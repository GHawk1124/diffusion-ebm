#!/bin/bash
# ============================================================================
# M5c — RMC benchmark evaluation.
#
# Runs all 6 methods on all 4 corpus×track files with the k=256 retrain
# checkpoint.  Supports dev-only mode (Day-7 gate) and full-test mode
# (Days 8-10).
#
# Submit (dev split — Day-7 gate):
#   sbatch --export=ALL,DEV_ONLY=1 slurm/m5c_eval_rmc.sbatch
#
# Submit (full test):
#   sbatch slurm/m5c_eval_rmc.sbatch
#
# Submit (smoke — 50 items per corpus×track, dev split):
#   sbatch --export=ALL,DEV_ONLY=1,MAX_ITEMS=50 slurm/m5c_eval_rmc.sbatch
#
# Override checkpoint:
#   sbatch --export=ALL,CKPT=/path/to/scorer_step*.pt \
#          slurm/m5c_eval_rmc.sbatch
# ============================================================================

#SBATCH --job-name=m5c_rmc_eval
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100,gpu-l40s,gpu-h200
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err
#SBATCH --account=gts-ccarr36
#SBATCH --qos=inferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=garrett.comes@gatech.edu

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"

if [[ -d "/storage/scratch1/1/${USER:-}" ]]; then
    OUT_DIR="${OUT_DIR:-/storage/scratch1/1/$USER/diffusion-ebm}"
else
    OUT_DIR="${OUT_DIR:-$REPO}"
fi
RESULTS_DIR="${RESULTS_DIR:-$OUT_DIR/results}"

# Default: k=256 native retrain checkpoint.
CKPT="${CKPT:-$(ls -t "$RESULTS_DIR/m5c_v3_k256"/scorer_step*.pt 2>/dev/null | head -1 || true)}"
if [[ -z "$CKPT" ]]; then
    echo "fatal: no checkpoint found under $RESULTS_DIR/m5c_v3_k256" >&2
    exit 1
fi

# Split selection — three-state, no silent dev/test mixing.
#   SPLIT=test | dev | full   (default: test)
# DEV_ONLY=1 kept as a legacy alias for SPLIT=dev.
SPLIT="${SPLIT:-test}"
DEV_ONLY="${DEV_ONLY:-0}"
if [[ "$DEV_ONLY" == "1" ]]; then SPLIT="dev"; fi

# Optional per-shard filters (default: all corpora / both tracks).
CORPUS="${CORPUS:-both}"          # both | owt_heldout | wikitext103
TRACK="${TRACK:-both}"            # both | single_chain | multi_chain
SUBSAMPLE_SEED="${SUBSAMPLE_SEED:-0}"

MAX_ITEMS="${MAX_ITEMS:-}"
# Tune these to trade speed vs coverage.  Defaults = full sweep.
WEIGHT_SCALES="${WEIGHT_SCALES:-0.5 1.0 2.0 5.0}"
BURN_IN="${BURN_IN:-100 500}"
GATE_ONLY="${GATE_ONLY:-0}"

module load cuda/12.6.1 python/3.12.5
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$OUT_DIR/.hf_cache}"
export PYTHONUNBUFFERED=1

cd "$REPO"

if [[ "$GATE_ONLY" == "1" ]]; then
    GATE_ONLY_ARG="--gate-only"
else
    GATE_ONLY_ARG=""
fi

case "$SPLIT" in
    dev)  RUN_TAG="dev";  SPLIT_ARG="--dev-only"  ;;
    test) RUN_TAG="test"; SPLIT_ARG="--test-only" ;;
    full) RUN_TAG="full"; SPLIT_ARG=""            ;;
    *) echo "fatal: SPLIT=$SPLIT (want test|dev|full)" >&2; exit 1 ;;
esac

# Embed corpus/track in the tag so each shard writes a unique JSON.
CORPUS_ARG=""
TRACK_ARG=""
if [[ "$CORPUS" != "both" ]]; then RUN_TAG="${RUN_TAG}_${CORPUS}"; CORPUS_ARG="--corpus $CORPUS"; fi
if [[ "$TRACK"  != "both" ]]; then RUN_TAG="${RUN_TAG}_${TRACK}";  TRACK_ARG="--track $TRACK";  fi

if [[ -n "$MAX_ITEMS" ]]; then
    RUN_TAG="${RUN_TAG}_smoke${MAX_ITEMS}"
    MAX_ITEMS_ARG="--max-items $MAX_ITEMS"
else
    MAX_ITEMS_ARG=""
fi

OUT_FILE="$RESULTS_DIR/m5c_rmc_${RUN_TAG}.json"

echo "================================================================"
echo "  job:     ${SLURM_JOB_NAME:-m5c_rmc_eval} (${SLURM_JOB_ID:-local})"
echo "  node:    ${SLURMD_NODENAME:-$HOSTNAME}"
echo "  ckpt:    $CKPT"
echo "  split:   $RUN_TAG"
echo "  out:     $OUT_FILE"
echo "  started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"
nvidia-smi || echo "(nvidia-smi unavailable)"
echo

uv sync

SECONDS=0
uv run python experiments/m5c_eval_rmc.py \
    --ckpt "$CKPT" \
    --data-dir "$REPO/data/rmc" \
    --weight-scales $WEIGHT_SCALES \
    --burn-in $BURN_IN \
    --k 256 \
    $SPLIT_ARG \
    $CORPUS_ARG \
    $TRACK_ARG \
    --subsample-seed "$SUBSAMPLE_SEED" \
    $MAX_ITEMS_ARG \
    $GATE_ONLY_ARG \
    --out "$OUT_FILE"
echo "==> eval done in $((SECONDS / 60)) min"

echo "================================================================"
echo "  finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  output:   $OUT_FILE"
echo "================================================================"
