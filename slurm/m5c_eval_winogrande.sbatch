#!/bin/bash
# ============================================================================
# M5c — WinoGrande forced-choice rerank (Stage 1, M5d).
#
# Runs the LM-rerank baseline and psi_rerank variants on the WinoGrande
# validation set.  Pass --max-items 100 for the pilot, or omit for the full
# 1267-item run.
#
# Submit (pilot):
#   sbatch --export=ALL,MAX_ITEMS=100 slurm/m5c_eval_winogrande.sbatch
#
# Submit (full):
#   sbatch slurm/m5c_eval_winogrande.sbatch
#
# Override checkpoint:
#   sbatch --export=ALL,CKPT=/path/to/scorer_step*.pt \
#          slurm/m5c_eval_winogrande.sbatch
# ============================================================================

#SBATCH --job-name=m5c_wg_eval
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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

# Default: k=256 native retrain checkpoint; override with CKPT env var.
CKPT="${CKPT:-$(ls -t "$RESULTS_DIR/m5c_v3_k256"/scorer_step*.pt 2>/dev/null | head -1 || true)}"
if [[ -z "$CKPT" ]]; then
    echo "fatal: no checkpoint found under $RESULTS_DIR/m5c_v3_k256" >&2
    exit 1
fi

MAX_ITEMS="${MAX_ITEMS:-}"   # empty = full 1267-item run

module load cuda/12.6.1 python/3.12.5
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$OUT_DIR/.hf_cache}"
export PYTHONUNBUFFERED=1

cd "$REPO"

if [[ -n "$MAX_ITEMS" ]]; then
    RUN_TAG="pilot${MAX_ITEMS}"
    MAX_ITEMS_ARG="--max-items $MAX_ITEMS"
else
    RUN_TAG="full"
    MAX_ITEMS_ARG=""
fi
OUT_FILE="$RESULTS_DIR/m5c_wg_${RUN_TAG}.json"

echo "================================================================"
echo "  job:     ${SLURM_JOB_NAME:-m5c_wg_eval} (${SLURM_JOB_ID:-local})"
echo "  node:    ${SLURMD_NODENAME:-$HOSTNAME}"
echo "  ckpt:    $CKPT"
echo "  run:     $RUN_TAG"
echo "  out:     $OUT_FILE"
echo "  started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"
nvidia-smi || echo "(nvidia-smi unavailable)"
echo

uv sync

SECONDS=0
uv run python experiments/m5c_eval_winogrande.py \
    --ckpt "$CKPT" \
    --weight-scales 0.5 1.0 2.0 5.0 \
    $MAX_ITEMS_ARG \
    --out "$OUT_FILE"
echo "==> eval done in ${SECONDS}s"

echo "================================================================"
echo "  finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  output:   $OUT_FILE"
echo "================================================================"
