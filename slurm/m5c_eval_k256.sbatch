#!/bin/bash
# ============================================================================
# M5c — eval-only job at k=256 (THRML uint8 ceiling).
#
# Tests whether ` x` and other rare tokens appear in MDLM's top-256 candidates
# for the failing templates (variable, room-polyseme, multi-group).  Uses the
# existing v3 step-200k scorer; no retraining.
#
# Submit:
#   sbatch slurm/m5c_eval_k256.sbatch
#
# Override the checkpoint:
#   sbatch --export=ALL,CKPT=/path/to/scorer_stepXXXXXXXX.pt \
#          slurm/m5c_eval_k256.sbatch
# ============================================================================

#SBATCH --job-name=m5c_eval_k256
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

# Default to the v3 step-200k checkpoint.
CKPT="${CKPT:-$(ls -t "$RESULTS_DIR/m5c_v3"/scorer_step*.pt 2>/dev/null | head -1 || true)}"
if [[ -z "$CKPT" ]]; then
    echo "fatal: no checkpoint found under $RESULTS_DIR/m5c_v3" >&2
    exit 1
fi

module load cuda/12.6.1 python/3.12.5
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$OUT_DIR/.hf_cache}"
export PYTHONUNBUFFERED=1

cd "$REPO"

echo "================================================================"
echo "  job:     ${SLURM_JOB_NAME:-m5c_eval_k256} (${SLURM_JOB_ID:-local})"
echo "  node:    ${SLURMD_NODENAME:-$HOSTNAME}"
echo "  ckpt:    $CKPT"
echo "  k:       256 (THRML uint8 ceiling)"
echo "  started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"
nvidia-smi || echo "(nvidia-smi unavailable)"
echo

uv sync

OUT_FILE="$RESULTS_DIR/m5c_v3_eval_k256.json"
echo "==> running eval at k=256 with temp-override sweep"
SECONDS=0
uv run python experiments/m5c_eval.py \
    --ckpt "$CKPT" \
    --templates all \
    --weight-scales 0.5 1.0 2.0 5.0 10.0 \
    --burn-in 100 500 \
    --k 256 \
    --temp-override -1.0 -0.5 0.0 0.5 \
    --out "$OUT_FILE"
echo "==> eval done in ${SECONDS}s"

echo "================================================================"
echo "  finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  output:   $OUT_FILE"
echo "================================================================"
