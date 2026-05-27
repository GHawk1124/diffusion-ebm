#!/bin/bash
# ============================================================================
# M5c — eval-only job with temperature-override sweep (A3).
#
# Runs the full template × weight_scale × burn_in sweep at each of four
# log_temp override values plus the trained log_temp (no-override baseline).
# Use this to determine whether v2's f/g heads encode real signal independent
# of the runaway log_temp.
#
# Submit:
#   sbatch slurm/m5c_eval_override.sbatch
#
# Override the checkpoint path:
#   sbatch --export=ALL,CKPT=/path/to/scorer_stepXXXXXXXX.pt \
#          slurm/m5c_eval_override.sbatch
# ============================================================================

#SBATCH --job-name=m5c_eval_override
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

# Default to the latest checkpoint in results/m5c.
CKPT="${CKPT:-$(ls -t "$RESULTS_DIR/m5c"/scorer_step*.pt 2>/dev/null | head -1 || true)}"
if [[ -z "$CKPT" ]]; then
    echo "fatal: no checkpoint found under $RESULTS_DIR/m5c" >&2
    exit 1
fi

module load cuda/12.6.1 python/3.12.5
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$OUT_DIR/.hf_cache}"
export PYTHONUNBUFFERED=1

cd "$REPO"

echo "================================================================"
echo "  job:   ${SLURM_JOB_NAME:-m5c_eval_override} (${SLURM_JOB_ID:-local})"
echo "  node:  ${SLURMD_NODENAME:-$HOSTNAME}"
echo "  ckpt:  $CKPT"
echo "  started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"
nvidia-smi || echo "(nvidia-smi unavailable)"
echo

uv sync

OUT_FILE="$RESULTS_DIR/m5c_eval_override.json"
echo "==> running eval with temp-override sweep"
SECONDS=0
uv run python experiments/m5c_eval.py \
    --ckpt "$CKPT" \
    --templates all \
    --weight-scales 0.5 1.0 2.0 5.0 10.0 \
    --burn-in 100 500 \
    --temp-override -1.0 -0.5 0.0 0.5 \
    --out "$OUT_FILE"
echo "==> eval done in ${SECONDS}s"

echo "================================================================"
echo "  finished:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  output:    $OUT_FILE"
echo "================================================================"
