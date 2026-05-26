#!/bin/bash
# ============================================================================
# M5c — module-based sbatch (no Apptainer).
#
# Uses the cluster module system instead of a container image.
# First submission triggers a flash-attn build (~15 min); subsequent runs
# reuse the uv wheel cache.
#
# Prerequisites (one-time, on the login node):
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Submit:
#   sbatch slurm/m5c_train_modules.sbatch
#
# Target any of A100 / L40S / H200 (whichever starts first):
#   sbatch --partition=gpu-a100,gpu-l40s,gpu-h200 --gres=gpu:1 \
#          slurm/m5c_train_modules.sbatch
#
# Override knobs:
#   sbatch --export=ALL,STEPS=50000,SEED=1 slurm/m5c_train_modules.sbatch
# ============================================================================

# ----- SLURM directives ------------------------------------------------------
#SBATCH --job-name=m5c_train
#SBATCH --time=24:00:00
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

# ----- knobs -----------------------------------------------------------------
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
STEPS="${STEPS:-200000}"
BATCH="${BATCH:-256}"
SEED="${SEED:-0}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

# ----- output dirs -----------------------------------------------------------
if [[ -d "/storage/scratch1/1/${USER:-}" ]]; then
    OUT_DIR="${OUT_DIR:-/storage/scratch1/1/$USER/diffusion-ebm}"
else
    OUT_DIR="${OUT_DIR:-$REPO}"
fi
RESULTS_DIR="${RESULTS_DIR:-$OUT_DIR/results}"
PLOTS_DIR="${PLOTS_DIR:-$OUT_DIR/plots}"

mkdir -p "$RESULTS_DIR/m5c" "$PLOTS_DIR"

# ----- modules ---------------------------------------------------------------
module load cuda/12.6.1 python/3.12.5

# ----- uv --------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "fatal: uv not found. Install on the login node with:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# flash-attn build: cover A100 (8.0) / L40S (8.9) / H100+H200 (9.0).
# Only matters on the first uv sync; cached on subsequent runs.
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"
export MAX_JOBS="4"

# HF cache to scratch so it doesn't fill the 20 GB home quota.
export HF_HOME="${HF_HOME:-$OUT_DIR/.hf_cache}"
mkdir -p "$HF_HOME"

cd "$REPO"

# ----- job header ------------------------------------------------------------
echo "================================================================"
echo "  job:      ${SLURM_JOB_NAME:-m5c_train} (${SLURM_JOB_ID:-local})"
echo "  node:     ${SLURMD_NODENAME:-$HOSTNAME}"
echo "  repo:     $REPO"
echo "  out dir:  $OUT_DIR"
echo "  steps:    $STEPS  batch: $BATCH  seed: $SEED"
echo "  started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"
nvidia-smi || echo "(nvidia-smi unavailable)"
echo

# ----- sync env (first run builds flash-attn, ~15 min) ----------------------
echo "==> uv sync"
uv sync
echo

# ----- smoke (≈ 30 s) --------------------------------------------------------
if [[ "$RUN_SMOKE" == "1" ]]; then
    echo "==> M5c smoke"
    SECONDS=0
    uv run python experiments/m5c_smoke.py
    echo "==> smoke done in ${SECONDS}s"
    echo
fi

# ----- training --------------------------------------------------------------
echo "==> M5c training: $STEPS steps, batch $BATCH, seed $SEED"
SECONDS=0
uv run python experiments/m5c_train.py \
    --corpus owt \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --neg-k 16 \
    --top-k 64 \
    --seq-len 64 \
    --min-pair-dist 4 \
    --lr 3e-4 \
    --wd 0.01 \
    --embed-dim 256 \
    --head-dim 128 \
    --mlp-dim 1024 \
    --log-every 200 \
    --ckpt-every 5000 \
    --ckpt-dir "$RESULTS_DIR/m5c" \
    --seed "$SEED"
echo "==> training done in $((SECONDS / 60)) min"
echo

# ----- eval ------------------------------------------------------------------
if [[ "$RUN_EVAL" == "1" ]]; then
    CKPT="$(ls -t "$RESULTS_DIR/m5c"/scorer_step*.pt 2>/dev/null | head -1 || true)"
    if [[ -z "$CKPT" ]]; then
        echo "warn: no checkpoint found, skipping eval" >&2
    else
        echo "==> M5c eval: $CKPT"
        SECONDS=0
        uv run python experiments/m5c_eval.py \
            --ckpt "$CKPT" \
            --templates all \
            --weight-scales 0.5 1.0 2.0 5.0 10.0 \
            --burn-in 100 500 \
            --out "$RESULTS_DIR/m5c_eval.json"
        echo "==> eval done in ${SECONDS}s"
    fi
fi

echo "================================================================"
echo "  finished:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  artefacts: $RESULTS_DIR/m5c/scorer_step*.pt"
echo "             $RESULTS_DIR/m5c_eval.json"
echo "================================================================"
