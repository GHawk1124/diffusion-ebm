#!/bin/bash
# ============================================================================
# M5c v3 k=256 — retrain with k=256 top-k support (THRML uint8 ceiling).
#
# Eliminates the train/eval k mismatch from the k=128 run: the NCE negatives
# are now calibrated to the same 256×256 table used at eval time.
#
# Submit:
#   sbatch slurm/m5c_train_k256.sbatch
#
# H200 only (fastest; k=256 adds ~20% per-step cost vs k=128):
#   sbatch --partition=gpu-h200 --gres=gpu:1 slurm/m5c_train_k256.sbatch
#
# Override knobs:
#   sbatch --export=ALL,STEPS=200000,SEED=0 slurm/m5c_train_k256.sbatch
# ============================================================================

#SBATCH --job-name=m5c_k256
#SBATCH --time=7:00:00
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
STEPS="${STEPS:-200000}"
BATCH="${BATCH:-256}"
SEED="${SEED:-0}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

if [[ -d "/storage/scratch1/1/${USER:-}" ]]; then
    OUT_DIR="${OUT_DIR:-/storage/scratch1/1/$USER/diffusion-ebm}"
else
    OUT_DIR="${OUT_DIR:-$REPO}"
fi
RESULTS_DIR="${RESULTS_DIR:-$OUT_DIR/results}"
PLOTS_DIR="${PLOTS_DIR:-$OUT_DIR/plots}"

mkdir -p "$RESULTS_DIR/m5c_v3_k256" "$PLOTS_DIR"

module load cuda/12.6.1 python/3.12.5

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "fatal: uv not found. Install on the login node with:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"
export MAX_JOBS="4"
export HF_HOME="${HF_HOME:-$OUT_DIR/.hf_cache}"
mkdir -p "$HF_HOME"
export PYTHONUNBUFFERED=1

cd "$REPO"

echo "================================================================"
echo "  job:      ${SLURM_JOB_NAME:-m5c_k256} (${SLURM_JOB_ID:-local})"
echo "  node:     ${SLURMD_NODENAME:-$HOSTNAME}"
echo "  repo:     $REPO"
echo "  out dir:  $OUT_DIR"
echo "  k:        256 (THRML uint8 ceiling)"
echo "  steps:    $STEPS  batch: $BATCH  seed: $SEED"
echo "  started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"
nvidia-smi || echo "(nvidia-smi unavailable)"
echo

echo "==> uv sync"
uv sync
echo

if [[ "$RUN_SMOKE" == "1" ]]; then
    echo "==> M5c smoke"
    SECONDS=0
    uv run python experiments/m5c_smoke.py
    echo "==> smoke done in ${SECONDS}s"
    echo
fi

echo "==> M5c training: $STEPS steps, batch $BATCH, seed $SEED, k=256"
SECONDS=0
uv run python experiments/m5c_train.py \
    --corpus owt \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --top-k 256 \
    --seq-len 64 \
    --min-pair-dist 4 \
    --mine-repeats \
    --lr 1e-4 \
    --wd 0.01 \
    --lr-warmup-steps 1000 \
    --embed-dim 256 \
    --head-dim 128 \
    --mlp-dim 1024 \
    --initial-buffer 50000 \
    --refresh-every 0 \
    --synth-frac-start 0.30 \
    --synth-frac-end 0.10 \
    --synth-anneal-steps 150000 \
    --temp-clamp-min -1.0 \
    --temp-clamp-max 0.5 \
    --log-every 200 \
    --ckpt-every 5000 \
    --ckpt-dir "$RESULTS_DIR/m5c_v3_k256" \
    --seed "$SEED"
echo "==> training done in $((SECONDS / 60)) min"
echo

if [[ "$RUN_EVAL" == "1" ]]; then
    CKPT="$(ls -t "$RESULTS_DIR/m5c_v3_k256"/scorer_step*.pt 2>/dev/null | head -1 || true)"
    if [[ -z "$CKPT" ]]; then
        echo "warn: no checkpoint found, skipping eval" >&2
    else
        echo "==> M5c eval at k=256: $CKPT"
        SECONDS=0
        uv run python experiments/m5c_eval.py \
            --ckpt "$CKPT" \
            --templates all \
            --weight-scales 0.5 1.0 2.0 5.0 10.0 \
            --burn-in 100 500 \
            --k 256 \
            --out "$RESULTS_DIR/m5c_v3_k256_eval.json"
        echo "==> eval done in ${SECONDS}s"
    fi
fi

echo "================================================================"
echo "  finished:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  artefacts: $RESULTS_DIR/m5c_v3_k256/scorer_step*.pt"
echo "             $RESULTS_DIR/m5c_v3_k256_eval.json"
echo "================================================================"
