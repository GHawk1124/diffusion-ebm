#!/bin/bash
# ============================================================================
# M5c — full training + smoke + eval, one PACE job.
#
# Submit from the repo root, with diffusion-ebm.sif already in $HOME:
#     sbatch slurm/m5c_train.sbatch
#
# Override any knob via --export, e.g.:
#     sbatch --export=ALL,STEPS=50000,SEED=1 slurm/m5c_train.sbatch
#     sbatch --export=ALL,SIF=/scratch/$USER/diffusion-ebm.sif ...
#
# Fill in the two FIXME directives before first submission.
# ============================================================================

# ----- SLURM directives ------------------------------------------------------
#SBATCH --job-name=m5c_train
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

# PACE Phoenix — paid no-preempt tier on the H100 partition.
#   account:    gts-ccarr36   (`pace-quota` → Job Charge Account Balances)
#   partition:  gpu-h100      (4 H100 nodes, sm_90; same image as H200)
#   qos:        inferno       (no preemption; embers is preemptible)
# Override on the cli with --account=, --partition=, --qos=, --gres= as needed.
#SBATCH --account=gts-ccarr36
#SBATCH --partition=gpu-h100
#SBATCH --qos=inferno

# Email on completion / failure.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=garrett.comes@gatech.edu

set -euo pipefail

# ----- knobs (override via sbatch --export=ALL,KEY=VAL ...) ------------------
SIF="${SIF:-$HOME/diffusion-ebm.sif}"
REPO="${REPO:-$SLURM_SUBMIT_DIR}"

# All bulk writes (checkpoints ~4 GB, HF streaming cache potentially several
# GB) go to $OUT_DIR. On PACE we default to per-user scratch (15 TB quota) so
# the 20 GB home filesystem doesn't fill mid-run. Off-PACE we fall back to the
# repo. Override with --export=ALL,OUT_DIR=/path/...
if [[ -d "/storage/scratch1/1/${USER:-}" ]]; then
    OUT_DIR="${OUT_DIR:-/storage/scratch1/1/$USER/diffusion-ebm}"
else
    OUT_DIR="${OUT_DIR:-$REPO}"
fi
RESULTS_DIR="${RESULTS_DIR:-$OUT_DIR/results}"
PLOTS_DIR="${PLOTS_DIR:-$OUT_DIR/plots}"
HF_CACHE="${HF_CACHE:-$OUT_DIR/.hf_cache}"

STEPS="${STEPS:-200000}"
BATCH="${BATCH:-256}"
SEED="${SEED:-0}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
# Compute capability of the GPU you actually land on. SIF bakes 8.0;8.9;9.0
# so this only matters for any runtime JIT (rare in our code). Default = H100.
#   A100  → 8.0    L40S → 8.9    H100/H200 → 9.0    Blackwell → not supported
CUDA_ARCH="${CUDA_ARCH:-9.0}"

# ----- environment -----------------------------------------------------------
# PACE wraps apptainer behind a Lmod module. The fallback covers clusters
# where it's already on PATH.
module load apptainer 2>/dev/null \
    || module load singularity 2>/dev/null \
    || true

if ! command -v apptainer >/dev/null 2>&1; then
    if command -v singularity >/dev/null 2>&1; then
        alias apptainer=singularity
        shopt -s expand_aliases
    else
        echo "fatal: neither apptainer nor singularity on PATH" >&2
        exit 1
    fi
fi

if [[ ! -f "$SIF" ]]; then
    echo "fatal: $SIF missing. Build locally with:" >&2
    echo "       nix run .#build-apptainer-image" >&2
    echo "       scp diffusion-ebm.sif <login>:$SIF" >&2
    exit 1
fi

cd "$REPO"
mkdir -p "$RESULTS_DIR/m5c" "$PLOTS_DIR" "$HF_CACHE"

# ----- job header ------------------------------------------------------------
echo "================================================================"
echo "  job:        ${SLURM_JOB_NAME:-m5c_train} (${SLURM_JOB_ID:-local})"
echo "  node:       ${SLURMD_NODENAME:-$HOSTNAME}"
echo "  repo:       $REPO"
echo "  sif:        $SIF"
echo "  out dir:    $OUT_DIR"
echo "  results:    $RESULTS_DIR"
echo "  plots:      $PLOTS_DIR"
echo "  hf cache:   $HF_CACHE"
echo "  steps:      $STEPS"
echo "  batch:      $BATCH"
echo "  seed:       $SEED"
echo "  started:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"
echo
nvidia-smi || echo "(nvidia-smi unavailable on this node)"
echo

# Reusable bind-mount arguments. The container expects results/plots inside
# /opt/diffusion-ebm; HF_HOME stays at the in-image default but redirected
# at the cache subdir so we can use cluster scratch.
APPTAINER_ARGS=(
    --nv
    --bind "$RESULTS_DIR:/opt/diffusion-ebm/results"
    --bind "$PLOTS_DIR:/opt/diffusion-ebm/plots"
    --bind "$HF_CACHE:/opt/diffusion-ebm/.hf_cache"
    --env  "TORCH_CUDA_ARCH_LIST=$CUDA_ARCH"
    # Pin the container's own gcc so Triton JIT doesn't pick up the PACE
    # spack-managed compiler (host CC/CXX leak in via the inherited env).
    --env  "CC=/usr/bin/gcc"
    --env  "CXX=/usr/bin/g++"
)

# ----- smoke (≈ 30 s) --------------------------------------------------------
if [[ "$RUN_SMOKE" == "1" ]]; then
    echo "==> M5c smoke (sanity check)"
    SECONDS=0
    apptainer run "${APPTAINER_ARGS[@]}" "$SIF" \
        python experiments/m5c_smoke.py
    echo "==> smoke done in ${SECONDS}s"
    echo
fi

# ----- training (≈ 18 h on A100) ---------------------------------------------
echo "==> M5c training: $STEPS steps, batch $BATCH, seed $SEED"
SECONDS=0
apptainer run "${APPTAINER_ARGS[@]}" "$SIF" \
    python experiments/m5c_train.py \
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
        --ckpt-dir results/m5c \
        --seed "$SEED"
echo "==> training done in $((SECONDS / 60)) min"
echo

# ----- eval (≈ 10 min) -------------------------------------------------------
if [[ "$RUN_EVAL" == "1" ]]; then
    # shellcheck disable=SC2012 # checkpoint filenames are scorer_step\d+\.pt — alphanumeric only.
    CKPT_HOST="$(ls -t "$RESULTS_DIR/m5c"/scorer_step*.pt 2>/dev/null | head -1 || true)"
    if [[ -z "$CKPT_HOST" ]]; then
        echo "warn: no checkpoint at $RESULTS_DIR/m5c/scorer_step*.pt — skipping eval" >&2
    else
        # Convert host path → in-container path (RESULTS_DIR is bind-mounted at
        # /opt/diffusion-ebm/results).
        CKPT="results/m5c/$(basename "$CKPT_HOST")"
        echo "==> M5c eval: $CKPT_HOST"
        SECONDS=0
        apptainer run "${APPTAINER_ARGS[@]}" "$SIF" \
            python experiments/m5c_eval.py \
                --ckpt "$CKPT" \
                --templates all \
                --weight-scales 0.5 1.0 2.0 5.0 10.0 \
                --burn-in 100 500 \
                --out results/m5c_eval.json
        echo "==> eval done in ${SECONDS}s"
        echo
    fi
fi

echo "================================================================"
echo "  finished:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  artefacts:  results/m5c/scorer_step*.pt"
echo "              results/m5c_eval.json"
echo "  log:        results/slurm-${SLURM_JOB_NAME:-m5c_train}-${SLURM_JOB_ID:-local}.out"
echo "================================================================"
