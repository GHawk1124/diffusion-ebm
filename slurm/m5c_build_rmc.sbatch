#!/bin/bash
# ============================================================================
# Build frozen RMC benchmark files (CPU-only, runs once).
#
# Produces data/rmc/{owt_heldout,wikitext103}_{single,multi}.jsonl.
# Commit those files after this job completes — they are the frozen
# benchmark artefacts used by m5c_eval_rmc.sbatch.
#
# Submit:
#   sbatch slurm/m5c_build_rmc.sbatch
#
# After completion, commit the .jsonl files:
#   git add data/rmc/*.jsonl
#   git commit -m "Add frozen RMC benchmark artefacts"
# ============================================================================

#SBATCH --job-name=rmc_build
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu-a100,gpu-l40s,gpu-h200
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err
#SBATCH --account=gts-ccarr36
#SBATCH --qos=inferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=garrett.comes@gatech.edu
# NOTE: if the compute node lacks internet access, run this interactively
# on the login node instead:
#   HF_HOME=/storage/scratch1/1/$USER/diffusion-ebm/.hf_cache \
#   uv run python experiments/build_rmc.py --out-dir data/rmc --print-stats

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"

if [[ -d "/storage/scratch1/1/${USER:-}" ]]; then
    OUT_DIR="${OUT_DIR:-/storage/scratch1/1/$USER/diffusion-ebm}"
else
    OUT_DIR="${OUT_DIR:-$REPO}"
fi

module load cuda/12.6.1 python/3.12.5
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$OUT_DIR/.hf_cache}"
export PYTHONUNBUFFERED=1

cd "$REPO"

echo "================================================================"
echo "  job:     ${SLURM_JOB_NAME:-rmc_build} (${SLURM_JOB_ID:-local})"
echo "  node:    ${SLURMD_NODENAME:-$HOSTNAME}"
echo "  out:     data/rmc/"
echo "  started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

uv sync

SECONDS=0
uv run python experiments/build_rmc.py \
    --out-dir "$REPO/data/rmc" \
    --print-stats
echo "==> extraction done in ${SECONDS}s"

echo "================================================================"
echo "  finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  outputs:  $(ls -lh $REPO/data/rmc/*.jsonl 2>/dev/null || echo 'none')"
echo "================================================================"
echo
echo "Next step: commit the .jsonl artefacts to the repo."
echo "  cd $REPO && git add data/rmc/*.jsonl && git commit -m 'Add frozen RMC benchmark artefacts'"
