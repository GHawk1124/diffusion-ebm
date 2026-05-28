#!/bin/bash
# ============================================================================
# Submit the RMC Day-7 dev-gate evaluation as 4 parallel A100 shards.
#
# One job per corpus×track so each fits comfortably under the wall; the
# per-shard outputs are merged offline.  Reduced sweep (ws ∈ {0.5,1.0,2.0},
# burn=500) — enough for the dev gate, ~1 h/shard instead of ~8 h combined.
#
# Run from the repo root on a PACE login node:
#   bash slurm/submit_rmc_dev.sh
#
# Each shard writes a distinct file:
#   results/m5c_rmc_dev_<corpus>_<track>_smoke200.json
# ============================================================================

set -euo pipefail

SBATCH_SCRIPT="slurm/m5c_eval_rmc.sbatch"
PARTITION="gpu-a100"
WALLTIME="2:00:00"
MAX_ITEMS="200"
WEIGHT_SCALES="0.5 1.0 2.0"
BURN_IN="500"

submit_shard() {
    local corpus="$1" track="$2"
    echo ">> submitting dev shard: ${corpus}/${track}"
    WEIGHT_SCALES="$WEIGHT_SCALES" BURN_IN="$BURN_IN" \
        sbatch \
            --partition="$PARTITION" \
            --time="$WALLTIME" \
            --export=ALL,DEV_ONLY=1,CORPUS="$corpus",TRACK="$track",MAX_ITEMS="$MAX_ITEMS" \
            "$SBATCH_SCRIPT"
}

submit_shard owt_heldout single_chain
submit_shard owt_heldout multi_chain
submit_shard wikitext103 single_chain
submit_shard wikitext103 multi_chain

echo
echo "All 4 dev shards submitted.  Monitor with:  squeue -u \$USER"
