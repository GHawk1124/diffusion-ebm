#!/bin/bash
# ============================================================================
# Submit the RMC Day-7 dev-gate evaluation as 4 parallel GPU shards.
#
# One job per corpus×track so each fits comfortably under the wall; the
# per-shard outputs are merged offline.  Reduced sweep (ws ∈ {0.5,1.0,2.0},
# burn=500) — enough for the dev gate, ~1 h/shard instead of ~8 h combined.
#
# GPU SELECTION — why a *typed* gres, not a --partition list:
#   PACE does site-side partition routing; a multi-partition --partition list
#   gets expanded and silently re-adds gpu-v100, so jobs land on V100s.  A
#   typed gres (gpu:a100:1) can only be satisfied by a node that actually has
#   that GPU type, so the scheduler physically cannot place it on a V100
#   regardless of partition routing.  Each shard is pinned to one non-V100
#   type; the heavier multi_chain shards go on the highest-capacity types.
#
# Run from the repo root on a PACE login node:
#   bash slurm/submit_rmc_dev.sh
#
# Each shard writes a distinct file:
#   results/m5c_rmc_dev_<corpus>_<track>_smoke200.json
# ============================================================================

set -euo pipefail

SBATCH_SCRIPT="slurm/m5c_eval_rmc.sbatch"
WALLTIME="2:00:00"
MAX_ITEMS="200"
WEIGHT_SCALES="0.5 1.0 2.0"
BURN_IN="500"

submit_shard() {
    local corpus="$1" track="$2" partition="$3" gpu="$4"
    echo ">> ${corpus}/${track}  ->  ${partition} (gpu:${gpu})"
    WEIGHT_SCALES="$WEIGHT_SCALES" BURN_IN="$BURN_IN" \
        sbatch \
            --partition="$partition" \
            --gres="gpu:${gpu}:1" \
            --cpus-per-task=4 \
            --time="$WALLTIME" \
            --export=ALL,DEV_ONLY=1,CORPUS="$corpus",TRACK="$track",MAX_ITEMS="$MAX_ITEMS" \
            "$SBATCH_SCRIPT"
}

#            corpus       track          partition   gpu-type
submit_shard owt_heldout  single_chain   gpu-l40s    l40s
submit_shard owt_heldout  multi_chain    gpu-h200    h200
submit_shard wikitext103  single_chain   gpu-a100    a100
submit_shard wikitext103  multi_chain    gpu-h200    h200

echo
echo "All 4 dev shards submitted.  Confirm GPU type (no v100) with:"
echo "  squeue -u \$USER -o '%.10i %.12P %.14b %.20j %.8T'"
