#!/bin/bash
# ============================================================================
# Track A — RMC test-split sweep as 4 parallel GPU shards.
#
# Gated on Track 0 being complete (the audit baselines hard_eq_map /
# mcmc_logits / best_of_n_strong / predicted_group_hard_eq must be in the
# eval — they are, as of 2026-05-29). Does NOT pass --gate-only, so the full
# method set (core + Track-0 baselines) runs and lands in the headline table.
#
# This is the locked Track-A headline run on the held-out 80% test split. The
# dev sweep already de-risked it (joint−argmax SIG on every cell; ψ↔hard_eq
# interchangeable; best_of_n_strong loses; mcmc_logits≈argmax); this run
# confirms it on test before the result becomes a paper headline.
#
# COST MODEL (from dev walls, ws∈{0.5,1.0}, burn=500):
#   single_chain ~11 s/item (psi_global skipped: 1 entity chain)
#   multi_chain  ~24 s/item (psi_global ×2 dominates)
# Per-cell MAX_ITEMS caps each shard near ~3 h compute; the 12 h wall is 4×
# that margin. m5c_eval_rmc.py writes JSON ONLY at the end — a wall-kill loses
# everything, hence the generous wall and the caps.
#
# Test-split item counts (frozen is_dev hash, 80% test):
#   owt_heldout/single 12579   owt_heldout/multi 2933
#   wikitext103/single  3682   wikitext103/multi  950
# Subsampling is deterministic (SUBSAMPLE_SEED, default 0): the same items are
# drawn every run. For the weeks-10/11 multi-seed buffer, resubmit with
# SUBSAMPLE_SEED=1,2,… (each writes the same filename — move results between).
#
# GPU SELECTION — typed gres, never a --partition list: PACE routing expands a
# multi-partition list and silently re-adds gpu-v100. A typed gres (gpu:a100:1
# etc.) can only land on a node with that GPU. l40s enforces a 4:1 CPU:GPU
# ratio, so --cpus-per-task=4 (fine on every type). Heavy multi_chain → h200.
# Retarget any shard if a type is busy, e.g.:
#   --partition=gpu-h200 --gres=gpu:h200:1   (e.g. if a100 is saturated)
#
# Run from the repo root on a PACE login node:
#   bash slurm/submit_rmc_test.sh
#
# Each shard writes:
#   results/m5c_rmc_test_<corpus>_<track>_smoke<MAX_ITEMS>.json
# Merge offline with:
#   .venv/bin/python experiments/analyze_rmc.py 'results/m5c_rmc_test_*.json'
# ============================================================================

set -euo pipefail

SBATCH_SCRIPT="slurm/m5c_eval_rmc.sbatch"
WALLTIME="12:00:00"
WEIGHT_SCALES="0.5 1.0"
BURN_IN="500"

submit_shard() {
    local corpus="$1" track="$2" partition="$3" gpu="$4" max_items="$5"
    echo ">> ${corpus}/${track}  ->  ${partition} (gpu:${gpu})  cap=${max_items}"
    WEIGHT_SCALES="$WEIGHT_SCALES" BURN_IN="$BURN_IN" \
        sbatch \
            --partition="$partition" \
            --gres="gpu:${gpu}:1" \
            --cpus-per-task=4 \
            --time="$WALLTIME" \
            --export=ALL,SPLIT=test,CORPUS="$corpus",TRACK="$track",MAX_ITEMS="$max_items" \
            "$SBATCH_SCRIPT"
}

#            corpus       track          partition   gpu-type  max_items
submit_shard owt_heldout  single_chain   gpu-a100    a100      900
submit_shard owt_heldout  multi_chain    gpu-h200    h200      450
submit_shard wikitext103  single_chain   gpu-a100    a100      900
submit_shard wikitext103  multi_chain    gpu-h200    h200      450

echo
echo "All 4 test shards submitted.  Confirm GPU type (no v100) with:"
echo "  squeue -u \$USER -o '%.10i %.12P %.14b %.20j %.8T'"
