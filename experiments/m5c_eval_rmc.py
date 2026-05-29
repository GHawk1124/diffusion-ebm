"""M5c — evaluate on the Repeated Mention Cloze (RMC) benchmark.

For each item in the frozen .jsonl benchmark files, runs all methods on the
same masked window and records 5 metrics:

  chain_em_supported  (PRIMARY) — exact chain recovery, conditioned on gold
                                  being in top-k at every mask position.
  chain_em            — exact chain recovery (denominator-honest).
  token_accuracy      — fraction of mask positions with gold token.
  unsupervised_agreement — fraction of chains/samples where all positions in
                            each entity chain agree (secondary; can reward the
                            variable-template failure mode).
  topk_support_rate   — fraction of mask positions where gold is in top-k.

Methods:
  mdlm_argmax          — single MDLM forward, argmax at each mask.
  mask_predict_T0      — Mask-Predict @ T=0, n_iters=4.
  best_of_n_cheap      — N=64 ancestral fills from 1 MDLM forward, ranked
                          by sum log-marginal (≈ argmax; 1 forward).
  hard_eq_thrml_oracle — THRML hard equality per identity chain (oracle labels).
  hard_eq_thrml_global — THRML hard equality over all mask positions (naïve).
  learned_psi_thrml    — locked ψ+THRML, oracle chain grouping, ws × burn sweep.
  learned_psi_thrml_global — locked ψ+THRML over ALL hole pairs, NO oracle
                          grouping (multi_chain only); tests whether the
                          learned factor recovers identity structure unaided.

Track-0 audit baselines (additive; show whether sampling earns its keep):
  hard_eq_map          — exact closed-form product-of-experts pooling MAP over
                          oracle groups (argmax_t Σ_{i∈g} log p_i(t)); no
                          sampling. Expected to tie hard_eq_thrml_oracle.
  mcmc_logits          — block-Gibbs with NO coupling factor (unary only);
                          isolates the factor as the source of any joint win.
                          Expected ≈ mdlm_argmax.
  best_of_n_strong     — N=64 ancestral fills scored by a real batched MDLM
                          joint forward (sum log p at former-mask positions),
                          ranked, best kept. ~1+N neural forwards (~65× FLOPs);
                          the matched-FLOPs headline must beat this.
  predicted_group_hard_eq — cluster holes WITHOUT gold (top-k Jaccard union-
                          find), then hard-eq within predicted groups. The
                          de-oracle probe; reports grouping ARI / pairwise F1 /
                          over-merge rate / n_pred_groups.

Run (dev split, Day-7 gate):
    uv run python experiments/m5c_eval_rmc.py \\
        --ckpt results/m5c_v3_k256/scorer_step00200000.pt \\
        --dev-only \\
        --out results/m5c_rmc_dev.json

Run (full test split, Days 8-10):
    uv run python experiments/m5c_eval_rmc.py \\
        --ckpt results/m5c_v3_k256/scorer_step00200000.pt \\
        --out results/m5c_rmc_test.json

Smoke test (50 items per corpus):
    uv run python experiments/m5c_eval_rmc.py \\
        --ckpt results/m5c_v3_k256/scorer_step00200000.pt \\
        --max-items 50 --dev-only --out results/m5c_rmc_smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from diffusion_ebm.backbones.mdlm import MASK_TOKEN_ID, MDLM  # noqa: E402
from diffusion_ebm.factors.learned import PairwiseScorer  # noqa: E402
from diffusion_ebm.sampler import thrml_joint, thrml_joint_learned  # noqa: E402
from diffusion_ebm.tasks.rmc import RMCItem, is_dev, load_items  # noqa: E402

N_CHAINS = 64
N_BON = 64   # best-of-N sample count


# ── Record schema ────────────────────────────────────────────────────────────

@dataclass
class Record:
    item_id: str
    corpus: str
    track: str
    method: str
    config: dict

    # Correctness metrics
    chain_em: bool          # exact match at every mask position
    chain_em_supported: Optional[bool]  # None when gold outside top-k at ≥1 position
    all_supported: bool     # gold in top-k at every mask position
    token_accuracy: float   # fraction of positions with gold token
    unsupervised_agreement: float  # fraction of samples where all chains agree
    topk_support_rate: float  # fraction of positions where gold is in top-k

    # Budget / timing
    n_lm_forwards: int
    wall_time_s: float
    seed: int

    # Grouping metrics — only populated for predicted_group_hard_eq; the
    # predicted partition (no gold) is scored against gold chain_labels.
    grouping_ari: Optional[float] = None
    grouping_pairwise_f1: Optional[float] = None
    over_merge_rate: Optional[float] = None       # cross-chain pairs merged
    n_pred_groups: Optional[int] = None


# ── Loader ───────────────────────────────────────────────────────────────────

def _load_scorer(ckpt_path: str, device: torch.device) -> PairwiseScorer:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = blob["config"]
    scorer = PairwiseScorer(
        hidden_dim=cfg["hidden_dim"],
        embed_dim=cfg["embed_dim"],
        head_dim=cfg["head_dim"],
        mlp_dim=blob.get("args", {}).get("mlp_dim", 256),
        vocab_size=cfg["vocab_size"],
    ).to(device)
    scorer.load_state_dict(blob["state_dict"])
    scorer.eval()
    return scorer


# ── Metrics helpers ──────────────────────────────────────────────────────────

def _compute_metrics(
    predicted: list[int],
    item: RMCItem,
    topk_ids: list[set[int]],            # top-k token IDs per mask position
    all_samples: Optional[list[list[int]]] = None,  # [N, n_holes] for agreement
) -> dict:
    gold = item.gold_token_ids
    chain_labels = item.chain_labels
    n_holes = len(item.mask_positions)
    n_entity_chains = max(chain_labels) + 1

    in_topk = [gold[i] in topk_ids[i] for i in range(n_holes)]
    topk_support_rate = sum(in_topk) / n_holes
    all_supported = all(in_topk)

    chain_em = all(predicted[i] == gold[i] for i in range(n_holes))
    chain_em_supported = chain_em if all_supported else None
    token_accuracy = sum(predicted[i] == gold[i] for i in range(n_holes)) / n_holes

    samples_for_agree = all_samples if all_samples is not None else [predicted]
    agree_count = 0
    for sample in samples_for_agree:
        ok = True
        for c in range(n_entity_chains):
            chain_toks = {sample[i] for i, cl in enumerate(chain_labels) if cl == c}
            if len(chain_toks) > 1:
                ok = False
                break
        if ok:
            agree_count += 1
    unsupervised_agreement = agree_count / len(samples_for_agree)

    return {
        "chain_em": chain_em,
        "chain_em_supported": chain_em_supported,
        "all_supported": all_supported,
        "token_accuracy": token_accuracy,
        "unsupervised_agreement": unsupervised_agreement,
        "topk_support_rate": topk_support_rate,
    }


def _majority_vote(samples: list[list[int]]) -> list[int]:
    """Plurality token at each hole across N samples."""
    n_holes = len(samples[0])
    return [Counter(s[h] for s in samples).most_common(1)[0][0]
            for h in range(n_holes)]


def _oracle_eq_groups(item: RMCItem) -> list[list[int]]:
    """Equality groups from gold chain labels (hole indices, 0-based)."""
    chain_to_holes: dict[int, list[int]] = defaultdict(list)
    for hole_idx, label in enumerate(item.chain_labels):
        chain_to_holes[label].append(hole_idx)
    return list(chain_to_holes.values())


def _all_pairs_within(groups: list[list[int]]) -> list[tuple[int, int]]:
    pairs = []
    for g in groups:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                pairs.append((g[a], g[b]))
    return pairs


def _predict_groups_jaccard(
    topk_sets: list[set[int]], threshold: float
) -> list[list[int]]:
    """Cluster holes WITHOUT gold via top-k Jaccard overlap + union-find.

    Two holes whose top-k candidate sets overlap with Jaccard ≥ threshold are
    merged. Connected components are the predicted entity groups. This is the
    de-oracle grouping signal: it only sees the corrupted input's MDLM top-k.
    """
    n = len(topk_sets)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a in range(n):
        for b in range(a + 1, n):
            inter = len(topk_sets[a] & topk_sets[b])
            union_sz = len(topk_sets[a] | topk_sets[b])
            jac = inter / union_sz if union_sz else 0.0
            if jac >= threshold:
                union(a, b)

    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    return list(comps.values())


def _adjusted_rand_index(labels_a: list[int], labels_b: list[int]) -> float:
    """ARI between two flat label assignments (1.0 = identical partition)."""
    n = len(labels_a)
    if n < 2:
        return 1.0
    contingency: dict[tuple[int, int], int] = Counter(zip(labels_a, labels_b))
    a_counts: Counter = Counter(labels_a)
    b_counts: Counter = Counter(labels_b)

    def comb2(x: int) -> int:
        return x * (x - 1) // 2

    sum_comb = sum(comb2(v) for v in contingency.values())
    sum_a = sum(comb2(v) for v in a_counts.values())
    sum_b = sum(comb2(v) for v in b_counts.values())
    total = comb2(n)
    expected = (sum_a * sum_b) / total if total else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if denom == 0:
        # Both partitions are trivial (e.g. all-singletons or all-one-cluster).
        return 1.0
    return (sum_comb - expected) / denom


def _grouping_metrics(
    pred_groups: list[list[int]], gold_labels: list[int]
) -> dict:
    """Compare predicted partition to gold chain labels.

    Returns adjusted Rand index, pairwise-link F1, over-merge rate (fraction of
    cross-chain hole pairs the prediction merged), and the predicted group count.
    """
    n = len(gold_labels)
    pred_labels = [0] * n
    for gi, g in enumerate(pred_groups):
        for i in g:
            pred_labels[i] = gi

    tp = fp = fn = 0
    cross_total = cross_merged = 0
    for a in range(n):
        for b in range(a + 1, n):
            gold_same = gold_labels[a] == gold_labels[b]
            pred_same = pred_labels[a] == pred_labels[b]
            if gold_same and pred_same:
                tp += 1
            elif gold_same and not pred_same:
                fn += 1
            elif (not gold_same) and pred_same:
                fp += 1
            if not gold_same:
                cross_total += 1
                if pred_same:
                    cross_merged += 1

    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    over_merge = cross_merged / cross_total if cross_total else 0.0
    return {
        "grouping_ari": _adjusted_rand_index(gold_labels, pred_labels),
        "grouping_pairwise_f1": f1,
        "over_merge_rate": over_merge,
        "n_pred_groups": len(pred_groups),
    }


# ── Per-item evaluation ──────────────────────────────────────────────────────

def evaluate_item(
    item: RMCItem,
    mdlm: MDLM,
    scorer: PairwiseScorer,
    weight_scales: list[float],
    burn_ins: list[int],
    k: int,
    seed: int,
    gate_only: bool = False,
    group_threshold: float = 0.3,
) -> list[Record]:
    """Run all methods on one RMC item. Returns list of Records."""
    device = mdlm.device
    input_ids = item.masked_input_ids.unsqueeze(0).to(device)  # [1, L]
    n_holes = len(item.mask_positions)
    mask_pos = item.mask_positions

    # ── Single MDLM forward (shared across all methods) ──────────────────
    t0 = time.time()
    logits, hidden = mdlm.forward_hidden(input_ids)
    logits_1d = logits[0].float()   # [L, V]
    hidden_1d = hidden[0].float()   # [L, d_h]
    logits_1d[:, MASK_TOKEN_ID] = -float("inf")
    mdlm_forward_time = time.time() - t0

    # Top-k candidates at each mask position
    logits_at_masks = logits_1d[mask_pos]  # [n_holes, V]
    ids_k, unary_k = mdlm.top_k_candidates(logits_1d.unsqueeze(0), k=k,
                                             exclude_mask_token=True)
    ids_k_holes = ids_k[0, mask_pos, :]      # [n_holes, k]
    unary_k_holes = unary_k[0, mask_pos, :]  # [n_holes, k]

    topk_sets = [set(ids_k_holes[h].tolist()) for h in range(n_holes)]

    unary_jnp = jnp.asarray(unary_k_holes.cpu().numpy(), dtype=jnp.float32)
    cand_jnp = jnp.asarray(ids_k_holes.cpu().numpy(), dtype=jnp.int32)
    cand_torch = ids_k_holes.to(device).long()

    oracle_groups = _oracle_eq_groups(item)
    global_groups = [list(range(n_holes))]
    psi_pairs = _all_pairs_within(oracle_groups)
    psi_pairs_global = _all_pairs_within(global_groups)
    n_entity_chains = max(item.chain_labels) + 1

    rng_key = jax.random.PRNGKey(seed)
    records: list[Record] = []

    def _make_record(method, config, metrics, n_lm, wall, **extra):
        return Record(
            item_id=item.item_id,
            corpus=item.corpus,
            track=item.track,
            method=method,
            config=config,
            n_lm_forwards=n_lm,
            wall_time_s=wall,
            seed=seed,
            **metrics,
            **extra,
        )

    # ── mdlm_argmax ──────────────────────────────────────────────────────
    t1 = time.time()
    pred_argmax = logits_at_masks.argmax(dim=-1).tolist()
    metrics_argmax = _compute_metrics(pred_argmax, item, topk_sets)
    records.append(_make_record(
        "mdlm_argmax", {"k": k}, metrics_argmax,
        n_lm=1, wall=mdlm_forward_time + (time.time() - t1),
    ))

    # ── mask_predict_T0 ──────────────────────────────────────────────────
    if not gate_only:
        t1 = time.time()
        filled = mdlm.mask_predict(item.masked_input_ids, n_iters=4, temperature=0.0)
        pred_mp = [filled[0, pos].item() for pos in mask_pos]
        metrics_mp = _compute_metrics(pred_mp, item, topk_sets)
        records.append(_make_record(
            "mask_predict_T0", {"n_iters": 4, "temperature": 0.0}, metrics_mp,
            n_lm=5, wall=time.time() - t1,
        ))

    # ── best_of_n_cheap ──────────────────────────────────────────────────
    if not gate_only:
        t1 = time.time()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        probs_at_masks = F.softmax(logits_at_masks.cpu(), dim=-1)  # [n_holes, V]
        fills = torch.multinomial(
            probs_at_masks.view(n_holes, -1).repeat(1, 1),
            N_BON, replacement=True, generator=gen,
        )  # [n_holes, N_BON]
        fills = fills.T  # [N_BON, n_holes]
        log_p = F.log_softmax(logits_at_masks.cpu(), dim=-1)  # [n_holes, V]
        scores = log_p[
            torch.arange(n_holes).unsqueeze(0),
            fills,
        ].sum(dim=-1)  # [N_BON]
        best_idx = scores.argmax().item()
        pred_bon = fills[best_idx].tolist()
        all_bon = fills.tolist()
        metrics_bon = _compute_metrics(pred_bon, item, topk_sets, all_samples=all_bon)
        records.append(_make_record(
            "best_of_n_cheap", {"n": N_BON}, metrics_bon,
            n_lm=1, wall=mdlm_forward_time + (time.time() - t1),
        ))

    # ── hard_eq_thrml_oracle ─────────────────────────────────────────────
    t1 = time.time()
    samp_oracle = thrml_joint.build(
        unary=unary_jnp,
        candidate_ids=cand_jnp,
        equality_groups=oracle_groups,
        equality_weight=5.0,
        k=k,
    )
    samples_oracle_jnp = thrml_joint.sample(
        samp_oracle, rng_key, n_chains=N_CHAINS, burn_in=200,
    )
    samples_oracle = np.asarray(samples_oracle_jnp).tolist()
    pred_oracle = _majority_vote(samples_oracle)
    metrics_oracle = _compute_metrics(pred_oracle, item, topk_sets,
                                       all_samples=samples_oracle)
    records.append(_make_record(
        "hard_eq_thrml_oracle",
        {"equality_weight": 5.0, "burn_in": 200, "n_chains": N_CHAINS, "k": k},
        metrics_oracle, n_lm=1, wall=mdlm_forward_time + (time.time() - t1),
    ))

    # ── hard_eq_thrml_global ─────────────────────────────────────────────
    if n_holes > 1 and not gate_only:
        t1 = time.time()
        samp_global = thrml_joint.build(
            unary=unary_jnp,
            candidate_ids=cand_jnp,
            equality_groups=global_groups,
            equality_weight=5.0,
            k=k,
        )
        samples_global_jnp = thrml_joint.sample(
            samp_global, rng_key, n_chains=N_CHAINS, burn_in=200,
        )
        samples_global = np.asarray(samples_global_jnp).tolist()
        pred_global = _majority_vote(samples_global)
        metrics_global = _compute_metrics(pred_global, item, topk_sets,
                                           all_samples=samples_global)
        records.append(_make_record(
            "hard_eq_thrml_global",
            {"equality_weight": 5.0, "burn_in": 200, "n_chains": N_CHAINS, "k": k},
            metrics_global, n_lm=1, wall=mdlm_forward_time + (time.time() - t1),
        ))

    # ── learned_psi_thrml ────────────────────────────────────────────────
    for ws in weight_scales:
        for burn in burn_ins:
            t1 = time.time()
            samp_psi = thrml_joint_learned.build(
                unary=unary_jnp,
                candidate_ids=cand_jnp,
                candidate_ids_torch=cand_torch,
                hidden=hidden_1d,
                hole_positions=mask_pos,
                pair_indices=psi_pairs,
                scorer=scorer,
                weight_scale=ws,
            )
            samples_psi_jnp = thrml_joint_learned.sample(
                samp_psi, rng_key, n_chains=N_CHAINS, burn_in=burn,
            )
            samples_psi = np.asarray(samples_psi_jnp).tolist()
            pred_psi = _majority_vote(samples_psi)
            metrics_psi = _compute_metrics(pred_psi, item, topk_sets,
                                            all_samples=samples_psi)
            records.append(_make_record(
                "learned_psi_thrml",
                {"weight_scale": ws, "burn_in": burn, "n_chains": N_CHAINS, "k": k},
                metrics_psi, n_lm=1, wall=mdlm_forward_time + (time.time() - t1),
            ))

    # ── learned_psi_thrml_global ─────────────────────────────────────────
    # ψ over ALL hole pairs with no oracle chain grouping: the scorer alone
    # must promote within-entity agreement and suppress cross-entity. This is
    # the only cell that tests whether the *learned* factor recovers identity
    # structure that hard equality can only get from oracle labels. For
    # single_chain global == oracle, so skip (would be a redundant copy).
    if not gate_only and n_entity_chains > 1:
        for ws in weight_scales:
            for burn in burn_ins:
                t1 = time.time()
                samp_psi_g = thrml_joint_learned.build(
                    unary=unary_jnp,
                    candidate_ids=cand_jnp,
                    candidate_ids_torch=cand_torch,
                    hidden=hidden_1d,
                    hole_positions=mask_pos,
                    pair_indices=psi_pairs_global,
                    scorer=scorer,
                    weight_scale=ws,
                )
                samples_psi_g_jnp = thrml_joint_learned.sample(
                    samp_psi_g, rng_key, n_chains=N_CHAINS, burn_in=burn,
                )
                samples_psi_g = np.asarray(samples_psi_g_jnp).tolist()
                pred_psi_g = _majority_vote(samples_psi_g)
                metrics_psi_g = _compute_metrics(pred_psi_g, item, topk_sets,
                                                 all_samples=samples_psi_g)
                records.append(_make_record(
                    "learned_psi_thrml_global",
                    {"weight_scale": ws, "burn_in": burn,
                     "n_chains": N_CHAINS, "k": k},
                    metrics_psi_g, n_lm=1,
                    wall=mdlm_forward_time + (time.time() - t1),
                ))

    # ══ Track-0 audit baselines ══════════════════════════════════════════
    if not gate_only:
        # ── hard_eq_map: exact closed-form pooling MAP over oracle groups ─
        # argmax_t Σ_{i∈g} log p_i(t) over the full vocab — the closed-form
        # product-of-experts MAP that hard equality + oracle grouping admits.
        # No sampling. If this ties hard_eq_thrml_oracle, sampling earns
        # nothing on the equality task (the Track-B-frustration green light).
        t1 = time.time()
        logp_full = F.log_softmax(logits_at_masks, dim=-1)  # [n_holes, V]
        pred_map = [0] * n_holes
        for g in oracle_groups:
            idx = torch.tensor(g, device=logp_full.device)
            pooled = logp_full.index_select(0, idx).sum(dim=0)  # [V]
            t_star = int(pooled.argmax().item())
            for i in g:
                pred_map[i] = t_star
        metrics_map = _compute_metrics(pred_map, item, topk_sets)
        records.append(_make_record(
            "hard_eq_map", {"support": "full_vocab"}, metrics_map,
            n_lm=1, wall=mdlm_forward_time + (time.time() - t1),
        ))

        # ── mcmc_logits: block-Gibbs with NO coupling factor (unary only) ─
        t1 = time.time()
        samp_nofac = thrml_joint.build(
            unary=unary_jnp,
            candidate_ids=cand_jnp,
            equality_groups=[],
            equality_weight=5.0,
            k=k,
        )
        samples_nofac_jnp = thrml_joint.sample(
            samp_nofac, rng_key, n_chains=N_CHAINS, burn_in=200,
        )
        samples_nofac = np.asarray(samples_nofac_jnp).tolist()
        pred_nofac = _majority_vote(samples_nofac)
        metrics_nofac = _compute_metrics(pred_nofac, item, topk_sets,
                                          all_samples=samples_nofac)
        records.append(_make_record(
            "mcmc_logits",
            {"factor": "none", "burn_in": 200, "n_chains": N_CHAINS, "k": k},
            metrics_nofac, n_lm=1, wall=mdlm_forward_time + (time.time() - t1),
        ))

        # ── best_of_n_strong: N ancestral fills, real joint-LM rerank ─────
        # Sample N fills per-hole, place them, run ONE batched MDLM forward
        # over the N filled sequences, score each by Σ log p(token | filled
        # context) at the former-mask positions, keep the best. ~1+N forwards.
        t1 = time.time()
        gen = torch.Generator(device="cpu").manual_seed(seed + 1)
        probs_at_masks = F.softmax(logits_at_masks.cpu(), dim=-1)  # [n_holes, V]
        fills = torch.multinomial(
            probs_at_masks, N_BON, replacement=True, generator=gen,
        ).T.contiguous()  # [N_BON, n_holes]
        mask_pos_t = torch.tensor(mask_pos, device=device)
        filled = item.masked_input_ids.unsqueeze(0).repeat(N_BON, 1).to(device)
        filled[:, mask_pos_t] = fills.to(device)
        strong_logits = mdlm.forward(filled)  # [N_BON, L, V]
        masked_logits = strong_logits[:, mask_pos_t, :].float()  # [N, n_holes, V]
        masked_logits[..., MASK_TOKEN_ID] = -float("inf")
        strong_logp = F.log_softmax(masked_logits, dim=-1)  # [N, n_holes, V]
        seq_scores = strong_logp.gather(
            -1, fills.to(device).unsqueeze(-1)
        ).squeeze(-1).sum(dim=-1)  # [N_BON]
        best_j = int(seq_scores.argmax().item())
        pred_bon_s = fills[best_j].tolist()
        all_bon_s = fills.tolist()
        metrics_bon_s = _compute_metrics(pred_bon_s, item, topk_sets,
                                          all_samples=all_bon_s)
        records.append(_make_record(
            "best_of_n_strong", {"n": N_BON, "scoring": "joint_lm"},
            metrics_bon_s, n_lm=1 + N_BON,
            wall=mdlm_forward_time + (time.time() - t1),
        ))

        # ── predicted_group_hard_eq: de-oracle grouping + hard equality ───
        t1 = time.time()
        pred_groups = _predict_groups_jaccard(topk_sets, group_threshold)
        samp_pg = thrml_joint.build(
            unary=unary_jnp,
            candidate_ids=cand_jnp,
            equality_groups=pred_groups,
            equality_weight=5.0,
            k=k,
        )
        samples_pg_jnp = thrml_joint.sample(
            samp_pg, rng_key, n_chains=N_CHAINS, burn_in=200,
        )
        samples_pg = np.asarray(samples_pg_jnp).tolist()
        pred_pg = _majority_vote(samples_pg)
        metrics_pg = _compute_metrics(pred_pg, item, topk_sets,
                                       all_samples=samples_pg)
        grouping = _grouping_metrics(pred_groups, item.chain_labels)
        records.append(_make_record(
            "predicted_group_hard_eq",
            {"equality_weight": 5.0, "burn_in": 200, "n_chains": N_CHAINS,
             "k": k, "group_threshold": group_threshold},
            metrics_pg, n_lm=1, wall=mdlm_forward_time + (time.time() - t1),
            **grouping,
        ))

    return records


# ── Summary helpers ──────────────────────────────────────────────────────────

def _print_summary(records: list[dict], n_items: int) -> None:
    from collections import defaultdict
    key_fn = lambda r: (
        r["corpus"], r["track"], r["method"],
        r["config"].get("weight_scale", ""),
        r["config"].get("burn_in", ""),
    )
    groups: dict = defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(r)

    prev_corpus_track = None
    for key in sorted(groups):
        corpus, track, method, ws, burn = key
        ct = f"{corpus}/{track}"
        if ct != prev_corpus_track:
            print(f"\n  --- {ct} ---")
            prev_corpus_track = ct
        recs = groups[key]

        n = len(recs)
        sup = [r for r in recs if r["all_supported"]]
        n_sup = len(sup)
        chain_em_rate = sum(r["chain_em"] for r in recs) / n if n else 0.0
        chain_em_sup_rate = (
            sum(r["chain_em_supported"] for r in sup) / n_sup if n_sup else float("nan")
        )
        tok_acc = sum(r["token_accuracy"] for r in recs) / n if n else 0.0
        agree = sum(r["unsupervised_agreement"] for r in recs) / n if n else 0.0
        topk = sum(r["topk_support_rate"] for r in recs) / n if n else 0.0

        cfg_str = ""
        if ws != "":
            cfg_str += f" ws={ws:.1f}"
        if burn != "":
            cfg_str += f" burn={burn}"

        grp_str = ""
        ari_vals = [r["grouping_ari"] for r in recs if r.get("grouping_ari") is not None]
        if ari_vals:
            f1_vals = [r["grouping_pairwise_f1"] for r in recs]
            om_vals = [r["over_merge_rate"] for r in recs]
            grp_str = (
                f"  ari={sum(ari_vals)/len(ari_vals):.3f}"
                f"  link_f1={sum(f1_vals)/len(f1_vals):.3f}"
                f"  over_merge={sum(om_vals)/len(om_vals):.3f}"
            )

        print(
            f"  {method:<28s}{cfg_str:<14s}"
            f"  items={n:>4d}  sup={n_sup:>4d}"
            f"  chain_em={chain_em_rate:.3f}"
            f"  chain_em_sup={chain_em_sup_rate:.3f}"
            f"  tok_acc={tok_acc:.3f}"
            f"  agree={agree:.3f}"
            f"  topk={topk:.3f}"
            f"{grp_str}"
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="PairwiseScorer checkpoint path")
    p.add_argument("--data-dir", default="data/rmc", help="Directory with .jsonl files")
    p.add_argument("--weight-scales", type=float, nargs="+",
                   default=[0.5, 1.0, 2.0, 5.0])
    p.add_argument("--burn-in", type=int, nargs="+", default=[100, 500])
    p.add_argument("--k", type=int, default=256,
                   help="Top-k candidates per mask position (default: 256)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dev-only", action="store_true",
                   help="Evaluate on dev split only (20%% by item_id hash)")
    p.add_argument("--test-only", action="store_true",
                   help="Evaluate on test split only (80%% by item_id hash)")
    p.add_argument("--gate-only", action="store_true",
                   help="Run only the 3 gate-relevant methods: mdlm_argmax, "
                        "hard_eq_thrml_oracle, learned_psi_thrml. Skips "
                        "mask_predict_T0, best_of_n_cheap, hard_eq_thrml_global, "
                        "and all Track-0 audit baselines.")
    p.add_argument("--group-threshold", type=float, default=0.3,
                   help="Top-k Jaccard threshold for predicted_group_hard_eq "
                        "de-oracle clustering (default: 0.3)")
    p.add_argument("--corpus", default="both",
                   choices=["both", "owt_heldout", "wikitext103"],
                   help="Restrict to one corpus (for sharded jobs)")
    p.add_argument("--track", default="both",
                   choices=["both", "single_chain", "multi_chain"],
                   help="Restrict to one track (for sharded jobs)")
    p.add_argument("--subsample-seed", type=int, default=0,
                   help="Seed for the deterministic shuffle applied before --max-items")
    p.add_argument("--max-items", type=int, default=None,
                   help="Cap items per corpus×track (deterministic random subsample)")
    p.add_argument("--out", default="results/m5c_rmc_eval.json")
    args = p.parse_args()

    if args.dev_only and args.test_only:
        print("fatal: --dev-only and --test-only are mutually exclusive", file=sys.stderr)
        return 1

    if torch.cuda.is_available():
        sm = torch.cuda.get_device_capability()
        if sm[0] < 8:
            print(
                f"fatal: GPU SM {sm[0]}.{sm[1]} < 8.0 — bf16 Triton kernels "
                "require Ampere+ (A100/L40s/H200). Resubmit to an SM 8.0+ partition.",
                file=sys.stderr,
            )
            return 1

    print("[rmc-eval] loading MDLM…")
    mdlm = MDLM.load()

    print(f"[rmc-eval] loading scorer from {args.ckpt}…")
    scorer = _load_scorer(args.ckpt, mdlm.device)
    log_temp = float(scorer.log_temp.detach())
    print(f"[rmc-eval] scorer log_temp={log_temp:.3f}  T={math.exp(log_temp):.2f}")

    data_dir = Path(args.data_dir)
    corpora = [
        ("owt_heldout", data_dir / "owt_heldout_single.jsonl",
                         data_dir / "owt_heldout_multi.jsonl"),
        ("wikitext103",  data_dir / "wikitext103_single.jsonl",
                         data_dir / "wikitext103_multi.jsonl"),
    ]

    all_records: list[dict] = []
    t_total = time.time()

    for corpus_tag, path_single, path_multi in corpora:
        if args.corpus != "both" and corpus_tag != args.corpus:
            continue
        for track, path in [("single_chain", path_single), ("multi_chain", path_multi)]:
            if args.track != "both" and track != args.track:
                continue
            print(f"\n[rmc-eval] === {corpus_tag}/{track} ===")
            items = load_items(path)

            if args.dev_only:
                items = [it for it in items if is_dev(it)]
                print(f"[rmc-eval]   dev split: {len(items)} items")
            elif args.test_only:
                items = [it for it in items if not is_dev(it)]
                print(f"[rmc-eval]   test split: {len(items)} items")
            else:
                print(f"[rmc-eval]   full set: {len(items)} items")

            if args.max_items is not None:
                rng = random.Random(args.subsample_seed)
                rng.shuffle(items)
                items = items[:args.max_items]
                print(f"[rmc-eval]   (random subsample to {len(items)}, "
                      f"seed={args.subsample_seed})")

            for idx, item in enumerate(items):
                recs = evaluate_item(
                    item, mdlm, scorer,
                    weight_scales=args.weight_scales,
                    burn_ins=args.burn_in,
                    k=args.k,
                    seed=args.seed,
                    gate_only=args.gate_only,
                    group_threshold=args.group_threshold,
                )
                all_records.extend(asdict(r) for r in recs)

                if (idx + 1) % 25 == 0 or idx == 0:
                    elapsed = time.time() - t_total
                    print(f"  [{idx+1}/{len(items)}]  {elapsed:.0f}s elapsed")

            print(f"  done — {len(items)} items")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────
    n_items_total = len({r["item_id"] + r["method"] for r in all_records})
    print(f"\n[rmc-eval] === RESULTS ({time.time() - t_total:.0f}s total) ===")
    _print_summary(all_records, n_items_total)
    print(f"\n[rmc-eval] wrote {len(all_records)} records → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
