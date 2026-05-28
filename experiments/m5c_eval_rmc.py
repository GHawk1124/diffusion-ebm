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
                          by sum log-marginal.
  hard_eq_thrml_oracle — THRML hard equality per identity chain (oracle labels).
  hard_eq_thrml_global — THRML hard equality over all mask positions (naïve).
  learned_psi_thrml    — locked ψ+THRML, ws sweep × burn_in sweep.

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


# ── Per-item evaluation ──────────────────────────────────────────────────────

def evaluate_item(
    item: RMCItem,
    mdlm: MDLM,
    scorer: PairwiseScorer,
    weight_scales: list[float],
    burn_ins: list[int],
    k: int,
    seed: int,
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

    rng_key = jax.random.PRNGKey(seed)
    records: list[Record] = []

    def _make_record(method, config, metrics, n_lm, wall):
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
    t1 = time.time()
    filled = mdlm.mask_predict(item.masked_input_ids, n_iters=4, temperature=0.0)
    pred_mp = [filled[0, pos].item() for pos in mask_pos]
    metrics_mp = _compute_metrics(pred_mp, item, topk_sets)
    records.append(_make_record(
        "mask_predict_T0", {"n_iters": 4, "temperature": 0.0}, metrics_mp,
        n_lm=5, wall=time.time() - t1,  # 1 mdlm_forward already done + 4 iters
    ))

    # ── best_of_n_cheap ──────────────────────────────────────────────────
    t1 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    probs_at_masks = F.softmax(logits_at_masks.cpu(), dim=-1)  # [n_holes, V]
    # Sample N_BON complete fills
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
    if n_holes > 1:
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

        print(
            f"  {method:<28s}{cfg_str:<14s}"
            f"  items={n:>4d}  sup={n_sup:>4d}"
            f"  chain_em={chain_em_rate:.3f}"
            f"  chain_em_sup={chain_em_sup_rate:.3f}"
            f"  tok_acc={tok_acc:.3f}"
            f"  agree={agree:.3f}"
            f"  topk={topk:.3f}"
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
    p.add_argument("--max-items", type=int, default=None,
                   help="Cap items per corpus×track (for smoke testing)")
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
        for track, path in [("single_chain", path_single), ("multi_chain", path_multi)]:
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
                items = items[:args.max_items]
                print(f"[rmc-eval]   (capped to {len(items)})")

            for idx, item in enumerate(items):
                recs = evaluate_item(
                    item, mdlm, scorer,
                    weight_scales=args.weight_scales,
                    burn_ins=args.burn_in,
                    k=args.k,
                    seed=args.seed,
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
