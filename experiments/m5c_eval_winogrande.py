"""M5c — WinoGrande forced-choice rerank pilot.

For each WinoGrande validation item, scores option1 and option2 under:

  lm_rerank     log p_LM(option | context) at the blank positions from a
                single MDLM forward.  This is the "best-of-N cheap" baseline
                for forced-choice tasks: ranking two specific options by their
                marginal LM log-prob.

  psi_rerank    LM score + weight_scale * ψ(h_blank, option_tok,
                                             h_ctx, ctx_tok)
                summed over N_CTX nearby context positions.  Tests whether
                the bilinear pairwise scorer adds signal beyond unary LM
                log-probs for a task the scorer was not explicitly trained on.

The pilot (--max-items 100) establishes feasibility before the full
1267-item validation run.  Success gate: psi_rerank accuracy > lm_rerank
accuracy + 3 pp with a directionally correct margin.

Run:
    uv run python experiments/m5c_eval_winogrande.py \\
        --ckpt results/m5c_v3_k256/scorer_step00200000.pt \\
        --max-items 100 \\
        --out results/m5c_wg_pilot.json

    uv run python experiments/m5c_eval_winogrande.py \\
        --ckpt results/m5c_v3_k256/scorer_step00200000.pt \\
        --out results/m5c_wg_full.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from diffusion_ebm.backbones.mdlm import MASK_TOKEN_ID, MDLM  # noqa: E402
from diffusion_ebm.factors.learned import PairwiseScorer        # noqa: E402
from diffusion_ebm.tasks.winogrande import WinoGrandeItem, build_items  # noqa: E402

N_CTX = 8   # context positions used for ψ scoring


@dataclass
class Record:
    item_id: str
    sentence: str
    option1: str
    option2: str
    gold: int
    method: str
    config: dict
    score1: float
    score2: float
    predicted: int   # 1 or 2
    correct: bool
    wall_time_s: float


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


def _lm_score(
    logits: torch.Tensor,        # [L, V] float
    option_positions: list[int],
    option_tok_ids: list[int],
) -> float:
    """Log-prob of option tokens at blank positions."""
    log_probs = F.log_softmax(logits, dim=-1)
    score = 0.0
    for pos, tok in zip(option_positions, option_tok_ids):
        score += log_probs[pos, tok].item()
    return score


def _context_positions(
    seq_len: int,
    option_positions: list[int],
    observed_ids: list[int],
    n_ctx: int,
) -> list[tuple[int, int]]:
    """Select up to n_ctx non-blank, content-word positions nearest the blank.

    Returns list of (position, token_id) pairs sorted by distance to blank centre.
    Skips punctuation-only tokens (token_id < 256) to focus on content words.
    """
    blank_centre = sum(option_positions) / len(option_positions)
    option_set = set(option_positions)
    candidates = []
    for pos, tok in enumerate(observed_ids):
        if pos in option_set:
            continue
        if tok < 256:  # punctuation / space-only tokens
            continue
        candidates.append((abs(pos - blank_centre), pos, tok))
    candidates.sort()
    return [(pos, tok) for _, pos, tok in candidates[:n_ctx]]


def _psi_score(
    scorer: PairwiseScorer,
    hidden: torch.Tensor,          # [L, d_h] on scorer device
    option_positions: list[int],
    option_tok_ids: list[int],
    ctx_pairs: list[tuple[int, int]],  # (position, token_id)
    weight_scale: float,
) -> float:
    """ψ score for an option against context positions."""
    device = scorer.log_temp.device
    total = 0.0
    for blank_pos, opt_tok in zip(option_positions, option_tok_ids):
        h_a = hidden[blank_pos]                                    # [d_h]
        ids_a = torch.tensor([opt_tok], dtype=torch.long, device=device)
        for ctx_pos, ctx_tok in ctx_pairs:
            h_b = hidden[ctx_pos]                                  # [d_h]
            ids_b = torch.tensor([ctx_tok], dtype=torch.long, device=device)
            psi = scorer.pair_table(h_a, h_b, ids_a, ids_b)       # [1, 1]
            total += psi.item()
    return total * weight_scale


def evaluate_item(
    item: WinoGrandeItem,
    mdlm: MDLM,
    scorer: Optional[PairwiseScorer],
    weight_scales: list[float],
) -> list[Record]:
    """Score one WG item under lm_rerank and psi_rerank variants."""
    t0 = time.time()
    input_ids = item.masked_input_ids.unsqueeze(0).to(mdlm.device)

    logits, hidden = mdlm.forward_hidden(input_ids)
    logits_1d = logits[0].float()     # [L, V]
    hidden_1d = hidden[0].float()     # [L, d_h]

    lm1 = _lm_score(logits_1d, item.option_positions, item.option1_token_ids)
    lm2 = _lm_score(logits_1d, item.option_positions, item.option2_token_ids)

    records: list[Record] = []
    wall = time.time() - t0
    pred_lm = 1 if lm1 >= lm2 else 2
    records.append(Record(
        item_id=item.item_id,
        sentence=item.sentence,
        option1=item.option1,
        option2=item.option2,
        gold=item.gold,
        method="lm_rerank",
        config={},
        score1=lm1,
        score2=lm2,
        predicted=pred_lm,
        correct=(pred_lm == item.gold),
        wall_time_s=wall,
    ))

    if scorer is not None:
        # Build the filled-in sequence for context position selection:
        # use option1 to build observed_ids (the context tokens don't change).
        filled_ids = (
            item.masked_input_ids.tolist()[:item.option_positions[0]]
            + item.option1_token_ids
            + item.masked_input_ids.tolist()[item.option_positions[-1] + 1:]
        )
        ctx_pairs = _context_positions(
            seq_len=len(item.masked_input_ids),
            option_positions=item.option_positions,
            observed_ids=filled_ids,
            n_ctx=N_CTX,
        )
        hidden_dev = hidden_1d.to(scorer.log_temp.device)

        for ws in weight_scales:
            t1 = time.time()
            psi1 = _psi_score(
                scorer, hidden_dev,
                item.option_positions, item.option1_token_ids,
                ctx_pairs, ws,
            )
            psi2 = _psi_score(
                scorer, hidden_dev,
                item.option_positions, item.option2_token_ids,
                ctx_pairs, ws,
            )
            s1 = lm1 + psi1
            s2 = lm2 + psi2
            pred = 1 if s1 >= s2 else 2
            records.append(Record(
                item_id=item.item_id,
                sentence=item.sentence,
                option1=item.option1,
                option2=item.option2,
                gold=item.gold,
                method="psi_rerank",
                config={"weight_scale": ws, "n_ctx": N_CTX},
                score1=s1,
                score2=s2,
                predicted=pred,
                correct=(pred == item.gold),
                wall_time_s=time.time() - t1,
            ))

    return records


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Scorer checkpoint path")
    p.add_argument(
        "--weight-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0],
    )
    p.add_argument("--max-items", type=int, default=None,
                   help="Cap for pilot runs (e.g. 100)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out", type=str,
        default=os.path.join(_REPO_ROOT, "results/m5c_wg_pilot.json"),
    )
    args = p.parse_args()

    print("[wg-eval] loading MDLM…")
    mdlm = MDLM.load()

    print(f"[wg-eval] loading scorer from {args.ckpt}…")
    scorer = _load_scorer(args.ckpt, mdlm.device)
    log_temp = float(scorer.log_temp.detach())
    print(f"[wg-eval] scorer log_temp={log_temp:.3f} (T={math.exp(log_temp):.2f})")

    print("[wg-eval] loading WinoGrande validation…")
    dataset = load_dataset("winogrande", "winogrande_xl", split="validation",
                           trust_remote_code=True)
    items, stats = build_items(
        dataset, mdlm.tokenizer, MASK_TOKEN_ID,
        max_items=args.max_items,
    )
    print(
        f"[wg-eval] {stats['ok']} items (skipped: "
        f"{stats['length_mismatch']} length-mismatch, "
        f"{stats['no_blank']} no-blank, "
        f"{stats['no_diff']} no-diff)"
    )

    records: list[dict] = []
    t_total = time.time()

    for idx, item in enumerate(items):
        recs = evaluate_item(item, mdlm, scorer, args.weight_scales)
        records.extend(asdict(r) for r in recs)

        if (idx + 1) % 10 == 0 or idx == 0:
            # Print running accuracy by method
            from collections import defaultdict
            correct_by_method: dict[str, list[bool]] = defaultdict(list)
            for r in records:
                key = r["method"] + (
                    f"_ws{r['config']['weight_scale']}"
                    if r["method"] == "psi_rerank" else ""
                )
                correct_by_method[key].append(r["correct"])
            acc_str = "  ".join(
                f"{k}={sum(v)/len(v):.3f}"
                for k, v in sorted(correct_by_method.items())
            )
            print(f"  [{idx+1}/{len(items)}]  {acc_str}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)

    # Final summary
    from collections import defaultdict
    correct_by_method: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        key = r["method"] + (
            f"_ws{r['config']['weight_scale']}"
            if r["method"] == "psi_rerank" else ""
        )
        correct_by_method[key].append(r["correct"])

    print(f"\n[wg-eval] === RESULTS ({len(items)} items) ===")
    for k in sorted(correct_by_method):
        v = correct_by_method[k]
        acc = sum(v) / len(v)
        print(f"  {k:35s}  {acc:.3f}  ({sum(v)}/{len(v)})")

    print(
        f"\n[wg-eval] wrote {len(records)} records to {args.out}"
        f"  ({time.time() - t_total:.1f}s total)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
