"""M5c — evaluate a trained learned scorer on the M5b template families.

For each (template, weight_scale) cell, builds a THRML factor graph
with the learned ψ and runs block-Gibbs. Compares the resulting
agreement rate and filled-sequence perplexity to:

  * the M4 hard-equality THRML baseline at the same equality_weight;
  * mask-predict @ T=0 (the strongest cascading baseline).

The output JSON has the same record schema as ``run_pareto.py`` so
``notebooks/06_learned.py`` can overlay the learned-ψ point cloud onto
the M5b boundary plot.

Run (after a training run):

    LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH" \\
    TRITON_LIBCUDA_PATH="/run/opengl-driver/lib" \\
        uv run python experiments/m5c_eval.py \\
            --ckpt results/m5c/scorer_step00200000.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from diffusion_ebm.backbones.mdlm import MASK_TOKEN_ID, MDLM  # noqa: E402
from diffusion_ebm.factors.learned import PairwiseScorer  # noqa: E402
from diffusion_ebm.metrics.agreement import (  # noqa: E402
    agreement_rate,
    lm_perplexity,
)
from diffusion_ebm.sampler import thrml_joint_learned  # noqa: E402
from diffusion_ebm.tasks.multihole import all_templates  # noqa: E402

N_CHAINS = 64


@dataclass
class Record:
    template: str
    method: str
    config: dict
    n_lm_forwards: int
    n_gibbs_sweeps: int
    n_chains: int
    agreement_rate: float
    lm_perplexity_mean: float
    wall_time_s: float
    seed: int


def _hole_groups(instance) -> list[list[int]]:
    pos_to_hole = {p: h for h, p in enumerate(instance.mask_positions)}
    return [[pos_to_hole[p] for p in g] for g in instance.equality_groups]


def _all_pairs(groups: list[list[int]]) -> list[tuple[int, int]]:
    edges = []
    for g in groups:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                edges.append((g[a], g[b]))
    return edges


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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--templates", choices=("core", "m5b", "all"), default="all")
    p.add_argument("--weight-scales", type=float, nargs="+",
                   default=[0.5, 1.0, 2.0, 5.0])
    p.add_argument("--burn-in", type=int, nargs="+", default=[100, 500])
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--temp-override", type=float, nargs="+", default=None,
        metavar="LOG_TEMP",
        help=(
            "Override log_temp to each value and run a full sweep per value. "
            "Values are in log-space: -1.0→T=0.37, -0.5→T=0.61, 0.0→T=1.0, "
            "0.5→T=1.65. If omitted, uses the trained log_temp."
        ),
    )
    p.add_argument(
        "--out",
        type=str,
        default=os.path.join(_REPO_ROOT, "results/m5c_eval.json"),
    )
    args = p.parse_args()

    print(f"[m5c-eval] loading MDLM…")
    mdlm = MDLM.load()
    scorer = _load_scorer(args.ckpt, mdlm.device)
    trained_log_temp = float(scorer.log_temp.detach())
    print(
        f"[m5c-eval] scorer loaded from {args.ckpt}  "
        f"(trained log_temp={trained_log_temp:.3f}, T={math.exp(trained_log_temp):.2f})"
    )

    templates = all_templates(mdlm.tokenizer, family=args.templates)
    records: list[dict] = []
    t_total = time.time()

    # Build MDLM features once per template (shared across all temp sweeps).
    template_feats: list[dict] = []
    for instance in templates:
        print(f"[m5c-eval] encoding: {instance.text[:60]!r}")
        logits, hidden = mdlm.forward_hidden(instance.masked_input_ids)
        logits = logits.float()
        ids, unary = mdlm.top_k_candidates(
            logits, k=args.k, exclude_mask_token=True
        )
        ids_holes = ids[0, instance.mask_positions, :]
        unary_holes = unary[0, instance.mask_positions, :]
        template_feats.append(dict(
            instance=instance,
            unary_jnp=jnp.asarray(unary_holes.cpu().numpy(), dtype=jnp.float32),
            ids_jnp=jnp.asarray(ids_holes.cpu().numpy(), dtype=jnp.int32),
            ids_torch=ids_holes.to(mdlm.device).long(),
            hidden=hidden[0],
            groups=_hole_groups(instance),
            pair_indices=_all_pairs(_hole_groups(instance)),
        ))

    # Determine which log_temp values to sweep.
    if args.temp_override:
        override_vals: list[float | None] = list(args.temp_override)
        print(
            f"[m5c-eval] temp overrides: {override_vals} "
            f"(trained={trained_log_temp:.3f})"
        )
    else:
        override_vals = [None]  # use trained log_temp as-is

    for temp_val in override_vals:
        if temp_val is not None:
            with torch.no_grad():
                scorer.log_temp.data.fill_(temp_val)
            print(
                f"\n[m5c-eval] === log_temp override {temp_val:.2f} "
                f"(T={math.exp(temp_val):.2f}) ==="
            )
        else:
            with torch.no_grad():
                scorer.log_temp.data.fill_(trained_log_temp)
            print(
                f"\n[m5c-eval] === trained log_temp {trained_log_temp:.3f} "
                f"(T={math.exp(trained_log_temp):.2f}) ==="
            )

        for tf in template_feats:
            instance = tf["instance"]
            unary_jnp = tf["unary_jnp"]
            ids_jnp = tf["ids_jnp"]
            ids_torch = tf["ids_torch"]
            groups = tf["groups"]
            pair_indices = tf["pair_indices"]
            print(f"[m5c-eval] template: {instance.text[:60]!r}")

            for ws in args.weight_scales:
                for burn in args.burn_in:
                    t0 = time.time()
                    sampler = thrml_joint_learned.build(
                        unary=unary_jnp,
                        candidate_ids=ids_jnp,
                        candidate_ids_torch=ids_torch,
                        hidden=tf["hidden"],
                        hole_positions=instance.mask_positions,
                        pair_indices=pair_indices,
                        scorer=scorer,
                        weight_scale=ws,
                    )
                    samples_jnp = thrml_joint_learned.sample(
                        sampler,
                        jax.random.PRNGKey(args.seed),
                        n_chains=N_CHAINS,
                        burn_in=burn,
                    )
                    samples = torch.from_numpy(np.asarray(samples_jnp)).long()
                    agree = agreement_rate(samples, groups)
                    ppl = float(lm_perplexity(mdlm, samples, instance).mean())
                    wall = time.time() - t0
                    rec = Record(
                        template=instance.text,
                        method="thrml_joint_learned",
                        config=dict(
                            weight_scale=ws,
                            gibbs_sweeps=burn,
                            k=args.k,
                            ckpt=os.path.basename(args.ckpt),
                            log_temp_override=temp_val,
                        ),
                        n_lm_forwards=1,
                        n_gibbs_sweeps=burn + N_CHAINS,
                        n_chains=N_CHAINS,
                        agreement_rate=agree,
                        lm_perplexity_mean=ppl,
                        wall_time_s=wall,
                        seed=args.seed,
                    )
                    records.append(asdict(rec))
                    print(
                        f"  ws={ws:5.2f} burn={burn:>4d}  "
                        f"agree={agree:.3f}  ppl={ppl:5.2f}  t={wall:5.2f}s"
                    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(
        f"[m5c-eval] wrote {len(records)} records to {args.out}"
        f"  ({time.time() - t_total:.1f}s total)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
