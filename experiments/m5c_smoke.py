"""M5c smoke — validate the full M5c pipeline locally before long training.

Three checks:

1. Train ``PairwiseScorer`` for a few hundred steps on the synthetic
   corpus and verify the InfoNCE loss decreases by ≥ 30 %.
2. Load the resulting checkpoint, build a ``thrml_joint_learned``
   sampler on one M3 template, and verify it runs without error.
3. Evaluate agreement on the polyseme M5b template with the trained
   scorer — must be ≥ 0 (we don't expect dominance from the smoke
   training run, only that the pipeline is wired correctly).

This is the contract that has to hold before a long training run.

Run:

    LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH" \\
    TRITON_LIBCUDA_PATH="/run/opengl-driver/lib" \\
        uv run python experiments/m5c_smoke.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from diffusion_ebm.backbones.mdlm import MDLM, MASK_TOKEN_ID  # noqa: E402
from diffusion_ebm.factors.learned import PairwiseScorer  # noqa: E402
from diffusion_ebm.metrics.agreement import agreement_rate  # noqa: E402
from diffusion_ebm.sampler import thrml_joint_learned  # noqa: E402
from diffusion_ebm.tasks.multihole import (  # noqa: E402
    color_template,
    polyseme_template,
)


CKPT_DIR = os.path.join(_REPO_ROOT, "results/m5c_smoke")
N_CHAINS = 64


def _hole_groups(instance) -> list[list[int]]:
    pos_to_hole = {p: h for h, p in enumerate(instance.mask_positions)}
    return [[pos_to_hole[p] for p in g] for g in instance.equality_groups]


def _all_pairs(equality_groups: list[list[int]]) -> list[tuple[int, int]]:
    edges = []
    for g in equality_groups:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                edges.append((g[a], g[b]))
    return edges


def _train_smoke() -> str:
    """Run the training script with a tiny config; return ckpt path."""
    cmd = [
        sys.executable,
        os.path.join(_REPO_ROOT, "experiments/m5c_train.py"),
        "--corpus", "synthetic",
        "--steps", "300",
        "--batch", "16",
        "--seq-len", "32",
        "--lr-warmup-steps", "30",
        "--min-pair-dist", "3",
        "--lr", "3e-4",
        "--embed-dim", "64",
        "--head-dim", "32",
        "--mlp-dim", "128",
        "--log-every", "30",
        "--ckpt-every", "300",
        "--ckpt-dir", CKPT_DIR,
        "--seed", "0",
    ]
    print("[smoke] launching training:")
    print("  " + " ".join(cmd))
    env = dict(os.environ)
    env.setdefault(
        "LD_LIBRARY_PATH",
        "/run/opengl-driver/lib:" + env.get("LD_LIBRARY_PATH", ""),
    )
    env.setdefault("TRITON_LIBCUDA_PATH", "/run/opengl-driver/lib")
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        raise SystemExit(f"training script returned {rc}")

    log_path = os.path.join(CKPT_DIR, "log.json")
    with open(log_path) as f:
        log = json.load(f)
    if not log:
        raise SystemExit("training log is empty")
    first, last = log[0]["loss"], log[-1]["loss"]
    drop = (first - last) / max(first, 1e-6)
    print(f"[smoke] loss {first:.3f} → {last:.3f}  ({drop * 100:.1f}% drop)")
    if drop < 0.3:
        raise SystemExit(
            f"loss decreased by only {drop * 100:.1f}% (< 30%); training broken"
        )

    ckpt = os.path.join(CKPT_DIR, "scorer_step00000300.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found at {ckpt}")
    return ckpt


def _load_scorer(ckpt_path: str, device: torch.device) -> PairwiseScorer:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = blob["config"]
    scorer = PairwiseScorer(
        hidden_dim=cfg["hidden_dim"],
        embed_dim=cfg["embed_dim"],
        head_dim=cfg["head_dim"],
        mlp_dim=blob.get("args", {}).get("mlp_dim", 128),
        vocab_size=cfg["vocab_size"],
    ).to(device)
    scorer.load_state_dict(blob["state_dict"])
    scorer.eval()
    return scorer


def _eval_factor_graph(scorer: PairwiseScorer, mdlm: MDLM) -> dict:
    """Build a learned-factor THRML sampler and run it on color + polyseme."""
    out = {}
    for label, builder in (("color", color_template), ("polyseme", polyseme_template)):
        instance = builder(mdlm.tokenizer)
        logits, hidden = mdlm.forward_hidden(instance.masked_input_ids)
        logits = logits.float()
        ids, unary = mdlm.top_k_candidates(logits, k=64, exclude_mask_token=True)
        ids_holes = ids[0, instance.mask_positions, :]   # [n_holes, k]
        unary_holes = unary[0, instance.mask_positions, :]

        unary_jnp = jnp.asarray(unary_holes.cpu().numpy(), dtype=jnp.float32)
        ids_jnp = jnp.asarray(ids_holes.cpu().numpy(), dtype=jnp.int32)
        ids_torch = ids_holes.to(mdlm.device).long()

        groups = _hole_groups(instance)
        pair_indices = _all_pairs(groups)
        sampler = thrml_joint_learned.build(
            unary=unary_jnp,
            candidate_ids=ids_jnp,
            candidate_ids_torch=ids_torch,
            hidden=hidden[0],
            hole_positions=instance.mask_positions,
            pair_indices=pair_indices,
            scorer=scorer,
            weight_scale=1.0,
        )
        samples_jnp = thrml_joint_learned.sample(
            sampler, jax.random.PRNGKey(0), n_chains=N_CHAINS, burn_in=200
        )
        samples = torch.from_numpy(np.asarray(samples_jnp)).long()
        ar = agreement_rate(samples, groups)
        out[label] = ar
        print(f"[smoke] {label}: agreement={ar:.3f}  (sample size {N_CHAINS})")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="reuse the existing smoke checkpoint",
    )
    args = parser.parse_args()

    if args.skip_train:
        ckpt = os.path.join(CKPT_DIR, "scorer_step00000300.pt")
        if not os.path.exists(ckpt):
            raise SystemExit(
                "no checkpoint at "
                + ckpt
                + "; drop --skip-train"
            )
    else:
        ckpt = _train_smoke()

    print("[smoke] loading MDLM and trained scorer…")
    mdlm = MDLM.load()
    scorer = _load_scorer(ckpt, mdlm.device)
    n_params = sum(p.numel() for p in scorer.parameters())
    print(f"[smoke] scorer params: {n_params/1e6:.2f}M")

    rates = _eval_factor_graph(scorer, mdlm)

    # Smoke contract: pipeline runs end-to-end and produces finite values.
    for k, v in rates.items():
        if not (0.0 <= v <= 1.0):
            raise SystemExit(f"{k} agreement out of range: {v}")

    print("[smoke] M5c smoke checks PASSED")
    print(json.dumps(rates, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
