"""M3 — multi-hole agreement headline experiment.

For each multi-hole template, we run four methods and compare:

* **thrml_joint** — single MDLM forward → top-k=64 → THRML block-Gibbs
  with hard equality factors over matched holes.
* **ancestral_topk** — single MDLM forward → independent multinomial
  draws from top-k softmax (same state space as THRML, no joint
  signal).
* **mask_predict** — MDLM's own iterative refinement, at a few budgets.
* **independent_full** — single MDLM forward, full-vocab independent
  multinomial.

Headline metric is **agreement_rate** under each template's equality
groups; we also report mean LM-perplexity of the filled sequence so a
high agreement on garbage tokens is visible.

Run:

    LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH" \
    TRITON_LIBCUDA_PATH="/run/opengl-driver/lib" \
        uv run python notebooks/03_mvp1_multihole.py
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from diffusion_ebm.backbones.mdlm import MDLM, MASK_TOKEN_ID  # noqa: E402
from diffusion_ebm.metrics.agreement import (  # noqa: E402
    Cost,
    agreement_rate,
    lm_perplexity,
)
from diffusion_ebm.sampler import baselines  # noqa: E402
from diffusion_ebm.sampler import thrml_joint  # noqa: E402
from diffusion_ebm.tasks.multihole import (  # noqa: E402
    MultiHoleInstance,
    all_templates,
)

torch.manual_seed(0)
SEED = 0
N_CHAINS = 64
K = 64
EQUALITY_WEIGHT = 5.0
THRML_BURN_IN = 200


def to_hole_indices(
    equality_groups: list[list[int]],
    mask_positions: list[int],
) -> list[list[int]]:
    """Translate token-position groups → hole-index groups (0..n_holes-1)."""
    pos_to_hole = {p: h for h, p in enumerate(mask_positions)}
    return [[pos_to_hole[p] for p in g] for g in equality_groups]


def topk_for_thrml(
    mdlm: MDLM, instance: MultiHoleInstance, k: int = K
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One MDLM forward → (candidate_ids, unary) at hole positions, as JAX f32/i32."""
    logits = mdlm.forward(instance.masked_input_ids)
    ids, unary = mdlm.top_k_candidates(logits, k=k, exclude_mask_token=True)
    ids_holes = ids[0, instance.mask_positions, :]            # torch (n_holes, k)
    unary_holes = unary[0, instance.mask_positions, :]        # torch bf16
    ids_jnp = jnp.asarray(ids_holes.cpu().numpy(), dtype=jnp.int32)
    unary_jnp = jnp.asarray(unary_holes.float().cpu().numpy(), dtype=jnp.float32)
    return ids_jnp, unary_jnp


def run_thrml(
    instance: MultiHoleInstance, candidate_ids, unary, n_chains: int = N_CHAINS
) -> torch.Tensor:
    hole_groups = to_hole_indices(instance.equality_groups, instance.mask_positions)
    sampler = thrml_joint.build(
        unary=unary,
        candidate_ids=candidate_ids,
        equality_groups=hole_groups,
        equality_weight=EQUALITY_WEIGHT,
        k=K,
    )
    out_jnp = thrml_joint.sample(
        sampler, jax.random.PRNGKey(SEED), n_chains=n_chains, burn_in=THRML_BURN_IN
    )
    return torch.from_numpy(np.asarray(out_jnp)).long()


def fmt_row(template_text: str, method: str, agree: float, ppl: float, cost: Cost) -> str:
    return (
        f"  {method:<22s}  agree={agree:5.3f}  ppl={ppl:8.2f}  "
        f"cost(LM={cost.n_lm_forwards}, gibbs={cost.n_gibbs_sweeps})"
    )


def main() -> None:
    mdlm = MDLM.load()
    print(f"Loaded MDLM on {mdlm.device}\n")

    methods = [
        ("thrml_joint",       lambda inst, cid, un: (run_thrml(inst, cid, un), Cost(1, THRML_BURN_IN + N_CHAINS))),
        ("ancestral_topk",    lambda inst, cid, un: (baselines.ancestral_topk(mdlm, inst, k=K, n_chains=N_CHAINS, seed=SEED), Cost(1, 0))),
        ("independent_full",  lambda inst, cid, un: (baselines.independent_full(mdlm, inst, n_chains=N_CHAINS, seed=SEED), Cost(1, 0))),
        ("mask_predict@1",    lambda inst, cid, un: (baselines.mask_predict(mdlm, inst, n_iters=1, n_chains=N_CHAINS, temperature=1.0, seed=SEED), Cost(1, 0))),
        ("mask_predict@4",    lambda inst, cid, un: (baselines.mask_predict(mdlm, inst, n_iters=4, n_chains=N_CHAINS, temperature=1.0, seed=SEED), Cost(4, 0))),
        ("mask_predict@8",    lambda inst, cid, un: (baselines.mask_predict(mdlm, inst, n_iters=8, n_chains=N_CHAINS, temperature=1.0, seed=SEED), Cost(8, 0))),
    ]

    overall: dict[str, dict[str, tuple[float, float, Cost]]] = {}

    for instance in all_templates(mdlm.tokenizer):
        print(f"= Template: {instance.text!r}")
        print(f"  ({len(instance.mask_positions)} holes at positions {instance.mask_positions})")
        candidate_ids, unary = topk_for_thrml(mdlm, instance, k=K)
        hole_groups = to_hole_indices(instance.equality_groups, instance.mask_positions)
        per_template: dict[str, tuple[float, float, Cost]] = {}
        for name, fn in methods:
            samples, cost = fn(instance, candidate_ids, unary)
            agree = agreement_rate(samples, hole_groups)
            ppl = float(lm_perplexity(mdlm, samples, instance).mean())
            per_template[name] = (agree, ppl, cost)
            print(fmt_row(instance.text, name, agree, ppl, cost))
        overall[instance.text] = per_template
        print()

    # Summary table
    print("=" * 78)
    print("Agreement-rate summary (cols = templates, rows = methods)")
    print("=" * 78)
    template_keys = list(overall.keys())
    short = [t[:30] for t in template_keys]
    print(f"  {'method':<22s}  " + "  ".join(f"{s:>30s}" for s in short))
    for name, _ in methods:
        cells = [f"{overall[t][name][0]:>30.3f}" for t in template_keys]
        print(f"  {name:<22s}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
