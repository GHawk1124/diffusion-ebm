"""THRML joint block-Gibbs sampler over per-position top-k candidate sets.

Given an MDLM forward pass, we extract `n_holes` masked positions, each
with `k` top-k candidate token IDs and corresponding LM logits.  Over
that finite state space we build:

* one **unary** `CategoricalEBMFactor` carrying the LM logits as
  per-position biases (single block, weights shape `(n_holes, k)`);
* one **equality** `CategoricalEBMFactor` per matched pair, stacked into
  a single factor with weights shape `(n_edges, k, k)` — entry `[a, b]`
  is `+equality_weight` iff the two candidate IDs match, else 0.

Free-block coloring is delegated to
``src/diffusion_ebm/utils/coloring.py`` (DSATUR via NetworkX).  The
resulting `FactorSamplingProgram` is consumed by ``thrml.sample_states``.

Sampling returns vocab-ID samples shaped ``(n_chains, n_holes)``; the
candidate-index → vocab-ID gather is done here so callers don't see
THRML's internal uint8 indexing.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from thrml import (
    Block,
    BlockGibbsSpec,
    CategoricalNode,
    FactorSamplingProgram,
    SamplingSchedule,
    sample_states,
)
from thrml.models import CategoricalEBMFactor, CategoricalGibbsConditional

from diffusion_ebm.factors.equality import stack_equality_factor
from diffusion_ebm.factors.unary import unary_weight_tensor
from diffusion_ebm.utils.coloring import chromatic_blocks


@dataclass
class JointSampler:
    program: FactorSamplingProgram
    target: Block
    free_blocks: list[Block]
    n_holes: int
    k: int
    candidate_ids: jnp.ndarray  # (n_holes, k) int32


def _all_pairs_within_groups(
    equality_groups: list[list[int]],
) -> list[tuple[int, int]]:
    """Edge list induced by 'all positions in a group must agree'."""
    edges: list[tuple[int, int]] = []
    for g in equality_groups:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                edges.append((g[a], g[b]))
    return edges


def build(
    unary: jnp.ndarray,                 # (n_holes, k) — LM logits at top-k candidates
    candidate_ids: jnp.ndarray,         # (n_holes, k) — vocab IDs the candidates decode to
    equality_groups: list[list[int]],
    equality_weight: float = 5.0,
    k: int = 64,
) -> JointSampler:
    """Assemble a `FactorSamplingProgram` for one MultiHoleInstance."""
    if unary.ndim != 2 or candidate_ids.shape != unary.shape:
        raise ValueError(
            f"unary {unary.shape} and candidate_ids {candidate_ids.shape} "
            "must both be (n_holes, k) and match"
        )
    n_holes, k_in = unary.shape
    if k_in != k:
        raise ValueError(f"unary last-dim {k_in} != requested k={k}")

    nodes = [CategoricalNode() for _ in range(n_holes)]
    edges = _all_pairs_within_groups(equality_groups)

    factors = [
        CategoricalEBMFactor([Block(nodes)], unary_weight_tensor(unary)),
    ]

    if edges:
        left_nodes = [nodes[i] for i, _ in edges]
        right_nodes = [nodes[j] for _, j in edges]
        pair_ids = jnp.stack(
            [jnp.stack([candidate_ids[i], candidate_ids[j]]) for i, j in edges]
        )  # (n_edges, 2, k)
        eq_weights = stack_equality_factor(pair_ids, weight=equality_weight)
        factors.append(
            CategoricalEBMFactor(
                [Block(left_nodes), Block(right_nodes)], eq_weights
            )
        )

    free_blocks = chromatic_blocks(nodes, edges)
    spec = BlockGibbsSpec(free_blocks, [])
    samp = CategoricalGibbsConditional(k)
    program = FactorSamplingProgram(spec, [samp] * len(free_blocks), factors, [])

    return JointSampler(
        program=program,
        target=Block(nodes),
        free_blocks=free_blocks,
        n_holes=n_holes,
        k=k,
        candidate_ids=candidate_ids,
    )


def _decode(samples: jnp.ndarray, candidate_ids: jnp.ndarray) -> jnp.ndarray:
    """Map (n_chains, n_holes) candidate-index samples → vocab IDs."""
    return jnp.take_along_axis(
        candidate_ids[None, :, :].astype(jnp.int32),
        samples.astype(jnp.int32)[..., None],
        axis=-1,
    ).squeeze(-1)


def sample(
    sampler: JointSampler,
    key,
    n_chains: int = 64,
    burn_in: int = 200,
) -> jnp.ndarray:
    """Run THRML block-Gibbs and return ``(n_chains, n_holes)`` vocab IDs.

    `n_chains` is realised as `n_chains` post-burn-in samples from a single
    chain (THRML's `SamplingSchedule(warmup, n_samples, thinning=1)` form).
    For first-pass M3 metrics this is sufficient; switch to `vmap` over
    PRNG keys later if independence between chains is required.
    """
    state_free_init = [
        jnp.zeros(len(b.nodes), dtype=jnp.uint8) for b in sampler.free_blocks
    ]
    schedule = SamplingSchedule(burn_in, n_chains, 1)
    out = sample_states(
        key,
        sampler.program,
        schedule,
        state_free_init,
        [],
        [sampler.target],
    )
    samples = out[0]  # (n_chains, n_holes), uint8 candidate indices
    return _decode(samples, sampler.candidate_ids)
