"""THRML block-Gibbs over a frustrated (attraction + repulsion) token graph.

Track B reframes repeated-mention recovery as **correlation clustering**:
holes that co-refer should share a token (attraction), holes that name distinct
entities should take different tokens (repulsion). Attraction + repulsion on
one graph is a *frustrated* Potts posterior — multimodal, NP-hard exact MAP,
and the canonical thermodynamic-hardware workload.

This module assembles that frustrated graph over per-position top-k candidate
sets and samples tokens with THRML block-Gibbs:

* one **unary** ``CategoricalEBMFactor`` (LM logits as per-position bias);
* one **equality** factor (``+attract_weight`` on the equal-id diagonal) over
  the attraction edges, via ``factors/equality.py``;
* one **inequality** factor (``-repel_weight`` on the equal-id diagonal) over
  the repulsion edges, via ``factors/inequality.py``.

Which pairs attract and which repel is set by an entity **partition** of the
holes (the latent of interest). ``partition_to_edges`` turns a hole->group
labelling into within-group attraction edges and cross-group repulsion edges;
``jaccard_partition`` produces an initial partition from the de-oracle top-k
overlap signal (the same feature ``predicted_group_hard_eq`` uses) so the graph
is built from the corrupted input alone — no gold grouping.

Scope: this file implements the **token step** — sampling ``x`` under a *fixed*
partition — plus the partition/edge plumbing. The full joint ``(z_i, x_i)``
block-Gibbs (alternating this token step with a partition-resampling step) is
the Track B research loop built on these primitives; see
``plans/shiny-watching-sundae.md``, Track B item 2.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from diffusion_ebm.factors.inequality import stack_inequality_factor
from diffusion_ebm.factors.unary import unary_weight_tensor
from diffusion_ebm.utils.coloring import chromatic_blocks


@dataclass
class FrustratedSampler:
    program: FactorSamplingProgram
    target: Block
    free_blocks: list[Block]
    n_holes: int
    k: int
    candidate_ids: jnp.ndarray  # (n_holes, k) int32
    attract_edges: list[tuple[int, int]]
    repel_edges: list[tuple[int, int]]


def partition_to_edges(
    partition: list[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split hole pairs into attraction (same group) and repulsion (distinct).

    Args:
        partition: ``partition[i]`` is the integer group label of hole ``i``.

    Returns:
        ``(attract_edges, repel_edges)``, each a list of ``(i, j)`` with
        ``i < j``. Same-label pairs attract; different-label pairs repel.
    """
    n = len(partition)
    attract: list[tuple[int, int]] = []
    repel: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            (attract if partition[i] == partition[j] else repel).append((i, j))
    return attract, repel


def jaccard_partition(topk_sets: list[set[int]], threshold: float) -> list[int]:
    """De-oracle partition: union-find over top-k Jaccard overlap.

    Two holes whose top-k candidate sets overlap with Jaccard >= ``threshold``
    are merged; connected components are the predicted entity groups. Mirrors
    ``experiments/m5c_eval_rmc.py:_predict_groups_jaccard`` but returns a flat
    hole->group label list. Sees only the corrupted input's MDLM top-k.
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
            union_sz = len(topk_sets[a] | topk_sets[b])
            jac = len(topk_sets[a] & topk_sets[b]) / union_sz if union_sz else 0.0
            if jac >= threshold:
                union(a, b)

    roots = {find(i): None for i in range(n)}
    label_of = {root: lbl for lbl, root in enumerate(roots)}
    return [label_of[find(i)] for i in range(n)]


def _stack_pair_ids(
    candidate_ids: jnp.ndarray, edges: list[tuple[int, int]]
) -> jnp.ndarray:
    """Gather candidate ids for an edge list into ``(n_edges, 2, k)``."""
    return jnp.stack(
        [jnp.stack([candidate_ids[i], candidate_ids[j]]) for i, j in edges]
    )


def build(
    unary: jnp.ndarray,                 # (n_holes, k) — LM logits at top-k candidates
    candidate_ids: jnp.ndarray,         # (n_holes, k) — vocab IDs the candidates decode to
    attract_edges: list[tuple[int, int]],
    repel_edges: list[tuple[int, int]],
    attract_weight: float = 5.0,
    repel_weight: float = 5.0,
    k: int = 64,
) -> FrustratedSampler:
    """Assemble a frustrated `FactorSamplingProgram` for one RMC item.

    Attraction edges carry an equality factor (reward shared token); repulsion
    edges carry an inequality factor (penalise shared token). Block coloring is
    over the union of both edge sets.
    """
    if unary.ndim != 2 or candidate_ids.shape != unary.shape:
        raise ValueError(
            f"unary {unary.shape} and candidate_ids {candidate_ids.shape} "
            "must both be (n_holes, k) and match"
        )
    n_holes, k_in = unary.shape
    if k_in != k:
        raise ValueError(f"unary last-dim {k_in} != requested k={k}")

    nodes = [CategoricalNode() for _ in range(n_holes)]
    factors = [CategoricalEBMFactor([Block(nodes)], unary_weight_tensor(unary))]

    if attract_edges:
        left = [nodes[i] for i, _ in attract_edges]
        right = [nodes[j] for _, j in attract_edges]
        w = stack_equality_factor(
            _stack_pair_ids(candidate_ids, attract_edges), weight=attract_weight
        )
        factors.append(CategoricalEBMFactor([Block(left), Block(right)], w))

    if repel_edges:
        left = [nodes[i] for i, _ in repel_edges]
        right = [nodes[j] for _, j in repel_edges]
        w = stack_inequality_factor(
            _stack_pair_ids(candidate_ids, repel_edges), weight=repel_weight
        )
        factors.append(CategoricalEBMFactor([Block(left), Block(right)], w))

    free_blocks = chromatic_blocks(nodes, attract_edges + repel_edges)
    spec = BlockGibbsSpec(free_blocks, [])
    samp = CategoricalGibbsConditional(k)
    program = FactorSamplingProgram(spec, [samp] * len(free_blocks), factors, [])

    return FrustratedSampler(
        program=program,
        target=Block(nodes),
        free_blocks=free_blocks,
        n_holes=n_holes,
        k=k,
        candidate_ids=candidate_ids,
        attract_edges=attract_edges,
        repel_edges=repel_edges,
    )


def _decode(samples: jnp.ndarray, candidate_ids: jnp.ndarray) -> jnp.ndarray:
    """Map (n_chains, n_holes) candidate-index samples -> vocab IDs."""
    return jnp.take_along_axis(
        candidate_ids[None, :, :].astype(jnp.int32),
        samples.astype(jnp.int32)[..., None],
        axis=-1,
    ).squeeze(-1)


def sample(
    sampler: FrustratedSampler,
    key,
    n_chains: int = 64,
    burn_in: int = 200,
) -> jnp.ndarray:
    """Run THRML block-Gibbs and return ``(n_chains, n_holes)`` vocab IDs."""
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
    return _decode(out[0], sampler.candidate_ids)
