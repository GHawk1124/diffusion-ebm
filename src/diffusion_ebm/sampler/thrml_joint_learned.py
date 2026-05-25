"""THRML block-Gibbs joint sampler with a *learned* pairwise factor.

This is the M5c parallel of ``thrml_joint``: same node + unary + coloring
machinery, but the per-edge weight tensor is produced by a torch
``PairwiseScorer`` (see ``diffusion_ebm.factors.learned``) instead of
the hard equality table from ``factors.equality``.

Use:

    from diffusion_ebm.factors.learned import PairwiseScorer
    from diffusion_ebm.sampler import thrml_joint_learned

    sampler = thrml_joint_learned.build(
        unary=unary_jnp,                # (n_holes, k)
        candidate_ids=cand_jnp,         # (n_holes, k)
        candidate_ids_torch=cand_torch, # (n_holes, k) long, on scorer device
        hidden=last_hidden,             # (L, d_h) torch float32
        hole_positions=instance.mask_positions,
        pair_indices=pair_indices,      # all-pairs-within-groups
        scorer=scorer,
        weight_scale=1.0,
    )
    out = thrml_joint_learned.sample(sampler, key, n_chains, burn_in)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import torch
from thrml import (
    Block,
    BlockGibbsSpec,
    CategoricalNode,
    FactorSamplingProgram,
    SamplingSchedule,
    sample_states,
)
from thrml.models import CategoricalEBMFactor, CategoricalGibbsConditional

from diffusion_ebm.factors.learned import PairwiseScorer, stack_learned_factor
from diffusion_ebm.factors.unary import unary_weight_tensor
from diffusion_ebm.utils.coloring import chromatic_blocks


@dataclass
class LearnedJointSampler:
    program: FactorSamplingProgram
    target: Block
    free_blocks: list[Block]
    n_holes: int
    k: int
    candidate_ids: jnp.ndarray  # (n_holes, k) int32


def build(
    unary: jnp.ndarray,                # (n_holes, k)
    candidate_ids: jnp.ndarray,        # (n_holes, k) — vocab IDs (jax)
    candidate_ids_torch: torch.Tensor,  # (n_holes, k) long — same vals on scorer device
    hidden: torch.Tensor,              # (L, d_h) — MDLM last hidden state
    hole_positions: list[int],
    pair_indices: list[tuple[int, int]],
    scorer: PairwiseScorer,
    weight_scale: float = 1.0,
) -> LearnedJointSampler:
    if unary.ndim != 2 or candidate_ids.shape != unary.shape:
        raise ValueError(
            f"unary {unary.shape} and candidate_ids {candidate_ids.shape}"
            " must match"
        )
    n_holes, k = unary.shape

    nodes = [CategoricalNode() for _ in range(n_holes)]
    factors = [
        CategoricalEBMFactor([Block(nodes)], unary_weight_tensor(unary)),
    ]

    if pair_indices:
        learned_w = stack_learned_factor(
            scorer=scorer,
            hidden=hidden,
            candidate_ids=candidate_ids_torch,
            hole_positions=hole_positions,
            pair_indices=pair_indices,
            weight_scale=weight_scale,
        )
        left_nodes = [nodes[i] for i, _ in pair_indices]
        right_nodes = [nodes[j] for _, j in pair_indices]
        factors.append(
            CategoricalEBMFactor(
                [Block(left_nodes), Block(right_nodes)], learned_w
            )
        )

    free_blocks = chromatic_blocks(nodes, list(pair_indices))
    spec = BlockGibbsSpec(free_blocks, [])
    samp = CategoricalGibbsConditional(k)
    program = FactorSamplingProgram(spec, [samp] * len(free_blocks), factors, [])

    return LearnedJointSampler(
        program=program,
        target=Block(nodes),
        free_blocks=free_blocks,
        n_holes=n_holes,
        k=k,
        candidate_ids=candidate_ids,
    )


def _decode(samples: jnp.ndarray, candidate_ids: jnp.ndarray) -> jnp.ndarray:
    return jnp.take_along_axis(
        candidate_ids[None, :, :].astype(jnp.int32),
        samples.astype(jnp.int32)[..., None],
        axis=-1,
    ).squeeze(-1)


def sample(
    sampler: LearnedJointSampler,
    key,
    n_chains: int = 64,
    burn_in: int = 200,
) -> jnp.ndarray:
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
    samples = out[0]
    return _decode(samples, sampler.candidate_ids)
