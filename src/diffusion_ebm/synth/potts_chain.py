"""Synthetic Potts chain for the M1 sampler sanity check.

A length-N chain of categorical-K variables with a pairwise ferromagnetic
factor table on the diagonal.  Per-position marginals are uniform (1/K) but
the Boltzmann distribution concentrates on the K all-equal configurations.
The MVP0 test compares THRML block-Gibbs samples against marginal-only
independent sampling.
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


@dataclass
class PottsChain:
    program: FactorSamplingProgram
    target: Block
    even: Block
    odd: Block
    n: int
    k: int


def ferromagnetic_table(k: int, J: float) -> jnp.ndarray:
    """Pairwise compatibility weights — `J` on the diagonal, 0 elsewhere."""
    return jnp.eye(k) * J


def make_chain(n: int = 20, k: int = 5, J: float = 5.0) -> PottsChain:
    """Build a length-N, K-state ferromagnetic Potts chain.

    The pairwise factor is shared across all (n - 1) edges via broadcasting,
    so we use a single `CategoricalEBMFactor` whose weight tensor has shape
    `(n - 1, k, k)` — the leading axis is the per-edge axis used by THRML.
    """
    nodes = [CategoricalNode() for _ in range(n)]

    table = ferromagnetic_table(k, J)
    weights = jnp.broadcast_to(table, (n - 1, k, k))
    factor = CategoricalEBMFactor([Block(nodes[:-1]), Block(nodes[1:])], weights)

    even = Block([nodes[i] for i in range(0, n, 2)])
    odd = Block([nodes[i] for i in range(1, n, 2)])
    spec = BlockGibbsSpec([even, odd], [])

    samp = CategoricalGibbsConditional(k)
    program = FactorSamplingProgram(spec, [samp, samp], [factor], [])

    return PottsChain(program=program, target=Block(nodes), even=even, odd=odd, n=n, k=k)


def aligned_fraction(samples: jnp.ndarray) -> jnp.ndarray:
    """Fraction of consecutive pairs that agree, per sample. Shape `[...]`."""
    return (samples[..., 1:] == samples[..., :-1]).mean(axis=-1)


def independent_samples(key, n: int, k: int, n_samples: int) -> jnp.ndarray:
    """Uniform per-position categorical — the marginal of the unconstrained chain."""
    return jax.random.randint(key, (n_samples, n), 0, k, dtype=jnp.uint8)


def thrml_samples(key, chain: PottsChain, n_samples: int, warmup: int = 500) -> jnp.ndarray:
    """Run THRML block-Gibbs and return samples shaped `[n_samples, n]` (uint8)."""
    state_free_init = [
        jnp.zeros(len(chain.even.nodes), dtype=jnp.uint8),
        jnp.zeros(len(chain.odd.nodes), dtype=jnp.uint8),
    ]
    schedule = SamplingSchedule(warmup, n_samples, 1)
    out = sample_states(
        key,
        chain.program,
        schedule,
        state_free_init,
        [],
        [chain.target],
    )
    return out[0]
