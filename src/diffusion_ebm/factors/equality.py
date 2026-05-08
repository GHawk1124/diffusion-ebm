"""Equality factor weight-table helpers for thrml integration.

These tables are fed into a thrml.CategoricalEBMFactor as the per-edge weight
tensor for hard "the two holes must take the same vocab id" factors. The
calling code lives in src/diffusion_ebm/sampler/thrml_joint.py.
"""

import jax
import jax.numpy as jnp


def equality_weight_table(
    ids_i: jnp.ndarray,
    ids_j: jnp.ndarray,
    weight: float = 5.0,
) -> jnp.ndarray:
    """Build a pairwise equality weight table.

    Args:
        ids_i: Integer vocab ids for the first hole, with shape ``(k,)``.
        ids_j: Integer vocab ids for the second hole, with shape ``(k,)``.
        weight: Weight assigned when vocab ids are equal.

    Returns:
        A float32 array with shape ``(k, k)`` where entry ``[a, b]`` is
        ``weight`` if ``ids_i[a] == ids_j[b]``, otherwise ``0.0``.
    """
    return jnp.where((ids_i[:, None] == ids_j[None, :]), weight, 0.0).astype(
        jnp.float32
    )


def stack_equality_factor(
    pair_ids: jnp.ndarray,
    weight: float = 5.0,
) -> jnp.ndarray:
    """Vectorize equality weight-table construction over pairs.

    Args:
        pair_ids: Integer vocab ids with shape ``(n_pairs, 2, k)``.
        weight: Weight assigned when vocab ids are equal.

    Returns:
        A float32 array with shape ``(n_pairs, k, k)`` containing one equality
        weight table per leading pair.
    """
    return jax.vmap(lambda ids: equality_weight_table(ids[0], ids[1], weight))(
        pair_ids
    ).astype(jnp.float32)
