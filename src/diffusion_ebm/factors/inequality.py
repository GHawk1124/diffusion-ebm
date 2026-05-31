"""Inequality (antiferromagnetic / repulsion) factor weight-table helpers.

Track B's frustrated correlation-clustering posterior couples masked holes
with two competing factors:

* the existing **equality** factor (``factors/equality.py``) — holes believed
  to be the *same* entity are rewarded for taking the same vocab id, and
* this **inequality** factor — holes believed to be *distinct* entities pay an
  energy penalty for taking the same vocab id (an antiferromagnetic coupling).

Both feed ``thrml.CategoricalEBMFactor`` as the per-edge weight tensor. THRML
adds the weight to the joint log-probability, so a *positive* weight on the
equal-id diagonal favours agreement (equality) while a *negative* weight on the
equal-id diagonal penalises it (inequality). The calling code lives in
``src/diffusion_ebm/sampler/thrml_latent_partition.py``.
"""

import jax
import jax.numpy as jnp


def inequality_weight_table(
    ids_i: jnp.ndarray,
    ids_j: jnp.ndarray,
    weight: float = 5.0,
) -> jnp.ndarray:
    """Build a pairwise repulsion weight table.

    Args:
        ids_i: Integer vocab ids for the first hole, with shape ``(k,)``.
        ids_j: Integer vocab ids for the second hole, with shape ``(k,)``.
        weight: Non-negative repulsion strength.

    Returns:
        A float32 array with shape ``(k, k)`` where entry ``[a, b]`` is
        ``-weight`` if ``ids_i[a] == ids_j[b]`` (two distinct entities would be
        sharing a token — penalised), otherwise ``0.0``.
    """
    return jnp.where((ids_i[:, None] == ids_j[None, :]), -weight, 0.0).astype(
        jnp.float32
    )


def stack_inequality_factor(
    pair_ids: jnp.ndarray,
    weight: float = 5.0,
) -> jnp.ndarray:
    """Vectorize repulsion-table construction over pairs.

    Args:
        pair_ids: Integer vocab ids with shape ``(n_pairs, 2, k)``.
        weight: Non-negative repulsion strength.

    Returns:
        A float32 array with shape ``(n_pairs, k, k)`` containing one repulsion
        weight table per leading pair.
    """
    return jax.vmap(lambda ids: inequality_weight_table(ids[0], ids[1], weight))(
        pair_ids
    ).astype(jnp.float32)
