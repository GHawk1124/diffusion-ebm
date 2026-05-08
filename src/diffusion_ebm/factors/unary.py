"""Unary factor weight helpers for THRML interop.

thrml.CategoricalEBMFactor with a single Block accepts weights shaped
``[n_holes, k]`` per discrete_ebm.py's invariant
``len(weights.shape) == 1 + len(node_groups)``. This module casts LM-derived
per-position top-k logits into that canonical shape and dtype.
"""

import jax.numpy as jnp


def unary_weight_tensor(unary: jnp.ndarray) -> jnp.ndarray:
    if unary.ndim != 2:
        raise ValueError(f"unary must have shape [n_holes, k], got {unary.shape}")
    return unary.astype(jnp.float32)
