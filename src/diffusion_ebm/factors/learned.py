"""Learned pairwise factor for the M5c EBM correction.

Given MDLM's last hidden state at two masked positions and a candidate
top-k token set per position, the ``PairwiseScorer`` produces a scalar
``ψ(x_i, x_j | context)`` that the THRML factor graph consumes as the
weight on a categorical edge between two ``CategoricalNode`` blocks.

The training objective is **Joint-Transition NCE** (M5c.1): positives are
true ``(x_i, x_j)`` from the corpus; negatives are pairs where each
token is independently high-probability under ``p_LM`` at its masked
position. The scorer learns to distinguish corpus pairs from
LM-marginal-resampled pairs — i.e., joint structure beyond the
per-position conditional.

Architectural notes:
* The scorer is a **bilinear factorisation**:
  ``ψ(x_i, x_j) = ⟨f(h_a, x_i), g(h_b, x_j)⟩``.
  At inference time, evaluating the full ``[k, k]`` pair table reduces
  to two ``[k, head_dim]`` MLP forwards plus a ``[k, k]`` matmul — the
  cheapest expressive form for THRML's per-edge weight tensor.
* Token embeddings are tied between the ``f`` and ``g`` heads so a token
  has a single learnable representation regardless of which side of the
  pair it appears on.
* ``forward(...)`` returns scores for one ``(x_i, x_j)`` per row — the
  training-loop call site. ``pair_table(...)`` returns the full
  ``[k_a, k_b]`` matrix used to populate one THRML edge weight.
"""

from __future__ import annotations

import jax.numpy as jnp
import torch
import torch.nn as nn


class PairwiseScorer(nn.Module):
    """Bilinear pairwise scorer for the M5c learned factor."""

    def __init__(
        self,
        hidden_dim: int = 768,
        embed_dim: int = 128,
        head_dim: int = 64,
        mlp_dim: int = 256,
        vocab_size: int = 50_258,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.f_head = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, head_dim),
        )
        self.g_head = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, head_dim),
        )

    def _f(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.f_head(torch.cat([h, self.token_embed(x)], dim=-1))

    def _g(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.g_head(torch.cat([h, self.token_embed(x)], dim=-1))

    def forward(
        self,
        h_a: torch.Tensor,  # [B, d_h]
        h_b: torch.Tensor,  # [B, d_h]
        x_i: torch.Tensor,  # [B] long
        x_j: torch.Tensor,  # [B] long
    ) -> torch.Tensor:
        """Return ψ(x_i, x_j | h_a, h_b), shape [B]."""
        scale = self.head_dim ** 0.5
        return (self._f(h_a, x_i) * self._g(h_b, x_j)).sum(dim=-1) / scale

    def forward_pos_neg(
        self,
        h_a: torch.Tensor,        # [B, d_h]
        h_b: torch.Tensor,        # [B, d_h]
        x_i_pos: torch.Tensor,    # [B] long
        x_j_pos: torch.Tensor,    # [B] long
        x_i_neg: torch.Tensor,    # [B, K] long
        x_j_neg: torch.Tensor,    # [B, K] long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pos_scores [B], neg_scores [B, K])."""
        scale = self.head_dim ** 0.5
        feat_a_pos = self._f(h_a, x_i_pos)
        feat_b_pos = self._g(h_b, x_j_pos)
        pos = (feat_a_pos * feat_b_pos).sum(dim=-1) / scale

        B, K = x_i_neg.shape
        h_a_exp = h_a.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        h_b_exp = h_b.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        feat_a_neg = self._f(h_a_exp, x_i_neg.reshape(-1)).view(B, K, -1)
        feat_b_neg = self._g(h_b_exp, x_j_neg.reshape(-1)).view(B, K, -1)
        neg = (feat_a_neg * feat_b_neg).sum(dim=-1) / scale
        return pos, neg

    @torch.no_grad()
    def pair_table(
        self,
        h_a: torch.Tensor,    # [d_h]
        h_b: torch.Tensor,    # [d_h]
        ids_a: torch.Tensor,  # [k_a] long
        ids_b: torch.Tensor,  # [k_b] long
    ) -> torch.Tensor:
        """Full ψ table over candidate pairs, shape [k_a, k_b].

        Used at THRML build time to populate one factor's weight tensor.
        """
        k_a = ids_a.shape[0]
        k_b = ids_b.shape[0]
        ha_exp = h_a.unsqueeze(0).expand(k_a, -1)
        hb_exp = h_b.unsqueeze(0).expand(k_b, -1)
        fa = self._f(ha_exp, ids_a)  # [k_a, head_dim]
        gb = self._g(hb_exp, ids_b)  # [k_b, head_dim]
        return fa @ gb.T / (self.head_dim ** 0.5)  # [k_a, k_b]


def stack_learned_factor(
    scorer: PairwiseScorer,
    hidden: torch.Tensor,           # [L, d_h] — last hidden state from MDLM
    candidate_ids: torch.Tensor,    # [n_holes, k] long, vocab IDs
    hole_positions: list[int],      # length n_holes
    pair_indices: list[tuple[int, int]],  # edge list in hole-index space
    weight_scale: float = 1.0,
) -> jnp.ndarray:
    """Build the (n_pairs, k, k) weight tensor for THRML.

    ``hidden`` is shaped [L, d_h] (single instance, not batched). For
    each (a, b) edge we evaluate the full ``[k, k]`` ``ψ`` table at the
    matched hole positions, multiply by ``weight_scale``, and stack.
    Caller passes the result into ``CategoricalEBMFactor``.
    """
    if hidden.dim() != 2:
        raise ValueError(f"hidden must be [L, d_h]; got {hidden.shape}")
    tables: list[torch.Tensor] = []
    for a, b in pair_indices:
        pos_a = hole_positions[a]
        pos_b = hole_positions[b]
        h_a = hidden[pos_a]
        h_b = hidden[pos_b]
        ids_a = candidate_ids[a]
        ids_b = candidate_ids[b]
        tables.append(
            scorer.pair_table(h_a, h_b, ids_a, ids_b) * weight_scale
        )
    stacked = torch.stack(tables, dim=0).float().cpu().numpy()
    return jnp.asarray(stacked, dtype=jnp.float32)
