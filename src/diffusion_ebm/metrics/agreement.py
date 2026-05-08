"""Metrics for the M3 experiment.

agreement_rate counts the fraction of chains whose hole values respect every
equality group.
lm_perplexity scores the filled-in sequence under MDLM.
Cost tracks LM forwards and Gibbs sweeps.
This lets the M4 Pareto plot be reconstructed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from diffusion_ebm.backbones.mdlm import MDLM, MASK_TOKEN_ID
from diffusion_ebm.tasks.multihole import MultiHoleInstance


@dataclass
class Cost:
    n_lm_forwards: int
    n_gibbs_sweeps: int = 0


def agreement_rate(
    samples: torch.Tensor,            # (n_chains, n_holes), torch.long
    equality_groups: list[list[int]],  # partition of n_holes positions
) -> float:
    # group_pos refers to the **hole index** (0..n_holes-1), NOT mask_positions in the original sequence.
    # Caller is responsible for either (a) passing groups in hole-index space, or (b) remapping.
    # For each chain: ALL groups must be internally consistent (every value in the group equal).
    # Return mean across chains (Python float).
    # Assert samples.dim() == 2.
    assert samples.dim() == 2

    ok = torch.ones(samples.shape[0], dtype=torch.bool, device=samples.device)
    for group in equality_groups:
        if len(group) <= 1:
            continue
        group_values = samples[:, group]
        ok &= (group_values == group_values[:, :1]).all(dim=1)
    return ok.float().mean().item()


@torch.no_grad()
def lm_perplexity(
    mdlm: MDLM,
    samples: torch.Tensor,             # (n_chains, n_holes), CPU long
    instance: MultiHoleInstance,
) -> torch.Tensor:                     # (n_chains,) float32 CPU
    # Build (n_chains, L) sequences by copying instance.masked_input_ids and writing each
    # row's sampled vocab IDs at instance.mask_positions.
    # Forward: logits = mdlm.forward(filled) -> (n_chains, L, V)
    # Cast logits to float32. Set logits[..., MASK_TOKEN_ID] = -inf BEFORE softmax.
    # log_probs = F.log_softmax(logits, dim=-1) -> (n_chains, L, V)
    # gather logp at the hole positions: logp[c, h] = log_probs[c, mask_positions[h], filled[c, mask_positions[h]]]
    # mean_neg_logp_per_chain = -logp.mean(dim=-1)            # (n_chains,) — average over the n_holes positions
    # ppl = exp(mean_neg_logp_per_chain)
    # Return ppl on CPU.
    n_chains = samples.shape[0]
    filled = instance.masked_input_ids.to(dtype=torch.long).unsqueeze(0).repeat(n_chains, 1)
    filled[:, instance.mask_positions] = samples.to(dtype=torch.long)

    logits = mdlm.forward(filled).float()
    logits[..., MASK_TOKEN_ID] = -float("inf")
    log_probs = F.log_softmax(logits, dim=-1)

    positions = torch.tensor(instance.mask_positions, device=log_probs.device)
    filled = filled.to(log_probs.device)
    targets = filled[:, positions].unsqueeze(-1)
    logp = log_probs[:, positions, :].gather(dim=-1, index=targets).squeeze(-1)
    mean_neg_logp_per_chain = -logp.mean(dim=-1)
    ppl = torch.exp(mean_neg_logp_per_chain)
    return ppl.cpu()
