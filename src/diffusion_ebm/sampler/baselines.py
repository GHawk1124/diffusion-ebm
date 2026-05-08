"""Baselines for the M3 multi-hole-agreement experiment.

Each baseline produces an ``(n_chains, n_holes)`` vocab-ID tensor at the hole
positions of a ``MultiHoleInstance``. ``ancestral_topk`` uses one MDLM forward
pass and samples from the top-k softmax. ``mask_predict`` wraps
``MDLM.mask_predict``. ``independent_full`` samples the unrestricted softmax.
None of them ever return ``MASK_TOKEN_ID`` (50257).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusion_ebm.backbones.mdlm import MDLM, MASK_TOKEN_ID
from diffusion_ebm.tasks.multihole import MultiHoleInstance


@torch.no_grad()
def ancestral_topk(
    mdlm: MDLM,
    instance: MultiHoleInstance,
    k: int = 64,
    n_chains: int = 64,
    seed: int | None = None,
) -> torch.Tensor:
    logits = mdlm.forward(instance.masked_input_ids).float()
    logits = logits.clone()
    logits[..., MASK_TOKEN_ID] = float("-inf")
    logits_holes = logits[0, instance.mask_positions, :]

    topk_logits, ids = torch.topk(logits_holes, k, dim=-1)
    probs = F.softmax(topk_logits, dim=-1)

    gen = torch.Generator(device=probs.device)
    if seed is not None:
        gen.manual_seed(seed)

    sampled = torch.multinomial(probs, n_chains, replacement=True, generator=gen)
    out = ids.gather(1, sampled).transpose(0, 1).to(device="cpu", dtype=torch.long)
    assert (out != MASK_TOKEN_ID).all()
    return out


@torch.no_grad()
def mask_predict(
    mdlm: MDLM,
    instance: MultiHoleInstance,
    n_iters: int = 8,
    n_chains: int = 64,
    temperature: float = 1.0,
    seed: int | None = None,
) -> torch.Tensor:
    batch = instance.masked_input_ids.unsqueeze(0).repeat(n_chains, 1)

    gen = None
    if temperature != 0.0 or seed is not None:
        gen = torch.Generator(device=mdlm.device)
        if seed is not None:
            gen.manual_seed(seed)

    filled = mdlm.mask_predict(
        batch,
        n_iters=n_iters,
        temperature=temperature,
        rng=gen,
    )
    out = filled[:, instance.mask_positions].to(device="cpu", dtype=torch.long)
    assert (out != MASK_TOKEN_ID).all()
    return out


@torch.no_grad()
def independent_full(
    mdlm: MDLM,
    instance: MultiHoleInstance,
    n_chains: int = 64,
    seed: int | None = None,
) -> torch.Tensor:
    logits = mdlm.forward(instance.masked_input_ids).float()
    logits = logits.clone()
    logits[..., MASK_TOKEN_ID] = float("-inf")
    logits_holes = logits[0, instance.mask_positions, :]

    probs = F.softmax(logits_holes, dim=-1).cpu()

    gen = torch.Generator(device=probs.device)
    if seed is not None:
        gen.manual_seed(seed)

    out = torch.multinomial(probs, n_chains, replacement=True, generator=gen)
    out = out.transpose(0, 1).to(device="cpu", dtype=torch.long)
    assert (out != MASK_TOKEN_ID).all()
    return out
