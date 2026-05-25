"""Baselines for the M3 multi-hole-agreement experiment.

Each baseline produces an ``(n_chains, n_holes)`` vocab-ID tensor at the hole
positions of a ``MultiHoleInstance``. ``ancestral_topk`` uses one MDLM forward
pass and samples from the top-k softmax. ``ancestral_topk_iterative`` is the
M5a strengthened baseline: it forwards MDLM ``n_iters`` times, each time
committing the most-confident remaining hole(s) per chain. ``mask_predict``
wraps ``MDLM.mask_predict``. ``independent_full`` samples the unrestricted
softmax. None of them ever return ``MASK_TOKEN_ID`` (50257).
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
def ancestral_topk_iterative(
    mdlm: MDLM,
    instance: MultiHoleInstance,
    k: int = 64,
    n_iters: int = 1,
    n_chains: int = 64,
    seed: int | None = None,
) -> torch.Tensor:
    """Iterative ancestral top-k fill (M5a baseline).

    At each iteration: forward MDLM on the current per-chain sequence, mask-token
    guard, sample a candidate at every still-masked hole from its top-k softmax,
    and commit the ``ceil(remaining / iters_left)`` most-confident holes per
    chain. ``n_iters=1`` is equivalent to ``ancestral_topk`` (all holes
    committed in one shot, distributionally identical up to RNG ordering).
    ``n_iters >= n_holes`` commits one hole per step.
    """
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1, got {n_iters}")

    hole_positions = instance.mask_positions
    n_holes = len(hole_positions)

    gen = torch.Generator(device=mdlm.device)
    if seed is not None:
        gen.manual_seed(seed)

    holes_idx = torch.tensor(hole_positions, device=mdlm.device, dtype=torch.long)

    out_chunks: list[torch.Tensor] = []
    for s in range(0, n_chains, _CHUNK):
        sub_n = min(_CHUNK, n_chains - s)
        ids = (
            instance.masked_input_ids
            .to(mdlm.device)
            .unsqueeze(0)
            .repeat(sub_n, 1)
        )
        for it in range(n_iters):
            is_masked = ids[:, holes_idx] == MASK_TOKEN_ID
            remaining = is_masked.sum(dim=-1)
            if int(remaining.max().item()) == 0:
                break

            logits = mdlm.forward(ids).float()
            logits[..., MASK_TOKEN_ID] = float("-inf")
            logits_holes = logits[:, holes_idx, :]
            topk_logits, topk_ids = torch.topk(logits_holes, k, dim=-1)
            topk_probs = F.softmax(topk_logits, dim=-1)
            conf = topk_probs[..., 0].masked_fill(~is_masked, float("-inf"))

            flat_probs = topk_probs.reshape(-1, k)
            sampled_idx = torch.multinomial(
                flat_probs, 1, generator=gen
            ).view(sub_n, n_holes)
            sampled_ids = topk_ids.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)

            iters_left = n_iters - it
            for b in range(sub_n):
                r = int(remaining[b].item())
                if r == 0:
                    continue
                keep = max(1, -(-r // iters_left))
                keep = min(keep, r)
                order = torch.topk(conf[b], keep).indices
                ids[b, holes_idx[order]] = sampled_ids[b, order]
            del logits, logits_holes, topk_logits, topk_probs

        out_chunks.append(ids[:, holes_idx].to(device="cpu", dtype=torch.long))
    out = torch.cat(out_chunks, dim=0)
    assert (out != MASK_TOKEN_ID).all()
    return out


_CHUNK = 16


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

    # Chunk the batch so the [B, L, V] float-cast inside MDLM.mask_predict
    # does not OOM on long M5b templates. Chains are independent here.
    out_chunks: list[torch.Tensor] = []
    for s in range(0, n_chains, _CHUNK):
        sub = batch[s : s + _CHUNK]
        filled = mdlm.mask_predict(
            sub, n_iters=n_iters, temperature=temperature, rng=gen,
        )
        out_chunks.append(
            filled[:, instance.mask_positions].to(device="cpu", dtype=torch.long)
        )
    out = torch.cat(out_chunks, dim=0)
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
