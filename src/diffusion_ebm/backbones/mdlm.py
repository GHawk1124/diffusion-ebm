"""HF wrapper around the MDLM masked-diffusion LM (kuleshov-group/mdlm-owt).

The MDLM checkpoint requires ``trust_remote_code=True`` and depends on
flash_attn (built into the env via the project's flake / pyproject).  This
module exposes:

* ``MDLM.load(...)`` — load weights + tokenizer.
* ``MDLM.forward(input_ids)`` — `[B, L]` long tensor → `[B, L, V]` logits.
* ``MDLM.top_k_candidates(logits, k)`` → ids and unary energies for THRML.
* ``MDLM.mask_predict(masked_ids, n_iters)`` — Mask-Predict-style iterative
  fill, used as the M3 baseline; ``n_iters=1`` reduces to a single-shot
  independent ancestral fill.

Vocabulary / mask-token convention:
GPT-2 has 50,257 tokens.  MDLM extends this with one absorbing/[MASK] state,
giving vocab_size = 50,258 and ``MASK_TOKEN_ID = 50_257``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

DEFAULT_MODEL = "kuleshov-group/mdlm-owt"
GPT2_VOCAB = 50_257
MASK_TOKEN_ID = GPT2_VOCAB  # 50_257
MDLM_VOCAB = 50_258         # 50_257 + 1


@dataclass
class MDLM:
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    vocab_size: int = MDLM_VOCAB
    mask_token_id: int = MASK_TOKEN_ID

    @classmethod
    def load(cls, name: str = DEFAULT_MODEL, device: Optional[str] = None) -> "MDLM":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = (
            AutoModelForMaskedLM
            .from_pretrained(name, trust_remote_code=True, dtype="auto")
            .to(device)
            .eval()
        )
        return cls(model=model, tokenizer=tokenizer, device=torch.device(device))

    @torch.no_grad()
    def forward_hidden(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One MDLM forward returning (logits, last_hidden_state).

        Used by the M5c learned EBM factor: the pairwise scorer ψ reads the
        last hidden state at the masked positions to produce token-pair
        scores. Same sigma convention as ``forward``.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        mask_ratio = (input_ids == self.mask_token_id).float().mean(dim=-1)
        sigma = mask_ratio.clamp(min=1e-3, max=1.0 - 1e-3)
        out = self.model(
            input_ids=input_ids,
            timesteps=sigma,
            return_dict=True,
            output_hidden_states=True,
        )
        # MDLM hidden_states is [embed, *per-block]; last is pre-output_layer.
        last_hidden = out.hidden_states[-1]
        return out.logits, last_hidden

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """One MDLM forward pass.

        ``input_ids``: ``[B, L]`` (or ``[L]``) long.  Mask tokens use
        ``MASK_TOKEN_ID``.  Returns logits ``[B, L, V]`` where ``V == 50258``.

        Sigma is set from the per-row mask ratio: 0 for clean text, 1 for fully
        masked.  When ``config.time_conditioning`` is False, MDLM zeros sigma
        internally and ignores it; when True, this is a sane default.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        mask_ratio = (input_ids == self.mask_token_id).float().mean(dim=-1)
        sigma = mask_ratio.clamp(min=1e-3, max=1.0 - 1e-3)
        # MDLM defaults to `return_dict=False` and yields a raw logits tensor;
        # force the dict form so we get a stable `.logits` accessor here.
        out = self.model(input_ids=input_ids, timesteps=sigma, return_dict=True)
        return out.logits

    def top_k_candidates(
        self,
        logits: torch.Tensor,
        k: int = 64,
        exclude_mask_token: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-position top-k extraction.

        Returns ``(ids[B, L, k], unary[B, L, k])`` where
        ``unary[b, l, c] == logits[b, l, ids[b, l, c]]`` — i.e., raw logits at
        the candidate ids, suitable for use as THRML unary energies (with the
        sign flipped, since THRML factor weights ``W`` give energy ``-W``).
        """
        if exclude_mask_token:
            logits = logits.clone()
            logits[..., self.mask_token_id] = -float("inf")
        unary, ids = logits.topk(k, dim=-1)
        return ids, unary

    @torch.no_grad()
    def mask_predict(
        self,
        masked_ids: torch.Tensor,
        n_iters: int = 8,
        temperature: float = 0.0,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Iterative mask-predict fill (Ghazvininejad et al., 2019).

        At each iteration: forward the current sequence, sample a candidate
        token at every masked position (greedy at temperature=0, multinomial
        otherwise), keep the top-confidence subset (``count // remaining_iters``
        per row), and re-mask the rest.

        ``n_iters=1`` is single-shot independent fill (the simplest 'ancestral'
        baseline).  Larger ``n_iters`` gives the model multiple chances to
        condition on its own outputs.
        """
        ids = masked_ids.clone().to(self.device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, L = ids.shape

        for it in range(n_iters):
            mask = ids == self.mask_token_id
            n_masked = int(mask.sum().item())
            if n_masked == 0:
                break
            logits = self.forward(ids)
            logits[..., self.mask_token_id] = -float("inf")

            if temperature == 0.0:
                sampled = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                sampled = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), 1, generator=rng,
                ).view(B, L)

            # Confidence: the softmax probability of the sampled token.
            conf = F.softmax(logits, dim=-1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            conf = conf.masked_fill(~mask, -float("inf"))

            iters_left = n_iters - it
            for b in range(B):
                row_mask = mask[b]
                row_n = int(row_mask.sum().item())
                if row_n == 0:
                    continue
                keep = max(1, row_n // iters_left)
                order = torch.argsort(conf[b], descending=True)[:keep]
                ids[b, order] = sampled[b, order]

        # Failsafe: greedy fill of any remaining mask tokens (shouldn't happen
        # with sensible ``n_iters``).
        remaining = ids == self.mask_token_id
        if remaining.any():
            logits = self.forward(ids)
            logits[..., self.mask_token_id] = -float("inf")
            ids = torch.where(remaining, logits.argmax(dim=-1), ids)
        return ids
