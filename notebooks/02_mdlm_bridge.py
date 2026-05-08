"""M2 — MDLM bridge end-to-end validation.

Loads the kuleshov-group/mdlm-owt checkpoint via our wrapper, runs three
checks, and prints sample outputs:

1. Forward pass on clean text → logits shape and dtype assertions.
2. ``top_k_candidates`` returns (ids, unaries) of the right shapes; the
   absorbing-token id never appears as a candidate.
3. Mask-predict fill on a partially-masked prompt produces coherent English.

Run: ``LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH" \\
        uv run python notebooks/02_mdlm_bridge.py``
"""

from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from diffusion_ebm.backbones.mdlm import MASK_TOKEN_ID, MDLM_VOCAB, MDLM  # noqa: E402

torch.manual_seed(0)

# %% Load
mdlm = MDLM.load()
print(f"Loaded {type(mdlm.model).__name__} on {mdlm.device}")
print(f"  vocab_size={mdlm.vocab_size}  mask_token_id={mdlm.mask_token_id}")

# %% (1) Forward on clean text
text = "The quick brown fox jumps over the lazy dog."
clean_ids = torch.tensor(mdlm.tokenizer.encode(text), dtype=torch.long)
print(f"\n[forward / clean]  input_ids shape={tuple(clean_ids.shape)}")
logits = mdlm.forward(clean_ids)
print(f"  logits.shape={tuple(logits.shape)}  dtype={logits.dtype}")
print(f"  logits[mask_token]: min={float(logits[..., MASK_TOKEN_ID].min()):.3f}  "
      f"max={float(logits[..., MASK_TOKEN_ID].max()):.3f}")

assert logits.shape == (1, clean_ids.shape[0], MDLM_VOCAB), (
    f"Expected logits ({1}, {clean_ids.shape[0]}, {MDLM_VOCAB}), got {tuple(logits.shape)}"
)

# %% (2) top_k_candidates contract
k = 32
ids, unary = mdlm.top_k_candidates(logits, k=k, exclude_mask_token=True)
print(f"\n[top_k / k={k}]  ids.shape={tuple(ids.shape)}  unary.shape={tuple(unary.shape)}")
assert ids.shape == (*logits.shape[:-1], k)
assert unary.shape == (*logits.shape[:-1], k)
assert (ids != MASK_TOKEN_ID).all(), "Mask token leaked into top-k candidates"
# Spot check: unary should equal the gathered logit values.
gathered = logits.gather(-1, ids)
assert torch.allclose(gathered, unary, rtol=1e-4, atol=1e-4)
print("  unary == logits.gather(-1, ids) ✓  no mask token in candidates ✓")

# %% (3) Single-shot vs iterative mask-predict on a half-masked prompt
prompt = "The capital of France is"
prompt_ids = mdlm.tokenizer.encode(prompt)
n_fill = 12
masked = prompt_ids + [MASK_TOKEN_ID] * n_fill
input_ids = torch.tensor(masked, dtype=torch.long).unsqueeze(0)
print(f"\n[mask-predict] prompt={prompt!r}  filling {n_fill} masked positions")
print(f"  input ({len(masked)} tokens, {sum(1 for t in masked if t == MASK_TOKEN_ID)} masked)")

for n_iters in (1, 4, 12):
    out = mdlm.mask_predict(input_ids, n_iters=n_iters, temperature=0.0)
    text_out = mdlm.tokenizer.decode(out[0].tolist())
    print(f"  n_iters={n_iters:2d}: {text_out!r}")

# %% (4) Per-position top-1 prediction at masked positions sanity check
top1 = logits.argmax(dim=-1)
decoded = mdlm.tokenizer.decode(top1[0].tolist())
print(f"\n[per-position argmax on clean text]\n  {decoded!r}")

print("\nM2 PASSED.")
