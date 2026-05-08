"""Probe the MDLM HuggingFace checkpoint to learn its public API.

Goals:
1. Confirm `AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)` loads.
2. Inspect config attributes (vocab_size, mask_token_id, special tokens).
3. Run one forward and inspect the output type and tensor shape.
4. Find the mask token id (could be on tokenizer, model config, or hardcoded).

Run: ``LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH" uv run python notebooks/02a_mdlm_probe.py``

Artefacts: prints discovered API to stdout, no files written.
"""

from __future__ import annotations

import os
import pprint

import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "kuleshov-group/mdlm-owt"

print("=" * 70)
print(f"Probing {MODEL_NAME}")
print("=" * 70)

# %% Tokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
print(f"\nTokenizer: gpt2  vocab_size={tok.vocab_size}  "
      f"mask_token={tok.mask_token!r}  pad_token={tok.pad_token!r}")
print(f"  special tokens: {tok.all_special_tokens}")
print(f"  bos={tok.bos_token_id}  eos={tok.eos_token_id}  unk={tok.unk_token_id}")

# %% Config — see if MDLM extends GPT-2 vocab with a mask token
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"\nConfig type: {type(config).__name__}")
print(f"  vocab_size: {getattr(config, 'vocab_size', '?')}")
print(f"  hidden_size / d_model: {getattr(config, 'hidden_size', getattr(config, 'd_model', '?'))}")
print(f"  architectures: {getattr(config, 'architectures', '?')}")
# Look for any mask-related attribute
mask_attrs = {
    k: v
    for k, v in config.to_dict().items()
    if "mask" in k.lower() or "absorb" in k.lower()
}
print(f"  mask/absorb-related attrs: {mask_attrs}")

# %% Load model (~500MB download on first run)
print("\nLoading model (this may download ~500MB on first run)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForMaskedLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, dtype="auto"
).to(device).eval()
print(f"Model class: {type(model).__name__}  on {device}")
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Total parameters: {n_params:.1f}M")

# Show top-level submodules to understand architecture.
print("\nTop-level submodules:")
for name, _ in model.named_children():
    print(f"  {name}")

# %% Forward pass on a small input
text = "The quick brown fox jumps over the lazy dog."
ids = tok.encode(text, return_tensors="pt").to(device)
print(f"\nInput ids shape: {tuple(ids.shape)}  vocab_max={int(ids.max())}")

# MDLM.forward requires `timesteps` (sigma).  For a clean (unmasked) input we
# pass a tiny epsilon; for the masked probe below we use ~0.5 to mimic a
# moderately-noised diffusion step.
sigma_clean = torch.full((ids.shape[0],), 1e-3, device=device)
with torch.no_grad():
    out = model(input_ids=ids, timesteps=sigma_clean)

print(f"Output type: {type(out).__name__}")
if hasattr(out, "logits"):
    logits = out.logits
elif isinstance(out, torch.Tensor):
    logits = out
else:
    print(f"Unexpected output: {out!r}")
    logits = None

if logits is not None:
    print(f"Logits shape: {tuple(logits.shape)}  dtype={logits.dtype}")
    print(f"Logits min/max: {float(logits.min()):.3f} / {float(logits.max()):.3f}")
    # Check whether the vocab dim matches GPT-2 (50257) or is extended (e.g. +1 for mask).
    V = logits.shape[-1]
    print(f"  vocab dim: {V}  (gpt2 vocab=50257, MDLM may add 1 for absorbing token)")
    if V > tok.vocab_size:
        print(f"  ➜ likely mask/absorb token id = {V - 1} (one past gpt2 vocab)")

# %% Try a partially-masked forward.  MDLM standard practice: mask token = vocab_size
mask_id_candidate = config.vocab_size - 1 if hasattr(config, "vocab_size") else 50257
print(f"\nProbing mask behaviour with candidate mask_id = {mask_id_candidate} ...")
masked = ids.clone()
# Mask the second word.
masked[0, 2] = mask_id_candidate
sigma_masked = torch.full((masked.shape[0],), 0.5, device=device)
with torch.no_grad():
    out2 = model(input_ids=masked, timesteps=sigma_masked)
logits2 = out2.logits if hasattr(out2, "logits") else out2
top5 = logits2[0, 2].topk(5)
print("Top-5 predictions at masked position 2:")
for v, i in zip(top5.values.tolist(), top5.indices.tolist()):
    tok_str = tok.decode([i])
    print(f"  {i:6d}  logit={v:7.3f}  {tok_str!r}")

print("\nProbe complete.")
