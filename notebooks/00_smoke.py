"""Smoke test for diffusion-ebm environment.

Run with: ``uv run python notebooks/00_smoke.py``

Verifies imports + accelerator detection only. Actual sampling correctness is the
job of M1 (notebooks/01_mvp0_potts.py). Keeping this test minimal so failures
point at install/env issues unambiguously.

Checks:
1. Python / JAX / Torch / THRML imports.
2. JAX devices and Torch CUDA availability.
3. THRML symbol table looks right (sampling primitives importable).
4. Construct a `SpinNode` and `CategoricalNode` (no sampling).
5. GPT-2 tokenizer round-trip (used by MDLM at M2).

Exits 0 on success.
"""

import platform
import sys

print(f"Python:   {platform.python_version()}")
print(f"Platform: {platform.platform()}")

# %% Core libs
import jax
import jax.numpy as jnp
import numpy as np
import torch

print(f"JAX:      {jax.__version__}  devices={jax.devices()}")
print(f"NumPy:    {np.__version__}")
print(f"Torch:    {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"          cuda device 0: {torch.cuda.get_device_name(0)}")

# %% THRML imports — these are the symbols M1+ rely on.
from thrml import (  # noqa: E402
    Block,
    BlockGibbsSpec,
    BlockSamplingProgram,
    CategoricalNode,
    FactorSamplingProgram,
    SamplingSchedule,
    SoftmaxConditional,
    SpinNode,
    sample_states,
)
from thrml.models import (  # noqa: E402
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    SpinEBMFactor,
    SpinGibbsConditional,
)

print(
    f"THRML:    OK  (CategoricalNode, SpinNode, BlockGibbsSpec, "
    f"FactorSamplingProgram, sample_states, "
    f"SpinEBMFactor, CategoricalEBMFactor, "
    f"SpinGibbsConditional, CategoricalGibbsConditional, "
    f"SoftmaxConditional, BlockSamplingProgram, SamplingSchedule, Block — all importable)"
)

# %% Construct nodes (no sampling — just verify object creation)
spin_nodes = [SpinNode() for _ in range(4)]
cat_nodes = [CategoricalNode() for _ in range(3)]

even = Block([spin_nodes[0], spin_nodes[2]])
odd = Block([spin_nodes[1], spin_nodes[3]])
print(f"Built Blocks: even={len(even.nodes)}  odd={len(odd.nodes)}")

# A categorical-5 sampler (cardinality is set on the sampler, not the node).
_cat_sampler = CategoricalGibbsConditional(n_categories=5)
print(f"Built CategoricalGibbsConditional(n_categories=5)")

# %% HuggingFace tokenizer round-trip (validates network / cache works)
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("gpt2")
text = "The quick brown fox jumps over the lazy dog."
ids = tok.encode(text)
back = tok.decode(ids)
print(f"\nTokenizer: gpt2  vocab_size={tok.vocab_size}")
print(f"  encode: {text!r} -> {ids[:10]}{'...' if len(ids) > 10 else ''}")
print(f"  decode: {back!r}")
assert back.strip() == text.strip(), "Tokenizer round-trip mismatch."

# %% Done
print("\nSmoke test passed.")
sys.exit(0)
