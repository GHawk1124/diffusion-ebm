"""M1 / MVP0 — ferromagnetic Potts chain sanity check.

Validates THRML's block-Gibbs sampler on a problem with a known target
distribution.  A length-N chain of K-state categorical variables with pairwise
ferromagnetic factors has uniform per-position marginals but a Boltzmann
distribution that concentrates on the K all-equal configurations.

Run: ``uv run python notebooks/01_mvp0_potts.py``
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

# Ensure the in-repo package is importable when running this script directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from diffusion_ebm.synth.potts_chain import (  # noqa: E402
    aligned_fraction,
    independent_samples,
    make_chain,
    thrml_samples,
)


N = 20
K = 5
J = 5.0
N_SAMPLES = 500
WARMUP = 500


# %% Build the chain
chain = make_chain(n=N, k=K, J=J)
print(f"Chain: N={N}, K={K}, J={J}")
print(f"  free blocks: even={len(chain.even.nodes)}, odd={len(chain.odd.nodes)}")

key = jax.random.PRNGKey(0)

# %% Independent baseline (uniform per-position marginal)
key, sub = jax.random.split(key)
indep = independent_samples(sub, n=N, k=K, n_samples=N_SAMPLES)
indep_align = aligned_fraction(indep)
indep_full = float((indep_align == 1.0).mean())
print(f"\n[Independent]")
print(f"  mean alignment fraction : {float(indep_align.mean()):.4f}  "
      f"(expect ~{1.0 / K:.4f})")
print(f"  fully-aligned rate      : {indep_full:.6f}  "
      f"(theoretical {K * K**-(N - 1):.2e})")

# %% THRML block-Gibbs
key, sub = jax.random.split(key)
samples = thrml_samples(sub, chain, n_samples=N_SAMPLES, warmup=WARMUP)
thrml_align = aligned_fraction(samples)
thrml_full = float((thrml_align == 1.0).mean())
print(f"\n[THRML]")
print(f"  mean alignment fraction : {float(thrml_align.mean()):.4f}")
print(f"  fully-aligned rate      : {thrml_full:.4f}")

# %% Mode distribution among fully-aligned THRML samples
aligned = samples[thrml_align == 1.0]
n_aligned = int(aligned.shape[0])
print(f"\nMode distribution among {n_aligned} aligned THRML samples:")
if n_aligned:
    modes = aligned[:, 0]
    width = 40
    for c in range(K):
        n_c = int((modes == c).sum())
        bar = "#" * (width * n_c // max(1, n_aligned))
        print(f"  mode {c}: {n_c:4d}  {bar}")
else:
    print("  (none)")

# %% Plot
os.makedirs("plots", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
ax[0].hist(np.asarray(indep_align), bins=21, range=(0, 1), color="C0")
ax[0].set_title(f"Independent  (full={indep_full:.3f})")
ax[0].set_xlabel("alignment fraction")
ax[0].set_ylabel("count")
ax[1].hist(np.asarray(thrml_align), bins=21, range=(0, 1), color="C1")
ax[1].set_title(f"THRML joint Gibbs  (full={thrml_full:.3f})")
ax[1].set_xlabel("alignment fraction")
fig.suptitle(f"MVP0 — ferromagnetic Potts chain  (N={N}, K={K}, J={J})")
plt.tight_layout()
out_path = os.path.join("plots", "mvp0_potts_alignment.png")
fig.savefig(out_path, dpi=120)
plt.close(fig)
print(f"\nSaved {out_path}")

# %% Sanity assertions
assert indep_full < 1e-3, (
    f"Independent fully-aligned rate {indep_full:.4f} unexpectedly high — bug?"
)
assert thrml_full > 0.5, (
    f"THRML fully-aligned rate {thrml_full:.4f} too low — increase WARMUP "
    "or sanity-check the factor wiring."
)
print("\nMVP0 PASSED.")
