# diffusion-ebm — project notes for Claude

This file is the canonical project context.  Read it first; treat it as the
source of truth for the research question, the design decisions we've
already debated, the env quirks specific to this NixOS host, and how far
along the milestones we are.

## What this project is

A research prototype that hybridises a **pretrained masked-diffusion language
model** with a **THRML block-Gibbs joint sampler** (the GPU/JAX simulator for
Extropic's TSU hardware).  The headline question:

> Given a masked diffusion LM that produces per-position categorical logits,
> can a THRML block-Gibbs sampler over a sparse factor graph improve
> **multi-hole consistency** at fixed neural-FLOPs budget?

The full plan is at `/home/ghawk/.claude/plans/nested-wishing-thimble.md`
(approved 2026-05-06).  Read it for milestone-level detail; the highlights
are below.

### The honest framing (the one load-bearing insight)

With energy `E = -Σ log p_LM(x_i | context)` and **no other terms**, joint
Gibbs samples from the same distribution as independent ancestral sampling.
Joint sampling only beats independent when ψ encodes information that's not
in `p_LM(x_i | context)`.  Sources of such signal:

1. **Hard combinatorial constraints** (regex, JSON, brackets) — clear win.
2. **A separately-trained sequence-level scorer** (out of M4 scope).
3. **Cross-position semantic agreement** (coreference, variable names,
   arithmetic consistency) — this is the M3 target.

What is *not* a source of joint signal:
- **Bigram / trigram corpus statistics** — already in the LM conditional.
- **Constrained decoding for hard syntax** — already solved by Outlines /
  xgrammar at near-zero cost.

### The MVP path (decided after two rounds of critique)

1. **MVP0 — synthetic Potts chain.**  THRML correctness sanity check.
2. **MVP1 — multi-hole agreement on MDLM.**  Top-k=64 candidate sets per
   masked position; unary = LM logit; hard equality factor between matched
   holes; baselines = ancestral, mask-predict, ancestral-top-k.
3. **M4 — Pareto plot.**  Agreement rate vs total LM forward passes, with
   Gibbs-sweep cost reported separately so the TSU cost model
   (`K · C_N > R · C_G`) can be applied post-hoc.
4. (Stretch, out of current scope) learned EBM corrections.

### Things we explicitly ruled out

- Training an LM from scratch (relies on a pretrained MDLM).
- **Binary spin codes for token IDs** — Hamming geometry is noise w.r.t.
  semantics; no useful inductive bias for local Gibbs moves.
- **MD4 backbone** — JAX-native but no public checkpoint; would be a
  multi-week training detour.
- **SEDD backbone** — has checkpoints but score-entropy parameterisation
  needs non-trivial conversion to per-position categorical.
- **Bit-flip noise as a forward process** — incoherent w.r.t. text.

## Stack

- **Python 3.12** (don't use 3.14 — torch wheels don't exist for it yet).
- **uv** for env management, **nix** for system deps (NixOS host).
- **THRML 0.1.3** (Oct 2025) — `CategoricalNode`/`SpinNode`,
  `CategoricalEBMFactor` / `SpinEBMFactor`, `BlockGibbsSpec`,
  `FactorSamplingProgram`, `sample_states`, `CategoricalGibbsConditional`.
- **JAX 0.5.3** with CUDA 12, `jax[cuda12]`.
- **PyTorch 2.6.0+cu124** for the MDLM backbone.  Pinned via
  `[tool.uv.sources] torch = { index = "pytorch-cu124" }` because PyPI's
  default ships cu130 wheels which need NVIDIA driver ≥580.
- **flash-attn** (built from source via `no-build-isolation`).
- **transformers**, **einops**, **networkx**, **matplotlib**, etc.

## Repo layout

```
diffusion-ebm/
  CLAUDE.md                    # this file
  flake.nix                    # FHS shell with nvcc + LD path workarounds
  .envrc                       # `use flake` (for nix-direnv users)
  .python-version              # 3.12
  pyproject.toml               # deps, torch+cu124 pin, no-build-isolation
  main.py                      # placeholder (delete before M5)

  src/diffusion_ebm/
    __init__.py
    backbones/
      __init__.py
      mdlm.py                  # M2: MDLM HF wrapper
    sampler/                   # M3 (empty)
    factors/                   # M3 (empty)
    tasks/                     # M3 (empty)
    metrics/                   # M3 (empty)
    synth/
      potts_chain.py           # M1: ferromagnetic Potts chain
    utils/                     # (empty)

  notebooks/                   # all are runnable .py with jupytext-style cells
    00_smoke.py                # M0: env check
    01_mvp0_potts.py           # M1: Potts sanity check
    02a_mdlm_probe.py          # M2: probe MDLM API surface
    02_mdlm_bridge.py          # M2: end-to-end MDLM forward + mask-predict
    03_mvp1_multihole.py       # M3: multi-hole agreement headline
    04_pareto.py               # M4: Pareto plots + summary table

  experiments/
    run_pareto.py              # M4: sweep driver → results/results.json

  plots/                       # gitignored; mvp0_potts_alignment.png lives here
  results/                     # gitignored
```

## Status (2026-05-08, M5a + M5b complete; M5c prepped, awaits A100 day)

- [x] **M0** — project skeleton, env install, smoke test (PASS).
- [x] **M1** — Potts chain MVP0 (PASS: independent 0/500 fully aligned;
      THRML 377/500 fully aligned, 98.4 % mean per-edge alignment; chain
      mode-locks at J=5 as expected).
- [x] **M2** — MDLM bridge (PASS: forward → `[B,L,50258]` bf16; top-k=32
      excludes mask token; mask-predict produces plausible English at
      n_iters ∈ {1,4,12} on the "capital of France" prompt).
- [x] **M3** — multi-hole agreement (PASS: at 1 LM forward + 264 Gibbs
      sweeps, THRML hits agreement 0.750 / 0.828 / 0.938 on the color /
      variable / repeat-3 templates vs 0.000 for both ancestral_topk and
      independent_full at the same LM budget; mask_predict@8 (8× LM
      forwards) only reaches 0.422 / 0.016 / 0.016. Perplexity ≈ 1.0
      across all methods — agreement is not bought with garbage tokens).
- [x] **M4** — Pareto sweep & writeup (PASS: 105-run sweep across
      `gibbs_sweeps × equality_weight × k` for THRML and `n_iters ×
      temperature` for mask-predict, ~2 min total. THRML at
      `equality_weight=10` saturates to **1.000 agreement** on all
      three templates with just 10 burn-in (74 total Gibbs sweeps) and
      a single LM forward. The strongest mask-predict baseline (T=0
      deterministic, n_iters=64) ties only on the *color* template
      (1.000) and stays at 0.000 on *variable* and *repeat-3* even at
      64 LM forwards. Headline averaged across templates: THRML ~0.84
      at 1 LM forward vs 0.33 (T=0) / 0.15 (T=1) for mask-predict at
      64 LM forwards. See `plots/{headline,pareto_flops,tsu_cost}.png`,
      `results/results.json`, and `README.md`).
- [x] **M5a** — strengthened iterative top-k baseline (PASS: 123-run
      sweep, ~2 min total. New `ancestral_topk_iterative` commits the
      most-confident hole per chain per iteration so subsequent
      forwards see real context. On *color* it lifts agreement
      0.000 → 0.469 at 2 LM forwards; on *variable* 0.016 → 0.047; on
      *repeat-3* it stays at 0.000. Still well below `mask_predict@T=0`
      on color (0.469 vs 1.000 @ 4 LM) and never beats THRML on any
      template. M4 headline survives the strengthening.
      `experiments/verify_m5a.py` covers the n_iters=1 distributional
      sanity check (top-3 token overlap, since MDLM bf16 attention
      shifts logits ~0.5 max between batch_size=1 and batch_size=64,
      making exact KL-match unreachable).
- [x] **M5b** — boundary templates (PASS: 328-run sweep across 8
      families, ~6 min total. Five new families
      (distance/many-holes/multi-group/distractor/polyseme) plus the
      original 3 core templates. THRML averages **0.80** agreement
      across the five M5b families at 1 LM forward vs 0.40 (T=0) /
      0.28 (T=1) for mask-predict at any budget; bimodal headline.
      THRML wins by ≥ 0.3 on variable, repeat-3, multi-group,
      polyseme; ties (≤ 0.1) on color, distance, distractor — all
      "favorite color is"-style cascadable templates where mp@T=0
      free-rides committed-argmax. The **many-holes** template is
      degenerate (all methods 0.000) because top-64 candidate sets at
      4 different syntactic roles share no common token — a property
      of the top-k state space, not the joint sampler. See
      `notebooks/05_boundary.py`, `plots/m5b_{headline,boundary}.png`,
      `results/results_all.json`. Required chunking
      `lm_perplexity` and the baseline forwards (16 chains/chunk) so
      the longer distance template did not OOM the 8 GB GPU.
- [~] **M5c** — learned EBM correction, **scaffolded and smoke-tested,
      awaits the A100 day**. Decision matrix locked in: objective =
      Joint-Transition NCE (positives are corpus pairs, negatives are
      independent top-k LM-marginal samples at each masked position);
      encoder = MDLM last hidden state via the new
      `MDLM.forward_hidden`; corpus = OpenWebText (HF streaming);
      eval primary = M5b distractor + polyseme. Files:
      `src/diffusion_ebm/factors/learned.py` (PairwiseScorer with
      bilinear factorisation `⟨f(h_a, x_i), g(h_b, x_j)⟩` so a full
      [k, k] pair table is two MLP forwards + one matmul);
      `src/diffusion_ebm/sampler/thrml_joint_learned.py` (parallel of
      `thrml_joint.build` swapping the equality table for the learned
      one); `experiments/m5c_train.py` (data + InfoNCE loop);
      `experiments/m5c_smoke.py` (300-step synthetic run + eval on
      color + polyseme; passes locally, 6.3 s); `experiments/m5c_eval.py`
      (M5b sweep with a trained checkpoint); `notebooks/06_learned.py`
      (overlay plot vs M5b boundary). Codex review caught one
      load-bearing bug — `(ids, unary)` swap from `top_k_candidates`
      in both smoke and eval — now fixed; smoke re-passes. The A100
      day plan is in `A100_RUNBOOK.md` (≈ 1 h setup, 18 h training,
      4 h eval).

## Environment quirks (NixOS-specific, **important**)

These cost real time to rediscover.  Read before running anything.

### 1. `LD_LIBRARY_PATH` for CUDA

NixOS keeps the kernel-driver-matched `libcuda.so` at `/run/opengl-driver/lib`,
**not** `/usr/lib`.  CUDA wheels won't find it without help.

- **Inside `nix develop`**: the flake's `profile` exports the right path
  automatically.
- **Outside `nix develop`** (e.g. when Claude runs scripts via Bash): prepend

```bash
LD_LIBRARY_PATH="/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}" uv run python ...
```

`nix develop -c <cmd>` swallowed stdout for non-interactive use, so the
`LD_LIBRARY_PATH=` wrapper is the practical workaround for scripted runs.

### 2. PyTorch wheel selection

PyPI's default `torch>=2.3` resolves to whatever the latest CUDA build is
(currently `2.11.0+cu130`).  cu130 needs NVIDIA driver ≥580 which our system
doesn't have.  We pin `torch` to PyTorch's cu124 index in
`pyproject.toml` (`[tool.uv.sources]`) and require `tool.uv.required-environments`
for x86_64 because cu124 wheels use the plain `linux_x86_64` platform tag
(not `manylinux_*`), which uv normally rejects as system-specific.

### 3. flash-attn from source

PyPI only ships sdists; pre-built wheels live in GitHub releases for
specific torch+CUDA combos.  We build from source via `no-build-isolation`
inside the FHS shell.  The flake exports:

- `CUDA_HOME` / `CUDA_PATH` → nixpkgs `cudaPackages_12.cudatoolkit`
- `TORCH_CUDA_ARCH_LIST="8.9"` → cuts build to one arch (RTX 3000 Ada).
  Bump for other GPUs.
- `MAX_JOBS="2"` → keep peak RAM ≤ 20 GB during compile.
- `LIBRARY_PATH=${glibc}/lib:${stdenv.cc.cc.lib}/lib` → the unwrapped
  binutils `ld` invoked by PyTorch's cpp_extension can't find `crti.o` /
  `crtn.o` otherwise; final `.so` link fails with "cannot find crti.o".

First `uv sync` after this is set up takes ~10 min.  Subsequent syncs reuse
the wheel cache.

### 4. Triton (used by flash-attn rotary kernel)

Triton's NVIDIA backend shells out to `/sbin/ldconfig` to discover
`libcuda.so` — NixOS doesn't ship ldconfig there.  The flake exports
`TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` which the driver checks first
and uses to skip the ldconfig probe entirely.  Without this, the very first
attention call dies with `FileNotFoundError: '/sbin/ldconfig'`.

### 5. MDLM checkpoint mechanics

- `kuleshov-group/mdlm-owt` requires `trust_remote_code=True`.
- The remote-code file uses transformers' legacy `_tied_weights_keys` API.
  **Pin `transformers<5`** in `pyproject.toml`; v5 renamed this to
  `all_tied_weights_keys` and crashes during `from_pretrained`.
- Use `dtype="auto"` (not `torch_dtype="auto"`) — transformers 4.57+
  deprecated the latter.
- Tokenizer is plain `gpt2` (50 257 vocab).  MDLM extends it with
  **one absorbing/[MASK] token at id 50 257**, so model `vocab_size = 50 258`.
- `MDLM.forward(input_ids, timesteps, return_dict=True)` returns
  `MaskedLMOutput`; **without `return_dict=True` it returns a raw Tensor**
  (config default is `use_return_dict=False`).  Our wrapper sets the kwarg
  for a stable `.logits` accessor.
- `timesteps` is `sigma` ∈ [0, 1].  Our wrapper sets sigma per-row from the
  current mask ratio; that's a sane default whether or not
  `config.time_conditioning` is True.
- Inside the model, attention runs under `torch.cuda.amp.autocast(bf16)`.
- MDLM **zeros** the `logits[..., 50_257]` slot internally (so it isn't
  the strict argmax) but it can still rank above negative logits, so set
  it to `-inf` before any sampling/top-k.  `MDLM.top_k_candidates` does
  this by default.

## How to run things

All commands assume CWD = `/home/ghawk/diffusion-ebm`.  Inside `nix develop`
the `LD_LIBRARY_PATH` is already set; outside, prepend it manually.

```bash
# Enter dev shell (FHS + CUDA + nvcc env vars).
nix develop

# Sync deps.  Triggers flash-attn build on first run (~10 min).
uv sync

# Smoke test (M0 — imports + GPU detection + tokenizer round-trip).
uv run python notebooks/00_smoke.py

# Potts chain (M1 — synthetic sampler sanity, ~30s after JIT).
uv run python notebooks/01_mvp0_potts.py

# MDLM bridge (M2 — first run downloads ~500 MB checkpoint).
uv run python notebooks/02_mdlm_bridge.py
```

Outside `nix develop` (e.g. scripted/agent contexts):

```bash
LD_LIBRARY_PATH="/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}" \
    uv run python notebooks/01_mvp0_potts.py
```

## THRML API gotchas worth remembering

- `CategoricalNode` stores state as `uint8` → **k ≤ 256**.  Top-k candidate
  sets at k=64 fit comfortably; full GPT-2 vocab does not.
- `BlockGibbsSpec(free_blocks, clamped_blocks)` has no `node_shape_dtypes`
  arg in current versions despite older docs claiming so.
- `CategoricalEBMFactor([Block(left), Block(right)], weights)` weight shape
  is `(num_edges, k_left, k_right)` — leading axis is "parallel edges that
  share this factor type", not a literal batch dim.
- Block coloring is **manual**.  Use
  `networkx.coloring.greedy_color(graph, strategy="DSATUR")` and pass the
  resulting Blocks into `free_blocks`.
- `sample_states(key, program, schedule, state_free, state_clamped, target_nodes)`
  takes positional args; returns a list of arrays, one per target Block.
- `FactorSamplingProgram(spec, samplers, factors, [])` — 4th arg is `[]`
  (other interactions); `samplers` is **one per free block**, not one per
  factor.  Same sampler instance can be reused: `[samp, samp]`.

The verbatim categorical pattern (from `tests/test_discrete_ebm.py`) is the
canonical reference if a future API change confuses things.

## MDLM internals worth remembering

- The model class is `MDLM(transformers.PreTrainedModel)` defined in
  the trust_remote_code file
  `~/.cache/huggingface/hub/models--kuleshov-group--mdlm-owt/.../modeling_mdlm.py`.
- Backbone is `DITBackbone` (DiT-style) with rotary embeddings.
- Two flash_attn calls inside `DDiTBlock.forward`:
  - `flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)`
  - `flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, max_seqlen, 0., causal=False)`
- `transformers` itself probes `flash_attn` via `find_spec` at import of
  `transformers.integrations.flash_attention`.  We tried a torch-native
  shim earlier and the spec-less fake module failed that probe; current
  approach is to install real flash-attn instead.

## Open questions / decisions deferred

- **Mask-predict baseline strength** — the strongest baseline at M3.  If it
  dominates the Pareto front at all budgets, the project pivots to "where
  in (sparsity × constraint-density) space does Gibbs win" — still
  publishable.
- **Equality factor weight** — too small → no effect, too large → freezes
  Gibbs.  Sweep `equality_weight ∈ {2, 5, 10}` at M3.
- **Top-k truncation** — high-entropy positions may need k > 64 to keep
  the gold token in candidates.  Consider top-p (nucleus) candidate sets
  if k=64 hurts quality; report at multiple k.
- **Mode mixing in MVP0** — at J=5 the chain locks into one mode.  This is
  expected and not blocking, but if we ever want to demonstrate mode-mixing
  we'd need annealing or lower J.

## File-locating tips

- THRML source (read-only): `.venv/lib/python3.12/site-packages/thrml/`
- THRML categorical reference test:
  https://github.com/extropic-ai/thrml/blob/main/tests/test_discrete_ebm.py
  (function `test_categorical`).
- MDLM modeling file (downloaded by trust_remote_code):
  `~/.cache/huggingface/hub/models--kuleshov-group--mdlm-owt/snapshots/<hash>/modeling_mdlm.py`
- The approved plan: `/home/ghawk/.claude/plans/nested-wishing-thimble.md`.
