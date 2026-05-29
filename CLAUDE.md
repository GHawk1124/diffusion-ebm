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

  src/diffusion_ebm/
    backbones/
      mdlm.py                  # M2: MDLM HF wrapper + forward_hidden
    factors/
      equality.py              # M3: hard equality [k,k] table
      learned.py               # M5c: PairwiseScorer bilinear ψ + log_temp
      unary.py
    sampler/
      thrml_joint.py           # M3: hard-equality THRML factor graph
      thrml_joint_learned.py   # M5c: learned-ψ THRML factor graph
      baselines.py             # M5a: ancestral, mask-predict, iterative-topk
    tasks/
      multihole.py             # M3–M5b: 8 template families + all_templates()
    metrics/
      agreement.py             # agreement_rate, lm_perplexity
    synth/
      potts_chain.py           # M1: ferromagnetic Potts chain
    utils/
      coloring.py              # graph coloring for THRML block assignment

  notebooks/                   # all are runnable .py with jupytext-style cells
    00_smoke.py                # M0: env check
    01_mvp0_potts.py           # M1: Potts sanity check
    02a_mdlm_probe.py          # M2: probe MDLM API surface
    02_mdlm_bridge.py          # M2: end-to-end MDLM forward + mask-predict
    03_mvp1_multihole.py       # M3: multi-hole agreement headline
    04_pareto.py               # M4: Pareto plots + summary table
    05_boundary.py             # M5b: boundary template sweep
    06_learned.py              # M5c: learned-ψ overlay vs M5b boundary
    07_tour.py                 # demo tour
    08_tour_marimo.py          # marimo interactive version

  data/
    rmc/                       # frozen benchmark artefacts (committed)
      owt_heldout_single.jsonl
      owt_heldout_multi.jsonl
      wikitext103_single.jsonl
      wikitext103_multi.jsonl

  experiments/
    run_pareto.py              # M4: sweep driver
    verify_m5a.py              # M5a: distributional sanity check
    m5c_smoke.py               # M5c: 300-step synth train + pipeline check
    m5c_train.py               # M5c: OWT + synth training loop
    m5c_eval.py                # M5c: full template sweep with --temp-override
    m5c_eval_winogrande.py     # M5d Stage 1 (WG, out-of-scope; appendix)
    build_rmc.py               # M5d Stage 1': deterministic RMC extractor
    m5c_eval_rmc.py            # M5d Stage 1': RMC eval (built; 11 methods —
                               #   core + Track-0 audit baselines: hard_eq_map,
                               #   mcmc_logits, best_of_n_strong,
                               #   predicted_group_hard_eq (+grouping ARI/F1/
                               #   over-merge); --corpus/--track/--subsample-seed/
                               #   --group-threshold shard flags; writes JSON
                               #   ONLY at end — wall-kill loses everything)
    analyze_rmc.py             # Track-0/A offline analysis: item-level bootstrap
                               #   CIs + paired Δ-vs-baseline on chain_em_supported
                               #   + de-oracle grouping + n_lm cost over sharded
                               #   JSONs; pure stdlib (runs on system python, no
                               #   CUDA/LD dance); reproduces dev findings exactly

  src/diffusion_ebm/tasks/
    winogrande.py              # WG dataclass + build_items (appendix material)
    rmc.py                     # RMC dataclass + load_items + is_dev split

  slurm/
    m5c_train_modules.sbatch   # v3 training job (k=128, 200k steps)
    m5c_train_k256.sbatch      # v3 k=256 retrain (THRML uint8 ceiling)
    m5c_eval_override.sbatch   # A3 temp-override eval (historical)
    m5c_eval_k256.sbatch       # k=256 eval on existing checkpoint
    m5c_eval_winogrande.sbatch # Stage 1 WG eval (appendix; do not delete)
    m5c_eval_rmc.sbatch        # Stage 1' RMC eval (built; 3-state SPLIT,
                               #   CORPUS/TRACK/SUBSAMPLE_SEED/GROUP_THRESHOLD
                               #   shard env vars)
    submit_rmc_dev.sh          # Stage 1' dev-gate launcher: 4 sharded jobs,
                               #   typed gres (gpu:<type>:1, never v100),
                               #   --cpus-per-task=4 (l40s 4:1 cap)

  plots/                       # gitignored
  results/                     # gitignored; PACE scratch at
                               #   /storage/scratch1/1/gcomes3/diffusion-ebm/
```

## Status (2026-05-28, M5c complete at k=256; M5d RMC dev gate PASSED but reframed — joint decoding wins, learned ψ ≈ hard equality. PIVOT decided: two-track plan (safe systems result + frustrated-posterior hiring artifact), method UNLOCKED — see M5d block + plan file)

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
      64 LM forwards.)
- [x] **M5a** — strengthened iterative top-k baseline (PASS: 123-run
      sweep, ~2 min total. New `ancestral_topk_iterative` commits the
      most-confident hole per chain per iteration. On *color* lifts
      0.000 → 0.469 at 2 LM forwards; never beats THRML on any template.
      M4 headline survives the strengthening.)
- [x] **M5b** — boundary templates (PASS: 328-run sweep across 8
      families. Five new families (distance/many-holes/multi-group/
      distractor/polyseme). THRML averages **0.80** agreement across the
      five M5b families at 1 LM forward vs 0.40/0.28 for mask-predict.
      The **many-holes** template is degenerate at k=64 — top-k candidate
      sets share no common token, a property of the state space not the
      sampler.)
- [x] **M5c** — learned EBM correction (COMPLETE after 3 PACE runs on
      H200). Final configuration: `PairwiseScorer` bilinear ψ
      `⟨f(h_a, x_i), g(h_b, x_j)⟩`, tabular NCE with full [B, k, k]
      cross-entropy, repeat-token tier-1 mining, expanded synthetic corpus
      (~300 seed sentences across variable/name/polyseme/multi-group axes),
      log_temp clamped to [-1.0, 0.5], synth-frac 0.30→0.10 over 150k
      steps, k=128 training.

      **v3 + k=256 eval results (step 200k checkpoint, 2026-05-27):**

      | Template | k=128 eval | k=256 eval | Hard-equality THRML |
      |---|---|---|---|
      | color | 1.000 | 1.000 | 1.000 |
      | distractor | 1.000 | 1.000 | 1.000 |
      | car-boot polyseme | 1.000 | 1.000 | 1.000 |
      | color-cascade | 1.000 | 1.000 | 1.000 |
      | variable | 1.000* | 1.000* | 1.000 |
      | repeat-3 (name) | 0.062 | 0.047 | 0.000 |
      | **room-polyseme** | 0.000 | **0.656** | 0.000 |
      | **multi-group** | ~0.000 | **1.000** | 0.000 |

      *Variable "1.000" is agreement on `' variable'`/`' that'` — the
      true target token `' x'` is outside MDLM's top-256 in prose context
      (documented top-k support ceiling; not a scorer failure).

      **Key diagnostic finding (C2):** all remaining failures are top-k
      support bottlenecks, not scorer quality. Room-polyseme and multi-group
      recovered at k=256 because subject pronouns / proper names entered the
      candidate set at positions 129-256. Repeat-3 hole 2 (`"I said [M]
      three times"`) still predicts "it"/"that" — no name token at k=256.

      Artefacts on PACE scratch:
      - v3 checkpoint: `results/m5c_v3/scorer_step00200000.pt`
      - v3 eval (k=128 train, k=128 eval): `results/m5c_v3_eval.json`
      - k=256 eval (k=128 train, k=256 eval): `results/m5c_v3_eval_k256.json`

- [~] **M5d** — publication preparation (EMNLP Findings target, in
      progress). Staged plan at
      `/home/ghawk/.claude/plans/shiny-watching-sundae.md`.

      **Stage 0 (k=256 native retrain) — COMPLETE.**
      `slurm/m5c_train_k256.sbatch` ran on H200 for 175 min (200k steps).
      All 8 templates → 1.000 agreement at some ws; three genuine wins over
      hard-equality (which scores 0.000 by construction on these):
      repeat-3 (name), room-polyseme, multi-group.
      Checkpoint: `results/m5c_v3_k256/scorer_step00200000.pt`
      Eval JSON: `results/m5c_v3_k256_eval.json`

      **Stage 1 (WinoGrande forced-choice rerank) — FAILED, reframed.**
      Pilot (100 items) + full run (1082 items after length-mismatch drop):
      best psi_rerank (ws=1.0) = 0.507, lm_rerank = 0.491. Gate was +3 pp;
      achieved only +1.6 pp. Diagnosis (confirmed via gpt-5.5-xhigh codex):
      ψ encodes *identity* consistency (tabular NCE + repeat-token mining),
      WG measures *reference* consistency (pronoun → distinct antecedent).
      Additionally, the WG rerank adapter sums ψ(option, nearby_context)
      with no joint Gibbs — it never exercises the method's core inference
      claim. Reframed as out-of-scope appendix material.

      **Stage 1' (Repeated Mention Cloze, RMC) — DEV GATE PASSED 2026-05-28,
      BUT REFRAMED.** 10-day time-box, Day-7 dev gate. Real-corpus benchmark
      where same single-token entity appears ≥ 2 times in a 64-token window;
      all occurrences masked; method must recover jointly. Two tracks:
      `single_chain` (one entity; hard-eq is upper bound) and `multi_chain`
      (2–4 entities). Corpora: OWT held-out (last 1000 docs by HF ordering) +
      WikiText-103 validation. Frozen .jsonl artefacts in `data/rmc/`.
      Day-7 gate: learned_psi_thrml `chain_em_supported` ≥ mdlm_argmax +5 pp
      on `multi_chain` for ≥1 corpus AND ties hard_eq_oracle within 3 pp
      on `single_chain`. **Both conditions met — gate PASSES.**

      **CRITICAL FINDING (2026-05-28 dev re-run, sharded, ws∈{0.5,1.0,2.0},
      burn=500, 200 items/cell, paired bootstrap on the method-independent
      `all_supported` set):** the gate passed on the *joint-decoding* claim,
      NOT the *learned-scorer* claim. The learned ψ is statistically
      interchangeable with parameter-free hard equality everywhere, and a new
      diagnostic method (`learned_psi_thrml_global`, ψ over ALL hole pairs
      with no oracle grouping) is *negative*. Three robust conclusions:

      1. **Joint factor-graph decoding beats independent argmax** by ~9–14 pp
         (`chain_em_supported`), CIs exclude 0 on every cell:
         | cell (supported n) | argmax | hard_eq_oracle | ψ best (ws=0.5) | ψ−argmax | ψ−hard_eq |
         |---|---|---|---|---|---|
         | OWT multi (116) | 0.181 | **0.319** | 0.284 | +0.103 SIG | −0.034 n.s. |
         | OWT single (170) | 0.424 | 0.524 | 0.524 | +0.100 SIG (hard_eq) | +0.000 n.s. |
         | WT single (151) | 0.298 | 0.384 | 0.404 | +0.086 SIG (hard_eq) | +0.020 n.s. |
         | WT multi (93) | 0.108 | **0.183** | 0.172 (ws=1.0) | +0.065 SIG | −0.011 n.s. |
         (WT multi confirmed 2026-05-29 via analyze_rmc.py — all 4 cells agree.)

      2. **Learned ψ ≈ hard equality (`==`).** ψ ties hard_eq_oracle within
         noise on all measured cells (Δ ∈ [−0.034, +0.020], all n.s.); at
         ws≥1.0 ψ is *worse*. The learned factor is interchangeable with
         hard equality on RMC — no learned-scorer advantage.

      3. **`learned_psi_thrml_global` is negative.** Without oracle grouping
         the learned factor does NOT recover identity structure: it ties
         naive `hard_eq_global` (OWT: ψ_global 0.078 vs 0.060, Δ=+0.017 n.s.)
         and both sit near the floor vs oracle's 0.319. At ws≥1.0 ψ_global
         collapses to 0.009 (over-merges distinct entity chains). The one
         test that could have justified the learned scorer fails.

      **Reframed honest claim:** "Joint factor-graph decoding (THRML
      block-Gibbs) over MDLM candidates with known entity grouping improves
      multi-mask identity consistency by ~9–14 pp over independent sampling
      at matched neural FLOPs; a learned pairwise scorer is interchangeable
      with hard equality on this benchmark." The learned-ψ headline from the
      template experiments (M5c) does NOT survive contact with real-corpus
      RMC. See [[m5d-rmc-dev-findings]] decision point in the plan.

      **wt_multi RESOLVED (2026-05-29):** all 4 dev shards are local in
      `results/m5c_rmc_dev_*_smoke200.json` and folded into the 4-cell table
      above via `experiments/analyze_rmc.py`. wt_multi confirms the other
      three: hard_eq−argmax +0.075 SIG, ψ−argmax +0.065 SIG, ψ−hard_eq
      −0.011 n.s. The old-dev shards predate the Track-0 baselines, so the
      Track-0 dev sweep (`bash slurm/submit_rmc_dev.sh`, no --gate-only) must
      be (re)run to populate hard_eq_map / mcmc_logits / best_of_n_strong /
      predicted_group_hard_eq before Track-A test-split execution.

      **PIVOT (2026-05-28): two-track plan, method UNLOCKED.** After the
      codex gpt-5.5-xhigh second opinion + critique, the dead learned-ψ
      thesis is replaced by a two-track plan (active plan:
      `/home/ghawk/.claude/plans/shiny-watching-sundae.md`). The key
      realisation: hard-eq + *oracle grouping* has a closed-form
      product-of-experts MAP (`argmax_t Σ_i log p_i(t)`) — no sampling
      needed — so the current RMC benchmark is exactly the case where
      sampling hardware buys nothing. Bad for the Extropic goal.

      - **Track 0 (do first, ~1 wk, gates everything):** additive audit
        baselines on the *dev* split — `hard_eq_map` (exact pooling, will
        tie THRML and prove sampling earns nothing here), `mcmc_logits`
        (no-factor Gibbs control, promoted to REQUIRED), `best_of_n_strong`
        (real joint-LM-scored best-of-N, not the ≈argmax `best_of_n_cheap`),
        and `predicted_group_hard_eq` (de-oracle probe). Decision rule in
        the plan.
      - **Track A (weeks 2–4, safety net, NOT where effort goes):** ship
        framing (a) — "cheap joint decoding helps ~9–14 pp at matched FLOPs;
        hard equality suffices, learned factor adds nothing; no-free-lunch
        boundary." Do NOT run the test-split sweep until Track 0 adds the
        honest baselines.
      - **Track B (weeks 2–7, the hiring artifact, marginal effort goes
        here):** latent-partition RMC as **correlation clustering** —
        attraction (equality) + a NEW antiferromagnetic **repulsion /
        inequality factor** = a *frustrated* Potts posterior. Removes the
        oracle crutch, is multimodal, exact MAP is NP-hard, and is the
        canonical thermodynamic-hardware (frustrated Ising/Potts) workload.
        Report grouping ARI/F1 + over-merge rate, mixing diagnostics,
        hardware-native graph properties, FLOPs↔wall-clock Pareto.

      **Method status: UNLOCKED for Track B.** `factors/learned.py`,
      `sampler/thrml_joint*.py`, factor design, temperature/schedule are now
      in scope. New files expected: `factors/inequality.py`,
      `sampler/thrml_latent_partition.py`. Still frozen for comparability:
      the RMC benchmark itself (`tasks/rmc.py`, `build_rmc.py`,
      `data/rmc/*.jsonl`) — Track B adds a latent-partition *view*, never
      mutates the frozen artefacts.

      Three seeds + item-level bootstrap CIs and wall-clock cost measurement
      remain as the weeks 10–11 buffer for whichever track becomes the
      headline.

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

## PACE submission quirks (learned 2026-05-28, **important**)

- **GPU type selection: use a typed gres, NOT a `--partition` list.** PACE
  does site-side partition routing — a multi-partition `--partition` list
  gets expanded and silently re-adds `gpu-v100`, so jobs land on V100s
  despite v100 not being in the list. A typed gres (`--gres=gpu:a100:1`,
  `gpu:h200:1`, `gpu:l40s:1`, `gpu:h100:1`) can only be satisfied by a node
  that actually has that GPU, so it cannot land on v100 regardless of
  routing. Partitions are GPU-type-pure (`gpu-a100` holds only a100, etc.).
- **`gpu-l40s` enforces a 4:1 CPU:GPU ratio.** The base eval sbatch requests
  `--cpus-per-task=8` with 1 GPU (8:1) → rejected with "Invalid gres". Pass
  `--cpus-per-task=4` (4 cores is plenty for this eval; satisfies the cap on
  all partitions). `submit_rmc_dev.sh` already does this.
- **Find free GPUs before submitting:**
  `sinfo -p gpu-a100,gpu-l40s,gpu-h200,gpu-h100 -t idle,mix -o "%P %t %D %G"`
  (`mix`=partially free, `drng`/`drain`=draining, avoid).
- **`m5c_eval_rmc.py` writes its JSON only as the final step.** A wall-kill
  (`#SBATCH --time` exceeded) loses ALL in-memory records — no incremental
  checkpoint. Always shard so each job finishes well under the wall, and
  size `--time` generously (single_chain ~40 min, multi_chain heavier).
- **Pull dev shards down (run on the local NixOS host, not PACE):**
  `scp 'gcomes3@login-phoenix.pace.gatech.edu:/storage/scratch1/1/gcomes3/diffusion-ebm/results/m5c_rmc_dev_*_smoke200.json' results/`

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

## Settled questions (previously open)

- **Mask-predict baseline strength** — resolved. mask-predict@T=0 wins on
  cascadable templates (color, distractor) but fails on templates requiring
  true joint constraints (variable, multi-group, polyseme). Learned-ψ THRML
  is the winner there.
- **Equality factor weight** — resolved. `equality_weight=1.0` with a
  learned ψ is the working configuration. `weight_scale` in [0.5, 2.0] is
  the sweet spot; ws≥5 over-constrains and can hurt ppl.
- **Top-k truncation** — resolved empirically. k=64 is too small (12% miss
  rate on positive pairs, and room-polyseme/multi-group fail completely).
  k=128 reduces miss rate to ~11% but still misses pronoun case at hole 3.
  k=256 (THRML uint8 hard ceiling) unlocks room-polyseme (0.656) and
  multi-group (1.000). **k=256 is the final training configuration.**
- **Runaway log_temp** — resolved. v2 hit T=15.5 (log_temp=2.74) via
  unconstrained learning. v3 clamps log_temp ∈ [-1.0, 0.5] (T ∈ [0.37,
  1.65]); the model hits the upper clamp by step ~50k and stays there.
- **ψ scope: identity vs reference consistency** — resolved. ψ encodes
  *identity* consistency (same surface token at multiple masked positions),
  NOT *reference* consistency (pronoun → distinct antecedent). This is
  set by the tabular NCE + repeat-token mining objective.  WinoGrande
  (pronoun resolution) is out of scope.  Expanding to reference consistency
  would require a retrain (Stage 1'' / Option E).
- **Does the learned ψ beat hard equality on real text?** — resolved
  2026-05-28 (RMC dev, paired bootstrap): NO. ψ ties `==` within noise on
  every measured cell and is worse at ws≥1.0. The M5c template "wins" were
  artefacts of hand-built templates; on RMC the identity signal ψ encodes
  is already captured by hard equality. The *joint decoding* (any coupling
  factor vs independent argmax) is the real, significant effect (+9–14 pp).
- **Can ψ recover entity grouping without oracle labels?** — resolved
  2026-05-28: NO. `learned_psi_thrml_global` (all-pairs, no grouping) ties
  naive global hard-equality at the floor and collapses at ws≥1.0.

## Open questions / M5d decisions pending

- **RMC Day-7 dev gate** — RESOLVED 2026-05-28: gate PASSES (OWT multi
  ψ−argmax +10.3 pp SIG; both single-chain cells tie hard_eq within 3 pp).
  But it passed on the *joint-decoding* claim, not the *learned-scorer*
  claim — ψ is interchangeable with hard equality, and `ψ_global` is
  negative. See Stage 1' block above and the plan's decision point.
- **THE framing decision** — RESOLVED 2026-05-28: "both — safe + ambitious"
  + method UNLOCKED. Track A ships framing (a) as the safety net; Track B
  (frustrated-posterior latent-partition RMC) is the groundbreaking +
  Extropic-aligned hiring artifact. Framing (c) (coreference retrain) is
  dropped — Track B targets latent partition (a frustrated Potts inference
  `==` cannot express) instead. See M5d block + plan file.
- **Track 0 open questions — ALL RESOLVED 2026-05-29** (dev sweep, 4 shards,
  200 items/cell, ws∈{0.5,1.0,2.0}, burn=500, paired bootstrap B=10000 on the
  method-independent supported set; full report
  `results/rmc_dev_track0_analysis.md` via `experiments/analyze_rmc.py`):
  - `hard_eq_map` (exact pooling MAP) vs THRML hard-eq → **MATCHES OR BEATS,
    never loses.** Δ(map−oracle): owt_multi +0.026 n.s., owt_single +0.024
    SIG, wt_multi +0.032 n.s., wt_single +0.040 n.s. The closed-form argmax
    (n_lm=1, ~0.07s, agree=1.000 by construction) *strictly dominates* the
    sampler (~2.5s) — stronger than the predicted tie. **Sampling earns
    nothing on the oracle-grouped equality task → Track B frustration is
    REQUIRED and well-motivated. GREEN LIGHT for Track B.**
  - `mcmc_logits` (no-factor Gibbs) ≈ argmax → **CONFIRMED.** Δ(mcmc−argmax)
    ∈ [−0.011, +0.006], all n.s. on every cell. The +7–14 pp joint win comes
    from the *factor*, not from Gibbs (honest-framing insight, empirically).
  - `best_of_n_strong` (joint-LM-scored N=64, 65× FLOPs) → **does NOT threaten
    the headline; it LOSES badly.** Δ(oracle−BoN): owt_multi +0.224, owt_single
    +0.282, wt_multi +0.118, wt_single +0.225, all SIG. BoN_strong even
    underperforms plain argmax (re-ranks fluent-but-wrong ancestral fills).
    Matched-FLOPs headline is SAFE and strengthened.
  - `predicted_group_hard_eq` (de-oracle) → **SPLIT by track.** single_chain:
    recovers a meaningful, SIG fraction of oracle gain (owt 24%, wt 54%;
    over-merge 0.000) → honest NLP result on single. multi_chain: recovers
    ~0% / hurts (owt 0%, wt −29%; over-merge 0.11–0.13, ARI ~0.50) → Jaccard
    over-merges distinct entities — the SAME failure as ψ_global, and exactly
    the gap Track B's repulsion/inequality factor is designed to close.
  - **Track A headline gate (verification table) PASSES on dev:** (best joint
    − argmax) SIG on both multi cells; (joint − best_of_n_strong) SIG on all
    4; (ψ − hard_eq_oracle) CI includes 0 on all 4. Still must re-confirm on
    the test split before the paper's headline table.
- **Track B open questions:**
  - Can the attraction+repulsion (frustrated) graph stop the `ψ_global`
    over-merge collapse and recover entity partition (grouping ARI) without
    oracle labels?
  - Is there a Pareto regime where block-Gibbs strictly dominates argmax,
    exact pooling, and best-of-N — the hardware-justifying figure?
- **Variable template framing** — `' x'` is outside MDLM's top-256 in
  prose context. Current "1.000 agreement" is on `' variable'`/`' that'`.
  Must be framed as a top-k ceiling example in the paper, not a success.
  Possible fix: add a code-context variable template where MDLM predicts
  identifiers; deferred to M5d Stage 1 review.
- **Repeat-3 hole 2** — `"I said [M] three times"` context predicts
  "it"/"that", not names, even at k=256. Template may need redesign or
  drop from primary eval.
- **k=256 retrain calibration** — does room-polyseme stabilise across all
  weight-scales when trained natively at k=256 (vs current 0.656 only at
  ws=0.5)? Answered by the pending `m5c_train_k256.sbatch` run. (Now low
  priority — template results are superseded by RMC.)

## File-locating tips

- THRML source (read-only): `.venv/lib/python3.12/site-packages/thrml/`
- THRML categorical reference test:
  https://github.com/extropic-ai/thrml/blob/main/tests/test_discrete_ebm.py
  (function `test_categorical`).
- MDLM modeling file (downloaded by trust_remote_code):
  `~/.cache/huggingface/hub/models--kuleshov-group--mdlm-owt/snapshots/<hash>/modeling_mdlm.py`
- M5d plan (active): `/home/ghawk/.claude/plans/shiny-watching-sundae.md`
- PACE scratch (15 TB): `/storage/scratch1/1/gcomes3/diffusion-ebm/`
  - v3 checkpoint: `results/m5c_v3/scorer_step00200000.pt`
  - v3 eval JSONs: `results/m5c_v3_eval.json`, `results/m5c_v3_eval_k256.json`
  - k=256 native retrain checkpoint: `results/m5c_v3_k256/scorer_step00200000.pt`
  - k=256 native retrain eval: `results/m5c_v3_k256_eval.json`
  - RMC eval outputs (to create): `results/m5c_v3_k256_rmc_{owt_heldout,wikitext103}_{dev,test}.json`
- RMC frozen benchmark artefacts: `data/rmc/*.jsonl` (in repo, committed)
