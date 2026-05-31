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
      inequality.py            # Track B: antiferromagnetic repulsion [k,k] table
      learned.py               # M5c: PairwiseScorer bilinear ψ + log_temp
      unary.py
    sampler/
      thrml_joint.py           # M3: hard-equality THRML factor graph
      thrml_joint_learned.py   # M5c: learned-ψ THRML factor graph
      thrml_latent_partition.py # Track B: frustrated attract+repel token sampler
                               #   (partition_to_edges / jaccard_partition / build / sample)
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
    probe_splitability.py      # Track B scout 1: exact partition posterior over
                               #   real over-merged dev components (YELLOW; 0.44
                               #   split-recovery ceiling — top-k bound). Builds
                               #   results/probe_splitability_cache.json (1 MDLM
                               #   forward/item) reused by scouts 2+3.
    probe_hardness_dial.py     # Track B scout 2: frustrated Potts w/ real MDLM
                               #   fields. EXP A Pareto crossover n≈11; EXP B
                               #   metastability wall at w≥8; EXP B2 burn-in
                               #   invariant. GREEN. (imports jax/flp)
    probe_tempering.py         # Track B scout 3: numpy block-Gibbs + annealing +
                               #   parallel tempering vs the wall. PT restores
                               #   TV→MC-floor (0.41→0.008) at ~9× cost. Hero fig.
                               #   EXP C barrier / EXP D sparsity / EXP E THRML↔
                               #   numpy cross-val (proxy confirmed; target peaked
                               #   not multimodal; wall = ensemble over-dispersion,
                               #   single-chain TV is high-variance).
    probe_partition_hardness.py # Track B scout 4: do REAL full-window RMC graphs
                               #   land in the hard regime? Exact Bell(n) partition
                               #   posterior at full L=256 via pooled product-of-
                               #   experts (per-group factorisation removes the L^n
                               #   blowup; n≤10 → Bell≤116k, <1s). VERDICT: NO —
                               #   posteriors are ambiguous (53% multimodal, p_MAP
                               #   ~0.66 @ β=4) but exact-in-<1s AND single-T Gibbs
                               #   mixes to MC floor (TV 0.008) even on the most
                               #   ambiguous items → no barrier; ambiguity≠hardness.
                               #   Closes the open question: hard regime is synthetic
                               #   only. (imports probe_splitability helpers; numpy)

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
    submit_rmc_test.sh         # Track-A test-split launcher: 4 shards, SPLIT=test,
                               #   full method set (no --gate-only), ws∈{0.5,1.0},
                               #   MAX_ITEMS caps (single 900 / multi 450), 12h wall

  plots/                       # gitignored
  results/                     # gitignored; PACE scratch at
                               #   /storage/scratch1/1/gcomes3/diffusion-ebm/
```

## Status (2026-05-31, M5d Track A test-split LOCKED — joint factor-graph decoding beats independent argmax by +7–15 pp on the held-out test split (all 4 cells SIG); hard_eq_map (closed-form pooling) DOMINATES the sampler; learned ψ ≤ hard equality (SIG-worse on wt_single). Track A is DONE. Track B SCOUTED via 4 fail-fast probes — RMC is too easy for the sampler (splitability ceiling 0.44; scout-4 full-window partition posterior is ambiguous but exact-in-<1s AND mixes to the MC floor TV=0.008 → no barrier, ambiguity≠hardness, real-hard-task question CLOSED negative), BUT a dialed-hardness frustrated Potts has a genuine regime (n≳12) where exact dies + single-T Gibbs hits a burn-in-invariant metastability wall (w≥8) + parallel tempering crosses it (TV 0.41→0.008 at ~9×). Thesis reshaped: hero fig = Pareto+metastability+tempering, legitimately SYNTHETIC, NOT RMC accuracy. See Track B SCOUT FINDINGS block + plan file)

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

      **TRACK A TEST-SPLIT — LOCKED HEADLINE (2026-05-29).** Full method set
      (core + Track-0 baselines, NO `--gate-only`) on the held-out 80% test
      split via `slurm/submit_rmc_test.sh` (4 GPU shards: single→a100,
      multi→h200) merged offline with `experiments/analyze_rmc.py`.
      `chain_em_supported`, paired bootstrap B=10000 on the method-independent
      supported set (per-shard MAX_ITEMS caps → supported n below):

      | cell (n) | argmax | hard_eq_oracle | hard_eq_map | ψ best | joint−argmax |
      |---|---|---|---|---|---|
      | owt multi (250)  | 0.104 | 0.224 | **0.252** | 0.212 | +0.120 SIG |
      | owt single (753) | 0.359 | 0.507 | **0.541** | 0.494 | +0.149 SIG |
      | wt multi (213)   | 0.042 | 0.108 | **0.127** | 0.085 | +0.066 SIG |
      | wt single (706)  | 0.303 | 0.432 | **0.466** | 0.409 | +0.129 SIG |

      Test reproduces dev on ~4× the items; every Track-0 audit verdict holds:
      - `mcmc_logits − argmax` n.s. on all 4 — the win is the *factor*, not Gibbs.
      - `joint − best_of_n_strong` SIG on all 4 — matched-FLOPs headline SAFE.
      - `hard_eq_map` (closed-form pooling, n_lm=1) SIG-DOMINATES the oracle
        *sampler* on both single cells (+0.033, +0.034) and ties/wins on multi.
        Sampling earns nothing on the oracle-grouped equality task → Track B
        frustration is REQUIRED. **GREEN LIGHT for Track B.**
      - `predicted_group_hard_eq` (de-oracle) recovers 30–32% of oracle gain on
        single_chain (over-merge 0.000) but only 7% / −14% on multi_chain
        (over-merge ~0.09–0.10) — the exact over-merge failure Track B's
        repulsion factor is designed to close.
      - **ψ − hard_eq_oracle:** n.s. on 3 cells but **SIG-WORSE on wt_single**
        (−0.023 [−0.044, −0.001]). The learned scorer is interchangeable-at-
        best and slightly *hurts* on one cell. Honest claim tightens from
        "ψ ≈ hard_eq" to **"ψ ≤ hard_eq (ties or loses); hard equality is the
        correct, parameter-free factor."** **Track A is DONE.**

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
      - **Track A (weeks 2–4, safety net, NOT where effort goes) — DONE
        2026-05-29:** ships framing (a) — "cheap joint decoding helps ~7–15 pp
        at matched FLOPs; hard equality suffices, learned factor adds nothing
        (ψ ≤ hard_eq); no-free-lunch boundary." Test-split sweep ran with the
        full Track-0 baseline set; see TRACK A TEST-SPLIT block above.
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

      **TRACK B SCOUT FINDINGS (2026-05-30) — three fail-fast probes, all on
      the dev split / real MDLM logit fields, pure-numpy exact ground truth.**
      Built before committing sampler effort; they reshaped the thesis.

      1. **Splitability probe (`experiments/probe_splitability.py`) — YELLOW.**
         Exact marginal partition posterior p(z) over the REAL over-merged dev
         multi_chain Jaccard components, under the proposed frustrated energy
         `Σ unary + Σ w_ij(2a−1) + λ Σ(2a−1)(2s−1)` (a=same group, s=same
         token, w_ij=β(J_ij−τ)), swept over a (λ,β) grid. Question: can a single
         (λ,β) recover the entity splits Jaccard over-merges WITHOUT collapsing
         correct merges? Best `go_score = min(split_recovery, merge_preservation)
         = 0.441` at (λ=0.5, β=4); split-recovery ceiling ~0.41–0.44, robust to
         the τ sweep. Diagnosis: bounded above by the **top-k support ceiling**
         (gold token in MDLM top-32 only 61% of masked entities; top-1 22%) +
         token-vs-entity confusion. Because this is the EXACT posterior, *any*
         sampler is bounded by it, and on the small components RMC produces
         (n≤5) exact enumeration is cheap → no sampling advantage on real RMC.
         Cache: `results/probe_splitability_cache.json` (one MDLM forward/item,
         reused by the other two probes); report `results/probe_splitability.json`.

      2. **Hardness-dial scout (`experiments/probe_hardness_dial.py`) — GREEN,
         thesis sharpened.** Dials hardness directly: frustrated Potts, n holes
         over a shared L=4 alphabet, unary = real MDLM logit fields, couplings =
         controlled spin-glass (random ± w, repel_frac 0.5). EXP A (Pareto):
         exact brute force vs THRML block-Gibbs as n grows — crossover at n≈11
         (exact 1.79 s vs Gibbs 1.60 s), exact INFEASIBLE at n≥12 (L^n>16.8M),
         Gibbs ~flat (0.55→2.7 s) AND faithful (TV≈0.02) in the checkable
         regime. EXP B (metastability): at fixed n=8, sweep w — TV(Gibbs,exact)
         is ~0 for w≤4 but jumps to ~0.58 at w≥8 (single-chain Egap 3.44 nats,
         mapAcc→0.375). **(EXP E correction: these w≥8 single-chain numbers are a
         single-instance draw from a HIGH-VARIANCE estimator — over 8 instances
         single-chain TV is 0.19±0.22; the robust, low-variance metastability
         signal is the independent-ensemble TV 0.49±0.11. The wall is real; the
         single-chain point estimate is not the way to report it.)** EXP B2: TV
         flat (~0.70) across burn_in 50→3200 → genuine **metastability** (energy
         barriers), not slow mixing. Report:
         `results/probe_hardness_dial.json`. CAVEATS: exact=brute force (but a
         complete graph has treewidth n−1, so smart exact is also exponential);
         Gibbs has a ~0.5 s CPU/JAX overhead floor (crossover is conservative);
         dense graphs force single-node DSATUR blocks — worst-case *throughput*
         (zero parallelism), NOT a mixing penalty (chromatic blocks are
         conditionally independent, so block size leaves the transition kernel
         identical to a single-site scan). EXP D (below) confirms the wall is
         frustration-driven and persists at every graph density.

      3. **Tempering probe (`experiments/probe_tempering.py`) — the barrier
         IS crossable; this is the hero figure.** Faithful pure-numpy block-Gibbs
         (exact Potts conditionals; identical algorithm to THRML on a dense graph,
         validated against the same exact enumerator) with temperature `logit/T`,
         comparing vanilla (T=1) vs geometric annealing vs **parallel tempering**
         (K=8 geometric temps to 2w, even/odd adjacent swaps, accept
         `min(1,exp((β_a−β_b)(U_b−U_a)))`) on the hard instances (n=8, w∈{8,16},
         8 instances, 512 chains). Result (`results/probe_tempering.json`):
         | w | vanilla TV | annealed TV | PT TV | PT cost |
         |---|---|---|---|---|
         | 8  | 0.369 | 0.198 | **0.006** | ~9× |
         | 16 | 0.410 | 0.312 | **0.008** | ~9× |
         PT restores marginal fidelity to the **Monte-Carlo noise floor**
         (~0.008 for 51 k samples over L=4) where single-T Gibbs is metastable;
         annealing only partially crosses. **Important correction to the
         hardness-dial Egap:** with 512 *independent* parallel chains, Egap=0 for
         ALL methods (random restarts find the MAP basin at n=8) — the 3.44-nat
         gap was specifically THRML's *single*-chain MAP failure. The
         restart-robust barrier lives in the **marginals / sampling task** (the
         one the TSU actually accelerates — partition function, latent-partition
         posterior, uncertainty), and PT is what crosses it, at a ~9× compute
         premium. That premium IS the hardware argument: native-temperature
         sampling hardware amortises the replica cost.

         **EXP D (sparsity sweep, n=8, w=16, ER density p∈{0.2…1.0}):** separates
         the two hardware-relevant axes the dense complete graph conflates.
         | p | edges | DSATUR colors | parallelism (n/colors) | vanilla TV | PT TV |
         |---|---|---|---|---|---|
         | 0.2 | 5.5 | 2.2 | 3.67 | 0.306 | 0.004 |
         | 0.5 | 13.9 | 3.4 | 2.42 | 0.349 | 0.004 |
         | 1.0 | 28.0 | 8.0 | 1.00 | 0.385 | 0.006 |
         (1) **Parallelism RISES as the graph sparsifies** (big color classes) —
         the throughput win; the complete graph is the worst case (1 node/step).
         (2) **The metastability wall PERSISTS at every density** (vanilla TV
         0.30–0.39 even at 5.5 edges) → it is a property of frustration
         *strength*, NOT of blocking; the earlier "single-node block = worse
         mixer" framing was wrong (chromatic blocks don't change the kernel).
         (3) **PT crosses it at every density** (TV 0.003–0.006, MC floor). So
         tempering is required regardless of sparsity — the two hardware wins are
         orthogonal: sparse-graph block parallelism (throughput) AND
         native-temperature replicas (barrier crossing).

         **EXP E (THRML↔numpy cross-validation + chain-strategy correction,
         n=8, w=16, p_T∝exp(U/T) temperature sweep, 8 instances).** Built to
         answer "is the numpy PT figure a faithful THRML proxy?" — and it
         corrected a framing error. THRML's `SamplingSchedule(burn,n_chains,1)`
         is ONE zero-init chain recording `n_chains` autocorrelated samples, NOT
         independent parallel chains. Separating the two strategies (mean±std TV
         vs exact over 8 instances):
         | T | w/T | H_exact | TV(thrml,ex) | TV(np-1chain,ex) | TV(np-512indep,ex) | TV(thrml,np-1chain) |
         |---|---|---|---|---|---|---|
         | 1  | 16 | 0.41 | 0.19±0.22 | 0.40±0.34 | **0.49±0.11** | 0.39 |
         | 16 | 1  | 1.37 | 0.06±0.03 | 0.05±0.01 | 0.005 | **0.069** |
         Three conclusions:
         (1) **Proxy claim CONFIRMED.** In the mixing regime (T≥16) numpy single
         chain reproduces THRML to sampling noise (TV(thrml,np)=0.069 at T=16,
         0.045 at T=32). The large low-T TV(thrml,np)=0.39 is two single chains
         trapping in *different* basins — not a kernel mismatch (both differ from
         exact by ±0.2–0.34). The numpy PT figure is a faithful extension of
         THRML's kernel; THRML simply doesn't expose replica state/swaps.
         (2) **The strong-coupling target is PEAKED, not equal-weight multimodal.**
         Exact marginal entropy at T=1 is 0.41 nats (uniform = log4 = 1.39) → the
         equilibrium concentrates on a dominant basin. "Hardness" = barrier
         trapping that stops chains reaching that basin, not averaging over many
         equal modes.
         (3) **State the wall via the ENSEMBLE, not a single chain.** The
         independent 512-chain ensemble robustly over-disperses (TV 0.49±0.11,
         low variance — chains stuck across init basins they should rarely
         occupy); this is the strategy hardware runs (many replicas) and the
         EXP-C "vanilla" baseline. A *single* chain has the same barrier but its
         TV is dominated by init luck (0.19–0.40 with std ±0.2–0.34). **The
         hardness-dial EXP B single-instance "TV=0.58 / Egap=3.44" was one
         unlucky draw from this high-variance single-chain estimator** — the
         robust metastability signal is the ensemble over-dispersion, which EXP C
         already uses as its baseline and PT robustly fixes. So the EXP-C hero
         figure stands (and is *better* motivated); only the single-chain EXP-B
         point estimates need the variance caveat.

      4. **Full-window partition-hardness probe
         (`experiments/probe_partition_hardness.py`) — closes the last open
         question: real RMC does NOT land in the hard regime.** Scout 1
         measured the wrong object (per-entity Jaccard *components*, n≤5, 7
         over-size dropped). Scout 4 measures the right one: the **joint
         latent-partition posterior over ALL n holes of a window** (n=4..10 in
         the cache), exact at the realistic L=256. Model = frustrated correlation
         clustering `log p(z)=Σ_g poolZ(g)+Σ_{i<j}β(J_ij−τ)(2[z_i=z_j]−1)`, where
         `poolZ(g)=logsumexp_t Σ_{h∈g} l_h(t)` over the intersection of the
         group's top-k cand vocab (hard within-group equality = Track A's
         hard_eq_map pooling, *summed* not *maxed*). The token marginal
         **factorises per group**, so precompute poolZ for all ≤2^n−1 hole-subsets
         once, then each Bell(n) partition is a sum over its groups — NO L^n joint
         enumeration (256^10≈1e24 hopeless → Bell(10)=115975, <1s). Engine
         validated: subset-precompute vs direct per-group max|err|=7e-15; β=0 MAP
         is all-singletons on every item (pooled-PoE structurally prefers
         splitting — merges come ONLY from the Jaccard prior); full posterior
         independent recompute max|Δp|=0, normalises to 1. Two readouts over all
         240 dev items:
         - **EXP1 census (β sweep 0/2/4/8, τ=0.3).** Real posteriors are
           **ambiguous but exactly tractable.** @β=4: mean p(MAP)=0.66, 80% have
           p(MAP)<0.9, **53% genuinely multimodal** (Hfrac>0.2); max Bell-enum
           wall **0.98s** (n=10). Gold largely **unrecoverable**: MAP==gold 0.11,
           ARI 0.34, p(gold) 0.09 (top-k support ceiling from scout 1, confirmed;
           rises only to 0.20/0.41/0.18 at β=8). So the partition posterior is a
           meaningful non-trivial object (good for motivating the latent-partition
           framing) yet computable exactly in sub-second time → **no sampler/
           hardware advantage on real RMC.**
         - **EXP2 metastability (top-8 most-ambiguous items, Hfrac 0.52–0.73,
           partition-space single-site block-Gibbs).** Even at maximal real
           ambiguity there is **no barrier**: single-T vanilla ensemble reproduces
           the exact co-clustering marginals to **TV=0.008** (MC floor); PT 0.005
           (also floor, swap 0.79–0.88). Tempering buys nothing because nothing is
           trapped. Contrast the synthetic dialed regime (scouts 2/3): single-T TV
           0.4–0.6, PT required. **ambiguity ≠ hardness** — real RMC frustration
           (weak Jaccard couplings, small n) yields posterior *uncertainty*
           without barrier-separated modes; the barrier-protected metastability
           that motivates the hardware requires the *dialed* strong frustration
           (w≥8, n≳12) that does not arise in n≤10 LM-grounded RMC graphs.
         Report: `results/probe_partition_hardness.json` (census_sweep ×4 β,
         census_rows ×240, metastability ×8). Reuses scout-1 cache + partition
         helpers; pure numpy, no GPU.

      **Reshaped Track B thesis (post-scout):** NOT "block-Gibbs beats exact"
      (false on real RMC — instances are too small/easy) but: *a frustrated,
      LM-grounded Potts model has a genuine intractable regime (n≳12, dense,
      strong coupling) where block-Gibbs is the only feasible sampler; the
      equilibrium target there is sharply peaked but barrier-protected, so an
      independent block-Gibbs replica ensemble robustly over-disperses
      (burn-in-invariant metastability wall); parallel tempering / replica
      exchange crosses it to the Monte-Carlo floor at a quantified ~9× compute
      premium that motivates native-temperature hardware.* The hero figure is the
      Pareto+metastability+tempering triptych, not an RMC accuracy number.

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
  - **Track A headline gate (verification table) PASSES on dev AND test
    (test confirmed 2026-05-29):** (best joint − argmax) SIG on all 4 cells;
    (joint − best_of_n_strong) SIG on all 4; (hard_eq_map ≥ oracle) on all 4.
    The (ψ − hard_eq_oracle) CI includes 0 on 3 cells but is **SIG-worse on
    wt_single** (−0.023 [−0.044, −0.001]) — the learned-ψ claim is now
    "ψ ≤ hard_eq (ties or loses)", not "ψ ≈ hard_eq". This does not threaten
    the Track A headline (hard equality is the shipped factor). **Track A is
    DONE; the paper's headline table is locked.**
- **Track B open questions — SCOUTED 2026-05-30 (3 probes, see Track B SCOUT
  FINDINGS block above), thesis reshaped:**
  - Can the attraction+repulsion graph recover entity partition without oracle
    labels on REAL RMC? → **Largely NO (splitability probe).** The exact
    partition posterior caps split-recovery at ~0.44 (top-k support ceiling,
    not a sampler failure); on RMC's small components (n≤5) exact is cheap, so
    sampling buys nothing on the real benchmark. RMC is the wrong place to
    showcase the sampler.
  - Is there a regime where block-Gibbs strictly dominates exact/pooling/BoN?
    → **YES, but it is the *dialed-hardness synthetic* regime, not RMC**
    (hardness-dial scout): n≳12 dense frustrated Potts with real MDLM fields —
    exact infeasible (L^n>16.8M), Gibbs flat+faithful. The hardware-justifying
    figure is the Pareto crossover + metastability wall + tempering rescue, NOT
    an RMC accuracy delta.
  - Does single-T block-Gibbs survive strong coupling? → **NO — metastability
    wall at w≥8 (TV→0.6), burn-in-invariant.** Does tempering cross it? →
    **YES (tempering probe): parallel tempering restores TV to the MC floor
    (0.41→0.008) at ~9× compute; annealing only partially.** The ~9× premium is
    itself the argument for native-temperature sampling hardware.
  - Does the wall survive on a *sparse* frustrated graph? → **YES (tempering
    EXP D, n=8 w=16, density sweep).** The metastability wall persists at EVERY
    density (vanilla TV 0.30–0.39 even at 5.5 edges); PT stays at the MC floor
    (0.003–0.006) throughout. Block size is **parallelism, not mixing** —
    chromatic blocks are conditionally independent, so sparsity raises THRML
    throughput (parallelism n/colors 1.0→3.67) WITHOUT changing the transition
    kernel. The two hardware wins are orthogonal: sparse-graph block parallelism
    (throughput) AND native-temperature replicas (barrier crossing).
  - Does the FULL-WINDOW real RMC partition posterior (not the per-entity
    components scout 1 measured) land in the hard regime? → **NO (scout 4,
    `probe_partition_hardness.py`).** Exact partition posterior over all n holes
    (n≤10), full L=256, via pooled-PoE per-group factorisation (Bell(10)=116k,
    <1s; the naive L^n=256^10≈1e24 never enumerated). Real posteriors are
    *ambiguous* (53% multimodal, p_MAP~0.66 @β=4) but exactly solvable in
    sub-second time, AND single-T partition-Gibbs mixes to the MC floor (TV
    0.008) even on the most ambiguous items — **no barrier, ambiguity≠hardness.**
    Gold grouping largely unrecoverable (MAP==gold 0.11 @β=4 — top-k ceiling).
    The hard regime (barrier-protected metastability needing PT) is reached only
    by *dialing* synthetic frustration; no real cache graph reaches it. **The
    open "is there a real hard task?" question is CLOSED with a measured negative
    → the hero figure is legitimately synthetic.**
  - REMAINING: (a) implement PT inside THRML or accept the numpy reference
    sampler for the paper figure? [(b) "is there a non-RMC benchmark in the hard
    regime?" — superseded: scout 4 closed it for RMC; finding such a task is now
    optional future work, not a blocker for the synthetic hero figure.]
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
