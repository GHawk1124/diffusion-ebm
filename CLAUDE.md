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
    probe_hw_common.py         # Classical-gauntlet shared harness (pure numpy):
                               #   Factor/FactorGraph/potts_pairwise, exact_marginals
                               #   (brute+co-cluster), variable_elimination_marginals,
                               #   block_gibbs, parallel_tempering, gauntlet_ais
                               #   (now returns log_z), gauntlet_mean_field/_trw_bp/
                               #   _best_of_n, mean_hellinger/co_tv, UAI .MAR IO.
    probe_factor_graph_inference.py # Probe D: loopy-BP + full gauntlet on real UAI
                               #   MAR grids, exact-VE gold, honest scorecard.
    probe_gauntlet_dial.py     # Gauntlet across the synthetic hardness dial
                               #   (AIS DOMINATES PT on marginals at every scale).
    probe_ais_vs_pt.py         # AIS-vs-PT first-order shot: EXP P (sweep w) /
                               #   EXP S (forced supercooling, schedule resolution) /
                               #   EXP M (matched compute). All negative → principled
                               #   first-order shot CLOSED. report probe_ais_vs_pt.json
    probe_logz.py              # Route-2 logZ/free-energy probe: AIS logẐ vs PT
                               #   thermodynamic-integration vs exact logZ. EXP ZF
                               #   (ferro schedule sweep: 1.46-nat logẐ bias @5 temps
                               #   while overlap=1.000, recovers @400 temps) + EXP ZG
                               #   (spin-glass no-escape NEGATIVE: AIS logẐ survives
                               #   ESS collapse ≤0.087 nats, PT-TI worse). Route 2
                               #   CLOSED. report results/probe_logz.json
    probe_first_order.py       # Route-3 (the last shot): engineered no-escape
                               #   first-order instance. Two EXTENDED ASYMMETRIC
                               #   basins (deep-narrow 3-body A vs wide-shallow 2-body
                               #   B = density-of-states competition, dodges ferro MF
                               #   escape). EXP FO-A/B single-basin funnel control;
                               #   FO-C/D two-basin double-well. VERDICT: genuine
                               #   first-order well BUILT (single-T Gibbs trapped, gap
                               #   0.137) but AIS crosses on every observable (basin-
                               #   occ err 0.000 @5→800 temps); AIS anneals from β=0
                               #   where wells haven't formed → never crosses barrier.
                               #   Route 3 CLOSED → only framing (i) survives. report
                               #   results/probe_first_order.json
    probe_planted.py           # Large-n PLANTED route (the last un-closed door):
                               #   frustrated Potts glass + planted reward on the
                               #   (t*_i,t*_j) edge entry → plant-overlap is ground
                               #   truth where exact Z is GONE. Triangulated validation
                               #   (soft/hard overlap, clamped-at-plant stability ctrl,
                               #   score vs score(t*), full gauntlet). VERDICT: sparse =
                               #   continuous transition, NO barrier; dense (n=48,q=5,
                               #   deg=12,frust=5; 5^48≈3.5e33) = genuine large-n
                               #   metastability wall (single-T Gibbs trapped, burn-in-
                               #   invariant, overlap gap ~0.06-0.12) but PT AND AIS both
                               #   cross to ≤0.024 → NO PT-unique window. Obstruction
                               #   holds at large n; cleanest framing-(i) confirmation.
                               #   report results/probe_planted.json (+_frust4/_dense_hunt)

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

## Status (2026-05-31, M5d Track A test-split LOCKED — joint factor-graph decoding beats independent argmax by +7–15 pp on the held-out test split (all 4 cells SIG); hard_eq_map (closed-form pooling) DOMINATES the sampler; learned ψ ≤ hard equality (SIG-worse on wt_single). Track A is DONE. Track B SCOUTED via 4 fail-fast probes — RMC is too easy for the sampler (splitability ceiling 0.44; scout-4 full-window partition posterior is ambiguous but exact-in-<1s AND mixes to the MC floor TV=0.008 → no barrier, ambiguity≠hardness, real-hard-task question CLOSED negative), BUT a dialed-hardness frustrated Potts has a genuine regime (n≳12) where exact dies + single-T Gibbs hits a burn-in-invariant metastability wall (w≥8) + parallel tempering crosses it (TV 0.41→0.008 at ~9×). Thesis reshaped: hero fig = Pareto+metastability+tempering, legitimately SYNTHETIC, NOT RMC accuracy. See Track B SCOUT FINDINGS block + plan file. **CAVEAT (2026-05-31, gauntlet+AIS): the metastability "wall" is only a wall vs the WEAK single-T-Gibbs incumbent — classical AIS (annealed importance sampling, single-machine) crosses it cheaply and DOMINATES PT on marginal inference at every scale with exact ground truth, including AIS's textbook first-order failure mode (probe_ais_vs_pt.py, three exps all negative). All FOUR escape routes are now CLOSED: marginal inference (probe_ais_vs_pt), logZ/free-energy (probe_logz — AIS logẐ survives ESS collapse on the no-escape spin glass), the engineered first-order double-well (probe_first_order — genuine barrier built, single-T Gibbs trapped, but AIS crosses on every observable), and the large-n PLANTED glass (probe_planted — at n=48,q=5,dense, 5^48 states/exact-gone, a real burn-in-invariant metastability wall traps single-T Gibbs but PT AND AIS both cross to ≤0.024 plant-overlap; the obstruction holds at the only scale where it could have broken). The "no classical escape" claim is RETIRED; only framing (i) survives — "PT vs single-T block-Gibbs at matched temperature count," with AIS acknowledged as a stronger classical baseline that also crosses the wall. See GAUNTLET + AIS-vs-PT FINDINGS block (items 1–6).**)

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

      **GAUNTLET + AIS-vs-PT FINDINGS (2026-05-31) — the "no classical escape"
      claim does NOT hold on marginal inference; the principled first-order shot
      is CLOSED.** Built the full classical gauntlet into a shared pure-numpy
      harness and ran it against both real published instances and the synthetic
      dial to test whether ANY validatable regime exists where the *whole*
      classical toolbox fails while PT/replica-exchange uniquely recovers
      marginals. It does not. New files (all pure numpy, run on system python, no
      GPU): `experiments/probe_hw_common.py` (shared harness: `Factor`/
      `FactorGraph`/`potts_pairwise`, `exact_marginals` brute+co-cluster,
      `variable_elimination_marginals` min-fill, `block_gibbs`,
      `parallel_tempering`, and the gauntlet `gauntlet_ais` / `gauntlet_mean_field`
      / `gauntlet_trw_bp` / `gauntlet_best_of_n`, plus `mean_hellinger` / `co_tv`
      / UAI `.MAR` IO); `experiments/probe_factor_graph_inference.py` (Probe D:
      loopy BP + full gauntlet on real UAI MAR instances, exact-VE gold, honest
      scorecard); `experiments/probe_gauntlet_dial.py` (gauntlet across the
      synthetic hardness dial); `experiments/probe_ais_vs_pt.py` (the
      first-order + matched-compute + schedule-resolution probe). Findings:

      1. **UAI MAR grids (Grids_12–14) — honest FAIL on "no escape" (G1b/G5).**
         Junction-tree VE solves all three EXACTLY (induced width 13–23). The
         gauntlet on these: loopy-BP 0.24–0.57, TRW-BP 0.22–0.46, mean-field
         0.45–0.53, **AIS 0.06–0.11** (strongest classical sampler but still
         >0.05 fail), best-of-N 0.35–0.63, single-T Gibbs 0.31–0.43, PT
         0.005–0.083 (Hellinger to exact). PT wins but the instance is *tractable*
         (VE exact) → no hardware argument. Verdict: TRACTABLE REGIME.

      2. **Synthetic dial (random-MDLM-field frustrated Potts) — AIS DOMINATES PT.**
         At n=10 sweeping w 0→16, AIS Hellinger stays 0.014–0.031 (PASSING < 0.05)
         while BP/TRW/MF/best-N fail at strong coupling and single-T Gibbs rises
         0.004→0.45. AIS at n=10,w=16 = 0.014 in 0.57 s vs PT 0.004 in 11 s — PT
         is more accurate but uses ~14× compute merely to tie a passing AIS. AIS
         ESS collapses (0.94→0.17 with w) so it *feels* hardness, but Hellinger
         stays low because the random-field target is **peaked** (one dominant
         basin). Field-scale sweep confirms robustness: at fs=0 (symmetric,
         multimodal) AIS still passes 0.019–0.038 while TRW=MF=0.000
         (variationally trivial — exact marginal uniform); at fs=0.25 AIS passes
         while BP/MF/best-N/Gibbs fail. AIS is unbreakable across the whole
         (field × coupling) plane at every n with exact ground truth.

      3. **The principled first-order shot (`probe_ais_vs_pt.py`) — CLOSED, three
         experiments, all negative.** AIS's one textbook failure mode is a
         *first-order phase transition* (chains supercool past a free-energy
         barrier → biased, not just noisy; PT tunnels). Tested on the cleanest
         exactly-solvable first-order system, the fully-connected q=4 ferromagnetic
         Potts (CAVEAT: it has a mean-field solver escape, so it demonstrates the
         MECHANISM with clean ground truth, it is NOT itself a no-escape instance).
         - **EXP P** (sweep w 0.5→8 at n=10, 400-temp ladder): AIS tracks exact
           *overlap AND marginals* at every coupling (overlap err <0.01 even at
           w=8 where exact overlap 0.998). ESS collapses to 0.71 but accuracy
           holds. The flat H_AIS≈0.036 vs PT's 0.003 is a sample-count floor (AIS
           = n_chains endpoint samples; PT = n_chains×n_measure), not bias.
         - **EXP S** (forced supercooling: fix w=6, exact overlap 0.985, throttle
           AIS to 5→400 temps): even at **5 temps** (ESS 0.019) AIS overlap stays
           0.982 — NO disordered-phase trap. The only degradation is per-node
           marginal Hellinger (0.244→0.037 as temps rise) = ESS-collapse
           symmetry-sector noise. Overlap is robust because all 4 ordered Potts
           sectors give overlap≈1, so a fast-quenched chain in the "wrong" sector
           still has correct overlap. At matched compute AIS is competitive-or-
           better than PT except at trivially tiny budgets where BOTH fail.
         - **EXP M** (matched site-sweep budget, w=3): PT's small edge (H 0.002–
           0.011 vs AIS 0.029–0.040) is sample-throughput on a barrier-free target
           (PT harvests n_measure× more β=1 samples; AIS gets a fine 800–12800-temp
           ladder, ESS 0.99), NOT replica-exchange tunneling. No cost-justified PT
           advantage. Report: `results/probe_ais_vs_pt.json`.

      4. **The logZ / free-energy pivot (`probe_logz.py`, route 2) — CLOSED on
         the no-escape instance; only the escape-instance mechanism survives.**
         AIS's importance weights also estimate logZ (logẐ = logZ_0 +
         logsumexp(logw) − log N, unbiased in Ẑ → biased LOW in logẐ by Jensen as
         ESS→0), so logZ is a *more sensitive* observable than the marginals. The
         decisive test: a cell where AIS marginals PASS (<0.05 H) but AIS logẐ is
         badly biased while PT-thermodynamic-integration (logZ = logZ_0 +
         ∫_0^1⟨U⟩_β dβ, replica exchange per rung) recovers exact logZ. Validated
         PT-TI vs exact to 0.002–0.003 nats. Two experiments (n=10, q=4, exact
         logZ gold):
         - **EXP ZF (ferro, schedule sweep, w=8):** the *genuinely new* finding —
           logZ is dramatically more schedule-sensitive than the order parameter.
           At 5 temps AIS logẐ is off by **1.46 nats** (0.15/site, ~4.3× in Z;
           ESS 0.025) **while overlap = 1.000 (perfect)**. PT-TI = 0.002 nats. So
           logZ exposes a failure the overlap completely hides. BUT well-resolved
           AIS recovers (400 temps → 0.029 nats) → it's an ESS/discretization
           Jensen bias fixed by *adding temps*, NOT a barrier PT *uniquely*
           crosses; and the ferro has the MF/sector-sum escape.
         - **EXP ZG (frustrated spin glass, the no-clean-escape instance, w up to
           16):** the decisive NEGATIVE. AIS marginals pass everywhere (H
           0.028–0.045) AND AIS logẐ *survives* ESS collapse (|Δ|≤0.087 nats even
           at w=16, ESS 0.18). Worse for the thesis, **PT-TI is *worse* than AIS at
           strong coupling** (|Δ| 0.403 vs 0.087 at w=16). logZ does NOT expose
           hardness the marginals hide on a no-escape instance, and PT is not
           uniquely better. Report: `results/probe_logz.json`.
         **Unifying obstruction (now crisp):** AIS's logZ failure needs a
         *first-order coexistence barrier* on the annealing path. The only small-n
         systems with that AND exact gold are mean-field-solvable (ferro → escape);
         the no-escape frustrated systems we can build are *peaked* (spin-glass,
         continuous transition) → no coexistence barrier → AIS survives on EVERY
         observable. The negative extends from marginals to logZ.

      5. **The engineered-first-order route (`probe_first_order.py`, route iii) —
         CLOSED; the obstruction confirmed empirically.** Route (iii) was the last
         shot: *explicitly construct* a no-escape first-order instance (not a
         collective spin-ordering transition, so no ferro MF/gauge escape) and check
         whether AIS supercools past the barrier while PT crosses, with exact gold at
         n≤9. The textbook AIS-killer is a **density-of-states / entropy-energy
         competition**: a DEEP+NARROW basin (low entropy) versus a WIDE+SHALLOW basin
         (high entropy), so AIS over-commits during annealing to whichever basin
         captures chains first. Construction = frustrated planted Potts (n=8, q=4):
         basin A = dense 3-body clauses toward t_A (cubic-steep mouth, depth jA);
         basin B = 2-body funnel toward an orthogonal t_B (gentle quadratic mouth,
         weight jB); random fields + random clause subsets break the match-count
         sufficient statistic → no 1-D/MF reduction. Order parameter = basin
         occupancy P_A / P_B (asymmetric → no sector-symmetry robustness). Two
         experiments (EXP FO-A/B single-basin funnel control; **FO-C/D two-basin
         double-well**), 3 instances, AIS to 800 temps, exact + PT-TI gold.
         - **We DID build a genuine first-order double-well.** At jA=1.4 there is real
           coexistence (exact P_A=0.602 / P_B=0.398, P_none≈0 → two deep wells with a
           barrier between them) and **single-T Gibbs is provably trapped**
           (Gibbs P_A=0.465 vs exact 0.602, gap 0.137; across the sweep Gibbs stays
           "sticky" near its ~0.5 init while true equilibrium swings 0.05→1.0). The
           barrier is real.
         - **AIS crosses it anyway, on every observable.** AIS P_A=0.573 (vs exact
           0.602) at jA=1.4; AIS tracks the full equilibrium swing to ≤0.03 at every
           jA; PT tracks to ≤0.01. In the schedule discriminator (FO-D) AIS basin
           occupancy error is **0.000 at every schedule** (5→800 temps); the *only*
           schedule-sensitive observable is logẐ (1.076 nats at 5 temps) which
           *recovers* to 0.008 at 800 temps — the same recoverable Jensen/ESS bias as
           EXP ZF, NOT a PT-unique barrier. Verdicts: obstruction holds (FO-A/B/C/D).
         - **The deep reason, now empirically confirmed:** AIS anneals from β=0 where
           the energy scale is washed out and the two wells have NOT formed; it
           distributes chains on a flat landscape and the wells crystallize *around*
           the already-placed chains. AIS never has to *cross* the barrier. Only a
           single-T sampler (starting at β=1 with the barrier already present) gets
           trapped. So a barrier that defeats naive Gibbs does NOT defeat AIS — and
           this holds for the most explicitly-engineered first-order landscape we can
           build at exact-gold scale. **There is no no-escape first-order instance at
           small exact-gold n. Route (iii) is CLOSED.** Report:
           `results/probe_first_order.json`.

      6. **The large-n PLANTED route (`probe_planted.py`) — CLOSED; the strongest
         large-n version of the obstruction, and the cleanest framing-(i)
         confirmation.** The one un-closed door after routes (i)–(iii): every prior
         test had exact gold only at n≤10, where first-order barriers are too weak to
         bias AIS. Go LARGE-n (exact Z gone) and replace exact gold with a PLANTED
         signal. Construction = frustrated Potts glass: random N(0,frust²) coupling
         tables on a graph + a planted reward `signal` added to the (t*_i,t*_j) entry
         of every edge (t* = fixed random colouring), which breaks the q!
         colour-permutation symmetry so **overlap with t* is an unambiguous recovery
         handle without needing Z**. Triangulated validation: soft/hard plant overlap;
         a **clamped-at-plant control** (init at t*, run β=1 Gibbs — overlap retention
         certifies t* is a deep stable state, so failure-to-recover is a barrier not an
         unstable plant); score vs score(t*); full gauntlet (single-T Gibbs / AIS / PT /
         mean-field / TRW-BP / best-of-N). The hoped-for win: in the glassy hard phase
         AIS supercools into decoys while PT recovers t*.
         - **Sparse/weak regimes give a CONTINUOUS transition → no barrier at all.**
           n=64, deg 3–4, frust 1–4: planting-on-every-edge turns recovery on smoothly;
           single-T Gibbs ≈ PT ≈ AIS ≈ clamp_ov at every signal (PT−AIS ≤0.006). Easy
           problems are easy — uninformative for the AIS-vs-PT question. (The barrier
           that traps local samplers needs a *dense* frustrated graph; sparse graphs are
           locally tree-like.)
         - **Dense strong regime gives a genuine large-n metastability wall — and AIS
           crosses it as well as PT.** n=48, q=5, avg_deg=12, frust=5 (state space
           5⁴⁸≈3.5×10³³, exact hopeless). At signal∈[1.5,3.0], **single-T Gibbs is
           provably trapped** below the clamp-certified stable level (gibbs plant-overlap
           0.295/0.340/0.406/0.495 vs clamp_ov 0.415/0.468/0.521/0.613 — gap ~0.06–0.12),
           and the gap is **burn-in-invariant** (unchanged from burn=400 hunt to burn=600
           confirmation) → a real barrier, the scout-3 wall reproduced at large n with
           plant ground truth. **Both PT and AIS cross it, tracking each other to ≤0.024**
           plant-overlap (matched compute, AIS 400 temps × 8 sweeps vs PT 8 levels;
           pt_so 0.355/0.384/0.470/0.544 vs ais_so 0.336/0.386/0.470/0.544). AIS sometimes
           slightly *above* PT. No supercooling: AIS recovers the plant wherever PT does.
           (t* is only partially recovered — strong glass — but the *relative* PT≈AIS≫Gibbs
           comparison is decisive; mean-field/TRW-BP/best-of-N all sit below Gibbs.)
         - **Verdict: NO PT-unique window; obstruction holds at large n.** This is the
           definitive large-n confirmation of framing (i): a genuine metastability wall
           that traps the weak single-T incumbent, at n where exact ground truth is gone,
           is crossed *equally* by PT and by single-machine AIS. The deep reason from
           route (iii) holds at scale — AIS anneals from β=0 where the glass hasn't frozen,
           so it never crosses the barrier; only single-T Gibbs (β=1 start) traps. Report:
           `results/probe_planted.json` (+ `_frust4.json` continuous-transition negative,
           `_dense_hunt.json` barrier discovery).

      **CONSEQUENCE for Track B (honest, load-bearing).** Track B's existing hero
      figure (`probe_tempering.py`) shows PT crossing a metastability wall — but
      the incumbent it beats is **single-T block-Gibbs ensemble over-dispersion**,
      a WEAK classical baseline. AIS (also classical, single-machine, no special
      hardware) crosses that same wall cheaply on every instance with exact ground
      truth. So the marginal-inference route to "PT/TSU uniquely wins where the
      whole classical gauntlet fails" is **closed at every validatable scale**,
      including AIS's principled first-order failure mode: the only systems where
      we can compute exact marginals (n≤10) have first-order barriers too weak to
      bias a properly-tempered AIS, and the order parameter is robust to schedule
      under-resolution. The only place PT could uniquely win is large-n where
      exact ground truth is gone (unfalsifiable — the trap to avoid). The logZ /
      free-energy pivot that *could* have rescued this was TESTED (item 4,
      `probe_logz.py`) and is **CLOSED**: logẐ is indeed dramatically more
      schedule-sensitive than the order parameter (EXP ZF: 1.46-nat bias at 5
      temps while overlap=1.000), but on the no-escape spin glass AIS logẐ
      *survives* ESS collapse (≤0.087 nats) and PT-TI is actually *worse* at
      strong coupling — the ESS-collapse prediction failed exactly where a
      first-order barrier was needed. The third option, (iii) — *explicitly
      engineering* a no-escape first-order instance (a deep-narrow vs wide-shallow
      density-of-states competition, dodging the ferro's MF/gauge escape) — was the
      last shot and was TESTED (item 5, `probe_first_order.py`) and is **CLOSED**: we
      DID build a genuine first-order double-well (single-T Gibbs provably trapped,
      gap 0.137 at the coexistence cell) but AIS crosses it on every observable
      (basin occupancy error 0.000 at every schedule 5→800 temps; only logẐ is
      schedule-sensitive and it recovers with temps). The deep reason is now
      empirically confirmed: AIS anneals from β=0 where the wells haven't formed, so
      it places chains on a flat landscape and the wells crystallize around them — it
      never crosses the barrier, only single-T Gibbs (starting at β=1) does. So **all
      three resolution options are now gone; only framing (i) survives.** The one
      remaining hope — that the obstruction was an artefact of small exact-gold n and
      a genuine large-n glassy hard phase would finally trap AIS — was TESTED (item 6,
      `probe_planted.py`, large-n planted glass with plant-overlap as ground truth) and
      is **CLOSED**: at n=48, q=5, dense (5⁴⁸ states, exact gone) we DID build a real
      large-n metastability wall (single-T Gibbs trapped, burn-in-invariant, overlap gap
      ~0.06–0.12 below the clamp-certified-stable level) but **both PT and AIS cross it
      to within ≤0.024 plant-overlap** — the same obstruction, now confirmed at the only
      scale where it could have broken. The honest
      Track-B synthetic figure must be framed as **"(i) PT vs single-T block-Gibbs at
      matched temperature count"** — a real metastability wall that defeats the naive
      single-T ensemble — while explicitly acknowledging AIS as a stronger classical
      single-machine baseline that ALSO crosses the wall on the marginal, the logZ,
      AND the basin-occupancy task. The "no classical escape" claim is retired; the
      defensible claim is the weaker, true one (PT > single-T Gibbs, and the ~9×
      replica premium motivates native-temperature hardware), and the double-well
      where Gibbs is trapped but AIS+PT cross is the *cleanest illustration* of why
      that incumbent is the honest baseline to beat.

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
  - Is the synthetic metastability wall a wall vs the FULL classical gauntlet,
    or only vs single-T Gibbs? → **Only vs single-T Gibbs (gauntlet+AIS step 1,
    2026-05-31).** Classical AIS crosses the wall cheaply and DOMINATES PT on
    marginal inference at every validatable scale; AIS's principled first-order
    failure mode does NOT materialise at n≤10 where exact ground truth exists
    (probe_ais_vs_pt.py EXP P/S/M, all negative). See the GAUNTLET + AIS-vs-PT
    FINDINGS block. This is the binding threat to the marginal-inference hero
    figure and must be addressed before the synthetic figure can claim "no
    classical escape."
  - Is there a genuinely no-escape FIRST-ORDER instance AIS supercools past
    while PT crosses? → **NO (route iii, `probe_first_order.py`, 2026-05-31).**
    Explicitly engineered a deep-narrow vs wide-shallow two-basin Potts (a
    density-of-states competition that dodges the ferro's MF escape). We DID get a
    genuine first-order double-well — single-T Gibbs provably trapped (P_A 0.465 vs
    exact 0.602, gap 0.137 at the coexistence cell) — but **AIS crosses it on every
    observable** (basin-occupancy error 0.000 at every schedule 5→800 temps; only
    logẐ is schedule-sensitive and recovers with temps). AIS anneals from β=0 where
    the wells haven't formed, so it never crosses the barrier; only single-T Gibbs
    (β=1 start) traps. Route (iii) CLOSED. See GAUNTLET + AIS-vs-PT FINDINGS item 5.
  - Was the obstruction just a small-n (n≤10 exact-gold) artefact — does a genuine
    LARGE-n glassy hard phase finally trap AIS? → **NO (large-n planted route,
    `probe_planted.py`, 2026-06-01).** Replaced exact gold with a planted signal
    (overlap with t* as ground truth where Z is gone). Sparse planting → continuous
    transition, no barrier. Dense (n=48, q=5, deg=12, frust=5; 5^48≈3.5e33 states)
    → a **real large-n metastability wall** (single-T Gibbs trapped, burn-in-
    invariant, plant-overlap gap ~0.06–0.12 below the clamp-certified-stable level)
    but **PT and AIS both cross it to ≤0.024** plant-overlap (AIS matched-compute,
    400 temps). The obstruction holds at the only scale where it could have broken.
    See GAUNTLET + AIS-vs-PT FINDINGS item 6.
  - REMAINING: (a) implement PT inside THRML or accept the numpy reference
    sampler for the paper figure? (b) DECIDE the Track-B framing — **RESOLVED by
    elimination: only (i) survives.** Of the three options: ~~(ii) pivot to
    partition-function / free-energy estimation~~ — **CLOSED 2026-05-31
    (probe_logz.py): on the no-escape spin glass AIS logẐ survives ESS collapse
    (≤0.087 nats) and PT-TI is *worse* at strong coupling**; ~~(iii) construct a
    genuinely no-escape first-order instance~~ — **CLOSED 2026-05-31
    (probe_first_order.py): the engineered double-well has a real barrier (Gibbs
    trapped) but AIS crosses it at every schedule**; leaving only **(i) honestly
    narrow the claim to "PT vs single-T block-Gibbs at matched temperature count"**
    — the shippable framing, with AIS acknowledged as a stronger classical
    single-machine baseline that also crosses the wall. The "no classical escape"
    claim is retired. **Net: (i) is THE framing; the double-well (Gibbs trapped,
    AIS+PT cross) is its cleanest small-n illustration, and the large-n planted glass
    (`probe_planted.py`, item 6 — Gibbs trapped at n=48/5^48-states, AIS+PT cross) is
    its large-n confirmation that the obstruction is not a small-n exact-gold
    artefact.**
    [(c) "is there a non-RMC benchmark in the hard regime?" — superseded by
    scout 4 for RMC; optional future work.]
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
- Classical-gauntlet probes (pure numpy, system python, no GPU):
  - `experiments/probe_hw_common.py` — shared harness (factor graph, exact,
    VE, block-Gibbs, PT, gauntlet AIS/MF/TRW-BP/best-of-N, UAI `.MAR` IO).
  - `experiments/probe_factor_graph_inference.py` — Probe D, real UAI MAR
    grids + gauntlet + honest scorecard; gold cache via `--gold-cache`.
  - `experiments/probe_gauntlet_dial.py` — gauntlet across the synthetic dial.
  - `experiments/probe_ais_vs_pt.py` — first-order (EXP P) + schedule-resolution
    (EXP S) + matched-compute (EXP M) AIS-vs-PT; report
    `results/probe_ais_vs_pt.json`. See GAUNTLET + AIS-vs-PT FINDINGS block.
  - `experiments/probe_logz.py` — route-2 logZ / free-energy probe: AIS logẐ vs
    PT thermodynamic-integration vs exact logZ. EXP ZF (ferro schedule sweep:
    logẐ schedule-sensitivity the order parameter hides) + EXP ZG (spin-glass
    no-escape negative); report `results/probe_logz.json`. Route 2 CLOSED.
  - `experiments/probe_first_order.py` — route-3 (last shot): engineered no-escape
    first-order instance via a deep-narrow vs wide-shallow two-basin Potts (EXP
    FO-A/B single-basin funnel control; FO-C/D two-basin double-well). Genuine
    first-order well built (single-T Gibbs trapped) but AIS crosses on every
    observable; report `results/probe_first_order.json`. Route 3 CLOSED → only
    framing (i) survives.
  - `experiments/probe_planted.py` — large-n PLANTED route (last un-closed door):
    frustrated Potts glass + planted reward on each (t*_i,t*_j) edge entry →
    plant-overlap is ground truth where exact Z is gone. Triangulated validation
    (soft/hard overlap, clamped-at-plant stability control, score vs score(t*),
    full gauntlet). Sparse = continuous transition / no barrier; dense (n=48,q=5,
    deg=12,frust=5) = real large-n metastability wall (single-T Gibbs trapped,
    burn-in-invariant) but PT AND AIS both cross to ≤0.024 plant-overlap. Report
    `results/probe_planted.json` (+ `_frust4.json`, `_dense_hunt.json`). Obstruction
    holds at large n; framing (i) confirmed. See GAUNTLET + AIS-vs-PT FINDINGS item 6.
