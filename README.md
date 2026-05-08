# diffusion-ebm

Hybridising a pretrained masked-diffusion language model with a THRML
block-Gibbs joint sampler over a sparse factor graph, to test whether
joint sampling improves multi-hole consistency at a fixed neural-FLOPs
budget.

## What this is

The research question:

> Given a masked diffusion LM that produces per-position categorical
> logits, can a THRML block-Gibbs sampler over a sparse factor graph
> improve **multi-hole consistency** at fixed neural-FLOPs budget?

We compare four strategies for filling multiple `[MASK]` holes that are
constrained to take the same vocab id:

- **`thrml_joint`** — one MDLM forward → top-k=64 candidate sets per
  hole → THRML block-Gibbs with hard equality factors over the matched
  positions.
- **`mask_predict`** — MDLM's own iterative refinement
  (Ghazvininejad et al., 2019), at temperature 0 (deterministic) and 1
  (stochastic), with `n_iters` ∈ {1, 4, 16, 64}.
- **`ancestral_topk`** — one MDLM forward, then independent
  multinomial draws from the top-k softmax. Same state space as
  `thrml_joint`; no joint signal.
- **`independent_full`** — one MDLM forward, full-vocab independent
  multinomial.

Templates (`src/diffusion_ebm/tasks/multihole.py`):

- *color*: `Alice's favorite color is [M]. Bob's favorite color is also [M].`
- *variable*: `The variable [M] was assigned the value 7. Later, [M] was used in a loop.`
- *repeat-3*: `My name is [M]. You can call me [M]. I said [M] three times.`

## Result

![headline](plots/headline.png)

At a single LM forward pass, THRML block-Gibbs reaches average
agreement **0.84 ± ~0.1** across the three templates — the error bar
spans the equality-weight × burn-in × k axes — while every iterative
mask-predict configuration up to **64 LM forwards** plateaus at 0.33
(deterministic) and 0.15 (stochastic). The independent baselines
(`ancestral_topk`, `independent_full`) sit at 0 across the board.

This is the headline the project was set up to test. The result holds
under the **TSU cost-model framing**: a neuromorphic accelerator running
factor-graph Gibbs at amortised cost
`C_G ≪ C_N` (the per-LM-forward cost) makes every point on the THRML
budget curve below cost-equivalent to a single LM forward.

![tsu_cost](plots/tsu_cost.png)

The full per-LM-forward Pareto, broken out by template:

![pareto_flops](plots/pareto_flops.png)

### Per-template results

| template | thrml_joint (best) | mask_predict (best) | ancestral_topk (best) | independent_full |
|---|---|---|---|---|
| color    | **1.000** @ 1 LM, 74 Gibbs | **1.000** @ 4 LM (T=0)  | 0.000 | 0.000 |
| variable | **1.000** @ 1 LM, 74 Gibbs | **0.031** @ 1 LM (T=1)  | 0.016 | 0.000 |
| repeat-3 | **1.000** @ 1 LM, 74 Gibbs | **0.016** @ 4 LM (T=1)  | 0.000 | 0.000 |

THRML reaches **perfect agreement** on all three templates at
`equality_weight = 10`, `gibbs_sweeps = 10` (i.e. 74 total Gibbs sweeps,
1 LM forward). Mask-predict at temperature 0 happens to tie on the
*color* template at `n_iters = 4`: the deterministic argmax at hole 1
becomes context for hole 2's argmax, and the model picks the same
token. On the *variable* and *repeat-3* templates, no number of
mask-predict iterations recovers consistent fills — the model's
per-hole modes are different even when the previous hole has been
committed. Joint Gibbs over the equality factor cuts through this
without re-querying the LM.

The mean LM perplexity of the filled sequences stays in the
**1.00–1.20** range across all winning configurations — agreement is
not bought with garbage tokens.

## Reproduce

This repo has NixOS-specific env quirks (CUDA driver path,
flash-attn-from-source, transformers pin). They are documented in
`CLAUDE.md`. The short path:

```bash
nix develop                 # FHS shell with nvcc + LD_LIBRARY_PATH set
uv sync                     # ~10 min on first run (flash-attn build)

# M0: env smoke check
uv run python notebooks/00_smoke.py

# M1: synthetic Potts chain — THRML correctness sanity
uv run python notebooks/01_mvp0_potts.py

# M2: MDLM forward + mask-predict bridge
uv run python notebooks/02_mdlm_bridge.py

# M3: multi-hole agreement headline
uv run python notebooks/03_mvp1_multihole.py

# M4: full Pareto sweep (~2 min on RTX 3000 Ada)
uv run python experiments/run_pareto.py
uv run python notebooks/04_pareto.py    # writes plots/*.png + summary
```

If you are running outside the dev shell, prepend the CUDA driver path:

```bash
LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH" \
TRITON_LIBCUDA_PATH="/run/opengl-driver/lib" \
    uv run python experiments/run_pareto.py
```

The sweep driver supports `--quick` (one config per method, 12 runs in
under a minute) and writes per-run records to `results/results.json`.
The plotting script is purely a function of that JSON; tweak it without
re-running the GPU sweep.

## What's not in scope

The master plan
(`/home/ghawk/.claude/plans/nested-wishing-thimble.md`) deliberately
narrows the M4 grid relative to the original proposal. Three things
that *would* strengthen the comparison are explicitly deferred:

- **Multi-step `ancestral_topk`** (T > 1). Implementing MDLM's native
  iterative diffusion sampler restricted to the top-k state space is
  its own ~half-day of work and is effectively a stricter
  mask-predict; the deterministic mask-predict at T=0 is a reasonable
  upper bound on what that baseline could achieve.
- **Multi-step THRML** (one Gibbs round per partial unmask, with new
  LM forwards re-evaluating the unary terms after each unmask). The
  headline claim — "joint sampling beats independent at fixed LM
  budget" — does not require this. We are not testing
  "diffusion + THRML beats diffusion + mask-predict end-to-end".
- **Learned EBM corrections.** The project's stretch goal:
  cross-position semantic agreement signal that isn't a hard equality
  constraint. The plan tagged it as out of M4 scope; we'd need a
  separately-trained sequence-level scorer.

Two further caveats worth flagging:

- All three templates encode the *same* type of constraint — exact
  vocab-id equality. Real "consistency" failures in modern LMs are
  softer (coreference, shared variable names, arithmetic). The
  equality-factor formulation is a clean test bed but it doesn't yet
  exercise the harder cases.
- The TSU cost model is a *post-hoc framing*: the JAX-on-GPU sampler
  used here pays one Gibbs sweep ≈ one CUDA kernel launch. The
  hardware claim `C_G ≪ C_N` belongs to the THRML/Extropic
  literature; this repo only verifies that joint sampling produces
  the agreement signal we want at the algorithmic level.

## Acknowledgements

- THRML and the underlying TSU cost model:
  [extropic-ai/thrml](https://github.com/extropic-ai/thrml).
- MDLM checkpoint: `kuleshov-group/mdlm-owt`
  ([Sahoo et al., 2024](https://arxiv.org/abs/2406.07524)).
- Mask-Predict baseline: Ghazvininejad et al., *Mask-Predict:
  Parallel Decoding of Conditional Masked Language Models*, 2019.
