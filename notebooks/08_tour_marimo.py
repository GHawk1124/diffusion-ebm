# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23",
# ]
# ///
"""Interactive tour of the diffusion-ebm project.

Run interactively (browser opens automatically):

    LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH" \\
        uv run marimo edit notebooks/08_tour_marimo.py

Or via the flake app:

    nix run .#tour
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import os

    return json, mo, os


@app.cell
def intro(mo):
    mo.md(r"""
    # diffusion-ebm — project tour

    Hybridising a pretrained masked-diffusion LM with a THRML
    block-Gibbs joint sampler over a sparse factor graph, to test
    whether joint sampling improves multi-hole consistency at fixed
    neural-FLOPs budget.

    **Honest framing.** With $E = -\sum_i \log p_{\text{LM}}(x_i \mid \text{ctx})$
    and no other terms, joint Gibbs samples from the same distribution as
    independent ancestral. Joint sampling only beats independent when
    $\psi$ encodes information that's not in $p_{\text{LM}}(x_i \mid \text{ctx})$.

    - **M3/M4** use a hard equality factor over matched holes — the simple test bed.
    - **M5b** stresses where it wins / ties / breaks.
    - **M5c** replaces hard equality with a learned scorer.
    """)
    return


@app.cell
def load_data(json, os):
    try:
        REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        REPO_ROOT = os.getcwd()
    RESULTS = os.path.join(REPO_ROOT, "results")
    PLOTS = os.path.join(REPO_ROOT, "plots")

    def load_json(name):
        path = os.path.join(RESULTS, name)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return json.load(f)

    core = load_json("results.json")
    full = load_json("results_all.json")
    return PLOTS, REPO_ROOT, core, full


@app.cell
def m0(mo):
    mo.md(r"""
    ## M0 — environment smoke

    GPU + tokenizer + JAX import + bf16 attention. Validated by
    `notebooks/00_smoke.py`. NixOS env quirks (`LD_LIBRARY_PATH`,
    `TRITON_LIBCUDA_PATH`, `transformers<5`, flash-attn from source)
    are documented in `CLAUDE.md`.
    """)
    return


@app.cell
def m1(PLOTS, mo, os):
    potts_path = os.path.join(PLOTS, "mvp0_potts_alignment.png")
    potts_fig = (
        mo.image(potts_path, width=600)
        if os.path.exists(potts_path)
        else mo.md(f"_(missing `{potts_path}`)_")
    )
    mo.vstack(
        [
            mo.md(
                r"""
    ## M1 — synthetic Potts chain (THRML correctness)

    Validate the THRML categorical block-Gibbs API on a ferromagnetic
    Potts chain before any LM work. At $J=5$, independent ancestral
    gets 0/500 fully aligned; THRML gets 377/500 fully aligned with
    98.4 % mean per-edge alignment. Chain mode-locks as expected.
                """
            ),
            potts_fig,
        ]
    )
    return


@app.cell
def m2(mo):
    mo.md(r"""
    ## M2 — MDLM bridge

    Wrap `kuleshov-group/mdlm-owt` as a stable forward + top-k +
    mask-predict. Vocab is GPT-2 (50 257) + one absorbing/`[MASK]`
    token at id 50 257 → `vocab_size = 50 258`. Forward returns
    `[B, L, 50258]` bf16 logits; top-k=32/64 excludes the mask token.

    **Added in M5c**: `MDLM.forward_hidden(input_ids)` returns
    `(logits, last_hidden_state)` so the learned scorer can read
    MDLM's pre-projection hidden states.
    """)
    return


@app.cell
def helpers():
    def short_template(text):
        if text.startswith("Alice's favorite color is [M]. <"):
            return "distance"
        if text.startswith("Whenever"):
            return "many-holes"
        if text.startswith("The author"):
            return "multi-group"
        if text.startswith("Alice's favorite color is red"):
            return "distractor"
        if text.startswith("She packed"):
            return "polyseme"
        if text.startswith("Alice's favorite color is [M]"):
            return "color"
        if text.startswith("The variable"):
            return "variable"
        if text.startswith("My name"):
            return "repeat-3"
        return text[:25]

    def best_per_method(rows):
        out = {}
        for r in rows:
            m = r["method"]
            if m == "mask_predict":
                m = f"mp@T={r['config']['temperature']:g}"
            cur = out.get(m, (-1.0, 0, 0))
            if r["agreement_rate"] > cur[0]:
                out[m] = (
                    r["agreement_rate"],
                    r["n_lm_forwards"],
                    r["n_gibbs_sweeps"],
                )
        return out

    return best_per_method, short_template


@app.cell
def m34_table(best_per_method, core, mo, short_template):
    m34_templates = ("color", "variable", "repeat-3")
    m34_rows = []
    for _tpl in m34_templates:
        _rows = [r for r in core if short_template(r["template"]) == _tpl]
        _b = best_per_method(_rows)
        _cells = [_tpl]
        for _m in (
            "thrml_joint",
            "mp@T=0",
            "mp@T=1",
            "ancestral_topk_iterative",
            "ancestral_topk",
        ):
            _v = _b.get(_m)
            _cells.append(f"**{_v[0]:.3f}** @ {_v[1]} LM" if _v else "—")
        m34_rows.append("| " + " | ".join(_cells) + " |")
    m34_table_md = "\n".join(
        [
            "| template | thrml | mp@T=0 | mp@T=1 | atk_iter | atk |",
            "|---|---|---|---|---|---|",
            *m34_rows,
        ]
    )

    mo.md(
        rf"""
    ## M3 / M4 — multi-hole agreement headline

    Three templates with `[MASK]` holes that must take the same vocab
    id (color, variable, repeat-3). At one MDLM forward, THRML averages
    ~0.84 across the three templates (1.000 at `equality_weight=10`)
    vs 0.33 (T=0) / 0.15 (T=1) for mask-predict at any budget up to
    64 LM forwards.

    ### Best agreement per (template, method)

    {m34_table_md}

    Filled-sequence perplexity stays in 1.00–1.20 across all winning
    configurations — agreement is not bought with garbage tokens.
        """
    )
    return


@app.cell
def m34_plots(PLOTS, mo, os):
    m34_paths = (
        os.path.join(PLOTS, "headline.png"),
        os.path.join(PLOTS, "pareto_flops.png"),
        os.path.join(PLOTS, "tsu_cost.png"),
    )
    mo.vstack(
        [
            mo.image(p, width=720)
            if os.path.exists(p)
            else mo.md(f"_(missing `{p}`)_")
            for p in m34_paths
        ]
    )
    return


@app.cell
def m5a(core, mo, short_template):
    m5a_templates = ("color", "variable", "repeat-3")
    m5a_rows = []
    for _tpl in m5a_templates:
        _sub = sorted(
            (
                r
                for r in core
                if short_template(r["template"]) == _tpl
                and r["method"] == "ancestral_topk_iterative"
                and r["config"].get("k") == 64
            ),
            key=lambda r: r["config"]["n_iters"],
        )
        for _r in _sub:
            m5a_rows.append(
                f"| {_tpl} | {_r['config']['n_iters']} | "
                f"{_r['agreement_rate']:.3f} | {_r['n_lm_forwards']} |"
            )
    m5a_table_md = "\n".join(
        [
            "| template | n_iters | agreement | LM fwds |",
            "|---|---|---|---|",
            *m5a_rows,
        ]
    )

    mo.md(
        rf"""
    ## M5a — strengthened iterative ancestral baseline

    New baseline `ancestral_topk_iterative`: at each of `n_iters` MDLM
    forwards, sample every still-masked hole's top-k softmax and
    commit the most-confident hole(s) per chain. Strongest
    "sample-one-hole-at-a-time" baseline.

    ### Effect of `n_iters` (k=64)

    {m5a_table_md}

    **Verdict.** Lifts color from 0.000 → 0.469 at 2 LM forwards but
    still loses to `mask_predict@T=0` on color (1.000 @ 4 LM) and
    stays pinned at ~0 on variable / repeat-3 even at 3 LM forwards.
    M4 headline survives the strengthening: only THRML hits 1.000 on
    all three templates at a single LM forward.
        """
    )
    return


@app.cell
def m5b_table(best_per_method, full, mo, short_template):
    m5b_families = (
        "color",
        "variable",
        "repeat-3",
        "distance",
        "many-holes",
        "multi-group",
        "distractor",
        "polyseme",
    )
    m5b_rows = []
    if full:
        for _fam in m5b_families:
            _rows = [r for r in full if short_template(r["template"]) == _fam]
            _b = best_per_method(_rows)
            _thrml_a = _b.get("thrml_joint", (0.0, 0, 0))[0]
            _baselines = max(
                _b.get("mp@T=0", (0.0, 0, 0))[0],
                _b.get("mp@T=1", (0.0, 0, 0))[0],
                _b.get("ancestral_topk_iterative", (0.0, 0, 0))[0],
            )
            if _thrml_a == 0.0 and _baselines == 0.0:
                _verdict = "degenerate"
            elif _thrml_a - _baselines >= 0.3:
                _verdict = "**THRML wins**"
            elif abs(_thrml_a - _baselines) < 0.1:
                _verdict = "_tie_"
            else:
                _verdict = "narrow"

            _cells = [_fam]
            for _m in (
                "thrml_joint",
                "mp@T=0",
                "mp@T=1",
                "ancestral_topk_iterative",
            ):
                _v = _b.get(_m)
                _cells.append(f"{_v[0]:.3f}" if _v else "—")
            _cells.append(_verdict)
            m5b_rows.append("| " + " | ".join(_cells) + " |")
        m5b_table_md = "\n".join(
            [
                "| family | thrml | mp@T=0 | mp@T=1 | atk_iter | verdict |",
                "|---|---|---|---|---|---|",
                *m5b_rows,
            ]
        )
    else:
        m5b_table_md = (
            "_(no `results/results_all.json` — run "
            "`experiments/run_pareto.py --templates all "
            "--out results/results_all.json` first)_"
        )

    mo.md(
        rf"""
    ## M5b — boundary template families

    Five new families designed to expose where THRML wins, ties, or
    breaks: **distance** (equality across long context), **many-holes**
    (4 equality holes), **multi-group** (two distinct groups in one
    prompt), **distractor** (prefix biases per-hole conditional),
    **polyseme** (local LM modes differ but a polyseme like "boot"
    fits both contexts).

    ### Verdict per family

    {m5b_table_md}

    Averaged over the 5 M5b families, THRML reaches 0.80 agreement at
    1 LM forward; mp@T=0 plateaus at 0.40 and mp@T=1 at 0.28 even at
    64 LM forwards. THRML wins by ≥ 0.3 on variable, repeat-3,
    multi-group, polyseme; ties on the cascadable patterns (color,
    distance, distractor); the many-holes family is **degenerate** —
    at top-k=64, no token sits in all four holes' candidate sets, so
    the hard equality factor admits no consistent fill. That's a
    property of the top-k state space, not the joint sampler.
        """
    )
    return


@app.cell
def m5b_plots(PLOTS, mo, os):
    m5b_plot_paths = (
        os.path.join(PLOTS, "m5b_boundary.png"),
        os.path.join(PLOTS, "m5b_headline.png"),
    )
    mo.vstack(
        [
            mo.image(p, width=720)
            if os.path.exists(p)
            else mo.md(f"_(missing `{p}`)_")
            for p in m5b_plot_paths
        ]
    )
    return


@app.cell
def m5c(REPO_ROOT, json, mo, os):
    m5c_decisions = [
        (
            "Objective",
            "Joint-Transition NCE — positives are corpus pairs, negatives are independent top-k LM-marginal samples at each masked position",
        ),
        (
            "Encoder",
            "MDLM last hidden state (free, shares tokenizer; the new `MDLM.forward_hidden` returns it)",
        ),
        (
            "Architecture",
            "Bilinear factorisation $\\langle f(h_a, x_i), g(h_b, x_j) \\rangle$ → full $[k,k]$ pair table = 2 MLP forwards + 1 matmul",
        ),
        ("Corpus", "OpenWebText (HF streaming) — matches MDLM pretraining"),
        (
            "Eval",
            "M5b distractor + polyseme as primary; OntoNotes deferred",
        ),
    ]
    m5c_decisions_md = "\n".join(
        [
            "| key | choice |",
            "|---|---|",
            *(f"| {k} | {v} |" for k, v in m5c_decisions),
        ]
    )

    m5c_files = [
        (
            "src/diffusion_ebm/factors/learned.py",
            "PairwiseScorer + stack_learned_factor",
        ),
        (
            "src/diffusion_ebm/sampler/thrml_joint_learned.py",
            "drop-in parallel of thrml_joint.build using learned weights",
        ),
        (
            "experiments/m5c_train.py",
            "data + InfoNCE loop; --corpus {synthetic, owt}",
        ),
        (
            "experiments/m5c_smoke.py",
            "300-step synthetic + factor-graph sanity (passes locally, 6.3 s)",
        ),
        (
            "experiments/m5c_eval.py",
            "(weight_scale × burn_in) sweep with a trained checkpoint",
        ),
        (
            "notebooks/06_learned.py",
            "overlay plot: learned-ψ vs hard-equality vs mp@T=0",
        ),
    ]
    m5c_file_rows = []
    for _path, _descr in m5c_files:
        _marker = "✓" if os.path.exists(os.path.join(REPO_ROOT, _path)) else "✗"
        m5c_file_rows.append(f"| {_marker} | `{_path}` | {_descr} |")
    m5c_files_md = "\n".join(
        ["|   | path | description |", "|---|---|---|", *m5c_file_rows]
    )

    m5c_log_path = os.path.join(REPO_ROOT, "results/m5c_smoke/log.json")
    if os.path.exists(m5c_log_path):
        with open(m5c_log_path) as f:
            m5c_log = json.load(f)
        _first, _last = m5c_log[0]["loss"], m5c_log[-1]["loss"]
        _last_acc = m5c_log[-1]["acc"]
        m5c_smoke_md = (
            f"**Smoke (RTX 3000 Ada, 6.3 s):** loss {_first:.3f} → {_last:.3f} "
            f"({(_first - _last) / _first * 100:.1f}% drop), "
            f"NCE acc {_last_acc:.2f}."
        )
    else:
        m5c_smoke_md = (
            "_(no smoke log; run `uv run python experiments/m5c_smoke.py`)_"
        )

    mo.md(
        rf"""
    ## M5c — learned EBM correction

    Replace the hard equality factor with a *learned* pairwise scorer
    $\psi(x_i, x_j \mid \text{{ctx}})$ trained with Joint-Transition
    NCE so the sampler can correct for cross-position structure that
    hard equality misses.

    ### Decision matrix

    {m5c_decisions_md}

    ### Files

    {m5c_files_md}

    {m5c_smoke_md}

    A pre-training check caught one load-bearing bug (an `(ids, unary)`
    tuple swap from `MDLM.top_k_candidates` in `m5c_smoke.py` and
    `m5c_eval.py`) and three minor issues (synthetic-doc EOS-padding,
    single-pass negative rejection); all fixed; smoke re-passed. The
    pipeline is wired correctly for full-scale training.
        """
    )
    return


@app.cell
def status(mo):
    mo.md(r"""
    ## Where the project stands

    | milestone | status |
    |---|---|
    | M0 / M1 / M2 | passed |
    | M3 multi-hole agreement headline | THRML wins at 1 LM forward |
    | M4 Pareto sweep | headline survives across burn-in × eq-weight × k |
    | M5a iterative ancestral baseline | ships, headline survives |
    | M5b boundary families | 5 new + 3 core; THRML wins on 4, ties on 3, degenerate on 1 |
    | M5c learned correction | scaffolded + smoke-tested |
    """)
    return


if __name__ == "__main__":
    app.run()
