"""End-to-end project tour — every milestone in one runnable file.

Walks the reader through the research question, the MVP pipeline, and
each milestone's headline result. The script is intentionally
narrative: it loads the JSON artefacts already on disk
(``results/{results,results_all}.json``, plot PNGs in ``plots/``) and
prints the same tables / surfaces the same figures as the per-milestone
notebooks, but in one pass.

Sections:

* The research question and the honest framing
* M0 — env smoke
* M1 — synthetic Potts chain (THRML correctness)
* M2 — MDLM bridge (forward + mask-predict)
* M3 — multi-hole agreement headline
* M4 — Pareto sweep
* M5a — strengthened iterative ancestral baseline
* M5b — boundary template families
* M5c — learned EBM correction scaffold

Run:

    LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH" \\
    TRITON_LIBCUDA_PATH="/run/opengl-driver/lib" \\
        uv run python notebooks/07_tour.py

By default the tour is data-only: it does NOT load MDLM, does NOT
re-run any sweep, and does NOT regenerate any plot. Pass ``--regen``
to additionally invoke the per-milestone notebooks (~30 s for the
plot scripts; the GPU sweeps are still skipped).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from typing import Iterable

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_REPO_ROOT, "results")
PLOTS = os.path.join(_REPO_ROOT, "plots")


def section(title: str) -> None:
    print(f"\n## {title}")


def subsection(title: str) -> None:
    print(f"\n### {title}")


def kv(items: Iterable[tuple[str, str]]) -> None:
    width = max(len(k) for k, _ in items)
    for k, v in items:
        print(f"  {k:<{width}}  {v}")


def fig_pointer(path: str, alt: str) -> None:
    rel = os.path.relpath(path, _REPO_ROOT)
    if os.path.exists(path):
        print(f"  [figure] {rel}  —  {alt}")
    else:
        print(f"  [figure] {rel}  (NOT FOUND; alt: {alt})")


def m0_m1_m2() -> None:
    section("M0 — environment smoke")
    print(
        "GPU + tokenizer + JAX import + bf16 attention. Validated by\n"
        "notebooks/00_smoke.py. NixOS env quirks (LD_LIBRARY_PATH,\n"
        "TRITON_LIBCUDA_PATH, transformers<5, flash-attn from source)\n"
        "are documented in CLAUDE.md."
    )

    section("M1 — synthetic Potts chain (THRML correctness)")
    print(
        "Goal: validate the THRML categorical block-Gibbs API on a\n"
        "ferromagnetic Potts chain before any LM work.\n\n"
        "Result: at J=5, independent ancestral 0/500 chains fully aligned;\n"
        "THRML 377/500 fully aligned, 98.4% mean per-edge alignment;\n"
        "chain mode-locks as expected."
    )
    fig_pointer(os.path.join(PLOTS, "mvp0_potts_alignment.png"),
                "Potts chain alignment histogram")

    section("M2 — MDLM bridge")
    print(
        "Goal: wrap kuleshov-group/mdlm-owt as a stable forward + top-k +\n"
        "mask-predict. Vocab is GPT-2 (50 257) + one absorbing/[MASK]\n"
        "token at id 50 257 → vocab_size = 50 258. forward returns\n"
        "[B, L, 50258] bf16 logits; top-k=32/64 excludes the mask token;\n"
        "mask-predict produces plausible English at n_iters in {1,4,12}\n"
        "on the 'capital of France' prompt.\n\n"
        "Added in M5c: MDLM.forward_hidden(input_ids) returns\n"
        "(logits, last_hidden_state) so the learned scorer can read\n"
        "MDLM's pre-projection hidden states without a forward hook."
    )


def short_template(text: str) -> str:
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


def best_per_method(rows: list[dict]) -> dict[str, tuple[float, int, int]]:
    """method -> (best agreement, n_lm_forwards, n_gibbs_sweeps)."""
    out: dict[str, tuple[float, int, int]] = {}
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


def m3_m4(records: list[dict]) -> None:
    section("M3 — multi-hole agreement headline")
    print(
        "Three templates with [MASK] holes that must take the same vocab\n"
        "id (color, variable, repeat-3). Compare:\n"
        "  thrml_joint        one MDLM forward + THRML block-Gibbs over\n"
        "                     the top-k state space with hard equality\n"
        "                     factors\n"
        "  mask_predict       MDLM iterative refinement (Ghazvininejad et\n"
        "                     al., 2019), at temperature 0/1\n"
        "  ancestral_topk     one MDLM forward + indep. multinomial draws\n"
        "  independent_full   one MDLM forward + full-vocab indep. draws\n"
    )

    section("M4 — Pareto sweep across methods + configs")
    templates_in_order = []
    for r in records:
        s = short_template(r["template"])
        if s in {"color", "variable", "repeat-3"} and s not in templates_in_order:
            templates_in_order.append(s)
    print(
        f"Sweep grid: {sum(1 for r in records if short_template(r['template']) in templates_in_order)}"
        f" runs across {len(templates_in_order)} core templates"
        f" (data: results/results.json)\n"
    )

    print("Best agreement per (template, method):")
    print()
    print("| template | thrml | mp@T=0 | mp@T=1 | atk_iter | atk |")
    print("|---|---|---|---|---|---|")
    for tpl_short in templates_in_order:
        rows = [r for r in records if short_template(r["template"]) == tpl_short]
        b = best_per_method(rows)
        cells = [tpl_short]
        for m in ("thrml_joint", "mp@T=0", "mp@T=1",
                  "ancestral_topk_iterative", "ancestral_topk"):
            v = b.get(m)
            cells.append(
                f"**{v[0]:.3f}** @ {v[1]}LM"
                if v else "—"
            )
        print("| " + " | ".join(cells) + " |")
    print()
    fig_pointer(os.path.join(PLOTS, "headline.png"),
                "M4 headline averaged across the 3 core templates")
    fig_pointer(os.path.join(PLOTS, "pareto_flops.png"),
                "M4 per-template agreement vs LM-forward budget")
    fig_pointer(os.path.join(PLOTS, "tsu_cost.png"),
                "THRML budget curve under the TSU cost model")
    print(
        "\nHeadline: at one MDLM forward, THRML averages ~0.84 agreement\n"
        "across the three templates (1.000 at equality_weight=10) vs 0.33\n"
        "(T=0) / 0.15 (T=1) for mask-predict at any budget up to 64 LM\n"
        "forwards. Filled-sequence perplexity stays in 1.00–1.20 across\n"
        "all winning configurations — agreement is not bought with\n"
        "garbage tokens."
    )


def m5a(records: list[dict]) -> None:
    section("M5a — strengthened iterative ancestral baseline")
    print(
        "Question: does committed-context cascading lift `ancestral_topk`\n"
        "enough to threaten THRML?\n\n"
        "New baseline `ancestral_topk_iterative`: at each of n_iters MDLM\n"
        "forwards, sample every still-masked hole's top-k softmax and\n"
        "commit ceil(remaining/iters_left) most-confident holes per chain.\n"
        "At n_iters=1 it reduces to ancestral_topk (modulo bf16 batch-size\n"
        "drift); at n_iters=n_holes it's the strongest 'sample-one-hole-\n"
        "at-a-time' baseline."
    )
    print()

    cores = ("color", "variable", "repeat-3")
    print("Effect of n_iters on the iterative baseline (k=64):")
    for tpl_short in cores:
        sub = sorted(
            (r for r in records
             if short_template(r["template"]) == tpl_short
             and r["method"] == "ancestral_topk_iterative"
             and r["config"].get("k") == 64),
            key=lambda r: r["config"]["n_iters"],
        )
        if not sub:
            continue
        print(f"  {tpl_short}:")
        for r in sub:
            print(f"    n_iters={r['config']['n_iters']}  "
                  f"agree={r['agreement_rate']:.3f}  "
                  f"({r['n_lm_forwards']} LM forwards)")
    print(
        "\nVerdict: lifts color from 0.000 → 0.469 at 2 LM forwards but\n"
        "still loses to mask_predict@T=0 on color (1.000 @ 4 LM) and\n"
        "stays pinned at ~0 on variable / repeat-3 even at 3 LM forwards.\n"
        "M4 headline survives the strengthening: only THRML hits 1.000 on\n"
        "all three templates at a single LM forward."
    )


def m5b(records_all: list[dict]) -> None:
    section("M5b — boundary template families")
    print(
        "Five new families designed to expose where THRML's joint sampler\n"
        "wins, ties, or loses:\n"
        "  distance     — equality across an 80-token distractor paragraph\n"
        "  many-holes   — 4 equality holes in tight quarters\n"
        "  multi-group  — two distinct equality groups in one prompt\n"
        "  distractor   — leading prefix biases per-hole conditional\n"
        "  polyseme     — local LM modes differ, joint optimum is a\n"
        "                 polyseme that fits both contexts (e.g. 'boot')\n"
    )

    fams = ("color", "variable", "repeat-3",
            "distance", "many-holes", "multi-group",
            "distractor", "polyseme")

    print("Best agreement per (family, method):\n")
    print("| family | thrml | mp@T=0 | mp@T=1 | atk_iter | verdict |")
    print("|---|---|---|---|---|---|")
    for fam in fams:
        rows = [r for r in records_all if short_template(r["template"]) == fam]
        b = best_per_method(rows)
        thrml_a = b.get("thrml_joint", (0.0, 0, 0))[0]
        baselines = max(
            b.get("mp@T=0", (0.0, 0, 0))[0],
            b.get("mp@T=1", (0.0, 0, 0))[0],
            b.get("ancestral_topk_iterative", (0.0, 0, 0))[0],
        )
        if thrml_a == 0.0 and baselines == 0.0:
            verdict = "degenerate"
        elif thrml_a - baselines >= 0.3:
            verdict = "**THRML wins**"
        elif abs(thrml_a - baselines) < 0.1:
            verdict = "_tie_"
        elif baselines - thrml_a >= 0.3:
            verdict = "_baseline wins_"
        else:
            verdict = "narrow"

        cells = [fam]
        for m in ("thrml_joint", "mp@T=0", "mp@T=1", "ancestral_topk_iterative"):
            v = b.get(m)
            cells.append(f"{v[0]:.3f}" if v else "—")
        cells.append(verdict)
        print("| " + " | ".join(cells) + " |")
    print()
    fig_pointer(os.path.join(PLOTS, "m5b_boundary.png"),
                "Per-family panel: agreement vs LM-forward budget")
    fig_pointer(os.path.join(PLOTS, "m5b_headline.png"),
                "M5b headline averaged across the 5 boundary families")
    print(
        "\nHeadline: averaged over the 5 M5b families, THRML reaches 0.80\n"
        "agreement at 1 LM forward; mp@T=0 plateaus at 0.40 and mp@T=1 at\n"
        "0.28 even at 64 LM forwards. THRML wins by ≥ 0.3 on variable,\n"
        "repeat-3, multi-group, polyseme; ties on color, distance,\n"
        "distractor (all cascadable patterns where mp@T=0 free-rides on\n"
        "committed-argmax). The many-holes family is degenerate — at\n"
        "top-k=64, no token sits in all four hole's candidate sets, so the\n"
        "hard equality factor admits no consistent fill. Property of the\n"
        "top-k state space, not the joint sampler."
    )


def m5c() -> None:
    section("M5c — learned EBM correction")
    print(
        "Goal: replace the hard equality factor with a *learned* pairwise\n"
        "scorer ψ(x_i, x_j | context) so the joint sampler can correct\n"
        "for cross-position structure that hard equality misses.\n"
    )

    subsection("Decision matrix")
    kv([
        ("Objective",
         "Joint-Transition NCE — positives are corpus pairs, negatives "
         "are independent top-k LM-marginal samples at each masked position"),
        ("Encoder",
         "MDLM last hidden state (free, shares tokenizer; the new "
         "MDLM.forward_hidden returns it)"),
        ("Architecture",
         "Bilinear factorisation ⟨f(h_a, x_i), g(h_b, x_j)⟩ → full [k,k] "
         "pair table = 2 MLP forwards + 1 matmul"),
        ("Corpus",
         "OpenWebText (HF streaming) — matches MDLM pretraining"),
        ("Eval",
         "M5b distractor + polyseme as primary; OntoNotes deferred"),
    ])

    subsection("Files")
    files = [
        ("src/diffusion_ebm/factors/learned.py",
         "PairwiseScorer + stack_learned_factor"),
        ("src/diffusion_ebm/sampler/thrml_joint_learned.py",
         "drop-in parallel of thrml_joint.build using learned weights"),
        ("experiments/m5c_train.py",
         "data + InfoNCE loop; --corpus {synthetic, owt}"),
        ("experiments/m5c_smoke.py",
         "300-step synthetic + factor-graph sanity (passes locally, 6.3s)"),
        ("experiments/m5c_eval.py",
         "(weight_scale × burn_in) sweep with a trained checkpoint"),
        ("notebooks/06_learned.py",
         "overlay plot: learned-ψ vs hard-equality vs mp@T=0"),
    ]
    for path, descr in files:
        full = os.path.join(_REPO_ROOT, path)
        marker = "✓" if os.path.exists(full) else "✗"
        print(f"  [{marker}] {path:<55s}  {descr}")

    subsection("Smoke result (RTX 3000 Ada, 6.3 s)")
    log_path = os.path.join(RESULTS, "m5c_smoke/log.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
        first, last = log[0]["loss"], log[-1]["loss"]
        last_acc = log[-1]["acc"]
        print(
            f"  loss {first:.3f} → {last:.3f}  "
            f"({(first - last) / first * 100:.1f}% drop), "
            f"NCE acc {last_acc:.2f}"
        )
    else:
        print(f"  no smoke log at {log_path}")

    print(
        "\nA pre-training check caught one load-bearing bug (an (ids, unary)\n"
        "tuple swap from MDLM.top_k_candidates in m5c_smoke.py and\n"
        "m5c_eval.py) and three minor issues (synthetic-doc EOS-padding,\n"
        "single-pass negative rejection); all fixed; smoke re-passed. The\n"
        "pipeline is wired correctly for full-scale training."
    )


def _maybe_regen(args: argparse.Namespace) -> None:
    if not args.regen:
        return
    notebook_dir = os.path.dirname(os.path.abspath(__file__))
    for nb in ("04_pareto.py", "05_boundary.py", "06_learned.py"):
        path = os.path.join(notebook_dir, nb)
        if not os.path.exists(path):
            continue
        print(f"[tour] regenerating {nb}…")
        rc = subprocess.call([sys.executable, path])
        if rc != 0:
            print(f"[tour] WARN: {nb} returned {rc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end project tour.")
    parser.add_argument(
        "--regen",
        action="store_true",
        help="re-run the per-milestone plot scripts before printing",
    )
    args = parser.parse_args()

    _maybe_regen(args)

    # Load JSON once
    core_path = os.path.join(RESULTS, "results.json")
    all_path = os.path.join(RESULTS, "results_all.json")
    if not os.path.exists(core_path):
        print(f"[tour] {core_path} missing — run experiments/run_pareto.py first")
        return 1
    with open(core_path) as f:
        core = json.load(f)
    if os.path.exists(all_path):
        with open(all_path) as f:
            full = json.load(f)
    else:
        print(
            f"[tour] WARN: {all_path} missing — M5b sections will be sparse;"
            " run `run_pareto.py --templates all --out results/results_all.json`"
        )
        full = []

    section("diffusion-ebm — project tour")
    print(
        "Hybridising a pretrained masked-diffusion LM with a THRML\n"
        "block-Gibbs joint sampler over a sparse factor graph, to test\n"
        "whether joint sampling improves multi-hole consistency at fixed\n"
        "neural-FLOPs budget.\n\n"
        "Honest framing: with E = -Σ log p_LM(x_i | context) and no other\n"
        "terms, joint Gibbs samples from the same distribution as\n"
        "independent ancestral. Joint sampling only beats independent when\n"
        "ψ encodes information that's not in p_LM(x_i | context). M3/M4\n"
        "use a hard equality factor over matched holes — that's the simple\n"
        "test bed. M5b stresses where it wins/ties/breaks. M5c\n"
        "replaces the hard equality with a learned scorer."
    )

    m0_m1_m2()
    m3_m4(core)
    m5a(core)
    if full:
        m5b(full)
    m5c()

    section("Where the project stands")
    kv([
        ("M0 / M1 / M2", "passed"),
        ("M3 multi-hole agreement headline", "THRML wins at 1 LM forward"),
        ("M4 Pareto sweep", "headline survives across burn-in × eq-weight × k"),
        ("M5a iterative ancestral baseline", "ships, headline survives"),
        ("M5b boundary families", "5 new + 3 core; THRML wins on 4, ties on 3, degenerate on 1"),
        ("M5c learned correction", "scaffolded + smoke-tested"),
    ])
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
