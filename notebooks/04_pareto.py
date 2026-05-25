"""M4 — Pareto plots and summary table.

Reads ``results/results.json`` (produced by ``experiments/run_pareto.py``)
and emits:

* ``plots/pareto_flops.png`` — agreement vs LM forwards, one panel per
  template, methods overlaid. THRML's vertical column at x=1 makes the
  central claim visible: at the same LM budget, joint Gibbs strictly
  dominates mask_predict on at least two templates.
* ``plots/tsu_cost.png`` — agreement vs total Gibbs sweeps for THRML
  alone. Annotated with the TSU cost-model framing: under
  ``C_G/C_N → 0`` every point on this curve costs the same as a single
  LM forward.
* ``plots/headline.png`` — average across templates, one line per method.
* A markdown summary table on stdout.

Run:

    LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH" \\
    TRITON_LIBCUDA_PATH="/run/opengl-driver/lib" \\
        uv run python notebooks/04_pareto.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_REPO_ROOT, "results/results.json")
PLOTS = os.path.join(_REPO_ROOT, "plots")
os.makedirs(PLOTS, exist_ok=True)

METHOD_COLORS = {
    "thrml_joint": "tab:red",
    "mask_predict": "tab:blue",
    "ancestral_topk": "tab:gray",
    "ancestral_topk_iterative": "tab:green",
    "independent_full": "tab:olive",
}
METHOD_MARKERS = {
    "thrml_joint": "*",
    "mask_predict": "o",
    "ancestral_topk": "s",
    "ancestral_topk_iterative": "D",
    "independent_full": "x",
}


def short(text: str) -> str:
    if "color" in text:
        return "color"
    if "variable" in text:
        return "variable"
    if "name" in text:
        return "repeat-3"
    return text[:20]


def load() -> list[dict]:
    with open(RESULTS) as f:
        return json.load(f)


def _plot_pareto_flops(records: list[dict]) -> None:
    templates = []
    for r in records:
        if r["template"] not in templates:
            templates.append(r["template"])

    fig, axes = plt.subplots(1, len(templates), figsize=(15, 4.5), sharey=True)
    if len(templates) == 1:
        axes = [axes]

    for ax, tpl in zip(axes, templates):
        rows = [r for r in records if r["template"] == tpl]

        # mask_predict: trace the curve at temperature=1 and at temperature=0.
        for temp, ls in ((1.0, "-"), (0.0, "--")):
            mp = sorted(
                (r for r in rows
                 if r["method"] == "mask_predict"
                 and r["config"]["temperature"] == temp),
                key=lambda r: r["n_lm_forwards"],
            )
            if mp:
                xs = [r["n_lm_forwards"] for r in mp]
                ys = [r["agreement_rate"] for r in mp]
                ax.plot(
                    xs, ys,
                    linestyle=ls, marker="o",
                    color=METHOD_COLORS["mask_predict"],
                    label=f"mask_predict (T={temp:g})",
                )

        # THRML: scatter all configs at x=1, star the best per (k, weight) sweep.
        thrml = [r for r in rows if r["method"] == "thrml_joint"]
        if thrml:
            xs = [r["n_lm_forwards"] for r in thrml]
            ys = [r["agreement_rate"] for r in thrml]
            ax.scatter(
                xs, ys,
                marker="*", s=80,
                color=METHOD_COLORS["thrml_joint"],
                alpha=0.55,
                label="thrml_joint (all cfgs)",
            )
            best = max(thrml, key=lambda r: r["agreement_rate"])
            ax.scatter(
                [best["n_lm_forwards"]], [best["agreement_rate"]],
                marker="*", s=240,
                edgecolor="black", linewidth=1.0,
                color=METHOD_COLORS["thrml_joint"],
                label="thrml_joint (best)",
                zorder=5,
            )

        # ancestral_topk + independent_full as single points.
        for method in ("ancestral_topk", "independent_full"):
            sub = [r for r in rows if r["method"] == method]
            if sub:
                xs = [r["n_lm_forwards"] for r in sub]
                ys = [r["agreement_rate"] for r in sub]
                ax.scatter(
                    xs, ys,
                    marker=METHOD_MARKERS[method], s=60,
                    color=METHOD_COLORS[method],
                    label=method,
                )

        # ancestral_topk_iterative: line at fixed k=64 across n_iters.
        ait = sorted(
            (r for r in rows
             if r["method"] == "ancestral_topk_iterative"
             and r["config"].get("k") == 64),
            key=lambda r: r["n_lm_forwards"],
        )
        if ait:
            xs = [r["n_lm_forwards"] for r in ait]
            ys = [r["agreement_rate"] for r in ait]
            ax.plot(
                xs, ys,
                marker=METHOD_MARKERS["ancestral_topk_iterative"], linestyle=":",
                color=METHOD_COLORS["ancestral_topk_iterative"],
                label="ancestral_topk_iterative (k=64)",
            )

        ax.set_xscale("log")
        ax.set_xlim(0.7, 100)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("LM forwards (log)")
        ax.set_title(short(tpl))
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("agreement rate")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Agreement vs LM-forward budget", y=1.02)
    fig.tight_layout()
    out = os.path.join(PLOTS, "pareto_flops.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def _plot_tsu_cost(records: list[dict]) -> None:
    templates = []
    for r in records:
        if r["template"] not in templates:
            templates.append(r["template"])

    thrml = [r for r in records if r["method"] == "thrml_joint"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for tpl in templates:
        # Hold equality_weight fixed at 5 — it's the most calibrated regime
        # (weight=10 saturates, weight=2 is too weak — see appendix).
        sub = sorted(
            (r for r in thrml
             if r["template"] == tpl
             and r["config"]["equality_weight"] == 5.0
             and r["config"]["k"] == 64),
            key=lambda r: r["n_gibbs_sweeps"],
        )
        if not sub:
            continue
        xs = [r["n_gibbs_sweeps"] for r in sub]
        ys = [r["agreement_rate"] for r in sub]
        ax.plot(xs, ys, marker="o", label=short(tpl))

    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("total Gibbs sweeps  (n_chains + burn-in)")
    ax.set_ylabel("agreement rate")
    ax.set_title("THRML budget curve  (k=64, equality_weight=5)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.text(
        0.98, 0.05,
        "Under  $C_G / C_N \\rightarrow 0$  (TSU regime)\n"
        "every point on this curve costs the same\n"
        "as a single LM forward.",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
    )
    fig.tight_layout()
    out = os.path.join(PLOTS, "tsu_cost.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def _plot_headline(records: list[dict]) -> None:
    by_method_x: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in records:
        if r["method"] == "thrml_joint":
            # take only equality_weight=5, k=64 to match the tsu plot
            cfg = r["config"]
            if cfg["equality_weight"] != 5.0 or cfg["k"] != 64:
                continue
        if r["method"] == "ancestral_topk" and r["config"]["k"] != 64:
            continue
        if r["method"] == "ancestral_topk_iterative" and r["config"].get("k") != 64:
            continue
        # for mask_predict, separate temp=0 and temp=1 visually
        method = r["method"]
        if method == "mask_predict":
            t = r["config"]["temperature"]
            method = f"mask_predict (T={t:g})"
        by_method_x[method][r["n_lm_forwards"]].append(r["agreement_rate"])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for method, x_to_ys in by_method_x.items():
        xs = sorted(x_to_ys.keys())
        ys = [float(np.mean(x_to_ys[x])) for x in xs]
        marker = "o"
        ls = "-"
        color = None
        if method.startswith("mask_predict"):
            color = METHOD_COLORS["mask_predict"]
            ls = "--" if "T=0" in method else "-"
        elif method.startswith("thrml"):
            color = METHOD_COLORS["thrml_joint"]
            marker = "*"
        elif method == "ancestral_topk_iterative":
            color = METHOD_COLORS["ancestral_topk_iterative"]
            marker = METHOD_MARKERS["ancestral_topk_iterative"]
            ls = ":"
        elif method.startswith("ancestral"):
            color = METHOD_COLORS["ancestral_topk"]
            marker = "s"
        elif method.startswith("independent"):
            color = METHOD_COLORS["independent_full"]
            marker = "x"
        if method.startswith("thrml") and len(xs) == 1:
            # vertical jitter: show min/max range across burn_in axis
            all_ys = [v for vs in x_to_ys.values() for v in vs]
            ax.errorbar(
                xs, [float(np.mean(all_ys))],
                yerr=[[float(np.mean(all_ys) - min(all_ys))],
                      [float(max(all_ys) - np.mean(all_ys))]],
                fmt=marker, color=color, markersize=14,
                capsize=4, label=method,
            )
        else:
            ax.plot(xs, ys, marker=marker, color=color, linestyle=ls, label=method)

    ax.set_xscale("log")
    ax.set_xlim(0.7, 100)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("LM forwards (log)")
    ax.set_ylabel("agreement rate (avg over 3 templates)")
    ax.set_title("Headline: joint Gibbs vs iterative diffusion")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    out = os.path.join(PLOTS, "headline.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def _print_summary(records: list[dict]) -> None:
    templates: list[str] = []
    for r in records:
        if r["template"] not in templates:
            templates.append(r["template"])

    methods = [
        "thrml_joint",
        "mask_predict",
        "ancestral_topk",
        "ancestral_topk_iterative",
        "independent_full",
    ]
    print()
    print("# Best-config agreement per (template, method)")
    print()
    header = "| template | " + " | ".join(methods) + " |"
    sep = "|" + "|".join(["---"] * (len(methods) + 1)) + "|"
    print(header)
    print(sep)
    for tpl in templates:
        row = [short(tpl)]
        for m in methods:
            sub = [r for r in records if r["template"] == tpl and r["method"] == m]
            if not sub:
                row.append("—")
                continue
            best = max(sub, key=lambda r: r["agreement_rate"])
            row.append(
                f"**{best['agreement_rate']:.3f}** "
                f"@ LM={best['n_lm_forwards']}, gibbs={best['n_gibbs_sweeps']}"
            )
        print("| " + " | ".join(row) + " |")
    print()


def main() -> None:
    records = load()
    print(f"[plot] loaded {len(records)} records from {RESULTS}")
    _plot_pareto_flops(records)
    _plot_tsu_cost(records)
    _plot_headline(records)
    _print_summary(records)


if __name__ == "__main__":
    main()
