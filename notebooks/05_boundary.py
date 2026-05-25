"""M5b — boundary analysis across template families.

Reads ``results/results_all.json`` (produced by
``experiments/run_pareto.py --templates all``) and emits a per-family
panel plot plus a markdown summary table that names which families
THRML wins, ties, or loses.

Run:

    LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH" \\
    TRITON_LIBCUDA_PATH="/run/opengl-driver/lib" \\
        uv run python notebooks/05_boundary.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_REPO_ROOT, "results/results_all.json")
PLOTS = os.path.join(_REPO_ROOT, "plots")
os.makedirs(PLOTS, exist_ok=True)


# Map the verbatim text-prefix of each template onto a short family label.
def family_of(text: str) -> str:
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


CORE = ("color", "variable", "repeat-3")
M5B = ("distance", "many-holes", "multi-group", "distractor", "polyseme")

METHOD_COLORS = {
    "thrml_joint": "tab:red",
    "mask_predict": "tab:blue",
    "ancestral_topk": "tab:gray",
    "ancestral_topk_iterative": "tab:green",
    "independent_full": "tab:olive",
}


def load() -> list[dict]:
    with open(RESULTS) as f:
        return json.load(f)


def best_per_method(rows: list[dict]) -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    for r in rows:
        m = r["method"]
        if m == "mask_predict":
            t = r["config"]["temperature"]
            m = f"mask_predict@T={t:g}"
        cur = out.get(m, (-1.0, 0))
        if r["agreement_rate"] > cur[0]:
            out[m] = (r["agreement_rate"], r["n_lm_forwards"])
    return out


def panel(records: list[dict]) -> None:
    families = CORE + M5B
    rows_by_family: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        rows_by_family[family_of(r["template"])].append(r)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    axes = axes.flatten()
    for ax, fam in zip(axes, families):
        rows = rows_by_family.get(fam, [])
        if not rows:
            ax.set_visible(False)
            continue

        # mask_predict curves (T=0 dashed, T=1 solid)
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
                    xs, ys, linestyle=ls, marker="o",
                    color=METHOD_COLORS["mask_predict"],
                    label=f"mask_predict (T={temp:g})",
                )

        # ancestral_topk_iterative @ k=64
        ait = sorted(
            (r for r in rows
             if r["method"] == "ancestral_topk_iterative"
             and r["config"].get("k") == 64),
            key=lambda r: r["n_lm_forwards"],
        )
        if ait:
            ax.plot(
                [r["n_lm_forwards"] for r in ait],
                [r["agreement_rate"] for r in ait],
                linestyle=":", marker="D",
                color=METHOD_COLORS["ancestral_topk_iterative"],
                label="ancestral_topk_iterative (k=64)",
            )

        # thrml_joint best across configs (single point at x=1)
        thrml = [r for r in rows if r["method"] == "thrml_joint"]
        if thrml:
            best = max(thrml, key=lambda r: r["agreement_rate"])
            ax.scatter(
                [1], [best["agreement_rate"]],
                marker="*", s=240,
                color=METHOD_COLORS["thrml_joint"],
                edgecolor="black", linewidth=1.0,
                zorder=5, label="thrml_joint (best)",
            )

        ax.set_xscale("log")
        ax.set_xlim(0.7, 100)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(fam + ("  (M5b)" if fam in M5B else "  (core)"))
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("LM forwards (log)")

    axes[0].set_ylabel("agreement rate")
    axes[4].set_ylabel("agreement rate")
    axes[3].legend(fontsize=8, loc="lower right")

    # Hide last subplot if we have 8 families exactly fitting 2x4 we use them all.
    if len(families) < len(axes):
        for ax in axes[len(families):]:
            ax.set_visible(False)

    fig.suptitle("M5b boundary: agreement vs LM-forward budget per template family", y=1.0)
    fig.tight_layout()
    out = os.path.join(PLOTS, "m5b_boundary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def headline(records: list[dict]) -> None:
    """One-panel headline averaged across M5b families only."""
    by_method_x: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if family_of(r["template"]) not in M5B:
            continue
        if r["method"] == "thrml_joint":
            cfg = r["config"]
            if cfg["equality_weight"] != 10.0 or cfg["k"] != 64:
                continue
        if r["method"] == "ancestral_topk" and r["config"]["k"] != 64:
            continue
        if r["method"] == "ancestral_topk_iterative" and r["config"].get("k") != 64:
            continue
        method = r["method"]
        if method == "mask_predict":
            method = f"mask_predict (T={r['config']['temperature']:g})"
        by_method_x[method][r["n_lm_forwards"]].append(r["agreement_rate"])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method, x_to_ys in by_method_x.items():
        xs = sorted(x_to_ys.keys())
        ys = [float(np.mean(x_to_ys[x])) for x in xs]
        marker, ls, color = "o", "-", None
        if method.startswith("thrml"):
            color = METHOD_COLORS["thrml_joint"]
            marker = "*"
        elif method.startswith("mask_predict"):
            color = METHOD_COLORS["mask_predict"]
            ls = "--" if "T=0" in method else "-"
        elif method == "ancestral_topk_iterative":
            color = METHOD_COLORS["ancestral_topk_iterative"]
            marker = "D"
            ls = ":"
        elif method == "ancestral_topk":
            color = METHOD_COLORS["ancestral_topk"]
            marker = "s"
        elif method == "independent_full":
            color = METHOD_COLORS["independent_full"]
            marker = "x"
        ax.plot(xs, ys, marker=marker, color=color, linestyle=ls, label=method)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("LM forwards (log)")
    ax.set_ylabel("agreement rate (avg over M5b families)")
    ax.set_title("M5b headline: average over the 5 boundary families")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = os.path.join(PLOTS, "m5b_headline.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def summary(records: list[dict]) -> None:
    families = CORE + M5B
    rows_by_family: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        rows_by_family[family_of(r["template"])].append(r)

    print()
    print("# Best agreement per (family, method)\n")
    methods = [
        "thrml_joint",
        "mask_predict@T=0",
        "mask_predict@T=1",
        "ancestral_topk_iterative",
        "ancestral_topk",
    ]
    print("| family | thrml | mp@T=0 | mp@T=1 | atk_iter | atk | verdict |")
    print("|---|---|---|---|---|---|---|")
    for fam in families:
        rows = rows_by_family.get(fam, [])
        b = best_per_method(rows)
        cells = []
        for m in methods:
            if m in b:
                a, lm = b[m]
                cells.append(f"{a:.3f} @ {lm}LM")
            else:
                cells.append("—")
        # Determine verdict for M5b families
        thrml_a = b.get("thrml_joint", (0.0, 0))[0]
        best_baseline = max(
            b.get("mask_predict@T=0", (0.0, 0))[0],
            b.get("mask_predict@T=1", (0.0, 0))[0],
            b.get("ancestral_topk_iterative", (0.0, 0))[0],
        )
        if thrml_a - best_baseline >= 0.3:
            verdict = "**THRML wins**"
        elif abs(thrml_a - best_baseline) < 0.1:
            verdict = "_tie_"
        elif best_baseline - thrml_a >= 0.3:
            verdict = "_baseline wins_"
        else:
            verdict = "narrow"
        print("| " + fam + " | " + " | ".join(cells) + " | " + verdict + " |")
    print()


def main() -> None:
    records = load()
    print(f"[plot] loaded {len(records)} records from {RESULTS}")
    panel(records)
    headline(records)
    summary(records)


if __name__ == "__main__":
    main()
