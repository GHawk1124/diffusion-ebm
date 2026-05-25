"""M5c — plot learned-ψ THRML against the M5b boundary.

Reads:
  * ``results/results_all.json``  (M5b sweep — hard equality + baselines)
  * ``results/m5c_eval.json``     (output of experiments/m5c_eval.py)
  * ``results/m5c/log.json``      (training loss curve)

Emits:
  * ``plots/m5c_loss.png``        — InfoNCE training loss vs step
  * ``plots/m5c_overlay.png``     — per-template panel: hard-equality
    THRML, mask_predict@T=0, and learned-ψ THRML at multiple weight
    scales overlaid.

Run:

    LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH" \\
        uv run python notebooks/06_learned.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(_REPO_ROOT, "plots")
os.makedirs(PLOTS, exist_ok=True)


def _short(text: str) -> str:
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


def _loss_plot(log_path: str) -> None:
    if not os.path.exists(log_path):
        print(f"[plot] no training log at {log_path}; skipping loss plot")
        return
    with open(log_path) as f:
        log = json.load(f)
    steps = [r["step"] for r in log]
    losses = [r["loss"] for r in log]
    accs = [r["acc"] for r in log]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(steps, losses, color="tab:red", label="InfoNCE loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(steps, accs, color="tab:blue", linestyle="--", label="pos>max(neg)")
    ax2.set_ylabel("NCE accuracy", color="tab:blue")
    ax2.set_ylim(0, 1)
    ax1.set_title("M5c training")
    fig.tight_layout()
    out = os.path.join(PLOTS, "m5c_loss.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def _overlay_plot(boundary_path: str, eval_path: str) -> None:
    if not os.path.exists(eval_path):
        print(f"[plot] no eval data at {eval_path}; skipping overlay")
        return
    with open(boundary_path) as f:
        b = json.load(f)
    with open(eval_path) as f:
        e = json.load(f)

    families = ("color", "variable", "repeat-3",
                "distance", "many-holes", "multi-group",
                "distractor", "polyseme")

    def by_fam(records):
        out = defaultdict(list)
        for r in records:
            out[_short(r["template"])].append(r)
        return out
    bf = by_fam(b)
    ef = by_fam(e)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    axes = axes.flatten()
    for ax, fam in zip(axes, families):
        # mask_predict @ T=0 reference
        mp = sorted(
            (r for r in bf.get(fam, [])
             if r["method"] == "mask_predict"
             and r["config"]["temperature"] == 0.0),
            key=lambda r: r["n_lm_forwards"],
        )
        if mp:
            ax.plot([r["n_lm_forwards"] for r in mp],
                    [r["agreement_rate"] for r in mp],
                    linestyle="--", marker="o", color="tab:blue",
                    label="mask_predict@T=0")
        # hard-equality THRML best
        thrml_hard = [r for r in bf.get(fam, []) if r["method"] == "thrml_joint"]
        if thrml_hard:
            best = max(thrml_hard, key=lambda r: r["agreement_rate"])
            ax.scatter([1], [best["agreement_rate"]], marker="*", s=240,
                       color="tab:red", edgecolor="black",
                       label="THRML hard-eq (best)", zorder=5)
        # learned-ψ THRML, point per (weight_scale, burn_in)
        for r in ef.get(fam, []):
            ax.scatter([1], [r["agreement_rate"]], marker="P", s=80,
                       color="tab:purple", alpha=0.6)
        if ef.get(fam):
            best_l = max(ef[fam], key=lambda r: r["agreement_rate"])
            ax.scatter([1], [best_l["agreement_rate"]], marker="P", s=200,
                       color="tab:purple", edgecolor="black",
                       label="THRML learned-ψ (best)", zorder=6)
        ax.set_xscale("log")
        ax.set_xlim(0.7, 100)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(fam)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("LM forwards (log)")

    axes[0].set_ylabel("agreement rate")
    axes[3].legend(loc="lower right", fontsize=8)
    fig.suptitle("M5c: learned ψ vs hard-equality THRML and mask_predict@T=0", y=1.0)
    fig.tight_layout()
    out = os.path.join(PLOTS, "m5c_overlay.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def main() -> None:
    _loss_plot(os.path.join(_REPO_ROOT, "results/m5c/log.json"))
    _overlay_plot(
        os.path.join(_REPO_ROOT, "results/results_all.json"),
        os.path.join(_REPO_ROOT, "results/m5c_eval.json"),
    )


if __name__ == "__main__":
    main()
