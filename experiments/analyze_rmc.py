"""Offline analysis of RMC eval shards — bootstrap CIs, paired Δ, cost, table.

Reads one or more `m5c_eval_rmc.py` output JSONs (sharded by corpus×track),
and for each (corpus, track) cell reports:

  * a per-variant table on the method-independent supported subset
    (chain_em_supported = chain_em restricted to items where gold is in
    top-k at every mask position),
  * paired Δ-vs-baseline on chain_em_supported with item-level bootstrap CIs
    (resample item_id) — the primary significance test,
  * the Track-0 audit comparisons (hard_eq_map vs sampling; mcmc_logits vs
    argmax; joint vs best_of_n_strong; ψ vs hard equality interchangeability),
  * de-oracle grouping quality + oracle-gain-recovery fraction for
    predicted_group_hard_eq,
  * a neural-FLOPs cost column (mean n_lm_forwards) for the Pareto figure.

Pure stdlib — no numpy/jax/torch — so it runs with the system Python on the
NixOS host without the CUDA LD_LIBRARY_PATH workaround.

Usage:
    python experiments/analyze_rmc.py results/m5c_rmc_dev_*_smoke200.json
    python experiments/analyze_rmc.py results/m5c_rmc_dev_*.json --out analysis.md
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import defaultdict


# ── Loading ──────────────────────────────────────────────────────────────────

def load_records(patterns: list[str]) -> list[dict]:
    """Load + concatenate records from every file matching the glob patterns.

    De-dups on (item_id, method, config) keeping the last seen, so re-globbing
    overlapping files (e.g. a re-run shard) does not double-count.
    """
    paths: list[str] = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        print(f"fatal: no files match {patterns}", file=sys.stderr)
        raise SystemExit(1)

    seen: dict[tuple, dict] = {}
    for path in paths:
        with open(path) as f:
            for r in json.load(f):
                key = (r["item_id"], r["method"],
                       json.dumps(r["config"], sort_keys=True))
                seen[key] = r
    print(f"[analyze] loaded {len(seen)} records from {len(paths)} file(s)")
    return list(seen.values())


# ── Variant keying ─────────────────────────────────────────────────────────

def variant_key(r: dict) -> tuple:
    """(method, weight_scale, burn_in) — splits the ψ sweep into its configs."""
    cfg = r["config"]
    if "weight_scale" in cfg:
        return (r["method"], cfg["weight_scale"], cfg.get("burn_in"))
    return (r["method"], None, None)


def variant_label(key: tuple) -> str:
    method, ws, burn = key
    if ws is not None:
        return f"{method}[ws={ws:g},burn={burn}]"
    return method


# ── Cells ────────────────────────────────────────────────────────────────────

def build_cells(records: list[dict]) -> dict:
    """cell[(corpus, track)] = {support: {iid: bool}, variants: {key: {iid: r}}}."""
    cells: dict = defaultdict(
        lambda: {"support": {}, "variants": defaultdict(dict)}
    )
    for r in records:
        cell = cells[(r["corpus"], r["track"])]
        # all_supported is method-independent (k-determined); any record sets it.
        cell["support"][r["item_id"]] = r["all_supported"]
        cell["variants"][variant_key(r)][r["item_id"]] = r
    return cells


# ── Paired bootstrap ─────────────────────────────────────────────────────────

def paired_diffs(map_a: dict, map_b: dict) -> list[float]:
    common = sorted(set(map_a) & set(map_b))
    return [map_a[i] - map_b[i] for i in common
            if map_a[i] is not None and map_b[i] is not None]


def bootstrap_ci(diffs: list[float], n_boot: int, seed: int) -> tuple:
    """Returns (point, lo, hi, n). Item-level resample of paired differences."""
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    point = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(int(0.975 * n_boot), n_boot - 1)]
    return (point, lo, hi, n)


# ── Per-variant metric extraction ────────────────────────────────────────────

def value_map(variant: dict, metric: str, item_filter: set | None) -> dict:
    out = {}
    for iid, r in variant.items():
        if item_filter is None or iid in item_filter:
            v = r[metric]
            out[iid] = (float(v) if v is not None else None)
    return out


def supported_rate(variant: dict, supported_ids: set, metric: str = "chain_em") -> float:
    vals = [variant[i][metric] for i in variant if i in supported_ids]
    vals = [float(v) for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def best_psi(variants: dict, method: str, supported_ids: set):
    cands = [k for k in variants if k[0] == method]
    best, best_rate = None, -1.0
    for k in sorted(cands):
        rate = supported_rate(variants[k], supported_ids)
        if rate == rate and rate > best_rate:  # rate==rate filters NaN
            best, best_rate = k, rate
    return best


# ── Reporting ────────────────────────────────────────────────────────────────

# Stable display order: baselines first, then the sweeps, then audit baselines.
METHOD_ORDER = {
    "mdlm_argmax": 0,
    "mask_predict_T0": 1,
    "best_of_n_cheap": 2,
    "best_of_n_strong": 3,
    "mcmc_logits": 4,
    "hard_eq_map": 5,
    "hard_eq_thrml_oracle": 6,
    "hard_eq_thrml_global": 7,
    "predicted_group_hard_eq": 8,
    "learned_psi_thrml": 9,
    "learned_psi_thrml_global": 10,
}


def fmt(x: float) -> str:
    return "  nan " if x != x else f"{x:+.3f}"


def analyze_cell(corpus: str, track: str, cell: dict,
                 n_boot: int, seed: int, out: list[str]) -> None:
    support = cell["support"]
    variants = cell["variants"]
    supported_ids = {iid for iid, ok in support.items() if ok}
    n_items = len(support)
    n_sup = len(supported_ids)

    out.append(f"\n## {corpus} / {track}")
    out.append(f"supported: {n_sup} / {n_items} items "
               f"({100*n_sup/n_items:.0f}% gold-in-top-k at every hole)\n")

    # ── Per-variant table ─────────────────────────────────────────────────
    out.append("### Per-variant")
    out.append("| method | n | n_sup | chain_em | chain_em_sup | tok_acc | "
               "agree | n_lm | wall_s |")
    out.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")

    def vsort(k):
        return (METHOD_ORDER.get(k[0], 99),
                k[1] if k[1] is not None else -1,
                k[2] if k[2] is not None else -1)

    for k in sorted(variants, key=vsort):
        v = variants[k]
        n = len(v)
        sup_here = [v[i] for i in v if i in supported_ids]
        cem = sum(r["chain_em"] for r in v.values()) / n if n else float("nan")
        cem_sup = (sum(r["chain_em"] for r in sup_here) / len(sup_here)
                   if sup_here else float("nan"))
        tok = sum(r["token_accuracy"] for r in v.values()) / n
        agr = sum(r["unsupervised_agreement"] for r in v.values()) / n
        nlm = sum(r["n_lm_forwards"] for r in v.values()) / n
        wall = sum(r["wall_time_s"] for r in v.values()) / n
        out.append(
            f"| {variant_label(k)} | {n} | {len(sup_here)} | "
            f"{cem:.3f} | {cem_sup:.3f} | {tok:.3f} | {agr:.3f} | "
            f"{nlm:.1f} | {wall:.2f} |"
        )

    # ── Paired Δ comparisons (chain_em_supported) ─────────────────────────
    psi_b = best_psi(variants, "learned_psi_thrml", supported_ids)
    psi_g = best_psi(variants, "learned_psi_thrml_global", supported_ids)

    arg = ("mdlm_argmax", None, None)
    oracle = ("hard_eq_thrml_oracle", None, None)
    bons = ("best_of_n_strong", None, None)
    hmap = ("hard_eq_map", None, None)
    pg = ("predicted_group_hard_eq", None, None)
    mcmc = ("mcmc_logits", None, None)

    comparisons = [
        ("hard_eq_oracle − argmax  (joint win)", oracle, arg),
        ("predicted_group − argmax  (de-oracle gain)", pg, arg),
        ("mcmc_logits − argmax  (no-factor ctrl, expect ~0)", mcmc, arg),
        ("hard_eq_map − hard_eq_oracle  (sampling earns nothing)", hmap, oracle),
        ("hard_eq_oracle − best_of_n_strong  (matched FLOPs)", oracle, bons),
    ]
    if psi_b:
        comparisons.append((f"{variant_label(psi_b)} − argmax", psi_b, arg))
        comparisons.append(
            (f"{variant_label(psi_b)} − hard_eq_oracle  (interchangeable?)",
             psi_b, oracle))
    if psi_g:
        comparisons.append((f"{variant_label(psi_g)} − argmax  (no-grouping)",
                            psi_g, arg))

    out.append("\n### Paired Δ on chain_em_supported "
               f"(95% bootstrap CI, B={n_boot})")
    out.append("| comparison | Δ | 95% CI | sig |")
    out.append("|---|--:|---|:--:|")
    for label, a_key, b_key in comparisons:
        if a_key not in variants or b_key not in variants:
            continue
        ma = value_map(variants[a_key], "chain_em", supported_ids)
        mb = value_map(variants[b_key], "chain_em", supported_ids)
        diffs = paired_diffs(ma, mb)
        point, lo, hi, n = bootstrap_ci(diffs, n_boot, seed)
        sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
        out.append(f"| {label} | {fmt(point)} | "
                   f"[{fmt(lo)}, {fmt(hi)}] (n={n}) | {sig} |")

    # ── Grouping quality (predicted_group_hard_eq) ────────────────────────
    if pg in variants:
        pgv = variants[pg]
        aris = [r["grouping_ari"] for r in pgv.values()
                if r.get("grouping_ari") is not None]
        f1s = [r["grouping_pairwise_f1"] for r in pgv.values()
               if r.get("grouping_pairwise_f1") is not None]
        oms = [r["over_merge_rate"] for r in pgv.values()
               if r.get("over_merge_rate") is not None]
        ngs = [r["n_pred_groups"] for r in pgv.values()
               if r.get("n_pred_groups") is not None]
        if aris:
            out.append("\n### De-oracle grouping (predicted_group_hard_eq)")
            out.append(f"- grouping ARI:        {sum(aris)/len(aris):.3f}")
            out.append(f"- pairwise link F1:    {sum(f1s)/len(f1s):.3f}")
            out.append(f"- over-merge rate:     {sum(oms)/len(oms):.3f}")
            out.append(f"- mean predicted groups: {sum(ngs)/len(ngs):.2f}")

            # Oracle-gain-recovery fraction (chain_em_supported).
            if oracle in variants and arg in variants:
                arg_rate = supported_rate(variants[arg], supported_ids)
                ora_rate = supported_rate(variants[oracle], supported_ids)
                pg_rate = supported_rate(pgv, supported_ids)
                gain = ora_rate - arg_rate
                if gain > 1e-9:
                    frac = (pg_rate - arg_rate) / gain
                    out.append(f"- oracle-gain recovered: {100*frac:.0f}% "
                               f"(argmax {arg_rate:.3f} → pred_group "
                               f"{pg_rate:.3f} → oracle {ora_rate:.3f})")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+",
                   help="JSON shard files or globs (e.g. results/m5c_rmc_dev_*.json)")
    p.add_argument("--n-boot", type=int, default=10000,
                   help="Bootstrap resamples (default: 10000)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="Write the markdown report here (also printed to stdout)")
    args = p.parse_args()

    records = load_records(args.files)
    cells = build_cells(records)

    out: list[str] = []
    out.append(f"# RMC analysis — {len(records)} records, {len(cells)} cells")

    for (corpus, track) in sorted(cells):
        analyze_cell(corpus, track, cells[(corpus, track)],
                     args.n_boot, args.seed, out)

    report = "\n".join(out)
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"\n[analyze] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
