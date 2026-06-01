"""Probe A — constrained generation as SAMPLING a frustrated LM posterior.

Constrained decoding is the textbook "joint sampling wins" case (regex, JSON,
brackets). But codex's gate G1b is exactly the trap here: regular / context-free
constraints are solved *exactly and cheaply* by automaton intersection
(Outlines / xgrammar), so they are a solver escape — no hardware win. The honest
project note (CLAUDE.md) already concedes this.

The constraint that does NOT have a cheap solver escape is a GLOBAL,
non-regular one whose normaliser is a #P-hard object: **all-different** across
the masked holes (generate n positions that must take distinct tokens — a
diverse list, distinct identifiers, a ranking/permutation). Sampling
x ∝ Π_i p_LM(x_i) · 1[all x_i distinct] is sampling from a *weighted bipartite
matching* distribution; its partition function is a weighted permanent (#P-hard),
and there is no tree / DP / automaton escape — only MCMC or the permanent FPRAS
(itself MCMC). That makes it a legitimate G0 (sampling) + G1 (no factorisation)
+ G1b (no solver escape) candidate, grounded in REAL MDLM logit fields.

We use a SOFT all-different (antiferromagnetic Potts repulsion, Track B's
`factors/inequality.py` idea) so the posterior is always defined and tunably
frustrated:

    log p(x) = Σ_i u_i(x_i)              # centred real MDLM logits over L tokens
             − R · Σ_{i<j} 1[x_i == x_j]  # repulsion: distinct fills preferred

On the frozen dev logit cache (no GPU), restricting each window's holes to a
shared L-token alphabet so the exact constrained marginals are enumerable
(L^n ≤ 2M). We ask: is the constrained posterior frustrated/multimodal, does
single-T block-Gibbs trap, and does parallel tempering recover the true
constrained marginals? And we run the gauntlet (best-of-N importance sampling)
since that is the matched-FLOPs classical sampler the hardware must beat.

    LD_LIBRARY_PATH="<zlib>:/run/opengl-driver/lib:..." \\
        .venv/bin/python experiments/probe_constrained_gen.py \\
            --cache results/probe_splitability_cache.json \\
            --L 6 --repel 6.0 --out results/probe_constrained_gen.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_hw_common import (  # noqa: E402
    Factor,
    FactorGraph,
    Scorecard,
    block_gibbs,
    co_tv,
    exact_marginals,
    gauntlet_best_of_n,
    marginal_tv,
    parallel_tempering,
    write_report,
)


# ───────────────────── build a constrained graph from a record ──────────────


def shared_alphabet(cand_ids, cand_logits, L: int):
    """Pick the L tokens with the highest total logit mass across all holes."""
    mass: dict[int, float] = {}
    for ids, lg in zip(cand_ids, cand_logits):
        for v, x in zip(ids, lg):
            mass[v] = mass.get(v, 0.0) + float(x)
    top = sorted(mass, key=lambda v: mass[v], reverse=True)[:L]
    return top


def build_constrained_graph(cand_ids, cand_logits, L: int, repel: float):
    """Antiferromagnetic Potts over n holes on a shared L-token alphabet.

    Unary = per-hole MDLM logit (centred to max 0; out-of-top-k tokens floored).
    Pairwise = −repel on the diagonal (same token), 0 off-diagonal.
    """
    alpha = shared_alphabet(cand_ids, cand_logits, L)
    n = len(cand_ids)
    unary = []
    for ids, lg in zip(cand_ids, cand_logits):
        d = {int(v): float(x) for v, x in zip(ids, lg)}
        floor = (min(d.values()) - 5.0) if d else -10.0
        row = np.array([d.get(v, floor) for v in alpha], dtype=np.float64)
        row -= row.max()  # centre so the scale is in nats of LM log-prob gap
        unary.append(row)
    factors = [Factor((i,), unary[i]) for i in range(n)]
    rep = np.zeros((L, L), dtype=np.float64)
    np.fill_diagonal(rep, -repel)
    for a in range(n):
        for b in range(a + 1, n):
            factors.append(Factor((a, b), rep))
    return FactorGraph(tuple([L] * n), factors), alpha


# ───────────────────────────────── main ────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/probe_splitability_cache.json")
    ap.add_argument("--L", type=int, default=6, help="shared alphabet size")
    ap.add_argument("--repel", type=float, default=6.0, help="all-diff penalty (nats)")
    ap.add_argument("--max-n", type=int, default=8, help="cap n so L^n is enumerable")
    ap.add_argument("--max-items", type=int, default=40)
    ap.add_argument("--chains", type=int, default=256)
    ap.add_argument("--burn-in", type=int, default=400)
    ap.add_argument("--n-measure", type=int, default=120)
    ap.add_argument("--pt-levels", type=int, default=8)
    ap.add_argument("--pt-tmax", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/probe_constrained_gen.json")
    args = ap.parse_args()

    blob = json.loads(Path(args.cache).read_text())
    records = [r for r in blob["records"]
               if 3 <= len(r["cand_ids"]) <= args.max_n
               and args.L ** len(r["cand_ids"]) <= 2_000_000]
    records = records[: args.max_items]
    print(f"[probe A] constrained generation (soft all-different)  L={args.L}  "
          f"repel={args.repel}  items={len(records)}")
    print("  target: x ∝ Π p_LM(x_i) · exp(−repel·#collisions)  (weighted matching)")
    print("  exact normaliser = weighted permanent (#P-hard); no tree/automaton escape\n")

    rows = []
    for rec in records:
        g, _ = build_constrained_graph(rec["cand_ids"], rec["cand_logits"],
                                       args.L, args.repel)
        ex = exact_marginals(g, co_cluster=True)
        gb = block_gibbs(g, T=1.0, n_chains=args.chains, burn_in=args.burn_in,
                         n_measure=args.n_measure, seed=args.seed)
        pt = parallel_tempering(g, n_levels=args.pt_levels, t_max=args.pt_tmax,
                                n_chains=args.chains, burn_in=args.burn_in,
                                n_measure=args.n_measure, seed=args.seed)
        bon = gauntlet_best_of_n(g, n=args.chains * args.n_measure, seed=args.seed)
        bon_marg = [np.asarray(m) for m in bon.metric["marginals"]]
        rows.append({
            "item_id": rec["item_id"], "corpus": rec["corpus"],
            "n": len(rec["cand_ids"]),
            "entropy": ex.entropy, "p_map": ex.p_map,
            "tv_gibbs_exact": marginal_tv(gb.var_marginals, ex.var_marginals),
            "tv_pt_exact": marginal_tv(pt.var_marginals, ex.var_marginals),
            "tv_bestn_exact": marginal_tv(bon_marg, ex.var_marginals),
            "co_tv_gibbs": co_tv(gb.co_cluster, ex.co_cluster),
            "co_tv_pt": co_tv(pt.co_cluster, ex.co_cluster),
        })

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"  mean entropy (constrained)= {mean('entropy'):.3f} nats")
    print(f"  mean TV(gibbs,exact)      = {mean('tv_gibbs_exact'):.3f}")
    print(f"  mean TV(PT,exact)         = {mean('tv_pt_exact'):.3f}")
    print(f"  mean TV(best-of-N,exact)  = {mean('tv_bestn_exact'):.3f}")
    print(f"  mean co_tv(gibbs)         = {mean('co_tv_gibbs'):.4f}")
    print(f"  mean co_tv(PT)            = {mean('co_tv_pt'):.4f}")

    gibbs_traps = mean("tv_gibbs_exact") > 0.10
    pt_crosses = mean("tv_pt_exact") < 0.05
    beats_bon = mean("tv_pt_exact") < mean("tv_bestn_exact") - 0.02

    sc = Scorecard("A: constrained generation (soft all-different)")
    sc.set("G0", "pass", "target is the constrained-posterior marginals (sampling)")
    sc.set("G1", "pass",
           "all-different couples every hole pair; no chain/tree factorisation")
    sc.set("G1b", "pass",
           "normaliser is a weighted permanent (#P-hard); no automaton/DP escape "
           "(unlike regular/CF constraints solved by Outlines/xgrammar)")
    sc.set("G2", "partial",
           f"exact enumerable here (L^n≤2M, L={args.L}); permanent infeasible at "
           "real list lengths / vocab")
    sc.set("G3", "pass" if gibbs_traps else "partial",
           f"mean TV(single-T Gibbs)={mean('tv_gibbs_exact'):.3f} "
           f"(frustration from repulsion={args.repel})")
    sc.set("G4", "pass" if pt_crosses else "partial",
           f"mean TV(PT)={mean('tv_pt_exact'):.3f} vs Gibbs "
           f"{mean('tv_gibbs_exact'):.3f}")
    sc.set("G5", "partial" if beats_bon else "fail",
           f"PT {'beats' if beats_bon else 'ties/loses to'} best-of-N "
           f"(TV {mean('tv_pt_exact'):.3f} vs {mean('tv_bestn_exact'):.3f}); "
           "the constraint is imposed, not an intrinsic task metric")
    sc.set("G6", "unknown", "needs scaling study + real hardware constants")
    sc.set("G7", "partial",
           f"all-different is DENSE all-to-all repulsion (n·(n−1)/2 edges) — high "
           "degree; a clean win needs a sparse/local constraint instead")
    sc.set("G8", "partial",
           "diverse-list generation tolerates approx samples, but #P normaliser "
           "means rare feasible modes may matter")
    sc.set("G9", "unknown", "depends on whether a real list-generation task lands here")
    print("\n" + sc.summary())
    print("\n  VERDICT: the all-different constraint gives a clean G0/G1/G1b "
          "(sampling, frustrated, #P normaliser, no solver escape) — the best of "
          "the four on those gates. The risks are G7 (dense all-to-all repulsion "
          "is not hardware-local) and whether a REAL scored task (diverse "
          "generation / ranking) naturally produces it. Promote if G3/G4 hold "
          "and a real list-generation benchmark can be wired in.")

    out = write_report(args.out, {
        "config": vars(args),
        "rows": rows,
        "aggregate": {k: mean(k) for k in
                      ["entropy", "tv_gibbs_exact", "tv_pt_exact",
                       "tv_bestn_exact", "co_tv_gibbs", "co_tv_pt"]},
        "scorecard": sc.to_dict(),
    })
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
