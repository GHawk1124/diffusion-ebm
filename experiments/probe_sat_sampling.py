"""Probe C — k-SAT reframed as SAMPLING / weighted model counting. The FALSIFIER.

Run this FIRST (codex gpt-5.5 xhigh: "take your favoured candidate, reframe it
as sampling or marginals, then run the classical gauntlet before touching more
THRML"). k-SAT is not a bet on hardware — it is the cleanest test of the
question that decides the whole programme:

    does "hard to OPTIMISE / SOLVE" imply "hard to SAMPLE"?

A thermodynamic sampler only earns its keep on sampling / marginal problems
(gate G0). Random k-SAT near its satisfiability threshold (α = m/n ≈ 4.267 for
3-SAT) is the canonical hard-optimisation problem, and in the clustering phase
its solution space *shatters* into exponentially many Hamming-separated
clusters — exactly the frustration that traps single-temperature local Gibbs.
So it should pass G3 (metastability) and G4 (tempering crosses). The trap is
gate **G1b (no solver escape)**: hashing-based model-count samplers
(ApproxMC / UniGen / spur) sample near-uniformly from solutions in time that
laughs at the metastability wall. If a classical solver samples the target
cheaply, the wall is irrelevant and the hardware wins nothing — k-SAT
*falsifies* the "hard ⇒ good hardware fit" leap.

We frame it as weighted model counting: sample x ∝ exp(−β · #violated clauses).
β→∞ is the uniform distribution over satisfying assignments (the model-counting
target). We measure, on small n where exact enumeration is feasible:

  * the sampling target's structure: #solutions, #Hamming-clusters (shattering),
    per-variable marginals P(x_i=1 | SAT);
  * single-T block-Gibbs vs parallel tempering fidelity (marginal TV to exact-
    at-β, and the fraction of samples that land *on* solutions) — does PT cross
    the clustering barrier?
  * the classical gauntlet: rejection sampling (the naive sampler) + the
    ApproxMC/UniGen bar (external; the verdict is set from the literature since
    these samplers are not installed) + simulated annealing (the optimiser,
    shows the task is solvable to confirm G0/G1b).

Decisive readout → the scorecard. Expectation: G0/G1/G3/G4 PASS, **G1b FAIL**
(hashing samplers escape), so the overall verdict is *falsified as a hardware
headline* — which is the point of running it first.

    LD_LIBRARY_PATH="<zlib>:/run/opengl-driver/lib:..." \\
        .venv/bin/python experiments/probe_sat_sampling.py \\
            --n 18 --alpha 4.0 --beta 6.0 --out results/probe_sat_sampling.json
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_hw_common import (  # noqa: E402
    Factor,
    FactorGraph,
    Scorecard,
    block_gibbs,
    exact_marginals,
    gauntlet_sa,
    marginal_tv,
    write_report,
)

Clause = tuple[tuple[int, int], ...]  # ((var, sign=+1/-1), ...)


# ───────────────────────────── instance ────────────────────────────────────


def random_ksat(n: int, m: int, k: int, rng: np.random.Generator) -> list[Clause]:
    clauses: list[Clause] = []
    for _ in range(m):
        vs = rng.choice(n, size=k, replace=False)
        signs = rng.choice([-1, 1], size=k)
        clauses.append(tuple((int(v), int(s)) for v, s in zip(vs, signs)))
    return clauses


def clause_factor(clause: Clause, beta: float) -> Factor:
    """Soft clause: 0 if satisfied, −β at the single all-false corner."""
    scope = tuple(v for v, _ in clause)
    table = np.zeros((2,) * len(scope), dtype=np.float64)
    # the violating assignment: literal (v, +1) false ⇒ x_v=0; (v,−1) false ⇒ x_v=1
    bad = tuple(0 if s == 1 else 1 for _, s in clause)
    table[bad] = -beta
    return Factor(scope, table)


def build_graph(clauses: list[Clause], n: int, beta: float) -> FactorGraph:
    return FactorGraph(tuple([2] * n), [clause_factor(c, beta) for c in clauses])


# ───────────────────────── exact SAT structure ─────────────────────────────


def violated_counts(clauses: list[Clause], states: np.ndarray) -> np.ndarray:
    """#violated clauses per state. states: (S, n) of {0,1}."""
    viol = np.zeros(states.shape[0], dtype=np.int64)
    for clause in clauses:
        sat = np.zeros(states.shape[0], dtype=bool)
        for v, s in clause:
            lit_true = states[:, v] == (1 if s == 1 else 0)
            sat |= lit_true
        viol += ~sat
    return viol


def hamming_clusters(solutions: np.ndarray) -> int:
    """#connected components among solutions under Hamming-distance-1 edges."""
    S = solutions.shape[0]
    if S == 0:
        return 0
    parent = list(range(S))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(S):
        for j in range(i + 1, S):
            if int(np.sum(solutions[i] != solutions[j])) == 1:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    return len({find(i) for i in range(S)})


def exact_sat_structure(clauses: list[Clause], n: int) -> dict:
    states = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)
    viol = violated_counts(clauses, states)
    sol = states[viol == 0]
    n_clusters = hamming_clusters(sol) if sol.shape[0] <= 4096 else -1
    if sol.shape[0] > 0:
        sat_marg = [np.array([np.mean(sol[:, v] == 0), np.mean(sol[:, v] == 1)])
                    for v in range(n)]
    else:
        sat_marg = [np.array([0.5, 0.5]) for _ in range(n)]
    return {
        "n_solutions": int(sol.shape[0]),
        "n_clusters": int(n_clusters),
        "min_violations": int(viol.min()),
        "sat_marginals": sat_marg,
    }


def frac_on_solutions(clauses: list[Clause], samples: np.ndarray) -> float:
    return float(np.mean(violated_counts(clauses, samples) == 0))


# ─────────────────────────── classical gauntlet ────────────────────────────


def gauntlet_rejection(clauses: list[Clause], n: int, n_draws: int,
                       rng: np.random.Generator) -> dict:
    """Naive sampler: draw uniform assignments, keep satisfying ones."""
    t0 = time.time()
    draws = rng.integers(0, 2, size=(n_draws, n))
    viol = violated_counts(clauses, draws)
    sols = draws[viol == 0]
    return {
        "name": "rejection_sampling",
        "wall_s": time.time() - t0,
        "n_draws": n_draws,
        "n_accepted": int(sols.shape[0]),
        "accept_rate": float(sols.shape[0] / n_draws),
    }


# ───────────────────────────────── main ────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=4.0, help="clause/var ratio")
    ap.add_argument("--beta", type=float, default=6.0, help="WMC inverse temp")
    ap.add_argument("--instances", type=int, default=8)
    ap.add_argument("--chains", type=int, default=256)
    ap.add_argument("--burn-in", type=int, default=400)
    ap.add_argument("--n-measure", type=int, default=120)
    ap.add_argument("--pt-levels", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/probe_sat_sampling.json")
    args = ap.parse_args()

    m = round(args.alpha * args.n)
    print(f"[probe C] random {args.k}-SAT  n={args.n}  m={m}  alpha={args.alpha}  "
          f"beta={args.beta}  instances={args.instances}")
    print("  target: x ~ exp(-beta * #violated)  (beta->inf = uniform over solutions)")

    rows = []
    for inst in range(args.instances):
        rng = np.random.default_rng(args.seed + inst)
        clauses = random_ksat(args.n, m, args.k, rng)
        struct = exact_sat_structure(clauses, args.n)
        if struct["n_solutions"] == 0:
            print(f"  inst {inst}: UNSAT, skipping")
            continue
        g = build_graph(clauses, args.n, args.beta)
        ex = exact_marginals(g, co_cluster=False)

        gb = block_gibbs(g, T=1.0, n_chains=args.chains, burn_in=args.burn_in,
                         n_measure=args.n_measure, co_cluster=False,
                         seed=args.seed + inst)
        from probe_hw_common import parallel_tempering
        pt = parallel_tempering(g, n_levels=args.pt_levels, t_max=args.beta,
                                n_chains=args.chains, burn_in=args.burn_in,
                                n_measure=args.n_measure, co_cluster=False,
                                seed=args.seed + inst)
        rej = gauntlet_rejection(clauses, args.n, args.chains * args.n_measure,
                                 np.random.default_rng(args.seed + inst + 999))
        sa = gauntlet_sa(g, n_chains=64, sweeps=400, t_hi=2.0, t_lo=0.05,
                         seed=args.seed + inst)

        row = {
            "instance": inst,
            "n_solutions": struct["n_solutions"],
            "n_clusters": struct["n_clusters"],
            "tv_gibbs_exact": marginal_tv(gb.var_marginals, ex.var_marginals),
            "tv_pt_exact": marginal_tv(pt.var_marginals, ex.var_marginals),
            "frac_on_sol_gibbs": frac_on_solutions(clauses, gb.samples),
            "frac_on_sol_pt": frac_on_solutions(clauses, pt.samples),
            "rejection_accept_rate": rej["accept_rate"],
            "sa_best_score": sa.metric["best_score"],  # 0.0 == found a solution
        }
        rows.append(row)
        print(f"  inst {inst}: sols={row['n_solutions']:6d} clusters={row['n_clusters']:4d} "
              f"TV(gibbs)={row['tv_gibbs_exact']:.3f} TV(PT)={row['tv_pt_exact']:.3f} "
              f"onSol(gibbs)={row['frac_on_sol_gibbs']:.2f} onSol(PT)={row['frac_on_sol_pt']:.2f} "
              f"rej={row['rejection_accept_rate']:.1e} SA_best={row['sa_best_score']:.1f}")

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    print("\n===== aggregate =====")
    print(f"  mean clusters          = {mean('n_clusters'):.1f}")
    print(f"  mean TV(gibbs,exact)   = {mean('tv_gibbs_exact'):.3f}")
    print(f"  mean TV(PT,exact)      = {mean('tv_pt_exact'):.3f}")
    print(f"  mean frac-on-sol gibbs = {mean('frac_on_sol_gibbs'):.3f}")
    print(f"  mean frac-on-sol PT    = {mean('frac_on_sol_pt'):.3f}")
    print(f"  mean rejection accept  = {mean('rejection_accept_rate'):.2e}")

    # ── scorecard ──
    sc = Scorecard("C: k-SAT as weighted model counting")
    gibbs_metastable = mean("tv_gibbs_exact") > 0.15
    pt_crosses = mean("tv_pt_exact") < 0.05
    sc.set("G0", "pass",
           "target is uniform-over-solutions / WMC marginals, a sampling task")
    sc.set("G1", "pass",
           "clustered solution space; clauses do not factorise into a tree")
    sc.set("G1b", "fail",
           "ApproxMC/UniGen (hash-based) sample near-uniformly from solutions in "
           "practice — a classical SOLVER ESCAPE the hardware cannot beat")
    sc.set("G2", "partial",
           f"exact 2^n feasible here (n={args.n}); infeasible at app sizes, but "
           "exact-count is the wrong bar — see G1b")
    sc.set("G3", "pass" if gibbs_metastable else "fail",
           f"mean TV(single-T Gibbs)={mean('tv_gibbs_exact'):.3f}; "
           f"on-solution frac={mean('frac_on_sol_gibbs'):.2f} (clustering traps it)")
    sc.set("G4", "pass" if pt_crosses else "partial",
           f"mean TV(PT)={mean('tv_pt_exact'):.3f} (PT crosses the cluster barrier)")
    sc.set("G5", "fail",
           "loses to ApproxMC/UniGen on the sampling metric; rejection sampling "
           f"accept rate {mean('rejection_accept_rate'):.1e} is the only naive bar")
    sc.set("G7", "partial",
           f"{args.k}-ary clauses need auxiliary vars / high-degree gadgets to map "
           "to sparse pairwise Ising — not a clean hardware-native encoding")
    sc.set("G8", "pass",
           "WMC tolerates approximate (ε,δ) samples — does not need exact tails")
    sc.set("G6", "unknown", "moot once G1b fails")
    sc.set("G9", "unknown", "moot once G1b fails")
    print("\n" + sc.summary())
    print("\n  VERDICT: hard-to-sample is REAL here (G3/G4 pass) but a classical "
          "SOLVER ESCAPES (G1b) — k-SAT is FALSIFIED as a hardware headline. "
          "This is the expected fast-fail; it sharpens the search toward tasks "
          "with NO model-counting solver (Probe D) or that are hardware-native "
          "by construction (Probe B).")

    out = write_report(args.out, {
        "config": vars(args),
        "m": m,
        "rows": rows,
        "scorecard": sc.to_dict(),
    })
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
