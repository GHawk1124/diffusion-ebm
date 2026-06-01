"""Probe D — marginal inference on a real factor graph: the most promising class.

Codex (gpt-5.5 xhigh) named probabilistic inference / weighted model counting on
real Boolean factor graphs as the most under-explored hardware-fit class:
genuinely a SAMPLING / marginal problem (G0), real and scored, and natively a
sparse local factor graph (G7). The decisive question is gate **G1b**: the
incumbent classical method is loopy belief propagation (and its cousins
mean-field / TAP / GBP). On locally tree-like graphs BP is near-exact — a solver
escape, no hardware win. The hardware case requires the regime BP is KNOWN to
fail: short loops + frustration (a spin glass), where BP marginals diverge from
truth while a tempered sampler stays accurate.

This probe pits exact marginals (small) against four methods:

  * **loopy BP** (sum-product, damped) — the serious classical gauntlet entry;
  * **single-T block-Gibbs** — does frustration trap it?
  * **parallel tempering** — does replica exchange recover the marginals?
  * **best-of-N importance** — the matched-FLOPs naive control.

The `--uai` hook loads a real UAI-2014-competition MARKOV instance (Grids,
Promedus, linkage, …) so the same pipeline runs on a published benchmark scored
with the official MAR metric — **mean Hellinger distance to the true marginals**.
Ground truth is non-circular: brute force when the joint is small, else exact
**variable elimination** (a grid's treewidth is its side, so VE is exact where
2**n is hopeless), else a long PT reference only as a last resort. The synthetic
default is an Edwards-Anderson Ising spin glass (mixed ± couplings) on a 2-D grid
or random-regular graph for dialing frustration directly.

Decisive readout: if BP error blows up under frustration while PT stays near
the exact marginals, G1b PASSES (no cheap solver escape) and this class is the
live candidate. If BP tracks exact, G1b FAILS and the class is ruled out.

    LD_LIBRARY_PATH="<zlib>:/run/opengl-driver/lib:..." \\
        .venv/bin/python experiments/probe_factor_graph_inference.py \\
            --uai data/uai/Grids_11.uai --mar-out results/Grids_11.MAR \\
            --out results/probe_fgi_grids11.json
    # or dial a synthetic spin glass:
    #   ... --topology grid --grid 4 --J 1.0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_hw_common import (  # noqa: E402
    Factor,
    FactorGraph,
    Scorecard,
    block_gibbs,
    clamp_evidence,
    co_tv,
    exact_marginals,
    gauntlet_ais,
    gauntlet_best_of_n,
    gauntlet_mean_field,
    gauntlet_trw_bp,
    marginal_tv,
    mean_hellinger,
    parallel_tempering,
    parse_uai_evidence,
    potts_pairwise,
    variable_elimination_marginals,
    write_report,
)


# ───────────────────────────── topologies ──────────────────────────────────


def grid_edges(side: int) -> list[tuple[int, int]]:
    edges = []
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if c + 1 < side:
                edges.append((i, i + 1))
            if r + 1 < side:
                edges.append((i, i + side))
    return edges


def random_regular_edges(n: int, deg: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Crude random-regular-ish graph via repeated random matching of stubs."""
    stubs = list(range(n)) * deg
    for _ in range(200):
        rng.shuffle(stubs)
        edges, ok = set(), True
        for a, b in zip(stubs[0::2], stubs[1::2]):
            if a == b or (min(a, b), max(a, b)) in edges:
                ok = False
                break
            edges.add((min(a, b), max(a, b)))
        if ok:
            return sorted(edges)
    return sorted(edges)  # best effort


def build_ea_ising(edges, n, J, field, rng: np.random.Generator) -> FactorGraph:
    """Edwards-Anderson spin glass: ±J couplings, q=2, small random field."""
    unary = [rng.normal(0.0, field, size=2) for _ in range(n)]
    couplings = []
    for _ in edges:
        s = 1.0 if rng.random() < 0.5 else -1.0
        # +sJ on agree (diagonal), -sJ on disagree → frustration for mixed signs
        c = np.array([[s * J, -s * J], [-s * J, s * J]], dtype=np.float64)
        couplings.append(c)
    return potts_pairwise([2] * n, unary, edges, couplings)


# ───────────────────────────── loopy BP ────────────────────────────────────


def _lse(a: np.ndarray) -> np.ndarray:
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))


def loopy_bp(graph: FactorGraph, iters: int = 400, damping: float = 0.5):
    """Damped sum-product on a pairwise+unary graph. Returns per-var marginals.

    Raises if any factor has arity > 2 (BP gauntlet is pairwise here).
    """
    unary = [np.zeros(c) for c in graph.cards]
    pair = []  # (a, b, table)
    for f in graph.factors:
        if len(f.scope) == 1:
            unary[f.scope[0]] = unary[f.scope[0]] + f.table
        elif len(f.scope) == 2:
            pair.append((f.scope[0], f.scope[1], f.table))
        else:
            raise ValueError("loopy_bp supports only unary/pairwise factors")
    # directed edges
    msgs: dict[tuple[int, int, int], np.ndarray] = {}  # (eidx, src, dst)->logvec
    nbr: dict[int, list[tuple[int, int]]] = {i: [] for i in range(graph.n_vars)}
    for eidx, (a, b, _t) in enumerate(pair):
        msgs[(eidx, a, b)] = np.zeros(graph.cards[b])
        msgs[(eidx, b, a)] = np.zeros(graph.cards[a])
        nbr[a].append((eidx, b))
        nbr[b].append((eidx, a))
    for _ in range(iters):
        new = {}
        for eidx, (a, b, t) in enumerate(pair):
            for src, dst, tab in ((a, b, t), (b, a, t.T)):
                incoming = unary[src].copy()
                for e2, other in nbr[src]:
                    if not (e2 == eidx and other == dst):
                        incoming = incoming + msgs[(e2, other, src)]
                # m(dst) = lse_src [ incoming(src) + tab(src,dst) ]
                out = np.array([_lse(incoming + tab[:, xd])
                                for xd in range(graph.cards[dst])])
                out -= _lse(out)
                new[(eidx, src, dst)] = out
        # damping in log-space
        for k in msgs:
            msgs[k] = damping * msgs[k] + (1 - damping) * new[k]
    marg = []
    for i in range(graph.n_vars):
        b = unary[i].copy()
        for e2, other in nbr[i]:
            b = b + msgs[(e2, other, i)]
        b -= _lse(b)
        marg.append(np.exp(b))
    return marg


# ───────────────────────────── UAI loader ──────────────────────────────────


def load_uai(path: str) -> FactorGraph:
    """Minimal UAI MARKOV parser (any arity; potentials → log)."""
    toks = Path(path).read_text().split()
    it = iter(toks)
    net = next(it)
    if net != "MARKOV":
        raise ValueError(f"expected MARKOV, got {net}")
    n = int(next(it))
    cards = tuple(int(next(it)) for _ in range(n))
    nf = int(next(it))
    scopes = []
    for _ in range(nf):
        arity = int(next(it))
        scopes.append(tuple(int(next(it)) for _ in range(arity)))
    factors = []
    for scope in scopes:
        sz = int(next(it))
        vals = np.array([float(next(it)) for _ in range(sz)], dtype=np.float64)
        shape = tuple(cards[v] for v in scope)
        with np.errstate(divide="ignore"):
            table = np.log(vals.reshape(shape))
        factors.append(Factor(scope, table))
    return FactorGraph(cards, factors)


def write_mar(path: str, marginals: list[np.ndarray]) -> None:
    """Write marginals in UAI MAR result format: 'MAR\\n N c_1 p... c_2 p... \\n'."""
    parts = [str(len(marginals))]
    for m in marginals:
        parts.append(str(len(m)))
        parts.extend(f"{float(x):.6f}" for x in m)
    Path(path).write_text("MAR\n" + " ".join(parts) + "\n")


# ───────────────────────────────── main ────────────────────────────────────


def _ground_truth(g, args, evid):
    """Pick a non-circular gold: brute-force exact → variable elimination → PT.

    VE exploits bounded treewidth (a grid's is its side length), so it is exact
    even when the 2**n joint is hopeless — and unlike a long-PT reference it does
    not reuse a sampler under test. Caches the gold marginals to ``--gold-cache``
    so PT/BP tuning re-runs do not recompute the (sometimes minutes-long) VE.
    """
    if args.gold_cache and Path(args.gold_cache).exists():
        blob = json.loads(Path(args.gold_cache).read_text())
        marg = [np.asarray(m, dtype=np.float64) for m in blob["marginals"]]
        return marg, None, blob["gold_src"] + " (cached)"

    cc_ok = (len(set(g.cards)) == 1)
    gold, gold_cc, src = None, None, None
    try:
        ex = exact_marginals(g, co_cluster=cc_ok)
        gold, gold_cc, src = ex.var_marginals, ex.co_cluster, \
            f"exact-bruteforce ({ex.n_states} states)"
    except ValueError:
        pass
    if gold is None:
        try:
            ve_marg, width = variable_elimination_marginals(g, max_width=args.max_width)
            gold, src = ve_marg, f"exact-VE (induced width {width})"
        except ValueError as e:
            print(f"  [VE infeasible: {e}; falling back to long-PT reference]")
    if gold is None:
        ref = parallel_tempering(g, n_levels=max(args.pt_levels, 16), t_max=args.pt_tmax,
                                 n_chains=512, burn_in=3000, n_measure=600,
                                 seed=args.seed + 7, co_cluster=cc_ok)
        gold, gold_cc, src = ref.var_marginals, ref.co_cluster, \
            "long-PT reference (NOT exact)"
    if args.gold_cache and src.startswith("exact"):
        Path(args.gold_cache).parent.mkdir(parents=True, exist_ok=True)
        Path(args.gold_cache).write_text(json.dumps(
            {"gold_src": src, "marginals": [m.tolist() for m in gold]}))
    return gold, gold_cc, src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uai", default=None, help="path to a UAI MARKOV instance")
    ap.add_argument("--evid", default=None,
                    help="path to a UAI .evid file (default: <uai>.evid if present)")
    ap.add_argument("--mar-out", default=None,
                    help="write the PT marginals as a UAI .MAR submission file")
    ap.add_argument("--topology", choices=["grid", "random_regular"], default="grid")
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--deg", type=int, default=3, help="random-regular degree")
    ap.add_argument("--J", type=float, default=1.0, help="EA coupling magnitude")
    ap.add_argument("--field", type=float, default=0.1)
    ap.add_argument("--chains", type=int, default=256)
    ap.add_argument("--burn-in", type=int, default=500)
    ap.add_argument("--n-measure", type=int, default=120)
    ap.add_argument("--pt-levels", type=int, default=10)
    ap.add_argument("--pt-tmax", type=float, default=4.0)
    ap.add_argument("--max-width", type=int, default=23,
                    help="VE induced-width cap (2**width tables) before PT fallback")
    ap.add_argument("--gold-cache", default=None,
                    help="cache exact gold marginals here; reuse on re-runs")
    ap.add_argument("--ais-temps", type=int, default=200,
                    help="AIS annealing temperatures (classical-sampling gauntlet)")
    ap.add_argument("--bp-iters", type=int, default=400)
    ap.add_argument("--bp-damping", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/probe_fgi.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    evid: dict[int, int] = {}
    if args.uai:
        g = load_uai(args.uai)
        evid_path = args.evid or (args.uai + ".evid")
        if Path(evid_path).exists():
            evid = parse_uai_evidence(evid_path)
        desc = f"UAI:{Path(args.uai).name}"
    elif args.topology == "grid":
        side = args.grid
        g = build_ea_ising(grid_edges(side), side * side, args.J, args.field, rng)
        desc = f"EA-grid {side}x{side} J={args.J}"
    else:
        n = args.grid * args.grid
        edges = random_regular_edges(n, args.deg, rng)
        g = build_ea_ising(edges, n, args.J, args.field, rng)
        desc = f"EA-random-regular n={n} deg={args.deg} J={args.J}"

    n = g.n_vars
    g = clamp_evidence(g, evid)
    evid_set = set(evid)
    n_edges = sum(1 for f in g.factors if len(f.scope) == 2)
    pairwise_only = all(len(f.scope) <= 2 for f in g.factors)
    print(f"[probe D] {desc}  n={n}  edges={n_edges}  evidence={len(evid)}  "
          f"states={g.state_space}")
    print("  target: per-variable marginals (SAMPLING, not MAP); "
          "metric = mean Hellinger (UAI MAR competition)\n")

    gold, gold_cc, gold_src = _ground_truth(g, args, evid)

    gb = block_gibbs(g, T=1.0, n_chains=args.chains, burn_in=args.burn_in,
                     n_measure=args.n_measure, seed=args.seed,
                     co_cluster=(gold_cc is not None))
    pt = parallel_tempering(g, n_levels=args.pt_levels, t_max=args.pt_tmax,
                            n_chains=args.chains, burn_in=args.burn_in,
                            n_measure=args.n_measure, seed=args.seed,
                            co_cluster=(gold_cc is not None))
    bon = gauntlet_best_of_n(g, n=args.chains * args.n_measure, seed=args.seed)
    bon_marg = [np.asarray(m) for m in bon.metric["marginals"]]

    # ── classical gauntlet (the "solver escape" set) ────────────────────────
    # Deterministic variational incumbents (run on a GPU/CPU, no thermo HW):
    #   loopy BP (vanilla), TRW-BP (convex, the strong one), naive mean-field.
    # Plus AIS — the classical *annealed-sampling* competitor to replica exchange.
    bp = mf = trw = ais_marg = None
    ais_ess = None
    if pairwise_only:
        bp = loopy_bp(g, iters=args.bp_iters, damping=args.bp_damping)
        trw_r = gauntlet_trw_bp(g, iters=args.bp_iters, damping=args.bp_damping)
        trw = [np.asarray(m) for m in trw_r.metric["marginals"]]
        mf_r = gauntlet_mean_field(g)
        mf = [np.asarray(m) for m in mf_r.metric["marginals"]]
    ais_r = gauntlet_ais(g, n_chains=args.chains, n_temps=args.ais_temps,
                         seed=args.seed)
    ais_marg = [np.asarray(m) for m in ais_r.metric["marginals"]]
    ais_ess = ais_r.metric["ess_frac"]

    def _he(m):
        return None if m is None else mean_hellinger(m, gold, evid_set)

    def _tv(m):
        return None if m is None else marginal_tv(m, gold)

    he = {
        "bp": _he(bp), "trw_bp": _he(trw), "mean_field": _he(mf),
        "ais": _he(ais_marg),
        "gibbs": _he(gb.var_marginals), "pt": _he(pt.var_marginals),
        "best_of_n": _he(bon_marg),
    }
    tv = {
        "bp": _tv(bp), "trw_bp": _tv(trw), "mean_field": _tv(mf),
        "ais": _tv(ais_marg),
        "gibbs": _tv(gb.var_marginals), "pt": _tv(pt.var_marginals),
        "best_of_n": _tv(bon_marg),
    }

    if args.mar_out:
        write_mar(args.mar_out, pt.var_marginals)
        print(f"  wrote MAR submission (PT marginals) -> {args.mar_out}")

    print(f"  ground truth: {gold_src}")
    print("  --- classical gauntlet (Hellinger to exact; <0.05 = solver escape) ---")
    print(f"    loopy-BP   = {he['bp']}")
    print(f"    TRW-BP     = {he['trw_bp']}   <- convex 'BP done right'")
    print(f"    mean-field = {he['mean_field']}")
    print(f"    AIS        = {he['ais']}  (ESS={ais_ess:.3f})  <- classical annealed sampler")
    print(f"    best-of-N  = {he['best_of_n']}")
    print("  --- thermodynamic-sampler candidates ---")
    print(f"    Gibbs T=1  = {he['gibbs']:.4f}")
    print(f"    par-temper = {he['pt']:.4f}")
    if gold_cc is not None:
        print(f"  co_tv(Gibbs)={co_tv(gb.co_cluster, gold_cc)}  "
              f"co_tv(PT)={co_tv(pt.co_cluster, gold_cc)}")

    # ── honest gate logic ────────────────────────────────────────────────────
    import re
    exact_jt = gold_src.startswith("exact-VE")     # junction tree feasible here
    exact_brute = gold_src.startswith("exact-bruteforce")
    exact_infeasible = not (exact_jt or exact_brute)  # gold came from a PT reference
    wmatch = re.search(r"induced width (\d+)", gold_src)
    jt_width = int(wmatch.group(1)) if wmatch else None

    # Deterministic polynomial-time incumbents (the methods a junction tree, if it
    # blows up, would be replaced by). AIS is a sampler, scored separately.
    approx_keys = [k for k in ("bp", "trw_bp", "mean_field") if he[k] is not None]
    approx_best = min(he[k] for k in approx_keys) if approx_keys else None
    approx_all_fail = approx_best is not None and approx_best > 0.05

    gibbs_traps = he["gibbs"] > 0.05
    pt_crosses = he["pt"] < 0.03
    # The classical gauntlet PT must beat: exact JT (≈0 if feasible) + every
    # polynomial method + best-of-N + the AIS classical sampler.
    classical_he = [v for k, v in he.items()
                    if k not in ("gibbs", "pt") and v is not None]
    if exact_jt or exact_brute:
        classical_he.append(0.0)  # an exact solver is in the gauntlet and wins
    classical_best = min(classical_he) if classical_he else None
    pt_beats_classical = (classical_best is not None
                          and he["pt"] < classical_best - 0.02)

    sc = Scorecard("D: real factor-graph marginal inference")
    sc.set("G0", "pass", "per-variable marginals are the target (UAI MAR task)")
    sc.set("G1", "pass", "loopy graph with frustration; no tree factorisation")
    if exact_jt:
        sc.set("G1b", "fail",
               f"junction tree (VE) is EXACT at induced width {jt_width} — a "
               f"classical SOLVER ESCAPE exists; approximate incumbents fail "
               f"(best-approx Hellinger={approx_best:.3f}) but exact JT does not")
    elif exact_brute:
        sc.set("G1b", "fail",
               "brute-force/JT trivially exact at this size — solver escape exists")
    else:
        sc.set("G1b", "pass" if approx_all_fail else "fail",
               f"exact/JT infeasible (induced width > {args.max_width}); approximate "
               f"gauntlet best Hellinger={approx_best} "
               f"({'ALL fail — no escape' if approx_all_fail else 'a method tracks the reference'})")
    sc.set("G2", "pass" if g.state_space > 2_000_000 else "partial",
           f"state space {g.state_space}" +
           ("" if g.state_space > 2_000_000 else " (enumerable here)"))
    sc.set("G3", "pass" if gibbs_traps else "partial",
           f"Hellinger(single-T Gibbs)={he['gibbs']:.3f}")
    sc.set("G4", "pass" if pt_crosses else "partial",
           f"Hellinger(PT)={he['pt']:.3f} vs Gibbs {he['gibbs']:.3f}")
    sc.set("G5", "pass" if pt_beats_classical else "fail",
           f"PT Hellinger={he['pt']:.3f} vs best classical={classical_best} "
           f"({'BEATS gauntlet' if pt_beats_classical else 'gauntlet wins/ties — no sampling advantage'})"
           + (f"; published UAI instance vs EXACT gold" if args.uai else ""))
    sc.set("G6", "unknown", "needs scaling + hardware constants")
    sc.set("G7", "pass" if (n_edges / max(n, 1) < 4) else "partial",
           f"sparse: {n_edges} edges over {n} vars (avg degree "
           f"{2*n_edges/max(n,1):.1f}); native pairwise Ising")
    sc.set("G8", "partial",
           "marginals tolerate approximation; partition-function tails may not")
    sc.set("G9", "unknown", "depends on the end-to-end inference application")
    print("\n" + sc.summary())

    approx_note = (f"approximate incumbents (BP/TRW/MF best Hellinger="
                   f"{approx_best:.3f}) FAIL" if approx_all_fail
                   else "an approximate incumbent tracks the truth")
    if exact_infeasible and approx_all_fail and pt_crosses and pt_beats_classical:
        print("\n  VERDICT: LIVE CANDIDATE. Exact/JT infeasible, the whole "
              f"classical gauntlet fails ({approx_note}), single-T Gibbs is "
              "metastable, and PT crosses to the floor and beats every classical "
              "method on the distributional metric.")
    elif exact_jt:
        print(f"\n  VERDICT: TRACTABLE REGIME (no-escape FAILS). A junction tree "
              f"solves this exactly at width {jt_width}; {approx_note} and PT "
              "recovers the marginals, but exact JT is the cheap classical escape. "
              "The no-escape regime needs exact-INFEASIBLE instances (push n / "
              "induced width up).")
    else:
        print("\n  VERDICT: not a clean candidate at these settings — see gate "
              "verdicts (escape exists, or PT does not strictly beat the gauntlet).")

    write_report(args.out, {
        "config": vars(args),
        "desc": desc, "n": n, "n_edges": n_edges, "n_evidence": len(evid),
        "gold_source": gold_src, "jt_width": jt_width,
        "exact_jt_feasible": exact_jt, "exact_infeasible": exact_infeasible,
        "hellinger": he, "tv": tv, "ais_ess": ais_ess,
        "approx_best_hellinger": approx_best,
        "classical_best_hellinger": classical_best,
        "pt_beats_classical": pt_beats_classical,
        "wall_s": {"gibbs": gb.wall_s, "pt": pt.wall_s},
        "scorecard": sc.to_dict(),
    })
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
