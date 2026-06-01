"""Is there ANY regime / cost where replica exchange beats classical AIS?

The gauntlet-dial result (probe_gauntlet_dial.py) was a clean negative for the
thermodynamic-sampling thesis on marginal inference: annealed importance
sampling (AIS) — a fully classical, single-machine sampler — recovers the
marginals everywhere parallel tempering (PT) does, at ~14× LESS compute, at
every coupling and every field scale we can validate against exact. The reason:
the random-field spin glass is *peaked* (one dominant basin), and AIS finds the
basin even as its ESS collapses. There was no "all classical methods fail" cell.

AIS has ONE textbook, principled failure mode: a **first-order phase
transition**. When the annealing path β: 0→1 crosses a temperature where two
phases coexist with a free-energy barrier between them, chains *supercool* —
they stay trapped in the metastable phase past the transition — and the
importance weights cannot repair a finite ensemble that never visited the
stable phase. The estimator is then BIASED, not merely high-variance, and more
chains do not fix it (only more temperatures spent near the transition do).
Replica exchange crosses it because the hot replicas tunnel between phases and
carry the stable phase down to β=1.

The cleanest exactly-solvable first-order system is the **fully-connected
q-state ferromagnetic Potts model** (q ≥ 3 → first-order in mean field). This
probe sweeps the coupling through the transition and asks, with EXACT ground
truth at small n:

  EXP P — does AIS develop a measurable BIAS (supercooling) while PT tracks
    exact? Score per-node marginal Hellinger AND the two-point overlap
    ⟨1[x_i = x_j]⟩ (the order parameter; symmetry-robust), and report AIS ESS.

  EXP M — matched compute. At a coupling in the ordered/coexistence region,
    sweep a SHARED budget B (total site-sweeps) and configure AIS and PT to each
    spend exactly B. If PT's correctness survives at AIS-equal cost while AIS
    stays biased, the replica-exchange advantage is real and cost-justified —
    the first validatable evidence for it. If AIS catches up at equal B, the
    negative result stands and is now airtight.

CAVEAT, stated up front: a fully-connected ferromagnet has a classical solver
escape (mean field is asymptotically exact; logZ has an O(n^q) magnetization-
sector closed form). So EXP P demonstrates the AIS-failure MECHANISM with clean
ground truth; it is NOT itself a no-solver-escape instance. The open question it
sets up is whether the same bias reproduces on a FRUSTRATED, no-escape graph
that happens to sit at a first-order transition — which random spin glasses
(probe_gauntlet_dial) do not, because they are peaked, not coexistent.

Pure numpy; reuses probe_hw_common. Runs on the system python (no GPU):

    LD_LIBRARY_PATH="<zlib>:/run/opengl-driver/lib:..." \\
        .venv/bin/python experiments/probe_ais_vs_pt.py \\
            --out results/probe_ais_vs_pt.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_hw_common import (  # noqa: E402
    block_gibbs,
    co_tv,
    exact_marginals,
    gauntlet_ais,
    mean_hellinger,
    parallel_tempering,
    potts_pairwise,
)


def mean_overlap(cc: np.ndarray | None) -> float:
    """Mean off-diagonal co-clustering P(x_i = x_j) — the Potts order parameter."""
    if cc is None:
        return float("nan")
    iu = np.triu_indices_from(cc, k=1)
    return float(np.mean(cc[iu]))


def build_ferro_potts(n: int, L: int, w: float, field: float):
    """Fully-connected ferromagnetic q=L Potts; coupling w/(n-1) per edge so the
    total field per node is O(w); small ``field`` on state 0 breaks the q-fold
    symmetry so the per-node marginal is an informative order parameter."""
    unary = [np.zeros(L) for _ in range(n)]
    for u in unary:
        u[0] = field
    eye = np.eye(L, dtype=np.float64)
    edges, coup = [], []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            coup.append((w / (n - 1)) * eye)
    return potts_pairwise([L] * n, unary, edges, coup)


# ───────────────────────── EXP P: first-order sweep ────────────────────────


def exp_first_order(args) -> list[dict]:
    print(f"\n===== EXP P: first-order Potts sweep (n={args.n}, q={args.L}, "
          f"field={args.field}, instances={args.instances}) =====")
    print("  AIS anneals β:0→1 from the disordered phase; if it supercools past the "
          "transition its overlap/marginal stays BELOW exact (bias), PT should track")
    print(f"\n  {'w':>5} {'ovlp_exact':>10} {'ovlp_AIS':>9} {'ovlp_PT':>8} {'ovlp_Gibbs':>10} "
          f"{'H_AIS':>6} {'H_PT':>6} {'H_Gibbs':>7} {'AIS_ess':>8} {'coTV_AIS':>8} {'coTV_PT':>7}")
    rows = []
    for w in args.weight_list:
        accum = {k: [] for k in ("oe", "oa", "op", "og", "ha", "hp", "hg",
                                 "ess", "ca", "cp")}
        # build_ferro_potts is deterministic → graph + exact identical across
        # instances; compute exact once per w (n=10 brute force is ~25s).
        g = build_ferro_potts(args.n, args.L, w, args.field)
        ex = exact_marginals(g, max_states=args.exact_budget, co_cluster=True)
        for r in range(args.instances):
            seed = args.seed + 100 * r
            ais = gauntlet_ais(g, n_chains=args.ais_chains, n_temps=args.ais_temps,
                               co_cluster=True, seed=seed)
            pt = parallel_tempering(g, n_levels=args.pt_levels, t_max=max(2 * w, 4.0),
                                    n_chains=args.chains, burn_in=args.burn_in,
                                    n_measure=args.n_measure, co_cluster=True, seed=seed)
            gb = block_gibbs(g, T=1.0, n_chains=args.chains, burn_in=args.burn_in,
                             n_measure=args.n_measure, co_cluster=True, seed=seed)
            ais_m = [np.asarray(m) for m in ais.metric["marginals"]]
            ais_cc = np.asarray(ais.metric["co_cluster"]) if ais.metric["co_cluster"] else None
            accum["oe"].append(mean_overlap(ex.co_cluster))
            accum["oa"].append(mean_overlap(ais_cc))
            accum["op"].append(mean_overlap(pt.co_cluster))
            accum["og"].append(mean_overlap(gb.co_cluster))
            accum["ha"].append(mean_hellinger(ais_m, ex.var_marginals))
            accum["hp"].append(mean_hellinger(pt.var_marginals, ex.var_marginals))
            accum["hg"].append(mean_hellinger(gb.var_marginals, ex.var_marginals))
            accum["ess"].append(ais.metric["ess_frac"])
            accum["ca"].append(co_tv(ais_cc, ex.co_cluster))
            accum["cp"].append(co_tv(pt.co_cluster, ex.co_cluster))
        m = {k: float(np.mean(v)) for k, v in accum.items()}
        rows.append({"w": w, **m})
        print(f"  {w:>5.2f} {m['oe']:>10.3f} {m['oa']:>9.3f} {m['op']:>8.3f} {m['og']:>10.3f} "
              f"{m['ha']:>6.3f} {m['hp']:>6.3f} {m['hg']:>7.3f} {m['ess']:>8.4f} "
              f"{m['ca']:>8.3f} {m['cp']:>7.3f}")
    return rows


# ───────────────────────── EXP M: matched compute ──────────────────────────


def exp_matched_compute(args) -> list[dict]:
    print(f"\n===== EXP M: matched-compute AIS vs PT (n={args.n}, q={args.L}, "
          f"w={args.match_w}, instances={args.instances}) =====")
    print("  budget B = site-sweeps × chains, held EQUAL for both. AIS: n_temps×chains; "
          "PT: n_levels×total_sweeps×chains. Does PT stay correct at AIS-equal cost?")
    print(f"\n  {'budget':>9} {'AIS_temps':>9} {'PT_sweeps':>9} {'H_AIS':>6} {'H_PT':>6} "
          f"{'coTV_AIS':>8} {'coTV_PT':>7} {'AIS_ess':>8} {'ais_s':>7} {'pt_s':>7}")
    rows = []
    chains = args.chains
    # graph fixed at w=match_w → exact once for the whole sweep.
    g = build_ferro_potts(args.n, args.L, args.match_w, args.field)
    ex = exact_marginals(g, max_states=args.exact_budget, co_cluster=True)
    for S in args.match_budgets:  # PT total sweeps; budget B = pt_levels*S*chains
        B = args.pt_levels * S * chains
        n_temps = max(2, int(round(B / chains)))  # AIS: 1 sweep/temp, same chains
        acc = {k: [] for k in ("ha", "hp", "ca", "cp", "ess", "ta", "tp")}
        for r in range(args.instances):
            seed = args.seed + 100 * r
            t0 = time.time()
            ais = gauntlet_ais(g, n_chains=chains, n_temps=n_temps, co_cluster=True, seed=seed)
            ta = time.time() - t0
            n_meas = max(1, S // 4)
            burn = S - n_meas
            t0 = time.time()
            pt = parallel_tempering(g, n_levels=args.pt_levels, t_max=max(2 * args.match_w, 4.0),
                                    n_chains=chains, burn_in=burn, n_measure=n_meas,
                                    co_cluster=True, seed=seed)
            tp = time.time() - t0
            ais_m = [np.asarray(m) for m in ais.metric["marginals"]]
            ais_cc = np.asarray(ais.metric["co_cluster"]) if ais.metric["co_cluster"] else None
            acc["ha"].append(mean_hellinger(ais_m, ex.var_marginals))
            acc["hp"].append(mean_hellinger(pt.var_marginals, ex.var_marginals))
            acc["ca"].append(co_tv(ais_cc, ex.co_cluster))
            acc["cp"].append(co_tv(pt.co_cluster, ex.co_cluster))
            acc["ess"].append(ais.metric["ess_frac"])
            acc["ta"].append(ta)
            acc["tp"].append(tp)
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        rows.append({"budget": B, "ais_temps": n_temps, "pt_sweeps": S, **m})
        print(f"  {B:>9} {n_temps:>9} {S:>9} {m['ha']:>6.3f} {m['hp']:>6.3f} "
              f"{m['ca']:>8.3f} {m['cp']:>7.3f} {m['ess']:>8.4f} {m['ta']:>7.2f} {m['tp']:>7.2f}")
    return rows


# ──────────────── EXP S: schedule-resolution (forced supercooling) ──────────


def exp_schedule_resolution(args) -> list[dict]:
    """Does AIS supercool when its annealing ladder is UNDER-resolved relative to
    the barrier? Fix w in the ordered region (exact overlap ≈ 1) and throttle
    n_temps down. Few temps → AIS jumps β:0→1 too fast → trapped in the
    disordered phase → overlap stays LOW (bias). PT at MATCHED compute (same
    site-sweep budget split across its levels) should still track. This isolates
    the mechanism: AIS bias is a schedule-resolution failure, cheaply fixable by
    adding temps — which is why EXP P (400 temps) sees no bias."""
    w = args.sched_w
    print(f"\n===== EXP S: schedule-resolution / forced supercooling "
          f"(n={args.n}, q={args.L}, w={w}, instances={args.instances}) =====")
    g = build_ferro_potts(args.n, args.L, w, args.field)
    ex = exact_marginals(g, max_states=args.exact_budget, co_cluster=True)
    oe = mean_overlap(ex.co_cluster)
    print(f"  exact overlap (ordered target) = {oe:.3f}; AIS that supercools stays "
          f"near the disordered value ≈ {1.0/args.L:.3f}")
    print(f"\n  {'temps':>6} {'ovlp_AIS':>9} {'ovlp_PT':>8} {'H_AIS':>6} {'H_PT':>6} "
          f"{'AIS_ess':>8} {'coTV_AIS':>8} {'coTV_PT':>7}")
    rows = []
    chains = args.ais_chains
    for nt in args.sched_temps:
        acc = {k: [] for k in ("oa", "op", "ha", "hp", "ess", "ca", "cp")}
        B = nt * chains  # AIS site-sweep budget (1 sweep/temp)
        S = max(2, int(round(B / (args.pt_levels * args.chains))))  # PT total sweeps
        n_meas = max(1, S // 4)
        burn = max(1, S - n_meas)
        for r in range(args.instances):
            seed = args.seed + 100 * r
            ais = gauntlet_ais(g, n_chains=chains, n_temps=nt, co_cluster=True, seed=seed)
            pt = parallel_tempering(g, n_levels=args.pt_levels, t_max=max(2 * w, 4.0),
                                    n_chains=args.chains, burn_in=burn, n_measure=n_meas,
                                    co_cluster=True, seed=seed)
            ais_m = [np.asarray(m) for m in ais.metric["marginals"]]
            ais_cc = np.asarray(ais.metric["co_cluster"]) if ais.metric["co_cluster"] else None
            acc["oa"].append(mean_overlap(ais_cc))
            acc["op"].append(mean_overlap(pt.co_cluster))
            acc["ha"].append(mean_hellinger(ais_m, ex.var_marginals))
            acc["hp"].append(mean_hellinger(pt.var_marginals, ex.var_marginals))
            acc["ess"].append(ais.metric["ess_frac"])
            acc["ca"].append(co_tv(ais_cc, ex.co_cluster))
            acc["cp"].append(co_tv(pt.co_cluster, ex.co_cluster))
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        rows.append({"temps": nt, "pt_sweeps": S, "ovlp_exact": oe, **m})
        print(f"  {nt:>6} {m['oa']:>9.3f} {m['op']:>8.3f} {m['ha']:>6.3f} {m['hp']:>6.3f} "
              f"{m['ess']:>8.4f} {m['ca']:>8.3f} {m['cp']:>7.3f}")
    return rows


def _verdict_S(rows: list[dict]) -> str:
    if not rows:
        return ""
    oe = rows[0]["ovlp_exact"]
    sc = [r for r in rows if (oe - r["oa"]) > 0.1 and (oe - r["op"]) < 0.05]
    if sc:
        r = min(sc, key=lambda x: x["temps"])
        return (f"MECHANISM CONFIRMED: at {r['temps']} temps AIS supercools (overlap "
                f"{r['oa']:.3f} ≪ exact {oe:.3f}, ESS {r['ess']:.3f}) while PT at MATCHED "
                f"compute tracks (overlap {r['op']:.3f}). The first-order trap is REAL but is "
                f"a schedule-resolution failure — adding temps (EXP P used 400) fixes AIS "
                f"cheaply, which is why no bias survives at practical ladders.")
    return ("Even under-resolved AIS did not supercool at this w — the n-scale barrier is too "
            "weak. Raise w / q / n (but exact ground truth bounds n).")


def _verdict_P(rows: list[dict]) -> str:
    # AIS biased = overlap markedly below exact while PT tracks, on some w.
    biased = [r for r in rows
              if (r["oe"] - r["oa"]) > 0.1 and abs(r["oe"] - r["op"]) < 0.05]
    if biased:
        w0 = biased[0]["w"]
        return (f"AIS SUPERCOOLS: at w≈{w0} exact overlap {biased[0]['oe']:.3f} but AIS "
                f"{biased[0]['oa']:.3f} (bias {biased[0]['oe']-biased[0]['oa']:.3f}) while PT "
                f"{biased[0]['op']:.3f} tracks exact. The first-order failure mode is real and "
                f"PT crosses it — with EXACT ground truth (caveat: ferromagnet has an MF escape).")
    return ("No first-order AIS bias found in this sweep — either the finite-n transition is "
            "too smeared or AIS still tracks. Widen the w grid / raise n / sharpen q.")


def _verdict_M(rows: list[dict]) -> str:
    # PT beats AIS at equal budget = PT coTV lower than AIS coTV at small budgets.
    wins = [r for r in rows if (r["ca"] - r["cp"]) > 0.03]
    if wins:
        r = wins[0]
        return (f"PT BEATS AIS AT EQUAL COMPUTE: at budget {r['budget']} PT coTV {r['cp']:.3f} < "
                f"AIS coTV {r['ca']:.3f} (AIS ESS {r['ess']:.4f}). Replica-exchange advantage "
                f"survives matched FLOPs — first validatable evidence for it.")
    return ("AIS ties or beats PT at every matched budget — the negative result is airtight: "
            "no cost-justified replica-exchange advantage on this instance.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/probe_ais_vs_pt.json")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--L", type=int, default=4, help="q = number of Potts states")
    ap.add_argument("--field", type=float, default=0.15, help="symmetry-breaking field on state 0")
    ap.add_argument("--weight-list", default="0.5,1,1.5,2,3,4,6,8")
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--chains", type=int, default=256)
    ap.add_argument("--burn-in", type=int, default=500)
    ap.add_argument("--n-measure", type=int, default=200)
    ap.add_argument("--pt-levels", type=int, default=16)
    ap.add_argument("--ais-temps", type=int, default=400)
    ap.add_argument("--ais-chains", type=int, default=256)
    ap.add_argument("--match-w", type=float, default=4.0, help="coupling for EXP M")
    ap.add_argument("--match-budgets", default="50,100,200,400,800",
                    help="PT total-sweeps grid; AIS matched to pt_levels×S×chains")
    ap.add_argument("--exact-budget", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-matched", action="store_true")
    ap.add_argument("--skip-first-order", action="store_true")
    ap.add_argument("--skip-schedule", action="store_true")
    ap.add_argument("--sched-w", type=float, default=6.0,
                    help="coupling (ordered region) for EXP S forced-supercooling")
    ap.add_argument("--sched-temps", default="5,10,20,40,80,160,400",
                    help="AIS n_temps ladder for EXP S; PT matched to same budget")
    args = ap.parse_args()
    args.weight_list = [float(x) for x in args.weight_list.split(",")]
    args.match_budgets = [int(x) for x in args.match_budgets.split(",")]
    args.sched_temps = [int(x) for x in args.sched_temps.split(",")]

    p_rows = [] if args.skip_first_order else exp_first_order(args)
    if p_rows:
        print(f"\n  VERDICT(P): {_verdict_P(p_rows)}")
    s_rows = [] if args.skip_schedule else exp_schedule_resolution(args)
    if s_rows:
        print(f"\n  VERDICT(S): {_verdict_S(s_rows)}")
    m_rows = [] if args.skip_matched else exp_matched_compute(args)
    if m_rows:
        print(f"\n  VERDICT(M): {_verdict_M(m_rows)}")

    report = {"config": vars(args), "first_order": p_rows, "schedule_resolution": s_rows,
              "matched_compute": m_rows,
              "verdict_P": _verdict_P(p_rows) if p_rows else None,
              "verdict_S": _verdict_S(s_rows) if s_rows else None,
              "verdict_M": _verdict_M(m_rows) if m_rows else None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=lambda o: (
        o.tolist() if isinstance(o, np.ndarray) else float(o)
        if isinstance(o, (np.floating, np.integer)) else str(o))))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
