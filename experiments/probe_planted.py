"""Probe: large-n PLANTED frustrated Potts glass — the last un-closed route to a
"PT/replica-exchange uniquely beats the classical gauntlet" claim.

Context (read CLAUDE.md "GAUNTLET + AIS-vs-PT FINDINGS"). Three earlier routes to
a no-classical-escape regime are CLOSED:
  • marginal inference (probe_ais_vs_pt.py) — AIS tracks exact at every n≤10;
  • logZ / free-energy (probe_logz.py) — AIS logẐ survives ESS collapse;
  • engineered first-order coexistence (probe_first_order.py) — small-n barriers
    are too weak; AIS crosses every double-well we can build with exact gold.
The unifying obstruction: AIS's failure needs a first-order coexistence barrier
on the annealing path, but every system small enough for exact ground truth
(n≤10) either has a mean-field escape (ferro) or is peaked (spin glass).

This probe attacks the ONE remaining door: go LARGE-n, where exact Z is gone, and
replace exact ground truth with a PLANTED signal. We build a frustrated Potts
glass (random N(0,frust) coupling tables on a sparse graph) and add a planted
reward ``signal`` to the (t*_i, t*_j) entry of every edge table, where t* is a
fixed random "ground-truth" colouring. The planted reward breaks the q!
colour-permutation symmetry, so overlap with t* is an unambiguous recovery handle
WITHOUT needing Z. As ``signal`` grows the model sweeps:
    pure glass (signal=0, unrecoverable)  →  HARD planted phase (glassy, t*
    is the dominant state but barrier-protected)  →  easy (signal large).

The textbook claim from planted statistical physics: in the HARD phase simulated
annealing (≈ AIS) supercools into glassy metastable decoys and fails to find the
plant, while parallel tempering (replica exchange — invented to beat SA on
glasses) recovers it. If TRUE here, this is the legitimate thermodynamic-hardware
figure. If FALSE (AIS recovers wherever PT does, or BOTH fail together in the
glass), the obstruction holds at large n too and Track B must ship framing (i)
("PT vs single-T block-Gibbs at matched temperature count").

Validation WITHOUT exact Z (triangulated):
  1. plant overlap — soft (Σ_v P(x_v=t*_v)/n, baseline 1/q) + hard (per-chain
     fraction of sites matching t*, max over chains = "did ANY chain find it");
  2. clamped-at-plant control — init at t*, run β=1 Gibbs; if overlap/score is
     retained, t* is a CERTIFIED stable deep state, so failure-to-recover is a
     genuine barrier, not an unstable plant;
  3. score(t*) vs each method's mean score — PT-from-random ≈ score(t*) certifies
     PT reached the plant basin; AIS score < score(t*) certifies a metastable
     decoy trap;
  4. full gauntlet (single-T Gibbs, AIS, mean-field, TRW-BP, best-of-N) — a clean
     win requires PT recovers while ALL of these do not.

Honest risk: PT and AIS are both local; deep in the glass BOTH fail and there may
be NO window where PT alone recovers. The sweep maps where each method breaks.

Run (NixOS host, no GPU needed — pure numpy):
    LD_LIBRARY_PATH=... .venv/bin/python experiments/probe_planted.py
    .venv/bin/python experiments/probe_planted.py --smoke   # tiny + fast
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import probe_hw_common as hw


# ────────────────────────────── construction ───────────────────────────────


def build_planted_potts(
    n: int,
    q: int,
    *,
    avg_deg: float,
    signal: float,
    frust: float,
    field_scale: float,
    seed: int,
) -> tuple[hw.FactorGraph, np.ndarray, list[tuple[int, int]]]:
    """Sparse frustrated Potts glass with a planted ground-truth colouring t*.

    • Erdős–Rényi graph with edge prob avg_deg/(n-1) (forced connected-ish by a
      spanning path so no isolated nodes), q colours.
    • Each edge (a,b) gets a random coupling table J[a,b] ∈ R^{q×q},
      entries ~ N(0, frust²) — this is the *glass* (rugged, many decoys).
    • Plant: add ``signal`` to J[a,b][t*_a, t*_b] for every edge → t* is rewarded
      on the specific colour pair it uses, breaking the q! permutation symmetry.
    • Weak per-site fields ~ field_scale·N(0,1) (quenched disorder; default 0).

    Returns (graph, t_star, edges). Log-potential convention (higher score = more
    probable), matching probe_hw_common.
    """
    rng = np.random.default_rng(seed)
    t_star = rng.integers(0, q, size=n)

    # --- sparse graph: spanning path (connectivity) + ER extra edges ---
    edge_set: set[tuple[int, int]] = set()
    perm = rng.permutation(n)
    for k in range(n - 1):
        a, b = int(perm[k]), int(perm[k + 1])
        edge_set.add((min(a, b), max(a, b)))
    p_extra = max(0.0, (avg_deg - 2.0 * (n - 1) / n) / (n - 1))
    iu = np.triu_indices(n, k=1)
    mask = rng.random(iu[0].shape[0]) < p_extra
    for a, b in zip(iu[0][mask], iu[1][mask]):
        edge_set.add((int(a), int(b)))
    edges = sorted(edge_set)

    # --- unary fields ---
    unary = [field_scale * rng.standard_normal(q) for _ in range(n)]

    # --- coupling tables: glass + plant ---
    couplings: list[np.ndarray] = []
    for (a, b) in edges:
        J = frust * rng.standard_normal((q, q))
        J[t_star[a], t_star[b]] += signal
        couplings.append(J)

    graph = hw.potts_pairwise([q] * n, unary, edges, couplings)
    return graph, t_star, edges


# ──────────────────────────────── readouts ─────────────────────────────────


def _overlap_from_samples(samples: np.ndarray, t_star: np.ndarray) -> dict:
    """Hard plant-overlap stats over a batch of integer states (M, n)."""
    match = (samples == t_star[None, :])
    per = match.mean(axis=1)
    return {
        "mean_overlap": float(per.mean()),
        "max_overlap": float(per.max()),
        "frac_ge_0.9": float(np.mean(per >= 0.9)),
    }


def _soft_overlap(marginals, t_star: np.ndarray) -> float:
    """Expected per-site match to the plant under a method's marginals.

    soft = (1/n) Σ_v marg_v[t*_v]; baseline (uniform) = 1/q. The universal,
    method-agnostic recovery handle (works for sample-free MF/BP too).
    """
    return float(np.mean([marginals[v][t_star[v]] for v in range(len(t_star))]))


def _mean_score(graph: hw.FactorGraph, samples: np.ndarray) -> float:
    return float(hw._energy_all(graph, samples).mean())


def _ais_states(
    graph: hw.FactorGraph,
    t_star: np.ndarray,
    *,
    n_chains: int,
    n_temps: int,
    n_sweeps: int,
    seed: int,
) -> dict:
    """AIS that RETURNS the endpoint states + importance weights (the harness
    gauntlet_ais only exposes marginals). Geometric-in-β path, Gibbs transitions;
    each chain's β=1 endpoint is one annealed sample of t. Lets us ask both
    "weighted soft overlap" AND "did ANY annealed chain find the plant".
    """
    rng = np.random.default_rng(seed)
    n = graph.n_vars
    betas = np.linspace(0.0, 1.0, n_temps)
    states = np.stack(
        [rng.integers(0, c, size=n_chains) for c in graph.cards], axis=1
    ).astype(np.int64)
    logw = np.zeros(n_chains, dtype=np.float64)
    order = list(range(n))
    for k in range(1, n_temps):
        logw += (betas[k] - betas[k - 1]) * hw._energy_all(graph, states)
        for _ in range(n_sweeps):
            for i in order:
                logits = hw._cond_logits(graph, states, i) * betas[k]
                states[:, i] = hw._gumbel_argmax(logits, rng)
    m = float(np.max(logw))
    w = np.exp(logw - m)
    w /= w.sum()
    ess = float(1.0 / np.sum(w ** 2) / n_chains)
    per = (states == t_star[None, :]).mean(axis=1)
    marg = []
    for v in range(n):
        mv = np.zeros(graph.cards[v])
        for val in range(graph.cards[v]):
            mv[val] = w[states[:, v] == val].sum()
        marg.append(mv)
    return {
        "soft_overlap": _soft_overlap(marg, t_star),
        "weighted_overlap": float(np.sum(w * per)),
        "max_overlap": float(per.max()),
        "mean_score": float(np.sum(w * hw._energy_all(graph, states))),
        "ess": ess,
    }


# ──────────────────────────────── experiment ───────────────────────────────


def exp_signal_sweep(
    *,
    n: int,
    q: int,
    avg_deg: float,
    frust: float,
    field_scale: float,
    signal_list: list[float],
    n_instances: int,
    samp_chains: int,
    samp_burn: int,
    samp_measure: int,
    pt_levels: int,
    pt_tmax: float,
    ais_temps: int,
    ais_sweeps: int,
    bon: int,
    seed0: int,
) -> dict:
    """Sweep the planted signal at fixed large n; map where each method recovers.

    Per signal, averaged over n_instances planted draws:
      ex(plant)   = score(t*) / n   (per-site planted score, the target energy)
      clamp_ov    = overlap retained after β=1 Gibbs started AT t* (stability)
      <method>_so = soft plant-overlap (baseline 1/q); samplers also report max
    """
    baseline = 1.0 / q
    rows = []
    for signal in signal_list:
        agg: dict[str, list[float]] = {}

        def add(key, val):
            agg.setdefault(key, []).append(val)

        for inst in range(n_instances):
            seed = seed0 + 1000 * inst
            graph, t_star, edges = build_planted_potts(
                n, q, avg_deg=avg_deg, signal=signal, frust=frust,
                field_scale=field_scale, seed=seed,
            )
            e_star = graph.score(t_star) / n
            add("e_star", e_star)
            add("n_edges", len(edges))

            # 1) clamped-at-plant control (is t* a stable deep state?)
            clamp = hw.block_gibbs(
                graph, T=1.0, n_chains=samp_chains, burn_in=samp_burn // 2,
                n_measure=samp_measure, co_cluster=False, seed=seed + 1,
                init=t_star,
            )
            add("clamp_ov", _overlap_from_samples(clamp.samples, t_star)["mean_overlap"])
            add("clamp_score", _mean_score(graph, clamp.samples) / n)

            # 2) single-T Gibbs (random start, the weak incumbent)
            g = hw.block_gibbs(
                graph, T=1.0, n_chains=samp_chains, burn_in=samp_burn,
                n_measure=samp_measure, co_cluster=False, seed=seed + 2,
            )
            gov = _overlap_from_samples(g.samples, t_star)
            add("gibbs_so", _soft_overlap(g.var_marginals, t_star))
            add("gibbs_max", gov["max_overlap"])
            add("gibbs_score", _mean_score(graph, g.samples) / n)

            # 3) parallel tempering (random start, the hope)
            pt = hw.parallel_tempering(
                graph, n_levels=pt_levels, t_max=pt_tmax, n_chains=samp_chains,
                burn_in=samp_burn, n_measure=samp_measure, co_cluster=False,
                seed=seed + 3,
            )
            ptov = _overlap_from_samples(pt.samples, t_star)
            add("pt_so", _soft_overlap(pt.var_marginals, t_star))
            add("pt_max", ptov["max_overlap"])
            add("pt_score", _mean_score(graph, pt.samples) / n)

            # 4) AIS (random start, the question — matched compute to PT)
            ais = _ais_states(
                graph, t_star, n_chains=samp_chains, n_temps=ais_temps,
                n_sweeps=ais_sweeps, seed=seed + 4,
            )
            add("ais_so", ais["soft_overlap"])
            add("ais_max", ais["max_overlap"])
            add("ais_score", ais["mean_score"] / n)
            add("ais_ess", ais["ess"])

            # 5) gauntlet: mean-field, TRW-BP, best-of-N (marginals → soft)
            mf = hw.gauntlet_mean_field(graph)
            add("mf_so", _soft_overlap([np.asarray(m) for m in mf.metric["marginals"]], t_star))
            trw = hw.gauntlet_trw_bp(graph)
            add("trw_so", _soft_overlap([np.asarray(m) for m in trw.metric["marginals"]], t_star))
            bo = hw.gauntlet_best_of_n(graph, n=bon, seed=seed + 5)
            add("bon_so", _soft_overlap([np.asarray(m) for m in bo.metric["marginals"]], t_star))

        row = {"signal": signal, "baseline": baseline}
        for k, vals in agg.items():
            row[k] = float(np.mean(vals))
        rows.append(row)
    return {"baseline": baseline, "rows": rows}


def _verdict_signal(res: dict, recov_norm: float = 0.7, clamp_min: float = 0.95) -> str:
    """Scan for a signal where t* is CERTIFIED stable, PT recovers, and the whole
    rest of the gauntlet (AIS, single-T Gibbs, mean-field, TRW-BP, best-of-N)
    does NOT. That window = route SURVIVES. Empty = obstruction holds at large n.
    """
    base = res["baseline"]

    def norm(so: float) -> float:  # normalised recovery in [0,1]
        return (so - base) / (1.0 - base + 1e-12)

    windows = []
    for r in res["rows"]:
        if r.get("clamp_ov", 0.0) < clamp_min:
            continue  # plant not certified stable here → uninformative
        pt_rec = norm(r["pt_so"]) >= recov_norm
        others = {
            "ais": norm(r["ais_so"]),
            "gibbs": norm(r["gibbs_so"]),
            "mf": norm(r["mf_so"]),
            "trw": norm(r["trw_so"]),
            "bon": norm(r["bon_so"]),
        }
        others_fail = all(v < recov_norm for v in others.values())
        if pt_rec and others_fail:
            worst = max(others, key=others.get)
            windows.append((r["signal"], norm(r["pt_so"]), worst, others[worst]))

    if windows:
        s, ptn, worst, wn = windows[0]
        return (
            f"VERDICT(PL): ROUTE SURVIVES — at signal={s:.2f} t* is clamp-certified "
            f"stable AND PT recovers (norm-overlap {ptn:.2f}) while the WHOLE "
            f"gauntlet fails (strongest competitor {worst} norm {wn:.2f} < {recov_norm}). "
            f"{len(windows)} such signal(s). PT/replica-exchange uniquely crosses the "
            f"planted glass barrier — the legitimate thermodynamic-hardware figure."
        )
    # No window: report whether PT even separates from AIS anywhere
    sep = max(
        (norm(r["pt_so"]) - norm(r["ais_so"]) for r in res["rows"]), default=0.0
    )
    best_pt = max((norm(r["pt_so"]) for r in res["rows"]), default=0.0)
    return (
        f"VERDICT(PL): NO PT-UNIQUE WINDOW — obstruction holds at large n. Best "
        f"PT−AIS soft-overlap separation across the sweep is {sep:+.3f} "
        f"(max PT norm-overlap {best_pt:.2f}). Either AIS recovers wherever PT does, "
        f"or both fail together in the glass. Ship framing (i): PT vs single-T "
        f"block-Gibbs at matched temperature count; AIS is a stronger classical "
        f"baseline that also crosses."
    )


def _print_sweep(res: dict, *, n: int, q: int, avg_deg: float, frust: float) -> None:
    print(
        f"\n===== EXP PL: planted Potts-glass signal sweep "
        f"(n={n}, q={q}, avg_deg={avg_deg}, frust={frust}) ====="
    )
    print(
        "  recovery = soft plant-overlap (1/n Σ_v P(x_v=t*_v)); baseline 1/q="
        f"{res['baseline']:.3f}. clamp_ov = overlap retained from a t*-start (t* stable?)."
    )
    print(
        "  sig  e*/n  clamp  cl_sc  gib_so gib_mx  pt_so  pt_mx  ais_so ais_mx ais_ess "
        " mf_so trw_so bon_so   pt_sc ais_sc"
    )
    for r in res["rows"]:
        print(
            f"  {r['signal']:.2f}"
            f" {r['e_star']:6.2f}"
            f" {r['clamp_ov']:6.3f}"
            f" {r['clamp_score']:6.2f}"
            f" {r['gibbs_so']:6.3f} {r['gibbs_max']:6.3f}"
            f" {r['pt_so']:6.3f} {r['pt_max']:6.3f}"
            f" {r['ais_so']:6.3f} {r['ais_max']:6.3f} {r['ais_ess']:6.3f}"
            f" {r['mf_so']:6.3f} {r['trw_so']:6.3f} {r['bon_so']:6.3f}"
            f"  {r['pt_score']:6.2f} {r['ais_score']:6.2f}"
        )


# ──────────────────────────────────── main ─────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument("--avg-deg", type=float, default=3.0)
    ap.add_argument("--frust", type=float, default=1.0)
    ap.add_argument("--field-scale", type=float, default=0.0)
    ap.add_argument("--signal-list", type=str, default="0.0,0.5,1.0,1.5,2.0,3.0,4.0")
    ap.add_argument("--n-instances", type=int, default=3)
    ap.add_argument("--samp-chains", type=int, default=128)
    ap.add_argument("--samp-burn", type=int, default=400)
    ap.add_argument("--samp-measure", type=int, default=80)
    ap.add_argument("--pt-levels", type=int, default=8)
    ap.add_argument("--pt-tmax", type=float, default=8.0)
    ap.add_argument("--ais-temps", type=int, default=400)
    ap.add_argument("--ais-sweeps", type=int, default=8)
    ap.add_argument("--bon", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/probe_planted.json")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny/fast settings to validate the machinery")
    args = ap.parse_args()

    if args.smoke:
        args.n = 16
        args.signal_list = "0.0,1.0,2.0"
        args.n_instances = 2
        args.samp_chains = 48
        args.samp_burn = 120
        args.samp_measure = 40
        args.ais_temps = 80
        args.ais_sweeps = 2
        args.bon = 1024

    signal_list = [float(s) for s in args.signal_list.split(",") if s.strip()]

    res = exp_signal_sweep(
        n=args.n, q=args.q, avg_deg=args.avg_deg, frust=args.frust,
        field_scale=args.field_scale, signal_list=signal_list,
        n_instances=args.n_instances, samp_chains=args.samp_chains,
        samp_burn=args.samp_burn, samp_measure=args.samp_measure,
        pt_levels=args.pt_levels, pt_tmax=args.pt_tmax,
        ais_temps=args.ais_temps, ais_sweeps=args.ais_sweeps,
        bon=args.bon, seed0=args.seed,
    )
    _print_sweep(res, n=args.n, q=args.q, avg_deg=args.avg_deg, frust=args.frust)
    verdict = _verdict_signal(res)
    print("\n  " + verdict + "\n")

    payload = {
        "config": vars(args),
        "signal_sweep": res,
        "verdict": verdict,
    }
    out = hw.write_report(args.out, payload)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
