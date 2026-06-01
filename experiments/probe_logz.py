"""Route 2: does the PARTITION FUNCTION expose hardness the marginals hide?

The marginal-inference route to "PT/TSU uniquely wins where the classical
gauntlet fails" is closed (probe_gauntlet_dial, probe_ais_vs_pt): annealed
importance sampling (AIS) — a classical single-machine sampler — recovers the
marginals everywhere PT does, even as its effective sample size (ESS) collapses,
because the targets we can validate are *peaked* (one dominant basin) and the
dominant-basin marginals survive a degenerate weight set.

But AIS's importance weights estimate ONE more quantity the marginals do not:
the log-partition function logZ (free energy). And logẐ_AIS = logZ_0 +
logsumexp(logw) − log N is **unbiased in Ẑ but biased LOW in logẐ by Jensen**,
with the bias growing as the weights degenerate (ESS → 0). So logZ is the
*sensitive* observable: AIS can nail the marginals while its logẐ is badly
biased, precisely when ESS collapses. A replica-exchange estimator (PT +
thermodynamic integration, logZ = logZ_0 + ∫_0^1 ⟨U⟩_β dβ) equilibrates each
temperature rung via swaps and should recover exact logZ where AIS's single
annealing pass supercools.

The decisive cell for route 2: AIS marginal Hellinger PASSES (< 0.05) while AIS
logẐ error is large AND PT-TI logZ tracks exact. If found, logZ / free-energy
estimation is the task where the thermodynamic-sampling thesis survives the
classical gauntlet; if AIS logẐ also tracks, route 2 closes and the negative is
airtight on every observable.

Two experiments, both with EXACT logZ ground truth at small n:

  EXP ZF — fully-connected q-state ferromagnetic Potts (clean first-order;
    CAVEAT: has an MF/sector-sum solver escape, so it demonstrates the MECHANISM
    with clean gold, it is NOT a no-escape instance). Fix w near/above the
    transition, sweep the AIS temperature count: logẐ bias should grow as the
    schedule under-resolves (ESS ↓) while PT-TI stays accurate.

  EXP ZG — frustrated random-field spin-glass Potts (the no-clean-escape
    instance from probe_gauntlet_dial, where single-T Gibbs is metastable). Sweep
    the coupling: does AIS logẐ fail where its marginals passed? Does PT-TI cross?

Pure numpy; reuses probe_hw_common. Runs on the system python (no GPU):

    LD_LIBRARY_PATH="<zlib>:/run/opengl-driver/lib:..." \\
        .venv/bin/python experiments/probe_logz.py --out results/probe_logz.json
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
    _cond_logits,
    _energy_all,
    _gumbel_argmax,
    exact_marginals,
    gauntlet_ais,
    mean_hellinger,
    potts_pairwise,
)


def mean_overlap(cc: np.ndarray | None) -> float:
    if cc is None:
        return float("nan")
    iu = np.triu_indices_from(cc, k=1)
    return float(np.mean(cc[iu]))


# ───────────────────────── instance builders ───────────────────────────────


def build_ferro_potts(n: int, L: int, w: float, field: float):
    """Fully-connected ferromagnetic q=L Potts; coupling w/(n-1) per edge so the
    per-node field is O(w); small ``field`` on state 0 breaks the q-fold symmetry."""
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


def build_spinglass_potts(n: int, L: int, w: float, rng: np.random.Generator,
                          repel_fraction: float = 0.5, field_scale: float = 1.0,
                          p_edge: float = 1.0):
    """Frustrated random Potts: Gaussian unary fields (×field_scale) + ±w·eye
    couplings (sign antiferro with prob ``repel_fraction``) over an ER(p_edge)
    graph. Mirrors probe_gauntlet_dial's no-clean-escape instance."""
    unary = [field_scale * rng.standard_normal(L) for _ in range(n)]
    eye = np.eye(L, dtype=np.float64)
    edges, coup = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if p_edge < 1.0 and rng.random() >= p_edge:
                continue
            sign = -1.0 if rng.random() < repel_fraction else 1.0
            edges.append((i, j))
            coup.append(sign * w * eye)
    if not edges:  # guarantee at least one edge
        edges.append((0, 1))
        coup.append(w * eye)
    return potts_pairwise([L] * n, unary, edges, coup)


# ───────────────── PT + thermodynamic-integration logZ ──────────────────────


def pt_ti_logz(graph, *, betas: np.ndarray, n_chains: int = 256,
               burn_in: int = 500, n_measure: int = 200, seed: int = 0,
               co_cluster: bool = True):
    """Replica exchange across an explicit β-grid (must include 0 and 1), with
    per-rung ⟨U⟩ measured after burn-in. Returns logZ via
    logZ = logZ_0 + ∫_0^1 ⟨U⟩_β dβ (trapezoid), plus the β=1 marginals.

    Swaps cross barriers → each rung equilibrates → ⟨U⟩_β is unbiased; the only
    residual error is β-grid resolution of the integral (controllable), unlike
    AIS whose single annealing pass supercools at a fixed schedule.
    """
    rng = np.random.default_rng(seed)
    N = graph.n_vars
    Lr = len(betas)
    cards = graph.cards
    states = np.stack(
        [np.stack([rng.integers(0, c, size=n_chains) for c in cards], axis=1)
         for _ in range(Lr)]
    ).astype(np.int64)  # (Lr, C, N)
    order = list(range(N))
    usum = np.zeros(Lr, dtype=np.float64)  # accumulated mean-U per rung
    ucount = 0
    base_samples: list[np.ndarray] = []
    t0 = time.time()
    total = burn_in + n_measure
    for it in range(total):
        for lvl in range(Lr):
            b = betas[lvl]
            for i in order:
                logits = _cond_logits(graph, states[lvl], i) * b
                states[lvl][:, i] = _gumbel_argmax(logits, rng)
        # even/odd adjacent swaps
        start = it % 2
        for lvl in range(start, Lr - 1, 2):
            ua = _energy_all(graph, states[lvl])
            ub = _energy_all(graph, states[lvl + 1])
            delta = (betas[lvl] - betas[lvl + 1]) * (ub - ua)
            accept = rng.random(n_chains) < np.exp(np.minimum(0.0, delta))
            tmp = states[lvl][accept].copy()
            states[lvl][accept] = states[lvl + 1][accept]
            states[lvl + 1][accept] = tmp
        if it >= burn_in:
            for lvl in range(Lr):
                usum[lvl] += float(np.mean(_energy_all(graph, states[lvl])))
            ucount += 1
            base_samples.append(states[-1].copy())  # betas[-1] == 1
    mean_u = usum / max(1, ucount)
    # trapezoidal ∫⟨U⟩dβ (manual: np.trapz is deprecated/renamed in numpy 2.x)
    integral = float(np.sum(0.5 * (mean_u[1:] + mean_u[:-1]) * np.diff(betas)))
    logz0 = float(np.sum(np.log(np.asarray(cards, dtype=np.float64))))
    logz = logz0 + integral
    samp = np.concatenate(base_samples, axis=0)
    marg = []
    for v in range(N):
        counts = np.bincount(samp[:, v], minlength=cards[v]).astype(np.float64)
        marg.append(counts / counts.sum())
    cc = None
    if co_cluster and len(set(cards)) == 1:
        cc = np.zeros((N, N))
        for a in range(N):
            for b2 in range(a + 1, N):
                cc[a, b2] = cc[b2, a] = float(np.mean(samp[:, a] == samp[:, b2]))
        np.fill_diagonal(cc, 1.0)
    return {"log_z": logz, "marginals": marg, "co_cluster": cc,
            "mean_u": mean_u.tolist(), "wall_s": time.time() - t0}


def _ais_aggregate(graph, *, n_chains, n_temps, instances, seed):
    """Run AIS over ``instances`` seeds; return mean/std of logẐ + mean ESS,
    marginal Hellinger (vs exact passed separately) handled by caller."""
    logzs, esss, margs, ccs = [], [], [], []
    for r in range(instances):
        a = gauntlet_ais(graph, n_chains=n_chains, n_temps=n_temps,
                         co_cluster=True, seed=seed + 100 * r)
        logzs.append(a.metric["log_z"])
        esss.append(a.metric["ess_frac"])
        margs.append([np.asarray(m) for m in a.metric["marginals"]])
        ccs.append(np.asarray(a.metric["co_cluster"]) if a.metric["co_cluster"] else None)
    return logzs, esss, margs, ccs


# ───────────────────────── EXP ZF: ferro schedule ──────────────────────────


def exp_ferro_schedule(args) -> dict:
    print(f"\n===== EXP ZF: logZ vs AIS schedule resolution (ferro, n={args.n}, "
          f"q={args.L}, w={args.zf_w}, field={args.field}, inst={args.instances}) =====")
    g = build_ferro_potts(args.n, args.L, args.zf_w, args.field)
    ex = exact_marginals(g, max_states=args.exact_budget, co_cluster=True)
    betas = np.linspace(0.0, 1.0, args.pt_rungs)
    pt = pt_ti_logz(g, betas=betas, n_chains=args.chains, burn_in=args.burn_in,
                    n_measure=args.n_measure, seed=args.seed)
    pt_err = abs(pt["log_z"] - ex.log_z)
    pt_h = mean_hellinger(pt["marginals"], ex.var_marginals)
    print(f"  exact logZ = {ex.log_z:.4f} (logZ/n = {ex.log_z/args.n:.4f}); exact "
          f"overlap = {mean_overlap(ex.co_cluster):.3f}")
    print(f"  PT-TI ({args.pt_rungs} rungs): logZ = {pt['log_z']:.4f}  |Δ| = {pt_err:.4f} "
          f"({pt_err/args.n:.4f}/site)  margH = {pt_h:.4f}")
    print(f"\n  {'temps':>6} {'AIS_logZ':>9} {'|ΔlogZ|':>8} {'Δ/site':>7} {'AIS_ess':>8} "
          f"{'AIS_margH':>9} {'AIS_ovlp':>8}")
    rows = []
    for nt in args.zf_temps:
        logzs, esss, margs, ccs = _ais_aggregate(
            g, n_chains=args.ais_chains, n_temps=nt, instances=args.instances, seed=args.seed)
        ais_logz = float(np.mean(logzs))
        derr = abs(ais_logz - ex.log_z)
        mh = float(np.mean([mean_hellinger(m, ex.var_marginals) for m in margs]))
        ov = float(np.mean([mean_overlap(c) for c in ccs]))
        rows.append({"temps": nt, "ais_logz": ais_logz, "ais_logz_std": float(np.std(logzs)),
                     "abs_err": derr, "err_per_site": derr / args.n,
                     "ess": float(np.mean(esss)), "marg_h": mh, "overlap": ov})
        print(f"  {nt:>6} {ais_logz:>9.4f} {derr:>8.4f} {derr/args.n:>7.4f} "
              f"{float(np.mean(esss)):>8.4f} {mh:>9.4f} {ov:>8.3f}")
    return {"exact_logz": ex.log_z, "exact_overlap": mean_overlap(ex.co_cluster),
            "pt_logz": pt["log_z"], "pt_abs_err": pt_err, "pt_marg_h": pt_h,
            "pt_rungs": args.pt_rungs, "rows": rows}


# ──────────────────── EXP ZG: spin-glass coupling sweep ─────────────────────


def exp_spinglass_coupling(args) -> list[dict]:
    print(f"\n===== EXP ZG: does logZ expose hardness marginals hide? (spin-glass, "
          f"n={args.n}, q={args.L}, repel={args.repel_fraction}, inst={args.instances}) =====")
    print("  looking for: AIS margH < 0.05 (marginals PASS) but |ΔlogZ_AIS| large "
          "while PT-TI logZ tracks exact → logZ is the sensitive observable")
    print(f"\n  {'w':>5} {'exactlogZ':>9} {'AIS_logZ':>9} {'|ΔAIS|':>7} {'PT_logZ':>9} "
          f"{'|ΔPT|':>6} {'AIS_ess':>8} {'AIS_mH':>7} {'PT_mH':>6} {'sgGibbsH':>8}")
    betas = np.linspace(0.0, 1.0, args.pt_rungs)
    rows = []
    for w in args.zg_weights:
        acc = {k: [] for k in ("elz", "alz", "ae", "plz", "pe", "ess", "amh",
                               "pmh", "ov_e", "ov_a", "ov_p")}
        for r in range(args.instances):
            seed = args.seed + 100 * r
            rng = np.random.default_rng(seed)
            g = build_spinglass_potts(args.n, args.L, w, rng,
                                      repel_fraction=args.repel_fraction,
                                      field_scale=args.field_scale, p_edge=args.p_edge)
            ex = exact_marginals(g, max_states=args.exact_budget, co_cluster=True)
            a = gauntlet_ais(g, n_chains=args.ais_chains, n_temps=args.ais_temps,
                             co_cluster=True, seed=seed)
            pt = pt_ti_logz(g, betas=betas, n_chains=args.chains, burn_in=args.burn_in,
                            n_measure=args.n_measure, seed=seed)
            a_m = [np.asarray(m) for m in a.metric["marginals"]]
            a_cc = np.asarray(a.metric["co_cluster"]) if a.metric["co_cluster"] else None
            acc["elz"].append(ex.log_z)
            acc["alz"].append(a.metric["log_z"])
            acc["ae"].append(abs(a.metric["log_z"] - ex.log_z))
            acc["plz"].append(pt["log_z"])
            acc["pe"].append(abs(pt["log_z"] - ex.log_z))
            acc["ess"].append(a.metric["ess_frac"])
            acc["amh"].append(mean_hellinger(a_m, ex.var_marginals))
            acc["pmh"].append(mean_hellinger(pt["marginals"], ex.var_marginals))
            acc["ov_e"].append(mean_overlap(ex.co_cluster))
            acc["ov_a"].append(mean_overlap(a_cc))
            acc["ov_p"].append(mean_overlap(pt["co_cluster"]))
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        rows.append({"w": w, **m})
        # sgGibbsH placeholder via AIS marg (kept compact); single-T failure is
        # documented in probe_gauntlet_dial — here we focus on logZ.
        print(f"  {w:>5.2f} {m['elz']:>9.3f} {m['alz']:>9.3f} {m['ae']:>7.3f} "
              f"{m['plz']:>9.3f} {m['pe']:>6.3f} {m['ess']:>8.4f} {m['amh']:>7.4f} "
              f"{m['pmh']:>6.4f} {'-':>8}")
    return rows


def _verdict_ZF(z: dict) -> str:
    rows = z["rows"]
    few = min(rows, key=lambda r: r["temps"])
    many = max(rows, key=lambda r: r["temps"])
    grew = few["abs_err"] - many["abs_err"]
    if few["abs_err"] > 0.5 and z["pt_abs_err"] < few["abs_err"] and grew > 0.2:
        return (f"AIS logẐ BIAS under-resolution: at {few['temps']} temps |ΔlogZ|="
                f"{few['abs_err']:.3f} ({few['err_per_site']:.3f}/site, ESS {few['ess']:.3f}, "
                f"margH {few['marg_h']:.3f}) shrinks to {many['abs_err']:.3f} at {many['temps']} "
                f"temps; PT-TI |ΔlogZ|={z['pt_abs_err']:.3f}. logZ is schedule-sensitive where "
                f"the marginals/overlap were not (caveat: ferromagnet has an MF escape).")
    return (f"AIS logẐ tracks even under-resolved (min-temp |Δ|={few['abs_err']:.3f}, "
            f"PT-TI |Δ|={z['pt_abs_err']:.3f}) — the n={ 'small' } ferro barrier is too weak to "
            f"bias logZ; route-2 mechanism not demonstrated here, push w / n / q.")


def _verdict_ZG(rows: list[dict], fail_h: float = 0.05) -> str:
    hits = [r for r in rows
            if r["amh"] < fail_h and r["ae"] > 0.5 and r["pe"] < r["ae"] - 0.2]
    if hits:
        r = max(hits, key=lambda x: x["ae"] - x["pe"])
        return (f"ROUTE-2 POSITIVE: at w={r['w']} AIS marginals PASS (H={r['amh']:.3f}<{fail_h}) "
                f"but AIS logẐ is biased |Δ|={r['ae']:.3f} (ESS {r['ess']:.4f}) while PT-TI "
                f"tracks |Δ|={r['pe']:.3f}. logZ / free-energy is the observable where "
                f"replica exchange beats classical AIS on a no-clean-escape instance.")
    worst = max(rows, key=lambda x: x["ae"]) if rows else None
    if worst:
        return (f"Route-2 NEGATIVE on this sweep: no cell with passing marginals AND a large "
                f"AIS-logZ bias that PT fixes. Worst AIS |ΔlogZ|={worst['ae']:.3f} at w={worst['w']} "
                f"(margH {worst['amh']:.3f}, PT |Δ|={worst['pe']:.3f}). Peaked target → AIS logẐ "
                f"survives ESS collapse; the negative extends from marginals to logZ.")
    return "no rows"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/probe_logz.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--L", type=int, default=4, help="q = number of Potts states")
    ap.add_argument("--field", type=float, default=0.15, help="ferro symmetry-breaking field")
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--chains", type=int, default=256, help="PT chains per rung")
    ap.add_argument("--ais-chains", type=int, default=256)
    ap.add_argument("--burn-in", type=int, default=500)
    ap.add_argument("--n-measure", type=int, default=200)
    ap.add_argument("--pt-rungs", type=int, default=33, help="PT-TI β-grid points (incl 0 and 1)")
    ap.add_argument("--ais-temps", type=int, default=400, help="AIS temps for EXP ZG")
    # EXP ZF
    ap.add_argument("--zf-w", type=float, default=6.0, help="ferro coupling (ordered region)")
    ap.add_argument("--zf-temps", default="5,10,20,40,80,160,400",
                    help="AIS temperature ladder for EXP ZF schedule sweep")
    # EXP ZG
    ap.add_argument("--zg-weights", default="0.5,1,2,4,8,16")
    ap.add_argument("--repel-fraction", type=float, default=0.5)
    ap.add_argument("--field-scale", type=float, default=1.0)
    ap.add_argument("--p-edge", type=float, default=1.0)
    ap.add_argument("--exact-budget", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-ferro", action="store_true")
    ap.add_argument("--skip-spinglass", action="store_true")
    args = ap.parse_args()
    args.zf_temps = [int(x) for x in args.zf_temps.split(",")]
    args.zg_weights = [float(x) for x in args.zg_weights.split(",")]

    zf = None if args.skip_ferro else exp_ferro_schedule(args)
    if zf:
        print(f"\n  VERDICT(ZF): {_verdict_ZF(zf)}")
    zg = [] if args.skip_spinglass else exp_spinglass_coupling(args)
    if zg:
        print(f"\n  VERDICT(ZG): {_verdict_ZG(zg)}")

    report = {"config": vars(args), "ferro_schedule": zf, "spinglass_coupling": zg,
              "verdict_ZF": _verdict_ZF(zf) if zf else None,
              "verdict_ZG": _verdict_ZG(zg) if zg else None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=lambda o: (
        o.tolist() if isinstance(o, np.ndarray) else float(o)
        if isinstance(o, (np.floating, np.integer)) else str(o))))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
