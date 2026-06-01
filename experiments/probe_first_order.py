"""Route (iii): a genuinely no-escape frustrated graph sitting AT a first-order point.

Background (the obstruction this probe attacks). The marginal-inference and the
logZ routes are both CLOSED (probe_gauntlet_dial, probe_ais_vs_pt, probe_logz):
classical annealed importance sampling (AIS) — single machine, no special
hardware — matches parallel tempering (PT) on every observable we can validate,
because the only small-n systems where we have exact ground truth are either
*peaked* (random spin glass → continuous transition, one dominant basin → AIS
survives) or *mean-field-solvable* (the q≥3 ferromagnet → first-order but has a
sector-sum / gauge escape, AND its barrier is crossable by simply adding AIS
temperatures, EXP ZF). AIS only fails at a genuine **first-order coexistence
barrier**, and the obstruction is: every no-escape instance we could build was
peaked; every first-order instance was MF-solvable.

This probe tries to thread that needle with an explicitly *engineered* first-order
landscape that is NOT a collective spin-ordering transition (so it dodges the
ferro's gauge/MF escape) but a **density-of-states competition** — the textbook
AIS-killer:

    a single DEEP, NARROW planted basin (low energy, ~zero entropy) competing
    with a WIDE, shallow disordered phase (high energy, entropy ≈ n·log q).

Construction — "planted funnel + 3-body sharpener + random background":

    log p(x) ∝  Σ_v h_v(x_v)                                   (weak random fields)
              + j2 · Σ_{(i,j)∈P}  [x_i=t*_i][x_j=t*_j]          (smooth quadratic mouth)
              + j3 · Σ_{(i,j,k)∈T} [x_i=t*_i][x_j=t*_j][x_k=t*_k] (cubic barrier)

with t* a random planted target, P / T random subsets of pairs / triples (so the
match-count k = #{v : x_v = t*_v} is NOT a sufficient statistic → no 1-D / mean-
field reduction → no solver escape; MF and BP fail). The j2 funnel gives a
gradient (a "mouth" PT can slide into); the j3 cubic term sharpens the basin into
a first-order well separated from the disordered phase by an entropic barrier.

Order parameter: overlap m(x) = (1/n) Σ_v [x_v = t*_v] ∈ [1/q, 1]. The two phases
are asymmetric (one planted basin, not q symmetric sectors), so basin occupancy
P(m ≥ thr) is a sharp 0/1 observable with NO sector-symmetry robustness — the
escape that saved the ferro's overlap in EXP S cannot operate here.

Decisive (iii) win = a cell where (a) the EXACT overlap distribution is BIMODAL
(mass at m≈1/q AND m≈1 → genuine first-order coexistence), (b) AIS supercools:
wrong basin occupancy + biased logẐ that does NOT vanish as n_temps grows (the
barrier is intra-temperature, monotone cooling overshoots it), while (c) PT
crosses (basin occupancy + logZ track exact). If instead AIS tracks (or its bias
vanishes with more temps), route (iii) closes and the obstruction is confirmed
empirically: framing (i) "PT vs single-T Gibbs" is the honest shippable claim.

Two experiments, exact gold at n ≤ 9:
  EXP FO-A — coexistence sweep: sweep the planting depth j3, find where exact
    P(basin) ∈ (0.1, 0.9) and the overlap histogram is bimodal; tabulate
    exact / AIS / PT / single-T-Gibbs basin occupancy + logZ + marginal H.
  EXP FO-B — schedule discriminator: at the coexistence cell, sweep AIS n_temps
    (and a matched-sweeps row); first-order ⇒ bias persists, continuous ⇒ it
    vanishes. PT-TI logZ as the replica-exchange reference.

Pure numpy; reuses probe_hw_common. Runs on system python (no GPU):

    LD_LIBRARY_PATH="<zlib>:/run/opengl-driver/lib:..." \\
        .venv/bin/python experiments/probe_first_order.py --out results/probe_first_order.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_hw_common import (  # noqa: E402
    Factor,
    FactorGraph,
    _cond_logits,
    _energy_all,
    _gumbel_argmax,
    block_gibbs,
    mean_hellinger,
)


# ───────────────────────── instance builder ────────────────────────────────


def build_planted_instance(
    n: int,
    L: int,
    *,
    j2: float,
    j3: float,
    p_pair: float,
    n_triples: int,
    bg_scale: float,
    seed: int,
):
    """Planted deep-narrow basin + entropic background (see module docstring).

    Returns (FactorGraph, t_star). Pairs are an ER(p_pair) subset; triples are
    ``n_triples`` random distinct 3-subsets. Random unary fields (×bg_scale) make
    the disordered phase non-uniform and, with the random pair/triple subsets,
    break the match-count sufficiency so no 1-D / mean-field reduction is exact.
    """
    rng = np.random.default_rng(seed)
    t_star = rng.integers(0, L, size=n).astype(np.int64)
    factors: list[Factor] = []

    # weak random unary fields (entropy/roughness of the wide phase)
    for v in range(n):
        factors.append(Factor((v,), bg_scale * rng.standard_normal(L)))

    # planted pairwise funnel: reward matching the planted pattern on a pair
    for i in range(n):
        for jj in range(i + 1, n):
            if rng.random() >= p_pair:
                continue
            tab = np.zeros((L, L), dtype=np.float64)
            tab[t_star[i], t_star[jj]] = j2
            factors.append(Factor((i, jj), tab))

    # planted 3-body sharpener: reward matching on a random triple
    all_triples = list(itertools.combinations(range(n), 3))
    rng.shuffle(all_triples)
    for (i, jj, k) in all_triples[: min(n_triples, len(all_triples))]:
        tab = np.zeros((L, L, L), dtype=np.float64)
        tab[t_star[i], t_star[jj], t_star[k]] = j3
        factors.append(Factor((i, jj, k), tab))

    return FactorGraph(tuple([L] * n), factors), t_star


def build_two_basin_instance(
    n: int,
    L: int,
    *,
    jA: float,
    jB: float,
    nA_triples: int,
    p_pair_B: float,
    bg_scale: float,
    seed: int,
):
    """Two EXTENDED, ASYMMETRIC basins — the textbook entropy-energy first-order
    competition that is AIS's one principled failure mode.

        basin A (DEEP, NARROW): a dense set of 3-body clauses toward a planted
            target t_A. Reward ≈ jA · nA_triples · m_A³ (cubic falloff with the
            overlap m_A = fraction of positions matching t_A) → steep mouth, few
            high-score configs → LOW entropy, but the peak depth jA·nA_triples can
            be made large → a deep well.

        basin B (SHALLOW, WIDE): a 2-body funnel toward an ~orthogonal target t_B
            over an ER(p_pair_B) pair set. Reward ≈ jB · n_pairs · m_B² (quadratic,
            gentler) → wide mouth, many moderate-score configs → HIGH entropy, but
            a shallow peak.

    t_A and t_B are forced to disagree at EVERY position, so m_A + m_B ≤ 1 and (for
    thr ≥ 0.5) the basins are mutually exclusive — a config is in A, in B, or in the
    disordered "none" region. Sweeping jA moves the β=1 equilibrium from B-dominant
    (jA small) through coexistence (F_A ≈ F_B) to A-dominant (jA large).

    The AIS-killer mechanism: at high T (small β, early annealing) entropy dominates
    the free energy, so chains pour into the WIDE basin B; the cubic mouth of A is
    too steep to capture them. As β→1, if A is the deeper well, equilibrium shifts to
    A — but a monotonically-cooled chain trapped in B cannot tunnel across the
    entropic barrier → AIS over-reports B (supercooling). PT keeps replicas at all
    temperatures and re-partitions A/B mass via swaps. Returns (FactorGraph, t_A, t_B).
    """
    rng = np.random.default_rng(seed)
    t_A = rng.integers(0, L, size=n).astype(np.int64)
    t_B = rng.integers(0, L, size=n).astype(np.int64)
    # force t_B to disagree with t_A everywhere → m_A + m_B ≤ 1 (disjoint basins)
    for v in range(n):
        if L > 1:
            while t_B[v] == t_A[v]:
                t_B[v] = rng.integers(0, L)
    factors: list[Factor] = []

    # weak random unary fields (roughness of the disordered phase)
    for v in range(n):
        factors.append(Factor((v,), bg_scale * rng.standard_normal(L)))

    # basin A: deep + narrow, dense 3-body clauses toward t_A
    all_triples = list(itertools.combinations(range(n), 3))
    rng.shuffle(all_triples)
    for (i, jj, k) in all_triples[: min(nA_triples, len(all_triples))]:
        tab = np.zeros((L, L, L), dtype=np.float64)
        tab[t_A[i], t_A[jj], t_A[k]] = jA
        factors.append(Factor((i, jj, k), tab))

    # basin B: shallow + wide, 2-body funnel toward t_B over ER(p_pair_B)
    for i in range(n):
        for jj in range(i + 1, n):
            if rng.random() >= p_pair_B:
                continue
            tab = np.zeros((L, L), dtype=np.float64)
            tab[t_B[i], t_B[jj]] = jB
            factors.append(Factor((i, jj), tab))

    return FactorGraph(tuple([L] * n), factors), t_A, t_B


# ───────────────────────── exact (brute force) ─────────────────────────────


def _all_states(cards: tuple[int, ...]) -> np.ndarray:
    return np.array(list(itertools.product(*[range(c) for c in cards])), dtype=np.int64)


def exact_full(graph: FactorGraph, t_star: np.ndarray, thr: float,
               *, max_states: int) -> dict:
    """Vectorised brute force: logZ, marginals, ⟨overlap⟩, P(basin), P(k) hist."""
    S = graph.state_space
    if S > max_states:
        raise ValueError(f"state space {S} > {max_states}; infeasible")
    states = _all_states(graph.cards)
    scores = _energy_all(graph, states)
    m = float(np.max(scores))
    w = np.exp(scores - m)
    z = float(w.sum())
    p = w / z
    log_z = m + float(np.log(z))
    N = graph.n_vars
    marg = []
    for v in range(N):
        mv = np.zeros(graph.cards[v])
        for val in range(graph.cards[v]):
            mv[val] = p[states[:, v] == val].sum()
        marg.append(mv)
    kmatch = (states == t_star[None, :]).sum(axis=1)
    overlap = kmatch / N
    mean_ov = float(np.sum(p * overlap))
    p_basin = float(np.sum(p[overlap >= thr]))
    p_k = np.zeros(N + 1)
    for kk in range(N + 1):
        p_k[kk] = p[kmatch == kk].sum()
    return {"log_z": log_z, "marginals": marg, "mean_overlap": mean_ov,
            "p_basin": p_basin, "p_k": p_k.tolist(), "n_states": int(S),
            "states": states, "probs": p}


# ───────────────────────── AIS (weighted, full readout) ─────────────────────


def ais_full(graph: FactorGraph, t_star: np.ndarray, thr: float, *,
             n_chains: int, n_temps: int, n_sweeps: int = 1, seed: int = 0) -> dict:
    """AIS with geometric-in-β path; returns logẐ, ESS, marginals and the
    weighted ⟨overlap⟩ / P(basin) / P(k). ``n_sweeps`` Gibbs sweeps per temp."""
    rng = np.random.default_rng(seed)
    N = graph.n_vars
    betas = np.linspace(0.0, 1.0, n_temps)
    states = np.stack(
        [rng.integers(0, c, size=n_chains) for c in graph.cards], axis=1
    ).astype(np.int64)
    logw = np.zeros(n_chains, dtype=np.float64)
    order = list(range(N))
    t0 = time.time()
    for k in range(1, n_temps):
        logw += (betas[k] - betas[k - 1]) * _energy_all(graph, states)
        for _ in range(n_sweeps):
            for i in order:
                logits = _cond_logits(graph, states, i) * betas[k]
                states[:, i] = _gumbel_argmax(logits, rng)
    m = float(np.max(logw))
    w_un = np.exp(logw - m)
    wsum = float(w_un.sum())
    logz0 = float(np.sum(np.log(np.asarray(graph.cards, dtype=np.float64))))
    log_z = logz0 + m + float(np.log(wsum)) - float(np.log(n_chains))
    w = w_un / wsum
    ess = float(1.0 / np.sum(w ** 2) / n_chains)
    marg = []
    for v in range(N):
        mv = np.zeros(graph.cards[v])
        for val in range(graph.cards[v]):
            mv[val] = w[states[:, v] == val].sum()
        marg.append(mv)
    kmatch = (states == t_star[None, :]).sum(axis=1)
    overlap = kmatch / N
    mean_ov = float(np.sum(w * overlap))
    p_basin = float(np.sum(w[overlap >= thr]))
    p_k = np.zeros(N + 1)
    for kk in range(N + 1):
        p_k[kk] = w[kmatch == kk].sum()
    return {"log_z": log_z, "ess": ess, "marginals": marg, "mean_overlap": mean_ov,
            "p_basin": p_basin, "p_k": p_k.tolist(), "wall_s": time.time() - t0,
            "states": states, "weights": w}


# ───────────────── PT + thermodynamic-integration logZ + overlap ────────────


def pt_full(graph: FactorGraph, t_star: np.ndarray, thr: float, *,
            betas: np.ndarray, n_chains: int, burn_in: int, n_measure: int,
            seed: int = 0) -> dict:
    """Replica exchange across an explicit β-grid (incl 0 and 1). logZ via
    thermodynamic integration logZ = logZ_0 + ∫⟨U⟩dβ; β=1 replica supplies
    marginals + ⟨overlap⟩ + P(basin) + P(k)."""
    rng = np.random.default_rng(seed)
    N = graph.n_vars
    Lr = len(betas)
    cards = graph.cards
    states = np.stack(
        [np.stack([rng.integers(0, c, size=n_chains) for c in cards], axis=1)
         for _ in range(Lr)]
    ).astype(np.int64)  # (Lr, C, N)
    order = list(range(N))
    usum = np.zeros(Lr, dtype=np.float64)
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
    integral = float(np.sum(0.5 * (mean_u[1:] + mean_u[:-1]) * np.diff(betas)))
    logz0 = float(np.sum(np.log(np.asarray(cards, dtype=np.float64))))
    log_z = logz0 + integral
    samp = np.concatenate(base_samples, axis=0)
    marg = []
    for v in range(N):
        counts = np.bincount(samp[:, v], minlength=cards[v]).astype(np.float64)
        marg.append(counts / counts.sum())
    kmatch = (samp == t_star[None, :]).sum(axis=1)
    overlap = kmatch / N
    mean_ov = float(np.mean(overlap))
    p_basin = float(np.mean(overlap >= thr))
    p_k = np.zeros(N + 1)
    for kk in range(N + 1):
        p_k[kk] = float(np.mean(kmatch == kk))
    return {"log_z": log_z, "marginals": marg, "mean_overlap": mean_ov,
            "p_basin": p_basin, "p_k": p_k.tolist(), "swap_accept": None,
            "wall_s": time.time() - t0, "samples": samp}


def gibbs_overlap(graph: FactorGraph, t_star: np.ndarray, thr: float, *,
                  n_chains: int, burn_in: int, n_measure: int, seed: int = 0) -> dict:
    """Vanilla single-T (T=1) independent-ensemble Gibbs; overlap readouts."""
    res = block_gibbs(graph, T=1.0, n_chains=n_chains, burn_in=burn_in,
                      n_measure=n_measure, co_cluster=False, seed=seed)
    samp = res.samples
    N = graph.n_vars
    kmatch = (samp == t_star[None, :]).sum(axis=1)
    overlap = kmatch / N
    return {"marginals": res.var_marginals,
            "mean_overlap": float(np.mean(overlap)),
            "p_basin": float(np.mean(overlap >= thr)),
            "wall_s": res.wall_s}


# ───────────────────────── generic-arity mean field ────────────────────────


def generic_mean_field(graph: FactorGraph, *, iters: int = 3000,
                       damping: float = 0.5, tol: float = 1e-9) -> list[np.ndarray]:
    """Naive mean-field coordinate ascent for ARBITRARY-arity factors.

    q_i(a) ∝ exp(Σ_{f∋i} E_{q_{scope(f)\\i}}[θ_f | x_i=a]). If MF nailed this
    instance it would be variationally trivial (a solver escape); frustration +
    the random planted structure should break it (mode-collapse onto one phase)."""
    cards = graph.cards
    q = [np.ones(c) / c for c in cards]
    touch = {i: graph.touching(i) for i in range(graph.n_vars)}
    for _ in range(iters):
        maxd = 0.0
        for i in range(graph.n_vars):
            logb = np.zeros(cards[i])
            for f in touch[i]:
                pos = f.scope.index(i)
                t = np.moveaxis(f.table, pos, 0)  # (card_i, *others)
                others = [v for v in f.scope if v != i]
                contrib = t
                for v in others:
                    contrib = np.tensordot(contrib, q[v], axes=([1], [0]))
                logb = logb + contrib
            logb -= logb.max()
            nq = np.exp(logb)
            nq /= nq.sum()
            nq = damping * q[i] + (1.0 - damping) * nq
            nq /= nq.sum()
            maxd = max(maxd, float(np.abs(nq - q[i]).max()))
            q[i] = nq
        if maxd < tol:
            break
    return q


def _bimodality(p_k: list[float]) -> float:
    """A crude bimodality score: 1 − (mass in the dominant contiguous mode).

    Splits the k-histogram at its global trough; returns the mass on the
    *minority* side of the trough that still contains a local peak. >0.1 ⇒ real
    two-phase coexistence (mass on both the disordered and the ordered side)."""
    a = np.asarray(p_k, dtype=np.float64)
    n = len(a) - 1
    # low side = k <= n//3 (disordered), high side = k >= 2n//3 (ordered)
    lo = float(a[: max(1, n // 3 + 1)].sum())
    hi = float(a[(2 * n) // 3:].sum())
    return float(min(lo, hi))


def two_basin_readout(states: np.ndarray, weights: np.ndarray | None,
                      t_A: np.ndarray, t_B: np.ndarray, thr: float) -> dict:
    """Assign each (weighted) state to basin A / B / none by planted-target overlap.

    Since t_A, t_B disagree everywhere, m_A + m_B ≤ 1, so for thr ≥ 0.5 the
    indicators are disjoint. ``weights`` None ⇒ uniform (raw Gibbs/PT samples)."""
    Ns = states.shape[0]
    N = states.shape[1]
    w = (np.full(Ns, 1.0 / Ns) if weights is None else np.asarray(weights, float))
    ovA = (states == t_A[None, :]).sum(axis=1) / N
    ovB = (states == t_B[None, :]).sum(axis=1) / N
    inA = ovA >= thr
    inB = ovB >= thr
    return {"P_A": float(w[inA].sum()), "P_B": float(w[inB].sum()),
            "P_none": float(w[~(inA | inB)].sum()),
            "mean_ovA": float(np.sum(w * ovA)), "mean_ovB": float(np.sum(w * ovB))}


# ───────────────────────── EXP FO-A: coexistence sweep ──────────────────────


def exp_coexistence(args) -> dict:
    print(f"\n===== EXP FO-A: coexistence sweep (planted basin, n={args.n}, q={args.L}, "
          f"j2={args.j2}, p_pair={args.p_pair}, n_triples={args.n_triples}, "
          f"bg={args.bg_scale}, inst={args.instances}) =====")
    print("  looking for: exact overlap BIMODAL (P(basin)∈(0.1,0.9)) AND AIS misses the "
          "basin (p_basin→0, logẐ biased) while PT tracks → first-order, AIS supercools")
    print(f"\n  {'j3':>5} {'ex_pB':>6} {'bimod':>6} {'ex_ovl':>6} {'ex_logZ':>8} "
          f"{'AIS_pB':>6} {'AISovl':>6} {'AISdlz':>7} {'ESS':>6} "
          f"{'PT_pB':>6} {'PTovl':>6} {'PTdlz':>6} {'Gb_pB':>6} {'MFovl':>6}")
    betas = np.linspace(0.0, 1.0, args.pt_rungs)
    rows = []
    for j3 in args.j3_list:
        acc = {k: [] for k in ("ex_pB", "bimod", "ex_ovl", "ex_lz", "a_pB", "a_ovl",
                               "a_dlz", "ess", "p_pB", "p_ovl", "p_dlz", "g_pB",
                               "mf_ovl", "a_mh", "p_mh")}
        for r in range(args.instances):
            seed = args.seed + 1000 * r
            g, t_star = build_planted_instance(
                args.n, args.L, j2=args.j2, j3=j3, p_pair=args.p_pair,
                n_triples=args.n_triples, bg_scale=args.bg_scale, seed=seed)
            ex = exact_full(g, t_star, args.thr, max_states=args.exact_budget)
            a = ais_full(g, t_star, args.thr, n_chains=args.ais_chains,
                         n_temps=args.ais_temps, n_sweeps=args.ais_sweeps, seed=seed)
            pt = pt_full(g, t_star, args.thr, betas=betas, n_chains=args.chains,
                         burn_in=args.burn_in, n_measure=args.n_measure, seed=seed)
            gb = gibbs_overlap(g, t_star, args.thr, n_chains=args.chains,
                               burn_in=args.burn_in, n_measure=args.n_measure, seed=seed)
            mf = generic_mean_field(g)
            mf_ov = float(np.mean([mf[v][t_star[v]] for v in range(args.n)]))
            acc["ex_pB"].append(ex["p_basin"])
            acc["bimod"].append(_bimodality(ex["p_k"]))
            acc["ex_ovl"].append(ex["mean_overlap"])
            acc["ex_lz"].append(ex["log_z"])
            acc["a_pB"].append(a["p_basin"])
            acc["a_ovl"].append(a["mean_overlap"])
            acc["a_dlz"].append(abs(a["log_z"] - ex["log_z"]))
            acc["ess"].append(a["ess"])
            acc["a_mh"].append(mean_hellinger(a["marginals"], ex["marginals"]))
            acc["p_pB"].append(pt["p_basin"])
            acc["p_ovl"].append(pt["mean_overlap"])
            acc["p_dlz"].append(abs(pt["log_z"] - ex["log_z"]))
            acc["p_mh"].append(mean_hellinger(pt["marginals"], ex["marginals"]))
            acc["g_pB"].append(gb["p_basin"])
            acc["mf_ovl"].append(mf_ov)
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        rows.append({"j3": j3, **m})
        print(f"  {j3:>5.2f} {m['ex_pB']:>6.3f} {m['bimod']:>6.3f} {m['ex_ovl']:>6.3f} "
              f"{m['ex_lz']:>8.3f} {m['a_pB']:>6.3f} {m['a_ovl']:>6.3f} {m['a_dlz']:>7.3f} "
              f"{m['ess']:>6.3f} {m['p_pB']:>6.3f} {m['p_ovl']:>6.3f} {m['p_dlz']:>6.3f} "
              f"{m['g_pB']:>6.3f} {m['mf_ovl']:>6.3f}")
    return {"rows": rows}


# ───────────────────────── EXP FO-B: schedule discriminator ──────────────────


def _pick_coexistence_j3(rows: list[dict]) -> float | None:
    """Pick the j3 with exact P(basin) nearest 0.5 among bimodal cells."""
    cand = [r for r in rows if 0.08 <= r["ex_pB"] <= 0.92 and r["bimod"] >= 0.05]
    if not cand:
        return None
    return min(cand, key=lambda r: abs(r["ex_pB"] - 0.5))["j3"]


def exp_schedule(args, j3: float) -> dict:
    print(f"\n===== EXP FO-B: AIS schedule discriminator at coexistence j3={j3:.2f} "
          f"(n={args.n}, q={args.L}) =====")
    print("  first-order ⇒ AIS basin-occ error & logẐ bias PERSIST as temps grow "
          "(intra-temp barrier); continuous ⇒ they vanish. PT-TI = replica reference.")
    g, t_star = build_planted_instance(
        args.n, args.L, j2=args.j2, j3=j3, p_pair=args.p_pair,
        n_triples=args.n_triples, bg_scale=args.bg_scale, seed=args.seed)
    ex = exact_full(g, t_star, args.thr, max_states=args.exact_budget)
    betas = np.linspace(0.0, 1.0, args.pt_rungs)
    pt = pt_full(g, t_star, args.thr, betas=betas, n_chains=args.chains,
                 burn_in=args.burn_in, n_measure=args.n_measure, seed=args.seed)
    print(f"  exact: P(basin)={ex['p_basin']:.3f}  ⟨overlap⟩={ex['mean_overlap']:.3f}  "
          f"logZ={ex['log_z']:.3f}  bimod={_bimodality(ex['p_k']):.3f}")
    print(f"  PT-TI ({args.pt_rungs} rungs): P(basin)={pt['p_basin']:.3f}  "
          f"⟨overlap⟩={pt['mean_overlap']:.3f}  logZ={pt['log_z']:.3f}  "
          f"|Δ|={abs(pt['log_z']-ex['log_z']):.3f}  margH="
          f"{mean_hellinger(pt['marginals'], ex['marginals']):.3f}")
    print(f"\n  {'temps':>6} {'sweeps':>6} {'AIS_pB':>6} {'AISovl':>6} {'AIS_logZ':>9} "
          f"{'|Δlz|':>6} {'Δ/site':>7} {'ESS':>6} {'margH':>6}")
    rows = []
    configs = [(nt, 1) for nt in args.fob_temps] + \
              [(args.fob_temps[-1], s) for s in args.fob_sweeps if s > 1]
    for nt, sw in configs:
        acc = {k: [] for k in ("pB", "ovl", "lz", "ess", "mh")}
        for r in range(args.instances):
            a = ais_full(g, t_star, args.thr, n_chains=args.ais_chains,
                         n_temps=nt, n_sweeps=sw, seed=args.seed + 100 * r)
            acc["pB"].append(a["p_basin"])
            acc["ovl"].append(a["mean_overlap"])
            acc["lz"].append(a["log_z"])
            acc["ess"].append(a["ess"])
            acc["mh"].append(mean_hellinger(a["marginals"], ex["marginals"]))
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        dlz = abs(m["lz"] - ex["log_z"])
        rows.append({"temps": nt, "sweeps": sw, "p_basin": m["pB"],
                     "overlap": m["ovl"], "ais_logz": m["lz"], "abs_err": dlz,
                     "err_per_site": dlz / args.n, "ess": m["ess"], "marg_h": m["mh"]})
        print(f"  {nt:>6} {sw:>6} {m['pB']:>6.3f} {m['ovl']:>6.3f} {m['lz']:>9.3f} "
              f"{dlz:>6.3f} {dlz/args.n:>7.4f} {m['ess']:>6.3f} {m['mh']:>6.3f}")
    return {"j3": j3, "exact": {"p_basin": ex["p_basin"], "mean_overlap": ex["mean_overlap"],
                                "log_z": ex["log_z"], "p_k": ex["p_k"],
                                "bimod": _bimodality(ex["p_k"])},
            "pt": {"p_basin": pt["p_basin"], "mean_overlap": pt["mean_overlap"],
                   "log_z": pt["log_z"], "abs_err": abs(pt["log_z"] - ex["log_z"]),
                   "marg_h": mean_hellinger(pt["marginals"], ex["marginals"])},
            "rows": rows}


# ──────────────── EXP FO-C: two-basin depth sweep (the (iii) shot) ──────────


def exp_two_basin(args) -> dict:
    print(f"\n===== EXP FO-C: two-basin depth sweep (deep-narrow A vs wide-shallow B, "
          f"n={args.n}, q={args.L}, jB={args.jB}, nA_triples={args.nA_triples}, "
          f"pB={args.p_pair_B}, bg={args.bg_scale}, inst={args.instances}) =====")
    print("  looking for: exact splits A/B (coexistence) while AIS over-commits to the WIDE "
          "basin B (a_PA≪ex_PA, a_PB≫ex_PB) and PT tracks exact → first-order, AIS supercools")
    print(f"\n  {'jA':>5} {'ex_PA':>6} {'ex_PB':>6} {'ex_Pn':>6} {'ex_logZ':>8} "
          f"{'AIS_PA':>6} {'AIS_PB':>6} {'AISdlz':>7} {'ESS':>6} "
          f"{'PT_PA':>6} {'PT_PB':>6} {'PTdlz':>6} {'Gb_PA':>6} {'Gb_PB':>6}")
    betas = np.linspace(0.0, 1.0, args.pt_rungs)
    rows = []
    for jA in args.jA_list:
        acc = {k: [] for k in ("ex_PA", "ex_PB", "ex_Pn", "ex_lz", "a_PA", "a_PB",
                               "a_dlz", "ess", "a_mh", "p_PA", "p_PB", "p_dlz",
                               "p_mh", "g_PA", "g_PB")}
        for r in range(args.instances):
            seed = args.seed + 1000 * r
            g, t_A, t_B = build_two_basin_instance(
                args.n, args.L, jA=jA, jB=args.jB, nA_triples=args.nA_triples,
                p_pair_B=args.p_pair_B, bg_scale=args.bg_scale, seed=seed)
            ex = exact_full(g, t_A, args.thr, max_states=args.exact_budget)
            a = ais_full(g, t_A, args.thr, n_chains=args.ais_chains,
                         n_temps=args.ais_temps, n_sweeps=args.ais_sweeps, seed=seed)
            pt = pt_full(g, t_A, args.thr, betas=betas, n_chains=args.chains,
                         burn_in=args.burn_in, n_measure=args.n_measure, seed=seed)
            gb = block_gibbs(g, T=1.0, n_chains=args.chains, burn_in=args.burn_in,
                             n_measure=args.n_measure, co_cluster=False, seed=seed)
            ex_rd = two_basin_readout(ex["states"], ex["probs"], t_A, t_B, args.thr)
            a_rd = two_basin_readout(a["states"], a["weights"], t_A, t_B, args.thr)
            p_rd = two_basin_readout(pt["samples"], None, t_A, t_B, args.thr)
            g_rd = two_basin_readout(gb.samples, None, t_A, t_B, args.thr)
            acc["ex_PA"].append(ex_rd["P_A"]); acc["ex_PB"].append(ex_rd["P_B"])
            acc["ex_Pn"].append(ex_rd["P_none"]); acc["ex_lz"].append(ex["log_z"])
            acc["a_PA"].append(a_rd["P_A"]); acc["a_PB"].append(a_rd["P_B"])
            acc["a_dlz"].append(abs(a["log_z"] - ex["log_z"])); acc["ess"].append(a["ess"])
            acc["a_mh"].append(mean_hellinger(a["marginals"], ex["marginals"]))
            acc["p_PA"].append(p_rd["P_A"]); acc["p_PB"].append(p_rd["P_B"])
            acc["p_dlz"].append(abs(pt["log_z"] - ex["log_z"]))
            acc["p_mh"].append(mean_hellinger(pt["marginals"], ex["marginals"]))
            acc["g_PA"].append(g_rd["P_A"]); acc["g_PB"].append(g_rd["P_B"])
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        rows.append({"jA": jA, **m})
        print(f"  {jA:>5.2f} {m['ex_PA']:>6.3f} {m['ex_PB']:>6.3f} {m['ex_Pn']:>6.3f} "
              f"{m['ex_lz']:>8.3f} {m['a_PA']:>6.3f} {m['a_PB']:>6.3f} {m['a_dlz']:>7.3f} "
              f"{m['ess']:>6.3f} {m['p_PA']:>6.3f} {m['p_PB']:>6.3f} {m['p_dlz']:>6.3f} "
              f"{m['g_PA']:>6.3f} {m['g_PB']:>6.3f}")
    return {"rows": rows}


def _pick_coexistence_jA(rows: list[dict]) -> float | None:
    """Pick the coexistence jA where a barrier is most provably present.

    Among cells with both basins alive (P_A, P_B ≥ 0.05) we pick the one where
    single-T Gibbs is most TRAPPED — the largest |Gibbs P_A − exact P_A|. A large
    Gibbs trapping gap is direct evidence of an energy barrier (a barrier-blind
    ensemble can't reach equilibrium), so it is the cell that most stresses
    whether AIS *also* fails (route-iii win) or anneals past it (obstruction)."""
    cand = [r for r in rows if r["ex_PA"] >= 0.05 and r["ex_PB"] >= 0.05]
    if not cand:
        return None
    return max(cand, key=lambda r: abs(r["g_PA"] - r["ex_PA"]))["jA"]


def exp_two_basin_schedule(args, jA: float) -> dict:
    print(f"\n===== EXP FO-D: two-basin AIS schedule discriminator at coexistence "
          f"jA={jA:.2f} (n={args.n}, q={args.L}) =====")
    print("  first-order ⇒ AIS A/B misallocation PERSISTS as temps grow (intra-temp "
          "entropic barrier); continuous ⇒ it vanishes. PT-TI = replica reference.")
    g, t_A, t_B = build_two_basin_instance(
        args.n, args.L, jA=jA, jB=args.jB, nA_triples=args.nA_triples,
        p_pair_B=args.p_pair_B, bg_scale=args.bg_scale, seed=args.seed)
    ex = exact_full(g, t_A, args.thr, max_states=args.exact_budget)
    ex_rd = two_basin_readout(ex["states"], ex["probs"], t_A, t_B, args.thr)
    betas = np.linspace(0.0, 1.0, args.pt_rungs)
    pt = pt_full(g, t_A, args.thr, betas=betas, n_chains=args.chains,
                 burn_in=args.burn_in, n_measure=args.n_measure, seed=args.seed)
    pt_rd = two_basin_readout(pt["samples"], None, t_A, t_B, args.thr)
    print(f"  exact:  P_A={ex_rd['P_A']:.3f}  P_B={ex_rd['P_B']:.3f}  "
          f"P_none={ex_rd['P_none']:.3f}  logZ={ex['log_z']:.3f}")
    print(f"  PT-TI ({args.pt_rungs} rungs): P_A={pt_rd['P_A']:.3f}  P_B={pt_rd['P_B']:.3f}  "
          f"logZ={pt['log_z']:.3f}  |Δlz|={abs(pt['log_z']-ex['log_z']):.3f}  "
          f"margH={mean_hellinger(pt['marginals'], ex['marginals']):.3f}")
    print(f"\n  {'temps':>6} {'sweeps':>6} {'AIS_PA':>6} {'AIS_PB':>6} {'|ΔP_A|':>7} "
          f"{'AIS_logZ':>9} {'|Δlz|':>6} {'ESS':>6} {'margH':>6}")
    rows = []
    configs = [(nt, 1) for nt in args.fob_temps] + \
              [(args.fob_temps[-1], s) for s in args.fob_sweeps if s > 1]
    for nt, sw in configs:
        acc = {k: [] for k in ("PA", "PB", "lz", "ess", "mh")}
        for r in range(args.instances):
            a = ais_full(g, t_A, args.thr, n_chains=args.ais_chains,
                         n_temps=nt, n_sweeps=sw, seed=args.seed + 100 * r)
            a_rd = two_basin_readout(a["states"], a["weights"], t_A, t_B, args.thr)
            acc["PA"].append(a_rd["P_A"]); acc["PB"].append(a_rd["P_B"])
            acc["lz"].append(a["log_z"]); acc["ess"].append(a["ess"])
            acc["mh"].append(mean_hellinger(a["marginals"], ex["marginals"]))
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        dpa = abs(m["PA"] - ex_rd["P_A"])
        dlz = abs(m["lz"] - ex["log_z"])
        rows.append({"temps": nt, "sweeps": sw, "P_A": m["PA"], "P_B": m["PB"],
                     "abs_err_PA": dpa, "ais_logz": m["lz"], "abs_err_lz": dlz,
                     "ess": m["ess"], "marg_h": m["mh"]})
        print(f"  {nt:>6} {sw:>6} {m['PA']:>6.3f} {m['PB']:>6.3f} {dpa:>7.3f} "
              f"{m['lz']:>9.3f} {dlz:>6.3f} {m['ess']:>6.3f} {m['mh']:>6.3f}")
    return {"jA": jA,
            "exact": {"P_A": ex_rd["P_A"], "P_B": ex_rd["P_B"],
                      "P_none": ex_rd["P_none"], "log_z": ex["log_z"]},
            "pt": {"P_A": pt_rd["P_A"], "P_B": pt_rd["P_B"], "log_z": pt["log_z"],
                   "abs_err_lz": abs(pt["log_z"] - ex["log_z"]),
                   "abs_err_PA": abs(pt_rd["P_A"] - ex_rd["P_A"]),
                   "marg_h": mean_hellinger(pt["marginals"], ex["marginals"])},
            "rows": rows}


# ───────────────────────── verdicts ─────────────────────────────────────────


def _verdict_A(rows: list[dict]) -> str:
    # a "win" cell: bimodal + AIS misses basin while PT tracks
    hits = [r for r in rows
            if r["bimod"] >= 0.05
            and abs(r["a_pB"] - r["ex_pB"]) > 0.15
            and abs(r["p_pB"] - r["ex_pB"]) < 0.10]
    if hits:
        r = max(hits, key=lambda x: abs(x["a_pB"] - x["ex_pB"]))
        return (f"FIRST-ORDER + AIS-SUPERCOOLS candidate at j3={r['j3']}: exact "
                f"P(basin)={r['ex_pB']:.3f} (bimod {r['bimod']:.3f}) but AIS "
                f"P(basin)={r['a_pB']:.3f} (Δ={abs(r['a_pB']-r['ex_pB']):.3f}, ESS "
                f"{r['ess']:.3f}, logẐ |Δ|={r['a_dlz']:.3f}) while PT P(basin)="
                f"{r['p_pB']:.3f} (|Δlz|={r['p_dlz']:.3f}) tracks. Confirm persistence "
                f"in EXP FO-B (schedule).")
    bm = [r for r in rows if r["bimod"] >= 0.05]
    if bm:
        r = max(bm, key=lambda x: x["bimod"])
        return (f"Bimodal coexistence reached (j3={r['j3']}, bimod {r['bimod']:.3f}, exact "
                f"P(basin)={r['ex_pB']:.3f}) but AIS TRACKS it (AIS P(basin)={r['a_pB']:.3f}, "
                f"logẐ |Δ|={r['a_dlz']:.3f}, ESS {r['ess']:.3f}) — annealing finds the basin; "
                f"no AIS-unique failure. Obstruction holds on this construction.")
    return ("No bimodal coexistence in the j3 sweep — basin never competes with the "
            "disordered phase at these (j2, n_triples, n) settings. Widen j3 / n_triples.")


def _verdict_B(fob: dict) -> str:
    rows = fob["rows"]
    base = [r for r in rows if r["sweeps"] == 1]
    if not base:
        return "no schedule rows"
    few = min(base, key=lambda r: r["temps"])
    many = max(base, key=lambda r: r["temps"])
    ex_pB = fob["exact"]["p_basin"]
    pt_err = fob["pt"]["abs_err"]
    persist_lz = many["abs_err"]
    persist_pB = abs(many["p_basin"] - ex_pB)
    if persist_pB > 0.15 and pt_err < 0.2 and many["ess"] < 0.5:
        return (f"ROUTE-(iii) POSITIVE: at the densest AIS schedule ({many['temps']} temps) "
                f"AIS still misses the basin (P_basin {many['p_basin']:.3f} vs exact "
                f"{ex_pB:.3f}, Δ={persist_pB:.3f}; logẐ |Δ|={persist_lz:.3f}, ESS "
                f"{many['ess']:.3f}) while PT-TI tracks (|Δlz|={pt_err:.3f}). The bias does "
                f"NOT vanish with temps → genuine first-order barrier AIS cannot anneal past; "
                f"replica exchange is required. No-escape first-order instance FOUND.")
    if few["abs_err"] - many["abs_err"] > 0.2 or abs(few["p_basin"] - ex_pB) - persist_pB > 0.15:
        return (f"AIS bias VANISHES with temps (P_basin err {abs(few['p_basin']-ex_pB):.3f}@"
                f"{few['temps']} → {persist_pB:.3f}@{many['temps']}; logẐ |Δ| "
                f"{few['abs_err']:.3f}→{many['abs_err']:.3f}) → the barrier is crossable by "
                f"annealing alone (continuous / weak first-order, like the ferro EXP ZF). NOT "
                f"a PT-unique barrier; route (iii) closes on this instance.")
    return (f"AIS TRACKS at all schedules (P_basin err {persist_pB:.3f}, logẐ |Δ| "
            f"{persist_lz:.3f}, ESS {many['ess']:.3f}) — even the coarse schedule finds the "
            f"basin. No AIS failure; obstruction confirmed empirically here.")


def _verdict_C(rows: list[dict]) -> str:
    # win cell: exact coexistence (both basins live) but AIS over-commits to wide B
    # while PT tracks exact A/B occupancy.
    hits = [r for r in rows
            if r["ex_PA"] >= 0.05 and r["ex_PB"] >= 0.05      # coexistence
            and abs(r["a_PA"] - r["ex_PA"]) > 0.15            # AIS misallocates A
            and abs(r["p_PA"] - r["ex_PA"]) < 0.10            # PT tracks A
            and r["a_PB"] - r["ex_PB"] > 0.10]                # AIS over-commits to wide B
    if hits:
        r = max(hits, key=lambda x: abs(x["a_PA"] - x["ex_PA"]))
        return (f"FIRST-ORDER + AIS-SUPERCOOLS candidate at jA={r['jA']}: exact splits "
                f"P_A={r['ex_PA']:.3f}/P_B={r['ex_PB']:.3f} but AIS over-commits to the WIDE "
                f"basin (AIS P_A={r['a_PA']:.3f}, P_B={r['a_PB']:.3f}, ESS {r['ess']:.3f}, "
                f"logẐ |Δ|={r['a_dlz']:.3f}) while PT tracks (PT P_A={r['p_PA']:.3f}, "
                f"|Δlz|={r['p_dlz']:.3f}). Confirm persistence in EXP FO-D (schedule).")
    coex = [r for r in rows if r["ex_PA"] >= 0.05 and r["ex_PB"] >= 0.05]
    if coex:
        # report the cell where the barrier is most provably present (Gibbs trapped)
        r = max(coex, key=lambda x: abs(x["g_PA"] - x["ex_PA"]))
        g_gap = abs(r["g_PA"] - r["ex_PA"])
        barrier = (f" Barrier is REAL: single-T Gibbs is trapped at this cell "
                   f"(Gibbs P_A={r['g_PA']:.3f} vs exact {r['ex_PA']:.3f}, gap {g_gap:.3f}) "
                   f"yet AIS still crosses it.") if g_gap > 0.12 else ""
        return (f"Coexistence reached (jA={r['jA']}, exact P_A={r['ex_PA']:.3f}/"
                f"P_B={r['ex_PB']:.3f}) but AIS TRACKS the split (AIS P_A={r['a_PA']:.3f}/"
                f"P_B={r['a_PB']:.3f}, logẐ |Δ|={r['a_dlz']:.3f}, ESS {r['ess']:.3f}) — "
                f"annealing partitions A/B correctly; no AIS-unique failure. Obstruction "
                f"holds.{barrier}")
    return ("No A/B coexistence in the jA sweep — one basin dominates at every depth "
            "(no first-order competition). Widen jA / adjust jB·n_pairs vs jA·nA_triples.")


def _verdict_D(fod: dict) -> str:
    rows = fod["rows"]
    base = [r for r in rows if r["sweeps"] == 1]
    if not base:
        return "no schedule rows"
    few = min(base, key=lambda r: r["temps"])
    many = max(base, key=lambda r: r["temps"])
    ex_PA = fod["exact"]["P_A"]
    pt_err = fod["pt"]["abs_err_PA"]
    persist_PA = abs(many["P_A"] - ex_PA)
    persist_lz = many["abs_err_lz"]
    if persist_PA > 0.15 and pt_err < 0.10:
        return (f"ROUTE-(iii) POSITIVE: at the densest AIS schedule ({many['temps']} temps) "
                f"AIS still misallocates A/B (P_A {many['P_A']:.3f} vs exact {ex_PA:.3f}, "
                f"Δ={persist_PA:.3f}; logẐ |Δ|={persist_lz:.3f}, ESS {many['ess']:.3f}) while "
                f"PT tracks (|ΔP_A|={pt_err:.3f}). The bias does NOT vanish with temps → genuine "
                f"first-order entropic barrier AIS cannot anneal past; replica exchange required. "
                f"No-escape first-order instance FOUND.")
    if abs(few["P_A"] - ex_PA) - persist_PA > 0.15:
        return (f"AIS bias VANISHES with temps (P_A err {abs(few['P_A']-ex_PA):.3f}@"
                f"{few['temps']} → {persist_PA:.3f}@{many['temps']}) → the barrier is crossable "
                f"by annealing alone (continuous / weak first-order). NOT a PT-unique barrier; "
                f"route (iii) closes on this instance.")
    return (f"AIS TRACKS at all schedules (P_A err {persist_PA:.3f}, logẐ |Δ| {persist_lz:.3f}, "
            f"ESS {many['ess']:.3f}) — even the coarse schedule partitions A/B. No AIS failure; "
            f"obstruction confirmed empirically here.")


# ───────────────────────── main ─────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/probe_first_order.json")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--L", type=int, default=4, help="q = number of Potts states")
    ap.add_argument("--thr", type=float, default=0.6, help="basin overlap threshold")
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--chains", type=int, default=256, help="PT / Gibbs chains per rung")
    ap.add_argument("--ais-chains", type=int, default=512)
    ap.add_argument("--ais-temps", type=int, default=400, help="AIS temps in EXP FO-A")
    ap.add_argument("--ais-sweeps", type=int, default=1, help="Gibbs sweeps per AIS temp (FO-A)")
    ap.add_argument("--burn-in", type=int, default=500)
    ap.add_argument("--n-measure", type=int, default=200)
    ap.add_argument("--pt-rungs", type=int, default=33, help="PT-TI β-grid points (incl 0,1)")
    # instance knobs
    ap.add_argument("--j2", type=float, default=0.5, help="planted pairwise funnel weight")
    ap.add_argument("--p-pair", type=float, default=0.6, help="ER prob a planted pair is kept")
    ap.add_argument("--n-triples", type=int, default=24, help="number of planted 3-body clauses")
    ap.add_argument("--bg-scale", type=float, default=0.3, help="random unary field scale")
    ap.add_argument("--j3-list", default="0.0,0.3,0.6,0.9,1.2,1.6,2.0,2.5",
                    help="planted 3-body depths to sweep (coexistence)")
    # FO-B
    ap.add_argument("--fob-temps", default="5,10,20,40,80,160,400,800",
                    help="AIS temperature ladder for the schedule discriminator")
    ap.add_argument("--fob-sweeps", default="1,4,16",
                    help="AIS sweeps-per-temp ladder (at the densest temp count)")
    ap.add_argument("--fob-j3", type=float, default=None,
                    help="force the FO-B coexistence j3 (else auto-pick from FO-A)")
    # two-basin (FO-C / FO-D) knobs — the principled (iii) shot
    ap.add_argument("--jB", type=float, default=0.5,
                    help="basin-B (wide, shallow) 2-body funnel weight")
    ap.add_argument("--p-pair-B", type=float, default=0.6,
                    help="ER prob a basin-B funnel pair is kept")
    ap.add_argument("--nA-triples", type=int, default=20,
                    help="basin-A (deep, narrow) 3-body clause count")
    ap.add_argument("--jA-list", default="0.3,0.6,0.9,1.2,1.6,2.0,2.5,3.0",
                    help="basin-A depths to sweep (find A/B coexistence)")
    ap.add_argument("--foc-jA", type=float, default=None,
                    help="force the FO-D coexistence jA (else auto-pick from FO-C)")
    ap.add_argument("--exact-budget", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-coexistence", action="store_true")
    ap.add_argument("--skip-schedule", action="store_true")
    ap.add_argument("--skip-two-basin", action="store_true")
    ap.add_argument("--skip-two-basin-schedule", action="store_true")
    args = ap.parse_args()
    args.j3_list = [float(x) for x in args.j3_list.split(",")]
    args.fob_temps = [int(x) for x in args.fob_temps.split(",")]
    args.fob_sweeps = [int(x) for x in args.fob_sweeps.split(",")]
    args.jA_list = [float(x) for x in args.jA_list.split(",")]

    fo_a = None if args.skip_coexistence else exp_coexistence(args)
    if fo_a:
        print(f"\n  VERDICT(FO-A): {_verdict_A(fo_a['rows'])}")

    fo_b = None
    if not args.skip_schedule:
        j3 = args.fob_j3
        if j3 is None and fo_a is not None:
            j3 = _pick_coexistence_j3(fo_a["rows"])
        if j3 is None:
            print("\n  EXP FO-B skipped: no coexistence j3 found in FO-A "
                  "(pass --fob-j3 to force).")
        else:
            fo_b = exp_schedule(args, j3)
            print(f"\n  VERDICT(FO-B): {_verdict_B(fo_b)}")

    fo_c = None if args.skip_two_basin else exp_two_basin(args)
    if fo_c:
        print(f"\n  VERDICT(FO-C): {_verdict_C(fo_c['rows'])}")

    fo_d = None
    if not args.skip_two_basin_schedule and fo_c is not None:
        jA = args.foc_jA
        if jA is None:
            jA = _pick_coexistence_jA(fo_c["rows"])
        if jA is None:
            print("\n  EXP FO-D skipped: no A/B coexistence jA found in FO-C "
                  "(pass --foc-jA to force).")
        else:
            fo_d = exp_two_basin_schedule(args, jA)
            print(f"\n  VERDICT(FO-D): {_verdict_D(fo_d)}")

    report = {"config": vars(args), "coexistence": fo_a, "schedule": fo_b,
              "two_basin": fo_c, "two_basin_schedule": fo_d,
              "verdict_A": _verdict_A(fo_a["rows"]) if fo_a else None,
              "verdict_B": _verdict_B(fo_b) if fo_b else None,
              "verdict_C": _verdict_C(fo_c["rows"]) if fo_c else None,
              "verdict_D": _verdict_D(fo_d) if fo_d else None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=lambda o: (
        o.tolist() if isinstance(o, np.ndarray) else float(o)
        if isinstance(o, (np.floating, np.integer)) else str(o))))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
