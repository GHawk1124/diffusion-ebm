"""Track B barrier-crossing scout: can tempering close the metastability gap?

The hardness-dial scout (``probe_hardness_dial.py``) found the regime that
justifies the thermodynamic-hardware thesis: a frustrated, LM-grounded Potts
model with n holes over a shared L-state alphabet has a genuine
intractable-but-multimodal regime (exact brute force dies past n≈11–12 on a
complete graph) where THRML block-Gibbs is the only feasible solver — BUT at
strong coupling (w≥8) single-temperature block-Gibbs hits a **metastability
wall**: marginal TV jumps to ~0.6, the MAP-mode energy gap opens to ~3.4 nats,
and *more burn-in does not help* (TV is flat across burn_in 50→3200). That is a
barrier-crossing failure, not slow mixing.

The textbook fix for barrier crossing is **tempering**: melt the barriers at
high temperature, then cool. This scout asks the one question that decides
whether Track B has a hero figure:

    On the hard instances where single-temperature block-Gibbs is metastable
    (n=8, w∈{8,16}, spin-glass), do **annealing** and **parallel tempering**
    cross the barrier — restore marginal fidelity (TV→exact) and find the MAP
    mode (Egap→0) — and at what compute premium over vanilla Gibbs?

Implementation note (why numpy, not THRML): ``FrustratedSampler.sample`` runs a
**single** Markov chain (its ``n_chains`` argument collects autocorrelated
samples one sweep apart from one chain; init is zeros, no per-replica state or
temperature is exposed). Parallel tempering needs K replicas at K temperatures
with per-sweep state exchange — none of which THRML's current API surfaces. So
this scout reimplements block-Gibbs as a faithful pure-numpy single-site Gibbs
sampler with **exact Potts conditionals** — the identical algorithm THRML runs
on a dense graph (where DSATUR coloring forces single-node blocks anyway) — and
validates it against the same brute-force exact enumerator from the hardness
probe. Temperature enters as ``logit / T`` in the conditional; PT adds replica
swaps with the Metropolis acceptance ``min(1, exp((β_a−β_b)(U_b−U_a)))``.

    LD_LIBRARY_PATH="<zlib>:/run/opengl-driver/lib:..." JAX_PLATFORMS=cpu \\
        .venv/bin/python experiments/probe_tempering.py \\
            --cache results/probe_splitability_cache.json \\
            --out results/probe_tempering.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe_hardness_dial import (  # noqa: E402
    config_energy,
    exact_solve,
    load_field_bank,
    make_edges,
    gibbs_solve,
    make_instance,
    marginal_tv,
)


# ─────────────────────── numpy block-Gibbs core ────────────────────────────


def build_adj(
    n: int,
    attract: list[tuple[int, int]],
    repel: list[tuple[int, int]],
) -> list[list[tuple[int, bool]]]:
    """Per-variable neighbour list of ``(neighbour, is_attract)``."""
    adj: list[list[tuple[int, bool]]] = [[] for _ in range(n)]
    for (i, j) in attract:
        adj[i].append((j, True))
        adj[j].append((i, True))
    for (i, j) in repel:
        adj[i].append((j, False))
        adj[j].append((i, False))
    return adj


def gibbs_sweep(
    states: np.ndarray,       # (C, n) int — updated in place
    unary: np.ndarray,        # (n, L)
    adj: list[list[tuple[int, bool]]],
    w_a: float,
    w_r: float,
    T: float,
    rng: np.random.Generator,
) -> None:
    """One systematic-scan single-site Gibbs sweep at temperature T.

    For hole i the exact Potts conditional log-weight at state s is
    ``unary[i,s] + Σ_{attract j} w_a·1[s=x_j] − Σ_{repel j} w_r·1[s=x_j]``.
    Sampling ∝ softmax(logit / T) is done with the Gumbel-max trick across all
    C chains at once. Updates are in place, so later holes in the scan see the
    freshly resampled earlier holes (a proper Gibbs sweep).
    """
    C, n = states.shape
    L = unary.shape[1]
    rows = np.arange(C)
    for i in range(n):
        logit = np.tile(unary[i], (C, 1)).astype(np.float64)  # (C, L)
        for (j, is_att) in adj[i]:
            logit[rows, states[:, j]] += w_a if is_att else -w_r
        g = rng.gumbel(size=(C, L))
        states[:, i] = np.argmax(logit / T + g, axis=1)


def _marginals(samples: np.ndarray, n: int, L: int) -> np.ndarray:
    marg = np.zeros((n, L), dtype=np.float64)
    for i in range(n):
        marg[i] = np.bincount(samples[:, i], minlength=L) / samples.shape[0]
    return marg


def _best_energy(samples, unary, attract, repel, w_a, w_r) -> float:
    return float(config_energy(samples, unary, attract, repel, w_a, w_r).max())


# ───────────────────────────── solvers ─────────────────────────────────────


def vanilla_gibbs(
    unary, attract, repel, w_a, w_r,
    n_chains: int, burn_in: int, n_measure: int, seed: int,
) -> dict:
    """Single-temperature (T=1) block-Gibbs with C independent parallel chains.

    Honest reproduction of the metastability wall: each chain starts at a random
    state and equilibrates at T=1, so a mode-locked chain stays locked.
    """
    rng = np.random.default_rng(seed)
    n, L = unary.shape
    adj = build_adj(n, attract, repel)
    states = rng.integers(0, L, size=(n_chains, n))
    t0 = time.time()
    for _ in range(burn_in):
        gibbs_sweep(states, unary, adj, w_a, w_r, 1.0, rng)
    coll = []
    best = -np.inf
    for _ in range(n_measure):
        gibbs_sweep(states, unary, adj, w_a, w_r, 1.0, rng)
        coll.append(states.copy())
        best = max(best, _best_energy(states, unary, attract, repel, w_a, w_r))
    wall = time.time() - t0
    samples = np.concatenate(coll, axis=0)
    return {"marginals": _marginals(samples, n, L), "best_energy": best,
            "wall": wall, "sweeps": (burn_in + n_measure) * n_chains}


def single_chain_gibbs(
    unary, attract, repel, w_a, w_r,
    burn_in: int, n_collect: int, T: float, seed: int,
    zero_init: bool = True,
) -> dict:
    """Mirror THRML's *strategy*: ONE chain, zero-init, warm up, then collect
    ``n_collect`` autocorrelated samples one sweep apart.

    THRML's ``SamplingSchedule(burn_in, n_chains, 1)`` is not 512 independent
    chains — it warms up a single zero-initialised chain for ``burn_in`` sweeps
    then records ``n_chains`` consecutive samples (``steps_per_sample=1``). To
    test whether the numpy kernel matches THRML we must match that strategy, not
    just the kernel. ``zero_init`` toggles THRML's zero start vs a random start.
    """
    rng = np.random.default_rng(seed)
    n, L = unary.shape
    adj = build_adj(n, attract, repel)
    states = (np.zeros((1, n), dtype=int) if zero_init
              else rng.integers(0, L, size=(1, n)))
    for _ in range(burn_in):
        gibbs_sweep(states, unary, adj, w_a, w_r, T, rng)
    coll = []
    for _ in range(n_collect):
        gibbs_sweep(states, unary, adj, w_a, w_r, T, rng)
        coll.append(states.copy())
    samples = np.concatenate(coll, axis=0)  # (n_collect, n)
    return {"marginals": _marginals(samples, n, L)}


def annealed_gibbs(
    unary, attract, repel, w_a, w_r,
    n_chains: int, n_anneal: int, n_measure: int, T_hi: float, seed: int,
) -> dict:
    """Geometric temperature anneal T_hi→1, then measure at T=1.

    Each chain melts the frustration barriers while hot, then freezes into a
    (hopefully) low-energy mode as it cools.
    """
    rng = np.random.default_rng(seed)
    n, L = unary.shape
    adj = build_adj(n, attract, repel)
    states = rng.integers(0, L, size=(n_chains, n))
    schedule = np.geomspace(T_hi, 1.0, n_anneal)
    t0 = time.time()
    for T in schedule:
        gibbs_sweep(states, unary, adj, w_a, w_r, float(T), rng)
    coll = []
    best = -np.inf
    for _ in range(n_measure):
        gibbs_sweep(states, unary, adj, w_a, w_r, 1.0, rng)
        coll.append(states.copy())
        best = max(best, _best_energy(states, unary, attract, repel, w_a, w_r))
    wall = time.time() - t0
    samples = np.concatenate(coll, axis=0)
    return {"marginals": _marginals(samples, n, L), "best_energy": best,
            "wall": wall, "sweeps": (n_anneal + n_measure) * n_chains,
            "T_hi": T_hi}


def parallel_tempering(
    unary, attract, repel, w_a, w_r,
    n_chains: int, K: int, T_max: float,
    n_sweep: int, n_measure: int, seed: int,
) -> dict:
    """Replica-exchange MCMC: K geometric temperatures, swap adjacent levels.

    Maintains a (K, n_chains, n) ensemble — n_chains independent PT runs, each
    with one replica per temperature T_1=1 < … < T_K=T_max. Every sweep, after
    a per-level Gibbs update, adjacent levels attempt a config swap with
    acceptance ``min(1, exp((β_a−β_b)(U_b−U_a)))`` (U = log p). Samples are the
    cold (T=1) replica. Even/odd parity alternation keeps swap pairs disjoint.
    """
    rng = np.random.default_rng(seed)
    n, L = unary.shape
    adj = build_adj(n, attract, repel)
    Ts = np.geomspace(1.0, T_max, K)
    betas = 1.0 / Ts
    states = rng.integers(0, L, size=(K, n_chains, n))
    rows = np.arange(n_chains)
    attempts = 0
    accepts = 0

    def swap(parity: int) -> None:
        nonlocal attempts, accepts
        U = np.stack([
            config_energy(states[k], unary, attract, repel, w_a, w_r)
            for k in range(K)
        ])  # (K, C)
        for k in range(parity, K - 1, 2):
            a, b = k, k + 1  # a colder (lower T), b hotter
            delta = (betas[a] - betas[b]) * (U[b] - U[a])  # (C,)
            acc = rng.random(n_chains) < np.exp(np.minimum(delta, 0.0))
            if acc.any():
                tmp = states[a, acc].copy()
                states[a, acc] = states[b, acc]
                states[b, acc] = tmp
                ua = U[a, acc].copy()
                U[a, acc] = U[b, acc]
                U[b, acc] = ua
            attempts += int(acc.size)
            accepts += int(acc.sum())

    t0 = time.time()
    for s in range(n_sweep):
        for k in range(K):
            gibbs_sweep(states[k], unary, adj, w_a, w_r, float(Ts[k]), rng)
        swap(s % 2)
    coll = []
    best = -np.inf
    for s in range(n_measure):
        for k in range(K):
            gibbs_sweep(states[k], unary, adj, w_a, w_r, float(Ts[k]), rng)
        swap(s % 2)
        cold = states[0].copy()
        coll.append(cold)
        best = max(best, _best_energy(cold, unary, attract, repel, w_a, w_r))
    wall = time.time() - t0
    samples = np.concatenate(coll, axis=0)
    return {"marginals": _marginals(samples, n, L), "best_energy": best,
            "wall": wall, "sweeps": (n_sweep + n_measure) * n_chains * K,
            "swap_rate": accepts / max(attempts, 1), "T_max": T_max, "K": K}


# ─────────────────────── graph / schedule stats ────────────────────────────


def graph_stats(n: int, attract, repel) -> dict:
    """DSATUR coloring stats — THRML's actual block schedule on this graph.

    A chromatic color class is an independent set, so its nodes are
    conditionally independent given the rest and are updated *in parallel* in
    one THRML block step. Block size is therefore a **parallelism / throughput**
    property, NOT a mixing property: a chromatic sweep is the same transition
    kernel as a systematic single-site scan regardless of block size. Dense
    graph → many colors → tiny blocks → no parallelism; sparse → few colors →
    big blocks → massive parallel updates (the hardware throughput win).
    """
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from([(i, j) for (i, j) in (list(attract) + list(repel))])
    coloring = nx.coloring.greedy_color(g, strategy="DSATUR")
    classes: dict[int, int] = {}
    for c in coloring.values():
        classes[c] = classes.get(c, 0) + 1
    sizes = list(classes.values())
    return {
        "n_edges": g.number_of_edges(),
        "n_colors": len(sizes),
        "max_block": max(sizes),
        "mean_block": float(np.mean(sizes)),
        "parallelism": n / len(sizes),  # avg nodes updated per parallel step
    }


# ──────────────────────────── experiment ───────────────────────────────────


def exp_barrier(bank, args) -> list[dict]:
    print("\n===== EXP C: tempering vs the metastability wall =====")
    print(f"  n={args.n}  L={args.L}  repel_frac={args.repel_fraction}  "
          f"instances={args.n_instances}  chains={args.n_chains}")
    print(f"  vanilla burn={args.burn_in}+meas={args.n_measure} | "
          f"anneal {args.n_anneal}+{args.n_measure} (T_hi={args.temp_mult}·w) | "
          f"PT K={args.pt_levels} T_max={args.temp_mult}·w sweeps={args.pt_sweeps}+{args.n_measure}")
    print("  metric = mean over instances; TV vs exact marginals, "
          "Egap = exactMAP − bestSample energy (nats; 0 = MAP mode found)")

    rows = []
    for w in args.weight_list:
        agg = {m: {"tv": [], "egap": [], "wall": []}
               for m in ("vanilla", "annealed", "pt")}
        swap_rates = []
        n_feasible = 0
        for inst in range(args.n_instances):
            rng = np.random.default_rng(args.seed + 1000 * inst)
            attract, repel = make_edges(
                args.n, "spinglass", rng,
                repel_fraction=args.repel_fraction, p_edge=args.p_edge,
            )
            unary = make_instance(bank, args.n, args.L, rng)
            ex = exact_solve(unary, attract, repel, args.L, w, w, args.exact_budget)
            if not ex["feasible"]:
                continue
            n_feasible += 1
            T_top = args.temp_mult * w

            van = vanilla_gibbs(unary, attract, repel, w, w,
                                args.n_chains, args.burn_in, args.n_measure,
                                args.seed + inst)
            ann = annealed_gibbs(unary, attract, repel, w, w,
                                 args.n_chains, args.n_anneal, args.n_measure,
                                 T_top, args.seed + inst)
            pt = parallel_tempering(unary, attract, repel, w, w,
                                    args.n_chains, args.pt_levels, T_top,
                                    args.pt_sweeps, args.n_measure,
                                    args.seed + inst)
            for name, res in (("vanilla", van), ("annealed", ann), ("pt", pt)):
                agg[name]["tv"].append(marginal_tv(res["marginals"], ex["marginals"]))
                agg[name]["egap"].append(ex["map_energy"] - res["best_energy"])
                agg[name]["wall"].append(res["wall"])
            swap_rates.append(pt["swap_rate"])

        if n_feasible == 0:
            print(f"  w={w}: all instances infeasible at n={args.n}; skip")
            continue

        print(f"\n  --- w={w} ({n_feasible} instances, PT swap-rate "
              f"{np.mean(swap_rates):.2f}) ---")
        print(f"  {'method':>10} {'TV':>8} {'Egap':>8} {'wall_s':>8} {'×vanilla':>9}")
        van_wall = float(np.mean(agg["vanilla"]["wall"]))
        wrow = {"weight": w, "n_feasible": n_feasible,
                "pt_swap_rate": float(np.mean(swap_rates))}
        for name in ("vanilla", "annealed", "pt"):
            tv = float(np.mean(agg[name]["tv"]))
            eg = float(np.mean(agg[name]["egap"]))
            wl = float(np.mean(agg[name]["wall"]))
            print(f"  {name:>10} {tv:>8.3f} {eg:>8.3f} {wl:>8.3f} "
                  f"{wl / van_wall:>9.1f}")
            wrow[name] = {"tv": tv, "egap": eg, "wall": wl}
        rows.append(wrow)
    return rows


def exp_sparsity(bank, args) -> list[dict]:
    """EXP D: does the metastability wall persist as the graph sparsifies, and
    how does the DSATUR block schedule (hardware parallelism) scale?

    Fixes n and a hard coupling, sweeps Erdős–Rényi edge density p. Reports, per
    density: the block schedule (n_colors / max_block / parallelism = n/n_colors)
    and vanilla-vs-PT marginal TV. Separates the two hardware-relevant axes that
    the dense complete-graph case conflates:
      * parallelism  — RISES as the graph sparsifies (big color classes);
      * metastability — a property of frustration strength, NOT of blocking, so
        it should track edge count / contradiction density, and PT should cross
        it at every density where vanilla fails.
    """
    print("\n===== EXP D: sparsity — block schedule + does the wall persist? =====")
    n = args.n
    w = args.sparse_weight
    T_top = args.temp_mult * w
    print(f"  n={n}  L={args.L}  w={w}  repel_frac={args.repel_fraction}  "
          f"instances={args.n_instances}  chains={args.n_chains}")
    print("  parallelism = n / n_colors (avg nodes per parallel THRML block step; "
          "higher = more hardware throughput)")
    print(f"\n  {'p_edge':>7} {'edges':>6} {'colors':>7} {'maxblk':>7} "
          f"{'parlsm':>7} {'vanTV':>7} {'ptTV':>7} {'van_s':>7} {'pt_s':>7}")
    rows = []
    for p in args.density_list:
        gs_agg = {"n_edges": [], "n_colors": [], "max_block": [], "parallelism": []}
        van_tv, pt_tv, van_wall, pt_wall = [], [], [], []
        swap_rates = []
        n_feasible = 0
        for inst in range(args.n_instances):
            rng = np.random.default_rng(args.seed + 1000 * inst + 7)
            attract, repel = make_edges(
                n, "spinglass", rng, repel_fraction=args.repel_fraction, p_edge=p
            )
            unary = make_instance(bank, n, args.L, rng)
            ex = exact_solve(unary, attract, repel, args.L, w, w, args.exact_budget)
            if not ex["feasible"]:
                continue
            n_feasible += 1
            gs = graph_stats(n, attract, repel)
            for kk in gs_agg:
                gs_agg[kk].append(gs[kk])
            van = vanilla_gibbs(unary, attract, repel, w, w,
                                args.n_chains, args.burn_in, args.n_measure,
                                args.seed + inst)
            pt = parallel_tempering(unary, attract, repel, w, w,
                                    args.n_chains, args.pt_levels, T_top,
                                    args.pt_sweeps, args.n_measure,
                                    args.seed + inst)
            van_tv.append(marginal_tv(van["marginals"], ex["marginals"]))
            pt_tv.append(marginal_tv(pt["marginals"], ex["marginals"]))
            van_wall.append(van["wall"])
            pt_wall.append(pt["wall"])
            swap_rates.append(pt["swap_rate"])
        if n_feasible == 0:
            continue
        row = {"p_edge": p, "n_feasible": n_feasible,
               "pt_swap_rate": float(np.mean(swap_rates))}
        for kk in gs_agg:
            row[kk] = float(np.mean(gs_agg[kk]))
        row["vanilla_tv"] = float(np.mean(van_tv))
        row["pt_tv"] = float(np.mean(pt_tv))
        row["vanilla_wall"] = float(np.mean(van_wall))
        row["pt_wall"] = float(np.mean(pt_wall))
        print(f"  {p:>7.2f} {row['n_edges']:>6.1f} {row['n_colors']:>7.1f} "
              f"{row['max_block']:>7.1f} {row['parallelism']:>7.2f} "
              f"{row['vanilla_tv']:>7.3f} {row['pt_tv']:>7.3f} "
              f"{row['vanilla_wall']:>7.3f} {row['pt_wall']:>7.3f}")
        rows.append(row)
    return rows


def exp_validate(bank, args) -> list[dict]:
    """EXP E: is the numpy reference sampler a faithful THRML proxy?

    The PT hero figure is generated by the numpy block-Gibbs (THRML's public API
    exposes only one zero-init chain at fixed temperature — no replica state or
    swaps). For that figure to stand in for the TSU, the numpy kernel must be the
    *same* sampler THRML runs. A first cut compared THRML against the EXP-C
    ``vanilla_gibbs`` (512 *independent random-init* chains) and they diverged
    badly at T=1 — but that is a **chain-strategy** difference, not a kernel one:
    ``SamplingSchedule(burn_in, n_chains, 1)`` runs ONE zero-init chain and
    records ``n_chains`` autocorrelated samples. So we separate the two:

      * ``numpy_single`` — matched strategy: 1 zero-init chain, ``n_chains``
        samples 1 sweep apart (the THRML twin). TV(THRML, numpy_single)≈0 across
        all T ⇒ the kernels are identical (the proxy claim).
      * ``numpy_multi`` — the EXP-C vanilla strategy: 512 independent random-init
        chains. Its TV-vs-exact at low T diagnoses whether the EXP-C "metastable"
        baseline is the canonical barrier or a strategy artefact.

    Sampling the tempered target p_T ∝ exp(U/T) is done by feeding every sampler
    ``unary/T`` and couplings ``±w/T`` (each run at its internal T=1). The exact
    marginal entropy (max log L ≈ 1.386 nats for L=4) is reported per T as a
    multimodality indicator: a low-entropy (peaked) target explains why a single
    zero-init chain can match exact while independent chains over-disperse.
    """
    print("\n===== EXP E: THRML ↔ numpy cross-validation (temperature sweep) =====")
    n = args.n
    w = args.val_weight
    print(f"  n={n}  L={args.L}  w={w}  repel_frac={args.repel_fraction}  "
          f"instances={args.val_instances}  chains={args.n_chains}")
    print("  sampling p_T ∝ exp(U/T): w/T is the effective coupling "
          "(w/T≫1 metastable, w/T≲1 easy)")
    print("  numpy_single = THRML twin (1 zero-init chain); "
          "numpy_multi = 512 independent random-init chains (EXP-C vanilla)")
    print(f"\n  {'T':>5} {'w/T':>5} {'H_exact':>8} "
          f"{'TV(th,ex)':>16} {'TV(np1,ex)':>16} {'TV(npM,ex)':>16} "
          f"{'TV(th,np1)':>12}")
    rows = []
    for T in args.val_temps:
        th_ex, np1_ex, npM_ex, th_np1, entropy = [], [], [], [], []
        n_feasible = 0
        for inst in range(args.val_instances):
            rng = np.random.default_rng(args.seed + 1000 * inst + 13)
            attract, repel = make_edges(
                n, "spinglass", rng,
                repel_fraction=args.repel_fraction, p_edge=args.p_edge,
            )
            unary = make_instance(bank, n, args.L, rng)
            uT, wT = unary / T, w / T
            ex = exact_solve(uT, attract, repel, args.L, wT, wT, args.exact_budget)
            if not ex["feasible"]:
                continue
            n_feasible += 1
            th = gibbs_solve(uT, attract, repel, args.L, wT, wT,
                             args.n_chains, args.burn_in, args.seed + inst)
            np1 = single_chain_gibbs(uT, attract, repel, wT, wT,
                                     args.burn_in, args.n_chains, 1.0,
                                     args.seed + inst, zero_init=True)
            npM = vanilla_gibbs(uT, attract, repel, wT, wT,
                                args.n_chains, args.burn_in, args.n_measure,
                                args.seed + inst)
            th_ex.append(marginal_tv(th["marginals"], ex["marginals"]))
            np1_ex.append(marginal_tv(np1["marginals"], ex["marginals"]))
            npM_ex.append(marginal_tv(npM["marginals"], ex["marginals"]))
            th_np1.append(marginal_tv(th["marginals"], np1["marginals"]))
            entropy.append(ex["marg_entropy"])
        if n_feasible == 0:
            continue
        def ms(xs):  # mean, std
            return float(np.mean(xs)), float(np.std(xs))
        row = {"T": T, "w_over_T": w / T, "n_feasible": n_feasible,
               "exact_entropy": float(np.mean(entropy)),
               "tv_thrml_exact": ms(th_ex)[0], "tv_thrml_exact_std": ms(th_ex)[1],
               "tv_numpy_single_exact": ms(np1_ex)[0],
               "tv_numpy_single_exact_std": ms(np1_ex)[1],
               "tv_numpy_multi_exact": ms(npM_ex)[0],
               "tv_numpy_multi_exact_std": ms(npM_ex)[1],
               "tv_thrml_numpy_single": ms(th_np1)[0],
               "tv_thrml_numpy_single_std": ms(th_np1)[1]}
        print(f"  {T:>5.1f} {w / T:>5.1f} {row['exact_entropy']:>8.3f} "
              f"{row['tv_thrml_exact']:>7.3f}±{row['tv_thrml_exact_std']:<7.3f} "
              f"{row['tv_numpy_single_exact']:>7.3f}±{row['tv_numpy_single_exact_std']:<7.3f} "
              f"{row['tv_numpy_multi_exact']:>7.3f}±{row['tv_numpy_multi_exact_std']:<7.3f} "
              f"{row['tv_thrml_numpy_single']:>11.3f}")
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="results/probe_splitability_cache.json")
    ap.add_argument("--out", default="results/probe_tempering.json")
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--n", type=int, default=8, help="holes (exact must be tractable)")
    ap.add_argument("--weight-list", default="8,16", help="hard coupling weights")
    ap.add_argument("--repel-fraction", type=float, default=0.5)
    ap.add_argument("--p-edge", type=float, default=1.0)
    ap.add_argument("--n-instances", type=int, default=8)
    ap.add_argument("--n-chains", type=int, default=512)
    ap.add_argument("--burn-in", type=int, default=500)
    ap.add_argument("--n-measure", type=int, default=100)
    ap.add_argument("--n-anneal", type=int, default=500)
    ap.add_argument("--pt-levels", type=int, default=8)
    ap.add_argument("--pt-sweeps", type=int, default=500)
    ap.add_argument("--temp-mult", type=float, default=2.0,
                    help="hot-end temperature as a multiple of the coupling w")
    ap.add_argument("--exact-budget", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sparse-weight", type=float, default=16.0,
                    help="fixed coupling w for the EXP D density sweep")
    ap.add_argument("--density-list", default="0.2,0.35,0.5,0.75,1.0",
                    help="Erdős–Rényi edge densities for EXP D")
    ap.add_argument("--val-weight", type=float, default=16.0,
                    help="fixed coupling w for the EXP E cross-validation")
    ap.add_argument("--val-temps", default="1,2,4,8,16,32",
                    help="temperatures for EXP E (p_T ∝ exp(U/T))")
    ap.add_argument("--val-instances", type=int, default=4,
                    help="instances for EXP E (THRML recompiles per instance)")
    ap.add_argument("--experiments", default="barrier,sparsity",
                    help="comma list: barrier (EXP C), sparsity (EXP D), validate (EXP E)")
    args = ap.parse_args()
    args.weight_list = [float(x) for x in args.weight_list.split(",")]
    args.density_list = [float(x) for x in args.density_list.split(",")]
    args.val_temps = [float(x) for x in args.val_temps.split(",")]
    exps = set(args.experiments.split(","))

    bank = load_field_bank(Path(args.cache), args.L)
    print(f"[bank] {bank.shape[0]} real MDLM logit fields (top-{args.L})")

    # Merge into any existing report so a single-experiment rerun doesn't clobber
    # the other experiment's saved results.
    report = {}
    if Path(args.out).exists():
        report = json.loads(Path(args.out).read_text())
    report["config"] = vars(args)
    report["bank_size"] = int(bank.shape[0])
    if "barrier" in exps:
        report["barrier"] = exp_barrier(bank, args)
    if "sparsity" in exps:
        report["sparsity"] = exp_sparsity(bank, args)
    if "validate" in exps:
        report["validate"] = exp_validate(bank, args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
