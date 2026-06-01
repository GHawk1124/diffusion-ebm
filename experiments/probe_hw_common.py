"""Shared harness for the Phase-0 hardware-fit probes (Track B, post-scout).

Track B's first four scouts proved the *mechanism* (Pareto crossover,
metastability wall, tempering rescue) on a dialed synthetic frustrated Potts,
and proved the negative (real RMC partition posteriors are ambiguous but
exactly tractable and barrier-free — "ambiguity ≠ hardness"). The open
question that decides whether the hero figure can attach to a REAL task is:

    is there a real workload whose natural inference is (a) a SAMPLING /
    marginal problem — not a MAP / optimization problem — that (b) does not
    factorise or admit a solver escape, (c) is exact-infeasible at its natural
    size, (d) is frustrated enough that single-temperature block-Gibbs is
    metastable, (e) is crossed by replica exchange / tempering, (f) maps to a
    sparse local Ising/Potts a thermodynamic sampler runs natively, and (g)
    beats the classical sampling gauntlet at a favourable amortised cost?

That list is the scorecard below. It folds in the second-opinion critique
(codex gpt-5.5, xhigh) that reshaped the plan:

  * **The linchpin is SAMPLING, not MAP (G0).** A thermodynamic sampler earns
    its keep on partition functions, marginals, free energies, and posterior
    *expectations* — tasks where you need the distribution, not its argmax.
    Track A died precisely because oracle-grouped RMC has a closed-form MAP
    (`hard_eq_map`), so the headline must be a marginal/expectation task or it
    is anti-hardware by construction. Every probe states its sampling target
    explicitly and is scored on a *distributional* metric (TV to the true
    marginals / co-clustering), never on argmax accuracy alone.

  * **No solver escape (G1b).** "Exact enumeration is infeasible" is necessary
    but not sufficient: if junction-tree / belief-propagation / DPLL+component
    caching / a SAT model counter solves it cheaply, the hardware wins nothing.
    The classical gauntlet (below) must be run *before* claiming hardness.

  * **k-SAT is a FALSIFIER, not a bet (Probe C).** Reframed as uniform /
    weighted sampling over solutions (model counting), it is the cleanest test
    of whether "hard to optimise" implies "hard to sample" — and ApproxMC /
    UniGen are brutal classical baselines. Run it first to fail fast.

  * **DTM realignment (Probe B).** Extropic's own direction (Denoising
    Thermodynamic Models) favours *sparse, local, well-mixed* EBM conditionals
    sampled approximately — NOT rugged spin glasses rescued by parallel
    tempering. G8 (approximation-insensitivity) credits tasks that tolerate
    fast approximate samples; a task that *needs* exact rare modes is a poor
    hardware fit even if it is "hard".

  * **WMC / probabilistic inference on real Boolean factor graphs (Probe D)**
    is the most promising under-explored class: real, scored, genuinely a
    marginal problem, and natively a sparse factor graph.

This module is pure numpy (+ the host zlib LD path). It provides a generic
categorical factor graph, exact marginals, single-site block-Gibbs (the same
kernel chromatic blocking runs — see scout-3 EXP D), parallel tempering,
distributional TV utilities, a classical sampling gauntlet, and the G0–G9
scorecard with a JSON report writer. The four probes import it and only supply
their task graph + gate evidence.

Run the self-test:

    LD_LIBRARY_PATH="<zlib>:/run/opengl-driver/lib:..." \\
        .venv/bin/python experiments/probe_hw_common.py
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

# ───────────────────────────── factor graph ────────────────────────────────


@dataclass(frozen=True)
class Factor:
    """A log-potential over a tuple of variables.

    ``scope`` is variable indices; ``table`` has one axis per scope variable in
    scope order, holding the *log* potential (so the model is
    ``log p(x) ∝ Σ_f table_f[x[scope_f]]``). Hard constraints use ``-inf``.
    """

    scope: tuple[int, ...]
    table: np.ndarray

    def __post_init__(self) -> None:
        if self.table.ndim != len(self.scope):
            raise ValueError(
                f"factor table ndim {self.table.ndim} != |scope| {len(self.scope)}"
            )


@dataclass
class FactorGraph:
    """Categorical MRF: per-variable cardinalities + a list of log-potentials."""

    cards: tuple[int, ...]
    factors: list[Factor]

    @property
    def n_vars(self) -> int:
        return len(self.cards)

    @property
    def state_space(self) -> int:
        prod = 1
        for c in self.cards:
            prod *= int(c)
        return prod

    def touching(self, i: int) -> list[Factor]:
        return [f for f in self.factors if i in f.scope]

    def score(self, state: np.ndarray) -> float:
        """log p(x) up to the global constant, for a single integer state."""
        total = 0.0
        for f in self.factors:
            total += float(f.table[tuple(int(state[v]) for v in f.scope)])
        return total


# ─────────────────────── conditionals (vectorised over chains) ──────────────


def _cond_logits(graph: FactorGraph, states: np.ndarray, i: int) -> np.ndarray:
    """Conditional log p(x_i | x_{-i}) for every chain. states: (C, N) int.

    Returns (C, cards[i]); rows are unnormalised log-probabilities.
    """
    C = states.shape[0]
    logits = np.zeros((C, int(graph.cards[i])), dtype=np.float64)
    for f in graph.touching(i):
        pos_i = f.scope.index(i)
        tperm = np.moveaxis(f.table, pos_i, 0)  # (card_i, *other_cards)
        other_vars = [v for v in f.scope if v != i]
        if other_vars:
            other_idx = tuple(states[:, v] for v in other_vars)
            contrib = tperm[(slice(None),) + other_idx]  # (card_i, C)
        else:
            contrib = tperm[:, None] * np.ones((1, C))  # (card_i, C)
        logits += np.asarray(contrib).T
    return logits


def _gumbel_argmax(logits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one category per row from unnormalised ``logits`` (C, K)."""
    finite = np.isfinite(logits)
    g = rng.gumbel(size=logits.shape)
    noised = np.where(finite, logits + g, -np.inf)
    return np.argmax(noised, axis=1)


# ───────────────────────────── exact marginals ─────────────────────────────


@dataclass
class ExactResult:
    log_z: float
    entropy: float  # nats, of the joint
    var_marginals: list[np.ndarray]
    co_cluster: np.ndarray | None  # (N, N) P(x_i == x_j), only if shared card
    map_state: np.ndarray
    p_map: float
    n_states: int


def exact_marginals(
    graph: FactorGraph, max_states: int = 2_000_000, co_cluster: bool = True
) -> ExactResult:
    """Brute-force enumeration. Raises if the state space exceeds ``max_states``."""
    S = graph.state_space
    if S > max_states:
        raise ValueError(f"state space {S} > max_states {max_states}; infeasible")
    cards = graph.cards
    states = np.array(list(itertools.product(*[range(c) for c in cards])), dtype=np.int64)
    scores = np.empty(states.shape[0], dtype=np.float64)
    for idx in range(states.shape[0]):
        scores[idx] = graph.score(states[idx])
    m = float(np.max(scores))
    w = np.exp(scores - m)
    z = float(w.sum())
    p = w / z
    log_z = m + np.log(z)
    ent = float(-np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)))
    marg = [np.zeros(c, dtype=np.float64) for c in cards]
    for v in range(graph.n_vars):
        for val in range(cards[v]):
            marg[v][val] = p[states[:, v] == val].sum()
    cc = None
    if co_cluster and len(set(cards)) == 1:
        N = graph.n_vars
        cc = np.zeros((N, N), dtype=np.float64)
        for a in range(N):
            for b in range(a + 1, N):
                pr = p[states[:, a] == states[:, b]].sum()
                cc[a, b] = cc[b, a] = pr
        np.fill_diagonal(cc, 1.0)
    amap = int(np.argmax(scores))
    return ExactResult(
        log_z=log_z,
        entropy=ent,
        var_marginals=marg,
        co_cluster=cc,
        map_state=states[amap].copy(),
        p_map=float(p[amap]),
        n_states=int(states.shape[0]),
    )


# ───────────────────── exact marginals via variable elimination ────────────


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m.squeeze(axis) + np.log(np.sum(np.exp(a - m), axis=axis))
    return out


def _canon(scope: tuple[int, ...], table: np.ndarray) -> tuple[tuple[int, ...], np.ndarray]:
    """Return (sorted_scope, table transposed so axes follow sorted scope)."""
    order = sorted(range(len(scope)), key=lambda k: scope[k])
    new_scope = tuple(scope[k] for k in order)
    return new_scope, np.transpose(table, order)


def _fill_order(n: int, adj: dict[int, set[int]]) -> tuple[list[int], int]:
    """Greedy min-fill elimination order; returns (order, induced width).

    Min-fill (pick the vertex adding the fewest fill edges, min-degree tie-break)
    finds near-optimal orderings on grids — an L×L grid's treewidth is L, where
    naive min-degree can balloon to ~2L and blow up the 2**width VE tables.
    """
    g = {v: set(ns) for v, ns in adj.items()}
    remaining = set(range(n))
    order, width = [], 0
    while remaining:
        best, best_key = None, None
        for u in remaining:
            nb = g[u] & remaining
            nbl = list(nb)
            fill = sum(
                1
                for a in range(len(nbl))
                for b in range(a + 1, len(nbl))
                if nbl[b] not in g[nbl[a]]
            )
            key = (fill, len(nb))
            if best_key is None or key < best_key:
                best_key, best = key, u
        nb = g[best] & remaining
        width = max(width, len(nb))
        for a in nb:
            g[a] |= (nb - {a})
        remaining.discard(best)
        order.append(best)
    return order, width


def variable_elimination_marginals(
    graph: FactorGraph, *, max_width: int = 23
) -> tuple[list[np.ndarray], int]:
    """Exact single-variable marginals by bucket elimination (sum-product).

    Ground truth for bounded-treewidth graphs (e.g. an L×L grid has treewidth L)
    where brute force over the joint is hopeless but VE is cheap. Non-circular:
    unlike a long-PT reference, it does not reuse a sampler under test. Raises
    ``ValueError`` if the induced width exceeds ``max_width`` (2**width tables).
    """
    n = graph.n_vars
    canon = [_canon(f.scope, f.table) for f in graph.factors]
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for s, _t in canon:
        for a in s:
            for b in s:
                if a != b:
                    adj[a].add(b)
    order, width = _fill_order(n, adj)
    if width > max_width:
        raise ValueError(f"induced width {width} > max_width {max_width}; VE infeasible")

    cards = graph.cards
    marg: list[np.ndarray] = [None] * n  # type: ignore

    def reshape_to(union: list[int], s: tuple[int, ...], t: np.ndarray) -> np.ndarray:
        bshape = [cards[u] if u in s else 1 for u in union]
        return t.reshape(bshape)

    for target in range(n):
        factors = [(s, t) for s, t in canon]  # shallow copy of (scope, table)
        for v in order:
            if v == target:
                continue
            bucket = [(s, t) for s, t in factors if v in s]
            if not bucket:
                continue
            factors = [(s, t) for s, t in factors if v not in s]
            union = sorted(set().union(*[set(s) for s, _ in bucket]))
            shape = tuple(cards[u] for u in union)
            acc = np.zeros(shape, dtype=np.float64)
            for s, t in bucket:
                acc = acc + reshape_to(union, s, t)
            vaxis = union.index(v)
            with np.errstate(divide="ignore", invalid="ignore"):
                newt = _logsumexp(acc, axis=vaxis)
            newscope = tuple(u for u in union if u != v)
            factors.append((newscope, newt))
        logvec = np.zeros(cards[target], dtype=np.float64)
        for s, t in factors:
            if s == (target,):
                logvec = logvec + t
            elif s == ():
                pass  # global constant, irrelevant after normalisation
            else:
                raise RuntimeError(f"leftover factor scope {s} for target {target}")
        logvec -= np.max(logvec[np.isfinite(logvec)]) if np.any(np.isfinite(logvec)) else 0.0
        p = np.exp(logvec)
        marg[target] = p / p.sum()
    return marg, width


# ───────────────────────────── samplers ────────────────────────────────────


@dataclass
class SampleResult:
    var_marginals: list[np.ndarray]
    co_cluster: np.ndarray | None
    samples: np.ndarray  # (n_measure * n_chains, N)
    wall_s: float


def _marginals_from_samples(
    samples: np.ndarray, cards: tuple[int, ...], co_cluster: bool
) -> tuple[list[np.ndarray], np.ndarray | None]:
    marg = []
    for v in range(len(cards)):
        counts = np.bincount(samples[:, v], minlength=cards[v]).astype(np.float64)
        marg.append(counts / counts.sum())
    cc = None
    if co_cluster and len(set(cards)) == 1:
        N = len(cards)
        cc = np.zeros((N, N))
        for a in range(N):
            for b in range(a + 1, N):
                cc[a, b] = cc[b, a] = float(np.mean(samples[:, a] == samples[:, b]))
        np.fill_diagonal(cc, 1.0)
    return marg, cc


def block_gibbs(
    graph: FactorGraph,
    *,
    T: float = 1.0,
    n_chains: int = 256,
    burn_in: int = 300,
    n_measure: int = 100,
    thin: int = 1,
    scan: Sequence[int] | None = None,
    co_cluster: bool = True,
    seed: int = 0,
    init: np.ndarray | None = None,
) -> SampleResult:
    """Independent-ensemble single-site Gibbs at temperature ``T``.

    ``n_chains`` random-init replicas (the hardware-relevant strategy, scout-3
    EXP E). Single-site scan is the same transition kernel chromatic blocking
    runs (scout-3 EXP D), so this faithfully proxies THRML block-Gibbs on a
    dense graph; block parallelism is a throughput axis, not a mixing one.
    """
    rng = np.random.default_rng(seed)
    N = graph.n_vars
    if init is not None:
        states = np.tile(init.astype(np.int64), (n_chains, 1))
    else:
        states = np.stack(
            [rng.integers(0, c, size=n_chains) for c in graph.cards], axis=1
        ).astype(np.int64)
    order = list(range(N)) if scan is None else list(scan)
    t0 = time.time()
    measured: list[np.ndarray] = []
    total = burn_in + n_measure * thin
    for it in range(total):
        for i in order:
            logits = _cond_logits(graph, states, i) / T
            states[:, i] = _gumbel_argmax(logits, rng)
        if it >= burn_in and (it - burn_in) % thin == 0:
            measured.append(states.copy())
    samples = np.concatenate(measured, axis=0)
    marg, cc = _marginals_from_samples(samples, graph.cards, co_cluster)
    return SampleResult(marg, cc, samples, time.time() - t0)


def parallel_tempering(
    graph: FactorGraph,
    *,
    n_levels: int = 8,
    t_max: float = 8.0,
    n_chains: int = 256,
    burn_in: int = 300,
    n_measure: int = 100,
    thin: int = 1,
    scan: Sequence[int] | None = None,
    co_cluster: bool = True,
    seed: int = 0,
) -> SampleResult:
    """Replica exchange over ``n_levels`` geometric temperatures in [1, t_max].

    Even/odd adjacent swaps each sweep, Metropolis accept
    ``min(1, exp((β_a−β_b)(U_b−U_a)))`` with U = score (log p). Base-temperature
    (T=1) replicas supply the reported marginals. This is the sampler that
    crosses the metastability wall (scout-3): if PT ≈ exact where single-T Gibbs
    is far, the barrier is real and tempering pays for it (~n_levels× cost).
    """
    rng = np.random.default_rng(seed)
    N = graph.n_vars
    temps = np.geomspace(1.0, t_max, n_levels)
    betas = 1.0 / temps
    states = np.stack(
        [
            np.stack([rng.integers(0, c, size=n_chains) for c in graph.cards], axis=1)
            for _ in range(n_levels)
        ]
    ).astype(np.int64)  # (L, C, N)
    order = list(range(N)) if scan is None else list(scan)

    def energies(level: int) -> np.ndarray:
        u = np.zeros(n_chains)
        for f in graph.factors:
            sub = states[level][:, list(f.scope)]
            u += f.table[tuple(sub[:, k] for k in range(len(f.scope)))]
        return u

    t0 = time.time()
    measured: list[np.ndarray] = []
    total = burn_in + n_measure * thin
    for it in range(total):
        for lvl in range(n_levels):
            for i in order:
                logits = _cond_logits(graph, states[lvl], i) * betas[lvl]
                states[lvl][:, i] = _gumbel_argmax(logits, rng)
        start = it % 2
        for lvl in range(start, n_levels - 1, 2):
            ua, ub = energies(lvl), energies(lvl + 1)
            delta = (betas[lvl] - betas[lvl + 1]) * (ub - ua)
            accept = rng.random(n_chains) < np.exp(np.minimum(0.0, delta))
            tmp = states[lvl][accept].copy()
            states[lvl][accept] = states[lvl + 1][accept]
            states[lvl + 1][accept] = tmp
        if it >= burn_in and (it - burn_in) % thin == 0:
            measured.append(states[0].copy())
    samples = np.concatenate(measured, axis=0)
    marg, cc = _marginals_from_samples(samples, graph.cards, co_cluster)
    return SampleResult(marg, cc, samples, time.time() - t0)


# ───────────────────────── distributional metrics ──────────────────────────


def marginal_tv(p: list[np.ndarray], q: list[np.ndarray]) -> float:
    """Mean per-variable total-variation distance between two marginal sets."""
    tvs = [0.5 * float(np.abs(pi - qi).sum()) for pi, qi in zip(p, q)]
    return float(np.mean(tvs))


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance between two discrete distributions, in [0, 1].

    H(p,q) = (1/√2)·sqrt(Σ_k (√p_k − √q_k)²). This is the UAI inference-
    competition MAR error metric (averaged over unobserved variables).
    """
    return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2.0))


def mean_hellinger(
    approx: list[np.ndarray], true: list[np.ndarray], skip: set[int] | None = None
) -> float:
    """Average Hellinger error over (non-evidence) variables — the UAI MAR score."""
    skip = skip or set()
    errs = [hellinger(np.asarray(a), np.asarray(t))
            for i, (a, t) in enumerate(zip(approx, true)) if i not in skip]
    return float(np.mean(errs)) if errs else 0.0


def co_tv(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Mean |Δ| over the upper triangle of two co-clustering matrices."""
    if a is None or b is None:
        return None
    iu = np.triu_indices_from(a, k=1)
    return float(np.mean(np.abs(a[iu] - b[iu])))


# ───────────────────────── classical gauntlet ──────────────────────────────


@dataclass
class GauntletResult:
    name: str
    wall_s: float
    metric: dict  # task-defined (e.g. {"tv": ..., "solutions_found": ...})
    note: str = ""


def gauntlet_best_of_n(
    graph: FactorGraph, *, n: int = 1024, seed: int = 0
) -> GauntletResult:
    """Independent random states scored by the model — the matched-FLOPs control.

    Returns the empirical marginals over the top-weighted draws (softmax over
    scores). If best-of-N marginals already match exact, the task is easy.
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()
    states = np.stack(
        [rng.integers(0, c, size=n) for c in graph.cards], axis=1
    ).astype(np.int64)
    scores = np.array([graph.score(states[i]) for i in range(n)])
    m = float(np.max(scores))
    w = np.exp(scores - m)
    w /= w.sum()
    marg = []
    for v in range(graph.n_vars):
        mv = np.zeros(graph.cards[v])
        for val in range(graph.cards[v]):
            mv[val] = w[states[:, v] == val].sum()
        marg.append(mv)
    cc = None
    if len(set(graph.cards)) == 1:
        N = graph.n_vars
        cc = np.zeros((N, N))
        for a in range(N):
            for b in range(a + 1, N):
                cc[a, b] = cc[b, a] = float(w[states[:, a] == states[:, b]].sum())
        np.fill_diagonal(cc, 1.0)
    return GauntletResult(
        name="best_of_n_importance",
        wall_s=time.time() - t0,
        metric={"marginals": [mv.tolist() for mv in marg],
                "co_cluster": None if cc is None else cc.tolist()},
        note=f"n={n} importance-weighted random draws",
    )


def gauntlet_sa(
    graph: FactorGraph,
    *,
    n_chains: int = 64,
    sweeps: int = 500,
    t_hi: float = 4.0,
    t_lo: float = 0.05,
    seed: int = 0,
) -> GauntletResult:
    """Simulated annealing → best score found (the optimisation gauntlet, G0).

    If SA reaches the MAP cheaply, the task is an *optimisation* problem and a
    sampler buys nothing (Track A's lesson). The probe should compare this best
    score to its own samples' modes.
    """
    rng = np.random.default_rng(seed)
    N = graph.n_vars
    states = np.stack(
        [rng.integers(0, c, size=n_chains) for c in graph.cards], axis=1
    ).astype(np.int64)
    schedule = np.geomspace(t_hi, t_lo, sweeps)
    t0 = time.time()
    best = -np.inf
    for T in schedule:
        for i in range(N):
            logits = _cond_logits(graph, states, i) / T
            states[:, i] = _gumbel_argmax(logits, rng)
        cur = max(graph.score(states[c]) for c in range(n_chains))
        best = max(best, cur)
    return GauntletResult(
        name="simulated_annealing",
        wall_s=time.time() - t0,
        metric={"best_score": float(best)},
        note=f"{n_chains} chains × {sweeps} sweeps, geometric {t_hi}->{t_lo}",
    )


def _unary_pairwise(graph: FactorGraph):
    """Split a unary+pairwise graph into (unary logvecs, [(a,b,table)]).

    Raises on arity > 2 (the variational gauntlet here is pairwise, like BP).
    """
    unary = [np.zeros(c) for c in graph.cards]
    pair: list[tuple[int, int, np.ndarray]] = []
    for f in graph.factors:
        if len(f.scope) == 1:
            unary[f.scope[0]] = unary[f.scope[0]] + f.table
        elif len(f.scope) == 2:
            pair.append((f.scope[0], f.scope[1], f.table))
        else:
            raise ValueError("variational gauntlet supports only unary/pairwise factors")
    return unary, pair


def gauntlet_mean_field(
    graph: FactorGraph, *, iters: int = 2000, damping: float = 0.5, tol: float = 1e-8
) -> GauntletResult:
    """Naive mean-field coordinate ascent — the variational *lower*-bound baseline.

    q_i(x_i) ∝ exp(θ_i(x_i) + Σ_{j∈N(i)} E_{q_j}[θ_ij(x_i,x_j)]). Frustrated graphs
    break mean-field (symmetry breaking / mode collapse), so if MF were enough the
    task would be easy. Returns per-variable marginals q_i.
    """
    unary, pair = _unary_pairwise(graph)
    nbr: dict[int, list[tuple[int, np.ndarray]]] = {i: [] for i in range(graph.n_vars)}
    for a, b, t in pair:
        nbr[a].append((b, t))       # θ oriented (card_a, card_b)
        nbr[b].append((a, t.T))     # θ oriented (card_b, card_a)
    q = [np.ones(c) / c for c in graph.cards]
    t0 = time.time()
    for _ in range(iters):
        maxd = 0.0
        for i in range(graph.n_vars):
            logb = unary[i].copy()
            for j, tij in nbr[i]:
                logb = logb + tij @ q[j]
            logb -= logb.max()
            nq = np.exp(logb)
            nq /= nq.sum()
            nq = damping * q[i] + (1.0 - damping) * nq
            nq /= nq.sum()
            maxd = max(maxd, float(np.abs(nq - q[i]).max()))
            q[i] = nq
        if maxd < tol:
            break
    return GauntletResult(
        name="mean_field",
        wall_s=time.time() - t0,
        metric={"marginals": [qi.tolist() for qi in q]},
        note=f"naive MF, damping={damping}",
    )


def gauntlet_trw_bp(
    graph: FactorGraph,
    *,
    iters: int = 1000,
    damping: float = 0.5,
    rho: float | None = None,
) -> GauntletResult:
    """Tree-reweighted BP (Wainwright-Jaakkola-Willsky) — convexified sum-product.

    The principled "BP done right": a convex free energy with a unique optimum and
    an upper bound on logZ, far stronger than vanilla loopy BP on frustrated grids.
    Edge appearance probability ``rho`` defaults to a uniform spanning-tree weight
    min(1, (V-1)/E); at rho=1 the updates reduce exactly to loopy BP. Returns the
    TRW pseudomarginals.
    """
    unary, pair = _unary_pairwise(graph)
    E = len(pair)
    if rho is None:
        rho = min(1.0, (graph.n_vars - 1) / max(E, 1))
    rho = float(np.clip(rho, 1e-3, 1.0))
    msgs: dict[tuple[int, int, int], np.ndarray] = {}
    nbr: dict[int, list[tuple[int, int]]] = {i: [] for i in range(graph.n_vars)}
    for eidx, (a, b, _t) in enumerate(pair):
        msgs[(eidx, a, b)] = np.zeros(graph.cards[b])
        msgs[(eidx, b, a)] = np.zeros(graph.cards[a])
        nbr[a].append((eidx, b))
        nbr[b].append((eidx, a))
    t0 = time.time()
    for _ in range(iters):
        new = {}
        for eidx, (a, b, t) in enumerate(pair):
            for src, dst, tab in ((a, b, t), (b, a, t.T)):
                inc = unary[src].copy()
                for e2, other in nbr[src]:
                    if e2 == eidx and other == dst:
                        inc = inc - (1.0 - rho) * msgs[(e2, dst, src)]
                    else:
                        inc = inc + rho * msgs[(e2, other, src)]
                out = np.array([_logsumexp(inc + tab[:, xd] / rho, axis=0)
                                for xd in range(graph.cards[dst])])
                out -= _logsumexp(out, axis=0)
                new[(eidx, src, dst)] = out
        for k in msgs:
            msgs[k] = damping * msgs[k] + (1.0 - damping) * new[k]
    marg = []
    for i in range(graph.n_vars):
        b = unary[i].copy()
        for e2, other in nbr[i]:
            b = b + rho * msgs[(e2, other, i)]
        b -= _logsumexp(b, axis=0)
        marg.append(np.exp(b))
    return GauntletResult(
        name="trw_bp",
        wall_s=time.time() - t0,
        metric={"marginals": [m.tolist() for m in marg], "rho": rho},
        note=f"tree-reweighted BP, rho={rho:.3f}, damping={damping}",
    )


def _energy_all(graph: FactorGraph, states: np.ndarray) -> np.ndarray:
    """Vectorised log p(x) (up to const) for a batch of states (C, N)."""
    u = np.zeros(states.shape[0], dtype=np.float64)
    for f in graph.factors:
        sub = states[:, list(f.scope)]
        u = u + f.table[tuple(sub[:, k] for k in range(len(f.scope)))]
    return u


def gauntlet_ais(
    graph: FactorGraph,
    *,
    n_chains: int = 512,
    n_temps: int = 200,
    n_sweeps: int = 1,
    co_cluster: bool = False,
    seed: int = 0,
) -> GauntletResult:
    """Annealed importance sampling — the classical *single-temperature-path*
    sampler, the honest competitor to parallel tempering.

    Geometric-in-β path π_β ∝ exp(β·U) from uniform (β=0) to target (β=1), Gibbs
    transitions at each β, importance weights log w += Δβ·U(x). If AIS recovers the
    marginals at matched cost, the barrier does NOT specifically need replica
    exchange; if its weights degenerate (ESS collapses) while PT crosses, the
    barrier needs *exchange*, not just annealing. Reports marginals + ESS fraction.
    """
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
    w_unnorm = np.exp(logw - m)
    wsum = float(w_unnorm.sum())
    # AIS log-partition estimate: logẐ = logZ_0 + logsumexp(logw) − log N, with
    # the uniform base Z_0 = Π cards. Unbiased in Ẑ → biased LOW in logẐ by
    # Jensen, with bias growing as the weights degenerate (ESS → 0).
    logz0 = float(np.sum(np.log(np.asarray(graph.cards, dtype=np.float64))))
    log_z = logz0 + m + float(np.log(wsum)) - float(np.log(n_chains))
    w = w_unnorm / wsum
    ess = float(1.0 / np.sum(w ** 2) / n_chains)  # normalised effective sample size
    marg = []
    for v in range(N):
        mv = np.zeros(graph.cards[v])
        for val in range(graph.cards[v]):
            mv[val] = w[states[:, v] == val].sum()
        marg.append(mv)
    cc = None
    if co_cluster and len(set(graph.cards)) == 1:
        cc = np.zeros((N, N))
        for a in range(N):
            for b in range(a + 1, N):
                cc[a, b] = cc[b, a] = float(w[states[:, a] == states[:, b]].sum())
        np.fill_diagonal(cc, 1.0)
    return GauntletResult(
        name="ais",
        wall_s=time.time() - t0,
        metric={"marginals": [mv.tolist() for mv in marg],
                "ess_frac": ess,
                "log_z": log_z,
                "co_cluster": None if cc is None else cc.tolist()},
        note=f"AIS {n_temps} temps × {n_chains} chains, ESS={ess:.3f}",
    )


# ───────────────────────────── scorecard ───────────────────────────────────

GATES: list[tuple[str, str, str]] = [
    ("G0", "sampling-required",
     "Is the headline a SAMPLING / marginal / partition-function task, not a "
     "MAP / optimization with a closed-form or solver answer?"),
    ("G1", "no-factorization",
     "Does the posterior fail to factorise into independent / tractable pieces "
     "(no product-of-experts MAP, no chain/tree structure)?"),
    ("G1b", "no-solver-escape",
     "Do junction-tree / belief-propagation / DP / SAT-model-counter solvers "
     "fail or blow up at the natural size? (run the gauntlet)"),
    ("G2", "exact-infeasible",
     "Is brute-force / exact enumeration infeasible at the natural problem size?"),
    ("G3", "frustration->metastability",
     "Is single-temperature block-Gibbs metastable (TV to truth stays high, "
     "burn-in-invariant) due to frustration?"),
    ("G4", "tempering-crosses",
     "Does parallel tempering / replica exchange cross the barrier (TV -> MC "
     "floor), establishing the barrier is real and PT pays for it?"),
    ("G5", "real-scored-beats-gauntlet",
     "Is the task REAL and SCORED, and does sampling beat the classical "
     "gauntlet (best-of-N, GPU-SA, ApproxMC/UniGen, BP) on the distributional "
     "metric?"),
    ("G6", "cost-model-favorable",
     "Does the cost model (K·C_neural vs R·C_gibbs / replica overhead) favour "
     "the sampler, and does it scale the right way with problem size?"),
    ("G7", "hardware-native-encoding",
     "Does the factor graph map to a SPARSE, LOCAL Ising/Potts a thermodynamic "
     "sampler runs natively (low degree, bounded cardinality, no dense all-to-"
     "all blowup)?"),
    ("G8", "approximation-insensitive",
     "Does the task tolerate fast APPROXIMATE samples (DTM-style) rather than "
     "needing exact rare modes? (a task that needs exact tails is a poor fit)"),
    ("G9", "amortized-cost",
     "Does the END-TO-END amortised cost (encoding + sampling + decoding + the "
     "neural passes it replaces) favour the hardware, not just one kernel?"),
]

_VERDICTS = {"pass", "fail", "partial", "unknown", "na"}


@dataclass
class Scorecard:
    """G0–G9 hardware-fit verdicts for one probe."""

    probe: str
    verdicts: dict[str, str] = field(default_factory=dict)  # gate id -> verdict
    evidence: dict[str, str] = field(default_factory=dict)  # gate id -> note

    def set(self, gate: str, verdict: str, evidence: str = "") -> None:
        if gate not in {g[0] for g in GATES}:
            raise KeyError(f"unknown gate {gate}")
        if verdict not in _VERDICTS:
            raise ValueError(f"verdict must be one of {_VERDICTS}")
        self.verdicts[gate] = verdict
        self.evidence[gate] = evidence

    def summary(self) -> str:
        lines = [f"===== SCORECARD: {self.probe} ====="]
        for gid, short, _q in GATES:
            v = self.verdicts.get(gid, "unknown")
            ev = self.evidence.get(gid, "")
            mark = {"pass": "PASS", "fail": "FAIL", "partial": "~~~~",
                    "unknown": "????", "na": "n/a "}[v]
            lines.append(f"  [{mark}] {gid:3s} {short:28s} {ev}")
        passes = sum(1 for g in GATES if self.verdicts.get(g[0]) == "pass")
        fails = sum(1 for g in GATES if self.verdicts.get(g[0]) == "fail")
        lines.append(f"  -> {passes} pass / {fails} fail / {len(GATES)} gates")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "probe": self.probe,
            "gates": [
                {"id": g[0], "short": g[1], "question": g[2],
                 "verdict": self.verdicts.get(g[0], "unknown"),
                 "evidence": self.evidence.get(g[0], "")}
                for g in GATES
            ],
        }


# ───────────────────────────── report writer ───────────────────────────────


def write_report(path: str | Path, payload: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=_json_default))
    return out


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dict__"):
        return asdict(o) if hasattr(o, "__dataclass_fields__") else o.__dict__
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ───────────────────────────── builders ────────────────────────────────────


def potts_pairwise(
    cards: Sequence[int],
    unary: Sequence[np.ndarray],
    edges: Sequence[tuple[int, int]],
    couplings: Sequence[np.ndarray],
) -> FactorGraph:
    """Build a pairwise Potts/categorical MRF from unary + edge log-potentials."""
    factors = [Factor((i,), np.asarray(u, dtype=np.float64)) for i, u in enumerate(unary)]
    for (a, b), w in zip(edges, couplings):
        factors.append(Factor((a, b), np.asarray(w, dtype=np.float64)))
    return FactorGraph(tuple(int(c) for c in cards), factors)


# ───────────────────────────── evidence ────────────────────────────────────


def parse_uai_evidence(path: str | Path) -> dict[int, int]:
    """Parse a UAI ``.evid`` file: ``k var_1 val_1 ... var_k val_k`` → {var: val}.

    A lone ``0`` (the common grid case) means no evidence.
    """
    toks = [int(x) for x in Path(path).read_text().split()]
    if not toks:
        return {}
    k = toks[0]
    pairs = toks[1 : 1 + 2 * k]
    return {pairs[2 * j]: pairs[2 * j + 1] for j in range(k)}


def clamp_evidence(graph: FactorGraph, evid: dict[int, int]) -> FactorGraph:
    """Pin evidence variables by adding hard unary log-potentials (−inf elsewhere).

    Variable indices are preserved (pinned vars stay in the graph as deltas), so
    downstream marginal indexing is unchanged; evidence vars are excluded from
    the MAR average by the caller. Works uniformly for exact/VE/Gibbs/PT/BP.
    """
    if not evid:
        return graph
    extra: list[Factor] = []
    for v, val in evid.items():
        u = np.full(graph.cards[v], -np.inf, dtype=np.float64)
        u[val] = 0.0
        extra.append(Factor((v,), u))
    return FactorGraph(graph.cards, list(graph.factors) + extra)


# ───────────────────────────── self-test ───────────────────────────────────


def _selftest() -> None:
    """A small frustrated Potts: exact vs Gibbs vs PT vs gauntlet, sanity TV."""
    rng = np.random.default_rng(0)
    n, L, w = 6, 3, 6.0
    unary = [rng.normal(0, 0.5, size=L) for _ in range(n)]
    edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
    couplings = []
    for _ in edges:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        couplings.append(sign * w * np.eye(L))  # ferro / antiferro Potts
    g = potts_pairwise([L] * n, unary, edges, couplings)
    print(f"[selftest] n={n} L={L} w={w} states={g.state_space}")

    ex = exact_marginals(g)
    gb = block_gibbs(g, T=1.0, n_chains=128, burn_in=200, n_measure=80, seed=1)
    pt = parallel_tempering(g, n_levels=6, t_max=2 * w, n_chains=128,
                            burn_in=200, n_measure=80, seed=1)
    bon = gauntlet_best_of_n(g, n=2048, seed=2)
    bon_marg = [np.asarray(m) for m in bon.metric["marginals"]]

    ve_marg, ve_width = variable_elimination_marginals(g)
    ve_err = marginal_tv(ve_marg, ex.var_marginals)
    print(f"  VE-vs-brute TV  = {ve_err:.2e}  (induced width {ve_width}) "
          f"{'OK' if ve_err < 1e-9 else 'MISMATCH!'}")

    print(f"  exact: logZ={ex.log_z:.3f}  H={ex.entropy:.3f}  p_map={ex.p_map:.3f}")
    print(f"  TV(gibbs,exact) = {marginal_tv(gb.var_marginals, ex.var_marginals):.4f}")
    print(f"  TV(PT,exact)    = {marginal_tv(pt.var_marginals, ex.var_marginals):.4f}")
    print(f"  TV(bestN,exact) = {marginal_tv(bon_marg, ex.var_marginals):.4f}")
    print(f"  co_tv(gibbs)    = {co_tv(gb.co_cluster, ex.co_cluster)}")
    print(f"  co_tv(PT)       = {co_tv(pt.co_cluster, ex.co_cluster)}")

    # ── gauntlet correctness validation ──────────────────────────────────────
    # (1) TRW@rho=1 must be EXACT on a tree (reduces to loopy BP, exact on trees).
    tn = 6
    tunary = [rng.normal(0, 0.8, size=2) for _ in range(tn)]
    tedges = [(i, i + 1) for i in range(tn - 1)]  # a path = tree
    tcoup = [rng.normal(0, 1.0, size=(2, 2)) for _ in tedges]
    tree = potts_pairwise([2] * tn, tunary, tedges, tcoup)
    tex = exact_marginals(tree, co_cluster=False)
    trw_tree = gauntlet_trw_bp(tree, rho=1.0, iters=400, damping=0.3)
    trw_tree_marg = [np.asarray(m) for m in trw_tree.metric["marginals"]]
    trw_tree_err = marginal_tv(trw_tree_marg, tex.var_marginals)
    print(f"  [validate] TRW(rho=1)-vs-exact on a TREE = {trw_tree_err:.2e} "
          f"{'OK' if trw_tree_err < 1e-6 else 'BROKEN!'}")

    # (2) AIS with a generous schedule must converge to exact on the frustrated g.
    ais = gauntlet_ais(g, n_chains=2048, n_temps=400, n_sweeps=1, seed=3)
    ais_marg = [np.asarray(m) for m in ais.metric["marginals"]]
    ais_err = marginal_tv(ais_marg, ex.var_marginals)
    print(f"  [validate] AIS(400 temps)-vs-exact = {ais_err:.4f} "
          f"(ESS={ais.metric['ess_frac']:.3f}) "
          f"{'OK' if ais_err < 0.05 else 'CHECK'}")

    # (3) MF + TRW(uniform rho) on the frustrated graph — expected imperfect.
    mf = gauntlet_mean_field(g)
    mf_marg = [np.asarray(m) for m in mf.metric["marginals"]]
    trw = gauntlet_trw_bp(g)
    trw_marg = [np.asarray(m) for m in trw.metric["marginals"]]
    print(f"  TV(mean_field,exact) = {marginal_tv(mf_marg, ex.var_marginals):.4f}")
    print(f"  TV(trw_bp,exact)     = {marginal_tv(trw_marg, ex.var_marginals):.4f} "
          f"(rho={trw.metric['rho']:.3f})")

    sc = Scorecard("selftest")
    sc.set("G0", "pass", "marginals are the target")
    sc.set("G1", "pass", "frustrated complete graph, no factorisation")
    sc.set("G2", "na", f"state space {g.state_space} still enumerable here")
    print(sc.summary())


if __name__ == "__main__":
    _selftest()
