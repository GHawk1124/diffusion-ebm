"""Track B scout 4: do REAL full-window RMC graphs land in the hard regime?

The first three scouts established the synthetic-hardness thesis (Pareto
crossover, metastability wall, tempering rescue) on a *dialed* frustrated Potts
with LM-grounded fields. The one open research question they left: is there a
**real** task whose natural graph is hard, or is the hero figure forever
synthetic? The splitability probe (scout 1) said RMC is too easy — but it
measured the wrong object: it split each window into per-entity Jaccard
*components* (capped at n<=5) and threw away 7 over-size components. The
genuinely hard object is the **joint latent-partition posterior over ALL masked
holes in a window** (n=4..10 in the cache) with the grouping unknown.

This probe measures that object directly. The model is frustrated correlation
clustering over the n holes of one window:

    log p(z) = Σ_{groups g of z}  poolZ(g)                # token evidence
             + Σ_{i<j} β (J_ij − τ) (2·[z_i==z_j] − 1)     # Jaccard prior

where the **token marginal factorises per group** under hard within-group
equality (one shared token per entity): a group emits a single vocab token, so

    poolZ(g) = logsumexp_t  Σ_{h in g} l_h(t)            # pooled product-of-experts

summed over the *intersection* of the group's top-k candidate sets (tokens
outside a member's top-k contribute −inf). A singleton's poolZ is just
``logsumexp`` over its own top-k logits; the per-hole full-vocab normaliser is
constant across partitions and drops out. This is exactly Track A's
``hard_eq_map`` pooling, *summed* (marginal) rather than *maxed* (MAP), and it
is what makes the exact partition posterior tractable at the realistic L=256:
precompute poolZ for every hole-subset once (≤2^n−1 ≤ 1023 subsets for n≤10),
then every Bell(n) partition (Bell(10)=115975) is a sum over its groups — no
L^n joint-token enumeration (256^10 ≈ 1e24, hopeless).

Two experiments, both on the frozen dev cache (no GPU, numpy only):

  * **EXP1 census** — exact p(z) per item: partition entropy (ambiguity),
    p(MAP), p(gold grouping), ARI(MAP, gold), co-clustering uncertainty, and the
    exact cost (Bell(n)) vs the infeasible naive joint (L^n). Decisive readout:
    are real full-window partition posteriors *multimodal / ambiguous*, or
    peaked (→ easy, like the components)?
  * **EXP2 metastability** — on the most-ambiguous items, a partition-space
    block-Gibbs sampler (single-site z_i reassignment, tokens marginalised via
    poolZ) at T=1 (independent-ensemble, the hardware-relevant strategy) vs
    parallel tempering, both vs the exact co-clustering marginals. Decisive
    readout: even where exact is feasible, is single-T Gibbs metastable on real
    LM-grounded frustration, and does tempering cross it?

Reuses the scout-1 cache and partition helpers. Run from repo root (no nix
needed for the numpy stage; numpy still wants the zlib LD path on this host):

    LD_LIBRARY_PATH="<zlib>:/run/opengl-driver/lib:..." \\
        .venv/bin/python experiments/probe_partition_hardness.py \\
            --cache results/probe_splitability_cache.json \\
            --out results/probe_partition_hardness.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe_splitability import (  # noqa: E402
    adjusted_rand,
    canonical_rgs,
    set_partitions,
)


# ───────────────────────── exact partition posterior ───────────────────────


def _logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(a - m))))


def pooled_logz_by_subset(cand_ids, cand_logits, n: int) -> dict[int, float]:
    """poolZ(S) for every non-empty hole-subset S (keyed by bitmask).

    poolZ(S) = logsumexp_t Σ_{h in S} l_h(t) over the *intersection* of the
    members' candidate vocab ids. Built over the item's union vocab support as a
    dense (n, V) logit table with −inf off-support, so a subset's pooled score is
    ``logsumexp`` of the summed rows (the sum is −inf wherever any member lacks
    the token → enforces the intersection automatically).
    """
    support = sorted({v for h in range(n) for v in cand_ids[h]})
    vpos = {v: c for c, v in enumerate(support)}
    V = len(support)
    M = np.full((n, V), -np.inf, dtype=np.float64)
    for h in range(n):
        for v, lg in zip(cand_ids[h], cand_logits[h]):
            M[h, vpos[v]] = lg

    out: dict[int, float] = {}
    for mask in range(1, 1 << n):
        rows = [h for h in range(n) if mask & (1 << h)]
        summed = M[rows].sum(axis=0)  # −inf where the intersection is empty
        out[mask] = _logsumexp(summed)
    return out


def _group_masks(labels) -> list[int]:
    """Bitmask per group of a label tuple."""
    masks: dict[int, int] = {}
    for h, lbl in enumerate(labels):
        masks[lbl] = masks.get(lbl, 0) | (1 << h)
    return list(masks.values())


def exact_partition_posterior(
    cand_ids, cand_logits, jac: np.ndarray, beta: float, tau: float
) -> dict:
    """Exact normalised p(z) over all Bell(n) partitions at full L.

    Returns the partition log-probs, entropy, MAP, co-clustering matrix, and the
    per-pair same-group probability — everything EXP1/EXP2 need.
    """
    n = len(cand_ids)
    pooled = pooled_logz_by_subset(cand_ids, cand_logits, n)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    w = {(i, j): beta * (float(jac[i, j]) - tau) for (i, j) in pairs}

    parts = list(set_partitions(n))
    logp = np.empty(len(parts), dtype=np.float64)
    for pi, z in enumerate(parts):
        lik = sum(pooled[m] for m in _group_masks(z))
        prior = sum(w[(i, j)] * (1.0 if z[i] == z[j] else -1.0) for (i, j) in pairs)
        logp[pi] = lik + prior
    logp -= _logsumexp(logp)
    p = np.exp(logp)

    entropy = float(-np.sum(p * np.log(np.where(p > 0, p, 1.0))))
    map_idx = int(np.argmax(p))
    co = np.zeros((n, n), dtype=np.float64)
    for pi, z in enumerate(parts):
        for (i, j) in pairs:
            if z[i] == z[j]:
                co[i, j] += p[pi]
                co[j, i] += p[pi]
    return {
        "n": n,
        "n_partitions": len(parts),
        "parts": parts,
        "p": p,
        "entropy": entropy,
        "max_entropy": float(np.log(len(parts))),
        "map_z": canonical_rgs(parts[map_idx]),
        "p_map": float(p[map_idx]),
        "co": co,
    }


def jaccard_matrix(cand_ids) -> np.ndarray:
    n = len(cand_ids)
    sets = [set(c) for c in cand_ids]
    J = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(a + 1, n):
            u = len(sets[a] | sets[b])
            J[a, b] = J[b, a] = (len(sets[a] & sets[b]) / u) if u else 0.0
    return J


# ───────────────────── partition-space block-Gibbs sampler ─────────────────


def _logp_one(z: np.ndarray, pooled: dict[int, float], w: np.ndarray) -> float:
    """log p(z) for a single label vector (unnormalised)."""
    lik = 0.0
    for m in _group_masks(tuple(int(x) for x in z)):
        lik += pooled[m]
    n = len(z)
    prior = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            prior += w[i, j] * (1.0 if z[i] == z[j] else -1.0)
    return lik + prior


def _gibbs_sweep_partition(
    Z: np.ndarray, pooled: dict[int, float], w: np.ndarray, T: float, rng
) -> None:
    """One systematic-scan single-site Gibbs sweep over partition labels (in place).

    For each chain and hole i, the candidate labels are the labels currently used
    by the other holes plus one fresh label (so splits and merges are both
    reachable). Samples z_i ∝ exp(logp(z)/T) via Gumbel-max.
    """
    C, n = Z.shape
    for c in range(C):
        z = Z[c]
        for i in range(n):
            others = sorted(set(int(z[h]) for h in range(n) if h != i))
            fresh = (max(others) + 1) if others else 0
            cand_labels = others + [fresh]
            scores = np.empty(len(cand_labels), dtype=np.float64)
            for li, lbl in enumerate(cand_labels):
                z[i] = lbl
                scores[li] = _logp_one(z, pooled, w)
            g = rng.gumbel(size=len(cand_labels))
            z[i] = cand_labels[int(np.argmax(scores / T + g))]


def _co_from_samples(samples: np.ndarray, n: int) -> np.ndarray:
    """Co-clustering matrix P(i,j same group) from (S, n) label samples."""
    co = np.zeros((n, n), dtype=np.float64)
    S = samples.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            same = float(np.mean(samples[:, i] == samples[:, j]))
            co[i, j] = co[j, i] = same
    return co


def vanilla_partition_gibbs(
    pooled, w, n, n_chains, burn_in, n_measure, seed
) -> np.ndarray:
    """Independent-ensemble single-T (T=1) partition Gibbs → co-clustering matrix.

    Each chain starts at the all-singletons partition with a random relabel and
    equilibrates at T=1 (the hardware-relevant many-replica strategy; the robust
    metastability signal per scout-3 EXP E).
    """
    rng = np.random.default_rng(seed)
    Z = np.tile(np.arange(n), (n_chains, 1))
    # random init: each hole an independent label in 0..n-1
    Z = rng.integers(0, n, size=(n_chains, n))
    for _ in range(burn_in):
        _gibbs_sweep_partition(Z, pooled, w, 1.0, rng)
    coll = []
    for _ in range(n_measure):
        _gibbs_sweep_partition(Z, pooled, w, 1.0, rng)
        coll.append(Z.copy())
    return _co_from_samples(np.concatenate(coll, axis=0), n)


def pt_partition_gibbs(
    pooled, w, n, n_chains, K, T_max, n_sweep, n_measure, seed
) -> tuple[np.ndarray, float]:
    """Parallel tempering over partition labels → cold-replica co-clustering.

    K geometric temperatures 1..T_max, per-sweep Gibbs at each level then even/odd
    adjacent swaps with acceptance min(1, exp((β_a−β_b)(U_b−U_a))), U=log p.
    """
    rng = np.random.default_rng(seed)
    Ts = np.geomspace(1.0, T_max, K)
    betas = 1.0 / Ts
    Z = rng.integers(0, n, size=(K, n_chains, n))
    swaps = acc = 0

    def sweep(measure):
        nonlocal swaps, acc
        for k in range(K):
            _gibbs_sweep_partition(Z[k], pooled, w, float(Ts[k]), rng)
        U = np.empty((K, n_chains), dtype=np.float64)
        for k in range(K):
            for c in range(n_chains):
                U[k, c] = _logp_one(Z[k, c], pooled, w)
        parity = measure & 1
        for k in range(parity, K - 1, 2):
            d = (betas[k] - betas[k + 1]) * (U[k + 1] - U[k])
            a = np.minimum(1.0, np.exp(np.minimum(d, 0.0)))
            r = rng.random(n_chains)
            m = r < a
            swaps += n_chains
            acc += int(m.sum())
            tmp = Z[k, m].copy()
            Z[k, m] = Z[k + 1, m]
            Z[k + 1, m] = tmp

    for s in range(n_sweep):
        sweep(s)
    coll = []
    for s in range(n_measure):
        sweep(s)
        coll.append(Z[0].copy())
    co = _co_from_samples(np.concatenate(coll, axis=0), n)
    return co, (acc / swaps if swaps else 0.0)


def co_tv(a: np.ndarray, b: np.ndarray) -> float:
    """Mean |Δ| over the off-diagonal co-clustering pairs (∈ [0,1])."""
    n = a.shape[0]
    iu = np.triu_indices(n, 1)
    return float(np.mean(np.abs(a[iu] - b[iu]))) if len(iu[0]) else 0.0


# ──────────────────────────── experiments ──────────────────────────────────


def exp_census(records, beta: float, tau: float) -> dict:
    """EXP1: exact partition-posterior hardness census over all full-window items."""
    print(f"\n===== EXP1 census (β={beta}, τ={tau}) =====")
    rows = []
    for rec in records:
        cand_ids = rec["cand_ids"]
        cand_logits = rec["cand_logits"]
        gold = canonical_rgs(rec["chain_labels"])
        n = len(cand_ids)
        J = jaccard_matrix(cand_ids)
        t0 = time.time()
        post = exact_partition_posterior(cand_ids, cand_logits, J, beta, tau)
        wall = time.time() - t0
        iu = np.triu_indices(n, 1)
        co_uncert = float(np.mean(np.minimum(post["co"][iu], 1 - post["co"][iu]))) \
            if len(iu[0]) else 0.0
        # gold partition probability
        gp = 0.0
        for z, pv in zip(post["parts"], post["p"]):
            if canonical_rgs(z) == gold:
                gp += float(pv)
        rows.append({
            "item_id": rec["item_id"], "corpus": rec["corpus"], "n": n,
            "n_entities_gold": len(set(gold)),
            "n_partitions": post["n_partitions"],
            "entropy": post["entropy"], "max_entropy": post["max_entropy"],
            "entropy_frac": post["entropy"] / post["max_entropy"]
                if post["max_entropy"] > 0 else 0.0,
            "p_map": post["p_map"], "p_gold": gp,
            "map_is_gold": post["map_z"] == gold,
            "ari_map_gold": adjusted_rand(post["map_z"], gold),
            "co_uncertainty": co_uncert, "wall": wall,
        })
    return rows


def summarise_census(rows: list[dict]) -> dict:
    import collections
    n_dist = dict(sorted(collections.Counter(r["n"] for r in rows).items()))
    arr = lambda key: np.array([r[key] for r in rows], dtype=np.float64)
    ent = arr("entropy")
    entf = arr("entropy_frac")
    pmap = arr("p_map")
    # "ambiguous" = posterior not dominated by one partition
    ambiguous = float(np.mean(pmap < 0.9))
    multimodal = float(np.mean(entf > 0.2))
    summ = {
        "n_items": len(rows),
        "n_distribution": n_dist,
        "max_n": int(max(r["n"] for r in rows)),
        "mean_entropy": float(ent.mean()),
        "mean_entropy_frac": float(entf.mean()),
        "mean_p_map": float(pmap.mean()),
        "frac_ambiguous_pmap_lt_0.9": ambiguous,
        "frac_multimodal_entfrac_gt_0.2": multimodal,
        "map_is_gold_rate": float(np.mean([r["map_is_gold"] for r in rows])),
        "mean_ari_map_gold": float(arr("ari_map_gold").mean()),
        "mean_p_gold": float(arr("p_gold").mean()),
        "mean_co_uncertainty": float(arr("co_uncertainty").mean()),
        "max_bell_wall": float(arr("wall").max()),
    }
    print(f"  items={summ['n_items']}  n-dist={n_dist}  max n={summ['max_n']}")
    print(f"  mean entropy={summ['mean_entropy']:.3f} "
          f"(frac of max {summ['mean_entropy_frac']:.3f})  "
          f"mean p(MAP)={summ['mean_p_map']:.3f}")
    print(f"  ambiguous (p(MAP)<0.9): {ambiguous:.3f}   "
          f"multimodal (Hfrac>0.2): {multimodal:.3f}")
    print(f"  MAP==gold: {summ['map_is_gold_rate']:.3f}   "
          f"mean ARI(MAP,gold)={summ['mean_ari_map_gold']:.3f}   "
          f"mean p(gold)={summ['mean_p_gold']:.3f}")
    print(f"  max Bell-enum wall={summ['max_bell_wall']:.3f}s")
    return summ


def exp_metastability(records, rows, args) -> list[dict]:
    """EXP2: partition Gibbs (vanilla ensemble + PT) vs exact on hard items."""
    # rank by ambiguity (entropy_frac), break ties toward larger n
    order = sorted(range(len(rows)),
                   key=lambda i: (rows[i]["entropy_frac"], rows[i]["n"]),
                   reverse=True)
    picks = order[: args.meta_items]
    print(f"\n===== EXP2 metastability (top {len(picks)} by ambiguity) =====")
    print(f"  {'item':>26} {'n':>2} {'Hfrac':>6} {'TV(van,ex)':>11} "
          f"{'TV(PT,ex)':>10} {'swap':>5}")
    out = []
    for idx in picks:
        rec = records[idx]
        cand_ids, cand_logits = rec["cand_ids"], rec["cand_logits"]
        n = len(cand_ids)
        J = jaccard_matrix(cand_ids)
        post = exact_partition_posterior(cand_ids, cand_logits, J, args.beta, args.tau)
        pooled = pooled_logz_by_subset(cand_ids, cand_logits, n)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        w = np.zeros((n, n))
        for (i, j) in pairs:
            w[i, j] = w[j, i] = args.beta * (float(J[i, j]) - args.tau)
        van = vanilla_partition_gibbs(
            pooled, w, n, args.meta_chains, args.burn_in, args.n_measure, args.seed + idx)
        pt, swap = pt_partition_gibbs(
            pooled, w, n, args.meta_chains, args.pt_levels, args.pt_tmax,
            args.burn_in, args.n_measure, args.seed + idx)
        tv_v = co_tv(van, post["co"])
        tv_p = co_tv(pt, post["co"])
        r = {"item_id": rec["item_id"], "n": n,
             "entropy_frac": rows[idx]["entropy_frac"],
             "tv_vanilla_exact": tv_v, "tv_pt_exact": tv_p, "swap_rate": swap}
        out.append(r)
        print(f"  {rec['item_id'][:26]:>26} {n:>2} {rows[idx]['entropy_frac']:>6.3f} "
              f"{tv_v:>11.3f} {tv_p:>10.3f} {swap:>5.2f}")
    if out:
        mv = float(np.mean([r["tv_vanilla_exact"] for r in out]))
        mp = float(np.mean([r["tv_pt_exact"] for r in out]))
        print(f"  mean TV(vanilla,exact)={mv:.3f}   mean TV(PT,exact)={mp:.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="results/probe_splitability_cache.json")
    ap.add_argument("--out", default="results/probe_partition_hardness.json")
    ap.add_argument("--beta", type=float, default=4.0, help="Jaccard prior strength")
    ap.add_argument("--tau", type=float, default=0.3, help="Jaccard merge threshold")
    ap.add_argument("--beta-list", default="0,2,4,8",
                    help="β values for the census sweep")
    ap.add_argument("--max-n", type=int, default=11,
                    help="skip items with more holes (Bell(n) guard)")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--experiments", default="census,meta",
                    help="comma list: census (EXP1), meta (EXP2)")
    ap.add_argument("--meta-items", type=int, default=16)
    ap.add_argument("--meta-chains", type=int, default=128)
    ap.add_argument("--burn-in", type=int, default=300)
    ap.add_argument("--n-measure", type=int, default=100)
    ap.add_argument("--pt-levels", type=int, default=8)
    ap.add_argument("--pt-tmax", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.beta_list = [float(x) for x in args.beta_list.split(",")]
    exps = set(args.experiments.split(","))

    blob = json.loads(Path(args.cache).read_text())
    records = [r for r in blob["records"] if len(r["cand_ids"]) <= args.max_n]
    skipped = len(blob["records"]) - len(records)
    if args.max_items:
        records = records[: args.max_items]
    print(f"[cache] {len(records)} items (skipped {skipped} with n>{args.max_n})")

    report = {}
    if Path(args.out).exists():
        report = json.loads(Path(args.out).read_text())
    report["config"] = vars(args)

    rows = None
    if "census" in exps:
        sweep = {}
        for beta in args.beta_list:
            rows_b = exp_census(records, beta, args.tau)
            sweep[str(beta)] = summarise_census(rows_b)
            if beta == args.beta:
                rows = rows_b
        report["census_sweep"] = sweep
        if rows is None:  # default β not in the sweep list
            rows = exp_census(records, args.beta, args.tau)
        report["census_rows"] = rows
    if "meta" in exps:
        if rows is None:
            rows = exp_census(records, args.beta, args.tau)
        report["metastability"] = exp_metastability(records, rows, args)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=_json_default))
    print(f"\nwrote {args.out}")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    main()
