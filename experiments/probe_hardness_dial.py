"""Track B hardware-regime scout: when does THRML block-Gibbs earn its keep?

Track A (oracle) and the splitability probe (de-oracle) both found the *same*
thing: every inference instance RMC actually produces is computationally easy —
closed-form pooling or cheap exact enumeration dominates the sampler. The common
cause is the benchmark, not the method: RMC windows are 64 tokens with 2–4
entities, so the frustrated graph never gets big or dense enough for sampling to
matter. So the real question for the thermodynamic-hardware thesis is NOT "tune
the energy harder" but "is there a regime where block-Gibbs strictly beats exact
inference — and does block-Gibbs stay faithful as the graph hardens, or does it
fall to metastability right where exact dies?"

This scout dials hardness directly. It builds a **frustrated Potts model with
real MDLM logit fields**: n categorical variables (holes) over a shared L-state
alphabet; the unary fields are real per-hole top-L MDLM logits drawn from the
splitability-probe cache (so the *fields* are LM-grounded); the couplings are a
controlled frustrated graph (attraction = +w on the equal-state diagonal,
repulsion = −w, exactly the equality/inequality factors). Using a shared
alphabet makes equality/repulsion *active* and the frustration tunable. This is
the canonical thermodynamic-hardware workload (frustrated Ising/Potts), and the
``FrustratedSampler`` (sampler/thrml_latent_partition.py) runs THRML block-Gibbs
on it unchanged — so this also exercises that scaffold.

The energy matches THRML bit-for-bit (sample ∝ exp(Σ factor weights)):

    log p(x) = Σ_i unary_i(x_i)
             + w_a Σ_{(i,j)∈attract} 1[x_i = x_j]
             − w_r Σ_{(i,j)∈repel}   1[x_i = x_j]

so a pure-numpy brute-force enumeration over L^n states is the exact ground
truth wherever it is still tractable.

Two experiments:

  EXP A — Pareto crossover. Spin-glass instances (random ± couplings) of growing
    size n. Wherever L^n ≤ budget, run exact brute force; always run block-Gibbs.
    Record wall-time(exact) vs wall-time(Gibbs) and the marginal TV between them.
    The hero result: exact wall-time explodes ~L^n and goes infeasible while
    Gibbs stays ~flat AND faithful (low TV) in the regime where we can still
    check it.

  EXP B — Metastability. Fixed n (small enough that exact is cheap), sweep the
    coupling weight w and the repulsion fraction f. Record TV(Gibbs, exact). The
    killer risk: as the posterior hardens into many symmetric modes, does block-
    Gibbs stop mixing (TV spikes)? Under label symmetry the exact marginal is
    near-flat, so a mode-locked sampler shows up as high TV — a clean detector.

Reuses the cached real logits from the splitability probe (no GPU forward):

    LD_LIBRARY_PATH="<zlib>:/run/opengl-driver/lib:..." JAX_PLATFORMS=cpu \\
        .venv/bin/python experiments/probe_hardness_dial.py \\
            --cache results/probe_splitability_cache.json \\
            --out results/probe_hardness_dial.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion_ebm.sampler import thrml_latent_partition as flp  # noqa: E402


# ───────────────────────────── field bank ──────────────────────────────────


def load_field_bank(cache_path: Path, L: int) -> np.ndarray:
    """Real per-hole top-L MDLM logits from the splitability cache → (M, L)."""
    blob = json.loads(cache_path.read_text())
    rows = []
    for rec in blob["records"]:
        for logits in rec["cand_logits"]:
            if len(logits) >= L:
                rows.append(logits[:L])
    bank = np.asarray(rows, dtype=np.float64)
    # Center each field (constant shift is irrelevant to the distribution) so
    # the top state sits at 0 and coupling weights are on a comparable scale.
    bank -= bank.max(axis=1, keepdims=True)
    return bank


# ─────────────────────────── instance builder ──────────────────────────────


def make_edges(
    n: int,
    structure: str,
    rng: np.random.Generator,
    repel_fraction: float = 0.5,
    n_groups: int = 2,
    p_edge: float = 1.0,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Build attraction / repulsion edge lists for a frustrated graph.

    structure='spinglass': each present pair (Erdős–Rényi p_edge) is repulsion
      with prob repel_fraction, else attraction. Random ± couplings → contradictory
      cycles → genuine frustration (the canonical hard / spin-glass workload).
    structure='planted': planted n_groups partition; attract within, repel across
      (satisfiable / unfrustrated up to label symmetry — the easy RMC-aligned case).
    """
    attract: list[tuple[int, int]] = []
    repel: list[tuple[int, int]] = []
    if structure == "planted":
        labels = [i % n_groups for i in range(n)]
        return flp.partition_to_edges(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if p_edge < 1.0 and rng.random() > p_edge:
                continue
            (repel if rng.random() < repel_fraction else attract).append((i, j))
    return attract, repel


def make_instance(
    bank: np.ndarray, n: int, L: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample n real logit fields → unary (n, L)."""
    idx = rng.integers(0, bank.shape[0], size=n)
    return bank[idx].copy()


def config_energy(
    states: np.ndarray,
    unary: np.ndarray,
    attract: list[tuple[int, int]],
    repel: list[tuple[int, int]],
    w_a: float,
    w_r: float,
) -> np.ndarray:
    """log p (up to const) for each config row in ``states`` (m, n) — the exact
    THRML energy, reused to score both brute-force states and Gibbs samples."""
    n = unary.shape[0]
    e = np.zeros(states.shape[0], dtype=np.float64)
    for i in range(n):
        e += unary[i, states[:, i]]
    for (i, j) in attract:
        e += w_a * (states[:, i] == states[:, j])
    for (i, j) in repel:
        e -= w_r * (states[:, i] == states[:, j])
    return e


# ─────────────────────────── exact ground truth ────────────────────────────


def exact_solve(
    unary: np.ndarray,
    attract: list[tuple[int, int]],
    repel: list[tuple[int, int]],
    L: int,
    w_a: float,
    w_r: float,
    budget: int,
) -> dict:
    """Brute-force Boltzmann marginals + MAP over all L^n states.

    Returns feasible=False (with wall time) when L^n exceeds the budget — the
    honest 'exact died here' signal for the Pareto plot.
    """
    n = unary.shape[0]
    n_states = L ** n
    if n_states > budget:
        return {"feasible": False, "n_states": n_states, "wall": float("nan")}

    t0 = time.time()
    grids = np.meshgrid(*[np.arange(L, dtype=np.int16) for _ in range(n)], indexing="ij")
    idx = np.stack([g.ravel() for g in grids], axis=1)  # (S, n)

    energy = config_energy(idx, unary, attract, repel, w_a, w_r)
    m = energy.max()
    p = np.exp(energy - m)
    p /= p.sum()
    marg = np.zeros((n, L), dtype=np.float64)
    for i in range(n):
        marg[i] = np.bincount(idx[:, i], weights=p, minlength=L)
    map_states = idx[int(np.argmax(energy))]
    return {
        "feasible": True,
        "n_states": n_states,
        "wall": time.time() - t0,
        "marginals": marg,
        "map_states": map_states.tolist(),
        "map_energy": float(m),
        "marg_entropy": float(
            np.mean([-np.sum(marg[i] * np.log(marg[i] + 1e-12)) for i in range(n)])
        ),
    }


# ───────────────────────────── block-Gibbs ─────────────────────────────────

import jax  # noqa: E402


def gibbs_solve(
    unary: np.ndarray,
    attract: list[tuple[int, int]],
    repel: list[tuple[int, int]],
    L: int,
    w_a: float,
    w_r: float,
    n_chains: int,
    burn_in: int,
    seed: int,
) -> dict:
    """THRML block-Gibbs marginals over the shared L-state alphabet.

    candidate_ids = arange(L) per hole, so decoded samples ARE state indices and
    equality/repulsion fire on matching states (a Potts model). Times a warm
    (post-compile) sample call so the cost is steady-state, not JIT overhead.
    """
    import jax.numpy as jnp  # noqa: PLC0415

    n = unary.shape[0]
    cand_ids = jnp.tile(jnp.arange(L, dtype=jnp.int32), (n, 1))
    sampler = flp.build(
        jnp.asarray(unary, dtype=jnp.float32),
        cand_ids,
        attract,
        repel,
        attract_weight=w_a,
        repel_weight=w_r,
        k=L,
    )
    n_blocks = len(sampler.free_blocks)

    # Warm up (compile), then time a second run for steady-state cost.
    _ = flp.sample(sampler, jax.random.PRNGKey(seed), n_chains=n_chains, burn_in=burn_in)
    t0 = time.time()
    samples = flp.sample(
        sampler, jax.random.PRNGKey(seed + 1), n_chains=n_chains, burn_in=burn_in
    )
    samples = np.asarray(samples)  # (n_chains, n) state indices
    wall = time.time() - t0

    marg = np.zeros((n, L), dtype=np.float64)
    for i in range(n):
        marg[i] = np.bincount(samples[:, i], minlength=L) / samples.shape[0]
    sample_e = config_energy(samples, unary, attract, repel, w_a, w_r)
    return {
        "marginals": marg,
        "wall": wall,
        "n_blocks": n_blocks,
        "best_energy": float(sample_e.max()),
        "mean_energy": float(sample_e.mean()),
    }


def marginal_tv(a: np.ndarray, b: np.ndarray) -> float:
    """Mean total-variation distance between per-hole marginals."""
    return float(np.mean(0.5 * np.sum(np.abs(a - b), axis=1)))


def map_hole_acc(gibbs_marg: np.ndarray, exact_map: list[int]) -> float:
    """Fraction of holes whose Gibbs-modal state matches the exact MAP state."""
    pred = np.argmax(gibbs_marg, axis=1)
    return float(np.mean(pred == np.asarray(exact_map)))


# ──────────────────────────── experiments ──────────────────────────────────


def exp_pareto(bank, args, rng) -> list[dict]:
    print("\n===== EXP A: Pareto crossover (spin-glass, growing n) =====")
    print(f"  L={args.L}  w={args.weight}  repel_frac={args.repel_fraction}  "
          f"chains={args.n_chains}  burn_in={args.burn_in}  exact_budget={args.exact_budget:.0e}")
    print(f"\n  {'n':>3} {'states':>12} {'exact_s':>9} {'gibbs_s':>9} "
          f"{'speedup':>8} {'TV':>7} {'mapAcc':>7} {'blocks':>7}")
    rows = []
    for n in args.n_list:
        attract, repel = make_edges(
            n, "spinglass", rng, repel_fraction=args.repel_fraction, p_edge=args.p_edge
        )
        unary = make_instance(bank, n, args.L, rng)
        ex = exact_solve(unary, attract, repel, args.L, args.weight, args.weight, args.exact_budget)
        gb = gibbs_solve(unary, attract, repel, args.L, args.weight, args.weight,
                         args.n_chains, args.burn_in, args.seed)
        row = {"n": n, "n_states": ex["n_states"], "gibbs_wall": gb["wall"],
               "n_blocks": gb["n_blocks"], "exact_feasible": ex["feasible"]}
        if ex["feasible"]:
            tv = marginal_tv(gb["marginals"], ex["marginals"])
            acc = map_hole_acc(gb["marginals"], ex["map_states"])
            speed = ex["wall"] / gb["wall"] if gb["wall"] > 0 else float("inf")
            row.update({"exact_wall": ex["wall"], "tv": tv, "map_acc": acc,
                        "marg_entropy": ex["marg_entropy"]})
            print(f"  {n:>3} {ex['n_states']:>12} {ex['wall']:>9.3f} {gb['wall']:>9.3f} "
                  f"{speed:>8.1f} {tv:>7.3f} {acc:>7.3f} {gb['n_blocks']:>7}")
        else:
            row.update({"exact_wall": None, "tv": None, "map_acc": None})
            print(f"  {n:>3} {ex['n_states']:>12} {'INFEAS':>9} {gb['wall']:>9.3f} "
                  f"{'--':>8} {'--':>7} {'--':>7} {gb['n_blocks']:>7}")
        rows.append(row)
    return rows


def exp_metastability(bank, args, rng) -> list[dict]:
    print("\n===== EXP B: Metastability (fixed n, hardening couplings) =====")
    n = args.meta_n
    print(f"  n={n}  L={args.L}  repel_frac={args.repel_fraction}  "
          f"chains={args.n_chains}  burn_in={args.burn_in}")
    # Fix the instance (edges + fields) so only the coupling weight changes.
    attract, repel = make_edges(
        n, "spinglass", rng, repel_fraction=args.repel_fraction, p_edge=args.p_edge
    )
    unary = make_instance(bank, n, args.L, rng)
    print(f"  graph: {len(attract)} attract + {len(repel)} repel edges")
    print(f"\n  {'weight':>7} {'TV':>7} {'Egap':>8} {'mapAcc':>7} {'exactEntropy':>13} {'gibbs_s':>8}")
    print("  (TV=marginal fidelity; Egap=exactMAP−bestGibbs energy, ~0 means the "
          "optimal mode IS found despite TV)")
    rows = []
    for w in args.weight_list:
        ex = exact_solve(unary, attract, repel, args.L, w, w, args.exact_budget)
        gb = gibbs_solve(unary, attract, repel, args.L, w, w,
                         args.n_chains, args.burn_in, args.seed)
        if not ex["feasible"]:
            print(f"  {w:>7.2f}  exact infeasible at n={n}, L={args.L}")
            continue
        tv = marginal_tv(gb["marginals"], ex["marginals"])
        acc = map_hole_acc(gb["marginals"], ex["map_states"])
        egap = ex["map_energy"] - gb["best_energy"]
        rows.append({"weight": w, "tv": tv, "map_acc": acc, "energy_gap": egap,
                     "map_energy": ex["map_energy"], "best_gibbs_energy": gb["best_energy"],
                     "exact_entropy": ex["marg_entropy"], "gibbs_wall": gb["wall"]})
        print(f"  {w:>7.2f} {tv:>7.3f} {egap:>8.3f} {acc:>7.3f} "
              f"{ex['marg_entropy']:>13.3f} {gb['wall']:>8.3f}")
    return rows


def exp_metastability_burnin(bank, args, rng) -> list[dict]:
    """At the hardest weight, does more burn-in close the TV gap (mixing) or not
    (true metastability)?"""
    print("\n===== EXP B2: does burn-in rescue mixing at the hard weight? =====")
    n = args.meta_n
    w = args.weight_list[-1]
    attract, repel = make_edges(
        n, "spinglass", rng, repel_fraction=args.repel_fraction, p_edge=args.p_edge
    )
    unary = make_instance(bank, n, args.L, rng)
    ex = exact_solve(unary, attract, repel, args.L, w, w, args.exact_budget)
    if not ex["feasible"]:
        print("  exact infeasible; skipping")
        return []
    print(f"  n={n} w={w} repel_frac={args.repel_fraction}")
    print(f"\n  {'burn_in':>8} {'TV':>7} {'gibbs_s':>8}")
    rows = []
    for b in args.burnin_list:
        gb = gibbs_solve(unary, attract, repel, args.L, w, w, args.n_chains, b, args.seed)
        tv = marginal_tv(gb["marginals"], ex["marginals"])
        rows.append({"burn_in": b, "tv": tv, "gibbs_wall": gb["wall"]})
        print(f"  {b:>8} {tv:>7.3f} {gb['wall']:>8.3f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="results/probe_splitability_cache.json")
    ap.add_argument("--out", default="results/probe_hardness_dial.json")
    ap.add_argument("--L", type=int, default=4, help="shared alphabet size (states/hole)")
    ap.add_argument("--n-list", default="4,6,8,10,12,14,16")
    ap.add_argument("--weight", type=float, default=2.0, help="coupling |w| for Exp A")
    ap.add_argument("--weight-list", default="0,0.5,1,2,4,8", help="coupling sweep for Exp B")
    ap.add_argument("--repel-fraction", type=float, default=0.5)
    ap.add_argument("--p-edge", type=float, default=1.0, help="Erdős–Rényi edge prob")
    ap.add_argument("--meta-n", type=int, default=8, help="fixed n for Exp B")
    ap.add_argument("--burnin-list", default="50,200,800,2000")
    ap.add_argument("--n-chains", type=int, default=512)
    ap.add_argument("--burn-in", type=int, default=500)
    ap.add_argument("--exact-budget", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.n_list = [int(x) for x in args.n_list.split(",")]
    args.weight_list = [float(x) for x in args.weight_list.split(",")]
    args.burnin_list = [int(x) for x in args.burnin_list.split(",")]

    bank = load_field_bank(Path(args.cache), args.L)
    print(f"[bank] {bank.shape[0]} real MDLM logit fields (top-{args.L})")
    rng = np.random.default_rng(args.seed)

    report = {
        "config": vars(args),
        "bank_size": int(bank.shape[0]),
        "pareto": exp_pareto(bank, args, rng),
        "metastability": exp_metastability(bank, args, rng),
        "burnin": exp_metastability_burnin(bank, args, rng),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
