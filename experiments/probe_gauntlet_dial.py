"""Where does the classical solver escape close? — gauntlet across the dial.

The UAI-grids Probe D result is honest but NOT groundbreaking: a junction tree
solves those grids exactly (induced width 13–23), so G1b FAILS — there is a
cheap classical *solver escape* even though the approximate incumbent (loopy BP)
fails under frustration. "PT beats BP" is a textbook phenomenon when an exact
solver still exists.

The path to a defensible hardware claim is a PHASE DIAGRAM: a regime where the
WHOLE classical gauntlet fails — not just vanilla loopy BP, but tree-reweighted
BP (convex "BP done right"), naive mean-field, annealed importance sampling (the
classical tempering competitor), and best-of-N — AND exact inference (brute
force / junction tree) is infeasible, while parallel tempering uniquely recovers
the validated marginals. That is the "no solver escape" gate (G1b) passing for
real, on a sampling target (G0), at a quantifiable replica cost (the G6 hook).

This driver dials the frustrated Potts directly (same energy as the hardness-dial
and tempering scouts, real MDLM logit fields) and runs the FULL gauntlet on every
instance:

    exact (brute / VE)  loopy-BP  TRW-BP  mean-field  AIS  best-of-N      <- classical
    single-T block-Gibbs  parallel-tempering                              <- thermodynamic

Two sweeps locate the boundary where the escape closes:

  EXP A — coupling sweep at FIXED, exact-feasible n. As w grows the approximate
    classical methods (BP/TRW/MF/AIS/best-of-N) cross the Hellinger>0.05 failure
    line one by one; single-T Gibbs hits the metastability wall; PT stays at the
    floor. This finds the coupling w* where the *approximate* escape closes —
    while exact is STILL feasible (so PT is validated against truth here).

  EXP B — size sweep at FIXED strong w (in the metastable regime from EXP A). As
    n grows the exact state space L^n explodes and brute force / VE die; in the
    feasible regime PT is validated against exact, and past exact-death the
    PT-vs-Gibbs divergence is the standing hardness signal. This finds where the
    *exact* escape closes.

The escape is fully closed only where BOTH conditions hold (strong w AND large
n). That intersection — every classical entrant failing while exact is
infeasible and PT alone recovers — is the regime the hero figure must live in.

Pure numpy (reuses probe_hw_common + loopy_bp from probe_factor_graph_inference);
no JAX / GPU, so it runs on the system python without the CUDA LD dance:

    LD_LIBRARY_PATH="<zlib>:/run/opengl-driver/lib:..." \\
        .venv/bin/python experiments/probe_gauntlet_dial.py \\
            --cache results/probe_splitability_cache.json \\
            --out results/probe_gauntlet_dial.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_factor_graph_inference import loopy_bp  # noqa: E402
from probe_hw_common import (  # noqa: E402
    block_gibbs,
    exact_marginals,
    gauntlet_ais,
    gauntlet_best_of_n,
    gauntlet_mean_field,
    gauntlet_trw_bp,
    marginal_tv,
    mean_hellinger,
    parallel_tempering,
    potts_pairwise,
    variable_elimination_marginals,
)

FAIL = 0.05   # Hellinger above this = the method failed (solver-escape closed)
FLOOR = 0.03  # Hellinger below this = recovered to the Monte-Carlo floor


# ───────────────────────────── field bank ──────────────────────────────────


def load_field_bank(cache_path: Path, L: int, rng: np.random.Generator) -> np.ndarray:
    """Real per-hole top-L MDLM logits from the splitability cache → (M, L).

    Falls back to random Gaussian fields if the cache is absent, so the driver is
    self-contained; LM-grounded fields are preferred (they set a realistic
    unary/coupling scale) but the hardness phenomenon is a property of the
    couplings, not the fields.
    """
    if cache_path.exists():
        blob = json.loads(cache_path.read_text())
        rows = [logits[:L] for rec in blob["records"] for logits in rec["cand_logits"]
                if len(logits) >= L]
        if rows:
            bank = np.asarray(rows, dtype=np.float64)
            bank -= bank.max(axis=1, keepdims=True)
            return bank
    print(f"[bank] cache {cache_path} missing/empty — using random Gaussian fields")
    bank = rng.normal(0.0, 1.0, size=(4096, L))
    bank -= bank.max(axis=1, keepdims=True)
    return bank


# ─────────────────────────── instance builder ──────────────────────────────


def make_spinglass_edges(
    n: int, rng: np.random.Generator, repel_fraction: float, p_edge: float
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Erdős–Rényi(p_edge) frustrated graph: each present pair is repulsion with
    prob ``repel_fraction`` else attraction (random ± couplings → frustration)."""
    attract: list[tuple[int, int]] = []
    repel: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if p_edge < 1.0 and rng.random() > p_edge:
                continue
            (repel if rng.random() < repel_fraction else attract).append((i, j))
    return attract, repel


def build_potts(
    unary: np.ndarray,
    attract: list[tuple[int, int]],
    repel: list[tuple[int, int]],
    w: float,
    L: int,
):
    """Frustrated Potts as a probe_hw_common FactorGraph: edge table ±w·eye(L)."""
    eye = np.eye(L, dtype=np.float64)
    edges = list(attract) + list(repel)
    couplings = [w * eye for _ in attract] + [-w * eye for _ in repel]
    return potts_pairwise([L] * unary.shape[0], list(unary), edges, couplings)


# ─────────────────────────── exact ground truth ────────────────────────────


def gold_marginals(g, n: int, L: int, args) -> tuple[list[np.ndarray] | None, str]:
    """Exact gold via brute force → variable elimination → None (infeasible)."""
    if L ** n <= args.exact_budget:
        try:
            return exact_marginals(g, max_states=args.exact_budget, co_cluster=False).var_marginals, \
                f"brute({L**n})"
        except ValueError:
            pass
    try:
        marg, width = variable_elimination_marginals(g, max_width=args.max_width)
        return marg, f"VE(w={width})"
    except ValueError:
        return None, "INFEASIBLE"


# ───────────────────────────── the gauntlet ────────────────────────────────


def run_gauntlet(g, w: float, args, seed: int) -> dict:
    """Run every method on graph ``g`` and time each. Marginals only (no co-cl)."""
    t_max = max(2.0 * w, 4.0)
    out: dict[str, dict] = {}

    def timed(name, fn):
        t0 = time.time()
        marg = fn()
        out[name] = {"marg": [np.asarray(m) for m in marg], "wall": time.time() - t0}

    timed("loopy_bp", lambda: loopy_bp(g, iters=args.bp_iters, damping=0.5))
    timed("trw_bp", lambda: gauntlet_trw_bp(g, iters=args.bp_iters, damping=0.5).metric["marginals"])
    timed("mean_field", lambda: gauntlet_mean_field(g).metric["marginals"])
    ais = gauntlet_ais(g, n_chains=args.ais_chains, n_temps=args.ais_temps, seed=seed)
    out["ais"] = {"marg": [np.asarray(m) for m in ais.metric["marginals"]],
                  "wall": ais.wall_s, "ess": ais.metric["ess_frac"]}
    timed("best_of_n", lambda: gauntlet_best_of_n(g, n=args.bon_n, seed=seed).metric["marginals"])
    timed("gibbs", lambda: block_gibbs(
        g, T=1.0, n_chains=args.chains, burn_in=args.burn_in, n_measure=args.n_measure,
        co_cluster=False, seed=seed).var_marginals)
    timed("pt", lambda: parallel_tempering(
        g, n_levels=args.pt_levels, t_max=t_max, n_chains=args.chains, burn_in=args.burn_in,
        n_measure=args.n_measure, co_cluster=False, seed=seed).var_marginals)
    return out


def _score(out: dict, gold: list[np.ndarray] | None) -> dict:
    """Hellinger of each method vs gold (or vs PT when exact is infeasible)."""
    ref = gold if gold is not None else out["pt"]["marg"]
    he = {}
    for name, rec in out.items():
        if gold is None and name == "pt":
            he[name] = None  # PT is the reference; can't score against itself
        else:
            he[name] = mean_hellinger(rec["marg"], ref)
    he["_pt_vs_gibbs_tv"] = marginal_tv(out["pt"]["marg"], out["gibbs"]["marg"])
    return he


_METHODS = ["loopy_bp", "trw_bp", "mean_field", "ais", "best_of_n", "gibbs", "pt"]
_CLASSICAL = ["loopy_bp", "trw_bp", "mean_field", "ais", "best_of_n"]  # the escape set


def _avg_over_instances(bank, n, w, args, base_seed) -> dict:
    """Mean Hellinger per method over ``args.instances`` fixed-w instances."""
    accum: dict[str, list[float]] = {m: [] for m in _METHODS}
    pt_gibbs_tv, ais_ess, walls = [], [], {m: [] for m in _METHODS}
    feasible_any = False
    src = "INFEASIBLE"
    for r in range(args.instances):
        rng = np.random.default_rng(base_seed + 1000 * r)
        attract, repel = make_spinglass_edges(n, rng, args.repel_fraction, args.p_edge)
        idx = rng.integers(0, bank.shape[0], size=n)
        unary = args.field_scale * bank[idx].copy()
        g = build_potts(unary, attract, repel, w, args.L)
        gold, src = gold_marginals(g, n, args.L, args)
        feasible_any = feasible_any or (gold is not None)
        out = run_gauntlet(g, w, args, seed=base_seed + 1000 * r)
        he = _score(out, gold)
        for m in _METHODS:
            if he[m] is not None:
                accum[m].append(he[m])
            walls[m].append(out[m]["wall"])
        pt_gibbs_tv.append(he["_pt_vs_gibbs_tv"])
        ais_ess.append(out["ais"].get("ess", float("nan")))
    mean = {m: (float(np.mean(accum[m])) if accum[m] else None) for m in _METHODS}
    return {
        "n": n, "w": w, "gold_src": src, "exact_feasible": feasible_any,
        "hellinger": mean,
        "pt_vs_gibbs_tv": float(np.mean(pt_gibbs_tv)),
        "ais_ess": float(np.nanmean(ais_ess)),
        "wall_s": {m: float(np.mean(walls[m])) for m in _METHODS},
        "n_edges": len(attract) + len(repel),
    }


def _classical_best(row: dict) -> float | None:
    vals = [row["hellinger"][m] for m in _CLASSICAL if row["hellinger"][m] is not None]
    return min(vals) if vals else None


def _print_row(row: dict, *, ref_label: str) -> None:
    h = row["hellinger"]
    def f(m):
        v = h[m]
        return "  --  " if v is None else f"{v:6.3f}"
    cb = _classical_best(row)
    flag = ""
    if cb is not None:
        approx_closed = cb > FAIL
        pt_ok = (h["pt"] is None) or (h["pt"] < FLOOR)
        if approx_closed and pt_ok and h["gibbs"] is not None and h["gibbs"] > FAIL:
            flag = "  <== ESCAPE CLOSED (PT alone)"
        elif approx_closed:
            flag = "  approx-escape closed"
    print(f"  {row['w']:>5.1f} {row['n']:>3} {row['gold_src']:>11} "
          f"{f('loopy_bp')} {f('trw_bp')} {f('mean_field')} {f('ais')} {f('best_of_n')} "
          f"| {f('gibbs')} {f('pt')} | {row['pt_vs_gibbs_tv']:5.3f}{flag}")


def _header(sweep: str) -> None:
    print(f"\n  {'w':>5} {'n':>3} {'gold':>11} "
          f"{'loopyBP':>6} {'TRW':>6} {'MF':>6} {'AIS':>6} {'bestN':>6} "
          f"| {'GibbsT1':>6} {'PT':>6} | {'PT~Gb':>5}")
    print(f"  {'-'*5} {'-'*3} {'-'*11} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} "
          f"| {'-'*6} {'-'*6} | {'-'*5}   ({sweep}; Hellinger to {'exact' }; "
          f"PT~Gb=TV(PT,Gibbs))")


# ──────────────────────────── experiments ──────────────────────────────────


def exp_weight_sweep(bank, args) -> list[dict]:
    print("\n===== EXP A: coupling sweep at fixed exact-feasible n "
          f"(n={args.meta_n}, L={args.L}, instances={args.instances}) =====")
    print("  finds w* where the APPROXIMATE classical escape (BP/TRW/MF/AIS/bestN) "
          "closes while exact is still feasible (PT validated vs truth)")
    _header("EXP A: coupling sweep")
    rows = []
    for w in args.weight_list:
        row = _avg_over_instances(bank, args.meta_n, w, args, base_seed=args.seed)
        _print_row(row, ref_label="exact")
        rows.append(row)
    return rows


def exp_size_sweep(bank, args) -> list[dict]:
    print(f"\n===== EXP B: size sweep at fixed strong w={args.hard_weight} "
          f"(L={args.L}, instances={args.instances}) =====")
    print("  finds where the EXACT escape closes (brute/VE die ~L^n); past that, "
          "PT-vs-Gibbs TV is the standing hardness signal (PT col '--' = is the ref)")
    _header("EXP B: size sweep")
    rows = []
    for n in args.n_list:
        row = _avg_over_instances(bank, n, args.hard_weight, args, base_seed=args.seed)
        _print_row(row, ref_label="exact/PT-ref")
        rows.append(row)
    return rows


def _verdict(weight_rows: list[dict], size_rows: list[dict]) -> str:
    # w* where the approximate escape closes (all classical > FAIL) with exact feasible.
    w_star = None
    for r in weight_rows:
        cb = _classical_best(r)
        if r["exact_feasible"] and cb is not None and cb > FAIL \
                and r["hellinger"]["pt"] is not None and r["hellinger"]["pt"] < FLOOR:
            w_star = r["w"]
            break
    # n where exact dies.
    n_exact_dead = next((r["n"] for r in size_rows if not r["exact_feasible"]), None)
    # fully-closed: strong w AND exact infeasible AND classical all fail AND PT/Gibbs split.
    closed = [r for r in size_rows if not r["exact_feasible"]
              and (_classical_best(r) or 0) > FAIL and r["pt_vs_gibbs_tv"] > 0.1]
    if w_star is not None and closed:
        return (f"PHASE BOUNDARY FOUND. Approximate-classical escape closes at "
                f"w*≈{w_star} (exact still feasible there → PT validated vs truth). "
                f"Exact escape closes at n≈{n_exact_dead} (L^n past budget). At strong "
                f"w AND n≥{closed[0]['n']} every classical entrant fails, exact is "
                f"infeasible, and PT/Gibbs diverge (metastability) — G1b passes for "
                f"real in this regime. This is where the hero figure lives.")
    if w_star is not None:
        return (f"APPROXIMATE escape closes at w*≈{w_star}, but no fully-closed "
                f"(exact-infeasible + all-classical-fail) cell in the size sweep — "
                f"push n_list higher or w harder.")
    return ("No w where ALL approximate classical methods fail while PT recovers — "
            "the escape stays open at these settings; raise the coupling / repel "
            "fraction or shrink the alphabet L.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="results/probe_splitability_cache.json")
    ap.add_argument("--out", default="results/probe_gauntlet_dial.json")
    ap.add_argument("--L", type=int, default=4, help="shared alphabet size")
    ap.add_argument("--meta-n", type=int, default=8, help="fixed n for EXP A (exact-feasible)")
    ap.add_argument("--weight-list", default="0,1,2,4,8,12,16", help="coupling sweep, EXP A")
    ap.add_argument("--n-list", default="6,8,10,12,14,16", help="size sweep, EXP B")
    ap.add_argument("--hard-weight", type=float, default=8.0, help="fixed strong w for EXP B")
    ap.add_argument("--repel-fraction", type=float, default=0.5)
    ap.add_argument("--field-scale", type=float, default=1.0,
                    help="scale on the unary MDLM fields; →0 restores label symmetry "
                         "(competing degenerate modes) where AIS importance weights "
                         "collapse but PT replica-exchange visits all modes")
    ap.add_argument("--p-edge", type=float, default=1.0, help="Erdős–Rényi edge prob")
    ap.add_argument("--instances", type=int, default=3, help="instances averaged per cell")
    ap.add_argument("--chains", type=int, default=256)
    ap.add_argument("--burn-in", type=int, default=500)
    ap.add_argument("--n-measure", type=int, default=200)
    ap.add_argument("--pt-levels", type=int, default=16)
    ap.add_argument("--ais-temps", type=int, default=400)
    ap.add_argument("--ais-chains", type=int, default=512)
    ap.add_argument("--bon-n", type=int, default=4096)
    ap.add_argument("--bp-iters", type=int, default=600)
    ap.add_argument("--exact-budget", type=int, default=20_000_000)
    ap.add_argument("--max-width", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-size", action="store_true", help="run only EXP A")
    args = ap.parse_args()
    args.weight_list = [float(x) for x in args.weight_list.split(",")]
    args.n_list = [int(x) for x in args.n_list.split(",")]

    rng = np.random.default_rng(args.seed)
    bank = load_field_bank(Path(args.cache), args.L, rng)
    print(f"[bank] {bank.shape[0]} logit fields (top-{args.L}); "
          f"FAIL>{FAIL} FLOOR<{FLOOR}")

    weight_rows = exp_weight_sweep(bank, args)
    size_rows = [] if args.skip_size else exp_size_sweep(bank, args)
    verdict = _verdict(weight_rows, size_rows)
    print(f"\n  VERDICT: {verdict}")

    report = {
        "config": vars(args),
        "bank_size": int(bank.shape[0]),
        "fail_threshold": FAIL,
        "floor_threshold": FLOOR,
        "weight_sweep": weight_rows,
        "size_sweep": size_rows,
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=lambda o: (
        o.tolist() if isinstance(o, np.ndarray) else float(o)
        if isinstance(o, (np.floating, np.integer)) else str(o))))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
