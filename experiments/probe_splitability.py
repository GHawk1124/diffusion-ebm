"""Track B fail-fast probe: is the over-merge *splitable* under the proposed
frustrated energy, in posterior mass — before any sampler or hardware?

Track A proved that on oracle-grouped RMC the closed-form pooling MAP
(`hard_eq_map`) strictly dominates the THRML sampler: sampling earns nothing
when the partition is known. Track B removes the oracle and treats the entity
partition ``z`` as a latent, coupling it to the tokens ``x`` with a *frustrated*
energy

    log p(z, x) = Σ_i  l_i(x_i)                       # MDLM unary (logit)
                + Σ_{i<j} w_ij (2 a_ij − 1)            # Jaccard partition prior
                + λ      Σ_{i<j} (2 a_ij − 1)(2 s_ij − 1)   # frustration

with ``a_ij = 1[z_i == z_j]`` (same group), ``s_ij = 1[x_i == x_j]`` (same
token), and ``w_ij = β (J_ij − τ)`` from the top-k Jaccard overlap ``J_ij``.
The coupling rewards (same group, same token) and (diff group, diff token),
penalises the two frustrated corners. The hope: when Jaccard *over-merges* two
distinct entities (their candidate sets overlap, J ≥ τ) but the high-logit
tokens *disagree*, the λ term supplies the evidence to split the group.

THE KILLER RISK (per two independent design reviews) is metastability: a λ
strong enough to split a bad merge may also suppress the token disagreement
that triggers the split, so a *sampler* never sees the split. But metastability
is a sampler property. This probe sidesteps it by computing the **exact marginal
posterior over partitions** by enumeration (marginalising x analytically over
the per-hole top-L candidates). That answers the strictly prior question:

    Does the correct split even *have* dominant posterior mass?

If it does not, no sampler and no hardware can recover it ("hardware cannot fix
absent posterior mass") — Track B is dead and we learn it in CPU-seconds. If it
does, metastability becomes the next (separate) sampler-level question.

Decisive readout: over Jaccard *over-merged* multi_chain components (predicted
group spanning >1 gold chain), does a (λ, β) window put argmax posterior mass on
the correct split — WITHOUT collapsing *correctly-merged* components (predicted
group == one gold chain, n≥2) into spurious splits? We report, per grid cell,
the over-merge split-recovery rate and the correct-merge preservation rate, and
the cell maximising their minimum (the go/no-go score).

Two stages:
  * STAGE 1 (GPU): one MDLM forward per dev multi_chain item; cache per-hole
    top-k candidate ids + raw logits to JSON. Lazy-imports torch/MDLM.
  * STAGE 2 (CPU, numpy only): Jaccard grouping, component triage, exact
    partition-posterior enumeration over the (λ, β) grid, aggregate report.

Reads only the frozen dev split (``tasks.rmc.is_dev``); never mutates the
benchmark. Run from repo root:

    # outside nix develop, GPU stage needs the LD path dance:
    LD_LIBRARY_PATH="/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}" \\
        uv run python experiments/probe_splitability.py \\
            --max-items 120 --cache results/probe_splitability_cache.json \\
            --out results/probe_splitability.json

    # re-run the CPU analysis only (cache exists, no torch needed):
    python experiments/probe_splitability.py \\
        --cache results/probe_splitability_cache.json --out results/probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# Make `import diffusion_ebm...` work when run as a script from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion_ebm.tasks.rmc import is_dev, load_items  # noqa: E402

CORPORA = {
    "owt_heldout": "owt_heldout_multi.jsonl",
    "wikitext103": "wikitext103_multi.jsonl",
}


# ───────────────────────────── Stage 1: cache ──────────────────────────────


def build_cache(
    data_dir: Path,
    corpora: list[str],
    k: int,
    max_items: int | None,
    cache_path: Path,
) -> dict:
    """Run one MDLM forward per dev multi_chain item; cache top-k ids+logits.

    Lazy-imports torch/MDLM so the CPU analysis stage can run without them.
    """
    import torch  # noqa: PLC0415

    from diffusion_ebm.backbones.mdlm import MDLM  # noqa: PLC0415

    mdlm = MDLM.load()
    mask_id = mdlm.mask_token_id

    records: list[dict] = []
    for corpus in corpora:
        path = data_dir / CORPORA[corpus]
        items = [it for it in load_items(path) if is_dev(it)]
        if max_items is not None:
            items = items[:max_items]
        print(f"[cache] {corpus}: {len(items)} dev multi_chain items", flush=True)

        for n, item in enumerate(items):
            input_ids = item.masked_input_ids.unsqueeze(0).to(mdlm.device)
            logits, _ = mdlm.forward_hidden(input_ids)
            logits_1d = logits[0].float()
            logits_1d[:, mask_id] = -float("inf")
            ids_k, unary_k = mdlm.top_k_candidates(
                logits_1d.unsqueeze(0), k=k, exclude_mask_token=True
            )
            mp = item.mask_positions
            ids_holes = ids_k[0, mp, :].cpu().tolist()      # [n_holes, k]
            logit_holes = unary_k[0, mp, :].cpu().tolist()  # [n_holes, k]
            records.append(
                {
                    "item_id": item.item_id,
                    "corpus": item.corpus,
                    "chain_labels": list(item.chain_labels),
                    "gold_token_ids": list(item.gold_token_ids),
                    "cand_ids": ids_holes,
                    "cand_logits": logit_holes,
                }
            )
            if (n + 1) % 50 == 0:
                print(f"  ... {n + 1} done", flush=True)

    blob = {"k": k, "records": records}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(blob))
    print(f"[cache] wrote {len(records)} records -> {cache_path}", flush=True)
    return blob


# ─────────────────────── Stage 2 helpers: partitions ───────────────────────


def set_partitions(n: int):
    """Yield every set partition of range(n) as a canonical RGS label tuple."""
    if n == 0:
        yield ()
        return
    labels = [0] * n

    def rec(i: int, max_label: int):
        if i == n:
            yield tuple(labels)
            return
        for lbl in range(max_label + 1):
            labels[i] = lbl
            yield from rec(i + 1, max_label)
        labels[i] = max_label + 1
        yield from rec(i + 1, max_label + 1)

    yield from rec(1, 0)  # hole 0 is always label 0 (canonical RGS)


def canonical_rgs(labels) -> tuple[int, ...]:
    """Relabel an arbitrary label sequence to canonical restricted-growth form."""
    remap: dict[int, int] = {}
    out = []
    for x in labels:
        if x not in remap:
            remap[x] = len(remap)
        out.append(remap[x])
    return tuple(out)


def adjusted_rand(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Adjusted Rand index between two label tuples over the same items.

    A softer, standard partition-similarity metric than exact equality: 1.0 for
    identical clusterings, ~0 for chance. Degenerate denominator (all-same or
    all-distinct on both sides) → 1.0 iff the partitions are equal.
    """
    from math import comb

    n = len(a)
    table: dict[tuple[int, int], int] = {}
    for x, y in zip(a, b):
        table[(x, y)] = table.get((x, y), 0) + 1
    rows: dict[int, int] = {}
    cols: dict[int, int] = {}
    for (x, y), c in table.items():
        rows[x] = rows.get(x, 0) + c
        cols[y] = cols.get(y, 0) + c
    sum_comb = sum(comb(c, 2) for c in table.values())
    sa = sum(comb(c, 2) for c in rows.values())
    sb = sum(comb(c, 2) for c in cols.values())
    total = comb(n, 2)
    expected = (sa * sb / total) if total else 0.0
    max_index = (sa + sb) / 2.0
    denom = max_index - expected
    if denom == 0:
        return 1.0 if a == b else 0.0
    return (sum_comb - expected) / denom


def jaccard_groups(topk_sets: list[set[int]], threshold: float) -> list[list[int]]:
    """Union-find over top-k Jaccard overlap; mirrors the eval's de-oracle probe."""
    n = len(topk_sets)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(n):
        for b in range(a + 1, n):
            union_sz = len(topk_sets[a] | topk_sets[b])
            jac = len(topk_sets[a] & topk_sets[b]) / union_sz if union_sz else 0.0
            if jac >= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    comp: dict[int, list[int]] = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(i)
    return list(comp.values())


# ─────────────────── Stage 2 core: exact partition posterior ───────────────


def _enumerate_token_configs(
    cand_logits: list[list[float]], L: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """All top-L token-index configs for a component.

    Returns ``(base, eq, pairs)`` where ``base[c] = Σ_i logit_i(x_i)`` for config
    ``c`` (the per-hole full-vocab normaliser is constant across configs and
    partitions, so raw logits suffice), ``eq[p, c] = 1[token_i == token_j]`` for
    pair ``p=(i,j)``, and ``pairs`` lists the (i<j) hole-index pairs.

    Token *equality* is by candidate-rank identity only if ids match; we compare
    actual vocab ids so distinct holes that happen to share a candidate token
    register as equal.
    """
    n = len(cand_logits)
    Ls = [min(L, len(c)) for c in cand_logits]
    logit_arrs = [np.asarray(cand_logits[h][: Ls[h]], dtype=np.float64) for h in range(n)]
    grids = np.meshgrid(*[np.arange(l) for l in Ls], indexing="ij")
    idx = np.stack([g.ravel() for g in grids], axis=1)  # (C, n) candidate ranks
    C = idx.shape[0]

    base = np.zeros(C, dtype=np.float64)
    for h in range(n):
        base += logit_arrs[h][idx[:, h]]
    return base, idx, [(i, j) for i in range(n) for j in range(i + 1, n)]


def _pairwise_eq(
    idx: np.ndarray, cand_ids: list[list[int]], L: int, pairs: list[tuple[int, int]]
) -> np.ndarray:
    """``eq[p, c]`` boolean: do the two holes of pair p share a vocab id in config c."""
    n = len(cand_ids)
    Ls = [min(L, len(c)) for c in cand_ids]
    id_arrs = [np.asarray(cand_ids[h][: Ls[h]], dtype=np.int64) for h in range(n)]
    tok = np.empty(idx.shape, dtype=np.int64)  # (C, n) actual vocab ids
    for h in range(n):
        tok[:, h] = id_arrs[h][idx[:, h]]
    return np.stack([tok[:, i] == tok[:, j] for (i, j) in pairs], axis=0)


def _logsumexp(a: np.ndarray) -> float:
    m = np.max(a)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.sum(np.exp(a - m))))


def component_posterior(
    cand_ids: list[list[int]],
    cand_logits: list[list[float]],
    jac_matrix: np.ndarray,         # (n, n) pairwise Jaccard, symmetric
    lambdas: list[float],
    betas: list[float],
    tau: float,
    enum_budget: int,
    L_max: int,
) -> dict:
    """Exact p(z) over the (λ, β) grid for one component of holes.

    For each (λ, β): log p(z) ∝ [Σ_{i<j} w_ij (2a_ij−1)] + log Z(z; λ), with
    w_ij = β(J_ij − τ) and log Z(z; λ) = logsumexp_x[ base(x) +
    λ Σ_{i<j}(2a_ij−1)(2 s_ij(x)−1) ]. Returns per-(λ,β) the full normalised
    p(z) keyed by partition tuple.
    """
    n = len(cand_ids)
    # Adaptive L so the token enumeration stays within budget.
    L = min(L_max, max(2, int(math.floor(enum_budget ** (1.0 / n)))))

    base, idx, pairs = _enumerate_token_configs(cand_logits, L)
    eq = _pairwise_eq(idx, cand_ids, L, pairs).astype(np.float64)  # (P, C)
    s = 2.0 * eq - 1.0  # {−1, +1}
    w_pair = {(i, j): float(jac_matrix[i, j]) for (i, j) in pairs}

    partitions = list(set_partitions(n))
    # sign_z[(i,j)] = 2 a_ij − 1 per partition
    sign_z = {}
    for z in partitions:
        sign_z[z] = np.array(
            [1.0 if z[i] == z[j] else -1.0 for (i, j) in pairs], dtype=np.float64
        )

    # log Z(z; λ): does NOT depend on β. Cache per (λ, z).
    logZ = {}  # (lam, z) -> float
    for lam in lambdas:
        for z in partitions:
            coupling = sign_z[z] @ s  # (C,)
            logZ[(lam, z)] = _logsumexp(base + lam * coupling)

    out = {"n": n, "L_used": int(L), "n_configs": int(base.shape[0]), "grid": {}}
    for lam in lambdas:
        for beta in betas:
            # partition prior log-weight: Σ_{i<j} β(J−τ)(2a−1)
            logp = {}
            for z in partitions:
                prior = sum(
                    beta * (w_pair[(i, j)] - tau) * sign_z[z][p_idx]
                    for p_idx, (i, j) in enumerate(pairs)
                )
                logp[z] = prior + logZ[(lam, z)]
            Znorm = _logsumexp(np.array(list(logp.values())))
            probs = {z: math.exp(logp[z] - Znorm) for z in partitions}
            out["grid"][f"{lam}|{beta}"] = probs
    return out


# ──────────────────────────── Stage 2 driver ───────────────────────────────


def analyse(
    blob: dict,
    lambdas: list[float],
    betas: list[float],
    tau: float,
    group_threshold: float,
    enum_budget: int,
    L_max: int,
    max_component: int,
) -> dict:
    """Triage Jaccard components and aggregate split/preserve rates over the grid."""
    k = blob["k"]
    cells = [f"{lam}|{beta}" for lam in lambdas for beta in betas]
    # Per-cell tallies.
    over_total = 0
    merge_total = 0
    over_skipped_big = 0
    over_recovered = {c: 0 for c in cells}     # argmax == correct split
    over_pz_star = {c: 0.0 for c in cells}      # mass on correct split
    over_pz_merge = {c: 0.0 for c in cells}     # mass on all-merged (wrong)
    over_ari = {c: 0.0 for c in cells}          # ARI(argmax z, gold) on over-merges
    merge_preserved = {c: 0 for c in cells}     # argmax == all-merged (correct)
    merge_pz_star = {c: 0.0 for c in cells}
    merge_ari = {c: 0.0 for c in cells}         # ARI(argmax z, gold) on correct merges

    comp_details: list[dict] = []

    for rec in blob["records"]:
        cand_ids = rec["cand_ids"]
        cand_logits = rec["cand_logits"]
        chain = rec["chain_labels"]
        n_holes = len(cand_ids)
        topk_sets = [set(cand_ids[h]) for h in range(n_holes)]
        groups = jaccard_groups(topk_sets, group_threshold)

        # Pairwise Jaccard matrix (full top-k) for the prior.
        J = np.zeros((n_holes, n_holes), dtype=np.float64)
        for a in range(n_holes):
            for b in range(a + 1, n_holes):
                u = len(topk_sets[a] | topk_sets[b])
                jab = len(topk_sets[a] & topk_sets[b]) / u if u else 0.0
                J[a, b] = J[b, a] = jab

        for g in groups:
            if len(g) < 2:
                continue
            gold_in_g = [chain[h] for h in g]
            spans = len(set(gold_in_g))
            is_over = spans > 1
            is_merge = spans == 1
            if not (is_over or is_merge):
                continue
            if len(g) > max_component:
                if is_over:
                    over_skipped_big += 1
                continue

            sub_ids = [cand_ids[h] for h in g]
            sub_logits = [cand_logits[h] for h in g]
            sub_J = J[np.ix_(g, g)]
            z_star = canonical_rgs(gold_in_g)
            z_merge = tuple([0] * len(g))

            post = component_posterior(
                sub_ids, sub_logits, sub_J, lambdas, betas, tau, enum_budget, L_max
            )

            if is_over:
                over_total += 1
            else:
                merge_total += 1

            per_cell = {}
            for c in cells:
                probs = post["grid"][c]
                argmax_z = max(probs, key=probs.get)
                pz_star = probs.get(z_star, 0.0)
                pz_merge = probs.get(z_merge, 0.0)
                ari = adjusted_rand(argmax_z, z_star)
                per_cell[c] = (argmax_z == z_star, pz_star, pz_merge)
                if is_over:
                    over_recovered[c] += int(argmax_z == z_star)
                    over_pz_star[c] += pz_star
                    over_pz_merge[c] += pz_merge
                    over_ari[c] += ari
                else:
                    merge_preserved[c] += int(argmax_z == z_merge)
                    merge_pz_star[c] += pz_star
                    merge_ari[c] += ari

            comp_details.append(
                {
                    "item_id": rec["item_id"],
                    "corpus": rec["corpus"],
                    "kind": "over_merge" if is_over else "correct_merge",
                    "holes": g,
                    "gold_labels": gold_in_g,
                    "n": len(g),
                    "L_used": post["L_used"],
                    "n_configs": post["n_configs"],
                }
            )

    # Aggregate grid: rates + the go/no-go score per cell.
    grid_summary = {}
    best_cell, best_score = None, -1.0
    for c in cells:
        rec_rate = over_recovered[c] / over_total if over_total else 0.0
        pres_rate = merge_preserved[c] / merge_total if merge_total else 1.0
        score = min(rec_rate, pres_rate)
        grid_summary[c] = {
            "over_split_recovery": rec_rate,
            "over_mean_pz_star": (over_pz_star[c] / over_total) if over_total else 0.0,
            "over_mean_pz_merge": (over_pz_merge[c] / over_total) if over_total else 0.0,
            "over_mean_ari": (over_ari[c] / over_total) if over_total else 0.0,
            "merge_preservation": pres_rate,
            "merge_mean_pz_star": (merge_pz_star[c] / merge_total) if merge_total else 0.0,
            "merge_mean_ari": (merge_ari[c] / merge_total) if merge_total else 1.0,
            "go_score": score,
        }
        if score > best_score:
            best_score, best_cell = score, c

    return {
        "config": {
            "k": k,
            "lambdas": lambdas,
            "betas": betas,
            "tau": tau,
            "group_threshold": group_threshold,
            "enum_budget": enum_budget,
            "L_max": L_max,
            "max_component": max_component,
        },
        "counts": {
            "n_items": len(blob["records"]),
            "over_merged_components": over_total,
            "correct_merged_components": merge_total,
            "over_skipped_too_big": over_skipped_big,
        },
        "grid": grid_summary,
        "best_cell": best_cell,
        "best_go_score": best_score,
        "components": comp_details,
    }


def print_report(report: dict) -> None:
    c = report["counts"]
    print("\n================ SPLITABILITY PROBE ================")
    print(f"items={c['n_items']}  over-merged comps={c['over_merged_components']}"
          f"  correct-merged comps={c['correct_merged_components']}"
          f"  (over skipped too-big={c['over_skipped_too_big']})")
    cfg = report["config"]
    print(f"grid: λ∈{cfg['lambdas']}  β∈{cfg['betas']}  τ={cfg['tau']}")
    print("\n  λ|β     splitRec  ariOver  pzMerge_over  mergePres  ariMerge   GO")
    for cell, g in report["grid"].items():
        marker = " <<" if cell == report["best_cell"] else ""
        print(f"  {cell:>8}  {g['over_split_recovery']:.3f}    "
              f"{g['over_mean_ari']:.3f}    {g['over_mean_pz_merge']:.3f}"
              f"        {g['merge_preservation']:.3f}     "
              f"{g['merge_mean_ari']:.3f}   {g['go_score']:.3f}{marker}")
    print(f"\nBEST CELL {report['best_cell']}  go_score={report['best_go_score']:.3f}")
    verdict = (
        "GO — a (λ,β) window splits over-merges while preserving correct merges"
        if report["best_go_score"] >= 0.5
        else "WEAK/NO-GO — no cell jointly recovers splits and preserves merges"
    )
    print(f"VERDICT: {verdict}")
    print("===================================================\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/rmc")
    ap.add_argument("--corpus", choices=["both", *CORPORA], default="both")
    ap.add_argument("--k", type=int, default=256, help="top-k for Jaccard grouping")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--cache", default="results/probe_splitability_cache.json")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--out", default="results/probe_splitability.json")
    ap.add_argument("--lambdas", default="0,1,2,4,8")
    ap.add_argument("--betas", default="0,1,2,4,8")
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--group-threshold", type=float, default=0.3)
    ap.add_argument("--enum-budget", type=int, default=500_000)
    ap.add_argument("--L-max", type=int, default=32)
    ap.add_argument("--max-component", type=int, default=5)
    args = ap.parse_args()

    corpora = list(CORPORA) if args.corpus == "both" else [args.corpus]
    cache_path = Path(args.cache)
    lambdas = [float(x) for x in args.lambdas.split(",")]
    betas = [float(x) for x in args.betas.split(",")]

    if args.rebuild_cache or not cache_path.exists():
        blob = build_cache(
            Path(args.data_dir), corpora, args.k, args.max_items, cache_path
        )
    else:
        print(f"[cache] reusing {cache_path} (pass --rebuild-cache to refresh)")
        blob = json.loads(cache_path.read_text())

    report = analyse(
        blob,
        lambdas=lambdas,
        betas=betas,
        tau=args.tau,
        group_threshold=args.group_threshold,
        enum_budget=args.enum_budget,
        L_max=args.L_max,
        max_component=args.max_component,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print_report(report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
