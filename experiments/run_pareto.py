"""M4 — Pareto sweep driver.

Loads MDLM once and enumerates a grid of (template, method, config) runs,
writing one JSON record per run to ``--out`` (default
``results/results.json``). Reuses the M3 modules verbatim; nothing here
adds new sampling logic.

Run:

    LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH" \\
    TRITON_LIBCUDA_PATH="/run/opengl-driver/lib" \\
        uv run python experiments/run_pareto.py [--quick] [--seed 0] \\
            [--out results/results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from diffusion_ebm.backbones.mdlm import MDLM  # noqa: E402
from diffusion_ebm.metrics.agreement import (  # noqa: E402
    agreement_rate,
    lm_perplexity,
)
from diffusion_ebm.sampler import baselines, thrml_joint  # noqa: E402
from diffusion_ebm.tasks.multihole import (  # noqa: E402
    MultiHoleInstance,
    all_templates,
)

N_CHAINS = 64


@dataclass
class RunResult:
    template: str
    method: str
    config: dict
    n_lm_forwards: int
    n_gibbs_sweeps: int
    n_chains: int
    agreement_rate: float
    lm_perplexity_mean: float
    wall_time_s: float
    seed: int


def _hole_groups(instance: MultiHoleInstance) -> list[list[int]]:
    pos_to_hole = {p: h for h, p in enumerate(instance.mask_positions)}
    return [[pos_to_hole[p] for p in g] for g in instance.equality_groups]


def _topk_jnp(
    mdlm: MDLM, instance: MultiHoleInstance, k: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    logits = mdlm.forward(instance.masked_input_ids)
    ids, unary = mdlm.top_k_candidates(logits, k=k, exclude_mask_token=True)
    ids_holes = ids[0, instance.mask_positions, :]
    unary_holes = unary[0, instance.mask_positions, :]
    ids_jnp = jnp.asarray(ids_holes.cpu().numpy(), dtype=jnp.int32)
    unary_jnp = jnp.asarray(unary_holes.float().cpu().numpy(), dtype=jnp.float32)
    return ids_jnp, unary_jnp


def _score(
    mdlm: MDLM, instance: MultiHoleInstance, samples: torch.Tensor
) -> tuple[float, float]:
    groups = _hole_groups(instance)
    agree = agreement_rate(samples, groups)
    ppl = float(lm_perplexity(mdlm, samples, instance).mean())
    return agree, ppl


# ---------------- method runners --------------------------------------------


def _run_thrml(
    mdlm: MDLM,
    instance: MultiHoleInstance,
    cfg: dict,
    seed: int,
) -> tuple[torch.Tensor, int, int]:
    k = int(cfg["k"])
    burn_in = int(cfg["gibbs_sweeps"])
    weight = float(cfg["equality_weight"])
    candidate_ids, unary = _topk_jnp(mdlm, instance, k=k)
    sampler = thrml_joint.build(
        unary=unary,
        candidate_ids=candidate_ids,
        equality_groups=_hole_groups(instance),
        equality_weight=weight,
        k=k,
    )
    out_jnp = thrml_joint.sample(
        sampler,
        jax.random.PRNGKey(seed),
        n_chains=N_CHAINS,
        burn_in=burn_in,
    )
    samples = torch.from_numpy(np.asarray(out_jnp)).long()
    return samples, 1, burn_in + N_CHAINS


def _run_mask_predict(
    mdlm: MDLM,
    instance: MultiHoleInstance,
    cfg: dict,
    seed: int,
) -> tuple[torch.Tensor, int, int]:
    n_iters = int(cfg["n_iters"])
    temperature = float(cfg["temperature"])
    samples = baselines.mask_predict(
        mdlm,
        instance,
        n_iters=n_iters,
        n_chains=N_CHAINS,
        temperature=temperature,
        seed=seed,
    )
    return samples, n_iters, 0


def _run_ancestral_topk(
    mdlm: MDLM,
    instance: MultiHoleInstance,
    cfg: dict,
    seed: int,
) -> tuple[torch.Tensor, int, int]:
    k = int(cfg["k"])
    samples = baselines.ancestral_topk(
        mdlm, instance, k=k, n_chains=N_CHAINS, seed=seed
    )
    return samples, 1, 0


def _run_independent_full(
    mdlm: MDLM,
    instance: MultiHoleInstance,
    cfg: dict,
    seed: int,
) -> tuple[torch.Tensor, int, int]:
    samples = baselines.independent_full(
        mdlm, instance, n_chains=N_CHAINS, seed=seed
    )
    return samples, 1, 0


Runner = Callable[
    [MDLM, MultiHoleInstance, dict, int], tuple[torch.Tensor, int, int]
]
RUNNERS: dict[str, Runner] = {
    "thrml_joint": _run_thrml,
    "mask_predict": _run_mask_predict,
    "ancestral_topk": _run_ancestral_topk,
    "independent_full": _run_independent_full,
}


# ---------------- grid -------------------------------------------------------


def _full_grid() -> dict[str, list[dict]]:
    thrml: list[dict] = []
    for sweeps in (10, 50, 200, 1000):
        for w in (2.0, 5.0, 10.0):
            for k in (32, 64):
                thrml.append(
                    {"gibbs_sweeps": sweeps, "equality_weight": w, "k": k}
                )

    mask: list[dict] = []
    for n_iters in (1, 4, 16, 64):
        for temp in (0.0, 1.0):
            mask.append({"n_iters": n_iters, "temperature": temp})

    ancestral = [{"k": 32}, {"k": 64}]
    indep: list[dict] = [{}]

    return {
        "thrml_joint": thrml,
        "mask_predict": mask,
        "ancestral_topk": ancestral,
        "independent_full": indep,
    }


def _quick_grid() -> dict[str, list[dict]]:
    return {
        "thrml_joint": [{"gibbs_sweeps": 200, "equality_weight": 5.0, "k": 64}],
        "mask_predict": [{"n_iters": 4, "temperature": 1.0}],
        "ancestral_topk": [{"k": 64}],
        "independent_full": [{}],
    }


# ---------------- main -------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 Pareto sweep driver")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=str, default=os.path.join(_REPO_ROOT, "results/results.json")
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    grid = _quick_grid() if args.quick else _full_grid()

    n_runs = sum(len(v) for v in grid.values()) * 3  # 3 templates
    print(f"[pareto] grid: {n_runs} runs, quick={args.quick}, seed={args.seed}")

    mdlm = MDLM.load()
    print(f"[pareto] MDLM on {mdlm.device}")

    records: list[dict[str, Any]] = []
    templates = all_templates(mdlm.tokenizer)
    t_total0 = time.time()

    for instance in templates:
        print(f"[pareto] template: {instance.text!r}")
        for method, configs in grid.items():
            runner = RUNNERS[method]
            for cfg in configs:
                t0 = time.time()
                samples, n_lm, n_gibbs = runner(mdlm, instance, cfg, args.seed)
                agree, ppl = _score(mdlm, instance, samples)
                wall = time.time() - t0
                rec = RunResult(
                    template=instance.text,
                    method=method,
                    config=cfg,
                    n_lm_forwards=n_lm,
                    n_gibbs_sweeps=n_gibbs,
                    n_chains=N_CHAINS,
                    agreement_rate=agree,
                    lm_perplexity_mean=ppl,
                    wall_time_s=wall,
                    seed=args.seed,
                )
                records.append(asdict(rec))
                print(
                    f"  {method:<18s} {cfg!s:<55s}  "
                    f"agree={agree:5.3f} ppl={ppl:6.2f} "
                    f"lm={n_lm:>3d} gibbs={n_gibbs:>5d} t={wall:5.2f}s"
                )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(
        f"[pareto] wrote {len(records)} records to {args.out} "
        f"in {time.time() - t_total0:.1f}s total"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
