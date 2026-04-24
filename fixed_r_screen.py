"""Phase 0 feasibility screen: can we place 15 semicircles inside a MEC of
target radius R? Searches 47D (scs[45] + cx + cy) with gauge pin (x0, y0, θ0).

Objective: continuous penalty (pair penetration + exact-containment deficit).
Solver: scipy L-BFGS-B with analytic gradient from numba gap primitives.

Exact-verifies any feasible candidate via geom.cnt + geom.mec.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.optimize as opt

import gap as gapmod
import geom


MARGIN = 5e-6  # scorer-safe margin — optimize to R_target - MARGIN to survive rnd


def contain_gap_exact(
    xi: float, yi: float, ti: float, cx: float, cy: float, R: float
) -> float:
    dx, dy = xi - cx, yi - cy
    d = math.hypot(dx, dy)
    if d < 1e-12:
        return R - 1.0
    ct, st = math.cos(ti), math.sin(ti)
    un = (dx * ct + dy * st) / d
    ut = (dx * st - dy * ct) / d
    far = d + 1.0 if un >= 0.0 else math.sqrt(d * d + 1.0 + 2.0 * d * abs(ut))
    return R - far


# 47D packing: x[0..44] = (x_i, y_i, θ_i) for i=0..14; x[45] = cx; x[46] = cy.
# Gauge: pin x[0]=0, y[0]=0, θ[0]=0 → effective 44 free vars.
FREE_INDICES = np.array([i for i in range(47) if i not in (0, 1, 2)], dtype=int)


def pack47_to_full(free: np.ndarray, gauge_vals=(0.0, 0.0, 0.0)) -> np.ndarray:
    full = np.zeros(47)
    full[0], full[1], full[2] = gauge_vals
    full[FREE_INDICES] = free
    return full


def unpack47(full: np.ndarray):
    scs = full[:45].reshape(15, 3).copy()
    cx = full[45]
    cy = full[46]
    return scs, cx, cy


def penalty(free: np.ndarray, R_target: float) -> float:
    full = pack47_to_full(free)
    scs, cx, cy = unpack47(full)
    # Pair penetration (continuous)
    pair = float(gapmod.overlap_penalty_sq(scs, margin=0.0))
    # Exact containment deficit: sum max(0, -gap_exact)^2
    cont = 0.0
    for i in range(15):
        g = contain_gap_exact(scs[i, 0], scs[i, 1], scs[i, 2], cx, cy, R_target)
        if g < 0:
            cont += g * g
    return pair + cont


def check_feasible(free: np.ndarray, R_target: float) -> tuple[bool, float, float]:
    """Return (feasible, cnt, mec). Feasible iff cnt==0 and mec <= R_target + 1e-7."""
    full = pack47_to_full(free)
    scs, cx, cy = unpack47(full)
    rounded = geom.rnd(scs)
    c = int(geom.cnt(rounded))
    r = float(geom.mec(rounded))
    return (c == 0 and r <= R_target + 1e-7), c, r


def solve_one(
    scs0: np.ndarray, cx0: float, cy0: float, R_target: float, max_iter: int = 1000
) -> dict:
    """Minimize penalty from initial config at fixed R_target. Returns result dict."""
    full0 = np.zeros(47)
    full0[:45] = scs0.reshape(-1)
    full0[45] = cx0
    full0[46] = cy0
    # Apply gauge: translate/rotate so piece 0 at origin with θ=0.
    # Simpler: just use raw coords; gauge pinning via FREE_INDICES means we DON'T
    # optimize (x0, y0, θ0), so they stay at their starting values. To make the
    # gauge proper, we translate the entire config so piece 0 is at origin.
    full0[:45:3] -= full0[0]  # x_i -= x_0 (but x_0 also subtracted from itself → 0)
    full0[1:45:3] -= full0[1]
    # Rotate so θ_0 = 0
    th0 = full0[2]
    c, s = math.cos(-th0), math.sin(-th0)
    for i in range(15):
        x, y = full0[3 * i], full0[3 * i + 1]
        full0[3 * i] = c * x - s * y
        full0[3 * i + 1] = s * x + c * y
        full0[3 * i + 2] = (full0[3 * i + 2] - th0) % (2 * math.pi)
    # Rotate cx, cy too
    cx_r = c * full0[45] - s * full0[46]
    cy_r = s * full0[45] + c * full0[46]
    full0[45], full0[46] = cx_r, cy_r
    # Extract free vars
    free0 = full0[FREE_INDICES]

    R_solve = R_target - MARGIN
    # R-homotopy: start from a loose radius to untangle damage, then shrink in
    # fine steps toward R_target. Steps sized relative to target so compression
    # from >=3.0 to 2.9486 works.
    R_loose = max(R_target + 0.05, 3.0)
    n_stages = 8
    R_schedule = np.linspace(R_loose, R_solve, n_stages)
    per_stage_iter = max(max_iter // n_stages, 50)
    result = opt.minimize(
        penalty,
        free0,
        args=(R_schedule[0],),
        method="L-BFGS-B",
        options={"maxiter": per_stage_iter, "ftol": 1e-14, "gtol": 1e-9},
    )
    free0 = result.x
    for R_stage in R_schedule[1:]:
        result = opt.minimize(
            penalty,
            free0,
            args=(R_stage,),
            method="L-BFGS-B",
            options={"maxiter": per_stage_iter, "ftol": 1e-14, "gtol": 1e-9},
        )
        free0 = result.x
    feasible, cnt, r_mec = check_feasible(result.x, R_target)
    full_end = pack47_to_full(result.x)
    scs_end = full_end[:45].reshape(15, 3)
    return {
        "penalty_final": float(result.fun),
        "nit": int(result.nit),
        "feasible": feasible,
        "cnt": cnt,
        "mec": r_mec,
        "scs": scs_end.tolist(),
        "cx": float(full_end[45]),
        "cy": float(full_end[46]),
    }


def perturb_incumbent(scs: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Damage k pieces by random angle shift. Amplitude scales inverse to k so
    L-BFGS-B can still recover feasibility at target R. k=1: up to π/4, k=3: up to π/12."""
    out = scs.copy()
    idx = rng.choice(15, size=k, replace=False)
    amp_max = math.pi / max(4.0, 4.0 * k)
    amp_min = amp_max * 0.2
    for i in idx:
        shift = rng.uniform(amp_min, amp_max) * rng.choice([-1, 1])
        out[i, 2] = (out[i, 2] + shift) % (2 * math.pi)
    return out


def generate_seeds(
    incumbent: np.ndarray, pool_dir: Path, count_per_cat: int, rng: np.random.Generator
) -> list[tuple[str, np.ndarray]]:
    """Return list of (category, scs) tuples."""
    seeds = []
    # Damaged incumbent (k = 1, 2, 3)
    for _ in range(count_per_cat):
        k = int(rng.integers(1, 4))
        seeds.append((f"damaged_k{k}", perturb_incumbent(incumbent, k, rng)))
    # Diverse pool + structured
    pool_files = sorted(pool_dir.glob("diverse_*.json")) + sorted(
        pool_dir.glob("seed_structured_*.json")
    )
    pool_configs = []
    for f in pool_files:
        try:
            d = json.load(open(f))
            pool_configs.append((f.stem, np.array(d["scs"], float)))
        except (json.JSONDecodeError, KeyError):
            pass
    for _ in range(count_per_cat):
        if not pool_configs:
            break
        name, cfg = pool_configs[int(rng.integers(len(pool_configs)))]
        # Light perturbation
        s = cfg.copy()
        for i in range(15):
            s[i, 2] = (s[i, 2] + rng.normal(0, 0.1)) % (2 * math.pi)
        seeds.append((f"pool_{name}", s))
    # Low-discrepancy Sobol (quasi-random) — simple fallback: uniform random
    for _ in range(count_per_cat):
        s = np.zeros((15, 3))
        s[:, 0] = rng.uniform(-2.8, 2.8, 15)
        s[:, 1] = rng.uniform(-2.8, 2.8, 15)
        s[:, 2] = rng.uniform(0, 2 * math.pi, 15)
        seeds.append(("sobol_uniform", s))
    return seeds


def mec_init(scs: np.ndarray) -> tuple[float, float]:
    """Compute initial (cx, cy) as MEC center."""
    cx, cy, _ = geom.mec_info(scs)
    return float(cx), float(cy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, required=True, help="target R")
    ap.add_argument("--n-seeds", type=int, default=16)
    ap.add_argument("--incumbent", default="pool/best.json")
    ap.add_argument("--pool-dir", default="pool")
    ap.add_argument("--out", default="runs/phase0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    incumbent = np.array(json.load(open(args.incumbent))["scs"], float)
    # Normalize seed count per category so total = n-seeds
    per_cat = max(args.n_seeds // 3, 1)
    seeds = generate_seeds(incumbent, Path(args.pool_dir), per_cat, rng)[: args.n_seeds]
    print(f"[screen] R={args.R} n_seeds={len(seeds)} out={out}")

    results = []
    t0 = time.time()
    for idx, (cat, scs0) in enumerate(seeds):
        cx0, cy0 = mec_init(scs0)
        res = solve_one(scs0, cx0, cy0, args.R, max_iter=args.max_iter)
        res["seed_idx"] = idx
        res["category"] = cat
        results.append(res)
        status = "FEAS" if res["feasible"] else "infeas"
        print(
            f"[screen] seed {idx:2d} {cat:25s} -> {status:6s} "
            f"pen={res['penalty_final']:.3e} cnt={res['cnt']} "
            f"mec={res['mec']:.6f} nit={res['nit']}"
        )

    elapsed = time.time() - t0
    feasible_count = sum(1 for r in results if r["feasible"])
    near_feas = sum(
        1 for r in results if not r["feasible"] and r["penalty_final"] < 1e-4
    )
    summary = {
        "R_target": args.R,
        "n_seeds": len(seeds),
        "feasible_count": feasible_count,
        "near_feasible_count": near_feas,
        "elapsed_s": elapsed,
        "results": results,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"\n[screen] DONE R={args.R} feasible={feasible_count}/{len(seeds)} "
        f"near={near_feas} elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
