#!/usr/bin/env python3
"""Approach D': CMA-ES basin hopping on the 45-DOF semicircle configuration.

Why CMA-ES, not L-BFGS+penalty:
- Exact geom.mec() and geom.ov() are non-smooth. L-BFGS-B with finite
  differences oscillates into tiny overlaps it cannot detect.
- CMA-ES is a mature blackbox evolutionary optimizer: population-based
  search with adaptive covariance, no gradients required. Scales well
  to 45-50 DOFs and handles discontinuous fitness landscapes natively.

Fitness function is a hard feasibility gate: configurations with
geom.cnt(rnd(scs)) > 0 get a large penalty proportional to overlap count,
so CMA-ES learns to stay feasible. Clean configs are scored by exact mec.

Outer basin hopping: repeatedly launch CMA-ES from current best +
Gaussian perturbation at several sigma scales. Keep any improvement.
"""

import argparse
import time

import cma
import numpy as np

import common
import geom

N = geom.N


def fitness(x, baseline_r):
    """x: flat (45,) array. Returns a score to minimize.
    Feasible configs score = mec(rnd(x)). Infeasible score = baseline + overlap_count.
    Rounded output matches the official scorer.
    """
    scs = x.reshape(N, 3)
    rounded = geom.rnd(scs)
    c = int(geom.cnt(rounded))
    if c > 0:
        return baseline_r + 1.0 + float(c)
    return float(geom.mec(rounded))


def run_cma(init_scs, baseline_r, sigma0, max_iter, popsize):
    x0 = init_scs.reshape(-1).astype(np.float64).copy()
    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "popsize": popsize,
            "maxiter": max_iter,
            "verbose": -9,
            "tolx": 1e-9,
            "tolfun": 1e-9,
        },
    )
    while not es.stop():
        sols = es.ask()
        fits = [fitness(s, baseline_r) for s in sols]
        es.tell(sols, fits)
    best_x = np.array(es.best.x)
    return best_x.reshape(N, 3), float(es.best.f)


def perturb(scs, rng, sigma_pos, sigma_theta):
    out = scs.copy()
    out[:, 0] += rng.normal(0.0, sigma_pos, size=N)
    out[:, 1] += rng.normal(0.0, sigma_pos, size=N)
    out[:, 2] += rng.normal(0.0, sigma_theta, size=N)
    return out


SIGMA_MENU = [
    # (outer_pos, outer_theta, cma_sigma0, cma_popsize, cma_maxiter)
    (0.00, 0.00, 0.005, 16, 300),  # tight re-polish of current best
    (0.02, 0.01, 0.010, 16, 400),
    (0.05, 0.03, 0.020, 20, 500),
    (0.10, 0.05, 0.040, 24, 600),
    (0.30, 0.15, 0.080, 32, 800),
    (0.80, 0.40, 0.200, 32, 1000),
]


def run(hours, seed, out_name):
    rng = np.random.default_rng(seed)
    score0, base = common.load_best()
    if score0 is None or base is None:
        common.log_line("D'", "no starting solution found; aborting")
        return
    common.log_line("D'", f"starting from {score0:.6f}")
    best_scs = base.copy()
    best_score = float(score0)

    t0 = time.time()
    n_trials = 0
    n_feasible = 0
    n_improved = 0
    last_log = t0

    while not common.timeout_reached(t0, hours):
        n_trials += 1
        recipe = SIGMA_MENU[rng.integers(0, len(SIGMA_MENU))]
        op, ot, s0, pop, mit = recipe
        init = perturb(best_scs, rng, op, ot)
        try:
            polished, fit = run_cma(
                init,
                baseline_r=best_score,
                sigma0=s0,
                max_iter=mit,
                popsize=pop,
            )
        except Exception as e:
            if n_trials % 50 == 0:
                common.log_line("D'", f"cma exception: {e}")
            continue
        rounded = geom.rnd(polished)
        c = int(geom.cnt(rounded))
        if c > 0:
            continue
        n_feasible += 1
        r = float(geom.mec(rounded))
        if r < best_score - 1e-8:
            n_improved += 1
            best_score = r
            best_scs = rounded.copy()
            common.save_named_best(out_name, r, rounded)
            common.log_line(
                "D'",
                f"trial {n_trials} recipe={recipe} -> {r:.6f} NEW BEST",
            )
        now = time.time()
        if now - last_log > 30:
            feas_rate = n_feasible / max(1, n_trials)
            common.log_line(
                "D'",
                f"trial {n_trials} feas={n_feasible} ({feas_rate:.2f}) impr={n_improved} best={best_score:.6f} elapsed={(now - t0) / 3600:.2f}h",
            )
            last_log = now

    common.save_named_best(out_name, best_score, geom.rnd(best_scs))
    common.write_done_flag("d_prime")
    common.log_line(
        "D'",
        f"done trials={n_trials} feasible={n_feasible} improved={n_improved} best={best_score:.6f}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="d_prime")
    args = ap.parse_args()
    run(args.hours, args.seed, args.out)


if __name__ == "__main__":
    main()
