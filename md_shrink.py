#!/usr/bin/env python3
"""Approach B: Shrinking-container molecular dynamics for semicircle packing.

Dynamics:
  - Start from random placement in disk of radius R0 = 4.0.
  - Repulsive force between overlapping pairs (detected by exact ov()).
    Force direction: -(c_i - c_j)/d, magnitude constant. This implicitly allows
    paired and flat-mating configurations since ov() returns False for those.
  - Container force on arc sample points outside radius R(t) around origin.
    Pulls the whole semicircle inward.
  - Small random torque on theta for rotational mixing.
  - R(t) shrinks linearly from R0 to R_target over N_steps, then holds.
  - After jamming, binary-search the tightest R with cnt(rnd(scs)) == 0.
  - Final polish with greedy coord-descent to improve MEC.

Validates every trajectory end-state with exact cnt() and mec().
"""

import argparse
import math
import time
import numpy as np
import numba as nb

import geom
import common

N_ARC = 12  # arc samples per semicircle for container forces
K_PAIR = 0.15
K_CONTAIN = 0.30
K_THETA = 0.03
MAX_STEP = 0.05
DT = 1.0


@nb.njit(cache=True)
def md_step(scs, R, k_pair, k_contain, k_theta, max_step, rng_theta_noise):
    """One Euler step. Mutates scs in place. Returns current overlap count."""
    N = geom.N
    fx = np.zeros(N)
    fy = np.zeros(N)
    ft = np.zeros(N)
    c = 0
    for i in range(N):
        for j in range(i + 1, N):
            if geom.ov(
                scs[i, 0], scs[i, 1], scs[i, 2], scs[j, 0], scs[j, 1], scs[j, 2]
            ):
                c += 1
                dx = scs[i, 0] - scs[j, 0]
                dy = scs[i, 1] - scs[j, 1]
                d = math.sqrt(dx * dx + dy * dy) + 1e-9
                ux = dx / d
                uy = dy / d
                fx[i] += k_pair * ux
                fy[i] += k_pair * uy
                fx[j] -= k_pair * ux
                fy[j] -= k_pair * uy
    for i in range(N):
        x = scs[i, 0]
        y = scs[i, 1]
        th = scs[i, 2]
        ct = math.cos(th)
        st = math.sin(th)
        fi_x = 0.0
        fi_y = 0.0
        samples = 0
        for s_idx in range(-1, N_ARC + 1):
            if s_idx == -1:
                sx = x + st
                sy = y - ct
            elif s_idx == N_ARC:
                sx = x - st
                sy = y + ct
            else:
                a = th - math.pi / 2 + math.pi * s_idx / (N_ARC - 1)
                sx = x + math.cos(a)
                sy = y + math.sin(a)
            dsq = sx * sx + sy * sy
            if dsq > R * R:
                d = math.sqrt(dsq)
                over = d - R
                fi_x -= k_contain * over * sx / d
                fi_y -= k_contain * over * sy / d
                samples += 1
        if samples > 0:
            fx[i] += fi_x / samples
            fy[i] += fi_y / samples
        ft[i] = rng_theta_noise[i] * k_theta
    for i in range(N):
        dx = fx[i]
        dy = fy[i]
        dmag = math.sqrt(dx * dx + dy * dy)
        if dmag > max_step:
            dx *= max_step / dmag
            dy *= max_step / dmag
        scs[i, 0] += dx
        scs[i, 1] += dy
        scs[i, 2] += ft[i]
    return c


@nb.njit(cache=True)
def greedy_polish(scs_init, iters, step0, seed):
    """Greedy coord-descent. Same kernel as contact_search; copied for numba cache."""
    np.random.seed(seed)
    scs = scs_init.copy()
    if geom.cnt(scs) > 0:
        return scs, 1e9
    score = geom.mec(scs)
    best = scs.copy()
    best_score = score
    step = step0
    stale = 0
    for it in range(iters):
        idx = np.random.randint(0, geom.N)
        ox = scs[idx, 0]
        oy = scs[idx, 1]
        ot = scs[idx, 2]
        scs[idx, 0] = ox + np.random.randn() * step
        scs[idx, 1] = oy + np.random.randn() * step
        scs[idx, 2] = ot + np.random.randn() * step * 0.3
        if geom.chk(scs, idx):
            scs[idx, 0] = ox
            scs[idx, 1] = oy
            scs[idx, 2] = ot
            stale += 1
        else:
            ns = geom.mec(scs)
            if ns < score:
                score = ns
                stale = 0
                if score < best_score:
                    best_score = score
                    best = scs.copy()
            else:
                scs[idx, 0] = ox
                scs[idx, 1] = oy
                scs[idx, 2] = ot
                stale += 1
        if stale > 5000:
            step *= 0.6
            stale = 0
        if step < 5e-6:
            step = 5e-6
    return best, best_score


def random_init(rng, R0):
    """Random semicircles uniformly in disk of radius R0 - 1."""
    scs = np.empty((geom.N, 3), dtype=np.float64)
    for i in range(geom.N):
        r = math.sqrt(rng.random()) * (R0 - 1.0)
        phi = rng.random() * 2 * math.pi
        scs[i, 0] = r * math.cos(phi)
        scs[i, 1] = r * math.sin(phi)
        scs[i, 2] = rng.random() * 2 * math.pi
    return scs


def jam_trial(rng, R0=4.0, R_end=3.5, n_shrink=2000, n_hold=1000, n_relax=3000):
    """One MD trajectory. Returns (final_scs, R_jam, feasible).

    Phases:
      n_shrink steps: linearly shrink container R0 -> R_end while repulsing
      n_hold steps: hold R_end, let overlaps dissipate
      n_relax steps: stronger pair repulsion, no container shrink - push out
                     remaining overlaps. If still overlapping at the end, trial
                     is infeasible.
    """
    scs = random_init(rng, R0)
    for step in range(n_shrink):
        R = R0 + (R_end - R0) * (step / n_shrink)
        noise = rng.normal(0, 1.0, geom.N).astype(np.float64)
        md_step(scs, R, K_PAIR, K_CONTAIN, K_THETA, MAX_STEP, noise)
    for step in range(n_hold):
        noise = rng.normal(0, 1.0, geom.N).astype(np.float64)
        md_step(scs, R_end, K_PAIR, K_CONTAIN, K_THETA, MAX_STEP, noise)
    for step in range(n_relax):
        noise = rng.normal(0, 0.5, geom.N).astype(np.float64)
        md_step(
            scs, R_end, K_PAIR * 2.0, K_CONTAIN * 0.5, K_THETA * 0.3, MAX_STEP, noise
        )

    rounded = geom.rnd(scs)
    if geom.cnt(rounded) > 0:
        return scs, float("inf"), False

    R_jam = float(geom.mec(rounded))
    return scs, R_jam, True


def run(hours, seed, out_name, polish_iters=200_000, r_end=3.5):
    rng = np.random.default_rng(seed)
    best_scs = None
    best_score = float("inf")

    t0 = time.time()
    n_trials = 0
    n_feasible = 0
    last_log = t0

    while not common.timeout_reached(t0, hours):
        n_trials += 1
        scs, r_jam, feasible = jam_trial(rng, R_end=r_end)
        if not feasible:
            continue
        n_feasible += 1
        polished, _ = greedy_polish(scs, polish_iters, 0.01, rng.integers(1 << 30))
        rounded = geom.rnd(polished)
        if geom.cnt(rounded) > 0:
            continue
        final = float(geom.mec(rounded))
        if final < best_score:
            best_score = final
            best_scs = polished.copy()
            common.save_named_best(out_name, final, rounded)
            common.log_line(
                "B",
                f"trial {n_trials} r_jam={r_jam:.4f} -> polished {final:.6f} NEW BEST",
            )
        now = time.time()
        if now - last_log > 30:
            feas_rate = n_feasible / max(1, n_trials)
            common.log_line(
                "B",
                f"trial {n_trials} feas={n_feasible} ({feas_rate:.2f}) best={best_score:.6f} elapsed={(now - t0) / 3600:.2f}h",
            )
            last_log = now

    if best_scs is not None:
        common.save_named_best(out_name, best_score, geom.rnd(best_scs))
    common.write_done_flag("md")
    common.log_line(
        "B",
        f"done trials={n_trials} feasible={n_feasible} best={best_score:.6f}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="md")
    ap.add_argument("--polish-iters", type=int, default=200_000)
    ap.add_argument(
        "--r-end",
        type=float,
        default=3.5,
        help="container shrink target; 3.5 gives ~97%% feasibility, tighter targets lose feasibility fast",
    )
    args = ap.parse_args()
    run(args.hours, args.seed, args.out, args.polish_iters, args.r_end)


if __name__ == "__main__":
    main()
