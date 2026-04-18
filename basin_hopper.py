#!/usr/bin/env python3
"""Basin-hopping optimizer for semicircle packing.

Key insight from compusophy's winning approach:
- Overlap resolution phase allows arbitrarily large kicks
- Workers at different temperatures explore different basins
- Greedy polishers continuously refine global best
- Simple Gaussian random walk, no fancy move types

Usage:
    python basin_hopper.py [hours]    # default: 40 hours
"""

import numpy as np
import numba as nb
import json
import time
import math
import os
import sys
from multiprocessing import Pool, cpu_count
from shapely.geometry import Polygon  # noqa: E402

N = 15
N_WORKERS = max(1, cpu_count() - 2)
BEST_FILE = "/tmp/semicircle_best.json"
SOLUTION_FILE = "/tmp/solution.json"
ITERS_PER_BATCH = 5000
BATCHES_PER_ROUND = 400  # 2M iters per round

# ============================================================
# Numba JIT core (overlap check, MEC, count)
# ============================================================


@nb.njit(cache=True)
def ov(x1, y1, t1, x2, y2, t2):
    EPS = 1e-9
    TH = 1e-6
    dSq = (x1 - x2) ** 2 + (y1 - y2) ** 2
    if dSq > 4 + EPS:
        return False
    d1x = math.cos(t1)
    d1y = math.sin(t1)
    d2x = math.cos(t2)
    d2y = math.sin(t2)
    if dSq < TH:
        if d1x * d2x + d1y * d2y > -1 + TH:
            return True
    f1ax = x1 + d1y
    f1ay = y1 - d1x
    f1bx = x1 - d1y
    f1by = y1 + d1x
    f2ax = x2 + d2y
    f2ay = y2 - d2x
    f2bx = x2 - d2y
    f2by = y2 + d2x
    cp1 = (f2ay - f1ay) * (f1bx - f1ax) - (f1by - f1ay) * (f2ax - f1ax)
    cp2 = (f2by - f1ay) * (f1bx - f1ax) - (f1by - f1ay) * (f2bx - f1ax)
    cp3 = (f1ay - f2ay) * (f2bx - f2ax) - (f2by - f2ay) * (f1ax - f2ax)
    cp4 = (f1by - f2ay) * (f2bx - f2ax) - (f2by - f2ay) * (f1bx - f2ax)
    if ((cp1 > TH and cp2 < -TH) or (cp1 < -TH and cp2 > TH)) and (
        (cp3 > TH and cp4 < -TH) or (cp3 < -TH and cp4 > TH)
    ):
        return True
    for ia in range(2):
        if ia == 0:
            ax, ay, bx, by = f1ax, f1ay, f1bx, f1by
            cx, cy, dx, dy, ox, oy = x2, y2, d2x, d2y, x1, y1
        else:
            ax, ay, bx, by = f2ax, f2ay, f2bx, f2by
            cx, cy, dx, dy, ox, oy = x1, y1, d1x, d1y, x2, y2
        vx = bx - ax
        vy = by - ay
        wx = ax - cx
        wy = ay - cy
        A = vx * vx + vy * vy
        B = 2 * (vx * wx + vy * wy)
        C = wx * wx + wy * wy - 1.0
        disc = B * B - 4 * A * C
        if disc >= -EPS:
            sq = math.sqrt(max(0.0, disc))
            for s in (1.0, -1.0):
                t = (-B + s * sq) / (2 * A)
                if -EPS <= t <= 1 + EPS:
                    px = ax + t * vx
                    py = ay + t * vy
                    if (px - ox) ** 2 + (py - oy) ** 2 < 1 - TH:
                        if (px - cx) * dx + (py - cy) * dy > TH:
                            return True
    d = math.sqrt(dSq)
    if d > EPS and d <= 2 + EPS:
        a = dSq / (2 * d)
        hSq = 1 - a * a
        if hSq >= 0:
            h = math.sqrt(max(0.0, hSq))
            mx = x1 + a * (x2 - x1) / d
            my = y1 + a * (y2 - y1) / d
            for s in (1.0, -1.0):
                px = mx + s * h * (y2 - y1) / d
                py = my - s * h * (x2 - x1) / d
                if (px - x1) * d1x + (py - y1) * d1y > TH and (px - x2) * d2x + (
                    py - y2
                ) * d2y > TH:
                    return True
    return False


@nb.njit(cache=True)
def cnt(scs):
    c = 0
    for i in range(N):
        for j in range(i + 1, N):
            if ov(scs[i, 0], scs[i, 1], scs[i, 2], scs[j, 0], scs[j, 1], scs[j, 2]):
                c += 1
    return c


@nb.njit(cache=True)
def cnt_one(scs, idx):
    c = 0
    for i in range(N):
        if i != idx and ov(
            scs[idx, 0], scs[idx, 1], scs[idx, 2], scs[i, 0], scs[i, 1], scs[i, 2]
        ):
            c += 1
    return c


@nb.njit(cache=True)
def welzl_mec(px, py, k):
    if k == 0:
        return 0.0, 0.0, 0.0
    if k == 1:
        return px[0], py[0], 0.0
    idx = np.arange(k)
    for i in range(k - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    cx = (px[idx[0]] + px[idx[1]]) / 2
    cy = (py[idx[0]] + py[idx[1]]) / 2
    r = math.sqrt((px[idx[0]] - px[idx[1]]) ** 2 + (py[idx[0]] - py[idx[1]]) ** 2) / 2
    for _p in range(3):
        for ii in range(k):
            i = idx[ii]
            if math.sqrt((px[i] - cx) ** 2 + (py[i] - cy) ** 2) > r + 1e-10:
                cx2 = px[i]
                cy2 = py[i]
                r2 = 0.0
                for jj in range(ii):
                    j = idx[jj]
                    if math.sqrt((px[j] - cx2) ** 2 + (py[j] - cy2) ** 2) > r2 + 1e-10:
                        cx3 = (px[i] + px[j]) / 2
                        cy3 = (py[i] + py[j]) / 2
                        r3 = math.sqrt((px[i] - px[j]) ** 2 + (py[i] - py[j]) ** 2) / 2
                        for kk in range(jj):
                            kv = idx[kk]
                            if (
                                math.sqrt((px[kv] - cx3) ** 2 + (py[kv] - cy3) ** 2)
                                > r3 + 1e-10
                            ):
                                bx_ = px[j] - px[i]
                                by_ = py[j] - py[i]
                                ccx = px[kv] - px[i]
                                ccy = py[kv] - py[i]
                                D = bx_ * ccy - by_ * ccx
                                if abs(D) < 1e-12:
                                    d_ij = (px[i] - px[j]) ** 2 + (py[i] - py[j]) ** 2
                                    d_ik = (px[i] - px[kv]) ** 2 + (py[i] - py[kv]) ** 2
                                    d_jk = (px[j] - px[kv]) ** 2 + (py[j] - py[kv]) ** 2
                                    if d_ij >= d_ik and d_ij >= d_jk:
                                        cx3 = (px[i] + px[j]) / 2
                                        cy3 = (py[i] + py[j]) / 2
                                        r3 = math.sqrt(d_ij) / 2
                                    elif d_ik >= d_jk:
                                        cx3 = (px[i] + px[kv]) / 2
                                        cy3 = (py[i] + py[kv]) / 2
                                        r3 = math.sqrt(d_ik) / 2
                                    else:
                                        cx3 = (px[j] + px[kv]) / 2
                                        cy3 = (py[j] + py[kv]) / 2
                                        r3 = math.sqrt(d_jk) / 2
                                else:
                                    B2 = bx_ * bx_ + by_ * by_
                                    C2 = ccx * ccx + ccy * ccy
                                    ux = (ccy * B2 - by_ * C2) / (2 * D)
                                    uy = (bx_ * C2 - ccx * B2) / (2 * D)
                                    cx3 = ux + px[i]
                                    cy3 = uy + py[i]
                                    r3 = math.sqrt(ux * ux + uy * uy)
                        cx2 = cx3
                        cy2 = cy3
                        r2 = r3
                cx = cx2
                cy = cy2
                r = r2
    return cx, cy, r


@nb.njit(cache=True)
def mec(scs):
    n_arc = 30
    k = 0
    M = N * (3 + n_arc + 1)
    px = np.empty(M)
    py = np.empty(M)
    for i in range(N):
        x, y, th = scs[i, 0], scs[i, 1], scs[i, 2]
        ct = math.cos(th)
        st = math.sin(th)
        px[k] = x
        py[k] = y
        k += 1
        px[k] = x + st
        py[k] = y - ct
        k += 1
        px[k] = x - st
        py[k] = y + ct
        k += 1
        for j in range(n_arc + 1):
            a = th - math.pi / 2 + math.pi * j / n_arc
            px[k] = x + math.cos(a)
            py[k] = y + math.sin(a)
            k += 1
    _, _2, r = welzl_mec(px, py, k)
    return r


@nb.njit(cache=True)
def rnd(scs):
    out = np.empty_like(scs)
    for i in range(N):
        out[i, 0] = round(scs[i, 0] * 1e6) / 1e6
        out[i, 1] = round(scs[i, 1] * 1e6) / 1e6
        out[i, 2] = round(scs[i, 2] * 1e6) / 1e6
    return out


@nb.njit(cache=True)
def find_boundary_mask(scs):
    n_arc = 30
    k = 0
    M = N * (3 + n_arc + 1)
    px = np.empty(M)
    py = np.empty(M)
    owner = np.empty(M, dtype=nb.int64)
    for i in range(N):
        x, y, th = scs[i, 0], scs[i, 1], scs[i, 2]
        ct = math.cos(th)
        st = math.sin(th)
        px[k] = x
        py[k] = y
        owner[k] = i
        k += 1
        px[k] = x + st
        py[k] = y - ct
        owner[k] = i
        k += 1
        px[k] = x - st
        py[k] = y + ct
        owner[k] = i
        k += 1
        for j in range(n_arc + 1):
            a = th - math.pi / 2 + math.pi * j / n_arc
            px[k] = x + math.cos(a)
            py[k] = y + math.sin(a)
            owner[k] = i
            k += 1
    cx, cy, r = welzl_mec(px, py, k)
    mask = np.zeros(N, dtype=nb.boolean)
    eps = 0.01
    for i in range(k):
        dist = math.sqrt((px[i] - cx) ** 2 + (py[i] - cy) ** 2)
        if dist > r - eps:
            mask[owner[i]] = True
    return mask, cx, cy, r


# ============================================================
# Basin-hopping worker
# ============================================================


@nb.njit(cache=True)
def basin_hop_worker(scs_init, iters, worker_id, n_workers, seed):
    """Single basin-hopping worker.

    Two phases:
    1. Overlap resolution: accept moves that reduce overlap count
    2. SA optimization: Metropolis-Hastings with adaptive step size

    When temperature drops below threshold, kick from global best
    and restart (basin hop).
    """
    np.random.seed(seed)

    is_polisher = worker_id >= (n_workers - n_workers // 4)  # top 25% are polishers

    scs = scs_init.copy()
    overlaps = cnt(scs)

    if overlaps == 0:
        score = mec(rnd(scs))
    else:
        score = 1e9

    best_scs = scs.copy()
    best_score = score

    # Worker-stratified temperature and kick strength
    worker_frac = worker_id / max(1.0, n_workers - 1.0)

    if is_polisher:
        temp = 1e-7
        step = 0.005
        step_min = 0.00001
        step_max = 0.01
    else:
        temp = 0.005 + worker_frac * 0.02  # 0.005 to 0.025
        step = 0.1
        step_min = 0.0001
        step_max = 0.5

    kick_strength = 0.01 + worker_frac * 0.20  # 0.01 to 0.21
    restarts = 0

    total_batches = iters // ITERS_PER_BATCH

    for batch in range(total_batches):
        batch_accepted = 0

        for _ in range(ITERS_PER_BATCH):
            idx = np.random.randint(0, N)

            # Gaussian random walk
            dx = np.random.randn() * step
            dy = np.random.randn() * step
            dtheta = np.random.randn() * step * math.pi

            old_x = scs[idx, 0]
            old_y = scs[idx, 1]
            old_t = scs[idx, 2]

            scs[idx, 0] += dx
            scs[idx, 1] += dy
            scs[idx, 2] += dtheta

            if overlaps > 0:
                # Phase 1: Overlap resolution
                new_overlaps = cnt(scs)
                if new_overlaps < overlaps or (
                    new_overlaps == overlaps and np.random.random() < 0.5
                ):
                    overlaps = new_overlaps
                    if overlaps == 0:
                        # Verify rounded solution is also overlap-free
                        if cnt(rnd(scs)) > 0:
                            overlaps = 1  # keep resolving
                        else:
                            score = mec(rnd(scs))
                            if score < best_score:
                                best_score = score
                                best_scs = scs.copy()
                else:
                    scs[idx, 0] = old_x
                    scs[idx, 1] = old_y
                    scs[idx, 2] = old_t
            else:
                # Phase 2: SA optimization - check overlap on ROUNDED coords
                # (official scorer rounds to 6 decimals before checking)
                rscs = rnd(scs)
                has_overlap = False
                for i in range(N):
                    if i != idx and ov(
                        rscs[idx, 0],
                        rscs[idx, 1],
                        rscs[idx, 2],
                        rscs[i, 0],
                        rscs[i, 1],
                        rscs[i, 2],
                    ):
                        has_overlap = True
                        break

                if has_overlap:
                    scs[idx, 0] = old_x
                    scs[idx, 1] = old_y
                    scs[idx, 2] = old_t
                else:
                    new_score = mec(rnd(scs))
                    delta = new_score - score

                    if delta < 0 or np.random.random() < math.exp(
                        -delta / max(temp, 1e-15)
                    ):
                        score = new_score
                        batch_accepted += 1
                        if score < best_score:
                            best_score = score
                            best_scs = scs.copy()
                    else:
                        scs[idx, 0] = old_x
                        scs[idx, 1] = old_y
                        scs[idx, 2] = old_t

        # Adaptive step size targeting 0.234 acceptance
        if overlaps == 0:
            rate = batch_accepted / ITERS_PER_BATCH
            if rate > 0.234:
                step *= 1.02
            else:
                step *= 0.98
            step = max(step_min, min(step_max, step))

            if not is_polisher:
                # Cool
                temp *= 0.95

                # Basin hop: when frozen, kick from best and reheat
                if temp < 0.00001:
                    restarts += 1
                    scs = best_scs.copy()

                    # Graduated kick - different strength per worker
                    for i in range(N):
                        scs[i, 0] += np.random.randn() * kick_strength
                        scs[i, 1] += np.random.randn() * kick_strength
                        scs[i, 2] += np.random.randn() * kick_strength * math.pi

                    overlaps = cnt(scs)
                    if overlaps == 0:
                        score = mec(rnd(scs))
                    else:
                        score = 1e9

                    # Reheat
                    temp = 0.005 + worker_frac * 0.02
                    step = 0.1
        else:
            # Polisher: stay cold, periodically restart from best
            step = max(step_min, min(step_max, step))
            if batch % 50 == 49:
                scs = best_scs.copy()
                score = best_score

    return best_score, best_scs, restarts


# ============================================================
# Pair-swap basin hopper: try all 105 position swaps
# ============================================================


@nb.njit(cache=True)
def swap_and_optimize(scs_init, i, j, sa_iters, seed):
    """Swap positions of semicircles i and j, then run SA to basin floor."""
    np.random.seed(seed)

    scs = scs_init.copy()
    # Swap positions but keep headings (creates new topology)
    scs[i, 0], scs[j, 0] = scs[j, 0], scs[i, 0]
    scs[i, 1], scs[j, 1] = scs[j, 1], scs[i, 1]

    # Resolve overlaps first
    overlaps = cnt(scs)
    for _ in range(sa_iters // 2):
        if overlaps == 0:
            break
        idx = np.random.randint(0, N)
        old_x = scs[idx, 0]
        old_y = scs[idx, 1]
        old_t = scs[idx, 2]
        scs[idx, 0] += np.random.randn() * 0.1
        scs[idx, 1] += np.random.randn() * 0.1
        scs[idx, 2] += np.random.randn() * 0.1 * math.pi
        new_overlaps = cnt(scs)
        if new_overlaps < overlaps or (
            new_overlaps == overlaps and np.random.random() < 0.5
        ):
            overlaps = new_overlaps
        else:
            scs[idx, 0] = old_x
            scs[idx, 1] = old_y
            scs[idx, 2] = old_t

    if overlaps > 0:
        return 1e9, scs

    # SA polish to basin floor
    score = mec(rnd(scs))
    best_score = score
    best_scs = scs.copy()
    temp = 0.005
    step = 0.05

    for _ in range(sa_iters // 2):
        idx = np.random.randint(0, N)
        old_x = scs[idx, 0]
        old_y = scs[idx, 1]
        old_t = scs[idx, 2]
        scs[idx, 0] += np.random.randn() * step
        scs[idx, 1] += np.random.randn() * step
        scs[idx, 2] += np.random.randn() * step * math.pi

        rscs = rnd(scs)
        has_overlap = False
        for k in range(N):
            if k != idx and ov(
                rscs[idx, 0],
                rscs[idx, 1],
                rscs[idx, 2],
                rscs[k, 0],
                rscs[k, 1],
                rscs[k, 2],
            ):
                has_overlap = True
                break

        if has_overlap:
            scs[idx, 0] = old_x
            scs[idx, 1] = old_y
            scs[idx, 2] = old_t
        else:
            new_score = mec(rnd(scs))
            delta = new_score - score
            if delta < 0 or np.random.random() < math.exp(-delta / max(temp, 1e-15)):
                score = new_score
                if score < best_score:
                    best_score = score
                    best_scs = scs.copy()
            else:
                scs[idx, 0] = old_x
                scs[idx, 1] = old_y
                scs[idx, 2] = old_t

        temp *= 0.9999
        # Adaptive step
        step *= 0.99999
        step = max(0.0001, min(0.1, step))

    return best_score, best_scs


# ============================================================
# Multiprocessing wrappers
# ============================================================


def run_basin_hop(args):
    worker_id, n_workers, scs_flat, iters, seed = args
    scs = np.array(scs_flat).reshape(N, 3)
    score, best_scs, restarts = basin_hop_worker(scs, iters, worker_id, n_workers, seed)
    return score, best_scs.tolist(), restarts, worker_id


def run_swap_search(args):
    i, j, scs_flat, sa_iters, seed = args
    scs = np.array(scs_flat).reshape(N, 3)
    score, best_scs = swap_and_optimize(scs, i, j, sa_iters, seed)
    return score, best_scs.tolist(), i, j


# ============================================================
# I/O
# ============================================================


def load_solution(path):
    """Load solution from JSON (array of {x,y,theta} dicts)."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "solution" in data:
        data = data["solution"]
    scs = np.zeros((N, 3))
    for i, d in enumerate(data[:N]):
        scs[i, 0] = d["x"]
        scs[i, 1] = d["y"]
        scs[i, 2] = d["theta"]
    return scs


def shapely_verify(scs):
    """Verify solution using Shapely polygon intersection (matches official scorer).
    Returns True if no overlaps with area > 1e-6."""
    ARC_PTS = 4096
    OVERLAP_TOL = 1e-6
    polys = []
    for i in range(N):
        x, y, th = round(scs[i, 0], 6), round(scs[i, 1], 6), round(scs[i, 2], 6)
        angles = np.linspace(th - math.pi / 2, th + math.pi / 2, ARC_PTS)
        coords = list(zip(x + np.cos(angles), y + np.sin(angles)))
        polys.append(Polygon(coords))
    for i in range(N):
        for j in range(i + 1, N):
            ci = polys[i].centroid
            cj = polys[j].centroid
            if math.hypot(ci.x - cj.x, ci.y - cj.y) > 2.0:
                continue
            if polys[i].intersection(polys[j]).area > OVERLAP_TOL:
                return False
    return True


def save_best(score, scs):
    # Save as solution array
    sol = [
        {
            "x": round(scs[i, 0], 6),
            "y": round(scs[i, 1], 6),
            "theta": round(scs[i, 2], 6),
        }
        for i in range(N)
    ]
    with open(SOLUTION_FILE, "w") as f:
        json.dump(sol, f, indent=2)
    # Save with score
    with open(BEST_FILE, "w") as f:
        json.dump({"score": score, "solution": sol}, f, indent=2)


# ============================================================
# Main loop
# ============================================================


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
    deadline = time.time() + hours * 3600

    # Load starting solution
    start_scs = None
    for path in [BEST_FILE, SOLUTION_FILE, "semicircle_best.json", "solution.json"]:
        if os.path.exists(path):
            try:
                start_scs = load_solution(path)
                print(f"Loaded from {path}")
                break
            except Exception:
                continue

    if start_scs is None:
        print("ERROR: No starting solution found")
        sys.exit(1)

    global_best_scs = start_scs.copy()
    global_best_score = mec(rnd(global_best_scs))

    if not shapely_verify(global_best_scs):
        print(
            f"WARNING: Starting solution has Shapely overlaps! Score {global_best_score:.6f} is invalid."
        )
        print("Will only accept Shapely-verified improvements.", flush=True)
        global_best_score = 1e9  # Force re-search

    print("=== Basin Hopper ===")
    print(
        f"Workers: {N_WORKERS}, Time: {hours:.1f}h, Iters/round: {ITERS_PER_BATCH * BATCHES_PER_ROUND // 1000}K"
    )

    # Compile numba
    print("Compiling...", flush=True)
    _dummy = np.random.randn(N, 3)
    cnt(_dummy)
    mec(_dummy)
    basin_hop_worker(_dummy, 1000, 0, 4, 42)
    swap_and_optimize(_dummy, 0, 1, 1000, 42)
    print("Ready", flush=True)
    print(f"\nStarting from: {global_best_score:.6f}\n", flush=True)

    save_best(global_best_score, global_best_scs)

    round_num = 0
    stale = 0
    swap_round_interval = 10  # Do swap search every N rounds

    pool = Pool(N_WORKERS)

    try:
        while time.time() < deadline:
            round_num += 1
            t0 = time.time()

            is_swap_round = round_num % swap_round_interval == 0

            if is_swap_round:
                # Swap search: try all 105 position swaps
                pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
                sa_per_swap = 500_000
                tasks = [
                    (
                        i,
                        j,
                        global_best_scs.flatten().tolist(),
                        sa_per_swap,
                        np.random.randint(0, 2**31),
                    )
                    for i, j in pairs
                ]
                results = pool.map(run_swap_search, tasks)

                swap_best = None
                for score, scs_list, si, sj in results:
                    if score < global_best_score:
                        if swap_best is None or score < swap_best[0]:
                            swap_best = (
                                score,
                                np.array(scs_list).reshape(N, 3),
                                si,
                                sj,
                            )

                dt = time.time() - t0
                left = (deadline - time.time()) / 3600

                if swap_best and shapely_verify(swap_best[1]):
                    global_best_score = swap_best[0]
                    global_best_scs = swap_best[1]
                    save_best(global_best_score, global_best_scs)
                    stale = 0
                    print(
                        f"  *** SWAP IMPROVEMENT: {global_best_score:.6f} (swap {swap_best[2]}<->{swap_best[3]}) ***",
                        flush=True,
                    )
                elif swap_best:
                    print(
                        f"  (swap {swap_best[2]}<->{swap_best[3]} score {swap_best[0]:.6f} REJECTED by Shapely)",
                        flush=True,
                    )
                else:
                    stale += 1

                print(
                    f"  R{round_num}: SWAP SEARCH best={global_best_score:.6f} stale={stale} {left:.1f}h left ({dt:.0f}s)",
                    flush=True,
                )
            else:
                # Normal basin-hopping round
                iters_per_worker = ITERS_PER_BATCH * BATCHES_PER_ROUND
                tasks = [
                    (
                        wid,
                        N_WORKERS,
                        global_best_scs.flatten().tolist(),
                        iters_per_worker,
                        np.random.randint(0, 2**31),
                    )
                    for wid in range(N_WORKERS)
                ]
                results = pool.map(run_basin_hop, tasks)

                scores = []
                restarts_total = 0
                for score, scs_list, restarts, wid in results:
                    scores.append(score)
                    restarts_total += restarts
                    if score < global_best_score:
                        candidate = np.array(scs_list).reshape(N, 3)
                        if shapely_verify(candidate):
                            global_best_score = score
                            global_best_scs = candidate
                            save_best(global_best_score, global_best_scs)
                            stale = 0
                            print(
                                f"  *** NEW BEST: {global_best_score:.6f} (worker {wid}, R{round_num}) ***",
                                flush=True,
                            )
                        else:
                            print(
                                f"  (worker {wid} score {score:.6f} REJECTED by Shapely)",
                                flush=True,
                            )

                if all(s >= global_best_score for s in scores):
                    stale += 1

                dt = time.time() - t0
                left = (deadline - time.time()) / 3600

                scores_sorted = sorted(scores)
                top3 = " ".join(f"{s:.4f}" for s in scores_sorted[:3])
                print(
                    f"  R{round_num}: best={global_best_score:.6f} top3=[{top3}] hops={restarts_total} stale={stale} {left:.1f}h left ({dt:.0f}s)",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        pool.terminate()
        pool.join()

    print(f"\nFinal: {global_best_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
