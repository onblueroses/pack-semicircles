#!/usr/bin/env python3
"""Parallel Tempering MCMC optimizer for semicircle packing.

Replica exchange with basin hopping within each chain.
Uses all available CPU cores. Saves best solution atomically.

Usage:
    python pt_optimizer.py [hours]    # default: 120 hours
"""

import numpy as np
import numba as nb
import json
import time
import math
import os
import sys
from multiprocessing import Pool, cpu_count

N = 15
N_WORKERS = max(1, cpu_count() - 1)

# Temperature ladder: log-spaced, wide range
# Cold chains exploit known basins, hot chains explore new territory
TEMPS = np.geomspace(0.0003, 5.0, N_WORKERS)
STEPS = np.geomspace(0.002, 0.25, N_WORKERS)

BEST_FILE = "semicircle_best.json"
SOLUTION_FILE = "solution.json"
ITERS_PER_ROUND = 8_000_000


# ---- Core geometry (numba JIT) ----


@nb.njit(cache=True)
def ov(x1, y1, t1, x2, y2, t2):
    """Check if two semicircles overlap. Matches official scorer thresholds."""
    EPS = 1e-9
    TH = 1e-6
    dSq = (x1 - x2) ** 2 + (y1 - y2) ** 2
    if dSq > 4 + EPS:
        return False
    d1x = math.cos(t1)
    d1y = math.sin(t1)
    d2x = math.cos(t2)
    d2y = math.sin(t2)
    # Coincident centers: overlap unless facing exactly opposite
    if dSq < TH:
        if d1x * d2x + d1y * d2y > -1 + TH:
            return True
    # Flat edge endpoints
    f1ax = x1 + d1y
    f1ay = y1 - d1x
    f1bx = x1 - d1y
    f1by = y1 + d1x
    f2ax = x2 + d2y
    f2ay = y2 - d2x
    f2bx = x2 - d2y
    f2by = y2 + d2x
    # Flat-flat: strict cross-product intersection
    cp1 = (f2ay - f1ay) * (f1bx - f1ax) - (f1by - f1ay) * (f2ax - f1ax)
    cp2 = (f2by - f1ay) * (f1bx - f1ax) - (f1by - f1ay) * (f2bx - f1ax)
    cp3 = (f1ay - f2ay) * (f2bx - f2ax) - (f2by - f2ay) * (f1ax - f2ax)
    cp4 = (f1by - f2ay) * (f2bx - f2ax) - (f2by - f2ay) * (f1bx - f2ax)
    if ((cp1 > TH and cp2 < -TH) or (cp1 < -TH and cp2 > TH)) and (
        (cp3 > TH and cp4 < -TH) or (cp3 < -TH and cp4 > TH)
    ):
        return True
    # Flat-arc intersections
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
    # Arc-arc intersection
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
    """Count overlap pairs."""
    c = 0
    for i in range(N):
        for j in range(i + 1, N):
            if ov(scs[i, 0], scs[i, 1], scs[i, 2], scs[j, 0], scs[j, 1], scs[j, 2]):
                c += 1
    return c


@nb.njit(cache=True)
def chk(scs, idx):
    """Check if semicircle idx overlaps any other."""
    for i in range(N):
        if i != idx and ov(
            scs[idx, 0], scs[idx, 1], scs[idx, 2], scs[i, 0], scs[i, 1], scs[i, 2]
        ):
            return True
    return False


@nb.njit(cache=True)
def welzl_mec(px, py, k):
    """Iterative Welzl MEC with 3-pass robustness. Returns (cx, cy, r)."""
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
    """MEC radius via Welzl on boundary sample points (matches official scorer)."""
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
    """Round to 6 decimal places (matches official scorer)."""
    out = np.empty_like(scs)
    for i in range(N):
        out[i, 0] = np.round(scs[i, 0] * 1e6) / 1e6
        out[i, 1] = np.round(scs[i, 1] * 1e6) / 1e6
        out[i, 2] = np.round((scs[i, 2] % (2 * math.pi)) * 1e6) / 1e6
    return out


@nb.njit(cache=True)
def pt_chain(scs_init, iters, temp, step, seed):
    """Single MCMC chain at fixed temperature with basin hopping."""
    np.random.seed(seed)
    scs = scs_init.copy()
    olaps = cnt(scs)
    score = mec(scs) if olaps == 0 else 1e9
    best = scs.copy()
    best_score = score

    # Basin hopping: 4 sub-runs with kicks between
    sub_iters = iters // 4
    for sub in range(4):
        if sub > 0:
            scs = best.copy()
            kick = 0.01 + temp * 0.05
            for i in range(N):
                scs[i, 0] += np.random.randn() * kick
                scs[i, 1] += np.random.randn() * kick
                scs[i, 2] += np.random.randn() * kick * 0.5
            olaps = cnt(scs)
            score = mec(scs) if olaps == 0 else 1e9

        for it in range(sub_iters):
            idx = np.random.randint(0, N)
            ox = scs[idx, 0]
            oy = scs[idx, 1]
            ot = scs[idx, 2]
            scs[idx, 0] += np.random.randn() * step
            scs[idx, 1] += np.random.randn() * step
            scs[idx, 2] += np.random.randn() * step * 0.8
            if olaps > 0:
                nv = cnt(scs)
                if nv < olaps or (nv == olaps and np.random.random() < 0.3):
                    olaps = nv
                    if olaps == 0:
                        score = mec(scs)
                        if score < best_score:
                            best_score = score
                            best = scs.copy()
                else:
                    scs[idx, 0] = ox
                    scs[idx, 1] = oy
                    scs[idx, 2] = ot
                if it % 500_000 == 499_999 and olaps > 0:
                    scs = best.copy()
                    olaps = cnt(scs)
                    score = mec(scs) if olaps == 0 else 1e9
            else:
                if chk(scs, idx):
                    scs[idx, 0] = ox
                    scs[idx, 1] = oy
                    scs[idx, 2] = ot
                else:
                    ns = mec(scs)
                    delta = ns - score
                    if delta < 0 or np.random.random() < math.exp(
                        -delta / max(temp, 1e-15)
                    ):
                        score = ns
                        if score < best_score:
                            best_score = score
                            best = scs.copy()
                    else:
                        scs[idx, 0] = ox
                        scs[idx, 1] = oy
                        scs[idx, 2] = ot

    return best, best_score, scs, score


@nb.njit(cache=True)
def polish_q(scs_init, iters, seed):
    """Greedy descent in quantized (6-decimal) coordinate space."""
    np.random.seed(seed)
    scs = scs_init.copy()
    if cnt(scs) > 0:
        return scs, 1e9
    score = mec(scs)
    best = scs.copy()
    best_score = score
    step = 0.003
    stale = 0
    for it in range(iters):
        idx = np.random.randint(0, N)
        ox = scs[idx, 0]
        oy = scs[idx, 1]
        ot = scs[idx, 2]
        scs[idx, 0] = np.round((ox + np.random.randn() * step) * 1e6) / 1e6
        scs[idx, 1] = np.round((oy + np.random.randn() * step) * 1e6) / 1e6
        scs[idx, 2] = (
            np.round(((ot + np.random.randn() * step * 0.3) % (2 * math.pi)) * 1e6)
            / 1e6
        )
        if chk(scs, idx):
            scs[idx, 0] = ox
            scs[idx, 1] = oy
            scs[idx, 2] = ot
            stale += 1
        else:
            ns = mec(scs)
            if ns < score:
                score = ns
                stale = 0
            else:
                scs[idx, 0] = ox
                scs[idx, 1] = oy
                scs[idx, 2] = ot
                stale += 1
            if score < best_score:
                best_score = score
                best = scs.copy()
        if stale > 30_000:
            step *= 0.5
            stale = 0
        if step < 2e-6:
            step = 2e-6
    return best, best_score


# ---- Initialization ----


def flower(nc: int, cr: float, rr: float, off: float = 0.0) -> np.ndarray:
    """Generate a flower config: pairs of opposing semicircles on inner ring,
    singles on outer ring."""
    scs = np.zeros((N, 3))
    k = 0
    for i in range(nc):
        a = 2 * math.pi * i / max(nc, 1) + off
        cx, cy = cr * math.cos(a), cr * math.sin(a)
        scs[k] = [cx, cy, a]
        k += 1
        scs[k] = [cx, cy, a + math.pi]
        k += 1
    rem = N - k
    for i in range(rem):
        a = 2 * math.pi * i / rem + off * 0.5
        scs[k] = [rr * math.cos(a), rr * math.sin(a), a]
        k += 1
        if k >= N:
            break
    return scs


# ---- Parallel tempering orchestration ----


def run_chain(args):
    chain_id, scs_flat, temp, step, seed, iters = args
    scs = scs_flat.reshape(N, 3)
    best, best_score, final, final_score = pt_chain(scs, iters, temp, step, seed)
    return chain_id, best.flatten(), best_score, final.flatten(), final_score


def save_best(score, scs, best_file, solution_file):
    s = scs.reshape(N, 3) if scs.ndim == 1 else scs
    r = rnd(s)
    if cnt(r) > 0:
        return False
    rs = mec(r)
    data = {
        "score": rs,
        "scs": r.tolist(),
        "solution": [
            {"x": float(r[i, 0]), "y": float(r[i, 1]), "theta": float(r[i, 2])}
            for i in range(N)
        ],
    }
    # Atomic write
    tmp = best_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, best_file)
    # Also write submission format
    with open(solution_file, "w") as f:
        json.dump(data["solution"], f, indent=2)
    return True


def main():
    t0 = time.time()
    TIME_HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    TIME_LIMIT = TIME_HOURS * 3600

    print("=== Parallel Tempering Optimizer ===", flush=True)
    print(
        f"Chains: {N_WORKERS}, Time: {TIME_HOURS:.0f}h, Iters/round: {ITERS_PER_ROUND // 1_000_000}M",
        flush=True,
    )
    print(f"Temps: {' '.join(f'{t:.4f}' for t in TEMPS)}", flush=True)

    # Compile numba functions
    print("Compiling...", flush=True)
    d = np.random.randn(N, 3)
    cnt(d)
    mec(d)
    rnd(d)
    pt_chain(d, 100, 0.01, 0.01, 0)
    polish_q(d, 100, 0)
    print(f"Ready ({time.time() - t0:.1f}s)\n", flush=True)

    # Load previous best (try atomic best file first, then solution file)
    global_best_score = 1e9
    global_best = None
    for load_file in [BEST_FILE, SOLUTION_FILE]:
        try:
            with open(load_file) as f:
                data = json.load(f)
            if isinstance(data, dict) and "scs" in data:
                ps = np.array(data["scs"])
            else:
                ps = np.array([[s["x"], s["y"], s["theta"]] for s in data])
            if cnt(ps) == 0:
                sc = mec(ps)
                if sc < global_best_score:
                    global_best_score = sc
                    global_best = ps.copy()
        except Exception:
            continue
    if global_best is not None:
        save_best(global_best_score, global_best, BEST_FILE, SOLUTION_FILE)
        print(f"Starting from: {global_best_score:.6f}\n", flush=True)

    # Diverse initial configs for hot chains
    hot_configs = [
        flower(3, 0.7, 2.2),
        flower(3, 0.5, 2.0),
        flower(4, 0.7, 2.0),
        flower(2, 0.5, 2.0),
        flower(3, 0.9, 2.3),
        flower(4, 0.9, 2.3),
        flower(3, 0.6, 2.1, 0.3),
        flower(5, 0.9, 2.5),
        flower(3, 0.7, 1.8),
    ]

    # Initialize chains: cold from best, hot from diverse configs
    chains = []
    n_cold = min(5, N_WORKERS)
    for i in range(N_WORKERS):
        if i < n_cold and global_best is not None:
            scs = global_best.copy()
            scs += np.random.randn(N, 3) * (0.002 + i * 0.003)
        else:
            scs = hot_configs[(i - n_cold) % len(hot_configs)].copy()
            scs += np.random.randn(N, 3) * (0.05 + (i - n_cold) * 0.02)
        chains.append(scs.flatten())
    chain_scores = [1e9] * N_WORKERS

    round_num = 0
    swap_count = 0
    swap_accept = 0
    rounds_since_improve = 0
    pool = Pool(N_WORKERS)

    try:
        while time.time() - t0 < TIME_LIMIT - min(300, TIME_LIMIT * 0.05):
            round_num += 1
            rounds_since_improve += 1

            # Double iterations when stuck
            cur_iters = ITERS_PER_ROUND
            if rounds_since_improve > 20:
                cur_iters = min(ITERS_PER_ROUND * 2, 16_000_000)

            args = [
                (
                    i,
                    chains[i],
                    TEMPS[i],
                    STEPS[i],
                    round_num * 100000 + i * 10000 + int(time.time()) % 10000,
                    cur_iters,
                )
                for i in range(N_WORKERS)
            ]

            results = pool.map(run_chain, args)

            for cid, best_flat, best_score, final_flat, final_score in results:
                chains[cid] = final_flat
                chain_scores[cid] = final_score
                if best_score < global_best_score:
                    scs = best_flat.reshape(N, 3)
                    r = rnd(scs)
                    if cnt(r) == 0:
                        rs = mec(r)
                        if rs < global_best_score:
                            global_best_score = rs
                            global_best = r.copy()
                            save_best(rs, r, BEST_FILE, SOLUTION_FILE)
                            rounds_since_improve = 0
                            hours = (time.time() - t0) / 3600
                            print(
                                f"  *** NEW BEST: {rs:.6f} (chain={cid}, T={TEMPS[cid]:.4f}, R{round_num}, {hours:.1f}h) ***",
                                flush=True,
                            )

            # Replica exchange: random swap attempts between adjacent chains
            if N_WORKERS < 2:
                pass  # no swaps with a single chain
            for _ in range(N_WORKERS * 3 if N_WORKERS > 1 else 0):
                i = np.random.randint(0, N_WORKERS - 1)
                j = i + 1
                si, sj = chain_scores[i], chain_scores[j]
                if si < 1e8 and sj < 1e8:
                    beta_i, beta_j = 1.0 / TEMPS[i], 1.0 / TEMPS[j]
                    delta = (beta_i - beta_j) * (sj - si)
                    swap_count += 1
                    if delta > 0 or np.random.random() < math.exp(max(-500, delta)):
                        chains[i], chains[j] = chains[j], chains[i]
                        chain_scores[i], chain_scores[j] = (
                            chain_scores[j],
                            chain_scores[i],
                        )
                        swap_accept += 1

            # Reinject global best into coldest chain
            if round_num % 5 == 0 and global_best is not None:
                chains[0] = global_best.flatten()
                chain_scores[0] = global_best_score

            # Randomize hottest chains for fresh exploration
            if round_num % 50 == 0:
                for i in range(max(0, N_WORKERS - 3), N_WORKERS):
                    cfg = hot_configs[np.random.randint(len(hot_configs))].copy()
                    cfg += np.random.randn(N, 3) * 0.2
                    chains[i] = cfg.flatten()
                    chain_scores[i] = 1e9

            # Status
            elapsed = time.time() - t0
            remaining = TIME_LIMIT - elapsed
            sr = swap_accept / max(swap_count, 1)
            cold_scores = " ".join(
                f"{s:.4f}" if s < 1e8 else "---" for s in chain_scores[:n_cold]
            )
            hot_valid = [s for s in chain_scores[n_cold:] if s < 1e8]
            hot_min = min(hot_valid) if hot_valid else -1
            print(
                f"  R{round_num}: best={global_best_score:.6f} cold=[{cold_scores}] "
                f"hot_min={hot_min:.4f} swaps={sr:.0%} stale={rounds_since_improve} "
                f"{remaining / 3600:.1f}h left",
                flush=True,
            )

            # Periodic polish
            if round_num % 100 == 0 and global_best is not None:
                print("  Polishing...", flush=True)
                r = rnd(global_best)
                if cnt(r) == 0:
                    r2, rs2 = polish_q(r, 2_000_000, round_num)
                    if rs2 < global_best_score:
                        global_best_score = rs2
                        global_best = r2
                        save_best(rs2, r2, BEST_FILE, SOLUTION_FILE)
                        print(f"  Polish improved: {rs2:.6f}", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted!", flush=True)
    finally:
        pool.terminate()
        pool.join()

    # Final polish
    if global_best is not None:
        print("\nFinal polish (5M)...", flush=True)
        r = rnd(global_best)
        if cnt(r) == 0:
            r, rs = polish_q(r, 5_000_000, 99999)
            if rs < global_best_score:
                global_best_score = rs
                global_best = r
                save_best(rs, r, BEST_FILE, SOLUTION_FILE)
        print(f"  {global_best_score:.6f}", flush=True)

    print(
        f"\nFINAL: {global_best_score:.6f} ({(time.time() - t0) / 3600:.1f}h)",
        flush=True,
    )


if __name__ == "__main__":
    main()
