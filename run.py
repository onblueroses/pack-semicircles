#!/usr/bin/env python3
"""Proper Welzl MEC in numba + proxy SA + rounding fixup."""

import numpy as np
import numba as nb
import json
import time
import math

N = 15


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
def chk(scs, idx):
    for i in range(N):
        if i != idx and ov(
            scs[idx, 0], scs[idx, 1], scs[idx, 2], scs[i, 0], scs[i, 1], scs[i, 2]
        ):
            return True
    return False


@nb.njit(cache=True)
def welzl_mec(px, py, k):
    """Iterative Welzl MEC. Returns (cx, cy, r)."""
    if k == 0:
        return 0.0, 0.0, 0.0
    if k == 1:
        return px[0], py[0], 0.0

    # Shuffle
    idx = np.arange(k)
    for i in range(k - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        idx[i], idx[j] = idx[j], idx[i]

    # Start with first 2 points
    cx = (px[idx[0]] + px[idx[1]]) / 2
    cy = (py[idx[0]] + py[idx[1]]) / 2
    r = math.sqrt((px[idx[0]] - px[idx[1]]) ** 2 + (py[idx[0]] - py[idx[1]]) ** 2) / 2

    for _pass in range(3):
        for ii in range(k):
            i = idx[ii]
            if math.sqrt((px[i] - cx) ** 2 + (py[i] - cy) ** 2) > r + 1e-10:
                # New MEC must include point i on boundary
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
                                # Circle through 3 points
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
    """Proper Welzl MEC on boundary points."""
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
        out[i, 0] = np.round(scs[i, 0] * 1e6) / 1e6
        out[i, 1] = np.round(scs[i, 1] * 1e6) / 1e6
        out[i, 2] = np.round((scs[i, 2] % (2 * math.pi)) * 1e6) / 1e6
    return out


@nb.njit(cache=True)
def sa_direct(scs_init, iters, seed):
    """SA with proper Welzl MEC. Basin hopping."""
    np.random.seed(seed)
    scs = scs_init.copy()
    olaps = cnt(scs)
    score = mec(scs) if olaps == 0 else 1e9
    best = scs.copy()
    best_score = score
    temp = 0.05
    step = 0.07
    cool = 1 - 5e-6
    hops = 6
    chunk = iters // hops

    for hop in range(hops):
        if hop > 0:
            scs = best.copy()
            kick = 0.02 + hop * 0.015
            for i in range(N):
                scs[i, 0] += np.random.randn() * kick
                scs[i, 1] += np.random.randn() * kick
                scs[i, 2] += np.random.randn() * kick * 0.5
            olaps = cnt(scs)
            score = mec(scs) if olaps == 0 else 1e9
            temp = 0.03
            step = 0.05

        for it in range(chunk):
            temp *= cool
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
                if it % 200_000 == 199_999 and olaps > 0:
                    scs = best.copy()
                    olaps = cnt(scs)
                    score = mec(scs) if olaps == 0 else 1e9
                    step = 0.06
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
    return best, best_score


@nb.njit(cache=True)
def polish_q(scs_init, iters, seed):
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


def flower(nc, cr, rr):
    scs = np.zeros((N, 3))
    k = 0
    for i in range(nc):
        a = 2 * math.pi * i / max(nc, 1)
        cx, cy = cr * math.cos(a), cr * math.sin(a)
        scs[k] = [cx, cy, a]
        k += 1
        scs[k] = [cx, cy, a + math.pi]
        k += 1
    rem = N - k
    for i in range(rem):
        a = 2 * math.pi * i / rem
        scs[k] = [rr * math.cos(a), rr * math.sin(a), a]
        k += 1
        if k >= N:
            break
    return scs


def main():
    t0 = time.time()
    print("Compiling...", flush=True)
    d = np.random.randn(N, 3)
    cnt(d)
    mec(d)
    rnd(d)
    sa_direct(d, 100, 0)
    polish_q(d, 100, 0)
    print(f"Done {time.time() - t0:.1f}s")

    best_r = None
    best_r_score = 1e9

    # Load previous best if available
    try:
        with open("solution.json") as f:
            prev = json.load(f)
        ps = np.array([[s["x"], s["y"], s["theta"]] for s in prev])
        if cnt(ps) == 0:
            sc = mec(ps)
            best_r = ps.copy()
            best_r_score = sc
            print(f"Loaded previous: {sc:.6f}")
    except Exception:
        pass

    TIME_LIMIT = 1700  # ~28 min, leave 2 min for polish
    configs = [
        flower(3, 0.7, 2.2),
        flower(3, 0.5, 2.0),
        flower(4, 0.7, 2.0),
        flower(3, 0.9, 2.3),
        flower(2, 0.5, 2.0),
    ]

    ci = 0
    while time.time() - t0 < TIME_LIMIT:
        elapsed = time.time() - t0
        remaining = TIME_LIMIT - elapsed
        iters = min(8_000_000, int(remaining * 15_000))
        if iters < 1_000_000:
            break
        scs = configs[ci % len(configs)]
        # Also try perturbed versions of best
        if best_r is not None and ci >= len(configs):
            scs = best_r.copy()
            scs += np.random.randn(N, 3) * (0.05 + (ci % 5) * 0.03)
        print(
            f"SA {ci} ({iters // 1000}K, {remaining:.0f}s left)...", end=" ", flush=True
        )
        scs_opt, score = sa_direct(scs, iters, ci * 7919 + 42)
        r = rnd(scs_opt)
        c = cnt(r)
        if c == 0:
            rs = mec(r)
            print(f"{rs:.6f} VALID", end="")
            if rs < best_r_score:
                best_r_score = rs
                best_r = r.copy()
                # Save intermediate best
                j = json.dumps(
                    [
                        {
                            "x": float(r[i, 0]),
                            "y": float(r[i, 1]),
                            "theta": float(r[i, 2]),
                        }
                        for i in range(N)
                    ]
                )
                with open("solution.json", "w") as f:
                    f.write(j)
                print(" *** SAVED", end="")
        else:
            print(f"{score:.6f} ({c} ov)", end="")
        print()
        ci += 1

    # Long polish
    if best_r is not None:
        rem = 1790 - (time.time() - t0)
        fi = min(5_000_000, int(max(0, rem) * 4_000))
        if fi > 100_000:
            print(f"\nPolish ({fi // 1000}K)...", flush=True)
            best_r, best_r_score = polish_q(best_r, fi, 42)
            print(f"  {best_r_score:.6f}")

    if best_r is not None and cnt(best_r) == 0:
        print(f"\nFINAL: {mec(best_r):.6f} ({time.time() - t0:.1f}s)")
        j = json.dumps(
            [
                {
                    "x": float(best_r[i, 0]),
                    "y": float(best_r[i, 1]),
                    "theta": float(best_r[i, 2]),
                }
                for i in range(N)
            ],
            indent=2,
        )
        print(j)
        with open("solution.json", "w") as f:
            f.write(j)
        print("Saved")


if __name__ == "__main__":
    main()
