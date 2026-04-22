"""Numba-jit geometry primitives for semicircle packing.

Copied verbatim from hetero_optimizer.py - bit-for-bit compatible with verify.mjs.
Do not modify kernels without re-running the verify.mjs cross-check.
"""

import math
import numpy as np
import numba as nb

N = 15

__all__ = ["N", "ov", "cnt", "chk", "welzl_mec", "mec", "find_boundary_mask", "rnd"]


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
def find_boundary_mask(scs):
    """Find semicircles with sample points on/near the MEC boundary.
    Returns (boolean mask, cx, cy, r)."""
    n_arc = 30
    k = 0
    M = N * (3 + n_arc + 1)
    px = np.empty(M)
    py = np.empty(M)
    owner = np.empty(M, dtype=np.int64)
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
    mask = np.zeros(N, dtype=np.bool_)
    eps = 0.01
    for i in range(k):
        dist = math.sqrt((px[i] - cx) ** 2 + (py[i] - cy) ** 2)
        if dist > r - eps:
            mask[owner[i]] = True
    return mask, cx, cy, r


@nb.njit(cache=True)
def rnd(scs):
    """Round to 6 decimal places (matches official scorer)."""
    out = np.empty_like(scs)
    for i in range(N):
        out[i, 0] = np.round(scs[i, 0] * 1e6) / 1e6
        out[i, 1] = np.round(scs[i, 1] * 1e6) / 1e6
        out[i, 2] = np.round((scs[i, 2] % (2 * math.pi)) * 1e6) / 1e6
    return out
