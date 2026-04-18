#!/usr/bin/env python3
"""Heterogeneous multi-day optimizer for semicircle packing.

Runs 14 specialized workers with distinct move operators:
  A: Cold refiners (single perturbation, low temp)
  B: Hot explorers (large steps, diverse inits)
  C: Cluster movers (translate/rotate groups, pair swaps)
  D: Elastic relaxation (soft overlap penalty, harden over time)
  E: Geometric constructors (structured seeds + quick polish)
  F: Monitor/polisher (quantized polish, checkpointing)
  I: Internal replica-exchange search with semicircle-specific moves
  Q: Quantized adaptive SA with discrete permutations
  R: Random restart worker with structured kicks

Communication via shared filesystem pool/ directory.
Phase-adaptive scheduling over 3 phases (explore -> exploit -> polish).

Usage:
    python hetero_optimizer.py [hours]    # default: 120 hours
"""

import numpy as np
import numba as nb
import json
import time
import math
import os
import sys
import signal
from multiprocessing import Pool, cpu_count

N = 15
N_WORKERS = max(1, cpu_count() - 2)

BEST_FILE = "semicircle_best.json"
SOLUTION_FILE = "solution.json"
POOL_DIR = "pool"
ITERS_PER_ROUND = 8_000_000

# Phase boundaries (hours)
PHASE_1_END = 24.0
PHASE_2_END = 96.0
CATASTROPHE_STALL_HOURS = 18.0
CATASTROPHE_STALL_ROUNDS = 5  # Chaos pulse every 5 stale rounds

# Diverse pool (archive)
DIVERSE_SLOTS = 20
DIVERSE_MIN_L2 = 0.08


# ============================================================
# Numba JIT functions (copied verbatim from pt_optimizer.py)
# ============================================================


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


def flower(nc: int, cr: float, rr: float, off: float = 0.0) -> np.ndarray:
    """Generate a flower config: pairs on inner ring, singles on outer ring."""
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


# ============================================================
# Cluster move operators (Phase 2)
# ============================================================


@nb.njit(cache=True)
def select_cluster(scs, seed_idx, k):
    """Select seed_idx + k-1 nearest neighbors. Returns boolean mask of length N."""
    mask = np.zeros(N, dtype=nb.boolean)
    mask[seed_idx] = True
    if k <= 1:
        return mask
    # Compute distances from seed to all others
    dists = np.empty(N)
    sx, sy = scs[seed_idx, 0], scs[seed_idx, 1]
    for i in range(N):
        if i == seed_idx:
            dists[i] = 1e9
        else:
            dists[i] = (scs[i, 0] - sx) ** 2 + (scs[i, 1] - sy) ** 2
    # Pick k-1 nearest
    for _ in range(k - 1):
        best_i = -1
        best_d = 1e9
        for i in range(N):
            if not mask[i] and dists[i] < best_d:
                best_d = dists[i]
                best_i = i
        if best_i >= 0:
            mask[best_i] = True
    return mask


@nb.njit(cache=True)
def move_cluster_translate(scs, mask, dx, dy):
    """Translate all selected semicircles by (dx, dy). Returns modified copy."""
    out = scs.copy()
    for i in range(N):
        if mask[i]:
            out[i, 0] += dx
            out[i, 1] += dy
    return out


@nb.njit(cache=True)
def move_cluster_rotate(scs, mask, angle):
    """Rotate selected semicircles around cluster centroid by angle. Returns modified copy."""
    out = scs.copy()
    # Compute cluster centroid
    cx, cy = 0.0, 0.0
    count = 0
    for i in range(N):
        if mask[i]:
            cx += scs[i, 0]
            cy += scs[i, 1]
            count += 1
    if count == 0:
        return out
    cx /= count
    cy /= count
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for i in range(N):
        if mask[i]:
            rx = scs[i, 0] - cx
            ry = scs[i, 1] - cy
            out[i, 0] = cx + rx * cos_a - ry * sin_a
            out[i, 1] = cy + rx * sin_a + ry * cos_a
            out[i, 2] = scs[i, 2] + angle
    return out


@nb.njit(cache=True)
def move_pair_swap(scs, i, j, swap_angle):
    """Swap positions of semicircles i and j. Optionally swap angles too."""
    out = scs.copy()
    out[i, 0], out[j, 0] = scs[j, 0], scs[i, 0]
    out[i, 1], out[j, 1] = scs[j, 1], scs[i, 1]
    if swap_angle:
        out[i, 2], out[j, 2] = scs[j, 2], scs[i, 2]
    return out


@nb.njit(cache=True)
def nearest_neighbor(scs, idx):
    """Return (nearest_index, center_distance) for semicircle idx."""
    best_j = -1
    best_d2 = 1e18
    x, y = scs[idx, 0], scs[idx, 1]
    for j in range(N):
        if j == idx:
            continue
        d2 = (scs[j, 0] - x) ** 2 + (scs[j, 1] - y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_j = j
    return best_j, math.sqrt(best_d2)


@nb.njit(cache=True)
def move_slide_on_flat(scs, mover, support, dist, normal_shift, theta_shift):
    """Slide mover along support's flat edge with a small normal correction."""
    out = scs.copy()
    th = scs[support, 2]
    tx = math.sin(th)
    ty = -math.cos(th)
    nx = math.cos(th)
    ny = math.sin(th)
    side = (scs[mover, 0] - scs[support, 0]) * nx + (scs[mover, 1] - scs[support, 1]) * ny
    if side < 0:
        nx = -nx
        ny = -ny
    out[mover, 0] += tx * dist + nx * normal_shift
    out[mover, 1] += ty * dist + ny * normal_shift
    out[mover, 2] = scs[mover, 2] + theta_shift
    return out


@nb.njit(cache=True)
def move_roll_around_neighbor(scs, pivot, mover, angle, radial_shift):
    """Roll mover around pivot by rotating its center and heading together."""
    out = scs.copy()
    vx = scs[mover, 0] - scs[pivot, 0]
    vy = scs[mover, 1] - scs[pivot, 1]
    ca = math.cos(angle)
    sa = math.sin(angle)
    rvx = vx * ca - vy * sa
    rvy = vx * sa + vy * ca
    radius = math.sqrt(rvx * rvx + rvy * rvy)
    if radius > 1e-12 and radial_shift != 0.0:
        target = max(0.05, radius + radial_shift)
        scale = target / radius
        rvx *= scale
        rvy *= scale
    out[mover, 0] = scs[pivot, 0] + rvx
    out[mover, 1] = scs[pivot, 1] + rvy
    out[mover, 2] = scs[mover, 2] + angle
    return out


@nb.njit(cache=True)
def move_pair_heading_swap(scs, i, j, push):
    """Swap the local flat-edge orientations of a pair and separate slightly."""
    out = scs.copy()
    vx = scs[i, 0] - scs[j, 0]
    vy = scs[i, 1] - scs[j, 1]
    norm = math.sqrt(vx * vx + vy * vy)
    if norm < 1e-12:
        vx = math.cos(scs[i, 2])
        vy = math.sin(scs[i, 2])
        norm = math.sqrt(vx * vx + vy * vy)
    vx /= norm
    vy /= norm
    thi = scs[i, 2]
    thj = scs[j, 2]
    out[i, 2] = thj + math.pi
    out[j, 2] = thi + math.pi
    out[i, 0] += vx * push
    out[i, 1] += vy * push
    out[j, 0] -= vx * push
    out[j, 1] -= vy * push
    return out


@nb.njit(cache=True)
def move_boundary_orbit(scs, idx, cx, cy, orbit_angle, inward_shift):
    """Move idx around the current MEC center while preserving relative heading."""
    out = scs.copy()
    vx = scs[idx, 0] - cx
    vy = scs[idx, 1] - cy
    ca = math.cos(orbit_angle)
    sa = math.sin(orbit_angle)
    rvx = vx * ca - vy * sa
    rvy = vx * sa + vy * ca
    radius = math.sqrt(rvx * rvx + rvy * rvy)
    if radius > 1e-12 and inward_shift != 0.0:
        target = max(0.05, radius - inward_shift)
        scale = target / radius
        rvx *= scale
        rvy *= scale
    out[idx, 0] = cx + rvx
    out[idx, 1] = cy + rvy
    out[idx, 2] = scs[idx, 2] + orbit_angle
    return out


@nb.njit(cache=True)
def structured_kick(scs_init, scale, seed):
    """Topology-aware restart kick built from boundary and pair-contact moves."""
    np.random.seed(seed)
    scs = scs_init.copy()
    moves = 3 + np.random.randint(0, 5)
    mask, cx, cy, _ = find_boundary_mask(scs)
    for _ in range(moves):
        r = np.random.random()
        if r < 0.25:
            idx = np.random.randint(0, N)
            j, _2 = nearest_neighbor(scs, idx)
            scs = move_slide_on_flat(
                scs,
                idx,
                j,
                np.random.randn() * scale,
                np.random.randn() * scale * 0.25,
                np.random.randn() * scale * 0.2,
            )
        elif r < 0.50:
            idx = np.random.randint(0, N)
            j, _2 = nearest_neighbor(scs, idx)
            scs = move_roll_around_neighbor(
                scs,
                j,
                idx,
                np.random.randn() * scale * 0.4,
                np.random.randn() * scale * 0.2,
            )
        elif r < 0.75:
            if np.random.random() < 0.7:
                n_boundary = 0
                for i in range(N):
                    if mask[i]:
                        n_boundary += 1
                if n_boundary > 0:
                    target = np.random.randint(0, n_boundary)
                    count = 0
                    idx = 0
                    for i in range(N):
                        if mask[i]:
                            if count == target:
                                idx = i
                                break
                            count += 1
                else:
                    idx = np.random.randint(0, N)
            else:
                idx = np.random.randint(0, N)
            scs = move_boundary_orbit(
                scs,
                idx,
                cx,
                cy,
                np.random.randn() * scale * 0.3,
                np.random.randn() * scale * 0.4,
            )
        else:
            i = np.random.randint(0, N)
            j, _2 = nearest_neighbor(scs, i)
            scs = move_pair_heading_swap(scs, i, j, abs(np.random.randn()) * scale * 0.3)
        if _ % 2 == 1:
            mask, cx, cy, _ = find_boundary_mask(scs)
    return scs


@nb.njit(cache=True)
def cluster_sa(scs_init, iters, temp, step, seed):
    """SA with mixed cluster/contact moves.
    Returns (best, best_score, final, final_score)."""
    np.random.seed(seed)
    scs = scs_init.copy()
    olaps = cnt(scs)
    score = mec(scs) if olaps == 0 else 1e9
    best = scs.copy()
    best_score = score

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
            r = np.random.random()
            old_scs = scs.copy()

            if r < 0.25:
                # Single perturbation
                idx = np.random.randint(0, N)
                scs[idx, 0] += np.random.randn() * step
                scs[idx, 1] += np.random.randn() * step
                scs[idx, 2] += np.random.randn() * step * 0.8
            elif r < 0.40:
                # Cluster translate
                seed_idx = np.random.randint(0, N)
                k = 2 + np.random.randint(0, 4)  # 2-5
                if k > N:
                    k = N
                mask = select_cluster(scs, seed_idx, k)
                dx = np.random.randn() * step * 0.7
                dy = np.random.randn() * step * 0.7
                scs = move_cluster_translate(scs, mask, dx, dy)
            elif r < 0.50:
                # Cluster rotate
                seed_idx = np.random.randint(0, N)
                k = 2 + np.random.randint(0, 4)
                if k > N:
                    k = N
                mask = select_cluster(scs, seed_idx, k)
                angle = np.random.randn() * step * 0.5
                scs = move_cluster_rotate(scs, mask, angle)
            elif r < 0.65:
                # Pair swap
                i = np.random.randint(0, N)
                j = np.random.randint(0, N - 1)
                if j >= i:
                    j += 1
                swap_angle = np.random.random() < 0.5
                scs = move_pair_swap(scs, i, j, swap_angle)
            elif r < 0.80:
                i = np.random.randint(0, N)
                j, _d = nearest_neighbor(scs, i)
                scs = move_slide_on_flat(
                    scs,
                    i,
                    j,
                    np.random.randn() * step,
                    np.random.randn() * step * 0.2,
                    np.random.randn() * step * 0.2,
                )
            elif r < 0.90:
                i = np.random.randint(0, N)
                j, _d = nearest_neighbor(scs, i)
                scs = move_roll_around_neighbor(
                    scs,
                    j,
                    i,
                    np.random.randn() * step * 0.35,
                    np.random.randn() * step * 0.2,
                )
            else:
                i = np.random.randint(0, N)
                j, _d = nearest_neighbor(scs, i)
                scs = move_pair_heading_swap(scs, i, j, abs(np.random.randn()) * step * 0.25)

            # Evaluate
            new_olaps = cnt(scs)
            if olaps > 0:
                if new_olaps < olaps or (
                    new_olaps == olaps and np.random.random() < 0.3
                ):
                    olaps = new_olaps
                    if olaps == 0:
                        score = mec(scs)
                        if score < best_score:
                            best_score = score
                            best = scs.copy()
                else:
                    scs = old_scs
                if it % 500_000 == 499_999 and olaps > 0:
                    scs = best.copy()
                    olaps = cnt(scs)
                    score = mec(scs) if olaps == 0 else 1e9
            else:
                if new_olaps > 0:
                    scs = old_scs
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
                        scs = old_scs

    return best, best_score, scs, score


# ============================================================
# Elastic relaxation (Phase 3)
# ============================================================


@nb.njit(cache=True)
def overlap_penalty(scs):
    """Count of actual overlapping pairs (using exact ov() check).
    Discrete but uses the correct overlap definition. Temperature handles non-smoothness."""
    return float(cnt(scs))


@nb.njit(cache=True)
def energy_elastic(scs, lam):
    """Composite energy: MEC radius + lambda * overlap penalty."""
    return mec(scs) + lam * overlap_penalty(scs)


@nb.njit(cache=True)
def elastic_sa(scs_init, iters, seed):
    """SA with soft overlap penalty that hardens over time.
    Lambda decays from 50 to 0.1 exponentially, then hard constraint for final 10%.
    Uses cluster moves in soft phase, single perturbation in hard phase.
    Returns (best, best_score, final, final_score)."""
    np.random.seed(seed)
    scs = scs_init.copy()
    temp = 0.05
    step = 0.08
    cool = 1 - 3e-6

    # Soft phase: 90% of iterations
    soft_iters = int(iters * 0.9)
    hard_iters = iters - soft_iters

    # Lambda schedule: 50 * (0.1/50)^(t/T) = 50 * exp(t/T * ln(0.1/50))
    ln_ratio = math.log(0.1 / 50.0)

    score = energy_elastic(scs, 50.0)
    best_valid = scs.copy()
    best_valid_score = 1e9
    if cnt(scs) == 0:
        best_valid_score = mec(scs)

    # Soft phase: allow overlaps, use cluster moves
    for it in range(soft_iters):
        temp *= cool
        frac = it / max(soft_iters - 1, 1)
        lam = 50.0 * math.exp(frac * ln_ratio)

        old_scs = scs.copy()
        r = np.random.random()

        if r < 0.50:
            # Single perturbation
            idx = np.random.randint(0, N)
            scs[idx, 0] += np.random.randn() * step
            scs[idx, 1] += np.random.randn() * step
            scs[idx, 2] += np.random.randn() * step * 0.8
        elif r < 0.75:
            # Cluster translate
            seed_idx = np.random.randint(0, N)
            k = 2 + np.random.randint(0, 3)
            mask = select_cluster(scs, seed_idx, k)
            dx = np.random.randn() * step * 0.5
            dy = np.random.randn() * step * 0.5
            scs = move_cluster_translate(scs, mask, dx, dy)
        else:
            # Cluster rotate
            seed_idx = np.random.randint(0, N)
            k = 2 + np.random.randint(0, 3)
            mask = select_cluster(scs, seed_idx, k)
            angle = np.random.randn() * step * 0.3
            scs = move_cluster_rotate(scs, mask, angle)

        new_score = energy_elastic(scs, lam)
        delta = new_score - score
        if delta < 0 or np.random.random() < math.exp(-delta / max(temp, 1e-15)):
            score = new_score
            # Track best valid
            if cnt(scs) == 0:
                ms = mec(scs)
                if ms < best_valid_score:
                    best_valid_score = ms
                    best_valid = scs.copy()
        else:
            scs = old_scs

        # Adaptive step
        if it % 100_000 == 99_999:
            step *= 0.9
            if step < 0.005:
                step = 0.005

    # Hard phase: strict feasibility, single perturbation only, greedy
    if cnt(scs) > 0 and best_valid_score < 1e9:
        scs = best_valid.copy()
    olaps = cnt(scs)
    score = mec(scs) if olaps == 0 else 1e9
    step = 0.01

    for it in range(hard_iters):
        idx = np.random.randint(0, N)
        ox, oy, ot = scs[idx, 0], scs[idx, 1], scs[idx, 2]
        scs[idx, 0] += np.random.randn() * step
        scs[idx, 1] += np.random.randn() * step
        scs[idx, 2] += np.random.randn() * step * 0.8

        if olaps > 0:
            nv = cnt(scs)
            if nv < olaps or (nv == olaps and np.random.random() < 0.3):
                olaps = nv
                if olaps == 0:
                    score = mec(scs)
                    if score < best_valid_score:
                        best_valid_score = score
                        best_valid = scs.copy()
            else:
                scs[idx, 0] = ox
                scs[idx, 1] = oy
                scs[idx, 2] = ot
        else:
            if chk(scs, idx):
                scs[idx, 0] = ox
                scs[idx, 1] = oy
                scs[idx, 2] = ot
            else:
                ns = mec(scs)
                if ns < score:
                    score = ns
                    if score < best_valid_score:
                        best_valid_score = score
                        best_valid = scs.copy()
                else:
                    scs[idx, 0] = ox
                    scs[idx, 1] = oy
                    scs[idx, 2] = ot

    return best_valid, best_valid_score, scs, score


# ============================================================
# Boundary-targeting SA
# ============================================================


@nb.njit(cache=True)
def boundary_sa(scs_init, iters, seed):
    """SA that targets MEC boundary contacts using orbit/slide/roll moves."""
    np.random.seed(seed)
    scs = scs_init.copy()
    if cnt(scs) > 0:
        return scs, 1e9, scs, 1e9

    score = mec(scs)
    best = scs.copy()
    best_score = score
    step = 0.0025
    temp = 0.0003
    cool = 1 - 1.5e-6

    boundary_interval = 20_000
    mask, cx, cy, r = find_boundary_mask(scs)
    n_boundary = 0
    for i in range(N):
        if mask[i]:
            n_boundary += 1
    stale = 0

    for it in range(iters):
        temp *= cool

        if n_boundary > 0 and np.random.random() < 0.8:
            count = 0
            target = np.random.randint(0, n_boundary)
            idx = 0
            for i in range(N):
                if mask[i]:
                    if count == target:
                        idx = i
                        break
                    count += 1
        else:
            idx = np.random.randint(0, N)

        old_scs = scs.copy()
        if mask[idx]:
            mv = np.random.random()
            if mv < 0.45:
                scs = move_boundary_orbit(
                    scs,
                    idx,
                    cx,
                    cy,
                    np.random.randn() * step * 0.25,
                    abs(np.random.randn()) * step * 0.9,
                )
            elif mv < 0.75:
                j, _d = nearest_neighbor(scs, idx)
                scs = move_slide_on_flat(
                    scs,
                    idx,
                    j,
                    np.random.randn() * step,
                    np.random.randn() * step * 0.2,
                    np.random.randn() * step * 0.2,
                )
            else:
                j, _d = nearest_neighbor(scs, idx)
                scs = move_roll_around_neighbor(
                    scs,
                    j,
                    idx,
                    np.random.randn() * step * 0.35,
                    np.random.randn() * step * 0.2,
                )
        else:
            if np.random.random() < 0.5:
                scs[idx, 0] += np.random.randn() * step
                scs[idx, 1] += np.random.randn() * step
                scs[idx, 2] += np.random.randn() * step * 0.25
            else:
                j, _d = nearest_neighbor(scs, idx)
                scs = move_slide_on_flat(
                    scs,
                    idx,
                    j,
                    np.random.randn() * step,
                    np.random.randn() * step * 0.2,
                    np.random.randn() * step * 0.15,
                )

        if chk(scs, idx):
            scs = old_scs
            stale += 1
        else:
            ns = mec(scs)
            delta = ns - score
            if delta < 0 or np.random.random() < math.exp(-delta / max(temp, 1e-15)):
                score = ns
                stale = 0
                if score < best_score:
                    best_score = score
                    best = scs.copy()
            else:
                scs = old_scs
                stale += 1

        if it % boundary_interval == boundary_interval - 1:
            mask, cx, cy, r = find_boundary_mask(scs)
            n_boundary = 0
            for i in range(N):
                if mask[i]:
                    n_boundary += 1
            if stale > boundary_interval // 2:
                temp = max(temp, 0.00015)
                step *= 1.1
                stale = 0
            else:
                step *= 0.95
            if step < 0.00008:
                step = 0.00008
            if step > 0.01:
                step = 0.01

    return best, best_score, scs, score


# ============================================================
# Coordinate descent
# ============================================================


@nb.njit(cache=True)
def coord_descent(scs_init, max_evals, seed):
    """Systematic local search over Cartesian and semicircle-specific move bases."""
    np.random.seed(seed)
    scs = scs_init.copy()
    if cnt(scs) > 0:
        return scs, 1e9, scs, 1e9

    score = mec(scs)
    best = scs.copy()
    best_score = score
    total_evals = 0

    steps = np.array([0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005])

    for si in range(len(steps)):
        step = steps[si]
        improved = True
        passes = 0
        while improved and total_evals < max_evals and passes < 12:
            improved = False
            passes += 1
            mask, cx, cy, _r = find_boundary_mask(scs)
            order = np.random.permutation(N)
            for ii in range(N):
                if total_evals >= max_evals:
                    break
                idx = order[ii]
                j, _d = nearest_neighbor(scs, idx)

                for coord in range(3):
                    if total_evals >= max_evals:
                        break
                    delta = step if coord < 2 else step * 0.3
                    for sign in range(2):
                        direction = 1.0 if sign == 0 else -1.0
                        trial = scs.copy()
                        trial[idx, coord] += direction * delta
                        total_evals += 1

                        if not chk(trial, idx):
                            ns = mec(trial)
                            if ns < score:
                                scs = trial
                                score = ns
                                improved = True
                                if score < best_score:
                                    best_score = score
                                    best = scs.copy()

                dx = cx - scs[idx, 0]
                dy = cy - scs[idx, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 1e-12:
                    rx = dx / dist
                    ry = dy / dist
                    tx = -ry
                    ty = rx
                    for sign in range(2):
                        if total_evals >= max_evals:
                            break
                        direction = 1.0 if sign == 0 else -1.0
                        trial = scs.copy()
                        trial[idx, 0] += rx * step * direction
                        trial[idx, 1] += ry * step * direction
                        total_evals += 1
                        if not chk(trial, idx):
                            ns = mec(trial)
                            if ns < score:
                                scs = trial
                                score = ns
                                improved = True
                                if score < best_score:
                                    best_score = score
                                    best = scs.copy()

                        if total_evals >= max_evals:
                            break
                        trial = scs.copy()
                        trial[idx, 0] += tx * step * direction
                        trial[idx, 1] += ty * step * direction
                        trial[idx, 2] += direction * step * 0.1
                        total_evals += 1
                        if not chk(trial, idx):
                            ns = mec(trial)
                            if ns < score:
                                scs = trial
                                score = ns
                                improved = True
                                if score < best_score:
                                    best_score = score
                                    best = scs.copy()

                for sign in range(2):
                    if total_evals >= max_evals:
                        break
                    direction = 1.0 if sign == 0 else -1.0
                    trial = move_slide_on_flat(
                        scs, idx, j, direction * step, 0.0, direction * step * 0.12
                    )
                    total_evals += 1
                    if not chk(trial, idx):
                        ns = mec(trial)
                        if ns < score:
                            scs = trial
                            score = ns
                            improved = True
                            if score < best_score:
                                best_score = score
                                best = scs.copy()

                    if total_evals >= max_evals:
                        break
                    trial = move_roll_around_neighbor(
                        scs, j, idx, direction * step * 0.18, 0.0
                    )
                    total_evals += 1
                    if not chk(trial, idx):
                        ns = mec(trial)
                        if ns < score:
                            scs = trial
                            score = ns
                            improved = True
                            if score < best_score:
                                best_score = score
                                best = scs.copy()

                if total_evals < max_evals and not mask[idx]:
                    trial = move_pair_heading_swap(scs, idx, j, step * 0.12)
                    total_evals += 1
                    if cnt(trial) == 0:
                        ns = mec(trial)
                        if ns < score:
                            scs = trial
                            score = ns
                            improved = True
                            if score < best_score:
                                best_score = score
                                best = scs.copy()

    return best, best_score, scs, score


# ============================================================
# Adaptive SA (PT-MCMC inspired)
# ============================================================


@nb.njit(cache=True)
def adaptive_sa(scs_init, iters, temp_start, seed):
    """SA with adaptive step sizing targeting ~23% acceptance rate.
    Two-phase: resolve overlaps first, then Metropolis refinement."""
    np.random.seed(seed)
    scs = scs_init.copy()
    olaps = cnt(scs)
    score = mec(scs) if olaps == 0 else 1e9
    best = scs.copy()
    best_score = score

    step = 0.01
    step_min = 0.0001
    step_max = 0.5
    temp = temp_start
    cool = 1 - 3e-6

    # Acceptance tracking
    window = 10_000
    accepted = 0
    total_in_window = 0
    target_rate = 0.234

    for it in range(iters):
        temp *= cool
        idx = np.random.randint(0, N)
        ox = scs[idx, 0]
        oy = scs[idx, 1]
        ot = scs[idx, 2]
        scs[idx, 0] += np.random.randn() * step
        scs[idx, 1] += np.random.randn() * step
        scs[idx, 2] += np.random.randn() * step * 0.8

        total_in_window += 1
        did_accept = False

        if olaps > 0:
            nv = cnt(scs)
            if nv < olaps or (nv == olaps and np.random.random() < 0.3):
                olaps = nv
                did_accept = True
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
                    did_accept = True
                    if score < best_score:
                        best_score = score
                        best = scs.copy()
                else:
                    scs[idx, 0] = ox
                    scs[idx, 1] = oy
                    scs[idx, 2] = ot

        if did_accept:
            accepted += 1

        # Adapt step size every window iterations
        if total_in_window >= window:
            rate = accepted / window
            if rate > target_rate:
                step *= 1.1
                if step > step_max:
                    step = step_max
            else:
                step *= 0.9
                if step < step_min:
                    step = step_min
            accepted = 0
            total_in_window = 0

    return best, best_score, scs, score


# ============================================================
# Quantized adaptive SA
# ============================================================


@nb.njit(cache=True)
def quantized_adaptive_sa(scs_init, iters, temp_start, seed):
    """SA in 6-decimal space with discrete permutations for rounding-limited basins."""
    np.random.seed(seed)
    scs = rnd(scs_init.copy())
    if cnt(scs) > 0:
        return scs, 1e9, scs, 1e9

    score = mec(scs)
    best = scs.copy()
    best_score = score

    step = 0.003
    step_min = 1e-6
    step_max = 0.1
    temp = temp_start
    cool = 1 - 3e-6

    window = 10_000
    accepted = 0
    total_in_window = 0
    target_rate = 0.234

    for it in range(iters):
        temp *= cool

        r = np.random.random()

        if r < 0.10:
            a = np.random.randint(0, N)
            b = np.random.randint(0, N - 1)
            if b >= a:
                b += 1
            c = np.random.randint(0, N - 2)
            if c >= min(a, b):
                c += 1
            if c >= max(a, b):
                c += 1
            old_a = (scs[a, 0], scs[a, 1], scs[a, 2])
            old_b = (scs[b, 0], scs[b, 1], scs[b, 2])
            old_c = (scs[c, 0], scs[c, 1], scs[c, 2])
            scs[a, 0], scs[a, 1] = old_b[0], old_b[1]
            scs[b, 0], scs[b, 1] = old_c[0], old_c[1]
            scs[c, 0], scs[c, 1] = old_a[0], old_a[1]
            if np.random.random() < 0.5:
                scs[a, 2], scs[b, 2], scs[c, 2] = old_b[2], old_c[2], old_a[2]
            move_type = 2
        elif r < 0.20:
            a = np.random.randint(0, N)
            b = np.random.randint(0, N - 1)
            if b >= a:
                b += 1
            old_a = (scs[a, 0], scs[a, 1], scs[a, 2])
            old_b = (scs[b, 0], scs[b, 1], scs[b, 2])
            scs[a, 0], scs[a, 1] = old_b[0], old_b[1]
            scs[b, 0], scs[b, 1] = old_a[0], old_a[1]
            if np.random.random() < 0.5:
                scs[a, 2], scs[b, 2] = old_b[2], old_a[2]
            old_c = old_a
            c = a
            move_type = 1
        else:
            idx = np.random.randint(0, N)
            old_a = (scs[idx, 0], scs[idx, 1], scs[idx, 2])
            scs[idx, 0] = np.round((scs[idx, 0] + np.random.randn() * step) * 1e6) / 1e6
            scs[idx, 1] = np.round((scs[idx, 1] + np.random.randn() * step) * 1e6) / 1e6
            scs[idx, 2] = (
                np.round(
                    ((scs[idx, 2] + np.random.randn() * step * 0.3) % (2 * math.pi))
                    * 1e6
                )
                / 1e6
            )
            a = idx
            old_b = old_a
            old_c = old_a
            b = a
            c = a
            move_type = 0

        total_in_window += 1
        did_accept = False

        if move_type == 0:
            has_overlap = chk(scs, a)
        else:
            has_overlap = cnt(scs) > 0

        if has_overlap:
            if move_type == 2:
                scs[a, 0], scs[a, 1], scs[a, 2] = old_a[0], old_a[1], old_a[2]
                scs[b, 0], scs[b, 1], scs[b, 2] = old_b[0], old_b[1], old_b[2]
                scs[c, 0], scs[c, 1], scs[c, 2] = old_c[0], old_c[1], old_c[2]
            elif move_type == 1:
                scs[a, 0], scs[a, 1], scs[a, 2] = old_a[0], old_a[1], old_a[2]
                scs[b, 0], scs[b, 1], scs[b, 2] = old_b[0], old_b[1], old_b[2]
            else:
                scs[a, 0], scs[a, 1], scs[a, 2] = old_a[0], old_a[1], old_a[2]
        else:
            ns = mec(scs)
            delta = ns - score
            if delta < 0 or np.random.random() < math.exp(-delta / max(temp, 1e-15)):
                score = ns
                did_accept = True
                if score < best_score:
                    best_score = score
                    best = scs.copy()
            else:
                if move_type == 2:
                    scs[a, 0], scs[a, 1], scs[a, 2] = old_a[0], old_a[1], old_a[2]
                    scs[b, 0], scs[b, 1], scs[b, 2] = old_b[0], old_b[1], old_b[2]
                    scs[c, 0], scs[c, 1], scs[c, 2] = old_c[0], old_c[1], old_c[2]
                elif move_type == 1:
                    scs[a, 0], scs[a, 1], scs[a, 2] = old_a[0], old_a[1], old_a[2]
                    scs[b, 0], scs[b, 1], scs[b, 2] = old_b[0], old_b[1], old_b[2]
                else:
                    scs[a, 0], scs[a, 1], scs[a, 2] = old_a[0], old_a[1], old_a[2]

        if did_accept:
            accepted += 1

        if total_in_window >= window:
            rate = accepted / window
            if rate > target_rate:
                step *= 1.1
                if step > step_max:
                    step = step_max
            else:
                step *= 0.9
                if step < step_min:
                    step = step_min
            accepted = 0
            total_in_window = 0

    return best, best_score, scs, score


# ============================================================
# Internal Parallel Tempering With Semicircle-Specific Moves
# ============================================================


@nb.njit(cache=True)
def tempered_geometric_search(scs_init, iters, temp_start, step_start, seed):
    """Replica exchange over contact-aware moves to cross plateaus SA cannot."""
    np.random.seed(seed)
    n_rep = 6
    temps = np.empty(n_rep)
    steps = np.empty(n_rep)
    replicas = np.empty((n_rep, N, 3))
    scores = np.empty(n_rep)
    olaps = np.empty(n_rep, dtype=nb.int64)

    base_temp = max(temp_start * 0.25, 5e-5)
    base_step = max(step_start * 0.35, 7e-4)

    best = scs_init.copy()
    best_score = 1e9

    for r in range(n_rep):
        temps[r] = min(1.5, base_temp * (3.0**r))
        steps[r] = min(0.18, base_step * (1.9**r))
        if r == 0:
            replicas[r] = scs_init.copy()
        else:
            replicas[r] = structured_kick(
                scs_init, steps[r] * (1.0 + 0.2 * r), seed + r * 997
            )
        olaps[r] = cnt(replicas[r])
        scores[r] = mec(replicas[r]) if olaps[r] == 0 else 1e9
        if olaps[r] == 0 and scores[r] < best_score:
            best_score = scores[r]
            best = replicas[r].copy()

    sweeps = max(1, iters // (n_rep * 200_000))
    per_rep = max(500, iters // (max(1, sweeps) * n_rep))

    for sw in range(sweeps):
        for r in range(n_rep):
            scs = replicas[r].copy()
            score = scores[r]
            olap = olaps[r]
            mask, cx, cy, _r = find_boundary_mask(scs)
            n_boundary = 0
            for i in range(N):
                if mask[i]:
                    n_boundary += 1
            accepted = 0

            for it in range(per_rep):
                if it % 256 == 0:
                    mask, cx, cy, _r = find_boundary_mask(scs)
                    n_boundary = 0
                    for i in range(N):
                        if mask[i]:
                            n_boundary += 1

                old_scs = scs.copy()
                mv = np.random.random()

                if mv < 0.28:
                    idx = np.random.randint(0, N)
                    scs[idx, 0] += np.random.randn() * steps[r]
                    scs[idx, 1] += np.random.randn() * steps[r]
                    scs[idx, 2] += np.random.randn() * steps[r] * 0.7
                elif mv < 0.50:
                    idx = np.random.randint(0, N)
                    j, _d = nearest_neighbor(scs, idx)
                    scs = move_slide_on_flat(
                        scs,
                        idx,
                        j,
                        np.random.randn() * steps[r],
                        np.random.randn() * steps[r] * 0.25,
                        np.random.randn() * steps[r] * 0.25,
                    )
                elif mv < 0.70:
                    idx = np.random.randint(0, N)
                    j, _d = nearest_neighbor(scs, idx)
                    scs = move_roll_around_neighbor(
                        scs,
                        j,
                        idx,
                        np.random.randn() * steps[r] * 0.35,
                        np.random.randn() * steps[r] * 0.2,
                    )
                elif mv < 0.86:
                    if n_boundary > 0:
                        target = np.random.randint(0, n_boundary)
                        count = 0
                        idx = 0
                        for i in range(N):
                            if mask[i]:
                                if count == target:
                                    idx = i
                                    break
                                count += 1
                    else:
                        idx = np.random.randint(0, N)
                    scs = move_boundary_orbit(
                        scs,
                        idx,
                        cx,
                        cy,
                        np.random.randn() * steps[r] * 0.25,
                        np.random.randn() * steps[r] * 0.4,
                    )
                else:
                    idx = np.random.randint(0, N)
                    j, _d = nearest_neighbor(scs, idx)
                    scs = move_pair_heading_swap(
                        scs, idx, j, abs(np.random.randn()) * steps[r] * 0.35
                    )

                new_olaps = cnt(scs)
                did_accept = False

                if olap > 0:
                    if new_olaps < olap or (
                        new_olaps == olap and np.random.random() < 0.15
                    ):
                        olap = new_olaps
                        did_accept = True
                        if olap == 0:
                            score = mec(scs)
                            if score < best_score:
                                best_score = score
                                best = scs.copy()
                    else:
                        scs = old_scs
                else:
                    if new_olaps > 0:
                        scs = old_scs
                    else:
                        ns = mec(scs)
                        delta = ns - score
                        if delta < 0 or np.random.random() < math.exp(
                            -delta / max(temps[r], 1e-15)
                        ):
                            score = ns
                            did_accept = True
                            if score < best_score:
                                best_score = score
                                best = scs.copy()
                        else:
                            scs = old_scs

                if did_accept:
                    accepted += 1

            replicas[r] = scs
            scores[r] = score
            olaps[r] = olap

            rate = accepted / max(per_rep, 1)
            if rate < 0.12:
                steps[r] *= 0.85
            elif rate > 0.35:
                steps[r] *= 1.12
            if steps[r] < 2e-4:
                steps[r] = 2e-4
            if steps[r] > 0.18:
                steps[r] = 0.18

        for parity in range(2):
            for i in range(parity, n_rep - 1, 2):
                j = i + 1
                if olaps[i] == 0 and olaps[j] == 0:
                    beta_i = 1.0 / max(temps[i], 1e-15)
                    beta_j = 1.0 / max(temps[j], 1e-15)
                    delta = (beta_i - beta_j) * (scores[j] - scores[i])
                    if delta > 0 or np.random.random() < math.exp(max(-500.0, delta)):
                        tmp = replicas[i].copy()
                        replicas[i] = replicas[j]
                        replicas[j] = tmp
                        tmp_score = scores[i]
                        scores[i] = scores[j]
                        scores[j] = tmp_score
                        tmp_olap = olaps[i]
                        olaps[i] = olaps[j]
                        olaps[j] = tmp_olap

        if sw % 2 == 1 and best_score < 1e8:
            replicas[0] = best.copy()
            scores[0] = best_score
            olaps[0] = 0

        if sw % 5 == 4:
            if best_score < 1e8:
                base = best
            else:
                base = scs_init
            replicas[n_rep - 1] = structured_kick(
                base, steps[n_rep - 1] * 2.5, seed + 50_000 + sw
            )
            olaps[n_rep - 1] = cnt(replicas[n_rep - 1])
            scores[n_rep - 1] = mec(replicas[n_rep - 1]) if olaps[n_rep - 1] == 0 else 1e9

    cold_final = replicas[0].copy()
    cold_score = scores[0] if olaps[0] == 0 else 1e9
    if best_score < 1e8:
        polished, polished_score, _, _ = coord_descent(
            best, min(500_000, max(20_000, per_rep)), seed + 777
        )
        if polished_score < best_score:
            best = polished
            best_score = polished_score

    return best, best_score, cold_final, cold_score


# ============================================================
# SA chain (copied from pt_optimizer.py)
# ============================================================


@nb.njit(cache=True)
def sa_chain(scs_init, iters, temp, step, seed):
    """Single SA chain with basin hopping. Returns (best, best_score, final, final_score)."""
    np.random.seed(seed)
    scs = scs_init.copy()
    olaps = cnt(scs)
    score = mec(scs) if olaps == 0 else 1e9
    best = scs.copy()
    best_score = score

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


# ============================================================
# Geometric constructors (Phase 4)
# ============================================================


@nb.njit(cache=True)
def seed_ring(n_inner, r_inner, n_outer, r_outer, off):
    """Place semicircles in two concentric rings with angular offset."""
    scs = np.zeros((N, 3))
    k = 0
    for i in range(min(n_inner, N)):
        a = 2 * math.pi * i / max(n_inner, 1) + off
        scs[k, 0] = r_inner * math.cos(a)
        scs[k, 1] = r_inner * math.sin(a)
        scs[k, 2] = a + math.pi  # face inward
        k += 1
        if k >= N:
            break
    for i in range(min(n_outer, N - k)):
        a = 2 * math.pi * i / max(n_outer, 1) + off * 0.5
        scs[k, 0] = r_outer * math.cos(a)
        scs[k, 1] = r_outer * math.sin(a)
        scs[k, 2] = a  # face outward
        k += 1
        if k >= N:
            break
    # Fill remaining with center-ish positions
    while k < N:
        a = 2 * math.pi * k / N
        scs[k, 0] = 0.3 * math.cos(a)
        scs[k, 1] = 0.3 * math.sin(a)
        scs[k, 2] = a
        k += 1
    return scs


@nb.njit(cache=True)
def seed_reflect(scs_base, axis_angle, seed):
    """Reflect base solution across an axis, then jitter to break symmetry."""
    np.random.seed(seed)
    out = np.zeros((N, 3))
    ca = math.cos(2 * axis_angle)
    sa = math.sin(2 * axis_angle)
    for i in range(N):
        x, y = scs_base[i, 0], scs_base[i, 1]
        # Reflect position
        out[i, 0] = x * ca + y * sa
        out[i, 1] = x * sa - y * ca
        # Reflect angle
        out[i, 2] = 2 * axis_angle - scs_base[i, 2]
    # Jitter
    for i in range(N):
        out[i, 0] += np.random.randn() * 0.02
        out[i, 1] += np.random.randn() * 0.02
        out[i, 2] += np.random.randn() * 0.02
    return out


@nb.njit(cache=True)
def constructor_sa(scs_init, iters, seed):
    """Quick SA polish for geometric constructors. Returns (best, best_score, final, final_score)."""
    np.random.seed(seed)
    scs = scs_init.copy()
    olaps = cnt(scs)
    score = mec(scs) if olaps == 0 else 1e9
    best = scs.copy()
    best_score = score
    temp = 0.1
    step = 0.05
    cool = 1 - 1e-5

    for it in range(iters):
        temp *= cool
        idx = np.random.randint(0, N)
        ox, oy, ot = scs[idx, 0], scs[idx, 1], scs[idx, 2]
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
            if it % 100_000 == 99_999 and olaps > 0:
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


# ============================================================
# Pool I/O
# ============================================================


def pool_init():
    """Create pool directory if needed."""
    os.makedirs(POOL_DIR, exist_ok=True)


def pool_write_best(score, scs):
    """Atomically write best solution to pool and standard files."""
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
    # Pool best
    tmp = os.path.join(POOL_DIR, "best.json.tmp")
    dst = os.path.join(POOL_DIR, "best.json")
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, dst)
    # Standard files (for sync_best.sh compatibility)
    tmp2 = BEST_FILE + ".tmp"
    with open(tmp2, "w") as f:
        json.dump(data, f)
    os.replace(tmp2, BEST_FILE)
    with open(SOLUTION_FILE, "w") as f:
        json.dump(data["solution"], f, indent=2)
    return True


def pool_read_best():
    """Read best solution from pool. Returns (score, scs) or (None, None)."""
    for path in [os.path.join(POOL_DIR, "best.json"), BEST_FILE, SOLUTION_FILE]:
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict) and "scs" in data:
                scs = np.array(data["scs"])
                return data["score"], scs
            elif isinstance(data, list):
                scs = np.array([[s["x"], s["y"], s["theta"]] for s in data])
                if cnt(scs) == 0:
                    return mec(scs), scs
        except Exception:
            continue
    return None, None


def pool_write_diverse(slot, score, scs):
    """Atomically write a diverse solution to pool."""
    s = scs.reshape(N, 3) if scs.ndim == 1 else scs
    data = {"score": float(score), "scs": s.tolist()}
    tmp = os.path.join(POOL_DIR, f"diverse_{slot}.json.tmp")
    dst = os.path.join(POOL_DIR, f"diverse_{slot}.json")
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, dst)


def pool_read_diverse():
    """Read all diverse solutions. Returns list of (score, scs) tuples."""
    results = []
    for i in range(DIVERSE_SLOTS):
        path = os.path.join(POOL_DIR, f"diverse_{i}.json")
        try:
            with open(path) as f:
                data = json.load(f)
            results.append((data["score"], np.array(data["scs"])))
        except Exception:
            continue
    return results


def pool_write_checkpoint(state):
    """Write full optimizer state for crash recovery."""
    tmp = os.path.join(POOL_DIR, "checkpoint.json.tmp")
    dst = os.path.join(POOL_DIR, "checkpoint.json")
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, dst)


# ============================================================
# Diverse pool management
# ============================================================


def solution_signature(scs):
    """Rotation-invariant archive fingerprint using sorted polar/radial/orientation features."""
    s = scs.reshape(N, 3) if scs.ndim == 1 else scs
    pts = s[:, :2].astype(float)
    center = pts.mean(axis=0)
    rel = pts - center
    radii = np.sqrt(np.sum(rel**2, axis=1))
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    rel_theta = ((s[:, 2] - ang + np.pi) % (2 * np.pi)) - np.pi
    order = np.lexsort((radii, ang))
    radii = radii[order]
    ang = ang[order]
    rel_theta = rel_theta[order]
    ang_steps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
    return np.concatenate([radii, ang_steps, 0.35 * rel_theta])


def solution_l2(scs1, scs2):
    """Distance between archive fingerprints, not label-ordered raw coordinates."""
    a = solution_signature(scs1)
    b = solution_signature(scs2)
    return np.sqrt(np.sum((a - b) ** 2))


def update_diverse_pool(new_score, new_scs, current_pool):
    """Update diverse pool with a new solution. Returns updated pool."""
    s = new_scs.reshape(N, 3) if new_scs.ndim == 1 else new_scs

    if len(current_pool) < DIVERSE_SLOTS:
        # Check L2 distance from all existing
        for _, existing_scs in current_pool:
            if solution_l2(s, existing_scs) < DIVERSE_MIN_L2:
                # Too similar, only replace if better
                break
        else:
            current_pool.append((new_score, s.copy()))
            current_pool.sort(key=lambda x: x[0])
            return current_pool

    # Pool is full - check if new solution is diverse enough
    min_dist = min(solution_l2(s, ex_scs) for _, ex_scs in current_pool)

    if min_dist >= DIVERSE_MIN_L2:
        # Structurally distinct - replace worst
        current_pool[-1] = (new_score, s.copy())
        current_pool.sort(key=lambda x: x[0])
    elif new_score < current_pool[-1][0]:
        # Better than worst, find the closest and replace if better
        closest_idx = 0
        closest_dist = float("inf")
        for i, (_, ex_scs) in enumerate(current_pool):
            d = solution_l2(s, ex_scs)
            if d < closest_dist:
                closest_dist = d
                closest_idx = i
        if new_score < current_pool[closest_idx][0]:
            current_pool[closest_idx] = (new_score, s.copy())
            current_pool.sort(key=lambda x: x[0])

    return current_pool


# ============================================================
# Worker types
# ============================================================

# Worker type constants
TYPE_A = 0  # Cold refiner
TYPE_B = 1  # Hot explorer
TYPE_C = 2  # Cluster mover (Phase 2)
TYPE_D = 3  # Elastic relaxation (Phase 3)
TYPE_E = 4  # Geometric constructor (Phase 4)
TYPE_F = 5  # Polisher
TYPE_G = 6  # Boundary-targeting SA
TYPE_H = 7  # Coordinate descent
TYPE_I = 8  # Internal replica-exchange geometric search
TYPE_Q = 9  # Quantized adaptive SA
TYPE_R = 10  # Random restart worker


def run_worker(args):
    """Dispatch to appropriate worker type. Called by Pool.map."""
    worker_id, worker_type, scs_flat, temp, step, seed, iters = args
    scs = scs_flat.reshape(N, 3)

    if worker_type == TYPE_A:
        # Adaptive SA with low starting temp (exploiter)
        best, best_score, final, final_score = adaptive_sa(scs, iters, temp, seed)
        return (
            worker_id,
            worker_type,
            best.flatten(),
            best_score,
            final.flatten(),
            final_score,
        )
    elif worker_type == TYPE_B:
        # Adaptive SA with high starting temp (explorer)
        best, best_score, final, final_score = adaptive_sa(scs, iters, temp, seed)
        return (
            worker_id,
            worker_type,
            best.flatten(),
            best_score,
            final.flatten(),
            final_score,
        )
    elif worker_type == TYPE_F:
        # Polisher: greedy descent in quantized space
        r = rnd(scs)
        if cnt(r) == 0:
            best, best_score = polish_q(r, min(iters, 500_000), seed)
            return (
                worker_id,
                worker_type,
                best.flatten(),
                best_score,
                best.flatten(),
                best_score,
            )
        else:
            return worker_id, worker_type, scs_flat, 1e9, scs_flat, 1e9
    elif worker_type == TYPE_C:
        # Cluster mover: mixed move operators
        best, best_score, final, final_score = cluster_sa(scs, iters, temp, step, seed)
        return (
            worker_id,
            worker_type,
            best.flatten(),
            best_score,
            final.flatten(),
            final_score,
        )
    elif worker_type == TYPE_D:
        # Elastic relaxation
        best, best_score, final, final_score = elastic_sa(scs, iters, seed)
        return (
            worker_id,
            worker_type,
            best.flatten(),
            best_score,
            final.flatten(),
            final_score,
        )
    elif worker_type == TYPE_E:
        # Geometric constructor: try 3 different seeds, return best
        best_score = 1e9
        best_flat = scs_flat
        ring_configs = [
            (5, 0.8, 10, 2.0),
            (6, 1.0, 9, 2.2),
            (4, 0.6, 11, 2.1),
            (7, 1.2, 8, 2.3),
            (3, 0.5, 12, 2.0),
        ]
        for trial in range(3):
            sub_seed = seed + trial * 1000
            if trial == 0:
                # Ring layout
                cfg_idx = (seed // 1000) % len(ring_configs)
                ni, ri, no, ro = ring_configs[cfg_idx]
                trial_scs = seed_ring(ni, ri, no, ro, sub_seed * 0.01)
            elif trial == 1:
                # Reflected best
                trial_scs = seed_reflect(scs, (sub_seed % 628) / 100.0, sub_seed)
            else:
                # Perturbed best
                trial_scs = scs.copy()
                trial_scs += (
                    np.random.default_rng(sub_seed).standard_normal((N, 3)) * 0.1
                )
            b, bs, _, _ = constructor_sa(trial_scs, min(iters, 500_000), sub_seed)
            if bs < best_score:
                best_score = bs
                best_flat = b.flatten()
        return worker_id, worker_type, best_flat, best_score, best_flat, best_score
    elif worker_type == TYPE_G:
        # Boundary-targeting SA
        best, best_score, final, final_score = boundary_sa(scs, iters, seed)
        return (
            worker_id,
            worker_type,
            best.flatten(),
            best_score,
            final.flatten(),
            final_score,
        )
    elif worker_type == TYPE_H:
        # Coordinate descent
        best, best_score, final, final_score = coord_descent(scs, iters, seed)
        return (
            worker_id,
            worker_type,
            best.flatten(),
            best_score,
            final.flatten(),
            final_score,
        )
    elif worker_type == TYPE_I:
        # Replica exchange with geometry-preserving moves
        best, best_score, final, final_score = tempered_geometric_search(
            scs, iters, temp, step, seed
        )
        return (
            worker_id,
            worker_type,
            best.flatten(),
            best_score,
            final.flatten(),
            final_score,
        )
    elif worker_type == TYPE_Q:
        # Quantized adaptive SA with permutation moves
        best, best_score, final, final_score = quantized_adaptive_sa(
            scs, iters, temp, seed
        )
        return (
            worker_id,
            worker_type,
            best.flatten(),
            best_score,
            final.flatten(),
            final_score,
        )
    elif worker_type == TYPE_R:
        # Random restart worker: structured kicks from flower/reflection seeds
        best_score = 1e9
        best_flat = scs_flat
        ring_configs = [
            flower(3, 0.7, 2.2),
            flower(4, 0.7, 2.0),
            flower(5, 0.9, 2.5),
            seed_ring(6, 1.0, 9, 2.2, 0.0),
        ]
        for trial in range(3):
            sub_seed = seed + trial * 10_000
            if trial == 0:
                trial_scs = ring_configs[(sub_seed // 1000) % len(ring_configs)].copy()
            elif trial == 1:
                trial_scs = seed_reflect(scs, (sub_seed % 628) / 100.0, sub_seed)
            else:
                trial_scs = structured_kick(
                    HOT_CONFIGS[(sub_seed // 2000) % len(HOT_CONFIGS)],
                    0.08,
                    sub_seed,
                )
            b, bs, _, _ = tempered_geometric_search(
                trial_scs, max(100_000, iters // 3), max(temp, 0.002), max(step, 0.01), sub_seed
            )
            if bs < best_score:
                best_score = bs
                best_flat = b.flatten()
        return worker_id, worker_type, best_flat, best_score, best_flat, best_score
    else:
        # Fallback
        best, best_score, final, final_score = sa_chain(scs, iters, temp, step, seed)
        return (
            worker_id,
            worker_type,
            best.flatten(),
            best_score,
            final.flatten(),
            final_score,
        )


# ============================================================
# Phase schedule and worker assignment
# ============================================================

# Temperature and step size presets per worker type
TEMPS_COLD = np.geomspace(0.00005, 0.002, 10)  # for Type A (finer than before)
TEMPS_HOT = np.geomspace(0.5, 5.0, 10)  # for Type B
STEPS_COLD = np.geomspace(0.0005, 0.008, 10)  # for Type A (finer than before)
STEPS_HOT = np.geomspace(0.05, 0.25, 10)


def get_phase(elapsed_hours):
    """Return current phase (1, 2, or 3)."""
    if elapsed_hours < PHASE_1_END:
        return 1
    elif elapsed_hours < PHASE_2_END:
        return 2
    else:
        return 3


def get_worker_assignments(phase, n_workers):
    """Return list of (worker_type, temp, step) for each worker slot.

    Base mix for 14 workers, with extra slots filled by PT/quantized/local workers.
    """
    if phase == 1:
        counts = {
            TYPE_A: 2,
            TYPE_B: 1,
            TYPE_C: 1,
            TYPE_G: 2,
            TYPE_H: 1,
            TYPE_I: 3,
            TYPE_Q: 2,
            TYPE_R: 1,
            TYPE_F: 1,
        }
    elif phase == 2:
        counts = {
            TYPE_A: 3,
            TYPE_C: 1,
            TYPE_G: 2,
            TYPE_H: 2,
            TYPE_I: 3,
            TYPE_Q: 2,
            TYPE_F: 1,
        }
    else:
        counts = {
            TYPE_A: 3,
            TYPE_G: 2,
            TYPE_H: 3,
            TYPE_I: 3,
            TYPE_Q: 2,
            TYPE_F: 1,
        }

    assignments = []
    for wtype in [
        TYPE_A,
        TYPE_B,
        TYPE_C,
        TYPE_D,
        TYPE_E,
        TYPE_G,
        TYPE_H,
        TYPE_I,
        TYPE_Q,
        TYPE_R,
        TYPE_F,
    ]:
        for i in range(counts.get(wtype, 0)):
            if wtype == TYPE_A:
                idx = i % len(TEMPS_COLD)
                assignments.append((wtype, TEMPS_COLD[idx], STEPS_COLD[idx]))
            elif wtype == TYPE_B:
                idx = i % len(TEMPS_HOT)
                assignments.append((wtype, TEMPS_HOT[idx], STEPS_HOT[idx]))
            elif wtype == TYPE_C:
                assignments.append((wtype, 0.01 + i * 0.03, 0.01 + i * 0.005))
            elif wtype == TYPE_D:
                assignments.append((wtype, 0.005 + i * 0.01, 0.08 - i * 0.02))
            elif wtype == TYPE_E:
                assignments.append((wtype, 0.01, 0.05))
            elif wtype == TYPE_G:
                # Boundary targeting (temp/step unused, boundary_sa has its own)
                assignments.append((wtype, 0.0002, 0.002))
            elif wtype == TYPE_H:
                # Coordinate descent (temp/step unused)
                assignments.append((wtype, 0.0, 0.001))
            elif wtype == TYPE_I:
                assignments.append((wtype, 0.0003 * (2.5**i), 0.002 * (1.9**i)))
            elif wtype == TYPE_Q:
                assignments.append((wtype, 0.0004 * (2.0**i), 0.003))
            elif wtype == TYPE_R:
                assignments.append((wtype, 0.003, 0.04))
            elif wtype == TYPE_F:
                assignments.append((wtype, 0.0, 0.003))

    extra_cycle = [TYPE_I, TYPE_A, TYPE_Q, TYPE_G, TYPE_H, TYPE_R, TYPE_B]
    extra_i = 0
    while len(assignments) < n_workers:
        wtype = extra_cycle[extra_i % len(extra_cycle)]
        tier = extra_i // len(extra_cycle)
        if wtype == TYPE_I:
            assignments.append((wtype, 0.0006 * (1.8**tier), 0.003 * (1.6**tier)))
        elif wtype == TYPE_A:
            idx = (tier + 3) % len(TEMPS_COLD)
            assignments.append((wtype, TEMPS_COLD[idx], STEPS_COLD[idx]))
        elif wtype == TYPE_Q:
            assignments.append((wtype, 0.0008 * (1.6**tier), 0.003))
        elif wtype == TYPE_G:
            assignments.append((wtype, 0.0002, 0.002))
        elif wtype == TYPE_H:
            assignments.append((wtype, 0.0, 0.001))
        elif wtype == TYPE_R:
            assignments.append((wtype, 0.003, 0.05))
        else:
            idx = tier % len(TEMPS_HOT)
            assignments.append((wtype, TEMPS_HOT[idx], STEPS_HOT[idx]))
        extra_i += 1
    return assignments[:n_workers]


# ============================================================
# Initialization configs
# ============================================================

HOT_CONFIGS = [
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

TYPE_NAMES = {
    TYPE_A: "A",
    TYPE_B: "B",
    TYPE_C: "C",
    TYPE_D: "D",
    TYPE_E: "E",
    TYPE_F: "F",
    TYPE_G: "G",
    TYPE_H: "H",
    TYPE_I: "I",
    TYPE_Q: "Q",
    TYPE_R: "R",
}


# ============================================================
# Main loop
# ============================================================


def main():
    t0 = time.time()
    TIME_HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    TIME_LIMIT = TIME_HOURS * 3600

    print("=== Heterogeneous Optimizer ===", flush=True)
    print(
        f"Workers: {N_WORKERS}, Time: {TIME_HOURS:.1f}h, Iters/round: {ITERS_PER_ROUND // 1_000_000}M",
        flush=True,
    )

    # Compile numba functions
    print("Compiling...", flush=True)
    d = np.random.randn(N, 3)
    cnt(d)
    mec(d)
    rnd(d)
    sa_chain(d, 100, 0.01, 0.01, 0)
    adaptive_sa(d, 100, 0.01, 0)
    quantized_adaptive_sa(d, 100, 0.001, 0)
    polish_q(d, 100, 0)
    find_boundary_mask(d)
    structured_kick(d, 0.01, 0)
    boundary_sa(d, 100, 0)
    coord_descent(d, 100, 0)
    tempered_geometric_search(d, 100, 0.001, 0.003, 0)
    print(f"Ready ({time.time() - t0:.1f}s)\n", flush=True)

    # Init pool
    pool_init()

    # Load previous best
    global_best_score = 1e9
    global_best = None
    score, scs = pool_read_best()
    if score is not None:
        global_best_score = score
        global_best = scs.copy()
        pool_write_best(score, scs)
        print(f"Starting from: {global_best_score:.6f}\n", flush=True)

    # Initialize diverse pool
    diverse_pool = []
    if global_best is not None:
        diverse_pool.append((global_best_score, global_best.copy()))

    # Initialize worker seeds
    def make_seed(worker_type, worker_idx):
        """Create initial configuration for a worker."""
        seed_base = int(time.time() * 1000) + worker_idx * 1009 + round_num * 37
        if worker_type == TYPE_A or worker_type == TYPE_F or worker_type == TYPE_Q:
            # Cold/quantized: archive-biased, but preserve contact structure while kicking
            if diverse_pool and len(diverse_pool) > 1:
                di = int(np.random.random() ** 2 * len(diverse_pool))
                di = min(di, len(diverse_pool) - 1)
                _, base = diverse_pool[di]
                return structured_kick(base, 0.002 + worker_idx * 0.002, seed_base)
            if global_best is not None:
                return structured_kick(
                    global_best, 0.002 + worker_idx * 0.002, seed_base + 17
                )
            return HOT_CONFIGS[worker_idx % len(HOT_CONFIGS)].copy()
        elif worker_type == TYPE_B:
            # Hot: fresh flower configs with strong structured kicks
            cfg = HOT_CONFIGS[worker_idx % len(HOT_CONFIGS)].copy()
            return structured_kick(cfg, 0.05 + worker_idx * 0.03, seed_base + 29)
        elif worker_type == TYPE_C or worker_type == TYPE_I:
            # Contact movers / PT: archive-biased structural perturbations
            if diverse_pool:
                di = int(np.random.random() ** 2 * len(diverse_pool))
                di = min(di, len(diverse_pool) - 1)
                _, base = diverse_pool[di]
                return structured_kick(base, 0.01 + worker_idx * 0.004, seed_base + 41)
            if global_best is not None:
                return structured_kick(
                    global_best, 0.01 + worker_idx * 0.004, seed_base + 53
                )
            return HOT_CONFIGS[worker_idx % len(HOT_CONFIGS)].copy()
        elif worker_type == TYPE_D:
            # Elastic: best with moderate structural kick
            if global_best is not None:
                return structured_kick(global_best, 0.03, seed_base + 67)
            return HOT_CONFIGS[worker_idx % len(HOT_CONFIGS)].copy()
        elif worker_type == TYPE_E or worker_type == TYPE_R:
            # Constructor: fresh configs
            return HOT_CONFIGS[worker_idx % len(HOT_CONFIGS)].copy()
        elif worker_type == TYPE_G:
            # Boundary SA: start from best unperturbed (must be overlap-free)
            if global_best is not None:
                return global_best.copy()
            return HOT_CONFIGS[worker_idx % len(HOT_CONFIGS)].copy()
        elif worker_type == TYPE_H:
            # Coordinate descent: start from best (no perturbation for systematic search)
            if global_best is not None:
                return global_best.copy()
            return HOT_CONFIGS[worker_idx % len(HOT_CONFIGS)].copy()
        return np.random.randn(N, 3) * 2

    round_num = 0
    rounds_since_improve = 0
    last_improve_time = t0
    last_phase = 0
    pool = Pool(N_WORKERS)

    # Graceful shutdown
    shutdown = [False]

    def signal_handler(sig, frame):
        shutdown[0] = True
        print("\nShutdown requested...", flush=True)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while not shutdown[0] and time.time() - t0 < TIME_LIMIT - min(
            300, TIME_LIMIT * 0.05
        ):
            round_num += 1
            rounds_since_improve += 1
            elapsed_hours = (time.time() - t0) / 3600

            # Phase management
            phase = get_phase(elapsed_hours)
            if phase != last_phase:
                print(
                    f"\n  === PHASE {phase} ({'EXPLORE' if phase == 1 else 'EXPLOIT' if phase == 2 else 'POLISH'}) ===\n",
                    flush=True,
                )
                last_phase = phase

            assignments = get_worker_assignments(phase, N_WORKERS)

            # Double iterations when stuck
            cur_iters = ITERS_PER_ROUND
            if rounds_since_improve > 20:
                cur_iters = min(ITERS_PER_ROUND * 2, 16_000_000)

            # Build worker args
            args = []
            type_counter = {}
            for i in range(N_WORKERS):
                wtype, temp, step = assignments[i]
                type_counter[wtype] = type_counter.get(wtype, 0) + 1
                worker_idx = type_counter[wtype] - 1

                # Seed selection
                scs = make_seed(wtype, worker_idx)
                seed = round_num * 100000 + i * 10000 + int(time.time()) % 10000

                args.append((i, wtype, scs.flatten(), temp, step, seed, cur_iters))

            # Dispatch
            results = pool.map(run_worker, args)

            # Collect results
            type_bests = {}
            for wid, wtype, best_flat, best_score, final_flat, final_score in results:
                tname = TYPE_NAMES[wtype]
                if tname not in type_bests or best_score < type_bests[tname]:
                    type_bests[tname] = best_score

                if best_score < global_best_score:
                    scs = best_flat.reshape(N, 3)
                    r = rnd(scs)
                    if cnt(r) == 0:
                        rs = mec(r)
                        if rs < global_best_score:
                            global_best_score = rs
                            global_best = r.copy()
                            pool_write_best(rs, r)
                            rounds_since_improve = 0
                            last_improve_time = time.time()
                            hours = (time.time() - t0) / 3600
                            print(
                                f"  *** NEW BEST: {rs:.6f} (worker={wid}, type={tname}, R{round_num}, {hours:.1f}h) ***",
                                flush=True,
                            )

                # Update diverse pool
                if best_score < 1e8:
                    diverse_pool = update_diverse_pool(
                        best_score, best_flat.reshape(N, 3), diverse_pool
                    )

            # Write diverse pool to disk
            for i, (ds, dscs) in enumerate(diverse_pool):
                pool_write_diverse(i, ds, dscs)

            # Chaos pulse: frequent archive-based reseeding (every N stale rounds)
            if (
                rounds_since_improve >= CATASTROPHE_STALL_ROUNDS
                and rounds_since_improve % CATASTROPHE_STALL_ROUNDS == 0
            ):
                pulse_num = rounds_since_improve // CATASTROPHE_STALL_ROUNDS
                print(
                    f"  << CHAOS PULSE #{pulse_num} (stale={rounds_since_improve}) >>",
                    flush=True,
                )
                # Inject new diverse seeds from best with varying kick sizes
                if global_best is not None:
                    for kick_i in range(min(5, DIVERSE_SLOTS - len(diverse_pool))):
                        kick_size = 0.03 + kick_i * 0.04 + pulse_num * 0.02
                        kicked = structured_kick(
                            global_best,
                            kick_size,
                            round_num * 10_000 + pulse_num * 100 + kick_i,
                        )
                        if kick_i % 2 == 1:
                            kicked = seed_reflect(
                                kicked,
                                ((round_num + kick_i) % 628) / 100.0,
                                round_num + kick_i,
                            )
                        if cnt(kicked) == 0:
                            ks = mec(kicked)
                            diverse_pool = update_diverse_pool(ks, kicked, diverse_pool)
                # Perturb some archive entries (exponential bias toward worse ones)
                n_to_kick = min(5, len(diverse_pool))
                for ki in range(n_to_kick):
                    # Bias toward worse entries (higher index = worse score)
                    di = (
                        len(diverse_pool)
                        - 1
                        - int(abs(np.random.randn()) * len(diverse_pool) * 0.3)
                    )
                    di = max(1, min(di, len(diverse_pool) - 1))  # Never kick best
                    ds, dscs = diverse_pool[di]
                    kicked = structured_kick(
                        dscs,
                        0.04 + pulse_num * 0.015,
                        round_num * 20_000 + di * 131 + ki,
                    )
                    if cnt(kicked) == 0:
                        ks = mec(kicked)
                        diverse_pool[di] = (ks, kicked)
                diverse_pool.sort(key=lambda x: x[0])

            # Status line
            elapsed = time.time() - t0
            remaining = TIME_LIMIT - elapsed
            type_str = " ".join(
                f"{k}={v:.4f}" if v < 1e8 else f"{k}=---"
                for k, v in sorted(type_bests.items())
            )
            print(
                f"  R{round_num}: best={global_best_score:.6f} [{type_str}] "
                f"diverse={len(diverse_pool)} stale={rounds_since_improve} "
                f"phase={phase} {remaining / 3600:.1f}h left",
                flush=True,
            )

            # Periodic polish (every 50 rounds)
            if round_num % 50 == 0 and global_best is not None:
                print("  Polishing...", flush=True)
                c2, cs2, _, _ = coord_descent(global_best, 300_000, round_num)
                candidate = c2 if cs2 < global_best_score else global_best
                q2, qs2, _, _ = quantized_adaptive_sa(
                    rnd(candidate), 500_000, 0.0008, round_num + 123
                )
                if qs2 < global_best_score:
                    candidate = q2
                r = rnd(candidate)
                if cnt(r) == 0:
                    r2, rs2 = polish_q(r, 2_000_000, round_num)
                    best_polish = rs2
                    best_candidate = r2
                    if qs2 < best_polish:
                        best_polish = qs2
                        best_candidate = q2
                    if best_polish < global_best_score:
                        global_best_score = best_polish
                        global_best = best_candidate
                        pool_write_best(best_polish, best_candidate)
                        rounds_since_improve = 0
                        last_improve_time = time.time()
                        print(f"  Polish improved: {best_polish:.6f}", flush=True)

            # Checkpoint every 10 rounds
            if round_num % 10 == 0:
                pool_write_checkpoint(
                    {
                        "round": round_num,
                        "best_score": float(global_best_score),
                        "phase": phase,
                        "elapsed_hours": elapsed_hours,
                        "stall_rounds": rounds_since_improve,
                        "diverse_count": len(diverse_pool),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                )

    except Exception as e:
        print(f"\nERROR: {e}", flush=True)
    finally:
        pool.terminate()
        pool.join()

    # Final polish
    if global_best is not None:
        print("\nFinal polish (5M)...", flush=True)
        c_best, c_score, _, _ = coord_descent(global_best, 800_000, 99997)
        candidate = c_best if c_score < global_best_score else global_best
        q_best, q_score, _, _ = quantized_adaptive_sa(rnd(candidate), 1_000_000, 0.0006, 99998)
        if q_score < global_best_score:
            candidate = q_best
        r = rnd(candidate)
        if cnt(r) == 0:
            r, rs = polish_q(r, 5_000_000, 99999)
            final_best = rs
            final_scs = r
            if q_score < final_best:
                final_best = q_score
                final_scs = q_best
            if final_best < global_best_score:
                global_best_score = final_best
                global_best = final_scs
                pool_write_best(final_best, final_scs)
        print(f"  {global_best_score:.6f}", flush=True)

    print(
        f"\nFINAL: {global_best_score:.6f} ({(time.time() - t0) / 3600:.1f}h)",
        flush=True,
    )


if __name__ == "__main__":
    main()
