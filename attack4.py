"""Attack 4: active-set Newton on extracted contact graph.

Per plan_final.md (v10 + implementation-detail fixes), with one simplification
to ship tonight: features live inline here rather than in a separate package.
Splittable later if needed.

Variable layout x (length 48):
  x[3i:3i+3] = (x_i, y_i, t_i) for i in 0..14
  x[45], x[46] = (cx, cy)  (MEC center)
  x[47] = R                (objective)

Gauge: fix (cx, cy) at seed value + fix t_0 at seed value (3 equality
constraints). Using explicit equality constraints rather than Bounds lb=ub
to avoid trust-constr's pin-via-bounds Lagrangian issue.

Thresholds (hardcoded per plan_final.md -- calibrated to incumbent):
  tol_seed = 1e-4  (active-set seeding)
  guard_eps = 1e-6 (sector guards)
  case_eps = 1e-8  (degenerate branch)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
from scipy.optimize import NonlinearConstraint, LinearConstraint

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import geom  # noqa: E402
import gap as gapmod  # noqa: E402
import common  # noqa: E402

N = geom.N
NX = 3 * N + 3  # 48

TOL_SEED = 1e-4
GUARD_EPS = 1e-6
CASE_EPS = 1e-8


# ---------- packing ----------


def pack(scs: np.ndarray, cx: float, cy: float, R: float) -> np.ndarray:
    return np.concatenate([scs.reshape(-1), [cx, cy, R]])


def unpack(x: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    scs = x[: 3 * N].reshape(N, 3).copy()
    cx = float(x[3 * N])
    cy = float(x[3 * N + 1])
    R = float(x[3 * N + 2])
    return scs, cx, cy, R


# ---------- feature types ----------


@dataclass
class Feature:
    kind: str  # 'ARC_ARC', 'FLAT_ARC', 'ENDPOINT_ARC', 'CONTAIN_ARC', 'CONTAIN_FLAT'
    i: int
    j: int = -1  # partner index (pair features) or -1 (containment)
    eps: int = 0  # endpoint sign ±1 (ENDPOINT_ARC, CONTAIN_FLAT); flat-arc σ (FLAT_ARC)
    note: str = ""  # human-readable classification info

    def residual(self, x: np.ndarray) -> float:
        return _residual(self, x)

    def jac(self, x: np.ndarray) -> np.ndarray:
        return _jac(self, x)


def _sc(x, i):
    return x[3 * i], x[3 * i + 1], x[3 * i + 2]


def contain_gap_exact(xi, yi, ti, cx, cy, R):
    """Exact signed containment gap: R - (farthest point of semicircle from MEC
    center). Arc side: d + 1. Flat side: sqrt(d² + 1 + 2d|u·τ|) where τ = (sin t,
    -cos t). Returns R - far; nonneg => fully inside."""
    dx = xi - cx
    dy = yi - cy
    d = math.hypot(dx, dy)
    if d < 1e-12:
        return R - 1.0
    ct = math.cos(ti)
    st = math.sin(ti)
    un = (dx * ct + dy * st) / d
    ut = (dx * st - dy * ct) / d
    far = d + 1.0 if un >= 0.0 else math.sqrt(d * d + 1.0 + 2.0 * d * abs(ut))
    return R - far


_MARGIN_PAIR = 0.0  # added to pairwise contact distances
_MARGIN_CONTAIN = 0.0  # subtracted from containment radius


def _residual(f: Feature, x: np.ndarray) -> float:
    scs, cx, cy, R = unpack(x)
    mp = _MARGIN_PAIR
    mc = _MARGIN_CONTAIN
    if f.kind == "ARC_ARC":
        xi, yi, _ = scs[f.i]
        xj, yj, _ = scs[f.j]
        T = 2.0 + mp
        return (xi - xj) ** 2 + (yi - yj) ** 2 - T * T
    if f.kind == "FLAT_ARC":
        # i's flat-interior touches j's arc. σ ∈ {+1, -1} encodes which side
        # of the flat-line j is on. r = (c_j - c_i) · n_i + σ*(1 + mp).
        xi, yi, ti = scs[f.i]
        xj, yj, _ = scs[f.j]
        return (xj - xi) * math.cos(ti) + (yj - yi) * math.sin(ti) + f.eps * (1.0 + mp)
    if f.kind == "ENDPOINT_ARC":
        # Endpoint of i (sign eps) touches arc of j.
        xi, yi, ti = scs[f.i]
        xj, yj, _ = scs[f.j]
        e = f.eps
        ax = xi + e * math.sin(ti) - xj
        ay = yi - e * math.cos(ti) - yj
        T = 1.0 + mp
        return ax * ax + ay * ay - T * T
    if f.kind == "CONTAIN_ARC":
        xi, yi, _ = scs[f.i]
        dx = xi - cx
        dy = yi - cy
        T = R - 1.0 - mc
        return dx * dx + dy * dy - T * T
    if f.kind == "CONTAIN_FLAT":
        xi, yi, ti = scs[f.i]
        e = f.eps
        bx = xi + e * math.sin(ti) - cx
        by = yi - e * math.cos(ti) - cy
        T = R - mc
        return bx * bx + by * by - T * T
    raise ValueError(f"unknown kind {f.kind}")


def _jac(f: Feature, x: np.ndarray) -> np.ndarray:
    scs, cx, cy, R = unpack(x)
    J = np.zeros(NX)
    if f.kind == "ARC_ARC":
        xi, yi, _ = scs[f.i]
        xj, yj, _ = scs[f.j]
        dx = xi - xj
        dy = yi - yj
        J[3 * f.i + 0] = 2 * dx
        J[3 * f.i + 1] = 2 * dy
        J[3 * f.j + 0] = -2 * dx
        J[3 * f.j + 1] = -2 * dy
        return J
    if f.kind == "FLAT_ARC":
        xi, yi, ti = scs[f.i]
        xj, yj, _ = scs[f.j]
        ct = math.cos(ti)
        st = math.sin(ti)
        J[3 * f.i + 0] = -ct
        J[3 * f.i + 1] = -st
        J[3 * f.i + 2] = -(xj - xi) * st + (yj - yi) * ct
        J[3 * f.j + 0] = ct
        J[3 * f.j + 1] = st
        return J
    if f.kind == "ENDPOINT_ARC":
        xi, yi, ti = scs[f.i]
        xj, yj, _ = scs[f.j]
        e = f.eps
        ct = math.cos(ti)
        st = math.sin(ti)
        ax = xi + e * st - xj
        ay = yi - e * ct - yj
        J[3 * f.i + 0] = 2 * ax
        J[3 * f.i + 1] = 2 * ay
        J[3 * f.i + 2] = 2 * e * (ax * ct + ay * st)
        J[3 * f.j + 0] = -2 * ax
        J[3 * f.j + 1] = -2 * ay
        return J
    if f.kind == "CONTAIN_ARC":
        xi, yi, _ = scs[f.i]
        dx = xi - cx
        dy = yi - cy
        T = R - 1.0 - _MARGIN_CONTAIN
        J[3 * f.i + 0] = 2 * dx
        J[3 * f.i + 1] = 2 * dy
        J[3 * N + 0] = -2 * dx
        J[3 * N + 1] = -2 * dy
        J[3 * N + 2] = -2 * T
        return J
    if f.kind == "CONTAIN_FLAT":
        xi, yi, ti = scs[f.i]
        e = f.eps
        ct = math.cos(ti)
        st = math.sin(ti)
        bx = xi + e * st - cx
        by = yi - e * ct - cy
        T = R - _MARGIN_CONTAIN
        J[3 * f.i + 0] = 2 * bx
        J[3 * f.i + 1] = 2 * by
        J[3 * f.i + 2] = 2 * e * (bx * ct + by * st)
        J[3 * N + 0] = -2 * bx
        J[3 * N + 1] = -2 * by
        J[3 * N + 2] = -2 * T
        return J
    raise ValueError(f"unknown kind {f.kind}")


# ---------- classifier ----------


def _try_classify_pair(scs: np.ndarray, i: int, j: int) -> List[Feature]:
    """Return list of locked features for this pair. Empty => flat-flat witness
    or below-threshold; caller still monitors but no equality constraint."""
    xi, yi, ti = scs[i]
    xj, yj, tj = scs[j]
    n_i = np.array([math.cos(ti), math.sin(ti)])
    n_j = np.array([math.cos(tj), math.sin(tj)])
    tau_i = np.array([math.sin(ti), -math.cos(ti)])
    d = np.array([xj - xi, yj - yi])
    dnorm = float(np.linalg.norm(d))
    if dnorm < CASE_EPS:
        return []  # coincident, skip
    u = d / dnorm

    feats: List[Feature] = []

    # 1. ARC_ARC: center distance ≈ 2, both arc-hemispheres
    if (
        abs(dnorm - 2.0) < 2 * TOL_SEED
        and u @ n_i > GUARD_EPS
        and (-u) @ n_j > GUARD_EPS
    ):
        feats.append(Feature("ARC_ARC", i=i, j=j, note=f"d={dnorm:.6f}"))
        return feats

    # 2. FLAT_ARC: i-flat vs j-arc. Signed perp (c_j - c_i)·n_i ≈ ±1 with α interior.
    p_i = float(d @ n_i)
    alpha_i = float(d @ tau_i)
    if abs(alpha_i) < 1.0 - CASE_EPS:
        for sigma in (+1, -1):
            if abs(p_i + sigma) < TOL_SEED and (sigma * (n_i @ n_j) < -GUARD_EPS):
                # sigma = +1 means p_i ≈ -1 (j below i-flat), need n_j · n_i > 0
                # so check sigma * (n_i·n_j) < 0, i.e., n_i·n_j has opposite sign of sigma
                # Actually: sigma=+1, need n_j toward +n_i (i.e., n_i·n_j > 0)
                # sigma=-1, need n_j toward -n_i (n_i·n_j < 0)
                # So: sigma * (n_i·n_j) > guard (positive). Fix:
                pass
    # Clean FLAT_ARC check:
    if abs(alpha_i) < 1.0 - CASE_EPS:
        if abs(p_i + 1.0) < TOL_SEED and (n_i @ n_j) > GUARD_EPS:
            feats.append(
                Feature(
                    "FLAT_ARC", i=i, j=j, eps=+1, note=f"p={p_i:.6f} a={alpha_i:.3f}"
                )
            )
            return feats
        if abs(p_i - 1.0) < TOL_SEED and (n_i @ n_j) < -GUARD_EPS:
            feats.append(
                Feature(
                    "FLAT_ARC", i=i, j=j, eps=-1, note=f"p={p_i:.6f} a={alpha_i:.3f}"
                )
            )
            return feats

    # 3. Swapped: j-flat vs i-arc. Signed perp (c_i - c_j)·n_j ≈ ±1.
    d_rev = -d
    p_j = float(d_rev @ n_j)
    alpha_j = float(d_rev @ np.array([math.sin(tj), -math.cos(tj)]))
    if abs(alpha_j) < 1.0 - CASE_EPS:
        if abs(p_j + 1.0) < TOL_SEED and (n_j @ n_i) > GUARD_EPS:
            feats.append(
                Feature(
                    "FLAT_ARC",
                    i=j,
                    j=i,
                    eps=+1,
                    note=f"p={p_j:.6f} a={alpha_j:.3f} swap",
                )
            )
            return feats
        if abs(p_j - 1.0) < TOL_SEED and (n_j @ n_i) < -GUARD_EPS:
            feats.append(
                Feature(
                    "FLAT_ARC",
                    i=j,
                    j=i,
                    eps=-1,
                    note=f"p={p_j:.6f} a={alpha_j:.3f} swap",
                )
            )
            return feats

    # 4. ENDPOINT_ARC: i-endpoint touches j-arc, or j-endpoint touches i-arc.
    # i-endpoint eps ∈ {+1,-1}, check ||ep - c_j|| ≈ 1 AND ep is on arc-side of j.
    for a, b, na, tau_a, ta in (
        (i, j, n_i, tau_i, ti),
        (j, i, n_j, np.array([math.sin(tj), -math.cos(tj)]), tj),
    ):
        xa, ya, _ = scs[a]
        xb, yb, _ = scs[b]
        nb = np.array([math.cos(scs[b, 2]), math.sin(scs[b, 2])])
        for e in (+1, -1):
            ep = np.array([xa + e * math.sin(ta), ya - e * math.cos(ta)])
            delta = ep - np.array([xb, yb])
            rnorm = float(np.linalg.norm(delta))
            if rnorm < CASE_EPS:
                continue
            u_ep = delta / rnorm
            if abs(rnorm - 1.0) < TOL_SEED and (u_ep @ nb) > GUARD_EPS:
                feats.append(
                    Feature(
                        "ENDPOINT_ARC", i=a, j=b, eps=e, note=f"ep_dist={rnorm:.6f}"
                    )
                )
                return feats

    # 5. FLAT_FLAT_WITNESS — don't lock
    return []


def _try_classify_containment(
    scs: np.ndarray, cx: float, cy: float, R: float, i: int
) -> Optional[Feature]:
    xi, yi, ti = scs[i]
    dx = xi - cx
    dy = yi - cy
    d_i = math.hypot(dx, dy)
    if d_i < CASE_EPS:
        return None
    u = np.array([dx / d_i, dy / d_i])
    n_i = np.array([math.cos(ti), math.sin(ti)])
    if (u @ n_i) > GUARD_EPS:
        # Arc side: farthest point is c_i + u (but plan uses c_i + n_i times...).
        # Actually farthest from (cx,cy) on semicircle is: support maximizer
        # in direction u. For u·n>0 arc-hemisphere the support is c_i + u*1
        # only if u is within arc. If u·n > 0 strictly, support in direction u
        # is c_i + u (yes). So farthest dist = d_i + 1, R = d_i + 1.
        # Residual = d_i^2 - (R-1)^2 = 0.
        return Feature("CONTAIN_ARC", i=i, note=f"u.n={u @ n_i:.4f}")
    if (u @ n_i) < -GUARD_EPS:
        # Flat side: farthest point is an endpoint c_i + e*tau_i.
        # Pick endpoint farther from (cx,cy): sign e = sign(u · tau_i).
        tau = np.array([math.sin(ti), -math.cos(ti)])
        e = 1 if (u @ tau) > 0 else -1
        return Feature("CONTAIN_FLAT", i=i, eps=e, note=f"u.n={u @ n_i:.4f} e={e}")
    return None  # kink; skip lock


def extract_contact_graph(
    scs: np.ndarray, cx: float, cy: float, R: float
) -> Tuple[List[Feature], List[Tuple[int, int]]]:
    """Return (equality_features, near_active_but_unlocked_pairs).
    Unlocked pairs (flat-flat witnesses) get inequality constraints in Stage A.
    """
    feats: List[Feature] = []
    witnesses: List[Tuple[int, int]] = []
    # Containment: use EXACT farthest-boundary distance (gap_sb-based approximation
    # silently misses flat-side contacts). For semicircle i=2 on incumbent,
    # approximate gap_sb gives +0.174744 but exact gap is -1.24e-4 (active).
    for i in range(N):
        g_exact = contain_gap_exact(scs[i, 0], scs[i, 1], scs[i, 2], cx, cy, R)
        if abs(g_exact) < TOL_SEED:
            cf = _try_classify_containment(scs, cx, cy, R, i)
            if cf is not None:
                feats.append(cf)
    # Pairwise
    for i in range(N):
        for j in range(i + 1, N):
            g = gapmod.gap_ss(
                scs[i, 0], scs[i, 1], scs[i, 2], scs[j, 0], scs[j, 1], scs[j, 2]
            )
            if g < TOL_SEED:
                new_feats = _try_classify_pair(scs, i, j)
                if new_feats:
                    feats.extend(new_feats)
                else:
                    # Near-active but unclassified -> flat-flat witness
                    witnesses.append((i, j))
    return feats, witnesses


# ---------- problem assembly ----------


def objective(x: np.ndarray) -> float:
    return float(x[-1])


def objective_grad(x: np.ndarray) -> np.ndarray:
    g = np.zeros_like(x)
    g[-1] = 1.0
    return g


def build_constraints(
    feats: List[Feature],
    witnesses: List[Tuple[int, int]],
    seed_cx: float,
    seed_cy: float,
    seed_t0: float,
):
    """Build scipy constraints:
       - Equality: one per feature (residual = 0)
       - Gauge: cx=seed_cx, cy=seed_cy, t_0=seed_t0 (linear)
       - Inequality: pairwise gap_ss >= 0 for non-locked pairs (monitor/safety)
                     containment >= 0 for non-locked boundary
    We only enforce equality for locked features + inequality for safety
    (applied as SciPy NonlinearConstraint over ALL pairs using exact gap_ss
    with FD Jac as safety margin). To keep things clean, start WITHOUT safety
    inequalities — rely on the classifier to capture all active contacts. Can
    add safety inequalities later if rounding fails feasibility."""

    def eq_residuals(x, feats=feats):
        return np.array([f.residual(x) for f in feats])

    def eq_jac(x, feats=feats):
        return np.vstack([f.jac(x) for f in feats]) if feats else np.zeros((0, NX))

    eq_con = NonlinearConstraint(eq_residuals, 0.0, 0.0, jac=eq_jac)  # type: ignore[arg-type]

    # Gauge as LinearConstraint (rows pick out cx, cy, t_0)
    A_gauge = np.zeros((3, NX))
    A_gauge[0, 3 * N + 0] = 1.0  # cx
    A_gauge[1, 3 * N + 1] = 1.0  # cy
    A_gauge[2, 3 * 0 + 2] = 1.0  # t_0
    b_gauge = np.array([seed_cx, seed_cy, seed_t0])
    gauge_con = LinearConstraint(A_gauge, b_gauge, b_gauge)  # type: ignore[arg-type]

    # Safety inequalities: pairwise non-overlap (exact gap_ss, FD Jac).
    # We include these to prevent the solver from wandering into infeasible regions.
    def pair_all(x):
        scs, _, _, _ = unpack(x)
        out = np.empty(N * (N - 1) // 2)
        k = 0
        for i in range(N):
            for j in range(i + 1, N):
                out[k] = gapmod.gap_ss(
                    scs[i, 0], scs[i, 1], scs[i, 2], scs[j, 0], scs[j, 1], scs[j, 2]
                )
                k += 1
        return out

    def cont_all(x):
        scs, cx, cy, R = unpack(x)
        return np.array(
            [
                gapmod.gap_sb(scs[i, 0], scs[i, 1], scs[i, 2], cx, cy, R)
                for i in range(N)
            ]
        )

    # Witness inequalities: flat-flat pairs that are active but not equality-
    # locked per plan. Prevents them from opening/closing arbitrarily.
    if witnesses:

        def witness_gaps(x, witnesses=witnesses):
            scs, _, _, _ = unpack(x)
            return np.array(
                [
                    gapmod.gap_ss(
                        scs[i, 0],
                        scs[i, 1],
                        scs[i, 2],
                        scs[j, 0],
                        scs[j, 1],
                        scs[j, 2],
                    )
                    for (i, j) in witnesses
                ]
            )

        witness_con = NonlinearConstraint(witness_gaps, 0.0, np.inf, jac="2-point")
    else:
        witness_con = None

    # Containment safety only for semicircles NOT already pinned by equality
    # (avoids equality + inequality on same quantity).
    pinned_contain = {f.i for f in feats if f.kind in ("CONTAIN_ARC", "CONTAIN_FLAT")}
    non_pinned = [i for i in range(N) if i not in pinned_contain]

    def cont_nonpinned(x, ids=non_pinned):
        scs, cx, cy, R = unpack(x)
        return np.array(
            [gapmod.gap_sb(scs[i, 0], scs[i, 1], scs[i, 2], cx, cy, R) for i in ids]
        )

    cont_ineq = NonlinearConstraint(cont_nonpinned, 0.0, np.inf, jac="2-point")

    cons = [eq_con, gauge_con, cont_ineq]
    if witness_con is not None:
        cons.append(witness_con)
    return cons


# ---------- Stage A ----------


def _ineq_fd_grad(fn, x, eps=1e-6):
    """Central-difference gradient of scalar fn at x."""
    n = x.size
    g = np.zeros(n)
    for k in range(n):
        xp = x.copy()
        xp[k] += eps
        xm = x.copy()
        xm[k] -= eps
        g[k] = (fn(xp) - fn(xm)) / (2 * eps)
    return g


def stage_a(
    scs0: np.ndarray,
    cx0: float,
    cy0: float,
    R0: float,
    feats: List[Feature],
    witnesses: Optional[List[Tuple[int, int]]] = None,
    maxiter: int = 60,
    verbose: int = 0,
    trust: float = 0.01,
    trust_min: float = 1e-6,
    margin_pair: float = 0.0,
    margin_contain: float = 0.0,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    global _MARGIN_PAIR, _MARGIN_CONTAIN
    _MARGIN_PAIR = margin_pair
    _MARGIN_CONTAIN = margin_contain
    """LP-driven active-set SQP.

    At each iterate x_k:
      1. Build J_eq (analytic) from features + gauge rows.
      2. Build J_ineq (FD) for currently-active inequalities:
         - witness pairs (gap_ss) with slack < witness_active_tol
         - containment (gap_sb) for non-pinned semicircles with slack < cont_active_tol
      3. Solve LP:
            min  δR  = c·δ                (c = e_R)
            s.t. J_eq δ = -r_eq           (equality correction)
                 J_ineq δ ≥ -s_ineq       (inequality maintenance)
                 -trust ≤ δ_k ≤ trust
      4. Line-search α ∈ [0, 1] on merit R + μ · max(0, -gap) with μ = 100.
      5. If no progress: shrink trust; if trust < trust_min: stop.
    """
    from scipy.optimize import linprog

    if log is None:
        log = lambda s: None  # noqa: E731

    witnesses = witnesses or []
    pinned = {f.i for f in feats if f.kind in ("CONTAIN_ARC", "CONTAIN_FLAT")}
    nonpinned = [i for i in range(N) if i not in pinned]

    x = pack(scs0, cx0, cy0, R0)
    _seed_cx, _seed_cy, seed_t0 = cx0, cy0, float(scs0[0, 2])

    # Gauge jacobian (linear, constant)
    # Gauge: fix (x_0, y_0, t_0) at seed. (cx, cy) free so solver minimizes
    # R at the true MEC center. Fixing cx,cy at seed trapped the center.
    seed_x0 = float(scs0[0, 0])
    seed_y0 = float(scs0[0, 1])
    Jg = np.zeros((3, NX))
    Jg[0, 3 * 0 + 0] = 1.0
    Jg[1, 3 * 0 + 1] = 1.0
    Jg[2, 3 * 0 + 2] = 1.0

    def gauge_res(x):
        return np.array([x[0] - seed_x0, x[1] - seed_y0, x[2] - seed_t0])

    def wit_val(x, i, j):
        s = x[: 3 * N].reshape(N, 3)
        return gapmod.gap_ss(s[i, 0], s[i, 1], s[i, 2], s[j, 0], s[j, 1], s[j, 2])

    def cont_val(x, i):
        s = x[: 3 * N].reshape(N, 3)
        return contain_true(s[i, 0], s[i, 1], s[i, 2], x[-3], x[-2], x[-1])

    def contain_true(xi, yi, ti, cx_v, cy_v, R_v):
        """Correct containment: R - max_{p in semi} ||p - (cx,cy)||.
        Semicircle is convex; farthest point is on flat endpoint or arc apex
        in direction of c_mec->c_i. Check both candidates analytically."""
        dx = xi - cx_v
        dy = yi - cy_v
        d = math.hypot(dx, dy)
        if d < 1e-12:
            return R_v - 1.0
        ct = math.cos(ti)
        st = math.sin(ti)
        # u . n_i
        un = (dx * ct + dy * st) / d
        if un > 0:
            # arc side: farthest is c_i + u (arc apex in direction u), dist = d + 1
            return R_v - (d + 1.0)
        # Flat side: farthest is a flat endpoint. Check both ±tau.
        # tau = (sin t, -cos t); endpoint c_i + ε*tau.
        ep1_dx = dx + st
        ep1_dy = dy - ct
        ep2_dx = dx - st
        ep2_dy = dy + ct
        d1 = math.hypot(ep1_dx, ep1_dy)
        d2 = math.hypot(ep2_dx, ep2_dy)
        # Also check arc apex: at arc-hemisphere edge closest to u direction.
        # For un ≤ 0 the max of 2d cos(θ-φ) over θ in [-π/2,π/2] is 2d cos(|φ|-π/2)
        # = 2d sin(|φ|) where φ = atan2(u_y_local, u_x_local). u_x_local = un,
        # u_y_local = (dx st - dy ct)/d (rotation into body frame). sin|φ| = |u_y|/1.
        u_y_local = abs((-dx * st + dy * ct)) / d
        # Arc farthest-from-c_mec² = 1 + d² + 2d u_y_local (when un<0)
        d_arc = math.sqrt(1.0 + d * d + 2.0 * d * u_y_local)
        return R_v - max(d1, d2, d_arc)

    def total_violation(x):
        v = 0.0
        v += float(np.abs([f.residual(x) for f in feats]).sum()) if feats else 0.0
        v += float(np.abs(gauge_res(x)).sum())
        s = x[: 3 * N].reshape(N, 3)
        cxv = x[-3]
        cyv = x[-2]
        Rv = x[-1]
        for i in range(N):
            for j in range(i + 1, N):
                g = gapmod.gap_ss(s[i, 0], s[i, 1], s[i, 2], s[j, 0], s[j, 1], s[j, 2])
                if g < 0:
                    v += -g
            cg = contain_true(s[i, 0], s[i, 1], s[i, 2], cxv, cyv, Rv)
            if cg < 0:
                v += -cg
        return v

    last_R = float(x[-1])
    for it in range(maxiter):
        # Build J_eq with current analytic features
        if feats:
            J_feats = np.vstack([f.jac(x) for f in feats])
            r_feats = np.array([f.residual(x) for f in feats])
        else:
            J_feats = np.zeros((0, NX))
            r_feats = np.zeros(0)
        J_eq = np.vstack([J_feats, Jg])
        r_eq = np.concatenate([r_feats, gauge_res(x)])

        # Include ALL non-pinned containment + ALL witnesses unconditionally.
        # Previously active-set filtering missed constraints that became active
        # mid-step, letting R shrink below the true MEC.
        A_rows = []
        b_rows = []
        for i, j in witnesses:
            v = wit_val(x, i, j)
            g = _ineq_fd_grad(lambda xx, i=i, j=j: wit_val(xx, i, j), x)
            A_rows.append(-g)
            b_rows.append(v)
        for i in nonpinned:
            v = cont_val(x, i)
            g = _ineq_fd_grad(lambda xx, i=i: cont_val(xx, i), x)
            A_rows.append(-g)
            b_rows.append(v)
        # Also: pairwise inequalities for the non-witness non-feature pairs.
        # These are pairs with initially large gaps but may close during descent.
        # Track them lazily: include only if their gap is currently < 0.05.
        scs_now = x[: 3 * N].reshape(N, 3)
        feat_pairs = {tuple(sorted((f.i, f.j))) for f in feats if f.j != -1}
        wit_set = {tuple(sorted(p)) for p in witnesses}
        for i in range(N):
            for j in range(i + 1, N):
                key = (i, j)
                if key in feat_pairs or key in wit_set:
                    continue
                v = gapmod.gap_ss(
                    scs_now[i, 0],
                    scs_now[i, 1],
                    scs_now[i, 2],
                    scs_now[j, 0],
                    scs_now[j, 1],
                    scs_now[j, 2],
                )
                if v < 0.05:
                    g = _ineq_fd_grad(lambda xx, i=i, j=j: wit_val(xx, i, j), x)
                    A_rows.append(-g)
                    b_rows.append(v)

        A_ub = np.array(A_rows) if A_rows else np.zeros((0, NX))
        b_ub = np.array(b_rows) if b_rows else np.zeros(0)

        c = np.zeros(NX)
        c[-1] = 1.0
        bounds = [(-trust, trust)] * NX

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=J_eq,
            b_eq=-r_eq,
            bounds=bounds,
            method="highs",
        )

        if res.status != 0 or res.x is None:
            log(f"  it={it} LP infeasible ({res.message[:50]}); trust->{trust / 2:.2e}")
            trust *= 0.5
            if trust < trust_min:
                break
            continue

        delta = res.x
        dR = float(delta[-1])

        # Line search with merit function
        mu = 100.0

        def f_merit(x, _mu=mu):
            return float(x[-1]) + _mu * total_violation(x)

        f0 = f_merit(x)
        best_alpha = 0.0
        best_f = f0
        for alpha in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
            xt = x + alpha * delta
            ft = f_merit(xt)
            if ft < best_f - 1e-12:
                best_f = ft
                best_alpha = alpha
                break  # take first descent

        if best_alpha == 0.0:
            log(
                f"  it={it} no α reduces merit (dR_lp={dR:+.3e}); trust->{trust / 2:.2e}"
            )
            trust *= 0.5
            if trust < trust_min:
                break
            continue

        x = x + best_alpha * delta
        viol = total_violation(x)
        log(
            f"  it={it} α={best_alpha:.3f} dR_lp={dR:+.3e} "
            f"R={x[-1]:.9f} viol={viol:.3e} trust={trust:.2e} nAct={len(A_rows)}"
        )
        # Expand trust on good steps
        if best_alpha >= 0.5:
            trust = min(trust * 1.5, 0.02)

        if abs(last_R - x[-1]) < 1e-11 and viol < 1e-8:
            log(f"  converged at it={it}")
            break
        last_R = x[-1]

    scs_out = x[: 3 * N].reshape(N, 3).copy()
    cx_out = float(x[3 * N])
    cy_out = float(x[3 * N + 1])
    R_out = float(x[-1])
    del scs_out, cx_out, cy_out, R_out  # computed for clarity, returned via unpack
    scs, cx, cy, R = unpack(x)
    return dict(
        scs=scs,
        cx=cx,
        cy=cy,
        R=R,
        success=True,
        message="lp-sqp",
        nit=0,
        constr_violation=float(total_violation(x)),
        x=x,
        lagrangian=None,
    )


def validate(scs: np.ndarray) -> Optional[float]:
    rounded = geom.rnd(np.asarray(scs, dtype=np.float64))
    if geom.cnt(rounded) > 0:
        return None
    return float(geom.mec(rounded))


# ---------- Stage B: topology branching ----------


def _make_witness_feature(i: int, j: int, sigma: int) -> Feature:
    """Witness pair locked as FLAT_ARC-like equality. sigma ∈ {+1,-1} picks
    which side. In a PARALLEL_SHARED_LINE witness, both orientations are
    near-anti-parallel, so this lock may or may not hold geometrically."""
    return Feature("FLAT_ARC", i=i, j=j, eps=sigma, note="witness-lock")


def enumerate_moves(
    feats: List[Feature],
    witnesses: List[Tuple[int, int]],
    scs: np.ndarray,
    include_pairs: bool = False,
) -> List[dict]:
    """Return a list of topology-perturbation descriptors. Each item has
    'kind' + 'apply(feats, witnesses)' returning new (feats', witnesses').
    """
    moves = []
    # DEACTIVATE each feature
    for k in range(len(feats)):
        moves.append(
            dict(
                kind=f"deact[{k}]:{feats[k].kind}({feats[k].i},{feats[k].j})",
                _type="deact",
                idx=k,
            )
        )
    # ACTIVATE each witness as equality (both signs)
    for idx, (i, j) in enumerate(witnesses):
        for sigma in (+1, -1):
            moves.append(
                dict(
                    kind=f"act_wit[{idx}]=({i},{j}):σ={sigma}",
                    _type="act_wit",
                    idx=idx,
                    sigma=sigma,
                )
            )
    # TOGGLE non-pinned boundary
    pinned = {f.i for f in feats if f.kind in ("CONTAIN_ARC", "CONTAIN_FLAT")}
    for i in range(N):
        if i in pinned:
            continue
        moves.append(dict(kind=f"add_contain_arc[{i}]", _type="add_ca", i=i))
        moves.append(
            dict(kind=f"add_contain_flat[{i}]:+1", _type="add_cf", i=i, eps=+1)
        )
        moves.append(
            dict(kind=f"add_contain_flat[{i}]:-1", _type="add_cf", i=i, eps=-1)
        )
    # PAIR-DEACTIVATE: remove two features simultaneously (expands search
    # from single-move to 2-move). Only when include_pairs=True.
    if include_pairs:
        for k1 in range(len(feats)):
            for k2 in range(k1 + 1, len(feats)):
                moves.append(
                    dict(
                        kind=f"deact2[{k1},{k2}]",
                        _type="deact2",
                        idx1=k1,
                        idx2=k2,
                    )
                )
    return moves


def apply_move(move: dict, feats: List[Feature], witnesses: List[Tuple[int, int]]):
    feats = list(feats)
    witnesses = list(witnesses)
    t = move["_type"]
    if t == "deact":
        feats.pop(move["idx"])
    elif t == "deact2":
        # Remove higher-idx first to preserve idx1
        k1, k2 = move["idx1"], move["idx2"]
        feats.pop(k2)
        feats.pop(k1)
    elif t == "act_wit":
        i, j = witnesses[move["idx"]]
        feats.append(_make_witness_feature(i, j, move["sigma"]))
        witnesses.pop(move["idx"])
    elif t == "add_ca":
        feats.append(Feature("CONTAIN_ARC", i=move["i"]))
    elif t == "add_cf":
        feats.append(Feature("CONTAIN_FLAT", i=move["i"], eps=move["eps"]))
    return feats, witnesses


def stage_b_solve_move(args):
    """Worker: apply move, run Stage A, return result dict."""
    (
        move,
        scs0,
        cx0,
        cy0,
        R0,
        base_feats,
        base_wit,
        margin_pair,
        margin_contain,
        maxiter,
    ) = args
    try:
        feats, wit = apply_move(move, base_feats, base_wit)
        res = stage_a(
            scs0,
            cx0,
            cy0,
            R0,
            feats,
            wit,
            maxiter=maxiter,
            trust=0.005,
            margin_pair=margin_pair,
            margin_contain=margin_contain,
        )
        rnd = geom.rnd(res["scs"])
        c = int(geom.cnt(rnd))
        mec_r = float(geom.mec(rnd)) if c == 0 else None
        return dict(
            kind=move["kind"],
            move=move,
            unrounded_R=float(res["R"]),
            rounded_R=mec_r,
            overlaps=c,
            viol=float(res["constr_violation"]),
            scs=res["scs"],
        )
    except Exception as e:
        return dict(kind=move["kind"], move=move, error=repr(e))


def stage_b(
    root_scs: np.ndarray,
    root_cx: float,
    root_cy: float,
    root_R: float,
    base_feats: List[Feature],
    base_wit: List[Tuple[int, int]],
    current_best_R: float,
    workers: int = 1,
    maxiter: int = 300,
    margin_pair: float = 2e-6,
    margin_contain: float = 2e-6,
    include_pairs: bool = False,
    log: Callable[[str], None] = print,
) -> List[dict]:
    """Run all single-move perturbations in parallel. Return sorted results."""
    import multiprocessing as mp

    moves = enumerate_moves(base_feats, base_wit, root_scs, include_pairs=include_pairs)
    log(f"[stage_b] enumerating {len(moves)} moves (workers={workers})")
    tasks = [
        (
            m,
            root_scs,
            root_cx,
            root_cy,
            root_R,
            base_feats,
            base_wit,
            margin_pair,
            margin_contain,
            maxiter,
        )
        for m in moves
    ]
    results = []
    if workers > 1:
        with mp.Pool(workers) as pool:
            for r in pool.imap_unordered(stage_b_solve_move, tasks, chunksize=1):
                results.append(r)
                rr = r.get("rounded_R")
                if isinstance(rr, (int, float)) and rr < current_best_R - 1e-9:
                    log(f"  IMPROVE {r['kind']}: rounded={rr:.9f}")
    else:
        for t in tasks:
            r = stage_b_solve_move(t)
            results.append(r)
            rr = r.get("rounded_R")
            if isinstance(rr, (int, float)) and rr < current_best_R - 1e-9:
                log(f"  IMPROVE {r['kind']}: rounded={rr:.9f}")

    def _sort_key(r: dict) -> float:
        v = r.get("rounded_R")
        if v is None or not isinstance(v, (int, float)):
            return 999.0
        return float(v)

    results.sort(key=_sort_key)
    return results


# ---------- driver ----------


def load_seed(path: str) -> Tuple[np.ndarray, float, float, float]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "scs" in data:
        scs = np.asarray(data["scs"], dtype=np.float64)
    elif isinstance(data, list):
        scs = np.asarray([[s["x"], s["y"], s["theta"]] for s in data], dtype=np.float64)
    else:
        raise ValueError(f"Unrecognized seed format: {path}")
    cx, cy, R = geom.mec_info(scs)
    return scs, float(cx), float(cy), float(R)


def feat_to_dict(f: Feature) -> dict:
    return asdict(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="pool/best.json")
    ap.add_argument("--seed", action="append", default=[])
    ap.add_argument("--hours", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument(
        "--dry-classify",
        action="store_true",
        help="Classify incumbent and exit (no solve).",
    )
    ap.add_argument("--stage-b", action="store_true", help="Run Stage B branching")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--depth", type=int, default=2, help="Stage B hill-climb depth")
    ap.add_argument("--b-maxiter", type=int, default=300)
    ap.add_argument("--margin", type=float, default=2e-6)
    ap.add_argument(
        "--pairs",
        action="store_true",
        help="Include 2-feature-deactivation moves (O(f²))",
    )
    args = ap.parse_args()

    if args.out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.label}" if args.label else ""
        args.out = f"runs/attack4_{ts}{tag}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        common.log_line("attack4", msg)
        with open(out_dir / "driver.log", "a") as f:
            f.write(msg + "\n")

    log(f"out_dir={out_dir} root={args.root} seeds={args.seed}")

    scs0, cx0, cy0, R0 = load_seed(args.root)
    log(f"root R={R0:.9f}")
    feats, witnesses = extract_contact_graph(scs0, cx0, cy0, R0)
    log(f"extracted {len(feats)} features, {len(witnesses)} witnesses")
    counts = {}
    for f in feats:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    log(f"breakdown: {counts}")
    for f in feats:
        log(f"  {f.kind} i={f.i} j={f.j} eps={f.eps} {f.note}")
    for i, j in witnesses:
        log(f"  WITNESS pair=({i},{j})")

    common.write_json_atomic(
        str(out_dir / "contact_graph.json"),
        dict(
            seed=args.root,
            R=R0,
            features=[feat_to_dict(f) for f in feats],
            witnesses=witnesses,
        ),
    )
    if args.dry_classify:
        return

    log("Stage A solve")
    res_a = stage_a(scs0, cx0, cy0, R0, feats, witnesses, maxiter=500, verbose=0)
    log(
        f"Stage A: R={res_a['R']:.9f} success={res_a['success']} "
        f"violation={res_a['constr_violation']:.2e} nit={res_a['nit']}"
    )
    rounded = validate(res_a["scs"])
    log(f"Stage A rounded R = {rounded}")
    common.write_json_atomic(
        str(out_dir / "stage_a.json"),
        dict(
            R=res_a["R"],
            rounded_R=rounded,
            scs=res_a["scs"].tolist(),
            cx=res_a["cx"],
            cy=res_a["cy"],
            violation=res_a["constr_violation"],
            nit=res_a["nit"],
            success=res_a["success"],
            message=res_a["message"],
        ),
    )

    if not args.stage_b:
        return

    # ---------- Stage B: topology hill-climb ----------
    t0 = time.time()
    current_best_R = rounded if rounded is not None else R0
    current_scs = res_a["scs"]
    current_cx = res_a["cx"]
    current_cy = res_a["cy"]
    current_R = res_a["R"]
    current_feats = feats
    current_wit = witnesses
    history = [dict(stage="A", R=current_best_R)]

    for depth in range(1, args.depth + 1):
        if time.time() - t0 > args.hours * 3600:
            log(f"budget exhausted before depth {depth}")
            break
        log(
            f"Stage B depth {depth}: from R={current_best_R:.9f}, "
            f"feats={len(current_feats)} wit={len(current_wit)}"
        )
        results = stage_b(
            current_scs,
            current_cx,
            current_cy,
            current_R,
            current_feats,
            current_wit,
            current_best_R,
            workers=args.workers,
            maxiter=args.b_maxiter,
            margin_pair=args.margin,
            margin_contain=args.margin,
            include_pairs=args.pairs,
            log=log,
        )
        # Save all results
        common.write_json_atomic(
            str(out_dir / f"stage_b_depth{depth}.json"),
            dict(
                depth=depth,
                parent_R=current_best_R,
                results=[
                    dict(
                        kind=r["kind"],
                        rounded_R=r.get("rounded_R"),
                        unrounded_R=r.get("unrounded_R"),
                        overlaps=r.get("overlaps"),
                        viol=r.get("viol"),
                    )
                    for r in results
                ],
            ),
        )
        # Pick best improver
        best = None
        for r in results:
            if (
                r.get("rounded_R") is not None
                and r["rounded_R"] < current_best_R - 1e-9
            ):
                best = r
                break
        if best is None:
            log(f"  no depth-{depth} improvement; top 3 non-improving rounded R:")
            for r in results[:3]:
                log(f"    {r.get('rounded_R')} {r['kind']}")
            break
        log(
            f"  BEST: {best['kind']} rounded={best['rounded_R']:.9f} "
            f"(Δ={current_best_R - best['rounded_R']:.2e})"
        )
        # Accept and re-extract active set from the new optimum
        current_best_R = best["rounded_R"]
        current_scs = best["scs"]
        rnd = geom.rnd(current_scs)
        current_cx, current_cy, current_R = geom.mec_info(rnd)
        current_feats, current_wit = extract_contact_graph(
            np.asarray(rnd, dtype=np.float64),
            float(current_cx),
            float(current_cy),
            float(current_R),
        )
        history.append(dict(stage=f"B_d{depth}", R=current_best_R, move=best["kind"]))
        # Save snapshot of new champion
        out = dict(
            score=float(current_best_R),
            scs=rnd.tolist(),
            solution=[
                {
                    "x": float(rnd[i, 0]),
                    "y": float(rnd[i, 1]),
                    "theta": float(rnd[i, 2]),
                }
                for i in range(N)
            ],
        )
        common.write_json_atomic(str(out_dir / f"champion_d{depth}.json"), out)

    log(f"FINAL best rounded R = {current_best_R:.9f}")
    log(f"history: {history}")
    common.write_json_atomic(
        str(out_dir / "summary.json"), dict(best_R=current_best_R, history=history)
    )


if __name__ == "__main__":
    main()
