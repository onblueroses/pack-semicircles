"""Analytic support-function-based separation gap for unit semicircles.

Theta convention matches geom.ov(): a semicircle at (x, y, t) has its
material (arc + flat) on the +t side, i.e., the flat edge is perpendicular
to (cos t, sin t) passing through (x, y), and the arc extends in the +t
direction. Two semicircles don't overlap iff their signed separation is
>= 0.

Since semicircles are convex, the signed separation between two of them is
exactly || c_A - c_B || - h_A(u_A) - h_B(u_B), where h is the support
function and u_A, u_B are unit vectors pointing toward the other center.
This matches geom.ov() sign (modulo the 1e-6 threshold) because convex
non-overlap has a unique characterization via separating distance.

Computational cost: ~20 flops per pair. Numba-jit for speed.
"""

import math
import numba as nb

__all__ = [
    "h_semicircle",
    "gap_ss",
    "gap_sb",
    "overlap_penalty_sq",
    "smooth_mec_sq",
    "smooth_penalty_objective",
]


@nb.njit(cache=True, inline="always")
def h_semicircle(nx, ny, t):
    """Support function of unit semicircle at origin, orientation t.

    h(n) = max over material of (point . n). Material: {(x,y): x^2+y^2 <= 1,
    and (x,y).t_hat >= 0} where t_hat = (cos t, sin t).

    In the local frame (t_hat along +x): h = |n| if n_x_local >= 0, else |n_y_local|.
    C^1 continuous everywhere except at the origin (n = 0, which never arises here).
    """
    ct = math.cos(t)
    st = math.sin(t)
    nx_l = nx * ct + ny * st
    ny_l = -nx * st + ny * ct
    if nx_l >= 0.0:
        return math.sqrt(nx * nx + ny * ny)
    return abs(ny_l)


@nb.njit(cache=True)
def gap_ss_collinear(x1, y1, t1, x2, y2, t2, eps=1e-9):
    """Lower-bound separation in the center-connecting direction only.
    Fast but NOT tight for semicircles with directional flat edges - use gap_ss
    for the true separation maximized over all directions."""
    dx = x1 - x2
    dy = y1 - y2
    d = math.sqrt(dx * dx + dy * dy + eps * eps)
    ux = dx / d
    uy = dy / d
    h1 = h_semicircle(-ux, -uy, t1)
    h2 = h_semicircle(ux, uy, t2)
    return d - h1 - h2


@nb.njit(cache=True)
def _sep_at(dx, dy, t1, t2, alpha):
    nx = math.cos(alpha)
    ny = math.sin(alpha)
    ha = h_semicircle(nx, ny, t1)
    hb = h_semicircle(-nx, -ny, t2)
    return dx * nx + dy * ny - ha - hb


@nb.njit(cache=True)
def gap_ss(x1, y1, t1, x2, y2, t2, n_samples=128):
    """Signed separation distance between two unit semicircles.
    sep = max over unit n of [(c2 - c1) . n - h_1(n) - h_2(-n)].
    >= 0 no overlap, < 0 penetration depth along the optimal direction.

    Strategy: 128-grid to locate coarse argmax, then golden-section refinement
    within the best grid bracket (20 iters -> ~2e-5 rad precision). Also
    evaluates at 4 piecewise kink angles t_i +/- pi/2 where h switches
    between its arc and flat branches."""
    dx = x2 - x1
    dy = y2 - y1
    best = -1.0e18
    best_alpha = 0.0
    # Grid sweep
    step = 2.0 * math.pi / n_samples
    for k in range(n_samples):
        alpha = step * k
        val = _sep_at(dx, dy, t1, t2, alpha)
        if val > best:
            best = val
            best_alpha = alpha
    # Kink angles
    for kink_t in (
        t1 + math.pi / 2,
        t1 - math.pi / 2,
        t2 + math.pi / 2,
        t2 - math.pi / 2,
    ):
        val = _sep_at(dx, dy, t1, t2, kink_t)
        if val > best:
            best = val
            best_alpha = kink_t
    # Golden-section refinement within +/- step around best_alpha
    lo = best_alpha - step
    hi = best_alpha + step
    # Phi = (sqrt(5) - 1) / 2 ~= 0.618
    phi_inv = 0.6180339887498949
    c = hi - phi_inv * (hi - lo)
    d = lo + phi_inv * (hi - lo)
    fc = _sep_at(dx, dy, t1, t2, c)
    fd = _sep_at(dx, dy, t1, t2, d)
    for _ in range(25):
        if fc > fd:
            hi = d
            d = c
            fd = fc
            c = hi - phi_inv * (hi - lo)
            fc = _sep_at(dx, dy, t1, t2, c)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + phi_inv * (hi - lo)
            fd = _sep_at(dx, dy, t1, t2, d)
    refined = max(fc, fd)
    if refined > best:
        best = refined
    return best


@nb.njit(cache=True)
def gap_ss_smooth(x1, y1, t1, x2, y2, t2, n_samples=64, T=0.01):
    """Smooth soft-max version of gap_ss for gradient-based optimization.
    log-sum-exp over n_samples directions. T -> 0 recovers gap_ss."""
    dx = x2 - x1
    dy = y2 - y1
    best = -1.0e18
    for k in range(n_samples):
        alpha = 2.0 * math.pi * k / n_samples
        nx = math.cos(alpha)
        ny = math.sin(alpha)
        ha = h_semicircle(nx, ny, t1)
        hb = h_semicircle(-nx, -ny, t2)
        val = dx * nx + dy * ny - ha - hb
        if val > best:
            best = val
    acc = 0.0
    for k in range(n_samples):
        alpha = 2.0 * math.pi * k / n_samples
        nx = math.cos(alpha)
        ny = math.sin(alpha)
        ha = h_semicircle(nx, ny, t1)
        hb = h_semicircle(-nx, -ny, t2)
        val = dx * nx + dy * ny - ha - hb
        acc += math.exp((val - best) / T)
    return best + T * math.log(acc)


@nb.njit(cache=True)
def gap_sb(x, y, t, cx, cy, R, eps=1e-9):
    """Signed gap from semicircle (x,y,t) to MEC boundary at (cx,cy) radius R.
    >= 0 means fully inside, < 0 means extending outside by |gap|."""
    dx = x - cx
    dy = y - cy
    d2 = dx * dx + dy * dy
    d = math.sqrt(d2 + eps * eps)
    ux = dx / d
    uy = dy / d
    h = h_semicircle(ux, uy, t)
    return R - d - h


@nb.njit(cache=True)
def overlap_penalty_sq(scs, margin=0.0):
    """Sum of squared pairwise penetration depths below `margin`.
    margin=0 gives raw overlap penalty; margin>0 creates a safety buffer
    (penalty active whenever gap < margin, so optimizer maintains slack)."""
    N = scs.shape[0]
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            g = gap_ss(
                scs[i, 0],
                scs[i, 1],
                scs[i, 2],
                scs[j, 0],
                scs[j, 1],
                scs[j, 2],
            )
            deficit = margin - g
            if deficit > 0.0:
                pen += deficit * deficit
    return pen


@nb.njit(cache=True)
def smooth_mec_sq(scs, cx, cy, T):
    """Log-sum-exp upper bound on max squared distance from (cx,cy) to any
    boundary sample point. Smooth in scs and (cx,cy). As T -> 0, approaches
    exact R^2.

    Samples: 2 flat endpoints + 8 arc samples per semicircle (150 points total).
    """
    N = scs.shape[0]
    # Stabilized log-sum-exp: first compute max, then sum exp((x-max)/T)
    # First pass: find max squared distance
    r2_max = 0.0
    for i in range(N):
        x = scs[i, 0]
        y = scs[i, 1]
        t = scs[i, 2]
        ct = math.cos(t)
        st = math.sin(t)
        # Flat endpoint A: (x + sin t, y - cos t), B: (x - sin t, y + cos t)
        for sign in (1.0, -1.0):
            px = x + sign * st
            py = y - sign * ct
            r2 = (px - cx) ** 2 + (py - cy) ** 2
            if r2 > r2_max:
                r2_max = r2
        # Arc samples: angles t + s*pi/2 for s in [-1, 1] across 8 samples
        for s in range(8):
            a = t - math.pi / 2 + math.pi * s / 7
            px = x + math.cos(a)
            py = y + math.sin(a)
            r2 = (px - cx) ** 2 + (py - cy) ** 2
            if r2 > r2_max:
                r2_max = r2
    # Second pass: sum exp((r2 - r2_max)/T)
    acc = 0.0
    for i in range(N):
        x = scs[i, 0]
        y = scs[i, 1]
        t = scs[i, 2]
        ct = math.cos(t)
        st = math.sin(t)
        for sign in (1.0, -1.0):
            px = x + sign * st
            py = y - sign * ct
            r2 = (px - cx) ** 2 + (py - cy) ** 2
            acc += math.exp((r2 - r2_max) / T)
        for s in range(8):
            a = t - math.pi / 2 + math.pi * s / 7
            px = x + math.cos(a)
            py = y + math.sin(a)
            r2 = (px - cx) ** 2 + (py - cy) ** 2
            acc += math.exp((r2 - r2_max) / T)
    return r2_max + T * math.log(acc)


@nb.njit(cache=True)
def smooth_penalty_objective(x, T_mec, lam):
    """Packed objective for L-BFGS-B. x is a flat array of length 3N + 2:
    [x_0,y_0,t_0, ..., x_{N-1},y_{N-1},t_{N-1}, cx, cy].
    Returns R2_smooth + lam * overlap_penalty_sq."""
    N = (x.shape[0] - 2) // 3
    scs = x[: 3 * N].reshape(N, 3)
    cx = x[3 * N]
    cy = x[3 * N + 1]
    r2 = smooth_mec_sq(scs, cx, cy, T_mec)
    pen = overlap_penalty_sq(scs)
    return r2 + lam * pen
