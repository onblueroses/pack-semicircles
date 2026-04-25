"""Perturbation library for MBH outer loop.

Provides 4 core moves (flip-one, reseat-interior, rim-swap, rotate-cluster)
and a weighted damage estimator D. Returns a PerturbResult for the driver to
feed through overlap-resolve → Stage A → archive.

Deferred to post-pilot (per plan v4 M6-lean): vacancy-reseat, multi-scale
composition, translate-pair, 45°-θ-pulse. The 4 moves below cover the Lai
2023/2025 ILS vocabulary sufficient for first basin-break signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.optimize as opt

import attack4
import gap as gapmod
import geom


# D weights (plan v4 §1.1): instrument during pilot, re-tune from histograms.
D_BOUNDARY = 2.0
D_INTERIOR = 1.0
D_WITNESS = 0.5
# Empirically calibrated on incumbent (2.9486936795) after plan v4 spec-correct
# weighted_d landed: flip_one D~1-2, reseat/rim D~2-5, rotate_cluster D~5-10.
# Widened from plan v4's [3.0, 6.0] so flip_one isn't silently dead; re-tune
# from pilot histograms across a diverse seed set.
D_BAND = (2.0, 7.0)

# Contact tolerance for weighted-D graph comparison. Matches attack4.TOL_SEED.
TOL_CONTACT = attack4.TOL_SEED


# ---------- damage measurement ----------


def _contacts_and_witnesses(
    scs: np.ndarray, cx: float, cy: float, R: float
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]:
    """Extract (pair_contacts, flat_flat_witnesses, boundary_pieces) for a
    config measured against a FIXED circle (cx, cy, R). Caller is responsible
    for supplying the same circle when comparing two configs.

    Uses attack4.extract_contact_graph for exact classification; the returned
    sets are *unordered* pair keys and boundary piece indices.
    """
    feats, witnesses = attack4.extract_contact_graph(scs, cx, cy, R)
    pair_contacts: set[tuple[int, int]] = set()
    boundary: set[int] = set()
    for f in feats:
        if f.kind in ("CONTAIN_ARC", "CONTAIN_FLAT"):
            boundary.add(f.i)
        else:
            a, b = sorted((f.i, f.j))
            pair_contacts.add((a, b))
    witness_set: set[tuple[int, int]] = set()
    for w in witnesses:
        a, b = sorted(w)
        witness_set.add((a, b))
    return pair_contacts, witness_set, boundary


def weighted_d(scs_before: np.ndarray, scs_after: np.ndarray, R: float) -> float:
    """Damage estimate per plan v4 §1.1.

        D = 2.0 · #boundary_broken
          + 1.0 · #interior_pair_broken  (neither endpoint on boundary in BEFORE)
          + 0.5 · #witnesses_broken

    Both configs are scored against the SAME (cx, cy, R) taken from the
    before-state MEC. Otherwise the boundary set "moves" with the circle and
    every piece that was on the old boundary looks artificially broken.
    """
    cx, cy, _ = geom.mec_info(scs_before)
    pb, wb, bb = _contacts_and_witnesses(scs_before, cx, cy, R)
    pa, wa, ba = _contacts_and_witnesses(scs_after, cx, cy, R)
    broken_boundary = len(bb - ba)
    # Interior pair = pair contact in BEFORE whose endpoints are both non-
    # boundary (per plan v4). Rim-involving pair breaks are counted via the
    # boundary term already.
    broken = pb - pa
    broken_interior_pairs = sum(1 for a, b in broken if a not in bb and b not in bb)
    broken_witnesses = len(wb - wa)
    return (
        D_BOUNDARY * broken_boundary
        + D_INTERIOR * broken_interior_pairs
        + D_WITNESS * broken_witnesses
    )


# ---------- result ----------


@dataclass
class PerturbResult:
    scs: np.ndarray
    move_type: str
    D: float
    metadata: dict = field(default_factory=dict)


# ---------- move: flip-one ----------


def move_flip_one(scs: np.ndarray, R: float, rng: np.random.Generator) -> PerturbResult:
    """Rotate one piece by π about its own position. Piece selection weighted
    by pair-contact degree (interior high-degree pieces preferred)."""
    cx, cy, _ = geom.mec_info(scs)
    pair_contacts, _, _ = _contacts_and_witnesses(scs, cx, cy, R)
    degree = np.ones(geom.N)  # +1 baseline so isolated pieces can still be picked
    for a, b in pair_contacts:
        degree[a] += 1
        degree[b] += 1
    probs = degree / degree.sum()
    i = int(rng.choice(geom.N, p=probs))
    out = scs.copy()
    out[i, 2] = (out[i, 2] + math.pi) % (2 * math.pi)
    D = weighted_d(scs, out, R)
    return PerturbResult(out, "flip_one", D, {"piece": i, "degree": float(degree[i])})


# ---------- move: reseat-interior ----------


def move_reseat_interior(
    scs: np.ndarray, R: float, rng: np.random.Generator
) -> PerturbResult:
    """Relocate an interior piece to an unoccupied rim position with outward θ.

    Rim grid: 72 candidate θ's on the MEC; pick the one maximising min pairwise
    gap to existing pieces (excluding the relocated one).
    """
    cx, cy, mec_r = geom.mec_info(scs)
    mask, _, _, _ = geom.find_boundary_mask(scs)
    interior_pieces = [i for i in range(geom.N) if not mask[i]]
    if not interior_pieces:
        # Fallback: pick any piece with least boundary contact
        interior_pieces = list(range(geom.N))
    i = int(rng.choice(interior_pieces))

    # Rim candidate positions at radius (mec_r - 1.0 - epsilon) so the arc tip
    # lands at the MEC boundary when θ points outward.
    arm = max(mec_r - 1.0, 0.1)
    grid = rng.permutation(72)
    best_scs = None
    best_slack = -float("inf")
    for k in grid[:24]:  # limit evaluations
        phi = 2 * math.pi * k / 72
        qx = cx + arm * math.cos(phi)
        qy = cy + arm * math.sin(phi)
        t_outward = phi  # unit normal = +arc direction
        candidate = scs.copy()
        candidate[i] = (qx, qy, t_outward)
        # Min slack to other pieces
        slacks = []
        for j in range(geom.N):
            if j == i:
                continue
            g = gapmod.gap_ss(qx, qy, t_outward, scs[j, 0], scs[j, 1], scs[j, 2])
            slacks.append(g)
        if not slacks:
            continue
        m = min(slacks)
        if m > best_slack:
            best_slack = m
            best_scs = candidate
    if best_scs is None:
        best_scs = scs.copy()
    D = weighted_d(scs, best_scs, R)
    return PerturbResult(
        best_scs,
        "reseat_interior",
        D,
        {"piece": i, "min_slack": float(best_slack)},
    )


# ---------- move: rim-swap ----------


def move_rim_swap(scs: np.ndarray, R: float, rng: np.random.Generator) -> PerturbResult:
    """Swap positions of two boundary pieces. Orientations stay, so the newly
    placed pieces will generally violate containment until resolve; that's
    intentional — the resolve step reseats them into feasible rim angles."""
    mask, _, _, _ = geom.find_boundary_mask(scs)
    rim = [i for i in range(geom.N) if mask[i]]
    if len(rim) < 2:
        # Fallback: swap any two pieces
        rim = list(range(geom.N))
    i, j = rng.choice(rim, size=2, replace=False)
    out = scs.copy()
    out[i, 0], out[j, 0] = out[j, 0], out[i, 0]
    out[i, 1], out[j, 1] = out[j, 1], out[i, 1]
    D = weighted_d(scs, out, R)
    return PerturbResult(out, "rim_swap", D, {"pieces": [int(i), int(j)]})


# ---------- move: rotate-cluster ----------


def _contact_neighbors(scs: np.ndarray, R: float) -> list[set[int]]:
    """Adjacency from pair contacts (computed at the config's own MEC)."""
    cx, cy, _ = geom.mec_info(scs)
    pair_contacts, _, _ = _contacts_and_witnesses(scs, cx, cy, R)
    adj: list[set[int]] = [set() for _ in range(geom.N)]
    for a, b in pair_contacts:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def move_rotate_cluster(
    scs: np.ndarray,
    R: float,
    rng: np.random.Generator,
    k_cluster: int | None = None,
) -> PerturbResult:
    """Pick a connected cluster of 3-4 pieces and rotate them about their
    centroid by α ∈ [π/6, π/3]."""
    if k_cluster is None:
        k_cluster = int(rng.integers(3, 5))
    adj = _contact_neighbors(scs, R)
    # BFS from a random seed piece
    seed = int(rng.integers(geom.N))
    cluster = [seed]
    frontier = list(adj[seed])
    rng.shuffle(frontier)
    while frontier and len(cluster) < k_cluster:
        n = frontier.pop()
        if n in cluster:
            continue
        cluster.append(n)
        for m in adj[n]:
            if m not in cluster and m not in frontier:
                frontier.append(m)
        rng.shuffle(frontier)
    if len(cluster) < 2:
        # Isolated piece — fall back to flip-one to avoid a no-op.
        return move_flip_one(scs, R, rng)
    cluster_arr = np.array(cluster, dtype=int)
    center = scs[cluster_arr, :2].mean(axis=0)
    alpha = float(rng.uniform(math.pi / 6, math.pi / 3)) * float(rng.choice([-1, 1]))
    c, s = math.cos(alpha), math.sin(alpha)
    out = scs.copy()
    for i in cluster:
        dx = out[i, 0] - center[0]
        dy = out[i, 1] - center[1]
        out[i, 0] = center[0] + c * dx - s * dy
        out[i, 1] = center[1] + s * dx + c * dy
        out[i, 2] = (out[i, 2] + alpha) % (2 * math.pi)
    D = weighted_d(scs, out, R)
    return PerturbResult(
        out,
        "rotate_cluster",
        D,
        {"cluster": [int(x) for x in cluster], "alpha": alpha},
    )


# ---------- move: contact-surgery (topology-changing) ----------
#
# Codex review of the earlier vacancy_reseat design demonstrated:
#  - blind grid sampling is alias-dead on the incumbent (0/15 hits)
#  - negative-slack seeds + global resolve scramble topology (post-D ~25)
# Replacement design: enumerate candidate poses from EXACT analytic tangency
# constructions seeded by existing piece centers and the rim, then run a LOCAL
# resolve that only moves the placed piece + its neighbors. The driver detects
# metadata["pre_resolved"]=True and skips its own global resolve, going straight
# to Stage A.

CONTACT_SURGERY_TOL = -0.05  # min_slack ≥ this; -0.05 catches ~7/15 pieces on incumbent
CONTACT_SURGERY_R_RESOLVE_SLACK = 0.10  # local-resolve at R + this (mirrors driver)
CONTACT_SURGERY_THETA_GRID = 36  # number of orientations probed per (qx, qy)


def _circle_circle_intersect(
    cx1: float, cy1: float, r1: float, cx2: float, cy2: float, r2: float
) -> list[tuple[float, float]]:
    """Up to 2 intersection points of two circles. Empty if disjoint, contained,
    or coincident. Tangent case returns 1 point."""
    d = math.hypot(cx2 - cx1, cy2 - cy1)
    if d > r1 + r2 or d < abs(r1 - r2) or d < 1e-12:
        return []
    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h2 = r1 * r1 - a * a
    if h2 < -1e-12:
        return []
    h = math.sqrt(max(h2, 0.0))
    px = cx1 + a * (cx2 - cx1) / d
    py = cy1 + a * (cy2 - cy1) / d
    if h < 1e-9:
        return [(px, py)]
    ox = -h * (cy2 - cy1) / d
    oy = h * (cx2 - cx1) / d
    return [(px + ox, py + oy), (px - ox, py - oy)]


def _surgery_local_penalty(
    x_active: np.ndarray,
    scs_full: np.ndarray,
    active_idx: np.ndarray,
    cx: float,
    cy: float,
    R: float,
) -> float:
    """Sum-of-squares penalty over violated pair gaps + containments.
    Active piece coords are taken from x_active; frozen pieces from scs_full."""
    s = scs_full.copy()
    s[active_idx] = x_active.reshape(len(active_idx), 3)
    pen = 0.0
    for i in range(geom.N):
        for j in range(i + 1, geom.N):
            g = gapmod.gap_ss(s[i, 0], s[i, 1], s[i, 2], s[j, 0], s[j, 1], s[j, 2])
            if g < 0:
                pen += g * g
        cg = attack4.contain_gap_exact(s[i, 0], s[i, 1], s[i, 2], cx, cy, R)
        if cg < 0:
            pen += cg * cg
    return pen


def _local_resolve(
    scs_init: np.ndarray,
    active_idx: set[int],
    cx: float,
    cy: float,
    R: float,
    maxiter: int = 500,
) -> tuple[np.ndarray, float]:
    """L-BFGS-B over only the active pieces' (x, y, theta) — frozen pieces stay
    put. Returns (resolved_scs, final_penalty)."""
    if not active_idx:
        return scs_init.copy(), 0.0
    active_arr = np.array(sorted(active_idx), dtype=int)
    x0 = scs_init[active_arr].reshape(-1).copy()
    res = opt.minimize(
        _surgery_local_penalty,
        x0,
        args=(scs_init, active_arr, cx, cy, R),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-9},
    )
    out = scs_init.copy()
    out[active_arr] = res.x.reshape(len(active_arr), 3)
    return out, float(res.fun)


def move_contact_surgery(
    scs: np.ndarray, R: float, rng: np.random.Generator
) -> PerturbResult:
    """Topology-changing move: remove a piece, place it at an analytic-tangency
    pose seeded by existing geometry, run a LOCAL resolve over piece + neighbors.

    Returns metadata["pre_resolved"]=True so the driver skips its global resolve.
    On feasibility failure, returns scs unchanged with metadata["fallback"] set
    (move_type stays "contact_surgery" so scheduler accounts truthfully)."""
    cx, cy, mec_r = geom.mec_info(scs)
    p = int(rng.integers(geom.N))
    others_list = [j for j in range(geom.N) if j != p]

    # 1. Generate candidate (qx, qy, anchors) from arc-arc × arc-arc and
    #    arc-arc × arc-rim tangency constructions. anchors = list of piece
    #    indices (and possibly "rim") that determined this candidate; used
    #    later to build the active set for local resolve.
    cands: list[tuple[float, float, list]] = []
    for ii, i in enumerate(others_list):
        for j in others_list[ii + 1 :]:
            for x, y in _circle_circle_intersect(
                scs[i, 0], scs[i, 1], 2.0, scs[j, 0], scs[j, 1], 2.0
            ):
                cands.append((x, y, [i, j]))
    for i in others_list:
        for x, y in _circle_circle_intersect(
            scs[i, 0], scs[i, 1], 2.0, cx, cy, mec_r - 1.0
        ):
            cands.append((x, y, [i, "rim"]))

    # Dedup by L2 (within 1e-4) and keep only candidates inside MEC interior.
    uniq: list[tuple[float, float, list]] = []
    for c in cands:
        if any(math.hypot(c[0] - u[0], c[1] - u[1]) < 1e-4 for u in uniq):
            continue
        uniq.append(c)
    inside = [c for c in uniq if math.hypot(c[0] - cx, c[1] - cy) <= mec_r - 1.0 + 1e-3]

    # 2. For each candidate, sweep theta grid; pick best (qx, qy, theta, anchors)
    #    by min_slack against ALL other pieces + rim containment.
    best: tuple[float, tuple[float, float, float], list] | None = None
    for qx, qy, anchors in inside:
        for k in range(CONTACT_SURGERY_THETA_GRID):
            theta = 2.0 * math.pi * k / CONTACT_SURGERY_THETA_GRID
            ms = float("inf")
            for j in others_list:
                g = gapmod.gap_ss(qx, qy, theta, scs[j, 0], scs[j, 1], scs[j, 2])
                if g < ms:
                    ms = g
                if ms < CONTACT_SURGERY_TOL:
                    break
            if ms < CONTACT_SURGERY_TOL:
                continue
            cg = attack4.contain_gap_exact(qx, qy, theta, cx, cy, R)
            ms = min(ms, cg)
            if ms < CONTACT_SURGERY_TOL:
                continue
            if best is None or ms > best[0]:
                best = (ms, (qx, qy, theta), anchors)

    if best is None:
        # No tangency-derived candidate is near-feasible. Return unchanged with
        # fallback flag — preserves move_type so the scheduler's mix-accounting
        # remains truthful (Codex P3).
        return PerturbResult(
            scs.copy(),
            "contact_surgery",
            0.0,
            {
                "piece": p,
                "fallback": "no_feasible_pose",
                "n_candidates": len(inside),
                "pre_resolved": False,
            },
        )

    ms_seed, pose, anchors = best
    placed = scs.copy()
    placed[p] = pose

    # 3. Build active set for local resolve.
    pair_contacts, _, _ = _contacts_and_witnesses(scs, cx, cy, R)
    old_n = {b for a, b in pair_contacts if a == p} | {
        a for a, b in pair_contacts if b == p
    }
    new_n = {a for a in anchors if isinstance(a, int)}
    active = {p} | old_n | new_n

    # 4. Local resolve at R + slack (mirrors driver's R_resolve).
    R_res = R + CONTACT_SURGERY_R_RESOLVE_SLACK
    resolved, pen = _local_resolve(placed, active, cx, cy, R_res)

    D = weighted_d(scs, resolved, R)
    return PerturbResult(
        resolved,
        "contact_surgery",
        D,
        {
            "piece": p,
            "anchors": [a if isinstance(a, str) else int(a) for a in anchors],
            "active_size": len(active),
            "seed_min_slack": float(ms_seed),
            "local_resolve_penalty": float(pen),
            "n_candidates": len(inside),
            "pre_resolved": True,
        },
    )


# ---------- scheduler ----------


MOVES = {
    "flip_one": move_flip_one,
    "reseat_interior": move_reseat_interior,
    "rim_swap": move_rim_swap,
    "rotate_cluster": move_rotate_cluster,
    "contact_surgery": move_contact_surgery,
}
DEFAULT_WEIGHTS = {
    "flip_one": 0.27,
    "reseat_interior": 0.23,
    "rim_swap": 0.18,
    "rotate_cluster": 0.22,
    "contact_surgery": 0.10,
}


def propose(
    scs: np.ndarray,
    R: float,
    rng: np.random.Generator | None = None,
    move_type: str | None = None,
    weights: dict[str, float] | None = None,
    max_retries: int = 6,
) -> PerturbResult:
    """Sample a move, accept if D falls in D_BAND. Retry up to max_retries with
    a fresh random choice if out of band. On exhaustion return the last
    proposal regardless (driver logs out-of-band fraction for weight tuning)."""
    if rng is None:
        rng = np.random.default_rng()
    w = weights or DEFAULT_WEIGHTS
    names = list(w.keys())
    ps = np.array([w[n] for n in names], dtype=float)
    ps = ps / ps.sum()
    retries = max(1, max_retries)
    picked = move_type or str(rng.choice(names, p=ps))
    last: PerturbResult = MOVES[picked](scs, R, rng)
    if D_BAND[0] <= last.D <= D_BAND[1]:
        last.metadata["in_band"] = True
        return last
    for _ in range(retries - 1):
        picked = move_type or str(rng.choice(names, p=ps))
        last = MOVES[picked](scs, R, rng)
        if D_BAND[0] <= last.D <= D_BAND[1]:
            last.metadata["in_band"] = True
            return last
    last.metadata["in_band"] = False
    return last
