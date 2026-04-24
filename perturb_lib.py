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


# ---------- scheduler ----------


MOVES = {
    "flip_one": move_flip_one,
    "reseat_interior": move_reseat_interior,
    "rim_swap": move_rim_swap,
    "rotate_cluster": move_rotate_cluster,
}
DEFAULT_WEIGHTS = {
    "flip_one": 0.30,
    "reseat_interior": 0.25,
    "rim_swap": 0.20,
    "rotate_cluster": 0.25,
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
