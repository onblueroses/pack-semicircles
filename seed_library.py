"""Diverse 15-piece seed generator for the contact-graph search pipeline.

Generates seeds from four constructive sources (random, perturbation, ring, hex)
then runs an analytic overlap resolver before scoring with geom.mec. Output
schema mirrors pool/best.json so attack4.load_seed can read the files
unmodified.

Why four sources, not one: each source biases the contact topology that
emerges after relaxation. attack4's Stage A explores from whatever topology a
seed lands on, so monoculture seeds give monoculture topologies. The unit
test test_topology_diversity enforces >=4 distinct contact-feature signatures
across the library, which is the structural guarantee this module exists to
provide.

Infeasibility: a seed that the resolver cannot push apart in maxiter=200 is
written with score=null (JSON null, parsed as Python None). See
_score_payload for the serialization contract; tests handle both null and
finite-float scores.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

import gap
import geom

N = geom.N  # 15
RESOLVE_MAXITER = 200
RESOLVE_EPS = 0.05  # per-iteration push magnitude; small enough to avoid blowups
DEFAULT_RADIUS = 3.0
HEX_SPACING = 2.05
RING_INNER = 1.1
RING_OUTER = 2.5
PERTURB_SIGMA_XY = 0.20
PERTURB_SIGMA_THETA = 0.40

__all__ = [
    "make_library",
    "from_random",
    "from_perturbation",
    "from_ring",
    "from_hex",
    "_resolve_overlaps",
]


# ---------- overlap resolution ----------


def _worst_pair(scs: np.ndarray) -> Tuple[int, int, float]:
    """Return (i, j, g) for the pair with most-negative gap_ss.

    gap_ss is signed: negative is interpenetration, positive is clearance. We
    only care about the worst (most negative) pair; if no pair is negative we
    return g=+inf as a non-actionable signal.
    """
    worst = math.inf
    wi, wj = -1, -1
    for i in range(N):
        for j in range(i + 1, N):
            g = gap.gap_ss(
                scs[i, 0],
                scs[i, 1],
                scs[i, 2],
                scs[j, 0],
                scs[j, 1],
                scs[j, 2],
            )
            if g < worst:
                worst = g
                wi, wj = i, j
    return wi, wj, worst


def _resolve_overlaps(scs: np.ndarray, maxiter: int = RESOLVE_MAXITER) -> np.ndarray:
    """Iteratively push apart the worst-overlapping pair until cnt==0 or maxiter.

    Each iteration: find pair with most-negative gap_ss, displace centers along
    the connecting axis by RESOLVE_EPS each (symmetric push). Theta is left
    untouched - rotational adjustments would invalidate the contact-topology
    diversity that downstream stages rely on.

    Modifies a copy and returns it. Caller checks geom.cnt on the rounded
    result for feasibility.
    """
    scs = np.asarray(scs, dtype=np.float64).copy()
    for _ in range(maxiter):
        if geom.cnt(geom.rnd(scs)) == 0:
            return scs
        i, j, g = _worst_pair(scs)
        if i < 0 or g >= 0:
            # geom.cnt sees overlap (1e-6 thresholds) but gap_ss reads >=0;
            # nudge the closest center pair instead.
            best_d2 = math.inf
            for a in range(N):
                for b in range(a + 1, N):
                    d2 = (scs[a, 0] - scs[b, 0]) ** 2 + (scs[a, 1] - scs[b, 1]) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        i, j = a, b
        dx = scs[j, 0] - scs[i, 0]
        dy = scs[j, 1] - scs[i, 1]
        d = math.hypot(dx, dy)
        if d < 1e-9:
            # Coincident centers: push along an arbitrary axis seeded by index
            # so successive iterations don't oscillate.
            ang = (i * 0.7 + j * 1.3) % (2 * math.pi)
            ux, uy = math.cos(ang), math.sin(ang)
        else:
            ux, uy = dx / d, dy / d
        scs[i, 0] -= RESOLVE_EPS * ux
        scs[i, 1] -= RESOLVE_EPS * uy
        scs[j, 0] += RESOLVE_EPS * ux
        scs[j, 1] += RESOLVE_EPS * uy
    return scs


# ---------- generators ----------


def from_random(rng: np.random.Generator, radius: float = DEFAULT_RADIUS) -> np.ndarray:
    """15 (x, y) uniform in disk of radius `radius`, theta uniform in [0, 2pi)."""
    # rejection-sample for uniform-in-disk to avoid the sqrt-trick artifacts
    out = np.empty((N, 3), dtype=np.float64)
    k = 0
    while k < N:
        x = rng.uniform(-radius, radius)
        y = rng.uniform(-radius, radius)
        if x * x + y * y <= radius * radius:
            out[k, 0] = x
            out[k, 1] = y
            out[k, 2] = rng.uniform(0.0, 2 * math.pi)
            k += 1
    return out


def _load_seed_pool(repo_root: Path) -> List[np.ndarray]:
    """Read pool/best.json and any pool/diverse_*.json as base scs arrays."""
    pool_dir = repo_root / "pool"
    pool: List[np.ndarray] = []
    for path in [pool_dir / "best.json", *sorted(pool_dir.glob("diverse_*.json"))]:
        try:
            with path.open() as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "scs" in data:
            scs = np.asarray(data["scs"], dtype=np.float64)
            if scs.shape == (N, 3):
                pool.append(scs)
    return pool


def from_perturbation(
    rng: np.random.Generator,
    base_pool: Optional[List[np.ndarray]] = None,
    sigma_xy: float = PERTURB_SIGMA_XY,
    sigma_theta: float = PERTURB_SIGMA_THETA,
    repo_root: str | os.PathLike | None = None,
) -> np.ndarray:
    """Pick a base from pool (best.json + diverse_*.json), add Gaussian noise."""
    if base_pool is None:
        repo_root = (
            Path(repo_root)
            if repo_root is not None
            else Path(__file__).resolve().parent
        )
        base_pool = _load_seed_pool(repo_root)
    if not base_pool:
        # No incumbent available: fall back to random so make_library never
        # silently produces zero perturbation seeds.
        return from_random(rng)
    base = base_pool[rng.integers(0, len(base_pool))]
    out = base.copy()
    out[:, 0] += rng.normal(0.0, sigma_xy, size=N)
    out[:, 1] += rng.normal(0.0, sigma_xy, size=N)
    out[:, 2] += rng.normal(0.0, sigma_theta, size=N)
    return out


def from_ring(rng: np.random.Generator) -> np.ndarray:
    """5 pieces on inner ring r=1.1, 10 on outer ring r=2.5.

    Theta perturbed from radial direction by N(0, 0.3) - small enough to
    preserve the ring topology, large enough to break degenerate symmetry
    that would push every ring seed onto the same contact graph.
    """
    out = np.empty((N, 3), dtype=np.float64)
    inner_count = 5
    outer_count = N - inner_count

    # phase-randomize so different seeds aren't all rotations of one config
    phase_inner = rng.uniform(0.0, 2 * math.pi)
    phase_outer = rng.uniform(0.0, 2 * math.pi)

    for k in range(inner_count):
        a = phase_inner + 2 * math.pi * k / inner_count
        out[k, 0] = RING_INNER * math.cos(a)
        out[k, 1] = RING_INNER * math.sin(a)
        out[k, 2] = (a + rng.normal(0.0, 0.3)) % (2 * math.pi)
    for k in range(outer_count):
        a = phase_outer + 2 * math.pi * k / outer_count
        out[inner_count + k, 0] = RING_OUTER * math.cos(a)
        out[inner_count + k, 1] = RING_OUTER * math.sin(a)
        out[inner_count + k, 2] = (a + rng.normal(0.0, 0.3)) % (2 * math.pi)
    return out


def from_hex(
    rng: np.random.Generator,
    spacing: float = HEX_SPACING,
    radius: float = DEFAULT_RADIUS,
) -> np.ndarray:
    """15 centers from a hex grid (spacing 2.05) within a radius-3 patch.

    Hex with spacing 2.05 just clears the diameter-2 disk hull of each
    semicircle, so feasibility usually holds with mild theta perturbation.
    """
    candidates: List[Tuple[float, float]] = []
    # Generate enough rows/cols to comfortably cover the disk
    rows = range(-4, 5)
    cols = range(-4, 5)
    dx_row = spacing
    dy_row = spacing * math.sqrt(3) / 2
    for r in rows:
        for c in cols:
            x = c * dx_row + (0.5 * dx_row if r % 2 else 0.0)
            y = r * dy_row
            if x * x + y * y <= radius * radius:
                candidates.append((x, y))
    if len(candidates) < N:
        # Pathologically tight radius - widen by adding outer ring fallback
        # rather than failing. Keeps make_library deterministic.
        candidates.extend(
            (radius * math.cos(t), radius * math.sin(t))
            for t in np.linspace(0, 2 * math.pi, N - len(candidates), endpoint=False)
        )
    # Sort by distance to origin and take inner-most 15 to keep MEC small
    candidates.sort(key=lambda p: p[0] ** 2 + p[1] ** 2)
    chosen = candidates[:N]
    # Light shuffle so identical hex layouts get different theta assignments
    chosen_idx = rng.permutation(len(chosen))
    out = np.empty((N, 3), dtype=np.float64)
    for k, idx in enumerate(chosen_idx[:N]):
        x, y = chosen[idx]
        out[k, 0] = x
        out[k, 1] = y
        out[k, 2] = rng.uniform(0.0, 2 * math.pi)
    return out


# ---------- payload + atomic write ----------


def _score_payload(scs: np.ndarray) -> Tuple[Optional[float], np.ndarray]:
    """Return (score_or_none, rounded_scs).

    Score is geom.mec(rounded) when feasible, None when infeasible. None
    serializes as JSON null - we deliberately avoid float('inf') because
    json.dumps with default flags refuses NaN/Inf and allow_nan=True emits
    non-standard literals; null parses cleanly anywhere.
    """
    rounded = geom.rnd(scs)
    if geom.cnt(rounded) > 0:
        return None, rounded
    return float(geom.mec(rounded)), rounded


def _write_seed(path: Path, scs: np.ndarray) -> Optional[float]:
    """Score, serialize, and atomically write one seed. Returns score or None."""
    score, rounded = _score_payload(scs)
    payload = {
        "score": score,
        "scs": rounded.tolist(),
        "solution": [
            {
                "x": float(rounded[i, 0]),
                "y": float(rounded[i, 1]),
                "theta": float(rounded[i, 2]),
            }
            for i in range(N)
        ],
    }
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)
    return score


# ---------- top-level orchestration ----------


_SOURCES = ("random", "perturbation", "ring", "hex")


def _generate_one(
    kind: str,
    rng: np.random.Generator,
    base_pool: List[np.ndarray],
    repo_root: Path,
) -> np.ndarray:
    if kind == "random":
        return from_random(rng)
    if kind == "perturbation":
        return from_perturbation(rng, base_pool=base_pool, repo_root=repo_root)
    if kind == "ring":
        return from_ring(rng)
    if kind == "hex":
        return from_hex(rng)
    raise ValueError(f"unknown kind: {kind}")


def make_library(
    out_dir: str | os.PathLike = "pool/seeds",
    n_per_kind: int = 6,
    seed: int = 0,
    sources: Iterable[str] = _SOURCES,
    repo_root: str | os.PathLike | None = None,
) -> List[Path]:
    """Generate n_per_kind seeds for each source, resolve overlaps, write JSON.

    Returns the list of written paths. Each generator gets a deterministic
    sub-stream so adding/removing one source doesn't shift others' RNG.
    """
    sources = list(sources)
    repo_root_path = (
        Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    )
    out_path = Path(out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(
        tempfile.mkdtemp(dir=out_path.parent, prefix=f".seeds.tmp.{os.getpid()}.")
    )
    base_pool = _load_seed_pool(repo_root_path)
    written: List[Path] = []
    parent_seeds = np.random.SeedSequence(seed).spawn(len(sources))
    for src_idx, kind in enumerate(sources):
        rng = np.random.default_rng(parent_seeds[src_idx])
        for i in range(n_per_kind):
            raw = _generate_one(kind, rng, base_pool, repo_root_path)
            resolved = _resolve_overlaps(raw)
            final_path = out_path / f"seed_{kind}_{i}.json"
            tmp_file = tmp_path / f"seed_{kind}_{i}.json"
            _write_seed(tmp_file, resolved)
            written.append(final_path)
    if out_path.exists():
        # Single-writer assumption: accept the small gap between removing the old
        # directory and swapping in the fully written replacement.
        shutil.rmtree(out_path)
    os.replace(tmp_path, out_path)
    return written


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pool/seeds")
    ap.add_argument("--n-per-kind", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    paths = make_library(args.out, args.n_per_kind, args.seed)
    feasible = 0
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if d["score"] is not None:
            feasible += 1
    print(f"wrote {len(paths)} seeds, {feasible} feasible")
