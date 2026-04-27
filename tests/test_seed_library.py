"""Acceptance tests for the seed_library module.

Tests run against a fresh build in a session temp directory (n_per_kind=6,
seed=0). The session-scoped fixture regenerates once per pytest run so the
assertions always reflect the current library builder.

The eight tests guard, in order:
  C1 schema: every file parses to the agreed shape so attack4 can read it.
  C2 feasibility: at least 12 of 24 seeds are valid packings.
  C3 attack4 compatibility: the load_seed entry point accepts our output.
  C4 topology diversity: the four baseline sources produce >=4 contact graphs.
  C5 near-incumbent pooling: stale diverse junk is filtered out.
  C6 tight perturbations: incumbent-biased seeds stay in a near-best basin.
  C7 fail-loud behavior: missing incumbent seed raises instead of falling back.
  C8 extra source counts: opt-in extras append files without replacing baseline.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))

import attack4  # noqa: E402
import geom  # noqa: E402
import seed_library  # noqa: E402


@pytest.fixture(scope="session")
def seed_paths(tmp_path_factory: pytest.TempPathFactory):
    """Build the 24-seed baseline library once per test session."""
    seeds_dir = tmp_path_factory.mktemp("seed-library") / "seeds"
    seed_library.make_library(
        out_dir=seeds_dir,
        n_per_kind=6,
        seed=0,
        repo_root=REPO_ROOT,
    )
    return sorted(seeds_dir.glob("seed_*.json"))


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _is_feasible_score(score) -> bool:
    return (
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and np.isfinite(score)
    )


def _contact_signature(scs: np.ndarray) -> tuple[int, int, int, int, int]:
    rnd = geom.rnd(scs)
    cx, cy, R = geom.mec_info(rnd)
    feats, _ = attack4.extract_contact_graph(rnd, float(cx), float(cy), float(R))
    return (
        sum(1 for f in feats if f.kind == "ARC_ARC"),
        sum(1 for f in feats if f.kind == "FLAT_ARC"),
        sum(1 for f in feats if f.kind == "CONTAIN_ARC"),
        sum(1 for f in feats if f.kind == "CONTAIN_FLAT"),
        sum(1 for f in feats if f.kind == "ENDPOINT_ARC"),
    )


def _write_seed_json(path: Path, scs: np.ndarray) -> None:
    rounded = geom.rnd(np.asarray(scs, dtype=np.float64))
    score = None if geom.cnt(rounded) > 0 else float(geom.mec(rounded))
    payload = {
        "score": score,
        "scs": rounded.tolist(),
        "solution": [
            {
                "x": float(rounded[i, 0]),
                "y": float(rounded[i, 1]),
                "theta": float(rounded[i, 2]),
            }
            for i in range(geom.N)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def _best_scs() -> np.ndarray:
    return np.asarray(_load(REPO_ROOT / "pool" / "best.json")["scs"], dtype=np.float64)


def _make_diverse_junk(best_scs: np.ndarray) -> np.ndarray:
    rounded = geom.rnd(best_scs)
    cx, cy, _ = geom.mec_info(rounded)
    distances = np.hypot(best_scs[:, 0] - cx, best_scs[:, 1] - cy)
    for idx in np.argsort(distances)[::-1]:
        dx = float(best_scs[idx, 0] - cx)
        dy = float(best_scs[idx, 1] - cy)
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            continue
        for delta in np.linspace(1.0, 2.5, 16):
            candidate = best_scs.copy()
            candidate[idx, 0] += delta * dx / norm
            candidate[idx, 1] += delta * dy / norm
            candidate_rounded = geom.rnd(candidate)
            if (
                geom.cnt(candidate_rounded) == 0
                and float(geom.mec(candidate_rounded)) > 3.75
            ):
                return candidate
    raise AssertionError("failed to construct a feasible diverse junk seed")


def test_seed_schema(seed_paths):
    assert len(seed_paths) == 24, f"expected 24 seed files, got {len(seed_paths)}"
    for path in seed_paths:
        data = _load(path)
        assert set(data.keys()) >= {"score", "scs", "solution"}, path
        assert len(data["scs"]) == 15, path
        assert len(data["solution"]) == 15, path
        for entry in data["solution"]:
            assert set(entry.keys()) >= {"x", "y", "theta"}, path
        for row in data["scs"]:
            assert len(row) == 3, path
            for value in row:
                assert isinstance(value, float), path


def test_at_least_twelve_feasible(seed_paths):
    n_feasible = 0
    for path in seed_paths:
        data = _load(path)
        score = data["score"]
        scs = np.asarray(data["scs"], dtype=np.float64)
        if _is_feasible_score(score):
            n_feasible += 1
            assert geom.cnt(geom.rnd(scs)) == 0, path
            assert abs(float(score) - float(geom.mec(geom.rnd(scs)))) < 1e-9, path
        else:
            assert score is None or score == "Infinity" or score == float("inf"), path
    assert n_feasible >= 12, f"only {n_feasible}/24 feasible seeds"


def test_attack4_can_load(seed_paths):
    feasible = [path for path in seed_paths if _load(path)["score"] is not None]
    assert feasible, "no feasible seeds to load via attack4"
    scs, cx, cy, R = attack4.load_seed(str(feasible[0]))
    assert scs.shape == (15, 3)
    assert np.isfinite(R)
    assert np.isfinite(cx)
    assert np.isfinite(cy)


def test_topology_diversity(seed_paths):
    distinct = set()
    for path in seed_paths:
        data = _load(path)
        if data["score"] is None:
            continue
        scs = np.asarray(data["scs"], dtype=np.float64)
        distinct.add(_contact_signature(scs))
    assert len(distinct) >= 4, f"only {len(distinct)} distinct topologies: {distinct}"


def test_near_incumbent_pool_filters_diverse_junk(tmp_path: Path):
    repo_root = tmp_path / "repo"
    pool_dir = repo_root / "pool"
    best_scs = _best_scs()
    junk_scs = _make_diverse_junk(best_scs)
    _write_seed_json(pool_dir / "best.json", best_scs)
    _write_seed_json(pool_dir / "diverse_0.json", junk_scs)
    pool = seed_library._load_near_incumbent_pool(repo_root, max_R_delta=0.02)
    assert len(pool) == 1
    np.testing.assert_allclose(pool[0], geom.rnd(best_scs))


def test_from_perturbation_tight_stays_near_incumbent():
    """Tight perturbations stay close to a near-incumbent base in raw state space.

    Raw R after perturbation+resolve is not a useful proxy: perturbing a
    tight-contact incumbent always increases R (pieces drift outward, MEC
    grows). The property we want is *state-space proximity* — the perturbed
    seed is a small step from some near-incumbent base in (x, y, theta) so
    attack4's Stage A can re-relax to a nearby topology without escaping to
    far-away R>3.5 basins.
    """
    rng = np.random.default_rng(0)
    pool = seed_library._load_near_incumbent_pool(
        REPO_ROOT, max_R_delta=seed_library.TIGHT_NEAR_INCUMBENT_DELTA
    )
    feasible_pre_resolve = 0
    for _ in range(30):
        scs = seed_library.from_perturbation_tight(rng, repo_root=REPO_ROOT)
        # Closest base in the pool — we don't know which one was picked.
        best_xy_delta = min(
            float(np.max(np.abs(scs[:, :2] - base[:, :2]))) for base in pool
        )
        best_theta_delta = min(
            float(np.max(np.abs(scs[:, 2] - base[:, 2]))) for base in pool
        )
        assert best_xy_delta < 0.6, best_xy_delta
        assert best_theta_delta < 1.0, best_theta_delta
        resolved = seed_library._resolve_overlaps(scs)
        if geom.cnt(geom.rnd(resolved)) == 0:
            feasible_pre_resolve += 1
    assert feasible_pre_resolve / 30 >= 0.40, feasible_pre_resolve


def test_from_perturbation_tight_raises_without_incumbent(tmp_path: Path):
    repo_root = tmp_path / "repo"
    pool_dir = repo_root / "pool"
    pool_dir.mkdir(parents=True)
    _write_seed_json(pool_dir / "diverse_0.json", _best_scs())
    with pytest.raises(ValueError):
        seed_library.from_perturbation_tight(
            np.random.default_rng(0), repo_root=repo_root
        )


def test_extra_per_source_produces_extra_files(tmp_path: Path):
    seeds_dir = tmp_path / "seeds"
    paths = seed_library.make_library(
        out_dir=seeds_dir,
        n_per_kind=4,
        seed=0,
        repo_root=REPO_ROOT,
        extra_per_source={"perturbation_tight": 8},
    )
    seed_paths = sorted(seeds_dir.glob("seed_*.json"))
    tight_paths = sorted(
        seeds_dir.glob("seed_perturbation_tight_*.json"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    assert len(paths) == 24
    assert len(seed_paths) == 24
    assert {path.name for path in tight_paths} == {
        f"seed_perturbation_tight_{i}.json" for i in range(4, 12)
    }
