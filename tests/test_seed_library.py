"""Acceptance tests for the seed_library module.

Tests run against a fresh build in a session temp directory (n_per_kind=6,
seed=0). The session-scoped fixture regenerates once per pytest run so the
assertions always reflect the current library builder.

The four tests guard, in order:
  C1 schema: every file parses to the agreed shape so attack4 can read it.
  C2 feasibility: at least 12 of 24 seeds are valid packings.
  C3 attack4 compatibility: the load_seed entry point accepts our output.
  C4 topology diversity: the four sources produce >=4 distinct contact graphs.
"""

import json
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
    """Build the 24-seed library once per test session in a fresh temp dir."""
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


def test_seed_schema(seed_paths):
    assert len(seed_paths) >= 24, f"expected >=24 seed files, got {len(seed_paths)}"
    for path in seed_paths:
        data = _load(path)
        assert set(data.keys()) >= {"score", "scs", "solution"}, path
        assert len(data["scs"]) == 15, path
        assert len(data["solution"]) == 15, path
        for entry in data["solution"]:
            assert set(entry.keys()) >= {"x", "y", "theta"}, path
        for row in data["scs"]:
            assert len(row) == 3, path
            for v in row:
                assert isinstance(v, float), path


def test_at_least_twelve_feasible(seed_paths):
    n_feasible = 0
    for path in seed_paths:
        data = _load(path)
        score = data["score"]
        scs = np.asarray(data["scs"], dtype=np.float64)
        # We serialize infeasible seeds as JSON null -> Python None. Anything
        # finite is feasible; bool/None/Infinity-string => infeasible.
        is_feasible = (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and np.isfinite(score)
        )
        if is_feasible:
            n_feasible += 1
            assert geom.cnt(geom.rnd(scs)) == 0, path
            assert abs(float(score) - float(geom.mec(geom.rnd(scs)))) < 1e-9, path
        else:
            assert score is None or score == "Infinity" or score == float("inf"), path
    assert n_feasible >= 12, f"only {n_feasible}/24 feasible seeds"


def test_attack4_can_load(seed_paths):
    feasible = [p for p in seed_paths if _load(p)["score"] is not None]
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
        rnd = geom.rnd(scs)
        cx, cy, R = geom.mec_info(rnd)
        feats, _ = attack4.extract_contact_graph(rnd, float(cx), float(cy), float(R))
        counts = (
            sum(1 for f in feats if f.kind == "ARC_ARC"),
            sum(1 for f in feats if f.kind == "FLAT_ARC"),
            sum(1 for f in feats if f.kind == "CONTAIN_ARC"),
            sum(1 for f in feats if f.kind == "CONTAIN_FLAT"),
            sum(1 for f in feats if f.kind == "ENDPOINT_ARC"),
        )
        distinct.add(counts)
    assert len(distinct) >= 4, f"only {len(distinct)} distinct topologies: {distinct}"
