"""Acceptance tests for the seed_library module.

Tests run against a fresh build in a session temp directory (n_per_kind=6,
seed=0). The session-scoped fixture regenerates once per pytest run so the
assertions always reflect the current library builder.

The four tests guard, in order:
  C1 schema: every file parses to the agreed shape so attack4 can read it.
  C2 feasibility: at least 12 seeds are valid packings.
  C3 attack4 compatibility: the load_seed entry point accepts our output.
  C4 topology diversity: the sources produce >=4 distinct contact graphs.
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


def test_seed_schema(seed_paths):
    assert len(seed_paths) >= 30, f"expected >=30 seed files, got {len(seed_paths)}"
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
        if _is_feasible_score(score):
            n_feasible += 1
            assert geom.cnt(geom.rnd(scs)) == 0, path
            assert abs(float(score) - float(geom.mec(geom.rnd(scs)))) < 1e-9, path
        else:
            assert score is None or score == "Infinity" or score == float("inf"), path
    assert n_feasible >= 12, f"only {n_feasible} feasible seeds"


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
    tight_distinct = set()
    for path in seed_paths:
        data = _load(path)
        if data["score"] is None:
            continue
        scs = np.asarray(data["scs"], dtype=np.float64)
        signature = _contact_signature(scs)
        distinct.add(signature)
        if "perturbation_tight" in path.name:
            tight_distinct.add(signature)
    assert len(distinct) >= 4, f"only {len(distinct)} distinct topologies: {distinct}"
    assert len(tight_distinct) >= 2, (
        f"tight seeds collapsed to one topology: {tight_distinct}"
    )


def test_from_perturbation_tight_stays_near_incumbent():
    """Tight perturbations stay close to the incumbent in raw state space.

    Raw R after perturbation is not a useful proxy: perturbing a tight-contact
    incumbent always increases R (pieces drift outward, MEC grows). The
    property we want is *state-space proximity* - the perturbed seed is close
    enough to the incumbent that attack4's Stage A can re-relax to nearby
    topologies, not collapse back to the incumbent or escape to R~3.8 basins.
    """
    rng = np.random.default_rng(0)
    base_pool = seed_library._load_seed_pool(REPO_ROOT)
    incumbent = base_pool[0]
    max_xy_delta = 0.0
    max_theta_delta = 0.0
    for _ in range(20):
        scs = seed_library.from_perturbation_tight(
            rng,
            base_pool=base_pool,
            repo_root=REPO_ROOT,
        )
        resolved = seed_library._resolve_overlaps(scs)
        rounded = geom.rnd(resolved)
        assert geom.cnt(rounded) == 0
        max_xy_delta = max(
            max_xy_delta,
            float(np.max(np.abs(scs[:, :2] - incumbent[:, :2]))),
        )
        max_theta_delta = max(
            max_theta_delta,
            float(np.max(np.abs(scs[:, 2] - incumbent[:, 2]))),
        )
    # Targeted regime allows up to TIGHT_TARGETED_SIGMA_XY * ~3 sigma displacement
    # on K<=3 pieces; allow generous headroom but stay well below R-blowup distance.
    assert max_xy_delta < 0.6, max_xy_delta
    assert max_theta_delta < 1.0, max_theta_delta


def test_source_weight_cli_extra_seeds(tmp_path: Path):
    seeds_dir = tmp_path / "seeds"
    paths = seed_library.make_library(
        out_dir=seeds_dir,
        n_per_kind=6,
        seed=0,
        repo_root=REPO_ROOT,
        source_weights={"perturbation_tight": 8},
    )
    tight_paths = sorted(seeds_dir.glob("seed_perturbation_tight_*.json"))
    assert len(paths) == (6 * len(seed_library._SOURCES)) + 8
    assert len(tight_paths) == 14
    expected_names = {f"seed_perturbation_tight_{i}.json" for i in range(14)}
    assert {path.name for path in tight_paths} == expected_names
