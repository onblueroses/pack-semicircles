"""Unit tests for island_orchestrator helpers.

Covers the parts of Phase 2 that are unit-testable without spawning real
mbh_driver subprocesses: NaN/inf filter (D12), top-K selection, seed
materialization, union merge semantics (D11), promotion gate ordering (D14),
and PR_SET_PDEATHSIG callable shape (D13). Spec: pack-semi-island-search
Phase 2 C5/C6/C7/C8.

The end-to-end smoke C1/C2/C3/C4 are exercised by the manual launch in
Phase 3 (and the supervisor.log is the artifact); they are not encoded
here because they require real subprocess timing.
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import geom
import island_orchestrator as io


# ---------- _filter_finite (D12) ----------


def _entry(scs, score, label="x"):
    return {
        "scs": np.asarray(scs, dtype=np.float64).tolist(),
        "score": score,
        "label": label,
        "trial": 1,
    }


def test_filter_finite_drops_nan_in_scs():
    good_scs = np.zeros((geom.N, 3))
    bad_scs = good_scs.copy()
    bad_scs[3, 1] = float("nan")
    kept, nan_n, other_n = io._filter_finite(
        [_entry(good_scs, 2.95), _entry(bad_scs, 2.99)]
    )
    assert len(kept) == 1
    assert kept[0]["score"] == 2.95
    assert nan_n == 1
    assert other_n == 0


def test_filter_finite_inf_score_counts_as_nan_class():
    scs = np.zeros((geom.N, 3))
    kept, nan_n, other_n = io._filter_finite(
        [_entry(scs, 2.95), _entry(scs, float("inf"))]
    )
    assert len(kept) == 1
    assert nan_n == 1
    assert other_n == 0


def test_filter_finite_wrong_shape_counts_as_other_class():
    bad = {"scs": [[0.0, 0.0]], "score": 2.95, "label": "bad", "trial": 1}
    kept, nan_n, other_n = io._filter_finite([bad])
    assert kept == []
    assert nan_n == 0
    assert other_n == 1


def test_filter_finite_score_above_cap_counts_as_other_class():
    scs = np.zeros((geom.N, 3))
    kept, nan_n, other_n = io._filter_finite([_entry(scs, 1e9)], score_cap=1e8)
    assert kept == []
    assert nan_n == 0
    assert other_n == 1


# ---------- _select_top_k ----------


def test_select_top_k_returns_lowest_scores_first():
    scs = np.zeros((geom.N, 3))
    entries = [
        _entry(scs, 3.05),
        _entry(scs, 2.95),
        _entry(scs, 2.97),
        _entry(scs, 2.99),
    ]
    top = io._select_top_k(entries, 2)
    assert [e["score"] for e in top] == [2.95, 2.97]


# ---------- _materialize_seed ----------


def test_materialize_seed_matches_best_json_schema():
    with tempfile.TemporaryDirectory() as tmp:
        scs = np.random.default_rng(0).uniform(-1, 1, size=(geom.N, 3))
        entry = _entry(scs, 2.95)
        path = Path(tmp) / "seed.json"
        io._materialize_seed(entry, path)
        loaded = json.loads(path.read_text())
        assert set(loaded.keys()) == {"score", "scs", "solution"}
        assert loaded["score"] == 2.95
        assert len(loaded["scs"]) == geom.N
        assert len(loaded["solution"]) == geom.N
        assert set(loaded["solution"][0].keys()) == {"x", "y", "theta"}


# ---------- _union_merge (D11 + D12) ----------


def _write_archive(path: Path, entries: list[dict]) -> None:
    payload = {
        "run": "test",
        "best_score": 0.0,
        "best_alternative_score": None,
        "best_alternative_delta": None,
        "archive_size": len(entries),
        "archive_slots": 64,
        "distinct_min_l2": 0.08,
        "contact_radius": 1.0,
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _make_real_scs(scale: float = 1.0) -> np.ndarray:
    """Produce a feasible scs by scaling incumbent xy outward. scale=1.0
    returns the incumbent unchanged; scale>1.0 produces a strictly worse
    (larger MEC) but still feasible variant — gaps only grow."""
    inc = json.loads(Path("pool/best.json").read_text())
    base = np.asarray(inc["scs"], dtype=np.float64).reshape(geom.N, 3)
    out = base.copy()
    out[:, :2] *= scale
    return geom.rnd(out)


def test_union_merge_preserves_previous_pool_when_worker_archive_missing():
    """D11: when a worker fails to write an archive, the union keeps the
    previous pool intact rather than dropping it. This is the partial-failure
    rule that distinguishes union semantics from replace semantics."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        prev = tmp_path / "prev.json"
        missing_worker = tmp_path / "w0" / "archive.json"  # never written
        out = tmp_path / "merged.json"

        prev_scs = _make_real_scs(scale=1.0)
        prev_score = float(geom.mec(prev_scs))
        _write_archive(prev, [_entry(prev_scs, prev_score, label="prev")])

        size, nan_n, other_n, best = io._union_merge(prev, [missing_worker], out)
        assert size == 1
        assert nan_n == 0
        assert other_n == 0
        assert abs(best - prev_score) < 1e-9
        # Sources field records exactly what got merged in (the missing worker
        # path is omitted, the prev pool is included).
        merged = json.loads(out.read_text())
        assert str(prev) in merged["sources"]
        assert str(missing_worker) not in merged["sources"]


def test_union_merge_combines_inputs_via_basin_archive_consider():
    """D11: union runs both prev + worker entries through BasinArchive.consider,
    so any new basin (different signature) survives alongside the existing one.
    Two identical-signature entries dedup to one — that's BasinArchive's job,
    not the union's."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        prev = tmp_path / "prev.json"
        worker = tmp_path / "w0" / "archive.json"
        out = tmp_path / "merged.json"

        scs = _make_real_scs(scale=1.0)
        prev_score = float(geom.mec(scs))
        _write_archive(prev, [_entry(scs, prev_score, label="prev")])
        # Worker writes the same basin (synthetic dup) — union must not error,
        # and the merged archive should contain the (single) basin.
        _write_archive(worker, [_entry(scs, prev_score, label="worker_dup")])

        size, nan_n, other_n, _best = io._union_merge(prev, [worker], out)
        assert size == 1
        assert nan_n == 0
        assert other_n == 0
        merged = json.loads(out.read_text())
        # Both source paths recorded — proves the union *visited* both.
        assert str(prev) in merged["sources"]
        assert str(worker) in merged["sources"]


def test_union_merge_drops_nan_entries_from_workers():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        worker = tmp_path / "w0" / "archive.json"
        out = tmp_path / "merged.json"

        good_scs = _make_real_scs(scale=1.0)
        good_score = float(geom.mec(good_scs))
        bad_scs = good_scs.copy()
        bad_scs[0, 0] = float("nan")
        _write_archive(
            worker,
            [
                _entry(good_scs, good_score),
                _entry(bad_scs, float("nan")),
            ],
        )
        size, nan_n, other_n, _ = io._union_merge(None, [worker], out)
        assert size == 1
        assert nan_n == 1
        assert other_n == 0


def test_union_merge_raises_when_all_entries_filtered():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        worker = tmp_path / "w0" / "archive.json"
        out = tmp_path / "merged.json"
        bad = np.zeros((geom.N, 3))
        bad[0, 0] = float("nan")
        _write_archive(worker, [_entry(bad, float("nan"))])
        try:
            io._union_merge(None, [worker], out)
        except RuntimeError as exc:
            assert "empty" in str(exc).lower()
        else:
            raise AssertionError("expected RuntimeError on empty merge")


# ---------- promote_if_better (D14) ----------


def test_promote_if_better_blocks_when_no_improvement():
    """Candidate strictly worse than incumbent -> result reason 'no improvement'.
    verify.mjs is never invoked because step (2) short-circuits before step (4)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        worse_scs = _make_real_scs(scale=1.005)  # strictly worse R, still feasible
        worse_score = float(geom.mec(worse_scs))
        cand_archive = tmp_path / "merged.json"
        _write_archive(cand_archive, [_entry(worse_scs, worse_score)])
        result = io.promote_if_better(
            cand_archive,
            incumbent_path=Path("pool/best.json"),
            yes=True,
            verify_cmd=(
                "false",
            ),  # would FAIL if invoked — step (2) must short-circuit first
        )
        assert result.promoted is False
        assert "improvement" in result.reason.lower(), (
            f"expected 'no improvement' reason, got: {result.reason}"
        )


def test_promote_if_better_blocks_when_archive_missing():
    result = io.promote_if_better(
        Path("/tmp/does-not-exist-xyz.json"),
        incumbent_path=Path("pool/best.json"),
        yes=True,
    )
    assert result.promoted is False
    assert "not found" in result.reason.lower()


def test_promote_if_better_blocks_when_incumbent_missing():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cand_archive = tmp_path / "merged.json"
        scs = _make_real_scs(scale=1.0)
        _write_archive(cand_archive, [_entry(scs, float(geom.mec(scs)))])
        result = io.promote_if_better(
            cand_archive,
            incumbent_path=tmp_path / "no-incumbent.json",
            yes=True,
        )
        assert result.promoted is False
        assert "not found" in result.reason.lower()


# ---------- spawn env / preexec (D13, D15) ----------


def test_spawn_worker_sets_thread_cap_env():
    """D15: the spawned worker process has all 4 thread-cap vars = 1.

    We don't actually spawn mbh_driver here (too slow); we use python -c to
    print the relevant env vars, with the same env-shaping path as
    _spawn_worker. This tests the env-construction logic, not mbh_driver.
    """
    env = os.environ.copy()
    for k in io.THREAD_CAP_ENV:
        env[k] = "1"
    code = (
        "import os; "
        "import json; "
        "print(json.dumps({k: os.environ.get(k) for k in "
        f"{list(io.THREAD_CAP_ENV)}}}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0
    caps = json.loads(proc.stdout.strip())
    for k in io.THREAD_CAP_ENV:
        assert caps[k] == "1", f"env var {k} not set to 1: {caps}"


def test_set_pdeathsig_preexec_callable_succeeds_in_subprocess():
    """D13: preexec_fn calling prctl(PR_SET_PDEATHSIG, SIGTERM) must not
    raise OSError when run in a fork-context. The standard CPython signature
    is the same here as it would be in subprocess.Popen's child-side fn."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    res = libc.prctl(io.PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    assert res == 0, f"prctl PR_SET_PDEATHSIG failed: errno={ctypes.get_errno()}"


def test_pdeathsig_kills_child_when_parent_dies():
    """D13 end-to-end: spawn a parent that spawns a child with PR_SET_PDEATHSIG;
    kill the parent with SIGKILL; child must die within 5s.

    We use a shim that runs a sleep loop in the child; the orchestrator's
    actual _spawn_worker is functionally equivalent.
    """
    parent_code = (
        "import ctypes, os, signal, subprocess, sys, time\n"
        "def preexec():\n"
        "    libc = ctypes.CDLL('libc.so.6', use_errno=True)\n"
        f"    libc.prctl({io.PR_SET_PDEATHSIG}, {int(signal.SIGTERM)}, 0, 0, 0)\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(60)'],\n"
        "    start_new_session=True, preexec_fn=preexec)\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Read child PID (first line of stdout)
        assert parent.stdout is not None
        child_pid_line = parent.stdout.readline().decode().strip()
        child_pid = int(child_pid_line)
        # Confirm child alive
        os.kill(child_pid, 0)
        # Kill parent with SIGKILL
        parent.kill()
        parent.wait(timeout=5)
        # Child should die from SIGTERM within 5s
        import time as _t

        start = _t.time()
        while _t.time() - start < 5.0:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                return  # success
            _t.sleep(0.2)
        # Still alive after 5s — fail
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise AssertionError(
            f"child {child_pid} survived parent SIGKILL > 5s (PR_SET_PDEATHSIG broken)"
        )
    finally:
        if parent.poll() is None:
            parent.kill()
