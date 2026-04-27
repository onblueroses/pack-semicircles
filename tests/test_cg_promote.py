"""Acceptance tests for cg_promote.promote_from_campaign.

Tests guard the D3 staging contract (verify.mjs hardcodes solution.json) and
D4 single-writer invariant (only harvest.py writes pool/best.json):

  C1: a genuinely better candidate is promoted; pool/best.json updates and
      ./solution.json reflects the new best.
  C2: a worse candidate is rejected; pool/best.json is byte-for-byte
      unchanged and ./solution.json is restored byte-for-byte.
  C3: a verify.mjs failure aborts the promote; ./solution.json is restored
      byte-for-byte.
  C4: outside harvest.py, no module opens pool/best.json for writing.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))

import cg_promote  # noqa: E402
import geom  # noqa: E402
import island_orchestrator  # noqa: E402

SOLUTION_PATH = REPO_ROOT / "solution.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_scs(path: Path) -> np.ndarray:
    with open(path) as handle:
        data = json.load(handle)
    return np.asarray(data["scs"], dtype=np.float64).reshape(geom.N, 3)


def _rescore(scs: np.ndarray) -> float:
    return float(geom.mec(geom.rnd(scs)))


def _write_archive(path: Path, scs: np.ndarray) -> None:
    """Write an archive in cg_orchestrator's R-keyed schema (the production form).

    The wrapper is responsible for translating to harvest.py's score-keyed form.
    """
    rounded = geom.rnd(scs)
    score = float(geom.mec(rounded))
    payload = {
        "entries": [
            {
                "seed_id": "synthetic",
                "stage": "A",
                "R": score,
                "scs": rounded.tolist(),
                "source": "test",
            }
        ]
    }
    path.write_text(json.dumps(payload))


def _write_incumbent(path: Path, scs: np.ndarray) -> None:
    rounded = geom.rnd(scs)
    score = float(geom.mec(rounded))
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
    path.write_text(json.dumps(payload))


def _write_solution(path: Path, scs: np.ndarray) -> None:
    rounded = geom.rnd(scs)
    payload = [
        {
            "x": float(rounded[i, 0]),
            "y": float(rounded[i, 1]),
            "theta": float(rounded[i, 2]),
        }
        for i in range(geom.N)
    ]
    path.write_text(json.dumps(payload))


def _make_temp_repo_root(
    tmp_path: Path,
    solution_bytes: bytes | None,
    *,
    copy_toolchain: bool = False,
) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "pool").mkdir(parents=True, exist_ok=True)
    if copy_toolchain:
        for name in ("geom.py", "harvest.py", "verify.mjs"):
            shutil.copy2(REPO_ROOT / name, repo_root / name)
    if solution_bytes is not None:
        (repo_root / "solution.json").write_bytes(solution_bytes)
    return repo_root


def test_better_candidate_promotes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """C1: a candidate that re-scores genuinely better must be promoted.

    Setup: incumbent is pool/best_2_948891.json (R approx 2.94889);
    candidate is pool/best.json (R approx 2.94869). Re-score confirms the
    delta is greater than 1e-4.
    """
    candidate_scs = _load_scs(REPO_ROOT / "pool" / "best.json")
    incumbent_scs = _load_scs(REPO_ROOT / "pool" / "best_2_948891.json")
    cand_R = _rescore(candidate_scs)
    inc_R = _rescore(incumbent_scs)
    assert cand_R < inc_R - 1e-4, f"test fixture invalid: cand={cand_R} inc={inc_R}"

    monkeypatch.chdir(tmp_path)
    repo_root = _make_temp_repo_root(tmp_path, None, copy_toolchain=True)
    merged_path = repo_root / "merged.json"
    pool_best_path = repo_root / "pool" / "best.json"
    _write_archive(merged_path, candidate_scs)
    _write_incumbent(pool_best_path, incumbent_scs)
    _write_solution(repo_root / "solution.json", incumbent_scs)

    monkeypatch.setattr(cg_promote, "DEFAULT_INCUMBENT", pool_best_path)
    result = cg_promote.promote_from_campaign(
        merged_path,
        yes=True,
        repo_root=repo_root,
    )

    assert result.promoted is True, f"expected promote, got: {result}"
    # (b) target file updated and re-loadable with new R
    with open(pool_best_path) as handle:
        new = json.load(handle)
    new_scs = np.asarray(new["scs"], dtype=np.float64).reshape(geom.N, 3)
    new_R = _rescore(new_scs)
    assert new_R < inc_R - 1e-4, f"target not updated: new_R={new_R} inc_R={inc_R}"
    assert abs(new_R - cand_R) < 1e-9
    # (c) solution.json content matches the candidate (array shape; x/y/theta)
    sol = json.loads((repo_root / "solution.json").read_text())
    assert isinstance(sol, list) and len(sol) == geom.N
    rounded = geom.rnd(candidate_scs)
    for i in range(geom.N):
        assert abs(sol[i]["x"] - float(rounded[i, 0])) < 1e-9
        assert abs(sol[i]["y"] - float(rounded[i, 1])) < 1e-9
        assert abs(sol[i]["theta"] - float(rounded[i, 2])) < 1e-9


def test_worse_candidate_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """C2: a candidate that re-scores worse must be rejected.

    pool/best.json (or its tmp copy) must be byte-for-byte unchanged.
    solution.json must be restored byte-for-byte.
    """
    candidate_scs = _load_scs(REPO_ROOT / "pool" / "diverse_0.json")
    incumbent_scs = _load_scs(REPO_ROOT / "pool" / "best.json")
    cand_R = _rescore(candidate_scs)
    inc_R = _rescore(incumbent_scs)
    assert cand_R > inc_R + 1e-4, f"test fixture invalid: cand={cand_R} inc={inc_R}"

    monkeypatch.chdir(tmp_path)
    repo_root = _make_temp_repo_root(tmp_path, None, copy_toolchain=True)
    merged_path = repo_root / "merged.json"
    pool_best_path = repo_root / "pool" / "best.json"
    _write_archive(merged_path, candidate_scs)
    _write_incumbent(pool_best_path, incumbent_scs)
    _write_solution(repo_root / "solution.json", incumbent_scs)

    pre_incumbent_sha = _sha256(pool_best_path)
    solution_path = repo_root / "solution.json"
    pre_solution_sha = _sha256(solution_path) if solution_path.exists() else None

    result = cg_promote.promote_from_campaign(
        merged_path,
        incumbent_path=pool_best_path,
        yes=True,
        repo_root=repo_root,
    )

    assert result.promoted is False
    # (b) incumbent byte-for-byte unchanged
    assert _sha256(pool_best_path) == pre_incumbent_sha
    # (c) solution.json restored byte-for-byte
    if pre_solution_sha is None:
        assert not solution_path.exists()
    else:
        assert solution_path.exists()
        assert _sha256(solution_path) == pre_solution_sha


def test_solution_json_restored_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """C3: when verify.mjs fails, solution.json must be restored byte-for-byte."""
    candidate_scs = _load_scs(REPO_ROOT / "pool" / "best.json")
    incumbent_scs = _load_scs(REPO_ROOT / "pool" / "best_2_948891.json")
    monkeypatch.chdir(tmp_path)
    repo_root = _make_temp_repo_root(tmp_path, None, copy_toolchain=True)
    merged_path = repo_root / "merged.json"
    incumbent_path = repo_root / "incumbent.json"
    _write_archive(merged_path, candidate_scs)
    _write_incumbent(incumbent_path, incumbent_scs)
    _write_solution(repo_root / "solution.json", incumbent_scs)

    solution_path = repo_root / "solution.json"
    pre_solution_sha = _sha256(solution_path) if solution_path.exists() else None

    real_run = subprocess.run

    class _FailedVerify:
        def __init__(self) -> None:
            self.returncode = 1
            self.stdout = ""
            self.stderr = "forced verify.mjs failure"

    def fake_run(cmd, *args, **kwargs):
        if (
            isinstance(cmd, (list, tuple))
            and len(cmd) >= 2
            and cmd[0] == "node"
            and str(cmd[1]).endswith("verify.mjs")
        ):
            return _FailedVerify()
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(island_orchestrator.subprocess, "run", fake_run)

    result = cg_promote.promote_from_campaign(
        merged_path,
        incumbent_path=incumbent_path,
        yes=True,
        repo_root=repo_root,
    )

    assert result.promoted is False
    assert "verify.mjs" in result.reason.lower()
    if pre_solution_sha is None:
        assert not solution_path.exists()
    else:
        assert solution_path.exists()
        assert _sha256(solution_path) == pre_solution_sha


def test_exception_restores_solution_json(tmp_path: Path):
    candidate_scs = _load_scs(REPO_ROOT / "pool" / "best.json")
    incumbent_scs = _load_scs(REPO_ROOT / "pool" / "best_2_948891.json")
    repo_root = _make_temp_repo_root(tmp_path, b"sentinel-bytes\n")
    merged_path = repo_root / "incoming" / "merged.json"
    incumbent_path = repo_root / "pool" / "best.json"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    _write_archive(merged_path, candidate_scs)
    _write_incumbent(incumbent_path, incumbent_scs)

    original_promote = cg_promote.promote_if_better

    def raising_promote(*args, **kwargs):
        raise RuntimeError("forced promote failure")

    cg_promote.promote_if_better = raising_promote
    try:
        with pytest.raises(RuntimeError, match="forced promote failure"):
            cg_promote.promote_from_campaign(
                merged_path,
                incumbent_path=incumbent_path,
                yes=True,
                repo_root=repo_root,
            )
    finally:
        cg_promote.promote_if_better = original_promote

    assert (repo_root / "solution.json").read_bytes() == b"sentinel-bytes\n"


def test_setup_failure_on_solution_snapshot_cleans_scratch_dir(tmp_path: Path):
    candidate_scs = _load_scs(REPO_ROOT / "pool" / "best.json")
    incumbent_scs = _load_scs(REPO_ROOT / "pool" / "best_2_948891.json")
    repo_root = _make_temp_repo_root(tmp_path, None)
    merged_path = repo_root / "incoming" / "merged.json"
    incumbent_path = repo_root / "pool" / "best.json"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    _write_archive(merged_path, candidate_scs)
    _write_incumbent(incumbent_path, incumbent_scs)
    (repo_root / "solution.json").mkdir()

    scratch_dirs: list[Path] = []
    original_mkdtemp = tempfile.mkdtemp

    def fake_mkdtemp(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        path = Path(original_mkdtemp(*args, **kwargs))
        scratch_dirs.append(path)
        return str(path)

    cg_promote.tempfile.mkdtemp = fake_mkdtemp
    try:
        with pytest.raises(IsADirectoryError):
            cg_promote.promote_from_campaign(
                merged_path,
                incumbent_path=incumbent_path,
                yes=True,
                repo_root=repo_root,
            )
    finally:
        cg_promote.tempfile.mkdtemp = original_mkdtemp

    assert len(scratch_dirs) == 1
    assert not scratch_dirs[0].exists()


@pytest.mark.parametrize("should_raise", [False, True])
def test_scratch_dir_cleaned_after_return_or_raise(
    tmp_path: Path,
    should_raise: bool,
):
    candidate_scs = _load_scs(REPO_ROOT / "pool" / "best.json")
    incumbent_scs = _load_scs(REPO_ROOT / "pool" / "best_2_948891.json")
    repo_root = _make_temp_repo_root(tmp_path, b"scratch-snapshot\n")
    merged_path = repo_root / "incoming" / "merged.json"
    incumbent_path = repo_root / "pool" / "best.json"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    _write_archive(merged_path, candidate_scs)
    _write_incumbent(incumbent_path, incumbent_scs)

    scratch_dirs: list[Path] = []
    candidate_paths: list[Path] = []
    compat_paths: list[Path] = []
    original_mkdtemp = tempfile.mkdtemp
    original_promote = cg_promote.promote_if_better

    def fake_mkdtemp(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        path = Path(original_mkdtemp(*args, **kwargs))
        scratch_dirs.append(path)
        return str(path)

    def fake_promote(*args, **kwargs):
        candidate_out = Path(kwargs["candidate_out"])
        candidate_paths.append(candidate_out)
        compat_paths.append(Path(args[0]))
        if should_raise:
            raise RuntimeError("scratch cleanup raise path")
        return cg_promote.PromoteResult(False, "synthetic no-promote", 1.0, 2.0)

    cg_promote.tempfile.mkdtemp = fake_mkdtemp
    cg_promote.promote_if_better = fake_promote
    try:
        if should_raise:
            with pytest.raises(RuntimeError, match="scratch cleanup raise path"):
                cg_promote.promote_from_campaign(
                    merged_path,
                    incumbent_path=incumbent_path,
                    yes=True,
                    repo_root=repo_root,
                )
        else:
            result = cg_promote.promote_from_campaign(
                merged_path,
                incumbent_path=incumbent_path,
                yes=True,
                repo_root=repo_root,
            )
            assert result.promoted is False
    finally:
        cg_promote.tempfile.mkdtemp = original_mkdtemp
        cg_promote.promote_if_better = original_promote

    assert len(scratch_dirs) == 1
    assert len(candidate_paths) == 1
    assert len(compat_paths) == 1
    assert candidate_paths[0].name == "island-candidate.json"
    assert compat_paths[0].name == "merged-harvest-compat.json"
    assert candidate_paths[0].parent == scratch_dirs[0]
    assert compat_paths[0].parent == scratch_dirs[0]
    assert not scratch_dirs[0].exists()


def test_sigterm_restores_solution_and_cleans_scratch_dir(tmp_path: Path):
    candidate_scs = _load_scs(REPO_ROOT / "pool" / "best.json")
    incumbent_scs = _load_scs(REPO_ROOT / "pool" / "best_2_948891.json")
    repo_root = tmp_path / "repo"
    (repo_root / "pool").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "cg_promote.py", repo_root / "cg_promote.py")
    shutil.copy2(REPO_ROOT / "geom.py", repo_root / "geom.py")
    (repo_root / "island_orchestrator.py").write_text(
        "\n".join(
            [
                "from dataclasses import dataclass",
                "from pathlib import Path",
                "import time",
                "",
                "@dataclass",
                "class PromoteResult:",
                "    promoted: bool",
                "    reason: str",
                "    candidate_R: float | None = None",
                "    incumbent_R: float | None = None",
                "",
                "def promote_if_better(*args, **kwargs):",
                "    Path('promote-entered.marker').write_text('entered')",
                "    time.sleep(30.0)",
                "    return PromoteResult(False, 'slept', 1.0, 2.0)",
            ]
        )
    )

    merged_path = repo_root / "merged.json"
    incumbent_path = repo_root / "pool" / "best.json"
    _write_archive(merged_path, candidate_scs)
    _write_incumbent(incumbent_path, incumbent_scs)
    solution_bytes = b"sigterm-sentinel\n"
    solution_path = repo_root / "solution.json"
    solution_path.write_bytes(solution_bytes)

    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    env = os.environ.copy()
    env["TMPDIR"] = str(scratch_root)

    proc = subprocess.Popen(
        [
            sys.executable,
            str(repo_root / "cg_promote.py"),
            "--merged",
            str(merged_path),
            "--incumbent",
            str(incumbent_path),
            "--yes",
        ],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    marker = repo_root / "promote-entered.marker"
    deadline = time.time() + 10.0
    while not marker.exists():
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise AssertionError(
                f"cg_promote exited before SIGTERM probe: rc={proc.returncode}\n"
                f"stdout={stdout}\nstderr={stderr}"
            )
        if time.time() >= deadline:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise AssertionError(
                "timed out waiting for cg_promote to enter promote_if_better\n"
                f"stdout={stdout}\nstderr={stderr}"
            )
        time.sleep(0.05)

    time.sleep(0.2)
    os.kill(proc.pid, signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=10.0)

    assert proc.returncode == 128 + signal.SIGTERM
    assert stdout == ""
    assert "interrupted by SIGTERM" in stderr
    assert solution_path.read_bytes() == solution_bytes
    assert list(scratch_root.glob("cg_promote.*")) == []


def test_lock_serializes_concurrent_promotes(tmp_path: Path):
    candidate_scs = _load_scs(REPO_ROOT / "pool" / "best.json")
    incumbent_scs = _load_scs(REPO_ROOT / "pool" / "best_2_948891.json")
    solution_bytes = SOLUTION_PATH.read_bytes() if SOLUTION_PATH.exists() else None
    repo_root = _make_temp_repo_root(tmp_path, solution_bytes)
    merged_path = repo_root / "incoming" / "merged.json"
    incumbent_path = repo_root / "pool" / "best.json"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    _write_archive(merged_path, candidate_scs)
    _write_incumbent(incumbent_path, incumbent_scs)

    runner = tmp_path / "lock_runner.py"
    runner.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "import time",
                "from pathlib import Path",
                f"sys.path.insert(0, {str(REPO_ROOT)!r})",
                "import cg_promote",
                "repo_root = Path(sys.argv[1])",
                "merged_path = Path(sys.argv[2])",
                "incumbent_path = Path(sys.argv[3])",
                "delay = float(sys.argv[4])",
                "marker = Path(sys.argv[5])",
                "",
                "def fake_promote(*args, **kwargs):",
                "    marker.write_text('entered')",
                "    time.sleep(delay)",
                "    return cg_promote.PromoteResult(False, f'sleep={delay}', 1.0, 2.0)",
                "",
                "cg_promote.promote_if_better = fake_promote",
                "result = cg_promote.promote_from_campaign(",
                "    merged_path,",
                "    incumbent_path=incumbent_path,",
                "    yes=True,",
                "    repo_root=repo_root,",
                ")",
                "print(json.dumps({'promoted': result.promoted, 'reason': result.reason}))",
            ]
        )
    )

    first_marker = tmp_path / "first.marker"
    second_marker = tmp_path / "second.marker"
    first = subprocess.Popen(
        [
            sys.executable,
            str(runner),
            str(repo_root),
            str(merged_path),
            str(incumbent_path),
            "1.5",
            str(first_marker),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 10.0
    while not first_marker.exists():
        if first.poll() is not None:
            stdout, stderr = first.communicate()
            raise AssertionError(
                f"first promote exited before lock probe: rc={first.returncode}\n"
                f"stdout={stdout}\nstderr={stderr}"
            )
        if time.time() >= deadline:
            first.kill()
            stdout, stderr = first.communicate()
            raise AssertionError(
                f"timed out waiting for first promote to enter critical section\n"
                f"stdout={stdout}\nstderr={stderr}"
            )
        time.sleep(0.05)

    start = time.monotonic()
    second = subprocess.Popen(
        [
            sys.executable,
            str(runner),
            str(repo_root),
            str(merged_path),
            str(incumbent_path),
            "0.0",
            str(second_marker),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    second_stdout, second_stderr = second.communicate(timeout=10.0)
    elapsed = time.monotonic() - start
    first_stdout, first_stderr = first.communicate(timeout=10.0)

    assert first.returncode == 0, f"stdout={first_stdout}\nstderr={first_stderr}"
    assert second.returncode == 0, f"stdout={second_stdout}\nstderr={second_stderr}"
    assert second_marker.exists()
    assert elapsed >= 1.0, (
        "second promote did not wait for the repo lock: "
        f"elapsed={elapsed:.3f}s stdout={second_stdout} stderr={second_stderr}"
    )


def _assert_no_writer_matches(pattern: str, label: str) -> None:
    proc = subprocess.run(
        [
            "grep",
            "-rEn",
            "--include=*.py",
            "--exclude-dir=.git",
            "--exclude-dir=.local",
            "--exclude-dir=__pycache__",
            "--exclude-dir=runs",
            "--exclude-dir=.venv",
            "--exclude-dir=venv",
            "--exclude-dir=tests",
            "--exclude=harvest.py",
            pattern,
            ".",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode in (0, 1), (
        f"{label} grep error rc={proc.returncode} stderr={proc.stderr}"
    )
    if proc.returncode == 0:
        raise AssertionError(
            f"{label} forbidden pool/best.json writers found:\n{proc.stdout}"
        )


def test_no_other_writer():
    """C4: no module outside harvest.py opens pool/best.json for writing.

    Enforces the D4 single-writer invariant. Excludes harvest.py (the
    canonical writer), tests/, and ephemeral dirs (.git, .local, __pycache__,
    runs, .venv, venv).
    """
    pattern = (
        r'open\([^)]*pool/best\.json[^)]*"w"'
        r"|"
        r"pool/best\.json[\"']\s*,\s*[\"']w"
    )
    extra_pattern = (
        r'open\([^)]*pool/best\.json[^)]*["\']wb["\']'
        r"|"
        r"pool/best\.json[\"']\s*,\s*[\"']wb"
        r"|"
        r"write_text\([^)]*pool/best\.json"
        r"|"
        r"pool/best\.json[^#\n]*write_text\("
        r"|"
        r"os\.replace\([^)]*pool/best\.json"
        r"|"
        r"pool/best\.json[^#\n]*os\.replace\("
        r"|"
        r"os\.rename\([^)]*pool/best\.json"
        r"|"
        r"pool/best\.json[^#\n]*os\.rename\("
        r"|"
        r"shutil\.copy[a-zA-Z0-9_]*\([^)]*pool/best\.json"
        r"|"
        r"pool/best\.json[^#\n]*shutil\.copy[a-zA-Z0-9_]*\("
    )
    _assert_no_writer_matches(pattern, "legacy writer grep")
    _assert_no_writer_matches(extra_pattern, "extended writer grep")
