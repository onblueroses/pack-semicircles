import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))

import cg_orchestrator  # noqa: E402
import common  # noqa: E402
import geom  # noqa: E402
import seed_library  # noqa: E402


def _load(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _build_seed_library(tmp_path_factory: pytest.TempPathFactory, name: str) -> Path:
    seeds_dir = tmp_path_factory.mktemp(name) / "all-seeds"
    seed_library.make_library(
        out_dir=seeds_dir,
        n_per_kind=6,
        seed=0,
        repo_root=REPO_ROOT,
    )
    return seeds_dir


def _copy_feasible_seeds(source_dir: Path, dest_dir: Path, count: int) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in sorted(source_dir.glob("seed_*.json")):
        if _load(path)["score"] is None:
            continue
        target = dest_dir / path.name
        shutil.copy2(path, target)
        copied.append(target)
        if len(copied) == count:
            return copied
    raise AssertionError(f"expected at least {count} feasible seeds in {source_dir}")


@pytest.fixture(scope="session")
def smoke_campaign(tmp_path_factory: pytest.TempPathFactory):
    source_dir = _build_seed_library(tmp_path_factory, "cg-smoke-library")
    seed_dir = tmp_path_factory.mktemp("cg-smoke-seeds") / "seeds"
    copied = _copy_feasible_seeds(source_dir, seed_dir, 3)
    out_root = tmp_path_factory.mktemp("cg-smoke-out")
    merged_path = cg_orchestrator.run_campaign(seed_dir, 0.05, 2, out_root)
    return copied, out_root, merged_path


def test_smoke_run_completes(smoke_campaign):
    copied, out_root, merged_path = smoke_campaign
    payload = _load(merged_path)
    assert "entries" in payload
    assert len(payload["entries"]) >= 1
    for seed_path in copied:
        run_log = out_root / seed_path.stem / "run.log"
        assert run_log.exists()
        assert run_log.stat().st_size > 0


def test_stop_file_propagates(tmp_path_factory: pytest.TempPathFactory):
    source_dir = _build_seed_library(tmp_path_factory, "cg-stop-library")
    seed_dir = tmp_path_factory.mktemp("cg-stop-seeds") / "seeds"
    _copy_feasible_seeds(source_dir, seed_dir, 3)
    out_root = tmp_path_factory.mktemp("cg-stop-out")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "cg_orchestrator.py"),
            "--seed-dir",
            str(seed_dir),
            "--hours",
            "1.0",
            "--workers",
            "2",
            "--out",
            str(out_root),
        ],
        cwd=str(tmp_path_factory.mktemp("cg-stop-cwd")),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(5.0)
    (out_root / "STOP").touch()
    try:
        stdout, _ = proc.communicate(timeout=30.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()
        raise AssertionError(f"orchestrator did not exit within 30s\n{stdout}")
    assert proc.returncode == 0, stdout
    assert (out_root / "merged.json").exists()


def test_atomic_merge_no_partial(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
):
    source_dir = _build_seed_library(tmp_path_factory, "cg-atomic-library")
    seed_dir = tmp_path / "seeds"
    out_root = tmp_path / "out"
    _copy_feasible_seeds(source_dir, seed_dir, 3)
    called = {"write": 0, "replace": 0}
    original_write = common.write_json_atomic
    original_replace = common.os.replace

    def tracking_replace(src, dst):
        called["replace"] += 1
        return original_replace(src, dst)

    def wrapped_write(path, data):
        called["write"] += 1
        monkeypatch.setattr(common.os, "replace", tracking_replace)
        try:
            return original_write(path, data)
        finally:
            monkeypatch.setattr(common.os, "replace", original_replace)

    monkeypatch.setattr(common, "write_json_atomic", wrapped_write)
    merged_path = cg_orchestrator.run_campaign(seed_dir, 0.05, 2, out_root)
    assert merged_path.exists()
    assert called["write"] >= 1
    assert called["replace"] >= 1
    assert not (out_root / "merged.json.tmp").exists()


def test_merged_entries_sorted(smoke_campaign):
    _copied, _out_root, merged_path = smoke_campaign
    payload = _load(merged_path)
    entries = payload["entries"]
    scores = [float(entry["R"]) for entry in entries]
    assert scores == sorted(scores)
    for entry in entries:
        expected = float(geom.mec(geom.rnd(np.asarray(entry["scs"], dtype=np.float64))))
        assert abs(expected - float(entry["R"])) < 1e-9
