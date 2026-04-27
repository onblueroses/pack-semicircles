# ruff: noqa: E402
from __future__ import annotations

import os

THREAD_CAP_ENV = (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)

for _name in THREAD_CAP_ENV:
    os.environ[_name] = "1"

import argparse
import json
import multiprocessing as mp
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import common
import geom
from island_orchestrator import _set_pdeathsig_preexec

STOP_POLL_SECONDS = 1.0
STOP_GRACE_SECONDS = 30.0
STAGE_B_RE = re.compile(r"stage_b_depth(\d+)\.json$")


@dataclass
class _ActiveTask:
    seed_path: Path
    out_dir: Path
    async_result: Any


def _init_pool_worker() -> None:
    _set_pdeathsig_preexec()


def _prepare_seed_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def _run_seed(
    seed_path_str: str,
    hours: float,
    out_root_str: str,
    repo_root_str: str,
) -> dict[str, Any]:
    seed_path = Path(seed_path_str)
    out_root = Path(out_root_str)
    repo_root = Path(repo_root_str)
    seed_id = seed_path.stem
    out_dir = out_root / seed_id
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = out_dir / "run.log"
    env = os.environ.copy()
    cmd = [
        sys.executable,
        str(repo_root / "attack4.py"),
        "--root",
        str(seed_path),
        "--stage-b",
        "--workers",
        "1",
        "--hours",
        str(hours),
        "--out",
        str(out_dir),
    ]
    returncode = -1
    try:
        with open(run_log_path, "a", encoding="utf-8") as run_log:
            run_log.write(f"[cg_orchestrator] seed={seed_id}\n")
            run_log.flush()
            completed = subprocess.run(
                cmd,
                check=False,
                stdout=run_log,
                stderr=subprocess.STDOUT,
                cwd=str(repo_root),
                env=env,
                start_new_session=True,
                preexec_fn=_set_pdeathsig_preexec,
            )
            returncode = int(completed.returncode)
            run_log.write(f"[cg_orchestrator] returncode={returncode}\n")
    except Exception as exc:
        try:
            with open(run_log_path, "a", encoding="utf-8") as run_log:
                run_log.write(f"[cg_orchestrator] worker_error={exc!r}\n")
        except OSError:
            pass
    return {"seed_id": seed_id, "returncode": returncode}


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _fanout_stop(active_tasks: dict[str, _ActiveTask]) -> None:
    for task in active_tasks.values():
        try:
            _touch(task.out_dir / "STOP")
        except OSError:
            continue


def _load_json(path: Path) -> Any:
    with open(path) as handle:
        return json.load(handle)


def _build_entry(
    seed_id: str, stage: str, source_path: Path, payload: Any
) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or "scs" not in payload:
        return None
    scs = np.asarray(payload["scs"], dtype=np.float64)
    if scs.shape != (geom.N, 3):
        return None
    rounded = geom.rnd(scs)
    score = float(geom.mec(rounded))
    return {
        "seed_id": seed_id,
        "stage": stage,
        "R": score,
        "scs": rounded.tolist(),
        "source": source_path.name,
    }


def _build_stage_b_entry(
    seed_id: str, seed_out_dir: Path, stage_b_path: Path
) -> dict[str, Any] | None:
    match = STAGE_B_RE.fullmatch(stage_b_path.name)
    if match is None:
        return None
    depth = int(match.group(1))
    stage = f"B_depth{depth}"
    candidates = [stage_b_path, seed_out_dir / f"champion_d{depth}.json"]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = _load_json(candidate)
        except (OSError, json.JSONDecodeError):
            continue
        entry = _build_entry(seed_id, stage, candidate, payload)
        if entry is not None:
            return entry
    return None


def _merge_outputs(out_root: Path) -> Path:
    entries: list[dict[str, Any]] = []
    for seed_out_dir in sorted(path for path in out_root.iterdir() if path.is_dir()):
        seed_id = seed_out_dir.name
        stage_a_path = seed_out_dir / "stage_a.json"
        if stage_a_path.exists():
            try:
                payload = _load_json(stage_a_path)
            except (OSError, json.JSONDecodeError):
                payload = None
            entry = _build_entry(seed_id, "A", stage_a_path, payload)
            if entry is not None:
                entries.append(entry)
        for stage_b_path in sorted(seed_out_dir.glob("stage_b_depth*.json")):
            entry = _build_stage_b_entry(seed_id, seed_out_dir, stage_b_path)
            if entry is not None:
                entries.append(entry)
    entries.sort(key=lambda item: float(item["R"]))
    merged_path = out_root / "merged.json"
    common.write_json_atomic(str(merged_path), {"entries": entries})
    return merged_path


def run_campaign(
    seed_dir: Path,
    hours: float,
    workers: int,
    out_root: Path,
) -> Path:
    """Run attack4 against every seed_*.json in seed_dir in parallel."""
    seed_dir = Path(seed_dir)
    out_root = Path(out_root)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if not seed_dir.is_dir():
        raise FileNotFoundError(seed_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    seed_paths = sorted(seed_dir.glob("seed_*.json"))
    if not seed_paths:
        merged_path = out_root / "merged.json"
        common.write_json_atomic(str(merged_path), {"entries": []})
        return merged_path

    repo_root = Path(__file__).resolve().parent
    pool_size = min(workers, len(seed_paths))
    active_tasks: dict[str, _ActiveTask] = {}
    next_index = 0
    stop_requested = False
    stop_deadline: float | None = None
    pool = mp.Pool(processes=pool_size, initializer=_init_pool_worker)
    closed = False
    terminated = False
    try:
        while active_tasks or (next_index < len(seed_paths) and not stop_requested):
            while (
                not stop_requested
                and next_index < len(seed_paths)
                and len(active_tasks) < pool_size
            ):
                seed_path = seed_paths[next_index]
                seed_out_dir = out_root / seed_path.stem
                _prepare_seed_out_dir(seed_out_dir)
                async_result = pool.apply_async(
                    _run_seed,
                    (
                        str(seed_path),
                        hours,
                        str(out_root),
                        str(repo_root),
                    ),
                )
                active_tasks[seed_path.stem] = _ActiveTask(
                    seed_path=seed_path,
                    out_dir=seed_out_dir,
                    async_result=async_result,
                )
                next_index += 1

            for seed_id, task in list(active_tasks.items()):
                if not task.async_result.ready():
                    continue
                try:
                    task.async_result.get()
                except Exception:
                    pass
                active_tasks.pop(seed_id)

            if not stop_requested and (out_root / "STOP").exists():
                stop_requested = True
                stop_deadline = time.monotonic() + STOP_GRACE_SECONDS
                _fanout_stop(active_tasks)

            if (
                stop_requested
                and stop_deadline is not None
                and time.monotonic() >= stop_deadline
            ):
                pool.terminate()
                terminated = True
                break

            if active_tasks or (next_index < len(seed_paths) and not stop_requested):
                time.sleep(STOP_POLL_SECONDS)

        if terminated:
            pool.join()
        else:
            pool.close()
            closed = True
            pool.join()
    finally:
        if not terminated and not closed:
            pool.terminate()
            pool.join()

    return _merge_outputs(out_root)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", required=True)
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_root = (
        Path(args.out)
        if args.out is not None
        else Path("runs") / f"cg_{int(time.time())}"
    )
    merged_path = run_campaign(
        seed_dir=Path(args.seed_dir),
        hours=args.hours,
        workers=args.workers,
        out_root=out_root,
    )
    with open(merged_path) as handle:
        payload = json.load(handle)
    print(f"{merged_path} entries={len(payload.get('entries', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
