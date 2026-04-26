"""Island-model MBH orchestrator: N epochs of M parallel mbh_driver workers.

Between epochs: union-merge worker archives with the previous pool (D11) +
NaN/inf filter at the merge boundary (D12). Reseed workers from top-K basins.

Process management:
  - Each worker spawned with thread caps (D15: NUMBA/OMP/OPENBLAS/MKL=1).
  - start_new_session=True puts each worker in its own process group.
  - PR_SET_PDEATHSIG(SIGTERM) ensures kernel-enforced cleanup if the
    orchestrator dies abruptly (D13).

STOP propagation: touching <out_dir>/STOP (or sending SIGTERM/SIGINT to the
orchestrator) fans out STOP files to every active worker dir; the workers'
existing stop_flag handler in mbh_driver writes done_mbh.flag and exits.

Promotion gate (D14): single sequential `promote_if_better()` function.
  harvest --dry-run -> verify.mjs -> user prompt -> harvest --yes.
The function structurally enforces step ordering — bypassable only by code edit.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional

import numpy as np

import archive_merge
import basin_archive
import geom

PR_SET_PDEATHSIG = 1
EPOCH_TIMEOUT_FACTOR = 1.5
MIN_COMPLETE_FRACTION = 0.5
DONE_FLAG_NAME = "done_mbh.flag"
STOP_FLAG_NAME = "STOP"
DEFAULT_INCUMBENT = "pool/best.json"
PROFILES = ("default", "explore", "exploit")
THREAD_CAP_ENV = (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _set_pdeathsig_preexec():
    """Run in child between fork and exec. Installs PR_SET_PDEATHSIG so the
    kernel sends SIGTERM to the child if the orchestrator dies (D13).
    Then race-checks PPID — if parent already died during fork, exit."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    res = libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    if res != 0:
        raise OSError(ctypes.get_errno(), "prctl PR_SET_PDEATHSIG failed")
    if os.getppid() == 1:
        # Parent died after fork but before prctl took effect; exit cleanly.
        os._exit(0)


def _load_pool(path: Path) -> list[dict]:
    """Load entries from a merged-archive json (mbh_driver / archive_merge format)."""
    with open(path) as f:
        d = json.load(f)
    if "entries" not in d:
        raise ValueError(f"not an archive file: {path}")
    return d["entries"]


def _filter_finite(
    entries: list[dict], score_cap: float = 1e8
) -> tuple[list[dict], int, int]:
    """D12 NaN/inf guard at the merge boundary. Returns
    (kept, dropped_nan_count, dropped_other_count).

    `dropped_nan_count` counts entries dropped because scs OR score is NaN/inf
    — the headline D12 telemetry the spec asks for.
    `dropped_other_count` counts entries dropped for malformed shape or
    sentinel-cap scores (>= score_cap). Tracked separately so the union-merge
    log can surface them without conflating with NaN drops.
    """
    kept: list[dict] = []
    dropped_nan = 0
    dropped_other = 0
    for e in entries:
        try:
            scs = np.asarray(e.get("scs"), dtype=np.float64)
            score = float(e.get("score", float("nan")))
        except (TypeError, ValueError):
            dropped_other += 1
            continue
        if scs.shape != (geom.N, 3):
            dropped_other += 1
            continue
        if not np.isfinite(scs).all():
            dropped_nan += 1
            continue
        if not np.isfinite(score):
            dropped_nan += 1
            continue
        if score >= score_cap:
            dropped_other += 1
            continue
        kept.append(e)
    return kept, dropped_nan, dropped_other


def _select_top_k(entries: list[dict], k: int) -> list[dict]:
    """Top-K by score after dedup. archive_merge already dedups by sig_l2;
    here we just keep the K lowest-score entries."""
    return sorted(entries, key=lambda e: float(e["score"]))[:k]


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomic JSON write: temp -> flush -> fsync -> os.replace -> fsync parent.
    Mirrors mbh_driver._snapshot durability ordering. Propagates OSError so
    the supervisor can log a deliberate failure path rather than silently
    losing state on ENOSPC/EIO."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _clear_worker_artifacts(worker_dir: Path) -> None:
    """Remove stale STOP / done_mbh.flag / archive.json / events.jsonl from a
    reused worker dir before respawn. Without this, a leftover done flag from
    a prior attempt makes the next worker look already-complete to the
    supervisor's `_completion_count` poll."""
    for name in (STOP_FLAG_NAME, DONE_FLAG_NAME, "archive.json", "events.jsonl"):
        p = worker_dir / name
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _materialize_seed(entry: dict, path: Path) -> None:
    """Write a seed file matching pool/best.json schema for mbh_driver --incumbent."""
    scs = np.asarray(entry["scs"], dtype=np.float64).reshape(geom.N, 3)
    score = float(entry["score"])
    payload = {
        "score": score,
        "scs": scs.tolist(),
        "solution": [
            {"x": float(scs[i, 0]), "y": float(scs[i, 1]), "theta": float(scs[i, 2])}
            for i in range(geom.N)
        ],
    }
    _atomic_write_json(path, payload)


def _union_merge(
    previous_pool_path: Optional[Path],
    worker_archive_paths: list[Path],
    out_path: Path,
    slots: int = 64,
    min_l2: float = 0.08,
) -> tuple[int, int, int, float]:
    """D11 union: merge previous_pool ∪ worker archives via the same sig_l2
    dedup as basin_archive. D12: drop NaN/inf entries. Returns
    (size, dropped_nan_count, dropped_other_count, best_R)."""
    arch = basin_archive.BasinArchive(slots=slots, min_l2=min_l2)
    nan_total = 0
    other_total = 0
    sources: list[str] = []
    if previous_pool_path is not None and Path(previous_pool_path).exists():
        prev_entries = _load_pool(Path(previous_pool_path))
        kept, dn, do = _filter_finite(prev_entries)
        nan_total += dn
        other_total += do
        archive_merge._rebuild_into(arch, kept)
        sources.append(str(previous_pool_path))
    for p in worker_archive_paths:
        if not p.exists():
            continue
        try:
            entries = _load_pool(p)
        except (ValueError, json.JSONDecodeError):
            continue
        kept, dn, do = _filter_finite(entries)
        nan_total += dn
        other_total += do
        archive_merge._rebuild_into(arch, kept)
        sources.append(str(p))
    if arch.size() == 0:
        raise RuntimeError("union merge produced empty archive")
    best = float(arch.entries[0]["score"])
    payload = arch.payload("merged", best)
    payload["sources"] = sources
    _atomic_write_json(out_path, payload)
    return arch.size(), nan_total, other_total, best


@dataclass
class WorkerHandle:
    idx: int
    proc: subprocess.Popen
    worker_dir: Path
    seed_path: Path
    profile: str
    rng_seed: int
    respawned: bool = False
    stderr_file: Optional[IO[bytes]] = None


def _spawn_worker(
    epoch_dir: Path,
    worker_idx: int,
    seed_path: Path,
    profile: str,
    epoch_min: float,
    rng_seed: int,
    chaos_pulse_every_iters: int,
    repo_root: Path,
) -> tuple[subprocess.Popen, Path, IO[bytes]]:
    """Spawn an mbh_driver subprocess. Returns (proc, worker_dir, stderr_fh)."""
    worker_dir = epoch_dir / f"w{worker_idx}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    events_path = worker_dir / "events.jsonl"
    archive_path = worker_dir / "archive.json"
    stop_flag = worker_dir / STOP_FLAG_NAME
    cmd = [
        sys.executable,
        str(repo_root / "mbh_driver.py"),
        "--incumbent",
        str(seed_path),
        "--events",
        str(events_path),
        "--archive",
        str(archive_path),
        "--hours",
        f"{epoch_min / 60.0:.6f}",
        "--seed",
        str(rng_seed),
        "--stop-flag",
        str(stop_flag),
        "--profile",
        profile,
        "--chaos-pulse-every-iters",
        str(chaos_pulse_every_iters),
    ]
    env = os.environ.copy()
    for k in THREAD_CAP_ENV:
        env[k] = "1"
    stderr_fh = open(worker_dir / "stderr.log", "wb")
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=str(repo_root),
        start_new_session=True,
        preexec_fn=_set_pdeathsig_preexec,
        stdout=subprocess.DEVNULL,
        stderr=stderr_fh,
    )
    return proc, worker_dir, stderr_fh


def _done_flag_path(worker_dir: Path) -> Path:
    return worker_dir / DONE_FLAG_NAME


def _completion_count(workers: list[WorkerHandle]) -> int:
    """Count workers as complete only if BOTH done_mbh.flag is present AND
    the process has exited cleanly (returncode 0). A worker that wrote the
    done flag but then crashed would otherwise contaminate the merge with
    a possibly-truncated archive."""
    n = 0
    for w in workers:
        if not _done_flag_path(w.worker_dir).exists():
            continue
        ret = w.proc.poll()
        # Accept: process is gone with rc==0, OR process is still finishing
        # cleanup but already wrote done flag durably.
        if ret is None:
            # Brief wait for clean shutdown (done flag is written near end of run()).
            try:
                ret = w.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                # Process still alive after writing done flag — count as
                # complete-pending; will be SIGTERMed by stragglers if needed.
                n += 1
                continue
        if ret == 0:
            n += 1
    return n


def _fanout_stop(workers: list[WorkerHandle], log=print) -> list[WorkerHandle]:
    """Touch STOP in every worker dir. If touch fails (full disk, permissions),
    log it and return that worker so callers can SIGTERM it directly."""
    failed: list[WorkerHandle] = []
    for w in workers:
        try:
            (w.worker_dir / STOP_FLAG_NAME).touch()
        except OSError as exc:
            log(f"[orch] STOP fanout failed for w{w.idx}: {exc}; SIGTERM fallback")
            failed.append(w)
            if w.proc.poll() is None:
                try:
                    os.killpg(os.getpgid(w.proc.pid), signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass
    return failed


def _terminate_stragglers(workers: list[WorkerHandle], grace_s: float = 60.0) -> None:
    """Send SIGTERM to each still-alive worker's process group, wait grace
    period, then SIGKILL the rest."""
    for w in workers:
        if w.proc.poll() is None:
            try:
                os.killpg(os.getpgid(w.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if all(w.proc.poll() is not None for w in workers):
            break
        time.sleep(1.0)
    for w in workers:
        if w.proc.poll() is None:
            try:
                os.killpg(os.getpgid(w.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        if w.stderr_file is not None:
            try:
                w.stderr_file.close()
            except OSError:
                pass


STOP_GRACE_SECONDS = 30.0


def _wait_epoch(
    workers: list[WorkerHandle],
    epoch_dir: Path,
    out_dir: Path,
    timeout_s: float,
    chaos_pulse_every_iters: int,
    epoch_min: float,
    repo_root: Path,
    stop_holder: dict,
    log,
) -> None:
    """Poll until all workers finish (done flag) or epoch timeout. Detect
    crashes (exit non-zero, no done flag) and respawn ONCE per worker.
    Honors both the on-disk STOP file AND the in-memory SIGTERM/SIGINT
    latch (stop_holder['stop']).

    On STOP, the function fans out STOP files to worker dirs, then waits
    STOP_GRACE_SECONDS for workers to write their done flag via the
    stop_flag path before returning. This preserves the
    exit_reason='stop_flag' path so the merge boundary keeps clean
    archives instead of preempting them with SIGTERM."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        signal_stop = stop_holder.get("stop", False)
        file_stop = (out_dir / STOP_FLAG_NAME).exists()
        if signal_stop or file_stop:
            log(
                f"[orch] STOP detected (signal={signal_stop} file={file_stop}) "
                f"— fanning out to workers"
            )
            _fanout_stop(workers, log)
            # Grace window: let workers write done flags via stop_flag path.
            # We poll briefly so completed workers are recognized; only stragglers
            # past the grace window fall through to _terminate_stragglers SIGTERM.
            grace_deadline = time.time() + STOP_GRACE_SECONDS
            while time.time() < grace_deadline:
                if all(
                    _done_flag_path(w.worker_dir).exists() or w.proc.poll() is not None
                    for w in workers
                ):
                    log("[orch] all workers reported in within grace window")
                    return
                time.sleep(1.0)
            log(
                f"[orch] {STOP_GRACE_SECONDS:.0f}s grace expired; "
                f"completed={_completion_count(workers)}/{len(workers)}; "
                f"stragglers will be SIGTERMed"
            )
            return
        all_finished = True
        for w in workers:
            if _done_flag_path(w.worker_dir).exists():
                continue
            ret = w.proc.poll()
            if ret is None:
                all_finished = False
                continue
            # Process exited but no done flag → crashed
            if not w.respawned:
                log(
                    f"[orch] worker w{w.idx} crashed (exit={ret}, no done flag); "
                    f"respawning from same seed (rng_seed={w.rng_seed})"
                )
                if w.stderr_file is not None:
                    try:
                        w.stderr_file.close()
                    except OSError:
                        pass
                # Clear the worker dir's prior artifacts BEFORE respawn so the
                # respawned worker's done_mbh.flag write is unambiguous and
                # the new archive.json doesn't merge with the crashed one.
                _clear_worker_artifacts(w.worker_dir)
                new_proc, _wd, new_stderr = _spawn_worker(
                    epoch_dir,
                    w.idx,
                    w.seed_path,
                    w.profile,
                    epoch_min,
                    w.rng_seed,  # SAME seed — preserve deterministic retry contract
                    chaos_pulse_every_iters,
                    repo_root,
                )
                w.proc = new_proc
                w.respawned = True
                w.stderr_file = new_stderr
                all_finished = False
            else:
                log(f"[orch] worker w{w.idx} crashed twice (exit={ret}); giving up")
        if all_finished:
            return
        time.sleep(2.0)
    log(f"[orch] epoch timeout after {timeout_s:.0f}s")


@dataclass
class PromoteResult:
    promoted: bool
    reason: str
    candidate_R: Optional[float] = None
    incumbent_R: Optional[float] = None


def promote_if_better(
    merged_archive_path: Path,
    incumbent_path: Path = Path(DEFAULT_INCUMBENT),
    yes: bool = False,
    verify_cmd: tuple[str, ...] = ("node", "verify.mjs"),
    candidate_out: Path = Path("/tmp/island-candidate.json"),
) -> PromoteResult:
    """D14 single promotion gate. Sequential by construction:
      (1) harvest --dry-run sanity-checks the archive
      (2) compare candidate R to incumbent R; reject if not better
      (3) extract candidate to disk in pool/best.json schema
      (4) verify.mjs <candidate> must exit 0
      (5) prompt user (skip if yes=True)
      (6) harvest --yes
    Step (6) cannot run unless (1)-(5) succeed in order.
    """
    merged_archive_path = Path(merged_archive_path)
    incumbent_path = Path(incumbent_path)
    if not merged_archive_path.exists():
        return PromoteResult(False, f"archive not found: {merged_archive_path}")
    if not incumbent_path.exists():
        return PromoteResult(False, f"incumbent not found: {incumbent_path}")

    dry = subprocess.run(
        [
            sys.executable,
            "harvest.py",
            "--archive",
            str(merged_archive_path),
            "--incumbent",
            str(incumbent_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    if dry.returncode != 0:
        return PromoteResult(
            False, f"harvest --dry-run failed: {dry.stderr.strip()[:200]}"
        )

    with open(merged_archive_path) as f:
        arch = json.load(f)
    if not arch.get("entries"):
        return PromoteResult(False, "merged archive has no entries")
    cand_entry = arch["entries"][0]
    cand_scs = np.asarray(cand_entry["scs"], dtype=np.float64).reshape(geom.N, 3)
    cand_round = geom.rnd(cand_scs)
    if int(geom.cnt(cand_round)) != 0:
        return PromoteResult(False, "candidate fails rounded feasibility")
    cand_R = float(geom.mec(cand_round))
    with open(incumbent_path) as f:
        inc = json.load(f)
    inc_scs = np.asarray(inc["scs"], dtype=np.float64).reshape(geom.N, 3)
    inc_R = float(geom.mec(geom.rnd(inc_scs)))

    if cand_R >= inc_R:
        return PromoteResult(False, "no improvement", cand_R, inc_R)

    payload = {
        "score": cand_R,
        "scs": cand_round.tolist(),
        "solution": [
            {
                "x": float(cand_round[i, 0]),
                "y": float(cand_round[i, 1]),
                "theta": float(cand_round[i, 2]),
            }
            for i in range(geom.N)
        ],
    }
    candidate_out.parent.mkdir(parents=True, exist_ok=True)
    candidate_out.write_text(json.dumps(payload))

    ver = subprocess.run(
        list(verify_cmd) + [str(candidate_out)],
        capture_output=True,
        text=True,
    )
    if ver.returncode != 0:
        return PromoteResult(
            False,
            f"verify.mjs exit={ver.returncode}: {ver.stderr.strip()[:200]}",
            cand_R,
            inc_R,
        )

    if not yes:
        print(f"\nCandidate R = {cand_R:.12f}")
        print(f"Incumbent R = {inc_R:.12f}")
        print(f"Delta       = {inc_R - cand_R:+.3e}")
        print(f"Verify      = exit 0 ({candidate_out})")
        try:
            resp = input("Promote? [y/N] ").strip().lower()
        except EOFError:
            return PromoteResult(False, "no tty for confirmation", cand_R, inc_R)
        if resp != "y":
            return PromoteResult(False, "user declined", cand_R, inc_R)

    promote = subprocess.run(
        [
            sys.executable,
            "harvest.py",
            "--archive",
            str(merged_archive_path),
            "--incumbent",
            str(incumbent_path),
            "--yes",
        ],
        capture_output=True,
        text=True,
    )
    if promote.returncode != 0:
        return PromoteResult(
            False,
            f"harvest --yes failed: {promote.stderr.strip()[:200]}",
            cand_R,
            inc_R,
        )
    return PromoteResult(True, "promoted", cand_R, inc_R)


def _write_done_orchestrator(out_dir: Path, payload: dict) -> None:
    """Atomic write of done_orchestrator.flag. fsynced + parent-dir-fsynced
    so the supervisor's done-signal is durable across host crashes."""
    _atomic_write_json(out_dir / "done_orchestrator.flag", payload)


def _make_stop_handler(stop_holder: dict):
    def handler(signum, frame):
        stop_holder["stop"] = True

    return handler


def run_orchestrator(args, repo_root: Path, log_factory=None) -> int:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "supervisor.log"
    log_f = open(log_path, "a", buffering=1)

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"{ts} {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    if log_factory is not None:
        log = log_factory(log_path)  # noqa: F841 (test hook)

    stop_holder = {"stop": False}
    signal.signal(signal.SIGTERM, _make_stop_handler(stop_holder))
    signal.signal(signal.SIGINT, _make_stop_handler(stop_holder))

    pool_path = Path(args.seed_pool).resolve()
    if not pool_path.exists():
        log(f"[orch] FATAL: seed pool not found: {pool_path}")
        log_f.close()
        return 2

    log(
        f"[orch] start epochs={args.epochs} workers={args.workers} "
        f"epoch_min={args.epoch_min} top_k={args.top_k} "
        f"chaos_every={args.chaos_pulse_every_iters} pool={pool_path.name}"
    )

    t_start = time.time()
    current_pool_path: Path = pool_path
    final_merged_path: Path = current_pool_path
    epoch_summaries: list[dict] = []

    for epoch in range(args.epochs):
        if stop_holder["stop"] or (out_dir / STOP_FLAG_NAME).exists():
            log("[orch] stop detected before epoch start — exiting loop")
            break
        # Wall-time cap
        if (time.time() - t_start) > args.hours_cap * 3600.0:
            log(f"[orch] hours_cap={args.hours_cap}h hit — exiting loop")
            break
        log(f"[orch] === epoch {epoch} ===")
        try:
            entries = _load_pool(current_pool_path)
        except (ValueError, json.JSONDecodeError) as exc:
            log(f"[orch] FATAL: cannot read pool {current_pool_path}: {exc}")
            log_f.close()
            return 3
        kept, pool_nan, pool_other = _filter_finite(entries)
        if pool_nan or pool_other:
            log(
                f"[orch] filtered pool entries: dropped_nan_count={pool_nan} "
                f"dropped_other={pool_other}"
            )
        if not kept:
            log("[orch] FATAL: pool empty after finite filter")
            log_f.close()
            return 4
        top = _select_top_k(kept, args.top_k)
        epoch_dir = out_dir / f"epoch_{epoch}"
        seeds_dir = epoch_dir / "seeds"
        seeds_dir.mkdir(parents=True, exist_ok=True)
        seed_files: list[Path] = []
        for i, entry in enumerate(top):
            sp = seeds_dir / f"seed_{i}.json"
            _materialize_seed(entry, sp)
            seed_files.append(sp)
        log(
            f"[orch] materialized {len(seed_files)} seeds "
            f"(best R={top[0]['score']:.6f})"
        )

        workers: list[WorkerHandle] = []
        for w_idx in range(args.workers):
            seed_path = seed_files[w_idx % len(seed_files)]
            profile = PROFILES[w_idx % len(PROFILES)]
            rng_seed = args.seed + epoch * 1000 + w_idx
            # Clear stale artifacts from prior runs that reused this --out-dir.
            # Without this, a leftover done_mbh.flag would make a fresh worker
            # appear already-complete to _completion_count.
            _clear_worker_artifacts(epoch_dir / f"w{w_idx}")
            proc, worker_dir, stderr_fh = _spawn_worker(
                epoch_dir,
                w_idx,
                seed_path,
                profile,
                args.epoch_min,
                rng_seed,
                args.chaos_pulse_every_iters,
                repo_root,
            )
            log(
                f"[orch] spawned w{w_idx} pid={proc.pid} profile={profile} "
                f"seed={seed_path.name}"
            )
            workers.append(
                WorkerHandle(
                    idx=w_idx,
                    proc=proc,
                    worker_dir=worker_dir,
                    seed_path=seed_path,
                    profile=profile,
                    rng_seed=rng_seed,
                    stderr_file=stderr_fh,
                )
            )

        timeout_s = EPOCH_TIMEOUT_FACTOR * args.epoch_min * 60.0
        _wait_epoch(
            workers,
            epoch_dir,
            out_dir,
            timeout_s,
            args.chaos_pulse_every_iters,
            args.epoch_min,
            repo_root,
            stop_holder,
            log,
        )
        _terminate_stragglers(workers)
        completed = _completion_count(workers)
        log(f"[orch] epoch {epoch} completed: {completed}/{len(workers)} workers")

        worker_arch_paths = [w.worker_dir / "archive.json" for w in workers]
        merged_path = out_dir / f"merged_{epoch}.json"
        if completed >= MIN_COMPLETE_FRACTION * len(workers):
            try:
                size, dropped_nan, dropped_other, best = _union_merge(
                    current_pool_path,
                    worker_arch_paths,
                    merged_path,
                )
                log(
                    f"[orch] union merge: size={size} "
                    f"dropped_nan_count={dropped_nan} "
                    f"dropped_other={dropped_other} best_R={best:.6f}"
                )
                current_pool_path = merged_path
                final_merged_path = merged_path
            except RuntimeError as exc:
                log(f"[orch] union merge failed: {exc}; carry forward previous pool")
        else:
            log(
                f"[orch] failed epoch ({completed}/{len(workers)} < 50%); "
                f"carry forward previous pool"
            )

        epoch_summaries.append(
            {
                "epoch": epoch,
                "completed": completed,
                "total": len(workers),
                "pool_path": str(current_pool_path),
            }
        )

        if stop_holder["stop"] or (out_dir / STOP_FLAG_NAME).exists():
            log("[orch] stop detected after epoch — exiting loop")
            break

    log("[orch] final harvest dry-run:")
    try:
        dry = subprocess.run(
            [
                sys.executable,
                "harvest.py",
                "--archive",
                str(final_merged_path),
                "--incumbent",
                DEFAULT_INCUMBENT,
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(repo_root),
        )
        out_text = dry.stdout.strip()
        for line in out_text.splitlines():
            log(f"  | {line}")
        if dry.stderr.strip():
            log(f"stderr: {dry.stderr.strip()}")
    except (subprocess.SubprocessError, OSError) as exc:
        log(f"[orch] harvest dry-run failed: {exc}")

    elapsed = time.time() - t_start
    _write_done_orchestrator(
        out_dir,
        {
            "exit_reason": "stop" if stop_holder["stop"] else "complete",
            "elapsed_s": elapsed,
            "epochs_run": len(epoch_summaries),
            "final_pool": str(final_merged_path),
            "summaries": epoch_summaries,
        },
    )
    log(f"[orch] DONE elapsed={elapsed:.1f}s final_pool={final_merged_path}")
    log_f.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description="Island-model MBH orchestrator")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--epoch-min", type=float, required=True, help="minutes per epoch")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument(
        "--seed-pool", required=True, help="merged archive json (Phase 0 output)"
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chaos-pulse-every-iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--hours-cap",
        type=float,
        default=12.0,
        help="absolute wall-time cap for the whole run",
    )
    args = ap.parse_args()

    if args.epochs <= 0 or args.workers <= 0 or args.epoch_min <= 0 or args.top_k <= 0:
        print("ERROR: epochs/workers/epoch-min/top-k must be > 0", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent
    return run_orchestrator(args, repo_root)


if __name__ == "__main__":
    sys.exit(main())
