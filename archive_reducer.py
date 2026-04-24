"""Single-reducer JSONL archive for MBH workers.

Workers emit candidate events via `append_event` (one record = one os.write
syscall on an O_APPEND fd — atomic under Linux ext4 for payloads ≤ PAGE_SIZE).
Reducer process replays the log on startup (tail-truncating any torn trailing
record), then tails for new events, applies them to an in-memory BasinArchive,
and snapshots to pool/archive.json periodically.

Crash semantics:
  - O_APPEND places bytes atomically at EOF *if* the syscall completes. Short
    writes on regular files are possible (ENOSPC, signals) — we raise, not
    retry, since a partial record is corrupt.
  - Not crash-atomic without RWF_ATOMIC + hardware support. Startup replay
    therefore truncates any trailing partial line back to the last '\n'.
  - Mid-file JSON corruption raises RuntimeError — manual repair required.
  - WSL: only safe on Linux-backed paths (`~/`), not `/mnt/c`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

import basin_archive


MAX_EVENT_BYTES = 4096  # PAGE_SIZE; Linux ext4 atomic ceiling for O_APPEND
FSYNC_EVERY_EVENTS = 50
FSYNC_EVERY_SECONDS = 5.0
SNAPSHOT_EVERY_SECONDS = 30.0


def open_append_fd(path: str | os.PathLike) -> int:
    """Open a file for atomic append. Creates if missing."""
    return os.open(
        os.fspath(path),
        os.O_WRONLY | os.O_APPEND | os.O_CREAT,
        0o644,
    )


def append_event(fd: int, event: dict) -> None:
    """Append one JSON event as a single atomic os.write.

    Records must be ≤ MAX_EVENT_BYTES so the syscall fits in a PAGE_SIZE write
    (atomic under O_APPEND on Linux ext4). Short writes raise — a partial
    record is corrupt and must not be retried.
    """
    payload = (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")
    if len(payload) > MAX_EVENT_BYTES:
        raise ValueError(
            f"event too large for atomic append: {len(payload)} > {MAX_EVENT_BYTES}"
        )
    n = os.write(fd, payload)
    if n != len(payload):
        raise IOError(
            f"short write on archive log: {n} < {len(payload)}; archive may be corrupt"
        )


def replay_archive_events(path: str | os.PathLike, on_event) -> tuple[int, int]:
    """Read JSONL events on reducer startup, creating file if missing and
    truncating any torn trailing record. Mid-file corruption raises.

    Returns (n_events_replayed, post_replay_byte_offset). The offset is the
    file size as of the moment replay finished reading — callers must seed
    their tail position from this value, not from a fresh SEEK_END, or a
    race window between replay and tail will silently drop worker events.

    NOTE: concurrent appends during the truncation window can themselves be
    truncated. Caller must coordinate writers to start only after replay
    returns (e.g. via a ready-flag file).
    """
    fd = os.open(os.fspath(path), os.O_RDWR | os.O_CREAT, 0o644)
    count = 0
    with os.fdopen(fd, "rb+") as f:
        content = f.read()
        if not content:
            return 0, 0
        last_newline = content.rfind(b"\n")
        if last_newline < 0:
            # Only partial first record; truncate.
            f.seek(0)
            f.truncate()
            return 0, 0
        if last_newline != len(content) - 1:
            # Torn trailing record; truncate to last complete line.
            f.seek(last_newline + 1)
            f.truncate()
            content = content[: last_newline + 1]
        for line in content.decode("utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"corrupt archive event mid-file in {path}: {exc}"
                ) from exc
            on_event(event)
            count += 1
    return count, len(content)


def apply_event_to_archive(archive: basin_archive.BasinArchive, event: dict):
    """Default event applier: feeds {scs, score, trial, label} to archive.consider."""
    scs = np.asarray(event["scs"], dtype=np.float64)
    return archive.consider(
        scs,
        float(event["score"]),
        int(event.get("trial", -1)),
        label=event.get("label", "worker"),
    )


class ArchiveReducer:
    """Replays + tails a JSONL event log, applies to BasinArchive, snapshots JSON."""

    def __init__(
        self,
        events_path: str | os.PathLike,
        snapshot_path: str | os.PathLike,
        archive: basin_archive.BasinArchive,
        run_name: str = "mbh",
        snapshot_every: float = SNAPSHOT_EVERY_SECONDS,
        fsync_every_events: int = FSYNC_EVERY_EVENTS,
        fsync_every_seconds: float = FSYNC_EVERY_SECONDS,
    ):
        self.events_path = Path(events_path)
        self.snapshot_path = Path(snapshot_path)
        self.archive = archive
        self.run_name = run_name
        self.snapshot_every = snapshot_every
        self.fsync_every_events = fsync_every_events
        self.fsync_every_seconds = fsync_every_seconds

        self._read_fd: int | None = None
        self._fsync_fd: int | None = None
        self._pos = 0
        self._leftover = b""
        self._events_applied = 0
        self._events_since_fsync = 0
        self._last_fsync_ts = 0.0
        self._last_snapshot_ts = 0.0
        self._best_score = float("inf")

    def startup(self) -> int:
        replayed, post_bytes = replay_archive_events(
            self.events_path, lambda ev: self._apply(ev)
        )
        self._read_fd = os.open(self.events_path, os.O_RDONLY)
        # Seed tail offset from replay's post-read file length, NOT SEEK_END —
        # otherwise any append that lands between replay and SEEK_END is lost.
        self._pos = post_bytes
        try:
            self._fsync_fd = os.open(self.events_path, os.O_WRONLY | os.O_APPEND)
        except OSError:
            # Partial-init leak guard: close the read fd if fsync fd fails.
            os.close(self._read_fd)
            self._read_fd = None
            raise
        now = time.time()
        self._last_fsync_ts = now
        self._last_snapshot_ts = now
        return replayed

    def _apply(self, event: dict) -> None:
        apply_event_to_archive(self.archive, event)
        self._events_applied += 1
        self._events_since_fsync += 1
        score = float(event.get("score", float("inf")))
        if score < self._best_score:
            self._best_score = score

    def tick(self) -> int:
        """Read any new bytes, apply complete records, maybe fsync+snapshot.

        Fail-fast semantics: a tail-phase JSON parse error or apply exception
        raises RuntimeError. The reducer is the single authoritative writer of
        archive state; silently dropping a consumed byte range would desync the
        in-memory archive from the replayable log.
        """
        assert self._read_fd is not None and self._fsync_fd is not None
        new_count = 0
        os.lseek(self._read_fd, self._pos, os.SEEK_SET)
        chunk = os.read(self._read_fd, 1 << 20)
        if chunk:
            data = self._leftover + chunk
            last_nl = data.rfind(b"\n")
            if last_nl < 0:
                self._leftover = data
                self._pos += len(chunk)
            else:
                complete = data[: last_nl + 1]
                new_leftover = data[last_nl + 1 :]
                for line in complete.decode("utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"corrupt archive event in tail: {exc}"
                        ) from exc
                    try:
                        self._apply(event)
                    except Exception as exc:
                        raise RuntimeError(
                            f"failed to apply archive event: {exc}"
                        ) from exc
                    new_count += 1
                # Only commit position + leftover after all records applied cleanly.
                self._leftover = new_leftover
                self._pos += len(chunk)

        now = time.time()
        if self._events_since_fsync >= self.fsync_every_events or (
            self._events_since_fsync > 0
            and now - self._last_fsync_ts >= self.fsync_every_seconds
        ):
            # fdatasync failure must NOT reset the counter — otherwise unsynced
            # events are silently forgotten and later snapshots become unsafe.
            os.fdatasync(self._fsync_fd)
            self._last_fsync_ts = now
            self._events_since_fsync = 0

        if now - self._last_snapshot_ts >= self.snapshot_every:
            self.snapshot()
            self._last_snapshot_ts = now

        return new_count

    def _sync_log(self) -> None:
        """Force the event log to stable storage. Required before every snapshot
        so a reader can never observe a snapshot that includes events missing
        from the replayable log."""
        if self._fsync_fd is None:
            return
        os.fdatasync(self._fsync_fd)
        self._last_fsync_ts = time.time()
        self._events_since_fsync = 0

    def snapshot(self) -> None:
        # Durability ordering: sync the log FIRST. If log sync raises, don't
        # publish a snapshot that names events the log can't prove.
        self._sync_log()
        payload = self.archive.payload(self.run_name, self._best_score)
        payload["events_applied"] = self._events_applied
        tmp = self.snapshot_path.with_suffix(self.snapshot_path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        # Crash-durable write: fsync tmp, rename, fsync parent dir.
        with open(tmp, "w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.snapshot_path)
        dir_fd = os.open(self.snapshot_path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def run_until(self, stop_flag_path: str | os.PathLike, poll_s: float = 0.1) -> None:
        stop = Path(stop_flag_path)
        while not stop.exists():
            self.tick()
            time.sleep(poll_s)
        # Drain any remaining events, then final snapshot.
        while self.tick() > 0:
            pass
        self.snapshot()

    def close(self) -> None:
        for fd in (self._read_fd, self._fsync_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._read_fd = None
        self._fsync_fd = None


def main() -> None:
    """CLI: archive-reducer reads events, writes snapshot. Stop via --stop-flag file."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="JSONL event log path")
    ap.add_argument("--snapshot", required=True, help="archive.json snapshot path")
    ap.add_argument("--stop-flag", required=True, help="touch this file to stop")
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--min-l2", type=float, default=0.08)
    ap.add_argument("--run-name", default="mbh")
    args = ap.parse_args()

    archive = basin_archive.BasinArchive(slots=args.slots, min_l2=args.min_l2)
    reducer = ArchiveReducer(
        events_path=args.events,
        snapshot_path=args.snapshot,
        archive=archive,
        run_name=args.run_name,
    )
    replayed = reducer.startup()
    print(f"[reducer] replayed {replayed} events from {args.events}")
    try:
        reducer.run_until(args.stop_flag)
    finally:
        reducer.close()
    print(
        f"[reducer] stopped; applied={reducer._events_applied} best={reducer._best_score:.6f}"
    )


if __name__ == "__main__":
    main()
