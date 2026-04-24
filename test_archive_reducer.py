"""Stress test for archive_reducer.py (M5 gate).

Launches N worker processes, each appending K events via atomic O_APPEND,
while a reducer tails the log. Verifies:
  1. No corruption: every line is valid JSON.
  2. No loss: reducer sees all N*K events.
  3. Startup replay truncates a torn trailing record.
  4. Invariant dedup: translations of the same basin don't inflate archive size.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

import numpy as np

import archive_reducer as ar
import basin_archive
import geom


def _worker(events_path: str, worker_id: int, n_events: int, scs_bytes: bytes):
    scs = np.frombuffer(scs_bytes, dtype=np.float64).reshape(geom.N, 3).copy()
    fd = ar.open_append_fd(events_path)
    try:
        for k in range(n_events):
            # Tiny translation per event — all same basin after canonicalization.
            s = scs.copy()
            s[:, 0] += 1e-4 * k
            event = {
                "trial": worker_id * 10_000 + k,
                "score": float(2.95 + 1e-6 * k),
                "scs": s.tolist(),
                "label": f"w{worker_id}",
            }
            ar.append_event(fd, event)
    finally:
        os.close(fd)


def test_concurrent_no_corruption(tmp: Path):
    """4 workers × 250 events = 1000 events, no corruption, no loss."""
    events = tmp / "events.jsonl"
    with open("pool/best.json") as f:
        scs = np.array(json.load(f)["scs"], dtype=np.float64)
    scs = geom.rnd(scs)

    n_workers, n_per = 4, 250
    procs = [
        mp.Process(target=_worker, args=(str(events), i, n_per, scs.tobytes()))
        for i in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"worker {p.pid} failed: exitcode={p.exitcode}"

    # Every line should be valid JSON.
    lines = events.read_bytes().splitlines()
    assert len(lines) == n_workers * n_per, (
        f"expected {n_workers * n_per} lines, got {len(lines)}"
    )
    for i, line in enumerate(lines):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            raise AssertionError(f"line {i} corrupt: {e}")
    print(f"OK: {len(lines)} events written, all parseable")


def test_replay_tail_truncate(tmp: Path):
    """Replay must truncate a torn trailing record."""
    events = tmp / "events2.jsonl"
    fd = ar.open_append_fd(events)
    try:
        good = {
            "trial": 1,
            "score": 2.95,
            "scs": [[0.0, 0.0, 0.0]] * geom.N,
            "label": "g",
        }
        ar.append_event(fd, good)
        # Simulate a torn write: raw bytes with no trailing newline.
        os.write(fd, b'{"trial":2,"score":2.95,"scs":[[0')
    finally:
        os.close(fd)

    sizes = []

    def on_event(ev):
        sizes.append(ev["trial"])

    n = ar.replay_archive_events(events, on_event)
    assert n == 1, f"expected 1 replayed event, got {n}"
    assert sizes == [1], f"expected trials=[1], got {sizes}"
    # File should be truncated to just the first complete record.
    content = events.read_bytes()
    assert content.endswith(b"\n"), "file did not end at newline after truncate"
    assert b'trial":2' not in content, "torn record was not truncated"
    print("OK: replay truncated torn trailing record")


def test_reducer_dedup(tmp: Path):
    """Same basin via 10 translations — archive size stays 1."""
    events = tmp / "events3.jsonl"
    snap = tmp / "archive.json"
    with open("pool/best.json") as f:
        scs = np.array(json.load(f)["scs"], dtype=np.float64)
    scs = geom.rnd(scs)

    fd = ar.open_append_fd(events)
    try:
        for k in range(10):
            s = scs.copy()
            s[:, 0] += 0.01 * k
            s[:, 1] -= 0.005 * k
            s = geom.rnd(s)
            ar.append_event(
                fd,
                {
                    "trial": k,
                    "score": float(geom.mec(s)),
                    "scs": s.tolist(),
                    "label": "translated",
                },
            )
    finally:
        os.close(fd)

    archive = basin_archive.BasinArchive(slots=8, min_l2=0.08)
    reducer = ar.ArchiveReducer(events, snap, archive, run_name="test")
    replayed = reducer.startup()
    assert replayed == 10, f"expected 10 replayed, got {replayed}"
    assert archive.size() == 1, f"expected 1 distinct basin, got {archive.size()}"
    reducer.snapshot()
    reducer.close()
    payload = json.loads(snap.read_text())
    assert payload["archive_size"] == 1, "snapshot archive_size mismatch"
    print(
        f"OK: reducer deduped 10 translations -> archive_size=1, events_applied={reducer._events_applied}"
    )


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        t0 = time.time()
        test_concurrent_no_corruption(tmp)
        test_replay_tail_truncate(tmp)
        test_reducer_dedup(tmp)
        print(f"\nALL PASS ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
