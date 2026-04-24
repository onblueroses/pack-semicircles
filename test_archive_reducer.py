"""Stress test for archive_reducer.py (M5 gate).

Runs as a standalone script (`python test_archive_reducer.py`) and is also
pytest-compatible (`pytest -q test_archive_reducer.py`): uses the built-in
`tmp_path` fixture, which pytest auto-provides and the standalone `main()`
supplies manually via tempfile.TemporaryDirectory.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import tempfile
import threading
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


def test_concurrent_no_corruption(tmp_path: Path):
    """4 workers × 250 events with reducer tailing: no corruption, no loss.

    Reducer drains concurrently with workers and must observe every trial id.
    """
    events = tmp_path / "events.jsonl"
    snap = tmp_path / "archive.json"
    with open("pool/best.json") as f:
        scs = np.array(json.load(f)["scs"], dtype=np.float64)
    scs = geom.rnd(scs)

    n_workers, n_per = 4, 250
    # Touch the file so reducer can open it, then start reducer BEFORE workers —
    # this is the same startup handshake the driver will use.
    ar.open_append_fd(events)  # creates empty file
    os.close(_ := ar.open_append_fd(events))

    archive = basin_archive.BasinArchive(slots=8, min_l2=0.08)
    reducer = ar.ArchiveReducer(
        events, snap, archive, run_name="test", snapshot_every=1.0
    )
    reducer.startup()

    procs = [
        mp.Process(target=_worker, args=(str(events), i, n_per, scs.tobytes()))
        for i in range(n_workers)
    ]
    stop = threading.Event()
    observed_trials: set[int] = set()

    def _capture_apply(event):
        observed_trials.add(int(event["trial"]))

    reducer._apply = (  # type: ignore[method-assign]
        lambda ev, orig=reducer._apply: (orig(ev), _capture_apply(ev))[0]
    )

    def _drain():
        while not stop.is_set():
            reducer.tick()
            time.sleep(0.01)

    t = threading.Thread(target=_drain)
    t.start()
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0, f"worker {p.pid} failed: exitcode={p.exitcode}"
        # Give reducer a moment to drain the tail.
        deadline = time.time() + 10.0
        expected = n_workers * n_per
        while time.time() < deadline and reducer._events_applied < expected:
            time.sleep(0.05)
    finally:
        stop.set()
        t.join(timeout=5.0)
        reducer.close()

    lines = events.read_bytes().splitlines()
    assert len(lines) == n_workers * n_per, (
        f"expected {n_workers * n_per} lines on disk, got {len(lines)}"
    )
    for i, line in enumerate(lines):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            raise AssertionError(f"line {i} corrupt: {e}")
    assert reducer._events_applied == n_workers * n_per, (
        f"reducer applied {reducer._events_applied}, expected {n_workers * n_per}"
    )
    expected_trials = {i * 10_000 + k for i in range(n_workers) for k in range(n_per)}
    assert observed_trials == expected_trials, (
        f"reducer missed {len(expected_trials - observed_trials)} trials, "
        f"extra {len(observed_trials - expected_trials)}"
    )
    print(
        f"OK: {len(lines)} events on disk, reducer applied all "
        f"({reducer._events_applied}), trial set matches exactly"
    )


def test_replay_tail_truncate(tmp_path: Path):
    """Replay must truncate a torn trailing record and return the post-replay byte offset."""
    events = tmp_path / "events2.jsonl"
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

    n, post = ar.replay_archive_events(events, on_event)
    assert n == 1, f"expected 1 replayed event, got {n}"
    assert sizes == [1], f"expected trials=[1], got {sizes}"
    content = events.read_bytes()
    assert content.endswith(b"\n"), "file did not end at newline after truncate"
    assert b'trial":2' not in content, "torn record was not truncated"
    assert post == len(content), (
        f"post-replay offset {post} != file length {len(content)}"
    )
    print("OK: replay truncated torn trailing record; post_bytes correct")


def test_startup_post_replay_offset_prevents_race(tmp_path: Path):
    """Regression: an append that lands between replay returning and tail
    beginning must still be visible. The fix is seeding _pos from replay's
    post-read byte count rather than SEEK_END. We simulate by appending after
    startup() returns and checking tick() sees the event."""
    events = tmp_path / "events3.jsonl"
    snap = tmp_path / "archive.json"
    with open("pool/best.json") as f:
        scs = np.array(json.load(f)["scs"], dtype=np.float64)
    scs = geom.rnd(scs)

    fd = ar.open_append_fd(events)
    try:
        ar.append_event(
            fd,
            {
                "trial": 1,
                "score": float(geom.mec(scs)),
                "scs": scs.tolist(),
                "label": "pre",
            },
        )
    finally:
        os.close(fd)

    archive = basin_archive.BasinArchive(slots=8, min_l2=0.08)
    reducer = ar.ArchiveReducer(events, snap, archive, run_name="test")
    replayed = reducer.startup()
    assert replayed == 1

    fd = ar.open_append_fd(events)
    try:
        s2 = scs.copy()
        s2[:, 0] += 0.5
        ar.append_event(
            fd,
            {
                "trial": 2,
                "score": float(geom.mec(geom.rnd(s2))),
                "scs": geom.rnd(s2).tolist(),
                "label": "post",
            },
        )
    finally:
        os.close(fd)

    consumed = reducer.tick()
    assert consumed == 1, f"post-startup append not seen in tail (consumed={consumed})"
    assert reducer._events_applied == 2
    reducer.close()
    print("OK: startup→tail handoff captures post-replay appends")


def test_reducer_dedup(tmp_path: Path):
    """Same basin via 10 translations — archive size stays 1."""
    events = tmp_path / "events4.jsonl"
    snap = tmp_path / "archive.json"
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
        f"OK: reducer deduped 10 translations -> archive_size=1, "
        f"events_applied={reducer._events_applied}"
    )


def test_tail_corruption_fails_loud(tmp_path: Path):
    """Tail-phase JSON parse error must raise, not silently advance _pos."""
    events = tmp_path / "events5.jsonl"
    snap = tmp_path / "archive.json"

    ar.open_append_fd(events)
    os.close(_ := ar.open_append_fd(events))
    archive = basin_archive.BasinArchive(slots=4, min_l2=0.08)
    reducer = ar.ArchiveReducer(events, snap, archive)
    reducer.startup()

    fd = ar.open_append_fd(events)
    try:
        os.write(fd, b"not json\n")
    finally:
        os.close(fd)

    try:
        reducer.tick()
    except RuntimeError as exc:
        assert "corrupt archive event" in str(exc), str(exc)
        print("OK: tail corruption raises RuntimeError (fail-loud)")
        reducer.close()
        return
    finally:
        reducer.close()
    raise AssertionError("tail corruption did not raise")


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        t0 = time.time()
        for i, (name, fn) in enumerate(
            [
                ("replay_tail_truncate", test_replay_tail_truncate),
                (
                    "startup_post_replay_offset",
                    test_startup_post_replay_offset_prevents_race,
                ),
                ("reducer_dedup", test_reducer_dedup),
                ("tail_corruption_fails_loud", test_tail_corruption_fails_loud),
                ("concurrent_no_corruption", test_concurrent_no_corruption),
            ]
        ):
            sub = tmp / f"sub{i}_{name}"
            sub.mkdir()
            fn(sub)
        print(f"\nALL PASS ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
