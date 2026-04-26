"""Unit tests for the chaos pulse iteration knob in mbh_driver.

Spec: pack-semi-island-search Phase 1 Step 1.4. Three invariants:
  (a) chaos branch is a no-op when chaos_pulse_every_iters=0 (default).
  (b) chaos branch fires on the configured cadence and emits well-formed
      events; any accepted basin is feasible (no NaN, correct shape, cnt==0).
  (c) zero-kick chaos cannot lower the best score below the seeded incumbent
      (no archive corruption from the chaos path).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

import geom
import mbh_driver as md


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _make_cfg(tmp: Path, **overrides) -> md.DriverConfig:
    return md.DriverConfig(
        events_path=str(tmp / "events.jsonl"),
        archive_path=str(tmp / "archive.json"),
        incumbent_path="pool/best.json",
        max_iters=overrides.pop("max_iters", 12),
        seed=overrides.pop("seed", 12345),
        snapshot_every_s=999.0,
        **overrides,
    )


def test_chaos_off_emits_no_chaos_events():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_cfg(Path(tmp), max_iters=20, chaos_pulse_every_iters=0)
        md.run(cfg)
        events = _read_events(Path(tmp) / "events.jsonl")
        chaos = [e for e in events if e.get("type") == "chaos_pulse"]
        assert chaos == [], f"expected no chaos events, got {len(chaos)}"


def test_chaos_fires_when_enabled_and_emits_valid_events():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_cfg(
            Path(tmp),
            max_iters=12,
            chaos_pulse_every_iters=3,
            chaos_kick_xy=0.05,
            chaos_n_pieces=4,
        )
        md.run(cfg)
        events = _read_events(Path(tmp) / "events.jsonl")
        chaos = [e for e in events if e.get("type") == "chaos_pulse"]
        # Trials 3,6,9,12 -> at least 3 events (some may be reached, depends on
        # actual iters reached before max_iters; allow >= 3).
        assert len(chaos) >= 3, f"expected >=3 chaos events, got {len(chaos)}"
        for e in chaos:
            assert "trial" in e and e["trial"] > 0
            assert isinstance(e["accepted"], bool)
            assert e["kick"] == 0.05
            assert e["n_pieces"] == 4
            if e["accepted"]:
                scs = np.asarray(e["scs"], dtype=np.float64)
                assert scs.shape == (geom.N, 3)
                assert np.isfinite(scs).all()
                assert int(geom.cnt(scs)) == 0


def test_chaos_zero_kick_cannot_lower_best_below_incumbent():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_cfg(
            Path(tmp),
            max_iters=10,
            chaos_pulse_every_iters=2,
            chaos_kick_xy=0.0,
            chaos_n_pieces=4,
        )
        md.run(cfg)
        # Incumbent R from pool/best.json (rounded, as the driver does).
        with open("pool/best.json") as f:
            scs_inc = geom.rnd(np.asarray(json.load(f)["scs"], dtype=np.float64))
        R_inc = float(geom.mec(scs_inc))

        # Zero-kick chaos cannot drop below incumbent (it just runs resolve on
        # an already-feasible config). Any chaos accept event with score below
        # incumbent would mean the chaos path corrupted archive bookkeeping.
        events = _read_events(Path(tmp) / "events.jsonl")
        for e in events:
            if e.get("type") == "chaos_pulse" and e.get("accepted"):
                assert e["score"] >= R_inc - 1e-9, (
                    f"chaos with kick=0 reported R={e['score']} < incumbent {R_inc}"
                )


def test_done_flag_written_on_max_iters_exit():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_cfg(Path(tmp), max_iters=3, chaos_pulse_every_iters=0)
        md.run(cfg)
        flag = Path(tmp) / "done_mbh.flag"
        assert flag.exists(), "done_mbh.flag missing after graceful exit"
        payload = json.loads(flag.read_text())
        assert payload["exit_reason"] == "max_iters"
        assert payload["iters"] >= 1
        assert "final_R" in payload
        assert "archive_size" in payload
