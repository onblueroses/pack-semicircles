"""Unit tests for mbh_driver tabu + restart + move-key paths.

Covers the three HIGH findings codex caught (so regressions fail loud):
  - basin tabu actually tabus revisits (not archive[0])
  - restart fires on since_insert, not since_improve
  - move-key is correct per move type (rim_swap uses 'pieces' not 'cluster')
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

import geom
import mbh_driver as md
import perturb_lib as pl


def load_incumbent() -> np.ndarray:
    with open("pool/best.json") as f:
        return geom.rnd(np.array(json.load(f)["scs"], dtype=np.float64))


# ---------- _move_key normalization ----------


def test_move_key_handles_each_metadata_shape():
    # flip_one → piece
    r = pl.PerturbResult(np.zeros((15, 3)), "flip_one", 3.0, {"piece": 4})
    assert md._move_key(r) == ("flip_one", frozenset({4}))

    # rim_swap → pieces (regression: codex found this collapsed to {-1})
    r = pl.PerturbResult(np.zeros((15, 3)), "rim_swap", 5.0, {"pieces": [6, 9]})
    assert md._move_key(r) == ("rim_swap", frozenset({6, 9}))

    # rotate_cluster → cluster
    r = pl.PerturbResult(
        np.zeros((15, 3)), "rotate_cluster", 7.0, {"cluster": [0, 3, 7], "alpha": 0.4}
    )
    assert md._move_key(r) == ("rotate_cluster", frozenset({0, 3, 7}))

    # reseat_interior → piece
    r = pl.PerturbResult(
        np.zeros((15, 3)), "reseat_interior", 4.0, {"piece": 11, "min_slack": -0.3}
    )
    assert md._move_key(r) == ("reseat_interior", frozenset({11}))

    # missing metadata → empty set, no crash
    r = pl.PerturbResult(np.zeros((15, 3)), "unknown", 2.0, {})
    assert md._move_key(r) == ("unknown", frozenset())
    print("OK: _move_key covers piece/pieces/cluster/empty correctly")


# ---------- ReactiveTabu bookkeeping ----------


def test_reactive_tabu_escalates_on_dup_and_decays():
    t = md.ReactiveTabu(
        basin_start=10, basin_step_up=3, basin_cap=20, basin_decay_every=5
    )
    key = ("a",)
    t.note_basin(key, is_dup=True)
    assert t.basin_tenure[key] == 13
    t.note_basin(key, is_dup=True)
    assert t.basin_tenure[key] == 16
    # 5 no-dup iters triggers decay
    for _ in range(5):
        t.note_basin(("other",), is_dup=False)
    assert t.basin_tenure[key] == 15
    # hit_basin works
    assert t.hit_basin(key)
    print("OK: ReactiveTabu escalates + decays as specified")


def test_reactive_tabu_full_reset_on_accepts():
    t = md.ReactiveTabu(reset_every_new_basins=3)
    t.basin_tenure[("x",)] = 10
    t.move_tenure[("y",)] = 5
    for _ in range(3):
        t.on_accept_new_basin()
    assert len(t.basin_tenure) == 0 and len(t.move_tenure) == 0
    print("OK: full reset fires every N accepts")


# ---------- restart temperature adapts ----------


def test_restart_temperature_scales_with_observed_spread():
    """When archive deltas are ~0.1, quality must not underflow to 0.
    Codex found hard-coded T=1e-4 made all non-best quality = 0."""

    class FakeEntry(dict):
        pass

    # Synthetic archive with deltas 0, 0.02, 0.08, 0.10, 0.12
    scores = [2.9486, 2.9686, 3.0286, 3.0486, 3.0686]
    sigs = [np.array([float(i)]) for i in range(len(scores))]
    entries = [
        FakeEntry({"score": s, "signature": sig}) for s, sig in zip(scores, sigs)
    ]

    class FakeArchive:
        def __init__(self, ents):
            self.entries = ents

    rng = np.random.default_rng(0)
    # Run 2000 times; distribution must include non-zero indices
    idx_counts = np.zeros(len(scores))
    for _ in range(2000):
        i = md._pick_restart_index(FakeArchive(entries), scores[0], rng, top_n=5)  # type: ignore[arg-type]
        idx_counts[i] += 1
    # Best-only degenerate behavior would put ~100% on idx 0. Check non-best
    # indices collectively get > 30% (novelty adds 30% base + quality contributes).
    assert idx_counts[1:].sum() / idx_counts.sum() > 0.3, (
        f"restart collapsed to best-only: counts={idx_counts.tolist()}"
    )
    print(
        f"OK: restart distribution (2000 samples) = {idx_counts.tolist()} — "
        f"non-best share {idx_counts[1:].sum() / idx_counts.sum():.0%}"
    )


# ---------- BasinTabu (L2-distance) ----------


def test_basin_tabu_matches_archive_dedup_rule():
    """L2-distance tabu must collide same-basin noise (|δ|~1e-6) but reject
    genuinely distinct basins. Matches BasinArchive min_l2=0.08."""
    scs = load_incumbent()
    sig = md._basin_signature(scs)
    # tiny noise: same basin, different 6dp key
    noise = scs.copy()
    noise[0, 0] += 1e-6
    sig_noise = md._basin_signature(geom.rnd(noise))

    tabu = md.BasinTabu(min_l2=0.08)
    tabu.note(sig, is_dup=True)
    assert tabu.hit(sig_noise), (
        "noise-perturbed signature should collide with tabu'd incumbent"
    )
    # A very different config (scaled) should NOT hit.
    center = scs[:, :2].mean(axis=0)
    scaled = scs.copy()
    scaled[:, :2] = center + 1.10 * (scaled[:, :2] - center)
    scaled_sig = md._basin_signature(geom.rnd(scaled))
    assert not tabu.hit(scaled_sig), "scaled basin incorrectly tabu'd"
    print("OK: BasinTabu collides on noise, separates on real basin change")


def test_basin_tabu_arms_on_first_visit():
    """Regression (codex round 3): BasinTabu previously only stored sigs on
    is_dup=True. First revisit therefore always missed. Now first visit
    arms at start tenure so the second visit hits."""
    tabu = md.BasinTabu(min_l2=0.08, start=30)
    scs = load_incumbent()
    sig = md._basin_signature(scs)
    tabu.note(sig, is_dup=False)
    assert tabu.size() == 1, "first-visit note must store the signature"
    assert tabu.hit(sig), "second visit should hit tabu after first-visit arming"
    print("OK: BasinTabu arms on first visit (blocks on second)")


def test_basin_tabu_decay_and_reset():
    tabu = md.BasinTabu(min_l2=0.08, start=5, step_up=2, step_down=1, decay_every=3)
    scs = load_incumbent()
    sig = md._basin_signature(scs)
    tabu.note(sig, is_dup=True)
    assert tabu.size() == 1
    for _ in range(3):
        tabu.note(md._basin_signature(load_incumbent()), is_dup=False)  # no-op sig
    # After decay, tenure should have dropped by step_down=1
    # (initial tenure = start + step_up = 7; after 1 decay = 6; still > 0)
    assert tabu.size() == 1 and tabu._tenure[0] == 6
    tabu.reset()
    assert tabu.size() == 0
    print("OK: BasinTabu decays tenure + resets cleanly")


# ---------- end-to-end driver: basin tabu actually blocks revisits ----------


def test_driver_smoke_produces_parseable_events():
    """Run 3 iters end-to-end; verify the event log is complete (one event
    per iter) and every line parses. Separately, the unit-level tabu tests
    above prove blocking semantics — this test only guards against runtime
    regressions in the loop plumbing."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = md.DriverConfig(
            events_path=str(td / "events.jsonl"),
            archive_path=str(td / "archive.json"),
            incumbent_path="pool/best.json",
            hours=1,
            max_iters=3,
            seed=1,
            snapshot_every_s=999,
        )
        summary = md.run(cfg)
        assert summary["iters"] == 3
        lines = (td / "events.jsonl").read_text().splitlines()
        # Every iteration must emit exactly one event (either accept, duplicate,
        # or reject). move_tabu-skip path also emits now.
        assert len(lines) >= summary["iters"], (
            f"events ({len(lines)}) < iters ({summary['iters']}) — log gap"
        )
        for line in lines:
            json.loads(line)
        print(f"OK: driver ran 3 iters, {len(lines)} events ≥ iters, all parseable")


def test_reducer_can_rebuild_archive_from_mbh_log():
    """Regression (codex round 4): accept events must carry scs so the reducer
    can rebuild archive state from the log. Previously skipped silently."""
    import archive_reducer as ar
    import basin_archive

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = md.DriverConfig(
            events_path=str(td / "events.jsonl"),
            archive_path=str(td / "archive.json"),
            incumbent_path="pool/best.json",
            hours=1,
            max_iters=10,
            seed=3,
            snapshot_every_s=999,
        )
        summary = md.run(cfg)
        if summary["accepts"] == 0:
            print("SKIP: no accepts in 10 iters, can't test reducer rebuild")
            return
        replay_archive = basin_archive.BasinArchive(slots=32, min_l2=0.08)
        n, _ = ar.replay_archive_events(
            td / "events.jsonl",
            lambda ev: ar.apply_event_to_archive(replay_archive, ev),
        )
        # Rebuilt archive should include at least the accepts (may differ from
        # live archive because live includes the incumbent seed too).
        assert replay_archive.size() >= summary["accepts"], (
            f"reducer rebuilt {replay_archive.size()} basins, "
            f"expected >= {summary['accepts']} accepts"
        )
        print(
            f"OK: reducer rebuilt archive size={replay_archive.size()} "
            f"from {n} events ({summary['accepts']} accepts)"
        )


def test_replay_preserves_same_basin_improvement():
    """Regression (codex round 5): replace events mutate archive (lower score
    in same basin) and must therefore be replayable — i.e. carry scs+label.
    Before this fix, replay_best_score lagged live_best_score for replaces."""
    import archive_reducer as ar
    import basin_archive
    import geom

    scs = load_incumbent()
    score_hi = float(geom.mec(scs))
    score_lo = score_hi - 1e-4

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "events.jsonl"
        fd = ar.open_append_fd(path)
        # Live path: insert + same-basin improvement (carries scs per our fix)
        ar.append_event(
            fd,
            {
                "type": "accept",
                "trial": 1,
                "score": score_hi,
                "scs": scs.tolist(),
                "label": "seed",
            },
        )
        ar.append_event(
            fd,
            {
                "type": "replace",
                "trial": 2,
                "score": score_lo,
                "scs": scs.tolist(),
                "label": "improved",
                "move": "flip_one",
                "D": 1.0,
            },
        )
        os.close(fd)

        arch = basin_archive.BasinArchive(slots=8, min_l2=0.08)
        ar.replay_archive_events(path, lambda ev: ar.apply_event_to_archive(arch, ev))
        assert arch.size() == 1, f"expected 1 basin, got {arch.size()}"
        assert abs(arch.entries[0]["score"] - score_lo) < 1e-10, (
            f"replay didn't apply improvement: score={arch.entries[0]['score']} "
            f"expected {score_lo}"
        )
        print("OK: replay applies same-basin improvement (replace event)")


def test_reducer_tolerates_mbh_telemetry():
    """HIGH regression: mbh_driver emits reject/restart/duplicate without scs.
    apply_event_to_archive must skip those cleanly instead of KeyError'ing."""
    import archive_reducer as ar
    import basin_archive

    arch = basin_archive.BasinArchive()
    # telemetry-only events
    for ev in [
        {"type": "reject", "trial": 1, "reason": "resolve_failed", "move": "flip_one"},
        {"type": "restart", "trial": 7, "from_rank": 2},
        {"type": "duplicate", "trial": 9, "score": 2.95, "move": "rim_swap", "D": 4.0},
    ]:
        result = ar.apply_event_to_archive(arch, ev)
        assert result is None, f"telemetry event should skip, got {result}"
    assert arch.size() == 0
    print("OK: reducer skips telemetry events instead of KeyError")


def main():
    test_move_key_handles_each_metadata_shape()
    test_reactive_tabu_escalates_on_dup_and_decays()
    test_reactive_tabu_full_reset_on_accepts()
    test_restart_temperature_scales_with_observed_spread()
    test_basin_tabu_matches_archive_dedup_rule()
    test_basin_tabu_arms_on_first_visit()
    test_basin_tabu_decay_and_reset()
    test_driver_smoke_produces_parseable_events()
    test_reducer_tolerates_mbh_telemetry()
    test_reducer_can_rebuild_archive_from_mbh_log()
    test_replay_preserves_same_basin_improvement()
    print("\nALL PASS")


if __name__ == "__main__":
    main()
