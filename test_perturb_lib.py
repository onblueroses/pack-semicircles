"""Smoke + D-band tests for perturb_lib (M6 gate partial).

Runs each move 30× on the incumbent and reports D histogram, in-band rate,
and average latency. The driver consumes in-band fraction to re-tune weights.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np

import geom
import perturb_lib as pl


def load_incumbent() -> tuple[np.ndarray, float]:
    with open("pool/best.json") as f:
        scs = np.array(json.load(f)["scs"], dtype=np.float64)
    scs = geom.rnd(scs)
    return scs, float(geom.mec(scs))


def bench_move(name: str, scs, R, rng, n=30):
    ds, in_band, elapsed = [], 0, 0.0
    for _ in range(n):
        t0 = time.time()
        # Force the move_type so we can histogram each bucket independently.
        res = pl.propose(scs, R, rng=rng, move_type=name, max_retries=1)
        elapsed += time.time() - t0
        ds.append(res.D)
        if res.metadata.get("in_band", False):
            in_band += 1
    arr = np.array(ds)
    return {
        "name": name,
        "n": n,
        "D_mean": float(arr.mean()),
        "D_median": float(np.median(arr)),
        "D_min": float(arr.min()),
        "D_max": float(arr.max()),
        "in_band_rate": in_band / n,
        "avg_ms": 1000 * elapsed / n,
    }


def test_moves_produce_nonzero_d():
    """Each move must actually damage the incumbent (no silent no-ops).
    Averages over 10 samples so a single unlucky pick doesn't flunk."""
    scs, _ = load_incumbent()
    R = 2.9486936795
    rng = np.random.default_rng(42)
    for name in pl.MOVES:
        ds = []
        changed = False
        for _ in range(10):
            res = pl.propose(scs, R, rng=rng, move_type=name, max_retries=1)
            assert res.scs.shape == scs.shape, f"{name}: bad shape"
            assert res.D >= 0, f"{name}: negative D {res.D}"
            assert res.move_type == name
            ds.append(res.D)
            if not np.array_equal(res.scs, scs):
                changed = True
        assert max(ds) > 0, f"{name}: never damaged incumbent in 10 samples"
        assert changed, f"{name}: returned unmodified config across 10 samples"
    print("OK: each move damages the incumbent at least once in 10 samples")


def test_weighted_d_zero_on_identity():
    scs, _ = load_incumbent()
    R = 2.9486936795
    assert pl.weighted_d(scs, scs, R) == 0.0, "identity D should be 0"
    print("OK: identity perturbation has D=0")


def test_distribution():
    scs, _ = load_incumbent()
    R = 2.9486936795
    rng = np.random.default_rng(0)
    histograms = []
    for name in pl.MOVES:
        histograms.append(bench_move(name, scs, R, rng, n=30))
    print("\nPer-move D histogram (30 samples each):")
    print(
        f"{'move':<18} {'mean':>6} {'med':>6} {'min':>6} {'max':>6} {'in-band':>8} {'ms':>6}"
    )
    for h in histograms:
        print(
            f"{h['name']:<18} {h['D_mean']:>6.2f} {h['D_median']:>6.2f} "
            f"{h['D_min']:>6.2f} {h['D_max']:>6.2f} "
            f"{h['in_band_rate'] * 100:>7.0f}% {h['avg_ms']:>6.0f}"
        )
    # Sanity: every move must damage at least once and land in-band at least once
    # (a core move that's always out-of-band would be silently dead in the pilot).
    for h in histograms:
        assert h["D_max"] > 0, f"{h['name']}: zero damage across 30 samples"
        assert h["in_band_rate"] > 0, (
            f"{h['name']}: never landed in D-band [{pl.D_BAND[0]},{pl.D_BAND[1]}] "
            f"in 30 samples — move is dead for this incumbent"
        )


def test_scheduler_in_band_acceptance():
    scs, _ = load_incumbent()
    R = 2.9486936795
    rng = np.random.default_rng(7)
    counts = defaultdict(int)
    in_band = 0
    n = 60
    for _ in range(n):
        res = pl.propose(scs, R, rng=rng, max_retries=6)
        counts[res.move_type] += 1
        if res.metadata.get("in_band", False):
            in_band += 1
    rate = in_band / n
    print(
        f"\nScheduler: {n} proposals, in-band rate={rate * 100:.0f}%, mix={dict(counts)}"
    )
    # M6-lean acceptance: ≥40% in-band over the mix (re-calibrated post-pilot).
    assert rate >= 0.4, f"in-band rate too low: {rate:.2f}"


def test_contact_surgery_pre_resolved_flag():
    """Non-fallback contact_surgery results carry pre_resolved=True so the
    driver knows to skip its global resolve. Fallback path keeps move_type
    intact (per Codex P3 — fallback must not masquerade as another move)."""
    scs, _ = load_incumbent()
    R = 2.9486936795
    rng = np.random.default_rng(13)
    n_real, n_fallback = 0, 0
    for _ in range(30):
        res = pl.move_contact_surgery(scs, R, rng)
        assert res.move_type == "contact_surgery", (
            f"fallback masqueraded as {res.move_type}"
        )
        if res.metadata.get("fallback"):
            n_fallback += 1
            assert not res.metadata.get("pre_resolved", False)
            assert res.D == 0.0
        else:
            n_real += 1
            assert res.metadata.get("pre_resolved") is True
            assert res.D > 0
            assert "active_size" in res.metadata
    assert n_real >= 5, (
        f"contact_surgery produced only {n_real}/30 real moves (rest fallbacks). "
        "Move is dead for this incumbent — relax CONTACT_SURGERY_TOL or extend "
        "tangency constructions."
    )
    print(f"OK: contact_surgery: {n_real} real / {n_fallback} fallback in 30 samples")


def main():
    t0 = time.time()
    test_moves_produce_nonzero_d()
    test_weighted_d_zero_on_identity()
    test_distribution()
    test_scheduler_in_band_acceptance()
    test_contact_surgery_pre_resolved_flag()
    print(f"\nALL PASS ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
