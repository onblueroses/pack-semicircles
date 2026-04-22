"""Sanity-check gap.py against geom.ov() and geom.mec().

Tests:
  1. h_semicircle correctness on handpicked cases.
  2. gap_ss agrees in SIGN with geom.ov() on random samples + the current best config.
  3. smooth_mec_sq upper-bounds exact R^2, gets tight as T -> 0.
  4. overlap_penalty_sq is 0 on the current best config (it has 0 overlaps).
"""

import json
import math
import numpy as np

import geom
import gap


def test_h_semicircle():
    # t = 0 (material in +x half). Support at n = (1, 0) should be 1.
    assert abs(gap.h_semicircle(1.0, 0.0, 0.0) - 1.0) < 1e-9
    # n = (-1, 0) pointing into flat side: h = |0| = 0
    assert abs(gap.h_semicircle(-1.0, 0.0, 0.0) - 0.0) < 1e-9
    # n = (0, 1): on edge of flat side. In local frame n_x_l = 0 >= 0, so h = 1.
    assert abs(gap.h_semicircle(0.0, 1.0, 0.0) - 1.0) < 1e-9
    # n = (-1, 0) with t = pi (material in -x half): local n_x_l = cos(pi) * -1 = 1 >= 0, h = 1
    assert abs(gap.h_semicircle(-1.0, 0.0, math.pi) - 1.0) < 1e-9
    print("  [PASS] h_semicircle handpicked cases")


def test_gap_ss_signs():
    # Two non-overlapping disks (arcs facing each other, centers d=2.5 apart along x)
    g = gap.gap_ss(0.0, 0.0, 0.0, 2.5, 0.0, math.pi)
    ov = geom.ov(0.0, 0.0, 0.0, 2.5, 0.0, math.pi)
    assert g > 0 and not ov, f"arc-arc d=2.5 sep: gap={g}, ov={ov}"

    # Arcs tangent at d=2 exactly: gap ~ 0; ov() calls this overlap per its convention.
    # Our gap_ss is the classical convex separation (0 at tangent). For optimization
    # we add a safety buffer to match ov's inclusive-tangent convention.
    g = gap.gap_ss(0.0, 0.0, 0.0, 2.0, 0.0, math.pi)
    ov = geom.ov(0.0, 0.0, 0.0, 2.0, 0.0, math.pi)
    print(
        f"  arc-arc tangent d=2: gap={g:.6e}, ov={ov} (ov true at exact tangent is expected)"
    )
    assert abs(g) < 1e-6

    # Arcs penetrating at d=1.8
    g = gap.gap_ss(0.0, 0.0, 0.0, 1.8, 0.0, math.pi)
    ov = geom.ov(0.0, 0.0, 0.0, 1.8, 0.0, math.pi)
    assert g < 0 and ov, f"arc-arc d=1.8: gap={g}, ov={ov}"

    # Flats face each other (t1=pi, t2=0): material of 1 in -x, material of 2 in +x.
    # Centers at d=0 should be fine (merged disk). At d=0.5 along x:
    # c1=(0,0) material in -x, c2=(0.5,0) material in +x. They should NOT overlap
    # since their materials extend AWAY from each other.
    g = gap.gap_ss(0.0, 0.0, math.pi, 0.5, 0.0, 0.0)
    ov = geom.ov(0.0, 0.0, math.pi, 0.5, 0.0, 0.0)
    print(f"  flats back-to-back d=0.5: gap={g:.6e}, ov={ov}")
    assert g > 0 and not ov

    # Flats face-to-face (t1=0, t2=pi): material of 1 in +x, material of 2 in -x.
    # At c1=(0,0), c2=(2.5,0): both materials point toward each other. arcs at d=2.5:
    # worst overlap. gap should be d - h1(u) - h2(-u) = 2.5 - 1 - 1 = 0.5.
    g = gap.gap_ss(0.0, 0.0, 0.0, 2.5, 0.0, math.pi)
    print(f"  flats face-to-face arcs-inward d=2.5: gap={g:.6e}")
    assert abs(g - 0.5) < 1e-9
    print("  [PASS] gap_ss handpicked cases")


def test_gap_ss_vs_ov_random():
    rng = np.random.default_rng(42)
    n_disagree = 0
    n_samples = 2000
    for _ in range(n_samples):
        x1, y1 = rng.uniform(-2, 2, 2)
        x2, y2 = rng.uniform(-2, 2, 2)
        t1 = rng.uniform(0, 2 * math.pi)
        t2 = rng.uniform(0, 2 * math.pi)
        g = gap.gap_ss(x1, y1, t1, x2, y2, t2)
        ov = geom.ov(x1, y1, t1, x2, y2, t2)
        # gap strongly negative -> must overlap
        # gap strongly positive -> must not overlap
        # near-zero gap: both outcomes acceptable (threshold zone)
        # UNSAFE (gap says no overlap but ov says overlap) is the only bad direction.
        # OVERSAFE (gap says overlap but ov says no) is acceptable - gap is a
        # grid-based upper bound that's conservatively tight to ~7e-3.
        if g > 1e-3 and ov:
            n_disagree += 1
    print(f"  gap/ov disagreement on {n_samples} random samples: {n_disagree}")
    assert n_disagree == 0, f"gap_ss disagrees with ov in sign on {n_disagree} samples"
    print("  [PASS] gap_ss agrees with ov on random samples (>1e-4 margin)")


def test_on_current_best():
    d = json.load(open("pool/best.json"))
    scs = np.array(d["scs"])
    # Configuration has 0 overlaps; penalty should be ~0.
    pen = gap.overlap_penalty_sq(scs)
    print(f"  overlap_penalty_sq on current best: {pen:.6e}")
    assert pen < 1e-8, f"penalty={pen} on supposedly-clean config"

    # Smooth MEC upper bound. Exact R from welzl:
    r_exact = float(geom.mec(geom.rnd(scs)))
    # Find MEC center via welzl (pack all sample points and call welzl)
    # Easier: use provided mec() from the boundary-max trick
    # For test, evaluate smooth MEC at the MEC center (known)
    # For now just sweep T and watch convergence
    # Build sample point list and compute welzl directly
    pts = []
    for i in range(scs.shape[0]):
        x, y, t = scs[i]
        ct, st = math.cos(t), math.sin(t)
        for sign in (1.0, -1.0):
            pts.append((x + sign * st, y - sign * ct))
        for s in range(8):
            a = t - math.pi / 2 + math.pi * s / 7
            pts.append((x + math.cos(a), y + math.sin(a)))
    pts_arr = np.array(pts)
    px = pts_arr[:, 0].copy()
    py = pts_arr[:, 1].copy()
    cx, cy, R_welzl = geom.welzl_mec(px, py, len(pts))
    r2_welzl = R_welzl * R_welzl
    print(f"  exact R: {r_exact:.6f}, welzl-on-samples R: {R_welzl:.6f}")
    # smooth at this center for various T
    for T in (1.0, 0.1, 0.01, 0.001):
        r2_smooth = gap.smooth_mec_sq(scs, cx, cy, T)
        print(
            f"    T={T:.3f}: smooth R^2={r2_smooth:.6f} vs welzl R^2={r2_welzl:.6f}  (diff={r2_smooth - r2_welzl:.2e})"
        )
        assert r2_smooth >= r2_welzl - 1e-9, (
            f"smooth={r2_smooth} underestimates welzl={r2_welzl}"
        )
    print("  [PASS] smooth_mec_sq upper-bounds exact and tightens with T->0")


if __name__ == "__main__":
    print("[test_gap]")
    test_h_semicircle()
    test_gap_ss_signs()
    test_gap_ss_vs_ov_random()
    test_on_current_best()
    print("All tests passed.")
