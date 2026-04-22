#!/usr/bin/env python3
"""Approach C: Packomania-seeded starts for the existing hetero_optimizer.

Writes three structured seed configurations to pool/seed_structured_{5,6,7}.json:

  k=5: 10 paired semicircles at regular-pentagon positions (r=1.7013),
       5 loners placed at gap angles on an outer ring.
  k=6: 12 paired semicircles at regular-hexagon positions (r=2.0),
       3 loners placed at gap angles on an outer ring.
  k=7: 14 paired semicircles (1 central + 6 hexagon at r=2.0),
       1 loner placed on an outer ring.

Each pair is two semicircles at the same center with opposite thetas,
together forming a unit disk. Loners have flat edges oriented outward.

Seeds typically have MEC in [3.0, 4.5] - hetero_optimizer is expected to
compress these qualitatively new starts.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import numpy as np

import geom
import common


POOL_DIR = "pool"
MEC_GATE = 4.5

# Packomania-optimal enclosing radii for unit disks. We add a 0.02 buffer
# to the disk-center ring so adjacent disks are slightly separated instead
# of exactly tangent - exact tangency triggers ov() edge cases.
_BUFFER = 0.02
# k=7 needs extra headroom because the center disk's pair offset (+/-0.01 along
# its d axis) leaves the nearest hex disk only 2.0 away at _BUFFER=0.02, which
# is exactly the arc-arc-tangent threshold that ov() flags.
R_K = {
    5: 1.0 + 1.0 / math.sin(math.pi / 5) + _BUFFER,
    6: 2.0 + _BUFFER,
    7: 2.0 + _BUFFER + 0.03,
}


_PAIR_OFFSET = 0.010


def paired_at(cx, cy, theta_outward):
    """Two semicircles near (cx,cy) forming a full-disk equivalent, offset along
    the theta axis so their flats are parallel but 2*_PAIR_OFFSET apart. A pure
    same-center pair with opposite thetas trips ov() after rnd rounding (the
    6-decimal theta quantization produces a ~1e-6 flat-flat asymmetry that the
    strict-cross test reads as a crossing). Offsetting along +/- theta_outward
    keeps the two arcs facing away from each other with no overlap in any
    contact mode."""
    dx = math.cos(theta_outward)
    dy = math.sin(theta_outward)
    return [
        (cx + _PAIR_OFFSET * dx, cy + _PAIR_OFFSET * dy, theta_outward),
        (
            cx - _PAIR_OFFSET * dx,
            cy - _PAIR_OFFSET * dy,
            (theta_outward + math.pi) % (2 * math.pi),
        ),
    ]


def disk_centers(k):
    """Packomania-optimal k-disk centers inside a unit enclosing disk,
    scaled so each disk has radius 1."""
    centers = []
    if k == 5:
        r = R_K[5] - 1.0
        for i in range(5):
            a = 2 * math.pi * i / 5
            centers.append((r * math.cos(a), r * math.sin(a)))
    elif k == 6:
        r = R_K[6]
        for i in range(6):
            a = math.pi * i / 3
            centers.append((r * math.cos(a), r * math.sin(a)))
    elif k == 7:
        centers.append((0.0, 0.0))
        r = R_K[7]
        for i in range(6):
            a = math.pi * i / 3
            centers.append((r * math.cos(a), r * math.sin(a)))
    else:
        raise ValueError(f"unsupported k={k}")
    return centers


def build_seed(k):
    """Build a (15, 3) config for the given k. Returns (scs, R_target)."""
    centers = disk_centers(k)
    n_loners = 15 - 2 * k

    sem = []
    for cx, cy in centers:
        theta_out = math.atan2(cy, cx) if (cx * cx + cy * cy) > 1e-9 else 0.0
        sem.extend(paired_at(cx, cy, theta_out))

    # Loners: place on an outer ring at gap angles.
    if n_loners > 0:
        r_outer, ok = _find_loner_ring(k, n_loners, np.array(sem))
        if k == 5:
            gap_offset = math.pi / 5  # halfway between pentagon spokes
            base_angles = [2 * math.pi * i / 5 + gap_offset for i in range(n_loners)]
        elif k == 6:
            gap_offset = math.pi / 6
            base_angles = [
                2 * math.pi * i / n_loners + gap_offset for i in range(n_loners)
            ]
        else:  # k == 7, n_loners == 1
            base_angles = [0.0]
        for a in base_angles:
            cx = r_outer * math.cos(a)
            cy = r_outer * math.sin(a)
            theta_out = math.atan2(cy, cx)
            sem.append((cx, cy, theta_out))

    scs = np.array(sem, dtype=np.float64)
    if scs.shape[0] != geom.N:
        raise RuntimeError(f"built {scs.shape[0]} semicircles, expected {geom.N}")
    return scs


def _find_loner_ring(k, n_loners, paired_scs):
    """Find the smallest outer ring radius where all loners fit without overlap."""
    r = R_K[k] + 0.1
    for _ in range(200):
        scs = paired_scs.copy()
        if k == 5:
            gap_offset = math.pi / 5
            angles = [2 * math.pi * i / n_loners + gap_offset for i in range(n_loners)]
        elif k == 6:
            gap_offset = math.pi / 6
            angles = [2 * math.pi * i / n_loners + gap_offset for i in range(n_loners)]
        else:
            angles = [0.0]
        loners = np.array(
            [
                [r * math.cos(a), r * math.sin(a), math.atan2(math.sin(a), math.cos(a))]
                for a in angles
            ]
        )
        full = np.vstack([scs, loners])
        if geom.cnt(geom.rnd(full)) == 0:
            return r, True
        r += 0.03
    return r, False


def validate(scs, label):
    rounded = geom.rnd(scs)
    c = int(geom.cnt(rounded))
    m = float(geom.mec(rounded))
    print(f"  {label}: mec={m:.4f} overlaps={c}")
    if c > 0:
        raise RuntimeError(f"{label}: {c} overlap(s) after rounding")
    if m > MEC_GATE:
        raise RuntimeError(f"{label}: mec {m:.4f} exceeds gate {MEC_GATE}")
    return m


# Slot mapping: hetero_optimizer.pool_read_diverse reads pool/diverse_{0..19}.json.
# Existing state uses slots 0-4. We park structured seeds in high slots so hot
# workers (which reinit from the diverse archive when stuck) can sample them.
_SLOT_BY_K = {5: 15, 6: 16, 7: 17}


def write_seed(k, scs):
    slot = _SLOT_BY_K[k]
    path = os.path.join(POOL_DIR, f"diverse_{slot}.json")
    data = {"score": float(geom.mec(geom.rnd(scs))), "scs": scs.tolist()}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    # Also keep a named copy for provenance / debugging.
    named = os.path.join(POOL_DIR, f"seed_structured_{k}.json")
    tmp2 = named + ".tmp"
    with open(tmp2, "w") as f:
        json.dump(data, f)
    os.replace(tmp2, named)
    return path


def prepare():
    os.makedirs(POOL_DIR, exist_ok=True)
    seeds = {}
    for k in (5, 6, 7):
        scs = build_seed(k)
        validate(scs, f"k={k}")
        path = write_seed(k, scs)
        seeds[k] = path
        print(f"  wrote {path}")
    return seeds


def run_hetero(hours, seeds):
    """Launch hetero_optimizer.py in this process via subprocess.
    Does not capture stdout - let tmux log."""
    env = os.environ.copy()
    env["NUMBA_NUM_THREADS"] = env.get("NUMBA_NUM_THREADS", "5")
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "5")
    print(
        f"[C] launching hetero_optimizer for {hours}h with seeds {list(seeds.keys())}"
    )
    result = subprocess.run(
        [sys.executable, "hetero_optimizer.py", str(hours)],
        env=env,
    )
    common.write_done_flag("structured")
    return result.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=40.0)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seeds = prepare()
    if args.prepare_only:
        print("[C] prepare-only: seeds written, skipping hetero launch.")
        return 0
    return run_hetero(args.hours, seeds)


if __name__ == "__main__":
    sys.exit(main() or 0)
