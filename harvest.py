"""Promote an archive.json entry to pool/best.json if it beats the incumbent.

The MBH driver writes runs/<tag>/archive.json with a ranked list of basins.
This harvester:
  1. Loads the best entry from archive.json.
  2. Compares against pool/best.json's score.
  3. If strictly better AND verify.mjs-compatible (geom.cnt==0 after rnd),
     writes pool/best.json atomically and emits a diff summary.

Does nothing destructive on its own — always prompts (or takes --yes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import geom


def _load(path: Path) -> tuple[float, np.ndarray]:
    d = json.load(open(path))
    if "entries" in d:
        # archive.json from mbh_driver
        best = d["entries"][0]
        return float(best["score"]), np.array(best["scs"], dtype=np.float64)
    if "scs" in d and "score" in d:
        # pool/best.json or similar
        return float(d["score"]), np.array(d["scs"], dtype=np.float64)
    raise ValueError(f"unknown archive format: {path}")


def _rescore(scs: np.ndarray) -> tuple[float, np.ndarray]:
    r = geom.rnd(scs)
    if int(geom.cnt(r)) != 0:
        raise ValueError("candidate fails rounded feasibility (geom.cnt != 0)")
    return float(geom.mec(r)), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="runs/.../archive.json")
    ap.add_argument("--incumbent", default="pool/best.json")
    ap.add_argument("--yes", action="store_true", help="skip confirmation")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arch_path = Path(args.archive)
    inc_path = Path(args.incumbent)
    if not arch_path.exists():
        print(f"ERROR: archive not found: {arch_path}", file=sys.stderr)
        sys.exit(1)
    if not inc_path.exists():
        print(f"ERROR: incumbent not found: {inc_path}", file=sys.stderr)
        sys.exit(1)

    cand_score, cand_scs = _load(arch_path)
    inc_score, inc_scs = _load(inc_path)

    # Re-score both from coordinates (don't trust persisted score fields)
    cand_score_rescored, cand_rounded = _rescore(cand_scs)
    inc_score_rescored, _ = _rescore(inc_scs)

    delta = inc_score_rescored - cand_score_rescored
    print(f"incumbent:  R = {inc_score_rescored:.12f}  ({inc_path})")
    print(f"candidate:  R = {cand_score_rescored:.12f}  ({arch_path})")
    print(f"delta (inc - cand): {delta:+.3e}")

    if cand_score_rescored >= inc_score_rescored:
        print("NO PROMOTION: candidate not better than incumbent")
        return 0

    if args.dry_run:
        print("DRY RUN — not writing")
        return 0

    if not args.yes:
        resp = input("Promote candidate to pool/best.json? [y/N] ").strip().lower()
        if resp != "y":
            print("aborted")
            return 1

    payload = {
        "score": cand_score_rescored,
        "scs": cand_rounded.tolist(),
        "solution": [
            {
                "x": float(cand_rounded[i, 0]),
                "y": float(cand_rounded[i, 1]),
                "theta": float(cand_rounded[i, 2]),
            }
            for i in range(geom.N)
        ],
    }
    tmp = inc_path.with_suffix(inc_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(inc_path)
    print(f"PROMOTED: {inc_path} <- {arch_path} (R={cand_score_rescored:.12f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
