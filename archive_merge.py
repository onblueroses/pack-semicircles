"""Merge multiple MBH archive.json files into a single deduplicated archive.

For multi-worker / multi-machine pilots: each worker writes its own
runs/<tag>/archive.json with up-to-32 distinct basins. After the run,
merge them by signature L2 (same criterion BasinArchive uses) and keep
the top-N by score.

Usage:
  python archive_merge.py runs/a/archive.json runs/b/archive.json \
    --out runs/merged.json --slots 64 --min-l2 0.08
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import basin_archive
import geom


def _load_entries(path: Path) -> list[dict]:
    d = json.load(open(path))
    if "entries" not in d:
        raise ValueError(f"not an archive file: {path}")
    return d["entries"]


def _rebuild_into(archive: basin_archive.BasinArchive, entries: list[dict]) -> int:
    """Re-feed entries into a BasinArchive (its consider() handles dedup/replace)."""
    n_inserts = 0
    for e in entries:
        scs = np.asarray(e["scs"], dtype=np.float64)
        rounded = geom.rnd(scs)
        if int(geom.cnt(rounded)) != 0:
            print(f"  skip infeasible entry (trial={e.get('trial', '?')})")
            continue
        score = float(geom.mec(rounded))
        result = archive.consider(
            rounded, score, trial=e.get("trial", -1), label=e.get("label", "merged")
        )
        if result is not None and result.get("action") == "insert":
            n_inserts += 1
    return n_inserts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archives", nargs="+", help="archive.json files to merge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--min-l2", type=float, default=0.08)
    args = ap.parse_args()

    merged = basin_archive.BasinArchive(slots=args.slots, min_l2=args.min_l2)
    total = 0
    for p in args.archives:
        entries = _load_entries(Path(p))
        n = _rebuild_into(merged, entries)
        total += n
        print(f"[merge] {p}: {len(entries)} entries → +{n} new basins")

    # Write merged archive
    if merged.size() == 0:
        print("WARN: merged archive is empty")
        return 1
    best_score = merged.entries[0]["score"]
    payload = merged.payload("merged", best_score)
    payload["sources"] = args.archives
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"[merge] DONE: {merged.size()} distinct basins → {args.out} "
        f"(best R={best_score:.9f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
