"""Print a one-shot health summary of an MBH pilot run.

Reads both events.jsonl and archive.json from a run directory and reports:
  - iterations + accepts + archive size + best score
  - reject breakdown (counts per reason)
  - move breakdown (proposals, accepts, rejects per move type)
  - D-histogram percentiles
  - iter rate

Usage: python pilot_status.py runs/mbh-4h-.../
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()

    run = Path(args.run_dir)
    events = run / "events.jsonl"
    archive = run / "archive.json"
    if not events.exists():
        print(f"ERROR: {events} not found", file=sys.stderr)
        sys.exit(1)

    reasons: Counter[str] = Counter()
    move_proposals: Counter[str] = Counter()
    move_accepts: Counter[str] = Counter()
    ds_per_move: defaultdict[str, list[float]] = defaultdict(list)
    total_iters = 0
    total_accepts = 0
    for line in events.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        typ = e.get("type")
        move = e.get("move", "?")
        if typ == "accept":
            total_accepts += 1
            move_accepts[move] += 1
        elif typ == "reject":
            reasons[e.get("reason", "?")] += 1
        if typ in ("accept", "reject", "duplicate", "replace"):
            move_proposals[move] += 1
            total_iters += 1
        d = e.get("D")
        if d is not None:
            ds_per_move[move].append(float(d))

    print(f"=== pilot status: {run} ===")
    if archive.exists():
        a = json.loads(archive.read_text())
        print(
            f"snapshot: iters={a.get('iters')} accepts={a.get('accepts')} "
            f"archive_size={a.get('archive_size')} best={a.get('best_score'):.9f}"
        )
        stats_keys = [
            "resolve_failed",
            "stage_a_failed",
            "tabu_skips",
            "restarts",
            "since_insert",
        ]
        stat_str = " ".join(f"{k}={a.get(k, '?')}" for k in stats_keys)
        print(f"counters: {stat_str}")
    print(f"events  : iters={total_iters} accepts={total_accepts}")

    print("\n-- reject breakdown --")
    for r, n in reasons.most_common():
        print(f"  {r:<30} {n:4d}")

    print("\n-- move breakdown --")
    print(f"  {'move':<18} {'proposals':>9} {'accepts':>8} {'accept%':>8}")
    for m in sorted(move_proposals.keys()):
        p = move_proposals[m]
        a_ = move_accepts[m]
        pct = 100.0 * a_ / p if p else 0.0
        print(f"  {m:<18} {p:>9d} {a_:>8d} {pct:>7.0f}%")

    print("\n-- D distribution (proposed) --")
    for m, ds in sorted(ds_per_move.items()):
        if not ds:
            continue
        arr = np.array(ds)
        print(
            f"  {m:<18} n={len(ds):4d}  p50={np.median(arr):5.1f}  "
            f"p90={np.percentile(arr, 90):5.1f}  "
            f"min={arr.min():4.1f}  max={arr.max():5.1f}"
        )


if __name__ == "__main__":
    main()
