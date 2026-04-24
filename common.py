"""Shared I/O and utility helpers for the basin-finding trio.

Pure Python - no numba. Keeps the jit boundary clean.
"""

import json
import os
import time
import numpy as np

import geom

POOL_DIR = "pool"
BEST_FILE = "semicircle_best.json"
SOLUTION_FILE = "solution.json"


def _round_validate_and_score(scs):
    rounded = geom.rnd(np.asarray(scs, dtype=np.float64))
    if geom.cnt(rounded) > 0:
        return None, None
    return float(geom.mec(rounded)), rounded


def load_best():
    """Read best solution from pool/best.json -> semicircle_best.json -> solution.json.

    Returns (score, scs) or (None, None).
    Re-scores from coordinates rather than trusting persisted score fields.
    """
    for path in [os.path.join(POOL_DIR, "best.json"), BEST_FILE, SOLUTION_FILE]:
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict) and "scs" in data:
                score, scs = _round_validate_and_score(data["scs"])
                if score is not None:
                    return score, scs
            if isinstance(data, list):
                scs = [[s["x"], s["y"], s["theta"]] for s in data]
                score, rounded = _round_validate_and_score(scs)
                if score is not None:
                    return score, rounded
        except Exception:
            continue
    return None, None


def save_named_best(name, score, scs_rounded):
    """Atomically write best_<name>.json with the same schema as pool/best.json.

    name: short identifier (e.g., 'contact', 'md', 'structured').
    scs_rounded: (N,3) array already passed through geom.rnd.
    """
    exact_score, rounded = _round_validate_and_score(scs_rounded)
    if exact_score is None or rounded is None:
        return False
    data = {
        "score": exact_score,
        "scs": rounded.tolist(),
        "solution": [
            {
                "x": float(rounded[i, 0]),
                "y": float(rounded[i, 1]),
                "theta": float(rounded[i, 2]),
            }
            for i in range(geom.N)
        ],
    }
    dst = f"best_{name}.json"
    tmp = dst + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, dst)
    return True


def write_json_atomic(path, data):
    """Atomically write JSON payload to an arbitrary path."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return True


def write_done_flag(name):
    """Touch done_<name>.flag to signal graceful completion."""
    with open(f"done_{name}.flag", "w") as f:
        f.write(str(time.time()))


def timeout_reached(t0, hours):
    """True if wall-clock elapsed since t0 (seconds) has exceeded hours."""
    return (time.time() - t0) >= hours * 3600.0


def log_line(prefix, msg):
    """One-line timestamped log to stdout, flushed immediately."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"{ts} [{prefix}] {msg}", flush=True)
