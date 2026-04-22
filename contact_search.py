#!/usr/bin/env python3
"""Approach A: Topology-aware basin hopping for semicircle packing.

Differs from MCMC and existing basin-hopping optimizers by structuring the
perturbation around the contact topology rather than random-walk kicks:

  1. Extract near-contact pairs (center distance < CONTACT_R) from the incumbent.
  2. Propose a STRUCTURAL perturbation: break a contact (push two semicircles
     apart along c2-c1) or form a contact (pull a non-contact pair together).
  3. Add small random jitter to all coordinates to escape symmetries.
  4. Inner polish: greedy per-index perturbation with exact ov() rejection.
  5. Accept on MEC improvement.

Center distance is used only for the PERTURBATION (natural topology metric).
All overlap checks use the exact ov() function (bit-for-bit matches verify.mjs).
"""

import argparse
import json
import math
import time
import numpy as np
import numba as nb

import geom
import common

CONTACT_R = 2.1
SEPARATED_R = 2.3
KICK_DIR = 0.3
KICK_JITTER = 0.04
KICK_THETA = 0.10


@nb.njit(cache=True)
def resolve_overlaps(scs, max_iters, step, seed):
    """Accept only moves that reduce overlap count. Used when the inner
    polish starts from a configuration with overlaps (typical after a
    topology perturbation). Returns (scs, remaining_overlap_count)."""
    np.random.seed(seed)
    c = geom.cnt(scs)
    for it in range(max_iters):
        if c == 0:
            return scs, 0
        idx = np.random.randint(0, geom.N)
        ox = scs[idx, 0]
        oy = scs[idx, 1]
        ot = scs[idx, 2]
        scs[idx, 0] = ox + np.random.randn() * step
        scs[idx, 1] = oy + np.random.randn() * step
        scs[idx, 2] = ot + np.random.randn() * step * 0.5
        c_new = geom.cnt(scs)
        if c_new <= c:
            c = c_new
        else:
            scs[idx, 0] = ox
            scs[idx, 1] = oy
            scs[idx, 2] = ot
    return scs, c


@nb.njit(cache=True)
def greedy_polish(scs_init, iters, step0, seed):
    """Greedy coord-descent with exact ov() rejection. MEC monotonically decreases."""
    np.random.seed(seed)
    scs = scs_init.copy()
    if geom.cnt(scs) > 0:
        return scs, 1e9
    score = geom.mec(scs)
    best = scs.copy()
    best_score = score
    step = step0
    stale = 0
    for it in range(iters):
        idx = np.random.randint(0, geom.N)
        ox = scs[idx, 0]
        oy = scs[idx, 1]
        ot = scs[idx, 2]
        scs[idx, 0] = ox + np.random.randn() * step
        scs[idx, 1] = oy + np.random.randn() * step
        scs[idx, 2] = ot + np.random.randn() * step * 0.3
        if geom.chk(scs, idx):
            scs[idx, 0] = ox
            scs[idx, 1] = oy
            scs[idx, 2] = ot
            stale += 1
        else:
            ns = geom.mec(scs)
            if ns < score:
                score = ns
                stale = 0
                if score < best_score:
                    best_score = score
                    best = scs.copy()
            else:
                scs[idx, 0] = ox
                scs[idx, 1] = oy
                scs[idx, 2] = ot
                stale += 1
        if stale > 5000:
            step *= 0.6
            stale = 0
        if step < 5e-6:
            step = 5e-6
    return best, best_score


def extract_contacts(scs, r=CONTACT_R):
    """Pairs with center distance < r. These define the topology."""
    contacts = []
    for i in range(geom.N):
        for j in range(i + 1, geom.N):
            dx = scs[i, 0] - scs[j, 0]
            dy = scs[i, 1] - scs[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < r:
                contacts.append((i, j, d))
    return contacts


def extract_non_contacts(scs, r=SEPARATED_R):
    """Pairs with center distance > r. Candidates for new contact formation."""
    pairs = []
    for i in range(geom.N):
        for j in range(i + 1, geom.N):
            dx = scs[i, 0] - scs[j, 0]
            dy = scs[i, 1] - scs[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            if d > r:
                pairs.append((i, j, d))
    return pairs


def perturb_topology(scs, rng, break_prob=0.5):
    """Structural perturbation: break or form one near-contact.

    Returns (perturbed_scs, description_str).
    """
    contacts = extract_contacts(scs)
    non_contacts = extract_non_contacts(scs)
    out = scs.copy()

    if contacts and (not non_contacts or rng.random() < break_prob):
        i, j, d = contacts[rng.integers(len(contacts))]
        vx = out[j, 0] - out[i, 0]
        vy = out[j, 1] - out[i, 1]
        norm = math.sqrt(vx * vx + vy * vy) + 1e-9
        vx /= norm
        vy /= norm
        out[j, 0] += vx * KICK_DIR
        out[j, 1] += vy * KICK_DIR
        desc = f"break({i},{j}|d={d:.3f})"
    elif non_contacts:
        i, j, d = non_contacts[rng.integers(len(non_contacts))]
        vx = out[j, 0] - out[i, 0]
        vy = out[j, 1] - out[i, 1]
        norm = math.sqrt(vx * vx + vy * vy) + 1e-9
        vx /= norm
        vy /= norm
        out[j, 0] -= vx * KICK_DIR
        out[j, 1] -= vy * KICK_DIR
        desc = f"form({i},{j}|d={d:.3f})"
    else:
        desc = "noop"

    out[:, 0] += rng.normal(0, KICK_JITTER, geom.N)
    out[:, 1] += rng.normal(0, KICK_JITTER, geom.N)
    out[:, 2] += rng.normal(0, KICK_THETA, geom.N)
    return out, desc


def run(hours, from_file, seed, out_name, polish_iters=60_000):
    rng = np.random.default_rng(seed)
    with open(from_file) as f:
        d = json.load(f)
    scs = np.array(d["scs"], dtype=np.float64)
    if scs.shape != (geom.N, 3):
        raise SystemExit(f"bad seed shape {scs.shape}")
    if geom.cnt(geom.rnd(scs)) > 0:
        raise SystemExit("seed has overlaps")

    best_scs = scs.copy()
    best_score = float(geom.mec(best_scs))
    common.log_line("A", f"seed score {best_score:.6f}")
    common.log_line(
        "A",
        f"contacts={len(extract_contacts(scs))} non={len(extract_non_contacts(scs))}",
    )

    t0 = time.time()
    n_trials = 0
    n_accepts = 0
    last_log = t0

    while not common.timeout_reached(t0, hours):
        n_trials += 1
        proposed, desc = perturb_topology(best_scs, rng)
        resolved, remaining = resolve_overlaps(
            proposed.copy(), 40_000, 0.03, rng.integers(1 << 30)
        )
        if remaining > 0:
            continue
        polished, _ = greedy_polish(
            resolved, polish_iters, 0.008, rng.integers(1 << 30)
        )
        rounded = geom.rnd(polished)
        if geom.cnt(rounded) > 0:
            continue
        score = float(geom.mec(rounded))
        if score < best_score - 1e-7:
            best_score = score
            best_scs = polished.copy()
            n_accepts += 1
            common.save_named_best(out_name, score, rounded)
            common.log_line("A", f"trial {n_trials} {desc} -> ACCEPT {score:.6f}")
        now = time.time()
        if now - last_log > 30:
            common.log_line(
                "A",
                f"trial {n_trials} acc={n_accepts} best={best_score:.6f} elapsed={(now - t0) / 3600:.2f}h",
            )
            last_log = now

    common.save_named_best(out_name, best_score, geom.rnd(best_scs))
    common.write_done_flag("contact")
    common.log_line(
        "A", f"done trials={n_trials} accepts={n_accepts} best={best_score:.6f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=40.0)
    ap.add_argument("--from", dest="from_file", default="semicircle_best.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="contact")
    ap.add_argument("--polish-iters", type=int, default=60_000)
    args = ap.parse_args()
    run(args.hours, args.from_file, args.seed, args.out, args.polish_iters)


if __name__ == "__main__":
    main()
