"""Monotonic Basin Hopping driver for 15-semicircle MEC packing.

Single-process outer loop. On each iteration:
  1. Pick current config (archive-best, or weighted restart if stalled).
  2. Perturb via perturb_lib (flip/reseat/rim-swap/rotate-cluster).
  3. Overlap-resolve via continuous penalty (L-BFGS-B, ≤500 iters).
  4. Stage A active-set Newton (attack4.stage_a) on the resolved config.
  5. Round + validate; feed to BasinArchive; emit JSONL event.
  6. Update reactive tabu and restart counters.

Tabu semantics (plan v4 §1.5):
  - basin-key tabu: tenure starts 30; +5 per duplicate signature; -1 per 50
    no-dup cycles; cap 60. Duplicate = L2 to existing signature < min_l2.
  - move-signature tabu: tenure starts 60; +10 per duplicate; -2 per 50 no-dup;
    cap 100. Signature = (move_type, frozenset of affected piece indices).
  - Full reset every 20 accepted new basins.

Restart (plan v4 §1.7):
  After K=30 iters with no archive insert, sample next current from top-N
  archive entries with P ∝ 0.7·quality + 0.3·novelty.

Events (one JSON per os.write, atomic under O_APPEND ≤ 4KB):
  - {"type":"accept", "trial":N, "score":R, "scs":..., "move":..., "D":..., "nov":...}
  - {"type":"reject", "trial":N, "reason":"resolve_failed"|"stage_a_failed"|"tabu"}
  - {"type":"restart", "trial":N, "from_rank":k}
  Format is consumed by archive_reducer.ArchiveReducer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.optimize as opt

import archive_reducer as ar
import attack4
import basin_archive
import gap as gapmod
import geom
import perturb_lib as pl


# ---------- overlap resolve ----------


def _resolve_penalty(x: np.ndarray, cx: float, cy: float, R: float) -> float:
    scs = x.reshape(geom.N, 3)
    pair_pen = float(gapmod.overlap_penalty_sq(scs, margin=0.0))
    cont = 0.0
    for i in range(geom.N):
        g = attack4.contain_gap_exact(scs[i, 0], scs[i, 1], scs[i, 2], cx, cy, R)
        if g < 0:
            cont += g * g
    return pair_pen + cont


def resolve_overlap(
    scs: np.ndarray,
    cx: float,
    cy: float,
    R: float,
    maxiter: int = 1000,
    tol: float = 1e-3,
) -> tuple[np.ndarray, float, bool]:
    """Gradient-descent overlap-resolve at fixed (cx, cy, R). Containment is
    scored against the BEFORE-state MEC — Stage A subsequently re-optimizes
    the center. Returns (scs_out, penalty, ok)."""
    x0 = scs.reshape(-1).copy()
    res = opt.minimize(
        _resolve_penalty,
        x0,
        args=(cx, cy, R),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
    )
    pen = float(res.fun)
    return res.x.reshape(geom.N, 3), pen, pen < tol


# ---------- reactive tabu ----------


@dataclass
class ReactiveTabu:
    basin_start: int = 30
    basin_step_up: int = 5
    basin_step_down: int = 1
    basin_cap: int = 60
    basin_decay_every: int = 50
    move_start: int = 60
    move_step_up: int = 10
    move_step_down: int = 2
    move_cap: int = 100
    move_decay_every: int = 50
    reset_every_new_basins: int = 20
    basin_tenure: dict[tuple, int] = field(default_factory=dict)
    move_tenure: dict[tuple, int] = field(default_factory=dict)
    iters_since_basin_dup: int = 0
    iters_since_move_dup: int = 0
    accepts_since_reset: int = 0

    def hit_basin(self, key: tuple) -> bool:
        """Return True if basin is currently tabu'd."""
        return key in self.basin_tenure and self.basin_tenure[key] > 0

    def hit_move(self, key: tuple) -> bool:
        return key in self.move_tenure and self.move_tenure[key] > 0

    def note_basin(self, key: tuple, is_dup: bool):
        if is_dup:
            cur = self.basin_tenure.get(key, self.basin_start)
            self.basin_tenure[key] = min(cur + self.basin_step_up, self.basin_cap)
            self.iters_since_basin_dup = 0
        else:
            self.iters_since_basin_dup += 1
            if self.iters_since_basin_dup >= self.basin_decay_every:
                self._decay_basin()
                self.iters_since_basin_dup = 0

    def note_move(self, key: tuple, is_dup: bool):
        if is_dup:
            cur = self.move_tenure.get(key, self.move_start)
            self.move_tenure[key] = min(cur + self.move_step_up, self.move_cap)
            self.iters_since_move_dup = 0
        else:
            self.iters_since_move_dup += 1
            if self.iters_since_move_dup >= self.move_decay_every:
                self._decay_move()
                self.iters_since_move_dup = 0

    def on_accept_new_basin(self):
        self.accepts_since_reset += 1
        if self.accepts_since_reset >= self.reset_every_new_basins:
            self.basin_tenure.clear()
            self.move_tenure.clear()
            self.accepts_since_reset = 0

    def _decay_basin(self):
        for k in list(self.basin_tenure.keys()):
            self.basin_tenure[k] -= self.basin_step_down
            if self.basin_tenure[k] <= 0:
                del self.basin_tenure[k]

    def _decay_move(self):
        for k in list(self.move_tenure.keys()):
            self.move_tenure[k] -= self.move_step_down
            if self.move_tenure[k] <= 0:
                del self.move_tenure[k]


# ---------- restart ----------


def _pick_restart_index(
    archive: basin_archive.BasinArchive,
    best_score: float,
    rng: np.random.Generator,
    top_n: int = 5,
) -> int:
    """Sample index i ∈ [0, top_n) with P ∝ 0.7·quality + 0.3·novelty.

    Temperature for the quality term is derived from the observed top-N spread,
    not hard-coded: T = max(1e-6, max_delta / ln(1000)). Hard-coding 1e-4 made
    every non-best quality underflow to 0 once deltas exceeded ~1e-3 (which
    they do — the incumbent basin is ~0.08 away from the next).
    """
    entries = archive.entries[:top_n]
    if len(entries) == 0:
        return 0
    if len(entries) == 1:
        return 0
    deltas = np.array([e["score"] - best_score for e in entries], dtype=float)
    T = max(1e-6, float(deltas.max()) / math.log(1000.0))
    qualities = np.exp(-deltas / T)
    if qualities.max() > 0:
        qualities /= qualities.max()
    novelties = np.zeros(len(entries))
    for i, e_i in enumerate(entries):
        dists = [
            basin_archive.signature_l2(e_i["signature"], e_j["signature"])
            for j, e_j in enumerate(entries)
            if j != i
        ]
        novelties[i] = float(np.mean(dists)) if dists else 0.0
    if novelties.max() > 0:
        novelties /= novelties.max()
    weights = 0.7 * qualities + 0.3 * novelties
    s = weights.sum()
    if s <= 0:
        return 0
    probs = weights / s
    return int(rng.choice(len(entries), p=probs))


def _move_key(res: "pl.PerturbResult") -> tuple[str, frozenset[int]]:
    """Normalized tabu key across all move types. perturb_lib emits different
    metadata shapes per move (piece / pieces / cluster); unify them here so
    rim_swap doesn't collapse to a single universal key."""
    meta = res.metadata
    if "cluster" in meta:
        pieces = frozenset(int(i) for i in meta["cluster"])
    elif "pieces" in meta:
        pieces = frozenset(int(i) for i in meta["pieces"])
    elif "piece" in meta:
        pieces = frozenset({int(meta["piece"])})
    else:
        pieces = frozenset()
    return (res.move_type, pieces)


def _basin_key(scs: np.ndarray) -> tuple:
    """Canonical basin key for tabu (round to 6 decimals to compress near-
    duplicates through noisy Stage A outputs)."""
    sig = basin_archive.solution_signature(scs)
    return tuple(np.round(sig, 6).tolist())


# ---------- event emission ----------


def _trim_for_atomic_append(event: dict) -> dict:
    """Small-event events carry scs; large events drop scs and mark it offloaded.
    MAX_EVENT_BYTES is 4096. Typical scs with 15×3 floats at 20 chars each plus
    metadata fits, but over-the-limit events are rejected upstream. This helper
    is a no-op for now — scs arrays always fit — but keeps the API explicit."""
    return event


# ---------- main loop ----------


@dataclass
class DriverConfig:
    events_path: str
    archive_path: str
    incumbent_path: str
    R_target: float = 2.9486936795
    R_resolve_slack: float = 0.10  # resolve at R_target + slack (room to untangle)
    resolve_tol: float = 5e-2  # lenient — Stage A does the final polish
    hours: float = 0.5
    max_iters: int | None = None
    K_restart: int = 30
    top_n_restart: int = 5
    seed: int = 42
    stage_a_maxiter: int = 60
    stop_flag: str = ""
    snapshot_every_s: float = 30.0


def run(cfg: DriverConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    archive = basin_archive.BasinArchive(slots=32, min_l2=0.08)
    tabu = ReactiveTabu()

    # Seed archive with incumbent.
    scs_inc = np.array(json.load(open(cfg.incumbent_path))["scs"], dtype=np.float64)
    scs_inc = geom.rnd(scs_inc)
    R_inc = float(geom.mec(scs_inc))
    archive.consider(scs_inc, R_inc, trial=-1, label="incumbent")
    best_score = R_inc

    # Event log + periodic snapshotter.
    events_fd = ar.open_append_fd(cfg.events_path)
    snap_path = Path(cfg.archive_path)

    def _snapshot():
        payload = archive.payload("mbh", best_score)
        payload["iters"] = stats["iters"]
        payload["accepts"] = stats["accepts"]
        tmp = snap_path.with_suffix(snap_path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, snap_path)
        try:
            os.fdatasync(events_fd)
        except OSError:
            pass

    # Graceful stop: SIGTERM → stop_requested[0]=True.
    stop_requested = [False]

    def _on_sigterm(signum, frame):
        stop_requested[0] = True

    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    stats = {
        "iters": 0,
        "accepts": 0,
        "resolve_failed": 0,
        "stage_a_failed": 0,
        "tabu_skips": 0,
        "restarts": 0,
        # Restart counter: iterations since the last archive insert of any kind
        # (not just a new global best). Plan v4 §1.7 wording was "K=30 no-
        # improvement perturbations"; we interpret that as "no archive insert"
        # because distinct accepted basins still diversify the front.
        "since_insert": 0,
    }

    t_start = time.time()
    last_snap = t_start

    def _time_up() -> bool:
        if stop_requested[0]:
            return True
        if cfg.stop_flag and Path(cfg.stop_flag).exists():
            return True
        if cfg.max_iters is not None and stats["iters"] >= cfg.max_iters:
            return True
        return time.time() - t_start >= cfg.hours * 3600

    R_resolve = cfg.R_target + cfg.R_resolve_slack
    current_idx = 0  # rank in archive to use as current
    print(
        f"[mbh] start R_target={cfg.R_target} R_resolve={R_resolve} hours={cfg.hours}"
    )

    visited_basins: set[tuple] = set()
    while not _time_up():
        stats["iters"] += 1
        trial = stats["iters"]

        # ---- restart policy (fires on since_insert — any accept, not just new best)
        if stats["since_insert"] >= cfg.K_restart:
            current_idx = _pick_restart_index(
                archive, best_score, rng, top_n=cfg.top_n_restart
            )
            stats["restarts"] += 1
            stats["since_insert"] = 0
            tabu.basin_tenure.clear()
            tabu.move_tenure.clear()
            ar.append_event(
                events_fd,
                {"type": "restart", "trial": trial, "from_rank": current_idx},
            )

        if archive.size() == 0:
            break
        cur_idx = min(current_idx, archive.size() - 1)
        current = archive.entries[cur_idx]["scs"]

        # ---- perturb
        res = pl.propose(current, cfg.R_target, rng=rng, max_retries=6)
        move_key = _move_key(res)
        if tabu.hit_move(move_key):
            stats["tabu_skips"] += 1
            stats["since_insert"] += 1
            continue

        # ---- resolve + Stage A
        cx_before, cy_before, _ = geom.mec_info(current)
        scs_resolved, pen, ok = resolve_overlap(
            res.scs, cx_before, cy_before, R_resolve, tol=cfg.resolve_tol
        )
        if not ok:
            stats["resolve_failed"] += 1
            stats["since_insert"] += 1
            ar.append_event(
                events_fd,
                {
                    "type": "reject",
                    "trial": trial,
                    "reason": "resolve_failed",
                    "penalty": float(pen),
                    "move": res.move_type,
                },
            )
            continue

        cx, cy, _ = geom.mec_info(scs_resolved)
        try:
            feats, witnesses = attack4.extract_contact_graph(
                scs_resolved, cx, cy, float(geom.mec(scs_resolved))
            )
            res_a = attack4.stage_a(
                scs_resolved,
                cx,
                cy,
                float(geom.mec(scs_resolved)),
                feats,
                witnesses,
                maxiter=cfg.stage_a_maxiter,
                margin_pair=2e-6,
                margin_contain=2e-6,
            )
        except Exception as exc:
            stats["stage_a_failed"] += 1
            stats["since_insert"] += 1
            ar.append_event(
                events_fd,
                {
                    "type": "reject",
                    "trial": trial,
                    "reason": f"stage_a_exc:{type(exc).__name__}",
                    "move": res.move_type,
                },
            )
            continue

        if not res_a.get("success", False):
            stats["stage_a_failed"] += 1
            stats["since_insert"] += 1
            ar.append_event(
                events_fd,
                {
                    "type": "reject",
                    "trial": trial,
                    "reason": "stage_a_unsuccessful",
                    "move": res.move_type,
                },
            )
            continue

        scs_final = geom.rnd(res_a["scs"])
        if int(geom.cnt(scs_final)) != 0:
            stats["stage_a_failed"] += 1
            stats["since_insert"] += 1
            ar.append_event(
                events_fd,
                {
                    "type": "reject",
                    "trial": trial,
                    "reason": "rounded_infeasible",
                    "move": res.move_type,
                },
            )
            continue
        score_final = float(geom.mec(scs_final))

        # ---- basin tabu: key on the candidate's canonical signature, not
        # archive.entries[0]. Archive can evict entries past the 32-slot cap,
        # so we track visited separately.
        candidate_key = _basin_key(scs_final)
        was_visited = candidate_key in visited_basins
        if tabu.hit_basin(candidate_key):
            stats["tabu_skips"] += 1
            stats["since_insert"] += 1
            tabu.note_basin(candidate_key, is_dup=True)
            tabu.note_move(move_key, is_dup=True)
            ar.append_event(
                events_fd,
                {
                    "type": "reject",
                    "trial": trial,
                    "reason": "basin_tabu",
                    "move": res.move_type,
                },
            )
            continue
        visited_basins.add(candidate_key)

        # ---- archive
        before_size = archive.size()
        event = archive.consider(
            scs_final, score_final, trial=trial, label=res.move_type
        )
        is_new = event is not None
        is_dup_archive = not is_new
        # A basin we've *visited* before but isn't in the retained archive
        # still counts as a duplicate for tabu escalation purposes.
        is_dup = is_dup_archive or was_visited
        tabu.note_basin(candidate_key, is_dup=is_dup)
        tabu.note_move(move_key, is_dup=is_dup)

        if is_new:
            stats["accepts"] += 1
            stats["since_insert"] = 0
            tabu.on_accept_new_basin()
            if score_final < best_score:
                best_score = score_final
            ar.append_event(
                events_fd,
                {
                    "type": "accept",
                    "trial": trial,
                    "score": score_final,
                    "move": res.move_type,
                    "D": res.D,
                    "before_size": before_size,
                    "after_size": archive.size(),
                },
            )
        else:
            stats["since_insert"] += 1
            ar.append_event(
                events_fd,
                {
                    "type": "duplicate",
                    "trial": trial,
                    "score": score_final,
                    "move": res.move_type,
                    "D": res.D,
                },
            )

        now = time.time()
        if now - last_snap >= cfg.snapshot_every_s:
            _snapshot()
            last_snap = now
            elapsed = now - t_start
            rate = stats["iters"] / elapsed if elapsed > 0 else 0
            print(
                f"[mbh] t={elapsed:6.0f}s iter={stats['iters']:5d} "
                f"accepts={stats['accepts']:3d} best={best_score:.6f} "
                f"archive={archive.size():2d} resolved_fail={stats['resolve_failed']} "
                f"stage_a_fail={stats['stage_a_failed']} restarts={stats['restarts']} "
                f"rate={rate:.2f}/s"
            )

    _snapshot()
    os.close(events_fd)
    elapsed = time.time() - t_start
    summary = {
        "elapsed_s": elapsed,
        "best_score": best_score,
        "archive_size": archive.size(),
        **stats,
    }
    print(f"[mbh] done: {summary}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent", default="pool/best.json")
    ap.add_argument("--events", default="runs/mbh/events.jsonl")
    ap.add_argument("--archive", default="runs/mbh/archive.json")
    ap.add_argument("--R-target", type=float, default=2.9486936795)
    ap.add_argument("--hours", type=float, default=0.5)
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stop-flag", default="")
    args = ap.parse_args()

    Path(args.events).parent.mkdir(parents=True, exist_ok=True)
    cfg = DriverConfig(
        events_path=args.events,
        archive_path=args.archive,
        incumbent_path=args.incumbent,
        R_target=args.R_target,
        hours=args.hours,
        max_iters=args.max_iters,
        seed=args.seed,
        stop_flag=args.stop_flag,
    )
    summary = run(cfg)
    return summary


if __name__ == "__main__":
    main()
