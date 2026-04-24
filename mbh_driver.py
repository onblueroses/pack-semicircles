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


def _worst_gap(scs: np.ndarray, cx: float, cy: float, R: float) -> float:
    """Return the minimum (most negative) gap across pair + containment."""
    worst = float("inf")
    for i in range(geom.N):
        g = attack4.contain_gap_exact(scs[i, 0], scs[i, 1], scs[i, 2], cx, cy, R)
        if g < worst:
            worst = g
        for j in range(i + 1, geom.N):
            g = gapmod.gap_ss(
                scs[i, 0],
                scs[i, 1],
                scs[i, 2],
                scs[j, 0],
                scs[j, 1],
                scs[j, 2],
            )
            if g < worst:
                worst = g
    return worst


def resolve_overlap(
    scs: np.ndarray,
    cx: float,
    cy: float,
    R: float,
    maxiter: int = 1000,
    max_deficit: float = 5e-3,
) -> tuple[np.ndarray, float, float, bool]:
    """Gradient-descent overlap-resolve at fixed (cx, cy, R). Returns
    (scs_out, total_penalty, worst_negative_gap, ok).

    Gate is based on worst INDIVIDUAL negative gap (max_deficit), not the
    sum-of-squares total. sum-of-squares can be tiny while one pair is
    wildly overlapping; that single pair then poisons Stage A.
    """
    x0 = scs.reshape(-1).copy()
    res = opt.minimize(
        _resolve_penalty,
        x0,
        args=(cx, cy, R),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
    )
    scs_out = res.x.reshape(geom.N, 3)
    worst = _worst_gap(scs_out, cx, cy, R)
    # ok if worst negative gap is within tolerance (i.e. -worst < max_deficit)
    ok = worst > -max_deficit
    return scs_out, float(res.fun), worst, ok


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


def _basin_signature(scs: np.ndarray) -> np.ndarray:
    """Canonical basin signature (rigid-motion + permutation invariant).
    Use signature_l2 for equivalence comparisons — exact tuple equality is
    far too strict and fragments same-basin revisits on solver noise."""
    return basin_archive.solution_signature(scs)


class BasinTabu:
    """L2-distance tabu list for basin signatures. Matches BasinArchive's
    same-basin criterion (signature_l2 < min_l2), so tabu and archive agree on
    what counts as a revisit."""

    def __init__(
        self,
        min_l2: float = 0.08,
        start: int = 30,
        step_up: int = 5,
        step_down: int = 1,
        cap: int = 60,
        decay_every: int = 50,
    ):
        self.min_l2 = min_l2
        self.start = start
        self.step_up = step_up
        self.step_down = step_down
        self.cap = cap
        self.decay_every = decay_every
        # Parallel lists: sig rows + tenures + last-seen iter
        self._sigs: list[np.ndarray] = []
        self._tenure: list[int] = []
        self._no_dup_iters = 0

    def _find(self, sig: np.ndarray) -> int:
        for i, s in enumerate(self._sigs):
            if basin_archive.signature_l2(s, sig) < self.min_l2:
                return i
        return -1

    def hit(self, sig: np.ndarray) -> bool:
        i = self._find(sig)
        return i >= 0 and self._tenure[i] > 0

    def note(self, sig: np.ndarray, is_dup: bool) -> None:
        """Record a basin visit. On is_dup=True, escalate tenure. On
        is_dup=False (first visit), arm at start tenure so the NEXT visit
        actually hits. Previous behavior (only storing on is_dup=True) meant
        a basin had to be visited 3× before tabu could skip it."""
        i = self._find(sig)
        if is_dup:
            if i >= 0:
                self._tenure[i] = min(self._tenure[i] + self.step_up, self.cap)
            else:
                self._sigs.append(sig)
                self._tenure.append(min(self.start + self.step_up, self.cap))
            self._no_dup_iters = 0
        else:
            # First visit: arm at start tenure so the second visit is blocked.
            if i < 0:
                self._sigs.append(sig)
                self._tenure.append(self.start)
            self._no_dup_iters += 1
            if self._no_dup_iters >= self.decay_every:
                self._decay()
                self._no_dup_iters = 0

    def _decay(self) -> None:
        keep_sigs: list[np.ndarray] = []
        keep_ten: list[int] = []
        for s, t in zip(self._sigs, self._tenure):
            t -= self.step_down
            if t > 0:
                keep_sigs.append(s)
                keep_ten.append(t)
        self._sigs = keep_sigs
        self._tenure = keep_ten

    def reset(self) -> None:
        self._sigs.clear()
        self._tenure.clear()
        self._no_dup_iters = 0

    def size(self) -> int:
        return len(self._sigs)


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
    resolve_max_deficit: float = 5e-3  # worst single-gap deficit before reject
    min_pair_gap: float = -5e-6  # post-Stage A pair gate (scorer-safe margin)
    stage_a_max_violation: float = 5e-4  # attack4 constr_violation ceiling
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
    basin_tabu = BasinTabu(min_l2=0.08)  # L2-based, matches archive dedup rule

    # Seed archive with incumbent.
    scs_inc = np.array(json.load(open(cfg.incumbent_path))["scs"], dtype=np.float64)
    scs_inc = geom.rnd(scs_inc)
    R_inc = float(geom.mec(scs_inc))
    archive.consider(scs_inc, R_inc, trial=-1, label="incumbent")
    best_score = R_inc
    # Arm basin_tabu with the incumbent's signature immediately. Otherwise the
    # first revisit of the seeded basin slips through (the seed was inserted
    # via archive.consider, which doesn't go through the driver's basin_tabu).
    basin_tabu.note(_basin_signature(scs_inc), is_dup=False)

    # Event log + periodic snapshotter.
    events_fd = ar.open_append_fd(cfg.events_path)
    snap_path = Path(cfg.archive_path)

    # Emit a replayable seed event FIRST so archive_reducer replay can rebuild
    # the same state the driver started with. Without this, replay from a log
    # with zero iters produces an empty archive while the live snapshot has
    # the incumbent.
    ar.append_event(
        events_fd,
        {
            "type": "accept",
            "trial": -1,
            "score": R_inc,
            "scs": scs_inc.tolist(),
            "label": "incumbent",
        },
    )

    def _snapshot():
        # Durability ordering (mirrors archive_reducer.snapshot):
        # 1. fsync the event log FIRST — never publish a snapshot that
        #    names events the durable log doesn't contain. Propagate OSError.
        # 2. write tmp → flush → fsync tmp.
        # 3. os.replace(tmp, snap).
        # 4. fsync parent dir so the rename itself is durable.
        os.fdatasync(events_fd)
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
        dir_fd = os.open(snap_path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

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
            basin_tabu.reset()
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
            # Emit a reject so the event log has a record for every iter.
            ar.append_event(
                events_fd,
                {
                    "type": "reject",
                    "trial": trial,
                    "reason": "move_tabu",
                    "move": res.move_type,
                },
            )
            continue

        # ---- resolve + Stage A
        cx_before, cy_before, _ = geom.mec_info(current)
        scs_resolved, pen, worst_gap, ok = resolve_overlap(
            res.scs,
            cx_before,
            cy_before,
            R_resolve,
            max_deficit=cfg.resolve_max_deficit,
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
                    "worst_gap": float(worst_gap),
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

        # success flag is unreliable — attack4.stage_a returns True even on
        # trust-min exits. Gate on constr_violation + raw min-gap instead.
        violation = float(res_a.get("constr_violation", 0.0))
        if violation > cfg.stage_a_max_violation:
            stats["stage_a_failed"] += 1
            stats["since_insert"] += 1
            ar.append_event(
                events_fd,
                {
                    "type": "reject",
                    "trial": trial,
                    "reason": "stage_a_constr_violation",
                    "violation": violation,
                    "move": res.move_type,
                },
            )
            continue

        # Pre-round min-gap gate — catches genuine overlap regardless of
        # whether rnd flips an active constraint. Cheaper than geom.cnt on
        # rounded config and distinguishes real overlap from rounding noise.
        scs_stage = res_a["scs"]
        min_pair_gap = float("inf")
        for i in range(geom.N):
            for j in range(i + 1, geom.N):
                g = gapmod.gap_ss(
                    scs_stage[i, 0],
                    scs_stage[i, 1],
                    scs_stage[i, 2],
                    scs_stage[j, 0],
                    scs_stage[j, 1],
                    scs_stage[j, 2],
                )
                if g < min_pair_gap:
                    min_pair_gap = g
        if min_pair_gap < cfg.min_pair_gap:
            stats["stage_a_failed"] += 1
            stats["since_insert"] += 1
            ar.append_event(
                events_fd,
                {
                    "type": "reject",
                    "trial": trial,
                    "reason": "pair_overlap",
                    "min_pair_gap": min_pair_gap,
                    "move": res.move_type,
                },
            )
            continue

        scs_final = geom.rnd(scs_stage)
        if int(geom.cnt(scs_final)) != 0:
            # Should be rare given the pre-round gate. Still a safety net.
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

        # ---- basin tabu: compare candidate signature to the L2-distance tabu
        # list. Matches BasinArchive's dedup rule (min_l2=0.08) so tabu +
        # archive agree on what counts as "same basin".
        candidate_sig = _basin_signature(scs_final)
        if basin_tabu.hit(candidate_sig):
            stats["tabu_skips"] += 1
            stats["since_insert"] += 1
            basin_tabu.note(candidate_sig, is_dup=True)
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

        # ---- archive
        before_size = archive.size()
        event = archive.consider(
            scs_final, score_final, trial=trial, label=res.move_type
        )
        # archive.consider actions:
        #   None         → same basin, not better than existing: duplicate
        #   "replace"    → same basin, but better score: NOT a new basin
        #   "insert"     → truly new basin
        action = event["action"] if event is not None else "duplicate"
        is_new_basin = action == "insert"
        is_dup = not is_new_basin  # replace OR duplicate both hit the same basin
        basin_tabu.note(candidate_sig, is_dup=is_dup)
        tabu.note_move(move_key, is_dup=is_dup)

        if is_new_basin:
            stats["accepts"] += 1
            stats["since_insert"] = 0
            tabu.on_accept_new_basin()
            if score_final < best_score:
                best_score = score_final
            # Accept events carry scs + label so archive_reducer.replay can
            # rebuild archive state from the log (HIGH regression from round 4).
            ar.append_event(
                events_fd,
                {
                    "type": "accept",
                    "trial": trial,
                    "score": score_final,
                    "scs": scs_final.tolist(),
                    "label": res.move_type,
                    "move": res.move_type,
                    "D": res.D,
                    "before_size": before_size,
                    "after_size": archive.size(),
                },
            )
        else:
            stats["since_insert"] += 1
            # Same-basin improvements ("replace") still update best_score.
            if action == "replace" and score_final < best_score:
                best_score = score_final
            ev = {
                "type": action,  # "duplicate" or "replace"
                "trial": trial,
                "score": score_final,
                "move": res.move_type,
                "D": res.D,
            }
            # replace events mutate archive state (better score in existing
            # basin). They must carry scs+label so reducer replay stays
            # faithful. duplicate events don't mutate state — scs optional.
            if action == "replace":
                ev["scs"] = scs_final.tolist()
                ev["label"] = res.move_type
            ar.append_event(events_fd, ev)

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
