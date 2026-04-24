"""Distinct basin archive helpers for semicircle packing runs."""

import numpy as np

import geom

CONTACT_R = 2.1
SCORE_EPS = 1e-8


def solution_signature(scs, eps=1e-12):
    """Rigid-motion (translation + rotation + reflection) + permutation invariant
    fingerprint. Anchor-based: enumerates 2*N*(N-1) candidate frames (one per
    ordered non-coincident piece pair × reflection), rounds coords to 12 decimals
    before sorting, takes lex-min. Codex-verified L2=0.0 on all invariance classes
    including collinear, square, hex."""
    pts = np.asarray(scs, dtype=np.float64).reshape(geom.N, 3).copy()
    pts[:, :2] -= pts[:, :2].mean(axis=0)
    pos = pts[:, :2]
    dirs = np.column_stack([np.cos(pts[:, 2]), np.sin(pts[:, 2])])
    n = len(pts)
    best = None
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = pos[j] - pos[i]
            d = np.hypot(v[0], v[1])
            if d <= eps:
                continue
            ex = v / d
            ey = np.array([-ex[1], ex[0]])
            for refl in (+1, -1):
                B = np.vstack([ex, refl * ey])
                xy = pos @ B.T
                uv = dirs @ B.T
                keys = np.stack(
                    [
                        np.round(xy[:, 0], 12),
                        np.round(xy[:, 1], 12),
                        np.round(uv[:, 1], 12),
                        np.round(uv[:, 0], 12),
                    ],
                    axis=1,
                )
                order = np.lexsort(keys.T[::-1])
                cand = tuple(keys[order].flatten().tolist())
                if best is None or cand < best:
                    best = cand
    if best is None:
        raise ValueError("no distinct anchor pair")
    return np.array(best, dtype=np.float64)


def signature_l2(sig_a, sig_b):
    return float(np.sqrt(np.sum((sig_a - sig_b) ** 2)))


def _count_contacts(scs, r=CONTACT_R):
    c = 0
    for i in range(geom.N):
        for j in range(i + 1, geom.N):
            dx = scs[i, 0] - scs[j, 0]
            dy = scs[i, 1] - scs[j, 1]
            if dx * dx + dy * dy < r * r:
                c += 1
    return c


def _recipe_dict(recipe):
    if recipe is None:
        return None
    return {
        "outer_pos": float(recipe[0]),
        "outer_theta": float(recipe[1]),
        "cma_sigma0": float(recipe[2]),
        "cma_popsize": int(recipe[3]),
        "cma_maxiter": int(recipe[4]),
    }


def _solution_list(scs):
    return [
        {"x": float(scs[i, 0]), "y": float(scs[i, 1]), "theta": float(scs[i, 2])}
        for i in range(geom.N)
    ]


class BasinArchive:
    def __init__(self, slots=12, min_l2=0.08):
        self.slots = int(slots)
        self.min_l2 = float(min_l2)
        self.entries = []
        self._next_id = 0

    def size(self):
        return len(self.entries)

    def best_alternative_score(self, best_score):
        for entry in self.entries:
            if entry["score"] > best_score + SCORE_EPS:
                return float(entry["score"])
        return None

    def _make_entry(self, scs, score, trial, recipe, label, signature):
        rounded = geom.rnd(np.asarray(scs, dtype=np.float64).reshape(geom.N, 3))
        mask, _cx, _cy, _r = geom.find_boundary_mask(rounded)
        boundary = [int(i) for i in np.flatnonzero(mask)]
        entry = {
            "_id": self._next_id,
            "score": float(score),
            "trial": int(trial),
            "label": str(label),
            "recipe": _recipe_dict(recipe),
            "boundary": boundary,
            "boundary_count": len(boundary),
            "contact_count": int(_count_contacts(rounded)),
            "scs": rounded.copy(),
            "signature": signature.copy(),
        }
        self._next_id += 1
        return entry

    def _sort_trim(self):
        self.entries.sort(key=lambda entry: entry["score"])
        if len(self.entries) > self.slots:
            self.entries = self.entries[: self.slots]

    def consider(self, scs, score, trial, recipe=None, label="candidate"):
        rounded = geom.rnd(np.asarray(scs, dtype=np.float64).reshape(geom.N, 3))
        signature = solution_signature(rounded)

        closest_idx = None
        closest_dist = float("inf")
        for idx, existing in enumerate(self.entries):
            dist = signature_l2(signature, existing["signature"])
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = idx
            if dist < self.min_l2:
                if score + SCORE_EPS < existing["score"]:
                    self.entries[idx] = self._make_entry(
                        rounded, score, trial, recipe, label, signature
                    )
                    self._sort_trim()
                    rank = next(
                        i
                        for i, entry in enumerate(self.entries)
                        if entry["trial"] == int(trial)
                        and abs(entry["score"] - score) < 1e-12
                    )
                    return {
                        "action": "replace",
                        "rank": rank,
                        "closest_l2": float(dist),
                    }
                return None

        entry = self._make_entry(rounded, score, trial, recipe, label, signature)
        entry_id = entry["_id"]
        self.entries.append(entry)
        self._sort_trim()
        for rank, kept in enumerate(self.entries):
            if kept["_id"] == entry_id:
                return {
                    "action": "insert",
                    "rank": rank,
                    "closest_l2": None if closest_idx is None else float(closest_dist),
                }
        return None

    def payload(self, run_name, best_score):
        best_signature = None
        for entry in self.entries:
            if abs(entry["score"] - best_score) <= SCORE_EPS:
                best_signature = entry["signature"]
                break
        if best_signature is None and self.entries:
            best_signature = self.entries[0]["signature"]

        entries = []
        for rank, entry in enumerate(self.entries):
            sig_l2 = None
            if best_signature is not None:
                sig_l2 = signature_l2(entry["signature"], best_signature)
            entries.append(
                {
                    "rank": rank,
                    "score": float(entry["score"]),
                    "delta": float(entry["score"] - best_score),
                    "trial": int(entry["trial"]),
                    "label": entry["label"],
                    "recipe": entry["recipe"],
                    "boundary_count": int(entry["boundary_count"]),
                    "boundary": list(entry["boundary"]),
                    "contact_count": int(entry["contact_count"]),
                    "signature_l2_to_best": sig_l2,
                    "scs": entry["scs"].tolist(),
                    "solution": _solution_list(entry["scs"]),
                }
            )

        alt_score = self.best_alternative_score(best_score)
        return {
            "run": str(run_name),
            "best_score": float(best_score),
            "best_alternative_score": alt_score,
            "best_alternative_delta": None
            if alt_score is None
            else float(alt_score - best_score),
            "archive_size": len(entries),
            "archive_slots": int(self.slots),
            "distinct_min_l2": float(self.min_l2),
            "contact_radius": float(CONTACT_R),
            "entries": entries,
        }
