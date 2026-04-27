# Stages candidate as ./solution.json before verify.mjs (D3); restores prior content on non-promote.
"""Submission gate around island_orchestrator.promote_if_better.

verify.mjs hardcodes its read of ./solution.json (it ignores argv). This wrapper
snapshots the existing ./solution.json bytes, stages the candidate's array-shape
solution payload at ./solution.json so verify.mjs reads the candidate, calls
promote_if_better, and on any non-promote outcome restores ./solution.json
byte-for-byte. On promote, ./solution.json is left synchronized with the new
pool/best.json.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import shutil
import sys
import tempfile
from pathlib import Path
from types import FrameType
from typing import Callable, Iterator

import numpy as np

import geom
from island_orchestrator import PromoteResult, promote_if_better

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent
SOLUTION_NAME = "solution.json"
LOCK_NAME = ".cg_promote.lock"
DEFAULT_INCUMBENT = Path("pool/best.json")
SignalHandler = signal.Handlers | int | None | Callable[[int, FrameType | None], object]


class PromoteInterrupted(KeyboardInterrupt):
    """Raised when SIGTERM/SIGINT interrupts a promote in cleanup-sensitive code."""

    def __init__(self, signum: int):
        self.signum = signum
        self.signal_name = signal.Signals(signum).name
        super().__init__(f"cg_promote interrupted by {self.signal_name}")


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def _load_archive(merged_path: Path) -> dict[str, object]:
    """Load and validate a merged archive once so later steps share one view."""
    with open(merged_path) as handle:
        archive = json.load(handle)
    if not isinstance(archive, dict):
        raise ValueError("merged archive must be a JSON object")
    return archive


def _load_first_entry_scs(archive: dict[str, object]) -> np.ndarray:
    """Return the rounded scs array of the merged archive's top-ranked entry."""
    entries_obj = archive.get("entries")
    entries = entries_obj if isinstance(entries_obj, list) else []
    if not entries:
        raise ValueError("merged archive has no entries")
    first = entries[0]
    if not isinstance(first, dict):
        raise ValueError("merged archive entry 0 is not a JSON object")
    scs = np.asarray(first.get("scs"), dtype=np.float64).reshape(geom.N, 3)
    return geom.rnd(scs)


def _tmp_path(path: Path, suffix: str) -> Path:
    return path.parent / f"{path.name}{suffix}"


def _write_text_atomic(path: Path, text: str, tmp_suffix: str = ".tmp") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path, tmp_suffix)
    tmp.write_text(text)
    os.replace(tmp, path)


def _write_bytes_atomic(path: Path, payload: bytes, tmp_suffix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path, tmp_suffix)
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _write_harvest_compat_archive(archive: dict[str, object], dst: Path) -> Path:
    """Translate cg_orchestrator's R-keyed entries into harvest.py's score-keyed form.

    cg_orchestrator writes {"entries": [{"R": ..., "scs": ...}, ...]}; harvest.py
    reads `entries[0]["score"]`. We round-trip through a tmp file, copying R into
    score, so promote_if_better's harvest --dry-run / --yes invocations succeed.
    Only the first (best) entry needs to be valid for harvest.
    """
    entries_obj = archive.get("entries")
    entries = list(entries_obj) if isinstance(entries_obj, list) else []
    translated: list[object] = []
    for entry in entries:
        if not isinstance(entry, dict):
            translated.append(entry)
            continue
        new_entry = dict(entry)
        if "score" not in new_entry and "R" in new_entry:
            new_entry["score"] = new_entry["R"]
        translated.append(new_entry)
    _write_text_atomic(dst, json.dumps({"entries": translated}))
    return dst


def _array_solution_payload(rounded: np.ndarray) -> list[dict[str, float]]:
    """Build the verify.mjs-shaped array (list of {x, y, theta} dicts)."""
    return [
        {
            "x": float(rounded[i, 0]),
            "y": float(rounded[i, 1]),
            "theta": float(rounded[i, 2]),
        }
        for i in range(geom.N)
    ]


def _solution_payload_from_best(incumbent_path: Path) -> list[dict[str, float]]:
    """Reconstruct verify.mjs-shaped array from a pool/best.json-style file."""
    with open(incumbent_path) as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return [
            {"x": float(s["x"]), "y": float(s["y"]), "theta": float(s["theta"])}
            for s in data
        ]
    scs = np.asarray(data["scs"], dtype=np.float64).reshape(geom.N, 3)
    return _array_solution_payload(scs)


def _stage_solution_array(solution_path: Path, rounded: np.ndarray) -> None:
    """Write the verify.mjs-shaped array to solution.json via atomic replace."""
    _write_text_atomic(
        solution_path,
        json.dumps(_array_solution_payload(rounded)),
        tmp_suffix=".cg_stage.tmp",
    )


def _restore_solution(
    solution_path: Path, pre_existed: bool, snapshot: bytes | None
) -> None:
    if pre_existed and snapshot is not None:
        _write_bytes_atomic(
            solution_path,
            snapshot,
            tmp_suffix=".cg_restore.tmp",
        )
        return
    try:
        solution_path.unlink()
    except FileNotFoundError:
        pass


@contextlib.contextmanager
def _locked_repo(lock_path: Path) -> Iterator[None]:
    with open(lock_path, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _pushd(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@contextlib.contextmanager
def _temporary_interrupt_handlers() -> Iterator[None]:
    """Temporarily route SIGTERM/SIGINT through Python cleanup before exit."""

    previous_handlers: dict[int, SignalHandler] = {}

    def _raise_interrupt(signum: int, _frame: FrameType | None) -> None:
        raise PromoteInterrupted(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _raise_interrupt)
    try:
        yield
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _raise_cleanup_errors(
    original_error: BaseException | None,
    restore_error: BaseException | None,
    cleanup_error: BaseException | None,
) -> None:
    if restore_error is None and cleanup_error is None:
        return
    if restore_error is not None:
        if cleanup_error is not None:
            raise restore_error from cleanup_error
        if original_error is not None:
            raise restore_error from original_error
        raise restore_error
    if cleanup_error is not None and original_error is not None:
        raise cleanup_error from original_error
    if cleanup_error is not None:
        raise cleanup_error


def promote_from_campaign(
    merged_path: Path,
    incumbent_path: Path | None = None,
    yes: bool = False,
    repo_root: Path | None = None,
) -> PromoteResult:
    """Wrap island_orchestrator.promote_if_better with the D3 staging contract.

    Pre-call: snapshot current ./solution.json bytes (or note its absence) and
    stage the candidate's verify.mjs-shaped array at ./solution.json so the
    hardcoded verify.mjs read sees the candidate.

    Call promote_if_better with verify_cmd=("node", "verify.mjs") and a
    candidate_out parameter that points OUTSIDE the repo root (so the
    promote_if_better internal write does not clobber our staged
    ./solution.json before verify.mjs reads it).

    Post-call:
      - if promoted: re-stage ./solution.json from the freshly promoted
        pool/best.json (the candidate is now the incumbent).
      - if not promoted (any reason): restore ./solution.json byte-for-byte
        from the snapshot, or delete it if it did not exist before the call.

    While this call runs, SIGTERM/SIGINT are temporarily rebound to raise
    PromoteInterrupted so cleanup runs before the prior handlers are restored.
    """
    if repo_root is None:
        repo_root = DEFAULT_REPO_ROOT
    repo_root = Path(repo_root).resolve()
    merged_path = _resolve_repo_path(repo_root, Path(merged_path))
    if incumbent_path is None:
        incumbent_path = DEFAULT_INCUMBENT
    incumbent_path = _resolve_repo_path(repo_root, Path(incumbent_path))
    solution_path = repo_root / SOLUTION_NAME
    lock_path = repo_root / LOCK_NAME

    with _temporary_interrupt_handlers():
        with _locked_repo(lock_path):
            with _pushd(repo_root):
                scratch_dir: Path | None = None
                pre_existed = False
                snapshot: bytes | None = None
                solution_staged = False
                promoted_successfully = False
                try:
                    scratch_dir = Path(
                        tempfile.mkdtemp(
                            prefix="cg_promote.",
                            dir=tempfile.gettempdir(),
                        )
                    )
                    scratch_dir.chmod(0o700)
                    candidate_out = scratch_dir / "island-candidate.json"
                    compat_archive = scratch_dir / "merged-harvest-compat.json"
                    pre_existed = solution_path.exists()
                    snapshot = solution_path.read_bytes() if pre_existed else None
                    try:
                        archive = _load_archive(merged_path)
                        rounded = _load_first_entry_scs(archive)
                    except FileNotFoundError:
                        return PromoteResult(False, f"archive not found: {merged_path}")
                    except json.JSONDecodeError as exc:
                        return PromoteResult(
                            False, f"merged archive not valid JSON: {exc}"
                        )
                    except ValueError as exc:
                        return PromoteResult(False, str(exc))
                    except KeyError as exc:
                        missing = exc.args[0] if exc.args else "unknown"
                        return PromoteResult(
                            False,
                            f"merged archive entry missing key: {missing}",
                        )

                    _stage_solution_array(solution_path, rounded)
                    solution_staged = True
                    _write_harvest_compat_archive(archive, compat_archive)
                    result = promote_if_better(
                        compat_archive,
                        incumbent_path=incumbent_path,
                        yes=yes,
                        verify_cmd=("node", "verify.mjs"),
                        candidate_out=candidate_out,
                    )
                    if result.promoted:
                        try:
                            payload = _solution_payload_from_best(incumbent_path)
                            _write_text_atomic(
                                solution_path,
                                json.dumps(payload),
                                tmp_suffix=".cg_promoted.tmp",
                            )
                        except (OSError, ValueError, KeyError, json.JSONDecodeError):
                            # If re-derivation fails, leave the staged candidate
                            # in place. Promotion already succeeded.
                            pass
                        promoted_successfully = True
                        return result
                    return result
                finally:
                    original_error = sys.exc_info()[1]
                    restore_error: BaseException | None = None
                    cleanup_error: BaseException | None = None
                    if solution_staged and not promoted_successfully:
                        try:
                            _restore_solution(solution_path, pre_existed, snapshot)
                        except BaseException as exc:
                            restore_error = exc
                    if scratch_dir is not None:
                        try:
                            shutil.rmtree(scratch_dir)
                        except BaseException as exc:
                            cleanup_error = exc
                    _raise_cleanup_errors(
                        original_error,
                        restore_error,
                        cleanup_error,
                    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a cg_orchestrator merged.json candidate to pool/best.json."
    )
    parser.add_argument("--merged", required=True, help="path to merged.json")
    parser.add_argument(
        "--incumbent", default="pool/best.json", help="path to pool/best.json"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive y/N prompt in promote_if_better",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = DEFAULT_REPO_ROOT
    merged_path = _resolve_repo_path(repo_root, Path(args.merged))
    incumbent_path = _resolve_repo_path(repo_root, Path(args.incumbent))

    if not merged_path.exists():
        print(f"ERROR: merged not found: {merged_path}", file=sys.stderr)
        return 2
    if not incumbent_path.exists():
        print(f"ERROR: incumbent not found: {incumbent_path}", file=sys.stderr)
        return 2
    try:
        with open(merged_path) as handle:
            json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: merged not valid JSON: {exc}", file=sys.stderr)
        return 2

    incumbent_size = incumbent_path.stat().st_size
    try:
        result = promote_from_campaign(
            merged_path,
            incumbent_path=incumbent_path,
            yes=args.yes,
            repo_root=repo_root,
        )
    except PromoteInterrupted as exc:
        print(
            f"ERROR: interrupted by {exc.signal_name} during promote cleanup window",
            file=sys.stderr,
        )
        return 128 + exc.signum
    new_size = incumbent_path.stat().st_size if incumbent_path.exists() else -1

    cand_R = "?" if result.candidate_R is None else f"{result.candidate_R:.12f}"
    inc_R = "?" if result.incumbent_R is None else f"{result.incumbent_R:.12f}"
    print(f"merged    = {merged_path}")
    print(f"incumbent = {incumbent_path}")
    print(f"candidate R = {cand_R}")
    print(f"incumbent R = {inc_R}")
    print(f"decision  = {'PROMOTED' if result.promoted else 'REJECTED'}")
    print(f"reason    = {result.reason}")
    print(f"incumbent bytes: pre={incumbent_size} post={new_size}")
    return 0 if result.promoted else 1


if __name__ == "__main__":
    raise SystemExit(main())
