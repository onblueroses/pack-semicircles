# Contact-Graph Search Launch Recipe

Operational playbook for the contact-graph campaign. Three phases:
seed-library build (`seed_library.py`), parallel attack4 sweep
(`cg_orchestrator.py`), gated promotion (`cg_promote.py`).

## Prerequisites

The repo must be in a known-good state before launching. Verify each item:

```bash
# 1. Incumbent best exists. Promotion compares against this file.
ls -la pool/best.json

# 2. Seed pool exists (Phase 1 artifact). Each seed_*.json is a 15-piece
#    starting configuration consumed by attack4.py --root.
ls pool/seeds/seed_*.json | head
test -d pool/seeds || python seed_library.py --out pool/seeds --n-per-kind 8 --seed 0

# 3. verify.mjs is functional. It hardcodes ./solution.json (D3) and
#    ignores argv. The promotion wrapper stages solution.json around the
#    verify.mjs call.
node verify.mjs && echo OK || echo "verify.mjs broken; do not promote"

# 4. Tests pass.
python -m pytest tests/test_seed_library.py tests/test_cg_orchestrator.py tests/test_cg_promote.py -q

# 5. No stale workers from a previous run.
pgrep -f attack4.py    # expect empty
pgrep -f cg_orchestrator.py  # expect empty
```

## Command sequence

End-to-end Phase 1 to Phase 3. Edit the `HOURS` and `WORKERS` variables to match
the target host.

```bash
TS=$(date +%Y%m%d-%H%M%S)
OUT=runs/cg_${TS}
HOURS=2.0       # per-seed wall-clock budget for attack4
WORKERS=8       # parallel attack4 instances; <= core count
SEED_DIR=pool/seeds
mkdir -p "$OUT"

# Phase 1 (one-off; skip if pool/seeds already populated):
python seed_library.py --out $SEED_DIR --n-per-kind 8 --seed 0

# Phase 2: parallel campaign. Writes runs/cg_<ts>/<seed_id>/{stage_a.json,
# stage_b_depth*.json,run.log} per seed and runs/cg_<ts>/merged.json once
# all seeds finish (or STOP is honored).
python cg_orchestrator.py \
  --seed-dir $SEED_DIR \
  --hours $HOURS \
  --workers $WORKERS \
  --out $OUT \
  2>&1 | tee $OUT/launch.log

# Phase 3: gated promotion. Wraps island_orchestrator.promote_if_better with
# the D3 staging contract (snapshots ./solution.json, stages the candidate
# array form for verify.mjs, restores byte-for-byte on non-promote, leaves
# solution.json synced with the new pool/best.json on promote). During this
# call, cg_promote.py temporarily replaces SIGTERM/SIGINT handling so the
# wrapper can restore solution.json and remove its scratch dir before
# returning to the prior handlers.
python cg_promote.py --merged $OUT/merged.json
# Add --yes to skip the interactive y/N prompt:
# python cg_promote.py --merged $OUT/merged.json --yes
```

The CLI exits 0 on a successful promote, 1 on rejection, 2 on bad inputs.
Re-loading `pool/best.json` after a successful promote shows the new R.

## Abort

To stop the Phase 2 sweep gracefully without losing partial work:

```bash
# Find the run dir from the launch terminal or:
ls -td runs/cg_* | head -1

touch runs/cg_<ts>/STOP
```

`cg_orchestrator` polls the `STOP` file every second. On detection it fans
the file out to each active seed dir, gives workers a 30-second grace window
to write their own done flags, then terminates the worker pool. The
post-loop merge still runs; `merged.json` is written from whatever seeds
finished. Phase 3 can then be invoked against the partial merge as usual.

`SIGTERM` to the orchestrator PID produces the same code path because the
STOP file is what drives shutdown — there is no separate signal handler in
`cg_orchestrator`. If you must hard-kill, send `SIGKILL` after `SIGTERM`
times out; the kernel's PR_SET_PDEATHSIG ensures workers die with the parent.

## Rollback

If Phase 3 promoted a candidate that turned out to be wrong (e.g. verify.mjs
was buggy at the time of the promote), restore the prior `pool/best.json`:

```bash
# Inspect what is currently committed vs. on disk:
git diff pool/best.json
git log --oneline -- pool/best.json | head -5

# Discard the local change (only safe if pool/best.json is tracked):
git checkout pool/best.json

# Re-stage solution.json from the restored incumbent so downstream tools see
# the same state:
python -c "
import json
from pathlib import Path
d = json.load(open('pool/best.json'))
Path('solution.json').write_text(json.dumps(d['solution']))
"

# Confirm verify.mjs now passes against the restored state:
node verify.mjs
```

To inspect a single seed dir from the campaign (e.g. to understand why one
seed produced no improvement before re-running):

```bash
SEED_ID=seed_hex_3
ls runs/cg_<ts>/$SEED_ID/
tail -50 runs/cg_<ts>/$SEED_ID/run.log
python -c "
import json
for name in ['stage_a.json'] + [f'stage_b_depth{d}.json' for d in (1,2,3)]:
    p = f'runs/cg_<ts>/$SEED_ID/{name}'
    try:
        d = json.load(open(p))
        print(name, 'score=', d.get('score'))
    except FileNotFoundError:
        pass
"
```

A botched promote that did *not* commit is recoverable from the wrapper's
own restoration: any non-promote outcome already restores `solution.json`
byte-for-byte.

Test guard: `test_no_other_writer` greps the `*.py` file set for literal
`pool/best.json` writes (excluding `harvest.py`); this catches the most common
writer additions but does NOT prove a repo-wide invariant. The runtime
guarantee comes from the `.cg_promote.lock` `flock` plus `harvest.py`'s atomic
write; the test is a regression tripwire, not a proof.

## Threat model: known limitations

This wrapper is designed for single-user developer use. It does not defend
against:

- Local symlink-based attacks on `/tmp` scratch dirs (mitigation: `0o700`
  parent + per-pid name; full `O_EXCL`/`O_NOFOLLOW` would be needed for
  multi-tenant deployment).
- Concurrent invocations from outside this CLI (mitigation: `fcntl flock` on
  `.cg_promote.lock` — but only blocks other invocations using this wrapper).
- `SIGKILL` (cannot be trapped; will leak scratch dir and may leave
  `solution.json` staged).
- Multi-threaded callers: this wrapper uses `os.chdir()` process-globally for
  the duration of `promote_from_campaign`. It is NOT safe to call from a
  multi-threaded process if other threads do filesystem I/O. Single-process /
  single-thread CLI use is the supported pattern.
