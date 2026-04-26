# Island-Model Launch Recipe

Operational playbook for `island_orchestrator.py` against the 12-core target.
Reproduces the standing 6-hour overnight configuration: 12 epochs × 12 workers × 30 min,
chaos pulse every 50 iters, top-6 reseed per epoch.

## Pre-flight

```bash
# Verify a clean target host (no leftover workers, no stale tmux session)
pgrep -f mbh_driver       # expect empty
tmux ls                   # expect "no server running" or no `island` session

# Verify the seed pool exists (Phase 0 artifact: merged archives across prior runs)
ls -la /tmp/merged-all.json

# Verify tests pass on the target (one-off after first sync)
cd ~/pack-semicircles
python -m pytest test_chaos_pulse.py test_island_orchestrator.py test_mbh_driver.py
```

## Launch

`launch_island.sh`:

```bash
#!/bin/bash
set -e
TS=$(date +%Y%m%d-%H%M%S)
OUT=$HOME/pack-semicircles/runs/island-${TS}
mkdir -p $OUT
echo $OUT > /tmp/island_run_dir.txt
cd $HOME/pack-semicircles
exec python island_orchestrator.py \
  --epochs 12 \
  --workers 12 \
  --epoch-min 30 \
  --top-k 6 \
  --seed-pool /tmp/merged-all.json \
  --out-dir $OUT \
  --chaos-pulse-every-iters 50 \
  --seed 12345 \
  --hours-cap 7.0 \
  >> $OUT/launch.log 2>&1
```

Run inside tmux so the run survives SSH disconnects:

```bash
tmux new-session -d -s island /tmp/launch_island.sh
tmux ls                                # expect "island: 1 windows ..."
cat /tmp/island_run_dir.txt            # records the run output dir
```

## Monitor

```bash
RUN_DIR=$(cat /tmp/island_run_dir.txt)
tail -f $RUN_DIR/supervisor.log         # follow per-epoch progress
ls $RUN_DIR/epoch_*/w*/done_mbh.flag    # see which workers have reported
ls $RUN_DIR/merged_*.json               # see which epochs have merged

# Per-epoch best R from event logs (rough live view)
for f in $RUN_DIR/epoch_*/w*/archive.json; do
  python -c "import json,sys; d=json.load(open('$f')); print('$f', d.get('best_score'))"
done | sort -k2
```

## Stop early

Touch the orchestrator STOP file (graceful):

```bash
RUN_DIR=$(cat /tmp/island_run_dir.txt)
touch $RUN_DIR/STOP
```

Workers detect STOP within 1-2 iterations, write `done_mbh.flag`, and exit with
`exit_reason="stop_flag"`. The orchestrator runs the partial union merge,
writes `done_orchestrator.flag`, and exits.

`SIGTERM` to the orchestrator process produces the same path
(`stop_holder["stop"]=True` → STOP fanout → 30 s grace window).

## Wake-up check

```bash
RUN_DIR=$(cat /tmp/island_run_dir.txt)
tail -200 $RUN_DIR/supervisor.log

# Final harvest dry-run
cd ~/pack-semicircles
python harvest.py --archive $RUN_DIR/merged_$(ls $RUN_DIR/merged_*.json | wc -l | xargs -I{} expr {} - 1).json --incumbent pool/best.json --dry-run
```

## Promotion gate

If the dry-run reports a candidate with `R < 2.948694`, drive the gate via
`island_orchestrator.promote_if_better()`. Sequential, never auto-fires:

```bash
python -c "
import sys
sys.path.insert(0, 'home/onblueroses/pack-semicircles')  # adjust as needed
from island_orchestrator import promote_if_better
from pathlib import Path
res = promote_if_better(Path('$RUN_DIR/merged_FINAL.json'))
print(res)
"
```

The function:

1. Runs `harvest.py --dry-run` for sanity.
2. Compares candidate R to incumbent R; returns early on no improvement.
3. Extracts the candidate to `/tmp/island-candidate.json` matching `pool/best.json` schema.
4. Runs `node verify.mjs <candidate>`; returns early on non-zero exit.
5. Prompts `Promote? [y/N]` (skip with `yes=True`).
6. Only on user "y" does it invoke `harvest.py --yes`.

Steps (1)-(5) must succeed in order before step (6) runs. Bypass requires editing
the function.
