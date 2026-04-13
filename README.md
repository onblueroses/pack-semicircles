# Pack Semicircles

Pack 15 unit semicircles into the smallest possible enclosing circle.

An entry for the [Optimization Arena](https://www.optimizationarena.com/packing) semicircle packing challenge. The problem: given 15 semicircles of radius 1, find positions (x, y) and orientations (theta) that minimize the radius of the minimum enclosing circle (MEC), with no overlaps.

**Current best: 2.9589** (theoretical lower bound ~2.74)

## Approach

### Scoring

The official scorer computes the MEC over boundary sample points (center + 2 flat-edge endpoints + 31 arc points per semicircle = 510 total points), using Welzl's algorithm. Coordinates are rounded to 6 decimal places before validation. Our optimizer uses the same MEC computation and overlap detection to match the official scorer exactly.

### Overlap Detection

Semicircle overlap is checked analytically via five geometric tests, matching the official TypeScript implementation with 1e-6 thresholds:

1. **Coincident centers** - if two semicircles share a center, they overlap unless they face exactly opposite directions (forming a full circle)
2. **Flat-flat intersection** - strict cross-product test for the two diameter segments
3. **Flat-arc intersection** - segment-circle intersection filtered by the half-plane condition
4. **Arc-arc intersection** - circle-circle intersection points filtered by both half-planes

### Minimum Enclosing Circle

Welzl's algorithm (randomized incremental construction) gives the exact MEC in O(n) expected time. Our numba implementation runs 3 passes over shuffled point order for robustness, handling the degenerate collinear case explicitly. ~13us per call.

### Single-Machine Optimizer (`run.py`)

Simulated annealing with basin hopping (iterated local search). Runs multiple SA chains from diverse initial configurations ("flower" patterns - pairs of opposing semicircles at regular angles around concentric rings). Each chain uses:

- Gaussian random walk in (x, y, theta) space
- Metropolis acceptance with exponential cooling
- Overlap resolution phase: accepts moves that reduce overlap count
- Basin hopping: periodic restarts from best-known solution with increasing kick strength
- Post-optimization quantized polish in 6-decimal coordinate space

### Parallel Tempering (`pt_optimizer.py`)

Replica exchange MCMC with multiple chains at different temperatures. This is the heavy-hitter that pushes scores below 2.96.

- **N chains** with log-spaced temperatures from 0.0003 (cold, exploiting) to 5.0 (hot, exploring)
- **Step sizes** scale with temperature - cold chains make small refinements, hot chains take large jumps
- **Basin hopping** within each chain (4 sub-runs with kicks between)
- **Replica exchange** after each round: randomly propose swaps between adjacent-temperature chains using the Metropolis criterion `delta = (beta_i - beta_j) * (score_j - score_i)`. Target swap rate: 40-60%
- **Reinjection**: global best reinjected into coldest chain every 5 rounds
- **Exploration reset**: hottest chains randomized every 50 rounds
- **Adaptive iterations**: doubles chain length when stuck for >20 rounds
- **Quantized polish**: final greedy descent in 6-decimal space

The temperature ladder is critical. Too narrow (all chains similar temperature) gives >80% swap rate with no diversity. Too wide gives <30% and chains can't communicate. The geometric spacing from 0.0003 to 5.0 across 15 chains gives ~56% swap rate.

### Rounding

The official scorer rounds coordinates to 6 decimal places before overlap checking. A solution that's valid at full precision can become invalid after rounding due to borderline overlaps at the 1e-6 threshold. Both optimizers validate solutions after rounding and discard any that don't survive.

## Usage

### Quick start (single machine)

```bash
pip install numpy numba
python run.py
```

Runs SA with basin hopping for ~30 minutes, saves best to `solution.json`.

### Parallel tempering (multi-core, long runs)

```bash
python pt_optimizer.py [hours]
```

Default: 120 hours. Uses N-1 CPU cores. Saves best to `semicircle_best.json` (with score metadata) and `solution.json` (submission format).

### Verify against official scorer

```bash
npm install
node verify.mjs
```

Reads `solution.json`, rounds to 6 decimals, checks overlaps, computes MEC radius using the same algorithm as the challenge scorer.

### Web UI

The interactive web visualizer (from the Optimization Arena scaffold):

```bash
npm install
npm run dev
```

Drag semicircles manually or let the browser-based worker swarm optimize. The Python optimizers are significantly faster due to numba JIT compilation.

## Solution Format

`solution.json` contains an array of 15 semicircles:

```json
[
  { "x": 4.422941, "y": 0.052069, "theta": 2.574761 },
  ...
]
```

Each semicircle has center (x, y) and orientation theta (radians). The flat edge is perpendicular to the direction (cos(theta), sin(theta)), with the arc on the positive side.

## Key Insights

- **Welzl over iterative 1-center**: The iterative approach (move toward farthest point) had 0.018 error vs exact Welzl. When your SA optimizes the wrong objective, it finds the wrong minimum.
- **Full MEC at every step**: A 6-point proxy score was 20x faster but led to solutions 0.1+ worse. The exact MEC is only 13us - cheap enough to evaluate at every SA step.
- **Rounding matters**: Solutions near the validity boundary at full precision can become invalid after 6-decimal rounding. Always validate post-rounding.
- **Temperature ladder tuning**: The single most impactful parameter in parallel tempering. Monitor swap rates between adjacent chains.
- **Two semicircles at same center**: Placing two semicircles at the same (x, y) with theta and theta+pi creates a valid full circle. The flower initialization exploits this.

## Project Structure

```
run.py              # Single-machine SA optimizer (numba)
pt_optimizer.py     # Parallel tempering MCMC (multiprocessing + numba)
verify.mjs          # Solution verifier matching official scorer
solution.json       # Current best solution
src/lib/geometry.ts # Overlap detection (TypeScript, used by web UI)
src/lib/welzl.ts    # Welzl MEC algorithm (TypeScript)
src/lib/worker.ts   # Web Worker SA optimizer (TypeScript)
```

## Dependencies

**Python optimizer**: numpy, numba (optional: CUDA-capable GPU for accelerated MEC)

**Verifier / Web UI**: Node.js 18+
