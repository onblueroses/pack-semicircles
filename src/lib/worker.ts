import { getMinEnclosingCircle } from './welzl';
import { semicirclesOverlap } from './geometry';

interface Semicircle {
    x: number;
    y: number;
    theta: number;
}

const N_SEMICIRCLES = 15;

let globalBestScs: Semicircle[] = [];
let globalBestScore = Infinity;

let bestValidScs: Semicircle[] = [];
let bestValidScore = Infinity;

let currentScs: Semicircle[] = [];
let currentScore = Infinity;

let running = false;
let workerId = 0;
let workerType = 'explorer';

let temp = 0.01;
let stepSize = 0.05;
let restarts = 0;

function getScore(scs: Semicircle[]) {
    const points = [];
    for (const sc of scs) {
        points.push({ x: sc.x, y: sc.y });
        for (let i = 0; i <= 30; i++) {
            const angle = sc.theta - Math.PI / 2 + (Math.PI * i) / 30;
            points.push({ x: sc.x + Math.cos(angle), y: sc.y + Math.sin(angle) });
        }
    }
    return getMinEnclosingCircle(points).r;
}

function countOverlaps(scs: Semicircle[]): number {
    let count = 0;
    for (let i = 0; i < N_SEMICIRCLES; i++) {
        for (let j = i + 1; j < N_SEMICIRCLES; j++) {
            if (semicirclesOverlap(scs[i], scs[j])) count++;
        }
    }
    return count;
}

// Box-Muller transform for rigorous Gaussian sampling
function gaussianRandom() {
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

const runBatch = () => {
    if (!running) return;

    let improved = false;
    let acceptedMoves = 0;
    const BATCH_SIZE = 2000;

    let currentOverlaps = countOverlaps(currentScs);

    for (let iter = 0; iter < BATCH_SIZE; iter++) {
        const idx = Math.floor(Math.random() * N_SEMICIRCLES);
        const sc = currentScs[idx];

        // Pure Gaussian Random Walk (Brownian motion in configuration space)
        const dx = gaussianRandom() * stepSize;
        const dy = gaussianRandom() * stepSize;
        const dtheta = gaussianRandom() * stepSize * Math.PI;

        const newSc = {
            x: sc.x + dx,
            y: sc.y + dy,
            theta: sc.theta + dtheta
        };

        const nextScs = [...currentScs];
        nextScs[idx] = newSc;

        if (currentOverlaps > 0) {
            // Phase 1: Overlap Resolution (Minimize constraint violations)
            const nextOverlaps = countOverlaps(nextScs);
            // Accept if it strictly reduces overlaps, or randomly walk flat spaces
            if (nextOverlaps < currentOverlaps || (nextOverlaps === currentOverlaps && Math.random() < 0.5)) {
                currentScs = nextScs;
                currentOverlaps = nextOverlaps;
                if (currentOverlaps === 0) {
                    currentScore = getScore(currentScs);
                }
            }
        } else {
            // Phase 2: Rigorous Simulated Annealing (Metropolis-Hastings)
            let valid = true;
            for (let i = 0; i < N_SEMICIRCLES; i++) {
                if (i !== idx && semicirclesOverlap(newSc, currentScs[i])) {
                    valid = false;
                    break;
                }
            }

            if (valid) {
                const nextScore = getScore(nextScs);
                const deltaE = nextScore - currentScore;
                
                // Metropolis Acceptance Criterion
                if (deltaE < 0 || Math.random() < Math.exp(-deltaE / temp)) {
                    currentScs = nextScs;
                    currentScore = nextScore;
                    acceptedMoves++;

                    if (currentScore < bestValidScore) {
                        bestValidScore = currentScore;
                        bestValidScs = currentScs.map(s => ({...s}));
                        improved = true;
                    }
                }
            }
        }
    }

    if (currentOverlaps === 0) {
        // Adaptive Step Size (Target acceptance rate ~ 0.234 for optimal MCMC mixing)
        const acceptanceRate = acceptedMoves / BATCH_SIZE;
        if (acceptanceRate > 0.234) {
            stepSize *= 1.02;
        } else {
            stepSize *= 0.98;
        }
        
        if (workerType === 'greedy') {
            stepSize = Math.max(0.00001, Math.min(stepSize, 0.01)); // Micro-steps for polishing
            temp = 0.0000001; // Strictly greedy
        } else {
            stepSize = Math.max(0.0001, Math.min(stepSize, 0.5));
            // Thermodynamic Cooling Schedule
            temp *= 0.95;
            
            // Basin Hopping / Iterated Local Search Restart
            if (temp < 0.00001) {
                restarts++;
                
                // 1. Adopt the global best as our starting point
                currentScs = globalBestScs.map(s => ({...s}));
                currentScore = globalBestScore;
                
                // 2. Apply a "Kick" (Ruin operator) to jump out of the local minimum
                // Different workers apply different kick strengths
                const kickStrength = 0.01 + (workerId / 16.0) * 0.15; // 0.01 to 0.16
                
                for (let i = 0; i < N_SEMICIRCLES; i++) {
                    currentScs[i].x += gaussianRandom() * kickStrength;
                    currentScs[i].y += gaussianRandom() * kickStrength;
                    currentScs[i].theta += gaussianRandom() * kickStrength * Math.PI;
                }
                
                currentOverlaps = countOverlaps(currentScs);
                if (currentOverlaps === 0) {
                    currentScore = getScore(currentScs);
                } else {
                    currentScore = Infinity;
                }
                
                // 3. Reheat for the next simulated annealing run
                temp = 0.005 + (workerId / 16.0) * 0.02; // 0.005 to 0.025
                stepSize = 0.1;
            }
        }
    } else {
        // If overlapping, increase step size to help escape
        stepSize = 0.1;
    }

    if (improved) {
        // Update our local copy of global best if we beat it
        if (bestValidScore < globalBestScore) {
            globalBestScore = bestValidScore;
            globalBestScs = bestValidScs.map(s => ({...s}));
        }
        self.postMessage({ type: 'IMPROVED', payload: { semicircles: bestValidScs, score: bestValidScore } });
    }

    self.postMessage({ 
        type: 'STATUS', 
        payload: { 
            id: workerId, 
            currentScore: currentOverlaps > 0 ? Infinity : currentScore, 
            bestScore: bestValidScore, 
            restarts: restarts,
            semicircles: currentScs.map(s => ({...s}))
        } 
    });

    setTimeout(runBatch, 0);
};

self.onmessage = (e) => {
    if (e.data.type === 'START') {
        workerId = e.data.payload.id;
        workerType = e.data.payload.workerType;
        
        globalBestScs = e.data.payload.semicircles.map((s: any) => ({...s}));
        
        const overlaps = countOverlaps(globalBestScs);
        if (overlaps === 0) {
            globalBestScore = getScore(globalBestScs);
        } else {
            globalBestScore = Infinity;
        }
        
        bestValidScs = globalBestScs.map(s => ({...s}));
        bestValidScore = globalBestScore;
        
        currentScs = globalBestScs.map(s => ({...s}));
        currentScore = globalBestScore;
        
        // Initial heat based on worker ID
        temp = 0.005 + (workerId / 16.0) * 0.02;
        stepSize = 0.1;
        restarts = 0;
        
        running = true;
        runBatch();
    } else if (e.data.type === 'STOP') {
        running = false;
    } else if (e.data.type === 'SYNC') {
        const newScore = e.data.payload.score;
        if (newScore < globalBestScore) {
            globalBestScore = newScore;
            globalBestScs = e.data.payload.semicircles.map((s: any) => ({...s}));
            
            // If we are currently frozen or overlapping, immediately adopt the new global best
            if (temp < 0.0001 || countOverlaps(currentScs) > 0) {
                currentScs = globalBestScs.map(s => ({...s}));
                currentScore = globalBestScore;
                temp = 0.005 + (workerId / 16.0) * 0.02;
                stepSize = 0.1;
            }
        }
    } else if (e.data.type === 'PULSE') {
        // Pulse is no longer needed for heating, Basin Hopping handles it naturally
    }
};
