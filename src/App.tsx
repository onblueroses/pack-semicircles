import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Play, Square, RotateCcw, Download, Upload, Info } from 'lucide-react';
import { getMinEnclosingCircle } from './lib/welzl';
import { semicirclesOverlap, isValidPacking } from './lib/geometry';
import OptimizationWorker from './lib/worker.ts?worker';

interface Semicircle {
  x: number;
  y: number;
  theta: number;
}

const N_SEMICIRCLES = 15;
const RADIUS = 1;

const initialWorkerStats = Array.from({ length: 16 }).reduce((acc, _, i) => {
  acc[i] = {
    id: i,
    type: i < 4 ? 'greedy' : 'explorer',
    currentScore: Infinity,
    bestScore: Infinity,
    restarts: 0,
    semicircles: []
  };
  return acc;
}, {} as Record<number, WorkerStat>);

// Generate vertices for a semicircle of radius 1
const STEPS = 30;
const vertices = [{ x: 0, y: 0 }];
for (let i = 0; i <= STEPS; i++) {
  const angle = -Math.PI / 2 + (Math.PI * i) / STEPS;
  vertices.push({
    x: Math.cos(angle),
    y: Math.sin(angle)
  });
}

function generateInitial(): Semicircle[] {
  const arr: Semicircle[] = [];
  for (let i = 0; i < N_SEMICIRCLES; i++) {
    arr.push({
      x: (i % 4) * 2.5 - 3.75,
      y: Math.floor(i / 4) * 2.5 - 3.75,
      theta: Math.random() * Math.PI * 2
    });
  }
  return arr;
}

function generateFibonacci(): Semicircle[] {
  const arr: Semicircle[] = [];
  const goldenAngle = Math.PI * (3 - Math.sqrt(5)); // ~137.5 degrees
  
  for (let i = 0; i < N_SEMICIRCLES; i++) {
    const r = Math.sqrt(i) * 1.5;
    const theta = i * goldenAngle;
    
    arr.push({
      x: r * Math.cos(theta),
      y: r * Math.sin(theta),
      theta: theta + Math.PI
    });
  }
  
  return arr;
}

function generateDoubleSpiral(): Semicircle[] {
  const arr: Semicircle[] = [];
  const goldenAngle = Math.PI * (3 - Math.sqrt(5)); // ~137.5 degrees
  
  for (let i = 0; i < N_SEMICIRCLES; i++) {
    const arm = i % 2; // 0 or 1
    const step = Math.floor(i / 2);
    
    const r = 1.0 + Math.sqrt(step) * 1.5;
    const theta = step * goldenAngle + (arm * Math.PI);
    
    arr.push({
      x: r * Math.cos(theta),
      y: r * Math.sin(theta),
      theta: theta + Math.PI
    });
  }
  
  return arr;
}

interface WorkerStat {
  id: number;
  type: string;
  currentScore: number;
  bestScore: number;
  restarts: number;
  semicircles?: Semicircle[];
}

export default function App() {
  const [strategy, setStrategy] = useState<'evolution'>('evolution');
  const [semicircles, setSemicircles] = useState<Semicircle[]>(() => {
    const saved = localStorage.getItem('best_semicircles');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length === N_SEMICIRCLES) {
          return parsed;
        }
      } catch (e) {
        console.error('Failed to parse saved semicircles', e);
      }
    }
    return generateInitial();
  });
  const [simulating, setSimulating] = useState(false);
  const [workersCount, setWorkersCount] = useState(0);
  const [workerStats, setWorkerStats] = useState<Record<number, WorkerStat>>(initialWorkerStats);
  
  const [draggingIdx, setDraggingIdx] = useState<number | null>(null);
  const [isRotating, setIsRotating] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);
  const workersRef = useRef<Worker[]>([]);

  // Prevent default scrolling when using the mouse wheel on the canvas
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const preventScroll = (e: WheelEvent) => {
      e.preventDefault();
    };
    svg.addEventListener('wheel', preventScroll, { passive: false });
    return () => svg.removeEventListener('wheel', preventScroll);
  }, []);

  const enclosingCircle = useMemo(() => {
    const points: { x: number; y: number }[] = [];
    semicircles.forEach(sc => {
      points.push({ x: sc.x, y: sc.y });
      for (let i = 0; i <= STEPS; i++) {
        const angle = sc.theta - Math.PI / 2 + (Math.PI * i) / STEPS;
        points.push({
          x: sc.x + Math.cos(angle),
          y: sc.y + Math.sin(angle)
        });
      }
    });
    return getMinEnclosingCircle(points);
  }, [semicircles]);

  useEffect(() => {
    localStorage.setItem('best_semicircles', JSON.stringify(semicircles));
  }, [semicircles]);

  const archiveRef = useRef<{score: number, semicircles: Semicircle[]}[]>([]);
  const latestScsRef = useRef<Semicircle[]>(semicircles);
  useEffect(() => { latestScsRef.current = semicircles; }, [semicircles]);
  const [pulseActive, setPulseActive] = useState(false);

  useEffect(() => {
    if (!simulating) {
      workersRef.current.forEach(w => w.terminate());
      workersRef.current = [];
      setWorkersCount(0);
      archiveRef.current = [];
      return;
    }

    // Force exactly 16 workers as requested
    const numWorkers = 16;
    setWorkersCount(numWorkers);
    
    const workers: Worker[] = [];
    let currentBestScore = enclosingCircle.r;
    let lastImprovementTime = Date.now();

    for (let i = 0; i < numWorkers; i++) {
      const worker = new OptimizationWorker();
      workers.push(worker);
      
      // 4 greedy polishers, the rest are fractal explorers
      const workerType = i < 4 ? 'greedy' : 'explorer';

      worker.onmessage = (e) => {
        if (e.data.type === 'IMPROVED') {
          const { semicircles: newScs, score } = e.data.payload;
          if (score < currentBestScore) {
            currentBestScore = score;
            lastImprovementTime = Date.now();
            setSemicircles(newScs);
            
            // Add the new global best to the Archive so explorers can branch from it!
            const archive = archiveRef.current;
            const isDiverse = !archive.some(a => Math.abs(a.score - score) < 0.001);
            if (isDiverse) {
              archive.push({ score, semicircles: newScs });
              archive.sort((a, b) => a.score - b.score);
              if (archive.length > 20) archive.pop();
            }

            // Broadcast to others (only greedy workers will adopt it)
            workers.forEach(w => {
              if (w !== worker) {
                w.postMessage({ type: 'SYNC', payload: { semicircles: newScs, score } });
              }
            });
          }
        } else if (e.data.type === 'STATUS') {
          setWorkerStats(prev => ({
            ...prev,
            [e.data.payload.id]: { ...e.data.payload, type: workerType }
          }));
        } else if (e.data.type === 'STAGNATED') {
          const { score, semicircles, id } = e.data.payload;
          const archive = archiveRef.current;
          
          if (score !== Infinity) {
            // Simple diversity check: don't add if we have one with almost identical score
            const isDiverse = !archive.some(a => Math.abs(a.score - score) < 0.001);
            
            if (isDiverse) {
              archive.push({ score, semicircles });
              archive.sort((a, b) => a.score - b.score);
              if (archive.length > 20) archive.pop(); // Keep top 20
            }
          }

          // Pick a seed for the worker to restart from
          // Bias towards better scores (e.g. exponential distribution)
          const seedIdx = Math.floor(Math.pow(Math.random(), 2) * archive.length);
          const seed = archive[seedIdx] || archive[0] || { semicircles };

          worker.postMessage({ type: 'RESTART_SEED', payload: { semicircles: seed.semicircles } });
        }
      };
      
      worker.postMessage({ type: 'START', payload: { semicircles, workerType, id: i } });
    }
    
    workersRef.current = workers;

    const pulseInterval = setInterval(() => {
      if (Date.now() - lastImprovementTime > 5000) {
        setPulseActive(true);
        setTimeout(() => setPulseActive(false), 1500);
        workers.forEach(w => {
          w.postMessage({ type: 'PULSE', payload: { semicircles: latestScsRef.current } });
        });
        lastImprovementTime = Date.now();
      }
    }, 1000);

    return () => {
      clearInterval(pulseInterval);
      workers.forEach(w => w.terminate());
      workersRef.current = [];
    };
  }, [simulating]);

  const handleExport = () => {
    if (overlappingIndices.size > 0) {
      if (!window.confirm('There are overlapping semicircles. This is an invalid solution. Export anyway?')) {
        return;
      }
    }
    // Round to 6 decimal places as required
    const data = semicircles.map(sc => ({
      x: Number(sc.x.toFixed(6)),
      y: Number(sc.y.toFixed(6)),
      theta: Number(sc.theta.toFixed(6))
    }));
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'solution.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target?.result as string);
        if (Array.isArray(data) && data.length === N_SEMICIRCLES) {
          setSemicircles(data);
          setSimulating(false);
        } else {
          alert('Invalid JSON format. Expected array of 15 semicircles.');
        }
      } catch (err) {
        alert('Failed to parse JSON.');
      }
    };
    reader.readAsText(file);
  };

  const getSvgPoint = (e: React.PointerEvent) => {
    if (!svgRef.current) return { x: 0, y: 0 };
    const svg = svgRef.current;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    return pt.matrixTransform(svg.getScreenCTM()?.inverse());
  };

  const handlePointerDown = (e: React.PointerEvent) => {
    if (simulating) return;
    const pt = getSvgPoint(e);
    
    // Find clicked semicircle (reverse order for z-index)
    let clickedIdx = -1;
    for (let i = semicircles.length - 1; i >= 0; i--) {
      const sc = semicircles[i];
      const dx = pt.x - sc.x;
      const dy = pt.y - sc.y;
      const dist = Math.hypot(dx, dy);
      if (dist <= RADIUS) {
        // Check if inside semicircle
        const angle = Math.atan2(dy, dx);
        let diff = angle - sc.theta;
        while (diff < -Math.PI) diff += 2 * Math.PI;
        while (diff > Math.PI) diff -= 2 * Math.PI;
        if (Math.abs(diff) <= Math.PI / 2) {
          clickedIdx = i;
          break;
        }
      }
    }

    if (clickedIdx !== -1) {
      setDraggingIdx(clickedIdx);
      setIsRotating(e.shiftKey || e.button === 2); // Right click or shift for rotation
      if (e.button === 2) e.preventDefault();
      (e.target as Element).setPointerCapture(e.pointerId);
    }
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (draggingIdx === null || simulating) return;
    const pt = getSvgPoint(e);
    
    setSemicircles(prev => {
      const next = [...prev];
      const sc = { ...next[draggingIdx] };
      
      if (isRotating) {
        sc.theta = Math.atan2(pt.y - sc.y, pt.x - sc.x);
      } else {
        sc.x = pt.x;
        sc.y = pt.y;
      }
      
      next[draggingIdx] = sc;
      return next;
    });
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    if (draggingIdx !== null) {
      (e.target as Element).releasePointerCapture(e.pointerId);
      setDraggingIdx(null);
      setIsRotating(false);
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (simulating) return;
    const pt = getSvgPoint(e as any);
    let hoveredIdx = -1;
    for (let i = semicircles.length - 1; i >= 0; i--) {
      const sc = semicircles[i];
      const dx = pt.x - sc.x;
      const dy = pt.y - sc.y;
      const dist = Math.hypot(dx, dy);
      if (dist <= RADIUS) {
        const angle = Math.atan2(dy, dx);
        let diff = angle - sc.theta;
        while (diff < -Math.PI) diff += 2 * Math.PI;
        while (diff > Math.PI) diff -= 2 * Math.PI;
        if (Math.abs(diff) <= Math.PI / 2) {
          hoveredIdx = i;
          break;
        }
      }
    }
    if (hoveredIdx !== -1) {
      e.preventDefault();
      setSemicircles(prev => {
        const next = [...prev];
        next[hoveredIdx] = { ...next[hoveredIdx], theta: next[hoveredIdx].theta + Math.sign(e.deltaY) * 0.2 };
        return next;
      });
    }
  };

  // SVG ViewBox calculations
  const r = enclosingCircle.r === Infinity || enclosingCircle.r === 0 ? 5 : enclosingCircle.r;
  const viewBoxSize = r * 2.5;
  const viewBox = `${enclosingCircle.x - viewBoxSize/2} ${enclosingCircle.y - viewBoxSize/2} ${viewBoxSize} ${viewBoxSize}`;

  // Find overlapping indices
  const overlappingIndices = useMemo(() => {
    const overlaps = new Set<number>();
    for (let i = 0; i < semicircles.length; i++) {
      for (let j = i + 1; j < semicircles.length; j++) {
        if (semicirclesOverlap(semicircles[i], semicircles[j])) {
          overlaps.add(i);
          overlaps.add(j);
        }
      }
    }
    return overlaps;
  }, [semicircles]);

  return (
    <div className="h-screen w-screen overflow-hidden bg-gray-50 flex font-sans text-gray-900">
      {/* Sidebar */}
      <div className="w-80 bg-white border-r border-gray-200 p-5 flex flex-col shadow-sm z-10 overflow-y-auto shrink-0">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight">Pack Semicircles</h1>
          <p className="text-sm text-gray-500 mt-1">Challenge Optimizer</p>
        </div>

        <div className="bg-blue-50 rounded-xl p-5 mb-6 border border-blue-100 shrink-0">
          <div className="text-sm font-medium text-blue-800 mb-1">Enclosing Radius</div>
          <div className="text-4xl font-bold text-blue-600 font-mono">
            {enclosingCircle.r === Infinity ? '---' : enclosingCircle.r.toFixed(6)}
          </div>
          <div className="text-xs text-blue-500 mt-2 flex items-center">
            <Info className="w-3 h-3 mr-1" />
            Theoretical lower bound: ~2.74
          </div>
          {overlappingIndices.size > 0 && (
            <div className="mt-3 text-xs font-medium text-red-600 bg-red-50 p-2 rounded border border-red-100">
              Warning: {overlappingIndices.size} semicircles are overlapping. This is an invalid packing.
            </div>
          )}
        </div>

        <div className="text-sm text-gray-600 mb-6 bg-gray-50 p-4 rounded-lg border border-gray-200 shrink-0">
          <p className="font-medium mb-1 text-gray-800">Controls:</p>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>Drag</strong> to move semicircles</li>
            <li><strong>Shift + Drag</strong> (or Right-click + Drag) to rotate</li>
          </ul>
        </div>

        <div className="space-y-4 mb-6 shrink-0">
          <div className="grid grid-cols-1 gap-2 bg-gray-100 p-1 rounded-lg">
            <button
              onClick={() => setStrategy('evolution')}
              className={`py-2 px-2 rounded-md text-xs font-medium transition-colors ${
                strategy === 'evolution' ? 'bg-white text-indigo-700 shadow-sm' : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Parallel Tempering MCMC
            </button>
          </div>

          <button
            onClick={() => setSimulating(!simulating)}
            className={`w-full py-3 px-4 rounded-lg font-medium flex items-center justify-center transition-colors ${
              simulating 
                ? 'bg-red-100 text-red-700 hover:bg-red-200' 
                : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-sm'
            }`}
          >
            {simulating ? (
              <><Square className="w-5 h-5 mr-2" /> Stop Simulation</>
            ) : (
              <><Play className="w-5 h-5 mr-2" /> Start Packing</>
            )}
          </button>

          <button
            onClick={() => { setSimulating(false); setSemicircles(generateInitial()); }}
            className="w-full py-2.5 px-4 rounded-lg font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 flex items-center justify-center transition-colors"
          >
            <RotateCcw className="w-4 h-4 mr-2" /> Reset (Grid)
          </button>

          <button
            onClick={() => { setSimulating(false); setSemicircles(generateFibonacci()); }}
            className="w-full py-2.5 px-4 rounded-lg font-medium bg-amber-100 text-amber-800 hover:bg-amber-200 flex items-center justify-center transition-colors"
          >
            <RotateCcw className="w-4 h-4 mr-2" /> Reset (Fibonacci Seed)
          </button>

          <button
            onClick={() => { setSimulating(false); setSemicircles(generateDoubleSpiral()); }}
            className="w-full py-2.5 px-4 rounded-lg font-medium bg-emerald-100 text-emerald-800 hover:bg-emerald-200 flex items-center justify-center transition-colors"
          >
            <RotateCcw className="w-4 h-4 mr-2" /> Reset (Double Spiral)
          </button>
        </div>

        {simulating && (
          <div className="bg-indigo-50 rounded-xl p-4 mb-6 border border-indigo-100 shrink-0">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-indigo-900">Evolution Engine</span>
              <span className="flex h-3 w-3 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-indigo-500"></span>
              </span>
            </div>
            <div className="text-xs text-indigo-700">
              Running <strong>{workersCount}</strong> parallel workers.<br/>
              Exploring mutations and sharing best generations...
            </div>
            {pulseActive && (
              <div className="mt-3 text-xs font-bold text-amber-700 bg-amber-100 p-2 rounded border border-amber-200 animate-pulse flex items-center">
                <RotateCcw className="w-3 h-3 mr-1" />
                Stagnation detected. Injecting Chaos Pulse!
              </div>
            )}
          </div>
        )}

        <div className="mt-auto space-y-3 shrink-0 pt-4">
          <button
            onClick={handleExport}
            className="w-full py-2 px-4 rounded-lg font-medium border border-gray-300 text-gray-700 hover:bg-gray-50 flex items-center justify-center transition-colors"
          >
            <Download className="w-4 h-4 mr-2" /> Export JSON
          </button>
          
          <label className="w-full py-2 px-4 rounded-lg font-medium border border-gray-300 text-gray-700 hover:bg-gray-50 flex items-center justify-center cursor-pointer transition-colors">
            <Upload className="w-4 h-4 mr-2" /> Import JSON
            <input type="file" accept=".json" className="hidden" onChange={handleImport} />
          </label>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-row overflow-hidden">
        {/* Main View */}
        <div className="w-[450px] relative overflow-hidden bg-white flex flex-col border-r border-gray-200 shrink-0">
          <div className="p-4 border-b border-gray-200 bg-gray-50 shrink-0">
            <h2 className="text-sm font-bold text-gray-800 uppercase tracking-wider">Current Best</h2>
            <p className="text-xs text-gray-500 mt-1">Global minimum radius: {enclosingCircle.r === Infinity ? '---' : enclosingCircle.r.toFixed(6)}</p>
          </div>
          <div className="flex-1 relative flex items-center justify-center p-4">
            {/* Grid Background */}
            <div className="absolute inset-0 pointer-events-none opacity-[0.03]" 
                 style={{ backgroundImage: 'radial-gradient(circle at 2px 2px, black 1px, transparent 0)', backgroundSize: '40px 40px' }}>
            </div>

            <svg 
              ref={svgRef}
              className="w-full h-full drop-shadow-sm touch-none" 
              viewBox={viewBox}
              preserveAspectRatio="xMidYMid meet"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={handlePointerUp}
              onContextMenu={e => e.preventDefault()}
              onWheel={handleWheel}
            >
          {/* Enclosing Circle */}
          <circle 
            cx={enclosingCircle.x} 
            cy={enclosingCircle.y} 
            r={enclosingCircle.r} 
            fill="rgba(79, 70, 229, 0.03)" 
            stroke="rgba(79, 70, 229, 0.4)" 
            strokeWidth={viewBoxSize * 0.002} 
            strokeDasharray={`${viewBoxSize * 0.01} ${viewBoxSize * 0.01}`}
          />
          
          {/* Center Mark */}
          <circle cx={enclosingCircle.x} cy={enclosingCircle.y} r={viewBoxSize * 0.005} fill="rgba(79, 70, 229, 0.5)" />

          {/* Semicircles */}
          {semicircles.map((sc, i) => {
            const isOverlapping = overlappingIndices.has(i);
            const isDragging = draggingIdx === i;
            return (
            <g 
              key={i} 
              transform={`translate(${sc.x}, ${sc.y}) rotate(${sc.theta * 180 / Math.PI})`}
              className={isDragging ? 'cursor-grabbing' : 'cursor-grab'}
            >
              <path 
                d="M 0 -1 A 1 1 0 0 1 0 1 Z" 
                fill={isOverlapping ? "rgba(239, 68, 68, 0.2)" : "rgba(99, 102, 241, 0.2)"} 
                stroke={isOverlapping ? "rgba(220, 38, 38, 0.8)" : "rgba(79, 70, 229, 0.8)"} 
                strokeWidth={viewBoxSize * 0.002} 
                strokeLinejoin="round"
              />
              {/* Center of disk indicator */}
              <circle cx={0} cy={0} r={viewBoxSize * 0.005} fill={isOverlapping ? "rgba(220, 38, 38, 0.8)" : "rgba(79, 70, 229, 0.8)"} />
              {/* Direction indicator */}
              <line x1={0} y1={0} x2={0.5} y2={0} stroke={isOverlapping ? "rgba(220, 38, 38, 0.5)" : "rgba(79, 70, 229, 0.5)"} strokeWidth={viewBoxSize * 0.002} />
            </g>
          )})}
          </svg>
        </div>
      </div>

      {/* Right Area - Worker Visualization */}
      <div className="flex-1 bg-gray-50 flex flex-col shadow-sm z-10 overflow-hidden">
        <div className="p-4 border-b border-gray-200 bg-white shrink-0">
          <h2 className="text-sm font-bold text-gray-800 uppercase tracking-wider">Worker Swarm (4x4 Grid)</h2>
          <p className="text-xs text-gray-500 mt-1">Live configuration states - 4 Polishers, 12 Explorers</p>
        </div>
        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-4 gap-4 h-full auto-rows-fr">
            {Object.values(workerStats).sort((a, b) => a.id - b.id).map(stat => (
              <div key={stat.id} className="bg-white border border-gray-200 rounded-xl p-3 flex flex-col items-center shadow-sm hover:shadow-md transition-shadow">
                <div className="text-xs text-gray-500 w-full flex justify-between font-mono mb-2">
                  <span className="font-bold flex items-center">
                    <span className={`w-2 h-2 rounded-full mr-1.5 ${stat.type === 'greedy' ? 'bg-amber-400' : 'bg-indigo-400'}`}></span>
                    W{stat.id}
                  </span>
                  <span className={stat.bestScore < enclosingCircle.r ? "text-green-600 font-bold" : ""}>
                    {stat.bestScore === Infinity ? (stat.currentScore === Infinity ? '---' : stat.currentScore.toFixed(3)) : stat.bestScore.toFixed(3)}
                  </span>
                </div>
                <div className="flex-1 w-full flex items-center justify-center min-h-0">
                  <svg viewBox="-4 -4 8 8" className="w-full h-full max-h-full bg-gray-50 rounded-lg border border-gray-100">
                    {stat.semicircles && stat.semicircles.map((sc, i) => (
                      <g key={i} transform={`translate(${sc.x}, ${sc.y}) rotate(${sc.theta * 180 / Math.PI})`}>
                        <path 
                          d="M 0 -1 A 1 1 0 0 1 0 1 Z" 
                          fill={stat.type === 'greedy' ? "rgba(245, 158, 11, 0.15)" : "rgba(99, 102, 241, 0.15)"} 
                          stroke={stat.type === 'greedy' ? "rgba(217, 119, 6, 0.8)" : "rgba(79, 70, 229, 0.8)"} 
                          strokeWidth="0.05" 
                        />
                      </g>
                    ))}
                  </svg>
                </div>
                <div className="text-[10px] text-gray-400 w-full flex justify-between mt-2">
                  <span className="uppercase tracking-wider">{stat.type}</span>
                  <span>Restarts: {stat.restarts}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  </div>
  );
}
