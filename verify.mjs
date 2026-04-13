// Verify solution.json using the exact same logic as the challenge
import fs from "fs";

// --- geometry.ts ---
const EPSILON = 1e-9;

function distSq(a, b) {
	return (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
}
function dot(a, b) {
	return a.x * b.x + a.y * b.y;
}

function segmentCircleIntersect(a, b, c, r) {
	const v = { x: b.x - a.x, y: b.y - a.y };
	const w = { x: a.x - c.x, y: a.y - c.y };
	const A = dot(v, v);
	const B = 2 * dot(v, w);
	const C = dot(w, w) - r * r;
	const discriminant = B * B - 4 * A * C;
	if (discriminant < -EPSILON) return [];
	const t1 = (-B + Math.sqrt(Math.max(0, discriminant))) / (2 * A);
	const t2 = (-B - Math.sqrt(Math.max(0, discriminant))) / (2 * A);
	const pts = [];
	if (t1 >= -EPSILON && t1 <= 1 + EPSILON)
		pts.push({ x: a.x + t1 * v.x, y: a.y + t1 * v.y });
	if (t2 >= -EPSILON && t2 <= 1 + EPSILON)
		pts.push({ x: a.x + t2 * v.x, y: a.y + t2 * v.y });
	return pts;
}

function circlesIntersect(c1, r1, c2, r2) {
	const dSq_ = distSq(c1, c2);
	const d = Math.sqrt(dSq_);
	if (d > r1 + r2 + EPSILON || d < Math.abs(r1 - r2) - EPSILON || d < EPSILON)
		return [];
	const a = (r1 * r1 - r2 * r2 + dSq_) / (2 * d);
	const hSq = r1 * r1 - a * a;
	const h = Math.sqrt(Math.max(0, hSq));
	const p2 = {
		x: c1.x + (a * (c2.x - c1.x)) / d,
		y: c1.y + (a * (c2.y - c1.y)) / d,
	};
	return [
		{ x: p2.x + (h * (c2.y - c1.y)) / d, y: p2.y - (h * (c2.x - c1.x)) / d },
		{ x: p2.x - (h * (c2.y - c1.y)) / d, y: p2.y + (h * (c2.x - c1.x)) / d },
	];
}

function semicirclesOverlap(s1, s2) {
	const dSq_ = distSq(s1, s2);
	if (dSq_ > 4 + EPSILON) return false;
	const dir1 = { x: Math.cos(s1.theta), y: Math.sin(s1.theta) };
	const dir2 = { x: Math.cos(s2.theta), y: Math.sin(s2.theta) };
	if (dSq_ < 1e-6) {
		if (dot(dir1, dir2) > -1 + 1e-6) return true;
	}
	const getF = (s, dir) => {
		const dx = dir.y,
			dy = -dir.x;
		return [
			{ x: s.x + dx, y: s.y + dy },
			{ x: s.x - dx, y: s.y - dy },
		];
	};
	const f1 = getF(s1, dir1);
	const f2 = getF(s2, dir2);
	const ccw = (p1, p2, p3) =>
		(p3.y - p1.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p3.x - p1.x);
	const strictSeg = (a, b, c, d) => {
		const cp1 = ccw(a, b, c),
			cp2 = ccw(a, b, d),
			cp3 = ccw(c, d, a),
			cp4 = ccw(c, d, b);
		return (
			((cp1 > 1e-6 && cp2 < -1e-6) || (cp1 < -1e-6 && cp2 > 1e-6)) &&
			((cp3 > 1e-6 && cp4 < -1e-6) || (cp3 < -1e-6 && cp4 > 1e-6))
		);
	};
	if (strictSeg(f1[0], f1[1], f2[0], f2[1])) return true;
	const pts1 = segmentCircleIntersect(f1[0], f1[1], s2, 1);
	for (const p of pts1) {
		if (distSq(p, s1) < 1 - 1e-6) {
			if (dot({ x: p.x - s2.x, y: p.y - s2.y }, dir2) > 1e-6) return true;
		}
	}
	const pts2 = segmentCircleIntersect(f2[0], f2[1], s1, 1);
	for (const p of pts2) {
		if (distSq(p, s2) < 1 - 1e-6) {
			if (dot({ x: p.x - s1.x, y: p.y - s1.y }, dir1) > 1e-6) return true;
		}
	}
	const arcPts = circlesIntersect(s1, 1, s2, 1);
	for (const p of arcPts) {
		if (
			dot({ x: p.x - s1.x, y: p.y - s1.y }, dir1) > 1e-6 &&
			dot({ x: p.x - s2.x, y: p.y - s2.y }, dir2) > 1e-6
		)
			return true;
	}
	return false;
}

// --- welzl.ts ---
function getCircleCenter(bx, by, cx, cy) {
	const B = bx * bx + by * by;
	const C = cx * cx + cy * cy;
	const D = bx * cy - by * cx;
	return { x: (cy * B - by * C) / (2 * D), y: (bx * C - cx * B) / (2 * D) };
}

function circleFrom(A, B, C) {
	if (!C)
		return {
			x: (A.x + B.x) / 2,
			y: (A.y + B.y) / 2,
			r: Math.hypot(A.x - B.x, A.y - B.y) / 2,
		};
	const center = getCircleCenter(B.x - A.x, B.y - A.y, C.x - A.x, C.y - A.y);
	center.x += A.x;
	center.y += A.y;
	return {
		x: center.x,
		y: center.y,
		r: Math.hypot(center.x - A.x, center.y - A.y),
	};
}

function isValid(c, p) {
	return Math.hypot(c.x - p.x, c.y - p.y) <= c.r + 1e-6;
}

function welzlHelper(P, R, n) {
	if (n === 0 || R.length === 3) {
		if (R.length === 0) return { x: 0, y: 0, r: 0 };
		if (R.length === 1) return { x: R[0].x, y: R[0].y, r: 0 };
		if (R.length === 2) return circleFrom(R[0], R[1]);
		return circleFrom(R[0], R[1], R[2]);
	}
	const p = P[n - 1];
	const c = welzlHelper(P, R, n - 1);
	if (isValid(c, p)) return c;
	R.push(p);
	const res = welzlHelper(P, R, n - 1);
	R.pop();
	return res;
}

function getMinEnclosingCircle(points) {
	const P = [...points];
	for (let i = P.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[P[i], P[j]] = [P[j], P[i]];
	}
	return welzlHelper(P, [], P.length);
}

// --- worker.ts scoring ---
function getScore(scs) {
	const points = [];
	for (const sc of scs) {
		points.push({ x: sc.x, y: sc.y });
		for (let i = 0; i <= 30; i++) {
			const angle = sc.theta - Math.PI / 2 + (Math.PI * i) / 30;
			points.push({ x: sc.x + Math.cos(angle), y: sc.y + Math.sin(angle) });
		}
	}
	return getMinEnclosingCircle(points);
}

// --- Main ---
const data = JSON.parse(fs.readFileSync("solution.json", "utf-8"));

// Round to 6 decimal places
const scs = data.map((s) => ({
	x: Math.round(s.x * 1e6) / 1e6,
	y: Math.round(s.y * 1e6) / 1e6,
	theta: Math.round(s.theta * 1e6) / 1e6,
}));

console.log(`Semicircles: ${scs.length}`);

// Check overlaps
let overlaps = 0;
for (let i = 0; i < scs.length; i++) {
	for (let j = i + 1; j < scs.length; j++) {
		if (semicirclesOverlap(scs[i], scs[j])) {
			console.log(`  OVERLAP: ${i} and ${j}`);
			overlaps++;
		}
	}
}
console.log(`Overlaps: ${overlaps}`);

// Score (run multiple times due to random shuffle in Welzl)
let bestR = Infinity;
for (let trial = 0; trial < 10; trial++) {
	const mec = getScore(scs);
	if (mec.r < bestR) bestR = mec.r;
}
console.log(`MEC radius: ${bestR.toFixed(6)}`);
const valid = overlaps === 0 && scs.length === 15;
console.log(`Valid: ${valid}`);
if (!valid) process.exit(1);
