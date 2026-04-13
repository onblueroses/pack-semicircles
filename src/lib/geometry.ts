export interface Point { x: number; y: number; }
export interface Semicircle { x: number; y: number; theta: number; }

const EPSILON = 1e-9;

function distSq(a: Point, b: Point): number {
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
}

function dot(a: Point, b: Point): number {
    return a.x * b.x + a.y * b.y;
}

function pointInSemicircle(p: Point, sc: Semicircle): boolean {
    const dSq = distSq(p, sc);
    if (dSq > 1 - EPSILON) return false;
    const dir = { x: Math.cos(sc.theta), y: Math.sin(sc.theta) };
    const d = dot({ x: p.x - sc.x, y: p.y - sc.y }, dir);
    return d > EPSILON;
}

function segmentsIntersect(a: Point, b: Point, c: Point, d: Point): boolean {
    const ccw = (p1: Point, p2: Point, p3: Point) => 
        (p3.y - p1.y) * (p2.x - p1.x) > (p2.y - p1.y) * (p3.x - p1.x);
    return ccw(a, c, d) !== ccw(b, c, d) && ccw(a, b, c) !== ccw(a, b, d);
}

function segmentCircleIntersect(a: Point, b: Point, c: Point, r: number): Point[] {
    const v = { x: b.x - a.x, y: b.y - a.y };
    const w = { x: a.x - c.x, y: a.y - c.y };
    const A = dot(v, v);
    const B = 2 * dot(v, w);
    const C = dot(w, w) - r * r;
    const discriminant = B * B - 4 * A * C;
    
    if (discriminant < -EPSILON) return [];
    
    const t1 = (-B + Math.sqrt(Math.max(0, discriminant))) / (2 * A);
    const t2 = (-B - Math.sqrt(Math.max(0, discriminant))) / (2 * A);
    
    const pts: Point[] = [];
    if (t1 >= -EPSILON && t1 <= 1 + EPSILON) pts.push({ x: a.x + t1 * v.x, y: a.y + t1 * v.y });
    if (t2 >= -EPSILON && t2 <= 1 + EPSILON) pts.push({ x: a.x + t2 * v.x, y: a.y + t2 * v.y });
    return pts;
}

function circlesIntersect(c1: Point, r1: number, c2: Point, r2: number): Point[] {
    const dSq = distSq(c1, c2);
    const d = Math.sqrt(dSq);
    if (d > r1 + r2 + EPSILON || d < Math.abs(r1 - r2) - EPSILON || d < EPSILON) return [];
    
    const a = (r1 * r1 - r2 * r2 + dSq) / (2 * d);
    const hSq = r1 * r1 - a * a;
    const h = Math.sqrt(Math.max(0, hSq));
    
    const p2 = {
        x: c1.x + a * (c2.x - c1.x) / d,
        y: c1.y + a * (c2.y - c1.y) / d
    };
    
    return [
        {
            x: p2.x + h * (c2.y - c1.y) / d,
            y: p2.y - h * (c2.x - c1.x) / d
        },
        {
            x: p2.x - h * (c2.y - c1.y) / d,
            y: p2.y + h * (c2.x - c1.x) / d
        }
    ];
}

export function semicirclesOverlap(s1: Semicircle, s2: Semicircle): boolean {
    const dSq = distSq(s1, s2);
    if (dSq > 4 + EPSILON) return false;

    const dir1 = { x: Math.cos(s1.theta), y: Math.sin(s1.theta) };
    const dir2 = { x: Math.cos(s2.theta), y: Math.sin(s2.theta) };

    // 1. Check if identical or nearly identical centers
    if (dSq < 1e-6) {
        // If they don't form a perfect full circle (facing opposite directions), they overlap
        if (dot(dir1, dir2) > -1 + 1e-6) {
            return true;
        }
    }

    // Helper to get flat edge endpoints
    const getF = (s: Semicircle, dir: Point) => {
        const dx = dir.y; // sin(theta)
        const dy = -dir.x; // -cos(theta)
        return [
            { x: s.x + dx, y: s.y + dy },
            { x: s.x - dx, y: s.y - dy }
        ];
    };

    const f1 = getF(s1, dir1);
    const f2 = getF(s2, dir2);

    // 2. Flat edge crosses Flat edge (strict intersection)
    const ccw = (p1: Point, p2: Point, p3: Point) => 
        (p3.y - p1.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p3.x - p1.x);
    
    const strictSegmentsIntersect = (a: Point, b: Point, c: Point, d: Point) => {
        const cp1 = ccw(a, b, c);
        const cp2 = ccw(a, b, d);
        const cp3 = ccw(c, d, a);
        const cp4 = ccw(c, d, b);
        return ((cp1 > 1e-6 && cp2 < -1e-6) || (cp1 < -1e-6 && cp2 > 1e-6)) &&
               ((cp3 > 1e-6 && cp4 < -1e-6) || (cp3 < -1e-6 && cp4 > 1e-6));
    };

    if (strictSegmentsIntersect(f1[0], f1[1], f2[0], f2[1])) return true;

    // 3. Flat edge 1 crosses Arc 2
    const pts1 = segmentCircleIntersect(f1[0], f1[1], s2, 1);
    for (const p of pts1) {
        // Is intersection strictly inside the flat edge segment?
        if (distSq(p, s1) < 1 - 1e-6) {
            // Is intersection strictly inside the arc's half-plane?
            if (dot({ x: p.x - s2.x, y: p.y - s2.y }, dir2) > 1e-6) {
                return true;
            }
        }
    }

    // 4. Flat edge 2 crosses Arc 1
    const pts2 = segmentCircleIntersect(f2[0], f2[1], s1, 1);
    for (const p of pts2) {
        if (distSq(p, s2) < 1 - 1e-6) {
            if (dot({ x: p.x - s1.x, y: p.y - s1.y }, dir1) > 1e-6) {
                return true;
            }
        }
    }

    // 5. Arc 1 crosses Arc 2
    const arcPts = circlesIntersect(s1, 1, s2, 1);
    for (const p of arcPts) {
        if (dot({ x: p.x - s1.x, y: p.y - s1.y }, dir1) > 1e-6 &&
            dot({ x: p.x - s2.x, y: p.y - s2.y }, dir2) > 1e-6) {
            return true;
        }
    }

    return false;
}

export function isValidPacking(semicircles: Semicircle[]): boolean {
    for (let i = 0; i < semicircles.length; i++) {
        for (let j = i + 1; j < semicircles.length; j++) {
            if (semicirclesOverlap(semicircles[i], semicircles[j])) {
                return false;
            }
        }
    }
    return true;
}
