export interface Point { x: number; y: number; }
export interface Circle { x: number; y: number; r: number; }

function getCircleCenter(bx: number, by: number, cx: number, cy: number): Point {
    const B = bx * bx + by * by;
    const C = cx * cx + cy * cy;
    const D = bx * cy - by * cx;
    return {
        x: (cy * B - by * C) / (2 * D),
        y: (bx * C - cx * B) / (2 * D)
    };
}

function circleFrom(A: Point, B: Point, C?: Point): Circle {
    if (!C) {
        return {
            x: (A.x + B.x) / 2,
            y: (A.y + B.y) / 2,
            r: Math.hypot(A.x - B.x, A.y - B.y) / 2
        };
    }
    const center = getCircleCenter(B.x - A.x, B.y - A.y, C.x - A.x, C.y - A.y);
    center.x += A.x;
    center.y += A.y;
    return {
        x: center.x,
        y: center.y,
        r: Math.hypot(center.x - A.x, center.y - A.y)
    };
}

function isValid(c: Circle, p: Point): boolean {
    return Math.hypot(c.x - p.x, c.y - p.y) <= c.r + 1e-6;
}

function welzlHelper(P: Point[], R: Point[], n: number): Circle {
    if (n === 0 || R.length === 3) {
        if (R.length === 0) return { x: 0, y: 0, r: 0 };
        if (R.length === 1) return { x: R[0].x, y: R[0].y, r: 0 };
        if (R.length === 2) return circleFrom(R[0], R[1]);
        return circleFrom(R[0], R[1], R[2]);
    }

    const p = P[n - 1];
    const c = welzlHelper(P, R, n - 1);

    if (isValid(c, p)) {
        return c;
    }

    R.push(p);
    const res = welzlHelper(P, R, n - 1);
    R.pop();
    return res;
}

export function getMinEnclosingCircle(points: Point[]): Circle {
    const P = [...points];
    // Shuffle to ensure O(N) expected time
    for (let i = P.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [P[i], P[j]] = [P[j], P[i]];
    }
    return welzlHelper(P, [], P.length);
}
