// Outline geometry for the MSDF generator: points, and line, quadratic and
// cubic segments with what the field needs of each, after msdfgen's edge
// segments: the signed distance from a texel to a segment's nearest point,
// the pseudo-distance to the lines its ends point along, where a horizontal
// line crosses it (for the non-zero fill test), its bounds, and splitting.
//
// Distances are signed by which side of the segment's direction a point is
// on (positive where the cross product of the point's offset and the
// direction is positive); a shape is turned so that side is its inside.
use std::f64::consts::PI;
use std::ops::{Add, Mul, Sub};

/// Farther than any distance in a cell.
pub(super) const FAR: f64 = 1e240;

// msdfgen's search for a cubic's nearest point: Newton's method from five
// evenly spaced starts, four steps each.
const CUBIC_SEARCH_STARTS: u32 = 4;
const CUBIC_SEARCH_STEPS: u32 = 4;

// Halvings that find where a scanline crosses a curve: t to within 2^-48.
const BISECTIONS: u32 = 48;

/// A point or vector in pixels, x right and y down.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub(super) fn dot(self, other: Point) -> f64 {
        self.x * other.x + self.y * other.y
    }

    pub(super) fn cross(self, other: Point) -> f64 {
        self.x * other.y - self.y * other.x
    }

    pub(super) fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// This vector at unit length, or (0, 1) for a zero vector, as msdfgen
    /// has it.
    pub(super) fn normalize(self) -> Point {
        let length = self.length();
        if length == 0.0 { Point::new(0.0, 1.0) } else { self * (1.0 / length) }
    }
}

impl Add for Point {
    type Output = Point;
    fn add(self, other: Point) -> Point {
        Point::new(self.x + other.x, self.y + other.y)
    }
}

impl Sub for Point {
    type Output = Point;
    fn sub(self, other: Point) -> Point {
        Point::new(self.x - other.x, self.y - other.y)
    }
}

impl Mul<f64> for Point {
    type Output = Point;
    fn mul(self, scale: f64) -> Point {
        Point::new(self.x * scale, self.y * scale)
    }
}

/// The point a fraction `t` of the way from `a` to `b`, exactly `a` and `b`
/// at the ends.
fn mix(a: Point, b: Point, t: f64) -> Point {
    a * (1.0 - t) + b * t
}

/// 1 for a positive number, -1 otherwise.
fn non_zero_sign(value: f64) -> f64 {
    if value > 0.0 { 1.0 } else { -1.0 }
}

/// A signed distance to a segment, and how far from square on to the
/// segment the point lies (0 square on, 1 straight past an end along it):
/// of two segments equally near, as at a corner they share, the one the
/// point is more square to is the nearer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Distance {
    pub distance: f64,
    pub dot: f64,
}

impl Distance {
    /// Farther than anything, and outside.
    pub const FAR: Distance = Distance { distance: -FAR, dot: 0.0 };

    pub fn closer_than(self, other: Distance) -> bool {
        let (a, b) = (self.distance.abs(), other.distance.abs());
        a < b || (a == b && self.dot < other.dot)
    }
}

/// A piece of a contour: a line, or a quadratic or cubic Bézier curve, from
/// its first point to its last.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Segment {
    Line(Point, Point),
    Quad(Point, Point, Point),
    Cubic(Point, Point, Point, Point),
}

impl Segment {
    pub fn start(self) -> Point {
        match self {
            Segment::Line(p0, _) | Segment::Quad(p0, _, _) | Segment::Cubic(p0, _, _, _) => p0,
        }
    }

    pub fn end(self) -> Point {
        match self {
            Segment::Line(_, p1) => p1,
            Segment::Quad(_, _, p2) => p2,
            Segment::Cubic(_, _, _, p3) => p3,
        }
    }

    /// The point at `t`, from 0 at the start to 1 at the end.
    pub fn point(self, t: f64) -> Point {
        match self {
            Segment::Line(p0, p1) => mix(p0, p1, t),
            Segment::Quad(p0, p1, p2) => mix(mix(p0, p1, t), mix(p1, p2, t), t),
            Segment::Cubic(p0, p1, p2, p3) => {
                let p12 = mix(p1, p2, t);
                mix(mix(mix(p0, p1, t), p12, t), mix(p12, mix(p2, p3, t), t), t)
            }
        }
    }

    /// The same segment run the other way.
    pub fn reversed(self) -> Segment {
        match self {
            Segment::Line(p0, p1) => Segment::Line(p1, p0),
            Segment::Quad(p0, p1, p2) => Segment::Quad(p2, p1, p0),
            Segment::Cubic(p0, p1, p2, p3) => Segment::Cubic(p3, p2, p1, p0),
        }
    }

    /// Whether all its points are one: it has no length or direction.
    pub(super) fn is_degenerate(self) -> bool {
        match self {
            Segment::Line(p0, p1) => p0 == p1,
            Segment::Quad(p0, p1, p2) => p0 == p1 && p1 == p2,
            Segment::Cubic(p0, p1, p2, p3) => p0 == p1 && p1 == p2 && p2 == p3,
        }
    }

    /// The direction it runs at `t` (not unit length). Where a curve's control
    /// point sits on its end, the direction from the next control point on.
    pub(super) fn direction(self, t: f64) -> Point {
        match self {
            Segment::Line(p0, p1) => p1 - p0,
            Segment::Quad(p0, p1, p2) => {
                let tangent = mix(p1 - p0, p2 - p1, t);
                if tangent == Point::default() { p2 - p0 } else { tangent }
            }
            Segment::Cubic(p0, p1, p2, p3) => {
                let tangent = mix(mix(p1 - p0, p2 - p1, t), mix(p2 - p1, p3 - p2, t), t);
                if tangent != Point::default() {
                    tangent
                } else if t == 0.0 {
                    p2 - p0
                } else if t == 1.0 {
                    p3 - p1
                } else {
                    tangent
                }
            }
        }
    }

    /// The two halves either side of `t`, sharing the point there exactly.
    pub(super) fn split(self, t: f64) -> (Segment, Segment) {
        match self {
            Segment::Line(p0, p1) => {
                let m = mix(p0, p1, t);
                (Segment::Line(p0, m), Segment::Line(m, p1))
            }
            Segment::Quad(p0, p1, p2) => {
                let (a, b) = (mix(p0, p1, t), mix(p1, p2, t));
                let m = mix(a, b, t);
                (Segment::Quad(p0, a, m), Segment::Quad(m, b, p2))
            }
            Segment::Cubic(p0, p1, p2, p3) => {
                let (a, b, c) = (mix(p0, p1, t), mix(p1, p2, t), mix(p2, p3, t));
                let (ab, bc) = (mix(a, b, t), mix(b, c, t));
                let m = mix(ab, bc, t);
                (Segment::Cubic(p0, a, ab, m), Segment::Cubic(m, bc, c, p3))
            }
        }
    }

    pub(super) fn split_in_thirds(self) -> [Segment; 3] {
        let (first, rest) = self.split(1.0 / 3.0);
        let (second, third) = rest.split(0.5);
        [first, second, third]
    }

    /// Where, strictly between its ends, it turns back along `axis`
    /// (0 for x, 1 for y): at most two places, in order.
    fn turns(self, axis: usize) -> ([f64; 2], usize) {
        let along = |p: Point| if axis == 0 { p.x } else { p.y };
        let (mut roots, mut count) = match self {
            Segment::Line(..) => ([0.0; 2], 0),
            Segment::Quad(p0, p1, p2) => {
                let bend = along(p0) - 2.0 * along(p1) + along(p2);
                if bend == 0.0 { ([0.0; 2], 0) } else { ([(along(p0) - along(p1)) / bend, 0.0], 1) }
            }
            Segment::Cubic(p0, p1, p2, p3) => {
                let (a0, a1, a2) = (along(p1) - along(p0), along(p2) - along(p1), along(p3) - along(p2));
                solve_quadratic(a0 - 2.0 * a1 + a2, 2.0 * (a1 - a0), a0)
            }
        };
        let mut kept = 0;
        for i in 0..count {
            if roots[i] > 0.0 && roots[i] < 1.0 && (kept == 0 || roots[i] != roots[0]) {
                roots[kept] = roots[i];
                kept += 1;
            }
        }
        count = kept;
        if count == 2 && roots[1] < roots[0] {
            roots.swap(0, 1);
        }
        (roots, count)
    }

    /// The smallest box holding it: its least and greatest corners.
    pub(super) fn bounds(self) -> (Point, Point) {
        let (start, end) = (self.start(), self.end());
        let mut min = Point::new(start.x.min(end.x), start.y.min(end.y));
        let mut max = Point::new(start.x.max(end.x), start.y.max(end.y));
        for axis in 0..2 {
            let (roots, count) = self.turns(axis);
            for &t in &roots[..count] {
                let p = self.point(t);
                min = Point::new(min.x.min(p.x), min.y.min(p.y));
                max = Point::new(max.x.max(p.x), max.y.max(p.y));
            }
        }
        (min, max)
    }

    /// The box of its control points, which holds it: least and greatest
    /// corners.
    pub(super) fn hull(self) -> (Point, Point) {
        let points = match self {
            Segment::Line(p0, p1) => [p0, p1, p1, p1],
            Segment::Quad(p0, p1, p2) => [p0, p1, p2, p2],
            Segment::Cubic(p0, p1, p2, p3) => [p0, p1, p2, p3],
        };
        let min = points.iter().fold(points[0], |a, b| Point::new(a.x.min(b.x), a.y.min(b.y)));
        let max = points.iter().fold(points[0], |a, b| Point::new(a.x.max(b.x), a.y.max(b.y)));
        (min, max)
    }

    /// The signed distance from `origin` to its nearest point, and where
    /// along it (`t`) that is, past 0 or 1 when the nearest point is an end
    /// and `origin` lies beyond it along the segment's direction there
    /// (msdfgen's `signedDistance`).
    pub(super) fn signed_distance(self, origin: Point) -> (Distance, f64) {
        match self {
            Segment::Line(p0, p1) => {
                let aq = origin - p0;
                let ab = p1 - p0;
                let param = aq.dot(ab) / ab.dot(ab);
                let eq = if param > 0.5 { p1 } else { p0 } - origin;
                let endpoint = eq.length();
                if param > 0.0 && param < 1.0 {
                    let ortho = aq.cross(ab) / ab.length();
                    if ortho.abs() < endpoint {
                        return (Distance { distance: ortho, dot: 0.0 }, param);
                    }
                }
                let dot = ab.normalize().dot(eq.normalize()).abs();
                (Distance { distance: non_zero_sign(aq.cross(ab)) * endpoint, dot }, param)
            }
            Segment::Quad(p0, p1, p2) => {
                let qa = p0 - origin;
                let ab = p1 - p0;
                let br = p2 - p1 - ab;
                // Where the curve's tangent is square to the line to origin.
                let (roots, count) =
                    solve_cubic(br.dot(br), 3.0 * ab.dot(br), 2.0 * ab.dot(ab) + qa.dot(br), qa.dot(ab));
                let (mut min, mut param) = self.nearest_end(origin);
                for &t in &roots[..count] {
                    if t > 0.0 && t < 1.0 {
                        let qe = qa + ab * (2.0 * t) + br * (t * t);
                        let distance = qe.length();
                        if distance <= min.abs() {
                            min = non_zero_sign((ab + br * t).cross(qe)) * distance;
                            param = t;
                        }
                    }
                }
                self.distance_at(min, param, origin)
            }
            Segment::Cubic(p0, p1, p2, p3) => {
                let qa = p0 - origin;
                let ab = p1 - p0;
                let br = p2 - p1 - ab;
                let az = (p3 - p2) - (p2 - p1) - br;
                let offset = |t: f64| qa + ab * (3.0 * t) + br * (3.0 * t * t) + az * (t * t * t);
                let tangent = |t: f64| ab * 3.0 + br * (6.0 * t) + az * (3.0 * t * t);
                let (mut min, mut param) = self.nearest_end(origin);
                for start in 0..=CUBIC_SEARCH_STARTS {
                    let mut t = f64::from(start) / f64::from(CUBIC_SEARCH_STARTS);
                    let mut qe = offset(t);
                    for _ in 0..CUBIC_SEARCH_STEPS {
                        let d1 = tangent(t);
                        let d2 = br * 6.0 + az * (6.0 * t);
                        t -= qe.dot(d1) / (d1.dot(d1) + qe.dot(d2));
                        if t <= 0.0 || t >= 1.0 || t.is_nan() {
                            break;
                        }
                        qe = offset(t);
                        let distance = qe.length();
                        if distance < min.abs() {
                            min = non_zero_sign(tangent(t).cross(qe)) * distance;
                            param = t;
                        }
                    }
                }
                self.distance_at(min, param, origin)
            }
        }
    }

    /// The signed distance to the nearer end of a curve, and its `t`
    /// measured along the direction there (below 0 or above 1 past it).
    fn nearest_end(self, origin: Point) -> (f64, f64) {
        let qa = self.start() - origin;
        let dir = self.direction(0.0);
        let mut min = non_zero_sign(dir.cross(qa)) * qa.length();
        let mut param = -qa.dot(dir) / dir.dot(dir);
        let bq = self.end() - origin;
        let distance = bq.length();
        if distance < min.abs() {
            let dir = self.direction(1.0);
            min = non_zero_sign(dir.cross(bq)) * distance;
            param = 1.0 - bq.dot(dir) / dir.dot(dir);
        }
        (min, param)
    }

    fn distance_at(self, distance: f64, param: f64, origin: Point) -> (Distance, f64) {
        let dot = if (0.0..=1.0).contains(&param) {
            0.0
        } else if param < 0.5 {
            self.direction(0.0).normalize().dot((self.start() - origin).normalize()).abs()
        } else {
            self.direction(1.0).normalize().dot((self.end() - origin).normalize()).abs()
        };
        (Distance { distance, dot }, param)
    }

    /// Turns a distance to an end into the pseudo-distance to the line the
    /// segment runs along there, where `origin` lies beyond that end and the
    /// line is no farther (msdfgen's `distanceToPseudoDistance`).
    pub(super) fn pseudo_distance(self, distance: &mut Distance, origin: Point, param: f64) {
        let (offset, dir, beyond) = if param < 0.0 {
            let dir = self.direction(0.0).normalize();
            let aq = origin - self.start();
            (aq, dir, aq.dot(dir) < 0.0)
        } else if param > 1.0 {
            let dir = self.direction(1.0).normalize();
            let bq = origin - self.end();
            (bq, dir, bq.dot(dir) > 0.0)
        } else {
            return;
        };
        if beyond {
            let pseudo = offset.cross(dir);
            if pseudo.abs() <= distance.distance.abs() {
                *distance = Distance { distance: pseudo, dot: 0.0 };
            }
        }
    }

    /// Adds where the horizontal line at `y` crosses it: each crossing's x,
    /// and 1 where y grows along the segment there, -1 where it shrinks. Each
    /// stretch between the points where it turns back counts its smaller-y
    /// end and not its other, so a line through a joint of two segments or
    /// a turning point counts once, or not at all.
    pub(super) fn crossings(self, y: f64, out: &mut Vec<(f64, i32)>) {
        let (low, high) = self.hull();
        if y < low.y || y >= high.y {
            return;
        }
        if let Segment::Line(p0, p1) = self {
            if (y >= p0.y && y < p1.y) || (y >= p1.y && y < p0.y) {
                let x = p0.x + (p1.x - p0.x) * ((y - p0.y) / (p1.y - p0.y));
                out.push((x, if p1.y > p0.y { 1 } else { -1 }));
            }
            return;
        }
        let (turns, count) = self.turns(1);
        let mut ts = [0.0; 4];
        let mut ys = [self.start().y; 4];
        for (i, &t) in turns[..count].iter().enumerate() {
            ts[i + 1] = t;
            ys[i + 1] = self.point(t).y;
        }
        ts[count + 1] = 1.0;
        ys[count + 1] = self.end().y;
        for i in 0..=count {
            let (y0, y1) = (ys[i], ys[i + 1]);
            if (y >= y0 && y < y1) || (y >= y1 && y < y0) {
                let rising = y1 > y0;
                let (mut lo, mut hi) = (ts[i], ts[i + 1]);
                for _ in 0..BISECTIONS {
                    let mid = 0.5 * (lo + hi);
                    if (self.point(mid).y < y) == rising {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                out.push((self.point(0.5 * (lo + hi)).x, if rising { 1 } else { -1 }));
            }
        }
    }
}

/// Real roots of a x² + b x + c = 0 and how many there are (none for 0 = 0),
/// as msdfgen solves it: nearly linear equations as linear.
pub(super) fn solve_quadratic(a: f64, b: f64, c: f64) -> ([f64; 2], usize) {
    if a == 0.0 || b.abs() > 1e12 * a.abs() {
        if b == 0.0 {
            return ([0.0; 2], 0);
        }
        return ([-c / b, 0.0], 1);
    }
    let discriminant = b * b - 4.0 * a * c;
    if discriminant > 0.0 {
        let root = discriminant.sqrt();
        ([(-b + root) / (2.0 * a), (-b - root) / (2.0 * a)], 2)
    } else if discriminant == 0.0 {
        ([-b / (2.0 * a), 0.0], 1)
    } else {
        ([0.0; 2], 0)
    }
}

/// Real roots of x³ + a x² + b x + c = 0.
fn solve_cubic_normed(a: f64, b: f64, c: f64) -> ([f64; 3], usize) {
    let a2 = a * a;
    let q = (a2 - 3.0 * b) / 9.0;
    let r = (a * (2.0 * a2 - 9.0 * b) + 27.0 * c) / 54.0;
    let (r2, q3) = (r * r, q * q * q);
    let a = a / 3.0;
    if r2 < q3 {
        let t = (r / q3.sqrt()).clamp(-1.0, 1.0).acos();
        let q = -2.0 * q.sqrt();
        let roots = [(t / 3.0).cos(), ((t + 2.0 * PI) / 3.0).cos(), ((t - 2.0 * PI) / 3.0).cos()];
        (roots.map(|root| q * root - a), 3)
    } else {
        let u = if r < 0.0 { 1.0 } else { -1.0 } * (r.abs() + (r2 - q3).sqrt()).cbrt();
        let v = if u == 0.0 { 0.0 } else { q / u };
        let first = (u + v) - a;
        if u == v || (u - v).abs() < 1e-12 * (u + v).abs() {
            ([first, -0.5 * (u + v) - a, 0.0], 2)
        } else {
            ([first, 0.0, 0.0], 1)
        }
    }
}

/// Real roots of a x³ + b x² + c x + d = 0 and how many there are, as
/// msdfgen solves it: nearly quadratic equations as quadratic.
pub(super) fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> ([f64; 3], usize) {
    if a != 0.0 {
        let bn = b / a;
        // Past this ratio the error grows larger than treating a as 0 makes.
        if bn.abs() < 1e6 {
            return solve_cubic_normed(bn, c / a, d / a);
        }
    }
    let (roots, count) = solve_quadratic(b, c, d);
    ([roots[0], roots[1], 0.0], count)
}
