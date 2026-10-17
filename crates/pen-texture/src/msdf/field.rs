// The distance field before error correction: per texel, each contour's
// nearest edge for each channel (by true distance, ties to the edge the texel
// is more square to) gives that channel its pseudo-distance, and contours are
// combined as msdfgen's OverlappingContourCombiner does, so where glyphs
// overlap contours (as variable fonts do) the edges inside the union don't
// show. Then every texel's sign is checked against the shape's non-zero fill
// along its row, and flipped where they disagree (msdfgen's
// `distanceSignCorrection`), so no orientation or overlap of contours can
// turn a texel inside out.
use super::Shape;
use super::coloring::{self, Edge};
use super::geometry::{Distance, FAR, Point, Segment};

/// A shape ready to measure: degenerate segments gone, contours turned so
/// the inside is to the positive side, edges coloured, and each contour's
/// winding (1 around an inside, -1 around a hole).
pub(super) struct Prepared {
    pub contours: Vec<Vec<Edge>>,
    pub windings: Vec<i32>,
    /// Each edge's box (that of its control points, which holds it), for
    /// passing over edges no nearer than the nearest found.
    boxes: Vec<Vec<(Point, Point)>>,
}

/// Twice a contour's area, negated (msdfgen's shoelace sum): positive for a
/// contour with its inside to the positive side, after sampling its curves.
fn shoelace(contour: &[Segment]) -> f64 {
    let mut points = Vec::new();
    for segment in contour {
        points.push(segment.start());
        if !matches!(segment, Segment::Line(..)) {
            points.extend((1..8).map(|i| segment.point(f64::from(i) / 8.0)));
        }
    }
    let mut total = 0.0;
    let mut previous = points.last().copied().unwrap_or_default();
    for &point in &points {
        total += (point.x - previous.x) * (previous.y + point.y);
        previous = point;
    }
    total
}

fn sign(value: f64) -> i32 {
    i32::from(value > 0.0) - i32::from(value < 0.0)
}

impl Prepared {
    pub fn new(shape: &Shape) -> Self {
        let mut contours: Vec<Vec<Segment>> = shape
            .contours
            .iter()
            .map(|contour| contour.iter().copied().filter(|segment| !segment.is_degenerate()).collect())
            .filter(|contour: &Vec<Segment>| !contour.is_empty())
            .collect();
        // Fonts run their outer contours either way round (TrueType one,
        // PostScript outlines the other); the outer ones hold the most area.
        if contours.iter().map(|contour| shoelace(contour)).sum::<f64>() < 0.0 {
            for contour in &mut contours {
                contour.reverse();
                for segment in contour.iter_mut() {
                    *segment = segment.reversed();
                }
            }
        }
        let windings = contours.iter().map(|contour| sign(shoelace(contour))).collect();
        // A contour of one edge has no corner to colour from: make it three.
        let mut contours: Vec<Vec<Edge>> = contours
            .into_iter()
            .map(|contour| match contour[..] {
                [only] => only.split_in_thirds().map(Edge::white).to_vec(),
                _ => contour.into_iter().map(Edge::white).collect(),
            })
            .collect();
        coloring::color_edges(&mut contours);
        let boxes = contours.iter().map(|contour| contour.iter().map(|edge| edge.segment.hull()).collect()).collect();
        Prepared { contours, windings, boxes }
    }

    /// The field at `p`, in pixels, positive inside: the three channels, the
    /// true distance, and the single-channel pseudo-distance.
    pub fn sample(&self, p: Point, scratch: &mut Scratch) -> Sample {
        scratch.multi.clear();
        scratch.truth.clear();
        scratch.pseudo.clear();
        scratch.hints.resize(self.contours.len(), [usize::MAX; 3]);
        for ((contour, boxes), hints) in self.contours.iter().zip(&self.boxes).zip(&mut scratch.hints) {
            let mut nearest = [Nearest::NONE; 3];
            // The last sample's nearest edges, measured first, are likely
            // near here too, and let most others go unmeasured.
            for (i, &index) in hints.iter().enumerate() {
                if index < contour.len() && !hints[..i].contains(&index) {
                    measure(contour, index, p, &mut nearest);
                }
            }
            for (index, (edge, &(low, high))) in contour.iter().zip(boxes).enumerate() {
                if hints.contains(&index) {
                    continue;
                }
                let dx = (low.x - p.x).max(p.x - high.x).max(0.0);
                let dy = (low.y - p.y).max(p.y - high.y).max(0.0);
                let bound = (0..3)
                    .filter(|channel| edge.color & (1 << channel) != 0)
                    .map(|channel| nearest[channel].distance.distance.abs())
                    .fold(0.0, f64::max);
                if dx * dx + dy * dy <= bound * bound {
                    measure(contour, index, p, &mut nearest);
                }
            }
            *hints = nearest.map(|nearest| nearest.index);
            // Every edge has a channel, so the nearest of all is one of these.
            let all = nearest.into_iter().reduce(|a, b| if b.beats(&a) { b } else { a }).unwrap_or(Nearest::NONE);
            let pick = |nearest: Nearest| match contour.get(nearest.index) {
                None => Pick::NONE,
                Some(edge) => {
                    let mut distance = nearest.distance;
                    edge.segment.pseudo_distance(&mut distance, p, nearest.param);
                    Pick { nearest: nearest.distance, value: distance.distance }
                }
            };
            scratch.multi.push(nearest.map(pick));
            scratch.truth.push([Pick { nearest: all.distance, value: all.distance.distance }]);
            scratch.pseudo.push([pick(all)]);
        }
        let [truth] = combine(&scratch.truth, &self.windings);
        let [pseudo] = combine(&scratch.pseudo, &self.windings);
        Sample { multi: combine(&scratch.multi, &self.windings), truth, pseudo }
    }

    /// Where the horizontal line at `y` crosses the outline: x and direction
    /// (see `Segment::crossings`), sorted by x.
    pub fn crossings(&self, y: f64, out: &mut Vec<(f64, i32)>) {
        out.clear();
        for edge in self.contours.iter().flatten() {
            edge.segment.crossings(y, out);
        }
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
    }

    /// Whether `p` is inside the shape by the non-zero rule.
    pub fn inside(&self, p: Point, crossings: &mut Vec<(f64, i32)>) -> bool {
        crossings.clear();
        for edge in self.contours.iter().flatten() {
            edge.segment.crossings(p.y, crossings);
        }
        crossings.iter().filter(|crossing| crossing.0 < p.x).map(|crossing| crossing.1).sum::<i32>() != 0
    }
}

/// The nearest edge of a contour found so far for a channel: its distance,
/// its index, and where along it.
#[derive(Clone, Copy)]
struct Nearest {
    distance: Distance,
    index: usize,
    param: f64,
}

impl Nearest {
    const NONE: Nearest = Nearest { distance: Distance::FAR, index: usize::MAX, param: 0.0 };

    /// Nearer by distance, then by squareness, then the earlier edge: the
    /// same nearest whatever order the edges are measured in.
    fn beats(&self, other: &Nearest) -> bool {
        self.distance.closer_than(other.distance)
            || (!other.distance.closer_than(self.distance) && self.index < other.index)
    }
}

/// Measures edge `index` of `contour` from `p`, keeping it for each of its
/// channels it is the nearest for.
fn measure(contour: &[Edge], index: usize, p: Point, nearest: &mut [Nearest; 3]) {
    let edge = &contour[index];
    let (distance, param) = edge.segment.signed_distance(p);
    let found = Nearest { distance, index, param };
    for (channel, best) in nearest.iter_mut().enumerate() {
        if edge.color & (1 << channel) != 0 && found.beats(best) {
            *best = found;
        }
    }
}

/// What a sample finds at a point, in pixels, positive inside.
pub(super) struct Sample {
    pub multi: [f64; 3],
    pub truth: f64,
    pub pseudo: f64,
}

/// Room for a sample's per-contour picks, and its nearest edges as hints
/// for the next, kept between samples.
#[derive(Default)]
pub(super) struct Scratch {
    multi: Vec<[Pick; 3]>,
    truth: Vec<[Pick; 1]>,
    pseudo: Vec<[Pick; 1]>,
    /// Each contour's nearest edges for each channel at the last sample.
    hints: Vec<[usize; 3]>,
}

/// One contour's say in a channel at a point: the true distance to its
/// nearest edge there, to choose between contours by, and the value it gives.
#[derive(Clone, Copy)]
struct Pick {
    nearest: Distance,
    value: f64,
}

impl Pick {
    const NONE: Pick = Pick { nearest: Distance::FAR, value: -FAR };
}

pub(super) fn median(a: f64, b: f64, c: f64) -> f64 {
    a.min(b).max(a.max(b).min(c))
}

/// The distance picks stand for: the median of three channels, or the one.
fn resolve<const N: usize>(picks: &[Pick; N]) -> f64 {
    match picks.as_slice() {
        [r, g, b] => median(r.value, g.value, b.value),
        [one] => one.value,
        _ => unreachable!("fields have one or three channels"),
    }
}

/// Keeps, channel by channel, whichever of two contours' picks is nearer.
fn merge<const N: usize>(into: &mut [Pick; N], from: &[Pick; N]) {
    for (into, from) in into.iter_mut().zip(from) {
        if from.nearest.closer_than(into.nearest) {
            *into = *from;
        }
    }
}

/// The shape's value at a point from its contours' (msdfgen's
/// OverlappingContourCombiner): inside contours that overlap, the deepest of
/// them; in a hole, the hole's; elsewhere the nearest edges of all.
///
/// A contour with no corners has one two-channel colour, so its third channel
/// has no edge; where that contour's picks are kept, the channel takes their
/// median, not the farthest distance there is. The texel's median is the
/// same, but its channels then blend with neighbours that kept another
/// contour's picks (along a seam of overlapping contours) without a dip.
fn combine<const N: usize>(contours: &[[Pick; N]], windings: &[i32]) -> [f64; N] {
    let values = |picks: &[Pick; N]| {
        let m = resolve(picks);
        picks.map(|pick| if pick.nearest == Distance::FAR { m } else { pick.value })
    };
    let mut shape = [Pick::NONE; N];
    let mut inner = [Pick::NONE; N];
    let mut outer = [Pick::NONE; N];
    for (picks, &winding) in contours.iter().zip(windings) {
        let distance = resolve(picks);
        merge(&mut shape, picks);
        if winding > 0 && distance >= 0.0 {
            merge(&mut inner, picks);
        }
        if winding < 0 && distance <= 0.0 {
            merge(&mut outer, picks);
        }
    }
    let (inner_distance, outer_distance) = (resolve(&inner), resolve(&outer));
    let (mut best, best_winding) = if inner_distance >= 0.0 && inner_distance.abs() <= outer_distance.abs() {
        let mut best = inner;
        for (picks, &winding) in contours.iter().zip(windings) {
            let distance = resolve(picks);
            if winding > 0 && distance.abs() < outer_distance.abs() && distance > resolve(&best) {
                best = *picks;
            }
        }
        (best, 1)
    } else if outer_distance <= 0.0 && outer_distance.abs() < inner_distance.abs() {
        let mut best = outer;
        for (picks, &winding) in contours.iter().zip(windings) {
            let distance = resolve(picks);
            if winding < 0 && distance.abs() < inner_distance.abs() && distance < resolve(&best) {
                best = *picks;
            }
        }
        (best, -1)
    } else {
        return values(&shape);
    };
    for (picks, &winding) in contours.iter().zip(windings) {
        if winding != best_winding {
            let (distance, current) = (resolve(picks), resolve(&best));
            if distance * current >= 0.0 && distance.abs() < current.abs() {
                best = *picks;
            }
        }
    }
    if resolve(&best) == resolve(&shape) {
        best = shape;
    }
    values(&best)
}

/// Each texel's distances in pixels, positive inside, rows top first: the
/// three channels, then the true distance. Texel (x, y) is the one centred
/// on (x + 0.5, y + 0.5).
pub(super) fn distances(prepared: &Prepared, width: usize, height: usize) -> Vec<[f64; 4]> {
    let mut scratch = Scratch::default();
    let mut field = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let sample = prepared.sample(Point::new(x as f64 + 0.5, y as f64 + 0.5), &mut scratch);
            let [r, g, b] = sample.multi;
            field.push([r, g, b, sample.truth]);
        }
    }
    correct_signs(prepared, &mut field, width, height);
    field
}

/// Makes every texel's sign agree with the shape's non-zero fill: a texel
/// whose median disagrees has its channels negated, and one exactly on the
/// edge (a median of 0) follows most of its neighbours.
fn correct_signs(prepared: &Prepared, field: &mut [[f64; 4]], width: usize, height: usize) {
    let mut agrees = vec![0i8; field.len()];
    let mut ambiguous = false;
    let mut crossings = Vec::new();
    for y in 0..height {
        prepared.crossings(y as f64 + 0.5, &mut crossings);
        let (mut next, mut winding) = (0, 0);
        for x in 0..width {
            let center = x as f64 + 0.5;
            while next < crossings.len() && crossings[next].0 < center {
                winding += crossings[next].1;
                next += 1;
            }
            let inside = winding != 0;
            let i = y * width + x;
            let texel = &mut field[i];
            let m = median(texel[0], texel[1], texel[2]);
            if m == 0.0 {
                ambiguous = true;
            } else if (m > 0.0) != inside {
                for channel in &mut texel[..3] {
                    *channel = -*channel;
                }
                agrees[i] = -1;
            } else {
                agrees[i] = 1;
            }
            texel[3] = if inside { texel[3].abs() } else { -texel[3].abs() };
        }
    }
    if !ambiguous {
        return;
    }
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            if agrees[i] != 0 {
                continue;
            }
            let mut neighbours = 0;
            if x > 0 {
                neighbours += agrees[i - 1];
            }
            if x + 1 < width {
                neighbours += agrees[i + 1];
            }
            if y > 0 {
                neighbours += agrees[i - width];
            }
            if y + 1 < height {
                neighbours += agrees[i + width];
            }
            if neighbours < 0 {
                for channel in &mut field[i][..3] {
                    *channel = -*channel;
                }
            }
        }
    }
}
