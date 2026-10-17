// Error correction for the MSDF generator, msdfgen's MSDFErrorCorrection
// in its default mode (edge priority, distances checked at the edge). Where
// two neighbouring texels' channels, interpolated as a GPU filters them,
// would give a median that crosses the edge where neither texel does, or
// strays further than a distance could over a texel's step, the texel
// farther from the edge is an artifact, and correcting it sets its three
// channels to their median: a plain distance there. Texels around corners,
// and those whose channels draw the edge between them and a neighbour, are
// protected: they are corrected only for a median that crosses the edge, and
// only when the shape's own distance at that spot shows correcting helps.
use std::cell::RefCell;
use std::f64::consts::SQRT_2;

use super::coloring::{BLUE, GREEN, RED};
use super::field::{Prepared, Scratch, median};
use super::geometry::{Point, solve_quadratic};

const PROTECTED: u8 = 1;
const ERROR: u8 = 2;

// What a range test finds of an interpolated median: worth checking, and
// straying beyond what a distance could.
const CANDIDATE: u8 = 1;
const ARTIFACT: u8 = 2;

/// How much more than a texel's step's worth of distance an interpolated
/// median must stray by to be an artifact.
const MIN_DEVIATION_RATIO: f64 = 10.0 / 9.0;
/// How much nearer the shape's distance correcting must bring a median.
const MIN_IMPROVE_RATIO: f64 = 10.0 / 9.0;
/// Two texels are either side of an edge when their medians' magnitudes sum
/// to less than this many steps between them.
const PROTECTION_RADIUS_TOLERANCE: f64 = 1.001;
/// Channels meeting this near a texel do so in every texel; not artifacts.
const ARTIFACT_T_EPSILON: f64 = 0.01;

type Rgb = [f64; 3];

fn rgb(texel: &[f64; 4]) -> Rgb {
    [texel[0], texel[1], texel[2]]
}

fn median_of(channels: &Rgb) -> f64 {
    median(channels[0], channels[1], channels[2])
}

fn mix(a: f64, b: f64, t: f64) -> f64 {
    (1.0 - t) * a + t * b
}

/// The median of `a` and `b`'s channels interpolated `t` of the way.
fn interpolated_median(a: &Rgb, b: &Rgb, t: f64) -> f64 {
    median(mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t))
}

/// The median at `t` along the diagonal of four texels, each channel
/// a + l t + q t² there.
fn diagonal_median(a: &Rgb, l: &Rgb, q: &Rgb, t: f64) -> f64 {
    let at = |i: usize| t * (t * q[i] + l[i]) + a[i];
    median(at(0), at(1), at(2))
}

/// Texel (x, y)'s index, for a texel that is in the field.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "whole, non-negative, in the field")]
fn texel_index(x: f64, y: f64, width: usize, height: usize) -> Option<usize> {
    (x >= 0.0 && y >= 0.0 && x < width as f64 && y < height as f64).then(|| y as usize * width + x as usize)
}

/// A whole texel coordinate clamped into 0..size.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "whole, clamped to the field first")]
fn clamped(coordinate: f64, size: usize) -> usize {
    coordinate.clamp(0.0, (size - 1) as f64) as usize
}

/// The field's channels interpolated at `p` (in pixels), clamped at its
/// edges as a GPU samples it.
fn bilinear(field: &[[f64; 4]], width: usize, height: usize, p: Point) -> Rgb {
    let (x, y) = (p.x - 0.5, p.y - 0.5);
    let (left, top) = (x.floor(), y.floor());
    let (tx, ty) = (x - left, y - top);
    let (l, r) = (clamped(left, width), clamped(left + 1.0, width));
    let (t, b) = (clamped(top, height), clamped(top + 1.0, height));
    let at = |column: usize, row: usize, channel: usize| field[row * width + column][channel];
    [0, 1, 2].map(|c| mix(mix(at(l, t, c), at(r, t, c), tx), mix(at(l, b, c), at(r, b, c), tx), ty))
}

/// The shape's own distance, for checking candidate artifacts against: its
/// single-channel pseudo-distance, signed by its non-zero fill as the field
/// is.
struct Reference<'a> {
    prepared: &'a Prepared,
    field: &'a [[f64; 4]],
    width: usize,
    height: usize,
    scratch: RefCell<(Scratch, Vec<(f64, i32)>)>,
}

impl Reference<'_> {
    fn distance(&self, p: Point) -> f64 {
        let mut scratch = self.scratch.borrow_mut();
        let (scratch, crossings) = &mut *scratch;
        let distance = self.prepared.sample(p, scratch).pseudo.abs();
        if self.prepared.inside(p, crossings) { distance } else { -distance }
    }
}

/// Judges the medians interpolated between one texel and a neighbour.
struct Classifier<'a> {
    /// How far a distance may change over the step to the neighbour, by the
    /// deviation ratio.
    span: f64,
    protected: bool,
    /// For the distance check: the shape, the texel's centre and channels,
    /// and the step to the neighbour.
    exact: Option<(&'a Reference<'a>, Point, Rgb, Point)>,
}

impl Classifier<'_> {
    /// Whether the median `xm` at `xt`, between `am` at `at` and `bm` at
    /// `bt`, is a candidate artifact, and whether it strays beyond what a
    /// distance could. For a protected texel only a median that crosses the
    /// edge where neither end does is a candidate; for others, any outside
    /// the ends' range.
    fn range_test(&self, at: f64, bt: f64, xt: f64, am: f64, bm: f64, xm: f64) -> u8 {
        let crosses = (am > 0.0 && bm > 0.0 && xm <= 0.0) || (am < 0.0 && bm < 0.0 && xm >= 0.0);
        if !crosses && (self.protected || median(am, bm, xm) == xm) {
            return 0;
        }
        let (a_span, b_span) = ((xt - at) * self.span, (bt - xt) * self.span);
        let within = xm >= am - a_span && xm <= am + a_span && xm >= bm - b_span && xm <= bm + b_span;
        if within { CANDIDATE } else { CANDIDATE | ARTIFACT }
    }

    /// Whether what the range tests found at `t` makes the texel an
    /// artifact: when it strays, or, checking distances, when setting the
    /// texel to its median brings the interpolated median there nearer the
    /// shape's distance.
    fn evaluate(&self, t: f64, flags: u8) -> bool {
        if flags & ARTIFACT != 0 {
            return true;
        }
        let Some((reference, center, channels, step)) = self.exact.filter(|_| flags & CANDIDATE != 0) else {
            return false;
        };
        let offset = step * t;
        let at = center + offset;
        let old = bilinear(reference.field, reference.width, reference.height, at);
        let weight = (1.0 - offset.x.abs()) * (1.0 - offset.y.abs());
        let m = median_of(&channels);
        let new = [0, 1, 2].map(|i| old[i] + weight * (m - channels[i]));
        let exact = reference.distance(at);
        MIN_IMPROVE_RATIO * (median_of(&new) - exact).abs() < (median_of(&old) - exact).abs()
    }
}

/// Whether interpolating texel `a` toward its neighbour `b` makes an
/// artifact where two channels cross, which is where medians peak.
fn has_linear_artifact(classifier: &Classifier, am: f64, a: &Rgb, b: &Rgb) -> bool {
    let bm = median_of(b);
    // Of the pair, only the texel farther from the edge is reported.
    am.abs() >= bm.abs()
        && [(0, 1), (1, 2), (2, 0)].into_iter().any(|(i, j)| {
            let (da, db) = (a[j] - a[i], b[j] - b[i]);
            let t = da / (da - db);
            t > ARTIFACT_T_EPSILON && t < 1.0 - ARTIFACT_T_EPSILON && {
                let xm = interpolated_median(a, b, t);
                classifier.evaluate(t, classifier.range_test(0.0, 1.0, t, am, bm, xm))
            }
        })
}

/// Whether interpolating texel `a` toward `d`, diagonally across from it
/// with `b` and `c` the other two corners, makes an artifact.
fn has_diagonal_artifact(classifier: &Classifier, am: f64, a: &Rgb, b: &Rgb, c: &Rgb, d: &Rgb) -> bool {
    let dm = median_of(d);
    if am.abs() < dm.abs() {
        return false;
    }
    let abc = [0, 1, 2].map(|i| a[i] - b[i] - c[i]);
    let l = [0, 1, 2].map(|i| -a[i] - abc[i]);
    let q = [0, 1, 2].map(|i| d[i] + abc[i]);
    // Where each channel peaks along the diagonal.
    let peak = [0, 1, 2].map(|i| -0.5 * l[i] / q[i]);
    [(0, 1), (1, 2), (2, 0)].into_iter().any(|(i, j)| {
        let (da, dbc, dd) = (a[j] - a[i], b[j] - b[i] + c[j] - c[i], d[j] - d[i]);
        let (roots, count) = solve_quadratic(dd - dbc + da, dbc - da - da, da);
        roots[..count].iter().any(|&t| {
            if t <= ARTIFACT_T_EPSILON || t >= 1.0 - ARTIFACT_T_EPSILON {
                return false;
            }
            let xm = diagonal_median(a, &l, &q, t);
            let mut flags = classifier.range_test(0.0, 1.0, t, am, dm, xm);
            // Also hold it to the medians where either channel peaks.
            for extreme in [peak[i], peak[j]] {
                if extreme > 0.0 && extreme < 1.0 {
                    let (mut ends, mut medians) = ([0.0, 1.0], [am, dm]);
                    let side = usize::from(extreme > t);
                    ends[side] = extreme;
                    medians[side] = diagonal_median(a, &l, &q, extreme);
                    flags |= classifier.range_test(ends[0], ends[1], t, medians[0], medians[1], xm);
                }
            }
            classifier.evaluate(t, flags)
        })
    })
}

/// Protects the four texels around each corner, where the edge colour
/// changes: there channels disagree by design.
fn protect_corners(stencil: &mut [u8], width: usize, height: usize, prepared: &Prepared) {
    for contour in &prepared.contours {
        let Some(mut previous) = contour.last() else { continue };
        for edge in contour {
            let common = previous.color & edge.color;
            if common & common.wrapping_sub(1) == 0 {
                let p = edge.segment.start();
                let (left, top) = ((p.x - 0.5).floor(), (p.y - 0.5).floor());
                for (x, y) in [(left, top), (left + 1.0, top), (left, top + 1.0), (left + 1.0, top + 1.0)] {
                    if let Some(i) = texel_index(x, y, width, height) {
                        stencil[i] |= PROTECTED;
                    }
                }
            }
            previous = edge;
        }
    }
}

/// Whether `channel` draws the edge between texels `a` and `b`: it crosses
/// zero between them where it is the median.
fn edge_between_channel(a: &Rgb, b: &Rgb, channel: usize) -> bool {
    let t = a[channel] / (a[channel] - b[channel]);
    t > 0.0 && t < 1.0 && {
        let c = [0, 1, 2].map(|i| mix(a[i], b[i], t));
        median_of(&c) == c[channel]
    }
}

/// Protects texels `i` and `j`, a step `radius` apart with an edge between
/// them, where a channel drawing that edge is not their median.
fn protect_pair(stencil: &mut [u8], field: &[[f64; 4]], i: usize, j: usize, radius: f64) {
    let (a, b) = (rgb(&field[i]), rgb(&field[j]));
    let (am, bm) = (median_of(&a), median_of(&b));
    if am.abs() + bm.abs() >= radius {
        return;
    }
    let drawing = [RED, GREEN, BLUE];
    let mask = (0..3).filter(|&c| edge_between_channel(&a, &b, c)).fold(0, |mask, c| mask | drawing[c]);
    for (index, channels, m) in [(i, a, am), (j, b, bm)] {
        if (0..3).any(|c| mask & drawing[c] != 0 && channels[c] != m) {
            stencil[index] |= PROTECTED;
        }
    }
}

fn protect_edges(stencil: &mut [u8], field: &[[f64; 4]], width: usize, height: usize) {
    let radius = PROTECTION_RADIUS_TOLERANCE;
    for y in 0..height {
        for x in 0..width.saturating_sub(1) {
            let i = y * width + x;
            protect_pair(stencil, field, i, i + 1, radius);
        }
    }
    for y in 0..height.saturating_sub(1) {
        for x in 0..width {
            let i = y * width + x;
            protect_pair(stencil, field, i, i + width, radius);
        }
    }
    let radius = PROTECTION_RADIUS_TOLERANCE * SQRT_2;
    for y in 0..height.saturating_sub(1) {
        for x in 0..width.saturating_sub(1) {
            let i = y * width + x;
            protect_pair(stencil, field, i, i + width + 1, radius);
            protect_pair(stencil, field, i + 1, i + width, radius);
        }
    }
}

/// Whether texel (x, y), of median `cm`, could make an artifact with any of
/// its neighbours, a quick test passing most texels over. Medians peak only
/// where two channels cross, and channels interpolated between texels whose
/// order of the two agrees (or ties) never cross; and a protected texel's
/// artifacts cross the edge, which channels all on its median's side never do.
fn may_have_artifact(
    field: &[[f64; 4]],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    cm: f64,
    protected: bool,
) -> bool {
    let (mut above, mut below) = ([false; 3], [false; 3]);
    let mut one_sided = true;
    for row in y.saturating_sub(1)..(y + 2).min(height) {
        for column in x.saturating_sub(1)..(x + 2).min(width) {
            let texel = &field[row * width + column];
            for (pair, (i, j)) in [(0, 1), (1, 2), (2, 0)].into_iter().enumerate() {
                above[pair] |= texel[j] > texel[i];
                below[pair] |= texel[j] < texel[i];
            }
            one_sided &= texel[..3].iter().all(|&channel| if cm > 0.0 { channel > 0.0 } else { channel < 0.0 });
        }
    }
    (0..3).any(|pair| above[pair] && below[pair]) && !(protected && one_sided)
}

/// Marks texels whose interpolation with any of their eight neighbours
/// makes an artifact. With `reference`, texels already marked are passed
/// over and candidates are checked against the shape's distance.
fn find_errors(stencil: &mut [u8], field: &[[f64; 4]], width: usize, height: usize, reference: Option<&Reference>) {
    let (span, diagonal_span) = (MIN_DEVIATION_RATIO, MIN_DEVIATION_RATIO * SQRT_2);
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            if reference.is_some() && stencil[i] & ERROR != 0 {
                continue;
            }
            let c = rgb(&field[i]);
            let cm = median_of(&c);
            let protected = stencil[i] & PROTECTED != 0;
            if !may_have_artifact(field, width, height, x, y, cm, protected) {
                continue;
            }
            let center = Point::new(x as f64 + 0.5, y as f64 + 0.5);
            let neighbour = |dx: isize, dy: isize| {
                let (nx, ny) = (x.checked_add_signed(dx)?, y.checked_add_signed(dy)?);
                (nx < width && ny < height).then(|| rgb(&field[ny * width + nx]))
            };
            let classifier = |dx: isize, dy: isize, span: f64| Classifier {
                span,
                protected,
                exact: reference.map(|reference| (reference, center, c, Point::new(dx as f64, dy as f64))),
            };
            let linear = |dx: isize, dy: isize| {
                neighbour(dx, dy).is_some_and(|n| has_linear_artifact(&classifier(dx, dy, span), cm, &c, &n))
            };
            let diagonal = |dx: isize, dy: isize| match (neighbour(dx, 0), neighbour(0, dy), neighbour(dx, dy)) {
                (Some(b), Some(side), Some(d)) => {
                    has_diagonal_artifact(&classifier(dx, dy, diagonal_span), cm, &c, &b, &side, &d)
                }
                _ => false,
            };
            if linear(-1, 0)
                || linear(0, -1)
                || linear(1, 0)
                || linear(0, 1)
                || diagonal(-1, -1)
                || diagonal(1, -1)
                || diagonal(-1, 1)
                || diagonal(1, 1)
            {
                stencil[i] |= ERROR;
            }
        }
    }
}

/// Finds the field's artifacts and sets each one's channels to their median.
pub(super) fn correct(field: &mut [[f64; 4]], width: usize, height: usize, prepared: &Prepared) {
    let mut stencil = vec![0u8; field.len()];
    protect_corners(&mut stencil, width, height, prepared);
    protect_edges(&mut stencil, field, width, height);
    find_errors(&mut stencil, field, width, height, None);
    // Then any texel, now protected, only for medians crossing the edge
    // and where the shape's distance says correcting helps.
    for flags in &mut stencil {
        *flags |= PROTECTED;
    }
    let reference = Reference { prepared, field, width, height, scratch: RefCell::default() };
    find_errors(&mut stencil, field, width, height, Some(&reference));
    for (texel, flags) in field.iter_mut().zip(stencil) {
        if flags & ERROR != 0 {
            let m = median(texel[0], texel[1], texel[2]);
            texel[..3].fill(m);
        }
    }
}
