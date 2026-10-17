// Tests of the MSDF generator on shapes drawn here, judged against an
// independent rasterisation: each field is magnified eight times with
// bilinear filtering, as a GPU samples it, thresholded at 0.5, and compared
// with the non-zero fill of the shape's outline cut into short lines. The
// median of the three channels should match it everywhere but within an
// output pixel of the edge; the alpha channel, a one-channel field, rounds
// sharp corners, which is what the median should not.
use super::field::median;
use super::*;

/// Output pixels per field texel when magnifying.
const FACTOR: usize = 8;
/// How near a corner (in texels) an output pixel is judged as at it.
const CORNER_RADIUS: f64 = 1.5;
/// gdev's UI font, if it is beside this checkout; pen's repository has none.
const FREDOKA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../gdev/models/font/Fredoka.ttf");

fn p(x: f64, y: f64) -> Point {
    Point::new(x, y)
}

fn size(value: usize) -> u32 {
    u32::try_from(value).unwrap()
}

fn polygon(points: &[(f64, f64)]) -> Vec<Segment> {
    (0..points.len())
        .map(|i| {
            let (a, b) = (points[i], points[(i + 1) % points.len()]);
            Segment::Line(p(a.0, a.1), p(b.0, b.1))
        })
        .collect()
}

fn reverse(contour: &[Segment]) -> Vec<Segment> {
    contour.iter().rev().map(|segment| segment.reversed()).collect()
}

fn reversed(shape: &Shape) -> Shape {
    Shape { contours: shape.contours.iter().map(|contour| reverse(contour)).collect() }
}

/// `contour` turned a quarter round `center`, which keeps its direction.
fn quarter_turn(contour: &[Segment], center: Point) -> Vec<Segment> {
    let turn = |q: Point| p(center.x - (q.y - center.y), center.y + (q.x - center.x));
    contour
        .iter()
        .map(|segment| match *segment {
            Segment::Line(a, b) => Segment::Line(turn(a), turn(b)),
            Segment::Quad(a, b, c) => Segment::Quad(turn(a), turn(b), turn(c)),
            Segment::Cubic(a, b, c, d) => Segment::Cubic(turn(a), turn(b), turn(c), turn(d)),
        })
        .collect()
}

/// A square of half-side `half` about `center`, turned by `angle` radians.
fn turned_square(center: Point, half: f64, angle: f64) -> Vec<Segment> {
    let (sin, cos) = angle.sin_cos();
    let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        .map(|(x, y)| (center.x + half * (x * cos - y * sin), center.y + half * (x * sin + y * cos)));
    polygon(&corners)
}

// A quarter circle's cubic control points sit this far along its tangents,
// per unit of radius.
const KAPPA: f64 = 0.552_284_75;

/// A circle of four cubics.
fn circle(center: Point, radius: f64) -> Vec<Segment> {
    let k = KAPPA * radius;
    let at = |x: f64, y: f64| p(center.x + x, center.y + y);
    vec![
        Segment::Cubic(at(radius, 0.0), at(radius, k), at(k, radius), at(0.0, radius)),
        Segment::Cubic(at(0.0, radius), at(-k, radius), at(-radius, k), at(-radius, 0.0)),
        Segment::Cubic(at(-radius, 0.0), at(-radius, -k), at(-k, -radius), at(0.0, -radius)),
        Segment::Cubic(at(0.0, -radius), at(k, -radius), at(radius, -k), at(radius, 0.0)),
    ]
}

/// A level bar with round ends, its middle from `left` to `right` at `y`:
/// a contour with no corners, so of one colour, as Fredoka's strokes are.
fn stadium(left: f64, right: f64, y: f64, radius: f64) -> Vec<Segment> {
    let k = KAPPA * radius;
    vec![
        Segment::Line(p(left, y - radius), p(right, y - radius)),
        Segment::Cubic(p(right, y - radius), p(right + k, y - radius), p(right + radius, y - k), p(right + radius, y)),
        Segment::Cubic(p(right + radius, y), p(right + radius, y + k), p(right + k, y + radius), p(right, y + radius)),
        Segment::Line(p(right, y + radius), p(left, y + radius)),
        Segment::Cubic(p(left, y + radius), p(left - k, y + radius), p(left - radius, y + k), p(left - radius, y)),
        Segment::Cubic(p(left - radius, y), p(left - radius, y - k), p(left - k, y - radius), p(left, y - radius)),
    ]
}

/// The test shapes: a name, the shape, its cell's size, and the field's
/// range.
fn shapes() -> Vec<(&'static str, Shape, usize, usize, f32)> {
    let square = Shape { contours: vec![turned_square(p(20.0, 20.0), 11.0, 0.3)] };
    let triangle = Shape { contours: vec![polygon(&[(6.3, 33.6), (34.2, 28.9), (15.7, 6.4)])] };
    // An L with a square hole, the hole running the other way round.
    let notched = Shape {
        contours: vec![
            polygon(&[(6.2, 6.3), (33.8, 6.3), (33.8, 18.4), (20.3, 18.4), (20.3, 33.7), (6.2, 33.7)]),
            reverse(&polygon(&[(10.6, 10.2), (16.4, 10.2), (16.4, 14.1), (10.6, 14.1)])),
        ],
    };
    // A D, its bowl a cubic, with a lens of two quadratics cut out of it.
    let curves = Shape {
        contours: vec![
            vec![
                Segment::Line(p(7.3, 5.7), p(7.3, 34.2)),
                Segment::Cubic(p(7.3, 34.2), p(36.0, 34.2), p(36.0, 5.7), p(7.3, 5.7)),
            ],
            reverse(&[
                Segment::Quad(p(13.2, 12.1), p(25.0, 20.0), p(13.2, 27.8)),
                Segment::Quad(p(13.2, 27.8), p(17.0, 20.0), p(13.2, 12.1)),
            ]),
        ],
    };
    // A circle (no corners) and a teardrop (one corner, one edge).
    let drops = Shape {
        contours: vec![
            circle(p(14.0, 16.0), 9.0),
            vec![Segment::Cubic(p(30.0, 16.0), p(52.0, -2.0), p(52.0, 34.0), p(30.0, 16.0))],
        ],
    };
    // Two bars crossing: a plus sign by the non-zero rule.
    let cross = Shape {
        contours: vec![
            polygon(&[(6.0, 15.0), (34.0, 15.0), (34.0, 25.0), (6.0, 25.0)]),
            polygon(&[(15.0, 6.0), (25.0, 6.0), (25.0, 34.0), (15.0, 34.0)]),
        ],
    };
    // Two round-ended bars crossing, as Fredoka draws its +: overlapping
    // contours of one colour each.
    let bar = stadium(12.0, 28.0, 20.0, 4.6);
    let rounded_cross = Shape { contours: vec![quarter_turn(&bar, p(20.0, 20.0)), bar] };
    // A square holding another running the same way: filled, no hole.
    let filled = Shape {
        contours: vec![
            polygon(&[(6.0, 6.0), (34.0, 6.0), (34.0, 34.0), (6.0, 34.0)]),
            polygon(&[(13.0, 13.0), (27.0, 13.0), (27.0, 27.0), (13.0, 27.0)]),
        ],
    };
    vec![
        ("square", square, 40, 40, 4.0),
        ("triangle", triangle, 40, 40, 4.0),
        ("notched", notched, 40, 40, 4.0),
        ("curves", curves, 40, 40, 6.0),
        ("drops", drops, 56, 32, 4.0),
        ("cross", cross, 40, 40, 4.0),
        ("rounded cross", rounded_cross, 40, 40, 4.0),
        ("filled", filled, 40, 40, 4.0),
    ]
}

/// Each contour as a closed polyline, curves cut into 64 lines.
fn flatten(shape: &Shape) -> Vec<Vec<Point>> {
    shape
        .contours
        .iter()
        .map(|contour| {
            contour
                .iter()
                .flat_map(|segment| {
                    let steps = if matches!(segment, Segment::Line(..)) { 1 } else { 64 };
                    (0..steps).map(move |i| segment.point(f64::from(i) / f64::from(steps)))
                })
                .collect()
        })
        .collect()
}

fn lines(polylines: &[Vec<Point>]) -> impl Iterator<Item = (Point, Point)> + '_ {
    polylines.iter().flat_map(|line| (0..line.len()).map(move |i| (line[i], line[(i + 1) % line.len()])))
}

/// Where the horizontal line at `y` crosses `polylines`, and which way.
fn crossings(polylines: &[Vec<Point>], y: f64) -> Vec<(f64, i32)> {
    let mut crossings: Vec<(f64, i32)> = lines(polylines)
        .filter(|(a, b)| (a.y <= y) != (b.y <= y))
        .map(|(a, b)| (a.x + (y - a.y) / (b.y - a.y) * (b.x - a.x), if b.y > a.y { 1 } else { -1 }))
        .collect();
    crossings.sort_by(|a, b| a.0.total_cmp(&b.0));
    crossings
}

fn filled_at(crossings: &[(f64, i32)], x: f64) -> bool {
    crossings.iter().take_while(|crossing| crossing.0 < x).map(|crossing| crossing.1).sum::<i32>() != 0
}

/// The non-zero fill of `polylines` at each output pixel of a `width` x
/// `height` cell magnified `FACTOR` times.
fn rasterise(polylines: &[Vec<Point>], width: usize, height: usize) -> Vec<bool> {
    let (columns, rows) = (width * FACTOR, height * FACTOR);
    let mut inside = Vec::with_capacity(columns * rows);
    for row in 0..rows {
        let crossings = crossings(polylines, (row as f64 + 0.5) / FACTOR as f64);
        inside.extend((0..columns).map(|column| filled_at(&crossings, (column as f64 + 0.5) / FACTOR as f64)));
    }
    inside
}

fn edge_distance(polylines: &[Vec<Point>], q: Point) -> f64 {
    lines(polylines)
        .map(|(a, b)| {
            let ab = b - a;
            let t = ((q - a).dot(ab) / ab.dot(ab)).clamp(0.0, 1.0);
            (a + ab * t - q).length()
        })
        .fold(f64::INFINITY, f64::min)
}

/// The cell's channels at `q` (in pixels), 0 to 1, filtered bilinearly and
/// clamped at the cell's edges.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped whole texel coordinates")]
fn sample(rgba: &[u8], width: usize, height: usize, q: Point) -> [f64; 4] {
    let (x, y) = (q.x - 0.5, q.y - 0.5);
    let (left, top) = (x.floor(), y.floor());
    let (tx, ty) = (x - left, y - top);
    let column = |c: f64| c.clamp(0.0, (width - 1) as f64) as usize;
    let row = |r: f64| r.clamp(0.0, (height - 1) as f64) as usize;
    let at = |c: usize, r: usize, channel: usize| f64::from(rgba[(r * width + c) * 4 + channel]) / 255.0;
    let (l, r, t, b) = (column(left), column(left + 1.0), row(top), row(top + 1.0));
    [0, 1, 2, 3].map(|i| {
        let upper = at(l, t, i) + (at(r, t, i) - at(l, t, i)) * tx;
        let lower = at(l, b, i) + (at(r, b, i) - at(l, b, i)) * tx;
        upper + (lower - upper) * ty
    })
}

/// The joints of `shape` where its direction turns by more than 20 degrees.
fn corners(shape: &Shape) -> Vec<Point> {
    let mut corners = Vec::new();
    for contour in &shape.contours {
        for (i, segment) in contour.iter().enumerate() {
            let previous = contour[(i + contour.len() - 1) % contour.len()];
            if previous.direction(1.0).normalize().dot(segment.direction(0.0).normalize()) < 0.94 {
                corners.push(segment.start());
            }
        }
    }
    corners
}

/// How a field magnified `FACTOR` times differs from the shape: output pixels
/// whose median (MSDF) or alpha (one channel) puts them on the wrong side,
/// more than an output pixel from the edge, and within `CORNER_RADIUS` of a
/// sharp corner.
#[derive(Debug, Default)]
struct Mismatch {
    samples: usize,
    msdf_far: usize,
    alpha_far: usize,
    msdf_corners: usize,
    alpha_corners: usize,
}

fn compare(shape: &Shape, rgba: &[u8], width: usize, height: usize) -> Mismatch {
    let polylines = flatten(shape);
    let truth = rasterise(&polylines, width, height);
    let corners = corners(shape);
    let mut mismatch = Mismatch { samples: truth.len(), ..Mismatch::default() };
    for (i, &inside) in truth.iter().enumerate() {
        let (column, row) = (i % (width * FACTOR), i / (width * FACTOR));
        let q = p((column as f64 + 0.5) / FACTOR as f64, (row as f64 + 0.5) / FACTOR as f64);
        let texel = sample(rgba, width, height, q);
        let msdf = median(texel[0], texel[1], texel[2]) > 0.5;
        let alpha = texel[3] > 0.5;
        if msdf == inside && alpha == inside {
            continue;
        }
        let far = edge_distance(&polylines, q) > 1.0 / FACTOR as f64;
        let cornered = corners.iter().any(|&corner| (corner - q).length() < CORNER_RADIUS);
        if msdf != inside {
            mismatch.msdf_far += usize::from(far);
            mismatch.msdf_corners += usize::from(cornered);
        }
        if alpha != inside {
            mismatch.alpha_far += usize::from(far);
            mismatch.alpha_corners += usize::from(cornered);
        }
    }
    mismatch
}

fn encode_field(field: &[[f64; 4]], range: f64) -> Vec<u8> {
    field.iter().flat_map(|texel| texel.map(|distance| encode(0.5 + distance / (2.0 * range)))).collect()
}

#[test]
fn fields_redraw_their_shapes_and_keep_corners_sharp() {
    let (mut msdf_corners, mut alpha_corners) = (0, 0);
    for (name, shape, width, height, range) in shapes() {
        let field = generate(&shape, size(width), size(height), range);
        // Contours running the other way round make the same field.
        assert!(generate(&reversed(&shape), size(width), size(height), range) == field, "{name} reversed differs");
        let mismatch = compare(&shape, &field, width, height);
        assert!(mismatch.msdf_far * 10_000 <= mismatch.samples, "{name}: {mismatch:?}");
        assert!(mismatch.msdf_corners * 4 <= mismatch.alpha_corners, "{name}: {mismatch:?}");
        msdf_corners += mismatch.msdf_corners;
        alpha_corners += mismatch.alpha_corners;
    }
    // At sharp corners the median is right where one channel rounds them.
    assert!(alpha_corners >= 200 && msdf_corners * 20 <= alpha_corners, "{msdf_corners} vs {alpha_corners}");
}

#[test]
fn alpha_holds_the_true_distance() {
    for (name, shape, width, height, range) in shapes() {
        // Polygons, whose outlines are exact, that do not overlap themselves.
        if !["square", "triangle", "notched"].contains(&name) {
            continue;
        }
        let field = generate(&shape, size(width), size(height), range);
        let range = f64::from(range);
        let polylines = flatten(&shape);
        for y in 0..height {
            let crossings = crossings(&polylines, y as f64 + 0.5);
            for x in 0..width {
                let q = p(x as f64 + 0.5, y as f64 + 0.5);
                let distance = edge_distance(&polylines, q);
                if distance >= range {
                    continue;
                }
                let expected = if filled_at(&crossings, q.x) { distance } else { -distance };
                let alpha = (f64::from(field[(y * width + x) * 4 + 3]) / 255.0 - 0.5) * 2.0 * range;
                // Within half of one of the 255 steps across twice the range.
                assert!((alpha - expected).abs() <= range / 255.0 + 1e-9, "{name} at {q:?}: {alpha} vs {expected}");
            }
        }
    }
}

#[test]
fn error_correction_removes_clashes() {
    // A star whose points are a texel or two across: its uncorrected field's
    // channels, interpolated, put the median over the edge away from it.
    let star = [
        (15.6, 12.4),
        (15.8, 15.6),
        (11.9, 13.4),
        (8.0, 16.6),
        (9.4, 12.1),
        (7.5, 9.5),
        (11.7, 10.6),
        (14.2, 8.0),
        (13.4, 11.2),
    ];
    let shape = Shape { contours: vec![polygon(&star)] };
    let raw = distances(&shape, 24, 24, false);
    let corrected = distances(&shape, 24, 24, true);
    let before = compare(&shape, &encode_field(&raw, 3.0), 24, 24);
    let after = compare(&shape, &encode_field(&corrected, 3.0), 24, 24);
    assert!(before.msdf_far > 0, "{before:?}");
    assert_eq!(after.msdf_far, 0, "{after:?}");
    // Correcting a texel sets its channels to their median, so every
    // texel's median, and its true distance, stays as it was.
    assert!(raw != corrected);
    for (raw, corrected) in raw.iter().zip(&corrected) {
        assert_eq!(median(raw[0], raw[1], raw[2]), median(corrected[0], corrected[1], corrected[2]));
        assert_eq!(raw[3], corrected[3]);
    }
}

#[test]
fn an_empty_shape_is_all_outside() {
    assert_eq!(generate(&Shape::default(), 3, 2, 4.0), vec![0; 3 * 2 * 4]);
    let point = Shape { contours: vec![vec![Segment::Line(p(1.0, 1.0), p(1.0, 1.0))]] };
    assert_eq!(generate(&point, 3, 2, 4.0), vec![0; 3 * 2 * 4]);
}

#[test]
fn packs_tallest_first_in_shelves() {
    let (places, height) = pack(&[(10, 5), (8, 9), (7, 9), (6, 3)], 20).unwrap();
    // The two 9-high cells share the first shelf; the 10-wide one starts the
    // next, 9 down, with the 6-wide one beside it; 14 rows round up to 16.
    assert_eq!(places, vec![(0, 9), (0, 0), (8, 0), (10, 9)]);
    assert_eq!(height, 16);
    assert_eq!(pack(&[], 20).unwrap(), (vec![], 1));
    assert!(pack(&[(21, 4)], 20).is_err());
}

#[test]
fn work_in_parallel_comes_back_in_order() {
    let items: Vec<u64> = (0..97).collect();
    let squares = in_parallel(&items, |&item| item * item);
    assert_eq!(squares, items.iter().map(|item| item * item).collect::<Vec<_>>());
    assert!(in_parallel(&[] as &[u64], |&item| item).is_empty());
}

#[test]
fn refuses_what_is_not_a_font() {
    let options = Options { size: 64.0, range: 6.0, width: 1024, variations: Vec::new(), chars: vec!['A'] };
    assert!(atlas(b"not a font", &options).is_err());
}

#[test]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "whole texel coordinates in the atlas")]
fn fredoka_matches_gdev_and_redraws_its_glyphs() {
    let Ok(data) = std::fs::read(FREDOKA) else {
        eprintln!("skipped: no font at {FREDOKA}");
        return;
    };
    // Advances from gdev's src/font.blot, made by models/font/font.py from
    // the same font at weight 600.
    let advances = [
        (' ', 0.24097),
        ('!', 0.24194),
        ('A', 0.70898),
        ('M', 0.81396),
        ('a', 0.56299),
        ('g', 0.55200),
        ('~', 0.55493),
    ];
    let mut options = Options {
        size: 64.0,
        range: 6.0,
        width: 1024,
        variations: vec![(*b"wght", 600.0)],
        chars: advances.iter().map(|&(code, _)| code).chain(['\u{E000}', 'A']).collect(),
    };
    let atlas = atlas(&data, &options).unwrap();
    // font.py's ascent and descent are FreeType's, rounded out to whole
    // pixels at 64 per em (63 and 16); these are the font's own.
    assert!((atlas.ascent - 0.98438).abs() <= 1.0 / 64.0, "ascent {}", atlas.ascent);
    assert!((atlas.descent - 0.25).abs() <= 1.0 / 64.0, "descent {}", atlas.descent);
    assert!((atlas.spread - 0.09375).abs() < 1e-6);
    // The font lacks the private-use character, and 'A' comes once.
    assert_eq!(atlas.glyphs.len(), advances.len());
    for (glyph, &(code, advance)) in atlas.glyphs.iter().zip(&advances) {
        assert_eq!(glyph.code, code);
        assert!((glyph.advance - advance).abs() < 0.002, "{code:?} advance {} vs {advance}", glyph.advance);
    }
    let space = &atlas.glyphs[0];
    assert_eq!([space.u0, space.v0, space.u1, space.v1, space.width, space.height], [0.0; 6]);
    assert!(atlas.height.is_power_of_two() && atlas.pixels.len() == (atlas.width * atlas.height * 4) as usize);

    let font = font(&data, &options).unwrap();
    for code in ['A', 'M', 'g'] {
        let glyph = atlas.glyphs.iter().find(|glyph| glyph.code == code).unwrap();
        let cell = font.entries.iter().find(|entry| entry.code == code).and_then(|entry| entry.cell.as_ref()).unwrap();
        let (width, height) = (cell.width as usize, cell.height as usize);
        assert_eq!(((glyph.u1 - glyph.u0) * 1024.0).round() as usize, width);
        assert!(
            (glyph.width * 64.0 - cell.width as f32).abs() < 1e-3 && (glyph.top * 64.0 - cell.top as f32).abs() < 1e-3
        );
        let (x, y) = ((glyph.u0 * 1024.0).round() as usize, (glyph.v0 * atlas.height as f32).round() as usize);
        let cropped: Vec<u8> =
            (0..height).flat_map(|row| &atlas.pixels[((y + row) * 1024 + x) * 4..][..width * 4]).copied().collect();
        assert!(cropped == generate(&cell.shape, cell.width, cell.height, 6.0), "{code:?} is not where it says");
        let mismatch = compare(&cell.shape, &cropped, width, height);
        assert!(mismatch.msdf_far * 10_000 <= mismatch.samples, "{code:?}: {mismatch:?}");
        assert!(mismatch.msdf_corners <= mismatch.alpha_corners, "{code:?}: {mismatch:?}");
    }

    options.variations = vec![(*b"slnt", -10.0)];
    assert!(super::atlas(&data, &options).is_err(), "Fredoka has no slant axis");
}
