// Edge colouring for the MSDF generator, msdfgen's `edgeColoringSimple`
// with seed 0: a contour's corners split it into runs of edges, and each run
// takes two of the three channels (yellow, cyan or magenta), so the two runs
// meeting at a corner share exactly one. Away from the corner the three
// channels' median follows the nearer run; at it, each run's own channel
// keeps the run's edge running straight on past the corner, which is what
// keeps the corner sharp. A contour with no corners takes one colour, and
// one with a single corner is split into three runs, the middle one white.
use super::geometry::{Point, Segment};

pub(super) const RED: u8 = 1;
pub(super) const GREEN: u8 = 2;
pub(super) const BLUE: u8 = 4;
const CYAN: u8 = GREEN | BLUE;
pub(super) const WHITE: u8 = RED | GREEN | BLUE;

/// Where the direction turns by more than this (radians), the outline has a
/// corner; msdfgen's default.
const ANGLE_THRESHOLD: f64 = 3.0;

/// A segment of a contour and the channels whose fields it shapes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Edge {
    pub segment: Segment,
    pub color: u8,
}

impl Edge {
    pub fn white(segment: Segment) -> Self {
        Self { segment, color: WHITE }
    }
}

fn is_corner(a: Point, b: Point, cross_threshold: f64) -> bool {
    a.dot(b) <= 0.0 || a.cross(b).abs() > cross_threshold
}

/// The next of cyan, magenta and yellow after `color`.
fn switch_color(color: u8) -> u8 {
    let shifted = color << 1;
    (shifted | shifted >> 3) & WHITE
}

/// The next colour after `color` that also differs from `banned`.
fn switch_color_avoiding(color: u8, banned: u8) -> u8 {
    let common = color & banned;
    if common == RED || common == GREEN || common == BLUE { common ^ WHITE } else { switch_color(color) }
}

/// Which third of a contour of `count` edges the one `position` edges past
/// its corner falls in: 0 the first, 1 the middle, 2 the last (msdfgen's
/// `symmetricalTrichotomy`, shifted to count from 0).
fn third(position: usize, count: usize) -> usize {
    let at = 3.0 + 2.875 * position as f64 / (count - 1) as f64 - 1.4375 + 0.5;
    if at < 3.0 {
        0
    } else if at < 4.0 {
        1
    } else {
        2
    }
}

/// Colours every contour's edges, splitting the edges of a contour with one
/// corner and fewer than three edges into thirds.
pub(super) fn color_edges(contours: &mut [Vec<Edge>]) {
    let cross_threshold = ANGLE_THRESHOLD.sin();
    let mut color = CYAN;
    for contour in contours.iter_mut() {
        let Some(last) = contour.last() else { continue };
        let mut corners = Vec::new();
        let mut previous = last.segment.direction(1.0);
        for (index, edge) in contour.iter().enumerate() {
            if is_corner(previous.normalize(), edge.segment.direction(0.0).normalize(), cross_threshold) {
                corners.push(index);
            }
            previous = edge.segment.direction(1.0);
        }
        let count = contour.len();
        match corners[..] {
            [] => {
                color = switch_color(color);
                for edge in contour.iter_mut() {
                    edge.color = color;
                }
            }
            // A teardrop: three runs from the one corner round to it again.
            [corner] => {
                color = switch_color(color);
                let first = color;
                color = switch_color(color);
                let colors = [first, WHITE, color];
                if count >= 3 {
                    for i in 0..count {
                        contour[(corner + i) % count].color = colors[third(i, count)];
                    }
                } else {
                    let parts: Vec<Segment> =
                        (0..count).flat_map(|i| contour[(corner + i) % count].segment.split_in_thirds()).collect();
                    let total = parts.len();
                    *contour = parts
                        .into_iter()
                        .enumerate()
                        .map(|(i, segment)| Edge { segment, color: colors[i * 3 / total] })
                        .collect();
                }
            }
            _ => {
                let start = corners[0];
                let mut spline = 0;
                color = switch_color(color);
                let initial = color;
                for i in 0..count {
                    let index = (start + i) % count;
                    if spline + 1 < corners.len() && corners[spline + 1] == index {
                        spline += 1;
                        // The last run meets the first too, so differs from it.
                        let banned = if spline == corners.len() - 1 { initial } else { 0 };
                        color = switch_color_avoiding(color, banned);
                    }
                    contour[index].color = color;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Vec<Edge> {
        let p = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)].map(|(x, y)| Point::new(x, y));
        (0..4).map(|i| Edge::white(Segment::Line(p[i], p[(i + 1) % 4]))).collect()
    }

    #[test]
    fn edges_meeting_at_a_corner_share_one_channel() {
        let mut contours = vec![square(), square()[..3].to_vec()];
        // A triangle: close the three edges.
        contours[1][2] = Edge::white(Segment::Line(Point::new(4.0, 4.0), Point::new(0.0, 0.0)));
        color_edges(&mut contours);
        for contour in &contours {
            for (i, edge) in contour.iter().enumerate() {
                let next = contour[(i + 1) % contour.len()];
                assert_eq!((edge.color & next.color).count_ones(), 1, "{contour:?}");
                assert_eq!(edge.color.count_ones(), 2);
            }
        }
    }

    #[test]
    fn a_smooth_contour_is_one_colour_and_a_teardrop_three() {
        let (a, b) = (Point::new(0.0, 0.0), Point::new(8.0, 0.0));
        let mut contours = vec![
            vec![
                Edge::white(Segment::Cubic(a, Point::new(0.0, 4.0), Point::new(8.0, 4.0), b)),
                Edge::white(Segment::Cubic(b, Point::new(8.0, -4.0), Point::new(0.0, -4.0), a)),
            ],
            vec![
                Edge::white(Segment::Quad(a, Point::new(4.0, 6.0), b)),
                Edge::white(Segment::Quad(b, Point::new(4.0, -6.0), a)),
            ],
        ];
        // The first is a smooth loop (its joints run on straight); the second
        // a lens with two corners.
        color_edges(&mut contours);
        assert!(contours[0].iter().all(|edge| edge.color == contours[0][0].color));
        assert_ne!(contours[1][0].color, contours[1][1].color);

        let tip = Point::new(0.0, 0.0);
        let mut teardrop =
            vec![vec![Edge::white(Segment::Cubic(tip, Point::new(12.0, -8.0), Point::new(12.0, 8.0), tip))]];
        teardrop[0] = teardrop[0][0].segment.split_in_thirds().map(Edge::white).to_vec();
        color_edges(&mut teardrop);
        let colors: Vec<u8> = teardrop[0].iter().map(|edge| edge.color).collect();
        assert_eq!(colors[1], WHITE);
        assert_eq!((colors[0] & colors[2]).count_ones(), 1);
    }
}
