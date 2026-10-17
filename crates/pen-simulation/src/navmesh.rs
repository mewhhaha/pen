//! Paths over a navmesh: walkable areas less obstacles, triangulated
//! (spade's constrained Delaunay triangulation, with exact predicates), then
//! A* from triangle to triangle and the funnel algorithm pulling the way
//! taut round the corners. Unlike `pen_runtime::pathfinding`'s grid, the
//! way goes at any angle and the ground can have any shape.
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use pen_protocol::boundary::MAX_SERVICE_DATA;
use pen_protocol::{navmesh_area, navmesh_op, service};
use pen_runtime::services::{Answer, Request, Service};
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

use crate::{Lanes, counted};

/// Paths over a navmesh on the ground plane (x, z) as a service. The
/// request sends the areas each time and the navmesh is built for it, so,
/// as with the other services, nothing outlives the request.
pub struct Navmesh;

impl<Input> Service<Input> for Navmesh {
    fn code(&self) -> u32 {
        service::NAVMESH
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        let answer = match request.operation {
            navmesh_op::FIND => find(request),
            navmesh_op::TRIANGLES => triangles(request),
            _ => return Some(Answer::unknown()),
        };
        Some(answer.unwrap_or_else(Answer::failed))
    }
}

type Point = Point2<f64>;

/// The walkable triangles: each one's corners, counterclockwise, and
/// across each of its edges (from corner i to the next) the triangle next
/// to it, if that is walkable too.
struct Mesh {
    corners: Vec<[Point; 3]>,
    neighbours: Vec<[Option<usize>; 3]>,
}

/// An area's kind and its corners.
struct Area {
    walkable: bool,
    corners: Vec<Point>,
}

/// The agent's radius and the areas, as the request's values give them.
fn areas(values: &[f32]) -> Option<(f64, Vec<Area>)> {
    let mut lanes = Lanes::new(values);
    let radius = f64::from(lanes.number()?);
    let mut areas = Vec::new();
    while !lanes.done() {
        let walkable = lanes.count(navmesh_area::NAMES.len() - 1)? == navmesh_area::WALKABLE as usize;
        let count = lanes.count(MAX_SERVICE_DATA / 2)?;
        let corners: Vec<Point> = lanes
            .numbers(count * 2)?
            .chunks(2)
            .map(|corner| Point::new(f64::from(corner[0]), f64::from(corner[1])))
            .collect();
        if count < 3 {
            return None;
        }
        areas.push(Area { walkable, corners });
    }
    (radius >= 0.0 && areas.iter().any(|area| area.walkable)).then_some((radius, areas))
}

/// Whether `point` is inside the polygon (crossings, even-odd).
fn inside(point: Point, corners: &[Point]) -> bool {
    let mut within = false;
    let mut last = corners[corners.len() - 1];
    for &corner in corners {
        if (corner.y > point.y) != (last.y > point.y)
            && point.x < (last.x - corner.x) * (point.y - corner.y) / (last.y - corner.y) + corner.x
        {
            within = !within;
        }
        last = corner;
    }
    within
}

/// The navmesh of the areas: every area's outline is a constraint (split
/// where outlines cross), and a triangle is walkable when its centre lies in
/// some walkable area and in no obstacle.
fn mesh(areas: &[Area]) -> Option<Mesh> {
    let mut triangulation = ConstrainedDelaunayTriangulation::<Point>::new();
    for area in areas {
        let corners: Vec<_> =
            area.corners.iter().map(|corner| triangulation.insert(*corner).ok()).collect::<Option<_>>()?;
        for (index, from) in corners.iter().enumerate() {
            let to = corners[(index + 1) % corners.len()];
            if *from != to {
                triangulation.add_constraint_and_split(*from, to, |point| point);
            }
        }
    }
    let walkable = |center: Point| {
        areas.iter().any(|area| area.walkable && inside(center, &area.corners))
            && !areas.iter().any(|area| !area.walkable && inside(center, &area.corners))
    };
    // Faces by their index, so neighbours can be found by it.
    let mut place = vec![None; triangulation.all_faces().len()];
    let mut faces = Vec::new();
    for face in triangulation.inner_faces() {
        if walkable(face.center()) {
            place[face.fix().index()] = Some(faces.len());
            faces.push(face);
        }
    }
    let corners = faces.iter().map(|face| face.positions()).collect();
    let neighbours = faces
        .iter()
        .map(|face| {
            let [a, b, c] = face.vertices().map(|vertex| vertex.fix());
            // Across the edge from each corner to the next.
            [(a, b), (b, c), (c, a)].map(|(from, to)| {
                let edge = face
                    .adjacent_edges()
                    .into_iter()
                    .find(|edge| edge.from().fix() == from && edge.to().fix() == to)?;
                place[edge.rev().face().fix().index()]
            })
        })
        .collect();
    Some(Mesh { corners, neighbours })
}

fn distance(a: Point, b: Point) -> f64 {
    ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)).sqrt()
}

/// Twice the signed area of a, b, c: negative when c lies left of a to b.
fn area2(a: Point, b: Point, c: Point) -> f64 {
    (c.x - a.x) * (b.y - a.y) - (b.x - a.x) * (c.y - a.y)
}

/// The point of segment a to b nearest `point`.
fn nearest_on_segment(point: Point, a: Point, b: Point) -> Point {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let length = dx * dx + dy * dy;
    let t = if length > 0.0 { (((point.x - a.x) * dx + (point.y - a.y) * dy) / length).clamp(0.0, 1.0) } else { 0.0 };
    Point::new(a.x + dx * t, a.y + dy * t)
}

impl Mesh {
    /// The triangle `point` is in, or else the nearest point on the
    /// navmesh and its triangle; the first such triangle when several are.
    fn locate(&self, point: Point) -> Option<(usize, Point)> {
        let mut nearest: Option<(f64, usize, Point)> = None;
        for (index, [a, b, c]) in self.corners.iter().copied().enumerate() {
            if area2(a, b, point) <= 0.0 && area2(b, c, point) <= 0.0 && area2(c, a, point) <= 0.0 {
                return Some((index, point));
            }
            for (from, to) in [(a, b), (b, c), (c, a)] {
                let on = nearest_on_segment(point, from, to);
                let apart = distance(on, point);
                if nearest.is_none_or(|(best, _, _)| apart < best) {
                    nearest = Some((apart, index, on));
                }
            }
        }
        nearest.map(|(_, index, on)| (index, on))
    }

    /// The edge's portal as an agent of `radius` passes it, travelling out
    /// of triangle `from` across its `edge`: its left and right ends, each
    /// pulled the radius in; None when it is narrower than the agent.
    fn portal(&self, from: usize, edge: usize, radius: f64) -> Option<(Point, Point)> {
        let corners = self.corners[from];
        // Counterclockwise, the triangle lies left of its edge, so crossing
        // it the edge's end is on the left and its start on the right.
        let (right, left) = (corners[edge], corners[(edge + 1) % 3]);
        let width = distance(left, right);
        if width < 2.0 * radius {
            return None;
        }
        let pull = if width > 0.0 { radius / width } else { 0.0 };
        let (dx, dy) = (right.x - left.x, right.y - left.y);
        Some((Point::new(left.x + dx * pull, left.y + dy * pull), Point::new(right.x - dx * pull, right.y - dy * pull)))
    }

    /// The triangles from `start` to `goal` and the edges crossed, by A*
    /// over the portals' middles: by estimated length, then nearness to the
    /// goal, then triangle, the same order on every machine.
    fn corridor(&self, start: (usize, Point), goal: (usize, Point), radius: f64) -> Option<Vec<(usize, usize)>> {
        let count = self.corners.len();
        let mut cost = vec![f64::INFINITY; count];
        let mut entry = vec![start.1; count];
        let mut came: Vec<Option<(usize, usize)>> = vec![None; count];
        let mut closed = vec![false; count];
        let mut frontier = BinaryHeap::new();
        let key = |value: f64| value.to_bits(); // Non-negative and finite: ordered as their bits.
        cost[start.0] = 0.0;
        frontier.push(Reverse((key(distance(start.1, goal.1)), key(distance(start.1, goal.1)), start.0)));
        while let Some(Reverse((_, _, triangle))) = frontier.pop() {
            if triangle == goal.0 {
                let mut crossed = Vec::new();
                let mut at = triangle;
                while let Some((from, edge)) = came[at] {
                    crossed.push((from, edge));
                    at = from;
                }
                crossed.reverse();
                return Some(crossed);
            }
            if std::mem::replace(&mut closed[triangle], true) {
                continue;
            }
            for (edge, next) in self.neighbours[triangle].iter().enumerate() {
                let Some(next) = *next else { continue };
                let Some((left, right)) = self.portal(triangle, edge, radius) else { continue };
                let middle = Point::new((left.x + right.x) / 2.0, (left.y + right.y) / 2.0);
                let through = cost[triangle] + distance(entry[triangle], middle);
                if !closed[next] && through < cost[next] {
                    cost[next] = through;
                    entry[next] = middle;
                    came[next] = Some((triangle, edge));
                    let ahead = distance(middle, goal.1);
                    frontier.push(Reverse((key(through + ahead), key(ahead), next)));
                }
            }
        }
        None
    }
}

/// The way through the portals from `start` to `goal`, pulled taut (the
/// simple stupid funnel algorithm): the funnel's sides narrow portal by
/// portal, and where one crosses the other its corner is a turn.
fn funnel(start: Point, goal: Point, portals: &[(Point, Point)]) -> Vec<Point> {
    let mut all = vec![(start, start)];
    all.extend_from_slice(portals);
    all.push((goal, goal));
    let same = |a: Point, b: Point| distance(a, b) < 1e-9;
    let mut path = vec![start];
    let (mut apex, mut left, mut right) = (start, start, start);
    let (mut left_at, mut right_at) = (0, 0);
    let mut index = 1;
    while index < all.len() {
        let (next_left, next_right) = all[index];
        if area2(apex, right, next_right) <= 0.0 {
            if same(apex, right) || area2(apex, left, next_right) > 0.0 {
                right = next_right;
                right_at = index;
            } else {
                // The right side crossed the left: turn at the left corner,
                // and look on from the portal it came with.
                path.push(left);
                (apex, right, right_at) = (left, left, left_at);
                index = left_at + 1;
                continue;
            }
        }
        if area2(apex, left, next_left) >= 0.0 {
            if same(apex, left) || area2(apex, right, next_left) < 0.0 {
                left = next_left;
                left_at = index;
            } else {
                path.push(right);
                (apex, left, left_at) = (right, right, right_at);
                index = right_at + 1;
                continue;
            }
        }
        index += 1;
    }
    path.push(goal);
    path.dedup_by(|a, b| same(*a, *b));
    path
}

#[allow(clippy::cast_possible_truncation, reason = "the navmesh's f64 points narrowed to f32 lanes")]
fn lanes(points: &[Point]) -> Vec<f32> {
    points.iter().flat_map(|point| [point.x as f32, point.y as f32]).collect()
}

fn find(request: &Request) -> Option<Answer> {
    let (radius, areas) = areas(request.values)?;
    let mesh = mesh(&areas)?;
    let [x, z, goal_x, goal_z] = request.numbers.map(f64::from);
    if !request.numbers.iter().all(|number| number.is_finite()) {
        return None;
    }
    let (Some(start), Some(goal)) = (mesh.locate(Point::new(x, z)), mesh.locate(Point::new(goal_x, goal_z))) else {
        return Some(Answer::ok(0.0));
    };
    let Some(corridor) = mesh.corridor(start, goal, radius) else { return Some(Answer::ok(0.0)) };
    let portals: Vec<_> = corridor.iter().filter_map(|&(from, edge)| mesh.portal(from, edge, radius)).collect();
    let path = funnel(start.1, goal.1, &portals);
    Some(counted(path.len(), lanes(&path)))
}

fn triangles(request: &Request) -> Option<Answer> {
    let (_, areas) = areas(request.values)?;
    let mesh = mesh(&areas)?;
    let corners: Vec<Point> = mesh.corners.iter().flatten().copied().collect();
    Some(counted(mesh.corners.len(), lanes(&corners)))
}

#[cfg(test)]
mod tests {
    use pen_protocol::service_status;

    use super::*;

    /// An area's lanes: its kind, its corner count and corners.
    fn area(kind: u32, corners: &[[f32; 2]]) -> Vec<f32> {
        let mut lanes = vec![kind as f32, corners.len() as f32];
        lanes.extend(corners.iter().flatten());
        lanes
    }

    fn square(kind: u32, low: f32, high: f32) -> Vec<f32> {
        area(kind, &[[low, low], [high, low], [high, high], [low, high]])
    }

    fn ask(operation: u32, numbers: [f32; 4], values: &[f32]) -> Answer {
        Service::<()>::call(&mut Navmesh, &Request { id: 1, operation, numbers, values }).unwrap()
    }

    /// A 10 by 10 floor with a 4 by 4 block in its middle.
    fn floor(radius: f32) -> Vec<f32> {
        [vec![radius], square(navmesh_area::WALKABLE, 0.0, 10.0), square(navmesh_area::OBSTACLE, 3.0, 7.0)].concat()
    }

    fn points(answer: &Answer) -> Vec<[f32; 2]> {
        answer.values.chunks(2).map(|point| [point[0], point[1]]).collect()
    }

    #[test]
    fn the_way_goes_straight_where_it_can_and_round_corners_where_it_cannot() {
        // Along the bottom, clear of the block: straight there.
        let answer = ask(navmesh_op::FIND, [1.0, 1.0, 9.0, 1.0], &floor(0.0));
        assert_eq!(points(&answer), [[1.0, 1.0], [9.0, 1.0]]);
        // Across the block: round one of its sides, turning at its two corners.
        let answer = ask(navmesh_op::FIND, [5.0, 1.0, 5.0, 9.0], &floor(0.0));
        let way = points(&answer);
        assert_eq!(answer.value, 4.0);
        assert!(
            way == [[5.0, 1.0], [3.0, 3.0], [3.0, 7.0], [5.0, 9.0]]
                || way == [[5.0, 1.0], [7.0, 3.0], [7.0, 7.0], [5.0, 9.0]],
            "{way:?}"
        );
        // The same way every time.
        assert_eq!(ask(navmesh_op::FIND, [5.0, 1.0, 5.0, 9.0], &floor(0.0)), answer);
    }

    #[test]
    fn a_radius_keeps_the_way_off_corners_and_out_of_narrow_gaps() {
        let answer = ask(navmesh_op::FIND, [5.0, 1.0, 5.0, 9.0], &floor(0.5));
        let way = points(&answer);
        assert_eq!(way.len(), 4);
        // The turns sit half a metre in from the block's corners, along
        // the portals: never closer than the radius to a corner.
        for turn in &way[1..3] {
            let nearest = [[3.0f32, 3.0], [7.0, 3.0], [3.0, 7.0], [7.0, 7.0]]
                .iter()
                .map(|corner| ((turn[0] - corner[0]).powi(2) + (turn[1] - corner[1]).powi(2)).sqrt())
                .fold(f32::INFINITY, f32::min);
            assert!((nearest - 0.5).abs() < 1e-4, "{way:?}");
        }
        // Two blocks a metre apart: an agent of radius 0.6 cannot pass between.
        let gap = [
            vec![0.6],
            square(navmesh_area::WALKABLE, 0.0, 10.0),
            area(navmesh_area::OBSTACLE, &[[0.0, 4.0], [4.5, 4.0], [4.5, 6.0], [0.0, 6.0]]),
            area(navmesh_area::OBSTACLE, &[[5.5, 4.0], [10.0, 4.0], [10.0, 6.0], [5.5, 6.0]]),
        ]
        .concat();
        assert_eq!(ask(navmesh_op::FIND, [5.0, 1.0, 5.0, 9.0], &gap).value, 0.0);
        let mut narrow = gap.clone();
        narrow[0] = 0.4;
        assert!(ask(navmesh_op::FIND, [5.0, 1.0, 5.0, 9.0], &narrow).value >= 2.0);
    }

    #[test]
    fn ends_off_the_navmesh_move_onto_it_and_walls_without_a_way_through_answer_none() {
        // The goal inside the block moves to its nearest edge.
        let answer = ask(navmesh_op::FIND, [1.0, 5.0, 4.0, 5.0], &floor(0.0));
        assert_eq!(points(&answer), [[1.0, 5.0], [3.0, 5.0]]);
        // A wall right across: no way.
        let walled = [
            vec![0.0],
            square(navmesh_area::WALKABLE, 0.0, 10.0),
            area(navmesh_area::OBSTACLE, &[[-1.0, 4.0], [11.0, 4.0], [11.0, 6.0], [-1.0, 6.0]]),
        ]
        .concat();
        assert_eq!(ask(navmesh_op::FIND, [5.0, 1.0, 5.0, 9.0], &walled), Answer::ok(0.0));
    }

    #[test]
    fn the_navmesh_covers_the_floor_less_the_block() {
        let answer = ask(navmesh_op::TRIANGLES, [0.0; 4], &floor(0.0));
        let covered: f32 = answer
            .values
            .chunks(6)
            .map(|t| ((t[2] - t[0]) * (t[5] - t[1]) - (t[4] - t[0]) * (t[3] - t[1])) / 2.0)
            .sum();
        assert_eq!(covered, 84.0, "100 square metres less 16, every triangle counterclockwise");
        assert_eq!(answer.value, (answer.values.len() / 6) as f32);
        // Malformed: no walkable area, or a polygon of two corners.
        assert_eq!(
            ask(navmesh_op::TRIANGLES, [0.0; 4], &[vec![0.0], square(navmesh_area::OBSTACLE, 0.0, 1.0)].concat())
                .status,
            service_status::FAILED
        );
        assert_eq!(
            ask(
                navmesh_op::TRIANGLES,
                [0.0; 4],
                &[vec![0.0], area(navmesh_area::WALKABLE, &[[0.0, 0.0], [1.0, 0.0]])].concat()
            )
            .status,
            service_status::FAILED
        );
    }
}
