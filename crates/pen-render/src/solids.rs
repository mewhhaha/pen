// Meshes convex solids (the brushes of a level) from their face planes.
use pen_protocol::Solid;

use crate::asset_store::MESH_VERTEX_FLOATS;
use crate::matrix::narrow;

type Point = [f64; 3];

/// Triangles of one material: position3, normal3, UV2, attributes2 per vertex.
pub(crate) struct SolidBatch {
    /// The material's catalog index.
    pub material: u32,
    pub vertices: Vec<f32>,
}

const EPSILON: f64 = 1e-3;
// Texture repeats per world unit on brush faces.
const UV_SCALE: f64 = 0.5;

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: Point, b: Point) -> Point {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn scale(a: Point, s: f64) -> Point {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn add(a: Point, b: Point) -> Point {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn subtract(a: Point, b: Point) -> Point {
    add(a, scale(b, -1.0))
}
fn length(a: Point) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
fn normalize(a: Point) -> Point {
    scale(a, 1.0 / length(a))
}

struct Polygon {
    normal: Point,
    material: u32,
    points: Vec<Point>,
}

/// Each face's polygon, counter-clockwise seen from outside the solid.
fn solid_polygons(solid: &Solid) -> Vec<Polygon> {
    struct Plane {
        normal: Point,
        distance: f64,
        material: u32,
    }
    let planes: Vec<Plane> = solid
        .faces
        .iter()
        .map(|face| {
            let normal = [f64::from(face.normal.x), f64::from(face.normal.y), f64::from(face.normal.z)];
            Plane {
                normal: normalize(normal),
                distance: f64::from(face.distance) / length(normal),
                material: face.material,
            }
        })
        .collect();
    let inside = |point: Point| planes.iter().all(|plane| dot(plane.normal, point) <= plane.distance + EPSILON);
    let mut polygons = Vec::new();
    for (i, plane) in planes.iter().enumerate() {
        let mut points: Vec<Point> = Vec::new();
        for j in 0..planes.len() {
            for k in j + 1..planes.len() {
                if j == i || k == i {
                    continue;
                }
                let (b, c) = (&planes[j], &planes[k]);
                let bc = cross(b.normal, c.normal);
                let determinant = dot(plane.normal, bc);
                if determinant.abs() < 1e-9 {
                    continue;
                }
                let point = scale(
                    add(
                        add(scale(bc, plane.distance), scale(cross(c.normal, plane.normal), b.distance)),
                        scale(cross(plane.normal, b.normal), c.distance),
                    ),
                    1.0 / determinant,
                );
                if inside(point) && !points.iter().any(|&known| length(subtract(known, point)) < EPSILON) {
                    points.push(point);
                }
            }
        }
        if points.len() < 3 {
            continue;
        }
        let center = scale(points.iter().copied().fold([0.0; 3], add), 1.0 / points.len() as f64);
        let u =
            normalize(cross(plane.normal, if plane.normal[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] }));
        let v = cross(plane.normal, u);
        let angle = |point: Point| {
            let offset = subtract(point, center);
            dot(offset, v).atan2(dot(offset, u))
        };
        points.sort_by(|a, b| angle(*a).total_cmp(&angle(*b)));
        polygons.push(Polygon { normal: plane.normal, material: plane.material, points });
    }
    polygons
}

// World-aligned texture coordinates on the plane facing the dominant axis.
fn uv(point: Point, normal: Point) -> [f64; 2] {
    let [x, y, z] = normal.map(f64::abs);
    if x >= y && x >= z {
        return [point[2] * UV_SCALE, -point[1] * UV_SCALE];
    }
    if y >= z {
        return [point[0] * UV_SCALE, point[2] * UV_SCALE];
    }
    [point[0] * UV_SCALE, -point[1] * UV_SCALE]
}

/// Triangle lists for `solids`, one per material, in the order each material
/// first appears.
pub(crate) fn mesh_solids(solids: &[Solid]) -> Vec<SolidBatch> {
    let mut batches: Vec<SolidBatch> = Vec::new();
    for solid in solids {
        for polygon in solid_polygons(solid) {
            let index = match batches.iter().position(|batch| batch.material == polygon.material) {
                Some(index) => index,
                None => {
                    batches.push(SolidBatch { material: polygon.material, vertices: Vec::new() });
                    batches.len() - 1
                }
            };
            let vertices = &mut batches[index].vertices;
            let first = polygon.points[0];
            let rest = &polygon.points[1..];
            for index in 0..rest.len().saturating_sub(1) {
                for point in [first, rest[index], rest[index + 1]] {
                    let [u, v] = uv(point, polygon.normal);
                    let vertex = [
                        point[0],
                        point[1],
                        point[2],
                        polygon.normal[0],
                        polygon.normal[1],
                        polygon.normal[2],
                        u,
                        v,
                        0.0,
                        0.0,
                    ];
                    debug_assert_eq!(vertex.len(), MESH_VERTEX_FLOATS);
                    vertices.extend(vertex.map(narrow));
                }
            }
        }
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_protocol::{SolidFace, Vector};

    #[test]
    fn a_cube_meshes_to_twelve_triangles() {
        let face = |x: f32, y: f32, z: f32| SolidFace { normal: Vector { x, y, z }, distance: 1.0, material: 3 };
        let cube = Solid {
            faces: vec![
                face(1.0, 0.0, 0.0),
                face(-1.0, 0.0, 0.0),
                face(0.0, 1.0, 0.0),
                face(0.0, -1.0, 0.0),
                face(0.0, 0.0, 1.0),
                face(0.0, 0.0, -1.0),
            ],
        };
        let batches = mesh_solids(&[cube]);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].material, 3);
        assert_eq!(batches[0].vertices.len(), 12 * 3 * MESH_VERTEX_FLOATS);
    }
}
