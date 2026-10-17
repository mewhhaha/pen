//! Colliders as the physics and spatial services read them: a record of
//! [`RECORD_LANES`] numbers (its shape, `COLLIDER_SHAPE_*`; its size; its
//! position; its rotation as a quaternion x y z w), then the shape's data
//! (a convex hull's points, a heightfield's heights, a mesh's vertices and
//! triangles), as the schema's `ColliderShape` lays them out.
use pen_protocol::boundary::{MAX_SERVICE_DATA, whole};
use pen_protocol::collider_shape;
use rapier3d::parry::utils::Array2;
use rapier3d::prelude::{Pose, Rotation, SharedShape, Vector};

use crate::Lanes;

/// The numbers of a collider's record before its shape's data.
pub const RECORD_LANES: usize = 11;
/// The farthest coordinate, and the largest size, height or speed, a
/// collider or body may have, in metres (a second). Past it f32 keeps no
/// centimetres, and parry's contact queries break down (a heightfield a
/// thousand million million metres high makes its EPA panic).
pub const REACH: f32 = 1.0e6;

/// The next `count` numbers, each within [`REACH`].
pub fn reach<'a>(lanes: &mut Lanes<'a>, count: usize) -> Option<&'a [f32]> {
    lanes.numbers(count).filter(|numbers| numbers.iter().all(|number| number.abs() <= REACH))
}

/// A collider's record: its shape and size, and where it is.
pub struct Record {
    shape: u32,
    size: Vector,
    pub pose: Pose,
}

/// Three numbers as a vector.
pub fn vector(lanes: &mut Lanes) -> Option<Vector> {
    let &[x, y, z] = reach(lanes, 3)? else { return None };
    Some(Vector::new(x, y, z))
}

/// A collider's record; None for an unknown shape or a rotation that is no
/// rotation (zero).
pub fn record(lanes: &mut Lanes) -> Option<Record> {
    let shape = u32::try_from(lanes.count(collider_shape::NAMES.len() - 1)?).ok()?;
    let size = vector(lanes)?;
    let position = vector(lanes)?;
    let &[x, y, z, w] = lanes.numbers(4)? else { return None };
    let rotation = Rotation::from_xyzw(x, y, z, w);
    let length = rotation.length_squared();
    if !(length.is_finite() && length > 1e-12) {
        return None;
    }
    Some(Record { shape, size, pose: Pose::from_parts(position, rotation.normalize()) })
}

/// The record's shape, reading its data; None for a size or data that make
/// no shape.
pub fn shape(record: &Record, lanes: &mut Lanes) -> Option<SharedShape> {
    let Vector { x, y, z } = record.size;
    let positive = |values: &[f32]| values.iter().all(|value| *value > 0.0);
    match record.shape {
        collider_shape::BALL => positive(&[x]).then(|| SharedShape::ball(x)),
        collider_shape::CUBOID => positive(&[x, y, z]).then(|| SharedShape::cuboid(x, y, z)),
        collider_shape::CAPSULE => positive(&[x, y]).then(|| SharedShape::capsule_y(y, x)),
        collider_shape::CYLINDER => positive(&[x, y]).then(|| SharedShape::cylinder(y, x)),
        collider_shape::CONVEX => {
            let count =
                whole(x).map(|count| count as usize).filter(|count| (4..=MAX_SERVICE_DATA / 3).contains(count))?;
            let points: Vec<Vector> =
                reach(lanes, count * 3)?.chunks(3).map(|point| Vector::new(point[0], point[1], point[2])).collect();
            SharedShape::convex_hull(&points)
        }
        collider_shape::HEIGHTFIELD => {
            let (columns, rows) = (whole(x)? as usize, whole(y)? as usize);
            let samples = columns.checked_mul(rows).filter(|samples| *samples <= MAX_SERVICE_DATA)?;
            if columns < 2 || rows < 2 || !positive(&[z]) {
                return None;
            }
            // Row by row from -z on the wire; parry's array is column major,
            // its rows along z and its columns along x.
            let heights = reach(lanes, samples)?;
            let mut data = vec![0.0; samples];
            for (index, height) in heights.iter().enumerate() {
                data[index / columns + (index % columns) * rows] = *height;
            }
            let scale = Vector::new((columns - 1) as f32 * z, 1.0, (rows - 1) as f32 * z);
            Some(SharedShape::heightfield(Array2::new(rows, columns, data), scale))
        }
        collider_shape::TRIMESH => {
            let (vertices, triangles) = (whole(x)? as usize, whole(y)? as usize);
            if vertices < 3 || triangles == 0 || vertices.max(triangles) > MAX_SERVICE_DATA / 3 {
                return None;
            }
            let points: Vec<Vector> =
                reach(lanes, vertices * 3)?.chunks(3).map(|point| Vector::new(point[0], point[1], point[2])).collect();
            let corners = lanes.numbers(triangles * 3)?;
            let index = |corner: f32| whole(corner).filter(|corner| (*corner as usize) < vertices);
            let indices: Option<Vec<[u32; 3]>> = corners
                .chunks(3)
                .map(|corner| Some([index(corner[0])?, index(corner[1])?, index(corner[2])?]))
                .collect();
            SharedShape::trimesh(points, indices?).ok()
        }
        _ => None,
    }
}

/// A collider's record and its shape, as the spatial service's colliders
/// come.
pub fn collider(lanes: &mut Lanes) -> Option<(Pose, SharedShape)> {
    let record = record(lanes)?;
    let shape = shape(&record, lanes)?;
    Some((record.pose, shape))
}

#[cfg(test)]
mod tests {
    use rapier3d::prelude::Ray;

    use super::*;

    /// A record's lanes: the shape, size, position and rotation.
    fn lanes(shape: u32, size: [f32; 3], position: [f32; 3]) -> Vec<f32> {
        let mut lanes = vec![shape as f32];
        lanes.extend(size);
        lanes.extend(position);
        lanes.extend([0.0, 0.0, 0.0, 1.0]);
        lanes
    }

    #[test]
    fn a_heightfield_lies_row_by_row_from_its_low_corner() {
        // 3 columns by 2 rows, a metre apart: higher along +x, and the far
        // row (+z) a metre above the near one.
        let mut values = lanes(collider_shape::HEIGHTFIELD, [3.0, 2.0, 1.0], [0.0, 0.0, 0.0]);
        values.extend([0.0, 1.0, 2.0, 1.0, 2.0, 3.0]);
        let (pose, shape) = collider(&mut Lanes::new(&values)).unwrap();
        let down = |x: f32, z: f32| {
            let ray = Ray::new(Vector::new(x, 10.0, z), Vector::new(0.0, -1.0, 0.0));
            10.0 - shape.cast_ray(&pose, &ray, 100.0, true).unwrap()
        };
        // Centred: x from -1 to 1, z from -0.5 to 0.5.
        assert!((down(-1.0, -0.5) - 0.0).abs() < 1e-5);
        assert!((down(1.0, -0.5) - 2.0).abs() < 1e-5);
        assert!((down(1.0, 0.5) - 3.0).abs() < 1e-5);
        assert!((down(0.0, 0.0) - 1.5).abs() < 1e-5);
    }

    #[test]
    fn shapes_refuse_sizes_and_data_that_make_none() {
        let read = |values: Vec<f32>| collider(&mut Lanes::new(&values)).is_some();
        assert!(read(lanes(collider_shape::BALL, [0.5, 0.0, 0.0], [0.0; 3])));
        assert!(!read(lanes(collider_shape::BALL, [0.0, 0.0, 0.0], [0.0; 3])));
        assert!(!read(lanes(u32::try_from(collider_shape::NAMES.len()).unwrap(), [1.0; 3], [0.0; 3])));
        let mut spun = lanes(collider_shape::CUBOID, [1.0; 3], [0.0; 3]);
        spun[7..11].copy_from_slice(&[0.0; 4]);
        assert!(!read(spun), "no rotation");
        // A mesh whose triangle names a vertex it lacks, and one cut short.
        let mut mesh = lanes(collider_shape::TRIMESH, [3.0, 1.0, 0.0], [0.0; 3]);
        mesh.extend([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(read([mesh.clone(), vec![0.0, 1.0, 2.0]].concat()));
        assert!(!read([mesh.clone(), vec![0.0, 1.0, 3.0]].concat()));
        assert!(!read([mesh, vec![0.0, 1.0]].concat()));
    }
}
