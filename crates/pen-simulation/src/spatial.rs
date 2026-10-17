use pen_protocol::boundary::MAX_SERVICE_DATA;
use pen_protocol::{service, spatial_op};
use pen_runtime::services::{Answer, Request, Service};
use rapier3d::parry::query::{DefaultQueryDispatcher, QueryDispatcher};
use rapier3d::prelude::{
    Collider, ColliderBuilder, ColliderHandle, PhysicsWorld, Pose, QueryFilter, Ray, Shape, Vector,
};

use crate::shapes::{self, vector};
use crate::{Lanes, counted};

/// The numbers of a ray: its origin and direction.
pub const RAY_LANES: usize = 6;
/// The numbers the answer gives each ray or point: the collider (its index
/// + 1, 0 for none), the distance, and a normal or a point.
pub const FOUND_LANES: usize = 5;

/// Ray and shape queries against colliders the request sends: the first
/// collider each ray hits, those a shape touches, the nearest to each
/// point. The colliders go into a rapier world's broad phase for the
/// request, as `Physics`'s bodies do, so a query looks only near where it
/// goes; like it, nothing is kept for the next request.
pub struct Spatial;

impl<Input> Service<Input> for Spatial {
    fn code(&self) -> u32 {
        service::SPATIAL
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        if !matches!(request.operation, spatial_op::RAYCAST | spatial_op::OVERLAP | spatial_op::NEAREST) {
            return Some(Answer::unknown());
        }
        Some(query(request).unwrap_or_else(Answer::failed))
    }
}

/// The world of the request's `a` colliders, each knowing its index.
fn world(count: f32, lanes: &mut Lanes) -> Option<PhysicsWorld> {
    let count = pen_protocol::boundary::whole(count)? as usize;
    let mut world = PhysicsWorld::new();
    for index in 0..count.min(MAX_SERVICE_DATA) {
        let (pose, shape) = shapes::collider(lanes)?;
        world.insert_collider(ColliderBuilder::new(shape).position(pose).user_data(index as u128), None);
    }
    if world.colliders.len() != count {
        return None;
    }
    // Puts the colliders in the broad phase, which queries go through.
    world.detect_collisions(&(), &());
    Some(world)
}

/// Whether `shape` at `pose` touches the collider. parry has no
/// intersection test for some pairs (a capsule and a heightfield), which
/// rapier's own overlap query then leaves out; its contact query has them.
fn touches(pose: &Pose, shape: &dyn Shape, collider: &Collider) -> bool {
    let between = pose.inv_mul(collider.position());
    match DefaultQueryDispatcher.intersection_test(&between, shape, collider.shape()) {
        Ok(hit) => hit.intersecting,
        Err(_) => DefaultQueryDispatcher
            .contact(&between, shape, collider.shape(), 0.0)
            .is_ok_and(|contact| contact.is_some_and(|contact| contact.dist <= 0.0)),
    }
}

/// A collider's lane in an answer: its index + 1.
fn found(world: &PhysicsWorld, handle: ColliderHandle) -> f32 {
    world.colliders[handle].user_data as f32 + 1.0
}

fn query(request: &Request) -> Option<Answer> {
    let [count, reach, solid, _] = request.numbers;
    let mut lanes = Lanes::new(request.values);
    let world = world(count, &mut lanes)?;
    let mut values = Vec::new();
    let mut hits = 0;
    match request.operation {
        spatial_op::RAYCAST => {
            if reach.is_nan() || reach < 0.0 || !matches!(solid, 0.0 | 1.0) {
                return None;
            }
            while !lanes.done() {
                let &[x, y, z, dx, dy, dz] = shapes::reach(&mut lanes, RAY_LANES)? else { return None };
                let (origin, direction) = (Vector::new(x, y, z), Vector::new(dx, dy, dz));
                let length = direction.length();
                if !(length > 0.0 && length.is_finite()) {
                    return None;
                }
                let ray = Ray::new(origin, direction / length);
                match world.cast_ray_and_get_normal(&ray, reach, solid == 1.0, QueryFilter::default()) {
                    Some((handle, hit)) => {
                        hits += 1;
                        values.extend([found(&world, handle), hit.time_of_impact]);
                        values.extend(hit.normal.to_array());
                    }
                    None => values.extend([0.0; FOUND_LANES]),
                }
            }
        }
        spatial_op::OVERLAP => {
            let (pose, shape) = shapes::collider(&mut lanes)?;
            if !lanes.done() {
                return None;
            }
            let mut touched: Vec<f32> = world
                .intersect_aabb_conservative(shape.compute_aabb(&pose), QueryFilter::default())
                .filter(|(_, collider)| touches(&pose, shape.as_ref(), collider))
                .map(|(handle, _)| found(&world, handle) - 1.0)
                .collect();
            touched.sort_by(f32::total_cmp);
            hits = touched.len();
            values = touched;
        }
        _ => {
            if reach.is_nan() || reach < 0.0 {
                return None;
            }
            while !lanes.done() {
                let point = vector(&mut lanes)?;
                match world.project_point(point, reach, true, QueryFilter::default()) {
                    Some((handle, projection)) => {
                        hits += 1;
                        let distance = if projection.is_inside { 0.0 } else { (projection.point - point).length() };
                        values.extend([found(&world, handle), distance]);
                        values.extend(projection.point.to_array());
                    }
                    None => values.extend([0.0; FOUND_LANES]),
                }
            }
        }
    }
    Some(counted(hits, values))
}

#[cfg(test)]
mod tests {
    use pen_protocol::{collider_shape, service_status};

    use super::*;

    /// A collider's lanes: shape, size, position and no rotation.
    fn collider(shape: u32, size: [f32; 3], position: [f32; 3]) -> Vec<f32> {
        let mut lanes = vec![shape as f32];
        lanes.extend(size);
        lanes.extend(position);
        lanes.extend([0.0, 0.0, 0.0, 1.0]);
        lanes
    }

    fn ask(operation: u32, numbers: [f32; 4], values: &[f32]) -> Answer {
        Service::<()>::call(&mut Spatial, &Request { id: 1, operation, numbers, values }).unwrap()
    }

    /// A box at x 5 and a ball at x 10, along the x axis.
    fn colliders() -> Vec<f32> {
        [
            collider(collider_shape::CUBOID, [1.0, 1.0, 1.0], [5.0, 0.0, 0.0]),
            collider(collider_shape::BALL, [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]),
        ]
        .concat()
    }

    #[test]
    fn a_ray_hits_the_first_collider_in_its_way() {
        let rays = [
            // Along +x from the origin: the box's near face, 4 metres on.
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            // Down onto the ball from above: its top, a metre over its centre.
            [10.0, 5.0, 0.0, 0.0, -1.0, 0.0],
            // Away from both.
            [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        ];
        let values = [colliders(), rays.concat()].concat();
        let answer = ask(spatial_op::RAYCAST, [2.0, 100.0, 1.0, 0.0], &values);
        assert_eq!(answer.value, 2.0);
        assert_eq!(answer.values[..5], [1.0, 4.0, -1.0, 0.0, 0.0]);
        assert_eq!(answer.values[5..7], [2.0, 4.0]);
        assert!((answer.values[8] - 1.0).abs() < 1e-6, "the ball's top faces up");
        assert_eq!(answer.values[10..], [0.0; 5]);
        // Out of reach: no hit.
        let short = ask(spatial_op::RAYCAST, [2.0, 3.0, 1.0, 0.0], &values);
        assert_eq!(short.values[0], 0.0);
    }

    #[test]
    fn a_shape_touches_what_it_overlaps_and_points_find_the_nearest() {
        // A ball between the two, reaching both.
        let probe = collider(collider_shape::BALL, [2.6, 0.0, 0.0], [7.5, 0.0, 0.0]);
        let answer = ask(spatial_op::OVERLAP, [2.0, 0.0, 0.0, 0.0], &[colliders(), probe].concat());
        assert_eq!((answer.value, answer.values), (2.0, vec![0.0, 1.0]));
        // A capsule dipping into a flat heightfield, which parry tests by contact.
        let mut ground = collider(collider_shape::HEIGHTFIELD, [2.0, 2.0, 4.0], [0.0; 3]);
        ground.extend([0.0; 4]);
        let dipping = collider(collider_shape::CAPSULE, [0.5, 1.0, 0.0], [0.0, 1.2, 0.0]);
        assert_eq!(ask(spatial_op::OVERLAP, [1.0, 0.0, 0.0, 0.0], &[ground.clone(), dipping].concat()).values, [0.0]);
        let above = collider(collider_shape::CAPSULE, [0.5, 1.0, 0.0], [0.0, 1.6, 0.0]);
        assert_eq!(ask(spatial_op::OVERLAP, [1.0, 0.0, 0.0, 0.0], &[ground, above].concat()).value, 0.0);
        let probe = collider(collider_shape::BALL, [0.5, 0.0, 0.0], [7.5, 0.0, 0.0]);
        assert_eq!(ask(spatial_op::OVERLAP, [2.0, 0.0, 0.0, 0.0], &[colliders(), probe].concat()).value, 0.0);
        // Nearest: a point by the ball, one in the box, one far off.
        let points = [12.0, 0.0, 0.0, 5.0, 0.5, 0.0, 50.0, 0.0, 0.0];
        let answer = ask(spatial_op::NEAREST, [2.0, 10.0, 0.0, 0.0], &[colliders(), points.to_vec()].concat());
        assert_eq!(answer.value, 2.0);
        assert_eq!(answer.values[..5], [2.0, 1.0, 11.0, 0.0, 0.0]);
        assert_eq!(answer.values[5..7], [1.0, 0.0], "inside the box");
        assert_eq!(answer.values[10..], [0.0; 5]);
    }

    #[test]
    fn queries_that_break_their_layout_fail() {
        let failed =
            |operation, numbers, values: &[f32]| ask(operation, numbers, values).status == service_status::FAILED;
        assert!(failed(spatial_op::RAYCAST, [3.0, 10.0, 1.0, 0.0], &colliders()), "a collider short");
        assert!(
            failed(spatial_op::RAYCAST, [2.0, 10.0, 1.0, 0.0], &[colliders(), vec![0.0; 6]].concat()),
            "no direction"
        );
        assert!(
            failed(spatial_op::RAYCAST, [2.0, 10.0, 1.0, 0.0], &[colliders(), vec![1.0; 4]].concat()),
            "a ray cut short"
        );
        assert!(failed(spatial_op::OVERLAP, [2.0, 0.0, 0.0, 0.0], &colliders()), "nothing to overlap");
        assert_eq!(ask(9, [0.0; 4], &[]).status, service_status::UNKNOWN);
    }
}
