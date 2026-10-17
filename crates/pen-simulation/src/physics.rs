use pen_protocol::{physics_body, physics_op, service};
use pen_runtime::services::{Answer, Request, Service};
use rapier3d::prelude::{ColliderBuilder, PhysicsWorld, RigidBodyBuilder, Vector};

use crate::shapes::{self, RECORD_LANES};
use crate::{Lanes, counted};

/// The numbers of a body before its shape's data: its kind, its collider's
/// record, its linear and angular velocity, friction, restitution and
/// density.
pub const BODY_LANES: usize = 1 + RECORD_LANES + 9;
/// The numbers the answer gives each body: its position, rotation, and
/// linear and angular velocity.
pub const STATE_LANES: usize = 13;
/// The numbers of a contact in the answer: the two bodies, the deepest
/// point, the normal and the distance.
pub const CONTACT_LANES: usize = 9;
/// The most steps one request may ask for: ten seconds at 60 a second.
const MOST_STEPS: u32 = 600;

/// Rigid-body physics as a service: rapier steps the bodies the request
/// sends, with gravity along y, and answers where they went and what
/// touches. The world lives for one request: the game keeps its bodies in
/// its own state and sends them again, so a rewind, reload or replay asks
/// the same and gets the same, and nothing hides in the host. What a
/// longer-lived world would carry over (contact impulses that warm-start
/// the solver, sleeping) starts afresh each request; asking for several
/// steps at once keeps resting stacks steadier.
pub struct Physics;

impl<Input> Service<Input> for Physics {
    fn code(&self) -> u32 {
        service::PHYSICS
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        if request.operation != physics_op::STEP {
            return Some(Answer::unknown());
        }
        Some(step(request).unwrap_or_else(Answer::failed))
    }
}

/// A touching pair: the bodies (the lower first), the deepest point on the
/// first, the normal from it to the second, and their distance.
struct Contact {
    bodies: (usize, usize),
    point: Vector,
    normal: Vector,
    distance: f32,
}

fn step(request: &Request) -> Option<Answer> {
    let [steps, seconds, gravity, _] = request.numbers;
    let steps = pen_protocol::boundary::whole(steps).filter(|steps| (1..=MOST_STEPS).contains(steps))?;
    if !(seconds > 0.0 && seconds <= 1.0 && gravity.is_finite()) {
        return None;
    }
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::new(0.0, gravity, 0.0);
    world.integration_parameters.dt = seconds;
    let mut lanes = Lanes::new(request.values);
    let mut bodies = Vec::with_capacity(request.values.len() / BODY_LANES);
    while !lanes.done() {
        let kind = u32::try_from(lanes.count(physics_body::NAMES.len() - 1)?).ok()?;
        let record = shapes::record(&mut lanes)?;
        let (linear, angular) = (shapes::vector(&mut lanes)?, shapes::vector(&mut lanes)?);
        let &[friction, restitution, density] = lanes.numbers(3)? else { return None };
        if friction < 0.0 || restitution < 0.0 || density <= 0.0 {
            return None;
        }
        let shape = shapes::shape(&record, &mut lanes)?;
        let body = match kind {
            physics_body::FIXED => RigidBodyBuilder::fixed(),
            physics_body::DYNAMIC => RigidBodyBuilder::dynamic(),
            _ => RigidBodyBuilder::kinematic_velocity_based(),
        };
        let index = bodies.len();
        let collider = ColliderBuilder::new(shape)
            .friction(friction)
            .restitution(restitution)
            .density(density)
            .user_data(index as u128);
        let (body, _) = world.insert(body.pose(record.pose).linvel(linear).angvel(angular), collider);
        bodies.push(body);
    }
    for _ in 0..steps {
        world.step();
    }

    let mut values = Vec::with_capacity(bodies.len() * STATE_LANES);
    for handle in &bodies {
        let body = &world.bodies[*handle];
        let (position, rotation) = (body.translation(), body.rotation());
        values.extend(position.to_array());
        values.extend([rotation.x, rotation.y, rotation.z, rotation.w]);
        values.extend(body.linvel().to_array());
        values.extend(body.angvel().to_array());
    }
    let mut contacts: Vec<Contact> = world
        .contact_pairs()
        .filter(|pair| pair.has_any_active_contact())
        .filter_map(|pair| {
            let (first, second) = (&world.colliders[pair.collider1], &world.colliders[pair.collider2]);
            let (manifold, deepest) = pair
                .manifolds()
                .iter()
                .flat_map(|manifold| manifold.points.iter().map(move |point| (manifold, point)))
                .min_by(|(_, a), (_, b)| a.dist.total_cmp(&b.dist))?;
            let (a, b) = (usize::try_from(first.user_data).ok()?, usize::try_from(second.user_data).ok()?);
            Some(if a < b {
                let point = first.position().transform_point(deepest.local_p1);
                Contact { bodies: (a, b), point, normal: manifold.data.normal, distance: deepest.dist }
            } else {
                let point = second.position().transform_point(deepest.local_p2);
                Contact { bodies: (b, a), point, normal: -manifold.data.normal, distance: deepest.dist }
            })
        })
        .collect();
    contacts.sort_by_key(|contact| contact.bodies);
    // As many contacts as fit beside the bodies.
    contacts.truncate(pen_protocol::boundary::MAX_SERVICE_DATA.saturating_sub(values.len()) / CONTACT_LANES);
    for contact in &contacts {
        values.extend([contact.bodies.0 as f32, contact.bodies.1 as f32]);
        values.extend(contact.point.to_array());
        values.extend(contact.normal.to_array());
        values.push(contact.distance);
    }
    Some(counted(contacts.len(), values))
}

#[cfg(test)]
mod tests {
    use pen_protocol::{collider_shape, service_status};

    use super::*;

    /// A body's lanes: kind, shape, size, position, no rotation, velocity,
    /// no spin, friction 0.5, no bounce, density 1.
    fn body(kind: u32, shape: u32, size: [f32; 3], position: [f32; 3], velocity: [f32; 3]) -> Vec<f32> {
        let mut lanes = vec![kind as f32, shape as f32];
        lanes.extend(size);
        lanes.extend(position);
        lanes.extend([0.0, 0.0, 0.0, 1.0]);
        lanes.extend(velocity);
        lanes.extend([0.0, 0.0, 0.0, 0.5, 0.0, 1.0]);
        lanes
    }

    fn ask(numbers: [f32; 4], values: &[f32]) -> Answer {
        Service::<()>::call(&mut Physics, &Request { id: 1, operation: physics_op::STEP, numbers, values }).unwrap()
    }

    #[test]
    fn a_ball_falls_onto_the_ground_and_rests_touching_it() {
        let ground = body(physics_body::FIXED, collider_shape::CUBOID, [5.0, 0.5, 5.0], [0.0, -0.5, 0.0], [0.0; 3]);
        let ball = body(physics_body::DYNAMIC, collider_shape::BALL, [0.5, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0; 3]);
        let values = [ground, ball].concat();
        let answer = ask([180.0, 1.0 / 60.0, -9.81, 0.0], &values);
        assert_eq!(answer.status, service_status::OK);
        // The ground stays; the ball rests on it, its centre a radius up.
        assert_eq!(answer.values[..3], [0.0, -0.5, 0.0]);
        let ball = &answer.values[STATE_LANES..STATE_LANES * 2];
        assert!((ball[1] - 0.5).abs() < 0.02, "the ball rests at {}", ball[1]);
        assert!(ball[7..10].iter().all(|speed| speed.abs() < 0.05));
        // One contact: the ground (0) and the ball (1), the normal up from
        // the ground to the ball, at the ball's foot.
        assert_eq!(answer.value, 1.0);
        let contact = &answer.values[STATE_LANES * 2..];
        assert_eq!(contact[..2], [0.0, 1.0]);
        assert!((contact[3]).abs() < 0.02, "the point is at the floor, {}", contact[3]);
        assert!((contact[6] - 1.0).abs() < 1e-4, "the normal points up");
        // Asked again, the same bits.
        assert_eq!(ask([180.0, 1.0 / 60.0, -9.81, 0.0], &values), answer);
    }

    #[test]
    fn balls_roll_down_a_heightfield_and_stay_on_it() {
        // A slope rising 4 metres over 8 along x, its heights row by row.
        let mut ground = body(physics_body::FIXED, collider_shape::HEIGHTFIELD, [5.0, 3.0, 2.0], [0.0; 3], [0.0; 3]);
        ground.extend([[-2.0, -1.0, 0.0, 1.0, 2.0]; 3].concat());
        let balls: Vec<f32> = (0..3u8)
            .flat_map(|ball| {
                let at = [f32::from(ball) * 2.0 - 2.0, 3.0, 0.0];
                body(physics_body::DYNAMIC, collider_shape::BALL, [0.25, 0.0, 0.0], at, [0.0; 3])
            })
            .collect();
        let answer = ask([90.0, 1.0 / 60.0, -9.81, 0.0], &[ground, balls].concat());
        for ball in 1..4 {
            let state = &answer.values[ball * STATE_LANES..];
            // The ground under it is half its x, and it rolls downhill (-x).
            assert!(state[1] >= state[0] / 2.0, "ball {ball} at {:?} is above the slope", &state[..3]);
            assert!(state[1] - state[0] / 2.0 < 0.4, "and on it");
            assert!(state[7] < 0.0);
        }
    }

    #[test]
    fn a_kinematic_body_moves_by_its_velocity_alone() {
        let mover = body(physics_body::KINEMATIC, collider_shape::CUBOID, [0.5; 3], [0.0; 3], [2.0, 0.0, 0.0]);
        let answer = ask([60.0, 1.0 / 60.0, -9.81, 0.0], &mover);
        assert!((answer.values[0] - 2.0).abs() < 1e-4, "{}", answer.values[0]);
        assert_eq!(answer.values[1], 0.0, "no gravity on it");
    }

    #[test]
    fn a_request_that_breaks_its_layout_fails() {
        let ball = body(physics_body::DYNAMIC, collider_shape::BALL, [0.5, 0.0, 0.0], [0.0; 3], [0.0; 3]);
        let failed = |numbers: [f32; 4], values: &[f32]| ask(numbers, values).status == service_status::FAILED;
        assert!(failed([0.0, 1.0 / 60.0, -9.81, 0.0], &ball), "no steps");
        assert!(failed([1.0, 0.0, -9.81, 0.0], &ball), "no time");
        assert!(failed([1.0, 1.0 / 60.0, -9.81, 0.0], &ball[..BODY_LANES - 1]), "cut short");
        let mut unknown = ball.clone();
        unknown[0] = 3.0;
        assert!(failed([1.0, 1.0 / 60.0, -9.81, 0.0], &unknown), "no such kind");
        // No bodies is an empty world, which steps.
        assert_eq!(ask([1.0, 1.0 / 60.0, -9.81, 0.0], &[]), Answer::ok(0.0));
        let other = Request { id: 1, operation: 1, numbers: [1.0, 0.1, 0.0, 0.0], values: &ball };
        assert_eq!(Service::<()>::call(&mut Physics, &other).unwrap().status, service_status::UNKNOWN);
    }
}
