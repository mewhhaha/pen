//! The same answers on every host: fixed requests to each simulation
//! service, and the digest of what a game would get back (the exact
//! ServiceAnswer and ServiceData events, bit for bit) that every host must
//! reproduce. Native tests check it here and through native pen's own
//! registry; the browser host checks it in wasm32
//! (pen-web's `qualify_services`, which crates/pen-web/tests/services.mjs
//! runs under node and `just conformance` in headless Chromium).
use pen_protocol::{
    GameEvent, ServiceRequest, collider_shape, navmesh_area, navmesh_op, noise_op, physics_body, physics_op, service,
    spatial_op,
};
use pen_runtime::services::{Request, Service, Services};

/// The digest every host's answers to [`requests`] must have.
pub const EXPECTED: u64 = 0x18bd_afd0_943f_14cf;

/// FNV-1a over 32-bit words.
fn fnv(hash: u64, word: u32) -> u64 {
    word.to_le_bytes().iter().fold(hash, |hash, byte| (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3))
}

/// The digest of answer events: each one's kind, request and lanes, as bits.
pub fn digest(events: &[GameEvent]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for event in events {
        let words = match event {
            GameEvent::ServiceAnswer { request, status, value, length } => {
                [0, *request, *status, value.to_bits(), *length]
            }
            GameEvent::ServiceData { request, a, b, c } => [1, *request, a.to_bits(), b.to_bits(), c.to_bits()],
            _ => [2, 0, 0, 0, 0],
        };
        hash = words.into_iter().fold(hash, fnv);
    }
    hash
}

fn request(id: u32, service: u32, operation: u32, numbers: [f32; 4], values: Vec<f32>) -> ServiceRequest {
    let [a, b, c, d] = numbers;
    ServiceRequest {
        request: id,
        service,
        operation,
        a,
        b,
        c,
        d,
        value_count: u32::try_from(values.len()).expect("a fixture fits a request"),
        values,
    }
}

/// A collider record: shape, size, position and rotation.
fn collider(shape: u32, size: [f32; 3], position: [f32; 3], rotation: [f32; 4]) -> Vec<f32> {
    [&[shape as f32][..], &size, &position, &rotation].concat()
}

/// A body: its kind, collider record, velocities and material, then its
/// shape's data.
fn body(kind: u32, collider: Vec<f32>, velocity: [f32; 6], material: [f32; 3], data: &[f32]) -> Vec<f32> {
    [&[kind as f32][..], &collider, &velocity, &material, data].concat()
}

/// A navmesh area: its kind and corners.
fn area(kind: u32, corners: &[[f32; 2]]) -> Vec<f32> {
    [vec![kind as f32, corners.len() as f32], corners.iter().flatten().copied().collect()].concat()
}

/// An eighth of a turn about y, and none, as quaternions.
const TILTED: [f32; 4] = [0.0, 0.382_683_43, 0.0, 0.923_879_5];
const UPRIGHT: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// The fixed requests: noise as a heightfield and at points, bodies falling
/// onto that heightfield, rays, an overlap and nearest points against
/// colliders of every shape, and navmesh ways and triangles.
pub fn requests() -> Vec<ServiceRequest> {
    // 17 by 17 heights a metre apart, which the physics and rays below
    // stand on as a heightfield.
    let hills = vec![0.08, 2.5, 2.0, 0.5, 0.0, -8.0, -8.0, 1.0];
    let ground = {
        let asked =
            Request { id: 0, operation: noise_op::HEIGHTFIELD, numbers: [17.0, 17.0, 2024.0, 5.0], values: &hills };
        Service::<()>::call(&mut crate::Noise, &asked).map(|answer| answer.values).unwrap_or_default()
    };
    let field =
        |position: [f32; 3]| (collider(collider_shape::HEIGHTFIELD, [17.0, 17.0, 1.0], position, UPRIGHT), &ground);
    let tetrahedron = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let pyramid = [-1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 1.5, 0.0];
    let pyramid_faces = [0.0, 1.0, 4.0, 1.0, 2.0, 4.0, 2.0, 3.0, 4.0, 3.0, 0.0, 4.0, 0.0, 2.0, 1.0, 0.0, 3.0, 2.0];
    let still = [0.0; 6];
    let material = [0.6, 0.1, 1.0];

    let (ground_collider, ground_heights) = field([0.0, 0.0, 0.0]);
    let bodies = [
        body(physics_body::FIXED, ground_collider, still, [0.8, 0.0, 1.0], ground_heights),
        body(
            physics_body::DYNAMIC,
            collider(collider_shape::CUBOID, [0.5, 0.5, 0.5], [0.0, 4.0, 0.0], UPRIGHT),
            still,
            material,
            &[],
        ),
        body(
            physics_body::DYNAMIC,
            collider(collider_shape::CUBOID, [0.5, 0.5, 0.5], [0.1, 5.2, 0.0], TILTED),
            still,
            material,
            &[],
        ),
        body(
            physics_body::DYNAMIC,
            collider(collider_shape::BALL, [0.4, 0.0, 0.0], [2.0, 6.0, 1.0], UPRIGHT),
            [-1.0, 0.0, 0.0, 0.0, 0.0, 3.0],
            [0.4, 0.6, 2.0],
            &[],
        ),
        body(
            physics_body::DYNAMIC,
            collider(collider_shape::CAPSULE, [0.3, 0.6, 0.0], [-2.0, 5.0, -1.0], TILTED),
            [0.5, 0.0, 0.5, 1.0, 0.0, 0.0],
            material,
            &[],
        ),
        body(
            physics_body::DYNAMIC,
            collider(collider_shape::CYLINDER, [0.5, 0.3, 0.0], [1.0, 7.0, -2.0], UPRIGHT),
            still,
            material,
            &[],
        ),
        body(
            physics_body::DYNAMIC,
            collider(collider_shape::CONVEX, [4.0, 0.0, 0.0], [-1.0, 8.0, 2.0], TILTED),
            still,
            material,
            &tetrahedron,
        ),
        body(
            physics_body::KINEMATIC,
            collider(collider_shape::CUBOID, [1.0, 0.2, 1.0], [-4.0, 3.0, 0.0], UPRIGHT),
            [1.5, 0.0, 0.0, 0.0, 0.5, 0.0],
            material,
            &[],
        ),
        body(
            physics_body::FIXED,
            collider(collider_shape::TRIMESH, [5.0, 6.0, 0.0], [3.0, 2.0, 3.0], UPRIGHT),
            still,
            material,
            &[&pyramid[..], &pyramid_faces].concat(),
        ),
    ]
    .concat();

    let (field_collider, field_heights) = field([0.0, -1.0, 0.0]);
    let colliders = [
        collider(collider_shape::CUBOID, [1.0, 2.0, 0.5], [4.0, 1.0, 0.0], TILTED),
        collider(collider_shape::BALL, [1.5, 0.0, 0.0], [-4.0, 2.0, 1.0], UPRIGHT),
        [
            collider(collider_shape::TRIMESH, [5.0, 6.0, 0.0], [0.0, 1.0, 4.0], TILTED),
            pyramid.to_vec(),
            pyramid_faces.to_vec(),
        ]
        .concat(),
        [field_collider, field_heights.clone()].concat(),
        collider(collider_shape::CYLINDER, [0.7, 1.5, 0.0], [0.0, 1.5, -4.0], UPRIGHT),
    ]
    .concat();
    // Rays down, a little aslant, from a 4 by 4 grid over the colliders.
    let rays: Vec<f32> = (0..16u8)
        .flat_map(|ray| {
            let (column, row) = (f32::from(ray % 4), f32::from(ray / 4));
            [column * 3.0 - 4.5, 8.0, row * 3.0 - 4.5, 0.2, -1.0, -0.1]
        })
        .collect();
    let probe = collider(collider_shape::CAPSULE, [2.2, 3.0, 0.0], [2.0, 1.0, -2.0], TILTED);
    let points = [5.0, 1.0, 1.0, -4.0, 2.5, 1.0, 0.0, 5.0, 4.0, 0.3, 3.0, -0.2, 9.0, 9.0, 9.0];

    let floor = [
        vec![0.3],
        area(navmesh_area::WALKABLE, &[[0.0, 0.0], [20.0, 0.0], [20.0, 12.0], [0.0, 12.0]]),
        area(navmesh_area::OBSTACLE, &[[4.0, 2.0], [6.0, 2.0], [6.0, 8.0], [10.0, 8.0], [10.0, 10.0], [4.0, 10.0]]),
        area(navmesh_area::OBSTACLE, &[[12.0, 1.0], [15.0, 5.0], [11.0, 6.0]]),
        area(navmesh_area::OBSTACLE, &[[16.5, 7.0], [18.0, 8.5], [16.5, 10.0], [15.0, 8.5]]),
        // An obstacle reaching out past the floor's edge, and overlapping another.
        area(navmesh_area::OBSTACLE, &[[8.0, -1.0], [9.0, -1.0], [9.0, 3.0], [8.0, 3.0]]),
        area(navmesh_area::OBSTACLE, &[[5.0, 7.0], [7.0, 7.0], [7.0, 9.0], [5.0, 9.0]]),
    ]
    .concat();

    vec![
        request(1, service::NOISE, noise_op::HEIGHTFIELD, [17.0, 17.0, 2024.0, 5.0], hills),
        request(
            2,
            service::NOISE,
            noise_op::SAMPLE,
            [0.0, 0.0, 99.0, 7.0],
            vec![0.03, 10.0, 2.1, 0.45, 1.0, 0.0, 0.0, -3.25, 7.5, 12345.678, -98765.4, 0.5, 0.25, -1000.0, 1000.0],
        ),
        request(3, service::PHYSICS, physics_op::STEP, [120.0, 1.0 / 60.0, -9.81, 0.0], bodies),
        request(4, service::SPATIAL, spatial_op::RAYCAST, [5.0, 20.0, 1.0, 0.0], [colliders.clone(), rays].concat()),
        request(5, service::SPATIAL, spatial_op::OVERLAP, [5.0, 0.0, 0.0, 0.0], [colliders.clone(), probe].concat()),
        request(6, service::SPATIAL, spatial_op::NEAREST, [5.0, 6.0, 0.0, 0.0], [colliders, points.to_vec()].concat()),
        request(7, service::NAVMESH, navmesh_op::FIND, [1.0, 1.0, 19.0, 11.0], floor.clone()),
        request(8, service::NAVMESH, navmesh_op::FIND, [2.0, 9.0, 13.0, 3.0], floor.clone()),
        request(9, service::NAVMESH, navmesh_op::TRIANGLES, [0.0; 4], floor),
    ]
}

/// Asks `services` the fixed requests; Ok with what they answered when the
/// answers' digest is [`EXPECTED`], an error saying what differs otherwise.
pub fn check<Input>(services: &mut Services<Input>) -> Result<String, String> {
    let requests = requests();
    services.answer(&requests);
    let events = services.take_answers();
    let answered = events.iter().filter(|event| matches!(event, GameEvent::ServiceAnswer { .. })).count();
    let digest = digest(&events);
    if digest != EXPECTED {
        return Err(format!(
            "simulation services answered {answered} of {} requests in {} events with digest {digest:#018x}, not {EXPECTED:#018x}",
            requests.len(),
            events.len()
        ));
    }
    Ok(format!("simulation services: {answered} answers in {} events, digest {digest:#018x}", events.len()))
}

#[cfg(test)]
mod tests {
    use pen_protocol::service_status;
    use pen_runtime::services::Answer;

    use super::*;

    fn ask(service: &mut dyn Service<()>, operation: u32, numbers: [f32; 4], values: &[f32]) -> Answer {
        service.call(&Request { id: 1, operation, numbers, values }).unwrap()
    }

    /// gdev's tests/simulation.blot makes these requests through its Blot
    /// binding (src/simulation.blot), checking each lane it sends, and
    /// reads these answers back into records.
    #[test]
    fn the_answers_gdevs_blot_test_reads() {
        // A ball resting on a slab, one step.
        let slab =
            [0.0, 1.0, 5.0, 0.5, 5.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0];
        let ball =
            [1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0];
        let answer = ask(&mut crate::Physics, physics_op::STEP, [1.0, 1.0 / 60.0, -9.81, 0.0], &[slab, ball].concat());
        let mut values = vec![0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        values.extend([0.0, 0.499_982_48, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        values.extend([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(answer, Answer { values, ..Answer::ok(1.0) });

        // A box at x 5 and a ball at x 10: a ray along +x, a ball between
        // them touching both, and the nearest to a point past the ball.
        let colliders = [
            [1.0, 1.0, 1.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
        .concat();
        let ray = [colliders.clone(), vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]].concat();
        let answer = ask(&mut crate::Spatial, spatial_op::RAYCAST, [2.0, 100.0, 1.0, 0.0], &ray);
        assert_eq!(answer, Answer { values: vec![1.0, 4.0, -1.0, 0.0, 0.0], ..Answer::ok(1.0) });
        let probe = [colliders.clone(), vec![0.0, 2.6, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]].concat();
        let answer = ask(&mut crate::Spatial, spatial_op::OVERLAP, [2.0, 0.0, 0.0, 0.0], &probe);
        assert_eq!(answer, Answer { values: vec![0.0, 1.0], ..Answer::ok(2.0) });
        let point = [colliders, vec![12.0, 0.0, 0.0]].concat();
        let answer = ask(&mut crate::Spatial, spatial_op::NEAREST, [2.0, 10.0, 0.0, 0.0], &point);
        assert_eq!(answer, Answer { values: vec![2.0, 1.0, 11.0, 0.0, 0.0], ..Answer::ok(1.0) });

        // A 10 by 10 floor with a block in its middle: to a goal inside the
        // block, which moves to the block's edge.
        let floor = [
            vec![0.0, 4.0, 0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
            vec![1.0, 4.0, 3.0, 3.0, 7.0, 3.0, 7.0, 7.0, 3.0, 7.0],
        ]
        .concat();
        let answer = ask(&mut crate::Navmesh, navmesh_op::FIND, [1.0, 5.0, 4.0, 5.0], &[vec![0.0], floor].concat());
        assert_eq!(answer, Answer { values: vec![1.0, 5.0, 3.0, 5.0], ..Answer::ok(2.0) });

        // Five octaves of hills, seed 7, at one point.
        let hills = [0.1, 8.0, 2.0, 0.5, 0.0, -0.5, 6.0];
        let answer = ask(&mut crate::Noise, noise_op::SAMPLE, [0.0, 0.0, 7.0, 5.0], &hills);
        assert_eq!(answer, Answer { values: vec![-2.375_610_6], ..Answer::ok(1.0) });
    }

    /// A game's mistakes reach the host as requests: the fixed ones cut
    /// short, and with odd or extreme numbers in their lanes, are answered
    /// (most `failed`), never a panic, which would end the host.
    #[test]
    fn broken_requests_are_answered_not_panicked_on() {
        let mut services: Services = Services::new();
        crate::register(&mut services);
        let odd = [0.0, -1.0, 0.5, 3.0, 7.0, 1e6, -1e6, 1e30, -1e30, 1e-30, 65536.0];
        let mut asked = 0;
        for mut request in requests() {
            // One step is enough to reach every body's handling.
            if request.service == service::PHYSICS {
                request.a = 1.0;
            }
            let length = request.values.len();
            let mut broken = Vec::new();
            for cut in [0, 1, 11, length / 2, length.saturating_sub(1)] {
                broken.push(ServiceRequest { values: request.values[..cut.min(length)].to_vec(), ..request.clone() });
            }
            for lane in (0..length).step_by(5) {
                for value in odd {
                    let mut values = request.values.clone();
                    values[lane] = value;
                    broken.push(ServiceRequest { values, ..request.clone() });
                }
            }
            for value in odd {
                broken.extend([
                    ServiceRequest { a: value, ..request.clone() },
                    ServiceRequest { b: value, ..request.clone() },
                    ServiceRequest { c: value, ..request.clone() },
                ]);
            }
            for request in &mut broken {
                request.value_count = u32::try_from(request.values.len()).unwrap();
            }
            asked += broken.len();
            services.answer(&broken);
        }
        let answers =
            services.take_answers().iter().filter(|event| matches!(event, GameEvent::ServiceAnswer { .. })).count();
        assert_eq!(answers, asked);
    }

    /// The requests as a game's render packet carries them: the header, then
    /// each request's lanes; the decoder must take every service code.
    #[test]
    fn every_fixed_request_crosses_the_render_packet() {
        use pen_protocol::boundary::{PACKET_HEADER_FLOATS, PACKET_MAGIC, decode_frame, lanes};

        let requests = requests();
        let mut packet = vec![0.0; PACKET_HEADER_FLOATS];
        packet[lanes::header::MAGIC] = PACKET_MAGIC;
        packet[lanes::header::SCHEMA] = 1.0;
        packet[lanes::header::SERVICE_COUNT] = requests.len() as f32;
        for asked in &requests {
            packet.extend([asked.request, asked.service, asked.operation].map(|lane| lane as f32));
            packet.extend([asked.a, asked.b, asked.c, asked.d, asked.value_count as f32]);
            packet.extend(&asked.values);
        }
        packet[lanes::header::SERVICE_LANES] = (packet.len() - PACKET_HEADER_FLOATS) as f32;
        let catalog = pen_protocol::Catalog { meshes: vec![], materials: vec![], sounds: vec![] };
        let decoded = decode_frame(&packet, 1, &catalog).unwrap();
        assert_eq!(decoded.render.services, requests);
    }

    #[test]
    fn every_fixed_request_is_answered_ok_with_the_expected_bits() {
        let mut services: Services = Services::new();
        crate::register(&mut services);
        services.answer(&requests());
        let events = services.take_answers();
        let statuses: Vec<(u32, u32, f32)> = events
            .iter()
            .filter_map(|event| match event {
                GameEvent::ServiceAnswer { request, status, value, .. } => Some((*request, *status, *value)),
                _ => None,
            })
            .collect();
        println!("{statuses:?} digest {:#018x}", digest(&events));
        assert!(statuses.iter().all(|(_, status, _)| *status == service_status::OK), "{statuses:?}");
        assert_eq!(statuses.len(), 9);
        let mut again: Services = Services::new();
        crate::register(&mut again);
        assert_eq!(
            check(&mut again),
            Ok(format!("simulation services: 9 answers in {} events, digest {EXPECTED:#018x}", events.len()))
        );
    }
}
