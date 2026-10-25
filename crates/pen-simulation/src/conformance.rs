//! The same answers on every host: fixed requests to each simulation
//! service, and the digest of what a game would get back (the exact
//! ServiceAnswer and ServiceData events, bit for bit) that every host must
//! reproduce. Native tests check it here and through native pen's own
//! registry; the browser host checks it in wasm32
//! (pen-web's `qualify_services`, which crates/pen-web/tests/services.mjs
//! runs under node and `just conformance` in headless Chromium).
use pen_protocol::{GameEvent, ServiceRequest, noise_op, service};
use pen_runtime::services::Services;

/// The digest every host's answers to [`requests`] must have.
pub const EXPECTED: u64 = 0x25b2_9823_ddf7_d625;

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

/// Fixed noise requests, without bundled physics or navigation.
pub fn requests() -> Vec<ServiceRequest> {
    let hills = vec![0.08, 2.5, 2.0, 0.5, 0.0, -8.0, -8.0, 1.0];

    vec![
        request(1, service::NOISE, noise_op::HEIGHTFIELD, [17.0, 17.0, 2024.0, 5.0], hills),
        request(
            2,
            service::NOISE,
            noise_op::SAMPLE,
            [0.0, 0.0, 99.0, 7.0],
            vec![0.03, 10.0, 2.1, 0.45, 1.0, 0.0, 0.0, -3.25, 7.5, 12345.678, -98765.4, 0.5, 0.25, -1000.0, 1000.0],
        ),
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
    use pen_runtime::services::{Answer, Request, Service};

    use super::*;

    fn ask(service: &mut dyn Service<()>, operation: u32, numbers: [f32; 4], values: &[f32]) -> Answer {
        service.call(&Request { id: 1, operation, numbers, values }).unwrap()
    }

    /// gdev's tests/simulation.blot makes these requests through its Blot
    /// binding (src/simulation.blot), checking each lane it sends, and
    /// reads these answers back into records.
    #[test]
    fn the_answers_gdevs_blot_test_reads() {
        // Five octaves of hills, seed 7, at one point.
        let hills = [0.1, 8.0, 2.0, 0.5, 0.0, -0.5, 6.0];
        let answer = ask(&mut crate::Noise, noise_op::SAMPLE, [0.0, 0.0, 7.0, 5.0], &hills);
        assert_eq!(answer, Answer { values: vec![-2.375_610_6], ..Answer::ok(1.0) });
    }

    #[test]
    fn removed_simulation_services_report_unknown() {
        let mut services: Services = Services::new();
        crate::register(&mut services);
        services.answer(&[
            request(1, service::PHYSICS, 0, [0.0; 4], vec![]),
            request(2, service::SPATIAL, 0, [0.0; 4], vec![]),
            request(3, service::NAVMESH, 0, [0.0; 4], vec![]),
            request(4, service::PATHFINDING, 0, [0.0; 4], vec![]),
        ]);
        let events = services.take_answers();
        assert_eq!(events.len(), 4);
        assert!(events.iter().all(|event| matches!(event,
            GameEvent::ServiceAnswer { status, length: 0, .. } if *status == service_status::UNKNOWN)));
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
        for request in requests() {
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
        assert_eq!(statuses.len(), 2);
        let mut again: Services = Services::new();
        crate::register(&mut again);
        assert_eq!(
            check(&mut again),
            Ok(format!("simulation services: 2 answers in {} events, digest {EXPECTED:#018x}", events.len()))
        );
    }
}
