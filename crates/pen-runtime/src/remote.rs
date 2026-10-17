//! The dev tools service across a connection: a client elsewhere (`pen
//! remote`, or a dev tools program on another machine) asks a running
//! game's host what its own dev tools would ask, and the host answers
//! through the same [`crate::devtools::Devtools`]. Each message is a
//! little-endian u32 count of float32 lanes, then the lanes, little-endian:
//! a request is its id, service, operation, `a` to `d`, its values' count
//! and its values; an answer is the request's id, its status, value, its
//! values' count and its values.
use anyhow::{Result, ensure};
use pen_protocol::ServiceRequest;
use pen_protocol::boundary::whole;

use crate::services::Answer;

/// The most lanes a message holds: an answer's values and its header.
pub const MESSAGE_LANES: usize = pen_protocol::boundary::MAX_SERVICE_DATA + 16;

fn message(lanes: &[f32]) -> Vec<u8> {
    let count = u32::try_from(lanes.len()).expect("a message's lanes fit a u32");
    let mut bytes = Vec::with_capacity(4 + lanes.len() * 4);
    bytes.extend_from_slice(&count.to_le_bytes());
    for lane in lanes {
        bytes.extend_from_slice(&lane.to_le_bytes());
    }
    bytes
}

/// A request as a message.
pub fn request_message(request: &ServiceRequest) -> Vec<u8> {
    let mut lanes = vec![
        request.request as f32,
        request.service as f32,
        request.operation as f32,
        request.a,
        request.b,
        request.c,
        request.d,
        request.values.len() as f32,
    ];
    lanes.extend(request.values.iter().take(pen_protocol::boundary::MAX_SERVICE_DATA));
    message(&lanes)
}

/// An answer to request `id` as a message.
pub fn answer_message(id: u32, answer: &Answer) -> Vec<u8> {
    let mut lanes = vec![id as f32, answer.status as f32, answer.value, answer.values.len() as f32];
    lanes.extend(answer.values.iter().take(pen_protocol::boundary::MAX_SERVICE_DATA));
    message(&lanes)
}

/// The values a message's header says follow at `at`, all of them there.
fn values(lanes: &[f32], at: usize) -> Result<Vec<f32>> {
    let count = whole(lanes[at - 1]).map(|count| count as usize);
    ensure!(count == Some(lanes.len() - at), "a message's values are not all there");
    ensure!(lanes[at..].iter().all(|lane| lane.is_finite()), "a message holds a value that is not a number");
    Ok(lanes[at..].to_vec())
}

/// A request from a message's lanes.
pub fn read_request(lanes: &[f32]) -> Result<ServiceRequest> {
    ensure!(lanes.len() >= 8, "a request needs its header");
    let number =
        |at: usize| whole(lanes[at]).ok_or_else(|| anyhow::anyhow!("a request's id, service and operation are whole"));
    ensure!(lanes[3..7].iter().all(|lane| lane.is_finite()), "a request's numbers are numbers");
    let values = values(lanes, 8)?;
    Ok(ServiceRequest {
        request: number(0)?,
        service: number(1)?,
        operation: number(2)?,
        a: lanes[3],
        b: lanes[4],
        c: lanes[5],
        d: lanes[6],
        value_count: u32::try_from(values.len())?,
        values,
    })
}

/// An answer, and the id of the request it answers, from a message's lanes.
pub fn read_answer(lanes: &[f32]) -> Result<(u32, Answer)> {
    ensure!(lanes.len() >= 4, "an answer needs its header");
    let id = whole(lanes[0]).ok_or_else(|| anyhow::anyhow!("an answer's id is whole"))?;
    let status = whole(lanes[1]).ok_or_else(|| anyhow::anyhow!("an answer's status is whole"))?;
    Ok((id, Answer { status, value: lanes[2], values: values(lanes, 4)? }))
}

/// Messages as they arrive, a few bytes at a time: what came, until a whole
/// message is there.
#[derive(Debug, Default)]
pub struct Messages {
    bytes: Vec<u8>,
}

impl Messages {
    pub fn arrived(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    /// The next whole message's lanes, if one is there; an error for one
    /// longer than any message.
    pub fn next_message(&mut self) -> Result<Option<Vec<f32>>> {
        let Some(header) = self.bytes.get(..4) else { return Ok(None) };
        let count = u32::from_le_bytes(header.try_into().unwrap()) as usize;
        ensure!(count <= MESSAGE_LANES, "a message of {count} lanes is past the protocol's limit");
        let Some(body) = self.bytes.get(4..4 + count * 4) else { return Ok(None) };
        let lanes = body.as_chunks::<4>().0.iter().map(|lane| f32::from_le_bytes(*lane)).collect();
        self.bytes.drain(..4 + count * 4);
        Ok(Some(lanes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::code_points;

    #[test]
    fn requests_and_answers_cross_as_messages() {
        let request = ServiceRequest {
            request: 7,
            service: 5,
            operation: 21,
            a: 100.0,
            b: 3.0,
            c: 3.0,
            d: -2.5,
            value_count: 4,
            values: code_points("fast"),
        };
        let answer = Answer { values: vec![1.0, 2.0], ..Answer::ok(4.0) };
        let mut messages = Messages::default();
        let bytes = [request_message(&request), answer_message(7, &answer)].concat();
        // A byte at a time: nothing until a message is whole.
        for byte in &bytes[..bytes.len() - 1] {
            messages.arrived(std::slice::from_ref(byte));
        }
        assert_eq!(read_request(&messages.next_message().unwrap().unwrap()).unwrap(), request);
        assert_eq!(messages.next_message().unwrap(), None);
        messages.arrived(&bytes[bytes.len() - 1..]);
        assert_eq!(read_answer(&messages.next_message().unwrap().unwrap()).unwrap(), (7, answer));
        // A message with values missing, or one too long, is refused.
        assert!(read_request(&[1.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0]).is_err());
        assert!(read_answer(&[1.0, 0.0]).is_err());
        let mut long = Messages::default();
        long.arrived(&u32::MAX.to_le_bytes());
        assert!(long.next_message().is_err());
    }
}
