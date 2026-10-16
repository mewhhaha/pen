use std::ops::Deref;

use anyhow::{Result, ensure};
use pen_protocol::boundary::validate_schema;
use web_time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Operation {
    Stop = 0,
    Create = 1,
    Frame = 2,
    Snapshot = 3,
    Restore = 4,
    Scene = 5,
    LoadScene = 6,
}

/// What a program says first: this magic number (0x474450), its protocol
/// version, its state schema and whether it keeps scenes.
pub const HANDSHAKE_MAGIC: f32 = 4670544.0;
pub const PROGRAM_VERSION: f32 = 1.0;

/// What a program answers in place of a state it refused.
const REFUSED_STATE: [f32; 1] = [-1.0];

pub fn handshake(description: &[f32]) -> Result<(u32, bool)> {
    ensure!(
        description.len() == 4
            && description[0] == HANDSHAKE_MAGIC
            && description[1] == PROGRAM_VERSION
            && (description[3] == 0.0 || description[3] == 1.0),
        "invalid game program handshake"
    );
    Ok((validate_schema(description[2])?, description[3] == 1.0))
}

/// A request: whether the program keeps what it did for the last one (the
/// host accepted it), the operation, then the payload. Written over
/// `packet`, reusing its allocation.
pub fn write_request(packet: &mut Vec<f32>, commit: bool, operation: Operation, payload: &[f32]) {
    packet.clear();
    packet.reserve(payload.len() + 2);
    packet.push(if commit { 1.0 } else { 0.0 });
    packet.push(operation as u8 as f32);
    packet.extend_from_slice(payload);
}

pub fn request(commit: bool, operation: Operation, payload: &[f32]) -> Vec<f32> {
    let mut packet = Vec::new();
    write_request(&mut packet, commit, operation, payload);
    packet
}

/// A program's answer to a request: the lanes after the operation it echoes.
/// They stay where they arrived rather than move down over that lane.
#[derive(Debug)]
pub struct Response(Vec<f32>);

impl Response {
    /// The lanes, in a vector of their own.
    pub fn into_lanes(mut self) -> Vec<f32> {
        self.0.remove(0);
        self.0
    }

    /// The buffer they arrived in, for a later request to reuse
    /// ([`Program::recycle`]).
    pub fn into_buffer(self) -> Vec<f32> {
        self.0
    }
}

impl Deref for Response {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.0[1..]
    }
}

/// Checks `packet` answers `operation`.
pub fn response(operation: Operation, packet: Vec<f32>) -> Result<Response> {
    ensure!(packet != REFUSED_STATE, "game state must be a bounded array of finite float32 values");
    ensure!(packet.first() == Some(&(operation as u8 as f32)), "game response does not match its request");
    Ok(Response(packet))
}

/// The program stopped answering: it ran past its deadline and was stopped,
/// or its thread never replied. Its frame is rejected like an invalid one,
/// but only restoring a moment runs the game again.
#[derive(Clone, Copy, Debug)]
pub struct Unresponsive;

impl std::fmt::Display for Unresponsive {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("the game stopped answering")
    }
}

#[allow(async_fn_in_trait)]
pub trait Backend: Clone {
    type Module: Clone;
    type Program: Program;
    async fn start(&self, module: &Self::Module) -> Result<Self::Program>;
}

#[allow(async_fn_in_trait)]
pub trait Program {
    fn schema(&self) -> u32;
    fn scenes(&self) -> bool;
    fn send(&mut self, operation: Operation, payload: &[f32]) -> Result<()>;
    /// The answer to the request in flight. An error with [`Unresponsive`]
    /// in it means the program is gone.
    async fn response(&mut self) -> Result<Response>;
    async fn request(&mut self, operation: Operation, payload: &[f32]) -> Result<Vec<f32>> {
        self.send(operation, payload)?;
        Ok(self.response().await?.into_lanes())
    }
    /// Takes back a response's buffer, for the next request to reuse.
    fn recycle(&mut self, _buffer: Vec<f32>) {}
    fn accept(&mut self);
    fn answered(&self) -> Option<Instant>;
    fn memory_bytes(&self) -> usize;
    async fn close(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn program_boundary_rejects_invalid_handshake_and_responses() {
        assert_eq!(handshake(&[HANDSHAKE_MAGIC, 1.0, 5.0, 1.0]).unwrap(), (5, true));
        assert!(handshake(&[HANDSHAKE_MAGIC, 2.0, 5.0, 1.0]).is_err());
        assert!(handshake(&[HANDSHAKE_MAGIC, 1.0, 5.5, 1.0]).is_err());
        assert_eq!(request(false, Operation::Frame, &[8.0]), [0.0, 2.0, 8.0]);
        let mut packet = vec![9.0; 10];
        write_request(&mut packet, true, Operation::Snapshot, &[]);
        assert_eq!(packet, [1.0, 3.0]);
        let answer = response(Operation::Frame, vec![2.0, 8.0]).unwrap();
        assert_eq!(*answer, [8.0]);
        assert_eq!(answer.into_buffer(), [2.0, 8.0]);
        assert_eq!(response(Operation::Frame, vec![2.0, 8.0]).unwrap().into_lanes(), [8.0]);
        assert!(response(Operation::Frame, vec![3.0]).is_err());
        assert!(response(Operation::Restore, vec![-1.0]).is_err());
    }
}
