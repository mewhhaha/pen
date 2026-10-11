use anyhow::{Result, ensure};
use pen_protocol::boundary::validate_schema;
use web_time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Operation { Stop = 0, Create = 1, Frame = 2, Snapshot = 3, Restore = 4, Scene = 5, LoadScene = 6 }

pub fn handshake(description: &[f32]) -> Result<(u32, bool)> {
    ensure!(description.len() == 4 && description[0] == 4670544.0 && description[1] == 1.0
        && (description[3] == 0.0 || description[3] == 1.0), "invalid game program handshake");
    Ok((validate_schema(description[2])?, description[3] == 1.0))
}

pub fn request(commit: bool, operation: Operation, payload: &[f32]) -> Vec<f32> {
    let mut packet = Vec::with_capacity(payload.len() + 2);
    if commit { packet.push(1.0); } else { packet.push(0.0); }
    packet.push(operation as u8 as f32);
    packet.extend_from_slice(payload);
    packet
}

pub fn response(operation: Operation, mut packet: Vec<f32>) -> Result<Vec<f32>> {
    ensure!(packet != [-1.0], "game state must be a bounded array of finite float32 values");
    ensure!(packet.first() == Some(&(operation as u8 as f32)), "game response does not match its request");
    packet.remove(0);
    Ok(packet)
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
    async fn response(&mut self) -> Result<Vec<f32>>;
    async fn request(&mut self, operation: Operation, payload: &[f32]) -> Result<Vec<f32>> {
        self.send(operation, payload)?;
        self.response().await
    }
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
        assert_eq!(handshake(&[4670544.0, 1.0, 5.0, 1.0]).unwrap(), (5, true));
        assert!(handshake(&[4670544.0, 2.0, 5.0, 1.0]).is_err());
        assert!(handshake(&[4670544.0, 1.0, 5.5, 1.0]).is_err());
        assert_eq!(request(false, Operation::Frame, &[8.0]), [0.0, 2.0, 8.0]);
        assert_eq!(response(Operation::Frame, vec![2.0, 8.0]).unwrap(), [8.0]);
        assert!(response(Operation::Frame, vec![3.0]).is_err());
        assert!(response(Operation::Restore, vec![-1.0]).is_err());
    }
}
