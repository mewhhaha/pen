//! What crosses between gdev's host and its Blot program: frames in, render
//! packets out (gdev's `protocol.ts` and `boundary.ts`), generated from gdev's
//! protocol/schema.ts into packets.rs.
pub mod boundary;
mod packets;
pub mod protocol;

pub use protocol::*;
