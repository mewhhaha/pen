//! What crosses between gdev's host and its Blot program: frames in, render
//! packets out (gdev's `protocol.ts` and `boundary.ts`).
pub mod boundary;
pub mod protocol;

pub use protocol::*;
