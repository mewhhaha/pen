//! What crosses between the host and a game's Blot program: frames in,
//! render packets out, generated from protocol/schema.json into packets.rs.
pub mod boundary;
pub mod diagnostic;
// Generated: every cast in it narrows a lane its decoder has checked is a
// whole number in range, or an f64 clock to its f32 lane. `pen codegen --check`
// compares it byte for byte, so rustfmt leaves it as generated.
#[rustfmt::skip]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "lanes checked before narrowing")]
mod packets;
pub mod protocol;

pub use protocol::*;
