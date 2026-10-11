//! The packets' meaning, as gdev's `protocol.ts` describes it, generated from
//! gdev's protocol/schema.json into packets.rs. Asset references are indices into
//! the [`Catalog`]; `boundary` has checked each one is in range.
pub use crate::packets::types::*;
