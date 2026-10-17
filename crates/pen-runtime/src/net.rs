//! Net sessions: several peers play one game, each its own player, and
//! exchange nothing but their players' input. The game is a pure function
//! of its state and input (docs/determinism.md), so peers that start from
//! the same moment and give every frame the same input give the same
//! frames, bit for bit; nobody sends state, and the game needs no netcode
//! of its own beyond declaring how many players it takes. A frame's input
//! lists every player in order, each behind a `Player` event, with the
//! events its peer gave it; the session host's (player 0's) own events
//! (assets, preferences, tunables, service answers) travel as player 0's,
//! and other peers keep theirs, so every peer's game hears the same.
//!
//! Each player's input plays `delay` frames after it was given, which
//! covers that much latency for free. Past it a peer does not wait: it
//! predicts that the players it has not heard from yet did nothing new,
//! keeps a moment before each frame that ran on a prediction (the
//! timeline's moments, as time travel keeps them), and when a prediction
//! proves wrong, gives those frames their real input and runs them again
//! from the moment, unseen: a rollback, through `Timeline::rewind`. A peer
//! runs at most `window` frames past the last frame whose every input is
//! known, which bounds a rollback. Every `CHECK_INTERVAL` frames the peers
//! compare the hash of a frame whose input was all known; a difference
//! means a game kept something outside its state ([`play::Desync`]).
//!
//! `wire` is what peers send: hellos that agree on the session's terms
//! (the host's: its delay, window, frame rate and viewport) and the game's
//! build, input messages that repeat every frame not yet acknowledged (so
//! a lost datagram costs nothing), and the host's starting moment, chunk
//! by chunk, for a peer that started from another. A [`play::Link`]
//! carries them: UDP natively, a lossy queue in the tests.
//!
//! `stream` is spectating: a session's final frames streamed to viewers,
//! who play them as a replay does.
pub mod inputs;
pub mod play;
pub mod stream;
pub mod wire;

pub use play::{Desync, Link, Netplay, Settings, Stats};

#[cfg(test)]
mod tests;
