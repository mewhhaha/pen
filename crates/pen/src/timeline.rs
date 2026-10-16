//! Time travel and replays: pen-runtime's timeline (`pen_runtime::timeline`,
//! which the browser runs too), its methods that run the game waited for
//! here as the native runtime's are (`game`). The game is deterministic
//! given its input, so a history keeps each frame's input, the states it
//! loaded from files and a moment every `MOMENT_INTERVAL` frames; a rewind
//! restores the nearest earlier moment and runs the recorded frames again,
//! unseen. The contract tests below check every frame a rewind runs again
//! gives the packets it gave the first time.
use std::ops::{Deref, DerefMut};

use anyhow::Result;
use pen_protocol::RenderFrame;

#[cfg(test)]
use crate::game::Moment;
use crate::game::{GameRuntime, Host};
use crate::program::NativeBackend;
#[cfg(test)]
use crate::snapshot::SavedState;
#[cfg(test)]
use pen_runtime::timeline::MOMENT_INTERVAL;
pub use pen_runtime::timeline::{KEEP_SECONDS, Load, REPLAY_SECONDS, Recorded, Recording};

/// The shared runtime a `GameRuntime` runs: what a rewind tells of each
/// frame it runs unseen.
pub type Runtime = pen_runtime::game::GameRuntime<NativeBackend>;

/// Told of each frame a rewind runs unseen: its position and input, and
/// the runtime it ran on (for its packet's hash).
pub type Unseen<'a> = pen_runtime::timeline::Unseen<'a, NativeBackend>;

/// Loads a recorded state again.
pub fn apply(runtime: &mut GameRuntime, host: &mut dyn Host, load: &Load) -> Result<()> {
    pollster::block_on(pen_runtime::timeline::apply(&mut **runtime, host, load))
}

#[derive(Default)]
pub struct Timeline(pen_runtime::timeline::Timeline);

impl Timeline {
    /// Before the frame at the cursor runs: keeps a moment when one is due,
    /// and restores the game from its own state where the recorded run was
    /// restored, so the frames after run as they did then.
    pub fn prepare(&mut self, runtime: &mut GameRuntime, host: &dyn Host) -> Result<()> {
        pollster::block_on(self.0.prepare(&mut **runtime, host))
    }

    /// Goes back (or forward, within the record) to `target`: restores the
    /// nearest moment before it and runs the recorded frames from there
    /// unseen. Returns the frame `target - 1` showed, and where the
    /// restored moment was.
    pub fn rewind(
        &mut self,
        runtime: &mut GameRuntime,
        host: &dyn Host,
        target: u64,
        unseen: &mut Unseen,
    ) -> Result<(RenderFrame, u64)> {
        pollster::block_on(self.0.rewind(&mut **runtime, host, target, unseen))
    }

    /// Restores the latest moment at or before `from`, for the recorded
    /// frames after it to play again (on whatever build runs now). Returns
    /// the restored program's first frame and where the replay starts.
    pub fn restart(&mut self, runtime: &mut GameRuntime, host: &dyn Host, from: u64) -> Result<(RenderFrame, u64)> {
        pollster::block_on(self.0.restart(&mut **runtime, host, from))
    }
}

/// The timeline's methods that do not run the game.
impl Deref for Timeline {
    type Target = pen_runtime::timeline::Timeline;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Timeline {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod contract_tests;
