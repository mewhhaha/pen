//! Time travel and replays. The game is deterministic given its input: the
//! same moment (state and edit history) and the same frames of input give
//! the same packets, bit for bit. So a history needs only each frame's input
//! (`encode_frame`'s payload: elapsed time, viewport and events), the states
//! the frame loaded from files, and a moment every `MOMENT_INTERVAL` frames.
//! Rewinding restores the nearest earlier moment and runs the recorded
//! frames again, unseen, up to the frame asked for.
//!
//! A restore starts a new program from the state, as a quicksave load does.
//! gdev saves everything later frames depend on (menus, focus, easing,
//! service requests in flight), so the frames a rewind runs again give the
//! packets they gave the first time; native pen's contract tests check each
//! one. A game that kept something outside its state would start it over at
//! a restore, so frames after one are marked `restored` and replays restore
//! there too: a replay gives exactly the packets the recorded run did even
//! then.
//!
//! Positions count frames: position N is the moment before frame N runs.
//! The cursor is the next frame to run. Frames at and after it were
//! recorded and wait to run again: a replay plays them on, and stepping
//! while paused in the past runs them one at a time.
use std::collections::{BTreeMap, VecDeque};

use anyhow::{Context, Result, ensure};
use pen_protocol::RenderFrame;

use crate::clock::STEP_SECONDS;
use crate::game::{GameRuntime, Host, Moment, Quiet};
use crate::program::Backend;
use crate::snapshot::SavedState;

/// Frames between the moments a history keeps: a rewind runs at most this
/// many frames again.
pub const MOMENT_INTERVAL: u64 = 30;
/// How much game time a rolling history keeps.
pub const KEEP_SECONDS: f64 = 10.0;
/// How much game time the replay shortcut (Ctrl+Shift+R) plays again.
pub const REPLAY_SECONDS: f64 = 5.0;
/// Frames kept behind the cursor at most, however short they were.
const MAX_KEPT_FRAMES: u64 = 20_000;

/// A state a frame loaded from a file, which a replay cannot read again
/// (the file may have changed since).
#[derive(Clone, Debug, PartialEq)]
pub enum Load {
    /// A quicksave (`load`, F9 or `load=FILE`).
    Save(SavedState),
    /// The editor's scene (Ctrl+R).
    Scene(SavedState),
}

/// One frame as the game received it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Recorded {
    /// The `encode_frame` payload.
    pub input: Vec<f32>,
    /// The states it loaded afterwards, in order.
    pub loads: Vec<Load>,
    /// The game was restored from its own state just before it (a rewind
    /// or replay started there); a replay restores there too.
    pub restored: bool,
}

impl Recorded {
    /// The game seconds the frame advanced.
    pub fn elapsed(&self) -> f64 {
        self.input.first().map_or(0.0, |elapsed| f64::from(*elapsed))
    }
}

/// A run's frames from a moment on: what native pen's `--record-input`
/// writes and `--replay` plays.
#[derive(Clone, Debug, PartialEq)]
pub struct Recording {
    pub start: Moment,
    pub frames: Vec<Recorded>,
}

/// Loads a recorded state again.
pub async fn apply<B: Backend>(runtime: &mut GameRuntime<B>, host: &mut dyn Host, load: &Load) -> Result<()> {
    match load {
        Load::Save(saved) => runtime.load(host, saved).await,
        Load::Scene(scene) => runtime.load_scene(host, scene).await,
    }
}

/// Runs a recorded frame without showing it; returns the frame it would
/// have shown.
async fn run_unseen<B: Backend>(
    runtime: &mut GameRuntime<B>,
    host: &dyn Host,
    recorded: &Recorded,
) -> Result<RenderFrame> {
    let mut quiet = Quiet(host);
    runtime.send(&recorded.input)?;
    let rendered = runtime.receive(&quiet).await?;
    runtime.follow(&mut quiet, &rendered).await?;
    for load in &recorded.loads {
        apply(runtime, &mut quiet, load).await?;
    }
    Ok(runtime.take_drawn().unwrap_or(rendered))
}

/// Told of each frame a rewind runs unseen: its position and input, and
/// the runtime it ran on (for its packet's hash).
pub type Unseen<'a, B> = dyn FnMut(u64, &Recorded, &GameRuntime<B>) -> Result<()> + 'a;

struct Kept {
    /// Which build of the game ran up to it (hot reloads count up).
    build: u64,
    moment: Moment,
}

pub struct Timeline {
    /// Frames from position `base` on.
    frames: VecDeque<Recorded>,
    base: u64,
    moments: BTreeMap<u64, Kept>,
    cursor: u64,
    build: u64,
    /// Keep frames and moments as frames run.
    history: bool,
    /// How much game time to keep behind the cursor; None keeps all.
    keep: Option<f64>,
    /// The game was restored at the cursor, which is the head: the next
    /// frame recorded follows a restore.
    restored_next: bool,
    /// Where the running program was last started from a restore.
    fresh: Option<u64>,
}

impl Default for Timeline {
    fn default() -> Timeline {
        Timeline {
            frames: VecDeque::new(),
            base: 0,
            moments: BTreeMap::new(),
            cursor: 0,
            build: 0,
            history: false,
            keep: Some(KEEP_SECONDS),
            restored_next: false,
            fresh: None,
        }
    }
}

impl Timeline {
    /// Keep a history to travel in: the last `seconds` of game time, or
    /// everything.
    pub fn keep_history(&mut self, seconds: Option<f64>) {
        self.history = true;
        self.keep = seconds;
    }

    pub fn history(&self) -> bool {
        self.history
    }

    /// The next frame to run; frames before it have run.
    pub fn position(&self) -> u64 {
        self.cursor
    }

    /// The end of what was recorded.
    pub fn head(&self) -> u64 {
        self.base + self.frames.len() as u64
    }

    /// The earliest position a rewind can reach.
    pub fn earliest(&self) -> Option<u64> {
        self.moments.keys().next().copied()
    }

    /// The game time from the earliest moment to the cursor.
    pub fn seconds(&self) -> f64 {
        let from = self.earliest().unwrap_or(self.cursor).max(self.base);
        (from..self.cursor).map(|position| self.frames[self.index(position)].elapsed()).sum()
    }

    /// Where in `frames` the frame at `position` (at or after `base`) is.
    fn index(&self, position: u64) -> usize {
        usize::try_from(position - self.base).expect("frames held fit in memory")
    }

    /// A frame kept in the history.
    pub fn recorded(&self, position: u64) -> Option<&Recorded> {
        let index = position.checked_sub(self.base).and_then(|index| usize::try_from(index).ok())?;
        self.frames.get(index)
    }

    /// Where the moments kept are, earliest first.
    pub fn moments(&self) -> impl DoubleEndedIterator<Item = u64> + '_ {
        self.moments.keys().copied()
    }

    /// The history as a recording: the earliest moment and every frame
    /// recorded from it on.
    pub fn recording(&self) -> Option<Recording> {
        let (&at, kept) = self.moments.iter().next()?;
        let frames = self.frames.iter().skip(self.index(at.max(self.base))).cloned().collect();
        Some(Recording { start: kept.moment.clone(), frames })
    }

    /// The frame `seconds` of game time before `frame`, as far back as the
    /// history reaches (frames it does not hold count a step's time).
    pub fn frames_before(&self, frame: u64, seconds: f64) -> u64 {
        let earliest = self.earliest().unwrap_or(frame).min(frame);
        let (mut position, mut time) = (frame, 0.0);
        while position > earliest && time < seconds {
            position -= 1;
            time += self.recorded(position).map_or(STEP_SECONDS, Recorded::elapsed);
        }
        position
    }

    /// The recorded frame at the cursor, when the cursor is in the past.
    pub fn next(&self) -> Option<&Recorded> {
        self.recorded(self.cursor)
    }

    /// Whether the frame at the cursor follows a restore.
    pub fn restored_here(&self) -> bool {
        self.next().map_or(self.restored_next, |recorded| recorded.restored)
    }

    /// Whether to keep a moment at the cursor before the next frame runs.
    pub fn wants_moment(&self) -> bool {
        self.history
            && match self.moments.get(&self.cursor) {
                Some(kept) => kept.build != self.build,
                None => self.moments.is_empty() || self.cursor.is_multiple_of(MOMENT_INTERVAL),
            }
    }

    pub fn keep_moment(&mut self, moment: Moment) {
        self.moments.insert(self.cursor, Kept { build: self.build, moment });
        self.trim();
    }

    /// Before the frame at the cursor runs: keeps a moment when one is due,
    /// and restores the game from its own state where the recorded run was
    /// restored, so the frames after run as they did then.
    pub async fn prepare<B: Backend>(&mut self, runtime: &mut GameRuntime<B>, host: &dyn Host) -> Result<()> {
        if self.wants_moment() {
            let moment = runtime.moment().await?;
            self.keep_moment(moment);
        }
        if self.restored_here() && self.fresh != Some(self.cursor) {
            runtime.reroot(host).await?;
            self.fresh = Some(self.cursor);
        }
        Ok(())
    }

    /// The frame at the cursor ran: a recorded one again, or a new one.
    pub fn ran(&mut self, mut recorded: Recorded) {
        if self.cursor == self.head() {
            recorded.restored |= std::mem::take(&mut self.restored_next);
            if self.history {
                self.frames.push_back(recorded);
            } else {
                // Nothing to travel in (a replay's frames are done with).
                self.frames.clear();
                self.moments.clear();
                self.base = self.cursor + 1;
            }
        }
        self.cursor += 1;
        self.fresh = None;
    }

    /// The recorded future is dropped: play continues live from the cursor.
    pub fn go_live(&mut self) {
        self.restored_next = self.restored_here();
        self.frames.truncate(self.index(self.cursor));
        self.moments.retain(|position, _| *position <= self.cursor);
    }

    /// A new build of the game runs from here on.
    pub fn rebuilt(&mut self) {
        self.build += 1;
    }

    /// Plays a recording: its frames wait to run from its start, which the
    /// caller restores.
    pub fn load(&mut self, recording: Recording) {
        self.frames = recording.frames.into();
        self.base = 0;
        self.cursor = 0;
        self.restored_next = false;
        self.fresh = Some(0);
        self.moments.clear();
        self.moments.insert(0, Kept { build: self.build, moment: recording.start });
    }

    /// The position `seconds` of game time before the end of the record.
    pub fn seconds_back(&self, seconds: f64) -> u64 {
        let mut time = 0.0;
        let mut position = self.head();
        while position > self.base && time < seconds {
            position -= 1;
            time += self.frames[self.index(position)].elapsed();
        }
        position
    }

    /// Drops what is older than the history keeps: moments before the
    /// newest one at least `keep` seconds back, and their frames.
    fn trim(&mut self) {
        let Some(keep) = self.keep else { return };
        let mut time = 0.0;
        let mut position = self.cursor;
        let mut cut = None;
        for &at in self.moments.range(..=self.cursor).rev().map(|(position, _)| position) {
            while position > at {
                position -= 1;
                time += self.frames[self.index(position)].elapsed();
            }
            if time >= keep || self.cursor - at >= MAX_KEPT_FRAMES {
                cut = Some(at);
                break;
            }
        }
        if let Some(at) = cut {
            self.moments = self.moments.split_off(&at);
            let end = self.index(at);
            self.frames.drain(..end);
            self.base = at;
        }
    }

    /// Moves the cursor to the moment kept at `at` and returns it. Moments
    /// after it that an older build reached no longer follow from it (the
    /// frames will run on the current build), so they go.
    fn seek(&mut self, at: u64) -> Moment {
        let kept = &self.moments[&at];
        let moment = kept.moment.clone();
        if kept.build != self.build {
            self.moments.retain(|position, _| *position <= at);
        }
        self.cursor = at;
        moment
    }

    /// The game was restored at the cursor: the frame there follows a restore.
    fn restored(&mut self) {
        let cursor = self.cursor;
        let index = self.index(cursor);
        match self.frames.get_mut(index) {
            Some(recorded) => recorded.restored = true,
            None => self.restored_next = true,
        }
        self.fresh = Some(cursor);
    }

    /// Goes back (or forward, within the record) to `target`: restores the
    /// nearest moment before it and runs the recorded frames from there
    /// unseen. Returns the frame `target - 1` showed, and where the
    /// restored moment was.
    pub async fn rewind<B: Backend>(
        &mut self,
        runtime: &mut GameRuntime<B>,
        host: &dyn Host,
        target: u64,
        unseen: &mut Unseen<'_, B>,
    ) -> Result<(RenderFrame, u64)> {
        let earliest = self.earliest().context("there is no history to travel in")?;
        ensure!(target >= earliest, "the history starts at frame {earliest}");
        let target = target.min(self.head());
        // A moment before the target, so the frame shown is the one the
        // target followed; the target itself only when it is the earliest.
        let at = self.moments.range(..target).next_back().map_or(earliest, |(position, _)| *position);
        let moment = self.seek(at);
        let mut shown = runtime.restore(&mut Quiet(host), &moment).await?;
        self.restored();
        while self.cursor < target {
            self.prepare(runtime, host).await?;
            let recorded = self.frames[self.index(self.cursor)].clone();
            shown = run_unseen(runtime, host, &recorded).await?;
            unseen(self.cursor, &recorded, runtime)?;
            self.ran(recorded);
        }
        Ok((shown, at))
    }

    /// Restores the latest moment at or before `from`, for the recorded
    /// frames after it to play again (on whatever build runs now). Returns
    /// the restored program's first frame and where the replay starts.
    pub async fn restart<B: Backend>(
        &mut self,
        runtime: &mut GameRuntime<B>,
        host: &dyn Host,
        from: u64,
    ) -> Result<(RenderFrame, u64)> {
        let earliest = self.earliest().context("there is no history to replay")?;
        let at = self.moments.range(..=from).next_back().map_or(earliest, |(position, _)| *position);
        let moment = self.seek(at);
        let shown = runtime.restore(&mut Quiet(host), &moment).await?;
        self.restored();
        Ok((shown, at))
    }
}

#[cfg(test)]
mod tests;
