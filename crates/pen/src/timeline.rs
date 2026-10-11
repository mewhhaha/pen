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
//! packets they gave the first time; the contract tests below check each
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

use crate::game::{GameRuntime, Host, Moment, Quiet};
use crate::snapshot::SavedState;

/// Frames between the moments a history keeps: a rewind runs at most this
/// many frames again.
pub const MOMENT_INTERVAL: u64 = 30;
/// How much game time a rolling history keeps.
pub const KEEP_SECONDS: f64 = 10.0;
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

/// A session's frames from a moment on: what `--record-input` writes and
/// `--replay` plays.
#[derive(Clone, Debug, PartialEq)]
pub struct Recording {
    pub start: Moment,
    pub frames: Vec<Recorded>,
}

/// Loads a recorded state again.
pub fn apply(runtime: &mut GameRuntime, host: &mut dyn Host, load: &Load) -> Result<()> {
    match load {
        Load::Save(saved) => runtime.load(host, saved),
        Load::Scene(scene) => runtime.load_scene(host, scene),
    }
}

/// Runs a recorded frame without showing it; returns the frame it would
/// have shown.
fn run_unseen(runtime: &mut GameRuntime, host: &dyn Host, recorded: &Recorded) -> Result<RenderFrame> {
    let mut quiet = Quiet(host);
    runtime.send(&recorded.input)?;
    let rendered = runtime.receive(&quiet)?;
    runtime.follow(&mut quiet, &rendered)?;
    for load in &recorded.loads {
        apply(runtime, &mut quiet, load)?;
    }
    Ok(runtime.take_drawn().unwrap_or(rendered))
}

/// Told of each frame a rewind runs unseen: its position and input, and
/// the runtime it ran on (for its packet's hash).
pub type Unseen<'a> = dyn FnMut(u64, &Recorded, &GameRuntime) -> Result<()> + 'a;

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
        (from..self.cursor).map(|position| self.frames[(position - self.base) as usize].elapsed()).sum()
    }

    /// A frame kept in the history.
    pub fn recorded(&self, position: u64) -> Option<&Recorded> {
        position.checked_sub(self.base).and_then(|index| self.frames.get(index as usize))
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
    pub fn prepare(&mut self, runtime: &mut GameRuntime, host: &dyn Host) -> Result<()> {
        if self.wants_moment() {
            let moment = runtime.moment()?;
            self.keep_moment(moment);
        }
        if self.restored_here() && self.fresh != Some(self.cursor) {
            runtime.reroot(host)?;
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
        self.frames.truncate((self.cursor - self.base) as usize);
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
            time += self.frames[(position - self.base) as usize].elapsed();
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
                time += self.frames[(position - self.base) as usize].elapsed();
            }
            if time >= keep || self.cursor - at >= MAX_KEPT_FRAMES {
                cut = Some(at);
                break;
            }
        }
        if let Some(at) = cut {
            self.moments = self.moments.split_off(&at);
            self.frames.drain(..(at - self.base) as usize);
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
        match self.frames.get_mut((cursor - self.base) as usize) {
            Some(recorded) => recorded.restored = true,
            None => self.restored_next = true,
        }
        self.fresh = Some(cursor);
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
        let earliest = self.earliest().context("there is no history to travel in")?;
        ensure!(target >= earliest, "the history starts at frame {earliest}");
        let target = target.min(self.head());
        // A moment before the target, so the frame shown is the one the
        // target followed; the target itself only when it is the earliest.
        let at = self.moments.range(..target).next_back().map_or(earliest, |(position, _)| *position);
        let moment = self.seek(at);
        let mut shown = runtime.restore(&mut Quiet(host), &moment)?;
        self.restored();
        while self.cursor < target {
            self.prepare(runtime, host)?;
            let recorded = self.frames[(self.cursor - self.base) as usize].clone();
            shown = run_unseen(runtime, host, &recorded)?;
            unseen(self.cursor, &recorded, runtime)?;
            self.ran(recorded);
        }
        Ok((shown, at))
    }

    /// Restores the latest moment at or before `from`, for the recorded
    /// frames after it to play again (on whatever build runs now). Returns
    /// the restored program's first frame and where the replay starts.
    pub fn restart(&mut self, runtime: &mut GameRuntime, host: &dyn Host, from: u64) -> Result<(RenderFrame, u64)> {
        let earliest = self.earliest().context("there is no history to replay")?;
        let at = self.moments.range(..=from).next_back().map_or(earliest, |(position, _)| *position);
        let moment = self.seek(at);
        let shown = runtime.restore(&mut Quiet(host), &moment)?;
        self.restored();
        Ok((shown, at))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Edits;

    fn moment(value: f32) -> Moment {
        Moment { saved: SavedState { schema: 1, state: vec![value] }, edits: Edits::default() }
    }

    fn frame(elapsed: f32) -> Recorded {
        Recorded { input: vec![elapsed, 64.0, 64.0, 0.0, 0.0], ..Recorded::default() }
    }

    /// Runs `count` frames as a session does: a moment when wanted, then the frame.
    fn run(timeline: &mut Timeline, count: u64) {
        for _ in 0..count {
            if timeline.wants_moment() {
                let position = timeline.position() as f32;
                timeline.keep_moment(moment(position));
            }
            timeline.ran(frame(1.0 / 60.0));
        }
    }

    #[test]
    fn keeps_moments_every_interval_and_trims_to_the_window() {
        let mut timeline = Timeline::default();
        timeline.keep_history(Some(2.0));
        run(&mut timeline, 600);
        assert_eq!(timeline.position(), 600);
        assert_eq!(timeline.head(), 600);
        // Two seconds are 120 frames: the newest moment at least that far back.
        let earliest = timeline.earliest().unwrap();
        assert!(earliest.is_multiple_of(MOMENT_INTERVAL) && (450..=480).contains(&earliest), "{earliest}");
        assert!(timeline.seconds() >= 2.0 && timeline.seconds() < 2.6);
        assert_eq!(timeline.seconds_back(1.0), 540);
    }

    #[test]
    fn without_history_nothing_is_kept() {
        let mut timeline = Timeline::default();
        run(&mut timeline, 100);
        assert_eq!((timeline.position(), timeline.head(), timeline.earliest()), (100, 100, None));
        assert!(timeline.next().is_none());

        // A recording's frames play, then play goes on live with nothing kept.
        timeline.load(Recording { start: moment(0.0), frames: vec![frame(0.5); 3] });
        assert_eq!((timeline.position(), timeline.head(), timeline.earliest()), (0, 3, Some(0)));
        run(&mut timeline, 5);
        assert_eq!((timeline.position(), timeline.head(), timeline.earliest()), (5, 5, None));
    }

    #[test]
    fn going_live_in_the_past_drops_the_future() {
        let mut timeline = Timeline::default();
        timeline.keep_history(None);
        run(&mut timeline, 100);
        let moment = timeline.seek(60);
        timeline.restored();
        assert_eq!(moment.saved.state, [60.0]);
        assert!(timeline.restored_here());
        timeline.ran(frame(1.0 / 60.0));
        assert!(timeline.recorded(60).unwrap().restored);
        timeline.go_live();
        assert_eq!((timeline.position(), timeline.head()), (61, 61));
        assert_eq!(timeline.moments.keys().copied().collect::<Vec<_>>(), [0, 30, 60]);

        // Restored at the cursor, then live: the next frame recorded follows a restore.
        timeline.seek(60);
        timeline.restored();
        timeline.go_live();
        assert_eq!(timeline.head(), 60);
        timeline.ran(frame(1.0 / 60.0));
        timeline.ran(frame(1.0 / 60.0));
        assert!(timeline.recorded(60).unwrap().restored && !timeline.recorded(61).unwrap().restored);
    }

    #[test]
    fn a_new_build_replaces_the_moments_after_a_restart() {
        let mut timeline = Timeline::default();
        timeline.keep_history(None);
        run(&mut timeline, 100);
        timeline.rebuilt();
        // The same build: later moments still follow.
        timeline.seek(90);
        assert_eq!(timeline.moments.len(), 4);
        // An older build's moment: those after it are stale.
        timeline.seek(30);
        assert_eq!(timeline.moments.keys().copied().collect::<Vec<_>>(), [0, 30]);
        assert!(timeline.wants_moment());
        run(&mut timeline, 70);
        assert_eq!(timeline.moments.keys().copied().collect::<Vec<_>>(), [0, 30, 60, 90]);
        assert!(timeline.moments.values().skip(1).all(|kept| kept.build == 1));
    }
}

/// The determinism contract on the game itself (`dist/gdev`, from `just
/// export`), without a GPU: the same input gives the same packets, replays
/// give the packets of the run they recorded (rewinds and all), and every
/// frame a rewind runs again gives the packets and state it gave the first
/// time, whatever the restored moment held: menus, cards and the keyboard
/// focus open, panels easing in, and service requests in flight to a host
/// that answers them. Skipped when the bundle is missing.
#[cfg(test)]
mod contract {
    use std::collections::{HashMap, VecDeque};
    use std::path::{Path, PathBuf};

    use pen_protocol::boundary::read_catalog;
    use pen_protocol::{Catalog, Frame, UiNode, Viewport, service};

    use super::*;
    use crate::game::hash_lanes;
    use crate::input::Input;
    use crate::script::{self, Script};
    use crate::services::{Answer, Request, Service, Services};
    use crate::snapshot;

    const VIEWPORT: Viewport = Viewport { width: 960, height: 640 };

    struct Null;

    impl Host for Null {
        fn viewport(&self) -> Viewport {
            VIEWPORT
        }

        fn validate(&self, frame: &RenderFrame) -> Result<()> {
            pen_render::validate_frame(frame)
        }

        fn render(&mut self, _frame: &RenderFrame, _catalog: &Catalog) -> Result<()> {
            Ok(())
        }

        fn report(&mut self, _message: &str) {}
    }

    /// Answers every request of its service with the text "fake", `delay`
    /// frames after it was asked (as a dialog closing would), so requests
    /// wait for their answers across moments.
    struct Late {
        code: u32,
        delay: u32,
        frame: u32,
        /// The frame each answer is due, and its request.
        due: VecDeque<(u32, u32)>,
    }

    impl Service for Late {
        fn code(&self) -> u32 {
            self.code
        }

        fn call(&mut self, request: &Request) -> Option<Answer> {
            self.due.push_back((self.frame + self.delay, request.id));
            None
        }

        fn poll(&mut self) -> Vec<(u32, Answer)> {
            let mut ready = Vec::new();
            while self.due.front().is_some_and(|(due, _)| *due <= self.frame) {
                let (_, id) = self.due.pop_front().unwrap();
                ready.push((id, Answer::text("fake")));
            }
            ready
        }

        fn frame(&mut self, _input: &mut Input) {
            self.frame += 1;
        }
    }

    /// A host whose every service answers `delay` frames late.
    fn late_services(delay: u32) -> Services {
        let mut services = Services::new();
        for code in [service::STEAM, service::CLIPBOARD, service::URL, service::PLATFORM, service::PATHFINDING] {
            services.register(Box::new(Late { code, delay, frame: 0, due: VecDeque::new() }));
        }
        services
    }

    fn bundle() -> Option<PathBuf> {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dist/gdev");
        let found = directory.join("game.wasm").exists();
        if !found {
            eprintln!("skipped: no dist/gdev (`just export`)");
        }
        found.then_some(directory)
    }

    /// What each position's frame gave when it last ran: its packet's hash,
    /// the frame itself (to say how two differ) and the state after it.
    #[derive(Clone, Default)]
    struct Pass {
        packets: Vec<u64>,
        frames: Vec<RenderFrame>,
        states: Vec<u64>,
    }

    fn keep<T>(list: &mut Vec<T>, position: u64, value: T) {
        list.truncate(position as usize);
        list.push(value);
    }

    impl Pass {
        fn keep(&mut self, position: u64, packet: u64, frame: RenderFrame) {
            keep(&mut self.packets, position, packet);
            keep(&mut self.frames, position, frame);
        }

        /// Panics where this pass first differs from `first`, saying how
        /// the frames differ.
        fn assert_same(&self, first: &Pass, context: &str) {
            assert_eq!(self.packets.len(), first.packets.len(), "{context}: the run ends at another frame");
            if let Some(position) = (0..first.packets.len()).find(|&index| self.packets[index] != first.packets[index]) {
                panic!(
                    "{context}: frame {position} gives other packets than the first time\n{}",
                    difference(&first.frames[position], &self.frames[position])
                );
            }
            if let Some(position) = (0..first.states.len()).find(|&index| self.states.get(index) != first.states.get(index)) {
                panic!("{context}: the state after frame {position} differs from the first time");
            }
        }
    }

    /// The entries of `lines` (a `{:#?}` form) indented `indent` spaces: a
    /// struct's fields or a list's elements, each with what is nested in it.
    fn entries<'a>(lines: &[&'a str], indent: usize) -> Vec<Vec<&'a str>> {
        let mut found: Vec<Vec<&str>> = Vec::new();
        for &line in lines {
            let depth = line.len() - line.trim_start().len();
            if depth == indent && !line.trim_start().starts_with(['}', ']', ')']) {
                found.push(vec![line]);
            } else if depth > indent
                && let Some(last) = found.last_mut()
            {
                last.push(line);
            }
        }
        found
    }

    /// Where two `{:#?}` forms first differ: the line in each, after the
    /// lines that enclose it.
    fn first_difference(was: &[&str], now: &[&str]) -> String {
        let at = was.iter().zip(now).position(|(a, b)| a != b).unwrap_or(was.len().min(now.len()));
        let depth = |line: &str| line.len() - line.trim_start().len();
        let line = |lines: &[&str]| lines.get(at).map_or("(none)", |line| line.trim()).to_owned();
        let mut path = Vec::new();
        let mut inside = was.get(at).map_or(usize::MAX, |line| depth(line));
        for enclosing in was[..at.min(was.len())].iter().rev() {
            if depth(enclosing) < inside {
                inside = depth(enclosing);
                path.push(enclosing.trim().trim_end_matches(['{', '[', '(']).trim());
            }
        }
        path.reverse();
        format!("{} > {}\n  again: {}", path.join(" > "), line(was), line(now))
    }

    /// How two frames differ: each field that does, and in it the first
    /// line that does (for a list, of the first element that does).
    fn difference(first: &RenderFrame, again: &RenderFrame) -> String {
        let (first, again) = (format!("{first:#?}"), format!("{again:#?}"));
        let (first, again): (Vec<&str>, Vec<&str>) = (first.lines().collect(), again.lines().collect());
        let mut said = Vec::new();
        for (was, now) in entries(&first, 4).iter().zip(entries(&again, 4)) {
            if *was == now {
                continue;
            }
            let name = was[0].trim().split(':').next().unwrap_or_default();
            let (was_items, now_items) = (entries(&was[1..], 8), entries(&now[1..], 8));
            if was_items.len() == now_items.len()
                && let Some(at) = was_items.iter().zip(&now_items).position(|(a, b)| a != b)
            {
                said.push(format!("{name}[{at}]: {}", first_difference(&was_items[at], &now_items[at])));
                continue;
            }
            if was_items.len() != now_items.len() {
                said.push(format!("{name}: {} entries, {} again", was_items.len(), now_items.len()));
            }
            said.push(first_difference(was, &now));
        }
        said.join("\n")
    }

    /// A game as a session runs it, minus the GPU: frames from a script,
    /// quicksaves kept in memory, a timeline keeping everything, and
    /// perhaps services answering live frames' requests.
    struct Game {
        runtime: GameRuntime,
        module: blot_host::GuestModule,
        timeline: Timeline,
        /// Quicksaves by name ("" for the game's own F6 and F9).
        saves: HashMap<PathBuf, SavedState>,
        services: Option<Services>,
        pass: Pass,
    }

    impl Game {
        fn start(directory: &Path, start: Option<&Moment>) -> Game {
            let engine = blot_host::engine().unwrap();
            let wasm = std::fs::read(directory.join("game.wasm")).unwrap();
            let cache = std::env::temp_dir().join("pen-test-wasm");
            let module = blot_host::GuestModule::new(&engine, &wasm, Some(&cache)).unwrap();
            let catalog = read_catalog(&directory.join("assets")).unwrap();
            let scene = snapshot::read(&directory.join("main.scene")).ok();
            let mut runtime = GameRuntime::new(&engine, module.clone(), catalog, &mut Null, None, scene).unwrap();
            runtime.hash_packets(true);
            let mut timeline = Timeline::default();
            timeline.keep_history(None);
            if let Some(start) = start {
                timeline.load(Recording { start: start.clone(), frames: Vec::new() });
                runtime.restore(&mut Null, start).unwrap();
            }
            Game { runtime, module, timeline, saves: HashMap::new(), services: None, pass: Pass::default() }
        }

        /// Notes what the frame at `position` gave, and every tenth frame
        /// the state after it (a snapshot takes as long as the frame).
        fn note(&mut self, position: u64) {
            let state = if position % 10 == 9 { hash_lanes(&self.runtime.snapshot().unwrap().state) } else { 0 };
            let frame = self.runtime.received().unwrap().clone();
            self.pass.keep(position, self.runtime.packet_hash(), frame);
            keep(&mut self.pass.states, position, state);
        }

        /// Runs one frame of input, live or recorded, with its loads.
        fn run(&mut self, input: Vec<f32>, saves: &[PathBuf], loads: &[PathBuf]) {
            self.timeline.prepare(&mut self.runtime, &Null).unwrap();
            let replayed = self.timeline.next().cloned();
            let input = replayed.as_ref().map_or(input, |recorded| recorded.input.clone());
            self.runtime.send(&input).unwrap();
            let rendered = self.runtime.receive(&Null).unwrap();
            self.runtime.follow(&mut Null, &rendered).unwrap();
            let mut done = Vec::new();
            match &replayed {
                Some(recorded) => {
                    for load in &recorded.loads {
                        apply(&mut self.runtime, &mut Null, load).unwrap();
                    }
                    done.clone_from(&recorded.loads);
                }
                None => {
                    // Only live frames ask the services.
                    if let Some(services) = &mut self.services {
                        services.answer(&rendered.services);
                    }
                    let mut saving: Vec<PathBuf> = saves.to_vec();
                    let mut loading: Vec<PathBuf> = loads.to_vec();
                    if rendered.save_requested {
                        saving.insert(0, PathBuf::new());
                    }
                    if rendered.load_requested {
                        loading.insert(0, PathBuf::new());
                    }
                    for path in saving {
                        self.saves.insert(path, self.runtime.snapshot().unwrap());
                    }
                    for path in loading {
                        let saved = self.saves[&path].clone();
                        self.runtime.load(&mut Null, &saved).unwrap();
                        if let Some(services) = &mut self.services {
                            services.forget();
                        }
                        done.push(Load::Save(saved));
                    }
                }
            }
            let position = self.timeline.position();
            self.note(position);
            let restored = self.timeline.restored_here();
            self.timeline.ran(Recorded { input, loads: done, restored });
        }

        /// Plays a script on a steady 60 Hz clock, the services' answers
        /// coming in with the frame after.
        fn play(&mut self, steps: &str) {
            let mut script = Script::new(script::parse(&[steps.to_owned()]).unwrap());
            let mut time = 0.0;
            while !script.done() {
                let tick = script.tick(VIEWPORT);
                let mut events = tick.events;
                if let Some(services) = &mut self.services {
                    services.frame(&mut Input::default());
                    events.extend(services.take_answers());
                }
                let frame = Frame { timestamp: time, viewport: VIEWPORT, events };
                time += 1.0 / 60.0;
                let input = self.runtime.input(&frame).unwrap();
                self.run(input, &tick.saves, &tick.loads);
            }
        }

        /// Goes to `target`, noting what the frames run unseen gave.
        fn rewind(&mut self, target: u64) {
            let mut unseen = Vec::new();
            let mut note = |position: u64, _: &Recorded, runtime: &GameRuntime| {
                unseen.push((position, runtime.packet_hash(), runtime.received().unwrap().clone()));
                Ok(())
            };
            self.timeline.rewind(&mut self.runtime, &Null, target, &mut note).unwrap();
            if let Some(services) = &mut self.services {
                services.forget();
            }
            for (position, packet, frame) in unseen {
                self.pass.keep(position, packet, frame);
            }
            assert_eq!(self.timeline.position(), target);
        }

        /// Runs recorded frames again up to the end of the record.
        fn catch_up(&mut self) {
            while self.timeline.next().is_some() {
                self.run(Vec::new(), &[], &[]);
            }
        }

        /// Rewinds to `target`, runs the record again to its end, and
        /// checks every frame gave what it did the first time.
        fn travel(&mut self, target: u64, first: &Pass) {
            self.rewind(target);
            self.catch_up();
            self.pass.assert_same(first, &format!("after a rewind to {target}"));
        }

        fn recording(&self) -> Recording {
            let start = self.timeline.moments[&0].moment.clone();
            Recording { start, frames: self.timeline.frames.iter().cloned().collect() }
        }
    }

    /// The editor and play, edits (undo history), undo and redo, and
    /// quicksaves and loads from the script and the game's own keys.
    const STEPS: &str = "wait=30 key=Tab wait=40 click=470,150 wait=20 click=470,152 wait=20 \
        key=Tab wait=30 drag=300,500,600,450 wait=20 drag=200,400,500,350 wait=20 \
        key=KeyU wait=10 key=KeyU wait=10 down=Shift key=KeyU up=Shift wait=10 \
        save=a wait=20 key=F6 wait=20 drag=300,300,400,450 wait=20 load=a wait=20 \
        drag=250,450,450,350 wait=20 key=F9 wait=20 key=Tab wait=30 wheel=-300,480,320 wait=30";

    /// Play and the interface, laid out so a moment (every 30 frames)
    /// falls in each of these, which `rewinds_into_the_interface_...`
    /// checks: the build bar easing in as play starts (frame 20, moment
    /// 30); the keyboard focus on the build bar's roll (F10 at 51: moments
    /// 60 and 90); a plan in hand and its route asked of pathfinding (104),
    /// answered after moment 120; the station's card opening (117: 120),
    /// open, the god's hand swaying after the pointer's last move (148:
    /// 150), and closing (178: 180); the pause menu flying up (235: 240),
    /// its focus moved to the second row (256: 270), its Settings sliding
    /// in (295: 300), a preference changed (316: 330) and the menu flying
    /// back down (352: 360).
    const INTERFACE: &str = "wait=20 key=Tab wait=30 key=F10 wait=40 key=Escape \
        ui=0x20001,activate wait=10 ui=0x20010,activate move=480,420 wait=11 \
        ui=0x30000,activate wait=30 move=500,400 wait=29 ui=0x30000,collapse ui=0x20001,collapse wait=55 \
        key=Escape wait=20 key=ArrowDown wait=38 key=Enter wait=20 key=ArrowRight wait=18 \
        key=Escape wait=16 key=Escape wait=60";

    /// Play with a crowd: twelve gnorps hired, salt knocked out of the rock
    /// for them to haul, a gust sent, a building placed for them to raise
    /// (its route asked of pathfinding first), the pointer left over them.
    const CROWD: &str = "wait=10 key=Tab wait=20 console=3,12 wait=30 \
        click=470,150 wait=15 click=470,152 wait=15 click=471,150 wait=15 click=470,151 wait=30 \
        console=6 wait=30 ui=0x20001,activate wait=10 ui=0x20010,activate move=680,140 wait=20 \
        click=680,140 wait=10 ui=0x20001,collapse move=480,330 wait=480";

    /// Plays a recording in a new program; returns its packets.
    fn replay(directory: &Path, recording: Recording) -> Vec<u64> {
        let mut game = Game::start(directory, Some(&recording.start));
        game.timeline.load(recording);
        game.catch_up();
        game.runtime.close();
        game.pass.packets
    }

    /// The lanes of the section of `state` with `key` (packages/persist.blot).
    fn section(state: &[f32], key: u32) -> Option<Vec<f32>> {
        let mut at = 3;
        for _ in 0..state[2] as usize {
            let length = state[at + 3] as usize;
            if state[at] as u32 == key {
                return Some(state[at + 4..at + 4 + length].to_vec());
            }
            at += 4 + length;
        }
        None
    }

    fn node(frame: &RenderFrame, id: u32) -> Option<&UiNode> {
        frame.ui.iter().find(|node| node.id == id)
    }

    #[test]
    fn the_same_input_gives_the_same_packets() {
        let Some(directory) = bundle() else { return };
        let mut first = Game::start(&directory, None);
        first.play(STEPS);
        let mut second = Game::start(&directory, None);
        second.play(STEPS);
        let packets = &first.pass.packets;
        assert!(packets.len() > 400);
        assert_eq!(*packets, second.pass.packets);
        // Packets differ from frame to frame, so equal runs are not trivially equal.
        assert!(packets.windows(2).filter(|pair| pair[0] != pair[1]).count() > packets.len() / 2);
        // A recording of it replays the same packets.
        assert_eq!(replay(&directory, first.recording()), *packets);
        first.runtime.close();
        second.runtime.close();
    }

    #[test]
    fn the_last_seconds_replay_on_a_new_build() {
        let Some(directory) = bundle() else { return };
        let mut game = Game::start(&directory, None);
        game.play(STEPS);
        let first = game.pass.clone();
        // A hot reload (of the same build here, so the frames must match).
        let catalog = game.runtime.catalog().clone();
        let module = game.module.clone();
        assert_eq!(game.runtime.reload(&mut Null, module, catalog), crate::game::ReloadResult::Activated);
        game.timeline.rebuilt();
        let end = game.timeline.position();
        let from = game.timeline.seconds_back(2.0);
        let (_, at) = game.timeline.restart(&mut game.runtime, &Null, from).unwrap();
        assert!(at <= from && end - at >= 120, "{at}");
        // The moments after it came from the old build: they go, and come
        // back from the new one as the frames replay.
        assert_eq!(game.timeline.moments.keys().last(), Some(&at));
        game.catch_up();
        game.pass.assert_same(&first, "replaying the last seconds after a reload");
        assert!(game.timeline.moments.range(at + 1..).count() >= 3);
        game.runtime.close();
    }

    #[test]
    fn rewinds_give_the_first_packets_and_replay_exactly() {
        let Some(directory) = bundle() else { return };
        let mut game = Game::start(&directory, None);
        game.play(STEPS);
        let first = game.pass.clone();
        let end = game.timeline.position();
        // Into play (frames 30 to 160 play; 110 is between moments), over
        // the undo and loads, to the start and to the last frame.
        for target in [110, end - 200, end - 75, 1, end - 1] {
            game.travel(target, &first);
        }
        // The run with its rewinds, recorded, replays its packets exactly:
        // restores where it was restored included.
        let recording = game.recording();
        assert!(recording.frames.iter().filter(|recorded| recorded.restored).count() >= 4);
        assert_eq!(replay(&directory, recording), game.pass.packets);
        game.runtime.close();
    }

    #[test]
    fn rewinds_into_the_interface_give_the_first_packets() {
        let Some(directory) = bundle() else { return };
        let mut game = Game::start(&directory, None);
        game.services = Some(late_services(40));
        game.play(INTERFACE);
        let first = game.pass.clone();
        let end = game.timeline.position();
        assert_eq!(end, 413);

        // The moments fall where INTERFACE says: each holds what the frame
        // before it left.
        let frames = &first.frames;
        let focused = |at: usize, id: u32| node(&frames[at], id).is_some_and(|node| node.focused);
        let (bar, roll, card) = (0x20000, 0x20001, 0x30001);
        let (paused, settings, second_row) = (0x40001, 0x40002, 0x40101);
        // Answers come 40 frames late: the game's first requests (what it
        // runs on, Steam's stat and presence) wait over moment 30, the
        // route's over 120.
        assert!(frames[..30].iter().any(|frame| frame.services.len() >= 4));
        let route = frames[100..120]
            .iter()
            .position(|frame| frame.services.iter().any(|request| request.service == service::PATHFINDING));
        assert!(route.is_some_and(|at| at + 100 + 40 > 120), "a route asked before 120, answered after");
        let y = |at: usize| node(&frames[at], bar).map(|node| node.y);
        assert!(y(29).is_some() && y(29) != y(30), "the build bar eases in over moment 30");
        assert!(focused(59, roll) && focused(89, roll), "the focus is on the build bar over 60 and 90");
        let card_width = |at: usize| node(&frames[at], card).map(|node| node.width);
        assert!(card_width(119).is_some() && card_width(119) != card_width(130), "the card opens over 120");
        assert!(card_width(149).is_some() && card_width(179).is_some() && card_width(200).is_none(), "and closes over 180");
        assert!(node(&frames[239], paused).is_some() && frames[239].blur > 0.0, "the menu flies up over 240");
        assert!(focused(269, second_row), "its second row has the focus over 270");
        assert!(node(&frames[299], settings).is_some(), "its settings show over 300");
        assert!(frames[316..330].iter().any(|frame| frame.preferences.is_some()), "a preference changed before 330");
        assert!(node(&frames[359], paused).is_none() && frames[359].blur > 0.0, "it flies down over 360");

        // A rewind restores the moment before its target: every moment's.
        for moment in (MOMENT_INTERVAL..end).step_by(MOMENT_INTERVAL as usize) {
            game.travel(moment + 1, &first);
        }
        let recording = game.recording();
        assert_eq!(replay(&directory, recording), game.pass.packets);
        game.runtime.close();
    }

    #[test]
    fn rewinds_among_the_crowd_give_the_first_packets() {
        let Some(directory) = bundle() else { return };
        let mut game = Game::start(&directory, None);
        game.services = Some(late_services(40));
        game.play(CROWD);
        let first = game.pass.clone();
        let end = game.timeline.position();
        let catalog = game.runtime.catalog().clone();
        let mesh = |at: usize, name: &str| first.frames[at].draws.iter().filter(|draw| catalog.mesh(draw.mesh) == name).count();
        let material =
            |at: usize, name: &str| first.frames[at].draws.iter().any(|draw| catalog.material(draw.material) == name);
        let bubble = |at: usize| {
            first.frames[at].sprites.iter().any(|sprite| catalog.material(sprite.material) == "bubble.material.json")
        };
        // What the frames show: the crowd, a gust, a building going up and
        // gnorps speaking, and a route asked.
        assert!((60..end as usize).all(|at| mesh(at, "robot3d.mesh.json") == 12), "twelve gnorps");
        assert!((150..300).any(|at| mesh(at, "wind_strip.mesh.json") > 0), "a gust");
        assert!(material(end as usize - 1, "building_stage.material.json"), "a building going up");
        assert!((0..end as usize).filter(|&at| bubble(at)).count() > 60, "gnorps speaking");
        assert!(first.frames.iter().any(|frame| frame.services.iter().any(|request| request.service == service::PATHFINDING)));
        let state = game.runtime.snapshot().unwrap().state;
        let builders = section(&state, 116).unwrap();
        assert!(builders[0] >= 1.0, "gnorps punching away at it: {builders:?}");

        // Every third moment's: the frames are dearer than the interface's.
        for moment in (MOMENT_INTERVAL..end).step_by(3 * MOMENT_INTERVAL as usize) {
            game.travel(moment + 1, &first);
        }
        let recording = game.recording();
        assert_eq!(replay(&directory, recording), game.pass.packets);
        game.runtime.close();
    }
}
