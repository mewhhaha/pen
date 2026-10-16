//! The dev tools service (`pen_protocol::service::DEVTOOLS`,
//! DevtoolsOperation in `protocol/schema.json`): what the dev tools program
//! run beside the game asks of the host. It reads the log, the last frame's
//! numbers and the GPU's name, and has the host carry out game commands,
//! time travel, input recordings, screenshots and tunables. [`Devtools`]
//! answers the requests the same way in every host; a host says what it has
//! and does through [`Host`]. [`Routing`] says which input the dev tools
//! take from the game.
use std::collections::VecDeque;

use pen_protocol::boundary::whole;
use pen_protocol::{GameEvent, ServiceRequest, UiNode, devtools_frame, devtools_op, ui_role};

use crate::game::FrameTimings;
use crate::services::{Answer, code_points, text};
use crate::snapshot::SavedState;
use crate::tunables::tuned_values;

/// The most values a log answer carries; the program asks again for the rest.
pub const LOG_VALUES: usize = 12_000;

/// Where a log line came from, as the dev tools number them.
pub mod log_source {
    /// The game runtime's reports.
    pub const GAME: u32 = 0;
    /// The compiler watching the checkout.
    pub const BUILD: u32 = 1;
    /// New builds of the game: compiled, reloaded or refused.
    pub const RELOAD: u32 = 2;
    /// Asset files reloading, and the renderer's word on them.
    pub const ASSET: u32 = 3;
    /// The checkout's asset recipes.
    pub const PIPELINE: u32 = 4;
    /// The host: time travel, recordings, saves, loads and bakes.
    pub const SESSION: u32 = 5;
}

/// Where the game stands in time, for the dev tools.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimelineStatus {
    /// Frames run so far: the next frame's position. What shows is the
    /// frame before it.
    pub frame: u64,
    /// The end of the recorded frames; beyond `frame` when paused after a
    /// rewind (stepping runs them again).
    pub latest: u64,
    /// The earliest frame a rewind reaches; none without a history.
    pub earliest: Option<u64>,
    /// Game seconds of history before `frame`.
    pub seconds: f64,
    pub paused: bool,
    /// Recorded frames are playing (a replay).
    pub replaying: bool,
    /// A history is kept.
    pub history: bool,
    /// Frames overlap (native pen's `--overlap`).
    pub overlapped: bool,
}

/// What the host says of the last frame besides where it stands in time.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameReport {
    /// The last frame's timings, every phase measured, while frames are
    /// timed (`Host::measure_frames`).
    pub timings: Option<FrameTimings>,
    pub draw_calls: u32,
    pub instanced_draw_calls: u32,
    pub instances: u32,
    /// Meshes and brushes the frame drew.
    pub meshes: usize,
    pub brushes: usize,
    /// The game's linear memory, in bytes.
    pub memory: usize,
    /// What the frame was drawn into, in pixels.
    pub size: (u32, u32),
    /// The frame the input recording starts at, while one is made.
    pub recording: Option<u64>,
}

/// A line of the log as the dev tools read it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogLine<'a> {
    /// Counts up over the host's lines.
    pub serial: u64,
    /// Where it came from ([`log_source`]).
    pub source: u32,
    pub error: bool,
    pub text: &'a str,
}

/// The log's `lines` from serial `from` on, as many as one answer carries:
/// each its source, whether an error, its length and its code points (a
/// line longer than an answer cut short); the value is the serial to ask
/// from next.
pub fn log_answer<'a>(lines: impl IntoIterator<Item = LogLine<'a>>, from: u64) -> Answer {
    let mut values = Vec::new();
    let mut next = from;
    for line in lines.into_iter().filter(|line| line.serial >= from) {
        let text: Vec<f32> = code_points(line.text).into_iter().take(LOG_VALUES - 3).collect();
        if !values.is_empty() && values.len() + 3 + text.len() > LOG_VALUES {
            break;
        }
        values.extend([line.source as f32, f32::from(u8::from(line.error)), text.len() as f32]);
        values.extend(text);
        next = line.serial + 1;
    }
    Answer { values, ..Answer::ok(next as f32) }
}

/// A log kept in memory for the dev tools: its last `keep` lines.
pub struct Log {
    lines: VecDeque<(u64, u32, bool, String)>,
    next: u64,
    keep: usize,
}

impl Log {
    pub fn new(keep: usize) -> Log {
        Log { lines: VecDeque::new(), next: 0, keep }
    }

    pub fn push(&mut self, source: u32, error: bool, text: impl Into<String>) {
        self.lines.push_back((self.next, source, error, text.into()));
        self.next += 1;
        if self.lines.len() > self.keep {
            self.lines.pop_front();
        }
    }

    /// The answer to a log request from serial `from` on.
    pub fn answer(&self, from: u64) -> Answer {
        let lines = self.lines.iter().map(|(serial, source, error, text)| LogLine {
            serial: *serial,
            source: *source,
            error: *error,
            text,
        });
        log_answer(lines, from)
    }
}

/// What the dev tools ask of the host running the game: what it reports,
/// and what it does for them. Time travel takes effect when the game is
/// next idle, as the host's frame loop sees to.
#[allow(async_fn_in_trait)]
pub trait Host {
    fn timeline(&self) -> TimelineStatus;
    /// The frame `seconds` of game time before `frame`, as far back as the
    /// history reaches.
    fn frames_before(&self, frame: u64, seconds: f64) -> u64;
    /// The answer to a log request from serial `from` on ([`log_answer`]).
    fn log(&self, from: u64) -> Answer;
    fn report(&self) -> FrameReport;
    /// Times the game's frames, or stops.
    fn measure_frames(&mut self, enabled: bool);
    /// The graphics adapter's name.
    fn gpu(&self) -> String;
    /// An event for the game's next live frame.
    fn enqueue(&mut self, event: GameEvent);
    fn pause(&mut self);
    fn resume(&mut self);
    /// Runs `frames` frames and stays paused.
    fn step_frames(&mut self, frames: u32);
    /// Goes to frame `frame`, paused.
    fn rewind_to(&mut self, frame: u64);
    /// Plays the last `seconds` again, then plays on live.
    fn replay_recent(&mut self, seconds: f64);
    /// Keeps the past to travel in from now on.
    fn keep_history(&mut self);
    /// Starts recording the input to `path` (a new file when none), or
    /// stops the recording being made.
    fn record(&mut self, _path: Option<String>) -> Answer {
        Answer::unavailable()
    }
    /// Saves the next frame with the dev tools over it to `path` (a new
    /// file when none).
    fn shot(&mut self, _path: Option<String>) -> Answer {
        Answer::unavailable()
    }
    /// The game's state, as a save holds it; none when it cannot say now.
    async fn snapshot(&mut self) -> Option<SavedState>;
    /// Writes the game's tuned values into its source as the defaults.
    async fn bake(&mut self) -> Answer;
}

/// Answers the dev tools' requests, and keeps what that takes: whether the
/// game's frames are timed for them, and where the last frame answer was.
#[derive(Debug, Default)]
pub struct Devtools {
    measuring: bool,
    /// The game's position at the last frame answer: a frame that did not
    /// run has no timings.
    position: Option<u64>,
}

impl Devtools {
    /// Does what a devtools request asks, or answers what it asks about.
    /// `interval` is the time since the dev tools' last frame, in ms.
    pub async fn answer(&mut self, request: &ServiceRequest, host: &mut impl Host, interval: f64) -> Answer {
        let (a, b) = (request.a, request.b);
        let status = host.timeline();
        let path = || Some(text(&request.values)).filter(|path| !path.is_empty());
        match request.operation {
            devtools_op::LOG => host.log(whole(a).map_or(0, u64::from)),
            devtools_op::FRAME => self.numbers(host, interval, a == 1.0),
            devtools_op::GPU => Answer::text(&host.gpu()),
            devtools_op::COMMAND => {
                host.enqueue(GameEvent::Console { command: a, argument: b });
                Answer::ok(0.0)
            }
            devtools_op::PAUSE => {
                host.pause();
                Answer::ok(0.0)
            }
            devtools_op::PLAY => {
                host.resume();
                Answer::ok(0.0)
            }
            devtools_op::STEP => {
                host.step_frames(whole(a).unwrap_or(1).clamp(1, 600));
                Answer::ok(0.0)
            }
            devtools_op::REWIND | devtools_op::REPLAY if !status.history => Answer::unavailable(),
            devtools_op::REWIND => {
                host.rewind_to(host.frames_before(status.frame, f64::from(a.max(0.0))));
                Answer::ok(0.0)
            }
            devtools_op::REPLAY => {
                host.replay_recent(f64::from(a.max(0.0)));
                Answer::ok(0.0)
            }
            devtools_op::HISTORY => {
                host.keep_history();
                Answer::ok(0.0)
            }
            devtools_op::RECORD => host.record(path()),
            devtools_op::SHOT => host.shot(path()),
            devtools_op::TUNE => match whole(a) {
                Some(index) => {
                    host.enqueue(GameEvent::Tune { index, value: b });
                    Answer::ok(0.0)
                }
                None => Answer::failed(),
            },
            devtools_op::TUNED => match host.snapshot().await {
                Some(saved) => {
                    let mut tuned: Vec<(u32, f32)> = tuned_values(&saved.state).into_iter().collect();
                    tuned.sort_by_key(|(id, _)| *id);
                    let values = tuned.into_iter().flat_map(|(id, value)| [id as f32, value]).collect();
                    Answer { values, ..Answer::ok(0.0) }
                }
                None => Answer::unavailable(),
            },
            devtools_op::BAKE => host.bake().await,
            _ => Answer::unknown(),
        }
    }

    /// The last frame's numbers, in DevtoolsFrame's order; frames are timed
    /// while `measure` says so, and a frame that did not run has no timings.
    fn numbers(&mut self, host: &mut impl Host, interval: f64, measure: bool) -> Answer {
        if measure != self.measuring {
            host.measure_frames(measure);
            self.measuring = measure;
        }
        let status = host.timeline();
        let ran = self.position.replace(status.frame) != Some(status.frame);
        let report = host.report();
        let timings = report.timings.filter(|_| ran && measure);
        let mut values = vec![0.0; devtools_frame::NAMES.len()];
        #[allow(clippy::cast_possible_truncation, reason = "numbers to show")]
        let mut set = |at: u32, value: f64| values[at as usize] = value as f32;
        set(devtools_frame::INTERVAL, interval);
        let phase = |pick: fn(&FrameTimings) -> f64| timings.as_ref().map_or(-1.0, pick);
        set(devtools_frame::GUEST, phase(|t| t.guest_ms));
        set(devtools_frame::WAIT, phase(|t| t.wait_ms));
        set(devtools_frame::DECODE, phase(|t| t.decode_ms));
        set(devtools_frame::VALIDATE, phase(|t| t.validate_ms));
        set(devtools_frame::SUBMIT, phase(|t| t.submit_ms));
        set(devtools_frame::HOUSEKEEPING, phase(|t| t.housekeeping_ms));
        set(devtools_frame::TOTAL, phase(|t| t.total_ms));
        set(devtools_frame::PACKET, timings.map_or(-1.0, |t| t.packet_bytes as f64));
        set(devtools_frame::DRAW_CALLS, f64::from(report.draw_calls));
        set(devtools_frame::INSTANCED_DRAW_CALLS, f64::from(report.instanced_draw_calls));
        set(devtools_frame::INSTANCES, f64::from(report.instances));
        set(devtools_frame::MESHES, report.meshes as f64);
        set(devtools_frame::BRUSHES, report.brushes as f64);
        set(devtools_frame::MEMORY, report.memory as f64);
        set(devtools_frame::WIDTH, f64::from(report.size.0));
        set(devtools_frame::HEIGHT, f64::from(report.size.1));
        set(devtools_frame::POSITION, status.frame as f64);
        set(devtools_frame::PAUSED, f64::from(u8::from(status.paused)));
        set(devtools_frame::REPLAYING, f64::from(u8::from(status.replaying)));
        set(devtools_frame::HISTORY, f64::from(u8::from(status.history)));
        set(devtools_frame::PIPELINED, f64::from(u8::from(status.overlapped)));
        set(devtools_frame::RECORDING, report.recording.map_or(-1.0, |from| from as f64));
        Answer { values, ..Answer::ok(0.0) }
    }
}

/// Whether `point` is inside one of the interface's dialogs.
pub fn inside_dialog(ui: &[UiNode], (x, y): (f32, f32)) -> bool {
    ui.iter().any(|node| {
        node.role == ui_role::DIALOG
            && !node.hidden
            && x >= node.x
            && y >= node.y
            && x < node.x + node.width
            && y < node.y + node.height
    })
}

/// A pointer button (numbered as `GameEvent`s number them) as a bit of a
/// set; buttons past the fifth share one.
pub fn button_bit(button: u32) -> u32 {
    1 << button.min(5)
}

/// Which input the dev tools take from the game: every key while one of
/// their nodes has the focus (their console's input) and the backquote key
/// always, and the pointer and wheel over their dialogs, but for a drag the
/// game started, which stays the game's.
#[derive(Debug, Default)]
pub struct Routing {
    /// One of their nodes has the focus.
    pub typing: bool,
    /// The pointer is over one of their dialogs, and where it is.
    pub over: bool,
    pub pointer: (f32, f32),
    /// Buttons the game got pressed, whose drags and releases stay its own.
    pub buttons: u32,
}

impl Routing {
    /// An event for the game, the dev tools' last interface being `ui` and
    /// `running` whether their program runs: None for an event they do not
    /// take, else whether the game gets it too.
    pub fn route(&mut self, event: &GameEvent, ui: &[UiNode], running: bool) -> Option<bool> {
        match event {
            GameEvent::KeyDown(_) | GameEvent::KeyUp(_) | GameEvent::FocusLost => {}
            GameEvent::PointerMove { x, y }
            | GameEvent::PointerDown { x, y, .. }
            | GameEvent::PointerUp { x, y, .. }
            | GameEvent::Wheel { x, y, .. } => {
                self.pointer = (*x, *y);
                self.over = inside_dialog(ui, self.pointer);
            }
            _ => return None,
        }
        if matches!(event, GameEvent::KeyDown(key) | GameEvent::KeyUp(key) if key == "`") {
            return Some(false);
        }
        if !running {
            return Some(true);
        }
        Some(match event {
            GameEvent::KeyDown(_) | GameEvent::KeyUp(_) => !self.typing,
            GameEvent::PointerMove { .. } => self.buttons != 0 || !self.over,
            GameEvent::PointerDown { button, .. } => {
                let game = self.buttons != 0 || !self.over;
                if game {
                    self.buttons |= button_bit(*button);
                }
                game
            }
            GameEvent::PointerUp { button, .. } => {
                self.buttons &= !button_bit(*button);
                true
            }
            GameEvent::Wheel { .. } => !self.over,
            _ => true,
        })
    }

    /// The dev tools laid out `ui`: whether it took the keyboard just now
    /// (keys held then would never come up for the game, which should let
    /// go of them as when the window loses the focus).
    pub fn laid_out(&mut self, ui: &[UiNode]) -> bool {
        let typing = ui.iter().any(|node| node.focused);
        let took = typing && !self.typing;
        self.typing = typing;
        self.over = inside_dialog(ui, self.pointer);
        took
    }

    /// Their program stopped: it takes nothing.
    pub fn stopped(&mut self) {
        self.typing = false;
        self.over = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_protocol::service;

    fn line(serial: u64, text: &str) -> LogLine<'_> {
        LogLine { serial, source: log_source::BUILD, error: serial % 2 == 1, text }
    }

    #[test]
    fn answers_the_log_a_whole_line_at_a_time() {
        let lines = [line(4, "ab"), line(5, "c"), line(6, "")];
        let answer = log_answer(lines, 5);
        assert_eq!(answer.value, 7.0);
        assert_eq!(answer.values, [1.0, 1.0, 1.0, 99.0, 1.0, 0.0, 0.0]);
        assert_eq!(log_answer(lines, 7), Answer::ok(7.0));
        // As many lines as fit; the next answer goes on from the first left out.
        let long = "x".repeat(LOG_VALUES / 2);
        let answer = log_answer([line(0, &long), line(1, &long), line(2, "y")], 0);
        assert_eq!((answer.value, answer.values.len()), (1.0, 3 + long.len()));
        // A line longer than an answer is cut short, and its length says so.
        let longest = "z".repeat(LOG_VALUES * 2);
        let answer = log_answer([line(0, &longest)], 0);
        assert_eq!(answer.values.len(), LOG_VALUES);
        assert_eq!(answer.values[2], (LOG_VALUES - 3) as f32);
    }

    #[test]
    fn a_kept_log_drops_its_oldest_lines() {
        let mut log = Log::new(2);
        for text in ["a", "b", "c"] {
            log.push(log_source::SESSION, false, text);
        }
        let answer = log.answer(0);
        assert_eq!(answer.value, 3.0);
        assert_eq!(answer.values, [5.0, 0.0, 1.0, 98.0, 5.0, 0.0, 1.0, 99.0]);
    }

    /// A host that keeps what it was asked.
    #[derive(Default)]
    struct Recorder {
        status: Option<TimelineStatus>,
        asked: Vec<String>,
        events: Vec<GameEvent>,
        measuring: bool,
        state: Vec<f32>,
    }

    impl Host for Recorder {
        fn timeline(&self) -> TimelineStatus {
            self.status.unwrap_or(TimelineStatus {
                frame: 100,
                latest: 100,
                earliest: Some(40),
                seconds: 1.0,
                paused: false,
                replaying: false,
                history: true,
                overlapped: false,
            })
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "the tests' whole seconds")]
        fn frames_before(&self, frame: u64, seconds: f64) -> u64 {
            frame - (seconds * 30.0) as u64
        }
        fn log(&self, from: u64) -> Answer {
            Answer::ok(from as f32)
        }
        fn report(&self) -> FrameReport {
            let timings = FrameTimings { guest_ms: 2.0, ..FrameTimings::default() };
            FrameReport { timings: Some(timings), size: (64, 48), recording: Some(7), ..FrameReport::default() }
        }
        fn measure_frames(&mut self, enabled: bool) {
            self.measuring = enabled;
        }
        fn gpu(&self) -> String {
            "test adapter".into()
        }
        fn enqueue(&mut self, event: GameEvent) {
            self.events.push(event);
        }
        fn pause(&mut self) {
            self.asked.push("pause".into());
        }
        fn resume(&mut self) {
            self.asked.push("resume".into());
        }
        fn step_frames(&mut self, frames: u32) {
            self.asked.push(format!("step {frames}"));
        }
        fn rewind_to(&mut self, frame: u64) {
            self.asked.push(format!("rewind {frame}"));
        }
        fn replay_recent(&mut self, seconds: f64) {
            self.asked.push(format!("replay {seconds}"));
        }
        fn keep_history(&mut self) {
            self.asked.push("history".into());
        }
        async fn snapshot(&mut self) -> Option<SavedState> {
            Some(SavedState { schema: 1, state: self.state.clone() })
        }
        async fn bake(&mut self) -> Answer {
            Answer::failed()
        }
    }

    /// The service's futures never wait on this host.
    fn ready<T>(future: impl Future<Output = T>) -> T {
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match std::pin::pin!(future).poll(&mut context) {
            std::task::Poll::Ready(value) => value,
            std::task::Poll::Pending => unreachable!("the test host answers at once"),
        }
    }

    fn ask(devtools: &mut Devtools, host: &mut Recorder, operation: u32, a: f32, values: &str) -> Answer {
        let request = ServiceRequest {
            service: service::DEVTOOLS,
            operation,
            a,
            values: code_points(values),
            ..ServiceRequest::default()
        };
        ready(devtools.answer(&request, host, 16.5))
    }

    #[test]
    fn time_travel_goes_by_game_time_and_needs_a_history() {
        let (mut devtools, mut host) = (Devtools::default(), Recorder::default());
        ask(&mut devtools, &mut host, devtools_op::STEP, 0.0, "");
        ask(&mut devtools, &mut host, devtools_op::STEP, 5000.0, "");
        ask(&mut devtools, &mut host, devtools_op::REWIND, 2.0, "");
        ask(&mut devtools, &mut host, devtools_op::REPLAY, 1.5, "");
        assert_eq!(host.asked, ["step 1", "step 600", "rewind 40", "replay 1.5"]);
        host.status = Some(TimelineStatus { history: false, ..host.timeline() });
        assert_eq!(ask(&mut devtools, &mut host, devtools_op::REWIND, 1.0, ""), Answer::unavailable());
        assert_eq!(ask(&mut devtools, &mut host, devtools_op::RECORD, 0.0, "a.penrec"), Answer::unavailable());
        assert_eq!(ask(&mut devtools, &mut host, 999, 0.0, ""), Answer::unknown());
    }

    #[test]
    fn frame_numbers_are_timed_while_asked_and_once_per_frame() {
        let (mut devtools, mut host) = (Devtools::default(), Recorder::default());
        let frame = ask(&mut devtools, &mut host, devtools_op::FRAME, 1.0, "");
        assert!(host.measuring);
        let at = |answer: &Answer, code: u32| answer.values[code as usize];
        assert_eq!(at(&frame, devtools_frame::GUEST), 2.0);
        assert_eq!(at(&frame, devtools_frame::INTERVAL), 16.5);
        assert_eq!((at(&frame, devtools_frame::WIDTH), at(&frame, devtools_frame::POSITION)), (64.0, 100.0));
        assert_eq!(at(&frame, devtools_frame::RECORDING), 7.0);
        // The game did not run since: no timings again.
        let again = ask(&mut devtools, &mut host, devtools_op::FRAME, 1.0, "");
        assert_eq!(at(&again, devtools_frame::GUEST), -1.0);
        ask(&mut devtools, &mut host, devtools_op::FRAME, 0.0, "");
        assert!(!host.measuring);
    }

    #[test]
    fn numbers_buttons_as_bits() {
        let bits: Vec<u32> = (0..8).map(button_bit).collect();
        assert_eq!(bits, [1, 2, 4, 8, 16, 32, 32, 32]);
    }

    fn node(role: u32, x: f32, y: f32, width: f32, height: f32) -> UiNode {
        UiNode { role, x, y, width, height, ..UiNode::default() }
    }

    #[test]
    fn takes_the_pointer_over_dialogs_only() {
        let ui = [node(ui_role::DIALOG, 10.0, 10.0, 100.0, 50.0), node(ui_role::STATUS, 500.0, 0.0, 200.0, 200.0)];
        assert!(inside_dialog(&ui, (10.0, 10.0)));
        assert!(!inside_dialog(&ui, (110.0, 30.0)));
        assert!(!inside_dialog(&ui, (600.0, 50.0)));
        let hidden = UiNode { hidden: true, ..ui[0].clone() };
        assert!(!inside_dialog(&[hidden], (20.0, 20.0)));
    }

    #[test]
    fn routes_keys_while_typing_and_keeps_the_games_drags() {
        let ui = [node(ui_role::DIALOG, 0.0, 0.0, 100.0, 100.0)];
        let mut routing = Routing::default();
        let key = |name: &str| GameEvent::KeyDown(name.to_owned());
        assert_eq!(routing.route(&key("`"), &ui, true), Some(false), "the console's key is theirs");
        assert_eq!(routing.route(&GameEvent::Console { command: 1.0, argument: 0.0 }, &ui, true), None);
        assert_eq!(routing.route(&key("a"), &ui, true), Some(true));
        assert!(routing.laid_out(&[UiNode { focused: true, ..ui[0].clone() }]) && routing.typing);
        assert_eq!(routing.route(&key("a"), &ui, true), Some(false), "typed into the console");
        assert_eq!(routing.route(&key("a"), &ui, false), Some(true), "unless their program stopped");
        // A drag the game started over open ground stays its own over a dialog.
        assert_eq!(routing.route(&GameEvent::PointerDown { button: 0, x: 200.0, y: 50.0 }, &ui, true), Some(true));
        assert_eq!(routing.route(&GameEvent::PointerMove { x: 50.0, y: 50.0 }, &ui, true), Some(true));
        assert_eq!(routing.route(&GameEvent::PointerUp { button: 0, x: 50.0, y: 50.0 }, &ui, true), Some(true));
        assert_eq!(routing.route(&GameEvent::PointerDown { button: 0, x: 50.0, y: 50.0 }, &ui, true), Some(false));
        assert_eq!(routing.route(&GameEvent::Wheel { x: 50.0, y: 50.0, delta: 1.0 }, &ui, true), Some(false));
    }

    #[test]
    fn tunables_go_to_the_game_and_come_back_from_its_state() {
        let (mut devtools, mut host) = (Devtools::default(), Recorder::default());
        ask(&mut devtools, &mut host, devtools_op::TUNE, 3.0, "");
        assert!(matches!(host.events[..], [GameEvent::Tune { index: 3, .. }]));
        // A state with one section: the tuned values, 9 = 0.5 and 4 = 2.
        host.state = vec![5678.0, 0.0, 1.0, 11.0, 0.0, 0.0, 4.0, 9.0, 0.5, 4.0, 2.0];
        let tuned = ask(&mut devtools, &mut host, devtools_op::TUNED, 0.0, "");
        assert_eq!(tuned.values, [4.0, 2.0, 9.0, 0.5]);
        assert_eq!(ask(&mut devtools, &mut host, devtools_op::GPU, 0.0, ""), Answer::text("test adapter"));
    }
}
