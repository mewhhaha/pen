//! The dev tools service (`pen_protocol::service::DEVTOOLS`,
//! DevtoolsOperation in `protocol/schema.json`): what the dev tools program
//! run beside the game asks of the host. It reads the log, the last frame's
//! numbers and the GPU's name, and has the host carry out game commands,
//! time travel, input recordings, screenshots and tunables, and looks into
//! the game: its packets, its state (and a watch on it), its guest's
//! samples and memory, its assets and materials (`crate::inspect` works out
//! what hosts alike can). [`Devtools`] answers the requests the same way in
//! every host; a host says what it has and does through [`Host`].
//! [`Routing`] says which input the dev tools take from the game.
use std::collections::VecDeque;

use pen_protocol::boundary::{MAX_BUSES, whole};
use pen_protocol::{GameEvent, ServiceRequest, UiNode, devtools_compare, devtools_frame, devtools_op, ui_role};

use crate::game::FrameTimings;
use crate::inspect::{self, Packets, Watch};
use crate::mix::AudioReport;
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
    /// What the renderer left out of the camera's view and of the shadow
    /// maps, the triangles it drew, and the render scale it drew at.
    pub culled: u32,
    pub shadow_culled: u32,
    pub triangles: u64,
    pub scale: f32,
    /// The GPU's milliseconds over a recent frame, where it has timestamps:
    /// shadow maps, culling, scene, the passes after it, sprites and insets,
    /// and the whole frame.
    pub gpu: Option<[f32; 6]>,
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
    /// The accessibility audit of the game's interface as a recent frame
    /// showed it (`crate::audit::answer`), the next frame read back for it.
    fn audit(&mut self) -> Answer {
        Answer::unavailable()
    }
    /// The mix as it plays, its meters read afresh; none without sound.
    fn audio(&mut self) -> Option<AudioReport> {
        None
    }
    /// Mutes (`mute`) or solos bus `bus`, or stops; false without sound.
    fn monitor(&mut self, _bus: usize, _mute: bool, _on: bool) -> bool {
        false
    }
    /// Hands the game's state over as a save file named `path` (a new
    /// one when none); answers where it went.
    async fn export_save(&mut self, _path: Option<String>) -> Answer {
        Answer::unavailable()
    }
    /// Loads the save at `path`, or one the player hands over.
    fn import_save(&mut self, _path: Option<String>) -> Answer {
        Answer::unavailable()
    }
    /// The game's state, as a save holds it; none when it cannot say now.
    async fn snapshot(&mut self) -> Option<SavedState>;
    /// Writes the game's tuned values into its source as the defaults.
    async fn bake(&mut self) -> Answer;
    /// Writes only tunable `id`'s value into its source as its default.
    async fn bake_one(&mut self, _id: u32) -> Answer {
        Answer::unavailable()
    }
    /// A line for the log, from the dev tools' service.
    fn say(&mut self, _line: &str) {}
    /// The last frames' packets, for a host that keeps them.
    fn packets(&mut self) -> Option<&mut Packets> {
        None
    }
    /// The state the history kept `seconds` of game time back (at the
    /// latest moment it kept by then) and the frame it was kept at.
    fn kept_state(&self, _seconds: f64) -> Option<(u64, SavedState)> {
        None
    }
    /// Takes pictures of the game's frames for the dev tools' atlas (the
    /// texture of material `atlas`, when named), or stops; answers the
    /// pictures (each its slot and frame).
    fn pictures(&mut self, _taking: bool, _atlas: Option<u32>) -> Answer {
        Answer::unavailable()
    }
    /// Samples the game's guest (`operation` 1), stops (0) or forgets the
    /// samples (2), or only asks (anything else); answers how many there
    /// are and whether it samples.
    fn profile(&mut self, _operation: u32) -> Answer {
        Answer::unavailable()
    }
    /// The samples as a flame graph (`crate::inspect::flame`).
    fn flame(&mut self) -> Answer {
        Answer::unavailable()
    }
    /// The game guest's memory and ownership counters.
    fn memory(&mut self) -> Answer {
        Answer::unavailable()
    }
    /// The checkout's tunable presets' names, a line each.
    fn presets(&mut self) -> Answer {
        Answer::unavailable()
    }
    /// Saves the game's tuned values as preset `name`.
    async fn save_preset(&mut self, _name: &str) -> Answer {
        Answer::unavailable()
    }
    /// The values preset `name` holds, by tunable id.
    fn preset(&mut self, _name: &str) -> Result<Vec<(u32, f32)>, Answer> {
        Err(Answer::unavailable())
    }
    /// The game's catalog of `kind` (DevtoolsAsset) and where each stands.
    fn assets(&mut self, _kind: u32) -> Answer {
        Answer::unavailable()
    }
    /// Draws catalog asset `index` of `kind` with `other` (a mesh's
    /// material, a material's mesh) into atlas slot `slot`.
    fn preview(&mut self, _kind: u32, _index: u32, _other: u32, _slot: u32) -> Answer {
        Answer::unavailable()
    }
    /// Draws catalog asset `index` of `kind` from asset `key` instead (its
    /// own again for an empty key).
    fn swap(&mut self, _kind: u32, _index: u32, _key: &str) -> Answer {
        Answer::unavailable()
    }
    /// Material `index`'s color and instance parameters, and its shader and
    /// texture.
    fn material(&mut self, _index: u32) -> Answer {
        Answer::unavailable()
    }
    /// Sets number `field` of material `index` (its color's four, then its
    /// instance parameters) in its file to `value`.
    fn set_material(&mut self, _index: u32, _field: u32, _value: f32) -> Answer {
        Answer::unavailable()
    }
}

/// Answers the dev tools' requests, and keeps what that takes: whether the
/// game's frames are timed for them, where the last frame answer was, and a
/// watch on the game's state.
#[derive(Debug, Default)]
pub struct Devtools {
    measuring: bool,
    /// The game's position at the last frame answer: a frame that did not
    /// run has no timings.
    position: Option<u64>,
    watch: Option<Watch>,
    /// The game's position when the watch last looked.
    watched: Option<u64>,
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
            devtools_op::BAKE => match whole(a) {
                Some(0) | None => host.bake().await,
                Some(id) => host.bake_one(id).await,
            },
            devtools_op::PACKETS => match host.packets() {
                Some(packets) => {
                    packets.keep(a == 1.0);
                    packets.answer()
                }
                None => Answer::unavailable(),
            },
            devtools_op::PACKET_DIFF => match (host.packets(), whole(a), whole(b)) {
                (Some(packets), Some(a), Some(b)) => packets.diff(u64::from(a), u64::from(b)),
                (None, ..) => Answer::unavailable(),
                _ => Answer::failed(),
            },
            devtools_op::STATE => {
                let state = if request.c > 0.0 {
                    host.kept_state(f64::from(request.c)).map(|(_, saved)| saved)
                } else {
                    host.snapshot().await
                };
                match (state, whole(a)) {
                    (Some(saved), None) => inspect::sections_answer(saved.schema, &saved.state),
                    (Some(saved), Some(key)) => {
                        inspect::lanes_answer(&saved.state, key, whole(b).map_or(0, |from| from as usize))
                    }
                    (None, _) => Answer::unavailable(),
                }
            }
            devtools_op::STATE_DIFF => match host.kept_state(f64::from(a.max(0.0))) {
                Some((frame, then)) => match host.snapshot().await {
                    Some(now) => {
                        Answer { values: inspect::state_diff(&then.state, &now.state), ..Answer::ok(frame as f32) }
                    }
                    None => Answer::unavailable(),
                },
                None => Answer::unavailable(),
            },
            devtools_op::PICTURES => host.pictures(a == 1.0, whole(b)),
            devtools_op::SEEK if !status.history => Answer::unavailable(),
            devtools_op::SEEK => match whole(a) {
                Some(frame) => {
                    host.rewind_to(u64::from(frame));
                    Answer::ok(0.0)
                }
                None => Answer::failed(),
            },
            devtools_op::WATCH => {
                if a == -1.0 {
                    self.watch = None;
                } else if let (Some(key), Some(lane), Some(compare)) = (whole(a), whole(b), whole(request.c)) {
                    self.watch = Some(Watch::new(key, lane, compare, request.d));
                    self.watched = None;
                    // It starts from the lane as it is now.
                    if let Some(saved) = host.snapshot().await
                        && let Some(watch) = &mut self.watch
                    {
                        watch.look(&saved.state, status.frame);
                        watch.paused = None;
                        self.watched = Some(status.frame);
                    }
                } else if a != -2.0 {
                    return Answer::failed();
                }
                let values = self.watch.as_ref().map(Watch::values).unwrap_or_default();
                Answer { values, ..Answer::ok(0.0) }
            }
            devtools_op::PROFILE => match whole(a) {
                Some(operation) => host.profile(operation),
                None => Answer::failed(),
            },
            devtools_op::FLAME => host.flame(),
            devtools_op::MEMORY => host.memory(),
            devtools_op::PRESETS => host.presets(),
            devtools_op::PRESET_SAVE => match path() {
                Some(name) => host.save_preset(&name).await,
                None => Answer::failed(),
            },
            devtools_op::PRESET_LOAD => match path().map(|name| host.preset(&name)) {
                Some(Ok(values)) => {
                    let count = values.len();
                    for (index, value) in values {
                        host.enqueue(GameEvent::Tune { index, value });
                    }
                    Answer::ok(count as f32)
                }
                Some(Err(answer)) => answer,
                None => Answer::failed(),
            },
            devtools_op::ASSETS => match whole(a) {
                Some(kind) => host.assets(kind),
                None => Answer::failed(),
            },
            devtools_op::PREVIEW => match (whole(a), whole(b), whole(request.c), whole(request.d)) {
                (Some(kind), Some(index), Some(other), Some(slot)) => host.preview(kind, index, other, slot),
                _ => Answer::failed(),
            },
            devtools_op::SWAP => match (whole(a), whole(b)) {
                (Some(kind), Some(index)) => host.swap(kind, index, &text(&request.values)),
                _ => Answer::failed(),
            },
            devtools_op::MATERIAL => match whole(a) {
                Some(index) => host.material(index),
                None => Answer::failed(),
            },
            devtools_op::MATERIAL_SET => match (whole(a), whole(b)) {
                (Some(index), Some(field @ 0..8)) if request.c.is_finite() => {
                    host.set_material(index, field, request.c)
                }
                _ => Answer::failed(),
            },
            devtools_op::AUDIT => host.audit(),
            devtools_op::AUDIO => host.audio().map_or_else(Answer::unavailable, |report| report.answer()),
            devtools_op::MUTE | devtools_op::SOLO => match whole(a).map(|bus| bus as usize) {
                Some(bus) if bus < MAX_BUSES => {
                    if host.monitor(bus, request.operation == devtools_op::MUTE, b == 1.0) {
                        Answer::ok(0.0)
                    } else {
                        Answer::unavailable()
                    }
                }
                _ => Answer::failed(),
            },
            devtools_op::EXPORT => host.export_save(path()).await,
            devtools_op::IMPORT => host.import_save(path()),
            _ => Answer::unknown(),
        }
    }

    /// Once a dev tools frame: when a watch is set and the game ran since
    /// it last looked, looks at the game's state and pauses the game when
    /// the watch comes to hold, saying why in the log.
    pub async fn follow(&mut self, host: &mut impl Host) {
        let Some(watch) = &mut self.watch else { return };
        let status = host.timeline();
        if status.paused || self.watched == Some(status.frame) {
            return;
        }
        let Some(saved) = host.snapshot().await else { return };
        self.watched = Some(status.frame);
        if watch.look(&saved.state, status.frame) {
            host.pause();
            let shown = status.frame.saturating_sub(1);
            let compare = devtools_compare::NAMES.get(watch.compare as usize).copied().unwrap_or("?");
            let (key, lane, value) = (watch.key, watch.lane, watch.now.unwrap_or_default());
            host.say(&format!(
                "watch: paused after frame {shown}: lane {lane} of section {key} is {value} ({compare} {})",
                watch.value
            ));
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
        set(devtools_frame::CULLED, f64::from(report.culled));
        set(devtools_frame::SHADOW_CULLED, f64::from(report.shadow_culled));
        set(devtools_frame::TRIANGLES, report.triangles as f64);
        set(devtools_frame::RENDER_SCALE, f64::from(report.scale));
        let gpu = |at: usize| report.gpu.map_or(-1.0, |gpu| f64::from(gpu[at]));
        set(devtools_frame::GPU_SHADOW, gpu(0));
        set(devtools_frame::GPU_CULLING, gpu(1));
        set(devtools_frame::GPU_SCENE, gpu(2));
        set(devtools_frame::GPU_POST, gpu(3));
        set(devtools_frame::GPU_OVERLAY, gpu(4));
        set(devtools_frame::GPU_TOTAL, gpu(5));
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

/// Which input the dev tools take from the game: every key and character
/// of text while one of their nodes has the focus (their console's input)
/// and the backquote key always, and the pointer and wheel over their
/// dialogs, but for a drag the game started, which stays the game's.
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
            GameEvent::KeyDown(_)
            | GameEvent::KeyUp(_)
            | GameEvent::TextInput { .. }
            | GameEvent::Composition { .. }
            | GameEvent::FocusLost => {}
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
            GameEvent::KeyDown(_)
            | GameEvent::KeyUp(_)
            | GameEvent::TextInput { .. }
            | GameEvent::Composition { .. } => !self.typing,
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
        packets: Packets,
        said: Vec<String>,
        /// The mix's mute and solo, when the host has sound.
        mixing: Option<crate::mix::Monitor>,
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
            FrameReport {
                timings: Some(timings),
                size: (64, 48),
                recording: Some(7),
                culled: 12,
                scale: 0.75,
                gpu: Some([0.5, 0.0, 1.5, 0.25, 0.125, 2.5]),
                ..FrameReport::default()
            }
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
        fn audio(&mut self) -> Option<AudioReport> {
            let monitor = self.mixing?;
            Some(AudioReport {
                rate: 48_000.0,
                buses: AudioReport::buses(&[], &[true], monitor),
                ..AudioReport::default()
            })
        }
        fn monitor(&mut self, bus: usize, mute: bool, on: bool) -> bool {
            self.mixing.as_mut().map(|monitor| monitor.set(bus, mute, on)).is_some()
        }
        async fn snapshot(&mut self) -> Option<SavedState> {
            Some(SavedState { schema: 1, state: self.state.clone() })
        }
        async fn bake(&mut self) -> Answer {
            Answer::failed()
        }
        fn say(&mut self, line: &str) {
            self.said.push(line.to_owned());
        }
        fn packets(&mut self) -> Option<&mut Packets> {
            Some(&mut self.packets)
        }
        fn kept_state(&self, seconds: f64) -> Option<(u64, SavedState)> {
            (seconds <= 1.0)
                .then(|| (40, SavedState { schema: 1, state: vec![5678.0, 1.0, 1.0, 20.0, 1.0, 0.0, 1.0, 1.0] }))
        }
        fn preset(&mut self, name: &str) -> Result<Vec<(u32, f32)>, Answer> {
            if name == "fast" { Ok(vec![(4, 2.0), (9, 0.5)]) } else { Err(Answer::failed()) }
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
        ask_all(devtools, host, operation, [a, 0.0, 0.0, 0.0], values)
    }

    fn ask_all(
        devtools: &mut Devtools,
        host: &mut Recorder,
        operation: u32,
        [a, b, c, d]: [f32; 4],
        values: &str,
    ) -> Answer {
        let request = ServiceRequest {
            service: service::DEVTOOLS,
            operation,
            a,
            b,
            c,
            d,
            values: code_points(values),
            ..ServiceRequest::default()
        };
        ready(devtools.answer(&request, host, 16.5))
    }

    /// A state with one section, key 20, whose lane 1 is `value`.
    fn watched(value: f32) -> Vec<f32> {
        vec![5678.0, 1.0, 1.0, 20.0, 1.0, 0.0, 2.0, 0.0, value]
    }

    #[test]
    fn a_watch_pauses_the_game_when_its_lane_comes_to_hold() {
        let (mut devtools, mut host) = (Devtools::default(), Recorder::default());
        host.state = watched(1.0);
        let watch = ask_all(&mut devtools, &mut host, devtools_op::WATCH, [20.0, 1.0, 3.0, 5.0], "");
        assert_eq!(watch.values, [20.0, 1.0, 3.0, 5.0, 1.0, -1.0]);
        // The game has not run since: it does not look again.
        host.state = watched(9.0);
        ready(devtools.follow(&mut host));
        assert!(host.asked.is_empty());
        host.status = Some(TimelineStatus { frame: 101, ..host.timeline() });
        ready(devtools.follow(&mut host));
        assert_eq!(host.asked, ["pause"]);
        assert!(
            host.said[0].starts_with("watch: paused after frame 100: lane 1 of section 20 is 9"),
            "{:?}",
            host.said
        );
        let asked = ask_all(&mut devtools, &mut host, devtools_op::WATCH, [-2.0, 0.0, 0.0, 0.0], "");
        assert_eq!(asked.values, [20.0, 1.0, 3.0, 5.0, 9.0, 101.0]);
        ask(&mut devtools, &mut host, devtools_op::WATCH, -1.0, "");
        assert!(ask(&mut devtools, &mut host, devtools_op::WATCH, -2.0, "").values.is_empty());
    }

    #[test]
    fn looks_into_states_packets_and_presets() {
        let (mut devtools, mut host) = (Devtools::default(), Recorder::default());
        host.state = watched(3.0);
        let sections = ask(&mut devtools, &mut host, devtools_op::STATE, -1.0, "");
        assert_eq!((sections.value, sections.values), (1.0, vec![20.0, 1.0, 0.0, 2.0]));
        let lanes = ask_all(&mut devtools, &mut host, devtools_op::STATE, [20.0, 1.0, 0.0, 0.0], "");
        assert_eq!((lanes.value, lanes.values), (2.0, vec![3.0]));
        let then = ask_all(&mut devtools, &mut host, devtools_op::STATE, [20.0, 0.0, 1.0, 0.0], "");
        assert_eq!(then.values, [1.0], "the state kept a second back");
        let diff = ask(&mut devtools, &mut host, devtools_op::STATE_DIFF, 1.0, "");
        assert_eq!((diff.value, diff.values), (40.0, vec![20.0, 1.0, 2.0, 2.0, 0.0]));
        assert_eq!(ask(&mut devtools, &mut host, devtools_op::STATE_DIFF, 5.0, ""), Answer::unavailable());
        // Packets are kept once asked for.
        assert_eq!(ask(&mut devtools, &mut host, devtools_op::PACKETS, 1.0, "").value, 0.0);
        host.packets.ran(7, &[0.0; 5], 3, &pen_protocol::RenderFrame::default());
        assert_eq!(ask(&mut devtools, &mut host, devtools_op::PACKETS, 1.0, "").value, 1.0);
        let same = ask_all(&mut devtools, &mut host, devtools_op::PACKET_DIFF, [7.0, 7.0, 0.0, 0.0], "");
        assert!(same.values.chunks(3).all(|triple| triple[2] == 0.0));
        assert_eq!(ask(&mut devtools, &mut host, devtools_op::PACKETS, 0.0, "").value, 0.0, "forgotten");
        // A preset tunes each of its values.
        assert_eq!(ask(&mut devtools, &mut host, devtools_op::PRESET_LOAD, 0.0, "fast"), Answer::ok(2.0));
        assert!(matches!(host.events[..], [GameEvent::Tune { index: 4, .. }, GameEvent::Tune { index: 9, .. }]));
        assert_eq!(ask(&mut devtools, &mut host, devtools_op::PRESET_LOAD, 0.0, "slow"), Answer::failed());
        assert_eq!(ask(&mut devtools, &mut host, devtools_op::PROFILE, 1.0, ""), Answer::unavailable());
        ask(&mut devtools, &mut host, devtools_op::SEEK, 55.0, "");
        assert_eq!(host.asked, ["rewind 55"]);
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
        // A host without save files says so.
        assert_eq!(ask(&mut devtools, &mut host, devtools_op::EXPORT, 0.0, "a.save"), Answer::unavailable());
        assert_eq!(ask(&mut devtools, &mut host, devtools_op::IMPORT, 0.0, ""), Answer::unavailable());
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
        assert_eq!((at(&frame, devtools_frame::CULLED), at(&frame, devtools_frame::RENDER_SCALE)), (12.0, 0.75));
        assert_eq!((at(&frame, devtools_frame::GPU_SCENE), at(&frame, devtools_frame::GPU_TOTAL)), (1.5, 2.5));
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
        // Gamepads and touches are the game's alone.
        assert_eq!(routing.route(&GameEvent::GamepadConnected { pad: 0, standard: 1 }, &ui, true), None);
        assert_eq!(routing.route(&key("a"), &ui, true), Some(true));
        assert!(routing.laid_out(&[UiNode { focused: true, ..ui[0].clone() }]) && routing.typing);
        assert_eq!(routing.route(&key("a"), &ui, true), Some(false), "typed into the console");
        let text = GameEvent::TextInput { character: 'a' as u32, composing: 0 };
        assert_eq!(routing.route(&text, &ui, true), Some(false), "text too");
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

    /// The mix answers with its numbers and each bus's, and mutes and
    /// solos buses; a host without sound answers unavailable.
    #[test]
    fn the_mix_answers_and_mutes_and_solos() {
        let (mut devtools, mut host) = (Devtools::default(), Recorder::default());
        let monitor = |devtools: &mut Devtools, host: &mut Recorder, operation: u32, bus: f32, on: f32| {
            let request =
                ServiceRequest { service: service::DEVTOOLS, operation, a: bus, b: on, ..ServiceRequest::default() };
            ready(devtools.answer(&request, host, 16.5))
        };
        assert_eq!(ask(&mut devtools, &mut host, devtools_op::AUDIO, 0.0, ""), Answer::unavailable());
        assert_eq!(monitor(&mut devtools, &mut host, devtools_op::MUTE, 1.0, 1.0), Answer::unavailable());
        host.mixing = Some(crate::mix::Monitor::default());
        assert_eq!(monitor(&mut devtools, &mut host, devtools_op::MUTE, 1.0, 1.0), Answer::ok(0.0));
        assert_eq!(monitor(&mut devtools, &mut host, devtools_op::SOLO, 2.0, 1.0), Answer::ok(0.0));
        assert_eq!(monitor(&mut devtools, &mut host, devtools_op::SOLO, 16.0, 1.0), Answer::failed());
        let answer = ask(&mut devtools, &mut host, devtools_op::AUDIO, 0.0, "");
        let head = pen_protocol::devtools_audio::NAMES.len();
        let state = |bus: usize| answer.values[head + bus * 5 + pen_protocol::devtools_bus::STATE as usize];
        assert_eq!((state(0), state(1), state(2)), (1.0, 2.0, 4.0));
        assert_eq!(answer.values[pen_protocol::devtools_audio::RATE as usize], 48_000.0);
    }
}
