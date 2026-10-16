//! One running game: its GPU, renderer, program, audio and input, and the
//! frame loop the window and the headless runner share. Frames can also be
//! captured (screenshots, recordings), scripted, timed on a steady clock and
//! hot reloaded.
//!
//! The game is deterministic given its input, so a session can also record
//! its input and replay it exactly, pause, step, rewind to any recent frame
//! and play the last seconds again on a new build (see `timeline`), and
//! overlap frames: the guest computes the next frame while this one draws
//! (`Playback::overlap`, `--overlap`: one frame more input latency).
//!
//! In a window, the dev tools (`Overlay`) can draw over the game's frames.
//!
//! Its parts: the GPU and what frames are drawn into (`gpu`), the player's
//! preferences (`preferences`) and frame measurements (`profile`); the
//! game's clock (`pen_runtime::clock`) and what was asked of time
//! (`pen_runtime::transport`) are the runtime's, which the browser shares.
mod gpu;
mod preferences;
mod profile;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use blot_host::GuestModule;
use pen_protocol::{Catalog, GameEvent, RenderFrame, UiNode, Viewport};
use pen_render::RenderTarget;
use pen_runtime::clock::Clock;
use pen_runtime::devtools::TimelineStatus;
#[cfg(all(test, feature = "compiler"))]
use pen_runtime::transport;
use pen_runtime::transport::{Replaying, Rewind, Transport};
use winit::event::WindowEvent;
use winit::window::Window;

use crate::audio::Audio;
use crate::dev::{Dev, Update};
use crate::devlog::{self, Source};
use crate::game::{FrameTimings, GameRuntime, Host, InvalidFrame, Quiet, ReloadResult, Unresponsive, hash_lanes};
use crate::input::{Hotkey, Input};
use crate::record::Recorder;
use crate::recording;
use crate::script::{self, Script, Step, Tick, Travel};
use crate::services::Services;
use crate::snapshot::{self, SavedState};
use crate::timeline::{self, KEEP_SECONDS, Load, REPLAY_SECONDS, Recorded, Timeline};
#[cfg(feature = "devtools")]
use gpu::Drawn;
pub use gpu::Gpu;
use gpu::{CaptureTarget, Output, Readback, SurfaceTarget, settle, write_overlay_shot, write_png};
use preferences::Preferences;
use profile::Profile;

/// Where the game's files are and where the player's are kept.
pub struct Paths {
    pub game: PathBuf,
    pub assets: PathBuf,
    pub quicksave: PathBuf,
    /// The scene the editor saves: the player's own, or the given file.
    pub scene: PathBuf,
    pub bundled_scene: PathBuf,
    pub preferences: PathBuf,
    /// Where `shot=NAME` steps write their PNGs.
    pub shots: PathBuf,
}

impl Paths {
    /// `data` replaces the platform data directory, where saves and
    /// preferences live (tests use a fresh one).
    pub fn new(
        game: PathBuf,
        assets: Option<PathBuf>,
        scene: Option<PathBuf>,
        shots: Option<PathBuf>,
        data: Option<PathBuf>,
    ) -> Paths {
        let data = data.unwrap_or_else(crate::data_dir);
        Paths {
            assets: assets.unwrap_or_else(|| game.join("assets")),
            bundled_scene: scene.clone().unwrap_or_else(|| game.join("main.scene")),
            scene: scene.unwrap_or_else(|| data.join("main.scene")),
            quicksave: data.join("quicksave.bin"),
            preferences: data.join("preferences.json"),
            shots: shots.unwrap_or_else(|| PathBuf::from("shots")),
            game,
        }
    }

    pub fn scene_to_read(&self) -> &Path {
        if self.scene.exists() { &self.scene } else { &self.bundled_scene }
    }
}

/// The dev tools over the window (`crate::devtools`): they see the window's
/// events before the game, run before each step (acting on the session),
/// and draw over the window's image after the game's frame and before it is
/// presented. They never draw into captures, recordings or headless output,
/// only into overlay shots.
pub trait Overlay {
    /// A window event, before the game: whether the game gets it too.
    fn window_event(&mut self, window: &Window, event: &WindowEvent) -> bool;
    /// Scripted input follows the same ownership rules as window input.
    fn script_event(&mut self, _event: &GameEvent) -> bool {
        true
    }
    /// Before a step: what the overlay shows, and what it asks of the
    /// session; headless, it has no window and lays itself out on the frame.
    fn update(&mut self, session: &mut Session, window: Option<&Window>);
    /// A line for its console, as if typed there (a script's `overlay=`),
    /// run on the next update.
    fn command(&mut self, _line: String) {}
    /// Draws what was laid out last over `target`, an image about to be shown.
    fn paint(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture);
    /// The pointer is the overlay's (over one of its dialogs), so it shows
    /// even where the game hides it.
    fn wants_pointer(&self) -> bool;
    /// Its interface for assistive technology from its last layout, a tree
    /// of its own (`accessibility`); none while it shows none.
    fn interface(&self) -> Option<&[UiNode]> {
        None
    }
    /// An assistive technology's request for one of its nodes (in its own
    /// tree's terms), for its next layout.
    fn accesskit_action(&mut self, _request: accesskit::ActionRequest) {}
}

/// What to keep of the frames drawn during one step.
#[derive(Default)]
struct Capture {
    screenshots: Vec<PathBuf>,
    recorder: Option<Recorder>,
    /// The recording's frames on their way back from the GPU.
    readback: Readback,
    /// Always draw offscreen (headless).
    offscreen: bool,
    /// Screenshots with the overlay, as the window shows it.
    overlay_shots: Vec<PathBuf>,
}

/// Everything the game runtime may touch while a frame runs.
struct Shell<'a> {
    gpu: &'a mut Gpu,
    window: Option<&'a Window>,
    audio: Option<&'a mut Audio>,
    input: &'a mut Input,
    preferences: &'a mut Preferences,
    services: &'a mut Services,
    capture: &'a mut Capture,
    overlay: Option<&'a mut Box<dyn Overlay>>,
}

impl Host for Shell<'_> {
    fn viewport(&self) -> Viewport {
        Viewport { width: self.gpu.size.0.max(1), height: self.gpu.size.1.max(1) }
    }

    fn validate(&self, frame: &RenderFrame) -> Result<()> {
        pen_render::validate_frame(frame)
    }

    fn render(&mut self, frame: &RenderFrame, catalog: &Catalog) -> Result<()> {
        let capture = &mut *self.capture;
        let offscreen = capture.offscreen
            || !capture.screenshots.is_empty()
            || capture.recorder.is_some()
            || !capture.overlay_shots.is_empty();
        let texture = offscreen.then(|| self.gpu.capture_texture());
        let gpu = &mut *self.gpu;
        #[cfg(feature = "devtools")]
        {
            gpu.drawn = Drawn { meshes: frame.draws.len(), brushes: frame.solids.len() };
        }
        let output = Output {
            device: &gpu.device,
            queue: &gpu.queue,
            surface: gpu.surface.as_ref(),
            size: gpu.size,
            blitter: gpu.blitter.as_ref(),
            overlay: self.overlay.as_deref_mut(),
        };
        match &texture {
            Some(texture) => {
                let mut target = CaptureTarget { output, texture: texture.clone() };
                gpu.renderer.submit(frame, catalog, &mut target)?;
            }
            None => {
                let mut target = SurfaceTarget { output, texture: None };
                gpu.renderer.submit(frame, catalog, &mut target)?;
                target.present();
            }
        }
        if let Some(texture) = &texture {
            for path in capture.overlay_shots.drain(..) {
                write_overlay_shot(gpu, texture, self.overlay.as_deref_mut(), &path)?;
            }
        }
        if let Some(texture) = &texture
            && !capture.screenshots.is_empty()
        {
            let pixels = gpu.read(texture)?;
            for path in capture.screenshots.drain(..) {
                write_png(&path, &pixels, texture.width(), texture.height(), gpu.bgra())?;
            }
        }
        if let Some(texture) = &texture
            && let Some(recorder) = &mut capture.recorder
        {
            recorder.sound(frame, catalog)?;
            for (pixels, (width, height)) in capture.readback.start(&gpu.device, &gpu.queue, texture)? {
                recorder.picture(pixels, width, height)?;
            }
        }
        if let Some(audio) = self.audio.as_deref_mut() {
            audio.frame(frame, catalog);
        }
        show_cursor(self.window, frame, self.overlay.as_ref().is_some_and(|overlay| overlay.wants_pointer()));
        if let Some(preferences) = &frame.preferences {
            self.preferences.store(preferences)?;
        }
        Ok(())
    }

    fn report(&mut self, message: &str) {
        // The runtime's word on a new build starts "Blot reload".
        match message.strip_prefix("Blot reload") {
            Some(rest) if rest.starts_with("ed") => devlog::info(Source::Reload, message),
            Some(_) => devlog::error(Source::Reload, message),
            None => devlog::info(Source::Game, message),
        }
    }

    fn restored(&mut self) {
        self.preferences.replay(self.input);
        // The answers waiting were for the world that asked.
        self.services.forget();
    }
}

/// The pointer shows unless the frame hides it, and always while the
/// overlay wants it.
fn show_cursor(window: Option<&Window>, frame: &RenderFrame, overlay: bool) {
    if let Some(window) = window {
        window.set_cursor_visible(!frame.hide_cursor || overlay);
    }
}

pub struct Options {
    /// Stop after this many frames, or 0 to run until closed (or, headless,
    /// until the script ends).
    pub frames: usize,
    /// Write the last frame here.
    pub screenshot: Option<PathBuf>,
    /// Record every frame to this video.
    pub record: Option<PathBuf>,
    /// Steady game time: this many frames per game second.
    pub fps: Option<f64>,
    pub mute: bool,
    /// Reload the game and its assets when their files change.
    pub dev: bool,
    pub script: Vec<Step>,
    pub headless: bool,
    /// Before the game clock starts, draw the first frame until every asset
    /// has loaded and faded in, so later frames depend only on game time
    /// and input (repeatable shots, recordings without pop-in).
    pub settle: bool,
}

/// What a session does besides playing: replay a recording, record the
/// input, keep a history to travel in, overlap frames, trace packets.
#[derive(Clone, Debug, Default)]
pub struct Playback {
    /// Play this recording's input instead of the player's (`--replay`).
    pub replay: Option<PathBuf>,
    /// Record the input to this file (`--record-input`).
    pub record_input: Option<PathBuf>,
    /// Keep the last seconds to rewind and replay (`--history`; always
    /// while developing or when a script travels back).
    pub history: bool,
    /// Compute the next frame while this one draws (`--overlap`).
    pub overlap: bool,
    /// Write a line per frame: its position, and its input's and packet's
    /// hashes (`--trace`).
    pub trace: Option<PathBuf>,
}

/// Overlapping frames (`Playback::overlap`): the guest computes the next
/// frame while this one draws.
#[derive(Default)]
struct Overlap {
    enabled: bool,
    /// The frame the guest computes while the last one draws.
    flight: Option<Planned>,
    /// A script tick taken early that travels in time, so its frame waits.
    early: Option<Tick>,
}

/// A frame on its way to the game: its input, and what to do with it.
struct Planned {
    position: u64,
    input: Vec<f32>,
    /// A recorded frame's loads, which it applies instead of reading files;
    /// None for a live frame.
    replayed: Option<Vec<Load>>,
    shots: Vec<PathBuf>,
    /// Shots with the overlay.
    overlay_shots: Vec<PathBuf>,
    saves: Vec<PathBuf>,
    loads: Vec<PathBuf>,
    /// The run ends after this frame.
    stop: bool,
}

pub struct Session {
    options: Options,
    paths: Paths,
    engine: wasmtime::Engine,
    pub gpu: Gpu,
    runtime: GameRuntime,
    audio: Option<Audio>,
    pub input: Input,
    preferences: Preferences,
    /// What the game asks of the platform and native libraries: called for
    /// live frames only, their answers the next live frame's input.
    pub services: Services,
    clock: Clock,
    script: Option<Script>,
    capture: Capture,
    dev: Option<Dev>,
    profile: Option<Profile>,
    frame_times: bool,
    /// Steps taken (frames run or paused ones presented).
    frames: usize,
    /// Paces a window's frames to game time (a steady clock, a replay):
    /// when it started and how much game time has passed since.
    pacer: Option<(Instant, f64)>,
    timeline: Timeline,
    /// Playing or paused, recorded frames playing, and the time travel and
    /// input recordings waiting for the guest to be idle.
    transport: Transport,
    /// A new build waiting for the guest to be idle.
    build: Option<(GuestModule, Catalog)>,
    overlap: Overlap,
    /// The frame last shown, shown again while paused.
    last: Option<RenderFrame>,
    /// The GPU's still texture holds `last`, drawn.
    still: bool,
    input_recorder: Option<recording::Writer>,
    trace: Option<BufWriter<File>>,
    /// The renderer's asset events while recorded frames play: the game
    /// gets them once it plays live again.
    held: Vec<GameEvent>,
    /// Draws over the window's frames (`Overlay`).
    overlay: Option<Box<dyn Overlay>>,
    /// The last frame's timings, while measured.
    timings: Option<FrameTimings>,
    /// The input recording's file and the frame it starts at.
    recording_input: Option<(PathBuf, u64)>,
    /// A screenshot with the overlay of the last frame (`--overlay-shot`).
    final_overlay_shot: Option<PathBuf>,
}

fn read_optional(path: &Path) -> Option<SavedState> {
    if !path.exists() {
        return None;
    }
    snapshot::read(path).map_err(|error| eprintln!("{error:#}")).ok()
}

/// Everything the game runtime may touch while a frame runs, borrowed from
/// the session's fields so the runtime and timeline stay free.
macro_rules! shell {
    ($session:expr, $window:expr) => {
        Shell {
            gpu: &mut $session.gpu,
            window: $window,
            audio: $session.audio.as_mut(),
            input: &mut $session.input,
            preferences: &mut $session.preferences,
            services: &mut $session.services,
            capture: &mut $session.capture,
            overlay: $session.overlay.as_mut(),
        }
    };
}

/// A save or load the frame or script asked for; a failure is reported, not fatal.
fn report(result: Result<()>, done: String) -> bool {
    match result {
        Ok(()) => {
            devlog::info(Source::Session, done);
            true
        }
        Err(error) => {
            devlog::error(Source::Session, format!("{error:#}"));
            false
        }
    }
}

/// Loads a quicksave or (with `scene`) the editor's scene from a file;
/// returns the state loaded, for the input recording.
fn load_file(runtime: &mut GameRuntime, host: &mut dyn Host, path: &Path, scene: bool) -> Option<Load> {
    let saved = snapshot::read(path).map_err(|error| eprintln!("{error:#}")).ok()?;
    let (result, what) =
        if scene { (runtime.load_scene(host, &saved), "scene ") } else { (runtime.load(host, &saved), "") };
    let load = if scene { Load::Scene(saved) } else { Load::Save(saved) };
    report(result, format!("loaded {what}{}", path.display())).then_some(load)
}

impl Session {
    pub fn start(
        options: Options,
        paths: Paths,
        engine: wasmtime::Engine,
        module: GuestModule,
        catalog: Catalog,
        mut gpu: Gpu,
        window: Option<&Window>,
    ) -> Result<Session> {
        gpu.renderer.preload(&catalog);
        // Headless runs and recordings play the sound into the recording, not the speakers.
        let audio = if options.mute || options.headless || options.record.is_some() {
            None
        } else {
            Audio::new(paths.assets.clone(), &catalog).map_err(|error| eprintln!("audio is off: {error:#}")).ok()
        };
        let fps = options.fps.or((options.headless || options.record.is_some()).then_some(60.0));
        let clock = match fps {
            Some(fps) => Clock::steady(fps),
            None => Clock::real(),
        };
        let recorder = match &options.record {
            Some(path) => Some(Recorder::start(
                path,
                gpu.size,
                fps.unwrap_or(60.0),
                gpu.bgra(),
                (!options.mute).then_some((paths.assets.as_path(), &catalog)),
            )?),
            None => None,
        };
        let mut capture = Capture { recorder, offscreen: options.headless, ..Capture::default() };
        let mut input = Input::default();
        let mut preferences = Preferences::load(paths.preferences.clone());
        preferences.replay(&mut input);
        let mut services = crate::services::standard();
        let dev = if options.dev { Some(Dev::new(&paths.game, &paths.assets)?) } else { None };
        let mut audio = audio;
        let runtime = {
            let mut shell = Shell {
                gpu: &mut gpu,
                window,
                audio: audio.as_mut(),
                input: &mut input,
                preferences: &mut preferences,
                services: &mut services,
                capture: &mut capture,
                overlay: None,
            };
            GameRuntime::new(
                &engine,
                module,
                catalog,
                &mut shell,
                read_optional(&paths.quicksave),
                read_optional(paths.scene_to_read()),
            )?
        };
        let mut runtime = runtime;
        let first = runtime.take_drawn();
        if let Some(frame) = &first
            && options.settle
        {
            settle(&mut gpu, frame, runtime.catalog())?;
        }
        let profile = Profile::from_env()?;
        let frame_times = std::env::var_os("PEN_FRAME_TIMES").is_some();
        runtime.measure_frames(profile.is_some() || frame_times);
        let script = (!options.script.is_empty()).then(|| Script::new(options.script.clone()));
        let mut timeline = Timeline::default();
        if options.dev || script::travels_back(&options.script) {
            timeline.keep_history(Some(KEEP_SECONDS));
        }
        Ok(Session {
            options,
            paths,
            engine,
            gpu,
            runtime,
            audio,
            input,
            preferences,
            services,
            clock,
            script,
            capture,
            dev,
            profile,
            frame_times,
            frames: 0,
            pacer: None,
            timeline,
            transport: Transport::default(),
            build: None,
            overlap: Overlap::default(),
            last: first,
            still: false,
            input_recorder: None,
            trace: None,
            held: Vec::new(),
            overlay: None,
            timings: None,
            recording_input: None,
            final_overlay_shot: None,
        })
    }

    /// Sets up what the session does besides playing; before the first step.
    pub fn playback(&mut self, playback: Playback, window: Option<&Window>) -> Result<()> {
        self.overlap.enabled = playback.overlap;
        if playback.history || playback.replay.is_some() {
            self.timeline.keep_history(Some(KEEP_SECONDS));
        }
        if let Some(path) = &playback.trace {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let file = File::create(path).with_context(|| format!("writing {}", path.display()))?;
            self.trace = Some(BufWriter::new(file));
            self.runtime.hash_packets(true);
        }
        if let Some(path) = &playback.replay {
            let recording = recording::read(path)?;
            let start = recording.start.clone();
            let frames = recording.frames.len();
            let seconds: f64 = recording.frames.iter().map(Recorded::elapsed).sum();
            self.timeline.load(recording);
            let shell = shell!(self, window);
            let shown = self
                .runtime
                .restore(&mut Quiet(&shell), &start)
                .with_context(|| format!("replaying {}", path.display()))?;
            if self.options.settle {
                settle(&mut self.gpu, &shown, self.runtime.catalog())?;
            }
            self.show(shown);
            self.services.forget();
            self.transport.replay(Replaying::File);
            eprintln!("replaying {frames} frames ({seconds:.1} s of game time) from {}", path.display());
        }
        if let Some(path) = &playback.record_input {
            let moment = self.runtime.moment()?;
            self.input_recorder = Some(recording::Writer::create(path, self.timeline.position(), &moment)?);
            self.recording_input = Some((path.clone(), self.timeline.position()));
        }
        Ok(())
    }

    /// Stops the game where it is; the window keeps showing (and drawing)
    /// the last frame.
    pub fn pause(&mut self) {
        let was_playing = self.transport.pause();
        self.stopped(was_playing);
    }

    /// The game time stops, if it had not (`was_playing`).
    fn stopped(&mut self, was_playing: bool) {
        if was_playing {
            self.clock.pause();
            self.pacer = None;
        }
    }

    /// Plays on. After a rewind, play continues live from there, and the
    /// frames that had followed are forgotten; a paused replay plays on.
    pub fn resume(&mut self) {
        if !self.transport.resume() {
            return;
        }
        if self.transport.replaying().is_none() && self.timeline.next().is_some() {
            self.timeline.go_live();
        }
        self.clock.resume();
        self.pacer = None;
    }

    /// Runs `frames` frames and stays paused (pausing first if need be).
    /// Paused after a rewind, it runs the next recorded frames again.
    pub fn step_frames(&mut self, frames: u32) {
        let was_playing = self.transport.step(frames);
        self.stopped(was_playing);
    }

    /// Goes to frame `frame` (pausing there) by restoring the nearest
    /// moment before it and running the recorded frames again unseen.
    /// Takes effect on the next step.
    pub fn rewind_to(&mut self, frame: u64) {
        let was_playing = self.transport.rewind_to(frame);
        self.stopped(was_playing);
    }

    /// Plays the last `seconds` of recorded input again, on whatever build
    /// runs now (after a hot reload, the new one), then plays on live.
    /// Takes effect on the next step.
    pub fn replay_recent(&mut self, seconds: f64) {
        self.transport.replay_recent(seconds);
    }

    pub fn timeline(&self) -> TimelineStatus {
        let frame = self.timeline.position();
        TimelineStatus {
            frame,
            latest: self.timeline.head().max(frame),
            earliest: self.timeline.earliest(),
            seconds: self.timeline.seconds(),
            paused: self.transport.paused(),
            replaying: self.transport.replaying().is_some(),
            history: self.timeline.history(),
            overlapped: self.overlap.enabled,
        }
    }

    #[cfg(feature = "devtools")]
    /// The frame `seconds` of game time before `frame`, as far back as the
    /// history reaches.
    pub fn frames_before(&self, frame: u64, seconds: f64) -> u64 {
        self.timeline.frames_before(frame, seconds)
    }

    #[cfg(feature = "devtools")]
    /// Draws `overlay` over the window's frames from the next step on.
    pub fn set_overlay(&mut self, overlay: Box<dyn Overlay>) {
        self.overlay = Some(overlay);
    }

    /// A window event, for the overlay first: whether the game gets it too.
    pub fn overlay_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.overlay.as_mut().is_none_or(|overlay| overlay.window_event(window, event))
    }

    /// A new frame shows; its image is not drawn yet.
    fn show(&mut self, frame: RenderFrame) {
        // The next frame decodes into the one this replaces.
        if let Some(shown) = self.last.replace(frame) {
            self.runtime.recycle(shown);
        }
        self.still = false;
    }

    /// The interface the frame shown declares for assistive technology
    /// (`RenderFrame::ui`, see `accessibility`).
    pub fn interface(&self) -> &[UiNode] {
        self.last.as_ref().map_or(&[], |frame| &frame.ui)
    }

    /// The overlay's interface for assistive technology, while it shows one.
    pub fn overlay_interface(&self) -> Option<&[UiNode]> {
        self.overlay.as_ref().and_then(|overlay| overlay.interface())
    }

    /// An assistive technology's request for one of the overlay's nodes.
    pub fn overlay_accesskit_action(&mut self, request: accesskit::ActionRequest) {
        if let Some(overlay) = &mut self.overlay {
            overlay.accesskit_action(request);
        }
    }

    #[cfg(feature = "devtools")]
    /// Times each frame's phases (`frame_timings`), as PEN_PROFILE_FRAMES does.
    pub fn measure_frames(&mut self, enabled: bool) {
        self.runtime.measure_frames(enabled || self.profile.is_some() || self.frame_times);
    }

    #[cfg(feature = "devtools")]
    /// The last frame's timings, while measured.
    pub fn frame_timings(&self) -> Option<FrameTimings> {
        self.timings
    }

    #[cfg(feature = "devtools")]
    /// The game's linear memory, in bytes.
    pub fn guest_memory(&self) -> usize {
        self.runtime.memory_bytes()
    }

    #[cfg(feature = "devtools")]
    /// The running game's state, as a quicksave holds it (the dev tools
    /// read the tuned values in it); none while an overlapped frame
    /// computes, or if the game cannot say.
    pub fn snapshot(&mut self) -> Option<SavedState> {
        if self.overlap.flight.is_some() {
            return None;
        }
        self.runtime.snapshot().map_err(|error| devlog::error(Source::Session, format!("{error:#}"))).ok()
    }

    #[cfg(feature = "devtools")]
    /// Keeps the last seconds to travel in from now on, as `--history` does
    /// from the start.
    pub fn keep_history(&mut self) {
        if !self.timeline.history() {
            self.timeline.keep_history(Some(KEEP_SECONDS));
        }
    }

    #[cfg(feature = "devtools")]
    /// Starts recording the input to `path` (a `.penrec`, from the frame
    /// it then is), or with None stops, once the guest is idle.
    pub fn record_input(&mut self, path: Option<PathBuf>) {
        self.transport.record(path);
    }

    #[cfg(feature = "devtools")]
    /// The input recording's file and the frame it starts at, while recording.
    pub fn input_recording(&self) -> Option<(&Path, u64)> {
        self.recording_input.as_ref().map(|(path, from)| (path.as_path(), *from))
    }

    #[cfg(feature = "devtools")]
    /// Writes the next frame shown with the overlay over it, as the window
    /// shows it (screenshots and recordings never have it).
    pub fn overlay_shot(&mut self, path: PathBuf) {
        self.capture.overlay_shots.push(path);
    }

    /// The same of the last frame (`--frames`), as `--screenshot` writes it.
    pub fn overlay_shot_at_end(&mut self, path: PathBuf) {
        self.final_overlay_shot = Some(path);
    }

    fn stop_recording(&mut self) -> Result<()> {
        if let Some(recorder) = self.input_recorder.take() {
            recorder.finish()?;
            if let Some((path, from)) = self.recording_input.take() {
                let frames = self.timeline.position().saturating_sub(from);
                devlog::note(Source::Session, format!("recorded {frames} frames of input to {}", path.display()));
            }
        }
        Ok(())
    }

    /// Records the input to `path` from here, once the guest is idle. A
    /// replay starts by restoring the recording's moment, so the game is
    /// restored from it here too (as a rewind does, the clock running on):
    /// what it keeps outside its state then starts over in both alike, and
    /// the replay gives the packets this run does.
    fn start_recording(&mut self, window: Option<&Window>, path: PathBuf) -> Result<()> {
        self.stop_recording()?;
        let moment = self.runtime.moment()?;
        let position = self.timeline.position();
        self.input_recorder = Some(recording::Writer::create(&path, position, &moment)?);
        let shell = shell!(self, window);
        let shown = if self.timeline.history() {
            // Rewinds after it then restore from no earlier than its start.
            self.timeline.keep_moment(moment);
            self.timeline.restart(&mut self.runtime, &shell, position)?.0
        } else {
            self.runtime.restore(&mut Quiet(&shell), &moment)?
        };
        self.runtime.continue_clock(self.clock.now());
        self.show(shown);
        devlog::info(Source::Session, format!("recording input to {} from frame {position}", path.display()));
        self.recording_input = Some((path, position));
        Ok(())
    }

    /// Going back to `at`: the input recording goes back too, or ends if
    /// it started later.
    fn rewind_recording(&mut self, at: u64) -> Result<()> {
        if let Some((path, from)) = &self.recording_input
            && at < *from
        {
            devlog::info(Source::Session, format!("stopped recording {}: went back before it started", path.display()));
            self.stop_recording()?;
        }
        if let Some(recorder) = &mut self.input_recorder {
            recorder.rewind(at)?;
        }
        Ok(())
    }

    /// The frame a rewind queued now starts from.
    fn target(&self) -> u64 {
        self.transport.target(self.timeline.position())
    }

    fn hotkey(&mut self, hotkey: Hotkey) {
        match hotkey {
            Hotkey::Pause if self.transport.paused() => {
                self.resume();
                devlog::info(Source::Session, "playing");
            }
            Hotkey::Pause => {
                self.pause();
                let status = self.timeline();
                let back = if status.history { ", Ctrl+Shift+comma back" } else { "" };
                devlog::info(
                    Source::Session,
                    format!("paused at frame {} (Ctrl+Shift+period steps{back}, Ctrl+Shift+P plays)", status.frame),
                );
            }
            Hotkey::Step => self.step_frames(1),
            Hotkey::Back => self.rewind_to(self.target().saturating_sub(1)),
            Hotkey::Replay => self.replay_recent(REPLAY_SECONDS),
        }
    }

    fn travel(&mut self, travel: Travel) {
        match travel {
            Travel::Pause => self.pause(),
            Travel::Resume => self.resume(),
            Travel::Step(frames) => self.step_frames(frames),
            Travel::Rewind(frames) => self.rewind_to(self.target().saturating_sub(frames)),
            Travel::Replay(seconds) => self.replay_recent(seconds),
        }
    }

    /// Whether the game plays the player's input (not recorded frames).
    fn live(&self) -> bool {
        self.transport.replaying().is_none() && self.timeline.next().is_none()
    }

    /// Goes to `target`, paused.
    fn rewind(&mut self, window: Option<&Window>, target: u64) -> Result<()> {
        let started = Instant::now();
        let shell = shell!(self, window);
        let trace = &mut self.trace;
        // The trace shows the frames run again unseen too: a replay of the
        // input recording runs them as they ran last.
        let mut unseen = |position: u64, recorded: &Recorded, runtime: &timeline::Runtime| -> Result<()> {
            if let Some(trace) = trace {
                writeln!(trace, "{position} {:016x} {:016x}", hash_lanes(&recorded.input), runtime.packet_hash())?;
            }
            Ok(())
        };
        let (shown, at) = self.timeline.rewind(&mut self.runtime, &shell, target, &mut unseen)?;
        self.services.forget();
        let position = self.timeline.position();
        self.runtime.continue_clock(self.clock.now());
        self.show(shown);
        self.pacer = None;
        // The recording goes back to the moment restored, and on again
        // from there marked as restored.
        self.rewind_recording(at)?;
        if let Some(recorder) = &mut self.input_recorder {
            for frame in at..position {
                recorder.frame(self.timeline.recorded(frame).context("a rewound frame is gone")?)?;
            }
        }
        let ahead = self.timeline.head() - position;
        devlog::info(
            Source::Session,
            format!(
                "at frame {position} in {:.0} ms ({ahead} recorded frames ahead: Ctrl+Shift+period runs them, \
                 Ctrl+Shift+P plays on from here)",
                started.elapsed().as_secs_f64() * 1000.0
            ),
        );
        Ok(())
    }

    /// Plays the last `seconds` of recorded frames again.
    fn replay(&mut self, window: Option<&Window>, seconds: f64) -> Result<()> {
        let from = self.timeline.seconds_back(seconds);
        let shell = shell!(self, window);
        let (shown, at) = self.timeline.restart(&mut self.runtime, &shell, from)?;
        self.services.forget();
        self.show(shown);
        self.transport.replay(Replaying::Recent);
        self.clock.resume();
        self.pacer = None;
        self.rewind_recording(at)?;
        let frames = self.timeline.head() - at;
        devlog::info(Source::Session, format!("replaying {frames} frames from frame {at} on the running build"));
        Ok(())
    }

    /// A replay's frames ran out; returns whether the run ends.
    fn replay_ended(&mut self) -> bool {
        let replaying = self.transport.replay_ended();
        self.runtime.continue_clock(self.clock.now());
        let position = self.timeline.position();
        match replaying {
            Some(Replaying::File) if self.options.frames == 0 && self.options.headless => {
                devlog::info(Source::Session, format!("the replay ended at frame {position}"));
                true
            }
            Some(Replaying::File) if self.options.frames == 0 => {
                self.pause();
                devlog::info(
                    Source::Session,
                    format!("the replay ended at frame {position}; paused (Ctrl+Shift+P plays on, live)"),
                );
                false
            }
            Some(_) => {
                devlog::info(Source::Session, format!("the replay ended at frame {position}; playing live"));
                false
            }
            None => false,
        }
    }

    /// One frame of the script: its input goes to the game, the rest is
    /// done around the frame.
    fn tick(&mut self) -> Tick {
        let viewport = Viewport { width: self.gpu.size.0, height: self.gpu.size.1 };
        let mut tick = self.script.as_mut().map(|script| script.tick(viewport)).unwrap_or_default();
        for event in tick.events.drain(..) {
            let game = self.overlay.as_mut().is_none_or(|overlay| overlay.script_event(&event));
            if game {
                self.input.enqueue(event);
            }
        }
        if let Some(overlay) = &mut self.overlay {
            for line in tick.overlay.drain(..) {
                overlay.command(line);
            }
        }
        tick
    }

    /// The script ran out (headless, without --frames): stop.
    fn script_over(&self) -> bool {
        self.script.as_ref().is_some_and(Script::done) && self.options.headless && self.options.frames == 0
    }

    /// Waits until a frame's game time has come (a window plays at the
    /// game's speed).
    fn pace(&mut self, elapsed: f64) {
        let (started, time) = self.pacer.get_or_insert_with(|| (Instant::now(), 0.0));
        *time += elapsed;
        let due = *started + std::time::Duration::from_secs_f64(*time);
        if let Some(wait) = due.checked_duration_since(Instant::now()) {
            std::thread::sleep(wait);
        }
    }

    /// The next frame's input (recorded, or the player's) and its script
    /// tick's requests. Keeps a moment first when the history wants one,
    /// and restores where a replayed run was restored.
    fn plan(&mut self, window: Option<&Window>, tick: Tick) -> Result<Planned> {
        let shell = shell!(self, window);
        self.timeline.prepare(&mut self.runtime, &shell)?;
        self.frames += 1;
        let last = (self.options.frames != 0 && self.frames >= self.options.frames) || tick.quit || self.script_over();
        let viewport = Viewport { width: self.gpu.size.0, height: self.gpu.size.1 };
        let (input, replayed) = match self.timeline.next() {
            Some(recorded) => {
                // Recorded frames bring their own input.
                let planned = (recorded.input.clone(), Some(recorded.loads.clone()));
                self.input.discard();
                planned
            }
            None => {
                let frame = self.input.next_frame(self.clock.advance(), viewport);
                (self.runtime.input(&frame)?, None)
            }
        };
        if window.is_some() && (self.clock.is_steady() || replayed.is_some()) {
            self.pace(f64::from(input[0]));
        }
        let mut shots: Vec<PathBuf> =
            tick.shots.iter().map(|name| self.paths.shots.join(format!("{name}.png"))).collect();
        if last && let Some(path) = &self.options.screenshot {
            shots.push(path.clone());
        }
        let overlay_shots = self.final_overlay_shot.iter().filter(|_| last).cloned().collect();
        Ok(Planned {
            position: self.timeline.position(),
            input,
            replayed,
            shots,
            overlay_shots,
            saves: tick.saves,
            loads: tick.loads,
            stop: last,
        })
    }

    /// Whether the next frame can go to the guest before this one draws:
    /// nothing this frame does afterwards needs the guest or changes its
    /// state, and the next frame is the player's.
    fn overlaps(&self, planned: &Planned, rendered: &RenderFrame) -> bool {
        self.overlap.enabled
            && !planned.stop
            && planned.replayed.is_none()
            && planned.saves.is_empty()
            && planned.loads.is_empty()
            && !self.transport.paused()
            && self.live()
            && self.build.is_none()
            && !self.transport.travelling()
            // Its requests' answers go to the next frame, as without overlap.
            && rendered.services.is_empty()
            && !(rendered.checkpoint_requested
                || rendered.undo_requested
                || rendered.redo_requested
                || rendered.save_requested
                || rendered.load_requested
                || rendered.scene_save_requested
                || rendered.scene_load_requested
                || rendered.exit_requested)
    }

    /// A frame ran: into the history, the input recording and the trace.
    fn ran(&mut self, position: u64, input: Vec<f32>, loads: Vec<Load>) -> Result<()> {
        debug_assert_eq!(position, self.timeline.position());
        if let Some(trace) = &mut self.trace {
            writeln!(trace, "{position} {:016x} {:016x}", hash_lanes(&input), self.runtime.packet_hash())?;
        }
        let recorded = Recorded { input, loads, restored: self.timeline.restored_here() };
        if let Some(recorder) = &mut self.input_recorder {
            recorder.frame(&recorded)?;
        }
        self.timeline.ran(recorded);
        Ok(())
    }

    fn reload(&mut self, window: Option<&Window>, module: GuestModule, catalog: Catalog) {
        let mut shell = shell!(self, window);
        if self.runtime.reload(&mut shell, module, catalog) == ReloadResult::Activated {
            self.timeline.rebuilt();
            if let Some(drawn) = self.runtime.take_drawn() {
                self.show(drawn);
            }
            if self.transport.reloaded() {
                self.resume();
            }
            if self.timeline.history() {
                devlog::info(
                    Source::Reload,
                    format!("dev: Ctrl+Shift+R replays the last {REPLAY_SECONDS} s on the new build"),
                );
            }
        }
    }

    /// Takes reloads as they come: assets at once, a new build when the
    /// guest is idle.
    fn poll_dev(&mut self) {
        let updates = match &mut self.dev {
            Some(dev) => dev.poll(&self.engine),
            None => Vec::new(),
        };
        for update in updates {
            match update {
                Update::Build(module, catalog) => {
                    self.gpu.renderer.preload(&catalog);
                    if let Some(audio) = &mut self.audio {
                        audio.preload(&catalog);
                    }
                    self.build = Some((module, catalog));
                }
                Update::Assets(keys) => {
                    self.gpu.renderer.reload(&keys);
                    if let Some(audio) = &mut self.audio {
                        audio.reload(&keys);
                    }
                }
            }
        }
    }

    /// Paused: the last frame again (without its sounds, and not into a
    /// recording), and the script's shots of it.
    fn present_paused(&mut self, window: Option<&Window>, tick: Tick) -> Result<bool> {
        self.frames += 1;
        let last = (self.options.frames != 0 && self.frames >= self.options.frames) || tick.quit || self.script_over();
        for name in &tick.shots {
            self.capture.screenshots.push(self.paths.shots.join(format!("{name}.png")));
        }
        if last && let Some(path) = &self.options.screenshot {
            self.capture.screenshots.push(path.clone());
        }
        if last && let Some(path) = &self.final_overlay_shot {
            self.capture.overlay_shots.push(path.clone());
        }
        self.present_last(window)?;
        Ok(!last)
    }

    /// Shows the last frame again and writes the shots waiting for it. Its
    /// image is drawn once, into the still texture, and copied to the window
    /// after that, with the overlay drawn over the copy as it changes. It is
    /// drawn again when the frame's size changes, and while the renderer
    /// loads or fades in assets (an asset reloaded while paused shows).
    fn present_last(&mut self, window: Option<&Window>) -> Result<()> {
        let Some(frame) = &self.last else { return Ok(()) };
        let gpu = &mut self.gpu;
        let current = self.still
            && gpu.still.as_ref().is_some_and(|still| (still.width(), still.height()) == gpu.size)
            && gpu.renderer.quiet()
            && gpu.renderer.pending() == 0;
        let capture = &mut self.capture;
        let shots = !capture.screenshots.is_empty() || !capture.overlay_shots.is_empty();
        if current && gpu.surface.is_none() && !shots {
            return Ok(());
        }
        let texture = gpu.still_texture();
        let output = Output {
            device: &gpu.device,
            queue: &gpu.queue,
            surface: gpu.surface.as_ref(),
            size: gpu.size,
            blitter: gpu.blitter.as_ref(),
            overlay: self.overlay.as_mut(),
        };
        let mut target = CaptureTarget { output, texture: texture.clone() };
        if current {
            target.present();
        } else {
            gpu.renderer.submit(frame, self.runtime.catalog(), &mut target)?;
            self.still = true;
        }
        for path in capture.overlay_shots.drain(..) {
            write_overlay_shot(gpu, &texture, self.overlay.as_mut(), &path)?;
        }
        if !capture.screenshots.is_empty() {
            let pixels = gpu.read(&texture)?;
            for path in capture.screenshots.drain(..) {
                write_png(&path, &pixels, texture.width(), texture.height(), gpu.bgra())?;
            }
        }
        show_cursor(window, frame, self.overlay.as_ref().is_some_and(|overlay| overlay.wants_pointer()));
        match gpu.take_fault() {
            Some(fault) => Err(anyhow!(fault)),
            None => Ok(()),
        }
    }

    /// Invalid guest output never commits its candidate state. Keep the last
    /// valid frame visible and the development watcher running until a fix.
    fn reject_frame(&mut self, window: Option<&Window>, planned: Planned, error: anyhow::Error) -> Result<bool> {
        if !self.options.dev || !error.is::<InvalidFrame>() {
            return Err(error);
        }
        let was_playing = self.transport.reject();
        self.stopped(was_playing);
        self.timings = None;
        self.input.discard();
        self.input.enqueue(GameEvent::FocusLost);
        // A program stopped at its deadline is gone: a new one restores the last frame's state.
        if error.is::<Unresponsive>() {
            self.transport.restore_at(self.timeline.position());
        }
        devlog::error(
            Source::Game,
            format!(
                "dev: frame {} rejected: {error:#}\ngame paused; fix the source to reload, or use Play to retry",
                planned.position,
            ),
        );
        self.capture.screenshots.extend(planned.shots);
        self.capture.overlay_shots.extend(planned.overlay_shots);
        self.present_last(window)?;
        Ok(!planned.stop)
    }

    /// Runs one frame (or, paused, shows the last again); returns whether
    /// to keep going.
    pub fn step(&mut self, window: Option<&Window>) -> Result<bool> {
        if self.gpu.size.0 == 0 || self.gpu.size.1 == 0 {
            return Ok(true);
        }
        // Laid out before the frame's time starts: its drawing counts in render+present.
        if let Some(mut overlay) = self.overlay.take() {
            overlay.update(self, window);
            self.overlay = Some(overlay);
        }
        let started = Instant::now();
        self.poll_dev();
        for event in self.gpu.renderer.take_events() {
            // While developing, say when a reloaded asset is in (or why not).
            if self.dev.is_some() {
                match &event {
                    GameEvent::AssetReady { key, revision, .. } if *revision > 1 => {
                        devlog::info(Source::Asset, format!("dev: {key} is in (revision {revision})"));
                    }
                    GameEvent::AssetFailed { key, message, .. } => {
                        devlog::error(Source::Asset, format!("dev: {key} failed: {message}"));
                    }
                    _ => {}
                }
            }
            self.held.push(event);
        }
        self.services.frame(&mut self.input);
        if self.live() {
            for event in self.held.drain(..) {
                self.input.enqueue(event);
            }
            for event in self.services.take_answers() {
                self.input.enqueue(event);
            }
        }
        for hotkey in self.input.take_hotkeys() {
            self.hotkey(hotkey);
        }
        let planned = match self.overlap.flight.take() {
            Some(planned) => planned,
            None => {
                // The guest is idle: what waited for it happens now.
                if let Some((module, catalog)) = self.build.take() {
                    self.reload(window, module, catalog);
                }
                let tick = match self.overlap.early.take() {
                    Some(tick) => tick,
                    None => self.tick(),
                };
                if let Some(travel) = tick.travel {
                    self.travel(travel);
                }
                while let Some(rewind) = self.transport.next_rewind() {
                    let result = match rewind {
                        Rewind::To(frame) => self.rewind(window, frame),
                        Rewind::Replay(seconds) => self.replay(window, seconds),
                    };
                    if let Err(error) = result {
                        devlog::error(Source::Session, format!("time travel: {error:#}"));
                    }
                }
                if let Some(path) = self.transport.take_record()
                    && let Err(error) = match path {
                        Some(path) => self.start_recording(window, path),
                        None => self.stop_recording(),
                    }
                {
                    devlog::error(Source::Session, format!("recording input: {error:#}"));
                }
                if !self.transport.take_frame() {
                    return self.present_paused(window, tick);
                }
                let planned = self.plan(window, tick)?;
                self.runtime.send(&planned.input)?;
                planned
            }
        };
        let rendered = {
            let shell = shell!(self, window);
            self.runtime.receive(&shell)
        };
        let rendered = match rendered {
            Ok(rendered) => rendered,
            Err(error) => return self.reject_frame(window, planned, error),
        };
        let overlapped = self.overlaps(&planned, &rendered);
        let mut planned = planned;
        if overlapped {
            // This frame changes nothing after it draws: the next one goes
            // to the guest now and computes while this one draws.
            self.ran(planned.position, std::mem::take(&mut planned.input), Vec::new())?;
            let tick = self.tick();
            if tick.travel.is_some() {
                self.overlap.early = Some(tick);
            } else {
                let next = self.plan(window, tick)?;
                self.runtime.send(&next.input)?;
                self.overlap.flight = Some(next);
            }
        }
        let stop = self.complete(window, planned, rendered, overlapped, started)?;
        Ok(!stop)
    }

    /// Draws a frame and does what it and its script tick asked; returns
    /// whether the run ends.
    fn complete(
        &mut self,
        window: Option<&Window>,
        planned: Planned,
        rendered: RenderFrame,
        overlapped: bool,
        started: Instant,
    ) -> Result<bool> {
        self.capture.screenshots.extend(planned.shots);
        self.capture.overlay_shots.extend(planned.overlay_shots);
        let submitting = Instant::now();
        let mut loads = Vec::new();
        let submit_ms;
        {
            let mut shell = shell!(self, window);
            shell.render(&rendered, self.runtime.catalog())?;
            submit_ms = submitting.elapsed().as_secs_f64() * 1000.0;
            if let Some(fault) = shell.gpu.take_fault() {
                return Err(anyhow!(fault));
            }
            // A live frame's requests (a recorded one's answers are in the
            // recording), before an undo or load replaces the world asking.
            if planned.replayed.is_none() {
                shell.services.answer(&rendered.services);
            }
            self.runtime.follow(&mut shell, &rendered)?;
            match &planned.replayed {
                // The states it loaded, not the files: they may have changed
                // since. A replay writes no files.
                Some(recorded) => {
                    for load in recorded {
                        timeline::apply(&mut self.runtime, &mut shell, load)?;
                    }
                    loads.clone_from(recorded);
                }
                None => {
                    let paths = &self.paths;
                    if rendered.scene_save_requested {
                        report(self.runtime.save_scene(&paths.scene), format!("saved scene {}", paths.scene.display()));
                    }
                    if rendered.scene_load_requested {
                        loads.extend(load_file(&mut self.runtime, &mut shell, paths.scene_to_read(), true));
                    }
                    if rendered.save_requested {
                        report(self.runtime.save(&paths.quicksave), format!("saved {}", paths.quicksave.display()));
                    }
                    if rendered.load_requested {
                        loads.extend(load_file(&mut self.runtime, &mut shell, &paths.quicksave, false));
                    }
                    for path in &planned.saves {
                        report(self.runtime.save(path), format!("saved {}", path.display()));
                    }
                    for path in &planned.loads {
                        loads.extend(load_file(&mut self.runtime, &mut shell, path, false));
                    }
                }
            }
        }
        if !overlapped {
            self.ran(planned.position, planned.input, loads)?;
        }
        let exit = rendered.exit_requested;
        let shown = self.runtime.take_drawn().unwrap_or(rendered);
        self.show(shown);
        let mut stop = exit || planned.stop;
        if self.transport.replaying().is_some() && self.timeline.next().is_none() {
            stop |= self.replay_ended();
        }
        // Only while measured (a profile, or the dev tools' profiler).
        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.timings = self.runtime.last_timings().map(|timings| timings.completed(submit_ms, total_ms));
        if self.frame_times
            && let Some(timing) = &self.timings
        {
            profile::print(self.frames, timing);
        }
        if let Some(profile) = &mut self.profile
            && profile.sample(self.frames, self.timings)
        {
            profile.report(self.overlap.enabled);
            return Ok(true);
        }
        Ok(stop)
    }

    /// Stops the game and finishes any recording; the GPU, with its loaded
    /// assets, can start another session.
    pub fn finish(mut self) -> Result<Gpu> {
        self.runtime.close();
        self.services.report();
        if let Some(recorder) = self.input_recorder.take() {
            recorder.finish()?;
        }
        if let Some(mut trace) = self.trace.take() {
            trace.flush()?;
        }
        if let Some(mut recorder) = self.capture.recorder.take() {
            for (pixels, (width, height)) in self.capture.readback.finish(&self.gpu.device)? {
                recorder.picture(pixels, width, height)?;
            }
            recorder.finish()?;
        }
        Ok(self.gpu)
    }
}

#[cfg(all(test, feature = "compiler"))]
mod recovery_tests;
