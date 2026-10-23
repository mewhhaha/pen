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
//! preferences (`preferences`), action bindings (`bindings`), accessibility
//! audits (`audit`) and frame measurements (`profile`); the game's clock
//! (`pen_runtime::clock`) and what was asked of time (`pen_runtime::transport`)
//! are the runtime's, which the browser shares.
mod audit;
mod bindings;
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
use pen_runtime::captions::{self, Captions, CaptionsService};
use pen_runtime::clock::Clock;
use pen_runtime::devtools::TimelineStatus;
use pen_runtime::state::{self, StateAdapter};
#[cfg(all(test, feature = "compiler"))]
use pen_runtime::transport;
use pen_runtime::transport::{Replaying, Rewind, Transport};
use std::sync::Arc;
use winit::event::WindowEvent;
use winit::window::Window;

use crate::audio::Audio;
use crate::bugreport;
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
    /// The game's actions and the player's bindings (`bindings`).
    pub bindings: PathBuf,
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
            bindings: data.join("bindings.json"),
            shots: shots.unwrap_or_else(|| PathBuf::from("shots")),
            game,
        }
    }

    pub fn scene_to_read(&self) -> &Path {
        if self.scene.exists() { &self.scene } else { &self.bundled_scene }
    }

    /// Whether `path` is one of the player's own saves in the data
    /// directory, the quick save and their scene, which are stamped and
    /// backed up (`saves`).
    fn kept(&self, path: &Path) -> bool {
        path == self.quicksave || path == self.quicksave.with_file_name("main.scene")
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
    /// Whether it wants a picture of the frame at `position` (the timeline's).
    fn wants_picture(&self, _position: u64) -> bool {
        false
    }
    /// The frame at `position`, drawn offscreen into `texture`, for its
    /// pictures.
    fn picture(&mut self, _device: &wgpu::Device, _queue: &wgpu::Queue, _texture: &wgpu::Texture, _position: u64) {}
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
    /// The overlay's picture of the frame at this position.
    picture: Option<u64>,
    /// Frames the accessibility audit reads back (`audit`).
    audits: audit::Audits,
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
    state_adapter: &'a Arc<dyn StateAdapter>,
}

impl Host for Shell<'_> {
    fn viewport(&self) -> Viewport {
        Viewport { width: self.gpu.size.0.max(1), height: self.gpu.size.1.max(1) }
    }

    fn validate(&self, frame: &RenderFrame) -> Result<()> {
        pen_render::validate_frame(frame)
    }

    fn render(&mut self, frame: &RenderFrame, catalog: &Catalog) -> Result<()> {
        let started = Instant::now();
        self.gpu.draw_timings = Default::default();
        self.preferences.apply(&mut self.gpu.renderer)?;
        let capture = &mut *self.capture;
        let audited = capture.audits.due(&frame.ui);
        let offscreen = capture.offscreen
            || !capture.screenshots.is_empty()
            || capture.recorder.is_some()
            || !capture.overlay_shots.is_empty()
            || capture.picture.is_some()
            || audited;
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
            timings: &mut gpu.draw_timings,
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
        gpu.draw_timings.render_ms = (started.elapsed().as_secs_f64() * 1000.0 - gpu.draw_timings.total_ms()).max(0.0);
        if let Some(texture) = texture.as_ref().filter(|_| audited) {
            capture.audits.capture(gpu, texture, &frame.ui)?;
        }
        crate::startup::drawn(&gpu.renderer);
        if let Some(texture) = &texture {
            for path in capture.overlay_shots.drain(..) {
                write_overlay_shot(gpu, texture, self.overlay.as_deref_mut(), &path)?;
            }
        }
        if let (Some(position), Some(texture), Some(overlay)) =
            (capture.picture.take(), &texture, self.overlay.as_deref_mut())
        {
            overlay.picture(&gpu.device, &gpu.queue, texture, position);
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
        self.services.rendered(frame, catalog);
        show_cursor(self.window, frame, self.overlay.as_ref().is_some_and(|overlay| overlay.wants_pointer()));
        self.input.text_field(self.window, &frame.ui);
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

    fn state_adapter(&self) -> Arc<dyn StateAdapter> {
        self.state_adapter.clone()
    }
}

/// Loaded at startup and configuration/build changes, never on tooling queries.
/// Invalid declarations remain explicit errors until the next change.
pub(crate) fn load_state_adapter(assets: &Path) -> Arc<dyn StateAdapter> {
    let path = assets.join(state::CONFIG_FILE);
    match std::fs::read_to_string(&path) {
        Ok(text) => pen_ink::adapter(&text).unwrap_or_else(|error| {
            devlog::error(Source::Game, format!("{}: {error:#}", path.display()));
            state::invalid(format!("{}: {error:#}", path.display()))
        }),
        Err(error) => {
            if error.kind() != std::io::ErrorKind::NotFound {
                let message = format!("{}: {error}", path.display());
                devlog::error(Source::Game, &message);
                return state::invalid(message);
            }
            state::opaque()
        }
    }
}

/// The assets' captions (`captions.json`), none if they have none; a file
/// that does not read is reported and gives none.
fn load_captions(assets: &Path) -> Captions {
    let Ok(json) = std::fs::read_to_string(assets.join("captions.json")) else { return Captions::default() };
    Captions::parse(&json).unwrap_or_else(|error| {
        devlog::error(Source::Asset, format!("{error:#}"));
        Captions::default()
    })
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
    /// The audio device's buffer, in samples (`--audio-buffer`); none for
    /// the system's.
    pub audio_buffer: Option<u32>,
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
    /// Play as one peer of a net session (`--net`, `crate::net`).
    pub net: Option<crate::net::Spec>,
    /// Stream the final frames to viewers on this address (`--broadcast`).
    pub stream: Option<String>,
    /// Play the stream at this address as it comes (`--spectate`).
    pub spectate: Option<String>,
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
    #[cfg(feature = "devtools")]
    pub system_trace: Option<std::sync::Arc<blot_host::SystemTrace>>,
    runtime: GameRuntime,
    state_adapter: Arc<dyn StateAdapter>,
    audio: Option<Audio>,
    pub input: Input,
    preferences: Preferences,
    /// The game's actions, which live frames' events move (`bindings`).
    pub actions: pen_runtime::actions::Shared,
    /// What the game asks of the platform and native libraries: called for
    /// live frames only, their answers the next live frame's input.
    pub services: Services,
    /// The captions of what plays, shared with the captions service.
    captions: captions::Shared,
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
    /// Mods beside the game (`--mod`): they see each live frame's input
    /// and add to it.
    mods: Option<crate::mods::Mods>,
    /// The last frames' packets, while the dev tools look at them.
    #[cfg(feature = "devtools")]
    packets: pen_runtime::inspect::Packets,
    /// The profile the game's guest is sampled into, while profiled.
    #[cfg(feature = "devtools")]
    sampling: Option<std::sync::Arc<blot_host::Profile>>,
    /// A net session this peer plays in (`crate::net`).
    net: Option<crate::net::Net>,
    /// Viewers this session streams to, and a stream it plays (`crate::spectate`).
    stream: Option<crate::spectate::Stream>,
    watch: Option<crate::spectate::Watch>,
}

/// How much game time a history keeps: enough to rewind, and a bug
/// report's worth.
fn history_seconds() -> f64 {
    KEEP_SECONDS.max(bugreport::SECONDS)
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
            state_adapter: &$session.state_adapter,
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
        // What the scene draws first loads first, side by side, while the
        // program starts; the rest of the catalog streams in behind it.
        let first = crate::scene_assets::read(&paths.assets, paths.scene_to_read()).unwrap_or_default();
        if let Some(prefetch) = crate::scene_assets::warmed(&paths.assets) {
            gpu.renderer.warm(prefetch);
        }
        gpu.renderer.preload_scene(&first.assets, first.samples);
        gpu.renderer.preload(&catalog);
        // Headless runs and recordings play the sound into the recording, not the speakers.
        let audio = if options.mute || options.headless || options.record.is_some() {
            None
        } else {
            Audio::new(paths.assets.clone(), &catalog, &first.sounds, options.audio_buffer)
                .map_err(|error| eprintln!("audio is off: {error:#}"))
                .ok()
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
        let mut capture =
            Capture { recorder, offscreen: options.headless, audits: audit::Audits::from_env()?, ..Capture::default() };
        let mut input = Input::default();
        let mut preferences = Preferences::load(paths.preferences.clone());
        if preferences.untouched() {
            let adapter = gpu.device.adapter_info();
            preferences.preset(crate::presets::first_run(&paths.assets, &adapter, options.headless));
        }
        preferences.replay(&mut input);
        let mut services = crate::services::standard();
        let actions = bindings::load(&paths.bindings);
        services.register(Box::new(pen_runtime::actions::InputService(actions.clone())));
        services.register(Box::new(preferences.service()));
        if let Some(locale) = crate::services::locale::service(&paths.assets) {
            services.register(Box::new(locale));
        }
        let captions = Captions::shared(load_captions(&paths.assets));
        services.register(Box::new(CaptionsService(captions.clone())));
        crate::saves::check_all([&paths.quicksave, &paths.scene].into_iter().filter(|path| paths.kept(path)));
        #[cfg(feature = "devtools")]
        let system_trace = module.system_trace();
        let dev = if options.dev { Some(Dev::new(&paths.game, &paths.assets, module.system_trace())?) } else { None };
        let mut audio = audio;
        let state_adapter = load_state_adapter(&paths.assets);
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
                state_adapter: &state_adapter,
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
        crate::startup::mark("game's first frame");
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
        // A window keeps one for Ctrl+Shift+B's bug reports too.
        if options.dev || script::travels_back(&options.script) || !options.headless {
            timeline.keep_history(Some(history_seconds()));
        }
        Ok(Session {
            #[cfg(feature = "devtools")]
            system_trace,
            options,
            paths,
            engine,
            gpu,
            runtime,
            state_adapter,
            audio,
            input,
            preferences,
            actions,
            services,
            captions,
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
            mods: None,
            #[cfg(feature = "devtools")]
            packets: pen_runtime::inspect::Packets::default(),
            #[cfg(feature = "devtools")]
            sampling: None,
            net: None,
            stream: None,
            watch: None,
        })
    }

    /// Sets up what the session does besides playing; before the first step.
    pub fn playback(&mut self, playback: Playback, window: Option<&Window>) -> Result<()> {
        self.overlap.enabled = playback.overlap;
        if playback.history || playback.replay.is_some() {
            self.timeline.keep_history(Some(history_seconds()));
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
        if let Some(spec) = &playback.net {
            self.start_net(spec)?;
        }
        if let Some(address) = &playback.stream {
            self.start_stream(address)?;
        }
        if let Some(address) = &playback.spectate {
            self.start_watch(address, window)?;
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
        if self.travel_refused() {
            return;
        }
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
        if self.travel_refused() {
            return;
        }
        let was_playing = self.transport.step(frames);
        self.stopped(was_playing);
    }

    /// Goes to frame `frame` (pausing there) by restoring the nearest
    /// moment before it and running the recorded frames again unseen.
    /// Takes effect on the next step.
    pub fn rewind_to(&mut self, frame: u64) {
        if self.travel_refused() {
            return;
        }
        let was_playing = self.transport.rewind_to(frame);
        self.stopped(was_playing);
    }

    /// Plays the last `seconds` of recorded input again, on whatever build
    /// runs now (after a hot reload, the new one), then plays on live.
    /// Takes effect on the next step.
    pub fn replay_recent(&mut self, seconds: f64) {
        if self.travel_refused() {
            return;
        }
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

    pub fn set_mods(&mut self, mods: crate::mods::Mods) {
        self.mods = Some(mods);
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

    /// The accessibility audit of the last frame read back for it (or of the
    /// interface now, without its pixels), and the next frame read back.
    #[cfg(feature = "devtools")]
    pub fn audit(&mut self) -> Vec<pen_runtime::audit::Finding> {
        let ui = self.interface().to_vec();
        self.capture.audits.ask(&ui)
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

    /// The captions of the sounds playing now, newest first, a line each;
    /// only when the game requests them in its accessibility presentation.
    pub fn captions(&self) -> Option<String> {
        let state = captions::lock(&self.captions);
        state.visible(self.last.as_ref().is_some_and(|frame| frame.show_captions))
    }

    /// The sound playing, unless the session is silent.
    #[cfg(feature = "devtools")]
    pub fn audio(&mut self) -> Option<&mut Audio> {
        self.audio.as_mut()
    }

    #[cfg(feature = "devtools")]
    /// The game's linear memory, in bytes.
    pub fn guest_memory(&self) -> usize {
        self.runtime.memory_bytes()
    }

    #[cfg(feature = "devtools")]
    /// The adapter selected at startup or the latest configuration/build reload.
    pub fn state_adapter(&self) -> Arc<dyn StateAdapter> {
        self.state_adapter.clone()
    }

    /// The running game's shared state; unavailable during an overlapped frame
    /// or when the guest cannot serialize it.
    #[cfg(feature = "devtools")]
    pub fn snapshot(&mut self) -> Option<Arc<SavedState>> {
        if self.overlap.flight.is_some() {
            return None;
        }
        self.runtime.shared_snapshot().map_err(|error| devlog::error(Source::Session, format!("{error:#}"))).ok()
    }

    #[cfg(feature = "devtools")]
    /// Keeps the last seconds to travel in from now on, as `--history` does
    /// from the start.
    pub fn keep_history(&mut self) {
        if !self.timeline.history() {
            self.timeline.keep_history(Some(history_seconds()));
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

    #[cfg(feature = "devtools")]
    /// The last frames' packets, kept while the dev tools look at them.
    pub fn packets(&mut self) -> &mut pen_runtime::inspect::Packets {
        &mut self.packets
    }

    #[cfg(feature = "devtools")]
    pub fn observations(&mut self) -> Option<&mut pen_runtime::semantic::Observations> {
        self.net.is_none().then_some(&mut self.timeline.observations)
    }

    #[cfg(feature = "devtools")]
    /// The state the history kept `seconds` of game time back, at the
    /// latest moment by then, and the frame it was kept at.
    pub fn kept_state(&self, seconds: f64) -> Option<(u64, SavedState)> {
        let position = self.timeline.frames_before(self.timeline.position(), seconds);
        self.timeline.moment_before(position).map(|(at, moment)| (at, moment.saved.clone()))
    }

    #[cfg(feature = "devtools")]
    /// The profile the game's guest is sampled into from the next frame
    /// on, and the engine it runs on (whose epoch a profile ticks).
    pub fn profile(&mut self) -> (&mut Option<std::sync::Arc<blot_host::Profile>>, &wasmtime::Engine) {
        (&mut self.sampling, &self.engine)
    }

    #[cfg(feature = "devtools")]
    /// What the game's guest showed its probe: its memory and counters.
    pub fn guest_seen(&self) -> blot_host::Seen {
        self.runtime.program().probe().seen()
    }

    #[cfg(feature = "devtools")]
    /// The game's catalog, for the dev tools to draw an entry from another
    /// asset until the next build.
    pub fn catalog_mut(&mut self) -> &mut Catalog {
        self.runtime.catalog_mut()
    }

    #[cfg(feature = "devtools")]
    /// The game's assets directory.
    pub fn assets(&self) -> &Path {
        &self.paths.assets
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

    /// Writes a bug report (`bugreport`): the history, the state now and
    /// the log, to `path`, or (empty) to `reports/` in the data directory.
    fn bug_report(&mut self, path: PathBuf) {
        let written = self.runtime.snapshot().and_then(|state| {
            let capture = bugreport::Capture {
                recording: self.timeline.recording(),
                state,
                game: &self.paths.game,
                position: self.timeline.position(),
            };
            let data = self.paths.quicksave.parent().unwrap_or(Path::new("."));
            bugreport::write(&capture, (!path.as_os_str().is_empty()).then_some(path), data)
        });
        match written {
            Ok(path) => devlog::info(
                Source::Session,
                format!("bug report written to {} (`pen report FILE` says what it holds)", path.display()),
            ),
            Err(error) => devlog::error(Source::Session, format!("bug report: {error:#}")),
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
        if let Some(net) = &mut self.net {
            net.play.prepare(&mut self.timeline);
        }
        self.timeline.prepare(&mut self.runtime, &shell)?;
        self.frames += 1;
        let last = (self.options.frames != 0 && self.frames >= self.options.frames)
            || tick.quit
            || self.script_over()
            || self.watched_to_the_end();
        let viewport = Viewport { width: self.gpu.size.0, height: self.gpu.size.1 };
        let (input, replayed) = match self.timeline.next() {
            Some(recorded) => {
                // Recorded frames bring their own input.
                let planned = (recorded.input.clone(), Some(recorded.loads.clone()));
                self.input.discard();
                planned
            }
            None => {
                let mut frame = self.input.next_frame(self.clock.advance(), viewport);
                self.actions.borrow_mut().map(&mut frame.events);
                let mut input = match &mut self.net {
                    // Every player's input, as far as it is known.
                    Some(net) => net.play.frame(self.timeline.position(), &frame.events, viewport)?,
                    None => self.runtime.input(&frame)?,
                };
                if let Some(mods) = &mut self.mods {
                    mods.frame(&mut input);
                }
                (input, None)
            }
        };
        // Headless on the AT-SPI bus, too: assistive technology keeps up.
        let paced = window.is_some() || crate::accessibility::paces_headless();
        if paced && (self.clock.is_steady() || replayed.is_some()) {
            self.pace(f64::from(input[0]));
        }
        let mut shots: Vec<PathBuf> =
            tick.shots.iter().map(|name| self.paths.shots.join(format!("{name}.png"))).collect();
        if last && let Some(path) = &self.options.screenshot {
            shots.push(path.clone());
        }
        let named = tick.overlay_shots.iter().map(|name| self.paths.shots.join(format!("{name}.png")));
        let overlay_shots = named.chain(self.final_overlay_shot.iter().filter(|_| last).cloned()).collect();
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
            && !self.timeline.observations.watching()
            && self.net.is_none()
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
        if let Some(net) = &mut self.net {
            net.play.ran(self.runtime.packet_hash());
        }
        self.stream_final();
        Ok(())
    }

    fn reload(&mut self, window: Option<&Window>, module: GuestModule, catalog: Catalog) {
        if self.net.is_some() {
            devlog::error(Source::Reload, "net: the new build waits; the peers play the build they started with");
            return;
        }
        self.state_adapter = load_state_adapter(&self.paths.assets);
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
                        audio.preload(&catalog, &[]);
                    }
                    self.build = Some((module, catalog));
                }
                Update::Assets(keys) => {
                    if keys.iter().any(|key| key == state::CONFIG_FILE) {
                        self.state_adapter = load_state_adapter(&self.paths.assets);
                    }
                    if keys.iter().any(|key| key == "captions.json") {
                        captions::lock(&self.captions).captions = load_captions(&self.paths.assets);
                    }
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
        for name in &tick.overlay_shots {
            self.capture.overlay_shots.push(self.paths.shots.join(format!("{name}.png")));
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
            timings: &mut Default::default(),
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
        // Laid out before the frame's time starts: its drawing counts in the render phase.
        if let Some(mut overlay) = self.overlay.take() {
            overlay.update(self, window);
            self.overlay = Some(overlay);
        }
        // The guest running now (a rewind or reload starts another) is sampled while profiled.
        #[cfg(feature = "devtools")]
        self.runtime.program().probe().attach(self.sampling.clone());
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
        self.input.poll_gamepads();
        self.services.frame(&mut self.input);
        bindings::keep(&self.actions, &self.paths.bindings);
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
                if !self.net_turn(window)? {
                    return Ok(true);
                }
                if let Some(going) = self.watch_turn(window)? {
                    return Ok(going);
                }
                // The guest is idle: what waited for it happens now.
                if let Some((module, catalog)) = self.build.take() {
                    self.reload(window, module, catalog);
                }
                let mut tick = match self.overlap.early.take() {
                    Some(tick) => tick,
                    None => self.tick(),
                };
                let mut reports = std::mem::take(&mut tick.reports);
                if self.input.take_report() {
                    reports.push(PathBuf::new());
                }
                for path in reports {
                    self.bug_report(path);
                }
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
        #[cfg(feature = "devtools")]
        self.packets.ran(planned.position, &planned.input, self.runtime.packet_lanes(), &rendered);
        let overlapped = self.overlaps(&planned, &rendered);
        let mut planned = planned;
        if overlapped {
            // This frame changes nothing after it draws: the next one goes
            // to the guest now and computes while this one draws.
            self.timeline.observe(&rendered, false);
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
        // Tagged with the position that shows it, as the timeline counts.
        if self.overlay.as_ref().is_some_and(|overlay| overlay.wants_picture(planned.position + 1)) {
            self.capture.picture = Some(planned.position + 1);
        }
        let mut loads = Vec::new();
        let draw_timings;
        {
            let mut shell = shell!(self, window);
            shell.render(&rendered, self.runtime.catalog())?;
            draw_timings = shell.gpu.draw_timings;
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
                    let local = self.net.is_none();
                    if !local && (rendered.scene_load_requested || rendered.load_requested || !planned.loads.is_empty())
                    {
                        devlog::error(
                            Source::Session,
                            "net: loads are off in a net session (they read this peer's files)",
                        );
                    }
                    if rendered.scene_save_requested {
                        let kept = paths.kept(&paths.scene);
                        report(
                            self.runtime.save_scene(&paths.scene, kept),
                            format!("saved scene {}", paths.scene.display()),
                        );
                    }
                    if rendered.scene_load_requested && local {
                        loads.extend(load_file(&mut self.runtime, &mut shell, paths.scene_to_read(), true));
                    }
                    if rendered.save_requested {
                        report(
                            self.runtime.save_kept(&paths.quicksave),
                            format!("saved {}", paths.quicksave.display()),
                        );
                    }
                    if rendered.load_requested && local {
                        loads.extend(load_file(&mut self.runtime, &mut shell, &paths.quicksave, false));
                    }
                    for path in &planned.saves {
                        report(self.runtime.save(path), format!("saved {}", path.display()));
                    }
                    for path in planned.loads.iter().filter(|_| local) {
                        loads.extend(load_file(&mut self.runtime, &mut shell, path, false));
                    }
                }
            }
        }
        let exit = rendered.exit_requested;
        let replacement = self.runtime.take_drawn();
        let replaced = replacement.is_some();
        let shown = replacement.unwrap_or(rendered);
        if !overlapped {
            self.timeline.observe(&shown, replaced);
            self.ran(planned.position, planned.input, loads)?;
        }
        self.show(shown);
        if let Some(position) = self.timeline.observations.take_break() {
            let was_playing = self.transport.breakpoint();
            self.stopped(was_playing);
            devlog::info(Source::Session, format!("semantic watch: paused at position {position}"));
        }
        let mut stop = exit || planned.stop;
        if self.transport.replaying().is_some()
            && self.timeline.next().is_none()
            && self.watch.as_ref().is_none_or(crate::spectate::Watch::ended)
        {
            stop |= self.replay_ended();
        }
        // Only while measured (a profile, or the dev tools' profiler).
        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.timings = self.runtime.last_timings().map(|timings| timings.completed(draw_timings, total_ms));
        if self.frame_times
            && let Some(timing) = &self.timings
        {
            profile::print(self.frames, timing, self.gpu.renderer.stats());
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
        self.finish_net()?;
        self.finish_stream();
        self.runtime.close();
        self.services.report();
        bindings::keep(&self.actions, &self.paths.bindings);
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

mod net;
mod spectate;

#[cfg(all(test, feature = "compiler"))]
mod recovery_tests;

#[cfg(all(test, feature = "compiler", feature = "devtools"))]
mod semantic_tests;
