//! The dev tools: the checkout's own program for them (gdev's
//! `devtools/main.blot`: its console and diagnostics), which the runtime runs
//! beside the game in `pen dev` and `pen run`, drawing its sprites over the
//! game's frame. It is a second Blot program with its own state, outside the
//! game's saves, recordings and timeline, so it keeps running while the game
//! is paused, rewinding, failing to build or stopped on an error, and what
//! the runtime tells it (frame timings, the log) never reaches the game.
//!
//! Each frame it runs first, on every window event the game gets and the
//! backquote key, which the game never sees. Its interface says what it takes
//! from the game: the keyboard while one of its nodes has the focus (its
//! console's input), and the pointer and wheel over its dialogs. It asks the
//! runtime for what it shows and does through the `devtools` service
//! (`protocol/schema.json`, DevtoolsOperation): the log, the last frame's
//! numbers and the GPU's name, which it answers in its next frame, and game
//! commands, time travel, input recordings, screenshots and tunables, which it
//! carries out. pen-runtime answers the service as the browser does
//! (`pen_runtime::devtools`); the session is its host here (`Answering`).
//!
//! Its build (`devtools.wasm`, beside the game's) is picked up when it
//! appears and whenever it changes (the session's file watcher says when),
//! with its state carried over; its state is kept in the checkout's
//! `.pen/devtools.state` between windowed sessions. A checkout without the
//! program runs without dev tools.
//!
//! The host draws pictures of the game's frames and previews of its assets
//! into an atlas texture of the dev tools' (`pictures`), samples the game's
//! guest for their profiler and reads its memory and ownership counters
//! (`blot_host::Probe`), keeps its packets and states for their inspectors
//! (`pen_runtime::inspect`), and edits tunable presets and materials in the
//! checkout for them (`tunables`, `materials`).
mod pictures;

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use blot_host::GuestModule;
use pen_protocol::{Catalog, Frame, GameEvent, RenderFrame, UiNode, Viewport, devtools_asset, service};
use pen_runtime::devtools::{
    self as tools, FrameReport, LogLine, Routing, TimelineStatus, button_bit, inside_dialog, log_answer,
};
use pen_runtime::inspect::{self, Packets};
use pen_runtime::mix::AudioReport;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

use crate::dev::watch;
use crate::devlog::{self, Source};
use crate::game::{GameRuntime, Host};
use crate::services::Answer;
use crate::session::{Gpu, Overlay, Session};
use crate::snapshot::{self, SavedState};
use crate::{materials, tunables};

/// The program's build, beside the game's.
pub const PROGRAM: &str = "devtools.wasm";
/// Where its state is kept between windowed sessions, in the checkout.
const STATE: &str = ".pen/devtools.state";
/// How many of the shots and input recordings the dev tools name themselves
/// each of `.pen/shots` and `.pen/recordings` keeps; older ones go.
const KEEP_FILES: usize = 50;

/// How the dev tools start.
pub struct Settings {
    /// The checkout: its program's build directory, and where recordings,
    /// shots and the program's state go.
    pub project: PathBuf,
    /// Where the program's build is (the game's build directory).
    pub build: PathBuf,
    /// The console shows from the start (`--overlay`, PEN_OVERLAY=1).
    pub open: bool,
    /// Keep the program's state for the next session (windowed runs).
    pub keep: bool,
    /// Answer remote dev tools here too (`--devtools-listen`).
    pub listen: Option<String>,
}

/// Whether to show the console from the start: `--overlay`, or PEN_OVERLAY
/// set to anything but 0.
pub fn requested(flag: bool) -> bool {
    flag || std::env::var_os("PEN_OVERLAY").is_some_and(|value| value != "0")
}

/// What the program's runtime needs of its host: the viewport, and checks.
/// Its frames are drawn over the game's by `Devtools::paint`, not here.
struct Shell {
    viewport: Viewport,
}

impl Host for Shell {
    fn viewport(&self) -> Viewport {
        self.viewport
    }

    fn validate(&self, frame: &RenderFrame) -> Result<()> {
        pen_render::validate_frame(frame)
    }

    fn render(&mut self, _frame: &RenderFrame, _catalog: &Catalog) -> Result<()> {
        Ok(())
    }

    fn report(&mut self, message: &str) {
        devlog::note(Source::Reload, format!("devtools: {message}"));
    }
}

/// The dev tools program, run beside the game (`session::Overlay`).
pub struct Devtools {
    engine: wasmtime::Engine,
    catalog: Catalog,
    build: PathBuf,
    project: PathBuf,
    keep: bool,
    /// The program, once its build is there and starts.
    runtime: Option<GameRuntime>,
    /// The build it runs (its file's time), and a newer one compiling.
    built: Option<SystemTime>,
    compiling: Option<Receiver<Result<(GuestModule, SystemTime)>>>,
    /// Its build's file changed since it last looked, as the watcher says.
    changed: Arc<AtomicBool>,
    _watch: Option<watch::Subscription>,
    /// Its own renderer, for its sprites over the game's frame: its assets'
    /// events stay out of the game's input.
    renderer: pen_render::Renderer,
    /// Its last frame: the sprites drawn and the interface routing input.
    frame: Option<RenderFrame>,
    /// What it is laid out on: the window, or the frame headless.
    viewport: Viewport,
    /// Window events since its last frame, the answers to its requests for
    /// its next, and frames of events a script typed.
    events: Vec<GameEvent>,
    answers: Vec<GameEvent>,
    typed: VecDeque<Vec<GameEvent>>,
    /// What input it takes from the game.
    routing: Routing,
    started: Instant,
    previous: Option<Instant>,
    /// Answers its requests of the session.
    service: tools::Devtools,
    adapter: String,
    baking: Option<Receiver<Result<String, String>>>,
    /// The last trouble it had, said once.
    trouble: Option<String>,
    /// Headless: a shot waits for its sprites' assets, since nothing else
    /// draws them before.
    headless: bool,
    /// Pictures of the game's frames and previews of its assets.
    atlas: pictures::Atlas,
    /// The catalog entries a swap draws from another asset: their own keys.
    swapped: HashMap<(u32, u32), String>,
    /// Remote dev tools it answers too (`crate::remote`).
    remote: Option<crate::remote::Remote>,
}

/// The date and time `since` the Unix epoch, in UTC, as a file name sorts
/// it: `20261015-142301-123` (to the millisecond).
fn stamp(since: Duration) -> String {
    let seconds = since.as_secs();
    let (days, time) = (seconds / 86_400, seconds % 86_400);
    // Days to a civil date (Howard Hinnant's `civil_from_days`).
    let days = days + 719_468;
    let era = days / 146_097;
    let day_of_era = days % 146_097;
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted + 2) / 5 + 1;
    let month = if shifted < 10 { shifted + 3 } else { shifted - 9 };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    format!(
        "{year:04}{month:02}{day:02}-{:02}{:02}{:02}-{:03}",
        time / 3600,
        time / 60 % 60,
        time % 60,
        since.subsec_millis()
    )
}

/// Whether `name` is one `stamp` made (perhaps with a count after it), or
/// an older one: Unix seconds.
fn stamped(name: &str) -> bool {
    !name.is_empty() && name.bytes().all(|byte| byte.is_ascii_digit() || byte == b'-')
}

/// A new file in `directory` with `extension`, named for now; of the files
/// named so there, the oldest beyond `KEEP_FILES` go first.
fn new_file(directory: &Path, extension: &str, now: Duration) -> PathBuf {
    let mut kept: Vec<PathBuf> = std::fs::read_dir(directory)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().is_some_and(|found| found == extension)
                && path.file_stem().and_then(|stem| stem.to_str()).is_some_and(stamped)
        })
        .collect();
    kept.sort();
    let surplus = (kept.len() + 1).saturating_sub(KEEP_FILES);
    for old in &kept[..surplus] {
        if let Err(error) = std::fs::remove_file(old) {
            eprintln!("devtools: removing {}: {error}", old.display());
        }
    }
    let name = stamp(now);
    let mut path = directory.join(format!("{name}.{extension}"));
    let mut count = 1;
    while path.exists() {
        count += 1;
        path = directory.join(format!("{name}-{count}.{extension}"));
    }
    path
}

/// A key pressed and let go, as the program reads it.
fn tap(key: &str) -> [GameEvent; 2] {
    [GameEvent::KeyDown(key.to_owned()), GameEvent::KeyUp(key.to_owned())]
}

/// A line typed into the console and run, a frame of keys after the frame
/// that opens it (unless `open`).
fn typed_line(line: &str, open: bool) -> Vec<Vec<GameEvent>> {
    let mut frames = Vec::new();
    if !open {
        frames.push(tap("`").to_vec());
    }
    let mut keys: Vec<GameEvent> = line.chars().flat_map(|key| tap(&key.to_string())).collect();
    keys.extend(tap("Enter"));
    frames.push(keys);
    frames
}

impl Devtools {
    pub fn new(
        window: Option<&Window>,
        gpu: &Gpu,
        catalog: Catalog,
        assets: &Path,
        settings: Settings,
    ) -> Result<Devtools> {
        // On the game renderer's device, loader threads and pipeline cache.
        let renderer = gpu.renderer.share(assets.to_owned())?;
        let info = gpu.device.adapter_info();
        let viewport = match window.map(Window::inner_size) {
            Some(size) => Viewport { width: size.width.max(1), height: size.height.max(1) },
            None => Viewport { width: gpu.size.0.max(1), height: gpu.size.1.max(1) },
        };
        // It looks once at the start, then whenever its build is written.
        let changed = Arc::new(AtomicBool::new(true));
        let watched = changed.clone();
        let build = settings.build.join(PROGRAM);
        let watch = watch::subscribe(&[(&settings.build, false)], move |event| {
            if let Ok(event) = event
                && !matches!(event.kind, notify::EventKind::Access(_))
                && event.paths.contains(&build)
            {
                watched.store(true, Ordering::Relaxed);
            }
        });
        let watch = watch.map_err(|error| {
            devlog::error(Source::Reload, format!("devtools: new builds are not picked up: {error:#}"))
        });
        let mut devtools = Devtools {
            engine: blot_host::dev_engine()?,
            catalog,
            build: settings.build.join(PROGRAM),
            project: settings.project,
            keep: settings.keep,
            runtime: None,
            built: None,
            compiling: None,
            changed,
            _watch: watch.ok(),
            renderer,
            frame: None,
            viewport,
            events: Vec::new(),
            answers: Vec::new(),
            typed: VecDeque::new(),
            routing: Routing::default(),
            started: Instant::now(),
            previous: None,
            service: tools::Devtools::default(),
            adapter: format!("{} ({:?})", info.name, info.backend),
            baking: None,
            trouble: None,
            headless: window.is_none(),
            atlas: pictures::Atlas::new(gpu.format),
            swapped: HashMap::new(),
            remote: settings.listen.as_deref().map(crate::remote::Remote::listen).transpose()?,
        };
        if settings.open {
            devtools.typed.push_back(tap("`").to_vec());
        }
        Ok(devtools)
    }

    /// Starts the program's build there is now, without waiting a frame
    /// for it to compile (a headless run's first frames show it).
    pub fn start_now(&mut self) -> Result<()> {
        let time = std::fs::metadata(&self.build)
            .and_then(|metadata| metadata.modified())
            .with_context(|| format!("the dev tools' program is not built ({})", self.build.display()))?;
        let module = GuestModule::new(&self.engine, &std::fs::read(&self.build)?, None)?;
        self.built = Some(time);
        self.start(module);
        Ok(())
    }

    /// Says a trouble once, until another.
    fn trouble(&mut self, said: String) {
        if self.trouble.as_ref() != Some(&said) {
            devlog::error(Source::Reload, &said);
            self.trouble = Some(said);
        }
    }

    /// A new build of the program, once the watcher says its file changed:
    /// compiled on a thread of its own, then started, or put in place of the
    /// running one with its state.
    fn look(&mut self) {
        if let Some(compiling) = &self.compiling {
            let Ok(compiled) = compiling.try_recv() else { return };
            self.compiling = None;
            match compiled {
                Ok((module, time)) => {
                    self.built = Some(time);
                    self.start(module);
                }
                Err(error) => self.trouble(format!("devtools: its build does not load: {error:#}")),
            }
            return;
        }
        if !self.changed.swap(false, Ordering::Relaxed) {
            return;
        }
        let Ok(time) = std::fs::metadata(&self.build).and_then(|metadata| metadata.modified()) else { return };
        if self.built == Some(time) {
            return;
        }
        let (sender, compiled) = channel();
        let (engine, path) = (self.engine.clone(), self.build.clone());
        let spawned = std::thread::Builder::new().name("pen-devtools-compile".into()).spawn(move || {
            let module = std::fs::read(&path)
                .map_err(anyhow::Error::from)
                .and_then(|bytes| GuestModule::new(&engine, &bytes, None));
            let _ = sender.send(module.map(|module| (module, time)));
        });
        match spawned {
            Ok(_) => self.compiling = Some(compiled),
            Err(error) => self.trouble(format!("devtools: cannot compile its build: {error}")),
        }
    }

    fn start(&mut self, module: GuestModule) {
        let mut shell = Shell { viewport: self.viewport };
        if let Some(runtime) = &mut self.runtime {
            runtime.reload(&mut shell, module, self.catalog.clone());
            return;
        }
        let state = self.project.join(STATE);
        let saved = state.is_file().then(|| snapshot::read(&state)).transpose();
        let started = match saved {
            Ok(saved) => GameRuntime::new(&self.engine, module.clone(), self.catalog.clone(), &mut shell, saved, None)
                // A state it cannot take starts it afresh.
                .or_else(|_| GameRuntime::new(&self.engine, module, self.catalog.clone(), &mut shell, None, None)),
            Err(error) => Err(error),
        };
        match started {
            Ok(runtime) => {
                self.runtime = Some(runtime);
                self.trouble = None;
            }
            Err(error) => self.trouble(format!("devtools: the program does not start: {error:#}")),
        }
    }

    /// Runs its frame on the events since its last, and does what it asks.
    fn run(&mut self, session: &mut Session, interval: f64) {
        let Some(runtime) = &mut self.runtime else { return };
        let mut events = std::mem::take(&mut self.answers);
        events.append(&mut self.events);
        if let Some(typed) = self.typed.pop_front() {
            events.extend(typed);
        }
        let frame = Frame { timestamp: self.started.elapsed().as_secs_f64(), viewport: self.viewport, events };
        let shell = Shell { viewport: self.viewport };
        let ran = runtime.input(&frame).and_then(|input| runtime.send(&input)).and_then(|()| runtime.receive(&shell));
        let rendered = match ran {
            Ok(rendered) => rendered,
            Err(error) => {
                // Stopped on an error: it waits for a new build.
                self.runtime = None;
                self.frame = None;
                self.routing.stopped();
                self.trouble(format!("devtools: stopped: {error:#}"));
                return;
            }
        };
        let mut reloads = Vec::new();
        let mut host = Answering {
            session: &mut *session,
            adapter: &self.adapter,
            project: &self.project,
            baking: &mut self.baking,
            atlas: &mut self.atlas,
            swapped: &mut self.swapped,
            reloads: &mut reloads,
        };
        for request in &rendered.services {
            let answer = if request.service == service::DEVTOOLS {
                pollster::block_on(self.service.answer(request, &mut host, interval))
            } else {
                Answer::unknown()
            };
            self.answers.extend(answer.events(request.request));
        }
        pollster::block_on(self.service.follow(&mut host));
        self.reloaded(&reloads);
        // Keys held when it took the keyboard would never come up for the
        // game: it lets go of them, as when the window loses focus.
        if self.routing.laid_out(&rendered.ui) {
            session.input.enqueue(GameEvent::FocusLost);
        }
        // What it draws with starts loading now, not when it is first drawn.
        if let Err(error) = self.renderer.prepare_sprites(&rendered.sprites, &self.catalog) {
            self.trouble(format!("devtools: its sprites do not draw: {error:#}"));
        }
        self.frame = Some(rendered);
    }

    /// Asset files edited for the dev tools: their renderer loads them again,
    /// and a material's previews are drawn again from its new file.
    fn reloaded(&mut self, reloads: &[String]) {
        if reloads.is_empty() {
            return;
        }
        self.renderer.reload(reloads);
        for key in reloads {
            if let Some(index) = self.catalog.materials.iter().position(|material| material == key) {
                self.atlas.changed(u32::try_from(index).unwrap_or(u32::MAX));
            }
        }
    }

    /// Answers the remote dev tools' requests since the last frame, as its
    /// own program's are, and looks at each one's watch.
    fn serve_remote(&mut self, session: &mut Session) {
        let Some(remote) = &mut self.remote else { return };
        let mut reloads = Vec::new();
        let mut host = Answering {
            session,
            adapter: &self.adapter,
            project: &self.project,
            baking: &mut self.baking,
            atlas: &mut self.atlas,
            swapped: &mut self.swapped,
            reloads: &mut reloads,
        };
        remote.serve(|service, request, interval| pollster::block_on(service.answer(request, &mut host, interval)));
        remote.follow(|service| pollster::block_on(service.follow(&mut host)));
        self.reloaded(&reloads);
    }

    /// A window event as the program reads it: every key (repeats too), the
    /// pointer, the wheel and losing the focus.
    fn take(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event: key, .. } => {
                // The console's key is the one left of 1, whatever the
                // layout makes of it (§ on a Swedish one, ~ with Shift).
                let name = if key.physical_key == PhysicalKey::Code(KeyCode::Backquote) {
                    "`".to_owned()
                } else {
                    let Some(name) = crate::input::key_name(&key.logical_key) else { return };
                    name
                };
                self.events.push(match key.state {
                    ElementState::Pressed => GameEvent::KeyDown(name),
                    ElementState::Released => GameEvent::KeyUp(name),
                });
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.routing.pointer = (crate::input::narrow(position.x), crate::input::narrow(position.y));
                let (x, y) = self.routing.pointer;
                self.events.push(GameEvent::PointerMove { x, y });
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let (x, y) = self.routing.pointer;
                let button = crate::input::button_number(*button);
                self.events.push(match state {
                    ElementState::Pressed => GameEvent::PointerDown { button, x, y },
                    ElementState::Released => GameEvent::PointerUp { button, x, y },
                });
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    MouseScrollDelta::LineDelta(_, lines) => -lines * crate::input::PIXELS_PER_LINE,
                    MouseScrollDelta::PixelDelta(pixels) => -crate::input::narrow(pixels.y),
                };
                let (x, y) = self.routing.pointer;
                self.events.push(GameEvent::Wheel { x, y, delta });
            }
            WindowEvent::Focused(false) => self.events.push(GameEvent::FocusLost),
            _ => {}
        }
    }
}

/// The session as the dev tools service's host: what it reports, and what
/// it does for them, with the dev tools' own part (the adapter's name, the
/// checkout files are named in, a bake under way, their atlas, the entries
/// swapped and the asset files edited, for their renderer to reload).
struct Answering<'a> {
    session: &'a mut Session,
    adapter: &'a str,
    project: &'a Path,
    baking: &'a mut Option<Receiver<Result<String, String>>>,
    atlas: &'a mut pictures::Atlas,
    swapped: &'a mut HashMap<(u32, u32), String>,
    reloads: &'a mut Vec<String>,
}

impl Answering<'_> {
    /// A new file in the checkout's `.pen/DIRECTORY`, named for now.
    fn new_file(&self, directory: &str, extension: &str) -> PathBuf {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        new_file(&self.project.join(".pen").join(directory), extension, now)
    }

    /// Writes the game's tuned values (or only tunable `only`'s) into the
    /// checkout as the defaults, on a thread of its own (Rust number-token
    /// edits); the log says how it went.
    fn start_bake(&mut self, only: Option<u32>) -> Answer {
        if self.baking.is_some() {
            return Answer::unavailable();
        }
        let declared = match tunables::load(self.project) {
            Ok(declared) => declared,
            Err(problem) => {
                devlog::error(Source::Session, format!("bake: {problem}"));
                return Answer::failed();
            }
        };
        let Some(saved) = self.session.snapshot() else { return Answer::unavailable() };
        let tuned = tunables::tuned_values(&saved.state);
        let values: Vec<(String, f32)> = declared
            .iter()
            .filter(|tunable| tuned.contains_key(&tunable.id) && only.is_none_or(|id| id == tunable.id))
            .map(|tunable| (tunable.name.clone(), tunable.live(&tuned)))
            .collect();
        if values.is_empty() {
            devlog::info(Source::Session, "bake: nothing to bake; each value is its default");
            return Answer::ok(0.0);
        }
        let (sender, receiver) = channel();
        let root = self.project.to_owned();
        let spawned = std::thread::Builder::new().name("pen-bake".into()).spawn(move || {
            let _ = sender.send(tunables::bake(&root, &values).map_err(|error| format!("{error:#}")));
        });
        match spawned {
            Ok(_) => {
                *self.baking = Some(receiver);
                Answer::ok(0.0)
            }
            Err(error) => {
                devlog::error(Source::Session, format!("bake: {error}"));
                Answer::failed()
            }
        }
    }

    /// The game's catalog entries of DevtoolsAsset `kind`.
    fn entries(&mut self, kind: u32) -> Option<&mut Vec<String>> {
        let catalog = self.session.catalog_mut();
        match kind {
            devtools_asset::MESH => Some(&mut catalog.meshes),
            devtools_asset::MATERIAL => Some(&mut catalog.materials),
            devtools_asset::SOUND => Some(&mut catalog.sounds),
            _ => None,
        }
    }

    /// Material `index`'s key and file in the game's assets.
    fn material_file(&mut self, index: u32) -> Option<(String, PathBuf)> {
        let key = self.session.catalog_mut().materials.get(index as usize)?.clone();
        let path = self.session.assets().join(&key);
        Some((key, path))
    }
}

/// A failure the log tells of, answered as failed.
fn failed(what: &str, problem: impl std::fmt::Display) -> Answer {
    devlog::error(Source::Session, format!("{what}: {problem}"));
    Answer::failed()
}

impl tools::Host for Answering<'_> {
    fn timeline(&self) -> TimelineStatus {
        self.session.timeline()
    }

    fn frames_before(&self, frame: u64, seconds: f64) -> u64 {
        self.session.frames_before(frame, seconds)
    }

    fn log(&self, from: u64) -> Answer {
        let lines = devlog::since(from);
        let lines = lines.iter().map(|line| LogLine {
            serial: line.serial,
            source: line.source as u32,
            error: line.error,
            text: &line.text,
        });
        log_answer(lines, from)
    }

    fn report(&self) -> FrameReport {
        let session = &self.session;
        let stats = session.gpu.renderer.stats();
        FrameReport {
            timings: session.frame_timings(),
            draw_calls: stats.draw_calls,
            instanced_draw_calls: stats.instanced_draw_calls,
            instances: stats.instances,
            meshes: session.gpu.drawn.meshes,
            brushes: session.gpu.drawn.brushes,
            memory: session.guest_memory(),
            size: session.gpu.size,
            recording: session.input_recording().map(|(_, from)| from),
            culled: stats.culled,
            shadow_culled: stats.shadow_culled,
            triangles: stats.triangles,
            scale: stats.scale,
            gpu: stats.gpu.map(|gpu| [gpu.shadow, gpu.culling, gpu.scene, gpu.post, gpu.overlay, gpu.total]),
        }
    }

    fn measure_frames(&mut self, enabled: bool) {
        self.session.measure_frames(enabled);
        self.session.gpu.renderer.time_passes(enabled);
    }

    fn gpu(&self) -> String {
        self.adapter.to_owned()
    }

    fn enqueue(&mut self, event: GameEvent) {
        self.session.input.enqueue(event);
    }

    fn pause(&mut self) {
        self.session.pause();
    }

    fn resume(&mut self) {
        self.session.resume();
    }

    fn step_frames(&mut self, frames: u32) {
        self.session.step_frames(frames);
    }

    fn rewind_to(&mut self, frame: u64) {
        self.session.rewind_to(frame);
    }

    fn replay_recent(&mut self, seconds: f64) {
        self.session.replay_recent(seconds);
    }

    fn keep_history(&mut self) {
        self.session.keep_history();
    }

    fn record(&mut self, path: Option<String>) -> Answer {
        let path = match self.session.input_recording() {
            Some(_) => None,
            None => Some(path.map_or_else(|| self.new_file("recordings", "penrec"), PathBuf::from)),
        };
        self.session.record_input(path);
        Answer::ok(0.0)
    }

    fn audit(&mut self) -> Answer {
        pen_runtime::audit::answer(&self.session.audit())
    }

    fn shot(&mut self, path: Option<String>) -> Answer {
        let path = path.map_or_else(|| self.new_file("shots", "png"), PathBuf::from);
        devlog::info(Source::Session, format!("a shot with the dev tools to {}", path.display()));
        self.session.overlay_shot(path);
        Answer::ok(0.0)
    }

    fn audio(&mut self) -> Option<AudioReport> {
        self.session.audio().map(crate::audio::Audio::report)
    }

    fn monitor(&mut self, bus: usize, mute: bool, on: bool) -> bool {
        self.session.audio().map(|audio| audio.monitor(bus, mute, on)).is_some()
    }

    async fn snapshot(&mut self) -> Option<SavedState> {
        self.session.snapshot()
    }

    async fn bake(&mut self) -> Answer {
        self.start_bake(None)
    }

    async fn bake_one(&mut self, id: u32) -> Answer {
        self.start_bake(Some(id))
    }

    fn say(&mut self, line: &str) {
        devlog::info(Source::Session, line);
    }

    fn packets(&mut self) -> Option<&mut Packets> {
        Some(self.session.packets())
    }

    fn kept_state(&self, seconds: f64) -> Option<(u64, SavedState)> {
        self.session.kept_state(seconds)
    }

    fn pictures(&mut self, taking: bool, atlas: Option<u32>) -> Answer {
        self.atlas.pictures(atlas, taking)
    }

    /// Samples the game's guest at about a millisecond of its running time;
    /// only the development engine's guests can be (`pen dev`).
    fn profile(&mut self, operation: u32) -> Answer {
        let (profile, engine) = self.session.profile();
        if !blot_host::samples(engine) {
            return Answer::unavailable();
        }
        match operation {
            1 => {
                let started = profile.get_or_insert_with(blot_host::Profile::new).start(engine);
                if let Err(error) = started {
                    return failed("profile", error);
                }
            }
            0 => profile.iter().for_each(|profile| profile.stop()),
            2 => profile.iter().for_each(|profile| profile.clear()),
            _ => {}
        }
        let (samples, sampling) =
            profile.as_ref().map_or((0, false), |profile| (profile.samples(), profile.sampling()));
        Answer { values: vec![samples as f32, f32::from(u8::from(sampling))], ..Answer::ok(samples as f32) }
    }

    fn flame(&mut self) -> Answer {
        let (profile, _) = self.session.profile();
        let Some(profile) = profile.clone() else { return Answer::ok(0.0) };
        let stacks = profile.stacks();
        inspect::flame(stacks.iter().map(|(stack, samples)| (stack.as_slice(), *samples)), |index| profile.name(index))
    }

    fn memory(&mut self) -> Answer {
        let seen = self.session.guest_seen();
        let mut counters = seen.counters.clone();
        counters.extend([
            ("calls to the host".to_owned(), seen.callbacks),
            ("calls into the guest".to_owned(), seen.calls),
            ("calls that failed".to_owned(), seen.failed),
        ]);
        inspect::memory_answer(seen.memory, seen.peak, &counters)
    }

    fn presets(&mut self) -> Answer {
        match tunables::load_presets(self.project) {
            Ok(presets) => Answer::text(&presets.keys().cloned().collect::<Vec<_>>().join("\n")),
            Err(problem) => failed("presets", problem),
        }
    }

    /// Every tunable at the value the game reads.
    async fn save_preset(&mut self, name: &str) -> Answer {
        let declared = match tunables::load(self.project) {
            Ok(declared) => declared,
            Err(problem) => return failed("preset", problem),
        };
        let Some(saved) = self.session.snapshot() else { return Answer::unavailable() };
        let tuned = tunables::tuned_values(&saved.state);
        let values = declared.iter().map(|tunable| (tunable.name.clone(), tunable.live(&tuned))).collect();
        match tunables::save_preset(self.project, name, values) {
            Ok(()) => {
                let path = tunables::presets_path(self.project);
                devlog::info(Source::Session, format!("preset {}: saved in {}", name.trim(), path.display()));
                Answer::ok(declared.len() as f32)
            }
            Err(error) => failed("preset", format!("{error:#}")),
        }
    }

    fn preset(&mut self, name: &str) -> Result<Vec<(u32, f32)>, Answer> {
        let presets = tunables::load_presets(self.project).map_err(|problem| failed("preset", problem))?;
        let preset = presets.get(name.trim()).ok_or_else(|| failed("preset", format!("there is no preset {name}")))?;
        let declared = tunables::load(self.project).map_err(|problem| failed("preset", problem))?;
        let tunes = tunables::preset_tunes(&declared, preset).map_err(|problem| failed("preset", problem))?;
        devlog::info(Source::Session, format!("preset {}: tuned {} values", name.trim(), tunes.len()));
        Ok(tunes)
    }

    fn assets(&mut self, kind: u32) -> Answer {
        let Some(entries) = self.entries(kind).map(|entries| entries.clone()) else { return Answer::failed() };
        let mut values = Vec::new();
        for key in &entries {
            let (state, revision) = self.session.gpu.renderer.standing(kind, key);
            let name = pen_runtime::services::code_points(key);
            values.extend([state as f32, revision as f32, name.len() as f32]);
            values.extend(name);
        }
        Answer { values, ..Answer::ok(entries.len() as f32) }
    }

    fn preview(&mut self, kind: u32, index: u32, other: u32, slot: u32) -> Answer {
        self.atlas.preview(kind, index, other, slot)
    }

    /// Until the next build; an empty key swaps the entry's own back.
    fn swap(&mut self, kind: u32, index: u32, key: &str) -> Answer {
        let assets = self.session.assets().to_owned();
        if !key.is_empty() && (pen_render::validate_asset_key(key).is_err() || !assets.join(key).is_file()) {
            return failed("swap", format!("there is no asset {key} in {}", assets.display()));
        }
        let Some(entry) = self.entries(kind).and_then(|entries| entries.get(index as usize).cloned()) else {
            return Answer::failed();
        };
        let own = self.swapped.entry((kind, index)).or_insert(entry).clone();
        let drawn = if key.is_empty() { own.clone() } else { key.to_owned() };
        if let Some(entry) = self.entries(kind).and_then(|entries| entries.get_mut(index as usize)) {
            entry.clone_from(&drawn);
        }
        if drawn == own {
            self.swapped.remove(&(kind, index));
            devlog::info(Source::Asset, format!("swap: {own} is its own again"));
        } else {
            devlog::info(Source::Asset, format!("swap: {own} draws from {drawn} until the next build"));
        }
        Answer::ok(0.0)
    }

    fn material(&mut self, index: u32) -> Answer {
        let Some((key, path)) = self.material_file(index) else { return Answer::failed() };
        let numbers = match std::fs::read_to_string(&path)
            .map_err(anyhow::Error::from)
            .and_then(|text| materials::numbers(&text))
        {
            Ok(numbers) => numbers,
            Err(error) => return failed(&key, format!("{error:#}")),
        };
        let mut values = numbers.values.to_vec();
        values.extend([f32::from(u8::from(numbers.color)), f32::from(u8::from(numbers.parameters))]);
        for text in [&numbers.shader, &numbers.texture] {
            let codes = pen_runtime::services::code_points(text);
            values.push(codes.len() as f32);
            values.extend(codes);
        }
        Answer { values, ..Answer::ok(0.0) }
    }

    /// In the checkout's file, which both renderers then load again.
    fn set_material(&mut self, index: u32, field: u32, value: f32) -> Answer {
        let Some((key, path)) = self.material_file(index) else { return Answer::failed() };
        if let Err(error) = materials::write(&path, field as usize, value) {
            return failed(&key, format!("{error:#}"));
        }
        self.session.gpu.renderer.reload(std::slice::from_ref(&key));
        self.reloads.push(key);
        Answer::ok(0.0)
    }
}

impl Overlay for Devtools {
    fn script_event(&mut self, event: &GameEvent) -> bool {
        let ui = self.frame.as_ref().map_or(&[][..], |frame| &frame.ui);
        let Some(game) = self.routing.route(event, ui, self.runtime.is_some()) else { return true };
        self.events.push(event.clone());
        game
    }

    fn window_event(&mut self, _window: &Window, event: &WindowEvent) -> bool {
        self.take(event);
        // The backquote key is the dev tools' alone, as in the browser.
        if let WindowEvent::KeyboardInput { event: key, .. } = event
            && key.physical_key == PhysicalKey::Code(KeyCode::Backquote)
        {
            return false;
        }
        if self.runtime.is_none() {
            return true;
        }
        let routing = &mut self.routing;
        match event {
            WindowEvent::KeyboardInput { .. } | WindowEvent::Ime(_) => !routing.typing,
            WindowEvent::CursorMoved { .. } => {
                let ui = self.frame.as_ref().map_or(&[][..], |frame| &frame.ui);
                routing.over = inside_dialog(ui, routing.pointer);
                // A drag the game started stays the game's, over a dialog or not.
                routing.buttons != 0 || !routing.over
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button, .. } => {
                let game = routing.buttons != 0 || !routing.over;
                if game {
                    routing.buttons |= button_bit(crate::input::button_number(*button));
                }
                game
            }
            // Releases always reach the game, as the browser's do.
            WindowEvent::MouseInput { state: ElementState::Released, button, .. } => {
                routing.buttons &= !button_bit(crate::input::button_number(*button));
                true
            }
            WindowEvent::MouseWheel { .. } | WindowEvent::Touch(_) => !routing.over,
            _ => true,
        }
    }

    fn update(&mut self, session: &mut Session, window: Option<&Window>) {
        if let Some(size) = window.map(Window::inner_size).filter(|size| size.width > 0 && size.height > 0) {
            self.viewport = Viewport { width: size.width, height: size.height };
        } else if window.is_none() {
            self.viewport = Viewport { width: session.gpu.size.0.max(1), height: session.gpu.size.1.max(1) };
        }
        if let Some(baking) = &self.baking
            && let Ok(baked) = baking.try_recv()
        {
            self.baking = None;
            match baked {
                Ok(said) => devlog::info(Source::Session, format!("bake: {said}")),
                Err(error) => devlog::error(Source::Session, format!("bake: {error}")),
            }
        }
        self.look();
        let now = Instant::now();
        let interval = self.previous.replace(now).map_or(0.0, |previous| (now - previous).as_secs_f64() * 1000.0);
        self.run(session, interval);
        self.serve_remote(session);
        let gpu = (&session.gpu.device, &session.gpu.queue);
        if let Err(error) = self.atlas.draw_previews(&mut self.renderer, &self.catalog, session.assets(), gpu) {
            self.trouble(format!("devtools: a preview: {error:#}"));
        }
    }

    fn command(&mut self, line: String) {
        // Typed into its console, opened first unless its text box has the
        // keyboard or it is opening already.
        let opening = self.typed.iter().any(|typed| typed.first() == Some(&GameEvent::KeyDown("`".to_owned())));
        let console_focused = self.frame.as_ref().is_some_and(|frame| {
            frame.ui.iter().any(|node| node.role == pen_protocol::ui_role::TEXTBOX && node.focused)
        });
        self.typed.extend(typed_line(&line, console_focused || opening));
    }

    fn paint(&mut self, _device: &wgpu::Device, _queue: &wgpu::Queue, target: &wgpu::Texture) {
        let Some(frame) = &self.frame else { return };
        if frame.sprites.is_empty() {
            return;
        }
        if self.headless {
            let waited = Instant::now();
            while !self.renderer.prepare_sprites(&frame.sprites, &self.catalog).unwrap_or(true) {
                let Some(left) = Duration::from_secs(10).checked_sub(waited.elapsed()) else { break };
                self.renderer.wait(left);
            }
        }
        let view = target.create_view(&Default::default());
        let size = (self.viewport.width, self.viewport.height);
        let drawn = self.renderer.overlay(&frame.sprites, &self.catalog, &view, size);
        // Its assets' events are its own; the game never hears of them.
        let _ = self.renderer.take_events();
        if let Err(error) = drawn {
            self.trouble(format!("devtools: its sprites do not draw: {error:#}"));
        }
    }

    fn wants_pointer(&self) -> bool {
        self.runtime.is_some() && self.routing.over
    }

    fn wants_picture(&self, position: u64) -> bool {
        self.atlas.wants_picture(position)
    }

    fn picture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, position: u64) {
        let taken = self.atlas.picture(&mut self.renderer, &self.catalog, (device, queue), texture, position);
        if let Err(error) = taken {
            self.trouble(format!("devtools: a picture of frame {position}: {error:#}"));
        }
    }

    fn interface(&self) -> Option<&[UiNode]> {
        self.frame.as_ref().map(|frame| frame.ui.as_slice()).filter(|ui| !ui.is_empty())
    }

    fn accesskit_action(&mut self, request: accesskit::ActionRequest) {
        let Some(frame) = &self.frame else { return };
        if let Some(event) = crate::accessibility::program_event(&frame.ui, &request) {
            self.events.push(event);
        }
    }
}

impl Drop for Devtools {
    fn drop(&mut self) {
        let Some(runtime) = &mut self.runtime else { return };
        if self.keep
            && let Err(error) = runtime.save(&self.project.join(STATE))
        {
            eprintln!("devtools: its state was not kept: {error:#}");
        }
        runtime.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_a_line_after_opening_the_console() {
        let frames = typed_line("/a b", false);
        assert_eq!(frames[0], tap("`"));
        let keys: Vec<&GameEvent> = frames[1].iter().step_by(2).collect();
        let down = |key: &str| GameEvent::KeyDown(key.to_owned());
        assert_eq!(keys, [&down("/"), &down("a"), &down(" "), &down("b"), &down("Enter")]);
        assert_eq!(typed_line("/a", true).len(), 1);
    }

    #[test]
    fn names_files_for_when_they_are_made() {
        let at = |seconds: u64, millis: u64| stamp(Duration::from_millis(seconds * 1000 + millis));
        assert_eq!(at(0, 0), "19700101-000000-000");
        assert_eq!(at(1_792_074_181, 123), "20261015-142301-123");
        assert_eq!(at(1_709_251_199, 999), "20240229-235959-999");
        assert_eq!(at(951_868_800, 7), "20000301-000000-007");
        assert!(stamped("20261015-142301-123-2") && stamped("1760538181"));
        assert!(!stamped("boss fight") && !stamped(""));
    }

    #[test]
    fn keeps_the_newest_files_and_never_overwrites() {
        let directory = std::env::temp_dir().join(format!("pen-devtools-files-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let now = Duration::from_secs(1_792_074_181);
        let first = new_file(&directory, "png", now);
        std::fs::write(&first, "").unwrap();
        // The same millisecond: another name.
        let second = new_file(&directory, "png", now);
        assert_eq!(second.file_name().unwrap(), "20261015-142301-000-2.png");
        // Full: the oldest go; files it did not name stay.
        std::fs::write(directory.join("mine.png"), "").unwrap();
        std::fs::write(directory.join("1760538181.penrec"), "").unwrap();
        for second in 0..KEEP_FILES as u64 + 3 {
            std::fs::write(new_file(&directory, "png", now + Duration::from_secs(second + 1)), "").unwrap();
        }
        let mut names: Vec<String> = std::fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names.len(), KEEP_FILES + 2);
        assert!(names.contains(&"mine.png".to_owned()) && names.contains(&"1760538181.penrec".to_owned()));
        assert!(!names.contains(&"20261015-142301-000.png".to_owned()));
        let _ = std::fs::remove_dir_all(&directory);
    }
}
