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
//! carries out.
//!
//! Its build (`devtools.wasm`, beside the game's) is picked up when it
//! appears and whenever it changes, with its state carried over; its state
//! is kept in the checkout's `.pen/devtools.state` between windowed sessions.
//! A checkout without the program runs without dev tools.
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use blot_host::GuestModule;
use pen_protocol::{
    Catalog, Frame, GameEvent, RenderFrame, ServiceRequest, UiNode, Viewport, devtools_frame, devtools_op, service,
};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

use crate::game::{GameRuntime, Host};
use crate::overlay::{self, Source, tunables};
use crate::services::{Answer, code_points, text};
use crate::session::{Gpu, Overlay, Session};
use crate::snapshot;

/// The program's build, beside the game's.
pub const PROGRAM: &str = "devtools.wasm";
/// Where its state is kept between windowed sessions, in the checkout.
const STATE: &str = ".pen/devtools.state";
/// How often to look for a new build of it.
const LOOK_EVERY: Duration = Duration::from_millis(250);
/// The most values a log answer carries; the program asks again for the rest.
const LOG_VALUES: usize = 12_000;

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
        overlay::note(Source::Reload, format!("devtools: {message}"));
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
    looked: Option<Instant>,
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
    /// One of its nodes has the focus: the game gets no keys.
    typing: bool,
    /// The pointer is over one of its dialogs, and where it is.
    over: bool,
    pointer: (f32, f32),
    /// Buttons the game got pressed, whose drags and releases stay its own.
    buttons: u32,
    started: Instant,
    previous: Option<Instant>,
    /// The game's position at the last frame answer: a frame that did not
    /// run has no timings.
    position: Option<u64>,
    measuring: bool,
    adapter: String,
    baking: Option<Receiver<Result<String, String>>>,
    /// The last trouble it had, said once.
    trouble: Option<String>,
    /// Headless: a shot waits for its sprites' assets, since nothing else
    /// draws them before.
    headless: bool,
}

fn button_bit(button: MouseButton) -> u32 {
    match button {
        MouseButton::Left => 1,
        MouseButton::Right => 2,
        MouseButton::Middle => 4,
        MouseButton::Back => 8,
        MouseButton::Forward => 16,
        MouseButton::Other(_) => 32,
    }
}

/// A name for a new recording or shot.
fn stamp() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |since| since.as_secs())
}

/// Whether `point` is inside one of the interface's dialogs.
fn inside_dialog(ui: &[UiNode], (x, y): (f32, f32)) -> bool {
    ui.iter().any(|node| {
        node.role == pen_protocol::ui_role::DIALOG
            && !node.hidden
            && x >= node.x
            && y >= node.y
            && x < node.x + node.width
            && y < node.y + node.height
    })
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

/// The log's `lines` from serial `from` on, as many as one answer carries:
/// each its source, whether an error, its length and its code points (a
/// line longer than an answer cut short); the value is the serial to ask
/// from next.
fn log_answer(lines: &[overlay::Line], from: u64) -> Answer {
    let mut values = Vec::new();
    let mut next = from;
    for line in lines.iter().filter(|line| line.serial >= from) {
        let text: Vec<f32> = code_points(&line.text).into_iter().take(LOG_VALUES - 3).collect();
        if !values.is_empty() && values.len() + 3 + text.len() > LOG_VALUES {
            break;
        }
        values.extend([line.source as u32 as f32, f32::from(u8::from(line.error)), text.len() as f32]);
        values.extend(text);
        next = line.serial + 1;
    }
    Answer { values, ..Answer::ok(next as f32) }
}

impl Devtools {
    pub fn new(window: Option<&Window>, gpu: &Gpu, catalog: Catalog, assets: &Path, settings: Settings) -> Result<Devtools> {
        let options = pen_render::RendererOptions { format: gpu.format, assets: assets.to_owned(), veil: None };
        let renderer = pen_render::Renderer::new(gpu.device.clone(), gpu.queue.clone(), options)?;
        let info = gpu.device.adapter_info();
        let viewport = match window.map(Window::inner_size) {
            Some(size) => Viewport { width: size.width.max(1), height: size.height.max(1) },
            None => Viewport { width: gpu.size.0.max(1), height: gpu.size.1.max(1) },
        };
        let mut devtools = Devtools {
            engine: blot_host::dev_engine()?,
            catalog,
            build: settings.build.join(PROGRAM),
            project: settings.project,
            keep: settings.keep,
            runtime: None,
            built: None,
            compiling: None,
            looked: None,
            renderer,
            frame: None,
            viewport,
            events: Vec::new(),
            answers: Vec::new(),
            typed: VecDeque::new(),
            typing: false,
            over: false,
            pointer: (0.0, 0.0),
            buttons: 0,
            started: Instant::now(),
            previous: None,
            position: None,
            measuring: false,
            adapter: format!("{} ({:?})", info.name, info.backend),
            baking: None,
            trouble: None,
            headless: window.is_none(),
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
            overlay::error(Source::Reload, &said);
            self.trouble = Some(said);
        }
    }

    /// A new build of the program: compiled on a thread of its own, then
    /// started, or put in place of the running one with its state.
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
        if self.looked.is_some_and(|at| at.elapsed() < LOOK_EVERY) {
            return;
        }
        self.looked = Some(Instant::now());
        let Ok(time) = std::fs::metadata(&self.build).and_then(|metadata| metadata.modified()) else { return };
        if self.built == Some(time) {
            return;
        }
        let (sender, compiled) = channel();
        let (engine, path) = (self.engine.clone(), self.build.clone());
        let spawned = std::thread::Builder::new().name("pen-devtools-compile".into()).spawn(move || {
            let module = std::fs::read(&path).map_err(anyhow::Error::from).and_then(|bytes| GuestModule::new(&engine, &bytes, None));
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
                self.typing = false;
                self.over = false;
                self.trouble(format!("devtools: stopped: {error:#}"));
                return;
            }
        };
        for request in &rendered.services {
            let answer = if request.service == service::DEVTOOLS {
                self.devtools(request, session, interval)
            } else {
                Answer::unknown()
            };
            self.answers.extend(answer.events(request.request));
        }
        // Keys held when it took the keyboard would never come up for the
        // game: it lets go of them, as when the window loses focus.
        let typing = rendered.ui.iter().any(|node| node.focused);
        if typing && !self.typing {
            session.input.enqueue(GameEvent::FocusLost);
        }
        self.typing = typing;
        self.over = inside_dialog(&rendered.ui, self.pointer);
        // What it draws with starts loading now, not when it is first drawn.
        if let Err(error) = self.renderer.prepare_sprites(&rendered.sprites, &self.catalog) {
            self.trouble(format!("devtools: its sprites do not draw: {error:#}"));
        }
        self.frame = Some(rendered);
    }

    /// Does what a devtools request asks, or answers what it asks about.
    fn devtools(&mut self, request: &ServiceRequest, session: &mut Session, interval: f64) -> Answer {
        let (a, b) = (request.a, request.b);
        let status = session.timeline();
        let path = || Some(PathBuf::from(text(&request.values))).filter(|path| !path.as_os_str().is_empty());
        match request.operation {
            devtools_op::LOG => log_answer(&overlay::since(a.max(0.0) as u64), a.max(0.0) as u64),
            devtools_op::FRAME => self.numbers(session, interval, a == 1.0),
            devtools_op::GPU => Answer::text(&self.adapter),
            devtools_op::COMMAND => {
                session.input.enqueue(GameEvent::Console { command: a, argument: b });
                Answer::ok(0.0)
            }
            devtools_op::PAUSE => {
                session.pause();
                Answer::ok(0.0)
            }
            devtools_op::PLAY => {
                session.resume();
                Answer::ok(0.0)
            }
            devtools_op::STEP => {
                for _ in 0..(a.max(1.0) as u32).min(600) {
                    session.step_frame();
                }
                Answer::ok(0.0)
            }
            devtools_op::REWIND | devtools_op::REPLAY if !status.history => Answer::unavailable(),
            devtools_op::REWIND => {
                session.rewind_to(session.frames_before(status.frame, f64::from(a.max(0.0))));
                Answer::ok(0.0)
            }
            devtools_op::REPLAY => {
                session.replay_recent(f64::from(a.max(0.0)));
                Answer::ok(0.0)
            }
            devtools_op::HISTORY => {
                session.keep_history();
                Answer::ok(0.0)
            }
            devtools_op::RECORD => {
                let path = match session.input_recording() {
                    Some(_) => None,
                    None => Some(path().unwrap_or_else(|| self.new_file("recordings", "penrec"))),
                };
                session.record_input(path);
                Answer::ok(0.0)
            }
            devtools_op::SHOT => {
                let path = path().unwrap_or_else(|| self.new_file("shots", "png"));
                overlay::info(Source::Session, format!("a shot with the dev tools to {}", path.display()));
                session.overlay_shot(path);
                Answer::ok(0.0)
            }
            devtools_op::TUNE => {
                session.input.enqueue(GameEvent::Tune { index: a as u32, value: b });
                Answer::ok(0.0)
            }
            devtools_op::TUNED => match session.snapshot() {
                Some(saved) => {
                    let mut tuned: Vec<(u32, f32)> = tunables::tuned_values(&saved.state).into_iter().collect();
                    tuned.sort_by_key(|(id, _)| *id);
                    let values = tuned.into_iter().flat_map(|(id, value)| [id as f32, value]).collect();
                    Answer { values, ..Answer::ok(0.0) }
                }
                None => Answer::unavailable(),
            },
            devtools_op::BAKE => self.bake(session),
            _ => Answer::unknown(),
        }
    }

    /// The last frame's numbers, in DevtoolsFrame's order; frames are timed
    /// while `measure` says so.
    fn numbers(&mut self, session: &mut Session, interval: f64, measure: bool) -> Answer {
        if measure != self.measuring {
            session.measure_frames(measure);
            self.measuring = measure;
        }
        let status = session.timeline();
        let ran = self.position.replace(status.frame) != Some(status.frame);
        let timings = session.frame_timings().filter(|_| ran && measure);
        let stats = session.gpu.renderer.stats();
        let drawn = session.gpu.drawn;
        let mut values = vec![0.0; devtools_frame::NAMES.len()];
        let mut set = |at: u32, value: f64| values[at as usize] = value as f32;
        set(devtools_frame::INTERVAL, interval);
        let phase = |pick: fn(&crate::game::FrameTimings) -> f64| timings.as_ref().map_or(-1.0, pick);
        set(devtools_frame::GUEST, phase(|t| t.guest_ms));
        set(devtools_frame::WAIT, phase(|t| t.wait_ms));
        set(devtools_frame::DECODE, phase(|t| t.decode_ms));
        set(devtools_frame::VALIDATE, phase(|t| t.validate_ms));
        set(devtools_frame::SUBMIT, phase(|t| t.submit_ms));
        set(devtools_frame::HOUSEKEEPING, phase(|t| t.housekeeping_ms));
        set(devtools_frame::TOTAL, phase(|t| t.total_ms));
        set(devtools_frame::PACKET, timings.map_or(-1.0, |t| t.packet_bytes as f64));
        set(devtools_frame::DRAW_CALLS, f64::from(stats.draw_calls));
        set(devtools_frame::INSTANCED_DRAW_CALLS, f64::from(stats.instanced_draw_calls));
        set(devtools_frame::INSTANCES, f64::from(stats.instances));
        set(devtools_frame::MESHES, drawn.meshes as f64);
        set(devtools_frame::BRUSHES, drawn.brushes as f64);
        set(devtools_frame::MEMORY, session.guest_memory() as f64);
        set(devtools_frame::WIDTH, f64::from(session.gpu.size.0));
        set(devtools_frame::HEIGHT, f64::from(session.gpu.size.1));
        set(devtools_frame::POSITION, status.frame as f64);
        set(devtools_frame::PAUSED, f64::from(u8::from(status.paused)));
        set(devtools_frame::REPLAYING, f64::from(u8::from(status.replaying)));
        set(devtools_frame::HISTORY, f64::from(u8::from(status.history)));
        set(devtools_frame::PIPELINED, f64::from(u8::from(status.pipelined)));
        set(devtools_frame::RECORDING, session.input_recording().map_or(-1.0, |(_, from)| from as f64));
        Answer { values, ..Answer::ok(0.0) }
    }

    /// Writes the game's tuned values into the checkout as the defaults, on
    /// a thread of its own (Rust number-token edits); the log says how it went.
    fn bake(&mut self, session: &mut Session) -> Answer {
        if self.baking.is_some() {
            return Answer::unavailable();
        }
        let declared = match tunables::load(&self.project) {
            Ok(declared) => declared,
            Err(problem) => {
                overlay::error(Source::Session, format!("bake: {problem}"));
                return Answer::failed();
            }
        };
        let Some(saved) = session.snapshot() else { return Answer::unavailable() };
        let tuned = tunables::tuned_values(&saved.state);
        let values: Vec<(String, f32)> = declared
            .iter()
            .filter(|tunable| tuned.contains_key(&tunable.id))
            .map(|tunable| (tunable.name.clone(), tunable.live(&tuned)))
            .collect();
        if values.is_empty() {
            overlay::info(Source::Session, "bake: nothing to bake; each value is its default");
            return Answer::ok(0.0);
        }
        let (sender, receiver) = channel();
        let root = self.project.clone();
        let spawned = std::thread::Builder::new().name("pen-bake".into()).spawn(move || {
            let _ = sender.send(tunables::bake(&root, &values).map_err(|error| format!("{error:#}")));
        });
        match spawned {
            Ok(_) => {
                self.baking = Some(receiver);
                Answer::ok(0.0)
            }
            Err(error) => {
                overlay::error(Source::Session, format!("bake: {error}"));
                Answer::failed()
            }
        }
    }

    /// A new file in the checkout's `.pen/DIRECTORY`.
    fn new_file(&self, directory: &str, extension: &str) -> PathBuf {
        self.project.join(".pen").join(directory).join(format!("{}.{extension}", stamp()))
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
                self.pointer = (position.x as f32, position.y as f32);
                let (x, y) = self.pointer;
                self.events.push(GameEvent::PointerMove { x, y });
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let (x, y) = self.pointer;
                let button = crate::input::button_number(*button);
                self.events.push(match state {
                    ElementState::Pressed => GameEvent::PointerDown { button, x, y },
                    ElementState::Released => GameEvent::PointerUp { button, x, y },
                });
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    MouseScrollDelta::LineDelta(_, lines) => -lines * crate::input::PIXELS_PER_LINE,
                    MouseScrollDelta::PixelDelta(pixels) => -pixels.y as f32,
                };
                let (x, y) = self.pointer;
                self.events.push(GameEvent::Wheel { x, y, delta });
            }
            WindowEvent::Focused(false) => self.events.push(GameEvent::FocusLost),
            _ => {}
        }
    }
}

impl Overlay for Devtools {
    fn script_event(&mut self, event: &GameEvent) -> bool {
        match event {
            GameEvent::KeyDown(_) | GameEvent::KeyUp(_) | GameEvent::FocusLost => {},
            GameEvent::PointerMove { x, y } | GameEvent::PointerDown { x, y, .. }
            | GameEvent::PointerUp { x, y, .. } | GameEvent::Wheel { x, y, .. } => {
                self.pointer = (*x, *y);
                let ui = self.frame.as_ref().map_or(&[][..], |frame| &frame.ui);
                self.over = inside_dialog(ui, self.pointer);
            }
            _ => return true,
        }
        self.events.push(event.clone());
        if matches!(event, GameEvent::KeyDown(key) | GameEvent::KeyUp(key) if key == "`") {
            return false;
        }
        if self.runtime.is_none() {
            return true;
        }
        let button_bit = |button: u32| match button { 0 => 1, 1 => 4, 2 => 2, 3 => 8, 4 => 16, _ => 32 };
        match event {
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
        }
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
        match event {
            WindowEvent::KeyboardInput { .. } | WindowEvent::Ime(_) => !self.typing,
            WindowEvent::CursorMoved { .. } => {
                let ui = self.frame.as_ref().map_or(&[][..], |frame| &frame.ui);
                self.over = inside_dialog(ui, self.pointer);
                // A drag the game started stays the game's, over a dialog or not.
                self.buttons != 0 || !self.over
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button, .. } => {
                let game = self.buttons != 0 || !self.over;
                if game {
                    self.buttons |= button_bit(*button);
                }
                game
            }
            // Releases always reach the game, as the browser's do.
            WindowEvent::MouseInput { state: ElementState::Released, button, .. } => {
                self.buttons &= !button_bit(*button);
                true
            }
            WindowEvent::MouseWheel { .. } | WindowEvent::Touch(_) => !self.over,
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
                Ok(said) => overlay::info(Source::Session, format!("bake: {said}")),
                Err(error) => overlay::error(Source::Session, format!("bake: {error}")),
            }
        }
        self.look();
        let now = Instant::now();
        let interval = self.previous.replace(now).map_or(0.0, |previous| (now - previous).as_secs_f64() * 1000.0);
        self.run(session, interval);
    }

    fn command(&mut self, line: String) {
        // Typed into its console, opened first unless it has the keyboard or
        // is opening already.
        let opening = self.typed.iter().any(|typed| typed.first() == Some(&GameEvent::KeyDown("`".to_owned())));
        let console_focused = self.frame.as_ref().is_some_and(|frame| {
            frame.ui.iter().any(|node| node.id == 0x1_0002 && node.focused)
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
            while !self.renderer.prepare_sprites(&frame.sprites, &self.catalog).unwrap_or(true)
                && waited.elapsed() < Duration::from_secs(10)
            {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        let view = target.create_view(&Default::default());
        let size = (self.viewport.width, self.viewport.height);
        if let Err(error) = self.renderer.overlay(&frame.sprites, &self.catalog, &view, size) {
            let said = format!("devtools: its sprites do not draw: {error:#}");
            if self.trouble.as_ref() != Some(&said) {
                overlay::error(Source::Reload, &said);
                self.trouble = Some(said);
            }
        }
        // Its assets' events are its own; the game never hears of them.
        let _ = self.renderer.take_events();
    }

    fn wants_pointer(&self) -> bool {
        self.runtime.is_some() && self.over
    }

    fn accesskit_shown(&self) -> bool {
        self.frame.as_ref().is_some_and(|frame| !frame.ui.is_empty())
    }

    fn accesskit_update(&mut self) -> Option<accesskit::TreeUpdate> {
        let frame = self.frame.as_ref()?;
        (!frame.ui.is_empty()).then(|| crate::accessibility::program_tree(&frame.ui))
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
    use pen_protocol::ui_role;

    fn line(serial: u64, text: &str) -> overlay::Line {
        overlay::Line { serial, source: Source::Build, error: serial % 2 == 1, text: text.to_owned() }
    }

    #[test]
    fn answers_the_log_a_whole_line_at_a_time() {
        let lines = [line(4, "ab"), line(5, "c"), line(6, "")];
        let answer = log_answer(&lines, 5);
        assert_eq!(answer.value, 7.0);
        assert_eq!(answer.values, [1.0, 1.0, 1.0, 99.0, 1.0, 0.0, 0.0]);
        assert_eq!(log_answer(&lines, 7), Answer::ok(7.0));
        // As many lines as fit; the next answer goes on from the first left out.
        let long = "x".repeat(LOG_VALUES / 2);
        let lines = [line(0, &long), line(1, &long), line(2, "y")];
        let answer = log_answer(&lines, 0);
        assert_eq!((answer.value, answer.values.len()), (1.0, 3 + long.len()));
        // A line longer than an answer is cut short, and its length says so.
        let answer = log_answer(&[line(0, &"z".repeat(LOG_VALUES * 2))], 0);
        assert_eq!(answer.values.len(), LOG_VALUES);
        assert_eq!(answer.values[2] as usize, LOG_VALUES - 3);
    }

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
    fn takes_the_pointer_over_dialogs_only() {
        let node = |role, x, y, width, height| UiNode { role, x, y, width, height, ..UiNode::default() };
        let ui = [node(ui_role::DIALOG, 10.0, 10.0, 100.0, 50.0), node(ui_role::STATUS, 500.0, 0.0, 200.0, 200.0)];
        assert!(inside_dialog(&ui, (10.0, 10.0)));
        assert!(!inside_dialog(&ui, (110.0, 30.0)));
        assert!(!inside_dialog(&ui, (600.0, 50.0)));
        let hidden = UiNode { hidden: true, ..ui[0].clone() };
        assert!(!inside_dialog(&[hidden], (20.0, 20.0)));
    }
}
