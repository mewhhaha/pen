//! The page's frame loop. Each iteration runs the dev tools, carries
//! out what was asked of time (pauses, steps, rewinds and replays, through
//! pen-runtime's timeline and transport, as native pen does), runs the
//! game's frame, recorded or live, then draws in the next repaint callback.
//! Saves and development endpoints keep off the frame's path.
use std::cell::RefCell;
use std::rc::Rc;

use anyhow::Result;
use pen_protocol::{Catalog, Frame, GameEvent, RenderFrame, Viewport};
use pen_runtime::clock::Clock;
use pen_runtime::devtools::{self, FrameReport, Log, TimelineStatus, log_source};
use pen_runtime::game::{FrameTimings, GameRuntime, Host, InvalidFrame, Quiet, ReloadResult, hash_lanes};
use pen_runtime::input::Hotkey;
use pen_runtime::program::Unresponsive;
use pen_runtime::recording;
use pen_runtime::services::{Answer, Services};
use pen_runtime::snapshot::{self, SavedState};
use pen_runtime::state::StateAdapter;
use pen_runtime::text::text_box;
use pen_runtime::timeline::{self, KEEP_SECONDS, Load, REPLAY_SECONDS, Recorded, Recording, Timeline};
use pen_runtime::transport::{Replaying, Rewind, Transport};
use std::sync::Arc;
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::HtmlCanvasElement;
use web_time::Instant;

use crate::accessibility::Accessibility;
use crate::audio::Audio;
use crate::development::{Development, Update};
use crate::files::{self, Arrival, Files};
use crate::input::Input;
use crate::preferences::Preferences;
use crate::program::{BrowserBackend, error};
use crate::replay::{self, Recorder};
use crate::saves;
use crate::screen::{FRAMES_IN_FLIGHT, Screen};
use crate::tools::Tools;

pub type Runtime = GameRuntime<BrowserBackend>;

/// Seconds between autosaves.
const AUTOSAVE_SECONDS: f64 = 5.0;
/// Log lines the dev tools can still read.
pub const LOG_LINES: usize = 2000;

/// Says `text` on the console and keeps it for the dev tools' log.
pub fn say(log: &mut Log, source: u32, error: bool, text: impl Into<String>) {
    let text = text.into();
    if error {
        web_sys::console::error_1(&text.as_str().into());
    } else {
        web_sys::console::log_1(&text.as_str().into());
    }
    log.push(source, error, text);
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

/// What the game's runtime may touch while it runs a frame or restores.
pub struct Shell<'a> {
    viewport: Viewport,
    events: &'a mut Vec<GameEvent>,
    preferences: &'a mut Preferences,
    services: &'a mut Services,
    log: &'a mut Log,
    state_adapter: &'a Arc<dyn StateAdapter>,
}

impl Host for Shell<'_> {
    fn viewport(&self) -> Viewport {
        self.viewport
    }

    fn validate(&self, frame: &RenderFrame) -> Result<()> {
        pen_render::validate_frame(frame)
    }

    /// A frame the runtime shows itself (a load's, a new build's) is drawn
    /// with the page's next frame; its preferences are kept now.
    fn render(&mut self, frame: &RenderFrame, _catalog: &Catalog) -> Result<()> {
        if let Some(values) = &frame.preferences {
            self.preferences.store(values);
        }
        Ok(())
    }

    fn report(&mut self, message: &str) {
        // The runtime's word on a new build starts "Blot reload".
        match message.strip_prefix("Blot reload") {
            Some(rest) => say(self.log, log_source::RELOAD, !rest.starts_with("ed"), message),
            None => say(self.log, log_source::GAME, false, message),
        }
    }

    fn restored(&mut self) {
        self.events.extend(self.preferences.events());
        // The answers waiting were for the world that asked.
        self.services.forget();
    }

    fn state_adapter(&self) -> Arc<dyn StateAdapter> {
        self.state_adapter.clone()
    }
}

/// Everything the game's runtime may touch, borrowed from the page's fields
/// so that the runtime and timeline stay free.
macro_rules! shell {
    ($browser:expr) => {
        Shell {
            viewport: $browser.screen.viewport(),
            events: &mut $browser.events,
            preferences: &mut $browser.preferences,
            services: &mut $browser.services,
            log: &mut $browser.log,
            state_adapter: &$browser.state_adapter,
        }
    };
}

/// What the browser harnesses wait on (`?qualify`), of the last frame the
/// GPU finished: its position, its quality, the assets still loading,
/// whether any are, and its packet's hash (which a replay must match).
/// `data-events` holds the input events of the frames that ran, recorded or
/// live, the latest `EVENT_HOOKS`, each as its five lanes (tag, code, x, y,
/// delta); assets' and services' are left out.
pub struct Hooks {
    canvas: HtmlCanvasElement,
    /// Submitted frames, written once their GPU queue slot is released.
    drawn: std::collections::VecDeque<[String; 5]>,
    written: [String; 5],
    events: std::collections::VecDeque<[f32; 5]>,
}

/// The input events `data-events` keeps.
const EVENT_HOOKS: usize = 512;

impl Hooks {
    const NAMES: [&str; 5] = ["data-frame", "data-quality", "data-pending", "aria-busy", "data-packet"];

    pub fn new(canvas: HtmlCanvasElement) -> Hooks {
        Hooks { canvas, drawn: Default::default(), written: Default::default(), events: Default::default() }
    }

    /// A frame ran with `input`, an encoded frame.
    fn ran(&mut self, input: &[f32]) -> Result<()> {
        use pen_protocol::boundary::{EVENT_FLOATS, FRAME_HEADER_FLOATS, event_tags};
        let count = pen_protocol::boundary::whole(input[3]).unwrap_or(0) as usize;
        let before = self.events.len();
        for event in input[FRAME_HEADER_FLOATS..].as_chunks::<EVENT_FLOATS>().0.iter().take(count) {
            let tag = pen_protocol::boundary::whole(event[0]).unwrap_or(0);
            if ![event_tags::ASSET_CHANGED, event_tags::SERVICE_ANSWER, event_tags::SERVICE_DATA].contains(&tag) {
                self.events.push_back(*event);
            }
        }
        if self.events.len() == before {
            return Ok(());
        }
        while self.events.len() > EVENT_HOOKS {
            self.events.pop_front();
        }
        let json = serde_json::to_string(&self.events).map_err(anyhow::Error::from)?;
        self.canvas.set_attribute("data-events", &json).map_err(error)
    }

    /// `drawn` was just submitted; the frame that freed its queue slot finished.
    fn drawn(&mut self, drawn: [String; 5]) -> Result<()> {
        self.drawn.push_back(drawn);
        if self.drawn.len() <= FRAMES_IN_FLIGHT {
            return Ok(());
        }
        let finished = self.drawn.pop_front().unwrap();
        for ((name, value), written) in Hooks::NAMES.iter().zip(finished).zip(&mut self.written) {
            if value != *written {
                self.canvas.set_attribute(name, &value).map_err(error)?;
                *written = value;
            }
        }
        Ok(())
    }
}

/// When the states were last saved and whether they may have changed.
#[derive(Default)]
struct Autosave {
    at: f64,
    /// A frame ran, or the game was restored, since.
    changed: bool,
    /// The hashes of the states written last.
    game: Option<u64>,
    tools: Option<u64>,
}

/// Writes `saved` as the save `name` off the frame's path, unless it is
/// the state written last (`last`, its hash); returns its hash.
fn keep(name: &'static str, saved: SavedState, last: Option<u64>) -> Option<u64> {
    let hash = hash_lanes(&saved.state) ^ u64::from(saved.schema);
    if last != Some(hash) {
        spawn_local(async move {
            if let Err(error) = saves::write(name, &saved).await {
                web_sys::console::warn_1(&format!("keeping {name}: {error:#}").into());
            }
        });
    }
    Some(hash)
}

/// How the page runs the game.
pub struct Options {
    /// Served by `pen web`: new builds load as they come, Bake writes the
    /// checkout, a history is kept from the start, and an invalid frame
    /// pauses the game instead of stopping the page.
    pub development: bool,
    /// Write the hooks the browser harnesses read (`Hooks`).
    pub qualify: bool,
    /// The explicit state adapter selected by `assets/saved.json`.
    pub state_adapter: Arc<dyn StateAdapter>,
    pub assets: String,
    /// The captions the game's sounds carry (the assets' captions.json).
    pub captions: pen_runtime::captions::Captions,
    /// The game's name, which the files it hands the player are named after.
    pub name: String,
    /// Where the page's link has a recording to play (`replay::linked`).
    pub replay: Option<String>,
}

pub struct Browser {
    game: Runtime,
    catalog: Catalog,
    pub screen: Screen,
    input: Rc<RefCell<Input>>,
    audio: Audio,
    accessibility: Accessibility,
    /// The captions of what plays, shared with the captions service.
    captions: pen_runtime::captions::Shared,
    tools: Option<Tools>,
    services: Services,
    preferences: Preferences,
    /// The game's actions, which live frames' events move.
    actions: pen_runtime::actions::Shared,
    /// The game's next live frame's events.
    events: Vec<GameEvent>,
    /// The renderer's asset events while recorded frames play: the game
    /// gets them once it plays live again.
    held: Vec<GameEvent>,
    /// The frame shown, drawn again while paused.
    shown: RenderFrame,
    timeline: Timeline,
    transport: Transport,
    clock: Clock,
    /// Paces recorded frames to their game time: when they started playing
    /// and how much game time has played since.
    pacer: Option<(Instant, f64)>,
    pub log: Log,
    /// The last frame's timings, while measured.
    timings: Option<FrameTimings>,
    development: Option<Development>,
    hooks: Option<Hooks>,
    autosave: Autosave,
    /// The page's status line still says something (it clears once the
    /// game shows).
    status: bool,
    /// Keep a rejected frame's explanation visible until a valid frame replaces it.
    rejected: bool,
    /// Consecutive complete frames; readiness is published after one leaves the GPU queue.
    ready_frames: usize,
    cursor_hidden: bool,
    /// The last animation frame's time.
    previous: Option<f64>,
    state_adapter: Arc<dyn StateAdapter>,
    assets: String,
    /// Files the player hands the page (saves, recordings), and the word on
    /// files handed over.
    files: Files,
    /// The game's name, for the files it hands over.
    name: String,
    /// A save the player handed the page, loaded with the next live frame.
    importing: Option<(String, SavedState)>,
    /// The input recording being made (`/record`).
    recorder: Option<Recorder>,
    /// The recording the page's link names, played first.
    linked: Option<String>,
}

impl Browser {
    pub fn new(
        mut game: Runtime,
        tools: Option<Tools>,
        screen: Screen,
        input: Rc<RefCell<Input>>,
        audio: Audio,
        log: Log,
        options: Options,
    ) -> Result<Browser> {
        let preferences = Preferences::load();
        let actions = Rc::new(RefCell::new(crate::preferences::load_bindings()));
        let mut services = crate::services::with_preferences(preferences.service());
        services.register(Box::new(pen_runtime::actions::InputService(actions.clone())));
        let mut timeline = Timeline::default();
        if options.development {
            timeline.keep_history(Some(KEEP_SECONDS));
        }
        let hooks = options.qualify.then(|| Hooks::new(screen.canvas.element.clone()));
        let captions = options.captions.shared();
        let mut services = crate::services::standard();
        services.register(Box::new(pen_runtime::captions::CaptionsService(captions.clone())));
        game.hash_packets(options.qualify);
        Ok(Browser {
            shown: game.take_drawn().unwrap_or_default(),
            catalog: game.catalog().clone(),
            game,
            screen,
            accessibility: Accessibility::new(input.clone())?,
            input,
            audio,
            tools,
            services,
            captions,
            events: preferences.events().collect(),
            preferences,
            actions,
            held: Vec::new(),
            timeline,
            transport: Transport::default(),
            clock: Clock::real(),
            pacer: None,
            log,
            timings: None,
            development: options.development.then(Development::watch),
            hooks,
            autosave: Autosave::default(),
            status: true,
            rejected: false,
            ready_frames: 0,
            cursor_hidden: false,
            previous: None,
            state_adapter: options.state_adapter,
            assets: options.assets,
            files: Files::install(),
            name: options.name,
            importing: None,
            recorder: None,
            linked: options.replay,
        })
    }

    /// Runs the game until it asks to exit.
    pub async fn run(mut self) -> Result<()> {
        let mut timestamp = web_sys::window().unwrap().performance().unwrap().now() / 1000.0;
        loop {
            let (running, presented) = self.frame(timestamp).await?;
            timestamp = presented;
            if !running {
                break;
            }
        }
        self.finish().await
    }

    /// Compute a frame, then draw at repaint; whether to go on and when it drew.
    async fn frame(&mut self, timestamp: f64) -> Result<(bool, f64)> {
        let interval = self.previous.replace(timestamp).map_or(0.0, |previous| (timestamp - previous) * 1000.0);
        self.screen.fit();
        self.read_mail();
        self.input.borrow_mut().poll_gamepads();
        let hotkeys = self.take_input();
        self.run_tools(timestamp, interval).await;
        self.qualification_request().await?;
        let started = Instant::now();
        for hotkey in hotkeys {
            self.hotkey(hotkey);
        }
        self.take_answers();
        let ran = self.step().await?;
        let overlay = self.tools.as_ref().and_then(|tools| tools.frame.as_ref());
        self.preferences.apply(&mut self.screen.renderer);
        let (presented, draw_timings) = self.screen.draw(&self.shown, overlay, &self.catalog).await?;
        // The dev tools' assets are their own; the game never hears of them.
        let _ = self.screen.overlay.take_events();
        if ran.is_some() {
            self.timings = self.game.last_timings().map(|timings| timings.completed(draw_timings, ms(started)));
        }
        self.present()?;
        crate::preferences::keep_bindings(&self.actions);
        self.autosave(presented).await;
        Ok((ran != Some(true), presented))
    }

    /// The input since the last frame: the game's events, which the dev
    /// tools see first, and the time travel shortcuts, for after the dev
    /// tools' frame.
    fn take_input(&mut self) -> Vec<Hotkey> {
        let (events, actions, hotkeys) = {
            let mut input = self.input.borrow_mut();
            (std::mem::take(&mut input.events), std::mem::take(&mut input.tools), std::mem::take(&mut input.hotkeys))
        };
        for event in events {
            let game = self.tools.as_mut().is_none_or(|tools| tools.route(&event));
            if game {
                self.events.push(event);
            }
        }
        if let Some(tools) = &mut self.tools {
            tools.take(actions);
        }
        hotkeys
    }

    /// The dev tools' frame, which may ask things of the game's.
    async fn run_tools(&mut self, timestamp: f64, interval: f64) {
        if let Some(mut tools) = self.tools.take() {
            tools.frame(self, timestamp, interval).await;
            self.tools = Some(tools);
        }
    }

    /// Headless harnesses exercise the real devtools host without a second
    /// guest/UI fixture. No command input is read outside qualification mode.
    async fn qualification_request(&mut self) -> Result<()> {
        let Some(hooks) = &self.hooks else { return Ok(()) };
        let canvas = hooks.canvas.clone();
        let Some(text) = canvas.get_attribute("data-devtools-request") else { return Ok(()) };
        canvas.remove_attribute("data-devtools-request").map_err(error)?;
        let json: serde_json::Value = serde_json::from_str(&text)?;
        let operation = json
            .get("operation")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| anyhow::anyhow!("qualification request needs an operation"))?;
        let number =
            |key| json.get(key).and_then(|value| serde_json::from_value::<f32>(value.clone()).ok()).unwrap_or(0.0);
        let values =
            json.get("text").map_or_else(Vec::new, |value| pen_runtime::services::code_points(&value.to_string()));
        let request = pen_protocol::ServiceRequest {
            request: 1,
            service: pen_protocol::service::DEVTOOLS,
            operation,
            a: number("a"),
            b: number("b"),
            c: number("c"),
            d: number("d"),
            value_count: u32::try_from(values.len()).map_err(anyhow::Error::from)?,
            values,
        };
        let answer = devtools::Devtools::default().answer(&request, self, 0.0).await;
        let reply = serde_json::json!({"status":answer.status,"value":answer.value,"text":pen_runtime::services::text(&answer.values),"values":answer.values});
        canvas.set_attribute("data-devtools-answer", &reply.to_string()).map_err(error)?;
        Ok(())
    }

    /// What the development endpoints said, into the log.
    fn read_mail(&mut self) {
        if let Some(development) = &self.development {
            for (error, line) in development.take_lines() {
                say(&mut self.log, log_source::SESSION, error, line);
            }
        }
    }

    /// The renderer's asset events and the services' answers, for the
    /// game's next live frame (recorded frames carry their own).
    fn take_answers(&mut self) {
        self.held.extend(self.screen.renderer.take_events());
        self.services.frame(&mut ());
        if self.live() {
            self.events.append(&mut self.held);
            self.events.extend(self.services.take_answers());
        }
    }

    /// Whether the game plays the player's input (not recorded frames).
    fn live(&self) -> bool {
        self.transport.replaying().is_none() && self.timeline.next().is_none()
    }

    fn hotkey(&mut self, hotkey: Hotkey) {
        use devtools::Host as _;
        match hotkey {
            Hotkey::Pause if self.transport.paused() => {
                self.resume();
                say(&mut self.log, log_source::SESSION, false, "playing");
            }
            Hotkey::Pause => {
                self.pause();
                let back = if self.timeline.history() { ", Ctrl+Shift+comma back" } else { "" };
                let frame = self.timeline.position();
                let said = format!("paused at frame {frame} (Ctrl+Shift+period steps{back}, Ctrl+Shift+P plays)");
                say(&mut self.log, log_source::SESSION, false, said);
            }
            Hotkey::Step => self.step_frames(1),
            Hotkey::Back => self.rewind_to(self.transport.target(self.timeline.position()).saturating_sub(1)),
            Hotkey::Replay => self.replay_recent(REPLAY_SECONDS),
        }
    }

    /// The game time stops, if it had not (`was_playing`).
    fn stopped(&mut self, was_playing: bool) {
        if was_playing {
            self.clock.pause();
            self.pacer = None;
        }
    }

    /// Recorded frames play at the game time they took: while the next
    /// one's time has not come, page frames show the last again.
    fn waiting(&mut self) -> bool {
        if self.transport.replaying().is_none() || self.transport.paused() {
            return false;
        }
        let Some(recorded) = self.timeline.next() else { return false };
        let (started, time) = self.pacer.get_or_insert_with(|| (Instant::now(), 0.0));
        if started.elapsed().as_secs_f64() < *time {
            return true;
        }
        *time += recorded.elapsed();
        false
    }

    /// Carries out what was asked of time, then runs the game's frame if
    /// one runs now: Some(whether the game asked to exit) if it did.
    async fn step(&mut self) -> Result<Option<bool>> {
        if let Some(update) = self.development.as_ref().and_then(Development::take_update) {
            self.reload(update).await;
        }
        self.arrivals().await;
        if let Some(path) = self.transport.take_record() {
            match path {
                Some(_) => self.start_recording().await,
                None => self.stop_recording(),
            }
        }
        while let Some(rewind) = self.transport.next_rewind() {
            let result = match rewind {
                Rewind::To(frame) => self.rewind(frame).await,
                Rewind::Replay(seconds) => self.replay(seconds).await,
            };
            if let Err(error) = result {
                say(&mut self.log, log_source::SESSION, true, format!("time travel: {error:#}"));
            }
        }
        if self.waiting() || !self.transport.take_frame() {
            return Ok(None);
        }
        {
            let shell = shell!(self);
            self.timeline.prepare(&mut self.game, &shell).await?;
        }
        let position = self.timeline.position();
        let (input, replayed) = match self.timeline.next() {
            Some(recorded) => {
                // Recorded frames bring their own input.
                self.events.clear();
                (recorded.input.clone(), Some(recorded.loads.clone()))
            }
            None => {
                let mut frame = Frame {
                    timestamp: self.clock.advance(),
                    viewport: self.screen.viewport(),
                    events: std::mem::take(&mut self.events),
                };
                self.actions.borrow_mut().map(&mut frame.events);
                (self.game.input(&frame)?, None)
            }
        };
        if let Some(hooks) = &mut self.hooks {
            hooks.ran(&input)?;
        }
        self.game.send(&input)?;
        let received = {
            let shell = shell!(self);
            self.game.receive(&shell).await
        };
        match received {
            Ok(rendered) => self.complete(input, replayed, rendered).await.map(Some),
            Err(error) => self.reject_frame(position, error).map(|()| None),
        }
    }

    /// Does what a frame and the player asked around it; whether the game
    /// asked to exit.
    async fn complete(&mut self, input: Vec<f32>, replayed: Option<Vec<Load>>, rendered: RenderFrame) -> Result<bool> {
        self.audio.submit(&rendered, &self.catalog).await;
        self.services.rendered(&rendered, &self.catalog);
        if let Some(values) = &rendered.preferences {
            self.preferences.store(values);
        }
        // A live frame's requests (a recorded one's answers are in the
        // recording), before an undo or load replaces the world asking.
        if replayed.is_none() {
            self.services.answer(&rendered.services);
        }
        {
            let mut shell = shell!(self);
            self.game.follow(&mut shell, &rendered).await?;
        }
        let mut loads = Vec::new();
        match replayed {
            // The states it loaded, as they were then.
            Some(recorded) => {
                for load in &recorded {
                    let mut shell = shell!(self);
                    timeline::apply(&mut self.game, &mut shell, load).await?;
                }
                loads = recorded;
            }
            None => {
                if rendered.scene_save_requested {
                    let saved = self.game.scene().await;
                    self.save("pen.scene", saved, "saved the scene").await;
                }
                if rendered.scene_load_requested {
                    loads.extend(self.load("pen.scene", true).await);
                }
                if rendered.save_requested {
                    let saved = self.game.snapshot().await;
                    self.save("pen.quicksave", saved, "saved").await;
                }
                if rendered.load_requested {
                    loads.extend(self.load("pen.quicksave", false).await);
                }
                if let Some((name, saved)) = self.importing.take() {
                    loads.extend(self.load_state(&name, saved, false).await);
                }
            }
        }
        let recorded = Recorded { input, loads, restored: self.timeline.restored_here() };
        if let Some(recorder) = &mut self.recorder {
            recorder.frame(&recorded);
        }
        let exit = rendered.exit_requested;
        let replacement = self.game.take_drawn();
        let replaced = replacement.is_some();
        let shown = replacement.unwrap_or(rendered);
        self.timeline.observe(&shown, replaced);
        self.timeline.ran(recorded);
        self.autosave.changed = true;
        self.show(shown);
        if let Some(position) = self.timeline.observations.take_break() {
            let was_playing = self.transport.breakpoint();
            self.stopped(was_playing);
            say(&mut self.log, log_source::SESSION, false, format!("semantic watch: paused at position {position}"));
        }
        if self.transport.replaying().is_some() && self.timeline.next().is_none() {
            self.replay_ended();
        }
        Ok(exit)
    }

    /// Invalid guest output never commits its candidate state. While
    /// developing, the last valid frame stays and the game pauses until a
    /// new build; otherwise the page stops.
    fn reject_frame(&mut self, position: u64, error: anyhow::Error) -> Result<()> {
        if self.development.is_none() || !error.is::<InvalidFrame>() {
            return Err(error);
        }
        let was_playing = self.transport.reject();
        self.stopped(was_playing);
        self.timings = None;
        self.events.clear();
        self.events.push(GameEvent::FocusLost);
        // A program that stopped answering is gone: a new one restores the last frame's state.
        if error.is::<Unresponsive>() {
            self.transport.restore_at(self.timeline.position());
        }
        let said = format!(
            "dev: frame {position} rejected: {error:#}\ngame paused; fix the source to reload, or use Play to retry"
        );
        self.rejected = true;
        self.status = false;
        if let Some(status) = web_sys::window().unwrap().document().unwrap().get_element_by_id("status") {
            status.set_text_content(Some(&said));
        }
        say(&mut self.log, log_source::GAME, true, said);
        Ok(())
    }

    /// A new frame shows; the one it replaces is decoded into next.
    fn show(&mut self, frame: RenderFrame) {
        if std::mem::take(&mut self.rejected) {
            self.status = true;
        }
        let shown = std::mem::replace(&mut self.shown, frame);
        self.game.recycle(shown);
    }

    async fn save(&mut self, name: &str, saved: Result<SavedState>, done: &str) {
        let written = match saved {
            Ok(saved) => saves::write(name, &saved).await,
            Err(error) => Err(error),
        };
        match written {
            Ok(()) => say(&mut self.log, log_source::SESSION, false, done),
            Err(error) => say(&mut self.log, log_source::SESSION, true, format!("{error:#}")),
        }
    }

    /// Loads a quicksave or (with `scene`) the editor's scene; returns the
    /// state loaded, for the history.
    async fn load(&mut self, name: &str, scene: bool) -> Option<Load> {
        let Some(saved) = saves::read(name).await else {
            say(&mut self.log, log_source::SESSION, true, format!("there is no {name} to load"));
            return None;
        };
        self.load_state(name, saved, scene).await
    }

    /// Loads `saved`, a quicksave or (with `scene`) the editor's scene that
    /// `name` held; returns the state loaded, for the history.
    async fn load_state(&mut self, name: &str, saved: SavedState, scene: bool) -> Option<Load> {
        let loaded = {
            let mut shell = shell!(self);
            if scene {
                self.game.load_scene(&mut shell, &saved).await
            } else {
                self.game.load(&mut shell, &saved).await
            }
        };
        match loaded {
            Ok(()) => {
                self.status = true;
                say(&mut self.log, log_source::SESSION, false, format!("loaded {name}"));
                Some(if scene { Load::Scene(saved) } else { Load::Save(saved) })
            }
            Err(error) => {
                if error.downcast_ref::<pen_protocol::diagnostic::FrameDiagnostic>().is_some() {
                    self.status = false;
                    if let Some(status) = web_sys::window().unwrap().document().unwrap().get_element_by_id("status") {
                        status.set_text_content(Some(&format!("could not load {name}: {error:#}")));
                    }
                }
                say(&mut self.log, log_source::SESSION, true, format!("{error:#}"));
                None
            }
        }
    }

    /// Goes to `target`, paused.
    async fn rewind(&mut self, target: u64) -> Result<()> {
        let started = Instant::now();
        let (shown, at) = {
            let shell = shell!(self);
            let mut unseen = |_: u64, _: &Recorded, _: &Runtime| Ok(());
            self.timeline.rewind(&mut self.game, &shell, target, &mut unseen).await?
        };
        self.services.forget();
        self.game.continue_clock(self.clock.now());
        self.show(shown);
        self.rewind_recording(at, self.timeline.position());
        self.pacer = None;
        self.autosave.changed = true;
        let position = self.timeline.position();
        let ahead = self.timeline.head() - position;
        let said = format!(
            "at frame {position} in {:.0} ms ({ahead} recorded frames ahead: Ctrl+Shift+period runs them, \
             Ctrl+Shift+P plays on from here)",
            ms(started)
        );
        say(&mut self.log, log_source::SESSION, false, said);
        Ok(())
    }

    /// Plays the last `seconds` of recorded frames again.
    async fn replay(&mut self, seconds: f64) -> Result<()> {
        let from = self.timeline.seconds_back(seconds);
        let (shown, at) = {
            let shell = shell!(self);
            self.timeline.restart(&mut self.game, &shell, from).await?
        };
        self.services.forget();
        self.show(shown);
        self.rewind_recording(at, at);
        self.transport.replay(Replaying::Recent);
        self.clock.resume();
        self.pacer = None;
        self.autosave.changed = true;
        let frames = self.timeline.head() - at;
        say(
            &mut self.log,
            log_source::SESSION,
            false,
            format!("replaying {frames} frames from frame {at} on the running build"),
        );
        Ok(())
    }

    /// The replay's frames ran out: play goes on live, or after a recording
    /// played from a file or a link, pauses there.
    fn replay_ended(&mut self) {
        let replayed = self.transport.replay_ended();
        self.game.continue_clock(self.clock.now());
        let position = self.timeline.position();
        let said = if replayed == Some(Replaying::File) {
            devtools::Host::pause(self);
            format!("the replay ended at frame {position}; paused (Ctrl+Shift+P plays on, live)")
        } else {
            format!("the replay ended at frame {position}; playing live")
        };
        say(&mut self.log, log_source::SESSION, false, said);
    }

    /// Files the player handed the page, and the word on those handed over:
    /// a recording (the page's link's first) plays, and a save loads with
    /// the next live frame.
    async fn arrivals(&mut self) {
        if let Some(source) = self.linked.take() {
            match replay::fetch(&source).await {
                Ok((recording, name)) => self.play(recording, &name).await,
                Err(error) => say(&mut self.log, log_source::SESSION, true, format!("replay: {error:#}")),
            }
        }
        for arrival in self.files.take() {
            match arrival {
                Arrival::Note { error, text } => say(&mut self.log, log_source::SESSION, error, text),
                Arrival::File { name, bytes } if recording::is_recording(&bytes) => match recording::decode(&bytes) {
                    Ok(decoded) => self.play(decoded.recording, &name).await,
                    Err(error) => say(&mut self.log, log_source::SESSION, true, format!("{name}: {error:#}")),
                },
                Arrival::File { name, bytes } => match snapshot::decode(&bytes) {
                    Ok(saved) => {
                        // It is the quicksave from now on, as a save from this page would be.
                        let kept = saved.clone();
                        spawn_local(async move {
                            if let Err(error) = saves::write("pen.quicksave", &kept).await {
                                web_sys::console::warn_1(&format!("keeping the imported save: {error:#}").into());
                            }
                        });
                        self.importing = Some((name, saved));
                    }
                    Err(_) => {
                        let said = format!("{name} is neither a save nor an input recording");
                        say(&mut self.log, log_source::SESSION, true, said);
                    }
                },
            }
        }
    }

    /// Plays `recording` (from `name`) in place of the player's input, from
    /// its start; it pauses where it ends.
    async fn play(&mut self, recording: Recording, name: &str) {
        if self.recorder.take().is_some() {
            say(&mut self.log, log_source::SESSION, false, "stopped recording input: a replay starts");
        }
        // Its frames were laid out on its viewport, whatever the page's size.
        self.screen.pin(recording::viewport(&recording));
        let restored = {
            let shell = shell!(self);
            self.game.restore(&mut Quiet(&shell), &recording.start).await
        };
        let shown = match restored {
            Ok(shown) => shown,
            Err(error) => {
                self.screen.pin(None);
                say(&mut self.log, log_source::SESSION, true, format!("replaying {name}: {error:#}"));
                return;
            }
        };
        let frames = recording.frames.len();
        let seconds: f64 = recording.frames.iter().map(Recorded::elapsed).sum();
        self.timeline.keep_history(Some(KEEP_SECONDS));
        self.timeline.load(recording);
        self.services.forget();
        self.show(shown);
        self.transport.replay_ended();
        self.transport.replay(Replaying::File);
        self.pacer = None;
        self.autosave.changed = true;
        let said = format!("replaying {frames} frames ({seconds:.1} s of game time) from {name}");
        say(&mut self.log, log_source::SESSION, false, said);
    }

    /// Records the input from here (`/record`). A replay starts by
    /// restoring the recording's moment, so the game is restored from it
    /// here too, as native pen does: what it keeps outside its state then
    /// starts over in both alike.
    async fn start_recording(&mut self) {
        self.stop_recording();
        let position = self.timeline.position();
        let started = async {
            let moment = self.game.moment().await?;
            let shell = shell!(self);
            let shown = if self.timeline.history() {
                // Rewinds after it then restore from no earlier than its start.
                self.timeline.keep_moment(moment.clone());
                self.timeline.restart(&mut self.game, &shell, position).await?.0
            } else {
                self.game.restore(&mut Quiet(&shell), &moment).await?
            };
            Ok::<_, anyhow::Error>((moment, shown))
        }
        .await;
        match started {
            Ok((moment, shown)) => {
                self.game.continue_clock(self.clock.now());
                self.show(shown);
                self.recorder = Some(Recorder::new(position, moment));
                say(&mut self.log, log_source::SESSION, false, format!("recording input from frame {position}"));
            }
            Err(error) => say(&mut self.log, log_source::SESSION, true, format!("recording input: {error:#}")),
        }
    }

    /// Stops the recording being made, if one is, and hands it to the
    /// player: as a `.penrec` download and as a link that plays it, which
    /// goes on the clipboard (and the browser's console).
    fn stop_recording(&mut self) {
        let Some(recorder) = self.recorder.take() else { return };
        let frames = recorder.frames();
        let bytes = recorder.finish();
        let name = files::named(&self.name, "penrec");
        if let Err(error) = files::download(&name, &bytes, "application/octet-stream") {
            say(&mut self.log, log_source::SESSION, true, format!("handing over {name}: {error:#}"));
        }
        let link = replay::link(&bytes);
        web_sys::console::log_1(&format!("a link that replays {name}: {link}").into());
        say(
            &mut self.log,
            log_source::SESSION,
            false,
            format!(
                "recorded {frames} frames of input as {name} ({} KB; a link to it is {} KB)",
                bytes.len() / 1000,
                link.len() / 1000
            ),
        );
        let files = self.files.clone();
        let copying = web_sys::window().unwrap().navigator().clipboard().write_text(&link);
        spawn_local(async move {
            match JsFuture::from(copying).await {
                Ok(_) => files.note(false, "the link that replays it is on the clipboard"),
                Err(_) => files.note(false, "the link that replays it is in the browser's console"),
            }
        });
    }

    /// The game went back to `at` and ran on to `position` (unseen): the
    /// input recording goes back too, or stops if it started later.
    fn rewind_recording(&mut self, at: u64, position: u64) {
        let Some(recorder) = &mut self.recorder else { return };
        if at < recorder.first {
            self.recorder = None;
            say(&mut self.log, log_source::SESSION, false, "stopped recording input: went back before it started");
            return;
        }
        recorder.rewind(at);
        for frame in at..position {
            if let Some(recorded) = self.timeline.recorded(frame) {
                recorder.frame(recorded);
            }
        }
    }

    /// A new build from `pen web`, taken while the game is idle.
    async fn reload(&mut self, update: Update) {
        self.state_adapter = crate::load_state_adapter(&self.assets).await;
        let catalog = update.catalog.unwrap_or_else(|| self.catalog.clone());
        if let Some(module) = update.game {
            let result = {
                let mut shell = shell!(self);
                self.game.reload(&mut shell, module, catalog.clone()).await
            };
            if result == ReloadResult::Activated {
                self.status = true;
                self.catalog = catalog;
                self.timeline.rebuilt();
                self.autosave.changed = true;
                if let Some(drawn) = self.game.take_drawn() {
                    self.show(drawn);
                }
                if self.transport.reloaded() {
                    devtools::Host::resume(self);
                }
                if self.timeline.history() {
                    let said = format!("dev: Ctrl+Shift+R replays the last {REPLAY_SECONDS} s on the new build");
                    say(&mut self.log, log_source::RELOAD, false, said);
                }
            }
        }
        if let Some(module) = update.devtools {
            let viewport = self.screen.viewport();
            match &mut self.tools {
                Some(tools) => tools.reload(module, &self.catalog, viewport, &mut self.log).await,
                None => match Tools::start(module, &self.catalog, viewport, None, &mut self.log).await {
                    Ok(started) => self.tools = Some(started),
                    Err(error) => say(&mut self.log, log_source::RELOAD, true, format!("devtools: {error:#}")),
                },
            }
        }
        self.screen.renderer.preload(&self.catalog);
        self.screen.renderer.reload_all();
        self.screen.overlay.reload_all();
        self.audio.reload();
    }

    /// What the page shows besides the canvas: the interface for assistive
    /// technology, the pointer, the harnesses' hooks and the status line.
    fn present(&mut self) -> Result<()> {
        let tools = self.tools.as_ref().and_then(|tools| tools.frame.as_ref()).filter(|frame| !frame.ui.is_empty());
        match tools {
            Some(frame) => self.accessibility.update(&frame.ui, true)?,
            None => self.accessibility.update(&self.shown.ui, false)?,
        }
        // The game's text box; the dev tools' console types keys, which an
        // input method would take.
        self.input.borrow_mut().text_field(text_box(&self.shown.ui));
        let captions = pen_runtime::captions::lock(&self.captions);
        let showing = captions.visible(self.shown.show_captions);
        drop(captions);
        self.accessibility.captions(showing.as_deref())?;
        let hidden = self.shown.hide_cursor && !self.tools.as_ref().is_some_and(Tools::wants_pointer);
        if hidden != self.cursor_hidden {
            self.cursor_hidden = hidden;
            let cursor = if hidden { "none" } else { "auto" };
            self.screen.canvas.element.style().set_property("cursor", cursor).map_err(error)?;
        }
        // This one-shot production signal lets a page cover startup without
        // enabling qualification's packet hashing and event instrumentation.
        if self.ready_frames <= FRAMES_IN_FLIGHT {
            self.ready_frames = if self.screen.busy() { 0 } else { self.ready_frames + 1 };
            if self.ready_frames > FRAMES_IN_FLIGHT {
                self.screen.canvas.element.set_attribute("data-ready", "true").map_err(error)?;
            }
        }
        if let Some(hooks) = &mut self.hooks {
            let quality = self.shown.quality;
            hooks.drawn([
                self.timeline.position().to_string(),
                format!("msaa={} scale={}", quality.msaa, quality.scale),
                format!("game={} tools={}", self.screen.renderer.pending(), self.screen.overlay.pending()),
                self.screen.busy().to_string(),
                format!("{:016x}", self.game.packet_hash()),
            ])?;
        }
        if std::mem::take(&mut self.status)
            && let Some(status) = web_sys::window().unwrap().document().unwrap().get_element_by_id("status")
        {
            status.set_text_content(None);
        }
        Ok(())
    }

    /// Every few seconds, the game's and the dev tools' states into the
    /// page's saves, when they changed; written off the frame's path.
    async fn autosave(&mut self, timestamp: f64) {
        if timestamp - self.autosave.at < AUTOSAVE_SECONDS {
            return;
        }
        self.autosave.at = timestamp;
        if std::mem::take(&mut self.autosave.changed) {
            match self.game.snapshot().await {
                Ok(saved) => self.autosave.game = keep("pen.autosave", saved, self.autosave.game),
                Err(error) => say(&mut self.log, log_source::SESSION, true, format!("autosave: {error:#}")),
            }
        }
        if let Some(tools) = &mut self.tools
            && let Some(saved) = tools.snapshot().await
        {
            self.autosave.tools = keep("pen.devtools", saved, self.autosave.tools);
        }
    }

    /// The game asked to exit: its state is kept, and the programs stop.
    async fn finish(mut self) -> Result<()> {
        saves::write("pen.autosave", &self.game.snapshot().await?).await?;
        self.game.close().await;
        if let Some(tools) = &mut self.tools {
            if let Some(saved) = tools.snapshot().await {
                saves::write("pen.devtools", &saved).await?;
            }
            tools.close().await;
        }
        Ok(())
    }
}

impl devtools::Host for Browser {
    fn state_adapter(&self) -> Arc<dyn StateAdapter> {
        self.state_adapter.clone()
    }

    fn observations(&mut self) -> Option<&mut pen_runtime::semantic::Observations> {
        Some(&mut self.timeline.observations)
    }

    fn timeline(&self) -> TimelineStatus {
        let frame = self.timeline.position();
        TimelineStatus {
            frame,
            latest: self.timeline.head().max(frame),
            earliest: self.timeline.earliest(),
            seconds: self.timeline.seconds(),
            paused: self.transport.paused(),
            replaying: self.transport.replaying().is_some(),
            history: self.timeline.history(),
            overlapped: false,
        }
    }

    fn frames_before(&self, frame: u64, seconds: f64) -> u64 {
        self.timeline.frames_before(frame, seconds)
    }

    fn log(&self, from: u64) -> Answer {
        self.log.answer(from)
    }

    fn report(&self) -> FrameReport {
        let stats = self.screen.renderer.stats();
        let viewport = self.screen.viewport();
        FrameReport {
            timings: self.timings,
            draw_calls: stats.draw_calls,
            instanced_draw_calls: stats.instanced_draw_calls,
            instances: stats.instances,
            meshes: self.shown.draws.len(),
            brushes: self.shown.solids.len(),
            memory: self.game.memory_bytes(),
            size: (viewport.width, viewport.height),
            recording: self.recorder.as_ref().map(|recorder| recorder.first),
            culled: stats.culled,
            shadow_culled: stats.shadow_culled,
            triangles: stats.triangles,
            scale: stats.scale,
            gpu: stats.gpu.map(|gpu| [gpu.shadow, gpu.culling, gpu.scene, gpu.post, gpu.overlay, gpu.total]),
        }
    }

    fn measure_frames(&mut self, enabled: bool) {
        self.game.measure_frames(enabled);
        self.screen.renderer.time_passes(enabled);
    }

    fn gpu(&self) -> String {
        self.screen.adapter.clone()
    }

    fn enqueue(&mut self, event: GameEvent) {
        self.events.push(event);
    }

    fn pause(&mut self) {
        let was_playing = self.transport.pause();
        self.stopped(was_playing);
    }

    /// Plays on. After a rewind, play continues live from there, and the
    /// frames that had followed are forgotten; a paused replay plays on.
    fn resume(&mut self) {
        if !self.transport.resume() {
            return;
        }
        if self.transport.replaying().is_none() && self.timeline.next().is_some() {
            self.timeline.go_live();
        }
        // Live play lays out on the page's size again.
        if self.transport.replaying().is_none() {
            self.screen.pin(None);
        }
        self.clock.resume();
        self.pacer = None;
    }

    fn step_frames(&mut self, frames: u32) {
        let was_playing = self.transport.step(frames);
        self.stopped(was_playing);
    }

    fn rewind_to(&mut self, frame: u64) {
        let was_playing = self.transport.rewind_to(frame);
        self.stopped(was_playing);
    }

    fn replay_recent(&mut self, seconds: f64) {
        self.transport.replay_recent(seconds);
    }

    fn keep_history(&mut self) {
        if !self.timeline.history() {
            self.timeline.keep_history(Some(KEEP_SECONDS));
        }
    }

    fn audio(&mut self) -> Option<pen_runtime::mix::AudioReport> {
        self.audio.report()
    }

    fn monitor(&mut self, bus: usize, mute: bool, on: bool) -> bool {
        self.audio.monitor(bus, mute, on)
    }

    /// Starts recording the page's input, or stops and hands the recording
    /// over (a download and a link); a file name is the browser's to give.
    fn record(&mut self, _path: Option<String>) -> Answer {
        self.transport.record(self.recorder.is_none().then(std::path::PathBuf::new));
        Answer::ok(0.0)
    }

    /// Hands the game's state to the player as a save file (`path` names it).
    async fn export_save(&mut self, path: Option<String>) -> Answer {
        let Some(saved) = devtools::Host::snapshot(self).await else { return Answer::failed() };
        let name = path
            .as_deref()
            .map(files::base_name)
            .filter(|name| !name.is_empty())
            .map_or_else(|| files::named(&self.name, "save"), str::to_owned);
        match snapshot::encode(&saved).and_then(|bytes| files::download(&name, &bytes, "application/octet-stream")) {
            Ok(()) => {
                say(&mut self.log, log_source::SESSION, false, format!("exported the game's state as {name}"));
                Answer::text(&name)
            }
            Err(error) => {
                say(&mut self.log, log_source::SESSION, true, format!("exporting the game's state: {error:#}"));
                Answer::failed()
            }
        }
    }

    /// Asks the player for a save (or a recording, which plays), which
    /// loads with the next live frame once chosen; the browser asks.
    fn import_save(&mut self, _path: Option<String>) -> Answer {
        match self.files.choose(".save,.bin,.penrec") {
            Ok(()) => Answer::text("choose a save or a recording"),
            Err(error) => {
                say(&mut self.log, log_source::SESSION, true, format!("importing a save: {error:#}"));
                Answer::failed()
            }
        }
    }

    async fn snapshot(&mut self) -> Option<Arc<SavedState>> {
        match self.game.shared_snapshot().await {
            Ok(saved) => Some(saved),
            Err(error) => {
                say(&mut self.log, log_source::SESSION, true, format!("{error:#}"));
                None
            }
        }
    }

    /// Has `pen web` write the game's tuned values into the checkout as
    /// the defaults; the log says how it went. A static export cannot.
    async fn bake(&mut self) -> Answer {
        if self.development.is_none() {
            return Answer::unavailable();
        }
        let Some(saved) = devtools::Host::snapshot(self).await else { return Answer::unavailable() };
        let Some(tuned) = self.state_adapter.tuned_values(&saved.state) else { return Answer::unavailable() };
        let tuned: Vec<(u32, f32)> = tuned.into_iter().collect();
        if tuned.is_empty() {
            say(&mut self.log, log_source::SESSION, false, "bake: nothing to bake; each value is its default");
            return Answer::ok(0.0);
        }
        match &self.development {
            Some(development) if development.bake(tuned) => Answer::ok(0.0),
            _ => Answer::unavailable(),
        }
    }
}
