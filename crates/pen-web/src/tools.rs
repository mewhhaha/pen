//! The dev tools in the browser: the game's own Blot program for them
//! (`devtools.wasm`), run beside the game as native pen runs it, its
//! sprites drawn over the game's frame. Its requests are answered through
//! pen-runtime's dev tools service, the page's frame loop (`Browser`) being
//! its host; the input it takes from the game follows the same `Routing`.
use anyhow::Result;
use pen_protocol::{Catalog, Frame, GameEvent, RenderFrame, Viewport, service};
use pen_runtime::devtools::{Devtools, Host as _, Log, Routing, log_source};
use pen_runtime::game::Host;
use pen_runtime::services::Answer;
use pen_runtime::snapshot::SavedState;

use crate::program::BrowserBackend;
use crate::run::{Browser, Runtime, say};

/// What the program's runtime needs of the page: the viewport, checks, and
/// the log for its word on new builds. Its frames are drawn over the game's
/// by the frame loop, not here.
struct Shell<'a> {
    viewport: Viewport,
    log: &'a mut Log,
}

impl Host for Shell<'_> {
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
        say(self.log, log_source::RELOAD, false, format!("devtools: {message}"));
    }
}

pub struct Tools {
    /// The program, until it stops on an error (a new build starts it again).
    runtime: Option<Runtime>,
    /// Its last frame: the sprites drawn and the interface routing input.
    pub frame: Option<RenderFrame>,
    /// Events since its last frame, and the answers to its requests for its next.
    events: Vec<GameEvent>,
    answers: Vec<GameEvent>,
    pub routing: Routing,
    service: Devtools,
}

impl Tools {
    /// Starts the program, from `saved` if it can take that state.
    pub async fn start(
        module: Vec<u8>,
        catalog: &Catalog,
        viewport: Viewport,
        saved: Option<SavedState>,
        log: &mut Log,
    ) -> Result<Tools> {
        let mut shell = Shell { viewport, log };
        let started =
            match Runtime::new(&BrowserBackend, module.clone(), catalog.clone(), &mut shell, saved, None).await {
                Ok(runtime) => runtime,
                // A state it cannot take starts it afresh.
                Err(_) => Runtime::new(&BrowserBackend, module, catalog.clone(), &mut shell, None, None).await?,
            };
        Ok(Tools {
            runtime: Some(started),
            frame: None,
            events: Vec::new(),
            answers: Vec::new(),
            routing: Routing::default(),
            service: Devtools::default(),
        })
    }

    /// A new build of the program: in place of the running one, with its
    /// state, or started afresh if it had stopped.
    pub async fn reload(&mut self, module: Vec<u8>, catalog: &Catalog, viewport: Viewport, log: &mut Log) {
        match &mut self.runtime {
            Some(runtime) => {
                runtime.reload(&mut Shell { viewport, log }, module, catalog.clone()).await;
            }
            None => match Tools::start(module, catalog, viewport, None, log).await {
                Ok(started) => self.runtime = started.runtime,
                Err(error) => {
                    say(log, log_source::RELOAD, true, format!("devtools: the program does not start: {error:#}"))
                }
            },
        }
    }

    /// An event for the game, the program seeing it first: whether the
    /// game gets it too.
    pub fn route(&mut self, event: &GameEvent) -> bool {
        let ui = self.frame.as_ref().map_or(&[][..], |frame| &frame.ui);
        let Some(game) = self.routing.route(event, ui, self.runtime.is_some()) else { return true };
        self.events.push(event.clone());
        game
    }

    /// Events for the program alone (assistive technology's actions on its
    /// interface).
    pub fn take(&mut self, events: Vec<GameEvent>) {
        self.events.extend(events);
    }

    /// Whether the pointer is the program's (over one of its dialogs), so
    /// it shows even where the game hides it.
    pub fn wants_pointer(&self) -> bool {
        self.runtime.is_some() && self.routing.over
    }

    /// Runs its frame on the events since its last, and does what it asks
    /// of `browser`; `interval` is the time since its last frame, in ms.
    pub async fn frame(&mut self, browser: &mut Browser, timestamp: f64, interval: f64) {
        let Some(runtime) = &mut self.runtime else { return };
        let mut events = std::mem::take(&mut self.answers);
        events.append(&mut self.events);
        let viewport = browser.screen.viewport();
        let shell = Shell { viewport, log: &mut browser.log };
        let ran = match runtime.input(&Frame { timestamp, viewport, events }).and_then(|input| runtime.send(&input)) {
            Ok(()) => runtime.receive(&shell).await,
            Err(error) => Err(error),
        };
        let rendered = match ran {
            Ok(rendered) => rendered,
            Err(error) => {
                // Stopped on an error: it waits for a new build.
                self.runtime = None;
                self.frame = None;
                self.routing.stopped();
                say(&mut browser.log, log_source::RELOAD, true, format!("devtools: stopped: {error:#}"));
                return;
            }
        };
        for request in &rendered.services {
            let answer = if request.service == service::DEVTOOLS {
                self.service.answer(request, browser, interval).await
            } else {
                Answer::unknown()
            };
            self.answers.extend(answer.events(request.request));
        }
        // Keys held when it took the keyboard would never come up for the
        // game: it lets go of them, as when the page loses the focus.
        if self.routing.laid_out(&rendered.ui) {
            browser.enqueue(GameEvent::FocusLost);
        }
        if let Some(runtime) = &mut self.runtime
            && let Some(last) = self.frame.replace(rendered)
        {
            runtime.recycle(last);
        }
    }

    /// Its state, to keep between visits.
    pub async fn snapshot(&mut self) -> Option<SavedState> {
        self.runtime.as_mut()?.snapshot().await.ok()
    }

    pub async fn close(&mut self) {
        if let Some(runtime) = &mut self.runtime {
            runtime.close().await;
        }
    }
}
