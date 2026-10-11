//! Services: what the platform and native libraries do for the game, through
//! one deterministic boundary (gdev's `packages/services.blot`). The game asks
//! in its render packet's `services` section; a [`Service`] answers each
//! request once, and the answer reaches the game as input, a ServiceAnswer
//! event and then ServiceData events carrying its values three at a time, in
//! the next frame that runs live. Answers are input, so an input recording
//! holds them and a replay, rewind or step through recorded frames plays them
//! back without calling a service again: the session calls services for live
//! frames only.
//!
//! The registry has one service per code (`pen_protocol::service`): Steam's
//! achievements, stats and rich presence, the clipboard, opening a web
//! address, what the game runs on, and native grid pathfinding. A request
//! for a code nobody registered is answered `unknown`; one a service cannot
//! do here (Steam is off) `unavailable`.
//!
//! Linking a native library as a service: implement [`Service`] (its code,
//! and an answer for each request, from the request's operation, four
//! numbers and values) and [`Services::register`] it; give it a code in
//! `protocol/schema.json` (`Service`, and its operations' enumeration),
//! regenerate (`just sync-protocol`), and add the request to
//! `packages/services.blot`. `pathfinding.rs` is the example.
//!
//! A service answers at once, or later: [`Service::call`] returns None and
//! [`Service::poll`] hands the answer over when it is ready (a dialog
//! closing). Answers wait in the registry until the next live frame, and a
//! restore (a load, undo, reload or time travel) drops those waiting: they
//! were for the world it replaced. The game saves its record of what it
//! asked with its state, so time travel's recorded frames bring it the
//! answers it waited for, and after a load it asks again, with the same ids,
//! for what the loaded world still waits on (gdev's `retry_ticks` later).
use pen_protocol::{GameEvent, ServiceRequest, service, service_status};

use crate::input::Input;

pub mod clipboard;
pub mod pathfinding;
pub mod platform;
pub mod steam;
pub mod url;

pub use pen_runtime::services::{Answer, Request, text};
#[cfg(any(feature = "devtools", test))]
pub use pen_runtime::services::code_points;

/// A service the game can ask: the platform's, or a native library's.
pub trait Service {
    /// The service code it answers (`pen_protocol::service`).
    fn code(&self) -> u32;

    /// Answers a request now, or returns None and answers it later through
    /// `poll`. Called for live frames only, once per request.
    fn call(&mut self, request: &Request) -> Option<Answer>;

    /// Answers that came in since the last frame, by request id.
    fn poll(&mut self) -> Vec<(u32, Answer)> {
        Vec::new()
    }

    /// Once a frame, before the game's (Steam pumps its callbacks here, and
    /// may tell the game focus was lost).
    fn frame(&mut self, _input: &mut Input) {}
}

/// The services a session answers with, and the answers waiting for the
/// next live frame.
#[derive(Default)]
pub struct Services {
    services: Vec<Box<dyn Service>>,
    answers: Vec<GameEvent>,
    /// Requests answered (or taken to answer later) so far.
    calls: u64,
    /// A line on stderr for each request (`PEN_SERVICES_LOG`).
    log: bool,
}

/// What a request was, for the log: `steam unlock_achievement "FIRST_BRICK"`.
fn describe(request: &ServiceRequest) -> String {
    let name = |names: &[&str], code: u32| names.get(code as usize).map_or(code.to_string(), |name| (*name).to_owned());
    let operations: &[&str] = match request.service {
        service::STEAM => &pen_protocol::steam_op::NAMES,
        service::CLIPBOARD => &pen_protocol::clipboard_op::NAMES,
        service::URL => &pen_protocol::url_op::NAMES,
        service::PLATFORM => &pen_protocol::platform_op::NAMES,
        service::PATHFINDING => &pen_protocol::pathfinding_op::NAMES,
        _ => &[],
    };
    let shown = if request.service == service::PATHFINDING {
        let blocked = request.values.iter().filter(|value| **value != 0.0).count();
        format!("{} cells, {blocked} blocked", request.values.len())
    } else {
        let text = text(&request.values);
        let text = if text.chars().count() > 40 { format!("{}…", text.chars().take(40).collect::<String>()) } else { text };
        format!("{text:?}")
    };
    format!(
        "{} {} {shown} [{}, {}, {}, {}]",
        name(&service::NAMES, request.service),
        name(operations, request.operation),
        request.a,
        request.b,
        request.c,
        request.d
    )
}

impl Services {
    /// None registered: every request is answered `unknown`.
    pub fn new() -> Services {
        Services { log: std::env::var_os("PEN_SERVICES_LOG").is_some(), ..Services::default() }
    }

    /// The runtime's own: Steam (unavailable until [`steam::SteamService`]
    /// with a client replaces it), the clipboard, web addresses, the
    /// platform and pathfinding.
    pub fn standard() -> Services {
        let mut services = Services::new();
        services.register(Box::new(steam::SteamService::off()));
        services.register(Box::new(clipboard::Clipboard::default()));
        services.register(Box::new(url::Url));
        services.register(Box::new(platform::Platform));
        services.register(Box::new(pathfinding::Pathfinding));
        services
    }

    /// Adds a service, in place of any with the same code.
    pub fn register(&mut self, added: Box<dyn Service>) {
        self.services.retain(|registered| registered.code() != added.code());
        self.services.push(added);
    }

    /// Requests answered so far.
    pub fn calls(&self) -> u64 {
        self.calls
    }

    /// Once a frame: the services' own work, and answers that came in.
    pub fn frame(&mut self, input: &mut Input) {
        for registered in &mut self.services {
            registered.frame(input);
            for (id, answer) in registered.poll() {
                if self.log {
                    eprintln!("service: request {id} answered {}", service_status::NAMES[answer.status as usize]);
                }
                self.answers.extend(answer.events(id));
            }
        }
    }

    /// Asks the services a live frame's requests; their answers go to the
    /// game with the next live frame.
    pub fn answer(&mut self, requests: &[ServiceRequest]) {
        for request in requests {
            self.calls += 1;
            let asked = Request {
                id: request.request,
                operation: request.operation,
                numbers: [request.a, request.b, request.c, request.d],
                values: &request.values,
            };
            let answer = match self.services.iter_mut().find(|registered| registered.code() == request.service) {
                Some(registered) => registered.call(&asked),
                None => Some(Answer::unknown()),
            };
            if self.log {
                let outcome = answer.as_ref().map_or("later".to_owned(), |answer| {
                    format!("{} {} ({} values)", service_status::NAMES[answer.status as usize], answer.value, answer.values.len())
                });
                eprintln!("service: request {} {} -> {outcome}", asked.id, describe(request));
            }
            if let Some(answer) = answer {
                self.answers.extend(answer.events(asked.id));
            }
        }
    }

    /// The answers for the next live frame, in the order they came.
    pub fn take_answers(&mut self) -> Vec<GameEvent> {
        std::mem::take(&mut self.answers)
    }

    /// The game's world was replaced (a load, undo, reload or time travel):
    /// the answers for the one that asked are dropped.
    pub fn forget(&mut self) {
        self.answers.clear();
    }

    /// How many requests a session answered, once it ends.
    pub fn report(&self) {
        if self.calls() > 0 {
            eprintln!("services: answered {} requests", self.calls());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts its calls and answers each with its id.
    struct Counting {
        calls: std::rc::Rc<std::cell::Cell<u32>>,
    }

    impl Service for Counting {
        fn code(&self) -> u32 {
            service::PLATFORM
        }

        fn call(&mut self, request: &Request) -> Option<Answer> {
            self.calls.set(self.calls.get() + 1);
            Some(Answer { status: service_status::OK, value: request.id as f32, values: code_points("linux") })
        }
    }

    fn request(id: u32, service: u32, values: Vec<f32>) -> ServiceRequest {
        ServiceRequest { request: id, service, value_count: values.len() as u32, values, ..ServiceRequest::default() }
    }

    #[test]
    fn answers_travel_as_events_three_values_at_a_time() {
        let events = Answer::text("linux").events(7);
        assert_eq!(
            events,
            [
                GameEvent::ServiceAnswer { request: 7, status: service_status::OK, value: 0.0, length: 5 },
                GameEvent::ServiceData { request: 7, a: 108.0, b: 105.0, c: 110.0 },
                GameEvent::ServiceData { request: 7, a: 117.0, b: 120.0, c: 0.0 },
            ]
        );
        assert_eq!(Answer::ok(2.5).events(1).len(), 1);
        assert_eq!(text(&code_points("Steam Deck é")), "Steam Deck é");
        assert_eq!(text(&[65.0, 0.5, -1.0]), "A\u{fffd}\u{fffd}");
    }

    #[test]
    fn requests_reach_their_service_once_and_unknown_ones_say_so() {
        let calls = std::rc::Rc::new(std::cell::Cell::new(0));
        let mut services = Services::new();
        services.register(Box::new(Counting { calls: calls.clone() }));
        services.answer(&[request(3, service::PLATFORM, vec![]), request(4, 99, vec![])]);
        assert_eq!(calls.get(), 1);
        assert_eq!(services.calls(), 2);
        let answers = services.take_answers();
        assert_eq!(answers[0], GameEvent::ServiceAnswer { request: 3, status: service_status::OK, value: 3.0, length: 5 });
        assert_eq!(answers.last(), Some(&GameEvent::ServiceAnswer { request: 4, status: service_status::UNKNOWN, value: 0.0, length: 0 }));
        assert!(services.take_answers().is_empty());
        // A restore drops what waits for the next frame.
        services.answer(&[request(5, service::PLATFORM, vec![])]);
        services.forget();
        assert!(services.take_answers().is_empty());
    }

    #[test]
    fn the_standard_services_answer() {
        let mut services = Services::standard();
        let os = pen_protocol::platform_op::OS;
        let mut asked = request(1, service::PLATFORM, vec![]);
        asked.operation = os;
        // Steam off: a clean unavailable answer, never a client started.
        let mut steam = request(2, service::STEAM, code_points("FIRST_BRICK"));
        steam.operation = pen_protocol::steam_op::UNLOCK_ACHIEVEMENT;
        services.answer(&[asked, steam]);
        let answers = services.take_answers();
        assert!(matches!(answers[0], GameEvent::ServiceAnswer { request: 1, status: service_status::OK, .. }));
        assert!(answers.contains(&GameEvent::ServiceAnswer {
            request: 2,
            status: service_status::UNAVAILABLE,
            value: 0.0,
            length: 0
        }));
    }
}

/// The game itself (`dist/gdev`, from `just export`), without a GPU: a
/// counting fake answers a scripted run's requests, and the run's input
/// replays the same packets without asking it again. Skipped when the
/// bundle is missing.
#[cfg(test)]
mod replay {
    use std::cell::Cell;
    use std::path::{Path, PathBuf};
    use std::rc::Rc;

    use anyhow::Result;
    use pen_protocol::boundary::read_catalog;
    use pen_protocol::{Catalog, Frame, RenderFrame, Viewport, service};

    use super::*;
    use crate::game::{GameRuntime, Host};
    use crate::script::{self, Script};
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

    /// Answers every request of its service with the text "fake", counting.
    struct Fake {
        code: u32,
        calls: Rc<Cell<u32>>,
    }

    impl Service for Fake {
        fn code(&self) -> u32 {
            self.code
        }

        fn call(&mut self, _request: &Request) -> Option<Answer> {
            self.calls.set(self.calls.get() + 1);
            Some(Answer::text("fake"))
        }
    }

    fn bundle() -> Option<PathBuf> {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dist/gdev");
        let found = directory.join("game.wasm").exists();
        if !found {
            eprintln!("skipped: no dist/gdev (`just export`)");
        }
        found.then_some(directory)
    }

    fn start(directory: &Path) -> GameRuntime {
        let engine = blot_host::engine().unwrap();
        let wasm = std::fs::read(directory.join("game.wasm")).unwrap();
        let cache = std::env::temp_dir().join("pen-test-wasm");
        let module = blot_host::GuestModule::new(&engine, &wasm, Some(&cache)).unwrap();
        let catalog = read_catalog(&directory.join("assets")).unwrap();
        let scene = snapshot::read(&directory.join("main.scene")).ok();
        let mut runtime = GameRuntime::new(&engine, module, catalog, &mut Null, None, scene).unwrap();
        runtime.hash_packets(true);
        runtime
    }

    /// Plays the editor for half a second, then opens the settings' About
    /// page, which shows what the platform service answered. Returns each
    /// frame's input and packet hash.
    fn play(runtime: &mut GameRuntime, services: &mut Services) -> (Vec<Vec<f32>>, Vec<u64>) {
        let steps = "wait=30 key=F2 wait=5 key=PageDown wait=2 key=PageDown wait=2 key=PageDown wait=30";
        let mut script = Script::new(script::parse(&[steps.to_owned()]).unwrap());
        let (mut inputs, mut packets) = (Vec::new(), Vec::new());
        let mut time = 0.0;
        while !script.done() {
            let mut events = script.tick(VIEWPORT).events;
            events.extend(services.take_answers());
            let input = runtime.input(&Frame { timestamp: time, viewport: VIEWPORT, events }).unwrap();
            time += 1.0 / 60.0;
            runtime.send(&input).unwrap();
            let rendered = runtime.receive(&Null).unwrap();
            services.answer(&rendered.services);
            runtime.follow(&mut Null, &rendered).unwrap();
            packets.push(runtime.packet_hash());
            inputs.push(input);
        }
        (inputs, packets)
    }

    #[test]
    fn a_recording_replays_the_answers_without_asking_again() {
        let Some(directory) = bundle() else { return };
        let calls = Rc::new(Cell::new(0));
        let mut services = Services::new();
        for code in [service::STEAM, service::CLIPBOARD, service::URL, service::PLATFORM, service::PATHFINDING] {
            services.register(Box::new(Fake { code, calls: calls.clone() }));
        }
        let mut live = start(&directory);
        let moment = live.moment().unwrap();
        let (inputs, packets) = play(&mut live, &mut services);
        live.close();
        // The platform's three questions, Steam's status, its stat and its
        // presence at least, each asked once.
        let asked = calls.get();
        assert!(asked >= 6, "{asked} calls");
        assert_eq!(u64::from(asked), services.calls());
        let answer_events = |input: &[f32]| {
            let count = input[3] as usize;
            (0..count).filter(|index| input[5 + index * 5] == 13.0).count()
        };
        assert_eq!(inputs.iter().map(|input| answer_events(input)).sum::<usize>(), asked as usize, "every answer is input");

        // The replay: the recorded input from the same moment, no services.
        let mut replay = start(&directory);
        replay.restore(&mut Null, &moment).unwrap();
        let mut replayed = Vec::new();
        for input in &inputs {
            replay.send(input).unwrap();
            let rendered = replay.receive(&Null).unwrap();
            replay.follow(&mut Null, &rendered).unwrap();
            replayed.push(replay.packet_hash());
        }
        replay.close();
        assert_eq!(replayed, packets, "the replay gives the packets the run did");
        assert_eq!(calls.get(), asked, "and asks nothing");

        // The answers matter: a host without services (every request
        // answered unknown) shows the About page otherwise.
        let mut bare = start(&directory);
        let (_, without) = play(&mut bare, &mut Services::new());
        bare.close();
        assert_ne!(without.last(), packets.last());
    }
}
