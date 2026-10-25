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
//! The registry ([`Services`], pen-runtime's, which the browser shares)
//! has one service per code (`pen_protocol::service`): Steam's
//! achievements, stats and rich presence, the clipboard, opening a web
//! address, what the game runs on, and noise (`pen_simulation`). A request for a code nobody registered is answered `unknown`;
//! one a service cannot do here (Steam is off) `unavailable`.
//!
//! Linking a native library as a service: implement [`Service`] (its code,
//! and an answer for each request, from the request's operation, four
//! numbers and values) and register it ([`Services::register`]); give it a
//! code in `protocol/schema.json` (`Service`, and its operations'
//! enumeration), regenerate (`just sync-protocol`), and add the request to
//! `packages/services.blot`. `pen_simulation::Noise` is one example.
//!
//! A service answers at once, or later: [`Service::call`] returns None and
//! [`Service::poll`] hands the answer over when it is ready (a dialog
//! closing). Answers wait in the registry until the next live frame, and a
//! restore (a load, undo, reload or time travel) drops those waiting: they
//! were for the world it replaced. The game saves its record of what it
//! asked with its state, so time travel's recorded frames bring it the
//! answers it waited for, and after a load it asks again, with the same ids,
//! for what the loaded world still waits on (gdev's `retry_ticks` later).
use crate::input::Input;

pub mod clipboard;
pub mod locale;
pub mod platform;
pub mod steam;
pub mod url;

#[cfg(test)]
use pen_runtime::services::code_points;
pub use pen_runtime::services::{Answer, Request};

// A service the game can ask: the platform's, or a native library's; once a
// frame it may tell the game's input queue something (Steam's overlay).
pub use pen_runtime::services::Service;

/// The services a session answers with, and the answers waiting for the
/// next live frame.
pub type Services = pen_runtime::services::Services<Input>;

/// The runtime's own: Steam (unavailable until [`steam::SteamService`] with
/// a client replaces it), the clipboard, web addresses, the platform,
/// noise (`pen_simulation`) and the
/// player's consent to reports (`crate::reports`).
pub fn standard() -> Services {
    let mut services = Services::new();
    services.register(Box::new(steam::SteamService::off()));
    services.register(Box::new(clipboard::Clipboard::default()));
    services.register(Box::new(url::Url));
    services.register(Box::new(platform::platform()));
    pen_simulation::register(&mut services);
    services.register(Box::new(crate::reports::Reports::new()));
    services
}

#[cfg(test)]
mod tests {
    use pen_protocol::{GameEvent, ServiceRequest, service, service_status};

    use super::*;

    fn request(id: u32, service: u32, values: Vec<f32>) -> ServiceRequest {
        ServiceRequest {
            request: id,
            service,
            value_count: u32::try_from(values.len()).unwrap(),
            values,
            ..ServiceRequest::default()
        }
    }

    #[test]
    fn the_standard_services_answer() {
        let mut services = standard();
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

    #[test]
    fn navigation_and_physics_are_guest_owned() {
        let mut services = standard();
        for code in [service::PATHFINDING, service::NAVMESH, service::PHYSICS, service::SPATIAL] {
            services.answer(&[request(code, code, vec![])]);
            assert_eq!(
                services.take_answers(),
                vec![GameEvent::ServiceAnswer {
                    request: code,
                    status: service_status::UNKNOWN,
                    value: 0.0,
                    length: 0,
                }]
            );
        }
    }

    /// The simulation services answer native pen's requests with the bits
    /// every host must give (the browser checks the same digest).
    #[test]
    fn the_simulation_services_answer_as_every_host_does() {
        pen_simulation::conformance::check(&mut standard()).unwrap();
    }
}

/// The game itself (`dist/gdev`, from `just export`), without a GPU: a
/// counting fake answers a scripted run's requests, and the run's input
/// replays the same packets without asking it again. Skipped when the
/// bundle is missing (`fixtures`).
#[cfg(test)]
mod replay {
    use std::cell::Cell;
    use std::path::Path;
    use std::rc::Rc;

    use anyhow::Result;
    use pen_protocol::boundary::read_catalog;
    use pen_protocol::{Catalog, Frame, RenderFrame, Viewport, service};

    use super::*;
    use crate::fixtures::bundle;
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

    impl Service<Input> for Fake {
        fn code(&self) -> u32 {
            self.code
        }

        fn call(&mut self, _request: &Request) -> Option<Answer> {
            self.calls.set(self.calls.get() + 1);
            Some(Answer::text("fake"))
        }
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
        let steps =
            "wait=100 console=7,0 wait=30 key=F2 wait=5 key=PageDown wait=2 key=PageDown wait=2 key=PageDown wait=30";
        let mut script = Script::new(script::parse(&[steps.to_owned()]).unwrap());
        let (mut inputs, mut packets) = (Vec::new(), Vec::new());
        let mut time = 0.0;
        while !script.done() {
            let mut events = script.tick(VIEWPORT).events;
            if inputs.is_empty() {
                events.insert(0, pen_protocol::GameEvent::RendererReady);
            }
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
        let codes = [
            service::STEAM,
            service::CLIPBOARD,
            service::URL,
            service::PLATFORM,
            service::PATHFINDING,
            service::PREFERENCES,
            service::LOCALE,
            service::REPORTS,
        ];
        for code in codes {
            services.register(Box::new(Fake { code, calls: calls.clone() }));
        }
        let mut live = start(&directory);
        let moment = live.moment().unwrap();
        let (inputs, packets) = play(&mut live, &mut services);
        live.close();
        // The platform's three questions, Steam's status, its stat and its
        // presence at least, and its words, each asked once.
        let asked = calls.get();
        assert!(asked >= 6, "{asked} calls");
        assert_eq!(u64::from(asked), services.calls());
        let answer_events = |input: &[f32]| {
            let count = pen_protocol::boundary::whole(input[3]).unwrap() as usize;
            (0..count).filter(|index| input[5 + index * 5] == 13.0).count()
        };
        assert_eq!(
            inputs.iter().map(|input| answer_events(input)).sum::<usize>(),
            asked as usize,
            "every answer is input"
        );

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
