//! Steam Input: the game's actions (`pen_runtime::actions`) read from the
//! controllers Steam configures, and the Steam service's input operations
//! (`pen_protocol::steam_op`): activating an action set, the glyph of what
//! is bound to an action, and Steam's binding panel, where the player
//! rebinds. Steam Input knows the game's actions from its action manifest
//! (the app's Steamworks configuration, or `PEN_STEAM_INPUT_MANIFEST`, a
//! local file, while developing), by the names the game declares them with:
//! a button action reads the digital action of its name (or an analog one's
//! x, a trigger), and an axis action the analog action of its name, or with
//! `.x` or `.y` after the analog action's name, that axis of it (`move.y`
//! reads `move`'s y), as Steam gives it. What Steam reads joins the action's
//! own bindings, and the Action events it moves say `INPUT_DEVICE_STEAM`.
//!
//! The logic is generic over [`Api`], so it is tested without Steam;
//! `Steamworks` (the `steam` feature) is the Steam client's.
// Without the `steam` feature, only the tests use it.
#![cfg_attr(not(feature = "steam"), allow(dead_code))]
use std::collections::HashMap;

use pen_runtime::actions::{Actions, Kind};
use pen_runtime::services::Answer;

/// What pen asks of Steam Input; handles are Steam's, 0 for none.
pub trait Api {
    /// Reads the controllers' newest state.
    fn run_frame(&mut self);
    fn controllers(&mut self) -> Vec<u64>;
    fn action_set(&mut self, name: &str) -> u64;
    fn activate(&mut self, controller: u64, set: u64);
    fn digital_action(&mut self, name: &str) -> u64;
    fn analog_action(&mut self, name: &str) -> u64;
    /// Whether a digital action is held, or None while it is not in the
    /// controller's active set.
    fn digital(&mut self, controller: u64, action: u64) -> Option<bool>;
    /// An analog action's x and y, or None while it is not active.
    fn analog(&mut self, controller: u64, action: u64) -> Option<(f32, f32)>;
    /// The glyph images (their paths) of what is bound to a digital action
    /// (or, when that is 0, an analog one) in a set, on a controller.
    fn glyphs(&mut self, controller: u64, set: u64, digital: u64, analog: u64) -> Vec<String>;
    /// Opens Steam's binding panel for a controller; whether it opened.
    fn binding_panel(&mut self, controller: u64) -> bool;
}

/// Where an action reads from in Steam Input.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Source {
    Digital(u64),
    /// An analog action's x (false) or y (true).
    Analog(u64, bool),
    Nothing,
}

pub struct SteamInput<A> {
    api: A,
    /// The action set the game activated, by name and handle.
    set: Option<(String, u64)>,
    /// Where each action reads from, by its name and kind.
    sources: HashMap<(String, bool), Source>,
}

impl<A: Api> SteamInput<A> {
    pub fn new(api: A) -> SteamInput<A> {
        SteamInput { api, set: None, sources: HashMap::new() }
    }

    fn source(&mut self, name: &str, kind: Kind) -> Source {
        let key = (name.to_owned(), kind == Kind::Axis);
        if let Some(source) = self.sources.get(&key) {
            return *source;
        }
        let api = &mut self.api;
        let analog = |api: &mut A, name: &str, y: bool| match api.analog_action(name) {
            0 => None,
            handle => Some(Source::Analog(handle, y)),
        };
        let digital = |api: &mut A, name: &str| match api.digital_action(name) {
            0 => None,
            handle => Some(Source::Digital(handle)),
        };
        let found = match kind {
            Kind::Button => digital(api, name).or_else(|| analog(api, name, false)),
            Kind::Axis => {
                let component = name.strip_suffix(".x").map(|stem| (stem, false));
                let component = component.or_else(|| name.strip_suffix(".y").map(|stem| (stem, true)));
                component
                    .and_then(|(stem, y)| analog(api, stem, y))
                    .or_else(|| analog(api, name, false))
                    .or_else(|| digital(api, name))
            }
        };
        let source = found.unwrap_or(Source::Nothing);
        self.sources.insert(key, source);
        source
    }

    /// Once a frame: every declared action as the controllers read it, the
    /// one furthest from rest, into `actions`.
    pub fn frame(&mut self, actions: &mut Actions) {
        self.api.run_frame();
        let controllers = self.api.controllers();
        if let Some((_, set)) = self.set {
            // Cheap, so controllers that came since get it too.
            for controller in &controllers {
                self.api.activate(*controller, set);
            }
        }
        let declared: Vec<(u32, String, Kind)> =
            actions.declared().map(|(index, action)| (index, action.name.clone(), action.kind)).collect();
        for (index, name, kind) in declared {
            let source = self.source(&name, kind);
            let mut value: f32 = 0.0;
            for controller in &controllers {
                let read = match source {
                    Source::Digital(handle) => {
                        self.api.digital(*controller, handle).map(|held| f32::from(u8::from(held)))
                    }
                    Source::Analog(handle, y) => {
                        self.api.analog(*controller, handle).map(|(along, down)| if y { down } else { along })
                    }
                    Source::Nothing => None,
                };
                if let Some(read) = read.filter(|read| read.abs() > value.abs()) {
                    value = read;
                }
            }
            actions.set_external(index, value);
        }
    }

    /// Activates the action set named `name` on every controller, and on
    /// those that come later.
    pub fn activate_set(&mut self, name: &str) -> Answer {
        let set = self.api.action_set(name);
        if set == 0 {
            return Answer::failed();
        }
        self.set = Some((name.to_owned(), set));
        let controllers = self.api.controllers();
        for controller in &controllers {
            self.api.activate(*controller, set);
        }
        if controllers.is_empty() { Answer::unavailable() } else { Answer::ok(controllers.len() as f32) }
    }

    /// The glyph image of what is bound first to action `name` (a digital
    /// action, or an analog one) on the first controller, in the active set.
    pub fn glyph(&mut self, name: &str) -> Answer {
        let Some(controller) = self.api.controllers().first().copied() else { return Answer::unavailable() };
        let set = self.set.as_ref().map_or(0, |(_, set)| *set);
        let (digital, analog) = match self.api.digital_action(name) {
            0 => (0, self.api.analog_action(name)),
            digital => (digital, 0),
        };
        if digital == 0 && analog == 0 {
            return Answer::failed();
        }
        match self.api.glyphs(controller, set, digital, analog).first() {
            Some(path) => Answer::text(path),
            None => Answer::failed(),
        }
    }

    /// Opens Steam's binding panel for the first controller.
    pub fn panel(&mut self) -> Answer {
        let Some(controller) = self.api.controllers().first().copied() else { return Answer::unavailable() };
        if self.api.binding_panel(controller) { Answer::ok(1.0) } else { Answer::failed() }
    }
}

/// The Steam client's Steam Input.
#[cfg(feature = "steam")]
pub struct Steamworks(pub steamworks::Input);

/// Steam Input stops with its user.
#[cfg(feature = "steam")]
impl Drop for Steamworks {
    fn drop(&mut self) {
        self.0.shutdown();
    }
}

#[cfg(feature = "steam")]
impl Api for Steamworks {
    fn run_frame(&mut self) {
        self.0.run_frame();
    }

    fn controllers(&mut self) -> Vec<u64> {
        self.0.get_connected_controllers()
    }

    fn action_set(&mut self, name: &str) -> u64 {
        self.0.get_action_set_handle(name)
    }

    fn activate(&mut self, controller: u64, set: u64) {
        self.0.activate_action_set_handle(controller, set);
    }

    fn digital_action(&mut self, name: &str) -> u64 {
        self.0.get_digital_action_handle(name)
    }

    fn analog_action(&mut self, name: &str) -> u64 {
        self.0.get_analog_action_handle(name)
    }

    fn digital(&mut self, controller: u64, action: u64) -> Option<bool> {
        let data = self.0.get_digital_action_data(controller, action);
        data.bActive.then_some(data.bState)
    }

    fn analog(&mut self, controller: u64, action: u64) -> Option<(f32, f32)> {
        let data = self.0.get_analog_action_data(controller, action);
        data.bActive.then_some((data.x, data.y))
    }

    fn glyphs(&mut self, controller: u64, set: u64, digital: u64, analog: u64) -> Vec<String> {
        let origins = if digital != 0 {
            self.0.get_digital_action_origins(controller, set, digital)
        } else {
            self.0.get_analog_action_origins(controller, set, analog)
        };
        origins.into_iter().map(|origin| self.0.get_glyph_for_action_origin(origin)).collect()
    }

    fn binding_panel(&mut self, controller: u64) -> bool {
        self.0.show_binding_panel(controller)
    }
}

#[cfg(test)]
mod tests {
    use pen_protocol::{GameEvent, input_device, input_op, service_status};
    use pen_runtime::services::{Request, code_points};

    use super::*;

    /// Steam Input as a test sets it up: action sets, actions and what each
    /// controller holds, by handle.
    #[derive(Default)]
    struct Fake {
        controllers: Vec<u64>,
        sets: HashMap<String, u64>,
        digital: HashMap<String, u64>,
        analog: HashMap<String, u64>,
        held: HashMap<(u64, u64), bool>,
        sticks: HashMap<(u64, u64), (f32, f32)>,
        activated: Vec<(u64, u64)>,
        frames: usize,
    }

    impl Api for Fake {
        fn run_frame(&mut self) {
            self.frames += 1;
        }
        fn controllers(&mut self) -> Vec<u64> {
            self.controllers.clone()
        }
        fn action_set(&mut self, name: &str) -> u64 {
            self.sets.get(name).copied().unwrap_or(0)
        }
        fn activate(&mut self, controller: u64, set: u64) {
            self.activated.push((controller, set));
        }
        fn digital_action(&mut self, name: &str) -> u64 {
            self.digital.get(name).copied().unwrap_or(0)
        }
        fn analog_action(&mut self, name: &str) -> u64 {
            self.analog.get(name).copied().unwrap_or(0)
        }
        fn digital(&mut self, controller: u64, action: u64) -> Option<bool> {
            self.held.get(&(controller, action)).copied()
        }
        fn analog(&mut self, controller: u64, action: u64) -> Option<(f32, f32)> {
            self.sticks.get(&(controller, action)).copied()
        }
        fn glyphs(&mut self, controller: u64, set: u64, digital: u64, analog: u64) -> Vec<String> {
            vec![format!("/glyphs/{controller}-{set}-{digital}-{analog}.png")]
        }
        fn binding_panel(&mut self, controller: u64) -> bool {
            controller == 1
        }
    }

    /// Jump (a button) and move.x and move.y (axes), as the game declares them.
    fn declared() -> Actions {
        let mut actions = Actions::default();
        for (index, kind, name) in [(0, 0.0, "jump"), (1, 1.0, "move.x"), (2, 1.0, "move.y")] {
            let values = code_points(name);
            let length = values.len() as f32;
            let request = Request {
                id: 1,
                operation: input_op::DECLARE,
                numbers: [index as f32, kind, length, 0.0],
                values: &values,
            };
            assert_eq!(actions.call(&request).map(|answer| answer.status), Some(service_status::OK));
        }
        actions
    }

    fn fake() -> Fake {
        let mut fake = Fake { controllers: vec![1, 2], ..Fake::default() };
        fake.sets.insert("InGame".into(), 40);
        fake.digital.insert("jump".into(), 10);
        fake.analog.insert("move".into(), 20);
        fake
    }

    #[test]
    fn steam_input_moves_the_games_actions_by_name() {
        let mut actions = declared();
        let mut fake = fake();
        fake.held.insert((2, 10), true);
        fake.sticks.insert((1, 20), (0.25, -0.5));
        fake.sticks.insert((2, 20), (-0.75, 0.0));
        let mut input = SteamInput::new(fake);
        input.frame(&mut actions);
        let mut events = Vec::new();
        actions.map(&mut events);
        assert_eq!(
            events,
            [
                GameEvent::Action { action: 0, value: 1.0, device: input_device::STEAM },
                GameEvent::Action { action: 1, value: -0.75, device: input_device::STEAM },
                GameEvent::Action { action: 2, value: -0.5, device: input_device::STEAM },
            ]
        );
        // The same again moves nothing; each name was looked up once.
        input.frame(&mut actions);
        let mut again = Vec::new();
        actions.map(&mut again);
        assert!(again.is_empty());
        assert_eq!(input.api.frames, 2);
        assert_eq!(input.sources.len(), 3);
    }

    #[test]
    fn action_sets_glyphs_and_the_binding_panel() {
        let mut input = SteamInput::new(fake());
        assert_eq!(input.activate_set("Menu"), Answer::failed());
        assert_eq!(input.activate_set("InGame"), Answer::ok(2.0));
        assert_eq!(input.api.activated, [(1, 40), (2, 40)]);
        // Kept active on every frame's controllers.
        input.frame(&mut declared());
        assert_eq!(input.api.activated.len(), 4);
        assert_eq!(input.glyph("jump"), Answer::text("/glyphs/1-40-10-0.png"));
        assert_eq!(input.glyph("move"), Answer::text("/glyphs/1-40-0-20.png"));
        assert_eq!(input.glyph("fly"), Answer::failed());
        assert_eq!(input.panel(), Answer::ok(1.0));
        // No controller: unavailable, but the set waits for one.
        input.api.controllers.clear();
        assert_eq!(input.activate_set("InGame"), Answer::unavailable());
        assert_eq!(input.glyph("jump"), Answer::unavailable());
        assert_eq!(input.panel(), Answer::unavailable());
    }
}
