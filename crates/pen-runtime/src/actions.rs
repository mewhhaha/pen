//! The game's actions: what the player does (jump, move, open the map),
//! apart from the keys, buttons and sticks that do it. The game declares
//! each action and its default bindings through the input service
//! (`pen_protocol::input_op`), and gets Action events when one changes; the
//! player rebinds them (the service's capture and bind, from the game's own
//! settings), and the host keeps the player's bindings with their
//! preferences ([`Actions::save`], [`Actions::load`]). Both hosts map the
//! same way, here: once a live frame's events are in, [`Actions::map`] reads
//! the keys, mouse buttons and gamepads among them and puts each Action
//! event right after the event that moved it. The Action events are then
//! input like any other, recorded and replayed: a replay maps nothing.
//!
//! An action is a button (1 while any binding holds it at half or more,
//! else 0) or an axis (its bindings summed, from -1 to 1). A binding reads a
//! key (by the code its KeyDown events carry), a mouse button, a gamepad
//! button (0 to 1) or a gamepad axis (-1 to 1), on every pad, times its
//! scale. Steam Input's actions arrive by name ([`Actions::set_external`]).
use std::cell::RefCell;
use std::rc::Rc;

use pen_protocol::boundary::{MAX_ACTION_BINDINGS, MAX_ACTIONS, key_code, whole};
use pen_protocol::{GameEvent, input_device, input_op, input_source, service, service_status};
use serde_json::{Value, json};

use crate::gamepad::{AXES, BUTTONS};
use crate::services::{Answer, Request, Service, text};

/// The key that gives up a capture.
const ESCAPE: &str = "Escape";

/// The code a key's events carry, as a binding reads it.
fn key(name: &str) -> u32 {
    whole(key_code(name)).unwrap_or(0)
}

/// A number from a bindings file as the f32 a binding holds.
#[allow(clippy::cast_possible_truncation, reason = "narrowing f64 to f32 is the point")]
fn narrow(value: f64) -> f32 {
    value as f32
}

/// What a binding reads: its source (`pen_protocol::input_source`), the
/// key, button or axis, and what it multiplies its reading by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Binding {
    pub source: u32,
    pub code: u32,
    pub scale: f32,
}

impl Binding {
    /// A binding from three request values, if they make one.
    fn read(values: &[f32]) -> Option<Binding> {
        let [source, code, scale] = values else { return None };
        let source = whole(*source).filter(|source| (*source as usize) < input_source::NAMES.len())?;
        let code = whole(*code)?;
        let fits = match source {
            input_source::PAD_BUTTON => (code as usize) < BUTTONS,
            input_source::PAD_AXIS => (code as usize) < AXES,
            _ => true,
        };
        (fits && scale.is_finite()).then_some(Binding { source, code, scale: *scale })
    }

    fn lanes(&self) -> [f32; 3] {
        [self.source as f32, self.code as f32, self.scale]
    }
}

/// A button action is pressed or not; an axis action sums its bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Button,
    Axis,
}

/// An action as the game declared it, with the player's bindings when they
/// rebound it.
#[derive(Clone, Debug, PartialEq)]
pub struct Action {
    pub name: String,
    pub kind: Kind,
    pub defaults: Vec<Binding>,
    pub custom: Option<Vec<Binding>>,
}

impl Action {
    pub fn bindings(&self) -> &[Binding] {
        self.custom.as_deref().unwrap_or(&self.defaults)
    }
}

/// The keys, mouse buttons and pads held now, as the bindings read them.
#[derive(Debug, Default)]
struct Held {
    keys: Vec<u32>,
    buttons: Vec<u32>,
    /// Each pad's buttons and axes, by pad number.
    pads: Vec<([f32; BUTTONS], [f32; AXES])>,
}

impl Held {
    fn pad(&mut self, pad: u32) -> &mut ([f32; BUTTONS], [f32; AXES]) {
        let pad = pad as usize;
        if self.pads.len() <= pad {
            self.pads.resize(pad + 1, ([0.0; BUTTONS], [0.0; AXES]));
        }
        &mut self.pads[pad]
    }

    /// What a binding reads, before its scale: a pad's button at its most
    /// pressed, an axis at its furthest from rest, over every pad.
    fn read(&self, binding: &Binding) -> f32 {
        let code = binding.code as usize;
        match binding.source {
            input_source::KEY => f32::from(u8::from(self.keys.contains(&binding.code))),
            input_source::MOUSE => f32::from(u8::from(self.buttons.contains(&binding.code))),
            input_source::PAD_BUTTON => self.pads.iter().map(|(buttons, _)| buttons[code]).fold(0.0, f32::max),
            input_source::PAD_AXIS => self
                .pads
                .iter()
                .map(|(_, axes)| axes[code])
                .fold(0.0, |most, value| if value.abs() > most.abs() { value } else { most }),
            _ => 0.0,
        }
    }
}

/// A rebinding waiting for the player's next press: the request it
/// answers, the action and the binding's slot.
#[derive(Clone, Copy, Debug)]
struct Capture {
    request: u32,
    action: usize,
    slot: usize,
}

/// The game's actions and the player's bindings, the input held now, and
/// the value each action was last sent with.
#[derive(Debug, Default)]
pub struct Actions {
    actions: Vec<Option<Action>>,
    /// Actions the player rebound that the game has not declared (yet),
    /// by name, kept for when it does.
    stored: Vec<(String, Vec<Binding>)>,
    held: Held,
    /// Steam Input's value for each action.
    external: Vec<f32>,
    sent: Vec<f32>,
    capture: Option<Capture>,
    /// Answers that came later (captures), for the service.
    answered: Vec<(u32, Answer)>,
    /// Action events that come with the next frame's events (Steam's).
    pending: Vec<GameEvent>,
    /// The player's bindings changed since they were last saved.
    changed: bool,
}

/// The actions both a host's input and its input service use.
pub type Shared = Rc<RefCell<Actions>>;

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Button => "button",
        Kind::Axis => "axis",
    }
}

fn bindings_json(bindings: &[Binding]) -> Value {
    Value::Array(bindings.iter().map(|binding| json!(binding.lanes())).collect())
}

fn bindings_from(value: &Value) -> Option<Vec<Binding>> {
    let list = value.as_array()?;
    let bindings: Vec<Binding> = list
        .iter()
        .filter_map(|binding| {
            let lanes: Vec<f32> =
                binding.as_array()?.iter().map(|lane| lane.as_f64().map(narrow)).collect::<Option<_>>()?;
            Binding::read(&lanes)
        })
        .collect();
    (bindings.len() == list.len() && bindings.len() <= MAX_ACTION_BINDINGS).then_some(bindings)
}

impl Actions {
    /// Actions and bindings as [`Actions::save`] wrote them; what does not
    /// read is left out.
    pub fn load(text: &str) -> Actions {
        let mut actions = Actions::default();
        let Ok(file) = serde_json::from_str::<Value>(text) else { return actions };
        for entry in file.get("actions").and_then(Value::as_array).into_iter().flatten() {
            let index = entry.get("index").and_then(Value::as_u64).and_then(|index| usize::try_from(index).ok());
            let name = entry.get("name").and_then(Value::as_str).filter(|name| !name.is_empty());
            let kind = match entry.get("kind").and_then(Value::as_str) {
                Some("button") => Some(Kind::Button),
                Some("axis") => Some(Kind::Axis),
                _ => None,
            };
            let defaults = entry.get("defaults").and_then(bindings_from);
            let custom = entry.get("bindings").map(bindings_from);
            let (Some(index), Some(name), Some(kind), Some(defaults)) = (index, name, kind, defaults) else { continue };
            if index >= MAX_ACTIONS || matches!(custom, Some(None)) {
                continue;
            }
            let action = Action { name: name.to_owned(), kind, defaults, custom: custom.flatten() };
            actions.put(index, action);
        }
        actions
    }

    /// The actions as declared and the player's bindings, as JSON.
    pub fn save(&self) -> String {
        let actions: Vec<Value> = self
            .actions
            .iter()
            .enumerate()
            .filter_map(|(index, action)| {
                let action = action.as_ref()?;
                let mut entry = json!({
                    "index": index,
                    "name": action.name,
                    "kind": kind_name(action.kind),
                    "defaults": bindings_json(&action.defaults),
                });
                if let Some(custom) = &action.custom {
                    entry["bindings"] = bindings_json(custom);
                }
                Some(entry)
            })
            .collect();
        let mut file = serde_json::to_string_pretty(&json!({ "actions": actions })).expect("JSON");
        file.push('\n');
        file
    }

    /// Whether the player's bindings changed since the last call: then the
    /// host saves them.
    pub fn take_changed(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }

    fn put(&mut self, index: usize, action: Action) {
        if self.actions.len() <= index {
            self.actions.resize(index + 1, None);
            self.sent.resize(index + 1, 0.0);
            self.external.resize(index + 1, 0.0);
        }
        self.actions[index] = Some(action);
    }

    pub fn action(&self, index: u32) -> Option<&Action> {
        self.actions.get(index as usize).and_then(Option::as_ref)
    }

    /// The declared actions, by number.
    pub fn declared(&self) -> impl Iterator<Item = (u32, &Action)> {
        self.actions
            .iter()
            .enumerate()
            .filter_map(|(index, action)| Some((u32::try_from(index).expect("MAX_ACTIONS"), action.as_ref()?)))
    }

    /// Declares action `index` (the game's, again on every start): its
    /// name, kind and default bindings. Bindings the player chose for an
    /// action of that name stay, whatever its number was.
    pub fn declare(&mut self, index: usize, name: &str, kind: Kind, defaults: Vec<Binding>) {
        let custom = match self.actions.iter().flatten().find(|action| action.name == name) {
            Some(action) => action.custom.clone(),
            None => self.stored.iter().position(|(stored, _)| stored == name).map(|at| self.stored.remove(at).1),
        };
        // Another action this index held keeps its player's bindings for later.
        if let Some(Some(old)) = self.actions.get_mut(index)
            && old.name != name
            && let Some(custom) = old.custom.take()
        {
            self.stored.push((old.name.clone(), custom));
        }
        self.put(index, Action { name: name.to_owned(), kind, defaults, custom });
        self.changed = true;
    }

    /// Steam Input says action `index` is at `value` now: its Action event
    /// comes with the next frame's events if that changed it.
    pub fn set_external(&mut self, index: u32, value: f32) {
        let index = index as usize;
        if self.actions.get(index).is_none_or(Option::is_none) || !value.is_finite() {
            return;
        }
        if self.external[index] != value {
            self.external[index] = value;
            if let Some(event) = self.update(index, input_device::STEAM) {
                self.pending.push(event);
            }
        }
    }

    /// An action's value from what is held now and Steam's.
    fn value(&self, index: usize) -> f32 {
        let Some(action) = &self.actions[index] else { return 0.0 };
        let external = self.external[index];
        let readings = action.bindings().iter().map(|binding| self.held.read(binding) * binding.scale);
        match action.kind {
            Kind::Button => {
                let pressed = external >= 0.5 || readings.into_iter().any(|reading| reading >= 0.5);
                f32::from(u8::from(pressed))
            }
            Kind::Axis => (readings.sum::<f32>() + external).clamp(-1.0, 1.0),
        }
    }

    /// The Action event for action `index` if its value changed, moved by
    /// `device`.
    fn update(&mut self, index: usize, device: u32) -> Option<GameEvent> {
        let value = self.value(index);
        if value == self.sent[index] {
            return None;
        }
        self.sent[index] = value;
        Some(GameEvent::Action { action: u32::try_from(index).expect("MAX_ACTIONS"), value, device })
    }

    /// Reads an event into what is held: the device it came from, or None
    /// for an event no binding reads.
    fn hold(&mut self, event: &GameEvent) -> Option<u32> {
        let held = &mut self.held;
        let set = |list: &mut Vec<u32>, code: u32, down: bool| {
            list.retain(|held| *held != code);
            if down {
                list.push(code);
            }
        };
        match event {
            GameEvent::KeyDown(key) | GameEvent::KeyUp(key) => {
                set(&mut held.keys, self::key(key), matches!(event, GameEvent::KeyDown(_)));
                Some(input_device::KEYBOARD)
            }
            GameEvent::PointerDown { button, .. } | GameEvent::PointerUp { button, .. } => {
                set(&mut held.buttons, *button, matches!(event, GameEvent::PointerDown { .. }));
                Some(input_device::KEYBOARD)
            }
            GameEvent::GamepadButton { button, pad, value, .. } if (*button as usize) < BUTTONS => {
                held.pad(*pad).0[*button as usize] = *value;
                Some(input_device::GAMEPAD)
            }
            GameEvent::GamepadAxis { axis, pad, value } if (*axis as usize) < AXES => {
                held.pad(*pad).1[*axis as usize] = *value;
                Some(input_device::GAMEPAD)
            }
            GameEvent::FocusLost => {
                held.keys.clear();
                held.buttons.clear();
                Some(input_device::KEYBOARD)
            }
            _ => None,
        }
    }

    /// The binding an event makes for a capture: a key, mouse button or pad
    /// button going down, or a stick pushed past half; Escape gives up.
    fn captured(event: &GameEvent) -> Option<Option<Binding>> {
        let binding = |source, code, scale| Some(Some(Binding { source, code, scale }));
        match event {
            GameEvent::KeyDown(key) if key == ESCAPE => Some(None),
            GameEvent::KeyDown(name) => binding(input_source::KEY, key(name), 1.0),
            GameEvent::PointerDown { button, .. } => binding(input_source::MOUSE, *button, 1.0),
            GameEvent::GamepadButton { button, pressed: 1, .. } => binding(input_source::PAD_BUTTON, *button, 1.0),
            GameEvent::GamepadAxis { axis, value, .. } if value.abs() >= 0.5 => {
                binding(input_source::PAD_AXIS, *axis, value.signum())
            }
            _ => None,
        }
    }

    /// A live frame's events: each Action event they move, right after the
    /// event that moved it, and Steam's first. A capture waiting takes the
    /// next press, which moves no action.
    pub fn map(&mut self, events: &mut Vec<GameEvent>) {
        if self.actions.is_empty() && self.capture.is_none() {
            // Nothing declared: only what is held is kept up.
            for event in events.iter() {
                self.hold(event);
            }
            return;
        }
        let raw = std::mem::take(events);
        events.append(&mut self.pending);
        for event in raw {
            // The press a capture takes is not held, so its release moves nothing either.
            if let Some(capture) = self.capture
                && let Some(binding) = Actions::captured(&event)
            {
                self.capture = None;
                let answer = match binding {
                    Some(binding) => self.bind(capture.action, capture.slot, binding),
                    None => Answer::failed(),
                };
                self.answered.push((capture.request, answer));
                events.push(event);
                continue;
            }
            let device = self.hold(&event);
            events.push(event);
            if let Some(device) = device {
                for index in 0..self.actions.len() {
                    if let Some(action) = self.update(index, device) {
                        events.push(action);
                    }
                }
            }
        }
    }

    /// Sets binding `slot` of action `index` (one past the last adds one).
    fn bind(&mut self, index: usize, slot: usize, binding: Binding) -> Answer {
        let Some(Some(action)) = self.actions.get_mut(index) else { return Answer::failed() };
        let mut bindings = action.bindings().to_vec();
        match slot.cmp(&bindings.len()) {
            std::cmp::Ordering::Less => bindings[slot] = binding,
            std::cmp::Ordering::Equal if slot < MAX_ACTION_BINDINGS => bindings.push(binding),
            _ => return Answer::failed(),
        }
        action.custom = Some(bindings);
        self.changed = true;
        Answer { values: binding.lanes().to_vec(), ..Answer::ok(slot as f32) }
    }

    /// Answers an input service request (`pen_protocol::input_op`), or
    /// None for a capture, answered when the player presses something.
    pub fn call(&mut self, request: &Request) -> Option<Answer> {
        let answer = self.answer(request);
        (request.operation != input_op::CAPTURE || answer.status != service_status::OK).then_some(answer)
    }

    fn answer(&mut self, request: &Request) -> Answer {
        let [a, b, c, _] = request.numbers;
        let index = whole(a).map(|index| index as usize).filter(|index| *index <= MAX_ACTIONS);
        let slot = whole(b).map(|slot| slot as usize);
        if request.operation == input_op::RESET && index == Some(MAX_ACTIONS) {
            for action in self.actions.iter_mut().flatten() {
                action.custom = None;
            }
            self.stored.clear();
            self.changed = true;
            return Answer::ok(0.0);
        }
        let Some(index) = index.filter(|index| *index < MAX_ACTIONS) else { return Answer::failed() };
        if request.operation == input_op::DECLARE {
            let kind = match whole(b) {
                Some(0) => Kind::Button,
                Some(1) => Kind::Axis,
                _ => return Answer::failed(),
            };
            let Some(length) = whole(c).map(|length| length as usize).filter(|length| *length <= request.values.len())
            else {
                return Answer::failed();
            };
            let (name, rest) = request.values.split_at(length);
            let defaults: Option<Vec<Binding>> = rest.chunks(3).map(Binding::read).collect();
            let name = text(name);
            match defaults {
                Some(defaults) if !name.is_empty() && defaults.len() <= MAX_ACTION_BINDINGS => {
                    self.declare(index, &name, kind, defaults);
                    let count = self.actions[index].as_ref().expect("declared").bindings().len();
                    return Answer::ok(count as f32);
                }
                _ => return Answer::failed(),
            }
        }
        let Some(action) = self.actions.get(index).and_then(Option::as_ref) else { return Answer::failed() };
        match request.operation {
            input_op::CAPTURE => {
                let Some(slot) = slot.filter(|slot| *slot <= action.bindings().len() && *slot < MAX_ACTION_BINDINGS)
                else {
                    return Answer::failed();
                };
                if let Some(earlier) = self.capture.replace(Capture { request: request.id, action: index, slot }) {
                    self.answered.push((earlier.request, Answer::failed()));
                }
                Answer::ok(0.0)
            }
            input_op::CANCEL => {
                if let Some(capture) = self.capture.take() {
                    self.answered.push((capture.request, Answer::failed()));
                }
                Answer::ok(0.0)
            }
            input_op::BIND => match (slot, Binding::read(request.values)) {
                (Some(slot), Some(binding)) => self.bind(index, slot, binding),
                _ => Answer::failed(),
            },
            input_op::UNBIND => {
                let mut bindings = action.bindings().to_vec();
                let Some(slot) = slot.filter(|slot| *slot < bindings.len()) else { return Answer::failed() };
                bindings.remove(slot);
                self.actions[index].as_mut().expect("declared").custom = Some(bindings);
                self.changed = true;
                Answer::ok(0.0)
            }
            input_op::RESET => {
                self.actions[index].as_mut().expect("declared").custom = None;
                self.changed = true;
                Answer::ok(0.0)
            }
            input_op::BINDINGS => {
                let bindings = action.bindings();
                Answer {
                    values: bindings.iter().flat_map(Binding::lanes).collect(),
                    ..Answer::ok(bindings.len() as f32)
                }
            }
            _ => Answer::unknown(),
        }
    }

    /// The answers that came since the last call (captures).
    pub fn take_answered(&mut self) -> Vec<(u32, Answer)> {
        std::mem::take(&mut self.answered)
    }
}

/// The input service: the game declares its actions and the player rebinds
/// them, through the host's [`Actions`].
pub struct InputService(pub Shared);

impl<Input> Service<Input> for InputService {
    fn code(&self) -> u32 {
        service::INPUT
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        self.0.borrow_mut().call(request)
    }

    fn poll(&mut self) -> Vec<(u32, Answer)> {
        self.0.borrow_mut().take_answered()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::{Services, code_points};
    use pen_protocol::{ServiceRequest, gamepad_axis, gamepad_button, service_status};

    fn request(operation: u32, numbers: [f32; 4], values: &[f32]) -> Request<'_> {
        Request { id: 7, operation, numbers, values }
    }

    /// Declares `name` as action `index` with `bindings`, as the game asks.
    fn declare(actions: &mut Actions, index: u32, kind: f32, name: &str, bindings: &[Binding]) -> Option<Answer> {
        let mut values = code_points(name);
        values.extend(bindings.iter().flat_map(Binding::lanes));
        let length = name.chars().count() as f32;
        actions.call(&request(input_op::DECLARE, [index as f32, kind, length, 0.0], &values))
    }

    fn key_binding(name: &str, scale: f32) -> Binding {
        Binding { source: input_source::KEY, code: key(name), scale }
    }

    fn actions_of(events: &[GameEvent]) -> Vec<(u32, f32, u32)> {
        events
            .iter()
            .filter_map(|event| match event {
                GameEvent::Action { action, value, device } => Some((*action, *value, *device)),
                _ => None,
            })
            .collect()
    }

    /// Jump on space or the south button; move on A and D, or the left stick.
    fn game() -> Actions {
        let mut actions = Actions::default();
        let jump = [
            key_binding(" ", 1.0),
            Binding { source: input_source::PAD_BUTTON, code: gamepad_button::SOUTH, scale: 1.0 },
        ];
        assert_eq!(declare(&mut actions, 0, 0.0, "jump", &jump), Some(Answer::ok(2.0)));
        let walk = [
            key_binding("a", -1.0),
            key_binding("d", 1.0),
            Binding { source: input_source::PAD_AXIS, code: gamepad_axis::LEFT_X, scale: 1.0 },
        ];
        assert_eq!(declare(&mut actions, 3, 1.0, "move.x", &walk), Some(Answer::ok(3.0)));
        actions
    }

    #[test]
    fn keys_and_pads_move_actions_right_after_their_events() {
        let mut actions = game();
        let mut events = vec![
            GameEvent::KeyDown(" ".into()),
            GameEvent::PointerMove { x: 1.0, y: 2.0 },
            GameEvent::KeyDown("D".into()),
            GameEvent::KeyDown("a".into()),
            GameEvent::KeyUp(" ".into()),
        ];
        actions.map(&mut events);
        assert_eq!(
            events,
            [
                GameEvent::KeyDown(" ".into()),
                GameEvent::Action { action: 0, value: 1.0, device: input_device::KEYBOARD },
                GameEvent::PointerMove { x: 1.0, y: 2.0 },
                GameEvent::KeyDown("D".into()),
                GameEvent::Action { action: 3, value: 1.0, device: input_device::KEYBOARD },
                GameEvent::KeyDown("a".into()),
                GameEvent::Action { action: 3, value: 0.0, device: input_device::KEYBOARD },
                GameEvent::KeyUp(" ".into()),
                GameEvent::Action { action: 0, value: 0.0, device: input_device::KEYBOARD },
            ]
        );
        // A stick adds to the keys, up to 1; the pad's button presses at half.
        let mut events = vec![
            GameEvent::KeyUp("a".into()),
            GameEvent::GamepadAxis { axis: gamepad_axis::LEFT_X, pad: 1, value: -0.5 },
            GameEvent::GamepadButton { button: gamepad_button::SOUTH, pad: 0, value: 0.4, pressed: 0 },
            GameEvent::GamepadButton { button: gamepad_button::SOUTH, pad: 0, value: 0.6, pressed: 1 },
        ];
        actions.map(&mut events);
        assert_eq!(
            actions_of(&events),
            [(3, 1.0, input_device::KEYBOARD), (3, 0.5, input_device::GAMEPAD), (0, 1.0, input_device::GAMEPAD),]
        );
        // Losing the focus lets go of the keys, not the pads.
        let mut events = vec![GameEvent::FocusLost];
        actions.map(&mut events);
        assert_eq!(actions_of(&events), [(3, -0.5, input_device::KEYBOARD)]);
    }

    #[test]
    fn a_capture_binds_the_next_press_and_escape_gives_up() {
        let mut actions = game();
        assert_eq!(actions.call(&request(input_op::CAPTURE, [0.0, 0.0, 0.0, 0.0], &[])), None);
        let mut events = vec![GameEvent::PointerMove { x: 0.0, y: 0.0 }, GameEvent::KeyDown("j".into())];
        actions.map(&mut events);
        // The press binds, and moves nothing.
        assert!(actions_of(&events).is_empty());
        let answered = actions.take_answered();
        let binding = key_binding("j", 1.0);
        assert_eq!(answered, [(7, Answer { values: binding.lanes().to_vec(), ..Answer::ok(0.0) })]);
        assert_eq!(actions.action(0).unwrap().bindings()[0], binding);
        assert!(actions.take_changed());
        // Letting go of it, and pressing it again, moves the action.
        let mut events = vec![GameEvent::KeyUp("j".into()), GameEvent::KeyDown("j".into())];
        actions.map(&mut events);
        assert_eq!(actions_of(&events), [(0, 1.0, input_device::KEYBOARD)]);
        // A stick past half binds its direction, as a new slot.
        assert_eq!(actions.call(&request(input_op::CAPTURE, [0.0, 2.0, 0.0, 0.0], &[])), None);
        let mut events = vec![
            GameEvent::GamepadAxis { axis: gamepad_axis::RIGHT_Y, pad: 0, value: -0.3 },
            GameEvent::GamepadAxis { axis: gamepad_axis::RIGHT_Y, pad: 0, value: -0.9 },
        ];
        actions.map(&mut events);
        let stick = Binding { source: input_source::PAD_AXIS, code: gamepad_axis::RIGHT_Y, scale: -1.0 };
        assert_eq!(actions.take_answered()[0].1.values, stick.lanes());
        assert_eq!(actions.action(0).unwrap().bindings().len(), 3);
        // Escape gives up; a second capture gives up the first.
        assert_eq!(actions.call(&request(input_op::CAPTURE, [0.0, 0.0, 0.0, 0.0], &[])), None);
        assert_eq!(actions.call(&request(input_op::CAPTURE, [3.0, 0.0, 0.0, 0.0], &[])), None);
        let mut events = vec![GameEvent::KeyDown("Escape".into())];
        actions.map(&mut events);
        let statuses: Vec<u32> = actions.take_answered().iter().map(|(_, answer)| answer.status).collect();
        assert_eq!(statuses, [service_status::FAILED, service_status::FAILED]);
        // A capture of a slot past the next one fails at once.
        let answer = actions.call(&request(input_op::CAPTURE, [3.0, 5.0, 0.0, 0.0], &[]));
        assert_eq!(answer, Some(Answer::failed()));
    }

    #[test]
    fn bind_unbind_reset_and_bindings() {
        let mut actions = game();
        let mouse = Binding { source: input_source::MOUSE, code: 2, scale: 1.0 };
        let answer = actions.call(&request(input_op::BIND, [0.0, 1.0, 0.0, 0.0], &mouse.lanes())).unwrap();
        assert_eq!(answer.status, service_status::OK);
        let answer = actions.call(&request(input_op::BINDINGS, [0.0, 0.0, 0.0, 0.0], &[])).unwrap();
        assert_eq!(answer.value, 2.0);
        assert_eq!(answer.values, [key_binding(" ", 1.0).lanes(), mouse.lanes()].concat());
        let mut events = vec![GameEvent::PointerDown { button: 2, x: 0.0, y: 0.0 }];
        actions.map(&mut events);
        assert_eq!(actions_of(&events), [(0, 1.0, input_device::KEYBOARD)]);
        actions.call(&request(input_op::UNBIND, [0.0, 0.0, 0.0, 0.0], &[])).unwrap();
        assert_eq!(actions.action(0).unwrap().bindings(), [mouse]);
        actions.call(&request(input_op::RESET, [0.0, 0.0, 0.0, 0.0], &[])).unwrap();
        assert_eq!(actions.action(0).unwrap().bindings().len(), 2);
        // Every action at once.
        actions.call(&request(input_op::BIND, [3.0, 0.0, 0.0, 0.0], &mouse.lanes())).unwrap();
        actions.call(&request(input_op::RESET, [MAX_ACTIONS as f32, 0.0, 0.0, 0.0], &[])).unwrap();
        assert!(actions.declared().all(|(_, action)| action.custom.is_none()));
        // What makes no sense fails.
        let failed = Some(Answer::failed());
        assert_eq!(actions.call(&request(input_op::BINDINGS, [9.0, 0.0, 0.0, 0.0], &[])), failed);
        assert_eq!(actions.call(&request(input_op::BIND, [0.0, 0.0, 0.0, 0.0], &[9.0, 0.0, 1.0])), failed);
        assert_eq!(actions.call(&request(input_op::UNBIND, [0.0, 4.0, 0.0, 0.0], &[])), failed);
        assert_eq!(declare(&mut actions, 1, 2.0, "x", &[]), failed);
        assert_eq!(declare(&mut actions, 1, 0.0, "", &[]), failed);
        assert_eq!(declare(&mut actions, u32::try_from(MAX_ACTIONS).unwrap(), 0.0, "x", &[]), failed);
        assert_eq!(actions.call(&request(99, [0.0, 0.0, 0.0, 0.0], &[])), Some(Answer::unknown()));
    }

    #[test]
    fn the_players_bindings_keep_and_survive_a_new_declaration() {
        let mut actions = game();
        let mouse = Binding { source: input_source::MOUSE, code: 0, scale: 1.0 };
        actions.call(&request(input_op::BIND, [0.0, 0.0, 0.0, 0.0], &mouse.lanes())).unwrap();
        let saved = actions.save();
        let mut loaded = Actions::load(&saved);
        assert_eq!(loaded.save(), saved);
        assert_eq!(loaded.action(0).unwrap().bindings()[0], mouse);
        // The game declares again on its next start, at another number:
        // the player's binding goes with the name.
        assert_eq!(declare(&mut loaded, 5, 0.0, "jump", &[key_binding(" ", 1.0)]), Some(Answer::ok(2.0)));
        assert_eq!(loaded.action(5).unwrap().bindings()[0], mouse);
        // A file that does not read gives no actions.
        assert_eq!(Actions::load("{").declared().count(), 0);
        assert_eq!(
            Actions::load(r#"{"actions": [{"index": 0, "name": "x", "kind": "dial", "defaults": []}]}"#)
                .declared()
                .count(),
            0
        );
    }

    #[test]
    fn steam_values_come_first_in_the_next_frame() {
        let mut actions = game();
        actions.set_external(0, 1.0);
        actions.set_external(0, 1.0);
        actions.set_external(42, 1.0);
        let mut events = vec![GameEvent::KeyDown(" ".into())];
        actions.map(&mut events);
        assert_eq!(
            events,
            [GameEvent::Action { action: 0, value: 1.0, device: input_device::STEAM }, GameEvent::KeyDown(" ".into())]
        );
    }

    #[test]
    fn the_service_answers_captures_in_a_later_frame() {
        let actions: Shared = Rc::new(RefCell::new(game()));
        let mut services: Services = Services::new();
        services.register(Box::new(InputService(actions.clone())));
        let capture = ServiceRequest {
            request: 12,
            service: service::INPUT,
            operation: input_op::CAPTURE,
            a: 3.0,
            b: 0.0,
            ..ServiceRequest::default()
        };
        services.answer(&[capture]);
        assert!(services.take_answers().is_empty());
        actions.borrow_mut().map(&mut vec![GameEvent::KeyDown("ArrowLeft".into())]);
        services.frame(&mut ());
        let answers = services.take_answers();
        assert_eq!(
            answers[0],
            GameEvent::ServiceAnswer { request: 12, status: service_status::OK, value: 0.0, length: 3 }
        );
        assert_eq!(
            answers[1],
            GameEvent::ServiceData { request: 12, a: input_source::KEY as f32, b: key("ArrowLeft") as f32, c: 1.0 }
        );
    }
}
