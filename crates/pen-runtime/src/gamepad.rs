//! Gamepads as the game's events, worked out the same way on every host.
//! A host reads each pad it has (gilrs natively, the Gamepad API in a
//! browser, a script's steps headless) into a [`Pad`] in the standard
//! gamepad's layout, as often as it likes; [`Gamepads`] numbers the pads
//! from 0, applies the sticks' dead zone, and turns what changed since the
//! pad was last read into GamepadConnected, GamepadButton, GamepadAxis and
//! GamepadDisconnected events. Like every input, those events are recorded
//! and replayed: a replay reads no pad.
use pen_protocol::GameEvent;
use pen_protocol::boundary::MAX_GAMEPADS;
use pen_protocol::{gamepad_axis, gamepad_button};

/// The standard gamepad's buttons (`pen_protocol::gamepad_button`).
pub const BUTTONS: usize = gamepad_button::NAMES.len();
/// The standard gamepad's stick axes (`pen_protocol::gamepad_axis`).
pub const AXES: usize = gamepad_axis::NAMES.len();

/// How far a stick moves before it reads anything, as a fraction of its
/// travel. Past it, the stick reads from 0 again, so it moves smoothly out
/// of the dead zone; measured round the stick (radial), so diagonals keep
/// their direction.
pub const DEAD_ZONE: f32 = 0.15;

/// The least change in an analog value worth an event: smaller ones are
/// noise from a stick or trigger at rest. A value that reaches its end (0,
/// or 1 either way) is always sent.
pub const STEP: f32 = 1.0 / 128.0;

/// A button counts as pressed from half its travel.
pub const PRESSED: f32 = 0.5;

/// What a pad reads: each button from 0 to 1 and each stick axis from -1 to
/// 1 (right and down positive), in the standard gamepad's order, and whether
/// the pad's own layout was known to be the standard one.
#[derive(Clone, Debug, PartialEq)]
pub struct Pad {
    pub buttons: [f32; BUTTONS],
    pub axes: [f32; AXES],
    pub standard: bool,
}

impl Default for Pad {
    fn default() -> Pad {
        Pad { buttons: [0.0; BUTTONS], axes: [0.0; AXES], standard: true }
    }
}

/// A stick's two axes past the radial dead zone `zone`: zero inside it, and
/// scaled so that its edge reads 0 and full travel 1.
pub fn dead_zone(x: f32, y: f32, zone: f32) -> (f32, f32) {
    let length = x.hypot(y);
    // A NaN length reads as rest too.
    if length.partial_cmp(&zone) != Some(std::cmp::Ordering::Greater) {
        return (0.0, 0.0);
    }
    let scaled = ((length - zone) / (1.0 - zone)).min(1.0);
    (x / length * scaled, y / length * scaled)
}

/// A reading as an event lane holds it: finite, within `min..=1`.
fn clean(value: f32, min: f32) -> f32 {
    if value.is_finite() { value.clamp(min, 1.0) } else { 0.0 }
}

/// Whether `now` is worth an event after `sent`.
fn changed(sent: f32, now: f32) -> bool {
    now != sent && ((now - sent).abs() >= STEP || now == 0.0 || now.abs() == 1.0)
}

/// A pad the game knows of: the host's key for it and what it was last told.
#[derive(Debug)]
struct Slot {
    key: u64,
    sent: Pad,
}

/// The pads the game knows of, by number.
#[derive(Debug, Default)]
pub struct Gamepads {
    slots: Vec<Option<Slot>>,
}

impl Gamepads {
    fn slot(&self, key: u64) -> Option<usize> {
        self.slots.iter().position(|slot| slot.as_ref().is_some_and(|slot| slot.key == key))
    }

    /// The pad the host knows by `key` reads `pad` now: the events for what
    /// changed, after GamepadConnected when the game did not know it. A pad
    /// past the `MAX_GAMEPADS`th is left out.
    pub fn update(&mut self, key: u64, pad: &Pad, events: &mut Vec<GameEvent>) {
        let number = match self.slot(key) {
            Some(number) => number,
            None => {
                let free = self.slots.iter().position(Option::is_none).unwrap_or(self.slots.len());
                if free >= MAX_GAMEPADS {
                    return;
                }
                if free == self.slots.len() {
                    self.slots.push(None);
                }
                self.slots[free] = Some(Slot { key, sent: Pad { standard: pad.standard, ..Pad::default() } });
                events.push(GameEvent::GamepadConnected {
                    pad: u32::try_from(free).expect("fewer than MAX_GAMEPADS pads"),
                    standard: u32::from(pad.standard),
                });
                free
            }
        };
        let slot = self.slots[number].as_mut().expect("the pad's slot");
        let pad_number = u32::try_from(number).expect("fewer than MAX_GAMEPADS pads");
        for (button, (sent, now)) in slot.sent.buttons.iter_mut().zip(pad.buttons).enumerate() {
            let now = clean(now, 0.0);
            let pressed = now >= PRESSED;
            if changed(*sent, now) || pressed != (*sent >= PRESSED) {
                *sent = now;
                events.push(GameEvent::GamepadButton {
                    button: u32::try_from(button).expect("a standard button"),
                    pad: pad_number,
                    value: now,
                    pressed: u32::from(pressed),
                });
            }
        }
        let sticks = [(gamepad_axis::LEFT_X, gamepad_axis::LEFT_Y), (gamepad_axis::RIGHT_X, gamepad_axis::RIGHT_Y)];
        for (x, y) in sticks {
            let (x, y) = (x as usize, y as usize);
            let (dx, dy) = dead_zone(clean(pad.axes[x], -1.0), clean(pad.axes[y], -1.0), DEAD_ZONE);
            for (axis, now) in [(x, dx), (y, dy)] {
                let sent = &mut slot.sent.axes[axis];
                if changed(*sent, now) {
                    *sent = now;
                    events.push(GameEvent::GamepadAxis {
                        axis: u32::try_from(axis).expect("a standard axis"),
                        pad: pad_number,
                        value: now,
                    });
                }
            }
        }
    }

    /// The pad the host knows by `key` went away: its buttons and axes come
    /// back to rest, then GamepadDisconnected, and its number is free again.
    pub fn disconnect(&mut self, key: u64, events: &mut Vec<GameEvent>) {
        let Some(number) = self.slot(key) else { return };
        let standard = self.slots[number].as_ref().is_some_and(|slot| slot.sent.standard);
        self.update(key, &Pad { standard, ..Pad::default() }, events);
        self.slots[number] = None;
        events
            .push(GameEvent::GamepadDisconnected { pad: u32::try_from(number).expect("fewer than MAX_GAMEPADS pads") });
    }

    /// The host's keys of the pads the game knows of, so a host that polls
    /// can tell which went away.
    pub fn keys(&self) -> Vec<u64> {
        self.slots.iter().flatten().map(|slot| slot.key).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_protocol::gamepad_button;

    fn pad() -> Pad {
        Pad::default()
    }

    #[test]
    fn a_stick_reads_nothing_inside_its_dead_zone_and_rises_smoothly_past_it() {
        assert_eq!(dead_zone(0.1, 0.1, DEAD_ZONE), (0.0, 0.0));
        assert_eq!(dead_zone(f32::NAN, 0.0, DEAD_ZONE), (0.0, 0.0));
        let (x, y) = dead_zone(1.0, 0.0, DEAD_ZONE);
        assert_eq!((x, y), (1.0, 0.0));
        // Just past the edge reads just above zero, not 0.15.
        let (x, _) = dead_zone(0.16, 0.0, DEAD_ZONE);
        assert!(x > 0.0 && x < 0.02, "{x}");
        // A diagonal keeps its direction.
        let (x, y) = dead_zone(0.6, -0.6, DEAD_ZONE);
        assert!((x + y).abs() < 1e-6 && x > 0.0);
        // A corner past full travel reads at most 1 round the stick.
        let (x, y) = dead_zone(1.0, 1.0, DEAD_ZONE);
        assert!((x.hypot(y) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_new_pad_connects_and_says_what_it_holds() {
        let mut gamepads = Gamepads::default();
        let mut events = Vec::new();
        let mut held = pad();
        held.buttons[gamepad_button::SOUTH as usize] = 1.0;
        gamepads.update(77, &held, &mut events);
        assert_eq!(
            events,
            [
                GameEvent::GamepadConnected { pad: 0, standard: 1 },
                GameEvent::GamepadButton { button: gamepad_button::SOUTH, pad: 0, value: 1.0, pressed: 1 },
            ]
        );
        // Reading the same again says nothing.
        events.clear();
        gamepads.update(77, &held, &mut events);
        assert!(events.is_empty());
    }

    #[test]
    fn buttons_and_triggers_report_their_value_and_press_at_half() {
        let mut gamepads = Gamepads::default();
        let mut events = Vec::new();
        gamepads.update(1, &pad(), &mut events);
        events.clear();
        let trigger = gamepad_button::RIGHT_TRIGGER;
        let mut reading = pad();
        for (value, pressed) in [(0.25, 0), (0.6, 1), (0.4, 0)] {
            reading.buttons[trigger as usize] = value;
            gamepads.update(1, &reading, &mut events);
            assert_eq!(events.pop(), Some(GameEvent::GamepadButton { button: trigger, pad: 0, value, pressed }));
        }
        // Noise smaller than a step says nothing.
        reading.buttons[trigger as usize] = 0.401;
        gamepads.update(1, &reading, &mut events);
        assert!(events.is_empty());
        // Out of range or not a number reads as its end, or rest.
        reading.buttons[trigger as usize] = 3.0;
        gamepads.update(1, &reading, &mut events);
        assert_eq!(events.pop(), Some(GameEvent::GamepadButton { button: trigger, pad: 0, value: 1.0, pressed: 1 }));
        reading.buttons[trigger as usize] = f32::NAN;
        gamepads.update(1, &reading, &mut events);
        assert_eq!(events.pop(), Some(GameEvent::GamepadButton { button: trigger, pad: 0, value: 0.0, pressed: 0 }));
    }

    #[test]
    fn sticks_report_past_their_dead_zone_and_return_to_zero() {
        let mut gamepads = Gamepads::default();
        let mut events = Vec::new();
        gamepads.update(1, &pad(), &mut events);
        events.clear();
        let mut reading = pad();
        // Resting a little off centre says nothing.
        reading.axes[gamepad_axis::LEFT_X as usize] = 0.1;
        gamepads.update(1, &reading, &mut events);
        assert!(events.is_empty());
        reading.axes[gamepad_axis::LEFT_X as usize] = 1.0;
        gamepads.update(1, &reading, &mut events);
        assert_eq!(events, [GameEvent::GamepadAxis { axis: gamepad_axis::LEFT_X, pad: 0, value: 1.0 }]);
        events.clear();
        reading.axes[gamepad_axis::LEFT_X as usize] = 0.05;
        gamepads.update(1, &reading, &mut events);
        assert_eq!(events, [GameEvent::GamepadAxis { axis: gamepad_axis::LEFT_X, pad: 0, value: 0.0 }]);
    }

    #[test]
    fn pads_take_the_lowest_free_number_and_let_go_when_they_leave() {
        let mut gamepads = Gamepads::default();
        let mut events = Vec::new();
        gamepads.update(10, &pad(), &mut events);
        gamepads.update(20, &pad(), &mut events);
        let mut held = pad();
        held.buttons[gamepad_button::START as usize] = 1.0;
        held.axes[gamepad_axis::RIGHT_Y as usize] = -1.0;
        gamepads.update(10, &held, &mut events);
        events.clear();
        gamepads.disconnect(10, &mut events);
        assert_eq!(
            events,
            [
                GameEvent::GamepadButton { button: gamepad_button::START, pad: 0, value: 0.0, pressed: 0 },
                GameEvent::GamepadAxis { axis: gamepad_axis::RIGHT_Y, pad: 0, value: 0.0 },
                GameEvent::GamepadDisconnected { pad: 0 },
            ]
        );
        assert_eq!(gamepads.keys(), [20]);
        events.clear();
        // A pad that is not the standard layout says so, and takes pad 0 again.
        gamepads.update(30, &Pad { standard: false, ..pad() }, &mut events);
        assert_eq!(events, [GameEvent::GamepadConnected { pad: 0, standard: 0 }]);
        // Disconnecting a pad the game never heard of says nothing.
        events.clear();
        gamepads.disconnect(99, &mut events);
        assert!(events.is_empty());
    }

    #[test]
    fn pads_past_the_limit_are_left_out() {
        let mut gamepads = Gamepads::default();
        let mut events = Vec::new();
        for key in 0..=MAX_GAMEPADS as u64 {
            gamepads.update(key, &pad(), &mut events);
        }
        assert_eq!(events.len(), MAX_GAMEPADS);
        assert_eq!(gamepads.keys().len(), MAX_GAMEPADS);
    }
}
