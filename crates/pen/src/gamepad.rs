//! The player's gamepads, through gilrs (evdev on Linux, Windows.Gaming.Input
//! on Windows), read into pen-runtime's standard layout
//! (`pen_runtime::gamepad`), which numbers them, applies the sticks' dead
//! zone and makes the game's events, as the browser host does with the
//! Gamepad API. gilrs maps each pad's buttons and axes to one layout with
//! SDL's controller database; its own filters are off, so pen-runtime's dead
//! zone is the only one, as in the browser. Only a window reads pads: a
//! headless run's pads are its script's.
use std::collections::HashMap;

use gilrs::ff::{BaseEffect, BaseEffectType, Effect, EffectBuilder, Repeat, Replay, Ticks};
use gilrs::{Axis, Button, EventType, GamepadId, Gilrs, GilrsBuilder};
use pen_protocol::{GameEvent, Haptic, gamepad_axis, gamepad_button};
use pen_runtime::gamepad::{Gamepads, Pad};

/// gilrs's buttons by the standard gamepad's.
const BUTTONS: [(Button, u32); 17] = [
    (Button::South, gamepad_button::SOUTH),
    (Button::East, gamepad_button::EAST),
    (Button::West, gamepad_button::WEST),
    (Button::North, gamepad_button::NORTH),
    (Button::LeftTrigger, gamepad_button::LEFT_SHOULDER),
    (Button::RightTrigger, gamepad_button::RIGHT_SHOULDER),
    (Button::LeftTrigger2, gamepad_button::LEFT_TRIGGER),
    (Button::RightTrigger2, gamepad_button::RIGHT_TRIGGER),
    (Button::Select, gamepad_button::SELECT),
    (Button::Start, gamepad_button::START),
    (Button::LeftThumb, gamepad_button::LEFT_STICK),
    (Button::RightThumb, gamepad_button::RIGHT_STICK),
    (Button::DPadUp, gamepad_button::DPAD_UP),
    (Button::DPadDown, gamepad_button::DPAD_DOWN),
    (Button::DPadLeft, gamepad_button::DPAD_LEFT),
    (Button::DPadRight, gamepad_button::DPAD_RIGHT),
    (Button::Mode, gamepad_button::HOME),
];

/// gilrs's stick axes by the standard gamepad's, and whether the axis runs
/// the other way: gilrs's sticks read up as positive, the standard
/// gamepad's down.
const AXES: [(Axis, u32, bool); 4] = [
    (Axis::LeftStickX, gamepad_axis::LEFT_X, false),
    (Axis::LeftStickY, gamepad_axis::LEFT_Y, true),
    (Axis::RightStickX, gamepad_axis::RIGHT_X, false),
    (Axis::RightStickY, gamepad_axis::RIGHT_Y, true),
];

/// What one of gilrs's pads reads, in the standard layout. A directional
/// pad that reports as a hat (two axes) reads as its four buttons.
fn read(gamepad: &gilrs::Gamepad<'_>) -> Pad {
    let mut pad = Pad::default();
    for (button, standard) in BUTTONS {
        pad.buttons[standard as usize] = gamepad.button_data(button).map_or(0.0, gilrs::ev::state::ButtonData::value);
    }
    for (axis, standard, flipped) in AXES {
        let value = gamepad.value(axis);
        pad.axes[standard as usize] = if flipped { -value } else { value };
    }
    let (hat_x, hat_y) = (gamepad.value(Axis::DPadX), gamepad.value(Axis::DPadY));
    for (button, held) in [
        (gamepad_button::DPAD_LEFT, hat_x <= -0.5),
        (gamepad_button::DPAD_RIGHT, hat_x >= 0.5),
        (gamepad_button::DPAD_UP, hat_y >= 0.5),
        (gamepad_button::DPAD_DOWN, hat_y <= -0.5),
    ] {
        if held {
            pad.buttons[button as usize] = 1.0;
        }
    }
    pad
}

fn key(id: GamepadId) -> u64 {
    usize::from(id) as u64
}

/// The window's gamepads, through gilrs.
pub struct Pads {
    gilrs: Gilrs,
    gamepads: Gamepads,
    ids: HashMap<u64, GamepadId>,
    effects: HashMap<u32, Effect>,
}

impl Pads {
    /// gilrs, or None (and the game plays without pads) where it cannot
    /// read them: no udev, no permission, or a platform it does not know.
    pub fn open() -> Option<Pads> {
        let built = match GilrsBuilder::new().with_default_filters(false).with_force_feedback(true).build() {
            Ok(gilrs) => Ok(gilrs),
            Err(_) => GilrsBuilder::new().with_default_filters(false).with_force_feedback(false).build(),
        };
        match built {
            Ok(gilrs) => {
                Some(Pads { gilrs, gamepads: Gamepads::default(), ids: HashMap::new(), effects: HashMap::new() })
            }
            Err(error) => {
                eprintln!("gamepads are off: {error}");
                None
            }
        }
    }

    /// The events of what the pads did since the last call: each change
    /// gilrs saw, so a press shorter than a frame is not lost.
    pub fn poll(&mut self, events: &mut Vec<GameEvent>) {
        while let Some(event) = self.gilrs.next_event() {
            let k = key(event.id);
            match event.event {
                EventType::Disconnected => {
                    self.ids.remove(&k);
                    self.gamepads.disconnect(k, events);
                }
                EventType::Dropped | EventType::ForceFeedbackEffectCompleted => {}
                _ => {
                    self.ids.insert(k, event.id);
                    let pad = read(&self.gilrs.gamepad(event.id));
                    self.gamepads.update(k, &pad, events);
                }
            }
        }
    }

    /// Drives dual-motor force feedback on connected gamepads from frame haptic commands.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped motor magnitude and ms")]
    pub fn apply_haptics(&mut self, haptics: &[Haptic]) {
        for cmd in haptics {
            let Some(k) = self.gamepads.key_for(cmd.pad) else { continue };
            let Some(&id) = self.ids.get(&k) else { continue };
            if !self.gilrs.gamepad(id).is_ff_supported() {
                continue;
            }
            if let Some(old) = self.effects.remove(&cmd.pad) {
                let _ = old.stop();
            }
            let low = if cmd.low.is_finite() { cmd.low.clamp(0.0, 1.0) } else { 0.0 };
            let high = if cmd.high.is_finite() { cmd.high.clamp(0.0, 1.0) } else { 0.0 };
            if low <= 0.0 && high <= 0.0 {
                continue;
            }
            let ms = if cmd.duration.is_finite() && cmd.duration > 0.0 {
                (cmd.duration.clamp(0.016, 10.0) * 1000.0).round() as u32
            } else {
                50
            };
            let scheduling = Replay { play_for: Ticks::from_ms(ms), ..Default::default() };
            let mut builder = EffectBuilder::new();
            if low > 0.0 {
                builder.add_effect(BaseEffect {
                    kind: BaseEffectType::Strong { magnitude: (low * 65535.0).round() as u16 },
                    scheduling,
                    ..Default::default()
                });
            }
            if high > 0.0 {
                builder.add_effect(BaseEffect {
                    kind: BaseEffectType::Weak { magnitude: (high * 65535.0).round() as u16 },
                    scheduling,
                    ..Default::default()
                });
            }
            if let Ok(effect) = builder.repeat(Repeat::For(Ticks::from_ms(ms))).gamepads(&[id]).finish(&mut self.gilrs)
            {
                let _ = effect.play();
                self.effects.insert(cmd.pad, effect);
            }
        }
    }
}
