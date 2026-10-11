//! Window input as the game's events, in the browser host's terms (gdev's
//! `event_pump.ts` and `web/client.ts`): key names as `KeyboardEvent.key`
//! gives them, pointer positions in window pixels, buttons numbered as
//! `MouseEvent.button`, and wheel deltas in pixels. Holding the secondary
//! button locks the pointer, so looking around has no screen edge.
//!
//! Time travel takes Ctrl+Shift with a key, which the game never sees: P
//! pauses and resumes, period steps a frame, comma steps a frame back, and
//! R replays the last seconds (after a hot reload, on the new build).
//!
//! In `pen dev` and `pen run` the dev tools (`devtools`) see window
//! events before this does (`Session::overlay_event`): the backquote key is
//! its own, and while one of its panels has the keyboard or the pointer, the
//! game gets neither.
use pen_protocol::{Frame, GameEvent, Viewport};
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Window};

/// Pixels a wheel notch scrolls, as browsers report `deltaY` for one line.
pub(crate) const PIXELS_PER_LINE: f32 = 100.0;

pub use pen_runtime::input::Hotkey;

#[derive(Default)]
pub struct Input {
    pending: Vec<GameEvent>,
    pointer: (f32, f32),
    locked: bool,
    modifiers: ModifiersState,
    hotkeys: Vec<Hotkey>,
    /// Hotkeys held down, whose release the game must not see either.
    held: Vec<KeyCode>,
}

fn hotkey(code: KeyCode) -> Option<Hotkey> {
    Some(match code {
        KeyCode::KeyP => Hotkey::Pause,
        KeyCode::Period => Hotkey::Step,
        KeyCode::Comma => Hotkey::Back,
        KeyCode::KeyR => Hotkey::Replay,
        _ => return None,
    })
}

/// The `KeyboardEvent.key` value for a key, or None for keys the browser
/// would not report either.
pub(crate) fn key_name(key: &Key) -> Option<String> {
    Some(match key {
        Key::Character(text) => text.to_string(),
        Key::Named(NamedKey::Space) => " ".to_owned(),
        // W3C names the logo key "Meta"; winit calls it Super.
        Key::Named(NamedKey::Super | NamedKey::Meta) => "Meta".to_owned(),
        // winit's named keys carry the W3C key values' names.
        Key::Named(named) => format!("{named:?}"),
        _ => return None,
    })
}

pub(crate) fn button_number(button: MouseButton) -> u32 {
    match button {
        MouseButton::Left => 0,
        MouseButton::Middle => 1,
        MouseButton::Right => 2,
        MouseButton::Back => 3,
        MouseButton::Forward => 4,
        MouseButton::Other(other) => u32::from(other),
    }
}

impl Input {
    pub fn enqueue(&mut self, event: GameEvent) {
        self.pending.push(event);
    }

    pub fn window_event(&mut self, window: &Window, event: &WindowEvent) {
        match event {
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;
                    let chord = self.modifiers.control_key()
                        && self.modifiers.shift_key()
                        && !self.modifiers.alt_key()
                        && !self.modifiers.super_key();
                    // A key held since before the chord is the game's, repeats and all.
                    let ours = !event.repeat || self.held.contains(&code);
                    if pressed
                        && chord
                        && ours
                        && let Some(hotkey) = hotkey(code)
                    {
                        // Held, step and step back repeat.
                        if !event.repeat || matches!(hotkey, Hotkey::Step | Hotkey::Back) {
                            self.hotkeys.push(hotkey);
                        }
                        if !self.held.contains(&code) {
                            self.held.push(code);
                        }
                        return;
                    }
                    if !pressed && let Some(index) = self.held.iter().position(|held| *held == code) {
                        self.held.swap_remove(index);
                        return;
                    }
                }
                if event.repeat && event.state == ElementState::Pressed {
                    return;
                }
                // The window's own key, as F11 is the browser's: the game never sees it.
                if event.logical_key == Key::Named(NamedKey::F11) {
                    if event.state == ElementState::Pressed {
                        let fullscreen = window.fullscreen().is_none().then_some(Fullscreen::Borderless(None));
                        window.set_fullscreen(fullscreen);
                    }
                    return;
                }
                let Some(name) = key_name(&event.logical_key) else { return };
                self.pending.push(match event.state {
                    ElementState::Pressed => GameEvent::KeyDown(name),
                    ElementState::Released => GameEvent::KeyUp(name),
                });
            }
            WindowEvent::CursorMoved { position, .. } => {
                // While locked, motion comes from raw device deltas instead.
                if !self.locked {
                    self.pointer = (position.x as f32, position.y as f32);
                    let (x, y) = self.pointer;
                    self.pending.push(GameEvent::PointerMove { x, y });
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let (x, y) = self.pointer;
                let button = button_number(*button);
                match state {
                    ElementState::Pressed => {
                        if button == 2 {
                            self.lock(window, true);
                        }
                        self.pending.push(GameEvent::PointerDown { button, x, y });
                    }
                    ElementState::Released => {
                        if button == 2 {
                            self.lock(window, false);
                        }
                        self.pending.push(GameEvent::PointerUp { button, x, y });
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    MouseScrollDelta::LineDelta(_, lines) => -lines * PIXELS_PER_LINE,
                    MouseScrollDelta::PixelDelta(pixels) => -pixels.y as f32,
                };
                let (x, y) = self.pointer;
                self.pending.push(GameEvent::Wheel { x, y, delta });
            }
            WindowEvent::Focused(false) => {
                self.lock(window, false);
                self.pending.push(GameEvent::FocusLost);
            }
            _ => {}
        }
    }

    pub fn device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event
            && self.locked
        {
            self.pointer.0 += *dx as f32;
            self.pointer.1 += *dy as f32;
            let (x, y) = self.pointer;
            self.pending.push(GameEvent::PointerMove { x, y });
        }
    }

    fn lock(&mut self, window: &Window, lock: bool) {
        if lock == self.locked {
            return;
        }
        if lock {
            // Not every platform can lock; motion then comes from positions.
            self.locked = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
                .is_ok();
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            self.locked = false;
        }
    }

    pub fn next_frame(&mut self, timestamp: f64, viewport: Viewport) -> Frame {
        Frame { timestamp, viewport, events: std::mem::take(&mut self.pending) }
    }

    /// Forgets the input so far (a replay's frames bring their own).
    pub fn discard(&mut self) {
        self.pending.clear();
    }

    /// The time travel keys pressed since the last call.
    pub fn take_hotkeys(&mut self) -> Vec<Hotkey> {
        std::mem::take(&mut self.hotkeys)
    }
}
