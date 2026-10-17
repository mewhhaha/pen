//! Window input as the game's events, in the browser host's terms: key
//! names as `KeyboardEvent.key` gives them, pointer positions in the
//! physical pixels of the frame the game draws (the window's, or a pinned
//! frame's stretched over it; `pen_runtime::scale`), buttons numbered as
//! `MouseEvent.button`, and wheel deltas in logical pixels, as a browser's
//! CSS pixels. Holding the secondary
//! button locks the pointer, so looking around has no screen edge; while it
//! is locked, the mouse's raw motion comes as MouseMotion events too.
//!
//! Text comes as TextInput events, typed or from an input method (IME),
//! whose composition comes as Composition events; the input method is on
//! only while a text box has the focus (`text_field`). Touches, and the
//! pans and pinches they make, and gamepads (`crate::gamepad`) go through
//! pen-runtime's `touch` and `gamepad`, which the browser shares.
//!
//! Time travel takes Ctrl+Shift with a key, which the game never sees: P
//! pauses and resumes, period steps a frame, comma steps a frame back, and
//! R replays the last seconds (after a hot reload, on the new build).
//!
//! In `pen dev` and `pen run` the dev tools (`devtools`) see window
//! events before this does (`Session::overlay_event`): the backquote key is
//! theirs, the keyboard while one of their nodes has the focus, and the
//! pointer over their dialogs.
use pen_protocol::{Frame, GameEvent, UiNode, Viewport, touch_phase};
use pen_runtime::{text, touch::Touches};
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{DeviceEvent, ElementState, Ime, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Window};

use crate::gamepad::Pads;

/// Pixels a wheel notch scrolls, as browsers report `deltaY` for one line.
pub(crate) const PIXELS_PER_LINE: f32 = 100.0;

/// A number (a position, a delta, a value) as the f32 a game event holds.
#[allow(clippy::cast_possible_truncation, reason = "narrowing f64 to f32 is the point")]
pub(crate) fn narrow(value: f64) -> f32 {
    value as f32
}

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
    /// The size frames keep whatever the window's (a recording's), which
    /// pointer positions are mapped onto.
    pinned: Option<(u32, u32)>,
    /// Ctrl+Shift+B asked for a bug report (`bugreport`).
    report: bool,
    touches: Touches,
    /// The window's gamepads, once `open_gamepads` found any way to read them.
    pads: Option<Pads>,
    /// The focused text box the input method composes over, while one has
    /// the focus, in window pixels.
    text_box: Option<(f32, f32, f32, f32)>,
    /// An input method is composing.
    composing: bool,
}

/// winit's touch phase as the protocol's.
fn phase(phase: TouchPhase) -> u32 {
    match phase {
        TouchPhase::Started => touch_phase::START,
        TouchPhase::Moved => touch_phase::MOVE,
        TouchPhase::Ended => touch_phase::END,
        TouchPhase::Cancelled => touch_phase::CANCEL,
    }
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

    /// Frames keep `size` however large the window is.
    pub fn pin(&mut self, size: (u32, u32)) {
        self.pinned = Some(size);
    }

    /// A position in the window's pixels on the frame the game draws.
    fn on_frame(&self, window: &Window, x: f64, y: f64) -> (f32, f32) {
        let (x, y) = match self.pinned {
            Some(frame) => {
                let shown = window.inner_size();
                pen_runtime::scale::to_frame((x, y), (f64::from(shown.width), f64::from(shown.height)), frame)
            }
            None => (x, y),
        };
        (narrow(x), narrow(y))
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
                    if pressed && chord && !event.repeat && code == KeyCode::KeyB {
                        self.report = true;
                        if !self.held.contains(&code) {
                            self.held.push(code);
                        }
                        return;
                    }
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
                // The window's own key, as F11 is the browser's: the game never sees it.
                if event.logical_key == Key::Named(NamedKey::F11) {
                    if event.state == ElementState::Pressed && !event.repeat {
                        let fullscreen = window.fullscreen().is_none().then_some(Fullscreen::Borderless(None));
                        window.set_fullscreen(fullscreen);
                    }
                    return;
                }
                // The key's text follows it, repeating while it is held, as a
                // text box expects; with Ctrl or the logo key held, a key is a
                // shortcut, and while an input method composes, it is the
                // input method's.
                let typed = event
                    .text
                    .as_ref()
                    .filter(|_| !self.composing && !self.modifiers.control_key() && !self.modifiers.super_key());
                if event.state == ElementState::Pressed && !event.repeat {
                    if let Some(name) = key_name(&event.logical_key) {
                        self.pending.push(GameEvent::KeyDown(name));
                    }
                } else if event.state == ElementState::Released {
                    if let Some(name) = key_name(&event.logical_key) {
                        self.pending.push(GameEvent::KeyUp(name));
                    }
                    return;
                }
                if let Some(typed) = typed {
                    text::commit(typed, &mut self.pending);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                // While locked, motion comes from raw device deltas instead.
                if !self.locked {
                    self.pointer = self.on_frame(window, position.x, position.y);
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
                    // Physical pixels, which a browser would give in CSS pixels.
                    MouseScrollDelta::PixelDelta(pixels) => {
                        -narrow(pen_runtime::scale::wheel_pixels(pixels.y, window.scale_factor()))
                    }
                };
                let (x, y) = self.pointer;
                self.pending.push(GameEvent::Wheel { x, y, delta });
            }
            WindowEvent::Focused(false) => {
                self.lock(window, false);
                self.touches.cancel(&mut self.pending);
                self.pending.push(GameEvent::FocusLost);
            }
            WindowEvent::Touch(touch) => {
                // On the frame, as the pointer the first touch drives.
                let (x, y) = self.on_frame(window, touch.location.x, touch.location.y);
                self.touches.touch(touch.id, phase(touch.phase), x, y, &mut self.pending);
            }
            WindowEvent::Ime(ime) => self.ime(ime),
            _ => {}
        }
    }

    /// An input method's composition and what it commits.
    fn ime(&mut self, ime: &Ime) {
        match ime {
            Ime::Preedit(composed, cursor) => {
                if !(composed.is_empty() && !self.composing) {
                    let cursor = cursor.map(|(start, _)| text::characters_before_byte(composed, start));
                    text::compose(composed, cursor, &mut self.pending);
                }
                self.composing = !composed.is_empty();
            }
            Ime::Commit(committed) => {
                if self.composing {
                    text::compose("", None, &mut self.pending);
                    self.composing = false;
                }
                text::commit(committed, &mut self.pending);
            }
            Ime::Disabled if self.composing => {
                text::compose("", None, &mut self.pending);
                self.composing = false;
            }
            Ime::Enabled | Ime::Disabled => {}
        }
    }

    /// The game's interface: the input method is on while one of its text
    /// boxes has the focus, composing over it. The dev tools' console types
    /// keys, which an input method would take.
    pub fn text_field(&mut self, window: Option<&Window>, ui: &[UiNode]) {
        let text_box = text::text_box(ui);
        if text_box == self.text_box {
            return;
        }
        if let Some(window) = window {
            if text_box.is_some() != self.text_box.is_some() {
                window.set_ime_allowed(text_box.is_some());
            }
            if let Some((x, y, width, height)) = text_box {
                window.set_ime_cursor_area(PhysicalPosition::new(x, y), PhysicalSize::new(width, height));
            }
        }
        self.text_box = text_box;
    }

    /// Reads the window's gamepads from now on.
    pub fn open_gamepads(&mut self) {
        self.pads = Pads::open();
    }

    /// What the gamepads did since the last frame.
    pub fn poll_gamepads(&mut self) {
        if let Some(pads) = &mut self.pads {
            pads.poll(&mut self.pending);
        }
    }

    pub fn device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event
            && self.locked
        {
            let (dx, dy) = (narrow(*dx), narrow(*dy));
            self.pending.push(GameEvent::MouseMotion { dx, dy });
            self.pointer.0 += dx;
            self.pointer.1 += dy;
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

    /// Whether Ctrl+Shift+B asked for a bug report since the last call.
    pub fn take_report(&mut self) -> bool {
        std::mem::take(&mut self.report)
    }
}
