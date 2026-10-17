//! The page's keyboard, pointer and wheel as the game's events, with the
//! time travel shortcuts (Ctrl+Shift+P, period, comma and R) taken first.
//! Text comes as TextInput events, typed or from an input method, which
//! composes into a hidden text field while the game's focused node is a
//! text box (`text_field`); its composition comes as Composition events. While the pointer is locked, the mouse's motion comes
//! as MouseMotion events too. Touches (pointer events whose pointerType is
//! touch), their pans and pinches, and gamepads (the Gamepad API, read each
//! frame by `poll_gamepads`) go through pen-runtime's `touch` and `gamepad`,
//! as natively.
#![allow(clippy::cast_possible_truncation, reason = "page positions and wheel deltas narrow to f32 lanes")]
use pen_protocol::{GameEvent, touch_phase};
use pen_runtime::gamepad::{AXES, BUTTONS, Gamepads, Pad};
use pen_runtime::{text, touch::Touches};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::{JsCast, closure::Closure};
use web_sys::{
    CompositionEvent, Event, EventTarget, Gamepad, GamepadButton, GamepadMappingType, HtmlCanvasElement,
    HtmlTextAreaElement, InputEvent, KeyboardEvent, MouseEvent, PointerEvent, WheelEvent,
};

/// What came in since the frame loop last looked: the game's events (which
/// the dev tools see first), assistive technology's actions on the dev
/// tools' interface, and the time travel shortcuts.
pub struct Input {
    pub events: Vec<GameEvent>,
    pub tools: Vec<GameEvent>,
    pub hotkeys: Vec<pen_runtime::input::Hotkey>,
    /// Where the pointer is, in the canvas's backing pixels.
    pointer: (f32, f32),
    held: std::collections::HashSet<String>,
    canvas: HtmlCanvasElement,
    touches: Touches,
    gamepads: Gamepads,
    /// The hidden field an input method composes into, and the text box
    /// (in the canvas's backing pixels) it lies over while one has the focus.
    field: HtmlTextAreaElement,
    text_box: Option<(f32, f32, f32, f32)>,
    composing: bool,
}
fn listen(target: &EventTarget, name: &str, call: impl FnMut(Event) + 'static) {
    let callback = Closure::<dyn FnMut(Event)>::new(call);
    target.add_event_listener_with_callback(name, callback.as_ref().unchecked_ref()).unwrap();
    callback.forget();
}

/// The canvas's backing pixels per CSS pixel.
fn ratio(canvas: &HtmlCanvasElement) -> f64 {
    let rect = canvas.get_bounding_client_rect();
    if rect.width() > 0.0 { f64::from(canvas.width()) / rect.width() } else { 1.0 }
}

/// CSS pixels on the canvas as its backing pixels, which the game lays out
/// on, each axis apart (`pen_runtime::scale`).
fn to_backing(canvas: &HtmlCanvasElement, x: f64, y: f64) -> (f32, f32) {
    let rect = canvas.get_bounding_client_rect();
    let (x, y) = pen_runtime::scale::to_frame((x, y), (rect.width(), rect.height()), (canvas.width(), canvas.height()));
    (x as f32, y as f32)
}

/// Where a mouse or pointer event is, in the canvas's backing pixels.
fn place(canvas: &HtmlCanvasElement, mouse: &MouseEvent) -> (f32, f32) {
    let rect = canvas.get_bounding_client_rect();
    to_backing(canvas, f64::from(mouse.client_x()) - rect.x(), f64::from(mouse.client_y()) - rect.y())
}

/// A gamepad as the standard layout reads it: the first buttons and axes,
/// in the order the browser gives them.
fn read(gamepad: &Gamepad) -> Pad {
    let mut pad = Pad { standard: gamepad.mapping() == GamepadMappingType::Standard, ..Pad::default() };
    for (slot, button) in pad.buttons.iter_mut().zip(gamepad.buttons().iter().take(BUTTONS)) {
        *slot = button.dyn_into::<GamepadButton>().map_or(0.0, |button| button.value() as f32);
    }
    for (slot, axis) in pad.axes.iter_mut().zip(gamepad.axes().iter().take(AXES)) {
        *slot = axis.as_f64().unwrap_or(0.0) as f32;
    }
    pad
}

impl Input {
    pub fn install(canvas: &HtmlCanvasElement) -> Rc<RefCell<Self>> {
        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();
        // Invisible but focusable, so input methods compose into it.
        let field: HtmlTextAreaElement = document.create_element("textarea").unwrap().unchecked_into();
        let _ = field.set_attribute(
            "style",
            "position:absolute;opacity:0;pointer-events:none;resize:none;border:0;padding:0;\
             margin:0;font-size:16px;left:0;top:0;width:1px;height:1px",
        );
        let _ = field.set_attribute("aria-hidden", "true");
        let _ = field.set_attribute("autocomplete", "off");
        let _ = field.set_attribute("tabindex", "-1");
        document.body().unwrap().append_child(&field).unwrap();
        // Touches are the game's, not the page's to pan or zoom.
        let _ = canvas.style().set_property("touch-action", "none");
        let input = Rc::new(RefCell::new(Input {
            events: Vec::new(),
            tools: Vec::new(),
            hotkeys: Vec::new(),
            pointer: (0.0, 0.0),
            held: Default::default(),
            canvas: canvas.clone(),
            touches: Touches::default(),
            gamepads: Gamepads::default(),
            field: field.clone(),
            text_box: None,
            composing: false,
        }));
        for (name, down) in [("keydown", true), ("keyup", false)] {
            let state = input.clone();
            listen(&window, name, move |event| {
                let key: KeyboardEvent = event.unchecked_into();
                let mut input = state.borrow_mut();
                // What an input method takes, it composes.
                if key.is_composing() || key.key() == "Process" {
                    return;
                }
                if !down && input.held.remove(&key.code()) {
                    return;
                }
                if down && key.ctrl_key() && key.shift_key() && !key.alt_key() && !key.meta_key() {
                    let hotkey = match key.code().as_str() {
                        "KeyP" => Some(pen_runtime::input::Hotkey::Pause),
                        "Period" => Some(pen_runtime::input::Hotkey::Step),
                        "Comma" => Some(pen_runtime::input::Hotkey::Back),
                        "KeyR" => Some(pen_runtime::input::Hotkey::Replay),
                        _ => None,
                    };
                    if let Some(hotkey) = hotkey {
                        input.hotkeys.push(hotkey);
                        input.held.insert(key.code());
                        key.prevent_default();
                        return;
                    }
                }
                if key.key() == "F11" {
                    return;
                }
                let typing = input.text_box.is_some();
                // In the text field, space types; elsewhere it would scroll the page.
                let page_keys = ["Tab", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "`"];
                if page_keys.contains(&key.key().as_str()) || (key.key() == " " && !typing) {
                    key.prevent_default();
                }
                if !down {
                    input.events.push(GameEvent::KeyUp(key.key()));
                    return;
                }
                if !key.repeat() {
                    input.events.push(GameEvent::KeyDown(key.key()));
                }
                // A key's text follows it, repeating as it is held; the text
                // field's own input brings it while a text box has the focus.
                if !typing && !key.ctrl_key() && !key.meta_key() && key.key().chars().count() == 1 {
                    text::commit(&key.key(), &mut input.events);
                }
            });
        }
        // The field's input: typed text, or the composition so far, once
        // the field holds it (its caret is the composition's).
        let state = input.clone();
        listen(&field, "input", move |event| {
            let typed: InputEvent = event.unchecked_into();
            let mut input = state.borrow_mut();
            match typed.input_type().as_str() {
                "insertText" => text::commit(&typed.data().unwrap_or_default(), &mut input.events),
                "insertCompositionText" if input.composing => {
                    let value = input.field.value();
                    let cursor = input.field.selection_start().ok().flatten();
                    let cursor = cursor.map(|at| text::characters_before_utf16(&value, at as usize));
                    text::compose(&value, cursor, &mut input.events);
                }
                _ => {}
            }
            if !input.composing {
                input.field.set_value("");
            }
        });
        let state = input.clone();
        listen(&field, "compositionstart", move |_| state.borrow_mut().composing = true);
        let state = input.clone();
        listen(&field, "compositionend", move |event| {
            let composition: CompositionEvent = event.unchecked_into();
            let mut input = state.borrow_mut();
            input.composing = false;
            text::compose("", None, &mut input.events);
            text::commit(&composition.data().unwrap_or_default(), &mut input.events);
            input.field.set_value("");
        });
        let state = input.clone();
        let element = canvas.clone();
        listen(&window, "mousemove", move |event| {
            let mouse: MouseEvent = event.unchecked_into();
            let mut state = state.borrow_mut();
            // CSS pixels to the canvas's backing pixels, which the game lays out on.
            if web_sys::window().unwrap().document().unwrap().pointer_lock_element().is_some() {
                let (dx, dy) = (mouse.movement_x() as f32, mouse.movement_y() as f32);
                state.events.push(GameEvent::MouseMotion { dx, dy });
                let (dx, dy) = to_backing(&element, f64::from(mouse.movement_x()), f64::from(mouse.movement_y()));
                state.pointer.0 += dx;
                state.pointer.1 += dy;
            } else {
                state.pointer = place(&element, &mouse);
            }
            let (x, y) = state.pointer;
            state.events.push(GameEvent::PointerMove { x, y });
        });
        for (name, down) in [("mousedown", true), ("mouseup", false)] {
            let state = input.clone();
            let element = canvas.clone();
            listen(&window, name, move |event| {
                let mouse: MouseEvent = event.unchecked_into();
                let mut state = state.borrow_mut();
                let (x, y) = state.pointer;
                let Ok(button) = u32::try_from(mouse.button()) else { return };
                if button == 2 {
                    if down {
                        element.request_pointer_lock();
                    } else {
                        web_sys::window().unwrap().document().unwrap().exit_pointer_lock();
                    }
                }
                if down {
                    state.events.push(GameEvent::PointerDown { button, x, y });
                } else {
                    state.events.push(GameEvent::PointerUp { button, x, y });
                }
            });
        }
        let phases = [
            ("pointerdown", touch_phase::START),
            ("pointermove", touch_phase::MOVE),
            ("pointerup", touch_phase::END),
            ("pointercancel", touch_phase::CANCEL),
        ];
        for (name, phase) in phases {
            let state = input.clone();
            let element = canvas.clone();
            listen(canvas, name, move |event| {
                let pointer: PointerEvent = event.unchecked_into();
                if pointer.pointer_type() != "touch" {
                    return;
                }
                // The touch makes the pointer's events itself (pen-runtime's
                // touch), so the browser's mouse events for it would be twice.
                if phase == touch_phase::START {
                    pointer.prevent_default();
                }
                let (x, y) = place(&element, &pointer);
                let key = u64::from(pointer.pointer_id().cast_unsigned());
                let mut state = state.borrow_mut();
                let state = &mut *state;
                state.touches.touch(key, phase, x, y, &mut state.events);
            });
        }
        let state = input.clone();
        listen(canvas, "wheel", move |event| {
            let wheel: WheelEvent = event.unchecked_into();
            wheel.prevent_default();
            let mut state = state.borrow_mut();
            let (x, y) = state.pointer;
            let mut delta = -wheel.delta_y() as f32;
            if wheel.delta_mode() == 1 {
                delta *= 100.0;
            }
            if wheel.delta_mode() == 2 {
                delta *= web_sys::window().unwrap().inner_height().unwrap().as_f64().unwrap() as f32;
            }
            state.events.push(GameEvent::Wheel { x, y, delta });
        });
        let state = input.clone();
        listen(&window, "blur", move |_| {
            let mut state = state.borrow_mut();
            let state = &mut *state;
            state.touches.cancel(&mut state.events);
            state.events.push(GameEvent::FocusLost);
        });
        listen(canvas, "contextmenu", move |event| event.prevent_default());
        input
    }

    /// Reads the gamepads (the Gamepad API has no events for their buttons
    /// and sticks): what changed since the last frame, and those that left.
    pub fn poll_gamepads(&mut self) {
        let Ok(list) = web_sys::window().unwrap().navigator().get_gamepads() else { return };
        let mut present = Vec::new();
        for gamepad in list.iter().filter_map(|gamepad| gamepad.dyn_into::<Gamepad>().ok()) {
            if !gamepad.connected() {
                continue;
            }
            let key = u64::from(gamepad.index());
            present.push(key);
            self.gamepads.update(key, &read(&gamepad), &mut self.events);
        }
        for key in self.gamepads.keys() {
            if !present.contains(&key) {
                self.gamepads.disconnect(key, &mut self.events);
            }
        }
    }

    /// The game's focused text box (in backing pixels), or None: the
    /// hidden field takes the focus over it, so an input method's window
    /// opens there, or gives it back to the canvas.
    pub fn text_field(&mut self, text_box: Option<(f32, f32, f32, f32)>) {
        if text_box == self.text_box {
            return;
        }
        let was = std::mem::replace(&mut self.text_box, text_box);
        let Some((x, y, width, height)) = text_box else {
            if self.composing {
                self.composing = false;
                text::compose("", None, &mut self.events);
            }
            let _ = self.field.blur();
            let _ = self.canvas.focus();
            return;
        };
        // Backing pixels to the page's CSS pixels.
        let window = web_sys::window().unwrap();
        let rect = self.canvas.get_bounding_client_rect();
        let ratio = ratio(&self.canvas);
        let left = rect.x() + window.scroll_x().unwrap_or(0.0) + f64::from(x) / ratio;
        let top = rect.y() + window.scroll_y().unwrap_or(0.0) + f64::from(y) / ratio;
        let style = self.field.style();
        for (property, value) in [
            ("left", left),
            ("top", top),
            ("width", (f64::from(width) / ratio).max(1.0)),
            ("height", (f64::from(height) / ratio).max(1.0)),
        ] {
            let _ = style.set_property(property, &format!("{value}px"));
        }
        if was.is_none() {
            self.field.set_value("");
            let _ = self.field.focus();
        }
    }
}
