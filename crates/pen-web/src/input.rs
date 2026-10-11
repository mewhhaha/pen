use std::cell::RefCell;
use std::rc::Rc;
use pen_protocol::GameEvent;
use wasm_bindgen::{JsCast, closure::Closure};
use web_sys::{Event, EventTarget, HtmlCanvasElement, KeyboardEvent, MouseEvent, WheelEvent};

#[derive(Default)]
pub struct Input { pub events: Vec<GameEvent>, pub pointer: (f32, f32), pub hotkeys: Vec<pen_runtime::input::Hotkey>, held: std::collections::HashSet<String> }
fn listen(target: &EventTarget, name: &str, call: impl FnMut(Event) + 'static) {
    let callback = Closure::<dyn FnMut(Event)>::new(call);
    target.add_event_listener_with_callback(name, callback.as_ref().unchecked_ref()).unwrap();
    callback.forget();
}
impl Input {
    pub fn install(canvas: &HtmlCanvasElement) -> Rc<RefCell<Self>> {
        let input = Rc::new(RefCell::new(Self::default()));
        let window = web_sys::window().unwrap();
        for (name, down) in [("keydown", true), ("keyup", false)] {
            let state = input.clone();
            listen(&window, name, move |event| {
                let key: KeyboardEvent = event.unchecked_into();
                let mut input = state.borrow_mut();
                if !down && input.held.remove(&key.code()) { return; }
                if down && key.ctrl_key() && key.shift_key() && !key.alt_key() && !key.meta_key() {
                    let hotkey = match key.code().as_str() {
                        "KeyP" => Some(pen_runtime::input::Hotkey::Pause),
                        "Period" => Some(pen_runtime::input::Hotkey::Step),
                        "Comma" => Some(pen_runtime::input::Hotkey::Back),
                        "KeyR" => Some(pen_runtime::input::Hotkey::Replay),
                        _ => None,
                    };
                    if let Some(hotkey) = hotkey {
                        input.hotkeys.push(hotkey); input.held.insert(key.code()); key.prevent_default(); return;
                    }
                }
                if key.repeat() || key.key() == "F11" { return; }
                if ["Tab", " ", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "`"].contains(&key.key().as_str()) {
                    key.prevent_default();
                }
                if down { input.events.push(GameEvent::KeyDown(key.key())); }
                else { input.events.push(GameEvent::KeyUp(key.key())); }
            });
        }
        let state = input.clone();
        let element = canvas.clone();
        listen(&window, "mousemove", move |event| {
            let mouse: MouseEvent = event.unchecked_into();
            let mut state = state.borrow_mut();
            if web_sys::window().unwrap().document().unwrap().pointer_lock_element().is_some() {
                state.pointer.0 += mouse.movement_x() as f32;
                state.pointer.1 += mouse.movement_y() as f32;
            } else {
                let rect = element.get_bounding_client_rect();
                state.pointer = ((mouse.client_x() as f64 - rect.x()) as f32, (mouse.client_y() as f64 - rect.y()) as f32);
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
                let button = mouse.button() as u32;
                if button == 2 {
                    if down { element.request_pointer_lock(); }
                    else { web_sys::window().unwrap().document().unwrap().exit_pointer_lock(); }
                }
                if down { state.events.push(GameEvent::PointerDown { button, x, y }); }
                else { state.events.push(GameEvent::PointerUp { button, x, y }); }
            });
        }
        let state = input.clone();
        listen(canvas, "wheel", move |event| {
            let wheel: WheelEvent = event.unchecked_into();
            wheel.prevent_default();
            let mut state = state.borrow_mut();
            let (x, y) = state.pointer;
            let mut delta = -wheel.delta_y() as f32;
            if wheel.delta_mode() == 1 { delta *= 100.0; }
            if wheel.delta_mode() == 2 { delta *= web_sys::window().unwrap().inner_height().unwrap().as_f64().unwrap() as f32; }
            state.events.push(GameEvent::Wheel { x, y, delta });
        });
        let state = input.clone();
        listen(&window, "blur", move |_| state.borrow_mut().events.push(GameEvent::FocusLost));
        listen(canvas, "contextmenu", move |event| event.prevent_default());
        input
    }
}
