//! The protocol's interface as a screen-reader tree beside the canvas.
use crate::input::Input;
use crate::program::error;
use anyhow::Result;
use pen_protocol::{GameEvent, UiNode, ui_action, ui_live, ui_role, ui_state};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::{JsCast, closure::Closure};
use web_sys::{Element, Event, HtmlElement, KeyboardEvent};

/// The interface shown last, and whose it is: the dev tools' (their actions
/// go to them) or the game's.
pub struct Accessibility {
    root: Element,
    last: Vec<UiNode>,
    tools: bool,
    replacing: Rc<Cell<bool>>,
    input: Rc<RefCell<Input>>,
    /// The captions' live region, while the game's sounds carry captions,
    /// and what it says.
    captions: Option<(Element, String)>,
}
impl Accessibility {
    pub fn new(input: Rc<RefCell<Input>>) -> Result<Self> {
        let document = web_sys::window().unwrap().document().unwrap();
        let root = document.create_element("div").map_err(error)?;
        root.set_attribute(
            "style",
            "position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);top:0;left:0",
        )
        .map_err(error)?;
        root.set_attribute("aria-label", "Game interface").map_err(error)?;
        document.body().unwrap().append_child(&root).map_err(error)?;
        Ok(Self { root, last: Vec::new(), tools: false, replacing: Rc::default(), input, captions: None })
    }

    /// The reading direction of the selected game language.
    pub fn set_rtl(&self, rtl: bool) -> Result<()> {
        self.root.set_attribute("dir", if rtl { "rtl" } else { "ltr" }).map_err(error)
    }
    /// The captions showing, a line each, in a polite live region after the
    /// interface; none when the game's sounds carry none.
    pub fn captions(&mut self, text: Option<&str>) -> Result<()> {
        let Some(text) = text else {
            if let Some((element, _)) = self.captions.take() {
                element.remove();
            }
            return Ok(());
        };
        if self.captions.is_none() {
            let document = web_sys::window().unwrap().document().unwrap();
            let element = document.create_element("div").map_err(error)?;
            for (key, value) in [("role", "status"), ("aria-live", "polite"), ("aria-label", "Captions")] {
                element.set_attribute(key, value).map_err(error)?;
            }
            self.root.after_with_node_1(&element).map_err(error)?;
            self.captions = Some((element, String::new()));
        }
        if let Some((element, shown)) = &mut self.captions
            && shown != text
        {
            element.set_text_content(Some(text));
            text.clone_into(shown);
        }
        Ok(())
    }
    /// Shows `nodes`, the dev tools' interface when `tools`, else the game's.
    pub fn update(&mut self, nodes: &[UiNode], tools: bool) -> Result<()> {
        if self.last == nodes && self.tools == tools {
            return Ok(());
        }
        let document = web_sys::window().unwrap().document().unwrap();
        let focused = document
            .active_element()
            .filter(|element| self.tools == tools && self.root.contains(Some(element)))
            .map(|element| element.id());
        let mut elements: Vec<Element> = Vec::with_capacity(nodes.len());
        let mut top_level = Vec::new();
        let mut listboxes = vec![false; nodes.len()];
        for (index, node) in nodes.iter().enumerate() {
            if option(node)
                && let Some(parent) = parent(index, node)
            {
                listboxes[parent] = nodes[parent].role == ui_role::LIST;
            }
        }
        for (index, node) in nodes.iter().enumerate() {
            let element = document.create_element("div").map_err(error)?;
            if node.hidden {
                element.set_attribute("aria-hidden", "true").map_err(error)?;
            }
            let role = if listboxes[index] {
                "listbox"
            } else if option(node) {
                "option"
            } else {
                ui_role::NAMES.get(node.role as usize).copied().unwrap_or("group")
            };
            element.set_attribute("role", role).map_err(error)?;
            element.set_attribute("id", &format!("pen-ui-{}", node.id)).map_err(error)?;
            let name_end = (node.name_length as usize).min(node.text.len());
            let description_end = name_end.saturating_add(node.description_length as usize).min(node.text.len());
            for (key, points) in [
                ("aria-label", &node.text[..name_end]),
                ("aria-description", &node.text[name_end..description_end]),
                ("aria-valuetext", &node.text[description_end..]),
            ] {
                if !points.is_empty() {
                    element.set_attribute(key, &pen_runtime::services::text(points)).map_err(error)?;
                }
            }
            if node.disabled {
                element.set_attribute("aria-disabled", "true").map_err(error)?;
            }
            if node.modal {
                element.set_attribute("aria-modal", "true").map_err(error)?;
            }
            let live = match node.live {
                ui_live::POLITE => Some("polite"),
                ui_live::ASSERTIVE => Some("assertive"),
                _ => None,
            };
            if let Some(live) = live {
                element.set_attribute("aria-live", live).map_err(error)?;
            }
            if node.checked != ui_state::NONE {
                let checked = match node.checked {
                    ui_state::TRUE => "true",
                    ui_state::MIXED => "mixed",
                    _ => "false",
                };
                element.set_attribute("aria-checked", checked).map_err(error)?;
            }
            for (key, state) in [("aria-expanded", node.expanded), ("aria-selected", node.selected)] {
                if state != ui_state::NONE {
                    element
                        .set_attribute(key, if state == ui_state::FALSE { "false" } else { "true" })
                        .map_err(error)?;
                }
            }
            if node.focusable || takes(node, ui_action::FOCUS) {
                element.set_attribute("tabindex", "-1").map_err(error)?;
            }
            let ranged = matches!(node.role, ui_role::SLIDER | ui_role::SPINBUTTON | ui_role::PROGRESSBAR)
                || [ui_action::INCREMENT, ui_action::DECREMENT, ui_action::SET_VALUE]
                    .into_iter()
                    .any(|action| takes(node, action));
            if ranged {
                for (key, value) in
                    [("aria-valuenow", node.value), ("aria-valuemin", node.minimum), ("aria-valuemax", node.maximum)]
                {
                    if value.is_finite() {
                        element.set_attribute(key, &value.to_string()).map_err(error)?;
                    }
                }
            }
            self.listen(&element, node, tools)?;
            if let Some(parent) = parent(index, node) {
                elements[parent].append_child(&element).map_err(error)?;
            } else {
                top_level.push(index);
            }
            elements.push(element);
        }

        // Keep a screen reader's DOM focus as the game redraws the same
        // control, without turning the replacement itself into a game action.
        self.replacing.set(true);
        let replaced: Result<()> = (|| {
            self.root.set_text_content(None);
            for index in top_level {
                self.root.append_child(&elements[index]).map_err(error)?;
            }
            if let Some(element) = elements
                .iter()
                .find(|element| focused.as_ref().is_some_and(|id| element.id() == *id))
                .and_then(|element| element.dyn_ref::<HtmlElement>())
                .filter(|element| element.has_attribute("tabindex"))
            {
                element.focus().map_err(error)?;
            }
            Ok(())
        })();
        self.replacing.set(false);
        replaced?;
        self.last = nodes.to_vec();
        self.tools = tools;
        Ok(())
    }

    fn listen(&self, element: &Element, node: &UiNode, tools: bool) -> Result<()> {
        let callbacks = js_sys::Array::new();
        let id = node.id;
        let input = self.input.clone();
        let send = move |action| {
            let mut input = input.borrow_mut();
            let queue = if tools { &mut input.tools } else { &mut input.events };
            queue.push(GameEvent::UiAction { node: id, action, value: 0.0 });
        };
        let click = if takes(node, ui_action::ACTIVATE) {
            Some(ui_action::ACTIVATE)
        } else {
            let action = if matches!(node.expanded, ui_state::TRUE | ui_state::MIXED) {
                ui_action::COLLAPSE
            } else {
                ui_action::EXPAND
            };
            takes(node, action).then_some(action)
        };
        for (event, action) in [
            ("focus", (node.focusable || takes(node, ui_action::FOCUS)).then_some(ui_action::FOCUS)),
            ("blur", (node.focusable || takes(node, ui_action::BLUR)).then_some(ui_action::BLUR)),
            ("click", click),
        ] {
            if let Some(action) = action {
                let send = send.clone();
                let replacing = self.replacing.clone();
                listen(element, &callbacks, event, move |event| {
                    if !replacing.get() {
                        event.stop_propagation();
                        send(action);
                    }
                })?;
            }
        }
        let increment = takes(node, ui_action::INCREMENT);
        let decrement = takes(node, ui_action::DECREMENT);
        if increment || decrement {
            listen(element, &callbacks, "keydown", move |event| {
                let Some(key) = event.dyn_ref::<KeyboardEvent>() else { return };
                let action = match key.key().as_str() {
                    "ArrowUp" | "ArrowRight" if increment => ui_action::INCREMENT,
                    "ArrowDown" | "ArrowLeft" if decrement => ui_action::DECREMENT,
                    _ => return,
                };
                event.prevent_default();
                event.stop_propagation();
                send(action);
            })?;
        }
        // Keep listeners with their DOM node so collecting it releases them.
        js_sys::Reflect::set(element, &"penActions".into(), &callbacks).map_err(error)?;
        Ok(())
    }
}

fn parent(index: usize, node: &UiNode) -> Option<usize> {
    (node.parent as usize).checked_sub(1).filter(|parent| *parent < index)
}

fn option(node: &UiNode) -> bool {
    node.role == ui_role::LISTITEM && node.focusable
}

fn takes(node: &UiNode, action: u32) -> bool {
    node.actions & (1 << action) != 0
}

fn listen(
    element: &Element,
    callbacks: &js_sys::Array,
    name: &str,
    callback: impl FnMut(Event) + 'static,
) -> Result<()> {
    let callback = Closure::<dyn FnMut(Event)>::new(callback);
    element.add_event_listener_with_callback(name, callback.as_ref().unchecked_ref()).map_err(error)?;
    callbacks.push(&callback.into_js_value());
    Ok(())
}
