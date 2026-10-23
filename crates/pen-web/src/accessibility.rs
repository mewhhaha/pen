//! The protocol's interface as a screen-reader tree beside the canvas.
use crate::input::Input;
use crate::program::error;
use anyhow::Result;
use pen_protocol::{GameEvent, UiNode, ui_action, ui_role, ui_state};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::{JsCast, closure::Closure};
use web_sys::Element;

/// The interface shown last, and whose it is: the dev tools' (their actions
/// go to them) or the game's.
pub struct Accessibility {
    root: Element,
    last: Vec<UiNode>,
    tools: bool,
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
        Ok(Self { root, last: Vec::new(), tools: false, input, captions: None })
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
        self.last = nodes.to_vec();
        self.tools = tools;
        self.root.set_text_content(None);
        let document = web_sys::window().unwrap().document().unwrap();
        let mut elements = Vec::with_capacity(nodes.len());
        for node in nodes {
            let element = document.create_element("div").map_err(error)?;
            if node.hidden {
                element.set_attribute("aria-hidden", "true").map_err(error)?;
            }
            let role = ui_role::NAMES.get(node.role as usize).copied().unwrap_or("group");
            element.set_attribute("role", role).map_err(error)?;
            element.set_attribute("id", &format!("pen-ui-{}", node.id)).map_err(error)?;
            let name = pen_runtime::services::text(&node.text[..(node.name_length as usize).min(node.text.len())]);
            element.set_attribute("aria-label", &name).map_err(error)?;
            let description_at = node.name_length as usize;
            let description_end = (description_at + node.description_length as usize).min(node.text.len());
            if let Some(values) = node.text.get(description_at..description_end) {
                element.set_attribute("aria-description", &pen_runtime::services::text(values)).map_err(error)?;
            }
            if node.disabled {
                element.set_attribute("aria-disabled", "true").map_err(error)?;
            }
            if node.modal {
                element.set_attribute("aria-modal", "true").map_err(error)?;
            }
            if node.live > 0 {
                element.set_attribute("aria-live", "polite").map_err(error)?;
            }
            if node.checked != ui_state::NONE {
                let checked = match node.checked {
                    ui_state::TRUE => "true",
                    ui_state::MIXED => "mixed",
                    _ => "false",
                };
                element.set_attribute("aria-checked", checked).map_err(error)?;
            }
            if node.focusable {
                element.set_attribute("tabindex", "-1").map_err(error)?;
            }
            if node.role == ui_role::SLIDER || node.role == ui_role::SPINBUTTON {
                for (key, value) in
                    [("aria-valuenow", node.value), ("aria-valuemin", node.minimum), ("aria-valuemax", node.maximum)]
                {
                    element.set_attribute(key, &value.to_string()).map_err(error)?;
                }
            }
            let id = node.id;
            let input = self.input.clone();
            let callback = Closure::<dyn FnMut(web_sys::Event)>::new(move |_| {
                let mut input = input.borrow_mut();
                let queue = if tools { &mut input.tools } else { &mut input.events };
                queue.push(GameEvent::UiAction { node: id, action: ui_action::ACTIVATE, value: 0.0 });
            });
            element.add_event_listener_with_callback("click", callback.as_ref().unchecked_ref()).map_err(error)?;
            // The removed DOM node drops the listener, and therefore its Rust closure.
            // Store it on the element so the function remains live without leaking.
            js_sys::Reflect::set(&element, &"penAction".into(), &callback.into_js_value()).map_err(error)?;
            let parent: &Element = if node.parent > 0 && (node.parent as usize) <= elements.len() {
                &elements[node.parent as usize - 1]
            } else {
                &self.root
            };
            parent.append_child(&element).map_err(error)?;
            elements.push(element);
        }
        Ok(())
    }
}
