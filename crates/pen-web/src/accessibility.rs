//! The protocol's interface as a screen-reader tree beside the canvas.
use crate::input::Input;
use crate::program::error;
use anyhow::Result;
use pen_protocol::{GameEvent, UiNode, ui_action, ui_live, ui_role, ui_state};
use pen_runtime::role_testing::AccessibilityTree;
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
    generation: Rc<Cell<u64>>,
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
        Ok(Self {
            root,
            last: Vec::new(),
            tools: false,
            replacing: Rc::default(),
            generation: Rc::default(),
            input,
            captions: None,
        })
    }

    /// The selected game language and its reading direction.
    pub fn set_locale(&self, tag: &str, rtl: bool) -> Result<()> {
        self.root.set_attribute("lang", tag).map_err(error)?;
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
        if self.tools == tools && a11y_nodes_equal(&self.last, nodes) {
            return Ok(());
        }
        let document = web_sys::window().unwrap().document().unwrap();
        let focused = document
            .active_element()
            .filter(|element| self.tools == tools && self.root.contains(Some(element)))
            .map(|element| element.id());
        let tree = AccessibilityTree::new(nodes);
        let modal = tree.active_modal();
        let generation = self.generation.get().wrapping_add(1);
        let mut elements: Vec<Option<Element>> = vec![None; nodes.len()];
        let mut top_level = Vec::new();
        let mut restore_focus = None;
        let mut game_focus = None;
        for (index, (node, info)) in nodes.iter().zip(tree.nodes()).enumerate() {
            if !info.in_tree {
                continue;
            }
            let element = document.create_element("div").map_err(error)?;
            if info.blocked {
                element.set_attribute("inert", "").map_err(error)?;
            }
            let role = if info.effective_role == "unknown" { "group" } else { info.effective_role };
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
            if info.disabled {
                element.set_attribute("aria-disabled", "true").map_err(error)?;
            }
            if modal == Some(index) {
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
            let focusable = tree.allows_action(node.id, ui_action::FOCUS);
            if focusable || modal == Some(index) {
                element.set_attribute("tabindex", "-1").map_err(error)?;
                if focused.as_ref().is_some_and(|id| element.id() == *id) {
                    restore_focus = Some(index);
                }
                if focusable && node.focused {
                    game_focus.get_or_insert(index);
                }
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
            if info.exposed() && !info.disabled {
                self.listen(&element, node, tools, &tree, generation)?;
            }
            // A nested active modal needs to sit outside its inert ancestors.
            if let Some(parent) = info.parent.filter(|_| modal != Some(index)) {
                elements[parent].as_ref().unwrap().append_child(&element).map_err(error)?;
            } else {
                top_level.push(index);
            }
            elements[index] = Some(element);
        }

        // Preserve an eligible focused control across redraws. If a modal
        // displaced it, focus the game's selection in the modal or the modal itself.
        let focus = if let Some(modal) = modal {
            restore_focus.filter(|&index| index != modal).or(game_focus).or(Some(modal))
        } else {
            restore_focus
        };
        self.replacing.set(true);
        self.generation.set(generation);
        let replaced: Result<()> = (|| {
            self.root.set_text_content(None);
            for index in top_level {
                self.root.append_child(elements[index].as_ref().unwrap()).map_err(error)?;
            }
            if let Some(element) =
                focus.and_then(|index| elements[index].as_ref()).and_then(|element| element.dyn_ref::<HtmlElement>())
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

    fn listen(
        &self,
        element: &Element,
        node: &UiNode,
        tools: bool,
        tree: &AccessibilityTree<'_>,
        generation: u64,
    ) -> Result<()> {
        let callbacks = js_sys::Array::new();
        let id = node.id;
        let input = self.input.clone();
        let current = self.generation.clone();
        let replacing = self.replacing.clone();
        let accepts = move |event: &Event| {
            current.get() == generation
                && !replacing.get()
                && event.target().and_then(|target| target.dyn_into::<Element>().ok()).is_some_and(|target| {
                    target.is_connected() && target.closest("[inert], [aria-disabled=true]").ok().flatten().is_none()
                })
        };
        let send = move |action| {
            let mut input = input.borrow_mut();
            let queue = if tools { &mut input.tools } else { &mut input.events };
            queue.push(GameEvent::UiAction { node: id, action, value: 0.0 });
        };
        let allowed = |action| tree.allows_action(id, action);
        let click = if allowed(ui_action::ACTIVATE) {
            Some(ui_action::ACTIVATE)
        } else {
            let action = if matches!(node.expanded, ui_state::TRUE | ui_state::MIXED) {
                ui_action::COLLAPSE
            } else {
                ui_action::EXPAND
            };
            allowed(action).then_some(action)
        };
        for (event, action) in [
            ("focus", allowed(ui_action::FOCUS).then_some(ui_action::FOCUS)),
            ("blur", allowed(ui_action::BLUR).then_some(ui_action::BLUR)),
            ("click", click),
        ] {
            if let Some(action) = action {
                let send = send.clone();
                let accepts = accepts.clone();
                listen(element, &callbacks, event, move |event| {
                    if accepts(&event) {
                        event.stop_propagation();
                        send(action);
                    }
                })?;
            }
        }
        let increment = allowed(ui_action::INCREMENT);
        let decrement = allowed(ui_action::DECREMENT);
        if increment || decrement {
            listen(element, &callbacks, "keydown", move |event| {
                if !accepts(&event) {
                    return;
                }
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

impl Drop for Accessibility {
    fn drop(&mut self) {
        self.generation.set(self.generation.get().wrapping_add(1));
        self.root.remove();
        if let Some((captions, _)) = &self.captions {
            captions.remove();
        }
    }
}

fn a11y_nodes_equal(a: &[UiNode], b: &[UiNode]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.id == y.id
                && x.parent == y.parent
                && x.role == y.role
                && x.focusable == y.focusable
                && x.focused == y.focused
                && x.disabled == y.disabled
                && x.hidden == y.hidden
                && x.modal == y.modal
                && x.checked == y.checked
                && x.expanded == y.expanded
                && x.selected == y.selected
                && x.live == y.live
                && x.value.to_bits() == y.value.to_bits()
                && x.minimum.to_bits() == y.minimum.to_bits()
                && x.maximum.to_bits() == y.maximum.to_bits()
                && x.actions == y.actions
                && x.text_length == y.text_length
                && x.name_length == y.name_length
                && x.description_length == y.description_length
                && x.text == y.text
        })
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
