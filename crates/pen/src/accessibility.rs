//! The game's interface for assistive technology: the nodes each frame
//! declares (`RenderFrame::ui`, gdev's packages/ui.blot) as an AccessKit
//! tree for the window's platform adapter (AT-SPI on Linux, UI Automation on
//! Windows, NSAccessibility on macOS), and what assistive technology asks of
//! them back to the game as `UiAction` input events.
//!
//! The game drives its own keyboard focus (Tab, the arrows, Enter; F10 moves
//! it into its interface) and marks the node that has it; the host only
//! mirrors that focus. Actions reach the game as input with its next frame,
//! so input recordings keep them and replays repeat them; scripts send them
//! with `ui=ID,ACTION[,VALUE]`.
//!
//! The game sends its whole interface every frame; the tree is built and
//! handed to the adapter only when it changes, and only while an assistive
//! technology listens. The dev tools' own tree (their program's) hangs from a graft
//! node under the root as a subtree with ids of its own (`OVERLAY_TREE`), and
//! has the focus while one of its widgets has the keyboard.
//!
//! PEN_A11Y_DUMP=FILE writes the tree as text to FILE (from the start of the
//! run) whenever it changes, in a window or headless.
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs::File;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use accesskit::{
    Action, ActionData, ActionHandler, ActionRequest, ActivationHandler, DeactivationHandler, Live, Node, NodeId, Rect,
    Role, Toggled, TreeId, TreeInfo, TreeUpdate, Uuid,
};
use anyhow::{Context, Result};
use pen_protocol::{GameEvent, UiNode, ui_action, ui_live, ui_role, ui_state};
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use crate::session::Session;

/// The root: the window.
const ROOT: NodeId = NodeId(0);
/// The dev tools' graft node, which their program's tree hangs from.
const GRAFT: NodeId = NodeId(1);
/// Game nodes' ids (1 to MAX_UI_ID) are offset by this, away from the
/// host's own nodes.
const GAME: u64 = 1 << 32;
/// The dev tools' tree: a subtree with ids of its own.
const OVERLAY_TREE: TreeId = TreeId(Uuid::from_u128(0x5d1c_7a2e_93b4_4c8f_a6e0_2f71_c0de_9a11));

/// The game's actions and AccessKit's, by the game's codes.
const ACTIONS: [(u32, Action); 8] = [
    (ui_action::FOCUS, Action::Focus),
    (ui_action::BLUR, Action::Blur),
    (ui_action::ACTIVATE, Action::Click),
    (ui_action::INCREMENT, Action::Increment),
    (ui_action::DECREMENT, Action::Decrement),
    (ui_action::SET_VALUE, Action::SetValue),
    (ui_action::EXPAND, Action::Expand),
    (ui_action::COLLAPSE, Action::Collapse),
];

/// What the tree is built from.
#[derive(Clone, Debug, PartialEq)]
struct Interface {
    /// The game's nodes, as the frame shown declares them.
    ui: Vec<UiNode>,
    /// Window pixels per frame pixel, across and down: frames can keep a size
    /// the window does not have, and the window shows them stretched.
    scale: [f64; 2],
    /// The dev tools show (their tree hangs from the graft node), and
    /// whether one of its widgets has the keyboard.
    overlay: Option<bool>,
}

fn game_node(id: u32) -> NodeId {
    NodeId(GAME + u64::from(id))
}

/// The game's id of a node of the tree, if it is the game's.
fn game_id(node: NodeId) -> Option<u32> {
    node.0.checked_sub(GAME).and_then(|id| u32::try_from(id).ok())
}

/// The place of a node's parent, if one comes before it; else it is at the top.
fn parent(index: usize, node: &UiNode) -> Option<usize> {
    (node.parent as usize).checked_sub(1).filter(|parent| *parent < index)
}

/// Which nodes the tree has: those not hidden nor under a node it leaves
/// out, the first with each id.
fn included(ui: &[UiNode]) -> Vec<bool> {
    let mut ids = HashSet::with_capacity(ui.len());
    let mut included: Vec<bool> = Vec::with_capacity(ui.len());
    for (index, node) in ui.iter().enumerate() {
        let kept = !node.hidden && parent(index, node).is_none_or(|parent| included[parent]) && ids.insert(node.id);
        included.push(kept);
    }
    included
}

/// Code points as text; one that is no character (or NUL, which D-Bus
/// strings cannot carry) becomes U+FFFD.
fn decode(points: &[f32]) -> String {
    points
        .iter()
        .map(|&point| {
            let character = pen_protocol::boundary::whole(point).and_then(char::from_u32);
            character.filter(|character| *character != '\0').unwrap_or(char::REPLACEMENT_CHARACTER)
        })
        .collect()
}

/// A node's name, description and value text.
fn texts(node: &UiNode) -> [String; 3] {
    let text = &node.text;
    let name = (node.name_length as usize).min(text.len());
    let description = name.saturating_add(node.description_length as usize).min(text.len());
    [decode(&text[..name]), decode(&text[name..description]), decode(&text[description..])]
}

/// `plain`: a group with nothing to say or do, which assistive technology
/// passes through to its children.
fn role(role: u32, plain: bool) -> Role {
    match role {
        ui_role::GROUP if plain => Role::GenericContainer,
        ui_role::GROUP => Role::Group,
        ui_role::TEXT => Role::Label,
        ui_role::HEADING => Role::Heading,
        ui_role::BUTTON => Role::Button,
        ui_role::CHECKBOX => Role::CheckBox,
        ui_role::SWITCH => Role::Switch,
        ui_role::SLIDER => Role::Slider,
        ui_role::SPINBUTTON => Role::SpinButton,
        ui_role::LIST => Role::List,
        ui_role::LISTITEM => Role::ListItem,
        ui_role::DIALOG => Role::Dialog,
        ui_role::MENU => Role::Menu,
        ui_role::MENUITEM => Role::MenuItem,
        ui_role::TABLIST => Role::TabList,
        ui_role::TAB => Role::Tab,
        ui_role::TABPANEL => Role::TabPanel,
        ui_role::TOOLBAR => Role::Toolbar,
        ui_role::IMAGE => Role::Image,
        ui_role::STATUS => Role::Status,
        ui_role::PROGRESSBAR => Role::ProgressIndicator,
        ui_role::RADIOGROUP => Role::RadioGroup,
        ui_role::RADIO => Role::RadioButton,
        ui_role::TEXTBOX => Role::TextInput,
        ui_role::LOG => Role::Log,
        _ => Role::Unknown,
    }
}

fn state(code: u32) -> Option<Toggled> {
    match code {
        ui_state::FALSE => Some(Toggled::False),
        ui_state::TRUE => Some(Toggled::True),
        ui_state::MIXED => Some(Toggled::Mixed),
        _ => None,
    }
}

/// Whether a node takes an action: it lists it, or the action is focus and
/// the node is focusable.
fn takes(node: &UiNode, action: u32) -> bool {
    let listed = 1u32.checked_shl(action).is_some_and(|bit| node.actions & bit != 0);
    listed || (action == ui_action::FOCUS && node.focusable)
}

/// Whether a node has a value in a range.
fn ranged(node: &UiNode) -> bool {
    matches!(node.role, ui_role::SLIDER | ui_role::SPINBUTTON | ui_role::PROGRESSBAR)
        || [ui_action::INCREMENT, ui_action::DECREMENT, ui_action::SET_VALUE].iter().any(|action| takes(node, *action))
}

/// Where a node is in the window, if it is drawn.
fn bounds(node: &UiNode, [across, down]: [f64; 2]) -> Option<Rect> {
    let [x, y, width, height] = [node.x, node.y, node.width, node.height].map(f64::from);
    let drawn = width > 0.0 && height > 0.0 && [x, y, width, height].iter().all(|value| value.is_finite());
    drawn.then(|| Rect::new(x * across, y * down, (x + width) * across, (y + height) * down))
}

fn node(ui: &UiNode, scale: [f64; 2]) -> Node {
    let [name, description, value] = texts(ui);
    let plain = name.is_empty() && description.is_empty() && !ui.focusable && ui.actions == 0;
    let role = role(ui.role, plain);
    let mut node = Node::new(role);
    if role == Role::Label {
        // A label's text is its value, in AccessKit's terms.
        let text = if name.is_empty() { value } else { name };
        if !text.is_empty() {
            node.set_value(text);
        }
    } else {
        if !name.is_empty() {
            node.set_label(name);
        }
        if !value.is_empty() {
            node.set_value(value);
        }
    }
    if !description.is_empty() {
        node.set_description(description);
    }
    if ranged(ui) {
        let finite = |value: f32| Some(f64::from(value)).filter(|value| value.is_finite());
        if let Some(value) = finite(ui.value) {
            node.set_numeric_value(value);
        }
        if let Some(minimum) = finite(ui.minimum) {
            node.set_min_numeric_value(minimum);
        }
        if let Some(maximum) = finite(ui.maximum) {
            node.set_max_numeric_value(maximum);
        }
        if let Some(step) = finite(ui.step).filter(|step| *step > 0.0) {
            node.set_numeric_value_step(step);
        }
    }
    if let Some(checked) = state(ui.checked) {
        node.set_toggled(checked);
    }
    if let Some(expanded) = state(ui.expanded) {
        node.set_expanded(expanded != Toggled::False);
    }
    if let Some(selected) = state(ui.selected) {
        node.set_selected(selected != Toggled::False);
    }
    if ui.disabled {
        node.set_disabled();
    }
    if ui.modal {
        node.set_modal();
    }
    match ui.live {
        ui_live::POLITE => node.set_live(Live::Polite),
        ui_live::ASSERTIVE => node.set_live(Live::Assertive),
        _ => {}
    }
    if let Some(bounds) = bounds(ui, scale) {
        node.set_bounds(bounds);
    }
    for (code, action) in ACTIONS {
        if takes(ui, code) {
            node.add_action(action);
        }
    }
    node
}

/// The tree an interface makes, as assistive technology should see it.
fn tree(interface: &Interface) -> TreeUpdate {
    build(interface, true)
}

/// `grafted`: the adapter has the overlay's subtree, so its graft node can
/// have the focus (an adapter fails on a focused graft without one).
fn build(interface: &Interface, grafted: bool) -> TreeUpdate {
    let ui = &interface.ui;
    let included = included(ui);
    let mut root = Node::new(Role::Window);
    // The window's title, as the game's pen.json or package names it.
    root.set_label(crate::identity().title.as_str());
    let mut nodes = Vec::with_capacity(ui.len() + 2);
    nodes.push((ROOT, Node::default()));
    // Where each node the tree has is in `nodes`.
    let mut places = vec![0; ui.len()];
    let mut focus = ROOT;
    // A list whose items are chosen, and those items, are a list box and its
    // options, as the browser host has them.
    let option = |game: &UiNode| game.role == ui_role::LISTITEM && game.focusable;
    let mut listbox = vec![false; ui.len()];
    for (index, game) in ui.iter().enumerate() {
        if let Some(parent) = parent(index, game).filter(|_| option(game)) {
            listbox[parent] = ui[parent].role == ui_role::LIST;
        }
    }
    for (index, game) in ui.iter().enumerate() {
        if !included[index] {
            continue;
        }
        let id = game_node(game.id);
        // A node the tree has is under a parent it has, or at the top.
        match parent(index, game) {
            Some(parent) => nodes[places[parent]].1.push_child(id),
            None => root.push_child(id),
        }
        if game.focused && focus == ROOT {
            focus = id;
        }
        places[index] = nodes.len();
        let mut made = node(game, interface.scale);
        if option(game) {
            made.set_role(Role::ListBoxOption);
        } else if listbox[index] {
            made.set_role(Role::ListBox);
        }
        nodes.push((id, made));
    }
    if let Some(focused) = interface.overlay {
        root.push_child(GRAFT);
        let mut graft = Node::new(Role::GenericContainer);
        graft.set_tree_id(OVERLAY_TREE);
        nodes.push((GRAFT, graft));
        if focused && grafted {
            focus = GRAFT;
        }
    }
    nodes[0].1 = root;
    TreeUpdate {
        nodes,
        tree: Some(TreeInfo {
            root: ROOT,
            toolkit_name: Some("pen".into()),
            toolkit_version: Some(env!("CARGO_PKG_VERSION").into()),
        }),
        tree_id: TreeId::ROOT,
        focus,
    }
}

/// The game's event for an assistive technology's request, when the request
/// is for a node of the game's tree that takes that action.
fn game_event(ui: &[UiNode], request: &ActionRequest) -> Option<GameEvent> {
    if request.target_tree != TreeId::ROOT {
        return None;
    }
    let id = game_id(request.target_node)?;
    let (action, _) = ACTIONS.into_iter().find(|(_, action)| *action == request.action)?;
    let value = match (action, &request.data) {
        (ui_action::SET_VALUE, Some(ActionData::NumericValue(value))) => crate::input::narrow(*value),
        (ui_action::SET_VALUE, Some(ActionData::Value(text))) => text.trim().parse().ok()?,
        (ui_action::SET_VALUE, _) => return None,
        _ => 0.0,
    };
    if !value.is_finite() {
        return None;
    }
    let included = included(ui);
    let (node, _) = ui.iter().zip(included).find(|(node, included)| *included && node.id == id)?;
    takes(node, action).then_some(GameEvent::UiAction { node: id, action, value })
}

/// The dev tools program's interface as a tree of its own, as their
/// subtree: its nodes under a group, the focus on its focused node.
fn program_tree(ui: &[UiNode]) -> TreeUpdate {
    let mut update = build(&Interface { ui: ui.to_vec(), scale: [1.0, 1.0], overlay: None }, false);
    if let Some((_, root)) = update.nodes.iter_mut().find(|(id, _)| *id == ROOT) {
        root.set_role(Role::Group);
        root.set_label("Dev tools");
    }
    overlay_subtree(update)
}

/// Whether one of the dev tools program's nodes has the keyboard.
fn program_focused(ui: &[UiNode]) -> bool {
    ui.iter().zip(included(ui)).any(|(node, included)| included && node.focused)
}

/// An assistive technology's request for one of the dev tools program's
/// nodes, as the UiAction event its next frame takes.
#[cfg(feature = "devtools")]
pub fn program_event(ui: &[UiNode], request: &ActionRequest) -> Option<GameEvent> {
    game_event(ui, request)
}

/// An assistive technology's request for the dev tools' subtree, as they
/// takes it: for its own (root) tree.
fn overlay_request(request: &ActionRequest) -> Option<ActionRequest> {
    (request.target_tree == OVERLAY_TREE).then(|| ActionRequest { target_tree: TreeId::ROOT, ..request.clone() })
}

/// The dev tools' tree as their subtree, its focus on one of its nodes.
fn overlay_subtree(mut update: TreeUpdate) -> TreeUpdate {
    update.tree_id = OVERLAY_TREE;
    if let Some(tree) = &update.tree
        && !update.nodes.iter().any(|(id, _)| *id == update.focus)
    {
        update.focus = tree.root;
    }
    update
}

/// The tree as text: a line per node, under its parent, with AccessKit's
/// roles, the game's ids (`#ID`) and the actions by the game's names (as
/// script steps take them). Numbers show at the game's precision.
fn describe(update: &TreeUpdate) -> String {
    let nodes: HashMap<NodeId, &Node> = update.nodes.iter().map(|(id, node)| (*id, node)).collect();
    let mut text = String::new();
    let mut stack = vec![(update.tree.as_ref().map_or(ROOT, |tree| tree.root), 0)];
    while let Some((id, depth)) = stack.pop() {
        let Some(node) = nodes.get(&id) else { continue };
        line(&mut text, id, node, depth, id == update.focus);
        stack.extend(node.children().iter().rev().map(|child| (*child, depth + 1)));
    }
    text
}

fn line(text: &mut String, id: NodeId, node: &Node, depth: usize, focused: bool) {
    let _ = write!(text, "{:indent$}{:?}", "", node.role(), indent = depth * 2);
    if let Some(game) = game_id(id) {
        let _ = write!(text, " #{game}");
    }
    if node.tree_id() == Some(OVERLAY_TREE) {
        text.push_str(" (the dev tools)");
    }
    if let Some(label) = node.label() {
        let _ = write!(text, " {label:?}");
    }
    if let Some(value) = node.value() {
        let _ = write!(text, " value={value:?}");
    }
    if let Some(description) = node.description() {
        let _ = write!(text, " description={description:?}");
    }
    if let Some(toggled) = node.toggled() {
        let toggled = match toggled {
            Toggled::False => "false",
            Toggled::True => "true",
            Toggled::Mixed => "mixed",
        };
        let _ = write!(text, " toggled={toggled}");
    }
    if let Some(expanded) = node.is_expanded() {
        let _ = write!(text, " expanded={expanded}");
    }
    if let Some(selected) = node.is_selected() {
        let _ = write!(text, " selected={selected}");
    }
    if node.is_disabled() {
        text.push_str(" disabled");
    }
    if node.is_modal() {
        text.push_str(" modal");
    }
    if let Some(live) = node.live() {
        let _ = write!(text, " live={}", format!("{live:?}").to_lowercase());
    }
    let number = crate::input::narrow;
    if let Some(value) = node.numeric_value() {
        let _ = write!(text, " number={}", number(value));
    }
    if node.min_numeric_value().is_some() || node.max_numeric_value().is_some() {
        let end = |value: Option<f64>| value.map_or(String::new(), |value| number(value).to_string());
        let _ = write!(text, " range={}..{}", end(node.min_numeric_value()), end(node.max_numeric_value()));
    }
    if let Some(step) = node.numeric_value_step() {
        let _ = write!(text, " step={}", number(step));
    }
    if let Some(bounds) = node.bounds() {
        let [x, y, width, height] = [bounds.x0, bounds.y0, bounds.width(), bounds.height()].map(number);
        let _ = write!(text, " at={x},{y} size={width}x{height}");
    }
    let actions: Vec<&str> = ACTIONS
        .iter()
        .filter(|(_, action)| node.supports_action(*action))
        .map(|(code, _)| ui_action::NAMES[*code as usize])
        .collect();
    if !actions.is_empty() {
        let _ = write!(text, " actions={}", actions.join(","));
    }
    if focused {
        text.push_str(" <- focus");
    }
    text.push('\n');
}

fn lock(shared: &Mutex<Shared>) -> MutexGuard<'_, Shared> {
    shared.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What the adapter's handlers share with the window's thread; they run on
/// other threads.
#[derive(Default)]
struct Shared {
    /// What the tree was last built from, for an assistive technology that
    /// starts now.
    latest: Option<Arc<Interface>>,
    /// An assistive technology started since the last update: the adapter
    /// has the tree its activation handler gave (or waits for a whole one),
    /// without the overlay's subtree.
    activated: bool,
    /// Requests the window's thread has yet to take.
    requests: Vec<ActionRequest>,
}

/// Gives the adapter the whole tree as it stands when an assistive
/// technology starts, so it has no placeholder.
struct Activation(Arc<Mutex<Shared>>);

impl ActivationHandler for Activation {
    fn request_initial_tree(&mut self) -> Option<TreeUpdate> {
        let latest = {
            let mut shared = lock(&self.0);
            shared.activated = true;
            shared.latest.clone()
        };
        latest.map(|interface| build(&interface, false))
    }
}

/// Keeps the requests for the window's thread.
struct Requests(Arc<Mutex<Shared>>);

impl ActionHandler for Requests {
    fn do_action(&mut self, request: ActionRequest) {
        lock(&self.0).requests.push(request);
    }
}

struct Deactivation;

impl DeactivationHandler for Deactivation {
    fn deactivate_accessibility(&mut self) {}
}

/// What the adapter has, as far as the window's thread knows.
#[derive(Default)]
struct Sent {
    /// The tree changed since the adapter last had it.
    stale: bool,
    /// The adapter has the overlay's subtree.
    grafted: bool,
    /// The focus of the adapter's tree.
    focus: Option<NodeId>,
}

/// Whether an assistive technology started since the last update (see
/// `Shared::activated`).
fn started(sent: &mut Sent, shared: &Mutex<Shared>) -> bool {
    let started = std::mem::take(&mut lock(shared).activated);
    if started {
        sent.grafted = false;
    }
    started
}

/// The whole tree, the focus off the overlay until the adapter has its subtree.
fn whole(sent: &mut Sent, shared: &Mutex<Shared>, interface: &Interface) -> TreeUpdate {
    started(sent, shared);
    if interface.overlay.is_none() {
        sent.grafted = false;
    }
    let update = build(interface, sent.grafted);
    sent.focus = Some(update.focus);
    update
}

/// Brings the adapter's tree up to date: the whole tree when it changed,
/// then the dev tools' subtree when it changed (`program`, their interface,
/// and whether it did) or the adapter lacks it, then the focus on it if it
/// was not the adapter's to have. `push` hands the adapter an update, made
/// only while an assistive technology listens; each one first sees whether
/// one has started since the last, and is then the whole tree.
fn sync(
    sent: &mut Sent,
    shared: &Mutex<Shared>,
    interface: &Interface,
    program: Option<(&[UiNode], bool)>,
    push: &mut dyn FnMut(&mut dyn FnMut() -> TreeUpdate),
) {
    if std::mem::take(&mut sent.stale) {
        push(&mut || whole(sent, shared, interface));
    }
    if let Some((ui, changed)) = program
        && interface.overlay.is_some()
        && (changed || !sent.grafted)
    {
        push(&mut || {
            if started(sent, shared) {
                return whole(sent, shared, interface);
            }
            sent.grafted = true;
            program_tree(ui)
        });
    }
    if interface.overlay == Some(true) && sent.grafted && sent.focus != Some(GRAFT) {
        push(&mut || {
            if started(sent, shared) {
                return whole(sent, shared, interface);
            }
            sent.focus = Some(GRAFT);
            TreeUpdate { nodes: Vec::new(), tree: None, tree_id: TreeId::ROOT, focus: GRAFT }
        });
    }
}

/// PEN_A11Y_DUMP: the tree as text, written to a file as it changes.
struct Dump {
    path: PathBuf,
    file: Option<File>,
    /// The tree last written.
    last: Option<TreeUpdate>,
}

impl Dump {
    fn from_env() -> Result<Option<Dump>> {
        let Some(path) = std::env::var_os("PEN_A11Y_DUMP").filter(|path| !path.is_empty()).map(PathBuf::from) else {
            return Ok(None);
        };
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(&path).with_context(|| format!("writing PEN_A11Y_DUMP {}", path.display()))?;
        Ok(Some(Dump { path, file: Some(file), last: None }))
    }

    /// `frame`: the frame shown's position, as `--trace` numbers frames.
    fn write(&mut self, frame: u64, tree: TreeUpdate) {
        if self.last.as_ref() == Some(&tree) {
            return;
        }
        let text = format!("frame {frame}\n{}\n", describe(&tree));
        if let Some(file) = &mut self.file
            && let Err(error) = file.write_all(text.as_bytes())
        {
            eprintln!("PEN_A11Y_DUMP: writing {}: {error}", self.path.display());
            self.file = None;
        }
        self.last = Some(tree);
    }
}

/// The game's interface (and the dev tools') for assistive technology:
/// the window's AccessKit adapter, and the PEN_A11Y_DUMP file.
pub struct Accessibility {
    /// None headless.
    adapter: Option<accesskit_winit::Adapter>,
    shared: Arc<Mutex<Shared>>,
    /// What the tree was last built from.
    current: Option<Arc<Interface>>,
    /// The dev tools' interface as last seen, their subtree's.
    program: Option<Vec<UiNode>>,
    sent: Sent,
    dump: Option<Dump>,
}

impl Accessibility {
    fn new(dump: Option<Dump>) -> Accessibility {
        Accessibility {
            adapter: None,
            shared: Arc::default(),
            current: None,
            program: None,
            sent: Sent::default(),
            dump,
        }
    }

    /// For a window not yet shown: platform adapters must exist before it
    /// first shows.
    pub fn windowed(event_loop: &ActiveEventLoop, window: &Window) -> Result<Accessibility> {
        let mut accessibility = Accessibility::new(Dump::from_env()?);
        let shared = &accessibility.shared;
        accessibility.adapter = Some(accesskit_winit::Adapter::with_direct_handlers(
            event_loop,
            window,
            Activation(shared.clone()),
            Requests(shared.clone()),
            Deactivation,
        ));
        Ok(accessibility)
    }

    /// Headless: only PEN_A11Y_DUMP's file, if asked for.
    pub fn headless() -> Result<Option<Accessibility>> {
        Ok(Dump::from_env()?.map(|dump| Accessibility::new(Some(dump))))
    }

    /// Every window event, before anything else handles it.
    pub fn window_event(&mut self, window: &Window, event: &WindowEvent) {
        if let Some(adapter) = &mut self.adapter {
            adapter.process_event(window, event);
        }
    }

    /// After a step: the tree from the frame shown, to the adapter and the
    /// dump when it changed (the dev tools' subtree, to the adapter when
    /// theirs did); and what assistive technology asked for since, to the
    /// game's next frame (or the dev tools' next).
    pub fn frame(&mut self, session: &mut Session, window: Option<&Window>) {
        let previous = self.current.as_deref();
        let scale = match window.map(Window::inner_size) {
            Some(size) if size.width > 0 && size.height > 0 => {
                let (width, height) = session.gpu.size;
                [f64::from(size.width) / f64::from(width.max(1)), f64::from(size.height) / f64::from(height.max(1))]
            }
            // Minimized, the window shows nothing.
            Some(_) => previous.map_or([1.0, 1.0], |previous| previous.scale),
            None => [1.0, 1.0],
        };
        let program = session.overlay_interface();
        let overlay = program.map(program_focused);
        let program_changed = program != self.program.as_deref();
        if program_changed {
            self.program = program.map(<[UiNode]>::to_vec);
        }
        let ui = session.interface();
        let changed =
            previous.is_none_or(|previous| previous.scale != scale || previous.overlay != overlay || previous.ui != ui);
        if changed {
            let interface = Arc::new(Interface { ui: ui.to_vec(), scale, overlay });
            lock(&self.shared).latest = Some(interface.clone());
            if let Some(dump) = &mut self.dump {
                dump.write(session.timeline().frame.saturating_sub(1), tree(&interface));
            }
            self.current = Some(interface);
            self.sent.stale = true;
        }
        let requests = {
            let mut shared = lock(&self.shared);
            self.sent.stale |= shared.activated;
            std::mem::take(&mut shared.requests)
        };
        if let (Some(adapter), Some(interface)) = (&mut self.adapter, &self.current) {
            let program = self.program.as_deref().map(|ui| (ui, program_changed));
            sync(&mut self.sent, &self.shared, interface, program, &mut |update| adapter.update_if_active(update));
        }
        for request in requests {
            if let Some(request) = overlay_request(&request) {
                session.overlay_accesskit_action(request);
            } else if let Some(event) = game_event(session.interface(), &request) {
                session.input.enqueue(event);
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, reason = "short test texts")]
mod tests {
    use super::*;

    /// A node with `text` as its name (and `parent` the place plus one).
    fn named(id: u32, parent: u32, role: u32, name: &str) -> UiNode {
        let text: Vec<f32> = name.chars().map(|character| character as u32 as f32).collect();
        UiNode {
            id,
            parent,
            role,
            name_length: text.len() as u32,
            text_length: text.len() as u32,
            text,
            ..UiNode::default()
        }
    }

    fn interface(ui: Vec<UiNode>) -> Interface {
        Interface { ui, scale: [1.0, 1.0], overlay: None }
    }

    fn find(update: &TreeUpdate, id: NodeId) -> &Node {
        &update.nodes.iter().find(|(node, _)| *node == id).unwrap_or_else(|| panic!("no node {id:?}")).1
    }

    fn children(update: &TreeUpdate, id: NodeId) -> Vec<NodeId> {
        find(update, id).children().to_vec()
    }

    /// A menu: a modal dialog with a heading, a focused button, a slider
    /// with value text, a hidden group with a child, and a checkbox.
    fn menu() -> Vec<UiNode> {
        let mut dialog = named(10, 0, ui_role::DIALOG, "Paused");
        (dialog.modal, dialog.x, dialog.y, dialog.width, dialog.height) = (true, 100.0, 50.0, 400.0, 300.0);
        let mut resume = named(12, 1, ui_role::BUTTON, "Resume");
        (resume.focusable, resume.focused, resume.actions) = (true, true, 1 << ui_action::ACTIVATE);
        (resume.x, resume.y, resume.width, resume.height) = (120.0, 80.0, 200.0, 40.0);
        let mut volume = named(13, 1, ui_role::SLIDER, "Volume");
        volume.text.extend("80%".chars().map(|character| character as u32 as f32));
        volume.text_length = volume.text.len() as u32;
        (volume.value, volume.minimum, volume.maximum, volume.step) = (0.8, 0.0, 1.0, 0.1);
        volume.actions = 1 << ui_action::INCREMENT | 1 << ui_action::DECREMENT | 1 << ui_action::SET_VALUE;
        volume.focusable = true;
        let mut hidden = named(14, 1, ui_role::GROUP, "Debug");
        hidden.hidden = true;
        let under_hidden = named(15, 5, ui_role::BUTTON, "Secret");
        let mut music = named(16, 1, ui_role::CHECKBOX, "Music");
        (music.checked, music.actions) = (ui_state::TRUE, 1 << ui_action::ACTIVATE);
        vec![dialog, named(11, 1, ui_role::HEADING, "Paused"), resume, volume, hidden, under_hidden, music]
    }

    #[test]
    fn maps_every_role() {
        let expected = [
            Role::Group,
            Role::Label,
            Role::Heading,
            Role::Button,
            Role::CheckBox,
            Role::Switch,
            Role::Slider,
            Role::SpinButton,
            Role::List,
            Role::ListItem,
            Role::Dialog,
            Role::Menu,
            Role::MenuItem,
            Role::TabList,
            Role::Tab,
            Role::TabPanel,
            Role::Toolbar,
            Role::Image,
            Role::Status,
            Role::ProgressIndicator,
            Role::RadioGroup,
            Role::RadioButton,
            Role::TextInput,
            Role::Log,
        ];
        assert_eq!(expected.len(), ui_role::NAMES.len());
        let ui: Vec<UiNode> = (0..expected.len() as u32).map(|code| named(code + 1, 0, code, "x")).collect();
        let update = tree(&interface(ui));
        for (code, role) in expected.iter().enumerate() {
            assert_eq!(find(&update, game_node(code as u32 + 1)).role(), *role, "{}", ui_role::NAMES[code]);
        }
        // A group with nothing to say or do is passed through.
        let update = tree(&interface(vec![named(1, 0, ui_role::GROUP, "")]));
        assert_eq!(find(&update, game_node(1)).role(), Role::GenericContainer);
        let root = find(&update, ROOT);
        assert_eq!((root.role(), root.label()), (Role::Window, Some(crate::identity().title.as_str())));
    }

    #[test]
    fn splits_names_descriptions_and_value_text() {
        let mut node = named(1, 0, ui_role::BUTTON, "Play");
        node.text.extend("Start a new game".chars().map(|character| character as u32 as f32));
        node.description_length = 16;
        node.text.extend("on".chars().map(|character| character as u32 as f32));
        node.text_length = node.text.len() as u32;
        // Not characters: a surrogate, a fraction, a negative, NaN, NUL.
        let mut odd = named(2, 0, ui_role::TEXT, "");
        odd.text = vec![0xD800 as f32, 66.5, -1.0, f32::NAN, 0.0, 'é' as u32 as f32];
        (odd.text_length, odd.name_length) = (6, 6);
        // Lengths beyond the text end with it.
        let mut short = named(3, 0, ui_role::BUTTON, "Hi");
        (short.name_length, short.description_length) = (5, 9);
        let update = tree(&interface(vec![node, odd, short]));
        let play = find(&update, game_node(1));
        assert_eq!(play.label(), Some("Play"));
        assert_eq!(play.description(), Some("Start a new game"));
        assert_eq!(play.value(), Some("on"));
        // A label's text is its value.
        let odd = find(&update, game_node(2));
        assert_eq!((odd.label(), odd.value()), (None, Some("\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}é")));
        let short = find(&update, game_node(3));
        assert_eq!((short.label(), short.description(), short.value()), (Some("Hi"), None, None));
    }

    #[test]
    fn carries_states_ranges_and_actions() {
        let update = tree(&interface(menu()));
        let dialog = find(&update, game_node(10));
        assert!(dialog.is_modal());
        assert!(ACTIONS.iter().all(|(_, action)| !dialog.supports_action(*action)));
        let resume = find(&update, game_node(12));
        assert!(resume.supports_action(Action::Focus) && resume.supports_action(Action::Click));
        assert!(!resume.supports_action(Action::Increment));
        assert_eq!(resume.numeric_value(), None);
        let volume = find(&update, game_node(13));
        assert_eq!((volume.label(), volume.value()), (Some("Volume"), Some("80%")));
        assert_eq!(volume.numeric_value(), Some(f64::from(0.8f32)));
        assert_eq!((volume.min_numeric_value(), volume.max_numeric_value()), (Some(0.0), Some(1.0)));
        assert_eq!(volume.numeric_value_step(), Some(f64::from(0.1f32)));
        for action in [Action::Focus, Action::Increment, Action::Decrement, Action::SetValue] {
            assert!(volume.supports_action(action), "{action:?}");
        }
        assert!(!volume.supports_action(Action::Click));
        assert_eq!(find(&update, game_node(16)).toggled(), Some(Toggled::True));

        let mut node = named(1, 0, ui_role::LISTITEM, "Save 1");
        (node.checked, node.expanded, node.selected, node.disabled, node.live) =
            (ui_state::MIXED, ui_state::FALSE, ui_state::TRUE, true, ui_live::ASSERTIVE);
        node.actions = 1 << ui_action::EXPAND | 1 << ui_action::COLLAPSE | 1 << ui_action::BLUR;
        let mut status = named(2, 0, ui_role::STATUS, "Saved");
        status.live = ui_live::POLITE;
        let update = tree(&interface(vec![node, status]));
        let node = find(&update, game_node(1));
        assert_eq!(node.toggled(), Some(Toggled::Mixed));
        assert_eq!((node.is_expanded(), node.is_selected()), (Some(false), Some(true)));
        assert!(node.is_disabled());
        assert_eq!(node.live(), Some(Live::Assertive));
        for action in [Action::Expand, Action::Collapse, Action::Blur] {
            assert!(node.supports_action(action), "{action:?}");
        }
        assert!(!node.supports_action(Action::Focus));
        let status = find(&update, game_node(2));
        assert_eq!((status.live(), status.toggled(), status.is_expanded()), (Some(Live::Polite), None, None));
    }

    #[test]
    fn scales_bounds_to_the_window() {
        let mut interface = interface(menu());
        interface.scale = [2.0, 0.5];
        let update = tree(&interface);
        assert_eq!(find(&update, game_node(12)).bounds(), Some(Rect::new(240.0, 40.0, 640.0, 60.0)));
        // Not drawn: no bounds.
        assert_eq!(find(&update, game_node(11)).bounds(), None);
    }

    #[test]
    fn leaves_out_hidden_subtrees_and_repeated_ids() {
        let mut ui = menu();
        let mut again = named(12, 1, ui_role::BUTTON, "Again");
        again.focused = true;
        ui.push(again);
        let update = tree(&interface(ui));
        assert_eq!(children(&update, ROOT), [game_node(10)]);
        assert_eq!(children(&update, game_node(10)), [game_node(11), game_node(12), game_node(13), game_node(16)]);
        assert!(!update.nodes.iter().any(|(id, _)| [game_node(14), game_node(15)].contains(id)));
        assert_eq!(find(&update, game_node(12)).label(), Some("Resume"));
        assert_eq!(update.nodes.len(), 1 + 5);
    }

    #[test]
    fn puts_nodes_without_a_parent_before_them_at_the_top() {
        let ui = vec![
            named(1, 2, ui_role::GROUP, "Points ahead"),
            named(2, 0, ui_role::LIST, "Top"),
            named(3, 2, ui_role::LISTITEM, "Child"),
            named(4, 4, ui_role::LISTITEM, "Its own parent"),
            named(5, 99, ui_role::LISTITEM, "Beyond the end"),
        ];
        let update = tree(&interface(ui));
        assert_eq!(children(&update, ROOT), [game_node(1), game_node(2), game_node(4), game_node(5)]);
        assert_eq!(children(&update, game_node(2)), [game_node(3)]);
    }

    #[test]
    fn mirrors_the_games_focus() {
        let update = tree(&interface(menu()));
        assert_eq!(update.focus, game_node(12));
        let mut ui = menu();
        ui[2].focused = false;
        assert_eq!(tree(&interface(ui.clone())).focus, ROOT);
        // A hidden node's focus is no one's.
        ui[5].focused = true;
        assert_eq!(tree(&interface(ui)).focus, ROOT);
        // The overlay's widget with the keyboard has it over the game's.
        let mut interface = interface(menu());
        interface.overlay = Some(true);
        let update = tree(&interface);
        assert_eq!(update.focus, GRAFT);
        assert_eq!(children(&update, ROOT), [game_node(10), GRAFT]);
        assert_eq!(find(&update, GRAFT).tree_id(), Some(OVERLAY_TREE));
        // Until the adapter has the overlay's subtree, the game keeps it.
        assert_eq!(build(&interface, false).focus, game_node(12));
        interface.overlay = Some(false);
        assert_eq!(tree(&interface).focus, game_node(12));
    }

    fn request(node: NodeId, action: Action, data: Option<ActionData>) -> ActionRequest {
        ActionRequest { action, target_tree: TreeId::ROOT, target_node: node, data }
    }

    #[test]
    fn turns_requests_into_ui_actions() {
        let ui = menu();
        let event = |node, action, data| game_event(&ui, &request(node, action, data));
        let action = |node, action, value| Some(GameEvent::UiAction { node, action, value });
        assert_eq!(event(game_node(12), Action::Click, None), action(12, ui_action::ACTIVATE, 0.0));
        assert_eq!(event(game_node(12), Action::Focus, None), action(12, ui_action::FOCUS, 0.0));
        assert_eq!(event(game_node(13), Action::Increment, None), action(13, ui_action::INCREMENT, 0.0));
        assert_eq!(event(game_node(13), Action::Decrement, None), action(13, ui_action::DECREMENT, 0.0));
        assert_eq!(
            event(game_node(13), Action::SetValue, Some(ActionData::NumericValue(0.25))),
            action(13, ui_action::SET_VALUE, 0.25)
        );
        assert_eq!(
            event(game_node(13), Action::SetValue, Some(ActionData::Value(" 0.5 ".into()))),
            action(13, ui_action::SET_VALUE, 0.5)
        );
        // Not without a value, nor a value that is no number.
        assert_eq!(event(game_node(13), Action::SetValue, None), None);
        assert_eq!(event(game_node(13), Action::SetValue, Some(ActionData::NumericValue(f64::NAN))), None);
        assert_eq!(event(game_node(13), Action::SetValue, Some(ActionData::Value("loud".into()))), None);
        // Not an action the node does not take, nor one the game has no code for.
        assert_eq!(event(game_node(12), Action::Increment, None), None);
        assert_eq!(event(game_node(10), Action::Focus, None), None);
        assert_eq!(event(game_node(12), Action::ShowContextMenu, None), None);
        // Not for a node the tree leaves out, one it has not, or the host's own.
        assert_eq!(event(game_node(15), Action::Click, None), None);
        assert_eq!(event(game_node(99), Action::Click, None), None);
        assert_eq!(event(ROOT, Action::Focus, None), None);
        assert_eq!(event(GRAFT, Action::Focus, None), None);
        // The overlay's go to it, as requests for its own tree.
        let mut overlaid = request(NodeId(7), Action::Click, None);
        overlaid.target_tree = OVERLAY_TREE;
        assert_eq!(game_event(&ui, &overlaid), None);
        assert_eq!(overlay_request(&overlaid), Some(request(NodeId(7), Action::Click, None)));
        assert_eq!(overlay_request(&request(game_node(12), Action::Click, None)), None);
    }

    #[test]
    fn describes_the_tree() {
        let mut interface = interface(menu());
        interface.overlay = Some(false);
        let text = describe(&tree(&interface));
        assert_eq!(
            text,
            format!("Window \"{}\"\n", crate::identity().title)
                + "\x20 Dialog #10 \"Paused\" modal at=100,50 size=400x300\n\
             \x20   Heading #11 \"Paused\"\n\
             \x20   Button #12 \"Resume\" at=120,80 size=200x40 actions=focus,activate <- focus\n\
             \x20   Slider #13 \"Volume\" value=\"80%\" number=0.8 range=0..1 step=0.1 \
             actions=focus,increment,decrement,set_value\n\
             \x20   CheckBox #16 \"Music\" toggled=true actions=activate\n\
             \x20 GenericContainer (the dev tools)\n"
        );
    }

    /// An adapter as far as AccessKit's own consumer goes: what the
    /// platform adapters keep, which fails (panics) on an inconsistent tree
    /// or update.
    #[derive(Default)]
    struct Adapter {
        tree: Option<accesskit_consumer::Tree>,
        /// An assistive technology listens.
        active: bool,
        updates: usize,
    }

    struct Ignore;

    impl accesskit_consumer::TreeChangeHandler for Ignore {
        fn node_added(&mut self, _: &accesskit_consumer::NodeRef) {}
        fn node_updated(&mut self, _: &accesskit_consumer::NodeRef, _: &accesskit_consumer::NodeRef) {}
        fn focus_moved(&mut self, _: Option<&accesskit_consumer::NodeRef>, _: Option<&accesskit_consumer::NodeRef>) {}
        fn node_removed(&mut self, _: &accesskit_consumer::NodeRef) {}
    }

    impl Adapter {
        /// As platform adapters do when an assistive technology starts.
        fn activate(&mut self, shared: &Arc<Mutex<Shared>>) {
            self.active = true;
            let initial = Activation(shared.clone()).request_initial_tree();
            self.tree = initial.map(|update| accesskit_consumer::Tree::new(update, true));
        }

        fn push(&mut self, update: &mut dyn FnMut() -> TreeUpdate) {
            if !self.active {
                return;
            }
            self.updates += 1;
            let update = update();
            match &mut self.tree {
                Some(tree) => tree.update_and_process_changes(update, &mut Ignore),
                // Activated without a tree: the first update is a whole one.
                None => self.tree = Some(accesskit_consumer::Tree::new(update, true)),
            }
        }

        /// The focused node, through the grafts: its id and tree.
        fn focus(&self) -> (NodeId, TreeId) {
            self.tree.as_ref().unwrap().state().focus().unwrap().locate()
        }
    }

    /// The dev tools' interface: a console whose text box has the keyboard
    /// when `focused`.
    fn console(focused: bool) -> Vec<UiNode> {
        let mut input = named(2, 1, ui_role::TEXTBOX, "Command");
        (input.focusable, input.focused) = (true, focused);
        vec![named(1, 0, ui_role::DIALOG, "Console"), input]
    }

    /// Where the console's text box is in the adapter's tree.
    const INPUT: (NodeId, TreeId) = (NodeId(GAME + 2), OVERLAY_TREE);

    /// The window thread's side of a frame in which the game's interface
    /// changed, as `Accessibility::frame` does it: `program`, the dev
    /// tools' interface, and whether it changed.
    fn frame(
        sent: &mut Sent,
        shared: &Arc<Mutex<Shared>>,
        adapter: &mut Adapter,
        interface: &Interface,
        program: Option<(&[UiNode], bool)>,
    ) {
        sent.stale = true;
        lock(shared).latest = Some(Arc::new(interface.clone()));
        sync(sent, shared, interface, program, &mut |update| adapter.push(update));
    }

    /// A frame in which the game's interface stayed the same: the tree goes
    /// again only to an assistive technology that started since.
    fn same(
        sent: &mut Sent,
        shared: &Arc<Mutex<Shared>>,
        adapter: &mut Adapter,
        interface: &Interface,
        program: Option<(&[UiNode], bool)>,
    ) {
        sent.stale |= lock(shared).activated;
        sync(sent, shared, interface, program, &mut |update| adapter.push(update));
    }

    #[test]
    fn platform_adapters_take_the_tree_and_the_overlay_graft() {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let mut sent = Sent::default();
        let mut adapter = Adapter::default();
        let mut game = interface(menu());
        // Nothing is made while no assistive technology listens.
        frame(&mut sent, &shared, &mut adapter, &game, None);
        assert_eq!(adapter.updates, 0);
        // One starts: its tree is the latest, at once.
        adapter.activate(&shared);
        assert_eq!(adapter.focus(), (game_node(12), TreeId::ROOT));
        same(&mut sent, &shared, &mut adapter, &game, None);
        // The game moves its focus.
        game.ui[2].focused = false;
        game.ui[3].focused = true;
        frame(&mut sent, &shared, &mut adapter, &game, None);
        assert_eq!(adapter.focus(), (game_node(13), TreeId::ROOT));
        // The dev tools show, their console with the keyboard: the graft,
        // their subtree, then the focus on it, in one frame.
        let (typing, idle) = (console(true), console(false));
        game.overlay = Some(true);
        frame(&mut sent, &shared, &mut adapter, &game, Some((&typing, true)));
        assert_eq!(adapter.focus(), INPUT);
        assert!(sent.grafted);
        // Their tree comes again only when it changes, even as the game's does.
        let updates = adapter.updates;
        same(&mut sent, &shared, &mut adapter, &game, Some((&typing, false)));
        assert_eq!(adapter.updates, updates);
        frame(&mut sent, &shared, &mut adapter, &game, Some((&typing, false)));
        assert_eq!((adapter.updates, adapter.focus()), (updates + 1, INPUT));
        // The console lets go of the keyboard: the game's focus again.
        game.overlay = Some(false);
        frame(&mut sent, &shared, &mut adapter, &game, Some((&idle, true)));
        assert_eq!(adapter.focus(), (game_node(13), TreeId::ROOT));
        // An assistive technology starts again while the dev tools have the
        // keyboard: its tree comes first, then their subtree (unchanged, but
        // the adapter lacks it) and the focus.
        game.overlay = Some(true);
        frame(&mut sent, &shared, &mut adapter, &game, Some((&typing, true)));
        adapter.activate(&shared);
        assert_eq!(adapter.focus(), (game_node(13), TreeId::ROOT));
        same(&mut sent, &shared, &mut adapter, &game, Some((&typing, false)));
        assert_eq!(adapter.focus(), INPUT);
        // They hide: their graft and subtree go.
        game.overlay = None;
        frame(&mut sent, &shared, &mut adapter, &game, None);
        let state = adapter.tree.as_ref().unwrap().state();
        assert_eq!(state.subtree_root(OVERLAY_TREE), None);
        assert_eq!(adapter.focus(), (game_node(13), TreeId::ROOT));
        // And show again, unchanged: the adapter gets their subtree again.
        game.overlay = Some(false);
        frame(&mut sent, &shared, &mut adapter, &game, Some((&idle, false)));
        assert!(adapter.tree.as_ref().unwrap().state().subtree_root(OVERLAY_TREE).is_some());
    }

    #[test]
    fn says_whether_the_dev_tools_have_the_keyboard() {
        assert!(program_focused(&console(true)));
        assert!(!program_focused(&console(false)));
        let mut hidden = console(true);
        hidden[0].hidden = true;
        assert!(!program_focused(&hidden));
        // Their subtree has ids of its own, the focus on the text box.
        let update = program_tree(&console(true));
        assert_eq!((update.tree_id, update.focus), (OVERLAY_TREE, NodeId(GAME + 2)));
        assert_eq!(find(&update, ROOT).label(), Some("Dev tools"));
    }

    #[test]
    fn an_activation_before_the_first_tree_waits_for_a_whole_one() {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let (mut sent, mut adapter) = (Sent::default(), Adapter::default());
        adapter.activate(&shared);
        assert!(adapter.tree.is_none());
        let mut game = interface(menu());
        game.overlay = Some(true);
        // The dev tools' subtree may not come first.
        frame(&mut sent, &shared, &mut adapter, &game, Some((&console(true), true)));
        assert_eq!(adapter.focus(), INPUT);
    }

    #[test]
    fn unchanged_interfaces_are_cheap_to_compare() {
        // A menu of 64 nodes with 24 code points of text each.
        let ui: Vec<UiNode> = (1..=64).map(|id| named(id, 0, ui_role::BUTTON, "Some button with a name")).collect();
        let (previous, now) = (ui.clone(), ui);
        let started = std::time::Instant::now();
        let rounds = 1000;
        for _ in 0..rounds {
            assert!(std::hint::black_box(&previous) == std::hint::black_box(&now));
        }
        let micros = started.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
        eprintln!("comparing 64 unchanged nodes: {micros:.2} µs");
    }
}
