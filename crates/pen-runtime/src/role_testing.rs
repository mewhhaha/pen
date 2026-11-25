//! Query accessibility frames by role and build user actions against the current frame.

use std::collections::HashMap;
use std::fmt::{self, Write as _};
use std::ops::Deref;
use std::sync::Arc;

use pen_protocol::{GameEvent, UiNode, boundary::whole, ui_action, ui_role, ui_state};

use crate::audit::{self, Finding};

/// A protocol role code or a case-insensitive ARIA role name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoleMatcher {
    Code(u32),
    Name(String),
}

/// Converts a protocol role constant or an ARIA role name into a query matcher.
pub trait IntoRole {
    fn into_role(self) -> RoleMatcher;
}

impl IntoRole for RoleMatcher {
    fn into_role(self) -> RoleMatcher {
        self
    }
}

impl IntoRole for u32 {
    fn into_role(self) -> RoleMatcher {
        RoleMatcher::Code(self)
    }
}

impl IntoRole for &str {
    fn into_role(self) -> RoleMatcher {
        RoleMatcher::Name(self.to_owned())
    }
}

impl IntoRole for String {
    fn into_role(self) -> RoleMatcher {
        RoleMatcher::Name(self)
    }
}

impl IntoRole for &String {
    fn into_role(self) -> RoleMatcher {
        self.as_str().into_role()
    }
}

fn role_name(code: u32) -> &'static str {
    ui_role::NAMES.get(code as usize).copied().unwrap_or("unknown")
}

impl RoleMatcher {
    fn matches(&self, code: u32, effective: &str) -> bool {
        match self {
            Self::Code(role) => *role == code,
            Self::Name(name) => {
                let name = if name.eq_ignore_ascii_case("img") {
                    "image"
                } else if name.eq_ignore_ascii_case("label") {
                    "text"
                } else {
                    name.as_str()
                };
                effective.eq_ignore_ascii_case(name) || role_name(code).eq_ignore_ascii_case(name)
            }
        }
    }
}

/// An owned snapshot. The original protocol fields (including the role code,
/// action bits, state codes and range) are also accessible through deref.
#[derive(Clone, Debug, PartialEq)]
pub struct RoleNode {
    pub node: UiNode,
    pub name: String,
    pub description: String,
    pub value_text: String,
    pub role_name: &'static str,
    /// The ARIA role after any listbox or option promotion.
    pub effective_role: &'static str,
    /// Zero-based position in the original frame.
    pub index: usize,
    /// Zero-based position of a valid preceding parent in the original frame.
    pub parent_index: Option<usize>,
    /// Membership in the tree before applying the active modal's boundary.
    pub in_tree: bool,
    /// Excluded from the tree by the node's own visibility or an ancestor.
    pub hidden: bool,
    /// Includes a disabled ancestor.
    pub disabled: bool,
    pub blocked_by_modal: bool,
    /// Bounds in frame pixels, as [x, y, width, height].
    pub rect: [f32; 4],
}

impl Deref for RoleNode {
    type Target = UiNode;

    fn deref(&self) -> &UiNode {
        &self.node
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryError {
    NotFound { query: String, tree: String },
    MultipleMatches { query: String, count: usize, tree: String },
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound { query, tree } => write!(f, "No nodes matched {query}\n\n{tree}"),
            Self::MultipleMatches { query, count, tree } => {
                write!(f, "{count} nodes matched {query}\n\n{tree}")
            }
        }
    }
}

impl std::error::Error for QueryError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActionError {
    NotFound { id: u32 },
    Hidden { id: u32 },
    Disabled { id: u32 },
    BlockedByModal { id: u32 },
    Unsupported { id: u32, action: u32 },
    NonFiniteValue { id: u32 },
}

impl fmt::Display for ActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NotFound { id } => write!(f, "Node #{id} is outside the current screen or no longer exists"),
            Self::Hidden { id } => write!(f, "Node #{id} is hidden"),
            Self::Disabled { id } => write!(f, "Node #{id} is disabled"),
            Self::BlockedByModal { id } => write!(f, "Node #{id} is blocked by the active modal dialog"),
            Self::Unsupported { id, action } => {
                let name = ui_action::NAMES.get(action as usize).copied().unwrap_or("the requested action");
                write!(f, "Node #{id} does not support {name}")
            }
            Self::NonFiniteValue { id } => write!(f, "Node #{id} requires a finite value"),
        }
    }
}

impl std::error::Error for ActionError {}

/// Effective accessibility state for a node in frame order.
pub struct NodeInfo {
    pub parent: Option<usize>,
    pub in_tree: bool,
    pub disabled: bool,
    pub blocked: bool,
    pub effective_role: &'static str,
}

impl NodeInfo {
    /// Whether the node is visible and belongs to the active modal scope.
    pub fn exposed(&self) -> bool {
        self.in_tree && !self.blocked
    }
}

/// Visibility, modal scope and action eligibility shared by the host projections.
pub struct AccessibilityTree<'a> {
    ui: &'a [UiNode],
    nodes: Vec<NodeInfo>,
    indices: HashMap<u32, usize>,
    modal: Option<usize>,
}

impl<'a> AccessibilityTree<'a> {
    pub fn new(ui: &'a [UiNode]) -> Self {
        let mut nodes: Vec<NodeInfo> = Vec::with_capacity(ui.len());
        let mut indices = HashMap::with_capacity(ui.len());
        let mut modal = None;
        for (index, node) in ui.iter().enumerate() {
            // The wire parent is one-based; invalid and forward parents become roots.
            let parent = (node.parent as usize).checked_sub(1).filter(|&parent| parent < index);
            let first = *indices.entry(node.id).or_insert(index) == index;
            let in_tree = first && !node.hidden && parent.is_none_or(|parent| nodes[parent].in_tree);
            let disabled = node.disabled || parent.is_some_and(|parent| nodes[parent].disabled);
            if in_tree && node.role == ui_role::DIALOG && node.modal {
                modal = Some(index);
            }
            nodes.push(NodeInfo { parent, in_tree, disabled, blocked: false, effective_role: role_name(node.role) });
        }
        // The last visible modal in frame order wins, including any nested modal.
        if let Some(modal) = modal {
            for index in 0..nodes.len() {
                nodes[index].blocked = index != modal && nodes[index].parent.is_none_or(|parent| nodes[parent].blocked);
            }
        }
        for (index, node) in ui.iter().enumerate() {
            if nodes[index].exposed()
                && node.role == ui_role::LISTITEM
                && node.focusable
                && let Some(parent) = nodes[index].parent
                && nodes[parent].exposed()
                && ui[parent].role == ui_role::LIST
            {
                nodes[parent].effective_role = "listbox";
                nodes[index].effective_role = "option";
            }
        }
        Self { ui, nodes, indices, modal }
    }

    pub fn nodes(&self) -> &[NodeInfo] {
        &self.nodes
    }

    /// The active modal's frame index.
    pub fn active_modal(&self) -> Option<usize> {
        self.modal
    }

    pub fn allows_action(&self, id: u32, action: u32) -> bool {
        self.check_action(id, action).is_ok()
    }

    fn check_action(&self, id: u32, action: u32) -> Result<(), ActionError> {
        let &index = self.indices.get(&id).ok_or(ActionError::NotFound { id })?;
        let info = &self.nodes[index];
        let node = &self.ui[index];
        if !info.in_tree {
            return Err(ActionError::Hidden { id });
        }
        if info.disabled {
            return Err(ActionError::Disabled { id });
        }
        if info.blocked {
            return Err(ActionError::BlockedByModal { id });
        }
        let listed = 1u32.checked_shl(action).is_some_and(|bit| node.actions & bit != 0);
        if listed || (node.focusable && matches!(action, ui_action::FOCUS | ui_action::BLUR)) {
            Ok(())
        } else {
            Err(ActionError::Unsupported { id, action })
        }
    }
}

/// A borrowed accessibility frame. Queries return owned snapshots in frame order.
#[derive(Clone)]
pub struct RoleScreen<'a> {
    ui: &'a [UiNode],
    tree: Arc<AccessibilityTree<'a>>,
    scope: Option<Arc<[bool]>>,
}

impl<'a> RoleScreen<'a> {
    pub fn new(ui: &'a [UiNode]) -> Self {
        Self { ui, tree: Arc::new(AccessibilityTree::new(ui)), scope: None }
    }

    fn in_scope(&self, index: usize) -> bool {
        self.scope.as_ref().is_none_or(|scope| scope[index])
    }

    /// Restricts queries to strict descendants of the container in this screen.
    /// A missing or out-of-scope container yields an empty screen.
    pub fn within(&self, container_id: u32) -> Self {
        let container = self.tree.indices.get(&container_id).copied().filter(|&index| self.in_scope(index));
        let mut scope = vec![false; self.ui.len()];
        if let Some(container) = container {
            for index in container + 1..self.ui.len() {
                scope[index] = self.tree.nodes[index].parent.is_some_and(|parent| parent == container || scope[parent]);
            }
        }
        Self { ui: self.ui, tree: self.tree.clone(), scope: Some(scope.into()) }
    }

    /// Restricts queries to strict descendants of the node's ID in this screen.
    pub fn within_node(&self, node: &RoleNode) -> Self {
        self.within(node.id)
    }

    pub fn by_role(&self, role: impl IntoRole) -> RoleQuery<'a> {
        RoleQuery { screen: self.clone(), role: Some(role.into_role()), filters: Filters::default() }
    }

    pub fn get_by_role(&self, role: impl IntoRole) -> Result<RoleNode, QueryError> {
        self.by_role(role).get()
    }

    pub fn query_by_role(&self, role: impl IntoRole) -> Result<Option<RoleNode>, QueryError> {
        self.by_role(role).query()
    }

    pub fn get_all_by_role(&self, role: impl IntoRole) -> Result<Vec<RoleNode>, QueryError> {
        self.by_role(role).get_all()
    }

    pub fn query_all_by_role(&self, role: impl IntoRole) -> Vec<RoleNode> {
        self.by_role(role).query_all()
    }

    pub fn get_by_role_and_name(&self, role: impl IntoRole, name: &str) -> Result<RoleNode, QueryError> {
        self.by_role(role).name(name).get()
    }

    /// Finds an exactly named node of any role by its decoded accessible name.
    pub fn get_by_label_text(&self, label: &str) -> Result<RoleNode, QueryError> {
        RoleQuery { screen: self.clone(), role: None, filters: Filters::default() }.name(label).get()
    }

    fn snapshot(&self, index: usize) -> RoleNode {
        let node = &self.ui[index];
        let info = &self.tree.nodes[index];
        let [name, description, value_text] = texts(node);
        RoleNode {
            node: node.clone(),
            name,
            description,
            value_text,
            role_name: role_name(node.role),
            effective_role: info.effective_role,
            index,
            parent_index: info.parent,
            in_tree: info.in_tree,
            hidden: !info.in_tree,
            disabled: info.disabled,
            blocked_by_modal: info.blocked,
            rect: [node.x, node.y, node.width, node.height],
        }
    }

    fn searchable(&self, index: usize, include_hidden: bool) -> bool {
        let info = &self.tree.nodes[index];
        self.in_scope(index) && self.tree.indices[&self.ui[index].id] == index && (include_hidden || info.exposed())
    }

    fn indices(&self, include_hidden: bool) -> impl Iterator<Item = usize> + '_ {
        (0..self.ui.len()).filter(move |&index| self.searchable(index, include_hidden))
    }

    /// Prints the accessible tree in this scope with state, bounds, and focus.
    pub fn debug_tree(&self) -> String {
        self.display_tree(false)
    }

    fn display_tree(&self, include_hidden: bool) -> String {
        let mut text = String::from("Accessible tree");
        if let Some(modal) = self.tree.modal {
            let _ = write!(text, " (active modal #{})", self.ui[modal].id);
        }
        text.push('\n');

        let mut roots = Vec::new();
        let mut children = vec![None; self.ui.len()];
        for index in self.indices(include_hidden) {
            let parent = self.tree.nodes[index].parent.filter(|&parent| self.searchable(parent, include_hidden));
            if let Some(parent) = parent {
                children[parent].get_or_insert_with(Vec::new).push(index);
            } else {
                roots.push(index);
            }
        }
        if roots.is_empty() {
            text.push_str("  (empty)\n");
        }
        let mut stack: Vec<_> = roots.into_iter().rev().map(|index| (index, 0)).collect();
        while let Some((index, depth)) = stack.pop() {
            let node = self.snapshot(index);
            let _ = write!(
                text,
                "{:indent$}{} #{} {:?} {:?}",
                "",
                node.effective_role,
                node.id,
                node.name,
                node.rect,
                indent = 2 * depth
            );
            for (key, value) in [("description", &node.description), ("value_text", &node.value_text)] {
                if !value.is_empty() {
                    let _ = write!(text, " {key}={value:?}");
                }
            }
            for (key, state) in [("checked", node.checked), ("expanded", node.expanded), ("selected", node.selected)] {
                if state != ui_state::NONE {
                    let value = ui_state::NAMES.get(state as usize).copied().unwrap_or("unknown");
                    let _ = write!(text, " {key}={value}");
                }
            }
            if matches!(node.role, ui_role::SLIDER | ui_role::SPINBUTTON | ui_role::PROGRESSBAR)
                || node.actions
                    & ((1 << ui_action::SET_VALUE) | (1 << ui_action::INCREMENT) | (1 << ui_action::DECREMENT))
                    != 0
            {
                let _ =
                    write!(text, " value={} range={}..={} step={}", node.value, node.minimum, node.maximum, node.step);
            }
            for (yes, marker) in [
                (node.disabled, "disabled"),
                (node.hidden, "hidden"),
                (node.blocked_by_modal, "blocked by modal"),
                (node.modal && node.role == ui_role::DIALOG, "modal"),
            ] {
                if yes {
                    let _ = write!(text, " [{marker}]");
                }
            }
            if node.focused {
                text.push_str(" <- focus");
            }
            text.push('\n');
            if let Some(children) = &children[index] {
                stack.extend(children.iter().rev().map(|&child| (child, depth + 1)));
            }
        }
        text
    }

    pub fn activate(&self, node: &RoleNode) -> Result<GameEvent, ActionError> {
        self.action(node, ui_action::ACTIVATE, 0.0)
    }

    pub fn click(&self, node: &RoleNode) -> Result<GameEvent, ActionError> {
        self.activate(node)
    }

    pub fn focus(&self, node: &RoleNode) -> Result<GameEvent, ActionError> {
        self.action(node, ui_action::FOCUS, 0.0)
    }

    pub fn blur(&self, node: &RoleNode) -> Result<GameEvent, ActionError> {
        self.action(node, ui_action::BLUR, 0.0)
    }

    pub fn increment(&self, node: &RoleNode) -> Result<GameEvent, ActionError> {
        self.action(node, ui_action::INCREMENT, 0.0)
    }

    pub fn decrement(&self, node: &RoleNode) -> Result<GameEvent, ActionError> {
        self.action(node, ui_action::DECREMENT, 0.0)
    }

    pub fn set_value(&self, node: &RoleNode, value: f32) -> Result<GameEvent, ActionError> {
        self.action(node, ui_action::SET_VALUE, value)
    }

    pub fn expand(&self, node: &RoleNode) -> Result<GameEvent, ActionError> {
        self.action(node, ui_action::EXPAND, 0.0)
    }

    pub fn collapse(&self, node: &RoleNode) -> Result<GameEvent, ActionError> {
        self.action(node, ui_action::COLLAPSE, 0.0)
    }

    fn action(&self, node: &RoleNode, action: u32, value: f32) -> Result<GameEvent, ActionError> {
        let id = node.id;
        if !self.tree.indices.get(&id).is_some_and(|&index| self.in_scope(index)) {
            return Err(ActionError::NotFound { id });
        }
        self.tree.check_action(id, action)?;
        if !value.is_finite() {
            return Err(ActionError::NonFiniteValue { id });
        }
        Ok(GameEvent::UiAction { node: id, action, value })
    }

    /// Audits the entire underlying frame without a rendered picture.
    pub fn audit(&self) -> Vec<Finding> {
        audit::audit(self.ui, None)
    }

    /// Returns all findings and the tree when the frame fails the audit.
    pub fn assert_audit_clean(&self) -> Result<(), String> {
        let findings = self.audit();
        if findings.is_empty() {
            Ok(())
        } else {
            Err(format!("Accessibility audit failed\n{}\n{}", audit::report(&findings), self.debug_tree()))
        }
    }
}

fn texts(node: &UiNode) -> [String; 3] {
    let text = &node.text;
    let name_end = (node.name_length as usize).min(text.len());
    let description_end = name_end.saturating_add(node.description_length as usize).min(text.len());
    [&text[..name_end], &text[name_end..description_end], &text[description_end..]].map(|points| {
        points
            .iter()
            .map(|&point| {
                whole(point).and_then(char::from_u32).filter(|&c| c != '\0').unwrap_or(char::REPLACEMENT_CHARACTER)
            })
            .collect()
    })
}

#[derive(Debug, Default)]
struct Filters {
    name: Option<String>,
    name_contains: Option<String>,
    case_insensitive: bool,
    description: Option<String>,
    value_text: Option<String>,
    checked: Option<u32>,
    selected: Option<u32>,
    expanded: Option<u32>,
    disabled: Option<bool>,
    focused: Option<bool>,
    include_hidden: bool,
}

fn boolean_state(value: bool) -> u32 {
    if value { ui_state::TRUE } else { ui_state::FALSE }
}

/// A fluent role query. State filters require the specified state; a node
/// without a checked, selected or expanded state does not match `false`.
#[must_use]
pub struct RoleQuery<'a> {
    screen: RoleScreen<'a>,
    role: Option<RoleMatcher>,
    filters: Filters,
}

impl RoleQuery<'_> {
    pub fn name(mut self, name: &str) -> Self {
        self.filters.name = Some(name.to_owned());
        self
    }

    pub fn name_contains(mut self, name: &str) -> Self {
        self.filters.name_contains = Some(name.to_owned());
        self
    }

    /// Compares names by their Unicode lowercase forms.
    pub fn case_insensitive(mut self) -> Self {
        self.filters.case_insensitive = true;
        self
    }

    pub fn description(mut self, description: &str) -> Self {
        self.filters.description = Some(description.to_owned());
        self
    }

    pub fn value_text(mut self, value_text: &str) -> Self {
        self.filters.value_text = Some(value_text.to_owned());
        self
    }

    pub fn checked(self, checked: bool) -> Self {
        self.checked_state(boolean_state(checked))
    }

    pub fn checked_state(mut self, checked: u32) -> Self {
        self.filters.checked = Some(checked);
        self
    }

    pub fn selected(mut self, selected: bool) -> Self {
        self.filters.selected = Some(boolean_state(selected));
        self
    }

    pub fn expanded(mut self, expanded: bool) -> Self {
        self.filters.expanded = Some(boolean_state(expanded));
        self
    }

    pub fn disabled(mut self, disabled: bool) -> Self {
        self.filters.disabled = Some(disabled);
        self
    }

    pub fn focused(mut self, focused: bool) -> Self {
        self.filters.focused = Some(focused);
        self
    }

    /// Also searches hidden nodes and nodes outside the active modal.
    pub fn include_hidden(mut self, include: bool) -> Self {
        self.filters.include_hidden = include;
        self
    }

    pub fn get(&self) -> Result<RoleNode, QueryError> {
        self.query()?.ok_or_else(|| self.error(0))
    }

    pub fn query(&self) -> Result<Option<RoleNode>, QueryError> {
        let mut matches = self.matches();
        let first = matches.next();
        let remaining = matches.count();
        if remaining == 0 { Ok(first.map(|index| self.screen.snapshot(index))) } else { Err(self.error(remaining + 1)) }
    }

    pub fn get_all(&self) -> Result<Vec<RoleNode>, QueryError> {
        let matches = self.query_all();
        if matches.is_empty() { Err(self.error(0)) } else { Ok(matches) }
    }

    pub fn query_all(&self) -> Vec<RoleNode> {
        self.matches().map(|index| self.screen.snapshot(index)).collect()
    }

    fn matches(&self) -> impl Iterator<Item = usize> + '_ {
        let filters = &self.filters;
        let lower = |value: &str| {
            if filters.case_insensitive { value.to_lowercase() } else { value.to_owned() }
        };
        let name = filters.name.as_deref().map(lower);
        let contains = filters.name_contains.as_deref().map(lower);
        self.screen.indices(filters.include_hidden).filter(move |&index| {
            let node = &self.screen.ui[index];
            let info = &self.screen.tree.nodes[index];
            if !(self.role.as_ref().is_none_or(|role| role.matches(node.role, info.effective_role))
                && filters.checked.is_none_or(|state| state == node.checked)
                && filters.selected.is_none_or(|state| state == node.selected)
                && filters.expanded.is_none_or(|state| state == node.expanded)
                && filters.disabled.is_none_or(|state| state == info.disabled)
                && filters.focused.is_none_or(|state| state == node.focused))
            {
                return false;
            }
            if name.is_none() && contains.is_none() && filters.description.is_none() && filters.value_text.is_none() {
                return true;
            }
            let [node_name, description, value_text] = texts(node);
            let node_name = if filters.case_insensitive { node_name.to_lowercase() } else { node_name };
            (self.role.is_some() || !node_name.is_empty())
                && name.as_ref().is_none_or(|name| *name == node_name)
                && contains.as_ref().is_none_or(|contains| node_name.contains(contains))
                && filters.description.as_ref().is_none_or(|value| *value == description)
                && filters.value_text.as_ref().is_none_or(|value| *value == value_text)
        })
    }

    fn error(&self, count: usize) -> QueryError {
        let filters = &self.filters;
        let mut query = match &self.role {
            Some(RoleMatcher::Code(code)) => format!("role {:?} ({code})", role_name(*code)),
            Some(RoleMatcher::Name(name)) => format!("role {name:?}"),
            None => "label text".into(),
        };
        for (key, value) in [
            ("name", &filters.name),
            ("name containing", &filters.name_contains),
            ("description", &filters.description),
            ("value text", &filters.value_text),
        ] {
            if let Some(value) = value {
                let _ = write!(query, " with {key} {value:?}");
            }
        }
        for (key, value) in
            [("checked", filters.checked), ("selected", filters.selected), ("expanded", filters.expanded)]
        {
            if let Some(value) = value {
                let name = ui_state::NAMES.get(value as usize).copied().unwrap_or("unknown");
                let _ = write!(query, " {key}={name}");
            }
        }
        for (key, value) in [("disabled", filters.disabled), ("focused", filters.focused)] {
            if let Some(value) = value {
                let _ = write!(query, " {key}={value}");
            }
        }
        if filters.case_insensitive {
            query.push_str(" (case-insensitive name)");
        }
        let tree = self.screen.display_tree(filters.include_hidden);
        if count == 0 {
            QueryError::NotFound { query, tree }
        } else {
            QueryError::MultipleMatches { query, count, tree }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_protocol::audit_finding;

    fn node(id: u32, parent: u32, role: u32, name: &str) -> UiNode {
        let text: Vec<_> = name.chars().map(|c| u32::from(c) as f32).collect();
        let len = u32::try_from(text.len()).unwrap();
        UiNode {
            id,
            parent,
            role,
            text,
            name_length: len,
            text_length: len,
            width: 48.0,
            height: 32.0,
            ..Default::default()
        }
    }

    #[test]
    fn matches_protocol_roles_names_and_aliases() {
        let mut ui: Vec<_> = ui_role::NAMES
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let role = u32::try_from(i).unwrap();
                node(role + 1, 0, role, name)
            })
            .collect();
        ui[ui_role::BUTTON as usize] = node(4, 0, ui_role::BUTTON, "Édit Profile");
        let screen = RoleScreen::new(&ui);
        for (i, name) in ui_role::NAMES.iter().enumerate() {
            let code = u32::try_from(i).unwrap();
            assert_eq!(screen.get_by_role(name.to_ascii_uppercase()).unwrap().role, code);
            assert_eq!(screen.get_by_role(code).unwrap().role, code);
        }
        assert_eq!(screen.get_by_role("ImG").unwrap().role, ui_role::IMAGE);
        assert_eq!(screen.get_by_role("LaBeL").unwrap().role, ui_role::TEXT);
        assert_eq!(screen.get_by_label_text("Édit Profile").unwrap().id, 4);
        assert_eq!(screen.get_by_label_text("heading").unwrap().role, ui_role::HEADING);
        assert_eq!(screen.get_by_role_and_name("BuTtOn", "Édit Profile").unwrap().id, 4);
        assert_eq!(screen.by_role("button").name("édit profile").query().unwrap(), None);
        assert_eq!(screen.by_role("button").name("édit profile").case_insensitive().get().unwrap().id, 4);
        assert_eq!(screen.by_role("button").name_contains("dit PRO").case_insensitive().get().unwrap().id, 4);
        assert_eq!(screen.by_role("button").name_contains("Profile").get().unwrap().id, 4);
        assert_eq!(screen.query_by_role("invented").unwrap(), None);
    }

    #[test]
    fn reports_missing_and_ambiguous_matches_with_the_scoped_tree() {
        let ui = [
            node(1, 0, ui_role::GROUP, "Controls"),
            node(2, 1, ui_role::BUTTON, "Save"),
            node(3, 1, ui_role::BUTTON, "Save as"),
            node(4, 0, ui_role::BUTTON, "Outside"),
        ];
        let screen = RoleScreen::new(&ui).within(1);
        assert_eq!(screen.query_by_role("link").unwrap(), None);
        assert!(screen.query_all_by_role("link").is_empty());
        for error in [screen.get_by_role("link").unwrap_err(), screen.get_all_by_role("link").unwrap_err()] {
            assert!(matches!(error, QueryError::NotFound { .. }));
            let message = error.to_string();
            assert!(
                message.contains("role \"link\"") && message.contains("Accessible tree") && message.contains("Save as")
            );
            assert!(!message.contains("Outside"));
        }
        let query = screen.by_role("button").name_contains("Save");
        let error = query.get().unwrap_err();
        assert!(matches!(error, QueryError::MultipleMatches { count: 2, .. }));
        assert_eq!(query.query().unwrap_err(), error);
        assert!(error.to_string().contains("2 nodes matched role \"button\" with name containing \"Save\""));
        assert!(error.to_string().contains(&screen.debug_tree()));
        assert_eq!(query.get_all().unwrap().iter().map(|node| node.id).collect::<Vec<_>>(), [2, 3]);
    }

    #[test]
    fn scopes_to_descendants_with_preceding_parents_and_stable_ids() {
        let mut hidden = node(20, 0, ui_role::GROUP, "Hidden");
        hidden.hidden = true;
        let ui = [
            node(10, 0, ui_role::GROUP, "Container"),
            node(11, 1, ui_role::GROUP, "Nested"),
            node(12, 2, ui_role::BUTTON, "Inside"),
            node(13, 4, ui_role::BUTTON, "Invalid parent"),
            hidden,
            node(21, 5, ui_role::BUTTON, "Hidden descendant"),
            node(12, 0, ui_role::BUTTON, "Duplicate"),
            node(20, 0, ui_role::GROUP, "Duplicate hidden"),
        ];
        let screen = RoleScreen::new(&ui);
        let container = screen.get_by_label_text("Container").unwrap();
        let scoped = screen.within_node(&container);
        assert_eq!(scoped.get_by_role("group").unwrap().id, 11);
        let inside = scoped.within(11).get_by_role("button").unwrap();
        assert_eq!((inside.id, inside.index, inside.parent_index), (12, 2, Some(1)));
        assert!(scoped.within(10).query_all_by_role("button").is_empty());
        assert!(screen.within(999).query_all_by_role("button").is_empty());
        assert_eq!(screen.get_by_label_text("Invalid parent").unwrap().parent_index, None);
        let all = screen.by_role("button").include_hidden(true).get_all().unwrap();
        assert_eq!(all.iter().map(|node| node.id).collect::<Vec<_>>(), [12, 13, 21]);
        assert!(all[2].hidden && !all[2].in_tree);
        assert!(screen.by_role("group").include_hidden(true).name("Duplicate hidden").query().unwrap().is_none());
        assert_eq!(screen.click(&all[2]), Err(ActionError::Hidden { id: 21 }));
    }

    #[test]
    fn promotes_lists_and_focusable_items_while_retaining_the_base_roles() {
        let mut option = node(2, 1, ui_role::LISTITEM, "Easy");
        option.focusable = true;
        let ui = [
            node(1, 0, ui_role::LIST, "Difficulty"),
            option,
            node(3, 1, ui_role::LISTITEM, "Help"),
            node(4, 0, ui_role::LIST, "News"),
            node(5, 4, ui_role::LISTITEM, "Release"),
        ];
        let screen = RoleScreen::new(&ui);
        let listbox = screen.get_by_role("ListBox").unwrap();
        assert_eq!((listbox.role_name, listbox.effective_role), ("list", "listbox"));
        assert_eq!(listbox, screen.get_by_role_and_name(ui_role::LIST, "Difficulty").unwrap());
        assert_eq!(screen.get_all_by_role("list").unwrap().len(), 2);
        let option = screen.get_by_role("OPTION").unwrap();
        assert_eq!((option.role_name, option.effective_role), ("listitem", "option"));
        assert_eq!(screen.get_by_role_and_name("listitem", "Easy").unwrap(), option);
        assert_eq!(screen.within_node(&listbox).get_all_by_role("listitem").unwrap().len(), 2);
        assert_eq!(screen.get_by_label_text("Release").unwrap().effective_role, "listitem");
    }

    #[test]
    fn promotes_only_exposed_focusable_children_of_lists() {
        let item = |id, parent, name| UiNode { focusable: true, ..node(id, parent, ui_role::LISTITEM, name) };
        let ui = [
            node(1, 0, ui_role::LIST, "List"),
            UiNode { hidden: true, ..item(2, 1, "Hidden") },
            item(3, 0, "Orphan"),
            node(4, 0, ui_role::GROUP, "Group"),
            item(5, 4, "Under group"),
        ];
        let screen = RoleScreen::new(&ui);
        assert_eq!(screen.get_by_role("list").unwrap().effective_role, "list");
        assert!(screen.by_role("option").include_hidden(true).query_all().is_empty());
        for name in ["Orphan", "Under group"] {
            assert_eq!(screen.get_by_label_text(name).unwrap().effective_role, "listitem");
        }
    }

    #[test]
    fn shared_action_eligibility_checks_ancestors_and_modal_scope() {
        let button = |id, parent| UiNode {
            focusable: true,
            actions: 1 << ui_action::ACTIVATE,
            ..node(id, parent, ui_role::BUTTON, "Button")
        };
        let ui = [
            button(1, 0),
            UiNode { modal: true, ..node(2, 0, ui_role::DIALOG, "Modal") },
            UiNode { hidden: true, ..node(3, 2, ui_role::GROUP, "Hidden") },
            button(4, 3),
            UiNode { disabled: true, ..node(5, 2, ui_role::GROUP, "Disabled") },
            button(6, 5),
            button(7, 2),
        ];
        let tree = AccessibilityTree::new(&ui);
        assert_eq!(tree.active_modal(), Some(1));
        for action in [ui_action::ACTIVATE, ui_action::FOCUS, ui_action::BLUR] {
            for id in [1, 4, 6] {
                assert!(!tree.allows_action(id, action), "node {id}, action {action}");
            }
            assert!(tree.allows_action(7, action));
        }
        assert!(!tree.allows_action(7, ui_action::INCREMENT));
        assert!(!tree.allows_action(7, u32::MAX));
    }

    #[test]
    fn filters_states_and_decoded_text_and_exposes_the_owned_snapshot() {
        let mut mixed = node(4, 1, ui_role::CHECKBOX, "Some");
        mixed.description_length = 4;
        mixed.text.extend("HintHalf".chars().map(|c| u32::from(c) as f32));
        mixed.text_length = u32::try_from(mixed.text.len()).unwrap();
        mixed.checked = ui_state::MIXED;
        mixed.expanded = ui_state::FALSE;
        mixed.selected = ui_state::TRUE;
        mixed.focused = true;
        mixed.value = 0.5;
        mixed.maximum = 1.0;
        mixed.step = 0.25;
        let ui = [
            UiNode { disabled: true, ..node(1, 0, ui_role::GROUP, "Controls") },
            UiNode { checked: ui_state::TRUE, expanded: ui_state::TRUE, ..node(2, 0, ui_role::CHECKBOX, "All") },
            UiNode { checked: ui_state::FALSE, selected: ui_state::FALSE, ..node(3, 0, ui_role::CHECKBOX, "None") },
            mixed,
            node(5, 0, ui_role::CHECKBOX, "Unset"),
        ];
        let screen = RoleScreen::new(&ui);
        let boxes = || screen.by_role("checkbox");
        assert_eq!(boxes().checked(true).expanded(true).get().unwrap().id, 2);
        assert_eq!(boxes().checked(false).selected(false).disabled(false).focused(false).get().unwrap().id, 3);
        let result = boxes()
            .checked_state(ui_state::MIXED)
            .selected(true)
            .expanded(false)
            .disabled(true)
            .focused(true)
            .description("Hint")
            .value_text("Half")
            .get()
            .unwrap();
        assert_eq!(
            (result.id, result.name.as_str(), result.description.as_str(), result.value_text.as_str()),
            (4, "Some", "Hint", "Half")
        );
        assert!(result.disabled && !result.node.disabled);
        assert_eq!([result.value, result.minimum, result.maximum, result.step], [0.5, 0.0, 1.0, 0.25]);
        assert_eq!(result.rect, [0.0, 0.0, 48.0, 32.0]);
        assert!(screen.debug_tree().contains("checked=mixed expanded=false selected=true [disabled] <- focus"));
        assert_eq!(boxes().description("hint").case_insensitive().query().unwrap(), None);
    }

    #[test]
    fn traps_queries_and_actions_inside_the_topmost_visible_dialog() {
        let modal = |id, parent, name| UiNode { modal: true, ..node(id, parent, ui_role::DIALOG, name) };
        let button =
            |id, parent, name| UiNode { actions: 1 << ui_action::ACTIVATE, ..node(id, parent, ui_role::BUTTON, name) };
        let ui = [
            button(1, 0, "Outside"),
            modal(2, 0, "Settings"),
            button(3, 2, "Back"),
            modal(4, 2, "Confirm"),
            button(5, 4, "Continue"),
            UiNode { hidden: true, ..modal(7, 0, "Hidden") },
        ];
        let previous = RoleScreen::new(&ui[..1]).get_by_role("button").unwrap();
        let screen = RoleScreen::new(&ui);
        assert_eq!(screen.get_by_role("dialog").unwrap().id, 4);
        assert_eq!(screen.get_by_role("button").unwrap().id, 5);
        assert_eq!(screen.within(4).get_by_role("button").unwrap().id, 5);
        assert!(screen.within(2).by_role("button").name("Back").query().unwrap().is_none());
        let outside = screen.by_role("button").include_hidden(true).name("Outside").get().unwrap();
        assert!(outside.in_tree && !outside.hidden && outside.blocked_by_modal);
        assert_eq!(screen.click(&outside), Err(ActionError::BlockedByModal { id: 1 }));
        assert_eq!(screen.click(&previous), Err(ActionError::BlockedByModal { id: 1 }));
        assert!(screen.debug_tree().contains("active modal #4"));
        assert!(screen.debug_tree().contains("dialog #4 \"Confirm\""));
        assert!(!screen.debug_tree().contains("Outside"));
        let error = screen.by_role("tab").include_hidden(true).get().unwrap_err().to_string();
        assert!(error.contains("[blocked by modal]") && error.contains("[hidden]"));

        let mut ui = ui.to_vec();
        ui.push(UiNode { disabled: true, ..modal(6, 0, "Disabled") });
        let screen = RoleScreen::new(&ui);
        assert_eq!(screen.get_by_role("dialog").unwrap().id, 6);
        assert!(screen.query_all_by_role("button").is_empty());
    }

    #[test]
    fn builds_events_and_validates_current_action_support_and_disabled_state() {
        let slider = UiNode {
            actions: [
                ui_action::ACTIVATE,
                ui_action::INCREMENT,
                ui_action::DECREMENT,
                ui_action::SET_VALUE,
                ui_action::EXPAND,
                ui_action::COLLAPSE,
            ]
            .into_iter()
            .fold(0, |bits, action| bits | (1 << action)),
            focusable: true,
            ..node(51, 1, ui_role::SLIDER, "Volume")
        };
        let mut group = node(50, 0, ui_role::GROUP, "Audio");
        let ui = [group.clone(), slider.clone(), node(52, 0, ui_role::BUTTON, "Unavailable")];
        let screen = RoleScreen::new(&ui);
        let slider_node = screen.get_by_role("slider").unwrap();
        for (result, action, value) in [
            (screen.activate(&slider_node), ui_action::ACTIVATE, 0.0),
            (screen.click(&slider_node), ui_action::ACTIVATE, 0.0),
            (screen.focus(&slider_node), ui_action::FOCUS, 0.0),
            (screen.blur(&slider_node), ui_action::BLUR, 0.0),
            (screen.increment(&slider_node), ui_action::INCREMENT, 0.0),
            (screen.decrement(&slider_node), ui_action::DECREMENT, 0.0),
            (screen.set_value(&slider_node, 0.75), ui_action::SET_VALUE, 0.75),
            (screen.expand(&slider_node), ui_action::EXPAND, 0.0),
            (screen.collapse(&slider_node), ui_action::COLLAPSE, 0.0),
        ] {
            assert_eq!(result.unwrap(), GameEvent::UiAction { node: 51, action, value });
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(screen.set_value(&slider_node, value), Err(ActionError::NonFiniteValue { id: 51 }));
        }
        let unsupported = screen.get_by_role("button").unwrap();
        assert_eq!(screen.click(&unsupported), Err(ActionError::Unsupported { id: 52, action: ui_action::ACTIVATE }));
        assert_eq!(screen.focus(&unsupported), Err(ActionError::Unsupported { id: 52, action: ui_action::FOCUS }));
        assert_eq!(screen.within(51).click(&slider_node), Err(ActionError::NotFound { id: 51 }));
        group.disabled = true;
        assert_eq!(RoleScreen::new(&[group, slider]).click(&slider_node), Err(ActionError::Disabled { id: 51 }));
    }

    #[test]
    fn audits_the_whole_frame_and_reports_every_finding() {
        let clean = [node(1, 0, ui_role::BUTTON, "Save")];
        assert_eq!(RoleScreen::new(&clean).assert_audit_clean(), Ok(()));
        let ui = [
            node(1, 0, ui_role::BUTTON, ""),
            UiNode { focusable: true, width: 12.0, ..node(2, 0, ui_role::BUTTON, "Small") },
        ];
        let screen = RoleScreen::new(&ui);
        let findings = screen.within(2).audit();
        assert_eq!(findings, audit::audit(&ui, None));
        assert_eq!(
            findings.iter().map(|finding| (finding.node, finding.kind)).collect::<Vec<_>>(),
            [(1, audit_finding::UNNAMED), (2, audit_finding::SIZE)]
        );
        let error = screen.assert_audit_clean().unwrap_err();
        for finding in findings {
            assert!(error.contains(&finding.message));
        }
        assert!(error.contains(&screen.debug_tree()));
    }
}
