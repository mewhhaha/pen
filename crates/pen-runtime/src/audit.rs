//! The accessibility audit the dev tools show (DEVTOOLS_OP_AUDIT, and
//! AuditFinding in `protocol/schema.json`): what in the game's interface, as
//! a frame declares it (`RenderFrame::ui`) and shows it (its pixels), shuts
//! a player out. It finds
//!
//! - controls, images and dialogs a screen reader cannot name;
//! - focus problems: a modal dialog with nothing in it to focus (the
//!   keyboard is trapped there), the focus outside the open modal dialog or
//!   on a node the tree leaves out, two focused nodes, and a focusable
//!   control partly off the screen; and, from presses of Tab
//!   ([`traversal`]), a focus the keyboard cannot move on and nodes it
//!   never reaches;
//! - text whose contrast with what lies behind it is below WCAG 2's 4.5:1
//!   (3:1 for large text), measured in the frame's pixels;
//! - controls smaller than WCAG 2.2's 24 by 24 pixels.
//!
//! Contrast is measured, not declared: the background is the median
//! luminance round the node's edge, the text the luminance farthest from it
//! that a twentieth of the pixels inside reach, so a few stray pixels (an
//! icon's rim, antialiasing) do not count. Text is large when its node is
//! at least 37 pixels high (text about half its row, 18.66 px and up). A
//! node with nothing drawn on it that differs from its background is not
//! measured: it may show no text at all.
//!
//! The same audit runs headless (`PEN_A11Y_AUDIT`) and for the dev tools'
//! panel, in every host that has the frame's pixels.
use std::fmt::Write as _;

use pen_protocol::boundary::{MAX_SERVICE_DATA, whole};
use pen_protocol::{UiNode, audit_finding, ui_role};

use crate::services::{Answer, code_points};

/// WCAG 2's least contrast for text, and for large text.
pub const TEXT_CONTRAST: f32 = 4.5;
pub const LARGE_CONTRAST: f32 = 3.0;
/// A node this high or more holds large text.
pub const LARGE_TEXT_ROW: f32 = 37.0;
/// WCAG 2.2's least target size, in pixels.
pub const TARGET: f32 = 24.0;

/// Something the audit found: its kind (AUDIT_FINDING_*), the node's id (0
/// for none), a measure (a contrast ratio, a size in pixels, or 0) and what
/// to fix, in words.
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    pub kind: u32,
    pub node: u32,
    pub measure: f32,
    pub message: String,
}

/// A frame's pixels: RGBA, 8 bits a channel, sRGB, rows top to bottom.
pub struct Picture<'a> {
    pub width: u32,
    pub height: u32,
    pub rgba: &'a [u8],
}

impl Picture<'_> {
    /// A pixel's relative luminance (WCAG 2), 0 to 1.
    fn luminance(&self, x: u32, y: u32) -> f32 {
        let at = (y as usize * self.width as usize + x as usize) * 4;
        let channel = |value: u8| {
            let value = f32::from(value) / 255.0;
            if value <= 0.040_45 { value / 12.92 } else { ((value + 0.055) / 1.055).powf(2.4) }
        };
        let [r, g, b] = [0, 1, 2].map(|offset| channel(self.rgba.get(at + offset).copied().unwrap_or(0)));
        0.2126 * r + 0.7152 * g + 0.0722 * b
    }
}

/// WCAG 2's contrast ratio of two luminances, 1 to 21.
pub fn contrast(one: f32, other: f32) -> f32 {
    (one.max(other) + 0.05) / (one.min(other) + 0.05)
}

/// A node's name and its value text.
fn words(node: &UiNode) -> (String, String) {
    let text = |points: &[f32]| -> String {
        points.iter().filter_map(|point| whole(*point).and_then(char::from_u32)).collect()
    };
    let name = (node.name_length as usize).min(node.text.len());
    let value = name.saturating_add(node.description_length as usize).min(node.text.len());
    (text(&node.text[..name]), text(&node.text[value..]))
}

/// The place of a node's parent, if one comes before it.
fn parent(index: usize, node: &UiNode) -> Option<usize> {
    (node.parent as usize).checked_sub(1).filter(|parent| *parent < index)
}

/// Which nodes the tree has, as hosts build it: not hidden nor under a node
/// it leaves out, the first with each id.
fn included(ui: &[UiNode]) -> Vec<bool> {
    let mut ids = std::collections::HashSet::with_capacity(ui.len());
    let mut included: Vec<bool> = Vec::with_capacity(ui.len());
    for (index, node) in ui.iter().enumerate() {
        let kept = !node.hidden && parent(index, node).is_none_or(|parent| included[parent]) && ids.insert(node.id);
        included.push(kept);
    }
    included
}

/// Whether the node at `index` is `ancestor` or lies under it.
fn under(ui: &[UiNode], mut index: usize, ancestor: usize) -> bool {
    loop {
        if index == ancestor {
            return true;
        }
        match parent(index, &ui[index]) {
            Some(above) => index = above,
            None => return false,
        }
    }
}

/// How a node is spoken of: its role, id and name.
fn called(node: &UiNode) -> String {
    let role = ui_role::NAMES.get(node.role as usize).copied().unwrap_or("node");
    let (name, _) = words(node);
    if name.is_empty() { format!("{role} #{:#x}", node.id) } else { format!("{role} #{:#x} {name:?}", node.id) }
}

fn interactive(node: &UiNode) -> bool {
    node.focusable || node.actions != 0
}

fn drawn(node: &UiNode) -> bool {
    node.width > 0.0 && node.height > 0.0 && [node.x, node.y, node.width, node.height].iter().all(|v| v.is_finite())
}

/// Roles a screen reader announces by their name, which must have one.
fn named_role(role: u32) -> bool {
    matches!(
        role,
        ui_role::BUTTON
            | ui_role::CHECKBOX
            | ui_role::SWITCH
            | ui_role::SLIDER
            | ui_role::SPINBUTTON
            | ui_role::MENUITEM
            | ui_role::TAB
            | ui_role::RADIO
            | ui_role::TEXTBOX
            | ui_role::IMAGE
            | ui_role::DIALOG
    )
}

/// Roles whose node shows its words as text where it lies.
fn shows_text(role: u32) -> bool {
    matches!(
        role,
        ui_role::TEXT
            | ui_role::HEADING
            | ui_role::BUTTON
            | ui_role::CHECKBOX
            | ui_role::SWITCH
            | ui_role::SLIDER
            | ui_role::SPINBUTTON
            | ui_role::LISTITEM
            | ui_role::MENUITEM
            | ui_role::TAB
            | ui_role::RADIO
            | ui_role::TEXTBOX
            | ui_role::STATUS
    )
}

/// What the audit finds in `ui`, shown as `picture` (None: no contrast,
/// and nothing about the screen's edges).
pub fn audit(ui: &[UiNode], picture: Option<&Picture>) -> Vec<Finding> {
    let included = included(ui);
    let mut findings = Vec::new();
    let mut find = |kind: u32, node: &UiNode, measure: f32, message: String| {
        findings.push(Finding { kind, node: node.id, measure, message });
    };
    // Names.
    for (node, _) in ui.iter().zip(&included).filter(|(_, kept)| **kept) {
        let (name, value) = words(node);
        let role = ui_role::NAMES.get(node.role as usize).copied().unwrap_or("node");
        let silent = if matches!(node.role, ui_role::TEXT | ui_role::HEADING | ui_role::STATUS) {
            name.is_empty() && value.is_empty()
        } else {
            name.is_empty() && (named_role(node.role) || interactive(node))
        };
        if silent {
            let message = format!("{} has no name: a screen reader says only \"{role}\"", called(node));
            find(audit_finding::UNNAMED, node, 0.0, message);
        }
    }
    // The focus.
    let focused: Vec<usize> = (0..ui.len()).filter(|index| ui[*index].focused).collect();
    for index in &focused {
        if !included[*index] {
            let message = format!("{} has the focus but the tree leaves it out", called(&ui[*index]));
            find(audit_finding::FOCUS, &ui[*index], 0.0, message);
        }
    }
    let shown_focus: Vec<usize> = focused.iter().copied().filter(|index| included[*index]).collect();
    if shown_focus.len() > 1 {
        let all: Vec<String> = shown_focus.iter().map(|index| called(&ui[*index])).collect();
        let message = format!("{} nodes have the focus at once: {}", all.len(), all.join(", "));
        find(audit_finding::FOCUS, &ui[shown_focus[1]], 0.0, message);
    }
    for (index, dialog) in ui.iter().enumerate() {
        if !included[index] || dialog.role != ui_role::DIALOG || !dialog.modal {
            continue;
        }
        let holds = (0..ui.len())
            .any(|inner| inner != index && included[inner] && ui[inner].focusable && under(ui, inner, index));
        if !holds {
            let message =
                format!("{} is modal but holds nothing to focus: the keyboard is trapped in it", called(dialog));
            find(audit_finding::FOCUS, dialog, 0.0, message);
        }
        if let Some(outside) = shown_focus.iter().copied().find(|focus| !under(ui, *focus, index)) {
            let message =
                format!("the focus is on {}, outside the open modal {}", called(&ui[outside]), called(dialog));
            find(audit_finding::FOCUS, &ui[outside], 0.0, message);
        }
    }
    if let Some(picture) = picture {
        let (width, height) = (picture.width as f32, picture.height as f32);
        for (node, _) in ui.iter().zip(&included).filter(|(node, kept)| **kept && node.focusable && drawn(node)) {
            let off = node.x < 0.0 || node.y < 0.0 || node.x + node.width > width || node.y + node.height > height;
            if off {
                let message = format!(
                    "{} lies partly off the screen (at {:.0},{:.0}, {:.0}x{:.0}): a keyboard user cannot see it focused",
                    called(node),
                    node.x,
                    node.y,
                    node.width,
                    node.height
                );
                find(audit_finding::FOCUS, node, 0.0, message);
            }
        }
    }
    // Contrast.
    if let Some(picture) = picture {
        for (node, _) in ui.iter().zip(&included).filter(|(node, kept)| **kept && shows_text(node.role) && drawn(node))
        {
            let (name, value) = words(node);
            if name.is_empty() && value.is_empty() {
                continue;
            }
            let Some(ratio) = measure(picture, node) else { continue };
            let least = if node.height >= LARGE_TEXT_ROW { LARGE_CONTRAST } else { TEXT_CONTRAST };
            if ratio < least {
                let message = format!("{}: its text's contrast is {ratio:.1}:1, below {least}:1", called(node));
                find(audit_finding::CONTRAST, node, ratio, message);
            }
        }
    }
    // Target size.
    for (node, _) in ui.iter().zip(&included).filter(|(node, kept)| **kept && interactive(node) && drawn(node)) {
        let least = node.width.min(node.height);
        if least < TARGET {
            let message = format!(
                "{} is {:.0}x{:.0} pixels, smaller than {TARGET:.0}x{TARGET:.0}",
                called(node),
                node.width,
                node.height
            );
            find(audit_finding::SIZE, node, least, message);
        }
    }
    findings
}

/// The contrast of what is drawn on `node` against what lies round it, if
/// anything drawn on it stands out (None for a node off the picture, too
/// small to measure, or with nothing on it).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "pixels, clipped to the picture")]
fn measure(picture: &Picture, node: &UiNode) -> Option<f32> {
    let clip = |value: f32, most: u32| value.max(0.0).min(most as f32) as u32;
    let [left, right] = [node.x, node.x + node.width].map(|x| clip(x, picture.width));
    let [top, bottom] = [node.y, node.y + node.height].map(|y| clip(y, picture.height));
    if right < left + 4 || bottom < top + 4 {
        return None;
    }
    let mut edge = Vec::new();
    let mut inside = Vec::new();
    for y in top..bottom {
        for x in left..right {
            let luminance = picture.luminance(x, y);
            if x == left || y == top || x + 1 == right || y + 1 == bottom {
                edge.push(luminance);
            } else {
                inside.push(luminance);
            }
        }
    }
    edge.sort_by(f32::total_cmp);
    let background = edge[edge.len() / 2];
    let mut apart: Vec<(f32, f32)> =
        inside.iter().map(|luminance| (contrast(*luminance, background), *luminance)).collect();
    apart.sort_by(|one, other| one.0.total_cmp(&other.0));
    // The text: what a twentieth of the pixels inside reach.
    let at = apart.len().saturating_sub(apart.len() / 20 + 1);
    let (ratio, _) = *apart.get(at)?;
    // Nothing stands out: no text drawn here to measure.
    (ratio >= 1.15).then_some(ratio)
}

/// What moving the focus through `ui` with the keyboard shows: `visited`
/// is the focused node's id after each press of Tab (0 for none), from the
/// state `ui` shows. The keyboard is trapped when presses stop moving the
/// focus while other nodes could take it; a focusable node a full round
/// never reaches cannot be reached by keyboard at all.
pub fn traversal(ui: &[UiNode], visited: &[u32]) -> Vec<Finding> {
    let included = included(ui);
    let focusable: Vec<&UiNode> =
        ui.iter().zip(&included).filter(|(node, kept)| **kept && node.focusable).map(|(node, _)| node).collect();
    let mut findings = Vec::new();
    if focusable.len() < 2 || visited.len() < 2 {
        return findings;
    }
    let stuck = visited.windows(2).all(|pair| pair[0] == pair[1]);
    if stuck && let Some(node) = focusable.iter().find(|node| node.id == visited[0]) {
        let message = format!("Tab does not move the focus off {}: the keyboard is trapped there", called(node));
        findings.push(Finding { kind: audit_finding::FOCUS, node: node.id, measure: 0.0, message });
        return findings;
    }
    if visited.len() > focusable.len() {
        for node in focusable.iter().filter(|node| !visited.contains(&node.id)) {
            let message = format!("Tab never reaches {}", called(node));
            findings.push(Finding { kind: audit_finding::FOCUS, node: node.id, measure: 0.0, message });
        }
    }
    findings
}

/// The findings as a devtools answer: value their count, and for each its
/// kind, node, measure, the message's length and its code points, as many
/// as one answer carries.
pub fn answer(findings: &[Finding]) -> Answer {
    let mut values = Vec::new();
    let mut count = 0;
    for finding in findings {
        let message = code_points(&finding.message);
        if values.len() + 4 + message.len() > MAX_SERVICE_DATA {
            break;
        }
        values.extend([finding.kind as f32, finding.node as f32, finding.measure, message.len() as f32]);
        values.extend(message);
        count += 1;
    }
    Answer { values, ..Answer::ok(count as f32) }
}

/// The findings as text, a line each: its kind, then the message.
pub fn report(findings: &[Finding]) -> String {
    let mut text = String::new();
    for finding in findings {
        let kind = audit_finding::NAMES.get(finding.kind as usize).copied().unwrap_or("finding");
        let _ = writeln!(text, "{kind}: {}", finding.message);
    }
    if findings.is_empty() {
        text.push_str("no findings\n");
    }
    text
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, reason = "short test texts")]
mod tests {
    use super::*;

    fn node(id: u32, parent: u32, role: u32, name: &str, rect: [f32; 4]) -> UiNode {
        let text: Vec<f32> = name.chars().map(|character| character as u32 as f32).collect();
        UiNode {
            id,
            parent,
            role,
            x: rect[0],
            y: rect[1],
            width: rect[2],
            height: rect[3],
            name_length: text.len() as u32,
            text_length: text.len() as u32,
            text,
            ..UiNode::default()
        }
    }

    fn kinds(findings: &[Finding]) -> Vec<(u32, u32)> {
        findings.iter().map(|finding| (finding.kind, finding.node)).collect()
    }

    #[test]
    fn finds_controls_without_names() {
        let mut silent = node(2, 0, ui_role::BUTTON, "", [0.0, 0.0, 40.0, 40.0]);
        silent.focusable = true;
        let image = node(3, 0, ui_role::IMAGE, "", [0.0, 0.0, 40.0, 40.0]);
        let empty_text = node(4, 0, ui_role::TEXT, "", [0.0, 0.0, 40.0, 40.0]);
        // A group with nothing to say is fine, and so is a named button.
        let group = node(5, 0, ui_role::GROUP, "", [0.0, 0.0, 40.0, 40.0]);
        let mut named = node(6, 0, ui_role::BUTTON, "Play", [0.0, 0.0, 40.0, 40.0]);
        named.focusable = true;
        // A hidden one is not in the tree at all.
        let mut hidden = node(7, 0, ui_role::BUTTON, "", [0.0, 0.0, 40.0, 40.0]);
        hidden.hidden = true;
        let findings = audit(&[silent, image, empty_text, group, named, hidden], None);
        let unnamed = audit_finding::UNNAMED;
        assert_eq!(kinds(&findings), [(unnamed, 2), (unnamed, 3), (unnamed, 4)]);
        assert_eq!(findings[0].message, "button #0x2 has no name: a screen reader says only \"button\"");
    }

    #[test]
    fn finds_focus_traps_and_lost_focus() {
        let mut dialog = node(1, 0, ui_role::DIALOG, "Paused", [0.0, 0.0, 100.0, 100.0]);
        dialog.modal = true;
        let heading = node(2, 1, ui_role::HEADING, "Paused", [0.0, 0.0, 100.0, 20.0]);
        let mut outside = node(3, 0, ui_role::BUTTON, "Hire", [0.0, 0.0, 30.0, 30.0]);
        (outside.focusable, outside.focused) = (true, true);
        let focus = audit_finding::FOCUS;
        assert_eq!(kinds(&audit(&[dialog.clone(), heading.clone(), outside.clone()], None)), [(focus, 1), (focus, 3)]);
        // With a button inside that has the focus, all is well.
        let mut inside = node(4, 1, ui_role::BUTTON, "Continue", [0.0, 30.0, 100.0, 30.0]);
        (inside.focusable, inside.focused) = (true, true);
        outside.focused = false;
        assert_eq!(audit(&[dialog.clone(), heading.clone(), inside.clone(), outside.clone()], None), []);
        // Two focused, and a focused node the tree leaves out.
        outside.focused = true;
        let mut gone = node(5, 0, ui_role::BUTTON, "Gone", [0.0, 0.0, 30.0, 30.0]);
        (gone.focused, gone.hidden) = (true, true);
        let findings = audit(&[inside, outside, gone], None);
        assert_eq!(kinds(&findings), [(focus, 5), (focus, 3)]);
        assert!(findings[1].message.starts_with("2 nodes have the focus at once"), "{}", findings[1].message);
    }

    #[test]
    fn finds_small_and_offscreen_controls() {
        let mut small = node(1, 0, ui_role::BUTTON, "x", [10.0, 10.0, 18.0, 30.0]);
        small.focusable = true;
        let mut off = node(2, 0, ui_role::BUTTON, "Hire", [930.0, -20.0, 60.0, 60.0]);
        off.focusable = true;
        let pixels = vec![0; 960 * 640 * 4];
        let picture = Picture { width: 960, height: 640, rgba: &pixels };
        let findings = audit(&[small, off], Some(&picture));
        assert_eq!(kinds(&findings), [(audit_finding::FOCUS, 2), (audit_finding::SIZE, 1)]);
        assert_eq!(findings[1].measure, 18.0);
    }

    /// A 100x40 picture, `background` grey, with "text" of grey `ink` in
    /// its middle: bars two pixels wide, every other pair of columns.
    fn picture(background: u8, ink: u8) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(100 * 40 * 4);
        for y in 0..40 {
            for x in 0..100 {
                let text = (10..30).contains(&y) && (20..80).contains(&x) && (x / 2) % 2 == 0;
                let grey = if text { ink } else { background };
                pixels.extend([grey, grey, grey, 255]);
            }
        }
        pixels
    }

    #[test]
    fn measures_contrast_in_the_pixels() {
        let label = node(1, 0, ui_role::TEXT, "Salt", [0.0, 0.0, 100.0, 36.0]);
        // White on black: 21:1, nothing to say.
        let pixels = picture(0, 255);
        let shown = Picture { width: 100, height: 40, rgba: &pixels };
        assert_eq!(audit(std::slice::from_ref(&label), Some(&shown)), []);
        assert!((measure(&shown, &label).unwrap() - 21.0).abs() < 0.01);
        // Mid grey on a little darker grey: low.
        let pixels = picture(100, 140);
        let shown = Picture { width: 100, height: 40, rgba: &pixels };
        let findings = audit(std::slice::from_ref(&label), Some(&shown));
        assert_eq!(kinds(&findings), [(audit_finding::CONTRAST, 1)]);
        let expected = contrast(
            Picture { width: 1, height: 1, rgba: &[140, 140, 140, 255] }.luminance(0, 0),
            Picture { width: 1, height: 1, rgba: &[100, 100, 100, 255] }.luminance(0, 0),
        );
        assert!((findings[0].measure - expected).abs() < 0.01, "{} against {expected}", findings[0].measure);
        // Large text takes 3:1; nothing drawn is not measured.
        let large = node(2, 0, ui_role::HEADING, "Title", [0.0, 0.0, 100.0, 40.0]);
        let pixels = picture(0, 100);
        let shown = Picture { width: 100, height: 40, rgba: &pixels };
        let ratio = measure(&shown, &large).unwrap();
        assert!(ratio > 3.0 && ratio < 4.5, "{ratio}");
        assert_eq!(audit(&[large], Some(&shown)), []);
        let pixels = picture(80, 80);
        let blank = Picture { width: 100, height: 40, rgba: &pixels };
        assert_eq!(measure(&blank, &label), None);
    }

    #[test]
    fn finds_keyboard_traps_by_moving_the_focus() {
        let mut ui = Vec::new();
        for id in 1..=3 {
            let mut button = node(id, 0, ui_role::BUTTON, "b", [0.0, 0.0, 30.0, 30.0]);
            button.focusable = true;
            ui.push(button);
        }
        // Round and round: fine.
        assert_eq!(traversal(&ui, &[1, 2, 3, 1, 2]), []);
        // Stuck on one: a trap.
        let trapped = traversal(&ui, &[2, 2, 2, 2]);
        assert_eq!(kinds(&trapped), [(audit_finding::FOCUS, 2)]);
        assert!(trapped[0].message.contains("trapped"), "{}", trapped[0].message);
        // Between two, never the third.
        assert_eq!(kinds(&traversal(&ui, &[1, 2, 1, 2, 1])), [(audit_finding::FOCUS, 3)]);
        // Too few presses to tell.
        assert_eq!(traversal(&ui, &[1, 2]), []);
    }

    #[test]
    fn answers_and_reports_the_findings() {
        let findings = vec![
            Finding { kind: audit_finding::SIZE, node: 7, measure: 18.0, message: "ab".into() },
            Finding { kind: audit_finding::UNNAMED, node: 8, measure: 0.0, message: "c".into() },
        ];
        let answer = answer(&findings);
        assert_eq!(answer.value, 2.0);
        assert_eq!(answer.values, [3.0, 7.0, 18.0, 2.0, 97.0, 98.0, 0.0, 8.0, 0.0, 1.0, 99.0]);
        assert_eq!(report(&findings), "size: ab\nunnamed: c\n");
        assert_eq!(report(&[]), "no findings\n");
    }
}
