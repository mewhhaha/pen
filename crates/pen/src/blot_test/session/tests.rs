use super::*;

fn button(id: u32, name: &str) -> UiNode {
    let text = pen_runtime::services::code_points(name);
    UiNode {
        id,
        role: pen_protocol::ui_role::BUTTON,
        name_length: u32::try_from(text.len()).unwrap(),
        text_length: u32::try_from(text.len()).unwrap(),
        text,
        width: 40.0,
        height: 20.0,
        ..UiNode::default()
    }
}

#[test]
fn selectors_reject_ambiguity_and_exclude_hidden_subtrees() {
    let first = button(1, "Continue");
    let second = button(2, "Continue");
    assert!(selected(&[first.clone(), second.clone()], first.role, "Continue").is_err());
    let mut hidden = first.clone();
    hidden.hidden = true;
    let mut child = second;
    child.parent = 1;
    assert!(selected(&[hidden, child], first.role, "Continue").unwrap().is_none());
    assert_eq!(selected(std::slice::from_ref(&first), first.role, "Continue").unwrap(), Some(first));
}

#[test]
fn pointer_targets_use_the_visible_intersection_and_wait_for_motion() {
    let mut node = button(1, "Continue");
    node.x = -30.0;
    assert_eq!(center(&node, (100, 100)).unwrap(), (5.0, 10.0));
    let mut moved = node.clone();
    moved.x += 2.0;
    assert!(!stable(&node, &moved));
    moved.x = node.x + 0.1;
    assert!(stable(&node, &moved));
    moved.id = 2;
    assert!(!stable(&node, &moved));
    node.x = 101.0;
    assert!(center(&node, (100, 100)).is_err());
}

#[test]
fn malformed_driver_commands_and_unsafe_artifact_names_are_refused() {
    for values in [&[f32::NAN][..], &[1.5], &[-1.0]] {
        assert!(Input { values, at: 0 }.integer().is_err());
    }
    assert!(Input { values: &[2.0, 4.0], at: 0 }.array(10).is_err());
    assert!(Input { values: &[11.0], at: 0 }.array(10).is_err());
    assert!(Input { values: &[1.0, 0xd800 as f32], at: 0 }.text().is_err());
    assert!(Input { values: &[3601.0], at: 0 }.ticks().is_err());
    assert!(Input { values: &[0.0], at: 0 }.end().is_err());
    for name in ["", "../outside", "/absolute", "name.png", "a/b", "a\\b"] {
        assert!(safe_name(name).is_err());
    }
    assert!(safe_name("continued-colony_1").is_ok());
}
