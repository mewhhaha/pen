//! Text as the game's events, the same on every host: committed text, typed
//! or from an input method (IME), as TextInput events, a code point each;
//! and an input method's composition, the text being composed before it is
//! committed, as a Composition event (its length and caret) followed by its
//! characters as TextInput events marked composing. Hosts turn on their
//! input method only while a text box has the focus ([`text_box`]): it
//! takes keys a game would otherwise get.
use pen_protocol::{GameEvent, UiNode, ui_role};

/// Whether a character is text, not a control (Enter, Backspace, Tab and
/// Escape arrive as KeyDown events, with their names).
fn printable(character: char) -> bool {
    !character.is_control()
}

/// Committed text: a TextInput event for each of its printable characters.
pub fn commit(text: &str, events: &mut Vec<GameEvent>) {
    for character in text.chars().filter(|character| printable(*character)) {
        events.push(GameEvent::TextInput { character: u32::from(character), composing: 0 });
    }
}

/// The composition is now `text`, its caret `cursor` characters from its
/// start (at its end when None): a Composition event, then its characters.
/// Empty text ends the composition.
pub fn compose(text: &str, cursor: Option<usize>, events: &mut Vec<GameEvent>) {
    let characters: Vec<char> = text.chars().filter(|character| printable(*character)).collect();
    let length = u32::try_from(characters.len()).expect("a composition shorter than 2^32");
    let cursor = cursor.map_or(length, |cursor| u32::try_from(cursor).unwrap_or(u32::MAX).min(length));
    events.push(GameEvent::Composition { length, cursor });
    for character in characters {
        events.push(GameEvent::TextInput { character: u32::from(character), composing: 1 });
    }
}

/// How many characters of `text` come before byte `index` (winit gives an
/// input method's caret in bytes); an index inside a character counts it.
pub fn characters_before_byte(text: &str, index: usize) -> usize {
    text.char_indices().take_while(|(at, _)| *at < index).count()
}

/// How many characters of `text` come before UTF-16 unit `index` (a
/// browser gives a text field's caret in UTF-16 units).
pub fn characters_before_utf16(text: &str, index: usize) -> usize {
    let mut units = 0;
    text.chars()
        .take_while(|character| {
            units += character.len_utf16();
            units <= index
        })
        .count()
}

/// The focused text box of an interface, if one has the focus: where an
/// input method shows what it composes, in window pixels (x, y, width,
/// height).
pub fn text_box(ui: &[UiNode]) -> Option<(f32, f32, f32, f32)> {
    ui.iter()
        .find(|node| node.focused && !node.hidden && node.role == ui_role::TEXTBOX)
        .map(|node| (node.x, node.y, node.width, node.height))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn characters(events: &[GameEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                GameEvent::TextInput { character, .. } => char::from_u32(*character),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn committed_text_is_a_character_an_event_without_controls() {
        let mut events = Vec::new();
        commit("añ\r漢😀\u{8}", &mut events);
        assert_eq!(characters(&events), "añ漢😀");
        assert!(events.iter().all(|event| matches!(event, GameEvent::TextInput { composing: 0, .. })));
        assert_eq!(events[3], GameEvent::TextInput { character: 0x1F600, composing: 0 });
    }

    #[test]
    fn a_composition_says_its_length_and_caret_then_its_characters() {
        let mut events = Vec::new();
        compose("かん", Some(1), &mut events);
        assert_eq!(
            events,
            [
                GameEvent::Composition { length: 2, cursor: 1 },
                GameEvent::TextInput { character: 'か' as u32, composing: 1 },
                GameEvent::TextInput { character: 'ん' as u32, composing: 1 },
            ]
        );
        events.clear();
        compose("漢", None, &mut events);
        assert_eq!(events[0], GameEvent::Composition { length: 1, cursor: 1 });
        events.clear();
        compose("", None, &mut events);
        assert_eq!(events, [GameEvent::Composition { length: 0, cursor: 0 }]);
        // A caret past the end stays at the end.
        events.clear();
        compose("a", Some(9), &mut events);
        assert_eq!(events[0], GameEvent::Composition { length: 1, cursor: 1 });
    }

    #[test]
    fn carets_count_characters_from_bytes_and_utf16_units() {
        let text = "aか😀b";
        assert_eq!(characters_before_byte(text, 0), 0);
        assert_eq!(characters_before_byte(text, 1), 1);
        assert_eq!(characters_before_byte(text, 4), 2);
        assert_eq!(characters_before_byte(text, 8), 3);
        assert_eq!(characters_before_byte(text, 100), 4);
        assert_eq!(characters_before_utf16(text, 2), 2);
        // Inside the emoji's surrogate pair: before it.
        assert_eq!(characters_before_utf16(text, 3), 2);
        assert_eq!(characters_before_utf16(text, 4), 3);
        assert_eq!(characters_before_utf16(text, 5), 4);
    }

    #[test]
    fn only_a_focused_visible_text_box_takes_text() {
        let node = |role, focused, hidden| UiNode {
            role,
            focused,
            hidden,
            x: 1.0,
            y: 2.0,
            width: 3.0,
            height: 4.0,
            ..UiNode::default()
        };
        assert_eq!(text_box(&[node(ui_role::BUTTON, true, false)]), None);
        assert_eq!(text_box(&[node(ui_role::TEXTBOX, false, false)]), None);
        assert_eq!(text_box(&[node(ui_role::TEXTBOX, true, true)]), None);
        assert_eq!(
            text_box(&[node(ui_role::BUTTON, false, false), node(ui_role::TEXTBOX, true, false)]),
            Some((1.0, 2.0, 3.0, 4.0))
        );
    }
}
