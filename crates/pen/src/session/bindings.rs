//! The game's actions and the player's bindings (`pen_runtime::actions`),
//! kept between runs in `bindings.json` beside the preferences: the input
//! service changes them, and the session writes them when they changed.
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use pen_runtime::actions::{Actions, Shared};

/// The actions last kept at `path`, or none.
pub(super) fn load(path: &Path) -> Shared {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    Rc::new(RefCell::new(Actions::load(&text)))
}

/// Writes the actions to `path` if they changed since they were last
/// written; a failure is reported, and tried again with the next change.
pub(super) fn keep(actions: &Shared, path: &Path) {
    let mut actions = actions.borrow_mut();
    if !actions.take_changed() {
        return;
    }
    let written =
        path.parent().map_or(Ok(()), std::fs::create_dir_all).and_then(|()| std::fs::write(path, actions.save()));
    if let Err(error) = written {
        eprintln!("keeping the player's bindings in {}: {error}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use pen_protocol::{input_op, input_source};
    use pen_runtime::services::{Request, code_points};

    use super::*;

    #[test]
    fn bindings_are_written_when_they_change_and_read_back() {
        let directory = std::env::temp_dir().join(format!("pen-bindings-{}", std::process::id()));
        let path = directory.join("data/bindings.json");
        let _ = std::fs::remove_dir_all(&directory);
        let actions = load(&path);
        keep(&actions, &path);
        assert!(!path.exists(), "nothing changed");
        let mut values = code_points("jump");
        values.extend([input_source::KEY as f32, 32.0, 1.0]);
        let declare = Request { id: 1, operation: input_op::DECLARE, numbers: [0.0, 0.0, 4.0, 0.0], values: &values };
        actions.borrow_mut().call(&declare);
        keep(&actions, &path);
        let loaded = load(&path);
        assert_eq!(loaded.borrow().action(0), actions.borrow().action(0));
        let _ = std::fs::remove_dir_all(&directory);
    }
}
