//! Snapshots of the real game's tree, as assistive technology gets it, and
//! of the accessibility audit's findings in it: the exported game
//! (`dist/gdev`, `just export`) runs a script without a GPU, and at each of
//! its `shot=NAME` steps the tree's text (`describe`, as PEN_A11Y_DUMP
//! writes it) and the audit's report (without pixels: names, focus and
//! sizes) must match `crates/pen/tests/a11y/NAME.txt` and `NAME.audit.txt`.
//! A `shot=NAME-keys` step after presses of Tab writes `NAME-keys.audit.txt`
//! instead: where the focus went, and what the audit's `traversal` finds in
//! it. A change to the game's interface shows here as a diff to look at;
//! with PEN_UPDATE_A11Y=1 the run writes the files instead. Skipped without
//! the bundle (`fixtures`).
use std::path::{Path, PathBuf};

use anyhow::Result;
use pen_protocol::boundary::read_catalog;
use pen_protocol::{Catalog, Frame, RenderFrame, Viewport};

use crate::fixtures::bundle;
use crate::game::{GameRuntime, Host};
use crate::script::{self, Script};
use crate::snapshot;

const VIEWPORT: Viewport = Viewport { width: 960, height: 640 };

struct Null;

impl Host for Null {
    fn viewport(&self) -> Viewport {
        VIEWPORT
    }

    fn validate(&self, frame: &RenderFrame) -> Result<()> {
        pen_render::validate_frame(frame)
    }

    fn render(&mut self, _frame: &RenderFrame, _catalog: &Catalog) -> Result<()> {
        Ok(())
    }

    fn report(&mut self, _message: &str) {}
}

fn expected() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/a11y")
}

/// The editor, the pause menu over it and Tab round it, and Settings'
/// Accessibility page (F2, then three sections on) and Tab round that, with
/// a snapshot of each.
const STEPS: &str = "wait=30 shot=editing key=Escape wait=40 shot=paused \
                     key=Tab wait=2 key=Tab wait=2 key=Tab wait=2 key=Tab wait=2 shot=paused-keys \
                     key=Escape wait=20 key=F2 wait=20 key=PageDown key=PageDown key=PageDown wait=20 shot=access \
                     key=Tab wait=2 key=Tab wait=2 key=Tab wait=2 key=Tab wait=2 key=Tab wait=2 key=Tab wait=2 \
                     key=Tab wait=2 shot=access-keys";

/// What a `shot=NAME-keys` step writes: where Tab took the focus from the
/// interface `before`, by name, and the audit's traversal findings.
fn keyed(before: &[pen_protocol::UiNode], visited: &[u32]) -> String {
    let name = |id: u32| {
        before.iter().find(|node| node.id == id).map_or(format!("#{id:#x}"), |node| {
            let length = (node.name_length as usize).min(node.text.len());
            node.text[..length]
                .iter()
                .filter_map(|point| pen_protocol::boundary::whole(*point).and_then(char::from_u32))
                .collect()
        })
    };
    let went: Vec<String> = visited.iter().map(|id| name(*id)).collect();
    let findings = pen_runtime::audit::traversal(before, visited);
    format!("Tab: {}\n{}", went.join(" -> "), pen_runtime::audit::report(&findings))
}

/// The files of `made` that differ from their snapshots (or, updating,
/// none: it writes them).
fn compare(made: &[(String, String)], update: bool) -> Vec<String> {
    let mut differ = Vec::new();
    for (file, text) in made {
        let path = expected().join(file);
        if update {
            std::fs::create_dir_all(expected()).unwrap();
            std::fs::write(&path, text).unwrap();
        } else if std::fs::read_to_string(&path).ok().as_deref() != Some(text.as_str()) {
            eprintln!("{file} differs; it is now:\n{text}");
            differ.push(file.clone());
        }
    }
    differ
}

#[test]
fn the_games_tree_and_audit_match_their_snapshots() {
    let Some(directory) = bundle() else { return };
    let engine = blot_host::engine().unwrap();
    let wasm = std::fs::read(directory.join("game.wasm")).unwrap();
    let cache = std::env::temp_dir().join("pen-test-wasm");
    let module = blot_host::GuestModule::new(&engine, &wasm, Some(&cache)).unwrap();
    let catalog = read_catalog(&directory.join("assets")).unwrap();
    let scene = snapshot::read(&directory.join("main.scene")).ok();
    let mut runtime = GameRuntime::new(&engine, module, catalog, &mut Null, None, scene).unwrap();
    let mut script = Script::new(script::parse(&[STEPS.to_owned()]).unwrap());
    let update = std::env::var_os("PEN_UPDATE_A11Y").is_some_and(|value| !value.is_empty() && value != "0");
    let mut time = 0.0;
    let mut differ = Vec::new();
    // The interface at the last snapshot, and where Tab took the focus since.
    let mut before = Vec::new();
    let mut visited = Vec::new();
    while !script.done() {
        let tick = script.tick(VIEWPORT);
        let tabbed =
            tick.events.iter().any(|event| matches!(event, pen_protocol::GameEvent::KeyDown(key) if key == "Tab"));
        let input = runtime.input(&Frame { timestamp: time, viewport: VIEWPORT, events: tick.events }).unwrap();
        time += 1.0 / 60.0;
        runtime.send(&input).unwrap();
        let rendered = runtime.receive(&Null).unwrap();
        runtime.follow(&mut Null, &rendered).unwrap();
        if tabbed {
            visited.push(rendered.ui.iter().find(|node| node.focused).map_or(0, |node| node.id));
        }
        for name in tick.shots {
            if name.ends_with("-keys") {
                let made = [(format!("{name}.audit.txt"), keyed(&before, &visited))];
                differ.extend(compare(&made, update));
                visited.clear();
                continue;
            }
            before.clone_from(&rendered.ui);
            visited.clear();
            let made = [
                // The window's title is the checkout's, wherever it is.
                (format!("{name}.txt"), super::describe_ui(&rendered.ui).replacen(&crate::identity().title, "gdev", 1)),
                (
                    format!("{name}.audit.txt"),
                    pen_runtime::audit::report(&pen_runtime::audit::audit(&rendered.ui, None)),
                ),
            ];
            differ.extend(compare(&made, update));
        }
    }
    runtime.close();
    assert!(differ.is_empty(), "{differ:?} differ from crates/pen/tests/a11y (PEN_UPDATE_A11Y=1 writes them)");
}
