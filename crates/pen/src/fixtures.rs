//! What tests read that the repository does not hold: the game bundle `just
//! export` writes (`dist/gdev`) and the sibling gdev checkout. A test whose
//! fixture is missing skips, saying why; with PEN_REQUIRE_FIXTURES=1 (CI's
//! game job sets it, after making them) it fails instead, so a run meant to
//! have the fixtures cannot pass without running those tests.
use std::path::{Path, PathBuf};

/// Whether a missing fixture fails the test.
fn required() -> bool {
    std::env::var_os("PEN_REQUIRE_FIXTURES").is_some_and(|value| !value.is_empty() && value != "0")
}

/// A test's fixture is missing (`what` says which): it skips, or fails
/// under PEN_REQUIRE_FIXTURES.
#[track_caller]
pub fn missing(what: &str) {
    assert!(!required(), "{what}, and PEN_REQUIRE_FIXTURES is set");
    eprintln!("skipped: {what}");
}

/// The exported game (`dist/gdev`), if it is there.
pub fn bundle() -> Option<PathBuf> {
    let override_path = std::env::var_os("PEN_TEST_BUNDLE");
    let directory = override_path
        .as_ref()
        .map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dist/gdev"), PathBuf::from);
    assert!(
        override_path.is_none() || directory.join("game.wasm").is_file(),
        "PEN_TEST_BUNDLE has no game.wasm: {}",
        directory.display()
    );
    if directory.join("game.wasm").is_file() {
        return Some(directory);
    }
    missing("no dist/gdev (`just export`)");
    None
}
