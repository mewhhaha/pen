//! The Blot compiler plugin (plugins/blot) hosted for real: pen.json grants it
//! blot2's packaged blotc, which it starts through the host's process
//! interface and drives over `serve-project`. Skipped without the plugin's
//! component (`just plugins`) or the sibling blot2 checkout.
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use pen_plugins::{BuildResult, Level, Outcome, Overlay, Plugins};

const MAIN: &str = "import * as lib from \"./lib\"\n\nentry const answer = fn () => lib.base + 1\n";
const LIB: &str = "const base: U32 = 41\n";

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// The plugin's component and blot2's compiler and std, if they are there.
fn parts() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let component = [
        std::env::var_os("PEN_TEST_BLOT_PLUGIN").map(PathBuf::from),
        Some(repository().join("plugins/target/wasm32-wasip2/release/pen_plugin_blot.wasm")),
    ]
    .into_iter()
    .flatten()
    .find(|path| path.is_file())?;
    let blot2 = repository().join("../blot2").canonicalize().ok()?;
    let compiler = blot2.join("generated/compiler/blotc.gz");
    let std = blot2.join("std");
    (compiler.is_file() && std.is_dir()).then_some((component, compiler, std))
}

/// A checkout of two files and its pen.json, in pen's cache.
fn checkout(component: &Path, compiler: &Path, std: &Path) -> PathBuf {
    let root = dirs::cache_dir()
        .unwrap()
        .join(format!("pen/tmp/pen-plugins-blot-{}-{:?}", std::process::id(), std::thread::current().id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("main.blot"), MAIN).unwrap();
    std::fs::write(root.join("lib.blot"), LIB).unwrap();
    let config = serde_json::json!({
        "plugins": {
            "blot": {
                "wasm": component,
                "read": [std],
                "run": { "blotc": compiler },
            },
        },
        "compiler": { "plugin": "blot", "settings": { "std": std } },
    });
    std::fs::write(root.join("pen.json"), config.to_string()).unwrap();
    root.canonicalize().unwrap()
}

fn plugins(root: &Path) -> (Arc<Plugins>, Arc<Mutex<Vec<String>>>) {
    let said = Arc::new(Mutex::new(Vec::new()));
    let log = said.clone();
    let logger: pen_plugins::Logger = Arc::new(move |level: Level, message: &str| {
        log.lock().unwrap().push(format!("{level:?}: {message}"));
    });
    (Plugins::load(root, logger).unwrap().expect("pen.json is there"), said)
}

#[test]
fn builds_and_rejects_through_the_blot_plugin() {
    let Some((component, compiler, std)) = parts() else {
        eprintln!("skipped: no Blot plugin component or blot2 checkout");
        return;
    };
    let root = checkout(&component, &compiler, &std);
    let (plugins, said) = plugins(&root);
    let entry = root.join("main.blot");
    let mut session = plugins.compiler(&entry).unwrap();
    let BuildResult::Built(wasm) = session.build(&[]).unwrap() else { panic!("the program compiles") };
    assert_eq!(&wasm[..4], b"\0asm");
    // An overlay that does not type checks is the program's failure, not
    // the compiler's: the session goes on.
    let wrong = "import * as lib from \"./lib\"\n\nconst wrong: Bool = lib.base\n\nentry const answer = fn () => wrong\n";
    let overlay = Overlay { path: entry.display().to_string(), contents: Some(wrong.into()) };
    let BuildResult::Rejected(diagnostics) = session.build(&[overlay]).unwrap() else { panic!("the overlay rejects") };
    assert_eq!(diagnostics[0].file, entry.display().to_string());
    assert!(diagnostics[0].end > diagnostics[0].start);
    let BuildResult::Built(again) = session.build(&[]).unwrap() else { panic!("the disk compiles again") };
    assert_eq!(again, wasm);
    let watched = session.watched().unwrap();
    assert!(watched.contains(&root), "{watched:?}");
    drop(session);
    // What a whole build comes to, as pen's builds use it.
    let Outcome::Built(built) = pen_plugins::build(plugins, &entry) else { panic!("it builds") };
    assert_eq!(built.wasm, wasm);
    assert!(said.lock().unwrap().iter().any(|line| line.starts_with("Info: ")), "{:?}", said.lock().unwrap());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn refuses_programs_the_checkout_does_not_grant() {
    let Some((component, compiler, std)) = parts() else {
        eprintln!("skipped: no Blot plugin component or blot2 checkout");
        return;
    };
    let root = checkout(&component, &compiler, &std);
    let config = serde_json::json!({
        "plugins": { "blot": { "wasm": component } },
        "compiler": { "plugin": "blot", "settings": { "std": std } },
    });
    std::fs::write(root.join("pen.json"), config.to_string()).unwrap();
    let (plugins, _) = plugins(&root);
    let error = plugins.compiler(&root.join("main.blot")).err().expect("no blotc granted").to_string();
    assert!(error.contains("blotc"), "{error}");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn builds_again_when_a_watched_file_changes() {
    let Some((component, compiler, std)) = parts() else {
        eprintln!("skipped: no Blot plugin component or blot2 checkout");
        return;
    };
    let root = checkout(&component, &compiler, &std);
    let (plugins, _) = plugins(&root);
    let (sender, outcomes) = std::sync::mpsc::channel();
    let watching = pen_plugins::watch(plugins, &root.join("main.blot"), Vec::new(), move |outcome| {
        let _ = sender.send(match outcome {
            Outcome::Built(built) => Ok(built.wasm),
            Outcome::Rejected(diagnostics) => Err(format!("{diagnostics:?}")),
            Outcome::Failed(error) => Err(error),
        });
    })
    .unwrap();
    let wait = || outcomes.recv_timeout(std::time::Duration::from_secs(60)).expect("a build");
    let first = wait().expect("it compiles");
    // An edit to a file the program imports builds it again, differently.
    std::fs::write(root.join("lib.blot"), "const base: U32 = 99\n").unwrap();
    let second = wait().expect("the edit compiles");
    assert_ne!(first, second);
    // A broken edit is reported, and fixing it builds again.
    std::fs::write(root.join("lib.blot"), "const base: Bool = 99\n").unwrap();
    assert!(wait().is_err());
    std::fs::write(root.join("lib.blot"), LIB).unwrap();
    assert_eq!(wait().expect("the fix compiles"), first);
    drop(watching);
    let _ = std::fs::remove_dir_all(&root);
}
