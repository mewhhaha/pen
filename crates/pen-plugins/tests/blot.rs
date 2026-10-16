//! The Blot compiler plugin (plugins/blot) hosted for real: pen.json grants it
//! blot2's packaged blotc, which it starts through the host's process
//! interface and drives over `serve-project`, and it lays out a checkout's
//! tests. Skipped without the plugin's component (`just plugins`), or, where
//! they compile, the sibling blot2 checkout.
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pen_plugins::{BuildResult, Builder, Expectation, Level, Outcome, Overlay, Plugins};

const MAIN: &str = "import * as lib from \"./lib\"\n\nentry const answer = fn () => lib.base + 1\n";
const LIB: &str = "const base: U32 = 41\n";

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// The plugin's component, if it is there.
fn component() -> Option<PathBuf> {
    [
        std::env::var_os("PEN_TEST_BLOT_PLUGIN").map(PathBuf::from),
        Some(repository().join("plugins/target/wasm32-wasip2/release/pen_plugin_blot.wasm")),
    ]
    .into_iter()
    .flatten()
    .find(|path| path.is_file())
}

/// The plugin's component and blot2's compiler and std, if they are there.
fn parts() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let component = component()?;
    let blot2 = repository().join("../blot2").canonicalize().ok()?;
    let compiler = blot2.join("generated/compiler/blotc.gz");
    let std = blot2.join("std");
    (compiler.is_file() && std.is_dir()).then_some((component, compiler, std))
}

/// An empty checkout in pen's cache, for this test.
fn empty() -> PathBuf {
    let root = dirs::cache_dir().unwrap().join(format!(
        "pen/tmp/pen-plugins-blot-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

/// A checkout of two files and its pen.json, in pen's cache.
fn checkout(component: &Path, compiler: &Path, std: &Path) -> PathBuf {
    let root = empty();
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
    let wrong =
        "import * as lib from \"./lib\"\n\nconst wrong: Bool = lib.base\n\nentry const answer = fn () => wrong\n";
    let overlay = Overlay { path: entry.display().to_string(), contents: Some(wrong.into()) };
    let BuildResult::Rejected(diagnostics) = session.build(&[overlay]).unwrap() else { panic!("the overlay rejects") };
    assert_eq!(diagnostics[0].file, entry.display().to_string());
    assert!(diagnostics[0].end > diagnostics[0].start);
    let BuildResult::Built(again) = session.build(&[]).unwrap() else { panic!("the disk compiles again") };
    assert_eq!(again, wasm);
    let watched = session.watched().unwrap();
    assert!(watched.contains(&root), "{watched:?}");
    drop(session);
    // The component, compiled once, is in pen's cache.
    let components = dirs::cache_dir().unwrap().join("pen/components");
    let recent = std::fs::read_dir(&components).unwrap().flatten().any(|entry| {
        let modified = entry.metadata().and_then(|metadata| metadata.modified()).unwrap();
        entry.path().extension().is_some_and(|extension| extension == "cwasm")
            && modified.elapsed().unwrap_or_default() < Duration::from_secs(600)
    });
    assert!(recent, "no compiled component in {}", components.display());
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

/// A pen.json that grants the plugin `run` as blotc, with `timeout`.
fn granting(root: &Path, component: &Path, run: &Path, timeout: Option<f64>) {
    let mut config = serde_json::json!({
        "plugins": { "blot": { "wasm": component, "run": { "blotc": run } } },
        "compiler": { "plugin": "blot" },
    });
    if let Some(timeout) = timeout {
        config["plugins"]["blot"]["timeout"] = timeout.into();
    }
    std::fs::write(root.join("pen.json"), config.to_string()).unwrap();
}

#[test]
fn lays_out_the_checkouts_tests() {
    let Some(component) = component() else {
        eprintln!("skipped: no Blot plugin component");
        return;
    };
    let root = empty();
    for (file, source) in [
        ("src/main.blot", MAIN),
        ("tests/b.blot", "entry const test_b = fn () => 0\n"),
        ("tests/a.blot", "entry const test_a = fn () => 0\n"),
        ("tests/traps/t.blot", "entry const test_t = fn () => 0\n"),
        ("tests/rejected/r.blot", "// rejects: type_mismatch\nconst a: Bool = 1\n"),
        ("tests/support/check.blot", "const a = 1\n"),
    ] {
        std::fs::create_dir_all(root.join(file).parent().unwrap()).unwrap();
        std::fs::write(root.join(file), source).unwrap();
    }
    // Any program will do: nothing starts it.
    granting(&root, &component, &root.join("src/main.blot"), None);
    let (plugins, _) = plugins(&root);
    let entry = root.join("src/main.blot");
    let modules = plugins.tests(&entry, &[], None).unwrap();
    let shown: Vec<&str> =
        modules.iter().map(|module| module.path.strip_prefix(&format!("{}/", root.display())).unwrap()).collect();
    assert_eq!(shown, ["tests/a.blot", "tests/b.blot", "tests/traps/t.blot", "tests/rejected/r.blot"]);
    let fixture = root.join("src/test_fixture.blot").display().to_string();
    assert!(modules.iter().all(|module| module.entry == fixture && module.prefix == "test_"));
    assert_eq!(modules[0].overlays[0].path, fixture);
    assert_eq!(modules[0].overlays[0].contents.as_deref(), Some("entry const test_a = fn () => 0\n"));
    assert!(matches!(modules[2].expect, Expectation::Traps));
    assert!(matches!(&modules[3].expect, Expectation::Rejected(text) if text == "type_mismatch"));
    // Given modules, and a filter.
    let given = [root.join("tests/b.blot")];
    assert_eq!(plugins.tests(&entry, &given, None).unwrap().len(), 1);
    let filtered = plugins.tests(&entry, &[], Some("test_t")).unwrap();
    assert!(filtered.len() == 1 && matches!(filtered[0].expect, Expectation::Traps));
    // A module that must not compile says why.
    std::fs::write(root.join("tests/rejected/r.blot"), "const a: Bool = 1\n").unwrap();
    let error = plugins.tests(&entry, &[], None).unwrap_err().to_string();
    assert!(error.contains("tests/rejected/r.blot must not compile"), "{error}");
    // Modules outside what the checkout grants are not read.
    let outside = root.with_extension("outside.blot");
    std::fs::write(&outside, "entry const test_o = fn () => 0\n").unwrap();
    let error = plugins.tests(&entry, std::slice::from_ref(&outside), None).unwrap_err().to_string();
    let _ = std::fs::remove_file(&outside);
    assert!(error.contains("outside what pen.json lets this plugin read"), "{error}");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn builds_a_test_module_as_its_entry() {
    let Some((component, compiler, std)) = parts() else {
        eprintln!("skipped: no Blot plugin component or blot2 checkout");
        return;
    };
    let root = checkout(&component, &compiler, &std);
    std::fs::create_dir_all(root.join("tests")).unwrap();
    let test = "import * as lib from \"./lib\"\n\nentry const test_base = fn () => lib.base\n";
    std::fs::write(root.join("tests/base.blot"), test).unwrap();
    let (plugins, _) = plugins(&root);
    let modules = plugins.tests(&root.join("main.blot"), &[], None).unwrap();
    assert_eq!(modules.len(), 1);
    // It stands in beside the entry, so `./lib` is the checkout's.
    let mut builder = Builder::new(plugins, Path::new(&modules[0].entry));
    let Outcome::Built(built) = builder.build(&modules[0].overlays) else { panic!("the test module compiles") };
    assert_eq!(&built.wasm[..4], b"\0asm");
    let _ = builder.close();
    let _ = std::fs::remove_dir_all(&root);
}

#[cfg(unix)]
#[test]
fn stops_a_compiler_that_hangs_at_the_calls_deadline() {
    use std::os::unix::fs::PermissionsExt;
    let Some(component) = component() else {
        eprintln!("skipped: no Blot plugin component");
        return;
    };
    let root = empty();
    std::fs::write(root.join("main.blot"), MAIN).unwrap();
    // A compiler that never says hello.
    let hung = root.join("hung");
    std::fs::write(&hung, "#!/bin/sh\nexec sleep 600\n").unwrap();
    std::fs::set_permissions(&hung, std::fs::Permissions::from_mode(0o755)).unwrap();
    granting(&root, &component, &hung, Some(1.0));
    let (plugins, _) = plugins(&root);
    let started = Instant::now();
    let error = format!("{:#}", plugins.compiler(&root.join("main.blot")).err().expect("it hangs"));
    assert!(error.contains("did not answer in time"), "{error}");
    assert!(started.elapsed() < Duration::from_secs(4), "{:?}", started.elapsed());
    // A build fails as the toolchain's, after a fresh compiler's one more try.
    let started = Instant::now();
    let Outcome::Failed(error) = pen_plugins::build(plugins, &root.join("main.blot")) else { panic!("it built") };
    assert!(error.contains("did not answer in time"), "{error}");
    assert!(started.elapsed() < Duration::from_secs(8), "{:?}", started.elapsed());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn builds_again_on_the_changes_a_callers_watcher_sends() {
    let Some((component, compiler, std)) = parts() else {
        eprintln!("skipped: no Blot plugin component or blot2 checkout");
        return;
    };
    let root = checkout(&component, &compiler, &std);
    let (plugins, _) = plugins(&root);
    let (outcomes, built) = std::sync::mpsc::channel();
    let (changes, changed) = std::sync::mpsc::channel();
    let ignored = vec![root.join(".pen")];
    let watching =
        pen_plugins::watch_changes(plugins, &root.join("main.blot"), ignored, changed, move |outcome, watched| {
            let _ = outcomes.send((matches!(outcome, Outcome::Built(_)), watched.clone()));
        })
        .unwrap();
    let wait = |within: u64| built.recv_timeout(Duration::from_secs(within));
    let (first, watched) = wait(60).expect("a build");
    assert!(first && watched.contains(&root), "{watched:?}");
    // Changes it does not watch, or ignores, build nothing.
    changes.send(vec![root.with_extension("elsewhere"), root.join(".pen/game.wasm")]).unwrap();
    assert!(wait(1).is_err());
    // A change it watches builds again.
    std::fs::write(root.join("lib.blot"), "const base: U32 = 99\n").unwrap();
    changes.send(vec![root.join("lib.blot")]).unwrap();
    assert!(wait(60).expect("a build again").0);
    // Closing the changes ends the builds.
    drop(changes);
    drop(watching);
    let _ = std::fs::remove_dir_all(&root);
}
