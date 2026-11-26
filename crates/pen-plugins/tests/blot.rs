//! The Blot compiler plugin (plugins/blot) hosted for real: pen.json grants it
//! blot2's packaged blotc, which it starts through the host's process
//! interface and drives over `serve-project`, and it lays out a checkout's
//! tests. Skipped without the plugin's component (`just plugins`), or, where
//! they compile, the sibling blot2 checkout.
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pen_plugins::{BuildResult, Builder, Expectation, Level, Outcome, Overlay, Plugins};

const MAIN: &str = "import * as lib from \"./lib\"\n\nio const answer = func () => return lib.base + 1\n";
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
    let compiler = std::env::var_os("BLOT_TEST_COMPILER")
        .map(PathBuf::from)
        .unwrap_or_else(|| blot2.join("generated/compiler/blotc.gz"));
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
        "import * as lib from \"./lib\"\n\nconst wrong: Bool = lib.base\n\nio const answer = func () => return wrong\n";
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
        ("tests/b.blot", "io const test_b = func () => return 0\n"),
        ("tests/a.blot", "io const test_a = func () => return 0\n"),
        ("tests/traps/t.blot", "io const test_t = func () => return 0\n"),
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
    assert_eq!(modules[0].overlays[0].contents.as_deref(), Some("io const test_a = func () => return 0\n"));
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
    std::fs::write(&outside, "io const test_o = func () => return 0\n").unwrap();
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
    let test = "import * as lib from \"./lib\"\n\nio const test_base = func () => return lib.base\n";
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

#[test]
fn library_tests_compile_relative_helpers_without_a_game() {
    let Some((component, compiler, std)) = parts() else {
        eprintln!("skipped: no Blot plugin component or blot2 checkout");
        return;
    };
    let root = checkout(&component, &compiler, &std);
    std::fs::remove_file(root.join("main.blot")).unwrap();
    std::fs::create_dir_all(root.join("tests")).unwrap();
    std::fs::write(root.join("tests/helper.blot"), "const value: F32 = 42.0\n").unwrap();
    std::fs::write(
        root.join("tests/check.blot"),
        "import * as helper from \"./helper\"\nio const test_answer = func () => return #[helper.value]\n",
    )
    .unwrap();
    // An unrelated fixture must not get compiled as a standalone test.
    std::fs::write(root.join("tests/not-a-test.blot"), "not valid Blot").unwrap();
    let config_path = root.join("pen.json");
    let mut config: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    config["compiler"]["settings"]["tests"] = serde_json::json!({
        "entry": "tests/.fixture.blot", "files": ["tests/check.blot"]
    });
    std::fs::write(config_path, config.to_string()).unwrap();
    let (plugins, _) = plugins(&root);
    let modules = plugins.tests(&root.join("src/main.blot"), &[], None).unwrap();
    assert_eq!(modules.len(), 1);
    assert_eq!(Path::new(&modules[0].entry), root.join("tests/.fixture.blot"));
    assert!(!root.join("tests/.fixture.blot").exists());
    let mut builder = Builder::new(plugins, Path::new(&modules[0].entry));
    match builder.build(&modules[0].overlays) {
        Outcome::Built(built) => assert_eq!(&built.wasm[..4], b"\0asm"),
        Outcome::Rejected(diagnostics) => panic!("relative helper did not compile: {diagnostics:?}"),
        Outcome::Failed(error) => panic!("relative helper did not compile: {error}"),
    }
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

#[cfg(unix)]
#[test]
fn stage0_wgsl_and_csv_generators_cache_adapt_and_report_provenance() {
    use std::os::unix::fs::PermissionsExt;

    let Some((component, compiler, std)) = parts() else {
        eprintln!("skipped: no Blot plugin component or blot2 checkout");
        return;
    };
    let examples = repository().join("../blot2/examples/generators");
    if !examples.join("wgsl_gen.blot").is_file() {
        eprintln!("skipped: no blot2/examples/generators");
        return;
    }

    let root = empty();
    std::fs::create_dir_all(root.join("tools")).unwrap();
    std::fs::create_dir_all(root.join("assets")).unwrap();
    std::fs::create_dir_all(root.join("raw")).unwrap();
    std::fs::create_dir_all(root.join("data")).unwrap();

    // Unpack blotc if `.gz` so our counting wrapper runs the exact compiler binary.
    let real_compiler = if compiler.extension().is_some_and(|e| e == "gz") {
        let unpacked = root.join("blotc-bin");
        let packed = std::fs::read(&compiler).unwrap();
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(&packed[..]), &mut bytes).unwrap();
        std::fs::write(&unpacked, bytes).unwrap();
        std::fs::set_permissions(&unpacked, std::fs::Permissions::from_mode(0o755)).unwrap();
        unpacked
    } else {
        compiler
    };
    let spawn_log = root.join("spawns.log");
    let wrapper = root.join("blotc-wrap");
    std::fs::write(
        &wrapper,
        format!(
            "#!/bin/sh\nprintf 'spawn\\n' >> '{}'\nexec '{}' \"$@\"\n",
            spawn_log.display(),
            real_compiler.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
    let spawn_count = || -> usize { std::fs::read_to_string(&spawn_log).map(|s| s.lines().count()).unwrap_or(0) };

    std::fs::copy(examples.join("wgsl_gen.blot"), root.join("tools/wgsl_gen.blot")).unwrap();
    std::fs::copy(examples.join("csv_gen.blot"), root.join("tools/csv_gen.blot")).unwrap();
    std::fs::copy(examples.join("literals.blot"), root.join("tools/literals.blot")).unwrap();
    // Record-input `{ path, text }` -> plain `Text` generator.
    std::fs::write(
        root.join("tools/path_gen.blot"),
        r#"import * as text from "std/text"
private io const make_meta = func (input: { path: Text, text: Text }): Text =>
  let out = text.builder ()
  out := text.append "const path_len: U32 = " self
  out := text.append_u32 input.path.length self
  out := text.append "\nconst text_len: U32 = " self
  out := text.append_u32 input.text.length self
  out := text.append "\n" self
  return text.finish out
"#,
    )
    .unwrap();
    // Direct `bytes` -> `Text` generator (no Result wrapper).
    std::fs::write(
        root.join("tools/bytes_gen.blot"),
        r#"import * as text from "std/text"
io generate (raw: #[U32]): Text =>
  let total: U32 = 0
  for b in raw:
    total := self + b
  let out = text.builder ()
  out := text.append "const checksum: U32 = " self
  out := text.append_u32 total self
  out := text.append "\n" self
  return text.finish out
"#,
    )
    .unwrap();

    let wgsl = r#"struct Parameters { clear: f32, active: bool };
const stage_light_slot = 16u;
const building_marker = -1.0;
"#;
    std::fs::write(root.join("assets/salt.wgsl"), wgsl).unwrap();
    std::fs::write(root.join("raw/units.csv"), "name,hp,speed\nScout,25,3.5\nTank,100,1.25\n").unwrap();
    std::fs::write(root.join("raw/note.txt"), "hello").unwrap();
    std::fs::write(root.join("raw/blob.bin"), [1u8, 2, 3, 4]).unwrap();

    std::fs::write(
        root.join("main.blot"),
        r#"import { Parameters, stage_light_slot, building_marker } from "./assets/salt.wgsl"
import { specs } from "./data/units.csv"
import { path_len, text_len } from "./data/note.txt"
import { checksum } from "./data/blob.bin"

io summary (): U32 =>
  let p = $Parameters { clear: 1.0, active: true }
  let bonus = if p.active then stage_light_slot else 0
  let mark = if building_marker < 0.0 then 100 else 0
  return bonus + mark + specs[0].hp + specs[1].hp + text_len + checksum + (if path_len > 0 then 1 else 0)
"#,
    )
    .unwrap();

    let config = serde_json::json!({
        "plugins": {
            "blot": {
                "wasm": component,
                "read": [std],
                "run": { "blotc": wrapper },
            },
        },
        "compiler": {
            "plugin": "blot",
            "settings": {
                "std": std,
                "generators": [
                    { "match": "assets/*.wgsl", "generator": "tools/wgsl_gen.blot", "inputMode": "text" },
                    { "module": "data/units.csv", "input": "raw/units.csv", "generator": "tools/csv_gen.blot", "inputMode": "text" },
                    { "module": "data/note.txt", "input": "raw/note.txt", "generator": "tools/path_gen.blot", "exportName": "make_meta", "inputMode": "text" },
                    { "module": "data/blob.bin", "input": "raw/blob.bin", "generator": "tools/bytes_gen.blot" }
                ]
            },
        },
    });
    std::fs::write(root.join("pen.json"), config.to_string()).unwrap();
    let root = root.canonicalize().unwrap();
    let (plugins, _) = plugins(&root);
    let entry = root.join("main.blot");
    let mut session = plugins.compiler(&entry).unwrap();

    // Build 1: 1 Stage-1 session + 4 Stage-0 compilations = 5 blotc spawns.
    let BuildResult::Built(wasm1) = session.build(&[]).unwrap() else { panic!("build 1 must succeed") };
    assert_eq!(&wasm1[..4], b"\0asm");
    let spawns_after_1 = spawn_count();
    assert_eq!(spawns_after_1, 5, "expected 1 Stage-1 + 4 Stage-0 spawns");

    // Build 2 (unchanged): Level-1 & Level-2 cache hits -> zero extra spawns, byte-identical Wasm.
    let BuildResult::Built(wasm2) = session.build(&[]).unwrap() else { panic!("build 2 must succeed") };
    assert_eq!(wasm2, wasm1);
    assert_eq!(spawn_count(), spawns_after_1);

    // Build 3 (edit asset input only): Level-1 cache hit -> zero Stage-0 recompiles, updated Wasm!
    std::fs::write(root.join("assets/salt.wgsl"), wgsl.replace("16u", "24u")).unwrap();
    let BuildResult::Built(wasm3) = session.build(&[]).unwrap() else { panic!("build 3 must succeed") };
    assert_ne!(wasm3, wasm1);
    assert_eq!(spawn_count(), spawns_after_1, "editing .wgsl must reuse compiled Stage-0 Wasm");

    // Build 4 (edit generator itself): recompiles only wgsl_gen.blot (+1 spawn).
    let wgsl_gen_path = root.join("tools/wgsl_gen.blot");
    let orig_gen = std::fs::read_to_string(&wgsl_gen_path).unwrap();
    std::fs::write(&wgsl_gen_path, orig_gen.replace("text.from_u32 value", "text.from_u32 (value + 1)")).unwrap();
    let BuildResult::Built(wasm4) = session.build(&[]).unwrap() else { panic!("build 4 must succeed") };
    assert_ne!(wasm4, wasm3);
    assert_eq!(spawn_count(), spawns_after_1 + 1, "editing wgsl_gen.blot must recompile Stage 0 once");

    // Build 5 (edit transitive import `literals.blot`): invalidates generators importing it.
    let lit_path = root.join("tools/literals.blot");
    let orig_lit = std::fs::read_to_string(&lit_path).unwrap();
    std::fs::write(&lit_path, orig_lit.replace("if start == 1:", "if false:")).unwrap();
    let BuildResult::Built(wasm5) = session.build(&[]).unwrap() else { panic!("build 5 must succeed") };
    assert_ne!(wasm5, wasm4);
    assert!(spawn_count() > spawns_after_1 + 1, "editing transitive import must recompile Stage 0");

    // Restore valid generator/helper, then verify `$Err` diagnostic provenance.
    std::fs::write(&wgsl_gen_path, &orig_gen).unwrap();
    std::fs::write(&lit_path, &orig_lit).unwrap();
    std::fs::write(root.join("assets/salt.wgsl"), "struct Parameters { clear: texture_2d };\n").unwrap();
    let BuildResult::Rejected(diags) = session.build(&[]).unwrap() else { panic!("malformed wgsl must reject") };
    assert_eq!(diags[0].stage, "asset");
    assert_eq!(diags[0].code, "asset_import");
    assert!(diags[0].message.contains("Unsupported WGSL field type in Parameters"), "{diags:?}");
    let details: serde_json::Value = serde_json::from_str(diags[0].details.as_deref().unwrap()).unwrap();
    assert_eq!(details["provenance"]["phase"], "result");

    // Verify Stage-0 runtime `@panic` and bounds trap provenance (decoded `blot:traps` v1 + exact generator spans).
    std::fs::write(root.join("assets/salt.wgsl"), wgsl).unwrap();
    std::fs::write(root.join("tools/bytes_gen.blot"), "io generate (raw: #[U32]): Text =>\n  @panic \"boom stage0\"\n")
        .unwrap();
    let BuildResult::Rejected(diags) = session.build(&[]).unwrap() else { panic!("panicking generator must reject") };
    assert_eq!(diags[0].code, "asset_import");
    assert!(diags[0].message.contains("panic: \"boom stage0\""), "{diags:?}");
    assert_eq!(Path::new(&diags[0].file), root.join("tools/bytes_gen.blot"), "{diags:?}");
    assert!(diags[0].end > diags[0].start, "expected non-empty panic trap span: {diags:?}");
    let details: serde_json::Value = serde_json::from_str(diags[0].details.as_deref().unwrap()).unwrap();
    assert_eq!(details["provenance"]["phase"], "execute");
    assert_eq!(Path::new(details["provenance"]["generator"].as_str().unwrap()), root.join("tools/bytes_gen.blot"));

    std::fs::write(
        root.join("tools/bytes_gen.blot"),
        "io generate (raw: #[U32]): Text =>\n  let _ = raw[999]\n  return \"\"\n",
    )
    .unwrap();
    let BuildResult::Rejected(diags) = session.build(&[]).unwrap() else { panic!("OOB generator must reject") };
    assert_eq!(diags[0].code, "asset_import");
    assert!(diags[0].message.contains("index out of bounds"), "{diags:?}");
    assert_eq!(Path::new(&diags[0].file), root.join("tools/bytes_gen.blot"), "{diags:?}");
    assert!(diags[0].end > diags[0].start, "expected non-empty bounds trap span: {diags:?}");

    // Verify Stage-1 error inside a generated virtual module carries `virtual_source` + `provenance`.
    std::fs::write(
        root.join("tools/bytes_gen.blot"),
        "io generate (raw: #[U32]): Text => return \"const checksum: Bool = 123\\n\"\n",
    )
    .unwrap();
    let BuildResult::Rejected(diags) = session.build(&[]).unwrap() else { panic!("bad generated Blot must reject") };
    let vdiag = diags.iter().find(|d| d.file.ends_with("data/blob.bin.blot")).expect("virtual module diag");
    let details: serde_json::Value = serde_json::from_str(vdiag.details.as_deref().unwrap()).unwrap();
    assert!(details["virtual_source"].as_str().unwrap().contains("const checksum: Bool = 123"));
    assert_eq!(details["provenance"]["phase"], "generated_source");

    let watched = session.watched().unwrap();
    assert!(watched.contains(&root.join("tools/wgsl_gen.blot")), "{watched:?}");
    assert!(watched.contains(&root.join("tools/literals.blot")), "{watched:?}");
    assert!(watched.contains(&root.join("assets")), "{watched:?}");

    drop(session);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn stage0_watch_rebuilds_on_asset_edit_and_generator_edit() {
    let Some((component, compiler, std)) = parts() else {
        eprintln!("skipped: no Blot plugin component or blot2 checkout");
        return;
    };
    let examples = repository().join("../blot2/examples/generators");
    if !examples.join("wgsl_gen.blot").is_file() {
        eprintln!("skipped: no blot2/examples/generators");
        return;
    }

    let root = empty();
    std::fs::create_dir_all(root.join("tools")).unwrap();
    std::fs::create_dir_all(root.join("assets")).unwrap();
    std::fs::copy(examples.join("wgsl_gen.blot"), root.join("tools/wgsl_gen.blot")).unwrap();
    std::fs::copy(examples.join("literals.blot"), root.join("tools/literals.blot")).unwrap();
    std::fs::write(root.join("assets/salt.wgsl"), "const slot = 10u;\n").unwrap();
    std::fs::write(
        root.join("main.blot"),
        "import { slot } from \"./assets/salt.wgsl\"\nio answer (): U32 => return slot\n",
    )
    .unwrap();
    let config = serde_json::json!({
        "plugins": { "blot": { "wasm": component, "read": [std], "run": { "blotc": compiler } } },
        "compiler": {
            "plugin": "blot",
            "settings": {
                "std": std,
                "generators": [{ "match": "assets/*.wgsl", "generator": "tools/wgsl_gen.blot", "inputMode": "text" }]
            }
        }
    });
    std::fs::write(root.join("pen.json"), config.to_string()).unwrap();
    let root = root.canonicalize().unwrap();
    let (plugins, _) = plugins(&root);
    let (outcomes, built) = std::sync::mpsc::channel();
    let (changes, changed) = std::sync::mpsc::channel();
    let watching =
        pen_plugins::watch_changes(plugins, &root.join("main.blot"), Vec::new(), changed, move |outcome, watched| {
            let wasm = match outcome {
                Outcome::Built(b) => Ok(b.wasm),
                Outcome::Rejected(d) => Err(format!("{d:?}")),
                Outcome::Failed(e) => Err(e),
            };
            let _ = outcomes.send((wasm, watched.clone()));
        })
        .unwrap();

    let (first, watched) = built.recv_timeout(Duration::from_secs(60)).expect("initial build");
    let wasm1 = first.expect("initial build succeeds");
    assert!(watched.contains(&root.join("assets")), "{watched:?}");
    assert!(watched.contains(&root.join("tools/wgsl_gen.blot")), "{watched:?}");
    assert!(watched.contains(&root.join("tools/literals.blot")), "{watched:?}");

    // Editing the WGSL file triggers an automatic rebuild with new Wasm.
    std::fs::write(root.join("assets/salt.wgsl"), "const slot = 99u;\n").unwrap();
    changes.send(vec![root.join("assets/salt.wgsl")]).unwrap();
    let (second, _) = built.recv_timeout(Duration::from_secs(60)).expect("asset rebuild");
    let wasm2 = second.expect("asset edit builds");
    assert_ne!(wasm2, wasm1);

    // Editing the Stage-0 generator itself also triggers a rebuild.
    let gen_src = std::fs::read_to_string(root.join("tools/wgsl_gen.blot")).unwrap();
    std::fs::write(
        root.join("tools/wgsl_gen.blot"),
        gen_src.replace("text.from_u32 value", "text.from_u32 (value + 5)"),
    )
    .unwrap();
    changes.send(vec![root.join("tools/wgsl_gen.blot")]).unwrap();
    let (third, _) = built.recv_timeout(Duration::from_secs(60)).expect("generator rebuild");
    let wasm3 = third.expect("generator edit builds");
    assert_ne!(wasm3, wasm2);

    drop(changes);
    drop(watching);
    let _ = std::fs::remove_dir_all(&root);
}
