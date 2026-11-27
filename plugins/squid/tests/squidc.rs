//! The client against a real Squid compiler: SQUID_TEST_COMPILER when set,
//! else squid-lang's packaged one (the sibling checkout's
//! `generated/compiler/squidc.gz`, unpacked into pen's temporary directory),
//! with squid-lang's standard library. The packaged compiler is Linux's.
#![cfg(unix)]
use std::hash::{DefaultHasher, Hasher};
use std::io::Read;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use pen_plugin_squid::native::{NativeChild, NativeFiles};
use pen_plugin_squid::{Build, Diagnostic, Expectation, Overlay, Program, Session, Settings, find_tests};
use serde_json::{Value, json};

/// The sibling checkouts: squid-lang and gdev.
fn sibling(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..").join(name)
}

/// pen's temporary directory, on disk rather than in a RAM-backed /tmp.
fn temporary() -> PathBuf {
    let cache = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|cache| cache.is_absolute())
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").expect("a home directory")).join(".cache"));
    let directory = cache.join("pen/tmp");
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// The compiler the tests run. The packaged one is unpacked once per
/// version, named as pen's sweep of its temporary directory expects, and
/// kept fresh while it is used.
fn compiler() -> &'static Path {
    static COMPILER: OnceLock<PathBuf> = OnceLock::new();
    COMPILER.get_or_init(|| {
        if let Some(compiler) = std::env::var_os("SQUID_TEST_COMPILER") {
            return PathBuf::from(compiler);
        }
        let packed =
            std::fs::read(sibling("squid-lang/generated/compiler/squidc.gz")).expect("squid-lang's packaged compiler");
        let mut hasher = DefaultHasher::new();
        hasher.write(&packed);
        let path = temporary().join(format!("squidc-pen-plugin-squid-{:016x}", hasher.finish()));
        if let Ok(file) = std::fs::File::open(&path) {
            let _ = file.set_modified(std::time::SystemTime::now());
            return path;
        }
        let mut unpacked = Vec::new();
        flate2::read::GzDecoder::new(&packed[..]).read_to_end(&mut unpacked).unwrap();
        let partial = path.with_extension(format!("{}.partial", std::process::id()));
        std::fs::write(&partial, unpacked).unwrap();
        std::fs::set_permissions(&partial, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::rename(&partial, &path).unwrap();
        path
    })
}

/// Whether squid-lang's compiler and standard library are here. Without the
/// sibling checkout (CI's Linux job has none) a test skips, saying why; with
/// PEN_REQUIRE_FIXTURES=1 (CI's game job, which checks squid-lang out) it fails.
fn present() -> bool {
    let compiler = std::env::var_os("SQUID_TEST_COMPILER").is_some()
        || sibling("squid-lang/generated/compiler/squidc.gz").is_file();
    if compiler && sibling("squid-lang/std").is_dir() {
        return true;
    }
    let required = std::env::var_os("PEN_REQUIRE_FIXTURES").is_some_and(|value| !value.is_empty() && value != "0");
    assert!(!required, "no ../squid-lang compiler and std, and PEN_REQUIRE_FIXTURES is set");
    eprintln!("skipped: no ../squid-lang compiler and std");
    false
}

/// squid-lang's standard library.
fn std_root() -> String {
    std::fs::canonicalize(sibling("squid-lang/std")).unwrap().display().to_string()
}

/// A program's files in a directory of their own, removed when dropped.
struct Project {
    root: PathBuf,
}

impl Project {
    fn new(files: &[(&str, &str)]) -> Project {
        static PROJECTS: AtomicUsize = AtomicUsize::new(0);
        let name = format!("pen-plugin-squid-{}-{}", std::process::id(), PROJECTS.fetch_add(1, Ordering::Relaxed));
        let root = temporary().join(name);
        std::fs::create_dir_all(root.join("cache")).unwrap();
        let root = std::fs::canonicalize(root).unwrap();
        for (name, contents) in files {
            std::fs::write(root.join(name), contents).unwrap();
        }
        Project { root }
    }

    fn path(&self, name: &str) -> String {
        self.root.join(name).display().to_string()
    }

    fn open(&self, settings: Value) -> Result<Session<NativeChild>, String> {
        open(&self.root, &self.path("main.sq"), settings, &mut None)
    }
}

impl Drop for Project {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// Opens `entry` with the test compiler as `squidc`, keeping its process id
/// in `id`. Its cache is the checkout's, under `root`.
fn open(root: &Path, entry: &str, settings: Value, id: &mut Option<u32>) -> Result<Session<NativeChild>, String> {
    let program = Program { root: root.display().to_string(), entry: entry.into(), settings: settings.to_string() };
    let cache = root.join("cache").display().to_string();
    Session::open(&program, &cache, |name, args, env| {
        assert_eq!(name, "squidc");
        let child = NativeChild::spawn(compiler(), args, env, root).map_err(|error| error.to_string())?;
        *id = Some(child.id());
        Ok(child)
    })
}

const MAIN: &str = "import * as lib from \"./lib\"\n\nio const answer = func () => return lib.base + 1\n";
const LIB: &str = "const base: U32 = 41\n";

/// What a build made, which must be a module.
fn built(build: Result<Build, String>) -> Vec<u8> {
    match build {
        Ok(Build::Built(wasm)) => {
            assert!(wasm.starts_with(b"\0asm"), "not Wasm: {:?}", &wasm[..wasm.len().min(8)]);
            wasm
        }
        Ok(Build::Rejected(diagnostics)) => panic!("rejected: {diagnostics:#?}"),
        Err(error) => panic!("failed: {error}"),
    }
}

/// Why a build was rejected.
fn rejected(build: Result<Build, String>) -> Vec<Diagnostic> {
    match build {
        Ok(Build::Rejected(diagnostics)) => diagnostics,
        Ok(Build::Built(_)) => panic!("built"),
        Err(error) => panic!("failed: {error}"),
    }
}

#[test]
fn says_hello_opens_and_closes() {
    if !present() {
        return;
    }
    let project = Project::new(&[("main.sq", MAIN), ("lib.sq", LIB)]);
    let mut session = project.open(json!({ "std": std_root() })).unwrap();
    assert_eq!(session.identity().len(), 64);
    assert_eq!(session.revision(), 0);
    let std = std_root();
    assert_eq!(session.watched(), [project.root.display().to_string(), std.clone(), format!("{std}/prelude.sq")]);
    session.close().unwrap();
    assert!(session.build(&[]).unwrap_err().contains("closed"));
}

#[test]
fn builds_a_program() {
    if !present() {
        return;
    }
    let project = Project::new(&[("main.sq", MAIN), ("lib.sq", LIB)]);
    let mut session = project.open(json!({ "std": std_root() })).unwrap();
    built(session.build(&[]));
    assert_eq!(session.revision(), 1);
    // Unchanged, it builds the same module again.
    let wasm = built(session.build(&[]));
    assert_eq!(built(session.build(&[])), wasm);
    assert_eq!(session.revision(), 3);
    // Closing saves the restart cache, and the compiler exits.
    session.close().unwrap();
}

#[test]
fn native_assets_survive_edits_rejections_and_restarts() {
    if !present() {
        return;
    }
    let project = Project::new(&[
        (
            "main.sq",
            "import * as settings from \"./settings.json\"\nio const answer = func () => return (settings.value).answer\n",
        ),
        ("settings.json", "{\"answer\":41}"),
    ]);
    let settings = json!({
        "std": std_root(),
        "nativeAssets": {"imports": [{"module":"settings.json.sq", "path":"settings.json", "kind":"json"}]},
    });
    let mut session = project.open(settings.clone()).unwrap();
    let first = built(session.build(&[]));
    assert!(session.watched().contains(&project.path("settings.json")));
    std::fs::write(project.path("settings.json"), "{\"answer\":99}").unwrap();
    let changed = built(session.build(&[]));
    assert_ne!(first, changed);
    std::fs::write(project.path("settings.json"), "{broken}").unwrap();
    let problems = rejected(session.build(&[]));
    assert_eq!(problems[0].stage, "asset");
    assert_eq!(problems[0].file, project.path("settings.json"));
    std::fs::write(project.path("settings.json"), "{\"answer\":41}").unwrap();
    assert_eq!(built(session.build(&[])), first);
    session.close().unwrap();
    session.close().unwrap();
    assert_eq!(built(project.open(settings).unwrap().build(&[])), first);
}

#[test]
fn rejects_a_type_error_where_it_is() {
    if !present() {
        return;
    }
    let main =
        "import * as lib from \"./lib\"\n\nconst wrong: Bool = lib.base\n\nio const answer = func () => return wrong\n";
    let project = Project::new(&[("main.sq", main), ("lib.sq", LIB)]);
    let mut session = project.open(json!({ "std": std_root() })).unwrap();
    let diagnostics = rejected(session.build(&[]));
    let diagnostic = &diagnostics[0];
    assert_eq!(diagnostic.file, project.path("main.sq"));
    assert!(diagnostic.stage.starts_with("check"), "{diagnostic:?}");
    assert_eq!(diagnostic.code, "type_mismatch");
    assert!(!diagnostic.message.is_empty());
    let (start, end) = (diagnostic.start as usize, diagnostic.end as usize);
    assert!(start < end && end <= main.len(), "{diagnostic:?}");
    assert!(main[start..end].contains("wrong: Bool"), "{:?}", &main[start..end]);
    // A rejected build leaves the revision, and the session goes on.
    assert_eq!(session.revision(), 0);
    let fixed = Overlay { path: project.path("main.sq"), contents: Some(MAIN.into()) };
    built(session.build(&[fixed]));
    assert_eq!(session.revision(), 1);
}

#[test]
fn builds_overlays_over_the_files_for_one_build() {
    if !present() {
        return;
    }
    let project = Project::new(&[("main.sq", MAIN), ("lib.sq", LIB)]);
    let mut session = project.open(json!({ "std": std_root() })).unwrap();
    let on_disk = built(session.build(&[]));
    let other = Overlay { path: project.path("lib.sq"), contents: Some("const base: U32 = 99\n".into()) };
    let overlaid = built(session.build(&[other]));
    assert_ne!(overlaid, on_disk);
    assert_eq!(session.revision(), 2);
    // The next build reads the disk again.
    assert_eq!(built(session.build(&[])), on_disk);
    // An overlay can be a file that is not on disk, by a path from the root.
    let main = "import * as extra from \"./extra\"\n\nio const answer = func () => return extra.base\n";
    let virtual_files = [
        Overlay { path: "main.sq".into(), contents: Some(main.into()) },
        Overlay { path: "extra.sq".into(), contents: Some("const base: U32 = 7\n".into()) },
    ];
    built(session.build(&virtual_files));
    assert_eq!(session.revision(), 4);
}

#[test]
fn an_overlay_without_contents_hides_a_file() {
    if !present() {
        return;
    }
    let project = Project::new(&[("main.sq", MAIN), ("lib.sq", LIB)]);
    let mut session = project.open(json!({ "std": std_root() })).unwrap();
    let hidden = Overlay { path: project.path("lib.sq"), contents: None };
    let diagnostics = rejected(session.build(&[hidden]));
    assert_eq!((diagnostics[0].code.as_str(), diagnostics[0].stage.as_str()), ("missing_file", "load"));
    // At the import of it.
    let (start, end) = (diagnostics[0].start as usize, diagnostics[0].end as usize);
    assert_eq!(&MAIN[start..end], "\"./lib\"");
    assert_eq!(session.revision(), 0);
    built(session.build(&[]));
}

#[test]
fn a_killed_compiler_fails_and_stays_failed() {
    if !present() {
        return;
    }
    let project = Project::new(&[("main.sq", MAIN), ("lib.sq", LIB)]);
    let mut id = None;
    let mut session = open(&project.root, &project.path("main.sq"), json!({ "std": std_root() }), &mut id).unwrap();
    built(session.build(&[]));
    let killed = Command::new("kill").args(["-9", &id.unwrap().to_string()]).status().unwrap();
    assert!(killed.success());
    let error = session.build(&[]).unwrap_err();
    eprintln!("{error}");
    assert!(error.starts_with("the Squid compiler"), "{error}");
    let again = session.build(&[]).unwrap_err();
    assert!(again.contains("stopped after an earlier failure"), "{again}");
    // pen then opens a new session, which builds.
    built(project.open(json!({ "std": std_root() })).unwrap().build(&[]));
}

#[test]
fn refuses_a_compiler_it_did_not_expect() {
    if !present() {
        return;
    }
    let project = Project::new(&[("main.sq", MAIN), ("lib.sq", LIB)]);
    let identity = "0".repeat(64);
    let error = project.open(json!({ "std": std_root(), "expectedIdentity": identity })).err().unwrap();
    assert!(error.contains("not the expected"), "{error}");
}

/// The sibling gdev checkout's `tests/math.sq`, laid out as pen's
/// `test-squid` has the plugin lay it out, and compiled alone as it says: in
/// place of its configured test entry, with gdev's compiler settings.
/// Skipped without the checkout.
fn gdev_math() -> Option<(PathBuf, Vec<u8>)> {
    if !present() {
        return None;
    }
    let gdev = std::fs::canonicalize(sibling("gdev")).ok()?;
    let math = gdev.join("tests/math.sq");
    let Ok(config) = std::fs::read_to_string(gdev.join("pen.json")) else {
        eprintln!("skipped: no ../gdev/pen.json");
        return None;
    };
    if !math.is_file() {
        eprintln!("skipped: no ../gdev/tests/math.sq");
        return None;
    }
    let config: Value = serde_json::from_str(&config).unwrap();
    let settings = config["compiler"]["settings"].to_string();
    let root = gdev.display().to_string();
    let main = Program { root: root.clone(), entry: format!("{root}/src/main.sq"), settings: settings.clone() };
    let mut modules = find_tests(&main, &[math.display().to_string()], None, &mut NativeFiles).unwrap();
    let test = modules.pop().unwrap();
    let layout = Settings::parse(&settings, &root).unwrap().tests;
    let entry = layout.entry.unwrap_or_else(|| format!("{root}/src/test_fixture.sq"));
    assert_eq!((test.entry.as_str(), &test.expect), (entry.as_str(), &Expectation::Checks));
    let cache = Project::new(&[]);
    let program = Program { root, entry: test.entry, settings };
    let mut session = Session::open(&program, &cache.path("cache"), |_, args, env| {
        NativeChild::spawn(compiler(), args, env, &gdev).map_err(|error| error.to_string())
    })
    .unwrap();
    let wasm = built(session.build(&test.overlays));
    Some((gdev, wasm))
}

#[test]
fn compiles_gdevs_math_tests() {
    gdev_math();
}
