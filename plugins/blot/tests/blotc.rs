//! The client against a real Blot compiler: BLOT_TEST_COMPILER when set,
//! else blot2's packaged one (the sibling checkout's
//! `generated/compiler/blotc.gz`, unpacked into pen's temporary directory),
//! with blot2's standard library. The packaged compiler is Linux's.
#![cfg(unix)]
use std::hash::{DefaultHasher, Hasher};
use std::io::Read;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use pen_plugin_blot::native::NativeChild;
use pen_plugin_blot::{Build, Diagnostic, Overlay, Program, Session};
use serde_json::{Value, json};

/// The sibling checkouts: blot2 and gdev.
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
        if let Some(compiler) = std::env::var_os("BLOT_TEST_COMPILER") {
            return PathBuf::from(compiler);
        }
        let packed = std::fs::read(sibling("blot2/generated/compiler/blotc.gz")).expect("blot2's packaged compiler");
        let mut hasher = DefaultHasher::new();
        hasher.write(&packed);
        let path = temporary().join(format!("blotc-pen-plugin-blot-{:016x}", hasher.finish()));
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

/// blot2's standard library.
fn std_root() -> String {
    std::fs::canonicalize(sibling("blot2/std")).unwrap().display().to_string()
}

/// A program's files in a directory of their own, removed when dropped.
struct Project {
    root: PathBuf,
}

impl Project {
    fn new(files: &[(&str, &str)]) -> Project {
        static PROJECTS: AtomicUsize = AtomicUsize::new(0);
        let name = format!("pen-plugin-blot-{}-{}", std::process::id(), PROJECTS.fetch_add(1, Ordering::Relaxed));
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
        open(&self.root, &self.path("main.blot"), settings, &mut None)
    }
}

impl Drop for Project {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// Opens `entry` with the test compiler as `blotc`, keeping its process id
/// in `id`. Its cache is the checkout's, under `root`.
fn open(root: &Path, entry: &str, settings: Value, id: &mut Option<u32>) -> Result<Session<NativeChild>, String> {
    let program = Program { root: root.display().to_string(), entry: entry.into(), settings: settings.to_string() };
    let cache = root.join("cache").display().to_string();
    Session::open(&program, &cache, |name, args, env| {
        assert_eq!(name, "blotc");
        let child = NativeChild::spawn(compiler(), args, env, root).map_err(|error| error.to_string())?;
        *id = Some(child.id());
        Ok(child)
    })
}

const MAIN: &str = "import * as lib from \"./lib\"\n\nentry const answer = fn () => lib.base + 1\n";
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
    let project = Project::new(&[("main.blot", MAIN), ("lib.blot", LIB)]);
    let mut session = project.open(json!({ "std": std_root() })).unwrap();
    assert_eq!(session.identity().len(), 64);
    assert_eq!(session.revision(), 0);
    let std = std_root();
    assert_eq!(session.watched(), [project.root.display().to_string(), std.clone(), format!("{std}/prelude.blot")]);
    session.close().unwrap();
    assert!(session.build(&[]).unwrap_err().contains("closed"));
}

#[test]
fn builds_a_program() {
    let project = Project::new(&[("main.blot", MAIN), ("lib.blot", LIB)]);
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
fn rejects_a_type_error_where_it_is() {
    let main =
        "import * as lib from \"./lib\"\n\nconst wrong: Bool = lib.base\n\nentry const answer = fn () => wrong\n";
    let project = Project::new(&[("main.blot", main), ("lib.blot", LIB)]);
    let mut session = project.open(json!({ "std": std_root() })).unwrap();
    let diagnostics = rejected(session.build(&[]));
    let diagnostic = &diagnostics[0];
    assert_eq!(diagnostic.file, project.path("main.blot"));
    assert!(diagnostic.stage.starts_with("check"), "{diagnostic:?}");
    assert_eq!(diagnostic.code, "type_mismatch");
    assert!(!diagnostic.message.is_empty());
    let (start, end) = (diagnostic.start as usize, diagnostic.end as usize);
    assert!(start < end && end <= main.len(), "{diagnostic:?}");
    assert!(main[start..end].contains("wrong: Bool"), "{:?}", &main[start..end]);
    // A rejected build leaves the revision, and the session goes on.
    assert_eq!(session.revision(), 0);
    let fixed = Overlay { path: project.path("main.blot"), contents: Some(MAIN.into()) };
    built(session.build(&[fixed]));
    assert_eq!(session.revision(), 1);
}

#[test]
fn builds_overlays_over_the_files_for_one_build() {
    let project = Project::new(&[("main.blot", MAIN), ("lib.blot", LIB)]);
    let mut session = project.open(json!({ "std": std_root() })).unwrap();
    let on_disk = built(session.build(&[]));
    let other = Overlay { path: project.path("lib.blot"), contents: Some("const base: U32 = 99\n".into()) };
    let overlaid = built(session.build(&[other]));
    assert_ne!(overlaid, on_disk);
    assert_eq!(session.revision(), 2);
    // The next build reads the disk again.
    assert_eq!(built(session.build(&[])), on_disk);
    // An overlay can be a file that is not on disk, by a path from the root.
    let main = "import * as extra from \"./extra\"\n\nentry const answer = fn () => extra.base\n";
    let virtual_files = [
        Overlay { path: "main.blot".into(), contents: Some(main.into()) },
        Overlay { path: "extra.blot".into(), contents: Some("const base: U32 = 7\n".into()) },
    ];
    built(session.build(&virtual_files));
    assert_eq!(session.revision(), 4);
}

#[test]
fn an_overlay_without_contents_hides_a_file() {
    let project = Project::new(&[("main.blot", MAIN), ("lib.blot", LIB)]);
    let mut session = project.open(json!({ "std": std_root() })).unwrap();
    let hidden = Overlay { path: project.path("lib.blot"), contents: None };
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
    let project = Project::new(&[("main.blot", MAIN), ("lib.blot", LIB)]);
    let mut id = None;
    let mut session = open(&project.root, &project.path("main.blot"), json!({ "std": std_root() }), &mut id).unwrap();
    built(session.build(&[]));
    let killed = Command::new("kill").args(["-9", &id.unwrap().to_string()]).status().unwrap();
    assert!(killed.success());
    let error = session.build(&[]).unwrap_err();
    eprintln!("{error}");
    assert!(error.starts_with("the Blot compiler"), "{error}");
    let again = session.build(&[]).unwrap_err();
    assert!(again.contains("stopped after an earlier failure"), "{again}");
    // pen then opens a new session, which builds.
    built(project.open(json!({ "std": std_root() })).unwrap().build(&[]));
}

#[test]
fn refuses_a_compiler_it_did_not_expect() {
    let project = Project::new(&[("main.blot", MAIN), ("lib.blot", LIB)]);
    let identity = "0".repeat(64);
    let error = project.open(json!({ "std": std_root(), "expectedIdentity": identity })).err().unwrap();
    assert!(error.contains("not the expected"), "{error}");
}

/// The sibling gdev checkout's `tests/math.blot`, as pen's `test-blot`
/// compiles it alone: in place of `src/test_fixture.blot`, with gdev's
/// packages behind `gdev/`. Skipped without the checkout.
fn gdev_math() -> Option<(PathBuf, Vec<u8>)> {
    let gdev = std::fs::canonicalize(sibling("gdev")).ok()?;
    let math = gdev.join("tests/math.blot");
    let Ok(source) = std::fs::read_to_string(&math) else {
        eprintln!("skipped: no ../gdev/tests/math.blot");
        return None;
    };
    let fixture = gdev.join("src/test_fixture.blot").display().to_string();
    let settings = json!({ "imports": { "gdev/": "packages/" }, "std": std_root() });
    let cache = Project::new(&[]);
    let program = Program { root: gdev.display().to_string(), entry: fixture.clone(), settings: settings.to_string() };
    let mut session = Session::open(&program, &cache.path("cache"), |_, args, env| {
        NativeChild::spawn(compiler(), args, env, &gdev).map_err(|error| error.to_string())
    })
    .unwrap();
    let wasm = built(session.build(&[Overlay { path: fixture, contents: Some(source) }]));
    Some((gdev, wasm))
}

#[test]
fn compiles_gdevs_math_tests() {
    gdev_math();
}

/// What `pen test-blot` compiles gdev's math tests to today, through Deno
/// (crates/pen/scripts/blot_test_build.ts), is what the plugin compiles them
/// to. Run with `--ignored`: it needs Deno, and gdev's Deno packages.
#[test]
#[ignore]
fn compiles_gdevs_math_tests_as_deno_does() {
    let Some((gdev, wasm)) = gdev_math() else { return };
    let lock = std::fs::read(gdev.join("deno.lock")).ok();
    let script = std::fs::canonicalize(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/pen/scripts/blot_test_build.ts"),
    )
    .expect("pen's Blot test build script");
    let out = Project::new(&[]);
    let deno_wasm = out.root.join("math.wasm");
    let mut command = Command::new("deno");
    command
        .args(["run", "-A", "--no-lock", "--config"])
        .arg(gdev.join("deno.json"))
        .arg(&script)
        .arg(&gdev)
        .arg(gdev.join("tests/math.blot"))
        .arg("alone")
        .arg(&deno_wasm)
        .current_dir(&gdev)
        .env("TMPDIR", temporary());
    if let Some(compiler) = std::env::var_os("BLOT_TEST_COMPILER") {
        command.env("GDEV_BLOT_COMPILER", compiler);
    }
    let output = command.output().expect("deno");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(std::fs::read(gdev.join("deno.lock")).ok(), lock, "gdev's deno.lock changed");
    let deno = std::fs::read(&deno_wasm).unwrap();
    eprintln!("the plugin: {} bytes; Deno: {} bytes", wasm.len(), deno.len());
    assert!(wasm == deno, "the plugin's module differs from Deno's");
}
