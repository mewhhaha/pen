//! `pen test-blot`: a gdev checkout's Blot tests (its PEN_MIGRATION.md,
//! phase 2). Each test module (`tests/*.blot`) compiles to Wasm: against the
//! game on the tests' level when its first line is `// test: game`, else
//! alone, as if it stood at `src/test_fixture.blot`. Every `test_…` entry it
//! exports, Unit to Array F32, runs in a fresh instance and returns the
//! report of its checks (gdev's `packages/check.blot`): the checks made, the
//! failures, then each failure's place (from 1), its expected lanes' count
//! and lanes, and its actual lanes' count and lanes.
//!
//! The entries take no capability, so nothing a test runs calls the host.
//! Should a module also export a function that takes one (`main(host)`),
//! blot-host links its adapter as it does for the game, and no test calls it.
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use blot_host::{Guest, GuestFunction, GuestModule, Parameter, Value, ValueType};
use wasmtime::Engine;

const BUILD_SCRIPT: &str = include_str!("../scripts/blot_test_build.ts");

/// The first line of a module that compiles against the game.
const AGAINST_GAME: &str = "// test: game";

/// What the names of the entries that are tests start with.
const TEST_PREFIX: &str = "test_";

pub struct Options {
    /// The modules to run; every `tests/*.blot` of the checkout when empty.
    pub files: Vec<PathBuf>,
    /// Only the tests whose names contain this.
    pub filter: Option<String>,
}

/// Runs the modules and prints each test's outcome; fails when a module
/// does not compile or a test fails or traps.
pub fn run(checkout: &Path, options: Options) -> Result<()> {
    ensure!(
        blot_host::COMPILER,
        "this pen is a player runtime, built without the Wasm compiler: it cannot run a checkout's \
         tests (use a full build, `just install`)"
    );
    let modules = if options.files.is_empty() {
        test_modules(checkout)?
    } else {
        options.files.iter().map(|file| locate(checkout, file)).collect::<Result<_>>()?
    };
    ensure!(!modules.is_empty(), "no Blot test modules in {}", checkout.join("tests").display());
    // Each module that compiles against the game repeats it; the
    // development engine compiles each function it has seen only once.
    let engine = blot_host::dev_engine()?;
    let started = Instant::now();
    let (mut passed, mut ran, mut broken) = (0, 0, 0);
    let mut failed = Vec::new();
    for module in &modules {
        let shown = module.strip_prefix(checkout).unwrap_or(module).display().to_string();
        // A test's name is in its module's source: a module without the
        // filter's text has no test the filter lets through.
        if let Some(filter) = &options.filter
            && std::fs::read_to_string(module).is_ok_and(|source| !source.contains(filter.as_str()))
        {
            continue;
        }
        ran += 1;
        eprintln!("{shown}");
        let compiling = Instant::now();
        let outcome =
            compile_test_module(checkout, module).and_then(|wasm| run_tests(&engine, &wasm, options.filter.as_deref()));
        let tests = match outcome {
            Ok(tests) => tests,
            Err(error) => {
                eprintln!("  FAILED to compile:\n{}", indent(&format!("{error:#}"), 4));
                broken += 1;
                failed.push(format!("{shown} (did not compile)"));
                continue;
            }
        };
        let blot = compiling.elapsed().saturating_sub(tests.compile + tests.run);
        for skipped in &tests.skipped {
            eprintln!("  warning: skipped {skipped}");
        }
        if tests.tests.is_empty() {
            eprintln!("  no {TEST_PREFIX}… entries{}", if options.filter.is_some() { " match" } else { "" });
        }
        let mut module_passed = 0;
        for (name, outcome) in &tests.tests {
            for line in describe(name, outcome) {
                eprintln!("{line}");
            }
            if outcome.passed() {
                module_passed += 1;
            } else {
                failed.push(format!("{shown}: {name}"));
            }
        }
        passed += module_passed;
        eprintln!(
            "  {module_passed} passed, {} failed (Blot {:.1} s, Wasm {}, tests {})",
            tests.tests.len() - module_passed,
            blot.as_secs_f64(),
            milliseconds(tests.compile),
            milliseconds(tests.run),
        );
    }
    let did_not_compile = if broken > 0 { format!("; {broken} did not compile") } else { String::new() };
    eprintln!(
        "\n{passed} passed, {} failed, in {ran} module{}{did_not_compile} ({:.1} s)",
        failed.len() - broken,
        if ran == 1 { "" } else { "s" },
        started.elapsed().as_secs_f64(),
    );
    if !failed.is_empty() {
        bail!("Blot tests failed:\n  {}", failed.join("\n  "));
    }
    Ok(())
}

/// Every `tests/*.blot` of the checkout, sorted.
fn test_modules(checkout: &Path) -> Result<Vec<PathBuf>> {
    let directory = checkout.join("tests");
    let mut found: Vec<PathBuf> = std::fs::read_dir(&directory)
        .with_context(|| format!("reading {}", directory.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|extension| extension == "blot") && path.is_file())
        .collect();
    found.sort();
    Ok(found)
}

/// A module named on the command line: from the current directory, else
/// from the checkout.
fn locate(checkout: &Path, file: &Path) -> Result<PathBuf> {
    [std::path::absolute(file)?, checkout.join(file)]
        .into_iter()
        .find(|path| path.is_file())
        .with_context(|| format!("there is no Blot test module {}", file.display()))
}

/// Whether a module compiles against the game.
fn against_game(source: &str) -> bool {
    source.lines().next() == Some(AGAINST_GAME)
}

/// Compiles the Blot test module at `module` to Wasm with `checkout`'s
/// compiler setup: against the game on the tests' level when its first line
/// is `// test: game` (gdev's `test_support/compile.ts`, `gameFixture`),
/// else alone, as if it stood at `src/test_fixture.blot`. A module that does
/// not compile fails with the compiler's diagnostics.
///
/// For now this runs the checkout's TypeScript compiler setup through Deno,
/// from a script embedded here, as `project.rs` builds the game; a Rust
/// client of the Blot compiler is to take its place.
pub fn compile_test_module(checkout: &Path, module: &Path) -> Result<Vec<u8>> {
    static BUILDS: AtomicUsize = AtomicUsize::new(0);
    let checkout = std::path::absolute(checkout)?;
    let module = std::path::absolute(module)?;
    let source = std::fs::read_to_string(&module).with_context(|| format!("reading {}", module.display()))?;
    let mode = if against_game(&source) { "game" } else { "alone" };
    let directory = checkout.join(".pen/blot-tests");
    std::fs::create_dir_all(&directory)?;
    let out = directory.join(format!("{}-{}.wasm", std::process::id(), BUILDS.fetch_add(1, Ordering::Relaxed)));
    // The Blot compiler unpacks itself into the temporary directory: the
    // cache directory's, where what a killed one leaves is swept.
    let temporary = crate::project::temporary_dir()?;
    let mut command = Command::new("deno");
    command
        .args(["run", "-A", "--no-lock", "--config"])
        .arg(checkout.join("deno.json"))
        .arg(script()?)
        .arg(&checkout)
        .arg(&module)
        .arg(mode)
        .arg(&out)
        .current_dir(&checkout)
        .env("TMPDIR", &temporary)
        .stdin(Stdio::null());
    #[cfg(windows)]
    command.env("TMP", &temporary).env("TEMP", &temporary);
    let output = command.output().context("running deno, which compiles Blot test modules")?;
    let bytes = std::fs::read(&out);
    let _ = std::fs::remove_file(&out);
    if !output.status.success() {
        let diagnostics = String::from_utf8_lossy(&output.stderr);
        let diagnostics = diagnostics.trim();
        if diagnostics.is_empty() {
            bail!("deno exited with {} while compiling {}", output.status, module.display());
        }
        bail!("{diagnostics}");
    }
    bytes.with_context(|| format!("reading the compiled {}", out.display()))
}

fn pen_directory() -> Result<PathBuf> {
    let directory = dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("pen");
    std::fs::create_dir_all(&directory)?;
    Ok(directory)
}

/// The embedded build script, written out for Deno beside project.rs's.
fn script() -> Result<PathBuf> {
    let path = pen_directory()?.join(format!("blot_test_build-{}.ts", env!("CARGO_PKG_VERSION")));
    if std::fs::read_to_string(&path).ok().as_deref() != Some(BUILD_SCRIPT) {
        // In one rename, since other runs may be reading it.
        let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&temporary, BUILD_SCRIPT)?;
        std::fs::rename(&temporary, &path)?;
    }
    Ok(path)
}

/// A module's tests, run.
struct Tests {
    /// Each test's name and outcome, in the module's order.
    tests: Vec<(String, Outcome)>,
    /// The `test_…` exports that are not Unit to Array F32, and what they are.
    skipped: Vec<String>,
    /// Compiling the Wasm, and running the tests.
    compile: Duration,
    run: Duration,
}

#[derive(Debug, PartialEq)]
enum Outcome {
    /// What it returned: it passed when no check failed.
    Reported(Report),
    /// It trapped, or what it returned is not a report.
    Broken(String),
}

impl Outcome {
    fn passed(&self) -> bool {
        matches!(self, Outcome::Reported(report) if report.failures.is_empty())
    }
}

/// Compiles `wasm` and runs its `test_…` entries whose names contain
/// `filter`, each in an instance of its own, on a thread with the stack a
/// guest needs.
fn run_tests(engine: &Engine, wasm: &[u8], filter: Option<&str>) -> Result<Tests> {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("pen-test".into())
            .stack_size(blot_host::GUEST_STACK_BYTES + (8 << 20))
            .spawn_scoped(scope, || run_tests_here(engine, wasm, filter))?
            .join()
            .map_err(|_| anyhow!("the test thread panicked"))?
    })
}

fn run_tests_here(engine: &Engine, wasm: &[u8], filter: Option<&str>) -> Result<Tests> {
    let started = Instant::now();
    let module = GuestModule::new(engine, wasm, None)?;
    let compile = started.elapsed();
    let started = Instant::now();
    let mut tests = Vec::new();
    let mut skipped = Vec::new();
    let entries = module.abi().functions.iter().filter(|function| {
        function.name.starts_with(TEST_PREFIX) && filter.is_none_or(|filter| function.name.contains(filter))
    });
    for function in entries {
        if function.parameter != Parameter::Value(ValueType::Unit) || function.result != ValueType::ArrayF32 {
            skipped.push(format!("{}: it is {}, not Unit -> Array F32", function.name, signature(function)));
            continue;
        }
        let called = Guest::new(engine, &module).and_then(|mut guest| guest.call(&function.name, Value::Unit));
        let outcome = match called.and_then(Value::into_f32_array) {
            Ok(lanes) => match decode(&lanes) {
                Ok(report) => Outcome::Reported(report),
                Err(error) => Outcome::Broken(format!("its report is malformed ({error:#}): {}", format_lanes(&lanes))),
            },
            Err(error) => Outcome::Broken(explain(&error)),
        };
        tests.push((function.name.clone(), outcome));
    }
    Ok(Tests { tests, skipped, compile, run: started.elapsed() })
}

fn signature(function: &GuestFunction) -> String {
    let parameter = match function.parameter {
        Parameter::Value(value) => value.to_string(),
        Parameter::Callback(callback) => format!("({} -> {} ! {{Foreign}})", callback.parameter, callback.result),
    };
    format!("{parameter} -> {}", function.result)
}

/// What `check.report` returns.
#[derive(Debug, PartialEq)]
struct Report {
    checks: usize,
    failures: Vec<Failure>,
}

#[derive(Debug, PartialEq)]
struct Failure {
    /// The check's place in its test, from 1.
    place: usize,
    expected: Vec<f32>,
    actual: Vec<f32>,
}

/// A report's lanes not yet read.
struct Lanes<'a>(&'a [f32]);

impl<'a> Lanes<'a> {
    fn take(&mut self, count: usize, what: &str) -> Result<&'a [f32]> {
        ensure!(count <= self.0.len(), "it ends within {what}");
        let (taken, rest) = self.0.split_at(count);
        self.0 = rest;
        Ok(taken)
    }

    fn count(&mut self, what: &str) -> Result<usize> {
        let lane = self.take(1, what)?[0];
        // Lanes count exactly up to 2^24.
        ensure!(lane >= 0.0 && lane.fract() == 0.0 && lane <= 16_777_216.0, "{what} is {lane}, not a count");
        Ok(lane as usize)
    }
}

/// Reads a report: the checks made, the failures, then each failure's
/// place, its expected lanes' count and lanes, and its actual lanes' count
/// and lanes.
fn decode(lanes: &[f32]) -> Result<Report> {
    let mut lanes = Lanes(lanes);
    let checks = lanes.count("the number of checks")?;
    let failed = lanes.count("the number of failures")?;
    ensure!(failed <= checks, "{failed} of {checks} checks failed");
    let mut failures = Vec::with_capacity(failed);
    for index in 1..=failed {
        let place = lanes.count(&format!("failure {index}'s place"))?;
        ensure!((1..=checks).contains(&place), "failure {index} is check {place} of {checks}");
        let count = lanes.count(&format!("failure {index}'s expected count"))?;
        let expected = lanes.take(count, &format!("failure {index}'s expected lanes"))?.to_vec();
        let count = lanes.count(&format!("failure {index}'s actual count"))?;
        let actual = lanes.take(count, &format!("failure {index}'s actual lanes"))?.to_vec();
        failures.push(Failure { place, expected, actual });
    }
    ensure!(lanes.0.is_empty(), "{} lanes follow its failures", lanes.0.len());
    Ok(Report { checks, failures })
}

/// A test's lines: `ok` or `FAILED`, and under a failure what each failed
/// check expected and got.
fn describe(name: &str, outcome: &Outcome) -> Vec<String> {
    match outcome {
        Outcome::Reported(report) if report.failures.is_empty() => vec![format!("  ok      {name}")],
        Outcome::Reported(report) => {
            let mut lines = vec![format!("  FAILED  {name} ({} of {} checks)", report.failures.len(), report.checks)];
            lines.extend(report.failures.iter().map(|failure| {
                format!(
                    "    check {}: expected {} got {}",
                    failure.place,
                    format_lanes(&failure.expected),
                    format_lanes(&failure.actual)
                )
            }));
            lines
        }
        Outcome::Broken(message) => vec![format!("  FAILED  {name}"), indent(message, 4)],
    }
}

/// An error's cause first (a trap's reason), then what it happened in (its
/// backtrace), a line each.
fn explain(error: &anyhow::Error) -> String {
    let mut chain: Vec<String> = error.chain().map(ToString::to_string).collect();
    chain.reverse();
    chain.join("\n")
}

fn format_lanes(lanes: &[f32]) -> String {
    let shown: Vec<String> = lanes.iter().take(64).map(f32::to_string).collect();
    let more =
        if lanes.len() > shown.len() { format!(", … {} more", lanes.len() - shown.len()) } else { String::new() };
    format!("[{}{more}]", shown.join(", "))
}

fn indent(text: &str, spaces: usize) -> String {
    let margin = " ".repeat(spaces);
    text.lines().map(|line| format!("{margin}{line}")).collect::<Vec<_>>().join("\n")
}

fn milliseconds(duration: Duration) -> String {
    format!("{:.0} ms", duration.as_secs_f64() * 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leb(mut value: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                bytes.push(byte);
                return bytes;
            }
            bytes.push(byte | 0x80);
        }
    }

    fn signed_leb(mut value: i32) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            let done = (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0);
            bytes.push(if done { byte } else { byte | 0x80 });
            if done {
                return bytes;
            }
        }
    }

    fn vector(items: Vec<Vec<u8>>) -> Vec<u8> {
        let mut bytes = leb(items.len() as u32);
        items.into_iter().for_each(|item| bytes.extend(item));
        bytes
    }

    fn name(text: &str) -> Vec<u8> {
        let mut bytes = leb(text.len() as u32);
        bytes.extend(text.as_bytes());
        bytes
    }

    fn section(id: u8, payload: Vec<u8>) -> Vec<u8> {
        let mut bytes = vec![id];
        bytes.extend(leb(payload.len() as u32));
        bytes.extend(payload);
        bytes
    }

    /// `i32.const at`, as a function body or an offset.
    fn constant(at: i32) -> Vec<u8> {
        [vec![0x41], signed_leb(at), vec![0x0b]].concat()
    }

    /// A Blot guest as the compiler lays one out, with these entries
    /// (`test_…` among them) and a `main(host)` that imports its adapter:
    /// each entry is a name, its manifest tags and its Wasm type and body.
    /// The arrays it returns lie at fixed places in its memory.
    fn guest(entries: &[(&str, &[u8], usize, Vec<u8>)], arrays: &[(i32, &[f32])]) -> Vec<u8> {
        let function_type = |parameters: &[u8], results: &[u8]| {
            [
                vec![0x60],
                vector(parameters.iter().map(|&p| vec![p]).collect()),
                vector(results.iter().map(|&r| vec![r]).collect()),
            ]
            .concat()
        };
        // 0: Unit, U32 or an array to one (and the arena's functions); 1: F32 to F32;
        // 2: the adapter; 3: main.
        let types = vector(vec![
            function_type(&[I32], &[I32]),
            function_type(&[F32], &[F32]),
            function_type(&[EXTERNREF, I32], &[I32]),
            function_type(&[EXTERNREF], &[I32]),
        ]);
        let imports =
            vector(vec![[name(blot_host::HOST_MODULE), name("call_array_f32_array_f32"), vec![0x00, 2]].concat()]);
        // Functions after the import: blot:allocate, blot:reset, main, then the entries.
        let mut bodies = vec![constant(1024), constant(0), constant(0)];
        let mut type_indices = vec![0, 0, 3];
        let mut exports = vec![
            [name("blot:memory"), vec![0x02, 0]].concat(),
            [name("blot:allocate"), vec![0x00, 1]].concat(),
            [name("blot:reset"), vec![0x00, 2]].concat(),
            [name("main"), vec![0x00, 3]].concat(),
        ];
        // main takes (Array F32 -> Array F32 ! {Foreign}) and returns Unit.
        let mut manifest = vec![[name("main"), vec![4, 6, 6, 0]].concat()];
        for (index, (entry, tags, type_index, body)) in entries.iter().enumerate() {
            bodies.push(body.clone());
            type_indices.push(*type_index as u32);
            exports.push([name(entry), vec![0x00], leb(4 + index as u32)].concat());
            manifest.push([name(entry), tags.to_vec()].concat());
        }
        let code =
            vector(bodies.into_iter().map(|body| [leb(body.len() as u32 + 1), vec![0], body].concat()).collect());
        let data = vector(
            arrays
                .iter()
                .map(|(at, lanes)| {
                    let mut bytes = (lanes.len() as u32).to_le_bytes().to_vec();
                    lanes.iter().for_each(|lane| bytes.extend(lane.to_le_bytes()));
                    [vec![0x00], constant(*at), vector(bytes.into_iter().map(|byte| vec![byte]).collect())].concat()
                })
                .collect(),
        );
        let abi = [name("blot:abi"), leb(2), vector(manifest), leb(0)].concat();
        [
            b"\0asm\x01\0\0\0".to_vec(),
            section(1, types),
            section(2, imports),
            section(3, vector(type_indices.into_iter().map(leb).collect())),
            section(5, vector(vec![vec![0x00, 1]])),
            section(7, vector(exports)),
            section(10, code),
            section(11, data),
            section(0, abi),
        ]
        .concat()
    }

    const I32: u8 = 0x7f;
    const F32: u8 = 0x7d;
    const EXTERNREF: u8 = 0x6f;
    /// Manifest tags: Unit to Array F32, and F32 to F32.
    const UNIT_TO_ARRAY: &[u8] = &[0, 6];
    const F32_TO_F32: &[u8] = &[3, 3];

    fn sample() -> Vec<u8> {
        guest(
            &[
                ("test_passes", UNIT_TO_ARRAY, 0, constant(8)),
                ("test_fails", UNIT_TO_ARRAY, 0, constant(24)),
                ("test_traps", UNIT_TO_ARRAY, 0, vec![0x00, 0x0b]),
                ("test_takes_f32", F32_TO_F32, 1, vec![0x20, 0, 0x0b]),
                ("failing_on_purpose", UNIT_TO_ARRAY, 0, constant(24)),
                ("test_malformed", UNIT_TO_ARRAY, 0, constant(64)),
            ],
            &[(8, &[2.0, 0.0]), (24, &[2.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 4.0]), (64, &[1.0, 1.0, 1.0, 5.0])],
        )
    }

    #[test]
    fn reads_reports() {
        assert_eq!(decode(&[0.0, 0.0]).unwrap(), Report { checks: 0, failures: vec![] });
        assert_eq!(decode(&[5.0, 0.0]).unwrap(), Report { checks: 5, failures: vec![] });
        // gdev's tests/hand_salt.blot's `failing_on_purpose`.
        assert_eq!(
            decode(&[2.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 4.0]).unwrap(),
            Report {
                checks: 2,
                failures: vec![Failure { place: 2, expected: vec![2.0, 3.0], actual: vec![2.0, 4.0] }]
            }
        );
        let two = decode(&[3.0, 2.0, 1.0, 0.0, 1.0, 0.5, 3.0, 1.0, 1.0, 1.0, 0.0]).unwrap();
        assert_eq!(two.failures[0], Failure { place: 1, expected: vec![], actual: vec![0.5] });
        assert_eq!(two.failures[1], Failure { place: 3, expected: vec![1.0], actual: vec![0.0] });
        for malformed in [
            &[][..],
            &[1.0],
            &[1.0, 2.0],
            &[1.0, 1.0],
            &[1.0, 1.0, 1.0, 2.0, 1.0],
            &[1.0, 1.0, 2.0, 0.0, 0.0],
            &[1.0, 0.0, 7.0],
            &[1.5, 0.0],
            &[-1.0, 0.0],
            &[f32::NAN, 0.0],
            &[f32::INFINITY, 0.0],
        ] {
            assert!(decode(malformed).is_err(), "{malformed:?}");
        }
    }

    #[test]
    fn describes_outcomes() {
        let passed = Outcome::Reported(Report { checks: 2, failures: vec![] });
        assert_eq!(describe("test_a", &passed), ["  ok      test_a"]);
        let failed = Outcome::Reported(decode(&[2.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 4.0]).unwrap());
        assert_eq!(
            describe("test_b", &failed),
            ["  FAILED  test_b (1 of 2 checks)", "    check 2: expected [2, 3] got [2, 4]"]
        );
        let near = Failure { place: 1, expected: vec![0.39, -0.0], actual: vec![1e-7, f32::NAN] };
        let near = Outcome::Reported(Report { checks: 1, failures: vec![near] });
        assert_eq!(describe("test_c", &near)[1], "    check 1: expected [0.39, -0] got [0.0000001, NaN]");
        let broken = Outcome::Broken("wasm trap\nunreachable".into());
        assert_eq!(describe("test_d", &broken), ["  FAILED  test_d", "    wasm trap\n    unreachable"]);
    }

    #[test]
    fn chooses_how_a_module_compiles() {
        assert!(against_game("// test: game\nimport * as check from \"gdev/check\"\n"));
        assert!(against_game("// test: game\r\n"));
        assert!(!against_game("// test: games\n"));
        assert!(!against_game("import * as salt from \"./salt\"\n// test: game\n"));
        assert!(!against_game(""));
    }

    #[test]
    fn runs_a_modules_tests() {
        let engine = blot_host::engine().unwrap();
        let tests = run_tests(&engine, &sample(), None).unwrap();
        let names: Vec<&str> = tests.tests.iter().map(|(name, _)| name.as_str()).collect();
        // Only `test_…` entries, Unit to Array F32, in the module's order.
        assert_eq!(names, ["test_passes", "test_fails", "test_traps", "test_malformed"]);
        assert_eq!(tests.skipped, ["test_takes_f32: it is F32 -> F32, not Unit -> Array F32"]);
        let outcome = |name: &str| &tests.tests.iter().find(|(test, _)| test == name).unwrap().1;
        assert_eq!(*outcome("test_passes"), Outcome::Reported(Report { checks: 2, failures: vec![] }));
        assert!(outcome("test_passes").passed());
        assert_eq!(describe("test_fails", outcome("test_fails"))[1], "    check 2: expected [2, 3] got [2, 4]");
        let Outcome::Broken(trap) = outcome("test_traps") else { panic!("test_traps did not trap") };
        assert!(trap.starts_with("wasm trap: wasm `unreachable`"), "{trap}");
        let Outcome::Broken(malformed) = outcome("test_malformed") else { panic!("test_malformed passed") };
        assert!(malformed.starts_with("its report is malformed (it ends within failure 1's expected"), "{malformed}");
        assert!(malformed.ends_with("[1, 1, 1, 5]"), "{malformed}");

        let filtered = run_tests(&engine, &sample(), Some("fail")).unwrap();
        assert_eq!(filtered.tests.len(), 1);
        assert_eq!(filtered.tests[0].0, "test_fails");
        assert!(run_tests(&engine, b"\0asm\x01\0\0\0", None).is_err());
    }

    #[test]
    fn runs_the_checkouts_pilot() {
        // The sibling gdev checkout's pilot module, when this repo has one
        // and Deno is there to compile it.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../gdev");
        let pilot = root.join("tests/hand_salt.blot");
        if !pilot.is_file() || Command::new("deno").arg("--version").output().is_err() {
            eprintln!("skipped: no ../gdev/tests/hand_salt.blot, or no deno");
            return;
        }
        let wasm = compile_test_module(&root, &pilot).unwrap();
        let engine = blot_host::dev_engine().unwrap();
        let tests = run_tests(&engine, &wasm, None).unwrap();
        assert!(!tests.tests.is_empty());
        for (name, outcome) in &tests.tests {
            assert!(outcome.passed(), "{}", describe(name, outcome).join("\n"));
        }
        // A test that fails on purpose (and is no `test_…`, so it does not run).
        let module = GuestModule::new(&engine, &wasm, None).unwrap();
        if module.abi().function("failing_on_purpose").is_some() {
            let lanes = std::thread::scope(|scope| {
                std::thread::Builder::new()
                    .stack_size(blot_host::GUEST_STACK_BYTES + (8 << 20))
                    .spawn_scoped(scope, || {
                        Guest::new(&engine, &module).unwrap().call("failing_on_purpose", Value::Unit).unwrap()
                    })
                    .unwrap()
                    .join()
                    .unwrap()
            });
            let outcome = Outcome::Reported(decode(&lanes.into_f32_array().unwrap()).unwrap());
            assert!(!outcome.passed());
            let lines = describe("failing_on_purpose", &outcome);
            eprintln!("{}", lines.join("\n"));
            assert!(lines[1].starts_with("    check "), "{lines:?}");
        }
        // A module that does not compile fails with the compiler's diagnostics.
        let broken = std::env::temp_dir().join(format!("pen-broken-{}.blot", std::process::id()));
        std::fs::write(&broken, "entry const test_broken = fn () => missing_name\n").unwrap();
        let error = compile_test_module(&root, &broken).unwrap_err();
        let _ = std::fs::remove_file(&broken);
        assert!(format!("{error:#}").contains("missing_name"), "{error:#}");
    }
}
