//! `pen test-blot`: a gdev checkout's Blot tests (its PEN_MIGRATION.md,
//! phase 2). Each test module compiles to Wasm through the checkout's
//! compiler plugin (its `pen.json`, `pen-plugins`), as if it stood at
//! `src/test_fixture.blot`, with the game's asset modules. Each worker keeps
//! one compiler session from module to module, so each compiles
//! incrementally. Its directory says what its tests must do:
//!
//! - `tests/*.blot`: every `test_…` entry it exports, Unit to Array F32,
//!   runs in a fresh instance and returns the report of its checks (gdev's
//!   `packages/check.blot`): the checks made, the failures, then each
//!   failure's place (from 1), its expected lanes' count and lanes, and its
//!   actual lanes' count and lanes. It passes when no check failed.
//! - `tests/traps/*.blot`: every such entry must trap at `unreachable`, as
//!   Blot's `@panic` stops a program (a decoder refusing bad input). Any
//!   other trap, or returning, fails.
//! - `tests/rejected/*.blot`: the module must not compile, and the
//!   compiler's diagnostics must contain the text of its `// rejects: TEXT`
//!   line, among the comment lines it opens with.
//!
//! Never `tests/blocked/` (tests waiting on compiler fixes) nor
//! `tests/codec_support/` (modules the tests import). Up to `--jobs` modules
//! compile at once, each running its tests once compiled, and the modules
//! print in order.
//!
//! The entries take no capability, so nothing a test runs calls the host.
//! Should a module also export a function that takes one (`main(host)`),
//! blot-host links its adapter as it does for the game, and no test calls it.
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use blot_host::{Guest, GuestFunction, GuestModule, Parameter, Value, ValueType};
use pen_plugins::{Builder, Outcome as Building, Overlay, Plugins};
use wasmtime::{Engine, Trap};

/// Where each test module compiles, as if it stood there, beside the game's
/// own sources.
const FIXTURE: &str = "src/test_fixture.blot";

/// What the line of a module that must not compile starts with, before the
/// text its diagnostics must contain.
const REJECTS: &str = "// rejects:";

/// What the names of the entries that are tests start with.
const TEST_PREFIX: &str = "test_";

pub struct Options {
    /// The modules to run; every module of the checkout's `tests/`,
    /// `tests/traps/` and `tests/rejected/` when empty.
    pub files: Vec<PathBuf>,
    /// Only the tests whose names contain this.
    pub filter: Option<String>,
    /// How many modules compile at once; about half the cores, at most 8,
    /// when None.
    pub jobs: Option<usize>,
}

/// What a module's tests must do, by the directory it is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// `tests/`: each `test_…` entry returns a report whose checks pass.
    Checks,
    /// `tests/traps/`: each `test_…` entry traps at `unreachable`.
    Traps,
    /// `tests/rejected/`: the module does not compile, for the reason its
    /// `// rejects:` line gives.
    Rejected,
}

impl Kind {
    /// In the order they run.
    const ALL: [Kind; 3] = [Kind::Checks, Kind::Traps, Kind::Rejected];

    /// A module's kind, by its directory's name.
    fn of(module: &Path) -> Kind {
        match module.parent().and_then(Path::file_name).and_then(|name| name.to_str()) {
            Some("traps") => Kind::Traps,
            Some("rejected") => Kind::Rejected,
            _ => Kind::Checks,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Kind::Checks => "checks",
            Kind::Traps => "must trap",
            Kind::Rejected => "must not compile",
        }
    }
}

/// A module to run.
struct Module {
    path: PathBuf,
    /// Its path as shown: from the checkout.
    shown: String,
    kind: Kind,
    /// What a module that must not compile has its diagnostics contain.
    rejects: Option<String>,
}

impl Module {
    /// The module at `path`, or None when `filter` lets none of its tests
    /// through. One that must not compile has to say what its diagnostics
    /// contain.
    fn read(checkout: &Path, path: PathBuf, filter: Option<&str>) -> Result<Option<Module>> {
        let source = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let shown = path.strip_prefix(checkout).unwrap_or(&path).display().to_string();
        let kind = Kind::of(&path);
        // A module that must not compile is one test, named for its file;
        // the names of any other's tests are in its source.
        let named = match kind {
            Kind::Rejected => path.file_stem().unwrap_or_default().to_string_lossy(),
            Kind::Checks | Kind::Traps => Cow::Borrowed(source.as_str()),
        };
        if filter.is_some_and(|filter| !named.contains(filter)) {
            return Ok(None);
        }
        let rejects = match kind {
            Kind::Rejected => Some(rejects(&source).map(str::to_owned).with_context(|| {
                format!(
                    "{shown} must not compile, but has no `{REJECTS} TEXT` line among the comment lines it opens \
                     with, to say what the compiler's diagnostics must contain"
                )
            })?),
            Kind::Checks | Kind::Traps => None,
        };
        Ok(Some(Module { path, shown, kind, rejects }))
    }
}

/// Runs the modules, up to `jobs` compiling at once, and prints each one's
/// tests' outcomes in order; fails when a module does not compile (or one
/// that must not, does) or a test fails.
pub fn run(checkout: &Path, options: Options) -> Result<()> {
    ensure!(
        blot_host::COMPILER,
        "this pen is a player runtime, built without the Wasm compiler: it cannot run a checkout's \
         tests (use a full build, `just install`)"
    );
    let paths = if options.files.is_empty() {
        test_modules(checkout)?
    } else {
        options.files.iter().map(|file| locate(checkout, file)).collect::<Result<_>>()?
    };
    ensure!(!paths.is_empty(), "no Blot test modules in {}", checkout.join("tests").display());
    let filter = options.filter.as_deref();
    let modules: Vec<Module> =
        paths.into_iter().filter_map(|path| Module::read(checkout, path, filter).transpose()).collect::<Result<_>>()?;
    let jobs = options.jobs.unwrap_or_else(default_jobs);
    ensure!(jobs > 0, "--jobs is how many modules compile at once: 1 or more");
    let jobs = jobs.min(modules.len()).max(1);
    let plugins = test_plugins(checkout)?;
    let entry = plugins.root().join(FIXTURE);
    // The modules share much of their code; the development engine
    // compiles each function it has seen only once.
    let engine = blot_host::dev_engine()?;
    let started = Instant::now();
    let mut tallies = [Tally::default(); Kind::ALL.len()];
    let mut failed = Vec::new();
    in_order(
        &modules,
        jobs,
        || Builder::new(plugins.clone(), &entry),
        |builder, module| run_module(builder, &engine, module, filter),
        |module, ran| {
            eprintln!("{}", module_lines(module, &ran, filter).join("\n"));
            let tally = &mut tallies[module.kind as usize];
            tally.modules += 1;
            let Ok(tests) = &ran.tests else {
                tally.broken += 1;
                failed.push(format!("{} (did not compile)", module.shown));
                return;
            };
            for (name, outcome) in &tests.tests {
                if outcome.passed() {
                    tally.passed += 1;
                } else {
                    tally.failed += 1;
                    failed.push(match module.kind {
                        Kind::Rejected => module.shown.clone(),
                        Kind::Checks | Kind::Traps => format!("{}: {name}", module.shown),
                    });
                }
            }
        },
    );
    eprintln!();
    let mut all = Tally::default();
    for kind in Kind::ALL {
        let tally = tallies[kind as usize];
        if tally.modules > 0 {
            eprintln!("{:<18}{tally}", kind.label());
        }
        all.add(tally);
    }
    eprintln!(
        "{:<18}{all} ({:.1} s, {jobs} job{})",
        "all",
        started.elapsed().as_secs_f64(),
        if jobs == 1 { "" } else { "s" }
    );
    if !failed.is_empty() {
        bail!("Blot tests failed:\n  {}", failed.join("\n  "));
    }
    Ok(())
}

/// How many modules compile at once unless told: about half the cores, at
/// most 8.
fn default_jobs() -> usize {
    std::thread::available_parallelism().map_or(1, |cores| cores.get() / 2).clamp(1, 8)
}

/// How the modules of a kind, or all of them, went.
#[derive(Clone, Copy, Default)]
struct Tally {
    modules: usize,
    passed: usize,
    failed: usize,
    /// Modules that did not compile, so none of their tests ran.
    broken: usize,
}

impl Tally {
    fn add(&mut self, other: Tally) {
        self.modules += other.modules;
        self.passed += other.passed;
        self.failed += other.failed;
        self.broken += other.broken;
    }
}

impl std::fmt::Display for Tally {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        let plural = if self.modules == 1 { "" } else { "s" };
        write!(formatter, "{} passed, {} failed, in {} module{plural}", self.passed, self.failed, self.modules)?;
        if self.broken > 0 {
            write!(formatter, "; {} did not compile", self.broken)?;
        }
        Ok(())
    }
}

/// Runs `work` on each of `items`, up to `jobs` at once and earliest first,
/// each worker with a `state` of its own (`start` makes it), and hands each
/// result to `done` on this thread in the items' order: each as soon as it
/// and every one before it are ready.
fn in_order<T: Sync, S, R: Send>(
    items: &[T],
    jobs: usize,
    start: impl Fn() -> S + Sync,
    work: impl Fn(&mut S, &T) -> R + Sync,
    mut done: impl FnMut(&T, R),
) {
    let next = AtomicUsize::new(0);
    let (sender, results) = mpsc::channel();
    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1).min(items.len()) {
            let (next, start, work, sender) = (&next, &start, &work, sender.clone());
            scope.spawn(move || {
                let mut state = start();
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(index) else { break };
                    // Only fails once `done` has panicked.
                    if sender.send((index, work(&mut state, item))).is_err() {
                        break;
                    }
                }
            });
        }
        drop(sender);
        let mut ready = BTreeMap::new();
        let mut handed = 0;
        for (index, result) in results {
            ready.insert(index, result);
            while let Some(result) = ready.remove(&handed) {
                done(&items[handed], result);
                handed += 1;
            }
        }
    });
}

/// A module, run: its tests' outcomes, or why none ran (it did not
/// compile), and how long the Blot compiler took.
struct Ran {
    tests: Result<Tests>,
    blot: Duration,
}

/// Compiles `module` and runs its tests whose names contain `filter`, or,
/// for one that must not compile, judges how compiling it went.
fn run_module(builder: &mut Builder, engine: &Engine, module: &Module, filter: Option<&str>) -> Ran {
    let compiling = Instant::now();
    let compiled = compile_test_module(builder, &module.path);
    let blot = compiling.elapsed();
    let tests = match &module.rejects {
        Some(rejects) => Ok(rejection(rejects, compiled)),
        None => compiled.and_then(|wasm| run_tests(engine, &wasm, module.kind, filter)),
    };
    Ran { tests, blot }
}

/// A module's lines: its path and kind, each test's outcome, and how many
/// passed.
fn module_lines(module: &Module, ran: &Ran, filter: Option<&str>) -> Vec<String> {
    let mut lines = vec![format!("{} ({})", module.shown, module.kind.label())];
    let tests = match &ran.tests {
        Ok(tests) => tests,
        Err(error) => {
            lines.push(format!("  FAILED to compile:\n{}", indent(&format!("{error:#}"), 4)));
            return lines;
        }
    };
    lines.extend(tests.skipped.iter().map(|skipped| format!("  warning: skipped {skipped}")));
    if tests.tests.is_empty() {
        lines.push(format!("  no {TEST_PREFIX}… entries{}", if filter.is_some() { " match" } else { "" }));
    }
    lines.extend(tests.tests.iter().flat_map(|(name, outcome)| describe(name, outcome)));
    let passed = tests.tests.iter().filter(|(_, outcome)| outcome.passed()).count();
    let blot = format!("Blot {:.1} s", ran.blot.as_secs_f64());
    let times = match module.kind {
        Kind::Rejected => blot,
        Kind::Checks | Kind::Traps => {
            format!("{blot}, Wasm {}, tests {}", milliseconds(tests.compile), milliseconds(tests.run))
        }
    };
    lines.push(format!("  {passed} passed, {} failed ({times})", tests.tests.len() - passed));
    lines
}

/// The checkout's modules: `tests/*.blot`, then `tests/traps/*.blot` and
/// `tests/rejected/*.blot`, each directory's sorted. Never `tests/blocked/`
/// (tests waiting on compiler fixes) nor `tests/codec_support/` (modules the
/// tests import).
fn test_modules(checkout: &Path) -> Result<Vec<PathBuf>> {
    let tests = checkout.join("tests");
    let mut found = blot_files(&tests)?;
    for directory in ["traps", "rejected"].map(|name| tests.join(name)) {
        if directory.is_dir() {
            found.extend(blot_files(&directory)?);
        }
    }
    Ok(found)
}

/// The `.blot` files in `directory`, sorted.
fn blot_files(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(directory)
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

/// The checkout's plugins, which compile its tests; their log's errors go
/// to stderr, the rest stays quiet.
fn test_plugins(checkout: &Path) -> Result<Arc<Plugins>> {
    let log: pen_plugins::Logger = Arc::new(|level, message| {
        if level == pen_plugins::Level::Error {
            eprintln!("{message}");
        }
    });
    Plugins::load(checkout, log)?.filter(|plugins| plugins.compiles()).ok_or_else(|| {
        anyhow!(
            "pen test-blot compiles through the checkout's Wasm plugins, and {} has no pen.json that \
             chooses a compiler",
            checkout.display()
        )
    })
}

/// Compiles the Blot test module at `module` to Wasm in `builder`'s session
/// (the checkout's compiler plugin, for `src/test_fixture.blot`), its source
/// standing in at that entry. A module that does not compile fails with the
/// compiler's diagnostics, placed in the module's own source.
pub fn compile_test_module(builder: &mut Builder, module: &Path) -> Result<Vec<u8>> {
    let source = std::fs::read_to_string(module).with_context(|| format!("reading {}", module.display()))?;
    let entry = builder.entry().display().to_string();
    let standing_in = Overlay { path: entry.clone(), contents: Some(source.clone()) };
    match builder.build(&[standing_in]) {
        Building::Built(built) => Ok(built.wasm),
        Building::Rejected(diagnostics) => {
            let module = module.display().to_string();
            let lines: Vec<String> = diagnostics
                .iter()
                .map(|diagnostic| {
                    crate::plugin_build::describe(diagnostic, Some((&entry, &source))).replacen(&entry, &module, 1)
                })
                .collect();
            bail!("{}", lines.join("\n"))
        }
        Building::Failed(error) => bail!("the compiler failed: {error}"),
    }
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
    /// It trapped (where it must not), or what it returned is not a report.
    Broken(String),
    /// It trapped at `unreachable`, as a test that must trap must: passed.
    Panicked,
    /// A test that must trap returned these lanes instead.
    Returned(Vec<f32>),
    /// A test that must trap trapped some other way: the trap.
    Mistrapped(String),
    /// A module that must not compile did not, with the diagnostics it
    /// must: passed.
    Rejected,
    /// A module that must not compile compiled.
    Compiled,
    /// A module that must not compile did not, with other diagnostics:
    /// these.
    Misrejected(String),
}

impl Outcome {
    fn passed(&self) -> bool {
        match self {
            Outcome::Reported(report) => report.failures.is_empty(),
            Outcome::Panicked | Outcome::Rejected => true,
            Outcome::Broken(_)
            | Outcome::Returned(_)
            | Outcome::Mistrapped(_)
            | Outcome::Compiled
            | Outcome::Misrejected(_) => false,
        }
    }
}

/// The one test of a module that must not compile, named for what its
/// diagnostics must contain: passed when compiling it failed with
/// diagnostics that contain that.
fn rejection(rejects: &str, compiled: Result<Vec<u8>>) -> Tests {
    let outcome = match compiled {
        Ok(_) => Outcome::Compiled,
        Err(error) => {
            let diagnostics = format!("{error:#}");
            if diagnostics.contains(rejects) { Outcome::Rejected } else { Outcome::Misrejected(diagnostics) }
        }
    };
    Tests {
        tests: vec![(rejects.to_owned(), outcome)],
        skipped: Vec::new(),
        compile: Duration::ZERO,
        run: Duration::ZERO,
    }
}

/// What a module that must not compile has its diagnostics contain: the
/// text of its `// rejects: TEXT` line, among the comment lines it opens
/// with (after `// test: game`, when it compiles against the game).
fn rejects(source: &str) -> Option<&str> {
    source
        .lines()
        .take_while(|line| line.starts_with("//"))
        .find_map(|line| line.strip_prefix(REJECTS))
        .map(str::trim)
        .filter(|text| !text.is_empty())
}

/// Compiles `wasm` and runs its `test_…` entries whose names contain
/// `filter`, each in an instance of its own, on a thread with the stack a
/// guest needs; each must do what `kind` says.
fn run_tests(engine: &Engine, wasm: &[u8], kind: Kind, filter: Option<&str>) -> Result<Tests> {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("pen-test".into())
            .stack_size(blot_host::GUEST_STACK_BYTES + (8 << 20))
            .spawn_scoped(scope, || run_tests_here(engine, wasm, kind, filter))?
            .join()
            .map_err(|_| anyhow!("the test thread panicked"))?
    })
}

fn run_tests_here(engine: &Engine, wasm: &[u8], kind: Kind, filter: Option<&str>) -> Result<Tests> {
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
        let outcome = match kind {
            Kind::Traps => trapped(called),
            Kind::Checks | Kind::Rejected => reported(called),
        };
        tests.push((function.name.clone(), outcome));
    }
    Ok(Tests { tests, skipped, compile, run: started.elapsed() })
}

/// A test's outcome by the report it returned.
fn reported(called: Result<Value>) -> Outcome {
    match called.and_then(Value::into_f32_array) {
        Ok(lanes) => match decode(&lanes) {
            Ok(report) => Outcome::Reported(report),
            Err(error) => Outcome::Broken(format!("its report is malformed ({error:#}): {}", format_lanes(&lanes))),
        },
        Err(error) => Outcome::Broken(explain(&error)),
    }
}

/// The outcome of a test that must trap at `unreachable`, as Blot's `@panic`
/// stops a program. Wasmtime's trap code tells it from other traps
/// (out-of-bounds memory, stack overflow, division by zero, …).
fn trapped(called: Result<Value>) -> Outcome {
    match called {
        Ok(value) => match value.into_f32_array() {
            Ok(lanes) => Outcome::Returned(lanes),
            Err(error) => Outcome::Broken(explain(&error)),
        },
        Err(error) => match error.downcast_ref::<Trap>() {
            Some(Trap::UnreachableCodeReached) => Outcome::Panicked,
            Some(_) => Outcome::Mistrapped(explain(&error)),
            None => Outcome::Broken(explain(&error)),
        },
    }
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
/// check expected and got, the trap, what a test that must trap returned,
/// or the diagnostics of a module that must not compile. The one test of
/// such a module is named for what they must contain.
fn describe(name: &str, outcome: &Outcome) -> Vec<String> {
    match outcome {
        Outcome::Reported(report) if report.failures.is_empty() => vec![format!("  ok      {name}")],
        Outcome::Reported(report) => {
            let mut lines = vec![format!("  FAILED  {name} ({} of {} checks)", report.failures.len(), report.checks)];
            lines.extend(failed_checks(report));
            lines
        }
        Outcome::Broken(message) => vec![format!("  FAILED  {name}"), indent(message, 4)],
        Outcome::Panicked => vec![format!("  ok      {name}")],
        Outcome::Returned(lanes) => {
            let mut lines = vec![format!("  FAILED  {name} (returned instead of trapping)")];
            match decode(lanes) {
                Ok(report) => {
                    lines.push(format!("    its report: {} of {} checks failed", report.failures.len(), report.checks));
                    lines.extend(failed_checks(&report));
                }
                Err(_) => lines.push(format!("    it returned {}", format_lanes(lanes))),
            }
            lines
        }
        Outcome::Mistrapped(trap) => {
            vec![format!("  FAILED  {name} (trapped, but not at `unreachable`)"), indent(trap, 4)]
        }
        Outcome::Rejected => vec![format!("  ok      rejected: {name}")],
        Outcome::Compiled => vec![format!("  FAILED  compiled, but should be rejected: {name}")],
        Outcome::Misrejected(diagnostics) => {
            vec![format!("  FAILED  rejected, but the diagnostics lack: {name}"), indent(diagnostics, 4)]
        }
    }
}

/// What each failed check of `report` expected and got, a line each.
fn failed_checks(report: &Report) -> impl Iterator<Item = String> + '_ {
    report.failures.iter().map(|failure| {
        format!(
            "    check {}: expected {} got {}",
            failure.place,
            format_lanes(&failure.expected),
            format_lanes(&failure.actual)
        )
    })
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

    /// Entries for a module that must trap: one stopping at `unreachable`, as
    /// Blot's `@panic` does; two returning, a report and lanes that are
    /// none; and three trapping otherwise.
    fn trapping_sample() -> Vec<u8> {
        guest(
            &[
                ("test_panics", UNIT_TO_ARRAY, 0, vec![0x00, 0x0b]),
                ("test_returns_a_report", UNIT_TO_ARRAY, 0, constant(24)),
                ("test_returns_lanes", UNIT_TO_ARRAY, 0, constant(64)),
                // i32.load at the address space's last byte.
                (
                    "test_reads_out_of_bounds",
                    UNIT_TO_ARRAY,
                    0,
                    [vec![0x41], signed_leb(-1), vec![0x28, 2, 0, 0x0b]].concat(),
                ),
                // 1 / 0 (i32.div_s).
                ("test_divides_by_zero", UNIT_TO_ARRAY, 0, vec![0x41, 1, 0x41, 0, 0x6d, 0x0b]),
                // Calls itself, function 9: the import, blot:allocate,
                // blot:reset, main, then these entries.
                ("test_recurses", UNIT_TO_ARRAY, 0, vec![0x20, 0, 0x10, 9, 0x0b]),
            ],
            &[(24, &[2.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 4.0]), (64, &[1.0, 1.0, 1.0, 5.0])],
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
    fn runs_tests_that_must_trap() {
        let engine = blot_host::engine().unwrap();
        let tests = run_tests(&engine, &trapping_sample(), Kind::Traps, None).unwrap();
        assert_eq!(tests.tests.len(), 6);
        let outcome = |name: &str| &tests.tests.iter().find(|(test, _)| test == name).unwrap().1;
        // Trapping at `unreachable`, as `@panic` does, passes.
        assert_eq!(*outcome("test_panics"), Outcome::Panicked);
        assert!(outcome("test_panics").passed());
        assert_eq!(describe("test_panics", outcome("test_panics")), ["  ok      test_panics"]);
        // Returning fails, showing the report it returned, or its lanes.
        assert!(!outcome("test_returns_a_report").passed());
        assert_eq!(
            describe("test_returns_a_report", outcome("test_returns_a_report")),
            [
                "  FAILED  test_returns_a_report (returned instead of trapping)",
                "    its report: 1 of 2 checks failed",
                "    check 2: expected [2, 3] got [2, 4]",
            ]
        );
        assert_eq!(*outcome("test_returns_lanes"), Outcome::Returned(vec![1.0, 1.0, 1.0, 5.0]));
        assert_eq!(
            describe("test_returns_lanes", outcome("test_returns_lanes")),
            ["  FAILED  test_returns_lanes (returned instead of trapping)", "    it returned [1, 1, 1, 5]"]
        );
        // Any other trap fails, showing it.
        for (name, trap) in [
            ("test_reads_out_of_bounds", "out of bounds memory access"),
            ("test_divides_by_zero", "integer divide by zero"),
            ("test_recurses", "call stack exhausted"),
        ] {
            let Outcome::Mistrapped(message) = outcome(name) else { panic!("{name}: {:?}", outcome(name)) };
            assert!(message.starts_with(&format!("wasm trap: {trap}")), "{message}");
            let lines = describe(name, outcome(name));
            assert_eq!(lines[0], format!("  FAILED  {name} (trapped, but not at `unreachable`)"));
            assert!(lines[1].starts_with(&format!("    wasm trap: {trap}")), "{}", lines[1]);
        }
        // Where a test must not trap, `unreachable` fails it.
        let checks = run_tests(&engine, &trapping_sample(), Kind::Checks, Some("panics")).unwrap();
        let Outcome::Broken(trap) = &checks.tests[0].1 else { panic!("{:?}", checks.tests) };
        assert!(trap.starts_with("wasm trap: wasm `unreachable`"), "{trap}");
    }

    #[test]
    fn judges_modules_that_must_not_compile() {
        let rejects = "component type already registered";
        // Not compiling, with diagnostics that contain the text, passes.
        let diagnostics = "/gdev/packages/ecs.blot:0-0: const_panic: component type already registered";
        let rejected = rejection(rejects, Err(anyhow!("{diagnostics}")));
        assert_eq!(rejected.tests, [(rejects.to_owned(), Outcome::Rejected)]);
        assert!(rejected.tests[0].1.passed());
        assert_eq!(describe(rejects, &Outcome::Rejected), ["  ok      rejected: component type already registered"]);
        // Compiling fails.
        let compiled = rejection(rejects, Ok(sample()));
        assert_eq!(compiled.tests, [(rejects.to_owned(), Outcome::Compiled)]);
        assert!(!compiled.tests[0].1.passed());
        assert_eq!(
            describe(rejects, &Outcome::Compiled),
            ["  FAILED  compiled, but should be rejected: component type already registered"]
        );
        // Not compiling with other diagnostics fails, showing them.
        for other in [
            "/gdev/src/test_fixture.blot:9532-9551: type_mismatch: Types do not match",
            "/gdev/packages/ecs.blot:0-0: const_panic: component type already initialized",
        ] {
            let misrejected = rejection(rejects, Err(anyhow!("{other}")));
            assert_eq!(misrejected.tests, [(rejects.to_owned(), Outcome::Misrejected(other.to_owned()))]);
            assert!(!misrejected.tests[0].1.passed());
            assert_eq!(
                describe(rejects, &misrejected.tests[0].1),
                [
                    "  FAILED  rejected, but the diagnostics lack: component type already registered".to_owned(),
                    format!("    {other}")
                ]
            );
        }
    }

    #[test]
    fn reads_what_a_module_must_be_rejected_for() {
        let ecs = "// rejects: component type already registered\n// A component type registered twice\nimport * as ecs from \"gdev/ecs\"\n";
        assert_eq!(rejects(ecs), Some("component type already registered"));
        assert_eq!(
            rejects("// test: game\r\n// rejects: type_mismatch \r\n// A fixed update\r\n"),
            Some("type_mismatch")
        );
        assert_eq!(rejects("// test: game\n// Why it does not compile\n// rejects: x y\nconst a = 1\n"), Some("x y"));
        // Only among the comment lines it opens with, and never empty.
        assert_eq!(rejects("// test: game\nconst a = 1\n// rejects: x\n"), None);
        assert_eq!(rejects("\n// rejects: x\n"), None);
        assert_eq!(rejects("// rejects:\n"), None);
        assert_eq!(rejects("// rejects:  \n// rejects: x\n"), None);
        assert_eq!(rejects("// rejected: x\n"), None);
        assert_eq!(rejects(""), None);
    }

    #[test]
    fn finds_the_checkouts_modules() {
        let checkout = std::env::temp_dir().join(format!("pen-blot-tests-{}", std::process::id()));
        let tests = checkout.join("tests");
        let files = [
            "b.blot",
            "a.blot",
            "notes.md",
            "traps/z.blot",
            "traps/c.blot",
            "rejected/r.blot",
            "rejected/unsaid.blot",
            "blocked/x.blot",
            "codec_support/s.blot",
        ];
        for file in files {
            let path = tests.join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let said = if file == "rejected/unsaid.blot" { "" } else { "// rejects: what test_d says\n" };
            std::fs::write(&path, format!("// test: game\n{said}entry const test_d = fn () => 0\n")).unwrap();
        }
        let found = test_modules(&checkout).unwrap();
        // Each directory's sorted, in this order; never blocked/ nor codec_support/.
        let expected = ["a.blot", "b.blot", "traps/c.blot", "traps/z.blot", "rejected/r.blot", "rejected/unsaid.blot"];
        assert_eq!(found, expected.map(|file| tests.join(file)));
        let kinds: Vec<Kind> = found.iter().map(|module| Kind::of(module)).collect();
        assert_eq!(kinds, [Kind::Checks, Kind::Checks, Kind::Traps, Kind::Traps, Kind::Rejected, Kind::Rejected]);
        // One that must not compile says why.
        let module = Module::read(&checkout, tests.join("rejected/r.blot"), None).unwrap().unwrap();
        assert_eq!(
            (module.shown.as_str(), module.kind),
            (Path::new("tests/rejected/r.blot").to_str().unwrap(), Kind::Rejected)
        );
        assert_eq!(module.rejects.as_deref(), Some("what test_d says"));
        let unsaid = Module::read(&checkout, tests.join("rejected/unsaid.blot"), None).err().unwrap();
        assert!(format!("{unsaid:#}").contains("has no `// rejects: TEXT` line"), "{unsaid:#}");
        // The filter: the names of a module's tests are in its source, and
        // one that must not compile is named for its file.
        assert!(Module::read(&checkout, tests.join("a.blot"), Some("test_d")).unwrap().is_some());
        assert!(Module::read(&checkout, tests.join("a.blot"), Some("test_e")).unwrap().is_none());
        assert!(Module::read(&checkout, tests.join("rejected/r.blot"), Some("test_d")).unwrap().is_none());
        assert!(Module::read(&checkout, tests.join("rejected/unsaid.blot"), Some("test_d")).unwrap().is_none());
        assert!(Module::read(&checkout, tests.join("rejected/r.blot"), Some("r")).unwrap().is_some());
        // Without traps/ and rejected/.
        std::fs::remove_dir_all(tests.join("traps")).unwrap();
        std::fs::remove_dir_all(tests.join("rejected")).unwrap();
        assert_eq!(test_modules(&checkout).unwrap(), ["a.blot", "b.blot"].map(|file| tests.join(file)));
        std::fs::remove_dir_all(&checkout).unwrap();
        assert!(test_modules(&checkout).is_err());
    }

    #[test]
    fn hands_results_over_in_order() {
        let items: Vec<u64> = (0..12).collect();
        let (running, most) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let mut handed = Vec::new();
        in_order(
            &items,
            4,
            || (),
            |(), &item| {
                most.fetch_max(running.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                // Later items finish sooner.
                std::thread::sleep(Duration::from_millis(3 * (12 - item)));
                running.fetch_sub(1, Ordering::SeqCst);
                item * 10
            },
            |&item, result| handed.push((item, result)),
        );
        assert_eq!(handed, items.iter().map(|&item| (item, item * 10)).collect::<Vec<_>>());
        let most = most.load(Ordering::SeqCst);
        assert!((2..=4).contains(&most), "{most} at once");
        // Nothing to do, and one at a time.
        in_order(&[] as &[u64], 4, || (), |(), _| unreachable!(), |_, ()| unreachable!());
        let mut one = Vec::new();
        in_order(&items, 1, || (), |(), &item| item, |_, result| one.push(result));
        assert_eq!(one, items);
    }

    #[test]
    fn runs_a_modules_tests() {
        let engine = blot_host::engine().unwrap();
        let tests = run_tests(&engine, &sample(), Kind::Checks, None).unwrap();
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

        let filtered = run_tests(&engine, &sample(), Kind::Checks, Some("fail")).unwrap();
        assert_eq!(filtered.tests.len(), 1);
        assert_eq!(filtered.tests[0].0, "test_fails");
        assert!(run_tests(&engine, b"\0asm\x01\0\0\0", Kind::Checks, None).is_err());
    }

    #[test]
    fn runs_the_checkouts_pilot() {
        // The sibling gdev checkout's pilot module, when this repo has one
        // and its plugins are there to compile it.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../gdev");
        let pilot = root.join("tests/hand_salt.blot");
        let plugins = pilot.is_file().then(|| test_plugins(&root).ok()).flatten();
        let Some(plugins) = plugins else {
            eprintln!("skipped: no ../gdev/tests/hand_salt.blot, or no plugins to compile it");
            return;
        };
        let entry = plugins.root().join(FIXTURE);
        let mut builder = Builder::new(plugins, &entry);
        let wasm = compile_test_module(&mut builder, &pilot).unwrap();
        let engine = blot_host::dev_engine().unwrap();
        let tests = run_tests(&engine, &wasm, Kind::Checks, None).unwrap();
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
        // A module that does not compile, in the same session, fails with the
        // compiler's diagnostics, placed in the module.
        let broken = std::env::temp_dir().join(format!("pen-broken-{}.blot", std::process::id()));
        std::fs::write(&broken, "entry const test_broken = fn () => missing_name\n").unwrap();
        let error = compile_test_module(&mut builder, &broken).unwrap_err();
        let _ = std::fs::remove_file(&broken);
        let error = format!("{error:#}");
        assert!(error.contains("missing_name"), "{error}");
        assert!(error.starts_with(&format!("{}:1:", broken.display())), "{error}");
    }
}
