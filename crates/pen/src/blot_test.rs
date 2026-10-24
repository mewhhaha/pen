//! `pen test-blot`: a checkout's tests, as its compiler plugin lays them
//! out (its `pen.json`, `pen-plugins`; for Blot, gdev's tests/README.md):
//! which modules, what each builds as, and what its tests must do. pen
//! builds each module in a compiler session for its entry, each worker
//! keeping its session from module to module so that each compiles
//! incrementally, and runs its tests as Wasm. A test is an export of its
//! module named with the prefix the plugin gives, Unit to Array F32, run in
//! a fresh instance:
//!
//! - In a module of checks, it returns the report of its checks (gdev's
//!   `tests/support/check.blot`): the checks made, the failures, then each
//!   failure's place (from 1), its expected lanes' count and lanes, and its
//!   actual lanes' count and lanes. It passes when no check failed.
//! - In a module whose tests must trap, it must trap at `unreachable`, as
//!   Blot's `@panic` stops a program (a decoder refusing bad input). Any
//!   other trap, or returning, fails.
//! - A module that must not compile is one test, which passes when the
//!   compiler's diagnostics contain the text the plugin gives.
//!
//! Up to `--jobs` modules compile at once, each running its tests once
//! compiled, and the modules print in order. Up to `--jobs` tests run at
//! once too, over all the modules, a module's own in parallel: gdev's codec
//! module is many tests of minutes each, nearly all of it in the guest
//! runtime's cycle collector (docs/testing.md, "Blot tests' speed").
//!
//! Ordinary tests take no capability. With `--session`, tests receive the
//! pen/test driver capability and control a separate headless game session.
//! These GPU scenarios compile and run sequentially.
mod session;
pub use session::Options as SessionOptions;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, PoisonError, mpsc};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use blot_host::{Guest, GuestFunction, GuestModule, Parameter, Value, ValueType};
use pen_plugins::{Builder, Expectation, Outcome as Building, Plugins, TestModule};
use wasmtime::{Engine, Trap};

/// The checkout's main program, which the plugin lays its tests out by.
const ENTRY: &str = "src/main.blot";

pub struct Options {
    pub session: Option<SessionOptions>,
    /// The modules to run; every one of the checkout's when empty.
    pub files: Vec<PathBuf>,
    /// Only the tests whose names contain this.
    pub filter: Option<String>,
    /// How many modules compile at once; about half the cores, at most 8,
    /// when None.
    pub jobs: Option<usize>,
    /// How long one test may run before it fails as stuck; [`TEST_DEADLINE`]
    /// when None.
    pub deadline: Option<Duration>,
}

/// What a module's tests must do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Each test returns a report whose checks pass.
    Checks,
    /// Each test traps at `unreachable`.
    Traps,
    /// The module does not compile, for the reason the plugin gives.
    Rejected,
}

impl Kind {
    /// In the order they are tallied.
    const ALL: [Kind; 3] = [Kind::Checks, Kind::Traps, Kind::Rejected];

    fn of(expect: &Expectation) -> Kind {
        match expect {
            Expectation::Checks => Kind::Checks,
            Expectation::Traps => Kind::Traps,
            Expectation::Rejected(_) => Kind::Rejected,
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
    test: TestModule,
    /// Its path as shown: from the checkout.
    shown: String,
    kind: Kind,
}

impl Module {
    fn new(checkout: &Path, test: TestModule) -> Module {
        let path = Path::new(&test.path);
        let shown = path.strip_prefix(checkout).unwrap_or(path).display().to_string();
        Module { kind: Kind::of(&test.expect), test, shown }
    }
}

/// How long a test runs before its line says how long it took.
const SLOW_TEST: Duration = Duration::from_secs(1);

/// How long one test may run before it fails as stuck: a test plays many
/// frames without calling the host, so far past a game's
/// [`blot_host::DEADLINE`]. gdev's codec tests take minutes each (their
/// world's cycle collection), more on a busy machine, and the deadline counts
/// the time they wait for a core; `--deadline` sets another.
const TEST_DEADLINE: Duration = Duration::from_secs(600);

/// Runs the modules, up to `jobs` compiling at once, and prints each one's
/// tests' outcomes in order; fails when a module does not compile (or one
/// that must not, does) or a test fails.
pub fn run(checkout: &Path, options: Options) -> Result<()> {
    ensure!(
        blot_host::COMPILER,
        "this pen is a player runtime, built without the Wasm compiler: it cannot run a checkout's \
         tests (use a full build, `just install`)"
    );
    let jobs = if options.session.is_some() { 1 } else { options.jobs.unwrap_or_else(default_jobs) };
    ensure!(jobs > 0, "--jobs is how many modules compile at once: 1 or more");
    let plugins = test_plugins(checkout)?;
    let checkout = plugins.root();
    let paths: Vec<PathBuf> = options.files.iter().map(|file| locate(checkout, file)).collect::<Result<_>>()?;
    let filter = options.filter.as_deref();
    let modules: Vec<Module> = plugins
        .tests(&checkout.join(ENTRY), &paths, filter)?
        .into_iter()
        .map(|test| Module::new(checkout, test))
        .collect();
    ensure!(!modules.is_empty() || filter.is_some(), "{} has no test modules", checkout.display());
    let session = options.session.map(|options| session::Prepared::new(checkout, options)).transpose()?;
    // As many tests run at once as modules compile, however few the modules.
    let slots = Slots::new(jobs).deadline(options.deadline.unwrap_or(TEST_DEADLINE));
    let jobs = jobs.min(modules.len()).max(1);
    // The modules share much of their code; the development engine
    // compiles each function it has seen only once.
    let engine = blot_host::dev_engine()?;
    let started = Instant::now();
    let mut tallies = [Tally::default(); Kind::ALL.len()];
    let mut failed = Vec::new();
    in_order(
        &modules,
        jobs,
        || None,
        |builder, module| run_module(builder, &plugins, &engine, module, filter, &slots, session.as_ref()),
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
        bail!("tests failed:\n  {}", failed.join("\n  "));
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
/// compile), and how long the compiler took.
struct Ran {
    tests: Result<Tests>,
    build: Duration,
}

/// Compiles `module` in a session for its entry (the worker's last one,
/// when it was for the same entry), and runs its tests whose names contain
/// `filter`, or, for one that must not compile, judges how compiling it
/// went.
fn run_module(
    session: &mut Option<Builder>,
    plugins: &Arc<Plugins>,
    engine: &Engine,
    module: &Module,
    filter: Option<&str>,
    slots: &Slots,
    session_tests: Option<&session::Prepared>,
) -> Ran {
    let entry = Path::new(&module.test.entry);
    if session.as_ref().is_none_or(|builder| builder.entry() != entry) {
        if let Some(mut last) = session.take() {
            let _ = last.close();
        }
        *session = Some(Builder::new(plugins.clone(), entry));
    }
    let builder = session.as_mut().expect("opened above");
    let compiling = Instant::now();
    let compiled = build_test_module(builder, &module.test);
    let build = compiling.elapsed();
    let tests = match &module.test.expect {
        Expectation::Rejected(rejects) => Ok(rejection(rejects, compiled)),
        Expectation::Checks | Expectation::Traps => {
            let run = session_tests.map(|prepared| prepared.module(Path::new(&module.test.path)));
            compiled.and_then(|wasm| {
                run_tests_with_session(engine, &wasm, module.kind, &module.test.prefix, filter, slots, run.as_ref())
            })
        }
    };
    Ran { tests, build }
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
        let prefix = &module.test.prefix;
        lines.push(format!("  no {prefix}… entries{}", if filter.is_some() { " match" } else { "" }));
    }
    for ((name, outcome), timing) in tests.tests.iter().zip(&tests.times) {
        let mut described = describe(name, outcome);
        // A test that takes a second or more says how long, and how often
        // the guest runtime collected cycles meanwhile.
        if timing.took >= SLOW_TEST {
            let collections =
                timing.cycle_passes.map(|passes| format!("; {passes} cycle collections")).unwrap_or_default();
            described[0].push_str(&format!(" ({:.1} s{collections})", timing.took.as_secs_f64()));
        }
        lines.extend(described);
    }
    let passed = tests.tests.iter().filter(|(_, outcome)| outcome.passed()).count();
    let build = format!("build {:.1} s", ran.build.as_secs_f64());
    let times = match module.kind {
        Kind::Rejected => build,
        Kind::Checks | Kind::Traps => {
            format!("{build}, Wasm {}, tests {}", milliseconds(tests.compile), milliseconds(tests.run))
        }
    };
    lines.push(format!("  {passed} passed, {} failed ({times})", tests.tests.len() - passed));
    lines
}

/// A module named on the command line: from the current directory, else
/// from the checkout.
fn locate(checkout: &Path, file: &Path) -> Result<PathBuf> {
    let found = [std::path::absolute(file)?, checkout.join(file)].into_iter().find(|path| path.is_file());
    let found = found.with_context(|| format!("there is no test module {}", file.display()))?;
    Ok(found.canonicalize()?)
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

/// Compiles the test module `test` to Wasm in `builder`'s session (the
/// checkout's compiler plugin, for the module's entry), with the overlays it
/// builds with. A module that does not compile fails with the compiler's
/// diagnostics, those at its entry placed in the module's own source, which
/// stands in there.
pub fn build_test_module(builder: &mut Builder, test: &TestModule) -> Result<Vec<u8>> {
    match builder.build(&test.overlays) {
        Building::Built(built) => Ok(built.wasm),
        Building::Rejected(diagnostics) => {
            let standing_in = test
                .overlays
                .iter()
                .find(|overlay| overlay.path == test.entry)
                .and_then(|overlay| Some((overlay.path.as_str(), overlay.contents.as_deref()?)));
            let lines: Vec<String> = diagnostics
                .iter()
                .map(|diagnostic| {
                    let line = crate::plugin_build::describe(diagnostic, standing_in);
                    if diagnostic.file == test.entry { line.replacen(&test.entry, &test.path, 1) } else { line }
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
    /// The exports named as tests that are not Unit to Array F32, and what
    /// they are.
    skipped: Vec<String>,
    /// How long each test ran, in the same order.
    times: Vec<Timing>,
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
        times: vec![Timing::default()],
        compile: Duration::ZERO,
        run: Duration::ZERO,
    }
}

/// Tests running at once, over every module, and how long each may: each
/// holds a slot while it runs, so a module of many slow tests spreads over
/// the cores (gdev's codec tests each build a world of 10,000 gnorps) without
/// the modules' workers multiplying them.
pub struct Slots {
    count: usize,
    free: Mutex<usize>,
    freed: Condvar,
    deadline: Duration,
}

impl Slots {
    pub fn new(count: usize) -> Slots {
        let count = count.max(1);
        Slots { count, free: Mutex::new(count), freed: Condvar::new(), deadline: TEST_DEADLINE }
    }

    /// Each test may run this long before it fails as stuck.
    pub fn deadline(self, deadline: Duration) -> Slots {
        Slots { deadline, ..self }
    }

    /// Runs `work` once a slot is free.
    fn hold<T>(&self, work: impl FnOnce() -> T) -> T {
        /// Gives the slot back, however `work` ends.
        struct Held<'a>(&'a Slots);
        impl Drop for Held<'_> {
            fn drop(&mut self) {
                *self.0.free.lock().unwrap_or_else(PoisonError::into_inner) += 1;
                self.0.freed.notify_one();
            }
        }
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        while *free == 0 {
            free = self.freed.wait(free).unwrap_or_else(PoisonError::into_inner);
        }
        *free -= 1;
        drop(free);
        let _held = Held(self);
        work()
    }
}

/// Compiles `wasm` and runs its tests, the entries named with `prefix`,
/// whose names contain `filter`, each in an instance of its own, on a
/// thread with the stack a guest needs, as many at once as `slots` lets;
/// each must do what `kind` says.
#[cfg(test)]
fn run_tests(
    engine: &Engine,
    wasm: &[u8],
    kind: Kind,
    prefix: &str,
    filter: Option<&str>,
    slots: &Slots,
) -> Result<Tests> {
    run_tests_with_session(engine, wasm, kind, prefix, filter, slots, None)
}

fn run_tests_with_session(
    engine: &Engine,
    wasm: &[u8],
    kind: Kind,
    prefix: &str,
    filter: Option<&str>,
    slots: &Slots,
    session: Option<&session::Run<'_>>,
) -> Result<Tests> {
    let started = Instant::now();
    let module = GuestModule::new(engine, wasm, None)?;
    let compile = started.elapsed();
    let started = Instant::now();
    let mut skipped = Vec::new();
    let mut runnable = Vec::new();
    let entries = module.abi().functions.iter().filter(|function| {
        function.name.starts_with(prefix) && filter.is_none_or(|filter| function.name.contains(filter))
    });
    for function in entries {
        if session.is_some() {
            ensure!(
                kind == Kind::Checks
                    && function.parameter == Parameter::Callback(pen_runtime::abi::GAME_HOST)
                    && function.result == ValueType::ArrayF32,
                "{}: session tests take the driver (Array F32 -> Array F32 ! {{ Foreign }}) and return Array F32 through pen/test.run",
                function.name
            );
        } else if function.parameter != Parameter::Value(ValueType::Unit) || function.result != ValueType::ArrayF32 {
            skipped.push(format!("{}: it is {}, not Unit -> Array F32", function.name, signature(function)));
            continue;
        }
        runnable.push(function.name.clone());
    }
    ensure!(session.is_none() || !runnable.is_empty(), "no runnable session tests match");
    let next = AtomicUsize::new(0);
    let ran: Mutex<Vec<Option<(Outcome, Timing)>>> = Mutex::new(runnable.iter().map(|_| None).collect());
    std::thread::scope(|scope| -> Result<()> {
        let workers = (0..slots.count.min(runnable.len()))
            .map(|_| {
                std::thread::Builder::new()
                    .name("pen-test".into())
                    .stack_size(blot_host::GUEST_STACK_BYTES + (8 << 20))
                    .spawn_scoped(scope, || {
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            let Some(name) = runnable.get(index) else { break };
                            let outcome = slots.hold(|| match session {
                                Some(run) => {
                                    let started = Instant::now();
                                    (
                                        reported(run.call(engine, &module, name, slots.deadline)),
                                        Timing { took: started.elapsed(), cycle_passes: None },
                                    )
                                }
                                None => run_test(engine, &module, kind, name, slots.deadline),
                            });
                            ran.lock().unwrap_or_else(PoisonError::into_inner)[index] = Some(outcome);
                        }
                    })
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        for worker in workers {
            worker.join().map_err(|_| anyhow!("a test thread panicked"))?;
        }
        Ok(())
    })?;
    let ran = ran.into_inner().unwrap_or_else(PoisonError::into_inner);
    let (tests, times) = runnable
        .into_iter()
        .zip(ran)
        .map(|(name, ran)| {
            let (outcome, time) = ran.expect("every test ran");
            ((name, outcome), time)
        })
        .unzip();
    Ok(Tests { tests, skipped, times, compile, run: started.elapsed() })
}

/// How long a test ran, and how many passes the guest runtime's cycle
/// collector made meanwhile (when its guest counts them).
#[derive(Clone, Copy, Debug, Default)]
struct Timing {
    took: Duration,
    cycle_passes: Option<u32>,
}

/// Runs the test `name` in an instance of its own: its outcome, and how
/// long it took.
fn run_test(engine: &Engine, module: &GuestModule, kind: Kind, name: &str, deadline: Duration) -> (Outcome, Timing) {
    let running = Instant::now();
    let mut cycle_passes = None;
    let called = Guest::new(engine, module).and_then(|mut guest| {
        guest.set_deadline(deadline);
        let called = guest.call(name, Value::Unit);
        cycle_passes = guest
            .runtime_counters()
            .into_iter()
            .find_map(|(counter, value)| (counter == "blot:cycle_passes").then_some(value));
        called
    });
    let outcome = match kind {
        Kind::Traps => trapped(called),
        Kind::Checks | Kind::Rejected => reported(called),
    };
    (outcome, Timing { took: running.elapsed(), cycle_passes })
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
        let count = pen_protocol::boundary::whole(lane).filter(|&count| count <= 16_777_216);
        count.map(|count| count as usize).with_context(|| format!("{what} is {lane}, not a count"))
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
pub(crate) mod tests;
