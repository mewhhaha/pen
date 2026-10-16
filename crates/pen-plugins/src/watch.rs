//! A program built through the checkout's compiler plugin, once or again
//! whenever a file the compiler named changes: watched by a watcher of its
//! own ([`watch`]), or by the caller's ([`watch_changes`]).
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::{Duration, Instant};

use anyhow::Result;
use notify::{EventKind, RecursiveMode, Watcher};

use crate::{BuildResult, Compiler, Diagnostic, Plugins};

/// How long files must be quiet before a new build: long enough to see an
/// editor's save through.
const SETTLE: Duration = Duration::from_millis(15);

/// What a build came to.
pub enum Outcome {
    Built(Built),
    /// The program does not compile: at least one diagnostic.
    Rejected(Vec<Diagnostic>),
    /// The toolchain failed (a plugin, or the compiler it runs).
    Failed(String),
}

/// A program's Wasm, and how long it took.
pub struct Built {
    pub wasm: Vec<u8>,
    pub seconds: f64,
}

/// One program and the compiler session that builds it, kept open from
/// build to build: an edit rebuilds incrementally.
pub struct Builder {
    plugins: Arc<Plugins>,
    entry: PathBuf,
    compiler: Option<Compiler>,
    /// The paths whose changes need a new build.
    watched: BTreeSet<PathBuf>,
}

impl Builder {
    pub fn new(plugins: Arc<Plugins>, entry: &Path) -> Builder {
        Builder { plugins, entry: entry.to_owned(), compiler: None, watched: BTreeSet::new() }
    }

    /// The program's entry.
    pub fn entry(&self) -> &Path {
        &self.entry
    }

    /// The paths (files, or directories with everything under them) whose
    /// changes need a new build, as the builds so far named them.
    pub fn watched(&self) -> &BTreeSet<PathBuf> {
        &self.watched
    }

    /// Whether a change to `path` needs a new build.
    pub fn needs(&self, path: &Path) -> bool {
        self.watched.iter().any(|watched| path.starts_with(watched))
    }

    pub fn close(&mut self) -> Result<()> {
        if let Some(mut compiler) = self.compiler.take() {
            compiler.close()?;
        }
        Ok(())
    }

    /// Builds the program, with `overlays` over the files on disk (a test
    /// module standing in at its entry, say).
    pub fn build(&mut self, overlays: &[crate::Overlay]) -> Outcome {
        let outcome = self.attempt(overlays);
        // A build that failed before anything named its inputs (a compiler
        // that does not start) watches the whole checkout, so that the fix
        // sets the next build off.
        if self.watched.is_empty() {
            self.watched.insert(self.plugins.root().to_owned());
        }
        outcome
    }

    fn attempt(&mut self, overlays: &[crate::Overlay]) -> Outcome {
        let started = Instant::now();
        // A compiler that stopped (its process crashed on a source, or hung
        // past its deadline and was killed) fails every later build: a
        // fresh one gets one more try.
        let mut tries = 0;
        let result = loop {
            tries += 1;
            let built = self.compiler().and_then(|compiler| compiler.build(overlays));
            match built {
                Ok(result) => break result,
                Err(error) => {
                    self.compiler = None;
                    if tries == 2 {
                        return Outcome::Failed(format!("{error:#}"));
                    }
                }
            }
        };
        if let Some(compiler) = &mut self.compiler
            && let Ok(paths) = compiler.watched()
        {
            self.watched.extend(paths);
        }
        match result {
            BuildResult::Built(wasm) => Outcome::Built(Built { wasm, seconds: started.elapsed().as_secs_f64() }),
            BuildResult::Rejected(diagnostics) => Outcome::Rejected(diagnostics),
        }
    }

    fn compiler(&mut self) -> Result<&mut Compiler> {
        if self.compiler.is_none() {
            self.compiler = Some(self.plugins.compiler(&self.entry)?);
        }
        Ok(self.compiler.as_mut().expect("opened above"))
    }
}

/// Builds the program at `entry` once.
pub fn build(plugins: Arc<Plugins>, entry: &Path) -> Outcome {
    let mut builder = Builder::new(plugins, entry);
    let outcome = builder.build(&[]);
    let _ = builder.close();
    outcome
}

/// A program being built again as its files change; dropping it stops.
pub struct Watching {
    stop: Option<std::sync::mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Watching {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Builds the program at `entry`, hands `each` the outcome, and does so again
/// whenever a watched file changes, on a thread of its own until the result
/// is dropped. `ignored` are directories whose changes never count (what pen
/// writes the builds to). It watches the files with a watcher of its own.
pub fn watch(
    plugins: Arc<Plugins>,
    entry: &Path,
    ignored: Vec<PathBuf>,
    mut each: impl FnMut(Outcome) + Send + 'static,
) -> Result<Watching> {
    let (sender, changes) = channel();
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
        if let Ok(event) = event
            && !matches!(event.kind, EventKind::Access(_))
        {
            let _ = sender.send(event.paths);
        }
    })?;
    let mut watching = BTreeSet::new();
    watch_changes(plugins, entry, ignored, changes, move |outcome, watched| {
        for path in watched.difference(&watching).cloned().collect::<Vec<_>>() {
            let mode = if path.is_dir() { RecursiveMode::Recursive } else { RecursiveMode::NonRecursive };
            // A file not there yet is watched through its directory.
            let target = if path.exists() { path.clone() } else { path.parent().unwrap_or(&path).to_owned() };
            if watcher.watch(&target, mode).is_ok() {
                watching.insert(path);
            }
        }
        each(outcome);
    })
}

/// [`watch`], with the changes from a watcher of the caller's: `changes`
/// carries the paths of each change to a file or directory (not a read of
/// one), and ends the builds when it closes. `each` gets each outcome with
/// the paths whose changes count, files or directories with everything under
/// them: the caller's watcher must cover them before `each` returns, so that
/// an edit made as soon as the outcome is seen does not slip past.
pub fn watch_changes(
    plugins: Arc<Plugins>,
    entry: &Path,
    ignored: Vec<PathBuf>,
    changes: Receiver<Vec<PathBuf>>,
    mut each: impl FnMut(Outcome, &BTreeSet<PathBuf>) + Send + 'static,
) -> Result<Watching> {
    let mut program = Builder::new(plugins, entry);
    let (stop, stopped) = channel::<()>();
    let thread = std::thread::Builder::new().name("pen-plugin-build".into()).spawn(move || {
        loop {
            let outcome = program.build(&[]);
            each(outcome, &program.watched);
            // Wait for a change that counts, then for the files to settle.
            let mut changed: Option<Instant> = None;
            loop {
                if stopped.try_recv() != Err(std::sync::mpsc::TryRecvError::Empty) {
                    let _ = program.close();
                    return;
                }
                let wait = changed.map_or(Duration::from_millis(100), |at| SETTLE.saturating_sub(at.elapsed()));
                match changes.recv_timeout(wait) {
                    Ok(paths) => {
                        let counts = paths.iter().any(|path| {
                            !ignored.iter().any(|ignored| path.starts_with(ignored)) && program.needs(path)
                        });
                        if counts {
                            changed = Some(Instant::now());
                        }
                    }
                    Err(RecvTimeoutError::Timeout) if changed.is_some_and(|at| at.elapsed() >= SETTLE) => break,
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => {
                        let _ = program.close();
                        return;
                    }
                }
            }
        }
    })?;
    Ok(Watching { stop: Some(stop), thread: Some(thread) })
}
