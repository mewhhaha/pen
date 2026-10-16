//! What a plugin may reach: the imports of `wit/pen-plugin.wit` (`log`,
//! `files`, `process`) and WASI's clocks, randomness and output, under the
//! grants its checkout's `pen.json` gives it, within its store's limits and
//! its calls' deadlines.
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use wasmtime::component::{Resource, ResourceTable};
use wasmtime::{StoreLimits, StoreLimitsBuilder};
use wasmtime_wasi::cli::{IsTerminal, StdoutStream};
use wasmtime_wasi::p2::{OutputStream, Pollable, StreamResult};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

use crate::bindings::pen::plugin::{files, log, process, types};

/// How much of a child's standard error is kept, and the longest line of a
/// plugin's own output.
const ERROR_TAIL: usize = 64 * 1024;
/// The most memory a plugin instance may have, and the most elements a
/// table of its may.
const MEMORY: usize = 1 << 30;
const TABLE_ELEMENTS: usize = 1 << 20;

/// Where a plugin's log lines go.
pub type Logger = Arc<dyn Fn(Level, &str) + Send + Sync>;

/// A log line's weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info,
    Error,
}

/// What a checkout grants one plugin: the roots it may read under, the
/// programs it may run by name, and its own cache directory.
pub struct Grants {
    pub root: PathBuf,
    pub read: Vec<PathBuf>,
    pub run: HashMap<String, PathBuf>,
    pub cache: PathBuf,
}

impl Grants {
    /// `path`, made canonical, if it lies under a granted root.
    fn readable(&self, path: &str) -> std::result::Result<PathBuf, String> {
        let path = Path::new(path);
        if !path.is_absolute() {
            return Err(format!("{} is not an absolute path", path.display()));
        }
        let canonical = path.canonicalize().map_err(|error| format!("{}: {error}", path.display()))?;
        if self.read.iter().any(|root| canonical.starts_with(root)) {
            Ok(canonical)
        } else {
            Err(format!("{} is outside what pen.json lets this plugin read", path.display()))
        }
    }
}

/// One plugin instance's state in its store.
pub struct State {
    wasi: WasiCtx,
    table: ResourceTable,
    grants: Arc<Grants>,
    log: Logger,
    pub limits: StoreLimits,
    /// When the call into the plugin under way must end: the host's waits
    /// end there, and the plugin's code traps `GRACE` later.
    pub deadline: Option<Instant>,
}

impl State {
    pub fn new(grants: Arc<Grants>, log: Logger) -> State {
        // Its output's lines reach pen's log; no files, environment or
        // network.
        let wasi = WasiCtxBuilder::new()
            .stdout(LogOutput::new(log.clone(), Level::Info))
            .stderr(LogOutput::new(log.clone(), Level::Error))
            .build();
        let limits = StoreLimitsBuilder::new().memory_size(MEMORY).table_elements(TABLE_ELEMENTS).build();
        State { wasi, table: ResourceTable::new(), grants, log, limits, deadline: None }
    }

    /// How long a wait of up to `timeout_ms` (none: no end) may last: until
    /// the call's deadline at most, and once past it, until half its grace
    /// is up (a plugin winding up waits for what it stopped). None when
    /// nothing ends it.
    fn bounded(&self, timeout_ms: Option<u32>) -> Option<Duration> {
        let asked = timeout_ms.map(|milliseconds| Duration::from_millis(milliseconds.into()));
        let now = Instant::now();
        let end = match self.deadline {
            None => return asked,
            Some(deadline) if now < deadline => deadline,
            Some(deadline) => deadline + crate::GRACE / 2,
        };
        let left = end.saturating_duration_since(now);
        Some(asked.map_or(left, |asked| asked.min(left)))
    }
}

/// A plugin's standard output or error: each line it writes reaches pen's
/// log at `level`.
#[derive(Clone)]
struct LogOutput {
    log: Logger,
    level: Level,
    /// What was written after the last line's end.
    partial: Arc<Mutex<Vec<u8>>>,
}

impl LogOutput {
    fn new(log: Logger, level: Level) -> LogOutput {
        LogOutput { log, level, partial: Arc::default() }
    }

    /// Takes `bytes`, logging each line they end; a line longer than
    /// [`ERROR_TAIL`] is logged in parts.
    fn take(&self, bytes: &[u8]) {
        let mut partial = self.partial.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        partial.extend_from_slice(bytes);
        loop {
            let end = match partial.iter().position(|byte| *byte == b'\n') {
                Some(end) => end + 1,
                None if partial.len() >= ERROR_TAIL => ERROR_TAIL,
                None => return,
            };
            let line: Vec<u8> = partial.drain(..end).collect();
            let line = String::from_utf8_lossy(&line);
            (self.log)(self.level, line.trim_end_matches(['\n', '\r']));
        }
    }
}

impl IsTerminal for LogOutput {
    fn is_terminal(&self) -> bool {
        false
    }
}

impl StdoutStream for LogOutput {
    fn p2_stream(&self) -> Box<dyn OutputStream> {
        Box::new(self.clone())
    }

    fn async_stream(&self) -> Box<dyn tokio::io::AsyncWrite + Send + Sync> {
        Box::new(self.clone())
    }
}

#[wasmtime_wasi::async_trait]
impl Pollable for LogOutput {
    async fn ready(&mut self) {}
}

impl OutputStream for LogOutput {
    fn write(&mut self, bytes: bytes::Bytes) -> StreamResult<()> {
        self.take(&bytes);
        Ok(())
    }

    fn flush(&mut self) -> StreamResult<()> {
        Ok(())
    }

    fn check_write(&mut self) -> StreamResult<usize> {
        Ok(ERROR_TAIL)
    }
}

impl tokio::io::AsyncWrite for LogOutput {
    fn poll_write(self: Pin<&mut Self>, _: &mut TaskContext<'_>, bytes: &[u8]) -> Poll<std::io::Result<usize>> {
        self.take(bytes);
        Poll::Ready(Ok(bytes.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _: &mut TaskContext<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _: &mut TaskContext<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

impl WasiView for State {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView { ctx: &mut self.wasi, table: &mut self.table }
    }
}

// What plugins and pen exchange: types only.
impl types::Host for State {}

impl log::Host for State {
    fn write(&mut self, level: log::Level, message: String) {
        let level = match level {
            log::Level::Info => Level::Info,
            log::Level::Error => Level::Error,
        };
        (self.log)(level, &message);
    }
}

impl files::Host for State {
    fn read(&mut self, path: String) -> std::result::Result<Vec<u8>, String> {
        let path = self.grants.readable(&path)?;
        std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))
    }

    fn read_directory(&mut self, directory: String) -> std::result::Result<Vec<String>, String> {
        let directory = self.grants.readable(&directory)?;
        let entries = std::fs::read_dir(&directory).map_err(|error| format!("{}: {error}", directory.display()))?;
        let mut names: Vec<String> =
            entries.flatten().map(|entry| entry.file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        Ok(names)
    }

    fn exists(&mut self, path: String) -> bool {
        self.grants.readable(&path).is_ok()
    }

    fn is_file(&mut self, path: String) -> bool {
        self.grants.readable(&path).is_ok_and(|path| path.is_file())
    }

    fn canonical(&mut self, path: String) -> std::result::Result<String, String> {
        self.grants.readable(&path).map(|path| crate::portable(&path))
    }
}

/// A program a plugin runs, its output read on threads of its own: standard
/// output in chunks, standard error into a bounded tail.
pub struct ChildProcess {
    child: Child,
    input: Option<ChildStdin>,
    output: Receiver<Vec<u8>>,
    pending: Vec<u8>,
    errors: Arc<Mutex<String>>,
    /// The thread reading standard error, until it has read the last line.
    reading_errors: Option<std::thread::JoinHandle<()>>,
}

impl ChildProcess {
    fn spawn(program: &Path, args: &[String], env: &[(String, String)], directory: &Path) -> Result<ChildProcess> {
        let mut child = Command::new(program)
            .args(args)
            .env_clear()
            .envs(env.iter().map(|(name, value)| (name, value)))
            .current_dir(directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("starting {}", program.display()))?;
        let (sender, output) = channel();
        let mut stdout = child.stdout.take().context("no standard output")?;
        std::thread::Builder::new().name("pen-plugin-output".into()).spawn(move || {
            let mut buffer = vec![0; 64 * 1024];
            while let Ok(count) = stdout.read(&mut buffer) {
                if count == 0 || sender.send(buffer[..count].to_vec()).is_err() {
                    break;
                }
            }
        })?;
        let errors = Arc::new(Mutex::new(String::new()));
        let mut stderr = child.stderr.take().context("no standard error")?;
        let tail = errors.clone();
        let reading_errors = std::thread::Builder::new().name("pen-plugin-errors".into()).spawn(move || {
            let mut buffer = vec![0; 8 * 1024];
            while let Ok(count) = stderr.read(&mut buffer) {
                if count == 0 {
                    break;
                }
                let mut tail = tail.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                tail.push_str(&String::from_utf8_lossy(&buffer[..count]));
                if tail.len() > ERROR_TAIL {
                    let mut cut = tail.len() - ERROR_TAIL;
                    while !tail.is_char_boundary(cut) {
                        cut += 1;
                    }
                    tail.drain(..cut);
                }
            }
        })?;
        let input = child.stdin.take();
        Ok(ChildProcess { child, input, output, pending: Vec::new(), errors, reading_errors: Some(reading_errors) })
    }

    /// Once it has exited, its standard error is read to the end: the last
    /// line a failing program writes (blotc's `blotc: <ErrorName>`) is
    /// what says why.
    fn finish_errors(&mut self) {
        if matches!(self.child.try_wait(), Ok(Some(_)))
            && let Some(thread) = self.reading_errors.take()
        {
            let _ = thread.join();
        }
    }
}

impl Drop for ChildProcess {
    fn drop(&mut self) {
        drop(self.input.take());
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

/// A gzipped program (`blotc.gz`) unpacked once into pen's cache, by its
/// digest, and made executable; any other program as it is.
fn runnable(program: &Path) -> Result<PathBuf> {
    if program.extension().is_none_or(|extension| extension != "gz") {
        return Ok(program.to_owned());
    }
    let packed = std::fs::read(program).with_context(|| format!("reading {}", program.display()))?;
    let digest = format!("{:x}", Sha256::digest(&packed));
    let name = program.file_stem().map_or_else(|| "program".into(), |stem| stem.to_owned());
    let directory = crate::cache_root().join("programs").join(&digest[..16]);
    let unpacked = directory.join(name);
    if unpacked.is_file() {
        crate::cache::used(&unpacked);
        return Ok(unpacked);
    }
    std::fs::create_dir_all(&directory)?;
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(&packed[..])
        .read_to_end(&mut bytes)
        .with_context(|| format!("unpacking {}", program.display()))?;
    let temporary = directory.join(format!(".unpacking.{}", std::process::id()));
    std::fs::write(&temporary, &bytes)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&temporary, std::fs::Permissions::from_mode(0o755))?;
    }
    std::fs::rename(&temporary, &unpacked)?;
    Ok(unpacked)
}

impl process::Host for State {
    fn spawn(
        &mut self,
        name: String,
        args: Vec<String>,
        env: Vec<(String, String)>,
    ) -> std::result::Result<Resource<ChildProcess>, String> {
        let Some(program) = self.grants.run.get(&name) else {
            return Err(format!("pen.json lets this plugin run no program named {name}"));
        };
        let child = runnable(program)
            .and_then(|program| ChildProcess::spawn(&program, &args, &env, &self.grants.root))
            .map_err(|error| format!("{error:#}"))?;
        self.table.push(child).map_err(|error| error.to_string())
    }

    fn cache_directory(&mut self) -> String {
        let _ = std::fs::create_dir_all(&self.grants.cache);
        crate::portable(&self.grants.cache)
    }
}

impl process::HostChild for State {
    fn write(&mut self, child: Resource<ChildProcess>, bytes: Vec<u8>) -> std::result::Result<(), String> {
        let child = self.table.get_mut(&child).map_err(|error| error.to_string())?;
        let input = child.input.as_mut().ok_or("its input is closed")?;
        input.write_all(&bytes).and_then(|()| input.flush()).map_err(|error| error.to_string())
    }

    fn close_input(&mut self, child: Resource<ChildProcess>) {
        if let Ok(child) = self.table.get_mut(&child) {
            drop(child.input.take());
        }
    }

    fn read(
        &mut self,
        child: Resource<ChildProcess>,
        max: u32,
        timeout_ms: Option<u32>,
    ) -> std::result::Result<Vec<u8>, process::ReadError> {
        let timeout = self.bounded(timeout_ms);
        let child = self.table.get_mut(&child).map_err(|error| process::ReadError::Other(error.to_string()))?;
        if child.pending.is_empty() {
            child.pending = match timeout {
                Some(timeout) => match child.output.recv_timeout(timeout) {
                    Ok(chunk) => chunk,
                    Err(RecvTimeoutError::Timeout) => return Err(process::ReadError::Timeout),
                    Err(RecvTimeoutError::Disconnected) => return Err(process::ReadError::Closed),
                },
                None => child.output.recv().map_err(|_| process::ReadError::Closed)?,
            };
        }
        let count = child.pending.len().min(max.max(1) as usize);
        Ok(child.pending.drain(..count).collect())
    }

    fn error_tail(&mut self, child: Resource<ChildProcess>) -> String {
        let Ok(child) = self.table.get_mut(&child) else { return String::new() };
        child.finish_errors();
        child.errors.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
    }

    fn wait(&mut self, child: Resource<ChildProcess>, timeout_ms: Option<u32>) -> Option<i32> {
        let deadline = self.bounded(timeout_ms).map(|timeout| Instant::now() + timeout);
        let child = self.table.get_mut(&child).ok()?;
        loop {
            match child.child.try_wait() {
                Ok(Some(status)) => {
                    child.finish_errors();
                    return status.code();
                }
                Ok(None) if deadline.is_none_or(|deadline| Instant::now() < deadline) => {
                    std::thread::sleep(Duration::from_millis(5))
                }
                _ => return None,
            }
        }
    }

    fn kill(&mut self, child: Resource<ChildProcess>) {
        if let Ok(child) = self.table.get_mut(&child) {
            let _ = child.child.kill();
        }
    }

    fn drop(&mut self, child: Resource<ChildProcess>) -> wasmtime::Result<()> {
        self.table.delete(child)?;
        Ok(())
    }
}

/// The checkout-relative or absolute `path`, made canonical when it exists.
pub fn resolve(root: &Path, path: &str) -> PathBuf {
    let joined = root.join(path);
    joined.canonicalize().unwrap_or(joined)
}

/// A program's grants, checked: every program it may run exists.
pub fn grants(root: &Path, read: &[String], run: &HashMap<String, String>, cache: PathBuf) -> Result<Grants> {
    let mut programs = HashMap::new();
    for (name, path) in run {
        let path = resolve(root, path);
        if !path.is_file() {
            bail!("pen.json grants program {name} at {}, which is not there", path.display());
        }
        programs.insert(name.clone(), path);
    }
    let mut roots = vec![root.to_owned()];
    roots.extend(read.iter().map(|path| resolve(root, path)));
    Ok(Grants { root: root.to_owned(), read: roots, run: programs, cache })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logs_a_plugins_output_by_the_line() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let said = lines.clone();
        let log: Logger = Arc::new(move |level, line| said.lock().unwrap().push(format!("{level:?} {line}")));
        let mut output = LogOutput::new(log, Level::Error);
        output.write(bytes::Bytes::from_static(b"panicked at src/lib.rs:1:\nbad")).unwrap();
        output.write(bytes::Bytes::from_static(b" input\r\n\n")).unwrap();
        assert_eq!(*lines.lock().unwrap(), ["Error panicked at src/lib.rs:1:", "Error bad input", "Error "]);
        // A line without an end is cut where the tail is.
        output.take(&vec![b'x'; ERROR_TAIL + 1]);
        assert_eq!(lines.lock().unwrap().len(), 4);
        assert_eq!(output.partial.lock().unwrap().len(), 1);
    }

    #[test]
    fn bounds_waits_by_the_calls_deadline() {
        let grants = Grants { root: PathBuf::new(), read: Vec::new(), run: HashMap::new(), cache: PathBuf::new() };
        let mut state = State::new(Arc::new(grants), Arc::new(|_, _| {}));
        assert_eq!(state.bounded(None), None);
        assert_eq!(state.bounded(Some(20)), Some(Duration::from_millis(20)));
        state.deadline = Some(Instant::now() + Duration::from_secs(5));
        assert!(state.bounded(None).is_some_and(|left| left <= Duration::from_secs(5)));
        assert_eq!(state.bounded(Some(20)), Some(Duration::from_millis(20)));
        assert!(state.bounded(Some(u32::MAX)).is_some_and(|left| left <= Duration::from_secs(5)));
        // Past the deadline, a plugin winding up still waits briefly, well
        // before its code traps.
        state.deadline = Some(Instant::now() - Duration::from_millis(1));
        assert_eq!(state.bounded(Some(20)), Some(Duration::from_millis(20)));
        for timeout in [None, Some(u32::MAX)] {
            assert!(state.bounded(timeout).is_some_and(|left| left < crate::GRACE / 2), "{timeout:?}");
        }
    }
}
