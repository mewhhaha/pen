//! What a plugin may reach: the imports of `wit/pen-plugin.wit` (`log`,
//! `files`, `process`) and WASI's clocks, randomness and output, under the
//! grants its checkout's `pen.json` gives it.
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use wasmtime::component::{Resource, ResourceTable};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

use crate::bindings::pen::plugin::{files, log, process, types};

/// How much of a child's standard error is kept.
const ERROR_TAIL: usize = 64 * 1024;

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
}

impl State {
    pub fn new(grants: Arc<Grants>, log: Logger) -> State {
        // Its output reaches pen's; no files, environment or network.
        let wasi = WasiCtxBuilder::new().inherit_stdout().inherit_stderr().build();
        State { wasi, table: ResourceTable::new(), grants, log }
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
        let child = self.table.get_mut(&child).map_err(|error| process::ReadError::Other(error.to_string()))?;
        if child.pending.is_empty() {
            child.pending = match timeout_ms {
                Some(milliseconds) => match child.output.recv_timeout(Duration::from_millis(milliseconds.into())) {
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
        let child = self.table.get_mut(&child).ok()?;
        let deadline = timeout_ms.map(|milliseconds| Instant::now() + Duration::from_millis(milliseconds.into()));
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
