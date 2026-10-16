//! pen's Wasm plugins (`wit/pen-plugin.wit`): WebAssembly components a
//! checkout chooses in its `pen.json` to do what is specific to its language
//! and toolchain, so that pen itself only runs Wasm. A compiler plugin turns
//! a program's sources into its Wasm module, again as they change, and lays
//! out the checkout's tests: which modules, how each builds, and what each
//! test must do.
//!
//! ```json
//! {
//!   "plugins": {
//!     "blot": { "read": ["../blot2/std"], "run": { "blotc": "../blot2/generated/compiler/blotc.gz" } }
//!   },
//!   "compiler": { "plugin": "blot", "settings": { "imports": { "gdev/": "packages/" } } }
//! }
//! ```
//!
//! - `plugins` names each plugin with its grants: `read`, the directories it
//!   may read besides the checkout, which it always may; `run`, the programs
//!   it may start, by the names it asks for them (a `.gz` one is unpacked into
//!   pen's cache first); `wasm`, the component itself, when it is not one of
//!   pen's own (`PEN_PLUGINS`, `plugins/` beside the pen executable, or this
//!   repository's `plugins/target`, in that order); `timeout`, the seconds one
//!   call into it may take (a build, say: 300 unless given).
//! - `compiler` says which plugin compiles, and the `settings` (any JSON) it
//!   gets with each program.
//!
//! A plugin gets WASI's clocks, randomness and output (its lines reach the
//! log), and nothing else of the machine: no files, environment or network
//! but through its grants. Its memory is bounded, and a call that outlives
//! its `timeout` ends: what the plugin waits on times out, and soon after the
//! call traps, which pen takes as the compiler's failure and opens a fresh
//! one. Each component is compiled once and kept in pen's cache
//! (`components/`), which [`cache`] keeps from growing without end.
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use wasmtime::component::{HasSelf, Linker, ResourceAny};
use wasmtime::{Engine, Store};

pub mod cache;
mod host;
mod watch;

pub use host::{Level, Logger};
pub use watch::{Builder, Built, Outcome, Watching, build, watch, watch_changes};

/// The bindings `wit/pen-plugin.wit` makes for the compiler world.
pub(crate) mod bindings {
    wasmtime::component::bindgen!({
        path: "../../wit",
        world: "compiler-plugin",
        with: { "pen:plugin/process.child": crate::host::ChildProcess },
    });
}

pub use bindings::exports::pen::plugin::compiler::BuildResult;
pub use bindings::exports::pen::plugin::tests::{Expectation, TestModule};
pub use bindings::pen::plugin::types::{Diagnostic, Overlay};

/// How long one call into a plugin may take unless its pen.json says: a
/// cold build of a large program, with room to spare.
const TIMEOUT: Duration = Duration::from_secs(300);
/// How long past its deadline a call has to wind up (stop what it runs, say
/// why) before it traps.
const GRACE: Duration = Duration::from_secs(5);
/// How often the engine's epoch, which the deadlines count in, ticks.
const TICK: Duration = Duration::from_millis(100);

/// Plugin paths have one spelling on every platform.
pub(crate) fn portable(path: &Path) -> String {
    let text = path.to_string_lossy().into_owned();
    #[cfg(windows)]
    let text = text.replace('\\', "/");
    text
}

/// pen's cache directory for plugins and what they run.
pub(crate) fn cache_root() -> PathBuf {
    dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("pen")
}

/// A checkout's `pen.json`.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub plugins: BTreeMap<String, PluginConfig>,
    pub compiler: Option<Choice>,
    /// The game's name and title, which pen reads for itself.
    pub game: Option<serde_json::Value>,
}

/// One plugin's component and grants.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginConfig {
    pub wasm: Option<String>,
    #[serde(default)]
    pub read: Vec<String>,
    #[serde(default)]
    pub run: HashMap<String, String>,
    /// The seconds one call into the plugin may take.
    pub timeout: Option<f64>,
}

/// Which plugin does a job, and its settings.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Choice {
    pub plugin: String,
    #[serde(default)]
    pub settings: Option<serde_json::Value>,
}

/// A checkout's plugins: its `pen.json`, an engine for components, and each
/// plugin compiled (or loaded from pen's cache) and linked once, when first
/// used.
pub struct Plugins {
    root: PathBuf,
    config: Config,
    engine: Engine,
    log: Logger,
    linked: Mutex<HashMap<String, bindings::CompilerPluginPre<host::State>>>,
}

/// A program for a plugin: the checkout, its entry and the plugin's settings.
fn program(root: &Path, entry: &Path, settings: &Option<serde_json::Value>) -> bindings::pen::plugin::types::Program {
    bindings::pen::plugin::types::Program {
        root: portable(root),
        entry: portable(entry),
        settings: settings.as_ref().map_or_else(|| "{}".to_owned(), |settings| settings.to_string()),
    }
}

/// Ticks `engine`'s epoch for as long as it is used.
fn tick(engine: &Engine) -> Result<()> {
    let engine = engine.weak();
    std::thread::Builder::new().name("pen-plugin-epoch".into()).spawn(move || {
        while let Some(engine) = engine.upgrade() {
            engine.increment_epoch();
            drop(engine);
            std::thread::sleep(TICK);
        }
    })?;
    Ok(())
}

/// Runs `call` into the plugin in `store` with its deadline `timeout` away:
/// the host's waits end there, and the plugin's own code traps `GRACE`
/// after.
fn within<T>(store: &mut Store<host::State>, timeout: Duration, call: impl FnOnce(&mut Store<host::State>) -> T) -> T {
    store.data_mut().deadline = Some(Instant::now() + timeout);
    let ticks = (timeout + GRACE).as_millis() / TICK.as_millis() + 1;
    store.set_epoch_deadline(ticks.try_into().unwrap_or(u64::MAX));
    let result = call(store);
    store.data_mut().deadline = None;
    result
}

/// A failed call into plugin `name`: a trap at its deadline says so.
fn failed(name: &str, timeout: Duration, error: wasmtime::Error) -> anyhow::Error {
    if error.downcast_ref::<wasmtime::Trap>() == Some(&wasmtime::Trap::Interrupt) {
        return anyhow!("plugin {name} did not return within {} s (its `timeout` in pen.json)", timeout.as_secs_f64());
    }
    anyhow::Error::from(error).context(format!("plugin {name}"))
}

impl Plugins {
    /// The plugins of the checkout at `root`, if it has a `pen.json`. `log`
    /// takes what plugins write to pen's log.
    pub fn load(root: &Path, log: Logger) -> Result<Option<Arc<Plugins>>> {
        let file = root.join("pen.json");
        if !file.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&file).with_context(|| format!("reading {}", file.display()))?;
        let config: Config = serde_json::from_str(&text).with_context(|| format!("reading {}", file.display()))?;
        if let Some(choice) = &config.compiler
            && !config.plugins.contains_key(&choice.plugin)
        {
            bail!("{} uses plugin {}, which its `plugins` does not name", file.display(), choice.plugin);
        }
        for (name, plugin) in &config.plugins {
            if plugin.timeout.is_some_and(|seconds| !(seconds > 0.0 && seconds <= 1e6)) {
                bail!("{}: plugin {name}'s `timeout` is not a number of seconds above 0", file.display());
            }
        }
        cache::prune_once();
        let mut engine = wasmtime::Config::new();
        engine.wasm_component_model(true).epoch_interruption(true);
        let engine = Engine::new(&engine)?;
        tick(&engine)?;
        let root = root.canonicalize()?;
        Ok(Some(Arc::new(Plugins { root, config, engine, log, linked: Mutex::new(HashMap::new()) })))
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The checkout's root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Whether the checkout compiles through a plugin.
    pub fn compiles(&self) -> bool {
        self.config.compiler.is_some()
    }

    /// The compiler's choice: its plugin and settings.
    fn chosen(&self) -> Result<&Choice> {
        self.config.compiler.as_ref().ok_or_else(|| anyhow!("pen.json chooses no compiler"))
    }

    /// How long one call into plugin `name` may take.
    fn timeout(&self, name: &str) -> Duration {
        let seconds = self.config.plugins.get(name).and_then(|plugin| plugin.timeout);
        seconds.map_or(TIMEOUT, Duration::from_secs_f64)
    }

    /// Where a plugin's component is.
    fn locate(&self, name: &str) -> Result<PathBuf> {
        let configured = self.config.plugins.get(name).and_then(|plugin| plugin.wasm.as_deref());
        if let Some(path) = configured {
            return Ok(host::resolve(&self.root, path));
        }
        let file = format!("{name}.wasm");
        let mut tried = Vec::new();
        if let Some(directory) = std::env::var_os("PEN_PLUGINS") {
            tried.push(PathBuf::from(directory).join(&file));
        }
        if let Some(directory) = std::env::current_exe().ok().and_then(|exe| Some(exe.parent()?.join("plugins"))) {
            tried.push(directory.join(&file));
        }
        tried.push(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../plugins/target/wasm32-wasip2/release")
                .join(format!("pen_plugin_{}.wasm", name.replace('-', "_"))),
        );
        tried.iter().find(|path| path.is_file()).cloned().ok_or_else(|| {
            let looked: Vec<String> = tried.iter().map(|path| path.display().to_string()).collect();
            anyhow!("no plugin {name}: looked for {}", looked.join(", "))
        })
    }

    /// Plugin `name`'s component, compiled and linked with pen's imports.
    fn linked(&self, name: &str) -> Result<bindings::CompilerPluginPre<host::State>> {
        let mut linked = self.linked.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(plugin) = linked.get(name) {
            return Ok(plugin.clone());
        }
        let path = self.locate(name)?;
        let loading = || -> Result<bindings::CompilerPluginPre<host::State>> {
            let bytes = std::fs::read(&path)?;
            let component = cache::component(&self.engine, &bytes)?;
            let mut linker = Linker::<host::State>::new(&self.engine);
            wasmtime_wasi::p2::add_to_linker_sync(&mut linker)?;
            bindings::CompilerPlugin::add_to_linker::<_, HasSelf<_>>(&mut linker, |state| state)?;
            Ok(bindings::CompilerPluginPre::new(linker.instantiate_pre(&component)?)?)
        };
        let plugin = loading().with_context(|| format!("loading plugin {name} from {}", path.display()))?;
        linked.insert(name.to_owned(), plugin.clone());
        Ok(plugin)
    }

    /// Plugin `name` started in a fresh store, with its grants.
    fn start(&self, name: &str) -> Result<(Store<host::State>, bindings::CompilerPlugin)> {
        let linked = self.linked(name)?;
        let plugin = self.config.plugins.get(name).ok_or_else(|| anyhow!("pen.json names no plugin {name}"))?;
        let cache = cache_root().join("plugins").join(name);
        let grants = host::grants(&self.root, &plugin.read, &plugin.run, cache)?;
        let mut store = Store::new(&self.engine, host::State::new(Arc::new(grants), self.log.clone()));
        store.limiter(|state| &mut state.limits);
        let timeout = self.timeout(name);
        let plugin = within(&mut store, timeout, |store| linked.instantiate(store))
            .map_err(|error| failed(name, timeout, error))
            .with_context(|| format!("starting plugin {name}"))?;
        Ok((store, plugin))
    }

    /// Opens a compiler session for the program whose entry is `entry`.
    pub fn compiler(&self, entry: &Path) -> Result<Compiler> {
        let choice = self.chosen()?;
        let name = choice.plugin.clone();
        let (mut store, plugin) = self.start(&name)?;
        let program = program(&self.root, entry, &choice.settings);
        let timeout = self.timeout(&name);
        let session =
            within(&mut store, timeout, |store| plugin.pen_plugin_compiler().session().call_open(store, &program))
                .map_err(|error| failed(&name, timeout, error))?
                .map_err(|error| anyhow!("{name}: {error}"))?;
        Ok(Compiler { store, plugin, session, name, timeout })
    }

    /// The checkout's test modules to run, in order, as its compiler plugin
    /// lays them out: those at `paths`, or all of them when none, less those
    /// none of whose tests can have a name holding `filter`. `entry` is the
    /// checkout's main program.
    pub fn tests(&self, entry: &Path, paths: &[PathBuf], filter: Option<&str>) -> Result<Vec<TestModule>> {
        let choice = self.chosen()?;
        let name = &choice.plugin;
        let (mut store, plugin) = self.start(name)?;
        let program = program(&self.root, entry, &choice.settings);
        let paths: Vec<String> = paths.iter().map(|path| portable(path)).collect();
        let timeout = self.timeout(name);
        within(&mut store, timeout, |store| plugin.pen_plugin_tests().call_find(store, &program, &paths, filter))
            .map_err(|error| failed(name, timeout, error))?
            .map_err(|error| anyhow!("{name}: {error}"))
    }
}

/// A compiler plugin's session for one program.
pub struct Compiler {
    store: Store<host::State>,
    plugin: bindings::CompilerPlugin,
    session: ResourceAny,
    name: String,
    timeout: Duration,
}

impl Compiler {
    /// Builds the program, with `overlays` over the files on disk. An error
    /// is the compiler's own failure (it may have hung), not the program's.
    pub fn build(&mut self, overlays: &[Overlay]) -> Result<BuildResult> {
        let Compiler { store, plugin, session, name, timeout } = self;
        within(store, *timeout, |store| plugin.pen_plugin_compiler().session().call_build(store, *session, overlays))
            .map_err(|error| failed(name, *timeout, error))?
            .map_err(|error| anyhow!(error))
    }

    /// The paths whose changes need a new build.
    pub fn watched(&mut self) -> Result<Vec<PathBuf>> {
        let Compiler { store, plugin, session, name, timeout } = self;
        let paths =
            within(store, *timeout, |store| plugin.pen_plugin_compiler().session().call_watched(store, *session))
                .map_err(|error| failed(name, *timeout, error))?;
        Ok(paths.into_iter().map(PathBuf::from).collect())
    }

    /// Gracefully saves the compiler's retained work. Drop itself is fast.
    pub fn close(&mut self) -> Result<()> {
        let Compiler { store, plugin, session, name, timeout } = self;
        within(store, *timeout, |store| plugin.pen_plugin_compiler().session().call_close(store, *session))
            .map_err(|error| failed(name, *timeout, error))?
            .map_err(anyhow::Error::msg)
    }
}

impl Drop for Compiler {
    fn drop(&mut self) {
        // What it runs is killed with the store, should this trap.
        let session = self.session;
        let _ = within(&mut self.store, self.timeout, |store| session.resource_drop(store));
    }
}
