//! pen's Wasm plugins (`wit/pen-plugin.wit`): WebAssembly components a
//! checkout chooses in its `pen.json` to do what is specific to its language
//! and toolchain, so that pen itself only runs Wasm. A compiler plugin turns
//! a program's sources into its Wasm module, again as they change; an
//! assets plugin makes the virtual modules a program imports its assets
//! through, which pen hands the compiler as overlays.
//!
//! ```json
//! {
//!   "plugins": {
//!     "blot": { "read": ["../blot2/std"], "run": { "blotc": "../blot2/generated/compiler/blotc.gz" } },
//!     "gdev-assets": {}
//!   },
//!   "compiler": { "plugin": "blot", "settings": { "imports": { "gdev/": "packages/" } } },
//!   "assets": { "plugin": "gdev-assets" }
//! }
//! ```
//!
//! - `plugins` names each plugin with its grants: `read`, the directories it
//!   may read besides the checkout, which it always may; `run`, the programs
//!   it may start, by the names it asks for them (a `.gz` one is unpacked into
//!   pen's cache first); `wasm`, the component itself, when it is not one of
//!   pen's own (`PEN_PLUGINS`, `plugins/` beside the pen executable, or this
//!   repository's `plugins/target`, in that order).
//! - `compiler` and `assets` say which plugin does each, and the `settings`
//!   (any JSON) it gets with each program.
//!
//! A plugin gets WASI's clocks, randomness and output, and nothing else of
//! the machine: no files, environment or network but through its grants.
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use wasmtime::component::{Component, HasSelf, Linker, ResourceAny};
use wasmtime::{Engine, Store};

mod host;
mod watch;

pub use host::{Level, Logger};
pub use watch::{Builder, Built, Outcome, Watching, build, watch};

/// The bindings `wit/pen-plugin.wit` makes for the compiler world, whose
/// imports the assets world shares.
pub(crate) mod bindings {
    wasmtime::component::bindgen!({
        path: "../../wit",
        world: "compiler-plugin",
        with: { "pen:plugin/process.child": crate::host::ChildProcess },
    });
}

mod assets_bindings {
    wasmtime::component::bindgen!({
        path: "../../wit",
        world: "assets-plugin",
        with: {
            "pen:plugin/types": crate::bindings::pen::plugin::types,
            "pen:plugin/log": crate::bindings::pen::plugin::log,
            "pen:plugin/files": crate::bindings::pen::plugin::files,
        },
    });
}

pub use bindings::exports::pen::plugin::compiler::BuildResult;
pub use bindings::pen::plugin::types::{Diagnostic, Overlay};

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
    pub assets: Option<Choice>,
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
/// plugin compiled once, when first used.
pub struct Plugins {
    root: PathBuf,
    config: Config,
    engine: Engine,
    log: Logger,
    compiled: Mutex<HashMap<String, Component>>,
}

/// A program for a plugin: the checkout, its entry and the plugin's settings.
fn program(root: &Path, entry: &Path, settings: &Option<serde_json::Value>) -> bindings::pen::plugin::types::Program {
    bindings::pen::plugin::types::Program {
        root: portable(root),
        entry: portable(entry),
        settings: settings.as_ref().map_or_else(|| "{}".to_owned(), |settings| settings.to_string()),
    }
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
        for choice in config.compiler.iter().chain(&config.assets) {
            if !config.plugins.contains_key(&choice.plugin) {
                bail!("{} uses plugin {}, which its `plugins` does not name", file.display(), choice.plugin);
            }
        }
        let mut engine = wasmtime::Config::new();
        engine.wasm_component_model(true);
        let engine = Engine::new(&engine)?;
        let root = root.canonicalize()?;
        Ok(Some(Arc::new(Plugins { root, config, engine, log, compiled: Mutex::new(HashMap::new()) })))
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

    fn component(&self, name: &str) -> Result<Component> {
        let mut compiled = self.compiled.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(component) = compiled.get(name) {
            return Ok(component.clone());
        }
        let path = self.locate(name)?;
        let component = Component::from_file(&self.engine, &path)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("loading plugin {name} from {}", path.display()))?;
        compiled.insert(name.to_owned(), component.clone());
        Ok(component)
    }

    /// A fresh store for plugin `name`, with its grants.
    fn store(&self, name: &str) -> Result<Store<host::State>> {
        let plugin = self.config.plugins.get(name).ok_or_else(|| anyhow!("pen.json names no plugin {name}"))?;
        let cache = cache_root().join("plugins").join(name);
        let grants = host::grants(&self.root, &plugin.read, &plugin.run, cache)?;
        let log = self.log.clone();
        Ok(Store::new(&self.engine, host::State::new(Arc::new(grants), log)))
    }

    /// Opens a compiler session for the program whose entry is `entry`.
    pub fn compiler(&self, entry: &Path) -> Result<Compiler> {
        let choice = self.config.compiler.as_ref().ok_or_else(|| anyhow!("pen.json chooses no compiler"))?;
        let component = self.component(&choice.plugin)?;
        let mut store = self.store(&choice.plugin)?;
        let mut linker = Linker::<host::State>::new(&self.engine);
        wasmtime_wasi::p2::add_to_linker_sync(&mut linker)?;
        bindings::CompilerPlugin::add_to_linker::<_, HasSelf<_>>(&mut linker, |state| state)?;
        let plugin = bindings::CompilerPlugin::instantiate(&mut store, &component, &linker)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("starting plugin {}", choice.plugin))?;
        let program = program(&self.root, entry, &choice.settings);
        let session = plugin
            .pen_plugin_compiler()
            .session()
            .call_open(&mut store, &program)?
            .map_err(|error| anyhow!("{}: {error}", choice.plugin))?;
        Ok(Compiler { store, plugin, session })
    }

    /// The checkout's assets plugin, if it chose one, for the program whose
    /// entry is `entry`.
    pub fn assets(&self, entry: &Path) -> Result<Option<Assets>> {
        let Some(choice) = &self.config.assets else { return Ok(None) };
        let component = self.component(&choice.plugin)?;
        let mut store = self.store(&choice.plugin)?;
        let mut linker = Linker::<host::State>::new(&self.engine);
        wasmtime_wasi::p2::add_to_linker_sync(&mut linker)?;
        assets_bindings::AssetsPlugin::add_to_linker::<_, HasSelf<_>>(&mut linker, |state| state)?;
        let plugin = assets_bindings::AssetsPlugin::instantiate(&mut store, &component, &linker)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("starting plugin {}", choice.plugin))?;
        let program = program(&self.root, entry, &choice.settings);
        Ok(Some(Assets { store, plugin, program, name: choice.plugin.clone() }))
    }
}

/// A compiler plugin's session for one program.
pub struct Compiler {
    store: Store<host::State>,
    plugin: bindings::CompilerPlugin,
    session: ResourceAny,
}

impl Compiler {
    /// Builds the program, with `overlays` over the files on disk. An error
    /// is the compiler's own failure, not the program's.
    pub fn build(&mut self, overlays: &[Overlay]) -> Result<BuildResult> {
        self.plugin
            .pen_plugin_compiler()
            .session()
            .call_build(&mut self.store, self.session, overlays)?
            .map_err(|error| anyhow!(error))
    }

    /// The paths whose changes need a new build.
    pub fn watched(&mut self) -> Result<Vec<PathBuf>> {
        let paths = self.plugin.pen_plugin_compiler().session().call_watched(&mut self.store, self.session)?;
        Ok(paths.into_iter().map(PathBuf::from).collect())
    }

    /// Gracefully saves the compiler's retained work. Drop itself is fast.
    pub fn close(&mut self) -> Result<()> {
        self.plugin.pen_plugin_compiler().session().call_close(&mut self.store, self.session)?
            .map_err(anyhow::Error::msg)
    }
}

impl Drop for Compiler {
    fn drop(&mut self) {
        let _ = self.session.resource_drop(&mut self.store);
    }
}

/// An assets plugin, for one program.
pub struct Assets {
    store: Store<host::State>,
    plugin: assets_bindings::AssetsPlugin,
    program: bindings::pen::plugin::types::Program,
    name: String,
}

/// What an assets plugin made: the modules, and what they depend on.
pub struct Generated {
    pub modules: Vec<Overlay>,
    pub dependencies: Vec<PathBuf>,
}

#[derive(Debug)]
pub struct GenerationError {
    pub message: String,
    pub dependencies: Vec<PathBuf>,
}

impl Assets {
    pub fn generate(&mut self) -> std::result::Result<Generated, GenerationError> {
        let generated = self
            .plugin
            .pen_plugin_assets()
            .call_generate(&mut self.store, &self.program)
            .map_err(|error| GenerationError { message: format!("{}: {error}", self.name), dependencies: Vec::new() })?
            .map_err(|error| GenerationError {
                message: format!("{}: {}", self.name, error.message),
                dependencies: error.dependencies.into_iter().map(PathBuf::from).collect(),
            })?;
        Ok(Generated {
            modules: generated.modules,
            dependencies: generated.dependencies.into_iter().map(PathBuf::from).collect(),
        })
    }
}
