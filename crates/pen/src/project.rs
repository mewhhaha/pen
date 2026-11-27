//! A checkout as a project the binary develops, runs and packages: its
//! programs compile through the Wasm plugins its `pen.json` chooses
//! (`plugin_build`), and its assets and scene are used in place.
#[cfg(feature = "compiler")]
use std::collections::{BTreeSet, HashSet};
use std::path::Path;
#[cfg(any(feature = "compiler", test))]
use std::path::PathBuf;
#[cfg(feature = "compiler")]
use std::process::Command;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

/// Who a game is: `name` names its data directory (saves, preferences and
/// crash reports) and the executable `pen build` ships it as; `title` is its
/// window's, which assistive technology reads too. `players` is how many
/// players it takes, each its own peer in a net session (docs/netplay.md):
/// all the netcode a game declares.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Identity {
    pub name: String,
    pub title: String,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub players: u32,
}

fn one() -> u32 {
    1
}

fn is_one(players: &u32) -> bool {
    *players == 1
}

impl Identity {
    /// A game called `name` (made a file name), with that title too.
    pub fn named(name: &str) -> Identity {
        let name: String =
            name.chars().map(|character| if file_name_character(character) { character } else { '-' }).collect();
        let name = name.trim_start_matches('.').trim_end_matches([' ', '.']);
        let name = if name.is_empty() { "game" } else { name };
        Identity { name: name.to_owned(), title: name.to_owned(), players: 1 }
    }

    /// The `game` block of a checkout's `pen.json`, `{"name": "gdev",
    /// "title": "gdev — Blot 3D", "players": 2}`, any of which may be left
    /// out: the name is then the checkout directory's, the title the name,
    /// and the players one.
    #[cfg(feature = "compiler")]
    fn of_checkout(root: &Path) -> Result<Identity> {
        let file = root.join("pen.json");
        let reading = || format!("reading {}", file.display());
        let text = std::fs::read_to_string(&file).with_context(reading)?;
        let config: serde_json::Value = serde_json::from_str(&text).with_context(reading)?;
        let field = |key: &str| -> Result<Option<String>> {
            match config.get("game").and_then(|game| game.get(key)) {
                None => Ok(None),
                Some(serde_json::Value::String(text)) => Ok(Some(text.clone())),
                Some(_) => bail!("{}: game.{key} must be a string", file.display()),
            }
        };
        let directory = root.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        let mut identity = Identity::named(&directory);
        if let Some(name) = field("name")? {
            identity = Identity { title: name.clone(), name, players: 1 };
            identity.check().with_context(|| format!("{}: game.name", file.display()))?;
        }
        if let Some(title) = field("title")? {
            identity.title = title;
        }
        if let Some(players) = config.get("game").and_then(|game| game.get("players")) {
            identity.players = players
                .as_u64()
                .and_then(|players| u32::try_from(players).ok())
                .filter(|players| (1..=32).contains(players))
                .with_context(|| format!("{}: game.players must be a whole number from 1 to 32", file.display()))?;
        }
        Ok(identity)
    }

    /// A name that is a file name on every platform.
    pub(crate) fn check(&self) -> Result<()> {
        let name = &self.name;
        let reserved = ["CON", "PRN", "AUX", "NUL"].contains(&name.to_ascii_uppercase().as_str())
            || ["COM", "LPT"].iter().any(|device| {
                name.len() == 4
                    && name.get(..3).is_some_and(|start| start.eq_ignore_ascii_case(device))
                    && name.ends_with(|digit: char| digit.is_ascii_digit())
            });
        ensure!(
            !name.is_empty()
                && name.len() <= 64
                && !reserved
                && !name.starts_with('.')
                && !name.ends_with([' ', '.'])
                && name.chars().all(file_name_character),
            "`{name}` cannot name a directory and an executable on every platform: use letters, digits, \
             spaces, `-`, `_`, `+` and `.`"
        );
        Ok(())
    }
}

fn file_name_character(character: char) -> bool {
    character.is_alphanumeric() || " -_+.".contains(character)
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// What a bundle says of itself in its `game.json`: its game, and the
/// runtime its `game.cwasm` was precompiled for, if it has one.
#[derive(Debug, Deserialize, Serialize)]
pub struct Manifest {
    #[serde(flatten)]
    pub game: Identity,
    /// [`RUNTIME`] of the pen that precompiled `game.cwasm`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<String>,
    /// Whether this package also ships epoch-interrupted `workers.cwasm` + `workers.meta`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub worker_deadlines: bool,
    /// Optional startup files present in a static browser export.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web: Option<pen_runtime::web::WebFiles>,
}

impl Manifest {
    /// The manifest of the bundle in `directory`, if it has one.
    pub fn read(directory: &Path) -> Result<Option<Manifest>> {
        let path = directory.join("game.json");
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
        };
        let manifest: Manifest = serde_json::from_str(&text).with_context(|| format!("reading {}", path.display()))?;
        manifest.game.check().with_context(|| format!("{}: name", path.display()))?;
        Ok(Some(manifest))
    }

    #[cfg(feature = "compiler")]
    pub fn write(&self, directory: &Path) -> Result<()> {
        std::fs::write(directory.join("game.json"), serde_json::to_string_pretty(self)? + "\n")?;
        Ok(())
    }
}

/// The game of the bundle in `directory`: its manifest's, else named after
/// the directory, or, a package's `game/`, after the executable beside it,
/// which `pen build` names after the game.
pub fn bundle_identity(directory: &Path) -> Result<Identity> {
    if let Some(manifest) = Manifest::read(directory)? {
        return Ok(manifest.game);
    }
    let stem = |path: &Path| path.file_stem().map(|name| name.to_string_lossy().into_owned());
    let directory = std::fs::canonicalize(directory).unwrap_or_else(|_| directory.to_owned());
    let name = match stem(&directory) {
        Some(name) if name != "game" => Some(name),
        _ => std::env::current_exe().ok().and_then(|exe| stem(&exe)),
    };
    Ok(Identity::named(name.as_deref().unwrap_or("game")))
}

/// This runtime's compatibility mark (build.rs): a `game.cwasm` loads in it
/// only when a pen with the same mark precompiled it.
pub const RUNTIME: &str = env!("PEN_RUNTIME_COMPATIBILITY");

/// The mark as it stands in the executable, where `pen build` reads it from
/// the runtime it ships.
#[used]
static RUNTIME_MARK: &str = concat!("pen-runtime-compatibility=", env!("PEN_RUNTIME_COMPATIBILITY"), "\n");

/// Keeps the mark in the executable. `#[used]` keeps it in the object file,
/// but MSVC's linker still drops data nothing reads, so the runtime's start
/// (and `runtime_mark`, for test executables) reads it.
pub fn keep_mark() {
    std::hint::black_box(RUNTIME_MARK);
}

/// What a mark starts with, made here rather than written out: this
/// executable's bytes then hold it only in its own mark, and nowhere two of
/// its strings happen to lie side by side.
#[cfg(feature = "compiler")]
fn mark_prefix() -> Vec<u8> {
    b"=ytilibitapmoc-emitnur-nep".iter().rev().copied().collect()
}

/// Whether `text` is a mark as build.rs makes them.
#[cfg(feature = "compiler")]
fn well_formed(text: &str) -> bool {
    let Some((version, hash)) = text.strip_prefix("wasmtime ").and_then(|rest| rest.split_once(", engine ")) else {
        return false;
    };
    !version.is_empty()
        && version.chars().all(|character| character.is_ascii_alphanumeric() || ".-+".contains(character))
        && hash.len() == 16
        && hash.chars().all(|character| character.is_ascii_hexdigit())
}

/// The compatibility mark of a runtime executable, if it has one.
#[cfg(feature = "compiler")]
fn runtime_mark(executable: &[u8]) -> Result<Option<String>> {
    keep_mark();
    let prefix = mark_prefix();
    let mut marks = BTreeSet::new();
    for (at, _) in executable.iter().enumerate().filter(|(_, byte)| **byte == prefix[0]) {
        if !executable[at..].starts_with(&prefix) {
            continue;
        }
        let rest = &executable[at + prefix.len()..];
        let end = rest.iter().take(128).position(|byte| *byte == b'\n');
        let mark = end.and_then(|end| std::str::from_utf8(&rest[..end]).ok()).filter(|mark| well_formed(mark));
        if let Some(mark) = mark {
            marks.insert(mark.to_owned());
        }
    }
    ensure!(marks.len() <= 1, "the runtime has more than one compatibility mark: {marks:?}");
    Ok(marks.pop_first())
}

/// Optional `web` block of `pen.json`:
/// `{"web": {"mode": "2d" | "3d", "features": ["render-2d", "gamepad"]}}`.
#[cfg(feature = "compiler")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WebConfig {
    /// Overrides 2D vs 3D auto-detection (`"2d"` or `"3d"`).
    pub mode: Option<String>,
    /// Explicit `pen-web` Cargo feature list (overridden by CLI `--web-features`).
    pub features: Option<Vec<String>>,
}

#[cfg(feature = "compiler")]
impl WebConfig {
    fn of_checkout(root: &Path) -> Result<Self> {
        let file = root.join("pen.json");
        let reading = || format!("reading {}", file.display());
        let text = std::fs::read_to_string(&file).with_context(reading)?;
        let config: serde_json::Value = serde_json::from_str(&text).with_context(reading)?;
        let Some(web) = config.get("web") else {
            return Ok(Self::default());
        };
        let Some(obj) = web.as_object() else {
            bail!("{}: web must be an object", file.display());
        };
        let mode = match obj.get("mode") {
            None => None,
            Some(serde_json::Value::String(mode)) if matches!(mode.as_str(), "2d" | "3d") => Some(mode.clone()),
            Some(_) => bail!("{}: web.mode must be \"2d\" or \"3d\"", file.display()),
        };
        let features = match obj.get("features") {
            None => None,
            Some(serde_json::Value::Array(items)) => {
                let mut list = Vec::with_capacity(items.len());
                for item in items {
                    let Some(text) = item.as_str() else {
                        bail!("{}: web.features entries must be strings", file.display());
                    };
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        list.push(trimmed.to_owned());
                    }
                }
                Some(list)
            }
            Some(_) => bail!("{}: web.features must be an array of strings", file.display()),
        };
        Ok(Self { mode, features })
    }
}

#[cfg(feature = "compiler")]
fn checkout_compiler_entries(root: &Path) -> Result<(PathBuf, Option<PathBuf>)> {
    let file = root.join("pen.json");
    let reading = || format!("reading {}", file.display());
    let text = std::fs::read_to_string(&file).with_context(reading)?;
    let config: serde_json::Value = serde_json::from_str(&text).with_context(reading)?;
    let compiler = config.get("compiler");
    let entry = match compiler.and_then(|c| c.get("entry")) {
        None | Some(serde_json::Value::Null) => PathBuf::from("src/main.blot"),
        Some(serde_json::Value::String(path)) if !path.trim().is_empty() => PathBuf::from(path),
        Some(_) => bail!("{}: compiler.entry must be a non-empty path string", file.display()),
    };
    let devtools = match compiler.and_then(|c| c.get("devtools")) {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(path)) if !path.trim().is_empty() => Some(PathBuf::from(path)),
        Some(_) => bail!("{}: compiler.devtools must be a non-empty path string", file.display()),
    };
    Ok((entry, devtools))
}

#[cfg(feature = "compiler")]
fn has_game_entry(directory: &Path) -> bool {
    let pen_json = directory.join("pen.json");
    if !pen_json.is_file() {
        return false;
    }
    if let Ok(text) = std::fs::read_to_string(&pen_json)
        && let Ok(config) = serde_json::from_str::<serde_json::Value>(&text)
        && let Some(entry) = config.get("compiler").and_then(|c| c.get("entry")).and_then(serde_json::Value::as_str)
    {
        return directory.join(entry).exists();
    }
    directory.join("src/main.blot").is_file()
}

#[cfg(feature = "compiler")]
pub struct Project {
    pub root: PathBuf,
    pub identity: Identity,
    pub web: WebConfig,
    /// Relative path to the game entry (from `pen.json` `compiler.entry`, else `src/main.blot`).
    pub entry: PathBuf,
    /// Optional relative path to the dev tools entry (from `pen.json` `compiler.devtools`).
    pub devtools_entry: Option<PathBuf>,
    /// Its Wasm plugins, loaded when first asked for.
    #[cfg(feature = "compiler")]
    plugins: std::sync::OnceLock<Option<std::sync::Arc<pen_plugins::Plugins>>>,
}

#[cfg(feature = "compiler")]
impl Project {
    /// The checkout at `root`, or the one the current directory is in.
    pub fn find(root: Option<PathBuf>) -> Result<Project> {
        Self::find_in(root, true)
    }

    /// A test checkout needs only pen.json; its plugin owns the test layout.
    pub fn find_tests(root: Option<PathBuf>) -> Result<Project> {
        Self::find_in(root, false)
    }

    fn find_in(root: Option<PathBuf>, require_game: bool) -> Result<Project> {
        let start = match root {
            Some(root) => root,
            None => std::env::current_dir()?,
        };
        let start = std::fs::canonicalize(&start).with_context(|| format!("{} does not exist", start.display()))?;
        let root = start
            .ancestors()
            .find(|directory| directory.join("pen.json").is_file() && (!require_game || has_game_entry(directory)));
        match root {
            Some(root) => {
                let (entry, devtools_entry) = checkout_compiler_entries(root)?;
                Ok(Project {
                    identity: Identity::of_checkout(root)?,
                    web: WebConfig::of_checkout(root)?,
                    entry,
                    devtools_entry,
                    root: root.to_owned(),
                    #[cfg(feature = "compiler")]
                    plugins: std::sync::OnceLock::new(),
                })
            }
            None => bail!(
                "{} is not inside a checkout (a directory with {}); pass --project DIR",
                start.display(),
                if require_game { "pen.json and its configured compiler.entry (or src/main.blot)" } else { "pen.json" }
            ),
        }
    }

    /// The checkout `directory` is in, if any.
    pub fn around(directory: &Path) -> Option<PathBuf> {
        directory.ancestors().find(|directory| has_game_entry(directory)).map(Path::to_owned)
    }

    pub fn entry_path(&self) -> PathBuf {
        self.root.join(&self.entry)
    }

    pub fn assets(&self) -> PathBuf {
        self.root.join("src/assets")
    }

    pub fn scene(&self) -> PathBuf {
        self.root.join("src/main.scene")
    }

    /// Ships the game's graphics presets (`crate::presets`), its
    /// `src/presets.json`, into the bundle `game`, if it has them.
    fn copy_presets(&self, game: &Path) -> Result<()> {
        let presets = self.root.join("src/presets.json");
        if presets.is_file() {
            std::fs::copy(&presets, game.join("presets.json")).context("copying src/presets.json")?;
        }
        Ok(())
    }

    /// Where `dev` and `run` keep the compiled program.
    pub fn native(&self) -> PathBuf {
        self.root.join(".pen/runtime")
    }

    /// The dev tools' program (`compiler.devtools` in `pen.json`, else
    /// `devtools/main.blot` or `src/devtools/main.blot`), relative to the
    /// checkout, if it has one.
    pub fn devtools(&self) -> Option<PathBuf> {
        if let Some(entry) = &self.devtools_entry {
            return self.root.join(entry).exists().then(|| entry.clone());
        }
        ["devtools/main.blot", "src/devtools/main.blot"]
            .into_iter()
            .find(|entry| self.root.join(entry).is_file())
            .map(PathBuf::from)
    }

    /// The checkout's plugins, when its `pen.json` chooses a compiler: they
    /// build its programs.
    #[cfg(feature = "compiler")]
    fn plugins(&self) -> Result<Option<std::sync::Arc<pen_plugins::Plugins>>> {
        if let Some(plugins) = self.plugins.get() {
            return Ok(plugins.clone());
        }
        let loaded = crate::plugin_build::load(&self.root)?;
        Ok(self.plugins.get_or_init(|| loaded).clone())
    }

    /// What a plugin build never watches: what pen writes.
    #[cfg(feature = "compiler")]
    fn written(&self) -> Vec<PathBuf> {
        vec![self.root.join(".pen"), self.root.join("target")]
    }

    /// Builds the checkout's optional dev tools, and again as their sources
    /// change.
    pub fn build_devtools(&self, out: &Path) -> Result<Option<Watcher>> {
        let Some(entry) = self.devtools() else { return Ok(None) };
        std::fs::create_dir_all(out)?;
        #[cfg(feature = "compiler")]
        {
            let plugins = self.plugins()?.context("pen.json must choose a compiler plugin")?;
            let built = crate::plugin_build::watch(
                plugins,
                "dev tools",
                &self.root.join(entry),
                &out.join("devtools.wasm"),
                self.written(),
            )?;
            Ok(Some(Watcher { plugin: Some(built) }))
        }
        #[cfg(not(feature = "compiler"))]
        {
            let _ = entry;
            bail!("this player runtime cannot build a checkout; use a full pen build")
        }
    }

    /// Compiles the game's Wasm once through the checkout's compiler plugin.
    pub fn compile(&self, out: &Path) -> Result<()> {
        std::fs::create_dir_all(out)?;
        #[cfg(feature = "compiler")]
        {
            let plugins = self.plugins()?.context("pen.json must choose a compiler plugin")?;
            crate::plugin_build::compile(plugins, "game", &self.entry_path(), &out.join("game.wasm"))
        }
        #[cfg(not(feature = "compiler"))]
        bail!("this player runtime cannot build a checkout; use a full pen build")
    }

    /// Compiles the optional dev tools once, for a browser export.
    pub fn compile_devtools(&self, out: &Path) -> Result<()> {
        let Some(entry) = self.devtools() else { return Ok(()) };
        #[cfg(feature = "compiler")]
        {
            let plugins = self.plugins()?.context("pen.json must choose a compiler plugin")?;
            crate::plugin_build::compile(plugins, "dev tools", &self.root.join(entry), &out.join("devtools.wasm"))
        }
        #[cfg(not(feature = "compiler"))]
        {
            let _ = (entry, out);
            bail!("this player runtime cannot build a checkout; use a full pen build")
        }
    }

    /// Watches the program and waits for its first successful build.
    pub fn watch(&self, out: &Path) -> Result<Watcher> {
        std::fs::create_dir_all(out)?;
        #[cfg(feature = "compiler")]
        {
            let plugins = self.plugins()?.context("pen.json must choose a compiler plugin")?;
            let (watching, built) = crate::plugin_build::watch(
                plugins,
                "game",
                &self.entry_path(),
                &out.join("game.wasm"),
                self.written(),
            )?;
            loop {
                match built.recv() {
                    Ok(true) => return Ok(Watcher { plugin: Some((watching, built)) }),
                    Ok(false) => overlay_waiting(),
                    Err(_) => bail!("the game's plugin build stopped"),
                }
            }
        }
        #[cfg(not(feature = "compiler"))]
        bail!("this player runtime cannot build a checkout; use a full pen build")
    }
}

/// On-disk temporary storage; shared compiler caches are not swept by age.
#[cfg(test)]
pub(crate) fn temporary_dir() -> Result<PathBuf> {
    let directory = dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("pen/tmp");
    std::fs::create_dir_all(&directory)?;
    Ok(directory)
}

/// Says once, while waiting for a game that does not compile yet, that its
/// next edit is awaited.
#[cfg(feature = "compiler")]
#[cfg(feature = "compiler")]
fn overlay_waiting() {
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| crate::devlog::info(crate::devlog::Source::Build, "waiting for an edit that compiles"));
}

/// Plugin build workers; dropping them joins their explicit shutdown.
#[cfg(feature = "compiler")]
pub struct Watcher {
    #[cfg(feature = "compiler")]
    plugin: Option<(pen_plugins::Watching, std::sync::mpsc::Receiver<bool>)>,
}

#[cfg(feature = "compiler")]
impl Watcher {
    pub fn wait(&mut self) -> Result<bool> {
        #[cfg(feature = "compiler")]
        if let Some((_, built)) = &self.plugin {
            return Ok(built.recv().unwrap_or(false));
        }
        Ok(true)
    }
}

/// Every file under `directory`, as asset keys: relative, `/` between
/// directories.
#[cfg(feature = "compiler")]
pub(crate) fn asset_keys(directory: &Path) -> Result<Vec<String>> {
    fn walk(directory: &Path, prefix: &str, keys: &mut Vec<String>) -> Result<()> {
        for entry in std::fs::read_dir(directory).with_context(|| format!("reading {}", directory.display()))? {
            let entry = entry?;
            let key = format!("{prefix}{}", entry.file_name().to_string_lossy());
            if entry.file_type()?.is_dir() {
                walk(&entry.path(), &format!("{key}/"), keys)?;
            } else {
                keys.push(key);
            }
        }
        Ok(())
    }
    let mut keys = Vec::new();
    walk(directory, "", &mut keys)?;
    keys.sort();
    Ok(keys)
}

/// Kinds of asset the renderer and audio load only when the catalog, or a
/// material it names, names them.
#[cfg(feature = "compiler")]
const NAMED_KINDS: [&str; 9] = [".json", ".wgsl", ".png", ".jpg", ".jpeg", ".gtex", ".wav", ".ogg", ".sound"];

/// Every string in a JSON value.
#[cfg(feature = "compiler")]
fn strings<'a>(value: &'a serde_json::Value, each: &mut impl FnMut(&'a str)) {
    match value {
        serde_json::Value::String(text) => each(text),
        serde_json::Value::Array(values) => values.iter().for_each(|value| strings(value, each)),
        serde_json::Value::Object(fields) => fields.values().for_each(|value| strings(value, each)),
        _ => {}
    }
}

/// Of `keys` (in `assets`), those a game uses: `catalog.json`, what it
/// names, and what those name in turn (a material's shader and textures),
/// and its sounds' captions (`captions.json`).
/// To be safe it keeps any file of a kind it does not know, and any `game`'s
/// Wasm names; without a catalog, all of them.
#[cfg(feature = "compiler")]
pub(crate) fn used_assets(assets: &Path, keys: &[String], game: &[u8]) -> BTreeSet<String> {
    let known: HashSet<&str> = keys.iter().map(String::as_str).collect();
    if !known.contains("catalog.json") {
        return keys.iter().cloned().collect();
    }
    let mut used = BTreeSet::new();
    let mut pending = vec!["catalog.json".to_owned()];
    if known.contains("captions.json") {
        pending.push("captions.json".to_owned());
    }
    while let Some(key) = pending.pop() {
        // Meshes are numbers only, and the bulk of the assets.
        if !used.insert(key.clone()) || !key.ends_with(".json") || key.ends_with(".mesh.json") {
            continue;
        }
        let parsed = std::fs::read(assets.join(&key)).ok().and_then(|bytes| serde_json::from_slice(&bytes).ok());
        if let Some(value) = parsed {
            strings(&value, &mut |text| {
                if known.contains(text) {
                    pending.push(text.to_owned());
                }
            });
        }
    }
    // A substring search per key: the program is megabytes, the keys hundreds.
    let named = |key: &str| memchr::memmem::find(game, key.as_bytes()).is_some();
    for key in keys {
        if !NAMED_KINDS.iter().any(|kind| key.ends_with(kind)) || named(key) {
            used.insert(key.clone());
        }
    }
    // The optional state adapter declaration, which the host reads beside the
    // catalog to migrate old saves.
    if known.contains(pen_runtime::state::CONFIG_FILE) {
        used.insert(pen_runtime::state::CONFIG_FILE.to_owned());
    }
    used
}

/// What [`ship_assets`] shipped, for a package's summary.
#[cfg(feature = "compiler")]
struct Shipped {
    files: usize,
    of: usize,
    bytes: u64,
    of_bytes: u64,
    meshes: usize,
}

#[cfg(feature = "compiler")]
impl std::fmt::Display for Shipped {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            formatter,
            "assets {:.1} MB: {} of {} files ({:.1} MB as they are), {} meshes in binary",
            self.bytes as f64 / 1e6,
            self.files,
            self.of,
            self.of_bytes as f64 / 1e6,
            self.meshes,
        )
    }
}

/// Copies the assets the game in `wasm` uses from `from` to `to`, its mesh
/// JSON in the renderer's binary form ([`pen_render::pack_mesh_with_lods`]),
/// which loads faster, is a fraction of the size, and embeds simpler levels of
/// detail plus pre-baked GPU meshlets and Cluster DAGs. Packs meshes in parallel
/// across available CPU cores.
#[cfg(feature = "compiler")]
fn ship_assets(from: &Path, to: &Path, wasm: &[u8]) -> Result<Shipped> {
    let keys = asset_keys(from)?;
    let used = used_assets(from, &keys, wasm);
    let mut shipped = Shipped { files: used.len(), of: keys.len(), bytes: 0, of_bytes: 0, meshes: 0 };
    let mut used_keys: Vec<(&String, u64)> = Vec::with_capacity(used.len());
    for key in &keys {
        let source = from.join(key);
        let len = std::fs::metadata(&source)?.len();
        shipped.of_bytes += len;
        if used.contains(key) {
            used_keys.push((key, len));
        }
    }
    // Schedule largest assets first so multi-core mesh packing avoids straggler tails.
    used_keys.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    for &(key, _) in &used_keys {
        let target = to.join(key);
        std::fs::create_dir_all(target.parent().context("an asset has no directory")?)?;
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let workers =
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get).min(used_keys.len().max(1));
    let mut results: Vec<Result<(u64, usize)>> = Vec::with_capacity(workers);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            handles.push(scope.spawn(|| -> Result<(u64, usize)> {
                let mut bytes = 0u64;
                let mut meshes = 0usize;
                #[cfg(feature = "compiler")]
                let pack = pen_render::pack_mesh_with_lods;
                #[cfg(not(feature = "compiler"))]
                let pack = pen_render::pack_mesh;
                loop {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(key, _)) = used_keys.get(index) else { break };
                    let source = from.join(key);
                    let target = to.join(key);
                    let packed = if key.ends_with(".mesh.json") {
                        let mesh = std::fs::read(&source)?;
                        match pack(&mesh) {
                            // A package's game/, packaged again, has them packed already.
                            Err(_) if mesh.starts_with(b"PMSH") => None,
                            Ok(packed) => Some(packed),
                            Err(error) => {
                                eprintln!("shipping {key} as it is: {error:#}");
                                None
                            }
                        }
                    } else {
                        None
                    };
                    // A new file, never one sharing the source's: tools rewrite assets in place.
                    bytes += match packed {
                        Some(packed) => {
                            std::fs::write(&target, &packed)?;
                            meshes += 1;
                            packed.len() as u64
                        }
                        None => {
                            std::fs::copy(&source, &target).with_context(|| format!("copying {}", source.display()))?
                        }
                    };
                }
                Ok((bytes, meshes))
            }));
        }
        for handle in handles {
            results.push(handle.join().expect("asset shipping worker panicked"));
        }
    });
    for result in results {
        let (bytes, meshes) = result?;
        shipped.bytes += bytes;
        shipped.meshes += meshes;
    }
    Ok(shipped)
}

/// Shared preparation for native packages and static browser exports.
/// Platform-specific metadata, native code and hosting files follow this.
#[cfg(feature = "compiler")]
struct PreparedGame {
    wasm: Vec<u8>,
    assets: Shipped,
}

#[cfg(feature = "compiler")]
fn prepare_game(source: &GameSource<'_>, game: &Path, optimize: bool) -> Result<PreparedGame> {
    source.compile(game)?;
    let wasm = shipping_wasm(&game.join("game.wasm"), optimize)?;
    let from = source.assets();
    let assets = ship_assets(&from, &game.join("assets"), &wasm)?;
    let scene = source.scene();
    if scene.is_file() {
        std::fs::copy(&scene, game.join("main.scene"))?;
    }
    match crate::scene_assets::write(&wasm, &from, &scene, &game.join("assets")) {
        Ok(()) => {}
        // A bundle built elsewhere need not implement this host's scene
        // analysis. Its first frame then loads assets as it goes.
        Err(error) if matches!(source, GameSource::Bundle(_)) => {
            eprintln!("not working out what the bundle draws first: {error:#}");
        }
        Err(error) => return Err(error),
    }
    Ok(PreparedGame { wasm, assets })
}

/// What `pen build` ships.
#[cfg(feature = "compiler")]
pub struct PackageOptions {
    /// Optimize shipping Wasm before native precompilation.
    pub optimize: bool,
    /// The runtime to ship. By default the player runtime (no compiler)
    /// that `just install` installs beside this executable, in `player/`.
    pub runtime: Option<PathBuf>,
    /// The Rust target the runtime was built for, when its executable header
    /// does not say. cargo-zigbuild's glibc suffix (`.2.28`) may be left on.
    pub target: Option<String>,
    /// The CPU game.cwasm is compiled for: a Cranelift preset, or `native`
    /// (see blot_host::portable_engine; x86-64-v2 by default).
    pub cpu: Option<String>,
    /// A macOS application bundle's identifier (`app_bundle`).
    pub bundle_id: Option<String>,
    /// The store package versions are kept in, for patches between them
    /// (patch.rs); by default the checkout's `.pen/store`.
    pub store: Option<PathBuf>,
    /// Also precompile epoch-interrupted `workers.cwasm` + `workers.meta` for
    /// deadline-bounded shipping worker pools.
    pub worker_deadlines: bool,
}

/// Where `pen build` takes the game from.
#[cfg(feature = "compiler")]
pub enum GameSource<'a> {
    /// A checkout, whose program compiles now.
    Checkout(&'a Project),
    /// A bundle compiled already (`pen build --game DIR`): `pen export`'s, a
    /// package's `game/`, or a game built in another language: its
    /// `game.wasm`, `assets/`, optional `main.scene` and `game.json`, as
    /// `pen --game` plays it. What a machine without the checkout or its
    /// compiler plugin packages for the runtime it has, as CI does for
    /// Windows and macOS.
    Bundle(&'a Path),
}

#[cfg(feature = "compiler")]
impl GameSource<'_> {
    fn identity(&self) -> Result<Identity> {
        match self {
            GameSource::Checkout(project) => Ok(project.identity.clone()),
            GameSource::Bundle(directory) => bundle_identity(directory),
        }
    }

    /// Writes game.wasm into `game`.
    fn compile(&self, game: &Path) -> Result<()> {
        let wasm = game.join("game.wasm");
        match self {
            GameSource::Checkout(project) => project.compile(game)?,
            GameSource::Bundle(directory) => {
                let from = directory.join("game.wasm");
                std::fs::copy(&from, &wasm).with_context(|| format!("copying {}", from.display()))?;
            }
        }
        Ok(())
    }

    fn assets(&self) -> PathBuf {
        match self {
            GameSource::Checkout(project) => project.assets(),
            GameSource::Bundle(directory) => directory.join("assets"),
        }
    }

    /// The starting scene's file, which a game without scenes leaves out: it
    /// starts from what its program creates.
    fn scene(&self) -> PathBuf {
        match self {
            GameSource::Checkout(project) => project.scene(),
            GameSource::Bundle(directory) => directory.join("main.scene"),
        }
    }

    /// The checkout, or the bundle's directory: where a package stages and
    /// its versions are kept by default (`.pen/store`).
    fn root(&self) -> &Path {
        match self {
            GameSource::Checkout(project) => &project.root,
            GameSource::Bundle(directory) => directory,
        }
    }
}

/// The runtime `pen build` ships by default.
#[cfg(feature = "compiler")]
fn default_runtime() -> Result<PathBuf> {
    installed_player(&std::env::current_exe()?)
}

#[cfg(feature = "compiler")]
fn installed_player(exe: &Path) -> Result<PathBuf> {
    let player = exe.with_file_name("player").join(exe.file_name().context("the executable has no name")?);
    if player.is_file() {
        return Ok(player);
    }
    bail!(
        "the player runtime is missing at {}: run `just install-player` in Pen's checkout, \
         or pass --runtime EXE to choose the runtime to ship",
        player.display()
    )
}

/// The Rust target an executable was built for, from its header.
#[cfg(feature = "compiler")]
pub(crate) fn executable_target(header: &[u8]) -> Option<&'static str> {
    let u16_at = |at: usize| Some(u16::from_le_bytes(header.get(at..at + 2)?.try_into().ok()?));
    let u32_at = |at: usize| Some(u32::from_le_bytes(header.get(at..at + 4)?.try_into().ok()?));
    if header.starts_with(b"\x7fELF") {
        return match u16_at(18)? {
            0x3e => Some("x86_64-unknown-linux-gnu"),
            0xb7 => Some("aarch64-unknown-linux-gnu"),
            _ => None,
        };
    }
    if header.starts_with(b"MZ") {
        let pe = u32_at(0x3c)? as usize;
        return match (header.get(pe..pe + 4)? == b"PE\0\0").then(|| u16_at(pe + 4)).flatten()? {
            0x8664 => Some("x86_64-pc-windows-gnu"),
            0xaa64 => Some("aarch64-pc-windows-msvc"),
            _ => None,
        };
    }
    if u32_at(0)? == 0xfeed_facf {
        return match u32_at(4)? {
            0x0100_0007 => Some("x86_64-apple-darwin"),
            0x0100_000c => Some("aarch64-apple-darwin"),
            _ => None,
        };
    }
    None
}

/// The Steam library `runtime` (for `target`) links, if it was built with
/// Steam: beside it, or where cargo's steamworks-sys build put it.
#[cfg(feature = "compiler")]
fn steam_library(runtime: &Path, bytes: &[u8], target: &str) -> Result<Option<PathBuf>> {
    // A function every runtime built with Steam imports, put together here
    // so that this executable's own bytes do not hold the name.
    let import = ["SteamAPI", "_InitFlat"].concat();
    if !bytes.windows(import.len()).any(|window| window == import.as_bytes()) {
        return Ok(None);
    }
    let name = if target.contains("windows") {
        "steam_api64.dll"
    } else if target.contains("apple") {
        "libsteam_api.dylib"
    } else {
        "libsteam_api.so"
    };
    let directory = runtime.parent().unwrap_or(Path::new("."));
    let beside = directory.join(name);
    if beside.is_file() {
        return Ok(Some(beside));
    }
    let built = std::fs::read_dir(directory.join("build"))
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("steamworks-sys-"))
        .map(|entry| entry.path().join("out").join(name))
        .filter(|path| path.is_file())
        .max_by_key(|path| std::fs::metadata(path).and_then(|metadata| metadata.modified()).ok());
    built.map(Some).with_context(|| format!("{} needs {name} beside it, and it is not there", runtime.display()))
}

/// Refuses to ship a runtime that would refuse the game.cwasm this pen
/// precompiles.
#[cfg(feature = "compiler")]
fn check_runtime(runtime: &Path, bytes: &[u8]) -> Result<()> {
    match runtime_mark(bytes)? {
        Some(mark) if mark == RUNTIME => Ok(()),
        Some(mark) => bail!(
            "{} loads games precompiled by a pen of {mark}, and this pen is {RUNTIME}: it would refuse the \
             game. Rebuild the player: just install-player (or pass --runtime)",
            runtime.display()
        ),
        None => bail!(
            "{} has no runtime compatibility mark: it is older than this pen, or not pen. Rebuild the \
             player: just install-player (or pass --runtime)",
            runtime.display()
        ),
    }
}

#[cfg(feature = "compiler")]
fn notices(out: &Path) -> Result<()> {
    std::fs::write(out.join("LICENSE"), include_str!("../../../LICENSE"))?;
    std::fs::write(out.join("THIRD_PARTY_NOTICES.md"), include_str!("../../../THIRD_PARTY_NOTICES.md"))?;
    Ok(())
}

#[cfg(feature = "compiler")]
pub(crate) fn size_of(path: &Path) -> u64 {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => {
            std::fs::read_dir(path).into_iter().flatten().flatten().map(|entry| size_of(&entry.path())).sum()
        }
        Ok(metadata) => metadata.len(),
        Err(_) => 0,
    }
}

/// Identity of the original Wasm, including native packages which omit it.
pub(crate) fn game_fingerprint(directory: &Path) -> Result<u64> {
    if directory.join("game.meta").is_file() {
        blot_host::native_fingerprint(&std::fs::read(directory.join("game.meta"))?)
    } else {
        #[cfg(feature = "compiler")]
        return Ok(pen_runtime::net::wire::hash_bytes(&std::fs::read(directory.join("game.wasm"))?));
        #[cfg(not(feature = "compiler"))]
        bail!("game.meta is missing; package the game again with `pen build`")
    }
}

/// A shippable folder: the runtime, the libraries it needs beside it, and
/// the game: its compiled program, precompiled for the runtime's platform
/// and a CPU baseline players' machines have, the assets it uses, its
/// starting scene and its manifest.
#[cfg(feature = "compiler")]
pub fn package(source: GameSource, out: &Path, options: PackageOptions) -> Result<()> {
    let staging = crate::staging::Staging::new(source.root(), out)?;
    package_into(source, &staging.directory, out, options)?;
    crate::reproducible::settle_times(&staging.directory)?;
    staging.publish()
}

#[cfg(feature = "compiler")]
fn package_into(source: GameSource, out: &Path, destination: &Path, options: PackageOptions) -> Result<()> {
    let runtime = match options.runtime {
        Some(runtime) => runtime,
        None => default_runtime()?,
    };
    let runtime_bytes =
        std::fs::read(&runtime).with_context(|| format!("reading the runtime {}", runtime.display()))?;
    check_runtime(&runtime, &runtime_bytes)?;
    let target = match options.target {
        Some(target) => target.split('.').next().unwrap_or_default().to_owned(),
        None => executable_target(&runtime_bytes)
            .with_context(|| format!("cannot tell what {} runs on: pass --target TRIPLE", runtime.display()))?
            .to_owned(),
    };
    let steam = steam_library(&runtime, &runtime_bytes, &target)?;
    let engine = blot_host::portable_engine(Some(&target), options.cpu.as_deref())?;

    let identity = source.identity()?;
    let name = identity.name.clone();
    // Named after the game, whatever the runtime is called.
    let layout = crate::app_bundle::Layout::new(out, &identity, &target, options.bundle_id.as_deref())?;
    layout.create()?;
    let game = &layout.game;
    let PreparedGame { wasm, assets } = prepare_game(&source, game, options.optimize)?;
    crate::services::locale::ship_tables(&source.assets(), game)?;
    match &source {
        GameSource::Checkout(project) => {
            project.copy_presets(game)?;
            crate::reports::ship_config(&project.root, game)?;
        }
        // A bundle carries both already, as `pen export` writes it.
        GameSource::Bundle(directory) => {
            for file in ["presets.json", "reports.json"] {
                let from = directory.join(file);
                if from.is_file() {
                    std::fs::copy(&from, game.join(file)).with_context(|| format!("copying {}", from.display()))?;
                }
            }
        }
    }
    let precompiled = blot_host::precompile(&engine, &wasm)?;
    std::fs::write(game.join("game.cwasm"), crate::compiled::encode(&precompiled)?)?;
    let meta = blot_host::native_metadata(&wasm)?;
    std::fs::write(game.join("game.meta"), &meta)?;
    if options.worker_deadlines {
        let worker_engine = blot_host::bounded_portable_engine(
            Some(&target),
            options.cpu.as_deref(),
            std::time::Duration::from_millis(2),
        )?;
        let worker_precompiled = blot_host::precompile(&worker_engine, &wasm)?;
        std::fs::write(game.join("workers.cwasm"), crate::compiled::encode(&worker_precompiled)?)?;
        std::fs::write(game.join("workers.meta"), &meta)?;
    }
    std::fs::remove_file(game.join("game.wasm"))?;
    Manifest {
        game: identity,
        runtime: Some(RUNTIME.to_owned()),
        worker_deadlines: options.worker_deadlines,
        web: None,
    }
    .write(game)?;
    std::fs::copy(&runtime, &layout.executable)?;
    if let Some(library) = &steam {
        std::fs::copy(library, layout.libraries.join(library.file_name().unwrap()))?;
    }
    notices(&layout.notices)?;
    let store = options.store.unwrap_or_else(|| source.root().join(".pen/store"));
    let version = crate::patch::record(out, &name, &store)?;
    let megabytes = |path: &Path| size_of(path) as f64 / 1e6;
    let x86_64 = target.starts_with("x86_64");
    let cpu = options.cpu.as_deref().unwrap_or(if x86_64 { blot_host::PORTABLE_X86_64 } else { "baseline" });
    eprintln!(
        "packaged {} for {target} ({} CPU): runtime {:.1} MB{}, game.cwasm {:.1} MB, {assets}; {:.1} MB in all",
        destination.display(),
        cpu,
        runtime_bytes.len() as f64 / 1e6,
        steam.as_ref().map(|library| format!(" + Steam {:.1} MB", megabytes(library))).unwrap_or_default(),
        megabytes(&game.join("game.cwasm")),
        megabytes(out),
    );
    eprintln!("version {version}, kept in {} (pen patch makes patches between versions)", store.display());
    Ok(())
}

/// What optional host JS files and Cargo features a static web export actually needs.
#[cfg(feature = "compiler")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebHostPlan {
    /// Whether the game uses retired `blot:tasks` structured concurrency (`structured.js`).
    pub structured_js: bool,
    /// Whether the game declares any compute kernels in `blot:abi` (`kernels.js` + `kernel_worker.js`).
    pub kernels_js: bool,
    /// Exact Cargo features to pass to `build.sh --features` when tailoring a static host.
    pub features: Vec<&'static str>,
}

#[cfg(feature = "compiler")]
fn wasm_contains_ext_tag(wasm: &[u8], tag: u32) -> bool {
    let f32_le = (tag as f32).to_le_bytes();
    if memchr::memmem::find(wasm, &f32_le).is_some() {
        return true;
    }
    let mut leb = [0u8; 5];
    let mut len = 0;
    let mut value = tag;
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            leb[len] = byte;
            len += 1;
            break;
        }
        leb[len] = byte | 0x80;
        len += 1;
    }
    memchr::memmem::find(wasm, &leb[..len]).is_some()
}

#[cfg(feature = "compiler")]
const WEB_2D_FEATURES: &[&str] = &["render-2d", "gamepad"];

#[cfg(feature = "compiler")]
const WEB_3D_FEATURES: &[&str] = &[
    "render-3d",
    "deferred-shaders",
    "audio-advanced",
    "gamepad",
    "render-oit",
    "render-atmosphere",
    "render-pbr-decals",
    "render-post-fx",
];

#[cfg(feature = "compiler")]
const WEB_COMPACT_FEATURES: &[&str] = &[
    "render-3d",
    "deferred-shaders",
    "audio-advanced",
    "gamepad",
    "kernels",
    "locales",
    "saved-adapter",
    "sound-recipes",
    "render-2d",
    "render-env",
    "render-particles",
    "render-lines",
    "render-jfa-outline",
    "render-meshlets",
    "render-oit",
    "render-atmosphere",
    "render-pbr-decals",
    "render-post-fx",
    "render-volume",
    "render-ssgi",
    "render-subsurface",
    "render-taa",
    "render-dof",
    "render-area-lights",
    "render-decals",
    "render-particles-cpu",
    "render-gpu-culling",
];

#[cfg(feature = "compiler")]
impl WebHostPlan {
    /// Inspects the exported `game.wasm`, `WebFiles` manifest, shipped `assets/` directory,
    /// and optional `pen.json` web config so static exports never ship unused JS glue or
    /// unused host subsystems.
    pub fn detect(
        wasm: &[u8],
        web: &pen_runtime::web::WebFiles,
        assets_dir: &Path,
        config: &WebConfig,
    ) -> Result<Self> {
        let abi = pen_runtime::abi::read_abi(wasm).ok();
        let structured_js = abi.as_ref().is_some_and(|a| a.tasks.is_some());
        let kernels_js = abi.as_ref().is_some_and(|a| !a.kernels.is_empty());

        let mut sound_recipes = false;
        let mut shader_graph = false;
        let mut texture_recipes = false;
        let mut mesh_json = false;
        let mut has_meshes = false;
        let mut has_materials = false;
        let mut render_meshlets = false;
        let mut video = false;

        if assets_dir.is_dir() {
            for key in asset_keys(assets_dir)? {
                let path = assets_dir.join(&key);
                let mut head = [0u8; 6];
                let head_len = std::fs::File::open(&path)
                    .and_then(|mut file| {
                        use std::io::Read;
                        file.read(&mut head)
                    })
                    .unwrap_or(0);
                if head_len >= 5 && &head[..4] == b"GTEX" && head[4] == 3 {
                    texture_recipes = true;
                }
                if head_len >= 6 && &head[..4] == b"PMSH" {
                    has_meshes = true;
                    // Byte 5 carries the mesh flags in both PMSH v1 and v2; bit 7 (128) is MESH_MESHLETS.
                    if (head[5] & 128) != 0 {
                        render_meshlets = true;
                    }
                } else if (head_len >= 4 && &head[..4] == b"PMRP") || key.ends_with(".mesh.json") {
                    has_meshes = true;
                    mesh_json = true;
                }
                if key.ends_with(".sound") {
                    sound_recipes = true;
                } else if key.ends_with(".matgraph.json") {
                    has_materials = true;
                    shader_graph = true;
                } else if key.ends_with(".material.json") {
                    has_materials = true;
                } else if key.ends_with(".y4m")
                    || key.ends_with(".ivf")
                    || key.ends_with(".webm")
                    || key.ends_with(".mp4")
                {
                    video = true;
                }
            }
        }

        let render_2d = wasm_contains_ext_tag(wasm, 65_008) || wasm_contains_ext_tag(wasm, 65_009);
        let render_env = wasm_contains_ext_tag(wasm, 65_001)
            || wasm_contains_ext_tag(wasm, 65_002)
            || wasm_contains_ext_tag(wasm, 65_003);
        let render_particles = wasm_contains_ext_tag(wasm, 65_005);
        let render_lines = wasm_contains_ext_tag(wasm, 65_006);
        let render_jfa_outline = wasm_contains_ext_tag(wasm, 65_007);

        let needs_3d = match config.mode.as_deref() {
            Some("2d") => false,
            Some("3d") => true,
            _ => {
                has_meshes
                    || has_materials
                    || web.scene
                    || video
                    || render_env
                    || render_particles
                    || render_lines
                    || render_jfa_outline
                    || !render_2d
            }
        };

        let mut features = Vec::new();
        if needs_3d {
            features.extend(["render-3d", "deferred-shaders", "audio-advanced"]);
        }
        features.push("gamepad");
        if web.devtools {
            features.push("devtools");
        }
        if kernels_js {
            features.push("kernels");
        }
        if web.locales {
            features.push("locales");
        }
        if web.saved {
            features.push("saved-adapter");
        }
        if sound_recipes {
            features.push("sound-recipes");
        }
        if texture_recipes {
            features.push("texture-recipes");
        }
        if render_2d {
            features.push("render-2d");
        }
        if needs_3d {
            if mesh_json {
                features.push("mesh-json");
            }
            if shader_graph {
                features.push("shader-graph");
            }
            if video {
                features.push("video");
            }
            if render_env {
                features.push("render-env");
            }
            if render_particles {
                features.push("render-particles");
            }
            if render_lines {
                features.push("render-lines");
            }
            if render_jfa_outline {
                features.push("render-jfa-outline");
            }
            if render_meshlets {
                features.push("render-meshlets");
            }
            // Standard RenderFrame header passes (deferred into shaders/*.wgsl so browsers only
            // fetch them if activated; authors can drop any via `pen.json` or `--web-features`).
            features.extend(["render-oit", "render-atmosphere", "render-pbr-decals", "render-post-fx"]);
        }

        Ok(Self { structured_js, kernels_js, features })
    }

    /// Tightest prebuilt release host directory (`web-2d`, `web-3d`, `web-compact`, or `web`)
    /// that supports every detected feature.
    pub fn prebuilt_tier(&self) -> &'static str {
        if !self.features.contains(&"render-3d") && self.features.iter().all(|f| WEB_2D_FEATURES.contains(f)) {
            "web-2d"
        } else if self.features.iter().all(|f| WEB_3D_FEATURES.contains(f)) {
            "web-3d"
        } else if self.features.iter().all(|f| WEB_COMPACT_FEATURES.contains(f)) {
            "web-compact"
        } else {
            "web"
        }
    }
}

/// Copies the runtime browser host files (plus deferred `shaders/` when present) from `from` into `out`.
#[cfg(feature = "compiler")]
fn copy_web_host_dir(from: &Path, out: &Path, ship_structured: bool, ship_kernels: bool) -> Result<()> {
    for file in ["pen_web.js", "pen_web_bg.wasm", "guest.js", "index.html"] {
        std::fs::copy(from.join(file), out.join(file)).with_context(|| format!("copying browser host {file}"))?;
    }
    if ship_structured && from.join("structured.js").is_file() {
        std::fs::copy(from.join("structured.js"), out.join("structured.js"))
            .context("copying browser host structured.js")?;
    }
    if ship_kernels {
        for file in ["kernels.js", "kernel_worker.js"] {
            if from.join(file).is_file() {
                std::fs::copy(from.join(file), out.join(file))
                    .with_context(|| format!("copying browser host {file}"))?;
            }
        }
    }
    let shaders = from.join("shaders");
    if shaders.is_dir() {
        copy_tree(&shaders, &out.join("shaders")).context("copying browser host deferred shaders")?;
    }
    Ok(())
}

/// Content-addressed cache directory (`~/.cache/pen/web-hosts/<hash>/`) for a tailored
/// `pen-web` build from this source checkout.
#[cfg(feature = "compiler")]
fn tailored_web_cache_dir(features: &str) -> Result<Option<PathBuf>> {
    use sha2::{Digest, Sha256};

    if std::env::var_os("PEN_WEB_NO_CACHE").is_some_and(|v| v != "0") {
        return Ok(None);
    }
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut canonical: Vec<&str> = features.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    canonical.sort_unstable();
    canonical.dedup();

    let mut hash = Sha256::new();
    hash.update(b"pen-web-host-v2\0");
    hash.update(canonical.join(",").as_bytes());
    hash.update(b"\0");
    let build_std = std::env::var("PEN_WEB_BUILD_STD").unwrap_or_else(|_| "auto".to_owned());
    hash.update(build_std.as_bytes());
    hash.update(b"\0");

    fn hash_file(hash: &mut Sha256, root: &Path, path: &Path) -> Result<()> {
        let rel = path.strip_prefix(root).unwrap_or(path);
        hash.update(rel.to_string_lossy().as_bytes());
        hash.update([0]);
        let bytes = std::fs::read(path).with_context(|| format!("hashing {}", path.display()))?;
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(&bytes);
        Ok(())
    }

    fn hash_tree(hash: &mut Sha256, root: &Path, dir: &Path, skip_dirs: &[&str]) -> Result<()> {
        if !dir.is_dir() {
            return Ok(());
        }
        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
        entries.sort();
        for path in entries {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if path.is_dir() {
                if !skip_dirs.contains(&name) {
                    hash_tree(hash, root, &path, skip_dirs)?;
                }
            } else if path.is_file() {
                hash_file(hash, root, &path)?;
            }
        }
        Ok(())
    }

    let lock = repo.join("Cargo.lock");
    if lock.is_file() {
        hash_file(&mut hash, &repo, &lock)?;
    }
    hash_tree(&mut hash, &repo, &repo.join("crates/pen-web"), &["pkg", "tests", "target"])?;
    for crate_name in ["pen-render", "pen-runtime", "pen-protocol", "pen-simulation"] {
        let crate_dir = repo.join("crates").join(crate_name);
        for file in ["Cargo.toml", "build.rs"] {
            let path = crate_dir.join(file);
            if path.is_file() {
                hash_file(&mut hash, &repo, &path)?;
            }
        }
        hash_tree(&mut hash, &repo, &crate_dir.join("src"), &[])?;
    }

    let digest: String = hash.finalize()[..8].iter().map(|b| format!("{b:02x}")).collect();
    let base = dirs::cache_dir().unwrap_or_else(|| PathBuf::from("/var/tmp")).join("pen/web-hosts");
    Ok(Some(base.join(digest)))
}

/// The browser host's files, into `out`: PEN_WEB_HOST's, those `just
/// install` put beside this executable, else built from this repository.
#[cfg(feature = "compiler")]
pub fn web_host(out: &Path, compact: bool) -> Result<()> {
    web_host_with_plan(out, compact, None, None)
}

#[cfg(feature = "compiler")]
fn find_prebuilt_beside(candidates: &[&str]) -> Option<PathBuf> {
    let parent = std::env::current_exe().ok()?.parent()?.to_path_buf();
    for dir in candidates {
        let beside = parent.join(dir);
        if beside.join("pen_web_bg.wasm").is_file() {
            return Some(beside);
        }
    }
    None
}

#[cfg(feature = "compiler")]
fn web_host_with_plan(
    out: &Path,
    compact: bool,
    plan: Option<&WebHostPlan>,
    explicit_features: Option<&str>,
) -> Result<()> {
    let ship_structured = plan.is_none_or(|p| p.structured_js);
    let ship_kernels = match explicit_features {
        Some(list) => list.split(',').any(|f| matches!(f.trim(), "kernels" | "compact" | "default")),
        None => plan.is_none_or(|p| p.kernels_js),
    };
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../pen-web/build.sh");
    let tailored_features = explicit_features.map(str::to_owned).or_else(|| {
        (compact && std::env::var_os("PEN_WEB_COMPACT_HOST").is_none() && script.is_file())
            .then(|| plan.map(|p| p.features.join(",")))
            .flatten()
    });

    if let Some(features) = tailored_features {
        // Binary-only installs without build.sh can still serve named preset tiers directly.
        if !script.is_file() {
            let preset_dir = match features.trim() {
                "web-2d" => Some("web-2d"),
                "web-3d" => Some("web-3d"),
                "compact" => Some("web-compact"),
                "default" => Some("web"),
                _ => None,
            };
            if let Some(dir) = preset_dir
                && let Some(host) = find_prebuilt_beside(&[dir])
            {
                return copy_web_host_dir(&host, out, ship_structured, ship_kernels);
            }
            bail!("tailoring the browser host requires crates/pen-web/build.sh");
        }
        if let Some(cache_dir) = tailored_web_cache_dir(&features)? {
            if cache_dir.join("pen_web_bg.wasm").is_file() && cache_dir.join("pen_web.js").is_file() {
                eprintln!("using cached browser host {}", cache_dir.display());
                copy_web_host_dir(&cache_dir, out, ship_structured, ship_kernels)?;
                if ship_structured && !out.join("structured.js").is_file() {
                    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../pen-web/structured.js");
                    std::fs::copy(&src, out.join("structured.js")).context("copying browser host structured.js")?;
                }
                return Ok(());
            }
            if let Some(parent) = cache_dir.parent() {
                std::fs::create_dir_all(parent)?;
                let staging = parent.join(format!(
                    ".tmp-{}-{}",
                    cache_dir.file_name().unwrap_or_default().to_string_lossy(),
                    std::process::id()
                ));
                let _ = std::fs::remove_dir_all(&staging);
                let mut build = Command::new("sh");
                build.arg(&script).arg(&staging).arg("--features").arg(&features);
                let status = build.status().context("building tailored pen-web")?;
                if !status.success() {
                    let _ = std::fs::remove_dir_all(&staging);
                    bail!("the browser host did not build");
                }
                let final_dir = if std::fs::rename(&staging, &cache_dir).is_ok() { &cache_dir } else { &staging };
                copy_web_host_dir(final_dir, out, ship_structured, ship_kernels)?;
                if final_dir == &staging {
                    let _ = std::fs::remove_dir_all(&staging);
                }
                if ship_structured && !out.join("structured.js").is_file() {
                    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../pen-web/structured.js");
                    std::fs::copy(&src, out.join("structured.js")).context("copying browser host structured.js")?;
                }
                return Ok(());
            }
        }
        let mut build = Command::new("sh");
        build.arg(&script).arg(out).arg("--features").arg(&features);
        let status = build.status().context("building tailored pen-web")?;
        ensure!(status.success(), "the browser host did not build");
        if ship_structured && !out.join("structured.js").is_file() {
            let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../pen-web/structured.js");
            std::fs::copy(&src, out.join("structured.js")).context("copying browser host structured.js")?;
        }
        return Ok(());
    }

    let variable = if compact { "PEN_WEB_COMPACT_HOST" } else { "PEN_WEB_HOST" };
    let host = std::env::var_os(variable).map(PathBuf::from).or_else(|| {
        if compact {
            let tier = plan.map_or("web-compact", WebHostPlan::prebuilt_tier);
            let candidates: &[&str] = match tier {
                "web-2d" => &["web-2d", "web-3d", "web-compact", "web"],
                "web-3d" => &["web-3d", "web-compact", "web"],
                "web-compact" => &["web-compact", "web"],
                _ => &["web", "web-compact"],
            };
            find_prebuilt_beside(candidates)
        } else {
            find_prebuilt_beside(&["web"])
        }
    });
    if let Some(host) = host {
        copy_web_host_dir(&host, out, ship_structured, ship_kernels)?;
    } else {
        ensure!(script.is_file(), "set {variable} to a built pen-web host directory");
        let mut build = Command::new("sh");
        build.arg(&script).arg(out);
        if compact {
            build.arg("--compact");
        }
        let status = build.status().context("building pen-web")?;
        ensure!(status.success(), "the browser host did not build");
        if !ship_structured {
            let _ = std::fs::remove_file(out.join("structured.js"));
        }
        if !ship_kernels {
            let _ = std::fs::remove_file(out.join("kernels.js"));
            let _ = std::fs::remove_file(out.join("kernel_worker.js"));
        }
    }
    Ok(())
}

/// The browser front end, guest programs, the assets the game uses and its
/// manifest as a complete static site, ready to host (`web_export`): its
/// files precompressed and a service worker; and, given `zip`, as a ZIP.
#[cfg(feature = "compiler")]
pub fn export_web(
    source: GameSource<'_>,
    out: &Path,
    zip: Option<&Path>,
    optimize: bool,
    compact: bool,
    explicit_features: Option<&str>,
) -> Result<()> {
    let has_devtools = match &source {
        GameSource::Checkout(project) => project.devtools().is_some(),
        GameSource::Bundle(directory) => directory.join("devtools.wasm").is_file(),
    };
    ensure!(!compact || !has_devtools, "--web-compact cannot include a dev tools program");
    let staging = crate::staging::Staging::new(source.root(), out)?;
    let site = &staging.directory;
    let PreparedGame { wasm, assets } = prepare_game(&source, site, optimize)?;
    crate::services::locale::ship_tables(&source.assets(), site)?;
    match &source {
        GameSource::Checkout(project) => project.compile_devtools(site)?,
        GameSource::Bundle(directory) => {
            let devtools = directory.join("devtools.wasm");
            if devtools.is_file() {
                std::fs::copy(&devtools, site.join("devtools.wasm"))
                    .with_context(|| format!("copying {}", devtools.display()))?;
            }
        }
    }
    if site.join("devtools.wasm").is_file() {
        shipping_wasm(&site.join("devtools.wasm"), optimize)?;
    }
    let catalog = source.assets().join("catalog.json");
    if catalog.is_file() {
        std::fs::copy(&catalog, site.join("catalog.json")).with_context(|| format!("copying {}", catalog.display()))?;
    }
    if site.join("main.scene").is_file() {
        std::fs::rename(site.join("main.scene"), site.join("scene.bin"))?;
    }
    let assets_dir = site.join("assets");
    let web = pen_runtime::web::WebFiles {
        scene: site.join("scene.bin").is_file(),
        devtools: site.join("devtools.wasm").is_file(),
        saved: assets_dir.join(pen_runtime::state::CONFIG_FILE).is_file(),
        captions: assets_dir.join("captions.json").is_file(),
        locales: site.join("text.json").is_file(),
    };
    let web_config = match &source {
        GameSource::Checkout(project) => project.web.clone(),
        GameSource::Bundle(directory) => {
            if directory.join("pen.json").is_file() {
                WebConfig::of_checkout(directory)?
            } else {
                WebConfig::default()
            }
        }
    };
    let plan = WebHostPlan::detect(&wasm, &web, &assets_dir, &web_config)?;
    let configured_features = web_config.features.as_ref().map(|list| list.join(","));
    let effective_features = explicit_features.or(configured_features.as_deref());
    let identity = source.identity()?;
    Manifest { game: identity, runtime: None, worker_deadlines: false, web: Some(web) }.write(site)?;
    web_host_with_plan(site, compact, Some(&plan), effective_features)?;
    notices(site)?;
    let hosting = crate::web_export::prepare(site)?;
    let host_note = match effective_features {
        Some(list) => format!("host features=[{list}]"),
        None if compact => format!("tailored host features=[{}]", plan.features.join(",")),
        None => format!(
            "host js={}",
            [plan.structured_js.then_some("structured"), plan.kernels_js.then_some("kernels"),]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(",")
        ),
    };
    eprintln!(
        "exported {}: {assets}; {host_note}; {hosting}; {:.1} MB in all",
        out.display(),
        size_of(site) as f64 / 1e6
    );
    crate::reproducible::settle_times(site)?;
    staging.publish()?;
    if let Some(zip) = zip {
        crate::web_export::zip(out, zip)?;
    }
    Ok(())
}

#[cfg(feature = "compiler")]
fn shipping_wasm(path: &Path, optimize: bool) -> Result<Vec<u8>> {
    if optimize {
        #[cfg(feature = "compiler")]
        crate::optimize::file(path)?;
        #[cfg(not(feature = "compiler"))]
        bail!("shipping optimization needs a full pen build");
    }
    Ok(std::fs::read(path)?)
}

/// Checks the explicit state adapter declaration beside the asset catalog,
/// including any adapter-owned migration layouts (docs/state-adapters.md).
#[cfg(feature = "compiler")]
pub fn check_saved_types(assets: &Path) -> Result<()> {
    let path = assets.join(pen_runtime::state::CONFIG_FILE);
    let Ok(text) = std::fs::read_to_string(&path) else { return Ok(()) };
    pen_ink::adapter(&text).with_context(|| format!("reading {}", path.display()))?;
    eprintln!("{}: state adapter checked", path.display());
    Ok(())
}

/// A compiled bundle for tests and packet tools, without a player executable.
#[cfg(feature = "compiler")]
pub fn export_bundle(project: &Project, out: &Path) -> Result<()> {
    let staging = crate::staging::Staging::new(&project.root, out)?;
    project.compile(&staging.directory)?;
    copy_tree(&project.assets(), &staging.directory.join("assets"))?;
    copy_scene(project, &staging.directory.join("main.scene"))?;
    crate::services::locale::ship_tables(&project.assets(), &staging.directory)?;
    project.copy_presets(&staging.directory)?;
    crate::reports::ship_config(&project.root, &staging.directory)?;
    let wasm = std::fs::read(staging.directory.join("game.wasm"))?;
    crate::scene_assets::write(&wasm, &project.assets(), &project.scene(), &staging.directory.join("assets"))?;
    Manifest { game: project.identity.clone(), runtime: None, worker_deadlines: false, web: None }
        .write(&staging.directory)?;
    staging.publish()
}

/// The checkout's starting scene to `to`, if it has one: a game without
/// scenes starts from what its program creates.
#[cfg(feature = "compiler")]
fn copy_scene(project: &Project, to: &Path) -> Result<()> {
    let scene = project.scene();
    if scene.is_file() {
        std::fs::copy(&scene, to).with_context(|| format!("copying {}", scene.display()))?;
    }
    Ok(())
}

#[cfg(any(feature = "compiler", test))]
pub(crate) fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "compiler"))]
mod tests {
    use super::*;

    #[test]
    fn executable_targets() {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            let exe = std::fs::read(std::env::current_exe().unwrap()).unwrap();
            assert_eq!(executable_target(&exe), Some("x86_64-unknown-linux-gnu"));
            // Its own source names Steam's libraries, but only imports Steam with the feature.
            #[cfg(not(feature = "steam"))]
            assert!(steam_library(Path::new("test"), &exe, "x86_64-unknown-linux-gnu").unwrap().is_none());
        }
        let mut pe = vec![0u8; 0x90];
        pe[..2].copy_from_slice(b"MZ");
        pe[0x3c] = 0x80;
        pe[0x80..0x84].copy_from_slice(b"PE\0\0");
        pe[0x84..0x86].copy_from_slice(&0x8664u16.to_le_bytes());
        assert_eq!(executable_target(&pe), Some("x86_64-pc-windows-gnu"));
        let macho = [0xcf, 0xfa, 0xed, 0xfe, 0x0c, 0x00, 0x00, 0x01];
        assert_eq!(executable_target(&macho), Some("aarch64-apple-darwin"));
        let elf_arm = {
            let mut elf = vec![0u8; 20];
            elf[..4].copy_from_slice(b"\x7fELF");
            elf[18..20].copy_from_slice(&0xb7u16.to_le_bytes());
            elf
        };
        assert_eq!(executable_target(&elf_arm), Some("aarch64-unknown-linux-gnu"));
        assert_eq!(executable_target(b"#!/bin/sh"), None);
        // A header cut short is unknown, not a panic.
        assert_eq!(executable_target(b"MZ"), None);
        assert_eq!(executable_target(b"\x7fELF"), None);
    }

    #[test]
    fn runtimes_carry_their_compatibility_mark() {
        // This test executable is a pen runtime too.
        let exe = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        assert_eq!(runtime_mark(&exe).unwrap().as_deref(), Some(RUNTIME));
        assert!(check_runtime(Path::new("pen"), &exe).is_ok());
        assert!(well_formed(RUNTIME), "{RUNTIME}");
        // Marks made here, not literals, which this executable would hold.
        let mark = |id: &str| [mark_prefix(), id.as_bytes().to_vec(), b"\n".to_vec()].concat();
        let older = [b"ELF...".to_vec(), mark("wasmtime 1.0.0, engine 0123456789abcdef"), b"...".to_vec()].concat();
        assert_eq!(runtime_mark(&older).unwrap().as_deref(), Some("wasmtime 1.0.0, engine 0123456789abcdef"));
        let refused = check_runtime(Path::new("player/pen"), &older).unwrap_err().to_string();
        assert!(refused.contains("just install-player"), "{refused}");
        assert!(check_runtime(Path::new("player/pen"), b"no mark").unwrap_err().to_string().contains("no runtime"));
        // Whatever else follows the prefix is not a mark.
        assert_eq!(runtime_mark(&[mark("8"), mark("wasmtime 1, engine 0")].concat()).unwrap(), None);
        let two = [mark("wasmtime 1, engine 0000000000000000"), mark("wasmtime 2, engine 0000000000000000")].concat();
        assert!(runtime_mark(&two).is_err());
    }

    fn fixture(name: &str) -> PathBuf {
        let directory = temporary_dir().unwrap().join(format!("project-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    #[test]
    fn default_shipping_requires_the_installed_player() {
        let root = fixture("default-player");
        std::fs::create_dir_all(&root).unwrap();
        let exe = root.join("pen");
        std::fs::write(&exe, b"development runtime").unwrap();
        let error = installed_player(&exe).unwrap_err().to_string();
        assert!(error.contains("just install-player") && error.contains("--runtime"));
        let player = root.join("player/pen");
        std::fs::create_dir_all(player.parent().unwrap()).unwrap();
        std::fs::write(&player, b"player runtime").unwrap();
        assert_eq!(installed_player(&exe).unwrap(), player);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn library_tests_find_configuration_without_a_game_entry() {
        let root = fixture("library-tests");
        std::fs::create_dir_all(root.join("tests")).unwrap();
        std::fs::write(root.join("pen.json"), "{}").unwrap();
        let found = Project::find_tests(Some(root.join("tests"))).unwrap();
        assert_eq!(found.root, root.canonicalize().unwrap());
        assert!(Project::find(Some(root.clone())).is_err());
        // Game commands and session scenarios still need an actual game.
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/main.blot"), "").unwrap();
        assert_eq!(Project::find(Some(root.clone())).unwrap().root, found.root);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn custom_compiler_entry_and_devtools_work_without_blot_files() {
        let root = fixture("custom-entry");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join("devtools")).unwrap();
        std::fs::write(root.join("Cargo.toml"), "").unwrap();
        std::fs::write(root.join("devtools/Cargo.toml"), "").unwrap();
        std::fs::write(
            root.join("pen.json"),
            r#"{"compiler": {"plugin": "rust", "entry": "Cargo.toml", "devtools": "devtools/Cargo.toml"}}"#,
        )
        .unwrap();
        let found = Project::find(Some(root.join("src"))).unwrap();
        assert_eq!(found.root, root.canonicalize().unwrap());
        assert_eq!(found.entry, PathBuf::from("Cargo.toml"));
        assert_eq!(found.devtools(), Some(PathBuf::from("devtools/Cargo.toml")));
        assert_eq!(Project::around(&root.join("src")), Some(root.clone()));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn identities_come_from_pen_json_or_the_checkout() {
        let root = fixture("identity").join("salt game");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/main.blot"), "").unwrap();
        std::fs::write(root.join("pen.json"), "{}").unwrap();
        let project = Project::find(Some(root.join("src"))).unwrap();
        assert_eq!(project.identity, Identity::named("salt game"));
        std::fs::write(root.join("pen.json"), r#"{"game": {"name": "gdev", "title": "gdev — Blot 3D"}}"#).unwrap();
        let identity = Identity::of_checkout(&root).unwrap();
        assert_eq!((identity.name.as_str(), identity.title.as_str()), ("gdev", "gdev — Blot 3D"));
        std::fs::write(root.join("pen.json"), r#"{"game": {"title": "Salt"}}"#).unwrap();
        let salt = Identity { name: "salt game".into(), title: "Salt".into(), players: 1 };
        assert_eq!(Identity::of_checkout(&root).unwrap(), salt);
        std::fs::write(root.join("pen.json"), r#"{"game": {"players": 4}}"#).unwrap();
        assert_eq!(Identity::of_checkout(&root).unwrap().players, 4);
        for players in ["0", "33", "2.5", r#""2""#] {
            std::fs::write(root.join("pen.json"), format!(r#"{{"game": {{"players": {players}}}}}"#)).unwrap();
            assert!(Identity::of_checkout(&root).is_err(), "{players}");
        }
        for name in [r#""../up""#, r#""a/b""#, r#""""#, r#"".hidden""#, r#""CON""#, r#""lpt1""#, "3"] {
            std::fs::write(root.join("pen.json"), format!(r#"{{"game": {{"name": {name}}}}}"#)).unwrap();
            assert!(Identity::of_checkout(&root).is_err(), "{name}");
        }
        assert_eq!(Identity::named("a/b: c").name, "a-b- c");
        assert_eq!(Identity::named("..").name, "game");
        std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn bundles_name_their_game_in_their_manifest() {
        let root = fixture("manifest");
        let bundle = root.join("gdev");
        std::fs::create_dir_all(&bundle).unwrap();
        // Without a manifest, after the directory.
        assert_eq!(bundle_identity(&bundle).unwrap(), Identity::named("gdev"));
        let salt = Identity { name: "salt".into(), title: "Salt!".into(), players: 1 };
        let manifest = Manifest { game: salt, runtime: Some(RUNTIME.into()), worker_deadlines: false, web: None };
        manifest.write(&bundle).unwrap();
        assert!(!std::fs::read_to_string(bundle.join("game.json")).unwrap().contains("players"), "one goes unsaid");
        let read = Manifest::read(&bundle).unwrap().unwrap();
        assert_eq!((read.game, read.runtime.as_deref()), (manifest.game.clone(), Some(RUNTIME)));
        assert_eq!(bundle_identity(&bundle).unwrap(), manifest.game);
        // A game of two players says so, and its package keeps it.
        let pair = Manifest {
            game: Identity { players: 2, ..manifest.game.clone() },
            runtime: None,
            worker_deadlines: false,
            web: None,
        };
        pair.write(&bundle).unwrap();
        assert_eq!(bundle_identity(&bundle).unwrap().players, 2);
        std::fs::write(bundle.join("game.json"), r#"{"name": "../x", "title": "x"}"#).unwrap();
        assert!(bundle_identity(&bundle).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    // The runtime it packages is an x86-64 ELF, and builds compile guests
    // only for their own architecture.
    #[test]
    #[cfg(all(feature = "compiler", target_arch = "x86_64"))]
    fn bundles_package_like_checkouts() {
        let root = fixture("bundle");
        // A game built elsewhere: its Wasm (here only a manifest), assets
        // and name, without a scene.
        let bundle = root.join("built");
        std::fs::create_dir_all(bundle.join("assets")).unwrap();
        let catalog = r#"{"meshes": ["quad.mesh.json"], "materials": ["fill.material.json"]}"#;
        std::fs::write(bundle.join("assets/catalog.json"), catalog).unwrap();
        let mesh = r#"{"positions": [0,0,0, 1,0,0, 0,1,0], "normals": [0,0,1, 0,0,1, 0,0,1], "uvs": [0,0, 1,0, 0,1], "indices": [0,1,2]}"#;
        std::fs::write(bundle.join("assets/quad.mesh.json"), mesh).unwrap();
        std::fs::write(bundle.join("assets/fill.material.json"), r#"{"shader": "fill.wgsl"}"#).unwrap();
        std::fs::write(bundle.join("assets/fill.wgsl"), "").unwrap();
        let wasm = [&b"\0asm\x01\0\0\0\0\x0c\x08blot:abi"[..], &[2, 0, 0]].concat();
        std::fs::write(bundle.join("game.wasm"), &wasm).unwrap();
        Manifest { game: Identity::named("gather"), runtime: None, worker_deadlines: false, web: None }
            .write(&bundle)
            .unwrap();
        // A runtime: an x86-64 ELF header and this pen's mark.
        let mut runtime = vec![0u8; 64];
        runtime[..4].copy_from_slice(b"\x7fELF");
        runtime[18..20].copy_from_slice(&0x3eu16.to_le_bytes());
        runtime.extend([mark_prefix(), RUNTIME.as_bytes().to_vec(), b"\n".to_vec()].concat());
        std::fs::write(root.join("player"), &runtime).unwrap();
        let options = PackageOptions {
            optimize: false,
            runtime: Some(root.join("player")),
            target: None,
            cpu: None,
            bundle_id: None,
            store: None,
            worker_deadlines: false,
        };
        let out = bundle.join("dist/package");
        package(GameSource::Bundle(&bundle), &out, options).unwrap();
        assert_eq!(std::fs::read(out.join("gather")).unwrap(), runtime, "the runtime, named after the game");
        assert!(!out.join("game/game.wasm").exists());
        assert_eq!(std::fs::read(out.join("game/game.meta")).unwrap(), blot_host::native_metadata(&wasm).unwrap());
        assert!(out.join("game/game.cwasm").is_file() && !out.join("game/main.scene").exists());
        let manifest = Manifest::read(&out.join("game")).unwrap().unwrap();
        assert_eq!((manifest.game.name.as_str(), manifest.runtime.as_deref()), ("gather", Some(RUNTIME)));
        // Exercise the player's decompression and Wasmtime loading path.
        crate::load_module(&blot_host::engine().unwrap(), &out.join("game"), false, false).unwrap();
        assert!(std::fs::read(out.join("game/assets/quad.mesh.json")).unwrap().starts_with(b"PMSH"));
        assert!(out.join("game/assets/fill.wgsl").is_file());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn packages_ship_the_assets_the_catalog_reaches() {
        let root = fixture("assets");
        let assets = root.join("assets");
        std::fs::create_dir_all(assets.join("sky")).unwrap();
        let catalog = r#"{"meshes": ["cube.mesh.json"], "materials": ["rock.material.json"], "sounds": ["hit.wav"]}"#;
        std::fs::write(assets.join("catalog.json"), catalog).unwrap();
        std::fs::write(assets.join("captions.json"), r#"{"hit.wav": "A thud"}"#).unwrap();
        let mesh = r#"{"positions": [0,0,0, 1,0,0, 0,1,0], "normals": [0,0,1, 0,0,1, 0,0,1], "uvs": [0,0, 1,0, 0,1], "indices": [0,1,2]}"#;
        std::fs::write(assets.join("cube.mesh.json"), mesh).unwrap();
        std::fs::write(assets.join("unused.mesh.json"), mesh).unwrap();
        let material =
            r#"{"shader": "scene.wgsl", "texture": "sky/rock.png", "normalTexture": null, "color": [1,1,1,1]}"#;
        std::fs::write(assets.join("rock.material.json"), material).unwrap();
        for file in [
            "scene.wgsl",
            "sky/rock.png",
            "hit.wav",
            "unused.wgsl",
            "unused.png",
            "notes.txt",
            "named.gtex",
            "saved.json",
        ] {
            std::fs::write(assets.join(file), file).unwrap();
        }
        let keys = asset_keys(&assets).unwrap();
        assert_eq!(keys.len(), 13);
        let used = used_assets(&assets, &keys, b"\0asm...named.gtex...");
        let expected = [
            "captions.json",
            "catalog.json",
            "cube.mesh.json",
            "hit.wav",
            "named.gtex",
            "notes.txt",
            "rock.material.json",
            "saved.json",
            "scene.wgsl",
            "sky/rock.png",
        ];
        assert_eq!(used.iter().map(String::as_str).collect::<Vec<_>>(), expected);
        // Without a catalog, nothing is left out.
        let uncatalogued: Vec<String> = keys.iter().filter(|key| *key != "catalog.json").cloned().collect();
        assert_eq!(used_assets(&assets, &uncatalogued, b"").len(), keys.len() - 1);

        let out = root.join("out");
        let shipped = ship_assets(&assets, &out, b"").unwrap();
        assert_eq!((shipped.files, shipped.of, shipped.meshes), (9, 13, 1));
        assert!(std::fs::read(out.join("cube.mesh.json")).unwrap().starts_with(b"PMSH"));
        assert_eq!(std::fs::read_to_string(out.join("sky/rock.png")).unwrap(), "sky/rock.png");
        assert!(!out.join("unused.mesh.json").exists() && !out.join("named.gtex").exists());
        // The source is never written through.
        assert_eq!(std::fs::read_to_string(assets.join("cube.mesh.json")).unwrap(), mesh);
        assert_eq!(shipped.bytes, size_of(&out));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn web_config_reads_from_pen_json() {
        let root = fixture("web_config");
        std::fs::write(root.join("pen.json"), "{}").unwrap();
        assert_eq!(WebConfig::of_checkout(&root).unwrap(), WebConfig::default());
        std::fs::write(root.join("pen.json"), r#"{"web": {"mode": "2d", "features": ["render-2d", "gamepad"]}}"#)
            .unwrap();
        let cfg = WebConfig::of_checkout(&root).unwrap();
        assert_eq!(cfg.mode.as_deref(), Some("2d"));
        assert_eq!(cfg.features, Some(vec!["render-2d".to_owned(), "gamepad".to_owned()]));
        for bad in [
            r#"{"web": 1}"#,
            r#"{"web": {"mode": "4d"}}"#,
            r#"{"web": {"features": "render-2d"}}"#,
            r#"{"web": {"features": [1]}}"#,
        ] {
            std::fs::write(root.join("pen.json"), bad).unwrap();
            assert!(WebConfig::of_checkout(&root).is_err(), "{bad}");
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn web_host_plan_only_includes_features_the_game_uses() {
        let root = fixture("web_plan");
        let assets = root.join("assets");
        std::fs::create_dir_all(&assets).unwrap();
        let web =
            pen_runtime::web::WebFiles { scene: false, devtools: false, saved: false, captions: false, locales: false };

        // 1. Pure 2D game (no meshes, materials, or scene; uses RENDER2D_EXT_TAG 65_008):
        //    auto-detected as 2D -> omits render-3d, deferred-shaders, audio-advanced, and 3D passes.
        let mut wasm_2d = b"\0asm\x01\0\0\0".to_vec();
        wasm_2d.extend_from_slice(&65_008.0f32.to_le_bytes());
        let pure_2d = WebHostPlan::detect(&wasm_2d, &web, &assets, &WebConfig::default()).unwrap();
        assert!(!pure_2d.structured_js);
        assert!(!pure_2d.kernels_js);
        assert_eq!(pure_2d.features, vec!["gamepad", "render-2d"]);
        assert_eq!(pure_2d.prebuilt_tier(), "web-2d");

        // 2. Forced `"mode": "2d"` sprite-only game (even without 2D extension tags):
        let mode_2d = WebConfig { mode: Some("2d".into()), features: None };
        let sprites_only = WebHostPlan::detect(b"\0asm\x01\0\0\0", &web, &assets, &mode_2d).unwrap();
        assert_eq!(sprites_only.features, vec!["gamepad"]);
        assert_eq!(sprites_only.prebuilt_tier(), "web-2d");

        // 3. Minimal static 3D game with only a packed PMSH mesh (without meshlets) and plain PNGs:
        //    selects render-3d + deferred-shaders + audio-advanced + gamepad + standard RenderFrame passes,
        //    fitting the slim `web-3d` prebuilt tier.
        std::fs::write(assets.join("cube.mesh.json"), b"PMSH\x01\0\0\0\0\0\0\0\0\0\0\0").unwrap();
        let minimal = WebHostPlan::detect(b"\0asm\x01\0\0\0", &web, &assets, &WebConfig::default()).unwrap();
        assert!(!minimal.structured_js);
        assert!(!minimal.kernels_js);
        assert_eq!(
            minimal.features,
            vec![
                "render-3d",
                "deferred-shaders",
                "audio-advanced",
                "gamepad",
                "render-oit",
                "render-atmosphere",
                "render-pbr-decals",
                "render-post-fx",
            ]
        );
        assert_eq!(minimal.prebuilt_tier(), "web-3d");

        // 4. 3D game with locales + 2D UI (still fits `web-compact`, no runtime recipes):
        let web_locales = pen_runtime::web::WebFiles { locales: true, ..web };
        let compact_3d = WebHostPlan::detect(&wasm_2d, &web_locales, &assets, &WebConfig::default()).unwrap();
        assert!(compact_3d.features.contains(&"render-3d"));
        assert!(compact_3d.features.contains(&"locales"));
        assert!(compact_3d.features.contains(&"render-2d"));
        assert_eq!(compact_3d.prebuilt_tier(), "web-compact");

        // 5. Game using locales, saved.json, .sound recipe, .matgraph.json, GTEX v3 (even under .png),
        //    PMSH mesh with MESH_MESHLETS flag, unpacked JSON mesh, and 2D + lines extension tags:
        std::fs::write(assets.join("boom.sound"), b"{}").unwrap();
        std::fs::write(assets.join("look.matgraph.json"), b"{}").unwrap();
        std::fs::write(assets.join("proc.png"), b"GTEX\x03\x04\x01\x01").unwrap();
        std::fs::write(assets.join("hi.mesh.json"), b"PMSH\x02\x80\0\0\0\0\0\0\0\0\0\0").unwrap();
        std::fs::write(assets.join("raw.mesh.json"), b"{}").unwrap();
        let web_full =
            pen_runtime::web::WebFiles { scene: true, devtools: false, saved: true, captions: true, locales: true };
        let mut wasm_with_tags = b"\0asm\x01\0\0\0".to_vec();
        wasm_with_tags.extend_from_slice(&65_008.0f32.to_le_bytes());
        wasm_with_tags.extend_from_slice(&65_006.0f32.to_le_bytes());
        let detected = WebHostPlan::detect(&wasm_with_tags, &web_full, &assets, &WebConfig::default()).unwrap();
        for expected in [
            "render-3d",
            "deferred-shaders",
            "audio-advanced",
            "gamepad",
            "locales",
            "saved-adapter",
            "sound-recipes",
            "mesh-json",
            "texture-recipes",
            "shader-graph",
            "render-2d",
            "render-lines",
            "render-meshlets",
        ] {
            assert!(detected.features.contains(&expected), "missing {expected} in {:?}", detected.features);
        }
        assert_eq!(detected.prebuilt_tier(), "web");
        assert!(!detected.features.contains(&"render-env"));
        assert!(!detected.features.contains(&"render-particles"));
        assert!(!detected.features.contains(&"render-jfa-outline"));
        assert!(!detected.features.contains(&"kernels"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
