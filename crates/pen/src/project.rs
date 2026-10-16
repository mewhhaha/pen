//! A checkout as a project the binary develops, runs and packages: its
//! programs compile through the Wasm plugins its `pen.json` chooses
//! (`plugin_build`), and its assets and scene are used in place.
use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

/// Who a game is: `name` names its data directory (saves, preferences and
/// crash reports) and the executable `pen build` ships it as; `title` is its
/// window's, which assistive technology reads too.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Identity {
    pub name: String,
    pub title: String,
}

impl Identity {
    /// A game called `name` (made a file name), with that title too.
    pub fn named(name: &str) -> Identity {
        let name: String =
            name.chars().map(|character| if file_name_character(character) { character } else { '-' }).collect();
        let name = name.trim_start_matches('.').trim_end_matches([' ', '.']);
        let name = if name.is_empty() { "game" } else { name };
        Identity { name: name.to_owned(), title: name.to_owned() }
    }

    /// The `game` block of a checkout's `pen.json`, `{"name": "gdev",
    /// "title": "gdev — Blot 3D"}`, either of which may be left out: the
    /// name is then the checkout directory's, and the title the name.
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
            identity = Identity { title: name.clone(), name };
            identity.check().with_context(|| format!("{}: game.name", file.display()))?;
        }
        if let Some(title) = field("title")? {
            identity.title = title;
        }
        Ok(identity)
    }

    /// A name that is a file name on every platform.
    fn check(&self) -> Result<()> {
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

/// What a bundle says of itself in its `game.json`: its game, and the
/// runtime its `game.cwasm` was precompiled for, if it has one.
#[derive(Debug, Deserialize, Serialize)]
pub struct Manifest {
    #[serde(flatten)]
    pub game: Identity,
    /// [`RUNTIME`] of the pen that precompiled `game.cwasm`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<String>,
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

/// What a mark starts with, made here rather than written out: this
/// executable's bytes then hold it only in its own mark, and nowhere two of
/// its strings happen to lie side by side.
fn mark_prefix() -> Vec<u8> {
    b"=ytilibitapmoc-emitnur-nep".iter().rev().copied().collect()
}

/// Whether `text` is a mark as build.rs makes them.
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
fn runtime_mark(executable: &[u8]) -> Result<Option<String>> {
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

pub struct Project {
    pub root: PathBuf,
    pub identity: Identity,
    /// Its Wasm plugins, loaded when first asked for.
    #[cfg(feature = "compiler")]
    plugins: std::sync::OnceLock<Option<std::sync::Arc<pen_plugins::Plugins>>>,
}

impl Project {
    /// The checkout at `root`, or the one the current directory is in.
    pub fn find(root: Option<PathBuf>) -> Result<Project> {
        let start = match root {
            Some(root) => root,
            None => std::env::current_dir()?,
        };
        let start = std::fs::canonicalize(&start).with_context(|| format!("{} does not exist", start.display()))?;
        match Project::around(&start) {
            Some(root) => Ok(Project {
                identity: Identity::of_checkout(&root)?,
                root,
                #[cfg(feature = "compiler")]
                plugins: std::sync::OnceLock::new(),
            }),
            None => bail!(
                "{} is not inside a checkout (a directory with pen.json and src/main.blot); pass --project DIR",
                start.display()
            ),
        }
    }

    /// The checkout `directory` is in, if any.
    pub fn around(directory: &Path) -> Option<PathBuf> {
        directory
            .ancestors()
            .find(|directory| directory.join("pen.json").is_file() && directory.join("src/main.blot").is_file())
            .map(Path::to_owned)
    }

    pub fn assets(&self) -> PathBuf {
        self.root.join("src/assets")
    }

    pub fn scene(&self) -> PathBuf {
        self.root.join("src/main.scene")
    }

    /// Where `dev` and `run` keep the compiled program.
    pub fn native(&self) -> PathBuf {
        self.root.join(".pen/runtime")
    }

    /// The dev tools' program (gdev's `devtools/main.blot`), relative to
    /// the checkout, if it has one.
    pub fn devtools(&self) -> Option<&'static str> {
        ["devtools/main.blot", "src/devtools/main.blot"].into_iter().find(|entry| self.root.join(entry).is_file())
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
            crate::plugin_build::compile(plugins, "game", &self.root.join("src/main.blot"), &out.join("game.wasm"))
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
                &self.root.join("src/main.blot"),
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
fn overlay_waiting() {
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| crate::devlog::info(crate::devlog::Source::Build, "waiting for an edit that compiles"));
}

/// Plugin build workers; dropping them joins their explicit shutdown.
pub struct Watcher {
    #[cfg(feature = "compiler")]
    plugin: Option<(pen_plugins::Watching, std::sync::mpsc::Receiver<bool>)>,
}

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
fn asset_keys(directory: &Path) -> Result<Vec<String>> {
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
const NAMED_KINDS: [&str; 8] = [".json", ".wgsl", ".png", ".jpg", ".jpeg", ".gtex", ".wav", ".ogg"];

/// Every string in a JSON value.
fn strings<'a>(value: &'a serde_json::Value, each: &mut impl FnMut(&'a str)) {
    match value {
        serde_json::Value::String(text) => each(text),
        serde_json::Value::Array(values) => values.iter().for_each(|value| strings(value, each)),
        serde_json::Value::Object(fields) => fields.values().for_each(|value| strings(value, each)),
        _ => {}
    }
}

/// Of `keys` (in `assets`), those a game uses: `catalog.json`, what it
/// names, and what those name in turn (a material's shader and textures).
/// To be safe it keeps any file of a kind it does not know, and any `game`'s
/// Wasm names; without a catalog, all of them.
fn used_assets(assets: &Path, keys: &[String], game: &[u8]) -> BTreeSet<String> {
    let known: HashSet<&str> = keys.iter().map(String::as_str).collect();
    if !known.contains("catalog.json") {
        return keys.iter().cloned().collect();
    }
    let mut used = BTreeSet::new();
    let mut pending = vec!["catalog.json".to_owned()];
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
    let named = |key: &str| game.windows(key.len()).any(|window| window == key.as_bytes());
    for key in keys {
        if !NAMED_KINDS.iter().any(|kind| key.ends_with(kind)) || named(key) {
            used.insert(key.clone());
        }
    }
    used
}

/// What [`ship_assets`] shipped, for a package's summary.
struct Shipped {
    files: usize,
    of: usize,
    bytes: u64,
    of_bytes: u64,
    meshes: usize,
}

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
/// JSON in the renderer's binary form ([`pen_render::pack_mesh`]), which
/// loads faster and is a fraction of the size.
fn ship_assets(from: &Path, to: &Path, wasm: &[u8]) -> Result<Shipped> {
    let keys = asset_keys(from)?;
    let used = used_assets(from, &keys, wasm);
    let mut shipped = Shipped { files: used.len(), of: keys.len(), bytes: 0, of_bytes: 0, meshes: 0 };
    for key in &keys {
        let source = from.join(key);
        shipped.of_bytes += std::fs::metadata(&source)?.len();
        if !used.contains(key) {
            continue;
        }
        let target = to.join(key);
        std::fs::create_dir_all(target.parent().context("an asset has no directory")?)?;
        let packed = if key.ends_with(".mesh.json") {
            match pen_render::pack_mesh(&std::fs::read(&source)?) {
                Ok(packed) => Some(packed),
                Err(error) => {
                    eprintln!("shipping {key} as it is: {error:#}");
                    None
                }
            }
        } else {
            None
        };
        // A new file, never one sharing the source's: tools rewrite assets
        // in place.
        match packed {
            Some(packed) => {
                std::fs::write(&target, packed)?;
                shipped.meshes += 1;
            }
            None => {
                std::fs::copy(&source, &target).with_context(|| format!("copying {}", source.display()))?;
            }
        }
        shipped.bytes += std::fs::metadata(&target)?.len();
    }
    Ok(shipped)
}

/// What `pen build` ships.
pub struct PackageOptions {
    /// The runtime to ship. By default the player runtime (no compiler)
    /// that `just install-player` installs beside this executable, in
    /// `player/`, else this executable.
    pub runtime: Option<PathBuf>,
    /// The Rust target the runtime was built for, when its executable header
    /// does not say. cargo-zigbuild's glibc suffix (`.2.28`) may be left on.
    pub target: Option<String>,
    /// The CPU game.cwasm is compiled for: a Cranelift preset, or `native`
    /// (see blot_host::portable_engine; x86-64-v2 by default).
    pub cpu: Option<String>,
}

/// The runtime `pen build` ships by default.
fn default_runtime() -> Result<PathBuf> {
    let exe = std::env::current_exe()?;
    let player = exe.with_file_name("player").join(exe.file_name().context("the executable has no name")?);
    if player.is_file() {
        return Ok(player);
    }
    eprintln!(
        "shipping this pen, which includes the Wasm compiler; `just install-player` installs the smaller \
         player runtime `pen build` ships instead"
    );
    Ok(exe)
}

/// The Rust target an executable was built for, from its header.
fn executable_target(header: &[u8]) -> Option<&'static str> {
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

fn notices(out: &Path) -> Result<()> {
    std::fs::write(out.join("LICENSE"), include_str!("../../../LICENSE"))?;
    std::fs::write(out.join("THIRD_PARTY_NOTICES.md"), include_str!("../../../THIRD_PARTY_NOTICES.md"))?;
    Ok(())
}

pub(crate) fn size_of(path: &Path) -> u64 {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => {
            std::fs::read_dir(path).into_iter().flatten().flatten().map(|entry| size_of(&entry.path())).sum()
        }
        Ok(metadata) => metadata.len(),
        Err(_) => 0,
    }
}

/// A shippable folder: the runtime, the libraries it needs beside it, and
/// the game: its compiled program, precompiled for the runtime's platform
/// and a CPU baseline players' machines have, the assets it uses, its
/// starting scene and its manifest.
pub fn package(project: &Project, out: &Path, options: PackageOptions) -> Result<()> {
    let staging = crate::staging::Staging::new(&project.root, out)?;
    package_into(project, &staging.directory, out, options)?;
    staging.publish()
}

fn package_into(project: &Project, out: &Path, destination: &Path, options: PackageOptions) -> Result<()> {
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

    let game = out.join("game");
    std::fs::create_dir_all(&game)?;
    project.compile(&game)?;
    let wasm = std::fs::read(game.join("game.wasm"))?;
    let assets = ship_assets(&project.assets(), &game.join("assets"), &wasm)?;
    std::fs::copy(project.scene(), game.join("main.scene"))?;
    std::fs::write(game.join("game.cwasm"), blot_host::precompile(&engine, &wasm)?)?;
    Manifest { game: project.identity.clone(), runtime: Some(RUNTIME.to_owned()) }.write(&game)?;
    // Named after the game, whatever the runtime is called.
    let name = &project.identity.name;
    let executable = if target.contains("windows") { format!("{name}.exe") } else { name.clone() };
    std::fs::copy(&runtime, out.join(executable))?;
    if let Some(library) = &steam {
        std::fs::copy(library, out.join(library.file_name().unwrap()))?;
    }
    notices(out)?;
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
    Ok(())
}

/// The browser host's files, into `out`: PEN_WEB_HOST's, those `just
/// install` put beside this executable, else built from this repository.
pub fn web_host(out: &Path) -> Result<()> {
    let host = std::env::var_os("PEN_WEB_HOST").map(PathBuf::from).or_else(|| {
        let beside = std::env::current_exe().ok()?.parent()?.join("web");
        beside.join("pen_web_bg.wasm").is_file().then_some(beside)
    });
    if let Some(host) = host {
        for file in ["pen_web.js", "pen_web_bg.wasm", "guest.js", "index.html"] {
            std::fs::copy(host.join(file), out.join(file)).with_context(|| format!("copying browser host {file}"))?;
        }
    } else {
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../pen-web/build.sh");
        ensure!(script.is_file(), "set PEN_WEB_HOST to a built pen-web host directory");
        let status = Command::new("sh").arg(script).arg(out).status().context("building pen-web")?;
        ensure!(status.success(), "the browser host did not build");
    }
    Ok(())
}

/// The browser front end, guest programs, the assets the game uses and its
/// manifest as a complete static site.
pub fn export_web(project: &Project, out: &Path) -> Result<()> {
    let staging = crate::staging::Staging::new(&project.root, out)?;
    let site = &staging.directory;
    project.compile(site)?;
    project.compile_devtools(site)?;
    let wasm = std::fs::read(site.join("game.wasm"))?;
    let assets = ship_assets(&project.assets(), &site.join("assets"), &wasm)?;
    std::fs::copy(project.assets().join("catalog.json"), site.join("catalog.json"))?;
    if project.scene().is_file() {
        std::fs::copy(project.scene(), site.join("scene.bin"))?;
    }
    Manifest { game: project.identity.clone(), runtime: None }.write(site)?;
    web_host(site)?;
    notices(site)?;
    eprintln!("exported {}: {assets}; {:.1} MB in all", out.display(), size_of(site) as f64 / 1e6);
    staging.publish()
}

/// A compiled bundle for tests and packet tools, without a player executable.
pub fn export_bundle(project: &Project, out: &Path) -> Result<()> {
    let staging = crate::staging::Staging::new(&project.root, out)?;
    project.compile(&staging.directory)?;
    copy_tree(&project.assets(), &staging.directory.join("assets"))?;
    std::fs::copy(project.scene(), staging.directory.join("main.scene"))?;
    Manifest { game: project.identity.clone(), runtime: None }.write(&staging.directory)?;
    staging.publish()
}

fn copy_tree(from: &Path, to: &Path) -> Result<()> {
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

#[cfg(test)]
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
        assert_eq!(Identity::of_checkout(&root).unwrap(), Identity { name: "salt game".into(), title: "Salt".into() });
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
        let manifest =
            Manifest { game: Identity { name: "salt".into(), title: "Salt!".into() }, runtime: Some(RUNTIME.into()) };
        manifest.write(&bundle).unwrap();
        let read = Manifest::read(&bundle).unwrap().unwrap();
        assert_eq!((read.game, read.runtime.as_deref()), (manifest.game.clone(), Some(RUNTIME)));
        assert_eq!(bundle_identity(&bundle).unwrap(), manifest.game);
        std::fs::write(bundle.join("game.json"), r#"{"name": "../x", "title": "x"}"#).unwrap();
        assert!(bundle_identity(&bundle).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn packages_ship_the_assets_the_catalog_reaches() {
        let root = fixture("assets");
        let assets = root.join("assets");
        std::fs::create_dir_all(assets.join("sky")).unwrap();
        let catalog = r#"{"meshes": ["cube.mesh.json"], "materials": ["rock.material.json"], "sounds": ["hit.wav"]}"#;
        std::fs::write(assets.join("catalog.json"), catalog).unwrap();
        let mesh = r#"{"positions": [0,0,0, 1,0,0, 0,1,0], "normals": [0,0,1, 0,0,1, 0,0,1], "uvs": [0,0, 1,0, 0,1], "indices": [0,1,2]}"#;
        std::fs::write(assets.join("cube.mesh.json"), mesh).unwrap();
        std::fs::write(assets.join("unused.mesh.json"), mesh).unwrap();
        let material =
            r#"{"shader": "scene.wgsl", "texture": "sky/rock.png", "normalTexture": null, "color": [1,1,1,1]}"#;
        std::fs::write(assets.join("rock.material.json"), material).unwrap();
        for file in ["scene.wgsl", "sky/rock.png", "hit.wav", "unused.wgsl", "unused.png", "notes.txt", "named.gtex"] {
            std::fs::write(assets.join(file), file).unwrap();
        }
        let keys = asset_keys(&assets).unwrap();
        assert_eq!(keys.len(), 11);
        let used = used_assets(&assets, &keys, b"\0asm...named.gtex...");
        let expected = [
            "catalog.json",
            "cube.mesh.json",
            "hit.wav",
            "named.gtex",
            "notes.txt",
            "rock.material.json",
            "scene.wgsl",
            "sky/rock.png",
        ];
        assert_eq!(used.iter().map(String::as_str).collect::<Vec<_>>(), expected);
        // Without a catalog, nothing is left out.
        assert_eq!(used_assets(&assets, &keys[1..], b"").len(), keys.len() - 1);

        let out = root.join("out");
        let shipped = ship_assets(&assets, &out, b"").unwrap();
        assert_eq!((shipped.files, shipped.of, shipped.meshes), (7, 11, 1));
        assert!(std::fs::read(out.join("cube.mesh.json")).unwrap().starts_with(b"PMSH"));
        assert_eq!(std::fs::read_to_string(out.join("sky/rock.png")).unwrap(), "sky/rock.png");
        assert!(!out.join("unused.mesh.json").exists() && !out.join("named.gtex").exists());
        // The source is never written through.
        assert_eq!(std::fs::read_to_string(assets.join("cube.mesh.json")).unwrap(), mesh);
        assert_eq!(shipped.bytes, size_of(&out));
        std::fs::remove_dir_all(root).unwrap();
    }
}
