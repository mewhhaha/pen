//! A gdev checkout as a project the binary develops, runs and packages:
//! its programs compile through the Wasm plugins its `pen.json` chooses
//! (`plugin_build`), or else with its own compiler setup through Deno (the
//! script is embedded here), and its assets and scene are used in place.
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};

const BUILD_SCRIPT: &str = include_str!("../scripts/blot_build.ts");

pub struct Project {
    pub root: PathBuf,
    /// Its Wasm plugins, loaded when first asked for.
    #[cfg(feature = "plugins")]
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
        for directory in start.ancestors() {
            if directory.join("blot_runtime.ts").is_file() && directory.join("src/main.blot").is_file() {
                return Ok(Project {
                    root: directory.to_owned(),
                    #[cfg(feature = "plugins")]
                    plugins: std::sync::OnceLock::new(),
                });
            }
        }
        bail!(
            "{} is not inside a gdev checkout (no blot_runtime.ts and src/main.blot); pass --project DIR",
            start.display()
        )
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

    fn script(&self) -> Result<PathBuf> {
        let directory = dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("pen");
        std::fs::create_dir_all(&directory)?;
        let path = directory.join(format!("blot_build-{}.ts", env!("CARGO_PKG_VERSION")));
        if std::fs::read_to_string(&path).ok().as_deref() != Some(BUILD_SCRIPT) {
            std::fs::write(&path, BUILD_SCRIPT)?;
        }
        Ok(path)
    }

    /// The dev tools' program (gdev's `devtools/main.blot`), relative to
    /// the checkout, if it has one.
    pub fn devtools(&self) -> Option<&'static str> {
        ["devtools/main.blot", "src/devtools/main.blot"].into_iter().find(|entry| self.root.join(entry).is_file())
    }

    /// The checkout's plugins, when its `pen.json` chooses a compiler: they
    /// build its programs instead of Deno.
    #[cfg(feature = "plugins")]
    fn plugins(&self) -> Result<Option<std::sync::Arc<pen_plugins::Plugins>>> {
        if let Some(plugins) = self.plugins.get() {
            return Ok(plugins.clone());
        }
        let loaded = crate::plugin_build::load(&self.root)?;
        Ok(self.plugins.get_or_init(|| loaded).clone())
    }

    /// What a plugin build never watches: what pen writes.
    #[cfg(feature = "plugins")]
    fn written(&self) -> Vec<PathBuf> {
        vec![self.root.join(".pen"), self.root.join("target")]
    }

    fn command(&self, out: &Path, watch: bool) -> Result<Command> {
        ensure!(
            blot_host::COMPILER,
            "this pen is a player runtime, built without the Wasm compiler: it plays packaged games \
             but cannot compile a checkout's program (use a full build, `just install`)"
        );
        let temporary = temporary_dir()?;
        let mut command = Command::new("deno");
        command
            .args(["run", "-A", "--no-lock", "--config"])
            .arg(self.root.join("deno.json"))
            .arg(self.script()?)
            .arg(&self.root)
            // Absolute, since deno runs in the checkout.
            .arg(std::path::absolute(out)?)
            .args(watch.then_some("--watch"))
            .current_dir(&self.root)
            .env("TMPDIR", &temporary);
        #[cfg(windows)]
        command.env("TMP", &temporary).env("TEMP", &temporary);
        Ok(command)
    }

    /// Starts building the dev tools' program into `out/devtools.wasm`
    /// beside the game's build, once or (`watch`) again as its sources
    /// change, until the returned watcher is dropped; none when the checkout
    /// has none. The runtime picks the program up when its file appears.
    pub fn build_devtools(&self, out: &Path, watch: bool) -> Result<Option<Watcher>> {
        let Some(entry) = self.devtools() else { return Ok(None) };
        std::fs::create_dir_all(out)?;
        #[cfg(feature = "plugins")]
        if let Some(plugins) = self.plugins()? {
            // Built again as it changes, in `run` too: the dev tools reload.
            let built = crate::plugin_build::watch(
                plugins,
                "dev tools",
                &self.root.join(entry),
                &out.join("devtools.wasm"),
                self.written(),
            )?;
            return Ok(Some(Watcher { child: None, plugin: Some(built) }));
        }
        let _ = watch;
        let mut child = self
            .command(out, watch)?
            .args(["--entry", entry, "--name", "devtools.wasm"])
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("running deno, which compiles the dev tools' Blot program")?;
        if let Some(stderr) = child.stderr.take() {
            crate::overlay::forward_build_output(stderr);
        }
        Ok(Some(Watcher::deno(child)))
    }

    /// Compiles the Blot program into `out/game.wasm` once.
    pub fn compile(&self, out: &Path) -> Result<()> {
        std::fs::create_dir_all(out)?;
        #[cfg(feature = "plugins")]
        if let Some(plugins) = self.plugins()? {
            return crate::plugin_build::compile(plugins, "game", &self.root.join("src/main.blot"), &out.join("game.wasm"));
        }
        let started = Instant::now();
        let status = self
            .command(out, false)?
            .status()
            .context("running deno, which compiles the game's Blot program")?;
        if !status.success() {
            bail!("the game's Blot program did not compile");
        }
        eprintln!("compiled {} in {:.1} s", out.join("game.wasm").display(), started.elapsed().as_secs_f64());
        Ok(())
    }

    /// Compiles into `out/game.wasm`, then keeps compiling as sources change,
    /// until the returned watcher is dropped. Returns once the first build is
    /// there.
    pub fn watch(&self, out: &Path) -> Result<Watcher> {
        std::fs::create_dir_all(out)?;
        let wasm = out.join("game.wasm");
        let _ = std::fs::remove_file(&wasm);
        #[cfg(feature = "plugins")]
        if let Some(plugins) = self.plugins()? {
            let (watching, built) =
                crate::plugin_build::watch(plugins, "game", &self.root.join("src/main.blot"), &wasm, self.written())?;
            // As with Deno, play starts with the first build that compiles.
            loop {
                match built.recv() {
                    Ok(true) => return Ok(Watcher { child: None, plugin: Some((watching, built)) }),
                    Ok(false) => overlay_waiting(),
                    Err(_) => bail!("the game's plugin build stopped"),
                }
            }
        }
        let mut child = self
            .command(out, true)?
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("running deno, which compiles the game's Blot program")?;
        // To stderr as before, and to the dev tools' console.
        if let Some(stderr) = child.stderr.take() {
            crate::overlay::forward_build_output(stderr);
        }
        loop {
            if wasm.exists() {
                return Ok(Watcher::deno(child));
            }
            if let Some(status) = child.try_wait()? {
                bail!("the game's Blot program did not compile ({status})");
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

/// The Blot compiler's temporary directory, on disk in the cache directory
/// rather than a RAM-backed /tmp: each compiler it opens unpacks a 35 MB
/// `blotc-*` there and removes it only when closed normally, so every
/// killed build leaves one behind. Those older than an hour are swept away.
pub(crate) fn temporary_dir() -> Result<PathBuf> {
    let directory = dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("pen/tmp");
    std::fs::create_dir_all(&directory)?;
    let hour_ago = std::time::SystemTime::now() - Duration::from_secs(3600);
    for entry in std::fs::read_dir(&directory)?.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let stale = entry.metadata().and_then(|metadata| metadata.modified()).is_ok_and(|time| time < hour_ago);
        if stale && (name.starts_with("blotc-") || name.starts_with("blot-std-")) {
            let path = entry.path();
            let _ = if path.is_dir() { std::fs::remove_dir_all(&path) } else { std::fs::remove_file(&path) };
        }
    }
    Ok(directory)
}

/// Says once, while waiting for a game that does not compile yet, that its
/// next edit is awaited.
#[cfg(feature = "plugins")]
fn overlay_waiting() {
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| crate::overlay::info(crate::overlay::Source::Build, "waiting for an edit that compiles"));
}

/// The compiler watching the project (Deno's, or the plugins' build
/// thread); stops when dropped.
pub struct Watcher {
    child: Option<Child>,
    #[cfg(feature = "plugins")]
    plugin: Option<(pen_plugins::Watching, std::sync::mpsc::Receiver<bool>)>,
}

impl Watcher {
    fn deno(child: Child) -> Watcher {
        Watcher {
            child: Some(child),
            #[cfg(feature = "plugins")]
            plugin: None,
        }
    }

    /// Waits for a build that does not watch (or a plugin build's first) to
    /// finish; whether it built.
    pub fn wait(&mut self) -> Result<bool> {
        #[cfg(feature = "plugins")]
        if let Some((_, built)) = &self.plugin {
            return Ok(built.recv().unwrap_or(false));
        }
        match &mut self.child {
            Some(child) => Ok(child.wait()?.success()),
            None => Ok(true),
        }
    }
}

impl Drop for Watcher {
    fn drop(&mut self) {
        let Some(child) = &mut self.child else { return };
        // Closing stdin asks the script to finish its build and stop.
        drop(child.stdin.take());
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

pub fn copy_tree(from: &Path, to: &Path) -> Result<()> {
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

/// What `pen build` ships.
pub struct PackageOptions {
    /// The runtime to ship. By default the player runtime (no compiler)
    /// that `just player` installs beside this executable, in `player/`,
    /// else this executable.
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
        "shipping this pen, which includes the Wasm compiler; `just player` installs the smaller \
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
    built
        .map(Some)
        .with_context(|| format!("{} needs {name} beside it, and it is not there", runtime.display()))
}

fn size_of(path: &Path) -> u64 {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => std::fs::read_dir(path)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| size_of(&entry.path()))
            .sum(),
        Ok(metadata) => metadata.len(),
        Err(_) => 0,
    }
}

/// A shippable folder: the runtime, the libraries it needs beside it, and
/// the game: its compiled program, precompiled for the runtime's platform
/// and a CPU baseline players' machines have, its assets and starting scene.
pub fn package(project: &Project, out: &Path, options: PackageOptions) -> Result<()> {
    let runtime = match options.runtime {
        Some(runtime) => runtime,
        None => default_runtime()?,
    };
    let runtime_bytes = std::fs::read(&runtime).with_context(|| format!("reading the runtime {}", runtime.display()))?;
    let target = match options.target {
        Some(target) => target.split('.').next().unwrap_or_default().to_owned(),
        None => executable_target(&runtime_bytes)
            .with_context(|| format!("cannot tell what {} runs on: pass --target TRIPLE", runtime.display()))?
            .to_owned(),
    };
    let steam = steam_library(&runtime, &runtime_bytes, &target)?;
    let engine = blot_host::portable_engine(Some(&target), options.cpu.as_deref())?;

    let game = out.join("game");
    let _ = std::fs::remove_dir_all(out);
    std::fs::create_dir_all(&game)?;
    project.compile(&game)?;
    copy_tree(&project.assets(), &game.join("assets"))?;
    std::fs::copy(project.scene(), game.join("main.scene"))?;
    let wasm = std::fs::read(game.join("game.wasm"))?;
    std::fs::write(game.join("game.cwasm"), blot_host::precompile(&engine, &wasm)?)?;
    // Named after the game, whatever the runtime is called.
    let executable = if target.contains("windows") { format!("{}.exe", crate::GAME) } else { crate::GAME.to_owned() };
    std::fs::copy(&runtime, out.join(executable))?;
    if let Some(library) = &steam {
        std::fs::copy(library, out.join(library.file_name().unwrap()))?;
    }
    let megabytes = |path: &Path| size_of(path) as f64 / 1e6;
    let x86_64 = target.starts_with("x86_64");
    let cpu = options.cpu.as_deref().unwrap_or(if x86_64 { blot_host::PORTABLE_X86_64 } else { "baseline" });
    eprintln!(
        "packaged {} for {target} ({} CPU): runtime {:.1} MB{}, game.cwasm {:.1} MB, assets {:.1} MB, {:.1} MB in all",
        out.display(),
        cpu,
        runtime_bytes.len() as f64 / 1e6,
        steam.as_ref().map(|library| format!(" + Steam {:.1} MB", megabytes(library))).unwrap_or_default(),
        megabytes(&game.join("game.cwasm")),
        megabytes(&game.join("assets")),
        megabytes(out),
    );
    Ok(())
}

/// The browser build as static files for any web host, from the checkout's
/// `scripts/export_web.ts`: its page and bundled code, the compiled game, its
/// assets and its starting scene.
pub fn export_web(project: &Project, out: &Path) -> Result<()> {
    let script = project.root.join("scripts/export_web.ts");
    ensure!(script.is_file(), "{} has no scripts/export_web.ts to export the browser build", project.root.display());
    let temporary = temporary_dir()?;
    let mut command = Command::new("deno");
    command
        .args(["run", "--allow-all"])
        .arg(&script)
        .arg(std::path::absolute(out)?)
        .current_dir(&project.root)
        .env("TMPDIR", &temporary);
    #[cfg(windows)]
    command.env("TMP", &temporary).env("TEMP", &temporary);
    let status = command.status().context("running deno, which exports the browser build")?;
    ensure!(status.success(), "the browser build did not export");
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
        assert_eq!(executable_target(b"#!/bin/sh"), None);
    }
}
