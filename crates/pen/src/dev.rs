//! Development reloads: watches the game's compiled program and its asset
//! directory (through `watch`, the session's one file watcher). A new
//! `game.wasm` produced after each Blot change compiles on a background
//! thread and then replaces the running build with its state carried over;
//! changed assets reload in the renderer and audio.
pub mod watch;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::time::{Duration, Instant};

use anyhow::Result;
use blot_host::GuestModule;
use notify::EventKind;
use pen_protocol::Catalog;
use pen_protocol::boundary::read_catalog;

use crate::devlog::{self, Source};

/// How long assets must be quiet before a reload: long enough to see an
/// editor's save through, short enough not to notice. game.wasm needs none:
/// it arrives in one rename.
const DEBOUNCE: Duration = Duration::from_millis(15);

pub enum Update {
    /// A new build, compiled, with the catalog it was built against.
    Build(GuestModule, Catalog),
    /// Asset keys whose files changed.
    Assets(Vec<String>),
}

pub struct Dev {
    _watch: watch::Subscription,
    events: Receiver<watch::Event>,
    wasm: PathBuf,
    assets: PathBuf,
    wasm_changed: bool,
    assets_changed: Option<Instant>,
    changed_keys: BTreeSet<String>,
    compiling: Option<Receiver<Result<(GuestModule, Catalog)>>>,
    /// Another change arrived while compiling.
    again: bool,
    trace: Option<std::sync::Arc<blot_host::SystemTrace>>,
}

/// An asset key for a changed file, or None for files that are not assets.
fn asset_key(assets: &Path, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(assets).ok()?;
    let name = relative.file_name()?.to_string_lossy();
    // Editors' temporary files, and the catalog, which the program imports:
    // a catalog change arrives as a new build.
    if name.starts_with('.') || name.ends_with('~') || name.ends_with(".tmp") || name == "catalog.json" {
        return None;
    }
    let parts: Vec<String> =
        relative.components().map(|part| part.as_os_str().to_string_lossy().into_owned()).collect();
    Some(parts.join("/"))
}

impl Dev {
    pub fn new(game: &Path, assets: &Path, trace: Option<std::sync::Arc<blot_host::SystemTrace>>) -> Result<Dev> {
        let (sender, events) = channel();
        let watch = watch::subscribe(&[(game, false), (assets, true)], move |event| {
            let _ = sender.send(event);
        })?;
        let game = std::fs::canonicalize(game)?;
        let assets = std::fs::canonicalize(assets)?;
        eprintln!("dev: reloading {} and {}", game.join("game.wasm").display(), assets.display());
        Ok(Dev {
            _watch: watch,
            events,
            wasm: game.join("game.wasm"),
            assets,
            wasm_changed: false,
            assets_changed: None,
            changed_keys: BTreeSet::new(),
            compiling: None,
            again: false,
            trace,
        })
    }

    fn compile(&mut self, engine: &wasmtime::Engine) {
        let (sender, result) = channel();
        let (engine, wasm, assets) = (engine.clone(), self.wasm.clone(), self.assets.clone());
        let started = Instant::now();
        let trace = self.trace.clone();
        let spawned = std::thread::Builder::new().name("pen-dev-compile".into()).spawn(move || {
            let built = (|| {
                let bytes = std::fs::read(&wasm)?;
                let catalog = read_catalog(&assets)?;
                // No cache: every edit would leave another compiled copy behind.
                let module = match trace {
                    Some(trace) => GuestModule::traced(&engine, &bytes, None, trace)?,
                    None => GuestModule::new(&engine, &bytes, None)?,
                };
                let milliseconds = started.elapsed().as_secs_f64() * 1000.0;
                devlog::info(Source::Reload, format!("dev: compiled the new build in {milliseconds:.0} ms"));
                Ok((module, catalog))
            })();
            let _ = sender.send(built);
        });
        match spawned {
            Ok(_) => self.compiling = Some(result),
            Err(error) => eprintln!("dev: cannot compile the new build: {error}"),
        }
    }

    /// What changed since the last call, once the files have settled.
    pub fn poll(&mut self, engine: &wasmtime::Engine) -> Vec<Update> {
        let now = Instant::now();
        loop {
            match self.events.try_recv() {
                Ok(Ok(event)) => {
                    if matches!(event.kind, EventKind::Access(_)) {
                        continue;
                    }
                    for path in &event.paths {
                        if path == &self.wasm {
                            self.wasm_changed = true;
                        } else if let Some(key) = asset_key(&self.assets, path) {
                            self.changed_keys.insert(key);
                            self.assets_changed = Some(now);
                        }
                    }
                }
                Ok(Err(error)) => eprintln!("dev: watching files: {error}"),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        let mut updates = Vec::new();
        if self.assets_changed.is_some_and(|at| now - at >= DEBOUNCE) {
            self.assets_changed = None;
            // Files gone again were editors' and tools' temporaries; a
            // deleted asset keeps its last version until it comes back. The
            // optional state declaration must also reload when removed, so its
            // cached adapter returns to opaque state handling.
            let keys: Vec<String> = std::mem::take(&mut self.changed_keys)
                .into_iter()
                .filter(|key| key == pen_runtime::state::CONFIG_FILE || self.assets.join(key).is_file())
                .collect();
            if !keys.is_empty() {
                devlog::info(Source::Asset, format!("dev: reloading {}", keys.join(", ")));
                updates.push(Update::Assets(keys));
            }
        }
        if let Some(compiling) = &self.compiling {
            match compiling.try_recv() {
                Ok(result) => {
                    self.compiling = None;
                    match result {
                        Ok((module, catalog)) => updates.push(Update::Build(module, catalog)),
                        Err(error) => {
                            devlog::error(Source::Reload, format!("dev: the new build does not load: {error:#}"))
                        }
                    }
                    if std::mem::take(&mut self.again) {
                        self.compile(engine);
                    }
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => self.compiling = None,
            }
        }
        if self.wasm_changed && self.wasm.exists() {
            self.wasm_changed = false;
            if self.compiling.is_some() {
                self.again = true;
            } else {
                self.compile(engine);
            }
        }
        updates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_keys_are_relative_and_skip_temporaries() {
        let root = Path::new("/game/assets");
        assert_eq!(asset_key(root, Path::new("/game/assets/rock.wgsl")).as_deref(), Some("rock.wgsl"));
        assert_eq!(asset_key(root, Path::new("/game/assets/a/b.png")).as_deref(), Some("a/b.png"));
        assert_eq!(asset_key(root, Path::new("/game/assets/.rock.wgsl.swp")), None);
        assert_eq!(asset_key(root, Path::new("/game/assets/catalog.json")), None);
        assert_eq!(asset_key(root, Path::new("/elsewhere/rock.wgsl")), None);
    }

    #[test]
    fn deleting_the_optional_state_declaration_still_delivers_a_reload() {
        let root = std::env::temp_dir().join(format!("pen-state-reload-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let mut dev = Dev::new(&root, &root, None).unwrap();
        // A settled watcher batch after both files were removed. No filesystem
        // timing is needed: this exercises the same delivery path as real edits.
        dev.changed_keys = BTreeSet::from([pen_runtime::state::CONFIG_FILE.into(), "removed.png".into()]);
        dev.assets_changed = Some(Instant::now() - DEBOUNCE);
        let updates = dev.poll(&wasmtime::Engine::default());
        assert_eq!(updates.len(), 1);
        let Update::Assets(keys) = &updates[0] else { panic!("expected asset reload") };
        assert_eq!(keys, &[pen_runtime::state::CONFIG_FILE]);
        drop(dev);
        std::fs::remove_dir_all(root).unwrap();
    }
}
