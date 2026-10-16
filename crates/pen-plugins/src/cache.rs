//! What plugins keep in pen's cache directory (`pen/` in the platform's),
//! and how it is kept from growing without end:
//!
//! - `components/`: each plugin component compiled for this pen, by the
//!   SHA-256 of its bytes and of what compiled code must match (wasmtime's
//!   version and configuration).
//! - `programs/`: the programs plugins run, unpacked (`blotc.gz`), one
//!   directory each.
//! - `plugins/NAME/`: each plugin's own directory, which the programs it runs
//!   keep their caches in (blotc's restart cache).
//!
//! Using a component or a program marks it used. Once per process, the
//! first time a checkout's plugins load, components and programs go when
//! unused for [`UNUSED`] or past the [`KEEP`] used last, and the files of a
//! plugin's directory when unchanged for [`UNUSED`] or, oldest first, while
//! it holds over [`PLUGIN_BYTES`]. Each goes at worst to a slower start: a
//! component compiles again, a program unpacks again, and a compiler
//! starts without its restart cache.
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use anyhow::Result;
use sha2::{Digest, Sha256};
use wasmtime::Engine;
use wasmtime::component::Component;

/// How long a cache entry may go unused.
pub const UNUSED: Duration = Duration::from_secs(14 * 24 * 60 * 60);
/// How many components, and how many programs, are kept: those used last.
pub const KEEP: usize = 4;
/// The most a plugin's own directory holds.
pub const PLUGIN_BYTES: u64 = 2 << 30;

/// A [`Hasher`] that feeds a SHA-256 digest what it is given.
struct Feed<'a>(&'a mut Sha256);

impl Hasher for Feed<'_> {
    fn write(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    fn finish(&self) -> u64 {
        0
    }
}

/// The component `bytes` are, compiled for `engine`: from pen's cache when
/// an earlier launch compiled it, else compiled now and kept there.
pub fn component(engine: &Engine, bytes: &[u8]) -> Result<Component> {
    let mut digest = Sha256::new();
    digest.update(bytes);
    engine.precompile_compatibility_hash().hash(&mut Feed(&mut digest));
    let name = format!("{:x}", digest.finalize());
    let path = crate::cache_root().join("components").join(format!("{}.cwasm", &name[..32]));
    if path.is_file() {
        // SAFETY: the file is one this function wrote, from these bytes, for
        // an engine of this wasmtime and configuration; wasmtime still checks
        // its header.
        match unsafe { Component::deserialize_file(engine, &path) } {
            Ok(component) => {
                used(&path);
                return Ok(component);
            }
            Err(_) => {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
    let component = Component::from_binary(engine, bytes)?;
    // A cache that cannot be written only costs the next launch a compile.
    let _ = (|| -> Result<()> {
        std::fs::create_dir_all(path.parent().expect("in components/"))?;
        let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&temporary, component.serialize()?)?;
        std::fs::rename(&temporary, &path)?;
        Ok(())
    })();
    Ok(component)
}

/// Marks the file at `path` used now.
pub(crate) fn used(path: &Path) {
    if let Ok(file) = std::fs::File::open(path) {
        let _ = file.set_modified(SystemTime::now());
    }
}

/// Prunes pen's cache, once per process.
pub(crate) fn prune_once() {
    static PRUNED: std::sync::Once = std::sync::Once::new();
    PRUNED.call_once(|| prune(&crate::cache_root(), SystemTime::now()));
}

/// Prunes the plugins' part of the cache at `root`, as of `now`.
pub fn prune(root: &Path, now: SystemTime) {
    prune_entries(&root.join("components"), now);
    prune_entries(&root.join("programs"), now);
    for plugin in entries(&root.join("plugins")) {
        if plugin.is_dir() {
            prune_files(&plugin, now, PLUGIN_BYTES);
        }
    }
}

/// The entries of `directory`; none when it cannot be read.
fn entries(directory: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(directory) else { return Vec::new() };
    entries.flatten().map(|entry| entry.path()).collect()
}

/// When `path` (a file, or a directory and what is in it) last changed.
fn changed(path: &Path) -> SystemTime {
    let own =
        std::fs::symlink_metadata(path).and_then(|metadata| metadata.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
    if !path.is_dir() || path.is_symlink() {
        return own;
    }
    entries(path).iter().map(|entry| changed(entry)).fold(own, SystemTime::max)
}

/// Removes the entries of `directory` (files or directories) unused for
/// [`UNUSED`] or past the [`KEEP`] used last; never one in the making.
fn prune_entries(directory: &Path, now: SystemTime) {
    let mut found: Vec<(SystemTime, PathBuf)> = entries(directory)
        .into_iter()
        .filter(|path| !path.file_name().is_some_and(|name| name.to_string_lossy().contains(".tmp")))
        .map(|path| (changed(&path), path))
        .collect();
    found.sort_by_key(|(at, _)| std::cmp::Reverse(*at));
    for (index, (at, path)) in found.into_iter().enumerate() {
        if index >= KEEP || stale(at, now) {
            let _ = if path.is_dir() { std::fs::remove_dir_all(&path) } else { std::fs::remove_file(&path) };
        }
    }
}

/// Removes the files under `directory` unchanged for [`UNUSED`], then, oldest
/// first, while the rest hold over `limit` bytes; then the directories under
/// it left empty.
fn prune_files(directory: &Path, now: SystemTime, limit: u64) {
    let mut files = Vec::new();
    walk(directory, &mut files);
    files.sort_by_key(|(at, _, _)| std::cmp::Reverse(*at));
    let mut kept = 0u64;
    for (at, size, path) in files {
        if stale(at, now) || kept + size > limit {
            let _ = std::fs::remove_file(&path);
        } else {
            kept += size;
        }
    }
    remove_empty(directory);
}

/// The files under `directory`: when each changed, its size and its path.
fn walk(directory: &Path, files: &mut Vec<(SystemTime, u64, PathBuf)>) {
    for path in entries(directory) {
        let Ok(metadata) = std::fs::symlink_metadata(&path) else { continue };
        if metadata.is_dir() {
            walk(&path, files);
        } else {
            files.push((metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH), metadata.len(), path));
        }
    }
}

/// Removes the directories under `directory` that are (or are left) empty.
fn remove_empty(directory: &Path) {
    for path in entries(directory) {
        if std::fs::symlink_metadata(&path).is_ok_and(|metadata| metadata.is_dir()) {
            remove_empty(&path);
            let _ = std::fs::remove_dir(&path);
        }
    }
}

/// Whether what changed `at` is unused for [`UNUSED`] by `now`.
fn stale(at: SystemTime, now: SystemTime) -> bool {
    now.duration_since(at).is_ok_and(|age| age > UNUSED)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, bytes: usize, age: Duration, now: SystemTime) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, vec![0u8; bytes]).unwrap();
        std::fs::File::options().write(true).open(path).unwrap().set_modified(now - age).unwrap();
    }

    fn set_changed(path: &Path, age: Duration, now: SystemTime) {
        std::fs::File::open(path).unwrap().set_modified(now - age).unwrap();
    }

    fn day(days: u64) -> Duration {
        Duration::from_secs(days * 24 * 60 * 60)
    }

    #[test]
    fn prunes_what_went_unused() {
        let root = crate::cache_root().join(format!("tmp/pen-plugins-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let now = SystemTime::now();
        // Six programs, used one to six days ago, and one a month ago: the
        // four used last stay.
        for (index, days) in [1, 2, 3, 4, 5, 6, 30].into_iter().enumerate() {
            let program = root.join(format!("programs/{index:016x}"));
            write(&program.join("blotc"), 8, day(days), now);
            set_changed(&program, day(days), now);
        }
        // A program unpacking now.
        write(&root.join("programs/0123456789abcdef/.unpacking.1"), 8, day(0), now);
        set_changed(&root.join("programs/0123456789abcdef"), day(40), now);
        // Components: one fresh, one a month old, one being written.
        write(&root.join("components/fresh.cwasm"), 8, day(1), now);
        write(&root.join("components/old.cwasm"), 8, day(30), now);
        write(&root.join("components/new.cwasm.7.tmp"), 8, day(30), now);
        // A plugin's directory: a month-old file, and fresh ones.
        let plugin = root.join("plugins/blot");
        write(&plugin.join("v1/old/a.blotcache"), 8, day(30), now);
        write(&plugin.join("v1/new/b.blotcache"), 16, day(1), now);
        write(&plugin.join("v1/new/c.blotcache"), 16, day(2), now);
        write(&plugin.join("v1/new/d.blotcache"), 16, day(3), now);
        prune(&root, now);
        let mut programs: Vec<String> = entries(&root.join("programs"))
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into())
            .collect();
        programs.sort();
        assert_eq!(programs, ["0000000000000000", "0000000000000001", "0000000000000002", "0123456789abcdef"]);
        assert!(root.join("components/fresh.cwasm").is_file());
        assert!(!root.join("components/old.cwasm").exists());
        assert!(root.join("components/new.cwasm.7.tmp").is_file());
        assert!(!plugin.join("v1/old").exists(), "an emptied directory goes");
        let left = |name: &str| plugin.join("v1/new").join(name).is_file();
        assert!(left("b.blotcache") && left("c.blotcache") && left("d.blotcache"));
        // Over its limit, the oldest go until the rest fit.
        prune_files(&plugin, now, 40);
        assert!(left("b.blotcache") && left("c.blotcache") && !left("d.blotcache"));
        prune_files(&plugin, now, 8);
        assert!(!plugin.join("v1").exists());
        assert!(plugin.is_dir(), "the plugin's own directory stays");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
