//! A content-addressed store: each file kept under the SHA-256 of its bytes,
//! written once and never changed, so one store can serve several
//! checkouts, CI runs and machines, and be copied or synced as plain files.
//! The asset pipeline caches its steps' outputs in one (asset_graph.rs), and
//! `pen build` keeps each package version in one, between which patches are
//! made (patch.rs).
//!
//! `objects/ab/abcdef…` holds the blob whose digest is `abcdef…`;
//! `records/KIND/NAME.json` holds small named documents that point at blobs
//! by their digests (a step's outputs, a package version's files). Writes
//! go through `tmp/` and are renamed into place, so a reader never sees half
//! a file, and two writers of the same blob write the same bytes.
#[cfg(any(feature = "compiler", test))]
use std::io::{Read, Write};
use std::path::Path;
#[cfg(any(feature = "compiler", test))]
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(any(feature = "compiler", test))]
use std::time::Duration;
use std::time::SystemTime;

#[cfg(any(feature = "compiler", test))]
use anyhow::bail;
use anyhow::{Context, Result, ensure};
#[cfg(any(feature = "compiler", test))]
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest as _, Sha256};

/// A SHA-256 digest, written as 64 lowercase hex digits.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    pub fn of(bytes: &[u8]) -> Digest {
        Digest(Sha256::digest(bytes).into())
    }

    pub fn parse(text: &str) -> Result<Digest> {
        ensure!(
            text.len() == 64 && text.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
            "`{text}` is not a SHA-256 digest"
        );
        let mut digest = [0; 32];
        for (index, byte) in digest.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16)?;
        }
        Ok(Digest(digest))
    }
}

impl std::fmt::Display for Digest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.0.iter().try_for_each(|byte| write!(formatter, "{byte:02x}"))
    }
}

impl std::fmt::Debug for Digest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "{self}")
    }
}

impl Serialize for Digest {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Digest, D::Error> {
        let text = String::deserialize(deserializer)?;
        Digest::parse(&text).map_err(serde::de::Error::custom)
    }
}

/// The digest and size of a file's content, read in pieces.
#[cfg(any(feature = "compiler", test))]
pub fn digest_file(path: &Path) -> Result<(Digest, u64)> {
    let file = std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
    hash_into(file, std::io::sink()).with_context(|| format!("reading {}", path.display()))
}

/// Copies `from` into `to`, hashing what passes.
#[cfg(any(feature = "compiler", test))]
fn hash_into(mut from: impl Read, mut to: impl Write) -> std::io::Result<(Digest, u64)> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 1 << 16];
    let mut size = 0;
    loop {
        let read = match from.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        hasher.update(&buffer[..read]);
        to.write_all(&buffer[..read])?;
        size += read as u64;
    }
    to.flush()?;
    Ok((Digest(hasher.finalize().into()), size))
}

/// A name no other writer uses, for a file written before it is renamed.
fn unique(prefix: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |time| time.as_nanos());
    format!("{prefix}{}-{}-{nanos}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed))
}

/// Renames `from` to `to`; another writer having put the same blob there
/// first (Windows will not rename over it) is as good.
#[cfg(any(feature = "compiler", test))]
fn rename_into(from: &Path, to: &Path) -> Result<()> {
    match std::fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(_) if to.is_file() => {
            let _ = std::fs::remove_file(from);
            Ok(())
        }
        Err(error) => Err(error).with_context(|| format!("writing {}", to.display())),
    }
}

/// Writes `to` whole or not at all: into a file beside it, then renamed.
pub fn write_atomically(to: &Path, write: impl FnOnce(&mut std::fs::File) -> Result<()>) -> Result<()> {
    let parent = to.parent().context("a file to write has no directory")?;
    std::fs::create_dir_all(parent)?;
    let name = to.file_name().context("a file to write has no name")?.to_string_lossy();
    let temporary = parent.join(unique(&format!(".{name}.pen-")));
    let written = std::fs::File::create(&temporary)
        .with_context(|| format!("writing {}", temporary.display()))
        .and_then(|mut file| write(&mut file).and_then(|()| Ok(file.sync_all()?)));
    match written.and_then(|()| Ok(std::fs::rename(&temporary, to)?)) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = std::fs::remove_file(&temporary);
            Err(error).with_context(|| format!("writing {}", to.display()))
        }
    }
}

#[cfg(any(feature = "compiler", test))]
pub struct Store {
    root: PathBuf,
}

#[cfg(any(feature = "compiler", test))]
impl Store {
    /// The store in `root`, made if it is not there.
    pub fn open(root: &Path) -> Result<Store> {
        for directory in ["objects", "records", "tmp"] {
            std::fs::create_dir_all(root.join(directory))
                .with_context(|| format!("making the store {}", root.display()))?;
        }
        Ok(Store { root: root.to_owned() })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Where the blob of `digest` is, if the store has it.
    pub fn object(&self, digest: &Digest) -> PathBuf {
        let hex = digest.to_string();
        self.root.join("objects").join(&hex[..2]).join(hex)
    }

    pub fn has(&self, digest: &Digest) -> bool {
        self.object(digest).is_file()
    }

    fn temporary(&self) -> PathBuf {
        self.root.join("tmp").join(unique(""))
    }

    /// Keeps a copy of the file at `path`, returning its digest and size.
    pub fn put_file(&self, path: &Path) -> Result<(Digest, u64)> {
        let source = std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
        let temporary = self.temporary();
        let target = std::fs::File::create(&temporary).with_context(|| format!("writing {}", temporary.display()))?;
        let copied = hash_into(source, std::io::BufWriter::new(target));
        let (digest, size) = match copied {
            Ok(copied) => copied,
            Err(error) => {
                let _ = std::fs::remove_file(&temporary);
                return Err(error).with_context(|| format!("keeping {}", path.display()));
            }
        };
        if self.has(&digest) {
            std::fs::remove_file(&temporary)?;
        } else {
            self.settle(&temporary, &digest)?;
        }
        Ok((digest, size))
    }

    fn settle(&self, temporary: &Path, digest: &Digest) -> Result<()> {
        let object = self.object(digest);
        std::fs::create_dir_all(object.parent().unwrap())?;
        rename_into(temporary, &object)
    }

    /// The blob of `digest`, checked against it.
    pub fn read(&self, digest: &Digest) -> Result<Vec<u8>> {
        let path = self.object(digest);
        let bytes = std::fs::read(&path).with_context(|| format!("the store has no {digest}"))?;
        if Digest::of(&bytes) != *digest {
            let _ = std::fs::remove_file(&path);
            bail!("the store's {digest} was damaged; it has been removed");
        }
        Ok(bytes)
    }

    /// Writes the blob of `digest` to `to`, whole or not at all, checking it
    /// on the way.
    pub fn copy_out(&self, digest: &Digest, to: &Path) -> Result<()> {
        let path = self.object(digest);
        let source = std::fs::File::open(&path).with_context(|| format!("the store has no {digest}"))?;
        let mut damaged = false;
        write_atomically(to, |file| {
            let (copied, _) = hash_into(source, std::io::BufWriter::new(file))?;
            damaged = copied != *digest;
            ensure!(!damaged, "the store's {digest} was damaged; it has been removed");
            Ok(())
        })
        .inspect_err(|_| {
            if damaged {
                let _ = std::fs::remove_file(&path);
            }
        })
    }

    fn record_path(&self, kind: &str, name: &str) -> Result<PathBuf> {
        ensure!(
            !name.is_empty()
                && name.chars().all(|character| character.is_ascii_alphanumeric() || "-_.".contains(character)),
            "`{name}` cannot name a record"
        );
        Ok(self.root.join("records").join(kind).join(format!("{name}.json")))
    }

    /// Writes the record `name` of `kind`, replacing any before it.
    pub fn write_record(&self, kind: &str, name: &str, value: &impl Serialize) -> Result<()> {
        let text = serde_json::to_vec_pretty(value)?;
        write_atomically(&self.record_path(kind, name)?, |file| Ok(file.write_all(&text)?))
    }

    /// The record `name` of `kind`, if there is one; reading it counts as
    /// using it, which keeps it from being pruned.
    pub fn read_record<T: DeserializeOwned>(&self, kind: &str, name: &str) -> Result<Option<T>> {
        let path = self.record_path(kind, name)?;
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
        };
        if let Ok(file) = std::fs::File::options().append(true).open(&path) {
            let _ = file.set_modified(SystemTime::now());
        }
        Ok(Some(serde_json::from_slice(&bytes).with_context(|| format!("reading {}", path.display()))?))
    }

    /// The names of the records of `kind`, sorted.
    pub fn records(&self, kind: &str) -> Result<Vec<String>> {
        let directory = self.root.join("records").join(kind);
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&directory).into_iter().flatten().flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(name) = name.strip_suffix(".json") {
                names.push(name.to_owned());
            }
        }
        names.sort();
        Ok(names)
    }

    /// Deletes the records of `kind` unused for `age`, then the oldest
    /// others while the store holds more than `bytes`, then every blob no
    /// record of any kind names, and what writers left in `tmp/` a day ago.
    /// Says what it freed.
    pub fn prune(&self, kind: &str, age: Duration, bytes: u64) -> Result<u64> {
        let now = SystemTime::now();
        let used = |path: &Path| std::fs::metadata(path).and_then(|metadata| metadata.modified()).unwrap_or(now);
        let directory = self.root.join("records").join(kind);
        let mut records: Vec<(SystemTime, PathBuf)> = std::fs::read_dir(&directory)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .map(|path| (used(&path), path))
            .collect();
        records.sort();
        let stale = |time: SystemTime| now.duration_since(time).unwrap_or_default() > age;
        let mut kept = Vec::new();
        for (time, path) in records {
            if stale(time) {
                std::fs::remove_file(&path)?;
            } else {
                kept.push(path);
            }
        }
        let objects = self.root.join("objects");
        let blobs = || -> Vec<(PathBuf, u64)> {
            let mut blobs = Vec::new();
            for shard in std::fs::read_dir(&objects).into_iter().flatten().flatten() {
                for entry in std::fs::read_dir(shard.path()).into_iter().flatten().flatten() {
                    let size = entry.metadata().map_or(0, |metadata| metadata.len());
                    blobs.push((entry.path(), size));
                }
            }
            blobs
        };
        let mut freed = 0;
        // The oldest of this kind go first while the blobs are too many.
        let mut total: u64 = blobs().iter().map(|(_, size)| size).sum();
        while total > bytes && kept.len() > 1 {
            std::fs::remove_file(kept.remove(0))?;
            freed += self.sweep(&blobs())?;
            total = blobs().iter().map(|(_, size)| size).sum();
        }
        freed += self.sweep(&blobs())?;
        for entry in std::fs::read_dir(self.root.join("tmp")).into_iter().flatten().flatten() {
            if now.duration_since(used(&entry.path())).unwrap_or_default() > Duration::from_secs(86_400) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
        Ok(freed)
    }

    /// Deletes the blobs no record names; what that freed.
    fn sweep(&self, blobs: &[(PathBuf, u64)]) -> Result<u64> {
        let mut named = std::collections::HashSet::new();
        for kind in std::fs::read_dir(self.root.join("records")).into_iter().flatten().flatten() {
            for record in std::fs::read_dir(kind.path()).into_iter().flatten().flatten() {
                let Ok(bytes) = std::fs::read(record.path()) else { continue };
                let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else { continue };
                digests(&value, &mut named);
            }
        }
        let mut freed = 0;
        for (path, size) in blobs {
            let name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
            if !named.contains(&name) {
                std::fs::remove_file(path)?;
                freed += size;
            }
        }
        Ok(freed)
    }
}

/// Every digest a record names: its strings of 64 hex digits.
#[cfg(any(feature = "compiler", test))]
fn digests(value: &serde_json::Value, found: &mut std::collections::HashSet<String>) {
    match value {
        serde_json::Value::String(text) if Digest::parse(text).is_ok() => {
            found.insert(text.clone());
        }
        serde_json::Value::Array(values) => values.iter().for_each(|value| digests(value, found)),
        serde_json::Value::Object(fields) => fields.values().for_each(|value| digests(value, found)),
        _ => {}
    }
}

#[cfg(test)]
pub(crate) fn scratch(name: &str) -> PathBuf {
    let directory = crate::project::temporary_dir().unwrap().join(unique(&format!("{name}-")));
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Keeps `bytes`, as a file of them.
    fn put(store: &Store, bytes: &[u8]) -> Digest {
        let file = store.temporary().with_extension("source");
        std::fs::write(&file, bytes).unwrap();
        let (digest, _) = store.put_file(&file).unwrap();
        std::fs::remove_file(file).unwrap();
        digest
    }

    #[test]
    fn blobs_are_kept_by_their_digest_and_checked() {
        let root = scratch("store");
        let store = Store::open(&root.join("store")).unwrap();
        let digest = put(&store, b"salt");
        assert_eq!(digest.to_string(), "63479ad69a090b258277ec8fba6f99419a2ffb248981510657c944ccd1148e97");
        assert_eq!(Digest::parse(&digest.to_string()).unwrap(), digest);
        assert!(Digest::parse("ABC").is_err() && Digest::parse(&"G".repeat(64)).is_err());
        assert_eq!(store.read(&digest).unwrap(), b"salt");
        // The same bytes from a file are the same blob.
        std::fs::write(root.join("salt.txt"), "salt").unwrap();
        assert_eq!(store.put_file(&root.join("salt.txt")).unwrap(), (digest, 4));
        assert_eq!(digest_file(&root.join("salt.txt")).unwrap(), (digest, 4));
        store.copy_out(&digest, &root.join("out/again.txt")).unwrap();
        assert_eq!(std::fs::read(root.join("out/again.txt")).unwrap(), b"salt");
        // A damaged blob is refused and removed, and nothing is written.
        std::fs::write(store.object(&digest), "pepper").unwrap();
        assert!(store.copy_out(&digest, &root.join("out/damaged.txt")).is_err());
        assert!(!root.join("out/damaged.txt").exists() && !store.has(&digest));
        assert!(std::fs::read_dir(root.join("out")).unwrap().count() == 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn records_keep_their_blobs_until_pruned() {
        let root = scratch("store-prune");
        let store = Store::open(&root).unwrap();
        let kept = put(&store, b"kept");
        let loose = put(&store, b"loose");
        store.write_record("steps", "a", &serde_json::json!({ "outputs": [{ "path": "a", "digest": kept }] })).unwrap();
        let record: serde_json::Value = store.read_record("steps", "a").unwrap().unwrap();
        assert_eq!(record["outputs"][0]["digest"], kept.to_string());
        assert!(store.read_record::<serde_json::Value>("steps", "b").unwrap().is_none());
        assert!(store.write_record("steps", "../escape", &1).is_err());
        assert_eq!(store.records("steps").unwrap(), ["a"]);
        // A blob no record names goes; one a record names stays.
        assert_eq!(store.prune("steps", Duration::from_secs(3600), u64::MAX).unwrap(), 5);
        assert!(store.has(&kept) && !store.has(&loose));
        // Over the size, the oldest records go first, but never the last.
        store.write_record("steps", "b", &serde_json::json!({ "digest": put(&store, b"newer") })).unwrap();
        let old = SystemTime::now() - Duration::from_secs(60);
        std::fs::File::options()
            .append(true)
            .open(root.join("records/steps/a.json"))
            .unwrap()
            .set_modified(old)
            .unwrap();
        store.prune("steps", Duration::from_secs(3600), 5).unwrap();
        assert_eq!(store.records("steps").unwrap(), ["b"]);
        assert!(!store.has(&kept));
        // Unused for longer than the age, a record goes whatever the size.
        store.prune("steps", Duration::ZERO, u64::MAX).unwrap();
        assert!(store.records("steps").unwrap().is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }
}
