//! Package versions and the patches between them.
//!
//! `pen build` writes `game/files.json` into each package: every file of it
//! (the runtime, its libraries, the game) by path, with its size and
//! SHA-256, and the package's version, the digest of that list. It keeps
//! the package in a content-addressed store (store.rs, the checkout's
//! `.pen/store` unless `--store DIR`): each file once however many versions
//! hold it, and each version's list as a record, so any version it made
//! can be patched from later.
//!
//! `pen patch OLD NEW --out FILE` makes a patch from one version to the
//! other (package folders, or versions in the store): for each file of the
//! new version, nothing when the old one has it at the same path, a copy
//! when it has the same bytes elsewhere, else a binary delta against the
//! old file at its path (delta.rs) when that is smaller than the file, else
//! the file; each deflated. `pen apply FILE [DIR]` applies one to the
//! package in DIR (by default the one beside the executable, so a shipped
//! game can patch itself): it checks that the package is the patch's old
//! version, builds the new one beside it, checks every file of that, and
//! only then swaps it in.
//!
//! A patch file: `PENPATCH`, a u32 format version, the length of its JSON
//! header (u64) and the header, then the payloads the header points into.
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::store::{Digest, write_atomically};
#[cfg(any(feature = "compiler", test))]
use crate::store::{Store, digest_file};

/// Where a package's file list is, from its root.
pub const MANIFEST: &str = "game/files.json";
const MAGIC: &[u8; 8] = b"PENPATCH";
const FORMAT: u32 = 1;
/// The store's records of package versions.
#[cfg(any(feature = "compiler", test))]
const VERSIONS: &str = "packages";

/// One file of a package.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Entry {
    /// From the package's root, `/` between names.
    pub path: String,
    pub digest: Digest,
    pub size: u64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub executable: bool,
}

/// A package's files, `game/files.json`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Manifest {
    pub game: String,
    /// The digest of its files' list: the same files, the same version.
    pub version: Digest,
    pub files: Vec<Entry>,
}

impl Manifest {
    /// The version of `files` (sorted by path).
    fn version_of(files: &[Entry]) -> Digest {
        let mut text = String::from("pen package 1\n");
        for entry in files {
            text += &format!("{} {} {} {}\n", entry.path, entry.digest, entry.size, u8::from(entry.executable));
        }
        Digest::of(text.as_bytes())
    }

    /// The manifest of the package at `root` as its files are now.
    #[cfg(any(feature = "compiler", test))]
    pub fn of_directory(root: &Path, game: &str) -> Result<Manifest> {
        let mut files = Vec::new();
        for path in package_files(root)? {
            let full = root.join(&path);
            let (digest, size) = digest_file(&full)?;
            files.push(Entry { path, digest, size, executable: executable(&full) });
        }
        Ok(Manifest { game: game.to_owned(), version: Manifest::version_of(&files), files })
    }

    pub fn read(root: &Path) -> Result<Manifest> {
        let path = root.join(MANIFEST);
        let bytes = std::fs::read(&path).with_context(|| format!("{} is not a package pen built", root.display()))?;
        let manifest: Manifest =
            serde_json::from_slice(&bytes).with_context(|| format!("reading {}", path.display()))?;
        ensure!(
            Manifest::version_of(&manifest.files) == manifest.version,
            "{}: its version does not match its files",
            path.display()
        );
        Ok(manifest)
    }

    pub fn write(&self, root: &Path) -> Result<()> {
        let text = serde_json::to_vec_pretty(self)?;
        write_atomically(&root.join(MANIFEST), |file| Ok(file.write_all(&text)?))
    }

    #[cfg(any(feature = "compiler", test))]
    fn total(&self) -> u64 {
        self.files.iter().map(|entry| entry.size).sum()
    }
}

/// Every file under `root` but the manifest, relative and sorted.
#[cfg(any(feature = "compiler", test))]
fn package_files(root: &Path) -> Result<Vec<String>> {
    fn walk(root: &Path, directory: &Path, files: &mut Vec<String>) -> Result<()> {
        for entry in std::fs::read_dir(directory).with_context(|| format!("reading {}", directory.display()))? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                walk(root, &path, files)?;
            } else {
                let parts: Vec<String> = path
                    .strip_prefix(root)?
                    .components()
                    .map(|part| part.as_os_str().to_string_lossy().into_owned())
                    .collect();
                files.push(parts.join("/"));
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(root, root, &mut files)?;
    files.retain(|path| path != MANIFEST);
    files.sort();
    Ok(files)
}

#[cfg(unix)]
#[cfg(any(feature = "compiler", test))]
fn executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).is_ok_and(|metadata| metadata.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
#[cfg(any(feature = "compiler", test))]
fn executable(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension == "exe")
}

#[cfg(unix)]
fn set_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(path)?.permissions();
    permissions.set_mode(permissions.mode() | 0o755);
    Ok(std::fs::set_permissions(path, permissions)?)
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> Result<()> {
    Ok(())
}

/// A path from a manifest, which must stay inside the package.
fn checked_path(path: &str) -> Result<&str> {
    ensure!(
        !path.is_empty()
            && !path.starts_with('/')
            && !path.contains('\\')
            && !path.contains(':')
            && path.split('/').all(|part| !part.is_empty() && part != "." && part != ".."),
        "`{path}` is not a path inside a package"
    );
    Ok(path)
}

/// Writes the package's manifest and keeps the package in the store, saying
/// so; returns its version.
#[cfg(any(feature = "compiler", test))]
pub fn record(root: &Path, game: &str, store: &Path) -> Result<Digest> {
    let manifest = Manifest::of_directory(root, game)?;
    manifest.write(root)?;
    let store = Store::open(store)?;
    for entry in &manifest.files {
        if !store.has(&entry.digest) {
            store.put_file(&root.join(&entry.path))?;
        }
    }
    store.write_record(VERSIONS, &manifest.version.to_string(), &manifest)?;
    Ok(manifest.version)
}

/// A version to patch from or to: a package folder, or a version in the
/// store (its digest, or the start of it).
#[cfg(any(feature = "compiler", test))]
enum Version {
    Directory(PathBuf, Manifest),
    Stored(Manifest),
}

#[cfg(any(feature = "compiler", test))]
impl Version {
    fn find(name: &Path, store: Option<&Store>) -> Result<Version> {
        if name.is_dir() {
            return Ok(Version::Directory(name.to_owned(), Manifest::read(name)?));
        }
        let text = name.to_string_lossy();
        let store = store.with_context(|| format!("{text} is not a package folder"))?;
        let found: Vec<String> =
            store.records(VERSIONS)?.into_iter().filter(|version| version.starts_with(&*text)).collect();
        match found.as_slice() {
            [version] => {
                Ok(Version::Stored(store.read_record(VERSIONS, version)?.context("a stored version went missing")?))
            }
            [] => bail!("{text} is neither a package folder nor a version in {}", store.root().display()),
            _ => bail!("{text} is the start of several versions in {}", store.root().display()),
        }
    }

    fn manifest(&self) -> &Manifest {
        match self {
            Version::Directory(_, manifest) | Version::Stored(manifest) => manifest,
        }
    }

    /// The bytes of one of its files.
    fn read(&self, entry: &Entry, store: Option<&Store>) -> Result<Vec<u8>> {
        match self {
            Version::Directory(root, _) => {
                let bytes = std::fs::read(root.join(checked_path(&entry.path)?))?;
                ensure!(Digest::of(&bytes) == entry.digest, "{} is not what the package's manifest says", entry.path);
                Ok(bytes)
            }
            Version::Stored(_) => store.context("no store")?.read(&entry.digest),
        }
    }
}

/// How the patch makes one file of the new version.
#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase", tag = "how")]
enum Make {
    /// The old version has it at the same path.
    Keep { path: String },
    /// The old version has the same bytes at `from`.
    Copy { path: String, from: String },
    /// The old file at `base`, changed by the delta in the payload.
    Delta { path: String, base: String, at: u64, length: u64 },
    /// The payload, deflated.
    Whole { path: String, at: u64, length: u64 },
}

#[derive(Debug, Deserialize, Serialize)]
struct Header {
    from: Digest,
    to: Manifest,
    files: Vec<Make>,
}

#[cfg(any(feature = "compiler", test))]
fn deflate(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
    encoder.write_all(bytes)?;
    Ok(encoder.finish()?)
}

fn inflate(bytes: &[u8], limit: u64) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    flate2::read::ZlibDecoder::new(bytes).take(limit + 1).read_to_end(&mut out)?;
    ensure!(out.len() as u64 <= limit, "a patch's payload inflates past its file's size");
    Ok(out)
}

/// What a patch holds, for its summary.
#[cfg(any(feature = "compiler", test))]
pub struct Made {
    pub kept: usize,
    pub copied: usize,
    pub deltas: usize,
    pub whole: usize,
    pub removed: usize,
    pub bytes: u64,
    pub of: u64,
}

#[cfg(any(feature = "compiler", test))]
impl std::fmt::Display for Made {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            formatter,
            "{:.2} MB for a {:.1} MB package: {} files unchanged, {} moved, {} sent as deltas, {} sent whole, {} removed",
            self.bytes as f64 / 1e6,
            self.of as f64 / 1e6,
            self.kept,
            self.copied,
            self.deltas,
            self.whole,
            self.removed,
        )
    }
}

/// Writes the patch from version `old` to version `new` to `out`.
#[cfg(any(feature = "compiler", test))]
pub fn make(old: &Path, new: &Path, store: Option<&Path>, out: &Path) -> Result<Made> {
    let store = store.filter(|store| store.is_dir()).map(Store::open).transpose()?;
    let old = Version::find(old, store.as_ref())?;
    let new = Version::find(new, store.as_ref())?;
    let (before, after) = (old.manifest(), new.manifest());
    ensure!(before.game == after.game, "the versions are of different games: {} and {}", before.game, after.game);
    let by_path: HashMap<&str, &Entry> = before.files.iter().map(|entry| (entry.path.as_str(), entry)).collect();
    let by_digest: HashMap<&Digest, &Entry> = before.files.iter().map(|entry| (&entry.digest, entry)).collect();
    let mut files = Vec::new();
    let mut payload = Vec::new();
    let mut made = Made { kept: 0, copied: 0, deltas: 0, whole: 0, removed: 0, bytes: 0, of: after.total() };
    for entry in &after.files {
        let path = entry.path.clone();
        if by_path.get(entry.path.as_str()).is_some_and(|old| old.digest == entry.digest) {
            files.push(Make::Keep { path });
            made.kept += 1;
            continue;
        }
        if let Some(same) = by_digest.get(&entry.digest) {
            files.push(Make::Copy { path, from: same.path.clone() });
            made.copied += 1;
            continue;
        }
        let bytes = new.read(entry, store.as_ref())?;
        let whole = deflate(&bytes)?;
        let delta = match by_path.get(entry.path.as_str()) {
            Some(base) => {
                let delta = deflate(&crate::delta::diff(&old.read(base, store.as_ref())?, &bytes))?;
                (delta.len() < whole.len()).then_some((base.path.clone(), delta))
            }
            None => None,
        };
        let at = payload.len() as u64;
        match delta {
            Some((base, delta)) => {
                files.push(Make::Delta { path, base, at, length: delta.len() as u64 });
                payload.extend(delta);
                made.deltas += 1;
            }
            None => {
                files.push(Make::Whole { path, at, length: whole.len() as u64 });
                payload.extend(whole);
                made.whole += 1;
            }
        }
    }
    let kept: std::collections::HashSet<&str> = after.files.iter().map(|entry| entry.path.as_str()).collect();
    made.removed = before.files.iter().filter(|entry| !kept.contains(entry.path.as_str())).count();
    let header = serde_json::to_vec(&Header { from: before.version, to: after.clone(), files })?;
    write_atomically(out, |file| {
        let mut file = std::io::BufWriter::new(file);
        file.write_all(MAGIC)?;
        file.write_all(&FORMAT.to_le_bytes())?;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(&header)?;
        file.write_all(&payload)?;
        Ok(file.flush()?)
    })?;
    made.bytes = std::fs::metadata(out)?.len();
    Ok(made)
}

/// Reads a patch's header, and where its payloads start.
fn read_header(bytes: &[u8]) -> Result<(Header, usize)> {
    ensure!(bytes.len() >= 20 && &bytes[..8] == MAGIC, "not a pen patch");
    let format = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    ensure!(format == FORMAT, "a patch of format {format}, which this pen does not read");
    let length = usize::try_from(u64::from_le_bytes(bytes[12..20].try_into().unwrap()))?;
    ensure!(bytes.len() - 20 >= length, "the patch ends inside its header");
    let header: Header = serde_json::from_slice(&bytes[20..20 + length]).context("the patch's header")?;
    ensure!(
        Manifest::version_of(&header.to.files) == header.to.version,
        "the patch's new version does not match its files"
    );
    Ok((header, 20 + length))
}

/// Applies the patch in `file` to the package at `root`; returns the
/// version it is now.
pub fn apply(file: &Path, root: &Path) -> Result<Digest> {
    let bytes = std::fs::read(file).with_context(|| format!("reading {}", file.display()))?;
    let (header, start) = read_header(&bytes)?;
    let payload = &bytes[start..];
    let manifest = Manifest::read(root)?;
    if manifest.version == header.to.version {
        bail!("{} is already version {}", root.display(), header.to.version);
    }
    ensure!(
        manifest.version == header.from,
        "{} is version {}, and the patch is from {}",
        root.display(),
        manifest.version,
        header.from
    );
    let old: HashMap<&str, &Entry> = manifest.files.iter().map(|entry| (entry.path.as_str(), entry)).collect();
    // The old file at `path`, as its manifest says it is.
    let read_old = |path: &str| -> Result<Vec<u8>> {
        let entry = old.get(path).with_context(|| format!("the patch names {path}, which the package lacks"))?;
        let bytes = std::fs::read(root.join(checked_path(path)?)).with_context(|| format!("reading {path}"))?;
        ensure!(Digest::of(&bytes) == entry.digest, "{path} has changed since the package was made; reinstall it");
        Ok(bytes)
    };
    let slice = |at: u64, length: u64| -> Result<&[u8]> {
        let at = usize::try_from(at)?;
        let end = at.checked_add(usize::try_from(length)?).filter(|&end| end <= payload.len());
        Ok(&payload[at..end.context("the patch ends inside a payload")?])
    };
    let staging = sibling(root, "patching")?;
    let built = (|| -> Result<()> {
        let entries: BTreeMap<&str, &Entry> =
            header.to.files.iter().map(|entry| (entry.path.as_str(), entry)).collect();
        ensure!(entries.len() == header.files.len(), "the patch does not make each file once");
        for make in &header.files {
            let (path, bytes) = match make {
                Make::Keep { path } => (path, read_old(path)?),
                Make::Copy { path, from } => (path, read_old(from)?),
                Make::Delta { path, base, at, length } => {
                    let entry =
                        entries.get(path.as_str()).with_context(|| format!("{path} is not in the patch's list"))?;
                    let delta = inflate(slice(*at, *length)?, entry.size.saturating_mul(2) + (1 << 20))?;
                    (path, crate::delta::apply(&read_old(base)?, &delta)?)
                }
                Make::Whole { path, at, length } => {
                    let entry =
                        entries.get(path.as_str()).with_context(|| format!("{path} is not in the patch's list"))?;
                    (path, inflate(slice(*at, *length)?, entry.size)?)
                }
            };
            let entry = entries.get(path.as_str()).with_context(|| format!("{path} is not in the patch's list"))?;
            ensure!(Digest::of(&bytes) == entry.digest, "the patch made {path} wrong");
            let target = staging.join(checked_path(path)?);
            std::fs::create_dir_all(target.parent().unwrap())?;
            std::fs::write(&target, &bytes)?;
            if entry.executable {
                set_executable(&target)?;
            }
        }
        header.to.write(&staging)
    })();
    if let Err(error) = built {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(error.context(format!("{} is as it was", root.display())));
    }
    // Swapped in by two renames; the old version is removed after.
    let previous = sibling(root, "previous")?;
    std::fs::remove_dir(&previous)?;
    std::fs::rename(root, &previous).with_context(|| format!("moving {} aside", root.display()))?;
    if let Err(error) = std::fs::rename(&staging, root) {
        std::fs::rename(&previous, root)?;
        return Err(error).with_context(|| format!("{} is as it was", root.display()));
    }
    std::fs::remove_dir_all(&previous)?;
    Ok(header.to.version)
}

/// A new empty directory beside `root`.
fn sibling(root: &Path, what: &str) -> Result<PathBuf> {
    let parent = root.parent().context("the package has no parent directory")?;
    let name = root.file_name().context("the package has no name")?.to_string_lossy();
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let directory = parent.join(format!(".{name}.{what}-{}-{nanos}", std::process::id()));
    std::fs::create_dir(&directory).with_context(|| format!("making {}", directory.display()))?;
    Ok(directory)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, path: &str, bytes: &[u8]) {
        let target = root.join(path);
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::write(target, bytes).unwrap();
    }

    fn noise(length: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..length)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                state.to_be_bytes()[0]
            })
            .collect()
    }

    #[test]
    fn patches_take_one_version_to_the_next() {
        let root = crate::store::scratch("patch");
        let (one, two, store) = (root.join("one/gdev"), root.join("two/gdev"), root.join("store"));
        let mesh = noise(300_000, 1);
        let texture = noise(100_000, 2);
        write(&one, "gdev", b"runtime 1");
        write(&one, "game/game.cwasm", b"program 1");
        write(&one, "game/assets/rock.mesh.json", &mesh);
        write(&one, "game/assets/rock.png", &texture);
        write(&one, "game/assets/old.wav", b"gone in two");
        #[cfg(unix)]
        set_executable(&one.join("gdev")).unwrap();
        let first = record(&one, "gdev", &store).unwrap();
        assert_eq!(Manifest::read(&one).unwrap().version, first);

        // The next version: the mesh edited in its middle, the texture
        // moved, a sound gone, a new one, the program rebuilt.
        write(&two, "gdev", b"runtime 1");
        write(&two, "game/game.cwasm", b"program 2");
        let mut edited = mesh.clone();
        edited.splice(150_000..150_010, noise(40, 3));
        write(&two, "game/assets/rock.mesh.json", &edited);
        write(&two, "game/assets/textures/rock.png", &texture);
        write(&two, "game/assets/new.wav", &noise(5000, 4));
        #[cfg(unix)]
        set_executable(&two.join("gdev")).unwrap();
        let second = record(&two, "gdev", &store).unwrap();

        let patch = root.join("update.penpatch");
        let made = make(&one, &two, Some(&store), &patch).unwrap();
        assert_eq!((made.kept, made.copied, made.deltas, made.whole, made.removed), (1, 1, 1, 2, 2));
        // The mesh travels as its edit, not its 300 kB.
        assert!(made.bytes < 10_000, "{made}");
        // Made from the store alone, by versions, it is the same patch.
        let stored = root.join("stored.penpatch");
        make(Path::new(&first.to_string()[..12]), Path::new(&second.to_string()), Some(&store), &stored).unwrap();
        assert_eq!(std::fs::read(&stored).unwrap(), std::fs::read(&patch).unwrap());

        let installed = root.join("installed/gdev");
        crate::project::copy_tree(&one, &installed).unwrap();
        assert_eq!(apply(&patch, &installed).unwrap(), second);
        assert_eq!(Manifest::read(&installed).unwrap(), Manifest::read(&two).unwrap());
        for path in package_files(&two).unwrap() {
            assert_eq!(
                std::fs::read(installed.join(&path)).unwrap(),
                std::fs::read(two.join(&path)).unwrap(),
                "{path}"
            );
        }
        assert!(!installed.join("game/assets/old.wav").exists());
        #[cfg(unix)]
        assert!(executable(&installed.join("gdev")));
        // Once is enough, and only from its own version.
        assert!(format!("{:#}", apply(&patch, &installed).unwrap_err()).contains("already"));
        let elsewhere = root.join("elsewhere/gdev");
        crate::project::copy_tree(&two, &elsewhere).unwrap();
        let back = root.join("back.penpatch");
        make(&two, &one, Some(&store), &back).unwrap();
        assert_eq!(apply(&back, &elsewhere).unwrap(), first);

        // A package changed by hand is refused, and left as it was.
        let tampered = root.join("tampered/gdev");
        crate::project::copy_tree(&one, &tampered).unwrap();
        write(&tampered, "game/assets/rock.mesh.json", b"edited");
        let refused = format!("{:#}", apply(&patch, &tampered).unwrap_err());
        assert!(refused.contains("has changed") && refused.contains("as it was"), "{refused}");
        assert_eq!(std::fs::read(tampered.join("game/assets/rock.mesh.json")).unwrap(), b"edited");
        let leftovers: Vec<_> = std::fs::read_dir(root.join("tampered")).unwrap().flatten().collect();
        assert_eq!(leftovers.len(), 1);
        // A damaged patch is refused before anything is touched.
        let mut damaged = std::fs::read(&patch).unwrap();
        let last = damaged.len() - 1;
        damaged[last] ^= 0xff;
        std::fs::write(&patch, damaged).unwrap();
        let fresh = root.join("fresh/gdev");
        crate::project::copy_tree(&one, &fresh).unwrap();
        assert!(apply(&patch, &fresh).is_err());
        assert_eq!(Manifest::read(&fresh).unwrap().version, first);
        assert!(checked_path("../escape").is_err() && checked_path("game/../../x").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
