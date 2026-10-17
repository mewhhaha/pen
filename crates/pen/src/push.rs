//! Pushing a package to another machine over the network: `pen serve`
//! receives packages (on a Steam Deck, say) and `pen push` sends one there,
//! only the files that changed since the last push, and can start it.
//!
//! The protocol, over TCP: each message is a little-endian u32 length and
//! that much JSON, and file contents follow their message raw. The pusher
//! says who it is (the server's token), the package's name and each file's
//! path, size, SHA-256 and whether it is executable; the server answers
//! which files it lacks (or why it refuses); the pusher sends those; the
//! server checks each against its hash, stages the package beside the one
//! it has (unchanged files linked or copied over), and puts it in place with
//! a rename, then answers how it went. It starts the package's executable
//! only when both ends asked for that (`--run`, `--allow-run`), after
//! stopping the one it started before.
//!
//! The server only listens where it is told (127.0.0.1 unless `--listen`
//! says otherwise), refuses a pusher without its token, and takes only
//! relative paths inside the package.
use std::collections::BTreeMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Component, Path, PathBuf};
use std::process::Child;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Where `pen serve` listens unless told otherwise, and `pen push` sends.
pub const PORT: u16 = 7811;
/// The protocol's version, which both ends must speak.
const VERSION: u32 = 1;
/// The largest message, and package, the server takes.
const MESSAGE_BYTES: u32 = 16 << 20;
const PACKAGE_BYTES: u64 = 8 << 30;
/// How long either end waits on the other before it gives up.
const PATIENCE: Duration = Duration::from_secs(120);

/// A file of a package as the pusher describes it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    /// Its path in the package, with `/` between parts.
    pub path: String,
    pub size: u64,
    /// Its SHA-256, in hex.
    pub hash: String,
    pub executable: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct Offer {
    pen: String,
    version: u32,
    token: String,
    /// The package's name: where the server keeps it.
    name: String,
    files: Vec<Entry>,
    /// Start this file of the package once it is in place.
    run: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Reply {
    /// Why the server refuses, if it does.
    refused: Option<String>,
    /// The files it lacks, by their place in the offer.
    need: Vec<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Done {
    pub files: usize,
    pub sent: usize,
    pub bytes: u64,
    /// Where it put the package.
    pub at: String,
    /// What it started, if anything.
    pub started: Option<String>,
    /// Why it failed, if it did.
    pub failed: Option<String>,
}

fn write_message(stream: &mut impl Write, message: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec(message)?;
    stream.write_all(&u32::try_from(bytes.len())?.to_le_bytes())?;
    stream.write_all(&bytes)?;
    stream.flush()?;
    Ok(())
}

fn read_message<T: for<'a> Deserialize<'a>>(stream: &mut impl Read) -> Result<T> {
    let mut length = [0; 4];
    stream.read_exact(&mut length)?;
    let length = u32::from_le_bytes(length);
    ensure!(length <= MESSAGE_BYTES, "a message of {length} bytes is past the protocol's limit");
    let mut bytes = vec![0; length as usize];
    stream.read_exact(&mut bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Whether `path` names a file inside a package: relative, its parts
/// plain names, `/` between them.
fn inside(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && Path::new(path).components().all(|part| matches!(part, Component::Normal(_)))
        && path.split('/').all(|part| !part.is_empty() && part != "." && part != "..")
}

/// Whether `name` will do as a package's name: a plain file name.
fn plain(name: &str) -> bool {
    inside(name) && !name.contains('/') && !name.starts_with('.')
}

#[cfg(unix)]
fn executable(metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;
    metadata.permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn executable(_metadata: &std::fs::Metadata) -> bool {
    false
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(path)?.permissions();
    permissions.set_mode(permissions.mode() | 0o755);
    std::fs::set_permissions(path, permissions)?;
    Ok(())
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<()> {
    Ok(())
}

fn hash_file(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hex(&hasher.finalize()))
}

/// The files of the package in `directory`, by path, each hashed.
pub fn describe(directory: &Path) -> Result<Vec<Entry>> {
    let mut entries = Vec::new();
    let mut pending = vec![directory.to_owned()];
    while let Some(folder) = pending.pop() {
        for item in std::fs::read_dir(&folder).with_context(|| format!("reading {}", folder.display()))? {
            let item = item?;
            let kind = item.file_type()?;
            if kind.is_dir() {
                pending.push(item.path());
            } else if kind.is_file() {
                let path = item.path();
                let relative = path.strip_prefix(directory)?;
                let parts: Vec<String> =
                    relative.components().map(|part| part.as_os_str().to_string_lossy().into_owned()).collect();
                let metadata = item.metadata()?;
                entries.push(Entry {
                    path: parts.join("/"),
                    size: metadata.len(),
                    hash: hash_file(&path)?,
                    executable: executable(&metadata),
                });
            }
        }
    }
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(entries)
}

/// How a push should go.
pub struct Push<'a> {
    pub address: &'a str,
    pub token: &'a str,
    /// The package's directory, and its name on the server.
    pub package: &'a Path,
    pub name: &'a str,
    /// Start this file of the package there.
    pub run: Option<&'a str>,
}

/// Pushes a package; returns what the server says it did.
pub fn push(push: &Push) -> Result<Done> {
    let files = describe(push.package)?;
    ensure!(!files.is_empty(), "{} holds no files to push", push.package.display());
    if let Some(run) = push.run {
        ensure!(files.iter().any(|file| file.path == run), "the package has no {run} to start");
    }
    let address = if push.address.contains(':') { push.address.to_owned() } else { format!("{}:{PORT}", push.address) };
    let target = address.to_socket_addrs()?.next().with_context(|| format!("{address} names no address"))?;
    let stream = TcpStream::connect_timeout(&target, Duration::from_secs(10))
        .with_context(|| format!("connecting to pen serve at {address}"))?;
    stream.set_read_timeout(Some(PATIENCE))?;
    stream.set_write_timeout(Some(PATIENCE))?;
    let (mut reader, mut writer) = (BufReader::new(stream.try_clone()?), BufWriter::new(stream));
    let offer = Offer {
        pen: "push".into(),
        version: VERSION,
        token: push.token.to_owned(),
        name: push.name.to_owned(),
        files: files.clone(),
        run: push.run.map(str::to_owned),
    };
    write_message(&mut writer, &offer)?;
    let reply: Reply = read_message(&mut reader)?;
    if let Some(why) = reply.refused {
        bail!("pen serve at {address} refused the push: {why}");
    }
    for &index in &reply.need {
        let entry = files.get(index).context("the server asked for a file the package does not have")?;
        let mut file = std::fs::File::open(push.package.join(&entry.path))?;
        let copied = std::io::copy(&mut (&mut file).take(entry.size), &mut writer)?;
        ensure!(copied == entry.size, "{} changed while it was pushed", entry.path);
    }
    writer.flush()?;
    let done: Done = read_message(&mut reader)?;
    if let Some(why) = &done.failed {
        bail!("pen serve at {address}: {why}");
    }
    Ok(done)
}

/// How `pen serve` serves.
pub struct Serve {
    pub listen: String,
    pub token: String,
    /// Where packages go: a directory each, by name.
    pub directory: PathBuf,
    /// Whether a pusher may start what it pushed.
    pub allow_run: bool,
}

/// A token for a server not given one: 128 random bits in hex, from the
/// system's randomness (the standard library's hasher keys).
pub fn new_token() -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut words = String::new();
    for _ in 0..2 {
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_u128(
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos(),
        );
        words.push_str(&format!("{:016x}", hasher.finish()));
    }
    words
}

/// Compares the tokens in time that does not depend on where they differ.
fn same_token(given: &str, wanted: &str) -> bool {
    let (given, wanted) = (given.as_bytes(), wanted.as_bytes());
    given.len() == wanted.len() && given.iter().zip(wanted).fold(0, |differ, (a, b)| differ | (a ^ b)) == 0
}

/// What the server keeps between pushes: the packages it started.
#[derive(Default)]
pub struct Server {
    started: BTreeMap<String, Child>,
}

impl Server {
    /// Serves pushes one at a time, until `stop` says so (never, from the
    /// command line); says what each did.
    pub fn serve(&mut self, serve: &Serve, listener: &TcpListener, stop: impl Fn() -> bool) -> Result<()> {
        std::fs::create_dir_all(&serve.directory)?;
        listener.set_nonblocking(true)?;
        while !stop() {
            match listener.accept() {
                Ok((stream, from)) => {
                    let started = Instant::now();
                    match self.receive(serve, stream) {
                        Ok(done) => eprintln!(
                            "serve: {from} pushed {} ({} of {} files, {:.1} MB) in {:.1} s{}",
                            done.at,
                            done.sent,
                            done.files,
                            done.bytes as f64 / 1e6,
                            started.elapsed().as_secs_f64(),
                            done.started.map(|run| format!("; started {run}")).unwrap_or_default()
                        ),
                        Err(error) => eprintln!("serve: a push from {from} failed: {error:#}"),
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }

    /// Takes one push.
    fn receive(&mut self, serve: &Serve, stream: TcpStream) -> Result<Done> {
        stream.set_nonblocking(false)?;
        stream.set_read_timeout(Some(PATIENCE))?;
        stream.set_write_timeout(Some(PATIENCE))?;
        let (mut reader, mut writer) = (BufReader::new(stream.try_clone()?), BufWriter::new(stream));
        let offer: Offer = read_message(&mut reader)?;
        let refusal = refuse(serve, &offer);
        let destination = serve.directory.join(&offer.name);
        let need: Vec<usize> = match &refusal {
            Some(_) => Vec::new(),
            None => offer
                .files
                .iter()
                .enumerate()
                .filter(|(_, entry)| {
                    let held = destination.join(&entry.path);
                    !(held.metadata().is_ok_and(|metadata| metadata.len() == entry.size)
                        && hash_file(&held).is_ok_and(|hash| hash == entry.hash))
                })
                .map(|(index, _)| index)
                .collect(),
        };
        write_message(&mut writer, &Reply { refused: refusal.clone(), need: need.clone() })?;
        if let Some(why) = refusal {
            bail!("refused: {why}");
        }
        let staged = serve.directory.join(format!(".{}.incoming-{}", offer.name, std::process::id()));
        let result = self.stage(serve, &offer, &need, &mut reader, &staged, &destination);
        if result.is_err() {
            let _ = std::fs::remove_dir_all(&staged);
        }
        let done = match result {
            Ok(done) => done,
            Err(error) => Done {
                files: offer.files.len(),
                sent: 0,
                bytes: 0,
                at: destination.display().to_string(),
                started: None,
                failed: Some(format!("{error:#}")),
            },
        };
        write_message(&mut writer, &done)?;
        Ok(done)
    }

    /// Receives the files sent into `staged` beside the package, links or
    /// copies the rest from it, puts the new package in its place, and
    /// starts it when asked.
    fn stage(
        &mut self,
        serve: &Serve,
        offer: &Offer,
        need: &[usize],
        reader: &mut impl Read,
        staged: &Path,
        destination: &Path,
    ) -> Result<Done> {
        let _ = std::fs::remove_dir_all(staged);
        std::fs::create_dir_all(staged)?;
        let mut bytes = 0;
        for &index in need {
            let entry = &offer.files[index];
            let path = staged.join(&entry.path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut file = BufWriter::new(std::fs::File::create(&path)?);
            let mut hasher = Sha256::new();
            let mut left = entry.size;
            let mut buffer = vec![0; 1 << 16];
            while left > 0 {
                let want = usize::try_from(left.min(buffer.len() as u64))?;
                reader.read_exact(&mut buffer[..want])?;
                hasher.update(&buffer[..want]);
                file.write_all(&buffer[..want])?;
                left -= want as u64;
            }
            file.flush()?;
            ensure!(hex(&hasher.finalize()) == entry.hash, "{} came with another hash than its own", entry.path);
            bytes += entry.size;
        }
        for (index, entry) in offer.files.iter().enumerate() {
            let path = staged.join(&entry.path);
            if !need.contains(&index) {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let held = destination.join(&entry.path);
                if std::fs::hard_link(&held, &path).is_err() {
                    std::fs::copy(&held, &path).with_context(|| format!("keeping {}", entry.path))?;
                }
            }
            if entry.executable {
                make_executable(&path)?;
            }
        }
        // What it started from the package goes before the package does.
        if let Some(mut child) = self.started.remove(&offer.name) {
            let _ = child.kill();
            let _ = child.wait();
        }
        let old = serve.directory.join(format!(".{}.old-{}", offer.name, std::process::id()));
        let _ = std::fs::remove_dir_all(&old);
        if destination.exists() {
            std::fs::rename(destination, &old)?;
        }
        std::fs::rename(staged, destination)?;
        let _ = std::fs::remove_dir_all(&old);
        let started = match &offer.run {
            Some(run) => {
                let program = destination.join(run);
                let child = std::process::Command::new(&program)
                    .current_dir(destination)
                    .spawn()
                    .with_context(|| format!("starting {}", program.display()))?;
                self.started.insert(offer.name.clone(), child);
                Some(run.clone())
            }
            None => None,
        };
        Ok(Done {
            files: offer.files.len(),
            sent: need.len(),
            bytes,
            at: destination.display().to_string(),
            started,
            failed: None,
        })
    }
}

/// Why the server refuses `offer`, if it does.
fn refuse(serve: &Serve, offer: &Offer) -> Option<String> {
    if offer.pen != "push" || offer.version != VERSION {
        return Some(format!("it speaks push version {VERSION}"));
    }
    if !same_token(&offer.token, &serve.token) {
        return Some("wrong token".into());
    }
    if !plain(&offer.name) {
        return Some(format!("{} is not a package's name", offer.name));
    }
    if let Some(entry) = offer.files.iter().find(|entry| !inside(&entry.path) || entry.hash.len() != 64) {
        return Some(format!("{} is not a file inside a package", entry.path));
    }
    let mut seen = std::collections::HashSet::new();
    if let Some(entry) = offer.files.iter().find(|entry| !seen.insert(entry.path.as_str())) {
        return Some(format!("{} comes twice", entry.path));
    }
    if offer.files.iter().map(|entry| entry.size).sum::<u64>() > PACKAGE_BYTES {
        return Some("the package is too large".into());
    }
    match &offer.run {
        Some(_) if !serve.allow_run => Some("this server starts nothing it is sent (pen serve --allow-run)".into()),
        Some(run) if !offer.files.iter().any(|entry| entry.path == *run) => Some(format!("the package has no {run}")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn scratch(name: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!("pen-push-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    #[test]
    fn takes_only_paths_inside_a_package() {
        assert!(inside("game/assets/rock.png") && inside("gdev"));
        for path in ["", "/etc/passwd", "../x", "a/../../x", "a//b", "a\\b", "./a", "a/."] {
            assert!(!inside(path), "{path}");
        }
        assert!(plain("gdev") && !plain("a/b") && !plain(".hidden"));
        assert!(same_token("abc", "abc") && !same_token("abc", "abd") && !same_token("abc", "ab"));
        assert_eq!(new_token().len(), 32);
        assert_ne!(new_token(), new_token());
    }

    #[test]
    fn pushes_what_changed_and_refuses_strangers() {
        let (package, served) = (scratch("package"), scratch("served"));
        std::fs::create_dir_all(package.join("game/assets")).unwrap();
        std::fs::write(package.join("gdev"), b"#!/bin/sh\n").unwrap();
        make_executable(&package.join("gdev")).unwrap();
        std::fs::write(package.join("game/game.cwasm"), vec![7; 200_000]).unwrap();
        std::fs::write(package.join("game/assets/rock.png"), b"rock").unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let serve =
            Serve { listen: address.clone(), token: "secret".into(), directory: served.clone(), allow_run: false };
        let server = std::thread::spawn(move || {
            Server::default().serve(&serve, &listener, || stopping.load(Ordering::Relaxed)).unwrap();
        });
        let asked = |token: &str, run: Option<&str>| {
            push(&Push { address: &address, token, package: &package, name: "gdev", run })
        };
        let done = asked("secret", None).unwrap();
        assert_eq!((done.files, done.sent, done.bytes), (3, 3, 200_014));
        assert_eq!(std::fs::read(served.join("gdev/game/assets/rock.png")).unwrap(), b"rock");
        assert!(executable(&std::fs::metadata(served.join("gdev/gdev")).unwrap()) || cfg!(not(unix)));
        // Again, with one file changed: only it goes.
        std::fs::write(package.join("game/assets/rock.png"), b"boulder").unwrap();
        let done = asked("secret", None).unwrap();
        assert_eq!((done.sent, done.bytes), (1, 7));
        assert_eq!(std::fs::read(served.join("gdev/game/assets/rock.png")).unwrap(), b"boulder");
        assert_eq!(std::fs::read(served.join("gdev/game/game.cwasm")).unwrap().len(), 200_000);
        // A file gone from the package goes from the server's copy too.
        std::fs::remove_file(package.join("game/assets/rock.png")).unwrap();
        assert_eq!(asked("secret", None).unwrap().sent, 0);
        assert!(!served.join("gdev/game/assets/rock.png").exists());
        let error = asked("guess", None).unwrap_err();
        assert!(format!("{error:#}").contains("wrong token"), "{error:#}");
        let error = asked("secret", Some("gdev")).unwrap_err();
        assert!(format!("{error:#}").contains("--allow-run"), "{error:#}");
        stop.store(true, Ordering::Relaxed);
        server.join().unwrap();
        let leftovers: Vec<_> = std::fs::read_dir(&served).unwrap().map(|entry| entry.unwrap().file_name()).collect();
        assert_eq!(leftovers, ["gdev"], "nothing staged is left behind");
        let _ = std::fs::remove_dir_all(&package);
        let _ = std::fs::remove_dir_all(&served);
    }
}
