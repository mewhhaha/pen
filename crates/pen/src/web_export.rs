//! What `pen build --web` adds to the static site for hosting it: a Brotli
//! and a gzip copy of each file that compresses (`NAME.br`, `NAME.gz`),
//! which itch.io serves with their Content-Encoding and servers such as
//! nginx and Caddy negotiate; the service worker (crates/pen-web/
//! service-worker.js), told the export's files and version, which fetches
//! the Brotli copies where the host decodes them, keeps the game for
//! offline play and serves it cross-origin isolated; and, with `--zip`, the
//! site as one archive to upload, checked against itch.io's limits.
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};

const SERVICE_WORKER: &str = include_str!("../../pen-web/service-worker.js");
const SERVICE_WORKER_FILE: &str = "service-worker.js";
/// The service worker's line that each export fills in.
const PLACEHOLDER: &str = "const EXPORT = null;";

/// Formats that are compressed already, which no copy would shrink.
const COMPRESSED: &[&str] =
    &["png", "jpg", "jpeg", "webp", "avif", "gif", "ogg", "opus", "mp3", "mp4", "webm", "woff2", "zip", "gz", "br"];
/// Below this, a compressed copy saves too little to be worth a file.
const SMALLEST: usize = 1024;

/// itch.io's limits for an HTML game's ZIP (itch.io/docs/creators/html5).
const ITCH_FILES: usize = 1000;
const ITCH_TOTAL: u64 = 500_000_000;
const ITCH_FILE: u64 = 200_000_000;
const ITCH_PATH: usize = 240;

/// The files under `site`, by their paths relative to it with `/`
/// separators, in order.
fn files(site: &Path) -> Result<Vec<(String, PathBuf)>> {
    fn walk(directory: &Path, prefix: &str, out: &mut Vec<(String, PathBuf)>) -> Result<()> {
        for entry in std::fs::read_dir(directory).with_context(|| format!("reading {}", directory.display()))? {
            let entry = entry?;
            let name = entry.file_name().to_str().context("a file name that is not UTF-8")?.to_owned();
            let path = format!("{prefix}{name}");
            if entry.file_type()?.is_dir() {
                walk(&entry.path(), &format!("{path}/"), out)?;
            } else {
                out.push((path, entry.path()));
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(site, "", &mut out)?;
    out.sort();
    Ok(out)
}

fn extension(path: &str) -> &str {
    path.rsplit_once('.').map_or("", |(_, extension)| extension)
}

/// A copy of `bytes` is worth keeping when it is at most 90% of them.
fn worthwhile(copy: &[u8], bytes: &[u8]) -> bool {
    copy.len() * 10 <= bytes.len() * 9
}

fn gzip(bytes: &[u8]) -> Result<Vec<u8>> {
    // No name and a zero time: the same bytes give the same copy.
    let mut encoder = flate2::GzBuilder::new().write(Vec::new(), flate2::Compression::best());
    encoder.write_all(bytes)?;
    Ok(encoder.finish()?)
}

/// The player runtime, which exports nothing, leaves the encoder out.
#[cfg(feature = "compiler")]
fn brotli(bytes: &[u8]) -> Result<Vec<u8>> {
    let params =
        brotli::enc::BrotliEncoderParams { quality: 11, lgwin: 22, size_hint: bytes.len(), ..Default::default() };
    let mut out = Vec::new();
    brotli::BrotliCompress(&mut &bytes[..], &mut out, &params)?;
    Ok(out)
}

#[cfg(not(feature = "compiler"))]
fn brotli(_: &[u8]) -> Result<Vec<u8>> {
    anyhow::bail!("this player runtime cannot export a checkout; use a full pen build")
}

/// A file's compressed copies worth keeping.
#[derive(Default)]
struct Copies {
    brotli: Option<Vec<u8>>,
    gzip: Option<Vec<u8>>,
}

/// The copies of `bytes` worth keeping, if any: a file that gzip cannot
/// shrink is taken to be compressed already.
fn compress(path: &str, bytes: &[u8]) -> Result<Copies> {
    if bytes.len() < SMALLEST || COMPRESSED.contains(&extension(path).to_ascii_lowercase().as_str()) {
        return Ok(Copies::default());
    }
    let gzip = gzip(bytes)?;
    if !worthwhile(&gzip, bytes) {
        return Ok(Copies::default());
    }
    let brotli = brotli(bytes)?;
    Ok(Copies { brotli: worthwhile(&brotli, bytes).then_some(brotli), gzip: Some(gzip) })
}

/// What precompressing a site's files made.
struct Precompressed {
    /// Each file's copies: `br`, `gz`, both or none.
    encodings: BTreeMap<String, Vec<&'static str>>,
    /// The bytes of the files with a Brotli copy, and of those copies.
    original: u64,
    compressed: u64,
}

/// Writes the copies of `files` that compress beside them, on a few threads
/// (Brotli's best quality is slow).
fn precompress(files: &[(String, PathBuf)]) -> Result<Precompressed> {
    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get().min(4));
    let work = || -> Result<()> {
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some((path, file)) = files.get(index) else { return Ok(()) };
            let bytes = std::fs::read(file).with_context(|| format!("reading {}", file.display()))?;
            let Copies { brotli, gzip } = compress(path, &bytes)?;
            let mut encodings = Vec::new();
            let mut sizes = (0, 0);
            for (encoding, copy) in [("br", &brotli), ("gz", &gzip)] {
                if let Some(copy) = copy {
                    let mut name = file.as_os_str().to_owned();
                    name.push(format!(".{encoding}"));
                    std::fs::write(&name, copy).with_context(|| format!("writing {}", Path::new(&name).display()))?;
                    encodings.push(encoding);
                }
            }
            if let Some(copy) = &brotli {
                sizes = (bytes.len() as u64, copy.len() as u64);
            }
            results.lock().unwrap().push((path.clone(), encodings, sizes));
        }
    };
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads).map(|_| scope.spawn(work)).collect();
        workers.into_iter().try_for_each(|worker| worker.join().expect("a compression thread panicked"))
    })?;
    let mut made = Precompressed { encodings: BTreeMap::new(), original: 0, compressed: 0 };
    for (path, kept, (from, to)) in results.into_inner().unwrap() {
        made.encodings.insert(path, kept);
        made.original += from;
        made.compressed += to;
    }
    Ok(made)
}

/// The export's version: a hash of its files and of the service worker.
fn version(files: &[(String, PathBuf)]) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(SERVICE_WORKER);
    for (path, file) in files {
        let bytes = std::fs::read(file)?;
        hash.update(path.as_bytes());
        hash.update([0]);
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(&bytes);
    }
    Ok(hash.finalize()[..8].iter().map(|byte| format!("{byte:02x}")).collect())
}

/// The service worker for an export of `files` with these encodings.
fn service_worker(version: &str, encodings: &BTreeMap<String, Vec<&'static str>>) -> Result<String> {
    ensure!(SERVICE_WORKER.matches(PLACEHOLDER).count() == 1, "the service worker has no single `{PLACEHOLDER}`");
    let export = serde_json::json!({ "version": version, "files": encodings });
    Ok(SERVICE_WORKER.replacen(PLACEHOLDER, &format!("const EXPORT = {export};"), 1))
}

/// Precompresses the site's files and writes its service worker; returns
/// what it did, for the export's summary.
pub fn prepare(site: &Path) -> Result<String> {
    let files: Vec<_> = files(site)?.into_iter().filter(|(path, _)| path != SERVICE_WORKER_FILE).collect();
    let version = version(&files)?;
    let Precompressed { encodings, original, compressed } = precompress(&files)?;
    std::fs::write(site.join(SERVICE_WORKER_FILE), service_worker(&version, &encodings)?)?;
    let copies = encodings.values().filter(|kept| !kept.is_empty()).count();
    Ok(format!(
        "{copies} of {} files precompressed ({:.1} MB as Brotli, from {:.1} MB), service worker {version}",
        files.len(),
        compressed as f64 / 1e6,
        original as f64 / 1e6
    ))
}

/// What itch.io would refuse in a ZIP of these files (paths and sizes).
fn itch_refusals(files: &[(String, u64)]) -> Vec<String> {
    let mut refusals = Vec::new();
    if files.len() > ITCH_FILES {
        refusals.push(format!("{} files, more than {ITCH_FILES}", files.len()));
    }
    let total: u64 = files.iter().map(|(_, size)| size).sum();
    if total > ITCH_TOTAL {
        refusals.push(format!("{:.1} MB in all, more than {} MB", total as f64 / 1e6, ITCH_TOTAL / 1_000_000));
    }
    for (path, size) in files {
        if *size > ITCH_FILE {
            refusals.push(format!("{path} is {:.1} MB, more than {} MB", *size as f64 / 1e6, ITCH_FILE / 1_000_000));
        }
        if path.chars().count() > ITCH_PATH {
            refusals.push(format!("{path} is longer than {ITCH_PATH} characters"));
        }
    }
    refusals
}

/// Little-endian fields of a ZIP record.
trait Fields {
    fn u16(&mut self, value: u16);
    fn u32(&mut self, value: u32);
}

impl Fields for Vec<u8> {
    fn u16(&mut self, value: u16) {
        self.extend_from_slice(&value.to_le_bytes());
    }
    fn u32(&mut self, value: u32) {
        self.extend_from_slice(&value.to_le_bytes());
    }
}

/// An entry's fields that the local header and the central directory share,
/// from "version needed" to the name's length.
fn entry_fields(record: &mut Vec<u8>, name: &str, method: u16, crc: u32, stored: u32, size: u32) -> Result<()> {
    record.u16(20); // version 2.0: deflate
    record.u16(0x0800); // the name is UTF-8
    record.u16(method);
    // 1980-01-01 00:00, for the same bytes from the same files.
    record.u16(0);
    record.u16(0x21);
    record.u32(crc);
    record.u32(stored);
    record.u32(size);
    record.u16(u16::try_from(name.len()).context("a path too long for a ZIP")?);
    Ok(())
}

/// Writes the site as a ZIP archive at `zip`: the files deflated where that
/// shrinks them, stored otherwise (the precompressed copies among them).
/// Says what itch.io would refuse in it.
pub fn zip(site: &Path, zip: &Path) -> Result<()> {
    let files = files(site)?;
    ensure!(files.len() < usize::from(u16::MAX), "{} files are more than a ZIP holds", files.len());
    let mut out = Vec::new();
    let mut directory = Vec::new();
    let mut sizes = Vec::new();
    for (name, file) in &files {
        let bytes = std::fs::read(file).with_context(|| format!("reading {}", file.display()))?;
        let mut crc = flate2::Crc::new();
        crc.update(&bytes);
        let deflated = if COMPRESSED.contains(&extension(name).to_ascii_lowercase().as_str()) {
            None
        } else {
            let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::best());
            encoder.write_all(&bytes)?;
            Some(encoder.finish()?).filter(|deflated| deflated.len() < bytes.len())
        };
        let (method, data) = match &deflated {
            Some(deflated) => (8, deflated.as_slice()),
            None => (0, bytes.as_slice()),
        };
        let too_big = || format!("{name} is too big for a ZIP");
        let size = u32::try_from(bytes.len()).with_context(too_big)?;
        let stored = u32::try_from(data.len()).with_context(too_big)?;
        let offset = u32::try_from(out.len()).context("the site is too big for a ZIP")?;
        out.u32(0x0403_4b50);
        entry_fields(&mut out, name, method, crc.sum(), stored, size)?;
        out.u16(0); // no extra field
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);
        directory.u32(0x0201_4b50);
        directory.u16(0x0314); // made on Unix, by ZIP 2.0
        entry_fields(&mut directory, name, method, crc.sum(), stored, size)?;
        directory.u16(0); // no extra field
        directory.u16(0); // no comment
        directory.u16(0); // disk 0
        directory.u16(0); // binary
        directory.u32(0o100_644 << 16); // a regular file, rw-r--r--
        directory.u32(offset);
        directory.extend_from_slice(name.as_bytes());
        sizes.push((name.clone(), bytes.len() as u64));
    }
    let count = u16::try_from(files.len())?;
    let start = u32::try_from(out.len()).context("the site is too big for a ZIP")?;
    let length = u32::try_from(directory.len())?;
    out.extend_from_slice(&directory);
    out.u32(0x0605_4b50);
    out.u16(0);
    out.u16(0);
    out.u16(count);
    out.u16(count);
    out.u32(length);
    out.u32(start);
    out.u16(0); // no comment
    if let Some(parent) = zip.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let partial = zip.with_extension(format!("zip.{}.tmp", std::process::id()));
    std::fs::write(&partial, &out).with_context(|| format!("writing {}", partial.display()))?;
    std::fs::rename(&partial, zip).with_context(|| format!("writing {}", zip.display()))?;
    eprintln!("zipped {} files into {} ({:.1} MB)", files.len(), zip.display(), out.len() as f64 / 1e6);
    for refusal in itch_refusals(&sizes) {
        eprintln!("warning: itch.io would refuse this ZIP: {refusal}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn sample_site(name: &str) -> PathBuf {
        let site = crate::project::temporary_dir().unwrap().join(format!("web-export-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&site);
        std::fs::create_dir_all(site.join("assets")).unwrap();
        // Text that compresses, noise that does not, and a small file.
        let text: String = (0..400).map(|line| format!("line {line} of the game's data\n")).collect();
        let mut state = 0x2545_f491_u32;
        let noise: Vec<u8> = (0..8192)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state.to_le_bytes()[0]
            })
            .collect();
        std::fs::write(site.join("game.wasm"), &text).unwrap();
        std::fs::write(site.join("assets/rock.mesh.json"), &text).unwrap();
        std::fs::write(site.join("assets/rock.png"), &text).unwrap();
        std::fs::write(site.join("assets/noise.gtex"), &noise).unwrap();
        std::fs::write(site.join("index.html"), "<!doctype html>").unwrap();
        site
    }

    // The player runtime has no Brotli encoder, and exports nothing.
    #[cfg(feature = "compiler")]
    #[test]
    fn copies_what_compresses_and_tells_the_service_worker() {
        let site = sample_site("prepare");
        prepare(&site).unwrap();
        let text = std::fs::read(site.join("game.wasm")).unwrap();
        for name in ["game.wasm", "assets/rock.mesh.json"] {
            let mut decoded = Vec::new();
            let brotli = std::fs::read(site.join(format!("{name}.br"))).unwrap();
            brotli::BrotliDecompress(&mut brotli.as_slice(), &mut decoded).unwrap();
            assert_eq!(decoded, text, "{name}.br");
            let mut decoded = Vec::new();
            let gzip = std::fs::read(site.join(format!("{name}.gz"))).unwrap();
            flate2::read::GzDecoder::new(gzip.as_slice()).read_to_end(&mut decoded).unwrap();
            assert_eq!(decoded, text, "{name}.gz");
        }
        // Formats compressed already, noise and small files have no copies.
        for name in ["assets/rock.png", "assets/noise.gtex", "index.html"] {
            assert!(!site.join(format!("{name}.br")).exists() && !site.join(format!("{name}.gz")).exists(), "{name}");
        }
        let worker = std::fs::read_to_string(site.join(SERVICE_WORKER_FILE)).unwrap();
        let line = worker.lines().find(|line| line.starts_with("const EXPORT = ")).unwrap();
        let export: serde_json::Value =
            serde_json::from_str(line.trim_start_matches("const EXPORT = ").trim_end_matches(';')).unwrap();
        assert_eq!(export["files"]["game.wasm"], serde_json::json!(["br", "gz"]));
        assert_eq!(export["files"]["assets/noise.gtex"], serde_json::json!([]));
        assert_eq!(export["files"].as_object().unwrap().len(), 5, "{export}");
        let version = export["version"].as_str().unwrap().to_owned();
        assert_eq!(version.len(), 16);

        // The same files give the same version; a changed one another.
        let again = sample_site("again");
        assert!(prepare(&again).unwrap().ends_with(&version));
        std::fs::write(again.join("assets/rock.png"), "another rock").unwrap();
        std::fs::remove_file(again.join(SERVICE_WORKER_FILE)).unwrap();
        for copy in files(&again).unwrap().iter().filter(|(path, _)| path.ends_with(".br") || path.ends_with(".gz")) {
            std::fs::remove_file(&copy.1).unwrap();
        }
        assert!(!prepare(&again).unwrap().ends_with(&version));
        std::fs::remove_dir_all(&site).unwrap();
        std::fs::remove_dir_all(&again).unwrap();
    }

    /// Reads a ZIP's central directory back: each entry's name and contents.
    fn unzip(bytes: &[u8]) -> Vec<(String, Vec<u8>)> {
        let u16_at = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
        let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let end = bytes.len() - 22;
        assert_eq!(u32_at(end), 0x0605_4b50);
        let count = usize::from(u16_at(end + 10));
        let mut at = u32_at(end + 16) as usize;
        let mut entries = Vec::new();
        for _ in 0..count {
            assert_eq!(u32_at(at), 0x0201_4b50);
            let (method, crc, stored) = (u16_at(at + 10), u32_at(at + 16), u32_at(at + 20) as usize);
            let name_length = usize::from(u16_at(at + 28));
            let skip = name_length + usize::from(u16_at(at + 30)) + usize::from(u16_at(at + 32));
            let name = String::from_utf8(bytes[at + 46..at + 46 + name_length].to_vec()).unwrap();
            let local = u32_at(at + 42) as usize;
            assert_eq!(u32_at(local), 0x0403_4b50);
            let data = local + 30 + usize::from(u16_at(local + 26)) + usize::from(u16_at(local + 28));
            let raw = &bytes[data..data + stored];
            let contents = match method {
                0 => raw.to_vec(),
                8 => {
                    let mut out = Vec::new();
                    flate2::read::DeflateDecoder::new(raw).read_to_end(&mut out).unwrap();
                    out
                }
                method => panic!("method {method}"),
            };
            let mut check = flate2::Crc::new();
            check.update(&contents);
            assert_eq!(check.sum(), crc, "{name}");
            entries.push((name, contents));
            at += 46 + skip;
        }
        entries
    }

    #[test]
    fn zips_the_site_as_it_is() {
        let site = sample_site("zip");
        // A precompressed copy, stored as it is.
        std::fs::write(site.join("assets/rock.mesh.json.br"), [11, 0, 3]).unwrap();
        let out = site.with_extension("zip");
        zip(&site, &out).unwrap();
        let entries = unzip(&std::fs::read(&out).unwrap());
        let expected = files(&site).unwrap();
        assert_eq!(entries.len(), expected.len());
        for ((name, contents), (path, file)) in entries.iter().zip(&expected) {
            assert_eq!(name, path);
            assert_eq!(contents, &std::fs::read(file).unwrap(), "{name}");
        }
        assert!(entries.iter().any(|(name, _)| name == "assets/rock.mesh.json.br"));
        std::fs::remove_dir_all(&site).unwrap();
        std::fs::remove_file(&out).unwrap();
    }

    #[test]
    fn itch_limits() {
        let small = vec![("index.html".to_owned(), 100)];
        assert!(itch_refusals(&small).is_empty());
        let many: Vec<_> = (0..1001).map(|index| (format!("assets/{index}.json"), 1)).collect();
        assert_eq!(itch_refusals(&many), ["1001 files, more than 1000"]);
        let large = vec![("game.wasm".to_owned(), 300_000_000), ("assets/a.json".to_owned(), 250_000_000)];
        let refusals = itch_refusals(&large);
        assert_eq!(refusals.len(), 3, "{refusals:?}");
        assert!(refusals[0].contains("550.0 MB in all"));
        let long = vec![("a/".repeat(121), 1)];
        assert_eq!(itch_refusals(&long).len(), 1);
    }
}
