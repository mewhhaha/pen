//! Cache directory verification, content hashing, build locking, subprocess timeouts,
//! and verified shared-object compilation/loading for `wtime-aot`.

use std::io::{Read as _, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use libloading::Library;

use super::toolchain::*;
use super::{
    X64_ADMISSION_REJECTED_EXIT_CODE, X64FallbackError, X64FallbackReason, find_in_path, resolve_wtime_aot_bin,
    sanitized_path_env,
};

pub(super) fn default_cache_dir(cache: Option<&Path>) -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("PEN_WTIME_AOT_CACHE")
        && !dir.is_empty()
    {
        return Ok(PathBuf::from(dir));
    }
    if let Some(cache) = cache {
        return Ok(cache.join("wtime-aot"));
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME")
        && !xdg.is_empty()
    {
        return Ok(PathBuf::from(xdg).join("pen").join("wtime-aot"));
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.is_empty()
    {
        return Ok(PathBuf::from(home).join(".cache").join("pen").join("wtime-aot"));
    }
    if let Ok(rt) = std::env::var("XDG_RUNTIME_DIR")
        && !rt.is_empty()
    {
        return Ok(PathBuf::from(rt).join("pen").join("wtime-aot"));
    }
    bail!(
        "no owner-private wtime-aot cache directory available (set PEN_WTIME_AOT_CACHE, XDG_CACHE_HOME, HOME, or XDG_RUNTIME_DIR)"
    )
}

pub(super) fn wasm_sha256_hex(wasm: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(wasm))
}

pub(super) const WTIME_AOT_MERKLE256K_CHUNK_BYTES: usize = 256 << 10;
pub(super) const WTIME_AOT_MERKLE256K_PREFIX: &str = "merkle256k:";
pub(super) const WTIME_AOT_MERKLE256K_DOMAIN: &[u8] = b"WTIME-MERKLE256K-1";

/// Computes `wtime-aot`'s deterministic 256 KiB chunk SHA-256 Merkle root (`WTIME-MERKLE256K-1`),
/// matching `tools/wtime-aot/src/sha256.rs::parallel_merkle_digest`, hashing chunks in parallel
/// when `bytes.len() > 256 KiB`.
pub(super) fn merkle256k_sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut root = Sha256::new();
    root.update(WTIME_AOT_MERKLE256K_DOMAIN);
    root.update((bytes.len() as u64).to_le_bytes());
    if bytes.len() <= WTIME_AOT_MERKLE256K_CHUNK_BYTES {
        root.update(Sha256::digest(bytes));
        return format!("{:x}", root.finalize());
    }
    let n_chunks = bytes.len().div_ceil(WTIME_AOT_MERKLE256K_CHUNK_BYTES);
    let n_threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16).min(n_chunks);
    if n_threads <= 1 {
        for chunk in bytes.chunks(WTIME_AOT_MERKLE256K_CHUNK_BYTES) {
            root.update(Sha256::digest(chunk));
        }
        return format!("{:x}", root.finalize());
    }
    let mut leaves = vec![[0u8; 32]; n_chunks];
    let per = n_chunks.div_ceil(n_threads);
    let mut unspawned: Vec<(usize, usize)> = Vec::new();
    std::thread::scope(|s| {
        let mut rem = leaves.as_mut_slice();
        let mut chunk_idx = 0usize;
        while !rem.is_empty() {
            let take = per.min(rem.len());
            let (slot_slice, rest) = rem.split_at_mut(take);
            let start_chunk = chunk_idx;
            rem = rest;
            chunk_idx += take;
            if rem.is_empty() {
                for (i, slot) in slot_slice.iter_mut().enumerate() {
                    let off = (start_chunk + i) * WTIME_AOT_MERKLE256K_CHUNK_BYTES;
                    let end = (off + WTIME_AOT_MERKLE256K_CHUNK_BYTES).min(bytes.len());
                    let d = Sha256::digest(&bytes[off..end]);
                    slot.copy_from_slice(&d);
                }
            } else if std::thread::Builder::new()
                .spawn_scoped(s, move || {
                    for (i, slot) in slot_slice.iter_mut().enumerate() {
                        let off = (start_chunk + i) * WTIME_AOT_MERKLE256K_CHUNK_BYTES;
                        let end = (off + WTIME_AOT_MERKLE256K_CHUNK_BYTES).min(bytes.len());
                        let d = Sha256::digest(&bytes[off..end]);
                        slot.copy_from_slice(&d);
                    }
                })
                .is_err()
            {
                unspawned.push((start_chunk, take));
            }
        }
    });
    for (start_chunk, take) in unspawned {
        for (i, slot) in leaves[start_chunk..start_chunk + take].iter_mut().enumerate() {
            let off = (start_chunk + i) * WTIME_AOT_MERKLE256K_CHUNK_BYTES;
            let end = (off + WTIME_AOT_MERKLE256K_CHUNK_BYTES).min(bytes.len());
            let d = Sha256::digest(&bytes[off..end]);
            slot.copy_from_slice(&d);
        }
    }
    for leaf in &leaves {
        root.update(leaf);
    }
    format!("{:x}", root.finalize())
}

/// Content-addressing digest for a verified `.so` artifact: either legacy plain 64-hex SHA-256
/// or parallel 256 KiB chunk Merkle SHA-256 (`merkle256k:<64-hex>`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum VerifiedSoDigest {
    Sha256(String),
    Merkle256k(String),
}

impl VerifiedSoDigest {
    pub(super) fn parse(raw: &str) -> Option<Self> {
        let trimmed = raw.trim();
        if let Some(hex) = trimmed.strip_prefix(WTIME_AOT_MERKLE256K_PREFIX) {
            if hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Some(Self::Merkle256k(hex.to_ascii_lowercase()));
            }
            return None;
        }
        if trimmed.len() == 64 && trimmed.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Some(Self::Sha256(trimmed.to_ascii_lowercase()));
        }
        None
    }

    pub(super) fn filename(&self) -> String {
        match self {
            Self::Sha256(hex) => format!("{hex}.so"),
            Self::Merkle256k(hex) => format!("m-{hex}.so"),
        }
    }

    pub(super) fn tmp_stem(&self) -> String {
        match self {
            Self::Sha256(hex) => hex.clone(),
            Self::Merkle256k(hex) => format!("m-{hex}"),
        }
    }

    pub(super) fn line_text(&self) -> String {
        match self {
            Self::Sha256(hex) => hex.clone(),
            Self::Merkle256k(hex) => format!("{WTIME_AOT_MERKLE256K_PREFIX}{hex}"),
        }
    }

    pub(super) fn matches_bytes(&self, bytes: &[u8]) -> bool {
        match self {
            Self::Sha256(expected) => wasm_sha256_hex(bytes) == *expected,
            Self::Merkle256k(expected) => merkle256k_sha256_hex(bytes) == *expected,
        }
    }

    pub(super) fn matches_file(&self, path: &Path) -> bool {
        match self {
            Self::Sha256(expected) => file_sha256_hex(path).is_ok_and(|actual| actual == *expected),
            Self::Merkle256k(expected) => std::fs::read(path).is_ok_and(|b| merkle256k_sha256_hex(&b) == *expected),
        }
    }
}

/// Fast collision-resistant cache-key digest for large `.wasm` inputs (parallel `merkle256k`
/// above 256 KiB, plain SHA-256 at or below 256 KiB).
pub(super) fn fast_wasm_cache_digest(wasm: &[u8]) -> String {
    if wasm.len() > WTIME_AOT_MERKLE256K_CHUNK_BYTES {
        format!("m256k-{}", merkle256k_sha256_hex(wasm))
    } else {
        wasm_sha256_hex(wasm)
    }
}

/// Verifies a `wtime-aot` `.so.sha256` integrity sidecar line: either plain 64-hex SHA-256
/// (artifacts `<= 256 KiB`) or `merkle256k:<64-hex>` (artifacts `> 256 KiB`).
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn verify_artifact_sidecar_digest(bytes: &[u8], linear_sha256: &str, expected_trimmed: &str) -> bool {
    if let Some(expected_hex) = expected_trimmed.strip_prefix(WTIME_AOT_MERKLE256K_PREFIX) {
        expected_hex.len() == 64 && merkle256k_sha256_hex(bytes) == expected_hex
    } else {
        expected_trimmed.len() == 64 && linear_sha256 == expected_trimmed
    }
}

pub(super) fn reader_sha256_hex(f: &mut std::fs::File, path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        match f.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => hasher.update(&buf[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub(super) fn file_sha256_hex(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    reader_sha256_hex(&mut f, path)
}

/// Reads the calling thread's Linux proc status (`/proc/thread-self/status`, falling back to
/// `/proc/self/status` only when `/proc/thread-self/status` returns `NotFound`).
#[cfg(target_os = "linux")]
pub(super) fn read_thread_proc_status() -> std::io::Result<String> {
    match std::fs::read_to_string("/proc/thread-self/status") {
        Ok(s) => Ok(s),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => std::fs::read_to_string("/proc/self/status"),
        Err(e) => Err(e),
    }
}

/// Returns the current effective UID on Unix, failing closed if unavailable.
/// On Linux this parses `Uid:` field 2 (`euid`) from `/proc/thread-self/status` so it stays exact
/// for the calling thread even under `PR_SET_DUMPABLE=0` (where `/proc/self` reports `st_uid == 0`).
#[cfg(unix)]
pub(crate) fn current_euid() -> Result<u32> {
    #[cfg(target_os = "linux")]
    {
        let status = read_thread_proc_status().context("reading /proc/thread-self/status for effective uid")?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("Uid:") {
                let euid_str =
                    rest.split_whitespace().nth(1).ok_or_else(|| anyhow!("malformed Uid line in proc status"))?;
                return euid_str.parse::<u32>().context("parsing effective uid from proc status");
            }
        }
        bail!("missing Uid line in proc status")
    }
    #[cfg(not(target_os = "linux"))]
    {
        let out = Command::new("/usr/bin/id")
            .env_clear()
            .arg("-u")
            .output()
            .context("running /usr/bin/id -u for effective uid")?;
        ensure!(out.status.success(), "/usr/bin/id -u failed ({})", out.status);
        String::from_utf8_lossy(&out.stdout).trim().parse::<u32>().context("parsing effective uid from /usr/bin/id -u")
    }
}

#[cfg(unix)]
pub(super) fn validate_trusted_ancestor(anc: &Path, target: &Path, euid: u32) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let am = std::fs::symlink_metadata(anc).with_context(|| format!("lstat cache ancestor {}", anc.display()))?;
    ensure!(
        !am.file_type().is_symlink() && am.file_type().is_dir(),
        "cache ancestor {} (for {}) is not a real directory",
        anc.display(),
        target.display()
    );
    let uid = am.uid();
    ensure!(
        uid == 0 || uid == euid,
        "cache ancestor {} (for {}) is owned by untrusted uid {uid} (expected 0 or {euid})",
        anc.display(),
        target.display()
    );
    let mode = am.permissions().mode();
    ensure!(
        (mode & 0o022) == 0 || (mode & 0o1000) != 0,
        "cache ancestor {} (for {}) is writable without sticky bit (remediation: run `chmod 700` or set PEN_WTIME_AOT_CACHE to an owner-private directory)",
        anc.display(),
        target.display()
    );
    Ok(())
}

/// Enforces an owner-controlled (`0o700`), symlink-free private cache directory hierarchy
/// BEFORE performing any directory creation or `chmod`, returning the verified canonical path.
/// Fails closed if effective UID cannot be determined, `path` contains `..` or targets `/`,
/// traverses a non-root symlink, has untrusted/non-sticky-writable ancestors, targets a foreign-
/// owned or pre-existing group/world-writable directory, or fails `0o700` creation/tightening.
pub(crate) fn ensure_private_cache_dir(path: &Path) -> Result<PathBuf> {
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(path).with_context(|| format!("creating cache dir {}", path.display()))?;
        std::fs::canonicalize(path).with_context(|| format!("canonicalizing cache dir {}", path.display()))
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
        use std::path::Component;
        let euid = current_euid()?;
        let abs_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir().with_context(|| format!("resolving current_dir for {}", path.display()))?.join(path)
        };
        let mut existing = PathBuf::new();
        let mut missing: Vec<std::ffi::OsString> = Vec::new();
        let mut in_missing = false;
        for comp in abs_path.components() {
            match comp {
                Component::CurDir => continue,
                Component::ParentDir => {
                    bail!("cache path must not contain '..' components: {}", path.display());
                }
                Component::RootDir => {
                    ensure!(!in_missing, "malformed cache path: {}", path.display());
                    existing.push(comp.as_os_str());
                }
                Component::Prefix(_) => bail!("unsupported path prefix: {}", path.display()),
                Component::Normal(part) => {
                    if in_missing {
                        missing.push(part.to_os_string());
                        continue;
                    }
                    let candidate = existing.join(part);
                    match std::fs::symlink_metadata(&candidate) {
                        Ok(m) => {
                            if m.file_type().is_symlink() {
                                ensure!(
                                    m.uid() == 0,
                                    "cache hierarchy traverses non-root symlink {} (in {})",
                                    candidate.display(),
                                    path.display()
                                );
                            } else {
                                ensure!(
                                    m.file_type().is_dir(),
                                    "cache path component {} is not a directory",
                                    candidate.display()
                                );
                            }
                            existing = candidate;
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                            in_missing = true;
                            missing.push(part.to_os_string());
                        }
                        Err(e) => {
                            return Err(e).with_context(|| format!("lstat cache component {}", candidate.display()));
                        }
                    }
                }
            }
        }
        ensure!(!existing.as_os_str().is_empty(), "empty cache path: {}", path.display());
        let canon_existing =
            std::fs::canonicalize(&existing).with_context(|| format!("canonicalizing {}", existing.display()))?;
        if missing.is_empty() {
            ensure!(canon_existing.parent().is_some(), "refusing to use filesystem root as cache directory");
            let sym = std::fs::symlink_metadata(&existing)
                .with_context(|| format!("lstat cache dir {}", existing.display()))?;
            ensure!(
                !sym.file_type().is_symlink() && sym.file_type().is_dir(),
                "cache path must be a real directory (not a symlink): {}",
                path.display()
            );
            ensure!(
                sym.uid() == euid,
                "cache directory {} owner uid {} does not match effective uid {euid}",
                path.display(),
                sym.uid()
            );
            ensure!(
                sym.permissions().mode() & 0o022 == 0,
                "refusing pre-existing group/world-writable cache directory {} (mode {:o})",
                path.display(),
                sym.permissions().mode() & 0o7777
            );
            for anc in canon_existing.ancestors().skip(1) {
                validate_trusted_ancestor(anc, path, euid)?;
            }
            if sym.permissions().mode() & 0o077 != 0 {
                std::fs::set_permissions(&canon_existing, std::fs::Permissions::from_mode(0o700))
                    .with_context(|| format!("enforcing owner-only 0700 permissions on {}", path.display()))?;
            }
            let post = std::fs::symlink_metadata(&canon_existing)
                .with_context(|| format!("lstat cache dir {}", canon_existing.display()))?;
            ensure!(
                !post.file_type().is_symlink()
                    && post.file_type().is_dir()
                    && post.uid() == euid
                    && post.permissions().mode() & 0o077 == 0
                    && post.dev() == sym.dev()
                    && post.ino() == sym.ino(),
                "cache directory failed post-validation identity/mode checks: {}",
                path.display()
            );
            return Ok(canon_existing);
        }

        for anc in canon_existing.ancestors() {
            validate_trusted_ancestor(anc, path, euid)?;
        }
        let mut cur = canon_existing;
        for part in missing {
            cur.push(&part);
            let mut db = std::fs::DirBuilder::new();
            db.mode(0o700);
            match db.create(&cur) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let pre = std::fs::symlink_metadata(&cur)
                        .with_context(|| format!("lstat cache dir {}", cur.display()))?;
                    ensure!(
                        !pre.file_type().is_symlink()
                            && pre.file_type().is_dir()
                            && pre.uid() == euid
                            && pre.permissions().mode() & 0o022 == 0,
                        "concurrently created cache directory failed ownership/mode checks: {}",
                        cur.display()
                    );
                    if pre.permissions().mode() & 0o077 != 0 {
                        std::fs::set_permissions(&cur, std::fs::Permissions::from_mode(0o700))
                            .with_context(|| format!("enforcing 0700 permissions on {}", cur.display()))?;
                    }
                }
                Err(e) => return Err(e).with_context(|| format!("creating cache dir {}", cur.display())),
            }
            let post = std::fs::symlink_metadata(&cur).with_context(|| format!("lstat cache dir {}", cur.display()))?;
            ensure!(
                !post.file_type().is_symlink()
                    && post.file_type().is_dir()
                    && post.uid() == euid
                    && post.permissions().mode() & 0o077 == 0,
                "created cache directory failed 0700 ownership checks: {}",
                cur.display()
            );
        }
        let canon_leaf = std::fs::canonicalize(&cur)
            .with_context(|| format!("canonicalizing created cache dir {}", cur.display()))?;
        ensure!(canon_leaf == cur, "cache directory changed path during creation: {}", cur.display());
        Ok(canon_leaf)
    }
}

/// Checks that an existing verified `.so` is a non-symlink regular file owned by `euid` with no write bits set.
pub(super) fn is_immutable_regular_file(meta: &std::fs::Metadata) -> bool {
    if meta.file_type().is_symlink() || !meta.file_type().is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        if meta.permissions().mode() & 0o222 != 0 {
            return false;
        }
        if !current_euid().ok().is_some_and(|euid| meta.uid() == euid) {
            return false;
        }
    }
    true
}

/// Atomically publishes `bytes` into `verified_dir/<digest>.so` with `0o555` permissions via
/// `rename(&tmp, &dest)` (never `remove_file(&dest)` so there is no missing-file window).
pub(super) fn publish_immutable_verified_so(
    verified_dir: &Path,
    digest: &VerifiedSoDigest,
    bytes: &[u8],
    seq: u64,
) -> Result<PathBuf> {
    let dest = verified_dir.join(digest.filename());
    if let Ok(meta) = std::fs::symlink_metadata(&dest)
        && is_immutable_regular_file(&meta)
        && digest.matches_file(&dest)
    {
        return Ok(dest);
    }
    let tmp = verified_dir.join(format!(".{}.{}.{seq}.tmp", digest.tmp_stem(), std::process::id()));
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o600);
    let mut f = opts.open(&tmp).with_context(|| format!("creating immutable verified artifact {}", tmp.display()))?;
    if let Err(err) = f.write_all(bytes) {
        drop(f);
        let _ = std::fs::remove_file(&tmp);
        return Err(err.into());
    }
    drop(f);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Err(err) = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o555)) {
            let _ = std::fs::remove_file(&tmp);
            return Err(err).with_context(|| format!("marking verified artifact read-only {}", tmp.display()));
        }
    }
    if let Err(err) = std::fs::rename(&tmp, &dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err.into());
    }
    Ok(dest)
}

#[cfg(unix)]
pub(super) type BinStamp = (u64, u64, u64, i64, i64, i64, i64);
#[cfg(not(unix))]
pub(super) type BinStamp = (u64, u64, u32);

pub(super) fn bin_stamp_from_meta(meta: &std::fs::Metadata) -> BinStamp {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        (meta.dev(), meta.ino(), meta.len(), meta.mtime(), meta.mtime_nsec(), meta.ctime(), meta.ctime_nsec())
    }
    #[cfg(not(unix))]
    {
        let (secs, nanos) = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or((0, 0), |d| (d.as_secs(), d.subsec_nanos()));
        (meta.len(), secs, nanos)
    }
}

pub(super) fn bin_stamp(canon: &Path) -> Result<BinStamp> {
    let meta = std::fs::metadata(canon).with_context(|| format!("stat wtime-aot binary {}", canon.display()))?;
    Ok(bin_stamp_from_meta(&meta))
}

pub(super) fn read_private_digest_cache_entry(entry_path: &Path) -> Option<String> {
    const MAX_ENTRY_BYTES: u64 = 8192;
    let sym = std::fs::symlink_metadata(entry_path).ok()?;
    if sym.file_type().is_symlink() || !sym.file_type().is_file() || sym.len() > MAX_ENTRY_BYTES {
        return None;
    }
    #[cfg(unix)]
    let euid = current_euid().ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        if (sym.permissions().mode() & 0o022) != 0 || sym.uid() != euid {
            return None;
        }
    }
    let f = std::fs::File::open(entry_path).ok()?;
    let meta = f.metadata().ok()?;
    if !meta.file_type().is_file() || meta.len() > MAX_ENTRY_BYTES {
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        if meta.dev() != sym.dev()
            || meta.ino() != sym.ino()
            || meta.uid() != euid
            || (meta.permissions().mode() & 0o022) != 0
        {
            return None;
        }
    }
    let mut text = String::new();
    f.take(MAX_ENTRY_BYTES + 1).read_to_string(&mut text).ok()?;
    if (text.len() as u64) > MAX_ENTRY_BYTES {
        return None;
    }
    Some(text)
}

pub(super) fn bin_digest(bin: &Path, canon_cache: &Path) -> Result<(PathBuf, String)> {
    let canon =
        std::fs::canonicalize(bin).with_context(|| format!("canonicalizing wtime-aot binary {}", bin.display()))?;
    static CACHE: OnceLock<Mutex<std::collections::HashMap<(PathBuf, BinStamp), String>>> = OnceLock::new();
    let stamp = bin_stamp(&canon)?;
    let key = (canon.clone(), stamp);
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if let Some(hit) = cache.lock().unwrap_or_else(|p| p.into_inner()).get(&key).cloned() {
        return Ok((canon, hit));
    }

    // Fresh-process warm-path lookup persisted inside owner-private `canon_cache/bin-digests/`,
    // keyed strictly by canonical binary path + full `BinStamp` (`dev/ino/len/mtime/ctime`).
    // Cold builds still compute full pre- and post-build `file_sha256_hex` before publishing.
    let bin_cache_dir = ensure_private_cache_dir(&canon_cache.join("bin-digests"))?;
    let path_hash = wasm_sha256_hex(canon.as_os_str().as_encoded_bytes());
    let entry_path = bin_cache_dir.join(format!("{path_hash}.txt"));
    let expected_header = format!("v1|{}|{stamp:?}", canon.display());
    if let Some(text) = read_private_digest_cache_entry(&entry_path) {
        let mut lines = text.lines();
        if let (Some(hdr), Some(sha), None) = (lines.next(), lines.next(), lines.next())
            && hdr == expected_header
            && sha.len() == 64
            && sha.bytes().all(|b| b.is_ascii_hexdigit())
        {
            let digest = sha.to_string();
            cache.lock().unwrap_or_else(|p| p.into_inner()).insert(key, digest.clone());
            return Ok((canon, digest));
        }
    }

    // Open once and verify pre-read, post-read, and path `BinStamp` all match `stamp` before
    // binding `(canon, stamp) -> digest` in memory or on disk (prevents stamp-to-hash TOCTOU).
    let mut f = std::fs::File::open(&canon).with_context(|| format!("opening {}", canon.display()))?;
    let pre_meta = f.metadata().with_context(|| format!("fstat {}", canon.display()))?;
    let pre_stamp = bin_stamp_from_meta(&pre_meta);
    ensure!(
        pre_stamp == stamp,
        "wtime-aot binary {} changed before hashing; refusing to cache stale digest",
        canon.display()
    );
    let digest = reader_sha256_hex(&mut f, &canon)?;
    let post_meta = f.metadata().with_context(|| format!("fstat {}", canon.display()))?;
    let post_stamp = bin_stamp_from_meta(&post_meta);
    let path_stamp = bin_stamp(&canon)?;
    ensure!(
        post_stamp == stamp && path_stamp == stamp,
        "wtime-aot binary {} changed during hashing; refusing to cache stale digest",
        canon.display()
    );

    if !expected_header.contains('\n')
        && !expected_header.contains('\r')
        && let Ok((mut tmp_file, tmp_path)) = create_private_temp_file(&bin_cache_dir, &path_hash, ".tmp")
    {
        let ok = tmp_file.write_all(format!("{expected_header}\n{digest}\n").as_bytes()).is_ok();
        drop(tmp_file);
        if !ok || std::fs::rename(&tmp_path, &entry_path).is_err() {
            let _ = std::fs::remove_file(&tmp_path);
        }
    }
    cache.lock().unwrap_or_else(|p| p.into_inner()).insert(key, digest.clone());
    Ok((canon, digest))
}

/// Computes a deterministic 16-hex-char fingerprint of the adjacent `wtime-aot/rt` runtime crate
/// sources when using `--backend rust`, propagating read errors cleanly (matching `build.rs`).
pub(super) fn rt_sources_fingerprint(canon_bin: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let rt_dir = canon_bin
        .ancestors()
        .skip(1)
        .take(4)
        .map(|anc| anc.join("rt"))
        .find(|cand| cand.join("Cargo.toml").is_file() && cand.join("src").is_dir());
    let Some(rt_dir) = rt_dir else {
        return Ok("no-rt".to_string());
    };
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let mut pending = vec![rt_dir.join("Cargo.toml"), rt_dir.join("src")];
    while let Some(path) = pending.pop() {
        if path.is_dir() {
            let entries = std::fs::read_dir(&path)
                .with_context(|| format!("reading runtime source directory {}", path.display()))?;
            for entry in entries {
                let entry = entry.with_context(|| format!("iterating runtime source directory {}", path.display()))?;
                pending.push(entry.path());
            }
        } else {
            let bytes =
                std::fs::read(&path).with_context(|| format!("reading runtime source file {}", path.display()))?;
            let rel = path.strip_prefix(&rt_dir).unwrap_or(&path).to_string_lossy().into_owned();
            files.push((rel, bytes));
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let mut hasher = Sha256::new();
    for (rel, bytes) in files {
        hasher.update(rel.as_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    Ok(format!("{:x}", hasher.finalize())[..16].to_string())
}

pub(super) fn host_cpu_fingerprint() -> &'static str {
    static FP: OnceLock<String> = OnceLock::new();
    FP.get_or_init(|| {
        #[cfg_attr(not(target_os = "linux"), allow(unused_mut))]
        let mut raw = format!("arch={};", std::env::consts::ARCH);
        #[cfg(target_os = "linux")]
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if line.starts_with("model name") || line.starts_with("flags") || line.starts_with("Features") {
                    raw.push_str(line.trim());
                    raw.push('\n');
                }
                if line.trim().is_empty() && raw.len() > 32 {
                    break;
                }
            }
        }
        wasm_sha256_hex(raw.as_bytes())[..16].to_string()
    })
}

/// Resolves Pen's host-side build/lock timeout (`PEN_WTIME_AOT_BUILD_TIMEOUT_SECS`, default 120s).
pub(super) fn build_timeout() -> Result<Duration> {
    match std::env::var("PEN_WTIME_AOT_BUILD_TIMEOUT_SECS") {
        Ok(raw) if !raw.is_empty() => {
            let secs = raw
                .parse::<f64>()
                .ok()
                .filter(|s| s.is_finite() && *s > 0.0)
                .and_then(|s| Duration::try_from_secs_f64(s).ok())
                .ok_or_else(|| {
                    anyhow!("invalid PEN_WTIME_AOT_BUILD_TIMEOUT_SECS={raw:?} (expected positive number of seconds)")
                })?;
            Ok(secs)
        }
        _ => Ok(Duration::from_secs(120)),
    }
}

pub(super) fn ensure_before_deadline(deadline: Instant, phase: &str) -> Result<()> {
    ensure!(Instant::now() < deadline, "wtime-aot {phase} exceeded host build timeout");
    Ok(())
}

pub(super) fn lock_file_with_deadline(
    file: &std::fs::File,
    path: &Path,
    exclusive: bool,
    deadline: Instant,
) -> Result<()> {
    loop {
        let res = if exclusive { file.try_lock() } else { file.try_lock_shared() };
        match res {
            Ok(()) => return Ok(()),
            Err(std::fs::TryLockError::WouldBlock) => {
                ensure!(Instant::now() < deadline, "timed out waiting for lock {}", path.display());
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(std::fs::TryLockError::Error(e)) => {
                return Err(e).with_context(|| format!("locking {}", path.display()));
            }
        }
    }
}

#[cfg(unix)]
pub(super) fn kill_process_group(pgid: u32) {
    let kill_bin = if Path::new("/bin/kill").is_file() { "/bin/kill" } else { "/usr/bin/kill" };
    let group = format!("-{pgid}");
    let _ = Command::new(kill_bin)
        .env_clear()
        .args(["-KILL", "--", &group])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

pub(super) fn create_private_temp_file(dir: &Path, prefix: &str, suffix: &str) -> Result<(std::fs::File, PathBuf)> {
    static TMP_SEQ: AtomicU64 = AtomicU64::new(0);
    for _ in 0..32 {
        let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!(".{prefix}.{}.{seq}{suffix}", std::process::id()));
        let mut opts = std::fs::OpenOptions::new();
        opts.read(true).write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o600);
        match opts.open(&path) {
            Ok(f) => return Ok((f, path)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e).with_context(|| format!("creating private temp file {}", path.display())),
        }
    }
    bail!("failed to create unique private temp file under {}", dir.display())
}

pub(super) const MAX_CAPTURE_BYTES: usize = 16 << 20;
pub(super) const POST_EXIT_DRAIN_GRACE: Duration = Duration::from_secs(2);

pub(super) enum PipeCaptureErr {
    Overflow,
    Io(String),
}

pub(super) fn spawn_bounded_pipe_reader<R: std::io::Read + Send + 'static>(
    mut reader: R,
    limit: usize,
    overflowed: Arc<AtomicBool>,
) -> Result<mpsc::Receiver<Result<Vec<u8>, PipeCaptureErr>>> {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("pen-aot-pipe".into())
        .spawn(move || {
            let mut out = Vec::new();
            let mut chunk = [0u8; 8192];
            loop {
                match reader.read(&mut chunk) {
                    Ok(0) => {
                        let _ = tx.send(Ok(out));
                        return;
                    }
                    Ok(n) => {
                        let new_len = out.len().saturating_add(n);
                        if new_len > limit {
                            overflowed.store(true, Ordering::Release);
                            drop(reader);
                            let _ = tx.send(Err(PipeCaptureErr::Overflow));
                            return;
                        }
                        if new_len > out.capacity() {
                            // Never overshoot `limit` in allocated capacity.
                            let target = out.capacity().saturating_mul(2).max(new_len).min(limit);
                            out.reserve_exact(target - out.len());
                        }
                        out.extend_from_slice(&chunk[..n]);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => {
                        let _ = tx.send(Err(PipeCaptureErr::Io(e.to_string())));
                        return;
                    }
                }
            }
        })
        .context("spawning subprocess pipe reader thread")?;
    Ok(rx)
}

pub(super) fn recv_bounded_pipe(
    rx: mpsc::Receiver<Result<Vec<u8>, PipeCaptureErr>>,
    deadline: Instant,
    label: &str,
) -> Result<Vec<u8>> {
    let rem = deadline
        .checked_duration_since(Instant::now())
        .ok_or_else(|| anyhow!("{label} exceeded host build timeout"))?
        .min(POST_EXIT_DRAIN_GRACE);
    let res = match rx.recv_timeout(rem) {
        Ok(r) => r,
        Err(RecvTimeoutError::Timeout) => {
            if Instant::now() >= deadline {
                bail!("{label} exceeded host build timeout");
            }
            bail!("{label} output pipe did not close after exit (orphaned descendant holding pipe open)");
        }
        Err(RecvTimeoutError::Disconnected) => bail!("{label} capture reader disconnected"),
    };
    match res {
        Ok(buf) => Ok(buf),
        Err(PipeCaptureErr::Overflow) => bail!("{label} exceeded subprocess output cap ({MAX_CAPTURE_BYTES} bytes)"),
        Err(PipeCaptureErr::Io(e)) => bail!("{label} reading subprocess output: {e}"),
    }
}

/// Runs `cmd` in its own Unix process group, capturing stdout/stderr through bounded in-memory
/// pipes (zero disk amplification, hard `MAX_CAPTURE_BYTES` per-stream ceiling with exact
/// `reserve_exact` capacity cap and immediate `SIGPIPE` + process-group kill on overflow,
/// concurrent draining preventing pipe deadlocks, bounded post-exit `recv_timeout` preventing
/// descendant EOF hangs, and full read-error propagation).
pub(super) fn run_cmd_output_with_deadline(
    mut cmd: Command,
    _scratch_dir: &Path,
    label: &str,
    deadline: Instant,
) -> Result<std::process::Output> {
    ensure_before_deadline(deadline, label)?;
    cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped());
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
    let mut child = cmd.spawn().with_context(|| format!("spawning {label}"))?;
    let pid = child.id();
    let overflowed = Arc::new(AtomicBool::new(false));
    let cleanup_child = |child: &mut std::process::Child| {
        #[cfg(unix)]
        kill_process_group(pid);
        let _ = child.kill();
        let _ = child.wait();
    };
    let Some(stdout_pipe) = child.stdout.take() else {
        cleanup_child(&mut child);
        bail!("missing stdout pipe for {label}");
    };
    let Some(stderr_pipe) = child.stderr.take() else {
        cleanup_child(&mut child);
        bail!("missing stderr pipe for {label}");
    };
    let rx_out = match spawn_bounded_pipe_reader(stdout_pipe, MAX_CAPTURE_BYTES, Arc::clone(&overflowed)) {
        Ok(rx) => rx,
        Err(e) => {
            cleanup_child(&mut child);
            return Err(e);
        }
    };
    let rx_err = match spawn_bounded_pipe_reader(stderr_pipe, MAX_CAPTURE_BYTES, Arc::clone(&overflowed)) {
        Ok(rx) => rx,
        Err(e) => {
            cleanup_child(&mut child);
            return Err(e);
        }
    };
    let status = loop {
        if overflowed.load(Ordering::Acquire) {
            #[cfg(unix)]
            kill_process_group(pid);
            let _ = child.kill();
            let _ = child.wait();
            bail!("{label} exceeded subprocess output cap ({MAX_CAPTURE_BYTES} bytes)");
        }
        match child.try_wait() {
            Ok(Some(st)) => break st,
            Ok(None) => {
                if Instant::now() >= deadline {
                    #[cfg(unix)]
                    kill_process_group(pid);
                    let _ = child.kill();
                    let _ = child.wait();
                    bail!("{label} exceeded host build timeout");
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => {
                #[cfg(unix)]
                kill_process_group(pid);
                let _ = child.kill();
                let _ = child.wait();
                return Err(e).with_context(|| format!("waiting on {label}"));
            }
        }
    };
    #[cfg(unix)]
    kill_process_group(pid);
    ensure!(!overflowed.load(Ordering::Acquire), "{label} exceeded subprocess output cap ({MAX_CAPTURE_BYTES} bytes)");
    let stdout = recv_bounded_pipe(rx_out, deadline, label)?;
    let stderr = recv_bounded_pipe(rx_err, deadline, label)?;
    ensure_before_deadline(deadline, label)?;
    Ok(std::process::Output { status, stdout, stderr })
}

/// Default `wtime-aot` backend when `PEN_WTIME_AOT_BACKEND` is unset (`wtime` v0.1.7).
/// Uses native Linux `"aarch64"` on ARM64 Linux and `"x64"` elsewhere, falling back
/// automatically to `"c"` (`--c-opt auto --target-cpu native`) on [`X64FallbackError`].
#[cfg(all(target_os = "linux", target_arch = "aarch64", target_pointer_width = "64"))]
pub(super) const DEFAULT_AOT_BACKEND: &str = "aarch64";
#[cfg(not(all(target_os = "linux", target_arch = "aarch64", target_pointer_width = "64")))]
pub(super) const DEFAULT_AOT_BACKEND: &str = "x64";

/// Explicit compilation configuration threaded into `load_verified_so_with_config` so
/// default-path fallback (`x64 -> c --c-opt auto --target-cpu native`) never mutates env vars.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AotCompileConfig {
    pub(super) backend: String,
    pub(super) c_opt: String,
    pub(super) toolchain: String,
    pub(super) target_cpu: String,
    /// True when `PEN_WTIME_AOT_BACKEND` was explicitly set by the caller (disables fallback).
    pub(super) explicit_backend: bool,
}

impl AotCompileConfig {
    pub(super) fn from_env() -> Self {
        let explicit = std::env::var("PEN_WTIME_AOT_BACKEND").ok().filter(|s| !s.is_empty());
        let (backend, explicit_backend) = match explicit {
            Some(b) => (b, true),
            None => (DEFAULT_AOT_BACKEND.to_string(), false),
        };
        Self {
            backend,
            c_opt: std::env::var("PEN_WTIME_AOT_C_OPT")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "auto".to_string()),
            toolchain: std::env::var("PEN_WTIME_AOT_TOOLCHAIN")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "stable".to_string()),
            target_cpu: std::env::var("PEN_WTIME_AOT_TARGET_CPU")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "native".to_string()),
            explicit_backend,
        }
    }

    pub(super) fn x64_fallback_to_c(&self) -> Self {
        Self {
            backend: "c".to_string(),
            c_opt: self.c_opt.clone(),
            toolchain: self.toolchain.clone(),
            target_cpu: "native".to_string(),
            explicit_backend: true,
        }
    }
}

/// Runs `loader` with `cfg`, falling back cleanly to `--backend c --c-opt auto --target-cpu native`
/// ONLY when `cfg` selected `"x64"` as the implicit default (`!cfg.explicit_backend`) and `loader`
/// returned a typed [`X64FallbackError`]. Explicit `PEN_WTIME_AOT_BACKEND=x64` and non-fallback
/// errors (such as exit code `1` compiler/I/O/config failures) always propagate unchanged.
pub(super) fn load_verified_so_with_fallback<T, F>(
    wasm: &[u8],
    cache: Option<&Path>,
    cfg: &AotCompileConfig,
    mut loader: F,
) -> Result<T>
where
    F: FnMut(&[u8], Option<&Path>, &AotCompileConfig) -> Result<T>,
{
    match loader(wasm, cache, cfg) {
        Ok(lib) => Ok(lib),
        Err(err)
            if matches!(cfg.backend.as_str(), "x64" | "aarch64")
                && !cfg.explicit_backend
                && err.downcast_ref::<X64FallbackError>().is_some() =>
        {
            let fallback_cfg = cfg.x64_fallback_to_c();
            loader(wasm, cache, &fallback_cfg)
        }
        Err(err) => Err(err),
    }
}

pub(super) fn classify_build_failure(
    backend: &str,
    exit_code: Option<i32>,
    status_display: &str,
    stderr: &str,
) -> anyhow::Error {
    if matches!(backend, "x64" | "aarch64") && exit_code == Some(X64_ADMISSION_REJECTED_EXIT_CODE) {
        anyhow::Error::new(X64FallbackError(X64FallbackReason::AdmissionRejected(stderr.trim().to_string())))
    } else {
        anyhow!("wtime-aot build failed ({status_display}): {stderr}")
    }
}

/// Checks `fast_link` and loads the immutable content-addressed `verified/<digest>.so` while the
/// caller holds `canon_cache/.cache.lock` (shared or exclusive), keeping the verified artifact
/// completely stable across `dlopen`.
pub(super) fn try_load_warm_verified_so(
    fast_link: &Path,
    fast_key: &str,
    verified_dir: &Path,
    require_shstk: bool,
) -> Option<(PathBuf, Library)> {
    let recorded = std::fs::read_to_string(fast_link).ok()?;
    let mut lines = recorded.lines();
    let (rec_key, rec_so_digest_str, _rec_path) = (lines.next()?, lines.next()?, lines.next()?);
    if rec_key != fast_key {
        return None;
    }
    let rec_digest = VerifiedSoDigest::parse(rec_so_digest_str)?;
    let verified_so = verified_dir.join(rec_digest.filename());
    let sym = std::fs::symlink_metadata(&verified_so).ok()?;
    if !is_immutable_regular_file(&sym) {
        return None;
    }
    let canon_verified = std::fs::canonicalize(&verified_so).ok()?;
    if !canon_verified.starts_with(verified_dir) {
        return None;
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    if require_shstk {
        let bytes = std::fs::read(&canon_verified).ok()?;
        if !rec_digest.matches_bytes(&bytes) || !elf64_has_x86_shstk_property(&bytes) {
            return None;
        }
    } else if !rec_digest.matches_file(&canon_verified) {
        return None;
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let _ = require_shstk;
        if !rec_digest.matches_file(&canon_verified) {
            return None;
        }
    }
    // SAFETY: `canon_verified` is an immutable `0o555` content-addressed file inside our owner-only
    // `0o700` cache whose digest matches `rec_digest` while `.cache.lock` is held across `dlopen`.
    let lib = unsafe { Library::new(&canon_verified) }.ok()?;
    Some((canon_verified, lib))
}

pub(super) fn load_verified_so(wasm: &[u8], cache: Option<&Path>) -> Result<Library> {
    if let Ok(override_so) = std::env::var("PEN_WTIME_AOT_SO")
        && !override_so.is_empty()
    {
        // Explicit trusted-native-code override; canonicalize so a bare filename can never trigger
        // `dlopen`'s `LD_LIBRARY_PATH` search.
        let p = PathBuf::from(override_so);
        ensure!(p.is_file(), "PEN_WTIME_AOT_SO does not exist: {}", p.display());
        let canon =
            std::fs::canonicalize(&p).with_context(|| format!("canonicalizing PEN_WTIME_AOT_SO {}", p.display()))?;
        // SAFETY: explicit operator-supplied trusted native shared library override.
        return unsafe { Library::new(&canon) }
            .with_context(|| format!("loading PEN_WTIME_AOT_SO {}", canon.display()));
    }
    let cfg = AotCompileConfig::from_env();
    load_verified_so_with_fallback(wasm, cache, &cfg, load_verified_so_with_config)
}

pub(super) fn load_verified_so_with_config(
    wasm: &[u8],
    cache: Option<&Path>,
    cfg: &AotCompileConfig,
) -> Result<Library> {
    static NEXT_SEQ: AtomicU64 = AtomicU64::new(0);
    let backend = &cfg.backend;
    let c_opt = &cfg.c_opt;
    let toolchain = &cfg.toolchain;
    let target_cpu = &cfg.target_cpu;

    // Validate backend, target_cpu syntax (matching `build.rs`), build timeout, and host platform BEFORE touching the filesystem or hashing binaries.
    ensure!(
        !target_cpu.is_empty()
            && !target_cpu.starts_with('-')
            && target_cpu.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.')),
        "invalid PEN_WTIME_AOT_TARGET_CPU={target_cpu:?}"
    );
    let deadline = Instant::now() + build_timeout()?;
    let native_isa = match backend.as_str() {
        "c" => {
            ensure!(
                matches!(c_opt.as_str(), "auto" | "O2" | "O1" | "O0"),
                "unsupported PEN_WTIME_AOT_C_OPT={c_opt:?} (expected auto, O2, O1, or O0)"
            );
            None
        }
        "rust" => None,
        "x64" => Some(resolved_x64_isa_tag(target_cpu)?),
        "aarch64" => Some(resolved_aarch64_isa_tag(target_cpu)?),
        other => bail!("unsupported PEN_WTIME_AOT_BACKEND={other:?} (expected c, rust, x64, or aarch64)"),
    };

    let cache_dir = default_cache_dir(cache)?;
    let canon_cache = ensure_private_cache_dir(&cache_dir)?;
    let fast_dir = ensure_private_cache_dir(&canon_cache.join("by-wasm"))?;
    let verified_dir = ensure_private_cache_dir(&canon_cache.join("verified"))?;

    let bin = resolve_wtime_aot_bin().ok_or_else(|| {
        anyhow!(
            "wtime-aot binary not found (install wtime v0.1.7 with `just install-wtime-aot`, put `wtime-aot` on PATH, or set PEN_WTIME_AOT_BIN)"
        )
    })?;
    let (canon_bin, bin_hash) = bin_digest(&bin, &canon_cache)?;
    #[cfg(all(target_os = "linux", target_arch = "x86_64", target_pointer_width = "64"))]
    let require_c_shstk = backend == "c" && !matches!(linux_shstk_status(), ShstkStatus::ConfirmedOff);
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64", target_pointer_width = "64")))]
    let require_c_shstk = false;

    let mut resolved_cc: Option<ResolvedCcIdentity> = None;
    let mut resolved_rust: Option<(ResolvedRustcIdentity, String)> = None;
    let profile_tag = match backend.as_str() {
        "c" => {
            let cc_id = c_compiler_identity(target_cpu, &canon_cache, deadline)?;
            if require_c_shstk {
                ensure!(
                    cc_id.has_cet_return,
                    "C compiler/linker {} does not emit GNU_PROPERTY_X86_FEATURE_1_SHSTK required on CET-enabled/unknown host",
                    cc_id.canon_cc.display()
                );
            }
            let cc_ver_hash = wasm_sha256_hex(cc_id.cc_ver.as_bytes());
            let tag = format!(
                "c-{}-{}-{}-{}-{}-{}-{}-{}-cet{}-sc{}",
                sanitize_key_part(c_opt),
                sanitize_key_part(target_cpu),
                cc_ver_hash,
                cc_id.cc_target,
                cc_id.cc_sha256,
                cc_id.sys_headers_sha256,
                cc_id.linker_sha256,
                cc_id.link_env_sha256,
                u8::from(cc_id.has_cet_return),
                u8::from(cc_id.has_stack_clash),
            );
            resolved_cc = Some(cc_id);
            tag
        }
        "rust" => {
            let cpu_tag = if target_cpu == "native" {
                format!("native-{}", host_cpu_fingerprint())
            } else {
                sanitize_key_part(target_cpu)
            };
            let rust_id = rustc_identity(toolchain, &canon_cache, deadline, false)?;
            let rt_fp = rt_sources_fingerprint(&canon_bin)?;
            let tag = format!("rust-{}-{cpu_tag}-{}-{rt_fp}", sanitize_key_part(toolchain), rust_id.fp);
            resolved_rust = Some((rust_id, rt_fp));
            tag
        }
        "x64" => format!("x64-{}", native_isa.expect("validated above")),
        "aarch64" => format!("aarch64-{}", native_isa.expect("validated above")),
        _ => unreachable!(),
    };

    let digest = fast_wasm_cache_digest(wasm);
    let fast_key = format!("{digest}-{profile_tag}-{bin_hash}");
    // Hash `fast_key` to a fixed 64-hex-char filename so long profile/toolchain tags never exceed
    // Linux's 255-byte `NAME_MAX`, while still checking full `fast_key` exactness on line 1.
    let fast_link_name = wasm_sha256_hex(fast_key.as_bytes());
    let fast_link = fast_dir.join(format!("{fast_link_name}.path"));

    // Single cohesive cache publication RWLock (`canon_cache/.cache.lock`, strictly ordered BEFORE
    // any builder `entry/.publish.lock`), acquired with host-side deadline polling: shared lock
    // across warm verification + `dlopen`, exclusive lock across invalidation, rebuild, staging,
    // and cold `dlopen`.
    let cache_lock_path = canon_cache.join(".cache.lock");
    let cache_lock = std::fs::File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&cache_lock_path)
        .with_context(|| format!("opening cache lock {}", cache_lock_path.display()))?;
    lock_file_with_deadline(&cache_lock, &cache_lock_path, false, deadline)?;
    ensure_before_deadline(deadline, "warm cache check")?;
    if let Some((_path, lib)) = try_load_warm_verified_so(&fast_link, &fast_key, &verified_dir, require_c_shstk) {
        ensure_before_deadline(deadline, "warm cache verify")?;
        return Ok(lib);
    }
    cache_lock.unlock().with_context(|| format!("unlocking {}", cache_lock_path.display()))?;

    // Upgrade to exclusive cache publication lock for repair/build/staging/dlopen.
    lock_file_with_deadline(&cache_lock, &cache_lock_path, true, deadline)?;
    ensure_before_deadline(deadline, "exclusive cache check")?;
    if let Some((_path, lib)) = try_load_warm_verified_so(&fast_link, &fast_key, &verified_dir, require_c_shstk) {
        ensure_before_deadline(deadline, "exclusive cache verify")?;
        return Ok(lib);
    }

    // Mismatch or stale entry under the exclusive lock (no reader can be mid-`dlopen`):
    // restrict deletion strictly to `<canon_cache>/<entry>/library.so` (+ sidecar) and require
    // successful deadline-bounded `entry/.publish.lock` acquisition as a hard prerequisite.
    if let Ok(recorded) = std::fs::read_to_string(&fast_link) {
        let mut lines = recorded.lines();
        if let (Some(_), Some(_), Some(rec_path)) = (lines.next(), lines.next(), lines.next()) {
            let so_path = PathBuf::from(rec_path.trim());
            if let Ok(canon_so) = std::fs::canonicalize(&so_path)
                && canon_so.file_name() == Some(std::ffi::OsStr::new("library.so"))
                && let Some(entry_dir) = canon_so.parent()
                && entry_dir.parent() == Some(canon_cache.as_path())
            {
                let entry_lock_path = entry_dir.join(".publish.lock");
                if let Ok(entry_lock) =
                    std::fs::File::options().read(true).write(true).create(true).truncate(false).open(&entry_lock_path)
                    && lock_file_with_deadline(&entry_lock, &entry_lock_path, true, deadline).is_ok()
                {
                    let _ = std::fs::remove_file(&canon_so);
                    let _ = std::fs::remove_file(canon_so.with_extension("so.sha256"));
                    drop(entry_lock);
                }
            }
        }
        let _ = std::fs::remove_file(&fast_link);
    }

    // Capture exact compiler provenance BEFORE invoking it, and re-verify AFTER execution so a
    // concurrent compiler upgrade can never bind output from a new compiler to an old digest.
    let pre_stamp = bin_stamp(&canon_bin)?;
    let cold_bin_hash = file_sha256_hex(&canon_bin)?;
    let fast_key = format!("{digest}-{profile_tag}-{cold_bin_hash}");
    let fast_link_name = wasm_sha256_hex(fast_key.as_bytes());
    let fast_link = fast_dir.join(format!("{fast_link_name}.path"));

    let remaining = deadline
        .checked_duration_since(Instant::now())
        .ok_or_else(|| anyhow!("wtime-aot build exceeded host build timeout"))?;
    let time_limit_arg = format!("{:.3}", remaining.as_secs_f64().max(0.001));

    let seq = NEXT_SEQ.fetch_add(1, Ordering::Relaxed);
    let (mut wasm_file, tmp_wasm) = create_private_temp_file(&canon_cache, &format!("input-{digest}"), ".wasm")?;
    let _cleanup_wasm = RemoveOnDrop(tmp_wasm.clone());
    wasm_file.write_all(wasm)?;
    drop(wasm_file);
    let mut cmd = Command::new(&canon_bin);
    if let Some(path) = sanitized_path_env() {
        cmd.env("PATH", path);
    } else {
        cmd.env_remove("PATH");
    }
    // Keep `wtime-aot` and all compiler/linker children inside Pen's single process group so a
    // host timeout kills the entire build tree atomically.
    cmd.env("WTIME_AOT_INHERIT_PGID", "1").args(["build"]).arg(&tmp_wasm).args(["--cache"]).arg(&canon_cache).args([
        "--backend",
        backend.as_str(),
        "--blot-cdylib",
        "--target-cpu",
        target_cpu.as_str(),
        "--time-limit",
        &time_limit_arg,
    ]);
    match backend.as_str() {
        "c" => {
            cmd.args(["--c-opt", c_opt.as_str()]);
        }
        "rust" => {
            cmd.args(["--toolchain", toolchain.as_str()]);
        }
        "x64" | "aarch64" => {}
        _ => unreachable!(),
    }
    let output =
        run_cmd_output_with_deadline(cmd, &canon_cache, &format!("wtime-aot ({})", canon_bin.display()), deadline)?;
    drop(_cleanup_wasm);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(classify_build_failure(
            backend.as_str(),
            output.status.code(),
            &output.status.to_string(),
            &stderr,
        ));
    }
    ensure_before_deadline(deadline, "post-build verification")?;
    let post_stamp = bin_stamp(&canon_bin)?;
    let post_bin_hash = file_sha256_hex(&canon_bin)?;
    ensure!(
        pre_stamp == post_stamp && cold_bin_hash == post_bin_hash,
        "wtime-aot compiler binary {} changed during build; refusing to bind artifact to stale provenance",
        canon_bin.display()
    );
    if let Some(ref pre_cc) = resolved_cc {
        let cc_name = pre_cc.cc_name;
        for cand in ["clang", "cc", "gcc"] {
            if cand == cc_name {
                break;
            }
            if let Some(higher) = find_in_path(cand) {
                let mut ver_cmd = clean_cc_cmd(&higher);
                ver_cmd.arg("--version");
                let higher_out =
                    run_cmd_output_with_deadline(ver_cmd, &canon_cache, &format!("{cand} --version"), deadline)?;
                if higher_out.status.success() {
                    bail!(
                        "higher-precedence C compiler {cand} ({}) appeared on PATH during build (previously selected {cc_name}); refusing to bind artifact to stale provenance",
                        higher.display()
                    );
                }
            }
        }
        let post_found =
            find_in_path(cc_name).ok_or_else(|| anyhow!("C compiler {cc_name} disappeared from PATH during build"))?;
        let post_canon_cc = std::fs::canonicalize(&post_found)
            .with_context(|| format!("canonicalizing post-build C compiler {}", post_found.display()))?;
        let post_cc_stamp = bin_stamp(&post_canon_cc)?;
        let post_cc1 = resolve_driver_prog(&post_found, "cc1", false, &canon_cache, deadline)?;
        let post_as = resolve_driver_prog(&post_found, "as", true, &canon_cache, deadline)?;
        let mut post_cc_libs = probe_dynamic_shared_libs(&post_canon_cc, &canon_cache, deadline)?;
        if let Some((_, ref c1, _)) = post_cc1 {
            post_cc_libs.extend(probe_dynamic_shared_libs(c1, &canon_cache, deadline)?);
        }
        if let Some((_, ref a, _)) = post_as {
            post_cc_libs.extend(probe_dynamic_shared_libs(a, &canon_cache, deadline)?);
        }
        post_cc_libs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        post_cc_libs.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
        let post_cc_sha256 = hash_compiler_binaries(
            &post_found,
            &post_canon_cc,
            post_cc1.as_ref().map(|(_, c, _)| c.as_path()),
            post_as.as_ref().map(|(_, c, _)| c.as_path()),
            &post_cc_libs,
        )?;
        ensure!(
            post_found == pre_cc.cc_path
                && post_canon_cc == pre_cc.canon_cc
                && pre_cc.stamp == post_cc_stamp
                && post_cc1 == pre_cc.cc1_file
                && post_as == pre_cc.as_file
                && post_cc_sha256 == pre_cc.cc_sha256,
            "C compiler {cc_name} changed during build ({} -> {}); refusing to bind artifact to stale provenance",
            pre_cc.canon_cc.display(),
            post_canon_cc.display()
        );
        let post_lld_key = find_in_path("ld.lld")
            .and_then(|p| std::fs::canonicalize(&p).ok().map(|c| (p, c)))
            .and_then(|(p, c)| bin_stamp(&c).ok().map(|s| (p, c, s)));
        ensure!(
            post_lld_key == pre_cc.lld_key,
            "linker ld.lld changed on PATH during build; refusing to bind artifact to stale provenance"
        );
        let (post_linker_path, post_canon_linker, post_linker_stamp) =
            resolve_driver_linker(&post_found, pre_cc.has_lld, &canon_cache, deadline)?;
        let post_collect2 = resolve_driver_prog(&post_found, "collect2", false, &canon_cache, deadline)?;
        let mut post_ld_libs = probe_dynamic_shared_libs(&post_canon_linker, &canon_cache, deadline)?;
        if let Some((_, ref c2, _)) = post_collect2 {
            post_ld_libs.extend(probe_dynamic_shared_libs(c2, &canon_cache, deadline)?);
        }
        post_ld_libs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        post_ld_libs.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
        let post_linker_sha256 = hash_linker_binaries(
            &post_linker_path,
            &post_canon_linker,
            post_collect2.as_ref().map(|(_, c, _)| c.as_path()),
            &post_ld_libs,
        )?;
        ensure!(
            post_linker_path == pre_cc.linker_path
                && post_canon_linker == pre_cc.canon_linker
                && post_linker_stamp == pre_cc.linker_stamp
                && post_collect2 == pre_cc.collect2_file
                && post_linker_sha256 == pre_cc.linker_sha256,
            "driver-selected linker changed during build ({} -> {}); refusing to bind artifact to stale provenance",
            pre_cc.canon_linker.display(),
            post_canon_linker.display()
        );
        ensure!(
            verify_watched_files(&pre_cc.watched_files),
            "system headers, shared toolchain libraries, or C link inputs changed during build; refusing to bind artifact to stale provenance"
        );
    }
    if let Some((ref pre_rust, ref pre_rt_fp)) = resolved_rust {
        let post_rust = rustc_identity(toolchain, &canon_cache, deadline, true)?;
        let post_rt_fp = rt_sources_fingerprint(&canon_bin)?;
        ensure!(
            post_rust.proxy_path == pre_rust.proxy_path
                && post_rust.canon_real == pre_rust.canon_real
                && post_rust.real_stamp == pre_rust.real_stamp
                && post_rust.fp == pre_rust.fp
                && &post_rt_fp == pre_rt_fp,
            "Rust toolchain or runtime sources changed during build; refusing to bind artifact to stale provenance"
        );
    }
    let so_path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    ensure!(so_path.is_file(), "wtime-aot build did not produce {}", so_path.display());
    let canon_so = std::fs::canonicalize(&so_path)
        .with_context(|| format!("canonicalizing built library {}", so_path.display()))?;
    ensure!(
        canon_so.starts_with(&canon_cache),
        "wtime-aot build produced artifact outside cache root: {}",
        canon_so.display()
    );
    let so_meta =
        std::fs::symlink_metadata(&canon_so).with_context(|| format!("lstat built library {}", canon_so.display()))?;
    ensure!(
        is_immutable_regular_file(&so_meta),
        "wtime-aot build artifact failed immutable owner-file checks: {}",
        canon_so.display()
    );
    let so_bytes = std::fs::read(&canon_so).with_context(|| format!("reading built library {}", canon_so.display()))?;
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    if require_c_shstk {
        ensure!(
            elf64_has_x86_shstk_property(&so_bytes),
            "built C library {} is missing required GNU_PROPERTY_X86_FEATURE_1_SHSTK property for CET host",
            canon_so.display()
        );
    }
    let sidecar = canon_so.with_extension("so.sha256");
    let sidecar_meta = std::fs::symlink_metadata(&sidecar)
        .with_context(|| format!("wtime-aot build missing required digest sidecar {}", sidecar.display()))?;
    ensure!(
        !sidecar_meta.file_type().is_symlink() && sidecar_meta.file_type().is_file(),
        "digest sidecar {} is not a regular file",
        sidecar.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let euid = current_euid()?;
        ensure!(
            sidecar_meta.uid() == euid && (sidecar_meta.permissions().mode() & 0o022) == 0,
            "digest sidecar {} failed owner/mode checks",
            sidecar.display()
        );
    }
    let expected_sidecar =
        std::fs::read_to_string(&sidecar).with_context(|| format!("reading digest sidecar {}", sidecar.display()))?;
    let verified_digest = VerifiedSoDigest::parse(&expected_sidecar)
        .ok_or_else(|| anyhow!("malformed digest sidecar {}", sidecar.display()))?;
    ensure!(
        verified_digest.matches_bytes(&so_bytes),
        "built library digest mismatch with sidecar {}",
        sidecar.display()
    );
    let verified_so = publish_immutable_verified_so(&verified_dir, &verified_digest, &so_bytes, seq)?;
    if let Ok((mut link_file, tmp_link)) = create_private_temp_file(&fast_dir, &fast_link_name, ".tmp") {
        let written = link_file
            .write_all(format!("{fast_key}\n{}\n{}\n", verified_digest.line_text(), canon_so.display()).as_bytes());
        drop(link_file);
        if written.is_err() || std::fs::rename(&tmp_link, &fast_link).is_err() {
            let _ = std::fs::remove_file(&tmp_link);
        }
    }
    // SAFETY: `verified_so` was just built locally from validated Wasm, digest-verified against
    // its mandatory builder sidecar, staged immutable (`0o555`) inside our owner-only `0o700`
    // cache, and `.cache.lock` remains exclusively held across `Library::new`.
    unsafe { Library::new(&verified_so) }
        .with_context(|| format!("loading wtime-aot library {}", verified_so.display()))
}
