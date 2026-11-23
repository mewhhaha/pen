//! `wtime-aot` backend for Blot ABI 8 guest modules.
//!
//! Used by default for `pen run` builds when `wtime-aot` is available (disable
//! with `PEN_WTIME_AOT=0`), or for non-`dev_engine` engines when `PEN_WTIME_AOT=1`.
//! Compiles the guest Wasm module once via `wtime-aot build --backend c --c-opt auto --blot-cdylib`
//! (override backend with `PEN_WTIME_AOT_BACKEND=c|rust|x64`, C optimization mode
//! with `PEN_WTIME_AOT_C_OPT=auto|O2|O1|O0`, and x64 target CPU with
//! `PEN_WTIME_AOT_TARGET_CPU=native|x86-64-v2|x86-64-v3`) into a cached shared library,
//! then loads it with `libloading` and drives the exact same Blot ABI 8 lifecycle,
//! arena array copies, host callbacks, and snapshots as Wasmtime.

use std::any::Any;
use std::ffi::c_void;
#[cfg(feature = "compiler")]
use std::io::Read as _;
use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
#[cfg(feature = "compiler")]
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use libloading::Library;
use pen_runtime::abi::{CallbackType, GuestAbi, GuestFunction, INVOCATION_FINISH, Parameter, ValueType};
use pen_runtime::memory_image::{Captured, MemoryImage, PAGE};
use wasmtime::Val;

use crate::{
    Active, Argument, GuestImage, HOST_MODULE, LEFT_FAILING, LEFT_NORMALLY, Limits, OWNERSHIP_ENTER,
    OWNERSHIP_LEAVE, Probe, TEARDOWN, TOKENS, Value, array_range,
};

const DEFAULT_WTIME_AOT_BIN: &str = "/home/jacobt/src/wtime/tools/wtime-aot/target/release/wtime-aot";

fn parse_env_bool(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .map(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
}

fn find_in_path(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn resolve_wtime_aot_bin() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("PEN_WTIME_AOT_BIN")
        && !explicit.is_empty()
    {
        let p = PathBuf::from(&explicit);
        if p.is_file() {
            return Some(p);
        }
        return find_in_path(&explicit);
    }
    let default_path = PathBuf::from(DEFAULT_WTIME_AOT_BIN);
    if default_path.is_file() {
        return Some(default_path);
    }
    find_in_path("wtime-aot")
}

/// Returns whether the `wtime-aot` backend is explicitly forced on everywhere.
pub fn enabled() -> bool {
    parse_env_bool("PEN_WTIME_AOT").unwrap_or(false)
}

/// Returns whether `wtime-aot` should be used for `pen run` builds.
///
/// Defaults to `true` when `wtime-aot` (or `PEN_WTIME_AOT_SO`) is available,
/// unless opted out via `PEN_WTIME_AOT=0` / `false`.
pub fn preferred_for_run() -> bool {
    if let Some(explicit) = parse_env_bool("PEN_WTIME_AOT") {
        return explicit;
    }
    if std::env::var("PEN_WTIME_AOT_SO").ok().is_some_and(|v| !v.is_empty()) {
        return true;
    }
    resolve_wtime_aot_bin().is_some()
}

type HostFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u32, u64, u64, *mut u64) -> i32;

type FnAbiVersion = unsafe extern "C" fn() -> u32;
type FnTrapMessage = unsafe extern "C" fn(u8, *mut *const u8, *mut usize) -> i32;
type FnImportCount = unsafe extern "C" fn() -> u32;
type FnImportName = unsafe extern "C" fn(u32, *mut *const u8, *mut usize, *mut *const u8, *mut usize) -> i32;
type FnNew = unsafe extern "C" fn(*mut c_void, Option<HostFn>, u64, *mut u8) -> *mut c_void;
type FnDrop = unsafe extern "C" fn(*mut c_void);
type FnCallExport = unsafe extern "C" fn(*mut c_void, *const u8, usize, u64, *mut u64, *mut u8) -> i32;
type FnCallI32I32 = unsafe extern "C" fn(*mut c_void, i32, *mut i32, *mut u8) -> i32;
type FnCallVoidI32 = unsafe extern "C" fn(*mut c_void, *mut i32, *mut u8) -> i32;
type FnMemData = unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut usize);
type FnMemGrow = unsafe extern "C" fn(*mut c_void, u64) -> i64;
type FnGetGlobal = unsafe extern "C" fn(*mut c_void, *const u8, usize, *mut u64, *mut u8) -> i32;
type FnSetGlobal = unsafe extern "C" fn(*mut c_void, *const u8, usize, u64) -> i32;

struct AotSymbols {
    trap_message: FnTrapMessage,
    new: FnNew,
    drop: FnDrop,
    call_export: FnCallExport,
    call_allocate: FnCallI32I32,
    call_reset: FnCallI32I32,
    call_finish_invocation: FnCallVoidI32,
    mem_data: FnMemData,
    mem_grow: FnMemGrow,
    get_global: FnGetGlobal,
    set_global: FnSetGlobal,
    active_mem_data: FnMemData,
    active_allocate: FnCallI32I32,
    #[allow(dead_code)]
    active_reset: FnCallI32I32,
    active_get_global: FnGetGlobal,
}

struct AotLibrary {
    // Keep the loaded shared object alive for as long as any symbols or instances exist.
    _lib: Library,
    syms: AotSymbols,
    imports: Vec<CallbackType>,
    counter_names: Vec<String>,
    /// Declared memory 0 `(min_pages, max_pages)` parsed from the Wasm module, if any.
    memory_limits: Option<(u64, Option<u64>)>,
    /// Whether the module exports memory 0 as `"blot:memory"` (the memory `WasmtimeGuest` images).
    has_blot_memory_export: bool,
    /// Whether the module declares a mutable `externref` table (unsupported for exact snapshot rewind).
    has_mut_extern_table: bool,
}

// SAFETY: loaded C function pointers and immutable metadata are safe to share across threads.
unsafe impl Send for AotLibrary {}
unsafe impl Sync for AotLibrary {}

/// Dedicated exit code emitted by `wtime-aot --backend x64` when a Wasm module is cleanly
/// rejected by x64 admission checks (distinct from exit code `1` for usage/I/O/compiler errors),
/// so default-path fallback never needs to parse stderr.
pub const X64_ADMISSION_REJECTED_EXIT_CODE: i32 = 4;

/// Reason why `wtime-aot --backend x64` is not applicable to the current host or module,
/// enabling clean default-path fallback to `--backend c --c-opt auto` while `wasm` and `cache`
/// are still in scope (never swallowed when `PEN_WTIME_AOT_BACKEND=x64` is explicitly set).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum X64FallbackReason {
    /// Host OS, architecture, or pointer width is not Linux x86-64 LP64.
    UnsupportedPlatform(String),
    /// Host CPU lacks required x64 ISA features (SSE4.1+POPCNT / ABM+BMI1) or CET SHSTK is active.
    UnsupportedIsa(String),
    /// Module was rejected by `wtime-aot --backend x64` admission checks (exit code `4`).
    AdmissionRejected(String),
}

/// Typed error returned when `wtime-aot --backend x64` declines a host or module cleanly.
#[derive(Debug)]
pub struct X64FallbackError(pub X64FallbackReason);

impl std::fmt::Display for X64FallbackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            X64FallbackReason::UnsupportedPlatform(msg) => {
                write!(f, "unsupported host platform for wtime-aot x64: {msg}")
            }
            X64FallbackReason::UnsupportedIsa(msg) => {
                write!(f, "unsupported CPU ISA for wtime-aot x64: {msg}")
            }
            X64FallbackReason::AdmissionRejected(msg) => {
                write!(f, "module rejected by wtime-aot x64 admission: {msg}")
            }
        }
    }
}

impl std::error::Error for X64FallbackError {}

impl AotLibrary {
    fn trap_error(&self, code: u8) -> anyhow::Error {
        let mut ptr: *const u8 = std::ptr::null();
        let mut len: usize = 0;
        let ok = unsafe { (self.syms.trap_message)(code, &mut ptr, &mut len) };
        let msg = if ok == 0 && !ptr.is_null() {
            let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
            String::from_utf8_lossy(bytes).into_owned()
        } else {
            format!("code {code}")
        };
        if code == 22 {
            anyhow::Error::new(X64FallbackError(X64FallbackReason::UnsupportedIsa(msg)))
        } else {
            anyhow!("wtime-aot trap: {msg}")
        }
    }
}

#[derive(Clone)]
pub struct AotModule {
    lib: Arc<AotLibrary>,
}

fn default_cache_dir(cache: Option<&Path>) -> Result<PathBuf> {
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
    bail!("no owner-private wtime-aot cache directory available (set PEN_WTIME_AOT_CACHE, XDG_CACHE_HOME, HOME, or XDG_RUNTIME_DIR)")
}

#[cfg(feature = "compiler")]
fn wasm_sha256_hex(wasm: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(wasm))
}

#[cfg(feature = "compiler")]
fn file_sha256_hex(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(wasm_sha256_hex(&bytes))
}

/// Reads the calling thread's Linux proc status (`/proc/thread-self/status`, falling back to
/// `/proc/self/status` only when `/proc/thread-self/status` returns `NotFound`).
#[cfg(all(feature = "compiler", target_os = "linux"))]
fn read_thread_proc_status() -> std::io::Result<String> {
    match std::fs::read_to_string("/proc/thread-self/status") {
        Ok(s) => Ok(s),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => std::fs::read_to_string("/proc/self/status"),
        Err(e) => Err(e),
    }
}

/// Returns the current effective UID on Unix, failing closed if unavailable.
/// On Linux this parses `Uid:` field 2 (`euid`) from `/proc/thread-self/status` so it stays exact
/// for the calling thread even under `PR_SET_DUMPABLE=0` (where `/proc/self` reports `st_uid == 0`).
#[cfg(all(feature = "compiler", unix))]
pub(crate) fn current_euid() -> Result<u32> {
    #[cfg(target_os = "linux")]
    {
        let status = read_thread_proc_status().context("reading /proc/thread-self/status for effective uid")?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("Uid:") {
                let euid_str = rest
                    .split_whitespace()
                    .nth(1)
                    .ok_or_else(|| anyhow!("malformed Uid line in proc status"))?;
                return euid_str
                    .parse::<u32>()
                    .context("parsing effective uid from proc status");
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
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<u32>()
            .context("parsing effective uid from /usr/bin/id -u")
    }
}

#[cfg(all(feature = "compiler", unix))]
fn validate_trusted_ancestor(anc: &Path, target: &Path, euid: u32) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let am = std::fs::symlink_metadata(anc)
        .with_context(|| format!("lstat cache ancestor {}", anc.display()))?;
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
#[cfg(feature = "compiler")]
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
            std::env::current_dir()
                .with_context(|| format!("resolving current_dir for {}", path.display()))?
                .join(path)
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
        let canon_existing = std::fs::canonicalize(&existing)
            .with_context(|| format!("canonicalizing {}", existing.display()))?;
        if missing.is_empty() {
            ensure!(
                canon_existing.parent().is_some(),
                "refusing to use filesystem root as cache directory"
            );
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
            let post = std::fs::symlink_metadata(&cur)
                .with_context(|| format!("lstat cache dir {}", cur.display()))?;
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
#[cfg(feature = "compiler")]
fn is_immutable_regular_file(meta: &std::fs::Metadata) -> bool {
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

/// Atomically publishes `bytes` into `verified_dir/<so_sha>.so` with `0o555` permissions via
/// `rename(&tmp, &dest)` (never `remove_file(&dest)` so there is no missing-file window).
#[cfg(feature = "compiler")]
fn publish_immutable_verified_so(verified_dir: &Path, so_sha: &str, bytes: &[u8], seq: u64) -> Result<PathBuf> {
    let dest = verified_dir.join(format!("{so_sha}.so"));
    if let Ok(meta) = std::fs::symlink_metadata(&dest)
        && is_immutable_regular_file(&meta)
        && file_sha256_hex(&dest).is_ok_and(|actual| actual == so_sha)
    {
        return Ok(dest);
    }
    let tmp = verified_dir.join(format!(".{so_sha}.{}.{seq}.tmp", std::process::id()));
    let mut f = std::fs::File::options()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .with_context(|| format!("creating immutable verified artifact {}", tmp.display()))?;
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

#[cfg(all(feature = "compiler", unix))]
type BinStamp = (u64, u64, u64, i64, i64, i64, i64);
#[cfg(all(feature = "compiler", not(unix)))]
type BinStamp = (u64, u64, u32);

#[cfg(feature = "compiler")]
fn bin_stamp(canon: &Path) -> Result<BinStamp> {
    let meta = std::fs::metadata(canon).with_context(|| format!("stat wtime-aot binary {}", canon.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok((
            meta.dev(),
            meta.ino(),
            meta.len(),
            meta.mtime(),
            meta.mtime_nsec(),
            meta.ctime(),
            meta.ctime_nsec(),
        ))
    }
    #[cfg(not(unix))]
    {
        let (secs, nanos) = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or((0, 0), |d| (d.as_secs(), d.subsec_nanos()));
        Ok((meta.len(), secs, nanos))
    }
}

#[cfg(feature = "compiler")]
fn bin_digest(bin: &Path) -> Result<(PathBuf, String)> {
    let canon = std::fs::canonicalize(bin).with_context(|| format!("canonicalizing wtime-aot binary {}", bin.display()))?;
    static CACHE: OnceLock<Mutex<std::collections::HashMap<(PathBuf, BinStamp), String>>> = OnceLock::new();
    let stamp = bin_stamp(&canon)?;
    let key = (canon.clone(), stamp);
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if let Some(hit) = cache.lock().unwrap_or_else(|p| p.into_inner()).get(&key).cloned() {
        return Ok((canon, hit));
    }
    let digest = file_sha256_hex(&canon)?;
    cache.lock().unwrap_or_else(|p| p.into_inner()).insert(key, digest.clone());
    Ok((canon, digest))
}

/// Computes a deterministic 16-hex-char fingerprint of the adjacent `wtime-aot/rt` runtime crate
/// sources when using `--backend rust`, preventing stale Rust fast-cache hits after runtime edits.
#[cfg(feature = "compiler")]
fn rt_sources_fingerprint(canon_bin: &Path) -> String {
    use sha2::{Digest, Sha256};
    let rt_dir = canon_bin
        .ancestors()
        .skip(1)
        .take(4)
        .map(|anc| anc.join("rt"))
        .find(|cand| cand.join("Cargo.toml").is_file() && cand.join("src").is_dir());
    let Some(rt_dir) = rt_dir else {
        return "no-rt".to_string();
    };
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let mut pending = vec![rt_dir.join("Cargo.toml"), rt_dir.join("src")];
    while let Some(path) = pending.pop() {
        if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&path) {
                for entry in entries.flatten() {
                    pending.push(entry.path());
                }
            }
        } else if let Ok(bytes) = std::fs::read(&path) {
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
    format!("{:x}", hasher.finalize())[..16].to_string()
}

#[cfg(feature = "compiler")]
fn host_cpu_fingerprint() -> &'static str {
    static FP: OnceLock<String> = OnceLock::new();
    FP.get_or_init(|| {
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
#[cfg(feature = "compiler")]
fn build_timeout() -> Result<Duration> {
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

#[cfg(feature = "compiler")]
fn ensure_before_deadline(deadline: Instant, phase: &str) -> Result<()> {
    ensure!(
        Instant::now() < deadline,
        "wtime-aot {phase} exceeded host build timeout"
    );
    Ok(())
}

#[cfg(feature = "compiler")]
fn lock_file_with_deadline(file: &std::fs::File, path: &Path, exclusive: bool, deadline: Instant) -> Result<()> {
    loop {
        let res = if exclusive {
            file.try_lock()
        } else {
            file.try_lock_shared()
        };
        match res {
            Ok(()) => return Ok(()),
            Err(std::fs::TryLockError::WouldBlock) => {
                ensure!(
                    Instant::now() < deadline,
                    "timed out waiting for lock {}",
                    path.display()
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(std::fs::TryLockError::Error(e)) => {
                return Err(e).with_context(|| format!("locking {}", path.display()));
            }
        }
    }
}

/// Runs `cmd` in its own Unix process group, capturing stdout/stderr without pipe deadlocks and
/// terminating the entire process group if `deadline` expires.
#[cfg(feature = "compiler")]
fn run_cmd_output_with_deadline(mut cmd: Command, label: &str, deadline: Instant) -> Result<std::process::Output> {
    ensure_before_deadline(deadline, label)?;
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
    let mut child = cmd.spawn().with_context(|| format!("spawning {label}"))?;
    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();
    std::thread::scope(|scope| {
        let out_t = scope.spawn(move || {
            let mut v = Vec::new();
            if let Some(ref mut p) = stdout_pipe {
                let _ = p.read_to_end(&mut v);
            }
            v
        });
        let err_t = scope.spawn(move || {
            let mut v = Vec::new();
            if let Some(ref mut p) = stderr_pipe {
                let _ = p.read_to_end(&mut v);
            }
            v
        });
        let status = loop {
            match child.try_wait() {
                Ok(Some(st)) => break st,
                Ok(None) => {
                    if Instant::now() >= deadline {
                        #[cfg(unix)]
                        {
                            let group = format!("-{}", child.id());
                            let _ = Command::new("kill").args(["-KILL", "--", &group]).status();
                        }
                        let _ = child.kill();
                        let _ = child.wait();
                        let _ = out_t.join();
                        let _ = err_t.join();
                        bail!("{label} exceeded host build timeout");
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(e) => {
                    #[cfg(unix)]
                    {
                        let group = format!("-{}", child.id());
                        let _ = Command::new("kill").args(["-KILL", "--", &group]).status();
                    }
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = out_t.join();
                    let _ = err_t.join();
                    return Err(e).with_context(|| format!("waiting on {label}"));
                }
            }
        };
        let stdout = out_t.join().unwrap_or_default();
        let stderr = err_t.join().unwrap_or_default();
        Ok(std::process::Output { status, stdout, stderr })
    })
}

/// Computes the exact same `(cc_ver, full_64_hex_cc_target_digest, has_cet_return)` C compiler
/// identity as `wtime-aot`'s `cgen.rs` + `build.rs`.
///
/// Keyed by `(canon_cc, bin_stamp, host_cpu_fingerprint, target_cpu)` in BOTH an in-process
/// `OnceLock` cache AND an owner-private `<canon_cache>/cc-probe/` file so:
/// 1. Warm fresh-process `pen run` launches spawn ZERO compiler subprocesses (~20 us `stat`+read);
/// 2. Mid-process or cross-process C toolchain upgrades invalidate immediately (never stale);
/// 3. The hashed identity matches `wtime-aot/src/build.rs` 1:1.
#[cfg(feature = "compiler")]
type CcIdentityKey = (PathBuf, BinStamp, String, String);
#[cfg(feature = "compiler")]
type CcIdentityVal = (String, String, bool);
#[cfg(feature = "compiler")]
struct ResolvedCcIdentity {
    canon_cc: PathBuf,
    stamp: BinStamp,
    cc_ver: String,
    cc_target: String,
    has_cet_return: bool,
}

#[cfg(feature = "compiler")]
fn c_compiler_identity(target_cpu: &str, canon_cache: &Path, deadline: Instant) -> Result<ResolvedCcIdentity> {
    use sha2::{Digest, Sha256};
    static CACHE: OnceLock<Mutex<std::collections::HashMap<CcIdentityKey, CcIdentityVal>>> = OnceLock::new();
    let host_fp = host_cpu_fingerprint().to_string();
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let probe_dir = ensure_private_cache_dir(&canon_cache.join("cc-probe"))?;
    for cc in ["clang", "cc", "gcc"] {
        let Some(p) = find_in_path(cc) else { continue };
        let Ok(canon_cc) = std::fs::canonicalize(&p) else { continue };
        let Ok(stamp) = bin_stamp(&canon_cc) else { continue };
        let mem_key = (canon_cc.clone(), stamp, host_fp.clone(), target_cpu.to_string());
        if let Some((cc_ver, cc_target, has_cet_return)) =
            cache.lock().unwrap_or_else(|p| p.into_inner()).get(&mem_key).cloned()
        {
            return Ok(ResolvedCcIdentity { canon_cc, stamp, cc_ver, cc_target, has_cet_return });
        }

        let probe_key_line = format!(
            "v1|{}|{:?}|{}|{}",
            canon_cc.display(),
            stamp,
            host_fp,
            target_cpu
        );
        let probe_file = probe_dir.join(format!("{}.txt", wasm_sha256_hex(probe_key_line.as_bytes())));
        if let Ok(meta) = std::fs::symlink_metadata(&probe_file)
            && !meta.file_type().is_symlink()
            && meta.file_type().is_file()
        {
            let owner_and_mode_ok = {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::{MetadataExt, PermissionsExt};
                    (meta.permissions().mode() & 0o022) == 0
                        && current_euid().ok().is_some_and(|euid| meta.uid() == euid)
                }
                #[cfg(not(unix))]
                {
                    true
                }
            };
            if owner_and_mode_ok
                && let Ok(text) = std::fs::read_to_string(&probe_file)
            {
                let mut lines = text.lines();
                if let (Some(k), Some(ver), Some(digest), Some(cet)) =
                    (lines.next(), lines.next(), lines.next(), lines.next())
                    && k == probe_key_line
                    && digest.len() == 64
                    && digest.bytes().all(|b| b.is_ascii_hexdigit())
                    && (cet == "0" || cet == "1")
                {
                    let id = (ver.to_string(), digest.to_string(), cet == "1");
                    cache.lock().unwrap_or_else(|p| p.into_inner()).insert(mem_key, id.clone());
                    return Ok(ResolvedCcIdentity {
                        canon_cc,
                        stamp,
                        cc_ver: id.0,
                        cc_target: id.1,
                        has_cet_return: id.2,
                    });
                }
            }
        }

        let cc_str = canon_cc.to_string_lossy().into_owned();
        let clean_cc = || {
            let mut cmd = Command::new(&cc_str);
            cmd.env_clear();
            for var in ["PATH", "HOME", "TMPDIR"] {
                if let Some(v) = std::env::var_os(var) {
                    cmd.env(var, v);
                }
            }
            cmd
        };
        let mut ver_cmd = clean_cc();
        ver_cmd.arg("--version");
        let ver_out = run_cmd_output_with_deadline(ver_cmd, "C compiler --version probe", deadline)?;
        if !ver_out.status.success() {
            // Match `cgen::detect_cc_with_deadline`: fall through to next compiler if `--version` fails.
            continue;
        }
        let cc_ver = String::from_utf8_lossy(&ver_out.stdout)
            .lines()
            .next()
            .unwrap_or(&cc_str)
            .trim()
            .to_string();

        let mut target_cmd = clean_cc();
        target_cmd.args([&format!("-march={target_cpu}"), "-dM", "-E", "-x", "c", "-"]);
        let out = run_cmd_output_with_deadline(
            target_cmd,
            &format!("probing {cc_str} -march={target_cpu}"),
            deadline,
        )?;
        ensure!(
            out.status.success(),
            "{cc_str} -march={target_cpu} probe failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
        let mut lines: Vec<&str> = std::str::from_utf8(&out.stdout)
            .unwrap_or("")
            .lines()
            .filter(|l| l.starts_with("#define __"))
            .collect();
        lines.sort_unstable();
        let mut h = Sha256::new();
        h.update(b"cc-target-v1");
        h.update(target_cpu.as_bytes());
        for l in lines {
            h.update(l.as_bytes());
            h.update(b"\n");
        }
        let cc_target = format!("{:x}", h.finalize());

        #[cfg(target_arch = "x86_64")]
        let has_cet_return = {
            let mut cet_cmd = clean_cc();
            cet_cmd.args(["-x", "c", "-c", "-fcf-protection=return", "-o", "/dev/null", "-"]);
            run_cmd_output_with_deadline(cet_cmd, "probing -fcf-protection=return", deadline)?
                .status
                .success()
        };
        #[cfg(not(target_arch = "x86_64"))]
        let has_cet_return = false;

        static PROBE_SEQ: AtomicU64 = AtomicU64::new(0);
        let tmp = probe_dir.join(format!(
            ".probe.{}.{}.tmp",
            std::process::id(),
            PROBE_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let body = format!(
            "{probe_key_line}\n{cc_ver}\n{cc_target}\n{}\n",
            u8::from(has_cet_return)
        );
        if std::fs::write(&tmp, body).is_ok() {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
            }
            if std::fs::rename(&tmp, &probe_file).is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
        } else {
            let _ = std::fs::remove_file(&tmp);
        }

        let id = (cc_ver, cc_target, has_cet_return);
        cache.lock().unwrap_or_else(|p| p.into_inner()).insert(mem_key, id.clone());
        return Ok(ResolvedCcIdentity {
            canon_cc,
            stamp,
            cc_ver: id.0,
            cc_target: id.1,
            has_cet_return: id.2,
        });
    }
    bail!("no C compiler found (install clang or gcc)")
}

#[cfg(feature = "compiler")]
fn sanitize_key_part(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' })
        .collect()
}

#[cfg(feature = "compiler")]
struct RemoveOnDrop(PathBuf);
#[cfg(feature = "compiler")]
impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[cfg(feature = "compiler")]
type RustcFpKey = (PathBuf, BinStamp, String);

/// Probes `rustc -vV` under a cleaned environment and host deadline, memoized by canonical
/// `rustc` binary + file stamp + toolchain so mid-process toolchain upgrades invalidate cleanly.
#[cfg(feature = "compiler")]
fn rustc_version_fingerprint(toolchain: &str, deadline: Instant) -> Result<String> {
    static CACHE: OnceLock<Mutex<std::collections::HashMap<RustcFpKey, String>>> = OnceLock::new();
    let Some(rustc_path) = find_in_path("rustc") else {
        return Ok("no-rustc".to_string());
    };
    let canon_rustc = std::fs::canonicalize(&rustc_path)
        .with_context(|| format!("canonicalizing rustc {}", rustc_path.display()))?;
    let stamp = bin_stamp(&canon_rustc)?;
    let key = (canon_rustc.clone(), stamp, toolchain.to_string());
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if let Some(hit) = cache.lock().unwrap_or_else(|p| p.into_inner()).get(&key).cloned() {
        return Ok(hit);
    }
    let clean_rustc = || {
        let mut cmd = Command::new(&canon_rustc);
        cmd.env_clear();
        for var in ["PATH", "HOME", "RUSTUP_HOME", "CARGO_HOME"] {
            if let Some(v) = std::env::var_os(var) {
                cmd.env(var, v);
            }
        }
        cmd.env("RUSTUP_TOOLCHAIN", toolchain);
        cmd
    };
    let mut cmd = clean_rustc();
    cmd.arg("-vV");
    let out = run_cmd_output_with_deadline(cmd, "rustc -vV probe", deadline)?;
    let fp = if out.status.success() {
        wasm_sha256_hex(&out.stdout)[..16].to_string()
    } else {
        let mut fallback = Command::new(&canon_rustc);
        fallback.env_clear();
        for var in ["PATH", "HOME", "RUSTUP_HOME", "CARGO_HOME"] {
            if let Some(v) = std::env::var_os(var) {
                fallback.env(var, v);
            }
        }
        fallback.arg("-vV");
        let fb = run_cmd_output_with_deadline(fallback, "rustc -vV fallback probe", deadline)?;
        if fb.status.success() {
            wasm_sha256_hex(&fb.stdout)[..16].to_string()
        } else {
            "no-rustc".to_string()
        }
    };
    cache.lock().unwrap_or_else(|p| p.into_inner()).insert(key, fp.clone());
    Ok(fp)
}

#[cfg(all(feature = "compiler", target_os = "linux", target_arch = "x86_64", target_pointer_width = "64"))]
enum ShstkStatus {
    ConfirmedOff,
    Enabled,
    Unknown(String),
}

/// Inspects the calling Linux thread's CET hardware shadow-stack (`shstk`) state via
/// `/proc/thread-self/status` `x86_Thread_features:` (zero unsafe, zero syscalls).
/// Distinguishes confirmed-off from unavailable/indeterminate status so strict CET loaders never
/// fail opaquely behind a false "off" assumption.
#[cfg(all(feature = "compiler", target_os = "linux", target_arch = "x86_64", target_pointer_width = "64"))]
fn linux_shstk_status() -> ShstkStatus {
    let status = match read_thread_proc_status() {
        Ok(s) => s,
        Err(e) => return ShstkStatus::Unknown(format!("cannot read /proc/thread-self/status: {e}")),
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("x86_Thread_features:") {
            return if rest.split_whitespace().any(|tok| tok == "shstk") {
                ShstkStatus::Enabled
            } else {
                ShstkStatus::ConfirmedOff
            };
        }
    }
    // Pre-6.6 kernels omit `x86_Thread_features:`; confirm via `/proc/cpuinfo` that the kernel/CPU
    // does not expose user CET shadow stacks (`user_shstk` / `shstk`), otherwise fail closed.
    let cpuinfo = match std::fs::read_to_string("/proc/cpuinfo") {
        Ok(s) => s,
        Err(e) => {
            return ShstkStatus::Unknown(format!(
                "missing x86_Thread_features in proc status and cannot read /proc/cpuinfo: {e}"
            ));
        }
    };
    let mut saw_flags = false;
    for line in cpuinfo.lines() {
        if let Some(rest) = line.strip_prefix("flags")
            && let Some((_, flags)) = rest.split_once(':')
        {
            saw_flags = true;
            if flags.split_whitespace().any(|f| f == "user_shstk" || f == "shstk") {
                return ShstkStatus::Unknown(
                    "CPU/kernel advertises CET shstk but /proc/thread-self/status lacks x86_Thread_features".into(),
                );
            }
        }
    }
    if saw_flags {
        ShstkStatus::ConfirmedOff
    } else {
        ShstkStatus::Unknown("unable to confirm CET shstk status from procfs".into())
    }
}

/// Validates `--target-cpu` syntax independently of the host OS/architecture so an invalid CPU
/// name always fails loudly as a configuration error (never triggering platform/ISA fallback).
#[cfg(feature = "compiler")]
fn parse_x64_target_cpu_syntax(target_cpu: &str) -> Result<Option<&'static str>> {
    match target_cpu {
        "native" => Ok(None),
        "x86-64-v2" | "generic" => Ok(Some("x64-v2")),
        "x86-64-v3" | "x86-64-v4" | "haswell" | "broadwell" | "skylake" | "skylake-avx512" | "cascadelake"
        | "cooperlake" | "cannonlake" | "icelake-client" | "icelake-server" | "tigerlake" | "rocketlake"
        | "alderlake" | "raptorlake" | "sapphirerapids" | "emeraldrapids" | "graniterapids" => Ok(Some("x64-v3")),
        other
            if other
                .strip_prefix("znver")
                .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())) =>
        {
            Ok(Some("x64-v3"))
        }
        other => bail!(
            "unsupported PEN_WTIME_AOT_TARGET_CPU={other:?} (expected native, x86-64-v2, generic, x86-64-v3, x86-64-v4, znver<N>, or haswell+)"
        ),
    }
}

/// Canonical internal cache profile tag matching `wtime-aot --backend x64 --target-cpu`,
/// validating `target_cpu` syntax FIRST (unconditionally across all platforms), then checking host
/// platform (`Linux x86-64 LP64`), calling-thread CET `SHSTK` status (failing closed when enabled
/// or indeterminate), and CPU capability (`SSE4.1 + POPCNT`, plus `ABM/LZCNT + BMI1` for `x64-v3`)
/// BEFORE cache lookup so unsupported hosts fail with a typed `X64FallbackError` while `wasm` &
/// `cache` are still available for default-path fallback (and bad CPU names always fail loudly).
#[cfg(feature = "compiler")]
fn resolved_x64_isa_tag(target_cpu: &str) -> Result<&'static str> {
    let explicit_tag = parse_x64_target_cpu_syntax(target_cpu)?;
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64", target_pointer_width = "64")))]
    {
        let _ = explicit_tag;
        Err(anyhow::Error::new(X64FallbackError(X64FallbackReason::UnsupportedPlatform(
            "wtime-aot --backend x64 is only supported on Linux x86_64 LP64 hosts".into(),
        ))))
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64", target_pointer_width = "64"))]
    {
        match linux_shstk_status() {
            ShstkStatus::ConfirmedOff => {}
            ShstkStatus::Enabled => {
                return Err(anyhow::Error::new(X64FallbackError(X64FallbackReason::UnsupportedIsa(
                    "calling thread has Linux CET hardware shadow stack (SHSTK) enabled".into(),
                ))));
            }
            ShstkStatus::Unknown(why) => {
                return Err(anyhow::Error::new(X64FallbackError(X64FallbackReason::UnsupportedIsa(why))));
            }
        }
        if !std::is_x86_feature_detected!("sse4.1") || !std::is_x86_feature_detected!("popcnt") {
            return Err(anyhow::Error::new(X64FallbackError(X64FallbackReason::UnsupportedIsa(
                "host CPU lacks baseline SSE4.1 + POPCNT required by wtime-aot --backend x64".into(),
            ))));
        }
        let tag = match explicit_tag {
            Some(t) => t,
            None => {
                if std::is_x86_feature_detected!("lzcnt") && std::is_x86_feature_detected!("bmi1") {
                    "x64-v3"
                } else {
                    "x64-v2"
                }
            }
        };
        if tag == "x64-v3" && (!std::is_x86_feature_detected!("lzcnt") || !std::is_x86_feature_detected!("bmi1")) {
            return Err(anyhow::Error::new(X64FallbackError(X64FallbackReason::UnsupportedIsa(format!(
                "host CPU lacks ABM/LZCNT + BMI1 required by PEN_WTIME_AOT_TARGET_CPU={target_cpu:?}"
            )))));
        }
        Ok(tag)
    }
}

#[cfg(not(feature = "compiler"))]
fn load_verified_so(_wasm: &[u8], _cache: Option<&Path>) -> Result<Library> {
    bail!("wtime-aot compilation requires blot-host's `compiler` feature")
}

/// Checks `fast_link` and loads the immutable content-addressed `verified/<sha256>.so` while the
/// caller holds `canon_cache/.cache.lock` (shared or exclusive), keeping the verified artifact
/// completely stable across `dlopen`.
#[cfg(feature = "compiler")]
fn try_load_warm_verified_so(
    fast_link: &Path,
    fast_key: &str,
    verified_dir: &Path,
) -> Option<(PathBuf, Library)> {
    let recorded = std::fs::read_to_string(fast_link).ok()?;
    let mut lines = recorded.lines();
    let (rec_key, rec_so_sha, _rec_path) = (lines.next()?, lines.next()?, lines.next()?);
    if rec_key != fast_key || rec_so_sha.len() != 64 || !rec_so_sha.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let verified_so = verified_dir.join(format!("{rec_so_sha}.so"));
    let sym = std::fs::symlink_metadata(&verified_so).ok()?;
    if !is_immutable_regular_file(&sym) {
        return None;
    }
    let canon_verified = std::fs::canonicalize(&verified_so).ok()?;
    if !canon_verified.starts_with(verified_dir) {
        return None;
    }
    if !file_sha256_hex(&canon_verified).is_ok_and(|actual| actual == rec_so_sha) {
        return None;
    }
    // SAFETY: `canon_verified` is an immutable `0o555` content-addressed file inside our owner-only
    // `0o700` cache whose SHA-256 matches `rec_so_sha` while `.cache.lock` is held across `dlopen`.
    let lib = unsafe { Library::new(&canon_verified) }.ok()?;
    Some((canon_verified, lib))
}

#[cfg(feature = "compiler")]
fn load_verified_so(wasm: &[u8], cache: Option<&Path>) -> Result<Library> {
    static NEXT_SEQ: AtomicU64 = AtomicU64::new(0);
    if let Ok(override_so) = std::env::var("PEN_WTIME_AOT_SO")
        && !override_so.is_empty()
    {
        // Explicit trusted-native-code override; canonicalize so a bare filename can never trigger
        // `dlopen`'s `LD_LIBRARY_PATH` search.
        let p = PathBuf::from(override_so);
        ensure!(p.is_file(), "PEN_WTIME_AOT_SO does not exist: {}", p.display());
        let canon = std::fs::canonicalize(&p)
            .with_context(|| format!("canonicalizing PEN_WTIME_AOT_SO {}", p.display()))?;
        // SAFETY: explicit operator-supplied trusted native shared library override.
        return unsafe { Library::new(&canon) }
            .with_context(|| format!("loading PEN_WTIME_AOT_SO {}", canon.display()));
    }
    let backend = std::env::var("PEN_WTIME_AOT_BACKEND")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "c".to_string());
    let c_opt = std::env::var("PEN_WTIME_AOT_C_OPT")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "auto".to_string());
    let toolchain = std::env::var("PEN_WTIME_AOT_TOOLCHAIN")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "stable".to_string());
    let target_cpu = std::env::var("PEN_WTIME_AOT_TARGET_CPU")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "native".to_string());

    // Validate backend, build timeout, and target/host platform BEFORE touching the filesystem or hashing binaries.
    let deadline = Instant::now() + build_timeout()?;
    let x64_isa = match backend.as_str() {
        "c" | "rust" => None,
        "x64" => Some(resolved_x64_isa_tag(&target_cpu)?),
        other => bail!("unsupported PEN_WTIME_AOT_BACKEND={other:?} (expected c, rust, or x64)"),
    };

    let cache_dir = default_cache_dir(cache)?;
    let canon_cache = ensure_private_cache_dir(&cache_dir)?;
    let fast_dir = ensure_private_cache_dir(&canon_cache.join("by-wasm"))?;
    let verified_dir = ensure_private_cache_dir(&canon_cache.join("verified"))?;

    let bin = resolve_wtime_aot_bin().unwrap_or_else(|| PathBuf::from(DEFAULT_WTIME_AOT_BIN));
    let (canon_bin, bin_hash) = bin_digest(&bin)?;
    let mut resolved_cc: Option<(PathBuf, BinStamp)> = None;
    let profile_tag = match backend.as_str() {
        "c" => {
            let cc_id = c_compiler_identity(&target_cpu, &canon_cache, deadline)?;
            let cc_ver_hash = wasm_sha256_hex(cc_id.cc_ver.as_bytes());
            resolved_cc = Some((cc_id.canon_cc, cc_id.stamp));
            format!(
                "c-{}-{}-{}-{}-cet{}",
                sanitize_key_part(&c_opt),
                sanitize_key_part(&target_cpu),
                cc_ver_hash,
                cc_id.cc_target,
                u8::from(cc_id.has_cet_return)
            )
        }
        "rust" => {
            let cpu_tag = if target_cpu == "native" {
                format!("native-{}", host_cpu_fingerprint())
            } else {
                sanitize_key_part(&target_cpu)
            };
            format!(
                "rust-{}-{cpu_tag}-{}-{}",
                sanitize_key_part(&toolchain),
                rustc_version_fingerprint(&toolchain, deadline)?,
                rt_sources_fingerprint(&canon_bin)
            )
        }
        "x64" => format!("x64-{}", x64_isa.expect("validated above")),
        _ => unreachable!(),
    };

    let digest = wasm_sha256_hex(wasm);
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
    if let Some((_path, lib)) = try_load_warm_verified_so(&fast_link, &fast_key, &verified_dir) {
        ensure_before_deadline(deadline, "warm cache verify")?;
        return Ok(lib);
    }
    cache_lock
        .unlock()
        .with_context(|| format!("unlocking {}", cache_lock_path.display()))?;

    // Upgrade to exclusive cache publication lock for repair/build/staging/dlopen.
    lock_file_with_deadline(&cache_lock, &cache_lock_path, true, deadline)?;
    ensure_before_deadline(deadline, "exclusive cache check")?;
    if let Some((_path, lib)) = try_load_warm_verified_so(&fast_link, &fast_key, &verified_dir) {
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
                if let Ok(entry_lock) = std::fs::File::options()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(&entry_lock_path)
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
    let tmp_wasm = canon_cache.join(format!("input-{digest}-{}-{seq}.wasm", std::process::id()));
    std::fs::write(&tmp_wasm, wasm)?;
    let _cleanup_wasm = RemoveOnDrop(tmp_wasm.clone());
    let mut cmd = Command::new(&canon_bin);
    cmd.args(["build"])
        .arg(&tmp_wasm)
        .args(["--cache"])
        .arg(&canon_cache)
        .args([
            "--backend",
            &backend,
            "--blot-cdylib",
            "--target-cpu",
            &target_cpu,
            "--time-limit",
            &time_limit_arg,
        ]);
    match backend.as_str() {
        "c" => {
            cmd.args(["--c-opt", &c_opt]);
        }
        "rust" => {
            cmd.args(["--toolchain", &toolchain]);
        }
        "x64" => {}
        _ => unreachable!(),
    }
    let output = run_cmd_output_with_deadline(cmd, &format!("wtime-aot ({})", canon_bin.display()), deadline)?;
    drop(_cleanup_wasm);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if backend == "x64" && output.status.code() == Some(X64_ADMISSION_REJECTED_EXIT_CODE) {
            return Err(anyhow::Error::new(X64FallbackError(X64FallbackReason::AdmissionRejected(
                stderr.trim().to_string(),
            ))));
        }
        bail!("wtime-aot build failed ({}): {stderr}", output.status);
    }
    ensure_before_deadline(deadline, "post-build verification")?;
    let post_stamp = bin_stamp(&canon_bin)?;
    let post_bin_hash = file_sha256_hex(&canon_bin)?;
    ensure!(
        pre_stamp == post_stamp && cold_bin_hash == post_bin_hash,
        "wtime-aot compiler binary {} changed during build; refusing to bind artifact to stale provenance",
        canon_bin.display()
    );
    if let Some((ref canon_cc, pre_cc_stamp)) = resolved_cc {
        let post_cc_stamp = bin_stamp(canon_cc)?;
        ensure!(
            pre_cc_stamp == post_cc_stamp,
            "C compiler binary {} changed during build; refusing to bind artifact to stale provenance",
            canon_cc.display()
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
    let so_meta = std::fs::symlink_metadata(&canon_so)
        .with_context(|| format!("lstat built library {}", canon_so.display()))?;
    ensure!(
        is_immutable_regular_file(&so_meta),
        "wtime-aot build artifact failed immutable owner-file checks: {}",
        canon_so.display()
    );
    let so_bytes = std::fs::read(&canon_so)
        .with_context(|| format!("reading built library {}", canon_so.display()))?;
    let so_sha = wasm_sha256_hex(&so_bytes);
    let sidecar = canon_so.with_extension("so.sha256");
    let expected_sidecar = std::fs::read_to_string(&sidecar)
        .with_context(|| format!("wtime-aot build missing required digest sidecar {}", sidecar.display()))?;
    ensure!(
        expected_sidecar.trim() == so_sha,
        "built library digest mismatch with sidecar {}",
        sidecar.display()
    );
    let verified_so = publish_immutable_verified_so(&verified_dir, &so_sha, &so_bytes, seq)?;
    let tmp_link = fast_dir.join(format!(".{fast_link_name}.{}.{seq}.tmp", std::process::id()));
    if let Ok(mut link_file) = std::fs::File::options().write(true).create_new(true).open(&tmp_link) {
        let written = link_file.write_all(format!("{fast_key}\n{so_sha}\n{}\n", canon_so.display()).as_bytes());
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

#[cfg(feature = "compiler")]
fn counter_export_names(wasm: &[u8], abi: &GuestAbi) -> Vec<String> {
    let mut names = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::ExportSection(section) = payload {
            for export in section.into_iter().flatten() {
                if export.kind == wasmparser::ExternalKind::Global
                    && export.name.starts_with("blot:")
                    && abi.constants.iter().all(|c| c.name != export.name)
                {
                    names.push(export.name.to_owned());
                }
            }
        }
    }
    names
}

#[cfg(not(feature = "compiler"))]
fn counter_export_names(_wasm: &[u8], _abi: &GuestAbi) -> Vec<String> {
    Vec::new()
}

#[cfg(feature = "compiler")]
fn wasmparser_val(value: ValueType) -> wasmparser::ValType {
    match value {
        ValueType::F32 => wasmparser::ValType::F32,
        _ => wasmparser::ValType::I32,
    }
}

struct ValidatedAotWasm {
    memory_limits: Option<(u64, Option<u64>)>,
    has_blot_memory_export: bool,
    has_mut_extern_table: bool,
    expected_imports: Option<Vec<CallbackType>>,
}

/// Validates every import, entrypoint, constant, arena, ownership, and finish-invocation
/// Wasm signature before cache lookup or compilation, matching Wasmtime's exact ABI checks.
#[cfg(feature = "compiler")]
fn validate_aot_wasm(wasm: &[u8], abi: &GuestAbi) -> Result<ValidatedAotWasm> {
    let callbacks = abi.callbacks();
    let mut types: Vec<Option<wasmparser::FuncType>> = Vec::new();
    let mut func_type_idxs: Vec<u32> = Vec::new();
    let mut expected_imports: Vec<CallbackType> = Vec::new();
    let mut global_types: Vec<wasmparser::ValType> = Vec::new();
    let mut memory_limits: Option<(u64, Option<u64>)> = None;
    let mut has_mut_extern_table = false;
    let mut exports = std::collections::HashMap::new();

    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        match payload? {
            wasmparser::Payload::TypeSection(section) => {
                for group in section {
                    for st in group?.into_types() {
                        let fty = match st.composite_type.inner {
                            wasmparser::CompositeInnerType::Func(f) => Some(f),
                            _ => None,
                        };
                        types.push(fty);
                    }
                }
            }
            wasmparser::Payload::ImportSection(section) => {
                for import in section.into_imports() {
                    let import = import?;
                    ensure!(import.module == HOST_MODULE, "invalid_abi: unexpected import {}::{}", import.module, import.name);
                    let cb = callbacks
                        .iter()
                        .copied()
                        .find(|cb| cb.import_name() == import.name)
                        .ok_or_else(|| anyhow!("invalid_abi: unexpected import {}::{}", import.module, import.name))?;
                    let wasmparser::TypeRef::Func(ti) = import.ty else {
                        bail!("invalid_abi: import {}::{} must be a plain function", import.module, import.name);
                    };
                    let fty = types
                        .get(ti as usize)
                        .and_then(|o| o.as_ref())
                        .ok_or_else(|| anyhow!("invalid_abi: import {}::{} has invalid type index {ti}", import.module, import.name))?;
                    let expected_params = [wasmparser::ValType::EXTERNREF, wasmparser_val(cb.parameter)];
                    let expected_res = [wasmparser_val(cb.result)];
                    ensure!(
                        fty.params() == expected_params && fty.results() == expected_res,
                        "invalid_abi: import {}::{} signature mismatch (got {:?} -> {:?}, expected {:?} -> {:?})",
                        import.module,
                        import.name,
                        fty.params(),
                        fty.results(),
                        expected_params,
                        expected_res
                    );
                    expected_imports.push(cb);
                    func_type_idxs.push(ti);
                }
            }
            wasmparser::Payload::FunctionSection(section) => {
                for ti in section {
                    func_type_idxs.push(ti?);
                }
            }
            wasmparser::Payload::TableSection(section) => {
                for table in section {
                    let table = table?;
                    if table.ty.element_type.is_extern_ref() {
                        has_mut_extern_table = true;
                    }
                }
            }
            wasmparser::Payload::MemorySection(section) => {
                for mem in section {
                    let mem = mem?;
                    ensure!(
                        memory_limits.is_none(),
                        "invalid_abi: wtime-aot supports at most one linear memory"
                    );
                    ensure!(!mem.memory64, "invalid_abi: wtime-aot does not support memory64");
                    ensure!(!mem.shared, "invalid_abi: wtime-aot does not support shared memories");
                    ensure!(
                        mem.page_size_log2.is_none_or(|p| p == 16),
                        "invalid_abi: wtime-aot requires standard 64 KiB memory pages"
                    );
                    memory_limits = Some((mem.initial, mem.maximum));
                }
            }
            wasmparser::Payload::GlobalSection(section) => {
                for g in section {
                    let g = g?;
                    global_types.push(g.ty.content_type);
                }
            }
            wasmparser::Payload::ExportSection(section) => {
                for export in section {
                    let export = export?;
                    exports.insert(export.name.to_owned(), (export.kind, export.index));
                }
            }
            _ => {}
        }
    }

    let lookup_func_sig = |fidx: u32| -> Option<&wasmparser::FuncType> {
        func_type_idxs
            .get(fidx as usize)
            .and_then(|&ti| types.get(ti as usize))
            .and_then(|o| o.as_ref())
    };

    for function in &abi.functions {
        let Some((wasmparser::ExternalKind::Func, fidx)) = exports.get(&function.name).copied() else {
            bail!("invalid_abi: missing function export {}", function.name);
        };
        let fty = lookup_func_sig(fidx)
            .ok_or_else(|| anyhow!("invalid_abi: function export {} has invalid type", function.name))?;
        let expected_param = match function.parameter {
            Parameter::Callback(_) => wasmparser::ValType::EXTERNREF,
            Parameter::Value(v) => wasmparser_val(v),
        };
        let expected_result = wasmparser_val(function.result);
        ensure!(
            fty.params() == [expected_param] && fty.results() == [expected_result],
            "invalid_abi: function export {} signature mismatch",
            function.name
        );
    }
    for constant in &abi.constants {
        let Some((wasmparser::ExternalKind::Global, gidx)) = exports.get(&constant.name).copied() else {
            bail!("invalid_abi: missing constant export {}", constant.name);
        };
        let expected_ty = wasmparser_val(constant.value);
        ensure!(
            global_types.get(gidx as usize) == Some(&expected_ty),
            "invalid_abi: constant export {} has wrong global type",
            constant.name
        );
    }
    let has_blot_memory_export =
        exports.get("blot:memory").is_some_and(|&(k, idx)| k == wasmparser::ExternalKind::Memory && idx == 0);
    if abi.uses_arrays() {
        ensure!(has_blot_memory_export, "invalid_abi: missing array arena export blot:memory");
        for name in ["blot:allocate", "blot:reset"] {
            let Some((wasmparser::ExternalKind::Func, fidx)) = exports.get(name).copied() else {
                bail!("invalid_abi: missing array arena export {name}");
            };
            let ok = lookup_func_sig(fidx)
                .is_some_and(|fty| fty.params() == [wasmparser::ValType::I32] && fty.results() == [wasmparser::ValType::I32]);
            ensure!(ok, "invalid_abi: invalid {name} signature");
        }
    }
    if abi.ownership {
        let Some((wasmparser::ExternalKind::Func, enter_idx)) = exports.get(OWNERSHIP_ENTER).copied() else {
            bail!("invalid_abi: missing ownership export {OWNERSHIP_ENTER}");
        };
        ensure!(
            lookup_func_sig(enter_idx).is_some_and(|fty| fty.params().is_empty() && fty.results().is_empty()),
            "invalid_abi: invalid {OWNERSHIP_ENTER} signature"
        );
        let Some((wasmparser::ExternalKind::Func, leave_idx)) = exports.get(OWNERSHIP_LEAVE).copied() else {
            bail!("invalid_abi: missing ownership export {OWNERSHIP_LEAVE}");
        };
        ensure!(
            lookup_func_sig(leave_idx)
                .is_some_and(|fty| fty.params() == [wasmparser::ValType::I32] && fty.results().is_empty()),
            "invalid_abi: invalid {OWNERSHIP_LEAVE} signature"
        );
    }
    let invocation_finish = exports.get(INVOCATION_FINISH).copied();
    ensure!(
        invocation_finish.is_some() == abi.invocation_finish,
        "invalid_abi: {INVOCATION_FINISH} must match the invocation finish section"
    );
    if let Some((kind, fidx)) = invocation_finish {
        let valid_sig = kind == wasmparser::ExternalKind::Func
            && lookup_func_sig(fidx)
                .is_some_and(|fty| fty.params().is_empty() && fty.results() == [wasmparser::ValType::I32]);
        ensure!(valid_sig, "invalid_abi: invalid {INVOCATION_FINISH} signature");
    }
    Ok(ValidatedAotWasm {
        memory_limits,
        has_blot_memory_export,
        has_mut_extern_table,
        expected_imports: Some(expected_imports),
    })
}

#[cfg(not(feature = "compiler"))]
fn validate_aot_wasm(_wasm: &[u8], abi: &GuestAbi) -> Result<ValidatedAotWasm> {
    Ok(ValidatedAotWasm {
        memory_limits: None,
        has_blot_memory_export: abi.uses_arrays(),
        has_mut_extern_table: false,
        expected_imports: None,
    })
}

/// Decodes a static UTF-8 byte slice returned by an AOT library export.
///
/// # Safety
/// If `len > 0`, `ptr` must point to `len` readable immutable bytes that remain live for the
/// lifetime of `_lib` (the loaded `libloading::Library`).
unsafe fn read_aot_str(_lib: &Library, ptr: *const u8, len: usize) -> Result<&str> {
    if len == 0 {
        return Ok("");
    }
    ensure!(!ptr.is_null(), "invalid_abi: null string pointer with non-zero length {len}");
    // SAFETY: upheld by caller's contract tying `(ptr, len)` to `_lib`.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    std::str::from_utf8(bytes).context("invalid_abi: non-UTF-8 string returned by AOT library")
}

impl AotModule {
    pub fn compile(wasm: &[u8], abi: &GuestAbi, cache: Option<&Path>) -> Result<Self> {
        ensure!(abi.tasks.is_none(), "wtime-aot experimental backend does not support legacy tasks");
        abi.check_hostable()?;
        let validated = validate_aot_wasm(wasm, abi)?;
        let lib = load_verified_so(wasm, cache)?;
        let abi_version: FnAbiVersion = *unsafe { lib.get(b"blot_aot_abi_version\0")? };
        let version = unsafe { abi_version() };
        ensure!(version == 1, "unsupported wtime-aot blot ABI version {version}");

        let import_count_fn: FnImportCount = *unsafe { lib.get(b"blot_aot_import_count\0")? };
        let import_name_fn: FnImportName = *unsafe { lib.get(b"blot_aot_import_name\0")? };
        let callbacks = abi.callbacks();
        let n_imports = unsafe { import_count_fn() };
        if let Some(expected) = &validated.expected_imports {
            ensure!(
                n_imports as usize == expected.len(),
                "invalid_abi: loaded AOT library import count ({n_imports}) does not match validated Wasm ({})",
                expected.len()
            );
        }
        ensure!(n_imports <= 1024, "invalid_abi: excessive AOT import count {n_imports}");
        let mut imports = Vec::with_capacity(n_imports as usize);
        for idx in 0..n_imports {
            let (mut mp, mut ml, mut np, mut nl) = (std::ptr::null(), 0, std::ptr::null(), 0);
            ensure!(unsafe { import_name_fn(idx, &mut mp, &mut ml, &mut np, &mut nl) } == 0, "bad import index {idx}");
            let module = unsafe { read_aot_str(&lib, mp, ml) }?;
            let name = unsafe { read_aot_str(&lib, np, nl) }?;
            ensure!(module == HOST_MODULE, "invalid_abi: unexpected import {module}::{name}");
            let sig = callbacks
                .iter()
                .copied()
                .find(|cb| cb.import_name() == name)
                .ok_or_else(|| anyhow!("invalid_abi: unexpected import {module}::{name}"))?;
            imports.push(sig);
        }
        if let Some(expected) = &validated.expected_imports {
            ensure!(
                &imports == expected,
                "invalid_abi: loaded AOT library imports ({imports:?}) do not match validated Wasm imports ({expected:?})"
            );
        }

        let syms = unsafe {
            AotSymbols {
                trap_message: *lib.get(b"blot_aot_trap_message\0")?,
                new: *lib.get(b"blot_aot_new\0")?,
                drop: *lib.get(b"blot_aot_drop\0")?,
                call_export: *lib.get(b"blot_aot_call_export\0")?,
                call_allocate: *lib.get(b"blot_aot_call_allocate\0")?,
                call_reset: *lib.get(b"blot_aot_call_reset\0")?,
                call_finish_invocation: *lib.get(b"blot_aot_call_finish_invocation\0")?,
                mem_data: *lib.get(b"blot_aot_mem_data\0")?,
                mem_grow: *lib.get(b"blot_aot_mem_grow\0")?,
                get_global: *lib.get(b"blot_aot_get_global\0")?,
                set_global: *lib.get(b"blot_aot_set_global\0")?,
                active_mem_data: *lib.get(b"blot_aot_active_mem_data\0")?,
                active_allocate: *lib.get(b"blot_aot_active_allocate\0")?,
                active_reset: *lib.get(b"blot_aot_active_reset\0")?,
                active_get_global: *lib.get(b"blot_aot_active_get_global\0")?,
            }
        };
        let counter_names = counter_export_names(wasm, abi);
        Ok(Self {
            lib: Arc::new(AotLibrary {
                _lib: lib,
                syms,
                imports,
                counter_names,
                memory_limits: validated.memory_limits,
                has_blot_memory_export: validated.has_blot_memory_export,
                has_mut_extern_table: validated.has_mut_extern_table,
            }),
        })
    }
}

struct SendPtr(*mut c_void);
// SAFETY: `BlotHandle` is owned exclusively by one `AotGuest` and never aliased.
unsafe impl Send for SendPtr {}

struct AotHostState {
    lib: Arc<AotLibrary>,
    active: Option<Active>,
    memory: Option<Arc<AtomicUsize>>,
    probe: Option<Arc<Probe>>,
    spare: Vec<u32>,
    pending_error: Option<anyhow::Error>,
    pending_panic: Option<Box<dyn Any + Send>>,
    uses_arrays: bool,
}

/// Heap-allocated host context shared with the AOT guest via raw `user_ctx`.
///
/// Guarded by a short-locked `Mutex<AotHostState>` that is NEVER held across any guest FFI call,
/// host callback invocation, or reentrant `active_allocate` call — matching Wasmtime's exact
/// nested-callback semantics (a guest `blot:allocate` may invoke the restored active callback
/// during result encoding) with zero `UnsafeCell` aliasing UB.
struct AotHostBox {
    state: Mutex<AotHostState>,
}

struct SendHostPtr(*mut AotHostBox);
// SAFETY: `AotHostBox` is owned exclusively by one `AotGuest`, `AotHostState: Send`, and
// accesses are strictly serialized on the calling thread.
unsafe impl Send for SendHostPtr {}

unsafe extern "C" fn host_callback_trampoline(
    user_ctx: *mut c_void,
    active_call: *mut c_void,
    import_idx: u32,
    token: u64,
    arg_bits: u64,
    res_out: *mut u64,
) -> i32 {
    if user_ctx.is_null() {
        return 1;
    }
    // SAFETY: `user_ctx` points to a live `AotHostBox` owned by `AotGuest` for the handle's lifetime.
    let host_box = unsafe { &*(user_ctx as *const AotHostBox) };
    match catch_unwind(AssertUnwindSafe(|| invoke_aot_import(host_box, active_call, import_idx, token, arg_bits))) {
        Ok(Ok(bits)) => {
            if !res_out.is_null() {
                unsafe { *res_out = bits };
            }
            0
        }
        Ok(Err(err)) => {
            if let Ok(mut state) = host_box.state.lock()
                && state.pending_panic.is_none()
                && state.pending_error.is_none()
            {
                state.pending_error = Some(err);
            }
            1
        }
        Err(payload) => {
            // Never unwind across `extern "C"` / generated native frames; stash innermost payload to resume on host side.
            if let Ok(mut state) = host_box.state.lock()
                && state.pending_panic.is_none()
            {
                state.pending_panic = Some(payload);
            }
            1
        }
    }
}

fn aot_from_bits(
    mem_bytes: &[u8],
    uses_arrays: bool,
    spare: &mut Vec<u32>,
    kind: ValueType,
    bits: u64,
) -> Result<Value> {
    let non_canonical = || anyhow!("invalid_abi: non-canonical Wasm {kind} value");
    let low = (bits & 0xffff_ffff) as u32;
    match kind {
        ValueType::F32 => Ok(Value::F32(f32::from_bits(low))),
        ValueType::Unit => {
            if low == 0 {
                Ok(Value::Unit)
            } else {
                Err(non_canonical())
            }
        }
        ValueType::Bool => match low {
            0 => Ok(Value::Bool(false)),
            1 => Ok(Value::Bool(true)),
            _ => Err(non_canonical()),
        },
        ValueType::U32 => Ok(Value::U32(low)),
        ValueType::ArrayU32 | ValueType::ArrayF32 => {
            ensure!(uses_arrays, "invalid_abi: array value requires an arena");
            let mut words = std::mem::take(spare);
            words.clear();
            let address = array_range(mem_bytes.len(), low.cast_signed(), 0)?;
            let length = u32::from_le_bytes(mem_bytes[address..address + 4].try_into().unwrap()) as usize;
            array_range(mem_bytes.len(), low.cast_signed(), length)?;
            let body = &mem_bytes[address + 4..address + 4 + length * 4];
            match bytemuck::try_cast_slice::<u8, u32>(body) {
                Ok(aligned) if cfg!(target_endian = "little") => words.extend_from_slice(aligned),
                _ => words.extend(body.as_chunks::<4>().0.iter().map(|chunk| u32::from_le_bytes(*chunk))),
            }
            Ok(if kind == ValueType::ArrayU32 {
                Value::ArrayU32(words)
            } else {
                Value::ArrayF32(bytemuck::allocation::cast_vec(words))
            })
        }
    }
}

fn aot_to_bits(
    uses_arrays: bool,
    spare: &mut Vec<u32>,
    value: Value,
    allocate: impl FnOnce(i32) -> Result<i32>,
    with_mem_mut: impl FnOnce(&mut dyn FnMut(&mut [u8]) -> Result<()>) -> Result<()>,
) -> Result<u64> {
    let words: Vec<u32> = match value {
        Value::Unit => return Ok(0),
        Value::U32(v) => return Ok(u64::from(v)),
        Value::Bool(v) => return Ok(u64::from(u32::from(v))),
        Value::F32(v) => return Ok(u64::from(v.to_bits())),
        Value::ArrayU32(v) => v,
        Value::ArrayF32(v) => bytemuck::allocation::cast_vec(v),
    };
    ensure!(uses_arrays, "invalid_abi: array value requires an arena");
    ensure!(words.len() < 1_073_741_823, "array byte count exceeds the Wasm32 address range");
    let length = u32::try_from(words.len())?;
    let pointer = allocate(((length + 1) * 4).cast_signed())?;
    with_mem_mut(&mut |memory: &mut [u8]| {
        let address = array_range(memory.len(), pointer, words.len())?;
        memory[address..address + 4].copy_from_slice(&length.to_le_bytes());
        let body = &mut memory[address + 4..address + 4 + words.len() * 4];
        if cfg!(target_endian = "little") {
            body.copy_from_slice(bytemuck::cast_slice(&words));
        } else {
            for (chunk, word) in body.as_chunks_mut::<4>().0.iter_mut().zip(&words) {
                *chunk = word.to_le_bytes();
            }
        }
        Ok(())
    })?;
    *spare = words;
    Ok(u64::from(pointer.cast_unsigned()))
}

fn restore_spare(host_box: &AotHostBox, spare: Vec<u32>) {
    if let Ok(mut state) = host_box.state.lock()
        && spare.capacity() > state.spare.capacity()
    {
        state.spare = spare;
    }
}

fn invoke_aot_import(
    host_box: &AotHostBox,
    active_call: *mut c_void,
    import_idx: u32,
    token: u64,
    arg_bits: u64,
) -> Result<u64> {
    // Short lock #1: extract metadata and move `spare` out so NO lock/reference to `AotHostState`
    // is held across any FFI call (`active_mem_data`, `active_get_global`, `callback`, or `active_allocate`).
    let (signature, export, lib, uses_arrays, memory_gauge, probe, mut spare) = {
        let mut state = host_box
            .state
            .try_lock()
            .map_err(|_| anyhow!("reentrant_call: host state lock unexpectedly contended"))?;
        let signature = *state
            .lib
            .imports
            .get(import_idx as usize)
            .ok_or_else(|| anyhow!("unknown import index {import_idx}"))?;
        let (export, expected) = match &state.active {
            Some(active) if token != 0 && active.token == token => (active.export.clone(), active.signature),
            _ => bail!("stale_capability: host call has no active capability reference"),
        };
        ensure!(expected == signature, "capability_signature: host call signature does not match its capability");
        (
            signature,
            export,
            state.lib.clone(),
            state.uses_arrays,
            state.memory.clone(),
            state.probe.clone(),
            std::mem::take(&mut state.spare),
        )
    };

    let (mem_ptr, mem_len) = {
        let mut p: *mut u8 = std::ptr::null_mut();
        let mut n: usize = 0;
        unsafe { (lib.syms.active_mem_data)(active_call, &mut p, &mut n) };
        (p, n)
    };
    let mem_slice = if mem_len == 0 || mem_ptr.is_null() {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(mem_ptr, mem_len) }
    };
    let argument = match aot_from_bits(mem_slice, uses_arrays, &mut spare, signature.parameter, arg_bits) {
        Ok(arg) => arg,
        Err(err) => {
            restore_spare(host_box, spare);
            return Err(err);
        }
    };
    if uses_arrays
        && let Some(gauge) = &memory_gauge
    {
        gauge.store(mem_len, Ordering::Relaxed);
    }
    if let Some(probe) = probe {
        let mut vals = Vec::with_capacity(lib.counter_names.len());
        for name in &lib.counter_names {
            let (mut val, mut kind) = (0u64, 0u8);
            let rc = unsafe { (lib.syms.active_get_global)(active_call, name.as_ptr(), name.len(), &mut val, &mut kind) };
            let c = if rc == 0 && (kind == 0 || kind == 1) { (val & 0xffff_ffff) as u32 } else { 0 };
            vals.push(u64::from(c));
        }
        probe.called_host(if uses_arrays { mem_len } else { 0 }, vals.into_iter());
    }

    // Short lock #2: take callback out before invoking it.
    let mut callback = {
        let mut state = host_box
            .state
            .try_lock()
            .map_err(|_| anyhow!("reentrant_call: host state lock unexpectedly contended"))?;
        match state.active.as_mut().and_then(|active| active.callback.take()) {
            Some(cb) => cb,
            None => {
                if spare.capacity() > state.spare.capacity() {
                    state.spare = spare;
                }
                bail!("reentrant_call: cannot reenter an active guest instance");
            }
        }
    };
    let result = callback(argument);
    // Short lock #3: restore callback immediately after invocation, matching Wasmtime so guest
    // `blot:allocate` during result encoding may legally invoke the active callback.
    {
        let mut state = host_box
            .state
            .try_lock()
            .map_err(|_| anyhow!("reentrant_call: host state lock unexpectedly contended"))?;
        if let Some(active) = state.active.as_mut() {
            active.callback = Some(callback);
        }
    }
    let result = match result.with_context(|| format!("host_exception: host callback failed during {export}")) {
        Ok(val) => val,
        Err(err) => {
            restore_spare(host_box, spare);
            return Err(err);
        }
    };
    if result.kind() != signature.result {
        restore_spare(host_box, spare);
        bail!(
            "host_result: host callback during {export} returned {}, not {}",
            result.kind(),
            signature.result
        );
    }

    // Zero locks are held during `active_allocate` below.
    let encoded = aot_to_bits(
        uses_arrays,
        &mut spare,
        result,
        |bytes| {
            let mut ptr_out = 0i32;
            let mut trap_code = 0u8;
            let rc = unsafe { (lib.syms.active_allocate)(active_call, bytes, &mut ptr_out, &mut trap_code) };
            match rc {
                0 => Ok(ptr_out),
                1 => Err(lib.trap_error(trap_code)),
                2 => Err(anyhow!("host error during blot:allocate")),
                _ => bail!("invalid_abi: missing blot:allocate"),
            }
        },
        |write| {
            let mut p: *mut u8 = std::ptr::null_mut();
            let mut n: usize = 0;
            unsafe { (lib.syms.active_mem_data)(active_call, &mut p, &mut n) };
            let slice = if n == 0 || p.is_null() {
                &mut [][..]
            } else {
                // SAFETY: `active_call` uniquely borrows guest linear memory for this scoped write.
                unsafe { std::slice::from_raw_parts_mut(p, n) }
            };
            write(slice)
        },
    );
    // Short lock #4: restore `spare` buffer.
    restore_spare(host_box, spare);
    encoded
}

pub struct AotGuest {
    lib: Arc<AotLibrary>,
    handle: SendPtr,
    host: SendHostPtr,
    abi: Arc<GuestAbi>,
    last_reclaimed_owners: Option<u32>,
    imaged: Arc<Result<Vec<String>, String>>,
    entered: bool,
    disposed: bool,
}

impl Drop for AotGuest {
    fn drop(&mut self) {
        if !self.handle.0.is_null() {
            unsafe { (self.lib.syms.drop)(self.handle.0) };
            self.handle.0 = std::ptr::null_mut();
        }
        if !self.host.0.is_null() {
            // SAFETY: handle has been dropped, so no guest callback can ever dereference `self.host.0`.
            unsafe { drop(Box::from_raw(self.host.0)) };
            self.host.0 = std::ptr::null_mut();
        }
    }
}

impl AotGuest {
    fn with_host_mut<R>(&mut self, f: impl FnOnce(&mut AotHostState) -> R) -> R {
        // SAFETY: `self` exclusively owns `self.host.0` and no guest FFI call is active while this
        // short lock is held.
        let host_box = unsafe { &*self.host.0 };
        let mut state = host_box.state.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut state)
    }

    /// Instantiates an AOT guest handle.
    ///
    /// Signal-stack contract: `wtime-aot` allocates a dedicated 8 MiB guest stack with a 64 KiB
    /// `PROT_NONE` guard and a 128 KiB signal-safe floor reserve (budgeting up to 2 nested
    /// `<= 32 KiB` async signal handlers at `AT_MINSIGSTKSZ <= 16 KiB`). Pen installs no custom
    /// async signal handlers on `pen run` guest threads; embedders installing deeper/heavier
    /// async handlers or running on CPUs with `AT_MINSIGSTKSZ > 16 KiB` MUST configure
    /// `SA_ONSTACK` + `sigaltstack` on calling threads.
    pub fn start(
        module: &AotModule,
        abi: Arc<GuestAbi>,
        imaged: Arc<Result<Vec<String>, String>>,
        limits: Option<Limits>,
    ) -> Result<Self> {
        let limit_pages = match limits {
            Some(lim) => {
                let max_allowed_pages = u64::try_from(lim.memory / PAGE).unwrap_or(u64::MAX);
                if let Some((min_p, max_p)) = module.lib.memory_limits {
                    let initial_bytes = min_p.saturating_mul(PAGE as u64);
                    if initial_bytes > lim.memory as u64 {
                        bail!("memory_limit: guest initial memory ({initial_bytes} B) exceeds limit ({} B)", lim.memory);
                    }
                    if max_allowed_pages == 0 {
                        // Blot ABI v1 reserves `mem_limit_pages == 0` as unlimited; reject growable memories
                        // BEFORE `blot_aot_new` (so `start` never runs and `memory.grow` cannot bypass a <64 KiB cap).
                        if max_p != Some(0) {
                            bail!(
                                "memory_limit: wtime-aot ABI v1 cannot enforce sub-page limit ({} B) on growable memory",
                                lim.memory
                            );
                        }
                        1
                    } else {
                        max_allowed_pages
                    }
                } else {
                    // Module declares no memory; pass a non-zero page cap safely.
                    max_allowed_pages.max(1)
                }
            }
            None => 0,
        };
        let host_ptr = Box::into_raw(Box::new(AotHostBox {
            state: Mutex::new(AotHostState {
                lib: module.lib.clone(),
                active: None,
                memory: None,
                probe: None,
                spare: Vec::new(),
                pending_error: None,
                pending_panic: None,
                uses_arrays: abi.uses_arrays(),
            }),
        }));
        let mut trap_code = 0u8;
        let handle = unsafe {
            (module.lib.syms.new)(host_ptr.cast::<c_void>(), Some(host_callback_trampoline), limit_pages, &mut trap_code)
        };
        if handle.is_null() {
            // SAFETY: `new` returned null, so no handle retains `host_ptr`.
            let host = unsafe { Box::from_raw(host_ptr) }.state.into_inner().unwrap_or_else(|e| e.into_inner());
            if let Some(payload) = host.pending_panic {
                resume_unwind(payload);
            }
            if let Some(err) = host.pending_error {
                return Err(err);
            }
            return Err(module.lib.trap_error(trap_code));
        }
        if let Some(lim) = limits {
            let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
            unsafe { (module.lib.syms.mem_data)(handle, &mut p, &mut n) };
            if n > lim.memory {
                unsafe { (module.lib.syms.drop)(handle) };
                unsafe { drop(Box::from_raw(host_ptr)) };
                bail!("memory_limit: guest initial memory ({n} B) exceeds limit ({} B)", lim.memory);
            }
        }
        Ok(Self {
            lib: module.lib.clone(),
            handle: SendPtr(handle),
            host: SendHostPtr(host_ptr),
            abi,
            last_reclaimed_owners: None,
            imaged,
            entered: false,
            disposed: false,
        })
    }

    pub fn probe(&mut self, probe: Arc<Probe>) {
        probe.counters_named(
            self.lib
                .counter_names
                .iter()
                .map(|name| name.trim_start_matches("blot:").to_owned())
                .collect(),
        );
        self.with_host_mut(|host| host.probe = Some(probe));
    }

    pub fn memory_bytes(&self) -> usize {
        if !self.abi.uses_arrays() {
            return 0;
        }
        let mut p: *mut u8 = std::ptr::null_mut();
        let mut n: usize = 0;
        unsafe { (self.lib.syms.mem_data)(self.handle.0, &mut p, &mut n) };
        n
    }

    pub fn last_reclaimed_owners(&self) -> Option<u32> {
        self.last_reclaimed_owners
    }

    pub fn track_memory(&mut self, gauge: Arc<AtomicUsize>) {
        gauge.store(self.memory_bytes(), Ordering::Relaxed);
        self.with_host_mut(|host| host.memory = Some(gauge));
    }

    fn maybe_resume_panic(&mut self) {
        if let Some(payload) = self.with_host_mut(|host| host.pending_panic.take()) {
            // Poison the guest handle before resuming unwind so partially unwound state cannot be reused.
            self.disposed = true;
            resume_unwind(payload);
        }
    }

    fn call_export_raw(&mut self, name: &str, arg_bits: u64) -> Result<u64> {
        let mut res_out = 0u64;
        let mut trap_code = 0u8;
        let rc = unsafe {
            (self.lib.syms.call_export)(self.handle.0, name.as_ptr(), name.len(), arg_bits, &mut res_out, &mut trap_code)
        };
        self.maybe_resume_panic();
        match rc {
            0 => Ok(res_out),
            2 => Err(self
                .with_host_mut(|host| host.pending_error.take())
                .unwrap_or_else(|| self.lib.trap_error(trap_code))),
            1 => Err(self.lib.trap_error(trap_code)),
            _ => bail!("unknown_export: unknown function export {name}"),
        }
    }

    fn call_reset(&mut self) -> Result<()> {
        let mut out = 0i32;
        let mut trap_code = 0u8;
        let rc = unsafe { (self.lib.syms.call_reset)(self.handle.0, 0, &mut out, &mut trap_code) };
        self.maybe_resume_panic();
        match rc {
            0 => Ok(()),
            1 => Err(self.lib.trap_error(trap_code)),
            2 => Err(self
                .with_host_mut(|host| host.pending_error.take())
                .unwrap_or_else(|| self.lib.trap_error(trap_code))),
            _ => bail!("invalid_abi: missing blot:reset"),
        }
    }

    pub fn call(&mut self, name: &str, argument: Argument) -> Result<Value> {
        ensure!(!self.disposed, "disposed_guest: the guest instance is disposed");
        let function =
            self.abi.function(name).ok_or_else(|| anyhow!("unknown_export: unknown function export {name}"))?.clone();
        let result = self.begin_and_run(&function, argument);
        if let Some(probe) = self.with_host_mut(|host| host.probe.clone()) {
            probe.called(result.is_err());
        }
        let finish = self.finish(if result.is_ok() { LEFT_NORMALLY } else { LEFT_FAILING });
        let result = result?;
        finish?;
        Ok(result)
    }

    fn begin_and_run(&mut self, function: &GuestFunction, argument: Argument) -> Result<Value> {
        let name = &function.name;
        self.with_host_mut(|host| {
            host.pending_error = None;
            host.pending_panic = None;
        });
        if self.abi.ownership {
            self.call_export_raw(OWNERSHIP_ENTER, 0)?;
            self.entered = true;
        }
        if self.abi.uses_arrays() {
            self.call_reset()?;
        }
        let arg_bits = match (function.parameter, argument) {
            (Parameter::Value(expected), Argument::Value(value)) => {
                ensure!(value.kind() == expected, "invalid_argument: {name} argument must be {expected}, not {}", value.kind());
                let lib = self.lib.clone();
                let handle = self.handle.0;
                let (uses_arrays, mut spare) = self.with_host_mut(|host| (host.uses_arrays, std::mem::take(&mut host.spare)));
                let mut alloc_host_err = false;
                let encoded = aot_to_bits(
                    uses_arrays,
                    &mut spare,
                    value,
                    |bytes| {
                        let mut ptr_out = 0i32;
                        let mut trap_code = 0u8;
                        let rc = unsafe { (lib.syms.call_allocate)(handle, bytes, &mut ptr_out, &mut trap_code) };
                        match rc {
                            0 => Ok(ptr_out),
                            1 => Err(lib.trap_error(trap_code)),
                            2 => {
                                alloc_host_err = true;
                                Err(lib.trap_error(trap_code))
                            }
                            _ => bail!("invalid_abi: missing blot:allocate"),
                        }
                    },
                    |write| {
                        let mut p: *mut u8 = std::ptr::null_mut();
                        let mut n: usize = 0;
                        unsafe { (lib.syms.mem_data)(handle, &mut p, &mut n) };
                        let slice = if n == 0 || p.is_null() {
                            &mut [][..]
                        } else {
                            // SAFETY: `handle` uniquely owns guest linear memory for this scoped write.
                            unsafe { std::slice::from_raw_parts_mut(p, n) }
                        };
                        write(slice)
                    },
                );
                let pending = self.with_host_mut(|host| {
                    host.spare = spare;
                    if alloc_host_err { host.pending_error.take() } else { None }
                });
                self.maybe_resume_panic();
                if let Some(err) = pending {
                    return Err(err);
                }
                encoded?
            }
            (Parameter::Callback(signature), Argument::Callback(callback)) => {
                let token = TOKENS.fetch_add(1, Ordering::Relaxed);
                self.with_host_mut(|host| {
                    host.active = Some(Active { token, signature, callback: Some(callback), export: name.clone() });
                });
                token
            }
            (Parameter::Callback(_), Argument::Value(_)) => {
                bail!("invalid_capability: {name} requires a host callback")
            }
            (Parameter::Value(_), Argument::Callback(_)) => {
                bail!("invalid_argument: {name} does not take a host callback")
            }
        };
        let res_bits = self.call_export_raw(name, arg_bits)?;
        let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
        unsafe { (self.lib.syms.mem_data)(self.handle.0, &mut p, &mut n) };
        let mem = if n == 0 || p.is_null() { &[][..] } else { unsafe { std::slice::from_raw_parts(p, n) } };
        self.with_host_mut(|host| aot_from_bits(mem, host.uses_arrays, &mut host.spare, function.result, res_bits))
    }

    fn finish(&mut self, outcome: i32) -> Result<()> {
        self.with_host_mut(|host| host.active = None);
        let mut left = Ok(());
        if std::mem::take(&mut self.entered) && self.abi.ownership {
            left = self.call_export_raw(OWNERSHIP_LEAVE, u64::from(outcome.cast_unsigned())).map(|_| ());
        }
        let invocation = if self.abi.invocation_finish {
            self.last_reclaimed_owners = None;
            let mut released = 0i32;
            let mut trap_code = 0u8;
            let rc = unsafe { (self.lib.syms.call_finish_invocation)(self.handle.0, &mut released, &mut trap_code) };
            self.maybe_resume_panic();
            match rc {
                0 => {
                    self.last_reclaimed_owners = Some(released.cast_unsigned());
                    Ok(())
                }
                1 => Err(self.lib.trap_error(trap_code)),
                2 => Err(self
                    .with_host_mut(|host| host.pending_error.take())
                    .unwrap_or_else(|| self.lib.trap_error(trap_code))),
                _ => bail!("invalid_abi: missing blot:finish_invocation"),
            }
        } else {
            Ok(())
        };
        if self.abi.uses_arrays() {
            self.call_reset()?;
        }
        left?;
        invocation?;
        Ok(())
    }

    pub fn dispose(&mut self) -> Result<()> {
        if std::mem::replace(&mut self.disposed, true) {
            return Ok(());
        }
        self.with_host_mut(|host| {
            host.pending_error = None;
            host.pending_panic = None;
        });
        let mut left = Ok(());
        if self.abi.ownership {
            left = self.call_export_raw(OWNERSHIP_LEAVE, u64::from(TEARDOWN.cast_unsigned())).map(|_| ());
            if self.abi.uses_arrays() {
                let reset = self.call_reset();
                left?;
                reset?;
                return Ok(());
            }
        }
        left
    }

    pub fn capture(&mut self, previous: Option<&GuestImage>) -> Result<(GuestImage, Captured)> {
        ensure!(!self.disposed, "disposed_guest: the guest instance is disposed");
        ensure!(
            !self.lib.has_mut_extern_table,
            "this guest cannot be imaged: mutable externref tables are not captured by snapshots"
        );
        let names = self.imaged.as_ref().as_ref().map_err(|why| anyhow!("this guest cannot be imaged: {why}"))?;
        let bytes = if self.lib.has_blot_memory_export {
            let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
            unsafe { (self.lib.syms.mem_data)(self.handle.0, &mut p, &mut n) };
            if n == 0 || p.is_null() { &[][..] } else { unsafe { std::slice::from_raw_parts(p, n) } }
        } else {
            &[][..]
        };
        let (memory, captured) = MemoryImage::capture(bytes, previous.map(|prev| &prev.memory));
        let mut globals = Vec::with_capacity(names.len());
        for name in names {
            let (mut val, mut kind) = (0u64, 0u8);
            ensure!(
                unsafe { (self.lib.syms.get_global)(self.handle.0, name.as_ptr(), name.len(), &mut val, &mut kind) } == 0,
                "an imaged global disappeared: {name}"
            );
            let wval = match kind {
                0 => Val::I32(((val & 0xffff_ffff) as u32).cast_signed()),
                1 => Val::I64(val.cast_signed()),
                2 => Val::F32((val & 0xffff_ffff) as u32),
                3 => Val::F64(val),
                _ => bail!("unsupported imaged global kind {kind}"),
            };
            globals.push((name.clone(), wval));
        }
        Ok((GuestImage { memory, globals }, captured))
    }

    pub fn restore(&mut self, image: &GuestImage) -> Result<usize> {
        ensure!(!self.disposed, "disposed_guest: the guest instance is disposed");
        ensure!(
            !self.lib.has_mut_extern_table,
            "this guest cannot be restored: mutable externref tables are not captured by snapshots"
        );
        // Preflight complete imaged-global schema (exact mutable export set, no duplicates/extras)
        // AND Wasm value kinds BEFORE growing or mutating linear memory or any global.
        let allowed = self.imaged.as_ref().as_ref().map_err(|why| anyhow!("this guest cannot be restored: {why}"))?;
        ensure!(
            image.globals.len() == allowed.len(),
            "imaged global schema mismatch: guest expects {} mutable globals, snapshot has {}",
            allowed.len(),
            image.globals.len()
        );
        let mut seen = std::collections::HashSet::with_capacity(allowed.len());
        let mut validated_globals: Vec<(&str, u64)> = Vec::with_capacity(image.globals.len());
        for (name, value) in &image.globals {
            ensure!(
                allowed.iter().any(|a| a == name),
                "imaged global {name} is not a mutable imaged global of this guest"
            );
            ensure!(seen.insert(name.as_str()), "duplicate imaged global in snapshot: {name}");
            let (expected_kind, raw) = match *value {
                Val::I32(v) => (0u8, u64::from(v.cast_unsigned())),
                Val::I64(v) => (1u8, v.cast_unsigned()),
                Val::F32(bits) => (2u8, u64::from(bits)),
                Val::F64(bits) => (3u8, bits),
                _ => bail!("unsupported imaged global value"),
            };
            let (mut cur_val, mut cur_kind) = (0u64, u8::MAX);
            ensure!(
                unsafe { (self.lib.syms.get_global)(self.handle.0, name.as_ptr(), name.len(), &mut cur_val, &mut cur_kind) }
                    == 0,
                "an imaged global disappeared: {name}"
            );
            ensure!(
                cur_kind == expected_kind,
                "imaged global type mismatch for {name}: guest kind {cur_kind}, image kind {expected_kind}"
            );
            validated_globals.push((name.as_str(), raw));
        }
        let mut written = 0;
        if self.lib.has_blot_memory_export {
            let (mut p, mut current) = (std::ptr::null_mut(), 0usize);
            unsafe { (self.lib.syms.mem_data)(self.handle.0, &mut p, &mut current) };
            let missing = image.memory.len().saturating_sub(current);
            if missing > 0 {
                let delta = u64::try_from(missing.div_ceil(PAGE))?;
                let old = unsafe { (self.lib.syms.mem_grow)(self.handle.0, delta) };
                ensure!(old >= 0, "failed to grow guest memory during restore");
                unsafe { (self.lib.syms.mem_data)(self.handle.0, &mut p, &mut current) };
            }
            if current > 0 && !p.is_null() {
                // SAFETY: `handle` uniquely owns guest linear memory; even when `image.memory` is empty
                // (captured at 0 pages before a later grow), `restore` zeros committed pages beyond it
                // while preserving current page count (`memory.size`), matching `WasmtimeGuest::restore`.
                let bytes = unsafe { std::slice::from_raw_parts_mut(p, current) };
                written = image.memory.restore(bytes);
            }
        }
        for (name, raw) in validated_globals {
            ensure!(
                unsafe { (self.lib.syms.set_global)(self.handle.0, name.as_ptr(), name.len(), raw) } == 0,
                "imaged global {name} is not writable"
            );
        }
        Ok(written)
    }

    pub fn read(&mut self, name: &str) -> Result<Value> {
        ensure!(!self.disposed, "disposed_guest: the guest instance is disposed");
        let constant = self
            .abi
            .constants
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| anyhow!("unknown_export: unknown constant export {name}"))?
            .clone();
        let (mut val, mut kind) = (0u64, 0u8);
        ensure!(
            unsafe { (self.lib.syms.get_global)(self.handle.0, name.as_ptr(), name.len(), &mut val, &mut kind) } == 0,
            "validated constant export {name} disappeared"
        );
        self.with_host_mut(|host| aot_from_bits(&[], false, &mut host.spare, constant.value, val))
    }

    pub fn runtime_counters(&mut self) -> Vec<(String, u32)> {
        let mut out = Vec::with_capacity(self.lib.counter_names.len());
        for name in &self.lib.counter_names {
            let (mut val, mut kind) = (0u64, 0u8);
            let rc = unsafe { (self.lib.syms.get_global)(self.handle.0, name.as_ptr(), name.len(), &mut val, &mut kind) };
            if rc == 0 && (kind == 0 || kind == 1) {
                out.push((name.clone(), (val & 0xffff_ffff) as u32));
            }
        }
        out
    }
}
