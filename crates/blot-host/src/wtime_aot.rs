//! `wtime-aot` backend for Blot ABI 8 guest modules.
//!
//! Used by default for `pen run` builds when `wtime-aot` is available (disable
//! with `PEN_WTIME_AOT=0`), or for non-`dev_engine` engines when `PEN_WTIME_AOT=1`.
//! Compiles the guest Wasm module once via `wtime-aot build --backend x64 --blot-cdylib`
//! on Linux x86-64 (automatically falling back to `--backend c --c-opt auto --target-cpu native`
//! on unsupported platforms, ISAs/SHSTK, or admission rejections; override backend with
//! `PEN_WTIME_AOT_BACKEND=x64|c|rust`, C optimization mode with
//! `PEN_WTIME_AOT_C_OPT=auto|O2|O1|O0`, and x64 target CPU with
//! `PEN_WTIME_AOT_TARGET_CPU=native|x86-64-v2|x86-64-v3`) into a cached shared library,
//! then loads it with `libloading` and drives the exact same Blot ABI 8 lifecycle,
//! arena array copies, host callbacks, and snapshots as Wasmtime.

use std::any::Any;
use std::ffi::c_void;
#[cfg(feature = "compiler")]
use std::io::Write;
#[cfg(feature = "compiler")]
use std::io::{Read as _, Seek as _, SeekFrom};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
#[cfg(feature = "compiler")]
use std::path::{Path, PathBuf};
#[cfg(feature = "compiler")]
use std::process::Command;
#[cfg(feature = "compiler")]
use std::sync::OnceLock;
#[cfg(feature = "compiler")]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(feature = "compiler")]
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
#[cfg(feature = "compiler")]
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use libloading::Library;
#[cfg(feature = "compiler")]
use pen_runtime::abi::INVOCATION_FINISH;
use pen_runtime::abi::{CallbackType, GuestAbi, GuestFunction, Parameter, ValueType};
use pen_runtime::memory_image::{Captured, MemoryImage, PAGE};
use wasmtime::Val;

#[cfg(feature = "compiler")]
use crate::HOST_MODULE;
use crate::{
    Active, Argument, GuestImage, LEFT_FAILING, LEFT_NORMALLY, Limits, OWNERSHIP_ENTER, OWNERSHIP_LEAVE, Probe,
    TEARDOWN, TOKENS, Value, array_range,
};

#[cfg(feature = "compiler")]
fn parse_env_bool(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
}

#[cfg(feature = "compiler")]
fn is_executable_file(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        (meta.permissions().mode() & 0o111) != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

#[cfg(feature = "compiler")]
fn is_bare_prog_name(s: &str) -> bool {
    !s.is_empty() && !s.contains(std::path::MAIN_SEPARATOR) && !s.contains('/') && s != "." && s != ".."
}

/// Returns an absolute-only, empty-entry-free `PATH` so subprocess execution resolves bare names
/// identically to `find_in_path` with zero CWD lookup.
#[cfg(feature = "compiler")]
fn sanitized_path_env() -> Option<std::ffi::OsString> {
    let raw = std::env::var_os("PATH")?;
    let dirs: Vec<PathBuf> =
        std::env::split_paths(&raw).filter(|d| !d.as_os_str().is_empty() && d.is_absolute()).collect();
    if dirs.is_empty() {
        return None;
    }
    std::env::join_paths(dirs).ok()
}

#[cfg(feature = "compiler")]
fn find_in_path(name: &str) -> Option<PathBuf> {
    if !is_bare_prog_name(name) {
        return None;
    }
    let path_var = sanitized_path_env()?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if is_executable_file(&candidate) {
            return Some(candidate);
        }
    }
    None
}

#[cfg(feature = "compiler")]
fn resolve_wtime_aot_bin() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("PEN_WTIME_AOT_BIN")
        && !explicit.is_empty()
    {
        if is_bare_prog_name(&explicit) {
            return find_in_path(&explicit);
        }
        let p = PathBuf::from(&explicit);
        return is_executable_file(&p).then_some(p);
    }
    if let Some(in_path) = find_in_path("wtime-aot") {
        return Some(in_path);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let sibling = dir.join("wtime-aot");
        if is_executable_file(&sibling) {
            return Some(sibling);
        }
    }
    if let Some(home) = std::env::var_os("HOME").filter(|h| !h.is_empty()) {
        let home = PathBuf::from(home);
        for candidate in [home.join(".local/lib/pen-runtime/wtime-aot"), home.join(".local/bin/wtime-aot")] {
            if is_executable_file(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

/// Returns whether the `wtime-aot` backend is explicitly forced on everywhere.
#[cfg(feature = "compiler")]
pub fn enabled() -> bool {
    parse_env_bool("PEN_WTIME_AOT").unwrap_or(false)
}

/// Returns whether `wtime-aot` should be used for `pen run` builds.
///
/// Defaults to `true` when `wtime-aot` (or `PEN_WTIME_AOT_SO`) is available,
/// unless opted out via `PEN_WTIME_AOT=0` / `false`.
#[cfg(feature = "compiler")]
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

#[cfg(feature = "compiler")]
type FnAbiVersion = unsafe extern "C" fn() -> u32;
type FnTrapMessage = unsafe extern "C" fn(u8, *mut *const u8, *mut usize) -> i32;
#[cfg(feature = "compiler")]
type FnImportCount = unsafe extern "C" fn() -> u32;
#[cfg(feature = "compiler")]
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

/// Typed runtime trap returned by a `wtime-aot` guest instance, preserving the Blot ABI 8 trap code.
#[derive(Debug)]
pub(crate) struct AotTrap {
    pub(crate) code: u8,
    message: String,
}

impl std::fmt::Display for AotTrap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "wtime-aot trap: {}", self.message)
    }
}

impl std::error::Error for AotTrap {}

const AOT_TRAP_CALL_STACK: u8 = 8;

fn is_oom_trap_code(code: u8) -> bool {
    matches!(code, 9..=12)
}

pub(crate) fn is_unrecoverable_aot_trap(error: &anyhow::Error) -> bool {
    error.downcast_ref::<AotTrap>().is_some_and(|trap| trap.code == AOT_TRAP_CALL_STACK)
}

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
            anyhow::Error::new(AotTrap { code, message: msg })
        }
    }
}

#[derive(Clone)]
pub struct AotModule {
    lib: Arc<AotLibrary>,
}

#[cfg(feature = "compiler")]
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
    bail!(
        "no owner-private wtime-aot cache directory available (set PEN_WTIME_AOT_CACHE, XDG_CACHE_HOME, HOME, or XDG_RUNTIME_DIR)"
    )
}

#[cfg(feature = "compiler")]
fn wasm_sha256_hex(wasm: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(wasm))
}

#[cfg(feature = "compiler")]
const WTIME_AOT_MERKLE256K_CHUNK_BYTES: usize = 256 << 10;
#[cfg(feature = "compiler")]
const WTIME_AOT_MERKLE256K_PREFIX: &str = "merkle256k:";
#[cfg(feature = "compiler")]
const WTIME_AOT_MERKLE256K_DOMAIN: &[u8] = b"WTIME-MERKLE256K-1";

/// Computes `wtime-aot`'s deterministic 256 KiB chunk SHA-256 Merkle root (`WTIME-MERKLE256K-1`),
/// matching `tools/wtime-aot/src/sha256.rs::parallel_merkle_digest`, hashing chunks in parallel
/// when `bytes.len() > 256 KiB`.
#[cfg(feature = "compiler")]
fn merkle256k_sha256_hex(bytes: &[u8]) -> String {
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
#[cfg(feature = "compiler")]
#[derive(Clone, Debug, PartialEq, Eq)]
enum VerifiedSoDigest {
    Sha256(String),
    Merkle256k(String),
}

#[cfg(feature = "compiler")]
impl VerifiedSoDigest {
    fn parse(raw: &str) -> Option<Self> {
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

    fn filename(&self) -> String {
        match self {
            Self::Sha256(hex) => format!("{hex}.so"),
            Self::Merkle256k(hex) => format!("m-{hex}.so"),
        }
    }

    fn tmp_stem(&self) -> String {
        match self {
            Self::Sha256(hex) => hex.clone(),
            Self::Merkle256k(hex) => format!("m-{hex}"),
        }
    }

    fn line_text(&self) -> String {
        match self {
            Self::Sha256(hex) => hex.clone(),
            Self::Merkle256k(hex) => format!("{WTIME_AOT_MERKLE256K_PREFIX}{hex}"),
        }
    }

    fn matches_bytes(&self, bytes: &[u8]) -> bool {
        match self {
            Self::Sha256(expected) => wasm_sha256_hex(bytes) == *expected,
            Self::Merkle256k(expected) => merkle256k_sha256_hex(bytes) == *expected,
        }
    }

    fn matches_file(&self, path: &Path) -> bool {
        match self {
            Self::Sha256(expected) => file_sha256_hex(path).is_ok_and(|actual| actual == *expected),
            Self::Merkle256k(expected) => std::fs::read(path).is_ok_and(|b| merkle256k_sha256_hex(&b) == *expected),
        }
    }
}

/// Fast collision-resistant cache-key digest for large `.wasm` inputs (parallel `merkle256k`
/// above 256 KiB, plain SHA-256 at or below 256 KiB).
#[cfg(feature = "compiler")]
fn fast_wasm_cache_digest(wasm: &[u8]) -> String {
    if wasm.len() > WTIME_AOT_MERKLE256K_CHUNK_BYTES {
        format!("m256k-{}", merkle256k_sha256_hex(wasm))
    } else {
        wasm_sha256_hex(wasm)
    }
}

/// Verifies a `wtime-aot` `.so.sha256` integrity sidecar line: either plain 64-hex SHA-256
/// (artifacts `<= 256 KiB`) or `merkle256k:<64-hex>` (artifacts `> 256 KiB`).
#[cfg(feature = "compiler")]
#[cfg_attr(not(test), allow(dead_code))]
fn verify_artifact_sidecar_digest(bytes: &[u8], linear_sha256: &str, expected_trimmed: &str) -> bool {
    if let Some(expected_hex) = expected_trimmed.strip_prefix(WTIME_AOT_MERKLE256K_PREFIX) {
        expected_hex.len() == 64 && merkle256k_sha256_hex(bytes) == expected_hex
    } else {
        expected_trimmed.len() == 64 && linear_sha256 == expected_trimmed
    }
}

#[cfg(feature = "compiler")]
fn reader_sha256_hex(f: &mut std::fs::File, path: &Path) -> Result<String> {
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

#[cfg(feature = "compiler")]
fn file_sha256_hex(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    reader_sha256_hex(&mut f, path)
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

#[cfg(all(feature = "compiler", unix))]
fn validate_trusted_ancestor(anc: &Path, target: &Path, euid: u32) -> Result<()> {
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

/// Atomically publishes `bytes` into `verified_dir/<digest>.so` with `0o555` permissions via
/// `rename(&tmp, &dest)` (never `remove_file(&dest)` so there is no missing-file window).
#[cfg(feature = "compiler")]
fn publish_immutable_verified_so(
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

#[cfg(all(feature = "compiler", unix))]
type BinStamp = (u64, u64, u64, i64, i64, i64, i64);
#[cfg(all(feature = "compiler", not(unix)))]
type BinStamp = (u64, u64, u32);

#[cfg(feature = "compiler")]
fn bin_stamp_from_meta(meta: &std::fs::Metadata) -> BinStamp {
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

#[cfg(feature = "compiler")]
fn bin_stamp(canon: &Path) -> Result<BinStamp> {
    let meta = std::fs::metadata(canon).with_context(|| format!("stat wtime-aot binary {}", canon.display()))?;
    Ok(bin_stamp_from_meta(&meta))
}

#[cfg(feature = "compiler")]
fn read_private_digest_cache_entry(entry_path: &Path) -> Option<String> {
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

#[cfg(feature = "compiler")]
fn bin_digest(bin: &Path, canon_cache: &Path) -> Result<(PathBuf, String)> {
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
#[cfg(feature = "compiler")]
fn rt_sources_fingerprint(canon_bin: &Path) -> Result<String> {
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

#[cfg(feature = "compiler")]
fn host_cpu_fingerprint() -> &'static str {
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
    ensure!(Instant::now() < deadline, "wtime-aot {phase} exceeded host build timeout");
    Ok(())
}

#[cfg(feature = "compiler")]
fn lock_file_with_deadline(file: &std::fs::File, path: &Path, exclusive: bool, deadline: Instant) -> Result<()> {
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

#[cfg(all(feature = "compiler", unix))]
fn kill_process_group(pgid: u32) {
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

#[cfg(feature = "compiler")]
fn create_private_temp_file(dir: &Path, prefix: &str, suffix: &str) -> Result<(std::fs::File, PathBuf)> {
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

#[cfg(feature = "compiler")]
const MAX_CAPTURE_BYTES: usize = 16 << 20;
#[cfg(feature = "compiler")]
const POST_EXIT_DRAIN_GRACE: Duration = Duration::from_secs(2);

#[cfg(feature = "compiler")]
enum PipeCaptureErr {
    Overflow,
    Io(String),
}

#[cfg(feature = "compiler")]
fn spawn_bounded_pipe_reader<R: std::io::Read + Send + 'static>(
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

#[cfg(feature = "compiler")]
fn recv_bounded_pipe(
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
#[cfg(feature = "compiler")]
fn run_cmd_output_with_deadline(
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

/// Inspects an ELF64 Linux x86-64 shared library for `NT_GNU_PROPERTY_TYPE_0` carrying
/// `GNU_PROPERTY_X86_FEATURE_1_SHSTK`.
#[cfg(all(feature = "compiler", target_os = "linux", target_arch = "x86_64"))]
fn elf64_has_x86_shstk_property(elf: &[u8]) -> bool {
    if elf.len() < 64 || &elf[0..4] != b"\x7fELF" || elf[4] != 2 || elf[5] != 1 {
        return false;
    }
    let e_machine = u16::from_le_bytes(elf[18..20].try_into().unwrap());
    if e_machine != 62 {
        return false;
    }
    let Ok(e_phoff) = usize::try_from(u64::from_le_bytes(elf[32..40].try_into().unwrap())) else {
        return false;
    };
    let e_phentsize = usize::from(u16::from_le_bytes(elf[54..56].try_into().unwrap()));
    if e_phentsize < 56 {
        return false;
    }
    let e_phnum = usize::from(u16::from_le_bytes(elf[56..58].try_into().unwrap()));
    const PT_NOTE: u32 = 4;
    const PT_GNU_PROPERTY: u32 = 0x6474_e553;
    const NT_GNU_PROPERTY_TYPE_0: u32 = 5;
    const GNU_PROPERTY_X86_FEATURE_1_AND: u32 = 0xc000_0002;
    const GNU_PROPERTY_X86_FEATURE_1_SHSTK: u32 = 1 << 1;
    let align_up = |v: usize, a: usize| v.checked_add(a - 1).map(|x| x & !(a - 1));
    for i in 0..e_phnum {
        let Some(off) = i.checked_mul(e_phentsize).and_then(|d| e_phoff.checked_add(d)) else {
            break;
        };
        let Some(ph_end) = off.checked_add(56) else {
            break;
        };
        if ph_end > elf.len() {
            break;
        }
        let p_type = u32::from_le_bytes(elf[off..off + 4].try_into().unwrap());
        if p_type != PT_NOTE && p_type != PT_GNU_PROPERTY {
            continue;
        }
        let Ok(p_offset) = usize::try_from(u64::from_le_bytes(elf[off + 8..off + 16].try_into().unwrap())) else {
            continue;
        };
        let Ok(p_filesz) = usize::try_from(u64::from_le_bytes(elf[off + 32..off + 40].try_into().unwrap())) else {
            continue;
        };
        let Ok(p_align) = usize::try_from(u64::from_le_bytes(elf[off + 48..off + 56].try_into().unwrap())) else {
            continue;
        };
        let note_align = if p_type == PT_GNU_PROPERTY || p_align >= 8 { 8 } else { 4 };
        let mut cur = p_offset;
        let end = p_offset.saturating_add(p_filesz).min(elf.len());
        while let Some(name_start) = cur.checked_add(12) {
            if name_start > end {
                break;
            }
            let Ok(namesz) = usize::try_from(u32::from_le_bytes(elf[cur..cur + 4].try_into().unwrap())) else {
                break;
            };
            let Ok(descsz) = usize::try_from(u32::from_le_bytes(elf[cur + 4..cur + 8].try_into().unwrap())) else {
                break;
            };
            let ntype = u32::from_le_bytes(elf[cur + 8..cur + 12].try_into().unwrap());
            let Some(name_end) = name_start.checked_add(namesz) else {
                break;
            };
            let Some(desc_start) = align_up(name_end, note_align) else {
                break;
            };
            let Some(desc_end) = desc_start.checked_add(descsz) else {
                break;
            };
            if desc_end > end {
                break;
            }
            if ntype == NT_GNU_PROPERTY_TYPE_0 && &elf[name_start..name_end] == b"GNU\0" {
                let mut p = desc_start;
                while let Some(data_start) = p.checked_add(8) {
                    if data_start > desc_end {
                        break;
                    }
                    let pr_type = u32::from_le_bytes(elf[p..p + 4].try_into().unwrap());
                    let Ok(pr_datasz) = usize::try_from(u32::from_le_bytes(elf[p + 4..p + 8].try_into().unwrap()))
                    else {
                        break;
                    };
                    let Some(data_end) = data_start.checked_add(pr_datasz) else {
                        break;
                    };
                    if data_end > desc_end {
                        break;
                    }
                    if pr_type == GNU_PROPERTY_X86_FEATURE_1_AND && pr_datasz == 4 {
                        let bits = u32::from_le_bytes(elf[data_start..data_start + 4].try_into().unwrap());
                        if (bits & GNU_PROPERTY_X86_FEATURE_1_SHSTK) != 0 {
                            return true;
                        }
                    }
                    let Some(next_p) = align_up(data_end, 8) else {
                        break;
                    };
                    p = next_p;
                }
            }
            let Some(next_cur) = align_up(desc_end, note_align) else {
                break;
            };
            cur = next_cur;
        }
    }
    false
}

#[cfg(feature = "compiler")]
type WatchedFile = (PathBuf, PathBuf, BinStamp);

#[cfg(feature = "compiler")]
const SYS_HEADER_PROBE_C: &[u8] = b"#if defined(__linux__) && !defined(_GNU_SOURCE)\n\
#define _GNU_SOURCE\n\
#endif\n\
#include <stdint.h>\n\
#include <stddef.h>\n\
#include <stdlib.h>\n\
#include <string.h>\n\
#include <math.h>\n\
#if defined(__linux__)\n\
#include <pthread.h>\n\
#endif\n";

#[cfg(feature = "compiler")]
fn clean_cc_cmd(program: impl AsRef<std::ffi::OsStr>) -> Command {
    let mut cmd = Command::new(program);
    cmd.env_clear();
    if let Some(path) = sanitized_path_env() {
        cmd.env("PATH", path);
    }
    for var in ["HOME", "TMPDIR"] {
        if let Some(v) = std::env::var_os(var) {
            cmd.env(var, v);
        }
    }
    cmd
}

#[cfg(feature = "compiler")]
const LINK_PROBE_C: &[u8] = b"#if defined(__linux__) && !defined(_GNU_SOURCE)\n\
#define _GNU_SOURCE\n\
#endif\n\
#include <math.h>\n\
#include <stdlib.h>\n\
#include <string.h>\n\
#if defined(__linux__)\n\
#include <pthread.h>\n\
#endif\n\
int wt_cet_probe(size_t n) {\n\
    void *p = malloc(n ? n : 1);\n\
    if (!p) return 0;\n\
    memset(p, 0, 1);\n\
    int r = memcmp(p, p, 1);\n\
    free(p);\n\
    volatile float x = (float)n + 0.5f;\n\
    r += (int)truncf(x);\n\
#if defined(__linux__)\n\
    pthread_attr_t a;\n\
    if (pthread_getattr_np(pthread_self(), &a) == 0) pthread_attr_destroy(&a);\n\
#endif\n\
    return r;\n\
}\n";

#[cfg(feature = "compiler")]
fn stamp_len(stamp: BinStamp) -> u64 {
    #[cfg(unix)]
    {
        stamp.2
    }
    #[cfg(not(unix))]
    {
        stamp.0
    }
}

#[cfg(feature = "compiler")]
fn hash_stamp(h: &mut sha2::Sha256, stamp: BinStamp) {
    use sha2::Digest;
    #[cfg(unix)]
    {
        h.update(stamp.0.to_le_bytes());
        h.update(stamp.1.to_le_bytes());
        h.update(stamp.2.to_le_bytes());
        h.update(stamp.3.to_le_bytes());
        h.update(stamp.4.to_le_bytes());
        h.update(stamp.5.to_le_bytes());
        h.update(stamp.6.to_le_bytes());
    }
    #[cfg(not(unix))]
    {
        h.update(stamp.0.to_le_bytes());
        h.update(stamp.1.to_le_bytes());
        h.update(stamp.2.to_le_bytes());
    }
}

/// Resolves an auxiliary toolchain program via `cc -print-prog-name=<prog>` (with `PATH` fallback
/// when `allow_path_fallback` is true). Fails closed on non-zero exit or relative non-bare paths.
#[cfg(feature = "compiler")]
fn resolve_driver_prog(
    cc_path: &Path,
    prog: &str,
    allow_path_fallback: bool,
    scratch_dir: &Path,
    deadline: Instant,
) -> Result<Option<WatchedFile>> {
    let mut cmd = clean_cc_cmd(cc_path);
    cmd.arg(format!("-print-prog-name={prog}"));
    let out = run_cmd_output_with_deadline(
        cmd,
        scratch_dir,
        &format!("{} -print-prog-name={prog}", cc_path.display()),
        deadline,
    )?;
    ensure!(
        out.status.success(),
        "{} -print-prog-name={prog} failed ({}): {}",
        cc_path.display(),
        out.status,
        String::from_utf8_lossy(&out.stderr).trim()
    );
    let printed = String::from_utf8_lossy(&out.stdout).trim().to_string();
    ensure!(!printed.is_empty(), "{} -print-prog-name={prog} returned empty output", cc_path.display());
    let candidate = if is_bare_prog_name(&printed) {
        if allow_path_fallback {
            find_in_path(&printed).or_else(|| find_in_path(prog))
        } else {
            return Ok(None);
        }
    } else {
        let p = PathBuf::from(&printed);
        ensure!(
            p.is_absolute() && is_executable_file(&p),
            "{} -print-prog-name={prog} returned non-absolute or non-executable path {printed:?}",
            cc_path.display()
        );
        Some(p)
    };
    let Some(cand) = candidate else { return Ok(None) };
    let canon = std::fs::canonicalize(&cand).with_context(|| format!("canonicalizing {prog} {}", cand.display()))?;
    let stamp = bin_stamp(&canon)?;
    Ok(Some((cand, canon, stamp)))
}

#[cfg(feature = "compiler")]
fn resolve_driver_linker(cc_path: &Path, has_lld: bool, scratch_dir: &Path, deadline: Instant) -> Result<WatchedFile> {
    let prog = if has_lld { "ld.lld" } else { "ld" };
    resolve_driver_prog(cc_path, prog, true, scratch_dir, deadline)?
        .ok_or_else(|| anyhow!("could not resolve driver linker {prog} for C compiler {}", cc_path.display()))
}

#[cfg(all(feature = "compiler", target_os = "linux"))]
fn elf64_pt_interp(elf_path: &Path) -> Result<Option<PathBuf>> {
    let mut f = std::fs::File::open(elf_path).with_context(|| format!("opening {}", elf_path.display()))?;
    let file_len = f.metadata().with_context(|| format!("stat {}", elf_path.display()))?.len();
    if file_len < 4 {
        return Ok(None);
    }
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).with_context(|| format!("reading {}", elf_path.display()))?;
    if &magic != b"\x7fELF" {
        return Ok(None);
    }
    ensure!(file_len >= 64, "{}: truncated ELF header ({file_len} bytes)", elf_path.display());
    f.seek(SeekFrom::Start(0)).with_context(|| format!("seeking {}", elf_path.display()))?;
    let mut ehdr = [0u8; 64];
    f.read_exact(&mut ehdr).with_context(|| format!("reading ELF header {}", elf_path.display()))?;
    ensure!(
        ehdr[4] == 2 && ehdr[5] == 1,
        "{}: unsupported ELF class/endianness ({}, {}) for shared-library tracking",
        elf_path.display(),
        ehdr[4],
        ehdr[5]
    );
    let e_phoff = u64::from_le_bytes(ehdr[32..40].try_into().unwrap());
    let e_phentsize = u64::from(u16::from_le_bytes(ehdr[54..56].try_into().unwrap()));
    let e_phnum = u64::from(u16::from_le_bytes(ehdr[56..58].try_into().unwrap()));
    if e_phnum == 0 {
        return Ok(None);
    }
    ensure!(
        e_phentsize >= 56 && e_phnum <= 4096,
        "{}: invalid ELF program header table (phentsize={e_phentsize}, phnum={e_phnum})",
        elf_path.display()
    );
    let ph_table_bytes = e_phentsize
        .checked_mul(e_phnum)
        .ok_or_else(|| anyhow!("{}: ELF program header size overflow", elf_path.display()))?;
    let ph_end = e_phoff
        .checked_add(ph_table_bytes)
        .ok_or_else(|| anyhow!("{}: ELF program header offset overflow", elf_path.display()))?;
    ensure!(ph_end <= file_len, "{}: ELF program headers exceed file size", elf_path.display());
    f.seek(SeekFrom::Start(e_phoff)).with_context(|| format!("seeking phdrs in {}", elf_path.display()))?;
    let mut phdrs = vec![0u8; usize::try_from(ph_table_bytes)?];
    f.read_exact(&mut phdrs).with_context(|| format!("reading phdrs in {}", elf_path.display()))?;
    const PT_DYNAMIC: u32 = 2;
    const PT_INTERP: u32 = 3;
    let stride = usize::try_from(e_phentsize)?;
    let mut has_dynamic = false;
    let mut interp_span: Option<(u64, u64)> = None;
    for i in 0..usize::try_from(e_phnum)? {
        let off = i * stride;
        let p_type = u32::from_le_bytes(phdrs[off..off + 4].try_into().unwrap());
        if p_type == PT_DYNAMIC {
            has_dynamic = true;
        } else if p_type == PT_INTERP {
            let p_offset = u64::from_le_bytes(phdrs[off + 8..off + 16].try_into().unwrap());
            let p_filesz = u64::from_le_bytes(phdrs[off + 32..off + 40].try_into().unwrap());
            interp_span = Some((p_offset, p_filesz));
        }
    }
    let Some((p_offset, p_filesz)) = interp_span else {
        ensure!(!has_dynamic, "{}: dynamic ELF has PT_DYNAMIC but no PT_INTERP", elf_path.display());
        return Ok(None);
    };
    ensure!(
        p_filesz > 0 && p_filesz <= 4096 && p_offset.checked_add(p_filesz).is_some_and(|e| e <= file_len),
        "{}: invalid PT_INTERP span ({p_offset}, {p_filesz})",
        elf_path.display()
    );
    f.seek(SeekFrom::Start(p_offset)).with_context(|| format!("seeking PT_INTERP in {}", elf_path.display()))?;
    let mut raw = vec![0u8; usize::try_from(p_filesz)?];
    f.read_exact(&mut raw).with_context(|| format!("reading PT_INTERP in {}", elf_path.display()))?;
    let trimmed = raw.split(|&b| b == 0).next().unwrap_or(&raw);
    let s = std::str::from_utf8(trimmed).map_err(|_| anyhow!("{}: non-UTF-8 PT_INTERP path", elf_path.display()))?;
    let p = PathBuf::from(s);
    ensure!(
        p.is_absolute() && is_executable_file(&p),
        "{}: PT_INTERP {s:?} is not an absolute executable",
        elf_path.display()
    );
    Ok(Some(p))
}

#[cfg(feature = "compiler")]
fn probe_dynamic_shared_libs(elf_path: &Path, scratch_dir: &Path, deadline: Instant) -> Result<Vec<WatchedFile>> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (elf_path, scratch_dir, deadline);
        Ok(Vec::new())
    }
    #[cfg(target_os = "linux")]
    {
        let Some(interp) = elf64_pt_interp(elf_path)? else {
            return Ok(Vec::new());
        };
        let mut cmd = clean_cc_cmd(&interp);
        cmd.arg("--list").arg(elf_path);
        let out = run_cmd_output_with_deadline(
            cmd,
            scratch_dir,
            &format!("{} --list {}", interp.display(), elf_path.display()),
            deadline,
        )?;
        ensure!(
            out.status.success(),
            "{} --list {} failed: {}",
            interp.display(),
            elf_path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
        let mut libs = Vec::new();
        let interp_canon = std::fs::canonicalize(&interp)
            .with_context(|| format!("canonicalizing dynamic linker {}", interp.display()))?;
        let interp_stamp = bin_stamp(&interp_canon)?;
        libs.push((interp, interp_canon, interp_stamp));
        for raw_line in String::from_utf8_lossy(&out.stdout).lines() {
            let line = raw_line.trim();
            if line.is_empty()
                || line.starts_with("linux-vdso.")
                || line.starts_with("linux-gate.")
                || line == "statically linked"
            {
                continue;
            }
            let rhs = if let Some((_, after)) = line.split_once(" => ") { after.trim() } else { line };
            let path_part = match rhs.rfind(" (0x") {
                Some(idx) => rhs[..idx].trim(),
                None => rhs.trim(),
            };
            let p = Path::new(path_part);
            ensure!(
                path_part != "not found" && p.is_absolute() && p.is_file(),
                "unresolved or non-absolute shared library for {}: {line}",
                elf_path.display()
            );
            let canon =
                std::fs::canonicalize(p).with_context(|| format!("canonicalizing shared library {}", p.display()))?;
            let stamp = bin_stamp(&canon)?;
            libs.push((p.to_path_buf(), canon, stamp));
        }
        libs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        libs.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
        Ok(libs)
    }
}

#[cfg(feature = "compiler")]
fn hash_shared_lib_sample(h: &mut sha2::Sha256, canon: &Path, stamp: BinStamp) -> Result<()> {
    use sha2::Digest;
    let path_bytes = canon.as_os_str().as_encoded_bytes();
    h.update((path_bytes.len() as u64).to_le_bytes());
    h.update(path_bytes);
    hash_stamp(h, stamp);
    let mut f = std::fs::File::open(canon).with_context(|| format!("opening {}", canon.display()))?;
    let total_len = f.metadata().with_context(|| format!("stat {}", canon.display()))?.len();
    ensure!(
        total_len == stamp_len(stamp),
        "shared library {} changed size during fingerprint ({} -> {total_len})",
        canon.display(),
        stamp_len(stamp)
    );
    let mut head = Vec::new();
    (&mut f).take(65536).read_to_end(&mut head).with_context(|| format!("reading head of {}", canon.display()))?;
    h.update((head.len() as u64).to_le_bytes());
    h.update(&head);
    if total_len > 65536 {
        f.seek(SeekFrom::Start(total_len - 65536)).with_context(|| format!("seeking tail of {}", canon.display()))?;
        let mut tail = Vec::new();
        (&mut f).take(65536).read_to_end(&mut tail).with_context(|| format!("reading tail of {}", canon.display()))?;
        h.update((tail.len() as u64).to_le_bytes());
        h.update(&tail);
    } else {
        h.update(0u64.to_le_bytes());
    }
    Ok(())
}

#[cfg(feature = "compiler")]
fn hash_shared_lib_set(h: &mut sha2::Sha256, libs: &[WatchedFile]) -> Result<()> {
    use sha2::Digest;
    let mut unique: Vec<(&Path, BinStamp)> = Vec::new();
    for (_, canon, stamp) in libs {
        if unique.last().is_none_or(|(prev, _)| *prev != canon.as_path()) {
            unique.push((canon.as_path(), *stamp));
        }
    }
    h.update((unique.len() as u64).to_le_bytes());
    for (canon, stamp) in unique {
        hash_shared_lib_sample(h, canon, stamp)?;
    }
    Ok(())
}

#[cfg(feature = "compiler")]
fn hash_compiler_binaries(
    cc_path: &Path,
    canon_cc: &Path,
    canon_cc1: Option<&Path>,
    canon_as: Option<&Path>,
    shared_libs: &[WatchedFile],
) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"cc-bin-v2\0");
    let argv0 = cc_path.file_name().map_or(&[][..], |n| n.as_encoded_bytes());
    h.update((argv0.len() as u64).to_le_bytes());
    h.update(argv0);
    let cc_bytes = std::fs::read(canon_cc).with_context(|| format!("reading {}", canon_cc.display()))?;
    h.update((cc_bytes.len() as u64).to_le_bytes());
    h.update(&cc_bytes);
    for opt_bin in [canon_cc1, canon_as] {
        if let Some(p) = opt_bin {
            let bytes = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
            h.update([1u8]);
            h.update((bytes.len() as u64).to_le_bytes());
            h.update(&bytes);
        } else {
            h.update([0u8]);
        }
    }
    hash_shared_lib_set(&mut h, shared_libs)?;
    Ok(format!("{:x}", h.finalize()))
}

#[cfg(feature = "compiler")]
fn hash_linker_binaries(
    linker_path: &Path,
    canon_linker: &Path,
    canon_collect2: Option<&Path>,
    shared_libs: &[WatchedFile],
) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"ld-bin-v2\0");
    let argv0 = linker_path.file_name().map_or(&[][..], |n| n.as_encoded_bytes());
    h.update((argv0.len() as u64).to_le_bytes());
    h.update(argv0);
    let ld_bytes = std::fs::read(canon_linker).with_context(|| format!("reading {}", canon_linker.display()))?;
    h.update((ld_bytes.len() as u64).to_le_bytes());
    h.update(&ld_bytes);
    if let Some(p) = canon_collect2 {
        let bytes = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
        h.update([1u8]);
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(&bytes);
    } else {
        h.update([0u8]);
    }
    hash_shared_lib_set(&mut h, shared_libs)?;
    Ok(format!("{:x}", h.finalize()))
}

#[cfg(feature = "compiler")]
fn parse_ld_trace_files(stdout: &[u8], stderr: &[u8]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for buf in [stdout, stderr] {
        for raw_line in String::from_utf8_lossy(buf).lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let cand = if let Some(rest) = line.strip_prefix('(')
                && let Some((inner, _)) = rest.split_once(')')
            {
                inner.trim()
            } else if let Some((_, rest)) = line.rsplit_once(" (")
                && let Some(inner) = rest.strip_suffix(')')
            {
                inner.trim()
            } else if line.starts_with('/')
                && line.ends_with(')')
                && let Some((arch, _)) = line.split_once('(')
            {
                arch.trim()
            } else {
                line
            };
            let p = Path::new(cand);
            if p.is_absolute() && p.is_file() {
                out.push(p.to_path_buf());
            }
        }
    }
    out
}

#[cfg(feature = "compiler")]
fn hash_watched_files(domain: &[u8], files: &mut Vec<WatchedFile>) -> Result<String> {
    use sha2::{Digest, Sha256};
    files.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    files.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    let mut h = Sha256::new();
    h.update(domain);
    let mut last_canon: Option<&Path> = None;
    for (_, canon, _) in files.iter() {
        if last_canon == Some(canon.as_path()) {
            continue;
        }
        last_canon = Some(canon.as_path());
        let path_bytes = canon.as_os_str().as_encoded_bytes();
        h.update((path_bytes.len() as u64).to_le_bytes());
        h.update(path_bytes);
        let bytes = std::fs::read(canon).with_context(|| format!("reading {}", canon.display()))?;
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(&bytes);
    }
    Ok(format!("{:x}", h.finalize()))
}

#[cfg(feature = "compiler")]
fn verify_watched_files(watched: &[WatchedFile]) -> bool {
    watched.iter().all(|(orig, canon, stamp)| {
        std::fs::canonicalize(orig).ok().as_ref() == Some(canon) && bin_stamp(canon).is_ok_and(|s| s == *stamp)
    })
}

#[cfg(feature = "compiler")]
fn extract_linemarker_path(line: &str) -> Option<String> {
    let rest = line.strip_prefix("# ")?;
    let (num, after_num) = rest.split_once(' ')?;
    if num.is_empty() || !num.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut chars = after_num.strip_prefix('"')?.chars();
    let mut path = String::new();
    while let Some(c) = chars.next() {
        if c == '"' {
            return Some(path);
        }
        if c == '\\' {
            let esc = chars.next()?;
            path.push(esc);
        } else {
            path.push(c);
        }
    }
    None
}

#[cfg(feature = "compiler")]
fn probe_sys_headers(
    cc_path: &Path,
    target_cpu: &str,
    probe_dir: &Path,
    deadline: Instant,
) -> Result<(String, Vec<WatchedFile>)> {
    let (mut hdr_file, hdr_path) = create_private_temp_file(probe_dir, "hdrs", ".c")?;
    let _rm_hdr = RemoveOnDrop(hdr_path.clone());
    hdr_file.write_all(SYS_HEADER_PROBE_C)?;
    drop(hdr_file);
    let canon_src = std::fs::canonicalize(&hdr_path)?;
    let mut watched = Vec::new();
    let mut record_header = |path_str: &str, opt_flag: &str| -> Result<()> {
        let dep = PathBuf::from(path_str);
        ensure!(
            dep.is_absolute() && dep.is_file(),
            "{} {opt_flag} -H/-E reported non-absolute or missing header: {path_str:?}",
            cc_path.display()
        );
        let canon =
            std::fs::canonicalize(&dep).with_context(|| format!("canonicalizing system header {}", dep.display()))?;
        if canon == canon_src {
            return Ok(());
        }
        let stamp = bin_stamp(&canon)?;
        watched.push((dep, canon, stamp));
        Ok(())
    };
    for opt_flag in ["-O2", "-O0"] {
        let mut cmd = clean_cc_cmd(cc_path);
        cmd.args([opt_flag, "-fPIC", &format!("-march={target_cpu}"), "-H", "-E", "-x", "c"]).arg(&hdr_path);
        let out = run_cmd_output_with_deadline(
            cmd,
            probe_dir,
            &format!("{} {opt_flag} -H -E system headers", cc_path.display()),
            deadline,
        )?;
        ensure!(
            out.status.success(),
            "{} {opt_flag} -H -E system header probe failed: {}",
            cc_path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
        for raw_line in String::from_utf8_lossy(&out.stderr).lines() {
            let trimmed = raw_line.trim_end_matches('\r');
            let rest = trimmed.trim_start_matches('.');
            if rest.len() < trimmed.len()
                && let Some(path_str) = rest.strip_prefix(' ')
            {
                record_header(path_str, opt_flag)?;
            }
        }
        for raw_line in String::from_utf8_lossy(&out.stdout).lines() {
            if let Some(path_str) = extract_linemarker_path(raw_line.trim_end_matches('\r'))
                && path_str.starts_with('/')
            {
                record_header(&path_str, opt_flag)?;
            }
        }
    }
    #[cfg(unix)]
    ensure!(!watched.is_empty(), "{} -H -E returned no system headers", cc_path.display());
    let digest = hash_watched_files(b"sys-headers-v3\0", &mut watched)?;
    Ok((digest, watched))
}

#[cfg(feature = "compiler")]
fn probe_link_env_and_cet(
    cc_path: &Path,
    has_lld: bool,
    probe_dir: &Path,
    deadline: Instant,
) -> Result<(bool, String, Vec<WatchedFile>)> {
    let (mut c_file, c_path) = create_private_temp_file(probe_dir, "cet", ".c")?;
    let _rm_c = RemoveOnDrop(c_path.clone());
    c_file.write_all(LINK_PROBE_C)?;
    drop(c_file);

    struct LinkProbeOut {
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        so_bytes: Vec<u8>,
        canon_obj: PathBuf,
    }
    let compile_and_link = |with_cet: bool| -> Result<Option<LinkProbeOut>> {
        let (o_file, o_path) = create_private_temp_file(probe_dir, "cet", ".o")?;
        drop(o_file);
        let _rm_o = RemoveOnDrop(o_path.clone());
        let canon_o = std::fs::canonicalize(&o_path)?;
        let mut c_cmd = clean_cc_cmd(cc_path);
        c_cmd.args(["-c", "-O0", "-fPIC"]);
        if with_cet {
            c_cmd.arg("-fcf-protection=return");
        }
        c_cmd.arg(&c_path).arg("-o").arg(&o_path);
        let c_out =
            run_cmd_output_with_deadline(c_cmd, probe_dir, &format!("{} probe compile", cc_path.display()), deadline)?;
        if !c_out.status.success() {
            return Ok(None);
        }
        let (so_file, so_path) = create_private_temp_file(probe_dir, "cet", ".so")?;
        drop(so_file);
        let _rm_so = RemoveOnDrop(so_path.clone());
        let mut l_cmd = clean_cc_cmd(cc_path);
        l_cmd.args(["-shared", "-fPIC", "-Wl,-t"]);
        if with_cet {
            l_cmd.arg("-fcf-protection=return");
        }
        if has_lld {
            l_cmd.arg("-fuse-ld=lld");
        }
        l_cmd.arg(&o_path).arg("-o").arg(&so_path);
        #[cfg(target_os = "linux")]
        l_cmd.arg("-pthread");
        #[cfg(unix)]
        l_cmd.arg("-lm");
        let l_out =
            run_cmd_output_with_deadline(l_cmd, probe_dir, &format!("{} probe link", cc_path.display()), deadline)?;
        if !l_out.status.success() {
            return Ok(None);
        }
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        let so_bytes = std::fs::read(&so_path)?;
        Ok(Some(LinkProbeOut {
            stdout: l_out.stdout,
            stderr: l_out.stderr,
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            so_bytes,
            canon_obj: canon_o,
        }))
    };

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    let (has_cet_return, probe_out) = if let Some(out) = compile_and_link(true)?
        && elf64_has_x86_shstk_property(&out.so_bytes)
    {
        (true, out)
    } else {
        let out = compile_and_link(false)?.ok_or_else(|| anyhow!("{} -shared probe link failed", cc_path.display()))?;
        (false, out)
    };
    #[cfg(all(not(target_os = "linux"), target_arch = "x86_64"))]
    let (has_cet_return, probe_out) = if let Some(out) = compile_and_link(true)? {
        (true, out)
    } else {
        let out = compile_and_link(false)?.ok_or_else(|| anyhow!("{} -shared probe link failed", cc_path.display()))?;
        (false, out)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let (has_cet_return, probe_out) = {
        let out = compile_and_link(false)?.ok_or_else(|| anyhow!("{} -shared probe link failed", cc_path.display()))?;
        (false, out)
    };
    let LinkProbeOut { stdout, stderr, canon_obj: canon_o, .. } = probe_out;

    let canon_probe_dir = std::fs::canonicalize(probe_dir).ok();
    let mut watched = Vec::new();
    for path in parse_ld_trace_files(&stdout, &stderr) {
        let canon =
            std::fs::canonicalize(&path).with_context(|| format!("canonicalizing link input {}", path.display()))?;
        if canon == canon_o {
            continue;
        }
        if let Some(ref pd) = canon_probe_dir
            && canon.starts_with(pd)
        {
            continue;
        }
        let stamp = bin_stamp(&canon)?;
        watched.push((path, canon, stamp));
    }
    #[cfg(target_os = "linux")]
    ensure!(!watched.is_empty(), "{} -Wl,-t returned no link inputs", cc_path.display());
    let digest = hash_watched_files(b"link-env-v3\0", &mut watched)?;
    Ok((has_cet_return, digest, watched))
}

/// Computes the exact same C compiler, `cc1`, `as`, dynamic shared-library, driver-selected linker
/// and `collect2`, transitive system-header (`-O2` and `-O0`), and CRT/link-environment identity
/// as `wtime-aot`'s `cgen.rs` and `build.rs`.
#[cfg(feature = "compiler")]
type CcIdentityKey = (
    &'static str,
    PathBuf,
    PathBuf,
    BinStamp,
    Option<WatchedFile>,
    Option<WatchedFile>,
    Option<WatchedFile>,
    String,
    String,
);
#[cfg(feature = "compiler")]
#[derive(Clone)]
struct ResolvedCcIdentity {
    cc_name: &'static str,
    cc_path: PathBuf,
    canon_cc: PathBuf,
    stamp: BinStamp,
    cc1_file: Option<WatchedFile>,
    as_file: Option<WatchedFile>,
    collect2_file: Option<WatchedFile>,
    lld_key: Option<WatchedFile>,
    has_lld: bool,
    linker_path: PathBuf,
    canon_linker: PathBuf,
    linker_stamp: BinStamp,
    cc_sha256: String,
    linker_sha256: String,
    sys_headers_sha256: String,
    link_env_sha256: String,
    watched_files: Vec<WatchedFile>,
    cc_ver: String,
    cc_target: String,
    has_cet_return: bool,
    has_stack_clash: bool,
}

#[cfg(feature = "compiler")]
fn c_compiler_identity(target_cpu: &str, canon_cache: &Path, deadline: Instant) -> Result<ResolvedCcIdentity> {
    use sha2::{Digest, Sha256};
    static CACHE: OnceLock<Mutex<std::collections::HashMap<CcIdentityKey, ResolvedCcIdentity>>> = OnceLock::new();
    let host_fp = host_cpu_fingerprint().to_string();
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let probe_dir = ensure_private_cache_dir(&canon_cache.join("cc-probe"))?;
    let lld_key = find_in_path("ld.lld")
        .and_then(|p| std::fs::canonicalize(&p).ok().map(|c| (p, c)))
        .and_then(|(p, c)| bin_stamp(&c).ok().map(|s| (p, c, s)));
    let ld_key = find_in_path("ld")
        .and_then(|p| std::fs::canonicalize(&p).ok().map(|c| (p, c)))
        .and_then(|(p, c)| bin_stamp(&c).ok().map(|s| (p, c, s)));
    let as_key = find_in_path("as")
        .and_then(|p| std::fs::canonicalize(&p).ok().map(|c| (p, c)))
        .and_then(|(p, c)| bin_stamp(&c).ok().map(|s| (p, c, s)));
    let is_sha64 = |s: &str| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit());
    for cc in ["clang", "cc", "gcc"] {
        let Some(p) = find_in_path(cc) else { continue };
        let Ok(canon_cc) = std::fs::canonicalize(&p) else { continue };
        let Ok(stamp) = bin_stamp(&canon_cc) else { continue };
        let mem_key = (
            cc,
            p.clone(),
            canon_cc.clone(),
            stamp,
            lld_key.clone(),
            ld_key.clone(),
            as_key.clone(),
            host_fp.clone(),
            target_cpu.to_string(),
        );
        if let Some(hit) = cache.lock().unwrap_or_else(|p| p.into_inner()).get(&mem_key).cloned()
            && std::fs::canonicalize(&hit.linker_path).ok().as_ref() == Some(&hit.canon_linker)
            && bin_stamp(&hit.canon_linker).is_ok_and(|s| s == hit.linker_stamp)
            && verify_watched_files(&hit.watched_files)
        {
            return Ok(hit);
        }

        let probe_key_line = format!(
            "v7|{cc}|{}|{}|{:?}|{:?}|{:?}|{:?}|{}|{}",
            p.display(),
            canon_cc.display(),
            stamp,
            lld_key,
            ld_key,
            as_key,
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
            if owner_and_mode_ok && let Ok(text) = std::fs::read_to_string(&probe_file) {
                let mut lines = text.lines();
                if let (
                    Some(k),
                    Some(ver),
                    Some(digest),
                    Some(cet),
                    Some(sc),
                    Some(cc_sha),
                    Some(linker_orig_str),
                    Some(linker_canon_str),
                    Some(linker_stamp_str),
                    Some(linker_sha),
                    Some(has_lld_str),
                    Some(sys_hdrs_sha),
                    Some(link_env_sha),
                    Some(cc1_line),
                    Some(as_line),
                    Some(collect2_line),
                    Some(n_watched_str),
                ) = (
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                ) && k == probe_key_line
                    && is_sha64(digest)
                    && is_sha64(cc_sha)
                    && is_sha64(linker_sha)
                    && is_sha64(sys_hdrs_sha)
                    && is_sha64(link_env_sha)
                    && (cet == "0" || cet == "1")
                    && (sc == "0" || sc == "1")
                    && (has_lld_str == "0" || has_lld_str == "1")
                    && !linker_orig_str.is_empty()
                    && !linker_canon_str.is_empty()
                    && let Ok(n_watched) = n_watched_str.parse::<usize>()
                    && n_watched <= 4096
                    && (!cfg!(unix) || n_watched > 0)
                {
                    let linker_path = PathBuf::from(linker_orig_str);
                    let canon_linker = PathBuf::from(linker_canon_str);
                    let linker_ok = (std::fs::canonicalize(&linker_path).ok().as_ref() == Some(&canon_linker))
                        .then(|| bin_stamp(&canon_linker).ok().filter(|s| format!("{s:?}") == linker_stamp_str))
                        .flatten();
                    let parse_watched_line = |line: &str| -> Option<WatchedFile> {
                        let mut parts = line.split('\t');
                        let (orig_s, canon_s, stamp_s) = (parts.next()?, parts.next()?, parts.next()?);
                        if parts.next().is_some() || orig_s.is_empty() || canon_s.is_empty() {
                            return None;
                        }
                        let orig = PathBuf::from(orig_s);
                        let canon = PathBuf::from(canon_s);
                        if std::fs::canonicalize(&orig).ok().as_ref() != Some(&canon) {
                            return None;
                        }
                        let cur_stamp = bin_stamp(&canon).ok()?;
                        if format!("{cur_stamp:?}") != stamp_s {
                            return None;
                        }
                        Some((orig, canon, cur_stamp))
                    };
                    let parse_opt = |line: &str| -> Option<Option<WatchedFile>> {
                        if line == "-" { Some(None) } else { parse_watched_line(line).map(Some) }
                    };
                    if let (Some(cur_linker_stamp), Some(cc1_file), Some(as_file), Some(collect2_file)) =
                        (linker_ok, parse_opt(cc1_line), parse_opt(as_line), parse_opt(collect2_line))
                    {
                        let mut watched_files = Vec::with_capacity(n_watched);
                        let mut all_ok = true;
                        for _ in 0..n_watched {
                            if let Some(w) = lines.next().and_then(parse_watched_line) {
                                watched_files.push(w);
                            } else {
                                all_ok = false;
                                break;
                            }
                        }
                        if all_ok && lines.next().is_none() {
                            let resolved = ResolvedCcIdentity {
                                cc_name: cc,
                                cc_path: p,
                                canon_cc,
                                stamp,
                                cc1_file,
                                as_file,
                                collect2_file,
                                lld_key: lld_key.clone(),
                                has_lld: has_lld_str == "1",
                                linker_path,
                                canon_linker,
                                linker_stamp: cur_linker_stamp,
                                cc_sha256: cc_sha.to_string(),
                                linker_sha256: linker_sha.to_string(),
                                sys_headers_sha256: sys_hdrs_sha.to_string(),
                                link_env_sha256: link_env_sha.to_string(),
                                watched_files,
                                cc_ver: ver.to_string(),
                                cc_target: digest.to_string(),
                                has_cet_return: cet == "1",
                                has_stack_clash: sc == "1",
                            };
                            cache.lock().unwrap_or_else(|p| p.into_inner()).insert(mem_key, resolved.clone());
                            return Ok(resolved);
                        }
                    }
                }
            }
        }

        let mut ver_cmd = clean_cc_cmd(&p);
        ver_cmd.arg("--version");
        let ver_out = run_cmd_output_with_deadline(ver_cmd, &probe_dir, "C compiler --version probe", deadline)?;
        if !ver_out.status.success() {
            continue;
        }
        let cc_ver = String::from_utf8_lossy(&ver_out.stdout).lines().next().unwrap_or(cc).trim().to_string();

        let mut target_cmd = clean_cc_cmd(&p);
        target_cmd.args([&format!("-march={target_cpu}"), "-dM", "-E", "-x", "c", "-"]);
        let out = run_cmd_output_with_deadline(
            target_cmd,
            &probe_dir,
            &format!("probing {} -march={target_cpu}", p.display()),
            deadline,
        )?;
        ensure!(
            out.status.success(),
            "{} -march={target_cpu} probe failed: {}",
            p.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
        let mut lines: Vec<&str> =
            std::str::from_utf8(&out.stdout).unwrap_or("").lines().filter(|l| l.starts_with("#define __")).collect();
        lines.sort_unstable();
        let mut h = Sha256::new();
        h.update(b"cc-target-v1");
        h.update(target_cpu.as_bytes());
        for l in lines {
            h.update(l.as_bytes());
            h.update(b"\n");
        }
        let cc_target = format!("{:x}", h.finalize());

        let has_lld = if let Some((ref lld_prog, _, _)) = lld_key {
            let mut lld_cmd = clean_cc_cmd(lld_prog);
            lld_cmd.arg("--version");
            run_cmd_output_with_deadline(lld_cmd, &probe_dir, "probing ld.lld --version", deadline)?.status.success()
        } else {
            false
        };
        let (linker_path, canon_linker, linker_stamp) = resolve_driver_linker(&p, has_lld, &probe_dir, deadline)?;
        let cc1_file = resolve_driver_prog(&p, "cc1", false, &probe_dir, deadline)?;
        let as_file = resolve_driver_prog(&p, "as", true, &probe_dir, deadline)?;
        let collect2_file = resolve_driver_prog(&p, "collect2", false, &probe_dir, deadline)?;

        let mut cc_libs = probe_dynamic_shared_libs(&canon_cc, &probe_dir, deadline)?;
        if let Some((_, ref c1, _)) = cc1_file {
            cc_libs.extend(probe_dynamic_shared_libs(c1, &probe_dir, deadline)?);
        }
        if let Some((_, ref a, _)) = as_file {
            cc_libs.extend(probe_dynamic_shared_libs(a, &probe_dir, deadline)?);
        }
        cc_libs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        cc_libs.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

        let mut ld_libs = probe_dynamic_shared_libs(&canon_linker, &probe_dir, deadline)?;
        if let Some((_, ref c2, _)) = collect2_file {
            ld_libs.extend(probe_dynamic_shared_libs(c2, &probe_dir, deadline)?);
        }
        ld_libs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        ld_libs.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

        let (sys_headers_sha256, header_files) = probe_sys_headers(&p, target_cpu, &probe_dir, deadline)?;
        let (has_cet_return, link_env_sha256, link_files) = probe_link_env_and_cet(&p, has_lld, &probe_dir, deadline)?;
        let cc_sha256 = hash_compiler_binaries(
            &p,
            &canon_cc,
            cc1_file.as_ref().map(|(_, c, _)| c.as_path()),
            as_file.as_ref().map(|(_, c, _)| c.as_path()),
            &cc_libs,
        )?;
        let linker_sha256 = hash_linker_binaries(
            &linker_path,
            &canon_linker,
            collect2_file.as_ref().map(|(_, c, _)| c.as_path()),
            &ld_libs,
        )?;

        let has_stack_clash = {
            let mut sc_cmd = clean_cc_cmd(&p);
            sc_cmd.args(["-x", "c", "-c", "-fstack-clash-protection", "-o", "/dev/null", "-"]);
            run_cmd_output_with_deadline(sc_cmd, &probe_dir, "probing -fstack-clash-protection", deadline)?
                .status
                .success()
        };

        let mut watched_files = header_files;
        watched_files.extend(link_files);
        watched_files.extend(cc_libs);
        watched_files.extend(ld_libs);
        #[cfg(target_os = "linux")]
        {
            let ld_cache = Path::new("/etc/ld.so.cache");
            if ld_cache.is_file()
                && let Ok(canon) = std::fs::canonicalize(ld_cache)
                && let Ok(st) = bin_stamp(&canon)
            {
                watched_files.push((ld_cache.to_path_buf(), canon, st));
            }
        }
        if let Some(ref f) = cc1_file {
            watched_files.push(f.clone());
        }
        if let Some(ref f) = as_file {
            watched_files.push(f.clone());
        }
        if let Some(ref f) = collect2_file {
            watched_files.push(f.clone());
        }
        watched_files.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        watched_files.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

        fn clean_path_str(path: &Path) -> Option<&str> {
            let s = path.to_str()?;
            (!s.is_empty() && !s.contains('|') && !s.contains('\t') && !s.contains('\n') && !s.contains('\r'))
                .then_some(s)
        }
        let clean_opt_watched = |opt: &Option<WatchedFile>| -> bool {
            opt.as_ref().is_none_or(|(o, c, _)| clean_path_str(o).is_some() && clean_path_str(c).is_some())
        };
        let format_watched = |(orig, canon, st): &WatchedFile| -> Option<String> {
            Some(format!("{}\t{}\t{st:?}", clean_path_str(orig)?, clean_path_str(canon)?))
        };
        let format_opt = |opt: &Option<WatchedFile>| -> Option<String> {
            match opt {
                None => Some("-".to_string()),
                Some(w) => format_watched(w),
            }
        };
        if watched_files.len() <= 4096
            && (!cfg!(unix) || !watched_files.is_empty())
            && clean_path_str(&p).is_some()
            && clean_path_str(&canon_cc).is_some()
            && clean_opt_watched(&lld_key)
            && clean_opt_watched(&ld_key)
            && clean_opt_watched(&as_key)
            && let (Some(l_orig), Some(l_canon)) = (clean_path_str(&linker_path), clean_path_str(&canon_linker))
            && let (Some(cc1_str), Some(as_str), Some(collect2_str)) =
                (format_opt(&cc1_file), format_opt(&as_file), format_opt(&collect2_file))
            && let Some(watched_lines) = watched_files.iter().map(format_watched).collect::<Option<Vec<String>>>()
        {
            let mut body = format!(
                "{probe_key_line}\n{cc_ver}\n{cc_target}\n{}\n{}\n{cc_sha256}\n{l_orig}\n{l_canon}\n{linker_stamp:?}\n{linker_sha256}\n{}\n{sys_headers_sha256}\n{link_env_sha256}\n{cc1_str}\n{as_str}\n{collect2_str}\n{}\n",
                u8::from(has_cet_return),
                u8::from(has_stack_clash),
                u8::from(has_lld),
                watched_lines.len(),
            );
            for line in watched_lines {
                body.push_str(&line);
                body.push('\n');
            }
            if let Ok((mut tmp_file, tmp)) = create_private_temp_file(&probe_dir, "probe", ".tmp") {
                let ok = tmp_file.write_all(body.as_bytes()).is_ok();
                drop(tmp_file);
                if !ok || std::fs::rename(&tmp, &probe_file).is_err() {
                    let _ = std::fs::remove_file(&tmp);
                }
            }
        }

        let resolved = ResolvedCcIdentity {
            cc_name: cc,
            cc_path: p,
            canon_cc,
            stamp,
            cc1_file,
            as_file,
            collect2_file,
            lld_key: lld_key.clone(),
            has_lld,
            linker_path,
            canon_linker,
            linker_stamp,
            cc_sha256,
            linker_sha256,
            sys_headers_sha256,
            link_env_sha256,
            watched_files,
            cc_ver,
            cc_target,
            has_cet_return,
            has_stack_clash,
        };
        cache.lock().unwrap_or_else(|p| p.into_inner()).insert(mem_key, resolved.clone());
        return Ok(resolved);
    }
    bail!("no C compiler found (install clang or gcc)")
}

#[cfg(feature = "compiler")]
fn sanitize_key_part(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' }).collect()
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
type RustcProxyKey = (PathBuf, BinStamp, String);
#[cfg(feature = "compiler")]
#[derive(Clone)]
struct ResolvedRustcIdentity {
    proxy_path: PathBuf,
    canon_real: PathBuf,
    real_stamp: BinStamp,
    fp: String,
}

/// Probes the selected Rust toolchain preserving proxy `argv[0]` (so `rustup` proxies like
/// `/usr/bin/rustc -> /usr/bin/rustup` execute `rustc` rather than `rustup`), locates and stamps
/// the actual `<sysroot>/bin/rustc` executable, and fails loudly if `RUSTUP_TOOLCHAIN` fails
/// (matching `wtime-aot/src/build.rs`).
#[cfg(feature = "compiler")]
fn rustc_identity(
    toolchain: &str,
    scratch_dir: &Path,
    deadline: Instant,
    force_refresh: bool,
) -> Result<ResolvedRustcIdentity> {
    static CACHE: OnceLock<Mutex<std::collections::HashMap<RustcProxyKey, ResolvedRustcIdentity>>> = OnceLock::new();
    let proxy_path = find_in_path("rustc").ok_or_else(|| anyhow!("no rustc found on PATH for --backend rust"))?;
    let canon_proxy = std::fs::canonicalize(&proxy_path)
        .with_context(|| format!("canonicalizing rustc proxy {}", proxy_path.display()))?;
    let proxy_stamp = bin_stamp(&canon_proxy)?;
    let key = (proxy_path.clone(), proxy_stamp, toolchain.to_string());
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if !force_refresh
        && let Some(hit) = cache.lock().unwrap_or_else(|p| p.into_inner()).get(&key).cloned()
        && bin_stamp(&hit.canon_real).is_ok_and(|s| s == hit.real_stamp)
    {
        return Ok(hit);
    }

    let clean_rustc = || {
        // Execute `proxy_path` directly (NOT canonicalized) so rustup preserves `argv[0] = "rustc"`.
        let mut cmd = Command::new(&proxy_path);
        cmd.env_clear();
        if let Some(path) = sanitized_path_env() {
            cmd.env("PATH", path);
        }
        for var in ["HOME", "RUSTUP_HOME", "CARGO_HOME"] {
            if let Some(v) = std::env::var_os(var) {
                cmd.env(var, v);
            }
        }
        cmd.env("RUSTUP_TOOLCHAIN", toolchain);
        cmd
    };

    let mut sysroot_cmd = clean_rustc();
    sysroot_cmd.args(["--print", "sysroot"]);
    let sysroot_out = run_cmd_output_with_deadline(sysroot_cmd, scratch_dir, "rustc --print sysroot", deadline)?;
    ensure!(
        sysroot_out.status.success(),
        "rustc --print sysroot failed for toolchain {toolchain}: {}",
        String::from_utf8_lossy(&sysroot_out.stderr).trim()
    );
    let sysroot = PathBuf::from(String::from_utf8_lossy(&sysroot_out.stdout).trim());
    let real_rustc = sysroot.join("bin").join(if cfg!(windows) { "rustc.exe" } else { "rustc" });
    let canon_real = if is_executable_file(&real_rustc) {
        std::fs::canonicalize(&real_rustc)
            .with_context(|| format!("canonicalizing toolchain rustc {}", real_rustc.display()))?
    } else {
        canon_proxy
    };
    let real_stamp = bin_stamp(&canon_real)?;

    let mut ver_cmd = clean_rustc();
    ver_cmd.arg("-vV");
    let ver_out = run_cmd_output_with_deadline(ver_cmd, scratch_dir, "rustc -vV probe", deadline)?;
    ensure!(
        ver_out.status.success(),
        "rustc -vV failed for toolchain {toolchain}: {}",
        String::from_utf8_lossy(&ver_out.stderr).trim()
    );
    let fp = wasm_sha256_hex(&ver_out.stdout)[..16].to_string();
    let id = ResolvedRustcIdentity { proxy_path, canon_real, real_stamp, fp };
    cache.lock().unwrap_or_else(|p| p.into_inner()).insert(key, id.clone());
    Ok(id)
}

#[cfg(all(feature = "compiler", target_os = "linux", target_arch = "x86_64", target_pointer_width = "64"))]
enum ShstkStatus {
    ConfirmedOff,
    Enabled,
    Unknown(String),
}

/// Inspects the calling Linux thread's CET hardware shadow-stack (`shstk`) state via
/// `/proc/thread-self/status` `x86_Thread_features:` (zero unsafe).
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
            if other.strip_prefix("znver").is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())) =>
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

#[cfg(feature = "compiler")]
fn parse_aarch64_target_cpu_syntax(target_cpu: &str) -> Result<&'static str> {
    match target_cpu {
        "native" | "generic" | "armv8-a" | "armv8.1-a" | "armv8.2-a" | "armv8.3-a" | "armv8.4-a" | "armv8.5-a"
        | "armv8.6-a" | "armv9-a" => Ok("aarch64-v8a"),
        other
            if other.starts_with("neoverse-")
                || other.starts_with("cortex-")
                || other.starts_with("apple-")
                || other.starts_with("graviton") =>
        {
            Ok("aarch64-v8a")
        }
        other => bail!(
            "unsupported PEN_WTIME_AOT_TARGET_CPU={other:?} for aarch64 (expected native, generic, armv8-a, neoverse-*, cortex-*, apple-*, or graviton*)"
        ),
    }
}

#[cfg(feature = "compiler")]
fn resolved_aarch64_isa_tag(target_cpu: &str) -> Result<&'static str> {
    let tag = parse_aarch64_target_cpu_syntax(target_cpu)?;
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64", target_pointer_width = "64")))]
    {
        let _ = tag;
        Err(anyhow::Error::new(X64FallbackError(X64FallbackReason::UnsupportedPlatform(
            "wtime-aot --backend aarch64 is only supported on Linux aarch64 LP64 hosts".into(),
        ))))
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64", target_pointer_width = "64"))]
    {
        Ok(tag)
    }
}

/// Default `wtime-aot` backend when `PEN_WTIME_AOT_BACKEND` is unset (`wtime` v0.1.7).
/// Uses native Linux `"aarch64"` on ARM64 Linux and `"x64"` elsewhere, falling back
/// automatically to `"c"` (`--c-opt auto --target-cpu native`) on [`X64FallbackError`].
#[cfg(all(feature = "compiler", target_os = "linux", target_arch = "aarch64", target_pointer_width = "64"))]
const DEFAULT_AOT_BACKEND: &str = "aarch64";
#[cfg(all(feature = "compiler", not(all(target_os = "linux", target_arch = "aarch64", target_pointer_width = "64"))))]
const DEFAULT_AOT_BACKEND: &str = "x64";

/// Explicit compilation configuration threaded into `load_verified_so_with_config` so
/// default-path fallback (`x64 -> c --c-opt auto --target-cpu native`) never mutates env vars.
#[cfg(feature = "compiler")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct AotCompileConfig {
    backend: String,
    c_opt: String,
    toolchain: String,
    target_cpu: String,
    /// True when `PEN_WTIME_AOT_BACKEND` was explicitly set by the caller (disables fallback).
    explicit_backend: bool,
}

#[cfg(feature = "compiler")]
impl AotCompileConfig {
    fn from_env() -> Self {
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

    fn x64_fallback_to_c(&self) -> Self {
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
#[cfg(feature = "compiler")]
fn load_verified_so_with_fallback<T, F>(
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

#[cfg(feature = "compiler")]
fn classify_build_failure(backend: &str, exit_code: Option<i32>, status_display: &str, stderr: &str) -> anyhow::Error {
    if matches!(backend, "x64" | "aarch64") && exit_code == Some(X64_ADMISSION_REJECTED_EXIT_CODE) {
        anyhow::Error::new(X64FallbackError(X64FallbackReason::AdmissionRejected(stderr.trim().to_string())))
    } else {
        anyhow!("wtime-aot build failed ({status_display}): {stderr}")
    }
}

/// Checks `fast_link` and loads the immutable content-addressed `verified/<digest>.so` while the
/// caller holds `canon_cache/.cache.lock` (shared or exclusive), keeping the verified artifact
/// completely stable across `dlopen`.
#[cfg(feature = "compiler")]
fn try_load_warm_verified_so(
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

#[cfg(feature = "compiler")]
fn load_verified_so(wasm: &[u8], cache: Option<&Path>) -> Result<Library> {
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

#[cfg(feature = "compiler")]
fn load_verified_so_with_config(wasm: &[u8], cache: Option<&Path>, cfg: &AotCompileConfig) -> Result<Library> {
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

#[cfg(feature = "compiler")]
fn wasmparser_val(value: ValueType) -> wasmparser::ValType {
    match value {
        ValueType::F32 => wasmparser::ValType::F32,
        _ => wasmparser::ValType::I32,
    }
}

#[cfg(feature = "compiler")]
struct ValidatedAotWasm {
    memory_limits: Option<(u64, Option<u64>)>,
    has_blot_memory_export: bool,
    has_mut_extern_table: bool,
    expected_imports: Option<Vec<CallbackType>>,
    counter_names: Vec<String>,
    imaged: Result<Vec<String>, String>,
}

/// Validates every import, entrypoint, constant, arena, ownership, and finish-invocation
/// Wasm signature in a single section-skipping pass (skipping the Code section body in O(1))
/// while also collecting counter exports and imaged mutable globals.
#[cfg(feature = "compiler")]
fn validate_aot_wasm(wasm: &[u8], abi: &GuestAbi) -> Result<ValidatedAotWasm> {
    let callbacks = abi.callbacks();
    let mut types: Vec<Option<wasmparser::FuncType>> = Vec::new();
    let mut func_type_idxs: Vec<u32> = Vec::new();
    let mut expected_imports: Vec<CallbackType> = Vec::new();
    let mut global_types: Vec<wasmparser::ValType> = Vec::new();
    let mut mutable_globals: Vec<u32> = Vec::new();
    let mut exported_globals = std::collections::HashMap::new();
    let mut counter_names: Vec<String> = Vec::new();
    let mut imaged_err: Option<String> = None;
    let mut memory_limits: Option<(u64, Option<u64>)> = None;
    let mut has_mut_extern_table = false;
    let mut exports = std::collections::HashMap::new();

    let mut parser = wasmparser::Parser::new(0);
    let mut offset = 0usize;
    loop {
        let rest = &wasm[offset..];
        let (consumed, payload) = match parser.parse(rest, true)? {
            wasmparser::Chunk::Parsed { consumed, payload } => (consumed, payload),
            wasmparser::Chunk::NeedMoreData(_) => bail!("invalid_abi: unexpected EOF in wasm module"),
        };
        match payload {
            wasmparser::Payload::CodeSectionStart { size, .. } => {
                let size = usize::try_from(size).map_err(|_| anyhow!("invalid_abi: code section size overflow"))?;
                let skip =
                    consumed.checked_add(size).ok_or_else(|| anyhow!("invalid_abi: code section size overflow"))?;
                ensure!(skip <= rest.len(), "invalid_abi: truncated code section");
                offset += skip;
                parser.skip_section();
                continue;
            }
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
                    ensure!(
                        import.module == HOST_MODULE,
                        "invalid_abi: unexpected import {}::{}",
                        import.module,
                        import.name
                    );
                    let cb =
                        callbacks.iter().copied().find(|cb| cb.import_name() == import.name).ok_or_else(|| {
                            anyhow!("invalid_abi: unexpected import {}::{}", import.module, import.name)
                        })?;
                    let wasmparser::TypeRef::Func(ti) = import.ty else {
                        bail!("invalid_abi: import {}::{} must be a plain function", import.module, import.name);
                    };
                    let fty = types.get(ti as usize).and_then(|o| o.as_ref()).ok_or_else(|| {
                        anyhow!("invalid_abi: import {}::{} has invalid type index {ti}", import.module, import.name)
                    })?;
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
                        if imaged_err.is_none() {
                            imaged_err = Some("mutable externref tables are not captured by snapshots".into());
                        }
                    }
                }
            }
            wasmparser::Payload::MemorySection(section) => {
                for mem in section {
                    let mem = mem?;
                    ensure!(memory_limits.is_none(), "invalid_abi: wtime-aot supports at most one linear memory");
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
                for (index, g) in section.into_iter().enumerate() {
                    let g = g?;
                    global_types.push(g.ty.content_type);
                    if g.ty.mutable {
                        if !matches!(
                            g.ty.content_type,
                            wasmparser::ValType::I32
                                | wasmparser::ValType::I64
                                | wasmparser::ValType::F32
                                | wasmparser::ValType::F64
                        ) {
                            if imaged_err.is_none() {
                                imaged_err = Some(format!("its mutable global {index} holds a reference"));
                            }
                        } else if let Ok(gi) = u32::try_from(index) {
                            mutable_globals.push(gi);
                        } else if imaged_err.is_none() {
                            imaged_err = Some("global index exceeds u32".into());
                        }
                    }
                }
            }
            wasmparser::Payload::ExportSection(section) => {
                for export in section {
                    let export = export?;
                    if export.kind == wasmparser::ExternalKind::Global {
                        exported_globals.entry(export.index).or_insert_with(|| export.name.to_owned());
                        if export.name.starts_with("blot:") && abi.constants.iter().all(|c| c.name != export.name) {
                            counter_names.push(export.name.to_owned());
                        }
                    }
                    exports.insert(export.name.to_owned(), (export.kind, export.index));
                }
            }
            wasmparser::Payload::End(_) => {
                break;
            }
            _ => {}
        }
        offset += consumed;
    }

    let lookup_func_sig = |fidx: u32| -> Option<&wasmparser::FuncType> {
        func_type_idxs.get(fidx as usize).and_then(|&ti| types.get(ti as usize)).and_then(|o| o.as_ref())
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
            let ok = lookup_func_sig(fidx).is_some_and(|fty| {
                fty.params() == [wasmparser::ValType::I32] && fty.results() == [wasmparser::ValType::I32]
            });
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
    let imaged = if let Some(err) = imaged_err {
        Err(err)
    } else {
        let hidden = mutable_globals.iter().filter(|idx| !exported_globals.contains_key(idx)).count();
        if hidden > 0 {
            Err(format!(
                "{hidden} of its {} mutable globals are not exported, out of the host's reach",
                mutable_globals.len()
            ))
        } else {
            Ok(mutable_globals.iter().map(|idx| exported_globals[idx].clone()).collect())
        }
    };
    Ok(ValidatedAotWasm {
        memory_limits,
        has_blot_memory_export,
        has_mut_extern_table,
        expected_imports: Some(expected_imports),
        counter_names,
        imaged,
    })
}

/// Decodes a static UTF-8 byte slice returned by an AOT library export.
///
/// # Safety
/// If `len > 0`, `ptr` must point to `len` readable immutable bytes that remain live for the
/// lifetime of `_lib` (the loaded `libloading::Library`).
#[cfg(feature = "compiler")]
unsafe fn read_aot_str(_lib: &Library, ptr: *const u8, len: usize) -> Result<&str> {
    if len == 0 {
        return Ok("");
    }
    ensure!(!ptr.is_null(), "invalid_abi: null string pointer with non-zero length {len}");
    // SAFETY: upheld by caller's contract tying `(ptr, len)` to `_lib`.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    std::str::from_utf8(bytes).context("invalid_abi: non-UTF-8 string returned by AOT library")
}

#[cfg(feature = "compiler")]
impl AotModule {
    pub fn compile(wasm: &[u8], abi: &GuestAbi, cache: Option<&Path>) -> Result<(Self, Result<Vec<String>, String>)> {
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
        Ok((
            Self {
                lib: Arc::new(AotLibrary {
                    _lib: lib,
                    syms,
                    imports,
                    counter_names: validated.counter_names,
                    memory_limits: validated.memory_limits,
                    has_blot_memory_export: validated.has_blot_memory_export,
                    has_mut_extern_table: validated.has_mut_extern_table,
                }),
            },
            validated.imaged,
        ))
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
    has_pending_error: AtomicBool,
    has_pending_panic: AtomicBool,
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
                host_box.has_pending_error.store(true, Ordering::Release);
            }
            1
        }
        Err(payload) => {
            // Never unwind across `extern "C"` / generated native frames; stash innermost payload to resume on host side.
            if let Ok(mut state) = host_box.state.lock()
                && state.pending_panic.is_none()
            {
                state.pending_panic = Some(payload);
                host_box.has_pending_panic.store(true, Ordering::Release);
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
    // Short lock #1: extract metadata, callback, and `spare` in one lock so NO lock/reference to
    // `AotHostState` is held across any FFI call (`active_mem_data`, `active_get_global`,
    // `callback`, or `active_allocate`).
    let (signature, lib, uses_arrays, memory_gauge, probe, mut spare, mut callback) = {
        let mut state =
            host_box.state.try_lock().map_err(|_| anyhow!("reentrant_call: host state lock unexpectedly contended"))?;
        let signature =
            *state.lib.imports.get(import_idx as usize).ok_or_else(|| anyhow!("unknown import index {import_idx}"))?;
        let expected = match &state.active {
            Some(active) if token != 0 && active.token == token => active.signature,
            _ => bail!("stale_capability: host call has no active capability reference"),
        };
        ensure!(expected == signature, "capability_signature: host call signature does not match its capability");
        let callback = state
            .active
            .as_mut()
            .and_then(|active| active.callback.take())
            .ok_or_else(|| anyhow!("reentrant_call: cannot reenter an active guest instance"))?;
        (
            signature,
            state.lib.clone(),
            state.uses_arrays,
            state.memory.clone(),
            state.probe.clone(),
            std::mem::take(&mut state.spare),
            callback,
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
            if let Ok(mut state) = host_box.state.lock() {
                if let Some(active) = state.active.as_mut() {
                    active.callback = Some(callback);
                }
                if spare.capacity() > state.spare.capacity() {
                    state.spare = spare;
                }
            }
            return Err(err);
        }
    };
    if uses_arrays && let Some(gauge) = &memory_gauge {
        gauge.store(mem_len, Ordering::Relaxed);
    }
    if let Some(probe) = probe {
        let mem = if uses_arrays { mem_len } else { 0 };
        if probe.wants_counters() && !lib.counter_names.is_empty() {
            let mut vals = Vec::with_capacity(lib.counter_names.len());
            for name in &lib.counter_names {
                let (mut val, mut kind) = (0u64, 0u8);
                let rc = unsafe {
                    (lib.syms.active_get_global)(active_call, name.as_ptr(), name.len(), &mut val, &mut kind)
                };
                let c = if rc == 0 && (kind == 0 || kind == 1) { (val & 0xffff_ffff) as u32 } else { 0 };
                vals.push(u64::from(c));
            }
            probe.called_host(mem, vals.into_iter());
        } else {
            probe.called_host(mem, std::iter::empty());
        }
    }

    let result = callback(argument);
    // Short lock #2: restore callback immediately after invocation, matching Wasmtime so guest
    // `blot:allocate` during result encoding may legally invoke the active callback. Only clone
    // the export name if formatting an error message.
    let needs_export = match &result {
        Ok(val) => val.kind() != signature.result,
        Err(_) => true,
    };
    let export = {
        let mut state =
            host_box.state.try_lock().map_err(|_| anyhow!("reentrant_call: host state lock unexpectedly contended"))?;
        let export = if needs_export {
            state.active.as_ref().map(|active| active.export.clone()).unwrap_or_default()
        } else {
            String::new()
        };
        if let Some(active) = state.active.as_mut() {
            active.callback = Some(callback);
        }
        export
    };
    let result = match result.with_context(|| format!("host_exception: host callback failed during {export}")) {
        Ok(val) => val,
        Err(err) => {
            restore_spare(host_box, spare);
            return Err(err);
        }
    };
    if result.kind() != signature.result {
        restore_spare(host_box, spare);
        bail!("host_result: host callback during {export} returned {}, not {}", result.kind(), signature.result);
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
    // Short lock #3: restore `spare` buffer.
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
    saw_oom: bool,
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
                        bail!(
                            "memory_limit: guest initial memory ({initial_bytes} B) exceeds limit ({} B)",
                            lim.memory
                        );
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
            has_pending_error: AtomicBool::new(false),
            has_pending_panic: AtomicBool::new(false),
        }));
        let mut trap_code = 0u8;
        let handle = unsafe {
            (module.lib.syms.new)(
                host_ptr.cast::<c_void>(),
                Some(host_callback_trampoline),
                limit_pages,
                &mut trap_code,
            )
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
            saw_oom: false,
            entered: false,
            disposed: false,
        })
    }

    pub fn probe(&mut self, probe: Arc<Probe>) {
        probe.counters_named(
            self.lib.counter_names.iter().map(|name| name.trim_start_matches("blot:").to_owned()).collect(),
        );
        self.with_host_mut(|host| host.probe = Some(probe));
    }

    pub fn memory_bytes(&self) -> usize {
        if self.lib.memory_limits.is_none() {
            return 0;
        }
        let mut p: *mut u8 = std::ptr::null_mut();
        let mut n: usize = 0;
        unsafe { (self.lib.syms.mem_data)(self.handle.0, &mut p, &mut n) };
        n
    }

    pub(crate) fn saw_oom(&self) -> bool {
        self.saw_oom
    }

    pub(crate) fn clear_saw_oom(&mut self) {
        self.saw_oom = false;
    }

    pub(crate) fn has_arena(&self) -> bool {
        self.abi.uses_arrays() || self.abi.tasks.is_some()
    }

    pub(crate) fn arena_oom_flag(&self) -> u32 {
        if !self.has_arena() {
            return 0;
        }
        let mut p: *mut u8 = std::ptr::null_mut();
        let mut n: usize = 0;
        unsafe { (self.lib.syms.mem_data)(self.handle.0, &mut p, &mut n) };
        if p.is_null() || n < crate::ARENA_OOM_OFFSET + 4 {
            return 0;
        }
        // SAFETY: `self.handle.0` uniquely owns guest linear memory while no guest call is active.
        let data = unsafe { std::slice::from_raw_parts(p, n) };
        u32::from_le_bytes(data[crate::ARENA_OOM_OFFSET..crate::ARENA_OOM_OFFSET + 4].try_into().unwrap())
    }

    pub(crate) fn reset_arena(&mut self) -> Result<()> {
        if self.has_arena() {
            self.call_reset()?;
        }
        Ok(())
    }

    pub(crate) fn write_words_packet(&mut self, words: &[u32]) -> Result<i32> {
        ensure!(self.has_arena(), "invalid_abi: array value requires an arena");
        ensure!(words.len() < 1_073_741_823, "array byte count exceeds the Wasm32 address range");
        let length = u32::try_from(words.len())?;
        let byte_len = (length + 1).checked_mul(4).ok_or_else(|| anyhow!("packet byte count overflow"))?;
        self.clear_pending();
        let mut ptr_out = 0i32;
        let mut trap_code = 0u8;
        let rc = unsafe {
            (self.lib.syms.call_allocate)(self.handle.0, byte_len.cast_signed(), &mut ptr_out, &mut trap_code)
        };
        self.maybe_resume_panic();
        let pointer = match rc {
            0 => ptr_out,
            1 => {
                if is_oom_trap_code(trap_code) {
                    self.saw_oom = true;
                }
                return Err(self.lib.trap_error(trap_code));
            }
            2 => {
                if is_oom_trap_code(trap_code) {
                    self.saw_oom = true;
                }
                return Err(self.take_pending_error().unwrap_or_else(|| self.lib.trap_error(trap_code)));
            }
            _ => bail!("invalid_abi: missing blot:allocate"),
        };
        let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
        unsafe { (self.lib.syms.mem_data)(self.handle.0, &mut p, &mut n) };
        let memory = if n == 0 || p.is_null() {
            &mut [][..]
        } else {
            // SAFETY: `self.handle.0` uniquely owns guest linear memory while no guest call is active.
            unsafe { std::slice::from_raw_parts_mut(p, n) }
        };
        let address = array_range(memory.len(), pointer, words.len())?;
        memory[address..address + 4].copy_from_slice(&length.to_le_bytes());
        let body = &mut memory[address + 4..address + 4 + words.len() * 4];
        if cfg!(target_endian = "little") {
            body.copy_from_slice(bytemuck::cast_slice(words));
        } else {
            for (chunk, word) in body.as_chunks_mut::<4>().0.iter_mut().zip(words) {
                *chunk = word.to_le_bytes();
            }
        }
        Ok(pointer)
    }

    pub(crate) fn call_raw(&mut self, name: &str, param: Val, result_is_f32: bool) -> Result<Val> {
        self.clear_pending();
        let arg_bits = match param {
            Val::I32(v) => u64::from(v.cast_unsigned()),
            Val::F32(bits) => u64::from(bits),
            other => bail!("unsupported raw parameter value {other:?} for wtime-aot export {name}"),
        };
        let res_bits = self.call_export_raw(name, arg_bits)?;
        let low = (res_bits & 0xffff_ffff) as u32;
        Ok(if result_is_f32 { Val::F32(low) } else { Val::I32(low.cast_signed()) })
    }

    pub(crate) fn inspect_packet_header(&self, pointer: i32) -> Result<u32> {
        ensure!(self.has_arena(), "invalid_abi: array value requires an arena");
        let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
        unsafe { (self.lib.syms.mem_data)(self.handle.0, &mut p, &mut n) };
        let memory = if n == 0 || p.is_null() { &[][..] } else { unsafe { std::slice::from_raw_parts(p, n) } };
        let address = array_range(memory.len(), pointer, 0)?;
        let header = u32::from_le_bytes(memory[address..address + 4].try_into().unwrap());
        if header != u32::MAX {
            array_range(memory.len(), pointer, header as usize)?;
        }
        Ok(header)
    }

    pub(crate) fn read_packet_words_into(&self, pointer: i32, expected_words: usize, dst: &mut [u32]) -> Result<()> {
        ensure!(dst.len() == expected_words, "destination slice length mismatch");
        ensure!(self.has_arena(), "invalid_abi: array value requires an arena");
        let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
        unsafe { (self.lib.syms.mem_data)(self.handle.0, &mut p, &mut n) };
        let memory = if n == 0 || p.is_null() { &[][..] } else { unsafe { std::slice::from_raw_parts(p, n) } };
        let address = array_range(memory.len(), pointer, expected_words)?;
        let header = u32::from_le_bytes(memory[address..address + 4].try_into().unwrap()) as usize;
        ensure!(header == expected_words, "invalid_abi: packet length {header} != expected {expected_words}");
        let body = &memory[address + 4..address + 4 + expected_words * 4];
        if cfg!(target_endian = "little") {
            match bytemuck::try_cast_slice::<u8, u32>(body) {
                Ok(aligned) => dst.copy_from_slice(aligned),
                Err(_) => {
                    for (slot, chunk) in dst.iter_mut().zip(body.as_chunks::<4>().0) {
                        *slot = u32::from_le_bytes(*chunk);
                    }
                }
            }
        } else {
            for (slot, chunk) in dst.iter_mut().zip(body.as_chunks::<4>().0) {
                *slot = u32::from_le_bytes(*chunk);
            }
        }
        Ok(())
    }

    pub(crate) fn read_packet_words_vec(&self, pointer: i32, words: usize) -> Result<Vec<u32>> {
        let mut out = vec![0u32; words];
        self.read_packet_words_into(pointer, words, &mut out)?;
        Ok(out)
    }

    pub(crate) fn finish_invocation(&mut self) -> Result<Option<u32>> {
        if !self.abi.invocation_finish {
            return Ok(None);
        }
        self.last_reclaimed_owners = None;
        self.clear_pending();
        let mut released = 0i32;
        let mut trap_code = 0u8;
        let rc = unsafe { (self.lib.syms.call_finish_invocation)(self.handle.0, &mut released, &mut trap_code) };
        self.maybe_resume_panic();
        match rc {
            0 => {
                let released = released.cast_unsigned();
                self.last_reclaimed_owners = Some(released);
                Ok(Some(released))
            }
            1 => {
                if is_oom_trap_code(trap_code) {
                    self.saw_oom = true;
                }
                Err(self.lib.trap_error(trap_code))
            }
            2 => {
                if is_oom_trap_code(trap_code) {
                    self.saw_oom = true;
                }
                Err(self.take_pending_error().unwrap_or_else(|| self.lib.trap_error(trap_code)))
            }
            _ => bail!("invalid_abi: missing blot:finish_invocation"),
        }
    }

    pub fn last_reclaimed_owners(&self) -> Option<u32> {
        self.last_reclaimed_owners
    }

    pub fn track_memory(&mut self, gauge: Arc<AtomicUsize>) {
        gauge.store(self.memory_bytes(), Ordering::Relaxed);
        self.with_host_mut(|host| host.memory = Some(gauge));
    }

    #[inline]
    fn clear_pending(&mut self) {
        // SAFETY: `self.host.0` points to the live `AotHostBox` owned by `self`.
        let host_box = unsafe { &*self.host.0 };
        if host_box.has_pending_error.load(Ordering::Acquire) || host_box.has_pending_panic.load(Ordering::Acquire) {
            let mut host = host_box.state.lock().unwrap_or_else(|e| e.into_inner());
            host.pending_error = None;
            host.pending_panic = None;
            host_box.has_pending_error.store(false, Ordering::Release);
            host_box.has_pending_panic.store(false, Ordering::Release);
        }
    }

    #[inline]
    fn take_pending_error(&mut self) -> Option<anyhow::Error> {
        // SAFETY: `self.host.0` points to the live `AotHostBox` owned by `self`.
        let host_box = unsafe { &*self.host.0 };
        if !host_box.has_pending_error.load(Ordering::Acquire) {
            return None;
        }
        let mut host = host_box.state.lock().unwrap_or_else(|e| e.into_inner());
        host_box.has_pending_error.store(false, Ordering::Release);
        host.pending_error.take()
    }

    #[inline]
    fn maybe_resume_panic(&mut self) {
        // SAFETY: `self.host.0` points to the live `AotHostBox` owned by `self`.
        let host_box = unsafe { &*self.host.0 };
        if !host_box.has_pending_panic.load(Ordering::Acquire) {
            return;
        }
        let payload = {
            let mut host = host_box.state.lock().unwrap_or_else(|e| e.into_inner());
            host_box.has_pending_panic.store(false, Ordering::Release);
            host.pending_panic.take()
        };
        if let Some(payload) = payload {
            // Poison the guest handle before resuming unwind so partially unwound state cannot be reused.
            self.disposed = true;
            resume_unwind(payload);
        }
    }

    fn clear_trap_site(&mut self) {
        let mut p: *mut u8 = std::ptr::null_mut();
        let mut n: usize = 0;
        unsafe { (self.lib.syms.mem_data)(self.handle.0, &mut p, &mut n) };
        if !p.is_null() && n >= crate::TRAP_SITE_OFFSET + 4 {
            // SAFETY: `self.handle.0` uniquely owns guest linear memory while no guest call is active.
            let data = unsafe { std::slice::from_raw_parts_mut(p, n) };
            data[crate::TRAP_SITE_OFFSET..crate::TRAP_SITE_OFFSET + 4].copy_from_slice(&0u32.to_le_bytes());
        }
    }

    fn decorate_trap(&self, error: anyhow::Error) -> anyhow::Error {
        if self.abi.traps.is_empty() {
            return error;
        }
        let mut p: *mut u8 = std::ptr::null_mut();
        let mut n: usize = 0;
        unsafe { (self.lib.syms.mem_data)(self.handle.0, &mut p, &mut n) };
        if p.is_null() || n < crate::TRAP_SITE_OFFSET + 4 {
            return error;
        }
        // SAFETY: `self.handle.0` uniquely owns guest linear memory while no guest call is active.
        let data = unsafe { std::slice::from_raw_parts(p, n) };
        match crate::lookup_trap(&self.abi.traps, data) {
            Some(guest_trap) => error.context(guest_trap),
            None => error,
        }
    }

    fn call_export_raw(&mut self, name: &str, arg_bits: u64) -> Result<u64> {
        self.clear_trap_site();
        let mut res_out = 0u64;
        let mut trap_code = 0u8;
        let rc = unsafe {
            (self.lib.syms.call_export)(
                self.handle.0,
                name.as_ptr(),
                name.len(),
                arg_bits,
                &mut res_out,
                &mut trap_code,
            )
        };
        self.maybe_resume_panic();
        match rc {
            0 => Ok(res_out),
            2 => {
                if is_oom_trap_code(trap_code) {
                    self.saw_oom = true;
                }
                Err(self.take_pending_error().unwrap_or_else(|| self.decorate_trap(self.lib.trap_error(trap_code))))
            }
            1 => {
                if is_oom_trap_code(trap_code) {
                    self.saw_oom = true;
                }
                Err(self.decorate_trap(self.lib.trap_error(trap_code)))
            }
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
            1 => {
                if is_oom_trap_code(trap_code) {
                    self.saw_oom = true;
                }
                Err(self.lib.trap_error(trap_code))
            }
            2 => {
                if is_oom_trap_code(trap_code) {
                    self.saw_oom = true;
                }
                Err(self.take_pending_error().unwrap_or_else(|| self.lib.trap_error(trap_code)))
            }
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
        self.clear_pending();
        if self.abi.ownership {
            self.call_export_raw(OWNERSHIP_ENTER, 0)?;
            self.entered = true;
        }
        if self.abi.uses_arrays() {
            self.call_reset()?;
        }
        let arg_bits = match (function.parameter, argument) {
            (Parameter::Value(expected), Argument::Value(value)) => {
                ensure!(
                    value.kind() == expected,
                    "invalid_argument: {name} argument must be {expected}, not {}",
                    value.kind()
                );
                let lib = self.lib.clone();
                let handle = self.handle.0;
                let (uses_arrays, mut spare) =
                    self.with_host_mut(|host| (host.uses_arrays, std::mem::take(&mut host.spare)));
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
                let pending = {
                    let host_box = unsafe { &*self.host.0 };
                    let mut host = host_box.state.lock().unwrap_or_else(|e| e.into_inner());
                    host.spare = spare;
                    if alloc_host_err {
                        host_box.has_pending_error.store(false, Ordering::Release);
                        host.pending_error.take()
                    } else {
                        None
                    }
                };
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
                2 => Err(self.take_pending_error().unwrap_or_else(|| self.lib.trap_error(trap_code))),
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
        self.clear_pending();
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
                unsafe { (self.lib.syms.get_global)(self.handle.0, name.as_ptr(), name.len(), &mut val, &mut kind) }
                    == 0,
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
                unsafe {
                    (self.lib.syms.get_global)(self.handle.0, name.as_ptr(), name.len(), &mut cur_val, &mut cur_kind)
                } == 0,
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
            let rc =
                unsafe { (self.lib.syms.get_global)(self.handle.0, name.as_ptr(), name.len(), &mut val, &mut kind) };
            if rc == 0 && (kind == 0 || kind == 1) {
                out.push((name.clone(), (val & 0xffff_ffff) as u32));
            }
        }
        out
    }
}

#[cfg(all(test, feature = "compiler"))]
mod tests {
    use super::*;

    fn test_temp_dir(name: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "blot-host-wtime-aot-{name}-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create test temp dir");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).expect("chmod 0700");
        }
        dir
    }

    fn default_x64_cfg() -> AotCompileConfig {
        AotCompileConfig {
            backend: "x64".to_string(),
            c_opt: "auto".to_string(),
            toolchain: "stable".to_string(),
            target_cpu: "x86-64-v3".to_string(),
            explicit_backend: false,
        }
    }

    #[test]
    fn default_aot_backend_matches_host_arch() {
        #[cfg(all(target_os = "linux", target_arch = "aarch64", target_pointer_width = "64"))]
        assert_eq!(DEFAULT_AOT_BACKEND, "aarch64");
        #[cfg(not(all(target_os = "linux", target_arch = "aarch64", target_pointer_width = "64")))]
        assert_eq!(DEFAULT_AOT_BACKEND, "x64");
    }

    #[test]
    fn aarch64_backend_fallback_and_classification() {
        let mut cfg = AotCompileConfig {
            backend: "aarch64".to_string(),
            c_opt: "auto".to_string(),
            toolchain: "stable".to_string(),
            target_cpu: "native".to_string(),
            explicit_backend: false,
        };
        let mut seen = Vec::new();
        let res = load_verified_so_with_fallback(b"wasm", None, &cfg, |_w, _c, cur| {
            seen.push(cur.backend.clone());
            if cur.backend == "aarch64" {
                Err(classify_build_failure("aarch64", Some(X64_ADMISSION_REJECTED_EXIT_CODE), "exit status: 4", "rej"))
            } else {
                Ok(cur.backend.clone())
            }
        })
        .expect("implicit aarch64 admission rejection should fall back to c");
        assert_eq!(res, "c");
        assert_eq!(seen, vec!["aarch64", "c"]);

        cfg.explicit_backend = true;
        let err = load_verified_so_with_fallback(b"wasm", None, &cfg, |_w, _c, _cur| -> Result<()> {
            Err(classify_build_failure("aarch64", Some(X64_ADMISSION_REJECTED_EXIT_CODE), "exit status: 4", "rej"))
        })
        .expect_err("explicit aarch64 backend must propagate fallback error");
        assert!(err.downcast_ref::<X64FallbackError>().is_some());
        assert!(parse_aarch64_target_cpu_syntax("neoverse-v2").is_ok());
        assert!(parse_aarch64_target_cpu_syntax("bogus-cpu").is_err());
    }

    #[test]
    fn default_x64_fallback_triggers_on_all_three_fallback_reasons() {
        let reasons = [
            X64FallbackReason::UnsupportedPlatform("not linux x86_64".into()),
            X64FallbackReason::UnsupportedIsa("missing sse4.1".into()),
            X64FallbackReason::AdmissionRejected("unsupported proposal".into()),
        ];
        for reason in reasons {
            let cfg = default_x64_cfg();
            let mut seen = Vec::new();
            let result = load_verified_so_with_fallback(b"wasm", None, &cfg, |_w, _c, cur| {
                seen.push(cur.clone());
                if cur.backend == "x64" {
                    Err(anyhow::Error::new(X64FallbackError(reason.clone())))
                } else {
                    Ok(cur.backend.clone())
                }
            })
            .expect("default x64 fallback should recover with c backend");
            assert_eq!(result, "c");
            assert_eq!(seen.len(), 2);
            assert_eq!(seen[0].backend, "x64");
            assert_eq!(seen[1].backend, "c");
            assert_eq!(seen[1].c_opt, "auto");
            assert_eq!(seen[1].target_cpu, "native");
            assert!(seen[1].explicit_backend);
        }
    }

    #[test]
    fn explicit_x64_backend_propagates_fallback_errors_without_fallback() {
        let reasons = [
            X64FallbackReason::UnsupportedPlatform("not linux x86_64".into()),
            X64FallbackReason::UnsupportedIsa("cet shstk enabled".into()),
            X64FallbackReason::AdmissionRejected("memory64 unsupported".into()),
        ];
        for reason in reasons {
            let mut cfg = default_x64_cfg();
            cfg.explicit_backend = true;
            let mut calls = 0usize;
            let err = load_verified_so_with_fallback(b"wasm", None, &cfg, |_w, _c, _cur| -> Result<()> {
                calls += 1;
                Err(anyhow::Error::new(X64FallbackError(reason.clone())))
            })
            .expect_err("explicit x64 backend must propagate X64FallbackError");
            assert_eq!(calls, 1);
            let typed = err.downcast_ref::<X64FallbackError>().expect("typed X64FallbackError");
            assert_eq!(typed.0, reason);
        }
    }

    #[test]
    fn non_fallback_error_and_exit_code_1_propagate_even_on_default_x64() {
        let exit4 = classify_build_failure("x64", Some(X64_ADMISSION_REJECTED_EXIT_CODE), "exit status: 4", "bad op");
        assert!(matches!(
            exit4.downcast_ref::<X64FallbackError>(),
            Some(X64FallbackError(X64FallbackReason::AdmissionRejected(msg))) if msg == "bad op"
        ));

        let exit1 = classify_build_failure("x64", Some(1), "exit status: 1", "internal compiler error");
        assert!(exit1.downcast_ref::<X64FallbackError>().is_none());

        let c_exit4 = classify_build_failure("c", Some(X64_ADMISSION_REJECTED_EXIT_CODE), "exit status: 4", "c err");
        assert!(c_exit4.downcast_ref::<X64FallbackError>().is_none());

        let cfg = default_x64_cfg();
        let mut calls = 0usize;
        let err = load_verified_so_with_fallback(b"wasm", None, &cfg, |_w, _c, _cur| -> Result<()> {
            calls += 1;
            Err(classify_build_failure("x64", Some(1), "exit status: 1", "compiler io failure"))
        })
        .expect_err("exit code 1 must not trigger fallback");
        assert_eq!(calls, 1);
        assert!(err.downcast_ref::<X64FallbackError>().is_none());
        assert!(err.to_string().contains("compiler io failure"));
    }

    #[test]
    fn bin_digest_caches_invalidates_on_change_and_ignores_tampered_entries() {
        let root = test_temp_dir("bin-digest");
        let cache = ensure_private_cache_dir(&root.join("cache")).expect("private cache");
        let bin = root.join("fake-wtime-aot");
        std::fs::write(&bin, b"binary-version-1").expect("write v1");
        let (canon1, d1) = bin_digest(&bin, &cache).expect("digest v1");
        assert_eq!(d1, wasm_sha256_hex(b"binary-version-1"));

        let bin_cache_dir = cache.join("bin-digests");
        let path_hash = wasm_sha256_hex(canon1.as_os_str().as_encoded_bytes());
        let entry = bin_cache_dir.join(format!("{path_hash}.txt"));
        assert!(entry.is_file(), "bin-digest entry should be persisted");

        // Second lookup hits cache with exact digest.
        let (_, d1_again) = bin_digest(&bin, &cache).expect("digest v1 cached");
        assert_eq!(d1_again, d1);

        // Overwriting the binary with a different length/content must change BinStamp and recompute.
        std::fs::write(&bin, b"binary-version-2-longer").expect("write v2");
        let (_, d2) = bin_digest(&bin, &cache).expect("digest v2");
        assert_eq!(d2, wasm_sha256_hex(b"binary-version-2-longer"));
        assert_ne!(d2, d1);

        // Corrupting the on-disk entry (oversized or invalid mode) must fail closed and recompute cleanly.
        let bin3 = root.join("fake-wtime-aot-3");
        std::fs::write(&bin3, b"binary-version-3").expect("write v3");
        let canon3 = std::fs::canonicalize(&bin3).expect("canon v3");
        let entry3 = bin_cache_dir.join(format!("{}.txt", wasm_sha256_hex(canon3.as_os_str().as_encoded_bytes())));
        std::fs::write(&entry3, vec![b'x'; 16_384]).expect("write oversized entry");
        let (_, d3) = bin_digest(&bin3, &cache).expect("digest v3 ignores oversized entry");
        assert_eq!(d3, wasm_sha256_hex(b"binary-version-3"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let bin4 = root.join("fake-wtime-aot-4");
            std::fs::write(&bin4, b"binary-version-4").expect("write v4");
            let canon4 = std::fs::canonicalize(&bin4).expect("canon v4");
            let entry4 = bin_cache_dir.join(format!("{}.txt", wasm_sha256_hex(canon4.as_os_str().as_encoded_bytes())));
            let st4 = bin_stamp(&canon4).expect("stamp v4");
            let forged_sha = "0".repeat(64);
            std::fs::write(&entry4, format!("v1|{}|{st4:?}\n{forged_sha}\n", canon4.display())).expect("write forged");
            std::fs::set_permissions(&entry4, std::fs::Permissions::from_mode(0o666)).expect("chmod 0666");
            let (_, d4) = bin_digest(&bin4, &cache).expect("digest v4 ignores world-writable entry");
            assert_eq!(d4, wasm_sha256_hex(b"binary-version-4"));
            assert_ne!(d4, forged_sha);
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn validate_aot_wasm_rejects_huge_code_section_size_without_overflow_and_matches_imaged_globals() {
        use wasm_encoder::{
            CodeSection, ConstExpr, ExportKind, ExportSection, Function, FunctionSection, GlobalSection, GlobalType,
            Instruction, Module, TypeSection, ValType as EncValType,
        };
        let abi = GuestAbi::default();

        // Crafted Wasm with Code section (id=10) claiming size = u32::MAX (0xffff_ffff LEB128).
        let crafted = [
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic + version
            0x0a, // Code section id
            0xff, 0xff, 0xff, 0xff, 0x0f, // size = u32::MAX
            0x00, // truncated body
        ];
        assert!(validate_aot_wasm(&crafted, &abi).is_err());
        assert!(crate::imaged_globals(&crafted).is_err());

        // Valid module with 2 exported mutable i32 globals (one counter), 1 immutable global, and 1 function body:
        // verify validate_aot_wasm().imaged matches crate::imaged_globals() verbatim.
        let mut m = Module::new();
        let mut types = TypeSection::new();
        types.ty().function([], []);
        m.section(&types);
        let mut funcs = FunctionSection::new();
        funcs.function(0);
        m.section(&funcs);
        let mut globals = GlobalSection::new();
        globals
            .global(GlobalType { val_type: EncValType::I32, mutable: true, shared: false }, &ConstExpr::i32_const(7));
        globals
            .global(GlobalType { val_type: EncValType::I32, mutable: true, shared: false }, &ConstExpr::i32_const(0));
        globals
            .global(GlobalType { val_type: EncValType::I32, mutable: false, shared: false }, &ConstExpr::i32_const(42));
        m.section(&globals);
        let mut exports = ExportSection::new();
        exports.export("g_mut", ExportKind::Global, 0);
        exports.export("blot:ctr", ExportKind::Global, 1);
        m.section(&exports);
        let mut code = CodeSection::new();
        let mut body = Function::new([]);
        body.instruction(&Instruction::Nop);
        body.instruction(&Instruction::End);
        code.function(&body);
        m.section(&code);
        let valid = m.finish();

        let validated = validate_aot_wasm(&valid, &abi).expect("validate valid module");
        assert_eq!(validated.imaged, crate::imaged_globals(&valid));
        assert_eq!(validated.imaged, Ok(vec!["g_mut".to_string(), "blot:ctr".to_string()]));
        assert_eq!(validated.counter_names, vec!["blot:ctr".to_string()]);

        // Hidden mutable global produces the exact same error string in both paths.
        let mut hm = Module::new();
        let mut hglobals = GlobalSection::new();
        hglobals
            .global(GlobalType { val_type: EncValType::I32, mutable: true, shared: false }, &ConstExpr::i32_const(1));
        hm.section(&hglobals);
        let hidden = hm.finish();
        let v_hidden = validate_aot_wasm(&hidden, &abi).expect("validate hidden");
        assert_eq!(v_hidden.imaged, crate::imaged_globals(&hidden));
        assert!(v_hidden.imaged.is_err());
    }

    #[cfg(feature = "compiler")]
    #[test]
    fn verify_artifact_sidecar_digest_accepts_plain_sha256_and_merkle256k_over_256kib() {
        let small = b"small-so-artifact";
        let small_sha = wasm_sha256_hex(small);
        assert!(verify_artifact_sidecar_digest(small, &small_sha, &small_sha));
        assert!(!verify_artifact_sidecar_digest(small, &small_sha, &"0".repeat(64)));
        assert!(!verify_artifact_sidecar_digest(small, &small_sha, "short"));

        // > 256 KiB payload (600 KiB = 3 chunks: 256 KiB + 256 KiB + 88 KiB) matching wtime-aot's merkle256k format.
        let mut large: Vec<u8> = (0u32..(600 << 10))
            .map(|i| u8::try_from((i.wrapping_mul(31) ^ (i >> 7)) & 0xff).expect("masked byte"))
            .collect();
        let large_sha = wasm_sha256_hex(&large);
        let merkle_line = format!("merkle256k:{}", merkle256k_sha256_hex(&large));
        assert!(verify_artifact_sidecar_digest(&large, &large_sha, &merkle_line));
        assert!(verify_artifact_sidecar_digest(&large, &large_sha, &large_sha));
        assert!(!verify_artifact_sidecar_digest(&large, &large_sha, &format!("merkle256k:{}", "0".repeat(64))));

        // Flip a byte in the final partial chunk and confirm Merkle verification fails closed.
        let last = large.len() - 1;
        large[last] ^= 0x01;
        let mutated_sha = wasm_sha256_hex(&large);
        assert!(!verify_artifact_sidecar_digest(&large, &mutated_sha, &merkle_line));
    }

    #[cfg(feature = "compiler")]
    #[test]
    fn parallel_merkle256k_matches_serial_and_verified_so_digest_roundtrips() {
        use sha2::{Digest, Sha256};
        // 1.1 MiB payload across 5 chunks (4 full 256 KiB + 1 partial) exercises multi-thread path.
        let payload: Vec<u8> =
            (0u32..(1120 << 10)).map(|i| u8::try_from((i.wrapping_mul(17) ^ (i >> 9)) & 0xff).expect("byte")).collect();
        let mut serial_root = Sha256::new();
        serial_root.update(WTIME_AOT_MERKLE256K_DOMAIN);
        serial_root.update((payload.len() as u64).to_le_bytes());
        for chunk in payload.chunks(WTIME_AOT_MERKLE256K_CHUNK_BYTES) {
            serial_root.update(Sha256::digest(chunk));
        }
        let expected_hex = format!("{:x}", serial_root.finalize());
        assert_eq!(merkle256k_sha256_hex(&payload), expected_hex);

        let line = format!("{WTIME_AOT_MERKLE256K_PREFIX}{expected_hex}");
        let parsed = VerifiedSoDigest::parse(&line).expect("parse merkle256k");
        assert_eq!(parsed.line_text(), line);
        assert_eq!(parsed.filename(), format!("m-{expected_hex}.so"));
        assert!(parsed.matches_bytes(&payload));

        let plain_hex = wasm_sha256_hex(b"tiny");
        let parsed_plain = VerifiedSoDigest::parse(&plain_hex).expect("parse plain sha256");
        assert_eq!(parsed_plain.line_text(), plain_hex);
        assert_eq!(parsed_plain.filename(), format!("{plain_hex}.so"));
        assert!(parsed_plain.matches_bytes(b"tiny"));
        assert!(VerifiedSoDigest::parse("merkle256k:short").is_none());
    }
}
