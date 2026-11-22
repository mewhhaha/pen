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
use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

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

impl AotLibrary {
    fn trap_error(&self, code: u8) -> anyhow::Error {
        let mut ptr: *const u8 = std::ptr::null();
        let mut len: usize = 0;
        let ok = unsafe { (self.syms.trap_message)(code, &mut ptr, &mut len) };
        if ok == 0 && !ptr.is_null() {
            let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
            let msg = String::from_utf8_lossy(bytes);
            anyhow!("wtime-aot trap: {msg}")
        } else {
            anyhow!("wtime-aot trap (code {code})")
        }
    }
}

#[derive(Clone)]
pub struct AotModule {
    lib: Arc<AotLibrary>,
}

fn default_cache_dir(cache: Option<&Path>) -> PathBuf {
    if let Ok(dir) = std::env::var("PEN_WTIME_AOT_CACHE")
        && !dir.is_empty()
    {
        return PathBuf::from(dir);
    }
    if let Some(cache) = cache {
        return cache.join("wtime-aot");
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME")
        && !xdg.is_empty()
    {
        return PathBuf::from(xdg).join("pen").join("wtime-aot");
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.is_empty()
    {
        return PathBuf::from(home).join(".cache").join("pen").join("wtime-aot");
    }
    std::env::temp_dir().join("pen-wtime-aot")
}

fn wasm_sha256_hex(wasm: &[u8]) -> String {
    #[cfg(feature = "compiler")]
    {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(wasm))
    }
    #[cfg(not(feature = "compiler"))]
    {
        let mut h = 0xcbf29ce484222325u64;
        for &b in wasm {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
        format!("{h:016x}")
    }
}

fn file_sha256_hex(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(wasm_sha256_hex(&bytes))
}

fn bin_digest(bin: &Path) -> Result<String> {
    type StampKey = (PathBuf, u64, u64, u32);
    static CACHE: OnceLock<Mutex<std::collections::HashMap<StampKey, String>>> = OnceLock::new();
    let meta = std::fs::metadata(bin).with_context(|| format!("stat wtime-aot binary {}", bin.display()))?;
    let (secs, nanos) = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or((0, 0), |d| (d.as_secs(), d.subsec_nanos()));
    let key = (bin.to_path_buf(), meta.len(), secs, nanos);
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if let Some(hit) = cache.lock().unwrap_or_else(|p| p.into_inner()).get(&key).cloned() {
        return Ok(hit);
    }
    let digest = file_sha256_hex(bin)?;
    cache.lock().unwrap_or_else(|p| p.into_inner()).insert(key, digest.clone());
    Ok(digest)
}

fn sanitize_key_part(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' })
        .collect()
}

struct RemoveOnDrop(PathBuf);
impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Canonical internal cache profile tag matching `wtime-aot --backend x64 --target-cpu`.
///
/// NOTE: `x64-v2` / `x64-v3` are internal compiler feature profiles (SSE4.1+POPCNT vs
/// SSE4.1+POPCNT+ABM/LZCNT+BMI1/TZCNT), not full psABI levels.
fn resolved_x64_isa_tag(target_cpu: &str) -> Result<&'static str> {
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let _ = target_cpu;
        bail!("wtime-aot --backend x64 is only supported on Linux x86_64 hosts")
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    match target_cpu {
        "native" => {
            if std::is_x86_feature_detected!("lzcnt") && std::is_x86_feature_detected!("bmi1") {
                Ok("x64-v3")
            } else {
                Ok("x64-v2")
            }
        }
        "x86-64-v2" | "generic" => Ok("x64-v2"),
        "x86-64-v3" | "x86-64-v4" | "haswell" | "broadwell" | "skylake" | "skylake-avx512" | "cascadelake"
        | "cooperlake" | "cannonlake" | "icelake-client" | "icelake-server" | "tigerlake" | "rocketlake"
        | "alderlake" | "raptorlake" | "sapphirerapids" | "emeraldrapids" | "graniterapids" => Ok("x64-v3"),
        other if other.strip_prefix("znver").is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())) => {
            Ok("x64-v3")
        }
        other => bail!(
            "unsupported PEN_WTIME_AOT_TARGET_CPU={other:?} (expected native, x86-64-v2, generic, x86-64-v3, x86-64-v4, znver<N>, or haswell+)"
        ),
    }
}

fn build_or_find_so(wasm: &[u8], cache: Option<&Path>) -> Result<PathBuf> {
    static NEXT_SEQ: AtomicU64 = AtomicU64::new(0);
    if let Ok(override_so) = std::env::var("PEN_WTIME_AOT_SO")
        && !override_so.is_empty()
    {
        let p = PathBuf::from(override_so);
        ensure!(p.is_file(), "PEN_WTIME_AOT_SO does not exist: {}", p.display());
        // Canonicalize so a bare filename can never trigger `dlopen`'s `LD_LIBRARY_PATH` search.
        return std::fs::canonicalize(&p)
            .with_context(|| format!("canonicalizing PEN_WTIME_AOT_SO {}", p.display()));
    }
    let bin = resolve_wtime_aot_bin().unwrap_or_else(|| PathBuf::from(DEFAULT_WTIME_AOT_BIN));
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

    let cache_dir = default_cache_dir(cache);
    let digest = wasm_sha256_hex(wasm);
    let bin_hash = bin_digest(&bin)?;
    let fast_key = match backend.as_str() {
        "c" => format!("{digest}-c-{}-{bin_hash}", sanitize_key_part(&c_opt)),
        "rust" => format!("{digest}-rust-{}-{bin_hash}", sanitize_key_part(&toolchain)),
        "x64" => format!("{digest}-x64-{}-{bin_hash}", resolved_x64_isa_tag(&target_cpu)?),
        other => bail!("unsupported PEN_WTIME_AOT_BACKEND={other:?} (expected c, rust, or x64)"),
    };
    let fast_dir = cache_dir.join("by-wasm");
    let fast_link = fast_dir.join(format!("{fast_key}.path"));
    if let Ok(recorded) = std::fs::read_to_string(&fast_link) {
        let mut lines = recorded.lines();
        if let (Some(rec_key), Some(rec_so_sha), Some(rec_path)) = (lines.next(), lines.next(), lines.next())
            && rec_key == fast_key
        {
            let so_path = PathBuf::from(rec_path.trim());
            if let (Ok(canon_cache), Ok(canon_so)) = (std::fs::canonicalize(&cache_dir), std::fs::canonicalize(&so_path))
                && canon_so.starts_with(&canon_cache)
                && canon_so.is_file()
                && file_sha256_hex(&canon_so).is_ok_and(|actual| actual == rec_so_sha)
            {
                return Ok(canon_so);
            }
        }
    }

    std::fs::create_dir_all(&cache_dir)?;
    let seq = NEXT_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp_wasm = cache_dir.join(format!("input-{digest}-{}-{seq}.wasm", std::process::id()));
    std::fs::write(&tmp_wasm, wasm)?;
    let _cleanup_wasm = RemoveOnDrop(tmp_wasm.clone());
    let mut cmd = Command::new(&bin);
    cmd.args(["build"])
        .arg(&tmp_wasm)
        .args(["--cache"])
        .arg(&cache_dir)
        .args(["--backend", &backend, "--blot-cdylib"]);
    match backend.as_str() {
        "c" => {
            cmd.args(["--c-opt", &c_opt]);
        }
        "rust" => {
            cmd.args(["--toolchain", &toolchain]);
        }
        "x64" => {
            cmd.args(["--target-cpu", &target_cpu]);
        }
        _ => unreachable!(),
    }
    let output = cmd.output().with_context(|| format!("running wtime-aot ({})", bin.display()))?;
    drop(_cleanup_wasm);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("wtime-aot build failed ({}): {stderr}", output.status);
    }
    let so_path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    ensure!(so_path.is_file(), "wtime-aot build did not produce {}", so_path.display());
    let canon_so = std::fs::canonicalize(&so_path)
        .with_context(|| format!("canonicalizing built library {}", so_path.display()))?;
    let so_sha = file_sha256_hex(&canon_so)?;
    let _ = std::fs::create_dir_all(&fast_dir);
    let tmp_link = fast_dir.join(format!(".{fast_key}.{}.{seq}.tmp", std::process::id()));
    if let Ok(mut file) = std::fs::File::options().write(true).create_new(true).open(&tmp_link) {
        let written = file.write_all(format!("{fast_key}\n{so_sha}\n{}\n", canon_so.display()).as_bytes());
        drop(file);
        if written.is_err() || std::fs::rename(&tmp_link, &fast_link).is_err() {
            let _ = std::fs::remove_file(&tmp_link);
        }
    }
    Ok(canon_so)
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
                    if memory_limits.is_none() {
                        memory_limits = Some((mem.initial, mem.maximum));
                    }
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

fn read_aot_str<'a>(ptr: *const u8, len: usize) -> Result<&'a str> {
    if len == 0 {
        return Ok("");
    }
    ensure!(!ptr.is_null(), "invalid_abi: null string pointer with non-zero length {len}");
    // SAFETY: caller guarantees `(ptr, len)` is a valid static byte range from the loaded `.so`.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    std::str::from_utf8(bytes).context("invalid_abi: non-UTF-8 string returned by AOT library")
}

impl AotModule {
    pub fn compile(wasm: &[u8], abi: &GuestAbi, cache: Option<&Path>) -> Result<Self> {
        ensure!(abi.tasks.is_none(), "wtime-aot experimental backend does not support legacy tasks");
        abi.check_hostable()?;
        let validated = validate_aot_wasm(wasm, abi)?;
        let so_path = build_or_find_so(wasm, cache)?;
        // SAFETY: `so_path` was built locally by `wtime-aot` from validated Wasm and digest-verified.
        let lib = unsafe { Library::new(&so_path) }
            .with_context(|| format!("loading wtime-aot library {}", so_path.display()))?;
        let abi_version: FnAbiVersion = *unsafe { lib.get(b"blot_aot_abi_version\0")? };
        let version = unsafe { abi_version() };
        ensure!(version == 1, "unsupported wtime-aot blot ABI version {version}");

        let import_count_fn: FnImportCount = *unsafe { lib.get(b"blot_aot_import_count\0")? };
        let import_name_fn: FnImportName = *unsafe { lib.get(b"blot_aot_import_name\0")? };
        let callbacks = abi.callbacks();
        let n_imports = unsafe { import_count_fn() };
        let mut imports = Vec::with_capacity(n_imports as usize);
        for idx in 0..n_imports {
            let (mut mp, mut ml, mut np, mut nl) = (std::ptr::null(), 0, std::ptr::null(), 0);
            ensure!(unsafe { import_name_fn(idx, &mut mp, &mut ml, &mut np, &mut nl) } == 0, "bad import index {idx}");
            let module = read_aot_str(mp, ml)?;
            let name = read_aot_str(np, nl)?;
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
/// Stored behind a raw pointer in `AotGuest` (never `Box<AotHostState>`) and gated by
/// `in_trampoline` so `&mut AotHostState` is NEVER live across any guest FFI call or
/// reentrant callback (`active_allocate`), preventing Stacked/Tree Borrows aliasing UB.
struct AotHostBox {
    in_trampoline: AtomicBool,
    state: UnsafeCell<AotHostState>,
}

struct SendHostPtr(*mut AotHostBox);
// SAFETY: `AotHostBox` is owned exclusively by one `AotGuest`, `AotHostState: Send`, and
// accesses are strictly serialized on the calling thread.
unsafe impl Send for SendHostPtr {}

struct TrampolineGuard<'a>(&'a AtomicBool);
impl Drop for TrampolineGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

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
    if host_box.in_trampoline.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
        // Reject nested host callbacks (e.g. from inside `active_allocate`) BEFORE forming `&mut AotHostState`.
        return 1;
    }
    let _guard = TrampolineGuard(&host_box.in_trampoline);
    match catch_unwind(AssertUnwindSafe(|| invoke_aot_import(host_box, active_call, import_idx, token, arg_bits))) {
        Ok(Ok(bits)) => {
            if !res_out.is_null() {
                unsafe { *res_out = bits };
            }
            0
        }
        Ok(Err(err)) => {
            // SAFETY: `in_trampoline` is held exclusively and no `&mut AotHostState` is currently live.
            unsafe { (*host_box.state.get()).pending_error = Some(err) };
            1
        }
        Err(payload) => {
            // Never unwind across `extern "C"` / generated native frames; stash payload to resume on host side.
            unsafe { (*host_box.state.get()).pending_panic = Some(payload) };
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

fn invoke_aot_import(
    host_box: &AotHostBox,
    active_call: *mut c_void,
    import_idx: u32,
    token: u64,
    arg_bits: u64,
) -> Result<u64> {
    // Short borrow #1: extract metadata and move `spare` out so NO `&mut AotHostState` is live
    // across any FFI call (`active_mem_data`, `active_get_global`, `callback`, or `active_allocate`).
    let (signature, export, lib, uses_arrays, memory_gauge, probe, mut spare) = {
        // SAFETY: `in_trampoline` is exclusively acquired by `host_callback_trampoline`.
        let state = unsafe { &mut *host_box.state.get() };
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
            unsafe { (*host_box.state.get()).spare = spare };
            return Err(err);
        }
    };
    if let Some(gauge) = &memory_gauge {
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
        probe.called_host(mem_len, vals.into_iter());
    }

    // Short borrow #2: take callback out before invoking it.
    let mut callback = {
        // SAFETY: `in_trampoline` is exclusively held and no other reference to `state` exists.
        let state = unsafe { &mut *host_box.state.get() };
        match state.active.as_mut().and_then(|active| active.callback.take()) {
            Some(cb) => cb,
            None => {
                state.spare = spare;
                bail!("reentrant_call: cannot reenter an active guest instance");
            }
        }
    };
    let result = callback(argument);
    // Short borrow #3: restore callback immediately after invocation.
    {
        let state = unsafe { &mut *host_box.state.get() };
        if let Some(active) = state.active.as_mut() {
            active.callback = Some(callback);
        }
    }
    let result = match result.with_context(|| format!("host_exception: host callback failed during {export}")) {
        Ok(val) => val,
        Err(err) => {
            unsafe { (*host_box.state.get()).spare = spare };
            return Err(err);
        }
    };
    if result.kind() != signature.result {
        unsafe { (*host_box.state.get()).spare = spare };
        bail!(
            "host_result: host callback during {export} returned {}, not {}",
            result.kind(),
            signature.result
        );
    }

    // Zero `&mut AotHostState` references are live during `active_allocate` below.
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
    // Short borrow #4: restore `spare` buffer.
    unsafe { (*host_box.state.get()).spare = spare };
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
        // short borrow is held.
        let host_box = unsafe { &*self.host.0 };
        debug_assert!(!host_box.in_trampoline.load(Ordering::Acquire));
        let state = unsafe { &mut *host_box.state.get() };
        f(state)
    }

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
            in_trampoline: AtomicBool::new(false),
            state: UnsafeCell::new(AotHostState {
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
            let host = unsafe { Box::from_raw(host_ptr) }.state.into_inner();
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
                // (captured at 0 pages before a later grow), `restore` zeros committed pages beyond it.
                let bytes = unsafe { std::slice::from_raw_parts_mut(p, current) };
                written = image.memory.restore(bytes);
            }
        }
        for (name, value) in &image.globals {
            let raw = match *value {
                Val::I32(v) => u64::from(v.cast_unsigned()),
                Val::I64(v) => v.cast_unsigned(),
                Val::F32(bits) => u64::from(bits),
                Val::F64(bits) => bits,
                _ => bail!("unsupported imaged global value"),
            };
            ensure!(
                unsafe { (self.lib.syms.set_global)(self.handle.0, name.as_ptr(), name.len(), raw) } == 0,
                "an imaged global disappeared: {name}"
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
