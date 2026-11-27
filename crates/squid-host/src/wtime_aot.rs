//! `wtime-aot` backend for Squid ABI 8 guest modules.
//!
//! Used by default for `pen run` builds when `wtime-aot` is available (disable
//! with `PEN_WTIME_AOT=0`), or for non-`dev_engine` engines when `PEN_WTIME_AOT=1`.
//! Compiles the guest Wasm module once via `wtime-aot build --backend x64 --blot-cdylib`
//! on Linux x86-64 (automatically falling back to `--backend c --c-opt auto --target-cpu native`
//! on unsupported platforms, ISAs/SHSTK, or admission rejections; override backend with
//! `PEN_WTIME_AOT_BACKEND=x64|c|rust`, C optimization mode with
//! `PEN_WTIME_AOT_C_OPT=auto|O2|O1|O0`, and x64 target CPU with
//! `PEN_WTIME_AOT_TARGET_CPU=native|x86-64-v2|x86-64-v3`) into a cached shared library,
//! then loads it with `libloading` and drives the exact same Squid ABI 8 lifecycle,
//! arena array copies, host callbacks, and snapshots as Wasmtime.

#[cfg(feature = "compiler")]
mod cache;
#[cfg(all(test, feature = "compiler"))]
mod tests;
#[cfg(feature = "compiler")]
mod toolchain;

#[cfg(all(feature = "compiler", unix))]
pub(crate) use cache::current_euid;
#[cfg(feature = "compiler")]
pub(crate) use cache::ensure_private_cache_dir;

use std::any::Any;
use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
#[cfg(feature = "compiler")]
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

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
    /// Whether the module exports memory 0 as `"squid:memory"` (the memory `WasmtimeGuest` images).
    has_squid_memory_export: bool,
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

/// Typed runtime trap returned by a `wtime-aot` guest instance, preserving the Squid ABI 8 trap code.
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
fn wasmparser_val(value: ValueType) -> wasmparser::ValType {
    match value {
        ValueType::F32 => wasmparser::ValType::F32,
        _ => wasmparser::ValType::I32,
    }
}

#[cfg(feature = "compiler")]
struct ValidatedAotWasm {
    memory_limits: Option<(u64, Option<u64>)>,
    has_squid_memory_export: bool,
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
                        if export.name.starts_with("squid:") && abi.constants.iter().all(|c| c.name != export.name) {
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
    let has_squid_memory_export =
        exports.get("squid:memory").is_some_and(|&(k, idx)| k == wasmparser::ExternalKind::Memory && idx == 0);
    if abi.uses_arrays() {
        ensure!(has_squid_memory_export, "invalid_abi: missing array arena export squid:memory");
        for name in ["squid:allocate", "squid:reset"] {
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
        has_squid_memory_export,
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
        let lib = cache::load_verified_so(wasm, cache)?;
        let abi_version: FnAbiVersion = *unsafe { lib.get(b"blot_aot_abi_version\0")? };
        let version = unsafe { abi_version() };
        ensure!(version == 1, "unsupported wtime-aot squid ABI version {version}");

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
            ensure!(
                module == HOST_MODULE || module == "blot:host/1",
                "invalid_abi: unexpected import {module}::{name}"
            );
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
            let _active_reset: FnCallI32I32 = *lib.get(b"blot_aot_active_reset\0")?;
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
                    has_squid_memory_export: validated.has_squid_memory_export,
                    has_mut_extern_table: validated.has_mut_extern_table,
                }),
            },
            validated.imaged,
        ))
    }
}

struct SendPtr(*mut c_void);
// SAFETY: `SquidHandle` is owned exclusively by one `AotGuest` and never aliased.
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
/// nested-callback semantics (a guest `squid:allocate` may invoke the restored active callback
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
    // `squid:allocate` during result encoding may legally invoke the active callback. Only clone
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
                2 => Err(anyhow!("host error during squid:allocate")),
                _ => bail!("invalid_abi: missing squid:allocate"),
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
                        // Squid ABI v1 reserves `mem_limit_pages == 0` as unlimited; reject growable memories
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
            self.lib.counter_names.iter().map(|name| name.trim_start_matches("squid:").to_owned()).collect(),
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
            _ => bail!("invalid_abi: missing squid:allocate"),
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
            _ => bail!("invalid_abi: missing squid:finish_invocation"),
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
            _ => bail!("invalid_abi: missing squid:reset"),
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
                            _ => bail!("invalid_abi: missing squid:allocate"),
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
                _ => bail!("invalid_abi: missing squid:finish_invocation"),
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
        let bytes = if self.lib.has_squid_memory_export {
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
        if self.lib.has_squid_memory_export {
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
