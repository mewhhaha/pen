//! `wtime-aot` backend for Blot ABI 8 guest modules.
//!
//! Used by default for `pen run` builds when `wtime-aot` is available (disable
//! with `PEN_WTIME_AOT=0`), or everywhere when `PEN_WTIME_AOT=1`. Compiles the
//! guest Wasm module once via `wtime-aot build --backend c --c-opt auto --blot-cdylib`
//! (override backend with `PEN_WTIME_AOT_BACKEND=rust|c` and optimization mode
//! with `PEN_WTIME_AOT_C_OPT=auto|O2|O1|O0`) into a cached shared library, then
//! loads it with `libloading` and drives the exact same Blot ABI 8 lifecycle,
//! arena array copies, host callbacks, and snapshots as Wasmtime.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, anyhow, bail, ensure};
use libloading::Library;
use pen_runtime::abi::{CallbackType, GuestAbi, GuestFunction, Parameter, ValueType};
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

fn bin_stamp(bin: &Path) -> String {
    let Ok(meta) = std::fs::metadata(bin) else {
        return "unknown".to_string();
    };
    let secs = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    format!("{}-{secs}", meta.len())
}

fn build_or_find_so(wasm: &[u8], cache: Option<&Path>) -> Result<PathBuf> {
    if let Ok(override_so) = std::env::var("PEN_WTIME_AOT_SO")
        && !override_so.is_empty()
    {
        let p = PathBuf::from(override_so);
        ensure!(p.is_file(), "PEN_WTIME_AOT_SO does not exist: {}", p.display());
        return Ok(p);
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

    let cache_dir = default_cache_dir(cache);
    let digest = wasm_sha256_hex(wasm);
    let stamp = bin_stamp(&bin);
    let fast_key = if backend == "c" {
        format!("{digest}-c-{c_opt}-{stamp}")
    } else {
        format!("{digest}-{backend}-{toolchain}-{stamp}")
    };
    let fast_dir = cache_dir.join("by-wasm");
    let fast_link = fast_dir.join(format!("{fast_key}.path"));
    if let Ok(recorded) = std::fs::read_to_string(&fast_link) {
        let so_path = PathBuf::from(recorded.trim());
        if so_path.is_file() {
            return Ok(so_path);
        }
    }

    std::fs::create_dir_all(&cache_dir)?;
    let tmp_wasm = cache_dir.join(format!("input-{}-{}.wasm", digest, std::process::id()));
    std::fs::write(&tmp_wasm, wasm)?;
    let mut cmd = Command::new(&bin);
    cmd.args(["build"])
        .arg(&tmp_wasm)
        .args(["--cache"])
        .arg(&cache_dir)
        .args(["--backend", &backend, "--blot-cdylib"]);
    if backend == "c" {
        cmd.args(["--c-opt", &c_opt]);
    } else {
        cmd.args(["--toolchain", &toolchain]);
    }
    let output = cmd.output().with_context(|| format!("running wtime-aot ({})", bin.display()))?;
    let _ = std::fs::remove_file(&tmp_wasm);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("wtime-aot build failed ({}): {stderr}", output.status);
    }
    let so_path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    ensure!(so_path.is_file(), "wtime-aot build did not produce {}", so_path.display());
    let _ = std::fs::create_dir_all(&fast_dir);
    let _ = std::fs::write(&fast_link, format!("{}\n", so_path.display()));
    Ok(so_path)
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

impl AotModule {
    pub fn compile(wasm: &[u8], abi: &GuestAbi, cache: Option<&Path>) -> Result<Self> {
        ensure!(abi.tasks.is_none(), "wtime-aot experimental backend does not support legacy tasks");
        abi.check_hostable()?;
        let so_path = build_or_find_so(wasm, cache)?;
        // SAFETY: `so_path` was built locally by `wtime-aot` from validated Wasm.
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
            let module = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(mp, ml)) };
            let name = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(np, nl)) };
            ensure!(module == HOST_MODULE, "invalid_abi: unexpected import {module}::{name}");
            let sig = callbacks
                .iter()
                .copied()
                .find(|cb| cb.import_name() == name)
                .ok_or_else(|| anyhow!("invalid_abi: unexpected import {module}::{name}"))?;
            imports.push(sig);
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
        Ok(Self { lib: Arc::new(AotLibrary { _lib: lib, syms, imports, counter_names }) })
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
    uses_arrays: bool,
}

unsafe extern "C" fn host_callback_trampoline(
    user_ctx: *mut c_void,
    active_call: *mut c_void,
    import_idx: u32,
    token: u64,
    arg_bits: u64,
    res_out: *mut u64,
) -> i32 {
    let state = unsafe { &mut *(user_ctx as *mut AotHostState) };
    match invoke_aot_import(state, active_call, import_idx, token, arg_bits) {
        Ok(bits) => {
            if !res_out.is_null() {
                unsafe { *res_out = bits };
            }
            0
        }
        Err(err) => {
            state.pending_error = Some(err);
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
    mem_mut: impl FnOnce() -> &'static mut [u8],
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
    let memory = mem_mut();
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
    *spare = words;
    Ok(u64::from(pointer.cast_unsigned()))
}

fn invoke_aot_import(
    state: &mut AotHostState,
    active_call: *mut c_void,
    import_idx: u32,
    token: u64,
    arg_bits: u64,
) -> Result<u64> {
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
    let lib = state.lib.clone();
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
    let argument = aot_from_bits(mem_slice, state.uses_arrays, &mut state.spare, signature.parameter, arg_bits)?;
    if let Some(gauge) = &state.memory {
        gauge.store(mem_len, Ordering::Relaxed);
    }
    if let Some(probe) = state.probe.clone() {
        let mut vals = Vec::with_capacity(lib.counter_names.len());
        for name in &lib.counter_names {
            let (mut val, mut kind) = (0u64, 0u8);
            let rc = unsafe { (lib.syms.active_get_global)(active_call, name.as_ptr(), name.len(), &mut val, &mut kind) };
            let c = if rc == 0 && (kind == 0 || kind == 1) { (val & 0xffff_ffff) as u32 } else { 0 };
            vals.push(u64::from(c));
        }
        probe.called_host(mem_len, vals.into_iter());
    }
    let mut callback = state
        .active
        .as_mut()
        .and_then(|active| active.callback.take())
        .ok_or_else(|| anyhow!("reentrant_call: cannot reenter an active guest instance"))?;
    let result = callback(argument);
    if let Some(active) = state.active.as_mut() {
        active.callback = Some(callback);
    }
    let result = result.with_context(|| format!("host_exception: host callback failed during {export}"))?;
    ensure!(
        result.kind() == signature.result,
        "host_result: host callback during {export} returned {}, not {}",
        result.kind(),
        signature.result
    );
    aot_to_bits(
        state.uses_arrays,
        &mut state.spare,
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
        || {
            let mut p: *mut u8 = std::ptr::null_mut();
            let mut n: usize = 0;
            unsafe { (lib.syms.active_mem_data)(active_call, &mut p, &mut n) };
            if n == 0 || p.is_null() {
                &mut []
            } else {
                // SAFETY: `active_call` uniquely borrows guest linear memory for this write.
                unsafe { std::slice::from_raw_parts_mut(p, n) }
            }
        },
    )
}

pub struct AotGuest {
    lib: Arc<AotLibrary>,
    handle: SendPtr,
    host: Box<AotHostState>,
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
    }
}

impl AotGuest {
    pub fn start(
        module: &AotModule,
        abi: Arc<GuestAbi>,
        imaged: Arc<Result<Vec<String>, String>>,
        limits: Option<Limits>,
    ) -> Result<Self> {
        let mut host = Box::new(AotHostState {
            lib: module.lib.clone(),
            active: None,
            memory: None,
            probe: None,
            spare: Vec::new(),
            pending_error: None,
            uses_arrays: abi.uses_arrays(),
        });
        let limit_pages = match limits {
            Some(lim) => u64::try_from(lim.memory / PAGE).unwrap_or(u64::MAX).max(1),
            None => 0,
        };
        let user_ctx = (&mut *host as *mut AotHostState).cast::<c_void>();
        let mut trap_code = 0u8;
        let handle = unsafe { (module.lib.syms.new)(user_ctx, Some(host_callback_trampoline), limit_pages, &mut trap_code) };
        if handle.is_null() {
            if let Some(err) = host.pending_error.take() {
                return Err(err);
            }
            return Err(module.lib.trap_error(trap_code));
        }
        Ok(Self {
            lib: module.lib.clone(),
            handle: SendPtr(handle),
            host,
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
        self.host.probe = Some(probe);
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
        self.host.memory = Some(gauge);
    }

    fn call_export_raw(&mut self, name: &str, arg_bits: u64) -> Result<u64> {
        let mut res_out = 0u64;
        let mut trap_code = 0u8;
        let rc = unsafe {
            (self.lib.syms.call_export)(self.handle.0, name.as_ptr(), name.len(), arg_bits, &mut res_out, &mut trap_code)
        };
        match rc {
            0 => Ok(res_out),
            2 => Err(self.host.pending_error.take().unwrap_or_else(|| self.lib.trap_error(trap_code))),
            1 => Err(self.lib.trap_error(trap_code)),
            _ => bail!("unknown_export: unknown function export {name}"),
        }
    }

    fn call_reset(&mut self) -> Result<()> {
        let mut out = 0i32;
        let mut trap_code = 0u8;
        let rc = unsafe { (self.lib.syms.call_reset)(self.handle.0, 0, &mut out, &mut trap_code) };
        match rc {
            0 => Ok(()),
            1 => Err(self.lib.trap_error(trap_code)),
            2 => Err(self.host.pending_error.take().unwrap_or_else(|| self.lib.trap_error(trap_code))),
            _ => bail!("invalid_abi: missing blot:reset"),
        }
    }

    pub fn call(&mut self, name: &str, argument: Argument) -> Result<Value> {
        ensure!(!self.disposed, "disposed_guest: the guest instance is disposed");
        let function =
            self.abi.function(name).ok_or_else(|| anyhow!("unknown_export: unknown function export {name}"))?.clone();
        let result = self.begin_and_run(&function, argument);
        if let Some(probe) = &self.host.probe {
            probe.called(result.is_err());
        }
        let finish = self.finish(if result.is_ok() { LEFT_NORMALLY } else { LEFT_FAILING });
        let result = result?;
        finish?;
        Ok(result)
    }

    fn begin_and_run(&mut self, function: &GuestFunction, argument: Argument) -> Result<Value> {
        let name = &function.name;
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
                aot_to_bits(
                    self.host.uses_arrays,
                    &mut self.host.spare,
                    value,
                    |bytes| {
                        let mut ptr_out = 0i32;
                        let mut trap_code = 0u8;
                        let rc = unsafe { (lib.syms.call_allocate)(handle, bytes, &mut ptr_out, &mut trap_code) };
                        match rc {
                            0 => Ok(ptr_out),
                            1 => Err(lib.trap_error(trap_code)),
                            _ => bail!("invalid_abi: missing blot:allocate"),
                        }
                    },
                    || {
                        let mut p: *mut u8 = std::ptr::null_mut();
                        let mut n: usize = 0;
                        unsafe { (lib.syms.mem_data)(handle, &mut p, &mut n) };
                        if n == 0 || p.is_null() {
                            &mut []
                        } else {
                            unsafe { std::slice::from_raw_parts_mut(p, n) }
                        }
                    },
                )?
            }
            (Parameter::Callback(signature), Argument::Callback(callback)) => {
                let token = TOKENS.fetch_add(1, Ordering::Relaxed);
                self.host.active = Some(Active { token, signature, callback: Some(callback), export: name.clone() });
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
        aot_from_bits(mem, self.host.uses_arrays, &mut self.host.spare, function.result, res_bits)
    }

    fn finish(&mut self, outcome: i32) -> Result<()> {
        self.host.active = None;
        let mut left = Ok(());
        if std::mem::take(&mut self.entered) && self.abi.ownership {
            left = self.call_export_raw(OWNERSHIP_LEAVE, u64::from(outcome.cast_unsigned())).map(|_| ());
        }
        let invocation = if self.abi.invocation_finish {
            self.last_reclaimed_owners = None;
            let mut released = 0i32;
            let mut trap_code = 0u8;
            let rc = unsafe { (self.lib.syms.call_finish_invocation)(self.handle.0, &mut released, &mut trap_code) };
            match rc {
                0 => {
                    self.last_reclaimed_owners = Some(released.cast_unsigned());
                    Ok(())
                }
                1 => Err(self.lib.trap_error(trap_code)),
                2 => Err(self.host.pending_error.take().unwrap_or_else(|| self.lib.trap_error(trap_code))),
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
        let names = self.imaged.as_ref().as_ref().map_err(|why| anyhow!("this guest cannot be imaged: {why}"))?;
        let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
        unsafe { (self.lib.syms.mem_data)(self.handle.0, &mut p, &mut n) };
        let bytes = if n == 0 || p.is_null() { &[][..] } else { unsafe { std::slice::from_raw_parts(p, n) } };
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
        let mut written = 0;
        if self.abi.uses_arrays() {
            let current = self.memory_bytes();
            let missing = image.memory.len().saturating_sub(current);
            if missing > 0 {
                let delta = u64::try_from(missing.div_ceil(PAGE))?;
                let old = unsafe { (self.lib.syms.mem_grow)(self.handle.0, delta) };
                ensure!(old >= 0, "failed to grow guest memory during restore");
            }
            let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
            unsafe { (self.lib.syms.mem_data)(self.handle.0, &mut p, &mut n) };
            if n > 0 && !p.is_null() {
                let bytes = unsafe { std::slice::from_raw_parts_mut(p, n) };
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
        aot_from_bits(&[], false, &mut self.host.spare, constant.value, val)
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
