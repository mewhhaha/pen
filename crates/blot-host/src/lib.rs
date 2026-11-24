//! Runs Blot guest programs on wasmtime, following Blot's guest ABI 2
//! (`blot2/docs/host/guest-abi.md`): scalars and copied numeric arrays cross
//! the boundary, and an entrypoint may take one host callback.
//!
//! The JavaScript adapter needs JSPI to suspend a guest inside a host
//! callback. Here a callback simply blocks its thread, which preserves the
//! Wasm stack the same way: run a long-lived `main(host)` on its own thread
//! and let the callback wait on a channel.
//!
//! A guest of the development engine that runs past its [`DEADLINE`] without
//! returning or calling the host traps with an [`Overrun`]; time it spends
//! blocked in a callback does not count. A [`Probe`] shows the dev tools its
//! memory, ownership counters and, sampled, where it spends its time.
pub mod kernels;
mod native;
mod probe;
pub use native::native_fingerprint;
#[cfg(feature = "compiler")]
pub use native::native_metadata;
pub mod processes;
mod system_trace;
mod tasks;
#[cfg(feature = "compiler")]
mod trace_instrument;
mod wtime_aot;
pub use system_trace::SystemTrace;
pub use wtime_aot::{X64_ADMISSION_REJECTED_EXIT_CODE, X64FallbackError, X64FallbackReason};

use std::any::Any;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use pen_runtime::abi::INVOCATION_FINISH;
use pen_runtime::memory_image::{Captured, MemoryImage, PAGE};
use pen_runtime::symbols::{Frame, Symbols};
use wasmtime::{
    AsContext, AsContextMut, Caller, Engine, ExternRef, FuncType, Global, Instance, Linker, Memory, Module,
    ResourceLimiter, Store, Trap, TypedFunc, UpdateDeadline, Val, ValType, WasmBacktrace,
};

pub use probe::{Probe, Profile, Seen, function_name};

/// The host import module every capability adapter comes from.
pub const HOST_MODULE: &str = "blot:host/1";

/// Fixed unused Page-0 linear-memory offset written with `1u32` right before a
/// physical retained-chunk / free-list / `alloc_raw` arena OOM trap.
pub const ARENA_OOM_OFFSET: usize = 640;

// The manifest is pen-runtime's: the browser reads it too.
pub use pen_runtime::abi::{
    CallbackType, DEFAULT_TRANSPORT_BUDGET, GuestAbi, GuestConstant, GuestFunction, KernelManifest, MemoryShape,
    Parameter, ProcessManifest, ValueType, WireSchema, WireSchemaKind, parse_abi, read_abi, read_memory_shape,
};

/// The Wasm type a value crosses the boundary as.
fn wasm_type(value: ValueType) -> ValType {
    match value {
        ValueType::F32 => ValType::F32,
        _ => ValType::I32,
    }
}

#[derive(Clone)]
enum CompiledModule {
    Wasmtime(Module),
    WtimeAot(wtime_aot::AotModule),
}

/// A compiled guest module, its manifest and its debug information, ready
/// to instantiate many times.
#[derive(Clone)]
pub struct GuestModule {
    module: CompiledModule,
    trace: Option<Arc<SystemTrace>>,
    abi: Arc<GuestAbi>,
    symbols: Arc<Symbols>,
    /// The exported mutable globals an image of an instance keeps, or why
    /// one cannot ([`Guest::capture`]).
    imaged: Arc<Result<Vec<String>, String>>,
}

/// The mutable globals an image keeps: all of the module's, when it
/// exports them all, by their export names.
#[cfg(feature = "compiler")]
fn imaged_globals(wasm: &[u8]) -> Result<Vec<String>, String> {
    let mut mutable = Vec::new();
    let mut exported = std::collections::HashMap::new();
    let mut imported = 0;
    let mut parser = wasmparser::Parser::new(0);
    let mut offset = 0usize;
    loop {
        let rest = &wasm[offset..];
        let (consumed, payload) = match parser.parse(rest, true).map_err(|error| error.to_string())? {
            wasmparser::Chunk::Parsed { consumed, payload } => (consumed, payload),
            wasmparser::Chunk::NeedMoreData(_) => return Err("unexpected EOF in wasm module".into()),
        };
        match payload {
            wasmparser::Payload::CodeSectionStart { size, .. } => {
                let size = usize::try_from(size).map_err(|error| error.to_string())?;
                let skip = consumed.checked_add(size).ok_or_else(|| "code section size overflow".to_string())?;
                if skip > rest.len() {
                    return Err("truncated code section".into());
                }
                offset += skip;
                parser.skip_section();
                continue;
            }
            wasmparser::Payload::ImportSection(section) => {
                for import in section.into_imports() {
                    match import.map_err(|error| error.to_string())?.ty {
                        wasmparser::TypeRef::Global(_) => imported += 1,
                        wasmparser::TypeRef::Table(table_ty) if table_ty.element_type.is_extern_ref() => {
                            return Err("mutable externref tables are not captured by snapshots".into());
                        }
                        _ => {}
                    }
                }
            }
            wasmparser::Payload::TableSection(section) => {
                for table in section {
                    let table = table.map_err(|error| error.to_string())?;
                    if table.ty.element_type.is_extern_ref() {
                        return Err("mutable externref tables are not captured by snapshots".into());
                    }
                }
            }
            wasmparser::Payload::GlobalSection(section) => {
                for (index, global) in section.into_iter().enumerate() {
                    let global = global.map_err(|error| error.to_string())?;
                    if global.ty.mutable {
                        if !matches!(
                            global.ty.content_type,
                            wasmparser::ValType::I32
                                | wasmparser::ValType::I64
                                | wasmparser::ValType::F32
                                | wasmparser::ValType::F64
                        ) {
                            return Err(format!("its mutable global {index} holds a reference"));
                        }
                        mutable.push(imported + u32::try_from(index).map_err(|error| error.to_string())?);
                    }
                }
            }
            wasmparser::Payload::ExportSection(section) => {
                for export in section {
                    let export = export.map_err(|error| error.to_string())?;
                    if export.kind == wasmparser::ExternalKind::Global {
                        exported.entry(export.index).or_insert_with(|| export.name.to_owned());
                    }
                }
            }
            wasmparser::Payload::End(_) => break,
            _ => {}
        }
        offset += consumed;
    }
    let hidden = mutable.iter().filter(|index| !exported.contains_key(index)).count();
    if hidden > 0 {
        return Err(format!(
            "{hidden} of its {} mutable globals are not exported, out of the host's reach",
            mutable.len()
        ));
    }
    Ok(mutable.iter().map(|index| exported[index].clone()).collect())
}

/// Whether `engine` permits unmetered `wtime-aot` execution (neither epoch interruption nor fuel).
fn engine_allows_aot(engine: &Engine) -> bool {
    !engine.get_epoch_interruption() && !engine.get_consume_fuel()
}

impl GuestModule {
    /// Compiles `wasm`, reusing a compiled copy from `cache` when one exists
    /// for these bytes and this engine's configuration. Fails without the
    /// `compiler` feature.
    ///
    /// When `cache` is `Some`, the cache directory MUST be an owner-private
    /// (`0o700`) local directory with trusted ancestors; native `.cwasm`/`.so`
    /// caches fail closed before reading or writing if ownership or ancestor
    /// permissions cannot be verified.
    pub fn new(engine: &Engine, wasm: &[u8], cache: Option<&Path>) -> Result<GuestModule> {
        Self::new_with_aot(engine, wasm, cache, false)
    }

    /// Compiles `wasm` for a `pen run` session, selecting `wtime-aot` when
    /// available and permitted by the engine/ABI (unless `PEN_WTIME_AOT=0`),
    /// otherwise compiling with Wasmtime.
    ///
    /// When `cache` is `Some`, enforces the same owner-private (`0o700`) cache
    /// hierarchy trust boundary as [`GuestModule::new`].
    pub fn new_run(engine: &Engine, wasm: &[u8], cache: Option<&Path>) -> Result<GuestModule> {
        Self::new_with_aot(engine, wasm, cache, true)
    }

    #[cfg(feature = "compiler")]
    fn new_wasmtime(engine: &Engine, wasm: &[u8], cache: Option<&Path>) -> Result<GuestModule> {
        let abi = Arc::new(read_abi(wasm)?);
        let shape = read_memory_shape(wasm)?;
        validate_memory_shape(&shape)?;
        let module = compile(engine, wasm, cache)?;
        validate_raw_memory_shape(&module, &shape)?;
        validate_module(&module, &abi)?;
        let imaged = Arc::new(imaged_globals(wasm));
        Ok(GuestModule {
            module: CompiledModule::Wasmtime(module),
            trace: None,
            abi,
            symbols: Arc::new(Symbols::read(wasm)),
            imaged,
        })
    }

    fn new_with_aot(engine: &Engine, wasm: &[u8], cache: Option<&Path>, prefer_aot: bool) -> Result<GuestModule> {
        #[cfg(feature = "compiler")]
        {
            let abi = Arc::new(read_abi(wasm)?);
            let shape = read_memory_shape(wasm)?;
            validate_memory_shape(&shape)?;
            let use_aot = if prefer_aot { wtime_aot::preferred_for_run() } else { wtime_aot::enabled() };
            let (module, imaged) = if use_aot
                && engine_allows_aot(engine)
                && abi.tasks.is_none()
                && abi.kernels.is_empty()
                && abi.processes.is_empty()
            {
                let (aot, imaged) = wtime_aot::AotModule::compile(wasm, &abi, cache)?;
                (CompiledModule::WtimeAot(aot), Arc::new(imaged))
            } else {
                let module = compile(engine, wasm, cache)?;
                validate_raw_memory_shape(&module, &shape)?;
                validate_module(&module, &abi)?;
                (CompiledModule::Wasmtime(module), Arc::new(imaged_globals(wasm)))
            };
            Ok(GuestModule { module, trace: None, abi, symbols: Arc::new(Symbols::read(wasm)), imaged })
        }
        #[cfg(not(feature = "compiler"))]
        {
            let _ = (engine, wasm, cache, prefer_aot);
            bail!(NO_COMPILER)
        }
    }

    /// Underlying Wasmtime [`Module`], when not executing via `wtime-aot`.
    pub fn wasmtime_module(&self) -> Option<&Module> {
        match &self.module {
            CompiledModule::Wasmtime(module) => Some(module),
            CompiledModule::WtimeAot(_) => None,
        }
    }

    /// Underlying Wasmtime [`Engine`] this module was compiled/deserialized for.
    pub fn engine(&self) -> Option<&Engine> {
        self.wasmtime_module().map(Module::engine)
    }

    /// Backend executing this module (`"wtime-aot"` or `"wasmtime"`).
    pub fn backend_name(&self) -> &'static str {
        match &self.module {
            CompiledModule::Wasmtime(_) => "wasmtime",
            CompiledModule::WtimeAot(_) => "wtime-aot",
        }
    }

    /// Instruments allocation traffic and selects the game's explicit tracing
    /// entrypoint. The original bytes and normal entry remain unchanged.
    #[cfg(feature = "compiler")]
    pub fn traced(engine: &Engine, wasm: &[u8], cache: Option<&Path>, trace: Arc<SystemTrace>) -> Result<GuestModule> {
        // System tracing requires Wasmtime's `Tracing` hooks and is never routed to `wtime-aot`.
        let mut module = if trace.records_allocations() {
            Self::new_wasmtime(engine, &trace_instrument::instrument(wasm)?, cache)?
        } else {
            Self::new_wasmtime(engine, wasm, cache)?
        };
        let entry = module.abi.function("main_trace").context("system tracing needs a main_trace entrypoint")?;
        ensure!(
            entry.parameter == Parameter::Callback(pen_runtime::abi::GAME_HOST) && entry.result == ValueType::Unit,
            "main_trace must have the same signature as main"
        );
        module.trace = Some(trace);
        Ok(module)
    }

    pub fn system_trace(&self) -> Option<Arc<SystemTrace>> {
        self.trace.clone()
    }

    pub fn main_entry(&self) -> &'static str {
        if self.trace.is_some() { "main_trace" } else { "main" }
    }

    pub fn abi(&self) -> &GuestAbi {
        &self.abi
    }

    /// Its function names and source positions, which describe its traps
    /// (and are there for profiles).
    pub fn symbols(&self) -> &Symbols {
        &self.symbols
    }
}

/// Whether this build can compile Wasm (the `compiler` feature: Cranelift).
/// Without it, only modules [`precompile`]d by a build with it load.
pub const COMPILER: bool = cfg!(feature = "compiler");

#[cfg(not(feature = "compiler"))]
const NO_COMPILER: &str = "this runtime was built without a Wasm compiler (blot-host's `compiler` feature): it only loads \
     games precompiled for it";

/// Compiles `wasm` ahead of time for `engine` (see [`portable_engine`]), to
/// ship beside it.
pub fn precompile(engine: &Engine, wasm: &[u8]) -> Result<Vec<u8>> {
    read_abi(wasm)?;
    let shape = read_memory_shape(wasm)?;
    validate_memory_shape(&shape)?;
    #[cfg(feature = "compiler")]
    return Ok(engine.precompile_module(wasm)?);
    #[cfg(not(feature = "compiler"))]
    {
        let _ = engine;
        bail!(NO_COMPILER)
    }
}

#[cfg(feature = "compiler")]
fn compile(engine: &Engine, wasm: &[u8], cache: Option<&Path>) -> Result<Module> {
    match cache {
        Some(directory) => compile_cached(engine, wasm, directory),
        None => Ok(Module::new(engine, wasm)?),
    }
}

#[cfg(feature = "compiler")]
fn compile_cached(engine: &Engine, wasm: &[u8], directory: &Path) -> Result<Module> {
    let directory = wtime_aot::ensure_private_cache_dir(directory)?;
    let key = cache_key(engine, wasm);
    let path = directory.join(format!("{key}.cwasm"));
    if std::fs::symlink_metadata(&path).is_ok() {
        match read_cached_module(engine, &path, &key) {
            Some(module) => {
                // Its modification time says when it was last used, for `prune`.
                let _ = std::fs::File::options()
                    .write(true)
                    .open(&path)
                    .and_then(|file| file.set_modified(std::time::SystemTime::now()));
                return Ok(module);
            }
            None => {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
    let module = Module::new(engine, wasm)?;
    // A cache that cannot be written only costs the next launch a compile.
    let _ = (|| -> Result<()> {
        let serialized = module.serialize()?;
        if serialized.len() > CACHE_MODULE_BYTES {
            return Ok(());
        }
        static NEXT_WRITE: AtomicU64 = AtomicU64::new(0);
        let temporary =
            path.with_extension(format!("{}-{}.tmp", std::process::id(), NEXT_WRITE.fetch_add(1, Ordering::Relaxed)));
        write_cache_file(&temporary, &key, &serialized)?;
        if let Err(error) = std::fs::rename(&temporary, &path) {
            let _ = std::fs::remove_file(&temporary);
            return Err(error.into());
        }
        prune(&directory, std::time::SystemTime::now())?;
        Ok(())
    })();
    Ok(module)
}

/// This local cache contains native executable code. The directory is trusted;
/// the envelope detects accidental corruption and misplaced/stale artifacts,
/// not an attacker who can replace both the bytes and their checksum.
#[cfg(feature = "compiler")]
const CACHE_MAGIC: &[u8] = b"PENCWASM\x01";
#[cfg(feature = "compiler")]
const CACHE_HEADER_BYTES: usize = CACHE_MAGIC.len() + 64 + 32;
/// Optional disk entries above this size decline caching, without limiting
/// which modules can compile. With CACHE_ENTRIES this bounds retained disk data.
#[cfg(feature = "compiler")]
const CACHE_MODULE_BYTES: usize = 64 << 20;

#[cfg(feature = "compiler")]
fn write_cache_file(path: &Path, key: &str, serialized: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    // A colliding temporary file belongs to another writer. Only remove a
    // failed partial file after create_new has established our ownership.
    let mut opts = std::fs::File::options();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut file = opts.open(path)?;
    let written = (|| {
        file.write_all(&cache_header(key, serialized))?;
        file.write_all(serialized)
    })();
    if written.is_err() {
        drop(file);
        let _ = std::fs::remove_file(path);
    }
    written
}

#[cfg(feature = "compiler")]
fn cache_header(key: &str, serialized: &[u8]) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    [CACHE_MAGIC, key.as_bytes(), &Sha256::digest(serialized)].concat()
}

#[cfg(feature = "compiler")]
fn checked_cache_bytes<'a>(bytes: &'a [u8], key: &str) -> Option<&'a [u8]> {
    if key.len() != 64 || bytes.len() < CACHE_HEADER_BYTES || bytes.len() - CACHE_HEADER_BYTES > CACHE_MODULE_BYTES {
        return None;
    }
    let (header, serialized) = bytes.split_at(CACHE_HEADER_BYTES);
    if header != cache_header(key, serialized) {
        return None;
    }
    Some(serialized)
}

#[cfg(feature = "compiler")]
fn read_cached_module(engine: &Engine, path: &Path, key: &str) -> Option<Module> {
    use std::io::Read;
    let sym = std::fs::symlink_metadata(path).ok()?;
    if sym.file_type().is_symlink() || !sym.file_type().is_file() {
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let euid = wtime_aot::current_euid().ok()?;
        if sym.uid() != euid || sym.permissions().mode() & 0o022 != 0 {
            return None;
        }
    }
    let file = std::fs::File::open(path).ok()?;
    let meta = file.metadata().ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if meta.dev() != sym.dev() || meta.ino() != sym.ino() {
            return None;
        }
    }
    let limit = u64::try_from(CACHE_HEADER_BYTES + CACHE_MODULE_BYTES).ok()?;
    if meta.len() > limit {
        return None;
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes).ok()?;
    let serialized = checked_cache_bytes(&bytes, key)?;
    // SAFETY: the parent cache directory passed owner-only `0o700` hierarchy checks, the artifact
    // is a non-symlink regular file owned by `euid` with no group/other write bits, and the exact
    // source/engine key and payload checksum were checked before handing an owned buffer to Wasmtime.
    unsafe { Module::deserialize(engine, serialized) }.ok()
}

/// The module cache's name for `wasm` compiled by `engine`: SHA-256 over the
/// bytes and the engine's compatibility hash, which covers wasmtime's version,
/// the target and CPU features, the compiler's flags, the tunables (inlining
/// and epoch checks among them) and the Wasm features.
#[cfg(feature = "compiler")]
fn cache_key(engine: &Engine, wasm: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    use std::hash::Hash;
    struct Hasher(Sha256);
    impl std::hash::Hasher for Hasher {
        fn write(&mut self, bytes: &[u8]) {
            self.0.update(bytes);
        }
        fn finish(&self) -> u64 {
            unreachable!("only the digest is read")
        }
    }
    let mut hasher = Hasher(Sha256::new());
    hasher.0.update(wasm);
    engine.precompile_compatibility_hash().hash(&mut hasher);
    hasher.0.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The module cache keeps the compiled modules used last, this many (gdev's
/// is about 6 MB)...
#[cfg(feature = "compiler")]
const CACHE_ENTRIES: usize = 8;
/// ...and none unused for this long (two weeks).
#[cfg(feature = "compiler")]
const CACHE_UNUSED: Duration = Duration::from_secs(14 * 24 * 60 * 60);

/// Drops the module cache's entries beyond the newest [`CACHE_ENTRIES`] or
/// unused for [`CACHE_UNUSED`] as of `now`, by modification time, and
/// temporary files a write left behind an hour or more ago.
#[cfg(feature = "compiler")]
fn prune(directory: &Path, now: std::time::SystemTime) -> std::io::Result<()> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        let used = metadata.modified()?;
        let unused = now.duration_since(used).unwrap_or_default();
        match path.extension().and_then(|extension| extension.to_str()) {
            Some("cwasm") if metadata.len() > u64::try_from(CACHE_MODULE_BYTES + CACHE_HEADER_BYTES).unwrap() => {
                let _ = std::fs::remove_file(&path);
            }
            Some("cwasm") => entries.push((used, unused, path)),
            Some("tmp") if unused > Duration::from_secs(60 * 60) => {
                let _ = std::fs::remove_file(&path);
            }
            _ => {}
        }
    }
    entries.sort_by_key(|(used, ..)| std::cmp::Reverse(*used));
    for (index, (_, unused, path)) in entries.iter().enumerate() {
        if index >= CACHE_ENTRIES || *unused > CACHE_UNUSED {
            let _ = std::fs::remove_file(path);
        }
    }
    Ok(())
}

pub(crate) fn validate_memory_shape(shape: &MemoryShape) -> Result<()> {
    if shape.has_memory {
        ensure!(!shape.memory64, "invalid_abi: memory64 linear memories are unsupported");
        ensure!(!shape.shared, "invalid_abi: shared linear memories are unsupported");
        ensure!(
            shape.page_size_log2 == 16,
            "invalid_abi: custom Wasm page size 2^{} is unsupported (expected 64 KiB)",
            shape.page_size_log2
        );
    }
    Ok(())
}

pub(crate) fn validate_raw_memory_shape(module: &Module, shape: &MemoryShape) -> Result<()> {
    validate_memory_shape(shape)?;
    let req = module.resources_required();
    if !shape.has_memory {
        ensure!(
            req.num_memories == 0 && req.max_initial_memory_size.is_none(),
            "invalid_abi: module memory resources ({}, {:?}) do not match MemorySection (none)",
            req.num_memories,
            req.max_initial_memory_size
        );
    } else {
        ensure!(
            req.num_memories == 1 && req.max_initial_memory_size == Some(shape.initial_pages),
            "invalid_abi: module memory resources ({}, {:?}) do not match MemorySection (1, Some({}))",
            req.num_memories,
            req.max_initial_memory_size,
            shape.initial_pages
        );
    }
    Ok(())
}

pub(crate) fn validate_native_memory_shape(module: &Module, abi: &GuestAbi, shape: &MemoryShape) -> Result<()> {
    let req = module.resources_required();
    if shape.section_present {
        return validate_raw_memory_shape(module, shape);
    }
    if req.num_memories == 0 {
        ensure!(
            req.max_initial_memory_size.is_none(),
            "invalid_abi: module with 0 memories reported initial memory pages"
        );
        return Ok(());
    }
    // Legacy Metadata v2 boundary: allow missing MemorySection ONLY when the module
    // has no `blot:kernels`/`blot:processes` and exports `blot:memory` directly.
    if abi.kernels.is_empty()
        && abi.processes.is_empty()
        && req.num_memories == 1
        && let Some(memory) = module.get_export("blot:memory").and_then(|export| export.memory().cloned())
    {
        ensure!(
            !memory.is_64() && !memory.is_shared() && memory.page_size_log2() == 16,
            "invalid_abi: legacy exported blot:memory has unsupported memory shape"
        );
        return Ok(());
    }
    bail!(
        "native metadata is missing Wasm MemorySection (Section 5) and cannot verify memory shape without a compiler; \
         repackage with `pen build` (for Pen games) or `GuestModule::native_bytes` / `blot_host::native_metadata` \
         (for standalone modules)"
    )
}

fn validate_module(module: &Module, abi: &GuestAbi) -> Result<()> {
    abi.check_hostable()?;
    if let Some(count) = abi.tasks {
        tasks::validate(module, count)?;
    }
    let adapters: Vec<String> = abi.callbacks().into_iter().map(CallbackType::import_name).collect();
    for import in module.imports() {
        ensure!(
            (import.module() == HOST_MODULE && adapters.iter().any(|name| name == import.name()))
                || (abi.tasks.is_some() && import.module() == tasks::MODULE),
            "invalid_abi: unexpected import {}::{}",
            import.module(),
            import.name()
        );
    }
    for function in &abi.functions {
        ensure!(
            module.get_export(&function.name).is_some_and(|export| export.func().is_some()),
            "invalid_abi: missing function export {}",
            function.name
        );
    }
    for constant in &abi.constants {
        ensure!(
            module.get_export(&constant.name).is_some_and(|export| export.global().is_some()),
            "invalid_abi: missing constant export {}",
            constant.name
        );
    }
    if abi.uses_arrays() {
        for name in ["blot:memory", "blot:allocate", "blot:reset"] {
            ensure!(module.get_export(name).is_some(), "invalid_abi: missing array arena export {name}");
        }
    }
    if abi.ownership {
        for name in [OWNERSHIP_ENTER, OWNERSHIP_LEAVE] {
            ensure!(
                module.get_export(name).is_some_and(|export| export.func().is_some()),
                "invalid_abi: missing ownership export {name}"
            );
        }
    }
    let invocation_finish = module.get_export(INVOCATION_FINISH);
    ensure!(
        invocation_finish.is_some() == abi.invocation_finish,
        "invalid_abi: {INVOCATION_FINISH} must match the invocation finish section"
    );
    if let Some(export) = invocation_finish {
        ensure!(
            export.func().is_some_and(|ty| ty.params().len() == 0
                && ty.results().len() == 1
                && ty.results().next().is_some_and(|result| result.is_i32())),
            "invalid_abi: invalid {INVOCATION_FINISH} signature"
        );
    }
    Ok(())
}

/// A value crossing the boundary.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Unit,
    U32(u32),
    Bool(bool),
    F32(f32),
    ArrayU32(Vec<u32>),
    ArrayF32(Vec<f32>),
}

impl Value {
    pub fn kind(&self) -> ValueType {
        match self {
            Value::Unit => ValueType::Unit,
            Value::U32(_) => ValueType::U32,
            Value::Bool(_) => ValueType::Bool,
            Value::F32(_) => ValueType::F32,
            Value::ArrayU32(_) => ValueType::ArrayU32,
            Value::ArrayF32(_) => ValueType::ArrayF32,
        }
    }

    pub fn into_f32_array(self) -> Result<Vec<f32>> {
        match self {
            Value::ArrayF32(values) => Ok(values),
            other => bail!("expected Array F32, got {}", other.kind()),
        }
    }
}

/// A host callback. It may block: the guest waits on the same thread.
pub type Callback = Box<dyn FnMut(Value) -> Result<Value> + Send>;

/// What an entrypoint is called with.
pub enum Argument {
    Value(Value),
    Callback(Callback),
}

impl From<Value> for Argument {
    fn from(value: Value) -> Argument {
        Argument::Value(value)
    }
}

/// The opaque reference one invocation's callback is called through.
struct Token(u64);

static TOKENS: AtomicU64 = AtomicU64::new(1);

struct Active {
    token: u64,
    signature: CallbackType,
    callback: Option<Callback>,
    export: String,
}

#[derive(Clone)]
struct Arena {
    memory: Memory,
    allocate: TypedFunc<i32, i32>,
    reset: TypedFunc<i32, i32>,
}

/// ABI 3's hooks around each call ([`GuestAbi::ownership`]).
const OWNERSHIP_ENTER: &str = "blot:ownership_enter";
const OWNERSHIP_LEAVE: &str = "blot:ownership_leave";

/// `blot:ownership_leave`'s outcomes: a call returned or failed, or (idle)
/// the instance is torn down, releasing the module's roots.
const LEFT_NORMALLY: i32 = 0;
const LEFT_FAILING: i32 = 1;
const TEARDOWN: i32 = 2;

#[derive(Clone)]
struct Ownership {
    enter: TypedFunc<(), ()>,
    leave: TypedFunc<i32, ()>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DeadlineMode {
    Relative(Duration),
    Absolute(Instant),
}

/// Always-installed resource limiter tracking linear memory bytes (including
/// unexported hidden memories) and physical growth failures (`saw_oom`).
struct MemoryTrackerLimiter {
    inner: wasmtime::StoreLimits,
    bytes: usize,
    pending_grow: Option<usize>,
    saw_oom: bool,
}

impl Default for MemoryTrackerLimiter {
    fn default() -> Self {
        Self::new(None)
    }
}

impl MemoryTrackerLimiter {
    fn new(limits: Option<Limits>) -> Self {
        let inner = match limits {
            Some(limits) => wasmtime::StoreLimitsBuilder::new().memory_size(limits.memory).instances(1).build(),
            None => wasmtime::StoreLimitsBuilder::new().memory_size(usize::MAX).build(),
        };
        Self { inner, bytes: 0, pending_grow: None, saw_oom: false }
    }
}

impl ResourceLimiter for MemoryTrackerLimiter {
    fn memory_growing(&mut self, current: usize, desired: usize, maximum: Option<usize>) -> wasmtime::Result<bool> {
        let allowed = self.inner.memory_growing(current, desired, maximum)?;
        if allowed {
            self.pending_grow = Some(current);
            self.bytes = desired;
        } else {
            self.pending_grow = None;
            self.saw_oom = true;
        }
        Ok(allowed)
    }

    fn memory_grow_failed(&mut self, error: wasmtime::Error) -> wasmtime::Result<()> {
        if let Some(prev) = self.pending_grow.take() {
            self.bytes = prev;
        }
        self.saw_oom = true;
        self.inner.memory_grow_failed(error)
    }

    fn table_growing(&mut self, current: usize, desired: usize, maximum: Option<usize>) -> wasmtime::Result<bool> {
        self.inner.table_growing(current, desired, maximum)
    }

    fn table_grow_failed(&mut self, error: wasmtime::Error) -> wasmtime::Result<()> {
        self.inner.table_grow_failed(error)
    }

    fn instances(&self) -> usize {
        self.inner.instances()
    }

    fn tables(&self) -> usize {
        self.inner.tables()
    }

    fn memories(&self) -> usize {
        self.inner.memories()
    }
}

#[derive(Default)]
struct State {
    tasks: Option<tasks::Tasks>,
    tracing: Option<system_trace::Tracing>,
    active: Option<Active>,
    arena: Option<Arena>,
    ownership: Option<Ownership>,
    /// Whether this call entered ownership, so it must leave.
    entered: bool,
    /// Committed linear-memory bytes as of the guest's last host call.
    memory: Option<Arc<AtomicUsize>>,
    /// When the guest last started running (again), its deadline counting from then.
    resumed: Option<Instant>,
    /// Explicit relative or absolute deadline mode.
    deadline_mode: Option<DeadlineMode>,
    /// When true, `deadline_mode == None` means no relative `DEADLINE` (only `cancel_hook`).
    no_relative_deadline: bool,
    /// Optional cooperative/wave cancellation flag checked at epoch interrupts.
    cancel_hook: Option<Arc<AtomicBool>>,
    /// What the dev tools see of it, and the counters it exports for them.
    probe: Option<Arc<Probe>>,
    counters: Vec<Global>,
    /// Its module's symbols, which name the functions a profile samples.
    symbols: Arc<Symbols>,
    /// The last array passed in, its buffer reused for the next array out.
    spare: Vec<u32>,
    /// Torn down ([`Guest::dispose`]): it takes no more calls.
    disposed: bool,
    /// Whether the module declares a linear memory (`resources_required().num_memories > 0`).
    has_guest_memory: bool,
    /// Always-installed memory tracker and resource limiter.
    limiter: MemoryTrackerLimiter,
}

/// What a sandboxed guest ([`Guest::sandboxed`]) may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Linear memory, in bytes: growing past it fails (`memory.grow`
    /// answers -1), and a module that starts with more does not start.
    pub memory: usize,
}

/// Why a guest stopped: it ran this long without returning or calling the
/// host, past its [`DEADLINE`].
/// Errors from [`Guest::call`] carry it as context over the trap and its
/// Wasm backtrace, which say where the guest was.
#[derive(Clone, Copy, Debug)]
pub struct Overrun {
    pub ran: Duration,
}

impl fmt::Display for Overrun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the guest ran {:.1} s without calling the host", self.ran.as_secs_f64())
    }
}

/// Epoch ticks of [`dev_engine`] in `time`, at least one.
fn ticks(time: Duration) -> u64 {
    let tick = DEADLINE / DEADLINE_TICKS;
    u64::try_from(time.as_nanos().div_ceil(tick.as_nanos())).unwrap_or(u64::MAX).max(1)
}

/// Starts the guest's deadline afresh (for an engine that has one): it is
/// about to run (again), and only the time it runs counts. A sampled guest
/// looks at the next tick.
fn resume(mut store: impl AsContextMut<Data = State>) {
    let mut store = store.as_context_mut();
    let sampled = store.data().probe.as_ref().is_some_and(|probe| probe.sampling().is_some());
    let mode = store.data().deadline_mode;
    let no_relative = store.data().no_relative_deadline;
    let deadline_ticks = match mode {
        Some(DeadlineMode::Absolute(_)) => 1,
        _ if no_relative => 1,
        Some(DeadlineMode::Relative(limit)) => {
            if sampled {
                1
            } else {
                ticks(limit)
            }
        }
        None => {
            if sampled {
                1
            } else {
                ticks(DEADLINE)
            }
        }
    };
    store.set_epoch_deadline(deadline_ticks);
    store.data_mut().resumed = Some(Instant::now());
}

/// At the guest's epoch deadline: samples it while its probe samples, then
/// interrupts it once it has run past its deadline or its cancel hook fired,
/// else continues without falsely interrupting unrelated stores sharing an engine.
fn deadline_reached(mut store: wasmtime::StoreContextMut<'_, State>) -> wasmtime::Result<UpdateDeadline> {
    let profile = store.data().probe.as_ref().and_then(|probe| probe.sampling());
    if let Some(profile) = &profile {
        profile.sample(&WasmBacktrace::capture(&store), &store.data().symbols);
    }
    let state = store.data_mut();
    if state.cancel_hook.as_ref().is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Ok(UpdateDeadline::Interrupt);
    }
    match state.deadline_mode {
        Some(DeadlineMode::Absolute(deadline_at)) => {
            if Instant::now() >= deadline_at {
                Ok(UpdateDeadline::Interrupt)
            } else {
                Ok(UpdateDeadline::Continue(1))
            }
        }
        Some(DeadlineMode::Relative(limit)) => {
            let ran = state.resumed.map_or(Duration::ZERO, |resumed| resumed.elapsed());
            Ok(match limit.checked_sub(ran).filter(|left| !left.is_zero()) {
                Some(_) if profile.is_some() => UpdateDeadline::Continue(1),
                Some(left) => UpdateDeadline::Continue(ticks(left)),
                None => UpdateDeadline::Interrupt,
            })
        }
        None => {
            if state.no_relative_deadline {
                return Ok(UpdateDeadline::Continue(1));
            }
            let ran = state.resumed.map_or(Duration::ZERO, |resumed| resumed.elapsed());
            Ok(match DEADLINE.checked_sub(ran).filter(|left| !left.is_zero()) {
                Some(_) if profile.is_some() => UpdateDeadline::Continue(1),
                Some(left) => UpdateDeadline::Continue(ticks(left)),
                None => UpdateDeadline::Interrupt,
            })
        }
    }
}

/// `error`, marked as an [`Overrun`] when the guest's deadline stopped it.
fn overran(state: &State, error: anyhow::Error) -> anyhow::Error {
    match (error.downcast_ref::<Trap>(), state.resumed) {
        (Some(Trap::Interrupt), Some(resumed)) => error.context(Overrun { ran: resumed.elapsed() }),
        _ => error,
    }
}

struct WasmtimeGuest {
    store: Store<State>,
    instance: Instance,
    abi: Arc<GuestAbi>,
    symbols: Arc<Symbols>,
    invocation_finish: Option<TypedFunc<(), i32>>,
    last_reclaimed_owners: Option<u32>,
    imaged: Arc<Result<Vec<String>, String>>,
}

enum GuestBackend {
    Wasmtime(Box<WasmtimeGuest>),
    WtimeAot(Box<wtime_aot::AotGuest>),
}

/// One instance of a guest module.
pub struct Guest {
    backend: GuestBackend,
    abi: Arc<GuestAbi>,
}

/// An idle instance's state ([`Guest::capture`]): its memory and its
/// mutable globals.
#[derive(Clone, Debug, Default)]
pub struct GuestImage {
    pub memory: MemoryImage,
    globals: Vec<(String, Val)>,
}

impl Guest {
    pub fn new(engine: &Engine, module: &GuestModule) -> Result<Guest> {
        Guest::start(engine, module, None)
    }

    /// An instance that may use only so much (mods'). Its imports are only
    /// the callbacks its manifest declares, as every guest's are.
    pub fn sandboxed(engine: &Engine, module: &GuestModule, limits: Limits) -> Result<Guest> {
        Guest::start(engine, module, Some(limits))
    }

    fn start(engine: &Engine, module: &GuestModule, limits: Option<Limits>) -> Result<Guest> {
        match &module.module {
            CompiledModule::WtimeAot(aot) => {
                ensure!(
                    engine_allows_aot(engine),
                    "wtime-aot module cannot be instantiated on an epoch-interrupted or fuel-metered engine"
                );
                let guest = wtime_aot::AotGuest::start(aot, module.abi.clone(), module.imaged.clone(), limits)?;
                Ok(Guest { backend: GuestBackend::WtimeAot(Box::new(guest)), abi: module.abi.clone() })
            }
            CompiledModule::Wasmtime(wt_module) => {
                let guest = WasmtimeGuest::start(engine, wt_module, module, limits)?;
                Ok(Guest { backend: GuestBackend::Wasmtime(Box::new(guest)), abi: module.abi.clone() })
            }
        }
    }

    pub fn abi(&self) -> &GuestAbi {
        &self.abi
    }

    /// Lets the guest run up to `deadline` at a time instead of [`DEADLINE`],
    /// on [`dev_engine`], to within a fiftieth of [`DEADLINE`].
    pub fn set_deadline(&mut self, deadline: Duration) {
        match &mut self.backend {
            GuestBackend::Wasmtime(wt) => {
                wt.store.data_mut().no_relative_deadline = false;
                wt.store.data_mut().deadline_mode = Some(DeadlineMode::Relative(deadline));
            }
            GuestBackend::WtimeAot(_) => {
                panic!("Guest::set_deadline requires an epoch-interrupted Wasmtime engine (dev_engine), not wtime-aot");
            }
        }
    }

    /// Shows `probe` what the guest does from now on: its memory and the
    /// ownership counters it exports, as of each of its host calls, and the
    /// calls into it; and samples it into the profile attached to it.
    pub fn probe(&mut self, probe: Arc<Probe>) {
        match &mut self.backend {
            GuestBackend::Wasmtime(wt) => wt.probe(probe),
            GuestBackend::WtimeAot(aot) => aot.probe(probe),
        }
    }

    /// Committed linear-memory bytes, including temporary allocation capacity.
    pub fn memory_bytes(&self) -> usize {
        match &self.backend {
            GuestBackend::Wasmtime(wt) => wt.memory_bytes(),
            GuestBackend::WtimeAot(aot) => aot.memory_bytes(),
        }
    }

    /// Temporary owners measured and reclaimed by the previous ABI 8 invocation.
    pub fn last_reclaimed_owners(&self) -> Option<u32> {
        match &self.backend {
            GuestBackend::Wasmtime(wt) => wt.last_reclaimed_owners,
            GuestBackend::WtimeAot(aot) => aot.last_reclaimed_owners(),
        }
    }

    /// Keeps `gauge` at `memory_bytes`, updated whenever the guest calls the
    /// host, for other threads to read while it runs.
    pub fn track_memory(&mut self, gauge: Arc<AtomicUsize>) {
        match &mut self.backend {
            GuestBackend::Wasmtime(wt) => wt.track_memory(gauge),
            GuestBackend::WtimeAot(aot) => aot.track_memory(gauge),
        }
    }

    /// Calls the entrypoint `name`. A callback argument stays live only for
    /// this invocation, and the guest's arena is reset before and after it
    /// (and, under ABI 3, ownership entered before it and left after).
    pub fn call(&mut self, name: &str, argument: impl Into<Argument>) -> Result<Value> {
        match &mut self.backend {
            GuestBackend::Wasmtime(wt) => wt.call(name, argument.into()),
            GuestBackend::WtimeAot(aot) => aot.call(name, argument.into()),
        }
    }

    /// Tears the instance down, as Blot's adapter disposes of one.
    pub fn dispose(&mut self) -> Result<()> {
        match &mut self.backend {
            GuestBackend::Wasmtime(wt) => wt.dispose(),
            GuestBackend::WtimeAot(aot) => aot.dispose(),
        }
    }

    /// An image of the instance between calls: its memory (`blot:memory`)
    /// and its mutable globals, sharing the pages `previous` has unchanged.
    pub fn capture(&mut self, previous: Option<&GuestImage>) -> Result<(GuestImage, Captured)> {
        match &mut self.backend {
            GuestBackend::Wasmtime(wt) => wt.capture(previous),
            GuestBackend::WtimeAot(aot) => aot.capture(previous),
        }
    }

    /// Puts the instance back as `image` found it, growing its memory if
    /// it has less. Returns the pages written.
    pub fn restore(&mut self, image: &GuestImage) -> Result<usize> {
        match &mut self.backend {
            GuestBackend::Wasmtime(wt) => wt.restore(image),
            GuestBackend::WtimeAot(aot) => aot.restore(image),
        }
    }

    /// The ownership counters the guest exports, as they stand.
    pub fn ownership(&mut self) -> OwnershipCounters {
        let counters = self.runtime_counters();
        let counter = |name: &str| counters.iter().find(|(counter, _)| counter == name).map(|(_, value)| *value);
        OwnershipCounters {
            roots: counter("blot:rc_roots"),
            leases: counter("blot:rc_leases"),
            host_refs: counter("blot:rc_host_refs"),
            retains: counter("blot:rc_retains"),
            releases: counter("blot:rc_releases"),
            destroyed: counter("blot:rc_destroyed"),
            cycle_passes: counter("blot:cycle_passes"),
            cycle_vertices: counter("blot:cycle_vertices"),
            cycle_edges: counter("blot:cycle_edges"),
            cycle_bytes: counter("blot:cycle_bytes"),
        }
    }

    /// Reads an exported scalar constant.
    pub fn read(&mut self, name: &str) -> Result<Value> {
        match &mut self.backend {
            GuestBackend::Wasmtime(wt) => wt.read(name),
            GuestBackend::WtimeAot(aot) => aot.read(name),
        }
    }

    /// The guest runtime's own counters.
    pub fn runtime_counters(&mut self) -> Vec<(String, u32)> {
        match &mut self.backend {
            GuestBackend::Wasmtime(wt) => wt.runtime_counters(),
            GuestBackend::WtimeAot(aot) => aot.runtime_counters(),
        }
    }
}

impl WasmtimeGuest {
    fn start(engine: &Engine, wt_module: &Module, module: &GuestModule, limits: Option<Limits>) -> Result<Self> {
        Self::start_inner(engine, wt_module, module, limits, false, None, None, None)
    }

    pub(crate) fn start_controlled(
        engine: &Engine,
        wt_module: &Module,
        module: &GuestModule,
        limits: Option<Limits>,
        deadline_at: Option<Instant>,
        cancel_hook: Option<Arc<AtomicBool>>,
        initial_fuel: Option<u64>,
    ) -> Result<Self> {
        Self::start_inner(engine, wt_module, module, limits, true, deadline_at, cancel_hook, initial_fuel)
    }

    #[allow(clippy::too_many_arguments)]
    fn start_inner(
        engine: &Engine,
        wt_module: &Module,
        module: &GuestModule,
        limits: Option<Limits>,
        controlled: bool,
        deadline_at: Option<Instant>,
        cancel_hook: Option<Arc<AtomicBool>>,
        initial_fuel: Option<u64>,
    ) -> Result<Self> {
        let mut store = Store::new(
            engine,
            State {
                symbols: module.symbols.clone(),
                has_guest_memory: wt_module.resources_required().num_memories > 0,
                limiter: MemoryTrackerLimiter::new(limits),
                no_relative_deadline: controlled,
                deadline_mode: deadline_at.map(DeadlineMode::Absolute),
                cancel_hook,
                ..State::default()
            },
        );
        store.epoch_deadline_callback(deadline_reached);
        store.limiter(|state| &mut state.limiter);
        if engine.get_consume_fuel() {
            store.set_fuel(initial_fuel.unwrap_or(u64::MAX))?;
        }
        // Instantiating runs the module's start function, if it has one.
        resume(&mut store);
        let mut linker = Linker::new(engine);
        if module.abi.tasks.is_some() {
            tasks::link(&mut linker)?;
        }
        for signature in module.abi.callbacks() {
            let ty = FuncType::new(
                engine,
                [ValType::EXTERNREF, wasm_type(signature.parameter)],
                [wasm_type(signature.result)],
            );
            linker.func_new(HOST_MODULE, &signature.import_name(), ty, move |caller, params, results| {
                invoke_import(caller, signature, params, results).map_err(wasmtime::Error::from_anyhow)
            })?;
        }
        let instance =
            linker.instantiate(&mut store, wt_module).map_err(|error| overran(store.data(), error.into()))?;
        if module.abi.uses_arrays() || module.abi.tasks.is_some() {
            let memory = instance
                .get_memory(&mut store, "blot:memory")
                .ok_or_else(|| anyhow!("invalid_abi: missing array arena exports"))?;
            let allocate = instance.get_typed_func::<i32, i32>(&mut store, "blot:allocate")?;
            let reset = instance.get_typed_func::<i32, i32>(&mut store, "blot:reset")?;
            store.data_mut().arena = Some(Arena { memory, allocate, reset });
        }
        if module.abi.ownership {
            // Typed lookups check the hooks' exact signatures before either runs.
            let enter = instance.get_typed_func::<(), ()>(&mut store, OWNERSHIP_ENTER)?;
            let leave = instance.get_typed_func::<i32, ()>(&mut store, OWNERSHIP_LEAVE)?;
            store.data_mut().ownership = Some(Ownership { enter, leave });
        }
        if let Some(count) = module.abi.tasks {
            tasks::install(&mut store, instance, count)?;
        }
        if let Some(trace) = &module.trace {
            let globals = if trace.records_allocations() {
                let bytes = instance.get_global(&mut store, "pen:trace_bytes").context("missing trace byte counter")?;
                let allocations = instance
                    .get_global(&mut store, "pen:trace_allocations")
                    .context("missing trace allocation counter")?;
                Some([bytes, allocations])
            } else {
                None
            };
            store.data_mut().tracing = Some(system_trace::Tracing::new(trace.clone(), globals));
        }
        let invocation_finish = if module.abi.invocation_finish {
            Some(instance.get_typed_func::<(), i32>(&mut store, INVOCATION_FINISH)?)
        } else {
            None
        };
        Ok(WasmtimeGuest {
            store,
            instance,
            abi: module.abi.clone(),
            symbols: module.symbols.clone(),
            invocation_finish,
            last_reclaimed_owners: None,
            imaged: module.imaged.clone(),
        })
    }

    pub(crate) fn saw_oom(&self) -> bool {
        self.store.data().limiter.saw_oom
    }

    pub(crate) fn clear_saw_oom(&mut self) {
        self.store.data_mut().limiter.saw_oom = false;
    }

    pub(crate) fn arena_oom_flag(&self) -> u32 {
        let Some(arena) = &self.store.data().arena else {
            return 0;
        };
        let data = arena.memory.data(&self.store);
        if data.len() < ARENA_OOM_OFFSET + 4 {
            return 0;
        }
        u32::from_le_bytes(data[ARENA_OOM_OFFSET..ARENA_OOM_OFFSET + 4].try_into().unwrap())
    }

    pub(crate) fn has_arena(&self) -> bool {
        self.store.data().arena.is_some()
    }

    pub(crate) fn arm_execution(
        &mut self,
        deadline_at: Option<Instant>,
        cancel_hook: Option<Arc<AtomicBool>>,
        fuel: Option<u64>,
    ) -> Result<()> {
        if self.store.engine().get_consume_fuel() {
            self.store.set_fuel(fuel.unwrap_or(u64::MAX))?;
        }
        {
            let state = self.store.data_mut();
            state.no_relative_deadline = true;
            state.deadline_mode = deadline_at.map(DeadlineMode::Absolute);
            state.cancel_hook = cancel_hook;
        }
        resume(&mut self.store);
        Ok(())
    }

    pub(crate) fn reset_arena(&mut self) -> Result<()> {
        if let Some(arena) = self.store.data().arena.clone() {
            arena.reset.call(&mut self.store, 0).map_err(|error| symbolize(error, &self.symbols))?;
        }
        Ok(())
    }

    pub(crate) fn write_words_packet(&mut self, words: &[u32]) -> Result<i32> {
        let arena = self.store.data().arena.clone().ok_or_else(|| anyhow!("invalid_abi: array value requires an arena"))?;
        ensure!(words.len() < 1_073_741_823, "array byte count exceeds the Wasm32 address range");
        let length = u32::try_from(words.len())?;
        let byte_len = (length + 1).checked_mul(4).ok_or_else(|| anyhow!("packet byte count overflow"))?;
        let pointer = arena
            .allocate
            .call(&mut self.store, byte_len.cast_signed())
            .map_err(|error| symbolize(error, &self.symbols))?;
        let memory = arena.memory.data_mut(&mut self.store);
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

    pub(crate) fn call_raw(&mut self, name: &str, param: Val) -> Result<Val> {
        let func = self
            .instance
            .get_func(&mut self.store, name)
            .ok_or_else(|| anyhow!("validated function export {name} disappeared"))?;
        let mut results = [Val::I32(0)];
        func.call(&mut self.store, &[param], &mut results).map_err(|error| symbolize(error, &self.symbols))?;
        Ok(results[0])
    }

    /// Reads the raw 32-bit header word at `pointer` directly from Wasm linear memory
    /// BEFORE allocating any host copy (enabling `[0xffffffff]` sentinel and pre-copy size inspection).
    pub(crate) fn inspect_packet_header(&self, pointer: i32) -> Result<u32> {
        let arena = self.store.data().arena.as_ref().ok_or_else(|| anyhow!("invalid_abi: array value requires an arena"))?;
        let memory = arena.memory.data(&self.store);
        let address = array_range(memory.len(), pointer, 0)?;
        let header = u32::from_le_bytes(memory[address..address + 4].try_into().unwrap());
        if header != u32::MAX {
            array_range(memory.len(), pointer, header as usize)?;
        }
        Ok(header)
    }

    /// Copies `expected_words` words directly from Wasm linear memory into `dst` (zero intermediate buffers).
    pub(crate) fn read_packet_words_into(&self, pointer: i32, expected_words: usize, dst: &mut [u32]) -> Result<()> {
        ensure!(dst.len() == expected_words, "destination slice length mismatch");
        let arena = self.store.data().arena.as_ref().ok_or_else(|| anyhow!("invalid_abi: array value requires an arena"))?;
        let memory = arena.memory.data(&self.store);
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
        match &self.invocation_finish {
            Some(finish) => {
                self.last_reclaimed_owners = None;
                let released = finish.call(&mut self.store, ()).map_err(|error| symbolize(error, &self.symbols))?;
                let released = released.cast_unsigned();
                self.last_reclaimed_owners = Some(released);
                Ok(Some(released))
            }
            None => Ok(None),
        }
    }

    fn probe(&mut self, probe: Arc<Probe>) {
        let counters = self.counter_globals();
        probe.counters_named(counters.iter().map(|(name, _)| name.trim_start_matches("blot:").to_owned()).collect());
        let state = self.store.data_mut();
        state.counters = counters.into_iter().map(|(_, global)| global).collect();
        state.probe = Some(probe);
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        match &self.store.data().arena {
            Some(arena) => arena.memory.data_size(&self.store),
            None if self.store.data().has_guest_memory => self.store.data().limiter.bytes,
            None => 0,
        }
    }

    fn track_memory(&mut self, gauge: Arc<AtomicUsize>) {
        gauge.store(self.memory_bytes(), Ordering::Relaxed);
        self.store.data_mut().memory = Some(gauge);
    }

    fn call(&mut self, name: &str, argument: Argument) -> Result<Value> {
        ensure!(!self.store.data().disposed, "disposed_guest: the guest instance is disposed");
        let function =
            self.abi.function(name).ok_or_else(|| anyhow!("unknown_export: unknown function export {name}"))?.clone();
        let result = self.begin_and_run(&function, argument).map_err(|error| overran(self.store.data(), error));
        let trace_finish = if let Some(mut tracing) = self.store.data_mut().tracing.take() {
            let counts = tracing.counts(&mut self.store);
            let aborted = tracing.abort(Instant::now(), counts);
            self.store.data_mut().tracing = Some(tracing);
            aborted
        } else {
            Ok(())
        };
        if let Some(probe) = &self.store.data().probe {
            probe.called(result.is_err());
        }
        let finish = self.finish(if result.is_ok() { LEFT_NORMALLY } else { LEFT_FAILING });
        let result = result?;
        finish?;
        trace_finish?;
        Ok(result)
    }

    fn begin_and_run(&mut self, function: &GuestFunction, argument: Argument) -> Result<Value> {
        let name = &function.name;
        resume(&mut self.store);
        if let Some(ownership) = self.store.data().ownership.clone() {
            ownership.enter.call(&mut self.store, ())?;
            self.store.data_mut().entered = true;
        }
        tasks::begin(&mut self.store)?;
        if let Some(arena) = self.store.data().arena.clone() {
            arena.reset.call(&mut self.store, 0)?;
        }
        let parameter = match (function.parameter, argument) {
            (Parameter::Value(expected), Argument::Value(value)) => {
                ensure!(
                    value.kind() == expected,
                    "invalid_argument: {name} argument must be {expected}, not {}",
                    value.kind()
                );
                let arena = self.store.data().arena.clone();
                to_wasm(&mut self.store, arena.as_ref(), value)?
            }
            (Parameter::Callback(signature), Argument::Callback(callback)) => {
                let token = TOKENS.fetch_add(1, Ordering::Relaxed);
                let reference = ExternRef::new(&mut self.store, Token(token))?;
                self.store.data_mut().active =
                    Some(Active { token, signature, callback: Some(callback), export: name.clone() });
                Val::ExternRef(Some(reference))
            }
            (Parameter::Callback(_), Argument::Value(_)) => {
                bail!("invalid_capability: {name} requires a host callback")
            }
            (Parameter::Value(_), Argument::Callback(_)) => {
                bail!("invalid_argument: {name} does not take a host callback")
            }
        };
        let func = self
            .instance
            .get_func(&mut self.store, name)
            .ok_or_else(|| anyhow!("validated function export {name} disappeared"))?;
        let mut results = [Val::I32(0)];
        func.call(&mut self.store, &[parameter], &mut results).map_err(|error| symbolize(error, &self.symbols))?;
        let arena = self.store.data().arena.clone();
        from_wasm(&mut self.store, arena.as_ref(), function.result, &results[0])
    }

    fn finish(&mut self, outcome: i32) -> Result<()> {
        let task_result = tasks::finish(&mut self.store, outcome != LEFT_NORMALLY);
        self.store.data_mut().active = None;
        resume(&mut self.store);
        let mut left = Ok(());
        if std::mem::take(&mut self.store.data_mut().entered)
            && let Some(ownership) = self.store.data().ownership.clone()
        {
            left = ownership.leave.call(&mut self.store, if task_result.is_err() { LEFT_FAILING } else { outcome });
        }
        let invocation = match &self.invocation_finish {
            Some(finish) => {
                self.last_reclaimed_owners = None;
                finish.call(&mut self.store, ()).map(|released| {
                    self.last_reclaimed_owners = Some(released.cast_unsigned());
                })
            }
            None => Ok(()),
        };
        if let Some(arena) = self.store.data().arena.clone() {
            arena.reset.call(&mut self.store, 0)?;
        }
        task_result?;
        left?;
        invocation?;
        Ok(())
    }

    fn dispose(&mut self) -> Result<()> {
        if std::mem::replace(&mut self.store.data_mut().disposed, true) {
            return Ok(());
        }
        resume(&mut self.store);
        let tasks = tasks::dispose(&mut self.store);
        let mut left = Ok(());
        if let Some(ownership) = self.store.data().ownership.clone() {
            left = ownership.leave.call(&mut self.store, TEARDOWN).map_err(|error| symbolize(error, &self.symbols));
            if let Some(arena) = self.store.data().arena.clone() {
                let reset = arena.reset.call(&mut self.store, 0);
                tasks?;
                left?;
                reset?;
                return Ok(());
            }
        }
        tasks?;
        left
    }

    fn capture(&mut self, previous: Option<&GuestImage>) -> Result<(GuestImage, Captured)> {
        ensure!(self.abi.tasks.is_none(), "task contexts are host state and cannot be memory-imaged");
        ensure!(!self.store.data().disposed, "disposed_guest: the guest instance is disposed");
        let names = self.imaged.as_ref().as_ref().map_err(|why| anyhow!("this guest cannot be imaged: {why}"))?;
        let memory = self.instance.get_memory(&mut self.store, "blot:memory");
        let bytes = memory.map(|memory| memory.data(&self.store)).unwrap_or_default();
        let (memory, captured) = MemoryImage::capture(bytes, previous.map(|previous| &previous.memory));
        let mut globals = Vec::with_capacity(names.len());
        for name in names {
            let global = self.instance.get_global(&mut self.store, name).context("an imaged global disappeared")?;
            globals.push((name.clone(), global.get(&mut self.store)));
        }
        Ok((GuestImage { memory, globals }, captured))
    }

    fn restore(&mut self, image: &GuestImage) -> Result<usize> {
        ensure!(self.abi.tasks.is_none(), "task contexts are host state and cannot be memory-imaged");
        ensure!(!self.store.data().disposed, "disposed_guest: the guest instance is disposed");
        // Preflight complete imaged-global schema (exact mutable export set, no duplicates/extras),
        // destination mutability, and Wasm value types BEFORE growing or mutating linear memory.
        let allowed = self.imaged.as_ref().as_ref().map_err(|why| anyhow!("this guest cannot be restored: {why}"))?;
        ensure!(
            image.globals.len() == allowed.len(),
            "imaged global schema mismatch: guest expects {} mutable globals, snapshot has {}",
            allowed.len(),
            image.globals.len()
        );
        let mut seen = std::collections::HashSet::with_capacity(allowed.len());
        let mut validated: Vec<(Global, Val)> = Vec::with_capacity(image.globals.len());
        for (name, value) in &image.globals {
            ensure!(
                allowed.iter().any(|a| a == name),
                "imaged global {name} is not a mutable imaged global of this guest"
            );
            ensure!(seen.insert(name.as_str()), "duplicate imaged global in snapshot: {name}");
            let global = self.instance.get_global(&mut self.store, name).context("an imaged global disappeared")?;
            let ty = global.ty(&self.store);
            ensure!(
                ty.mutability() == wasmtime::Mutability::Var,
                "imaged global {name} is not mutable"
            );
            let kind_matches = matches!(
                (ty.content(), value),
                (wasmtime::ValType::I32, Val::I32(_))
                    | (wasmtime::ValType::I64, Val::I64(_))
                    | (wasmtime::ValType::F32, Val::F32(_))
                    | (wasmtime::ValType::F64, Val::F64(_))
            );
            ensure!(kind_matches, "imaged global type mismatch for {name}");
            validated.push((global, *value));
        }
        let mut written = 0;
        if let Some(memory) = self.instance.get_memory(&mut self.store, "blot:memory") {
            let missing = image.memory.len().saturating_sub(memory.data_size(&self.store));
            if missing > 0 {
                memory.grow(&mut self.store, u64::try_from(missing.div_ceil(PAGE))?)?;
            }
            written = image.memory.restore(memory.data_mut(&mut self.store));
        }
        for (global, value) in validated {
            global.set(&mut self.store, value)?;
        }
        Ok(written)
    }

    fn read(&mut self, name: &str) -> Result<Value> {
        ensure!(!self.store.data().disposed, "disposed_guest: the guest instance is disposed");
        let constant = self
            .abi
            .constants
            .iter()
            .find(|constant| constant.name == name)
            .ok_or_else(|| anyhow!("unknown_export: unknown constant export {name}"))?
            .clone();
        let global = self
            .instance
            .get_global(&mut self.store, name)
            .ok_or_else(|| anyhow!("validated constant export {name} disappeared"))?;
        let value = global.get(&mut self.store);
        from_wasm(&mut self.store, None, constant.value, &value)
    }

    fn runtime_counters(&mut self) -> Vec<(String, u32)> {
        let globals = self.counter_globals();
        globals.into_iter().filter_map(|(name, global)| Some((name, counter(global.get(&mut self.store))?))).collect()
    }

    fn counter_globals(&mut self) -> Vec<(String, Global)> {
        let globals: Vec<(String, Global)> = self
            .instance
            .exports(&mut self.store)
            .filter(|export| export.name().starts_with("blot:"))
            .filter_map(|export| Some((export.name().to_owned(), export.into_global()?)))
            .collect();
        globals
            .into_iter()
            .filter(|(name, _)| self.abi.constants.iter().all(|constant| constant.name != *name))
            .collect()
    }
}

/// A trap with its backtrace described by the module's symbols (names, and
/// source positions when it has a source map) in place of wasmtime's, as the
/// browser host describes one; any other error as it is.
fn symbolize(error: wasmtime::Error, symbols: &Symbols) -> anyhow::Error {
    let frames = error.downcast_ref::<WasmBacktrace>().map(|backtrace| {
        let frames = backtrace.frames().iter();
        frames.map(|frame| Frame { function: frame.func_index(), offset: frame.module_offset() }).collect::<Vec<_>>()
    });
    match (error.downcast_ref::<Trap>(), frames) {
        (Some(trap), Some(frames)) => anyhow::Error::new(*trap).context(symbols.backtrace(&frames)),
        _ => error.into(),
    }
}

pub(crate) fn is_trap_interrupt(error: &anyhow::Error) -> bool {
    error.downcast_ref::<Trap>() == Some(&Trap::Interrupt)
}

pub(crate) fn is_unrecoverable_trap(error: &anyhow::Error) -> bool {
    matches!(
        error.downcast_ref::<Trap>(),
        Some(Trap::Interrupt | Trap::OutOfFuel | Trap::StackOverflow)
    )
}

/// What an ABI 3 guest's runtime counts of the memory it owns, through the
/// globals Blot exports beside its ownership hooks (`blot:rc_*`, and
/// `blot:cycle_*` with cycle collection); each is None when the module does
/// not export it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OwnershipCounters {
    /// Live roots and leases: zero again after each call returns, and after
    /// teardown, unless something leaks.
    pub roots: Option<u32>,
    pub leases: Option<u32>,
    /// Host references (callbacks) the guest holds.
    pub host_refs: Option<u32>,
    /// Running totals: references retained and released, objects destroyed.
    pub retains: Option<u32>,
    pub releases: Option<u32>,
    pub destroyed: Option<u32>,
    /// Cycle collection's running totals: passes, the vertices and edges
    /// they walked, and the bytes they reclaimed.
    pub cycle_passes: Option<u32>,
    pub cycle_vertices: Option<u32>,
    pub cycle_edges: Option<u32>,
    pub cycle_bytes: Option<u32>,
}

fn invoke_import(
    mut caller: Caller<'_, State>,
    signature: CallbackType,
    params: &[Val],
    results: &mut [Val],
) -> Result<()> {
    let token = match &params[0] {
        Val::ExternRef(Some(reference)) => reference
            .data(caller.as_context())?
            .and_then(|data: &(dyn Any + Send + Sync)| data.downcast_ref::<Token>())
            .map(|token| token.0),
        _ => None,
    };
    let (export, expected) = match &caller.data().active {
        Some(active) if Some(active.token) == token => (active.export.clone(), active.signature),
        _ => bail!("stale_capability: host call has no active capability reference"),
    };
    ensure!(expected == signature, "capability_signature: host call signature does not match its capability");
    let arena = caller.data().arena.clone();
    let observed = caller.data().tracing.as_ref().map(|_| Instant::now());
    let argument = from_wasm(&mut caller, arena.as_ref(), signature.parameter, &params[1])?;
    if let Some(mut tracing) = caller.data_mut().tracing.take() {
        let counts = tracing.counts(&mut caller);
        let now = observed.unwrap();
        let marked = match &argument {
            Value::ArrayF32(packet) => tracing.marker(packet, now, counts),
            _ => Ok(false),
        };
        caller.data_mut().tracing = Some(tracing);
        if marked? {
            let begin = matches!(&argument, Value::ArrayF32(packet) if packet[1] == 1.0);
            results[0] = to_wasm(&mut caller, arena.as_ref(), Value::ArrayF32(Vec::new()))?;
            let mut tracing = caller.data_mut().tracing.take().unwrap();
            let counts = tracing.counts(&mut caller);
            tracing.resume_marker(begin, Instant::now(), counts);
            caller.data_mut().tracing = Some(tracing);
            return Ok(());
        }
        let mut tracing = caller.data_mut().tracing.take().unwrap();
        let finished = tracing.finish_command(now, counts);
        caller.data_mut().tracing = Some(tracing);
        finished?;
    }
    if let Some(gauge) = &caller.data().memory {
        let current = match &arena {
            Some(a) => a.memory.data_size(&caller),
            None if caller.data().has_guest_memory => caller.data().limiter.bytes,
            None => 0,
        };
        gauge.store(current, Ordering::Relaxed);
    }
    if let Some(probe) = caller.data().probe.clone() {
        let memory = match &arena {
            Some(arena) => arena.memory.data_size(&caller),
            None if caller.data().has_guest_memory => caller.data().limiter.bytes,
            None => 0,
        };
        let counters = std::mem::take(&mut caller.data_mut().counters);
        let values = counters.iter().map(|global| counter(global.get(&mut caller)).map_or(0, u64::from));
        probe.called_host(memory, values);
        caller.data_mut().counters = counters;
    }
    let mut callback = caller
        .data_mut()
        .active
        .as_mut()
        .and_then(|active| active.callback.take())
        .ok_or_else(|| anyhow!("reentrant_call: cannot reenter an active guest instance"))?;
    let result = callback(argument);
    resume(&mut caller);
    if let Some(active) = caller.data_mut().active.as_mut() {
        active.callback = Some(callback);
    }
    let result = result.with_context(|| format!("host_exception: host callback failed during {export}"))?;
    ensure!(
        result.kind() == signature.result,
        "host_result: host callback during {export} returned {}, not {}",
        result.kind(),
        signature.result
    );
    let command = match &result {
        Value::ArrayF32(packet) if caller.data().tracing.is_some() && packet.len() >= 2 => Some([packet[0], packet[1]]),
        _ => None,
    };
    results[0] = to_wasm(&mut caller, arena.as_ref(), result)?;
    if let Some(command) = command {
        let mut tracing = caller.data_mut().tracing.take().unwrap();
        let counts = tracing.counts(&mut caller);
        tracing.resume_command(&command, Instant::now(), counts);
        caller.data_mut().tracing = Some(tracing);
    }
    Ok(())
}

/// A counter global's value, unsigned and wrapping at 2^32 as the guest's
/// 32-bit counters do; None for a global of another type.
fn counter(value: Val) -> Option<u32> {
    match value {
        Val::I32(value) => Some(value.cast_unsigned()),
        Val::I64(value) => u32::try_from(value.cast_unsigned() & u64::from(u32::MAX)).ok(),
        _ => None,
    }
}

/// The byte address of an array header at `pointer` holding `length` words,
/// checked against the arena's current size.
fn array_range(memory_bytes: usize, pointer: i32, length: usize) -> Result<usize> {
    let address = pointer.cast_unsigned() as usize;
    ensure!(
        address >= 4
            && address.is_multiple_of(4)
            && address + 4 <= memory_bytes
            && length <= (memory_bytes - address - 4) / 4,
        "invalid_abi: array lies outside the guest arena"
    );
    Ok(address)
}

/// Copies an array into the guest's arena, then keeps its buffer for the
/// next array out of it.
fn to_wasm(mut store: impl AsContextMut<Data = State>, arena: Option<&Arena>, value: Value) -> Result<Val> {
    let words: Vec<u32> = match value {
        Value::Unit => return Ok(Val::I32(0)),
        Value::U32(value) => return Ok(Val::I32(value.cast_signed())),
        Value::Bool(value) => return Ok(Val::I32(value as i32)),
        Value::F32(value) => return Ok(Val::F32(value.to_bits())),
        Value::ArrayU32(values) => values,
        Value::ArrayF32(values) => bytemuck::allocation::cast_vec(values),
    };
    let arena = arena.ok_or_else(|| anyhow!("invalid_abi: array value requires an arena"))?;
    ensure!(words.len() < 1_073_741_823, "array byte count exceeds the Wasm32 address range");
    // So its length word and words' bytes fit a u32, which Wasm passes as an i32.
    let length = u32::try_from(words.len())?;
    let pointer = arena.allocate.call(&mut store, ((length + 1) * 4).cast_signed())?;
    // Allocation may grow memory, so look at its size only afterwards.
    let memory = arena.memory.data_mut(&mut store);
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
    store.as_context_mut().data_mut().spare = words;
    Ok(Val::I32(pointer))
}

fn from_wasm(
    mut store: impl AsContextMut<Data = State>,
    arena: Option<&Arena>,
    kind: ValueType,
    value: &Val,
) -> Result<Value> {
    let non_canonical = || anyhow!("invalid_abi: non-canonical Wasm {kind} value");
    match (kind, value) {
        (ValueType::F32, Val::F32(bits)) => Ok(Value::F32(f32::from_bits(*bits))),
        (ValueType::Unit, Val::I32(0)) => Ok(Value::Unit),
        (ValueType::Unit, Val::I32(_)) => Err(non_canonical()),
        (ValueType::Bool, Val::I32(value @ (0 | 1))) => Ok(Value::Bool(*value == 1)),
        (ValueType::Bool, Val::I32(_)) => Err(non_canonical()),
        (ValueType::U32, Val::I32(value)) => Ok(Value::U32(value.cast_unsigned())),
        (ValueType::ArrayU32 | ValueType::ArrayF32, Val::I32(pointer)) => {
            let arena = arena.ok_or_else(|| anyhow!("invalid_abi: array value requires an arena"))?;
            let mut words = std::mem::take(&mut store.as_context_mut().data_mut().spare);
            words.clear();
            // Guest execution may also grow memory; read it afresh.
            let memory = arena.memory.data(&store);
            let address = array_range(memory.len(), *pointer, 0)?;
            let length = u32::from_le_bytes(memory[address..address + 4].try_into().unwrap()) as usize;
            array_range(memory.len(), *pointer, length)?;
            let body = &memory[address + 4..address + 4 + length * 4];
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
        _ => bail!("invalid_abi: Wasm {kind} value has the wrong representation"),
    }
}

/// How long a guest of [`dev_engine`] may run without returning or calling
/// the host before it traps with an [`Overrun`]: an edit that loops costs a
/// few seconds. Its code checks the time at every function entry and loop
/// head, which on gdev retires 50 to 65% more instructions per frame (and
/// takes 30 to 50% more cycles), so no other engine does.
pub const DEADLINE: Duration = Duration::from_secs(5);

/// Epoch ticks per deadline: a guest traps within one tick before it.
const DEADLINE_TICKS: u32 = 50;

/// An engine for Blot guests: optimizing compiler with inlining (unless
/// BLOT_INLINING=0, see [`inlining`]) and a stack deep enough for recursive
/// programs. Run guests on threads whose stack exceeds [`GUEST_STACK_BYTES`]
/// plus the host's own needs. It compiles for this machine's exact CPU;
/// [`portable_engine`] compiles code to ship.
pub fn engine() -> Result<Engine> {
    Ok(Engine::new(&config(inlining(true)))?)
}

/// Whether Cranelift inlines calls, which makes gdev's frames about a
/// quarter faster for a slower compile. BLOT_INLINING=1 or 0 turns it on or
/// off for every engine, to compare; otherwise it is `default`: on for
/// [`engine`] and [`portable_engine`], off for [`dev_engine`].
fn inlining(default: bool) -> bool {
    match std::env::var("BLOT_INLINING").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => default,
    }
}

/// The same with and without the compiler: a precompiled module only loads
/// into an engine with the tunables (inlining among them) it was made with.
fn config(inlining: bool) -> wasmtime::Config {
    let mut config = wasmtime::Config::new();
    config.wasm_reference_types(true);
    config.wasm_simd(true);
    // Guests are core modules. pen's plugins turn wasmtime's component model
    // on in the builds that have them, which would otherwise enable it here
    // too, and a precompiled game records the features it was made with: a
    // player runtime (no plugins) would refuse one from a development build.
    config.wasm_features(wasmtime::WasmFeatures::COMPONENT_MODEL, false);
    config.max_wasm_stack(GUEST_STACK_BYTES);
    // Only checked against max_wasm_stack; guests run synchronously.
    config.async_stack_size(GUEST_STACK_BYTES + (1 << 20));
    config.compiler_inlining(if inlining { wasmtime::Inlining::Yes } else { wasmtime::Inlining::No });
    config
}

struct TickerState {
    tick_ns: u64,
}

struct TickerEntry {
    weak: wasmtime::EngineWeak,
    shared: Arc<(Mutex<TickerState>, Condvar)>,
}

static ENGINE_TICKERS: Mutex<Vec<TickerEntry>> = Mutex::new(Vec::new());

/// Guarantees a single deduplicated, wakeable epoch-ticker thread is running for
/// `engine` at `tick` (or faster if another caller already tightened it).
///
/// Tightening `tick` immediately wakes the sleeping ticker via `Condvar::notify_one()`
/// rather than waiting out the prior interval. Returns the effective tick duration.
/// Compiled unconditionally (works in compiler-free player runtimes).
pub fn ensure_engine_ticker(engine: &Engine, tick: Duration) -> Result<Duration> {
    let requested_ns = u64::try_from(tick.as_nanos().max(100_000)).unwrap_or(u64::MAX);
    let mut registry = ENGINE_TICKERS.lock().unwrap();
    registry.retain(|entry| entry.weak.upgrade().is_some());
    for entry in registry.iter() {
        if let Some(existing) = entry.weak.upgrade()
            && Engine::same(&existing, engine)
        {
            let (lock, cvar) = &*entry.shared;
            let mut state = lock.lock().unwrap();
            if requested_ns < state.tick_ns {
                state.tick_ns = requested_ns;
                cvar.notify_one();
            }
            return Ok(Duration::from_nanos(state.tick_ns));
        }
    }
    let shared = Arc::new((Mutex::new(TickerState { tick_ns: requested_ns }), Condvar::new()));
    let weak = engine.weak();
    let thread_weak = weak.clone();
    let thread_shared = shared.clone();
    std::thread::Builder::new().name("blot-deadline".into()).spawn(move || {
        loop {
            let Some(engine) = thread_weak.upgrade() else {
                break;
            };
            engine.increment_epoch();
            drop(engine);
            let (lock, cvar) = &*thread_shared;
            let mut state = lock.lock().unwrap();
            let slept_from = Instant::now();
            loop {
                let target = Duration::from_nanos(state.tick_ns);
                let Some(remaining) = target.checked_sub(slept_from.elapsed()) else {
                    break;
                };
                if remaining.is_zero() {
                    break;
                }
                let (next, result) = cvar.wait_timeout(state, remaining).unwrap();
                state = next;
                if result.timed_out() {
                    break;
                }
            }
        }
    })?;
    registry.push(TickerEntry { weak, shared });
    Ok(Duration::from_nanos(requested_ns))
}

/// An epoch-interrupted engine suitable for deadline-bounded workers and processes,
/// available both with and without the `compiler` feature (for deserializing
/// `workers.cwasm` in shipping players).
pub fn bounded_engine(tick: Duration) -> Result<Engine> {
    let mut cfg = config(inlining(true));
    cfg.epoch_interruption(true);
    let engine = Engine::new(&cfg)?;
    ensure_engine_ticker(&engine, tick)?;
    Ok(engine)
}

/// A fuel-metered AND epoch-interrupted engine for processes requiring both
/// deterministic fuel caps and wall-clock deadline preemption.
pub fn fuel_bounded_engine(tick: Duration) -> Result<Engine> {
    let mut cfg = config(inlining(true));
    cfg.consume_fuel(true);
    cfg.epoch_interruption(true);
    let engine = Engine::new(&cfg)?;
    ensure_engine_ticker(&engine, tick)?;
    Ok(engine)
}

/// A fuel-metered engine (without epoch interruption) for deterministic fuel-capped processes.
pub fn fuel_engine() -> Result<Engine> {
    let mut cfg = config(inlining(true));
    cfg.consume_fuel(true);
    Ok(Engine::new(&cfg)?)
}

/// A portable shipping engine with epoch interruption enabled, used by
/// `pen build --worker-deadlines` to precompile `workers.cwasm`.
#[cfg(feature = "compiler")]
pub fn bounded_portable_engine(target: Option<&str>, cpu: Option<&str>, tick: Duration) -> Result<Engine> {
    let mut cfg = config(inlining(true));
    cfg.epoch_interruption(true);
    if cpu == Some("native") {
        ensure!(target.is_none(), "`--cpu native` compiles for this machine, not for {}", target.unwrap_or_default());
        let engine = Engine::new(&cfg)?;
        ensure_engine_ticker(&engine, tick)?;
        return Ok(engine);
    }
    let triple = target.map_or_else(host_triple, str::to_owned);
    cfg.target(&triple)?;
    let preset = cpu.or(triple.starts_with("x86_64").then_some(PORTABLE_X86_64));
    if let Some(preset) = preset {
        // SAFETY: this only enables CPU features (which loading checks the
        // player's CPU has); it leaves the float ABI alone.
        unsafe {
            cfg.cranelift_flag_enable(preset);
        }
    }
    let engine = Engine::new(&cfg)
        .map_err(anyhow::Error::from)
        .with_context(|| format!("compiling for {triple} ({})", preset.unwrap_or("baseline")))?;
    ensure_engine_ticker(&engine, tick)?;
    Ok(engine)
}

#[cfg(not(feature = "compiler"))]
pub fn bounded_portable_engine(_target: Option<&str>, _cpu: Option<&str>, _tick: Duration) -> Result<Engine> {
    bail!(NO_COMPILER)
}

/// `engine`, its epoch advanced [`DEADLINE_TICKS`] times per `deadline` by a
/// deduplicated wakeable ticker thread.
#[cfg(feature = "compiler")]
fn ticking(engine: Engine, deadline: Duration) -> Result<Engine> {
    ensure_engine_ticker(&engine, deadline / DEADLINE_TICKS)?;
    Ok(engine)
}

/// The CPU [`portable_engine`] compiles for on x86-64 unless told otherwise:
/// x86-64-v2 (SSE4.2, POPCNT, CMPXCHG16B), which x86-64 CPUs have had since
/// about 2009 and Windows 11 requires. It keeps `floor`, `ceil`, `trunc` and
/// `nearest` single instructions (SSE4.1) rather than calls into the host.
pub const PORTABLE_X86_64: &str = "x86-64-v2";

/// An engine that compiles code to ship: for `target` (a Rust target triple
/// such as `x86_64-pc-windows-gnu`; this machine's when None) and, rather
/// than this machine's exact CPU, for `cpu`: a Cranelift preset such as
/// `baseline`, `x86-64-v2`, `x86-64-v3` or `znver2`, [`PORTABLE_X86_64`] on
/// x86-64 when None, or `native` for this machine. Loading checks that the
/// player's CPU has every feature a module was compiled for.
#[cfg(feature = "compiler")]
pub fn portable_engine(target: Option<&str>, cpu: Option<&str>) -> Result<Engine> {
    let mut config = config(inlining(true));
    if cpu == Some("native") {
        ensure!(target.is_none(), "`--cpu native` compiles for this machine, not for {}", target.unwrap_or_default());
        return Ok(Engine::new(&config)?);
    }
    // Any target, even this machine's, starts from its architecture's
    // baseline instead of the features this CPU has.
    let triple = target.map_or_else(host_triple, str::to_owned);
    config.target(&triple)?;
    let preset = cpu.or(triple.starts_with("x86_64").then_some(PORTABLE_X86_64));
    if let Some(preset) = preset {
        // SAFETY: this only enables CPU features (which loading checks the
        // player's CPU has); it leaves the float ABI alone.
        unsafe {
            config.cranelift_flag_enable(preset);
        }
    }
    Engine::new(&config)
        .map_err(anyhow::Error::from)
        .with_context(|| format!("compiling for {triple} ({})", preset.unwrap_or("baseline")))
}

#[cfg(not(feature = "compiler"))]
pub fn portable_engine(_target: Option<&str>, _cpu: Option<&str>) -> Result<Engine> {
    bail!(NO_COMPILER)
}

#[cfg(feature = "compiler")]
fn host_triple() -> String {
    let system = match std::env::consts::OS {
        "macos" => "apple-darwin",
        "windows" if cfg!(target_env = "gnu") => "pc-windows-gnu",
        "windows" => "pc-windows-msvc",
        _ => "unknown-linux-gnu",
    };
    format!("{}-{system}", std::env::consts::ARCH)
}

/// The engine for development, where the same program is compiled again
/// after every edit: Cranelift keeps each function's machine code in memory
/// and reuses it when the function comes back unchanged, so a small edit
/// recompiles only what it touched. Without inlining (see [`inlining`]): on
/// gdev that halves each reload's compile (about 100 to 50 ms) and the first
/// one (about 360 to 105 ms), for guest code about a quarter slower. Its
/// guests have a [`DEADLINE`]. Every call returns the same engine, so the
/// game and the dev tools share its cache.
pub fn dev_engine() -> Result<Engine> {
    #[cfg(feature = "compiler")]
    {
        let mut engine = DEV_ENGINE.lock().unwrap();
        if engine.is_none() {
            let mut config = config(inlining(false));
            config.enable_incremental_compilation(Arc::new(MemoryCache::new(DEV_CACHE_BYTES)))?;
            config.epoch_interruption(true);
            // A profile's samples need whole stacks (and traps show them).
            config.wasm_backtrace_max_frames(std::num::NonZeroUsize::new(BACKTRACE_FRAMES));
            *engine = Some(ticking(Engine::new(&config)?, DEADLINE)?);
        }
        Ok(engine.clone().unwrap())
    }
    #[cfg(not(feature = "compiler"))]
    bail!(NO_COMPILER)
}

/// Whether `engine` is [`dev_engine`], whose guests check its epoch and so
/// can be sampled ([`Profile`]); asking never makes one.
pub fn samples(engine: &Engine) -> bool {
    #[cfg(feature = "compiler")]
    {
        DEV_ENGINE.lock().unwrap().as_ref().is_some_and(|dev| Engine::same(dev, engine))
    }
    #[cfg(not(feature = "compiler"))]
    {
        let _ = engine;
        false
    }
}

/// The most frames of a guest's stack [`dev_engine`] keeps for a trap's
/// backtrace and a profile's sample.
#[cfg(feature = "compiler")]
const BACKTRACE_FRAMES: usize = 512;

/// The one [`dev_engine`], once made.
#[cfg(feature = "compiler")]
static DEV_ENGINE: std::sync::Mutex<Option<Engine>> = std::sync::Mutex::new(None);

/// What one generation of [`dev_engine`]'s cache keeps of compiled functions.
/// Both generations together retain at most twice this many key/value bytes;
/// allocator overhead and the compiler's active scratch are separate.
#[cfg(feature = "compiler")]
const DEV_CACHE_BYTES: usize = 32 << 20;

/// Compiled functions by Cranelift's cache key, for [`dev_engine`], in two
/// generations: a function found in the older moves to the newer, and once
/// the newer holds `limit` bytes the older is dropped and the newer takes
/// its place. What the edits keep compiling stays; the rest goes.
#[cfg(feature = "compiler")]
#[derive(Debug, Default)]
struct MemoryCache(std::sync::Mutex<Generations>);

#[cfg(feature = "compiler")]
#[derive(Debug, Default)]
struct Generations {
    newer: std::collections::HashMap<Vec<u8>, Vec<u8>>,
    older: std::collections::HashMap<Vec<u8>, Vec<u8>>,
    /// The newer generation's keys and values.
    bytes: usize,
    limit: usize,
}

#[cfg(feature = "compiler")]
impl MemoryCache {
    fn new(limit: usize) -> MemoryCache {
        MemoryCache(std::sync::Mutex::new(Generations { limit, ..Generations::default() }))
    }
}

#[cfg(feature = "compiler")]
impl Generations {
    fn insert(&mut self, key: Vec<u8>, value: Vec<u8>) -> bool {
        let Some(size) = key.len().checked_add(value.len()) else {
            return false;
        };
        if self.limit == 0 || size > self.limit {
            return false;
        }
        let replaced = self.newer.get(&key).map_or(0, |value| key.len() + value.len());
        if self.bytes - replaced > self.limit - size {
            self.older = std::mem::take(&mut self.newer);
            self.bytes = 0;
        }
        self.older.remove(&key);
        let (key_bytes, value_bytes) = (key.len(), value.len());
        if let Some(replaced) = self.newer.insert(key, value) {
            self.bytes -= key_bytes + replaced.len();
        }
        self.bytes += key_bytes + value_bytes;
        true
    }
}

#[cfg(feature = "compiler")]
impl wasmtime::CacheStore for MemoryCache {
    fn get(&self, key: &[u8]) -> Option<std::borrow::Cow<'_, [u8]>> {
        let mut cache = self.0.lock().unwrap();
        if let Some(value) = cache.newer.get(key) {
            return Some(std::borrow::Cow::Owned(value.clone()));
        }
        let value = cache.older.remove(key)?;
        cache.insert(key.to_vec(), value.clone());
        Some(std::borrow::Cow::Owned(value))
    }

    fn insert(&self, key: &[u8], value: Vec<u8>) -> bool {
        self.0.lock().unwrap().insert(key.to_vec(), value)
    }
}

/// Wasm stack available to a guest.
pub const GUEST_STACK_BYTES: usize = 8 << 20;

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "compiler"))]
    #[test]
    fn runtime_rejects_wasm_before_parsing_it() {
        let error = GuestModule::new(&engine().unwrap(), b"not Wasm", None).err().unwrap();
        assert!(error.to_string().contains("without a Wasm compiler"), "{error:#}");
    }

    /// A Wasm section: its id, then its contents' size.
    #[cfg(feature = "compiler")]
    fn section(id: u8, contents: &[u8]) -> Vec<u8> {
        assert!(contents.len() < 128, "one-byte LEB128 sizes only");
        [&[id, u8::try_from(contents.len()).unwrap()], contents].concat()
    }

    /// A Wasm name: its length, then its bytes.
    #[cfg(feature = "compiler")]
    fn name(text: &str) -> Vec<u8> {
        [&[u8::try_from(text.len()).unwrap()], text.as_bytes()].concat()
    }

    /// A guest exporting `spin: Unit -> Unit`, which never returns, and
    /// `main: (U32 -> U32 ! {Foreign}) -> U32`, which calls the host three
    /// times and then never returns either.
    #[cfg(feature = "compiler")]
    fn looping_guest() -> Vec<u8> {
        let abi =
            [&[0x02, 0x02][..], &name("spin"), &[0x00, 0x00], &name("main"), &[0x04, 0x01, 0x01, 0x01, 0x00]].concat();
        let forever = [0x03, 0x40, 0x0c, 0x00, 0x0b, 0x41, 0x00, 0x0b];
        let call_host = [0x20, 0x00, 0x41, 0x01, 0x10, 0x00, 0x1a];
        let spin = [&[0x00][..], &forever].concat();
        let main = [&[0x00][..], &call_host, &call_host, &call_host, &forever].concat();
        [
            &b"\0asm\x01\0\0\0"[..],
            &section(0, &[&name(pen_runtime::abi::ABI_SECTION)[..], &abi].concat()),
            // (externref, i32) -> i32, i32 -> i32, externref -> i32
            &section(
                1,
                &[0x03, 0x60, 0x02, 0x6f, 0x7f, 0x01, 0x7f, 0x60, 0x01, 0x7f, 0x01, 0x7f, 0x60, 0x01, 0x6f, 0x01, 0x7f],
            ),
            &section(2, &[&[0x01][..], &name(HOST_MODULE), &name("call_u32_u32"), &[0x00, 0x00]].concat()),
            &section(3, &[0x02, 0x01, 0x02]),
            &section(7, &[&[0x02][..], &name("spin"), &[0x00, 0x01], &name("main"), &[0x00, 0x02]].concat()),
            &section(
                10,
                &[&[0x02, u8::try_from(spin.len()).unwrap()][..], &spin, &[u8::try_from(main.len()).unwrap()], &main]
                    .concat(),
            ),
        ]
        .concat()
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn deadlines_stop_looping_guests_but_not_waiting_ones() {
        let deadline = Duration::from_millis(100);
        let mut config = config(false);
        config.epoch_interruption(true);
        let engine = ticking(Engine::new(&config).unwrap(), deadline).unwrap();
        let module = GuestModule::new(&engine, &looping_guest(), None).unwrap();
        let mut guest = Guest::new(&engine, &module).unwrap();
        guest.set_deadline(deadline);
        let started = Instant::now();
        let error = guest.call("spin", Value::Unit).unwrap_err();
        let overrun = error.downcast_ref::<Overrun>().expect("an overrun");
        assert!(overrun.ran >= deadline * 9 / 10 && started.elapsed() < deadline * 10, "{error:#}");
        // The host taking longer than the deadline to answer does not count.
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let callback = Box::new(move |value: Value| -> Result<Value> {
            counted.fetch_add(1, Ordering::Relaxed);
            std::thread::sleep(deadline * 2);
            Ok(value)
        });
        let error = guest.call("main", Argument::Callback(callback)).unwrap_err();
        assert!(error.is::<Overrun>(), "{error:#}");
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        // A stopped guest runs again with a deadline of its own.
        assert!(guest.call("spin", Value::Unit).unwrap_err().is::<Overrun>());
    }

    /// `looping_guest` with a counter: `main` adds one to its exported global
    /// `blot:rc_retains` before each of its host calls, and `spin` calls
    /// function 3, which never returns.
    #[cfg(feature = "compiler")]
    fn probed_guest() -> Vec<u8> {
        let abi =
            [&[0x02, 0x02][..], &name("spin"), &[0x00, 0x00], &name("main"), &[0x04, 0x01, 0x01, 0x01, 0x00]].concat();
        let forever = [0x03, 0x40, 0x0c, 0x00, 0x0b, 0x41, 0x00, 0x0b];
        // retains += 1, then call the host with 1 and drop its answer.
        let call_host = [0x23, 0x00, 0x41, 0x01, 0x6a, 0x24, 0x00, 0x20, 0x00, 0x41, 0x01, 0x10, 0x00, 0x1a];
        let spin = [0x00, 0x10, 0x03, 0x41, 0x00, 0x0b];
        let main = [&[0x00][..], &call_host, &call_host, &call_host, &forever].concat();
        let inner = [0x00, 0x03, 0x40, 0x0c, 0x00, 0x0b, 0x0b];
        let mut code = vec![0x03];
        for body in [&spin[..], &main, &inner] {
            code.push(u8::try_from(body.len()).unwrap());
            code.extend(body);
        }
        [
            &b"\0asm\x01\0\0\0"[..],
            &section(0, &[&name(pen_runtime::abi::ABI_SECTION)[..], &abi].concat()),
            // (externref, i32) -> i32, i32 -> i32, externref -> i32, () -> ()
            &section(
                1,
                &[
                    0x04, 0x60, 0x02, 0x6f, 0x7f, 0x01, 0x7f, 0x60, 0x01, 0x7f, 0x01, 0x7f, 0x60, 0x01, 0x6f, 0x01,
                    0x7f, 0x60, 0x00, 0x00,
                ],
            ),
            &section(2, &[&[0x01][..], &name(HOST_MODULE), &name("call_u32_u32"), &[0x00, 0x00]].concat()),
            &section(3, &[0x03, 0x01, 0x02, 0x03]),
            // A mutable i32, from 0.
            &section(6, &[0x01, 0x7f, 0x01, 0x41, 0x00, 0x0b]),
            &section(
                7,
                &[
                    &[0x03][..],
                    &name("spin"),
                    &[0x00, 0x01],
                    &name("main"),
                    &[0x00, 0x02],
                    &name("blot:rc_retains"),
                    &[0x03, 0x00],
                ]
                .concat(),
            ),
            &section(10, &code),
        ]
        .concat()
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn probes_count_calls_read_counters_and_sample_stacks() {
        let deadline = Duration::from_millis(100);
        let mut config = config(false);
        config.epoch_interruption(true);
        let engine = ticking(Engine::new(&config).unwrap(), deadline).unwrap();
        let module = GuestModule::new(&engine, &probed_guest(), None).unwrap();
        let mut guest = Guest::new(&engine, &module).unwrap();
        guest.set_deadline(deadline);
        let probe = Probe::new();
        guest.probe(probe.clone());
        assert_eq!(probe.seen().counters, [("rc_retains".to_owned(), 0)]);
        let callback = Box::new(|value: Value| -> Result<Value> { Ok(value) });
        assert!(guest.call("main", Argument::Callback(callback)).unwrap_err().is::<Overrun>());
        let seen = probe.seen();
        assert_eq!(seen.counters, [("rc_retains".to_owned(), 3)], "as of its last host call");
        assert_eq!((seen.calls, seen.failed, seen.callbacks), (1, 1, 3));
        assert!(seen.memory == 0 && seen.peak == 0, "it has no arena");
        // Sampled, `spin` is found in function 3 every time.
        let profile = Profile::new();
        profile.start(&engine).unwrap();
        probe.attach(Some(profile.clone()));
        let started = Instant::now();
        assert!(guest.call("spin", Value::Unit).unwrap_err().is::<Overrun>());
        assert!(started.elapsed() < deadline * 5, "the profile's ticks do not hasten its deadline either");
        profile.stop();
        let stacks = profile.stacks();
        assert!(profile.samples() >= 10, "{stacks:?}");
        assert!(stacks.iter().all(|(stack, _)| stack == &[1, 3]), "{stacks:?}");
        assert_eq!(profile.name(3), "function 3");
        profile.clear();
        assert_eq!(profile.samples(), 0);
        assert!(samples(&dev_engine().unwrap()) && !samples(&engine));
    }

    /// An ABI 3 guest exporting `ok: Unit -> U32`, which returns the
    /// constant `log`, and `fail: Unit -> Unit`, which traps. Its ownership
    /// hooks append to `log`: entering a 1, leaving 2 plus the outcome. With
    /// `hooks` false it leaves them out of its exports; with them, it also
    /// exports `log` as a runtime counter, `blot:cycle_passes`.
    #[cfg(feature = "compiler")]
    fn ownership_guest(hooks: bool) -> Vec<u8> {
        let abi = [
            &[0x03, 0x02][..],
            &name("ok"),
            &[0x00, 0x01],
            &name("fail"),
            &[0x00, 0x00],
            &[0x01],
            &name("log"),
            &[0x01],
        ]
        .concat();
        let mut exports = vec![[name("ok"), vec![0x00, 2]].concat(), [name("fail"), vec![0x00, 3]].concat()];
        exports.push([name("log"), vec![0x03, 0]].concat());
        // The log again as an ownership counter, to read after teardown.
        exports.push([name("blot:rc_destroyed"), vec![0x03, 0]].concat());
        if hooks {
            exports.push([name(OWNERSHIP_ENTER), vec![0x00, 0]].concat());
            exports.push([name(OWNERSHIP_LEAVE), vec![0x00, 1]].concat());
            // The same global again, as one of the runtime's counters.
            exports.push([name("blot:cycle_passes"), vec![0x03, 0]].concat());
        }
        // log = log * 10 + 1, and log = log * 10 + 2 + outcome.
        let enter = [0x00, 0x23, 0x00, 0x41, 10, 0x6c, 0x41, 1, 0x6a, 0x24, 0x00, 0x0b];
        let leave = [0x00, 0x23, 0x00, 0x41, 10, 0x6c, 0x41, 2, 0x6a, 0x20, 0x00, 0x6a, 0x24, 0x00, 0x0b];
        let ok = [0x00, 0x23, 0x00, 0x0b];
        let fail = [0x00, 0x00, 0x0b];
        let mut code = vec![0x04];
        for body in [&enter[..], &leave, &ok, &fail] {
            code.push(u8::try_from(body.len()).unwrap());
            code.extend(body);
        }
        [
            &b"\0asm\x01\0\0\0"[..],
            &section(0, &[&name(pen_runtime::abi::ABI_SECTION)[..], &abi].concat()),
            // () -> (), i32 -> (), i32 -> i32
            &section(1, &[0x03, 0x60, 0x00, 0x00, 0x60, 0x01, 0x7f, 0x00, 0x60, 0x01, 0x7f, 0x01, 0x7f]),
            &section(3, &[0x04, 0x00, 0x01, 0x02, 0x02]),
            // A mutable i32, from 0.
            &section(6, &[0x01, 0x7f, 0x01, 0x41, 0x00, 0x0b]),
            &section(7, &[&[u8::try_from(exports.len()).unwrap()][..], &exports.concat()].concat()),
            &section(10, &code),
        ]
        .concat()
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn abi_3_calls_run_between_entering_and_leaving_ownership() {
        let engine = engine().unwrap();
        let module = GuestModule::new(&engine, &ownership_guest(true), None).unwrap();
        assert!(module.abi().ownership);
        let mut guest = Guest::new(&engine, &module).unwrap();
        assert_eq!(guest.call("ok", Value::Unit).unwrap(), Value::U32(1), "entered before the call");
        assert_eq!(guest.read("log").unwrap(), Value::U32(12), "left normally after it");
        assert!(guest.call("fail", Value::Unit).is_err());
        assert_eq!(guest.read("log").unwrap(), Value::U32(1213), "left failing after a trap");
        assert_eq!(guest.call("ok", Value::Unit).unwrap(), Value::U32(12131), "and runs again");
        assert_eq!(guest.ownership().destroyed, Some(121312));
        let counters = [("blot:rc_destroyed".to_owned(), 121_312), ("blot:cycle_passes".to_owned(), 121_312)];
        assert_eq!(guest.runtime_counters(), counters);
        assert_eq!(guest.ownership().roots, None, "a counter the module does not export");
        guest.dispose().unwrap();
        assert_eq!(guest.ownership().destroyed, Some(1213124), "torn down once, idle");
        guest.dispose().unwrap();
        assert_eq!(guest.ownership().destroyed, Some(1213124), "and only once");
        let refused = format!("{:#}", guest.call("ok", Value::Unit).unwrap_err());
        assert!(refused.contains("disposed_guest"), "{refused}");
        assert!(guest.read("log").is_err());
        // An ABI 2 guest has no hooks to call.
        let module = GuestModule::new(&engine, &looping_guest(), None).unwrap();
        let mut guest = Guest::new(&engine, &module).unwrap();
        guest.dispose().unwrap();
        assert!(guest.call("spin", Value::Unit).is_err());
        let error = GuestModule::new(&engine, &ownership_guest(false), None).err().unwrap();
        assert!(format!("{error:#}").contains("missing ownership export"), "{error:#}");
    }

    /// A source map's base64 VLQ field.
    #[cfg(feature = "compiler")]
    fn vlq(value: i64) -> String {
        const DIGITS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut rest = if value < 0 { (-value << 1) | 1 } else { value << 1 };
        let mut text = String::new();
        loop {
            let digit = usize::try_from(rest & 0x1f).unwrap();
            rest >>= 5;
            text.push(char::from(DIGITS[digit | if rest > 0 { 0x20 } else { 0 }]));
            if rest == 0 {
                return text;
            }
        }
    }

    /// Standard base64.
    #[cfg(feature = "compiler")]
    fn base64(bytes: &[u8]) -> String {
        const DIGITS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut text = String::new();
        for chunk in bytes.chunks(3) {
            let word = chunk.iter().enumerate().fold(0u32, |word, (at, byte)| word | u32::from(*byte) << (16 - 8 * at));
            for at in 0..4 {
                let digit = usize::try_from((word >> (18 - 6 * at)) & 63).unwrap();
                text.push(if at <= chunk.len() { char::from(DIGITS[digit]) } else { '=' });
            }
        }
        text
    }

    /// A guest exporting `fail: Unit -> Unit`, which calls a function that
    /// traps, with a name section naming both and a source map placing the
    /// call at src/game.blot:3:1 and the trap at 7:3. Without `debug` it has
    /// neither.
    #[cfg(feature = "compiler")]
    fn trapping_guest(debug: bool) -> Vec<u8> {
        let abi = [&[0x02, 0x01][..], &name("fail"), &[0x00, 0x00, 0x00]].concat();
        let fail = [0x00, 0x10, 0x01, 0x41, 0x00, 0x0b];
        let explode = [0x00, 0x00, 0x0b];
        let code =
            [&[0x02, u8::try_from(fail.len()).unwrap()][..], &fail, &[u8::try_from(explode.len()).unwrap()], &explode]
                .concat();
        let before_code = [
            &b"\0asm\x01\0\0\0"[..],
            &section(0, &[&name(pen_runtime::abi::ABI_SECTION)[..], &abi].concat()),
            // (i32) -> i32, () -> ()
            &section(1, &[0x02, 0x60, 0x01, 0x7f, 0x01, 0x7f, 0x60, 0x00, 0x00]),
            &section(3, &[0x02, 0x00, 0x01]),
            &section(7, &[&[0x01][..], &name("fail"), &[0x00, 0x00]].concat()),
        ]
        .concat();
        let mut module = [before_code.clone(), section(10, &code)].concat();
        if debug {
            // The code section's contents start after its id and size.
            let call = before_code.len() + 2 + 2 + 1;
            let trap = before_code.len() + 2 + 2 + fail.len() + 1 + 1;
            let call = i64::try_from(call).unwrap();
            let trap = i64::try_from(trap).unwrap();
            let mappings = format!(
                "{}{}{}{},{}{}{}{}",
                vlq(call),
                vlq(0),
                vlq(2),
                vlq(0),
                vlq(trap - call),
                vlq(0),
                vlq(4),
                vlq(2)
            );
            let map = format!(r#"{{"version":3,"sources":["src/game.blot"],"names":[],"mappings":"{mappings}"}}"#);
            let url = format!("data:application/json;base64,{}", base64(map.as_bytes()));
            let functions = [&[0x02, 0x00][..], &name("game.fail"), &[0x01], &name("game.explode")].concat();
            let names = [&[0x01, u8::try_from(functions.len()).unwrap()][..], &functions].concat();
            let url_name = [&[u8::try_from(url.len()).unwrap(), 0x01][..], url.as_bytes()].concat();
            let source_map = [&name(pen_runtime::symbols::SOURCE_MAP_SECTION)[..], &url_name].concat();
            module.extend(section(0, &[&name(pen_runtime::symbols::NAME_SECTION)[..], &names].concat()));
            module.push(0);
            module.extend(leb(source_map.len()));
            module.extend(source_map);
        }
        module
    }

    /// An unsigned LEB128.
    #[cfg(feature = "compiler")]
    fn leb(mut value: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let byte = u8::try_from(value & 0x7f).unwrap();
            value >>= 7;
            if value == 0 {
                bytes.push(byte);
                return bytes;
            }
            bytes.push(byte | 0x80);
        }
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn traps_name_their_functions_and_source_lines() {
        let engine = Engine::new(&config(false)).unwrap();
        let module = GuestModule::new(&engine, &trapping_guest(true), None).unwrap();
        assert!(module.symbols().problems.is_empty(), "{:?}", module.symbols().problems);
        let mut guest = Guest::new(&engine, &module).unwrap();
        let error = guest.call("fail", Value::Unit).unwrap_err();
        assert_eq!(error.downcast_ref::<Trap>(), Some(&Trap::UnreachableCodeReached), "{error:#}");
        let text = format!("{error:#}");
        let explode = text.find("game.explode\n                at src/game.blot:7:3");
        let fail = text.find("game.fail\n                at src/game.blot:3:1");
        assert!(explode.is_some_and(|explode| fail.is_some_and(|fail| explode < fail)), "{text}");
        // Without debug information, frames go by index (and exports by name).
        let module = GuestModule::new(&engine, &trapping_guest(false), None).unwrap();
        let mut guest = Guest::new(&engine, &module).unwrap();
        let text = format!("{:#}", guest.call("fail", Value::Unit).unwrap_err());
        assert!(text.contains("<wasm function 1>") && text.contains(" - fail") && !text.contains(" at "), "{text}");
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn native_packages_preserve_traps_state_and_build_identity() {
        let engine = Engine::new(&config(false)).unwrap();
        let wasm = trapping_guest(true);
        let metadata = native_metadata(&wasm).unwrap();
        assert_eq!(native_fingerprint(&metadata).unwrap(), pen_runtime::net::wire::hash_bytes(&wasm));
        let code = precompile(&engine, &wasm).unwrap();
        // SAFETY: freshly precompiled with this engine from the same Wasm.
        let module = unsafe { GuestModule::from_native(&engine, &metadata, &code) }.unwrap();
        let mut guest = Guest::new(&engine, &module).unwrap();
        let error = format!("{:#}", guest.call("fail", Value::Unit).unwrap_err());
        assert!(error.contains("game.explode") && error.contains("src/game.blot:7:3"), "{error}");
        for hidden in [false, true] {
            let wasm = bumping_guest(hidden);
            let metadata = native_metadata(&wasm).unwrap();
            let code = precompile(&engine, &wasm).unwrap();
            // SAFETY: freshly precompiled from this metadata's Wasm.
            let module = unsafe { GuestModule::from_native(&engine, &metadata, &code) }.unwrap();
            let mut guest = Guest::new(&engine, &module).unwrap();
            assert_eq!(guest.call("bump", Value::U32(5)).unwrap(), Value::U32(5));
            if hidden {
                assert!(guest.capture(None).is_err());
            } else {
                let (saved, _) = guest.capture(None).unwrap();
                guest.call("bump", Value::U32(7)).unwrap();
                guest.restore(&saved).unwrap();
                assert_eq!(guest.call("bump", Value::U32(1)).unwrap(), Value::U32(6));
            }
            let mut invalid: serde_json::Value = serde_json::from_slice(&metadata).unwrap();
            invalid["version"] = 99.into();
            assert!(
                unsafe { GuestModule::from_native(&engine, &serde_json::to_vec(&invalid).unwrap(), &code) }.is_err()
            );
        }
    }

    /// A guest exporting `grow: U32 -> U32`, which grows its memory (one
    /// page to start) by that many pages and returns what `memory.grow`
    /// does.
    #[cfg(feature = "compiler")]
    fn growing_guest() -> Vec<u8> {
        let abi = [&[0x02, 0x01][..], &name("grow"), &[0x01, 0x01, 0x00]].concat();
        let grow = [0x00, 0x20, 0x00, 0x40, 0x00, 0x0b];
        [
            &b"\0asm\x01\0\0\0"[..],
            &section(0, &[&name(pen_runtime::abi::ABI_SECTION)[..], &abi].concat()),
            &section(1, &[0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]),
            &section(3, &[0x01, 0x00]),
            &section(5, &[0x01, 0x00, 0x01]),
            &section(7, &[&[0x01][..], &name("grow"), &[0x00, 0x00]].concat()),
            &section(10, &[&[0x01, u8::try_from(grow.len()).unwrap()][..], &grow].concat()),
        ]
        .concat()
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn sandboxes_keep_guests_to_their_memory() {
        let engine = engine().unwrap();
        let module = GuestModule::new(&engine, &growing_guest(), None).unwrap();
        let mut sandboxed = Guest::sandboxed(&engine, &module, Limits { memory: 3 << 16 }).unwrap();
        assert_eq!(sandboxed.call("grow", Value::U32(2)).unwrap(), Value::U32(1), "up to three pages");
        assert_eq!(sandboxed.call("grow", Value::U32(1)).unwrap(), Value::U32(u32::MAX), "and no more");
        let mut free = Guest::new(&engine, &module).unwrap();
        assert_eq!(free.call("grow", Value::U32(16)).unwrap(), Value::U32(1));
        assert!(Guest::sandboxed(&engine, &module, Limits { memory: 0 }).is_err(), "too little to start");
    }

    /// A guest exporting `bump: U32 -> U32`, which adds to a global and
    /// keeps the sum at byte 60 of its memory too. With `hidden`, it has a
    /// mutable global it does not export.
    #[cfg(feature = "compiler")]
    fn bumping_guest(hidden: bool) -> Vec<u8> {
        let abi = [&[0x02, 0x01][..], &name("bump"), &[0x01, 0x01, 0x00]].concat();
        let bump =
            [0x00, 0x23, 0x00, 0x20, 0x00, 0x6a, 0x24, 0x00, 0x41, 60, 0x23, 0x00, 0x36, 0x02, 0x00, 0x23, 0x00, 0x0b];
        let globals: &[u8] = if hidden {
            &[0x02, 0x7f, 0x01, 0x41, 0x00, 0x0b, 0x7f, 0x01, 0x41, 0x00, 0x0b]
        } else {
            &[0x01, 0x7f, 0x01, 0x41, 0x00, 0x0b]
        };
        let exports = [
            &[0x03][..],
            &name("bump"),
            &[0x00, 0x00],
            &name("blot:memory"),
            &[0x02, 0x00],
            &name("sum"),
            &[0x03, 0x00],
        ]
        .concat();
        [
            &b"\0asm\x01\0\0\0"[..],
            &section(0, &[&name(pen_runtime::abi::ABI_SECTION)[..], &abi].concat()),
            &section(1, &[0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]),
            &section(3, &[0x01, 0x00]),
            &section(5, &[0x01, 0x00, 0x01]),
            &section(6, globals),
            &section(7, &exports),
            &section(10, &[&[0x01, u8::try_from(bump.len()).unwrap()][..], &bump].concat()),
        ]
        .concat()
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn images_put_idle_guests_back() {
        let engine = engine().unwrap();
        let module = GuestModule::new(&engine, &bumping_guest(false), None).unwrap();
        let mut guest = Guest::new(&engine, &module).unwrap();
        assert_eq!(guest.call("bump", Value::U32(5)).unwrap(), Value::U32(5));
        let (five, captured) = guest.capture(None).unwrap();
        assert_eq!((captured.copied, five.memory.len()), (1, PAGE));
        assert_eq!(guest.call("bump", Value::U32(7)).unwrap(), Value::U32(12));
        let (twelve, captured) = guest.capture(Some(&five)).unwrap();
        assert_eq!((captured.copied, captured.shared), (1, 0), "the one page changed");
        let (again, captured) = guest.capture(Some(&twelve)).unwrap();
        assert_eq!((captured.copied, captured.shared, again.memory.unshared(&twelve.memory)), (0, 1, 0), "nothing did");
        assert_eq!(guest.restore(&five).unwrap(), 1);
        assert_eq!(guest.call("bump", Value::U32(1)).unwrap(), Value::U32(6), "the global came back");
        assert_eq!(guest.restore(&five).unwrap(), 1, "and the memory: byte 60 differed");
        // A guest whose state hides in a global the host cannot read.
        let module = GuestModule::new(&engine, &bumping_guest(true), None).unwrap();
        let mut guest = Guest::new(&engine, &module).unwrap();
        let refused = format!("{:#}", guest.capture(None).unwrap_err());
        assert!(refused.contains("1 of its 2 mutable globals are not exported"), "{refused}");
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn development_cache_keeps_what_is_used() {
        let cache = MemoryCache::new(10);
        let store: &dyn wasmtime::CacheStore = &cache;
        store.insert(b"a", vec![0; 4]);
        store.insert(b"b", vec![0; 4]);
        // Ten bytes: the next insert starts a new generation.
        store.insert(b"c", vec![0; 4]);
        assert!(store.get(b"a").is_some(), "found in the older generation, it moves to the newer");
        store.insert(b"d", vec![0; 4]);
        store.insert(b"e", vec![0; 4]);
        assert!(store.get(b"b").is_none(), "dropped with its generation");
        assert!(store.get(b"a").is_some() && store.get(b"e").is_some());
        let generations = cache.0.lock().unwrap();
        assert!(generations.bytes <= 10 && generations.newer.len() + generations.older.len() <= 4);
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn development_cache_rejects_oversized_entries_and_bounds_each_generation() {
        let cache = MemoryCache::new(10);
        let store: &dyn wasmtime::CacheStore = &cache;
        assert!(store.insert(b"a", vec![1; 9]));
        assert!(!store.insert(b"oversized", vec![2; 10]));
        assert_eq!(store.get(b"a").unwrap().as_ref(), &[1; 9]);
        assert!(store.get(b"oversized").is_none());
        for index in 0..128u8 {
            assert!(store.insert(&[index], vec![index; 6]));
            assert_eq!(store.get(&[index]).unwrap().as_ref(), &[index; 6]);
            let generations = cache.0.lock().unwrap();
            let size = |entries: &std::collections::HashMap<Vec<u8>, Vec<u8>>| {
                entries.iter().map(|(key, value)| key.len() + value.len()).sum::<usize>()
            };
            assert_eq!(generations.bytes, size(&generations.newer));
            assert!(generations.bytes <= 10 && size(&generations.older) <= 10);
        }
        assert!(store.get(b"a").is_none(), "eviction falls back to compilation");
        let disabled = MemoryCache::new(0);
        let disabled: &dyn wasmtime::CacheStore = &disabled;
        assert!(!disabled.insert(b"", vec![]));
        assert!(disabled.get(b"").is_none());
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn development_cache_replacement_uses_its_existing_space() {
        let cache = MemoryCache::new(10);
        let store: &dyn wasmtime::CacheStore = &cache;
        assert!(store.insert(b"a", vec![1; 4]));
        assert!(store.insert(b"b", vec![2; 4]));
        assert!(store.insert(b"a", vec![3; 4]));
        assert_eq!(store.get(b"a").unwrap().as_ref(), &[3; 4]);
        assert_eq!(store.get(b"b").unwrap().as_ref(), &[2; 4]);
        let generations = cache.0.lock().unwrap();
        assert!(generations.older.is_empty());
        assert_eq!(generations.bytes, 10);
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn module_cache_checks_complete_payload_and_exact_key_before_loading() {
        let key = "a".repeat(64);
        let payload = b"a serialized module is opaque to the envelope";
        let bytes = [cache_header(&key, payload), payload.to_vec()].concat();
        assert_eq!(checked_cache_bytes(&bytes, &key), Some(&payload[..]));
        assert!(checked_cache_bytes(&bytes, &"b".repeat(64)).is_none());
        for length in 0..bytes.len() {
            assert!(checked_cache_bytes(&bytes[..length], &key).is_none());
        }
        for index in 0..bytes.len() {
            let mut corrupt = bytes.clone();
            corrupt[index] ^= 1;
            assert!(checked_cache_bytes(&corrupt, &key).is_none());
        }
        let mut extended = bytes.clone();
        extended.push(0);
        assert!(checked_cache_bytes(&extended, &key).is_none());
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn module_cache_does_not_remove_another_writers_temporary_file() {
        let path = std::env::temp_dir().join(format!("pen-cache-collision-{}", std::process::id()));
        std::fs::write(&path, b"another writer's partial bytes").unwrap();
        let key = "a".repeat(64);
        assert_eq!(write_cache_file(&path, &key, b"payload").unwrap_err().kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&path).unwrap(), b"another writer's partial bytes");
        std::fs::remove_file(&path).unwrap();
        write_cache_file(&path, &key, b"payload").unwrap();
        assert_eq!(checked_cache_bytes(&std::fs::read(&path).unwrap(), &key), Some(&b"payload"[..]));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn module_cache_recovers_from_corruption_old_formats_and_wrong_artifacts() {
        static NEXT_TEST: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "pen-module-cache-{}-{}",
            std::process::id(),
            NEXT_TEST.fetch_add(1, Ordering::Relaxed)
        ));
        let engine = engine().unwrap();
        let wasm = looping_guest();
        let key = cache_key(&engine, &wasm);
        let path = directory.join(format!("{key}.cwasm"));
        GuestModule::new(&engine, &wasm, Some(&directory)).unwrap();
        let valid = std::fs::read(&path).unwrap();
        let payload = checked_cache_bytes(&valid, &key).unwrap();
        let mut corrupt = valid.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        let wrong = [cache_header(&"b".repeat(64), payload), payload.to_vec()].concat();
        for rejected in [vec![], payload.to_vec(), corrupt, wrong, valid[..valid.len() / 2].to_vec()] {
            std::fs::write(&path, rejected).unwrap();
            assert!(read_cached_module(&engine, &path, &key).is_none());
            GuestModule::new(&engine, &wasm, Some(&directory)).unwrap();
            assert!(read_cached_module(&engine, &path, &key).is_some());
        }
        let oversized = std::fs::File::options().write(true).open(&path).unwrap();
        oversized.set_len(u64::try_from(CACHE_HEADER_BYTES + CACHE_MODULE_BYTES + 1).unwrap()).unwrap();
        assert!(read_cached_module(&engine, &path, &key).is_none());
        drop(oversized);
        GuestModule::new(&engine, &wasm, Some(&directory)).unwrap();
        assert!(read_cached_module(&engine, &path, &key).is_some());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn module_cache_keys_and_pruning() {
        let engine = engine().unwrap();
        let wasm = looping_guest();
        let key = cache_key(&engine, &wasm);
        assert_eq!(key.len(), 64);
        assert_eq!(key, cache_key(&engine, &wasm));
        assert_ne!(key, cache_key(&dev_engine().unwrap(), &wasm), "the engine's configuration is in the key");
        assert_ne!(key, cache_key(&engine, &wasm[..wasm.len() - 1]));

        let directory = std::env::temp_dir().join(format!("blot-host-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        let now = std::time::SystemTime::now();
        let day = Duration::from_secs(24 * 60 * 60);
        let file = |name: &str, age: Duration| {
            let path = directory.join(name);
            std::fs::File::create(&path).unwrap().set_modified(now - age).unwrap();
            path
        };
        let stale = file("stale.cwasm", day * 15);
        let used = file("used.cwasm", day * 13);
        let abandoned = file("a.1234.tmp", day);
        let writing = file("b.1234.tmp", Duration::ZERO);
        let other = file("notes.txt", day * 30);
        let oversized = file("oversized.cwasm", Duration::ZERO);
        std::fs::File::options()
            .write(true)
            .open(&oversized)
            .unwrap()
            .set_len(u64::try_from(CACHE_MODULE_BYTES + CACHE_HEADER_BYTES + 1).unwrap())
            .unwrap();
        prune(&directory, now).unwrap();
        assert!(!stale.exists() && used.exists() && !abandoned.exists() && writing.exists() && other.exists());
        assert!(!oversized.exists(), "oversized legacy entries do not evade the disk bound");
        let recent: Vec<_> = (0..CACHE_ENTRIES)
            .map(|age| file(&format!("{age}.cwasm"), day * u32::try_from(age).unwrap() / 4))
            .collect();
        prune(&directory, now).unwrap();
        assert!(!used.exists(), "the oldest beyond the newest {CACHE_ENTRIES}");
        assert!(recent.iter().all(|path| path.exists()));

        // A compile writes its entry, and loads it the next time.
        let module = GuestModule::new(&engine, &wasm, Some(&directory)).unwrap();
        assert!(directory.join(format!("{key}.cwasm")).exists());
        GuestModule::new(&engine, &wasm, Some(&directory)).unwrap();
        assert_eq!(module.abi().functions.len(), 2);
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn inlining_modes_use_distinct_artifacts() {
        let wasm = looping_guest();
        for (producer, consumer) in [(true, false), (false, true)] {
            let writer = Engine::new(&config(producer)).unwrap();
            let reader = Engine::new(&config(consumer)).unwrap();
            assert_ne!(cache_key(&writer, &wasm), cache_key(&reader, &wasm));
            let artifact = writer.precompile_module(&wasm).unwrap();
            // SAFETY: the local writer compiled these bytes with this build of Wasmtime.
            unsafe { wasmtime::Module::deserialize(&writer, &artifact) }.unwrap();
            assert!(unsafe { wasmtime::Module::deserialize(&reader, &artifact) }.is_err());
        }
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn bounded_inlining_artifacts_require_matching_runtime_configuration() {
        let normal = Engine::new(&config(true)).unwrap();
        let mut bounded = config(true);
        // SAFETY: the pinned Wasmtime 49 setting only bounds inline expansion.
        unsafe {
            bounded.cranelift_flag_set("wasmtime_inlining_sum_size_threshold", "128");
        }
        let bounded = Engine::new(&bounded).unwrap();
        let compiled = wasmtime::Module::new(&bounded, looping_guest()).unwrap();
        let artifact = compiled.serialize().unwrap();
        // SAFETY: these bytes were just serialized by the same Wasmtime build.
        let mismatch = unsafe { wasmtime::Module::deserialize(&normal, &artifact) }.unwrap_err();
        assert!(mismatch.to_string().contains("function inlining sum-size threshold"));
        // SAFETY: the producer and consumer use the identical configuration.
        let loaded = unsafe { wasmtime::Module::deserialize(&bounded, &artifact) }.unwrap();
        assert_eq!(loaded.exports().count(), compiled.exports().count());
    }

    #[test]
    #[cfg(feature = "compiler")]
    fn allocator_can_invoke_active_callback_during_result_encoding() {
        // ABI 2 guest: `run: (U32 -> Array U32 ! {Foreign}) -> U32` whose `blot:allocate`
        // invokes the restored active host callback once while encoding an array result.
        let abi = [&[0x02, 0x01][..], &name("run"), &[0x04, 0x01, 0x05, 0x01, 0x00]].concat();
        // Types: 0=(externref,i32)->i32, 1=(externref)->i32, 2=(i32)->i32
        let types = [
            0x03, 0x60, 0x02, 0x6f, 0x7f, 0x01, 0x7f, 0x60, 0x01, 0x6f, 0x01, 0x7f, 0x60, 0x01, 0x7f, 0x01, 0x7f,
        ];
        let imports = [&[0x01][..], &name(HOST_MODULE), &name("call_u32_array_u32"), &[0x00, 0x00]].concat();
        let funcs = [0x03, 0x01, 0x02, 0x02];
        let memory = [0x01, 0x00, 0x01];
        // Globals: 0=mut externref (cap), 1=mut i32 (cursor, non-zero), 2=mut i32 (reentry guard)
        let globals = [
            0x03, 0x6f, 0x01, 0xd0, 0x6f, 0x0b, 0x7f, 0x01, 0x41, 0x10, 0x0b, 0x7f, 0x01, 0x41, 0x00, 0x0b,
        ];
        let exports = [
            &[0x04][..],
            &name("run"),
            &[0x00, 0x01],
            &name("blot:allocate"),
            &[0x00, 0x02],
            &name("blot:reset"),
            &[0x00, 0x03],
            &name("blot:memory"),
            &[0x02, 0x00],
        ]
        .concat();
        // fn 1 (run): local 1 i32; global.set 0(local 0); local 1 = call 0(local 0, 1); load(4) + load(8)
        let run_body = [
            0x01, 0x01, 0x7f, 0x20, 0x00, 0x24, 0x00, 0x20, 0x00, 0x41, 0x01, 0x10, 0x00, 0x21, 0x01, 0x20, 0x01,
            0x28, 0x02, 0x04, 0x20, 0x01, 0x28, 0x02, 0x08, 0x6a, 0x0b,
        ];
        // fn 2 (blot:allocate): local 1 i32; if !global2 { global2=1; drop(call 0(global0, 2)); } bump global1
        let alloc_body = [
            0x01, 0x01, 0x7f, 0x23, 0x02, 0x45, 0x04, 0x40, 0x41, 0x01, 0x24, 0x02, 0x23, 0x00, 0x41, 0x02, 0x10,
            0x00, 0x1a, 0x0b, 0x23, 0x01, 0x21, 0x01, 0x20, 0x01, 0x20, 0x00, 0x6a, 0x24, 0x01, 0x20, 0x01, 0x0b,
        ];
        // fn 3 (blot:reset): global1 = 16; global2 = 0; return 0
        let reset_body = [0x00, 0x41, 0x10, 0x24, 0x01, 0x41, 0x00, 0x24, 0x02, 0x41, 0x00, 0x0b];
        let mut code = vec![0x03];
        for body in [&run_body[..], &alloc_body[..], &reset_body[..]] {
            code.push(u8::try_from(body.len()).unwrap());
            code.extend(body);
        }
        let wasm = [
            &b"\0asm\x01\0\0\0"[..],
            &section(0, &[&name(pen_runtime::abi::ABI_SECTION)[..], &abi].concat()),
            &section(1, &types),
            &section(2, &imports),
            &section(3, &funcs),
            &section(5, &memory),
            &section(6, &globals),
            &section(7, &exports),
            &section(10, &code),
        ]
        .concat();

        let engine = engine().unwrap();
        let module = GuestModule::new(&engine, &wasm, None).unwrap();
        let mut guest = Guest::new(&engine, &module).unwrap();
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen_cb = seen.clone();
        let cb = Box::new(move |arg: Value| -> Result<Value> {
            let Value::U32(n) = arg else { bail!("expected U32") };
            seen_cb.lock().unwrap().push(n);
            Ok(if n == 1 { Value::ArrayU32(vec![10, 20]) } else { Value::ArrayU32(vec![7]) })
        });
        assert_eq!(guest.call("run", Argument::Callback(cb)).unwrap(), Value::U32(30));
        assert_eq!(*seen.lock().unwrap(), vec![1, 2]);
    }
}
