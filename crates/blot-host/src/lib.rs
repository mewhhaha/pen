//! Runs Blot guest programs on wasmtime, following Blot's guest ABI 2
//! (`blot2/compiler/guest-abi.md`): scalars and copied numeric arrays cross
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
mod probe;
mod system_trace;
mod tasks;
#[cfg(feature = "compiler")]
mod trace_instrument;
pub mod workers;
pub use system_trace::SystemTrace;

use std::any::Any;
use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use pen_runtime::memory_image::{Captured, MemoryImage, PAGE};
use pen_runtime::symbols::{Frame, Symbols};
use wasmtime::{
    AsContext, AsContextMut, Caller, Engine, ExternRef, FuncType, Global, Instance, Linker, Memory, Module, Store,
    Trap, TypedFunc, UpdateDeadline, Val, ValType, WasmBacktrace,
};

pub use probe::{Probe, Profile, Seen, function_name};

/// The host import module every capability adapter comes from.
pub const HOST_MODULE: &str = "blot:host/1";

// The manifest is pen-runtime's: the browser reads it too.
pub use pen_runtime::abi::{
    CallbackType, GuestAbi, GuestConstant, GuestFunction, Parameter, ValueType, parse_abi, read_abi,
};

/// The Wasm type a value crosses the boundary as.
fn wasm_type(value: ValueType) -> ValType {
    match value {
        ValueType::F32 => ValType::F32,
        _ => ValType::I32,
    }
}

/// A compiled guest module, its manifest and its debug information, ready
/// to instantiate many times.
#[derive(Clone)]
pub struct GuestModule {
    module: Module,
    trace: Option<Arc<SystemTrace>>,
    abi: Arc<GuestAbi>,
    symbols: Arc<Symbols>,
    /// The exported mutable globals an image of an instance keeps, or why
    /// one cannot ([`Guest::capture`]).
    imaged: Arc<Result<Vec<String>, String>>,
}

/// The mutable globals an image keeps: all of the module's, when it
/// exports them all, by their export names.
fn imaged_globals(wasm: &[u8]) -> Result<Vec<String>, String> {
    let mut mutable = Vec::new();
    let mut exported = std::collections::HashMap::new();
    let mut imported = 0;
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        match payload.map_err(|error| error.to_string())? {
            wasmparser::Payload::ImportSection(section) => {
                for import in section.into_imports() {
                    if let wasmparser::TypeRef::Global(_) = import.map_err(|error| error.to_string())?.ty {
                        imported += 1;
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
            _ => {}
        }
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

impl GuestModule {
    /// Compiles `wasm`, reusing a compiled copy from `cache` when one exists
    /// for these bytes and this engine's configuration. Fails without the
    /// `compiler` feature.
    pub fn new(engine: &Engine, wasm: &[u8], cache: Option<&Path>) -> Result<GuestModule> {
        let abi = Arc::new(read_abi(wasm)?);
        let module = compile(engine, wasm, cache)?;
        validate_module(&module, &abi)?;
        let imaged = Arc::new(imaged_globals(wasm));
        Ok(GuestModule { module, trace: None, abi, symbols: Arc::new(Symbols::read(wasm)), imaged })
    }

    /// Instruments allocation traffic and selects the game's explicit tracing
    /// entrypoint. The original bytes and normal entry remain unchanged.
    #[cfg(feature = "compiler")]
    pub fn traced(engine: &Engine, wasm: &[u8], cache: Option<&Path>, trace: Arc<SystemTrace>) -> Result<GuestModule> {
        let mut module = if trace.records_allocations() {
            Self::new(engine, &trace_instrument::instrument(wasm)?, cache)?
        } else {
            Self::new(engine, wasm, cache)?
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

    /// Like [`GuestModule::new`], but first tries `precompiled`, the output
    /// of [`precompile`] shipped beside the module. One made for another
    /// engine, or for CPU features this machine lacks, is compiled again, or
    /// refused without the `compiler` feature.
    ///
    /// # Safety
    ///
    /// `precompiled` runs as native code: it must come from [`precompile`],
    /// as trusted as the executable itself.
    pub unsafe fn with_precompiled(
        engine: &Engine,
        wasm: &[u8],
        precompiled: &[u8],
        cache: Option<&Path>,
    ) -> Result<GuestModule> {
        let abi = Arc::new(read_abi(wasm)?);
        // SAFETY: the caller vouches for the bytes; wasmtime checks they suit this engine.
        match unsafe { Module::deserialize(engine, precompiled) } {
            Ok(module) => {
                validate_module(&module, &abi)?;
                let imaged = Arc::new(imaged_globals(wasm));
                Ok(GuestModule { module, trace: None, abi, symbols: Arc::new(Symbols::read(wasm)), imaged })
            }
            Err(_) if COMPILER => GuestModule::new(engine, wasm, cache),
            Err(error) => Err(anyhow::Error::from(error).context(
                "the precompiled game does not suit this runtime (it was made by another pen \
                 version, or for CPU features this machine lacks), and this runtime has no \
                 compiler to compile it again",
            )),
        }
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

#[cfg(not(feature = "compiler"))]
fn compile(_engine: &Engine, _wasm: &[u8], _cache: Option<&Path>) -> Result<Module> {
    bail!(NO_COMPILER)
}

#[cfg(feature = "compiler")]
fn compile_cached(engine: &Engine, wasm: &[u8], directory: &Path) -> Result<Module> {
    let path = directory.join(format!("{}.cwasm", cache_key(engine, wasm)));
    if path.exists() {
        // SAFETY: the file is one this function wrote from these exact bytes
        // with a compatible engine; wasmtime still checks its header.
        match unsafe { Module::deserialize_file(engine, &path) } {
            Ok(module) => {
                // Its modification time says when it was last used, for `prune`.
                let _ = std::fs::File::options()
                    .write(true)
                    .open(&path)
                    .and_then(|file| file.set_modified(std::time::SystemTime::now()));
                return Ok(module);
            }
            Err(_) => {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
    let module = Module::new(engine, wasm)?;
    // A cache that cannot be written only costs the next launch a compile.
    let _ = (|| -> Result<()> {
        std::fs::create_dir_all(directory)?;
        let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&temporary, module.serialize()?)?;
        std::fs::rename(&temporary, &path)?;
        prune(directory, std::time::SystemTime::now())?;
        Ok(())
    })();
    Ok(module)
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
        let used = entry.metadata()?.modified()?;
        let unused = now.duration_since(used).unwrap_or_default();
        match path.extension().and_then(|extension| extension.to_str()) {
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
    /// Its deadline, when not [`DEADLINE`].
    deadline: Option<Duration>,
    /// What the dev tools see of it, and the counters it exports for them.
    probe: Option<Arc<Probe>>,
    counters: Vec<Global>,
    /// Its module's symbols, which name the functions a profile samples.
    symbols: Arc<Symbols>,
    /// The last array passed in, its buffer reused for the next array out.
    spare: Vec<u32>,
    /// Torn down ([`Guest::dispose`]): it takes no more calls.
    disposed: bool,
    /// What a sandboxed guest may use ([`Guest::sandboxed`]).
    limits: Option<wasmtime::StoreLimits>,
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
    let ticks = if sampled { 1 } else { ticks(store.data().deadline.unwrap_or(DEADLINE)) };
    store.set_epoch_deadline(ticks);
    store.data_mut().resumed = Some(Instant::now());
}

/// At the guest's epoch deadline: samples it while its probe samples, then
/// interrupts it once it has run past its deadline, which counts time
/// rather than ticks (a profile ticks the engine faster), else looks again
/// at the next tick (sampled) or when its deadline will be due.
fn deadline_reached(mut store: wasmtime::StoreContextMut<'_, State>) -> wasmtime::Result<UpdateDeadline> {
    let profile = store.data().probe.as_ref().and_then(|probe| probe.sampling());
    if let Some(profile) = &profile {
        profile.sample(&WasmBacktrace::capture(&store), &store.data().symbols);
    }
    let state = store.data_mut();
    let ran = state.resumed.map_or(Duration::ZERO, |resumed| resumed.elapsed());
    Ok(match state.deadline.unwrap_or(DEADLINE).checked_sub(ran).filter(|left| !left.is_zero()) {
        Some(_) if profile.is_some() => UpdateDeadline::Continue(1),
        Some(left) => UpdateDeadline::Continue(ticks(left)),
        None => UpdateDeadline::Interrupt,
    })
}

/// `error`, marked as an [`Overrun`] when the guest's deadline stopped it.
fn overran(state: &State, error: anyhow::Error) -> anyhow::Error {
    match (error.downcast_ref::<Trap>(), state.resumed) {
        (Some(Trap::Interrupt), Some(resumed)) => error.context(Overrun { ran: resumed.elapsed() }),
        _ => error,
    }
}

/// One instance of a guest module.
pub struct Guest {
    store: Store<State>,
    instance: Instance,
    abi: Arc<GuestAbi>,
    symbols: Arc<Symbols>,
    imaged: Arc<Result<Vec<String>, String>>,
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
        let mut store = Store::new(engine, State { symbols: module.symbols.clone(), ..State::default() });
        store.epoch_deadline_callback(deadline_reached);
        if let Some(limits) = limits {
            let limits = wasmtime::StoreLimitsBuilder::new().memory_size(limits.memory).instances(1).build();
            store.data_mut().limits = Some(limits);
            store.limiter(|state| state.limits.as_mut().expect("a sandbox's limits"));
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
            linker.instantiate(&mut store, &module.module).map_err(|error| overran(store.data(), error.into()))?;
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
        Ok(Guest {
            store,
            instance,
            abi: module.abi.clone(),
            symbols: module.symbols.clone(),
            imaged: module.imaged.clone(),
        })
    }

    pub fn abi(&self) -> &GuestAbi {
        &self.abi
    }

    /// Lets the guest run up to `deadline` at a time instead of [`DEADLINE`],
    /// on [`dev_engine`], to within a fiftieth of [`DEADLINE`].
    pub fn set_deadline(&mut self, deadline: Duration) {
        self.store.data_mut().deadline = Some(deadline);
    }

    /// Shows `probe` what the guest does from now on: its memory and the
    /// ownership counters it exports, as of each of its host calls, and the
    /// calls into it; and samples it into the profile attached to it.
    pub fn probe(&mut self, probe: Arc<Probe>) {
        let counters = self.counter_globals();
        probe.counters_named(counters.iter().map(|(name, _)| name.trim_start_matches("blot:").to_owned()).collect());
        let state = self.store.data_mut();
        state.counters = counters.into_iter().map(|(_, global)| global).collect();
        state.probe = Some(probe);
    }

    /// Committed linear-memory bytes, including temporary allocation capacity.
    pub fn memory_bytes(&self) -> usize {
        match &self.store.data().arena {
            Some(arena) => arena.memory.data_size(&self.store),
            None => 0,
        }
    }

    /// Keeps `gauge` at `memory_bytes`, updated whenever the guest calls the
    /// host, for other threads to read while it runs.
    pub fn track_memory(&mut self, gauge: Arc<AtomicUsize>) {
        gauge.store(self.memory_bytes(), Ordering::Relaxed);
        self.store.data_mut().memory = Some(gauge);
    }

    /// Calls the entrypoint `name`. A callback argument stays live only for
    /// this invocation, and the guest's arena is reset before and after it
    /// (and, under ABI 3, ownership entered before it and left after).
    pub fn call(&mut self, name: &str, argument: impl Into<Argument>) -> Result<Value> {
        ensure!(!self.store.data().disposed, "disposed_guest: the guest instance is disposed");
        let function =
            self.abi.function(name).ok_or_else(|| anyhow!("unknown_export: unknown function export {name}"))?.clone();
        let argument = argument.into();
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

    /// Ends a call: leaves ownership with `outcome` if it entered, then
    /// resets the arena, both attempted even when one fails.
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
        if let Some(arena) = self.store.data().arena.clone() {
            arena.reset.call(&mut self.store, 0)?;
        }
        task_result?;
        Ok(left?)
    }

    /// Tears the instance down, as Blot's adapter disposes of one: an ABI 3
    /// guest leaves ownership for good (outcome 2), releasing its module's
    /// roots and draining what it was destroying, and its arena is reset.
    /// Afterwards it takes no calls or reads, but its store stays to be
    /// looked at ([`Guest::ownership`], [`Guest::memory_bytes`]). Disposing
    /// again does nothing.
    pub fn dispose(&mut self) -> Result<()> {
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

    /// An image of the instance between calls: its memory (`blot:memory`)
    /// and its mutable globals, sharing the pages `previous` has unchanged.
    /// Restored ([`Guest::restore`]), the instance is as it was, for a
    /// module whose state is all there: one that exports every mutable
    /// global (others are refused). A long-lived `main(host)` is never
    /// between calls while it runs, so it cannot be imaged.
    pub fn capture(&mut self, previous: Option<&GuestImage>) -> Result<(GuestImage, Captured)> {
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

    /// Puts the instance back as `image` found it, growing its memory if
    /// it has less (memory never shrinks: what lies past the image is
    /// zeroed). Returns the pages written.
    pub fn restore(&mut self, image: &GuestImage) -> Result<usize> {
        ensure!(self.abi.tasks.is_none(), "task contexts are host state and cannot be memory-imaged");
        ensure!(!self.store.data().disposed, "disposed_guest: the guest instance is disposed");
        let mut written = 0;
        if let Some(memory) = self.instance.get_memory(&mut self.store, "blot:memory") {
            let missing = image.memory.len().saturating_sub(memory.data_size(&self.store));
            if missing > 0 {
                memory.grow(&mut self.store, u64::try_from(missing.div_ceil(PAGE))?)?;
            }
            written = image.memory.restore(memory.data_mut(&mut self.store));
        }
        for (name, value) in &image.globals {
            let global = self.instance.get_global(&mut self.store, name).context("an imaged global disappeared")?;
            global.set(&mut self.store, *value)?;
        }
        Ok(written)
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

    /// The guest runtime's own counters: its exported `blot:` 32-bit integer
    /// globals (under ABI 3 its reference counts' retains and releases, and
    /// its cycle collector's passes and the objects, edges and bytes they
    /// visited), by name, as unsigned numbers that wrap at 2^32. Empty for a
    /// guest that keeps none.
    pub fn runtime_counters(&mut self) -> Vec<(String, u32)> {
        let globals = self.counter_globals();
        globals.into_iter().filter_map(|(name, global)| Some((name, counter(global.get(&mut self.store))?))).collect()
    }

    /// The globals behind [`Guest::runtime_counters`], which
    /// [`Guest::ownership`] and a [`Probe`] read too: the guest's exported
    /// `blot:` globals that are not ABI constants, by name.
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
    if let (Some(gauge), Some(arena)) = (&caller.data().memory, &arena) {
        gauge.store(arena.memory.data_size(&caller), Ordering::Relaxed);
    }
    if let Some(probe) = caller.data().probe.clone() {
        let memory = arena.as_ref().map_or(0, |arena| arena.memory.data_size(&caller));
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
    // Guests are core modules. pen's plugins turn wasmtime's component model
    // on in the builds that have them, which would otherwise enable it here
    // too, and a precompiled game records the features it was made with: a
    // player runtime (no plugins) would refuse one from a development build.
    config.wasm_features(wasmtime::WasmFeatures::COMPONENT_MODEL, false);
    config.max_wasm_stack(GUEST_STACK_BYTES);
    // Only checked against max_wasm_stack; guests run synchronously.
    config.async_stack_size(GUEST_STACK_BYTES + (1 << 20));
    if inlining {
        config.compiler_inlining(wasmtime::Inlining::Yes);
    }
    config
}

/// `engine`, its epoch advanced [`DEADLINE_TICKS`] times per `deadline` by a
/// thread that ends with it, for an engine whose code checks it.
#[cfg(feature = "compiler")]
fn ticking(engine: Engine, deadline: Duration) -> Result<Engine> {
    let tick = deadline / DEADLINE_TICKS;
    let weak = engine.weak();
    std::thread::Builder::new().name("blot-deadline".into()).spawn(move || {
        while let Some(engine) = weak.upgrade() {
            engine.increment_epoch();
            drop(engine);
            std::thread::sleep(tick);
        }
    })?;
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

/// What [`dev_engine`]'s cache keeps of compiled functions, at most twice
/// over. Compiling gdev's game and dev tools puts about 3.7 MB in it.
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
    fn insert(&mut self, key: Vec<u8>, value: Vec<u8>) {
        if self.bytes >= self.limit {
            self.older = std::mem::take(&mut self.newer);
            self.bytes = 0;
        }
        let (key_bytes, value_bytes) = (key.len(), value.len());
        if let Some(replaced) = self.newer.insert(key, value) {
            self.bytes -= key_bytes + replaced.len();
        }
        self.bytes += key_bytes + value_bytes;
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
        self.0.lock().unwrap().insert(key.to_vec(), value);
        true
    }
}

/// Wasm stack available to a guest.
pub const GUEST_STACK_BYTES: usize = 8 << 20;

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(generations.bytes <= 10 + 5 && generations.newer.len() + generations.older.len() <= 4);
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
        prune(&directory, now).unwrap();
        assert!(!stale.exists() && used.exists() && !abandoned.exists() && writing.exists() && other.exists());
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
}
