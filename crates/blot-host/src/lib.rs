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
//! blocked in a callback does not count.
use std::any::Any;
use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use wasmtime::{
    AsContext, AsContextMut, Caller, Engine, ExternRef, FuncType, Instance, Linker, Memory, Module, Store, Trap,
    TypedFunc, Val, ValType,
};

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

/// A compiled guest module and its manifest, ready to instantiate many times.
#[derive(Clone)]
pub struct GuestModule {
    module: Module,
    abi: Arc<GuestAbi>,
}

impl GuestModule {
    /// Compiles `wasm`, reusing a compiled copy from `cache` when one exists
    /// for these bytes and this engine's configuration. Fails without the
    /// `compiler` feature.
    pub fn new(engine: &Engine, wasm: &[u8], cache: Option<&Path>) -> Result<GuestModule> {
        let abi = Arc::new(read_abi(wasm)?);
        let module = compile(engine, wasm, cache)?;
        validate_module(&module, &abi)?;
        Ok(GuestModule { module, abi })
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
                Ok(GuestModule { module, abi })
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
    let adapters: Vec<String> = abi.callbacks().into_iter().map(CallbackType::import_name).collect();
    for import in module.imports() {
        ensure!(
            import.module() == HOST_MODULE && adapters.iter().any(|name| name == import.name()),
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

/// `blot:ownership_leave`'s outcomes. (2, teardown, releases the module's
/// roots before the instance is discarded; dropping the store does as much.)
const LEFT_NORMALLY: i32 = 0;
const LEFT_FAILING: i32 = 1;

#[derive(Clone)]
struct Ownership {
    enter: TypedFunc<(), ()>,
    leave: TypedFunc<i32, ()>,
}

#[derive(Default)]
struct State {
    active: Option<Active>,
    arena: Option<Arena>,
    ownership: Option<Ownership>,
    /// Whether this call entered ownership, so it must leave.
    entered: bool,
    /// Committed linear-memory bytes as of the guest's last host call.
    memory: Option<Arc<AtomicUsize>>,
    /// When the guest last started running (again), its deadline counting from then.
    resumed: Option<Instant>,
    /// Its deadline in epoch ticks, when not [`DEADLINE_TICKS`].
    deadline: Option<u64>,
    /// The last array passed in, its buffer reused for the next array out.
    spare: Vec<u32>,
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

/// Starts the guest's deadline afresh (for an engine that has one): it is
/// about to run (again), and only the time it runs counts.
fn resume(mut store: impl AsContextMut<Data = State>) {
    let mut store = store.as_context_mut();
    let ticks = store.data().deadline.unwrap_or(u64::from(DEADLINE_TICKS));
    store.set_epoch_deadline(ticks);
    store.data_mut().resumed = Some(Instant::now());
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
}

impl Guest {
    pub fn new(engine: &Engine, module: &GuestModule) -> Result<Guest> {
        let mut store = Store::new(engine, State::default());
        // Instantiating runs the module's start function, if it has one.
        resume(&mut store);
        let mut linker = Linker::new(engine);
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
        if module.abi.uses_arrays() {
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
        Ok(Guest { store, instance, abi: module.abi.clone() })
    }

    pub fn abi(&self) -> &GuestAbi {
        &self.abi
    }

    /// Lets the guest run up to `deadline` at a time instead of [`DEADLINE`],
    /// on [`dev_engine`], to within a fiftieth of [`DEADLINE`].
    pub fn set_deadline(&mut self, deadline: Duration) {
        let tick = DEADLINE / DEADLINE_TICKS;
        let ticks = u64::try_from(deadline.as_nanos() / tick.as_nanos()).unwrap_or(u64::MAX);
        self.store.data_mut().deadline = Some(ticks.max(1));
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
        let function =
            self.abi.function(name).ok_or_else(|| anyhow!("unknown_export: unknown function export {name}"))?.clone();
        let argument = argument.into();
        let result = self.begin_and_run(&function, argument).map_err(|error| overran(self.store.data(), error));
        let finish = self.finish(if result.is_ok() { LEFT_NORMALLY } else { LEFT_FAILING });
        let result = result?;
        finish?;
        Ok(result)
    }

    fn begin_and_run(&mut self, function: &GuestFunction, argument: Argument) -> Result<Value> {
        let name = &function.name;
        resume(&mut self.store);
        if let Some(ownership) = self.store.data().ownership.clone() {
            ownership.enter.call(&mut self.store, ())?;
            self.store.data_mut().entered = true;
        }
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
        func.call(&mut self.store, &[parameter], &mut results)?;
        let arena = self.store.data().arena.clone();
        from_wasm(&mut self.store, arena.as_ref(), function.result, &results[0])
    }

    /// Ends a call: leaves ownership with `outcome` if it entered, then
    /// resets the arena, both attempted even when one fails.
    fn finish(&mut self, outcome: i32) -> Result<()> {
        self.store.data_mut().active = None;
        resume(&mut self.store);
        let mut left = Ok(());
        if std::mem::take(&mut self.store.data_mut().entered)
            && let Some(ownership) = self.store.data().ownership.clone()
        {
            left = ownership.leave.call(&mut self.store, outcome);
        }
        if let Some(arena) = self.store.data().arena.clone() {
            arena.reset.call(&mut self.store, 0)?;
        }
        Ok(left?)
    }

    /// Reads an exported scalar constant.
    pub fn read(&mut self, name: &str) -> Result<Value> {
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
    if let (Some(gauge), Some(arena)) = (&caller.data().memory, &arena) {
        gauge.store(arena.memory.data_size(&caller), Ordering::Relaxed);
    }
    let argument = from_wasm(&mut caller, arena.as_ref(), signature.parameter, &params[1])?;
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
    results[0] = to_wasm(&mut caller, arena.as_ref(), result)?;
    Ok(())
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
        static ENGINE: std::sync::Mutex<Option<Engine>> = std::sync::Mutex::new(None);
        let mut engine = ENGINE.lock().unwrap();
        if engine.is_none() {
            let mut config = config(inlining(false));
            config.enable_incremental_compilation(Arc::new(MemoryCache::new(DEV_CACHE_BYTES)))?;
            config.epoch_interruption(true);
            *engine = Some(ticking(Engine::new(&config)?, DEADLINE)?);
        }
        Ok(engine.clone().unwrap())
    }
    #[cfg(not(feature = "compiler"))]
    bail!(NO_COMPILER)
}

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

    /// An ABI 3 guest exporting `ok: Unit -> U32`, which returns the
    /// constant `log`, and `fail: Unit -> Unit`, which traps. Its ownership
    /// hooks append to `log`: entering a 1, leaving 2 plus the outcome. With
    /// `hooks` false it leaves them out of its exports.
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
        if hooks {
            exports.push([name(OWNERSHIP_ENTER), vec![0x00, 0]].concat());
            exports.push([name(OWNERSHIP_LEAVE), vec![0x00, 1]].concat());
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
        let error = GuestModule::new(&engine, &ownership_guest(false), None).err().unwrap();
        assert!(format!("{error:#}").contains("missing ownership export"), "{error:#}");
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
