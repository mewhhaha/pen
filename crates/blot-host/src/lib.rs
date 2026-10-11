//! Runs Blot guest programs on wasmtime, following Blot's guest ABI 2
//! (`blot2/compiler/guest-abi.md`): scalars and copied numeric arrays cross
//! the boundary, and an entrypoint may take one host callback.
//!
//! The JavaScript adapter needs JSPI to suspend a guest inside a host
//! callback. Here a callback simply blocks its thread, which preserves the
//! Wasm stack the same way: run a long-lived `main(host)` on its own thread
//! and let the callback wait on a channel.
use std::any::Any;
use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use anyhow::{Context, Result, anyhow, bail, ensure};
use wasmtime::{
    AsContext, AsContextMut, Caller, Engine, ExternRef, FuncType, Instance, Linker, Memory,
    Module, Store, TypedFunc, Val, ValType,
};

/// The host import module every capability adapter comes from.
pub const HOST_MODULE: &str = "blot:host/1";
const ABI_SECTION: &str = "blot:abi";
const ABI_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueType {
    Unit,
    U32,
    Bool,
    F32,
    ArrayU32,
    ArrayF32,
}

impl ValueType {
    fn is_array(self) -> bool {
        matches!(self, ValueType::ArrayU32 | ValueType::ArrayF32)
    }
    fn is_scalar(self) -> bool {
        !self.is_array()
    }
    fn import_name(self) -> &'static str {
        match self {
            ValueType::Unit => "unit",
            ValueType::U32 => "u32",
            ValueType::Bool => "bool",
            ValueType::F32 => "f32",
            ValueType::ArrayU32 => "array_u32",
            ValueType::ArrayF32 => "array_f32",
        }
    }
    fn wasm(self) -> ValType {
        match self {
            ValueType::F32 => ValType::F32,
            _ => ValType::I32,
        }
    }
    fn from_tag(tag: u8) -> Result<ValueType> {
        Ok(match tag {
            0 => ValueType::Unit,
            1 => ValueType::U32,
            2 => ValueType::Bool,
            3 => ValueType::F32,
            5 => ValueType::ArrayU32,
            6 => ValueType::ArrayF32,
            _ => bail!("invalid_abi: unknown value type tag {tag}"),
        })
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ValueType::Unit => "Unit",
            ValueType::U32 => "U32",
            ValueType::Bool => "Bool",
            ValueType::F32 => "F32",
            ValueType::ArrayU32 => "Array U32",
            ValueType::ArrayF32 => "Array F32",
        })
    }
}

/// A host callback's signature: one value in, one value out, `! {Foreign}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CallbackType {
    pub parameter: ValueType,
    pub result: ValueType,
}

impl CallbackType {
    /// `call_<parameter>_<result>`, the adapter this signature imports.
    pub fn import_name(self) -> String {
        format!("call_{}_{}", self.parameter.import_name(), self.result.import_name())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parameter {
    Value(ValueType),
    Callback(CallbackType),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuestFunction {
    pub name: String,
    pub parameter: Parameter,
    pub result: ValueType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuestConstant {
    pub name: String,
    pub value: ValueType,
}

/// The `blot:abi` manifest: the module's exports, in declaration order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GuestAbi {
    pub functions: Vec<GuestFunction>,
    pub constants: Vec<GuestConstant>,
}

impl GuestAbi {
    pub fn function(&self, name: &str) -> Option<&GuestFunction> {
        self.functions.iter().find(|function| function.name == name)
    }

    /// Callback signatures, deduplicated in first-export order.
    pub fn callbacks(&self) -> Vec<CallbackType> {
        let mut found = Vec::new();
        for function in &self.functions {
            if let Parameter::Callback(signature) = function.parameter
                && !found.contains(&signature)
            {
                found.push(signature);
            }
        }
        found
    }

    fn uses_arrays(&self) -> bool {
        self.functions.iter().any(|function| {
            function.result.is_array()
                || match function.parameter {
                    Parameter::Value(value) => value.is_array(),
                    Parameter::Callback(signature) => {
                        signature.parameter.is_array() || signature.result.is_array()
                    }
                }
        })
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn byte(&mut self) -> Result<u8> {
        let byte = *self.bytes.get(self.at).ok_or_else(|| anyhow!("invalid_abi: truncated manifest"))?;
        self.at += 1;
        Ok(byte)
    }

    /// A canonical unsigned LEB128 u32.
    fn u32(&mut self) -> Result<u32> {
        let mut value = 0u64;
        for index in 0..5 {
            let byte = self.byte()?;
            value |= u64::from(byte & 0x7f) << (index * 7);
            if byte & 0x80 == 0 {
                ensure!(index == 0 || byte != 0, "invalid_abi: non-canonical LEB128");
                ensure!(value <= u64::from(u32::MAX), "invalid_abi: LEB128 exceeds u32");
                return Ok(value as u32);
            }
        }
        bail!("invalid_abi: LEB128 too long")
    }

    fn name(&mut self) -> Result<String> {
        let length = self.u32()? as usize;
        let end = self.at.checked_add(length).filter(|end| *end <= self.bytes.len());
        let end = end.ok_or_else(|| anyhow!("invalid_abi: truncated name"))?;
        let name = std::str::from_utf8(&self.bytes[self.at..end])
            .map_err(|_| anyhow!("invalid_abi: name is not UTF-8"))?;
        self.at = end;
        Ok(name.to_owned())
    }
}

/// Parses a `blot:abi` custom section's payload.
pub fn parse_abi(payload: &[u8]) -> Result<GuestAbi> {
    let mut reader = Reader { bytes: payload, at: 0 };
    let version = reader.u32()?;
    ensure!(version == ABI_VERSION, "invalid_abi: unsupported ABI version {version}");
    let mut abi = GuestAbi::default();
    let mut names = std::collections::HashSet::new();
    for _ in 0..reader.u32()? {
        let name = reader.name()?;
        let tag = reader.byte()?;
        let parameter = if tag == 4 {
            Parameter::Callback(CallbackType {
                parameter: ValueType::from_tag(reader.byte()?)?,
                result: ValueType::from_tag(reader.byte()?)?,
            })
        } else {
            Parameter::Value(ValueType::from_tag(tag)?)
        };
        let result = ValueType::from_tag(reader.byte()?)?;
        ensure!(names.insert(name.clone()), "invalid_abi: duplicate export {name}");
        abi.functions.push(GuestFunction { name, parameter, result });
    }
    for _ in 0..reader.u32()? {
        let name = reader.name()?;
        let value = ValueType::from_tag(reader.byte()?)?;
        ensure!(value.is_scalar(), "invalid_abi: constant {name} is not a scalar");
        ensure!(names.insert(name.clone()), "invalid_abi: duplicate export {name}");
        abi.constants.push(GuestConstant { name, value });
    }
    ensure!(reader.at == payload.len(), "invalid_abi: trailing manifest bytes");
    Ok(abi)
}

/// Finds the single `blot:abi` custom section in a module's bytes.
pub fn read_abi(wasm: &[u8]) -> Result<GuestAbi> {
    ensure!(wasm.len() >= 8 && wasm[..4] == *b"\0asm", "not a Wasm module");
    let mut reader = Reader { bytes: wasm, at: 8 };
    let mut found = None;
    while reader.at < wasm.len() {
        let id = reader.byte()?;
        let size = reader.u32()? as usize;
        let end = reader.at.checked_add(size).filter(|end| *end <= wasm.len());
        let end = end.ok_or_else(|| anyhow!("truncated Wasm section"))?;
        if id == 0 {
            let mut section = Reader { bytes: &wasm[..end], at: reader.at };
            if section.name()? == ABI_SECTION {
                ensure!(found.is_none(), "invalid_abi: more than one {ABI_SECTION} section");
                found = Some(parse_abi(&wasm[section.at..end])?);
            }
        }
        reader.at = end;
    }
    found.ok_or_else(|| anyhow!("invalid_abi: missing {ABI_SECTION} section"))
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
const NO_COMPILER: &str =
    "this runtime was built without a Wasm compiler (blot-host's `compiler` feature): it only loads \
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
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    wasm.hash(&mut hasher);
    engine.precompile_compatibility_hash().hash(&mut hasher);
    let path = directory.join(format!("{:016x}.cwasm", hasher.finish()));
    if path.exists() {
        // SAFETY: the file is one this function wrote from these exact bytes
        // with a compatible engine; wasmtime still checks its header.
        match unsafe { Module::deserialize_file(engine, &path) } {
            Ok(module) => return Ok(module),
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
        Ok(())
    })();
    Ok(module)
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

#[derive(Default)]
struct State {
    active: Option<Active>,
    arena: Option<Arena>,
    /// Committed linear-memory bytes as of the guest's last host call.
    memory: Option<Arc<AtomicUsize>>,
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
        let mut linker = Linker::new(engine);
        for signature in module.abi.callbacks() {
            let ty = FuncType::new(
                engine,
                [ValType::EXTERNREF, signature.parameter.wasm()],
                [signature.result.wasm()],
            );
            linker.func_new(HOST_MODULE, &signature.import_name(), ty, move |caller, params, results| {
                invoke_import(caller, signature, params, results).map_err(wasmtime::Error::from_anyhow)
            })?;
        }
        let instance = linker.instantiate(&mut store, &module.module)?;
        if module.abi.uses_arrays() {
            let memory = instance
                .get_memory(&mut store, "blot:memory")
                .ok_or_else(|| anyhow!("invalid_abi: missing array arena exports"))?;
            let allocate = instance.get_typed_func::<i32, i32>(&mut store, "blot:allocate")?;
            let reset = instance.get_typed_func::<i32, i32>(&mut store, "blot:reset")?;
            store.data_mut().arena = Some(Arena { memory, allocate, reset });
        }
        Ok(Guest { store, instance, abi: module.abi.clone() })
    }

    pub fn abi(&self) -> &GuestAbi {
        &self.abi
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
    /// this invocation, and the guest's arena is reset before and after it.
    pub fn call(&mut self, name: &str, argument: impl Into<Argument>) -> Result<Value> {
        let function = self
            .abi
            .function(name)
            .ok_or_else(|| anyhow!("unknown_export: unknown function export {name}"))?
            .clone();
        let argument = argument.into();
        let result = self.begin_and_run(&function, argument);
        let finish = self.finish();
        let result = result?;
        finish?;
        Ok(result)
    }

    fn begin_and_run(&mut self, function: &GuestFunction, argument: Argument) -> Result<Value> {
        let name = &function.name;
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
                self.store.data_mut().active = Some(Active {
                    token,
                    signature,
                    callback: Some(callback),
                    export: name.clone(),
                });
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
        from_wasm(&self.store, arena.as_ref(), function.result, &results[0])
    }

    fn finish(&mut self) -> Result<()> {
        self.store.data_mut().active = None;
        if let Some(arena) = self.store.data().arena.clone() {
            arena.reset.call(&mut self.store, 0)?;
        }
        Ok(())
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
        from_wasm(&self.store, None, constant.value, &value)
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
    ensure!(
        expected == signature,
        "capability_signature: host call signature does not match its capability"
    );
    let arena = caller.data().arena.clone();
    if let (Some(gauge), Some(arena)) = (&caller.data().memory, &arena) {
        gauge.store(arena.memory.data_size(&caller), Ordering::Relaxed);
    }
    let argument = from_wasm(&caller, arena.as_ref(), signature.parameter, &params[1])?;
    let mut callback = caller
        .data_mut()
        .active
        .as_mut()
        .and_then(|active| active.callback.take())
        .ok_or_else(|| anyhow!("reentrant_call: cannot reenter an active guest instance"))?;
    let result = callback(argument);
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
    let address = pointer as u32 as usize;
    ensure!(
        address >= 4
            && address.is_multiple_of(4)
            && address + 4 <= memory_bytes
            && length <= (memory_bytes - address - 4) / 4,
        "invalid_abi: array lies outside the guest arena"
    );
    Ok(address)
}

fn to_wasm(mut store: impl AsContextMut, arena: Option<&Arena>, value: Value) -> Result<Val> {
    let words: Vec<u32> = match value {
        Value::Unit => return Ok(Val::I32(0)),
        Value::U32(value) => return Ok(Val::I32(value as i32)),
        Value::Bool(value) => return Ok(Val::I32(value as i32)),
        Value::F32(value) => return Ok(Val::F32(value.to_bits())),
        Value::ArrayU32(values) => values,
        Value::ArrayF32(values) => values.into_iter().map(f32::to_bits).collect(),
    };
    let arena = arena.ok_or_else(|| anyhow!("invalid_abi: array value requires an arena"))?;
    ensure!(words.len() < 1_073_741_823, "array byte count exceeds the Wasm32 address range");
    let pointer = arena.allocate.call(&mut store, ((words.len() + 1) * 4) as i32)?;
    // Allocation may grow memory, so look at its size only afterwards.
    let memory = arena.memory.data_mut(&mut store);
    let address = array_range(memory.len(), pointer, words.len())?;
    memory[address..address + 4].copy_from_slice(&(words.len() as u32).to_le_bytes());
    let body = &mut memory[address + 4..address + 4 + words.len() * 4];
    for (chunk, word) in body.as_chunks_mut::<4>().0.iter_mut().zip(&words) {
        *chunk = word.to_le_bytes();
    }
    Ok(Val::I32(pointer))
}

fn from_wasm(store: impl AsContext, arena: Option<&Arena>, kind: ValueType, value: &Val) -> Result<Value> {
    let non_canonical = || anyhow!("invalid_abi: non-canonical Wasm {kind} value");
    match (kind, value) {
        (ValueType::F32, Val::F32(bits)) => Ok(Value::F32(f32::from_bits(*bits))),
        (ValueType::Unit, Val::I32(0)) => Ok(Value::Unit),
        (ValueType::Unit, Val::I32(_)) => Err(non_canonical()),
        (ValueType::Bool, Val::I32(value @ (0 | 1))) => Ok(Value::Bool(*value == 1)),
        (ValueType::Bool, Val::I32(_)) => Err(non_canonical()),
        (ValueType::U32, Val::I32(value)) => Ok(Value::U32(*value as u32)),
        (ValueType::ArrayU32 | ValueType::ArrayF32, Val::I32(pointer)) => {
            let arena = arena.ok_or_else(|| anyhow!("invalid_abi: array value requires an arena"))?;
            // Guest execution may also grow memory; read it afresh.
            let memory = arena.memory.data(&store);
            let address = array_range(memory.len(), *pointer, 0)?;
            let length = u32::from_le_bytes(memory[address..address + 4].try_into().unwrap()) as usize;
            array_range(memory.len(), *pointer, length)?;
            let words = memory[address + 4..address + 4 + length * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| u32::from_le_bytes(*chunk));
            Ok(if kind == ValueType::ArrayU32 {
                Value::ArrayU32(words.collect())
            } else {
                Value::ArrayF32(words.map(f32::from_bits).collect())
            })
        }
        _ => bail!("invalid_abi: Wasm {kind} value has the wrong representation"),
    }
}

/// An engine configured for Blot guests: optimizing compiler with inlining
/// and a stack deep enough for recursive programs. Run guests on threads whose stack
/// exceeds [`GUEST_STACK_BYTES`] plus the host's own needs. It compiles for
/// this machine's exact CPU; [`portable_engine`] compiles code to ship.
pub fn engine() -> Result<Engine> {
    Ok(Engine::new(&config())?)
}

/// The same with and without the compiler: a precompiled module only loads
/// into an engine with the tunables (inlining among them) it was made with.
/// Inlining makes gdev's frames about a quarter faster for a slower first
/// compile (cached afterwards). BLOT_INLINING=0 turns it off to compare.
fn config() -> wasmtime::Config {
    config_inlining(std::env::var_os("BLOT_INLINING").is_none_or(|value| value != "0"))
}

fn config_inlining(inlining: bool) -> wasmtime::Config {
    let mut config = wasmtime::Config::new();
    config.wasm_reference_types(true);
    config.max_wasm_stack(GUEST_STACK_BYTES);
    // Only checked against max_wasm_stack; guests run synchronously.
    config.async_stack_size(GUEST_STACK_BYTES + (1 << 20));
    if inlining {
        config.compiler_inlining(wasmtime::Inlining::Yes);
    }
    config
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
    let mut config = config();
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

/// An engine for development, where the same program is compiled again
/// after every edit: Cranelift keeps each function's machine code in memory
/// and reuses it when the function comes back unchanged, so a small edit
/// recompiles only what it touched. Without inlining, unless BLOT_INLINING=1:
/// on gdev that halves each reload's compile (about 100 to 50 ms) and the
/// first one (about 360 to 105 ms), for guest code about a quarter slower.
pub fn dev_engine() -> Result<Engine> {
    #[cfg(feature = "compiler")]
    {
        let mut config = config_inlining(std::env::var_os("BLOT_INLINING").is_some_and(|value| value == "1"));
        config.enable_incremental_compilation(Arc::new(MemoryCache::default()))?;
        Ok(Engine::new(&config)?)
    }
    #[cfg(not(feature = "compiler"))]
    bail!(NO_COMPILER)
}

/// Compiled functions by Cranelift's cache key, for [`dev_engine`].
#[cfg(feature = "compiler")]
#[derive(Debug, Default)]
struct MemoryCache(std::sync::Mutex<std::collections::HashMap<Vec<u8>, Vec<u8>>>);

#[cfg(feature = "compiler")]
impl wasmtime::CacheStore for MemoryCache {
    fn get(&self, key: &[u8]) -> Option<std::borrow::Cow<'_, [u8]>> {
        self.0.lock().unwrap().get(key).cloned().map(std::borrow::Cow::Owned)
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

    #[test]
    fn manifest_test_vectors() {
        assert_eq!(parse_abi(&[0x02, 0x00, 0x00]).unwrap(), GuestAbi::default());
        let abi = parse_abi(&[0x02, 0x01, 0x04, 0x6d, 0x61, 0x69, 0x6e, 0x04, 0x01, 0x01, 0x01, 0x00]).unwrap();
        assert_eq!(
            abi.functions,
            vec![GuestFunction {
                name: "main".into(),
                parameter: Parameter::Callback(CallbackType {
                    parameter: ValueType::U32,
                    result: ValueType::U32
                }),
                result: ValueType::U32,
            }]
        );
        assert!(parse_abi(&[0x02, 0x00, 0x00, 0x00]).is_err());
        assert!(parse_abi(&[0x03, 0x00, 0x00]).is_err());
    }
}
