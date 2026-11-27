//! Sandboxed execution of pure, zero-import Squid ABI 8 Wasm modules built by a
//! compiler plugin (e.g. Stage-0 generators), with host-shared compiled module
//! caching, bounded linear memory, epoch deadline enforcement, and `squid:traps`
//! v1 runtime provenance decoding.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use pen_runtime::abi::{self, GuestAbi, Parameter, TRAP_SITE_OFFSET, TrapKind, ValueType};
use sha2::{Digest, Sha256};
use wasmtime::{Engine, Instance, Module, Store, StoreLimits, StoreLimitsBuilder, Trap};

use crate::bindings::pen::plugin::guest::{CallError, TrapInfo};

/// Maximum linear memory a Stage-0 guest instance may grow to (256 MiB).
const GUEST_MEMORY: usize = 256 << 20;
/// Maximum table elements a Stage-0 guest instance may allocate.
const GUEST_TABLE_ELEMENTS: usize = 1 << 20;
/// Maximum Wasm call stack for a Stage-0 guest (matches `squid-host`).
const GUEST_STACK_BYTES: usize = 2 * 1024 * 1024;
/// Upper bound on host-cached compiled Stage-0 Wasm modules per `Plugins`.
const MAX_CACHED_MODULES: usize = 64;
/// Maximum U32 packet length allowed across the Stage-0 ABI boundary (32M words / 128 MiB).
const MAX_PACKET_WORDS: usize = 32 * 1024 * 1024;

struct CompiledGuest {
    module: Module,
    abi: GuestAbi,
}

struct StoreState {
    limits: StoreLimits,
}

/// Core Wasmtime engine + compiled module cache shared across all plugin
/// instances opened from the same [`crate::Plugins`].
pub struct GuestRuntime {
    engine: Engine,
    cache: Mutex<HashMap<[u8; 32], Arc<CompiledGuest>>>,
}

impl GuestRuntime {
    pub fn new() -> Result<Self> {
        let mut config = wasmtime::Config::new();
        config.wasm_reference_types(true);
        config.wasm_simd(true);
        config.wasm_features(wasmtime::WasmFeatures::COMPONENT_MODEL, false);
        config.epoch_interruption(true);
        config.max_wasm_stack(GUEST_STACK_BYTES);
        config.compiler_inlining(wasmtime::Inlining::No);
        let engine = Engine::new(&config)?;
        Ok(Self { engine, cache: Mutex::new(HashMap::new()) })
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    fn compiled(&self, wasm: &[u8]) -> std::result::Result<Arc<CompiledGuest>, CallError> {
        let digest: [u8; 32] = Sha256::digest(wasm).into();
        if let Some(cached) = self.cache.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).get(&digest).cloned() {
            return Ok(cached);
        }
        let abi = abi::read_abi(wasm).map_err(|error| CallError::Invalid(format!("{error:#}")))?;
        if !abi.data_stacks {
            return Err(CallError::Invalid("Stage-0 guest does not use Squid ABI 8".into()));
        }
        let module = Module::new(&self.engine, wasm)
            .map_err(|error| CallError::Invalid(format!("invalid Wasm module: {error:#}")))?;
        if let Some(import) = module.imports().next() {
            return Err(CallError::Invalid(format!(
                "Stage-0 guest requires Wasm host import `{}.{}`; Stage 0 grants no capabilities",
                import.module(),
                import.name()
            )));
        }
        let entry = Arc::new(CompiledGuest { module, abi });
        let mut cache = self.cache.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if cache.len() >= MAX_CACHED_MODULES {
            cache.clear();
        }
        cache.insert(digest, entry.clone());
        Ok(entry)
    }

    /// Runs `export_name(input)` on `wasm` as a pure Squid ABI 8
    /// `Array U32 -> Array U32` call in a fresh sandboxed instance.
    pub fn run_u32_packet(
        &self,
        wasm: &[u8],
        export_name: &str,
        input: &[u32],
        deadline: Option<Instant>,
    ) -> std::result::Result<Vec<u32>, CallError> {
        if input.len() > MAX_PACKET_WORDS {
            return Err(CallError::Invalid(format!(
                "Stage-0 input packet of {} words exceeds the {MAX_PACKET_WORDS}-word limit",
                input.len()
            )));
        }
        let compiled = self.compiled(wasm)?;
        let Some(spec) = compiled.abi.functions.iter().find(|func| func.name == export_name) else {
            return Err(CallError::Invalid(format!("Stage-0 guest has no ABI export `{export_name}`")));
        };
        if spec.parameter != Parameter::Value(ValueType::ArrayU32) || spec.result != ValueType::ArrayU32 {
            return Err(CallError::Invalid(format!(
                "Stage-0 export `{export_name}` has unsupported ABI signature (expected Array U32 -> Array U32)"
            )));
        }

        let limits = StoreLimitsBuilder::new().memory_size(GUEST_MEMORY).table_elements(GUEST_TABLE_ELEMENTS).build();
        let mut store = Store::new(&self.engine, StoreState { limits });
        store.limiter(|state| &mut state.limits);

        let timeout = match deadline {
            Some(end) => {
                let now = Instant::now();
                if now >= end {
                    return Err(CallError::Timeout);
                }
                end.saturating_duration_since(now)
            }
            None => crate::TIMEOUT,
        };
        let ticks = (timeout.as_millis() / crate::TICK.as_millis()).saturating_add(1);
        store.set_epoch_deadline(u64::try_from(ticks).unwrap_or(u64::MAX));

        let instance =
            Instance::new(&mut store, &compiled.module, &[]).map_err(|error| map_trap(&compiled.abi, None, error))?;
        let memory = instance
            .get_memory(&mut store, "squid:memory")
            .ok_or_else(|| CallError::Invalid("Stage-0 guest does not export `squid:memory`".into()))?;
        let allocate = instance
            .get_typed_func::<i32, i32>(&mut store, "squid:allocate")
            .map_err(|error| CallError::Invalid(format!("Stage-0 guest has invalid `squid:allocate`: {error:#}")))?;
        let reset = instance
            .get_typed_func::<i32, i32>(&mut store, "squid:reset")
            .map_err(|error| CallError::Invalid(format!("Stage-0 guest has invalid `squid:reset`: {error:#}")))?;
        let entry = instance.get_typed_func::<i32, i32>(&mut store, export_name).map_err(|error| {
            CallError::Invalid(format!("Stage-0 export `{export_name}` is not (i32) -> i32: {error:#}"))
        })?;

        let input_len =
            u32::try_from(input.len()).map_err(|_| CallError::Invalid("Stage-0 input exceeds U32 length".into()))?;
        let byte_size = input_len
            .checked_add(1)
            .and_then(|words| words.checked_mul(4))
            .ok_or_else(|| CallError::Invalid("Stage-0 input byte length overflows U32".into()))?;

        clear_trap_site(&mut store, memory);
        reset.call(&mut store, 0).map_err(|error| map_trap(&compiled.abi, Some(memory.data(&store)), error))?;
        let input_ptr = allocate
            .call(&mut store, byte_size.cast_signed())
            .map_err(|error| map_trap(&compiled.abi, Some(memory.data(&store)), error))?;

        {
            let data = memory.data_mut(&mut store);
            let address = array_range(data.len(), input_ptr, input.len()).map_err(CallError::Invalid)?;
            data[address..address + 4].copy_from_slice(&input_len.to_le_bytes());
            let body = &mut data[address + 4..address + 4 + input.len() * 4];
            for (chunk, word) in body.as_chunks_mut::<4>().0.iter_mut().zip(input) {
                *chunk = word.to_le_bytes();
            }
        }

        clear_trap_site(&mut store, memory);
        let output_ptr = entry
            .call(&mut store, input_ptr)
            .map_err(|error| map_trap(&compiled.abi, Some(memory.data(&store)), error))?;

        let data = memory.data(&store);
        let address = array_range(data.len(), output_ptr, 0).map_err(CallError::Invalid)?;
        let out_len = u32::from_le_bytes(data[address..address + 4].try_into().unwrap()) as usize;
        if out_len > MAX_PACKET_WORDS {
            return Err(CallError::Invalid(format!(
                "Stage-0 output packet of {out_len} words exceeds the {MAX_PACKET_WORDS}-word limit"
            )));
        }
        array_range(data.len(), output_ptr, out_len).map_err(CallError::Invalid)?;
        let body = &data[address + 4..address + 4 + out_len * 4];
        let mut out = Vec::with_capacity(out_len);
        for chunk in body.as_chunks::<4>().0 {
            out.push(u32::from_le_bytes(*chunk));
        }
        Ok(out)
    }
}

fn clear_trap_site(store: &mut Store<StoreState>, memory: wasmtime::Memory) {
    let data = memory.data_mut(store);
    if data.len() >= TRAP_SITE_OFFSET + 4 {
        data[TRAP_SITE_OFFSET..TRAP_SITE_OFFSET + 4].copy_from_slice(&0u32.to_le_bytes());
    }
}

fn array_range(memory_bytes: usize, pointer: i32, length: usize) -> std::result::Result<usize, String> {
    let address = pointer.cast_unsigned() as usize;
    if address == 0
        || !address.is_multiple_of(4)
        || address + 4 > memory_bytes
        || length > (memory_bytes - address - 4) / 4
    {
        return Err("invalid_abi: array lies outside the Stage-0 guest arena".into());
    }
    Ok(address)
}

fn map_trap(abi: &GuestAbi, memory: Option<&[u8]>, error: wasmtime::Error) -> CallError {
    if error.downcast_ref::<Trap>() == Some(&Trap::Interrupt) {
        return CallError::Timeout;
    }
    let trap_info = memory.and_then(|bytes| lookup_trap(abi, bytes));
    let message = match &trap_info {
        Some(info) => match info.kind.as_str() {
            "panic" if !info.message.is_empty() => format!("panic: {:?}", info.message),
            "panic" => "panic".into(),
            "bounds" => "index out of bounds".into(),
            "unreachable" => "unreachable".into(),
            other => other.to_owned(),
        },
        None => match error.downcast_ref::<Trap>() {
            Some(trap) => trap.to_string(),
            None => format!("{error:#}"),
        },
    };
    CallError::Trapped((message, trap_info))
}

fn lookup_trap(abi: &GuestAbi, memory: &[u8]) -> Option<TrapInfo> {
    if abi.traps.is_empty() || memory.len() < TRAP_SITE_OFFSET + 4 {
        return None;
    }
    let id = u32::from_le_bytes(memory[TRAP_SITE_OFFSET..TRAP_SITE_OFFSET + 4].try_into().unwrap());
    let index = id.checked_sub(1)? as usize;
    let site = abi.traps.get(index)?;
    let kind = match site.kind {
        TrapKind::Panic => "panic",
        TrapKind::Bounds => "bounds",
        TrapKind::Unreachable => "unreachable",
    };
    Some(TrapInfo {
        kind: kind.to_owned(),
        message: site.message.clone(),
        unit: site.unit,
        start: site.start,
        end: site.end,
    })
}

/// Helper used by `GuestRuntime` tests to run a tick loop with the same period.
#[cfg(test)]
pub(crate) fn tick_for_test(engine: &Engine) {
    let weak = engine.weak();
    let _ = std::thread::Builder::new().name("pen-plugins-guest-test-epoch".into()).spawn(move || {
        while let Some(engine) = weak.upgrade() {
            engine.increment_epoch();
            drop(engine);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_encoder::{
        CodeSection, CustomSection, ExportKind, ExportSection, Function, FunctionSection, ImportSection, Instruction,
        MemorySection, MemoryType, Module as WasmModule, TypeSection, ValType,
    };

    fn abi_section(export_name: &str, param_tag: u8, result_tag: u8) -> Vec<u8> {
        let mut out = vec![8, 1, u8::try_from(export_name.len()).unwrap()];
        out.extend_from_slice(export_name.as_bytes());
        out.push(param_tag);
        out.push(result_tag);
        out.push(0); // 0 constants
        out
    }

    fn traps_section(kind: u8, unit: u32, start: u32, end: u32, message: &str) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(1); // version 1
        out.extend_from_slice(&1u32.to_le_bytes()); // 1 site
        out.push(kind);
        out.extend_from_slice(&unit.to_le_bytes());
        out.extend_from_slice(&start.to_le_bytes());
        out.extend_from_slice(&end.to_le_bytes());
        out.extend_from_slice(&u32::try_from(message.len()).unwrap().to_le_bytes());
        out.extend_from_slice(message.as_bytes());
        out
    }

    fn build_echo_guest(with_import: bool, trap_id: Option<u32>, infinite_loop: bool) -> Vec<u8> {
        let mut module = WasmModule::new();
        let mut types = TypeSection::new();
        types.ty().function([ValType::I32], [ValType::I32]); // type 0: (i32) -> i32
        module.section(&types);

        if with_import {
            let mut imports = ImportSection::new();
            imports.import("env", "nope", wasm_encoder::EntityType::Function(0));
            module.section(&imports);
        }

        let mut funcs = FunctionSection::new();
        funcs.function(0); // squid:allocate
        funcs.function(0); // squid:reset
        funcs.function(0); // generate
        module.section(&funcs);

        let mut memories = MemorySection::new();
        memories.memory(MemoryType { minimum: 2, maximum: None, memory64: false, shared: false, page_size_log2: None });
        module.section(&memories);

        let base = u32::from(with_import);
        let mut exports = ExportSection::new();
        exports.export("squid:memory", ExportKind::Memory, 0);
        exports.export("squid:allocate", ExportKind::Func, base);
        exports.export("squid:reset", ExportKind::Func, base + 1);
        exports.export("generate", ExportKind::Func, base + 2);
        module.section(&exports);

        let mut code = CodeSection::new();
        // squid:allocate: return 4096
        let mut alloc_fn = Function::new([]);
        alloc_fn.instruction(&Instruction::I32Const(4096));
        alloc_fn.instruction(&Instruction::End);
        code.function(&alloc_fn);

        // squid:reset: return 0
        let mut reset_fn = Function::new([]);
        reset_fn.instruction(&Instruction::I32Const(0));
        reset_fn.instruction(&Instruction::End);
        code.function(&reset_fn);

        // generate: either loop forever, write trap_id & trap, or echo input pointer
        let mut gen_fn = Function::new([]);
        if infinite_loop {
            gen_fn.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty));
            gen_fn.instruction(&Instruction::Br(0));
            gen_fn.instruction(&Instruction::End);
        }
        if let Some(id) = trap_id {
            gen_fn.instruction(&Instruction::I32Const(i32::try_from(TRAP_SITE_OFFSET).unwrap()));
            gen_fn.instruction(&Instruction::I32Const(id.cast_signed()));
            gen_fn.instruction(&Instruction::I32Store(wasm_encoder::MemArg { offset: 0, align: 2, memory_index: 0 }));
            gen_fn.instruction(&Instruction::Unreachable);
        } else {
            gen_fn.instruction(&Instruction::LocalGet(0));
        }
        gen_fn.instruction(&Instruction::End);
        code.function(&gen_fn);
        module.section(&code);

        // ArrayU32 = tag 5
        let abi = abi_section("generate", 5, 5);
        module.section(&CustomSection { name: "squid:abi".into(), data: abi.into() });
        if trap_id.is_some() {
            let traps = traps_section(1, 0, 12, 34, "bad wgsl");
            module.section(&CustomSection { name: "squid:traps".into(), data: traps.into() });
        }
        module.finish()
    }

    #[test]
    fn runs_echo_packet_rejects_imports_decodes_traps_and_enforces_deadlines() {
        let runtime = GuestRuntime::new().unwrap();
        tick_for_test(runtime.engine());

        let echo = build_echo_guest(false, None, false);
        let output = runtime.run_u32_packet(&echo, "generate", &[10, 20, 30], None).unwrap();
        assert_eq!(output, [10, 20, 30]);
        // Cached second call succeeds too.
        let again = runtime.run_u32_packet(&echo, "generate", &[42], None).unwrap();
        assert_eq!(again, [42]);

        let importing = build_echo_guest(true, None, false);
        let err = runtime.run_u32_packet(&importing, "generate", &[], None).unwrap_err();
        assert!(matches!(err, CallError::Invalid(msg) if msg.contains("no capabilities")));

        let trapping = build_echo_guest(false, Some(1), false);
        let err = runtime.run_u32_packet(&trapping, "generate", &[], None).unwrap_err();
        match err {
            CallError::Trapped((msg, Some(info))) => {
                assert_eq!(msg, "panic: \"bad wgsl\"");
                assert_eq!(
                    (info.kind.as_str(), info.message.as_str(), info.unit, info.start, info.end),
                    ("panic", "bad wgsl", 0, 12, 34)
                );
            }
            other => panic!("expected decoded trap, got {other:?}"),
        }

        let looping = build_echo_guest(false, None, true);
        let err = runtime
            .run_u32_packet(&looping, "generate", &[], Some(Instant::now() + std::time::Duration::from_millis(80)))
            .unwrap_err();
        assert!(matches!(err, CallError::Timeout), "expected timeout, got {err:?}");
    }
}
