#![cfg(feature = "compiler")]
//! Native `wtime-aot` execution tests for ABI 8 `KernelPool` and `Processes`
//! verifying zero Wasmtime fallback (`module.wasmtime_module().is_none()`).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Result;
use squid_host::{
    GuestModule, Limits,
    kernels::{KernelConfig, KernelError, KernelPool},
    processes::{
        AdmitResult, FuelPolicy, ProcessHostError, ProcessOutcome, ProcessPolicy, ProcessStatus, ProcessValue,
        Processes, SendResult,
    },
};

fn has_wtime_aot_binary() -> bool {
    if let Some(p) = std::env::var_os("PEN_WTIME_AOT_BIN")
        && !p.is_empty()
        && PathBuf::from(p).is_file()
    {
        return true;
    }
    if let Some(home) = std::env::var_os("HOME") {
        let cand = PathBuf::from(home).join(".local/lib/pen-runtime/wtime-aot");
        if cand.is_file() {
            return true;
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            if dir.join("wtime-aot").is_file() {
                return true;
            }
        }
    }
    false
}

fn temp_cache_dir(tag: &str) -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "squid-host-aot-conc-{tag}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create cache dir");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).expect("chmod 0700");
    }
    dir
}

fn leb_u32(mut v: u32) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let mut b = u8::try_from(v & 0x7f).unwrap();
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        out.push(b);
        if v == 0 {
            return out;
        }
    }
}

fn wasm_section(id: u8, body: &[u8]) -> Vec<u8> {
    let mut out = vec![id];
    out.extend(leb_u32(u32::try_from(body.len()).unwrap()));
    out.extend_from_slice(body);
    out
}

fn wasm_name(s: &str) -> Vec<u8> {
    let mut out = leb_u32(u32::try_from(s.len()).unwrap());
    out.extend_from_slice(s.as_bytes());
    out
}

fn wasm_custom(name: &str, payload: &[u8]) -> Vec<u8> {
    let mut body = wasm_name(name);
    body.extend_from_slice(payload);
    wasm_section(0, &body)
}

/// Crafts a self-contained ABI 8 module with a prepared-context fixed kernel `scale_add`
/// (`out[i] = in[i] * ctx[0] + ctx[1]`) and a variable-output kernel `dup_u32`.
fn crafted_aot_kernel_wasm() -> Vec<u8> {
    // ABI 8 exports: `scale_add`, `squid:kernel_ctx:scale_add`, `squid:kernel_clear:scale_add`, `dup_u32`
    let abi_payload = [
        vec![8, 4],
        wasm_name("scale_add"),
        vec![5, 5],
        wasm_name("squid:kernel_ctx:scale_add"),
        vec![5, 0],
        wasm_name("squid:kernel_clear:scale_add"),
        vec![0, 0],
        wasm_name("dup_u32"),
        vec![5, 5],
        vec![0],
    ]
    .concat();

    // squid:kernels v1
    let mut ksec = vec![1];
    ksec.extend(leb_u32(squid_host::DEFAULT_TRANSPORT_BUDGET));
    ksec.push(2); // 2 kernels

    // Kernel 0: scale_add (flags = 1: has_context, fixed output)
    ksec.extend(wasm_name("scale_add"));
    ksec.push(1);
    // in_schema: FlatFixed(1), ArrayU32(5), stride=1, word_tags=[U32(1)], hash
    ksec.extend([1, 5, 1, 1]);
    ksec.extend_from_slice(&0x101u64.to_le_bytes());
    // out_schema: FlatFixed(1), ArrayU32(5), stride=1, word_tags=[U32(1)], hash
    ksec.extend([1, 5, 1, 1]);
    ksec.extend_from_slice(&0x102u64.to_le_bytes());
    // io_hash, impl_hash, binding_hash
    ksec.extend_from_slice(&0x111u64.to_le_bytes());
    ksec.extend_from_slice(&0x112u64.to_le_bytes());
    ksec.extend_from_slice(&0x113u64.to_le_bytes());
    // ctx_schema (FlatFixed, ArrayU32, stride=2, [U32, U32]) + ctx_export + clear_export
    ksec.extend([1, 5, 2, 1, 1]);
    ksec.extend_from_slice(&0x103u64.to_le_bytes());
    ksec.extend(wasm_name("squid:kernel_ctx:scale_add"));
    ksec.extend(wasm_name("squid:kernel_clear:scale_add"));

    // Kernel 1: dup_u32 (flags = 2: variable_output)
    ksec.extend(wasm_name("dup_u32"));
    ksec.push(2);
    ksec.extend([1, 5, 1, 1]);
    ksec.extend_from_slice(&0x201u64.to_le_bytes());
    ksec.extend([2, 5, 0]); // VarU32(2), ArrayU32(5), stride=0
    ksec.extend_from_slice(&0x202u64.to_le_bytes());
    ksec.extend_from_slice(&0x211u64.to_le_bytes());
    ksec.extend_from_slice(&0x212u64.to_le_bytes());
    ksec.extend_from_slice(&0x213u64.to_le_bytes());

    // Types: 0: (i32)->(i32), 1: ()->(i32)
    let types = [2, 0x60, 1, 0x7f, 1, 0x7f, 0x60, 0, 1, 0x7f];
    // Funcs: 0=alloc, 1=reset, 2=finish, 3=ctx_set, 4=ctx_clear, 5=scale_add, 6=dup_u32
    let funcs = [7, 0, 0, 1, 0, 0, 0, 0];
    let mem = [1, 1, 1, 4]; // min=1, max=4
    // Globals: 0=ctx_mul (mut i32), 1=ctx_add (mut i32)
    let globals = [2, 0x7f, 1, 0x41, 0, 0x0b, 0x7f, 1, 0x41, 0, 0x0b];
    let exports = [
        vec![8],
        wasm_name("squid:allocate"),
        vec![0, 0],
        wasm_name("squid:reset"),
        vec![0, 1],
        wasm_name("squid:finish_invocation"),
        vec![0, 2],
        wasm_name("squid:kernel_ctx:scale_add"),
        vec![0, 3],
        wasm_name("squid:kernel_clear:scale_add"),
        vec![0, 4],
        wasm_name("scale_add"),
        vec![0, 5],
        wasm_name("dup_u32"),
        vec![0, 6],
        wasm_name("squid:memory"),
        vec![2, 0],
    ]
    .concat();

    // f0: allocate(_) -> 1024
    let mut f_alloc = vec![0, 0x41];
    f_alloc.extend(leb_u32(1024));
    f_alloc.push(0x0b);
    // f1: reset(_) -> 0
    let f_reset = vec![0, 0x41, 0, 0x0b];
    // f2: finish_invocation() -> 0
    let f_fin = vec![0, 0x41, 0, 0x0b];
    // f3: ctx_set(ptr) -> g0 = mem[ptr+4], g1 = mem[ptr+8], return 0
    let f_ctx = vec![0, 0x20, 0, 0x28, 2, 4, 0x24, 0, 0x20, 0, 0x28, 2, 8, 0x24, 1, 0x41, 0, 0x0b];
    // f4: ctx_clear(_) -> g0 = 0, g1 = 0, return 0
    let f_clear = vec![0, 0x41, 0, 0x24, 0, 0x41, 0, 0x24, 1, 0x41, 0, 0x0b];

    // f5: scale_add(in_ptr) -> out_ptr = 4096
    // local 1 = n, local 2 = i
    let mut f_scale = vec![1, 2, 0x7f];
    f_scale.extend([0x20, 0, 0x28, 2, 0, 0x21, 1]); // n = mem[in_ptr]
    f_scale.push(0x41);
    f_scale.extend(leb_u32(4096));
    f_scale.extend([0x20, 1, 0x36, 2, 0]); // mem[4096] = n
    f_scale.extend([0x02, 0x40, 0x03, 0x40]); // block; loop
    f_scale.extend([0x20, 2, 0x20, 1, 0x4f, 0x0d, 1]); // if i >=u n break
    // dst = 4100 + i*4
    f_scale.push(0x41);
    f_scale.extend(leb_u32(4100));
    f_scale.extend([0x20, 2, 0x41, 2, 0x74, 0x6a]);
    // val = mem[in_ptr + 4 + i*4] * g0 + g1
    f_scale.extend([0x20, 0, 0x20, 2, 0x41, 2, 0x74, 0x6a, 0x28, 2, 4, 0x23, 0, 0x6c, 0x23, 1, 0x6a]);
    f_scale.extend([0x36, 2, 0]); // store
    f_scale.extend([0x20, 2, 0x41, 1, 0x6a, 0x21, 2, 0x0c, 0, 0x0b, 0x0b]); // i++; br 0; end; end
    f_scale.push(0x41);
    f_scale.extend(leb_u32(4096));
    f_scale.push(0x0b);

    // f6: dup_u32(in_ptr) -> out_ptr = 4096: for each x in input, emits [x, x + 100] (2 words per item)
    let mut f_dup = vec![1, 3, 0x7f]; // 1=n, 2=i, 3=x
    f_dup.extend([0x20, 0, 0x28, 2, 0, 0x21, 1]);
    f_dup.push(0x41);
    f_dup.extend(leb_u32(4096));
    f_dup.extend([0x20, 1, 0x41, 1, 0x74, 0x36, 2, 0]); // mem[4096] = n * 2
    f_dup.extend([0x02, 0x40, 0x03, 0x40]);
    f_dup.extend([0x20, 2, 0x20, 1, 0x4f, 0x0d, 1]);
    f_dup.extend([0x20, 0, 0x20, 2, 0x41, 2, 0x74, 0x6a, 0x28, 2, 4, 0x21, 3]); // x
    // mem[4100 + i*8] = x
    f_dup.push(0x41);
    f_dup.extend(leb_u32(4100));
    f_dup.extend([0x20, 2, 0x41, 3, 0x74, 0x6a, 0x20, 3, 0x36, 2, 0]);
    // mem[4104 + i*8] = x + 100  (signed LEB128 for 100 is [0xe4, 0x00])
    f_dup.push(0x41);
    f_dup.extend(leb_u32(4104));
    f_dup.extend([0x20, 2, 0x41, 3, 0x74, 0x6a, 0x20, 3, 0x41, 0xe4, 0x00, 0x6a, 0x36, 2, 0]);
    f_dup.extend([0x20, 2, 0x41, 1, 0x6a, 0x21, 2, 0x0c, 0, 0x0b, 0x0b]);
    f_dup.push(0x41);
    f_dup.extend(leb_u32(4096));
    f_dup.push(0x0b);

    let mut code = vec![7];
    for body in [&f_alloc, &f_reset, &f_fin, &f_ctx, &f_clear, &f_scale, &f_dup] {
        code.extend(leb_u32(u32::try_from(body.len()).unwrap()));
        code.extend_from_slice(body);
    }

    [
        b"\0asm\x01\0\0\0".to_vec(),
        wasm_custom(pen_runtime::abi::ABI_SECTION, &abi_payload),
        wasm_custom(pen_runtime::abi::INVOCATION_SECTION, &[1]),
        wasm_custom(pen_runtime::abi::KERNELS_SECTION, &ksec),
        wasm_section(1, &types),
        wasm_section(3, &funcs),
        wasm_section(5, &mem),
        wasm_section(6, &globals),
        wasm_section(7, &exports),
        wasm_section(10, &code),
    ]
    .concat()
}

/// Crafts a self-contained ABI 8 module with:
/// - `f32_acc`: scalar `#process` (config: U32, cmd: F32, reply: F32) WITH an unexported
///   hidden 1-page linear memory (testing `f32` `call_raw` + hidden `memory_bytes() >= 65536`!).
/// - `vec_inc`: array `#process` (config: ArrayU32, cmd: ArrayU32, reply: ArrayU32)
///   when `with_arena == true`.
fn crafted_aot_process_wasm(with_arena: bool) -> Vec<u8> {
    let mut abi_payload = vec![8, if with_arena { 4 } else { 2 }];
    abi_payload.extend(wasm_name("squid:proc_init:f32_acc"));
    abi_payload.extend([1, 0]); // U32 -> Unit
    abi_payload.extend(wasm_name("squid:proc_step:f32_acc"));
    abi_payload.extend([3, 3]); // F32 -> F32
    if with_arena {
        abi_payload.extend(wasm_name("squid:proc_init:vec_inc"));
        abi_payload.extend([5, 0]); // ArrayU32 -> Unit
        abi_payload.extend(wasm_name("squid:proc_step:vec_inc"));
        abi_payload.extend([5, 5]); // ArrayU32 -> ArrayU32
    }
    abi_payload.push(0); // 0 constants

    let mut psec = vec![1];
    psec.extend(leb_u32(squid_host::DEFAULT_TRANSPORT_BUDGET));
    psec.push(if with_arena { 2 } else { 1 });

    // Process 0: f32_acc (config U32, cmd F32, reply F32)
    psec.extend(wasm_name("f32_acc"));
    psec.extend(wasm_name("squid:proc_init:f32_acc"));
    psec.extend(wasm_name("squid:proc_step:f32_acc"));
    psec.extend([1, 1, 1, 1]); // FlatFixed, U32, stride=1, [U32]
    psec.extend_from_slice(&0x301u64.to_le_bytes());
    psec.extend([1, 3, 1, 3]); // FlatFixed, F32, stride=1, [F32]
    psec.extend_from_slice(&0x302u64.to_le_bytes());
    psec.extend([1, 3, 1, 3]); // FlatFixed, F32, stride=1, [F32]
    psec.extend_from_slice(&0x303u64.to_le_bytes());

    if with_arena {
        // Process 1: vec_inc (config ArrayU32 VarU32, cmd ArrayU32 VarU32, reply ArrayU32 VarU32)
        psec.extend(wasm_name("vec_inc"));
        psec.extend(wasm_name("squid:proc_init:vec_inc"));
        psec.extend(wasm_name("squid:proc_step:vec_inc"));
        for h in [0x401u64, 0x402, 0x403] {
            psec.extend([2, 5, 0]); // VarU32(2), ArrayU32(5), stride=0
            psec.extend_from_slice(&h.to_le_bytes());
        }
    }

    // Types: 0=(i32)->(i32), 1=(f32)->(f32), 2=()->(i32)
    let types = [3, 0x60, 1, 0x7f, 1, 0x7f, 0x60, 1, 0x7d, 1, 0x7d, 0x60, 0, 1, 0x7f];
    let funcs = if with_arena { vec![7, 0, 1, 2, 0, 0, 0, 0] } else { vec![3, 0, 1, 2] };
    // Memory: 1 page min, 4 max (unexported when !with_arena!)
    let mem = [1, 1, 1, 4];
    // Globals: 0=acc (mut f32 = 0.0), 1=bias (mut i32 = 0)
    let globals = [2, 0x7d, 1, 0x43, 0, 0, 0, 0, 0x0b, 0x7f, 1, 0x41, 0, 0x0b];

    let mut exports = vec![if with_arena { 8 } else { 3 }];
    exports.extend(wasm_name("squid:proc_init:f32_acc"));
    exports.extend([0, 0]);
    exports.extend(wasm_name("squid:proc_step:f32_acc"));
    exports.extend([0, 1]);
    exports.extend(wasm_name("squid:finish_invocation"));
    exports.extend([0, 2]);
    if with_arena {
        exports.extend(wasm_name("squid:allocate"));
        exports.extend([0, 3]);
        exports.extend(wasm_name("squid:reset"));
        exports.extend([0, 4]);
        exports.extend(wasm_name("squid:proc_init:vec_inc"));
        exports.extend([0, 5]);
        exports.extend(wasm_name("squid:proc_step:vec_inc"));
        exports.extend([0, 6]);
        exports.extend(wasm_name("squid:memory"));
        exports.extend([2, 0]);
    }

    // f0: f32_acc init(cfg: u32) -> g0 = f32.convert_i32_u(cfg); return 0
    let f_f32_init = vec![0, 0x20, 0, 0xb3, 0x24, 0, 0x41, 0, 0x0b];
    // f1: f32_acc step(delta: f32) -> g0 = g0 + delta; return g0
    let f_f32_step = vec![0, 0x23, 0, 0x20, 0, 0x92, 0x24, 0, 0x23, 0, 0x0b];
    // f2: finish_invocation() -> 0
    let f_fin = vec![0, 0x41, 0, 0x0b];

    let mut bodies = vec![f_f32_init, f_f32_step, f_fin];
    if with_arena {
        // f3: allocate(_) -> 1024
        let mut f_alloc = vec![0, 0x41];
        f_alloc.extend(leb_u32(1024));
        f_alloc.push(0x0b);
        // f4: reset(_) -> 0
        let f_reset = vec![0, 0x41, 0, 0x0b];
        // f5: vec_inc init(ptr) -> g1 = mem[ptr + 4]; return 0
        let f_vinit = vec![0, 0x20, 0, 0x28, 2, 4, 0x24, 1, 0x41, 0, 0x0b];
        // f6: vec_inc step(ptr) -> out at 4096: len = mem[ptr], out[i] = mem[ptr + 4 + i*4] + g1, g1 += 1
        let mut f_vstep = vec![1, 2, 0x7f];
        f_vstep.extend([0x20, 0, 0x28, 2, 0, 0x21, 1]);
        f_vstep.push(0x41);
        f_vstep.extend(leb_u32(4096));
        f_vstep.extend([0x20, 1, 0x36, 2, 0]);
        f_vstep.extend([0x02, 0x40, 0x03, 0x40]);
        f_vstep.extend([0x20, 2, 0x20, 1, 0x4f, 0x0d, 1]);
        f_vstep.push(0x41);
        f_vstep.extend(leb_u32(4100));
        f_vstep.extend([0x20, 2, 0x41, 2, 0x74, 0x6a]);
        f_vstep.extend([0x20, 0, 0x20, 2, 0x41, 2, 0x74, 0x6a, 0x28, 2, 4, 0x23, 1, 0x6a, 0x36, 2, 0]);
        f_vstep.extend([0x20, 2, 0x41, 1, 0x6a, 0x21, 2, 0x0c, 0, 0x0b, 0x0b]);
        f_vstep.extend([0x23, 1, 0x41, 1, 0x6a, 0x24, 1]);
        f_vstep.push(0x41);
        f_vstep.extend(leb_u32(4096));
        f_vstep.push(0x0b);
        bodies.extend([f_alloc, f_reset, f_vinit, f_vstep]);
    }

    let mut code = vec![u8::try_from(bodies.len()).unwrap()];
    for body in &bodies {
        code.extend(leb_u32(u32::try_from(body.len()).unwrap()));
        code.extend_from_slice(body);
    }

    [
        b"\0asm\x01\0\0\0".to_vec(),
        wasm_custom(pen_runtime::abi::ABI_SECTION, &abi_payload),
        wasm_custom(pen_runtime::abi::INVOCATION_SECTION, &[1]),
        wasm_custom(pen_runtime::abi::PROCESSES_SECTION, &psec),
        wasm_section(1, &types),
        wasm_section(3, &funcs),
        wasm_section(5, &mem),
        wasm_section(6, &globals),
        wasm_section(7, &exports),
        wasm_section(10, &code),
    ]
    .concat()
}

#[test]
fn wtime_aot_executes_kernel_pool_natively_without_wasmtime_fallback() -> Result<()> {
    if !has_wtime_aot_binary() {
        return Ok(());
    }
    let engine = squid_host::engine()?;
    let other_engine = squid_host::engine()?;
    let cache = temp_cache_dir("kernels");
    let wasm = crafted_aot_kernel_wasm();

    let module = GuestModule::new_run(&engine, &wasm, Some(&cache))?;
    assert_eq!(module.backend_name(), "wtime-aot");
    assert!(module.wasmtime_module().is_none(), "wtime-aot module must not compile Wasmtime eagerly");

    // EngineMismatch & DeadlineRequiresBoundedEngine contracts remain intact on wtime-aot.
    assert_eq!(
        KernelPool::open(&other_engine, &module, "scale_add", KernelConfig::default()).err(),
        Some(KernelError::EngineMismatch)
    );
    assert_eq!(
        KernelPool::open(
            &engine,
            &module,
            "scale_add",
            KernelConfig { deadline: Some(Duration::from_millis(50)), ..KernelConfig::default() },
        )
        .err(),
        Some(KernelError::DeadlineRequiresBoundedEngine)
    );

    // Run prepared-context fixed kernel in both direct (workers=0) and parallel (workers=2) modes.
    for workers in [0usize, 2] {
        let mut pool = KernelPool::open(
            &engine,
            &module,
            "scale_add",
            KernelConfig { workers, default_chunk_items: 4, ..KernelConfig::default() },
        )?;
        pool.set_context(1, &[3, 7])?;
        let input: Vec<u32> = (0..16).collect();
        let out = pool.run_fixed(&input)?;
        let expected: Vec<u32> = input.iter().map(|&x| x * 3 + 7).collect();
        assert_eq!(out, expected);

        // Warm context replacement
        pool.set_context(2, &[5, 1])?;
        let out2 = pool.run_fixed(&input)?;
        let expected2: Vec<u32> = input.iter().map(|&x| x * 5 + 1).collect();
        assert_eq!(out2, expected2);
        pool.clear_context()?;
    }

    // Run variable-output kernel natively across 2 workers.
    let mut vpool = KernelPool::open(
        &engine,
        &module,
        "dup_u32",
        KernelConfig { workers: 2, default_chunk_items: 3, ..KernelConfig::default() },
    )?;
    let vout = vpool.run_variable(&[10, 20, 30, 40, 50])?;
    assert_eq!(vout, vec![10, 110, 20, 120, 30, 130, 40, 140, 50, 150]);

    assert!(
        module.wasmtime_module().is_none(),
        "KernelPool must run 100% natively on wtime-aot without lazily compiling Wasmtime"
    );
    let _ = std::fs::remove_dir_all(&cache);
    Ok(())
}

#[test]
fn wtime_aot_executes_processes_natively_with_hidden_memory_and_parallel_waves() -> Result<()> {
    if !has_wtime_aot_binary() {
        return Ok(());
    }
    let engine = squid_host::engine()?;
    let other_engine = squid_host::engine()?;
    let cache = temp_cache_dir("processes");

    // 1. Scalar-only process with UNEXPORTED hidden linear memory.
    let scalar_wasm = crafted_aot_process_wasm(false);
    let scalar_mod = GuestModule::new_run(&engine, &scalar_wasm, Some(&cache))?;
    assert_eq!(scalar_mod.backend_name(), "wtime-aot");
    assert!(scalar_mod.wasmtime_module().is_none());

    let base_policy = |name: &str| ProcessPolicy {
        name: name.to_string(),
        init_work: 1,
        step_work: 1,
        rust_wasmtime_fuel: None,
        deadline: None,
        memory: Limits { memory: 2 << 20 },
        mailbox: 8,
        mailbox_bytes: 4096,
        max_reply_bytes: 1024,
        slots: 4,
    };

    // Contract checks: EngineMismatch, FuelEngineRequired, BoundedEngineRequired.
    assert!(matches!(
        Processes::new(&other_engine, &scalar_mod, vec![base_policy("f32_acc")]),
        Err(ProcessHostError::EngineMismatch)
    ));
    let mut fuel_pol = base_policy("f32_acc");
    fuel_pol.rust_wasmtime_fuel = Some(FuelPolicy { init_fuel: 1000, step_fuel: 1000, cleanup_fuel: 1000 });
    assert!(matches!(Processes::new(&engine, &scalar_mod, vec![fuel_pol]), Err(ProcessHostError::FuelEngineRequired)));
    let mut dl_pol = base_policy("f32_acc");
    dl_pol.deadline = Some(Duration::from_millis(50));
    assert!(matches!(Processes::new(&engine, &scalar_mod, vec![dl_pol]), Err(ProcessHostError::BoundedEngineRequired)));

    let mut procs = Processes::new(&engine, &scalar_mod, vec![base_policy("f32_acc")])?;
    let h1 = match procs.admit("f32_acc", ProcessValue::U32(10)) {
        AdmitResult::Admitted(h) => h,
        other => panic!("expected Admitted, got {other:?}"),
    };
    let h2 = match procs.admit("f32_acc", ProcessValue::U32(100)) {
        AdmitResult::Admitted(h) => h,
        other => panic!("expected Admitted, got {other:?}"),
    };
    assert_eq!(procs.advance_all_parallel(10)?, 2);
    assert_eq!(procs.status(h1), ProcessStatus::Ready);
    assert_eq!(procs.status(h2), ProcessStatus::Ready);
    assert!(
        procs.memory_bytes() >= 2 * 65_536,
        "wtime-aot process host must report unexported hidden linear memory (got {})",
        procs.memory_bytes()
    );
    let t1 = match procs.send(h1, ProcessValue::F32(2.5)) {
        SendResult::Accepted(t) => t,
        other => panic!("expected Accepted, got {other:?}"),
    };
    let t2 = match procs.send(h2, ProcessValue::F32(-1.25)) {
        SendResult::Accepted(t) => t,
        other => panic!("expected Accepted, got {other:?}"),
    };
    assert_eq!(procs.advance_all_parallel(10)?, 2);
    let ev1 = procs.poll(h1).expect("reply 1");
    let ev2 = procs.poll(h2).expect("reply 2");
    assert_eq!(ev1.ticket, t1);
    assert_eq!(ev1.outcome, ProcessOutcome::Done(ProcessValue::F32(12.5)));
    assert_eq!(ev2.ticket, t2);
    assert_eq!(ev2.outcome, ProcessOutcome::Done(ProcessValue::F32(98.75)));
    assert!(scalar_mod.wasmtime_module().is_none());

    // 2. Array-bearing process (arena packet roundtrip + stateful progression).
    let arr_wasm = crafted_aot_process_wasm(true);
    let arr_mod = GuestModule::new_run(&engine, &arr_wasm, Some(&cache))?;
    assert_eq!(arr_mod.backend_name(), "wtime-aot");
    assert!(arr_mod.wasmtime_module().is_none());
    let mut aprocs = Processes::new(&engine, &arr_mod, vec![base_policy("f32_acc"), base_policy("vec_inc")])?;
    let hv = match aprocs.admit("vec_inc", ProcessValue::Words(vec![10])) {
        AdmitResult::Admitted(h) => h,
        other => panic!("expected Admitted, got {other:?}"),
    };
    assert_eq!(aprocs.advance(10)?, 1);
    let tv1 = match aprocs.send(hv, ProcessValue::Words(vec![1, 2, 3])) {
        SendResult::Accepted(t) => t,
        other => panic!("expected Accepted, got {other:?}"),
    };
    let tv2 = match aprocs.send(hv, ProcessValue::Words(vec![1, 2, 3])) {
        SendResult::Accepted(t) => t,
        other => panic!("expected Accepted, got {other:?}"),
    };
    assert_eq!(aprocs.advance(10)?, 2);
    assert_eq!(aprocs.poll(hv).unwrap().ticket, tv1);
    let second = aprocs.poll(hv).unwrap();
    assert_eq!(second.ticket, tv2);
    assert_eq!(second.outcome, ProcessOutcome::Done(ProcessValue::Words(vec![12, 13, 14])));
    assert!(arr_mod.wasmtime_module().is_none());

    let _ = std::fs::remove_dir_all(&cache);
    Ok(())
}
