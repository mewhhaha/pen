#![cfg(feature = "compiler")]
//! Phase 4 Rust Qualification Suite (`blot2/tasks/CONCURRENCY_VNEXT.md` §3.1, §5.1, §5.2, Laws #4 & #5).
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use blot_host::{
    GuestModule, Limits,
    kernels::{KernelConfig, KernelError, KernelPool},
    processes::{
        AdmitResult, FuelPolicy, ProcessHostError, ProcessOutcome, ProcessPolicy, ProcessStatus, ProcessValue,
        Processes, SendResult,
    },
};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(1);

fn compile_blot(source: &str) -> Result<Vec<u8>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let blot = root.join("../../../blot2");
    let compiler = std::env::var_os("BLOTC")
        .map(PathBuf::from)
        .unwrap_or_else(|| blot.join("compiler-goose/build/blotc-goose"));
    let id = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_dir = std::env::temp_dir().join(format!("pen-phase4-{}-{id}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir)?;
    let src_path = tmp_dir.join("mod.blot");
    let wasm_path = tmp_dir.join("mod.wasm");
    std::fs::write(&src_path, source)?;
    let out = Command::new(compiler)
        .arg("build")
        .arg(&src_path)
        .arg(&wasm_path)
        .arg("--prelude")
        .arg(blot.join("std/prelude.blot"))
        .output()
        .context("invoking blotc-goose")?;
    ensure!(
        out.status.success(),
        "blotc-goose failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = std::fs::read(&wasm_path)?;
    let _ = std::fs::remove_dir_all(&tmp_dir);
    Ok(bytes)
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

/// Strips Section ID 5 (`MemorySection`) from a Wasm container (used to test legacy Metadata v2).
fn strip_memory_section(wasm: &[u8]) -> Vec<u8> {
    let mut out = b"\0asm\x01\0\0\0".to_vec();
    for (id, body) in pen_runtime::abi::sections(wasm).unwrap() {
        if id != 5 {
            out.extend(wasm_section(id, body));
        }
    }
    out
}

/// Crafts a minimal ABI 8 variable-output kernel Wasm module with controllable:
/// - `spin_cleanup`: if true, `blot:finish_invocation` loops forever.
/// - `trap_cleanup_on_item`: if `Some(trigger)`, sets a global during `run` when input[0] == trigger so
///   `blot:finish_invocation` traps (`unreachable`) on that chunk!
/// - `overflow_on_item`: if `Some(trigger)`, returns `[0xffffffff]` sentinel when `input[0] == trigger`!
/// - `out_words_per_item`: number of output words per input item (`in_stride = 1`, `U32`).
fn crafted_var_kernel_wasm(
    spin_cleanup: bool,
    trap_cleanup_on_item: Option<u32>,
    overflow_on_item: Option<u32>,
    words_per_item: u32,
) -> Vec<u8> {
    // ABI 8 exports: `k` (ArrayU32 -> ArrayU32), `blot:memory`, `blot:allocate`, `blot:reset`, `blot:finish_invocation`.
    let abi_payload = [vec![8, 1], wasm_name("k"), vec![5, 5, 0]].concat();
    // blot:kernels v1: version=1, budget=2097152, count=1, name="k", flags=2 (variable_output)
    let mut ksec = vec![1];
    ksec.extend(leb_u32(blot_host::DEFAULT_TRANSPORT_BUDGET));
    ksec.push(1);
    ksec.extend(wasm_name("k"));
    ksec.push(2); // variable_output
    // in_schema: FlatFixed(1), ArrayU32(5), stride=1, word_tags=[1], hash=0x11
    ksec.extend([1, 5, 1, 1]);
    ksec.extend_from_slice(&0x11u64.to_le_bytes());
    // out_schema: VarU32(2), ArrayU32(5), stride=0, hash=0x22
    ksec.extend([2, 5, 0]);
    ksec.extend_from_slice(&0x22u64.to_le_bytes());
    // io_hash, impl_hash, binding_hash
    ksec.extend_from_slice(&0x33u64.to_le_bytes());
    ksec.extend_from_slice(&0x44u64.to_le_bytes());
    ksec.extend_from_slice(&0x55u64.to_le_bytes());

    // Types:
    // 0: (i32) -> (i32)  for allocate, reset, k
    // 1: () -> (i32)     for finish_invocation
    let types = [2, 0x60, 1, 0x7f, 1, 0x7f, 0x60, 0, 1, 0x7f];
    let funcs = [4, 0, 0, 0, 1];
    // Memory: 1 page (min=1, max=4)
    let mem = [1, 1, 1, 4];
    // Globals: 0 = mut i32 (trap_flag, init 0)
    let globals = [1, 0x7f, 1, 0x41, 0, 0x0b];
    let exports = [
        vec![5],
        wasm_name("blot:allocate"),
        vec![0, 0],
        wasm_name("blot:reset"),
        vec![0, 1],
        wasm_name("k"),
        vec![0, 2],
        wasm_name("blot:finish_invocation"),
        vec![0, 3],
        wasm_name("blot:memory"),
        vec![2, 0],
    ]
    .concat();

    // func 0: allocate(bytes) -> return 1024
    let mut f_alloc = vec![0, 0x41];
    f_alloc.extend(leb_u32(1024));
    f_alloc.push(0x0b);
    // func 1: reset(x) -> return 0
    let f_reset = vec![0, 0x41, 0, 0x0b];

    // func 2: k(in_ptr: i32) -> out_ptr (at 2048)
    // Reads count = mem[in_ptr], first_item = count > 0 ? mem[in_ptr + 4] : 0.
    // Sets global 0 = 1 if trap_cleanup_on_item == Some(first_item).
    // If overflow_on_item == Some(first_item) -> writes 0xffffffff at 2048, returns 2048.
    // Else writes total_words = count * words_per_item at 2048, fills each word with first_item, returns 2048.
    let mut f_k = vec![1, 3, 0x7f]; // 3 i32 locals: 1=count, 2=first, 3=total_words
    // local 1 = i32.load(local 0)
    f_k.extend([0x20, 0, 0x28, 2, 0, 0x21, 1]);
    // local 2 = i32.load(local 0 + 4)
    f_k.extend([0x20, 0, 0x28, 2, 4, 0x21, 2]);
    // global 0 = 0
    f_k.extend([0x41, 0, 0x24, 0]);
    if let Some(trig) = trap_cleanup_on_item {
        f_k.extend([0x20, 2, 0x41]);
        f_k.extend(leb_u32(trig));
        f_k.extend([0x46, 0x04, 0x40, 0x41, 1, 0x24, 0, 0x0b]);
    }
    if let Some(ov) = overflow_on_item {
        f_k.extend([0x20, 2, 0x41]);
        f_k.extend(leb_u32(ov));
        // if first == ov { i32.store(2048, -1); return 2048; }
        f_k.extend([0x46, 0x04, 0x40, 0x41]);
        f_k.extend(leb_u32(2048));
        f_k.extend([0x41, 0x7f, 0x36, 2, 0, 0x41]);
        f_k.extend(leb_u32(2048));
        f_k.extend([0x0f, 0x0b]);
    }
    // local 3 = count * words_per_item
    f_k.extend([0x20, 1, 0x41]);
    f_k.extend(leb_u32(words_per_item));
    f_k.extend([0x6c, 0x21, 3]);
    // i32.store(2048, local 3)
    f_k.push(0x41);
    f_k.extend(leb_u32(2048));
    f_k.extend([0x20, 3, 0x36, 2, 0]);
    // store first_item into first two output slots for visibility
    f_k.push(0x41);
    f_k.extend(leb_u32(2052));
    f_k.extend([0x20, 2, 0x36, 2, 0]);
    f_k.push(0x41);
    f_k.extend(leb_u32(2048));
    f_k.push(0x0b);

    // func 3: finish_invocation() -> i32
    let mut f_fin = vec![0];
    if spin_cleanup {
        // loop { br 0 }
        f_fin.extend([0x03, 0x40, 0x0c, 0, 0x0b, 0x41, 0, 0x0b]);
    } else {
        // if global 0 != 0 { unreachable } else { return 0 }
        f_fin.extend([0x23, 0, 0x04, 0x40, 0x00, 0x0b, 0x41, 0, 0x0b]);
    }

    let mut code = vec![4];
    for body in [&f_alloc, &f_reset, &f_k, &f_fin] {
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

/// Crafts a minimal ABI 8 scalar-only `#process` whose `blot:finish_invocation` spins forever
/// (testing Law #5(b): successful `proc_step` followed by cleanup Overrun must NOT be masked!).
fn crafted_spin_cleanup_process_wasm() -> Vec<u8> {
    // ABI 8 exports: `blot:proc_init:p` (U32->Unit), `blot:proc_step:p` (U32->U32), `blot:finish_invocation`
    let abi_payload = [
        vec![8, 2],
        wasm_name("blot:proc_init:p"),
        vec![1, 0],
        wasm_name("blot:proc_step:p"),
        vec![1, 1],
        vec![0],
    ]
    .concat();
    let mut psec = vec![1];
    psec.extend(leb_u32(blot_host::DEFAULT_TRANSPORT_BUDGET));
    psec.push(1);
    psec.extend(wasm_name("p"));
    psec.extend(wasm_name("blot:proc_init:p"));
    psec.extend(wasm_name("blot:proc_step:p"));
    for h in [0x11u64, 0x22, 0x33] {
        psec.extend([1, 1, 1, 1]); // FlatFixed, U32, stride=1, word_tag=U32
        psec.extend_from_slice(&h.to_le_bytes());
    }
    let types = [2, 0x60, 1, 0x7f, 1, 0x7f, 0x60, 0, 1, 0x7f];
    let funcs = [3, 0, 0, 1];
    let globals = [1, 0x7f, 1, 0x41, 0, 0x0b]; // global 0: spin_on_cleanup (0 during init, 1 during step!)
    let exports = [
        vec![3],
        wasm_name("blot:proc_init:p"),
        vec![0, 0],
        wasm_name("blot:proc_step:p"),
        vec![0, 1],
        wasm_name("blot:finish_invocation"),
        vec![0, 2],
    ]
    .concat();
    // init(x) -> global 0 = 0; return 0
    let f_init = vec![0, 0x41, 0, 0x24, 0, 0x41, 0, 0x0b];
    // step(x) -> global 0 = 1; return x + 1
    let f_step = vec![0, 0x41, 1, 0x24, 0, 0x20, 0, 0x41, 1, 0x6a, 0x0b];
    // finish_invocation() -> if global 0 != 0 { loop { br 0 } }; return 0
    let f_fin = vec![0, 0x23, 0, 0x04, 0x40, 0x03, 0x40, 0x0c, 0, 0x0b, 0x0b, 0x41, 0, 0x0b];
    let mut code = vec![3];
    for body in [&f_init, &f_step, &f_fin] {
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
        wasm_section(6, &globals),
        wasm_section(7, &exports),
        wasm_section(10, &code),
    ]
    .concat()
}

// =============================================================================
// 1. §3.1: Scalar-Only native_bytes -> from_native & Legacy Metadata v2 Boundary
// =============================================================================

#[test]
fn scalar_only_process_native_roundtrip_and_legacy_missing_memory_section_boundary() -> Result<()> {
    let wasm = compile_blot(
        r#"
func s_init (x: U32): U32 => return x
func s_step (st: own U32) -> (cmd: U32): (U32, U32) =>
  let next = @u32.add st cmd
  return (next, next)
#process const scalar_only: Process [U32, U32, U32] = { init: s_init, step: s_step }
"#,
    )?;
    let abi = blot_host::read_abi(&wasm)?;
    assert!(!abi.uses_arrays(), "scalar-only process must not require exported arena");
    let shape = blot_host::read_memory_shape(&wasm)?;
    assert!(shape.section_present && shape.has_memory, "scalar-only ABI 8 still defines internal linear memory");

    let engine = blot_host::engine()?;
    let (metadata, precompiled) = GuestModule::native_bytes(&engine, &wasm)?;
    // SAFETY: just precompiled by this engine.
    let module = unsafe { GuestModule::from_native(&engine, &metadata, &precompiled) }?;
    assert!(module.wasmtime_module().unwrap().get_export("blot:memory").is_none());

    // Execute via Processes and confirm MemoryTrackerLimiter tracks the unexported hidden memory!
    let mut procs = Processes::new(
        &engine,
        &module,
        vec![ProcessPolicy {
            name: "scalar_only".into(),
            init_work: 1,
            step_work: 1,
            rust_wasmtime_fuel: None,
            deadline: None,
            memory: Limits { memory: 16 << 20 },
            mailbox: 4,
            mailbox_bytes: 256,
            max_reply_bytes: 4,
            slots: 2,
        }],
    )?;
    let AdmitResult::Admitted(h) = procs.admit("scalar_only", ProcessValue::U32(10)) else {
        panic!("admit failed");
    };
    assert_eq!(procs.advance(1)?, 1);
    assert!(procs.memory_bytes() >= 65_536, "MemoryTrackerLimiter must report unexported hidden memory");
    let SendResult::Accepted(t1) = procs.send(h, ProcessValue::U32(32)) else {
        panic!("send failed");
    };
    assert_eq!(procs.advance(1)?, 1);
    assert_eq!(procs.poll(h).unwrap().outcome, ProcessOutcome::Done(ProcessValue::U32(42)));
    let _ = t1;

    // Strip MemorySection from native metadata JSON -> from_native MUST explicitly reject with repackage diagnostic!
    let mut json: serde_json::Value = serde_json::from_slice(&metadata)?;
    let orig_sections: Vec<u8> = serde_json::from_value(json["sections"].clone())?;
    let stripped = strip_memory_section(&orig_sections);
    json["sections"] = serde_json::to_value(&stripped)?;
    let legacy_bad_meta = serde_json::to_vec(&json)?;
    let err = unsafe { GuestModule::from_native(&engine, &legacy_bad_meta, &precompiled) }
        .err()
        .expect("missing MemorySection on hidden-memory process package must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("MemorySection") && msg.contains("repackage"),
        "unexpected error message: {msg}"
    );

    // Ordinary legacy ABI 8 module WITH exported blot:memory and ZERO kernels/processes still loads cleanly!
    let legacy_wasm = compile_blot(
        r#"
io echo (xs: #[U32]): #[U32] => return copy xs
"#,
    )?;
    let (legacy_meta, legacy_code) = GuestModule::native_bytes(&engine, &legacy_wasm)?;
    let mut ljson: serde_json::Value = serde_json::from_slice(&legacy_meta)?;
    let lsec: Vec<u8> = serde_json::from_value(ljson["sections"].clone())?;
    ljson["sections"] = serde_json::to_value(strip_memory_section(&lsec))?;
    let lmeta_stripped = serde_json::to_vec(&ljson)?;
    let loaded = unsafe { GuestModule::from_native(&engine, &lmeta_stripped, &legacy_code) }?;
    let mut g = blot_host::Guest::new(&engine, &loaded)?;
    assert_eq!(g.call("echo", blot_host::Value::ArrayU32(vec![7, 8]))?, blot_host::Value::ArrayU32(vec![7, 8]));
    Ok(())
}

// =============================================================================
// 2. §5.1 / Law #4: Engine Admission, Unticked & Slow-Ticking Condvar Wakeup
// =============================================================================

#[test]
fn kernel_pool_engine_admission_unticked_and_slow_ticking_condvar_wakeup() -> Result<()> {
    let wasm = compile_blot(
        r#"
#kernel func double (x: U32): U32 => return @u32.mul x 2
#kernel func spin (x: U32): U32 =>
  let acc = x
  for ever:
    acc := @u32.add acc 1
  return acc
"#,
    )?;

    // 1. EngineMismatch (!Engine::same)
    let eng_a = blot_host::engine()?;
    let eng_b = blot_host::engine()?;
    let mod_a = GuestModule::new(&eng_a, &wasm, None)?;
    assert_eq!(
        KernelPool::open(&eng_b, &mod_a, "double", KernelConfig::default()).err(),
        Some(KernelError::EngineMismatch)
    );

    // 2. FuelEngineUnsupportedForKernels (both fuel_engine and fuel_bounded_engine)
    let fuel_eng = blot_host::fuel_engine()?;
    let fuel_mod = GuestModule::new(&fuel_eng, &wasm, None)?;
    assert_eq!(
        KernelPool::open(&fuel_eng, &fuel_mod, "double", KernelConfig::default()).err(),
        Some(KernelError::FuelEngineUnsupportedForKernels)
    );
    let fb_eng = blot_host::fuel_bounded_engine(Duration::from_millis(2))?;
    let fb_mod = GuestModule::new(&fb_eng, &wasm, None)?;
    assert_eq!(
        KernelPool::open(&fb_eng, &fb_mod, "double", KernelConfig::default()).err(),
        Some(KernelError::FuelEngineUnsupportedForKernels)
    );

    // 3. DeadlineRequiresBoundedEngine on non-epoch engine()
    let cfg_dl = KernelConfig { deadline: Some(Duration::from_millis(50)), ..KernelConfig::default() };
    assert_eq!(
        KernelPool::open(&eng_a, &mod_a, "double", cfg_dl).err(),
        Some(KernelError::DeadlineRequiresBoundedEngine)
    );

    // 4. Caller-created UNTICKED epoch-enabled Engine with deadline = Some(...) -> auto-ticked!
    let mut raw_cfg = wasmtime::Config::new();
    raw_cfg.wasm_reference_types(true);
    raw_cfg.wasm_simd(true);
    raw_cfg.epoch_interruption(true);
    let unticked_eng = wasmtime::Engine::new(&raw_cfg)?;
    let unticked_mod = GuestModule::new(&unticked_eng, &wasm, None)?;
    let mut spin_pool = KernelPool::open(
        &unticked_eng,
        &unticked_mod,
        "spin",
        KernelConfig { workers: 2, deadline: Some(Duration::from_millis(35)), ..KernelConfig::default() },
    )?;
    let t0 = Instant::now();
    assert_eq!(spin_pool.run_fixed(&[1, 2]), Err(KernelError::Timeout));
    assert!(t0.elapsed() < Duration::from_millis(600), "unticked engine was not auto-ticked promptly");

    // 5. Caller-created UNTICKED epoch-enabled Engine with deadline = None -> spinning finish_invocation
    //    is interrupted cleanly after cleanup_grace = max(10ms, 2 * effective_tick)!
    let unticked_eng2 = wasmtime::Engine::new(&raw_cfg)?;
    let spin_clean_wasm = crafted_var_kernel_wasm(true, None, None, 1);
    let spin_clean_mod = GuestModule::new(&unticked_eng2, &spin_clean_wasm, None)?;
    let mut sc_pool = KernelPool::open(
        &unticked_eng2,
        &spin_clean_mod,
        "k",
        KernelConfig { workers: 1, deadline: None, ..KernelConfig::default() },
    )?;
    let t1 = Instant::now();
    let sc_err = sc_pool.run_variable(&[1]).unwrap_err();
    assert!(matches!(sc_err, KernelError::CleanupTrap { item_index: 0, .. }), "got {sc_err:?}");
    assert!(t1.elapsed() < Duration::from_millis(500), "spinning cleanup was not interrupted after cleanup_grace");

    // 6. Slow-ticking (2000ms) already-running epoch engine tightened via Condvar::notify_one()!
    let slow_eng = wasmtime::Engine::new(&raw_cfg)?;
    let initial_tick = blot_host::ensure_engine_ticker(&slow_eng, Duration::from_secs(2))?;
    assert_eq!(initial_tick, Duration::from_secs(2));
    // Give the ticker a few milliseconds to enter its 2000ms wait_timeout sleep.
    std::thread::sleep(Duration::from_millis(15));
    let slow_mod = GuestModule::new(&slow_eng, &wasm, None)?;
    let mut tightened_pool = KernelPool::open(
        &slow_eng,
        &slow_mod,
        "spin",
        KernelConfig { workers: 1, deadline: Some(Duration::from_millis(35)), ..KernelConfig::default() },
    )?;
    assert!(tightened_pool.stats().effective_tick.unwrap() <= Duration::from_millis(2));
    let t2 = Instant::now();
    assert_eq!(tightened_pool.run_fixed(&[1]), Err(KernelError::Timeout));
    assert!(
        t2.elapsed() < Duration::from_millis(500),
        "Condvar::notify_one did not immediately wake 2000ms sleeping ticker (took {:?})",
        t2.elapsed()
    );
    Ok(())
}

// =============================================================================
// 3. §5.1 / Law #4: Prefix Order, Prepared Empty Contract, [0xffffffff] Bisection,
//    Same-start_item Precedence, 12B Deadlock-Free Budget & 1 MiB Chunking
// =============================================================================

#[test]
fn kernel_pool_prefix_errors_prepared_empty_bisection_precedence_and_chunking() -> Result<()> {
    let wasm = compile_blot(
        r#"
func prep (ctx: #[U32]): #[U32] => return copy ctx
func run_ctx (ctx: #[U32]) -> (x: U32): U32 =>
  if x == 999:
    return @panic "trap_item"
  let sum = 0
  if x == 888:
    for i in 0..50000:
      sum := @u32.add sum i
    return @panic "slow_trap"
  return @u32.add x ctx[0]

#kernel const prepared_k: Kernel [#[U32], U32, U32] = { prepare: prep, run: run_ctx }

func var_row (n: U32): #[U32] =>
  let out: #[U32] = #[]
  for i in 0..n:
    @storage.push out (@u32.add n i)
  return out

#kernel func var_k (n: U32): #[U32] => return var_row n

// 1 MiB fixed row (262,144 U32 words = 1,048,576 bytes) under 2 MiB transport_budget!
#kernel func huge_fixed (row: #[262144; U32]): U32 =>
  return @u32.add row[0] row[262143]
"#,
    )?;
    let engine = blot_host::engine()?;
    let module = GuestModule::new(&engine, &wasm, None)?;

    // A. Prepared empty-batch contract + idempotent/mismatch set_context + prefix-ordered trap across workers 0,1,2,4,8
    for workers in [0usize, 1, 2, 4, 8] {
        let mut pool = KernelPool::open(
            &engine,
            &module,
            "prepared_k",
            KernelConfig { workers, default_chunk_items: 1, ..KernelConfig::default() },
        )?;
        assert_eq!(pool.run_fixed(&[]), Err(KernelError::ContextNotPrepared));
        pool.set_context(1, &[10])?;
        assert_eq!(pool.set_context(1, &[11]), Err(KernelError::RevisionContentMismatch));
        pool.set_context(1, &[10])?; // idempotent
        assert_eq!(pool.run_fixed(&[])?, Vec::<u32>::new());
        assert_eq!(pool.run_fixed(&[1, 2, 3])?, vec![11, 12, 13]);
        // Item 0 traps slowly ("slow_trap"), item 1 traps fast ("trap_item") -> item 0 MUST win!
        let err = pool.run_fixed(&[888, 999]).unwrap_err();
        assert!(
            matches!(err, KernelError::RunTrap { item_index: 0, .. }),
            "workers={workers}: expected item 0 trap to win, got {err:?}"
        );
        // After recoverable trap, pool stays reusable!
        assert_eq!(pool.run_fixed(&[5, 6])?, vec![15, 16]);
        pool.clear_context()?;
        assert_eq!(pool.run_fixed(&[]), Err(KernelError::ContextNotPrepared));
    }

    // B. Concurrent [0xffffffff] bisection parity across workers 0,1,2,4,8 (each item emits 200_000 words = 800 KB;
    //    4 items = 3.2 MB > 2 MiB transport_budget -> forces multi-level bisection!)
    let mut baseline: Option<Vec<u32>> = None;
    for workers in [0usize, 1, 2, 4, 8] {
        let mut pool = KernelPool::open(
            &engine,
            &module,
            "var_k",
            KernelConfig {
                workers,
                default_chunk_items: 4,
                max_total_output_bytes: 16 << 20,
                max_physical_output_bytes: 32 << 20,
                ..KernelConfig::default()
            },
        )?;
        let out = pool.run_variable(&[150_000, 150_000, 150_000, 150_000])?;
        assert!(pool.stats().bisections >= 1, "workers={workers} should bisect on [0xffffffff]");
        if let Some(ref exp) = baseline {
            assert_eq!(&out, exp, "bisected output must be identical for workers={workers}");
        } else {
            baseline = Some(out);
        }
        // Single item > max_out_words (524,287) -> SingleItemOutputOverflow at item 1!
        let ov = pool.run_variable(&[1, 524_288]).unwrap_err();
        assert_eq!(ov, KernelError::SingleItemOutputOverflow { item_index: 1 });
    }

    // C. Budget exhaustion with zero evictable results (12B cap: chunk 0 = 8B, chunk 1 = 8B) AND
    //    worker-count-invariant same-start_item precedence (OutputBudgetExceeded > CleanupTrap)!
    let clean_trap_wasm = crafted_var_kernel_wasm(false, Some(2), None, 2); // 2 words = 8B per item, item 2 traps cleanup!
    let clean_trap_mod = GuestModule::new(&engine, &clean_trap_wasm, None)?;
    for workers in [0usize, 1, 2, 4, 8] {
        // Under 12B cap: chunk 0 (item 1) = 8B, chunk 1 (item 2) = 8B + traps cleanup -> 8+8=16 > 12 -> OutputBudgetExceeded!
        let mut p_over = KernelPool::open(
            &engine,
            &clean_trap_mod,
            "k",
            KernelConfig {
                workers,
                default_chunk_items: 1,
                max_total_output_bytes: 12,
                max_physical_output_bytes: 24,
                ..KernelConfig::default()
            },
        )?;
        let err_over = p_over.run_variable(&[1, 2]).unwrap_err();
        assert!(
            matches!(err_over, KernelError::OutputBudgetExceeded { required_bytes: 16, limit_bytes: 12 }),
            "workers={workers}: OutputBudgetExceeded must beat CleanupTrap when 8+8>12, got {err_over:?}"
        );
        assert!(p_over.stats().peak_charged_bytes <= 12);

        // Under 16B cap: chunk 0 = 8B, chunk 1 = 8B + traps cleanup -> 8+8=16 <= 16 -> CleanupTrap!
        let mut p_fit = KernelPool::open(
            &engine,
            &clean_trap_mod,
            "k",
            KernelConfig {
                workers,
                default_chunk_items: 1,
                max_total_output_bytes: 16,
                max_physical_output_bytes: 32,
                ..KernelConfig::default()
            },
        )?;
        let err_fit = p_fit.run_variable(&[1, 2]).unwrap_err();
        assert!(
            matches!(err_fit, KernelError::CleanupTrap { item_index: 1, .. }),
            "workers={workers}: CleanupTrap must win when 8+8<=16, got {err_fit:?}"
        );
    }

    // D. Unified [0xffffffff] + OVERFLOW_CLEANUP_FAIL: len > 1 -> CleanupTrap WITHOUT bisecting;
    //    len == 1 -> SingleItemOutputOverflow!
    let ov_clean_wasm = crafted_var_kernel_wasm(false, Some(7), Some(7), 1);
    let ov_clean_mod = GuestModule::new(&engine, &ov_clean_wasm, None)?;
    for workers in [0usize, 1, 2] {
        let mut p_multi = KernelPool::open(
            &engine,
            &ov_clean_mod,
            "k",
            KernelConfig { workers, default_chunk_items: 2, ..KernelConfig::default() },
        )?;
        let err_multi = p_multi.run_variable(&[7, 8]).unwrap_err();
        assert!(matches!(err_multi, KernelError::CleanupTrap { item_index: 0, .. }), "{err_multi:?}");
        assert_eq!(p_multi.stats().bisections, 0, "OVERFLOW_CLEANUP_FAIL on len>1 must NOT bisect");

        let mut p_single = KernelPool::open(
            &engine,
            &ov_clean_mod,
            "k",
            KernelConfig { workers, default_chunk_items: 1, ..KernelConfig::default() },
        )?;
        let err_single = p_single.run_variable(&[7]).unwrap_err();
        assert_eq!(err_single, KernelError::SingleItemOutputOverflow { item_index: 0 });
    }

    // E. 1 MiB fixed row under 2 MiB transport_budget -> integer floor division caps max_chunk_items == 1!
    let mut huge_pool = KernelPool::open(
        &engine,
        &module,
        "huge_fixed",
        KernelConfig { workers: 2, default_chunk_items: 64, ..KernelConfig::default() },
    )?;
    assert_eq!(huge_pool.max_chunk_items(), 1, "two 1 MiB rows + 4B header exceed 2 MiB transport_budget");
    let mut two_rows = vec![0u32; 262_144 * 2];
    two_rows[0] = 10;
    two_rows[262_143] = 5;
    two_rows[262_144] = 20;
    two_rows[262_144 * 2 - 1] = 7;
    assert_eq!(huge_pool.run_fixed(&two_rows)?, vec![15, 27]);
    Ok(())
}

// =============================================================================
// 4. Law #4: Warm-Worker History-Independence, Fragmentation Retry & Zero Drift
// =============================================================================

#[test]
fn kernel_pool_warm_worker_fragmentation_retry_monotonicity_and_zero_drift() -> Result<()> {
    let wasm = compile_blot(
        r#"
// Witness (1) & (7): heap class fragmentation + pre-first-chunk replacement + fresh-no-retry
func alloc_class_blocks (class_words: U32) -> (blocks: U32): U32 =>
  let sum = 0
  for i in 0..blocks:
    let arr: #[U32] = #[]
    for j in 0..class_words:
      @storage.push arr (@u32.add i j)
    sum := @u32.add sum arr[0]
  return sum

// cmd encoding: 1 = class-A warmup (small blocks), 2 = class-B test (larger blocks), 3 = huge OOM always
#kernel func frag_heap (cmd: U32): U32 =>
  if cmd == 1:
    return alloc_class_blocks 2000 100
  if cmd == 2:
    return alloc_class_blocks 5000 40
  return alloc_class_blocks 500000 200

// Witness (2): retained stack chunks (60 MB then 80 MB) + reversed-order inner rewind
func touch_stack (n: U32): U32 =>
  let a = @array.fill n 7
  return a[0]

#kernel func stack_frag (n: U32): U32 => return touch_stack n

// Witness (5): fixed, variable, and variable-overflow zero heap_live_bytes drift across 100 batches
#kernel func drift_fixed (x: U32): #[4; U32] =>
  return #[x, @u32.add x 1, @u32.add x 2, @u32.add x 3]

#kernel func drift_var (n: U32): #[U32] =>
  let out: #[U32] = #[]
  for i in 0..n:
    @storage.push out i
  return out

// Witness (6): context replacement + rollback preserving rev 1 (with start-time static pool!)
const STATIC_POOL: #[U32] = #[100, 200, 300, 400]

func prep_ctx (w: #[U32]): #[U32] =>
  if w.length > 0 and w[0] == 999999:
    return @panic "bad_rev"
  let big: #[U32] = #[]
  let n = if w.length > 0 then w[0] else 1
  for i in 0..n:
    @storage.push big (@u32.add i STATIC_POOL[0])
  return big

func run_ctx (ctx: #[U32]) -> (x: U32): U32 =>
  return @u32.add x ctx[0]

#kernel const ctx_k: Kernel [#[U32], U32, U32] = { prepare: prep_ctx, run: run_ctx }
"#,
    )?;
    let engine = blot_host::engine()?;
    let module = GuestModule::new(&engine, &wasm, None)?;

    // Witness (2): retained stack chunk 60 MB (15M words) then 80 MB (20M words) on warm worker under 128 MiB stack budget!
    for workers in [0usize, 1, 2] {
        let mut sp = KernelPool::open(
            &engine,
            &module,
            "stack_frag",
            KernelConfig {
                workers,
                memory: Some(Limits { memory: 256 << 20 }),
                default_chunk_items: 1,
                ..KernelConfig::default()
            },
        )?;
        // Warm every worker slot with 60MB before running 80MB so every worker is fragmented.
        let warm_in = vec![15_000_000u32; workers.max(1)];
        let expected_warm = vec![7u32; workers.max(1)];
        assert_eq!(sp.run_fixed(&warm_in)?, expected_warm);
        let big_in = vec![20_000_000u32; workers.max(1)];
        let expected_big = vec![7u32; workers.max(1)];
        assert_eq!(sp.run_fixed(&big_in)?, expected_big);
        // Fresh retry must have kicked in cleanly since warm retained 60MB + 80MB > 128MB stack_budget!
        assert!(sp.stats().fresh_oom_retries >= 1, "workers={workers} should retire+retry fragmented stack store");
    }

    // Witness (4): bounded-policy.memory warm-vs-fresh transport symmetry!
    // Find exact fresh_required_pages for a variable output batch, then verify BOTH fresh and warm fail at
    // (fresh_required_pages - 1) and BOTH succeed at fresh_required_pages.
    let mut fresh_pages = 1usize;
    loop {
        let mut probe = KernelPool::open(
            &engine,
            &module,
            "drift_var",
            KernelConfig {
                workers: 0,
                memory: Some(Limits { memory: fresh_pages * 65_536 }),
                default_chunk_items: 1,
                ..KernelConfig::default()
            },
        )?;
        if probe.run_variable(&[16_384]).is_ok() {
            break;
        }
        fresh_pages += 1;
        ensure!(fresh_pages <= 128, "could not find fresh_required_pages");
    }
    let mut warm_tight = KernelPool::open(
        &engine,
        &module,
        "drift_var",
        KernelConfig {
            workers: 0,
            memory: Some(Limits { memory: (fresh_pages - 1) * 65_536 }),
            default_chunk_items: 1,
            ..KernelConfig::default()
        },
    )?;
    // Warm up with a smaller batch, then run the exact threshold batch -> MUST still fail identically to fresh!
    let _ = warm_tight.run_variable(&[16]);
    assert!(matches!(warm_tight.run_variable(&[16_384]), Err(KernelError::OutOfMemory { .. })));

    let mut warm_ok = KernelPool::open(
        &engine,
        &module,
        "drift_var",
        KernelConfig {
            workers: 0,
            memory: Some(Limits { memory: fresh_pages * 65_536 }),
            default_chunk_items: 1,
            ..KernelConfig::default()
        },
    )?;
    assert!(warm_ok.run_variable(&[32_768]).is_err() || warm_ok.run_variable(&[16_384]).is_ok());
    assert!(warm_ok.run_variable(&[16_384]).is_ok());

    // Witness (5): 100 repeated batches across fixed, variable, and variable-overflow pools with zero drift!
    let mut p_fix = KernelPool::open(&engine, &module, "drift_fixed", KernelConfig { workers: 1, ..KernelConfig::default() })?;
    let mut p_var = KernelPool::open(&engine, &module, "drift_var", KernelConfig { workers: 1, ..KernelConfig::default() })?;
    let mut p_ov = KernelPool::open(
        &engine,
        &module,
        "drift_var",
        KernelConfig { workers: 1, default_chunk_items: 4, ..KernelConfig::default() },
    )?;
    for i in 0..100u32 {
        assert_eq!(p_fix.run_fixed(&[i])?, vec![i, i + 1, i + 2, i + 3]);
        assert_eq!(p_var.run_variable(&[16])?.len(), 17);
        assert_eq!(p_ov.run_variable(&[200_000, 200_000, 200_000])?.len(), 600_003);
    }

    // Witness (6) & (7): warm context replacement + rollback preserving rev 1 + fresh-no-retry!
    let mut ctx_pool = KernelPool::open(
        &engine,
        &module,
        "ctx_k",
        KernelConfig { workers: 1, memory: Some(Limits { memory: 64 << 20 }), ..KernelConfig::default() },
    )?;
    ctx_pool.set_context(1, &[50_000])?;
    // Install rev 1 on worker 0 via empty count==0 batch!
    assert_eq!(ctx_pool.run_fixed(&[])?, Vec::<u32>::new());
    // Failed rev 2 candidate must preserve rev 1 untouched!
    assert!(matches!(ctx_pool.set_context(2, &[999_999]), Err(KernelError::ContextTrap { .. })));
    assert_eq!(ctx_pool.run_fixed(&[5])?, vec![105]);
    // Valid rev 2 replaces rev 1 via blot:kernel_clear first!
    ctx_pool.set_context(2, &[60_000])?;
    assert_eq!(ctx_pool.run_fixed(&[7])?, vec![107]);

    // Witness (7)(iv): brand-new fresh Store hitting physical OOM executes ZERO retries!
    let mut fresh_oom_pool = KernelPool::open(
        &engine,
        &module,
        "frag_heap",
        KernelConfig {
            workers: 1,
            memory: Some(Limits { memory: 2 << 20 }),
            default_chunk_items: 1,
            ..KernelConfig::default()
        },
    )?;
    assert!(matches!(fresh_oom_pool.run_fixed(&[3]), Err(KernelError::OutOfMemory { .. })));
    assert_eq!(
        fresh_oom_pool.stats().fresh_oom_retries,
        0,
        "already-fresh store must execute ZERO retries on physical OOM"
    );
    Ok(())
}

// =============================================================================
// 5. §5.2 / Law #5: Processes Engine Admission, Mixed Fuel+Deadline, Overrun
//    Precedence, Sibling Cancel, Cleanup Overrun & Exhaustive Serial == Parallel
// =============================================================================

#[test]
fn processes_law5_admission_mixed_fuel_overrun_precedence_and_parallel_equivalence() -> Result<()> {
    let wasm = compile_blot(
        r#"
func fast_init (x: U32): U32 => return x
func fast_step (st: own U32) -> (cmd: U32): (U32, U32) =>
  let next = @u32.add st cmd
  return (next, next)
#process const fast: Process [U32, U32, U32] = { init: fast_init, step: fast_step }

func slow_init (x: U32): U32 => return x
func slow_step (st: own U32) -> (cmd: U32): (U32, U32) =>
  let acc = st
  if cmd == 999999:
    for ever:
      acc := @u32.add acc 1
  else:
    for i in 0..cmd:
      acc := @u32.add acc 1
  return (acc, acc)
#process const slow: Process [U32, U32, U32] = { init: slow_init, step: slow_step }

func unit_init (_: Unit): U32 => return 0
func unit_step (st: own U32) -> (_: Unit): (U32, Unit) => return (@u32.add st 1, ())
#process const unit_p: Process [Unit, Unit, Unit] = { init: unit_init, step: unit_step }
"#,
    )?;

    // 1. EngineMismatch, FuelEngineRequired, BoundedEngineRequired, max_reply_bytes < 4 rejected
    let eng1 = blot_host::engine()?;
    let eng2 = blot_host::engine()?;
    let mod1 = GuestModule::new(&eng1, &wasm, None)?;
    let base_policy = |name: &str| ProcessPolicy {
        name: name.into(),
        init_work: 1,
        step_work: 1,
        rust_wasmtime_fuel: None,
        deadline: None,
        memory: Limits { memory: 16 << 20 },
        mailbox: 4,
        mailbox_bytes: 1024,
        max_reply_bytes: 4,
        slots: 4,
    };
    assert!(matches!(Processes::new(&eng2, &mod1, vec![base_policy("fast")]), Err(ProcessHostError::EngineMismatch)));
    let mut fuel_p = base_policy("fast");
    fuel_p.rust_wasmtime_fuel = Some(FuelPolicy { init_fuel: 100_000, step_fuel: 100_000, cleanup_fuel: 100_000 });
    assert!(matches!(Processes::new(&eng1, &mod1, vec![fuel_p.clone()]), Err(ProcessHostError::FuelEngineRequired)));
    let mut dl_p = base_policy("fast");
    dl_p.deadline = Some(Duration::from_millis(50));
    assert!(matches!(Processes::new(&eng1, &mod1, vec![dl_p]), Err(ProcessHostError::BoundedEngineRequired)));
    let mut bad_reply = base_policy("fast");
    bad_reply.max_reply_bytes = 3;
    assert!(matches!(Processes::new(&eng1, &mod1, vec![bad_reply]), Err(ProcessHostError::InvalidPolicy(_))));

    // 2. Mixed-policy fuel_bounded_engine: fuel: Some(...) AND fuel: None (seeded u64::MAX!) side by side!
    let fb_eng = blot_host::fuel_bounded_engine(Duration::from_millis(2))?;
    let fb_mod = GuestModule::new(&fb_eng, &wasm, None)?;
    let mut p_with_fuel = base_policy("fast");
    p_with_fuel.rust_wasmtime_fuel = Some(FuelPolicy { init_fuel: 1_000_000, step_fuel: 1_000_000, cleanup_fuel: 1_000_000 });
    p_with_fuel.deadline = Some(Duration::from_millis(500));
    let mut p_no_fuel = base_policy("slow");
    p_no_fuel.rust_wasmtime_fuel = None; // MUST be seeded u64::MAX on fuel_bounded_engine!
    p_no_fuel.deadline = Some(Duration::from_millis(500));
    let mut mixed = Processes::new(&fb_eng, &fb_mod, vec![p_with_fuel, p_no_fuel])?;
    let AdmitResult::Admitted(hf) = mixed.admit("fast", ProcessValue::U32(10)) else { panic!() };
    let AdmitResult::Admitted(hs) = mixed.admit("slow", ProcessValue::U32(20)) else { panic!() };
    assert_eq!(mixed.advance_all_parallel(2)?, 2);
    assert_eq!(mixed.status(hf), ProcessStatus::Ready);
    assert_eq!(mixed.status(hs), ProcessStatus::Ready);
    let SendResult::Accepted(tf) = mixed.send(hf, ProcessValue::U32(5)) else { panic!() };
    let SendResult::Accepted(ts) = mixed.send(hs, ProcessValue::U32(7)) else { panic!() };
    assert_eq!(mixed.advance_all_parallel(2)?, 2);
    assert_eq!(mixed.poll(hf).unwrap().outcome, ProcessOutcome::Done(ProcessValue::U32(15)));
    assert_eq!(mixed.poll(hs).unwrap().outcome, ProcessOutcome::Done(ProcessValue::U32(27)));
    let _ = (tf, ts);

    // 3. Law #5(c): Simultaneous fuel exhaustion + deadline expiry -> FatalOverrun wins over OutOfFuel!
    // Give enough fuel (~50ms of spinning) that the 3ms deadline fires inside the fuel-metered loop,
    // AND also test exact post-entry Overrun precedence when OutOfFuel returns after deadline_at elapsed.
    let mut sim_p3 = base_policy("slow");
    sim_p3.rust_wasmtime_fuel = Some(FuelPolicy { init_fuel: 5_000_000, step_fuel: 50_000_000, cleanup_fuel: 1_000_000 });
    sim_p3.deadline = Some(Duration::from_millis(3));
    let mut sim_procs3 = Processes::new(&fb_eng, &fb_mod, vec![sim_p3])?;
    let AdmitResult::Admitted(hsim3) = sim_procs3.admit("slow", ProcessValue::U32(0)) else { panic!() };
    assert_eq!(sim_procs3.advance(1)?, 1);
    let _ = sim_procs3.send(hsim3, ProcessValue::U32(999_999));
    let sim_res = sim_procs3.advance(1);
    assert!(matches!(sim_res, Err(ProcessHostError::Overrun { .. })), "expected Overrun to win, got {sim_res:?}");
    assert!(sim_procs3.is_aborted());

    // 4. Law #5(a): Sibling B whose OWN individual deadline is 5000ms in the future is interrupted
    //    promptly via wave_cancel_flag when sibling A (deadline 25ms) overruns!
    //    Also verifies unpolled Done + Cancelled drained_events and detached lease reclamation!
    let b_eng = blot_host::bounded_engine(Duration::from_millis(2))?;
    let b_mod = GuestModule::new(&b_eng, &wasm, None)?;
    let wasm_two_spin = compile_blot(
        r#"
func s_init (x: U32): U32 => return x
func s_step (st: own U32) -> (cmd: U32): (U32, U32) =>
  let acc = st
  if cmd == 999999:
    for ever:
      acc := @u32.add acc 1
  return (@u32.add acc cmd, @u32.add acc cmd)
#process const short_dl: Process [U32, U32, U32] = { init: s_init, step: s_step }
#process const long_dl: Process [U32, U32, U32] = { init: s_init, step: s_step }
#process const done_p: Process [U32, U32, U32] = { init: s_init, step: s_step }
"#,
    )?;
    let b_mod2 = GuestModule::new(&b_eng, &wasm_two_spin, None)?;
    let mut p_short = base_policy("short_dl");
    p_short.deadline = Some(Duration::from_millis(25));
    let mut p_long = base_policy("long_dl");
    p_long.deadline = Some(Duration::from_secs(5)); // 5s live individual deadline!
    let mut p_done = base_policy("done_p");
    p_done.deadline = Some(Duration::from_secs(5));
    let mut wave_procs = Processes::new(&b_eng, &b_mod2, vec![p_short, p_long, p_done])?;
    let AdmitResult::Admitted(hs1) = wave_procs.admit("short_dl", ProcessValue::U32(1)) else { panic!() };
    let AdmitResult::Admitted(hs2) = wave_procs.admit("long_dl", ProcessValue::U32(2)) else { panic!() };
    let AdmitResult::Admitted(hd) = wave_procs.admit("done_p", ProcessValue::U32(10)) else { panic!() };
    assert_eq!(wave_procs.advance_all_parallel(3)?, 3);
    // Produce an unpolled Done on `hd`, then trigger overrun on `hs1` (25ms) while `hs2` (5s) spins!
    let SendResult::Accepted(td) = wave_procs.send(hd, ProcessValue::U32(5)) else { panic!() };
    assert_eq!(wave_procs.advance_all_parallel(1)?, 1);
    let SendResult::Accepted(ts1) = wave_procs.send(hs1, ProcessValue::U32(999_999)) else { panic!() };
    let SendResult::Accepted(ts2) = wave_procs.send(hs2, ProcessValue::U32(999_999)) else { panic!() };
    let t_wave = Instant::now();
    let err = wave_procs.advance_all_parallel(2).unwrap_err();
    assert!(
        t_wave.elapsed() < Duration::from_millis(800),
        "long_dl (5s deadline) sibling must be interrupted via wave_cancel_flag in <800ms, took {:?}",
        t_wave.elapsed()
    );
    let ProcessHostError::Overrun { drained_events } = err else { panic!("expected Overrun, got {err:?}") };
    assert_eq!(drained_events[&hd][0].ticket, td);
    assert_eq!(drained_events[&hd][0].outcome, ProcessOutcome::Done(ProcessValue::U32(15)));
    assert_eq!(drained_events[&hs1][0].ticket, ts1);
    assert_eq!(drained_events[&hs1][0].outcome, ProcessOutcome::Cancelled);
    assert_eq!(drained_events[&hs2][0].ticket, ts2);
    assert_eq!(drained_events[&hs2][0].outcome, ProcessOutcome::Cancelled);
    // Wait briefly for any detached thread to finish exiting and verify all reserved leases drop to 0!
    std::thread::sleep(Duration::from_millis(40));
    assert_eq!(wave_procs.reserved_memory_bytes(), 0);
    assert_eq!(wave_procs.reserved_mailbox_bytes(), 0);
    let _ = &b_mod;

    // 5. Law #5(b): Successful `proc_step` followed by cleanup `Overrun` MUST NOT be masked!
    let spin_clean_proc_wasm = crafted_spin_cleanup_process_wasm();
    let spin_clean_proc_mod = GuestModule::new(&b_eng, &spin_clean_proc_wasm, None)?;
    let mut sc_procs = Processes::new(
        &b_eng,
        &spin_clean_proc_mod,
        vec![ProcessPolicy {
            name: "p".into(),
            init_work: 1,
            step_work: 1,
            rust_wasmtime_fuel: None,
            deadline: Some(Duration::from_millis(30)),
            memory: Limits { memory: 16 << 20 },
            mailbox: 4,
            mailbox_bytes: 256,
            max_reply_bytes: 4,
            slots: 2,
        }],
    )?;
    let AdmitResult::Admitted(hp) = sc_procs.admit("p", ProcessValue::U32(1)) else { panic!() };
    assert_eq!(sc_procs.advance(1)?, 1);
    let _ = sc_procs.send(hp, ProcessValue::U32(2));
    let sc_res = sc_procs.advance(1);
    assert!(matches!(sc_res, Err(ProcessHostError::Overrun { .. })), "cleanup overrun must NOT be masked, got {sc_res:?}");

    // 6. Mailbox-slot invariance on Unit -> Unit (0 bytes, mailbox = 1) + unpolled Done reply_bytes across close!
    let mut unit_pol = base_policy("unit_p");
    unit_pol.mailbox = 1;
    unit_pol.max_reply_bytes = 0;
    let mut u_procs = Processes::new(&eng1, &mod1, vec![unit_pol, base_policy("fast")])?;
    let AdmitResult::Admitted(hu) = u_procs.admit("unit_p", ProcessValue::Unit) else { panic!() };
    let AdmitResult::Admitted(hf2) = u_procs.admit("fast", ProcessValue::U32(5)) else { panic!() };
    assert_eq!(u_procs.advance(2)?, 2);
    let SendResult::Accepted(tu) = u_procs.send(hu, ProcessValue::Unit) else { panic!() };
    assert_eq!(u_procs.send(hu, ProcessValue::Unit), SendResult::Full);
    assert_eq!(u_procs.advance(1)?, 1);
    assert_eq!(u_procs.send(hu, ProcessValue::Unit), SendResult::Full, "unpolled Unit Done still holds mailbox slot");
    assert_eq!(u_procs.poll(hu).unwrap().ticket, tu);
    assert!(matches!(u_procs.send(hu, ProcessValue::Unit), SendResult::Accepted(_)));
    // Unpolled Done on fast retains its 4 logical_reply_bytes across close() until poll()!
    let _ = u_procs.send(hf2, ProcessValue::U32(3));
    assert_eq!(u_procs.advance(1)?, 1);
    u_procs.close(hf2);
    assert_eq!(u_procs.reserved_mailbox_bytes(), 4, "unpolled Done must remain charged across close()");
    assert_eq!(u_procs.poll(hf2).unwrap().outcome, ProcessOutcome::Done(ProcessValue::U32(8)));
    assert_eq!(u_procs.reserved_mailbox_bytes(), 0);

    // 7. Exhaustive 3-endpoint combinatorial `advance_all_parallel` == serial `advance` equivalence!
    for cost_a in [1u32, 2] {
        for cost_b in [1u32, 3] {
            for budget in [1u32, 2, 3, 4, 5] {
                let make = || -> Result<(Processes, [blot_host::processes::ProcessHandle; 3])> {
                    let mut p_fast = base_policy("fast");
                    p_fast.step_work = cost_a;
                    let mut p_slow = base_policy("slow");
                    p_slow.step_work = cost_b;
                    let mut p = Processes::new(&eng1, &mod1, vec![p_fast, p_slow])?;
                    let AdmitResult::Admitted(h0) = p.admit("fast", ProcessValue::U32(1)) else { panic!() };
                    let AdmitResult::Admitted(h1) = p.admit("slow", ProcessValue::U32(2)) else { panic!() };
                    let AdmitResult::Admitted(h2) = p.admit("fast", ProcessValue::U32(3)) else { panic!() };
                    Ok((p, [h0, h1, h2]))
                };
                let (mut serial, hs) = make()?;
                let (mut parallel, hp) = make()?;
                for round in 0..4u32 {
                    if round == 1 {
                        for i in 0..3 {
                            let _ = serial.send(hs[i], ProcessValue::U32(1 + u32::try_from(i).unwrap()));
                            let _ = parallel.send(hp[i], ProcessValue::U32(1 + u32::try_from(i).unwrap()));
                            let _ = serial.send(hs[i], ProcessValue::U32(10 + u32::try_from(i).unwrap()));
                            let _ = parallel.send(hp[i], ProcessValue::U32(10 + u32::try_from(i).unwrap()));
                        }
                    }
                    let s_spent = serial.advance(budget)?;
                    let p_spent = parallel.advance_all_parallel(budget)?;
                    assert_eq!(s_spent, p_spent, "cost_a={cost_a} cost_b={cost_b} budget={budget} round={round}");
                    for i in 0..3 {
                        assert_eq!(serial.status(hs[i]), parallel.status(hp[i]));
                        loop {
                            let se = serial.poll(hs[i]);
                            let pe = parallel.poll(hp[i]);
                            assert_eq!(se, pe, "event mismatch at ep={i} round={round}");
                            if se.is_none() {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
