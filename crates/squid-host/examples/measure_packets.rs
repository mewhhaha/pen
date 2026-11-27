//! Measure repeated two-word array replies using Pen's actual release engine.
//! Inputs are a local Wasm module and a JSON list of scalar packets and expected replies.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::json;
use std::time::Instant;
use wasmtime::{Instance, Module, Store};

#[derive(Deserialize)]
struct Case {
    input: Vec<u32>,
    expected: Option<[u32; 2]>,
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(args.len() == 5, "measure_packets WASM CASES_JSON AOT_OUT WARMUPS SAMPLES");
    let bytes = std::fs::read(&args[0])?;
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let warmups = args[3].parse::<usize>()?;
    let measured = args[4].parse::<usize>()?;
    ensure!(!cases.is_empty() && measured > 0, "expected cases and measurements");
    let start = Instant::now();
    let engine = squid_host::engine()?;
    let engine_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let start = Instant::now();
    let compiled = engine.precompile_module(&bytes)?;
    let compile_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    std::fs::write(&args[2], &compiled)?;
    let start = Instant::now();
    // SAFETY: these bytes were compiled above using this exact engine.
    let module = unsafe { Module::deserialize(&engine, &compiled)? };
    let load_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let start = Instant::now();
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;
    let memory = instance
        .get_memory(&mut store, "squid:memory")
        .or_else(|| instance.get_memory(&mut store, "memory"))
        .context("missing memory")?;
    let reset = instance.get_typed_func::<u32, u32>(&mut store, "squid:reset")?;
    let alloc = instance.get_typed_func::<u32, u32>(&mut store, "squid:allocate")?;
    let solve = instance.get_typed_func::<u32, u32>(&mut store, "solve")?;
    let finish = instance.get_typed_func::<(), u32>(&mut store, "squid:finish_invocation")?;
    let instantiate_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let mut records = Vec::new();
    for (index, case) in cases.iter().enumerate() {
        let words = u32::try_from(case.input.len())?;
        let input: Vec<u8> = std::iter::once(&words).chain(&case.input).flat_map(|word| word.to_le_bytes()).collect();
        for repeat in 0..=warmups + measured {
            let start = Instant::now();
            reset.call(&mut store, 0)?;
            let ptr = alloc.call(&mut store, input.len().try_into()?)?;
            memory.write(&mut store, ptr as usize, &input)?;
            let prepared = Instant::now();
            let result = solve.call(&mut store, ptr);
            let solved = Instant::now();
            match (result, case.expected) {
                (Ok(out), Some(expected)) => {
                    let mut reply = [0; 12];
                    memory.read(&store, out as usize, &mut reply)?;
                    let values: Vec<_> = reply.as_chunks::<4>().0.iter().map(|&v| u32::from_le_bytes(v)).collect();
                    ensure!(
                        values == [2, expected[0], expected[1]],
                        "case {index} got {values:?}, wanted {expected:?}"
                    );
                }
                (Err(err), None) => {
                    ensure!(err.downcast_ref::<wasmtime::Trap>().is_some(), "expected a guest trap: {err}")
                }
                (other, expected) => anyhow::bail!("case {index} got {other:?}, wanted {expected:?}"),
            }
            finish.call(&mut store, ())?;
            let end = Instant::now();
            if repeat == 0 || repeat > warmups {
                records.push(json!([
                    index,
                    repeat,
                    u64::try_from((prepared - start).as_nanos()).unwrap_or(u64::MAX),
                    u64::try_from((solved - prepared).as_nanos()).unwrap_or(u64::MAX),
                    u64::try_from((end - solved).as_nanos()).unwrap_or(u64::MAX)
                ]));
            }
        }
    }
    let rss = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:"))?.split_whitespace().nth(1)?.parse::<u64>().ok());
    println!(
        "{}",
        json!({"engine_ns":engine_ns,"aot_compile_ns":compile_ns,
        "cold_deserialize_ns":load_ns,"instantiate_and_bind_ns":instantiate_ns,"aot_bytes":compiled.len(),
        "wasm_bytes":bytes.len(),"guest_linear_bytes":memory.data_size(&store),"peak_rss_kib":rss,
        "samples":records})
    );
    Ok(())
}
