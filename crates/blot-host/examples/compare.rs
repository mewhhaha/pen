//! Compare U32 -> Array F32 kernels with identical output, through the native host.
//! compare MODULE.wasm WORKLOAD.json OUTPUT.json
use anyhow::{Context, Result, ensure};
use blot_host::{Guest, GuestModule, Value};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(args.len() == 3, "compare MODULE.wasm WORKLOAD.json OUTPUT.json");
    let workload: Json = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let engine = blot_host::engine()?;
    let bytes = std::fs::read(&args[0])?;
    let wasm = format!("{:x}", Sha256::digest(&bytes));
    ensure!(workload["wasm"].as_str() == Some(&wasm), "workload identifies a different Wasm module");
    let module = GuestModule::new(&engine, &bytes, None)?;
    let mut guest = Guest::new(&engine, &module)?;
    let mut rows = vec![];
    for row in workload["rows"].as_array().context("workload rows")? {
        let names: Vec<_> = row["names"]
            .as_array()
            .context("names")?
            .iter()
            .map(|name| name.as_str().context("name"))
            .collect::<Result<_>>()?;
        let size = u32::try_from(row["size"].as_u64().context("size")?)?;
        let mut reference = None;
        for name in &names {
            let actual = guest.call(name, Value::U32(size))?.into_f32_array()?;
            let bits: Vec<_> = actual.iter().map(|value| value.to_bits()).collect();
            if let Some(previous) = &reference {
                ensure!(*previous == bits, "{name}: changed output");
            } else {
                reference = Some(bits);
            }
            for _ in 0..30 {
                guest.call(name, Value::U32(size))?;
            }
        }
        let mut samples = vec![vec![]; names.len()];
        for pair in 0..11 {
            let mut order: Vec<_> = (0..names.len()).collect();
            if pair % 2 != 0 {
                order.reverse();
            }
            for index in order {
                let start = Instant::now();
                let mut last = Vec::new();
                for _ in 0..50 {
                    last = guest.call(names[index], Value::U32(size))?.into_f32_array()?;
                }
                samples[index].push(start.elapsed().as_secs_f64() * 1000.0 / 50.0);
                let bits: Vec<_> = last.iter().map(|value| value.to_bits()).collect();
                ensure!(reference.as_ref() == Some(&bits), "{}: output changed during timing", names[index]);
            }
        }
        let medians: Vec<_> = samples
            .iter()
            .map(|sample| {
                let mut sorted = sample.clone();
                sorted.sort_by(f64::total_cmp);
                sorted[5]
            })
            .collect();
        let ratios: Vec<_> = medians.iter().map(|value| value / medians[0]).collect();
        rows.push(json!({"names": names, "size": size, "samplesMs": samples, "medianMs": medians, "ratios": ratios}));
    }
    guest.dispose()?;
    std::fs::write(
        &args[2],
        serde_json::to_vec_pretty(&json!({
            "host": "Pen blot-host / Wasmtime; includes inputs and output copying", "rows": rows,
            "compiler": workload["compiler"], "wasm": wasm,
        }))?,
    )?;
    Ok(())
}
