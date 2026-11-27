//! Measure independent kernel batches using one versus four warm `KernelPool` workers.
use anyhow::{Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};
use squid_host::{
    GuestModule,
    kernels::{KernelConfig, KernelPool},
};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(args.len() == 4, "compare_workers MODULE.wasm KERNEL_NAME SIZE OUTPUT.json");
    let size: u32 = args[2].parse()?;
    let bytes = std::fs::read(&args[0])?;
    let wasm = format!("{:x}", Sha256::digest(&bytes));
    let engine = squid_host::engine()?;
    let module = GuestModule::new(&engine, &bytes, None)?;
    let mut pools = [
        KernelPool::open(
            &engine,
            &module,
            &args[1],
            KernelConfig { workers: 1, default_chunk_items: 4, ..KernelConfig::default() },
        )?,
        KernelPool::open(
            &engine,
            &module,
            &args[1],
            KernelConfig { workers: 4, default_chunk_items: 4, ..KernelConfig::default() },
        )?,
    ];
    let inputs: Vec<u32> = (0..32).map(|i| size + i).collect();
    let mut reference = None;
    for pool in &mut pools {
        for _ in 0..5 {
            let values = pool.run_fixed(&inputs)?;
            if let Some(previous) = &reference {
                ensure!(*previous == values, "worker outputs differ");
            } else {
                reference = Some(values);
            }
        }
    }
    let mut samples = [vec![], vec![]];
    for pair in 0..11 {
        for index in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
            let start = Instant::now();
            let values = pools[index].run_fixed(&inputs)?;
            samples[index].push(start.elapsed().as_secs_f64() * 1000.0);
            ensure!(reference.as_ref() == Some(&values), "worker outputs changed during timing");
        }
    }
    let median: Vec<_> = samples
        .iter()
        .map(|sample| {
            let mut x = sample.clone();
            x.sort_by(f64::total_cmp);
            x[5]
        })
        .collect();
    for pool in &mut pools {
        pool.close();
    }
    std::fs::write(
        &args[3],
        serde_json::to_vec_pretty(&json!({
            "host": "Pen warm Wasmtime KernelPool",
            "wasm": wasm,
            "kernel": args[1],
            "size": size,
            "items": 32,
            "workers": [1, 4],
            "samplesMs": samples,
            "medianMs": median,
            "timingContext": std::env::var("PEN_BENCH_CONTEXT").unwrap_or_else(|_| "not recorded".into()),
            "ratio": median[1] / median[0],
            "includes": "warm worker pool, copied input/output, work, cleanup; excludes shared module compilation"
        }))?,
    )?;
    Ok(())
}
