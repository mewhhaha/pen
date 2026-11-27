//! Compare a direct serial `KernelPool` (`workers = 0`) with 1-worker and 4-worker warm pools.
//! Input is a JSON array of scenarios: `{ label, inputs: [[finite F32, ...], ...] }`.
use anyhow::{Context, Result, ensure};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use squid_host::{
    GuestModule,
    kernels::{KernelConfig, KernelPool},
};
use std::time::Instant;

fn run_pool(pool: &mut KernelPool, inputs: &[u32]) -> Result<Vec<u32>> {
    let out = pool.run_fixed(inputs)?;
    ensure!(out.iter().all(|bits| f32::from_bits(*bits).is_finite()), "nonfinite kernel output");
    Ok(out)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(args.len() == 4, "compare_worker_batches MODULE.wasm KERNEL_NAME WORKLOAD.json OUTPUT.json");
    let bytes = std::fs::read(&args[0])?;
    let workload_bytes = std::fs::read(&args[2])?;
    let workload: Json = serde_json::from_slice(&workload_bytes)?;
    let engine = squid_host::engine()?;
    let module = GuestModule::new(&engine, &bytes, None)?;
    let mut pools = [
        KernelPool::open(&engine, &module, &args[1], KernelConfig { workers: 0, ..KernelConfig::default() })?,
        KernelPool::open(&engine, &module, &args[1], KernelConfig { workers: 1, ..KernelConfig::default() })?,
        KernelPool::open(&engine, &module, &args[1], KernelConfig { workers: 4, ..KernelConfig::default() })?,
    ];
    let mut rows = vec![];
    for scenario in workload.as_array().context("workload scenarios")? {
        let batches = scenario["inputs"].as_array().context("inputs")?;
        let mut flat: Vec<u32> = Vec::new();
        for input in batches {
            let lanes = input.as_array().context("input lanes")?;
            for lane in lanes {
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "guest input lanes are f32; non-finite rejected below"
                )]
                let value = lane.as_f64().context("numeric lane")? as f32;
                ensure!(value.is_finite(), "nonfinite input");
                flat.push(value.to_bits());
            }
        }
        let reference = run_pool(&mut pools[0], &flat)?;
        for pool in &mut pools {
            for _ in 0..5 {
                ensure!(reference == run_pool(pool, &flat)?, "warmup output differs");
            }
        }
        let mut samples = [vec![], vec![], vec![]];
        for pair in 0..11 {
            let order = if pair % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
            for mode in order {
                let start = Instant::now();
                for _ in 0..5 {
                    ensure!(reference == run_pool(&mut pools[mode], &flat)?, "measured output differs");
                }
                samples[mode].push(start.elapsed().as_secs_f64() * 1000.0 / 5.0);
            }
        }
        let medians: Vec<_> = samples
            .iter()
            .map(|values| {
                let mut ordered = values.clone();
                ordered.sort_by(f64::total_cmp);
                ordered[5]
            })
            .collect();
        let mut output_hash = Sha256::new();
        for lane in &reference {
            output_hash.update(lane.to_le_bytes());
        }
        rows.push(json!({
            "outputSha256": format!("{:x}", output_hash.finalize()),
            "label": scenario["label"],
            "outputLanes": reference.len(),
            "medianMs": medians,
            "samplesMs": samples,
        }));
    }
    for pool in &mut pools {
        pool.close();
    }
    std::fs::write(
        &args[3],
        serde_json::to_vec_pretty(&json!({
            "wasm": format!("{:x}", Sha256::digest(&bytes)),
            "workloadSha256": format!("{:x}", Sha256::digest(&workload_bytes)),
            "kernel": args[1],
            "modes": ["direct serial (workers=0)", "one warm worker", "four warm workers"],
            "rows": rows,
        }))?,
    )?;
    Ok(())
}
