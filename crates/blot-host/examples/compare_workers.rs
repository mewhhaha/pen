//! Measure independent value jobs using one versus four isolated CPU workers.
use anyhow::{Result, ensure};
use blot_host::{GuestModule, Value, workers::Workers};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::time::Instant;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(args.len() == 4, "compare_workers MODULE.wasm ENTRY SIZE OUTPUT.json");
    let size: u32 = args[2].parse()?;
    let bytes = std::fs::read(&args[0])?;
    let wasm = format!("{:x}", Sha256::digest(&bytes));
    let engine = blot_host::engine()?;
    let module = GuestModule::new(&engine, &bytes, None)?;
    let mut pools = [Workers::new(&engine, &module, &args[1], 1)?, Workers::new(&engine, &module, &args[1], 4)?];
    let inputs: Vec<_> = (0..32).map(|i| Value::U32(size + i)).collect();
    let mut reference = None;
    for pool in &mut pools {
        for _ in 0..5 {
            let values: Vec<_> = pool
                .run(inputs.clone())?
                .into_iter()
                .collect::<Result<_, _>>()
                .map_err(|error| anyhow::anyhow!(error))?;
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
            let values = pools[index].run(inputs.clone())?;
            samples[index].push(start.elapsed().as_secs_f64() * 1000.0);
            let values: Vec<_> =
                values.into_iter().collect::<Result<_, _>>().map_err(|error| anyhow::anyhow!(error))?;
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
    // Keep timing diagnostics out of the paired performance samples.
    let mut timelines = vec![];
    for pool in &mut pools {
        pool.set_tracing(true);
        let values: Vec<_> =
            pool.run(inputs.clone())?.into_iter().collect::<Result<_, _>>().map_err(|error| anyhow::anyhow!(error))?;
        ensure!(reference.as_ref() == Some(&values), "tracing changed worker results");
        timelines.push(
            pool.timings()
                .iter()
                .map(|timing| {
                    json!({
                        "job": timing.job, "worker": timing.worker,
                        "submittedMs": timing.submitted.as_secs_f64() * 1000.0,
                        "startedMs": timing.started.as_secs_f64() * 1000.0,
                        "finishedMs": timing.finished.as_secs_f64() * 1000.0,
                        "succeeded": timing.succeeded,
                    })
                })
                .collect::<Vec<_>>(),
        );
    }
    for pool in &mut pools {
        pool.close()?;
    }
    std::fs::write(
        &args[3],
        serde_json::to_vec_pretty(&json!({"host":"Pen isolated Wasmtime worker instances",
        "wasm":wasm,"entry":args[1], "size":size,"jobs":32,"workers":[1,4],"samplesMs":samples,"medianMs":median,
        "timelines":timelines,"timelineIncludes":"separate traced batch, not included in paired samples; queue wait and fresh instance execution/teardown",
        "timingContext":std::env::var("PEN_BENCH_CONTEXT").unwrap_or_else(|_| "not recorded".into()),
        "ratio":median[1]/median[0],"includes":"fresh instance, copied input/output, work, teardown; excludes shared module compilation"}))?,
    )?;
    Ok(())
}
