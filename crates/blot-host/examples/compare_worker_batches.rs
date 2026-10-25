//! Compare a persistent serial guest with copied, bounded CPU batches.
//! Input is a JSON array of scenarios: { label, inputs: [[finite F32, ...], ...] }.
use anyhow::{Context, Result, ensure};
use blot_host::{Guest, GuestModule, Value, workers::Workers};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::time::Instant;

fn run(
    mode: usize,
    entry: &str,
    inputs: &[Value],
    direct_input: &Value,
    guest: &mut Guest,
    pools: &mut [Workers; 2],
) -> Result<Vec<u32>> {
    let answers = if mode == 0 {
        inputs.iter().map(|input| guest.call(entry, input.clone())).collect::<Result<Vec<_>>>()?
    } else if mode == 3 {
        vec![guest.call(entry, direct_input.clone())?]
    } else {
        pools[mode - 1]
            .run(inputs.to_vec())?
            .into_iter()
            .map(|answer| answer.map_err(anyhow::Error::msg))
            .collect::<Result<Vec<_>>>()?
    };
    let chunks = answers
        .into_iter()
        .map(|value| {
            let lanes = value.into_f32_array()?;
            ensure!(lanes.iter().all(|x| x.is_finite()), "nonfinite worker output");
            Ok(lanes.into_iter().map(f32::to_bits).collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(chunks.into_iter().flatten().collect())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(args.len() == 4, "compare_worker_batches MODULE.wasm ENTRY WORKLOAD.json OUTPUT.json");
    let bytes = std::fs::read(&args[0])?;
    let workload_bytes = std::fs::read(&args[2])?;
    let workload: Json = serde_json::from_slice(&workload_bytes)?;
    let engine = blot_host::engine()?;
    let module = GuestModule::new(&engine, &bytes, None)?;
    let mut guest = Guest::new(&engine, &module)?;
    let mut pools = [Workers::new(&engine, &module, &args[1], 1)?, Workers::new(&engine, &module, &args[1], 4)?];
    let mut rows = vec![];
    for scenario in workload.as_array().context("workload scenarios")? {
        let batches = scenario["inputs"].as_array().context("inputs")?;
        ensure!(batches.len() <= 32, "at most 32 jobs per batch");
        let inputs = batches
            .iter()
            .map(|input| {
                let lanes = input.as_array().context("input lanes")?;
                ensure!(lanes.len() <= 16384, "input lane budget exceeded");
                let lanes = lanes
                    .iter()
                    .map(|lane| {
                        #[allow(
                            clippy::cast_possible_truncation,
                            reason = "guest input lanes are f32; overflow is rejected below"
                        )]
                        let value = lane.as_f64().context("numeric lane")? as f32;
                        ensure!(value.is_finite(), "nonfinite input");
                        Ok(value)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Value::ArrayF32(lanes))
            })
            .collect::<Result<Vec<_>>>()?;
        let direct_input = Value::ArrayF32(
            inputs
                .iter()
                .flat_map(|input| match input {
                    Value::ArrayF32(lanes) => lanes.clone(),
                    _ => unreachable!(),
                })
                .collect(),
        );
        let reference = run(0, &args[1], &inputs, &direct_input, &mut guest, &mut pools)?;
        for mode in 0..4 {
            for _ in 0..5 {
                ensure!(
                    reference == run(mode, &args[1], &inputs, &direct_input, &mut guest, &mut pools)?,
                    "warmup output differs"
                );
            }
        }
        let mut samples = [vec![], vec![], vec![], vec![]];
        for pair in 0..11 {
            let order = if pair % 2 == 0 { [0, 1, 2, 3] } else { [3, 2, 1, 0] };
            for mode in order {
                let start = Instant::now();
                for _ in 0..5 {
                    ensure!(
                        reference == run(mode, &args[1], &inputs, &direct_input, &mut guest, &mut pools)?,
                        "measured output differs"
                    );
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
        let mut timelines = vec![];
        for (index, pool) in pools.iter_mut().enumerate() {
            pool.set_tracing(true);
            let answers = pool.run(inputs.clone())?;
            ensure!(answers.iter().all(Result::is_ok), "traced batch failed");
            timelines.push(
                json!({"workers": if index == 0 { 1 } else { 4 }, "jobs": pool.timings().iter().map(|t| json!({
                "job": t.job, "worker": t.worker, "submittedMs": t.submitted.as_secs_f64() * 1000.0,
                "startedMs": t.started.as_secs_f64() * 1000.0, "finishedMs": t.finished.as_secs_f64() * 1000.0,
            })).collect::<Vec<_>>() }),
            );
            pool.set_tracing(false);
        }
        let mut output_hash = Sha256::new();
        for lane in &reference {
            output_hash.update(lane.to_le_bytes());
        }
        rows.push(json!({"outputSha256": format!("{:x}", output_hash.finalize()), "label": scenario["label"], "jobs": inputs.len(), "outputLanes": reference.len(),
            "medianMs": medians, "samplesMs": samples, "timelines": timelines}));
    }
    guest.dispose()?;
    for pool in &mut pools {
        pool.close()?;
    }
    std::fs::write(
        &args[3],
        serde_json::to_vec_pretty(&json!({
            "wasm": format!("{:x}", Sha256::digest(&bytes)), "workloadSha256": format!("{:x}", Sha256::digest(&workload_bytes)), "entry": args[1], "modes": ["persistent serial guest", "one isolated worker", "four isolated workers", "one direct serial call"],
            "includes": "input/output copies, fresh worker instances, ordered batch barrier; excludes shared module compilation and pool creation",
            "rows": rows,
        }))?,
    )?;
    Ok(())
}
