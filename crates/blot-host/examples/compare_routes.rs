//! Compare native route stages on the same ABI 8 artifact as the Blot host
//! benchmark. One-shot workers use a fresh sandboxed guest for each job.
use anyhow::{Context, Result, ensure};
use blot_host::{Guest, GuestModule, Value, workers::Workers};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::time::Instant;

const SEED: u32 = 11;
const WIDTH: usize = 392;
// Pen's copied value worker allows at most 16,384 returned numeric words.
const ROWS_PER_JOB: usize = 16_384 / WIDTH;

fn packet(rows: &[u32]) -> Value {
    Value::ArrayU32(std::iter::once(SEED).chain(rows.iter().copied()).collect())
}

fn direct_argument(rows: &[u32]) -> (&'static str, Value) {
    if rows.len() == 4 { ("serial_shared", Value::U32(SEED)) } else { ("routes", packet(rows)) }
}

fn direct(guest: &mut Guest, rows: &[u32]) -> Result<Vec<f32>> {
    let (entry, input) = direct_argument(rows);
    guest.call(entry, input)?.into_f32_array()
}

fn jobs(rows: &[u32], workers: usize) -> Vec<Value> {
    let count = rows.len().min(workers).max(rows.len().div_ceil(ROWS_PER_JOB));
    (0..count).map(|index| packet(&rows[index * rows.len() / count..(index + 1) * rows.len() / count])).collect()
}

fn batch(pool: &mut Workers, rows: &[u32], workers: usize) -> Result<Vec<f32>> {
    let mut packed = Vec::with_capacity(rows.len() * WIDTH);
    assemble(&mut packed, pool.run(jobs(rows, workers))?)?;
    apply(packed, rows.len())
}

fn assemble(packed: &mut Vec<f32>, replies: Vec<Result<Value, String>>) -> Result<()> {
    for reply in replies {
        packed.extend(reply.map_err(anyhow::Error::msg)?.into_f32_array()?);
    }
    Ok(())
}

fn apply(packed: Vec<f32>, count: usize) -> Result<Vec<f32>> {
    ensure!(packed.len() == count * WIDTH, "wrong route length");
    if count != 4 {
        return Ok(packed);
    }
    // Gdev prefers arrival, then remaining distance when no route arrives,
    // then traveled distance. Equal candidates keep the earlier index.
    let mut winner = 0;
    let mut result = vec![0.0];
    for (i, route) in packed.chunks_exact(WIDTH).enumerate() {
        let prior = &packed[winner * WIDTH..(winner + 1) * WIDTH];
        let better = if route[1] != prior[1] {
            route[1] > prior[1]
        } else if route[1] == 0.0 && route[4] != prior[4] {
            route[4] < prior[4]
        } else {
            route[3] < prior[3]
        };
        if better {
            winner = i;
        }
        let words =
            (1..WIDTH).find(|n| f32::from(u16::try_from(*n).unwrap()) == route[0]).context("invalid route length")?;
        result.extend_from_slice(&route[1..words + 1]);
    }
    result[0] = f32::from(u16::try_from(winner)?);
    Ok(result)
}

fn equal(found: &[f32], expected: &[f32]) -> Result<()> {
    ensure!(
        found.len() == expected.len() && found.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()),
        "route payload or application result changed"
    );
    Ok(())
}

fn copies(inputs: usize, outputs: usize, snapshot: usize, assembly: usize, application: usize) -> serde_json::Value {
    json!({
        "host_snapshot_query": 4 * snapshot, "host_to_wasm": 4 * inputs, "wasm_unpack": 4 * inputs,
        "wasm_pack": 4 * outputs, "wasm_to_host": 4 * outputs, "host_assembly": 4 * assembly,
        "host_application": 4 * application
    })
}

fn direct_stages(guest: &mut Guest, rows: &[u32], expected: &[f32]) -> Result<serde_json::Value> {
    let started = Instant::now();
    let (entry, input) = direct_argument(rows);
    let input_words = match &input {
        Value::ArrayU32(words) => words.len(),
        _ => 0,
    };
    let call = Instant::now();
    let result = guest.call(entry, input)?.into_f32_array()?;
    let done = Instant::now();
    equal(&result, expected)?;
    let snapshot_words = if rows.len() == 4 { 0 } else { rows.len() };
    Ok(json!({
        "caller_snapshot_ns": (call - started).as_nanos(),
        "fused_call_ns": (done - call).as_nanos(),
        "total_ns": (done - started).as_nanos(),
        "copies_bytes": copies(input_words, result.len(), snapshot_words, 0, 0)
    }))
}

fn worker_stages(
    guest: &mut Guest,
    pool: &mut Workers,
    rows: &[u32],
    workers: usize,
    expected: &[f32],
    inject_trap: bool,
) -> Result<serde_json::Value> {
    let started = Instant::now();
    let mut packed = Vec::with_capacity(rows.len() * WIDTH);
    let snapshot = Instant::now();
    let mut inputs = jobs(rows, workers);
    let job_count = inputs.len();
    let mut input_words = rows.len() + job_count;
    if inject_trap {
        // Reach a real Wasm bounds trap after admission. The saved query
        // slice stays immutable and still produces its entire direct answer.
        let Value::ArrayU32(input) = &mut inputs[0] else { unreachable!() };
        input_words -= input.len();
        input.clear();
    }
    let dispatch = Instant::now();
    let replies = pool.run(inputs)?;
    let joined = Instant::now();
    let count_failed = replies.iter().filter(|reply| reply.is_err()).count();
    ensure!(count_failed == usize::from(inject_trap), "injected guest failure count changed");
    let output_words = replies
        .iter()
        .map(|reply| match reply {
            Ok(Value::ArrayF32(words)) => words.len(),
            _ => 0,
        })
        .sum();
    let (assembly_ns, application_ns, recovery_ns, result) = if inject_trap {
        let failed = replies[0].as_ref().expect_err("first job trapped");
        ensure!(failed.contains("wasm trap"), "unexpected injected failure: {failed}");
        let recovery = Instant::now();
        drop(replies);
        let recovered = direct(guest, rows)?;
        (0, 0, recovery.elapsed().as_nanos(), recovered)
    } else {
        let assembly = Instant::now();
        assemble(&mut packed, replies)?;
        let application = Instant::now();
        let result = apply(packed, rows.len())?;
        ((application - assembly).as_nanos(), application.elapsed().as_nanos(), 0, result)
    };
    let done = Instant::now();
    equal(&result, expected)?;
    let timing = pool.batch_timing().context("missing worker caller timings")?;
    let job_times = pool
        .timings()
        .iter()
        .map(|job| {
            json!({
                "job": job.job, "worker": job.worker, "succeeded": job.succeeded,
                "queued_ns": (job.started - job.submitted).as_nanos(),
                "sandbox_call_and_dispose_ns": (job.finished - job.started).as_nanos()
            })
        })
        .collect::<Vec<_>>();
    let assembly_words = if inject_trap { 0 } else { output_words };
    let application_words = if !inject_trap && rows.len() == 4 { result.len() - 1 } else { 0 };
    let recovery_copies = if inject_trap {
        let (inputs, snapshot) = if rows.len() == 4 { (0, 0) } else { (rows.len() + 1, rows.len()) };
        Some(copies(inputs, result.len(), snapshot, 0, 0))
    } else {
        None
    };
    Ok(json!({
        "caller_snapshot_ns": (dispatch - snapshot).as_nanos(),
        "admission_ns": timing.admission.as_nanos(),
        "caller_receive_ns": timing.receive.as_nanos(),
        "run_ns": (joined - dispatch).as_nanos(),
        "assembly_ns": assembly_ns, "application_ns": application_ns,
        "recovery_ns": recovery_ns, "failed_jobs": count_failed,
        "total_ns": (done - started).as_nanos(),
        "copies_bytes": copies(input_words, output_words, rows.len(), assembly_words, application_words),
        "recovery_copies_bytes": recovery_copies,
        "jobs": job_times
    }))
}

fn stage_samples(rows: &[u32], workers: usize, guest: &mut Guest, pool: &mut Workers) -> Result<serde_json::Value> {
    let expected = direct(guest, rows)?;
    let mut direct_sample = vec![];
    let mut worker_sample = vec![];
    let mut recovery_sample = vec![];
    for repeat in 0..15 {
        for pass in 0..2 {
            if (repeat + pass) % 2 == 0 {
                let sample = direct_stages(guest, rows, &expected)?;
                if repeat >= 4 {
                    direct_sample.push(sample);
                }
            } else {
                let sample = worker_stages(guest, pool, rows, workers, &expected, false)?;
                if repeat >= 4 {
                    worker_sample.push(sample);
                }
            }
        }
    }
    if rows.len() == 4 {
        for repeat in 0..15 {
            let sample = worker_stages(guest, pool, rows, workers, &expected, true)?;
            if repeat >= 4 {
                recovery_sample.push(sample);
            }
        }
        equal(&batch(pool, rows, workers)?, &expected)?;
    }
    Ok(json!({
        "size": rows.len(), "workers": workers,
        "direct": direct_sample, "copied": worker_sample, "forced_trap": recovery_sample
    }))
}

fn measure(rows: &[u32], workers: usize, guest: &mut Guest, pool: &mut Workers) -> Result<serde_json::Value> {
    let started = Instant::now();
    let expected = direct(guest, rows)?;
    let direct_first_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let first = batch(pool, rows, workers)?;
    let copied_first_ms = started.elapsed().as_secs_f64() * 1000.0;
    equal(&first, &expected)?;
    let mut sample = [Vec::new(), Vec::new()];
    for repeat in 0..25 {
        for pass in 0..2 {
            let mode = (repeat + pass) % 2;
            let started = Instant::now();
            let found = if mode == 0 { direct(guest, rows)? } else { batch(pool, rows, workers)? };
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            equal(&found, &expected)?;
            if repeat >= 4 {
                sample[mode].push(elapsed);
            }
        }
    }
    let [direct_ms, copied_ms] = sample;
    Ok(json!({
        "size": rows.len(), "workers": workers, "jobs": jobs(rows, workers).len(),
        "direct_first_ms": direct_first_ms, "copied_first_ms": copied_first_ms,
        "direct_ms": direct_ms, "copied_ms": copied_ms
    }))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().context("compare_routes MODULE.wasm [--stages]")?;
    let stages = match args.next().as_deref() {
        None => false,
        Some("--stages") => true,
        Some(option) => anyhow::bail!("unknown option {option}"),
    };
    let wasm = std::fs::read(path)?;
    let engine = blot_host::engine()?;
    let module = GuestModule::new(&engine, &wasm, None)?;
    let mut results = Vec::new();
    for workers in [1, 2, 4, 8] {
        let mut direct_guest = Guest::new(&engine, &module)?;
        let started = Instant::now();
        let mut pool = Workers::new(&engine, &module, "routes", workers)?;
        let setup_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut cells = Vec::new();
        if stages {
            pool.set_tracing(true);
            for size in [4, 128] {
                let rows: Vec<u32> = (0..size).map(|i| i % 4).collect();
                cells.push(stage_samples(&rows, workers, &mut direct_guest, &mut pool)?);
            }
        } else {
            for size in [4, 32, 128] {
                let rows: Vec<u32> = (0..size).map(|i| i % 4).collect();
                cells.push(measure(&rows, workers, &mut direct_guest, &mut pool)?);
            }
        }
        for rows in [vec![], vec![3, 99, 3, 1, 0]] {
            equal(&batch(&mut pool, &rows, workers)?, &direct(&mut direct_guest, &rows)?)?;
        }
        pool.close()?;
        results.push(json!({
            "setup_ms": setup_ms, "persistent_direct_bytes": direct_guest.memory_bytes(), "cells": cells
        }));
    }
    println!(
        "{}",
        json!({
            "wasm_sha256": format!("{:x}", Sha256::digest(wasm)),
            "scope": "Native direct persistent guest and cold isolated guest per copied job; route stages, no renderer.",
            "results": results,
        })
    );
    Ok(())
}
