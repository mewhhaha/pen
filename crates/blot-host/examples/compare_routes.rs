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

fn direct(guest: &mut Guest, rows: &[u32]) -> Result<Vec<f32>> {
    let result = if rows.len() == 4 {
        guest.call("serial_shared", Value::U32(SEED))?
    } else {
        guest.call("routes", packet(rows))?
    };
    result.into_f32_array()
}

fn jobs(rows: &[u32], workers: usize) -> Vec<Value> {
    let count = rows.len().min(workers).max(rows.len().div_ceil(ROWS_PER_JOB));
    (0..count).map(|index| packet(&rows[index * rows.len() / count..(index + 1) * rows.len() / count])).collect()
}

fn batch(pool: &mut Workers, rows: &[u32], workers: usize) -> Result<Vec<f32>> {
    let mut packed = Vec::with_capacity(rows.len() * WIDTH);
    for reply in pool.run(jobs(rows, workers))? {
        packed.extend(reply.map_err(anyhow::Error::msg)?.into_f32_array()?);
    }
    ensure!(packed.len() == rows.len() * WIDTH, "wrong route length");
    if rows.len() != 4 {
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
    let path = std::env::args().nth(1).context("compare_routes MODULE.wasm")?;
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
        for size in [4, 32, 128] {
            let rows: Vec<u32> = (0..size).map(|i| i % 4).collect();
            cells.push(measure(&rows, workers, &mut direct_guest, &mut pool)?);
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
