//! Compare native route kernel performance across direct (`workers = 0`) and warm
//! multi-core `KernelPool` (`workers = 1, 2, 4, 8`) on an ABI 8 `#kernel` artifact.
use anyhow::{Context, Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};
use squid_host::{
    GuestModule,
    kernels::{KernelConfig, KernelPool},
};
use std::time::Instant;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().context("compare_routes MODULE.wasm [KERNEL_NAME]")?;
    let kernel_name = args.next().unwrap_or_else(|| "routes".to_owned());
    let wasm = std::fs::read(path)?;
    let engine = squid_host::engine()?;
    let module = GuestModule::new(&engine, &wasm, None)?;
    let manifest = module.abi().kernel(&kernel_name).context("missing #kernel")?.clone();
    let variable = manifest.variable_output();

    let mut results = Vec::new();
    for workers in [0usize, 1, 2, 4, 8] {
        let started = Instant::now();
        let mut pool = KernelPool::open(
            &engine,
            &module,
            &kernel_name,
            KernelConfig { workers, default_chunk_items: 8, ..KernelConfig::default() },
        )?;
        if let Some(ctx_schema) = &manifest.ctx_schema {
            let ctx_words = vec![11u32; (ctx_schema.stride as usize).max(1)];
            pool.set_context(1, &ctx_words)?;
        }
        let setup_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut cells = Vec::new();
        for size in [4usize, 32, 128] {
            let stride = manifest.in_schema.stride as usize;
            let input: Vec<u32> = (0..size * stride).map(|i| u32::try_from(i % 4).unwrap()).collect();
            let first = if variable { pool.run_variable(&input)? } else { pool.run_fixed(&input)? };
            let mut times_ms = Vec::new();
            for repeat in 0..25 {
                let t0 = Instant::now();
                let out = if variable { pool.run_variable(&input)? } else { pool.run_fixed(&input)? };
                let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
                ensure!(out == first, "kernel output changed across repeats");
                if repeat >= 4 {
                    times_ms.push(elapsed);
                }
            }
            cells.push(json!({
                "size": size,
                "workers": workers,
                "output_words": first.len(),
                "times_ms": times_ms,
            }));
        }
        let stats = pool.stats();
        pool.close();
        results.push(json!({
            "workers": workers,
            "setup_ms": setup_ms,
            "worker_starts": stats.worker_starts,
            "peak_charged_bytes": stats.peak_charged_bytes,
            "cells": cells,
        }));
    }
    println!(
        "{}",
        json!({
            "wasm_sha256": format!("{:x}", Sha256::digest(wasm)),
            "kernel": kernel_name,
            "scope": "Native warm KernelPool across workers=0,1,2,4,8; no renderer.",
            "results": results,
        })
    );
    Ok(())
}
