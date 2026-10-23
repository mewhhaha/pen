//! Times compiling Blot programs one after another, as development does:
//!   cargo run --release -p blot-host --example compile_time A.wasm B.wasm ...
//! Use `--engine plain|incremental` to isolate a process's peak memory, and
//! `--cache DIR` to compare a cold module compilation with cached loading.
use std::path::PathBuf;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut selected = "both".to_owned();
    let mut cache: Option<PathBuf> = None;
    let mut inline_budget: Option<u32> = None;
    let mut paths = Vec::new();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--engine" => {
                selected = args.next().ok_or_else(|| anyhow::anyhow!("--engine needs plain, incremental or both"))?;
                anyhow::ensure!(
                    matches!(selected.as_str(), "plain" | "incremental" | "both"),
                    "unknown engine: {selected}"
                );
            }
            "--cache" => cache = Some(args.next().ok_or_else(|| anyhow::anyhow!("--cache needs a directory"))?.into()),
            "--inline-budget" => {
                inline_budget =
                    Some(args.next().ok_or_else(|| anyhow::anyhow!("--inline-budget needs an integer"))?.parse()?);
            }
            "--help" => {
                println!(
                    "compile_time [--engine plain|incremental|both] [--inline-budget N] [--cache DIR] A.wasm [B.wasm ...]"
                );
                return Ok(());
            }
            _ if argument.starts_with('-') => anyhow::bail!("unknown option: {argument}"),
            _ => paths.push(argument),
        }
    }
    anyhow::ensure!(!paths.is_empty(), "supply at least one Wasm file; see --help");
    for label in ["plain", "incremental"] {
        if selected != "both" && selected != label {
            continue;
        }
        let mut engine = if label == "plain" { blot_host::engine()? } else { blot_host::dev_engine()? };
        if let Some(budget) = inline_budget {
            let mut config = engine.config().clone();
            // SAFETY: this Wasmtime 49 option only limits inline expansion. It
            // changes neither the target ISA nor any memory-safety checks.
            unsafe {
                config.cranelift_flag_set("wasmtime_inlining_sum_size_threshold", &budget.to_string());
            }
            engine = wasmtime::Engine::new(&config)?;
        }
        for path in &paths {
            let wasm = std::fs::read(path)?;
            let started = Instant::now();
            blot_host::GuestModule::new(&engine, &wasm, cache.as_deref())?;
            println!("{label}: {path}: {:.0} ms", started.elapsed().as_secs_f64() * 1000.0);
        }
    }
    Ok(())
}
