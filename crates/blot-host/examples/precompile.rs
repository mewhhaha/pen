//! Precompiles a Blot program as `pen build` does, for comparing CPU
//! baselines:
//!   cargo run --release -p blot-host --example precompile game.wasm game.cwasm [CPU] [TARGET]
//! CPU is a Cranelift preset (`baseline`, `x86-64-v2`, `x86-64-v3`, ...) or
//! `native`; the default (or `-`) is blot_host::PORTABLE_X86_64 on x86-64.
use std::time::Instant;

use anyhow::Context;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [wasm, out] = [0, 1].map(|index| args.get(index).cloned());
    let (wasm, out) = wasm.zip(out).context("usage: precompile game.wasm game.cwasm [CPU] [TARGET]")?;
    let cpu = args.get(2).map(String::as_str).filter(|cpu| *cpu != "-");
    let target = args.get(3).map(String::as_str);
    let engine = blot_host::portable_engine(target, cpu)?;
    let started = Instant::now();
    let precompiled = blot_host::precompile(&engine, &std::fs::read(&wasm)?)?;
    std::fs::write(&out, &precompiled)?;
    println!("{out}: {} bytes in {:.0} ms", precompiled.len(), started.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}
