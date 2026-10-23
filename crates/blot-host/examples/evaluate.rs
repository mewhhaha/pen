//! Compile a checkout's program and call an Array F32 -> Array F32 export.
//! Numeric probes and fixture generators use the real compiler plugin and ABI:
//!   cargo run --release -p blot-host --example evaluate -- ROOT ENTRY EXPORT INPUT.json OUTPUT.json
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use pen_plugins::{Builder, Outcome, Plugins};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(args.len() == 5, "usage: evaluate ROOT ENTRY EXPORT INPUT.json OUTPUT.json");
    let root = Path::new(&args[0]).canonicalize()?;
    let entry = root.join(&args[1]).canonicalize()?;
    let log: pen_plugins::Logger = Arc::new(|_, message| eprintln!("{message}"));
    let plugins = Plugins::load(&root, log)?.context("checkout has no compiler plugin")?;
    let mut builder = Builder::new(plugins, &entry);
    let outcome = builder.build(&[]);
    builder.close()?;
    let wasm = match outcome {
        Outcome::Built(built) => built.wasm,
        Outcome::Rejected(diagnostics) => bail!("program rejected: {diagnostics:?}"),
        Outcome::Failed(error) => bail!("compiler failed: {error}"),
    };
    let input: Vec<f32> = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    ensure!(input.iter().all(|n| n.is_finite()), "input contains nonfinite lanes");
    let engine = blot_host::engine()?;
    let module = blot_host::GuestModule::new(&engine, &wasm, None)?;
    let mut guest = blot_host::Guest::new(&engine, &module)?;
    let output = guest.call(&args[2], blot_host::Value::ArrayF32(input))?.into_f32_array()?;
    ensure!(output.iter().all(|n| n.is_finite()), "output contains nonfinite lanes");
    std::fs::write(&args[4], serde_json::to_vec(&output)?)?;
    Ok(())
}
