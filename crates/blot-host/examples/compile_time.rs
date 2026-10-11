//! Times compiling Blot programs one after another, as development does:
//!   cargo run --release -p blot-host --example compile_time A.wasm B.wasm ...
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    for (label, engine) in [("plain", blot_host::engine()?), ("incremental", blot_host::dev_engine()?)] {
        for path in &paths {
            let wasm = std::fs::read(path)?;
            let started = Instant::now();
            blot_host::GuestModule::new(&engine, &wasm, None)?;
            println!("{label}: {path}: {:.0} ms", started.elapsed().as_secs_f64() * 1000.0);
        }
    }
    Ok(())
}
