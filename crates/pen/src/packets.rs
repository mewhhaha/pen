//! The game's program alone, without rendering: guest timing (`--bench`),
//! and each frame's input and render packet for renderer tests, engine
//! conformance (`tools/conformance_v8.ts`) and `pen parity`.
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use blot_host::GuestModule;
use pen_protocol::boundary::{decode_frame, encode_frame, read_catalog};
use pen_protocol::{Frame, Viewport};

use crate::program::{Operation, Program};
use crate::script::{self, Script, Step};
use crate::snapshot;

pub struct Run<'a> {
    pub module: &'a GuestModule,
    pub engine: &'a wasmtime::Engine,
    pub assets: PathBuf,
    /// The scene a new game loads, if it exists.
    pub scene: PathBuf,
    pub viewport: Viewport,
    /// Write each frame's input and packet under `out/input` and `out/output`.
    pub out: Option<PathBuf>,
    /// Run this many frames, or as long as the steps last.
    pub frames: Option<usize>,
    pub steps: Vec<Step>,
}

/// Runs the program at 60 frames per game second and returns how many
/// frames ran.
pub fn run(run: Run) -> Result<usize> {
    let catalog = read_catalog(&run.assets)?;
    let mut program = Program::start(run.engine, run.module)?;
    program.request(Operation::Create, &[])?;
    program.accept();
    if let Ok(scene) = snapshot::read(&run.scene)
        && scene.schema == program.schema
    {
        program.request(Operation::LoadScene, &scene.state)?;
        program.accept();
    }
    let out = run.out.as_deref();
    if let Some(out) = out {
        // A longer earlier run's frames would otherwise stay behind.
        for directory in ["input", "output"] {
            let _ = std::fs::remove_dir_all(out.join(directory));
            std::fs::create_dir_all(out.join(directory))?;
        }
        std::fs::write(out.join("schema.txt"), program.schema.to_string())?;
    }
    // Without steps, the editor and back again, to show its interface.
    let steps = if run.steps.is_empty() { script::parse(&["wait=60 key=Tab".into()])? } else { run.steps };
    let scripted = run.frames.is_none();
    let mut script = Script::new(steps);
    let mut times = Vec::new();
    let write = |directory: &str, index: usize, lanes: &[f32]| -> Result<()> {
        if let Some(out) = out {
            let bytes: Vec<u8> = lanes.iter().flat_map(|value| value.to_le_bytes()).collect();
            std::fs::write(out.join(directory).join(format!("{index:05}.f32")), bytes)?;
        }
        Ok(())
    };
    for index in 0..run.frames.unwrap_or(usize::MAX) {
        if scripted && script.done() {
            break;
        }
        let events = script.tick(run.viewport).events;
        let frame = Frame { timestamp: index as f64 / 60.0, viewport: run.viewport, events };
        let elapsed = if index == 0 { 0.0 } else { 1.0 / 60.0 };
        let input = encode_frame(&frame, elapsed, &[])?;
        let started = Instant::now();
        let packet = program.request(Operation::Frame, &input)?;
        times.push(started.elapsed().as_secs_f64() * 1000.0);
        program.accept();
        decode_frame(&packet, program.schema, &catalog).with_context(|| format!("frame {index}"))?;
        write("input", index, &input)?;
        write("output", index, &packet)?;
    }
    let warm = times.len().min(120);
    let mut sorted = times[warm..].to_vec();
    sorted.sort_by(f64::total_cmp);
    if !sorted.is_empty() {
        let at = |q: f64| sorted[(q * (sorted.len() - 1) as f64) as usize];
        eprintln!(
            "wasmtime: frames {}, after {warm} warm-up: median {:.2} ms, p95 {:.2} ms",
            times.len(),
            at(0.5),
            at(0.95)
        );
    }
    if let Some(out) = out {
        eprintln!("wrote {} frames' inputs and packets to {}", times.len(), out.display());
    }
    program.close();
    Ok(times.len())
}
