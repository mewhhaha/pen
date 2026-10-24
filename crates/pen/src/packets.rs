//! The game's program alone, without rendering: guest timing (`--bench`),
//! and each frame's input and render packet for renderer tests, engine
//! native/browser qualification.
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use blot_host::GuestModule;
use pen_protocol::boundary::{decode_frame, encode_frame, read_catalog};
use pen_protocol::{Extension, Frame, GameEvent, Viewport};
use pen_runtime::timeline::MOMENT_INTERVAL;

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
    /// Mods beside the game (`--mod`), which add to each frame's input.
    pub mods: Option<crate::mods::Mods>,
    /// Also time a snapshot of the state every `MOMENT_INTERVAL` frames,
    /// as a history takes its moments (`pen bench`).
    pub snapshots: bool,
    /// One of the game's own sections as `out/extensions.jsonl` has it.
    pub describe: &'a dyn Fn(&Extension) -> serde_json::Value,
}

/// How long the guest took, in milliseconds: each frame, and each snapshot
/// taken.
#[derive(Debug, Default)]
pub struct Measured {
    pub frames: Vec<f64>,
    pub snapshots: Vec<f64>,
}

/// Frames a run's timings leave out at its start, while caches warm.
pub fn warmup(frames: usize) -> usize {
    frames.min(120)
}

/// Runs the program at 60 frames per game second and returns how long each
/// frame (and snapshot) took.
pub fn run(mut run: Run) -> Result<Measured> {
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
        let _ = std::fs::remove_file(out.join("extensions.jsonl"));
        std::fs::write(out.join("schema.txt"), program.schema.to_string())?;
    }
    // Default captures observe the authored startup. A workload supplies its
    // own game-specific controls instead of the host guessing an editor key.
    let steps = if run.steps.is_empty() { script::parse(&["wait=120".into()])? } else { run.steps };
    let scripted = run.frames.is_none();
    let mut script = Script::new(steps);
    let mut times = Vec::new();
    let mut snapshots = Vec::new();
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
        let mut events = script.tick(run.viewport).events;
        // No renderer loads asynchronously here. Record the same readiness
        // event a real host sends so captures can leave the loading screen.
        if index == 0 {
            events.insert(0, GameEvent::RendererReady);
            events.insert(0, GameEvent::SessionStarted { restored: 0 });
        }
        let frame = Frame { timestamp: index as f64 / 60.0, viewport: run.viewport, events };
        let elapsed = if index == 0 { 0.0 } else { 1.0 / 60.0 };
        let mut input = encode_frame(&frame, elapsed, &[])?;
        if let Some(mods) = &mut run.mods {
            mods.frame(&mut input);
        }
        let started = Instant::now();
        let packet = program.request(Operation::Frame, &input)?;
        times.push(started.elapsed().as_secs_f64() * 1000.0);
        program.accept();
        let decoded = decode_frame(&packet, program.schema, &catalog).with_context(|| format!("frame {index}"))?;
        write("input", index, &input)?;
        write("output", index, &packet)?;
        if run.snapshots && (index as u64 + 1).is_multiple_of(MOMENT_INTERVAL) {
            let started = Instant::now();
            program.request(Operation::Snapshot, &[])?;
            snapshots.push(started.elapsed().as_secs_f64() * 1000.0);
            program.accept();
        }
        // The game's own sections, a line a frame that has any.
        if let Some(out) = out
            && !decoded.render.extensions.is_empty()
        {
            let sections: Vec<_> = decoded.render.extensions.iter().map(run.describe).collect();
            let line = serde_json::json!({ "frame": index, "sections": sections });
            let mut file = std::fs::OpenOptions::new().create(true).append(true).open(out.join("extensions.jsonl"))?;
            writeln!(file, "{line}")?;
        }
    }
    let warm = warmup(times.len());
    let mut sorted = times[warm..].to_vec();
    sorted.sort_by(f64::total_cmp);
    if !sorted.is_empty() {
        let at = |percent: usize| sorted[(sorted.len() - 1) * percent / 100];
        eprintln!(
            "wasmtime: frames {}, after {warm} warm-up: median {:.2} ms, p95 {:.2} ms",
            times.len(),
            at(50),
            at(95)
        );
    }
    if let Some(out) = out {
        // Compare hidden simulation state as well as presentation packets.
        // This final witness is outside the frame/snapshot timing samples.
        let state = program.request(Operation::Snapshot, &[])?;
        program.accept();
        snapshot::write(&out.join("native.snapshot"), &snapshot::SavedState { schema: program.schema, state })?;
        eprintln!("wrote {} frames' inputs, packets and final snapshot to {}", times.len(), out.display());
    }
    program.close();
    Ok(Measured { frames: times, snapshots })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_capture_leaves_loading_and_reaches_the_scripted_scene() {
        let Some(bundle) = crate::fixtures::bundle() else { return };
        let engine = blot_host::engine().unwrap();
        let wasm = std::fs::read(bundle.join("game.wasm")).unwrap();
        let cache = std::env::temp_dir().join("pen-test-wasm");
        let module = GuestModule::new(&engine, &wasm, Some(&cache)).unwrap();
        let mut guest = Program::start(&engine, &module).unwrap();
        let schema = guest.schema;
        guest.close();
        let assets = bundle.join("assets");
        let catalog = read_catalog(&assets).unwrap();
        let out = std::env::temp_dir().join(format!("pen-packets-loading-{}", std::process::id()));
        let measured = run(Run {
            module: &module,
            engine: &engine,
            assets,
            scene: bundle.join("main.scene"),
            viewport: Viewport { width: 960, height: 640 },
            out: Some(out.clone()),
            frames: Some(110),
            steps: script::parse(&["wait=100 console=7,0 wait=10".into()]).unwrap(),
            mods: None,
            snapshots: false,
            describe: &|_| serde_json::Value::Null,
        })
        .unwrap();
        let read = |index: usize| {
            let bytes = std::fs::read(out.join("output").join(format!("{index:05}.f32"))).unwrap();
            let (words, remainder) = bytes.as_chunks::<4>();
            assert!(remainder.is_empty());
            let lanes: Vec<f32> = words.iter().copied().map(f32::from_le_bytes).collect();
            decode_frame(&lanes, schema, &catalog).unwrap().render
        };
        assert!(read(0).loading, "capture includes startup");
        let played = read(measured.frames.len() - 1);
        assert!(!played.loading, "readiness releases the game's loading gate");
        assert!(!played.draws.is_empty(), "the scripted level draws its world");
        let saved = snapshot::read(&out.join("native.snapshot")).unwrap();
        assert_eq!(saved.schema, schema);
        assert!(!saved.state.is_empty(), "the final witness uses the shared disk save format");
        std::fs::remove_dir_all(out).unwrap();
    }
}
