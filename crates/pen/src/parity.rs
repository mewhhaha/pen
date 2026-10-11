//! Renderer parity: the native renderer is a port of the checkout's
//! `renderer.ts`, so a change to one needs its twin in the other. This runs
//! a script on the game's program, draws chosen frames with both renderers
//! (the TypeScript one on Deno's WebGPU) and compares the images.
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use blot_host::GuestModule;
use pen_protocol::boundary::{decode_frame, read_catalog};
use pen_protocol::Viewport;

use crate::packets;
use crate::project::Project;
use crate::script::Step;
use crate::session::Gpu;

const REFERENCE_SCRIPT: &str = include_str!("../scripts/render_reference.ts");

pub struct ParityOptions {
    pub size: (u32, u32),
    pub steps: Vec<Step>,
    /// Frames to compare; default every 30th and the last.
    pub at: Vec<usize>,
    /// Largest RMSE (0..1 over the colour channels) a frame may differ by.
    pub threshold: f64,
    pub out: PathBuf,
}

fn rmse(a: &[u8], b: &[u8]) -> (f64, u8) {
    let mut squares = 0.0f64;
    let mut largest = 0u8;
    let mut count = 0usize;
    for (a, b) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
        for channel in 0..3 {
            let delta = a[channel].abs_diff(b[channel]);
            squares += f64::from(delta) * f64::from(delta);
            largest = largest.max(delta);
            count += 1;
        }
    }
    ((squares / count.max(1) as f64).sqrt() / 255.0, largest)
}

fn save(path: &Path, rgba: &[u8], (width, height): (u32, u32)) -> Result<()> {
    image::save_buffer(path, rgba, width, height, image::ColorType::Rgba8)
        .with_context(|| format!("writing {}", path.display()))
}

pub fn run(project: &Project, engine: &wasmtime::Engine, module: &GuestModule, options: ParityOptions) -> Result<()> {
    let assets = project.assets();
    let packets_dir = options.out.join("packets");
    let reference = options.out.join("reference");
    let _ = std::fs::remove_dir_all(&options.out);
    std::fs::create_dir_all(&reference)?;
    let frames = packets::run(packets::Run {
        module,
        engine,
        assets: assets.clone(),
        scene: project.scene(),
        viewport: Viewport { width: options.size.0, height: options.size.1 },
        out: Some(packets_dir.clone()),
        frames: None,
        steps: options.steps,
    })?;
    if frames == 0 {
        bail!("the script ran no frames");
    }
    let mut at = options.at.clone();
    if at.is_empty() {
        at = (0..frames).step_by(30).chain([frames - 1]).collect();
        at.dedup();
    }
    let names: Vec<String> = at.iter().filter(|&&index| index < frames).map(|index| format!("{index:05}")).collect();

    eprintln!("drawing {} frames with renderer.ts ...", names.len());
    let script = dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("pen").join("render_reference.ts");
    std::fs::create_dir_all(script.parent().unwrap())?;
    std::fs::write(&script, REFERENCE_SCRIPT)?;
    let status = Command::new("deno")
        .args(["run", "-A", "--no-lock", "--config"])
        .arg(project.root.join("deno.json"))
        .arg(&script)
        .arg(&project.root)
        .arg(&assets)
        .arg(&packets_dir)
        .arg(&reference)
        .arg(options.size.0.to_string())
        .arg(options.size.1.to_string())
        .args(&names)
        .status()
        .context("running deno for the reference renderer")?;
    if !status.success() {
        bail!("the reference renderer failed");
    }

    eprintln!("drawing them with the native renderer ...");
    let catalog = read_catalog(&assets)?;
    let schema: u32 = std::fs::read_to_string(packets_dir.join("schema.txt"))?.trim().parse()?;
    let mut gpu = Gpu::headless(options.size, &assets)?;
    gpu.renderer.preload(&catalog);
    let mut failures = Vec::new();
    let mut unchecked = Vec::new();
    for name in &names {
        // Deno's WebGPU refuses some allocations the native renderer makes
        // (a full-screen inset's 4K, 4x MSAA targets); such a frame has no reference.
        if let Ok(reason) = std::fs::read_to_string(reference.join(format!("{name}.failed"))) {
            eprintln!("  frame {name}: not compared, renderer.ts failed on Deno's WebGPU: {reason}");
            unchecked.push(name.clone());
            continue;
        }
        let bytes = std::fs::read(packets_dir.join("output").join(format!("{name}.f32")))?;
        let lanes: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect();
        let mut frame = decode_frame(&lanes, schema, &catalog)?.render;
        let stripped = reference.join(format!("{name}.stripped")).exists();
        if stripped {
            frame.insets.clear();
        }
        let native = gpu.draw_settled(&frame, &catalog)?;
        let expected = std::fs::read(reference.join(format!("{name}.rgba")))?;
        let (difference, largest) = rmse(&native, &expected);
        let verdict = if difference > options.threshold { "DIFFERS" } else { "ok" };
        let note = if stripped { " (without its insets, which renderer.ts cannot allocate on Deno's WebGPU)" } else { "" };
        eprintln!(
            "  frame {name}: {verdict}, RMSE {:.4}%, largest {:.1}%{note}",
            difference * 100.0,
            f64::from(largest) / 2.55
        );
        if difference > options.threshold {
            save(&options.out.join(format!("{name}.native.png")), &native, options.size)?;
            save(&options.out.join(format!("{name}.reference.png")), &expected, options.size)?;
            failures.push(name.clone());
        }
    }
    if !failures.is_empty() {
        bail!(
            "the native renderer differs from renderer.ts in frames {} (images in {}); port the renderer change",
            failures.join(", "),
            options.out.display()
        );
    }
    let compared = names.len() - unchecked.len();
    if compared == 0 {
        bail!("no frame could be compared: renderer.ts failed on every one");
    }
    eprintln!(
        "{compared} frames: the native renderer matches renderer.ts{}",
        if unchecked.is_empty() { String::new() } else { format!(" ({} not comparable)", unchecked.len()) }
    );
    Ok(())
}
