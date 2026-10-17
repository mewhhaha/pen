//! What memory images (pen_runtime::memory_image) would cost a game, beside
//! the moments time travel keeps today (docs/instant-rewind.md):
//!
//!   cargo run --release -p blot-host --example memory_images dist/gdev/game.wasm [FRAMES]
//!
//! Plays FRAMES frames (600 unless given; play starts at frame 60, as `pen
//! bench` does) on the inlining engine, imaging the guest's memory at every
//! frame boundary from inside its host call, and asking for a snapshot
//! every 30 frames as the timeline does. Then it restores the image of 30
//! frames before into a copy of the memory, and times today's restore: a
//! new instance run to its handshake, the Restore request and its first
//! frame. A suspended `main` cannot really be put back from an image (its
//! Wasm stack is out of reach), so the restore is timed on a copy.
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, ensure};
use pen_protocol::boundary::encode_frame;
use pen_protocol::{Frame, GameEvent, Viewport};
use pen_runtime::memory_image::{MemoryImage, PAGE};
use pen_runtime::program::{Operation, request};
use wasmtime::{Caller, Engine, ExternRef, Linker, Module, Rooted, Store};

/// Frames between the moments the timeline keeps (pen-runtime's).
const MOMENT_INTERVAL: usize = 30;

#[derive(Default)]
struct Driver {
    /// The requests still to send, in order.
    requests: VecDeque<Vec<f32>>,
    /// What the last request was.
    sent: Option<Operation>,
    sent_at: Option<Instant>,
    images: VecDeque<MemoryImage>,
    memory_bytes: usize,
    /// The memory at the last frame boundary, and the bytes each frame
    /// changed in blocks of 4 KiB and of 256 bytes.
    last: Vec<u8>,
    changed: Vec<(usize, usize)>,
    copied: Vec<usize>,
    capturing: Vec<Duration>,
    unshared: Vec<usize>,
    frames: Vec<Duration>,
    snapshots: Vec<Duration>,
    /// The last snapshot's lanes.
    state: Vec<f32>,
    /// Restoring the image of 30 frames before, into a copy of memory.
    restored: Option<(usize, Duration)>,
    /// When each response came, for the second run.
    answered: Vec<(Option<Operation>, Instant)>,
    imaging: bool,
}

fn exchange(mut caller: Caller<'_, Driver>, pointer: i32) -> Result<i32> {
    let now = Instant::now();
    let memory = caller.get_export("blot:memory").and_then(|export| export.into_memory()).context("no memory")?;
    let address = usize::try_from(pointer)?;
    let bytes = memory.data(&caller);
    let length = u32::from_le_bytes(bytes[address..address + 4].try_into()?) as usize;
    let lanes: Vec<f32> = bytes[address + 4..address + 4 + length * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    let driver = caller.data_mut();
    let sent = driver.sent;
    let took = driver.sent_at.map(|sent| now - sent);
    driver.answered.push((sent, now));
    match (sent, took) {
        (Some(Operation::Frame), Some(took)) => driver.frames.push(took),
        (Some(Operation::Snapshot), Some(took)) => {
            driver.snapshots.push(took);
            driver.state = lanes[1..].to_vec();
        }
        _ => {}
    }
    if sent == Some(Operation::Frame) && caller.data().imaging {
        let started = Instant::now();
        let previous = caller.data().images.back().cloned();
        let bytes = memory.data(&caller);
        let (image, captured) = MemoryImage::capture(bytes, previous.as_ref());
        let capturing = started.elapsed();
        let driver = caller.data_mut();
        driver.memory_bytes = image.len();
        driver.copied.push(captured.copied);
        driver.capturing.push(capturing);
        driver.unshared.push(previous.as_ref().map_or(image.len(), |previous| image.unshared(previous)));
        let bytes = memory.data(&caller).to_vec();
        let driver = caller.data_mut();
        if driver.last.len() == bytes.len() {
            let changed = |block: usize| {
                let blocks = bytes.chunks(block).zip(driver.last.chunks(block));
                blocks.filter(|(now, before)| now != before).count() * block
            };
            driver.changed.push((changed(4096), changed(256)));
        }
        driver.last = bytes;
        driver.images.push_back(image);
        if driver.images.len() > MOMENT_INTERVAL + 1 {
            driver.images.pop_front();
        }
    }
    let next = caller.data_mut().requests.pop_front().unwrap_or_else(|| request(true, Operation::Stop, &[]));
    if next[1] == Operation::Stop as u8 as f32 && caller.data().imaging {
        // The image of 30 frames before, back over a copy of the memory.
        let mut copy = memory.data(&caller).to_vec();
        let oldest = caller.data().images.front().cloned().context("no images")?;
        let started = Instant::now();
        let written = oldest.restore(&mut copy);
        caller.data_mut().restored = Some((written, started.elapsed()));
    }
    let operation = [
        Operation::Stop,
        Operation::Create,
        Operation::Frame,
        Operation::Snapshot,
        Operation::Restore,
        Operation::Scene,
        Operation::LoadScene,
    ]
    .into_iter()
    .find(|operation| *operation as u8 as f32 == next[1]);
    let allocate = caller.get_export("blot:allocate").and_then(|export| export.into_func()).context("no allocate")?;
    let target = allocate.typed::<i32, i32>(&caller)?.call(&mut caller, i32::try_from((next.len() + 1) * 4)?)?;
    let target = usize::try_from(target)?;
    let bytes = memory.data_mut(&mut caller);
    bytes[target..target + 4].copy_from_slice(&u32::try_from(next.len())?.to_le_bytes());
    for (index, lane) in next.iter().enumerate() {
        bytes[target + 4 + index * 4..target + 8 + index * 4].copy_from_slice(&lane.to_le_bytes());
    }
    let driver = caller.data_mut();
    driver.sent = operation;
    driver.sent_at = Some(Instant::now());
    Ok(i32::try_from(target)?)
}

/// Runs `main` with `driver`'s requests; returns the driver and how long
/// instantiating took.
fn run(engine: &Engine, module: &Module, driver: Driver) -> Result<(Driver, Duration)> {
    let mut store = Store::new(engine, driver);
    let mut linker = Linker::new(engine);
    linker.func_wrap(
        "blot:host/1",
        "call_array_f32_array_f32",
        |caller: Caller<'_, Driver>, _host: Option<Rooted<ExternRef>>, pointer: i32| {
            exchange(caller, pointer).map_err(wasmtime::Error::from_anyhow)
        },
    )?;
    let started = Instant::now();
    let instance = linker.instantiate(&mut store, module)?;
    let instantiated = started.elapsed();
    if let Some(enter) = instance.get_func(&mut store, "blot:ownership_enter") {
        enter.typed::<(), ()>(&store)?.call(&mut store, ())?;
    }
    instance.get_typed_func::<i32, i32>(&mut store, "blot:reset")?.call(&mut store, 0)?;
    let host = ExternRef::new(&mut store, ())?;
    let main = instance.get_typed_func::<Option<Rooted<ExternRef>>, i32>(&mut store, "main")?;
    store.data_mut().sent_at = Some(started);
    main.call(&mut store, Some(host)).map_err(|error| anyhow!("{error:?}"))?;
    if let Some(leave) = instance.get_func(&mut store, "blot:ownership_leave") {
        leave.typed::<i32, ()>(&store)?.call(&mut store, 0)?;
    }
    Ok((store.into_data(), instantiated))
}

fn median(values: &mut [Duration]) -> f64 {
    values.sort();
    values.get(values.len() / 2).map_or(0.0, |value| value.as_secs_f64() * 1000.0)
}

fn largest(values: &[Duration]) -> f64 {
    values.iter().max().map_or(0.0, |value| value.as_secs_f64() * 1000.0)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().context("usage: memory_images GAME.wasm [FRAMES]")?;
    let frames: usize = args.next().map(|frames| frames.parse()).transpose()?.unwrap_or(600);
    let wasm = std::fs::read(&path).with_context(|| format!("reading {path}"))?;
    let engine = blot_host::engine()?;
    let module = Module::new(&engine, &wasm)?;
    let viewport = Viewport { width: 1280, height: 720 };
    let input = |index: usize| {
        let events = if index == 60 { vec![GameEvent::KeyDown("Tab".into())] } else { Vec::new() };
        let frame = Frame { timestamp: index as f64 / 60.0, viewport, events };
        encode_frame(&frame, if index == 0 { 0.0 } else { 1.0 / 60.0 }, &[]).unwrap()
    };

    let mut driver = Driver { imaging: true, ..Driver::default() };
    driver.requests.push_back(request(true, Operation::Create, &[]));
    for index in 0..frames {
        if index % MOMENT_INTERVAL == 0 {
            driver.requests.push_back(request(true, Operation::Snapshot, &[]));
        }
        driver.requests.push_back(request(true, Operation::Frame, &input(index)));
    }
    let (mut played, _) = run(&engine, &module, driver)?;
    ensure!(!played.copied.is_empty() && !played.state.is_empty(), "the game played no frames");

    // Today's restore: a new program, its handshake, the Restore and a frame.
    let mut fresh = Driver::default();
    fresh.requests.push_back(request(true, Operation::Restore, &played.state));
    fresh.requests.push_back(request(true, Operation::Frame, &input(frames)));
    let started = Instant::now();
    let (restored, instantiated) = run(&engine, &module, fresh)?;
    let at = |operation: Option<Operation>| {
        restored.answered.iter().find(|(sent, _)| *sent == operation).map(|(_, at)| *at - started)
    };
    let ms = |duration: Option<Duration>| duration.map_or(0.0, |duration| duration.as_secs_f64() * 1000.0);
    let handshake = at(None);
    let restore = at(Some(Operation::Restore));
    let first = at(Some(Operation::Frame));

    let pages = played.memory_bytes / PAGE;
    let mean = |values: &[usize]| values.iter().sum::<usize>() as f64 / values.len().max(1) as f64;
    let (written, restoring) = played.restored.unwrap_or_default();
    println!("{path}: {frames} frames; memory {:.1} MB ({pages} pages)", played.memory_bytes as f64 / 1e6);
    println!(
        "memory images, each frame: {:.1} pages copied on average, {} at most; capture median {:.3} ms, max {:.3} ms; \
         {:.0} KB a frame kept",
        mean(&played.copied),
        played.copied.iter().max().unwrap_or(&0),
        median(&mut played.capturing),
        largest(&played.capturing),
        mean(&played.unshared) / 1024.0,
    );
    let finer = |pick: fn(&(usize, usize)) -> usize| {
        played.changed.iter().map(pick).sum::<usize>() as f64 / played.changed.len().max(1) as f64 / 1024.0
    };
    println!(
        "bytes a frame changes, in blocks of 4 KiB: {:.0} KB on average; of 256 bytes: {:.0} KB",
        finer(|changed| changed.0),
        finer(|changed| changed.1)
    );
    println!(
        "restoring the image of {MOMENT_INTERVAL} frames before: {written} pages written in {:.3} ms",
        restoring.as_secs_f64() * 1000.0
    );
    println!(
        "today, every {MOMENT_INTERVAL} frames: a snapshot of {} lanes ({:.1} KB) in median {:.3} ms, max {:.3} ms",
        played.state.len(),
        played.state.len() as f64 * 4.0 / 1024.0,
        median(&mut played.snapshots),
        largest(&played.snapshots),
    );
    println!(
        "today's restore: instantiate {:.2} ms, handshake at {:.2} ms, restored at {:.2} ms, first frame at {:.2} ms; \
         a frame's median {:.3} ms, so a rewind runs up to {} frames again (up to {:.1} ms more)",
        instantiated.as_secs_f64() * 1000.0,
        ms(handshake),
        ms(restore),
        ms(first),
        median(&mut played.frames),
        MOMENT_INTERVAL - 1,
        median(&mut played.frames) * (MOMENT_INTERVAL - 1) as f64,
    );
    Ok(())
}
