//! A stream of the game itself (`dist/gdev`) over a WebSocket on the
//! loopback: a viewer that joins late plays every frame from where it
//! started to the end exactly as the session did. Skipped when the bundle
//! is missing (`fixtures`).
#![allow(clippy::cast_possible_truncation, reason = "frame positions as indices")]
use std::path::Path;
use std::time::Duration;

use pen_protocol::boundary::read_catalog;
use pen_protocol::{Catalog, Frame, RenderFrame, Viewport};
use pen_runtime::timeline::Recording;

use super::*;
use crate::fixtures::bundle;
use crate::game::{GameRuntime, Host, Quiet};
use crate::script::{self, Script};
use crate::snapshot;
use crate::timeline::Recorded;

const VIEWPORT: Viewport = Viewport { width: 960, height: 640 };

struct Null;

impl Host for Null {
    fn viewport(&self) -> Viewport {
        VIEWPORT
    }

    fn validate(&self, frame: &RenderFrame) -> anyhow::Result<()> {
        pen_render::validate_frame(frame)
    }

    fn render(&mut self, _frame: &RenderFrame, _catalog: &Catalog) -> anyhow::Result<()> {
        Ok(())
    }

    fn report(&mut self, _message: &str) {}
}

#[test]
fn loopback_streams_take_native_viewers_and_loopback_pages() {
    let loopback: SocketAddr = "127.0.0.1:7200".parse().unwrap();
    let open: SocketAddr = "0.0.0.0:7200".parse().unwrap();
    assert!(allowed(loopback, None));
    assert!(allowed(loopback, Some("http://localhost:8000")) && allowed(loopback, Some("http://[::1]:8000")));
    assert!(!allowed(loopback, Some("https://elsewhere.example")));
    assert!(!allowed(loopback, Some("http://localhost.elsewhere.example")));
    assert!(allowed(open, Some("https://elsewhere.example")));
}

fn start(directory: &Path) -> GameRuntime {
    let engine = blot_host::engine().unwrap();
    let wasm = std::fs::read(directory.join("game.wasm")).unwrap();
    let cache = std::env::temp_dir().join("pen-test-wasm");
    let module = blot_host::GuestModule::new(&engine, &wasm, Some(&cache)).unwrap();
    let catalog = read_catalog(&directory.join("assets")).unwrap();
    let scene = snapshot::read(&directory.join("main.scene")).ok();
    let mut runtime = GameRuntime::new(&engine, module, catalog, &mut Null, None, scene).unwrap();
    runtime.hash_packets(true);
    runtime
}

/// Watches the stream at `address` to its end: where it started, and each
/// frame's packet from there.
fn watch(directory: &Path, address: &str) -> (u64, Vec<u64>) {
    let mut runtime = start(directory);
    let mut watch = Watch::connect(address).unwrap();
    let mut timeline = Timeline::default();
    timeline.keep_history(None);
    let (mut first, mut hashes) = (None, Vec::new());
    while let Some(message) = watch.next(Duration::from_secs(30)).unwrap() {
        match message {
            Message::Start { position, moment } => {
                runtime.restore(&mut Quiet(&Null), &moment).unwrap();
                timeline.load_at(Recording { start: moment, frames: Vec::new() }, position);
                first.get_or_insert(position);
            }
            Message::Frames { first, frames } => {
                assert_eq!(first, timeline.head(), "frames arrive in order");
                timeline.extend(frames);
            }
            Message::End => break,
        }
        // Plays what arrived.
        while let Some(recorded) = timeline.next().cloned() {
            timeline.prepare(&mut runtime, &Null).unwrap();
            runtime.send(&recorded.input).unwrap();
            let rendered = runtime.receive(&Null).unwrap();
            runtime.follow(&mut Null, &rendered).unwrap();
            hashes.push(runtime.packet_hash());
            timeline.ran(recorded);
        }
    }
    assert!(watch.ended(), "the stream ended, not broke");
    runtime.close();
    (first.expect("the stream started"), hashes)
}

#[test]
fn a_late_viewer_plays_what_the_session_played() {
    let Some(directory) = bundle() else { return };
    let mut runtime = start(&directory);
    let mut timeline = Timeline::default();
    timeline.keep_history(Some(10.0));
    let mut stream = Stream::serve("127.0.0.1:0").unwrap();
    let address = stream.address().to_string();
    let steps = "wait=10 key=Tab wait=20 console=3,6 wait=30 click=470,150 wait=15 click=470,152 wait=40 \
        move=480,330 wait=200";
    let mut script = Script::new(script::parse(&[steps.to_owned()]).unwrap());
    let mut hashes = Vec::new();
    let (mut time, mut viewer) = (0.0, None);
    while !script.done() {
        if timeline.position() == 100 {
            let (directory, address) = (directory.clone(), address.clone());
            viewer = Some(std::thread::spawn(move || watch(&directory, &address)));
            // Its handshake done, it joins.
            while stream.poll(&timeline, timeline.position()) == 0 {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        timeline.prepare(&mut runtime, &Null).unwrap();
        let frame = Frame { timestamp: time, viewport: VIEWPORT, events: script.tick(VIEWPORT).events };
        time += 1.0 / 60.0;
        let input = runtime.input(&frame).unwrap();
        runtime.send(&input).unwrap();
        let rendered = runtime.receive(&Null).unwrap();
        runtime.follow(&mut Null, &rendered).unwrap();
        hashes.push(runtime.packet_hash());
        let restored = timeline.restored_here();
        timeline.ran(Recorded { input, loads: Vec::new(), restored });
        stream.poll(&timeline, timeline.position());
    }
    stream.end();
    runtime.close();
    let (first, watched) = viewer.unwrap().join().unwrap();
    // It started at the moment kept at or before frame 100, and played on
    // to the end.
    assert_eq!(first, 90);
    assert_eq!(watched.len(), hashes.len() - first as usize);
    assert_eq!(watched, hashes[first as usize..], "the viewer's packets");
}
