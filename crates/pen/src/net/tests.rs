//! Net sessions of the game itself (`dist/gdev`, from `just export`)
//! without a GPU: two peers, each on its own thread with its own program,
//! talking over UDP on the loopback, end with the same input and packets
//! for every frame, a straight run's, however late their messages come.
//! Skipped when the bundle is missing (`fixtures`). `rollback_costs`
//! measures what a rollback costs (`--ignored`).
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, reason = "frame positions as indices")]
use std::collections::VecDeque;
use std::path::Path;
use std::time::{Duration, Instant};

use pen_protocol::boundary::read_catalog;
use pen_protocol::{Catalog, RenderFrame, Viewport};

use super::*;
use crate::fixtures::bundle;
use crate::game::{GameRuntime, Host, Quiet};
use crate::script::{self, Script};
use crate::snapshot;
use crate::timeline::{Recorded, Timeline};

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
fn specs_name_every_players_address() {
    let spec = Spec::parse("1:127.0.0.1:7000, 127.0.0.1:7001").unwrap();
    assert_eq!(spec.player, 1);
    assert_eq!(spec.addresses, ["127.0.0.1:7000".parse().unwrap(), "127.0.0.1:7001".parse::<SocketAddr>().unwrap()]);
    assert!(Spec::parse("2:127.0.0.1:7000,127.0.0.1:7001").is_err(), "a player past the list");
    assert!(Spec::parse("0:127.0.0.1:7000").is_err(), "one player is no session");
    assert!(Spec::parse("127.0.0.1:7000,127.0.0.1:7001").is_err());
    assert!(check_players(1, 2).is_err() && check_players(2, 3).is_err() && check_players(4, 2).is_ok());
}

#[test]
fn udp_carries_datagrams_between_listed_peers_only() {
    let sockets: Vec<UdpSocket> = (0..3).map(|_| UdpSocket::bind("127.0.0.1:0").unwrap()).collect();
    let addresses: Vec<SocketAddr> = sockets.iter().map(|socket| socket.local_addr().unwrap()).collect();
    let mut sockets = sockets.into_iter();
    let mut first = Udp::new(sockets.next().unwrap(), addresses[..2].to_vec()).unwrap();
    let mut second = Udp::new(sockets.next().unwrap(), addresses[..2].to_vec()).unwrap();
    let stranger = sockets.next().unwrap();
    stranger.send_to(b"hello", addresses[1]).unwrap();
    first.send(1, b"PNET");
    let deadline = Instant::now() + Duration::from_secs(2);
    let received = loop {
        if let Some(received) = second.receive() {
            break received;
        }
        assert!(Instant::now() < deadline, "the datagram never came");
        std::thread::sleep(Duration::from_millis(1));
    };
    assert_eq!(received, (0, b"PNET".to_vec()), "the stranger's is dropped, the peer's arrives");
    assert_eq!(second.receive(), None);
}

/// A link that holds what it sends for `latency` before handing it to UDP,
/// as a long way would.
struct Late {
    udp: Udp,
    latency: Duration,
    held: VecDeque<(Instant, usize, Vec<u8>)>,
}

impl Late {
    fn flush(&mut self) {
        while self.held.front().is_some_and(|(due, _, _)| *due <= Instant::now()) {
            let (_, player, bytes) = self.held.pop_front().unwrap();
            self.udp.send(player, &bytes);
        }
    }
}

impl Link for Late {
    fn send(&mut self, player: usize, bytes: &[u8]) {
        self.held.push_back((Instant::now() + self.latency, player, bytes.to_vec()));
        self.flush();
    }

    fn receive(&mut self) -> Option<(usize, Vec<u8>)> {
        self.flush();
        self.udp.receive()
    }
}

/// gdev, started as a session starts it: its scene, nothing saved.
fn start(directory: &Path) -> (GameRuntime, Vec<u8>) {
    let engine = blot_host::engine().unwrap();
    let wasm = std::fs::read(directory.join("game.wasm")).unwrap();
    let cache = std::env::temp_dir().join("pen-test-wasm");
    let module = blot_host::GuestModule::new(&engine, &wasm, Some(&cache)).unwrap();
    let catalog = read_catalog(&directory.join("assets")).unwrap();
    let scene = snapshot::read(&directory.join("main.scene")).ok();
    let mut runtime = GameRuntime::new(&engine, module, catalog, &mut Null, None, scene).unwrap();
    runtime.hash_packets(true);
    (runtime, wasm)
}

fn keep(hashes: &mut Vec<u64>, at: u64, hash: u64) {
    let at = at as usize;
    if hashes.len() <= at {
        hashes.resize(at + 1, 0);
    }
    hashes[at] = hash;
}

/// What a peer ended with: every frame's input and packet, and its stats.
struct Ended {
    inputs: Vec<Vec<f32>>,
    hashes: Vec<u64>,
    stats: Stats,
}

/// Plays `frames` frames of gdev as `player`, its input from `steps`,
/// after `alone` frames on its own (so it starts elsewhere than the host).
fn peer(directory: &Path, link: Box<dyn Link>, player: usize, steps: &str, alone: usize, frames: u64) -> Ended {
    let (mut runtime, wasm) = start(directory);
    let mut script = Script::new(script::parse(&[steps.to_owned()]).unwrap());
    for _ in 0..alone {
        let input = runtime.input(&pen_protocol::Frame { timestamp: 0.0, viewport: VIEWPORT, events: Vec::new() });
        runtime.send(&input.unwrap()).unwrap();
        runtime.receive(&Null).unwrap();
    }
    let moment = runtime.moment().unwrap();
    let settings = Settings { delay: 1, ..Settings::new(2, player, VIEWPORT) };
    let build = pen_runtime::net::wire::hash_bytes(&wasm);
    let mut net = Netplay::new(settings, link, build, &moment, 0x5eed).unwrap();
    let mut timeline = Timeline::default();
    timeline.keep_history(None);
    let clock = Instant::now();
    let now = || u64::try_from(clock.elapsed().as_millis()).unwrap();
    let mut hashes = Vec::new();
    let mut ended: Option<Instant> = None;
    loop {
        assert!(clock.elapsed() < Duration::from_secs(300), "player {player}'s session never ended");
        net.poll(now()).unwrap();
        if let Some(start) = net.take_start() {
            runtime.restore(&mut Null, &start).unwrap();
        }
        let mut note = |position: u64, _: &Recorded, runtime: &crate::timeline::Runtime| {
            keep(&mut hashes, position, runtime.packet_hash());
            Ok(())
        };
        pollster::block_on(net.correct(&mut timeline, &mut runtime, &Null, &mut note)).unwrap();
        if net.next() >= frames {
            // Done: say so, then linger a little for the peer to hear it.
            if net.done(frames) {
                net.leave();
                let since = *ended.get_or_insert_with(Instant::now);
                if since.elapsed() > Duration::from_millis(200) {
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }
        if !net.ready().unwrap() {
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }
        net.prepare(&mut timeline);
        timeline.prepare(&mut runtime, &Null).unwrap();
        let position = timeline.position();
        let events = script.tick(VIEWPORT).events;
        let input = net.frame(position, &events, VIEWPORT).unwrap();
        runtime.send(&input).unwrap();
        let rendered = runtime.receive(&Null).unwrap();
        runtime.follow(&mut Null, &rendered).unwrap();
        net.ran(runtime.packet_hash());
        keep(&mut hashes, position, runtime.packet_hash());
        let restored = timeline.restored_here();
        timeline.ran(Recorded { input, loads: Vec::new(), restored });
    }
    let inputs = (0..frames).map(|at| timeline.recorded(at).unwrap().input.clone()).collect();
    runtime.close();
    hashes.truncate(frames as usize);
    Ended { inputs, hashes, stats: net.stats() }
}

/// Into play, building: the editor's Tab, strikes on the rock, a drag.
const FIRST: &str = "wait=10 key=Tab wait=30 click=470,150 wait=12 click=470,152 wait=12 click=471,150 \
    wait=30 drag=300,500,600,450 wait=40";
/// The other god: the pointer over the meadow, clicks of its own, a wheel.
const SECOND: &str = "wait=50 move=480,330 wait=7 click=470,151 wait=9 move=520,300 wait=5 click=469,150 \
    wait=20 wheel=-200,480,320 wait=15 move=200,200 wait=40";

#[test]
fn two_peers_play_the_game_over_udp() {
    let Some(directory) = bundle() else { return };
    let sockets = [UdpSocket::bind("127.0.0.1:0").unwrap(), UdpSocket::bind("127.0.0.1:0").unwrap()];
    let addresses: Vec<SocketAddr> = sockets.iter().map(|socket| socket.local_addr().unwrap()).collect();
    let frames = 240;
    // 25 ms each way against a delay of one frame: most frames run on a
    // prediction of the other's input, and every click is rolled back.
    let ended: Vec<Ended> = std::thread::scope(|scope| {
        let threads: Vec<_> = sockets
            .into_iter()
            .enumerate()
            .map(|(player, socket)| {
                let (directory, addresses) = (&directory, addresses.clone());
                scope.spawn(move || {
                    let udp = Udp::new(socket, addresses).unwrap();
                    let link = Box::new(Late { udp, latency: Duration::from_millis(25), held: VecDeque::new() });
                    let (steps, alone) = if player == 0 { (FIRST, 0) } else { (SECOND, 20) };
                    peer(directory, link, player, steps, alone, frames)
                })
            })
            .collect();
        threads.into_iter().map(|thread| thread.join().unwrap()).collect()
    });
    let (host, guest) = (&ended[0], &ended[1]);
    assert_eq!(host.inputs.len(), frames as usize);
    for at in 0..frames as usize {
        assert_eq!(host.inputs[at], guest.inputs[at], "frame {at}'s input");
    }
    assert_eq!(host.hashes, guest.hashes, "the peers' packets");
    // A straight run of that input, from the host's start, gives them too:
    // the guest took the host's moment, and every rollback put it right.
    let (mut runtime, _) = start(&directory);
    let straight: Vec<u64> = host
        .inputs
        .iter()
        .map(|input| {
            runtime.send(input).unwrap();
            let rendered = runtime.receive(&Null).unwrap();
            runtime.follow(&mut Null, &rendered).unwrap();
            runtime.packet_hash()
        })
        .collect();
    runtime.close();
    assert_eq!(straight, host.hashes, "a straight run's packets");
    assert!(straight.windows(2).filter(|pair| pair[0] != pair[1]).count() > 100, "the packets change");
    // Each peer ran on predictions of the other's input and rolled back
    // where the other did something; most of the other's frames were empty
    // and so predicted right.
    for (player, end) in ended.iter().enumerate() {
        let stats = end.stats;
        eprintln!("player {player}: {stats:?}");
        assert!(stats.rollbacks >= 2 && stats.predicted >= 100, "player {player}: {stats:?}");
        assert!(stats.checked >= 6, "player {player} compared packets: {stats:?}");
    }
    assert!(ended.iter().map(|end| end.stats.rollbacks).sum::<u64>() >= 8);
}

/// Play with a crowd, as the contract tests' `CROWD` (timeline/contract_tests.rs).
const CROWD: &str = "wait=10 key=Tab wait=20 console=3,12 wait=30 \
    click=470,150 wait=15 click=470,152 wait=15 click=471,150 wait=15 click=470,151 wait=30 \
    console=6 wait=30 ui=0x20001,activate wait=10 ui=0x20010,activate move=680,140 wait=20 \
    click=680,140 wait=10 ui=0x20001,collapse move=480,330 wait=480";

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

/// What a rollback of N frames costs on gdev today: a moment (Snapshot),
/// a restore (a new program from it, and its proving frame), and the
/// frames run again; with a moment before every frame, as a net session
/// keeps while it predicts, and with today's moment every 30 frames.
/// `cargo test --release -p pen rollback_costs -- --ignored --nocapture`.
#[test]
#[ignore = "a measurement: run it with --release and read its table"]
fn rollback_costs() {
    let Some(directory) = bundle() else { return };
    let (mut runtime, _) = start(&directory);
    let mut script = Script::new(script::parse(&[CROWD.to_owned()]).unwrap());
    let (mut moments, mut inputs, mut snapshots) = (Vec::new(), Vec::new(), Vec::new());
    let mut time = 0.0;
    while !script.done() {
        let started = Instant::now();
        moments.push(runtime.moment().unwrap());
        snapshots.push(ms(started));
        let frame = pen_protocol::Frame { timestamp: time, viewport: VIEWPORT, events: script.tick(VIEWPORT).events };
        time += 1.0 / 60.0;
        let input = runtime.input(&frame).unwrap();
        runtime.send(&input).unwrap();
        let rendered = runtime.receive(&Null).unwrap();
        runtime.follow(&mut Null, &rendered).unwrap();
        inputs.push(input);
    }
    let head = inputs.len();
    let bytes = moments.iter().map(|moment| moment.saved.state.len() * 4).max().unwrap();
    // A rollback: restore the moment at `from`, run the frames to `to` unseen.
    let mut rollback = |from: usize, to: usize| {
        let started = Instant::now();
        runtime.restore(&mut Quiet(&Null), &moments[from]).unwrap();
        let restored = ms(started);
        let frames = Instant::now();
        for input in &inputs[from..to] {
            runtime.send(input).unwrap();
            let rendered = runtime.receive(&Null).unwrap();
            runtime.follow(&mut Quiet(&Null), &rendered).unwrap();
        }
        (restored, ms(frames))
    };
    let (mut all, mut per_frame) = (Vec::new(), Vec::new());
    println!("gdev, {head} frames of CROWD; moments up to {bytes} bytes; snapshot median {:.3} ms", median(snapshots));
    println!("| N | restore | N frames | rollback (moment every frame) | rollback (moment every 30) |");
    println!("|---|---|---|---|---|");
    for n in [1, 2, 4, 8, 16, 30] {
        let (mut restores, mut dense, mut sparse, mut frames_ms) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        // Ten heads, three frames apart: the last N frames before each run
        // again, from a moment before every frame or, today, from the
        // latest multiple of 30 at or before the first of them.
        for sample in 0..10 {
            let to = head - sample * 3;
            let from = to - n;
            let (restore, frames) = rollback(from, to);
            restores.push(restore);
            per_frame.push(frames / n as f64);
            frames_ms.push(frames);
            dense.push(restore + frames);
            let (restore, frames) = rollback(from - from % 30, to);
            sparse.push(restore + frames);
        }
        all.extend_from_slice(&restores);
        let restore = median(restores);
        println!(
            "| {n} | {restore:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms |",
            median(frames_ms),
            median(dense),
            median(sparse)
        );
    }
    println!("restore median {:.2} ms, a frame run again {:.3} ms (median)", median(all), median(per_frame));
    runtime.close();
}
