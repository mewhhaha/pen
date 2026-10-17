//! Peers playing a small deterministic game over a link that delays,
//! loses, repeats and reorders datagrams: whatever the link does, every
//! peer ends with the same input and the same packets for every frame, the
//! packets a straight run of that input gives.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the toy game's lanes are small whole numbers"
)]
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use pen_protocol::boundary::{PACKET_HEADER_FLOATS, PACKET_MAGIC, lanes};
use pen_protocol::{Catalog, GameEvent, RenderFrame, Viewport};
use web_time::Instant;

use super::*;
use crate::game::{Edits, GameRuntime, Host, Moment};
use crate::program::{Backend, Operation, Program, Response, response};
use crate::snapshot::SavedState;
use crate::timeline::{Recorded, Timeline};

const VIEWPORT: Viewport = Viewport { width: 960, height: 640 };

/// Runs a future of the toy game, which never waits.
fn ready<T>(future: impl Future<Output = T>) -> T {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match std::pin::pin!(future).poll(&mut context) {
        std::task::Poll::Ready(value) => value,
        std::task::Poll::Pending => unreachable!("the toy game answers at once"),
    }
}

/// A game of 9 lanes: frames run, game time, a hash of every event in
/// order with its player, each of four players' own hash, and the last
/// pointer position. Its packet shows them in the view matrix, so
/// packets differ whenever states do. `leak` keeps a number outside its
/// state that differs from peer to peer, as a buggy game would.
#[derive(Clone, Default)]
struct Toy {
    leak: f32,
    /// Programs started: the first, and one for each restore.
    starts: Rc<std::cell::Cell<u64>>,
}

struct Running {
    state: Vec<f32>,
    pending: Option<Vec<f32>>,
    request: Option<(Operation, Vec<f32>)>,
    leak: f32,
}

const MODULUS: f32 = 65521.0;

fn mix(hash: f32, value: f32) -> f32 {
    (hash * 31.0 + value).rem_euclid(MODULUS)
}

impl Running {
    fn frame(&self, input: &[f32]) -> Vec<f32> {
        let mut state = self.state.clone();
        state[0] += 1.0;
        state[1] += input[0];
        let mut player = 0usize;
        for event in input[5..].as_chunks::<5>().0 {
            if event[0] == 26.0 {
                player = event[1] as usize;
                continue;
            }
            state[2] = mix(state[2], (player + 1) as f32 * 1000.0 + event[0] * 37.0 + event[1]);
            state[3 + player] = mix(state[3 + player], event[0] * 37.0 + event[1] + event[2] + event[3]);
            if event[0] == 5.0 {
                state[7] = event[2];
                state[8] = event[3];
            }
        }
        state[2] = mix(state[2], self.leak);
        state
    }

    fn packet(state: &[f32]) -> Vec<f32> {
        let mut packet = vec![0.0; 1 + PACKET_HEADER_FLOATS];
        packet[0] = Operation::Frame as u8 as f32;
        packet[1 + lanes::header::MAGIC] = PACKET_MAGIC;
        packet[1 + lanes::header::SCHEMA] = 1.0;
        packet[1 + lanes::header::VIEW..1 + lanes::header::VIEW + state.len()].copy_from_slice(state);
        packet
    }
}

impl Backend for Toy {
    type Module = ();
    type Program = Running;
    async fn start(&self, _: &()) -> anyhow::Result<Running> {
        self.starts.set(self.starts.get() + 1);
        Ok(Running { state: vec![0.0; 9], pending: None, request: None, leak: self.leak })
    }
}

impl Program for Running {
    fn schema(&self) -> u32 {
        1
    }
    fn scenes(&self) -> bool {
        false
    }
    fn send(&mut self, operation: Operation, payload: &[f32]) -> anyhow::Result<()> {
        // What the last request did is kept only if the host accepted it.
        self.pending = None;
        self.request = Some((operation, payload.to_vec()));
        Ok(())
    }
    async fn response(&mut self) -> anyhow::Result<Response> {
        let (operation, payload) = self.request.take().expect("a request in flight");
        let answer = match operation {
            Operation::Create => vec![operation as u8 as f32],
            Operation::Restore => {
                self.pending = Some(payload);
                vec![operation as u8 as f32]
            }
            Operation::Snapshot => [&[operation as u8 as f32], self.state.as_slice()].concat(),
            Operation::Frame => {
                let state = self.frame(&payload);
                let packet = Running::packet(&state);
                self.pending = Some(state);
                packet
            }
            _ => anyhow::bail!("the toy game has no scenes"),
        };
        response(operation, answer)
    }
    fn accept(&mut self) {
        if let Some(state) = self.pending.take() {
            self.state = state;
        }
    }
    fn answered(&self) -> Option<Instant> {
        None
    }
    fn memory_bytes(&self) -> usize {
        0
    }
    async fn close(&mut self) {}
}

struct Viewer;

impl Host for Viewer {
    fn viewport(&self) -> Viewport {
        VIEWPORT
    }
    fn validate(&self, _: &RenderFrame) -> anyhow::Result<()> {
        Ok(())
    }
    fn render(&mut self, _: &RenderFrame, _: &Catalog) -> anyhow::Result<()> {
        Ok(())
    }
    fn report(&mut self, _: &str) {}
}

/// Datagrams between peers, delivered `latency` ms after they were sent,
/// give or take `jitter` (so some overtake others), a `loss` in a hundred
/// of them lost and as many sent twice.
#[derive(Default)]
struct Hub {
    now: u64,
    latency: u64,
    jitter: u64,
    loss: u64,
    dice: u64,
    /// (due, from, to, bytes)
    flying: VecDeque<(u64, usize, usize, Vec<u8>)>,
}

impl Hub {
    fn roll(&mut self, sides: u64) -> u64 {
        self.dice = self.dice.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.dice >> 33) % sides.max(1)
    }
}

struct Wire {
    hub: Rc<RefCell<Hub>>,
    player: usize,
}

impl Link for Wire {
    fn send(&mut self, player: usize, bytes: &[u8]) {
        let mut hub = self.hub.borrow_mut();
        let copies = match hub.roll(100) {
            roll if roll < hub.loss => 0,
            roll if roll < 2 * hub.loss => 2,
            _ => 1,
        };
        for _ in 0..copies {
            let jitter = hub.jitter;
            let due = hub.now + hub.latency + hub.roll(jitter + 1);
            hub.flying.push_back((due, self.player, player, bytes.to_vec()));
        }
    }

    fn receive(&mut self) -> Option<(usize, Vec<u8>)> {
        let mut hub = self.hub.borrow_mut();
        let now = hub.now;
        let at = hub.flying.iter().position(|(due, _, to, _)| *due <= now && *to == self.player)?;
        let (_, from, _, bytes) = hub.flying.remove(at)?;
        Some((from, bytes))
    }
}

fn moment(state: Vec<f32>) -> Moment {
    Moment { saved: SavedState { schema: 1, state }, edits: Edits::default() }
}

/// One player's events on its `step`th frame: keys and pointer moves at
/// moments of its own, different for each player.
fn script(player: usize, step: u64) -> Vec<GameEvent> {
    let mut events = Vec::new();
    if (step + 3 * player as u64).is_multiple_of(7) {
        events.push(GameEvent::KeyDown(["a", "b", "c"][player % 3].to_owned()));
    }
    if (step + player as u64).is_multiple_of(11) {
        events.push(GameEvent::PointerMove { x: (step * 3 + player as u64) as f32, y: step as f32 });
    }
    events
}

struct Peer {
    toy: Toy,
    net: Netplay,
    runtime: GameRuntime<Toy>,
    timeline: Timeline,
    /// Each frame's packet hash as it last ran.
    hashes: Vec<u64>,
    /// Ran its frames and the session can end.
    done: bool,
}

fn keep(hashes: &mut Vec<u64>, at: u64, hash: u64) {
    let at = at as usize;
    if hashes.len() <= at {
        hashes.resize(at + 1, 0);
    }
    hashes[at] = hash;
}

impl Peer {
    fn new(settings: Settings, hub: &Rc<RefCell<Hub>>, toy: Toy, start: &Moment) -> Peer {
        Peer::of_build(settings, hub, toy, start, 0xb17d)
    }

    fn of_build(settings: Settings, hub: &Rc<RefCell<Hub>>, toy: Toy, start: &Moment, build: u64) -> Peer {
        let link = Box::new(Wire { hub: hub.clone(), player: settings.player });
        let net = Netplay::new(settings, link, build, start, 77).unwrap();
        let mut runtime = ready(GameRuntime::new(&toy, (), Catalog::default(), &mut Viewer, None, None)).unwrap();
        let toy = toy.clone();
        runtime.hash_packets(true);
        ready(runtime.restore(&mut Viewer, start)).unwrap();
        let mut timeline = Timeline::default();
        timeline.keep_history(None);
        Peer { toy, net, runtime, timeline, hashes: Vec::new(), done: false }
    }

    /// One turn of a host's loop: take what arrived, correct, and run the
    /// next frame if it may.
    fn turn(&mut self, now: u64, frames: u64) -> anyhow::Result<()> {
        self.net.poll(now)?;
        if let Some(start) = self.net.take_start() {
            ready(self.runtime.restore(&mut Viewer, &start))?;
        }
        let hashes = &mut self.hashes;
        let mut note = |position: u64, _: &Recorded, runtime: &GameRuntime<Toy>| {
            keep(hashes, position, runtime.packet_hash());
            Ok(())
        };
        ready(self.net.correct(&mut self.timeline, &mut self.runtime, &Viewer, &mut note))?;
        if self.net.next() >= frames {
            self.done = self.net.done(frames);
            return Ok(());
        }
        if !self.net.ready()? {
            return Ok(());
        }
        self.net.prepare(&mut self.timeline);
        ready(self.timeline.prepare(&mut self.runtime, &Viewer))?;
        let position = self.timeline.position();
        let events = script(self.net.settings().player, position);
        let input = self.net.frame(position, &events, VIEWPORT)?;
        self.runtime.send(&input)?;
        let rendered = ready(self.runtime.receive(&Viewer))?;
        ready(self.runtime.follow(&mut Viewer, &rendered))?;
        self.net.ran(self.runtime.packet_hash());
        keep(&mut self.hashes, position, self.runtime.packet_hash());
        let restored = self.timeline.restored_here();
        self.timeline.ran(Recorded { input, loads: Vec::new(), restored });
        Ok(())
    }
}

/// Plays `frames` frames on every peer, a turn each per 4 ms; returns
/// them once every one is done.
fn play(peers: &mut [Peer], hub: &Rc<RefCell<Hub>>, frames: u64) -> anyhow::Result<()> {
    for _ in 0..100_000 {
        let now = hub.borrow().now;
        for peer in peers.iter_mut() {
            peer.turn(now, frames)?;
        }
        if peers.iter().all(|peer| peer.done) {
            return Ok(());
        }
        hub.borrow_mut().now += 4;
    }
    panic!("the session never ended");
}

/// The packets a straight run of `inputs` gives from `start`.
fn straight(start: &Moment, inputs: &[Vec<f32>]) -> Vec<u64> {
    let mut runtime =
        ready(GameRuntime::new(&Toy::default(), (), Catalog::default(), &mut Viewer, None, None)).unwrap();
    runtime.hash_packets(true);
    ready(runtime.restore(&mut Viewer, start)).unwrap();
    inputs
        .iter()
        .map(|input| {
            runtime.send(input).unwrap();
            ready(runtime.receive(&Viewer)).unwrap();
            runtime.packet_hash()
        })
        .collect()
}

fn hub(latency: u64, jitter: u64, loss: u64) -> Rc<RefCell<Hub>> {
    Rc::new(RefCell::new(Hub { latency, jitter, loss, dice: 9, ..Hub::default() }))
}

fn session(players: usize, delay: u8, hub: &Rc<RefCell<Hub>>, start: &Moment) -> Vec<Peer> {
    (0..players)
        .map(|player| {
            let settings = Settings { delay, ..Settings::new(players, player, VIEWPORT) };
            Peer::new(settings, hub, Toy::default(), start)
        })
        .collect()
}

/// Every peer ended with the same input and packets, a straight run's.
fn agree(peers: &[Peer], start: &Moment, frames: u64) -> Vec<Vec<f32>> {
    let inputs: Vec<Vec<f32>> = (0..frames).map(|at| peers[0].timeline.recorded(at).unwrap().input.clone()).collect();
    let expected = straight(start, &inputs);
    for (player, peer) in peers.iter().enumerate() {
        for at in 0..frames {
            assert_eq!(peer.timeline.recorded(at).unwrap().input, inputs[at as usize], "player {player}, frame {at}");
        }
        assert_eq!(peer.hashes[..frames as usize], expected, "player {player}'s packets");
    }
    inputs
}

#[test]
fn peers_agree_through_latency_loss_and_reordering() {
    let start = moment(vec![0.0; 9]);
    // 40 ms each way (10 turns) against a delay of 2 frames: most frames
    // run on predictions, and many are rolled back.
    let hub = hub(40, 12, 5);
    let mut peers = session(2, 2, &hub, &start);
    play(&mut peers, &hub, 240).unwrap();
    let inputs = agree(&peers, &start, 240);
    // Both players' keys are in it, behind their markers.
    let keys = |player: f32| {
        inputs
            .iter()
            .filter(|input| {
                let mut owner = -1.0;
                input[5..].as_chunks::<5>().0.iter().any(|event| {
                    if event[0] == 26.0 {
                        owner = event[1];
                    }
                    event[0] == 1.0 && owner == player
                })
            })
            .count()
    };
    assert!(keys(0.0) >= 30 && keys(1.0) >= 30, "{} and {}", keys(0.0), keys(1.0));
    for peer in &peers {
        let stats = peer.net.stats();
        assert!(stats.predicted > 100 && stats.rollbacks > 10, "{stats:?}");
        assert!(stats.longest <= 8, "rollbacks stay inside the window: {stats:?}");
        // Frames 0, 30, ... 210 compared, the last perhaps not yet.
        assert!((7..=8).contains(&stats.checked), "the peers compared packets: {stats:?}");
        // 40 ms each way, give or take 12, less what the peer held it.
        assert!(stats.round_trip.is_some_and(|trip| (80..=110).contains(&trip)), "{stats:?}");
        // A program for the start, its restore, and one a rollback: frames
        // an earlier rollback restored at do not restore again.
        assert_eq!(peer.toy.starts.get(), 2 + stats.rollbacks, "{stats:?}");
    }
}

#[test]
fn input_inside_the_delay_needs_no_rollback() {
    let start = moment(vec![0.0; 9]);
    let hub = hub(4, 0, 0);
    let mut peers = session(2, 3, &hub, &start);
    play(&mut peers, &hub, 120).unwrap();
    agree(&peers, &start, 120);
    for peer in &peers {
        assert_eq!((peer.net.stats().rollbacks, peer.net.stats().predicted), (0, 0), "{:?}", peer.net.stats());
    }
}

#[test]
fn three_players_agree() {
    let start = moment(vec![0.0; 9]);
    let hub = hub(20, 20, 10);
    let mut peers = session(3, 2, &hub, &start);
    play(&mut peers, &hub, 150).unwrap();
    agree(&peers, &start, 150);
    // One player's late input can take a rollback back past where another
    // player's began: it restores once, not again there.
    for peer in &peers {
        assert_eq!(peer.toy.starts.get(), 2 + peer.net.stats().rollbacks, "{:?}", peer.net.stats());
    }
}

#[test]
fn a_peer_that_started_elsewhere_takes_the_hosts_moment() {
    let start = moment(vec![5.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let hub = hub(10, 0, 0);
    let host = Peer::new(Settings::new(2, 0, VIEWPORT), &hub, Toy::default(), &start);
    // The other peer's own game is elsewhere, and its window another size:
    // the host's viewport is the session's.
    let other = Settings::new(2, 1, Viewport { width: 320, height: 200 });
    let guest = Peer::new(other, &hub, Toy::default(), &moment(vec![0.0; 9]));
    let mut peers = vec![host, guest];
    play(&mut peers, &hub, 60).unwrap();
    assert_eq!(peers[1].net.settings().viewport, VIEWPORT);
    agree(&peers, &start, 60);
}

#[test]
fn a_game_that_keeps_something_outside_its_state_is_caught() {
    let start = moment(vec![0.0; 9]);
    let hub = hub(8, 0, 0);
    let mut peers: Vec<Peer> = (0..2)
        .map(|player| {
            Peer::new(Settings::new(2, player, VIEWPORT), &hub, Toy { leak: player as f32, ..Toy::default() }, &start)
        })
        .collect();
    let error = play(&mut peers, &hub, 200).unwrap_err();
    let desync = error.downcast_ref::<Desync>().unwrap_or_else(|| panic!("{error:#}"));
    assert_eq!(desync.frame, 0, "the first frame compared");
}

#[test]
fn peers_of_another_build_or_session_are_refused() {
    let start = moment(vec![0.0; 9]);
    let wire = hub(1, 0, 0);
    let mut peers = vec![
        Peer::new(Settings::new(2, 0, VIEWPORT), &wire, Toy::default(), &start),
        Peer::of_build(Settings::new(2, 1, VIEWPORT), &wire, Toy::default(), &start, 0xbad),
    ];
    let error = play(&mut peers, &wire, 10).unwrap_err();
    assert!(format!("{error:#}").contains("another build"), "{error:#}");
    // Terms no session can keep.
    let settings = Settings { window: 0, ..Settings::new(2, 0, VIEWPORT) };
    let link = Box::new(Wire { hub: wire.clone(), player: 0 });
    assert!(Netplay::new(settings, link, 1, &start, 1).is_err());
    let hello = wire::Hello {
        session: 9,
        player: 0,
        players: 2,
        delay: 2,
        window: 8,
        fps: 0.0,
        width: 960,
        height: 640,
        build: 0xb17d,
        start: 0,
        heard: 1,
    };
    let alone = hub(1, 0, 0);
    let mut guest = vec![Peer::new(Settings::new(2, 1, VIEWPORT), &alone, Toy::default(), &start)];
    alone.borrow_mut().flying.push_back((0, 0, 1, wire::Message::Hello(hello).encode()));
    let error = play(&mut guest, &alone, 10).unwrap_err();
    assert!(format!("{error:#}").contains("no session can keep"), "{error:#}");
    // A stranger's datagrams are dropped, not trusted.
    let quiet = hub(1, 0, 0);
    let mut peers = session(2, 2, &quiet, &start);
    quiet.borrow_mut().flying.push_back((0, 1, 0, b"PNET\x01\x02junk".to_vec()));
    play(&mut peers, &quiet, 10).unwrap();
    assert!(peers[0].net.stats().dropped >= 1);
}
