//! A peer of a net session (`Netplay`): it shakes hands with every other
//! peer, sends its player's input to them and takes theirs, gives each
//! frame the input every player is known or predicted to have, and when a
//! prediction proves wrong rolls the game back through the timeline.
use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail, ensure};
use pen_protocol::{GameEvent, RenderFrame, Viewport};

use super::inputs::{self, Inputs};
use super::wire::{self, CHUNK_BYTES, DATAGRAM_BYTES, Hello, MAX_FRAMES, Message};
use crate::game::{GameRuntime, Host, Moment};
use crate::program::Backend;
use crate::timeline::{Timeline, Unseen};

/// Sends and receives datagrams between the peers, which may lose,
/// repeat or reorder them. Native pen's is UDP; a test's is a queue.
pub trait Link {
    /// Sends a datagram to `player`'s peer.
    fn send(&mut self, player: usize, bytes: &[u8]);
    /// A datagram that arrived, and which player's peer sent it; None when
    /// none waits.
    fn receive(&mut self) -> Option<(usize, Vec<u8>)>;
}

/// What a peer is told of its session. The session host's (player 0's)
/// delay, window, frame rate and viewport are everyone's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub players: usize,
    /// This peer's player.
    pub player: usize,
    /// Frames between a player's input and the frame it plays in: input
    /// that arrives within it needs no prediction.
    pub delay: u8,
    /// Frames a peer runs past the last one every player's input is known
    /// for, at most; a rollback runs at most this many again.
    pub window: u8,
    /// Each frame is `1 / fps` seconds of game time.
    pub fps: f32,
    /// The viewport every game runs with: the host's.
    pub viewport: Viewport,
    /// How long a peer may go unheard, in milliseconds.
    pub timeout: u64,
}

impl Settings {
    pub fn new(players: usize, player: usize, viewport: Viewport) -> Settings {
        Settings { players, player, delay: 2, window: 8, fps: 60.0, viewport, timeout: 5000 }
    }
}

/// How a session has gone, for diagnostics and the report at its end.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    /// Frames run, and of those, on a prediction of someone's input.
    pub frames: u64,
    pub predicted: u64,
    /// Rollbacks, the frames they ran again, and the most one did.
    pub rollbacks: u64,
    pub rerun: u64,
    pub longest: u64,
    /// Times a frame waited for input (past the window, or connecting).
    pub stalls: u64,
    /// Datagrams sent and received, and received ones that did not decode
    /// or belong.
    pub sent: u64,
    pub received: u64,
    pub dropped: u64,
    /// The round trip to the slowest peer, in milliseconds.
    pub round_trip: Option<u64>,
    /// Frames whose packets the peers compared and found equal.
    pub checked: u64,
}

/// The peers' games no longer agree: from the same input they gave
/// different packets, so one kept something outside its state or runs
/// another game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Desync {
    pub frame: u64,
    pub player: usize,
}

impl std::fmt::Display for Desync {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "the game parted ways with player {}'s at frame {}: the same input gave other packets",
            self.player, self.frame
        )
    }
}

impl std::error::Error for Desync {}

/// Frames between the packets peers compare.
pub const CHECK_INTERVAL: u64 = 30;
/// Milliseconds between hellos, and at most between input messages.
const HELLO_EVERY: u64 = 100;
const SEND_EVERY: u64 = 4;
/// Final hashes kept to compare with peers' late ones.
const KEPT_CHECKS: usize = 64;

/// A clock's milliseconds as an input message's stamp, never 0 (none).
fn stamp(now: u64) -> u32 {
    u32::try_from(now & 0xffff_ffff).unwrap_or(0).max(1)
}

/// What this peer knows of another.
#[derive(Debug, Default)]
struct Peer {
    hello: Option<Hello>,
    /// Frames of this peer's player's input it holds.
    acked: u64,
    /// Its next frame to run, as it last said.
    frame: u64,
    /// When it was last heard, and its last stamp and when that came.
    heard: u64,
    stamp: u32,
    stamped: u64,
    /// The round trip to it, smoothed, in milliseconds.
    trip: Option<u64>,
    /// Whether an input message came from it: it is in the session.
    joined: bool,
    left: bool,
    /// A frame and its packet's hash it compared, not yet checked here,
    /// and the last frame checked.
    check: Option<(u64, u64)>,
    compared: Option<u64>,
    /// The host's moment arriving, chunk by chunk (from player 0 only).
    moment: Vec<u8>,
    chunks: Vec<bool>,
}

pub struct Netplay {
    settings: Settings,
    link: Box<dyn Link>,
    /// The host's nonce: datagrams from another session are dropped.
    session: u32,
    build: u64,
    /// The moment this peer starts from, and its hash.
    start: Vec<u8>,
    start_hash: u64,
    /// The host's moment, taken in place of this peer's own.
    adopted: Option<Moment>,
    peers: Vec<Peer>,
    /// Every player's input, once connected.
    inputs: Option<Inputs>,
    /// The timeline position of frame 0.
    origin: Option<u64>,
    /// Each frame's packet hash as it last ran, while it may run again.
    hashes: BTreeMap<u64, u64>,
    /// Hashes of frames every input of which is known, to compare.
    checks: BTreeMap<u64, u64>,
    /// Frames before it are final and their checks taken.
    settled: u64,
    sent_hello: Option<u64>,
    sent_input: Option<u64>,
    /// New input to send.
    unsent: bool,
    now: u64,
    stats: Stats,
}

impl Netplay {
    /// A peer starting from `start` with the game build `build` (a hash
    /// of its program), talking over `link`. The host's nonce `session`
    /// tells this session's datagrams from an earlier one's; other peers
    /// take the host's.
    pub fn new(settings: Settings, link: Box<dyn Link>, build: u64, start: &Moment, session: u32) -> Result<Netplay> {
        ensure!(
            (2..=32).contains(&settings.players) && settings.player < settings.players,
            "a net session has 2 to 32 players, this peer one of them"
        );
        ensure!(settings.fps.is_finite() && settings.fps > 0.0, "a net session's frame rate must be positive");
        ensure!(settings.window >= 1, "a net session's peers run at least a frame past what is known");
        let start = wire::encode_moment(start);
        let start_hash = wire::hash_bytes(&start);
        let peers = (0..settings.players).map(|_| Peer::default()).collect();
        let session = if settings.player == 0 { session.max(1) } else { 0 };
        Ok(Netplay {
            settings,
            link,
            session,
            build,
            start,
            start_hash,
            adopted: None,
            peers,
            inputs: None,
            origin: None,
            hashes: BTreeMap::new(),
            checks: BTreeMap::new(),
            settled: 0,
            sent_hello: None,
            sent_input: None,
            unsent: false,
            now: 0,
            stats: Stats::default(),
        })
    }

    /// The session's terms: the host's, once connected.
    pub fn settings(&self) -> Settings {
        self.settings
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// Every peer has answered and agreed, and play can begin.
    pub fn connected(&self) -> bool {
        self.inputs.is_some()
    }

    /// The host's moment, once, when this peer started from another: the
    /// game restores it before frame 0.
    pub fn take_start(&mut self) -> Option<Moment> {
        self.adopted.take()
    }

    fn others(&self) -> impl Iterator<Item = usize> + use<> {
        let player = self.settings.player;
        (0..self.settings.players).filter(move |other| *other != player)
    }

    fn send(&mut self, player: usize, message: &Message) {
        self.link.send(player, &message.encode());
        self.stats.sent += 1;
    }

    /// Takes what arrived and sends what is due; `now` is a clock in
    /// milliseconds. Fails when a peer disagrees, leaves, goes quiet or
    /// its game parts ways with this one's ([`Desync`]).
    pub fn poll(&mut self, now: u64) -> Result<()> {
        self.now = now;
        while let Some((player, bytes)) = self.link.receive() {
            self.stats.received += 1;
            let message = match Message::decode(&bytes) {
                Ok(message) if player < self.settings.players && player != self.settings.player => message,
                _ => {
                    self.stats.dropped += 1;
                    continue;
                }
            };
            if !self.take(player, message)? {
                self.stats.dropped += 1;
            }
        }
        for player in self.others() {
            let peer = &self.peers[player];
            if !peer.left && peer.hello.is_some() && now.saturating_sub(peer.heard) > self.settings.timeout {
                bail!("player {player}'s peer went quiet");
            }
        }
        self.connect();
        // Hellos until every peer is in, then input.
        let hello_due = self.sent_hello.is_none_or(|sent| now >= sent + HELLO_EVERY);
        if hello_due && self.others().any(|player| !self.peers[player].joined) {
            self.sent_hello = Some(now);
            self.hello();
        }
        let input_due = self.unsent || self.sent_input.is_none_or(|sent| now >= sent + SEND_EVERY);
        if self.connected() && input_due {
            self.sent_input = Some(now);
            self.unsent = false;
            for player in self.others() {
                self.send_input(player);
            }
        }
        Ok(())
    }

    fn hello(&mut self) {
        let heard = self
            .peers
            .iter()
            .enumerate()
            .filter(|(_, peer)| peer.hello.is_some())
            .fold(1u32 << self.settings.player, |bits, (player, _)| bits | 1 << player);
        let settings = self.settings;
        let hello = Hello {
            session: self.session,
            player: u8::try_from(settings.player).expect("at most 32 players"),
            players: u8::try_from(settings.players).expect("at most 32 players"),
            delay: settings.delay,
            window: settings.window,
            fps: settings.fps,
            width: settings.viewport.width,
            height: settings.viewport.height,
            build: self.build,
            start: self.start_hash,
            heard,
        };
        for player in self.others() {
            self.send(player, &Message::Hello(hello.clone()));
            // A peer that started elsewhere takes the host's moment.
            if settings.player == 0
                && self.peers[player].hello.as_ref().is_some_and(|hello| hello.start != self.start_hash)
            {
                let total = u32::try_from(self.start.len()).expect("moments are bounded");
                for (index, chunk) in self.start.clone().chunks(CHUNK_BYTES).enumerate() {
                    let offset = u32::try_from(index * CHUNK_BYTES).expect("moments are bounded");
                    let message = Message::Moment { session: self.session, total, offset, bytes: chunk.to_vec() };
                    self.send(player, &message);
                }
            }
        }
    }

    /// Takes a message from `player`'s peer; whether it belonged.
    fn take(&mut self, player: usize, message: Message) -> Result<bool> {
        let session = match &message {
            Message::Hello(hello) => hello.session,
            Message::Input(inputs) => inputs.session,
            Message::Moment { session, .. } | Message::Bye { session, .. } => *session,
        };
        // Peers that have not heard the host yet say hello with no
        // session; the host's hello gives a peer its session.
        let belongs = match &message {
            Message::Hello(_) => session == 0 || session == self.session || self.session == 0 && player == 0,
            _ => self.session != 0 && session == self.session,
        };
        if !belongs {
            return Ok(false);
        }
        self.peers[player].heard = self.now;
        match message {
            Message::Hello(hello) => self.hear(player, hello)?,
            Message::Input(message) => {
                let Some(inputs) = &mut self.inputs else { return Ok(false) };
                if usize::from(message.player) != player {
                    return Ok(false);
                }
                for records in &message.frames {
                    inputs::check(player, records)?;
                }
                inputs.receive(player, u64::from(message.first), message.frames);
                let peer = &mut self.peers[player];
                peer.joined = true;
                peer.acked = peer.acked.max(u64::from(message.ack));
                peer.frame = u64::from(message.frame);
                peer.stamp = message.stamp;
                peer.stamped = self.now;
                if message.echo != 0 {
                    // Less the time the peer held the stamp before echoing it.
                    let sample = u64::from(stamp(self.now).wrapping_sub(message.echo).saturating_sub(message.held));
                    peer.trip = Some(peer.trip.map_or(sample, |trip| (trip * 7 + sample) / 8));
                }
                if message.checked != 0 || message.hash != 0 {
                    peer.check = Some((u64::from(message.checked), message.hash));
                }
                self.stats.round_trip = self.peers.iter().filter_map(|peer| peer.trip).max();
                self.compare()?;
            }
            Message::Moment { total, offset, bytes, .. } => {
                if player != 0 || self.settings.player == 0 || self.adopted.is_some() || self.connected() {
                    return Ok(false);
                }
                self.chunk(total as usize, offset as usize, &bytes)?;
            }
            Message::Bye { .. } => self.peers[player].left = true,
        }
        Ok(true)
    }

    fn hear(&mut self, player: usize, hello: Hello) -> Result<()> {
        ensure!(
            usize::from(hello.player) == player && usize::from(hello.players) == self.settings.players,
            "player {player}'s peer plays another session: player {} of {}",
            hello.player,
            hello.players
        );
        ensure!(hello.build == self.build, "player {player}'s peer runs another build of the game");
        let viewport = 1..=0xffffff;
        ensure!(
            hello.fps > 0.0 && hello.window >= 1 && viewport.contains(&hello.width) && viewport.contains(&hello.height),
            "player {player}'s peer offers terms no session can keep: {} fps, a window of {} frames, {}x{}",
            hello.fps,
            hello.window,
            hello.width,
            hello.height
        );
        if player == 0 && self.settings.player != 0 && !self.connected() {
            // The host's terms are the session's.
            self.session = hello.session;
            let settings = &mut self.settings;
            settings.delay = hello.delay;
            settings.window = hello.window;
            settings.fps = hello.fps;
            settings.viewport = Viewport { width: hello.width, height: hello.height };
        }
        self.peers[player].hello = Some(hello);
        Ok(())
    }

    /// A chunk of the host's moment; the whole of it, checked against the
    /// hash the host said it starts from, becomes this peer's start.
    fn chunk(&mut self, total: usize, offset: usize, bytes: &[u8]) -> Result<()> {
        let Some(host) = self.peers[0].hello.as_ref().map(|hello| hello.start) else { return Ok(()) };
        let peer = &mut self.peers[0];
        if peer.moment.len() != total {
            peer.moment = vec![0; total];
            peer.chunks = vec![false; total.div_ceil(CHUNK_BYTES)];
        }
        peer.moment[offset..offset + bytes.len()].copy_from_slice(bytes);
        peer.chunks[offset / CHUNK_BYTES] = true;
        if peer.chunks.iter().all(|chunk| *chunk) && wire::hash_bytes(&peer.moment) == host {
            let bytes = std::mem::take(&mut peer.moment);
            self.adopted = Some(wire::decode_moment(&bytes)?);
            self.start_hash = host;
            self.start = bytes;
        }
        Ok(())
    }

    /// Play begins once every peer's hello agrees and has heard this one.
    fn connect(&mut self) {
        if self.connected() || self.session == 0 {
            return;
        }
        let me = 1u32 << self.settings.player;
        let host = self.peers[0].hello.as_ref().map_or(self.start_hash, |hello| hello.start);
        let agreed = self.others().all(|player| {
            self.peers[player].hello.as_ref().is_some_and(|hello| hello.heard & me != 0 && hello.start == host)
        });
        if agreed && self.start_hash == host {
            self.inputs = Some(Inputs::new(self.settings.players, u32::from(self.settings.delay)));
        }
    }

    fn send_input(&mut self, player: usize) {
        let Some(inputs) = &self.inputs else { return };
        let local = self.settings.player;
        let from = self.peers[player].acked.max(inputs.base(local));
        let mut frames = Vec::new();
        let mut bytes = 64;
        for records in inputs.frames_from(local, from, MAX_FRAMES) {
            bytes += wire::frame_bytes(records);
            if !frames.is_empty() && bytes > DATAGRAM_BYTES {
                break;
            }
            frames.push(records.to_vec());
        }
        let (checked, hash) = self.checks.last_key_value().map_or((0, 0), |(frame, hash)| (*frame, *hash));
        let narrow = |value: u64| u32::try_from(value).unwrap_or(u32::MAX);
        let message = Message::Input(wire::Inputs {
            session: self.session,
            player: u8::try_from(local).expect("at most 32 players"),
            frame: narrow(inputs.next()),
            ack: narrow(inputs.known(player)),
            stamp: stamp(self.now),
            echo: self.peers[player].stamp,
            held: narrow(self.now.saturating_sub(self.peers[player].stamped)),
            checked: narrow(checked),
            hash,
            first: narrow(from),
            frames,
        });
        self.send(player, &message);
    }

    /// Whether the next frame may run: play has begun and it is within the
    /// window of the last frame every input is known for.
    pub fn ready(&mut self) -> Result<bool> {
        let Some(inputs) = &self.inputs else {
            self.stats.stalls += 1;
            return Ok(false);
        };
        let (next, confirmed) = (inputs.next(), inputs.confirmed());
        if next < confirmed + u64::from(self.settings.window) {
            return Ok(true);
        }
        // A peer that left sends nothing more.
        if let Some(player) = self.others().find(|player| self.peers[*player].left && inputs.known(*player) <= next) {
            bail!("player {player} left the session at frame {}", inputs.known(player));
        }
        self.stats.stalls += 1;
        Ok(false)
    }

    fn elapsed(&self) -> f32 {
        1.0 / self.settings.fps
    }

    /// The local player's `events` (from a window of size `local`) become
    /// its input `delay` frames on; returns the input the frame at
    /// timeline `position`, the next, runs on.
    pub fn frame(&mut self, position: u64, events: &[GameEvent], local: Viewport) -> Result<Vec<f32>> {
        let settings = self.settings;
        let inputs = self.inputs.as_mut().ok_or_else(|| anyhow!("the net session has not begun"))?;
        let origin = *self.origin.get_or_insert(position);
        ensure!(position == origin + inputs.next(), "a net session's frames run in order, from its first");
        let records = inputs::records(events, local, settings.viewport, settings.player == 0)?;
        inputs.push(settings.player, records);
        self.unsent = true;
        let next = inputs.next();
        if (0..inputs.players()).any(|player| inputs.get(player, next).is_none()) {
            self.stats.predicted += 1;
        }
        Ok(inputs.frame(next, 1.0 / settings.fps, settings.viewport))
    }

    /// The frame `frame` gave and ran on was taken: its packet's hash.
    pub fn ran(&mut self, packet: u64) {
        let Some(inputs) = &mut self.inputs else { return };
        let frame = inputs.next();
        inputs.ran(frame);
        self.hashes.insert(frame, packet);
        self.stats.frames += 1;
    }

    /// Before each frame: a moment before every frame that may run
    /// again, from the first one whose input is not all known. The first
    /// frame prepared is the session's frame 0.
    pub fn prepare(&mut self, timeline: &mut Timeline) {
        if self.inputs.is_some() && self.origin.is_none() {
            self.origin = Some(timeline.position());
        }
        let from = self.inputs.as_ref().zip(self.origin).map(|(inputs, origin)| origin + inputs.confirmed());
        timeline.moments_from(from);
    }

    /// When frames ran on a prediction that proved wrong: gives them the
    /// input now known, and runs the game again from the first of them to
    /// where it was (unseen, as a rewind does). Returns the frame the last
    /// one showed and where the restored moment was. Then compares the
    /// frames that became final with the peers'.
    pub async fn correct<B: Backend>(
        &mut self,
        timeline: &mut Timeline,
        runtime: &mut GameRuntime<B>,
        host: &dyn Host,
        unseen: &mut Unseen<'_, B>,
    ) -> Result<Option<(RenderFrame, u64)>> {
        let (elapsed, viewport) = (self.elapsed(), self.settings.viewport);
        let (Some(inputs), Some(origin)) = (&mut self.inputs, self.origin) else { return Ok(None) };
        let mut shown = None;
        if let Some(wrong) = inputs.take_wrong() {
            let next = inputs.next();
            for frame in wrong..next {
                timeline.amend(origin + frame, inputs.frame(frame, elapsed, viewport));
            }
            self.prepare(timeline);
            let hashes = &mut self.hashes;
            let mut note = |position: u64, recorded: &crate::timeline::Recorded, runtime: &GameRuntime<B>| {
                hashes.insert(position - origin, runtime.packet_hash());
                unseen(position, recorded, runtime)
            };
            shown = Some(timeline.rewind(runtime, host, origin + next, &mut note).await?);
            let stats = &mut self.stats;
            stats.rollbacks += 1;
            stats.rerun += next - wrong;
            stats.longest = stats.longest.max(next - wrong);
        }
        self.settle();
        self.compare()?;
        Ok(shown)
    }

    /// Frames before the confirmed one ran on known input and will not run
    /// again: every `CHECK_INTERVAL`th one's hash is kept to compare.
    fn settle(&mut self) {
        let acked = self.others().map(|player| self.peers[player].acked).min().unwrap_or(0);
        let Some(inputs) = &mut self.inputs else { return };
        let end = inputs.confirmed().min(inputs.next());
        for frame in self.settled..end {
            if let Some(hash) = self.hashes.remove(&frame)
                && frame.is_multiple_of(CHECK_INTERVAL)
            {
                self.checks.insert(frame, hash);
            }
        }
        self.settled = self.settled.max(end);
        while self.checks.len() > KEPT_CHECKS {
            self.checks.pop_first();
        }
        // What is needed no more: frames before the final ones, which no
        // rollback merges again; and this player's only once every peer
        // holds them too.
        for player in 0..inputs.players() {
            let before = if player == self.settings.player { acked.min(self.settled) } else { self.settled };
            inputs.forget_before(player, before);
        }
    }

    /// Compares the peers' final hashes with this one's.
    fn compare(&mut self) -> Result<()> {
        for player in self.others() {
            let peer = &mut self.peers[player];
            let Some((frame, hash)) = peer.check else { continue };
            if frame >= self.settled {
                continue;
            }
            peer.check = None;
            if peer.compared.is_some_and(|compared| compared >= frame) {
                continue;
            }
            match self.checks.get(&frame) {
                Some(ours) if *ours != hash => return Err(Desync { frame, player }.into()),
                Some(_) => {
                    peer.compared = Some(frame);
                    self.stats.checked += 1;
                }
                None => {}
            }
        }
        Ok(())
    }

    /// Whether the session can end after `frames` frames: every input for
    /// them is known here, and every peer holds this player's.
    pub fn done(&self, frames: u64) -> bool {
        let Some(inputs) = &self.inputs else { return false };
        inputs.confirmed() >= frames
            && self.settled >= frames.min(inputs.next())
            && self.others().all(|player| self.peers[player].acked >= frames || self.peers[player].left)
    }

    /// Says goodbye to every peer.
    pub fn leave(&mut self) {
        let message = Message::Bye { session: self.session, player: u8::try_from(self.settings.player).unwrap_or(0) };
        for player in self.others() {
            self.send(player, &message);
        }
    }

    /// Frames before this timeline position are final: they ran on every
    /// player's input and will not run again.
    pub fn settled(&self) -> Option<u64> {
        self.origin.map(|origin| origin + self.settled)
    }

    /// The next frame to run, counted from the session's first.
    pub fn next(&self) -> u64 {
        self.inputs.as_ref().map_or(0, Inputs::next)
    }

    /// Frames before it have every player's input known.
    pub fn confirmed(&self) -> u64 {
        self.inputs.as_ref().map_or(0, Inputs::confirmed)
    }
}
