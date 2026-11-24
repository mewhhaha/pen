//! Net sessions natively (`--net`): pen-runtime's `net` (the wire,
//! prediction and rollback; docs/netplay.md) over UDP, and what a session
//! does around it. Each peer is given every player's address in order and
//! its own player: `--net 1:127.0.0.1:7000,127.0.0.1:7001` is player 1 of
//! 2, listening on the second address. Datagrams from any other address
//! are dropped.
use std::collections::HashMap;
use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs, UdpSocket};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use pen_protocol::{RenderFrame, Viewport};
pub use pen_runtime::net::relay::{ChannelLink, ClientId, FrameDecoder, RelayBroker, RelayPacket, Transport};
pub use pen_runtime::net::{Link, Netplay, Settings, Stats};

use crate::game::{GameRuntime, Host, Moment};
use crate::timeline::{Timeline, Unseen};

/// The largest datagram a peer reads.
const DATAGRAM: usize = 65536;

/// NAT-traversal relay target parsed from `--net relay:PLAYER/PLAYERS@HOST:PORT/ROOM[#TOKEN]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelaySpec {
    pub server: SocketAddr,
    pub room: String,
    pub players: usize,
    pub token: u64,
}

/// Who this peer is and where every player's peer listens (direct UDP) or which
/// relay server & room to join.
#[derive(Clone, Debug, PartialEq)]
pub struct Spec {
    pub player: usize,
    pub addresses: Vec<SocketAddr>,
    pub relay: Option<RelaySpec>,
}

impl Spec {
    /// Parses either direct UDP `PLAYER:ADDRESS,ADDRESS,...` or
    /// relay `relay:PLAYER/PLAYERS@HOST:PORT/ROOM[#TOKEN]`.
    pub fn parse(text: &str) -> Result<Spec> {
        if let Some(rest) = text.strip_prefix("relay:") {
            let usage = "--net relay:PLAYER/PLAYERS@HOST:PORT/ROOM[#TOKEN]";
            let (who, target) = rest.split_once('@').context(usage)?;
            let (player_s, players_s) = who.split_once('/').context(usage)?;
            let player: usize = player_s.trim().parse().context(usage)?;
            let players: usize = players_s.trim().parse().context(usage)?;
            ensure!((2..=32).contains(&players) && player < players, "{usage}: 2 to 32 players, PLAYER < PLAYERS");
            let (addr_s, room_part) = target.split_once('/').context(usage)?;
            let server = addr_s
                .trim()
                .to_socket_addrs()?
                .next()
                .with_context(|| format!("{addr_s} names no address"))?;
            let (room, token) = match room_part.split_once('#') {
                Some((r, tok)) => (r.trim().to_owned(), tok.trim().parse::<u64>().context(usage)?),
                None => (room_part.trim().to_owned(), 0),
            };
            ensure!(!room.is_empty() && room.len() <= 255, "{usage}: room must be 1..=255 bytes");
            return Ok(Spec { player, addresses: Vec::new(), relay: Some(RelaySpec { server, room, players, token }) });
        }
        let usage = "--net takes PLAYER:ADDRESS,ADDRESS,... (every player's address, in order)";
        let (player, list) = text.split_once(':').context(usage)?;
        let player: usize = player.trim().parse().context(usage)?;
        let addresses = list
            .split(',')
            .map(|address| {
                let address = address.trim();
                address.to_socket_addrs()?.next().with_context(|| format!("{address} names no address"))
            })
            .collect::<Result<Vec<SocketAddr>>>()
            .context(usage)?;
        ensure!(
            (2..=32).contains(&addresses.len()) && player < addresses.len(),
            "{usage}: 2 to 32 players, and PLAYER one of them (from 0)"
        );
        Ok(Spec { player, addresses, relay: None })
    }

    /// Number of players in this session.
    pub fn players(&self) -> usize {
        self.relay.as_ref().map_or(self.addresses.len(), |r| r.players)
    }

    /// Opens the configured transport link (direct UDP or framed TCP relay).
    pub fn connect(&self) -> Result<Box<dyn Link>> {
        match &self.relay {
            Some(relay) => Ok(Box::new(TcpRelay::connect(relay.server, &relay.room, self.player, relay.players, relay.token)?)),
            None => Ok(Box::new(Udp::bind(self)?)),
        }
    }
}

/// Non-blocking length-prefixed TCP stream transport for [`ChannelLink`].
pub struct TcpTransport {
    stream: TcpStream,
    decoder: FrameDecoder,
    read_buf: Vec<u8>,
}

impl TcpTransport {
    pub fn connect(addr: SocketAddr) -> Result<Self> {
        let stream = TcpStream::connect(addr).with_context(|| format!("connecting relay {addr}"))?;
        let _ = stream.set_nodelay(true);
        stream.set_nonblocking(true)?;
        Ok(Self { stream, decoder: FrameDecoder::default(), read_buf: vec![0u8; 8192] })
    }
}

impl Transport for TcpTransport {
    fn send_frame(&mut self, bytes: &[u8]) {
        let Ok(len) = u16::try_from(bytes.len()) else { return };
        let mut framed = Vec::with_capacity(2 + bytes.len());
        framed.extend_from_slice(&len.to_le_bytes());
        framed.extend_from_slice(bytes);
        let _ = self.stream.write_all(&framed);
    }

    fn recv_frame(&mut self) -> Option<Vec<u8>> {
        loop {
            match self.stream.read(&mut self.read_buf) {
                Ok(0) => break,
                Ok(n) => self.decoder.push(&self.read_buf[..n]),
                Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }
        self.decoder.next_frame().ok().flatten()
    }
}

/// A framed TCP NAT-traversal relay [`Link`] connecting to a [`TcpRelayServer`].
pub struct TcpRelay(ChannelLink<TcpTransport>);

impl TcpRelay {
    pub fn connect(server: SocketAddr, room: &str, player: usize, players: usize, token: u64) -> Result<Self> {
        let transport = TcpTransport::connect(server)?;
        Ok(Self(ChannelLink::new(transport, room, player, players, token)?))
    }
}

impl Link for TcpRelay {
    fn send(&mut self, player: usize, bytes: &[u8]) {
        self.0.send(player, bytes);
    }

    fn receive(&mut self) -> Option<(usize, Vec<u8>)> {
        self.0.receive()
    }
}

#[cfg_attr(not(test), allow(dead_code))]
struct RelayClient {
    stream: TcpStream,
    decoder: FrameDecoder,
}

/// Non-blocking multi-room TCP relay server wrapping [`RelayBroker`].
#[cfg_attr(not(test), allow(dead_code))]
pub struct TcpRelayServer {
    listener: TcpListener,
    broker: RelayBroker,
    clients: HashMap<ClientId, RelayClient>,
    next_id: ClientId,
    read_buf: Vec<u8>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl TcpRelayServer {
    pub fn bind(addr: impl ToSocketAddrs) -> Result<Self> {
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        Ok(Self {
            listener,
            broker: RelayBroker::new(),
            clients: HashMap::new(),
            next_id: 1,
            read_buf: vec![0u8; 8192],
        })
    }

    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.listener.local_addr()?)
    }

    /// Accepts pending connections and routes all currently available packets.
    pub fn poll(&mut self) {
        loop {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    let _ = stream.set_nodelay(true);
                    if stream.set_nonblocking(true).is_ok() {
                        let id = self.next_id;
                        self.next_id += 1;
                        self.clients.insert(id, RelayClient { stream, decoder: FrameDecoder::default() });
                    }
                }
                Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }
        let ids: Vec<ClientId> = self.clients.keys().copied().collect();
        let mut outgoing: Vec<(ClientId, Vec<u8>)> = Vec::new();
        let mut dropped = Vec::new();
        for id in ids {
            let Some(client) = self.clients.get_mut(&id) else { continue };
            loop {
                match client.stream.read(&mut self.read_buf) {
                    Ok(0) => {
                        dropped.push(id);
                        break;
                    }
                    Ok(n) => client.decoder.push(&self.read_buf[..n]),
                    Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                    Err(_) => {
                        dropped.push(id);
                        break;
                    }
                }
            }
            while let Ok(Some(raw)) = client.decoder.next_frame() {
                if let Ok(packet) = RelayPacket::decode(&raw) {
                    for (dst, reply) in self.broker.handle(id, packet) {
                        if let Ok(framed) = reply.encode_framed() {
                            outgoing.push((dst, framed));
                        }
                    }
                }
            }
        }
        for (dst, framed) in outgoing {
            if let Some(target) = self.clients.get_mut(&dst)
                && target.stream.write_all(&framed).is_err()
            {
                dropped.push(dst);
            }
        }
        for id in dropped {
            self.clients.remove(&id);
            self.broker.disconnect(id);
        }
    }
}

/// UDP between the peers.
pub struct Udp {
    socket: UdpSocket,
    peers: Vec<SocketAddr>,
    buffer: Vec<u8>,
}

impl Udp {
    /// Listens on the spec's own address.
    pub fn bind(spec: &Spec) -> Result<Udp> {
        let own = spec.addresses[spec.player];
        let socket = UdpSocket::bind(own).with_context(|| format!("listening on {own}"))?;
        Udp::new(socket, spec.addresses.clone())
    }

    /// Talks through `socket` to `peers`, every player's address in order.
    pub fn new(socket: UdpSocket, peers: Vec<SocketAddr>) -> Result<Udp> {
        socket.set_nonblocking(true)?;
        Ok(Udp { socket, peers, buffer: vec![0; DATAGRAM] })
    }
}

impl Link for Udp {
    fn send(&mut self, player: usize, bytes: &[u8]) {
        // Lost is allowed: the next message says it again.
        let _ = self.socket.send_to(bytes, self.peers[player]);
    }

    fn receive(&mut self) -> Option<(usize, Vec<u8>)> {
        loop {
            match self.socket.recv_from(&mut self.buffer) {
                Ok((length, from)) => {
                    if let Some(player) = self.peers.iter().position(|peer| *peer == from) {
                        return Some((player, self.buffer[..length].to_vec()));
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return None,
                // A peer not listening yet answers with an ICMP error,
                // which a later read reports: it is only a datagram lost.
                Err(_) => {}
            }
        }
    }
}

/// A net session as native pen runs it: a peer, and the clock its
/// messages are timed by.
pub struct Net {
    pub play: Netplay,
    clock: Instant,
}

impl Net {
    /// Starts the peer `spec` names, for the game build `wasm`, starting
    /// from `start` in a window of size `viewport` (the session's, if this
    /// is the host).
    pub fn start(spec: &Spec, link: Box<dyn Link>, build: u64, start: &Moment, viewport: Viewport) -> Result<Net> {
        let settings = Settings::new(spec.players(), spec.player, viewport);
        // The host's nonce: this run's, not an earlier one's on the same port.
        let since = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        let seed = [since.as_nanos().to_le_bytes().as_slice(), &std::process::id().to_le_bytes()].concat();
        let nonce = u32::try_from(pen_runtime::net::wire::hash_bytes(&seed) >> 32).unwrap_or(1);
        Ok(Net { play: Netplay::new(settings, link, build, start, nonce)?, clock: Instant::now() })
    }

    /// Milliseconds since the session started.
    fn now(&self) -> u64 {
        u64::try_from(self.clock.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// Takes what arrived and sends what is due.
    pub fn poll(&mut self) -> Result<()> {
        let now = self.now();
        self.play.poll(now)
    }

    /// Rolls back what ran on a wrong prediction (`Netplay::correct`).
    pub fn correct(
        &mut self,
        timeline: &mut Timeline,
        runtime: &mut GameRuntime,
        host: &dyn Host,
        unseen: &mut Unseen,
    ) -> Result<Option<(RenderFrame, u64)>> {
        pollster::block_on(self.play.correct(timeline, &mut **runtime, host, unseen))
    }

    /// A line for the log: how the session went.
    pub fn report(&self) -> String {
        let Stats { frames, predicted, rollbacks, rerun, longest, stalls, round_trip, checked, .. } = self.play.stats();
        let trip = round_trip.map_or_else(String::new, |trip| format!(", {trip} ms round trip"));
        format!(
            "net: {frames} frames, {predicted} on predictions; {rollbacks} rollbacks ran {rerun} frames again (at most \
             {longest}); waited {stalls} times; {checked} frames compared with the peers{trip}"
        )
    }
}

/// Fails unless the game takes `players` players (its `players`).
pub fn check_players(declared: u32, players: usize) -> Result<()> {
    if declared < 2 {
        bail!(
            "this game takes one player: a net session needs a game that declares its players (pen.json's \
             \"game\": {{\"players\": N}})"
        );
    }
    ensure!(players <= declared as usize, "this game takes at most {declared} players, not {players}");
    Ok(())
}

#[cfg(test)]
mod tests;
