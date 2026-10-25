//! Net sessions natively (`--net`): pen-runtime's `net` (the wire,
//! prediction and rollback; docs/netplay.md) over UDP, and what a session
//! does around it. Each peer is given every player's address in order and
//! its own player: `--net 1:127.0.0.1:7000,127.0.0.1:7001` is player 1 of
//! 2, listening on the second address. Datagrams from any other address
//! are dropped.
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use pen_protocol::{RenderFrame, Viewport};
pub use pen_runtime::net::{Link, Netplay, Settings, Stats};

use crate::game::{GameRuntime, Host, Moment};
use crate::timeline::{Timeline, Unseen};

/// The largest datagram a peer reads.
const DATAGRAM: usize = 65536;

/// Who this peer is and where every player's peer listens.
#[derive(Clone, Debug, PartialEq)]
pub struct Spec {
    pub player: usize,
    pub addresses: Vec<SocketAddr>,
}

impl Spec {
    /// `PLAYER:ADDRESS,ADDRESS,...`, every player's address in order.
    pub fn parse(text: &str) -> Result<Spec> {
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
        Ok(Spec { player, addresses })
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
        let settings = Settings::new(spec.addresses.len(), spec.player, viewport);
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
