//! Framed NAT-traversal relay and WebRTC DataChannel transport for [`super::Netplay`].
//!
//! Direct UDP peer-to-peer requires open ports on every peer. This module defines
//! a compact binary framing protocol ([`RelayPacket`]), an embeddable multi-room
//! packet router ([`RelayBroker`]), and a generic [`ChannelLink`] implementing
//! [`super::Link`] over any framed message/stream [`Transport`] (TCP relay socket,
//! WebSocket, or WebRTC `RTCDataChannel`).
use std::collections::{HashMap, VecDeque};

use anyhow::{Result, bail, ensure};

use super::play::Link;

/// Maximum encoded packet size (fits in a `u16` length prefix).
pub const MAX_FRAME_BYTES: usize = 65500;

const TAG_JOIN: u8 = 1;
const TAG_JOINED: u8 = 2;
const TAG_FORWARD: u8 = 3;
const TAG_HEARTBEAT: u8 = 4;
const TAG_ACK: u8 = 5;
const TAG_LEAVE: u8 = 6;
const TAG_LIST_ROOMS: u8 = 7;
const TAG_ROOMS: u8 = 8;

/// Public lobby summary of an active relay room for room discovery.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoomSummary {
    pub room: String,
    pub occupied: u8,
    pub capacity: u8,
}

/// Control and data packets exchanged between a peer and a relay server or
/// multiplexed WebRTC DataChannel bridge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RelayPacket {
    /// Register this connection as `player` of `players` in `room` with auth `token`.
    Join { room: String, player: u8, players: u8, token: u64 },
    /// Relay acknowledges room membership.
    Joined { player: u8, players: u8 },
    /// Forward a `Netplay` datagram between two players in the same room.
    Forward { from: u8, to: u8, payload: Vec<u8> },
    /// Keepalive / RTT probe.
    Heartbeat { nonce: u32 },
    /// Reply to a [`RelayPacket::Heartbeat`].
    Ack { nonce: u32 },
    /// Graceful peer disconnect.
    Leave { player: u8 },
    /// Lobby room discovery request.
    ListRooms,
    /// Lobby room discovery response.
    Rooms(Vec<RoomSummary>),
}

impl RelayPacket {
    /// Encodes this packet into a compact binary payload (without outer length prefix).
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        match self {
            Self::Join { room, player, players, token } => {
                let bytes = room.as_bytes();
                let room_len =
                    u8::try_from(bytes.len()).map_err(|_| anyhow::anyhow!("relay room name exceeds 255 bytes"))?;
                out.push(TAG_JOIN);
                out.push(*player);
                out.push(*players);
                out.extend_from_slice(&token.to_le_bytes());
                out.push(room_len);
                out.extend_from_slice(bytes);
            }
            Self::Joined { player, players } => {
                out.push(TAG_JOINED);
                out.push(*player);
                out.push(*players);
            }
            Self::Forward { from, to, payload } => {
                ensure!(payload.len() + 3 <= MAX_FRAME_BYTES, "relay payload too large");
                out.push(TAG_FORWARD);
                out.push(*from);
                out.push(*to);
                out.extend_from_slice(payload);
            }
            Self::Heartbeat { nonce } => {
                out.push(TAG_HEARTBEAT);
                out.extend_from_slice(&nonce.to_le_bytes());
            }
            Self::Ack { nonce } => {
                out.push(TAG_ACK);
                out.extend_from_slice(&nonce.to_le_bytes());
            }
            Self::Leave { player } => {
                out.push(TAG_LEAVE);
                out.push(*player);
            }
            Self::ListRooms => out.push(TAG_LIST_ROOMS),
            Self::Rooms(rooms) => {
                out.push(TAG_ROOMS);
                let count = u16::try_from(rooms.len()).map_err(|_| anyhow::anyhow!("too many rooms"))?;
                out.extend_from_slice(&count.to_le_bytes());
                for r in rooms {
                    let bytes = r.room.as_bytes();
                    let rlen = u8::try_from(bytes.len()).map_err(|_| anyhow::anyhow!("room name > 255 bytes"))?;
                    out.push(r.occupied);
                    out.push(r.capacity);
                    out.push(rlen);
                    out.extend_from_slice(bytes);
                }
                ensure!(out.len() <= MAX_FRAME_BYTES, "rooms listing exceeds max frame");
            }
        }
        Ok(out)
    }

    /// Decodes a packet from its binary payload.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let (&tag, rest) = bytes.split_first().ok_or_else(|| anyhow::anyhow!("empty relay packet"))?;
        match tag {
            TAG_JOIN => {
                ensure!(rest.len() >= 11, "truncated Join packet");
                let player = rest[0];
                let players = rest[1];
                let token = u64::from_le_bytes(rest[2..10].try_into().unwrap());
                let room_len = usize::from(rest[10]);
                ensure!(rest.len() == 11 + room_len, "malformed Join room length");
                let room = std::str::from_utf8(&rest[11..])?.to_owned();
                Ok(Self::Join { room, player, players, token })
            }
            TAG_JOINED => {
                ensure!(rest.len() == 2, "truncated Joined packet");
                Ok(Self::Joined { player: rest[0], players: rest[1] })
            }
            TAG_FORWARD => {
                ensure!(rest.len() >= 2, "truncated Forward packet");
                Ok(Self::Forward { from: rest[0], to: rest[1], payload: rest[2..].to_vec() })
            }
            TAG_HEARTBEAT => {
                ensure!(rest.len() == 4, "truncated Heartbeat packet");
                Ok(Self::Heartbeat { nonce: u32::from_le_bytes(rest.try_into().unwrap()) })
            }
            TAG_ACK => {
                ensure!(rest.len() == 4, "truncated Ack packet");
                Ok(Self::Ack { nonce: u32::from_le_bytes(rest.try_into().unwrap()) })
            }
            TAG_LEAVE => {
                ensure!(rest.len() == 1, "truncated Leave packet");
                Ok(Self::Leave { player: rest[0] })
            }
            TAG_LIST_ROOMS => {
                ensure!(rest.is_empty(), "trailing bytes in ListRooms");
                Ok(Self::ListRooms)
            }
            TAG_ROOMS => {
                ensure!(rest.len() >= 2, "truncated Rooms packet");
                let count = usize::from(u16::from_le_bytes([rest[0], rest[1]]));
                let mut pos = 2;
                let mut rooms = Vec::with_capacity(count.min(256));
                for _ in 0..count {
                    ensure!(pos + 3 <= rest.len(), "truncated RoomSummary");
                    let occupied = rest[pos];
                    let capacity = rest[pos + 1];
                    let rlen = usize::from(rest[pos + 2]);
                    pos += 3;
                    ensure!(pos + rlen <= rest.len(), "truncated RoomSummary name");
                    let room = std::str::from_utf8(&rest[pos..pos + rlen])?.to_owned();
                    pos += rlen;
                    rooms.push(RoomSummary { room, occupied, capacity });
                }
                ensure!(pos == rest.len(), "trailing bytes in Rooms");
                Ok(Self::Rooms(rooms))
            }
            other => bail!("unknown relay packet tag {other}"),
        }
    }

    /// Encodes this packet with a 2-byte little-endian length prefix for byte streams (TCP/WS).
    pub fn encode_framed(&self) -> Result<Vec<u8>> {
        let body = self.encode()?;
        let len = u16::try_from(body.len()).map_err(|_| anyhow::anyhow!("relay frame exceeds u16"))?;
        let mut framed = Vec::with_capacity(2 + body.len());
        framed.extend_from_slice(&len.to_le_bytes());
        framed.extend_from_slice(&body);
        Ok(framed)
    }
}

/// Incremental length-prefixed frame decoder for byte-stream transports (TCP).
#[derive(Default, Debug)]
pub struct FrameDecoder {
    buffer: Vec<u8>,
    read_pos: usize,
}

impl FrameDecoder {
    pub fn push(&mut self, chunk: &[u8]) {
        if self.read_pos == self.buffer.len() {
            self.buffer.clear();
            self.read_pos = 0;
        }
        self.buffer.extend_from_slice(chunk);
    }

    pub fn next_frame(&mut self) -> Result<Option<Vec<u8>>> {
        let available = self.buffer.len() - self.read_pos;
        if available < 2 {
            return Ok(None);
        }
        let len = usize::from(u16::from_le_bytes([self.buffer[self.read_pos], self.buffer[self.read_pos + 1]]));
        ensure!(len <= MAX_FRAME_BYTES, "incoming relay frame too large ({len} bytes)");
        if available < 2 + len {
            return Ok(None);
        }
        let start = self.read_pos + 2;
        let payload = self.buffer[start..start + len].to_vec();
        self.read_pos = start + len;
        if self.read_pos == self.buffer.len() {
            self.buffer.clear();
            self.read_pos = 0;
        } else if self.read_pos >= 8192 {
            self.buffer.drain(..self.read_pos);
            self.read_pos = 0;
        }
        Ok(Some(payload))
    }
}

/// Any bidirectional framed message channel: TCP relay client, WebSocket, or WebRTC DataChannel.
pub trait Transport {
    fn send_frame(&mut self, bytes: &[u8]);
    fn recv_frame(&mut self) -> Option<Vec<u8>>;
}

/// A [`Link`] adapter that runs [`super::Netplay`] over any [`Transport`] (WebRTC DataChannel,
/// WebSocket, TCP relay stream, or in-memory broker channel).
pub struct ChannelLink<T: Transport> {
    transport: T,
    room: String,
    player: u8,
    players: u8,
    token: u64,
    joined: bool,
    last_ack: Option<u32>,
    inbox: VecDeque<(usize, Vec<u8>)>,
    tx_buf: Vec<u8>,
}

impl<T: Transport> ChannelLink<T> {
    /// Creates the link and immediately sends a [`RelayPacket::Join`] registration frame.
    pub fn new(mut transport: T, room: impl Into<String>, player: usize, players: usize, token: u64) -> Result<Self> {
        let room = room.into();
        let player_u8 = u8::try_from(player).map_err(|_| anyhow::anyhow!("player index exceeds u8"))?;
        let players_u8 = u8::try_from(players).map_err(|_| anyhow::anyhow!("player count exceeds u8"))?;
        let join = RelayPacket::Join { room: room.clone(), player: player_u8, players: players_u8, token }.encode()?;
        transport.send_frame(&join);
        Ok(Self {
            transport,
            room,
            player: player_u8,
            players: players_u8,
            token,
            joined: false,
            last_ack: None,
            inbox: VecDeque::new(),
            tx_buf: Vec::with_capacity(super::wire::DATAGRAM_BYTES + 3),
        })
    }

    pub fn joined(&self) -> bool {
        self.joined
    }

    pub fn room(&self) -> &str {
        &self.room
    }

    pub fn token(&self) -> u64 {
        self.token
    }

    pub fn last_ack(&self) -> Option<u32> {
        self.last_ack
    }

    /// Sends a keepalive / RTT probe over the underlying transport.
    pub fn send_heartbeat(&mut self, nonce: u32) {
        if let Ok(bytes) = (RelayPacket::Heartbeat { nonce }).encode() {
            self.transport.send_frame(&bytes);
        }
    }

    /// Gracefully leaves the relay room.
    pub fn leave(&mut self) {
        if let Ok(bytes) = (RelayPacket::Leave { player: self.player }).encode() {
            self.transport.send_frame(&bytes);
        }
    }

    fn drain_transport(&mut self) {
        while let Some(raw) = self.transport.recv_frame() {
            let Ok(packet) = RelayPacket::decode(&raw) else { continue };
            match packet {
                RelayPacket::Joined { player, players } if player == self.player && players == self.players => {
                    self.joined = true;
                }
                RelayPacket::Forward { from, to, payload }
                    if to == self.player && usize::from(from) < usize::from(self.players) =>
                {
                    self.inbox.push_back((usize::from(from), payload));
                }
                RelayPacket::Heartbeat { nonce } => {
                    if let Ok(ack) = (RelayPacket::Ack { nonce }).encode() {
                        self.transport.send_frame(&ack);
                    }
                }
                RelayPacket::Ack { nonce } => {
                    self.last_ack = Some(nonce);
                }
                _ => {}
            }
        }
    }
}

impl<T: Transport> Link for ChannelLink<T> {
    fn send(&mut self, player: usize, bytes: &[u8]) {
        let Ok(to) = u8::try_from(player) else { return };
        if bytes.len() + 3 > MAX_FRAME_BYTES {
            return;
        }
        self.tx_buf.clear();
        self.tx_buf.reserve(3 + bytes.len());
        self.tx_buf.push(TAG_FORWARD);
        self.tx_buf.push(self.player);
        self.tx_buf.push(to);
        self.tx_buf.extend_from_slice(bytes);
        self.transport.send_frame(&self.tx_buf);
    }

    fn receive(&mut self) -> Option<(usize, Vec<u8>)> {
        self.drain_transport();
        self.inbox.pop_front()
    }
}

/// Identifies a connected client slot inside [`RelayBroker`].
pub type ClientId = u64;

struct Room {
    players: u8,
    token: u64,
    members: HashMap<u8, ClientId>,
}

/// Multi-room packet router used by both the native TCP relay server and
/// WebRTC/WebSocket signaling bridges.
#[derive(Default)]
pub struct RelayBroker {
    rooms: HashMap<String, Room>,
    membership: HashMap<ClientId, (String, u8)>,
}

impl RelayBroker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Processes one decoded [`RelayPacket`] from `client` and returns `(recipient, packet)` frames to send.
    pub fn handle(&mut self, client: ClientId, packet: RelayPacket) -> Vec<(ClientId, RelayPacket)> {
        match packet {
            RelayPacket::Join { room, player, players, token } => {
                if !(2..=32).contains(&players) || player >= players || room.is_empty() || room.len() > 255 {
                    return Vec::new();
                }
                // Validate target room before modifying existing membership so joins are atomic.
                if let Some(existing) = self.rooms.get(&room) {
                    if existing.players != players || existing.token != token {
                        return Vec::new();
                    }
                    // Reject hijacking an active slot occupied by a different client.
                    if let Some(&occupant) = existing.members.get(&player)
                        && occupant != client
                    {
                        return Vec::new();
                    }
                }
                // Remove any prior room/slot membership for this client atomically.
                self.disconnect(client);
                let entry =
                    self.rooms.entry(room.clone()).or_insert_with(|| Room { players, token, members: HashMap::new() });
                entry.members.insert(player, client);
                self.membership.insert(client, (room, player));
                vec![(client, RelayPacket::Joined { player, players })]
            }
            RelayPacket::Forward { to, payload, .. } => {
                let Some((room_name, from)) = self.membership.get(&client).cloned() else {
                    return Vec::new();
                };
                let Some(room) = self.rooms.get(&room_name) else { return Vec::new() };
                let Some(&target) = room.members.get(&to) else { return Vec::new() };
                vec![(target, RelayPacket::Forward { from, to, payload })]
            }
            RelayPacket::Heartbeat { nonce } => vec![(client, RelayPacket::Ack { nonce })],
            RelayPacket::Leave { .. } => {
                self.disconnect(client);
                Vec::new()
            }
            RelayPacket::ListRooms => vec![(client, RelayPacket::Rooms(self.list_rooms_within_frame()))],
            RelayPacket::Joined { .. } | RelayPacket::Ack { .. } | RelayPacket::Rooms(_) => Vec::new(),
        }
    }

    /// Returns a deterministically sorted list of active rooms for lobby discovery.
    pub fn list_rooms(&self) -> Vec<RoomSummary> {
        let mut out: Vec<RoomSummary> = self
            .rooms
            .iter()
            .map(|(name, room)| RoomSummary {
                room: name.clone(),
                occupied: u8::try_from(room.members.len()).unwrap_or(u8::MAX),
                capacity: room.players,
            })
            .collect();
        crate::sort::shell_sort_by(&mut out, |a, b| a.room.cmp(&b.room));
        out
    }

    /// Returns a deterministically sorted page of active rooms.
    pub fn list_rooms_page(&self, offset: usize, limit: usize) -> Vec<RoomSummary> {
        self.list_rooms().into_iter().skip(offset).take(limit).collect()
    }

    /// Returns as many deterministically sorted active rooms as fit inside one
    /// [`MAX_FRAME_BYTES`] [`RelayPacket::Rooms`] wire frame (never fails `encode()`).
    pub fn list_rooms_within_frame(&self) -> Vec<RoomSummary> {
        let mut out = Vec::new();
        let mut used = 3usize; // TAG_ROOMS (1) + count (2)
        for summary in self.list_rooms() {
            let entry_bytes = 3 + summary.room.len();
            if used + entry_bytes > MAX_FRAME_BYTES || out.len() >= usize::from(u16::MAX) {
                break;
            }
            used += entry_bytes;
            out.push(summary);
        }
        out
    }

    /// Removes `client` from its room (cleaning up empty rooms).
    pub fn disconnect(&mut self, client: ClientId) {
        if let Some((room_name, player)) = self.membership.remove(&client)
            && let Some(room) = self.rooms.get_mut(&room_name)
        {
            if room.members.get(&player) == Some(&client) {
                room.members.remove(&player);
            }
            if room.members.is_empty() {
                self.rooms.remove(&room_name);
            }
        }
    }

    pub fn room_count(&self) -> usize {
        self.rooms.len()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;

    /// Shared in-memory broker wiring multiple [`ChannelLink`]s (models WebRTC DataChannel
    /// or relay connections deterministically inside unit tests).
    #[derive(Default)]
    struct Hub {
        broker: RelayBroker,
        inboxes: HashMap<ClientId, VecDeque<Vec<u8>>>,
    }

    impl Hub {
        fn send_from(&mut self, client: ClientId, raw: &[u8]) {
            let Ok(packet) = RelayPacket::decode(raw) else { return };
            for (dst, out) in self.broker.handle(client, packet) {
                if let Ok(bytes) = out.encode() {
                    self.inboxes.entry(dst).or_default().push_back(bytes);
                }
            }
        }
        fn recv_for(&mut self, client: ClientId) -> Option<Vec<u8>> {
            self.inboxes.get_mut(&client)?.pop_front()
        }
    }

    struct HubTransport {
        id: ClientId,
        hub: Rc<RefCell<Hub>>,
    }

    impl Transport for HubTransport {
        fn send_frame(&mut self, bytes: &[u8]) {
            self.hub.borrow_mut().send_from(self.id, bytes);
        }
        fn recv_frame(&mut self) -> Option<Vec<u8>> {
            self.hub.borrow_mut().recv_for(self.id)
        }
    }

    #[test]
    fn packet_and_stream_framing_round_trip() {
        let cases = [
            RelayPacket::Join { room: "arena-42".into(), player: 1, players: 4, token: 0xDEAD_BEEF_1234 },
            RelayPacket::Joined { player: 1, players: 4 },
            RelayPacket::Forward { from: 0, to: 1, payload: vec![1, 2, 3, 250, 0, 9] },
            RelayPacket::Heartbeat { nonce: 77 },
            RelayPacket::Ack { nonce: 77 },
            RelayPacket::Leave { player: 1 },
        ];
        let mut decoder = FrameDecoder::default();
        let mut stream = Vec::new();
        for pkt in &cases {
            let decoded = RelayPacket::decode(&pkt.encode().unwrap()).unwrap();
            assert_eq!(&decoded, pkt);
            stream.extend_from_slice(&pkt.encode_framed().unwrap());
        }
        // Feed odd 3-byte chunks to verify arbitrary TCP segmentation.
        let mut recovered = Vec::new();
        for chunk in stream.chunks(3) {
            decoder.push(chunk);
            while let Some(frame) = decoder.next_frame().unwrap() {
                recovered.push(RelayPacket::decode(&frame).unwrap());
            }
        }
        assert_eq!(recovered, cases);
    }

    #[test]
    fn broker_isolates_rooms_validates_tokens_and_forwards_link_datagrams() {
        let hub = Rc::new(RefCell::new(Hub::default()));
        let mut p0 = ChannelLink::new(HubTransport { id: 10, hub: hub.clone() }, "match-1", 0, 2, 999).unwrap();
        let mut p1 = ChannelLink::new(HubTransport { id: 20, hub: hub.clone() }, "match-1", 1, 2, 999).unwrap();
        let mut intruder = ChannelLink::new(HubTransport { id: 30, hub: hub.clone() }, "match-1", 1, 2, 111).unwrap();

        assert!(p0.receive().is_none() && p0.joined());
        assert!(p1.receive().is_none() && p1.joined());
        assert!(intruder.receive().is_none() && !intruder.joined(), "wrong room token must be rejected");

        p0.send(1, b"hello-from-0");
        p1.send(0, b"reply-from-1");
        assert_eq!(p1.receive(), Some((0, b"hello-from-0".to_vec())));
        assert_eq!(p0.receive(), Some((1, b"reply-from-1".to_vec())));

        p0.send_heartbeat(42);
        assert!(p0.receive().is_none());
        assert_eq!(p0.last_ack(), Some(42));

        p0.leave();
        p1.leave();
        assert_eq!(hub.borrow().broker.room_count(), 0);
    }

    #[test]
    fn relay_broker_joins_atomically_rejects_slot_hijack_and_bounds_room_list() {
        let mut broker = RelayBroker::new();
        assert!(
            !broker.handle(1, RelayPacket::Join { room: "alpha".into(), player: 0, players: 2, token: 5 }).is_empty()
        );
        assert!(
            broker.handle(2, RelayPacket::Join { room: "alpha".into(), player: 0, players: 2, token: 5 }).is_empty()
        );
        assert!(
            !broker.handle(1, RelayPacket::Join { room: "beta".into(), player: 1, players: 2, token: 9 }).is_empty()
        );
        assert_eq!(broker.room_count(), 1);
        assert_eq!(broker.list_rooms()[0].room, "beta");
        for i in 0..400u32 {
            let name = format!("room-{i:04}-{}", "x".repeat(200));
            broker.handle(100 + u64::from(i), RelayPacket::Join { room: name, player: 0, players: 2, token: 1 });
        }
        let replies = broker.handle(999, RelayPacket::ListRooms);
        assert_eq!(replies.len(), 1);
        let encoded = replies[0].1.encode().unwrap();
        assert!(encoded.len() <= MAX_FRAME_BYTES);
    }
}
