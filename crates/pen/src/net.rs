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
/// Unambiguous 4-byte native TCP relay stream preface (distinct from HTTP `GET `).
pub const TCP_PREFACE: &[u8; 4] = b"PENR";
const MAX_RELAY_CLIENTS: usize = 1024;
const MAX_HTTP_HEADER_BYTES: usize = 8192;
const MAX_OUTBOX_BYTES: usize = 262_144;
const MAX_READ_BYTES_PER_POLL: usize = 65_536;
const MAX_FRAMES_PER_POLL: usize = 64;
const DETECT_TIMEOUT_SECS: u64 = 5;
const IDLE_TIMEOUT_SECS: u64 = 60;

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
    write_buf: Vec<u8>,
    write_pos: usize,
    closed: bool,
}

impl TcpTransport {
    pub fn connect(addr: SocketAddr) -> Result<Self> {
        let stream = TcpStream::connect(addr).with_context(|| format!("connecting relay {addr}"))?;
        let _ = stream.set_nodelay(true);
        stream.set_nonblocking(true)?;
        let mut transport = Self {
            stream,
            decoder: FrameDecoder::default(),
            read_buf: vec![0u8; 8192],
            write_buf: TCP_PREFACE.to_vec(),
            write_pos: 0,
            closed: false,
        };
        transport.flush_writes();
        Ok(transport)
    }

    #[allow(dead_code)]
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    fn flush_writes(&mut self) {
        if self.closed {
            return;
        }
        while self.write_pos < self.write_buf.len() {
            match self.stream.write(&self.write_buf[self.write_pos..]) {
                Ok(0) => {
                    self.closed = true;
                    break;
                }
                Ok(n) => self.write_pos += n,
                Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                Err(_) => {
                    self.closed = true;
                    break;
                }
            }
        }
        if self.write_pos == self.write_buf.len() {
            self.write_buf.clear();
            self.write_pos = 0;
        } else if self.write_pos >= 8192 {
            self.write_buf.drain(..self.write_pos);
            self.write_pos = 0;
        }
    }
}

impl Transport for TcpTransport {
    fn send_frame(&mut self, bytes: &[u8]) {
        if self.closed {
            return;
        }
        let Ok(len) = u16::try_from(bytes.len()) else { return };
        if usize::from(len) > pen_runtime::net::relay::MAX_FRAME_BYTES {
            return;
        }
        let pending = self.write_buf.len().saturating_sub(self.write_pos);
        if pending + 2 + bytes.len() > MAX_OUTBOX_BYTES {
            return;
        }
        self.write_buf.extend_from_slice(&len.to_le_bytes());
        self.write_buf.extend_from_slice(bytes);
        self.flush_writes();
    }

    fn recv_frame(&mut self) -> Option<Vec<u8>> {
        self.flush_writes();
        if self.closed {
            return None;
        }
        let mut budget = MAX_READ_BYTES_PER_POLL;
        while budget > 0 {
            let cap = self.read_buf.len().min(budget);
            match self.stream.read(&mut self.read_buf[..cap]) {
                Ok(0) => {
                    self.closed = true;
                    break;
                }
                Ok(n) => {
                    budget -= n;
                    self.decoder.push(&self.read_buf[..n]);
                }
                Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                Err(_) => {
                    self.closed = true;
                    break;
                }
            }
        }
        match self.decoder.next_frame() {
            Ok(frame) => frame,
            Err(_) => {
                self.closed = true;
                None
            }
        }
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

#[derive(PartialEq, Eq)]
enum ClientMode {
    Detecting,
    Tcp,
    WebSocket,
}

struct RelayClient {
    stream: TcpStream,
    mode: ClientMode,
    raw_buf: Vec<u8>,
    decoder: FrameDecoder,
    write_buf: Vec<u8>,
    write_pos: usize,
    frag_buf: Vec<u8>,
    frag_active: bool,
    closing: bool,
    connected_at: Instant,
    last_activity: Instant,
}

impl RelayClient {
    fn queue_bytes(&mut self, bytes: &[u8]) -> bool {
        let pending = self.write_buf.len().saturating_sub(self.write_pos);
        if pending.saturating_add(bytes.len()) > MAX_OUTBOX_BYTES {
            return false;
        }
        self.write_buf.extend_from_slice(bytes);
        true
    }

    /// Flushes pending outbound bytes; returns `false` if the socket errored or closed.
    fn flush(&mut self) -> bool {
        while self.write_pos < self.write_buf.len() {
            match self.stream.write(&self.write_buf[self.write_pos..]) {
                Ok(0) => return false,
                Ok(n) => self.write_pos += n,
                Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                Err(_) => return false,
            }
        }
        if self.write_pos == self.write_buf.len() {
            self.write_buf.clear();
            self.write_pos = 0;
        } else if self.write_pos >= 8192 {
            self.write_buf.drain(..self.write_pos);
            self.write_pos = 0;
        }
        true
    }
}

fn encode_ws_frame(opcode: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 10);
    out.push(0x80 | (opcode & 0x0f));
    if payload.len() < 126 {
        #[allow(clippy::cast_possible_truncation, reason = "checked < 126")]
        out.push(payload.len() as u8);
    } else if payload.len() <= usize::from(u16::MAX) {
        out.push(126);
        #[allow(clippy::cast_possible_truncation, reason = "checked <= u16::MAX")]
        out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    } else {
        out.push(127);
        out.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    }
    out.extend_from_slice(payload);
    out
}

fn encode_ws_binary(payload: &[u8]) -> Vec<u8> {
    encode_ws_frame(2, payload)
}

#[derive(Debug)]
struct WsFrame {
    fin: bool,
    opcode: u8,
    payload: Vec<u8>,
}

enum WsParse {
    Incomplete,
    Frame(WsFrame),
    ProtocolError,
}

/// Validates and decodes one RFC 6455 client-to-server WebSocket frame from `buf`,
/// enforcing RSV == 0, client masking, canonical payload lengths, control-frame rules,
/// and `max_payload` before buffering any payload bytes.
fn pop_ws_frame(buf: &mut Vec<u8>, max_payload: usize) -> WsParse {
    if buf.len() < 2 {
        return WsParse::Incomplete;
    }
    let fin = (buf[0] & 0x80) != 0;
    let rsv = buf[0] & 0x70;
    if rsv != 0 {
        return WsParse::ProtocolError;
    }
    let opcode = buf[0] & 0x0f;
    let masked = (buf[1] & 0x80) != 0;
    // RFC 6455 §5.1: all client-to-server frames MUST be masked.
    if !masked {
        return WsParse::ProtocolError;
    }
    let is_control = (opcode & 0x08) != 0;
    if is_control && !fin {
        return WsParse::ProtocolError;
    }
    let mut pos = 2usize;
    let len = match buf[1] & 0x7f {
        126 => {
            if buf.len() < pos + 2 {
                return WsParse::Incomplete;
            }
            let n = usize::from(u16::from_be_bytes([buf[pos], buf[pos + 1]]));
            if n < 126 {
                return WsParse::ProtocolError; // non-canonical 16-bit length
            }
            pos += 2;
            n
        }
        127 => {
            if buf.len() < pos + 8 {
                return WsParse::Incomplete;
            }
            let raw = u64::from_be_bytes(buf[pos..pos + 8].try_into().unwrap());
            if (raw >> 63) != 0 || raw <= u64::from(u16::MAX) {
                return WsParse::ProtocolError; // MSB set or non-canonical 64-bit length
            }
            let Ok(n) = usize::try_from(raw) else {
                return WsParse::ProtocolError;
            };
            pos += 8;
            n
        }
        small => usize::from(small),
    };
    if is_control && len > 125 {
        return WsParse::ProtocolError;
    }
    if len > max_payload {
        return WsParse::ProtocolError;
    }
    if buf.len() < pos + 4 {
        return WsParse::Incomplete;
    }
    let mask = [buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]];
    pos += 4;
    if buf.len() < pos + len {
        return WsParse::Incomplete;
    }
    let mut payload = buf[pos..pos + len].to_vec();
    for (i, b) in payload.iter_mut().enumerate() {
        *b ^= mask[i & 3];
    }
    buf.drain(..pos + len);
    WsParse::Frame(WsFrame { fin, opcode, payload })
}

/// Validates an RFC 6455 HTTP/1.1 WebSocket upgrade header block and returns `Sec-WebSocket-Key`.
fn validate_ws_upgrade(head: &str) -> Option<String> {
    let mut lines = head.split("\r\n");
    let req_line = lines.next()?;
    let mut parts = req_line.split_whitespace();
    let method = parts.next()?;
    let _target = parts.next()?;
    let version = parts.next()?;
    if method != "GET" || !version.starts_with("HTTP/1.1") || parts.next().is_some() {
        return None;
    }
    let mut has_upgrade = false;
    let mut has_connection = false;
    let mut has_version = false;
    let mut ws_key: Option<String> = None;
    for line in lines {
        let Some((k, v)) = line.split_once(':') else { continue };
        let k = k.trim();
        let v = v.trim();
        if k.eq_ignore_ascii_case("upgrade") && v.eq_ignore_ascii_case("websocket") {
            has_upgrade = true;
        } else if k.eq_ignore_ascii_case("connection")
            && v.split(',').any(|tok| tok.trim().eq_ignore_ascii_case("upgrade"))
        {
            has_connection = true;
        } else if k.eq_ignore_ascii_case("sec-websocket-version") && v == "13" {
            has_version = true;
        } else if k.eq_ignore_ascii_case("sec-websocket-key")
            && v.len() == 24
            && v.ends_with("==")
            && v[..22].bytes().all(|b| b.is_ascii_alphanumeric() || b == b'+' || b == b'/')
        {
            ws_key = Some(v.to_owned());
        }
    }
    (has_upgrade && has_connection && has_version).then_some(ws_key?)
}

/// Non-blocking multi-room TCP + RFC 6455 WebSocket relay server wrapping [`RelayBroker`].
/// Auto-detects HTTP/1.1 WebSocket upgrades and `PENR`-prefaced length-prefixed native TCP clients.
pub struct TcpRelayServer {
    listener: TcpListener,
    broker: RelayBroker,
    clients: HashMap<ClientId, RelayClient>,
    next_id: ClientId,
    read_buf: Vec<u8>,
}

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

    /// Runs the relay server loop until `stop()` returns true.
    pub fn serve(&mut self, mut stop: impl FnMut() -> bool) -> Result<()> {
        while !stop() {
            self.poll();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        Ok(())
    }

    /// Accepts pending connections (up to `MAX_RELAY_CLIENTS`), enforces timeouts &
    /// per-poll budgets, routes available packets, and flushes nonblocking outboxes.
    pub fn poll(&mut self) {
        let now = Instant::now();
        let mut accepted = 0usize;
        while accepted < 32 {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    accepted += 1;
                    if self.clients.len() >= MAX_RELAY_CLIENTS {
                        drop(stream);
                        continue;
                    }
                    let _ = stream.set_nodelay(true);
                    if stream.set_nonblocking(true).is_ok() {
                        let id = self.next_id;
                        self.next_id += 1;
                        self.clients.insert(
                            id,
                            RelayClient {
                                stream,
                                mode: ClientMode::Detecting,
                                raw_buf: Vec::new(),
                                decoder: FrameDecoder::default(),
                                write_buf: Vec::new(),
                                write_pos: 0,
                                frag_buf: Vec::new(),
                                frag_active: false,
                                closing: false,
                                connected_at: now,
                                last_activity: now,
                            },
                        );
                    }
                }
                Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }
        let ids: Vec<ClientId> = self.clients.keys().copied().collect();
        let mut outgoing: Vec<(ClientId, RelayPacket)> = Vec::new();
        let mut dropped = Vec::new();
        for id in ids {
            let Some(client) = self.clients.get_mut(&id) else { continue };
            if !client.flush() {
                dropped.push(id);
                continue;
            }
            if client.closing {
                if client.write_buf.is_empty() {
                    dropped.push(id);
                }
                continue;
            }
            let max_idle = if client.mode == ClientMode::Detecting { DETECT_TIMEOUT_SECS } else { IDLE_TIMEOUT_SECS };
            if now.duration_since(client.last_activity).as_secs() >= max_idle
                || (client.mode == ClientMode::Detecting
                    && now.duration_since(client.connected_at).as_secs() >= DETECT_TIMEOUT_SECS)
            {
                dropped.push(id);
                continue;
            }
            let mut read_budget = MAX_READ_BYTES_PER_POLL;
            let mut peer_closed = false;
            while read_budget > 0 {
                let cap = self.read_buf.len().min(read_budget);
                match client.stream.read(&mut self.read_buf[..cap]) {
                    Ok(0) => {
                        peer_closed = true;
                        break;
                    }
                    Ok(n) => {
                        read_budget -= n;
                        client.last_activity = now;
                        let chunk = &self.read_buf[..n];
                        match client.mode {
                            ClientMode::Tcp => client.decoder.push(chunk),
                            ClientMode::Detecting => {
                                if client.raw_buf.len() + chunk.len() > MAX_HTTP_HEADER_BYTES {
                                    peer_closed = true;
                                    break;
                                }
                                client.raw_buf.extend_from_slice(chunk);
                            }
                            ClientMode::WebSocket => {
                                if client.raw_buf.len() + chunk.len()
                                    > pen_runtime::net::relay::MAX_FRAME_BYTES + 32
                                {
                                    peer_closed = true;
                                    break;
                                }
                                client.raw_buf.extend_from_slice(chunk);
                            }
                        }
                    }
                    Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                    Err(_) => {
                        peer_closed = true;
                        break;
                    }
                }
            }
            if peer_closed {
                dropped.push(id);
                continue;
            }
            if client.mode == ClientMode::Detecting && !client.raw_buf.is_empty() {
                let n = client.raw_buf.len().min(4);
                if client.raw_buf[..n] == TCP_PREFACE[..n] {
                    if client.raw_buf.len() >= 4 {
                        client.decoder.push(&client.raw_buf[4..]);
                        client.raw_buf.clear();
                        client.mode = ClientMode::Tcp;
                    }
                } else if client.raw_buf[..n] == b"GET "[..n] {
                    if let Some(end) = client.raw_buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&client.raw_buf[..end]).into_owned();
                        client.raw_buf.drain(..end + 4);
                        if let Some(key) = validate_ws_upgrade(&head) {
                            let accept = crate::spectate::websocket::accept_key(&key);
                            let response = format!(
                                "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
                            );
                            if !client.queue_bytes(response.as_bytes()) || !client.flush() {
                                dropped.push(id);
                                continue;
                            }
                            client.mode = ClientMode::WebSocket;
                        } else {
                            let _ = client.queue_bytes(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n");
                            let _ = client.flush();
                            dropped.push(id);
                            continue;
                        }
                    } else if client.raw_buf.len() >= MAX_HTTP_HEADER_BYTES {
                        dropped.push(id);
                        continue;
                    }
                } else {
                    dropped.push(id);
                    continue;
                }
            }
            let mut frames_handled = 0usize;
            match client.mode {
                ClientMode::Tcp => {
                    while frames_handled < MAX_FRAMES_PER_POLL {
                        match client.decoder.next_frame() {
                            Ok(Some(raw)) => {
                                frames_handled += 1;
                                if let Ok(packet) = RelayPacket::decode(&raw) {
                                    outgoing.extend(self.broker.handle(id, packet));
                                } else {
                                    dropped.push(id);
                                    break;
                                }
                            }
                            Ok(None) => break,
                            Err(_) => {
                                dropped.push(id);
                                break;
                            }
                        }
                    }
                }
                ClientMode::WebSocket => {
                    let max_msg = pen_runtime::net::relay::MAX_FRAME_BYTES;
                    while frames_handled < MAX_FRAMES_PER_POLL {
                        match pop_ws_frame(&mut client.raw_buf, max_msg) {
                            WsParse::Incomplete => break,
                            WsParse::ProtocolError => {
                                let _ = client.queue_bytes(&encode_ws_frame(8, &1002u16.to_be_bytes()));
                                let _ = client.flush();
                                dropped.push(id);
                                break;
                            }
                            WsParse::Frame(frame) => {
                                frames_handled += 1;
                                match frame.opcode {
                                    2 => {
                                        if client.frag_active {
                                            dropped.push(id);
                                            break;
                                        }
                                        if frame.fin {
                                            if let Ok(packet) = RelayPacket::decode(&frame.payload) {
                                                outgoing.extend(self.broker.handle(id, packet));
                                            } else {
                                                dropped.push(id);
                                                break;
                                            }
                                        } else {
                                            client.frag_buf.clear();
                                            client.frag_buf.extend_from_slice(&frame.payload);
                                            client.frag_active = true;
                                        }
                                    }
                                    0 => {
                                        if !client.frag_active
                                            || client.frag_buf.len() + frame.payload.len() > max_msg
                                        {
                                            dropped.push(id);
                                            break;
                                        }
                                        client.frag_buf.extend_from_slice(&frame.payload);
                                        if frame.fin {
                                            let assembled = std::mem::take(&mut client.frag_buf);
                                            client.frag_active = false;
                                            if let Ok(packet) = RelayPacket::decode(&assembled) {
                                                outgoing.extend(self.broker.handle(id, packet));
                                            } else {
                                                dropped.push(id);
                                                break;
                                            }
                                        }
                                    }
                                    8 => {
                                        // Echo close status code if present (or 1000 Normal Closure), flush, then close.
                                        let code = if frame.payload.len() >= 2 {
                                            &frame.payload[..2]
                                        } else {
                                            &1000u16.to_be_bytes()[..]
                                        };
                                        let _ = client.queue_bytes(&encode_ws_frame(8, code));
                                        let _ = client.flush();
                                        client.closing = true;
                                        if client.write_buf.is_empty() {
                                            dropped.push(id);
                                        }
                                        break;
                                    }
                                    9 => {
                                        let pong = encode_ws_frame(10, &frame.payload);
                                        if !client.queue_bytes(&pong) || !client.flush() {
                                            dropped.push(id);
                                            break;
                                        }
                                    }
                                    10 => {} // unsolicited Pong ignored per RFC 6455
                                    _ => {
                                        // Reject text or unknown opcodes on binary relay channel.
                                        let _ = client.queue_bytes(&encode_ws_frame(8, &1003u16.to_be_bytes()));
                                        let _ = client.flush();
                                        dropped.push(id);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                ClientMode::Detecting => {}
            }
        }
        for (dst, reply) in outgoing {
            if let Some(target) = self.clients.get_mut(&dst)
                && !target.closing
            {
                let wire = match target.mode {
                    ClientMode::WebSocket => reply.encode().ok().map(|b| encode_ws_binary(&b)),
                    ClientMode::Tcp => reply.encode_framed().ok(),
                    ClientMode::Detecting => None,
                };
                if let Some(bytes) = wire
                    && (!target.queue_bytes(&bytes) || !target.flush())
                {
                    dropped.push(dst);
                }
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
