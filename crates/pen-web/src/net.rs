//! Browser netplay transports: [`WebSocketTransport`] and
//! [`WebRtcDataChannelTransport`] implementing `pen_runtime::net::relay::Transport`
//! with bounded queues and explicit closed-state handling so browser sessions can
//! join `pen relay` rooms and WebRTC peer meshes with `pen_runtime::net::Netplay`.
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use anyhow::{Result, anyhow};
use js_sys::{ArrayBuffer, Uint8Array};
use pen_runtime::net::Link;
use pen_runtime::net::relay::{ChannelLink, MAX_FRAME_BYTES, Transport};
use wasm_bindgen::{JsCast, closure::Closure};
use web_sys::{BinaryType, MessageEvent, RtcDataChannel, RtcDataChannelState, RtcDataChannelType, WebSocket};

const MAX_QUEUE_FRAMES: usize = 256;
const MAX_QUEUE_BYTES: usize = 1024 * 1024;

#[derive(Default)]
struct BoundedQueue {
    frames: VecDeque<Vec<u8>>,
    bytes: usize,
    dropped: usize,
}

impl BoundedQueue {
    fn push_back(&mut self, frame: Vec<u8>) -> bool {
        if frame.len() > MAX_FRAME_BYTES
            || self.frames.len() >= MAX_QUEUE_FRAMES
            || self.bytes.saturating_add(frame.len()) > MAX_QUEUE_BYTES
        {
            self.dropped += 1;
            return false;
        }
        self.bytes += frame.len();
        self.frames.push_back(frame);
        true
    }

    fn push_front(&mut self, frame: Vec<u8>) {
        self.bytes += frame.len();
        self.frames.push_front(frame);
    }

    fn pop_front(&mut self) -> Option<Vec<u8>> {
        let frame = self.frames.pop_front()?;
        self.bytes = self.bytes.saturating_sub(frame.len());
        Some(frame)
    }

    fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

/// Non-blocking binary WebSocket transport backed by `web_sys::WebSocket`.
pub struct WebSocketTransport {
    socket: WebSocket,
    inbox: Rc<RefCell<BoundedQueue>>,
    outbox: BoundedQueue,
    _on_message: Closure<dyn FnMut(MessageEvent)>,
}

impl WebSocketTransport {
    /// Connects to a WebSocket relay URL (e.g. `ws://host:port` or `wss://...`).
    pub fn connect(url: &str) -> Result<Self> {
        let socket = WebSocket::new(url).map_err(|err| anyhow!("WebSocket::new({url}): {err:?}"))?;
        socket.set_binary_type(BinaryType::Arraybuffer);
        let inbox = Rc::new(RefCell::new(BoundedQueue::default()));
        let inbox_ref = Rc::clone(&inbox);
        let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
            if let Ok(buffer) = event.data().dyn_into::<ArrayBuffer>() {
                let len = buffer.byte_length() as usize;
                if len <= MAX_FRAME_BYTES {
                    inbox_ref.borrow_mut().push_back(Uint8Array::new(&buffer).to_vec());
                }
            }
        });
        socket.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        Ok(Self { socket, inbox, outbox: BoundedQueue::default(), _on_message: on_message })
    }

    pub fn is_closed(&self) -> bool {
        matches!(self.socket.ready_state(), WebSocket::CLOSING | WebSocket::CLOSED)
    }

    pub fn dropped_frames(&self) -> usize {
        self.outbox.dropped + self.inbox.borrow().dropped
    }

    fn flush_outbox(&mut self) {
        if self.socket.ready_state() != WebSocket::OPEN {
            if self.is_closed() {
                self.outbox.frames.clear();
                self.outbox.bytes = 0;
            }
            return;
        }
        while let Some(frame) = self.outbox.pop_front() {
            if self.socket.send_with_u8_array(&frame).is_err() {
                if !self.is_closed() {
                    self.outbox.push_front(frame);
                }
                break;
            }
        }
    }
}

impl Transport for WebSocketTransport {
    fn send_frame(&mut self, bytes: &[u8]) {
        if self.is_closed() || bytes.len() > MAX_FRAME_BYTES {
            self.outbox.dropped += 1;
            return;
        }
        if self.socket.ready_state() == WebSocket::OPEN && self.outbox.is_empty() {
            if self.socket.send_with_u8_array(bytes).is_err() && !self.is_closed() {
                self.outbox.push_back(bytes.to_vec());
            }
        } else {
            self.outbox.push_back(bytes.to_vec());
            self.flush_outbox();
        }
    }

    fn recv_frame(&mut self) -> Option<Vec<u8>> {
        self.flush_outbox();
        self.inbox.borrow_mut().pop_front()
    }
}

impl Drop for WebSocketTransport {
    fn drop(&mut self) {
        self.socket.set_onmessage(None);
        let _ = self.socket.close();
    }
}

/// Non-blocking binary WebRTC DataChannel transport backed by `web_sys::RtcDataChannel`.
pub struct WebRtcDataChannelTransport {
    channel: RtcDataChannel,
    inbox: Rc<RefCell<BoundedQueue>>,
    outbox: BoundedQueue,
    _on_message: Closure<dyn FnMut(MessageEvent)>,
}

impl WebRtcDataChannelTransport {
    /// Wraps an existing `RtcDataChannel` in binary arraybuffer mode.
    pub fn new(channel: RtcDataChannel) -> Self {
        channel.set_binary_type(RtcDataChannelType::Arraybuffer);
        let inbox = Rc::new(RefCell::new(BoundedQueue::default()));
        let inbox_ref = Rc::clone(&inbox);
        let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
            if let Ok(buffer) = event.data().dyn_into::<ArrayBuffer>() {
                let len = buffer.byte_length() as usize;
                if len <= MAX_FRAME_BYTES {
                    inbox_ref.borrow_mut().push_back(Uint8Array::new(&buffer).to_vec());
                }
            }
        });
        channel.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        Self { channel, inbox, outbox: BoundedQueue::default(), _on_message: on_message }
    }

    pub fn is_closed(&self) -> bool {
        matches!(self.channel.ready_state(), RtcDataChannelState::Closing | RtcDataChannelState::Closed)
    }

    pub fn dropped_frames(&self) -> usize {
        self.outbox.dropped + self.inbox.borrow().dropped
    }

    fn flush_outbox(&mut self) {
        if self.channel.ready_state() != RtcDataChannelState::Open {
            if self.is_closed() {
                self.outbox.frames.clear();
                self.outbox.bytes = 0;
            }
            return;
        }
        while let Some(frame) = self.outbox.pop_front() {
            if self.channel.send_with_u8_array(&frame).is_err() {
                if !self.is_closed() {
                    self.outbox.push_front(frame);
                }
                break;
            }
        }
    }
}

impl Transport for WebRtcDataChannelTransport {
    fn send_frame(&mut self, bytes: &[u8]) {
        if self.is_closed() || bytes.len() > MAX_FRAME_BYTES {
            self.outbox.dropped += 1;
            return;
        }
        if self.channel.ready_state() == RtcDataChannelState::Open && self.outbox.is_empty() {
            if self.channel.send_with_u8_array(bytes).is_err() && !self.is_closed() {
                self.outbox.push_back(bytes.to_vec());
            }
        } else {
            self.outbox.push_back(bytes.to_vec());
            self.flush_outbox();
        }
    }

    fn recv_frame(&mut self) -> Option<Vec<u8>> {
        self.flush_outbox();
        self.inbox.borrow_mut().pop_front()
    }
}

impl Drop for WebRtcDataChannelTransport {
    fn drop(&mut self) {
        self.channel.set_onmessage(None);
    }
}

/// Ready-to-use browser relay [`Link`] backed by a WebSocket connection.
pub struct BrowserRelayLink(ChannelLink<WebSocketTransport>);

impl BrowserRelayLink {
    pub fn connect(url: &str, room: &str, player: usize, players: usize, token: u64) -> Result<Self> {
        let transport = WebSocketTransport::connect(url)?;
        Ok(Self(ChannelLink::new(transport, room, player, players, token)?))
    }
}

impl Link for BrowserRelayLink {
    fn send(&mut self, player: usize, bytes: &[u8]) {
        self.0.send(player, bytes);
    }

    fn receive(&mut self) -> Option<(usize, Vec<u8>)> {
        self.0.receive()
    }
}
