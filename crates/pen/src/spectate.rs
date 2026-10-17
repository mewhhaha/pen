//! Spectating natively: a session streams its final frames to viewers over
//! WebSockets (`--broadcast ADDRESS`), and a viewer plays a stream as it comes
//! (`--spectate ADDRESS`), as a replay plays a recording
//! (`pen_runtime::net::stream`, docs/netplay.md). A stream carries every
//! player's input, keys typed included: serve it only where that may go. On
//! a loopback address only native viewers and pages from the loopback may
//! watch.
use std::net::{SocketAddr, TcpListener, ToSocketAddrs};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, channel, sync_channel};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use pen_runtime::net::stream::{Broadcast, Message};

use crate::timeline::Timeline;

pub mod websocket;

/// Messages held for a viewer that has not taken them yet; one that falls
/// further behind is let go.
const VIEWER_QUEUE: usize = 4096;

/// Whether a page at `origin` may watch a stream on `address`: any page
/// when the stream is served beyond the loopback, else native viewers and
/// pages from the loopback.
fn allowed(address: SocketAddr, origin: Option<&str>) -> bool {
    let Some(origin) = origin else { return true };
    if !address.ip().is_loopback() {
        return true;
    }
    let host = origin.split_once("://").map_or(origin, |(_, rest)| rest);
    let name = host.rsplit_once(':').map_or(host, |(name, port)| if port.parse::<u16>().is_ok() { name } else { host });
    ["127.0.0.1", "localhost", "[::1]"].contains(&name)
}

/// A session's stream to its viewers.
pub struct Stream {
    broadcast: Broadcast,
    address: SocketAddr,
    /// Viewers whose handshake is done, waiting to be started.
    joining: Receiver<SyncSender<Arc<Vec<u8>>>>,
    viewers: Vec<SyncSender<Arc<Vec<u8>>>>,
}

impl Stream {
    /// Listens on `address` for viewers, on a thread of its own; each
    /// viewer's messages are written by a thread of its own too.
    pub fn serve(address: &str) -> Result<Stream> {
        let listener = TcpListener::bind(address).with_context(|| format!("streaming on {address}"))?;
        let address = listener.local_addr()?;
        let (join, joining) = channel();
        std::thread::Builder::new().name("pen-stream".into()).spawn(move || {
            for connection in listener.incoming() {
                let Ok(connection) = connection else { continue };
                let join = join.clone();
                // The handshake and then the viewer's messages, on its own thread.
                let _ = std::thread::Builder::new().name("pen-viewer".into()).spawn(move || {
                    let _ = connection.set_nodelay(true);
                    let _ = connection.set_read_timeout(Some(Duration::from_secs(10)));
                    let Ok(mut socket) = websocket::accept(connection, |origin| allowed(address, origin)) else {
                        return;
                    };
                    let (sender, messages) = sync_channel::<Arc<Vec<u8>>>(VIEWER_QUEUE);
                    if join.send(sender).is_err() {
                        return;
                    }
                    for message in messages {
                        if socket.send(&message).is_err() {
                            return;
                        }
                    }
                    socket.close();
                });
            }
        })?;
        Ok(Stream { broadcast: Broadcast::default(), address, joining, viewers: Vec::new() })
    }

    /// Where viewers connect.
    pub fn address(&self) -> SocketAddr {
        self.address
    }

    fn send(viewers: &mut Vec<SyncSender<Arc<Vec<u8>>>>, message: &Message) {
        let bytes = Arc::new(message.encode());
        viewers.retain(|viewer| match viewer.try_send(bytes.clone()) {
            Ok(()) => true,
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => false,
        });
    }

    /// Frames before `end` of `timeline` are final: sends the ones not
    /// sent yet to every viewer, and starts the viewers that joined.
    /// Returns how many joined.
    pub fn poll(&mut self, timeline: &Timeline, end: u64) -> usize {
        for message in self.broadcast.advance(timeline, end) {
            Stream::send(&mut self.viewers, &message);
        }
        let mut joined = 0;
        while let Ok(viewer) = self.joining.try_recv() {
            let mut started = vec![viewer];
            for message in self.broadcast.join(timeline, end).unwrap_or_default() {
                Stream::send(&mut started, &message);
            }
            joined += started.len();
            self.viewers.extend(started);
        }
        joined
    }

    /// The session ended: so does every viewer's stream.
    pub fn end(&mut self) {
        Stream::send(&mut self.viewers, &Message::End);
        self.viewers.clear();
    }
}

/// A stream being watched: its messages as they arrive, read on a thread
/// of their own.
pub struct Watch {
    messages: Receiver<Result<Message>>,
    ended: bool,
}

impl Watch {
    /// Watches the stream at `address` (`HOST:PORT`, or a `ws://` URL).
    pub fn connect(address: &str) -> Result<Watch> {
        let rest = address.strip_prefix("ws://").unwrap_or(address);
        let (host, path) =
            rest.split_once('/').map_or((rest, "/".to_owned()), |(host, path)| (host, format!("/{path}")));
        host.to_socket_addrs().with_context(|| format!("{host} names no address"))?;
        let mut socket = websocket::client(host, &path)?;
        let (sender, messages) = channel();
        std::thread::Builder::new().name("pen-watch".into()).spawn(move || {
            loop {
                let message = match socket.receive() {
                    Ok(websocket::Incoming::Message(bytes)) => Message::decode(&bytes),
                    Ok(websocket::Incoming::Closed) => Err(anyhow!("the stream closed before it ended")),
                    Err(error) => Err(error.context("reading the stream")),
                };
                let last = !matches!(message, Ok(Message::Start { .. } | Message::Frames { .. }));
                if sender.send(message).is_err() || last {
                    return;
                }
            }
        })?;
        Ok(Watch { messages, ended: false })
    }

    /// The next message, waiting at most `timeout`; None when none came
    /// or the stream has ended.
    pub fn next(&mut self, timeout: Duration) -> Result<Option<Message>> {
        if self.ended {
            return Ok(None);
        }
        let message = match self.messages.recv_timeout(timeout) {
            Ok(message) => message?,
            Err(RecvTimeoutError::Timeout) => return Ok(None),
            Err(RecvTimeoutError::Disconnected) => Message::End,
        };
        self.ended = message == Message::End;
        Ok(Some(message))
    }

    /// Whether the stream has ended: no more frames will come.
    pub fn ended(&self) -> bool {
        self.ended
    }
}

#[cfg(test)]
mod tests;
