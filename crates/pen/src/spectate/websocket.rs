//! The little of WebSocket (RFC 6455) a stream needs, over a blocking
//! `TcpStream`: the opening handshake both ways, binary messages from the
//! server, and the client's close and pings. A browser page can watch a
//! stream with `new WebSocket(url)`; native pen with `client`.
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

use anyhow::{Context, Result, bail, ensure};

/// What the server appends to the client's key before hashing it.
const GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
/// The longest handshake read, and the largest message taken.
const MAX_HEADER: usize = 16 << 10;
const MAX_MESSAGE: usize = 256 << 20;

const CONTINUATION: u8 = 0;
const BINARY: u8 = 2;
const CLOSE: u8 = 8;
const PING: u8 = 9;
const PONG: u8 = 10;

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let word = (u32::from(chunk[0]) << 16)
            | (u32::from(chunk.get(1).copied().unwrap_or(0)) << 8)
            | u32::from(chunk.get(2).copied().unwrap_or(0));
        for index in 0..4 {
            if index <= chunk.len() {
                out.push(char::from(ALPHABET[usize::try_from(word >> (18 - 6 * index) & 63).unwrap_or(0)]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// The `Sec-WebSocket-Accept` answering a client's `Sec-WebSocket-Key`.
pub fn accept_key(key: &str) -> String {
    base64(&sha1_smol::Sha1::from(format!("{}{GUID}", key.trim())).digest().bytes())
}

/// A request's or response's head: its first line and its headers.
fn read_head(reader: &mut BufReader<TcpStream>) -> Result<(String, Vec<(String, String)>)> {
    let mut lines = Vec::new();
    let mut read = 0;
    loop {
        let mut line = String::new();
        let length = reader.read_line(&mut line)?;
        read += length;
        ensure!(length > 0 && read <= MAX_HEADER, "the WebSocket handshake ended early or ran long");
        let line = line.trim_end_matches(['\r', '\n']).to_owned();
        if line.is_empty() {
            break;
        }
        lines.push(line);
    }
    let first = lines.first().cloned().unwrap_or_default();
    let headers = lines
        .iter()
        .skip(1)
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_owned()))
        .collect();
    Ok((first, headers))
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers.iter().find(|(key, _)| key == name).map(|(_, value)| value.as_str())
}

/// One side of an open WebSocket.
pub struct Socket {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
    /// The client masks what it sends; the server does not.
    client: bool,
}

/// Answers a client's opening handshake on `stream`. `allowed` decides on
/// its `Origin` (None from a native client), so a loopback stream refuses
/// pages from elsewhere.
pub fn accept(stream: TcpStream, allowed: impl Fn(Option<&str>) -> bool) -> Result<Socket> {
    let writer = stream.try_clone()?;
    let mut reader = BufReader::new(stream);
    let (request, headers) = read_head(&mut reader)?;
    let mut socket = Socket { reader, writer, client: false };
    let upgrade = header(&headers, "upgrade").is_some_and(|value| value.eq_ignore_ascii_case("websocket"));
    let key = header(&headers, "sec-websocket-key");
    if !request.starts_with("GET ") || !upgrade || key.is_none() || !allowed(header(&headers, "origin")) {
        socket.writer.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")?;
        bail!("refused a request that was no WebSocket handshake for this stream: {request}");
    }
    let accept = accept_key(key.unwrap_or_default());
    write!(
        socket.writer,
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
    )?;
    Ok(socket)
}

/// Opens a WebSocket to `host` (an address with its port) at `path`.
pub fn client(host: &str, path: &str) -> Result<Socket> {
    let stream = TcpStream::connect(host).with_context(|| format!("connecting to {host}"))?;
    stream.set_nodelay(true)?;
    let writer = stream.try_clone()?;
    let mut socket = Socket { reader: BufReader::new(stream), writer, client: true };
    // The key only proves the server speaks WebSocket; it need not be secret.
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
    let key = base64(&nonce.to_le_bytes());
    write!(
        socket.writer,
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
    )?;
    let (status, headers) = read_head(&mut socket.reader)?;
    ensure!(status.split_whitespace().nth(1) == Some("101"), "{host} did not open a WebSocket: {status}");
    ensure!(
        header(&headers, "sec-websocket-accept") == Some(accept_key(&key).as_str()),
        "{host} answered the WebSocket handshake wrongly"
    );
    Ok(socket)
}

/// What arrived: a whole message, or the other side closing.
#[derive(Debug, PartialEq)]
pub enum Incoming {
    Message(Vec<u8>),
    Closed,
}

impl Socket {
    fn frame(&mut self, opcode: u8, payload: &[u8]) -> Result<()> {
        let mut head = vec![0x80 | opcode];
        let mask = if self.client { 0x80 } else { 0 };
        match payload.len() {
            length @ 0..=125 => head.push(mask | u8::try_from(length)?),
            length @ 126..=0xffff => {
                head.push(mask | 126);
                head.extend_from_slice(&u16::try_from(length)?.to_be_bytes());
            }
            length => {
                head.push(mask | 127);
                head.extend_from_slice(&u64::try_from(length)?.to_be_bytes());
            }
        }
        if self.client {
            let key: [u8; 4] = std::process::id().to_le_bytes();
            head.extend_from_slice(&key);
            let masked: Vec<u8> = payload.iter().enumerate().map(|(at, byte)| byte ^ key[at % 4]).collect();
            self.writer.write_all(&head)?;
            self.writer.write_all(&masked)?;
        } else {
            self.writer.write_all(&head)?;
            self.writer.write_all(payload)?;
        }
        Ok(())
    }

    /// Sends a binary message.
    pub fn send(&mut self, message: &[u8]) -> Result<()> {
        self.frame(BINARY, message)
    }

    /// Says goodbye.
    pub fn close(&mut self) {
        let _ = self.frame(CLOSE, &[]);
    }

    /// Waits for the next message, answering pings on the way.
    pub fn receive(&mut self) -> Result<Incoming> {
        let mut message = Vec::new();
        loop {
            let mut head = [0; 2];
            if let Err(error) = self.reader.read_exact(&mut head) {
                return if error.kind() == std::io::ErrorKind::UnexpectedEof {
                    Ok(Incoming::Closed)
                } else {
                    Err(error.into())
                };
            }
            let (last, opcode, masked) = (head[0] & 0x80 != 0, head[0] & 0x0f, head[1] & 0x80 != 0);
            ensure!(masked == !self.client, "a WebSocket frame masked the wrong way");
            let length = match head[1] & 0x7f {
                126 => {
                    let mut bytes = [0; 2];
                    self.reader.read_exact(&mut bytes)?;
                    usize::from(u16::from_be_bytes(bytes))
                }
                127 => {
                    let mut bytes = [0; 8];
                    self.reader.read_exact(&mut bytes)?;
                    usize::try_from(u64::from_be_bytes(bytes))?
                }
                length => usize::from(length),
            };
            ensure!(message.len() + length <= MAX_MESSAGE, "a WebSocket message past {MAX_MESSAGE} bytes");
            let mut key = [0; 4];
            if masked {
                self.reader.read_exact(&mut key)?;
            }
            let mut payload = vec![0; length];
            self.reader.read_exact(&mut payload)?;
            if masked {
                for (at, byte) in payload.iter_mut().enumerate() {
                    *byte ^= key[at % 4];
                }
            }
            match opcode {
                CLOSE => {
                    self.close();
                    return Ok(Incoming::Closed);
                }
                PING => self.frame(PONG, &payload)?,
                PONG => {}
                CONTINUATION | BINARY | 1 => {
                    message.extend_from_slice(&payload);
                    if last {
                        return Ok(Incoming::Message(message));
                    }
                }
                _ => bail!("an unknown WebSocket frame {opcode}"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use super::*;

    #[test]
    fn the_accept_key_is_rfc_6455s() {
        assert_eq!(accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn messages_of_every_length_cross_both_ways() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut socket = accept(stream, |origin| origin.is_none()).unwrap();
            for length in [0, 125, 126, 65535, 65536, 300_000] {
                socket.send(&vec![7; length]).unwrap();
            }
            // The client's masked message, then its close.
            let echoed = socket.receive().unwrap();
            assert_eq!(socket.receive().unwrap(), Incoming::Closed);
            echoed
        });
        let mut socket = client(&address, "/").unwrap();
        for length in [0, 125, 126, 65535, 65536, 300_000] {
            assert_eq!(socket.receive().unwrap(), Incoming::Message(vec![7; length]));
        }
        socket.send(b"bye").unwrap();
        socket.close();
        assert_eq!(server.join().unwrap(), Incoming::Message(b"bye".to_vec()));
        assert_eq!(socket.receive().unwrap(), Incoming::Closed);
    }

    #[test]
    fn pages_from_elsewhere_are_refused() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            accept(stream, |origin| origin.is_none()).is_err()
        });
        let mut stream = TcpStream::connect(address).unwrap();
        stream
            .write_all(
                b"GET / HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nOrigin: https://elsewhere.example\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n",
            )
            .unwrap();
        let mut answer = String::new();
        stream.read_to_string(&mut answer).unwrap();
        assert!(answer.starts_with("HTTP/1.1 400"), "{answer}");
        assert!(server.join().unwrap());
    }
}
