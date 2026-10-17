//! Remote dev tools: a running game answers the dev tools service over TCP
//! (`--devtools-listen ADDRESS`), as it answers its own dev tools, so a
//! packaged game played by pen or a game on another machine can be asked
//! for its log and frames, paused, stepped, rewound and the rest from
//! elsewhere. `pen remote ADDRESS [OPERATION [A [B [C [D]]]]]` asks it one
//! thing and prints the answer (`log --follow` tails the log). The messages
//! are pen-runtime's (`pen_runtime::remote`); each connection has a service
//! state of its own, as the dev tools program does.
#![cfg_attr(not(feature = "devtools"), allow(dead_code))]
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use pen_protocol::{ServiceRequest, devtools_frame, devtools_op, service, service_status};
use pen_runtime::devtools::Devtools;
use pen_runtime::remote::{Messages, answer_message, read_answer, read_request, request_message};
use pen_runtime::services::{Answer, code_points, text};

/// Where a game listens for remote dev tools unless told a port.
pub const PORT: u16 = 7812;

/// `address`, with the default port when it names none.
pub fn with_port(address: &str) -> String {
    if address.contains(':') { address.to_owned() } else { format!("{address}:{PORT}") }
}

/// A connection to a remote client: what it sent so far, what goes back,
/// and the service's state for it.
struct Client {
    stream: TcpStream,
    from: String,
    messages: Messages,
    out: Vec<u8>,
    service: Devtools,
    previous: Option<Instant>,
}

/// The game's end: listens, and answers each client's requests once a frame.
pub struct Remote {
    listener: TcpListener,
    clients: Vec<Client>,
}

impl Remote {
    pub fn listen(address: &str) -> Result<Remote> {
        let address = with_port(address);
        let listener =
            TcpListener::bind(&address).with_context(|| format!("listening for remote dev tools on {address}"))?;
        listener.set_nonblocking(true)?;
        eprintln!("devtools: remote dev tools listen on {}", listener.local_addr()?);
        Ok(Remote { listener, clients: Vec::new() })
    }

    #[cfg(test)]
    pub fn address(&self) -> Result<std::net::SocketAddr> {
        Ok(self.listener.local_addr()?)
    }

    /// Takes in new clients and what they sent, answers each request with
    /// `answer` (the client's service state, the request and the time since
    /// its last frame request in ms), and sends the answers. A client whose
    /// connection fails or who sends nonsense is let go.
    pub fn serve(&mut self, mut answer: impl FnMut(&mut Devtools, &ServiceRequest, f64) -> Answer) {
        while let Ok((stream, from)) = self.listener.accept() {
            if stream.set_nonblocking(true).is_ok() {
                eprintln!("devtools: remote dev tools from {from}");
                self.clients.push(Client {
                    stream,
                    from: from.to_string(),
                    messages: Messages::default(),
                    out: Vec::new(),
                    service: Devtools::default(),
                    previous: None,
                });
            }
        }
        self.clients.retain_mut(|client| match client.serve(&mut answer) {
            Ok(open) => open,
            Err(error) => {
                eprintln!("devtools: remote dev tools from {} let go: {error:#}", client.from);
                false
            }
        });
    }
}

impl Remote {
    /// Each client's service state, for what it does every frame (a watch).
    pub fn follow(&mut self, mut each: impl FnMut(&mut Devtools)) {
        for client in &mut self.clients {
            each(&mut client.service);
        }
    }
}

impl Client {
    /// Reads, answers and writes what it can now; whether it is still open.
    fn serve(&mut self, answer: &mut impl FnMut(&mut Devtools, &ServiceRequest, f64) -> Answer) -> Result<bool> {
        let mut buffer = [0; 1 << 14];
        let mut open = true;
        loop {
            match self.stream.read(&mut buffer) {
                Ok(0) => {
                    open = false;
                    break;
                }
                Ok(read) => self.messages.arrived(&buffer[..read]),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error.into()),
            }
        }
        while let Some(lanes) = self.messages.next_message()? {
            let request = read_request(&lanes)?;
            let answered = if request.service == service::DEVTOOLS {
                let now = Instant::now();
                let interval = match request.operation {
                    devtools_op::FRAME => {
                        self.previous.replace(now).map_or(0.0, |previous| (now - previous).as_secs_f64() * 1000.0)
                    }
                    _ => 0.0,
                };
                answer(&mut self.service, &request, interval)
            } else {
                Answer::unknown()
            };
            self.out.extend(answer_message(request.request, &answered));
        }
        while !self.out.is_empty() {
            match self.stream.write(&self.out) {
                Ok(0) => bail!("the connection closed"),
                Ok(written) => {
                    self.out.drain(..written);
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error.into()),
            }
        }
        Ok(open || !self.out.is_empty())
    }
}

/// The client's end: a connection to a game's remote dev tools.
pub struct Connection {
    stream: TcpStream,
    messages: Messages,
    next: u32,
}

impl Connection {
    pub fn open(address: &str) -> Result<Connection> {
        let address = with_port(address);
        let target = address.to_socket_addrs()?.next().with_context(|| format!("{address} names no address"))?;
        let stream = TcpStream::connect_timeout(&target, Duration::from_secs(5))
            .with_context(|| format!("connecting to the game's remote dev tools at {address}"))?;
        stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        Ok(Connection { stream, messages: Messages::default(), next: 1 })
    }

    /// Asks the game's dev tools service `operation` with `numbers` (`a` to
    /// `d`) and `values`, and waits for the answer.
    pub fn ask(&mut self, operation: u32, numbers: [f32; 4], values: Vec<f32>) -> Result<Answer> {
        let [a, b, c, d] = numbers;
        let id = self.next;
        self.next += 1;
        let value_count = u32::try_from(values.len())?;
        let request =
            ServiceRequest { request: id, service: service::DEVTOOLS, operation, a, b, c, d, value_count, values };
        self.stream.write_all(&request_message(&request))?;
        let mut buffer = [0; 1 << 14];
        loop {
            if let Some(lanes) = self.messages.next_message()? {
                let (answered, answer) = read_answer(&lanes)?;
                ensure!(answered == id, "the game answered request {answered}, not {id}");
                return Ok(answer);
            }
            let read = self.stream.read(&mut buffer).context("waiting for the game's answer")?;
            ensure!(read > 0, "the game closed the connection");
            self.messages.arrived(&buffer[..read]);
        }
    }
}

/// An answer's log lines, as `log_answer` lays them out: each its source,
/// whether an error, and its text.
fn log_lines(values: &[f32]) -> Vec<(u32, bool, String)> {
    let mut lines = Vec::new();
    let mut at = 0;
    while at + 3 <= values.len() {
        let Some(length) = pen_protocol::boundary::whole(values[at + 2]).map(|length| length as usize) else { break };
        let Some(codes) = values.get(at + 3..at + 3 + length) else { break };
        let source = pen_protocol::boundary::whole(values[at]).unwrap_or(0);
        lines.push((source, values[at + 1] > 0.5, text(codes)));
        at += 3 + length;
    }
    lines
}

/// An answer as `pen remote` prints it: the frame's numbers by name, the
/// log's lines, text as text, and other numbers as they are.
pub fn describe(operation: u32, answer: &Answer) -> String {
    let status = pen_protocol::service_status::NAMES.get(answer.status as usize).copied().unwrap_or("?");
    if answer.status != service_status::OK {
        let said = text(&answer.values);
        return if said.is_empty() { status.to_owned() } else { format!("{status}: {said}") };
    }
    match operation {
        devtools_op::FRAME => devtools_frame::NAMES
            .iter()
            .zip(&answer.values)
            .map(|(name, value)| format!("{name} {value}"))
            .collect::<Vec<_>>()
            .join("\n"),
        devtools_op::LOG => log_lines(&answer.values)
            .into_iter()
            .map(|(_, error, line)| if error { format!("error: {line}") } else { line })
            .collect::<Vec<_>>()
            .join("\n"),
        devtools_op::GPU | devtools_op::PRESETS => text(&answer.values),
        _ => {
            let values: Vec<String> = answer.values.iter().map(f32::to_string).collect();
            if values.is_empty() {
                format!("ok {}", answer.value)
            } else {
                format!("ok {}: {}", answer.value, values.join(" "))
            }
        }
    }
}

/// `pen remote ADDRESS [OPERATION [A [B [C [D]]]]]`: asks the game at
/// `address` (its `--devtools-listen`) one thing, `--text` the request's
/// text, and prints the answer; `log --follow` prints the log as it grows.
pub fn command(address: &str, words: &[String], text_value: Option<&str>, follow: bool) -> Result<()> {
    let name = words.first().map_or("frame", String::as_str);
    let Some(operation) = devtools_op::NAMES.iter().position(|known| *known == name) else {
        bail!("{name} is not a dev tools operation; they are {}", devtools_op::NAMES.join(", "));
    };
    let operation = u32::try_from(operation)?;
    let mut numbers = [0.0; 4];
    for (at, word) in words.iter().skip(1).enumerate() {
        ensure!(at < 4, "an operation takes at most four numbers");
        numbers[at] = word.parse().with_context(|| format!("{word} is not a number"))?;
    }
    let values = text_value.map(code_points).unwrap_or_default();
    let mut connection = Connection::open(address)?;
    if follow {
        ensure!(operation == devtools_op::LOG, "only the log can be followed");
        let mut from = numbers[0];
        loop {
            let answer = connection.ask(operation, [from, 0.0, 0.0, 0.0], Vec::new())?;
            for (_, error, line) in log_lines(&answer.values) {
                println!("{}{line}", if error { "error: " } else { "" });
            }
            from = answer.value;
            std::thread::sleep(Duration::from_millis(250));
        }
    }
    let answer = connection.ask(operation, numbers, values)?;
    println!("{}", describe(operation, &answer));
    if answer.status != service_status::OK {
        bail!("the game answered {name} with {}", pen_protocol::service_status::NAMES[answer.status as usize]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_runtime::devtools::LogLine;

    #[test]
    fn a_game_answers_remote_dev_tools_each_frame() {
        let mut remote = Remote::listen("127.0.0.1:0").unwrap();
        let address = remote.address().unwrap().to_string();
        let client = std::thread::spawn(move || {
            let mut connection = Connection::open(&address).unwrap();
            let paused = connection.ask(devtools_op::PAUSE, [0.0; 4], Vec::new()).unwrap();
            let log = connection.ask(devtools_op::LOG, [0.0; 4], Vec::new()).unwrap();
            (paused, log)
        });
        let mut asked = Vec::new();
        for _ in 0..500 {
            remote.serve(|_, request, _| {
                asked.push(request.operation);
                match request.operation {
                    devtools_op::LOG => pen_runtime::devtools::log_answer(
                        [LogLine { serial: 0, source: 5, error: false, text: "paused at frame 9" }],
                        0,
                    ),
                    _ => Answer::ok(0.0),
                }
            });
            if asked.len() == 2 {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let (paused, log) = client.join().unwrap();
        assert_eq!(asked, [devtools_op::PAUSE, devtools_op::LOG]);
        assert_eq!(paused, Answer::ok(0.0));
        assert_eq!(describe(devtools_op::LOG, &log), "paused at frame 9");
        assert_eq!(describe(devtools_op::PAUSE, &Answer::unavailable()), "unavailable");
        assert_eq!(with_port("deck.local"), format!("deck.local:{PORT}"));
    }
}
