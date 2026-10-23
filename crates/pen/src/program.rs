//! Drives one long-lived `main(host)`, the game's program. The guest runs on
//! its own thread; a pending host call is the frame boundary, and answering
//! it resumes the guest until it next exchanges a response.
#[cfg(feature = "devtools")]
use std::sync::Arc;
#[cfg(feature = "devtools")]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, SyncSender, channel, sync_channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use blot_host::{Argument, Guest, GuestModule, Overrun, Value};
use pen_runtime::program::{Response, Unresponsive};
use wasmtime::Engine;

pub use pen_runtime::program::Operation;

/// How long the host waits for the guest's reply before it gives the game
/// up. A guest of the development engine traps at its deadline long before
/// ([`blot_host::DEADLINE`]); one of another engine, whose code spends
/// nothing checking the time, may loop until this ends the game.
const REPLY_TIMEOUT: Duration = Duration::from_secs(60);

/// What the guest thread sends back.
enum Reply {
    /// A packet, and when the guest sent it.
    Packet(Vec<f32>, Instant),
    Finished(Result<()>),
}

pub struct Program {
    pub schema: u32,
    /// Whether the game keeps scenes (for the editor's save and undo).
    pub scenes: bool,
    requests: Option<SyncSender<Vec<f32>>>,
    replies: Receiver<Reply>,
    thread: Option<JoinHandle<()>>,
    failure: Option<String>,
    /// The guest thread stopped answering: closing leaves it be.
    stuck: bool,
    commit: bool,
    stopping: bool,
    /// The operation sent and not yet answered.
    pending: Option<Operation>,
    /// When the guest sent its last packet.
    answered: Option<Instant>,
    /// A response's buffer, for the next request (and, the guest thread
    /// keeping it, the response after) to reuse.
    spare: Vec<f32>,
    /// The guest's linear memory, as of its last host call (for the dev tools).
    #[cfg(feature = "devtools")]
    memory: Arc<AtomicUsize>,
    /// What the dev tools see of the guest: its memory, its ownership
    /// counters, and samples of it while profiled.
    #[cfg(feature = "devtools")]
    probe: Arc<blot_host::Probe>,
}

impl Program {
    pub fn start(engine: &Engine, module: &GuestModule) -> Result<Program> {
        module.abi().check_game()?;
        let (request_tx, request_rx) = sync_channel::<Vec<f32>>(1);
        // Replies never block, so the guest can always finish and be joined.
        let (reply_tx, reply_rx): (Sender<Reply>, _) = channel();
        let engine = engine.clone();
        let module = module.clone();
        #[cfg(feature = "devtools")]
        let memory = Arc::new(AtomicUsize::new(0));
        #[cfg(feature = "devtools")]
        let gauge = memory.clone();
        #[cfg(feature = "devtools")]
        let probe = blot_host::Probe::new();
        #[cfg(feature = "devtools")]
        let probing = probe.clone();
        let thread = std::thread::Builder::new()
            .name("pen-guest".into())
            .stack_size(blot_host::GUEST_STACK_BYTES + (8 << 20))
            .spawn(move || {
                let finished_tx = reply_tx.clone();
                let result = (|| -> Result<()> {
                    let mut guest = Guest::new(&engine, &module)?;
                    #[cfg(feature = "devtools")]
                    {
                        guest.track_memory(gauge);
                        guest.probe(probing);
                    }
                    let callback = Box::new(move |packet: Value| -> Result<Value> {
                        reply_tx
                            .send(Reply::Packet(packet.into_f32_array()?, Instant::now()))
                            .map_err(|_| anyhow!("host closed"))?;
                        let request = request_rx.recv().map_err(|_| anyhow!("host closed"))?;
                        Ok(Value::ArrayF32(request))
                    });
                    guest.call(module.main_entry(), Argument::Callback(callback))?;
                    // Torn down as Blot's adapter disposes of an instance.
                    guest.dispose()
                })();
                let _ = finished_tx.send(Reply::Finished(result));
            })?;
        let mut program = Program {
            schema: 0,
            scenes: false,
            requests: Some(request_tx),
            replies: reply_rx,
            thread: Some(thread),
            failure: None,
            stuck: false,
            commit: true,
            stopping: false,
            pending: None,
            answered: None,
            spare: Vec::new(),
            #[cfg(feature = "devtools")]
            memory,
            #[cfg(feature = "devtools")]
            probe,
        };
        let description = program.receive().context("game program handshake")?;
        (program.schema, program.scenes) = pen_runtime::program::handshake(&description)?;
        program.recycle(description);
        Ok(program)
    }

    fn receive(&mut self) -> Result<Vec<f32>> {
        if let Some(failure) = &self.failure {
            bail!("game program failed: {failure}");
        }
        match self.replies.recv_timeout(REPLY_TIMEOUT) {
            Ok(Reply::Packet(packet, sent)) => {
                self.answered = Some(sent);
                Ok(packet)
            }
            Ok(Reply::Finished(result)) => {
                let error = match result {
                    Ok(()) if self.stopping => return Ok(Vec::new()),
                    Ok(()) => anyhow!("game main returned before shutdown"),
                    Err(error) => error,
                };
                self.failure = Some(format!("{error:#}"));
                let error = error.context("game program failed");
                // Stopped at its deadline, the program is gone, but a moment
                // restored starts another.
                Err(if error.is::<Overrun>() { error.context(Unresponsive) } else { error })
            }
            Err(RecvTimeoutError::Timeout) => {
                self.failure = Some("it stopped answering".into());
                self.stuck = true;
                Err(anyhow!("game program sent nothing for {} s", REPLY_TIMEOUT.as_secs()).context(Unresponsive))
            }
            Err(RecvTimeoutError::Disconnected) => {
                self.failure = Some("game thread stopped".into());
                bail!("game thread stopped")
            }
        }
    }

    /// Sends `operation` with `payload` and returns the response after its
    /// echoed operation lane.
    pub fn request(&mut self, operation: Operation, payload: &[f32]) -> Result<Vec<f32>> {
        self.send(operation, payload)?;
        Ok(self.response()?.into_lanes())
    }

    /// Sends a request without waiting: the guest works on it while the
    /// caller does something else, until `response`.
    pub fn send(&mut self, operation: Operation, payload: &[f32]) -> Result<()> {
        ensure!(!self.stopping, "game program is closed");
        ensure!(self.pending.is_none(), "game program already has a request in flight");
        let sender = self.requests.as_ref().ok_or_else(|| anyhow!("game program is closed"))?;
        if let Some(failure) = &self.failure {
            bail!("game program failed: {failure}");
        }
        let mut request = std::mem::take(&mut self.spare);
        pen_runtime::program::write_request(&mut request, self.commit, operation, payload);
        sender.send(request).map_err(|_| anyhow!("game thread stopped"))?;
        // A candidate must be accepted by the runtime before a later command
        // commits it; rejection simply keeps the preceding guest-owned state.
        self.commit = false;
        self.pending = Some(operation);
        Ok(())
    }

    /// When the guest finished its last response (perhaps well before the
    /// host took it).
    pub fn answered(&self) -> Option<Instant> {
        self.answered
    }

    /// Waits for the response to the request in flight.
    pub fn response(&mut self) -> Result<Response> {
        let operation = self.pending.take().ok_or_else(|| anyhow!("game program has no request in flight"))?;
        pen_runtime::program::response(operation, self.receive()?)
    }

    /// Takes back a response's buffer, for the next request to reuse.
    pub fn recycle(&mut self, buffer: Vec<f32>) {
        if buffer.capacity() > self.spare.capacity() {
            self.spare = buffer;
        }
    }

    /// The guest's committed linear-memory bytes, as of its last response.
    #[cfg(feature = "devtools")]
    pub fn memory_bytes(&self) -> usize {
        self.memory.load(Ordering::Relaxed)
    }

    /// What the dev tools see of the guest.
    #[cfg(feature = "devtools")]
    pub fn probe(&self) -> &Arc<blot_host::Probe> {
        &self.probe
    }

    pub fn accept(&mut self) {
        self.commit = true;
    }

    /// Asks the program to stop and waits for its thread (unless it is stuck).
    pub fn close(&mut self) {
        if self.stopping {
            return;
        }
        self.stopping = true;
        if let Some(sender) = self.requests.take()
            && self.failure.is_none()
        {
            let _ = sender.send(vec![0.0, Operation::Stop as u8 as f32]);
        }
        if let Some(thread) = self.thread.take()
            && !self.stuck
        {
            let _ = thread.join();
        }
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        self.close();
    }
}

#[derive(Clone)]
pub struct NativeBackend(pub Engine);
impl pen_runtime::program::Backend for NativeBackend {
    type Module = GuestModule;
    type Program = Program;
    async fn start(&self, module: &GuestModule) -> Result<Program> {
        Program::start(&self.0, module)
    }
}
impl pen_runtime::program::Program for Program {
    fn schema(&self) -> u32 {
        self.schema
    }
    fn scenes(&self) -> bool {
        self.scenes
    }
    fn send(&mut self, operation: Operation, payload: &[f32]) -> Result<()> {
        self.send(operation, payload)
    }
    async fn response(&mut self) -> Result<Response> {
        self.response()
    }
    fn recycle(&mut self, buffer: Vec<f32>) {
        self.recycle(buffer);
    }
    fn accept(&mut self) {
        self.accept();
    }
    fn answered(&self) -> Option<Instant> {
        self.answered()
    }
    fn memory_bytes(&self) -> usize {
        #[cfg(feature = "devtools")]
        {
            self.memory_bytes()
        }
        #[cfg(not(feature = "devtools"))]
        {
            0
        }
    }
    async fn close(&mut self) {
        self.close();
    }
}
