//! Warm isolated guests with an explicit immutable prepared context. The guest
//! stays inside its callback entry; a failed job discards that entire session.
use crate::{Argument, GUEST_STACK_BYTES, Guest, GuestModule, Limits, Parameter, Value, ValueType};
use anyhow::{Context, Result, ensure};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};
use std::thread::{self, JoinHandle};
use wasmtime::Engine;

const MAGIC: f32 = -4670546.0;
const MAX_LANES: usize = 16_384;
const MAX_BATCH: usize = 32;
const MAX_MEMORY: usize = 64 << 20;
struct ContextValue {
    revision: u32,
    values: Vec<f32>,
}
struct Job {
    index: usize,
    input: Vec<f32>,
    context: Arc<ContextValue>,
}
type Answer = (usize, Result<Vec<f32>, String>);
#[derive(Default)]
struct Metrics {
    starts: AtomicUsize,
    contexts: AtomicUsize,
    context_lanes: AtomicUsize,
    jobs: AtomicUsize,
}
#[derive(Debug, serde::Serialize)]
pub struct Stats {
    /// Last observed committed guest bytes for each worker (zero after close).
    pub memory_bytes: Vec<usize>,
    pub starts: usize,
    pub contexts: usize,
    pub context_lanes: usize,
    pub jobs: usize,
}
#[derive(Default)]
struct Session {
    current: Option<Job>,
    context: Option<Arc<ContextValue>>,
    stage: u8,
}
impl Session {
    fn next(
        &mut self,
        response: Value,
        jobs: &mpsc::Receiver<Job>,
        answers: &mpsc::SyncSender<Answer>,
        metrics: &Metrics,
    ) -> Result<Value> {
        let response = response.into_f32_array()?;
        match self.stage {
            0 => ensure!(response == [MAGIC], "invalid worker session handshake"),
            1 => {
                ensure!(response == [1.0], "invalid context acknowledgement");
                self.context = self.current.as_ref().map(|job| job.context.clone());
                metrics.contexts.fetch_add(1, Ordering::Relaxed);
                metrics
                    .context_lanes
                    .fetch_add(self.context.as_ref().expect("context").values.len(), Ordering::Relaxed);
                self.stage = 2;
                return Ok(Value::ArrayF32(packet(2.0, &self.current.as_ref().expect("preparing job").input)));
            }
            2 => {
                ensure!(response.first() == Some(&2.0), "invalid worker job response");
                ensure!(
                    response.len() <= MAX_LANES + 1 && response.iter().all(|v| v.is_finite()),
                    "invalid worker output"
                );
                let job = self.current.take().expect("running job");
                metrics.jobs.fetch_add(1, Ordering::Relaxed);
                answers.send((job.index, Ok(response[1..].to_vec()))).context("worker answers closed")?;
            }
            _ => anyhow::bail!("worker called after shutdown"),
        }
        let Ok(job) = jobs.recv() else {
            self.stage = 3;
            return Ok(Value::ArrayF32(vec![0.0]));
        };
        let reuse = self.context.as_ref().is_some_and(|context| Arc::ptr_eq(context, &job.context));
        let request = if reuse { packet(2.0, &job.input) } else { packet(1.0, &job.context.values) };
        self.current = Some(job);
        self.stage = if reuse { 2 } else { 1 };
        Ok(Value::ArrayF32(request))
    }
}
fn packet(tag: f32, values: &[f32]) -> Vec<f32> {
    let mut packet = Vec::with_capacity(values.len() + 1);
    packet.push(tag);
    packet.extend_from_slice(values);
    packet
}
fn valid(values: &[f32]) -> bool {
    values.len() <= MAX_LANES && values.iter().all(|v| v.is_finite())
}

pub struct Workers {
    memory: Vec<Arc<AtomicUsize>>,
    metrics: Arc<Metrics>,
    send: Vec<mpsc::SyncSender<Job>>,
    receive: mpsc::Receiver<Answer>,
    threads: Vec<JoinHandle<()>>,
    context: Option<Arc<ContextValue>>,
}
impl Workers {
    pub fn new(engine: &Engine, module: &GuestModule, entry: &str, count: usize) -> Result<Self> {
        ensure!((1..=8).contains(&count), "worker count must be 1..=8");
        let function = module.abi().function(entry).context("missing worker service")?;
        ensure!(
            matches!(function.parameter, Parameter::Callback(signature)
            if signature.parameter == ValueType::ArrayF32 && signature.result == ValueType::ArrayF32)
                && function.result == ValueType::Unit,
            "worker service needs (Array F32 -> Array F32) -> Unit"
        );
        let (answers, receive) = mpsc::sync_channel(MAX_BATCH);
        let mut pool = Self {
            memory: vec![],
            metrics: Arc::new(Metrics::default()),
            send: vec![],
            receive,
            threads: vec![],
            context: None,
        };
        for index in 0..count {
            let (send, jobs) = mpsc::sync_channel::<Job>(MAX_BATCH);
            let engine = engine.clone();
            let module = module.clone();
            let entry = entry.to_owned();
            let answers = answers.clone();
            let metrics = pool.metrics.clone();
            let memory = Arc::new(AtomicUsize::new(0));
            pool.memory.push(memory.clone());
            let thread = thread::Builder::new()
                .name(format!("blot-session-{index}"))
                .stack_size(GUEST_STACK_BYTES + (2 << 20))
                .spawn(move || {
                    let jobs = Arc::new(Mutex::new(jobs));
                    loop {
                        let state = Arc::new(Mutex::new(Session::default()));
                        let callback_state = state.clone();
                        let callback_jobs = jobs.clone();
                        let callback_answers = answers.clone();
                        let callback_metrics = metrics.clone();
                        let callback = Box::new(move |response| {
                            callback_state.lock().map_err(|_| anyhow::anyhow!("poisoned session"))?.next(
                                response,
                                &*callback_jobs.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
                                &callback_answers,
                                &callback_metrics,
                            )
                        });
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                            let mut guest = Guest::sandboxed(&engine, &module, Limits { memory: MAX_MEMORY })?;
                            guest.track_memory(memory.clone());
                            metrics.starts.fetch_add(1, Ordering::Relaxed);
                            let called = guest.call(&entry, Argument::Callback(callback));
                            let disposed = guest.dispose();
                            called?;
                            disposed?;
                            Ok(())
                        }))
                        .map_err(|_| anyhow::anyhow!("worker session panicked"))
                        .and_then(|v| v);
                        memory.store(0, Ordering::Relaxed);
                        let mut state = state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                        if state.stage == 3 {
                            return;
                        }
                        let message =
                            result.err().map_or_else(|| "worker service exited early".to_owned(), |e| format!("{e:#}"));
                        if let Some(job) = state.current.take() {
                            if answers.send((job.index, Err(message))).is_err() {
                                return;
                            }
                        } else {
                            // Failure before requesting a job must still resolve the next
                            // admitted job, rather than spin indefinitely or lose a result.
                            let Ok(job) = jobs.lock().unwrap_or_else(|p| p.into_inner()).recv() else {
                                return;
                            };
                            if answers.send((job.index, Err(message))).is_err() {
                                return;
                            }
                        }
                        // Recreate after a trap, panic, invalid packet or early return.
                    }
                })?;
            pool.send.push(send);
            pool.threads.push(thread);
        }
        Ok(pool)
    }
    /// Reusing the current revision requires the same context bits. Input and result
    /// order is stable; worker completion order never chooses simulation order.
    pub fn run(
        &mut self,
        revision: u32,
        context: &[f32],
        inputs: Vec<Vec<f32>>,
    ) -> Result<Vec<Result<Vec<f32>, String>>> {
        // Reject the entire batch before changing context or submitting work.
        ensure!(inputs.len() <= MAX_BATCH && inputs.iter().all(|v| valid(v)), "worker batch exceeds numeric limits");
        self.configure(revision, context)?;
        self.run_prepared(inputs)
    }

    /// Set explicit context once, then reuse it across run_prepared batches.
    /// Reusing the current revision requires identical bits, including zeros.
    pub fn configure(&mut self, revision: u32, context: &[f32]) -> Result<()> {
        ensure!(!self.send.is_empty(), "worker pool is closed");
        ensure!(valid(context), "worker context exceeds numeric limits");
        if let Some(before) = &self.context
            && before.revision == revision
        {
            ensure!(
                before.values.len() == context.len()
                    && before.values.iter().zip(context).all(|(a, b)| a.to_bits() == b.to_bits()),
                "context changed without a new revision"
            );
        } else {
            self.context = Some(Arc::new(ContextValue { revision, values: context.to_vec() }));
        }
        Ok(())
    }

    pub fn run_prepared(&mut self, inputs: Vec<Vec<f32>>) -> Result<Vec<Result<Vec<f32>, String>>> {
        ensure!(!self.send.is_empty(), "worker pool is closed");
        ensure!(inputs.len() <= MAX_BATCH && inputs.iter().all(|v| valid(v)), "worker batch exceeds numeric limits");
        if inputs.is_empty() {
            return Ok(vec![]);
        }
        let selected = self.context.as_ref().context("worker context is not configured")?.clone();
        let count = inputs.len();
        for (index, input) in inputs.into_iter().enumerate() {
            self.send[index % self.send.len()]
                .send(Job { index, input, context: selected.clone() })
                .context("worker transport closed")?;
        }
        let mut results: Vec<_> = (0..count).map(|_| None).collect();
        for _ in 0..count {
            let (index, value) = self.receive.recv().context("worker transport failed")?;
            ensure!(index < count && results[index].is_none(), "worker result identity changed");
            results[index] = Some(value);
        }
        Ok(results.into_iter().map(Option::unwrap).collect())
    }

    /// Optional copied-lane callback for a client guest. It configures context
    /// separately from jobs, so steady batches do not resend geometry. A job
    /// failure rejects the callback result; callers publish no partial wave.
    pub fn call(&mut self, request: Value) -> Result<Value> {
        const CLIENT: f32 = -4670547.0;
        const BATCH: f32 = -4670545.0;
        let values = request.into_f32_array()?;
        ensure!(values.len() >= 2 && values[0] == CLIENT, "invalid prepared-worker request");
        ensure!(
            values.len() <= MAX_BATCH * (MAX_LANES + 1) + 4 && values.iter().all(|v| v.is_finite()),
            "invalid prepared-worker packet"
        );
        match values[1] {
            0.0 => {
                ensure!(values.len() == 2 && !self.send.is_empty(), "invalid worker reset");
                self.reset();
                Ok(Value::ArrayF32(vec![CLIENT, 0.0]))
            }
            1.0 => {
                ensure!(values.len() >= 5, "short worker context");
                let low = integer(values[2], 65535)?;
                let high = integer(values[3], 65535)?;
                let size = integer(values[4], MAX_LANES)?;
                ensure!(values.len() == 5 + size, "worker context framing");
                self.configure((low | high << 16) as u32, &values[5..])?;
                Ok(Value::ArrayF32(vec![CLIENT, 1.0]))
            }
            2.0 => {
                ensure!(values.len() >= 4 && values[2] == BATCH, "invalid worker batch");
                let count = integer(values[3], MAX_BATCH)?;
                let mut inputs = Vec::with_capacity(count);
                let mut cursor = 4;
                for _ in 0..count {
                    let size = integer(*values.get(cursor).context("missing worker input length")?, MAX_LANES)?;
                    cursor += 1;
                    let end = cursor + size;
                    inputs.push(values.get(cursor..end).context("short worker input")?.to_vec());
                    cursor = end;
                }
                ensure!(cursor == values.len(), "trailing worker input");
                let results = self.run_prepared(inputs)?;
                let mut output = vec![BATCH, count as f32];
                for (index, result) in results.into_iter().enumerate() {
                    let result = result.map_err(|error| anyhow::anyhow!("worker job {index}: {error}"))?;
                    output.push(result.len() as f32);
                    output.extend_from_slice(&result);
                }
                Ok(Value::ArrayF32(output))
            }
            _ => anyhow::bail!("unknown prepared-worker command"),
        }
    }
    pub fn stats(&self) -> Stats {
        Stats {
            memory_bytes: self.memory.iter().map(|v| v.load(Ordering::Relaxed)).collect(),
            starts: self.metrics.starts.load(Ordering::Relaxed),
            contexts: self.metrics.contexts.load(Ordering::Relaxed),
            context_lanes: self.metrics.context_lanes.load(Ordering::Relaxed),
            jobs: self.metrics.jobs.load(Ordering::Relaxed),
        }
    }
    /// The next batch re-prepares every worker it uses, even for the same revision.
    pub fn reset(&mut self) {
        self.context = None;
    }
    pub fn close(&mut self) -> Result<()> {
        self.send.clear();
        self.context = None;
        let mut panicked = false;
        for thread in self.threads.drain(..) {
            panicked |= thread.join().is_err();
        }
        ensure!(!panicked, "worker session thread panicked");
        Ok(())
    }
}
fn integer(value: f32, maximum: usize) -> Result<usize> {
    ensure!(
        value.is_finite() && value >= 0.0 && value <= maximum as f32 && value.fract() == 0.0,
        "worker integer outside range"
    );
    Ok(value as usize)
}
impl Drop for Workers {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
