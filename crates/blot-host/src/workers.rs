//! Experimental bounded CPU batches. Workers share compiled code, never guest
//! memory or pointers. Every job gets a fresh instance; completion order cannot
//! leak module state into another job. These are independent value calls, not
//! migrations of a live structured-task stack.
use crate::{GUEST_STACK_BYTES, Guest, GuestModule, Limits, Parameter, Value};
use anyhow::{Context, Result, ensure};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use wasmtime::Engine;

const MAX_WORKERS: usize = 8;
const MAX_BATCH: usize = 32;
const MAX_LANES: usize = 16_384;
const MAX_MEMORY: usize = 64 << 20;
struct Job {
    index: usize,
    input: Value,
    timing: Option<(Instant, Duration)>,
}
/// Host wall-clock evidence for one copied value job. This is diagnostic data,
/// never an input to simulation or a claim about a suspended guest task.
#[derive(Clone, Debug)]
pub struct Timing {
    pub job: usize,
    pub worker: usize,
    pub submitted: Duration,
    pub started: Duration,
    pub finished: Duration,
    pub succeeded: bool,
}
type Answer = (usize, Result<Value, String>, Option<Timing>);
pub struct Workers {
    send: Option<mpsc::SyncSender<Job>>,
    receive: mpsc::Receiver<Answer>,
    threads: Vec<JoinHandle<()>>,
    tracing: bool,
    timings: Vec<Timing>,
}
fn bounded(value: &Value) -> bool {
    match value {
        Value::ArrayU32(values) => values.len() <= MAX_LANES,
        Value::ArrayF32(values) => values.len() <= MAX_LANES,
        _ => true,
    }
}
impl Workers {
    pub fn new(engine: &Engine, module: &GuestModule, entry: &str, count: usize) -> Result<Self> {
        ensure!((1..=MAX_WORKERS).contains(&count), "worker count must be 1..={MAX_WORKERS}");
        let function = module.abi().function(entry).context("missing worker entry")?;
        ensure!(matches!(function.parameter, Parameter::Value(_)), "workers accept copied values, not callbacks");
        let (send, jobs) = mpsc::sync_channel::<Job>(MAX_BATCH);
        let jobs = Arc::new(Mutex::new(jobs));
        let (answers, receive) = mpsc::sync_channel::<Answer>(MAX_BATCH);
        let mut pool = Self { send: Some(send), receive, threads: vec![], tracing: false, timings: vec![] };
        for index in 0..count {
            let engine = engine.clone();
            let module = module.clone();
            let entry = entry.to_owned();
            let jobs = jobs.clone();
            let answers = answers.clone();
            let thread = thread::Builder::new()
                .name(format!("blot-worker-{index}"))
                .stack_size(GUEST_STACK_BYTES + (2 << 20))
                .spawn(move || {
                    loop {
                        let job = match jobs.lock() {
                            Ok(queue) => queue.recv(),
                            Err(_) => return,
                        };
                        let Ok(job) = job else { return };
                        let started = job.timing.map(|(origin, _)| origin.elapsed());
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Value> {
                            let mut guest = Guest::sandboxed(&engine, &module, Limits { memory: MAX_MEMORY })?;
                            let called = guest.call(&entry, job.input);
                            let disposed = guest.dispose();
                            let result = called?;
                            disposed?;
                            ensure!(bounded(&result), "worker result exceeds {MAX_LANES} lanes");
                            Ok(result)
                        }))
                        .map_err(|_| "worker execution panicked".to_owned())
                        .and_then(|result| result.map_err(|error| format!("{error:#}")));
                        let timing = job.timing.map(|(origin, submitted)| Timing {
                            job: job.index,
                            worker: index,
                            submitted,
                            started: started.expect("timed job has a start"),
                            finished: origin.elapsed(),
                            succeeded: result.is_ok(),
                        });
                        if answers.send((job.index, result, timing)).is_err() {
                            return;
                        }
                    }
                })?;
            pool.threads.push(thread);
        }
        Ok(pool)
    }
    /// Retain at most the latest accepted batch (32 jobs). Disabled by default;
    /// disabling clears old measurements and avoids per-job clock reads.
    pub fn set_tracing(&mut self, enabled: bool) {
        self.tracing = enabled;
        if !enabled {
            self.timings.clear();
        }
    }
    pub fn timings(&self) -> &[Timing] {
        &self.timings
    }
    /// A batch is admitted as a whole. All jobs finish before this returns;
    /// outputs and failures retain input order, independent of completion time.
    pub fn run(&mut self, inputs: Vec<Value>) -> Result<Vec<Result<Value, String>>> {
        ensure!(inputs.len() <= MAX_BATCH, "worker batch exceeds {MAX_BATCH} jobs");
        ensure!(inputs.iter().all(bounded), "worker input exceeds {MAX_LANES} lanes");
        let sender = self.send.as_ref().context("worker pool is closed")?;
        self.timings.clear();
        let origin = self.tracing.then(Instant::now);
        let count = inputs.len();
        for (index, input) in inputs.into_iter().enumerate() {
            sender
                .send(Job { index, input, timing: origin.map(|origin| (origin, origin.elapsed())) })
                .context("worker transport closed")?;
        }
        let mut results: Vec<_> = (0..count).map(|_| None).collect();
        for _ in 0..count {
            let (index, result, timing) = self.receive.recv().context("worker transport failed")?;
            ensure!(index < count && results[index].is_none(), "worker result identity changed");
            results[index] = Some(result);
            if let Some(timing) = timing {
                self.timings.push(timing);
            }
        }
        self.timings.sort_unstable_by_key(|timing| timing.job);
        Ok(results.into_iter().map(Option::unwrap).collect())
    }
    pub fn close(&mut self) -> Result<()> {
        self.send = None;
        let mut panicked = false;
        for thread in self.threads.drain(..) {
            panicked |= thread.join().is_err();
        }
        ensure!(!panicked, "worker thread panicked");
        Ok(())
    }
}
impl Drop for Workers {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
