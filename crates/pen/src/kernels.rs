//! Native Pen `service::KERNEL` (code 15) host service (`blot2/tasks/CONCURRENCY_VNEXT.md` §5.3).
//!
//! All host→guest kernel traffic travels exclusively on `GameEvent::ServiceAnswer { request: 0 }`
//! and immediately following contiguous `GameEvent::ServiceData { request: 0 }`:
//! - Live-only unrecorded `CAP` (`value = 0.0`, `status = OK`, `length = 6` — 3 events) is appended
//!   at the very tail of `guest_input` and stripped BEFORE constructing `Recorded` so recordings
//!   replay cleanly on both new and old (`service <= 14`) Pen hosts.
//! - Recorded self-identifying replies (`value = kernel_req_id >= 1`):
//!   - `REPLY_KIND_CTL = 1.0` (`length = 3` — 2 events): `[1.0, error_code, 0.0]`
//!   - `REPLY_KIND_BATCH_ERR = 2.0` (`length = 3` — 2 events): `[2.0, error_code, kernel_id]`
//!   - `REPLY_KIND_BATCH_OK = 3.0` (`length = 13 + payload`):
//!     `[3.0, kernel_id_lo16, kernel_id_hi16, binding_w0..w3, geom_w0..w3, count_lo16, count_hi16, outputs...]`

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use blot_host::GuestModule;
use blot_host::kernels::{KernelConfig, KernelError, KernelPool};
use pen_protocol::boundary::{EVENT_FLOATS, FRAME_HEADER_FLOATS, MAX_EVENTS, MAX_SERVICE_DATA, event_tags, whole};
use pen_protocol::{ServiceRequest, kernel_op, service, service_status};
use pen_runtime::abi::{KernelManifest, ValueType, WireSchema, WireSchemaKind};

pub const REPLY_KIND_CTL: f32 = 1.0;
pub const REPLY_KIND_BATCH_ERR: f32 = 2.0;
pub const REPLY_KIND_BATCH_OK: f32 = 3.0;

pub const ERR_NONE: u32 = 0;
pub const ERR_UNKNOWN_OP: u32 = 1;
pub const ERR_UNKNOWN_KERNEL: u32 = 2;
pub const ERR_STALE_EPOCH: u32 = 3;
pub const ERR_INVALID_REQUEST: u32 = 4;
pub const ERR_UPLOAD_SEQUENCE: u32 = 5;
pub const ERR_UPLOAD_TOO_LARGE: u32 = 6;
pub const ERR_HASH_MISMATCH: u32 = 7;
pub const ERR_BINDING_MISMATCH: u32 = 8;
pub const ERR_CTX_EVICTED: u32 = 9;
pub const ERR_CTX_MISSING: u32 = 10;
pub const ERR_EXECUTION_FAILED: u32 = 11;
pub const ERR_OUTPUT_TOO_LARGE: u32 = 12;
pub const ERR_NONFINITE_OUTPUT: u32 = 13;
pub const ERR_TIMEOUT: u32 = 14;
pub const ERR_EVENT_BUDGET: u32 = 15;
pub const ERR_UNAVAILABLE: u32 = 16;

/// Aggregate active context upload byte ceiling across all kernels (4 MiB).
const MAX_AGGREGATE_UPLOAD_BYTES: usize = 4 * 1024 * 1024;
/// Maximum cached committed contexts per kernel.
const MAX_LRU_ENTRIES_PER_KERNEL: usize = 2;
/// Maximum total bytes across all cached committed contexts (4 MiB).
const MAX_LRU_TOTAL_BYTES: usize = 4 * 1024 * 1024;
/// Maximum kernel reply events spliced into a single live frame.
const MAX_KERNEL_REPLY_EVENTS: usize = 65_536;
/// Number of events in a live-only `CAP` announcement (`ServiceAnswer` + 2 `ServiceData`).
pub const CAP_EVENTS: usize = 3;
/// Per-batch deadline when running on an epoch-enabled worker engine (`dev_engine` / `--worker-deadlines`).
const BOUNDED_BATCH_DEADLINE: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CtxKey {
    kernel_id: u32,
    binding_hash: u64,
    geom_hash: u64,
}

struct ActiveUpload {
    host_epoch: u64,
    upload_seq: u32,
    binding_hash: u64,
    geom_hash: u64,
    total_chunks: u32,
    total_words: usize,
    next_chunk: u32,
    words: Vec<u32>,
}

struct LruEntry {
    key: CtxKey,
    words: Arc<[u32]>,
    revision: u64,
    committed_upload_key: (u64, u32, usize),
    last_used_tick: u64,
}

#[derive(Clone, Debug)]
struct KernelReply {
    request_id: u32,
    status: u32,
    kernel_id: u32,
    is_batch_ok: bool,
    values: Vec<f32>,
}

impl KernelReply {
    fn ctl(request_id: u32, status: u32, error_code: u32) -> Self {
        Self {
            request_id,
            status,
            kernel_id: 0,
            is_batch_ok: false,
            values: vec![REPLY_KIND_CTL, error_code as f32, 0.0],
        }
    }

    fn batch_err(request_id: u32, status: u32, error_code: u32, kernel_id: u32) -> Self {
        Self {
            request_id,
            status,
            kernel_id,
            is_batch_ok: false,
            values: vec![REPLY_KIND_BATCH_ERR, error_code as f32, kernel_id as f32],
        }
    }

    fn batch_ok(
        request_id: u32,
        kernel_id: u32,
        binding_hash: u64,
        geom_hash: u64,
        item_count: u32,
        outputs: Vec<f32>,
    ) -> Self {
        let [kid_lo, kid_hi] = pack_u32(kernel_id);
        let bh = pack_u64(binding_hash);
        let gh = pack_u64(geom_hash);
        let [cnt_lo, cnt_hi] = pack_u32(item_count);
        let mut values = Vec::with_capacity(13 + outputs.len());
        values.push(REPLY_KIND_BATCH_OK);
        values.push(kid_lo);
        values.push(kid_hi);
        values.extend_from_slice(&bh);
        values.extend_from_slice(&gh);
        values.push(cnt_lo);
        values.push(cnt_hi);
        values.extend_from_slice(&outputs);
        Self { request_id, status: service_status::OK, kernel_id, is_batch_ok: true, values }
    }

    fn event_count(&self) -> usize {
        1 + self.values.len().div_ceil(3)
    }

    fn encode_events(&self, out: &mut Vec<[f32; EVENT_FLOATS]>) {
        let len = self.values.len().min(MAX_SERVICE_DATA);
        out.push([event_tags::SERVICE_ANSWER as f32, self.status as f32, 0.0, self.request_id as f32, len as f32]);
        for chunk in self.values[..len].chunks(3) {
            let at = |i: usize| chunk.get(i).copied().filter(|v| v.is_finite()).unwrap_or(0.0);
            out.push([event_tags::SERVICE_DATA as f32, 0.0, at(0), at(1), at(2)]);
        }
    }
}

struct BatchJob {
    epoch: u64,
    request_id: u32,
    kernel_id: u32,
    manifest: KernelManifest,
    binding_hash: u64,
    geom_hash: u64,
    item_count: u32,
    ctx: Option<(u64, Arc<[u32]>)>,
    items: Vec<u32>,
}

struct PoolSlot {
    pool: KernelPool,
    installed_revision: Option<u64>,
}

struct RunnerShared {
    active_epoch: AtomicU64,
    state: Mutex<RunnerState>,
    cvar: Condvar,
}

struct RunnerState {
    worker_engine: wasmtime::Engine,
    worker_module: GuestModule,
    worker_count: usize,
    has_deadlines: bool,
    module_generation: u64,
    pending_jobs: VecDeque<BatchJob>,
    active_jobs: usize,
    completed: Vec<(u64, KernelReply)>,
    shutdown: bool,
}

pub fn pack_u32(value: u32) -> [f32; 2] {
    [(value & 0xffff) as f32, (value >> 16) as f32]
}

pub fn unpack_u32(lo16: f32, hi16: f32) -> Option<u32> {
    let lo = whole(lo16).filter(|&v| v <= 0xffff)?;
    let hi = whole(hi16).filter(|&v| v <= 0xffff)?;
    Some(lo | (hi << 16))
}

pub fn pack_u64(value: u64) -> [f32; 4] {
    [
        (value & 0xffff) as f32,
        ((value >> 16) & 0xffff) as f32,
        ((value >> 32) & 0xffff) as f32,
        ((value >> 48) & 0xffff) as f32,
    ]
}

pub fn unpack_u64(words: &[f32]) -> Option<u64> {
    if words.len() < 4 {
        return None;
    }
    let mut out = 0u64;
    for (i, &lane) in words[..4].iter().enumerate() {
        let part = whole(lane).filter(|&v| v <= 0xffff)? as u64;
        out |= part << (i * 16);
    }
    Some(out)
}

pub fn fnv1a64_words(words: &[u32]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &word in words {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x00000100000001b3);
        }
    }
    hash
}

fn encode_cap_events(worker_count: usize, host_epoch: u64) -> [[f32; EVENT_FLOATS]; CAP_EVENTS] {
    let [e0, e1, e2, e3] = pack_u64(host_epoch);
    [
        [event_tags::SERVICE_ANSWER as f32, service_status::OK as f32, 0.0, 0.0, 6.0],
        [event_tags::SERVICE_DATA as f32, 0.0, MAX_SERVICE_DATA as f32, worker_count as f32, e0],
        [event_tags::SERVICE_DATA as f32, 0.0, e1, e2, e3],
    ]
}

/// Strips `strip` trailing event records from an encoded frame input packet BEFORE
/// recording it into `Recorded` / `Timeline` / trace.
pub fn strip_trailing_events(input: &mut Vec<f32>, strip: usize) {
    if strip == 0 {
        return;
    }
    let Some(count) = input.get(3).copied().and_then(whole) else {
        return;
    };
    let count = count as usize;
    if count < strip {
        return;
    }
    let end = FRAME_HEADER_FLOATS + count * EVENT_FLOATS;
    let start = end - strip * EVENT_FLOATS;
    if end <= input.len() {
        input.drain(start..end);
        input[3] = (count - strip) as f32;
    }
}

/// Partitions `requests` into canonical (`service != KERNEL`) and kernel (`service == KERNEL`)
/// without allocating when no kernel requests are present.
pub fn has_kernel_requests(requests: &[ServiceRequest]) -> bool {
    requests.iter().any(|req| req.service == service::KERNEL)
}

fn decode_ctx_chunk(schema: &WireSchema, offset: usize, chunk: &[f32], out: &mut Vec<u32>) -> Result<(), u32> {
    match schema.kind {
        WireSchemaKind::VarF32 => {
            for &lane in chunk {
                if !lane.is_finite() {
                    return Err(ERR_INVALID_REQUEST);
                }
                out.push(lane.to_bits());
            }
            Ok(())
        }
        WireSchemaKind::VarU32 => {
            for &lane in chunk {
                let value = whole(lane).ok_or(ERR_INVALID_REQUEST)?;
                out.push(value);
            }
            Ok(())
        }
        WireSchemaKind::FlatFixed => {
            for (i, &lane) in chunk.iter().enumerate() {
                let tag = schema.word_tags.get(offset + i).copied().ok_or(ERR_INVALID_REQUEST)?;
                let word = encode_scalar_lane(tag, lane)?;
                out.push(word);
            }
            Ok(())
        }
        WireSchemaKind::Unit => {
            if chunk.is_empty() {
                Ok(())
            } else {
                Err(ERR_INVALID_REQUEST)
            }
        }
    }
}

fn encode_scalar_lane(tag: ValueType, lane: f32) -> Result<u32, u32> {
    match tag {
        ValueType::F32 => {
            if !lane.is_finite() {
                return Err(ERR_INVALID_REQUEST);
            }
            Ok(lane.to_bits())
        }
        ValueType::U32 => whole(lane).ok_or(ERR_INVALID_REQUEST),
        ValueType::Bool => whole(lane).filter(|&b| b <= 1).ok_or(ERR_INVALID_REQUEST),
        _ => Err(ERR_INVALID_REQUEST),
    }
}

fn decode_items(schema: &WireSchema, item_count: usize, lanes: &[f32]) -> Result<Vec<u32>, u32> {
    let stride = schema.stride as usize;
    if schema.kind != WireSchemaKind::FlatFixed || stride == 0 || schema.word_tags.len() != stride {
        return Err(ERR_INVALID_REQUEST);
    }
    let expected = item_count.checked_mul(stride).ok_or(ERR_INVALID_REQUEST)?;
    if lanes.len() != expected {
        return Err(ERR_INVALID_REQUEST);
    }
    let mut out = Vec::with_capacity(expected);
    for item in lanes.chunks_exact(stride) {
        for (&tag, &lane) in schema.word_tags.iter().zip(item.iter()) {
            out.push(encode_scalar_lane(tag, lane)?);
        }
    }
    Ok(out)
}

fn encode_outputs(schema: &WireSchema, item_count: usize, raw: &[u32]) -> Result<Vec<f32>, u32> {
    match schema.kind {
        WireSchemaKind::VarF32 => {
            let mut pos = 0usize;
            let mut out = Vec::with_capacity(raw.len());
            for _ in 0..item_count {
                let len = *raw.get(pos).ok_or(ERR_EXECUTION_FAILED)? as usize;
                pos += 1;
                if len > 16_777_215 || pos + len > raw.len() {
                    return Err(ERR_EXECUTION_FAILED);
                }
                out.push(len as f32);
                for &word in &raw[pos..pos + len] {
                    let f = f32::from_bits(word);
                    if !f.is_finite() {
                        return Err(ERR_NONFINITE_OUTPUT);
                    }
                    out.push(f);
                }
                pos += len;
            }
            if pos != raw.len() {
                return Err(ERR_EXECUTION_FAILED);
            }
            if 13 + out.len() > MAX_SERVICE_DATA {
                return Err(ERR_OUTPUT_TOO_LARGE);
            }
            Ok(out)
        }
        WireSchemaKind::VarU32 => {
            let mut pos = 0usize;
            let mut out = Vec::with_capacity(raw.len());
            for _ in 0..item_count {
                let len = *raw.get(pos).ok_or(ERR_EXECUTION_FAILED)? as usize;
                pos += 1;
                if len > 16_777_215 || pos + len > raw.len() {
                    return Err(ERR_EXECUTION_FAILED);
                }
                out.push(len as f32);
                for &word in &raw[pos..pos + len] {
                    if word > 16_777_215 {
                        return Err(ERR_OUTPUT_TOO_LARGE);
                    }
                    out.push(word as f32);
                }
                pos += len;
            }
            if pos != raw.len() {
                return Err(ERR_EXECUTION_FAILED);
            }
            if 13 + out.len() > MAX_SERVICE_DATA {
                return Err(ERR_OUTPUT_TOO_LARGE);
            }
            Ok(out)
        }
        WireSchemaKind::FlatFixed => {
            let stride = schema.stride as usize;
            if stride == 0 || schema.word_tags.len() != stride || raw.len() != item_count * stride {
                return Err(ERR_EXECUTION_FAILED);
            }
            if 13 + raw.len() > MAX_SERVICE_DATA {
                return Err(ERR_OUTPUT_TOO_LARGE);
            }
            let mut out = Vec::with_capacity(raw.len());
            for row in raw.chunks_exact(stride) {
                for (&tag, &word) in schema.word_tags.iter().zip(row.iter()) {
                    match tag {
                        ValueType::F32 => {
                            let f = f32::from_bits(word);
                            if !f.is_finite() {
                                return Err(ERR_NONFINITE_OUTPUT);
                            }
                            out.push(f);
                        }
                        ValueType::U32 | ValueType::Bool => {
                            if word > 16_777_215 {
                                return Err(ERR_OUTPUT_TOO_LARGE);
                            }
                            out.push(word as f32);
                        }
                        _ => return Err(ERR_EXECUTION_FAILED),
                    }
                }
            }
            Ok(out)
        }
        WireSchemaKind::Unit => Ok(Vec::new()),
    }
}

fn runner_loop(shared: Arc<RunnerShared>) {
    let mut pools: HashMap<u32, PoolSlot> = HashMap::new();
    let mut pools_generation = 0u64;
    loop {
        let (job, engine, module, worker_count, has_deadlines, generation) = {
            let mut state = shared.state.lock().unwrap();
            loop {
                if state.shutdown {
                    return;
                }
                if let Some(job) = state.pending_jobs.pop_front() {
                    state.active_jobs += 1;
                    break (
                        job,
                        state.worker_engine.clone(),
                        state.worker_module.clone(),
                        state.worker_count,
                        state.has_deadlines,
                        state.module_generation,
                    );
                }
                state = shared.cvar.wait(state).unwrap();
            }
        };
        if pools_generation != generation {
            pools.clear();
            pools_generation = generation;
        }
        let reply = if job.epoch != shared.active_epoch.load(Ordering::Acquire) {
            None
        } else {
            Some(execute_batch_job(&mut pools, &engine, &module, worker_count, has_deadlines, &job))
        };
        let mut state = shared.state.lock().unwrap();
        state.active_jobs -= 1;
        if let Some(reply) = reply
            && job.epoch == shared.active_epoch.load(Ordering::Acquire)
        {
            state.completed.push((job.epoch, reply));
        }
        shared.cvar.notify_all();
    }
}

fn execute_batch_job(
    pools: &mut HashMap<u32, PoolSlot>,
    engine: &wasmtime::Engine,
    module: &GuestModule,
    worker_count: usize,
    has_deadlines: bool,
    job: &BatchJob,
) -> KernelReply {
    let slot = match pools.entry(job.kernel_id) {
        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
        std::collections::hash_map::Entry::Vacant(entry) => {
            let cfg = KernelConfig {
                workers: worker_count,
                deadline: if has_deadlines { Some(BOUNDED_BATCH_DEADLINE) } else { None },
                max_total_output_bytes: 2 << 20,
                max_physical_output_bytes: 4 << 20,
                ..KernelConfig::default()
            };
            match KernelPool::open(engine, module, &job.manifest.name, cfg) {
                Ok(pool) => entry.insert(PoolSlot { pool, installed_revision: None }),
                Err(_) => {
                    return KernelReply::batch_err(
                        job.request_id,
                        service_status::UNAVAILABLE,
                        ERR_UNAVAILABLE,
                        job.kernel_id,
                    );
                }
            }
        }
    };
    if let Some((revision, ref words)) = job.ctx
        && slot.installed_revision != Some(revision)
    {
        match slot.pool.set_context(revision, words) {
            Ok(()) => slot.installed_revision = Some(revision),
            Err(KernelError::Timeout) => {
                return KernelReply::batch_err(job.request_id, service_status::FAILED, ERR_TIMEOUT, job.kernel_id);
            }
            Err(_) => {
                return KernelReply::batch_err(
                    job.request_id,
                    service_status::FAILED,
                    ERR_EXECUTION_FAILED,
                    job.kernel_id,
                );
            }
        }
    }
    let raw = if job.manifest.variable_output() {
        slot.pool.run_variable(&job.items)
    } else {
        slot.pool.run_fixed(&job.items)
    };
    match raw {
        Ok(words) => match encode_outputs(&job.manifest.out_schema, job.item_count as usize, &words) {
            Ok(outputs) => KernelReply::batch_ok(
                job.request_id,
                job.kernel_id,
                job.binding_hash,
                job.geom_hash,
                job.item_count,
                outputs,
            ),
            Err(code) => KernelReply::batch_err(job.request_id, service_status::FAILED, code, job.kernel_id),
        },
        Err(KernelError::Timeout) => {
            KernelReply::batch_err(job.request_id, service_status::FAILED, ERR_TIMEOUT, job.kernel_id)
        }
        Err(KernelError::OutputBudgetExceeded { .. } | KernelError::SingleItemOutputOverflow { .. }) => {
            KernelReply::batch_err(job.request_id, service_status::FAILED, ERR_OUTPUT_TOO_LARGE, job.kernel_id)
        }
        Err(_) => KernelReply::batch_err(job.request_id, service_status::FAILED, ERR_EXECUTION_FAILED, job.kernel_id),
    }
}

fn default_worker_count() -> usize {
    if let Ok(raw) = std::env::var("PEN_KERNEL_WORKERS")
        && let Ok(parsed) = raw.trim().parse::<usize>()
    {
        return parsed.min(16);
    }
    std::thread::available_parallelism().map_or(2, |n| n.get().clamp(1, 8))
}

fn env_kernels_enabled() -> bool {
    std::env::var_os("PEN_KERNELS").is_none_or(|v| v != "0" && v != "false")
}

pub struct KernelService {
    kernels: Vec<KernelManifest>,
    transport_budget: usize,
    worker_count: usize,
    host_epoch: u64,
    pending_cap_announcement: bool,
    last_accepted_upload_seq: HashMap<u32, u32>,
    active_uploads: HashMap<u32, ActiveUpload>,
    lru: Vec<LruEntry>,
    next_revision: u64,
    lru_clock: u64,
    ready_replies: Vec<KernelReply>,
    runner_shared: Arc<RunnerShared>,
    runner_thread: Option<JoinHandle<()>>,
}

impl KernelService {
    /// Creates a `KernelService` iff `PEN_KERNELS != 0` and `worker_module` declares
    /// at least one `#kernel`.
    pub fn maybe_new(worker_engine: wasmtime::Engine, worker_module: GuestModule, has_deadlines: bool) -> Option<Self> {
        if !env_kernels_enabled() {
            return None;
        }
        let abi = worker_module.abi();
        if abi.kernels.is_empty() {
            return None;
        }
        let worker_count = default_worker_count();
        Some(Self::new_with_workers(worker_engine, worker_module, has_deadlines, worker_count))
    }

    pub fn new_with_workers(
        worker_engine: wasmtime::Engine,
        worker_module: GuestModule,
        has_deadlines: bool,
        worker_count: usize,
    ) -> Self {
        let abi = worker_module.abi();
        let kernels = abi.kernels.clone();
        let transport_budget = abi.transport_budget as usize;
        let host_epoch = 1u64;
        let runner_shared = Arc::new(RunnerShared {
            active_epoch: AtomicU64::new(host_epoch),
            state: Mutex::new(RunnerState {
                worker_engine,
                worker_module,
                worker_count,
                has_deadlines,
                module_generation: 1,
                pending_jobs: VecDeque::new(),
                active_jobs: 0,
                completed: Vec::new(),
                shutdown: false,
            }),
            cvar: Condvar::new(),
        });
        let runner_for_thread = runner_shared.clone();
        let runner_thread = std::thread::Builder::new()
            .name("pen-kernel-service".to_owned())
            .spawn(move || runner_loop(runner_for_thread))
            .ok();
        Self {
            kernels,
            transport_budget,
            worker_count,
            host_epoch,
            pending_cap_announcement: true,
            last_accepted_upload_seq: HashMap::new(),
            active_uploads: HashMap::new(),
            lru: Vec::new(),
            next_revision: 1,
            lru_clock: 0,
            ready_replies: Vec::new(),
            runner_shared,
            runner_thread,
        }
    }

    #[cfg(test)]
    pub fn host_epoch(&self) -> u64 {
        self.host_epoch
    }

    /// Replaces the worker module after a hot reload or packaged-worker upgrade.
    pub fn reload(&mut self, worker_engine: wasmtime::Engine, worker_module: GuestModule, has_deadlines: bool) {
        self.kernels = worker_module.abi().kernels.clone();
        self.transport_budget = worker_module.abi().transport_budget as usize;
        self.lru.clear();
        {
            let mut state = self.runner_shared.state.lock().unwrap();
            state.worker_engine = worker_engine;
            state.worker_module = worker_module;
            state.has_deadlines = has_deadlines;
            state.module_generation += 1;
            state.pending_jobs.clear();
            state.completed.clear();
        }
        self.fence();
    }

    /// Fences all ephemeral kernel state without touching canonical services:
    /// increments `host_epoch`, resets `last_accepted_upload_seq`, aborts partial uploads
    /// (preserving the byte-verified LRU cache!), discards in-flight/ready replies, and
    /// schedules a fresh live-only `CAP` announcement for the next live frame.
    pub fn fence(&mut self) {
        self.host_epoch = self.host_epoch.wrapping_add(1).max(1);
        self.runner_shared.active_epoch.store(self.host_epoch, Ordering::Release);
        self.last_accepted_upload_seq.clear();
        self.active_uploads.clear();
        self.ready_replies.clear();
        {
            let mut state = self.runner_shared.state.lock().unwrap();
            state.pending_jobs.clear();
            state.completed.clear();
        }
        self.pending_cap_announcement = !self.kernels.is_empty();
    }

    fn max_ctx_bytes(&self) -> usize {
        self.transport_budget.saturating_sub(4)
    }

    fn manifest(&self, kernel_id: u32) -> Option<&KernelManifest> {
        self.kernels.get(kernel_id as usize).filter(|k| k.id == kernel_id)
    }

    fn aggregate_upload_bytes_excluding(&self, kernel_id: u32) -> usize {
        self.active_uploads
            .iter()
            .filter(|&(&id, _)| id != kernel_id)
            .map(|(_, up)| up.total_words.saturating_mul(4))
            .sum()
    }

    fn touch_lru(&mut self, idx: usize) {
        self.lru_clock += 1;
        self.lru[idx].last_used_tick = self.lru_clock;
    }

    fn insert_lru(
        &mut self,
        key: CtxKey,
        words: Arc<[u32]>,
        committed_upload_key: (u64, u32, usize),
    ) -> Result<(), u32> {
        let new_bytes = words.len().saturating_mul(4);
        if new_bytes > MAX_LRU_TOTAL_BYTES {
            return Err(ERR_UPLOAD_TOO_LARGE);
        }
        if let Some(pos) = self.lru.iter().position(|entry| entry.key == key) {
            if self.lru[pos].words.as_ref() != words.as_ref() {
                return Err(ERR_HASH_MISMATCH);
            }
            self.lru[pos].committed_upload_key = committed_upload_key;
            self.touch_lru(pos);
            return Ok(());
        }
        while self.lru.iter().filter(|e| e.key.kernel_id == key.kernel_id).count() >= MAX_LRU_ENTRIES_PER_KERNEL {
            let victim = self
                .lru
                .iter()
                .enumerate()
                .filter(|(_, e)| e.key.kernel_id == key.kernel_id)
                .min_by_key(|(_, e)| e.last_used_tick)
                .map(|(i, _)| i);
            if let Some(idx) = victim {
                self.lru.remove(idx);
            } else {
                break;
            }
        }
        while self.lru.iter().map(|e| e.words.len() * 4).sum::<usize>() + new_bytes > MAX_LRU_TOTAL_BYTES {
            let victim = self.lru.iter().enumerate().min_by_key(|(_, e)| e.last_used_tick).map(|(i, _)| i);
            if let Some(idx) = victim {
                self.lru.remove(idx);
            } else {
                break;
            }
        }
        self.lru_clock += 1;
        let revision = self.next_revision;
        self.next_revision += 1;
        self.lru.push(LruEntry { key, words, revision, committed_upload_key, last_used_tick: self.lru_clock });
        Ok(())
    }

    /// Processes a live frame's `service::KERNEL` requests: `CTX_*` synchronously and
    /// `RUN_BATCH` dispatched onto the background runner thread (overlapping GPU render).
    pub fn submit(&mut self, requests: &[ServiceRequest]) {
        let mut batch_jobs = Vec::new();
        for req in requests {
            if req.service != service::KERNEL {
                continue;
            }
            match req.operation {
                kernel_op::CTX_BEGIN => self.handle_ctx_begin(req),
                kernel_op::CTX_CHUNK => self.handle_ctx_chunk(req),
                kernel_op::CTX_COMMIT => self.handle_ctx_commit(req),
                kernel_op::RUN_BATCH => {
                    if let Some(job) = self.prepare_run_batch(req) {
                        batch_jobs.push(job);
                    }
                }
                _ => {
                    self.ready_replies.push(KernelReply::ctl(req.request, service_status::UNKNOWN, ERR_UNKNOWN_OP));
                }
            }
        }
        if !batch_jobs.is_empty() {
            let mut state = self.runner_shared.state.lock().unwrap();
            state.pending_jobs.extend(batch_jobs);
            self.runner_shared.cvar.notify_all();
        }
    }

    fn handle_ctx_begin(&mut self, req: &ServiceRequest) {
        let Some(kernel_id) = whole(req.a) else {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
            return;
        };
        let Some(upload_seq) = unpack_u32(req.b, req.d) else {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
            return;
        };
        let Some(total_chunks) = whole(req.c) else {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
            return;
        };
        if req.values.len() != 14 {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
            return;
        }
        let (Some(req_epoch), Some(geom_hash), Some(binding_hash), Some(total_words_u32)) = (
            unpack_u64(&req.values[0..4]),
            unpack_u64(&req.values[4..8]),
            unpack_u64(&req.values[8..12]),
            unpack_u32(req.values[12], req.values[13]),
        ) else {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
            return;
        };
        if req_epoch != self.host_epoch {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_STALE_EPOCH));
            return;
        }
        let Some(manifest) = self.manifest(kernel_id) else {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_UNKNOWN_KERNEL));
            return;
        };
        if !manifest.has_context() {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
            return;
        }
        if manifest.binding_hash != binding_hash {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_BINDING_MISMATCH));
            return;
        }
        let last_seq = self.last_accepted_upload_seq.get(&kernel_id).copied().unwrap_or(0);
        if upload_seq == 0 || upload_seq <= last_seq {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_UPLOAD_SEQUENCE));
            return;
        }
        let total_words = total_words_u32 as usize;
        let total_bytes = total_words.saturating_mul(4);
        if total_bytes > self.max_ctx_bytes() {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_UPLOAD_TOO_LARGE));
            return;
        }
        let max_chunks = u32::try_from((self.max_ctx_bytes() / 4).div_ceil(16_380).max(1)).unwrap_or(u32::MAX);
        if (total_words > 0 && total_chunks == 0) || total_chunks > max_chunks {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
            return;
        }
        if self.aggregate_upload_bytes_excluding(kernel_id) + total_bytes > MAX_AGGREGATE_UPLOAD_BYTES {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_UPLOAD_TOO_LARGE));
            return;
        }
        self.last_accepted_upload_seq.insert(kernel_id, upload_seq);
        self.active_uploads.insert(
            kernel_id,
            ActiveUpload {
                host_epoch: req_epoch,
                upload_seq,
                binding_hash,
                geom_hash,
                total_chunks,
                total_words,
                next_chunk: 0,
                words: Vec::with_capacity(total_words),
            },
        );
        self.ready_replies.push(KernelReply::ctl(req.request, service_status::OK, ERR_NONE));
    }

    fn handle_ctx_chunk(&mut self, req: &ServiceRequest) {
        let (Some(kernel_id), Some(upload_seq), Some(chunk_index)) =
            (whole(req.a), unpack_u32(req.b, req.d), whole(req.c))
        else {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
            return;
        };
        if req.values.len() < 4 || req.values.len() > MAX_SERVICE_DATA {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
            return;
        }
        let Some(req_epoch) = unpack_u64(&req.values[0..4]) else {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
            return;
        };
        if req_epoch != self.host_epoch {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_STALE_EPOCH));
            return;
        }
        let Some(ctx_schema) = self.manifest(kernel_id).and_then(|m| m.ctx_schema.clone()) else {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_UNKNOWN_KERNEL));
            return;
        };
        let payload = &req.values[4..];
        let Some(active) = self.active_uploads.get_mut(&kernel_id) else {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
            return;
        };
        if active.host_epoch != req_epoch
            || active.upload_seq != upload_seq
            || active.next_chunk != chunk_index
            || chunk_index >= active.total_chunks
            || active.words.len().saturating_add(payload.len()) > active.total_words
        {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
            return;
        }
        let offset = active.words.len();
        if let Err(code) = decode_ctx_chunk(&ctx_schema, offset, payload, &mut active.words) {
            self.active_uploads.remove(&kernel_id);
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, code));
            return;
        }
        active.next_chunk += 1;
        self.ready_replies.push(KernelReply::ctl(req.request, service_status::OK, ERR_NONE));
    }

    fn handle_ctx_commit(&mut self, req: &ServiceRequest) {
        let (Some(kernel_id), Some(upload_seq), Some(total_chunks)) =
            (whole(req.a), unpack_u32(req.b, req.d), whole(req.c))
        else {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
            return;
        };
        if req.values.len() != 14 {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
            return;
        }
        let (Some(req_epoch), Some(geom_hash), Some(binding_hash), Some(expected_words_u32)) = (
            unpack_u64(&req.values[0..4]),
            unpack_u64(&req.values[4..8]),
            unpack_u64(&req.values[8..12]),
            unpack_u32(req.values[12], req.values[13]),
        ) else {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
            return;
        };
        if req_epoch != self.host_epoch {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_STALE_EPOCH));
            return;
        }
        let Some(manifest) = self.manifest(kernel_id) else {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_UNKNOWN_KERNEL));
            return;
        };
        if manifest.binding_hash != binding_hash {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_BINDING_MISMATCH));
            return;
        }
        let expected_words = expected_words_u32 as usize;
        let key = CtxKey { kernel_id, binding_hash, geom_hash };
        let commit_key = (req_epoch, upload_seq, expected_words);
        let has_matching_active = self
            .active_uploads
            .get(&kernel_id)
            .is_some_and(|up| up.host_epoch == req_epoch && up.upload_seq == upload_seq);
        if has_matching_active {
            let active = self.active_uploads.remove(&kernel_id).unwrap();
            if active.total_chunks != total_chunks
                || active.next_chunk != total_chunks
                || active.total_words != expected_words
                || active.words.len() != expected_words
                || active.geom_hash != geom_hash
                || active.binding_hash != binding_hash
            {
                self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
                return;
            }
            let actual_hash = fnv1a64_words(&active.words);
            if actual_hash != geom_hash {
                self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_HASH_MISMATCH));
                return;
            }
            let words: Arc<[u32]> = Arc::from(active.words);
            match self.insert_lru(key, words, commit_key) {
                Ok(()) => self.ready_replies.push(KernelReply::ctl(req.request, service_status::OK, ERR_NONE)),
                Err(code) => self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, code)),
            }
            return;
        }
        // Lost-ack retry: idempotent OK iff the exact entry is STILL PRESENT in the live LRU!
        if let Some(pos) =
            self.lru.iter().position(|entry| entry.key == key && entry.committed_upload_key == commit_key)
        {
            self.touch_lru(pos);
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::OK, ERR_NONE));
        } else {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_CTX_EVICTED));
        }
    }

    fn prepare_run_batch(&mut self, req: &ServiceRequest) -> Option<BatchJob> {
        let (Some(kernel_id), Some(item_count)) = (whole(req.a), whole(req.c)) else {
            self.ready_replies.push(KernelReply::batch_err(
                req.request,
                service_status::FAILED,
                ERR_INVALID_REQUEST,
                0,
            ));
            return None;
        };
        if req.values.len() < 8 {
            self.ready_replies.push(KernelReply::batch_err(
                req.request,
                service_status::FAILED,
                ERR_INVALID_REQUEST,
                kernel_id,
            ));
            return None;
        }
        let (Some(binding_hash), Some(geom_hash)) = (unpack_u64(&req.values[0..4]), unpack_u64(&req.values[4..8]))
        else {
            self.ready_replies.push(KernelReply::batch_err(
                req.request,
                service_status::FAILED,
                ERR_INVALID_REQUEST,
                kernel_id,
            ));
            return None;
        };
        let Some(manifest) = self.manifest(kernel_id).cloned() else {
            self.ready_replies.push(KernelReply::batch_err(
                req.request,
                service_status::FAILED,
                ERR_UNKNOWN_KERNEL,
                kernel_id,
            ));
            return None;
        };
        if manifest.binding_hash != binding_hash {
            self.ready_replies.push(KernelReply::batch_err(
                req.request,
                service_status::FAILED,
                ERR_BINDING_MISMATCH,
                kernel_id,
            ));
            return None;
        }
        let items = match decode_items(&manifest.in_schema, item_count as usize, &req.values[8..]) {
            Ok(items) => items,
            Err(code) => {
                self.ready_replies.push(KernelReply::batch_err(req.request, service_status::FAILED, code, kernel_id));
                return None;
            }
        };
        let ctx = if manifest.has_context() {
            let key = CtxKey { kernel_id, binding_hash, geom_hash };
            let Some(pos) = self.lru.iter().position(|e| e.key == key) else {
                self.ready_replies.push(KernelReply::batch_err(
                    req.request,
                    service_status::FAILED,
                    ERR_CTX_MISSING,
                    kernel_id,
                ));
                return None;
            };
            self.touch_lru(pos);
            let entry = &self.lru[pos];
            Some((entry.revision, entry.words.clone()))
        } else {
            None
        };
        Some(BatchJob {
            epoch: self.host_epoch,
            request_id: req.request,
            kernel_id,
            manifest,
            binding_hash,
            geom_hash,
            item_count,
            ctx,
            items,
        })
    }

    fn join_in_flight(&mut self) {
        let mut state = self.runner_shared.state.lock().unwrap();
        while !state.pending_jobs.is_empty() || state.active_jobs > 0 {
            state = self.runner_shared.cvar.wait(state).unwrap();
        }
        let epoch = self.host_epoch;
        for (reply_epoch, reply) in state.completed.drain(..) {
            if reply_epoch == epoch {
                self.ready_replies.push(reply);
            }
        }
    }

    /// Joins completed current-epoch `RUN_BATCH` replies, atomically splices bounded `request == 0`
    /// replies into `input` AFTER `mods.frame`, and appends live-only `CAP` at the tail when due.
    /// Returns the number of trailing live-only `CAP` events (`0` or `3`) that MUST be stripped
    /// via [`strip_trailing_events`] before constructing `Recorded`.
    pub fn splice_live_events(&mut self, input: &mut Vec<f32>) -> usize {
        self.join_in_flight();
        let Some(base_count) = input.get(3).copied().and_then(whole).map(|v| v as usize) else {
            self.ready_replies.clear();
            return 0;
        };
        let end = FRAME_HEADER_FLOATS + base_count * EVENT_FLOATS;
        if end > input.len() {
            self.ready_replies.clear();
            return 0;
        }
        let available = MAX_EVENTS.saturating_sub(base_count);
        let cap_reserve = if self.pending_cap_announcement && available >= CAP_EVENTS { CAP_EVENTS } else { 0 };
        let reply_budget = available.saturating_sub(cap_reserve).min(MAX_KERNEL_REPLY_EVENTS);
        let mut events: Vec<[f32; EVENT_FLOATS]> = Vec::new();
        let mut used = 0usize;
        for reply in self.ready_replies.drain(..) {
            let needed = reply.event_count();
            if used + needed <= reply_budget {
                reply.encode_events(&mut events);
                used += needed;
            } else if reply.is_batch_ok && used + 2 <= reply_budget {
                let downgraded =
                    KernelReply::batch_err(reply.request_id, service_status::FAILED, ERR_EVENT_BUDGET, reply.kernel_id);
                downgraded.encode_events(&mut events);
                used += 2;
            }
        }
        if cap_reserve == CAP_EVENTS {
            events.extend_from_slice(&encode_cap_events(self.worker_count, self.host_epoch));
            self.pending_cap_announcement = false;
        }
        if !events.is_empty() {
            input.splice(end..end, events.iter().flatten().copied());
            input[3] = (base_count + events.len()) as f32;
        }
        cap_reserve
    }
}

impl Drop for KernelService {
    fn drop(&mut self) {
        self.runner_shared.active_epoch.fetch_add(1, Ordering::Release);
        {
            let mut state = self.runner_shared.state.lock().unwrap();
            state.shutdown = true;
            state.pending_jobs.clear();
            self.runner_shared.cvar.notify_all();
        }
        if let Some(handle) = self.runner_thread.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(all(test, feature = "compiler"))]
#[path = "kernels_law6_tests.rs"]
mod kernels_law6_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use pen_protocol::boundary::encode_frame;
    use pen_protocol::{Frame, Viewport};

    fn dummy_service_from_blot(source: &str) -> KernelService {
        let blot_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../blot2");
        let blotc = blot_root.join("compiler-goose/build/blotc-goose");
        let prelude = blot_root.join("std/prelude.blot");
        let tmp = std::env::temp_dir().join(format!("pen-kernel-svc-test-{}.blot", std::process::id()));
        let wasm_path = tmp.with_extension("wasm");
        std::fs::write(&tmp, source).unwrap();
        let status = std::process::Command::new(&blotc)
            .args(["build", tmp.to_str().unwrap(), wasm_path.to_str().unwrap(), "--prelude", prelude.to_str().unwrap()])
            .status()
            .unwrap();
        assert!(status.success(), "blotc failed");
        let wasm = std::fs::read(&wasm_path).unwrap();
        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(&wasm_path);
        let engine = blot_host::engine().unwrap();
        let module = GuestModule::new(&engine, &wasm, None).unwrap();
        KernelService::new_with_workers(engine, module, false, 2)
    }

    fn req(id: u32, op: u32, a: f32, b: f32, c: f32, d: f32, values: Vec<f32>) -> ServiceRequest {
        ServiceRequest {
            request: id,
            service: service::KERNEL,
            operation: op,
            a,
            b,
            c,
            d,
            value_count: u32::try_from(values.len()).unwrap(),
            values,
        }
    }

    #[test]
    fn pack_unpack_round_trips_full_u32_and_u64_ranges() {
        for seq in [1u32, 0xffff, 0x1_0000, 16_777_215, 16_777_216, 0xdead_beef, u32::MAX] {
            let [lo, hi] = pack_u32(seq);
            assert_eq!(unpack_u32(lo, hi), Some(seq));
        }
        for hash in [0u64, 1, 0xcbf29ce484222325, 0x1234_5678_9abc_def0, u64::MAX] {
            let words = pack_u64(hash);
            assert_eq!(unpack_u64(&words), Some(hash));
        }
    }

    #[cfg(feature = "compiler")]
    #[test]
    fn context_upload_lru_retry_fence_and_cap_strip_contract() {
        let src = r#"
func prep (ctx: #[F32]) => #[ctx[0], ctx[1]]
func run_item (ctx: #[F32]) -> (x: F32) => #[x + ctx[0], x + ctx[1]]
#kernel const scale_ctx: Kernel [#[F32], F32, #[F32]] = { prepare: prep, run: run_item }
"#;
        let mut svc = dummy_service_from_blot(src);
        let manifest = svc.kernels[0].clone();
        let frame = Frame { timestamp: 0.0, viewport: Viewport { width: 640, height: 480 }, events: vec![] };
        let mut input0 = encode_frame(&frame, 1.0 / 60.0, &[]).unwrap();
        let cap = svc.splice_live_events(&mut input0);
        assert_eq!(cap, CAP_EVENTS);
        assert_eq!(input0[3], 3.0);
        assert_eq!(input0[FRAME_HEADER_FLOATS + 2], 0.0, "CAP uses request == 0");
        assert_eq!(input0[FRAME_HEADER_FLOATS + 3], 0.0, "CAP uses value == 0.0");
        strip_trailing_events(&mut input0, cap);
        assert_eq!(input0[3], 0.0, "live-only CAP stripped before Recorded");
        assert_eq!(input0.len(), FRAME_HEADER_FLOATS);

        // Upload context under high 32-bit upload_seq (> 16,777,215!)
        let ctx_floats = vec![10.0f32, 20.0f32];
        let ctx_words: Vec<u32> = ctx_floats.iter().map(|f| f.to_bits()).collect();
        let geom_hash = fnv1a64_words(&ctx_words);
        let seq = 0x0102_0304u32;
        let [seq_lo, seq_hi] = pack_u32(seq);
        let [w_lo, w_hi] = pack_u32(u32::try_from(ctx_floats.len()).unwrap());
        let epoch_w = pack_u64(svc.host_epoch());
        let geom_w = pack_u64(geom_hash);
        let bind_w = pack_u64(manifest.binding_hash);
        let mut begin_vals = Vec::new();
        begin_vals.extend_from_slice(&epoch_w);
        begin_vals.extend_from_slice(&geom_w);
        begin_vals.extend_from_slice(&bind_w);
        begin_vals.extend_from_slice(&[w_lo, w_hi]);
        let mut chunk_vals = epoch_w.to_vec();
        chunk_vals.extend_from_slice(&ctx_floats);
        svc.submit(&[
            req(1, kernel_op::CTX_BEGIN, 0.0, seq_lo, 1.0, seq_hi, begin_vals.clone()),
            req(2, kernel_op::CTX_CHUNK, 0.0, seq_lo, 0.0, seq_hi, chunk_vals),
            req(3, kernel_op::CTX_COMMIT, 0.0, seq_lo, 1.0, seq_hi, begin_vals.clone()),
            // Idempotent lost-ack CTX_COMMIT retry while live in LRU:
            req(4, kernel_op::CTX_COMMIT, 0.0, seq_lo, 1.0, seq_hi, begin_vals),
        ]);
        let mut input1 = encode_frame(&frame, 1.0 / 60.0, &[]).unwrap();
        assert_eq!(svc.splice_live_events(&mut input1), 0);
        // 4 CTL replies * 2 events each = 8 events
        assert_eq!(input1[3], 8.0);
        for i in 0..4 {
            let base = FRAME_HEADER_FLOATS + i * 2 * EVENT_FLOATS;
            assert_eq!(input1[base + 1], service_status::OK as f32, "reply {i} status OK");
            assert_eq!(input1[base + 2], 0.0, "request == 0");
            assert_eq!(input1[base + 3], (i + 1) as f32, "value == kernel_req_id");
            assert_eq!(input1[base + EVENT_FLOATS + 2], REPLY_KIND_CTL);
            assert_eq!(input1[base + EVENT_FLOATS + 3], ERR_NONE as f32);
        }

        // RUN_BATCH with 2 items -> [1.0, 2.0] => [[11.0, 21.0], [12.0, 22.0]]
        let mut run_vals = Vec::new();
        run_vals.extend_from_slice(&bind_w);
        run_vals.extend_from_slice(&geom_w);
        run_vals.extend_from_slice(&[1.0, 2.0]);
        svc.submit(&[req(5, kernel_op::RUN_BATCH, 0.0, 0.0, 2.0, 0.0, run_vals)]);
        let mut input2 = encode_frame(&frame, 1.0 / 60.0, &[]).unwrap();
        assert_eq!(svc.splice_live_events(&mut input2), 0);
        // Header (13) + row0 (1+2=3) + row1 (1+2=3) = 19 floats => 1 + ceil(19/3) = 8 events
        assert_eq!(input2[3], 8.0);
        assert_eq!(input2[FRAME_HEADER_FLOATS + 1], service_status::OK as f32);
        assert_eq!(input2[FRAME_HEADER_FLOATS + 3], 5.0);
        assert_eq!(input2[FRAME_HEADER_FLOATS + 4], 19.0);
        let mut unpacked = Vec::new();
        for ev in 1..8 {
            let off = FRAME_HEADER_FLOATS + ev * EVENT_FLOATS;
            unpacked.extend_from_slice(&input2[off + 2..off + 5]);
        }
        unpacked.truncate(19);
        assert_eq!(unpacked[0], REPLY_KIND_BATCH_OK);
        assert_eq!(unpack_u64(&unpacked[3..7]), Some(manifest.binding_hash));
        assert_eq!(unpack_u64(&unpacked[7..11]), Some(geom_hash));
        assert_eq!(unpack_u32(unpacked[11], unpacked[12]), Some(2));
        assert_eq!(&unpacked[13..], &[2.0, 11.0, 21.0, 2.0, 12.0, 22.0]);

        // Fence bumps epoch, re-arms CAP, rejects stale epoch CTX_BEGIN while keeping LRU!
        svc.fence();
        assert_eq!(svc.host_epoch(), 2);
        let mut stale_vals = Vec::new();
        stale_vals.extend_from_slice(&epoch_w); // old epoch 1
        stale_vals.extend_from_slice(&geom_w);
        stale_vals.extend_from_slice(&bind_w);
        stale_vals.extend_from_slice(&[w_lo, w_hi]);
        svc.submit(&[req(6, kernel_op::CTX_BEGIN, 0.0, 1.0, 1.0, 0.0, stale_vals)]);
        let mut input3 = encode_frame(&frame, 1.0 / 60.0, &[]).unwrap();
        let cap3 = svc.splice_live_events(&mut input3);
        assert_eq!(cap3, CAP_EVENTS);
        assert_eq!(input3[3], 5.0); // 2 CTL + 3 CAP
        assert_eq!(input3[FRAME_HEADER_FLOATS + 1], service_status::FAILED as f32);
        assert_eq!(input3[FRAME_HEADER_FLOATS + EVENT_FLOATS + 3], ERR_STALE_EPOCH as f32);
        strip_trailing_events(&mut input3, cap3);
        assert_eq!(input3[3], 2.0);
    }
}
