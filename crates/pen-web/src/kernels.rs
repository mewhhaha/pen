//! Browser Pen `service::KERNEL` (code 15) host service (`blot2/tasks/CONCURRENCY_VNEXT.md` §5.3).
//!
//! Mirrors native `pen/src/kernels.rs` over `./kernels.js`'s `BrowserKernelPool`:
//! - Unambiguous `request == 0` host→guest framing (`CAP`, `CTL`, `BATCH_ERR`, `BATCH_OK`)
//! - Live-only unrecorded `CAP` appended at the tail of `guest_input` and stripped before `Recorded`
//! - Lossless 32-bit `(host_epoch, upload_seq_lo16/hi16)` byte-verified context uploads & bounded LRU
//! - Async `join_live_kernels_async` called in `Browser::step` AFTER `timeline.prepare`

use std::collections::HashMap;

use js_sys::{Object, Promise, Reflect, Uint8Array, Uint32Array};
use pen_protocol::boundary::{EVENT_FLOATS, FRAME_HEADER_FLOATS, MAX_EVENTS, MAX_SERVICE_DATA, event_tags, whole};
use pen_protocol::{ServiceRequest, kernel_op, service, service_status};
use pen_runtime::abi::{KernelManifest, ValueType, WireSchema, WireSchemaKind};
use wasm_bindgen::{JsCast, prelude::*};
use wasm_bindgen_futures::{JsFuture, spawn_local};

#[wasm_bindgen(raw_module = "./kernels.js")]
extern "C" {
    #[wasm_bindgen(catch, js_name = compileKernelModule)]
    fn compile_kernel_module(bytes: &Uint8Array) -> Result<JsValue, JsValue>;

    type BrowserKernelPool;

    #[wasm_bindgen(static_method_of = BrowserKernelPool, catch)]
    fn open(source: &JsValue, name: &str, config: &JsValue) -> Result<Promise, JsValue>;

    #[wasm_bindgen(method, catch, js_name = setContext)]
    fn set_context(this: &BrowserKernelPool, revision: f64, bytes: &Uint8Array) -> Result<Promise, JsValue>;

    #[wasm_bindgen(method, catch, js_name = runFixed)]
    fn run_fixed(this: &BrowserKernelPool, items: &Uint32Array) -> Result<Promise, JsValue>;

    #[wasm_bindgen(method, catch, js_name = runVariable)]
    fn run_variable(this: &BrowserKernelPool, items: &Uint32Array) -> Result<Promise, JsValue>;

    #[wasm_bindgen(method)]
    fn close(this: &BrowserKernelPool) -> Promise;
}

const REPLY_KIND_CTL: f32 = 1.0;
const REPLY_KIND_BATCH_ERR: f32 = 2.0;
const REPLY_KIND_BATCH_OK: f32 = 3.0;

const ERR_NONE: u32 = 0;
const ERR_UNKNOWN_OP: u32 = 1;
const ERR_UNKNOWN_KERNEL: u32 = 2;
const ERR_STALE_EPOCH: u32 = 3;
const ERR_INVALID_REQUEST: u32 = 4;
const ERR_UPLOAD_SEQUENCE: u32 = 5;
const ERR_UPLOAD_TOO_LARGE: u32 = 6;
const ERR_HASH_MISMATCH: u32 = 7;
const ERR_BINDING_MISMATCH: u32 = 8;
const ERR_CTX_EVICTED: u32 = 9;
const ERR_CTX_MISSING: u32 = 10;
const ERR_EXECUTION_FAILED: u32 = 11;
const ERR_OUTPUT_TOO_LARGE: u32 = 12;
const ERR_NONFINITE_OUTPUT: u32 = 13;
const ERR_TIMEOUT: u32 = 14;
const ERR_EVENT_BUDGET: u32 = 15;
const ERR_UNAVAILABLE: u32 = 16;

const MAX_AGGREGATE_UPLOAD_BYTES: usize = 4 * 1024 * 1024;
const MAX_LRU_ENTRIES_PER_KERNEL: usize = 2;
const MAX_LRU_TOTAL_BYTES: usize = 4 * 1024 * 1024;
const MAX_KERNEL_REPLY_EVENTS: usize = 65_536;
pub const CAP_EVENTS: usize = 3;
const WEB_BATCH_DEADLINE_MS: f64 = 500.0;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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
    words: Vec<u32>,
    revision: u64,
    committed_upload_key: (u64, u32, usize),
    last_used_tick: u64,
}

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
        out.push([
            event_tags::SERVICE_ANSWER as f32,
            self.status as f32,
            0.0,
            self.request_id as f32,
            len as f32,
        ]);
        for chunk in self.values[..len].chunks(3) {
            let at = |i: usize| chunk.get(i).copied().filter(|v| v.is_finite()).unwrap_or(0.0);
            out.push([event_tags::SERVICE_DATA as f32, 0.0, at(0), at(1), at(2)]);
        }
    }
}

struct PendingBatch {
    epoch: u64,
    request_id: u32,
    kernel_id: u32,
    manifest: KernelManifest,
    binding_hash: u64,
    geom_hash: u64,
    item_count: u32,
    ctx: Option<(u64, Vec<u32>)>,
    items: Vec<u32>,
}

struct WebPoolSlot {
    pool: BrowserKernelPool,
    installed_revision: Option<u64>,
}

fn pack_u32(value: u32) -> [f32; 2] {
    [(value & 0xffff) as f32, (value >> 16) as f32]
}

fn unpack_u32(lo16: f32, hi16: f32) -> Option<u32> {
    let lo = whole(lo16).filter(|&v| v <= 0xffff)?;
    let hi = whole(hi16).filter(|&v| v <= 0xffff)?;
    Some(lo | (hi << 16))
}

fn pack_u64(value: u64) -> [f32; 4] {
    [
        (value & 0xffff) as f32,
        ((value >> 16) & 0xffff) as f32,
        ((value >> 32) & 0xffff) as f32,
        ((value >> 48) & 0xffff) as f32,
    ]
}

fn unpack_u64(words: &[f32]) -> Option<u64> {
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

fn fnv1a64_words(words: &[u32]) -> u64 {
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

pub fn has_kernel_requests(requests: &[ServiceRequest]) -> bool {
    requests.iter().any(|req| req.service == service::KERNEL)
}

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
                out.push(whole(lane).ok_or(ERR_INVALID_REQUEST)?);
            }
            Ok(())
        }
        WireSchemaKind::FlatFixed => {
            for (i, &lane) in chunk.iter().enumerate() {
                let tag = schema.word_tags.get(offset + i).copied().ok_or(ERR_INVALID_REQUEST)?;
                out.push(encode_scalar_lane(tag, lane)?);
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

fn words_to_u8(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for &w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

fn js_error_code(err: &JsValue) -> Option<String> {
    Reflect::get(err, &"code".into()).ok().and_then(|v| v.as_string())
}

fn browser_worker_count() -> usize {
    web_sys::window()
        .map(|w| w.navigator().hardware_concurrency() as usize)
        .filter(|&n| n >= 1)
        .map_or(2, |n| n.clamp(1, 4))
}

pub struct WebKernelService {
    receipt: JsValue,
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
    pending_batches: Vec<PendingBatch>,
    pools: HashMap<u32, WebPoolSlot>,
}

impl WebKernelService {
    pub fn maybe_new(wasm_bytes: &[u8]) -> Option<Self> {
        let abi = pen_runtime::abi::read_abi(wasm_bytes).ok()?;
        if abi.kernels.is_empty() {
            return None;
        }
        let receipt = compile_kernel_module(&Uint8Array::from(wasm_bytes)).ok()?;
        Some(Self {
            receipt,
            kernels: abi.kernels,
            transport_budget: abi.transport_budget as usize,
            worker_count: browser_worker_count(),
            host_epoch: 1,
            pending_cap_announcement: true,
            last_accepted_upload_seq: HashMap::new(),
            active_uploads: HashMap::new(),
            lru: Vec::new(),
            next_revision: 1,
            lru_clock: 0,
            ready_replies: Vec::new(),
            pending_batches: Vec::new(),
            pools: HashMap::new(),
        })
    }

    pub fn reload(&mut self, wasm_bytes: &[u8]) {
        self.close_all_pools();
        self.lru.clear();
        if let Ok(abi) = pen_runtime::abi::read_abi(wasm_bytes)
            && let Ok(receipt) = compile_kernel_module(&Uint8Array::from(wasm_bytes))
        {
            self.receipt = receipt;
            self.kernels = abi.kernels;
            self.transport_budget = abi.transport_budget as usize;
        } else {
            self.kernels.clear();
        }
        self.fence();
    }

    fn close_all_pools(&mut self) {
        for (_, slot) in self.pools.drain() {
            let promise = slot.pool.close();
            spawn_local(async move {
                let _ = JsFuture::from(promise).await;
            });
        }
    }

    pub fn fence(&mut self) {
        self.host_epoch = self.host_epoch.wrapping_add(1).max(1);
        self.last_accepted_upload_seq.clear();
        self.active_uploads.clear();
        self.ready_replies.clear();
        self.pending_batches.clear();
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
        words: Vec<u32>,
        committed_upload_key: (u64, u32, usize),
    ) -> Result<(), u32> {
        let new_bytes = words.len().saturating_mul(4);
        if new_bytes > MAX_LRU_TOTAL_BYTES {
            return Err(ERR_UPLOAD_TOO_LARGE);
        }
        if let Some(pos) = self.lru.iter().position(|entry| entry.key == key) {
            if self.lru[pos].words != words {
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
        self.lru.push(LruEntry {
            key,
            words,
            revision,
            committed_upload_key,
            last_used_tick: self.lru_clock,
        });
        Ok(())
    }

    pub fn submit(&mut self, requests: &[ServiceRequest]) {
        for req in requests {
            if req.service != service::KERNEL {
                continue;
            }
            match req.operation {
                kernel_op::CTX_BEGIN => self.handle_ctx_begin(req),
                kernel_op::CTX_CHUNK => self.handle_ctx_chunk(req),
                kernel_op::CTX_COMMIT => self.handle_ctx_commit(req),
                kernel_op::RUN_BATCH => self.handle_run_batch(req),
                _ => self.ready_replies.push(KernelReply::ctl(req.request, service_status::UNKNOWN, ERR_UNKNOWN_OP)),
            }
        }
    }

    fn handle_ctx_begin(&mut self, req: &ServiceRequest) {
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
        if total_bytes > self.max_ctx_bytes()
            || self.aggregate_upload_bytes_excluding(kernel_id) + total_bytes > MAX_AGGREGATE_UPLOAD_BYTES
        {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_UPLOAD_TOO_LARGE));
            return;
        }
        let max_chunks = (self.max_ctx_bytes() / 4).div_ceil(16_380).max(1) as u32;
        if (total_words > 0 && total_chunks == 0) || total_chunks > max_chunks {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_INVALID_REQUEST));
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
        let has_matching = self
            .active_uploads
            .get(&kernel_id)
            .is_some_and(|up| up.host_epoch == req_epoch && up.upload_seq == upload_seq);
        if has_matching {
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
            if fnv1a64_words(&active.words) != geom_hash {
                self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_HASH_MISMATCH));
                return;
            }
            match self.insert_lru(key, active.words, commit_key) {
                Ok(()) => self.ready_replies.push(KernelReply::ctl(req.request, service_status::OK, ERR_NONE)),
                Err(code) => self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, code)),
            }
            return;
        }
        if let Some(pos) =
            self.lru.iter().position(|entry| entry.key == key && entry.committed_upload_key == commit_key)
        {
            self.touch_lru(pos);
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::OK, ERR_NONE));
        } else {
            self.ready_replies.push(KernelReply::ctl(req.request, service_status::FAILED, ERR_CTX_EVICTED));
        }
    }

    fn handle_run_batch(&mut self, req: &ServiceRequest) {
        let (Some(kernel_id), Some(item_count)) = (whole(req.a), whole(req.c)) else {
            self.ready_replies.push(KernelReply::batch_err(
                req.request,
                service_status::FAILED,
                ERR_INVALID_REQUEST,
                0,
            ));
            return;
        };
        if req.values.len() < 8 {
            self.ready_replies.push(KernelReply::batch_err(
                req.request,
                service_status::FAILED,
                ERR_INVALID_REQUEST,
                kernel_id,
            ));
            return;
        }
        let (Some(binding_hash), Some(geom_hash)) =
            (unpack_u64(&req.values[0..4]), unpack_u64(&req.values[4..8]))
        else {
            self.ready_replies.push(KernelReply::batch_err(
                req.request,
                service_status::FAILED,
                ERR_INVALID_REQUEST,
                kernel_id,
            ));
            return;
        };
        let Some(manifest) = self.manifest(kernel_id).cloned() else {
            self.ready_replies.push(KernelReply::batch_err(
                req.request,
                service_status::FAILED,
                ERR_UNKNOWN_KERNEL,
                kernel_id,
            ));
            return;
        };
        if manifest.binding_hash != binding_hash {
            self.ready_replies.push(KernelReply::batch_err(
                req.request,
                service_status::FAILED,
                ERR_BINDING_MISMATCH,
                kernel_id,
            ));
            return;
        }
        let items = match decode_items(&manifest.in_schema, item_count as usize, &req.values[8..]) {
            Ok(items) => items,
            Err(code) => {
                self.ready_replies.push(KernelReply::batch_err(
                    req.request,
                    service_status::FAILED,
                    code,
                    kernel_id,
                ));
                return;
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
                return;
            };
            self.touch_lru(pos);
            let entry = &self.lru[pos];
            Some((entry.revision, entry.words.clone()))
        } else {
            None
        };
        self.pending_batches.push(PendingBatch {
            epoch: self.host_epoch,
            request_id: req.request,
            kernel_id,
            manifest,
            binding_hash,
            geom_hash,
            item_count,
            ctx,
            items,
        });
    }

    async fn ensure_pool(&mut self, kernel_id: u32, name: &str) -> Result<(), u32> {
        if self.pools.contains_key(&kernel_id) {
            return Ok(());
        }
        let config = Object::new();
        let _ = Reflect::set(&config, &"workers".into(), &JsValue::from(self.worker_count as u32));
        let _ = Reflect::set(&config, &"deadlineMs".into(), &JsValue::from(WEB_BATCH_DEADLINE_MS));
        let _ = Reflect::set(&config, &"maxTotalOutputBytes".into(), &JsValue::from((2u32 << 20) as f64));
        let _ = Reflect::set(&config, &"maxPhysicalOutputBytes".into(), &JsValue::from((4u32 << 20) as f64));
        let promise = BrowserKernelPool::open(&self.receipt, name, &config.into()).map_err(|_| ERR_UNAVAILABLE)?;
        let value = JsFuture::from(promise).await.map_err(|_| ERR_UNAVAILABLE)?;
        let pool: BrowserKernelPool = value.unchecked_into();
        self.pools.insert(kernel_id, WebPoolSlot { pool, installed_revision: None });
        Ok(())
    }

    async fn execute_batch(&mut self, job: PendingBatch) -> KernelReply {
        if let Err(code) = self.ensure_pool(job.kernel_id, &job.manifest.name).await {
            return KernelReply::batch_err(job.request_id, service_status::UNAVAILABLE, code, job.kernel_id);
        }
        let mut faulted = false;
        let result: Result<Vec<u32>, u32> = async {
            if let Some((revision, ref words)) = job.ctx {
                let needs_install = self.pools.get(&job.kernel_id).is_some_and(|s| s.installed_revision != Some(revision));
                if needs_install {
                    let bytes = words_to_u8(words);
                    let u8_arr = Uint8Array::from(bytes.as_slice());
                    let promise = self
                        .pools
                        .get(&job.kernel_id)
                        .unwrap()
                        .pool
                        .set_context(revision as f64, &u8_arr)
                        .map_err(|_| ERR_EXECUTION_FAILED)?;
                    if let Err(err) = JsFuture::from(promise).await {
                        let code = match js_error_code(&err).as_deref() {
                            Some("timeout") => ERR_TIMEOUT,
                            _ => ERR_EXECUTION_FAILED,
                        };
                        faulted = true;
                        return Err(code);
                    }
                    if let Some(slot) = self.pools.get_mut(&job.kernel_id) {
                        slot.installed_revision = Some(revision);
                    }
                }
            }
            let items_arr = Uint32Array::from(job.items.as_slice());
            let pool = &self.pools.get(&job.kernel_id).unwrap().pool;
            let promise = if job.manifest.variable_output() {
                pool.run_variable(&items_arr)
            } else {
                pool.run_fixed(&items_arr)
            }
            .map_err(|_| ERR_EXECUTION_FAILED)?;
            match JsFuture::from(promise).await {
                Ok(val) => Ok(Uint32Array::new(&val).to_vec()),
                Err(err) => {
                    let code = match js_error_code(&err).as_deref() {
                        Some("timeout") => {
                            faulted = true;
                            ERR_TIMEOUT
                        }
                        Some("output_budget_exceeded" | "single_item_output_overflow") => ERR_OUTPUT_TOO_LARGE,
                        _ => {
                            faulted = true;
                            ERR_EXECUTION_FAILED
                        }
                    };
                    Err(code)
                }
            }
        }
        .await;
        if faulted
            && let Some(slot) = self.pools.remove(&job.kernel_id)
        {
            let promise = slot.pool.close();
            spawn_local(async move {
                let _ = JsFuture::from(promise).await;
            });
        }
        match result {
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
            Err(code) => KernelReply::batch_err(job.request_id, service_status::FAILED, code, job.kernel_id),
        }
    }

    /// Awaits pending browser kernel batches for the current epoch, splices completed replies
    /// atomically within `MAX_EVENTS` and `65_536`, and appends live-only `CAP` at the tail.
    pub async fn join_live_kernels_async(&mut self, input: &mut Vec<f32>) -> usize {
        let batches = std::mem::take(&mut self.pending_batches);
        for batch in batches {
            if batch.epoch == self.host_epoch {
                let reply = self.execute_batch(batch).await;
                if self.host_epoch == self.host_epoch {
                    self.ready_replies.push(reply);
                }
            }
        }
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
                let downgraded = KernelReply::batch_err(
                    reply.request_id,
                    service_status::FAILED,
                    ERR_EVENT_BUDGET,
                    reply.kernel_id,
                );
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

impl Drop for WebKernelService {
    fn drop(&mut self) {
        self.close_all_pools();
    }
}
