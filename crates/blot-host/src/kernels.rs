//! Unified warm work-stealing multi-core `KernelPool` for ABI 8 `entry kernel`
//! modules (`blot2/tasks/CONCURRENCY_VNEXT.md` §5.1).
use std::cell::UnsafeCell;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use wasmtime::{Engine, Val};

use crate::{
    GUEST_STACK_BYTES, GuestModule, KernelManifest, Limits, ValueType, WasmtimeGuest, WireSchemaKind,
    ensure_engine_ticker, is_trap_interrupt, is_unrecoverable_trap,
};

/// Configuration for [`KernelPool::open`].
#[derive(Clone, Debug)]
pub struct KernelConfig {
    /// Worker count (`0` = direct serial execution on caller thread; `>= 1` = persistent worker pool).
    pub workers: usize,
    /// Optional per-worker linear-memory limit enforced by [`MemoryTrackerLimiter`].
    pub memory: Option<Limits>,
    /// Optional absolute per-operation deadline for `set_context`, `clear_context`, and `run_*`.
    /// Requires an epoch-interrupted [`Engine`] (`engine.get_epoch_interruption() == true`).
    pub deadline: Option<Duration>,
    /// Semantic logical output ceiling in bytes (`run_fixed`, `run_variable_chunks`, `run_variable`).
    pub max_total_output_bytes: usize,
    /// Physical host output memory ceiling in bytes (`>= max_total_output_bytes` for fixed/chunks,
    /// `>= 2 * max_total_output_bytes` for multi-chunk `run_variable`).
    pub max_physical_output_bytes: usize,
    /// Nominal items per chunk before transport-budget clamping (worker-count-independent).
    pub default_chunk_items: usize,
    /// Physical footprint reclamation threshold after a recoverable deterministic trap.
    pub retire_memory_threshold: usize,
    /// Maximum detached timed-out worker threads allowed in flight before returning `PoolExhausted`.
    pub max_retired_in_flight: usize,
    /// Requested epoch tick duration when running on an epoch-enabled engine (clamped to `<= 2 ms`).
    pub tick: Duration,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            workers: 4,
            memory: Some(Limits { memory: 64 << 20 }),
            deadline: None,
            max_total_output_bytes: 16 << 20,
            max_physical_output_bytes: 32 << 20,
            default_chunk_items: 64,
            retire_memory_threshold: 64 << 20,
            max_retired_in_flight: 8,
            tick: Duration::from_millis(2),
        }
    }
}

/// Errors returned by [`KernelPool`] operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelError {
    EngineMismatch,
    FuelEngineUnsupportedForKernels,
    DeadlineRequiresBoundedEngine,
    WtimeAotUnsupported,
    UnknownKernel(String),
    InvalidConfig(String),
    ContextNotSupported,
    ContextNotPrepared,
    RevisionContentMismatch,
    InvalidInputLength { words: usize, stride: usize },
    WrongOutputMode { expected_variable: bool },
    OutputBudgetExceeded { required_bytes: usize, limit_bytes: usize },
    PhysicalOutputBudgetExceeded { required_bytes: usize, limit_bytes: usize },
    SingleItemOutputOverflow { item_index: usize },
    OutOfMemory { item_index: Option<usize>, detail: String },
    RunTrap { item_index: usize, detail: String },
    CleanupTrap { item_index: usize, detail: String },
    ContextTrap { detail: String },
    Timeout,
    PoolExhausted,
    PoolClosed,
    Internal(String),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::EngineMismatch => write!(f, "kernel pool engine does not match module engine"),
            KernelError::FuelEngineUnsupportedForKernels => {
                write!(f, "fuel-metered engines are unsupported for KernelPool")
            }
            KernelError::DeadlineRequiresBoundedEngine => {
                write!(f, "KernelPool deadline requires an epoch-interrupted engine")
            }
            KernelError::WtimeAotUnsupported => write!(f, "KernelPool requires a Wasmtime module, not wtime-aot"),
            KernelError::UnknownKernel(name) => write!(f, "unknown entry kernel `{name}`"),
            KernelError::InvalidConfig(msg) => write!(f, "invalid KernelPool config: {msg}"),
            KernelError::ContextNotSupported => write!(f, "kernel does not accept a prepared context"),
            KernelError::ContextNotPrepared => write!(f, "prepared kernel context has not been set"),
            KernelError::RevisionContentMismatch => {
                write!(f, "context revision reused with different payload bytes")
            }
            KernelError::InvalidInputLength { words, stride } => {
                write!(f, "input length {words} words is not a multiple of kernel stride {stride}")
            }
            KernelError::WrongOutputMode { expected_variable } => {
                if *expected_variable {
                    write!(f, "kernel has fixed output; call run_fixed instead")
                } else {
                    write!(f, "kernel has variable output; call run_variable or run_variable_chunks instead")
                }
            }
            KernelError::OutputBudgetExceeded { required_bytes, limit_bytes } => {
                write!(f, "kernel output {required_bytes} bytes exceeds max_total_output_bytes ({limit_bytes})")
            }
            KernelError::PhysicalOutputBudgetExceeded { required_bytes, limit_bytes } => {
                write!(f, "kernel output {required_bytes} bytes exceeds max_physical_output_bytes ({limit_bytes})")
            }
            KernelError::SingleItemOutputOverflow { item_index } => {
                write!(f, "single item {item_index} overflowed guest transport_budget")
            }
            KernelError::OutOfMemory { item_index, detail } => match item_index {
                Some(idx) => write!(f, "kernel out of memory at item {idx}: {detail}"),
                None => write!(f, "kernel out of memory: {detail}"),
            },
            KernelError::RunTrap { item_index, detail } => write!(f, "kernel trapped at item {item_index}: {detail}"),
            KernelError::CleanupTrap { item_index, detail } => {
                write!(f, "kernel cleanup trapped at item {item_index}: {detail}")
            }
            KernelError::ContextTrap { detail } => write!(f, "kernel context preparation trapped: {detail}"),
            KernelError::Timeout => write!(f, "kernel operation exceeded deadline"),
            KernelError::PoolExhausted => write!(f, "kernel pool exhausted by retired stuck workers"),
            KernelError::PoolClosed => write!(f, "kernel pool is closed"),
            KernelError::Internal(msg) => write!(f, "kernel host internal error: {msg}"),
        }
    }
}

impl std::error::Error for KernelError {}

/// Runtime telemetry for [`KernelPool`] verification and benchmarks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KernelStats {
    pub worker_starts: u64,
    pub worker_retirements: u64,
    pub fresh_oom_retries: u64,
    pub context_installs: u64,
    pub chunks_completed: u64,
    pub bisections: u64,
    pub peak_charged_bytes: usize,
    pub active_workers: usize,
    pub retired_in_flight: usize,
    pub effective_tick: Option<Duration>,
}

struct StatsAtomic {
    worker_starts: AtomicU64,
    worker_retirements: AtomicU64,
    fresh_oom_retries: AtomicU64,
    context_installs: AtomicU64,
    chunks_completed: AtomicU64,
    bisections: AtomicU64,
    peak_charged_bytes: AtomicUsize,
}

impl Default for StatsAtomic {
    fn default() -> Self {
        Self {
            worker_starts: AtomicU64::new(0),
            worker_retirements: AtomicU64::new(0),
            fresh_oom_retries: AtomicU64::new(0),
            context_installs: AtomicU64::new(0),
            chunks_completed: AtomicU64::new(0),
            bisections: AtomicU64::new(0),
            peak_charged_bytes: AtomicUsize::new(0),
        }
    }
}

impl StatsAtomic {
    fn record_peak(&self, bytes: usize) {
        self.peak_charged_bytes.fetch_max(bytes, Ordering::Relaxed);
    }
}

struct PreparedContextSnapshot {
    snapshot_id: u64,
    revision: u64,
    words: Arc<[u32]>,
}

struct WorkerGuest {
    guest: WasmtimeGuest,
    installed_snapshot_id: Option<u64>,
    store_has_prior_history: bool,
}

/// Sound disjoint fixed-output backing storage (`1×` exact peak, zero-copy recovery via `Box::into_vec`).
struct DisjointStorage {
    data: UnsafeCell<Box<[u32]>>,
    len: usize,
}

// SAFETY: Every chunk writes strictly to its own disjoint `[start_word..end_word)` sub-range
// partitioned exclusively by the coordinator; no two threads ever access overlapping indices.
unsafe impl Sync for DisjointStorage {}
unsafe impl Send for DisjointStorage {}

impl DisjointStorage {
    fn new(words: usize) -> Self {
        Self { data: UnsafeCell::new(vec![0u32; words].into_boxed_slice()), len: words }
    }

    /// # Safety
    /// Caller must hold exclusive ownership of `[start_word..end_word)` within `0..self.len`.
    unsafe fn write_disjoint(&self, start_word: usize, end_word: usize, writer: impl FnOnce(&mut [u32]) -> anyhow::Result<()>) -> anyhow::Result<()> {
        debug_assert!(start_word <= end_word && end_word <= self.len);
        // SAFETY: `self.data` is a heap-allocated `Box<[u32]>` whose buffer never moves;
        // caller guarantees disjointness of `[start_word..end_word)`.
        let base_ptr = unsafe { (*self.data.get()).as_mut_ptr() };
        let slice = unsafe { std::slice::from_raw_parts_mut(base_ptr.add(start_word), end_word - start_word) };
        writer(slice)
    }

    fn into_vec(self) -> Vec<u32> {
        self.data.into_inner().into_vec()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct WorkRange {
    start_item: usize,
    end_item: usize,
    retried_fresh: bool,
}

impl WorkRange {
    fn len(self) -> usize {
        self.end_item - self.start_item
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchMode {
    Fixed,
    Variable,
}

enum ChunkState {
    FixedDone {
        end_item: usize,
    },
    VariableBuffered {
        end_item: usize,
        words: Vec<u32>,
    },
    VariableSettled {
        end_item: usize,
        words: Vec<u32>,
    },
    OverBudgetHint {
        result_bytes: usize,
    },
    CleanupFailed {
        result_bytes: usize,
        detail: String,
    },
    PrimaryError {
        error: KernelError,
    },
}

enum AdmitDecision {
    Copy,
    Discard,
    ParkAndRequeue,
}

struct BatchShared {
    mode: BatchMode,
    input: Arc<[u32]>,
    snapshot: Option<Arc<PreparedContextSnapshot>>,
    fixed_storage: Option<Arc<DisjointStorage>>,
    deadline_at: Option<Instant>,
    cancel_flag: Arc<AtomicBool>,
    state: Mutex<BatchCoord>,
    cvar: Condvar,
}

struct BatchCoord {
    pending: BinaryHeap<Reverse<WorkRange>>,
    active_ranges: BTreeMap<usize, usize>, // start_item -> end_item
    chunks: BTreeMap<usize, ChunkState>,
    frontier_item: usize,
    total_items: usize,
    settled_prefix_bytes: usize,
    higher_buffered_bytes: usize,
    in_flight_copy_bytes: usize,
    pending_frontier_copy: bool,
    max_total_output_bytes: usize,
    earliest_failed_item: usize,
    timed_out: bool,
    active_workers_on_batch: usize,
}

impl BatchCoord {
    fn new(
        total_items: usize,
        max_chunk_items: usize,
        max_total_output_bytes: usize,
        mode: BatchMode,
    ) -> Self {
        let mut pending = BinaryHeap::new();
        let mut start = 0;
        while start < total_items {
            let end = (start + max_chunk_items).min(total_items);
            pending.push(Reverse(WorkRange { start_item: start, end_item: end, retried_fresh: false }));
            start = end;
        }
        let _ = mode;
        Self {
            pending,
            active_ranges: BTreeMap::new(),
            chunks: BTreeMap::new(),
            frontier_item: 0,
            total_items,
            settled_prefix_bytes: 0,
            higher_buffered_bytes: 0,
            in_flight_copy_bytes: 0,
            pending_frontier_copy: false,
            max_total_output_bytes,
            earliest_failed_item: usize::MAX,
            timed_out: false,
            active_workers_on_batch: 0,
        }
    }

    fn current_charged_bytes(&self) -> usize {
        self.settled_prefix_bytes + self.higher_buffered_bytes + self.in_flight_copy_bytes
    }

    fn tighten_failure(&mut self, start_item: usize) {
        if start_item < self.earliest_failed_item {
            self.earliest_failed_item = start_item;
            // Drop queued unstarted ranges >= earliest_failed_item.
            let kept: Vec<_> = self
                .pending
                .drain()
                .filter(|Reverse(r)| r.start_item < self.earliest_failed_item)
                .collect();
            for item in kept {
                self.pending.push(item);
            }
            // Drop any buffered chunks strictly above earliest_failed_item to free budget immediately.
            let drop_keys: Vec<usize> =
                self.chunks.range((start_item + 1)..).map(|(&k, _)| k).collect();
            for key in drop_keys {
                if let Some(ChunkState::VariableBuffered { words, .. }) = self.chunks.remove(&key) {
                    self.higher_buffered_bytes = self.higher_buffered_bytes.saturating_sub(words.len() * 4);
                }
            }
        }
    }

    fn record_primary_error(&mut self, start_item: usize, error: KernelError) {
        // Primary error (`RunTrap`, `SingleItemOutputOverflow`, `OutOfMemory`) beats everything at the same `start_item`.
        if let Some(ChunkState::VariableBuffered { words, .. }) = self.chunks.remove(&start_item) {
            self.higher_buffered_bytes = self.higher_buffered_bytes.saturating_sub(words.len() * 4);
        }
        self.chunks.insert(start_item, ChunkState::PrimaryError { error });
        self.tighten_failure(start_item);
    }

    fn record_cleanup_failed(&mut self, range: WorkRange, result_bytes: usize, detail: String) {
        // Do not overwrite a higher-precedence PrimaryError at the same `start_item`.
        if matches!(self.chunks.get(&range.start_item), Some(ChunkState::PrimaryError { .. })) {
            return;
        }
        if let Some(ChunkState::VariableBuffered { words, .. }) = self.chunks.remove(&range.start_item) {
            self.higher_buffered_bytes = self.higher_buffered_bytes.saturating_sub(words.len() * 4);
        }
        self.chunks.insert(
            range.start_item,
            ChunkState::CleanupFailed { result_bytes, detail },
        );
        self.tighten_failure(range.start_item);
    }

    fn record_over_budget_frontier(&mut self, start_item: usize, required_bytes: usize) {
        let limit_bytes = self.max_total_output_bytes;
        self.record_primary_error(start_item, KernelError::OutputBudgetExceeded { required_bytes, limit_bytes });
    }

    /// Evicts highest-`start_item` completed buffered chunks strictly above `above_item`
    /// until `settled_prefix_bytes + higher_buffered_bytes + need_bytes <= max_total_output_bytes`
    /// (or no evictable chunks remain).
    fn evict_higher_buffered(&mut self, above_item: usize, need_bytes: usize) {
        while self.settled_prefix_bytes + self.higher_buffered_bytes + self.in_flight_copy_bytes + need_bytes
            > self.max_total_output_bytes
        {
            let candidate = self
                .chunks
                .range((above_item + 1)..)
                .rev()
                .find_map(|(&k, state)| match state {
                    ChunkState::VariableBuffered { end_item, .. } => Some((k, *end_item)),
                    _ => None,
                });
            let Some((key, end_item)) = candidate else {
                break;
            };
            if let Some(ChunkState::VariableBuffered { words, .. }) = self.chunks.remove(&key) {
                self.higher_buffered_bytes = self.higher_buffered_bytes.saturating_sub(words.len() * 4);
                if key < self.earliest_failed_item {
                    self.pending.push(Reverse(WorkRange {
                        start_item: key,
                        end_item,
                        retried_fresh: false,
                    }));
                }
            }
        }
    }

    /// Advances `frontier_item` across all contiguous resolved chunks, resolving
    /// same-`start_item` precedence (`PrimaryError` > `OutputBudgetExceeded` > `CleanupTrap`).
    fn advance_frontier(&mut self) {
        loop {
            if self.frontier_item >= self.total_items {
                break;
            }
            let Some(state) = self.chunks.remove(&self.frontier_item) else {
                break;
            };
            match state {
                ChunkState::FixedDone { end_item } => {
                    self.chunks.insert(self.frontier_item, ChunkState::FixedDone { end_item });
                    self.frontier_item = end_item;
                }
                ChunkState::VariableSettled { end_item, words } => {
                    self.chunks.insert(self.frontier_item, ChunkState::VariableSettled { end_item, words });
                    self.frontier_item = end_item;
                }
                ChunkState::VariableBuffered { end_item, words } => {
                    let bytes = words.len() * 4;
                    self.higher_buffered_bytes = self.higher_buffered_bytes.saturating_sub(bytes);
                    if self.settled_prefix_bytes.saturating_add(bytes) > self.max_total_output_bytes {
                        let required = self.settled_prefix_bytes.saturating_add(bytes);
                        drop(words);
                        self.record_over_budget_frontier(self.frontier_item, required);
                        break;
                    }
                    self.settled_prefix_bytes += bytes;
                    self.chunks.insert(self.frontier_item, ChunkState::VariableSettled { end_item, words });
                    self.frontier_item = end_item;
                }
                ChunkState::OverBudgetHint { result_bytes, .. } => {
                    let required = self.settled_prefix_bytes.saturating_add(result_bytes);
                    if required > self.max_total_output_bytes {
                        self.record_over_budget_frontier(self.frontier_item, required);
                        break;
                    }
                    // Shouldn't happen because OverBudgetHint is only stored when settled_prefix_bytes + result_bytes > max;
                    // defensively re-queue if ever reached.
                    unreachable!("OverBudgetHint cannot become in-budget monotonically");
                }
                ChunkState::CleanupFailed { result_bytes, detail, .. } => {
                    let required = self.settled_prefix_bytes.saturating_add(result_bytes);
                    let error = if required > self.max_total_output_bytes {
                        KernelError::OutputBudgetExceeded {
                            required_bytes: required,
                            limit_bytes: self.max_total_output_bytes,
                        }
                    } else {
                        KernelError::CleanupTrap { item_index: self.frontier_item, detail }
                    };
                    self.chunks.insert(self.frontier_item, ChunkState::PrimaryError { error });
                    self.tighten_failure(self.frontier_item);
                    break;
                }
                ChunkState::PrimaryError { error } => {
                    self.chunks.insert(self.frontier_item, ChunkState::PrimaryError { error });
                    self.tighten_failure(self.frontier_item);
                    break;
                }
            }
        }
    }

    fn is_done(&self) -> bool {
        if self.timed_out {
            return true;
        }
        if self.frontier_item >= self.total_items {
            return true;
        }
        if self.earliest_failed_item != usize::MAX && self.frontier_item >= self.earliest_failed_item {
            return true;
        }
        false
    }

    fn has_active_below_failure(&self) -> bool {
        self.active_ranges.keys().any(|&start| start < self.earliest_failed_item)
    }

    fn next_work(&mut self) -> Option<WorkRange> {
        while let Some(Reverse(range)) = self.pending.pop() {
            if range.start_item >= self.earliest_failed_item {
                continue;
            }
            self.active_ranges.insert(range.start_item, range.end_item);
            return Some(range);
        }
        None
    }
}

#[derive(Clone)]
struct PoolContext {
    engine: Engine,
    module: GuestModule,
    manifest: KernelManifest,
    config: KernelConfig,
    cleanup_grace: Option<Duration>,
    stats: Arc<StatsAtomic>,
}

struct WorkerSlot {
    dispatch: Arc<(Mutex<WorkerCommand>, Condvar)>,
    handle: Option<JoinHandle<()>>,
    retired_flag: Arc<AtomicBool>,
}

struct SyncSnapshotJob {
    snapshot: Option<Arc<PreparedContextSnapshot>>,
    deadline_at: Option<Instant>,
    cancel_flag: Arc<AtomicBool>,
    result: Mutex<Option<std::result::Result<(), KernelError>>>,
    cvar: Condvar,
}

struct WorkerCommand {
    batch: Option<Arc<BatchShared>>,
    sync_snapshot: Option<Arc<SyncSnapshotJob>>,
    shutdown: bool,
    busy: bool,
}

/// Unified warm work-stealing `KernelPool` for `entry kernel` modules.
pub struct KernelPool {
    ctx: PoolContext,
    max_chunk_items: usize,
    effective_tick: Option<Duration>,
    next_snapshot_id: u64,
    current_ctx: Option<Arc<PreparedContextSnapshot>>,
    /// Direct serial guest used when `config.workers == 0`.
    direct_guest: Option<WorkerGuest>,
    /// Persistent worker threads used when `config.workers >= 1`.
    workers: Vec<WorkerSlot>,
    retired_in_flight: Arc<AtomicUsize>,
    closed: bool,
}

impl KernelPool {
    /// Opens a warm [`KernelPool`] for `entry kernel` `name` on `engine` + `module`.
    pub fn open(
        engine: &Engine,
        module: &GuestModule,
        name: &str,
        config: KernelConfig,
    ) -> std::result::Result<Self, KernelError> {
        let _wt_module = module.ensure_wasmtime_module(engine)?;
        if engine.get_consume_fuel() {
            return Err(KernelError::FuelEngineUnsupportedForKernels);
        }
        if config.deadline.is_some() && !engine.get_epoch_interruption() {
            return Err(KernelError::DeadlineRequiresBoundedEngine);
        }
        if let Some(deadline) = config.deadline
            && deadline.is_zero()
        {
            return Err(KernelError::InvalidConfig("deadline must be non-zero".into()));
        }
        if config.max_physical_output_bytes < config.max_total_output_bytes {
            return Err(KernelError::InvalidConfig(
                "max_physical_output_bytes must be >= max_total_output_bytes".into(),
            ));
        }
        let manifest = module
            .abi()
            .kernel(name)
            .cloned()
            .ok_or_else(|| KernelError::UnknownKernel(name.to_owned()))?;

        // Unconditionally guarantee a live wakeable epoch ticker whenever the engine has epoch interruption,
        // even when `config.deadline.is_none()`, so epoch-timed cleanup grace can never hang!
        let (effective_tick, cleanup_grace) = if engine.get_epoch_interruption() {
            let requested = config.tick.min(Duration::from_millis(2));
            let tick = ensure_engine_ticker(engine, requested)
                .map_err(|e| KernelError::Internal(format!("starting epoch ticker: {e:#}")))?;
            let grace = Duration::from_millis(10).max(tick.saturating_mul(2));
            (Some(tick), Some(grace))
        } else {
            (None, None)
        };

        let transport_budget = module.abi().transport_budget as usize;
        let in_stride = manifest.in_schema.stride as usize;
        let max_in_items = (transport_budget.saturating_sub(4)) / (in_stride * 4);
        let mut max_chunk_items = config.default_chunk_items.max(1).min(max_in_items);
        if !manifest.variable_output() {
            let out_stride = manifest.out_schema.stride as usize;
            let max_out_items = (transport_budget.saturating_sub(4)) / (out_stride * 4);
            max_chunk_items = max_chunk_items.min(max_out_items);
        }
        let max_chunk_items = max_chunk_items.max(1);

        let ctx = PoolContext {
            engine: engine.clone(),
            module: module.clone(),
            manifest,
            config,
            cleanup_grace,
            stats: Arc::new(StatsAtomic::default()),
        };
        let retired_in_flight = Arc::new(AtomicUsize::new(0));
        let mut pool = Self {
            ctx,
            max_chunk_items,
            effective_tick,
            next_snapshot_id: 1,
            current_ctx: None,
            direct_guest: None,
            workers: Vec::new(),
            retired_in_flight,
            closed: false,
        };
        pool.ensure_workers()?;
        Ok(pool)
    }

    /// Convenience helper compiling raw Wasm bytes on `engine` and opening a [`KernelPool`].
    #[cfg(feature = "compiler")]
    pub fn open_wasm(
        engine: &Engine,
        wasm: &[u8],
        name: &str,
        config: KernelConfig,
    ) -> std::result::Result<Self, KernelError> {
        let module = GuestModule::new(engine, wasm, None)
            .map_err(|e| KernelError::InvalidConfig(format!("compiling kernel wasm: {e:#}")))?;
        Self::open(engine, &module, name, config)
    }

    pub fn manifest(&self) -> &KernelManifest {
        &self.ctx.manifest
    }

    pub fn max_chunk_items(&self) -> usize {
        self.max_chunk_items
    }

    pub fn stats(&self) -> KernelStats {
        KernelStats {
            worker_starts: self.ctx.stats.worker_starts.load(Ordering::Relaxed),
            worker_retirements: self.ctx.stats.worker_retirements.load(Ordering::Relaxed),
            fresh_oom_retries: self.ctx.stats.fresh_oom_retries.load(Ordering::Relaxed),
            context_installs: self.ctx.stats.context_installs.load(Ordering::Relaxed),
            chunks_completed: self.ctx.stats.chunks_completed.load(Ordering::Relaxed),
            bisections: self.ctx.stats.bisections.load(Ordering::Relaxed),
            peak_charged_bytes: self.ctx.stats.peak_charged_bytes.load(Ordering::Relaxed),
            active_workers: self.workers.len(),
            retired_in_flight: self.retired_in_flight.load(Ordering::Acquire),
            effective_tick: self.effective_tick,
        }
    }

    fn ensure_workers(&mut self) -> std::result::Result<(), KernelError> {
        let target = self.ctx.config.workers;
        if target == 0 {
            return Ok(());
        }
        while self.workers.len() < target {
            let retired = self.retired_in_flight.load(Ordering::Acquire);
            if self.workers.len() + retired >= target + self.ctx.config.max_retired_in_flight {
                return Err(KernelError::PoolExhausted);
            }
            let slot = spawn_worker_thread(self.ctx.clone(), self.retired_in_flight.clone(), self.workers.len())?;
            self.workers.push(slot);
        }
        Ok(())
    }

    /// Sets or idempotently re-validates the kernel's prepared context snapshot.
    ///
    /// Always validates a new revision on a brand-new empty candidate `Store`
    /// (`0` old context bytes) while retaining any previously published snapshot
    /// untouched on failure.
    pub fn set_context(&mut self, revision: u64, words: &[u32]) -> std::result::Result<(), KernelError> {
        if self.closed {
            return Err(KernelError::PoolClosed);
        }
        let Some(ctx_schema) = &self.ctx.manifest.ctx_schema else {
            return Err(KernelError::ContextNotSupported);
        };
        let transport_budget = self.ctx.module.abi().transport_budget as usize;
        let max_words = (transport_budget.saturating_sub(4)) / 4;
        if words.len() > max_words {
            return Err(KernelError::InvalidInputLength { words: words.len(), stride: max_words.max(1) });
        }
        if ctx_schema.kind == WireSchemaKind::FlatFixed && words.len() != ctx_schema.stride as usize {
            return Err(KernelError::InvalidInputLength {
                words: words.len(),
                stride: ctx_schema.stride as usize,
            });
        }
        let deadline_at = self.ctx.config.deadline.map(|d| Instant::now() + d);
        if let Some(existing) = &self.current_ctx
            && existing.revision == revision
        {
            if existing.words.as_ref() != words {
                return Err(KernelError::RevisionContentMismatch);
            }
            // Idempotent same-revision call still goes through `ensure_guest_snapshot`.
            if self.ctx.config.workers == 0 {
                let cancel = Arc::new(AtomicBool::new(false));
                ensure_guest_snapshot(&self.ctx, &mut self.direct_guest, Some(existing), deadline_at, &cancel)?;
            } else {
                let mut candidate: Option<WorkerGuest> = None;
                let cancel = Arc::new(AtomicBool::new(false));
                ensure_guest_snapshot(&self.ctx, &mut candidate, Some(existing), deadline_at, &cancel)?;
            }
            return Ok(());
        }
        let snapshot_id = self.next_snapshot_id;
        self.next_snapshot_id = self.next_snapshot_id.checked_add(1).unwrap_or(1);
        let next = Arc::new(PreparedContextSnapshot {
            snapshot_id,
            revision,
            words: Arc::from(words),
        });
        // ALWAYS validate new revision on a brand-new empty candidate Store so old context is
        // untouched on failure and candidate sees 0 old context bytes!
        let mut candidate: Option<WorkerGuest> = None;
        let cancel = Arc::new(AtomicBool::new(false));
        ensure_guest_snapshot(&self.ctx, &mut candidate, Some(&next), deadline_at, &cancel)?;
        if let Some(deadline) = deadline_at
            && Instant::now() >= deadline
        {
            return Err(KernelError::Timeout);
        }
        self.current_ctx = Some(next);
        Ok(())
    }

    pub fn set_context_f32(&mut self, revision: u64, floats: &[f32]) -> std::result::Result<(), KernelError> {
        let words: Vec<u32> = floats.iter().copied().map(f32::to_bits).collect();
        self.set_context(revision, &words)
    }

    /// Clears the published prepared context snapshot.
    pub fn clear_context(&mut self) -> std::result::Result<(), KernelError> {
        if self.closed {
            return Err(KernelError::PoolClosed);
        }
        if !self.ctx.manifest.has_context() {
            return Err(KernelError::ContextNotSupported);
        }
        let deadline_at = self.ctx.config.deadline.map(|d| Instant::now() + d);
        if self.ctx.config.workers == 0
            && self.direct_guest.is_some()
        {
            let cancel = Arc::new(AtomicBool::new(false));
            ensure_guest_snapshot(&self.ctx, &mut self.direct_guest, None, deadline_at, &cancel)?;
        }
        if let Some(deadline) = deadline_at
            && Instant::now() >= deadline
        {
            return Err(KernelError::Timeout);
        }
        self.current_ctx = None;
        Ok(())
    }

    /// Executes a fixed-output kernel batch into ONE exact-size `DisjointStorage` (`1×` peak, `0` copies).
    pub fn run_fixed(&mut self, input_words: &[u32]) -> std::result::Result<Vec<u32>, KernelError> {
        if self.closed {
            return Err(KernelError::PoolClosed);
        }
        if self.ctx.manifest.variable_output() {
            return Err(KernelError::WrongOutputMode { expected_variable: false });
        }
        let count = self.validate_input_count(input_words)?;
        let snapshot = self.require_batch_snapshot()?;
        let deadline_at = self.ctx.config.deadline.map(|d| Instant::now() + d);

        let out_stride = self.ctx.manifest.out_schema.stride as usize;
        let total_words = count
            .checked_mul(out_stride)
            .ok_or(KernelError::OutputBudgetExceeded {
                required_bytes: usize::MAX,
                limit_bytes: self.ctx.config.max_total_output_bytes,
            })?;
        let total_bytes = total_words
            .checked_mul(4)
            .ok_or(KernelError::OutputBudgetExceeded {
                required_bytes: usize::MAX,
                limit_bytes: self.ctx.config.max_total_output_bytes,
            })?;
        if total_bytes > self.ctx.config.max_total_output_bytes {
            return Err(KernelError::OutputBudgetExceeded {
                required_bytes: total_bytes,
                limit_bytes: self.ctx.config.max_total_output_bytes,
            });
        }
        if total_bytes > self.ctx.config.max_physical_output_bytes {
            return Err(KernelError::PhysicalOutputBudgetExceeded {
                required_bytes: total_bytes,
                limit_bytes: self.ctx.config.max_physical_output_bytes,
            });
        }

        if count == 0 {
            self.run_empty_batch(snapshot.as_ref(), deadline_at)?;
            return Ok(Vec::new());
        }

        let storage = Arc::new(DisjointStorage::new(total_words));
        self.ctx.stats.record_peak(total_bytes);

        if self.ctx.config.workers == 0 {
            self.run_fixed_direct(input_words, count, snapshot.as_ref(), &storage, deadline_at)?;
        } else {
            self.run_batch_parallel(
                BatchMode::Fixed,
                input_words,
                count,
                snapshot,
                Some(storage.clone()),
                deadline_at,
            )?;
        }

        if let Some(deadline) = deadline_at
            && Instant::now() >= deadline
        {
            return Err(KernelError::Timeout);
        }
        match Arc::try_unwrap(storage) {
            Ok(owned) => Ok(owned.into_vec()),
            Err(_) => Err(KernelError::Timeout),
        }
    }

    pub fn run_fixed_f32(&mut self, input_floats: &[f32]) -> std::result::Result<Vec<f32>, KernelError> {
        let input_words: Vec<u32> = input_floats.iter().copied().map(f32::to_bits).collect();
        let out_words = self.run_fixed(&input_words)?;
        Ok(out_words.into_iter().map(f32::from_bits).collect())
    }

    /// Executes a variable-output kernel batch returning ordered chunks (`1×` peak `<= max_total_output_bytes`).
    pub fn run_variable_chunks(&mut self, input_words: &[u32]) -> std::result::Result<Vec<Vec<u32>>, KernelError> {
        if self.closed {
            return Err(KernelError::PoolClosed);
        }
        if !self.ctx.manifest.variable_output() {
            return Err(KernelError::WrongOutputMode { expected_variable: true });
        }
        let count = self.validate_input_count(input_words)?;
        let snapshot = self.require_batch_snapshot()?;
        let deadline_at = self.ctx.config.deadline.map(|d| Instant::now() + d);

        if count == 0 {
            self.run_empty_batch(snapshot.as_ref(), deadline_at)?;
            return Ok(Vec::new());
        }

        let chunks = if self.ctx.config.workers == 0 {
            self.run_variable_direct(input_words, count, snapshot.as_ref(), deadline_at)?
        } else {
            self.run_batch_parallel(BatchMode::Variable, input_words, count, snapshot, None, deadline_at)?
        };
        if let Some(deadline) = deadline_at
            && Instant::now() >= deadline
        {
            return Err(KernelError::Timeout);
        }
        Ok(chunks)
    }

    /// Executes a variable-output kernel batch returning a single concatenated `Vec<u32>`
    /// (`1×` for single chunk, bounded `<= 2× max_total_output_bytes` validated against `max_physical_output_bytes`).
    pub fn run_variable(&mut self, input_words: &[u32]) -> std::result::Result<Vec<u32>, KernelError> {
        let deadline_at = self.ctx.config.deadline.map(|d| Instant::now() + d);
        let chunks = self.run_variable_chunks(input_words)?;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        if chunks.len() == 1 {
            return Ok(chunks.into_iter().next().unwrap());
        }
        let total_words: usize = chunks.iter().map(Vec::len).sum();
        let total_bytes = total_words.saturating_mul(4);
        let concat_peak = total_bytes.saturating_mul(2);
        if concat_peak > self.ctx.config.max_physical_output_bytes {
            return Err(KernelError::PhysicalOutputBudgetExceeded {
                required_bytes: concat_peak,
                limit_bytes: self.ctx.config.max_physical_output_bytes,
            });
        }
        let mut out = Vec::with_capacity(total_words);
        let mut remaining_chunk_bytes = total_bytes;
        for chunk in chunks {
            self.ctx.stats.record_peak(total_bytes + remaining_chunk_bytes);
            out.extend_from_slice(&chunk);
            remaining_chunk_bytes = remaining_chunk_bytes.saturating_sub(chunk.len() * 4);
            drop(chunk);
        }
        if let Some(deadline) = deadline_at
            && Instant::now() >= deadline
        {
            return Err(KernelError::Timeout);
        }
        Ok(out)
    }

    /// Idempotently closes the pool, detaching any stuck threads after a bounded `100 ms` grace.
    pub fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.direct_guest = None;
        let deadline = Instant::now() + Duration::from_millis(100);
        for slot in &mut self.workers {
            let (lock, cvar) = &*slot.dispatch;
            let mut cmd = lock.lock().unwrap();
            cmd.shutdown = true;
            cvar.notify_one();
            drop(cmd);
        }
        for mut slot in self.workers.drain(..) {
            let (lock, cvar) = &*slot.dispatch;
            let mut cmd = lock.lock().unwrap();
            while cmd.busy {
                let Some(rem) = deadline.checked_duration_since(Instant::now()) else {
                    break;
                };
                if rem.is_zero() {
                    break;
                }
                let (next, res) = cvar.wait_timeout(cmd, rem).unwrap();
                cmd = next;
                if res.timed_out() {
                    break;
                }
            }
            let still_busy = cmd.busy;
            drop(cmd);
            if !still_busy {
                if let Some(handle) = slot.handle.take() {
                    let _ = handle.join();
                }
            } else {
                // Non-blocking Drop: detach stuck thread cleanly with its Arc lease.
                if !slot.retired_flag.swap(true, Ordering::AcqRel) {
                    self.retired_in_flight.fetch_add(1, Ordering::AcqRel);
                }
                drop(slot.handle.take());
            }
        }
    }

    fn validate_input_count(&self, input_words: &[u32]) -> std::result::Result<usize, KernelError> {
        let in_stride = self.ctx.manifest.in_schema.stride as usize;
        if !input_words.len().is_multiple_of(in_stride) {
            return Err(KernelError::InvalidInputLength { words: input_words.len(), stride: in_stride });
        }
        Ok(input_words.len() / in_stride)
    }

    fn require_batch_snapshot(&self) -> std::result::Result<Option<Arc<PreparedContextSnapshot>>, KernelError> {
        if self.ctx.manifest.has_context() {
            match &self.current_ctx {
                Some(snap) => Ok(Some(snap.clone())),
                None => Err(KernelError::ContextNotPrepared),
            }
        } else {
            Ok(None)
        }
    }

    fn run_empty_batch(
        &mut self,
        snapshot: Option<&Arc<PreparedContextSnapshot>>,
        deadline_at: Option<Instant>,
    ) -> std::result::Result<(), KernelError> {
        if let Some(snap) = snapshot {
            if self.ctx.config.workers == 0 {
                let cancel = Arc::new(AtomicBool::new(false));
                ensure_guest_snapshot(&self.ctx, &mut self.direct_guest, Some(snap), deadline_at, &cancel)?;
            } else {
                self.ensure_workers()?;
                // Synchronize worker 0's persistent WorkerGuest directly so empty `count == 0`
                // batches warm-install or warm-replace its context snapshot cleanly!
                let cancel = Arc::new(AtomicBool::new(false));
                let job = Arc::new(SyncSnapshotJob {
                    snapshot: Some(snap.clone()),
                    deadline_at,
                    cancel_flag: cancel.clone(),
                    result: Mutex::new(None),
                    cvar: Condvar::new(),
                });
                {
                    let (lock, cvar) = &*self.workers[0].dispatch;
                    let mut cmd = lock.lock().unwrap();
                    cmd.sync_snapshot = Some(job.clone());
                    cmd.busy = true;
                    cvar.notify_one();
                }
                let res = {
                    let mut guard = job.result.lock().unwrap();
                    loop {
                        if let Some(res) = guard.take() {
                            break res;
                        }
                        if let Some(deadline) = deadline_at {
                            let Some(rem) = deadline.checked_duration_since(Instant::now()) else {
                                cancel.store(true, Ordering::Release);
                                self.ctx.engine.increment_epoch();
                                break Err(KernelError::Timeout);
                            };
                            if rem.is_zero() {
                                cancel.store(true, Ordering::Release);
                                self.ctx.engine.increment_epoch();
                                break Err(KernelError::Timeout);
                            }
                            let (next, _) = job.cvar.wait_timeout(guard, rem).unwrap();
                            guard = next;
                        } else {
                            guard = job.cvar.wait(guard).unwrap();
                        }
                    }
                };
                if matches!(res, Err(KernelError::Timeout)) {
                    self.reap_or_detach_busy_workers(Duration::from_millis(20));
                }
                res?;
            }
        }
        if let Some(deadline) = deadline_at
            && Instant::now() >= deadline
        {
            return Err(KernelError::Timeout);
        }
        Ok(())
    }

    fn run_fixed_direct(
        &mut self,
        input_words: &[u32],
        count: usize,
        snapshot: Option<&Arc<PreparedContextSnapshot>>,
        storage: &Arc<DisjointStorage>,
        deadline_at: Option<Instant>,
    ) -> std::result::Result<(), KernelError> {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut start = 0;
        while start < count {
            let end = (start + self.max_chunk_items).min(count);
            let range = WorkRange { start_item: start, end_item: end, retried_fresh: false };
            execute_fixed_chunk_direct(
                &self.ctx,
                &mut self.direct_guest,
                snapshot,
                input_words,
                range,
                storage,
                deadline_at,
                &cancel,
            )?;
            start = end;
        }
        Ok(())
    }

    fn run_variable_direct(
        &mut self,
        input_words: &[u32],
        count: usize,
        snapshot: Option<&Arc<PreparedContextSnapshot>>,
        deadline_at: Option<Instant>,
    ) -> std::result::Result<Vec<Vec<u32>>, KernelError> {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut pending = BinaryHeap::new();
        let mut start = 0;
        while start < count {
            let end = (start + self.max_chunk_items).min(count);
            pending.push(Reverse(WorkRange { start_item: start, end_item: end, retried_fresh: false }));
            start = end;
        }
        let mut settled_bytes = 0usize;
        let mut chunks = Vec::new();
        while let Some(Reverse(range)) = pending.pop() {
            match execute_variable_chunk_direct(
                &self.ctx,
                &mut self.direct_guest,
                snapshot,
                input_words,
                range,
                settled_bytes,
                deadline_at,
                &cancel,
            )? {
                DirectVarOutcome::Chunk(words) => {
                    settled_bytes += words.len() * 4;
                    self.ctx.stats.record_peak(settled_bytes);
                    chunks.push(words);
                }
                DirectVarOutcome::Bisect => {
                    self.ctx.stats.bisections.fetch_add(1, Ordering::Relaxed);
                    let mid = range.start_item + range.len() / 2;
                    pending.push(Reverse(WorkRange { start_item: mid, end_item: range.end_item, retried_fresh: false }));
                    pending.push(Reverse(WorkRange { start_item: range.start_item, end_item: mid, retried_fresh: false }));
                }
            }
        }
        Ok(chunks)
    }

    fn run_batch_parallel(
        &mut self,
        mode: BatchMode,
        input_words: &[u32],
        count: usize,
        snapshot: Option<Arc<PreparedContextSnapshot>>,
        fixed_storage: Option<Arc<DisjointStorage>>,
        deadline_at: Option<Instant>,
    ) -> std::result::Result<Vec<Vec<u32>>, KernelError> {
        self.ensure_workers()?;
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let batch = Arc::new(BatchShared {
            mode,
            input: Arc::from(input_words),
            snapshot,
            fixed_storage,
            deadline_at,
            cancel_flag: cancel_flag.clone(),
            state: Mutex::new(BatchCoord::new(
                count,
                self.max_chunk_items,
                self.ctx.config.max_total_output_bytes,
                mode,
            )),
            cvar: Condvar::new(),
        });

        let active_count = self.workers.len();
        {
            let mut guard = batch.state.lock().unwrap();
            guard.active_workers_on_batch = active_count;
        }
        for slot in &self.workers {
            let (lock, cvar) = &*slot.dispatch;
            let mut cmd = lock.lock().unwrap();
            cmd.batch = Some(batch.clone());
            cmd.busy = true;
            cvar.notify_one();
        }

        // Standard predicate loop over `cvar.wait_timeout`!
        let timed_out = {
            let mut guard = batch.state.lock().unwrap();
            loop {
                if guard.is_done() && (guard.active_workers_on_batch == 0 || !guard.has_active_below_failure()) {
                    break guard.timed_out;
                }
                if let Some(deadline) = deadline_at {
                    let Some(rem) = deadline.checked_duration_since(Instant::now()) else {
                        guard.timed_out = true;
                        cancel_flag.store(true, Ordering::Release);
                        self.ctx.engine.increment_epoch();
                        batch.cvar.notify_all();
                        break true;
                    };
                    if rem.is_zero() {
                        guard.timed_out = true;
                        cancel_flag.store(true, Ordering::Release);
                        self.ctx.engine.increment_epoch();
                        batch.cvar.notify_all();
                        break true;
                    }
                    let (next, _) = batch.cvar.wait_timeout(guard, rem).unwrap();
                    guard = next;
                } else {
                    guard = batch.cvar.wait(guard).unwrap();
                }
            }
        };

        if timed_out {
            cancel_flag.store(true, Ordering::Release);
            self.ctx.engine.increment_epoch();
            batch.cvar.notify_all();
            self.reap_or_detach_busy_workers(Duration::from_millis(20));
            return Err(KernelError::Timeout);
        }

        // Wait for remaining speculative workers (e.g. running >= earliest_failed_item) to finish their
        // bounded cleanup or detach them if deadline expires.
        {
            let mut guard = batch.state.lock().unwrap();
            while guard.active_workers_on_batch > 0 {
                if let Some(deadline) = deadline_at {
                    let Some(rem) = deadline.checked_duration_since(Instant::now()) else {
                        guard.timed_out = true;
                        cancel_flag.store(true, Ordering::Release);
                        self.ctx.engine.increment_epoch();
                        batch.cvar.notify_all();
                        drop(guard);
                        self.reap_or_detach_busy_workers(Duration::from_millis(20));
                        return Err(KernelError::Timeout);
                    };
                    let (next, _) = batch.cvar.wait_timeout(guard, rem).unwrap();
                    guard = next;
                } else {
                    guard = batch.cvar.wait(guard).unwrap();
                }
            }
        }

        if let Some(deadline) = deadline_at
            && Instant::now() >= deadline
        {
            return Err(KernelError::Timeout);
        }

        let mut guard = batch.state.lock().unwrap();
        if guard.timed_out {
            return Err(KernelError::Timeout);
        }
        if guard.earliest_failed_item != usize::MAX {
            let err_item = guard.earliest_failed_item;
            if let Some(ChunkState::PrimaryError { error }) = guard.chunks.remove(&err_item) {
                return Err(error);
            }
            return Err(KernelError::Internal(format!("missing error state at item {err_item}")));
        }

        if mode == BatchMode::Fixed {
            return Ok(Vec::new());
        }
        let mut out_chunks = Vec::with_capacity(guard.chunks.len());
        let keys: Vec<usize> = guard.chunks.keys().copied().collect();
        for key in keys {
            if let Some(ChunkState::VariableSettled { words, .. }) = guard.chunks.remove(&key) {
                out_chunks.push(words);
            }
        }
        Ok(out_chunks)
    }

    fn reap_or_detach_busy_workers(&mut self, grace: Duration) {
        let deadline = Instant::now() + grace;
        let mut kept = Vec::with_capacity(self.workers.len());
        for mut slot in self.workers.drain(..) {
            let (lock, cvar) = &*slot.dispatch;
            let mut cmd = lock.lock().unwrap();
            while cmd.busy {
                let Some(rem) = deadline.checked_duration_since(Instant::now()) else {
                    break;
                };
                if rem.is_zero() {
                    break;
                }
                let (next, res) = cvar.wait_timeout(cmd, rem).unwrap();
                cmd = next;
                if res.timed_out() {
                    break;
                }
            }
            let still_busy = cmd.busy;
            if still_busy {
                cmd.shutdown = true;
            }
            drop(cmd);
            if still_busy {
                if !slot.retired_flag.swap(true, Ordering::AcqRel) {
                    self.retired_in_flight.fetch_add(1, Ordering::AcqRel);
                }
                // Detached thread will exit once Wasm returns and decrement `retired_in_flight`.
                drop(slot.handle.take());
            } else {
                kept.push(slot);
            }
        }
        self.workers = kept;
    }
}

impl Drop for KernelPool {
    fn drop(&mut self) {
        self.close();
    }
}

fn spawn_worker_thread(
    ctx: PoolContext,
    retired_in_flight: Arc<AtomicUsize>,
    worker_idx: usize,
) -> std::result::Result<WorkerSlot, KernelError> {
    let dispatch = Arc::new((
        Mutex::new(WorkerCommand { batch: None, sync_snapshot: None, shutdown: false, busy: false }),
        Condvar::new(),
    ));
    let retired_flag = Arc::new(AtomicBool::new(false));
    let t_dispatch = dispatch.clone();
    let t_retired_flag = retired_flag.clone();
    let t_retired_count = retired_in_flight;

    let handle = std::thread::Builder::new()
        .name(format!("blot-kernel-{worker_idx}"))
        .stack_size(GUEST_STACK_BYTES + (2 << 20))
        .spawn(move || {
            let mut guest_slot: Option<WorkerGuest> = None;
            loop {
                let (batch, sync_job) = {
                    let (lock, cvar) = &*t_dispatch;
                    let mut cmd = lock.lock().unwrap();
                    while cmd.batch.is_none() && cmd.sync_snapshot.is_none() && !cmd.shutdown {
                        cmd = cvar.wait(cmd).unwrap();
                    }
                    if cmd.shutdown && cmd.batch.is_none() && cmd.sync_snapshot.is_none() {
                        break;
                    }
                    (cmd.batch.take(), cmd.sync_snapshot.take())
                };
                if let Some(job) = sync_job {
                    let res = ensure_guest_snapshot(
                        &ctx,
                        &mut guest_slot,
                        job.snapshot.as_ref(),
                        job.deadline_at,
                        &job.cancel_flag,
                    );
                    *job.result.lock().unwrap() = Some(res);
                    job.cvar.notify_all();
                }
                if let Some(batch) = batch {
                    run_worker_on_batch(&ctx, &mut guest_slot, &batch);
                }
                let should_exit = {
                    let (lock, cvar) = &*t_dispatch;
                    let mut cmd = lock.lock().unwrap();
                    cmd.busy = false;
                    cvar.notify_all();
                    cmd.shutdown
                };
                if should_exit {
                    break;
                }
            }
            drop(guest_slot);
            if t_retired_flag.load(Ordering::Acquire) {
                t_retired_count.fetch_sub(1, Ordering::AcqRel);
            }
        })
        .map_err(|e| KernelError::Internal(format!("spawning kernel worker thread: {e}")))?;

    Ok(WorkerSlot { dispatch, handle: Some(handle), retired_flag })
}

fn is_timed_out_or_cancelled(
    err: Option<&anyhow::Error>,
    deadline_at: Option<Instant>,
    cancel_flag: &AtomicBool,
) -> bool {
    if cancel_flag.load(Ordering::Acquire) {
        return true;
    }
    if err.is_some_and(is_trap_interrupt) {
        return true;
    }
    if let Some(deadline) = deadline_at
        && Instant::now() >= deadline
    {
        return true;
    }
    false
}

/// Runs `blot:finish_invocation` (+ `blot:reset`) with independent `cleanup_grace`
/// (detached from `batch_cancel_flag`). Returns `Ok(())` iff cleanup succeeded cleanly
/// without any physical OOM flag.
fn run_cleanup(ctx: &PoolContext, wg: &mut WorkerGuest) -> std::result::Result<bool, String> {
    let cleanup_deadline = ctx.cleanup_grace.map(|grace| Instant::now() + grace);
    if let Err(err) = wg.guest.arm_execution(cleanup_deadline, None, None) {
        return Err(format!("{err:#}"));
    }
    if let Err(err) = wg.guest.finish_invocation() {
        return Err(format!("{err:#}"));
    }
    if let Err(err) = wg.guest.reset_arena() {
        return Err(format!("{err:#}"));
    }
    if wg.guest.saw_oom() || wg.guest.arena_oom_flag() != 0 {
        return Ok(false);
    }
    Ok(wg.guest.memory_bytes() < ctx.config.retire_memory_threshold)
}

/// Single unified snapshot installation helper (§5.1).
///
/// Handles fresh `Store` creation, `blot:kernel_clear`-first warm replacement (marking
/// `store_has_prior_history = true`), and single fresh-`Store` retry on flagged physical OOM.
fn ensure_guest_snapshot(
    ctx: &PoolContext,
    guest_slot: &mut Option<WorkerGuest>,
    target: Option<&Arc<PreparedContextSnapshot>>,
    deadline_at: Option<Instant>,
    cancel_flag: &Arc<AtomicBool>,
) -> std::result::Result<(), KernelError> {
    let mut attempt_retried_fresh = false;
    loop {
        if is_timed_out_or_cancelled(None, deadline_at, cancel_flag) {
            return Err(KernelError::Timeout);
        }
        if guest_slot.is_none() {
            let wt_module = ctx.module.wasmtime_module().expect("validated Wasmtime module");
            match WasmtimeGuest::start_controlled(
                &ctx.engine,
                wt_module,
                &ctx.module,
                ctx.config.memory,
                deadline_at,
                Some(cancel_flag.clone()),
                None,
            ) {
                Ok(guest) => {
                    ctx.stats.worker_starts.fetch_add(1, Ordering::Relaxed);
                    *guest_slot = Some(WorkerGuest {
                        guest,
                        installed_snapshot_id: None,
                        store_has_prior_history: false,
                    });
                }
                Err(err) => {
                    if is_timed_out_or_cancelled(Some(&err), deadline_at, cancel_flag) {
                        return Err(KernelError::Timeout);
                    }
                    return Err(KernelError::OutOfMemory { item_index: None, detail: format!("{err:#}") });
                }
            }
        }

        let wg = guest_slot.as_mut().unwrap();
        let Some(snap) = target else {
            if wg.installed_snapshot_id.is_some()
                && let Some(clear_export) = &ctx.manifest.clear_export
            {
                if let Err(err) = wg.guest.arm_execution(deadline_at, Some(cancel_flag.clone()), None) {
                    *guest_slot = None;
                    ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                    return Err(KernelError::ContextTrap { detail: format!("{err:#}") });
                }
                let clear_res = wg
                    .guest
                    .reset_arena()
                    .and_then(|()| wg.guest.call_raw(clear_export, Val::I32(0)))
                    .and_then(|_| wg.guest.finish_invocation().map(|_| ()));
                if let Err(err) = clear_res {
                    *guest_slot = None;
                    ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                    if is_timed_out_or_cancelled(Some(&err), deadline_at, cancel_flag) {
                        return Err(KernelError::Timeout);
                    }
                    return Err(KernelError::ContextTrap { detail: format!("{err:#}") });
                }
                wg.installed_snapshot_id = None;
                wg.store_has_prior_history = true;
            }
            return Ok(());
        };

        if wg.installed_snapshot_id == Some(snap.snapshot_id) {
            return Ok(());
        }

        // Warm-worker replacement: run `blot:kernel_clear` FIRST so old context heap charges
        // drop back to the startup baseline AND mark `store_has_prior_history = true`!
        if wg.installed_snapshot_id.is_some()
            && let Some(clear_export) = &ctx.manifest.clear_export
        {
            if let Err(err) = wg.guest.arm_execution(deadline_at, Some(cancel_flag.clone()), None) {
                *guest_slot = None;
                ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                return Err(KernelError::ContextTrap { detail: format!("{err:#}") });
            }
            let clear_res = wg
                .guest
                .reset_arena()
                .and_then(|()| wg.guest.call_raw(clear_export, Val::I32(0)))
                .and_then(|_| wg.guest.finish_invocation().map(|_| ()));
            if let Err(err) = clear_res {
                *guest_slot = None;
                ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                if is_timed_out_or_cancelled(Some(&err), deadline_at, cancel_flag) {
                    return Err(KernelError::Timeout);
                }
                return Err(KernelError::ContextTrap { detail: format!("{err:#}") });
            }
            wg.installed_snapshot_id = None;
            wg.store_has_prior_history = true;
        }

        let eligible_for_fresh_retry = wg.store_has_prior_history && !attempt_retried_fresh;
        wg.guest.clear_saw_oom();
        let ctx_export = ctx
            .manifest
            .ctx_export
            .as_deref()
            .ok_or(KernelError::ContextNotSupported)?;
        let ctx_schema = ctx
            .manifest
            .ctx_schema
            .as_ref()
            .ok_or(KernelError::ContextNotSupported)?;

        let install_res = (|| -> anyhow::Result<()> {
            wg.guest.arm_execution(deadline_at, Some(cancel_flag.clone()), None)?;
            wg.guest.reset_arena()?;
            let param = encode_context_param(&mut wg.guest, ctx_schema, &snap.words)?;
            wg.guest.call_raw(ctx_export, param)?;
            wg.guest.finish_invocation()?;
            wg.guest.reset_arena()?;
            Ok(())
        })();

        match install_res {
            Ok(()) => {
                if is_timed_out_or_cancelled(None, deadline_at, cancel_flag) {
                    *guest_slot = None;
                    ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                    return Err(KernelError::Timeout);
                }
                wg.installed_snapshot_id = Some(snap.snapshot_id);
                ctx.stats.context_installs.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
            Err(err) => {
                let saw_oom = wg.guest.saw_oom() || wg.guest.arena_oom_flag() != 0;
                *guest_slot = None;
                ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                if is_timed_out_or_cancelled(Some(&err), deadline_at, cancel_flag) {
                    return Err(KernelError::Timeout);
                }
                if saw_oom && eligible_for_fresh_retry {
                    attempt_retried_fresh = true;
                    ctx.stats.fresh_oom_retries.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                if saw_oom {
                    return Err(KernelError::OutOfMemory { item_index: None, detail: format!("{err:#}") });
                }
                return Err(KernelError::ContextTrap { detail: format!("{err:#}") });
            }
        }
    }
}

fn encode_context_param(
    guest: &mut WasmtimeGuest,
    ctx_schema: &crate::WireSchema,
    words: &[u32],
) -> anyhow::Result<Val> {
    if ctx_schema.wire_tag.is_array() {
        let ptr = guest.write_words_packet(words)?;
        Ok(Val::I32(ptr))
    } else {
        let word = *words.first().ok_or_else(|| anyhow::anyhow!("missing scalar context word"))?;
        Ok(match ctx_schema.wire_tag {
            ValueType::F32 => Val::F32(word),
            _ => Val::I32(word.cast_signed()),
        })
    }
}

enum RawRunOutcome {
    Packet { out_ptr: i32, header: u32 },
    RetryFresh,
    TimedOut,
    Failed(KernelError),
}

/// Executes `ensure_guest_snapshot` + `reset_arena` + `write_words_packet` + `call_raw`
/// for one chunk attempt, handling pre-attempt `eligible_for_fresh_retry` and recoverable
/// trap cleanup identically across direct and parallel modes.
#[allow(clippy::too_many_arguments)]
fn run_chunk_wasm(
    ctx: &PoolContext,
    guest_slot: &mut Option<WorkerGuest>,
    snapshot: Option<&Arc<PreparedContextSnapshot>>,
    input_words: &[u32],
    range: WorkRange,
    deadline_at: Option<Instant>,
    cancel_flag: &Arc<AtomicBool>,
) -> RawRunOutcome {
    if let Err(err) = ensure_guest_snapshot(ctx, guest_slot, snapshot, deadline_at, cancel_flag) {
        return match err {
            KernelError::Timeout => RawRunOutcome::TimedOut,
            other => RawRunOutcome::Failed(other),
        };
    }

    let wg = guest_slot.as_mut().expect("guest installed");
    let eligible_for_fresh_retry = wg.store_has_prior_history && !range.retried_fresh;
    wg.guest.clear_saw_oom();

    let in_stride = ctx.manifest.in_schema.stride as usize;
    let chunk_in = &input_words[range.start_item * in_stride..range.end_item * in_stride];

    let arm_and_alloc = (|| -> anyhow::Result<i32> {
        wg.guest.arm_execution(deadline_at, Some(cancel_flag.clone()), None)?;
        wg.guest.reset_arena()?;
        wg.guest.write_words_packet(chunk_in)
    })();

    let in_ptr = match arm_and_alloc {
        Ok(ptr) => ptr,
        Err(err) => {
            let saw_oom = wg.guest.saw_oom() || wg.guest.arena_oom_flag() != 0;
            *guest_slot = None;
            ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
            if is_timed_out_or_cancelled(Some(&err), deadline_at, cancel_flag) {
                return RawRunOutcome::TimedOut;
            }
            if saw_oom && eligible_for_fresh_retry {
                ctx.stats.fresh_oom_retries.fetch_add(1, Ordering::Relaxed);
                return RawRunOutcome::RetryFresh;
            }
            if saw_oom {
                return RawRunOutcome::Failed(KernelError::OutOfMemory {
                    item_index: Some(range.start_item),
                    detail: format!("{err:#}"),
                });
            }
            return RawRunOutcome::Failed(KernelError::RunTrap {
                item_index: range.start_item,
                detail: format!("{err:#}"),
            });
        }
    };

    match wg.guest.call_raw(&ctx.manifest.name, Val::I32(in_ptr)) {
        Ok(Val::I32(out_ptr)) => {
            if is_timed_out_or_cancelled(None, deadline_at, cancel_flag) {
                *guest_slot = None;
                ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                return RawRunOutcome::TimedOut;
            }
            match wg.guest.inspect_packet_header(out_ptr) {
                Ok(header) => RawRunOutcome::Packet { out_ptr, header },
                Err(err) => {
                    *guest_slot = None;
                    ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                    RawRunOutcome::Failed(KernelError::RunTrap {
                        item_index: range.start_item,
                        detail: format!("{err:#}"),
                    })
                }
            }
        }
        Ok(_) => {
            *guest_slot = None;
            ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
            RawRunOutcome::Failed(KernelError::RunTrap {
                item_index: range.start_item,
                detail: "kernel returned non-pointer ABI value".into(),
            })
        }
        Err(err) => {
            let saw_oom = wg.guest.saw_oom() || wg.guest.arena_oom_flag() != 0;
            if is_timed_out_or_cancelled(Some(&err), deadline_at, cancel_flag) {
                *guest_slot = None;
                ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                return RawRunOutcome::TimedOut;
            }
            if saw_oom {
                *guest_slot = None;
                ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                if eligible_for_fresh_retry {
                    ctx.stats.fresh_oom_retries.fetch_add(1, Ordering::Relaxed);
                    return RawRunOutcome::RetryFresh;
                }
                return RawRunOutcome::Failed(KernelError::OutOfMemory {
                    item_index: Some(range.start_item),
                    detail: format!("{err:#}"),
                });
            }
            if is_unrecoverable_trap(&err) {
                *guest_slot = None;
                ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
            } else {
                // Recoverable deterministic logical trap: unwind via `blot:finish_invocation` with cleanup grace!
                match run_cleanup(ctx, wg) {
                    Ok(keep_warm) if keep_warm => {
                        wg.store_has_prior_history = true;
                    }
                    _ => {
                        *guest_slot = None;
                        ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            RawRunOutcome::Failed(KernelError::RunTrap {
                item_index: range.start_item,
                detail: format!("{err:#}"),
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_fixed_chunk_direct(
    ctx: &PoolContext,
    guest_slot: &mut Option<WorkerGuest>,
    snapshot: Option<&Arc<PreparedContextSnapshot>>,
    input_words: &[u32],
    mut range: WorkRange,
    storage: &Arc<DisjointStorage>,
    deadline_at: Option<Instant>,
    cancel_flag: &Arc<AtomicBool>,
) -> std::result::Result<(), KernelError> {
    let out_stride = ctx.manifest.out_schema.stride as usize;
    loop {
        match run_chunk_wasm(ctx, guest_slot, snapshot, input_words, range, deadline_at, cancel_flag) {
            RawRunOutcome::RetryFresh => {
                range.retried_fresh = true;
                continue;
            }
            RawRunOutcome::TimedOut => return Err(KernelError::Timeout),
            RawRunOutcome::Failed(err) => return Err(err),
            RawRunOutcome::Packet { out_ptr, header } => {
                let expected_words = range.len() * out_stride;
                let wg = guest_slot.as_mut().unwrap();
                if header as usize != expected_words {
                    *guest_slot = None;
                    ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                    return Err(KernelError::RunTrap {
                        item_index: range.start_item,
                        detail: format!("fixed kernel returned {header} words, expected {expected_words}"),
                    });
                }
                let start_word = range.start_item * out_stride;
                let end_word = range.end_item * out_stride;
                // SAFETY: direct mode processes ranges sequentially and disjointly.
                let copy_res = unsafe {
                    storage.write_disjoint(start_word, end_word, |dst| {
                        wg.guest.read_packet_words_into(out_ptr, expected_words, dst)
                    })
                };
                if let Err(err) = copy_res {
                    *guest_slot = None;
                    ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                    return Err(KernelError::RunTrap {
                        item_index: range.start_item,
                        detail: format!("{err:#}"),
                    });
                }
                match run_cleanup(ctx, wg) {
                    Ok(keep_warm) => {
                        if keep_warm {
                            wg.store_has_prior_history = true;
                        } else {
                            *guest_slot = None;
                            ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                        }
                        ctx.stats.chunks_completed.fetch_add(1, Ordering::Relaxed);
                        return Ok(());
                    }
                    Err(detail) => {
                        *guest_slot = None;
                        ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                        return Err(KernelError::CleanupTrap { item_index: range.start_item, detail });
                    }
                }
            }
        }
    }
}

enum DirectVarOutcome {
    Chunk(Vec<u32>),
    Bisect,
}

#[allow(clippy::too_many_arguments)]
fn execute_variable_chunk_direct(
    ctx: &PoolContext,
    guest_slot: &mut Option<WorkerGuest>,
    snapshot: Option<&Arc<PreparedContextSnapshot>>,
    input_words: &[u32],
    mut range: WorkRange,
    settled_bytes: usize,
    deadline_at: Option<Instant>,
    cancel_flag: &Arc<AtomicBool>,
) -> std::result::Result<DirectVarOutcome, KernelError> {
    loop {
        match run_chunk_wasm(ctx, guest_slot, snapshot, input_words, range, deadline_at, cancel_flag) {
            RawRunOutcome::RetryFresh => {
                range.retried_fresh = true;
                continue;
            }
            RawRunOutcome::TimedOut => return Err(KernelError::Timeout),
            RawRunOutcome::Failed(err) => return Err(err),
            RawRunOutcome::Packet { out_ptr, header } => {
                let wg = guest_slot.as_mut().unwrap();
                if header == u32::MAX {
                    // Unified `[0xffffffff]` transition: run `blot:finish_invocation` FIRST!
                    let cleanup = run_cleanup(ctx, wg);
                    match &cleanup {
                        Ok(true) => wg.store_has_prior_history = true,
                        _ => {
                            *guest_slot = None;
                            ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    if range.len() == 1 {
                        return Err(KernelError::SingleItemOutputOverflow { item_index: range.start_item });
                    }
                    return match cleanup {
                        Ok(_) => Ok(DirectVarOutcome::Bisect),
                        Err(detail) => Err(KernelError::CleanupTrap { item_index: range.start_item, detail }),
                    };
                }
                let words_len = header as usize;
                let result_bytes = words_len.saturating_mul(4);
                let required = settled_bytes.saturating_add(result_bytes);
                if required > ctx.config.max_total_output_bytes {
                    // Discard without copying! Run cleanup; OutputBudgetExceeded wins same-start_item precedence.
                    match run_cleanup(ctx, wg) {
                        Ok(true) => wg.store_has_prior_history = true,
                        _ => {
                            *guest_slot = None;
                            ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    return Err(KernelError::OutputBudgetExceeded {
                        required_bytes: required,
                        limit_bytes: ctx.config.max_total_output_bytes,
                    });
                }
                let words = match wg.guest.read_packet_words_vec(out_ptr, words_len) {
                    Ok(w) => w,
                    Err(err) => {
                        *guest_slot = None;
                        ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                        return Err(KernelError::RunTrap {
                            item_index: range.start_item,
                            detail: format!("{err:#}"),
                        });
                    }
                };
                match run_cleanup(ctx, wg) {
                    Ok(keep_warm) => {
                        if keep_warm {
                            wg.store_has_prior_history = true;
                        } else {
                            *guest_slot = None;
                            ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                        }
                        ctx.stats.chunks_completed.fetch_add(1, Ordering::Relaxed);
                        return Ok(DirectVarOutcome::Chunk(words));
                    }
                    Err(detail) => {
                        drop(words);
                        *guest_slot = None;
                        ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                        return Err(KernelError::CleanupTrap { item_index: range.start_item, detail });
                    }
                }
            }
        }
    }
}

fn run_worker_on_batch(ctx: &PoolContext, guest_slot: &mut Option<WorkerGuest>, batch: &Arc<BatchShared>) {
    loop {
        let range = {
            let mut guard = batch.state.lock().unwrap();
            if guard.timed_out || batch.cancel_flag.load(Ordering::Acquire) {
                break;
            }
            match guard.next_work() {
                Some(r) => r,
                None => break,
            }
        };
        execute_chunk_parallel(ctx, guest_slot, batch, range);
    }

    let mut guard = batch.state.lock().unwrap();
    guard.active_workers_on_batch = guard.active_workers_on_batch.saturating_sub(1);
    batch.cvar.notify_all();
}

fn execute_chunk_parallel(
    ctx: &PoolContext,
    guest_slot: &mut Option<WorkerGuest>,
    batch: &Arc<BatchShared>,
    mut range: WorkRange,
) {
    loop {
        match run_chunk_wasm(
            ctx,
            guest_slot,
            batch.snapshot.as_ref(),
            &batch.input,
            range,
            batch.deadline_at,
            &batch.cancel_flag,
        ) {
            RawRunOutcome::RetryFresh => {
                range.retried_fresh = true;
                continue;
            }
            RawRunOutcome::TimedOut => {
                let mut guard = batch.state.lock().unwrap();
                guard.active_ranges.remove(&range.start_item);
                guard.timed_out = true;
                batch.cancel_flag.store(true, Ordering::Release);
                batch.cvar.notify_all();
                return;
            }
            RawRunOutcome::Failed(error) => {
                let mut guard = batch.state.lock().unwrap();
                guard.active_ranges.remove(&range.start_item);
                guard.record_primary_error(range.start_item, error);
                guard.advance_frontier();
                batch.cvar.notify_all();
                return;
            }
            RawRunOutcome::Packet { out_ptr, header } => {
                let wg = guest_slot.as_mut().unwrap();
                if batch.mode == BatchMode::Fixed {
                    let out_stride = ctx.manifest.out_schema.stride as usize;
                    let expected_words = range.len() * out_stride;
                    let storage = batch.fixed_storage.as_ref().expect("fixed storage");
                    if header as usize != expected_words {
                        *guest_slot = None;
                        ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                        let mut guard = batch.state.lock().unwrap();
                        guard.active_ranges.remove(&range.start_item);
                        guard.record_primary_error(
                            range.start_item,
                            KernelError::RunTrap {
                                item_index: range.start_item,
                                detail: format!("fixed kernel returned {header} words, expected {expected_words}"),
                            },
                        );
                        guard.advance_frontier();
                        batch.cvar.notify_all();
                        return;
                    }
                    // Skip copy if already past earliest_failed_item, still run cleanup.
                    let should_copy = {
                        let guard = batch.state.lock().unwrap();
                        range.start_item < guard.earliest_failed_item && !guard.timed_out
                    };
                    if should_copy {
                        let start_word = range.start_item * out_stride;
                        let end_word = range.end_item * out_stride;
                        // SAFETY: coordinator partitioned `0..total_items` into disjoint fixed ranges.
                        let copy_res = unsafe {
                            storage.write_disjoint(start_word, end_word, |dst| {
                                wg.guest.read_packet_words_into(out_ptr, expected_words, dst)
                            })
                        };
                        if let Err(err) = copy_res {
                            *guest_slot = None;
                            ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                            let mut guard = batch.state.lock().unwrap();
                            guard.active_ranges.remove(&range.start_item);
                            guard.record_primary_error(
                                range.start_item,
                                KernelError::RunTrap {
                                    item_index: range.start_item,
                                    detail: format!("{err:#}"),
                                },
                            );
                            guard.advance_frontier();
                            batch.cvar.notify_all();
                            return;
                        }
                    }
                    match run_cleanup(ctx, wg) {
                        Ok(keep_warm) => {
                            if keep_warm {
                                wg.store_has_prior_history = true;
                            } else {
                                *guest_slot = None;
                                ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                            }
                            ctx.stats.chunks_completed.fetch_add(1, Ordering::Relaxed);
                            let mut guard = batch.state.lock().unwrap();
                            guard.active_ranges.remove(&range.start_item);
                            if range.start_item < guard.earliest_failed_item {
                                guard.chunks.insert(range.start_item, ChunkState::FixedDone { end_item: range.end_item });
                                guard.advance_frontier();
                            }
                            batch.cvar.notify_all();
                            return;
                        }
                        Err(detail) => {
                            *guest_slot = None;
                            ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                            let mut guard = batch.state.lock().unwrap();
                            guard.active_ranges.remove(&range.start_item);
                            guard.record_cleanup_failed(range, expected_words * 4, detail);
                            guard.advance_frontier();
                            batch.cvar.notify_all();
                            return;
                        }
                    }
                }

                // Variable output mode!
                if header == u32::MAX {
                    // Unified `[0xffffffff]` transition: run cleanup FIRST!
                    let cleanup = run_cleanup(ctx, wg);
                    match &cleanup {
                        Ok(true) => wg.store_has_prior_history = true,
                        _ => {
                            *guest_slot = None;
                            ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    let mut guard = batch.state.lock().unwrap();
                    guard.active_ranges.remove(&range.start_item);
                    if range.len() == 1 {
                        guard.record_primary_error(
                            range.start_item,
                            KernelError::SingleItemOutputOverflow { item_index: range.start_item },
                        );
                        guard.advance_frontier();
                    } else if let Err(detail) = cleanup {
                        // OVERFLOW_CLEANUP_FAIL on len > 1 -> CleanupTrap WITHOUT bisecting!
                        guard.record_primary_error(
                            range.start_item,
                            KernelError::CleanupTrap { item_index: range.start_item, detail },
                        );
                        guard.advance_frontier();
                    } else if range.start_item < guard.earliest_failed_item {
                        // OVERFLOW_OK on len > 1 -> bisect into min-heap!
                        ctx.stats.bisections.fetch_add(1, Ordering::Relaxed);
                        let mid = range.start_item + range.len() / 2;
                        guard.pending.push(Reverse(WorkRange {
                            start_item: mid,
                            end_item: range.end_item,
                            retried_fresh: false,
                        }));
                        guard.pending.push(Reverse(WorkRange {
                            start_item: range.start_item,
                            end_item: mid,
                            retried_fresh: false,
                        }));
                    }
                    batch.cvar.notify_all();
                    return;
                }

                let words_len = header as usize;
                let result_bytes = words_len.saturating_mul(4);

                // Pre-copy Wasm-memory size admission BEFORE allocating any host Vec!
                let decision = {
                    let mut guard = batch.state.lock().unwrap();
                    if guard.timed_out || range.start_item >= guard.earliest_failed_item {
                        AdmitDecision::Discard
                    } else if range.start_item == guard.frontier_item {
                        let required = guard.settled_prefix_bytes.saturating_add(result_bytes);
                        if required > guard.max_total_output_bytes {
                            guard.record_over_budget_frontier(range.start_item, required);
                            guard.advance_frontier();
                            batch.cvar.notify_all();
                            AdmitDecision::Discard
                        } else {
                            guard.pending_frontier_copy = true;
                            loop {
                                guard.evict_higher_buffered(range.start_item, result_bytes);
                                if guard.timed_out || batch.cancel_flag.load(Ordering::Acquire) {
                                    guard.pending_frontier_copy = false;
                                    break AdmitDecision::Discard;
                                }
                                if guard.settled_prefix_bytes
                                    + guard.higher_buffered_bytes
                                    + guard.in_flight_copy_bytes
                                    + result_bytes
                                    <= guard.max_total_output_bytes
                                {
                                    guard.pending_frontier_copy = false;
                                    guard.in_flight_copy_bytes += result_bytes;
                                    ctx.stats.record_peak(guard.current_charged_bytes());
                                    break AdmitDecision::Copy;
                                }
                                if let Some(deadline) = batch.deadline_at {
                                    let Some(rem) = deadline.checked_duration_since(Instant::now()) else {
                                        guard.timed_out = true;
                                        guard.pending_frontier_copy = false;
                                        batch.cancel_flag.store(true, Ordering::Release);
                                        break AdmitDecision::Discard;
                                    };
                                    if rem.is_zero() {
                                        guard.timed_out = true;
                                        guard.pending_frontier_copy = false;
                                        batch.cancel_flag.store(true, Ordering::Release);
                                        break AdmitDecision::Discard;
                                    }
                                    let (next, _) = batch.cvar.wait_timeout(guard, rem).unwrap();
                                    guard = next;
                                } else {
                                    guard = batch.cvar.wait(guard).unwrap();
                                }
                            }
                        }
                    } else if guard.pending_frontier_copy {
                        AdmitDecision::ParkAndRequeue
                    } else {
                        // Speculative higher range > frontier_item
                        let required = guard.settled_prefix_bytes.saturating_add(result_bytes);
                        if required > guard.max_total_output_bytes {
                            guard.chunks.insert(
                                range.start_item,
                                ChunkState::OverBudgetHint { result_bytes },
                            );
                            AdmitDecision::Discard
                        } else {
                            guard.evict_higher_buffered(range.start_item, result_bytes);
                            if guard.settled_prefix_bytes
                                + guard.higher_buffered_bytes
                                + guard.in_flight_copy_bytes
                                + result_bytes
                                <= guard.max_total_output_bytes
                            {
                                guard.in_flight_copy_bytes += result_bytes;
                                ctx.stats.record_peak(guard.current_charged_bytes());
                                AdmitDecision::Copy
                            } else {
                                AdmitDecision::ParkAndRequeue
                            }
                        }
                    }
                };

                match decision {
                    AdmitDecision::Copy => {
                        let words = match wg.guest.read_packet_words_vec(out_ptr, words_len) {
                            Ok(w) => w,
                            Err(err) => {
                                *guest_slot = None;
                                ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                                let mut guard = batch.state.lock().unwrap();
                                guard.in_flight_copy_bytes = guard.in_flight_copy_bytes.saturating_sub(result_bytes);
                                guard.active_ranges.remove(&range.start_item);
                                guard.record_primary_error(
                                    range.start_item,
                                    KernelError::RunTrap {
                                        item_index: range.start_item,
                                        detail: format!("{err:#}"),
                                    },
                                );
                                guard.advance_frontier();
                                batch.cvar.notify_all();
                                return;
                            }
                        };
                        match run_cleanup(ctx, wg) {
                            Ok(keep_warm) => {
                                if keep_warm {
                                    wg.store_has_prior_history = true;
                                } else {
                                    *guest_slot = None;
                                    ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                                }
                                ctx.stats.chunks_completed.fetch_add(1, Ordering::Relaxed);
                                let mut guard = batch.state.lock().unwrap();
                                guard.in_flight_copy_bytes = guard.in_flight_copy_bytes.saturating_sub(result_bytes);
                                guard.active_ranges.remove(&range.start_item);
                                if range.start_item >= guard.earliest_failed_item || guard.timed_out {
                                    drop(words);
                                } else if guard.pending_frontier_copy && range.start_item > guard.frontier_item {
                                    // Frontier is waiting for in-flight copy bytes to drain: drop and re-queue!
                                    drop(words);
                                    guard.pending.push(Reverse(WorkRange {
                                        start_item: range.start_item,
                                        end_item: range.end_item,
                                        retried_fresh: false,
                                    }));
                                } else {
                                    guard.higher_buffered_bytes += result_bytes;
                                    guard.chunks.insert(
                                        range.start_item,
                                        ChunkState::VariableBuffered { end_item: range.end_item, words },
                                    );
                                    guard.advance_frontier();
                                }
                                batch.cvar.notify_all();
                                return;
                            }
                            Err(detail) => {
                                drop(words);
                                *guest_slot = None;
                                ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                                let mut guard = batch.state.lock().unwrap();
                                guard.in_flight_copy_bytes = guard.in_flight_copy_bytes.saturating_sub(result_bytes);
                                guard.active_ranges.remove(&range.start_item);
                                guard.record_cleanup_failed(range, result_bytes, detail);
                                guard.advance_frontier();
                                batch.cvar.notify_all();
                                return;
                            }
                        }
                    }
                    AdmitDecision::Discard => {
                        match run_cleanup(ctx, wg) {
                            Ok(keep_warm) => {
                                if keep_warm {
                                    wg.store_has_prior_history = true;
                                } else {
                                    *guest_slot = None;
                                    ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                                }
                                let mut guard = batch.state.lock().unwrap();
                                guard.active_ranges.remove(&range.start_item);
                                guard.advance_frontier();
                                batch.cvar.notify_all();
                                return;
                            }
                            Err(detail) => {
                                *guest_slot = None;
                                ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                                let mut guard = batch.state.lock().unwrap();
                                guard.active_ranges.remove(&range.start_item);
                                guard.record_cleanup_failed(range, result_bytes, detail);
                                guard.advance_frontier();
                                batch.cvar.notify_all();
                                return;
                            }
                        }
                    }
                    AdmitDecision::ParkAndRequeue => {
                        match run_cleanup(ctx, wg) {
                            Ok(keep_warm) => {
                                if keep_warm {
                                    wg.store_has_prior_history = true;
                                } else {
                                    *guest_slot = None;
                                    ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            Err(detail) => {
                                *guest_slot = None;
                                ctx.stats.worker_retirements.fetch_add(1, Ordering::Relaxed);
                                let mut guard = batch.state.lock().unwrap();
                                guard.active_ranges.remove(&range.start_item);
                                guard.record_cleanup_failed(range, result_bytes, detail);
                                guard.advance_frontier();
                                batch.cvar.notify_all();
                                return;
                            }
                        }
                        let mut guard = batch.state.lock().unwrap();
                        guard.active_ranges.remove(&range.start_item);
                        if range.start_item < guard.earliest_failed_item && !guard.timed_out {
                            guard.pending.push(Reverse(WorkRange {
                                start_item: range.start_item,
                                end_item: range.end_item,
                                retried_fresh: false,
                            }));
                        }
                        let observed_frontier = guard.frontier_item;
                        // Park ONLY while `frontier_item` is actively running (so frontier is never starved!).
                        while !guard.is_done()
                            && !batch.cancel_flag.load(Ordering::Acquire)
                            && guard.frontier_item == observed_frontier
                            && guard.active_ranges.contains_key(&observed_frontier)
                        {
                            if let Some(deadline) = batch.deadline_at {
                                let Some(rem) = deadline.checked_duration_since(Instant::now()) else {
                                    guard.timed_out = true;
                                    batch.cancel_flag.store(true, Ordering::Release);
                                    break;
                                };
                                if rem.is_zero() {
                                    guard.timed_out = true;
                                    batch.cancel_flag.store(true, Ordering::Release);
                                    break;
                                }
                                let (next, _) = batch.cvar.wait_timeout(guard, rem).unwrap();
                                guard = next;
                            } else {
                                guard = batch.cvar.wait(guard).unwrap();
                            }
                        }
                        batch.cvar.notify_all();
                        return;
                    }
                }
            }
        }
    }
}
