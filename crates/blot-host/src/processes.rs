//! Deterministic standalone isolated `Processes` scheduler for ABI 8 `entry process`
//! modules (`blot2/tasks/CONCURRENCY_VNEXT.md` §5.2).
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use wasmtime::{Engine, Val};

use crate::{
    GUEST_STACK_BYTES, GuestModule, Limits, ProcessManifest, ValueType, WasmtimeGuest, WireSchema, WireSchemaKind,
    ensure_engine_ticker, is_trap_interrupt,
};

static NEXT_SCHEDULER_ID: AtomicU64 = AtomicU64::new(1);

const OVERRUN_JOIN_GRACE: Duration = Duration::from_millis(50);

/// Per-turn Wasmtime fuel limits (Rust-only; rejected in JS/Deno).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FuelPolicy {
    pub init_fuel: u64,
    pub step_fuel: u64,
    pub cleanup_fuel: u64,
}

/// Resource and scheduling policy bound to one `entry process` manifest.
#[derive(Clone, Debug)]
pub struct ProcessPolicy {
    pub name: String,
    pub init_work: u32,
    pub step_work: u32,
    pub rust_wasmtime_fuel: Option<FuelPolicy>,
    pub deadline: Option<Duration>,
    pub memory: Limits,
    pub mailbox: usize,
    pub mailbox_bytes: usize,
    pub max_reply_bytes: usize,
    pub slots: usize,
}

/// Scheduler-scoped, epoch-fenced handle identifying one live process instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProcessHandle {
    pub scheduler_id: u64,
    pub endpoint: u32,
    pub epoch: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessStatus {
    Starting,
    Ready,
    InitializationFailed,
    StepFailed,
    Closed,
    Stale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmitResult {
    Admitted(ProcessHandle),
    UnknownProcess,
    Full,
    InvalidConfig,
    Aborted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SendResult {
    Accepted(u64),
    Starting,
    Full,
    Closed,
    Stale,
    InvalidCommand,
}

/// Wire value exchanged with an `entry process` (`Config`, `Cmd`, or `Reply`).
#[derive(Clone, Debug, PartialEq)]
pub enum ProcessValue {
    Unit,
    U32(u32),
    Bool(bool),
    F32(f32),
    Words(Vec<u32>),
}

impl ProcessValue {
    pub fn from_f32s(floats: &[f32]) -> Self {
        Self::Words(floats.iter().copied().map(f32::to_bits).collect())
    }

    pub fn as_f32s(&self) -> Option<Vec<f32>> {
        match self {
            ProcessValue::Words(words) => Some(words.iter().copied().map(f32::from_bits).collect()),
            _ => None,
        }
    }

    /// Logical wire byte width per §5.2 (`0` for Unit, `4` for 1-word scalars, `words.len() * 4` for arrays/composites).
    pub fn logical_bytes(&self) -> usize {
        match self {
            ProcessValue::Unit => 0,
            ProcessValue::U32(_) | ProcessValue::Bool(_) | ProcessValue::F32(_) => 4,
            ProcessValue::Words(words) => words.len().saturating_mul(4),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ProcessOutcome {
    Done(ProcessValue),
    Cancelled,
    Failed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProcessEvent {
    pub ticket: u64,
    pub outcome: ProcessOutcome,
}

#[derive(Debug)]
pub enum ProcessHostError {
    EngineMismatch,
    WtimeAotUnsupported,
    FuelEngineRequired,
    BoundedEngineRequired,
    InvalidPolicy(String),
    UnknownProcess(String),
    Aborted,
    Overrun { drained_events: BTreeMap<ProcessHandle, VecDeque<ProcessEvent>> },
    Internal(String),
}

impl fmt::Display for ProcessHostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProcessHostError::EngineMismatch => write!(f, "processes engine does not match module engine"),
            ProcessHostError::WtimeAotUnsupported => write!(f, "processes require a Wasmtime module, not wtime-aot"),
            ProcessHostError::FuelEngineRequired => write!(f, "fuel policy requires a fuel-metered Wasmtime engine"),
            ProcessHostError::BoundedEngineRequired => {
                write!(f, "deadline policy requires an epoch-interrupted Wasmtime engine")
            }
            ProcessHostError::InvalidPolicy(msg) => write!(f, "invalid process policy: {msg}"),
            ProcessHostError::UnknownProcess(name) => write!(f, "unknown entry process `{name}`"),
            ProcessHostError::Aborted => write!(f, "process scheduler is permanently aborted after an overrun"),
            ProcessHostError::Overrun { drained_events } => {
                write!(f, "process turn overran its deadline ({} endpoints drained)", drained_events.len())
            }
            ProcessHostError::Internal(msg) => write!(f, "process host internal error: {msg}"),
        }
    }
}

impl std::error::Error for ProcessHostError {}

/// RAII byte lease held by live instances AND detached overrun sibling threads until physical exit.
struct ByteLease {
    gauge: Arc<AtomicUsize>,
    bytes: usize,
}

impl ByteLease {
    fn acquire(gauge: &Arc<AtomicUsize>, bytes: usize) -> Self {
        if bytes > 0 {
            gauge.fetch_add(bytes, Ordering::AcqRel);
        }
        Self { gauge: gauge.clone(), bytes }
    }
}

impl Drop for ByteLease {
    fn drop(&mut self) {
        if self.bytes > 0 {
            self.gauge.fetch_sub(self.bytes, Ordering::AcqRel);
        }
    }
}

struct RegisteredProcess {
    manifest: ProcessManifest,
    policy: ProcessPolicy,
    active_slots: usize,
}

struct PendingCommand {
    ticket: u64,
    cmd: ProcessValue,
    cmd_bytes: usize,
    max_reply_bytes: usize,
}

struct Child {
    handle: ProcessHandle,
    program: usize,
    status: ProcessStatus,
    init_config: Option<ProcessValue>,
    init_config_bytes: usize,
    guest: Option<WasmtimeGuest>,
    memory_lease: Option<ByteLease>,
    pending: VecDeque<PendingCommand>,
    events: VecDeque<ProcessEvent>,
    charged_mailbox_bytes: usize,
}

impl Child {
    fn settle_terminal(&mut self, status: ProcessStatus, pending_outcome: ProcessOutcome, mailbox_gauge: &Arc<AtomicUsize>) {
        self.status = status;
        self.init_config = None;
        if self.init_config_bytes > 0 {
            self.charged_mailbox_bytes = self.charged_mailbox_bytes.saturating_sub(self.init_config_bytes);
            mailbox_gauge.fetch_sub(self.init_config_bytes, Ordering::AcqRel);
            self.init_config_bytes = 0;
        }
        for cmd in self.pending.drain(..) {
            let release = cmd.cmd_bytes.saturating_add(cmd.max_reply_bytes);
            self.charged_mailbox_bytes = self.charged_mailbox_bytes.saturating_sub(release);
            if release > 0 {
                mailbox_gauge.fetch_sub(release, Ordering::AcqRel);
            }
            self.events.push_back(ProcessEvent { ticket: cmd.ticket, outcome: pending_outcome.clone() });
        }
        self.guest = None;
        self.memory_lease = None;
    }
}

enum RingSlot {
    Present(Child),
    InWave(usize), // index into wave_tasks
}

/// Deterministic isolated `entry process` scheduler supporting both serial `advance`
/// and serial-equivalent multi-wave `advance_all_parallel`.
pub struct Processes {
    scheduler_id: u64,
    epoch: u64,
    engine: Engine,
    module: GuestModule,
    transport_budget: usize,
    programs: Vec<RegisteredProcess>,
    children: VecDeque<Child>,
    next_endpoint: u32,
    next_ticket: u64,
    aborted: bool,
    reserved_memory_gauge: Arc<AtomicUsize>,
    reserved_mailbox_gauge: Arc<AtomicUsize>,
}

impl Processes {
    pub fn new(
        engine: &Engine,
        module: &GuestModule,
        policies: Vec<ProcessPolicy>,
    ) -> std::result::Result<Self, ProcessHostError> {
        if !module.abi().data_stacks {
            return Err(ProcessHostError::InvalidPolicy("processes require guest ABI 8".into()));
        }
        let Some(wt_module) = module.wasmtime_module() else {
            return Err(ProcessHostError::WtimeAotUnsupported);
        };
        if !Engine::same(engine, wt_module.engine()) {
            return Err(ProcessHostError::EngineMismatch);
        }
        let any_fuel = policies.iter().any(|p| p.rust_wasmtime_fuel.is_some());
        let any_deadline = policies.iter().any(|p| p.deadline.is_some());
        if any_fuel && !engine.get_consume_fuel() {
            return Err(ProcessHostError::FuelEngineRequired);
        }
        if any_deadline && !engine.get_epoch_interruption() {
            return Err(ProcessHostError::BoundedEngineRequired);
        }
        if engine.get_epoch_interruption() {
            ensure_engine_ticker(engine, Duration::from_millis(2))
                .map_err(|e| ProcessHostError::Internal(format!("starting epoch ticker: {e:#}")))?;
        }
        let mut programs = Vec::with_capacity(policies.len());
        let mut names = std::collections::HashSet::new();
        for policy in policies {
            if !names.insert(policy.name.clone()) {
                return Err(ProcessHostError::InvalidPolicy(format!("duplicate policy for `{}`", policy.name)));
            }
            if policy.init_work == 0 || policy.step_work == 0 || policy.slots == 0 || policy.mailbox == 0 {
                return Err(ProcessHostError::InvalidPolicy(format!(
                    "process `{}` requires positive init_work, step_work, slots, and mailbox",
                    policy.name
                )));
            }
            if policy.memory.memory == 0 || policy.slots.checked_mul(policy.memory.memory).is_none() {
                return Err(ProcessHostError::InvalidPolicy(format!(
                    "process `{}` has invalid memory limit",
                    policy.name
                )));
            }
            if let Some(deadline) = policy.deadline
                && deadline.is_zero()
            {
                return Err(ProcessHostError::InvalidPolicy(format!(
                    "process `{}` deadline must be non-zero",
                    policy.name
                )));
            }
            let manifest = module
                .abi()
                .process(&policy.name)
                .cloned()
                .ok_or_else(|| ProcessHostError::UnknownProcess(policy.name.clone()))?;
            validate_reply_budget(&manifest.reply_schema, policy.max_reply_bytes, &policy.name)?;
            programs.push(RegisteredProcess { manifest, policy, active_slots: 0 });
        }
        let scheduler_id = NEXT_SCHEDULER_ID.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            scheduler_id,
            epoch: 1,
            engine: engine.clone(),
            module: module.clone(),
            transport_budget: module.abi().transport_budget as usize,
            programs,
            children: VecDeque::new(),
            next_endpoint: 1,
            next_ticket: 1,
            aborted: false,
            reserved_memory_gauge: Arc::new(AtomicUsize::new(0)),
            reserved_mailbox_gauge: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn is_aborted(&self) -> bool {
        self.aborted
    }

    pub fn admit(&mut self, name: &str, config: ProcessValue) -> AdmitResult {
        if self.aborted {
            return AdmitResult::Aborted;
        }
        let Some(program_idx) = self.programs.iter().position(|p| p.manifest.name == name) else {
            return AdmitResult::UnknownProcess;
        };
        let prog = &mut self.programs[program_idx];
        if prog.active_slots >= prog.policy.slots {
            return AdmitResult::Full;
        }
        if !validate_value_against_schema(&config, &prog.manifest.config_schema, self.transport_budget) {
            return AdmitResult::InvalidConfig;
        }
        let config_bytes = config.logical_bytes();
        if config_bytes > prog.policy.mailbox_bytes {
            return AdmitResult::InvalidConfig;
        }
        let Some(next_ep) = self.next_endpoint.checked_add(1) else {
            return AdmitResult::Full;
        };
        let endpoint = self.next_endpoint;
        self.next_endpoint = next_ep;
        prog.active_slots += 1;
        if config_bytes > 0 {
            self.reserved_mailbox_gauge.fetch_add(config_bytes, Ordering::AcqRel);
        }
        let handle = ProcessHandle { scheduler_id: self.scheduler_id, endpoint, epoch: self.epoch };
        self.children.push_back(Child {
            handle,
            program: program_idx,
            status: ProcessStatus::Starting,
            init_config: Some(config),
            init_config_bytes: config_bytes,
            guest: None,
            memory_lease: None,
            pending: VecDeque::new(),
            events: VecDeque::new(),
            charged_mailbox_bytes: config_bytes,
        });
        AdmitResult::Admitted(handle)
    }

    fn is_valid_handle(&self, handle: ProcessHandle) -> bool {
        handle.scheduler_id == self.scheduler_id && handle.epoch == self.epoch
    }

    pub fn status(&self, handle: ProcessHandle) -> ProcessStatus {
        if !self.is_valid_handle(handle) {
            return ProcessStatus::Stale;
        }
        self.children.iter().find(|c| c.handle == handle).map_or(ProcessStatus::Closed, |c| c.status)
    }

    pub fn send(&mut self, handle: ProcessHandle, cmd: ProcessValue) -> SendResult {
        if !self.is_valid_handle(handle) {
            return SendResult::Stale;
        }
        if self.aborted {
            return SendResult::Closed;
        }
        let Some(child) = self.children.iter_mut().find(|c| c.handle == handle) else {
            return SendResult::Closed;
        };
        match child.status {
            ProcessStatus::Starting => return SendResult::Starting,
            ProcessStatus::Ready => {}
            _ => return SendResult::Closed,
        }
        let prog = &self.programs[child.program];
        if !validate_value_against_schema(&cmd, &prog.manifest.cmd_schema, self.transport_budget) {
            return SendResult::InvalidCommand;
        }
        if child.pending.len() + child.events.len() >= prog.policy.mailbox {
            return SendResult::Full;
        }
        let cmd_bytes = cmd.logical_bytes();
        let max_reply_bytes = prog.policy.max_reply_bytes;
        let reserve = cmd_bytes.saturating_add(max_reply_bytes);
        if child.charged_mailbox_bytes.saturating_add(reserve) > prog.policy.mailbox_bytes {
            return SendResult::Full;
        }
        let ticket = self.next_ticket;
        let Some(next_ticket) = ticket.checked_add(1) else {
            return SendResult::Full;
        };
        self.next_ticket = next_ticket;
        child.charged_mailbox_bytes += reserve;
        if reserve > 0 {
            self.reserved_mailbox_gauge.fetch_add(reserve, Ordering::AcqRel);
        }
        child.pending.push_back(PendingCommand { ticket, cmd, cmd_bytes, max_reply_bytes });
        SendResult::Accepted(ticket)
    }

    pub fn poll(&mut self, handle: ProcessHandle) -> Option<ProcessEvent> {
        if !self.is_valid_handle(handle) {
            return None;
        }
        let child = self.children.iter_mut().find(|c| c.handle == handle)?;
        let event = child.events.pop_front()?;
        if let ProcessOutcome::Done(ref val) = event.outcome {
            let bytes = val.logical_bytes();
            child.charged_mailbox_bytes = child.charged_mailbox_bytes.saturating_sub(bytes);
            if bytes > 0 {
                self.reserved_mailbox_gauge.fetch_sub(bytes, Ordering::AcqRel);
            }
        }
        Some(event)
    }

    pub fn cancel(&mut self, handle: ProcessHandle, ticket: u64) -> bool {
        if !self.is_valid_handle(handle) || self.aborted {
            return false;
        }
        let Some(child) = self.children.iter_mut().find(|c| c.handle == handle) else {
            return false;
        };
        if !matches!(child.status, ProcessStatus::Starting | ProcessStatus::Ready) {
            return false;
        }
        let Some(pos) = child.pending.iter().position(|c| c.ticket == ticket) else {
            return false;
        };
        let removed = child.pending.remove(pos).unwrap();
        let release = removed.cmd_bytes.saturating_add(removed.max_reply_bytes);
        child.charged_mailbox_bytes = child.charged_mailbox_bytes.saturating_sub(release);
        if release > 0 {
            self.reserved_mailbox_gauge.fetch_sub(release, Ordering::AcqRel);
        }
        child.events.push_back(ProcessEvent { ticket, outcome: ProcessOutcome::Cancelled });
        true
    }

    pub fn close(&mut self, handle: ProcessHandle) {
        if !self.is_valid_handle(handle) {
            return;
        }
        if let Some(child) = self.children.iter_mut().find(|c| c.handle == handle)
            && matches!(child.status, ProcessStatus::Starting | ProcessStatus::Ready)
        {
            child.settle_terminal(ProcessStatus::Closed, ProcessOutcome::Cancelled, &self.reserved_mailbox_gauge);
        }
    }

    pub fn release(&mut self, handle: ProcessHandle) -> bool {
        if !self.is_valid_handle(handle) {
            return false;
        }
        let Some(idx) = self.children.iter().position(|c| {
            c.handle == handle
                && c.events.is_empty()
                && c.pending.is_empty()
                && !matches!(c.status, ProcessStatus::Starting | ProcessStatus::Ready)
        }) else {
            return false;
        };
        let child = self.children.remove(idx).unwrap();
        self.programs[child.program].active_slots = self.programs[child.program].active_slots.saturating_sub(1);
        true
    }

    /// Fences all old handles by advancing `epoch`, tears down all live guests,
    /// and transfers ownership of all undelivered `Done` + terminal `Cancelled` events.
    pub fn restore(&mut self) -> BTreeMap<ProcessHandle, VecDeque<ProcessEvent>> {
        self.epoch = self.epoch.wrapping_add(1);
        let mut drained = BTreeMap::new();
        while let Some(mut child) = self.children.pop_front() {
            if matches!(child.status, ProcessStatus::Starting | ProcessStatus::Ready) {
                child.settle_terminal(ProcessStatus::Closed, ProcessOutcome::Cancelled, &self.reserved_mailbox_gauge);
            } else {
                child.guest = None;
                child.memory_lease = None;
            }
            self.programs[child.program].active_slots = self.programs[child.program].active_slots.saturating_sub(1);
            // Release all transferred `Done` reply_bytes upon ownership transfer!
            for ev in &child.events {
                if let ProcessOutcome::Done(ref val) = ev.outcome {
                    let bytes = val.logical_bytes();
                    child.charged_mailbox_bytes = child.charged_mailbox_bytes.saturating_sub(bytes);
                    if bytes > 0 {
                        self.reserved_mailbox_gauge.fetch_sub(bytes, Ordering::AcqRel);
                    }
                }
            }
            if !child.events.is_empty() {
                drained.insert(child.handle, child.events);
            }
        }
        drained
    }

    /// Observed Wasm linear-memory bytes across currently live guests (including unexported memories).
    pub fn memory_bytes(&self) -> usize {
        self.children.iter().filter_map(|c| c.guest.as_ref()).map(WasmtimeGuest::memory_bytes).sum()
    }

    /// Total resident memory limit bytes currently reserved across live and detached overrun guests.
    pub fn reserved_memory_bytes(&self) -> usize {
        self.reserved_memory_gauge.load(Ordering::Acquire)
    }

    /// Total mailbox bytes currently reserved across live and detached overrun commands/replies.
    pub fn reserved_mailbox_bytes(&self) -> usize {
        self.reserved_mailbox_gauge.load(Ordering::Acquire)
    }

    /// Runs complete guest invocations serially within one shared work budget.
    pub fn advance(&mut self, budget: u32) -> std::result::Result<u32, ProcessHostError> {
        if self.aborted {
            return Err(ProcessHostError::Aborted);
        }
        let mut spent = 0u32;
        let mut skipped = 0usize;
        let mut deferred: Option<u32> = None;
        while spent < budget && skipped < self.children.len() {
            let mut child = self.children.pop_front().expect("unvisited process");
            let prog = &self.programs[child.program];
            let cost = match child.status {
                ProcessStatus::Starting => prog.policy.init_work,
                ProcessStatus::Ready if !child.pending.is_empty() => prog.policy.step_work,
                _ => 0,
            };
            if cost == 0 || cost > budget - spent {
                if cost != 0 && cost <= budget && deferred.is_none() {
                    deferred = Some(child.handle.endpoint);
                }
                skipped += 1;
                self.children.push_back(child);
                continue;
            }
            skipped = 0;
            spent += cost;
            let cancel_flag = Arc::new(AtomicBool::new(false));
            let deadline_at = prog.policy.deadline.map(|d| Instant::now() + d);

            if child.status == ProcessStatus::Starting {
                let config = child.init_config.take().expect("starting child has config");
                // Reserve resident memory BEFORE Wasm instantiation & Wasm `start`!
                let mem_lease = ByteLease::acquire(&self.reserved_memory_gauge, prog.policy.memory.memory);
                let outcome = run_process_init(
                    &self.engine,
                    &self.module,
                    &prog.manifest,
                    &prog.policy,
                    &config,
                    deadline_at,
                    &cancel_flag,
                );
                match outcome {
                    InitOutcome::Ok(guest) => {
                        child.guest = Some(guest);
                        child.memory_lease = Some(mem_lease);
                        child.status = ProcessStatus::Ready;
                        if child.init_config_bytes > 0 {
                            child.charged_mailbox_bytes =
                                child.charged_mailbox_bytes.saturating_sub(child.init_config_bytes);
                            self.reserved_mailbox_gauge.fetch_sub(child.init_config_bytes, Ordering::AcqRel);
                            child.init_config_bytes = 0;
                        }
                        self.children.push_back(child);
                    }
                    InitOutcome::DeterministicFail => {
                        drop(mem_lease);
                        child.settle_terminal(
                            ProcessStatus::InitializationFailed,
                            ProcessOutcome::Failed,
                            &self.reserved_mailbox_gauge,
                        );
                        self.children.push_back(child);
                    }
                    InitOutcome::FatalOverrun => {
                        drop(mem_lease);
                        self.children.push_back(child);
                        let drained_events = self.abort_scheduler();
                        return Err(ProcessHostError::Overrun { drained_events });
                    }
                }
            } else {
                let cmd = child.pending.pop_front().expect("runnable process has command");
                let mut guest = child.guest.take().expect("ready process owns guest");
                let outcome = run_process_step(
                    &mut guest,
                    &prog.manifest,
                    &prog.policy,
                    &cmd.cmd,
                    deadline_at,
                    &cancel_flag,
                );
                match outcome {
                    StepOutcome::Ok(reply) => {
                        child.guest = Some(guest);
                        let reply_bytes = reply.logical_bytes();
                        let old_reserve = cmd.cmd_bytes.saturating_add(cmd.max_reply_bytes);
                        // Release cmd_bytes + max_reply_bytes, charge exact logical_reply_bytes.
                        child.charged_mailbox_bytes = child
                            .charged_mailbox_bytes
                            .saturating_sub(old_reserve)
                            .saturating_add(reply_bytes);
                        if old_reserve > reply_bytes {
                            self.reserved_mailbox_gauge.fetch_sub(old_reserve - reply_bytes, Ordering::AcqRel);
                        } else if reply_bytes > old_reserve {
                            self.reserved_mailbox_gauge.fetch_add(reply_bytes - old_reserve, Ordering::AcqRel);
                        }
                        child.events.push_back(ProcessEvent {
                            ticket: cmd.ticket,
                            outcome: ProcessOutcome::Done(reply),
                        });
                        self.children.push_back(child);
                    }
                    StepOutcome::DeterministicFail => {
                        drop(guest);
                        let old_reserve = cmd.cmd_bytes.saturating_add(cmd.max_reply_bytes);
                        child.charged_mailbox_bytes = child.charged_mailbox_bytes.saturating_sub(old_reserve);
                        if old_reserve > 0 {
                            self.reserved_mailbox_gauge.fetch_sub(old_reserve, Ordering::AcqRel);
                        }
                        child.events.push_back(ProcessEvent { ticket: cmd.ticket, outcome: ProcessOutcome::Failed });
                        child.settle_terminal(
                            ProcessStatus::StepFailed,
                            ProcessOutcome::Failed,
                            &self.reserved_mailbox_gauge,
                        );
                        self.children.push_back(child);
                    }
                    StepOutcome::FatalOverrun => {
                        drop(guest);
                        let old_reserve = cmd.cmd_bytes.saturating_add(cmd.max_reply_bytes);
                        child.charged_mailbox_bytes = child.charged_mailbox_bytes.saturating_sub(old_reserve);
                        if old_reserve > 0 {
                            self.reserved_mailbox_gauge.fetch_sub(old_reserve, Ordering::AcqRel);
                        }
                        child.events.push_back(ProcessEvent { ticket: cmd.ticket, outcome: ProcessOutcome::Cancelled });
                        self.children.push_back(child);
                        let drained_events = self.abort_scheduler();
                        return Err(ProcessHostError::Overrun { drained_events });
                    }
                }
            }
        }
        if let Some(endpoint) = deferred
            && let Some(index) = self.children.iter().position(|c| c.handle.endpoint == endpoint)
        {
            self.children.rotate_left(index);
        }
        Ok(spent)
    }

    /// Executes runnable independent endpoints concurrently in serial-equivalent waves.
    ///
    /// Preserves bit-identical state, event order, work spent, and ring rotation against
    /// [`Processes::advance`] across all non-fatal runs. On any [`ProcessHostError::Overrun`],
    /// immediately sets `wave_cancel_flag` BEFORE incrementing epoch, waits up to `50 ms`,
    /// detaches stuck siblings with their byte leases, permanently aborts the scheduler,
    /// and returns `Err(ProcessHostError::Overrun { drained_events })`.
    pub fn advance_all_parallel(&mut self, budget: u32) -> std::result::Result<u32, ProcessHostError> {
        if self.aborted {
            return Err(ProcessHostError::Aborted);
        }
        let mut spent = 0u32;
        let mut skipped = 0usize;
        let mut deferred: Option<u32> = None;

        while spent < budget && skipped < self.children.len() {
            let ring_len = self.children.len();
            let mut ring: VecDeque<RingSlot> = self.children.drain(..).map(RingSlot::Present).collect();
            let mut wave_children: Vec<Child> = Vec::new();

            // Select one wave of distinct runnable endpoints in exact serial ring order!
            while spent < budget && skipped < ring_len {
                match ring.front() {
                    Some(RingSlot::InWave(_)) => break,
                    None => break,
                    Some(RingSlot::Present(_)) => {}
                }
                let RingSlot::Present(child) = ring.pop_front().unwrap() else {
                    unreachable!()
                };
                let prog = &self.programs[child.program];
                let cost = match child.status {
                    ProcessStatus::Starting => prog.policy.init_work,
                    ProcessStatus::Ready if !child.pending.is_empty() => prog.policy.step_work,
                    _ => 0,
                };
                if cost == 0 || cost > budget - spent {
                    if cost != 0 && cost <= budget && deferred.is_none() {
                        deferred = Some(child.handle.endpoint);
                    }
                    skipped += 1;
                    ring.push_back(RingSlot::Present(child));
                } else {
                    skipped = 0;
                    spent += cost;
                    let idx = wave_children.len();
                    wave_children.push(child);
                    ring.push_back(RingSlot::InWave(idx));
                }
            }

            if wave_children.is_empty() {
                // Restore ring and exit.
                self.children = ring
                    .into_iter()
                    .map(|slot| match slot {
                        RingSlot::Present(c) => c,
                        RingSlot::InWave(_) => unreachable!(),
                    })
                    .collect();
                break;
            }

            // Execute all selected endpoints in this wave concurrently!
            let wave_cancel = Arc::new(AtomicBool::new(false));
            let wave_overrun = Arc::new(AtomicBool::new(false));
            let done_sync = Arc::new((Mutex::new(0usize), Condvar::new()));
            let wave_count = wave_children.len();

            struct TaskSlot {
                child_meta: Child,
                in_flight_cmd: Option<PendingCommand>,
                result: Arc<Mutex<Option<WaveTaskResult>>>,
            }

            enum WaveTaskResult {
                InitOk(WasmtimeGuest, ByteLease),
                InitFail,
                InitOverrun,
                StepOk(WasmtimeGuest, ByteLease, ProcessValue),
                StepFail,
                StepOverrun,
            }

            let mut task_slots: Vec<TaskSlot> = Vec::with_capacity(wave_count);
            let mut handles = Vec::with_capacity(wave_count);

            for mut child in wave_children {
                let prog = &self.programs[child.program];
                let manifest = prog.manifest.clone();
                let policy = prog.policy.clone();
                let engine = self.engine.clone();
                let module = self.module.clone();
                let cancel = wave_cancel.clone();
                let overrun_flag = wave_overrun.clone();
                let sync = done_sync.clone();
                let deadline_at = policy.deadline.map(|d| Instant::now() + d);
                let result_cell = Arc::new(Mutex::new(None));
                let t_result = result_cell.clone();

                if child.status == ProcessStatus::Starting {
                    let config = child.init_config.take().expect("starting child has config");
                    let mem_lease = ByteLease::acquire(&self.reserved_memory_gauge, policy.memory.memory);
                    // Hold an extra command-byte lease on the thread for init_config_bytes so if detached on overrun,
                    // bytes only release at physical thread exit!
                    let cfg_bytes = child.init_config_bytes;
                    let cfg_detach_lease = if cfg_bytes > 0 {
                        // Transfer charge ownership into thread lease during wave execution.
                        child.init_config_bytes = 0;
                        child.charged_mailbox_bytes = child.charged_mailbox_bytes.saturating_sub(cfg_bytes);
                        Some(ByteLease { gauge: self.reserved_mailbox_gauge.clone(), bytes: cfg_bytes })
                    } else {
                        None
                    };
                    task_slots.push(TaskSlot { child_meta: child, in_flight_cmd: None, result: result_cell });
                    let handle = std::thread::Builder::new()
                        .name("blot-proc-wave".into())
                        .stack_size(GUEST_STACK_BYTES + (2 << 20))
                        .spawn(move || {
                            let res = run_process_init(
                                &engine,
                                &module,
                                &manifest,
                                &policy,
                                &config,
                                deadline_at,
                                &cancel,
                            );
                            let out = match res {
                                InitOutcome::Ok(guest) => {
                                    drop(cfg_detach_lease);
                                    WaveTaskResult::InitOk(guest, mem_lease)
                                }
                                InitOutcome::DeterministicFail => {
                                    drop(mem_lease);
                                    drop(cfg_detach_lease);
                                    WaveTaskResult::InitFail
                                }
                                InitOutcome::FatalOverrun => {
                                    // Set cancel BEFORE incrementing epoch!
                                    cancel.store(true, Ordering::Release);
                                    overrun_flag.store(true, Ordering::Release);
                                    engine.increment_epoch();
                                    drop(mem_lease);
                                    drop(cfg_detach_lease);
                                    WaveTaskResult::InitOverrun
                                }
                            };
                            *t_result.lock().unwrap() = Some(out);
                            let (lock, cvar) = &*sync;
                            let mut finished = lock.lock().unwrap();
                            *finished += 1;
                            cvar.notify_all();
                        })
                        .map_err(|e| ProcessHostError::Internal(format!("spawning process wave thread: {e}")))?;
                    handles.push(handle);
                } else {
                    let cmd = child.pending.pop_front().expect("runnable process has command");
                    let mut guest = child.guest.take().expect("ready process owns guest");
                    let mem_lease = child.memory_lease.take().expect("ready process owns memory lease");
                    let cmd_reserve = cmd.cmd_bytes.saturating_add(cmd.max_reply_bytes);
                    // Transfer cmd_reserve lease to worker thread during wave so detached threads keep it until exit!
                    child.charged_mailbox_bytes = child.charged_mailbox_bytes.saturating_sub(cmd_reserve);
                    let cmd_lease = ByteLease { gauge: self.reserved_mailbox_gauge.clone(), bytes: cmd_reserve };
                    let cmd_val = cmd.cmd.clone();
                    task_slots.push(TaskSlot { child_meta: child, in_flight_cmd: Some(cmd), result: result_cell });
                    let handle = std::thread::Builder::new()
                        .name("blot-proc-wave".into())
                        .stack_size(GUEST_STACK_BYTES + (2 << 20))
                        .spawn(move || {
                            let res = run_process_step(
                                &mut guest,
                                &manifest,
                                &policy,
                                &cmd_val,
                                deadline_at,
                                &cancel,
                            );
                            let out = match res {
                                StepOutcome::Ok(reply) => {
                                    // Keep cmd_lease until coordinator updates reply bytes on non-overrun,
                                    // or drops cleanly on thread exit if detached!
                                    drop(cmd_lease);
                                    WaveTaskResult::StepOk(guest, mem_lease, reply)
                                }
                                StepOutcome::DeterministicFail => {
                                    drop(guest);
                                    drop(mem_lease);
                                    drop(cmd_lease);
                                    WaveTaskResult::StepFail
                                }
                                StepOutcome::FatalOverrun => {
                                    cancel.store(true, Ordering::Release);
                                    overrun_flag.store(true, Ordering::Release);
                                    engine.increment_epoch();
                                    drop(guest);
                                    drop(mem_lease);
                                    drop(cmd_lease);
                                    WaveTaskResult::StepOverrun
                                }
                            };
                            *t_result.lock().unwrap() = Some(out);
                            let (lock, cvar) = &*sync;
                            let mut finished = lock.lock().unwrap();
                            *finished += 1;
                            cvar.notify_all();
                        })
                        .map_err(|e| ProcessHostError::Internal(format!("spawning process wave thread: {e}")))?;
                    handles.push(handle);
                }
            }

            // Wait for wave completion OR overrun + 50 ms join grace!
            let any_overrun = {
                let (lock, cvar) = &*done_sync;
                let mut finished = lock.lock().unwrap();
                let mut grace_deadline: Option<Instant> = None;
                loop {
                    if *finished >= wave_count {
                        break wave_overrun.load(Ordering::Acquire);
                    }
                    if wave_overrun.load(Ordering::Acquire) {
                        wave_cancel.store(true, Ordering::Release);
                        self.engine.increment_epoch();
                        let dl = *grace_deadline.get_or_insert_with(|| Instant::now() + OVERRUN_JOIN_GRACE);
                        let Some(rem) = dl.checked_duration_since(Instant::now()) else {
                            break true;
                        };
                        if rem.is_zero() {
                            break true;
                        }
                        let (next, res) = cvar.wait_timeout(finished, rem).unwrap();
                        finished = next;
                        if res.timed_out() {
                            break true;
                        }
                    } else {
                        // Wake periodically so any overrun set right before wait is noticed promptly.
                        let (next, _) = cvar.wait_timeout(finished, Duration::from_millis(5)).unwrap();
                        finished = next;
                    }
                }
            };

            if any_overrun {
                wave_cancel.store(true, Ordering::Release);
                self.engine.increment_epoch();
                // Detach any still-running thread (holding its own `mem_lease` & `cmd_lease` until exit!);
                // join threads that already finished.
                for (handle, slot) in handles.into_iter().zip(task_slots.iter()) {
                    if slot.result.lock().unwrap().is_some() {
                        let _ = handle.join();
                    } else {
                        drop(handle);
                    }
                }
                // Reassemble all children into `self.children`, settling popped in-flight tickets as Cancelled
                // and discarding uncommitted sibling completions!
                let mut slots_opt: Vec<Option<TaskSlot>> = task_slots.into_iter().map(Some).collect();
                for entry in ring {
                    match entry {
                        RingSlot::Present(c) => self.children.push_back(c),
                        RingSlot::InWave(idx) => {
                            let mut slot = slots_opt[idx].take().unwrap();
                            // Drop any completed guest/lease immediately.
                            let _ = slot.result.lock().unwrap().take();
                            if let Some(cmd) = slot.in_flight_cmd.take() {
                                slot.child_meta
                                    .events
                                    .push_back(ProcessEvent { ticket: cmd.ticket, outcome: ProcessOutcome::Cancelled });
                            }
                            self.children.push_back(slot.child_meta);
                        }
                    }
                }
                let drained_events = self.abort_scheduler();
                return Err(ProcessHostError::Overrun { drained_events });
            }

            // All wave threads finished cleanly without overrun: join and commit results in exact ring order!
            for handle in handles {
                let _ = handle.join();
            }
            let mut completed_children: Vec<Option<Child>> = Vec::with_capacity(wave_count);
            for mut slot in task_slots {
                let res = slot.result.lock().unwrap().take().expect("finished wave task has result");
                match res {
                    WaveTaskResult::InitOk(guest, mem_lease) => {
                        slot.child_meta.guest = Some(guest);
                        slot.child_meta.memory_lease = Some(mem_lease);
                        slot.child_meta.status = ProcessStatus::Ready;
                    }
                    WaveTaskResult::InitFail => {
                        slot.child_meta.settle_terminal(
                            ProcessStatus::InitializationFailed,
                            ProcessOutcome::Failed,
                            &self.reserved_mailbox_gauge,
                        );
                    }
                    WaveTaskResult::StepOk(guest, mem_lease, reply) => {
                        let cmd = slot.in_flight_cmd.take().unwrap();
                        slot.child_meta.guest = Some(guest);
                        slot.child_meta.memory_lease = Some(mem_lease);
                        let reply_bytes = reply.logical_bytes();
                        slot.child_meta.charged_mailbox_bytes += reply_bytes;
                        if reply_bytes > 0 {
                            self.reserved_mailbox_gauge.fetch_add(reply_bytes, Ordering::AcqRel);
                        }
                        slot.child_meta.events.push_back(ProcessEvent {
                            ticket: cmd.ticket,
                            outcome: ProcessOutcome::Done(reply),
                        });
                    }
                    WaveTaskResult::StepFail => {
                        let cmd = slot.in_flight_cmd.take().unwrap();
                        slot.child_meta
                            .events
                            .push_back(ProcessEvent { ticket: cmd.ticket, outcome: ProcessOutcome::Failed });
                        slot.child_meta.settle_terminal(
                            ProcessStatus::StepFailed,
                            ProcessOutcome::Failed,
                            &self.reserved_mailbox_gauge,
                        );
                    }
                    WaveTaskResult::InitOverrun | WaveTaskResult::StepOverrun => {
                        unreachable!("handled by any_overrun branch above")
                    }
                }
                completed_children.push(Some(slot.child_meta));
            }
            for entry in ring {
                match entry {
                    RingSlot::Present(c) => self.children.push_back(c),
                    RingSlot::InWave(idx) => self.children.push_back(completed_children[idx].take().unwrap()),
                }
            }
        }

        if let Some(endpoint) = deferred
            && let Some(index) = self.children.iter().position(|c| c.handle.endpoint == endpoint)
        {
            self.children.rotate_left(index);
        }
        Ok(spent)
    }

    fn abort_scheduler(&mut self) -> BTreeMap<ProcessHandle, VecDeque<ProcessEvent>> {
        self.aborted = true;
        self.restore()
    }
}

enum InitOutcome {
    Ok(WasmtimeGuest),
    DeterministicFail,
    FatalOverrun,
}

enum StepOutcome {
    Ok(ProcessValue),
    DeterministicFail,
    FatalOverrun,
}

fn is_overrun_result<T>(
    res: &anyhow::Result<T>,
    cancel_flag: &AtomicBool,
    deadline_at: Option<Instant>,
) -> bool {
    if cancel_flag.load(Ordering::Acquire) {
        return true;
    }
    if res.as_ref().err().is_some_and(is_trap_interrupt) {
        return true;
    }
    if let Some(deadline) = deadline_at
        && Instant::now() >= deadline
    {
        return true;
    }
    false
}

fn is_overrun_now(cancel_flag: &AtomicBool, deadline_at: Option<Instant>) -> bool {
    cancel_flag.load(Ordering::Acquire) || deadline_at.is_some_and(|d| Instant::now() >= d)
}

/// Dedicated Rust `proc_init` turn executor with Overrun-first precedence (§5.2).
fn run_process_init(
    engine: &Engine,
    module: &GuestModule,
    manifest: &ProcessManifest,
    policy: &ProcessPolicy,
    config: &ProcessValue,
    deadline_at: Option<Instant>,
    cancel_flag: &Arc<AtomicBool>,
) -> InitOutcome {
    if is_overrun_now(cancel_flag, deadline_at) {
        return InitOutcome::FatalOverrun;
    }
    let wt_module = module.wasmtime_module().expect("validated Wasmtime module");
    let init_fuel = policy.rust_wasmtime_fuel.map(|f| f.init_fuel);
    let cleanup_fuel = policy.rust_wasmtime_fuel.map(|f| f.cleanup_fuel);

    let started = WasmtimeGuest::start_controlled(
        engine,
        wt_module,
        module,
        Some(policy.memory),
        deadline_at,
        Some(cancel_flag.clone()),
        init_fuel,
    );
    if is_overrun_result(&started, cancel_flag, deadline_at) {
        return InitOutcome::FatalOverrun;
    }
    let mut guest = match started {
        Ok(g) => g,
        Err(_) => return InitOutcome::DeterministicFail,
    };

    let primary = (|| -> anyhow::Result<()> {
        guest.arm_execution(deadline_at, Some(cancel_flag.clone()), init_fuel)?;
        if guest.has_arena() {
            guest.reset_arena()?;
        }
        let param = encode_process_value(&mut guest, &manifest.config_schema, config)?;
        guest.call_raw(&manifest.init_export, param)?;
        Ok(())
    })();

    if is_overrun_result(&primary, cancel_flag, deadline_at) {
        return InitOutcome::FatalOverrun;
    }
    if primary.is_err() {
        return InitOutcome::DeterministicFail;
    }

    let cleanup = (|| -> anyhow::Result<()> {
        guest.arm_execution(deadline_at, Some(cancel_flag.clone()), cleanup_fuel)?;
        guest.finish_invocation()?;
        if guest.has_arena() {
            guest.reset_arena()?;
        }
        Ok(())
    })();

    if is_overrun_result(&cleanup, cancel_flag, deadline_at) {
        return InitOutcome::FatalOverrun;
    }
    if cleanup.is_err() {
        return InitOutcome::DeterministicFail;
    }
    InitOutcome::Ok(guest)
}

/// Dedicated Rust `proc_step` turn executor with Overrun-first precedence (§5.2).
fn run_process_step(
    guest: &mut WasmtimeGuest,
    manifest: &ProcessManifest,
    policy: &ProcessPolicy,
    cmd: &ProcessValue,
    deadline_at: Option<Instant>,
    cancel_flag: &Arc<AtomicBool>,
) -> StepOutcome {
    if is_overrun_now(cancel_flag, deadline_at) {
        return StepOutcome::FatalOverrun;
    }
    let step_fuel = policy.rust_wasmtime_fuel.map(|f| f.step_fuel);
    let cleanup_fuel = policy.rust_wasmtime_fuel.map(|f| f.cleanup_fuel);

    let primary = (|| -> anyhow::Result<Val> {
        guest.arm_execution(deadline_at, Some(cancel_flag.clone()), step_fuel)?;
        if guest.has_arena() {
            guest.reset_arena()?;
        }
        let param = encode_process_value(guest, &manifest.cmd_schema, cmd)?;
        guest.call_raw(&manifest.step_export, param)
    })();

    // Post-entry Overrun-first precedence (checked BEFORE inspecting Ok/Err, closing Cranelift
    // fuel-before-epoch ordering!).
    if is_overrun_result(&primary, cancel_flag, deadline_at) {
        return StepOutcome::FatalOverrun;
    }
    let raw_reply = match primary {
        Ok(val) => val,
        Err(_) => return StepOutcome::DeterministicFail,
    };

    let decoded = decode_process_reply(guest, &manifest.reply_schema, &raw_reply, policy.max_reply_bytes);
    if is_overrun_now(cancel_flag, deadline_at) {
        return StepOutcome::FatalOverrun;
    }
    let reply = match decoded {
        Ok(val) => val,
        Err(_) => return StepOutcome::DeterministicFail,
    };

    let cleanup = (|| -> anyhow::Result<()> {
        guest.arm_execution(deadline_at, Some(cancel_flag.clone()), cleanup_fuel)?;
        guest.finish_invocation()?;
        if guest.has_arena() {
            guest.reset_arena()?;
        }
        Ok(())
    })();

    if is_overrun_result(&cleanup, cancel_flag, deadline_at) {
        return StepOutcome::FatalOverrun;
    }
    if cleanup.is_err() {
        return StepOutcome::DeterministicFail;
    }
    StepOutcome::Ok(reply)
}

fn validate_reply_budget(
    reply_schema: &WireSchema,
    max_reply_bytes: usize,
    process_name: &str,
) -> std::result::Result<(), ProcessHostError> {
    match reply_schema.wire_tag {
        ValueType::Unit => Ok(()),
        ValueType::U32 | ValueType::Bool | ValueType::F32 => {
            if max_reply_bytes < 4 {
                return Err(ProcessHostError::InvalidPolicy(format!(
                    "process `{process_name}` has scalar reply requiring max_reply_bytes >= 4 (got {max_reply_bytes})"
                )));
            }
            Ok(())
        }
        ValueType::ArrayU32 | ValueType::ArrayF32 => {
            if reply_schema.kind == WireSchemaKind::FlatFixed {
                let min_bytes = (reply_schema.stride as usize).saturating_mul(4);
                if max_reply_bytes < min_bytes {
                    return Err(ProcessHostError::InvalidPolicy(format!(
                        "process `{process_name}` has FlatFixed reply requiring max_reply_bytes >= {min_bytes} (got {max_reply_bytes})"
                    )));
                }
            }
            Ok(())
        }
    }
}

fn validate_value_against_schema(value: &ProcessValue, schema: &WireSchema, transport_budget: usize) -> bool {
    let max_words = (transport_budget.saturating_sub(4)) / 4;
    match (schema.wire_tag, value) {
        (ValueType::Unit, ProcessValue::Unit) => true,
        (ValueType::U32, ProcessValue::U32(_)) => true,
        (ValueType::Bool, ProcessValue::Bool(_)) => true,
        (ValueType::F32, ProcessValue::F32(_)) => true,
        (ValueType::ArrayU32 | ValueType::ArrayF32, ProcessValue::Words(words)) => {
            if words.len() > max_words {
                return false;
            }
            if schema.kind == WireSchemaKind::FlatFixed {
                words.len() == schema.stride as usize
            } else {
                true
            }
        }
        _ => false,
    }
}

fn encode_process_value(
    guest: &mut WasmtimeGuest,
    schema: &WireSchema,
    value: &ProcessValue,
) -> anyhow::Result<Val> {
    match (schema.wire_tag, value) {
        (ValueType::Unit, ProcessValue::Unit) => Ok(Val::I32(0)),
        (ValueType::U32, ProcessValue::U32(v)) => Ok(Val::I32(v.cast_signed())),
        (ValueType::Bool, ProcessValue::Bool(v)) => Ok(Val::I32(i32::from(*v))),
        (ValueType::F32, ProcessValue::F32(v)) => Ok(Val::F32(v.to_bits())),
        (ValueType::ArrayU32 | ValueType::ArrayF32, ProcessValue::Words(words)) => {
            let ptr = guest.write_words_packet(words)?;
            Ok(Val::I32(ptr))
        }
        _ => anyhow::bail!("process value does not match wire schema"),
    }
}

fn decode_process_reply(
    guest: &WasmtimeGuest,
    schema: &WireSchema,
    raw: &Val,
    max_reply_bytes: usize,
) -> anyhow::Result<ProcessValue> {
    match (schema.wire_tag, raw) {
        (ValueType::Unit, Val::I32(0)) => Ok(ProcessValue::Unit),
        (ValueType::U32, Val::I32(v)) => {
            anyhow::ensure!(4 <= max_reply_bytes, "scalar reply exceeds max_reply_bytes");
            Ok(ProcessValue::U32(v.cast_unsigned()))
        }
        (ValueType::Bool, Val::I32(v @ (0 | 1))) => {
            anyhow::ensure!(4 <= max_reply_bytes, "scalar reply exceeds max_reply_bytes");
            Ok(ProcessValue::Bool(*v == 1))
        }
        (ValueType::F32, Val::F32(bits)) => {
            anyhow::ensure!(4 <= max_reply_bytes, "scalar reply exceeds max_reply_bytes");
            Ok(ProcessValue::F32(f32::from_bits(*bits)))
        }
        (ValueType::ArrayU32 | ValueType::ArrayF32, Val::I32(ptr)) => {
            let header = guest.inspect_packet_header(*ptr)?;
            anyhow::ensure!(header != u32::MAX, "unexpected overflow sentinel in process reply");
            let words_len = header as usize;
            let bytes = words_len.saturating_mul(4);
            anyhow::ensure!(bytes <= max_reply_bytes, "process reply {bytes} bytes exceeds max_reply_bytes {max_reply_bytes}");
            if schema.kind == WireSchemaKind::FlatFixed {
                anyhow::ensure!(
                    words_len == schema.stride as usize,
                    "FlatFixed reply length {words_len} != stride {}",
                    schema.stride
                );
            }
            let words = guest.read_packet_words_vec(*ptr, words_len)?;
            Ok(ProcessValue::Words(words))
        }
        _ => anyhow::bail!("invalid or non-canonical process reply value"),
    }
}
