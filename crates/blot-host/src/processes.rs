//! Opt-in cooperative processes. All guest turns return to the caller's thread.
//! Register only entries whose initialization and step have bounded work.
use crate::{Guest, GuestModule, Limits, Parameter, Value, ValueType};
use anyhow::{Result, ensure};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::atomic::{AtomicU32, Ordering},
};
use wasmtime::Engine;

static NEXT_ENDPOINT: AtomicU32 = AtomicU32::new(1);

pub struct Program {
    pub entry: String,
    pub init_work: u32,
    pub step_work: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Starting,
    Ready,
    InitializationFailed,
    StepFailed,
    Closed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Send {
    Accepted(u64),
    Starting,
    Full,
    Closed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Done(u32),
    Cancelled,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event {
    pub ticket: u64,
    pub outcome: Outcome,
}

struct Registered {
    init: String,
    step: String,
    init_work: u32,
    step_work: u32,
}

struct Child {
    endpoint: u32,
    program: usize,
    status: Status,
    state: u32,
    guest: Option<Guest>,
    pending: VecDeque<(u64, u32)>,
    events: VecDeque<Event>,
}

impl Child {
    fn settle(&mut self, status: Status, outcome: Outcome) {
        self.status = status;
        for (ticket, _) in self.pending.drain(..) {
            self.events.push_back(Event { ticket, outcome });
        }
        if let Some(mut guest) = self.guest.take() {
            // Logical outcomes settle before the instance is reclaimed.
            let _ = guest.dispose();
        }
    }
}

pub struct Processes {
    engine: Engine,
    module: GuestModule,
    programs: Vec<Registered>,
    slots: usize,
    mailbox: usize,
    memory: Limits,
    children: VecDeque<Child>,
    next_ticket: u64,
}

impl Processes {
    pub fn new(
        engine: &Engine,
        module: &GuestModule,
        programs: Vec<Program>,
        slots: usize,
        mailbox: usize,
        memory: Limits,
    ) -> Result<Self> {
        ensure!(
            slots > 0 && mailbox > 0 && memory.memory > 0 && slots.checked_mul(memory.memory).is_some(),
            "invalid process limits"
        );
        ensure!(module.abi().data_stacks, "processes require guest ABI 8");
        let mut registered: Vec<Registered> = Vec::new();
        for p in programs {
            let step = format!("{}_step", p.entry);
            ensure!(
                p.init_work > 0
                    && p.step_work > 0
                    && p.entry.len() <= 64
                    && p.entry.bytes().next().is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
                    && p.entry.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                    && registered.iter().all(|other| p.entry != other.init),
                "invalid registered process"
            );
            for (entry, input) in [(&p.entry, ValueType::U32), (&step, ValueType::ArrayU32)] {
                let found = module.abi().function(entry);
                ensure!(
                    found.is_some_and(|f| f.parameter == Parameter::Value(input) && f.result == ValueType::U32),
                    "invalid process entry {entry}"
                );
            }
            registered.push(Registered { init: p.entry, step, init_work: p.init_work, step_work: p.step_work });
        }
        Ok(Self {
            engine: engine.clone(),
            module: module.clone(),
            programs: registered,
            slots,
            mailbox,
            memory,
            children: VecDeque::new(),
            next_ticket: 1,
        })
    }

    /// Synchronous provider for a scoped `@spawn` packet. Zero means refused.
    /// An accepted process starts on a later `advance` turn.
    pub fn admit(&mut self, packet: &[u32]) -> u32 {
        if self.children.len() == self.slots || !(2..=65).contains(&packet.len()) {
            return 0;
        }
        let Some(program) = self.programs.iter().position(|p| {
            p.init.len() == packet.len() - 1
                && p.init.bytes().zip(&packet[1..]).all(|(byte, word)| u32::from(byte) == *word)
        }) else {
            return 0;
        };
        let Ok(endpoint) = NEXT_ENDPOINT.try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        else {
            return 0;
        };
        self.children.push_back(Child {
            endpoint,
            program,
            status: Status::Starting,
            state: packet[0],
            guest: None,
            pending: VecDeque::new(),
            events: VecDeque::new(),
        });
        endpoint
    }

    pub fn status(&self, endpoint: u32) -> Status {
        self.children.iter().find(|c| c.endpoint == endpoint).map_or(Status::Closed, |c| c.status)
    }

    pub fn send(&mut self, endpoint: u32, command: u32) -> Send {
        let Some(child) = self.children.iter_mut().find(|c| c.endpoint == endpoint) else { return Send::Closed };
        match child.status {
            Status::Starting => return Send::Starting,
            Status::Ready => {}
            _ => return Send::Closed,
        }
        if child.pending.len() + child.events.len() >= self.mailbox {
            return Send::Full;
        }
        let ticket = self.next_ticket;
        let Some(next) = ticket.checked_add(1) else { return Send::Full };
        self.next_ticket = next;
        child.pending.push_back((ticket, command));
        Send::Accepted(ticket)
    }

    pub fn poll(&mut self, endpoint: u32) -> Option<Event> {
        self.children.iter_mut().find(|c| c.endpoint == endpoint)?.events.pop_front()
    }

    pub fn cancel(&mut self, endpoint: u32, ticket: u64) -> bool {
        let Some(child) = self.children.iter_mut().find(|c| c.endpoint == endpoint) else { return false };
        let Some(index) = child.pending.iter().position(|(id, _)| *id == ticket) else { return false };
        child.pending.remove(index);
        child.events.push_back(Event { ticket, outcome: Outcome::Cancelled });
        true
    }

    /// Run complete guest invocations within one shared allowance.
    pub fn advance(&mut self, budget: u32) -> u32 {
        let mut spent = 0;
        let mut skipped = 0;
        let mut deferred = None;
        while spent < budget && skipped < self.children.len() {
            let mut child = self.children.pop_front().expect("unvisited process");
            let program = &self.programs[child.program];
            let cost = match child.status {
                Status::Starting => program.init_work,
                Status::Ready if !child.pending.is_empty() => program.step_work,
                _ => 0,
            };
            if cost == 0 || cost > budget - spent {
                if cost != 0 && cost <= budget && deferred.is_none() {
                    deferred = Some(child.endpoint);
                }
                skipped += 1;
            } else {
                skipped = 0;
                spent += cost;
                if child.status == Status::Starting {
                    let init = Guest::sandboxed(&self.engine, &self.module, self.memory).and_then(|mut guest| {
                        let result = guest.call(&program.init, Value::U32(child.state));
                        child.guest = Some(guest);
                        result
                    });
                    if let Ok(Value::U32(state)) = init {
                        child.state = state;
                        child.status = Status::Ready;
                    } else {
                        child.settle(Status::InitializationFailed, Outcome::Failed);
                    }
                } else {
                    let &(ticket, command) = child.pending.front().expect("runnable process has a command");
                    let guest = child.guest.as_mut().expect("ready process owns its guest");
                    let result = guest.call(&program.step, Value::ArrayU32(vec![child.state, command]));
                    if let Ok(Value::U32(state)) = result {
                        child.state = state;
                        child.pending.pop_front();
                        child.events.push_back(Event { ticket, outcome: Outcome::Done(state) });
                    } else {
                        child.settle(Status::StepFailed, Outcome::Failed);
                    }
                }
            }
            self.children.push_back(child);
        }
        if let Some(endpoint) = deferred
            && let Some(index) = self.children.iter().position(|c| c.endpoint == endpoint)
        {
            self.children.rotate_left(index);
        }
        spent
    }

    pub fn close(&mut self, endpoint: u32) {
        if let Some(child) = self.children.iter_mut().find(|c| c.endpoint == endpoint)
            && matches!(child.status, Status::Ready | Status::Starting)
        {
            child.settle(Status::Closed, Outcome::Cancelled);
        }
    }

    pub fn release(&mut self, endpoint: u32) -> bool {
        let Some(index) = self.children.iter().position(|c| {
            c.endpoint == endpoint && c.events.is_empty() && !matches!(c.status, Status::Ready | Status::Starting)
        }) else {
            return false;
        };
        self.children.remove(index).is_some()
    }

    /// Fence old endpoints and return the old owner's undelivered outcomes.
    pub fn restore(&mut self) -> BTreeMap<u32, VecDeque<Event>> {
        let mut retired = BTreeMap::new();
        while let Some(mut child) = self.children.pop_front() {
            child.settle(Status::Closed, Outcome::Cancelled);
            if !child.events.is_empty() {
                retired.insert(child.endpoint, child.events);
            }
        }
        retired
    }

    /// Observed linear memory after completed turns. The configured hard
    /// ceiling is the slot count times each sandbox's limit.
    pub fn memory_bytes(&self) -> usize {
        self.children.iter().filter_map(|c| c.guest.as_ref()).map(Guest::memory_bytes).sum()
    }
}
