//! What the dev tools see of a guest as it runs ([`Probe`]): its committed
//! memory and the most it committed, the counters guest ABI 3's ownership
//! exports (`blot:rc_*` and `blot:cycle_*` globals: retains, releases,
//! objects destroyed, roots, cycle passes...), the host's calls into it and
//! those that failed, and, while a [`Profile`] is attached, samples of its
//! stack. A guest of [`crate::dev_engine`] is sampled at each tick of its
//! engine's epoch, which [`Profile::start`] makes about a millisecond.
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use anyhow::Result;
use pen_runtime::symbols::Symbols;
use wasmtime::{Engine, FrameInfo, WasmBacktrace};

/// What a probe saw of its guest.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Seen {
    /// Committed linear-memory bytes as of the guest's last host call, and
    /// the most it had then.
    pub memory: usize,
    pub peak: usize,
    /// The counters the guest exports, by name, as of its last host call.
    pub counters: Vec<(String, u64)>,
    /// Calls into the guest, and those that failed (trapped, or the host
    /// failed during them).
    pub calls: u64,
    pub failed: u64,
    /// The guest's calls to the host (a game's, one a frame).
    pub callbacks: u64,
}

/// Watches one guest for the dev tools; the guest updates it as it runs.
#[derive(Debug, Default)]
pub struct Probe {
    seen: Mutex<Seen>,
    profile: Mutex<Option<Arc<Profile>>>,
}

impl Probe {
    pub fn new() -> Arc<Probe> {
        Arc::new(Probe::default())
    }

    pub fn seen(&self) -> Seen {
        self.seen.lock().unwrap().clone()
    }

    /// Samples the guest into `profile` from now on, or stops with None.
    pub fn attach(&self, profile: Option<Arc<Profile>>) {
        *self.profile.lock().unwrap() = profile;
    }

    /// The profile its guest is sampled into, while it samples.
    pub(crate) fn sampling(&self) -> Option<Arc<Profile>> {
        self.profile.lock().unwrap().clone().filter(|profile| profile.sampling())
    }

    /// The counters its guest exports, by name, at 0 until it calls the host.
    pub(crate) fn counters_named(&self, names: Vec<String>) {
        self.seen.lock().unwrap().counters = names.into_iter().map(|name| (name, 0)).collect();
    }

    /// The guest called the host: its memory and its counters now, in the
    /// order they were named.
    pub(crate) fn called_host(&self, memory: usize, counters: impl Iterator<Item = u64>) {
        let mut seen = self.seen.lock().unwrap();
        seen.memory = memory;
        seen.peak = seen.peak.max(memory);
        seen.callbacks += 1;
        for ((_, kept), value) in seen.counters.iter_mut().zip(counters) {
            *kept = value;
        }
    }

    /// A call into the guest ended, failing or not.
    pub(crate) fn called(&self, failed: bool) {
        let mut seen = self.seen.lock().unwrap();
        seen.calls += 1;
        seen.failed += u64::from(failed);
    }
}

/// Function `index` as a profile names it: by the module's symbols (its
/// name section, else its exports; `pen_runtime::symbols`, which name a
/// trap's frames too), else by its index. The one place a profile looks
/// names up.
pub fn function_name(symbols: &Symbols, index: u32) -> String {
    symbols.function(index).map_or_else(|| format!("function {index}"), str::to_owned)
}

/// Samples of a guest's stack: how many found each stack of function
/// indices (from the outermost call in), and the functions' names.
#[derive(Debug, Default)]
pub struct Profile {
    sampling: AtomicBool,
    /// A thread advances the engine's epoch while it samples.
    ticking: AtomicBool,
    stacks: Mutex<HashMap<Vec<u32>, u64>>,
    names: Mutex<HashMap<u32, String>>,
}

/// How often a sampled guest's engine ticks.
const SAMPLE_EVERY: Duration = Duration::from_millis(1);

impl Profile {
    pub fn new() -> Arc<Profile> {
        Arc::new(Profile::default())
    }

    pub fn sampling(&self) -> bool {
        self.sampling.load(Ordering::Relaxed)
    }

    /// Samples the guests it is attached to from now on: their engine's
    /// epoch advances about every millisecond while it does. Guests of an
    /// engine that does not check its epoch (only [`crate::dev_engine`]
    /// does) are never sampled.
    pub fn start(self: &Arc<Profile>, engine: &Engine) -> Result<()> {
        self.sampling.store(true, Ordering::Relaxed);
        if self.ticking.swap(true, Ordering::Relaxed) {
            return Ok(());
        }
        let (profile, engine) = (Arc::downgrade(self), engine.weak());
        let ticker = move || {
            let ticks = |profile: &Weak<Profile>| profile.upgrade().filter(|profile| profile.sampling());
            while let Some(sampling) = ticks(&profile) {
                let Some(engine) = engine.upgrade() else { break };
                engine.increment_epoch();
                drop((engine, sampling));
                std::thread::sleep(SAMPLE_EVERY);
            }
            if let Some(profile) = profile.upgrade() {
                profile.ticking.store(false, Ordering::Relaxed);
            }
        };
        if let Err(error) = std::thread::Builder::new().name("blot-profile".into()).spawn(ticker) {
            self.ticking.store(false, Ordering::Relaxed);
            self.sampling.store(false, Ordering::Relaxed);
            return Err(error.into());
        }
        Ok(())
    }

    /// Stops sampling; what it has stays.
    pub fn stop(&self) {
        self.sampling.store(false, Ordering::Relaxed);
    }

    /// Forgets its samples.
    pub fn clear(&self) {
        self.stacks.lock().unwrap().clear();
    }

    /// Takes a sample of a guest's stack, its functions named by `symbols`.
    pub(crate) fn sample(&self, backtrace: &WasmBacktrace, symbols: &Symbols) {
        // Frames come innermost first.
        let stack: Vec<u32> = backtrace.frames().iter().rev().map(FrameInfo::func_index).collect();
        if stack.is_empty() {
            return;
        }
        {
            let mut names = self.names.lock().unwrap();
            for frame in backtrace.frames() {
                names.entry(frame.func_index()).or_insert_with(|| function_name(symbols, frame.func_index()));
            }
        }
        *self.stacks.lock().unwrap().entry(stack).or_default() += 1;
    }

    pub fn samples(&self) -> u64 {
        self.stacks.lock().unwrap().values().sum()
    }

    /// Each stack sampled and how many samples found it.
    pub fn stacks(&self) -> Vec<(Vec<u32>, u64)> {
        self.stacks.lock().unwrap().iter().map(|(stack, samples)| (stack.clone(), *samples)).collect()
    }

    /// Function `index`'s name as its samples found it.
    pub fn name(&self, index: u32) -> String {
        self.names.lock().unwrap().get(&index).cloned().unwrap_or_else(|| format!("function {index}"))
    }
}
