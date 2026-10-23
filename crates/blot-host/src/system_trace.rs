//! Streaming Chrome/Perfetto spans and an aggregate per-system report.
//! Only an explicitly traced guest has this state. It never enters game saves.
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use wasmtime::{AsContextMut, Global};

use crate::State;

pub const MAGIC: f32 = -4670544.0;
const STAGES: [&str; 7] = ["Start", "Route", "Interface", "Event", "Update", "Render", "Settle"];

/// One capture shared by every instance, including reloads and time travel.
/// Events stream to disk up to `limit`; summaries continue after that limit.
pub struct SystemTrace {
    started: Instant,
    next_guest: AtomicU64,
    output: Mutex<Output>,
    enabled: AtomicBool,
    live: Mutex<Live>,
}
struct Output {
    file: Option<BufWriter<File>>,
    summary: PathBuf,
    limit: u64,
    written: u64,
    dropped: u64,
    totals: BTreeMap<(String, String, u32), Total>,
}
#[derive(Clone, Default)]
struct Total {
    calls: u64,
    micros: f64,
    max_micros: f64,
    bytes: u64,
    allocations: u64,
    aborted: u64,
}

#[derive(Default)]
struct Live {
    totals: BTreeMap<(u32, u32, String), Total>,
    snapshot: Vec<((u32, u32, String), Total)>,
}

impl SystemTrace {
    /// A bounded in-memory timing collector. No allocation instrumentation or files.
    pub fn live() -> Arc<Self> {
        Arc::new(Self {
            started: Instant::now(),
            next_guest: AtomicU64::new(1),
            enabled: AtomicBool::new(false),
            live: Mutex::new(Live::default()),
            output: Mutex::new(Output {
                file: None,
                summary: PathBuf::new(),
                limit: 0,
                written: 0,
                dropped: 0,
                totals: BTreeMap::new(),
            }),
        })
    }

    pub fn records_allocations(&self) -> bool {
        self.output.lock().unwrap().file.is_some()
    }

    /// Freeze one polling window so pagination cannot mix samples from different frames.
    pub fn systems(&self, enabled: bool, first: usize) -> (usize, Vec<f32>) {
        self.enabled.store(enabled, Ordering::Relaxed);
        let mut live = self.live.lock().unwrap();
        if !enabled {
            for total in live.totals.values_mut() {
                *total = Total::default();
            }
            live.snapshot.clear();
            return (0, Vec::new());
        }
        if first == 0 {
            live.snapshot = live.totals.iter_mut().map(|(key, total)| (key.clone(), std::mem::take(total))).collect();
        }
        let mut values = Vec::new();
        for ((stage, id, name), total) in live.snapshot.iter().skip(first).take(64) {
            let mean = if total.calls == 0 { -1.0 } else { total.micros / total.calls as f64 / 1000.0 };
            let peak = if total.calls == 0 { -1.0 } else { total.max_micros / 1000.0 };
            // The explicit Ink observer runs a sequential stage fold on one guest thread.
            values.extend([
                *stage as f32,
                *id as f32,
                total.calls as f32,
                mean as f32,
                peak as f32,
                0.0,
                name.chars().count() as f32,
            ]);
            values.extend(name.chars().map(|c| u32::from(c) as f32));
        }
        (live.snapshot.len(), values)
    }

    pub fn create(path: &Path, limit: u64) -> Result<Arc<Self>> {
        ensure!(limit > 0, "trace event limit must be positive");
        let mut file =
            BufWriter::new(File::create(path).with_context(|| format!("creating trace {}", path.display()))?);
        file.write_all(b"{\"traceEvents\":[")?;
        Ok(Arc::new(Self {
            started: Instant::now(),
            next_guest: AtomicU64::new(1),
            enabled: AtomicBool::new(false),
            live: Mutex::new(Live::default()),
            output: Mutex::new(Output {
                file: Some(file),
                summary: path.with_extension("summary.json"),
                limit,
                written: 0,
                dropped: 0,
                totals: BTreeMap::new(),
            }),
        }))
    }

    fn record(&self, guest: u64, frame: u64, span: Span, now: Instant, counts: Counts, aborted: bool) -> Result<()> {
        let micros = now.duration_since(span.started).as_secs_f64() * 1e6;
        let bytes = counts.bytes.wrapping_sub(span.counts.bytes);
        let allocations = counts.allocations.wrapping_sub(span.counts.allocations);
        if let Some(stage) = STAGES.iter().position(|stage| *stage == span.category) {
            let mut live = self.live.lock().unwrap();
            let key = (stage as u32, span.id, span.name.clone());
            ensure!(live.totals.contains_key(&key) || live.totals.len() < 4096, "too many live systems");
            let total = live.totals.entry(key).or_default();
            if self.enabled.load(Ordering::Relaxed) {
                total.calls += 1;
                total.micros += micros;
                total.max_micros = total.max_micros.max(micros);
            }
        }
        let mut output = self.output.lock().unwrap();
        if output.file.is_none() {
            return Ok(());
        }
        let key = (span.category.clone(), span.name.clone(), span.id);
        // A module may supply arbitrary names; bound report memory as well as the file.
        ensure!(output.totals.contains_key(&key) || output.totals.len() < 4096, "too many distinct traced systems");
        let total = output.totals.entry(key).or_default();
        total.calls += 1;
        total.micros += micros;
        total.max_micros = total.max_micros.max(micros);
        total.bytes = total.bytes.saturating_add(bytes);
        total.allocations = total.allocations.saturating_add(allocations);
        total.aborted += u64::from(aborted);
        if output.written >= output.limit {
            output.dropped += 1;
            return Ok(());
        }
        let event = json!({
            "name": span.name, "cat": span.category, "ph": "X", "pid": 1, "tid": guest,
            "ts": span.started.duration_since(self.started).as_secs_f64() * 1e6, "dur": micros,
            "args": { "system": span.id, "frame": frame, "allocated_bytes": bytes, "allocations": allocations, "aborted": aborted },
        });
        if output.written > 0 {
            output.file.as_mut().unwrap().write_all(b",\n")?;
        }
        serde_json::to_writer(output.file.as_mut().unwrap(), &event)?;
        output.written += 1;
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        if let Some(file) = &mut self.output.lock().unwrap().file {
            file.flush()?;
        }
        Ok(())
    }
}

impl Drop for SystemTrace {
    fn drop(&mut self) {
        let output = self.output.get_mut().unwrap();
        let Some(file) = &mut output.file else { return };
        let mut finish = || -> Result<()> {
            writeln!(file, "],\"displayTimeUnit\":\"ms\",\"droppedEvents\":{}}}", output.dropped)?;
            file.flush()?;
            let rows: Vec<Value> = output.totals.iter().map(|((stage, name, id), total)| json!({
                "stage": stage, "name": name, "system": id, "calls": total.calls,
                "total_us": total.micros, "mean_us": total.micros / total.calls as f64, "max_us": total.max_micros,
                "allocated_bytes": total.bytes, "mean_allocated_bytes": total.bytes as f64 / total.calls as f64,
                "allocations": total.allocations, "aborted": total.aborted,
            })).collect();
            let mut summary = BufWriter::new(File::create(&output.summary)?);
            serde_json::to_writer_pretty(&mut summary, &json!({ "dropped_events": output.dropped, "systems": rows }))?;
            summary.flush()?;
            Ok(())
        };
        if let Err(error) = finish() {
            eprintln!("cannot finish system trace: {error:#}");
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Counts {
    pub bytes: u64,
    pub allocations: u64,
}
struct Span {
    category: String,
    name: String,
    id: u32,
    started: Instant,
    counts: Counts,
}

pub(crate) struct Tracing {
    capture: Arc<SystemTrace>,
    guest: u64,
    frame: u64,
    globals: Option<[Global; 2]>,
    stack: Vec<Span>,
    command: Option<Span>,
}

impl Tracing {
    pub(crate) fn new(capture: Arc<SystemTrace>, globals: Option<[Global; 2]>) -> Self {
        let guest = capture.next_guest.fetch_add(1, Ordering::Relaxed);
        Self { capture, guest, frame: 0, globals, stack: Vec::new(), command: None }
    }

    pub(crate) fn counts(&self, mut store: impl AsContextMut<Data = State>) -> Counts {
        let Some(globals) = self.globals else { return Counts::default() };
        let mut get = |global: Global| global.get(&mut store).i64().unwrap().cast_unsigned();
        Counts { bytes: get(globals[0]), allocations: get(globals[1]) }
    }

    pub(crate) fn marker(&mut self, packet: &[f32], now: Instant, counts: Counts) -> Result<bool> {
        if packet.first() != Some(&MAGIC) {
            return Ok(false);
        }
        ensure!(packet.len() >= 4 && packet.len() <= 260, "invalid system trace marker");
        let integer = |value: f32, limit: u32| -> Result<u32> {
            ensure!(
                value.is_finite() && value >= 0.0 && value <= limit as f32 && value.fract() == 0.0,
                "invalid system trace marker number"
            );
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Ok(value as u32)
        };
        let phase = integer(packet[1], 1)?;
        let stage = integer(packet[2], 6)?;
        let id = integer(packet[3], 16_777_215)?;
        if phase == 1 {
            ensure!(self.stack.len() < 128, "system trace nesting limit exceeded");
            let mut name = String::new();
            for value in &packet[4..] {
                name.push(char::from_u32(integer(*value, 0x10ffff)?).context("invalid system trace name")?);
            }
            if name.is_empty() {
                name = format!("system {id}");
            }
            self.stack.push(Span { category: STAGES[stage as usize].into(), name, id, started: now, counts });
        } else {
            ensure!(packet.len() == 4, "invalid system trace end marker");
            let span = self.stack.last().context("system trace end without a begin")?;
            ensure!(span.id == id && span.category == STAGES[stage as usize], "mismatched system trace end");
            let span = self.stack.pop().unwrap();
            self.capture.record(self.guest, self.frame, span, now, counts, false)?;
        }
        Ok(true)
    }

    /// Begin measurement after the host bridge has returned its empty reply.
    pub(crate) fn resume_marker(&mut self, begin: bool, now: Instant, counts: Counts) {
        if begin && let Some(span) = self.stack.last_mut() {
            span.started = now;
            span.counts = counts;
        }
    }

    pub(crate) fn finish_command(&mut self, now: Instant, counts: Counts) -> Result<()> {
        ensure!(self.stack.is_empty(), "guest yielded a response inside a system trace");
        if let Some(span) = self.command.take() {
            self.capture.record(self.guest, self.frame, span, now, counts, false)?;
        }
        self.capture.flush()
    }

    pub(crate) fn resume_command(&mut self, packet: &[f32], now: Instant, counts: Counts) {
        let Some(operation) = packet.get(1) else { return };
        let name = match *operation {
            1.0 => "Create",
            2.0 => "Frame",
            3.0 => "Snapshot",
            4.0 => "Restore",
            5.0 => "Scene",
            6.0 => "LoadScene",
            _ => return,
        };
        if *operation == 2.0 {
            self.frame += 1;
        }
        self.command = Some(Span { category: "Guest".into(), name: name.into(), id: 0, started: now, counts });
    }

    pub(crate) fn abort(&mut self, now: Instant, counts: Counts) -> Result<()> {
        while let Some(span) = self.stack.pop() {
            self.capture.record(self.guest, self.frame, span, now, counts, true)?;
        }
        if let Some(span) = self.command.take() {
            self.capture.record(self.guest, self.frame, span, now, counts, true)?;
        }
        self.capture.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn capture(limit: u64) -> (PathBuf, Arc<SystemTrace>, Tracing) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "pen-trace-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let trace = SystemTrace::create(&path, limit).unwrap();
        let engine = crate::engine().unwrap();
        let mut store = wasmtime::Store::new(&engine, State::default());
        let mut global = || {
            Global::new(
                &mut store,
                wasmtime::GlobalType::new(wasmtime::ValType::I64, wasmtime::Mutability::Var),
                wasmtime::Val::I64(0),
            )
            .unwrap()
        };
        let tracing = Tracing::new(trace.clone(), Some([global(), global()]));
        (path, trace, tracing)
    }

    #[test]
    fn spans_exclude_bridge_setup_keep_nested_totals_and_bound_only_the_timeline() {
        let (path, trace, mut tracing) = capture(1);
        let at = trace.started;
        tracing.resume_command(&[1.0, 2.0], at, Counts::default());
        tracing.marker(&[MAGIC, 1.0, 4.0, 7.0, 65.0], at, Counts::default()).unwrap();
        let counts = Counts { bytes: 100, allocations: 2 };
        tracing.resume_marker(true, at + Duration::from_micros(5), counts);
        tracing
            .marker(&[MAGIC, 0.0, 4.0, 7.0], at + Duration::from_micros(15), Counts { bytes: 164, allocations: 4 })
            .unwrap();
        tracing.finish_command(at + Duration::from_micros(20), Counts { bytes: 164, allocations: 4 }).unwrap();
        drop(tracing);
        drop(trace);
        let events: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(events["traceEvents"].as_array().unwrap().len(), 1);
        assert_eq!(events["droppedEvents"], 1);
        let event = &events["traceEvents"][0];
        assert_eq!(event["name"], "A");
        assert!((event["dur"].as_f64().unwrap() - 10.0).abs() < 1e-9);
        assert_eq!(event["args"]["allocated_bytes"], 64);
        assert_eq!(event["args"]["allocations"], 2);
        assert_eq!(event["args"]["frame"], 1);
        let summary: Value =
            serde_json::from_slice(&std::fs::read(path.with_extension("summary.json")).unwrap()).unwrap();
        assert_eq!(summary["systems"].as_array().unwrap().len(), 2, "summary includes the capped frame span");
        std::fs::remove_file(path.with_extension("summary.json")).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn traps_close_active_spans_and_malformed_markers_are_rejected() {
        let (path, trace, mut tracing) = capture(10);
        let at = trace.started;
        for packet in
            [vec![MAGIC], vec![MAGIC, 1.0, f32::NAN, 0.0], vec![MAGIC, 1.0, 7.0, 0.0], vec![MAGIC, 0.0, 4.0, 2.0]]
        {
            assert!(tracing.marker(&packet, at, Counts::default()).is_err());
        }
        tracing.marker(&[MAGIC, 1.0, 4.0, 3.0], at, Counts::default()).unwrap();
        tracing.abort(at + Duration::from_micros(4), Counts { bytes: 20, allocations: 1 }).unwrap();
        drop(tracing);
        drop(trace);
        let events: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(events["traceEvents"][0]["args"]["aborted"], true);
        assert_eq!(events["traceEvents"][0]["args"]["allocated_bytes"], 20);
        std::fs::remove_file(path.with_extension("summary.json")).unwrap();
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn live_windows_are_bounded_paged_and_stop_accumulating_when_closed() {
        let capture = SystemTrace::live();
        let at = capture.started;
        let record = |id, duration| {
            let span = Span {
                category: "Update".into(),
                name: format!("system {id}"),
                id,
                started: at,
                counts: Counts::default(),
            };
            capture.record(1, 1, span, at + Duration::from_micros(duration), Counts::default(), false).unwrap();
        };
        for id in 0..130 {
            record(id, 100);
        }
        let (count, first) = capture.systems(true, 0);
        assert_eq!(count, 130);
        assert_eq!(&first[..7], &[4.0, 0.0, 0.0, -1.0, -1.0, 0.0, 8.0]);
        for id in 0..130 {
            record(id, 200);
            record(id, 400);
        }
        let (_, first) = capture.systems(true, 0);
        assert_eq!(&first[..6], &[4.0, 0.0, 2.0, 0.3, 0.4, 0.0]);
        record(64, 900);
        let (_, second) = capture.systems(true, 64);
        assert_eq!(&second[..6], &[4.0, 64.0, 2.0, 0.3, 0.4, 0.0]);
        let (_, third) = capture.systems(true, 128);
        assert_eq!(third[1], 128.0);
        assert_eq!(third.len(), 34); // two headers plus ten characters each
        capture.systems(false, 0);
        record(0, 3000);
        let (_, first) = capture.systems(true, 0);
        assert_eq!(first[2], 0.0);
        assert_eq!(capture.live.lock().unwrap().totals.len(), 130);
        assert!(capture.output.lock().unwrap().file.is_none());
        assert!(capture.output.lock().unwrap().totals.is_empty());
    }
}
