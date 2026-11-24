//! Measuring frames: PEN_PROFILE_FRAMES's medians (`Profile`), and
//! PEN_FRAME_TIMES's line per frame.
use std::time::Instant;

use anyhow::Result;

use crate::game::FrameTimings;

/// PEN_PROFILE_FRAMES: after PEN_PROFILE_WARMUP frames (90), sample this
/// many, report each phase's median and stop.
pub struct Profile {
    warmup: usize,
    wanted: usize,
    timings: Vec<FrameTimings>,
    intervals: Vec<f64>,
    previous: Option<Instant>,
}

impl Profile {
    pub(super) fn from_env() -> Result<Option<Profile>> {
        let Some(wanted) = std::env::var("PEN_PROFILE_FRAMES").ok() else { return Ok(None) };
        let warmup = std::env::var("PEN_PROFILE_WARMUP").ok().map(|value| value.parse()).transpose()?;
        Ok(Some(Profile {
            warmup: warmup.unwrap_or(90),
            wanted: wanted.parse()?,
            timings: Vec::new(),
            intervals: Vec::new(),
            previous: None,
        }))
    }

    /// Records a frame; true once enough have been sampled.
    pub(super) fn sample(&mut self, frame: usize, timings: Option<FrameTimings>) -> bool {
        let now = Instant::now();
        if frame > self.warmup {
            if let Some(previous) = self.previous {
                self.intervals.push((now - previous).as_secs_f64() * 1000.0);
            }
            self.timings.extend(timings);
        }
        self.previous = Some(now);
        self.timings.len() >= self.wanted
    }

    pub(super) fn report(&self, overlapped: bool) {
        let median = |mut values: Vec<f64>| {
            values.sort_by(f64::total_cmp);
            values.get(values.len() / 2).copied().unwrap_or(0.0)
        };
        let phase = |pick: fn(&FrameTimings) -> f64| median(self.timings.iter().map(pick).collect());
        eprintln!(
            "profile of {} {}frames (medians, ms): guest {:.2} (waited for {:.2}), decode {:.2}, validate {:.2}, \
             render {:.2}, acquire {:.2}, present {:.2}, pace {:.2}, housekeeping {:.2}, total {:.2}; frame interval {:.2}; packet {:.0} KB",
            self.timings.len(),
            if overlapped { "overlapped " } else { "" },
            phase(|t| t.guest_ms),
            phase(|t| t.wait_ms),
            phase(|t| t.decode_ms),
            phase(|t| t.validate_ms),
            phase(|t| t.draw.render_ms),
            phase(|t| t.draw.acquire_ms),
            phase(|t| t.draw.present_ms),
            phase(|t| t.draw.pace_ms),
            phase(|t| t.housekeeping_ms),
            phase(|t| t.total_ms),
            median(self.intervals.clone()),
            phase(|t| t.packet_bytes as f64) / 1024.0,
        );
        if let Ok(path) = std::env::var("PEN_TRACE_OUT")
            && !path.is_empty()
            && let Err(err) = write_chrome_trace_with_intervals(std::path::Path::new(&path), &self.timings, &self.intervals)
        {
            eprintln!("failed writing Chrome trace to {path}: {err}");
        }
    }
}

/// Formats `timings` as a Chrome Trace Event Format (`chrome://tracing` /
/// Perfetto) JSON document with separate tracks for overlapped guest work
/// (`tid: 1`), host decode/validate/housekeeping (`tid: 2`), GPU draw (`tid: 3`),
/// whole-frame spans (`tid: 0`), and packet byte counters.
#[allow(dead_code)]
pub fn chrome_trace_json(timings: &[FrameTimings]) -> String {
    chrome_trace_json_with_intervals(timings, &[])
}

/// Formats `timings` using measured frame-to-frame `intervals` (in ms) when
/// available so frame start timestamps reflect real pacing instead of synthetic sums.
pub fn chrome_trace_json_with_intervals(timings: &[FrameTimings], intervals_ms: &[f64]) -> String {
    let mut events = Vec::with_capacity(timings.len() * 11);
    let mut cursor_us = 0.0f64;
    for (idx, t) in timings.iter().enumerate() {
        let frame_start = cursor_us;
        let total_us = t.total_ms.max(0.0) * 1000.0;
        let guest_us = t.guest_ms.max(0.0) * 1000.0;
        events.push(serde_json::json!({
            "name": "frame",
            "cat": "frame",
            "ph": "X",
            "ts": frame_start,
            "dur": total_us,
            "pid": 1,
            "tid": 0,
            "args": { "frame": idx }
        }));
        // Guest execution runs on its own track (overlapping host wait when overlapped).
        events.push(serde_json::json!({
            "name": "guest",
            "cat": "frame",
            "ph": "X",
            "ts": frame_start,
            "dur": guest_us,
            "pid": 1,
            "tid": 1,
            "args": { "frame": idx }
        }));
        let host_phases = [
            ("wait", 2u32, t.wait_ms),
            ("decode", 2, t.decode_ms),
            ("validate", 2, t.validate_ms),
            ("render", 3, t.draw.render_ms),
            ("acquire", 3, t.draw.acquire_ms),
            ("present", 3, t.draw.present_ms),
            ("pace", 3, t.draw.pace_ms),
            ("housekeeping", 2, t.housekeeping_ms),
        ];
        let mut phase_ts = frame_start;
        for (name, tid, dur_ms) in host_phases {
            let dur_us = dur_ms.max(0.0) * 1000.0;
            events.push(serde_json::json!({
                "name": name,
                "cat": "frame",
                "ph": "X",
                "ts": phase_ts,
                "dur": dur_us,
                "pid": 1,
                "tid": tid,
                "args": { "frame": idx }
            }));
            phase_ts += dur_us;
        }
        events.push(serde_json::json!({
            "name": "packet_bytes",
            "cat": "counters",
            "ph": "C",
            "ts": frame_start,
            "pid": 1,
            "args": { "bytes": t.packet_bytes }
        }));
        let measured = intervals_ms.get(idx).copied().filter(|v| v.is_finite() && *v > 0.0);
        let fallback_ms = ((phase_ts - frame_start) / 1000.0).max(t.total_ms).max(t.guest_ms);
        let step_ms = measured.unwrap_or(fallback_ms).max(0.001);
        cursor_us = frame_start + step_ms * 1000.0;
    }
    serde_json::json!({
        "displayTimeUnit": "ms",
        "traceEvents": events
    })
    .to_string()
}

/// Writes a Chrome Trace Event Format (`chrome://tracing` / Perfetto) JSON file.
#[allow(dead_code)]
pub fn write_chrome_trace(path: &std::path::Path, timings: &[FrameTimings]) -> Result<()> {
    write_chrome_trace_with_intervals(path, timings, &[])
}

pub fn write_chrome_trace_with_intervals(
    path: &std::path::Path,
    timings: &[FrameTimings],
    intervals_ms: &[f64],
) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, chrome_trace_json_with_intervals(timings, intervals_ms))?;
    Ok(())
}

/// PEN_FRAME_TIMES: frame `frame`'s timings, a line on stderr.
pub(super) fn print(frame: usize, timing: &FrameTimings, stats: pen_render::RenderStats) {
    eprintln!(
        "frame {frame}: guest {:.3} wait {:.3} decode {:.3} validate {:.3} render {:.3} acquire {:.3} present {:.3} pace {:.3} housekeeping {:.3} total {:.3} ms uploads {} {} fields {} {} {}",
        timing.guest_ms,
        timing.wait_ms,
        timing.decode_ms,
        timing.validate_ms,
        timing.draw.render_ms,
        timing.draw.acquire_ms,
        timing.draw.present_ms,
        timing.draw.pace_ms,
        timing.housekeeping_ms,
        timing.total_ms,
        stats.instance_upload_bytes,
        stats.instance_upload_calls,
        stats.field_upload_bytes,
        stats.field_upload_calls,
        stats.field_input_texels,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_after_the_warmup_until_it_has_enough() {
        let mut profile = Profile { warmup: 2, wanted: 3, timings: Vec::new(), intervals: Vec::new(), previous: None };
        let timed = Some(FrameTimings { guest_ms: 1.0, ..FrameTimings::default() });
        assert!(!profile.sample(1, timed) && !profile.sample(2, timed));
        assert!(profile.timings.is_empty());
        // A frame without timings (not measured) counts no sample.
        assert!(!profile.sample(3, None));
        assert!(!profile.sample(4, timed) && !profile.sample(5, timed));
        assert!(profile.sample(6, timed));
        assert_eq!((profile.timings.len(), profile.intervals.len()), (3, 4));
    }

    #[test]
    fn chrome_trace_json_emits_valid_perfetto_events() {
        let t = FrameTimings {
            guest_ms: 1.5,
            wait_ms: 0.4,
            decode_ms: 0.2,
            total_ms: 6.0,
            packet_bytes: 4096,
            draw: pen_runtime::game::DrawTimings { render_ms: 3.0, ..Default::default() },
            ..Default::default()
        };
        let json: serde_json::Value =
            serde_json::from_str(&chrome_trace_json_with_intervals(&[t, t], &[16.67, 16.67])).unwrap();
        let events = json["traceEvents"].as_array().unwrap();
        assert_eq!(events.len(), 22, "2 frames * (frame + guest + 8 host/gpu spans + 1 counter)");
        assert_eq!(events[0]["name"], "frame");
        assert_eq!(events[0]["tid"], 0);
        assert_eq!(events[1]["name"], "guest");
        assert_eq!(events[1]["tid"], 1);
        assert_eq!(events[1]["ts"], 0.0);
        assert_eq!(events[1]["dur"], 1500.0);
        // Host wait starts at frame_start on tid 2 (overlapping guest on tid 1, not stacked behind guest).
        assert_eq!(events[2]["name"], "wait");
        assert_eq!(events[2]["tid"], 2);
        assert_eq!(events[2]["ts"], 0.0);
        assert_eq!(events[10]["ph"], "C");
        assert_eq!(events[10]["args"]["bytes"], 4096);
        // Second frame uses the measured 16.67 ms interval.
        assert_eq!(events[11]["ts"], 16_670.0);
    }
}
