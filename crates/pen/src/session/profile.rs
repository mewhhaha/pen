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
             render+present {:.2}, housekeeping {:.2}, total {:.2}; frame interval {:.2}; packet {:.0} KB",
            self.timings.len(),
            if overlapped { "overlapped " } else { "" },
            phase(|t| t.guest_ms),
            phase(|t| t.wait_ms),
            phase(|t| t.decode_ms),
            phase(|t| t.validate_ms),
            phase(|t| t.submit_ms),
            phase(|t| t.housekeeping_ms),
            phase(|t| t.total_ms),
            median(self.intervals.clone()),
            phase(|t| t.packet_bytes as f64) / 1024.0,
        );
    }
}

/// PEN_FRAME_TIMES: frame `frame`'s timings, a line on stderr.
pub(super) fn print(frame: usize, timing: &FrameTimings) {
    eprintln!(
        "frame {frame}: guest {:.3} wait {:.3} decode {:.3} validate {:.3} submit {:.3} housekeeping {:.3} total {:.3} ms",
        timing.guest_ms,
        timing.wait_ms,
        timing.decode_ms,
        timing.validate_ms,
        timing.submit_ms,
        timing.housekeeping_ms,
        timing.total_ms,
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
}
