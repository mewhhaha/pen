// Dynamic resolution: with a frame budget set (`Renderer::frame_budget`),
// the scene's render scale follows how long the GPU takes over a frame, to
// hold the budget: it drops when frames take longer, as far as half the
// frame's own scale, and rises again, a step at a time, once they take well
// under it. GPU time grows with the scene's pixels, the square of the scale,
// so a step down aims straight for the budget. Scales move in sixteenths and
// at most twice a second, since a new scale means new targets.
//
// The time comes from the GPU's timestamps where the device has them;
// otherwise from when the queue finishes each frame's work, the time from
// its submit (or the frame before finishing, if later) to then.
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::platform::Clock;

/// The least share of the frame's scale dynamic resolution draws at.
pub(crate) const LEAST: f32 = 0.5;

// Scales are whole sixteenths of the frame's.
const STEP: f32 = 1.0 / 16.0;

// Frames between changes: what a change does takes a few frames to show in
// the GPU's times, and new targets cost.
const HOLD: u32 = 30;

// It aims at this share of the budget, and rises once frames take under the
// lower share.
const AIM: f32 = 0.9;
const RISE: f32 = 0.7;

// How much of each new time the smoothed one takes.
const SMOOTHING: f32 = 0.25;

/// When frames went to the queue and when their work finished, in renderer
/// seconds, for devices without timestamps.
#[derive(Default)]
struct Finished {
    frames: VecDeque<(f64, Option<f64>)>,
    /// When the last frame measured finished.
    last: Option<f64>,
}

pub(crate) struct Resolution {
    budget: Option<f32>,
    factor: f32,
    smoothed: Option<f32>,
    held: u32,
    clock: Clock,
    finished: Arc<Mutex<Finished>>,
}

impl Resolution {
    pub fn new() -> Self {
        Self { budget: None, factor: 1.0, smoothed: None, held: 0, clock: Clock::new(), finished: Arc::default() }
    }

    /// The milliseconds of GPU time a frame may take; None draws at the
    /// frame's own scale.
    pub fn set_budget(&mut self, budget: Option<f32>) {
        self.budget = budget.filter(|budget| budget.is_finite() && *budget > 0.0);
        if self.budget.is_none() {
            self.factor = 1.0;
        }
        self.smoothed = None;
        self.held = 0;
    }

    /// Whether a budget is set, which wants the GPU's frames timed.
    pub fn budgeted(&self) -> bool {
        self.budget.is_some()
    }

    /// The share of the frame's scale to draw at, given the GPU time of a
    /// recent frame in milliseconds (its timestamps), if known.
    pub fn factor(&mut self, timed: Option<f32>) -> f32 {
        let Some(budget) = self.budget else { return 1.0 };
        if let Some(time) = timed.or_else(|| self.queued()).filter(|time| time.is_finite() && *time >= 0.0) {
            self.smoothed = Some(self.smoothed.map_or(time, |smoothed| smoothed + (time - smoothed) * SMOOTHING));
        }
        self.held = self.held.saturating_add(1);
        let Some(time) = self.smoothed else { return self.factor };
        if self.held < HOLD {
            return self.factor;
        }
        let wanted = if time > budget {
            // Pixels, and so time, go as the square of the scale.
            let aimed = self.factor * (budget * AIM / time).sqrt();
            ((aimed / STEP).floor() * STEP).max(LEAST)
        } else if time < budget * RISE {
            (self.factor + STEP).min(1.0)
        } else {
            self.factor
        };
        if wanted != self.factor {
            self.factor = wanted;
            self.held = 0;
            self.smoothed = None;
        }
        self.factor
    }

    /// Notes that a frame's work went to the queue, to time it when it
    /// finishes, where there are no timestamps to.
    pub fn submitted(&mut self, queue: &wgpu::Queue, timed: bool) {
        if self.budget.is_none() || timed {
            return;
        }
        let now = self.clock.now();
        let mut finished = self.finished.lock().unwrap();
        // A few frames' worth at most.
        while finished.frames.len() >= 8 {
            finished.frames.pop_front();
        }
        finished.frames.push_back((now, None));
        let (clock, record) = (self.clock, self.finished.clone());
        queue.on_submitted_work_done(move || {
            let mut finished = record.lock().unwrap();
            if let Some(frame) = finished.frames.iter_mut().find(|(_, done)| done.is_none()) {
                frame.1 = Some(clock.now());
            }
        });
    }

    /// The time the GPU took over the last frame that finished, from its
    /// submit, or the frame before it finishing if that was later.
    fn queued(&mut self) -> Option<f32> {
        let mut finished = self.finished.lock().unwrap();
        let mut time = None;
        while let Some(&(submitted, Some(done))) = finished.frames.front() {
            finished.frames.pop_front();
            let started = finished.last.map_or(submitted, |last| last.max(submitted));
            finished.last = Some(done);
            time = Some(crate::matrix::narrow((done - started) * 1000.0));
        }
        time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The factors `resolution` picks over `frames` frames that each take
    /// `time` at the full scale (and as much less as they have fewer pixels).
    fn run(resolution: &mut Resolution, time: f32, frames: u32) -> f32 {
        for _ in 0..frames {
            let factor = resolution.factor;
            resolution.factor(Some(time * factor * factor));
        }
        resolution.factor
    }

    #[test]
    fn scale_drops_to_hold_the_budget_and_rises_when_there_is_room() {
        let mut resolution = Resolution::new();
        // No budget: the frame's own scale, however slow.
        assert_eq!(resolution.factor(Some(100.0)), 1.0);
        resolution.set_budget(Some(10.0));
        // Twice the budget: one step to about 0.67 (sixteenths, floored).
        let dropped = run(&mut resolution, 20.0, HOLD);
        assert_eq!(dropped, 10.0 / 16.0);
        assert!(20.0 * dropped * dropped <= 10.0);
        // It holds there while frames fit.
        assert_eq!(run(&mut resolution, 20.0, 3 * HOLD), dropped);
        // Ten times the budget: down to half, no further.
        assert_eq!(run(&mut resolution, 100.0, 4 * HOLD), LEAST);
        // A quick GPU: back up a step at a time, to the frame's scale.
        assert_eq!(run(&mut resolution, 5.0, HOLD + 1), LEAST + STEP);
        assert_eq!(run(&mut resolution, 5.0, 20 * HOLD), 1.0);
        // Without times it keeps what it has; without a budget, the full scale.
        resolution.set_budget(Some(1.0));
        assert_eq!(resolution.factor(None), 1.0);
        run(&mut resolution, 4.0, 2 * HOLD);
        assert!(resolution.factor < 1.0);
        resolution.set_budget(None);
        assert_eq!(resolution.factor(Some(4.0)), 1.0);
    }
}
