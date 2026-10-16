//! The game's clock (`Clock`).
use web_time::Instant;

/// The game's clock: real time, or a steady `1/fps` per frame however long
/// frames take, for recordings and repeatable headless runs. Paused, real
/// time stops counting, and a frame stepped is `STEP_SECONDS` long.
pub struct Clock {
    kind: ClockKind,
    /// The last frame's time.
    last: f64,
    /// Real time paused at `last`.
    frozen: bool,
    /// Real time not counted, while paused.
    skipped: f64,
}

enum ClockKind {
    Real(Instant),
    Steady { fps: f64, frame: u64 },
}

/// The game time a frame stepped while paused takes on a real clock.
pub const STEP_SECONDS: f64 = 1.0 / 60.0;

impl Clock {
    pub fn real() -> Clock {
        Clock { kind: ClockKind::Real(Instant::now()), last: 0.0, frozen: false, skipped: 0.0 }
    }

    pub fn steady(fps: f64) -> Clock {
        Clock { kind: ClockKind::Steady { fps, frame: 0 }, last: 0.0, frozen: false, skipped: 0.0 }
    }

    pub fn is_steady(&self) -> bool {
        matches!(self.kind, ClockKind::Steady { .. })
    }

    /// The next frame's time.
    pub fn advance(&mut self) -> f64 {
        let timestamp = match &mut self.kind {
            ClockKind::Real(_) if self.frozen => self.last + STEP_SECONDS,
            ClockKind::Real(started) => started.elapsed().as_secs_f64() - self.skipped,
            ClockKind::Steady { fps, frame } => {
                let timestamp = *frame as f64 / *fps;
                *frame += 1;
                timestamp
            }
        };
        self.last = timestamp;
        timestamp
    }

    /// The time now, for the next frame's elapsed time to count from.
    pub fn now(&self) -> f64 {
        match &self.kind {
            ClockKind::Real(started) if !self.frozen => started.elapsed().as_secs_f64() - self.skipped,
            _ => self.last,
        }
    }

    pub fn pause(&mut self) {
        self.frozen = true;
    }

    pub fn resume(&mut self) {
        if let ClockKind::Real(started) = &self.kind
            && self.frozen
        {
            self.skipped = started.elapsed().as_secs_f64() - self.last;
        }
        self.frozen = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_steady_clock_counts_frames_whatever_they_take() {
        let mut clock = Clock::steady(30.0);
        assert!(clock.is_steady());
        assert_eq!([clock.advance(), clock.advance(), clock.advance()], [0.0, 1.0 / 30.0, 2.0 / 30.0]);
        assert_eq!(clock.now(), 2.0 / 30.0);
        // Paused or not, its frames are as long.
        clock.pause();
        assert_eq!(clock.advance(), 3.0 / 30.0);
        clock.resume();
        assert_eq!(clock.advance(), 4.0 / 30.0);
    }

    #[test]
    fn a_real_clock_stops_while_paused_and_steps_a_frame_at_a_time() {
        let mut clock = Clock::real();
        assert!(!clock.is_steady());
        let first = clock.advance();
        clock.pause();
        let stepped = clock.advance();
        assert_eq!(stepped, first + STEP_SECONDS);
        assert_eq!(clock.advance(), stepped + STEP_SECONDS);
        assert_eq!(clock.now(), stepped + STEP_SECONDS);
        // Playing on, it counts from where it stopped, not from the time
        // that passed while paused.
        std::thread::sleep(std::time::Duration::from_millis(50));
        clock.resume();
        let after = clock.advance();
        assert!(after >= stepped + STEP_SECONDS && after < stepped + STEP_SECONDS + 0.04, "{after}");
    }
}
