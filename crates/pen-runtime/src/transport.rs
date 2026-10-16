//! What the player, a script or the dev tools asked of the game's time:
//! whether it plays or is paused (and how many frames to step while
//! paused), recorded frames playing, and time travel and input recordings
//! waiting for the guest to be idle. The host's frame loop carries these
//! out; this only keeps them straight.
use std::collections::VecDeque;
use std::path::PathBuf;

/// Playing, or paused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Run {
    Playing,
    /// Paused, with `steps` frames still to run one at a time, and whether
    /// a valid reload plays on (a rejected frame paused live play).
    Paused {
        steps: u32,
        resume_after_reload: bool,
    },
}

/// What a replay does when its frames run out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Replaying {
    /// `--replay`: stop (headless) or pause (a window).
    File,
    /// `replay_recent`: play on, live.
    Recent,
}

/// Time travel, once the guest is idle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rewind {
    /// Go to this frame, paused.
    To(u64),
    /// Play this many seconds of recorded input again.
    Replay(f64),
}

#[derive(Debug)]
pub struct Transport {
    run: Run,
    replaying: Option<Replaying>,
    rewinds: VecDeque<Rewind>,
    /// Start recording the input (to a file) or stop.
    record: Option<Option<PathBuf>>,
}

impl Default for Transport {
    fn default() -> Transport {
        Transport { run: Run::Playing, replaying: None, rewinds: VecDeque::new(), record: None }
    }
}

impl Transport {
    pub fn run(&self) -> Run {
        self.run
    }

    pub fn paused(&self) -> bool {
        self.run != Run::Playing
    }

    /// Pauses (a reload no longer plays on); whether it was playing.
    pub fn pause(&mut self) -> bool {
        match self.run {
            Run::Playing => {
                self.run = Run::Paused { steps: 0, resume_after_reload: false };
                true
            }
            Run::Paused { steps, .. } => {
                self.run = Run::Paused { steps, resume_after_reload: false };
                false
            }
        }
    }

    /// Plays on, forgetting the frames left to step; whether it was paused.
    pub fn resume(&mut self) -> bool {
        let paused = self.paused();
        self.run = Run::Playing;
        paused
    }

    /// Runs `frames` more frames one at a time and stays paused (pausing
    /// first); whether it was playing.
    pub fn step(&mut self, frames: u32) -> bool {
        let was_playing = self.pause();
        if let Run::Paused { steps, .. } = &mut self.run {
            *steps = steps.saturating_add(frames);
        }
        was_playing
    }

    /// A frame was rejected: paused with nothing to step, to play on after
    /// a valid reload if it was playing; whether it was.
    pub fn reject(&mut self) -> bool {
        let was_playing = self.pause();
        self.run = Run::Paused { steps: 0, resume_after_reload: was_playing };
        was_playing
    }

    /// A valid new build runs: whether to play on.
    pub fn reloaded(&self) -> bool {
        matches!(self.run, Run::Paused { resume_after_reload: true, .. })
    }

    /// Whether a frame runs now: playing, or a step taken while paused.
    pub fn take_frame(&mut self) -> bool {
        match &mut self.run {
            Run::Playing => true,
            Run::Paused { steps, .. } if *steps > 0 => {
                *steps -= 1;
                true
            }
            Run::Paused { .. } => false,
        }
    }

    /// Goes to `frame`, paused there with nothing to step, once the guest is
    /// idle; whether it was playing.
    pub fn rewind_to(&mut self, frame: u64) -> bool {
        let was_playing = self.pause();
        self.run = Run::Paused { steps: 0, resume_after_reload: false };
        self.rewinds.push_back(Rewind::To(frame));
        was_playing
    }

    /// Goes back to `frame` once the guest is idle, playing or paused as it
    /// is (a program stopped at its deadline starts again from there).
    pub fn restore_at(&mut self, frame: u64) {
        self.rewinds.push_back(Rewind::To(frame));
    }

    /// Plays the last `seconds` again, once the guest is idle.
    pub fn replay_recent(&mut self, seconds: f64) {
        self.rewinds.push_back(Rewind::Replay(seconds));
    }

    /// The frame time travel queued now starts from, or `position`.
    pub fn target(&self, position: u64) -> u64 {
        match self.rewinds.back() {
            Some(Rewind::To(frame)) => *frame,
            _ => position,
        }
    }

    /// Time travel is queued.
    pub fn travelling(&self) -> bool {
        !self.rewinds.is_empty()
    }

    /// The next time travel to do: of rewinds queued one after another (a
    /// held key), the last.
    pub fn next_rewind(&mut self) -> Option<Rewind> {
        loop {
            let rewind = self.rewinds.pop_front()?;
            if !matches!((rewind, self.rewinds.front()), (Rewind::To(_), Some(Rewind::To(_)))) {
                return Some(rewind);
            }
        }
    }

    /// Recorded frames play (of `kind`, unless some already do), from now
    /// on, with nothing to step; whether it was paused.
    pub fn replay(&mut self, kind: Replaying) -> bool {
        self.replaying.get_or_insert(kind);
        self.resume()
    }

    pub fn replaying(&self) -> Option<Replaying> {
        self.replaying
    }

    /// The recorded frames ran out: what was playing.
    pub fn replay_ended(&mut self) -> Option<Replaying> {
        self.replaying.take()
    }

    /// Start recording the input to `path`, or with None stop, once the
    /// guest is idle.
    pub fn record(&mut self, path: Option<PathBuf>) {
        self.record = Some(path);
    }

    /// A recording to start or stop, once.
    pub fn take_record(&mut self) -> Option<Option<PathBuf>> {
        self.record.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAUSED: Run = Run::Paused { steps: 0, resume_after_reload: false };

    #[test]
    fn pauses_steps_and_plays_on() {
        let mut transport = Transport::default();
        assert!(transport.take_frame() && transport.take_frame());
        assert!(transport.pause());
        assert!(!transport.pause());
        assert!(!transport.take_frame());
        // Steps add up, run one a frame, and stay paused.
        assert!(!transport.step(1));
        transport.step(2);
        assert_eq!(transport.run(), Run::Paused { steps: 3, resume_after_reload: false });
        assert!((0..3).all(|_| transport.take_frame()));
        assert!(!transport.take_frame() && transport.paused());
        // Stepping while playing pauses first.
        assert!(transport.resume());
        assert!(transport.step(1));
        assert!(transport.take_frame() && !transport.take_frame());
        // Playing on forgets what was left to step.
        transport.step(5);
        assert!(transport.resume() && !transport.resume());
        assert_eq!(transport.run(), Run::Playing);
    }

    #[test]
    fn a_rejected_frame_plays_on_after_a_reload_only_if_it_was_playing() {
        let mut transport = Transport::default();
        assert!(transport.reject());
        assert!(transport.reloaded());
        assert!(!transport.take_frame());
        // A step while paused by a rejection is the player's: it stays paused.
        transport.step(1);
        assert!(!transport.reloaded());
        assert!(transport.take_frame());
        assert!(!transport.reject());
        assert_eq!(transport.run(), PAUSED);
        assert!(!transport.reloaded());
        // Pausing by hand after a rejection keeps it paused too.
        transport.resume();
        transport.reject();
        transport.pause();
        assert!(!transport.reloaded());
        // Going back to where a stopped program was keeps a rejection's
        // promise to play on.
        transport.resume();
        transport.reject();
        transport.restore_at(7);
        assert!(transport.reloaded() && transport.travelling());
        assert_eq!(transport.next_rewind(), Some(Rewind::To(7)));
    }

    #[test]
    fn rewinds_pause_and_queue_and_held_keys_go_to_the_last() {
        let mut transport = Transport::default();
        transport.step(4);
        assert!(!transport.rewind_to(10));
        assert_eq!(transport.run(), PAUSED);
        assert_eq!(transport.target(50), 10);
        transport.rewind_to(transport.target(50) - 1);
        transport.rewind_to(transport.target(50) - 1);
        transport.replay_recent(5.0);
        assert!(transport.travelling());
        assert_eq!(transport.target(50), 50, "a replay starts where the frames are");
        transport.rewind_to(3);
        assert_eq!(transport.next_rewind(), Some(Rewind::To(8)));
        assert_eq!(transport.next_rewind(), Some(Rewind::Replay(5.0)));
        assert_eq!(transport.next_rewind(), Some(Rewind::To(3)));
        assert_eq!(transport.next_rewind(), None);
        assert!(!transport.travelling());
    }

    #[test]
    fn replays_play_and_end_once() {
        let mut transport = Transport::default();
        transport.reject();
        assert!(transport.replay(Replaying::Recent));
        assert_eq!(transport.run(), Run::Playing);
        // A replay of a file goes on being one.
        let mut file = Transport::default();
        file.replay(Replaying::File);
        file.pause();
        assert!(file.replay(Replaying::Recent));
        assert_eq!(file.replay_ended(), Some(Replaying::File));
        assert_eq!(file.replay_ended(), None);
        assert_eq!(transport.replaying(), Some(Replaying::Recent));
    }

    #[test]
    fn a_recording_request_is_taken_once() {
        let mut transport = Transport::default();
        assert_eq!(transport.take_record(), None);
        transport.record(Some(PathBuf::from("a.penrec")));
        transport.record(None);
        assert_eq!(transport.take_record(), Some(None));
        assert_eq!(transport.take_record(), None);
    }
}
