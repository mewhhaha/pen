//! What both hosts' mixes tell the dev tools as they play (DevtoolsOperation
//! `audio`, `mute` and `solo`): each bus's peak and RMS, the buses the
//! developer muted or soloed, and how late the output plays. Muting and
//! soloing are the developer's alone: they change what the host plays, never
//! the game's frames or a recording's sound.
use pen_protocol::boundary::MAX_BUSES;
use pen_protocol::{devtools_audio, devtools_bus};

use crate::services::Answer;

/// The developer's mute and solo, a bit for each bus.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Monitor {
    pub muted: u32,
    pub soloed: u32,
}

impl Monitor {
    /// Whether bus `index` is heard: not muted, and soloed while any bus
    /// is. The master is only ever muted: soloing others leaves its own
    /// voices, and what the soloed buses pass through it, heard.
    pub fn audible(&self, index: usize) -> bool {
        let bit = 1u32 << index.min(31);
        self.muted & bit == 0 && (index == 0 || self.soloed == 0 || self.soloed & bit != 0)
    }

    /// Mutes or unmutes bus `index`, or solos it or not.
    pub fn set(&mut self, index: usize, mute: bool, on: bool) {
        let bits = if mute { &mut self.muted } else { &mut self.soloed };
        let bit = 1u32 << index.min(31);
        if on {
            *bits |= bit;
        } else {
            *bits &= !bit;
        }
    }

    /// The monitor as one number, for an atomic.
    pub fn bits(self) -> u64 {
        u64::from(self.muted) | u64::from(self.soloed) << 32
    }

    pub fn from_bits(bits: u64) -> Monitor {
        let half = |bits: u64| u32::try_from(bits & 0xffff_ffff).unwrap_or_default();
        Monitor { muted: half(bits), soloed: half(bits >> 32) }
    }
}

/// One bus's output over the samples since it was last read: the peak and
/// the sum of squares on each side.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Meter {
    pub peak: [f32; 2],
    pub squares: [f64; 2],
    pub samples: u64,
}

impl Meter {
    pub fn add(&mut self, frame: [f32; 2]) {
        for ((peak, squares), sample) in self.peak.iter_mut().zip(&mut self.squares).zip(frame) {
            *peak = peak.max(sample.abs());
            *squares += f64::from(sample) * f64::from(sample);
        }
        self.samples += 1;
    }

    /// Both meters' samples as one's.
    pub fn merge(&mut self, other: &Meter) {
        for ((peak, squares), (other_peak, other_squares)) in
            self.peak.iter_mut().zip(&mut self.squares).zip(other.peak.iter().zip(other.squares))
        {
            *peak = peak.max(*other_peak);
            *squares += other_squares;
        }
        self.samples += other.samples;
    }

    #[allow(clippy::cast_possible_truncation, reason = "a level to show")]
    pub fn rms(&self) -> [f32; 2] {
        if self.samples == 0 {
            return [0.0; 2];
        }
        self.squares.map(|squares| (squares / self.samples as f64).sqrt() as f32)
    }
}

/// A bus as the dev tools see it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BusReport {
    pub peak: [f32; 2],
    pub rms: [f32; 2],
    /// A frame uses it.
    pub used: bool,
    pub muted: bool,
    pub soloed: bool,
}

/// What the host's mix says of itself.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AudioReport {
    /// From a frame's sound reaching the mixer to it leaving the speakers,
    /// in milliseconds, when the host can tell.
    pub latency_ms: Option<f64>,
    pub rate: f64,
    /// The device's buffer in samples, unless the system chose it.
    pub buffer: Option<u32>,
    /// Voices and sounds playing, and those of them streamed.
    pub voices: usize,
    pub streams: usize,
    /// How often a stream ran dry so far.
    pub underruns: u64,
    /// Every bus, the master first.
    pub buses: Vec<BusReport>,
}

impl AudioReport {
    /// The buses as `meters` measured them (`used` saying which a frame
    /// uses) and `monitor` has them.
    pub fn buses(meters: &[Meter], used: &[bool], monitor: Monitor) -> Vec<BusReport> {
        (0..MAX_BUSES)
            .map(|index| {
                let meter = meters.get(index).copied().unwrap_or_default();
                let bit = 1u32 << index;
                BusReport {
                    peak: meter.peak,
                    rms: meter.rms(),
                    used: used.get(index).copied().unwrap_or(false),
                    muted: monitor.muted & bit != 0,
                    soloed: monitor.soloed & bit != 0,
                }
            })
            .collect()
    }

    /// The dev tools' answer: DEVTOOLS_AUDIO's numbers, then DEVTOOLS_BUS's
    /// for each bus.
    #[allow(clippy::cast_possible_truncation, reason = "numbers to show")]
    pub fn answer(&self) -> Answer {
        let mut values = vec![0.0; devtools_audio::NAMES.len()];
        let mut set = |at: u32, value: f64| values[at as usize] = value as f32;
        set(devtools_audio::LATENCY, self.latency_ms.unwrap_or(-1.0));
        set(devtools_audio::RATE, self.rate);
        set(devtools_audio::BUFFER, self.buffer.map_or(-1.0, f64::from));
        set(devtools_audio::VOICES, self.voices as f64);
        set(devtools_audio::STREAMS, self.streams as f64);
        set(devtools_audio::UNDERRUNS, self.underruns as f64);
        set(devtools_audio::BUSES, self.buses.len() as f64);
        for bus in &self.buses {
            let mut numbers = [0.0; devtools_bus::NAMES.len()];
            numbers[devtools_bus::PEAK_LEFT as usize] = bus.peak[0];
            numbers[devtools_bus::PEAK_RIGHT as usize] = bus.peak[1];
            numbers[devtools_bus::RMS_LEFT as usize] = bus.rms[0];
            numbers[devtools_bus::RMS_RIGHT as usize] = bus.rms[1];
            numbers[devtools_bus::STATE as usize] =
                f32::from(u8::from(bus.used) | u8::from(bus.muted) << 1 | u8::from(bus.soloed) << 2);
            values.extend(numbers);
        }
        Answer { values, ..Answer::ok(0.0) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solo_and_mute_say_what_is_heard() {
        let mut monitor = Monitor::default();
        assert!((0..MAX_BUSES).all(|bus| monitor.audible(bus)));
        monitor.set(2, true, true);
        assert!(!monitor.audible(2) && monitor.audible(1));
        // Soloing 3 leaves 3 and the master heard; 2 stays muted even soloed.
        monitor.set(3, false, true);
        monitor.set(2, false, true);
        assert!(monitor.audible(0) && monitor.audible(3) && !monitor.audible(1) && !monitor.audible(2));
        monitor.set(2, true, false);
        assert!(monitor.audible(2));
        // Muting the master silences everything there is.
        monitor.set(0, true, true);
        assert!(!monitor.audible(0));
        assert_eq!(Monitor::from_bits(monitor.bits()), monitor);
    }

    #[test]
    fn meters_keep_peaks_and_rms() {
        let (mut first, mut second) = (Meter::default(), Meter::default());
        first.add([0.5, -1.0]);
        first.add([-0.5, 0.0]);
        second.add([0.5, 0.0]);
        second.add([0.5, 0.0]);
        first.merge(&second);
        assert_eq!(first.peak, [0.5, 1.0]);
        assert_eq!(first.rms(), [0.5, 0.5]);
        assert_eq!(Meter::default().rms(), [0.0; 2]);
    }

    /// The answer has DEVTOOLS_AUDIO's numbers, then each bus's.
    #[test]
    fn the_answer_lays_out_the_report() {
        let mut meter = Meter::default();
        meter.add([0.25, -0.5]);
        let monitor = Monitor { muted: 0b10, soloed: 0b100 };
        let report = AudioReport {
            latency_ms: Some(21.5),
            rate: 48_000.0,
            buffer: None,
            voices: 3,
            streams: 1,
            underruns: 2,
            buses: AudioReport::buses(&[meter, meter, meter], &[true, true, false], monitor),
        };
        let answer = report.answer();
        let head = devtools_audio::NAMES.len();
        assert_eq!(answer.values[..head], [21.5, 48_000.0, -1.0, 3.0, 1.0, 2.0, MAX_BUSES as f32]);
        let bus = |index: usize| &answer.values[head + index * 5..head + index * 5 + 5];
        assert_eq!(bus(0), [0.25, 0.5, 0.25, 0.5, 1.0]);
        assert_eq!(bus(1)[4], 3.0);
        assert_eq!(bus(2)[4], 4.0);
        assert_eq!(bus(3), [0.0; 5]);
        assert_eq!(answer.values.len(), head + MAX_BUSES * 5);
    }
}
