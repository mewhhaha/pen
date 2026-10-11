//! gdev's mix, as WebAudio plays it (gdev's `web/mixer.ts` builds the same
//! graph of nodes): the frame's buses, voices and sounds, sample by sample,
//! a render quantum (128 samples) at a time. The live output and video
//! recordings both play this, so they sound alike, and like the browser.
//!
//! Per voice, as its nodes do in the browser: the clip (AudioBufferSourceNode,
//! linearly interpolated, its pitch a k-rate playbackRate), a lowpass
//! (BiquadFilterNode, Butterworth), its fade (a GainNode ramped linearly), its
//! gain (a GainNode), its pan (StereoPannerNode, mono or stereo as the clip
//! is), then into its bus, and its reverb send (a GainNode) into the bus's
//! send. A bus filters and scales both alike, passes the first to the master
//! and the second, with `reverb` of the first, to the shared reverb
//! (ConvolverNode, `reverb.rs`), which returns into the master at the
//! master's `reverb`. Values that change glide as setTargetAtTime does, with
//! AUDIO_GLIDE_MS for its time constant; a frame's changes land at the start
//! of the next quantum, where the browser's `currentTime` is.
use std::f64::consts::PI;
use std::sync::Arc;

use pen_protocol::boundary::{AUDIO_FADE_MS, AUDIO_GLIDE_MS, AUDIO_OPEN, MAX_BUSES};

use super::resample::resample;
use super::reverb::Convolver;

/// WebAudio's render quantum.
pub const QUANTUM: usize = 128;
/// A lowpass cutoff at which the filter is open.
pub const OPEN: f32 = AUDIO_OPEN as f32;
/// The lowpass's Q, in WebAudio's decibels: 1/√2, a Butterworth.
pub const LOWPASS_Q_DB: f64 = -3.010_299_956_639_812;
const GLIDE_SECONDS: f64 = AUDIO_GLIDE_MS as f64 / 1000.0;
const FADE_SECONDS: f64 = AUDIO_FADE_MS as f64 / 1000.0;

/// A decoded clip: its frames at its own rate, and whether it is mono (both
/// channels the same), which WebAudio pans with the mono law.
pub struct Clip {
    pub frames: Arc<[kira::Frame]>,
    pub rate: f64,
    pub mono: bool,
}

impl Clip {
    pub fn new(frames: Arc<[kira::Frame]>, rate: f64) -> Clip {
        let mono = frames.iter().all(|frame| frame.left.to_bits() == frame.right.to_bits());
        Clip { frames, rate, mono }
    }

    /// The clip at `rate`, resampled as the browser's decoding resamples it
    /// (`resample.rs`), or as it is if it is at that rate already. The mixer
    /// plays a clip at any rate, but one at its own as the browser does.
    pub fn at_rate(self: Arc<Clip>, rate: f64) -> Arc<Clip> {
        if (self.rate - rate).abs() < 0.5 || self.frames.is_empty() {
            return self;
        }
        let channel = |pick: fn(&kira::Frame) -> f32| {
            let samples: Vec<f32> = self.frames.iter().map(pick).collect();
            resample(&samples, self.rate, rate)
        };
        let left = channel(|frame| frame.left);
        let frames: Arc<[kira::Frame]> = if self.mono {
            left.into_iter().map(|value| kira::Frame { left: value, right: value }).collect()
        } else {
            left.into_iter().zip(channel(|frame| frame.right)).map(|(left, right)| kira::Frame { left, right }).collect()
        };
        Arc::new(Clip { frames, rate, mono: self.mono })
    }
}

/// A bus as a frame declares it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BusSpec {
    pub gain: f32,
    pub lowpass: f32,
    pub reverb: f32,
}

/// What a frame without buses mixes through: an open master at gain 1,
/// returning the reverb whole, and buses it lacks open and dry at gain 1.
const MASTER: BusSpec = BusSpec { gain: 1.0, lowpass: OPEN, reverb: 1.0 };
const PLAIN: BusSpec = BusSpec { gain: 1.0, lowpass: OPEN, reverb: 0.0 };

/// A voice as a frame declares it.
#[derive(Clone)]
pub struct VoiceSpec {
    pub id: u32,
    pub clip: Arc<Clip>,
    pub bus: usize,
    pub gain: f32,
    pub pan: f32,
    pub pitch: f32,
    pub lowpass: f32,
    pub reverb: f32,
    pub looping: bool,
    pub start: f32,
    pub fade_in: f32,
    pub fade_out: f32,
}

/// A sound to play once.
#[derive(Clone)]
pub struct SoundSpec {
    pub clip: Arc<Clip>,
    pub bus: usize,
    pub gain: f32,
    pub pan: f32,
    pub pitch: f32,
    pub lowpass: f32,
    pub reverb: f32,
}

/// One frame's sound: its buses, voices and sounds.
#[derive(Default)]
pub struct Mix {
    pub buses: Vec<BusSpec>,
    pub voices: Vec<VoiceSpec>,
    pub sounds: Vec<SoundSpec>,
}

/// A value setTargetAtTime glides toward its target: each sample, the step's
/// share of the way there. It settles where WebAudio's does, within 1.5e-6
/// of the target.
#[derive(Clone, Copy, Debug)]
struct Glide {
    value: f64,
    target: f64,
}

impl Glide {
    fn at(value: f32) -> Glide {
        Glide { value: f64::from(value), target: f64::from(value) }
    }

    fn set(&mut self, target: f32) {
        self.target = f64::from(target);
    }

    /// This sample's value, then a step on.
    fn next(&mut self, step: f64) -> f32 {
        let value = self.value;
        if value != self.target {
            let next = value + (self.target - value) * step;
            let settled = if self.target == 0.0 {
                next.abs() < 1e-20
            } else {
                (self.target - next).abs() < 1.5e-6 * self.target.abs()
            };
            self.value = if settled { self.target } else { next };
        }
        value as f32
    }
}

/// A fade: linearRampToValueAtTime from `from` at sample `start` to `to`
/// `length` samples later.
#[derive(Clone, Copy, Debug)]
struct Ramp {
    from: f64,
    to: f64,
    start: u64,
    length: f64,
}

impl Ramp {
    fn hold(value: f64) -> Ramp {
        Ramp { from: value, to: value, start: 0, length: 0.0 }
    }

    fn at(&self, sample: u64) -> f64 {
        let since = sample.saturating_sub(self.start) as f64;
        if sample < self.start {
            self.from
        } else if since >= self.length {
            self.to
        } else {
            self.from + (self.to - self.from) * since / self.length
        }
    }

    fn done(&self, sample: u64) -> bool {
        sample >= self.start && (sample - self.start) as f64 >= self.length
    }
}

/// A BiquadFilterNode lowpass at WebAudio's Q of LOWPASS_Q_DB, its cutoff
/// gliding; its coefficients are worked out again whenever the cutoff
/// moves, every sample while it glides, as Chrome's are. Up to four
/// channels share them (a bus's dry pair and send pair).
struct Lowpass {
    cutoff: Glide,
    /// The cutoff the coefficients are for.
    computed: f32,
    b: [f64; 3],
    a: [f64; 2],
    /// Per channel: x1, x2, y1, y2.
    state: [[f64; 4]; 4],
}

impl Lowpass {
    fn new(cutoff: f32, rate: f64) -> Lowpass {
        let mut filter = Lowpass { cutoff: Glide::at(cutoff), computed: f32::NAN, b: [1.0, 0.0, 0.0], a: [0.0, 0.0], state: [[0.0; 4]; 4] };
        filter.compute(cutoff, rate);
        filter
    }

    /// WebAudio's lowpass coefficients (Chrome's `Biquad::SetLowpassParams`).
    fn compute(&mut self, cutoff: f32, rate: f64) {
        self.computed = cutoff;
        let nyquist = rate * 0.5;
        let normalized = (f64::from(cutoff).clamp(0.0, nyquist)) / nyquist;
        (self.b, self.a) = if normalized >= 1.0 {
            ([1.0, 0.0, 0.0], [0.0, 0.0])
        } else if normalized > 0.0 {
            let resonance = 10f64.powf(LOWPASS_Q_DB / 20.0);
            let theta = PI * normalized;
            let alpha = theta.sin() / (2.0 * resonance);
            let cosine = theta.cos();
            let beta = (1.0 - cosine) / 2.0;
            let a0 = 1.0 + alpha;
            ([beta / a0, 2.0 * beta / a0, beta / a0], [-2.0 * cosine / a0, (1.0 - alpha) / a0])
        } else {
            ([0.0, 0.0, 0.0], [0.0, 0.0])
        };
    }

    /// The cutoff for this sample, and the coefficients with it.
    fn step(&mut self, glide: f64, rate: f64) {
        let cutoff = self.cutoff.next(glide);
        if cutoff.to_bits() != self.computed.to_bits() {
            self.compute(cutoff, rate);
        }
    }

    fn run(&mut self, channel: usize, input: f32) -> f32 {
        let [x1, x2, y1, y2] = self.state[channel];
        let x = f64::from(input);
        let y = self.b[0] * x + self.b[1] * x1 + self.b[2] * x2 - self.a[0] * y1 - self.a[1] * y2;
        // Flushed, as denormals are.
        let y = if y.abs() < 1e-30 { 0.0 } else { y };
        self.state[channel] = [x, x1, y, y1];
        y as f32
    }
}

/// A lowpass cutoff as hosts set it: within 0 and half the sample rate, the
/// filter's nominal range, so that both glide the same way.
fn cutoff(lowpass: f32, rate: f64) -> f32 {
    lowpass.clamp(0.0, (rate * 0.5) as f32)
}

/// StereoPannerNode's gains for `pan`: (left, right) for a mono input, or
/// for a stereo one the share of one side moved into the other.
fn pan_gains(pan: f32) -> (f32, f32) {
    let pan = pan.clamp(-1.0, 1.0);
    let x = if pan <= 0.0 { pan + 1.0 } else { pan } * std::f32::consts::FRAC_PI_2;
    (x.cos(), x.sin())
}

fn mono_gains(pan: f32) -> (f32, f32) {
    let x = (pan.clamp(-1.0, 1.0) + 1.0) * 0.5 * std::f32::consts::FRAC_PI_2;
    (x.cos(), x.sin())
}

struct Voice {
    /// None for a sound played once, or a voice its id has moved on from.
    id: Option<u32>,
    clip: Arc<Clip>,
    bus: usize,
    /// Where it is in the clip, in the clip's frames.
    position: f64,
    looping: bool,
    pitch: Glide,
    gain: Glide,
    pan: Glide,
    send: Glide,
    lowpass: Lowpass,
    fade: Ramp,
    fade_out: f32,
    leaving: bool,
    ended: bool,
    /// Declared by the latest frame.
    seen: bool,
    /// The pan its gains are for, and they.
    panned: f32,
    gains: (f32, f32),
}

impl Voice {
    /// Starts at sample `now`, at the values of `sound`, `start` seconds
    /// into its clip.
    fn new(id: Option<u32>, sound: &SoundSpec, looping: bool, start: f32, fade_in: f32, now: u64, rate: f64) -> Voice {
        let SoundSpec { ref clip, bus, gain, pan, pitch, lowpass, reverb } = *sound;
        let length = clip.frames.len() as f64;
        let offset = f64::from(start) * clip.rate;
        let position = if looping && length > 0.0 { offset.rem_euclid(length) } else { offset.max(0.0) };
        let fade = if fade_in > 0.0 {
            Ramp { from: 0.0, to: 1.0, start: now, length: f64::from(fade_in) * rate }
        } else {
            Ramp::hold(1.0)
        };
        Voice {
            id,
            ended: position >= length,
            clip: clip.clone(),
            bus,
            position,
            looping,
            pitch: Glide::at(pitch.max(0.0)),
            gain: Glide::at(gain),
            pan: Glide::at(pan.clamp(-1.0, 1.0)),
            send: Glide::at(reverb),
            lowpass: Lowpass::new(cutoff(lowpass, rate), rate),
            fade,
            fade_out: 0.0,
            leaving: false,
            seen: true,
            panned: f32::NAN,
            gains: (1.0, 1.0),
        }
    }

    /// Fades out from where it is, then goes.
    fn leave(&mut self, now: u64, rate: f64) {
        let current = self.fade.at(now);
        let seconds = f64::from(self.fade_out).max(FADE_SECONDS);
        self.fade = Ramp { from: current, to: 0.0, start: now, length: seconds * rate };
        self.leaving = true;
    }

    fn gone(&self, now: u64) -> bool {
        (self.leaving && self.fade.done(now)) || (self.ended && (self.leaving || self.id.is_none()))
    }

    /// One quantum into its bus's dry and send inputs.
    fn render(&mut self, now: u64, rate: f64, glide: f64, quantum_glide: f64, dry: &mut [[f32; 2]; QUANTUM], send: &mut [[f32; 2]; QUANTUM]) {
        if self.ended {
            return;
        }
        let frames = &self.clip.frames;
        let length = frames.len();
        let end = length as f64;
        // playbackRate is k-rate: its value at the quantum's start.
        let step = f64::from(self.pitch.next(quantum_glide)) * self.clip.rate / rate;
        let mono = self.clip.mono;
        for i in 0..QUANTUM {
            if self.position >= end {
                if self.looping && length > 0 {
                    self.position %= end;
                } else {
                    self.ended = true;
                    return;
                }
            }
            let index = self.position as usize;
            let fraction = self.position - index as f64;
            let next = if index + 1 < length {
                index + 1
            } else if self.looping {
                0
            } else {
                index
            };
            let (a, b) = (frames[index], frames[next]);
            let left = (f64::from(a.left) * (1.0 - fraction) + f64::from(b.left) * fraction) as f32;
            self.position += step;
            self.lowpass.step(glide, rate);
            let fade = self.fade.at(now + i as u64) as f32;
            let gain = self.gain.next(glide);
            let pan = self.pan.next(glide);
            let reverb = self.send.next(glide);
            if pan.to_bits() != self.panned.to_bits() {
                self.panned = pan;
                self.gains = if mono { mono_gains(pan) } else { pan_gains(pan) };
            }
            let (out_left, out_right) = if mono {
                let x = self.lowpass.run(0, left) * fade * gain;
                (x * self.gains.0, x * self.gains.1)
            } else {
                let right = (f64::from(a.right) * (1.0 - fraction) + f64::from(b.right) * fraction) as f32;
                let l = self.lowpass.run(0, left) * fade * gain;
                let r = self.lowpass.run(1, right) * fade * gain;
                if pan <= 0.0 {
                    (l + r * self.gains.0, r * self.gains.1)
                } else {
                    (l * self.gains.0, r + l * self.gains.1)
                }
            };
            dry[i][0] += out_left;
            dry[i][1] += out_right;
            send[i][0] += out_left * reverb;
            send[i][1] += out_right * reverb;
        }
    }
}

struct Bus {
    active: bool,
    gain: Glide,
    reverb: Glide,
    /// Channels 0 and 1 filter the dry input, 2 and 3 the send.
    lowpass: Lowpass,
    dry: [[f32; 2]; QUANTUM],
    send: [[f32; 2]; QUANTUM],
}

impl Bus {
    fn new(rate: f64) -> Bus {
        Bus {
            active: false,
            gain: Glide::at(1.0),
            reverb: Glide::at(0.0),
            lowpass: Lowpass::new(OPEN, rate),
            dry: [[0.0; 2]; QUANTUM],
            send: [[0.0; 2]; QUANTUM],
        }
    }

    /// Starts it at `spec` at once, as a new bus's nodes do.
    fn start(&mut self, spec: BusSpec, rate: f64) {
        self.active = true;
        self.gain = Glide::at(spec.gain);
        self.reverb = Glide::at(spec.reverb);
        self.lowpass = Lowpass::new(cutoff(spec.lowpass, rate), rate);
    }

    fn glide_to(&mut self, spec: BusSpec, rate: f64) {
        self.gain.set(spec.gain);
        self.reverb.set(spec.reverb);
        self.lowpass.cutoff.set(cutoff(spec.lowpass, rate));
    }
}

/// The mix, as WebAudio would play it, at `rate` samples a second.
pub struct Mixer {
    rate: f64,
    glide: f64,
    quantum_glide: f64,
    /// The sample the next quantum starts at.
    now: u64,
    voices: Vec<Voice>,
    buses: Vec<Bus>,
    reverb: Convolver,
    master_in: [[f32; 2]; QUANTUM],
    reverb_in: [[f32; 2]; QUANTUM],
    reverb_out: [[f32; 2]; QUANTUM],
    out: [[f32; 2]; QUANTUM],
    /// How much of `out` has been handed out.
    taken: usize,
}

impl Mixer {
    pub fn new(rate: f64) -> Mixer {
        let mut buses: Vec<Bus> = (0..MAX_BUSES).map(|_| Bus::new(rate)).collect();
        buses[0].start(MASTER, rate);
        Mixer {
            rate,
            glide: 1.0 - (-1.0 / (GLIDE_SECONDS * rate)).exp(),
            quantum_glide: 1.0 - (-(QUANTUM as f64) / (GLIDE_SECONDS * rate)).exp(),
            now: 0,
            voices: Vec::with_capacity(1024),
            buses,
            reverb: Convolver::new(rate),
            master_in: [[0.0; 2]; QUANTUM],
            reverb_in: [[0.0; 2]; QUANTUM],
            reverb_out: [[0.0; 2]; QUANTUM],
            out: [[0.0; 2]; QUANTUM],
            taken: QUANTUM,
        }
    }

    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Voices and sounds playing, those fading out included.
    #[cfg(test)]
    fn playing(&self) -> usize {
        self.voices.len()
    }

    /// Bus `index`, and those before it, started plain if they are new, as
    /// the browser makes them.
    fn bus(&mut self, index: usize) -> usize {
        let index = index.min(MAX_BUSES - 1);
        for bus in self.buses[..=index].iter_mut().filter(|bus| !bus.active) {
            bus.start(PLAIN, self.rate);
        }
        index
    }

    /// A frame's sound, from the next quantum on: buses glide to theirs (those
    /// it lacks to plain), voices start, return, change or leave by id, and
    /// its sounds start.
    pub fn apply(&mut self, mix: &Mix) {
        let (now, rate) = (self.now, self.rate);
        for (index, bus) in self.buses.iter_mut().enumerate() {
            let spec = mix.buses.get(index).copied();
            match (spec, bus.active) {
                (Some(spec), true) => bus.glide_to(spec, rate),
                (Some(spec), false) => bus.start(spec, rate),
                (None, true) => bus.glide_to(if index == 0 { MASTER } else { PLAIN }, rate),
                (None, false) => {}
            }
        }
        for voice in &mut self.voices {
            voice.seen = false;
        }
        for spec in &mix.voices {
            let bus = self.bus(spec.bus);
            let found = self.voices.iter().position(|voice| voice.id == Some(spec.id));
            if let Some(index) = found {
                let voice = &mut self.voices[index];
                if voice.seen {
                    // A second voice with an id this frame already has.
                    continue;
                }
                if Arc::ptr_eq(&voice.clip, &spec.clip) && voice.bus == bus {
                    voice.seen = true;
                    voice.gain.set(spec.gain);
                    voice.pan.set(spec.pan.clamp(-1.0, 1.0));
                    voice.pitch.set(spec.pitch.max(0.0));
                    voice.send.set(spec.reverb);
                    voice.lowpass.cutoff.set(cutoff(spec.lowpass, rate));
                    voice.looping = spec.looping;
                    voice.fade_out = spec.fade_out;
                    if voice.leaving {
                        let current = voice.fade.at(now);
                        let seconds = f64::from(spec.fade_in).max(FADE_SECONDS);
                        voice.fade = Ramp { from: current, to: 1.0, start: now, length: seconds * rate };
                        voice.leaving = false;
                    }
                    continue;
                }
                // Another clip or bus: the old fades out as the new starts.
                voice.id = None;
                voice.fade_out = spec.fade_out;
                voice.leave(now, rate);
            }
            let sound = SoundSpec {
                clip: spec.clip.clone(),
                bus,
                gain: spec.gain,
                pan: spec.pan,
                pitch: spec.pitch,
                lowpass: spec.lowpass,
                reverb: spec.reverb,
            };
            let mut voice = Voice::new(Some(spec.id), &sound, spec.looping, spec.start, spec.fade_in, now, rate);
            voice.fade_out = spec.fade_out;
            self.voices.push(voice);
        }
        for voice in &mut self.voices {
            if voice.id.is_some() && !voice.seen && !voice.leaving {
                voice.leave(now, rate);
            }
        }
        for sound in &mix.sounds {
            let bus = self.bus(sound.bus);
            self.voices.push(Voice::new(None, &SoundSpec { bus, ..sound.clone() }, false, 0.0, 0.0, now, rate));
        }
    }

    /// The next quantum, into `out`.
    fn quantum(&mut self) {
        let (now, rate, glide) = (self.now, self.rate, self.glide);
        for bus in self.buses.iter_mut().filter(|bus| bus.active) {
            bus.dry = [[0.0; 2]; QUANTUM];
            bus.send = [[0.0; 2]; QUANTUM];
        }
        for voice in &mut self.voices {
            let bus = &mut self.buses[voice.bus];
            voice.render(now, rate, glide, self.quantum_glide, &mut bus.dry, &mut bus.send);
        }
        let end = now + QUANTUM as u64;
        self.voices.retain(|voice| !voice.gone(end));
        // The master's own voices go straight to it and to the reverb.
        let (master, buses) = self.buses.split_first_mut().expect("a master bus");
        self.master_in = master.dry;
        self.reverb_in = master.send;
        for bus in buses.iter_mut().filter(|bus| bus.active) {
            for i in 0..QUANTUM {
                bus.lowpass.step(glide, rate);
                let gain = bus.gain.next(glide);
                let reverb = bus.reverb.next(glide);
                let [dry_left, dry_right] = bus.dry[i];
                let [send_left, send_right] = bus.send[i];
                let left = bus.lowpass.run(0, dry_left) * gain;
                let right = bus.lowpass.run(1, dry_right) * gain;
                let wet_left = bus.lowpass.run(2, send_left) * gain;
                let wet_right = bus.lowpass.run(3, send_right) * gain;
                self.master_in[i][0] += left;
                self.master_in[i][1] += right;
                self.reverb_in[i][0] += wet_left + left * reverb;
                self.reverb_in[i][1] += wet_right + right * reverb;
            }
        }
        self.reverb_out = [[0.0; 2]; QUANTUM];
        self.reverb.process(&self.reverb_in, &mut self.reverb_out);
        for i in 0..QUANTUM {
            let back = master.reverb.next(glide);
            master.lowpass.step(glide, rate);
            let gain = master.gain.next(glide);
            let left = self.master_in[i][0] + self.reverb_out[i][0] * back;
            let right = self.master_in[i][1] + self.reverb_out[i][1] * back;
            self.out[i] = [master.lowpass.run(0, left) * gain, master.lowpass.run(1, right) * gain];
        }
        self.now = end;
    }

    /// The next `out.len()` samples of the mix.
    pub fn render(&mut self, out: &mut [[f32; 2]]) {
        let mut written = 0;
        while written < out.len() {
            if self.taken == QUANTUM {
                self.quantum();
                self.taken = 0;
            }
            let count = (QUANTUM - self.taken).min(out.len() - written);
            out[written..written + count].copy_from_slice(&self.out[self.taken..self.taken + count]);
            self.taken += count;
            written += count;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = 48_000.0;

    /// A mono sine at `frequency`, `seconds` long, at `RATE`.
    fn sine(frequency: f64, seconds: f64) -> Arc<Clip> {
        let frames: Arc<[kira::Frame]> = (0..(seconds * RATE) as usize)
            .map(|i| {
                let value = (std::f64::consts::TAU * frequency * i as f64 / RATE).sin() as f32;
                kira::Frame { left: value, right: value }
            })
            .collect();
        Arc::new(Clip::new(frames, RATE))
    }

    fn voice(id: u32, clip: &Arc<Clip>) -> VoiceSpec {
        VoiceSpec {
            id,
            clip: clip.clone(),
            bus: 0,
            gain: 1.0,
            pan: 0.0,
            pitch: 1.0,
            lowpass: OPEN,
            reverb: 0.0,
            looping: true,
            start: 0.0,
            fade_in: 0.0,
            fade_out: 0.0,
        }
    }

    /// Frames of 800 samples (60 per second): each frame's mix, then its sound.
    fn play(mixer: &mut Mixer, frames: impl IntoIterator<Item = Mix>) -> Vec<[f32; 2]> {
        let mut out = Vec::new();
        for mix in frames {
            mixer.apply(&mix);
            let mut chunk = vec![[0.0f32; 2]; 800];
            mixer.render(&mut chunk);
            out.extend(chunk);
        }
        out
    }

    fn rms(samples: &[[f32; 2]]) -> f64 {
        (samples.iter().map(|[l, r]| f64::from(*l).powi(2) + f64::from(*r).powi(2)).sum::<f64>() / (2 * samples.len()) as f64).sqrt()
    }

    fn largest_step(samples: &[[f32; 2]]) -> f32 {
        samples.windows(2).map(|pair| (pair[1][0] - pair[0][0]).abs().max((pair[1][1] - pair[0][1]).abs())).fold(0.0, f32::max)
    }

    #[test]
    fn glides_settle_where_webaudio_does() {
        let mut glide = Glide::at(0.0);
        glide.set(1.0);
        let step = 1.0 - (-1.0 / (GLIDE_SECONDS * RATE)).exp();
        let mut value = 0.0;
        // One time constant on: 1 - 1/e of the way.
        for _ in 0..=(GLIDE_SECONDS * RATE) as usize {
            value = glide.next(step);
        }
        assert!((f64::from(value) - (1.0 - (-1.0f64).exp())).abs() < 1e-3, "{value}");
        for _ in 0..20_000 {
            glide.next(step);
        }
        assert_eq!(glide.value, 1.0);
    }

    #[test]
    fn mono_and_stereo_panning_follow_webaudio() {
        // Mono in the middle is -3 dB a side; stereo in the middle passes as it is.
        let (l, r) = mono_gains(0.0);
        assert!((l - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6 && (r - l).abs() < 1e-6);
        assert_eq!(pan_gains(0.0), (std::f32::consts::FRAC_PI_2.cos(), 1.0));
        let (l, r) = mono_gains(1.0);
        assert!(l.abs() < 1e-6 && (r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn lowpass_passes_low_and_stops_high() {
        let mut filter = Lowpass::new(1000.0, RATE);
        let tone = |filter: &mut Lowpass, frequency: f64| {
            let mut peak = 0.0f32;
            for i in 0..48_000 {
                let y = filter.run(0, (std::f64::consts::TAU * frequency * i as f64 / RATE).sin() as f32);
                if i > 24_000 {
                    peak = peak.max(y.abs());
                }
            }
            peak
        };
        assert!((tone(&mut filter, 100.0) - 1.0).abs() < 0.01);
        // A Butterworth is 3 dB down at its cutoff, 12 dB an octave past it.
        assert!((tone(&mut filter, 1000.0) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.01);
        assert!(tone(&mut filter, 8000.0) < 0.02);
        let open = Lowpass::new(OPEN, RATE);
        assert_eq!((open.b, open.a), ([1.0, 0.0, 0.0], [0.0, 0.0]));
    }

    /// New ids start, missing ones fade out over their fade_out and go, and
    /// one that comes back while fading returns where it is.
    #[test]
    fn voices_reconcile_by_id() {
        let clip = sine(220.0, 1.0);
        let mut mixer = Mixer::new(RATE);
        let mut faded = voice(7, &clip);
        faded.fade_out = 0.5;
        let with = || Mix { voices: vec![faded.clone()], ..Mix::default() };
        let out = play(&mut mixer, [with(), with()]);
        assert_eq!(mixer.playing(), 1);
        // Mono at the middle: 1/√2 a side, a sine's RMS 1/√2 of that.
        assert!((rms(&out[400..]) - 0.5).abs() < 0.01, "{}", rms(&out[400..]));
        // Left out: it fades over half a second, linearly.
        let out = play(&mut mixer, (0..15).map(|_| Mix::default()));
        assert_eq!(mixer.playing(), 1);
        let early = rms(&out[..2400]);
        let late = rms(&out[9600..12000]);
        assert!((early - 0.5 * (1.0 - 1200.0 / 24000.0)).abs() < 0.01, "{early}");
        assert!((late - 0.5 * (1.0 - 10800.0 / 24000.0)).abs() < 0.01, "{late}");
        // Back again: it returns from where its fade had got to, not from
        // its start, so the sine carries on without a jump.
        let before = mixer.voices[0].position;
        let out = play(&mut mixer, [with()]);
        assert!(mixer.voices[0].position > before);
        assert!(largest_step(&out) < 0.05, "{}", largest_step(&out));
        // Left out for good: gone once the fade is over.
        play(&mut mixer, (0..40).map(|_| Mix::default()));
        assert_eq!(mixer.playing(), 0);
    }

    #[test]
    fn changes_glide_without_clicks() {
        let clip = sine(330.0, 1.0);
        let mut mixer = Mixer::new(RATE);
        // Gain, pan and lowpass jumping every frame.
        let frames = (0..120).map(|frame| {
            let mut spec = voice(1, &clip);
            let up = frame % 2 == 0;
            spec.gain = if up { 0.9 } else { 0.1 };
            spec.pan = if up { -0.8 } else { 0.8 };
            spec.lowpass = if up { 400.0 } else { OPEN };
            Mix { voices: vec![spec], ..Mix::default() }
        });
        let out = play(&mut mixer, frames);
        // A 330 Hz sine at full scale moves at most 2π·330/48000 ≈ 0.043 a sample.
        assert!(largest_step(&out) < 0.05, "{}", largest_step(&out));
    }

    /// A new clip for an id crossfades: the old fades out as the new comes in.
    #[test]
    fn clip_changes_crossfade() {
        let (low, high) = (sine(200.0, 2.0), sine(3000.0, 2.0));
        let mut mixer = Mixer::new(RATE);
        let track = |clip: &Arc<Clip>| {
            let mut spec = voice(1, clip);
            spec.fade_in = 0.5;
            spec.fade_out = 0.5;
            Mix { voices: vec![spec], ..Mix::default() }
        };
        play(&mut mixer, (0..60).map(|_| track(&low)));
        let out = play(&mut mixer, (0..60).map(|_| track(&high)));
        assert_eq!(mixer.playing(), 1);
        // Halfway through, both are there at about half.
        let middle = &out[12_000 - 960..12_000 + 960];
        let level = rms(middle);
        assert!((0.25..0.45).contains(&level), "{level}");
        assert!(largest_step(&out) < 0.45);
    }

    #[test]
    fn a_voice_that_does_not_loop_plays_once() {
        let clip = sine(440.0, 0.1);
        let mut mixer = Mixer::new(RATE);
        let mut once = voice(3, &clip);
        once.looping = false;
        let with = || Mix { voices: vec![once.clone()], ..Mix::default() };
        let out = play(&mut mixer, (0..30).map(|_| with()));
        assert!(rms(&out[..4800]) > 0.4);
        assert_eq!(rms(&out[6000..]), 0.0);
        assert_eq!(mixer.playing(), 1);
        // Left out, then declared again: it plays again.
        play(&mut mixer, [Mix::default()]);
        assert_eq!(mixer.playing(), 0);
        let out = play(&mut mixer, [with()]);
        assert!(rms(&out) > 0.4);
    }

    #[test]
    fn duplicate_ids_play_the_first() {
        let (a, b) = (sine(200.0, 1.0), sine(900.0, 1.0));
        let mut mixer = Mixer::new(RATE);
        play(&mut mixer, [Mix { voices: vec![voice(5, &a), voice(5, &b)], ..Mix::default() }]);
        assert_eq!(mixer.playing(), 1);
        assert!(Arc::ptr_eq(&mixer.voices[0].clip, &a));
    }

    /// Buses scale and filter their voices, and send to the reverb, which
    /// comes back through the master.
    #[test]
    fn buses_and_the_reverb() {
        let clip = sine(500.0, 1.0);
        let on_bus = |bus: usize, reverb: f32, buses: Vec<BusSpec>| {
            let mut spec = voice(1, &clip);
            spec.bus = bus;
            spec.reverb = reverb;
            Mix { voices: vec![spec], buses, ..Mix::default() }
        };
        let master = BusSpec { gain: 1.0, lowpass: OPEN, reverb: 1.0 };
        let mut dry = Mixer::new(RATE);
        let plain = rms(&play(&mut dry, (0..30).map(|_| on_bus(1, 0.0, vec![master, PLAIN])))[4800..]);
        let mut quiet = Mixer::new(RATE);
        let half = rms(&play(&mut quiet, (0..30).map(|_| on_bus(1, 0.0, vec![master, BusSpec { gain: 0.5, ..PLAIN }])))[4800..]);
        assert!((half / plain - 0.5).abs() < 0.01, "{half} {plain}");
        let mut dull = Mixer::new(RATE);
        let filtered = rms(&play(&mut dull, (0..30).map(|_| on_bus(1, 0.0, vec![master, BusSpec { lowpass: 250.0, ..PLAIN }])))[4800..]);
        assert!(filtered < plain * 0.35, "{filtered} {plain}");
        // A send adds the reverb; a master returning none takes it away.
        let mut wet = Mixer::new(RATE);
        let reverberant = rms(&play(&mut wet, (0..30).map(|_| on_bus(1, 1.0, vec![master, PLAIN])))[4800..]);
        assert!(reverberant > plain * 1.1, "{reverberant} {plain}");
        let mut returned = Mixer::new(RATE);
        let none = rms(&play(&mut returned, (0..30).map(|_| on_bus(1, 1.0, vec![BusSpec { reverb: 0.0, ..master }, PLAIN])))[4800..]);
        assert!((none - plain).abs() < 1e-4, "{none} {plain}");
    }

    /// The same frames give the same samples, bit for bit.
    #[test]
    fn mixing_is_deterministic() {
        let (a, b) = (sine(200.0, 0.7), sine(1300.0, 0.3));
        let frames = || {
            (0..90).map(|frame| {
                let mut spec = voice(1 + frame / 30, &a);
                spec.reverb = 0.5;
                spec.pitch = 1.0 + frame as f32 * 0.01;
                let sounds = if frame % 7 == 0 {
                    vec![SoundSpec { clip: b.clone(), bus: 2, gain: 0.5, pan: 0.3, pitch: 1.2, lowpass: 3000.0, reverb: 0.3 }]
                } else {
                    Vec::new()
                };
                Mix { voices: vec![spec], sounds, buses: vec![MASTER, PLAIN, BusSpec { gain: 0.7, lowpass: 5000.0, reverb: 0.2 }] }
            })
        };
        let first = play(&mut Mixer::new(RATE), frames());
        let second = play(&mut Mixer::new(RATE), frames());
        assert!(first.iter().zip(&second).all(|(a, b)| a[0].to_bits() == b[0].to_bits() && a[1].to_bits() == b[1].to_bits()));
    }
    /// The calibration scenario `web/mixer.ts` also plays in Chromium (see
    /// the report of hill #13): a ramp clip, whose samples are their own
    /// positions, panned hard left with its pitch changing every frame, and a
    /// clip of ones panned hard right with its gain changing every frame, so
    /// the output's channels are the read position and the gain as they go.
    /// PEN_CALIBRATE_OUT=FILE cargo test --release -p pen calibrate -- --ignored
    #[test]
    #[ignore]
    fn calibrate() {
        let out = std::env::var_os("PEN_CALIBRATE_OUT").expect("PEN_CALIBRATE_OUT is not set");
        let clip = |value: fn(usize) -> f32| {
            let frames: Arc<[kira::Frame]> = (0..48_000).map(|i| kira::Frame { left: value(i), right: value(i) }).collect();
            Arc::new(Clip::new(frames, RATE))
        };
        let (ramp, ones) = (clip(|i| i as f32 / 48_000.0), clip(|_| 1.0));
        let frames = (0..120).map(|frame| {
            let f = frame as f32;
            let mut position = voice(1, &ramp);
            position.pan = -1.0;
            position.pitch = 1.0 + 0.5 * (f / 10.0).sin();
            let mut gain = voice(2, &ones);
            gain.pan = 1.0;
            gain.gain = 0.5 + 0.4 * (f / 7.0).sin();
            gain.lowpass = 2000.0 + 1500.0 * (f / 5.0).sin();
            Mix { voices: vec![position, gain], ..Mix::default() }
        });
        let samples = play(&mut Mixer::new(RATE), frames);
        let bytes: Vec<u8> = samples.iter().flatten().flat_map(|sample| sample.to_le_bytes()).collect();
        std::fs::write(out, bytes).unwrap();
    }
}
