//! Core audio DSP primitives used by the mixer: parameter glides, linear fades,
//! Butterworth lowpass filters, binaural head-shadow/ITD filtering, and 3D spatialization.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "audio sample math")]
use std::f64::consts::PI;

use pen_runtime::panner::{DEFAULT_EAR, Pan, Placement, Vec3, azimuth_default_ear, equal_power};

/// A value setTargetAtTime glides toward its target: each sample, the step's
/// share of the way there. It settles where WebAudio's does, within 1.5e-6
/// of the target.
#[derive(Clone, Copy, Debug)]
pub(super) struct Glide {
    pub(super) value: f64,
    target: f64,
}

impl Glide {
    pub(super) fn at(value: f32) -> Glide {
        Glide { value: f64::from(value), target: f64::from(value) }
    }

    pub(super) fn set(&mut self, target: f32) {
        self.target = f64::from(target);
    }

    /// This sample's value, then a step on.
    #[inline]
    pub(super) fn next(&mut self, step: f64) -> f32 {
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

    #[inline]
    pub(super) fn is_settled(&self) -> bool {
        self.value == self.target
    }

    #[inline]
    pub(super) fn current(&self) -> f32 {
        self.value as f32
    }
}

/// A fade: linearRampToValueAtTime from `from` at sample `start` to `to`
/// `length` samples later.
#[derive(Clone, Copy, Debug)]
pub(super) struct Ramp {
    from: f64,
    to: f64,
    start: u64,
    length: f64,
    inv_length: f64,
}

impl Ramp {
    pub(super) fn new(from: f64, to: f64, start: u64, length: f64) -> Ramp {
        Ramp { from, to, start, length, inv_length: if length > 0.0 { 1.0 / length } else { 0.0 } }
    }

    pub(super) fn hold(value: f64) -> Ramp {
        Ramp { from: value, to: value, start: 0, length: 0.0, inv_length: 0.0 }
    }

    #[inline]
    pub(super) fn at(&self, sample: u64) -> f64 {
        let since = sample.saturating_sub(self.start) as f64;
        if sample < self.start {
            self.from
        } else if since >= self.length {
            self.to
        } else {
            self.from + (self.to - self.from) * (since * self.inv_length)
        }
    }

    #[inline]
    pub(super) fn settled_at(&self, start_sample: u64, end_sample: u64) -> Option<f32> {
        if start_sample >= self.start && (start_sample - self.start) as f64 >= self.length {
            Some(self.to as f32)
        } else if end_sample <= self.start {
            Some(self.from as f32)
        } else {
            None
        }
    }

    pub(super) fn done(&self, sample: u64) -> bool {
        sample >= self.start && (sample - self.start) as f64 >= self.length
    }
}

/// Exact linear Butterworth resonance `10^(LOWPASS_Q_DB / 20) == 1/√2`.
const LOWPASS_RESONANCE: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// A BiquadFilterNode lowpass at WebAudio's Q of LOWPASS_Q_DB, its cutoff
/// gliding; its coefficients are worked out again whenever the cutoff
/// moves, every sample while it glides, as Chrome's are. Up to four
/// channels share them (a bus's dry pair and send pair).
pub(super) struct Lowpass {
    pub(super) cutoff: Glide,
    /// The cutoff the coefficients are for.
    computed: f32,
    open: bool,
    pub(super) b: [f64; 3],
    pub(super) a: [f64; 2],
    /// Per channel: x1, x2, y1, y2.
    state: [[f64; 4]; 4],
}

impl Lowpass {
    pub(super) fn new(cutoff: f32, rate: f64) -> Lowpass {
        let mut filter = Lowpass {
            cutoff: Glide::at(cutoff),
            computed: f32::NAN,
            open: true,
            b: [1.0, 0.0, 0.0],
            a: [0.0, 0.0],
            state: [[0.0; 4]; 4],
        };
        filter.compute(cutoff, rate);
        filter
    }

    /// WebAudio's lowpass coefficients (Chrome's `Biquad::SetLowpassParams`).
    pub(super) fn compute(&mut self, cutoff: f32, rate: f64) {
        self.computed = cutoff;
        let nyquist = rate * 0.5;
        let normalized = (f64::from(cutoff).clamp(0.0, nyquist)) / nyquist;
        self.open = normalized >= 1.0;
        (self.b, self.a) = if self.open {
            ([1.0, 0.0, 0.0], [0.0, 0.0])
        } else if normalized > 0.0 {
            let theta = PI * normalized;
            let (sine, cosine) = theta.sin_cos();
            let alpha = sine / (2.0 * LOWPASS_RESONANCE);
            let beta = (1.0 - cosine) / 2.0;
            let inv_a0 = 1.0 / (1.0 + alpha);
            ([beta * inv_a0, 2.0 * beta * inv_a0, beta * inv_a0], [-2.0 * cosine * inv_a0, (1.0 - alpha) * inv_a0])
        } else {
            ([0.0, 0.0, 0.0], [0.0, 0.0])
        };
    }

    /// The cutoff for this sample, and the coefficients with it.
    #[inline]
    pub(super) fn step(&mut self, glide: f64, rate: f64) {
        let cutoff = self.cutoff.next(glide);
        if cutoff.to_bits() != self.computed.to_bits() {
            self.compute(cutoff, rate);
        }
    }

    #[inline]
    pub(super) fn is_open_settled(&self) -> bool {
        self.open && self.cutoff.is_settled()
    }

    #[inline]
    pub(super) fn seed_open_pair(&mut self, channel: usize, prev: f32, last: f32) {
        let x2 = f64::from(prev);
        let x1 = f64::from(last);
        self.state[channel] = [x1, x2, x1, x2];
    }

    #[inline]
    pub(super) fn run(&mut self, channel: usize, input: f32) -> f32 {
        if self.open {
            let x = f64::from(input);
            let [x1, _, y1, _] = self.state[channel];
            self.state[channel] = [x, x1, x, y1];
            return input;
        }
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
pub(super) fn cutoff(lowpass: f32, rate: f64) -> f32 {
    lowpass.clamp(0.0, (rate * 0.5) as f32)
}

/// StereoPannerNode's gains for `pan`: (left, right) for a mono input, or
/// for a stereo one the share of one side moved into the other.
pub(super) fn pan_gains(pan: f32) -> (f32, f32) {
    let pan = pan.clamp(-1.0, 1.0);
    let x = if pan <= 0.0 { pan + 1.0 } else { pan } * std::f32::consts::FRAC_PI_2;
    (x.cos(), x.sin())
}

pub(super) fn mono_gains(pan: f32) -> (f32, f32) {
    let x = (pan.clamp(-1.0, 1.0) + 1.0) * 0.5 * std::f32::consts::FRAC_PI_2;
    (x.cos(), x.sin())
}

/// Three values gliding together: a position or a direction.
pub(super) fn glides(value: Vec3) -> [Glide; 3] {
    value.map(Glide::at)
}

pub(super) fn glide_to(glides: &mut [Glide; 3], target: Vec3) {
    for (glide, value) in glides.iter_mut().zip(target) {
        glide.set(value);
    }
}

pub(super) fn next(glides: &mut [Glide; 3], step: f64) -> Vec3 {
    [glides[0].next(step), glides[1].next(step), glides[2].next(step)]
}

/// Fractional-sample binaural ITD delay line + contralateral head-shadowing ILD lowpass
/// approximating a spherical-head HRTF (Woodworth ITD + acoustic shadow + rear pinna cue).
pub(super) struct Binaural {
    ring: [[f32; 64]; 2],
    write: usize,
    shadow: [f64; 2],
    delay_l_i0: usize,
    delay_l_frac: f32,
    delay_r_i0: usize,
    delay_r_frac: f32,
    alpha_left: f64,
    alpha_right: f64,
    initialized: bool,
}

impl Binaural {
    pub(super) fn new() -> Self {
        Self {
            ring: [[0.0; 64]; 2],
            write: 0,
            shadow: [0.0; 2],
            delay_l_i0: 0,
            delay_l_frac: 0.0,
            delay_r_i0: 0,
            delay_r_frac: 0.0,
            alpha_left: 1.0,
            alpha_right: 1.0,
            initialized: false,
        }
    }

    pub(super) fn update(&mut self, position: Vec3, azimuth_deg: f64, rate: f64) {
        // Woodworth spherical-head ITD: r = 0.0875 m, c = 343 m/s.
        let az_rad = azimuth_deg.to_radians();
        let lateral = az_rad.sin().clamp(-1.0, 1.0);
        let itd_sec = (0.0875 / 343.0) * (lateral + lateral.asin());
        let itd_samples = (itd_sec * rate).clamp(-56.0, 56.0);
        let (delay_l, delay_r) = if itd_samples >= 0.0 { (itd_samples, 0.0) } else { (0.0, -itd_samples) };
        let dl = delay_l.clamp(0.0, 60.0);
        let dr = delay_r.clamp(0.0, 60.0);
        self.delay_l_i0 = dl.floor() as usize;
        self.delay_l_frac = (dl - self.delay_l_i0 as f64) as f32;
        self.delay_r_i0 = dr.floor() as usize;
        self.delay_r_frac = (dr - self.delay_r_i0 as f64) as f32;
        // Rear hemisphere (z > 0 in listener space) adds subtle pinna high-frequency dulling.
        let dist = (f64::from(position[0]).powi(2) + f64::from(position[1]).powi(2) + f64::from(position[2]).powi(2))
            .sqrt()
            .max(1e-6);
        let rear = (f64::from(position[2]) / dist).clamp(0.0, 1.0);
        let ear_alpha = |shadow_amount: f64| -> f64 {
            let amount = (shadow_amount + rear * 0.25).clamp(0.0, 1.0);
            if amount < 1e-4 {
                return 1.0;
            }
            let cutoff = 18_000.0 * (1600.0 / 18_000.0f64).powf(amount);
            (1.0 - (-2.0 * PI * cutoff / rate).exp()).clamp(0.01, 1.0)
        };
        // Positive lateral = source on right -> left ear is contralateral (shadowed + delayed).
        self.alpha_left = ear_alpha(lateral.max(0.0));
        self.alpha_right = ear_alpha((-lateral).max(0.0));
        self.initialized = true;
    }

    #[inline]
    pub(super) fn process(&mut self, left: f32, right: f32) -> (f32, f32) {
        let w = self.write;
        self.ring[0][w] = left;
        self.ring[1][w] = right;
        let l0 = (w + 64 - self.delay_l_i0) & 63;
        let l1 = (l0.wrapping_sub(1)) & 63;
        let dl = f64::from(self.ring[0][l0] * (1.0 - self.delay_l_frac) + self.ring[0][l1] * self.delay_l_frac);
        let r0 = (w + 64 - self.delay_r_i0) & 63;
        let r1 = (r0.wrapping_sub(1)) & 63;
        let dr = f64::from(self.ring[1][r0] * (1.0 - self.delay_r_frac) + self.ring[1][r1] * self.delay_r_frac);
        self.shadow[0] += self.alpha_left * (dl - self.shadow[0]);
        self.shadow[1] += self.alpha_right * (dr - self.shadow[1]);
        if self.shadow[0].abs() < 1e-30 {
            self.shadow[0] = 0.0;
        }
        if self.shadow[1].abs() < 1e-30 {
            self.shadow[1] = 0.0;
        }
        self.write = (w + 1) & 63;
        (self.shadow[0] as f32, self.shadow[1] as f32)
    }
}

/// A voice's PannerNode: its settings, which change at once, its position
/// and orientation in the listener's frame, which glide, and the pan and
/// gain worked out for where it was last. While it moves, they are worked
/// out again every sample, as Chrome does while a panner's values are
/// automated.
pub(super) struct Spatial {
    placement: Placement,
    position: [Glide; 3],
    orientation: [Glide; 3],
    occlusion: Glide,
    doppler: Glide,
    /// What `pan` and `base_gain` are for.
    placed: Option<(Vec3, Vec3)>,
    pan: Pan,
    base_gain: f32,
    binaural: Option<Binaural>,
}

impl Spatial {
    pub(super) fn new(placement: Placement) -> Spatial {
        let doppler = placement.doppler_ratio(placement.position);
        Spatial {
            binaural: placement.hrtf.then(Binaural::new),
            occlusion: Glide::at(placement.occlusion_gain()),
            doppler: Glide::at(doppler),
            placement,
            position: glides(placement.position),
            orientation: glides(placement.orientation),
            placed: None,
            pan: Pan { left: 1.0, right: 1.0, rightward: false },
            base_gain: 1.0,
        }
    }

    pub(super) fn set(&mut self, placement: Placement) {
        glide_to(&mut self.position, placement.position);
        glide_to(&mut self.orientation, placement.orientation);
        self.occlusion.set(placement.occlusion_gain());
        self.doppler.set(placement.doppler_ratio(placement.position));
        if placement.hrtf && self.binaural.is_none() {
            self.binaural = Some(Binaural::new());
        } else if !placement.hrtf {
            self.binaural = None;
        }
        if placement != self.placement {
            self.placement = placement;
            self.placed = None;
        }
    }

    pub(super) fn next_doppler(&mut self, quantum_step: f64) -> f32 {
        self.doppler.next(quantum_step)
    }

    pub(super) fn wet_scale(&self) -> f32 {
        self.placement.wet_scale
    }

    /// This sample's pan, direct gain (distance * cone * gliding occlusion), and pre-occlusion
    /// send gain (distance * cone), then a step on.
    pub(super) fn next(&mut self, step: f64, mono: bool, rate: f64, update_binaural: bool) -> (Pan, f32, f32) {
        let position = next(&mut self.position, step);
        let orientation = next(&mut self.orientation, step);
        let occ = self.occlusion.next(step);
        if self.placed != Some((position, orientation)) {
            self.placed = Some((position, orientation));
            let az = azimuth_default_ear(position);
            self.pan = equal_power(az, mono);
            self.base_gain = self.placement.gain(position, orientation, &DEFAULT_EAR);
            self.doppler.set(self.placement.doppler_ratio(position));
            if let Some(binaural) = &mut self.binaural
                && (update_binaural || !binaural.initialized)
            {
                binaural.update(position, az, rate);
            }
        }
        (self.pan, self.base_gain * occ, self.base_gain)
    }

    pub(super) fn filter_binaural(&mut self, left: f32, right: f32) -> (f32, f32) {
        match &mut self.binaural {
            Some(binaural) => binaural.process(left, right),
            None => (left, right),
        }
    }
}
