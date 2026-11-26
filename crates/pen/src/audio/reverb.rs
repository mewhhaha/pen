//! The shared reverb: the impulse response of `pen_runtime::reverb`, which
//! the browser host's ConvolverNode plays too, and a convolution with it
//! that, like the ConvolverNode, adds no latency.
//!
//! Why a convolution: WebAudio has no algorithmic reverb, but its
//! ConvolverNode convolves exactly, so the browser and this mixer agree to
//! float rounding with the same impulse response, which both generate from
//! the sample rate alone (deterministic noise, no file). The convolution is
//! uniformly partitioned in two stages: the response's first 2048 samples in
//! blocks of one render quantum (128), which gives each quantum its output at
//! once, and the rest in blocks of 2048, each computed as its input block
//! completes and played over the next, which is exactly when it is due.

use std::f64::consts::TAU;

use super::mixer::QUANTUM;

#[cfg(test)]
pub use pen_runtime::reverb::impulse_response;
pub use pen_runtime::reverb::{AcousticProfile, impulse_response_for, impulse_response_from_acoustics};

/// Precomputed stage-contiguous twiddles for one radix-4 butterfly stage (`m > 1`).
struct Radix4Stage {
    m: usize,
    w1_re: Vec<f32>,
    w1_im: Vec<f32>,
    w2_re: Vec<f32>,
    w2_im: Vec<f32>,
    w3_re: Vec<f32>,
    w3_im: Vec<f32>,
}

/// An iterative split-SoA radix-4 FFT (with an initial radix-2 pass for odd
/// powers of two) operating directly on separate real and imaginary slices so
/// butterflies and spectral MACs auto-vectorize without shuffle overhead.
struct Fft {
    size: usize,
    odd_bits: bool,
    stages: Vec<Radix4Stage>,
    reversed: Vec<usize>,
}

impl Fft {
    fn new(size: usize) -> Fft {
        assert!(size.is_power_of_two() && size >= 2);
        let bits = size.trailing_zeros();
        let odd_bits = (bits & 1) == 1;
        let mut stages = Vec::new();
        let mut m = if odd_bits { 2 } else { 1 };
        #[allow(clippy::cast_possible_truncation, reason = "f32 twiddles")]
        while m < size {
            if m > 1 {
                let mut w1_re = Vec::with_capacity(m);
                let mut w1_im = Vec::with_capacity(m);
                let mut w2_re = Vec::with_capacity(m);
                let mut w2_im = Vec::with_capacity(m);
                let mut w3_re = Vec::with_capacity(m);
                let mut w3_im = Vec::with_capacity(m);
                let base = -TAU / (4 * m) as f64;
                for j in 0..m {
                    let a1 = base * j as f64;
                    let a2 = base * (2 * j) as f64;
                    let a3 = base * (3 * j) as f64;
                    let (s1, c1) = a1.sin_cos();
                    let (s2, c2) = a2.sin_cos();
                    let (s3, c3) = a3.sin_cos();
                    w1_re.push(c1 as f32);
                    w1_im.push(s1 as f32);
                    w2_re.push(c2 as f32);
                    w2_im.push(s2 as f32);
                    w3_re.push(c3 as f32);
                    w3_im.push(s3 as f32);
                }
                stages.push(Radix4Stage { m, w1_re, w1_im, w2_re, w2_im, w3_re, w3_im });
            }
            m *= 4;
        }
        let reversed = (0..size).map(|index| index.reverse_bits() >> (usize::BITS - bits)).collect();
        Fft { size, odd_bits, stages, reversed }
    }

    fn transform(&self, re: &mut [f32], im: &mut [f32], inverse: bool) {
        if inverse {
            self.transform_inner::<true>(re, im);
        } else {
            self.transform_inner::<false>(re, im);
        }
    }

    fn transform_inner<const INVERSE: bool>(&self, re: &mut [f32], im: &mut [f32]) {
        let size = self.size;
        debug_assert_eq!(re.len(), size);
        debug_assert_eq!(im.len(), size);
        for index in 0..size {
            let other = self.reversed[index];
            if index < other {
                re.swap(index, other);
                im.swap(index, other);
            }
        }
        if self.odd_bits {
            for s in (0..size).step_by(2) {
                let (ar, ai) = (re[s], im[s]);
                let (br, bi) = (re[s + 1], im[s + 1]);
                re[s] = ar + br;
                im[s] = ai + bi;
                re[s + 1] = ar - br;
                im[s + 1] = ai - bi;
            }
        } else {
            // m = 1 radix-4 stage: all twiddles are 1.
            for s in (0..size).step_by(4) {
                let (a0r, a0i) = (re[s], im[s]);
                let (a1r, a1i) = (re[s + 1], im[s + 1]);
                let (a2r, a2i) = (re[s + 2], im[s + 2]);
                let (a3r, a3i) = (re[s + 3], im[s + 3]);
                let (v0r, v0i) = (a0r + a1r, a0i + a1i);
                let (v1r, v1i) = (a0r - a1r, a0i - a1i);
                let (v2r, v2i) = (a2r + a3r, a2i + a3i);
                let (v3r, v3i) = (a2r - a3r, a2i - a3i);
                re[s] = v0r + v2r;
                im[s] = v0i + v2i;
                re[s + 2] = v0r - v2r;
                im[s + 2] = v0i - v2i;
                if INVERSE {
                    re[s + 1] = v1r - v3i;
                    im[s + 1] = v1i + v3r;
                    re[s + 3] = v1r + v3i;
                    im[s + 3] = v1i - v3r;
                } else {
                    re[s + 1] = v1r + v3i;
                    im[s + 1] = v1i - v3r;
                    re[s + 3] = v1r - v3i;
                    im[s + 3] = v1i + v3r;
                }
            }
        }
        for stage in &self.stages {
            let m = stage.m;
            let span = 4 * m;
            let (w1_re, w1_im) = (&stage.w1_re, &stage.w1_im);
            let (w2_re, w2_im) = (&stage.w2_re, &stage.w2_im);
            let (w3_re, w3_im) = (&stage.w3_re, &stage.w3_im);
            for s in (0..size).step_by(span) {
                let (r0, rest_r) = re[s..s + span].split_at_mut(m);
                let (r1, rest_r) = rest_r.split_at_mut(m);
                let (r2, r3) = rest_r.split_at_mut(m);
                let (i0, rest_i) = im[s..s + span].split_at_mut(m);
                let (i1, rest_i) = rest_i.split_at_mut(m);
                let (i2, i3) = rest_i.split_at_mut(m);
                for j in 0..m {
                    let (a0r, a0i) = (r0[j], i0[j]);
                    let (b1r, b1i) = (r1[j], i1[j]);
                    let (b2r, b2i) = (r2[j], i2[j]);
                    let (b3r, b3i) = (r3[j], i3[j]);
                    let (t1r, t1i, t2r, t2i, t3r, t3i) = if INVERSE {
                        (
                            b1r * w2_re[j] + b1i * w2_im[j],
                            b1i * w2_re[j] - b1r * w2_im[j],
                            b2r * w1_re[j] + b2i * w1_im[j],
                            b2i * w1_re[j] - b2r * w1_im[j],
                            b3r * w3_re[j] + b3i * w3_im[j],
                            b3i * w3_re[j] - b3r * w3_im[j],
                        )
                    } else {
                        (
                            b1r * w2_re[j] - b1i * w2_im[j],
                            b1r * w2_im[j] + b1i * w2_re[j],
                            b2r * w1_re[j] - b2i * w1_im[j],
                            b2r * w1_im[j] + b2i * w1_re[j],
                            b3r * w3_re[j] - b3i * w3_im[j],
                            b3r * w3_im[j] + b3i * w3_re[j],
                        )
                    };
                    let (v0r, v0i) = (a0r + t1r, a0i + t1i);
                    let (v1r, v1i) = (a0r - t1r, a0i - t1i);
                    let (v2r, v2i) = (t2r + t3r, t2i + t3i);
                    let (v3r, v3i) = (t2r - t3r, t2i - t3i);
                    r0[j] = v0r + v2r;
                    i0[j] = v0i + v2i;
                    r2[j] = v0r - v2r;
                    i2[j] = v0i - v2i;
                    if INVERSE {
                        r1[j] = v1r - v3i;
                        i1[j] = v1i + v3r;
                        r3[j] = v1r + v3i;
                        i3[j] = v1i - v3r;
                    } else {
                        r1[j] = v1r + v3i;
                        i1[j] = v1i - v3r;
                        r3[j] = v1r - v3i;
                        i3[j] = v1i + v3r;
                    }
                }
            }
        }
        if INVERSE {
            let scale = 1.0 / size as f32;
            for value in re.iter_mut() {
                *value *= scale;
            }
            for value in im.iter_mut() {
                *value *= scale;
            }
        }
    }
}

/// One uniformly partitioned overlap-save convolution with a stretch of the
/// response, two channels at once: left in the real part, right in the
/// imaginary, split apart in the frequency domain where each meets its own
/// channel's response, and joined again for one inverse transform.
struct Stage {
    block: usize,
    partitions: usize,
    bins: usize,
    fft: Fft,
    /// Flat SoA slabs: `partitions * 2 * bins`, indexed as `(partition * 2 + channel) * bins + bin`.
    filters_re: Vec<f32>,
    filters_im: Vec<f32>,
    /// Flat SoA slab rings: `partitions * 2 * bins`, newest at `newest`.
    history_re: Vec<f32>,
    history_im: Vec<f32>,
    newest: usize,
    /// The previous input block, then the current one.
    input: Vec<[f32; 2]>,
    work_re: Vec<f32>,
    work_im: Vec<f32>,
    sums_re: [Vec<f32>; 2],
    sums_im: [Vec<f32>; 2],
}

impl Stage {
    fn new(block: usize, response: &[Vec<f32>; 2], from: usize, to: usize) -> Stage {
        let size = 2 * block;
        let fft = Fft::new(size);
        let partitions = (to.saturating_sub(from)).div_ceil(block).max(1);
        let bins = block + 1;
        let mut work_re = vec![0.0f32; size];
        let mut work_im = vec![0.0f32; size];
        let mut filters_re = vec![0.0f32; partitions * 2 * bins];
        let mut filters_im = vec![0.0f32; partitions * 2 * bins];
        for partition in 0..partitions {
            work_re.fill(0.0);
            work_im.fill(0.0);
            let start = from + partition * block;
            for offset in 0..block {
                let at = start + offset;
                if at < to {
                    work_re[offset] = response[0][at];
                    work_im[offset] = response[1][at];
                }
            }
            fft.transform(&mut work_re, &mut work_im, false);
            let base = partition * 2 * bins;
            let (l_re, r_re) = filters_re[base..base + 2 * bins].split_at_mut(bins);
            let (l_im, r_im) = filters_im[base..base + 2 * bins].split_at_mut(bins);
            split(&work_re, &work_im, l_re, l_im, r_re, r_im);
        }
        Stage {
            block,
            partitions,
            bins,
            fft,
            filters_re,
            filters_im,
            history_re: vec![0.0f32; partitions * 2 * bins],
            history_im: vec![0.0f32; partitions * 2 * bins],
            newest: 0,
            input: vec![[0.0; 2]; size],
            work_re,
            work_im,
            sums_re: [vec![0.0f32; bins], vec![0.0f32; bins]],
            sums_im: [vec![0.0f32; bins], vec![0.0f32; bins]],
        }
    }

    fn clear_sums(&mut self) {
        self.sums_re[0].fill(0.0);
        self.sums_re[1].fill(0.0);
        self.sums_im[0].fill(0.0);
        self.sums_im[1].fill(0.0);
    }

    /// Accumulates spectral products for `start..end` assuming the
    /// upcoming input block will land at `next_newest`.
    fn accumulate_partitions(&mut self, start: usize, end: usize, next_newest: usize) {
        let (partitions, bins) = (self.partitions, self.bins);
        for partition in start..end {
            let slot = (next_newest + partitions - partition) % partitions;
            let h_base = slot * 2 * bins;
            let f_base = partition * 2 * bins;
            for channel in 0..2 {
                let off = channel * bins;
                let sum_re = &mut self.sums_re[channel][..bins];
                let sum_im = &mut self.sums_im[channel][..bins];
                let in_re = &self.history_re[h_base + off..h_base + off + bins];
                let in_im = &self.history_im[h_base + off..h_base + off + bins];
                let flt_re = &self.filters_re[f_base + off..f_base + off + bins];
                let flt_im = &self.filters_im[f_base + off..f_base + off + bins];
                for bin in 0..bins {
                    let ar = in_re[bin];
                    let ai = in_im[bin];
                    let br = flt_re[bin];
                    let bi = flt_im[bin];
                    sum_re[bin] += ar * br - ai * bi;
                    sum_im[bin] += ar * bi + ai * br;
                }
            }
        }
    }

    /// Takes the next input block (in the second half of `input`), finishes
    /// convolution against all partitions, and writes into `output`.
    fn convolve(&mut self, output: &mut [[f32; 2]], preaccumulated_tail: bool) {
        let (block, size, bins, partitions) = (self.block, 2 * self.block, self.bins, self.partitions);
        for (i, &[left, right]) in self.input.iter().enumerate() {
            self.work_re[i] = left;
            self.work_im[i] = right;
        }
        self.fft.transform(&mut self.work_re, &mut self.work_im, false);
        self.newest = (self.newest + 1) % partitions;
        let base = self.newest * 2 * bins;
        let (l_re, r_re) = self.history_re[base..base + 2 * bins].split_at_mut(bins);
        let (l_im, r_im) = self.history_im[base..base + 2 * bins].split_at_mut(bins);
        split(&self.work_re, &self.work_im, l_re, l_im, r_re, r_im);
        if preaccumulated_tail {
            self.accumulate_partitions(0, 1, self.newest);
        } else {
            self.clear_sums();
            self.accumulate_partitions(0, partitions, self.newest);
        }
        // Both channels' real outputs as one complex signal: left + i right.
        let (s0_re, s1_re) = (&self.sums_re[0], &self.sums_re[1]);
        let (s0_im, s1_im) = (&self.sums_im[0], &self.sums_im[1]);
        for bin in 0..bins {
            self.work_re[bin] = s0_re[bin] - s1_im[bin];
            self.work_im[bin] = s0_im[bin] + s1_re[bin];
        }
        for bin in bins..size {
            let m = size - bin;
            self.work_re[bin] = s0_re[m] + s1_im[m];
            self.work_im[bin] = -s0_im[m] + s1_re[m];
        }
        self.fft.transform(&mut self.work_re, &mut self.work_im, true);
        for (i, out) in output.iter_mut().enumerate() {
            *out = [self.work_re[block + i], self.work_im[block + i]];
        }
        self.input.copy_within(block.., 0);
    }
}

/// The left and right spectra, bins 0 to `left_re.len() - 1`, of two real
/// signals transformed together as left + i right.
fn split(re: &[f32], im: &[f32], left_re: &mut [f32], left_im: &mut [f32], right_re: &mut [f32], right_im: &mut [f32]) {
    let size = re.len();
    let bins = left_re.len();
    left_re[0] = re[0];
    left_im[0] = 0.0;
    right_re[0] = im[0];
    right_im[0] = 0.0;
    for bin in 1..bins {
        let m = size - bin;
        let (zr, zi) = (re[bin], im[bin]);
        let (mr, mi) = (re[m], im[m]);
        left_re[bin] = (zr + mr) * 0.5;
        left_im[bin] = (zi - mi) * 0.5;
        right_re[bin] = (zi + mi) * 0.5;
        right_im[bin] = (mr - zr) * 0.5;
    }
}

/// The tail's blocks: 16 quanta each.
const TAIL_BLOCK: usize = 2048;
const TAIL_QUANTA: usize = TAIL_BLOCK / QUANTUM;

/// The reverb: stereo in, stereo out, a quantum at a time.
pub struct Convolver {
    head: Stage,
    tail: Stage,
    /// The tail's output for the block now playing, and how much has played.
    tail_output: Vec<[f32; 2]>,
    played: usize,
    head_output: Vec<[f32; 2]>,
    /// Quanta of silence needed after the last non-zero input until all head
    /// and tail rings and overlap buffers have flushed to exact zero.
    head_ir_quanta: usize,
    head_active_quanta: usize,
    ir_quanta: usize,
    active_quanta: usize,
}

impl Convolver {
    pub fn new(rate: f64) -> Convolver {
        Convolver::with_room(rate, 0.0, 0.0)
    }

    pub fn with_room(rate: f64, decay: f64, damping: f64) -> Convolver {
        Self::from_response(impulse_response_for(rate, decay, damping))
    }

    pub fn with_acoustics(rate: f64, profile: &AcousticProfile) -> Convolver {
        Self::from_response(impulse_response_from_acoustics(rate, profile))
    }

    fn from_response(response: [Vec<f32>; 2]) -> Convolver {
        let length = response[0].len();
        let split_at = TAIL_BLOCK.min(length);
        let head = Stage::new(QUANTUM, &response, 0, split_at);
        let tail = Stage::new(TAIL_BLOCK, &response, split_at, length);
        let head_ir_quanta = head.partitions + 2;
        let ir_quanta = (tail.partitions + 3) * TAIL_QUANTA;
        Convolver {
            head,
            tail,
            tail_output: vec![[0.0; 2]; TAIL_BLOCK],
            played: 0,
            head_output: vec![[0.0; 2]; QUANTUM],
            head_ir_quanta,
            head_active_quanta: 0,
            ir_quanta,
            active_quanta: 0,
        }
    }

    /// Convolves one quantum of input, adding the output to `output`.
    pub fn process(&mut self, input: &[[f32; 2]; QUANTUM], output: &mut [[f32; 2]; QUANTUM]) {
        let silent = input.iter().all(|&[l, r]| l == 0.0 && r == 0.0);
        if !silent {
            self.active_quanta = self.ir_quanta;
            self.head_active_quanta = self.head_ir_quanta;
        } else if self.active_quanta == 0 && self.played == 0 {
            return;
        } else {
            self.active_quanta = self.active_quanta.saturating_sub(1);
        }
        let start = self.played;
        let q = start / QUANTUM;
        if self.head_active_quanta > 0 {
            // The head: this quantum's own contribution, at once.
            self.head.input[QUANTUM..].copy_from_slice(input);
            self.head.convolve(&mut self.head_output, false);
            if silent {
                self.head_active_quanta -= 1;
            }
            for ((out, head), tail) in
                output.iter_mut().zip(&self.head_output).zip(&self.tail_output[start..start + QUANTUM])
            {
                out[0] += head[0] + tail[0];
                out[1] += head[1] + tail[1];
            }
        } else {
            for (out, tail) in output.iter_mut().zip(&self.tail_output[start..start + QUANTUM]) {
                out[0] += tail[0];
                out[1] += tail[1];
            }
        }
        self.tail.input[TAIL_BLOCK + start..TAIL_BLOCK + start + QUANTUM].copy_from_slice(input);
        // Spread partitions `1..tail.partitions` (which only depend on already-
        // completed history blocks) across quanta `0..15` so quantum 15 only
        // runs the FFTs plus partition 0.
        if q == 0 {
            self.tail.clear_sums();
        }
        if q < TAIL_QUANTA - 1 {
            let rest = self.tail.partitions.saturating_sub(1);
            let p_start = 1 + (q * rest) / (TAIL_QUANTA - 1);
            let p_end = 1 + ((q + 1) * rest) / (TAIL_QUANTA - 1);
            if p_start < p_end {
                let next_newest = (self.tail.newest + 1) % self.tail.partitions;
                self.tail.accumulate_partitions(p_start, p_end, next_newest);
            }
        }
        self.played += QUANTUM;
        if self.played == TAIL_BLOCK {
            self.played = 0;
            self.tail.convolve(&mut self.tail_output, true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The straightforward convolution, in f64.
    fn direct(input: &[[f32; 2]], response: &[Vec<f32>; 2]) -> Vec<[f64; 2]> {
        let mut output = vec![[0.0f64; 2]; input.len()];
        for (n, out) in output.iter_mut().enumerate() {
            for channel in 0..2 {
                let mut sum = 0.0f64;
                for k in 0..=n.min(response[channel].len() - 1) {
                    sum += f64::from(input[n - k][channel]) * f64::from(response[channel][k]);
                }
                out[channel] = sum;
            }
        }
        output
    }

    #[test]
    fn fft_round_trips() {
        for size in [16usize, 32, 256] {
            let fft = Fft::new(size);
            let orig_re: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let orig_im: Vec<f32> = (0..size).map(|i| (i * i % 7) as f32).collect();
            let (mut re, mut im) = (orig_re.clone(), orig_im.clone());
            fft.transform(&mut re, &mut im, false);
            // Bin 0 is the sum.
            let expected_dc = (0..size).sum::<usize>() as f32;
            assert!((re[0] - expected_dc).abs() < 1e-2, "size {size}: {} != {expected_dc}", re[0]);
            fft.transform(&mut re, &mut im, true);
            for i in 0..size {
                assert!(
                    (re[i] - orig_re[i]).abs() < 1e-3 && (im[i] - orig_im[i]).abs() < 1e-3,
                    "size {size} idx {i}: ({}, {}) vs ({}, {})",
                    re[i],
                    im[i],
                    orig_re[i],
                    orig_im[i]
                );
            }
        }
    }

    /// The partitioned convolution is the convolution, with no latency, in
    /// both channels and across the head and the tail.
    #[test]
    fn convolution_matches_direct() {
        let rate = 8000.0;
        let response = impulse_response(rate);
        let mut convolver = Convolver::new(rate);
        let quanta = 120;
        let mut state = 1u32;
        let input: Vec<[f32; 2]> = (0..quanta * QUANTUM)
            .map(|i| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (state >> 8) as f32 / (1 << 24) as f32 - 0.5;
                // An impulse, then noise, then silence.
                if i == 0 {
                    [1.0, 0.5]
                } else if i < 4000 {
                    [noise, -noise * 0.5]
                } else {
                    [0.0, 0.0]
                }
            })
            .collect();
        let mut output = Vec::new();
        for chunk in input.chunks(QUANTUM) {
            let mut out = [[0.0f32; 2]; QUANTUM];
            convolver.process(chunk.try_into().unwrap(), &mut out);
            output.extend_from_slice(&out);
        }
        let expected = direct(&input, &response);
        let peak = expected.iter().flat_map(|pair| pair.iter()).fold(0.0f64, |m, v| m.max(v.abs()));
        let worst = output
            .iter()
            .zip(&expected)
            .flat_map(|(a, b)| (0..2).map(move |c| (f64::from(a[c]) - b[c]).abs()))
            .fold(0.0f64, f64::max);
        assert!(worst < peak * 1e-4, "worst error {worst} against a peak of {peak}");
        // The impulse's response, from sample 0.
        assert!((f64::from(output[100][0]) - expected[100][0]).abs() < 1e-5);
    }

    /// The response is pinned: the browser host's ConvolverNode plays the
    /// same one, so it must not drift.
    #[test]
    fn impulse_response_is_pinned() {
        let response = impulse_response(48_000.0);
        assert_eq!(response[0].len(), 76_800);
        let energy: f64 = response.iter().flatten().map(|v| f64::from(*v) * f64::from(*v)).sum();
        assert!((energy - 2.0).abs() < 1e-4, "{energy}");
        // Silent through the predelay, then the noise.
        assert_eq!(response[0][575], 0.0);
        for (channel, index, expected) in [
            (0, 600, 4.367_595_5e-4),
            (1, 600, 6.779_858_4e-4),
            (0, 24_000, -8.966_267_3e-4),
            (1, 70_000, 5.950_865_7e-6),
        ] {
            let value = response[channel][index];
            assert!((value - expected).abs() <= expected.abs() * 1e-5, "{channel} {index}: {value:e}");
        }
    }
}
