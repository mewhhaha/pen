//! Noise and heightfields: two-dimensional simplex noise (Gustavson's),
//! its gradients picked by an integer hash of the lattice point and the
//! seed rather than a permutation table, summed in octaves. It takes only
//! additions, multiplications, divisions and floors, which IEEE rounds the
//! same on every machine, so a seed gives the same heights everywhere.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "heights narrowed to f32 lanes, lattice cells wrapped into u32"
)]
use pen_protocol::boundary::{MAX_SERVICE_DATA, whole};
use pen_protocol::{noise_op, noise_style, service};
use pen_runtime::services::{Answer, Request, Service};

use crate::{Lanes, counted};

/// The most octaves a request may sum.
const MOST_OCTAVES: u32 = 16;
/// Where the lattice skews the plane into triangles, and back:
/// (sqrt(3) - 1) / 2 and (3 - sqrt(3)) / 6.
const SKEW: f64 = 0.366_025_403_784_438_6;
const UNSKEW: f64 = 0.211_324_865_405_187_1;
/// The twelve gradients of Gustavson's simplex noise, in the plane.
const GRADIENTS: [(f64, f64); 12] = [
    (1.0, 1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (-1.0, -1.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (0.0, 1.0),
    (0.0, -1.0),
];

/// Noise as a service: heights over a grid, or at the points the game
/// sends, from a seed and the octaves' settings.
pub struct Noise;

impl<Input> Service<Input> for Noise {
    fn code(&self) -> u32 {
        service::NOISE
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        if !matches!(request.operation, noise_op::HEIGHTFIELD | noise_op::SAMPLE) {
            return Some(Answer::unknown());
        }
        Some(heights(request).unwrap_or_else(Answer::failed))
    }
}

/// How the octaves add up: their settings and the seed.
struct Fractal {
    frequency: f64,
    amplitude: f64,
    lacunarity: f64,
    gain: f64,
    ridged: bool,
    seed: u32,
    octaves: u32,
}

/// A 32-bit hash of a lattice point and a seed (wrapping multiplies and
/// shifted xors).
fn hash(x: u32, y: u32, seed: u32) -> u32 {
    let mut hash = x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x2c1b_3c6d);
    hash ^= hash >> 12;
    hash = hash.wrapping_mul(0x297a_2d39);
    hash ^ (hash >> 15)
}

/// The lattice cell of a floored coordinate, wrapped to 32 bits.
fn cell(floored: f64) -> u32 {
    floored.rem_euclid(4_294_967_296.0) as u32
}

/// Simplex noise at (x, y), within -1 and 1.
fn simplex(x: f64, y: f64, seed: u32) -> f64 {
    let skew = (x + y) * SKEW;
    let (i, j) = ((x + skew).floor(), (y + skew).floor());
    let unskew = (i + j) * UNSKEW;
    let (x0, y0) = (x - (i - unskew), y - (j - unskew));
    let (step_i, step_j) = if x0 > y0 { (1.0, 0.0) } else { (0.0, 1.0) };
    let corners = [
        (x0, y0, 0.0, 0.0),
        (x0 - step_i + UNSKEW, y0 - step_j + UNSKEW, step_i, step_j),
        (x0 - 1.0 + 2.0 * UNSKEW, y0 - 1.0 + 2.0 * UNSKEW, 1.0, 1.0),
    ];
    let mut total = 0.0;
    for (dx, dy, di, dj) in corners {
        let falloff = 0.5 - dx * dx - dy * dy;
        if falloff > 0.0 {
            let (gx, gy) = GRADIENTS[(hash(cell(i + di), cell(j + dj), seed) % 12) as usize];
            let falloff = falloff * falloff;
            total += falloff * falloff * (gx * dx + gy * dy);
        }
    }
    // Gustavson's scale, which brings the extremes near -1 and 1; the clamp
    // keeps the rare overshoot within them.
    (70.0 * total).clamp(-1.0, 1.0)
}

impl Fractal {
    /// The octaves summed at (x, z): each a new seed, `lacunarity` times
    /// the last's frequency and `gain` times its weight, then scaled so the
    /// heights stay within the amplitude.
    fn height(&self, x: f64, z: f64) -> f64 {
        let (mut frequency, mut weight, mut total, mut weights) = (self.frequency, 1.0, 0.0, 0.0);
        for octave in 0..self.octaves {
            let noise = simplex(x * frequency, z * frequency, hash(self.seed, octave, 0x5eed));
            let value = if self.ridged {
                let crest = 1.0 - noise.abs();
                crest * crest * 2.0 - 1.0
            } else {
                noise
            };
            total += weight * value;
            weights += weight;
            frequency *= self.lacunarity;
            weight *= self.gain;
        }
        if weights > 0.0 { self.amplitude * total / weights } else { 0.0 }
    }
}

fn heights(request: &Request) -> Option<Answer> {
    let [columns, rows, seed, octaves] = request.numbers;
    let mut lanes = Lanes::new(request.values);
    let &[frequency, amplitude, lacunarity, gain, style] = lanes.numbers(5)? else { return None };
    let fractal = Fractal {
        frequency: f64::from(frequency),
        amplitude: f64::from(amplitude),
        lacunarity: f64::from(lacunarity),
        gain: f64::from(gain),
        ridged: whole(style).filter(|style| *style <= noise_style::RIDGED)? == noise_style::RIDGED,
        seed: whole(seed)?,
        octaves: whole(octaves).filter(|octaves| (1..=MOST_OCTAVES).contains(octaves))?,
    };
    if !(fractal.frequency > 0.0 && fractal.lacunarity > 0.0 && fractal.gain > 0.0) {
        return None;
    }
    let heights: Vec<f32> = if request.operation == noise_op::HEIGHTFIELD {
        let (columns, rows) = (whole(columns)? as usize, whole(rows)? as usize);
        columns.checked_mul(rows).filter(|count| *count <= MAX_SERVICE_DATA)?;
        let &[x, z, spacing] = lanes.numbers(3)? else { return None };
        if !lanes.done() || spacing <= 0.0 {
            return None;
        }
        let (x, z, spacing) = (f64::from(x), f64::from(z), f64::from(spacing));
        (0..rows)
            .flat_map(|row| (0..columns).map(move |column| (column, row)))
            .map(|(column, row)| fractal.height(x + column as f64 * spacing, z + row as f64 * spacing) as f32)
            .collect()
    } else {
        let mut heights = Vec::new();
        while !lanes.done() {
            let &[x, z] = lanes.numbers(2)? else { return None };
            heights.push(fractal.height(f64::from(x), f64::from(z)) as f32);
        }
        heights
    };
    Some(counted(heights.len(), heights))
}

#[cfg(test)]
mod tests {
    use pen_protocol::service_status;

    use super::*;

    fn ask(operation: u32, numbers: [f32; 4], values: &[f32]) -> Answer {
        Service::<()>::call(&mut Noise, &Request { id: 1, operation, numbers, values }).unwrap()
    }

    /// Five octaves of fBm, a tenth of a cycle a metre, 8 metres high.
    const HILLS: [f32; 5] = [0.1, 8.0, 2.0, 0.5, 0.0];

    #[test]
    fn a_heightfield_is_the_samples_at_its_points() {
        let field = ask(noise_op::HEIGHTFIELD, [4.0, 3.0, 7.0, 5.0], &[&HILLS[..], &[-2.0, 5.0, 0.5]].concat());
        assert_eq!((field.status, field.value, field.values.len()), (service_status::OK, 12.0, 12));
        // The sample at column 3, row 2: (-2 + 1.5, 5 + 1).
        let point = ask(noise_op::SAMPLE, [0.0, 0.0, 7.0, 5.0], &[&HILLS[..], &[-0.5, 6.0]].concat());
        assert_eq!(point.values, [field.values[11]]);
        assert!(field.values.iter().all(|height| height.abs() <= 8.0));
        assert!(field.values.windows(2).all(|pair| pair[0] != pair[1]), "it varies");
    }

    #[test]
    fn seeds_and_styles_change_the_heights_and_stay_within_the_amplitude() {
        let grid = |seed: f32, style: f32| {
            let settings = [0.05, 3.0, 2.0, 0.5, style, 100.0, -40.0, 1.0];
            ask(noise_op::HEIGHTFIELD, [32.0, 32.0, seed, 6.0], &settings).values
        };
        let (fbm, other, ridged) = (grid(1.0, 0.0), grid(2.0, 0.0), grid(1.0, 1.0));
        assert_ne!(fbm, other);
        assert_ne!(fbm, ridged);
        for heights in [&fbm, &other, &ridged] {
            assert!(heights.iter().all(|height| height.abs() <= 3.0));
            let (low, high) = heights.iter().fold((f32::MAX, f32::MIN), |(low, high), h| (low.min(*h), high.max(*h)));
            assert!(high - low > 1.0, "hills, not a plain: {low} to {high}");
        }
        assert_eq!(grid(1.0, 0.0), fbm, "the same seed, the same heights");
    }

    #[test]
    fn simplex_noise_is_continuous_and_spans_its_range() {
        let mut extremes = (0.0f64, 0.0f64);
        for step in 0..20_000 {
            let (x, y) = (f64::from(step) * 0.013 - 100.0, f64::from(step % 97) * 0.21);
            let value = simplex(x, y, 3);
            extremes = (extremes.0.min(value), extremes.1.max(value));
            assert!((value - simplex(x + 1e-4, y, 3)).abs() < 0.01, "no jumps at {x}, {y}");
        }
        assert!(extremes.0 < -0.7 && extremes.1 > 0.7, "{extremes:?}");
    }

    #[test]
    fn settings_that_make_no_noise_fail() {
        let failed =
            |operation, numbers, values: &[f32]| ask(operation, numbers, values).status == service_status::FAILED;
        assert!(failed(noise_op::SAMPLE, [0.0, 0.0, 1.0, 0.0], &HILLS), "no octaves");
        assert!(failed(noise_op::SAMPLE, [0.0, 0.0, 1.0, 17.0], &HILLS), "too many octaves");
        assert!(failed(noise_op::SAMPLE, [0.0, 0.0, 1.0, 2.0], &[0.0, 1.0, 2.0, 0.5, 0.0]), "no frequency");
        assert!(failed(noise_op::SAMPLE, [0.0, 0.0, 1.0, 2.0], &[&HILLS[..], &[1.0]].concat()), "half a point");
        assert!(
            failed(noise_op::HEIGHTFIELD, [200.0, 200.0, 1.0, 2.0], &[&HILLS[..], &[0.0, 0.0, 1.0]].concat()),
            "too many heights"
        );
        assert!(
            failed(noise_op::HEIGHTFIELD, [2.0, 2.0, 1.0, 2.0], &[&HILLS[..], &[0.0, 0.0, 0.0]].concat()),
            "no spacing"
        );
        assert_eq!(ask(noise_op::SAMPLE, [0.0, 0.0, 1.0, 1.0], &HILLS), Answer::ok(0.0), "no points, no heights");
    }
}
