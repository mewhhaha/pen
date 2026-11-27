#![cfg_attr(not(feature = "render-subsurface"), allow(dead_code))]
// Screen-space subsurface scattering, for skin and other materials that light
// travels a little way into before it leaves (a material's `"subsurface":
// true`). The scene stamps such surfaces into the stencil (`MARK`), and once the
// opaque surfaces and their decals are drawn and resolved, two passes of
// `shaders/subsurface.wgsl` blur the picture there: along x, then along y,
// the separable approximation of Jimenez et al., "Separable Subsurface
// Scattering" (2015). Both draw before the sky, the occlusion and the mist
// (post.rs), so skin is diffused before it is misted.
//
// The blur is one kernel of seven pairs of taps and a centre, spread over a sum
// of Gaussians that approximates d'Eon and Luebke's profile of skin (GPU Gems 3,
// chapter 14), each channel's profile widened or narrowed by a falloff (red
// reaches farthest) and its strength mixing the blur with the sharp picture.
// This builds the kernel the way the paper's `calculateKernel` does, in f64:
// taps are spaced by `offset = range * sign * (t / range)^2`, denser near the
// centre, each weighted by the profile there times the width of its cell of the
// kernel, the weights of a channel normalised to sum to one, then the centre
// takes `1 - strength + strength * weight` and every other tap `strength *
// weight`, so a flat picture stays as it is. The taps' reach in the world is
// `WIDTH` and in pixels it follows the surface's depth.
/// The stencil bit of surfaces that are diffused on screen; bit 1 is the
/// aberration's and bit 2 the decals'.
pub(crate) const MARK: u32 = 4;

/// The world distance from a pixel to its outermost tap, in the world's units
/// (1 is a metre): a centimetre. The frame has no setting for it yet, so every
/// frame diffuses over the same distance.
pub(crate) const WIDTH: f32 = 0.01;

/// How many taps lie to each side of the centre.
pub(crate) const TAPS: usize = 7;

/// The most the kernel's offsets reach, and the power that spaces them, in the
/// paper's terms (`range` 2 below 20 taps, `exponent` 2).
const RANGE: f64 = 2.0;
const EXPONENT: f64 = 2.0;

/// The width of each channel's profile (the green and blue are narrower than the
/// red, which is what d'Eon's profile fits) and how much of the blur each takes
/// over the sharp picture.
pub(crate) const FALLOFF: [f64; 3] = [1.0, 0.37, 0.3];
pub(crate) const STRENGTH: [f64; 3] = [0.48, 0.41, 0.28];

/// d'Eon and Luebke's sum of Gaussians for the profile of skin: each one's
/// variance (in squared millimetres) and its weight; they sum to one. The
/// first, narrowest is directly bounced light, which the strength accounts for.
pub(crate) const PROFILE: [(f64, f64); 6] =
    [(0.0064, 0.233), (0.0484, 0.100), (0.187, 0.118), (0.567, 0.113), (1.99, 0.358), (7.41, 0.078)];

/// One pair of the kernel's taps: the weight of each of the two, by channel,
/// and how far out of the reach they lie (0 to 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Tap {
    pub weights: [f64; 3],
    pub offset: f64,
}

fn gaussian(variance: f64, r: f64, falloff: f64) -> f64 {
    let scaled = r / (0.001 + falloff);
    (-(scaled * scaled) / (2.0 * variance)).exp() / (std::f64::consts::TAU * variance)
}

fn profile(r: f64) -> [f64; 3] {
    std::array::from_fn(|channel| {
        PROFILE.iter().map(|&(variance, weight)| weight * gaussian(variance, r, FALLOFF[channel])).sum()
    })
}

/// The kernel: the centre first (its offset 0), then each pair of taps out to the
/// reach, nearest first. Both taps of a pair have the weights given.
pub(crate) fn kernel() -> [Tap; TAPS + 1] {
    let count = 2 * TAPS + 1;
    let step = 2.0 * RANGE / (count - 1) as f64;
    // Where each of the taps lies, from the far end of one side to the other's.
    let offsets: Vec<f64> = (0..count)
        .map(|index| {
            let spaced = -RANGE + index as f64 * step;
            let sign = if spaced < 0.0 { -1.0 } else { 1.0 };
            RANGE * sign * (spaced.abs().powf(EXPONENT) / RANGE.powf(EXPONENT))
        })
        .collect();
    let areas: Vec<f64> = (0..count)
        .map(|index| {
            let before = if index > 0 { (offsets[index] - offsets[index - 1]).abs() } else { 0.0 };
            let after = if index < count - 1 { (offsets[index] - offsets[index + 1]).abs() } else { 0.0 };
            (before + after) / 2.0
        })
        .collect();
    let weighed: Vec<[f64; 3]> =
        (0..count).map(|index| profile(offsets[index]).map(|weight| weight * areas[index])).collect();
    let sums: [f64; 3] = std::array::from_fn(|channel| weighed.iter().map(|weights| weights[channel]).sum());
    let normalised =
        |index: usize| -> [f64; 3] { std::array::from_fn(|channel| weighed[index][channel] / sums[channel]) };
    std::array::from_fn(|place| {
        if place == 0 {
            let weights: [f64; 3] = normalised(TAPS);
            Tap { weights: std::array::from_fn(|c| (1.0 - STRENGTH[c]) + STRENGTH[c] * weights[c]), offset: 0.0 }
        } else {
            // The positive side, nearest the centre first.
            let weights: [f64; 3] = normalised(TAPS + place);
            Tap { weights: std::array::from_fn(|c| STRENGTH[c] * weights[c]), offset: offsets[TAPS + place] / RANGE }
        }
    })
}

/// WGSL: the kernel as a constant table, a `vec4f` a place: the weights and the
/// offset out of the reach.
fn kernel_wgsl() -> String {
    let taps = kernel()
        .iter()
        .map(|tap| {
            let [r, g, b] = tap.weights;
            format!("vec4f({r:.9}, {g:.9}, {b:.9}, {:.9})", tap.offset)
        })
        .collect::<Vec<_>>()
        .join(",\n  ");
    format!("const KERNEL = array<vec4f, {}>(\n  {taps});", TAPS + 1)
}

const FULLSCREEN: &str = crate::wgsl!("fullscreen.wgsl");

/// One of the two passes' shader, reading the depth and the stencil marks of a
/// scene of `samples` per pixel.
pub(crate) fn shader(samples: u32) -> String {
    let depth = if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" };
    let marks = if samples > 1 { "texture_multisampled_2d<u32>" } else { "texture_2d<u32>" };
    crate::deferred_wgsl!("subsurface.wgsl")
        .replace("${depth_texture}", depth)
        .replace("${marks_texture}", marks)
        .replace("${fullscreen}", FULLSCREEN)
        .replace("${kernel}", &kernel_wgsl())
        .replace("${mark}", &MARK.to_string())
        .replace("${samples}", &samples.to_string())
}

/// The uniform of one pass: the projection's inverse, the step along its axis
/// (1, 0 or 0, 1), the reach in the world, and the picture's size.
pub(crate) fn uniform(inverse_projection: &[f32; 16], vertical: bool, size: (u32, u32)) -> [f32; 24] {
    let mut data = [0.0; 24];
    data[..16].copy_from_slice(inverse_projection);
    data[16..24].copy_from_slice(&[
        if vertical { 0.0 } else { 1.0 },
        if vertical { 1.0 } else { 0.0 },
        WIDTH,
        0.0,
        size.0 as f32,
        size.1 as f32,
        1.0 / size.0 as f32,
        1.0 / size.1 as f32,
    ]);
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kernel_keeps_a_flat_picture_and_widens_with_the_channel() {
        let taps = kernel();
        for channel in 0..3 {
            // Every tap but the centre is in the kernel twice, one each side.
            let total: f64 =
                taps[0].weights[channel] + 2.0 * taps[1..].iter().map(|tap| tap.weights[channel]).sum::<f64>();
            assert!((total - 1.0).abs() < 1e-12, "channel {channel}: {total}");
        }
        // Out from the centre, to the reach, nearest first.
        assert_eq!(taps[0].offset, 0.0);
        assert!(taps.windows(2).all(|pair| pair[1].offset > pair[0].offset));
        assert!((taps[TAPS].offset - 1.0).abs() < 1e-12);
        // Red reaches farthest: past the middle taps its weights are the largest.
        assert!(taps[TAPS].weights[0] > taps[TAPS].weights[1] && taps[TAPS].weights[1] > taps[TAPS].weights[2]);
        // The centre keeps the most of the weights it can: green and blue's strength is lower.
        assert!(taps[0].weights[2] > taps[0].weights[1] && taps[0].weights[1] > taps[0].weights[0]);
        assert!(PROFILE.iter().map(|&(_, weight)| weight).sum::<f64>() - 1.0 < 1e-12);
    }

    #[test]
    fn the_shader_fills_every_hole_for_both_sample_counts() {
        for samples in [1, 4] {
            let code = shader(samples);
            assert!(!code.contains("${"), "{code}");
            assert!(code.contains("const MARK = 4u;"));
            assert!(code.contains(&format!("sample < {samples}")));
        }
        assert!(
            shader(4).contains("texture_multisampled_2d<u32>") && shader(4).contains("texture_depth_multisampled_2d")
        );
        assert!(shader(1).contains("texture_2d<u32>"));
    }
}
