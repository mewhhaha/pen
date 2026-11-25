// Temporal antialiasing, which the frame's quality turns on. Each frame's
// samples move within their pixels along a Halton sequence (the projection
// is jittered), and a pass after the sky and fog (post.rs) blends the frame
// into the picture the frames before it made: each pixel's depth puts its
// surface in the world, the last frame's camera says where that surface was
// on the screen, and the history there, clipped to the range of the colours
// round the pixel now, takes 90% of the blend. Over eight frames a still
// picture gathers eight samples a pixel. Without motion vectors, what moves
// on its own keeps only what the clipping lets through.
use pen_protocol::Matrix;

use crate::matrix::{invert, multiply};

/// Default native-resolution frames before the jitter repeats.
const SEQUENCE: u64 = 8;

/// This frame's baseline share of the temporal blend.
const BLEND: f32 = 0.1;

/// AMD FSR 2.2 / 3.1 jitter phase count scaling: `ceil(8 * (out_w / in_w)^2)` clamped to `8..=72`.
/// Returns `8` at native resolution (`1.0x`), `18` at Quality (`1.5x`), `24` at Balanced (`1.7x`),
/// `32` at Performance (`2.0x`), and `72` at Ultra Performance (`3.0x`).
pub fn fsr_jitter_phase_count(in_width: u32, out_width: u32) -> u32 {
    if in_width == 0 || out_width <= in_width {
        return 8;
    }
    let ratio = f64::from(out_width) / f64::from(in_width);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped positive phase count")]
    let phases = (8.0 * ratio * ratio).ceil() as u32;
    phases.clamp(8, 72)
}

/// AMD FSR 2.2 / 3.1 negative texture LOD mip bias (`-log2(out_w / in_w)`, clamped to `[-2.5, 0.0]`)
/// so textures sampled at reduced internal render scale retain full native sharpness.
pub fn fsr_mip_bias(in_width: u32, out_width: u32) -> f32 {
    if in_width == 0 || out_width <= in_width {
        return 0.0;
    }
    let ratio = (out_width as f32) / (in_width as f32);
    (-ratio.log2()).clamp(-2.5, 0.0)
}

const SHADER: &str = include_str!("shaders/taa.wgsl");

pub(crate) fn shader(samples: u32) -> String {
    SHADER
        .replace("${depth_texture}", if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" })
        .replace("${fullscreen}", include_str!("shaders/fullscreen.wgsl"))
}

/// The `index`th number of the Halton sequence of `base`, in 0..1.
fn halton(mut index: u64, base: u64) -> f64 {
    let (mut result, mut fraction) = (0.0, 1.0);
    while index > 0 {
        fraction /= base as f64;
        result += fraction * (index % base) as f64;
        index /= base;
    }
    result
}

/// Where frame `frame`'s samples sit within their pixels across a `phase_count`-frame Halton(2, 3) cycle.
pub(crate) fn jitter_for(frame: u64, phase_count: u32) -> (f64, f64) {
    let phases = u64::from(phase_count.max(1));
    let index = frame % phases + 1;
    (halton(index, 2) - 0.5, halton(index, 3) - 0.5)
}

/// Where frame `frame`'s samples sit within their pixels, in pixels from
/// the middle: the Halton (2, 3) points, which spread evenly however many of
/// them are taken.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn jitter(frame: u64) -> (f64, f64) {
    jitter_for(frame, u32::try_from(SEQUENCE).unwrap_or(8))
}

/// `projection` with its picture moved by `offset` pixels across and down a
/// picture of `size`: clip-space x and y gain the offset times w, so it holds
/// for perspective and orthographic projections alike.
pub(crate) fn jittered(projection: &Matrix, offset: (f64, f64), size: (u32, u32)) -> Matrix {
    let x = offset.0 * 2.0 / f64::from(size.0.max(1));
    // Clip space's y goes up the screen.
    let y = -offset.1 * 2.0 / f64::from(size.1.max(1));
    let mut moved = *projection;
    for column in 0..4 {
        let w = f64::from(projection[column * 4 + 3]);
        moved[column * 4] = crate::matrix::narrow(f64::from(projection[column * 4]) + x * w);
        moved[column * 4 + 1] = crate::matrix::narrow(f64::from(projection[column * 4 + 1]) + y * w);
    }
    moved
}

/// What TAA remembers of the frames before: the camera of the last one,
/// and how many it has blended into the history, whose two textures take
/// turns being read and drawn.
#[derive(Default)]
pub(crate) struct Memory {
    /// The last frame's view-projection, unjittered.
    previous: Option<Matrix>,
    frames: u64,
}

impl Memory {
    /// The last frame's view-projection, unjittered, if there was one.
    pub fn previous(&self) -> Option<Matrix> {
        self.previous
    }

    /// Drops history across a camera cut so the next frame does not blend the old view.
    pub fn cut(&mut self) {
        self.previous = None;
    }

    /// Which history texture this frame reads; it draws into the other.
    pub fn read(&self) -> usize {
        usize::from(self.frames % 2 == 1)
    }

    /// The TAA / FSR 2.2-class temporal upscaler uniform for a frame seen through `view` and
    /// `projection`, jittered by `offset` pixels (across and down) into `jittered`, rendered at
    /// `scene_size` and accumulated into `output_size`; the frame is then remembered.
    #[allow(clippy::too_many_arguments)]
    pub fn uniform(
        &mut self,
        view: &Matrix,
        projection: &Matrix,
        (jittered, offset): (&Matrix, (f64, f64)),
        scene_size: (u32, u32),
        output_size: (u32, u32),
        reversed_z: bool,
        has_opaque_reactive: bool,
    ) -> [f32; 44] {
        let now = multiply(projection, view);
        let fresh = self.previous.is_none();
        let mut data = [0.0; 44];
        data[..16].copy_from_slice(&invert(&multiply(jittered, view)));
        data[16..32].copy_from_slice(&self.previous.unwrap_or(now));
        let reset = if fresh { 1.0 } else { 0.0 };
        let (across, down) = (offset.0 / f64::from(scene_size.0.max(1)), offset.1 / f64::from(scene_size.1.max(1)));
        let (across, down) = (crate::matrix::narrow(across), crate::matrix::narrow(down));
        let phases = fsr_jitter_phase_count(scene_size.0, output_size.0);
        data[32..40].copy_from_slice(&[
            across,
            down,
            BLEND,
            reset,
            output_size.0 as f32,
            output_size.1 as f32,
            scene_size.0 as f32,
            scene_size.1 as f32,
        ]);
        data[40..44].copy_from_slice(&[
            if reversed_z { 1.0 } else { 0.0 },
            phases as f32,
            if has_opaque_reactive { 1.0 } else { 0.0 },
            if reversed_z { 0.0 } else { 1.0 },
        ]);
        self.previous = Some(now);
        self.frames += 1;
        data
    }
}

/// A texture the history is kept in.
pub(crate) fn history(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("history"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::IDENTITY;

    #[test]
    fn jitter_spreads_within_the_pixel() {
        let points: Vec<(f64, f64)> = (0..SEQUENCE).map(jitter).collect();
        assert_eq!(points[0], (0.0, halton(1, 3) - 0.5));
        assert!((points[0].1 + 1.0 / 6.0).abs() < 1e-12);
        assert!(points.iter().all(|&(x, y)| x.abs() < 0.5 && y.abs() < 0.5));
        // Each half of the pixel, across and down, gets half the samples.
        assert_eq!(points.iter().filter(|point| point.0 < 0.0).count(), 4);
        assert_eq!(points.iter().filter(|point| point.1 < 0.0).count(), 4);
        let mean = points.iter().fold((0.0, 0.0), |sum, point| (sum.0 + point.0, sum.1 + point.1));
        assert!(mean.0.abs() / 8.0 < 0.07 && mean.1.abs() / 8.0 < 0.07, "{mean:?}");
        assert_eq!(jitter(SEQUENCE), points[0]);
    }

    #[test]
    fn jittered_projection_moves_the_picture_by_pixels() {
        // A perspective projection (w = -z) and an orthographic one (w = 1).
        let mut perspective = [0.0; 16];
        perspective[0] = 1.0;
        perspective[5] = 1.0;
        perspective[10] = -1.0;
        perspective[11] = -1.0;
        perspective[14] = -0.2;
        for projection in [perspective, IDENTITY] {
            let moved = jittered(&projection, (0.25, -0.5), (100, 50));
            let point = [0.3f32, -0.2, -4.0, 1.0];
            let clip = |m: &Matrix| -> [f64; 4] {
                let mut out = [0.0; 4];
                for (row, value) in out.iter_mut().enumerate() {
                    *value = (0..4).map(|k| f64::from(m[k * 4 + row]) * f64::from(point[k])).sum();
                }
                out
            };
            let (before, after) = (clip(&projection), clip(&moved));
            // A quarter pixel right, half a pixel up, in NDC units of 2 / size.
            let dx = after[0] / after[3] - before[0] / before[3];
            let dy = after[1] / after[3] - before[1] / before[3];
            assert!((dx - 0.25 * 2.0 / 100.0).abs() < 1e-6, "{dx}");
            assert!((dy - 0.5 * 2.0 / 50.0).abs() < 1e-6, "{dy}");
            assert_eq!((after[2], after[3]), (before[2], before[3]));
        }
    }

    #[test]
    fn memory_starts_fresh_then_remembers_the_camera() {
        let mut memory = Memory::default();
        let mut moved = IDENTITY;
        moved[12] = 1.0;
        let first = memory.uniform(&IDENTITY, &IDENTITY, (&IDENTITY, (0.0, 0.0)), (8, 8), (8, 8), false, false);
        assert_eq!((first[35], memory.read()), (1.0, 1));
        let second = memory.uniform(&moved, &IDENTITY, (&IDENTITY, (0.25, -0.5)), (8, 8), (16, 16), false, true);
        assert_eq!((second[35], memory.read()), (0.0, 0));
        // The jitter, in texture coordinates.
        assert_eq!(&second[32..34], &[0.25 / 8.0, -0.5 / 8.0]);
        // The last frame's camera, as it was, and this one's inverse.
        assert_eq!(&second[16..32], &IDENTITY);
        assert_eq!(second[12], -1.0);
        assert_eq!(&second[36..40], &[16.0, 16.0, 8.0, 8.0]);
        assert_eq!(&second[40..44], &[0.0, 32.0, 1.0, 1.0]);
    }

    #[test]
    fn fsr_jitter_phases_and_mip_bias_follow_upscale_ratio() {
        assert_eq!(fsr_jitter_phase_count(1920, 1920), 8);
        assert_eq!(fsr_jitter_phase_count(1280, 1920), 18); // 1.5x Quality
        assert_eq!(fsr_jitter_phase_count(960, 1920), 32); // 2.0x Performance
        assert_eq!(fsr_jitter_phase_count(640, 1920), 72); // 3.0x Ultra Performance
        assert!((fsr_mip_bias(1920, 1920) - 0.0).abs() < 1e-6);
        assert!((fsr_mip_bias(960, 1920) - (-1.0)).abs() < 1e-5);
        assert!((fsr_mip_bias(1280, 1920) - (-1.5f32.log2())).abs() < 1e-5);
    }
}
