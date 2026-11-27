#![cfg_attr(not(feature = "render-taa"), allow(dead_code))]
// Velocity motion blur pass (shaders/motion_blur.wgsl), run in linear HDR
// light after TAA and depth of field, before auto-exposure metering and bloom.
use pen_protocol::{Matrix, MotionBlur};

use crate::matrix::{invert, multiply, narrow};

pub(crate) fn shader(samples: u32) -> String {
    crate::deferred_wgsl!("motion_blur.wgsl")
        .replace("${depth_texture}", if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" })
        .replace("${fullscreen}", crate::wgsl!("fullscreen.wgsl"))
}

pub(crate) fn uniform(
    view: &Matrix,
    (jittered, offset): (&Matrix, (f64, f64)),
    previous_vp: Option<Matrix>,
    mb: &MotionBlur,
    size: (u32, u32),
) -> [f32; 40] {
    let mut data = [0.0f32; 40];
    data[..16].copy_from_slice(&invert(&multiply(jittered, view)));
    data[16..32].copy_from_slice(&previous_vp.unwrap_or_else(|| multiply(jittered, view)));
    let samples = if mb.samples == 0 { 8 } else { mb.samples.clamp(4, 32) };
    let has_prev = if previous_vp.is_some() { 1.0 } else { 0.0 };
    let (across, down) = (narrow(offset.0 / f64::from(size.0.max(1))), narrow(offset.1 / f64::from(size.1.max(1))));
    data[32..36].copy_from_slice(&[mb.strength, mb.max_radius, samples as f32, has_prev]);
    data[36..40].copy_from_slice(&[size.0 as f32, size.1 as f32, across, down]);
    data
}
