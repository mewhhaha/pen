// Bokeh depth of field pass (shaders/dof.wgsl), run in linear HDR light after
// TAA and before motion blur, exposure metering and bloom.
use pen_protocol::{Dof, Matrix};

use crate::matrix::invert;

const SHADER: &str = crate::wgsl!("dof.wgsl");

pub(crate) fn shader(samples: u32) -> String {
    SHADER
        .replace("${depth_texture}", if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" })
        .replace("${fullscreen}", crate::wgsl!("fullscreen.wgsl"))
}

pub(crate) fn uniform(projection: &Matrix, dof: &Dof, size: (u32, u32)) -> [f32; 24] {
    let mut data = [0.0f32; 24];
    data[..16].copy_from_slice(&invert(projection));
    let w = size.0.max(1) as f32;
    let h = size.1.max(1) as f32;
    data[16..20].copy_from_slice(&[dof.focus_distance, dof.focus_range, dof.aperture, dof.max_blur]);
    data[20..24].copy_from_slice(&[w, h, w / h, 0.0]);
    data
}
