// Screen-Space Global Illumination (SSGI) diffuse bounce pass.
use pen_protocol::Matrix;

use crate::matrix::invert;

const FULLSCREEN: &str = include_str!("shaders/fullscreen.wgsl");
const SSGI_SHADER: &str = include_str!("shaders/ssgi.wgsl");

/// World-space bounce march radius, indirect intensity, and depth-hit thickness.
const SSGI_RADIUS: f32 = 3.5;
const SSGI_INTENSITY: f32 = 1.6;
const SSGI_THICKNESS: f32 = 0.75;
const SSGI_MAX_PIXELS: f32 = 96.0;

pub(crate) fn shader(samples: u32) -> String {
    let depth_texture = if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" };
    SSGI_SHADER
        .replace("${depth_texture}", depth_texture)
        .replace("${fullscreen}", FULLSCREEN)
}

/// Packs the 160-byte `Ssgi` uniform buffer.
pub(crate) fn uniform(projection: &Matrix, size: (u32, u32)) -> [f32; 40] {
    let mut data = [0.0f32; 40];
    data[..16].copy_from_slice(projection);
    data[16..32].copy_from_slice(&invert(projection));
    data[32..36].copy_from_slice(&[SSGI_RADIUS, SSGI_INTENSITY, SSGI_THICKNESS, SSGI_MAX_PIXELS]);
    #[allow(clippy::cast_precision_loss, reason = "screen size in pixels")]
    let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
    data[36..40].copy_from_slice(&[w, h, 1.0 / w, 1.0 / h]);
    data
}
