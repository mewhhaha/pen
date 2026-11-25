//! Analytic shapes with premultiplied output and screen-space clipping.

use bytemuck::{Pod, Zeroable};

use super::{Draw2d, buffer, pipeline, prepared};

pub const SDF_UI_WGSL: &str = include_str!("../shaders/render2d_sdf_ui.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SdfShape {
    /// Top-left X/Y and size in physical pixels. Shape rotation is about its center.
    pub rect: [f32; 4],
    /// Radius in pixels, ordered top-left, top-right, bottom-right, bottom-left.
    pub corner_radii: [f32; 4],
    /// Colors use straight linear RGBA and may have HDR RGB.
    pub fill_color: [f32; 4],
    pub fill_color_end: [f32; 4],
    pub border_color: [f32; 4],
    pub shadow_color: [f32; 4],
    /// 0 is rounded rect, 1 is ellipse, and 2 is capsule.
    pub kind: u32,
    pub border_width_px: f32,
    /// 0 is solid, 1 is linear, and 2 is radial.
    pub gradient_kind: u32,
    /// Clockwise linear gradient angle in radians, relative to the shape.
    pub gradient_angle: f32,
    pub shadow_offset: [f32; 2],
    /// Gaussian standard deviation in pixels.
    pub shadow_blur_px: f32,
    pub clip_radius_px: f32,
    /// Clip rectangle in screen pixels. A nonpositive width disables clipping.
    pub clip_rect: [f32; 4],
    /// Radial gradient center in shape UV.
    pub gradient_center: [f32; 2],
    /// Zero derives the radius from the farthest corner.
    pub gradient_radius_px: f32,
    /// Clockwise shape rotation in radians, useful for capsule lines.
    pub rotation: f32,
}

impl Default for SdfShape {
    fn default() -> Self {
        Self {
            rect: [0.0, 0.0, 1.0, 1.0],
            corner_radii: [0.0; 4],
            fill_color: [1.0; 4],
            fill_color_end: [1.0; 4],
            border_color: [0.0; 4],
            shadow_color: [0.0; 4],
            kind: 0,
            border_width_px: 0.0,
            gradient_kind: 0,
            gradient_angle: 0.0,
            shadow_offset: [0.0; 2],
            shadow_blur_px: 0.0,
            clip_radius_px: 0.0,
            clip_rect: [0.0; 4],
            gradient_center: [0.5; 2],
            gradient_radius_px: 0.0,
            rotation: 0.0,
        }
    }
}

pub struct SdfShapePass {
    pipeline: wgpu::RenderPipeline,
}

impl SdfShapePass {
    /// Accepts any blendable color target format.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        Self {
            pipeline: pipeline(
                device,
                "2D SDF shapes",
                SDF_UI_WGSL,
                target_format,
                wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            ),
        }
    }

    /// Prepare one draw, with shapes in back-to-front order.
    pub fn prepare(&self, device: &wgpu::Device, viewport: [f32; 2], shapes: &[SdfShape]) -> Option<Draw2d> {
        if shapes.is_empty() {
            return None;
        }
        let viewport = [viewport[0], viewport[1], 0.0, 0.0];
        let uniform = buffer(device, "2D SDF viewport", bytemuck::bytes_of(&viewport), wgpu::BufferUsages::UNIFORM);
        let shapes_buffer =
            buffer(device, "2D SDF instances", bytemuck::cast_slice(shapes), wgpu::BufferUsages::STORAGE);
        Some(prepared(
            device,
            &self.pipeline,
            &[uniform.as_entire_binding(), shapes_buffer.as_entire_binding()],
            u32::try_from(shapes.len()).expect("shape instance count exceeds u32"),
        ))
    }
}
