//! Analytic shapes with premultiplied output and screen-space clipping.

use bytemuck::{Pod, Zeroable};

use super::{Draw2d, buffer, pipeline, prepared};

pub const SDF_UI_WGSL: &str = crate::wgsl!("render2d_sdf_ui.wgsl");

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
        let limits = device.limits();
        let bytes = std::mem::size_of_val(shapes) as u64;
        if shapes.is_empty() || bytes > limits.max_storage_buffer_binding_size.min(limits.max_buffer_size) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::util::DeviceExt as _;

    #[test]
    fn thin_ellipse_matches_an_independent_pixel_distance_reference() {
        let (device, queue) = crate::testing::device().expect("headless GPU adapter required");
        let points =
            [[999.0_f32, 0.0001], [999.5, 0.020], [1000.0, 0.015], [1000.1, 0.01], [985.0, 0.1], [500.0, 0.000001]];
        let wgsl = format!(
            "{SDF_UI_WGSL}
@group(0) @binding(8) var<storage, read> samples: array<vec2f>;
@group(0) @binding(9) var<storage, read_write> distances: array<f32>;
@compute @workgroup_size(1)
fn witness(@builtin(global_invocation_id) id: vec3u) {{
    distances[id.x] = ellipse_distance(samples[id.x], vec2f(1000.0, 0.5));
}}"
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("witness"),
            compilation_options: Default::default(),
            cache: crate::pipeline_cache::active().as_ref(),
        });
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&points),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (points.len() * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 8, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: output.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(u32::try_from(points.len()).unwrap(), 1, 1);
        }
        queue.submit([encoder.finish()]);
        let read = crate::testing::read(&device, &queue, &output, output.size());
        for (bits, p) in read.as_chunks::<4>().0.iter().zip(points) {
            let actual = f64::from(f32::from_le_bytes(*bits));
            let [x, y] = p.map(f64::from);
            let distance = |theta: f64| (x - 1000.0 * theta.cos()).hypot(y - 0.5 * theta.sin());
            let mut left = 0.0;
            let mut right = std::f64::consts::FRAC_PI_2;
            for _ in 0..80 {
                let a = (left * 2.0 + right) / 3.0;
                let b = (left + right * 2.0) / 3.0;
                if distance(a) < distance(b) {
                    right = b;
                } else {
                    left = a;
                }
            }
            let sign = if (x / 1000.0).powi(2) + (y * 2.0).powi(2) < 1.0 { -1.0 } else { 1.0 };
            let expected = sign * distance((left + right) * 0.5);
            assert!((actual - expected).abs() < 0.02, "point {p:?} gave {actual} px, expected {expected} px");
        }
    }
}
