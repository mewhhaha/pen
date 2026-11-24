//! Render-to-Texture Camera Targets, Planar Reflection / Oblique Clipping Math,
//! and Split-Screen Multi-Viewport Layout & Compositor.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "GPU viewport and pixel dimensions")]

use anyhow::{Result, ensure};

use crate::RenderTarget;

/// Explicit color transfer mode for [`SplitScreenCompositor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorTransfer {
    /// Copies sampled color channels directly (use when source and destination share the same encoding,
    /// e.g. `render_view()` -> `Rgba8Unorm` or linear -> linear).
    Passthrough,
    /// Encodes linear-light source samples to sRGB (use when sampling linear `Rgba16Float` or
    /// hardware-linearized `sample_view()` into a non-sRGB `Rgba8Unorm`/`Bgra8Unorm` surface).
    LinearToSrgb,
}

/// Standalone WGSL source for compositing split-screen viewports onto a single output surface.
pub const SPLIT_SCREEN_WGSL: &str = r#"
struct BlitUniform {
  rect_uv: vec4f, // xy = uv_min, zw = uv_max
  flags: vec4f,   // x = linear_to_srgb (>0.5), yzw = reserved
};
@group(0) @binding(0) var<uniform> u: BlitUniform;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var src_samp: sampler;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

fn linear_to_srgb_channel(c: f32) -> f32 {
  let x = clamp(c, 0.0, 1.0);
  return select(1.055 * pow(x, 1.0 / 2.4) - 0.055, x * 12.92, x <= 0.0031308);
}

@vertex fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
  let positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0),
  );
  let p = positions[vid];
  let uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return VsOut(vec4f(p, 0.0, 1.0), mix(u.rect_uv.xy, u.rect_uv.zw, uv));
}

@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
  let s = textureSample(src_tex, src_samp, in.uv);
  if (u.flags.x > 0.5) {
    return vec4f(
      linear_to_srgb_channel(s.r),
      linear_to_srgb_channel(s.g),
      linear_to_srgb_channel(s.b),
      s.a,
    );
  }
  return s;
}
"#;

/// Offscreen `RenderTarget` for secondary cameras (mirrors, portals, security monitors,
/// minimaps, or split-screen players) that can be sampled directly by scene materials.
pub struct OffscreenRenderTarget {
    texture: wgpu::Texture,
    render_view: wgpu::TextureView,
    sample_view: wgpu::TextureView,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

impl OffscreenRenderTarget {
    /// Creates an offscreen color target of `(width, height)` with `format`.
    /// For `Rgba8Unorm` / `Bgra8Unorm`, enables a matching sRGB view so PBR materials
    /// automatically linearize display-referred rendered cameras on sample.
    pub fn new(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Result<Self> {
        let limit = device.limits().max_texture_dimension_2d;
        ensure!(
            width >= 1 && height >= 1 && width <= limit && height <= limit,
            "offscreen render target size {width}x{height} exceeds valid range 1..={limit}"
        );
        let srgb_alt = match format {
            wgpu::TextureFormat::Rgba8Unorm => Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            wgpu::TextureFormat::Bgra8Unorm => Some(wgpu::TextureFormat::Bgra8UnormSrgb),
            _ => None,
        };
        let view_formats: Vec<wgpu::TextureFormat> = srgb_alt.into_iter().collect();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen_camera_target"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &view_formats,
        });
        let render_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sample_view = if let Some(srgb) = srgb_alt {
            texture.create_view(&wgpu::TextureViewDescriptor { format: Some(srgb), ..Default::default() })
        } else {
            texture.create_view(&wgpu::TextureViewDescriptor::default())
        };
        Ok(Self { texture, render_view, sample_view, width, height, format })
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// View in the target's native rendering format (for passthrough blitting or display-encoded readback).
    pub fn render_view(&self) -> &wgpu::TextureView {
        &self.render_view
    }

    /// View for material sampling (uses hardware sRGB linearization when backed by `Rgba8Unorm`/`Bgra8Unorm`).
    pub fn sample_view(&self) -> &wgpu::TextureView {
        &self.sample_view
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl RenderTarget for OffscreenRenderTarget {
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn view(&mut self) -> Option<wgpu::TextureView> {
        Some(self.render_view.clone())
    }

    fn present(&mut self) {}
}

fn mul4_f64(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut out = [0.0f64; 16];
    for c in 0..4 {
        for r in 0..4 {
            out[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
        }
    }
    out
}

fn to_f64(m: &[f32; 16]) -> [f64; 16] {
    let mut out = [0.0f64; 16];
    for i in 0..16 {
        out[i] = f64::from(m[i]);
    }
    out
}

fn to_f32(m: &[f64; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = m[i] as f32;
    }
    out
}

fn invert4_checked(m: &[f64; 16]) -> Result<[f64; 16]> {
    // Gauss-Jordan elimination with partial pivoting in f64 on column-major 4x4 matrix.
    let mut a = [[0.0f64; 8]; 4];
    for r in 0..4 {
        for c in 0..4 {
            a[r][c] = m[c * 4 + r];
        }
        a[r][r + 4] = 1.0;
    }
    for col in 0..4 {
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for (r, row) in a.iter().enumerate().skip(col + 1) {
            if row[col].abs() > best {
                best = row[col].abs();
                pivot = r;
            }
        }
        ensure!(best > 1e-12 && best.is_finite(), "singular 4x4 matrix cannot be inverted");
        if pivot != col {
            a.swap(col, pivot);
        }
        let inv_diag = 1.0 / a[col][col];
        for v in &mut a[col] {
            *v *= inv_diag;
        }
        let pivot_row = a[col];
        for (r, row) in a.iter_mut().enumerate() {
            if r != col {
                let factor = row[col];
                for (dst, &src) in row.iter_mut().zip(pivot_row.iter()) {
                    *dst -= factor * src;
                }
            }
        }
    }
    let mut inv = [0.0f64; 16];
    for c in 0..4 {
        for r in 0..4 {
            inv[c * 4 + r] = a[r][c + 4];
        }
    }
    Ok(inv)
}

fn normalize3_f64(v: [f32; 3]) -> Result<[f64; 3]> {
    let (x, y, z) = (f64::from(v[0]), f64::from(v[1]), f64::from(v[2]));
    ensure!(x.is_finite() && y.is_finite() && z.is_finite(), "non-finite plane normal");
    let len = (x * x + y * y + z * z).sqrt();
    ensure!(len > 1e-8, "degenerate zero plane normal");
    Ok([x / len, y / len, z / len])
}

/// Computes the raw Householder-reflected camera view matrix across `(plane_point, plane_normal)`.
/// Note: `det = -1` (improper reflection), which reverses triangle winding. For standard CCW
/// back-face-culled pipelines (including [`crate::Renderer::render_camera_to_texture`]), use
/// [`planar_reflection_camera`] instead, which preserves right-handed CCW winding (`det = +1`)
/// with a horizontally mirrored UV lookup `u' = 1 - u`.
pub fn planar_reflection_view(view: &[f32; 16], plane_point: [f32; 3], plane_normal: [f32; 3]) -> Result<[f32; 16]> {
    ensure!(view.iter().all(|v| v.is_finite()), "non-finite view matrix");
    ensure!(plane_point.iter().all(|v| v.is_finite()), "non-finite plane point");
    let [nx, ny, nz] = normalize3_f64(plane_normal)?;
    let d = -(nx * f64::from(plane_point[0]) + ny * f64::from(plane_point[1]) + nz * f64::from(plane_point[2]));
    let refl = [
        1.0 - 2.0 * nx * nx, -2.0 * ny * nx,      -2.0 * nz * nx,      0.0,
        -2.0 * nx * ny,      1.0 - 2.0 * ny * ny, -2.0 * nz * ny,      0.0,
        -2.0 * nx * nz,      -2.0 * ny * nz,      1.0 - 2.0 * nz * nz, 0.0,
        -2.0 * d * nx,       -2.0 * d * ny,       -2.0 * d * nz,       1.0,
    ];
    Ok(to_f32(&mul4_f64(&to_f64(view), &refl)))
}

/// Computes Lengyel's oblique near-plane clipping projection matrix (WebGPU `0..1` clip Z) so all
/// geometry on the negative side of `(plane_point, plane_normal)` is clipped by the hardware near plane.
/// Preserves the caller's exact `plane_normal` orientation (`dot(N, p - P) >= 0` is kept).
/// Supports both standard-Z (`reversed_z = false`, near at `z = 0`) and reversed-Z (`reversed_z = true`, near at `z = w`).
pub fn oblique_clip_projection(
    projection: &[f32; 16],
    view: &[f32; 16],
    plane_point: [f32; 3],
    plane_normal: [f32; 3],
    reversed_z: bool,
) -> Result<[f32; 16]> {
    ensure!(projection.iter().all(|v| v.is_finite()) && view.iter().all(|v| v.is_finite()), "non-finite camera matrix");
    ensure!(plane_point.iter().all(|v| v.is_finite()), "non-finite plane point");
    let [nx, ny, nz] = normalize3_f64(plane_normal)?;
    let [px, py, pz] = [f64::from(plane_point[0]), f64::from(plane_point[1]), f64::from(plane_point[2])];
    let v = to_f64(view);
    let inv_v = invert4_checked(&v)?;
    // Transform world plane `n . x + d = 0` into view space via inverse-transpose of view matrix: C_view = V^{-T} * C_world.
    let d_world = -(nx * px + ny * py + nz * pz);
    // Column-major `inv_v`: row k of V^{-T} is column k of V^{-1}.
    let cx = inv_v[0] * nx + inv_v[1] * ny + inv_v[2] * nz + inv_v[3] * d_world;
    let cy = inv_v[4] * nx + inv_v[5] * ny + inv_v[6] * nz + inv_v[7] * d_world;
    let cz = inv_v[8] * nx + inv_v[9] * ny + inv_v[10] * nz + inv_v[11] * d_world;
    let cw = inv_v[12] * nx + inv_v[13] * ny + inv_v[14] * nz + inv_v[15] * d_world;
    let clen = (cx * cx + cy * cy + cz * cz).sqrt();
    ensure!(clen > 1e-9 && clen.is_finite(), "degenerate view-space clip plane");
    let c = [cx / clen, cy / clen, cz / clen, cw / clen];

    let mut proj = to_f64(projection);
    let inv_proj = invert4_checked(&proj)?;
    let far_clip_z = if reversed_z { 0.0 } else { 1.0 };

    // Choose the far-frustum corner with the largest positive dot(C, q_view) so the oblique near plane
    // keeps the exact half-space `dot(C, p_view) >= 0` with maximum depth precision.
    let mut best_q = [0.0f64; 4];
    let mut best_dot = f64::NEG_INFINITY;
    for &sx in &[-1.0f64, 1.0] {
        for &sy in &[-1.0f64, 1.0] {
            let q = [
                inv_proj[0] * sx + inv_proj[4] * sy + inv_proj[8] * far_clip_z + inv_proj[12],
                inv_proj[1] * sx + inv_proj[5] * sy + inv_proj[9] * far_clip_z + inv_proj[13],
                inv_proj[2] * sx + inv_proj[6] * sy + inv_proj[10] * far_clip_z + inv_proj[14],
                inv_proj[3] * sx + inv_proj[7] * sy + inv_proj[11] * far_clip_z + inv_proj[15],
            ];
            let d_val = c[0] * q[0] + c[1] * q[1] + c[2] * q[2] + c[3] * q[3];
            if d_val > best_dot {
                best_dot = d_val;
                best_q = q;
            }
        }
    }
    ensure!(
        best_dot > 1e-7 && best_dot.is_finite(),
        "oblique clip plane excludes the entire view frustum (max dot(C, q) = {best_dot})"
    );
    let row3_dot_q = proj[3] * best_q[0] + proj[7] * best_q[1] + proj[11] * best_q[2] + proj[15] * best_q[3];
    ensure!(row3_dot_q > 1e-7, "degenerate projection W at opposite frustum corner");
    let scale = row3_dot_q / best_dot;
    if reversed_z {
        // Reversed-Z: near plane is z_clip = w_clip (w - z = 0) -> row2 = row3 - scale * C.
        // At q (far): (row3 - scale * C) . q = row3.q - row3.q = 0 (z = 0 far plane);
        // on plane (C.p = 0): z = w (near plane); kept side (C.p > 0): z < w (in [0, w]).
        proj[2] = proj[3] - c[0] * scale;
        proj[6] = proj[7] - c[1] * scale;
        proj[10] = proj[11] - c[2] * scale;
        proj[14] = proj[15] - c[3] * scale;
    } else {
        // Standard-Z: near plane is z_clip = 0 -> row2 = scale * C.
        // At q (far): z = row3.q = w (z/w = 1); on plane (C.p = 0): z = 0; kept side (C.p > 0): z > 0.
        proj[2] = c[0] * scale;
        proj[6] = c[1] * scale;
        proj[10] = c[2] * scale;
        proj[14] = c[3] * scale;
    }
    Ok(to_f32(&proj))
}

/// Computes a ready-to-render `(view, projection)` pair for a planar reflection camera that
/// preserves right-handed CCW triangle winding (`det(view) = +1`) for standard back-face-culled
/// pipelines (including [`crate::Renderer::render_camera_to_texture`]) and applies Lengyel's
/// oblique near-plane clip along the mirror plane.
///
/// The rendered texture is horizontally mirrored (`u' = 1 - u`) relative to the main camera so
/// sampling it on the mirror surface reflects left/right naturally without flipping cull modes.
pub fn planar_reflection_camera(
    view: &[f32; 16],
    projection: &[f32; 16],
    plane_point: [f32; 3],
    plane_normal: [f32; 3],
    reversed_z: bool,
) -> Result<([f32; 16], [f32; 16])> {
    let raw_refl = planar_reflection_view(view, plane_point, plane_normal)?;
    // Multiply on the left by diag(-1, 1, 1, 1) (flips view X): turns improper reflection (det = -1)
    // into a proper right-handed camera (det = +1) so CCW winding & back-face culling stay exact!
    let mut proper_view = raw_refl;
    proper_view[0] = -proper_view[0];
    proper_view[4] = -proper_view[4];
    proper_view[8] = -proper_view[8];
    proper_view[12] = -proper_view[12];
    let obl_proj = oblique_clip_projection(projection, &proper_view, plane_point, plane_normal, reversed_z)?;
    Ok((proper_view, obl_proj))
}

/// Pixel-space rectangle for one split-screen or picture-in-picture viewport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewportRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl ViewportRect {
    pub fn aspect_ratio(self) -> f32 {
        self.width.max(1) as f32 / self.height.max(1) as f32
    }
}

/// Split-screen viewport layout partitioner for 1..N independent cameras.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitScreenLayout {
    pub output_size: (u32, u32),
    pub divider_px: u32,
    pub viewports: Vec<ViewportRect>,
}

impl SplitScreenLayout {
    pub fn horizontal_2p(width: u32, height: u32, divider_px: u32) -> Result<Self> {
        ensure!(width >= 4 && height >= 4 && divider_px < height / 2, "invalid 2P horizontal split size");
        let top_h = (height - divider_px) / 2;
        let bot_y = top_h + divider_px;
        let bot_h = height - bot_y;
        Ok(Self {
            output_size: (width, height),
            divider_px,
            viewports: vec![
                ViewportRect { x: 0, y: 0, width, height: top_h },
                ViewportRect { x: 0, y: bot_y, width, height: bot_h },
            ],
        })
    }

    pub fn vertical_2p(width: u32, height: u32, divider_px: u32) -> Result<Self> {
        ensure!(width >= 4 && height >= 4 && divider_px < width / 2, "invalid 2P vertical split size");
        let left_w = (width - divider_px) / 2;
        let right_x = left_w + divider_px;
        let right_w = width - right_x;
        Ok(Self {
            output_size: (width, height),
            divider_px,
            viewports: vec![
                ViewportRect { x: 0, y: 0, width: left_w, height },
                ViewportRect { x: right_x, y: 0, width: right_w, height },
            ],
        })
    }

    pub fn quad_4p(width: u32, height: u32, divider_px: u32) -> Result<Self> {
        ensure!(
            width >= 4 && height >= 4 && divider_px < width / 2 && divider_px < height / 2,
            "invalid 4P quad split size"
        );
        let lw = (width - divider_px) / 2;
        let rx = lw + divider_px;
        let rw = width - rx;
        let th = (height - divider_px) / 2;
        let by = th + divider_px;
        let bh = height - by;
        Ok(Self {
            output_size: (width, height),
            divider_px,
            viewports: vec![
                ViewportRect { x: 0, y: 0, width: lw, height: th },
                ViewportRect { x: rx, y: 0, width: rw, height: th },
                ViewportRect { x: 0, y: by, width: lw, height: bh },
                ViewportRect { x: rx, y: by, width: rw, height: bh },
            ],
        })
    }

    pub fn picture_in_picture(width: u32, height: u32, inset: ViewportRect) -> Result<Self> {
        ensure!(width >= 4 && height >= 4, "invalid output size");
        ensure!(
            inset.width >= 1
                && inset.height >= 1
                && inset.x.saturating_add(inset.width) <= width
                && inset.y.saturating_add(inset.height) <= height,
            "PiP inset rect {inset:?} exceeds output {width}x{height}"
        );
        Ok(Self {
            output_size: (width, height),
            divider_px: 0,
            viewports: vec![ViewportRect { x: 0, y: 0, width, height }, inset],
        })
    }
}

/// GPU compositor blitting N independently rendered camera views into one split-screen output surface.
pub struct SplitScreenCompositor {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
}

impl SplitScreenCompositor {
    pub fn new(device: &wgpu::Device, output_format: wgpu::TextureFormat, transfer: ColorTransfer) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("split_screen_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(32),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("split_screen_shader"),
            source: wgpu::ShaderSource::Wgsl(SPLIT_SCREEN_WGSL.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("split_screen_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("split_screen_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("split_screen_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let encode_flag = match transfer {
            ColorTransfer::Passthrough => 0.0f32,
            ColorTransfer::LinearToSrgb => 1.0f32,
        };
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("split_screen_uv"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        uniform
            .slice(..)
            .get_mapped_range_mut()
            .unwrap()
            .copy_from_slice(bytemuck::cast_slice(&[0.0f32, 0.0, 1.0, 1.0, encode_flag, 0.0, 0.0, 0.0]));
        uniform.unmap();
        Self { pipeline, bgl, sampler, uniform }
    }

    /// Validates attachment dimensions, every viewport rect, and source/output non-aliasing BEFORE
    /// encoding the pass, then composites each view into `output` with hardware viewport + scissor isolation.
    pub fn composite(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        layout: &SplitScreenLayout,
        views: &[&wgpu::TextureView],
        divider_color: wgpu::Color,
    ) -> Result<()> {
        ensure!(
            views.len() == layout.viewports.len(),
            "viewport count {} does not match source view count {}",
            layout.viewports.len(),
            views.len()
        );
        let out_tex = output.texture();
        let out_extent = out_tex.size();
        let (out_w, out_h) = layout.output_size;
        ensure!(
            out_w <= out_extent.width && out_h <= out_extent.height,
            "layout output_size {out_w}x{out_h} exceeds target attachment {}x{}",
            out_extent.width,
            out_extent.height
        );
        for (&vp, &src_view) in layout.viewports.iter().zip(views.iter()) {
            ensure!(
                vp.width >= 1
                    && vp.height >= 1
                    && vp.x.saturating_add(vp.width) <= out_w
                    && vp.y.saturating_add(vp.height) <= out_h,
                "viewport {vp:?} exceeds output {out_w}x{out_h}"
            );
            ensure!(
                src_view.texture() != out_tex,
                "split-screen source texture cannot alias the destination attachment"
            );
        }

        let groups: Vec<wgpu::BindGroup> = views
            .iter()
            .map(|&src_view| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("split_screen_bg"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(src_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    ],
                })
            })
            .collect();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("split_screen_composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(divider_color), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        for (&vp, bg) in layout.viewports.iter().zip(groups.iter()) {
            pass.set_viewport(vp.x as f32, vp.y as f32, vp.width as f32, vp.height as f32, 0.0, 1.0);
            pass.set_scissor_rect(vp.x, vp.y, vp.width, vp.height);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transform_pt(m: &[f32; 16], p: [f32; 3]) -> [f32; 4] {
        [
            m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
            m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
            m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
            m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15],
        ]
    }

    fn det3(m: &[f32; 16]) -> f32 {
        m[0] * (m[5] * m[10] - m[6] * m[9])
            - m[4] * (m[1] * m[10] - m[2] * m[9])
            + m[8] * (m[1] * m[6] - m[2] * m[5])
    }

    #[test]
    fn planar_reflection_proper_winding_and_oblique_clip_standard_and_reversed_z() {
        let id = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let refl_v = planar_reflection_view(&id, [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]).unwrap();
        let p_refl = transform_pt(&refl_v, [1.0, 3.0, -5.0]);
        assert!((p_refl[0] - 1.0).abs() < 1e-5 && (p_refl[1] - -7.0).abs() < 1e-5 && (p_refl[2] - -5.0).abs() < 1e-5);

        let (near, far) = (0.5f32, 100.0f32);
        let proj_std = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, far / (near - far), -1.0,
            0.0, 0.0, (far * near) / (near - far), 0.0,
        ];
        // Standard-Z oblique clip plane at z = -5 keeping z <= -5 (normal (0, 0, -1)).
        let obl_std = oblique_clip_projection(&proj_std, &id, [0.0, 0.0, -5.0], [0.0, 0.0, -1.0], false).unwrap();
        let on_plane = transform_pt(&obl_std, [0.0, 0.0, -5.0]);
        let behind = transform_pt(&obl_std, [0.0, 0.0, -2.0]);
        let in_front = transform_pt(&obl_std, [0.0, 0.0, -20.0]);
        assert!((on_plane[2] / on_plane[3]).abs() < 1e-4, "plane point must lie on z=0: {on_plane:?}");
        assert!(behind[2] / behind[3] < -0.05, "behind plane must clip (<0): {behind:?}");
        assert!((0.0..1.0).contains(&(in_front[2] / in_front[3])));

        // Reversed-Z oblique clip (near = 1, far = 0).
        let proj_rev = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, near / (far - near), -1.0,
            0.0, 0.0, (far * near) / (far - near), 0.0,
        ];
        let obl_rev = oblique_clip_projection(&proj_rev, &id, [0.0, 0.0, -5.0], [0.0, 0.0, -1.0], true).unwrap();
        let on_p_rev = transform_pt(&obl_rev, [0.0, 0.0, -5.0]);
        let behind_rev = transform_pt(&obl_rev, [0.0, 0.0, -2.0]);
        let front_rev = transform_pt(&obl_rev, [0.0, 0.0, -20.0]);
        assert!((on_p_rev[2] / on_p_rev[3] - 1.0).abs() < 1e-4, "rev-Z plane point must lie on near z=1: {on_p_rev:?}");
        assert!(behind_rev[2] / behind_rev[3] > 1.05, "rev-Z behind plane must clip (>1): {behind_rev:?}");
        assert!((0.0..1.0).contains(&(front_rev[2] / front_rev[3])));

        // Proper planar_reflection_camera keeps det(view) == +1 so CCW pipelines render front faces cleanly!
        let (cam_v, _cam_p) = planar_reflection_camera(&id, &proj_std, [0.0, -2.0, 0.0], [0.0, 1.0, 0.0], false).unwrap();
        assert!((det3(&cam_v) - 1.0).abs() < 1e-5, "planar_reflection_camera must preserve det=+1 winding");
    }

    #[test]
    fn gpu_offscreen_targets_and_4p_split_screen_compositor_witness() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU adapter required");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        let colors = [
            wgpu::Color { r: 1.0, g: 0.0, b: 0.0, a: 1.0 },
            wgpu::Color { r: 0.5, g: 0.5, b: 0.5, a: 1.0 }, // Mid-gray tests encoding fidelity!
            wgpu::Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 },
            wgpu::Color { r: 1.0, g: 1.0, b: 0.0, a: 1.0 },
        ];
        let targets: Vec<OffscreenRenderTarget> = (0..4)
            .map(|_| OffscreenRenderTarget::new(&device, 16, 16, fmt).unwrap())
            .collect();
        let out = OffscreenRenderTarget::new(&device, 66, 66, fmt).unwrap();
        let layout = SplitScreenLayout::quad_4p(66, 66, 2).unwrap();
        let comp = SplitScreenCompositor::new(&device, fmt, ColorTransfer::Passthrough);

        // Source/output aliasing must fail BEFORE encoding any pass!
        {
            let mut bad_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let bad_views = [out.render_view(), targets[1].render_view(), targets[2].render_view(), targets[3].render_view()];
            assert!(comp.composite(&device, &mut bad_enc, out.render_view(), &layout, &bad_views, wgpu::Color::BLACK).is_err());
        }

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for (t, &c) in targets.iter().zip(colors.iter()) {
            let _rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fill_player"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: t.render_view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(c), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
        }
        let views: Vec<&wgpu::TextureView> = targets.iter().map(OffscreenRenderTarget::render_view).collect();
        comp.composite(&device, &mut enc, out.render_view(), &layout, &views, wgpu::Color { r: 0.2, g: 0.2, b: 0.2, a: 1.0 }).unwrap();

        let bpr = (66u32 * 4).next_multiple_of(256);
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("split_rb"),
            size: u64::from(bpr * 66),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: out.texture(), mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &rb,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(66) },
            },
            wgpu::Extent3d { width: 66, height: 66, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let bytes = slice.get_mapped_range().unwrap();
        let px = |x: u32, y: u32| {
            let o = (y * bpr + x * 4) as usize;
            [bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]
        };
        assert_eq!(px(10, 10), [255, 0, 0, 255], "P1 red");
        assert!((i16::from(px(50, 10)[0]) - 128).abs() <= 2, "P2 mid-gray must stay ~128 under Passthrough: {:?}", px(50, 10));
        assert_eq!(px(10, 50), [0, 0, 255, 255], "P3 blue");
        assert_eq!(px(50, 50), [255, 255, 0, 255], "P4 yellow");
        assert!((i16::from(px(32, 32)[0]) - 51).abs() <= 2, "divider strip must match clear color ~51: {:?}", px(32, 32));
        drop(bytes);
        rb.unmap();
    }
}
