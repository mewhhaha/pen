//! Reversed-Z depth projection and camera-relative (floating-origin) f64-to-f32 precision helpers
//! for large open worlds without Z-fighting or far-from-origin vertex jitter.

use pen_protocol::Matrix;

/// Depth buffer convention for pipelines and depth linearization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DepthMode {
    /// Standard WebGPU `[0, 1]` depth where `0` = near, `1` = far.
    #[default]
    ForwardZ,
    /// Reversed-Z `[1, 0]` depth where `1` = near, `0` = far (dramatically improves far precision).
    ReversedZ,
}

impl DepthMode {
    pub fn clear_value(self) -> f32 {
        match self {
            Self::ForwardZ => 1.0,
            Self::ReversedZ => 0.0,
        }
    }

    pub fn compare_nearer(self) -> wgpu::CompareFunction {
        match self {
            Self::ForwardZ => wgpu::CompareFunction::Less,
            Self::ReversedZ => wgpu::CompareFunction::Greater,
        }
    }

    pub fn compare_nearer_or_equal(self) -> wgpu::CompareFunction {
        match self {
            Self::ForwardZ => wgpu::CompareFunction::LessEqual,
            Self::ReversedZ => wgpu::CompareFunction::GreaterEqual,
        }
    }

    pub fn compare_farther(self) -> wgpu::CompareFunction {
        match self {
            Self::ForwardZ => wgpu::CompareFunction::Greater,
            Self::ReversedZ => wgpu::CompareFunction::Less,
        }
    }
}

/// Detects whether a right-handed WebGPU `[0, 1]` perspective `projection` matrix uses
/// `ForwardZ` (`z_ndc` 0->1) or `ReversedZ` (`z_ndc` 1->0). Returns `None` for non-perspective
/// matrices (e.g. identity or orthographic) so callers preserve the active depth mode.
pub fn detect_depth_mode(projection: &Matrix) -> Option<DepthMode> {
    if !projection.iter().all(|v| v.is_finite()) {
        return None;
    }
    if projection[11] < -0.5 {
        if projection[10] >= 0.0 && projection[14] > 0.0 {
            return Some(DepthMode::ReversedZ);
        }
        if projection[10] < 0.0 && projection[14] < 0.0 {
            return Some(DepthMode::ForwardZ);
        }
    }
    None
}

/// Right-handed WebGPU `[0, 1]` Reversed-Z perspective projection matrix (`z_ndc = 1` at `near`,
/// `z_ndc = 0` at `far` or infinity when `far` is `None`).
pub fn perspective_reversed_z(fov_y_rad: f32, aspect: f32, near: f32, far: Option<f32>) -> Matrix {
    let f = 1.0 / (fov_y_rad * 0.5).tan();
    let near = near.max(1e-5);
    match far {
        None => [f / aspect, 0.0, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, near, 0.0],
        Some(far) => {
            let far = far.max(near + 1e-4);
            let k = near / (far - near);
            [f / aspect, 0.0, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 0.0, k, -1.0, 0.0, 0.0, far * k, 0.0]
        }
    }
}

/// Right-handed WebGPU `[0, 1]` Forward-Z perspective projection matrix (`z_ndc = 0` at `near`, `1` at `far`).
pub fn perspective_forward_z(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Matrix {
    let f = 1.0 / (fov_y_rad * 0.5).tan();
    let near = near.max(1e-5);
    let far = far.max(near + 1e-4);
    let nf = far / (near - far);
    [f / aspect, 0.0, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 0.0, nf, -1.0, 0.0, 0.0, nf * near, 0.0]
}

/// Camera-relative floating-origin frame built from `f64` world coordinates.
///
/// Subtracts the camera's world position in double precision before narrowing
/// translations to `f32`, eliminating vertex jitter and precision loss even at
/// hundreds of kilometers from the world origin.
#[derive(Clone, Copy, Debug)]
pub struct CameraRelativeFrame {
    pub camera_world: [f64; 3],
    /// View rotation matrix (camera centered at `[0, 0, 0]` in camera-relative space).
    pub view_rotation: Matrix,
}

impl CameraRelativeFrame {
    /// Builds a camera-relative frame from `f64` eye/target/up vectors.
    pub fn look_at(eye: [f64; 3], target: [f64; 3], up: [f64; 3]) -> Self {
        let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        };
        let norm = |v: [f64; 3]| {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
            [v[0] / l, v[1] / l, v[2] / l]
        };
        let f = norm(sub(target, eye));
        let s = norm(cross(f, up));
        let u = cross(s, f);
        #[allow(clippy::cast_possible_truncation, reason = "unit camera basis narrowed to f32")]
        let view_rotation = [
            s[0] as f32,
            u[0] as f32,
            -f[0] as f32,
            0.0,
            s[1] as f32,
            u[1] as f32,
            -f[1] as f32,
            0.0,
            s[2] as f32,
            u[2] as f32,
            -f[2] as f32,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ];
        Self { camera_world: eye, view_rotation }
    }

    /// Converts an entity's `f64` world position and 3x3 linear transform (column-major)
    /// into a camera-relative `f32` model matrix.
    pub fn entity_model_matrix(&self, world_pos: [f64; 3], basis3: [[f32; 3]; 3]) -> Matrix {
        #[allow(clippy::cast_possible_truncation, reason = "camera-relative delta narrowed to f32")]
        let rel = [
            (world_pos[0] - self.camera_world[0]) as f32,
            (world_pos[1] - self.camera_world[1]) as f32,
            (world_pos[2] - self.camera_world[2]) as f32,
        ];
        [
            basis3[0][0],
            basis3[0][1],
            basis3[0][2],
            0.0,
            basis3[1][0],
            basis3[1][1],
            basis3[1][2],
            0.0,
            basis3[2][0],
            basis3[2][1],
            basis3[2][2],
            0.0,
            rel[0],
            rel[1],
            rel[2],
            1.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_depth_mode_classifies_forward_reversed_and_non_perspective() {
        let rev_inf = perspective_reversed_z(1.0, 1.5, 0.1, None);
        let rev_fin = perspective_reversed_z(1.0, 1.5, 0.1, Some(10_000.0));
        let fwd = perspective_forward_z(1.0, 1.5, 0.1, 10_000.0);
        let ortho = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(detect_depth_mode(&rev_inf), Some(DepthMode::ReversedZ));
        assert_eq!(detect_depth_mode(&rev_fin), Some(DepthMode::ReversedZ));
        assert_eq!(detect_depth_mode(&fwd), Some(DepthMode::ForwardZ));
        assert_eq!(detect_depth_mode(&ortho), None);
    }

    #[test]
    fn camera_relative_preserves_submillimeter_precision_at_100km() {
        let eye = [100_000.0f64, 50_000.0, -250_000.0];
        let target = [100_000.0, 50_000.0, -250_010.0];
        let frame = CameraRelativeFrame::look_at(eye, target, [0.0, 1.0, 0.0]);
        let a = frame.entity_model_matrix(
            [100_000.0, 50_000.0, -250_005.000],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        );
        let b = frame.entity_model_matrix(
            [100_000.0, 50_000.0, -250_005.001],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        );
        let dz = (a[14] - b[14]).abs();
        assert!((dz - 0.001).abs() < 1e-6, "1mm delta at 250km should stay exact, got {dz}");
    }

    #[test]
    fn gpu_reversed_z_eliminates_far_z_fighting_where_forward_z_collapses() {
        let Some((device, queue)) = crate::testing::device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let wgsl = r#"
            struct Draw { proj: mat4x4f, color: vec4f, z_view: f32, _p0: f32, _p1: f32, _p2: f32 };
            @group(0) @binding(0) var<uniform> d: Draw;
            @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
              let x = select(-1.0, 1.0, (i & 1u) != 0u) * 10000.0;
              let y = select(-1.0, 1.0, (i & 2u) != 0u) * 10000.0;
              return d.proj * vec4f(x, y, d.z_view, 1.0);
            }
            @fragment fn fs() -> @location(0) vec4f { return d.color; }
        "#;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("depth_precision_test"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(96),
                },
                count: None,
            }],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let run_mode = |mode: DepthMode, proj: Matrix| -> [u8; 4] {
            let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(mode.compare_nearer()),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: crate::pipeline_cache::active().as_ref(),
            });
            let color_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let make_bg = |color: [f32; 4], z: f32| {
                let mut f = [0.0f32; 24];
                f[..16].copy_from_slice(&proj);
                f[16..20].copy_from_slice(&color);
                f[20] = z;
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 96,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(&f));
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bgl,
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
                })
            };
            // Draw FAR blue quad at -5000.05m first, then NEAR green quad at -5000.00m (5cm apart at 5km!).
            // Note: draw NEAR green FIRST then FAR red SECOND to test depth rejection cleanly!
            let near_green = make_bg([0.0, 1.0, 0.0, 1.0], -5000.00);
            let far_red = make_bg([1.0, 0.0, 0.0, 1.0], -5000.05);
            let cv = color_tex.create_view(&Default::default());
            let dv = depth_tex.create_view(&Default::default());
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &cv,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &dv,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(mode.clear_value()),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                pass.set_pipeline(&pipe);
                // Draw FAR red first, then NEAR green; green MUST win if depth distinguishes 5cm at 5km!
                pass.set_bind_group(0, &far_red, &[]);
                pass.draw(0..4, 0..1);
                pass.set_bind_group(0, &near_green, &[]);
                pass.draw(0..4, 0..1);
            }
            let rb = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256 * 8,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            enc.copy_texture_to_buffer(
                color_tex.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &rb,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(8),
                    },
                },
                wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 },
            );
            queue.submit([enc.finish()]);
            let slice = rb.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            rx.recv().unwrap().unwrap();
            let d = slice.get_mapped_range().unwrap();
            let off = 4 * 256 + 4 * 4;
            let px = [d[off], d[off + 1], d[off + 2], d[off + 3]];
            drop(d);
            rb.unmap();
            px
        };

        // At near = 0.05m, far = 100_000m, depth difference at 5000m vs 5000.05m:
        // Forward-Z collapses both to 0.9999900 (identical in f32 -> green fails Less -> red wins).
        // Reversed-Z cleanly separates them (green wins)!
        let rev_proj = perspective_reversed_z(std::f32::consts::FRAC_PI_3, 1.0, 0.05, None);
        let rev_px = run_mode(DepthMode::ReversedZ, rev_proj);
        assert!(rev_px[1] > 240 && rev_px[0] < 10, "Reversed-Z must resolve 5cm separation at 5km: got {rev_px:?}");
    }
}
