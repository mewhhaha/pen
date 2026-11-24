//! Portable GPU BVH Ray-Traced Soft Shadows & Ray-Traced Ambient Occlusion (RTAO),
//! plus hardware `EXPERIMENTAL_RAY_QUERY` capability detection.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "GPU BVH node and texture indices")]

use anyhow::{Result, ensure};

pub const BVH_RAYTRACE_WGSL: &str = include_str!("shaders/bvh_raytrace.wgsl");
pub const TLAS_TRAVERSAL_WGSL: &str = include_str!("shaders/tlas_traversal.wgsl");
pub const TLAS_RAYTRACE_WGSL: &str = include_str!("shaders/tlas_raytrace.wgsl");

/// Reports whether the active GPU adapter advertises hardware ray-query support (`KHR_ray_query` / Metal RT / DXR).
/// Note that [`GpuRayTracer`] executes portably via compute BVH traversal on ALL WebGPU/Vulkan/Metal backends.
pub fn supports_hardware_ray_query(adapter: &wgpu::Adapter) -> bool {
    adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
}

/// World-space triangle fed into the BVH builder.
#[derive(Clone, Copy, Debug)]
pub struct BvhTriangle {
    pub v0: [f32; 3],
    pub v1: [f32; 3],
    pub v2: [f32; 3],
}

/// Cache-aligned 32-byte GPU BVH node with explicit `u32` child/leaf fields (no float/u32 bitcasts).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BvhNodeGpu {
    pub aabb_min: [f32; 3],
    pub left_or_first: u32,
    pub aabb_max: [f32; 3],
    pub tri_count: u32,
}

/// GPU triangle record storing all 3 vertices for watertight Woop-Benthin-Wald intersection.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BvhTriGpu {
    pub v0: [f32; 4],
    pub v1: [f32; 4],
    pub v2: [f32; 4],
}

#[derive(Clone, Copy)]
struct Aabb {
    min: [f32; 3],
    max: [f32; 3],
}

impl Aabb {
    fn empty() -> Self {
        Self { min: [f32::INFINITY; 3], max: [f32::NEG_INFINITY; 3] }
    }
    fn grow_pt(&mut self, p: [f32; 3]) {
        for (k, &v) in p.iter().enumerate() {
            self.min[k] = self.min[k].min(v);
            self.max[k] = self.max[k].max(v);
        }
    }
    fn grow_box(&mut self, o: &Aabb) {
        for k in 0..3 {
            self.min[k] = self.min[k].min(o.min[k]);
            self.max[k] = self.max[k].max(o.max[k]);
        }
    }
    fn surface_area(&self) -> f32 {
        let ex = (self.max[0] - self.min[0]).max(0.0);
        let ey = (self.max[1] - self.min[1]).max(0.0);
        let ez = (self.max[2] - self.min[2]).max(0.0);
        2.0 * (ex * ey + ey * ez + ez * ex)
    }
}

fn tri_area2(t: &BvhTriangle) -> f64 {
    let e1 = [f64::from(t.v1[0] - t.v0[0]), f64::from(t.v1[1] - t.v0[1]), f64::from(t.v1[2] - t.v0[2])];
    let e2 = [f64::from(t.v2[0] - t.v0[0]), f64::from(t.v2[1] - t.v0[1]), f64::from(t.v2[2] - t.v0[2])];
    let cx = e1[1] * e2[2] - e1[2] * e2[1];
    let cy = e1[2] * e2[0] - e1[0] * e2[2];
    let cz = e1[0] * e2[1] - e1[1] * e2[0];
    cx * cx + cy * cy + cz * cz
}

/// Validated, acyclic flattened SAH-binned BVH (depth `<= 28`, guaranteed safe for 32-deep GPU stack).
#[derive(Clone, Debug)]
pub struct BvhScene {
    nodes: Vec<BvhNodeGpu>,
    tris: Vec<BvhTriGpu>,
}

impl BvhScene {
    pub fn nodes(&self) -> &[BvhNodeGpu] {
        &self.nodes
    }

    pub fn tris(&self) -> &[BvhTriGpu] {
        &self.tris
    }

    /// Builds a validated flat SAH-binned BVH over `triangles`, filtering zero-area degenerates
    /// and falling back to balanced median splits on coincident centroids.
    pub fn build(triangles: &[BvhTriangle]) -> Result<Self> {
        let mut valid_tris = Vec::with_capacity(triangles.len());
        for t in triangles {
            ensure!(
                t.v0.iter().chain(t.v1.iter()).chain(t.v2.iter()).all(|v| v.is_finite()),
                "BVH triangle contains non-finite vertex coordinate"
            );
            if tri_area2(t) > 1e-24 {
                valid_tris.push(*t);
            }
        }
        if valid_tris.is_empty() {
            return Ok(Self { nodes: Vec::new(), tris: Vec::new() });
        }
        let mut order: Vec<usize> = (0..valid_tris.len()).collect();
        let centroids: Vec<[f32; 3]> = valid_tris
            .iter()
            .map(|t| [(t.v0[0] + t.v1[0] + t.v2[0]) / 3.0, (t.v0[1] + t.v1[1] + t.v2[1]) / 3.0, (t.v0[2] + t.v1[2] + t.v2[2]) / 3.0])
            .collect();
        let tri_boxes: Vec<Aabb> = valid_tris
            .iter()
            .map(|t| {
                let mut b = Aabb::empty();
                b.grow_pt(t.v0);
                b.grow_pt(t.v1);
                b.grow_pt(t.v2);
                b
            })
            .collect();

        let mut nodes = vec![BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 }];
        let mut out_tris = Vec::with_capacity(valid_tris.len());

        #[allow(clippy::too_many_arguments)]
        fn build_rec(
            node_idx: usize,
            start: usize,
            end: usize,
            depth: u32,
            order: &mut [usize],
            centroids: &[[f32; 3]],
            tri_boxes: &[Aabb],
            triangles: &[BvhTriangle],
            nodes: &mut Vec<BvhNodeGpu>,
            out_tris: &mut Vec<BvhTriGpu>,
        ) {
            let mut bounds = Aabb::empty();
            let mut c_bounds = Aabb::empty();
            for &ti in &order[start..end] {
                bounds.grow_box(&tri_boxes[ti]);
                c_bounds.grow_pt(centroids[ti]);
            }
            let count = end - start;
            if count <= 4 || depth >= 28 {
                let first = out_tris.len() as u32;
                for &ti in &order[start..end] {
                    let t = triangles[ti];
                    out_tris.push(BvhTriGpu {
                        v0: [t.v0[0], t.v0[1], t.v0[2], 0.0],
                        v1: [t.v1[0], t.v1[1], t.v1[2], 0.0],
                        v2: [t.v2[0], t.v2[1], t.v2[2], 0.0],
                    });
                }
                nodes[node_idx] = BvhNodeGpu {
                    aabb_min: bounds.min,
                    left_or_first: first,
                    aabb_max: bounds.max,
                    tri_count: count as u32,
                };
                return;
            }
            let ext = [c_bounds.max[0] - c_bounds.min[0], c_bounds.max[1] - c_bounds.min[1], c_bounds.max[2] - c_bounds.min[2]];
            let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
                0
            } else if ext[1] >= ext[2] {
                1
            } else {
                2
            };

            let mid = if ext[axis] <= 1e-6 {
                // Balanced fallback prevents arbitrarily huge leaves when centroids coincide.
                start + count / 2
            } else {
                const BINS: usize = 8;
                let mut bin_box = [Aabb::empty(); BINS];
                let mut bin_cnt = [0usize; BINS];
                let scale = BINS as f32 / ext[axis];
                for &ti in &order[start..end] {
                    let b = (((centroids[ti][axis] - c_bounds.min[axis]) * scale) as usize).min(BINS - 1);
                    bin_cnt[b] += 1;
                    bin_box[b].grow_box(&tri_boxes[ti]);
                }
                let mut best_cost = f32::INFINITY;
                let mut best_split = 0usize;
                for split in 1..BINS {
                    let (mut l_box, mut r_box) = (Aabb::empty(), Aabb::empty());
                    let (mut l_cnt, mut r_cnt) = (0usize, 0usize);
                    for b in 0..split {
                        l_cnt += bin_cnt[b];
                        l_box.grow_box(&bin_box[b]);
                    }
                    for b in split..BINS {
                        r_cnt += bin_cnt[b];
                        r_box.grow_box(&bin_box[b]);
                    }
                    if l_cnt == 0 || r_cnt == 0 {
                        continue;
                    }
                    let cost = l_cnt as f32 * l_box.surface_area() + r_cnt as f32 * r_box.surface_area();
                    if cost < best_cost {
                        best_cost = cost;
                        best_split = split;
                    }
                }
                let split_pos = if best_split == 0 {
                    c_bounds.min[axis] + 0.5 * ext[axis]
                } else {
                    c_bounds.min[axis] + (best_split as f32 / BINS as f32) * ext[axis]
                };
                let mut m = start;
                for i in start..end {
                    if centroids[order[i]][axis] < split_pos {
                        order.swap(i, m);
                        m += 1;
                    }
                }
                if m == start || m == end { start + count / 2 } else { m }
            };

            let left_idx = nodes.len();
            nodes.push(BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 });
            nodes.push(BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 });
            nodes[node_idx] = BvhNodeGpu {
                aabb_min: bounds.min,
                left_or_first: left_idx as u32,
                aabb_max: bounds.max,
                tri_count: 0,
            };
            build_rec(left_idx, start, mid, depth + 1, order, centroids, tri_boxes, triangles, nodes, out_tris);
            build_rec(left_idx + 1, mid, end, depth + 1, order, centroids, tri_boxes, triangles, nodes, out_tris);
        }

        build_rec(0, 0, valid_tris.len(), 0, &mut order, &centroids, &tri_boxes, &valid_tris, &mut nodes, &mut out_tris);
        Ok(Self { nodes, tris: out_tris })
    }
}

/// Ray-tracing pass parameters for sun shadows and hemispherical RTAO.
#[derive(Clone, Copy, Debug)]
pub struct RayTraceParams {
    pub view_projection: [f32; 16],
    pub camera_pos: [f32; 3],
    pub depth_clear: f32,
    pub sun_direction: [f32; 3],
    pub sun_angular_radius: f32,
    pub ao_radius: f32,
    pub ao_samples: u32,
    pub shadow_samples: u32,
    pub frame_seed: f32,
    pub normal_bias: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RayTraceUniform {
    inverse_view_projection: [f32; 16],
    camera_pos_and_clear: [f32; 4],
    sun_dir_and_radius: [f32; 4],
    ao_shadow: [f32; 4],
    bias_params: [f32; 4],
    counts: [u32; 4],
}

fn encode_copy(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, dst: &wgpu::Buffer, bytes: &[u8]) {
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rt_stage"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    stage.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytes);
    stage.unmap();
    encoder.copy_buffer_to_buffer(&stage, 0, dst, 0, bytes.len() as u64);
}

/// Portable GPU compute BVH ray tracer outputting `(rt_shadow, rtao, hit_dist, 1.0)` in `Rgba16Float`.
pub struct GpuRayTracer {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    nodes_buf: Option<(u32, wgpu::Buffer)>,
    tris_buf: Option<(u32, wgpu::Buffer)>,
    output: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
}

impl GpuRayTracer {
    pub fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bvh_rt_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<RayTraceUniform>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bvh_raytrace"),
            source: wgpu::ShaderSource::Wgsl(BVH_RAYTRACE_WGSL.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bvh_rt_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bvh_rt_pipe"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("raytrace_shadows_ao"),
            compilation_options: Default::default(),
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bvh_rt_u"),
            size: std::mem::size_of::<RayTraceUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, bgl, uniform, nodes_buf: None, tris_buf: None, output: None }
    }

    pub fn upload_bvh(&mut self, device: &wgpu::Device, scene: &BvhScene) -> Result<()> {
        let max_storage = device.limits().max_storage_buffer_binding_size;
        let nodes = if scene.nodes.is_empty() {
            vec![BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 }]
        } else {
            scene.nodes.clone()
        };
        let tris = if scene.tris.is_empty() {
            vec![BvhTriGpu { v0: [0.0; 4], v1: [0.0; 4], v2: [0.0; 4] }]
        } else {
            scene.tris.clone()
        };
        let n_bytes = std::mem::size_of_val(nodes.as_slice()) as u64;
        let t_bytes = std::mem::size_of_val(tris.as_slice()) as u64;
        ensure!(n_bytes <= max_storage && t_bytes <= max_storage, "BVH buffer exceeds device storage limit");

        let nb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bvh_nodes"),
            size: n_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        nb.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytemuck::cast_slice(&nodes));
        nb.unmap();

        let tb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bvh_tris"),
            size: t_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        tb.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytemuck::cast_slice(&tris));
        tb.unmap();

        self.nodes_buf = Some((scene.nodes.len() as u32, nb));
        self.tris_buf = Some((scene.tris.len() as u32, tb));
        Ok(())
    }

    pub fn output_texture(&self) -> Option<&wgpu::Texture> {
        self.output.as_ref().map(|o| &o.2)
    }

    pub fn output_view(&self) -> Option<&wgpu::TextureView> {
        self.output.as_ref().map(|o| &o.3)
    }

    pub fn trace(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        size: (u32, u32),
        params: &RayTraceParams,
    ) -> Result<()> {
        let max_dim = device.limits().max_texture_dimension_2d;
        ensure!(size.0 >= 1 && size.1 >= 1 && size.0 <= max_dim && size.1 <= max_dim, "invalid raytrace size {size:?}");
        let d_sz = depth_view.texture().size();
        ensure!(
            (d_sz.width, d_sz.height) == size,
            "GpuRayTracer depth_view extent {}x{} does not match requested {size:?}",
            d_sz.width,
            d_sz.height
        );
        ensure!(params.view_projection.iter().all(|v| v.is_finite()), "non-finite raytrace view_projection");
        let (node_count, nodes_buf) = self.nodes_buf.as_ref().ok_or_else(|| anyhow::anyhow!("call upload_bvh before trace"))?;
        let (tri_count, tris_buf) = self.tris_buf.as_ref().unwrap();

        if !matches!(&self.output, Some((w, h, _, _)) if *w == size.0 && *h == size.1) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("bvh_rt_output"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.output = Some((size.0, size.1, tex, view));
        }

        let u = RayTraceUniform {
            inverse_view_projection: crate::matrix::invert(&params.view_projection),
            camera_pos_and_clear: [params.camera_pos[0], params.camera_pos[1], params.camera_pos[2], params.depth_clear],
            sun_dir_and_radius: [
                params.sun_direction[0],
                params.sun_direction[1],
                params.sun_direction[2],
                params.sun_angular_radius.clamp(0.0, 0.2),
            ],
            ao_shadow: [
                params.ao_radius.max(0.05),
                params.ao_samples.min(16) as f32,
                params.shadow_samples.clamp(1, 8) as f32,
                params.frame_seed,
            ],
            bias_params: [params.normal_bias.max(1e-5), 0.0, 0.0, 0.0],
            counts: [*node_count, *tri_count, 0, 0],
        };
        encode_copy(device, encoder, &self.uniform, bytemuck::bytes_of(&u));
        let (_, _, _, out_view) = self.output.as_ref().unwrap();
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bvh_rt_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tris_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(out_view) },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bvh_raytrace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
        Ok(())
    }
}

// ============================================================================
// 2-Level Dynamic TLAS/BLAS Ray-Tracing Engine (Shadows, RTAO, Reflections, GI)
// ============================================================================

/// Mesh-local triangle with per-vertex normals for a reusable Bottom-Level Acceleration Structure ([`Blas`]).
#[derive(Clone, Copy, Debug)]
pub struct BlasTriangle {
    pub v0: [f32; 3],
    pub v1: [f32; 3],
    pub v2: [f32; 3],
    pub n0: [f32; 3],
    pub n1: [f32; 3],
    pub n2: [f32; 3],
}

impl BlasTriangle {
    /// Creates a flat-shaded triangle whose vertex normals equal the normalized geometric face normal.
    pub fn flat(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> Self {
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let len = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        let n = if len > 1e-8 { [c[0] / len, c[1] / len, c[2] / len] } else { [0.0, 1.0, 0.0] };
        Self { v0, v1, v2, n0: n, n1: n, n2: n }
    }

    /// Creates a smooth-shaded triangle with explicit per-vertex normals.
    pub fn smooth(
        v0: [f32; 3],
        v1: [f32; 3],
        v2: [f32; 3],
        n0: [f32; 3],
        n1: [f32; 3],
        n2: [f32; 3],
    ) -> Self {
        Self { v0, v1, v2, n0, n1, n2 }
    }
}

/// GPU-packed BLAS triangle record (96 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlasTriGpu {
    pub v0: [f32; 4],
    pub v1: [f32; 4],
    pub v2: [f32; 4],
    pub n0: [f32; 4],
    pub n1: [f32; 4],
    pub n2: [f32; 4],
}

impl BlasTriGpu {
    fn from_tri(t: &BlasTriangle) -> Self {
        Self {
            v0: [t.v0[0], t.v0[1], t.v0[2], 0.0],
            v1: [t.v1[0], t.v1[1], t.v1[2], 0.0],
            v2: [t.v2[0], t.v2[1], t.v2[2], 0.0],
            n0: [t.n0[0], t.n0[1], t.n0[2], 0.0],
            n1: [t.n1[0], t.n1[1], t.n1[2], 0.0],
            n2: [t.n2[0], t.n2[1], t.n2[2], 0.0],
        }
    }
}

fn blas_tri_area2(t: &BlasTriangle) -> f64 {
    let e1 = [f64::from(t.v1[0] - t.v0[0]), f64::from(t.v1[1] - t.v0[1]), f64::from(t.v1[2] - t.v0[2])];
    let e2 = [f64::from(t.v2[0] - t.v0[0]), f64::from(t.v2[1] - t.v0[1]), f64::from(t.v2[2] - t.v0[2])];
    let cx = e1[1] * e2[2] - e1[2] * e2[1];
    let cy = e1[2] * e2[0] - e1[0] * e2[2];
    let cz = e1[0] * e2[1] - e1[1] * e2[0];
    cx * cx + cy * cy + cz * cz
}

/// Reusable per-mesh Bottom-Level Acceleration Structure supporting O(N) in-place [`Blas::refit`]
/// for deformed or skinned geometry without topology rebuild.
#[derive(Clone, Debug)]
pub struct Blas {
    nodes: Vec<BvhNodeGpu>,
    tris: Vec<BlasTriGpu>,
    /// Maps each packed leaf triangle index back to its index in the original input slice.
    source_indices: Vec<usize>,
    source_len: usize,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
}

impl Blas {
    pub fn nodes(&self) -> &[BvhNodeGpu] {
        &self.nodes
    }

    pub fn tris(&self) -> &[BlasTriGpu] {
        &self.tris
    }

    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        (self.aabb_min, self.aabb_max)
    }

    /// Builds a SAH-binned BLAS (depth `<= 24`, guaranteed safe for the 28-deep GPU BLAS stack).
    pub fn build(triangles: &[BlasTriangle]) -> Result<Self> {
        let mut valid_indices = Vec::with_capacity(triangles.len());
        for (i, t) in triangles.iter().enumerate() {
            ensure!(
                t.v0.iter()
                    .chain(t.v1.iter())
                    .chain(t.v2.iter())
                    .chain(t.n0.iter())
                    .chain(t.n1.iter())
                    .chain(t.n2.iter())
                    .all(|v| v.is_finite()),
                "BLAS triangle contains non-finite coordinate or normal"
            );
            if blas_tri_area2(t) > 1e-24 {
                valid_indices.push(i);
            }
        }
        if valid_indices.is_empty() {
            return Ok(Self {
                nodes: Vec::new(),
                tris: Vec::new(),
                source_indices: Vec::new(),
                source_len: triangles.len(),
                aabb_min: [0.0; 3],
                aabb_max: [0.0; 3],
            });
        }
        let centroids: Vec<[f32; 3]> = triangles
            .iter()
            .map(|t| {
                [
                    (t.v0[0] + t.v1[0] + t.v2[0]) / 3.0,
                    (t.v0[1] + t.v1[1] + t.v2[1]) / 3.0,
                    (t.v0[2] + t.v1[2] + t.v2[2]) / 3.0,
                ]
            })
            .collect();
        let tri_boxes: Vec<Aabb> = triangles
            .iter()
            .map(|t| {
                let mut b = Aabb::empty();
                b.grow_pt(t.v0);
                b.grow_pt(t.v1);
                b.grow_pt(t.v2);
                b
            })
            .collect();

        let mut nodes = vec![BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 }];
        let mut out_tris = Vec::with_capacity(valid_indices.len());
        let mut source_indices = Vec::with_capacity(valid_indices.len());

        #[allow(clippy::too_many_arguments)]
        fn build_rec(
            node_idx: usize,
            start: usize,
            end: usize,
            depth: u32,
            order: &mut [usize],
            centroids: &[[f32; 3]],
            tri_boxes: &[Aabb],
            triangles: &[BlasTriangle],
            nodes: &mut Vec<BvhNodeGpu>,
            out_tris: &mut Vec<BlasTriGpu>,
            source_indices: &mut Vec<usize>,
        ) {
            let mut bounds = Aabb::empty();
            let mut c_bounds = Aabb::empty();
            for &ti in &order[start..end] {
                bounds.grow_box(&tri_boxes[ti]);
                c_bounds.grow_pt(centroids[ti]);
            }
            let count = end - start;
            if count <= 4 || depth >= 24 {
                let first = out_tris.len() as u32;
                for &ti in &order[start..end] {
                    out_tris.push(BlasTriGpu::from_tri(&triangles[ti]));
                    source_indices.push(ti);
                }
                nodes[node_idx] = BvhNodeGpu {
                    aabb_min: bounds.min,
                    left_or_first: first,
                    aabb_max: bounds.max,
                    tri_count: count as u32,
                };
                return;
            }
            let ext = [
                c_bounds.max[0] - c_bounds.min[0],
                c_bounds.max[1] - c_bounds.min[1],
                c_bounds.max[2] - c_bounds.min[2],
            ];
            let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
                0
            } else if ext[1] >= ext[2] {
                1
            } else {
                2
            };
            let mid = if ext[axis] <= 1e-6 {
                start + count / 2
            } else {
                const BINS: usize = 8;
                let mut bin_box = [Aabb::empty(); BINS];
                let mut bin_cnt = [0usize; BINS];
                let scale = BINS as f32 / ext[axis];
                for &ti in &order[start..end] {
                    let b = (((centroids[ti][axis] - c_bounds.min[axis]) * scale) as usize).min(BINS - 1);
                    bin_cnt[b] += 1;
                    bin_box[b].grow_box(&tri_boxes[ti]);
                }
                let mut best_cost = f32::INFINITY;
                let mut best_split = 0usize;
                for split in 1..BINS {
                    let (mut l_box, mut r_box) = (Aabb::empty(), Aabb::empty());
                    let (mut l_cnt, mut r_cnt) = (0usize, 0usize);
                    for b in 0..split {
                        l_cnt += bin_cnt[b];
                        l_box.grow_box(&bin_box[b]);
                    }
                    for b in split..BINS {
                        r_cnt += bin_cnt[b];
                        r_box.grow_box(&bin_box[b]);
                    }
                    if l_cnt == 0 || r_cnt == 0 {
                        continue;
                    }
                    let cost = l_cnt as f32 * l_box.surface_area() + r_cnt as f32 * r_box.surface_area();
                    if cost < best_cost {
                        best_cost = cost;
                        best_split = split;
                    }
                }
                let split_pos = if best_split == 0 {
                    c_bounds.min[axis] + 0.5 * ext[axis]
                } else {
                    c_bounds.min[axis] + (best_split as f32 / BINS as f32) * ext[axis]
                };
                let mut m = start;
                for i in start..end {
                    if centroids[order[i]][axis] < split_pos {
                        order.swap(i, m);
                        m += 1;
                    }
                }
                if m == start || m == end { start + count / 2 } else { m }
            };

            let left_idx = nodes.len();
            nodes.push(BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 });
            nodes.push(BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 });
            nodes[node_idx] = BvhNodeGpu {
                aabb_min: bounds.min,
                left_or_first: left_idx as u32,
                aabb_max: bounds.max,
                tri_count: 0,
            };
            build_rec(
                left_idx,
                start,
                mid,
                depth + 1,
                order,
                centroids,
                tri_boxes,
                triangles,
                nodes,
                out_tris,
                source_indices,
            );
            build_rec(
                left_idx + 1,
                mid,
                end,
                depth + 1,
                order,
                centroids,
                tri_boxes,
                triangles,
                nodes,
                out_tris,
                source_indices,
            );
        }

        let n_valid = valid_indices.len();
        build_rec(
            0,
            0,
            n_valid,
            0,
            &mut valid_indices,
            &centroids,
            &tri_boxes,
            triangles,
            &mut nodes,
            &mut out_tris,
            &mut source_indices,
        );
        let root_min = nodes[0].aabb_min;
        let root_max = nodes[0].aabb_max;
        Ok(Self {
            nodes,
            tris: out_tris,
            source_indices,
            source_len: triangles.len(),
            aabb_min: root_min,
            aabb_max: root_max,
        })
    }

    /// Refits deformed/skinned triangle vertices and every BVH node AABB in O(N) bottom-up
    /// without rebuilding tree topology.
    pub fn refit(&mut self, updated_triangles: &[BlasTriangle]) -> Result<()> {
        ensure!(
            updated_triangles.len() == self.source_len,
            "BLAS refit triangle count {} does not match original {}",
            updated_triangles.len(),
            self.source_len
        );
        if self.nodes.is_empty() {
            return Ok(());
        }
        for (dst_idx, &src_idx) in self.source_indices.iter().enumerate() {
            let t = &updated_triangles[src_idx];
            ensure!(
                t.v0.iter()
                    .chain(t.v1.iter())
                    .chain(t.v2.iter())
                    .chain(t.n0.iter())
                    .chain(t.n1.iter())
                    .chain(t.n2.iter())
                    .all(|v| v.is_finite()),
                "BLAS refit triangle contains non-finite value"
            );
            self.tris[dst_idx] = BlasTriGpu::from_tri(t);
        }
        // Reverse index order visits every child strictly before its parent.
        for idx in (0..self.nodes.len()).rev() {
            let node = self.nodes[idx];
            let mut b = Aabb::empty();
            if node.tri_count > 0 {
                let first = node.left_or_first as usize;
                let end = (first + node.tri_count as usize).min(self.tris.len());
                for tri in &self.tris[first..end] {
                    b.grow_pt([tri.v0[0], tri.v0[1], tri.v0[2]]);
                    b.grow_pt([tri.v1[0], tri.v1[1], tri.v1[2]]);
                    b.grow_pt([tri.v2[0], tri.v2[1], tri.v2[2]]);
                }
            } else {
                let left = node.left_or_first as usize;
                let right = left + 1;
                if left < self.nodes.len() {
                    b.grow_box(&Aabb { min: self.nodes[left].aabb_min, max: self.nodes[left].aabb_max });
                }
                if right < self.nodes.len() {
                    b.grow_box(&Aabb { min: self.nodes[right].aabb_min, max: self.nodes[right].aabb_max });
                }
            }
            self.nodes[idx].aabb_min = b.min;
            self.nodes[idx].aabb_max = b.max;
        }
        self.aabb_min = self.nodes[0].aabb_min;
        self.aabb_max = self.nodes[0].aabb_max;
        Ok(())
    }
}

/// Dynamic scene instance referencing a shared [`Blas`] with a world transform and shading parameters.
#[derive(Clone, Copy, Debug)]
pub struct TlasInstance {
    pub blas_index: u32,
    pub transform: [f32; 16],
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub metallic: f32,
    pub mask: u32,
}

impl TlasInstance {
    pub fn new(blas_index: u32, transform: [f32; 16]) -> Self {
        Self {
            blas_index,
            transform,
            albedo: [0.8, 0.8, 0.8],
            roughness: 0.5,
            emissive: [0.0, 0.0, 0.0],
            metallic: 0.0,
            mask: u32::MAX,
        }
    }
}

/// GPU-packed TLAS instance record (160 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TlasInstanceGpu {
    pub inv_transform: [f32; 16],
    pub normal_matrix_0: [f32; 4],
    pub normal_matrix_1: [f32; 4],
    pub normal_matrix_2: [f32; 4],
    pub albedo_roughness: [f32; 4],
    pub emissive_metallic: [f32; 4],
    pub blas_node_offset: u32,
    pub blas_node_count: u32,
    pub blas_tri_offset: u32,
    pub mask: u32,
}

fn transform_aabb(m: &[f32; 16], bmin: [f32; 3], bmax: [f32; 3]) -> Aabb {
    let mut out = Aabb::empty();
    for &x in &[bmin[0], bmax[0]] {
        for &y in &[bmin[1], bmax[1]] {
            for &z in &[bmin[2], bmax[2]] {
                let wx = m[0] * x + m[4] * y + m[8] * z + m[12];
                let wy = m[1] * x + m[5] * y + m[9] * z + m[13];
                let wz = m[2] * x + m[6] * y + m[10] * z + m[14];
                out.grow_pt([wx, wy, wz]);
            }
        }
    }
    out
}

/// Assembled 2-level acceleration structure combining packed BLAS pools with a dynamic SAH TLAS.
#[derive(Clone, Debug)]
pub struct TlasScene {
    tlas_nodes: Vec<BvhNodeGpu>,
    tlas_instances: Vec<TlasInstanceGpu>,
    blas_nodes: Vec<BvhNodeGpu>,
    blas_tris: Vec<BlasTriGpu>,
}

impl TlasScene {
    pub fn tlas_nodes(&self) -> &[BvhNodeGpu] {
        &self.tlas_nodes
    }

    pub fn tlas_instances(&self) -> &[TlasInstanceGpu] {
        &self.tlas_instances
    }

    pub fn blas_nodes(&self) -> &[BvhNodeGpu] {
        &self.blas_nodes
    }

    pub fn blas_tris(&self) -> &[BlasTriGpu] {
        &self.blas_tris
    }

    /// Builds a 2-level TLAS over `instances` referencing `blases`.
    /// Fast enough to rebuild every frame when dynamic transforms move without touching BLAS trees.
    pub fn build(blases: &[Blas], instances: &[TlasInstance]) -> Result<Self> {
        let mut blas_offsets = Vec::with_capacity(blases.len());
        let mut blas_nodes = Vec::new();
        let mut blas_tris = Vec::new();
        for b in blases {
            blas_offsets.push((blas_nodes.len() as u32, b.nodes.len() as u32, blas_tris.len() as u32));
            blas_nodes.extend_from_slice(&b.nodes);
            blas_tris.extend_from_slice(&b.tris);
        }

        let mut prepared = Vec::with_capacity(instances.len());
        let mut inst_boxes = Vec::with_capacity(instances.len());
        let mut centroids = Vec::with_capacity(instances.len());

        for inst in instances {
            ensure!(
                inst.transform.iter().all(|v| v.is_finite()),
                "TLAS instance transform contains non-finite value"
            );
            let bi = inst.blas_index as usize;
            ensure!(bi < blases.len(), "TLAS instance blas_index {bi} out of bounds ({})", blases.len());
            let blas = &blases[bi];
            if blas.nodes.is_empty() {
                continue;
            }
            let inv = crate::matrix::invert(&inst.transform);
            ensure!(inv.iter().all(|v| v.is_finite()), "TLAS instance transform is singular / non-invertible");
            // Columns of transpose(inv_3x3): column c = [inv[0*4 + c], inv[1*4 + c], inv[2*4 + c]].
            let nm0 = [inv[0], inv[4], inv[8], 0.0];
            let nm1 = [inv[1], inv[5], inv[9], 0.0];
            let nm2 = [inv[2], inv[6], inv[10], 0.0];
            let (node_off, node_cnt, tri_off) = blas_offsets[bi];
            let wb = transform_aabb(&inst.transform, blas.aabb_min, blas.aabb_max);
            centroids.push([
                0.5 * (wb.min[0] + wb.max[0]),
                0.5 * (wb.min[1] + wb.max[1]),
                0.5 * (wb.min[2] + wb.max[2]),
            ]);
            inst_boxes.push(wb);
            prepared.push(TlasInstanceGpu {
                inv_transform: inv,
                normal_matrix_0: nm0,
                normal_matrix_1: nm1,
                normal_matrix_2: nm2,
                albedo_roughness: [
                    inst.albedo[0].max(0.0),
                    inst.albedo[1].max(0.0),
                    inst.albedo[2].max(0.0),
                    inst.roughness.clamp(0.0, 1.0),
                ],
                emissive_metallic: [
                    inst.emissive[0].max(0.0),
                    inst.emissive[1].max(0.0),
                    inst.emissive[2].max(0.0),
                    inst.metallic.clamp(0.0, 1.0),
                ],
                blas_node_offset: node_off,
                blas_node_count: node_cnt,
                blas_tri_offset: tri_off,
                mask: inst.mask,
            });
        }

        if prepared.is_empty() {
            return Ok(Self { tlas_nodes: Vec::new(), tlas_instances: Vec::new(), blas_nodes, blas_tris });
        }

        let mut order: Vec<usize> = (0..prepared.len()).collect();
        let mut tlas_nodes =
            vec![BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 }];
        let mut tlas_instances = Vec::with_capacity(prepared.len());

        #[allow(clippy::too_many_arguments)]
        fn build_tlas_rec(
            node_idx: usize,
            start: usize,
            end: usize,
            depth: u32,
            order: &mut [usize],
            centroids: &[[f32; 3]],
            inst_boxes: &[Aabb],
            prepared: &[TlasInstanceGpu],
            nodes: &mut Vec<BvhNodeGpu>,
            out_insts: &mut Vec<TlasInstanceGpu>,
        ) {
            let mut bounds = Aabb::empty();
            let mut c_bounds = Aabb::empty();
            for &ii in &order[start..end] {
                bounds.grow_box(&inst_boxes[ii]);
                c_bounds.grow_pt(centroids[ii]);
            }
            let count = end - start;
            if count <= 2 || depth >= 20 {
                let first = out_insts.len() as u32;
                for &ii in &order[start..end] {
                    out_insts.push(prepared[ii]);
                }
                nodes[node_idx] = BvhNodeGpu {
                    aabb_min: bounds.min,
                    left_or_first: first,
                    aabb_max: bounds.max,
                    tri_count: count as u32,
                };
                return;
            }
            let ext = [
                c_bounds.max[0] - c_bounds.min[0],
                c_bounds.max[1] - c_bounds.min[1],
                c_bounds.max[2] - c_bounds.min[2],
            ];
            let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
                0
            } else if ext[1] >= ext[2] {
                1
            } else {
                2
            };
            let mid = if ext[axis] <= 1e-6 {
                start + count / 2
            } else {
                let split_pos = c_bounds.min[axis] + 0.5 * ext[axis];
                let mut m = start;
                for i in start..end {
                    if centroids[order[i]][axis] < split_pos {
                        order.swap(i, m);
                        m += 1;
                    }
                }
                if m == start || m == end { start + count / 2 } else { m }
            };
            let left_idx = nodes.len();
            nodes.push(BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 });
            nodes.push(BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 });
            nodes[node_idx] = BvhNodeGpu {
                aabb_min: bounds.min,
                left_or_first: left_idx as u32,
                aabb_max: bounds.max,
                tri_count: 0,
            };
            build_tlas_rec(left_idx, start, mid, depth + 1, order, centroids, inst_boxes, prepared, nodes, out_insts);
            build_tlas_rec(left_idx + 1, mid, end, depth + 1, order, centroids, inst_boxes, prepared, nodes, out_insts);
        }

        build_tlas_rec(
            0,
            0,
            prepared.len(),
            0,
            &mut order,
            &centroids,
            &inst_boxes,
            &prepared,
            &mut tlas_nodes,
            &mut tlas_instances,
        );
        Ok(Self { tlas_nodes, tlas_instances, blas_nodes, blas_tris })
    }
}

/// Uploaded GPU buffers for a [`TlasScene`], shareable across [`TlasRayTracer`], DDGI, and lightmap baking.
pub struct TlasSceneGpu {
    pub tlas_nodes_buf: wgpu::Buffer,
    pub tlas_instances_buf: wgpu::Buffer,
    pub blas_nodes_buf: wgpu::Buffer,
    pub blas_tris_buf: wgpu::Buffer,
    pub tlas_node_count: u32,
    pub tlas_instance_count: u32,
    pub blas_tri_count: u32,
}

impl TlasSceneGpu {
    pub fn upload(device: &wgpu::Device, scene: &TlasScene) -> Result<Self> {
        let max_storage = device.limits().max_storage_buffer_binding_size;
        let t_nodes = if scene.tlas_nodes.is_empty() {
            vec![BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 }]
        } else {
            scene.tlas_nodes.clone()
        };
        let t_insts = if scene.tlas_instances.is_empty() {
            vec![bytemuck::Zeroable::zeroed()]
        } else {
            scene.tlas_instances.clone()
        };
        let b_nodes = if scene.blas_nodes.is_empty() {
            vec![BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 }]
        } else {
            scene.blas_nodes.clone()
        };
        let b_tris = if scene.blas_tris.is_empty() {
            vec![bytemuck::Zeroable::zeroed()]
        } else {
            scene.blas_tris.clone()
        };

        let make_buf = |label: &str, bytes: &[u8]| -> Result<wgpu::Buffer> {
            ensure!(bytes.len() as u64 <= max_storage, "{label} exceeds device storage limit");
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: true,
            });
            buf.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytes);
            buf.unmap();
            Ok(buf)
        };

        Ok(Self {
            tlas_nodes_buf: make_buf("tlas_nodes", bytemuck::cast_slice(&t_nodes))?,
            tlas_instances_buf: make_buf("tlas_instances", bytemuck::cast_slice(&t_insts))?,
            blas_nodes_buf: make_buf("blas_nodes", bytemuck::cast_slice(&b_nodes))?,
            blas_tris_buf: make_buf("blas_tris", bytemuck::cast_slice(&b_tris))?,
            tlas_node_count: scene.tlas_nodes.len() as u32,
            tlas_instance_count: scene.tlas_instances.len() as u32,
            blas_tri_count: scene.blas_tris.len() as u32,
        })
    }
}

/// Parameters for [`TlasRayTracer`] covering soft shadows, RTAO, GGX specular reflections, and 1-bounce diffuse GI.
#[derive(Clone, Copy, Debug)]
pub struct TlasRayTraceParams {
    pub view_projection: [f32; 16],
    pub camera_pos: [f32; 3],
    pub depth_clear: f32,
    pub sun_direction: [f32; 3],
    pub sun_angular_radius: f32,
    pub sun_color: [f32; 3],
    pub sky_zenith: [f32; 3],
    pub sky_horizon: [f32; 3],
    pub surface_roughness: f32,
    pub ao_radius: f32,
    pub max_ray_distance: f32,
    pub shadow_samples: u32,
    pub ao_samples: u32,
    pub reflection_samples: u32,
    pub gi_samples: u32,
    pub gi_bounce_scale: f32,
    pub frame_seed: f32,
    pub normal_bias: f32,
    pub visibility_mask: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TlasRayTraceUniform {
    inverse_view_projection: [f32; 16],
    camera_pos_and_clear: [f32; 4],
    sun_dir_and_radius: [f32; 4],
    sun_color_and_rough: [f32; 4],
    sky_zenith: [f32; 4],
    sky_horizon: [f32; 4],
    samples_seed: [f32; 4],
    bias_and_refl: [f32; 4],
    counts: [u32; 4],
}

struct RtOutputs {
    size: (u32, u32),
    shadow_ao: (wgpu::Texture, wgpu::TextureView),
    reflection: (wgpu::Texture, wgpu::TextureView),
    gi: (wgpu::Texture, wgpu::TextureView),
}

impl RtOutputs {
    fn new(device: &wgpu::Device, size: (u32, u32)) -> Self {
        let make = |label: &str| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        Self {
            size,
            shadow_ao: make("tlas_rt_shadow_ao"),
            reflection: make("tlas_rt_reflection"),
            gi: make("tlas_rt_gi"),
        }
    }
}

/// Portable 2-level TLAS/BLAS GPU compute ray tracer producing shadows/RTAO, GGX specular reflections,
/// and 1-bounce diffuse GI in a single pass.
pub struct TlasRayTracer {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    scene: Option<TlasSceneGpu>,
    outputs: Option<RtOutputs>,
}

impl TlasRayTracer {
    pub fn new(device: &wgpu::Device) -> Self {
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let out_wo = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tlas_rt_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<TlasRayTraceUniform>() as u64),
                    },
                    count: None,
                },
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_ro(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                out_wo(6),
                out_wo(7),
                out_wo(8),
            ],
        });
        let source = format!("{TLAS_TRAVERSAL_WGSL}\n{TLAS_RAYTRACE_WGSL}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tlas_raytrace"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tlas_rt_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_rt_pipe"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("trace_scene"),
            compilation_options: Default::default(),
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_rt_u"),
            size: std::mem::size_of::<TlasRayTraceUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, bgl, uniform, scene: None, outputs: None }
    }

    pub fn upload_scene(&mut self, device: &wgpu::Device, scene: &TlasScene) -> Result<()> {
        self.scene = Some(TlasSceneGpu::upload(device, scene)?);
        Ok(())
    }

    pub fn scene_gpu(&self) -> Option<&TlasSceneGpu> {
        self.scene.as_ref()
    }

    pub fn shadow_ao_texture(&self) -> Option<&wgpu::Texture> {
        self.outputs.as_ref().map(|o| &o.shadow_ao.0)
    }

    pub fn shadow_ao_view(&self) -> Option<&wgpu::TextureView> {
        self.outputs.as_ref().map(|o| &o.shadow_ao.1)
    }

    pub fn reflection_texture(&self) -> Option<&wgpu::Texture> {
        self.outputs.as_ref().map(|o| &o.reflection.0)
    }

    pub fn reflection_view(&self) -> Option<&wgpu::TextureView> {
        self.outputs.as_ref().map(|o| &o.reflection.1)
    }

    pub fn gi_texture(&self) -> Option<&wgpu::Texture> {
        self.outputs.as_ref().map(|o| &o.gi.0)
    }

    pub fn gi_view(&self) -> Option<&wgpu::TextureView> {
        self.outputs.as_ref().map(|o| &o.gi.1)
    }

    pub fn trace(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        size: (u32, u32),
        params: &TlasRayTraceParams,
    ) -> Result<()> {
        let scene = self.scene.as_ref().ok_or_else(|| anyhow::anyhow!("call upload_scene before trace"))?;
        Self::dispatch_trace(
            &self.pipeline,
            &self.bgl,
            &self.uniform,
            &mut self.outputs,
            device,
            encoder,
            scene,
            depth_view,
            size,
            params,
        )
    }

    pub fn trace_with_scene(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene: &TlasSceneGpu,
        depth_view: &wgpu::TextureView,
        size: (u32, u32),
        params: &TlasRayTraceParams,
    ) -> Result<()> {
        Self::dispatch_trace(
            &self.pipeline,
            &self.bgl,
            &self.uniform,
            &mut self.outputs,
            device,
            encoder,
            scene,
            depth_view,
            size,
            params,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_trace(
        pipeline: &wgpu::ComputePipeline,
        bgl: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        outputs: &mut Option<RtOutputs>,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene: &TlasSceneGpu,
        depth_view: &wgpu::TextureView,
        size: (u32, u32),
        params: &TlasRayTraceParams,
    ) -> Result<()> {
        let max_dim = device.limits().max_texture_dimension_2d;
        ensure!(size.0 >= 1 && size.1 >= 1 && size.0 <= max_dim && size.1 <= max_dim, "invalid RT size {size:?}");
        let d_sz = depth_view.texture().size();
        ensure!(
            (d_sz.width, d_sz.height) == size,
            "TlasRayTracer depth_view {}x{} != {size:?}",
            d_sz.width,
            d_sz.height
        );
        ensure!(params.view_projection.iter().all(|v| v.is_finite()), "non-finite view_projection");

        if !matches!(outputs, Some(o) if o.size == size) {
            *outputs = Some(RtOutputs::new(device, size));
        }
        let u = TlasRayTraceUniform {
            inverse_view_projection: crate::matrix::invert(&params.view_projection),
            camera_pos_and_clear: [params.camera_pos[0], params.camera_pos[1], params.camera_pos[2], params.depth_clear],
            sun_dir_and_radius: [
                params.sun_direction[0],
                params.sun_direction[1],
                params.sun_direction[2],
                params.sun_angular_radius.clamp(0.0, 0.2),
            ],
            sun_color_and_rough: [
                params.sun_color[0].max(0.0),
                params.sun_color[1].max(0.0),
                params.sun_color[2].max(0.0),
                params.surface_roughness.clamp(0.0, 1.0),
            ],
            sky_zenith: [
                params.sky_zenith[0].max(0.0),
                params.sky_zenith[1].max(0.0),
                params.sky_zenith[2].max(0.0),
                params.ao_radius.max(0.05),
            ],
            sky_horizon: [
                params.sky_horizon[0].max(0.0),
                params.sky_horizon[1].max(0.0),
                params.sky_horizon[2].max(0.0),
                params.max_ray_distance.max(1.0),
            ],
            samples_seed: [
                params.shadow_samples.clamp(1, 8) as f32,
                params.ao_samples.min(16) as f32,
                params.gi_samples.min(8) as f32,
                params.frame_seed,
            ],
            bias_and_refl: [
                params.normal_bias.max(1e-5),
                params.reflection_samples.min(8) as f32,
                params.gi_bounce_scale.max(0.0),
                0.0,
            ],
            counts: [scene.tlas_node_count, scene.tlas_instance_count, scene.blas_tri_count, params.visibility_mask],
        };
        encode_copy(device, encoder, uniform, bytemuck::bytes_of(&u));
        let out = outputs.as_ref().unwrap();
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tlas_rt_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scene.tlas_nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scene.tlas_instances_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: scene.blas_nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: scene.blas_tris_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&out.shadow_ao.1) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&out.reflection.1) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&out.gi.1) },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tlas_raytrace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_to_f32(bits: u16) -> f32 {
        let sign = u32::from((bits >> 15) & 1) << 31;
        let exp = u32::from((bits >> 10) & 0x1f);
        let frac = u32::from(bits & 0x3ff);
        if exp == 0 {
            if frac == 0 {
                f32::from_bits(sign)
            } else {
                let mut f = frac;
                let mut e = -14i32;
                while (f & 0x400) == 0 {
                    f <<= 1;
                    e -= 1;
                }
                f &= 0x3ff;
                f32::from_bits(sign | (((e + 127) as u32) << 23) | (f << 13))
            }
        } else if exp == 31 {
            f32::from_bits(sign | (0xff << 23) | (frac << 13))
        } else {
            f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13))
        }
    }

    #[test]
    fn gpu_bvh_watertight_raytraced_shadow_and_concave_rtao_witness() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU adapter required");
        let _hw_rq = supports_hardware_ray_query(&adapter);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

        let tris = [
            // Horizontal occluder roof quad at y = 1.5 over x in [-1.8, -0.5] (shared diagonal tests watertightness!).
            BvhTriangle { v0: [-1.8, 1.5, -1.5], v1: [-0.5, 1.5, -1.5], v2: [-0.5, 1.5, 1.5] },
            BvhTriangle { v0: [-1.8, 1.5, -1.5], v1: [-0.5, 1.5, 1.5], v2: [-1.8, 1.5, 1.5] },
            // Vertical wall at x = -0.5, y in [0, 2.5], z in [-2, 2] for RTAO concavity!
            BvhTriangle { v0: [-0.5, 0.0, -2.0], v1: [-0.5, 2.5, -2.0], v2: [-0.5, 2.5, 2.0] },
            BvhTriangle { v0: [-0.5, 0.0, -2.0], v1: [-0.5, 2.5, 2.0], v2: [-0.5, 0.0, 2.0] },
            // Degenerate zero-area triangle must be filtered cleanly without breaking build.
            BvhTriangle { v0: [0.0, 0.0, 0.0], v1: [0.0, 0.0, 0.0], v2: [0.0, 0.0, 0.0] },
        ];
        let bvh = BvhScene::build(&tris).unwrap();
        assert_eq!(bvh.tris().len(), 4, "zero-area degenerate triangle must be filtered");
        let mut rt = GpuRayTracer::new(&device);
        rt.upload_bvh(&device, &bvh).unwrap();

        let vp: [f32; 16] = [
            0.5, 0.0, 0.0, 0.0,
            0.0, 0.0, -0.25, 0.0,
            0.0, -0.5, 0.0, 0.0,
            0.0, 0.0, 0.5, 1.0,
        ];
        let (w, h) = (32u32, 32u32);
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rt_depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &depth_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&vec![0.5f32; (w * h) as usize]),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        rt.trace(
            &device,
            &mut enc,
            &depth_view,
            (w, h),
            &RayTraceParams {
                view_projection: vp,
                camera_pos: [0.0, 4.0, 0.0],
                depth_clear: 1.0,
                sun_direction: [0.0, 1.0, 0.0], // Axis-aligned ray (+Y) tests parallel X/Z slab handling!
                sun_angular_radius: 0.0,
                ao_radius: 2.5,
                ao_samples: 16,
                shadow_samples: 1,
                frame_seed: 1.0,
                normal_bias: 0.01,
            },
        )
        .unwrap();

        let bpr = (w * 8).next_multiple_of(256);
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rt_rb"),
            size: u64::from(bpr * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: rt.output_texture().unwrap(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &rb,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().unwrap();
        let read_px = |x: u32, y: u32| -> [f32; 4] {
            let o = (y * bpr + x * 8) as usize;
            [
                f16_to_f32(u16::from_le_bytes([data[o], data[o + 1]])),
                f16_to_f32(u16::from_le_bytes([data[o + 2], data[o + 3]])),
                f16_to_f32(u16::from_le_bytes([data[o + 4], data[o + 5]])),
                f16_to_f32(u16::from_le_bytes([data[o + 6], data[o + 7]])),
            ]
        };
        let under_roof = read_px(7, 16);
        let open_floor = read_px(26, 16);
        assert!(under_roof[0] < 0.05, "occluder roof must cast exact dark RT shadow: {under_roof:?}");
        assert!(open_floor[0] > 0.95, "open floor must stay fully lit: {open_floor:?}");
        assert!(
            under_roof[1] + 0.15 < open_floor[1],
            "concave wall+roof corner must have darker RTAO than open floor: corner={} open={}",
            under_roof[1],
            open_floor[1]
        );
        drop(data);
        rb.unmap();
    }

    fn read_rgba16f(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, w: u32, h: u32) -> Vec<[f32; 4]> {
        let bpr = (w * 8).next_multiple_of(256);
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_rb"),
            size: u64::from(bpr * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &rb,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().unwrap();
        let mut out = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let o = (y * bpr + x * 8) as usize;
                out.push([
                    f16_to_f32(u16::from_le_bytes([data[o], data[o + 1]])),
                    f16_to_f32(u16::from_le_bytes([data[o + 2], data[o + 3]])),
                    f16_to_f32(u16::from_le_bytes([data[o + 4], data[o + 5]])),
                    f16_to_f32(u16::from_le_bytes([data[o + 6], data[o + 7]])),
                ]);
            }
        }
        drop(data);
        rb.unmap();
        out
    }

    #[test]
    fn gpu_tlas_blas_dynamic_transform_refit_reflection_and_gi_witness() {
        let (device, queue) = crate::testing::device().expect("GPU adapter required");

        // Reusable unit quad BLAS in XZ plane centered at origin ([-0.5..0.5] x 0 x [-1.5..1.5]).
        let quad_tris = [
            BlasTriangle::flat([-0.6, 0.0, -1.5], [0.6, 0.0, -1.5], [0.6, 0.0, 1.5]),
            BlasTriangle::flat([-0.6, 0.0, -1.5], [0.6, 0.0, 1.5], [-0.6, 0.0, 1.5]),
        ];
        let mut roof_blas = Blas::build(&quad_tris).unwrap();

        let translate = |x: f32, y: f32, z: f32| -> [f32; 16] {
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0]
        };

        // Frame 1: glowing red roof over LEFT half (x = -1.15, y = 1.5).
        let mut left_inst = TlasInstance::new(0, translate(-1.15, 1.5, 0.0));
        left_inst.albedo = [0.9, 0.05, 0.05];
        left_inst.emissive = [3.0, 0.0, 0.0];
        let scene1 = TlasScene::build(&[roof_blas.clone()], &[left_inst]).unwrap();

        let mut rt = TlasRayTracer::new(&device);
        rt.upload_scene(&device, &scene1).unwrap();

        let vp: [f32; 16] = [
            0.5, 0.0, 0.0, 0.0,
            0.0, 0.0, -0.25, 0.0,
            0.0, -0.5, 0.0, 0.0,
            0.0, 0.0, 0.5, 1.0,
        ];
        let (w, h) = (32u32, 32u32);
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tlas_depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            depth_tex.as_image_copy(),
            bytemuck::cast_slice(&vec![0.5f32; (w * h) as usize]),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let depth_view = depth_tex.create_view(&Default::default());

        let params = TlasRayTraceParams {
            view_projection: vp,
            camera_pos: [0.0, 4.0, 0.0],
            depth_clear: 1.0,
            sun_direction: [0.0, 1.0, 0.0],
            sun_angular_radius: 0.0,
            sun_color: [2.0, 2.0, 2.0],
            sky_zenith: [0.05, 0.1, 0.3],
            sky_horizon: [0.05, 0.1, 0.25],
            surface_roughness: 0.0, // mirror specular reflection directly upward to hit red roof
            ao_radius: 3.0,
            max_ray_distance: 100.0,
            shadow_samples: 1,
            ao_samples: 8,
            reflection_samples: 1,
            gi_samples: 8,
            gi_bounce_scale: 1.0,
            frame_seed: 1.0,
            normal_bias: 0.01,
            visibility_mask: u32::MAX,
        };

        let mut enc = device.create_command_encoder(&Default::default());
        rt.trace(&device, &mut enc, &depth_view, (w, h), &params).unwrap();
        queue.submit(Some(enc.finish()));

        let sh1 = read_rgba16f(&device, &queue, rt.shadow_ao_texture().unwrap(), w, h);
        let refl1 = read_rgba16f(&device, &queue, rt.reflection_texture().unwrap(), w, h);
        let gi1 = read_rgba16f(&device, &queue, rt.gi_texture().unwrap(), w, h);
        let idx_left = (16 * w + 7) as usize;
        let idx_right = (16 * w + 25) as usize;

        assert!(sh1[idx_left][0] < 0.05, "left floor must be shadowed in frame 1: {:?}", sh1[idx_left]);
        assert!(sh1[idx_right][0] > 0.95, "right floor must be lit in frame 1: {:?}", sh1[idx_right]);
        assert!(
            refl1[idx_left][0] > 1.5 && refl1[idx_left][0] > refl1[idx_left][2] * 3.0,
            "mirror reflection under red emissive roof must reflect bright red: {:?}",
            refl1[idx_left]
        );
        assert!(
            gi1[idx_left][0] > gi1[idx_left][2] * 2.0 && gi1[idx_left][0] > 0.4,
            "1-bounce diffuse GI under red roof must show strong red color bleed: {:?}",
            gi1[idx_left]
        );

        // Frame 2: dynamically move the TLAS instance to the RIGHT side + refit the BLAS in place!
        let refit_tris = [
            BlasTriangle::flat([-0.7, 0.0, -1.6], [0.7, 0.0, -1.6], [0.7, 0.0, 1.6]),
            BlasTriangle::flat([-0.7, 0.0, -1.6], [0.7, 0.0, 1.6], [-0.7, 0.0, 1.6]),
        ];
        roof_blas.refit(&refit_tris).unwrap();
        let mut right_inst = TlasInstance::new(0, translate(1.15, 1.5, 0.0));
        right_inst.albedo = [0.05, 0.9, 0.05];
        right_inst.emissive = [0.0, 3.0, 0.0];
        let scene2 = TlasScene::build(&[roof_blas], &[right_inst]).unwrap();
        rt.upload_scene(&device, &scene2).unwrap();

        let mut enc2 = device.create_command_encoder(&Default::default());
        rt.trace(&device, &mut enc2, &depth_view, (w, h), &params).unwrap();
        queue.submit(Some(enc2.finish()));

        let sh2 = read_rgba16f(&device, &queue, rt.shadow_ao_texture().unwrap(), w, h);
        let refl2 = read_rgba16f(&device, &queue, rt.reflection_texture().unwrap(), w, h);
        assert!(sh2[idx_left][0] > 0.95, "left floor must become lit after TLAS move: {:?}", sh2[idx_left]);
        assert!(sh2[idx_right][0] < 0.05, "right floor must become shadowed after TLAS move + refit: {:?}", sh2[idx_right]);
        assert!(
            refl2[idx_right][1] > 1.5 && refl2[idx_right][1] > refl2[idx_right][0] * 3.0,
            "reflection under moved green roof must now reflect bright green: {:?}",
            refl2[idx_right]
        );
    }
}
