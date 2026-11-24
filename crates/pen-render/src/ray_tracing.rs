//! Portable GPU BVH Ray-Traced Soft Shadows & Ray-Traced Ambient Occlusion (RTAO),
//! plus hardware `EXPERIMENTAL_RAY_QUERY` capability detection.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "GPU BVH node and texture indices")]

use anyhow::{Result, ensure};

pub const BVH_RAYTRACE_WGSL: &str = include_str!("shaders/bvh_raytrace.wgsl");

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
}
