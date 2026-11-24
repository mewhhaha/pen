//! GPU Meshlets & Cluster Culling:
//! - Partitions indexed triangle meshes into compact clusters (`<= 64` verts, `<= 124` tris)
//! - Computes conservative `f32`-rounded bounding spheres & local-space normal cones
//! - GPU compute culler (`MESHLET_CULL_WGSL`) performing exact local-space frustum & normal-cone
//!   backface culling (conservative under arbitrary shear/scale) and full-footprint Hi-Z culling
//!   into baseline-WebGPU `DrawIndexedIndirect` records (`first_instance = 0`).
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "GPU meshlet buffer indices")]

use std::collections::HashSet;

use anyhow::{Result, ensure};

pub const MESHLET_CULL_WGSL: &str = include_str!("shaders/meshlet_cull.wgsl");

pub const DEFAULT_MAX_MESHLET_VERTICES: usize = 64;
pub const DEFAULT_MAX_MESHLET_TRIANGLES: usize = 124;

/// GPU-ready 64-byte meshlet descriptor matching `meshlet_cull.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshletGpu {
    /// `[cx, cy, cz, radius]` bounding sphere in mesh local space (radius rounded outward in `f32`).
    pub sphere: [f32; 4],
    /// `[axis_x, axis_y, axis_z, cutoff]` where `cutoff = sin(max_half_angle)` (`>= 1.0` disables cone cull).
    pub cone_axis_cutoff: [f32; 4],
    /// `[first_index, index_count, base_vertex_bits, tri_count]` stored as exact `u32` (no `f32` precision loss).
    pub counts: [u32; 4],
    /// `[unique_verts, 0, 0, 0]`.
    pub meta: [u32; 4],
}

/// Partitioned meshlet representation ready for GPU cluster culling and indirect indexed draws.
#[derive(Clone, Debug)]
pub struct MeshletMesh {
    pub meshlets: Vec<MeshletGpu>,
    pub indices: Vec<u32>,
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f64; 3] {
    [f64::from(a[0] - b[0]), f64::from(a[1] - b[1]), f64::from(a[2] - b[2])]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3(a: [f64; 3]) -> Option<[f64; 3]> {
    let l = dot3(a, a).sqrt();
    if l > 1e-12 && l.is_finite() { Some([a[0] / l, a[1] / l, a[2] / l]) } else { None }
}

/// Builds spatially coherent meshlet clusters with conservative bounding spheres and normal cones.
pub fn build_meshlets(
    positions: &[[f32; 3]],
    indices: &[u32],
    max_vertices: usize,
    max_triangles: usize,
) -> Result<MeshletMesh> {
    ensure!(indices.len().is_multiple_of(3), "index count must be a multiple of 3");
    ensure!(u32::try_from(indices.len()).is_ok(), "mesh index count exceeds u32::MAX");
    ensure!((3..=256).contains(&max_vertices), "max_vertices must be in 3..=256");
    ensure!((1..=512).contains(&max_triangles), "max_triangles must be in 1..=512");
    let tri_count = indices.len() / 3;
    if tri_count == 0 {
        return Ok(MeshletMesh { meshlets: Vec::new(), indices: Vec::new() });
    }
    for &idx in indices {
        ensure!((idx as usize) < positions.len(), "mesh index {idx} out of bounds for {} vertices", positions.len());
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); positions.len()];
    let mut tri_centroids = Vec::with_capacity(tri_count);
    for t in 0..tri_count {
        let (a, b, c) = (indices[t * 3] as usize, indices[t * 3 + 1] as usize, indices[t * 3 + 2] as usize);
        adj[a].push(t);
        adj[b].push(t);
        adj[c].push(t);
        let (pa, pb, pc) = (positions[a], positions[b], positions[c]);
        tri_centroids.push([
            (pa[0] + pb[0] + pc[0]) / 3.0,
            (pa[1] + pb[1] + pc[1]) / 3.0,
            (pa[2] + pb[2] + pc[2]) / 3.0,
        ]);
    }

    let mut emitted = vec![false; tri_count];
    let mut in_frontier = vec![false; tri_count];
    let mut emitted_count = 0usize;
    let mut meshlets = Vec::new();
    let mut out_indices = Vec::with_capacity(indices.len());
    let mut next_seed = 0usize;

    while emitted_count < tri_count {
        while next_seed < tri_count && emitted[next_seed] {
            next_seed += 1;
        }
        if next_seed >= tri_count {
            break;
        }
        let mut cluster_tris: Vec<usize> = Vec::with_capacity(max_triangles);
        let mut cluster_verts: HashSet<u32> = HashSet::with_capacity(max_vertices);
        let mut frontier: Vec<usize> = Vec::new();
        let mut cx = 0.0f64;
        let mut cy = 0.0f64;
        let mut cz = 0.0f64;

        let push_tri = |t: usize,
                        emitted: &mut [bool],
                        in_frontier: &mut [bool],
                        cluster_tris: &mut Vec<usize>,
                        cluster_verts: &mut HashSet<u32>,
                        frontier: &mut Vec<usize>,
                        cx: &mut f64,
                        cy: &mut f64,
                        cz: &mut f64| {
            emitted[t] = true;
            in_frontier[t] = false;
            cluster_tris.push(t);
            let tc = tri_centroids[t];
            *cx += f64::from(tc[0]);
            *cy += f64::from(tc[1]);
            *cz += f64::from(tc[2]);
            for k in 0..3 {
                let v = indices[t * 3 + k];
                if cluster_verts.insert(v) {
                    for &nb in &adj[v as usize] {
                        if !emitted[nb] && !in_frontier[nb] {
                            in_frontier[nb] = true;
                            frontier.push(nb);
                        }
                    }
                }
            }
        };

        push_tri(
            next_seed,
            &mut emitted,
            &mut in_frontier,
            &mut cluster_tris,
            &mut cluster_verts,
            &mut frontier,
            &mut cx,
            &mut cy,
            &mut cz,
        );
        emitted_count += 1;

        while cluster_tris.len() < max_triangles {
            let inv_n = 1.0 / cluster_tris.len() as f64;
            let mean = [cx * inv_n, cy * inv_n, cz * inv_n];
            let mut best: Option<(usize, f64)> = None;
            frontier.retain(|&t| !emitted[t]);
            for &cand in &frontier {
                let (v0, v1, v2) = (indices[cand * 3], indices[cand * 3 + 1], indices[cand * 3 + 2]);
                // Count DISTINCT unseen vertices even for degenerate triangles with repeated indices.
                let mut added = usize::from(!cluster_verts.contains(&v0));
                if v1 != v0 && !cluster_verts.contains(&v1) {
                    added += 1;
                }
                if v2 != v0 && v2 != v1 && !cluster_verts.contains(&v2) {
                    added += 1;
                }
                if cluster_verts.len() + added > max_vertices {
                    continue;
                }
                let tc = tri_centroids[cand];
                let d2 = (f64::from(tc[0]) - mean[0]).powi(2)
                    + (f64::from(tc[1]) - mean[1]).powi(2)
                    + (f64::from(tc[2]) - mean[2]).powi(2);
                let score = (added as f64) * 100.0 + d2;
                if best.is_none_or(|(_, s)| score < s) {
                    best = Some((cand, score));
                }
            }
            let Some((chosen, _)) = best else {
                break;
            };
            push_tri(
                chosen,
                &mut emitted,
                &mut in_frontier,
                &mut cluster_tris,
                &mut cluster_verts,
                &mut frontier,
                &mut cx,
                &mut cy,
                &mut cz,
            );
            emitted_count += 1;
        }
        // Clear any leftover frontier markers before starting the next cluster.
        for &t in &frontier {
            in_frontier[t] = false;
        }

        let first_index = u32::try_from(out_indices.len())?;
        let mut min_p = [f64::INFINITY; 3];
        let mut max_p = [f64::NEG_INFINITY; 3];
        for &v in &cluster_verts {
            let p = positions[v as usize];
            for k in 0..3 {
                min_p[k] = min_p[k].min(f64::from(p[k]));
                max_p[k] = max_p[k].max(f64::from(p[k]));
            }
        }
        // Measure radius directly around the exact `f32`-stored center and round outward conservatively.
        let center_f32 = [
            (0.5 * (min_p[0] + max_p[0])) as f32,
            (0.5 * (min_p[1] + max_p[1])) as f32,
            (0.5 * (min_p[2] + max_p[2])) as f32,
        ];
        let mut radius = 0.0f64;
        for &v in &cluster_verts {
            let p = positions[v as usize];
            let d = ((f64::from(p[0]) - f64::from(center_f32[0])).powi(2)
                + (f64::from(p[1]) - f64::from(center_f32[1])).powi(2)
                + (f64::from(p[2]) - f64::from(center_f32[2])).powi(2))
            .sqrt();
            radius = radius.max(d);
        }
        let radius_f32 = (radius * (1.0 + 1e-6)) as f32 + f32::EPSILON;

        let mut normals = Vec::with_capacity(cluster_tris.len());
        let mut avg_n = [0.0f64; 3];
        for &t in &cluster_tris {
            let (i0, i1, i2) = (indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]);
            out_indices.extend_from_slice(&[i0, i1, i2]);
            let e1 = sub3(positions[i1 as usize], positions[i0 as usize]);
            let e2 = sub3(positions[i2 as usize], positions[i0 as usize]);
            if let Some(n) = norm3(cross3(e1, e2)) {
                normals.push(n);
                avg_n[0] += n[0];
                avg_n[1] += n[1];
                avg_n[2] += n[2];
            }
        }
        let (cone_axis, cone_cutoff) = if let Some(axis) = norm3(avg_n) {
            let mut min_dot = 1.0f64;
            for n in &normals {
                min_dot = min_dot.min(dot3(*n, axis));
            }
            if min_dot > 0.1 {
                let cutoff = (1.0 - min_dot * min_dot).max(0.0).sqrt();
                ([axis[0] as f32, axis[1] as f32, axis[2] as f32], cutoff as f32)
            } else {
                ([0.0, 1.0, 0.0], 1.0)
            }
        } else {
            ([0.0, 1.0, 0.0], 1.0)
        };

        let index_count = u32::try_from(cluster_tris.len() * 3)?;
        meshlets.push(MeshletGpu {
            sphere: [center_f32[0], center_f32[1], center_f32[2], radius_f32],
            cone_axis_cutoff: [cone_axis[0], cone_axis[1], cone_axis[2], cone_cutoff],
            counts: [first_index, index_count, 0u32, cluster_tris.len() as u32],
            meta: [cluster_verts.len() as u32, 0, 0, 0],
        });
    }

    Ok(MeshletMesh { meshlets, indices: out_indices })
}

/// Parameters for one GPU meshlet cluster culling dispatch.
#[derive(Clone, Copy, Debug)]
pub struct MeshletCullParams {
    pub model: [f32; 16],
    pub hiz_vp: [f32; 16],
    pub frustum_planes: [[f32; 4]; 6],
    pub camera_pos: [f32; 3],
    pub hiz_enabled: bool,
    pub hiz_mips: u32,
    pub backface_cull: bool,
    pub reversed_z: bool,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CullUniform {
    model: [f32; 16],
    inv_model: [f32; 16],
    hiz_vp: [f32; 16],
    planes: [[f32; 4]; 6],
    camera_pos: [f32; 4],
    counts_flags: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshletCullStats {
    pub visible: u32,
    pub frustum_culled: u32,
    pub backface_culled: u32,
    pub hiz_culled: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshletIndirectDraw {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

fn encode_copy(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, dst: &wgpu::Buffer, bytes: &[u8]) {
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("meshlet_stage"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    stage.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytes);
    stage.unmap();
    encoder.copy_buffer_to_buffer(&stage, 0, dst, 0, bytes.len() as u64);
}

/// GPU compute meshlet cluster culler (local-space frustum + shear-safe normal-cone backface + full-footprint Hi-Z).
pub struct MeshletCuller {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    stats: wgpu::Buffer,
    dummy_hiz: wgpu::Texture,
    dummy_hiz_view: wgpu::TextureView,
}

impl MeshletCuller {
    pub fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("meshlet_cull_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<CullUniform>() as u64),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("meshlet_cull_shader"),
            source: wgpu::ShaderSource::Wgsl(MESHLET_CULL_WGSL.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("meshlet_cull_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("meshlet_cull_pipe"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("cull_meshlets"),
            compilation_options: Default::default(),
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meshlet_cull_u"),
            size: std::mem::size_of::<CullUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stats = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meshlet_cull_stats"),
            size: std::mem::size_of::<MeshletCullStats>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dummy_hiz = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("meshlet_dummy_hiz"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_hiz_view = dummy_hiz.create_view(&wgpu::TextureViewDescriptor::default());
        Self { pipeline, bgl, uniform, stats, dummy_hiz, dummy_hiz_view }
    }

    pub fn stats_buffer(&self) -> &wgpu::Buffer {
        &self.stats
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cull(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        meshlets_buf: &wgpu::Buffer,
        draws_buf: &wgpu::Buffer,
        meshlet_count: u32,
        hiz_view: Option<&wgpu::TextureView>,
        params: &MeshletCullParams,
    ) -> Result<()> {
        if meshlet_count == 0 {
            return Ok(());
        }
        let limits = device.limits();
        let req_m_bytes = u64::from(meshlet_count) * std::mem::size_of::<MeshletGpu>() as u64;
        let req_d_bytes = u64::from(meshlet_count) * std::mem::size_of::<MeshletIndirectDraw>() as u64;
        ensure!(
            meshlets_buf.usage().contains(wgpu::BufferUsages::STORAGE) && meshlets_buf.size() >= req_m_bytes,
            "meshlets_buf must have STORAGE usage and at least {req_m_bytes} bytes"
        );
        ensure!(
            draws_buf.usage().contains(wgpu::BufferUsages::STORAGE) && draws_buf.size() >= req_d_bytes,
            "draws_buf must have STORAGE usage and at least {req_d_bytes} bytes"
        );
        ensure!(
            req_m_bytes <= limits.max_storage_buffer_binding_size && req_d_bytes <= limits.max_storage_buffer_binding_size,
            "meshlet buffers exceed device max_storage_buffer_binding_size"
        );
        let wg = meshlet_count.div_ceil(64);
        ensure!(
            wg <= limits.max_compute_workgroups_per_dimension,
            "meshlet workgroup count {wg} exceeds device limit {}",
            limits.max_compute_workgroups_per_dimension
        );
        let flags = u32::from(params.backface_cull) | (u32::from(params.reversed_z) << 1);
        let hiz_mips = hiz_view.map_or(0, |v| params.hiz_mips.min(v.texture().mip_level_count()));
        let u = CullUniform {
            model: params.model,
            inv_model: crate::matrix::invert(&params.model),
            hiz_vp: params.hiz_vp,
            planes: params.frustum_planes,
            camera_pos: [params.camera_pos[0], params.camera_pos[1], params.camera_pos[2], 0.0],
            counts_flags: [
                meshlet_count,
                u32::from(params.hiz_enabled && hiz_view.is_some()),
                hiz_mips,
                flags,
            ],
        };
        encode_copy(device, encoder, &self.uniform, bytemuck::bytes_of(&u));
        encode_copy(device, encoder, &self.stats, bytemuck::bytes_of(&MeshletCullStats::default()));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meshlet_cull_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: meshlets_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(req_m_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: draws_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(req_d_bytes),
                    }),
                },
                wgpu::BindGroupEntry { binding: 3, resource: self.stats.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(hiz_view.unwrap_or(&self.dummy_hiz_view)),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("meshlet_cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(wg, 1, 1);
        Ok(())
    }

    #[allow(dead_code)]
    fn _keep_alive(&self) -> &wgpu::Texture {
        &self.dummy_hiz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_meshlets_respects_limits_and_preserves_all_triangles() {
        let n = 20u32;
        let mut pos = Vec::new();
        for z in 0..=n {
            for x in 0..=n {
                pos.push([x as f32 - 10.0, 0.0, z as f32 - 10.0]);
            }
        }
        let mut idx = Vec::new();
        for z in 0..n {
            for x in 0..n {
                let i0 = z * (n + 1) + x;
                idx.extend_from_slice(&[i0, i0 + (n + 1), i0 + 1, i0 + 1, i0 + (n + 1), i0 + (n + 1) + 1]);
            }
        }
        let built = build_meshlets(&pos, &idx, 64, 124).unwrap();
        assert_eq!(built.indices.len(), idx.len(), "every input triangle must be preserved exactly once");
        assert!(built.meshlets.len() >= 7, "800 tris at max 124/cluster must yield >=7 meshlets");
        for m in &built.meshlets {
            assert!(m.meta[0] <= 64 && m.counts[3] <= 124, "meshlet violated vertex/tri cap: {:?} {:?}", m.meta, m.counts);
            assert!(m.cone_axis_cutoff[1] > 0.99 && m.cone_axis_cutoff[3] < 0.01, "flat grid normal cone must be sharp +Y");
        }
    }

    #[test]
    fn gpu_meshlet_culler_and_nonzero_index_indirect_draw_witness() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU adapter required");
        // Default device WITHOUT INDIRECT_FIRST_INSTANCE proves baseline WebGPU compatibility!
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

        // Place the VISIBLE meshlet at index 1 (NOT index 0!) so indirect draw at index > 0 is proven!
        // 0: Back-facing in frustum at (0, 0, -5), normal -Z -> BACKFACE CONE CULLED
        // 1: Front-facing in frustum at (0, 0, -5), normal +Z -> VISIBLE (index > 0!)
        // 2: Off-frustum at (50, 0, -5), normal +Z -> FRUSTUM CULLED
        // 3: Front-facing in frustum at (0, 0, -15) behind Hi-Z occluder -> HI-Z OCCLUDED
        let meshlets = [
            MeshletGpu { sphere: [0.0, 0.0, -5.0, 0.5], cone_axis_cutoff: [0.0, 0.0, -1.0, 0.0], counts: [0, 6, 0, 2], meta: [4, 0, 0, 0] },
            MeshletGpu { sphere: [0.0, 0.0, -5.0, 0.5], cone_axis_cutoff: [0.0, 0.0, 1.0, 0.0], counts: [6, 6, 0, 2], meta: [4, 0, 0, 0] },
            MeshletGpu { sphere: [50.0, 0.0, -5.0, 0.5], cone_axis_cutoff: [0.0, 0.0, 1.0, 0.0], counts: [12, 6, 0, 2], meta: [4, 0, 0, 0] },
            MeshletGpu { sphere: [0.0, 0.0, -15.0, 0.5], cone_axis_cutoff: [0.0, 0.0, 1.0, 0.0], counts: [18, 6, 0, 2], meta: [4, 0, 0, 0] },
        ];
        let m_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_meshlets"),
            size: std::mem::size_of_val(&meshlets) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        m_buf.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytemuck::cast_slice(&meshlets));
        m_buf.unmap();
        let d_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_draws"),
            size: (meshlets.len() * std::mem::size_of::<MeshletIndirectDraw>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let hiz_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_hiz"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &hiz_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::bytes_of(&0.85f32),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let hiz_view = hiz_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let (near, far) = (1.0f32, 100.0f32);
        let proj = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, far / (near - far), -1.0,
            0.0, 0.0, (far * near) / (near - far), 0.0,
        ];
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let planes = [
            [inv_sqrt2, 0.0, -inv_sqrt2, 0.0],
            [-inv_sqrt2, 0.0, -inv_sqrt2, 0.0],
            [0.0, inv_sqrt2, -inv_sqrt2, 0.0],
            [0.0, -inv_sqrt2, -inv_sqrt2, 0.0],
            [0.0, 0.0, -1.0, -near],
            [0.0, 0.0, 1.0, far],
        ];
        let id = crate::matrix::IDENTITY;
        let culler = MeshletCuller::new(&device);

        // Build a minimal render pass drawing meshlet 1 via `draw_indexed_indirect` on default device!
        let verts: [[f32; 4]; 4] = [[-0.5, -0.5, -5.0, 1.0], [0.5, -0.5, -5.0, 1.0], [0.5, 0.5, -5.0, 1.0], [-0.5, 0.5, -5.0, 1.0]];
        let indices: [u32; 24] = [
            0, 1, 2, 0, 2, 3, // meshlet 0
            0, 1, 2, 0, 2, 3, // meshlet 1 (visible!)
            0, 1, 2, 0, 2, 3, // meshlet 2
            0, 1, 2, 0, 2, 3, // meshlet 3
        ];
        let vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("m_vb"),
            size: std::mem::size_of_val(&verts) as u64,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: true,
        });
        vb.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytemuck::cast_slice(&verts));
        vb.unmap();
        let ib = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("m_ib"),
            size: std::mem::size_of_val(&indices) as u64,
            usage: wgpu::BufferUsages::INDEX,
            mapped_at_creation: true,
        });
        ib.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytemuck::cast_slice(&indices));
        ib.unmap();
        let draw_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("m_draw"),
            source: wgpu::ShaderSource::Wgsl(
                "@vertex fn vs(@location(0) p: vec4f) -> @builtin(position) vec4f { return vec4f(p.x, p.y, 0.5, 1.0); }
                 @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }".into(),
            ),
        });
        let draw_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("m_draw_pipe"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &draw_mod,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 16,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 0 }],
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &draw_mod,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let rt_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("m_rt"),
            size: wgpu::Extent3d { width: 16, height: 16, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let rt_view = rt_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        culler
            .cull(
                &device,
                &mut enc,
                &m_buf,
                &d_buf,
                4,
                Some(&hiz_view),
                &MeshletCullParams {
                    model: id,
                    hiz_vp: proj,
                    frustum_planes: planes,
                    camera_pos: [0.0, 0.0, 0.0],
                    hiz_enabled: true,
                    hiz_mips: 1,
                    backface_cull: true,
                    reversed_z: false,
                },
            )
            .unwrap();
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("meshlet_indirect_render"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &rt_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            rp.set_pipeline(&draw_pipe);
            rp.set_vertex_buffer(0, vb.slice(..));
            rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            for i in 0..4u64 {
                rp.draw_indexed_indirect(&d_buf, i * std::mem::size_of::<MeshletIndirectDraw>() as u64);
            }
        }

        let s_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("s_rb"),
            size: std::mem::size_of::<MeshletCullStats>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let d_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("d_rb"),
            size: d_buf.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bpr = 256u32;
        let img_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("img_rb"),
            size: u64::from(bpr * 16),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(culler.stats_buffer(), 0, &s_rb, 0, s_rb.size());
        enc.copy_buffer_to_buffer(&d_buf, 0, &d_rb, 0, d_rb.size());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &rt_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &img_rb,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(16) },
            },
            wgpu::Extent3d { width: 16, height: 16, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));

        let read = |b: &wgpu::Buffer| -> Vec<u8> {
            let s = b.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            s.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            rx.recv().unwrap().unwrap();
            let v = s.get_mapped_range().unwrap().to_vec();
            b.unmap();
            v
        };
        let st_bytes = read(&s_rb);
        let dr_bytes = read(&d_rb);
        let img = read(&img_rb);
        let st: &MeshletCullStats = bytemuck::from_bytes(&st_bytes);
        let dr: &[MeshletIndirectDraw] = bytemuck::cast_slice(&dr_bytes);

        assert_eq!(st.visible, 1, "only meshlet 1 should survive: {st:?}");
        assert_eq!(st.backface_culled, 1, "meshlet 0 backface culled: {st:?}");
        assert_eq!(st.frustum_culled, 1, "meshlet 2 frustum culled: {st:?}");
        assert_eq!(st.hiz_culled, 1, "meshlet 3 Hi-Z culled: {st:?}");
        assert_eq!([dr[0].instance_count, dr[1].instance_count, dr[2].instance_count, dr[3].instance_count], [0, 1, 0, 0]);
        assert!(dr.iter().all(|d| d.first_instance == 0), "first_instance must stay 0 for baseline WebGPU");
        // Verify meshlet 1 actually rasterized green pixels at screen center!
        let center_o = (8 * bpr + 8 * 4) as usize;
        assert_eq!(img[center_o + 1], 255, "indirect draw of meshlet index 1 must render green pixels");
    }
}
