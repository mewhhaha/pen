//! GPU Meshlets & Cluster Culling:
//! - Partitions indexed triangle meshes into compact clusters (`<= 64` verts, `<= 124` tris)
//! - Computes conservative `f32`-rounded bounding spheres & local-space normal cones
//! - GPU compute culler (`MESHLET_CULL_WGSL`) performing exact local-space frustum & normal-cone
//!   backface culling (conservative under arbitrary shear/scale) and full-footprint Hi-Z culling
//!   into baseline-WebGPU `DrawIndexedIndirect` records (`first_instance = 0`).
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "GPU meshlet buffer indices")]

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{Result, ensure};

pub const MESHLET_CULL_WGSL: &str = crate::wgsl!("meshlet_cull.wgsl");

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

fn expand_bits_10(v: u32) -> u32 {
    let mut x = v & 0x3ff;
    x = (x | (x << 16)) & 0x0300_00ff;
    x = (x | (x << 8)) & 0x0300_f00f;
    x = (x | (x << 4)) & 0x030c_30c3;
    x = (x | (x << 2)) & 0x0924_9249;
    x
}

fn morton3_10(p: [f32; 3], min_p: [f32; 3], inv_span: [f32; 3]) -> u32 {
    let q = |axis: usize| -> u32 {
        let t = ((p[axis] - min_p[axis]) * inv_span[axis]).clamp(0.0, 1.0);
        (t * 1023.0).round() as u32
    };
    expand_bits_10(q(0)) | (expand_bits_10(q(1)) << 1) | (expand_bits_10(q(2)) << 2)
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
    let mut cmin = [f32::INFINITY; 3];
    let mut cmax = [f32::NEG_INFINITY; 3];
    for t in 0..tri_count {
        let (a, b, c) = (indices[t * 3] as usize, indices[t * 3 + 1] as usize, indices[t * 3 + 2] as usize);
        adj[a].push(t);
        adj[b].push(t);
        adj[c].push(t);
        let (pa, pb, pc) = (positions[a], positions[b], positions[c]);
        let tc = [
            (pa[0] + pb[0] + pc[0]) / 3.0,
            (pa[1] + pb[1] + pc[1]) / 3.0,
            (pa[2] + pb[2] + pc[2]) / 3.0,
        ];
        for axis in 0..3 {
            cmin[axis] = cmin[axis].min(tc[axis]);
            cmax[axis] = cmax[axis].max(tc[axis]);
        }
        tri_centroids.push(tc);
    }
    let inv_span = [
        if cmax[0] > cmin[0] { 1.0 / (cmax[0] - cmin[0]) } else { 0.0 },
        if cmax[1] > cmin[1] { 1.0 / (cmax[1] - cmin[1]) } else { 0.0 },
        if cmax[2] > cmin[2] { 1.0 / (cmax[2] - cmin[2]) } else { 0.0 },
    ];
    let mut morton_order: Vec<(u32, usize)> = tri_centroids
        .iter()
        .enumerate()
        .map(|(t, &tc)| (morton3_10(tc, cmin, inv_span), t))
        .collect();
    morton_order.sort_unstable();
    let morton_tris: Vec<usize> = morton_order.into_iter().map(|(_, t)| t).collect();

    let mut emitted = vec![false; tri_count];
    let mut in_frontier = vec![false; tri_count];
    let mut vert_stamp = vec![0u32; positions.len()];
    let mut stamp = 0u32;
    let mut emitted_count = 0usize;
    let mut meshlets = Vec::new();
    let mut out_indices = Vec::with_capacity(indices.len());
    let mut next_seed = 0usize;
    let mut cluster_tris: Vec<usize> = Vec::with_capacity(max_triangles);
    let mut cluster_verts: Vec<u32> = Vec::with_capacity(max_vertices);
    let mut frontier: Vec<usize> = Vec::new();
    let mut normals: Vec<[f64; 3]> = Vec::with_capacity(max_triangles);

    while emitted_count < tri_count {
        while next_seed < tri_count && emitted[morton_tris[next_seed]] {
            next_seed += 1;
        }
        if next_seed >= tri_count {
            break;
        }
        let seed_tri = morton_tris[next_seed];
        stamp += 1;
        cluster_tris.clear();
        cluster_verts.clear();
        frontier.clear();
        let mut cx = 0.0f64;
        let mut cy = 0.0f64;
        let mut cz = 0.0f64;

        let push_tri = |t: usize,
                        stamp: u32,
                        emitted: &mut [bool],
                        in_frontier: &mut [bool],
                        vert_stamp: &mut [u32],
                        cluster_tris: &mut Vec<usize>,
                        cluster_verts: &mut Vec<u32>,
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
                if vert_stamp[v as usize] != stamp {
                    vert_stamp[v as usize] = stamp;
                    cluster_verts.push(v);
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
            seed_tri,
            stamp,
            &mut emitted,
            &mut in_frontier,
            &mut vert_stamp,
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
                let mut added = usize::from(vert_stamp[v0 as usize] != stamp);
                if v1 != v0 && vert_stamp[v1 as usize] != stamp {
                    added += 1;
                }
                if v2 != v0 && v2 != v1 && vert_stamp[v2 as usize] != stamp {
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
            // On flat-shaded / disconnected meshes (unshared vertices per face), shared-vertex
            // adjacency runs dry immediately; pack spatially adjacent triangles in Morton order
            // so meshes never degenerate into one meshlet per triangle.
            if best.is_none() && cluster_verts.len() + 3 <= max_vertices {
                while next_seed < tri_count && emitted[morton_tris[next_seed]] {
                    next_seed += 1;
                }
                let mut scanned = 0usize;
                let mut cursor = next_seed;
                while cursor < tri_count && scanned < 64 {
                    let cand = morton_tris[cursor];
                    cursor += 1;
                    if emitted[cand] {
                        continue;
                    }
                    scanned += 1;
                    let (v0, v1, v2) = (indices[cand * 3], indices[cand * 3 + 1], indices[cand * 3 + 2]);
                    let mut added = usize::from(vert_stamp[v0 as usize] != stamp);
                    if v1 != v0 && vert_stamp[v1 as usize] != stamp {
                        added += 1;
                    }
                    if v2 != v0 && v2 != v1 && vert_stamp[v2 as usize] != stamp {
                        added += 1;
                    }
                    if cluster_verts.len() + added > max_vertices {
                        continue;
                    }
                    let tc = tri_centroids[cand];
                    let d2 = (f64::from(tc[0]) - mean[0]).powi(2)
                        + (f64::from(tc[1]) - mean[1]).powi(2)
                        + (f64::from(tc[2]) - mean[2]).powi(2);
                    if best.is_none_or(|(_, s)| d2 < s) {
                        best = Some((cand, d2));
                    }
                }
            }
            let Some((chosen, _)) = best else {
                break;
            };
            push_tri(
                chosen,
                stamp,
                &mut emitted,
                &mut in_frontier,
                &mut vert_stamp,
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

        normals.clear();
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

/// One draw queued for batched GPU meshlet culling in a single compute pass.
pub struct MeshletBatchEntry<'a> {
    pub mesh_key: u64,
    pub draws_gen: u64,
    pub meshlets_buf: &'a wgpu::Buffer,
    pub draws_buf: &'a wgpu::Buffer,
    pub meshlet_count: u32,
    pub params: MeshletCullParams,
}

/// One draw shared by both phases of GPU meshlet culling.
/// Advance `draws_gen` whenever any of the bound buffers changes for this mesh key.
pub struct MeshletTwoPhaseBatchEntry<'a> {
    pub mesh_key: u64,
    pub draws_gen: u64,
    pub meshlets_buf: &'a wgpu::Buffer,
    pub draws_phase1_buf: &'a wgpu::Buffer,
    pub draws_phase2_buf: &'a wgpu::Buffer,
    /// Storage buffer with one `u32` candidate flag per meshlet.
    pub phase2_flags_buf: &'a wgpu::Buffer,
    pub meshlet_count: u32,
    pub params: MeshletCullParams,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CullPhase {
    First,
    Second,
}

impl CullPhase {
    fn select<T>(self, first: T, second: T) -> T {
        match self {
            Self::First => first,
            Self::Second => second,
        }
    }
}

struct TwoPhasePipelines {
    bgl: wgpu::BindGroupLayout,
    first: wgpu::ComputePipeline,
    second: wgpu::ComputePipeline,
}

type BatchGroupKey = (u64, u64, u32, Option<u64>);

struct BatchBuffers {
    uniform: wgpu::Buffer,
    capacity: usize,
    bytes: Vec<u8>,
    groups: HashMap<BatchGroupKey, wgpu::BindGroup>,
}

impl BatchBuffers {
    fn uniform_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meshlet_cull_u"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn new(device: &wgpu::Device, stride: u64) -> Self {
        let capacity = 16;
        Self {
            uniform: Self::uniform_buffer(device, stride * capacity as u64),
            capacity,
            bytes: Vec::new(),
            groups: HashMap::new(),
        }
    }
}

struct CullDispatch<'a> {
    meshlets: &'a wgpu::Buffer,
    draws: &'a wgpu::Buffer,
    flags: Option<&'a wgpu::Buffer>,
    count: u32,
    params: &'a MeshletCullParams,
}

impl CullDispatch<'_> {
    fn storage_sizes(&self) -> [u64; 3] {
        let count = u64::from(self.count);
        [
            count * std::mem::size_of::<MeshletGpu>() as u64,
            count * std::mem::size_of::<MeshletIndirectDraw>() as u64,
            count * std::mem::size_of::<u32>() as u64,
        ]
    }

    fn validate(&self, limits: &wgpu::Limits) -> Result<()> {
        let [meshlet_bytes, draw_bytes, flag_bytes] = self.storage_sizes();
        for (buffer, bytes, name) in [
            (Some(self.meshlets), meshlet_bytes, "meshlets_buf"),
            (Some(self.draws), draw_bytes, "draws_buf"),
            (self.flags, flag_bytes, "phase2_flags_buf"),
        ] {
            if let Some(buffer) = buffer {
                ensure!(
                    buffer.usage().contains(wgpu::BufferUsages::STORAGE) && buffer.size() >= bytes,
                    "{name} must have STORAGE usage and at least {bytes} bytes"
                );
                ensure!(
                    bytes <= limits.max_storage_buffer_binding_size,
                    "{name} exceeds device max_storage_buffer_binding_size"
                );
            }
        }
        let groups = self.count.div_ceil(64);
        ensure!(
            groups <= limits.max_compute_workgroups_per_dimension,
            "meshlet workgroup count {groups} exceeds device limit {}",
            limits.max_compute_workgroups_per_dimension
        );
        Ok(())
    }

    fn bind_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        stats: &wgpu::Buffer,
        hiz: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let buffer_entry = |binding, buffer, size| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: wgpu::BufferSize::new(size),
            }),
        };
        let [meshlet_bytes, draw_bytes, flag_bytes] = self.storage_sizes();
        let entries = [
            buffer_entry(0, uniform, std::mem::size_of::<CullUniform>() as u64),
            buffer_entry(1, self.meshlets, meshlet_bytes),
            buffer_entry(2, self.draws, draw_bytes),
            wgpu::BindGroupEntry { binding: 3, resource: stats.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(hiz) },
            buffer_entry(5, self.flags.unwrap_or(self.draws), flag_bytes),
        ];
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meshlet_cull_bg"),
            layout,
            entries: &entries[..5 + usize::from(self.flags.is_some())],
        })
    }
}

const BINDINGS: [wgpu::BindGroupLayoutEntry; 6] = [
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
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
    wgpu::BindGroupLayoutEntry {
        binding: 5,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
];

/// GPU compute meshlet cluster culler (local-space frustum + shear-safe normal-cone backface + full-footprint Hi-Z).
pub struct MeshletCuller {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    shader: wgpu::ShaderModule,
    uniform_stride: u64,
    batch: BatchBuffers,
    batch_phase1: Option<BatchBuffers>,
    batch_phase2: Option<BatchBuffers>,
    two_phase: OnceLock<TwoPhasePipelines>,
    stats: wgpu::Buffer,
    dummy_hiz: wgpu::Texture,
    dummy_hiz_view: wgpu::TextureView,
}

impl MeshletCuller {
    pub fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("meshlet_cull_bgl"),
            entries: &BINDINGS[..5],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("meshlet_cull_shader"),
            source: wgpu::ShaderSource::Wgsl(MESHLET_CULL_WGSL.into()),
        });
        let layout = Self::pipeline_layout(device, &bgl);
        let pipeline = Self::pipeline(device, &shader, &layout, "cull_meshlets");
        let align = u64::from(device.limits().min_uniform_buffer_offset_alignment).max(1);
        let uniform_stride = (std::mem::size_of::<CullUniform>() as u64).next_multiple_of(align);
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
        Self {
            pipeline,
            bgl,
            shader,
            uniform_stride,
            batch: BatchBuffers::new(device, uniform_stride),
            batch_phase1: None,
            batch_phase2: None,
            two_phase: OnceLock::new(),
            stats,
            dummy_hiz,
            dummy_hiz_view,
        }
    }

    fn pipeline_layout(device: &wgpu::Device, bgl: &wgpu::BindGroupLayout) -> wgpu::PipelineLayout {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("meshlet_cull_pl"),
            bind_group_layouts: &[Some(bgl)],
            immediate_size: 0,
        })
    }

    fn pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        entry_point: &str,
    ) -> wgpu::ComputePipeline {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(layout),
            module: shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn two_phase_pipelines<'a>(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        pipelines: &'a OnceLock<TwoPhasePipelines>,
    ) -> &'a TwoPhasePipelines {
        pipelines.get_or_init(|| {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("meshlet_cull_twophase_bgl"),
                entries: &BINDINGS,
            });
            let layout = Self::pipeline_layout(device, &bgl);
            TwoPhasePipelines {
                first: Self::pipeline(device, shader, &layout, "cull_meshlets_phase1"),
                second: Self::pipeline(device, shader, &layout, "cull_meshlets_phase2"),
                bgl,
            }
        })
    }

    pub fn stats_buffer(&self) -> &wgpu::Buffer {
        &self.stats
    }

    fn build_uniform(
        meshlet_count: u32,
        hiz_view: Option<&wgpu::TextureView>,
        params: &MeshletCullParams,
    ) -> CullUniform {
        let flags = u32::from(params.backface_cull) | (u32::from(params.reversed_z) << 1);
        let hiz_mips = hiz_view.map_or(0, |v| params.hiz_mips.min(v.texture().mip_level_count()));
        CullUniform {
            model: params.model,
            inv_model: crate::matrix::invert(&params.model),
            hiz_vp: params.hiz_vp,
            planes: params.frustum_planes,
            camera_pos: [params.camera_pos[0], params.camera_pos[1], params.camera_pos[2], 0.0],
            counts_flags: [meshlet_count, u32::from(params.hiz_enabled && hiz_view.is_some()), hiz_mips, flags],
        }
    }

    /// Dispatches GPU meshlet cluster culling for an entire frame's solo mesh draws inside
    /// a single compute pass, uploading all draw uniforms in one `queue.write_buffer` call
    /// and reusing cached bind groups across frames.
    #[allow(clippy::too_many_arguments)]
    pub fn cull_batch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        hiz: Option<(&wgpu::TextureView, u64)>,
        entries: &[MeshletBatchEntry<'_>],
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let dispatches = entries.iter().map(|entry| {
            (
                entry.mesh_key,
                entry.draws_gen,
                CullDispatch {
                    meshlets: entry.meshlets_buf,
                    draws: entry.draws_buf,
                    flags: None,
                    count: entry.meshlet_count,
                    params: &entry.params,
                },
            )
        });
        Self::dispatch_batch(
            device,
            queue,
            encoder,
            hiz,
            dispatches,
            timestamp_writes,
            &self.bgl,
            &self.pipeline,
            &mut self.batch,
            self.uniform_stride,
            &self.stats,
            &self.dummy_hiz_view,
            true,
        )
    }

    /// Clears the cull stats and emits the draws that survive the previous frame's Hi-Z.
    /// Set each entry's `params.hiz_vp` to the projection used to build `prev_hiz`.
    /// Batch all Phase 1 draws for this culler into one call per queue submission.
    #[allow(clippy::too_many_arguments)]
    pub fn cull_batch_phase1(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        prev_hiz: Option<(&wgpu::TextureView, u64)>,
        entries: &[MeshletTwoPhaseBatchEntry<'_>],
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<()> {
        self.cull_two_phase_batch(device, queue, encoder, prev_hiz, entries, timestamp_writes, CullPhase::First)
    }

    /// Retests Phase 1 candidates and adds their results to the existing cull stats.
    /// Set each entry's `params.hiz_vp` to the current projection used to build `curr_hiz`.
    /// When current Hi-Z is unavailable, all candidates are emitted.
    /// Batch all Phase 2 draws for this culler into one call per queue submission.
    #[allow(clippy::too_many_arguments)]
    pub fn cull_batch_phase2(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        curr_hiz: Option<(&wgpu::TextureView, u64)>,
        entries: &[MeshletTwoPhaseBatchEntry<'_>],
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<()> {
        self.cull_two_phase_batch(device, queue, encoder, curr_hiz, entries, timestamp_writes, CullPhase::Second)
    }

    #[allow(clippy::too_many_arguments)]
    fn cull_two_phase_batch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        hiz: Option<(&wgpu::TextureView, u64)>,
        entries: &[MeshletTwoPhaseBatchEntry<'_>],
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
        phase: CullPhase,
    ) -> Result<()> {
        let clear_stats = phase == CullPhase::First;
        if entries.is_empty() {
            if clear_stats {
                encoder.clear_buffer(&self.stats, 0, None);
            }
            return Ok(());
        }

        let pipelines = Self::two_phase_pipelines(device, &self.shader, &self.two_phase);
        // Queue writes happen before encoder execution. Each phase needs its own uniform slots.
        let batch = phase
            .select(&mut self.batch_phase1, &mut self.batch_phase2)
            .get_or_insert_with(|| BatchBuffers::new(device, self.uniform_stride));
        let dispatches = entries.iter().map(|entry| {
            (
                entry.mesh_key,
                entry.draws_gen,
                CullDispatch {
                    meshlets: entry.meshlets_buf,
                    draws: phase.select(entry.draws_phase1_buf, entry.draws_phase2_buf),
                    flags: Some(entry.phase2_flags_buf),
                    count: entry.meshlet_count,
                    params: &entry.params,
                },
            )
        });
        Self::dispatch_batch(
            device,
            queue,
            encoder,
            hiz,
            dispatches,
            timestamp_writes,
            &pipelines.bgl,
            phase.select(&pipelines.first, &pipelines.second),
            batch,
            self.uniform_stride,
            &self.stats,
            &self.dummy_hiz_view,
            clear_stats,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_batch<'a>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        hiz: Option<(&wgpu::TextureView, u64)>,
        dispatches: impl ExactSizeIterator<Item = (u64, u64, CullDispatch<'a>)> + Clone,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
        layout: &wgpu::BindGroupLayout,
        pipeline: &wgpu::ComputePipeline,
        batch: &mut BatchBuffers,
        uniform_stride: u64,
        stats: &wgpu::Buffer,
        dummy_hiz_view: &wgpu::TextureView,
        clear_stats: bool,
    ) -> Result<()> {
        let limits = device.limits();
        let stride = uniform_stride as usize;
        let count = dispatches.len();
        u32::try_from((count - 1) as u64 * uniform_stride)?;
        if count > batch.capacity {
            let capacity = count.next_power_of_two();
            ensure!(
                capacity as u64 * uniform_stride <= limits.max_buffer_size,
                "meshlet cull uniform buffer exceeds device max_buffer_size"
            );
            batch.uniform = BatchBuffers::uniform_buffer(device, capacity as u64 * uniform_stride);
            batch.capacity = capacity;
            batch.groups.clear();
        }
        if batch.groups.len() > 2048 {
            batch.groups.clear();
        }
        let total_bytes = count * stride;
        if batch.bytes.len() < total_bytes {
            batch.bytes.resize(total_bytes, 0);
        }
        let hiz_view = hiz.map(|(view, _)| view);
        let hiz_id = hiz.map(|(_, id)| id);
        let bound_hiz = hiz_view.unwrap_or(dummy_hiz_view);
        let u_size = std::mem::size_of::<CullUniform>();
        for (i, (mesh_key, draws_gen, dispatch)) in dispatches.clone().enumerate() {
            if dispatch.count == 0 {
                continue;
            }
            dispatch.validate(&limits)?;
            let uniform = Self::build_uniform(dispatch.count, hiz_view, dispatch.params);
            let offset = i * stride;
            batch.bytes[offset..offset + u_size].copy_from_slice(bytemuck::bytes_of(&uniform));
            let key = (mesh_key, draws_gen, dispatch.count, hiz_id);
            batch
                .groups
                .entry(key)
                .or_insert_with(|| dispatch.bind_group(device, layout, &batch.uniform, stats, bound_hiz));
        }
        queue.write_buffer(&batch.uniform, 0, &batch.bytes[..total_bytes]);
        if clear_stats {
            encoder.clear_buffer(stats, 0, None);
        }
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("meshlet_cull_batch"), timestamp_writes });
        pass.set_pipeline(pipeline);
        for (i, (mesh_key, draws_gen, dispatch)) in dispatches.enumerate() {
            if dispatch.count == 0 {
                continue;
            }
            let group = &batch.groups[&(mesh_key, draws_gen, dispatch.count, hiz_id)];
            pass.set_bind_group(0, group, &[(i as u64 * uniform_stride) as u32]);
            pass.dispatch_workgroups(dispatch.count.div_ceil(64), 1, 1);
        }
        Ok(())
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
        let dispatch =
            CullDispatch { meshlets: meshlets_buf, draws: draws_buf, flags: None, count: meshlet_count, params };
        self.dispatch_single(device, encoder, &dispatch, hiz_view, None)
    }

    /// Clears the cull stats and culls one draw against the previous frame's Hi-Z.
    #[allow(clippy::too_many_arguments)]
    pub fn cull_phase1(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        meshlets_buf: &wgpu::Buffer,
        draws_phase1_buf: &wgpu::Buffer,
        phase2_flags_buf: &wgpu::Buffer,
        meshlet_count: u32,
        prev_hiz: Option<&wgpu::TextureView>,
        params: &MeshletCullParams,
    ) -> Result<()> {
        let dispatch = CullDispatch {
            meshlets: meshlets_buf,
            draws: draws_phase1_buf,
            flags: Some(phase2_flags_buf),
            count: meshlet_count,
            params,
        };
        self.dispatch_single(device, encoder, &dispatch, prev_hiz, Some(CullPhase::First))
    }

    /// Retests one draw's Phase 1 candidates against the current frame's Hi-Z and accumulates stats.
    #[allow(clippy::too_many_arguments)]
    pub fn cull_phase2(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        meshlets_buf: &wgpu::Buffer,
        draws_phase2_buf: &wgpu::Buffer,
        phase2_flags_buf: &wgpu::Buffer,
        meshlet_count: u32,
        curr_hiz: Option<&wgpu::TextureView>,
        params: &MeshletCullParams,
    ) -> Result<()> {
        let dispatch = CullDispatch {
            meshlets: meshlets_buf,
            draws: draws_phase2_buf,
            flags: Some(phase2_flags_buf),
            count: meshlet_count,
            params,
        };
        self.dispatch_single(device, encoder, &dispatch, curr_hiz, Some(CullPhase::Second))
    }

    fn dispatch_single(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dispatch: &CullDispatch<'_>,
        hiz_view: Option<&wgpu::TextureView>,
        phase: Option<CullPhase>,
    ) -> Result<()> {
        if dispatch.count == 0 {
            if phase == Some(CullPhase::First) {
                encoder.clear_buffer(&self.stats, 0, None);
            }
            return Ok(());
        }
        dispatch.validate(&device.limits())?;
        let data = Self::build_uniform(dispatch.count, hiz_view, dispatch.params);
        let uniform;
        let (layout, pipeline, bound_uniform) = if let Some(phase) = phase {
            let pipelines = Self::two_phase_pipelines(device, &self.shader, &self.two_phase);
            uniform = wgpu::util::DeviceExt::create_buffer_init(
                device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("meshlet_cull_single_u"),
                    contents: bytemuck::bytes_of(&data),
                    usage: wgpu::BufferUsages::UNIFORM,
                },
            );
            (&pipelines.bgl, phase.select(&pipelines.first, &pipelines.second), &uniform)
        } else {
            encode_copy(device, encoder, &self.batch.uniform, bytemuck::bytes_of(&data));
            (&self.bgl, &self.pipeline, &self.batch.uniform)
        };
        if phase != Some(CullPhase::Second) {
            encoder.clear_buffer(&self.stats, 0, None);
        }
        let group =
            dispatch.bind_group(device, layout, bound_uniform, &self.stats, hiz_view.unwrap_or(&self.dummy_hiz_view));
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("meshlet_cull"), timestamp_writes: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[0]);
        pass.dispatch_workgroups(dispatch.count.div_ceil(64), 1, 1);
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
    fn build_meshlets_packs_disconnected_flat_shaded_triangles() {
        let mut pos = Vec::new();
        let mut idx = Vec::new();
        // 400 disconnected flat-shaded triangles (3 unique vertices each = 1200 vertices).
        for z in 0..20u32 {
            for x in 0..20u32 {
                let base = pos.len() as u32;
                let fx = x as f32;
                let fz = z as f32;
                pos.push([fx, 0.0, fz]);
                pos.push([fx, 0.0, fz + 0.8]);
                pos.push([fx + 0.8, 0.0, fz]);
                idx.extend_from_slice(&[base, base + 1, base + 2]);
            }
        }
        let built = build_meshlets(&pos, &idx, 64, 124).unwrap();
        assert_eq!(built.indices.len(), idx.len(), "every disconnected triangle must be preserved");
        assert!(
            built.meshlets.len() <= 25,
            "400 flat-shaded tris at 64 verts/cluster (~21 tris each) should pack into <=25 meshlets, got {}",
            built.meshlets.len()
        );
        for m in &built.meshlets {
            assert!(m.meta[0] <= 64 && m.counts[3] <= 124);
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

    #[test]
    fn gpu_two_phase_meshlet_occlusion_recovers_disoccluded_clusters_same_frame_witness() {
        use wgpu::util::{BufferInitDescriptor, DeviceExt};

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU adapter required");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let mut culler = MeshletCuller::new(&device);

        let (rings, sectors) = (32u32, 40u32);
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        for ring in 0..=rings {
            let (sin_phi, cos_phi) = (std::f32::consts::PI * ring as f32 / rings as f32).sin_cos();
            for sector in 0..sectors {
                let (sin_theta, cos_theta) = (std::f32::consts::TAU * sector as f32 / sectors as f32).sin_cos();
                positions.push([0.075 * sin_phi * cos_theta, 0.075 * sin_phi * sin_theta, 0.075 * cos_phi]);
                if ring < rings {
                    let a = ring * sectors + sector;
                    let b = ring * sectors + (sector + 1) % sectors;
                    indices.extend_from_slice(&[a, a + sectors, b, b, a + sectors, b + sectors]);
                }
            }
        }
        let sphere = build_meshlets(&positions, &indices, 32, 32).unwrap().meshlets;
        assert!(sphere.len() > 64, "exercise more than one workgroup per sphere");
        let mut recovered = sphere.clone();
        // Cover a Phase 1 draw, a candidate that the current VP projects offscreen, and a frustum rejection.
        // If the Phase 2 uniform clobbers Phase 1, that candidate survives Phase 1 incorrectly.
        for (x, z) in [(0.0, -0.47), (1.0, 0.0), (2.0, 0.0)] {
            recovered.push(MeshletGpu {
                sphere: [x, 0.0, z, 0.01],
                cone_axis_cutoff: [0.0, 0.0, 1.0, 1.0],
                counts: [3, 6, 0, 2],
                meta: [4, 0, 0, 0],
            });
        }
        let buffer = |size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage, mapped_at_creation: false })
        };
        let fixtures = [(&sphere, -0.65), (&recovered, -0.25)].map(|(meshlets, x)| {
            let count = meshlets.len() as u32;
            let draws_size = u64::from(count) * std::mem::size_of::<MeshletIndirectDraw>() as u64;
            let draws_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC;
            // Fill both outputs with nonzero data to catch untouched flags or indirect draws.
            let nonzero_buffer = |words, usage| {
                device.create_buffer_init(&BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&vec![1u32; words]),
                    usage,
                })
            };
            (
                x,
                count,
                device.create_buffer_init(&BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(meshlets),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
                buffer(draws_size, draws_usage),
                nonzero_buffer(draws_size as usize / 4, draws_usage),
                nonzero_buffer(count as usize, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            )
        });
        let hiz = |depth: [f32; 2]| {
            let size = wgpu::Extent3d { width: 2, height: 1, depth_or_array_layers: 1 };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                texture.as_image_copy(),
                bytemuck::cast_slice(&depth),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8), rows_per_image: None },
                size,
            );
            texture.create_view(&wgpu::TextureViewDescriptor::default())
        };

        for reversed_z in [false, true] {
            let (near_depth, clear_depth) = if reversed_z { (0.9, 0.0) } else { (0.1, 1.0) };
            let prev_hiz = hiz([near_depth; 2]);
            let curr_hiz = hiz([near_depth, clear_depth]);
            let mut vp = crate::matrix::IDENTITY;
            if reversed_z {
                vp[10] = -1.0;
                vp[14] = 1.0;
            }
            let mut entries = std::array::from_fn::<_, 2, _>(|i| {
                let (x, count, meshlets, draws1, draws2, flags) = &fixtures[i];
                let mut model = crate::matrix::IDENTITY;
                model[12] = *x;
                model[14] = 0.5;
                let mut planes = [[0.0; 4]; 6];
                planes[0] = [-1.0, 0.0, 0.0, 1.0];
                MeshletTwoPhaseBatchEntry {
                    mesh_key: i as u64,
                    draws_gen: 0,
                    meshlets_buf: meshlets,
                    draws_phase1_buf: draws1,
                    draws_phase2_buf: draws2,
                    phase2_flags_buf: flags,
                    meshlet_count: *count,
                    params: MeshletCullParams {
                        model,
                        hiz_vp: vp,
                        frustum_planes: planes,
                        camera_pos: [0.0, 0.0, -1.0],
                        hiz_enabled: true,
                        hiz_mips: 1,
                        backface_cull: true,
                        reversed_z,
                    },
                }
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let readback = |encoder: &mut wgpu::CommandEncoder, source: &wgpu::Buffer| {
                let dest = buffer(source.size(), wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ);
                encoder.copy_buffer_to_buffer(source, 0, &dest, 0, source.size());
                dest
            };
            let hiz_id = u64::from(reversed_z);
            culler.cull_batch_phase1(&device, &queue, &mut encoder, Some((&prev_hiz, hiz_id)), &entries, None).unwrap();
            let stats1 = readback(&mut encoder, culler.stats_buffer());
            for entry in &mut entries {
                entry.params.hiz_vp[12] = 0.5;
            }
            culler.cull_batch_phase2(&device, &queue, &mut encoder, Some((&curr_hiz, hiz_id)), &entries, None).unwrap();
            let stats2 = readback(&mut encoder, culler.stats_buffer());
            let draw_readbacks = entries.map(|entry| {
                (
                    readback(&mut encoder, entry.draws_phase1_buf),
                    readback(&mut encoder, entry.draws_phase2_buf),
                    readback(&mut encoder, entry.phase2_flags_buf),
                )
            });
            queue.submit(Some(encoder.finish()));

            let read = |buffer: &wgpu::Buffer| {
                let slice = buffer.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();
                let bytes = slice.get_mapped_range().unwrap().to_vec();
                buffer.unmap();
                bytes
            };
            let mut candidates = [0u32; 2];
            let mut backface_culled = 0;
            for (i, (draws1, draws2, flags)) in draw_readbacks.iter().enumerate() {
                let draws1 = read(draws1);
                let draws2 = read(draws2);
                let flags = read(flags);
                let draws1 = bytemuck::cast_slice::<_, MeshletIndirectDraw>(&draws1);
                let draws2 = bytemuck::cast_slice::<_, MeshletIndirectDraw>(&draws2);
                let flags = bytemuck::cast_slice::<_, u32>(&flags);
                for (idx, ((draw1, draw2), &flag)) in draws1.iter().zip(draws2).zip(flags).enumerate() {
                    let already_visible = i == 1 && idx == sphere.len();
                    assert!(flag <= 1);
                    assert_eq!(draw1.instance_count, u32::from(already_visible));
                    assert_eq!(draw2.instance_count, if i == 0 { 0 } else { flag });
                    assert_eq!(draw1.first_instance, 0);
                    assert_eq!(draw2.first_instance, 0);
                    if flag == 1 {
                        assert_eq!(draw2.first_index, draw1.first_index);
                        assert_eq!(draw2.index_count, draw1.index_count);
                        assert_eq!(draw2.base_vertex, draw1.base_vertex);
                    }
                    candidates[i] += flag;
                    backface_culled += u32::from(flag == 0 && idx < sphere.len());
                }
                assert!(candidates[i] > 1, "sphere must have multiple Hi-Z candidates");
                if i == 1 {
                    assert_eq!(&flags[sphere.len()..], &[0, 1, 0]);
                }
            }
            assert!(backface_culled > 0, "backfaces must skip both draws");
            let stats1 = bytemuck::pod_read_unaligned::<MeshletCullStats>(&read(&stats1));
            let stats2 = bytemuck::pod_read_unaligned::<MeshletCullStats>(&read(&stats2));
            assert_eq!(
                [stats1.visible, stats1.frustum_culled, stats1.backface_culled, stats1.hiz_culled],
                [1, 1, backface_culled, 0]
            );
            assert_eq!(
                [stats2.visible, stats2.frustum_culled, stats2.backface_culled, stats2.hiz_culled],
                [1 + candidates[1], 1, backface_culled, candidates[0]]
            );
        }
    }
}
