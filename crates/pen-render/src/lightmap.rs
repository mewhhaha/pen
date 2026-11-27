//! Automatic UV2 Lightmap Chart Unwrapper/Packer & Multi-Bounce GPU Path-Traced Lightmap Baker.
//!
//! Produces HDR irradiance lightmaps (`.exr`), dominant-light directional maps (`.png`),
//! and soft sun shadowmasks (`.png`) with gutter dilation and normal-aware bilateral denoising.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "UV2 rasterizer and atlas coordinates")]

use std::collections::VecDeque;

use anyhow::{Result, ensure};

use crate::hdri::HdriImage;
use crate::ray_tracing::{TLAS_TRAVERSAL_WGSL, TlasSceneGpu};

pub const LIGHTMAP_BAKE_WGSL: &str = crate::wgsl!("lightmap_bake.wgsl");

/// World-space mesh triangle fed into UV2 chart packing and lightmap baking.
#[derive(Clone, Copy, Debug)]
pub struct LightmapTriangle {
    pub positions: [[f32; 3]; 3],
    pub normals: [[f32; 3]; 3],
    pub uv2: Option<[[f32; 2]; 3]>,
}

/// Output of automatic UV2 chart unwrapping and skyline bin packing.
#[derive(Clone, Debug)]
pub struct PackedLightmapMesh {
    /// Per-triangle unpacked UV2 coordinates in `[0, 1]^2` with guaranteed non-overlapping gutters.
    pub triangle_uv2: Vec<[[f32; 2]; 3]>,
    pub chart_count: u32,
    pub utilization: f32,
    pub atlas_size: (u32, u32),
}

/// Lightmap baking quality and lighting settings.
#[derive(Clone, Copy, Debug)]
pub struct LightmapBakeParams {
    pub width: u32,
    pub height: u32,
    pub samples_per_texel: u32,
    pub max_bounces: u32,
    pub gutter_texels: u32,
    pub denoise: bool,
    pub sun_direction: [f32; 3],
    pub sun_angular_radius: f32,
    pub sun_color: [f32; 3],
    pub sky_zenith: [f32; 3],
    pub sky_horizon: [f32; 3],
    pub normal_bias: f32,
    pub max_ray_distance: f32,
    pub visibility_mask: u32,
}

impl Default for LightmapBakeParams {
    fn default() -> Self {
        Self {
            width: 128,
            height: 128,
            samples_per_texel: 64,
            max_bounces: 2,
            gutter_texels: 2,
            denoise: true,
            sun_direction: [0.4, 0.8, 0.45],
            sun_angular_radius: 0.02,
            sun_color: [3.0, 2.9, 2.7],
            sky_zenith: [0.15, 0.25, 0.5],
            sky_horizon: [0.35, 0.42, 0.55],
            normal_bias: 0.005,
            max_ray_distance: 100.0,
            visibility_mask: u32::MAX,
        }
    }
}

/// Baked CPU-accessible lightmap textures ready for `.exr`/`.png` export or `SurfaceMapAtlas` upload.
#[derive(Clone, Debug)]
pub struct BakedLightmap {
    pub width: u32,
    pub height: u32,
    /// Linear HDR RGB irradiance (`width * height * 4`, alpha = validity mask `0` or `1`).
    pub irradiance_rgba32f: Vec<[f32; 4]>,
    /// Dominant incoming direction packed in `[0, 255]` (`rgb = dir * 0.5 + 0.5`, `a = directionality`).
    pub directional_rgba8: Vec<u8>,
    /// Soft direct sun visibility in `[0, 255]` (`width * height`).
    pub shadowmask_r8: Vec<u8>,
}

impl BakedLightmap {
    /// Encodes the HDR irradiance map as a lossless 32-bit float OpenEXR byte buffer.
    pub fn encode_irradiance_exr(&self) -> Result<Vec<u8>> {
        let mut pixels = Vec::with_capacity((self.width * self.height) as usize);
        for px in &self.irradiance_rgba32f {
            pixels.push([px[0].max(0.0), px[1].max(0.0), px[2].max(0.0), 1.0]);
        }
        let img = HdriImage { width: self.width, height: self.height, pixels };
        Ok(img.to_exr_bytes())
    }

    /// Saves `lightmap_irradiance.exr`, `lightmap_directional.png`, and `lightmap_shadowmask.png` into `dir`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_to_dir(&self, dir: &std::path::Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("lightmap_irradiance.exr"), self.encode_irradiance_exr()?)?;
        let dir_img = image::RgbaImage::from_raw(self.width, self.height, self.directional_rgba8.clone())
            .ok_or_else(|| anyhow::anyhow!("invalid directional lightmap buffer"))?;
        dir_img.save(dir.join("lightmap_directional.png"))?;
        let sh_img = image::GrayImage::from_raw(self.width, self.height, self.shadowmask_r8.clone())
            .ok_or_else(|| anyhow::anyhow!("invalid shadowmask buffer"))?;
        sh_img.save(dir.join("lightmap_shadowmask.png"))?;
        Ok(())
    }
}

// ============================================================================
// Automatic UV2 Chart Unwrapper + Skyline Bin Packer
// ============================================================================

fn face_bucket(n: [f32; 3]) -> u8 {
    let ax = n[0].abs();
    let ay = n[1].abs();
    let az = n[2].abs();
    if ax >= ay && ax >= az {
        if n[0] >= 0.0 { 0 } else { 1 }
    } else if ay >= az {
        if n[1] >= 0.0 { 2 } else { 3 }
    } else if n[2] >= 0.0 {
        4
    } else {
        5
    }
}

fn project_bucket(p: [f32; 3], bucket: u8) -> [f32; 2] {
    match bucket {
        0 => [-p[2], p[1]],
        1 => [p[2], p[1]],
        2 => [p[0], -p[2]],
        3 => [p[0], p[2]],
        4 => [p[0], p[1]],
        _ => [-p[0], p[1]],
    }
}

fn quantize_pos(p: [f32; 3]) -> [i32; 3] {
    [(p[0] * 1000.0).round() as i32, (p[1] * 1000.0).round() as i32, (p[2] * 1000.0).round() as i32]
}

struct Chart {
    tris: Vec<usize>,
    local_uvs: Vec<[[f32; 2]; 3]>,
    width: f32,
    height: f32,
}

/// Skyline bin packer: tries packing all charts scaled by `density` (texels per world unit)
/// with `gutter` texels of padding around each chart into `(atlas_w, atlas_h)`.
fn try_skyline_pack(
    charts: &[Chart],
    order: &[usize],
    density: f32,
    gutter: u32,
    atlas_w: u32,
    atlas_h: u32,
) -> Option<Vec<(u32, u32, u32, u32)>> {
    let mut skyline: Vec<(u32, u32)> = vec![(0, 0)]; // (x, y) horizontal segments
    let mut placements = vec![(0u32, 0u32, 0u32, 0u32); charts.len()];
    let pad = gutter * 2;

    for &ci in order {
        let c = &charts[ci];
        let cw = ((c.width * density).ceil() as u32).max(1) + pad;
        let ch = ((c.height * density).ceil() as u32).max(1) + pad;
        if cw > atlas_w || ch > atlas_h {
            return None;
        }

        let mut best_y = u32::MAX;
        let mut best_x = 0u32;
        let mut best_idx = None;

        for i in 0..skyline.len() {
            let x = skyline[i].0;
            if x + cw > atlas_w {
                break;
            }
            let mut max_y = skyline[i].1;
            let mut j = i;
            while j < skyline.len() && skyline[j].0 < x + cw {
                max_y = max_y.max(skyline[j].1);
                j += 1;
            }
            if max_y + ch <= atlas_h && max_y < best_y {
                best_y = max_y;
                best_x = x;
                best_idx = Some(i);
            }
        }
        let i = best_idx?;
        placements[ci] = (best_x + gutter, best_y + gutter, cw - pad, ch - pad);

        let right_x = best_x + cw;
        let new_top = best_y + ch;
        let mut old_right_y = skyline[i].1;
        let mut remove_end = i;
        while remove_end < skyline.len() && skyline[remove_end].0 < right_x {
            old_right_y = skyline[remove_end].1;
            remove_end += 1;
        }
        let need_right = remove_end >= skyline.len() || skyline[remove_end].0 > right_x;
        skyline.drain(i..remove_end);
        skyline.insert(i, (best_x, new_top));
        if need_right && right_x < atlas_w {
            skyline.insert(i + 1, (right_x, old_right_y));
        }
        // Merge adjacent segments with identical height.
        let mut k = 0;
        while k + 1 < skyline.len() {
            if skyline[k].1 == skyline[k + 1].1 {
                skyline.remove(k + 1);
            } else {
                k += 1;
            }
        }
    }
    Some(placements)
}

/// Automatically unwraps world-space triangles into non-overlapping planar UV2 charts packed
/// into an `(atlas_w, atlas_h)` lightmap atlas with uniform world-space texel density.
pub fn pack_lightmap_uv2(
    triangles: &[LightmapTriangle],
    atlas_size: (u32, u32),
    gutter_texels: u32,
) -> Result<PackedLightmapMesh> {
    ensure!(atlas_size.0 >= 16 && atlas_size.1 >= 16, "lightmap atlas must be at least 16x16");
    if triangles.is_empty() {
        return Ok(PackedLightmapMesh { triangle_uv2: Vec::new(), chart_count: 0, utilization: 0.0, atlas_size });
    }

    // If every triangle already supplies explicit UV2 coordinates, preserve them verbatim.
    if triangles.iter().all(|t| t.uv2.is_some()) {
        let triangle_uv2 = triangles.iter().map(|t| t.uv2.unwrap()).collect();
        return Ok(PackedLightmapMesh { triangle_uv2, chart_count: 1, utilization: 1.0, atlas_size });
    }

    // Group triangles by 6 cardinal normal buckets + spatial vertex adjacency into planar charts.
    let buckets: Vec<u8> = triangles
        .iter()
        .map(|t| {
            let avg_n = [
                t.normals[0][0] + t.normals[1][0] + t.normals[2][0],
                t.normals[0][1] + t.normals[1][1] + t.normals[2][1],
                t.normals[0][2] + t.normals[1][2] + t.normals[2][2],
            ];
            face_bucket(avg_n)
        })
        .collect();

    let mut vert_to_tris: std::collections::HashMap<[i32; 3], Vec<usize>> = std::collections::HashMap::new();
    for (ti, t) in triangles.iter().enumerate() {
        for p in &t.positions {
            vert_to_tris.entry(quantize_pos(*p)).or_default().push(ti);
        }
    }

    let mut visited = vec![false; triangles.len()];
    let mut charts = Vec::new();
    for start in 0..triangles.len() {
        if visited[start] {
            continue;
        }
        let b = buckets[start];
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited[start] = true;
        let mut tris = Vec::new();
        while let Some(cur) = queue.pop_front() {
            tris.push(cur);
            for p in &triangles[cur].positions {
                if let Some(adj) = vert_to_tris.get(&quantize_pos(*p)) {
                    for &nxt in adj {
                        if !visited[nxt] && buckets[nxt] == b {
                            visited[nxt] = true;
                            queue.push_back(nxt);
                        }
                    }
                }
            }
        }

        let mut min_uv = [f32::INFINITY; 2];
        let mut max_uv = [f32::NEG_INFINITY; 2];
        let mut raw_uvs = Vec::with_capacity(tris.len());
        for &ti in &tris {
            let mut uvs = [[0.0f32; 2]; 3];
            for (uv, &pos) in uvs.iter_mut().zip(&triangles[ti].positions) {
                let p2 = project_bucket(pos, b);
                min_uv[0] = min_uv[0].min(p2[0]);
                min_uv[1] = min_uv[1].min(p2[1]);
                max_uv[0] = max_uv[0].max(p2[0]);
                max_uv[1] = max_uv[1].max(p2[1]);
                *uv = p2;
            }
            raw_uvs.push(uvs);
        }
        for uvs in &mut raw_uvs {
            for uv in uvs.iter_mut() {
                uv[0] -= min_uv[0];
                uv[1] -= min_uv[1];
            }
        }
        charts.push(Chart {
            tris,
            local_uvs: raw_uvs,
            width: (max_uv[0] - min_uv[0]).max(1e-4),
            height: (max_uv[1] - min_uv[1]).max(1e-4),
        });
    }

    // Sort tallest charts first for optimal skyline packing.
    let mut order: Vec<usize> = (0..charts.len()).collect();
    order.sort_by(|&a, &b| charts[b].height.total_cmp(&charts[a].height));

    let total_area: f32 = charts.iter().map(|c| c.width * c.height).sum::<f32>().max(1e-4);
    let mut hi = ((atlas_size.0 as f32 * atlas_size.1 as f32) / total_area).sqrt() * 2.0;
    let mut lo = 0.01f32;
    let mut best_pack = None;
    let mut best_density = lo;

    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        if let Some(p) = try_skyline_pack(&charts, &order, mid, gutter_texels, atlas_size.0, atlas_size.1) {
            best_pack = Some(p);
            best_density = mid;
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let placements = best_pack.ok_or_else(|| {
        anyhow::anyhow!("failed to pack {} UV2 charts into {}x{} atlas", charts.len(), atlas_size.0, atlas_size.1)
    })?;

    let mut triangle_uv2 = vec![[[0.0f32; 2]; 3]; triangles.len()];
    let mut used_texels = 0u64;
    let inv_w = 1.0 / atlas_size.0 as f32;
    let inv_h = 1.0 / atlas_size.1 as f32;
    for (ci, chart) in charts.iter().enumerate() {
        let (ox, oy, cw, ch) = placements[ci];
        used_texels += u64::from(cw) * u64::from(ch);
        for (local_i, &ti) in chart.tris.iter().enumerate() {
            let mut uvs = [[0.0f32; 2]; 3];
            for (uv, &local_uv) in uvs.iter_mut().zip(&chart.local_uvs[local_i]) {
                let px = ox as f32 + local_uv[0] * best_density;
                let py = oy as f32 + local_uv[1] * best_density;
                *uv = [(px * inv_w).clamp(0.0, 1.0), (py * inv_h).clamp(0.0, 1.0)];
            }
            triangle_uv2[ti] = uvs;
        }
    }
    let utilization = used_texels as f32 / (u64::from(atlas_size.0) * u64::from(atlas_size.1)) as f32;
    Ok(PackedLightmapMesh { triangle_uv2, chart_count: charts.len() as u32, utilization, atlas_size })
}

// ============================================================================
// Conservative UV2 Surface Rasterizer
// ============================================================================

fn rasterize_uv2_gbuffer(
    triangles: &[LightmapTriangle],
    uv2: &[[[f32; 2]; 3]],
    w: u32,
    h: u32,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let n = (w * h) as usize;
    let mut pos = vec![[0.0f32; 4]; n];
    let mut norm = vec![[0.0, 1.0, 0.0, 0.0]; n];
    let wf = w as f32;
    let hf = h as f32;

    for (ti, tri) in triangles.iter().enumerate() {
        let u = uv2[ti];
        let p0 = [u[0][0] * wf, u[0][1] * hf];
        let p1 = [u[1][0] * wf, u[1][1] * hf];
        let p2 = [u[2][0] * wf, u[2][1] * hf];

        let wi = w.cast_signed() - 1;
        let hi = h.cast_signed() - 1;
        let min_x = (p0[0].min(p1[0]).min(p2[0]).floor() as i32 - 1).clamp(0, wi) as u32;
        let max_x = (p0[0].max(p1[0]).max(p2[0]).ceil() as i32 + 1).clamp(0, wi) as u32;
        let min_y = (p0[1].min(p1[1]).min(p2[1]).floor() as i32 - 1).clamp(0, hi) as u32;
        let max_y = (p0[1].max(p1[1]).max(p2[1]).ceil() as i32 + 1).clamp(0, hi) as u32;

        let denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1]);
        if denom.abs() < 1e-8 {
            continue;
        }
        let inv_d = 1.0 / denom;

        for y in min_y..=max_y {
            let py = y as f32 + 0.5;
            for x in min_x..=max_x {
                let px = x as f32 + 0.5;
                let w0 = ((p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])) * inv_d;
                let w1 = ((p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])) * inv_d;
                let w2 = 1.0 - w0 - w1;
                // Slightly conservative tolerance captures exact boundary texels without skipping slivers.
                if w0 >= -0.05 && w1 >= -0.05 && w2 >= -0.05 {
                    let c0 = w0.max(0.0);
                    let c1 = w1.max(0.0);
                    let c2 = w2.max(0.0);
                    let sum = (c0 + c1 + c2).max(1e-6);
                    let (b0, b1, b2) = (c0 / sum, c1 / sum, c2 / sum);
                    let wp = [
                        tri.positions[0][0] * b0 + tri.positions[1][0] * b1 + tri.positions[2][0] * b2,
                        tri.positions[0][1] * b0 + tri.positions[1][1] * b1 + tri.positions[2][1] * b2,
                        tri.positions[0][2] * b0 + tri.positions[1][2] * b1 + tri.positions[2][2] * b2,
                        1.0,
                    ];
                    let wn_raw = [
                        tri.normals[0][0] * b0 + tri.normals[1][0] * b1 + tri.normals[2][0] * b2,
                        tri.normals[0][1] * b0 + tri.normals[1][1] * b1 + tri.normals[2][1] * b2,
                        tri.normals[0][2] * b0 + tri.normals[1][2] * b1 + tri.normals[2][2] * b2,
                    ];
                    let nl = (wn_raw[0] * wn_raw[0] + wn_raw[1] * wn_raw[1] + wn_raw[2] * wn_raw[2]).sqrt();
                    let wn = if nl > 1e-6 {
                        [wn_raw[0] / nl, wn_raw[1] / nl, wn_raw[2] / nl, 0.0]
                    } else {
                        [0.0, 1.0, 0.0, 0.0]
                    };
                    let idx = (y * w + x) as usize;
                    pos[idx] = wp;
                    norm[idx] = wn;
                }
            }
        }
    }
    (pos, norm)
}

// ============================================================================
// GPU Lightmap Baker
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BakeUniform {
    sun_dir_and_radius: [f32; 4],
    sun_color_and_bias: [f32; 4],
    sky_zenith_and_max: [f32; 4],
    sky_horizon: [f32; 4],
    counts: [u32; 4],
    bake_cfg: [u32; 4],
}

/// GPU multi-bounce path-traced lightmap baker running over [`TlasSceneGpu`].
pub struct GpuLightmapBaker {
    bake_pipe: wgpu::ComputePipeline,
    bake_bgl: wgpu::BindGroupLayout,
    dilate_pipe: wgpu::ComputePipeline,
    dilate_bgl: wgpu::BindGroupLayout,
    denoise_pipe: wgpu::ComputePipeline,
    denoise_bgl: wgpu::BindGroupLayout,
}

fn tex_rgba32f(device: &wgpu::Device, label: &str, w: u32, h: u32, usage: wgpu::TextureUsages) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage,
        view_formats: &[],
    })
}

fn readback_rgba32f(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    w: u32,
    h: u32,
) -> Result<Vec<[f32; 4]>> {
    let bpr = (w * 16).next_multiple_of(256);
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lm_rb"),
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
    rx.recv().unwrap()?;
    let data = slice.get_mapped_range()?;
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        let row_start = (y * bpr) as usize;
        let row_end = row_start + (w * 16) as usize;
        let row: &[[f32; 4]] = bytemuck::cast_slice(&data[row_start..row_end]);
        out.extend_from_slice(row);
    }
    drop(data);
    rb.unmap();
    Ok(out)
}

impl GpuLightmapBaker {
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
        let tex_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let tex_wo = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba32Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let bake_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lm_bake_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<BakeUniform>() as u64),
                    },
                    count: None,
                },
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_ro(4),
                tex_ro(5),
                tex_ro(6),
                tex_wo(7),
                tex_wo(8),
                tex_wo(9),
            ],
        });
        let dilate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lm_dilate_bgl"),
            entries: &[tex_ro(0), tex_wo(1)],
        });
        let denoise_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lm_denoise_bgl"),
            entries: &[tex_ro(0), tex_ro(1), tex_ro(2), tex_wo(3)],
        });

        let source = format!("{TLAS_TRAVERSAL_WGSL}\n{LIGHTMAP_BAKE_WGSL}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lightmap_bake_wgsl"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let make_pipe = |label: &str, bgl: &wgpu::BindGroupLayout, entry: &str| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            bake_pipe: make_pipe("lm_bake", &bake_bgl, "bake_lightmap"),
            bake_bgl,
            dilate_pipe: make_pipe("lm_dilate", &dilate_bgl, "dilate_lightmap"),
            dilate_bgl,
            denoise_pipe: make_pipe("lm_denoise", &denoise_bgl, "denoise_lightmap"),
            denoise_bgl,
        }
    }

    /// Bakes irradiance, directional dominant-light, and sun shadowmask lightmaps for `target_triangles`
    /// inside `scene` on the GPU, returning both the packed UV2 coordinates and the baked maps.
    pub fn bake(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &TlasSceneGpu,
        target_triangles: &[LightmapTriangle],
        params: &LightmapBakeParams,
    ) -> Result<(PackedLightmapMesh, BakedLightmap)> {
        let max_dim = device.limits().max_texture_dimension_2d;
        let (w, h) = (params.width, params.height);
        ensure!(w >= 16 && h >= 16 && w <= max_dim && h <= max_dim, "invalid lightmap size {w}x{h}");

        let packed = pack_lightmap_uv2(target_triangles, (w, h), params.gutter_texels)?;
        let (pos_data, norm_data) = rasterize_uv2_gbuffer(target_triangles, &packed.triangle_uv2, w, h);

        let gbuf_usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let rw_usage =
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC;

        let pos_tex = tex_rgba32f(device, "lm_gbuf_pos", w, h, gbuf_usage);
        let norm_tex = tex_rgba32f(device, "lm_gbuf_norm", w, h, gbuf_usage);
        queue.write_texture(
            pos_tex.as_image_copy(),
            bytemuck::cast_slice(&pos_data),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 16), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.write_texture(
            norm_tex.as_image_copy(),
            bytemuck::cast_slice(&norm_data),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 16), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let pos_view = pos_tex.create_view(&Default::default());
        let norm_view = norm_tex.create_view(&Default::default());

        let irr_a = tex_rgba32f(device, "lm_irr_a", w, h, rw_usage);
        let irr_b = tex_rgba32f(device, "lm_irr_b", w, h, rw_usage);
        let dir_a = tex_rgba32f(device, "lm_dir_a", w, h, rw_usage);
        let dir_b = tex_rgba32f(device, "lm_dir_b", w, h, rw_usage);
        let sh_a = tex_rgba32f(device, "lm_sh_a", w, h, rw_usage);
        let sh_b = tex_rgba32f(device, "lm_sh_b", w, h, rw_usage);

        let u = BakeUniform {
            sun_dir_and_radius: [
                params.sun_direction[0],
                params.sun_direction[1],
                params.sun_direction[2],
                params.sun_angular_radius.clamp(0.0, 0.2),
            ],
            sun_color_and_bias: [
                params.sun_color[0].max(0.0),
                params.sun_color[1].max(0.0),
                params.sun_color[2].max(0.0),
                params.normal_bias.max(1e-4),
            ],
            sky_zenith_and_max: [
                params.sky_zenith[0].max(0.0),
                params.sky_zenith[1].max(0.0),
                params.sky_zenith[2].max(0.0),
                params.max_ray_distance.max(1.0),
            ],
            sky_horizon: [
                params.sky_horizon[0].max(0.0),
                params.sky_horizon[1].max(0.0),
                params.sky_horizon[2].max(0.0),
                0.0,
            ],
            counts: [scene.tlas_node_count, scene.tlas_instance_count, scene.blas_tri_count, params.visibility_mask],
            bake_cfg: [w, h, params.samples_per_texel.clamp(1, 256), params.max_bounces.clamp(1, 4)],
        };
        let u_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lm_u"),
            size: std::mem::size_of::<BakeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        u_buf.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytemuck::bytes_of(&u));
        u_buf.unmap();

        let (gx, gy) = (w.div_ceil(8), h.div_ceil(8));
        let mut enc = device.create_command_encoder(&Default::default());

        // 1. Multi-bounce GPU path trace.
        let irr_av = irr_a.create_view(&Default::default());
        let irr_bv = irr_b.create_view(&Default::default());
        let dir_av = dir_a.create_view(&Default::default());
        let dir_bv = dir_b.create_view(&Default::default());
        let sh_av = sh_a.create_view(&Default::default());
        let sh_bv = sh_b.create_view(&Default::default());

        let bake_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lm_bake_bg"),
            layout: &self.bake_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scene.tlas_nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scene.tlas_instances_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: scene.blas_nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: scene.blas_tris_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&pos_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&norm_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&irr_av) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&dir_av) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&sh_av) },
            ],
        });
        {
            let mut pass =
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("lm_bake"), timestamp_writes: None });
            pass.set_pipeline(&self.bake_pipe);
            pass.set_bind_group(0, &bake_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // 2. Optional normal-aware bilateral denoise on irradiance.
        let mut irr_in_b = false;
        if params.denoise {
            let dn_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lm_dn_bg"),
                layout: &self.denoise_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&irr_av) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&norm_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&pos_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&irr_bv) },
                ],
            });
            let mut pass = enc
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("lm_denoise"), timestamp_writes: None });
            pass.set_pipeline(&self.denoise_pipe);
            pass.set_bind_group(0, &dn_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
            irr_in_b = true;
        }

        // 3. Gutter dilation ping-pong across all 3 maps.
        let mut dir_in_b = false;
        let mut sh_in_b = false;
        let dilate_steps = params.gutter_texels.min(8);
        for _ in 0..dilate_steps {
            let run_dilate = |enc: &mut wgpu::CommandEncoder, src: &wgpu::TextureView, dst: &wgpu::TextureView| {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("lm_dilate_bg"),
                    layout: &self.dilate_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(dst) },
                    ],
                });
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("lm_dilate"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.dilate_pipe);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            };
            if irr_in_b {
                run_dilate(&mut enc, &irr_bv, &irr_av);
            } else {
                run_dilate(&mut enc, &irr_av, &irr_bv);
            }
            irr_in_b = !irr_in_b;

            if dir_in_b {
                run_dilate(&mut enc, &dir_bv, &dir_av);
            } else {
                run_dilate(&mut enc, &dir_av, &dir_bv);
            }
            dir_in_b = !dir_in_b;

            if sh_in_b {
                run_dilate(&mut enc, &sh_bv, &sh_av);
            } else {
                run_dilate(&mut enc, &sh_av, &sh_bv);
            }
            sh_in_b = !sh_in_b;
        }

        queue.submit(Some(enc.finish()));

        let irr_final = if irr_in_b { &irr_b } else { &irr_a };
        let dir_final = if dir_in_b { &dir_b } else { &dir_a };
        let sh_final = if sh_in_b { &sh_b } else { &sh_a };

        let irr_data = readback_rgba32f(device, queue, irr_final, w, h)?;
        let dir_data = readback_rgba32f(device, queue, dir_final, w, h)?;
        let sh_data = readback_rgba32f(device, queue, sh_final, w, h)?;

        let mut directional_rgba8 = Vec::with_capacity((w * h * 4) as usize);
        for d in &dir_data {
            for c in d {
                directional_rgba8.push((c.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
        let mut shadowmask_r8 = Vec::with_capacity((w * h) as usize);
        for s in &sh_data {
            shadowmask_r8.push((s[0].clamp(0.0, 1.0) * 255.0).round() as u8);
        }

        Ok((
            packed,
            BakedLightmap { width: w, height: h, irradiance_rgba32f: irr_data, directional_rgba8, shadowmask_r8 },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_tracing::{Blas, BlasTriangle, TlasInstance, TlasScene};

    const ID: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    #[test]
    fn gpu_lightmap_uv2_packer_and_cornell_color_bleed_shadowmask_exr_witness() {
        let (device, queue) = crate::testing::device().expect("GPU adapter required");

        // Target floor quad spanning x in [-2, 2], z in [-2, 2] at y = 0 (NO pre-authored UV2!).
        let floor_lm_tris = [
            LightmapTriangle {
                positions: [[-2.0, 0.0, -2.0], [2.0, 0.0, -2.0], [2.0, 0.0, 2.0]],
                normals: [[0.0, 1.0, 0.0]; 3],
                uv2: None,
            },
            LightmapTriangle {
                positions: [[-2.0, 0.0, -2.0], [2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]],
                normals: [[0.0, 1.0, 0.0]; 3],
                uv2: None,
            },
        ];

        // Scene geometry:
        // - Emissive bright RED wall on the left (x = -2, y in [0, 3], z in [-2, 2])
        // - Shadow-casting roof overhang on the right (x in [0.5, 2.0], y = 1.5, z in [-2, 2])
        let red_wall_blas = Blas::build(&[
            BlasTriangle::flat([-2.0, 0.0, -2.0], [-2.0, 0.0, 2.0], [-2.0, 3.0, 2.0]),
            BlasTriangle::flat([-2.0, 0.0, -2.0], [-2.0, 3.0, 2.0], [-2.0, 3.0, -2.0]),
        ])
        .unwrap();
        let roof_blas = Blas::build(&[
            BlasTriangle::flat([0.5, 1.5, -2.0], [2.0, 1.5, -2.0], [2.0, 1.5, 2.0]),
            BlasTriangle::flat([0.5, 1.5, -2.0], [2.0, 1.5, 2.0], [0.5, 1.5, 2.0]),
        ])
        .unwrap();

        let mut red_inst = TlasInstance::new(0, ID);
        red_inst.albedo = [0.95, 0.05, 0.05];
        red_inst.emissive = [5.0, 0.0, 0.0];

        let mut roof_inst = TlasInstance::new(1, ID);
        roof_inst.albedo = [0.2, 0.2, 0.2];

        let scene = TlasScene::build(&[red_wall_blas, roof_blas], &[red_inst, roof_inst]).unwrap();
        let scene_gpu = TlasSceneGpu::upload(&device, &scene).unwrap();

        let baker = GpuLightmapBaker::new(&device);
        let params = LightmapBakeParams {
            width: 64,
            height: 64,
            samples_per_texel: 64,
            max_bounces: 2,
            gutter_texels: 2,
            denoise: true,
            sun_direction: [0.0, 1.0, 0.0],
            sun_angular_radius: 0.0,
            sun_color: [2.0, 2.0, 2.0],
            sky_zenith: [0.05, 0.08, 0.15],
            sky_horizon: [0.05, 0.08, 0.12],
            normal_bias: 0.01,
            max_ray_distance: 50.0,
            visibility_mask: u32::MAX,
        };

        let (packed, baked) = baker.bake(&device, &queue, &scene_gpu, &floor_lm_tris, &params).unwrap();
        assert_eq!(packed.chart_count, 1, "coplanar floor quads must merge into 1 chart");
        assert!(packed.utilization > 0.7, "UV2 packer utilization must be high: {}", packed.utilization);

        // Find texel near left red wall (x ~ -1.5, z ~ 0) and texel under right roof (x ~ +1.3, z ~ 0).
        let sample_at_world = |wx: f32, wz: f32| -> ([f32; 4], u8) {
            // Interpolate UV2 using triangle 0/1 bounds: x in [-2, 2], z in [-2, 2].
            let u_min = packed.triangle_uv2[0][0][0].min(packed.triangle_uv2[0][1][0]);
            let u_max = packed.triangle_uv2[0][0][0].max(packed.triangle_uv2[0][1][0]);
            let v_min = packed.triangle_uv2[0][0][1].min(packed.triangle_uv2[0][2][1]);
            let v_max = packed.triangle_uv2[0][0][1].max(packed.triangle_uv2[0][2][1]);
            let tx = (wx + 2.0) / 4.0;
            let tz = 1.0 - (wz + 2.0) / 4.0; // bucket 2 (+Y) projects [x, -z]
            let px = ((u_min + tx * (u_max - u_min)) * baked.width as f32).clamp(0.0, (baked.width - 1) as f32) as u32;
            let py =
                ((v_min + tz * (v_max - v_min)) * baked.height as f32).clamp(0.0, (baked.height - 1) as f32) as u32;
            let idx = (py * baked.width + px) as usize;
            (baked.irradiance_rgba32f[idx], baked.shadowmask_r8[idx])
        };

        let (near_red_irr, near_red_sh) = sample_at_world(-1.5, 0.0);
        let (under_roof_irr, under_roof_sh) = sample_at_world(1.3, 0.0);

        assert!(
            near_red_irr[0] > 0.5 && near_red_irr[0] > near_red_irr[2] * 3.0,
            "baked lightmap near red wall must capture strong red bounce irradiance: {near_red_irr:?}"
        );
        assert!(near_red_sh > 240, "open floor near red wall must be unshadowed: {near_red_sh}");
        assert!(under_roof_sh < 15, "floor under right roof must be shadowed in shadowmask: {under_roof_sh}");
        assert!(
            near_red_irr[0] > under_roof_irr[0] * 2.0,
            "red color bleed must fall off with distance across floor: near={near_red_irr:?}, far={under_roof_irr:?}"
        );

        // Verify EXR round-trip decode!
        let exr_bytes = baked.encode_irradiance_exr().unwrap();
        let decoded = HdriImage::from_exr_bytes(&exr_bytes).unwrap();
        assert_eq!((decoded.width, decoded.height), (64, 64));
    }
}
