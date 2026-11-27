//! GPU-packed 4x3 Surface Extension Atlas (`SurfaceMapAtlas`).
//!
//! Packs up to 12 per-material auxiliary maps (ORM, Emissive, Layer/Detail, Clearcoat,
//! Clearcoat Normal, Sheen, Anisotropy, Transmission/Thickness, Specular, Iridescence,
//! Lightmap Irradiance, Lightmap Direction+Shadowmask) into ONE `texture_2d<f32>` with
//! 1-texel wrap gutters so standard material shaders can sample every glTF PBR extension
//! map and baked lightmap within WebGPU's strict 15-sampled-textures-per-stage limit.

use bytemuck::{Pod, Zeroable};

/// Atlas grid dimensions: 4 columns x 3 rows = 12 tiles.
pub const ATLAS_COLS: u32 = 4;
pub const ATLAS_ROWS: u32 = 3;
/// Default interior resolution per tile (plus 1-texel wrap border on each side).
pub const DEFAULT_TILE_SIZE: u32 = 128;

/// Tile indices inside the 4x3 surface extension atlas.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceTile {
    Orm = 0,
    Emissive = 1,
    Layer = 2,
    Clearcoat = 3,
    ClearcoatNormal = 4,
    Sheen = 5,
    Anisotropy = 6,
    Transmission = 7,
    Specular = 8,
    Iridescence = 9,
    Lightmap = 10,
    LightmapDir = 11,
}

impl SurfaceTile {
    pub const fn bit(self) -> u32 {
        1u32 << (self as u32)
    }
}

/// WGSL helper functions to sample any tile from the 4x3 `SurfaceMapAtlas` cleanly
/// without cross-tile bilinear bleeding (interior UV mapped inside the 1-texel gutter).
pub const SURFACE_ATLAS_WGSL: &str = r#"
const ATLAS_TILE_ORM: u32 = 0u;
const ATLAS_TILE_EMISSIVE: u32 = 1u;
const ATLAS_TILE_LAYER: u32 = 2u;
const ATLAS_TILE_CLEARCOAT: u32 = 3u;
const ATLAS_TILE_CLEARCOAT_NORMAL: u32 = 4u;
const ATLAS_TILE_SHEEN: u32 = 5u;
const ATLAS_TILE_ANISOTROPY: u32 = 6u;
const ATLAS_TILE_TRANSMISSION: u32 = 7u;
const ATLAS_TILE_SPECULAR: u32 = 8u;
const ATLAS_TILE_IRIDESCENCE: u32 = 9u;
const ATLAS_TILE_LIGHTMAP: u32 = 10u;
const ATLAS_TILE_LIGHTMAP_DIR: u32 = 11u;

fn surface_atlas_uv(tile: u32, uv: vec2f, atlas_dims: vec2f, clamp_uv: bool) -> vec2f {
  let col = f32(tile % 4u);
  let row = f32(tile / 4u);
  let cell = atlas_dims / vec2f(4.0, 3.0);
  let wrapped = select(fract(uv), clamp(uv, vec2f(0.0), vec2f(1.0)), clamp_uv);
  let inner = cell - vec2f(2.0);
  let px = vec2f(col, row) * cell + vec2f(1.0) + wrapped * inner;
  return px / atlas_dims;
}

fn surface_atlas_sample(atlas: texture_2d<f32>, samp: sampler, tile: u32, uv: vec2f, clamp_uv: bool) -> vec4f {
  let dims = vec2f(textureDimensions(atlas, 0));
  let auv = surface_atlas_uv(tile, uv, dims, clamp_uv);
  return textureSampleLevel(atlas, samp, auv, 0.0);
}
"#;

const PACK_WGSL: &str = r#"
struct TileParams {
  dst_origin: vec2u,
  tile_size: u32,
  mode: u32, // 0 = copy RGBA from src0, 1 = RG from src0.r & src1.g, 2 = RGB from src0 & A from src1.r
  clamp_edges: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: TileParams;
@group(0) @binding(1) var src0: texture_2d<f32>;
@group(0) @binding(2) var src1: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var dst_atlas: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn pack_tile(@builtin(global_invocation_id) gid: vec3u) {
  let cell = params.tile_size + 2u;
  if (gid.x >= cell || gid.y >= cell) {
    return;
  }
  let inner = f32(params.tile_size);
  let raw_uv = (vec2f(f32(gid.x) - 0.5, f32(gid.y) - 0.5)) / inner;
  let uv = select(fract(raw_uv), clamp(raw_uv, vec2f(0.0), vec2f(1.0)), params.clamp_edges != 0u);
  let a = textureSampleLevel(src0, samp, uv, 0.0);
  var out_val = a;
  if (params.mode == 1u) {
    let b = textureSampleLevel(src1, samp, uv, 0.0);
    out_val = vec4f(a.r, b.g, a.b, 1.0);
  } else if (params.mode == 2u) {
    let b = textureSampleLevel(src1, samp, uv, 0.0);
    out_val = vec4f(a.rgb, b.a);
  }
  textureStore(dst_atlas, vec2i(params.dst_origin + gid.xy), out_val);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileParams {
    dst_origin: [u32; 2],
    tile_size: u32,
    mode: u32,
    clamp_edges: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Specification for packing one tile into a `SurfaceMapAtlas`.
pub struct TileCopy<'a> {
    pub tile: SurfaceTile,
    pub src0: &'a wgpu::TextureView,
    pub src1: Option<&'a wgpu::TextureView>,
    /// `0` = direct RGBA copy, `1` = `(src0.r, src1.g, src0.b, 1)`, `2` = `(src0.rgb, src1.a)`.
    pub mode: u32,
    /// Clamp edges instead of wrap (true for `TEXCOORD_1` lightmaps).
    pub clamp_edges: bool,
}

/// GPU tile packer creating 4x3 `SurfaceMapAtlas` textures.
pub struct SurfaceAtlasPacker {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl SurfaceAtlasPacker {
    pub fn new(device: &wgpu::Device, cache: Option<&wgpu::PipelineCache>) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("surface_atlas_pack"),
            source: wgpu::ShaderSource::Wgsl(PACK_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surface_atlas_pack_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<TileParams>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("surface_atlas_pack_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("surface_atlas_pack"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("pack_tile"),
            compilation_options: Default::default(),
            cache,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("surface_atlas_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self { pipeline, layout, sampler }
    }

    /// Creates and populates a 4x3 `SurfaceMapAtlas` texture, returning `(texture, tile_mask)`.
    pub fn pack(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tile_size: u32,
        copies: &[TileCopy<'_>],
        fallback: &wgpu::TextureView,
    ) -> (wgpu::Texture, u32) {
        let tile = tile_size.clamp(8, 512);
        let cell = tile + 2;
        let width = cell * ATLAS_COLS;
        let height = cell * ATLAS_ROWS;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("surface_map_atlas"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let dst_view = texture.create_view(&Default::default());
        if copies.is_empty() {
            return (texture, 0);
        }

        let stride = device.limits().min_uniform_buffer_offset_alignment.max(256) as usize;
        let mut uniform_bytes = vec![0u8; copies.len() * stride];
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_atlas_uniforms"),
            size: uniform_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut mask = 0u32;
        let mut groups = Vec::with_capacity(copies.len());
        for (i, copy) in copies.iter().enumerate() {
            let idx = copy.tile as u32;
            mask |= copy.tile.bit();
            let col = idx % ATLAS_COLS;
            let row = idx / ATLAS_COLS;
            let params = TileParams {
                dst_origin: [col * cell, row * cell],
                tile_size: tile,
                mode: copy.mode,
                clamp_edges: u32::from(copy.clamp_edges),
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let off = i * stride;
            uniform_bytes[off..off + std::mem::size_of::<TileParams>()].copy_from_slice(bytemuck::bytes_of(&params));
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("surface_atlas_tile_bg"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &uniform_buf,
                            offset: off as u64,
                            size: wgpu::BufferSize::new(std::mem::size_of::<TileParams>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(copy.src0) },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(copy.src1.unwrap_or(fallback)),
                    },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&dst_view) },
                ],
            });
            groups.push(bg);
        }
        queue.write_buffer(&uniform_buf, 0, &uniform_bytes);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pack_atlas") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surface_atlas_pack"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            let wg = cell.div_ceil(8);
            for bg in &groups {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(wg, wg, 1);
            }
        }
        queue.submit([encoder.finish()]);
        (texture, mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_surface_atlas_packs_multiple_pbr_maps_into_single_texture() {
        let Some((device, queue)) = crate::testing::device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let make_solid = |rgba: [u8; 4]| -> wgpu::Texture {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("solid"),
                size: wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let data: Vec<u8> = rgba.iter().copied().cycle().take(4 * 4 * 4).collect();
            queue.write_texture(
                t.as_image_copy(),
                &data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(16), rows_per_image: Some(4) },
                wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
            );
            t
        };

        let clearcoat = make_solid([200, 0, 0, 255]);
        let clearcoat_rough = make_solid([0, 100, 0, 255]);
        let lightmap = make_solid([50, 150, 250, 255]);
        let fallback = make_solid([255, 255, 255, 255]);

        let cc_v = clearcoat.create_view(&Default::default());
        let ccr_v = clearcoat_rough.create_view(&Default::default());
        let lm_v = lightmap.create_view(&Default::default());
        let fb_v = fallback.create_view(&Default::default());

        let packer = SurfaceAtlasPacker::new(&device, None);
        let (atlas, mask) = packer.pack(
            &device,
            &queue,
            16,
            &[
                TileCopy { tile: SurfaceTile::Clearcoat, src0: &cc_v, src1: Some(&ccr_v), mode: 1, clamp_edges: false },
                TileCopy { tile: SurfaceTile::Lightmap, src0: &lm_v, src1: None, mode: 0, clamp_edges: true },
            ],
            &fb_v,
        );
        assert_eq!(mask, SurfaceTile::Clearcoat.bit() | SurfaceTile::Lightmap.bit());

        // Read back the packed atlas and verify Clearcoat tile (col 3, row 0) and Lightmap tile (col 2, row 2).
        let w = atlas.width();
        let h = atlas.height();
        let padded_row = (w * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atlas_rb"),
            size: u64::from(padded_row * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            atlas.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.recv().unwrap().unwrap();
        let bytes = slice.get_mapped_range().unwrap();
        let cell = 18usize;
        let px = |col: usize, row: usize| -> [u8; 4] {
            let x = col * cell + 8;
            let y = row * cell + 8;
            let off = y * padded_row as usize + x * 4;
            [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]
        };
        let cc_px = px(3, 0);
        let lm_px = px(2, 2);
        assert!(
            (i32::from(cc_px[0]) - 200).abs() <= 2 && (i32::from(cc_px[1]) - 100).abs() <= 2,
            "clearcoat={cc_px:?}"
        );
        assert!(
            (i32::from(lm_px[0]) - 50).abs() <= 2
                && (i32::from(lm_px[1]) - 150).abs() <= 2
                && (i32::from(lm_px[2]) - 250).abs() <= 2,
            "lightmap={lm_px:?}"
        );
    }
}
