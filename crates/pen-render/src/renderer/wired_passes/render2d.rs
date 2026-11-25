use super::super::{GpuAsset, State, TextureAsset};
use super::{RENDER2D_EXT_TAG, RENDER2D_WORLD_EXT_TAG, WiredPasses};
use crate::asset_store::AssetKind;
use crate::render2d::{
    Camera2d, RadianceCascades2d, RadianceCascades2dConfig, Scene2dEmitter, SdfShape, SdfShapePass, Sprite2d,
    Sprites2dPass, TileCell, TilemapPass, TilemapUniform, rasterize_scene_2d, sort_sprites,
};
use anyhow::Result;
use pen_protocol::{Catalog, RenderFrame};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// World-space 2D sprite batch referencing a bound texture atlas.
#[derive(Clone)]
pub struct SpriteLayer2d {
    pub sprites: Vec<Sprite2d>,
    pub atlas: wgpu::TextureView,
    pub sampler: Option<wgpu::Sampler>,
    pub y_sort: bool,
}

/// Single-draw GPU tilemap layer referencing an uploaded `TileCell` storage buffer and atlas.
#[derive(Clone)]
pub struct TilemapLayer2d {
    pub uniform: TilemapUniform,
    pub cells: Arc<wgpu::Buffer>,
    pub atlas: wgpu::TextureView,
    pub sampler: Option<wgpu::Sampler>,
}

/// 2D Radiance Cascades global illumination layer composited over world-space 2D tilemaps & sprites.
#[derive(Clone)]
pub struct RadianceCascades2dLayer {
    /// Emissive lights & opaque walls rasterized into an `Rgba16Float` scene texture (pixel space).
    pub emitters: Vec<Scene2dEmitter>,
    /// Optional caller-owned `Rgba16Float` 2D scene texture (overrides `emitters` when `Some`).
    pub scene_view: Option<wgpu::TextureView>,
    pub config: RadianceCascades2dConfig,
    /// `[ambient, multiply, add]` weights applied to 2D world albedo + GI irradiance.
    pub composite_weights: [f32; 3],
    /// When `true` (default for pure 2D scenes), replaces the frame output with the GI-lit 2D world;
    /// when `false`, alpha-blends lit 2D layers over the underlying 3D scene.
    pub opaque_backdrop: bool,
}

impl Default for RadianceCascades2dLayer {
    fn default() -> Self {
        Self {
            emitters: Vec::new(),
            scene_view: None,
            config: RadianceCascades2dConfig::default(),
            composite_weights: [0.08, 1.0, 0.25],
            opaque_backdrop: true,
        }
    }
}

/// Complete first-class 2D world + vector SDF UI scene attached via [`crate::Renderer::set_scene_2d`].
#[derive(Clone)]
pub struct Scene2d {
    /// World camera for `tilemaps` and `sprite_layers`. Zero viewport auto-fills the output size.
    pub camera: Camera2d,
    pub tilemaps: Vec<TilemapLayer2d>,
    pub sprite_layers: Vec<SpriteLayer2d>,
    /// Optional 2D Radiance Cascades global illumination over the 2D tilemaps & sprites.
    pub radiance_cascades: Option<RadianceCascades2dLayer>,
    /// Vector SDF UI shapes drawn BEFORE `frame.sprites` (so Pen MSDF text draws cleanly on top).
    pub sdf_shapes: Vec<SdfShape>,
    /// Vector SDF UI shapes drawn AFTER `frame.sprites` (for overlays, tooltips, and debug panels).
    pub sdf_overlay_shapes: Vec<SdfShape>,
}

impl Default for Scene2d {
    fn default() -> Self {
        Self {
            camera: Camera2d { center: [0.0; 2], zoom: 1.0, rotation: 0.0, viewport: [0.0; 2] },
            tilemaps: Vec::new(),
            sprite_layers: Vec::new(),
            radiance_cascades: None,
            sdf_shapes: Vec::new(),
            sdf_overlay_shapes: Vec::new(),
        }
    }
}

const ALPHA_BLIT_WGSL: &str = r#"
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;

@vertex fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
  let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
  let y = 1.0 - f32(idx & 2u) * 2.0;
  return vec4f(x, y, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(src_tex, 0));
  let uv = pos.xy / max(dims, vec2f(1.0));
  return textureSampleLevel(src_tex, src_samp, uv, 0.0);
}
"#;

struct AlphaBlitShared {
    module: wgpu::ShaderModule,
    bgl: wgpu::BindGroupLayout,
    pl: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
}

struct CachedRc2d {
    size: [u32; 2],
    c0_spacing: u32,
    c0_dirs: u32,
    cascades: Option<u32>,
    pass: RadianceCascades2d,
}

#[derive(Default)]
pub(in crate::renderer) struct Render2dWiredState {
    pub scene: Option<Scene2d>,
    sprites_passes: HashMap<wgpu::TextureFormat, Sprites2dPass>,
    tilemap_passes: HashMap<wgpu::TextureFormat, TilemapPass>,
    sdf_passes: HashMap<wgpu::TextureFormat, SdfShapePass>,
    default_sampler: Option<wgpu::Sampler>,
    rc_state: Option<CachedRc2d>,
    rc_targets: Option<([u32; 2], wgpu::Texture, wgpu::TextureView, wgpu::Texture, wgpu::TextureView)>,
    rc_emitter_tex: Option<([u32; 2], u64, wgpu::Texture, wgpu::TextureView)>,
    ext_tilemap_buffers: HashMap<u64, Arc<wgpu::Buffer>>,
    alpha_blit_shared: Option<AlphaBlitShared>,
    alpha_blit_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "IEEE-754 binary16 bit packing")]
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff).cast_signed();
    let frac = bits & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | u16::from(frac != 0);
    }
    let unbiased = exp - 127 + 15;
    if unbiased >= 31 {
        return sign | 0x7c00;
    }
    if unbiased <= 0 {
        if unbiased < -10 {
            return sign;
        }
        let mant = (frac | 0x0080_0000) >> (14 - unbiased);
        return sign | (mant as u16);
    }
    sign | ((unbiased as u16) << 10) | ((frac >> 13) as u16)
}

fn hash_emitters(emitters: &[Scene2dEmitter]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    emitters.len().hash(&mut h);
    for e in emitters {
        match *e {
            Scene2dEmitter::Circle { center, radius, material } => {
                0u8.hash(&mut h);
                for f in [center[0], center[1], radius, material[0], material[1], material[2], material[3]] {
                    f.to_bits().hash(&mut h);
                }
            }
            Scene2dEmitter::Rect { rect, material } => {
                1u8.hash(&mut h);
                for f in [rect[0], rect[1], rect[2], rect[3], material[0], material[1], material[2], material[3]] {
                    f.to_bits().hash(&mut h);
                }
            }
        }
    }
    h.finish()
}

fn decode_ext_sdf_shapes(frame: &RenderFrame, base: &mut Vec<SdfShape>, overlay: &mut Vec<SdfShape>) {
    for ext in &frame.extensions {
        if ext.tag != RENDER2D_EXT_TAG || ext.values.len() < 36 {
            continue;
        }
        let count = ext.values.len() / 36;
        for i in 0..count {
            let v = &ext.values[i * 36..(i + 1) * 36];
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let raw_kind = v[24].round().clamp(0.0, 31.0) as u32;
            let is_overlay = raw_kind >= 16;
            let kind = (raw_kind & 15).min(2);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let gradient_kind = v[26].round().clamp(0.0, 2.0) as u32;
            let shape = SdfShape {
                rect: [v[0], v[1], v[2], v[3]],
                corner_radii: [v[4], v[5], v[6], v[7]],
                fill_color: [v[8], v[9], v[10], v[11]],
                fill_color_end: [v[12], v[13], v[14], v[15]],
                border_color: [v[16], v[17], v[18], v[19]],
                shadow_color: [v[20], v[21], v[22], v[23]],
                kind,
                border_width_px: v[25].max(0.0),
                gradient_kind,
                gradient_angle: v[27],
                shadow_offset: [v[28], v[29]],
                shadow_blur_px: v[30].max(0.0),
                clip_radius_px: v[31].max(0.0),
                clip_rect: [v[32], v[33], v[34], v[35]],
                ..SdfShape::default()
            };
            if is_overlay {
                overlay.push(shape);
            } else {
                base.push(shape);
            }
        }
    }
}

fn resolve_2d_material_view(
    material_idx_f: f32,
    catalog: &Catalog,
    assets: &crate::asset_store::AssetStore<GpuAsset>,
    white: &TextureAsset,
) -> wgpu::TextureView {
    if !material_idx_f.is_finite() || material_idx_f < 0.0 {
        return white.view.clone();
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "validated nonnegative catalog index")]
    let idx = material_idx_f.round() as usize;
    let Some(mat_key) = catalog.materials.get(idx) else {
        return white.view.clone();
    };
    let Some(mat_slot) = assets.id(AssetKind::Material, mat_key) else {
        return white.view.clone();
    };
    let Some(GpuAsset::Material(mat)) = assets.current(mat_slot) else {
        return white.view.clone();
    };
    let Some(tex_key) = mat.material.texture.as_deref() else {
        return white.view.clone();
    };
    let Some(tex_slot) = assets.id(AssetKind::Texture, tex_key) else {
        return white.view.clone();
    };
    match assets.current(tex_slot) {
        Some(GpuAsset::Texture(tex)) => tex.view.clone(),
        _ => white.view.clone(),
    }
}

fn decode_ext_world_2d(
    device: &wgpu::Device,
    wired: &mut Render2dWiredState,
    frame: &RenderFrame,
    catalog: &Catalog,
    assets: &crate::asset_store::AssetStore<GpuAsset>,
    white: &TextureAsset,
) -> Option<Scene2d> {
    let ext = frame.extensions.iter().rev().find(|e| e.tag == RENDER2D_WORLD_EXT_TAG && e.values.len() >= 8)?;
    let v = &ext.values;
    let mut cursor = 8usize;
    let camera = Camera2d {
        center: [v[0], v[1]],
        zoom: if v[2] > 0.0 { v[2] } else { 1.0 },
        rotation: v[3],
        viewport: [0.0, 0.0],
    };
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "bounded extension header counts")]
    let tilemap_count = v[4].round().clamp(0.0, 64.0) as usize;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "bounded extension header counts")]
    let sprite_count = v[5].round().clamp(0.0, 65_536.0) as usize;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "bounded extension header counts")]
    let emitter_count = v[6].round().clamp(0.0, 4096.0) as usize;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "bounded extension header flags")]
    let rc_flags = v[7].round().clamp(0.0, 255.0) as u32;
    let rc_mode = rc_flags & 3;
    let y_sort = (rc_flags & 4) != 0;

    let mut rc_layer: Option<RadianceCascades2dLayer> = None;
    if rc_mode != 0 {
        if cursor + 6 > v.len() {
            return None;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped RC2D config")]
        let c0_spacing = v[cursor].round().clamp(1.0, 64.0) as u32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped RC2D config")]
        let c0_dirs = v[cursor + 1].round().clamp(1.0, 64.0) as u32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped RC2D config")]
        let cascades_raw = v[cursor + 2].round().clamp(0.0, 6.0) as u32;
        rc_layer = Some(RadianceCascades2dLayer {
            emitters: Vec::new(),
            scene_view: None,
            config: RadianceCascades2dConfig {
                c0_spacing,
                c0_dirs,
                cascades: (cascades_raw > 0).then_some(cascades_raw),
            },
            composite_weights: [v[cursor + 3].max(0.0), v[cursor + 4].max(0.0), v[cursor + 5].max(0.0)],
            opaque_backdrop: rc_mode == 1,
        });
        cursor += 6;
    }

    let mut tilemaps = Vec::with_capacity(tilemap_count);
    for _ in 0..tilemap_count {
        if cursor + 15 > v.len() {
            break;
        }
        let mat_idx = v[cursor];
        let ox = v[cursor + 1];
        let oy = v[cursor + 2];
        let tw = v[cursor + 3].max(1e-3);
        let th = v[cursor + 4].max(1e-3);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped tilemap grid dims")]
        let gw = v[cursor + 5].round().clamp(1.0, 1024.0) as u32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped tilemap grid dims")]
        let gh = v[cursor + 6].round().clamp(1.0, 1024.0) as u32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped tilemap atlas dims")]
        let acols = v[cursor + 7].round().clamp(1.0, 256.0) as u32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped tilemap atlas dims")]
        let arows = v[cursor + 8].round().clamp(1.0, 256.0) as u32;
        let anim_time = v[cursor + 9];
        let anim_fps = v[cursor + 10];
        let tint = [v[cursor + 11], v[cursor + 12], v[cursor + 13], v[cursor + 14]];
        cursor += 15;
        let cell_count = (gw as usize).saturating_mul(gh as usize);
        if cursor + cell_count > v.len() {
            break;
        }
        let mut cells = Vec::with_capacity(cell_count);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        gw.hash(&mut hasher);
        gh.hash(&mut hasher);
        for &raw in &v[cursor..cursor + cell_count] {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "24-bit TileCell packed in f32")]
            let bits = raw.round().clamp(0.0, 16_777_215.0) as u32;
            bits.hash(&mut hasher);
            cells.push(TileCell(bits));
        }
        cursor += cell_count;
        let key = hasher.finish();
        if wired.ext_tilemap_buffers.len() > 32 && !wired.ext_tilemap_buffers.contains_key(&key) {
            wired.ext_tilemap_buffers.clear();
        }
        let buf = wired
            .ext_tilemap_buffers
            .entry(key)
            .or_insert_with(|| Arc::new(TilemapPass::upload_cells(device, &cells)))
            .clone();
        let atlas = resolve_2d_material_view(mat_idx, catalog, assets, white);
        tilemaps.push(TilemapLayer2d {
            uniform: TilemapUniform {
                camera_vp: [[0.0; 4]; 4],
                origin_and_tile_size: [ox, oy, tw, th],
                grid_size_and_atlas: [gw, gh, acols, arows],
                anim_time_and_tint: [anim_time, anim_fps, 0.0, 0.0, tint[0], tint[1], tint[2], tint[3]],
            },
            cells: buf,
            atlas,
            sampler: None,
        });
    }

    // Read sprites, sort globally by painter's key when y_sort is on, then batch contiguous runs by material index.
    let mut raw_sprites: Vec<(i32, Sprite2d)> = Vec::with_capacity(sprite_count);
    for _ in 0..sprite_count {
        if cursor + 25 > v.len() {
            break;
        }
        #[allow(clippy::cast_possible_truncation, reason = "signed catalog material index")]
        let mat_id = if v[cursor] < 0.0 { -1 } else { v[cursor].round().min(i32::MAX as f32) as i32 };
        let sprite = Sprite2d {
            pos: [v[cursor + 1], v[cursor + 2]],
            size: [v[cursor + 3], v[cursor + 4]],
            pivot_rot_layer: [v[cursor + 5], v[cursor + 6], v[cursor + 7], v[cursor + 8]],
            uv_rect: [v[cursor + 9], v[cursor + 10], v[cursor + 11], v[cursor + 12]],
            color: [v[cursor + 13], v[cursor + 14], v[cursor + 15], v[cursor + 16]],
            nine_slice: [v[cursor + 17], v[cursor + 18], v[cursor + 19], v[cursor + 20]],
            flags: [v[cursor + 21], v[cursor + 22], v[cursor + 23], v[cursor + 24]],
        };
        cursor += 25;
        raw_sprites.push((mat_id, sprite));
    }
    raw_sprites.sort_by(|a, b| {
        let k = |x: f32| x + 0.0;
        k(a.1.pivot_rot_layer[3])
            .total_cmp(&k(b.1.pivot_rot_layer[3]))
            .then_with(|| if y_sort { k(b.1.pos[1]).total_cmp(&k(a.1.pos[1])) } else { std::cmp::Ordering::Equal })
    });
    let mut sprite_layers: Vec<SpriteLayer2d> = Vec::new();
    let mut idx = 0;
    while idx < raw_sprites.len() {
        let mat_id = raw_sprites[idx].0;
        let start = idx;
        while idx < raw_sprites.len() && raw_sprites[idx].0 == mat_id {
            idx += 1;
        }
        let atlas = resolve_2d_material_view(mat_id as f32, catalog, assets, white);
        sprite_layers.push(SpriteLayer2d {
            sprites: raw_sprites[start..idx].iter().map(|(_, s)| *s).collect(),
            atlas,
            sampler: None,
            y_sort: false, // already globally sorted across atlases
        });
    }

    let mut emitters = Vec::with_capacity(emitter_count);
    for _ in 0..emitter_count {
        if cursor + 9 > v.len() {
            break;
        }
        let is_rect = v[cursor] > 0.5;
        let material = [v[cursor + 5], v[cursor + 6], v[cursor + 7], v[cursor + 8]];
        if is_rect {
            emitters.push(Scene2dEmitter::Rect {
                rect: [v[cursor + 1], v[cursor + 2], v[cursor + 3], v[cursor + 4]],
                material,
            });
        } else {
            emitters.push(Scene2dEmitter::Circle {
                center: [v[cursor + 1], v[cursor + 2]],
                radius: v[cursor + 3].max(0.0),
                material,
            });
        }
        cursor += 9;
    }
    if let Some(rc) = rc_layer.as_mut() {
        rc.emitters = emitters;
    }

    if tilemaps.is_empty() && sprite_layers.is_empty() && rc_layer.is_none() {
        return None;
    }
    Some(Scene2d {
        camera,
        tilemaps,
        sprite_layers,
        radiance_cascades: rc_layer,
        sdf_shapes: Vec::new(),
        sdf_overlay_shapes: Vec::new(),
    })
}

impl Render2dWiredState {
    fn default_sampler(&mut self, device: &wgpu::Device) -> wgpu::Sampler {
        self.default_sampler
            .get_or_insert_with(|| {
                device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("wired_render2d_sampler"),
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                })
            })
            .clone()
    }

    fn alpha_blit(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
    ) {
        let shared = self.alpha_blit_shared.get_or_insert_with(|| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("wired_render2d_alpha_blit"),
                source: wgpu::ShaderSource::Wgsl(ALPHA_BLIT_WGSL.into()),
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("wired_render2d_alpha_blit_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("wired_render2d_alpha_blit_pl"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("wired_render2d_alpha_blit_samp"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            AlphaBlitShared { module, bgl, pl, sampler }
        });
        let pipeline = self.alpha_blit_pipelines.entry(target_format).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("wired_render2d_alpha_blit"),
                layout: Some(&shared.pl),
                vertex: wgpu::VertexState {
                    module: &shared.module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shared.module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wired_render2d_alpha_blit_bg"),
            layout: &shared.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shared.sampler) },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("wired_render2d_alpha_blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

impl State {
    /// Encodes world-space 2D tilemaps, sprites2d, optional 2D Radiance Cascades GI, and base
    /// vector SDF UI shapes onto `output` BEFORE Pen's MSDF `frame.sprites`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_render2d_base_wired(
        wired: &mut WiredPasses,
        assets: &crate::asset_store::AssetStore<GpuAsset>,
        white: &TextureAsset,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        catalog: &Catalog,
        target_view: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
        size: (u32, u32),
    ) -> Result<Vec<SdfShape>> {
        let mut ext_base_sdf = Vec::new();
        let mut ext_overlay_sdf = Vec::new();
        decode_ext_sdf_shapes(frame, &mut ext_base_sdf, &mut ext_overlay_sdf);

        let scene_opt = wired
            .render2d
            .scene
            .clone()
            .or_else(|| decode_ext_world_2d(device, &mut wired.render2d, frame, catalog, assets, white));
        if scene_opt.is_none() && ext_base_sdf.is_empty() && ext_overlay_sdf.is_empty() {
            return Ok(Vec::new());
        }
        let (w, h) = (size.0.max(1), size.1.max(1));
        let viewport = [w as f32, h as f32];
        let mut all_base_sdf = Vec::new();
        let mut all_overlay_sdf = Vec::new();

        if let Some(scene) = scene_opt {
            let mut camera = scene.camera;
            if camera.viewport[0] <= 0.0 || camera.viewport[1] <= 0.0 {
                camera.viewport = viewport;
            }
            if camera.zoom <= 0.0 {
                camera.zoom = 1.0;
            }
            let has_world_2d = !scene.tilemaps.is_empty() || !scene.sprite_layers.is_empty();
            let default_sampler = wired.render2d.default_sampler(device);

            if let Some(rc_layer) = &scene.radiance_cascades {
                let rc_size = [w, h];
                let need_rc = !matches!(
                    &wired.render2d.rc_state,
                    Some(cached)
                        if cached.size == rc_size
                            && cached.c0_spacing == rc_layer.config.c0_spacing
                            && cached.c0_dirs == rc_layer.config.c0_dirs
                            && cached.cascades == rc_layer.config.cascades
                );
                if need_rc {
                    let pass = RadianceCascades2d::new(device, rc_size, rc_layer.config)?;
                    wired.render2d.rc_state = Some(CachedRc2d {
                        size: rc_size,
                        c0_spacing: rc_layer.config.c0_spacing,
                        c0_dirs: rc_layer.config.c0_dirs,
                        cascades: rc_layer.config.cascades,
                        pass,
                    });
                }
                if !matches!(&wired.render2d.rc_targets, Some((s, ..)) if *s == rc_size) {
                    let make_tex = |label, usage| {
                        let tex = device.create_texture(&wgpu::TextureDescriptor {
                            label: Some(label),
                            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba16Float,
                            usage,
                            view_formats: &[],
                        });
                        let view = tex.create_view(&Default::default());
                        (tex, view)
                    };
                    let (albedo_tex, albedo_view) = make_tex(
                        "wired_rc2d_albedo",
                        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    );
                    let (lit_tex, lit_view) = make_tex(
                        "wired_rc2d_lit",
                        wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                    );
                    wired.render2d.rc_targets = Some((rc_size, albedo_tex, albedo_view, lit_tex, lit_view));
                }
                let rc_scene_view = if let Some(view) = &rc_layer.scene_view {
                    view.clone()
                } else {
                    let ehash = hash_emitters(&rc_layer.emitters);
                    if !matches!(&wired.render2d.rc_emitter_tex, Some((s, prev, ..)) if *s == rc_size && *prev == ehash)
                    {
                        let tex = device.create_texture(&wgpu::TextureDescriptor {
                            label: Some("wired_rc2d_emitters"),
                            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba16Float,
                            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                            view_formats: &[],
                        });
                        let pixels = rasterize_scene_2d(rc_size, &rc_layer.emitters);
                        let mut f16_bytes = Vec::with_capacity(pixels.len() * 8);
                        for px in pixels {
                            for ch in px {
                                f16_bytes.extend_from_slice(&f32_to_f16(ch).to_le_bytes());
                            }
                        }
                        queue.write_texture(
                            tex.as_image_copy(),
                            &f16_bytes,
                            wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(w * 8),
                                rows_per_image: Some(h),
                            },
                            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                        );
                        let view = tex.create_view(&Default::default());
                        wired.render2d.rc_emitter_tex = Some((rc_size, ehash, tex, view));
                    }
                    wired.render2d.rc_emitter_tex.as_ref().unwrap().3.clone()
                };

                // Render 2D tilemaps & sprites into Rgba16Float albedo target.
                let mut draws = Vec::new();
                if has_world_2d {
                    let tm_pass = wired
                        .render2d
                        .tilemap_passes
                        .entry(wgpu::TextureFormat::Rgba16Float)
                        .or_insert_with(|| TilemapPass::new(device, wgpu::TextureFormat::Rgba16Float));
                    for tm in &scene.tilemaps {
                        let mut u = tm.uniform;
                        if u.camera_vp.iter().flatten().all(|v| *v == 0.0) {
                            u.camera_vp = camera.world_to_ndc();
                        }
                        let samp = tm.sampler.as_ref().unwrap_or(&default_sampler);
                        draws.push(tm_pass.prepare(device, &u, &tm.cells, &tm.atlas, samp));
                    }
                    let sp_pass = wired
                        .render2d
                        .sprites_passes
                        .entry(wgpu::TextureFormat::Rgba16Float)
                        .or_insert_with(|| Sprites2dPass::new(device, wgpu::TextureFormat::Rgba16Float));
                    for layer in &scene.sprite_layers {
                        let mut sorted = layer.sprites.clone();
                        sort_sprites(&mut sorted, layer.y_sort);
                        let samp = layer.sampler.as_ref().unwrap_or(&default_sampler);
                        if let Some(d) = sp_pass.prepare(device, camera, &sorted, &layer.atlas, samp) {
                            draws.push(d);
                        }
                    }
                }
                let (_, _, albedo_view, _, lit_view) = wired.render2d.rc_targets.as_ref().unwrap();
                let albedo_view = albedo_view.clone();
                let lit_view = lit_view.clone();
                {
                    let clear_albedo = if rc_layer.opaque_backdrop && !has_world_2d {
                        wgpu::Color::WHITE
                    } else {
                        wgpu::Color::TRANSPARENT
                    };
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("render2d_rc_albedo"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &albedo_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(clear_albedo),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        ..Default::default()
                    });
                    for draw in &draws {
                        draw.draw(&mut pass);
                    }
                }
                let rc = &wired.render2d.rc_state.as_ref().unwrap().pass;
                rc.build(device, encoder, &rc_scene_view);
                rc.composite(device, encoder, &albedo_view, &lit_view, rc_layer.composite_weights);
                if rc_layer.opaque_backdrop {
                    wired.blit(device, encoder, &lit_view, target_view, target_format);
                } else {
                    wired.render2d.alpha_blit(device, encoder, &lit_view, target_view, target_format);
                }
            } else if has_world_2d {
                let mut draws = Vec::new();
                let tm_pass = wired
                    .render2d
                    .tilemap_passes
                    .entry(target_format)
                    .or_insert_with(|| TilemapPass::new(device, target_format));
                for tm in &scene.tilemaps {
                    let mut u = tm.uniform;
                    if u.camera_vp.iter().flatten().all(|v| *v == 0.0) {
                        u.camera_vp = camera.world_to_ndc();
                    }
                    let samp = tm.sampler.as_ref().unwrap_or(&default_sampler);
                    draws.push(tm_pass.prepare(device, &u, &tm.cells, &tm.atlas, samp));
                }
                let sp_pass = wired
                    .render2d
                    .sprites_passes
                    .entry(target_format)
                    .or_insert_with(|| Sprites2dPass::new(device, target_format));
                for layer in &scene.sprite_layers {
                    let mut sorted = layer.sprites.clone();
                    sort_sprites(&mut sorted, layer.y_sort);
                    let samp = layer.sampler.as_ref().unwrap_or(&default_sampler);
                    if let Some(d) = sp_pass.prepare(device, camera, &sorted, &layer.atlas, samp) {
                        draws.push(d);
                    }
                }
                if !draws.is_empty() {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("render2d_world"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: target_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                        })],
                        ..Default::default()
                    });
                    for draw in &draws {
                        draw.draw(&mut pass);
                    }
                }
            }

            all_base_sdf.extend_from_slice(&scene.sdf_shapes);
            all_overlay_sdf.extend_from_slice(&scene.sdf_overlay_shapes);
        }

        all_base_sdf.extend_from_slice(&ext_base_sdf);
        all_overlay_sdf.extend_from_slice(&ext_overlay_sdf);

        if !all_base_sdf.is_empty() {
            let sdf_pass = wired
                .render2d
                .sdf_passes
                .entry(target_format)
                .or_insert_with(|| SdfShapePass::new(device, target_format));
            if let Some(draw) = sdf_pass.prepare(device, viewport, &all_base_sdf) {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("render2d_sdf_ui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    ..Default::default()
                });
                draw.draw(&mut pass);
            }
        }
        Ok(all_overlay_sdf)
    }

    /// Encodes overlay vector SDF UI shapes AFTER Pen's MSDF `frame.sprites`.
    pub(in crate::renderer) fn encode_render2d_overlay_wired(
        wired: &mut WiredPasses,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        shapes: &[SdfShape],
        target_view: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
        size: (u32, u32),
    ) {
        if shapes.is_empty() {
            return;
        }
        let viewport = [size.0.max(1) as f32, size.1.max(1) as f32];
        let sdf_pass =
            wired.render2d.sdf_passes.entry(target_format).or_insert_with(|| SdfShapePass::new(device, target_format));
        if let Some(draw) = sdf_pass.prepare(device, viewport, shapes) {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render2d_sdf_overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            draw.draw(&mut pass);
        }
    }
}
