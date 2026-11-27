//! Lean 2D-only renderer backend compiled when `feature = "render-3d"` is disabled.
//! Draws 2D worlds (`RENDER2D_WORLD_EXT_TAG`), vector SDF UI (`RENDER2D_EXT_TAG`),
//! protocol `frame.sprites` (including MSDF text and custom sprite shaders), and
//! accessibility color filters without linking the 3D PBR renderer.
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use pen_protocol::{Catalog, GameEvent, RenderFrame, Sprite};
use serde::Deserialize;

use crate::filter::Filter;
use crate::packed_texture::{PackedReader, is_packed_texture};
#[cfg(feature = "render-2d")]
use crate::render2d::{
    Camera2d, RadianceCascades2d, RadianceCascades2dConfig, Scene2dEmitter, SdfShape, SdfShapePass, Sprite2d,
    Sprites2dPass, TileCell, TilemapPass, TilemapUniform, sort_sprites,
};
use crate::sprites::{SpriteBatch, Sprites, sprite_group_layout};
use crate::{RenderStats, RenderTarget, RendererOptions};

#[cfg(feature = "render-2d")]
const RENDER2D_EXT_TAG: u32 = 65_008;
#[cfg(feature = "render-2d")]
const RENDER2D_WORLD_EXT_TAG: u32 = 65_009;

#[derive(Deserialize)]
struct MaterialJson {
    #[serde(default)]
    shader: Option<String>,
    #[serde(default)]
    texture: Option<String>,
    #[serde(default)]
    scene_filter: bool,
}

#[derive(Clone)]
struct Material2d {
    shader: Option<String>,
    texture: Option<String>,
    scene_filter: bool,
}

#[derive(Clone)]
struct Texture2d {
    view: wgpu::TextureView,
    view_id: u64,
}

enum LoadedAsset {
    Material { key: String, revision: u64, result: Result<Material2d, String> },
    Shader { key: String, revision: u64, result: Result<String, String> },
    Texture { key: String, revision: u64, result: Result<LoadedPixels, String> },
}

enum LoadedPixels {
    Rgba { width: u32, height: u32, pixels: Vec<u8> },
    Packed { width: u32, height: u32, format: wgpu::TextureFormat, pixels: Vec<u8> },
}

pub(crate) struct State2d {
    device: wgpu::Device,
    queue: wgpu::Queue,
    format: wgpu::TextureFormat,
    assets_root: String,
    sprite_layout: wgpu::PipelineLayout,
    sprites: Sprites,
    white: Texture2d,
    next_view_id: u64,
    revision: u64,
    pub(crate) filter: Filter,
    materials: HashMap<String, Material2d>,
    shaders: HashMap<String, (wgpu::RenderPipeline, usize)>,
    next_shader_id: usize,
    textures: HashMap<String, Texture2d>,
    in_flight: HashSet<(u8, String)>,
    failed: HashSet<(u8, String)>,
    inbox: Arc<Mutex<Vec<LoadedAsset>>>,
    events: Vec<GameEvent>,
    stats: RenderStats,
    #[cfg(feature = "render-2d")]
    sprites2d_pass: Option<Sprites2dPass>,
    #[cfg(feature = "render-2d")]
    tilemap_pass: Option<TilemapPass>,
    #[cfg(feature = "render-2d")]
    sdf_pass: Option<SdfShapePass>,
    #[cfg(feature = "render-2d")]
    rc2d_state: Option<([u32; 2], RadianceCascades2d)>,
    #[cfg(feature = "render-2d")]
    tilemap_cache: HashMap<u64, Arc<wgpu::Buffer>>,
    #[cfg(feature = "render-2d")]
    default_sampler: wgpu::Sampler,
}

const KIND_MATERIAL: u8 = 0;
const KIND_SHADER: u8 = 1;
const KIND_TEXTURE: u8 = 2;

pub fn validate_frame(frame: &RenderFrame) -> Result<()> {
    let finite = |v: f32| v.is_finite();
    ensure!(
        finite(frame.clear.r) && finite(frame.clear.g) && finite(frame.clear.b) && finite(frame.clear.a),
        "non-finite clear color"
    );
    for sprite in &frame.sprites {
        ensure!(
            finite(sprite.x) && finite(sprite.y) && finite(sprite.width) && finite(sprite.height),
            "non-finite sprite rectangle"
        );
    }
    Ok(())
}

impl State2d {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, options: RendererOptions) -> Result<Self> {
        let sprite_group = sprite_group_layout(&device);
        let sprite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sprite2d"),
            bind_group_layouts: &[Some(&sprite_group)],
            immediate_size: 0,
        });
        let sprites = Sprites::new(&device, sprite_group);
        let white_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("white2d"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            white_tex.as_image_copy(),
            &[255, 255, 255, 255],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let white = Texture2d { view: white_tex.create_view(&Default::default()), view_id: 1 };
        #[cfg(feature = "render-2d")]
        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Ok(Self {
            device,
            queue,
            format: options.format,
            assets_root: options.assets.to_string_lossy().into_owned(),
            sprite_layout,
            sprites,
            white,
            next_view_id: 2,
            revision: 1,
            filter: Filter::default(),
            materials: HashMap::new(),
            shaders: HashMap::new(),
            next_shader_id: 1,
            textures: HashMap::new(),
            in_flight: HashSet::new(),
            failed: HashSet::new(),
            inbox: Arc::new(Mutex::new(Vec::new())),
            events: Vec::new(),
            stats: RenderStats::default(),
            #[cfg(feature = "render-2d")]
            sprites2d_pass: None,
            #[cfg(feature = "render-2d")]
            tilemap_pass: None,
            #[cfg(feature = "render-2d")]
            sdf_pass: None,
            #[cfg(feature = "render-2d")]
            rc2d_state: None,
            #[cfg(feature = "render-2d")]
            tilemap_cache: HashMap::new(),
            #[cfg(feature = "render-2d")]
            default_sampler,
        })
    }

    pub fn sharing(&self, assets: PathBuf) -> Result<Self> {
        Self::new(
            self.device.clone(),
            self.queue.clone(),
            RendererOptions {
                format: self.format,
                display: crate::Display::SDR,
                assets,
                veil: None,
                pipeline_cache: None,
            },
        )
    }

    pub fn preload(&mut self, catalog: &Catalog) {
        for key in &catalog.materials {
            self.request_material(key);
        }
    }

    pub fn reload_all(&mut self) {
        self.revision += 1;
        self.failed.clear();
        let mat_keys: Vec<String> = self.materials.keys().cloned().collect();
        let shader_keys: Vec<String> = self.shaders.keys().cloned().collect();
        let tex_keys: Vec<String> = self.textures.keys().cloned().collect();
        for key in mat_keys {
            self.request_material(&key);
        }
        for key in shader_keys {
            self.request_shader(&key);
        }
        for key in tex_keys {
            self.request_texture(&key);
        }
    }

    pub fn take_events(&mut self) -> Vec<GameEvent> {
        self.drain_inbox();
        std::mem::take(&mut self.events)
    }

    pub fn pending(&self) -> usize {
        self.in_flight.len()
    }

    pub fn quiet(&self) -> bool {
        self.in_flight.is_empty()
    }

    pub fn stats(&self) -> RenderStats {
        self.stats
    }

    pub fn time_passes(&mut self, _enabled: bool) {}

    fn request_material(&mut self, key: &str) {
        if key.is_empty()
            || self.materials.contains_key(key)
            || self.failed.contains(&(KIND_MATERIAL, key.to_owned()))
            || !self.in_flight.insert((KIND_MATERIAL, key.to_owned()))
        {
            return;
        }
        let root = self.assets_root.clone();
        let key_owned = key.to_owned();
        let revision = self.revision;
        let inbox = self.inbox.clone();
        spawn_load(async move {
            let result = match crate::platform::read_asset(&root, &key_owned).await {
                Ok(bytes) => serde_json::from_slice::<MaterialJson>(&bytes)
                    .map(|m| Material2d { shader: m.shader, texture: m.texture, scene_filter: m.scene_filter })
                    .map_err(|e| e.to_string()),
                Err(e) => Err(format!("{e:#}")),
            };
            inbox.lock().unwrap().push(LoadedAsset::Material { key: key_owned, revision, result });
        });
    }

    fn request_shader(&mut self, key: &str) {
        if key.is_empty()
            || self.shaders.contains_key(key)
            || self.failed.contains(&(KIND_SHADER, key.to_owned()))
            || !self.in_flight.insert((KIND_SHADER, key.to_owned()))
        {
            return;
        }
        let root = self.assets_root.clone();
        let key_owned = key.to_owned();
        let revision = self.revision;
        let inbox = self.inbox.clone();
        spawn_load(async move {
            let result = match crate::platform::read_asset(&root, &key_owned).await {
                Ok(bytes) => String::from_utf8(bytes).map_err(|e| e.to_string()),
                Err(e) => Err(format!("{e:#}")),
            };
            inbox.lock().unwrap().push(LoadedAsset::Shader { key: key_owned, revision, result });
        });
    }

    fn request_texture(&mut self, key: &str) {
        if key.is_empty()
            || self.textures.contains_key(key)
            || self.failed.contains(&(KIND_TEXTURE, key.to_owned()))
            || !self.in_flight.insert((KIND_TEXTURE, key.to_owned()))
        {
            return;
        }
        let root = self.assets_root.clone();
        let key_owned = key.to_owned();
        let revision = self.revision;
        let inbox = self.inbox.clone();
        spawn_load(async move {
            let result = async {
                let bytes = crate::platform::read_asset(&root, &key_owned).await?;
                if is_packed_texture(&bytes) {
                    let mut packed = PackedReader::open_async(bytes).await?;
                    let header = packed.header;
                    let mut top = None;
                    while let Some(level) = packed.next_level()? {
                        if level.level == 0 {
                            top = Some(level);
                        }
                    }
                    let level = top.ok_or_else(|| anyhow::anyhow!("packed texture had no base level"))?;
                    Ok(LoadedPixels::Packed {
                        width: header.width,
                        height: header.height,
                        format: header.format,
                        pixels: level.pixels,
                    })
                } else {
                    let img = crate::platform::decode_image(bytes).await?;
                    Ok(LoadedPixels::Rgba { width: img.width, height: img.height, pixels: img.pixels })
                }
            }
            .await
            .map_err(|e: anyhow::Error| format!("{e:#}"));
            inbox.lock().unwrap().push(LoadedAsset::Texture { key: key_owned, revision, result });
        });
    }

    fn compile_sprite_pipeline(&self, code: &str) -> wgpu::RenderPipeline {
        use wgpu::{BlendComponent, BlendFactor, BlendOperation};
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sprite2d"),
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sprite2d"),
            layout: Some(&self.sprite_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.format,
                    blend: Some(wgpu::BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::SrcAlpha,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }

    fn upload_texture(&mut self, loaded: LoadedPixels) -> Texture2d {
        let (width, height, format, pixels, bpp) = match loaded {
            LoadedPixels::Rgba { width, height, pixels } => {
                (width, height, wgpu::TextureFormat::Rgba8UnormSrgb, pixels, 4u32)
            }
            LoadedPixels::Packed { width, height, format, pixels } => {
                let bpp = crate::packed_texture::texel_bytes(format);
                (width, height, format, pixels, bpp)
            }
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tex2d"),
            size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            texture.as_image_copy(),
            &pixels,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width * bpp), rows_per_image: Some(height) },
            wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
        );
        let view_id = self.next_view_id;
        self.next_view_id += 1;
        Texture2d { view: texture.create_view(&Default::default()), view_id }
    }

    fn drain_inbox(&mut self) {
        let arrived: Vec<LoadedAsset> = {
            let mut lock = self.inbox.lock().unwrap();
            std::mem::take(&mut *lock)
        };
        for item in arrived {
            match item {
                LoadedAsset::Material { key, revision, result } => {
                    self.in_flight.remove(&(KIND_MATERIAL, key.clone()));
                    match result {
                        Ok(mat) => {
                            if let Some(shader) = &mat.shader {
                                self.request_shader(shader);
                            }
                            if let Some(texture) = &mat.texture {
                                self.request_texture(texture);
                            }
                            self.materials.insert(key.clone(), mat);
                            self.events.push(GameEvent::AssetReady { key, kind: "material".into(), revision });
                        }
                        Err(message) => {
                            self.failed.insert((KIND_MATERIAL, key.clone()));
                            self.events.push(GameEvent::AssetFailed {
                                key,
                                kind: "material".into(),
                                revision,
                                message,
                            });
                        }
                    }
                }
                LoadedAsset::Shader { key, revision, result } => {
                    self.in_flight.remove(&(KIND_SHADER, key.clone()));
                    match result {
                        Ok(code) => {
                            let pipeline = self.compile_sprite_pipeline(&code);
                            let id = self.next_shader_id;
                            self.next_shader_id += 1;
                            self.shaders.insert(key.clone(), (pipeline, id));
                            self.events.push(GameEvent::AssetReady { key, kind: "shader".into(), revision });
                        }
                        Err(message) => {
                            self.failed.insert((KIND_SHADER, key.clone()));
                            self.events.push(GameEvent::AssetFailed { key, kind: "shader".into(), revision, message });
                        }
                    }
                }
                LoadedAsset::Texture { key, revision, result } => {
                    self.in_flight.remove(&(KIND_TEXTURE, key.clone()));
                    match result {
                        Ok(pixels) => {
                            let tex = self.upload_texture(pixels);
                            self.textures.insert(key.clone(), tex);
                            self.events.push(GameEvent::AssetReady { key, kind: "texture".into(), revision });
                        }
                        Err(message) => {
                            self.failed.insert((KIND_TEXTURE, key.clone()));
                            self.events.push(GameEvent::AssetFailed { key, kind: "texture".into(), revision, message });
                        }
                    }
                }
            }
        }
    }

    fn build_sprite_batches(&mut self, sprites: &[Sprite], catalog: &Catalog) -> (Vec<SpriteBatch>, Vec<u32>, bool) {
        let mut batches: Vec<SpriteBatch> = Vec::new();
        let mut order: Vec<u32> = Vec::new();
        let mut all_ready = true;
        for (index, sprite) in sprites.iter().enumerate() {
            let mat_key = catalog.material(sprite.material);
            if mat_key.is_empty() {
                continue;
            }
            let Some(mat) = self.materials.get(mat_key).cloned() else {
                self.request_material(mat_key);
                all_ready = false;
                continue;
            };
            let Some(shader_key) = mat.shader.as_deref() else { continue };
            let Some((pipeline, shader_id)) = self.shaders.get(shader_key).cloned() else {
                self.request_shader(shader_key);
                all_ready = false;
                continue;
            };
            let tex = match mat.texture.as_deref() {
                Some(tex_key) => match self.textures.get(tex_key) {
                    Some(tex) => tex.clone(),
                    None => {
                        self.request_texture(tex_key);
                        all_ready = false;
                        continue;
                    }
                },
                None => self.white.clone(),
            };
            let key = (tex.view_id, shader_id);
            let pos = u32::try_from(order.len()).unwrap_or(u32::MAX);
            order.push(u32::try_from(index).unwrap_or(u32::MAX));
            if let Some(last) = batches.last_mut()
                && last.key == key
                && last.scene_filter == mat.scene_filter
            {
                last.count += 1;
            } else {
                batches.push(SpriteBatch {
                    pipeline,
                    texture: tex.view,
                    key,
                    first: pos,
                    count: 1,
                    scene_filter: mat.scene_filter,
                });
            }
        }
        (batches, order, all_ready)
    }

    pub fn prepare_sprites(&mut self, sprites: &[Sprite], catalog: &Catalog) -> Result<bool> {
        self.drain_inbox();
        let (_, _, ready) = self.build_sprite_batches(sprites, catalog);
        Ok(ready)
    }

    pub fn overlay(
        &mut self,
        sprites: &[Sprite],
        catalog: &Catalog,
        output: &wgpu::TextureView,
        size: (u32, u32),
    ) -> Result<()> {
        self.drain_inbox();
        if sprites.is_empty() {
            return Ok(());
        }
        let (batches, order, _) = self.build_sprite_batches(sprites, catalog);
        if order.is_empty() {
            return Ok(());
        }
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("overlay2d") });
        self.sprites.draw(&self.device, &self.queue, &mut encoder, output, size, &batches, sprites, &order, None);
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    #[cfg(feature = "render-2d")]
    fn draw_sdf_shapes(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        size: (u32, u32),
        shapes: &[SdfShape],
    ) {
        if shapes.is_empty() {
            return;
        }
        let viewport = [size.0.max(1) as f32, size.1.max(1) as f32];
        let sdf = self.sdf_pass.get_or_insert_with(|| SdfShapePass::new(&self.device, self.format));
        let Some(draw) = sdf.prepare(&self.device, viewport, shapes) else { return };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render2d_sdf"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        draw.draw(&mut pass);
    }

    pub fn submit(&mut self, frame: &RenderFrame, catalog: &Catalog, target: &mut dyn RenderTarget) -> Result<()> {
        validate_frame(frame)?;
        self.drain_inbox();
        let size = target.size();
        if size.0 == 0 || size.1 == 0 {
            return Ok(());
        }
        let Some(output) = target.view() else {
            return Ok(());
        };
        let filtered = self.filter.begin(&self.device, &self.queue, self.format, None, size);
        let draw_target = filtered.as_ref().unwrap_or(&output);
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame2d") });

        // Clear the frame output.
        {
            let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear2d"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: draw_target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(frame.clear.r),
                            g: f64::from(frame.clear.g),
                            b: f64::from(frame.clear.b),
                            a: f64::from(frame.clear.a),
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
        }

        // Optional 2D world + vector SDF UI extension passes.
        #[cfg(feature = "render-2d")]
        let (sdf_base, sdf_overlay) = self.encode_render2d_world(&mut encoder, draw_target, size, frame, catalog);

        #[cfg(feature = "render-2d")]
        self.draw_sdf_shapes(&mut encoder, draw_target, size, &sdf_base);

        // Protocol sprites (MSDF text, UI, sprite shaders).
        let (batches, order, _) = self.build_sprite_batches(&frame.sprites, catalog);
        let draw_calls = u32::try_from(batches.len()).unwrap_or(0);
        if !order.is_empty() {
            self.sprites.draw(
                &self.device,
                &self.queue,
                &mut encoder,
                draw_target,
                size,
                &batches,
                &frame.sprites,
                &order,
                None,
            );
        }

        #[cfg(feature = "render-2d")]
        self.draw_sdf_shapes(&mut encoder, draw_target, size, &sdf_overlay);

        if filtered.is_some() {
            self.filter.finish(&mut encoder, &output);
        }
        self.queue.submit([encoder.finish()]);
        target.present();
        self.stats = RenderStats { draw_calls, scale: 1.0, ..RenderStats::default() };
        Ok(())
    }

    #[cfg(feature = "render-2d")]
    fn resolve_2d_view(&mut self, mat_idx_f: f32, catalog: &Catalog) -> wgpu::TextureView {
        if !mat_idx_f.is_finite() || mat_idx_f < 0.0 {
            return self.white.view.clone();
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "validated nonnegative index")]
        let idx = mat_idx_f.round() as usize;
        let Some(mat_key) = catalog.materials.get(idx) else { return self.white.view.clone() };
        let Some(mat) = self.materials.get(mat_key).cloned() else {
            self.request_material(mat_key);
            return self.white.view.clone();
        };
        let Some(tex_key) = mat.texture.as_deref() else { return self.white.view.clone() };
        if let Some(tex) = self.textures.get(tex_key) {
            tex.view.clone()
        } else {
            self.request_texture(tex_key);
            self.white.view.clone()
        }
    }

    #[cfg(feature = "render-2d")]
    fn encode_render2d_world(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        size: (u32, u32),
        frame: &RenderFrame,
        catalog: &Catalog,
    ) -> (Vec<SdfShape>, Vec<SdfShape>) {
        use std::hash::{Hash, Hasher};

        // Decode SDF shapes from RENDER2D_EXT_TAG (65_008).
        let mut base = Vec::new();
        let mut overlay = Vec::new();
        for ext in frame.extensions.iter().filter(|e| e.tag == RENDER2D_EXT_TAG) {
            for v in ext.values.as_chunks::<36>().0 {
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

        // Decode 2D world (tilemaps + sprites + optional RC2D) from RENDER2D_WORLD_EXT_TAG (65_009).
        let Some(ext) = frame.extensions.iter().rev().find(|e| e.tag == RENDER2D_WORLD_EXT_TAG) else {
            return (base, overlay);
        };
        let v = &ext.values;
        if v.len() < 8 {
            return (base, overlay);
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let count = |index: usize, max: f32| {
            let value = v[index];
            ((0.0..=max).contains(&value) && value.fract() == 0.0).then_some(value as usize)
        };
        let (Some(tm_count), Some(sp_count), Some(em_count)) = (count(4, 64.0), count(5, 65_536.0), count(6, 4096.0))
        else {
            return (base, overlay);
        };
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let rc_flags = v[7].round().clamp(0.0, 255.0) as u32;
        let rc_mode = rc_flags & 3;
        let y_sort = (rc_flags & 4) != 0;
        let header_size = if rc_mode == 0 { 8 } else { 14 };
        let Some(records) = v.get(header_size..) else { return (base, overlay) };
        let Some(tile_end) = records.len().checked_sub(sp_count * 25 + em_count * 9) else {
            return (base, overlay);
        };
        let (mut rest, tail) = records.split_at(tile_end);
        let (sp_slice, em_slice) = tail.split_at(sp_count * 25);

        let camera = Camera2d {
            center: [v[0], v[1]],
            zoom: if v[2] > 0.0 { v[2] } else { 1.0 },
            rotation: v[3],
            viewport: [size.0.max(1) as f32, size.1.max(1) as f32],
        };
        let vp = camera.world_to_ndc();

        // Tilemaps.
        let mut tilemaps = Vec::with_capacity(tm_count);
        for _ in 0..tm_count {
            let Some(tile) = rest.get(..15) else { break };
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let gw = tile[5].round().clamp(1.0, 1024.0) as u32;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let gh = tile[6].round().clamp(1.0, 1024.0) as u32;
            let cell_count = (gw * gh) as usize;
            let Some((record, next)) = rest.split_at_checked(15 + cell_count) else { break };
            rest = next;
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            gw.hash(&mut hasher);
            gh.hash(&mut hasher);
            let mut cells = Vec::with_capacity(cell_count);
            for &raw in &record[15..] {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let bits = raw.round().clamp(0.0, 16_777_215.0) as u32;
                bits.hash(&mut hasher);
                cells.push(TileCell(bits));
            }
            let key = hasher.finish();
            if self.tilemap_cache.len() > 32 && !self.tilemap_cache.contains_key(&key) {
                self.tilemap_cache.clear();
            }
            let buf = self
                .tilemap_cache
                .entry(key)
                .or_insert_with(|| Arc::new(TilemapPass::upload_cells(&self.device, &cells)))
                .clone();
            let atlas = self.resolve_2d_view(tile[0], catalog);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            tilemaps.push((
                TilemapUniform {
                    camera_vp: vp,
                    origin_and_tile_size: [tile[1], tile[2], tile[3].max(1e-3), tile[4].max(1e-3)],
                    grid_size_and_atlas: [
                        gw,
                        gh,
                        tile[7].round().clamp(1.0, 256.0) as u32,
                        tile[8].round().clamp(1.0, 256.0) as u32,
                    ],
                    anim_time_and_tint: [tile[9], tile[10], 0.0, 0.0, tile[11], tile[12], tile[13], tile[14]],
                },
                buf,
                atlas,
            ));
        }

        // World 2D sprites.
        let mut raw_sprites: Vec<(i32, Sprite2d)> = Vec::with_capacity(sp_count);
        for chunk in sp_slice.as_chunks::<25>().0 {
            #[allow(clippy::cast_possible_truncation)]
            let mat_id =
                if !chunk[0].is_finite() || chunk[0] < 0.0 { -1 } else { chunk[0].round().min(i32::MAX as f32) as i32 };
            raw_sprites.push((
                mat_id,
                Sprite2d {
                    pos: [chunk[1], chunk[2]],
                    size: [chunk[3], chunk[4]],
                    pivot_rot_layer: [chunk[5], chunk[6], chunk[7], chunk[8]],
                    uv_rect: [chunk[9], chunk[10], chunk[11], chunk[12]],
                    color: [chunk[13], chunk[14], chunk[15], chunk[16]],
                    nine_slice: [chunk[17], chunk[18], chunk[19], chunk[20]],
                    flags: [chunk[21], chunk[22], chunk[23], chunk[24]],
                },
            ));
        }
        crate::sort::sort_by_u64_key(&mut raw_sprites, |(_, s)| {
            let k = |x: f32| x + 0.0;
            let layer = u64::from(crate::sort::f32_sortable(k(s.pivot_rot_layer[3])));
            let y = if y_sort { u64::from(!crate::sort::f32_sortable(k(s.pos[1]))) } else { 0 };
            (layer << 32) | y
        });
        let mut sprite_batches: Vec<(wgpu::TextureView, Vec<Sprite2d>)> = Vec::new();
        let mut cur_mat: Option<i32> = None;
        for (mat_id, sp) in raw_sprites {
            if cur_mat == Some(mat_id)
                && let Some((_, list)) = sprite_batches.last_mut()
            {
                list.push(sp);
            } else {
                cur_mat = Some(mat_id);
                let view =
                    if mat_id < 0 { self.white.view.clone() } else { self.resolve_2d_view(mat_id as f32, catalog) };
                sprite_batches.push((view, vec![sp]));
            }
        }

        // Prepare & draw tilemaps + world sprites.
        let mut draws = Vec::new();
        if !tilemaps.is_empty() {
            let pass = self.tilemap_pass.get_or_insert_with(|| TilemapPass::new(&self.device, self.format));
            for (uniform, cells, atlas) in &tilemaps {
                draws.push(pass.prepare(&self.device, uniform, cells, atlas, &self.default_sampler));
            }
        }
        if !sprite_batches.is_empty() {
            let pass = self.sprites2d_pass.get_or_insert_with(|| Sprites2dPass::new(&self.device, self.format));
            for (atlas, list) in &mut sprite_batches {
                if !y_sort {
                    sort_sprites(list, false);
                }
                if let Some(d) = pass.prepare(&self.device, camera, list, atlas, &self.default_sampler) {
                    draws.push(d);
                }
            }
        }
        if !draws.is_empty() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render2d_world"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output,
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

        // Optional 2D Radiance Cascades lighting pass.
        if rc_mode != 0 && !em_slice.is_empty() {
            let mut emitters = Vec::with_capacity(em_count);
            for chunk in em_slice.as_chunks::<9>().0 {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let kind = chunk[0].round().clamp(0.0, 1.0) as u32;
                let mat = [chunk[5], chunk[6], chunk[7], chunk[8]];
                let emitter = if kind == 1 {
                    Scene2dEmitter::Rect { rect: [chunk[1], chunk[2], chunk[3], chunk[4]], material: mat }
                } else {
                    Scene2dEmitter::Circle { center: [chunk[1], chunk[2]], radius: chunk[3], material: mat }
                };
                emitters.push(emitter);
            }
            let rc_size = [size.0.max(1), size.1.max(1)];
            let pixels = crate::render2d::rasterize_scene_2d(rc_size, &emitters);
            let scene_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("rc2d_scene"),
                size: wgpu::Extent3d { width: rc_size[0], height: rc_size[1], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let half: Vec<u16> = pixels.iter().flat_map(|p| p.map(f32_to_f16)).collect();
            self.queue.write_texture(
                scene_tex.as_image_copy(),
                bytemuck::cast_slice(&half),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(rc_size[0] * 8),
                    rows_per_image: Some(rc_size[1]),
                },
                wgpu::Extent3d { width: rc_size[0], height: rc_size[1], depth_or_array_layers: 1 },
            );
            let scene_view = scene_tex.create_view(&Default::default());
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let rc_cfg = RadianceCascades2dConfig {
                c0_spacing: v[8].round().clamp(1.0, 64.0) as u32,
                c0_dirs: v[9].round().clamp(1.0, 64.0) as u32,
                cascades: {
                    let c = v[10].round().clamp(0.0, 6.0) as u32;
                    (c > 0).then_some(c)
                },
            };
            let need_new = !matches!(&self.rc2d_state, Some((s, _)) if *s == rc_size);
            if need_new && let Ok(pass) = RadianceCascades2d::new(&self.device, rc_size, rc_cfg) {
                self.rc2d_state = Some((rc_size, pass));
            }
            if let Some((_, rc)) = &self.rc2d_state {
                rc.build(&self.device, encoder, &scene_view);
            }
        }

        (base, overlay)
    }
}

#[cfg(feature = "render-2d")]
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits & 0x007f_ffff) >> 13;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7c00
    } else {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            sign | ((exp as u16) << 10) | (mant as u16)
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn spawn_load(fut: impl std::future::Future<Output = ()> + 'static) {
    wasm_bindgen_futures::spawn_local(fut);
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_load(fut: impl std::future::Future<Output = ()> + 'static) {
    pollster::block_on(fut);
}
