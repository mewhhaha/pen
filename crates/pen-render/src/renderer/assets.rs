// Assets arriving from the loader: made into GPU assets on the thread that
// submits frames, shaders staged until their pipelines compile, packed
// textures streaming in their levels, smallest first, and detail fading in.
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc::TryRecvError;

use anyhow::{Result, ensure};
use pen_protocol::{Catalog, GameEvent};
use wgpu::util::DeviceExt;

use super::{
    Checking, GpuAsset, INSET_SAMPLES, MaterialAsset, MeshAsset, MeshSkin, PendingTexture, ShaderAsset, Staged, State,
    TextureAsset, extent,
};
use crate::asset_store::AssetKind;
use crate::built_in::BuiltIn;
use crate::loader::{Job, JobQueue, LevelMessage, Loaded, Message, Priority};
use crate::mipmaps::Mipmapper;
use crate::packed_texture::{PackedLevel, mip_levels, texel_bytes};
use crate::shader::{self, Pipelines, Stage, Variants};

// How long (seconds) detail takes to settle once a finer level arrives.
const SHARPENING: f64 = 0.12;

// A streamed texture shows once its levels up to 128 texels across are in.
const SHOWN_LEVELS: u32 = 8;

impl ShaderAsset {
    /// `variants` as it has them: instanced only with `vertex_instanced`,
    /// skinned only with `vertex_skinned`.
    pub(super) fn own(&self, variants: Variants) -> Variants {
        let variants = if self.instancing { variants } else { variants.without(Variants::INSTANCED) };
        if self.skinning { variants } else { variants.without(Variants::SKINNED) }
    }

    /// Background compiles under way.
    pub(super) fn compiling(&self) -> usize {
        let shadows = self.shadow_compiling.iter().filter(|&&compiling| compiling).count();
        self.compiling.values().map(|variants| variants.each().count()).sum::<usize>() + shadows
    }

    /// Compiles in the background those of `variants` for `samples` it has
    /// not and is not compiling, each pipeline in a job of its own, so that
    /// they compile side by side.
    pub(super) fn request(&mut self, jobs: &JobQueue, slot: usize, samples: u32, variants: Variants) {
        let made = self.made.get(&samples).map_or(Variants::NONE, Pipelines::variants);
        let failed = self.failed.get(&samples).copied().unwrap_or_default();
        let missing = self.own(variants).without(made).without(failed);
        let compiling = self.compiling.entry(samples).or_default();
        let missing = missing.without(*compiling);
        *compiling |= missing;
        for variant in missing.each() {
            let (revision, module) = (self.revision, self.module.clone());
            let (instancing, skinning) = (self.instancing, self.skinning);
            jobs.push(
                Job::Compile { slot, revision, samples, variants: variant, module, instancing, skinning },
                Priority::Urgent,
            );
        }
    }

    /// Compiles its depth-only pipelines in the background, unless they are
    /// there or on their way.
    pub(super) fn request_shadow(&mut self, jobs: &JobQueue, slot: usize) {
        for stage in Stage::ALL {
            let index = stage as usize;
            let lacks = match stage {
                Stage::Plain => false,
                Stage::Instanced => !self.instancing,
                Stage::Skinned => !self.skinning,
            };
            if lacks || self.shadow[index].is_some() || self.shadow_compiling[index] {
                continue;
            }
            self.shadow_compiling[index] = true;
            let (revision, module) = (self.revision, self.module.clone());
            jobs.push(Job::Shadow { slot, revision, module, stage }, Priority::Urgent);
        }
    }

    /// Whether its compiles for `samples` and its shadows have all ended.
    pub(super) fn settled(&self, samples: u32) -> bool {
        self.compiling.get(&samples).is_none_or(|variants| variants.is_empty())
            && !self.shadow_compiling.contains(&true)
    }
}

pub(super) fn texture_views(texture: &wgpu::Texture, finest: u32) -> (wgpu::TextureView, wgpu::TextureView) {
    let levels = |format: Option<wgpu::TextureFormat>| wgpu::TextureViewDescriptor {
        format,
        base_mip_level: finest,
        mip_level_count: Some(texture.mip_level_count() - finest),
        ..Default::default()
    };
    let linear = texture.format().remove_srgb_suffix();
    (texture.create_view(&levels(None)), texture.create_view(&levels(Some(linear))))
}

/// The formats a texture of `format` is also viewed in: an sRGB colour texture
/// (a compressed one too) is read as linear data too (normal maps).
fn view_formats(format: wgpu::TextureFormat) -> Vec<wgpu::TextureFormat> {
    if format.is_srgb() { vec![format.remove_srgb_suffix()] } else { Vec::new() }
}

/// A texture with all its levels there.
pub(super) fn whole(texture: wgpu::Texture, view_id: u64) -> TextureAsset {
    let (view, linear) = texture_views(&texture, 0);
    TextureAsset { texture, view, linear, view_id, resident: 0, shown: 0.0, seen: false, stream: None }
}

/// Writes `level`, whole blocks of the texture's format (texels, for an
/// uncompressed one): a level smaller than a block is still a whole block.
pub(super) fn upload_level(queue: &wgpu::Queue, texture: &wgpu::Texture, level: &PackedLevel) {
    let format = texture.format();
    let (block_width, block_height) = format.block_dimensions();
    let block_bytes = format.block_copy_size(None).unwrap_or(4);
    let (columns, rows) = (level.width.div_ceil(block_width), level.height.div_ceil(block_height));
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: level.level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &level.pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(columns * block_bytes),
            rows_per_image: Some(rows),
        },
        extent(columns * block_width, rows * block_height),
    );
}

// Mipmapped textures get every smaller level drawn after their upload.
pub(super) fn texture_from(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mipmaps: &mut Mipmapper,
    (width, height): (u32, u32),
    pixels: &[u8],
    format: wgpu::TextureFormat,
    mipmapped: bool,
) -> Result<wgpu::Texture> {
    let limit = device.limits().max_texture_dimension_2d;
    ensure!(width <= limit && height <= limit, "texture exceeds device dimensions");
    let levels = if mipmapped { mip_levels(width, height) } else { 1 };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: extent(width, height),
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | if levels > 1 { wgpu::TextureUsages::RENDER_ATTACHMENT } else { wgpu::TextureUsages::empty() },
        // Reinterpret the same allocation as linear data for object-space normals.
        view_formats: &view_formats(format),
    });
    queue.write_texture(
        texture.as_image_copy(),
        pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * texel_bytes(format)),
            rows_per_image: Some(height),
        },
        extent(width, height),
    );
    if levels > 1 {
        mipmaps.generate(device, queue, &texture);
    }
    Ok(texture)
}

impl State {
    /// Loads the catalog's meshes and materials, and the materials' textures,
    /// ahead of need, behind whatever a frame asks for, so nothing pops in
    /// when first drawn.
    pub fn preload(&mut self, catalog: &Catalog) {
        // A scene's own assets first: the rest waits for them.
        if self.scene.iter().any(|&slot| self.assets.loading(slot)) {
            self.deferred = Some(catalog.clone());
            return;
        }
        self.preloading = true;
        for key in &catalog.meshes {
            if let Err(error) = self.assets.prefetch(AssetKind::Mesh, key) {
                self.reject(AssetKind::Mesh, key, error);
            }
        }
        for key in &catalog.materials {
            if let Err(error) = self.assets.prefetch(AssetKind::Material, key) {
                self.reject(AssetKind::Material, key, error);
            }
        }
    }

    /// Loads a scene's assets now, ahead of the frames that draw them, all
    /// side by side: `(kind, key)` pairs, kinds named as `AssetKind::name`
    /// says (a material's shader and textures among them, so they need not
    /// wait for it). The catalog's preload waits until they are in.
    pub fn preload_scene(&mut self, assets: &[(String, String)], samples: u32) {
        // The first frame takes its sample count at once (`quality`): the
        // scene's shaders and the passes round it compile for it now.
        if samples > 0 && self.frames == 0 {
            self.samples_now = samples;
            self.built_ins.prepare(BuiltIn::all(samples));
        }
        for (kind, key) in assets {
            let Some(kind) = AssetKind::named(kind) else { continue };
            match self.assets.get(kind, key) {
                Ok(slot) => self.scene.push(slot),
                Err(error) => self.reject(kind, key, error),
            }
        }
    }

    /// Has first loads take what `prefetch` read and decoded ahead.
    pub fn warm(&mut self, prefetch: crate::prefetch::Prefetch) {
        *self.loader.prefetch.lock().unwrap() = Some(prefetch);
    }

    /// Reports an asset that cannot even be asked for.
    pub(super) fn reject(&mut self, kind: AssetKind, key: &str, error: anyhow::Error) {
        self.assets.events.push(GameEvent::AssetFailed {
            key: key.to_owned(),
            kind: kind.name().to_owned(),
            revision: 0,
            message: format!("{error:#}"),
        });
    }

    pub(super) fn prefetch_shader(&mut self, key: &str) {
        if !self.sorted.insert(key.to_owned()) {
            return;
        }
        self.loader.jobs.push(Job::Sort { key: key.to_owned() }, Priority::Background);
    }

    /// Takes in one thing the loader brought.
    pub(super) fn take(&mut self, message: Message, now: f64) {
        match message {
            Message::Loaded { slot, revision, early, result } => {
                let prepared = result.and_then(|loaded| self.prepare(slot, revision, early, loaded, now));
                match prepared {
                    // A packed texture finishes loading once its first levels are in.
                    Ok(None) => {}
                    // Not published until its allocation and upload have
                    // passed the error scopes (at once natively).
                    Ok(Some((asset, Some(mut check)))) => match shader::settled(&mut check) {
                        Some(Ok(())) => {
                            let previous = self.assets.finish(slot, revision, early, Ok(asset), now);
                            self.retire(previous);
                        }
                        Some(Err(message)) => {
                            let previous = self.assets.finish(slot, revision, early, Err(message), now);
                            self.retire(previous);
                        }
                        None => self.checking.push(Checking { slot, revision, early, asset, check }),
                    },
                    Ok(Some((asset, None))) => {
                        let previous = self.assets.finish(slot, revision, early, Ok(asset), now);
                        self.retire(previous);
                    }
                    Err(message) => {
                        let previous = self.assets.finish(slot, revision, early, Err(message), now);
                        self.retire(previous);
                    }
                }
            }
            Message::Sorted { key, sprite } => {
                let kind = if sprite { AssetKind::SpriteShader } else { AssetKind::Shader };
                let _ = self.assets.prefetch(kind, &key);
            }
            Message::Compiled { slot, revision, samples, variants, result } => {
                let failure = self.compiled(slot, revision, |shader| {
                    let compiling = shader.compiling.entry(samples).or_default();
                    *compiling = compiling.without(variants);
                    match result {
                        Ok(pipelines) => {
                            shader.made.entry(samples).or_default().merge(pipelines);
                            None
                        }
                        Err(message) => {
                            *shader.failed.entry(samples).or_default() |= variants;
                            Some(message)
                        }
                    }
                });
                self.compiled_out(slot, revision, failure, now);
            }
            Message::BuiltIn { which, result } => self.built_ins.arrived(which, result),
            Message::Culling(result) => self.culling.arrived(result),
            // Taken in with the others' levels.
            Message::Level => {}
            Message::Shadow { slot, revision, stage, result } => {
                let failure = self.compiled(slot, revision, |shader| {
                    shader.shadow_compiling[stage as usize] = false;
                    match result {
                        Ok(pipeline) => {
                            shader.shadow[stage as usize] = Some(pipeline);
                            None
                        }
                        Err(message) => Some(message),
                    }
                });
                self.compiled_out(slot, revision, failure, now);
            }
        }
    }

    /// Applies a background compile's outcome to the shader revision it
    /// answers, staged or published, if that is still there. `apply` says
    /// why it failed, if it did; a published shader's failure comes back.
    pub(super) fn compiled(
        &mut self,
        slot: usize,
        revision: u64,
        apply: impl FnOnce(&mut ShaderAsset) -> Option<String>,
    ) -> Option<String> {
        if let Some(staged) = self.staging.get_mut(&slot).filter(|staged| staged.shader.revision == revision) {
            let failure = apply(&mut staged.shader);
            staged.failure = staged.failure.take().or(failure);
            return None;
        }
        match self.assets.current_mut(slot) {
            Some(GpuAsset::Shader(shader)) if shader.revision == revision => apply(shader),
            _ => None,
        }
    }

    /// After a compile: a published shader's failure is reported; a staged
    /// one is published, or fails, once its compiles have ended.
    pub(super) fn compiled_out(&mut self, slot: usize, revision: u64, failure: Option<String>, now: f64) {
        if let Some(message) = failure {
            self.assets.events.push(GameEvent::AssetFailed {
                key: self.assets.slots[slot].key.clone(),
                kind: AssetKind::Shader.name().to_owned(),
                revision,
                message,
            });
        }
        if self
            .staging
            .get(&slot)
            .is_some_and(|staged| staged.shader.revision == revision && staged.shader.settled(staged.samples))
        {
            let Staged { shader, early, failure, .. } = self.staging.remove(&slot).unwrap();
            let published = match failure {
                Some(message) => Err(message),
                None => Ok(GpuAsset::Shader(shader)),
            };
            let previous = self.assets.finish(slot, revision, early, published, now);
            self.retire(previous);
        }
    }

    /// Takes in whatever the loader has brought since the last frame.
    pub(super) fn receive(&mut self, now: f64) {
        while let Ok(message) = self.loader.results.try_recv() {
            self.take(message, now);
        }
        // Assets whose error scopes came back since the last frame.
        let mut index = 0;
        while index < self.checking.len() {
            let Some(result) = shader::settled(&mut self.checking[index].check) else {
                index += 1;
                continue;
            };
            let Checking { slot, revision, early, asset, .. } = self.checking.swap_remove(index);
            let previous = self.assets.finish(slot, revision, early, result.map(|()| asset), now);
            self.retire(previous);
        }
        self.receive_levels(now);
        // The scene's assets are in: the rest of the catalog may come.
        if self.deferred.is_some() && !self.scene.iter().any(|&slot| self.assets.loading(slot)) {
            self.scene.clear();
            let catalog = self.deferred.take().unwrap();
            self.preload(&catalog);
        }
    }

    /// Waits up to `timeout` for the loader to bring something (a load, a
    /// compiled pipeline), takes it in, and says whether anything came. In
    /// the browser it only looks: the page's thread must not block.
    pub fn wait(&mut self, timeout: std::time::Duration) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        let message = self.loader.results.recv_timeout(timeout).ok();
        #[cfg(target_arch = "wasm32")]
        let message = {
            let _ = timeout;
            self.loader.results.try_recv().ok()
        };
        let Some(message) = message else { return false };
        let now = self.now();
        self.take(message, now);
        self.receive(now);
        true
    }

    /// Makes what the loader brought into a GPU asset; None for a packed
    /// texture, which shows once its first levels are in.
    pub(super) fn prepare(
        &mut self,
        slot: usize,
        revision: u64,
        early: bool,
        loaded: Loaded,
        now: f64,
    ) -> Result<Option<(GpuAsset, Option<shader::Check>)>, String> {
        let device = self.device.clone();
        match loaded {
            Loaded::Mesh(mesh) => {
                let vertex_bytes: &[u8] = bytemuck::cast_slice(&mesh.vertices);
                let index_bytes: &[u8] = bytemuck::cast_slice(&mesh.indices);
                if vertex_bytes.len().max(index_bytes.len()) as u64 > device.limits().max_buffer_size {
                    return Err("mesh exceeds device buffer size".into());
                }
                let buffer = |bytes: &[u8], usage: wgpu::BufferUsages| {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytes, usage })
                };
                // A skinned mesh's joints and weights, beside its vertices
                // (`shader::SKIN_VERTEX_BYTES` a vertex).
                let skin_bytes: Option<Vec<u8>> = mesh.skin.as_ref().map(|skin| {
                    let mut bytes = Vec::with_capacity(skin.joints.len() * 24);
                    for (joints, weights) in skin.joints.iter().zip(&skin.weights) {
                        bytes.extend_from_slice(bytemuck::cast_slice(joints));
                        bytes.extend_from_slice(bytemuck::cast_slice(weights));
                    }
                    bytes
                });
                let (asset, check) = shader::scoped(&device, || {
                    GpuAsset::Mesh(MeshAsset {
                        vertices: buffer(vertex_bytes, wgpu::BufferUsages::VERTEX),
                        indices: buffer(index_bytes, wgpu::BufferUsages::INDEX),
                        bounds: mesh.bounds,
                        lods: mesh.lods.clone(),
                        skin: mesh.skin.zip(skin_bytes).map(|(skin, bytes)| MeshSkin {
                            vertices: buffer(&bytes, wgpu::BufferUsages::VERTEX),
                            rig: Arc::new(skin.rig),
                        }),
                    })
                });
                Ok(Some((asset, Some(check))))
            }
            Loaded::Material(material) => {
                // A material arriving after its shader asks for the variants
                // it draws with, ahead of the frames that draw it.
                if let Some(slot) = self.assets.id(AssetKind::Shader, &material.shader) {
                    let jobs = &self.loader.jobs;
                    if let Some(GpuAsset::Shader(shader)) = self.assets.current_mut(slot) {
                        shader.needs |= material.variants();
                        for samples in [self.samples_shown, self.samples_now] {
                            shader.request(jobs, slot, samples, shader.needs);
                        }
                        if material.casts_shadow() {
                            shader.request_shadow(jobs, slot);
                        }
                    }
                    if let Some(Staged { shader, samples, .. }) = self.staging.get_mut(&slot) {
                        shader.needs |= material.variants();
                        shader.request(jobs, slot, *samples, shader.needs);
                        if material.casts_shadow() {
                            shader.request_shadow(jobs, slot);
                        }
                    }
                }
                if self.preloading {
                    for key in
                        [&material.texture, &material.normal_texture, &material.layer_texture].into_iter().flatten()
                    {
                        let _ = self.assets.prefetch(AssetKind::Texture, key);
                    }
                    self.prefetch_shader(&material.shader);
                }
                let next = u32::try_from(self.states.len()).map_err(|error| error.to_string())?;
                let state = *self.states.entry(material.state_key()).or_insert(next);
                Ok(Some((GpuAsset::Material(Arc::new(MaterialAsset { material, state })), None)))
            }
            Loaded::Shader { module, instancing, skinning, margin, cascaded } => {
                // What the materials naming it draw with compiles now, for
                // the sample count frames draw with, each pipeline on a
                // thread of its own; other variants and counts when a
                // material or the settings first ask for them. A caster's
                // shadows are part of it: a hot reload's old shader draws on
                // until the new one has them.
                let need = self.loader.needs.lock().unwrap().of(&self.assets.slots[slot].key);
                let needs = if need.variants.is_empty() { Variants::OPAQUE } else { need.variants };
                let mut shader = ShaderAsset {
                    revision,
                    module,
                    instancing,
                    skinning,
                    cascaded,
                    margin,
                    needs,
                    made: HashMap::new(),
                    compiling: HashMap::new(),
                    failed: HashMap::new(),
                    shadow: Default::default(),
                    shadow_compiling: [false; 3],
                };
                let (jobs, samples) = (&self.loader.jobs, self.samples_now);
                shader.request(jobs, slot, samples, needs);
                if need.shadow {
                    shader.request_shadow(jobs, slot);
                }
                // Insets draw with 4 samples a pixel: their pipeline compiles
                // in the background now, so an inset first shown never waits
                // on it.
                // Their draws take the opaque or translucent pipeline alone.
                let inset_needs =
                    needs.without(Variants::INSTANCED | Variants::OCCLUDED | Variants::VISIBLE | Variants::SKINNED);
                let inset_needs = if inset_needs.is_empty() { Variants::OPAQUE } else { inset_needs };
                shader.request(jobs, slot, INSET_SAMPLES, inset_needs);
                let staged = Staged { shader, early, samples, failure: None };
                // A revision this one overtook is let go.
                if let Some(Staged { shader, early, .. }) = self.staging.insert(slot, staged) {
                    let _ = self.assets.finish(slot, shader.revision, early, Err(String::new()), now);
                }
                Ok(None)
            }
            Loaded::SpriteShader(pipeline) => Ok(Some((GpuAsset::SpriteShader(pipeline), None))),
            Loaded::Image(image) => {
                let id = self.view_id();
                let (queue, mipmaps) = (&self.queue, &mut self.mipmaps);
                let (texture, check) = shader::scoped(&device, || {
                    texture_from(
                        &device,
                        queue,
                        mipmaps,
                        (image.width, image.height),
                        &image.pixels,
                        wgpu::TextureFormat::Rgba8UnormSrgb,
                        true,
                    )
                });
                let texture = texture.map_err(|error| format!("{error:#}"))?;
                Ok(Some((GpuAsset::Texture(whole(texture, id)), Some(check))))
            }
            Loaded::Packed { header, levels } => {
                let limit = device.limits().max_texture_dimension_2d;
                if header.width > limit || header.height > limit {
                    return Err("texture exceeds device dimensions".into());
                }
                let (texture, mut check) = shader::scoped(&device, || {
                    device.create_texture(&wgpu::TextureDescriptor {
                        label: None,
                        size: extent(header.width, header.height),
                        mip_level_count: header.levels,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: header.format,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        // Reinterpret the same allocation as linear data for normal maps.
                        view_formats: &view_formats(header.format),
                    })
                });
                let check = match shader::settled(&mut check) {
                    Some(result) => result.map(|()| None)?,
                    None => Some(check),
                };
                self.pending.push(PendingTexture {
                    check,
                    slot,
                    revision,
                    early,
                    texture,
                    header,
                    finest: header.levels,
                    levels,
                });
                Ok(None)
            }
        }
    }

    /// Uploads the levels that have arrived: a pending texture's first ones,
    /// until it shows, and a shown one's finer ones, which fade in.
    pub(super) fn receive_levels(&mut self, now: f64) {
        let mut index = 0;
        while index < self.pending.len() {
            let pending = &mut self.pending[index];
            let shown_from = pending.header.levels.saturating_sub(SHOWN_LEVELS);
            let mut outcome: Option<Result<(), String>> = None;
            // Its levels wait until its allocation has passed the error
            // scopes (at once natively).
            if let Some(check) = &mut pending.check {
                match shader::settled(check) {
                    None => {
                        index += 1;
                        continue;
                    }
                    Some(Ok(())) => pending.check = None,
                    Some(Err(message)) => outcome = Some(Err(message)),
                }
            }
            while outcome.is_none() && pending.finest > shown_from {
                match pending.levels.try_recv() {
                    Ok(LevelMessage::Level(level)) => {
                        upload_level(&self.queue, &pending.texture, &level);
                        pending.finest = level.level;
                    }
                    Ok(LevelMessage::Done) | Err(TryRecvError::Disconnected) => {
                        outcome = Some(Err("packed texture ends early".into()));
                        break;
                    }
                    Ok(LevelMessage::Failed(message)) => {
                        outcome = Some(Err(message));
                        break;
                    }
                    Err(TryRecvError::Empty) => break,
                }
            }
            if outcome.is_none() && pending.finest <= shown_from {
                outcome = Some(Ok(()));
            }
            let Some(outcome) = outcome else {
                index += 1;
                continue;
            };
            let pending = self.pending.swap_remove(index);
            match outcome {
                Err(message) => {
                    let previous = self.assets.finish(pending.slot, pending.revision, pending.early, Err(message), now);
                    self.retire(previous);
                }
                Ok(()) => {
                    let (view, linear) = texture_views(&pending.texture, pending.finest);
                    let asset = TextureAsset {
                        texture: pending.texture,
                        view,
                        linear,
                        view_id: self.view_id(),
                        resident: pending.finest,
                        shown: f64::from(pending.finest),
                        seen: false,
                        stream: Some(pending.levels),
                    };
                    let current = self.assets.slots[pending.slot].revision == pending.revision;
                    let previous = self.assets.finish(
                        pending.slot,
                        pending.revision,
                        pending.early,
                        Ok(GpuAsset::Texture(asset)),
                        now,
                    );
                    self.retire(previous);
                    // The rest, in the background: each finer level joins the
                    // views and fades in. (A revision it replaced may still be
                    // listed; its stream went with it.)
                    if current && !self.streaming.contains(&pending.slot) {
                        self.streaming.push(pending.slot);
                    }
                }
            }
        }
        let mut index = 0;
        while index < self.streaming.len() {
            let slot = self.streaming[index];
            let key = self.assets.slots[slot].key.clone();
            let mut new_views = false;
            let mut ended = false;
            let mut old_views = None;
            if let Some(GpuAsset::Texture(texture)) = self.assets.slots[slot].current.as_mut() {
                old_views = Some(texture.view_id);
                loop {
                    let Some(levels) = &texture.stream else {
                        ended = true;
                        break;
                    };
                    match levels.try_recv() {
                        Ok(LevelMessage::Level(level)) => {
                            upload_level(&self.queue, &texture.texture, &level);
                            (texture.view, texture.linear) = texture_views(&texture.texture, level.level);
                            texture.resident = level.level;
                            new_views = true;
                            // A level nobody saw the coarser one before needs
                            // no fading in.
                            if !texture.seen {
                                texture.shown = f64::from(level.level);
                            }
                        }
                        Ok(LevelMessage::Done) | Err(TryRecvError::Disconnected) => {
                            texture.stream = None;
                        }
                        Ok(LevelMessage::Failed(message)) => {
                            log::warn!("texture {key}: {message}");
                            texture.stream = None;
                        }
                        Err(TryRecvError::Empty) => break,
                    }
                }
            } else {
                ended = true;
            }
            if new_views {
                // Groups of the coarser views are not drawn with again.
                if let Some(old) = old_views {
                    self.forget_views(old);
                }
                let id = self.view_id();
                if let Some(GpuAsset::Texture(texture)) = self.assets.slots[slot].current.as_mut() {
                    texture.view_id = id;
                    if texture.shown > f64::from(texture.resident) && !self.fading.contains(&slot) {
                        self.fading.push(slot);
                    }
                }
            }
            if ended {
                self.streaming.swap_remove(index);
            } else {
                index += 1;
            }
        }
    }

    /// Lets go of what depended on an asset a new revision replaced. Shader,
    /// sprite shader, mesh and material assets carry their own pipelines and
    /// buffers, and frames look them up afresh, so dropping them is enough;
    /// a texture's views are also held by cached bind groups.
    pub(super) fn retire(&mut self, previous: Option<GpuAsset>) {
        if let Some(GpuAsset::Texture(texture)) = previous {
            self.forget_views(texture.view_id);
        }
    }

    /// Drops the cached draw bind groups that bind views `id`.
    pub(super) fn forget_views(&mut self, id: u64) {
        self.groups.retain(|&(texture, normal, outline, _), _| texture != id && normal != id && outline != id);
    }

    /// Loads the changed `keys` again: every
    /// asset of those keys that was loaded or asked for gets a new revision,
    /// drawn once it is ready; until then, or if it fails, the last one
    /// draws on.
    pub fn reload(&mut self, keys: &[String]) {
        self.assets.reload(keys);
    }

    pub fn reload_all(&mut self) {
        self.assets.reload_all();
    }

    /// Detail that arrived since the last frame settles in.
    pub(super) fn fade(&mut self, now: f64) {
        let step = (now - self.last_time.unwrap_or(now)).min(0.1);
        self.last_time = Some(now);
        let assets = &mut self.assets;
        let mut settled = Vec::new();
        self.fading.retain(|&slot| {
            let Some(GpuAsset::Texture(texture)) = assets.current_mut(slot) else { return false };
            let resident = f64::from(texture.resident);
            texture.shown = resident + (texture.shown - resident) * (-step / SHARPENING).exp();
            if texture.shown - resident < 0.01 {
                texture.shown = resident;
                settled.push(texture.view_id);
                return false;
            }
            true
        });
        // The groups that drew a settled texture's detail fading in are not
        // drawn with again.
        for view in settled {
            self.groups.retain(|&(texture, normal, outline, sampler), _| {
                sampler == 0 || ![texture, normal, outline].contains(&view)
            });
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod shadow_tests {
    use super::*;
    use crate::RendererOptions;
    use std::time::{Duration, Instant};

    const SHADER: &str = r#"
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
    return vec4f(position, 1.0);
}
@vertex fn vertex_instanced(@location(0) position: vec3f) -> @builtin(position) vec4f {
    return vec4f(position, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f { return vec4f(1.0); }
"#;

    fn pump(state: &mut State, ready: impl Fn(&State) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(15);
        while !ready(state) {
            state.receive(state.now());
            assert!(Instant::now() < deadline, "shadow pipeline work did not finish");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    fn shader(state: &State, slot: usize) -> &ShaderAsset {
        let Some(GpuAsset::Shader(shader)) = state.assets.current(slot) else { panic!("shader not loaded") };
        shader
    }

    #[test]
    fn shadows_follow_casting_materials_and_reload_without_frame_thread_compilation() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("shadow pipeline regression requires a GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let root = std::env::temp_dir().join(format!("pen-shadow-test-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("scene.wgsl"), SHADER).unwrap();
        let unlit = br#"{"shader":"scene.wgsl","color":[1,1,1,1],"lighting":"unlit"}"#;
        let lit = br#"{"shader":"scene.wgsl","color":[1,1,1,1],"lighting":"lit"}"#;
        std::fs::write(root.join("paint.json"), unlit).unwrap();
        let mut state = State::new(
            device,
            queue,
            RendererOptions {
                format: wgpu::TextureFormat::Bgra8Unorm,
                display: crate::Display::SDR,
                assets: root.clone(),
                veil: None,
                pipeline_cache: None,
            },
        )
        .unwrap();
        let material_slot = state.assets.get(AssetKind::Material, "paint.json").unwrap();
        pump(&mut state, |state| state.assets.current(material_slot).is_some());
        let shader_slot = state.assets.get(AssetKind::Shader, "scene.wgsl").unwrap();
        pump(&mut state, |state| state.assets.current(shader_slot).is_some());
        assert!(shader(&state, shader_slot).shadow.iter().all(Option::is_none));
        assert!(!shader(&state, shader_slot).shadow_compiling.contains(&true));
        // Only what its materials draw with (an opaque material, alone or
        // instanced), and the inset's opaque pipeline.
        assert_eq!(shader(&state, shader_slot).made[&1].variants(), Variants::OPAQUE | Variants::INSTANCED);
        assert!(!state.ensure_variants(shader_slot, 4, Variants::INSTANCED));
        assert!(!state.ensure_variants(shader_slot, 4, Variants::INSTANCED));
        assert!(shader(&state, shader_slot).compiling[&4].contains(Variants::INSTANCED));
        assert!(state.pending() > 0);
        pump(&mut state, |state| shader(state, shader_slot).made.get(&4).is_some_and(|made| made.instanced.is_some()));
        assert!(state.ensure_variants(shader_slot, 4, Variants::INSTANCED));
        pump(&mut state, |state| shader(state, shader_slot).made[&4].opaque.is_some());
        assert_eq!(shader(&state, shader_slot).made[&4].variants(), Variants::OPAQUE | Variants::INSTANCED);

        // A caster arriving after its shared shader asks for background work.
        let caster_started = Instant::now();
        std::fs::write(root.join("paint.json"), lit).unwrap();
        state.assets.reload(&["paint.json".to_owned()]);
        pump(
            &mut state,
            |state| matches!(state.assets.current(material_slot), Some(GpuAsset::Material(material)) if material.material.lit),
        );
        assert_eq!(state.shader(&crate::asset_store::parse_material(unlit).unwrap()).unwrap(), Some(shader_slot));
        if shader(&state, shader_slot).shadow_compiling.contains(&true) {
            assert!(state.shader(&crate::asset_store::parse_material(lit).unwrap()).unwrap().is_none());
        }
        // Plain and instanced; a shader without `vertex_skinned` has no skinned one.
        // Other background work may still finish after them on a slow device
        // (WARP): wait for it too, which must not need the frame thread.
        pump(&mut state, |state| {
            shader(state, shader_slot).shadow[..2].iter().all(Option::is_some) && state.pending() == 0
        });
        assert!(shader(&state, shader_slot).shadow[2].is_none());
        eprintln!(
            "late caster material and both shadow variants ready: {:.2} ms",
            caster_started.elapsed().as_secs_f64() * 1000.0
        );

        // A known caster's new shader includes shadows before publication.
        state.assets.reload(&["scene.wgsl".to_owned()]);
        assert_eq!(shader(&state, shader_slot).revision, 1);
        assert!(shader(&state, shader_slot).shadow[0].is_some());
        pump(&mut state, |state| shader(state, shader_slot).revision == 2);
        assert!(shader(&state, shader_slot).shadow[0].is_some());
        assert!(!shader(&state, shader_slot).shadow_compiling.contains(&true));
        assert!(shader(&state, shader_slot).shadow[1].is_some());

        // Reloading the last caster as unlit removes the loader's demand.
        std::fs::write(root.join("paint.json"), unlit).unwrap();
        state.assets.reload(&["paint.json".to_owned()]);
        pump(
            &mut state,
            |state| matches!(state.assets.current(material_slot), Some(GpuAsset::Material(material)) if !material.material.lit),
        );
        state.assets.reload(&["scene.wgsl".to_owned()]);
        pump(&mut state, |state| shader(state, shader_slot).revision == 3);
        assert!(shader(&state, shader_slot).shadow.iter().all(Option::is_none));
        drop(state);
        std::fs::remove_dir_all(root).unwrap();
    }
}
