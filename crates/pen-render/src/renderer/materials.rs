// What a frame's draws bind: their materials, the shaders' pipelines for
// the frame's sample count, and bind groups of their textures with the
// sampler their filtering and settling detail ask for.
use std::sync::Arc;

use anyhow::Result;
use pen_protocol::{Catalog, RenderFrame};

use super::{DRAW_BYTES, GpuAsset, MaterialAsset, Scratch, State, TextureAsset, TextureRef};
use crate::asset_store::{AssetKind, Material};
use crate::built_in::BuiltIn;
use crate::shader::{Pipelines, Variants};

// Texture filtering as the frame's quality gives it: 0 reads only the
// finest level there is, 1 is trilinear, more is that many anisotropic
// samples. `blur` keeps it that many levels off the finest, while a level
// that has just arrived fades in.
pub(super) fn sampler_for(device: &wgpu::Device, filtering: u32, blur: f32) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: if filtering == 0 { wgpu::MipmapFilterMode::Nearest } else { wgpu::MipmapFilterMode::Linear },
        lod_min_clamp: if filtering == 0 { 0.0 } else { blur },
        lod_max_clamp: if filtering == 0 { 0.0 } else { 32.0 },
        // WebGPU clamps maxAnisotropy to what it supports; wgpu takes 1..=16.
        // Written as a clamp: Rust 1.99's optimizer turned the equivalent
        // `if filtering > 1 { filtering.min(16) } else { 1 }` into
        // `filtering.min(16)`, an invalid 0 for unfiltered frames.
        anisotropy_clamp: filtering.clamp(1, 16) as u16,
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        ..Default::default()
    })
}

impl State {
    /// The asset's slot, asking for it if it is not loaded yet; notes when a
    /// frame waits on it.
    pub(super) fn resolve(&mut self, kind: AssetKind, key: &str) -> Result<usize> {
        let slot = self.assets.get(kind, key)?;
        if self.assets.current(slot).is_none() && self.assets.loading(slot) {
            self.waiting = true;
        }
        Ok(slot)
    }

    pub(super) fn material(&mut self, catalog: &Catalog, index: u32) -> Result<Option<Arc<MaterialAsset>>> {
        let slot = self.resolve(AssetKind::Material, catalog.material(index))?;
        Ok(match self.assets.current(slot) {
            Some(GpuAsset::Material(material)) => Some(material.clone()),
            _ => None,
        })
    }

    /// The shader's slot, if it is loaded.
    pub(super) fn shader(&mut self, material: &Material) -> Result<Option<usize>> {
        let slot = self.resolve(AssetKind::Shader, &material.shader)?;
        match self.assets.current(slot) {
            Some(GpuAsset::Shader(shader)) if shader.shadow_compiling.contains(&true) && material.casts_shadow() => {
                self.waiting = true;
                Ok(None)
            }
            Some(GpuAsset::Shader(_)) => Ok(Some(slot)),
            _ => Ok(None),
        }
    }

    pub(super) fn texture(&self, texture: TextureRef) -> &TextureAsset {
        match texture {
            TextureRef::Slot(slot) => match self.assets.current(slot) {
                Some(GpuAsset::Texture(texture)) => texture,
                _ => unreachable!("a texture reference names a loaded texture"),
            },
            TextureRef::White => &self.white,
            TextureRef::FlatNormal => &self.flat_normal,
            TextureRef::Transparent => &self.transparent,
            TextureRef::Checker => &self.checker,
            TextureRef::Field(index) => &self.fields[index].1,
        }
    }

    /// The loaded texture of `key`, if any, asking for it if need be.
    pub(super) fn loaded_texture(&mut self, key: &Option<String>) -> Result<Option<usize>> {
        let Some(key) = key else { return Ok(None) };
        let slot = self.resolve(AssetKind::Texture, key)?;
        Ok(matches!(self.assets.current(slot), Some(GpuAsset::Texture(_))).then_some(slot))
    }

    /// Whether the shader has `variants` for `samples` per pixel. If not,
    /// they compile on the loader, with whatever else its materials draw
    /// with there, and the frame waits: a newly visible shader may have
    /// loaded at another sample count, or the settings changed it.
    pub(super) fn ensure_variants(&mut self, slot: usize, samples: u32, variants: Variants) -> bool {
        let Some(GpuAsset::Shader(shader)) = self.assets.current_mut(slot) else { return false };
        let made = shader.made.get(&samples).map_or(Variants::NONE, Pipelines::variants);
        if made.contains(shader.own(variants)) {
            return true;
        }
        shader.request(&self.loader.jobs, slot, samples, variants | shader.needs);
        self.waiting = true;
        false
    }

    pub(super) fn pipelines(&self, slot: usize, samples: u32) -> &Pipelines {
        match self.assets.current(slot) {
            Some(GpuAsset::Shader(shader)) => &shader.made[&samples],
            _ => unreachable!("pipelines of a loaded shader"),
        }
    }

    pub(super) fn sampler_at(&mut self, blur: f64) -> (i32, wgpu::Sampler) {
        #[allow(clippy::cast_possible_truncation, reason = "a few mip levels, in quarters")]
        let key = if self.filtering == 0 { 0 } else { (blur * 4.0).ceil() as i32 };
        let (device, filtering) = (&self.device, self.filtering);
        let sampler = self.samplers.entry(key).or_insert_with(|| sampler_for(device, filtering, key as f32 / 4.0));
        (key, sampler.clone())
    }

    /// The textures of a draw's material, or None while one is on its way:
    /// nothing is drawn half-dressed. Also whether the normal and layer
    /// textures loaded, as the draw's settings tell the shader.
    pub(super) fn dress(
        &mut self,
        material: &Material,
        field: Option<usize>,
    ) -> Result<Option<([TextureRef; 3], f32, f32)>> {
        let loaded_texture = self.loaded_texture(&material.texture)?;
        let loaded_normal = self.loaded_texture(&material.normal_texture)?;
        let loaded_outline = self.loaded_texture(&material.layer_texture)?;
        // A texture on its way holds the draw back; one that failed shows as
        // the checker.
        if [&material.texture, &material.normal_texture, &material.layer_texture]
            .into_iter()
            .flatten()
            .any(|key| self.assets.loading_key(AssetKind::Texture, key))
        {
            return Ok(None);
        }
        let mut texture = match (&material.texture, loaded_texture) {
            (None, _) => TextureRef::White,
            (Some(_), Some(slot)) => TextureRef::Slot(slot),
            (Some(_), None) => TextureRef::Checker,
        };
        let mut normal = match field {
            Some(field) => TextureRef::Field(field),
            None => loaded_normal.map_or(TextureRef::FlatNormal, TextureRef::Slot),
        };
        let mut outline = loaded_outline.map_or(TextureRef::Transparent, TextureRef::Slot);
        // Index and palette must become usable together: an ordinary fallback
        // texture must never be interpreted as palette indices or oct normals.
        let indexed = material.indexed;
        let packed_ready =
            indexed && loaded_texture.is_some() && loaded_normal.is_some() && loaded_outline.is_some() && {
                let (index, normals, palette) =
                    (&self.texture(texture).texture, &self.texture(normal).texture, &self.texture(outline).texture);
                index.format() == wgpu::TextureFormat::R8Unorm
                    && normals.format() == wgpu::TextureFormat::Rg8Unorm
                    && palette.format() == wgpu::TextureFormat::Rgba8UnormSrgb
                    && palette.width() == 256
                    && palette.height() == 2
                    && index.width() == normals.width()
                    && index.height() == normals.height()
            };
        if indexed && !packed_ready {
            texture = TextureRef::Checker;
            normal = TextureRef::FlatNormal;
            outline = TextureRef::Transparent;
        }
        let normal_setting = if indexed {
            if packed_ready { 1.0 } else { 0.0 }
        } else if loaded_normal.is_some() {
            1.0
        } else {
            0.0
        };
        let outline_setting = if indexed {
            if packed_ready { 2.0 } else { 0.0 }
        } else if loaded_outline.is_some() {
            1.0
        } else {
            0.0
        };
        Ok(Some(([texture, normal, outline], normal_setting, outline_setting)))
    }

    /// The draw bind group for these textures, with the sampler their
    /// settling detail asks for.
    pub(super) fn draw_group(&mut self, textures: [TextureRef; 3]) -> wgpu::BindGroup {
        for texture in textures {
            if let TextureRef::Slot(slot) = texture
                && let Some(GpuAsset::Texture(texture)) = self.assets.current_mut(slot)
            {
                texture.seen = true;
            }
        }
        let blur = textures
            .iter()
            .map(|&texture| {
                let texture = self.texture(texture);
                texture.shown - f64::from(texture.resident)
            })
            .fold(f64::NEG_INFINITY, f64::max);
        let (sampler_key, sampler) = self.sampler_at(blur);
        // Every texture's resident levels participate in cache identity.
        // Reloading any layer, or a finer level arriving, replaces its group.
        let [texture_id, normal_id, outline_id] = textures.map(|texture| self.texture(texture).view_id);
        let key = (texture_id, normal_id, outline_id, sampler_key);
        let frame = self.frames;
        if let Some((group, used)) = self.groups.get_mut(&key) {
            *used = frame;
            return group.clone();
        }
        let [texture, normal, outline] = textures.map(|texture| self.texture(texture));
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.context.draw_group,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.draw_uniform,
                        offset: 0,
                        size: wgpu::BufferSize::new(DRAW_BYTES),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&texture.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&normal.linear) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&outline.view) },
            ],
        });
        self.groups.insert(key, (group.clone(), frame));
        group
    }

    /// Takes in the frame's texture filtering and sample count, and returns
    /// the sample count it draws with.
    pub(super) fn quality(&mut self, frame: &RenderFrame, catalog: &Catalog, scratch: &mut Scratch) -> Result<u32> {
        if frame.quality.filtering != self.filtering {
            self.filtering = frame.quality.filtering;
            self.samplers.clear();
            // Draw bind groups hold the sampler.
            self.groups.clear();
        }
        // A new sample count waits until each shader this frame draws with,
        // and the passes around the scene, have compiled their pipelines for
        // it in the background; until then the frame keeps the count it had.
        // The first frame takes its count at once: nothing compiled before.
        self.samples_now = frame.quality.msaa;
        if self.frames == 1 {
            self.samples_shown = self.samples_now;
        }
        self.built_ins.prepare(BuiltIn::all(self.samples_now));
        if self.samples_now != self.samples_shown {
            let mut compiled = self.built_ins.ready(BuiltIn::all(self.samples_now));
            let materials = &mut scratch.materials;
            materials.extend(frame.draws.iter().map(|draw| draw.material));
            materials.extend(frame.solids.iter().flat_map(|solid| solid.faces.iter().map(|face| face.material)));
            materials.sort_unstable();
            materials.dedup();
            for index in std::mem::take(materials) {
                let Some(material) = self.material(catalog, index)? else { continue };
                let Some(slot) = self.shader(&material.material)? else { continue };
                let samples = self.samples_now;
                if !self.ensure_variants(slot, samples, material.material.variants()) {
                    compiled = false;
                }
            }
            if compiled {
                self.samples_shown = self.samples_now;
            } else {
                self.waiting = true;
            }
        }
        Ok(self.samples_shown)
    }
}
