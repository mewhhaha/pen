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

use crate::surface_atlas::{DEFAULT_TILE_SIZE, SurfaceAtlasPacker, SurfaceTile, TileCopy};

pub(super) type Dressed = ([TextureRef; 3], f32, f32, f32, f32, u32, f32);

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
        self.assets.touch(slot, self.frames);
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
            TextureRef::Video(index) => &self.videos[index].1,
            TextureRef::CameraTarget(index) => {
                let entry = &self.camera_targets[index];
                &entry.assets[entry.front]
            }
            TextureRef::SurfaceAtlas(index) => &self.surface_atlases[index],
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
        let depth = self.context.depth_mode();
        let made = shader.made.get(&(samples, depth)).map_or(Variants::NONE, Pipelines::variants);
        if made.contains(shader.own(variants)) {
            return true;
        }
        shader.request(&self.loader.jobs, slot, samples, depth, variants | shader.needs);
        self.waiting = true;
        false
    }

    pub(super) fn pipelines(&self, slot: usize, samples: u32) -> &Pipelines {
        match self.assets.current(slot) {
            Some(GpuAsset::Shader(shader)) => &shader.made[&(samples, self.context.depth_mode())],
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
    /// nothing is drawn half-dressed. Also whether the normal, layer/emissive,
    /// ORM, and emissive textures loaded, as the draw uniform tells the shader.
    pub(super) fn dress(&mut self, material: &Material, field: Option<usize>) -> Result<Option<Dressed>> {
        let video_texture = material.texture.as_deref().and_then(|key| self.video_ids.get(key).copied());
        let reserved_video = video_texture.is_none()
            && material.texture.as_deref().is_some_and(|key| self.reserved_videos.contains(key));
        let camera_texture = if video_texture.is_some() || reserved_video {
            None
        } else {
            material.texture.as_deref().and_then(|key| self.camera_target_ids.get(key).copied())
        };
        let has_dynamic_texture = video_texture.is_some() || reserved_video || camera_texture.is_some();
        let loaded_texture = if has_dynamic_texture { None } else { self.loaded_texture(&material.texture)? };
        let loaded_normal = self.loaded_texture(&material.normal_texture)?;
        let loaded_outline = self.loaded_texture(&material.layer_texture)?;
        let loaded_emissive = self.loaded_texture(&material.emissive_texture)?;
        let loaded_orm = self.loaded_texture(&material.orm_texture)?;
        let s = &material.surface;
        let loaded_cc = self.loaded_texture(&s.clearcoat_texture)?;
        let loaded_ccr = self.loaded_texture(&s.clearcoat_roughness_texture)?;
        let loaded_ccn = self.loaded_texture(&s.clearcoat_normal_texture)?;
        let loaded_sheen_c = self.loaded_texture(&s.sheen_color_texture)?;
        let loaded_sheen_r = self.loaded_texture(&s.sheen_roughness_texture)?;
        let loaded_aniso = self.loaded_texture(&s.anisotropy_texture)?;
        let loaded_trans = self.loaded_texture(&s.transmission_texture)?;
        let loaded_thick = self.loaded_texture(&s.thickness_texture)?;
        let loaded_spec = self.loaded_texture(&s.specular_texture)?;
        let loaded_spec_c = self.loaded_texture(&s.specular_color_texture)?;
        let loaded_irid = self.loaded_texture(&s.iridescence_texture)?;
        let loaded_lm = self.loaded_texture(&s.lightmap_texture)?;
        let loaded_lm_dir = self.loaded_texture(&s.lightmap_dir_texture)?;
        // A texture on its way holds the draw back; one that failed shows as
        // the checker.
        let disk_texture = if has_dynamic_texture { &None } else { &material.texture };
        if [
            disk_texture,
            &material.normal_texture,
            &material.layer_texture,
            &material.emissive_texture,
            &material.orm_texture,
            &s.clearcoat_texture,
            &s.clearcoat_roughness_texture,
            &s.clearcoat_normal_texture,
            &s.sheen_color_texture,
            &s.sheen_roughness_texture,
            &s.anisotropy_texture,
            &s.transmission_texture,
            &s.thickness_texture,
            &s.specular_texture,
            &s.specular_color_texture,
            &s.iridescence_texture,
            &s.lightmap_texture,
            &s.lightmap_dir_texture,
        ]
        .into_iter()
        .flatten()
        .any(|key| self.assets.loading_key(AssetKind::Texture, key))
        {
            return Ok(None);
        }
        let mut texture = match (video_texture, reserved_video, camera_texture, &material.texture, loaded_texture) {
            (Some(idx), _, _, _, _) => TextureRef::Video(idx),
            (None, true, _, _, _) => TextureRef::White,
            (None, false, Some(idx), _, _) => TextureRef::CameraTarget(idx),
            (None, false, None, None, _) => TextureRef::White,
            (None, false, None, Some(_), Some(slot)) => TextureRef::Slot(slot),
            (None, false, None, Some(_), None) => TextureRef::Checker,
        };
        let has_base_texture =
            if video_texture.is_some() || camera_texture.is_some() || loaded_texture.is_some() { 1.0 } else { 0.0 };
        let mut normal = match field {
            Some(field) => TextureRef::Field(field),
            None => loaded_normal.map_or(TextureRef::FlatNormal, TextureRef::Slot),
        };
        let has_surface_ext = [
            loaded_cc,
            loaded_ccr,
            loaded_ccn,
            loaded_sheen_c,
            loaded_sheen_r,
            loaded_aniso,
            loaded_trans,
            loaded_thick,
            loaded_spec,
            loaded_spec_c,
            loaded_irid,
            loaded_lm,
            loaded_lm_dir,
        ]
        .iter()
        .any(Option::is_some);
        let aux_count = usize::from(loaded_outline.is_some())
            + usize::from(loaded_emissive.is_some())
            + usize::from(loaded_orm.is_some());
        let needs_atlas = !material.indexed && (has_surface_ext || aux_count >= 2);
        let orm_setting = if loaded_orm.is_some() { 1.0 } else { 0.0 };
        let emissive_setting = if loaded_emissive.is_some() { 1.0 } else { 0.0 };

        // Binding 4 holds standalone layer / emissive / ORM texture, or the GPU-packed 4x3 SurfaceMapAtlas.
        let (mut outline, mut outline_setting, tile_mask) = if needs_atlas {
            let mut key: Vec<(u32, u64, u64)> = Vec::new();
            let view_id =
                |slot: Option<usize>| -> u64 { slot.map_or(0, |s| self.texture(TextureRef::Slot(s)).view_id) };
            let push_pair = |key: &mut Vec<(u32, u64, u64)>, tile: SurfaceTile, a: Option<usize>, b: Option<usize>| {
                if a.is_some() || b.is_some() {
                    key.push((tile as u32, view_id(a), view_id(b)));
                }
            };
            push_pair(&mut key, SurfaceTile::Orm, loaded_orm, None);
            push_pair(&mut key, SurfaceTile::Emissive, loaded_emissive, None);
            push_pair(&mut key, SurfaceTile::Layer, loaded_outline, None);
            push_pair(&mut key, SurfaceTile::Clearcoat, loaded_cc, loaded_ccr);
            push_pair(&mut key, SurfaceTile::ClearcoatNormal, loaded_ccn, None);
            push_pair(&mut key, SurfaceTile::Sheen, loaded_sheen_c, loaded_sheen_r);
            push_pair(&mut key, SurfaceTile::Anisotropy, loaded_aniso, None);
            push_pair(&mut key, SurfaceTile::Transmission, loaded_trans, loaded_thick);
            push_pair(&mut key, SurfaceTile::Specular, loaded_spec_c, loaded_spec);
            push_pair(&mut key, SurfaceTile::Iridescence, loaded_irid, None);
            push_pair(&mut key, SurfaceTile::Lightmap, loaded_lm, None);
            push_pair(&mut key, SurfaceTile::LightmapDir, loaded_lm_dir, None);

            let (atlas_idx, tile_mask) = if let Some(&(idx, mask)) = self.surface_atlas_ids.get(&key) {
                (idx, mask)
            } else {
                let device = self.device.clone();
                if self.surface_atlas_packer.is_none() {
                    let cache = self.context.cache.get();
                    self.surface_atlas_packer = Some(SurfaceAtlasPacker::new(&device, cache));
                }
                let packer = self.surface_atlas_packer.as_ref().unwrap();
                let white_v = &self.white.linear;
                let flat_n = &self.flat_normal.linear;
                let view_linear = |slot: Option<usize>, fallback: &'static str| -> &wgpu::TextureView {
                    match slot {
                        Some(s) => &self.texture(TextureRef::Slot(s)).linear,
                        None if fallback == "normal" => flat_n,
                        None => white_v,
                    }
                };
                let view_srgb = |slot: Option<usize>| -> &wgpu::TextureView {
                    match slot {
                        Some(s) => &self.texture(TextureRef::Slot(s)).view,
                        None => white_v,
                    }
                };
                let mut copies: Vec<TileCopy<'_>> = Vec::new();
                if loaded_orm.is_some() {
                    copies.push(TileCopy {
                        tile: SurfaceTile::Orm,
                        src0: view_linear(loaded_orm, "w"),
                        src1: None,
                        mode: 0,
                        clamp_edges: false,
                    });
                }
                if loaded_emissive.is_some() {
                    copies.push(TileCopy {
                        tile: SurfaceTile::Emissive,
                        src0: view_srgb(loaded_emissive),
                        src1: None,
                        mode: 0,
                        clamp_edges: false,
                    });
                }
                if loaded_outline.is_some() {
                    copies.push(TileCopy {
                        tile: SurfaceTile::Layer,
                        src0: view_srgb(loaded_outline),
                        src1: None,
                        mode: 0,
                        clamp_edges: false,
                    });
                }
                if loaded_cc.is_some() || loaded_ccr.is_some() {
                    copies.push(TileCopy {
                        tile: SurfaceTile::Clearcoat,
                        src0: view_linear(loaded_cc, "w"),
                        src1: Some(view_linear(loaded_ccr, "w")),
                        mode: 1,
                        clamp_edges: false,
                    });
                }
                if loaded_ccn.is_some() {
                    copies.push(TileCopy {
                        tile: SurfaceTile::ClearcoatNormal,
                        src0: view_linear(loaded_ccn, "normal"),
                        src1: None,
                        mode: 0,
                        clamp_edges: false,
                    });
                }
                if loaded_sheen_c.is_some() || loaded_sheen_r.is_some() {
                    copies.push(TileCopy {
                        tile: SurfaceTile::Sheen,
                        src0: view_srgb(loaded_sheen_c),
                        src1: Some(view_linear(loaded_sheen_r, "w")),
                        mode: 2,
                        clamp_edges: false,
                    });
                }
                if loaded_aniso.is_some() {
                    copies.push(TileCopy {
                        tile: SurfaceTile::Anisotropy,
                        src0: view_linear(loaded_aniso, "w"),
                        src1: None,
                        mode: 0,
                        clamp_edges: false,
                    });
                }
                if loaded_trans.is_some() || loaded_thick.is_some() {
                    copies.push(TileCopy {
                        tile: SurfaceTile::Transmission,
                        src0: view_linear(loaded_trans, "w"),
                        src1: Some(view_linear(loaded_thick, "w")),
                        mode: 1,
                        clamp_edges: false,
                    });
                }
                if loaded_spec.is_some() || loaded_spec_c.is_some() {
                    copies.push(TileCopy {
                        tile: SurfaceTile::Specular,
                        src0: view_srgb(loaded_spec_c),
                        src1: Some(view_linear(loaded_spec, "w")),
                        mode: 2,
                        clamp_edges: false,
                    });
                }
                if loaded_irid.is_some() {
                    copies.push(TileCopy {
                        tile: SurfaceTile::Iridescence,
                        src0: view_linear(loaded_irid, "w"),
                        src1: None,
                        mode: 0,
                        clamp_edges: false,
                    });
                }
                if loaded_lm.is_some() {
                    copies.push(TileCopy {
                        tile: SurfaceTile::Lightmap,
                        src0: view_linear(loaded_lm, "w"),
                        src1: None,
                        mode: 0,
                        clamp_edges: true,
                    });
                }
                if loaded_lm_dir.is_some() {
                    copies.push(TileCopy {
                        tile: SurfaceTile::LightmapDir,
                        src0: view_linear(loaded_lm_dir, "w"),
                        src1: None,
                        mode: 0,
                        clamp_edges: true,
                    });
                }
                let (tex, mask) = packer.pack(&device, &self.queue, DEFAULT_TILE_SIZE, &copies, white_v);
                let id = self.next_view_id;
                self.next_view_id += 1;
                let idx = self.surface_atlases.len();
                self.surface_atlases.push(super::assets::whole(tex, id));
                self.surface_atlas_ids.insert(key, (idx, mask));
                (idx, mask)
            };
            (TextureRef::SurfaceAtlas(atlas_idx), 5.0, tile_mask)
        } else if let Some(slot) = loaded_outline {
            (TextureRef::Slot(slot), 1.0, 0)
        } else if let Some(slot) = loaded_emissive {
            (TextureRef::Slot(slot), 3.0, 0)
        } else if let Some(slot) = loaded_orm {
            (TextureRef::Slot(slot), 4.0, 0)
        } else {
            (TextureRef::Transparent, 0.0, 0)
        };
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
        if indexed {
            outline_setting = if packed_ready { 2.0 } else { 0.0 };
        }
        Ok(Some((
            [texture, normal, outline],
            normal_setting,
            outline_setting,
            orm_setting,
            emissive_setting,
            tile_mask,
            has_base_texture,
        )))
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
        let ids = textures.map(|texture| self.texture(texture).view_id);
        let key = (ids, sampler_key);
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
            crate::sort::sort_by_u64_key(materials, |&m| u64::from(m));
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
