// What a frame draws, worked out: its draws placed alone or in instanced
// batches, the solids meshed, each item's assets resolved and uniforms
// written, sorted into passes; and its decals and sprites batched.
use anyhow::{Context, Result, anyhow, ensure};
use pen_protocol::boundary::MAX_DRAW_EXTRA;
use pen_protocol::{Catalog, Draw, Matrix, RenderFrame, Solid, Sprite};
use wgpu::util::DeviceExt;

use super::frame::stencil;
use super::validate::normal_matrix;
use super::{
    ALONE, GpuAsset, INSET_SAMPLES, InstanceBatch, Item, LEFT_OUT, Placement, ReadyEntry, Scratch, Shape, SolidMesh,
    Source, State, uniform_buffer,
};
use crate::asset_store::AssetKind;
use crate::decals::DecalBatch;
use crate::matrix::{IDENTITY, narrow};
use crate::shader::Variants;
use crate::solids::{SolidBatch, mesh_solids};
use crate::sprites::SpriteBatch;

impl Scratch {
    pub(super) fn clear(&mut self) {
        self.choices.clear();
        self.parameter_batches.clear();
        self.placements.clear();
        self.batches.clear();
        self.items.clear();
        self.ready.clear();
        self.opaque.clear();
        self.translucent.clear();
        self.overlays.clear();
        self.casters.clear();
        self.materials.clear();
        self.sprites.clear();
        self.sprite_order.clear();
        self.decals.clear();
        self.decal_order.clear();
        self.insets.clear();
    }
}

impl Item {
    pub(super) fn draw<'f>(&self, frame: &'f RenderFrame) -> Option<&'f Draw> {
        match self.source {
            Source::Draw(index) => Some(&frame.draws[index]),
            Source::Inset(inset, index) => Some(&frame.insets[inset].draws[index]),
            Source::World => None,
        }
    }

    pub(super) fn transform<'f>(&self, frame: &'f RenderFrame) -> &'f Matrix {
        self.draw(frame).map_or(&IDENTITY, |draw| &draw.transform)
    }

    /// The inset it belongs to, if any.
    pub(super) fn inset(&self) -> Option<usize> {
        match self.source {
            Source::Inset(inset, _) => Some(inset),
            _ => None,
        }
    }
}

impl State {
    /// Checks that every asset the frame names is in the catalog and stays
    /// inside the asset root, each once.
    pub(super) fn check_names(&mut self, frame: &RenderFrame, catalog: &Catalog) -> Result<()> {
        self.checks += 1;
        let check = self.checks;
        let (meshes, materials) = (&mut self.checked_meshes, &mut self.checked_materials);
        meshes.resize(catalog.meshes.len(), 0);
        materials.resize(catalog.materials.len(), 0);
        let checked = |keys: &[String], checked: &mut [u64], index: u32| -> Result<()> {
            let index = index as usize;
            let key = keys.get(index).ok_or_else(|| anyhow!("draw must name a mesh and material"))?;
            if checked[index] != check {
                crate::asset_store::validate_asset_key(key)?;
                checked[index] = check;
            }
            Ok(())
        };
        for draw in frame.draws.iter().chain(frame.insets.iter().flat_map(|inset| &inset.draws)) {
            checked(&catalog.meshes, meshes, draw.mesh)?;
            checked(&catalog.materials, materials, draw.material)?;
        }
        let named = frame.solids.iter().flat_map(|solid| solid.faces.iter().map(|face| face.material));
        let named = named
            .chain(frame.decals.iter().map(|decal| decal.material))
            .chain(frame.sprites.iter().map(|sprite| sprite.material))
            .chain(frame.fields.iter().map(|patch| patch.material));
        for material in named {
            checked(&catalog.materials, materials, material)?;
        }
        Ok(())
    }

    /// Places the frame's draws, alone or in instanced batches, writes the
    /// batches' instances and lists what the frame draws: mesh draws, then
    /// one draw per batch and per solid batch in world space, then the
    /// insets' draws.
    pub(super) fn batch(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
        samples: u32,
        scratch: &mut Scratch,
    ) -> Result<()> {
        for draw in &frame.draws {
            let placement = self.place(draw, catalog, samples, scratch)?;
            scratch.placements.push(placement);
        }
        // Each batch's instances together, in the order of their draws.
        let mut instance_count = 0;
        for batch in &mut scratch.batches {
            batch.first = instance_count;
            instance_count += batch.count;
            batch.count = 0;
        }
        let instance_count = instance_count as usize;
        if instance_count > self.instance_capacity {
            self.instance_capacity = instance_count.next_power_of_two();
            ensure!(
                self.instance_capacity as u64 * 80 <= self.device.limits().max_buffer_size,
                "instance list exceeds device buffer size"
            );
            self.instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instances"),
                size: self.instance_capacity as u64 * 80,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_data = vec![0.0; self.instance_capacity * 20];
        }
        for (draw, placement) in frame.draws.iter().zip(&scratch.placements) {
            let Some(batch) = scratch.batches.get_mut(placement.batch as usize) else { continue };
            let at = (batch.first + batch.count) as usize * 20;
            batch.count += 1;
            self.instance_data[at..at + 16].copy_from_slice(&draw.transform);
            self.instance_data[at + 16..at + 20].copy_from_slice(&draw.parameters.unwrap_or(placement.parameters));
        }
        for (index, (draw, placement)) in frame.draws.iter().zip(&scratch.placements).enumerate() {
            if placement.batch == ALONE {
                scratch.items.push(Item {
                    material: draw.material,
                    source: Source::Draw(index),
                    shape: Shape::Mesh(draw.mesh),
                });
            }
        }
        for (index, batch) in scratch.batches.iter().enumerate() {
            scratch.items.push(Item { material: batch.material, source: Source::World, shape: Shape::Batch(index) });
        }
        for (index, solid) in self.solid_meshes.iter().enumerate() {
            scratch.items.push(Item { material: solid.material, source: Source::World, shape: Shape::Solid(index) });
        }
        for (inset, drawn) in frame.insets.iter().enumerate() {
            for (index, draw) in drawn.draws.iter().enumerate() {
                scratch.items.push(Item {
                    material: draw.material,
                    source: Source::Inset(inset, index),
                    shape: Shape::Mesh(draw.mesh),
                });
            }
        }
        if instance_count > 0 {
            self.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instance_data[..instance_count * 20]),
            );
        }
        let item_count = scratch.items.len();
        if item_count > self.capacity {
            self.capacity = item_count.next_power_of_two();
            ensure!(
                self.capacity as u64 * self.stride <= self.device.limits().max_buffer_size,
                "draw list exceeds device buffer size"
            );
            self.draw_uniform = uniform_buffer(&self.device, "draws", self.capacity as u64 * self.stride);
            self.groups.clear();
            self.uniform_data = vec![0.0; self.capacity * (self.stride / 4) as usize];
        }
        Ok(())
    }

    /// Where `draw` goes this frame. An opaque material whose shader can
    /// instance it joins a batch of its mesh and material, or, with instance
    /// parameters, of its mesh and render state: only the parameters may
    /// differ, which lets animation poses share one draw.
    pub(super) fn place(
        &mut self,
        draw: &Draw,
        catalog: &Catalog,
        samples: u32,
        scratch: &mut Scratch,
    ) -> Result<Placement> {
        let alone = Placement { batch: ALONE, parameters: [0.0; 4] };
        // Its own uniform data: drawn alone.
        if draw.extra.as_ref().is_some_and(|extra| !extra.is_empty()) {
            return Ok(alone);
        }
        let choice = match scratch.choices.get(&(draw.mesh, draw.material)) {
            Some(choice) => *choice,
            None => {
                let material = self.material(catalog, draw.material)?;
                let shader = match &material {
                    Some(material) => self.shader(&material.material)?,
                    None => None,
                };
                let mesh = self.resolve(AssetKind::Mesh, catalog.mesh(draw.mesh))?;
                let mesh_loaded = matches!(self.assets.current(mesh), Some(GpuAsset::Mesh(_)));
                let instancing = shader.is_some_and(
                    |slot| matches!(self.assets.current(slot), Some(GpuAsset::Shader(shader)) if shader.instancing),
                );
                let choice = match (material, shader) {
                    (Some(material), Some(shader)) if material.material.opaque() && instancing && mesh_loaded => {
                        let key = material.material.instance_parameters.map(|_| (draw.mesh, material.state));
                        let batch = match key.and_then(|key| scratch.parameter_batches.get(&key).copied()) {
                            Some(batch) => batch,
                            None => {
                                // Left out until its instanced pipeline is there.
                                if !self.ensure_variants(shader, samples, Variants::INSTANCED) {
                                    return Ok(Placement { batch: LEFT_OUT, ..alone });
                                }
                                let pipeline = self.pipelines(shader, samples).instanced.clone().unwrap();
                                scratch.batches.push(InstanceBatch {
                                    mesh,
                                    material: draw.material,
                                    pipeline,
                                    first: 0,
                                    count: 0,
                                });
                                if let Some(key) = key {
                                    scratch.parameter_batches.insert(key, scratch.batches.len() - 1);
                                }
                                scratch.batches.len() - 1
                            }
                        };
                        let parameters =
                            material.material.instance_parameters.map_or([0.0; 4], |values| values.map(narrow));
                        Some((batch, parameters))
                    }
                    _ => None,
                };
                scratch.choices.insert((draw.mesh, draw.material), choice);
                choice
            }
        };
        Ok(match choice {
            Some((batch, parameters)) => {
                scratch.batches[batch].count += 1;
                Placement { batch: u32::try_from(batch)?, parameters }
            }
            None => alone,
        })
    }

    /// Resolves each item's assets and writes its uniforms once.
    pub(super) fn prepare_items(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
        samples: u32,
        scratch: &mut Scratch,
    ) -> Result<()> {
        let item_count = scratch.items.len();
        let floats = (self.stride / 4) as usize;
        for index in 0..item_count {
            let item = scratch.items[index];
            let Some(entry) = self.ready_entry(frame, catalog, scratch, index, item, samples)? else { continue };
            scratch.ready.push(entry);
        }
        if item_count > 0 {
            self.queue.write_buffer(
                &self.draw_uniform,
                0,
                bytemuck::cast_slice(&self.uniform_data[..item_count * floats]),
            );
        }
        Ok(())
    }

    /// Item `index` ready to draw, its uniforms written; None while
    /// something it draws with is on its way.
    pub(super) fn ready_entry(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
        scratch: &Scratch,
        index: usize,
        item: Item,
        samples: u32,
    ) -> Result<Option<ReadyEntry>> {
        let Some(material) = self.material(catalog, item.material)? else { return Ok(None) };
        let Some(shader) = self.shader(&material.material)? else { return Ok(None) };
        let field = self.field_ids.get(catalog.material(item.material)).copied();
        let Some((textures, normal_setting, outline_setting)) = self.dress(&material.material, field)? else {
            return Ok(None);
        };
        let group = self.draw_group(textures);
        let floats = (self.stride / 4) as usize;
        let data = &mut self.uniform_data[index * floats..(index + 1) * floats];
        let m = &material.material;
        let draw = item.draw(frame);
        let transform = item.transform(frame);
        data[..16].copy_from_slice(transform);
        data[16..32].copy_from_slice(&normal_matrix(transform)?);
        data[32..36].copy_from_slice(&m.color.map(narrow));
        data[36..40].copy_from_slice(&[
            if m.lit { 1.0 } else { 0.0 },
            narrow(m.alpha_cutoff),
            normal_setting,
            outline_setting,
        ]);
        let parameters = draw
            .and_then(|draw| draw.parameters)
            .or(m.instance_parameters.map(|values| values.map(narrow)))
            .unwrap_or([0.0; 4]);
        data[40..44].copy_from_slice(&parameters);
        // Unused extra vectors read as zero.
        let extra = &mut data[44..44 + MAX_DRAW_EXTRA * 4];
        extra.fill(0.0);
        if let Some(values) = draw.and_then(|draw| draw.extra.as_deref()) {
            let count = values.len().min(MAX_DRAW_EXTRA * 4);
            extra[..count].copy_from_slice(&values[..count]);
        }
        // Insets retain material blending, with their own sample count.
        let (pipeline, visible) = match (item.inset(), item.shape) {
            (Some(_), _) => {
                let variant = if m.opaque() { Variants::OPAQUE } else { Variants::TRANSLUCENT };
                if !self.ensure_variants(shader, INSET_SAMPLES, variant) {
                    return Ok(None);
                }
                let made = self.pipelines(shader, INSET_SAMPLES);
                let pipeline = if m.opaque() { made.opaque.clone() } else { made.translucent.clone() };
                (pipeline, None)
            }
            (None, Shape::Batch(batch)) => (Some(scratch.batches[batch].pipeline.clone()), None),
            (None, _) => {
                if !self.ensure_variants(shader, samples, m.variants().without(Variants::INSTANCED)) {
                    return Ok(None);
                }
                let made = self.pipelines(shader, samples);
                if m.overlay {
                    (made.occluded.clone(), made.visible.clone())
                } else if m.opaque() {
                    (made.opaque.clone(), None)
                } else {
                    (made.translucent.clone(), None)
                }
            }
        };
        let mesh = match item.shape {
            Shape::Mesh(mesh) => Some(self.resolve(AssetKind::Mesh, catalog.mesh(mesh))?),
            _ => None,
        };
        let pipeline = pipeline.expect("an ensured variant is compiled");
        let offset =
            u32::try_from(index as u64 * self.stride).context("draw uniforms lie past a dynamic offset's reach")?;
        Ok(Some(ReadyEntry { item: index, offset, group, material, shader, mesh, pipeline, visible }))
    }

    /// Sorts what is ready into the scene's passes: opaque surfaces by
    /// stencil mark and material, so that draws in a row share their
    /// pipeline (they may be drawn in any order, and catalog indices keep it
    /// the same from run to run), translucent ones back to front by their
    /// origin's view depth, overlays, and the shadows' casters.
    pub(super) fn sort_passes(&self, frame: &RenderFrame, scratch: &mut Scratch, shadowed: bool) {
        let items = &scratch.items;
        let ready = &scratch.ready;
        for (index, entry) in ready.iter().enumerate() {
            if items[entry.item].inset().is_some() {
                continue;
            }
            let m = &entry.material.material;
            if m.overlay {
                scratch.overlays.push(index);
            } else if m.opaque() {
                scratch.opaque.push(index);
            } else {
                scratch.translucent.push(index);
            }
        }
        scratch.opaque.sort_by_key(|&index| {
            let entry = &ready[index];
            let item = &items[entry.item];
            (stencil(&entry.material.material), item.material, matches!(item.shape, Shape::Batch(_)))
        });
        let view = &frame.view;
        let depth_of = |transform: &Matrix| {
            f64::from(view[2]) * f64::from(transform[12])
                + f64::from(view[6]) * f64::from(transform[13])
                + f64::from(view[10]) * f64::from(transform[14])
                + f64::from(view[14])
        };
        scratch.translucent.sort_by(|&a, &b| {
            let (a, b) = (items[ready[a].item].transform(frame), items[ready[b].item].transform(frame));
            depth_of(a).total_cmp(&depth_of(b))
        });
        // Lit, opaque surfaces cast shadows, unless their material says not.
        if shadowed {
            for &index in &scratch.opaque {
                let entry = &ready[index];
                let instanced = matches!(items[entry.item].shape, Shape::Batch(_));
                if entry.material.material.casts_shadow()
                    && matches!(self.assets.current(entry.shader), Some(GpuAsset::Shader(shader))
                        if shader.shadow[instanced as usize].is_some())
                {
                    scratch.casters.push(index);
                }
            }
        }
    }

    /// The frame's decals whose material and texture are ready, in batches by
    /// material and texture.
    pub(super) fn decal_batches(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
        scratch: &mut Scratch,
    ) -> Result<()> {
        // Each decal's batch, then the decals in batch order.
        let placed = &mut scratch.decal_order;
        for decal in &frame.decals {
            placed.push(LEFT_OUT);
            let material_slot = self.resolve(AssetKind::Material, catalog.material(decal.material))?;
            let Some(GpuAsset::Material(material)) = self.assets.current(material_slot) else { continue };
            let material = material.clone();
            let Some(key) = &material.material.texture else { continue };
            let image = self.resolve(AssetKind::Texture, key)?;
            let Some(GpuAsset::Texture(image)) = self.assets.current(image) else { continue };
            let paint = !material.material.lit;
            let found = scratch.decals.iter().position(|other| {
                other.texture_id == image.view_id && other.paint == paint && other.material == material_slot
            });
            let batch = found.unwrap_or_else(|| {
                scratch.decals.push(DecalBatch {
                    texture: image.view.clone(),
                    texture_id: image.view_id,
                    material: material_slot,
                    color: material.material.color,
                    paint,
                    first: 0,
                    count: 0,
                });
                scratch.decals.len() - 1
            });
            scratch.decals[batch].count += 1;
            *placed.last_mut().unwrap() = u32::try_from(batch)?;
        }
        let mut first = 0;
        for batch in &mut scratch.decals {
            batch.first = first;
            first += batch.count;
            batch.count = 0;
        }
        let batch_of = std::mem::take(placed);
        placed.resize(first as usize, 0);
        for (index, &batch) in batch_of.iter().enumerate() {
            let Some(batch) = scratch.decals.get_mut(batch as usize) else { continue };
            placed[(batch.first + batch.count) as usize] = u32::try_from(index)?;
            batch.count += 1;
        }
        Ok(())
    }

    /// `sprites` in batches of one texture and shader, in order, into
    /// `batches` (by their indices in `order`); a sprite whose material,
    /// shader or texture is still loading is left out.
    pub(super) fn sprite_batches(
        &mut self,
        sprites: &[Sprite],
        catalog: &Catalog,
        batches: &mut Vec<SpriteBatch>,
        order: &mut Vec<u32>,
    ) -> Result<()> {
        for (index, sprite) in sprites.iter().enumerate() {
            let Some(material) = self.material(catalog, sprite.material)? else { continue };
            let shader_slot = self.resolve(AssetKind::SpriteShader, &material.material.shader)?;
            let texture_slot = match &material.material.texture {
                None => None,
                Some(key) => Some(self.resolve(AssetKind::Texture, key)?),
            };
            let Some(GpuAsset::SpriteShader(pipeline)) = self.assets.current(shader_slot) else { continue };
            let texture = match texture_slot.map(|slot| self.assets.current(slot)) {
                None => &self.white,
                Some(Some(GpuAsset::Texture(texture))) => texture,
                Some(_) => continue,
            };
            let key = (texture.view_id, shader_slot);
            let first = u32::try_from(order.len())?;
            order.push(u32::try_from(index)?);
            match batches.last_mut() {
                Some(last) if last.key == key => last.count += 1,
                _ => batches.push(SpriteBatch {
                    pipeline: pipeline.clone(),
                    texture: texture.view.clone(),
                    key,
                    first,
                    count: 1,
                }),
            }
        }
        Ok(())
    }

    /// Meshes the frame's solids, again only those that changed, and
    /// uploads them by material when any did.
    pub(super) fn update_solids(&mut self, solids: &[Solid]) {
        let unchanged = self.solids.len() == solids.len()
            && self.solids.iter().zip(solids).all(|((known, _), solid)| known == solid);
        if unchanged {
            return;
        }
        self.solids.truncate(solids.len());
        for (index, solid) in solids.iter().enumerate() {
            match self.solids.get_mut(index) {
                Some((known, _)) if known == solid => {}
                Some(known) => *known = (solid.clone(), mesh_solids(std::slice::from_ref(solid))),
                None => self.solids.push((solid.clone(), mesh_solids(std::slice::from_ref(solid)))),
            }
        }
        // Each material's triangles, in the order materials first appear.
        let mut merged: Vec<SolidBatch> = Vec::new();
        for batch in self.solids.iter().flat_map(|(_, batches)| batches) {
            match merged.iter_mut().find(|merged| merged.material == batch.material) {
                Some(merged) => merged.vertices.extend_from_slice(&batch.vertices),
                None => merged.push(SolidBatch { material: batch.material, vertices: batch.vertices.clone() }),
            }
        }
        merged.retain(|batch| !batch.vertices.is_empty());
        let mut old = std::mem::take(&mut self.solid_meshes).into_iter();
        for batch in merged {
            let bytes: &[u8] = bytemuck::cast_slice(&batch.vertices);
            let count = u32::try_from(batch.vertices.len() / crate::asset_store::MESH_VERTEX_FLOATS)
                .expect("MAX_SOLIDS and MAX_SOLID_FACES keep solids' vertices below 2^32");
            // A buffer from before is written again if the batch fits it.
            let buffer = match old.next() {
                Some(mesh) if mesh.buffer.size() >= bytes.len() as u64 => {
                    self.queue.write_buffer(&mesh.buffer, 0, bytes);
                    mesh.buffer
                }
                _ => self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("solids"),
                    contents: bytes,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                }),
            };
            self.solid_meshes.push(SolidMesh { material: batch.material, buffer, count });
        }
    }
}
