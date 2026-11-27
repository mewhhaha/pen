// What a frame draws, worked out: its draws placed alone or in instanced
// batches, the solids meshed, each item's assets resolved and uniforms
// written, sorted into passes; and its decals and sprites batched.
use anyhow::{Context, Result, anyhow, ensure};
use pen_protocol::boundary::MAX_DRAW_EXTRA;
use pen_protocol::{Catalog, Draw, Matrix, RenderFrame, Solid, Sprite};
use wgpu::util::DeviceExt;

use super::cascades::Cascade;
use super::frame::stencil;
use super::skinning::Posed;
use super::validate::normal_matrix;
use super::{
    ALONE, Batching, Choice, GpuAsset, INSET_SAMPLES, InstanceBatch, Item, LEFT_OUT, MAPS_AT, Placement, ReadyEntry,
    SURFACE_AT, Scratch, Shape, SolidMesh, Source, State, uniform_buffer,
};
use crate::RenderStats;
use crate::asset_store::{AssetKind, MAPS_FLOATS, MAX_LODS, SURFACE_FLOATS};
use crate::culling::{ALL_VIEWS, CAMERA, Detail, Frustum, Sphere, ViewMask, Views, stretch};
use crate::decals::DecalBatch;
use crate::local_shadows::Selection;
use crate::matrix::{IDENTITY, narrow};
use crate::shader::Variants;
use crate::solids::{SolidBatch, mesh_solids};
use crate::sprites::SpriteBatch;
use pen_protocol::boundary::MAX_CASCADES;

/// The views `frame` draws: the camera's, then its sun shadow maps'
/// (`cascades`), then the faces of the local lights' maps (`local`), then the
/// view of a reflection probe's capture (`capture`, a view-projection), and
/// the bit of that one (0 without). Every draw may be in the capture's view,
/// casters or not.
fn views(
    frame: &RenderFrame,
    (maps, count): &([Cascade; MAX_CASCADES], usize),
    local: &Selection,
    capture: Option<&Matrix>,
) -> (Views, ViewMask) {
    let mut views = Views::new(Frustum::of(&frame.projection, &frame.view));
    for map in &maps[..*count] {
        views.push(Frustum::of(&map.projection, &frame.shadow.view));
    }
    for face in &local.faces {
        views.push(Frustum::new(&face.matrix));
    }
    let bit = capture.map_or(0, |matrix| {
        let bit = 1 << views.count();
        views.push(Frustum::new(matrix));
        bit
    });
    (views, bit)
}

/// Which `views` see `draw` (a bit each; a non-caster only the camera's,
/// and the capture's `capture`) and at which level of detail the camera has
/// it drawn, if its shader declares its bounds; otherwise every view, whole.
/// Counts into `stats` what it leaves out.
fn seen(
    choice: &Choice,
    draw: &Draw,
    views: &Views,
    detail: &Detail,
    capture: ViewMask,
    stats: &mut RenderStats,
) -> (ViewMask, u8) {
    let every = if choice.casts { views.all() } else { CAMERA | capture };
    let Some(bounds) = choice.bounds else { return (every, 0) };
    let placed = bounds.placed(&draw.transform, 0.0);
    let seen = views.mask_in(&placed, every);
    stats.culled += u32::from(seen & CAMERA == 0);
    stats.shadow_culled += (every & !seen & !CAMERA & !capture).count_ones();
    let level = if choice.levels > 1 && detail.pixels > 0.0 {
        detail.level(&placed, stretch(&draw.transform), &choice.errors[..usize::from(choice.levels)])
    } else {
        0
    };
    (seen, u8::try_from(level).expect("at most MAX_LODS"))
}

/// A buffer of `capacity` instances, which batches draw from and the GPU
/// culls from.
pub(super) fn instance_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("instances"),
        size: capacity as u64 * 80,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl Scratch {
    pub(super) fn clear(&mut self) {
        self.chosen.clear();
        self.choices.clear();
        self.parameter_batches.clear();
        self.placements.clear();
        self.batches.clear();
        self.gpu_order.clear();
        self.items.clear();
        self.ready.clear();
        self.opaque.clear();
        self.translucent.clear();
        self.overlays.clear();
        self.casters.clear();
        self.contact_casters.clear();
        self.capture_list.clear();
        self.capture_bit = 0;
        self.materials.clear();
        self.sprites.clear();
        self.sprite_order.clear();
        self.decals.clear();
        self.decal_order.clear();
        self.insets.clear();
        self.emitters.clear();
        self.ribbons.clear();
        self.motion_for_draw.clear();
    }

    /// Maps each scene draw index with a differing previous transform to its index in `frame.motions`.
    pub(super) fn group_motions(&mut self, frame: &RenderFrame) {
        self.motion_for_draw.clear();
        if frame.motions.is_empty() {
            return;
        }
        self.motion_for_draw.resize(frame.draws.len(), u32::MAX);
        for (index, motion) in frame.motions.iter().enumerate() {
            let draw = motion.draw as usize;
            if let Some(current) = frame.draws.get(draw)
                && current.transform != motion.previous
            {
                self.motion_for_draw[draw] = u32::try_from(index).unwrap_or(u32::MAX);
            }
        }
    }

    pub(super) fn has_motion(&self, draw: usize) -> bool {
        self.motion_for_draw.get(draw).is_some_and(|&slot| slot != u32::MAX)
    }

    pub(super) fn motion_previous<'a>(&self, frame: &'a RenderFrame, draw: usize) -> Option<&'a Matrix> {
        let slot = *self.motion_for_draw.get(draw)?;
        (slot != u32::MAX).then(|| &frame.motions[slot as usize].previous)
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
            .chain(frame.pbr_decals.iter().map(|decal| decal.material))
            .chain(frame.emitters.iter().map(|emitter| emitter.material))
            .chain(frame.ribbons.iter().map(|ribbon| ribbon.material))
            .chain(frame.sprites.iter().map(|sprite| sprite.material))
            .chain(frame.fields.iter().map(|patch| patch.material))
            .chain(frame.camera_targets.iter().filter(|cam| !cam.split_screen).map(|cam| cam.material))
            .chain(frame.video_streams.iter().map(|video| video.material))
            .chain(
                frame
                    .area_lights
                    .iter()
                    .filter(|light| light.cookie_material != 0)
                    .map(|light| light.cookie_material - 1),
            )
            .chain(
                [frame.grade.lut_a, frame.grade.lut_b].into_iter().filter(|&table| table != 0).map(|table| table - 1),
            );
        for material in named {
            checked(&catalog.materials, materials, material)?;
        }
        Ok(())
    }

    /// Places the frame's draws, alone or in instanced batches, culls them
    /// (a batch of many instances on the GPU, the rest here: each draw into
    /// the views that see it, at the level of detail the camera picks for
    /// it), writes the batches' instances and lists what the frame draws:
    /// mesh draws, then one draw per batch and per solid batch in world
    /// space, then the insets' draws. The scene is `height` pixels high.
    /// Counts what culling here left out.
    #[allow(clippy::too_many_arguments, reason = "frame batching inputs")]
    pub(super) fn batch(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
        samples: u32,
        height: f32,
        scratch: &mut Scratch,
        stats: &mut RenderStats,
        outline_scenes: &super::wired_passes::OutlineScenes,
    ) -> Result<()> {
        let capture = self.capture.as_ref().map(|step| {
            let at = &step.probe.position;
            crate::environment::box_view([at.x, at.y, at.z], step.probe.far)
        });
        (scratch.views, scratch.capture_bit) = views(frame, &self.cascades, &self.local, capture.as_ref());
        scratch.detail = Detail::new(&frame.projection, &frame.view, height);
        self.culling.begin(scratch.views, scratch.detail);
        scratch.group_poses(frame);
        scratch.group_morphs(frame);
        scratch.group_motions(frame);
        let mut last = ((u32::MAX, u32::MAX), u32::MAX);
        for (index, draw) in frame.draws.iter().enumerate() {
            // A posed, morphed, or explicitly moving draw has per-draw matrices/deformations: drawn alone.
            let alone = scratch.posed(index)
                || scratch.morphed(index)
                || scratch.has_motion(index)
                || super::wired_passes::outlined(outline_scenes, index);
            let placement = self.place_cached(draw, catalog, samples, scratch, alone, &mut last)?;
            scratch.placements.push(placement);
        }
        // Batches of many instances cull on the GPU: their instances go first
        // in the buffer, whole, once each, largest first so big static batches
        // stay at stable offsets when smaller dynamic batches change size.
        let mut instance_count = 0;
        scratch.gpu_order.clear();
        for i in 0..scratch.batches.len() {
            let batch = &scratch.batches[i];
            if batch.bounds.is_some()
                && matches!(self.assets.current(batch.mesh), Some(GpuAsset::Mesh(_)))
                && self.culling.takes(batch.count)
            {
                scratch.gpu_order.push(i);
            }
        }
        crate::sort::sort_by_u64_key(&mut scratch.gpu_order, |&i| u64::from(!scratch.batches[i].count));
        for i in 0..scratch.gpu_order.len() {
            let index = scratch.gpu_order[i];
            let batch = &mut scratch.batches[index];
            let (Some(bounds), Some(GpuAsset::Mesh(mesh))) = (batch.bounds, self.assets.current(batch.mesh)) else {
                continue;
            };
            let views = if batch.casts { scratch.views.all() } else { CAMERA | scratch.capture_bit };
            batch.gpu = self.culling.add(&bounds, &mesh.lods, instance_count, batch.count, views);
            if batch.gpu.is_some() {
                batch.first = instance_count;
                instance_count += batch.count;
            }
        }
        // Reserve a coarse aligned region for GPU-culled batches so fluctuating dynamic
        // batch counts (e.g. flying bricks) do not shift all subsequent CPU-culled batches.
        if instance_count > self.gpu_instance_base {
            self.gpu_instance_base = instance_count.next_multiple_of(512);
        } else if instance_count == 0 {
            self.gpu_instance_base = 0;
        } else if self.gpu_instance_base > instance_count + 2048 {
            self.gpu_instance_base = instance_count.next_multiple_of(512);
        }
        instance_count = self.gpu_instance_base;
        // The rest cull here, but posed draws: their joints move their
        // vertices past their mesh's bounds, so every view they may be in
        // draws them, whole.
        for (index, draw) in frame.draws.iter().enumerate() {
            let deformed = scratch.posed(index) || scratch.morphed(index);
            let placement = &mut scratch.placements[index];
            if placement.batch == LEFT_OUT {
                continue;
            }
            let batch = scratch.batches.get_mut(placement.batch as usize);
            if batch.as_ref().is_some_and(|batch| batch.gpu.is_some()) {
                continue;
            }
            let choice = &scratch.choices[placement.choice as usize];
            (placement.views, placement.lod) = if deformed {
                (if choice.casts { scratch.views.all() } else { CAMERA | scratch.capture_bit }, 0)
            } else {
                seen(choice, draw, &scratch.views, &scratch.detail, scratch.capture_bit, stats)
            };
            if let Some(batch) = batch {
                let mut mask = placement.views;
                let lod = usize::from(placement.lod);
                while mask != 0 {
                    let view = mask.trailing_zeros() as usize;
                    mask &= mask - 1;
                    batch.ranges[view][lod].1 += 1;
                }
            }
        }
        // Place singleton CPU-culled batches (mostly static buildings across views)
        // in a contiguous prefix ahead of dynamic multi-instance batches so fluctuating
        // actor counts do not shift static instances in the upload diff.
        for single in [true, false] {
            for view in 0..scratch.views.count() {
                for batch in
                    scratch.batches.iter_mut().filter(|batch| batch.gpu.is_none() && (batch.count == 1) == single)
                {
                    for range in &mut batch.ranges[view] {
                        let count = range.1;
                        *range = (instance_count, 0);
                        instance_count += count;
                    }
                }
            }
        }
        let instance_count = instance_count as usize;
        if instance_count > self.instance_capacity {
            self.instance_capacity = instance_count.next_power_of_two();
            ensure!(
                self.instance_capacity as u64 * 80 <= self.device.limits().max_buffer_size,
                "instance list exceeds device buffer size"
            );
            self.instance_buffer = instance_buffer(&self.device, self.instance_capacity);
            self.instance_data = vec![0.0; self.instance_capacity * 20];
            self.instance_uploads.invalidate();
        }
        for (draw, placement) in frame.draws.iter().zip(&scratch.placements) {
            let Some(batch) = scratch.batches.get_mut(placement.batch as usize) else { continue };
            let parameters = draw.parameters.unwrap_or(placement.parameters);
            let mut write = |at: u32| {
                let at = at as usize * 20;
                self.instance_data[at..at + 16].copy_from_slice(&draw.transform);
                self.instance_data[at + 16..at + 20].copy_from_slice(&parameters);
            };
            if batch.gpu.is_some() {
                write(batch.first + batch.written);
                batch.written += 1;
                continue;
            }
            let mut mask = placement.views;
            let lod = usize::from(placement.lod);
            while mask != 0 {
                let view = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let (first, count) = &mut batch.ranges[view][lod];
                write(*first + *count);
                *count += 1;
            }
        }
        for (index, (draw, placement)) in frame.draws.iter().zip(&scratch.placements).enumerate() {
            if placement.batch == ALONE {
                scratch.items.push(Item {
                    material: draw.material,
                    source: Source::Draw(index),
                    shape: Shape::Mesh(draw.mesh),
                    views: placement.views,
                    lod: placement.lod,
                });
            }
        }
        for (index, batch) in scratch.batches.iter().enumerate() {
            // One the GPU culls may be drawn in every view it is allowed.
            let views = match batch.gpu {
                Some(_) if batch.casts => scratch.views.all(),
                Some(_) => CAMERA | scratch.capture_bit,
                None => (0..scratch.views.count())
                    .filter(|&view| batch.ranges[view].iter().any(|range| range.1 > 0))
                    .fold(0, |views, view| views | 1 << view),
            };
            scratch.items.push(Item {
                material: batch.material,
                source: Source::World,
                shape: Shape::Batch(index),
                views,
                lod: 0,
            });
        }
        for (index, solid) in self.solid_meshes.iter().enumerate() {
            scratch.items.push(Item {
                material: solid.material,
                source: Source::World,
                shape: Shape::Solid(index),
                views: ALL_VIEWS,
                lod: 0,
            });
        }
        for (inset, drawn) in frame.insets.iter().enumerate() {
            for (index, draw) in drawn.draws.iter().enumerate() {
                scratch.items.push(Item {
                    material: draw.material,
                    source: Source::Inset(inset, index),
                    shape: Shape::Mesh(draw.mesh),
                    views: ALL_VIEWS,
                    lod: 0,
                });
            }
        }
        if instance_count > 0 {
            let current = &self.instance_data[..instance_count * 20];
            for range in self.instance_uploads.changed(current) {
                self.queue.write_buffer(
                    &self.instance_buffer,
                    range.start as u64 * 4,
                    bytemuck::cast_slice(&current[range.clone()]),
                );
                stats.instance_upload_bytes += range.len() as u64 * 4;
                stats.instance_upload_calls += 1;
            }
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

    /// Batches only `frame.insets` draws (used by the split-screen compositor path so insets
    /// render and interleave with sprites identically to single-camera frames).
    pub(super) fn batch_insets_only(&mut self, frame: &RenderFrame, scratch: &mut Scratch) -> Result<()> {
        scratch.group_poses(frame);
        scratch.group_morphs(frame);
        for (inset, drawn) in frame.insets.iter().enumerate() {
            for (index, draw) in drawn.draws.iter().enumerate() {
                scratch.items.push(Item {
                    material: draw.material,
                    source: Source::Inset(inset, index),
                    shape: Shape::Mesh(draw.mesh),
                    views: ALL_VIEWS,
                    lod: 0,
                });
            }
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

    /// Where `draw` goes this frame: alone or into a batch, or left out. An
    /// opaque material whose shader can instance it joins a batch of its
    /// mesh and material, or, with instance parameters, of its mesh and
    /// render state: only the parameters may differ, which lets animation
    /// poses share one draw. Which views see it is worked out after.
    pub(super) fn place_cached(
        &mut self,
        draw: &Draw,
        catalog: &Catalog,
        samples: u32,
        scratch: &mut Scratch,
        alone: bool,
        last: &mut ((u32, u32), u32),
    ) -> Result<Placement> {
        let key = (draw.mesh, draw.material);
        let choice = if key == last.0 {
            last.1
        } else {
            let c = match scratch.chosen.get(&key) {
                Some(&choice) => choice,
                None => {
                    let choice = self.choose(draw, catalog, samples, scratch)?;
                    scratch.choices.push(choice);
                    let index = u32::try_from(scratch.choices.len() - 1)?;
                    scratch.chosen.insert(key, index);
                    index
                }
            };
            *last = (key, c);
            c
        };
        let placement = Placement { batch: ALONE, choice, parameters: [0.0; 4], views: 0, lod: 0 };
        // A draw with its own uniform data draws alone, as one its caller
        // says must.
        if alone || draw.extra.as_ref().is_some_and(|extra| !extra.is_empty()) {
            return Ok(placement);
        }
        Ok(match scratch.choices[choice as usize].batch {
            Batching::Into(batch, parameters) => {
                scratch.batches[batch].count += 1;
                Placement { batch: u32::try_from(batch)?, parameters, ..placement }
            }
            Batching::Waiting => Placement { batch: LEFT_OUT, ..placement },
            Batching::Alone => placement,
        })
    }

    /// How a mesh and material's draws go this frame (see `place`).
    fn choose(&mut self, draw: &Draw, catalog: &Catalog, samples: u32, scratch: &mut Scratch) -> Result<Choice> {
        let material = self.material(catalog, draw.material)?;
        let shader = match &material {
            Some(material) => self.shader(&material.material)?,
            None => None,
        };
        let mesh = self.resolve(AssetKind::Mesh, catalog.mesh(draw.mesh))?;
        let mut errors = [0.0; MAX_LODS];
        let (sphere, levels) = match self.assets.current(mesh) {
            Some(GpuAsset::Mesh(mesh)) => {
                for (error, lod) in errors.iter_mut().zip(&mesh.lods) {
                    *error = lod.error;
                }
                (Some(mesh.bounds), mesh.lods.len().clamp(1, MAX_LODS))
            }
            _ => (None, 1),
        };
        let (instancing, margin) = match shader.and_then(|slot| self.assets.current(slot)) {
            Some(GpuAsset::Shader(shader)) => (shader.instancing, shader.margin),
            _ => (false, None),
        };
        // The mesh's sphere grown by its shader's margin; without one, it is
        // never culled, nor drawn simpler.
        let bounds = sphere.zip(margin).map(|(sphere, margin)| Sphere { radius: sphere.radius + margin, ..sphere });
        let casts = material.as_ref().is_some_and(|material| material.material.casts_shadow());
        // Mesh-owned records must never enter a frame's transform batches.
        let persistent = matches!(self.assets.current(mesh), Some(GpuAsset::Mesh(mesh)) if mesh.instances.is_some());
        let batch = match (material, shader) {
            (Some(material), Some(shader))
                if material.material.opaque() && instancing && sphere.is_some() && !persistent =>
            {
                let parameters = material.material.instance_parameters.map_or([0.0; 4], |values| values.map(narrow));
                let key = material.material.instance_parameters.map(|_| (draw.mesh, material.state));
                match key.and_then(|key| scratch.parameter_batches.get(&key).copied()) {
                    Some(batch) => Batching::Into(batch, parameters),
                    // Left out until its instanced pipeline is there.
                    None if !self.ensure_variants(shader, samples, Variants::INSTANCED) => Batching::Waiting,
                    None => {
                        let pipeline = self.pipelines(shader, samples).instanced.clone().unwrap();
                        scratch.batches.push(InstanceBatch {
                            mesh,
                            material: draw.material,
                            pipeline,
                            bounds,
                            casts,
                            count: 0,
                            gpu: None,
                            first: 0,
                            written: 0,
                            ranges: Default::default(),
                        });
                        if let Some(key) = key {
                            scratch.parameter_batches.insert(key, scratch.batches.len() - 1);
                        }
                        Batching::Into(scratch.batches.len() - 1, parameters)
                    }
                }
            }
            _ => Batching::Alone,
        };
        Ok(Choice { batch, bounds, casts, errors, levels: u8::try_from(levels)? })
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
        self.skinning.begin(frame.exposure.cut);
        self.morphing.begin();
        // Fast 1-entry caches avoid repeated material/dress/bind-group and mesh lookups
        // across contiguous runs of identical solo draws (e.g. wakes, earth patches, buildings).
        struct MatCache {
            index: u32,
            material: std::sync::Arc<super::MaterialAsset>,
            shader: usize,
            dressed: super::materials::Dressed,
            group: Option<wgpu::BindGroup>,
            surface_vecs: [f32; SURFACE_FLOATS],
            uv_maps: [f32; 8],
            color: [f32; 4],
            settings: [f32; 4],
            default_params: [f32; 4],
            maps_tail: [f32; 8],
        }
        let mut mat_cache: Option<MatCache> = None;
        let mut mesh_cache: Option<(u32, usize, bool)> = None;
        for index in 0..item_count {
            let item = scratch.items[index];
            if mat_cache.as_ref().is_none_or(|c| c.index != item.material) {
                let Some(material) = self.material(catalog, item.material)? else { continue };
                let Some(shader) = self.shader(&material.material)? else { continue };
                let field = if self.field_ids.is_empty() {
                    None
                } else {
                    self.field_ids.get(catalog.material(item.material)).copied()
                };
                let Some(dressed) = self.dress(&material.material, field)? else { continue };
                let m = &material.material;
                let (_, normal_setting, outline_setting, orm_setting, emissive_setting, tile_mask, has_base_texture) =
                    dressed;
                mat_cache = Some(MatCache {
                    index: item.material,
                    surface_vecs: m.surface.vectors(),
                    uv_maps: m.uv_transform.vectors(orm_setting, emissive_setting),
                    color: m.color.map(narrow),
                    settings: [if m.lit { 1.0 } else { 0.0 }, narrow(m.alpha_cutoff), normal_setting, outline_setting],
                    default_params: m.instance_parameters.map(|values| values.map(narrow)).unwrap_or([0.0; 4]),
                    maps_tail: [
                        tile_mask as f32,
                        has_base_texture,
                        0.0,
                        0.0,
                        m.surface.iridescence,
                        m.surface.iridescence_ior,
                        m.surface.iridescence_thickness_min,
                        m.surface.iridescence_thickness_max,
                    ],
                    material,
                    shader,
                    dressed,
                    group: None,
                });
            }
            let cached = mat_cache.as_mut().unwrap();
            let (mesh, persistent) = match item.shape {
                Shape::Mesh(m) => {
                    if let Some((cm, slot, p)) = mesh_cache
                        && cm == m
                    {
                        (Some(slot), p)
                    } else {
                        let slot = self.resolve(AssetKind::Mesh, catalog.mesh(m))?;
                        let p = matches!(
                            self.assets.current(slot),
                            Some(GpuAsset::Mesh(mesh)) if mesh.instances.is_some()
                        );
                        mesh_cache = Some((m, slot, p));
                        (Some(slot), p)
                    }
                }
                _ => (None, false),
            };
            if persistent {
                ensure!(cached.material.material.opaque(), "persistent mesh instances require an opaque material");
                ensure!(
                    matches!(self.assets.current(cached.shader), Some(GpuAsset::Shader(shader)) if shader.persistent),
                    "persistent mesh instances require a vertex_persistent shader entry point"
                );
                ensure!(
                    !scratch.target_of_source(item.source).is_some_and(|t| scratch.posed(t) || scratch.morphed(t)),
                    "persistent mesh instances cannot be posed or morphed"
                );
            }
            if item.views == 0 {
                // Out of every view: what it draws with still loads and
                // compiles, so that it shows as soon as it comes into view.
                if matches!(item.shape, Shape::Mesh(_)) {
                    self.ensure_variants(
                        cached.shader,
                        samples,
                        cached.material.material.variants().without(Variants::INSTANCED),
                    );
                }
                continue;
            }
            let group = match &cached.group {
                Some(g) => g.clone(),
                None => {
                    let g = self.draw_group(cached.dressed.0);
                    cached.group = Some(g.clone());
                    g
                }
            };
            let data = &mut self.uniform_data[index * floats..(index + 1) * floats];
            let m = &cached.material.material;
            let draw = item.draw(frame);
            if matches!(item.source, Source::World) {
                data[..16].copy_from_slice(&IDENTITY);
                data[16..32].copy_from_slice(&IDENTITY);
            } else {
                let transform = item.transform(frame);
                data[..16].copy_from_slice(transform);
                data[16..32].copy_from_slice(&normal_matrix(transform)?);
            }
            data[32..36].copy_from_slice(&cached.color);
            data[36..40].copy_from_slice(&cached.settings);
            let parameters = draw.and_then(|draw| draw.parameters).unwrap_or(cached.default_params);
            data[40..44].copy_from_slice(&parameters);
            // Unused extra vectors read as zero.
            let extra = &mut data[44..SURFACE_AT];
            extra.fill(0.0);
            if let Some(values) = draw.and_then(|draw| draw.extra.as_deref()) {
                let count = values.len().min(MAX_DRAW_EXTRA * 4);
                extra[..count].copy_from_slice(&values[..count]);
            }
            // Every path writes the material's surface: a batch's draws share its first's
            // material, or one of the same render state.
            data[SURFACE_AT..SURFACE_AT + SURFACE_FLOATS].copy_from_slice(&cached.surface_vecs);
            data[MAPS_AT..MAPS_AT + 8].copy_from_slice(&cached.uv_maps);
            data[MAPS_AT + 8..MAPS_AT + MAPS_FLOATS].copy_from_slice(&cached.maps_tail);
            // Insets preserve authored coverage, separately from scene sky alpha.
            let shader = cached.shader;
            let (pipeline, visible) = if persistent {
                let inset = item.inset().is_some();
                let samples = if inset { INSET_SAMPLES } else { samples };
                let variant = if inset { Variants::INSET_PERSISTENT } else { Variants::PERSISTENT };
                if !self.ensure_variants(shader, samples, variant) {
                    continue;
                }
                let made = self.pipelines(shader, samples);
                (if inset { made.inset_persistent.clone() } else { made.persistent.clone() }, None)
            } else {
                match (item.inset(), item.shape) {
                    (Some(_), _) => {
                        let mut variant = Variants::INSET;
                        if !m.opaque() {
                            variant = Variants::INSET_BLENDED;
                        }
                        if !self.ensure_variants(shader, INSET_SAMPLES, variant) {
                            continue;
                        }
                        let made = self.pipelines(shader, INSET_SAMPLES);
                        let mut pipeline = made.inset.clone();
                        if !m.opaque() {
                            pipeline = made.inset_blended.clone();
                        }
                        (pipeline, None)
                    }
                    (None, Shape::Batch(batch)) => (Some(scratch.batches[batch].pipeline.clone()), None),
                    (None, _) => {
                        if !self.ensure_variants(
                            shader,
                            samples,
                            m.variants().without(Variants::INSTANCED | Variants::SKINNED),
                        ) {
                            continue;
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
                }
            };
            let mut pipeline = pipeline.expect("an ensured variant is compiled");
            let mut skin = None;
            let mut morph = None;
            if let (Some(target), Some(mesh)) = (scratch.target_of_source(item.source), mesh) {
                if scratch.posed(target) {
                    match self.pose_draw(frame, scratch, target, (mesh, m, shader), samples, item.inset().is_some()) {
                        Posed::Unskinned => {}
                        Posed::Waiting => continue,
                        Posed::Ready(skinned, at) => {
                            pipeline = skinned;
                            skin = Some(at);
                        }
                    }
                }
                if scratch.morphed(target) {
                    morph = self.morph_draw(frame, scratch, target, mesh);
                }
            }
            let offset =
                u32::try_from(index as u64 * self.stride).context("draw uniforms lie past a dynamic offset's reach")?;
            scratch.ready.push(ReadyEntry {
                item: index,
                offset,
                group,
                material: cached.material.clone(),
                shader,
                mesh,
                pipeline,
                visible,
                skin,
                morph,
                meshlet_slot: None,
                dag_slot: None,
                persistent,
            });
        }
        self.skinning.upload(&self.device, &self.queue, &self.context.skin_group);
        self.morphing.upload(&self.device, &self.queue);
        if item_count > 0 {
            self.queue.write_buffer(
                &self.draw_uniform,
                0,
                bytemuck::cast_slice(&self.uniform_data[..item_count * floats]),
            );
        }
        Ok(())
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
            let item = &items[entry.item];
            if item.inset().is_some() {
                continue;
            }
            let m = &entry.material.material;
            // What a reflection probe's capture sees: opaque surfaces.
            if item.views & scratch.capture_bit != 0 && !m.overlay && m.opaque() {
                scratch.capture_list.push(index);
            }
            if item.views & CAMERA == 0 {
                continue;
            }
            if m.overlay {
                scratch.overlays.push(index);
            } else if m.opaque() {
                scratch.opaque.push(index);
            } else {
                scratch.translucent.push(index);
            }
        }
        for list in [&mut scratch.opaque, &mut scratch.capture_list] {
            crate::sort::sort_by_u64_key(list, |&index| {
                let entry = &ready[index];
                let item = &items[entry.item];
                (u64::from(stencil(&entry.material.material)) << 33)
                    | (u64::from(item.material) << 1)
                    | u64::from(matches!(item.shape, Shape::Batch(_)))
            });
        }
        let view = &frame.view;
        let depth_of = |transform: &Matrix| {
            f64::from(view[2]) * f64::from(transform[12])
                + f64::from(view[6]) * f64::from(transform[13])
                + f64::from(view[10]) * f64::from(transform[14])
                + f64::from(view[14])
        };
        crate::sort::sort_by_u64_key(&mut scratch.translucent, |&index| {
            crate::sort::f64_sortable(depth_of(items[ready[index].item].transform(frame)))
        });
        // Lit, opaque surfaces cast shadows, unless their material says not:
        // those a shadow map sees, whether the camera does or not, sorted
        // like the opaque ones.
        if shadowed {
            for (index, entry) in ready.iter().enumerate() {
                let item = &items[entry.item];
                let stage = entry.stage(items) as usize;
                if item.inset().is_none()
                    && item.views != 0
                    && entry.material.material.casts_shadow()
                    && matches!(self.assets.current(entry.shader), Some(GpuAsset::Shader(shader))
                        if shader.shadow[stage].is_some())
                {
                    if item.views & !CAMERA & !scratch.capture_bit != 0 {
                        scratch.casters.push(index);
                    }
                    if item.views & CAMERA != 0 {
                        scratch.contact_casters.push(index);
                    }
                }
            }
            for list in [&mut scratch.casters, &mut scratch.contact_casters] {
                crate::sort::sort_by_u64_key(list, |&index| {
                    let item = &items[ready[index].item];
                    (u64::from(item.material) << 1) | u64::from(matches!(item.shape, Shape::Batch(_)))
                });
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
            self.built_ins.prepare([crate::built_in::BuiltIn::Decal { samples: self.samples_shown, paint }]);
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
                Some(last) if last.key == key && last.scene_filter == material.material.scene_filter => last.count += 1,
                _ => batches.push(SpriteBatch {
                    pipeline: pipeline.clone(),
                    texture: texture.view.clone(),
                    scene_filter: material.material.scene_filter,
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
        self.solids_revision += 1;
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
            // Solids have no tangents: they read the zeros.
            self.ensure_zero_tangents(count as usize);
            self.ensure_default_extra_attrs(count as usize);
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
            #[cfg(feature = "render-rt")]
            let rt_source = Some(std::sync::Arc::new(super::RtMeshSource {
                id: self.view_id(),
                vertices: batch.vertices,
                indices: None,
            }));
            #[cfg(not(feature = "render-rt"))]
            let rt_source = None;
            self.solid_meshes.push(SolidMesh { material: batch.material, buffer, count, rt_source });
        }
    }
}
