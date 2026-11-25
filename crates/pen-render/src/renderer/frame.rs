// A frame drawn: what it draws worked out (batching.rs), then encoded: the
// shadow pass, the scene's passes (opaque surfaces, decals over them,
// translucent surfaces and overlays), the passes after the scene (post.rs),
// insets, sprites and the veil. And another program's sprites drawn over a
// finished frame.
use std::collections::HashSet;

use anyhow::{Result, anyhow};
use pen_protocol::{Catalog, RenderFrame, Sprite};

use super::timing::Span;
use super::validate::validate_frame;
use super::{Drawing, GpuAsset, MeshAsset, ReadyEntry, Scratch, ShaderAsset, Shape, State, particles, ribbons};
use crate::asset_store::{AssetKind, Material};
use crate::built_in::BuiltIn;
use crate::decals::{DECAL_MARK, Decals};
use crate::loader::{Job, Priority};
use crate::post::{Extras, Post};
use crate::shader::Layer;
use crate::sprites::Sprites;
use crate::subsurface::MARK as SUBSURFACE_MARK;
use crate::{RenderStats, RenderTarget};

// Bind groups no frame has drawn with for this many frames are let go.
const GROUP_LIFETIME: u64 = 120;

// How much of an overlay shows through the scene in front of it.
const OCCLUDED_STRENGTH: f64 = 0.3;

/// The stencil marks a material's surfaces stamp: the frame's aberration
/// splits them, decals lie on them, and the scene's picture is diffused over
/// them (subsurface.rs).
pub(super) fn stencil(material: &Material) -> u32 {
    (if material.aberration { 1 } else { 0 })
        | if material.decals { DECAL_MARK } else { 0 }
        | if material.subsurface { SUBSURFACE_MARK } else { 0 }
}

impl Drawing<'_> {
    /// Draws newly disoccluded Phase-2 meshlet clusters for `entry` (`draws_phase2` buffer),
    /// emitting zero overlapping triangles with Phase 1.
    pub(super) fn draw_meshlet_phase2(&self, pass: &mut wgpu::RenderPass<'_>, entry: &ReadyEntry) -> bool {
        let Some(slot) = entry.meshlet_slot else {
            return false;
        };
        let Some(slot_data) = self.meshlet_indirect.get(slot) else {
            return false;
        };
        let item = &self.items[entry.item];
        if item.lod != 0 || !matches!(item.shape, Shape::Mesh(_)) {
            return false;
        }
        let Some(GpuAsset::Mesh(mesh)) = entry.mesh.and_then(|m| self.assets.current(m)) else {
            return false;
        };
        let Some(ml) = &mesh.meshlets else {
            return false;
        };
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, self.tangents(mesh).slice(..));
        pass.set_vertex_buffer(2, self.extra_attrs(mesh).slice(..));
        pass.set_index_buffer(ml.indices.slice(..), wgpu::IndexFormat::Uint32);
        let ml_count = slot_data.count;
        let ind_buf = &slot_data.draws_phase2;
        if self.multi_draw_indirect {
            pass.multi_draw_indexed_indirect(ind_buf, 0, ml_count);
        } else {
            for k in 0..ml_count {
                pass.draw_indexed_indirect(ind_buf, u64::from(k) * 20);
            }
        }
        true
    }

    /// Draws `entry`'s item into `pass` as `view` sees it (a batch's
    /// instances in that view), counting it into `stats` if given.
    pub(super) fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        entry: &ReadyEntry,
        view: usize,
        stats: Option<&mut RenderStats>,
    ) -> bool {
        let item = &self.items[entry.item];
        match item.shape {
            Shape::Mesh(_) => {
                let Some(GpuAsset::Mesh(mesh)) = entry.mesh.and_then(|slot| self.assets.current(slot)) else {
                    return false;
                };
                // A posed draw: its joints and weights, and its joint matrices.
                let mut has_slot2 = false;
                if let (Some(at), Some(skin), Some(group)) = (entry.skin, &mesh.skin, self.skin) {
                    pass.set_vertex_buffer(2, skin.vertices.slice(..));
                    pass.set_bind_group(3, group, &[at]);
                    has_slot2 = true;
                }
                let lod = mesh.lods.get(usize::from(item.lod)).unwrap_or(&mesh.lods[0]);
                if let (Some((start, end)), Some(morphed)) = (entry.morph, self.morphed) {
                    pass.set_vertex_buffer(0, morphed.slice(start..end));
                } else {
                    pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                }
                pass.set_vertex_buffer(1, self.tangents(mesh).slice(..));
                let count = if let Some(instances) = &mesh.instances {
                    pass.set_vertex_buffer(2, instances.buffer.slice(..));
                    has_slot2 = true;
                    instances.count
                } else {
                    1
                };
                pass.set_vertex_buffer(if has_slot2 { 3 } else { 2 }, self.extra_attrs(mesh).slice(..));
                if view == 0
                    && item.lod == 0
                    && let (Some(slot), Some(dag)) = (entry.dag_slot, &mesh.meshlet_dag)
                    && let Some(out) = self.dag_outputs.get(slot)
                {
                    pass.set_index_buffer(dag.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                    if self.multi_draw_indirect {
                        pass.multi_draw_indexed_indirect_count(
                            &out.draws_buf,
                            0,
                            &out.draw_count_buf,
                            0,
                            dag.cluster_count,
                        );
                        if let Some(stats) = stats {
                            stats.draw_calls += 1;
                            stats.triangles += u64::from(lod.count / 3);
                        }
                    } else {
                        for k in 0..dag.cluster_count {
                            pass.draw_indexed_indirect(&out.draws_buf, u64::from(k) * 20);
                        }
                        if let Some(stats) = stats {
                            stats.draw_calls += dag.cluster_count;
                            stats.triangles += u64::from(lod.count / 3);
                        }
                    }
                } else if view == 0
                    && item.lod == 0
                    && let (Some(slot), Some(ml)) = (entry.meshlet_slot, &mesh.meshlets)
                    && let Some(slot_data) = self.meshlet_indirect.get(slot)
                {
                    let ml_count = slot_data.count;
                    let ind_buf = &slot_data.draws_phase1;
                    pass.set_index_buffer(ml.indices.slice(..), wgpu::IndexFormat::Uint32);
                    if self.multi_draw_indirect {
                        pass.multi_draw_indexed_indirect(ind_buf, 0, ml_count);
                        if let Some(stats) = stats {
                            stats.draw_calls += 1;
                            stats.triangles += u64::from(lod.count / 3);
                        }
                    } else {
                        for k in 0..ml_count {
                            pass.draw_indexed_indirect(ind_buf, u64::from(k) * 20);
                        }
                        if let Some(stats) = stats {
                            stats.draw_calls += ml_count;
                            stats.triangles += u64::from(lod.count / 3);
                        }
                    }
                } else {
                    pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(lod.first..lod.first + lod.count, 0, 0..count);
                    if let Some(stats) = stats {
                        stats.draw_calls += 1;
                        stats.triangles += u64::from(lod.count / 3) * u64::from(count);
                        if mesh.instances.is_some() {
                            stats.instanced_draw_calls += 1;
                            stats.instances += count;
                        }
                    }
                }
            }
            Shape::Batch(index) => {
                let batch = &self.batches[index];
                let Some(GpuAsset::Mesh(mesh)) = self.assets.current(batch.mesh) else { return false };
                // Culled on the GPU: its instances and triangles are
                // counted as they are read back.
                if let Some(culled) = &batch.gpu {
                    let draws = self.culling.draw(pass, culled, view, mesh, self.tangents(mesh), self.extra_attrs(mesh));
                    if let Some(stats) = stats {
                        stats.draw_calls += draws;
                        stats.instanced_draw_calls += draws;
                    }
                    return draws > 0;
                }
                let ranges = &batch.ranges[view];
                if ranges.iter().all(|&(_, count)| count == 0) {
                    return false;
                }
                pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                pass.set_vertex_buffer(1, self.tangents(mesh).slice(..));
                pass.set_vertex_buffer(2, self.instances.slice(..));
                pass.set_vertex_buffer(3, self.extra_attrs(mesh).slice(..));
                pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                let mut stats = stats;
                for (lod, &(first, count)) in mesh.lods.iter().zip(ranges) {
                    if count == 0 {
                        continue;
                    }
                    pass.draw_indexed(lod.first..lod.first + lod.count, 0, first..first + count);
                    if let Some(stats) = stats.as_deref_mut() {
                        stats.draw_calls += 1;
                        stats.instanced_draw_calls += 1;
                        stats.instances += count;
                        stats.triangles += u64::from(count) * u64::from(lod.count / 3);
                    }
                }
            }
            Shape::Solid(index) => {
                let solid = &self.solids[index];
                pass.set_vertex_buffer(0, solid.buffer.slice(..));
                pass.set_vertex_buffer(1, self.zero_tangents.slice(..));
                pass.set_vertex_buffer(2, self.default_extra_attrs.slice(..));
                pass.draw(0..solid.count, 0..1);
                if let Some(stats) = stats {
                    stats.draw_calls += 1;
                }
            }
        }
        true
    }

    /// The mesh's tangents, or zeros for each of its vertices.
    fn tangents<'b>(&'b self, mesh: &'b MeshAsset) -> &'b wgpu::Buffer {
        mesh.tangents.as_ref().unwrap_or(self.zero_tangents)
    }

    /// The mesh's extra vertex attributes (`uv1`, `color`), or the shared white defaults.
    fn extra_attrs<'b>(&'b self, mesh: &'b MeshAsset) -> &'b wgpu::Buffer {
        mesh.extra_attrs.as_ref().unwrap_or(self.default_extra_attrs)
    }

    pub(super) fn shader(&self, entry: &ReadyEntry) -> &ShaderAsset {
        match self.assets.current(entry.shader) {
            Some(GpuAsset::Shader(shader)) => shader,
            _ => unreachable!("a ready entry's shader is loaded"),
        }
    }
}

impl State {
    /// The passes after the scene the last frame drew.
    pub fn passes(&self) -> Vec<&'static str> {
        self.post.as_ref().map(Post::passes).unwrap_or_default()
    }

    /// The next frame the mist's volume draws takes no history from the ones before.
    pub fn reset_volumetrics(&mut self) {
        self.volumetrics.reset();
    }

    /// Reads a texture of the last frame's froxel volume back, waiting for the
    /// GPU (for tools and tests).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_volume(&self, layer: crate::volume::VolumeLayer) -> Option<crate::volume::VolumeImage> {
        self.volumetrics.read(layer)
    }

    /// Auto exposure takes the exposure its next frame's scene asks for at once.
    pub fn reset_exposure(&mut self) {
        if let Some(post) = &mut self.post {
            post.reset_exposure();
        }
    }

    /// Starts loading what `sprites` draw with (their materials, shaders and
    /// textures); whether all of it is ready to draw.
    pub fn prepare_sprites(&mut self, sprites: &[Sprite], catalog: &Catalog) -> Result<bool> {
        let _ = self.device.poll(wgpu::PollType::Poll);
        let now = self.now();
        self.receive(now);
        let mut ready = true;
        let mut seen = HashSet::new();
        for sprite in sprites.iter().filter(|sprite| seen.insert(sprite.material)) {
            let key = catalog
                .materials
                .get(sprite.material as usize)
                .ok_or_else(|| anyhow!("sprite must name a material"))?;
            crate::asset_store::validate_asset_key(key)?;
            let Some(material) = self.material(catalog, sprite.material)? else {
                ready = false;
                continue;
            };
            let shader = self.resolve(AssetKind::SpriteShader, &material.material.shader)?;
            ready &= matches!(self.assets.current(shader), Some(GpuAsset::SpriteShader(_)));
            if let Some(texture) = &material.material.texture {
                let slot = self.resolve(AssetKind::Texture, texture)?;
                ready &= matches!(self.assets.current(slot), Some(GpuAsset::Texture(_)));
            }
        }
        Ok(ready)
    }

    /// Draws `sprites` over `output`, a finished image of `size`, which stays
    /// under them: another program's interface over the game's frame.
    pub fn overlay(
        &mut self,
        sprites: &[Sprite],
        catalog: &Catalog,
        output: &wgpu::TextureView,
        size: (u32, u32),
    ) -> Result<()> {
        for sprite in sprites {
            let key = catalog
                .materials
                .get(sprite.material as usize)
                .ok_or_else(|| anyhow!("sprite must name a material"))?;
            crate::asset_store::validate_asset_key(key)?;
        }
        let _ = self.device.poll(wgpu::PollType::Poll);
        let now = self.now();
        self.receive(now);
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
        let batched = self.sprite_batches(sprites, catalog, &mut scratch.sprites, &mut scratch.sprite_order);
        if batched.is_ok() && !scratch.sprites.is_empty() {
            let device = self.device.clone();
            // Over the HDR picture the frame drew into, decoded again.
            let surface = output;
            let output = &self.screen.view(&device, &self.queue, surface, size);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("overlay") });
            let layout = self.context.sprite_group.clone();
            let drawer = self.sprites.get_or_insert_with(|| Sprites::new(&device, layout));
            drawer.draw(
                &device,
                &self.queue,
                &mut encoder,
                output,
                size,
                &scratch.sprites,
                sprites,
                &scratch.sprite_order,
                None,
            );
            self.screen.decode(&mut encoder, surface);
            self.queue.submit([encoder.finish()]);
        }
        self.scratch = scratch;
        batched
    }

    pub fn submit(&mut self, frame: &RenderFrame, catalog: &Catalog, target: &mut dyn RenderTarget) -> Result<()> {
        // What a frame works out reuses the last one's memory.
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
        let drawn = self.draw(frame, catalog, target, &mut scratch);
        self.scratch = scratch;
        drawn
    }

    pub(super) fn draw(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
        target: &mut dyn RenderTarget,
        scratch: &mut Scratch,
    ) -> Result<()> {
        validate_frame(frame)?;
        self.check_names(frame, catalog)?;
        if let Some(detected) = crate::camera_precision::detect_depth_mode(&frame.projection) {
            self.set_depth_mode(detected);
        }
        let depth_mode = self.context.depth_mode();
        let (mut field_upload_bytes, mut field_upload_calls, mut field_input_texels) = (0, 0, 0);
        for patch in &frame.fields {
            let (bytes, calls) = self.write_field(catalog.material(patch.material), patch);
            field_upload_bytes += bytes;
            field_upload_calls += calls;
            field_input_texels += patch.values.len() as u64;
        }

        // What the loader brought, and resources the GPU is done with.
        let _ = self.device.poll(wgpu::PollType::Poll);
        let now = self.now();
        self.receive(now);
        self.waiting = false;
        if !self.wired.in_secondary_camera {
            self.frames += 1;
            self.loading_seen = frame.loading;
            if !frame.loading {
                self.ready_sent = false;
            }
            if self.frames.is_multiple_of(GROUP_LIFETIME) {
                let oldest = self.frames - GROUP_LIFETIME;
                self.groups.retain(|_, (_, used)| *used >= oldest);
            }
            self.fade(now);
        }
        if self.post.is_none() {
            let mut post = Post::new(&self.device, &self.queue);
            post.set_tonemap(self.tonemap, self.context.display.encoded_peak());
            self.post = Some(post);
        }
        // Apply per-frame tonemap selection from RenderFrame when nonzero (or keep host override).
        let effective_tonemap = if frame.grade.tonemap != 0 {
            crate::tonemap::TonemapOperator::from_u32(frame.grade.tonemap)
        } else {
            self.tonemap
        };
        if let Some(post) = &mut self.post {
            post.set_tonemap(effective_tonemap, self.context.display.encoded_peak());
        }
        // Execute non-split-screen RTT secondary cameras before main-view batching so materials sample them fresh.
        self.run_rtt_cameras(frame, catalog)?;
        // Split-screen fast path: skip the unused primary 3D scene pass, render each split-screen
        // camera into cached targets, composite onto `output`, then draw HUD sprites and veil once on top.
        if !self.wired.in_secondary_camera && frame.camera_targets.iter().any(|c| c.split_screen) {
            return self.draw_split_screen(
                frame,
                catalog,
                target,
                scratch,
                now,
                (field_upload_bytes, field_upload_calls, field_input_texels),
            );
        }
        let samples = self.quality(frame, catalog, scratch)?;
        // The veil covers the frame while the passes around the scene
        // compile (it waits for them, see below). Without one, the frame
        // waits for those on their way; natively, rather than compiling them
        // again here.
        let covered = self.veil.as_ref().is_some_and(|veil| veil.clearing.is_none());
        #[cfg(not(target_arch = "wasm32"))]
        while !covered && self.built_ins.coming(BuiltIn::all(samples)) && self.wait(std::time::Duration::from_secs(10))
        {
        }
        self.update_solids(&frame.solids);
        self.fit_cascades(frame);
        // What the frame captures of its reflection probes is culled for with the rest.
        self.plan_reflections(frame);
        let reflected = super::reflections::screen_level(frame);
        // The render scale, less what holding a frame budget takes off it.
        let timed = self.timer.fresh();
        let scale = frame.quality.scale * self.resolution.factor(timed);
        let mut stats =
            RenderStats { field_upload_bytes, field_upload_calls, field_input_texels, ..Default::default() };
        // Only depth-writing objects can be regrouped freely. Transparent
        // objects retain their global back-to-front order.
        #[allow(clippy::cast_precision_loss, reason = "pixels")]
        let height = target.size().1 as f32 * scale;
        self.batch(frame, catalog, samples, height, scratch, &mut stats)?;
        let shadowed = self.light(frame) || reflected > 0;
        self.prepare_items(frame, catalog, samples, scratch)?;
        // Decals whose material and texture are ready, by material.
        self.decal_batches(frame, catalog, scratch)?;
        let prepared_pbr_decals = self.prepare_pbr_decals(frame, catalog)?;
        // Sprites over the finished image, in order.
        self.sprite_batches(&frame.sprites, catalog, &mut scratch.sprites, &mut scratch.sprite_order)?;
        // Particles and ribbon trails whose material, texture and pipeline are ready.
        self.prepare_particles(frame, catalog, samples, scratch)?;
        self.prepare_ribbons(frame, catalog, samples, scratch)?;
        self.sort_passes(frame, scratch, shadowed);
        // Until they are ready, draw the cover and any opted-in startup sprites.
        if covered && !self.built_ins.ready(BuiltIn::all(samples)) {
            self.waiting = true;
            self.last_stats = RenderStats::default();
            let Some(surface) = target.view() else { return Ok(()) };
            let output = self.screen.view(&self.device, &self.queue, &surface, target.size());
            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.cover(&mut encoder, &output, frame, now, target.size(), true);
            if frame.loading && !scratch.sprites.is_empty() {
                let layout = self.context.sprite_group.clone();
                let sprites = self.sprites.get_or_insert_with(|| Sprites::new(&self.device, layout));
                sprites.draw(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &output,
                    target.size(),
                    &scratch.sprites,
                    &frame.sprites,
                    &scratch.sprite_order,
                    None,
                );
            }
            self.screen.decode(&mut encoder, &surface);
            self.queue.submit([encoder.finish()]);
            target.present();
            return Ok(());
        }
        // Each inset's targets and scene, ready to draw into.
        let (output_width, output_height) = target.size();
        self.prepare_insets(frame, (output_width, output_height), &mut scratch.insets);
        let composite = (!scratch.insets.is_empty()).then(|| self.built_ins.get(BuiltIn::Composite).clone());

        let tables = self.look_up_tables(frame, catalog)?;
        // Screen-space reflections and temporal upscaling read TAA's jittered history.
        let quality = pen_protocol::Quality {
            msaa: samples,
            scale,
            taa: frame.quality.taa || reflected > 0 || frame.quality.upscale,
            ..frame.quality
        };
        // Translucent surfaces, particles, overlays, 3D debug lines and JFA outlines draw
        // after the opaque picture is misted, in a pass of their own; a frame without them
        // draws the scene as one picture.
        let line_scene = self.gather_line_scene(frame);
        let has_3d_lines = line_scene.as_ref().is_some_and(|s| s.batch.has_3d());
        let jfa_scene = self.gather_jfa_outlines(frame);
        let translucent = !scratch.translucent.is_empty()
            || !scratch.overlays.is_empty()
            || !scratch.emitters.is_empty()
            || !scratch.ribbons.is_empty()
            || has_3d_lines
            || jfa_scene.is_some();
        let mut opaque_mips = false;
        let mut opaque_depth = !scratch.emitters.is_empty() || !scratch.ribbons.is_empty();
        for &index in scratch.translucent.iter().chain(&scratch.overlays) {
            let entry = &scratch.ready[index];
            if let Some(GpuAsset::Shader(shader)) = self.assets.current(entry.shader) {
                opaque_mips |= shader.reads_opaque_color;
                opaque_depth |= shader.reads_opaque_depth;
            }
        }
        let volume = self.volumetrics.prepare(&self.context, frame, &quality, target.size(), translucent);
        let planned_scene_size = crate::post::scene_size(&self.device, target.size(), quality.scale);
        let physical_sky = self.prepare_physical_sky(frame, planned_scene_size)?;
        // Opaque surfaces of a subsurface material have the scene's picture
        // diffused over them once it is resolved; a frame with none makes no passes
        // and no textures for it.
        let subsurface = scratch.opaque.iter().any(|&index| scratch.ready[index].material.material.subsurface);
        let targets = self.post.as_mut().unwrap().begin(
            &mut self.built_ins,
            frame,
            quality,
            target.size(),
            &volume,
            physical_sky.as_ref(),
            Extras { translucent, opaque_depth, opaque_mips, subsurface },
        );
        if targets.targets_recreated {
            self.velocity.reset_target();
            self.hiz.reset_target();
        }
        if frame.exposure.cut {
            self.hiz.invalidate();
        }
        // With TAA the scene draws through a projection jittered within its
        // pixels: the scene uniform's, after its view.
        let jittered = self.post.as_ref().unwrap().jittered();
        let previous_vp = self.post.as_ref().unwrap().previous_vp();
        if let Some(projection) = jittered {
            self.queue.write_buffer(&self.scene_uniform, 64, bytemuck::cast_slice(&projection));
        }
        // FSR 2.2/3.1 negative texture mip-bias (`-log2(out_w / in_w)`) stored in `scene.shadow_params.w`
        // so textured surfaces rendered at reduced scale retain full native detail.
        let mip_bias = if quality.upscale {
            crate::taa::fsr_mip_bias(targets.size.0, output_width)
        } else {
            0.0
        };
        let mip_bias_offset = ((super::SCENE_FLOATS - 1) * 4) as u64;
        self.queue.write_buffer(&self.scene_uniform, mip_bias_offset, bytemuck::bytes_of(&mip_bias));
        self.volumetrics.set_projection(&frame.view, &jittered.unwrap_or(frame.projection));
        self.prepare_screen(reflected);
        self.prepare_contact(frame, targets.size, reflected > 0);
        if let Some(opaque) = &targets.opaque {
            self.bind_translucent(opaque, &volume);
        }
        let active_proj_for_trails = jittered.unwrap_or(frame.projection);
        self.bind_particles(targets.opaque.as_ref(), &volume, &mut scratch.emitters);
        self.bind_ribbons(frame, &active_proj_for_trails, targets.opaque.as_ref(), &volume, &mut scratch.ribbons);
        self.write_screen(frame, reflected);
        let (cookie_slots, cookie_uploads) = self.prepare_area_cookies(frame, catalog)?;
        self.clusters.update(
            &self.device,
            &self.queue,
            frame,
            targets.size,
            &self.local,
            super::shadows::local_atlas(&self.shadows),
            &cookie_slots,
        );
        let device = self.device.clone();
        let mut encoder = device.create_command_encoder(&Default::default());
        self.clusters.encode_cookies(&device, &mut encoder, &cookie_uploads);
        // The sky's reflection, drawn again if its colors changed.
        self.environment.encode(&device, &mut encoder);
        // Physical atmosphere LUTs and raymarched 3D volumetric clouds when enabled (matching jittered rays).
        self.encode_physical_sky(&mut encoder, frame, &active_proj_for_trails, targets.size)?;
        // Timestamps around the passes, read back a few frames later, while
        // the host or a frame budget wants them.
        self.timer.begin(self.timing || self.resolution.budgeted());
        // Evaluate morph targets and cull instanced batches on the GPU before any pass draws them.
        self.morphing.dispatch(&device, &mut encoder, &self.assets);
        self.culling.dispatch(&mut encoder, &self.timer, &self.instance_buffer, &self.hiz);
        let redraw = if shadowed {
            let signatures = self.local_signatures(frame, scratch);
            self.plan_local_shadows(&signatures, &mut stats)
        } else {
            Vec::new()
        };
        let capture = self.prepare_capture(frame, scratch, samples);
        let oit_pipelines = self.prepare_oit_pipelines(frame, scratch, samples);
        let tlas_ready = self.prepare_tlas_scene(&mut encoder, frame, scratch)?;
        if tlas_ready && frame.quality.ddgi {
            self.encode_ddgi_prepass(&mut encoder, frame)?;
        }
        let restir_di_active =
            tlas_ready && frame.quality.ray_tracing >= 3 && super::wired_passes::has_active_point_spot_lights(frame);
        #[allow(clippy::cast_precision_loss, reason = "viewport height in pixels")]
        let dag_viewport_height = targets.size.1 as f32;
        self.encode_meshlet_culling_phase1(
            &device,
            &mut encoder,
            frame,
            scratch,
            &active_proj_for_trails,
            dag_viewport_height,
            depth_mode,
        )?;
        let ready = &scratch.ready;
        let drawing = Drawing {
            assets: &self.assets,
            items: &scratch.items,
            batches: &scratch.batches,
            solids: &self.solid_meshes,
            instances: &self.instance_buffer,
            zero_tangents: &self.zero_tangents,
            default_extra_attrs: &self.default_extra_attrs,
            skin: self.skinning.group(),
            morphed: self.morphing.output(),
            culling: &self.culling,
            meshlet_indirect: &self.meshlet_indirect,
            dag_outputs: &self.meshlet_dag_outputs,
            multi_draw_indirect: self.device.features().contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT),
        };
        if let Some((vel_view, vel_depth)) = &targets.velocity {
            let active_proj = jittered.unwrap_or(frame.projection);
            self.velocity.encode(
                &device,
                &self.queue,
                &mut encoder,
                vel_view,
                vel_depth,
                frame,
                &active_proj,
                previous_vp,
                scratch,
                &drawing,
                &self.skinning,
                ready,
                &self.timer,
            );
        }
        if tlas_ready && frame.quality.ray_tracing > 0 {
            self.thin_gbuffer.encode(
                &device,
                &self.queue,
                &mut encoder,
                targets.size,
                depth_mode,
                frame,
                &active_proj_for_trails,
                scratch,
                &drawing,
                &self.skinning,
                ready,
                &self.timer,
            );
        }
        if shadowed {
            self.cast_shadows(
                &mut encoder,
                &drawing,
                frame,
                &scratch.casters,
                &scratch.contact_casters,
                &redraw,
                ready,
            );
        }
        // The probe's faces, once the shadows they read are drawn, ahead of the
        // scene that reads the cubes.
        if let Some(capture) = &capture {
            super::reflections::draw_capture(
                capture,
                &device,
                &self.queue,
                &mut encoder,
                &mut self.environment,
                &drawing,
                ready,
                &scratch.capture_list,
                self.capture_clusters.group(),
            );
        }
        self.environment.commit_reflections(&self.queue, frame, self.capture.as_ref(), capture.is_some());
        // The mist's froxels, once the shadow maps and the lights they read are drawn.
        self.volumetrics.encode(
            &mut encoder,
            &self.timer,
            &self.scene_group.as_ref().unwrap().0,
            self.clusters.group(),
        );
        let (volume_froxels, volume_history) = self.volumetrics.stats();
        stats.volume_froxels = volume_froxels;
        stats.volume_history = volume_history;
        if let Some(particles) = &self.particles {
            particles.simulate(&mut encoder);
        }
        // Acquired as late as possible. An HDR surface that needs decoding
        // gets the frame through a picture of its own (display.rs); with a
        // colour filter on, the frame draws into the filter's texture first,
        // which its pass draws into that picture (or the surface) last.
        let Some(surface) = target.view() else { return Ok(()) };
        let bypass = self.wired.bypass_display_output;
        let screen = if bypass {
            surface.clone()
        } else {
            self.screen.view(&self.device, &self.queue, &surface, (output_width, output_height))
        };
        let format = self.context.format;
        let filtered = if bypass {
            None
        } else {
            self.filter.begin(&device, &self.queue, format, self.context.cache.get(), (output_width, output_height))
        };
        let output = filtered.as_ref().unwrap_or(&screen).clone();
        // Opaque surfaces, then any decals over them. A frame with translucent
        // surfaces, particles or overlays then has its opaque picture misted, and
        // draws those over it in a second pass, with the mist and the opaque
        // picture to read.
        let has_gpu_particles = self.has_active_gpu_particles(frame);
        let layered = !scratch.decals.is_empty() || !prepared_pbr_decals.is_empty() || has_gpu_particles;
        let wants_meshlet_phase2 = self.wired.meshlet_phase1_used_hiz && !self.wired.meshlet_plan.is_empty();
        let scene_group = &self.scene_group.as_ref().unwrap().0;
        let lights = self.clusters.group();
        let opaque_lights = if restir_di_active { self.clusters.area_only_group() } else { lights };
        let draw_layers = |pass: &mut wgpu::RenderPass<'_>,
                           group: &wgpu::BindGroup,
                           light_group: &wgpu::BindGroup,
                           layers: &[(&[usize], Layer)],
                           stats: &mut RenderStats| {
            pass.set_bind_group(0, group, &[]);
            pass.set_bind_group(2, light_group, &[]);
            pass.set_blend_constant(wgpu::Color {
                r: OCCLUDED_STRENGTH,
                g: OCCLUDED_STRENGTH,
                b: OCCLUDED_STRENGTH,
                a: OCCLUDED_STRENGTH,
            });
            pass.set_stencil_reference(0);
            let mut mark = 0;
            let mut current: Option<&wgpu::RenderPipeline> = None;
            for (entries, layer) in layers {
                for &index in entries.iter() {
                    let entry = &ready[index];
                    let wanted = stencil(&entry.material.material);
                    if mark != wanted {
                        mark = wanted;
                        pass.set_stencil_reference(mark);
                    }
                    let pipeline = match layer {
                        Layer::Visible => entry.visible.as_ref().expect("an overlay has its full-strength pipeline"),
                        _ => &entry.pipeline,
                    };
                    if current != Some(pipeline) {
                        current = Some(pipeline);
                        pass.set_pipeline(pipeline);
                    }
                    pass.set_bind_group(1, &entry.group, &[entry.offset]);
                    drawing.draw(pass, entry, 0, Some(stats));
                }
            }
        };
        // The clear colour is what the sky and the mist take back out of the resolved
        // scene where no surface was; its alpha 1 is what surfaces draw 0 over.
        let clear = wgpu::Color {
            r: f64::from(frame.clear.r),
            g: f64::from(frame.clear.g),
            b: f64::from(frame.clear.b),
            a: 1.0,
        };
        {
            let mut first = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(targets.color(clear, true, !layered && !wants_meshlet_phase2))],
                timestamp_writes: self.timer.pass(Span::Scene),
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(depth_mode.clear_value()),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0), store: wgpu::StoreOp::Store }),
                }),
                ..Default::default()
            });
            draw_layers(&mut first, scene_group, opaque_lights, &[(&scratch.opaque, Layer::Opaque)], &mut stats);
            Self::encode_environment_passes(
                &mut self.wired,
                &device,
                &self.queue,
                &mut first,
                frame,
                &active_proj_for_trails,
                samples,
                depth_mode,
            )?;
        }
        let wants_hiz_build = self.hiz.enabled()
            && ((self.culling.supported() && self.culling.from.is_some())
                || self.wired.last_meshlet_cull_draws > 0
                || self.wired.last_meshlet_dag_draws > 0);
        if wants_hiz_build {
            let active_proj = jittered.unwrap_or(frame.projection);
            let active_vp = crate::matrix::multiply(&active_proj, &frame.view);
            self.hiz.build(
                &device,
                &mut encoder,
                &targets.depth_only,
                targets.size,
                samples,
                depth_mode,
                active_vp,
                self.context.cache.get(),
                &self.timer,
            );
        }
        if wants_meshlet_phase2 {
            Self::encode_meshlet_culling_phase2(
                &mut self.wired,
                &self.hiz,
                &self.assets,
                &self.meshlet_indirect,
                &self.timer,
                &device,
                &self.queue,
                &mut encoder,
                frame,
                &active_proj_for_trails,
                depth_mode,
            )?;
            {
                let mut phase2 = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("scene_meshlet_phase2"),
                    color_attachments: &[Some(targets.color(clear, false, !layered))],
                    timestamp_writes: self.timer.pass(Span::Scene),
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &targets.depth,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                        stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    }),
                    ..Default::default()
                });
                phase2.set_bind_group(0, scene_group, &[]);
                phase2.set_bind_group(2, opaque_lights, &[]);
                phase2.set_stencil_reference(0);
                let mut mark = 0;
                let mut current: Option<&wgpu::RenderPipeline> = None;
                for &index in &scratch.opaque {
                    let entry = &ready[index];
                    if entry.meshlet_slot.is_none() {
                        continue;
                    }
                    let wanted = stencil(&entry.material.material);
                    if mark != wanted {
                        mark = wanted;
                        phase2.set_stencil_reference(mark);
                    }
                    if current != Some(&entry.pipeline) {
                        current = Some(&entry.pipeline);
                        phase2.set_pipeline(&entry.pipeline);
                    }
                    phase2.set_bind_group(1, &entry.group, &[entry.offset]);
                    drawing.draw_meshlet_phase2(&mut phase2, entry);
                }
            }
            if wants_hiz_build {
                let active_proj = jittered.unwrap_or(frame.projection);
                let active_vp = crate::matrix::multiply(&active_proj, &frame.view);
                self.hiz.build(
                    &device,
                    &mut encoder,
                    &targets.depth_only,
                    targets.size,
                    samples,
                    depth_mode,
                    active_vp,
                    self.context.cache.get(),
                    &self.timer,
                );
            }
        }
        if !scratch.decals.is_empty() {
            let decals = self.decals.get_or_insert_with(|| Decals::new(&device));
            let mut decal_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("decals"),
                color_attachments: &[Some(targets.color(
                    clear,
                    false,
                    prepared_pbr_decals.is_empty() && !has_gpu_particles,
                ))],
                timestamp_writes: self.timer.pass(Span::Scene),
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: None,
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            decals.draw(
                &device,
                &self.queue,
                &mut self.built_ins,
                &mut decal_pass,
                frame,
                &scratch.decals,
                &scratch.decal_order,
                &targets.depth_only,
                samples,
                targets.size,
            );
        }
        if !prepared_pbr_decals.is_empty() {
            Self::encode_pbr_decals_prepared(
                &mut self.wired,
                &device,
                &self.queue,
                &mut encoder,
                frame,
                &prepared_pbr_decals,
                targets.color(clear, false, !has_gpu_particles),
                &targets.depth,
                &targets.depth_only,
                samples,
                targets.size,
                depth_mode,
            )?;
        }
        if has_gpu_particles {
            Self::encode_gpu_particles_wired(
                &mut self.wired,
                &mut self.built_ins,
                &device,
                &mut encoder,
                frame,
                &active_proj_for_trails,
                targets.color(clear, false, true),
                &targets.depth,
                &targets.depth_only,
                targets.size,
                samples,
                depth_mode,
            )?;
        }
        if tlas_ready && frame.quality.ray_tracing > 0 {
            let ddgi_buf = self.wired.ddgi_active_this_frame.then_some(self.environment.probe_data_buffer());
            let vel_view = targets.velocity.as_ref().map(|(v, _)| v);
            let thin_gbuf = match (
                self.thin_gbuffer.normal_roughness_view(),
                self.thin_gbuffer.albedo_metallic_view(),
                self.thin_gbuffer.depth_view(),
            ) {
                (Some(nr), Some(am), Some(d)) => Some((nr, am, d)),
                _ => None,
            };
            Self::encode_rt_wired(
                &mut self.wired,
                &mut self.built_ins,
                &device,
                &self.queue,
                &mut encoder,
                frame,
                self.frames,
                &active_proj_for_trails,
                &targets.scene,
                &targets.depth_only,
                vel_view,
                thin_gbuf,
                ddgi_buf,
                targets.size,
                samples,
                depth_mode,
            )?;
        }
        if subsurface {
            self.post.as_mut().unwrap().diffuse(&mut self.built_ins, &mut encoder, frame, &self.timer);
        }
        if targets.opaque.is_some() {
            let post = self.post.as_mut().unwrap();
            post.opaque(&mut self.built_ins, &mut encoder, frame, &self.timer);
            let group = &self.translucent_group.as_ref().expect("a translucent frame binds its own scene").0;
            let use_oit = frame.quality.oit && oit_pipelines.iter().any(Option::is_some);
            if use_oit {
                let mut fallback_translucent = Vec::new();
                for (slot, &entry_idx) in scratch.translucent.iter().enumerate() {
                    if oit_pipelines.get(slot).and_then(Option::as_ref).is_none() {
                        fallback_translucent.push(entry_idx);
                    }
                }
                let has_after_oit = !fallback_translucent.is_empty()
                    || !scratch.emitters.is_empty()
                    || !scratch.ribbons.is_empty()
                    || !scratch.overlays.is_empty()
                    || has_3d_lines;
                {
                    let mut backdrop_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("translucent_backdrop"),
                        color_attachments: &[Some(targets.color(wgpu::Color::BLACK, true, false))],
                        timestamp_writes: self.timer.pass(Span::Scene),
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &targets.depth,
                            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                            stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                        }),
                        ..Default::default()
                    });
                    post.draw_backdrop(&mut self.built_ins, &mut backdrop_pass);
                }
                {
                    let oit = self
                        .wired
                        .oit_pass
                        .get_or_insert_with(|| crate::oit::OitPass::new(&device, crate::post::SCENE_FORMAT, None));
                    oit.resize_with_samples(&device, targets.size, samples);
                    let mut oit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("translucent_oit"),
                        color_attachments: &oit.attachments(),
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &targets.depth,
                            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                            stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                        }),
                        ..Default::default()
                    });
                    oit_pass.set_bind_group(0, group, &[]);
                    oit_pass.set_bind_group(2, lights, &[]);
                    oit_pass.set_stencil_reference(0);
                    let mut current: Option<&wgpu::RenderPipeline> = None;
                    for (slot, &entry_idx) in scratch.translucent.iter().enumerate() {
                        let Some(pipeline) = oit_pipelines.get(slot).and_then(Option::as_ref) else { continue };
                        let entry = &ready[entry_idx];
                        if current != Some(pipeline) {
                            current = Some(pipeline);
                            oit_pass.set_pipeline(pipeline);
                        }
                        oit_pass.set_bind_group(1, &entry.group, &[entry.offset]);
                        drawing.draw(&mut oit_pass, entry, 0, Some(&mut stats));
                    }
                }
                let oit = self.wired.oit_pass.as_mut().unwrap();
                oit.composite_attachment(
                    &device,
                    &mut encoder,
                    targets.color(wgpu::Color::BLACK, false, !has_after_oit),
                    samples,
                );
                if has_after_oit {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("translucent_after_oit"),
                        color_attachments: &[Some(targets.color(wgpu::Color::BLACK, false, true))],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &targets.depth,
                            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                            stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                        }),
                        ..Default::default()
                    });
                    if !fallback_translucent.is_empty() {
                        draw_layers(&mut pass, group, lights, &[(&fallback_translucent, Layer::Translucent)], &mut stats);
                    }
                    particles::draw(&mut pass, &scratch.emitters);
                    ribbons::draw(&mut pass, self.ribbons.as_ref(), &scratch.ribbons);
                    let overlays = [(&scratch.overlays[..], Layer::Occluded), (&scratch.overlays[..], Layer::Visible)];
                    draw_layers(&mut pass, group, lights, &overlays, &mut stats);
                    if let Some(lines) = &line_scene {
                        Self::encode_lines_3d_wired(
                            &mut self.wired,
                            &device,
                            &self.queue,
                            &mut pass,
                            lines,
                            &frame.view,
                            &active_proj_for_trails,
                            targets.size,
                            samples,
                            depth_mode,
                        )?;
                    }
                }
            } else {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("translucent"),
                    color_attachments: &[Some(targets.translucent_color())],
                    timestamp_writes: self.timer.pass(Span::Scene),
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &targets.depth,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                        stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    }),
                    ..Default::default()
                });
                post.draw_backdrop(&mut self.built_ins, &mut pass);
                draw_layers(&mut pass, group, lights, &[(&scratch.translucent, Layer::Translucent)], &mut stats);
                particles::draw(&mut pass, &scratch.emitters);
                ribbons::draw(&mut pass, self.ribbons.as_ref(), &scratch.ribbons);
                let overlays = [(&scratch.overlays[..], Layer::Occluded), (&scratch.overlays[..], Layer::Visible)];
                draw_layers(&mut pass, group, lights, &overlays, &mut stats);
                if let Some(lines) = &line_scene {
                    Self::encode_lines_3d_wired(
                        &mut self.wired,
                        &device,
                        &self.queue,
                        &mut pass,
                        lines,
                        &frame.view,
                        &active_proj_for_trails,
                        targets.size,
                        samples,
                        depth_mode,
                    )?;
                }
            }
        }
        if let Some(jfa) = &jfa_scene {
            Self::encode_jfa_outlines_wired(
                &mut self.wired,
                &mut self.built_ins,
                &self.assets,
                &device,
                &self.queue,
                &mut encoder,
                frame,
                catalog,
                jfa,
                &active_proj_for_trails,
                &targets.scene,
                &targets.depth_only,
                targets.size,
                samples,
                depth_mode,
            )?;
        }
        let has_filters = scratch.sprites.iter().any(|batch| batch.scene_filter);
        let wants_polish = super::wired_passes::WiredPasses::wants_post_polish(frame);
        if has_filters {
            let layout = self.context.sprite_group.clone();
            let sprites = self.sprites.get_or_insert_with(|| Sprites::new(&device, layout));
            let images = sprites.filter_targets(&device, (output_width, output_height), self.context.format);
            self.post.as_mut().unwrap().finish_base(
                &mut self.built_ins,
                &mut encoder,
                frame,
                &images[0],
                &self.timer,
                true,
                &tables,
            );
            sprites.prepare(&device, &self.queue, (output_width, output_height), &frame.sprites, &scratch.sprite_order);
            let mut current = 0;
            // Each sprite is a pass, even if adjacent filters share a material.
            for batch in scratch.sprites.iter().filter(|batch| batch.scene_filter) {
                for index in batch.first..batch.first + batch.count {
                    let next = 1 - current;
                    encoder.copy_texture_to_texture(
                        images[current].texture().as_image_copy(),
                        images[next].texture().as_image_copy(),
                        wgpu::Extent3d { width: output_width, height: output_height, depth_or_array_layers: 1 },
                    );
                    sprites.set_scene(Some(images[current].clone()));
                    sprites.draw_pass(
                        &device,
                        &mut encoder,
                        &images[next],
                        std::slice::from_ref(batch),
                        index..index + 1,
                        self.timer.pass(Span::Post),
                        true,
                    );
                    current = next;
                }
            }
            let filter_dst = if wants_polish {
                self.wired.polish_target(&device, (output_width, output_height), self.context.format)
            } else {
                output.clone()
            };
            self.post.as_mut().unwrap().finish_filtered(
                &mut self.built_ins,
                &mut encoder,
                frame,
                &images[current],
                &filter_dst,
                self.timer.pass(Span::Post),
            );
            if wants_polish {
                Self::encode_post_polish_wired(
                    &mut self.wired,
                    &mut self.built_ins,
                    &device,
                    &mut encoder,
                    frame,
                    self.frames,
                    self.context.format,
                    &filter_dst,
                    &targets.depth_only,
                    targets.size,
                    &output,
                    (output_width, output_height),
                    samples,
                )?;
            }
        } else if wants_polish {
            let polish_in = self.wired.polish_target(&device, (output_width, output_height), self.context.format);
            self.post.as_mut().unwrap().finish(&mut self.built_ins, &mut encoder, frame, &polish_in, &self.timer, &tables);
            Self::encode_post_polish_wired(
                &mut self.wired,
                &mut self.built_ins,
                &device,
                &mut encoder,
                frame,
                self.frames,
                self.context.format,
                &polish_in,
                &targets.depth_only,
                targets.size,
                &output,
                (output_width, output_height),
                samples,
            )?;
        } else {
            self.post.as_mut().unwrap().finish(&mut self.built_ins, &mut encoder, frame, &output, &self.timer, &tables);
        }
        let render2d_overlay_sdf = Self::encode_render2d_base_wired(
            &mut self.wired,
            &device,
            &self.queue,
            &mut encoder,
            frame,
            &output,
            self.context.format,
            (output_width, output_height),
        )?;
        if !scratch.sprites.is_empty() {
            let layout = self.context.sprite_group.clone();
            let sprites = self.sprites.get_or_insert_with(|| Sprites::new(&device, layout));
            sprites.set_scene(Some(targets.scene.clone()));
            if !has_filters {
                sprites.prepare(
                    &device,
                    &self.queue,
                    (output_width, output_height),
                    &frame.sprites,
                    &scratch.sprite_order,
                );
            }
        }
        // Stable order within each layer; sprites under a modal are drawn
        // before its backdrop and 3D icons, then the modal's own text follows.
        scratch.insets.sort_by_key(|&(index, _)| frame.insets[index].after_sprites);
        let mut cursor = 0;
        for layer in 0..scratch.insets.len() {
            let index = scratch.insets[layer].0;
            let boundary = frame.insets[index].after_sprites;
            let end = scratch.sprite_order.partition_point(|&index| index < boundary);
            if end > cursor {
                self.sprites.as_mut().unwrap().draw_prepared(
                    &device,
                    &mut encoder,
                    &output,
                    &scratch.sprites,
                    u32::try_from(cursor).expect("bounded sprite count")
                        ..u32::try_from(end).expect("bounded sprite count"),
                    self.timer.pass(Span::Overlay),
                );
                cursor = end;
            }
            if let Some(composite) = &composite {
                self.draw_insets(
                    &mut encoder,
                    &output,
                    &drawing,
                    &scratch.insets[layer..layer + 1],
                    ready,
                    composite,
                    &mut stats,
                );
            }
        }
        if frame.loading {
            self.cover(&mut encoder, &output, frame, now, (output_width, output_height), false);
        }
        if cursor < scratch.sprite_order.len() {
            self.sprites.as_mut().unwrap().draw_prepared(
                &device,
                &mut encoder,
                &output,
                &scratch.sprites,
                u32::try_from(cursor).expect("bounded sprite count")
                    ..u32::try_from(scratch.sprite_order.len()).expect("bounded sprite count"),
                self.timer.pass(Span::Overlay),
            );
        }
        if !render2d_overlay_sdf.is_empty() {
            Self::encode_render2d_overlay_wired(
                &mut self.wired,
                &device,
                &mut encoder,
                &render2d_overlay_sdf,
                &output,
                self.context.format,
                (output_width, output_height),
            );
        }
        if let Some(lines) = &line_scene
            && lines.batch.has_2d()
        {
            Self::encode_lines_2d_wired(
                &mut self.wired,
                &device,
                &self.queue,
                &mut encoder,
                lines,
                &output,
                self.context.format,
                (output_width, output_height),
            )?;
        }
        if !frame.loading {
            self.cover(&mut encoder, &output, frame, now, (output_width, output_height), false);
        }
        if !bypass {
            if filtered.is_some() {
                self.filter.finish(&mut encoder, &screen);
            }
            self.screen.decode(&mut encoder, &surface);
        }
        // The scale the scene drew at, as `Post` sized it.
        #[allow(clippy::cast_precision_loss, reason = "a ratio of pixels")]
        let drawn = targets.size.0 as f32 / output_width.max(1) as f32;
        stats.scale = drawn;
        stats.gpu = self.timer.latest();
        stats.exposure = self.post.as_mut().and_then(Post::exposure_seen);
        let kept = self.culling.latest();
        stats.instances += kept.instances;
        stats.triangles += kept.triangles;
        stats.culled += kept.culled;
        stats.shadow_culled += kept.shadow_culled;
        self.last_stats = stats;
        self.timer.resolve(&mut encoder);
        self.culling.finish(&mut encoder);
        if let Some(post) = &mut self.post {
            post.exposure_finish(&mut encoder);
        }
        self.queue.submit([encoder.finish()]);
        if let Some(post) = &mut self.post {
            post.exposure_submitted();
        }
        self.timer.submitted();
        self.culling.submitted();
        if !self.wired.in_secondary_camera {
            self.resolution.submitted(&self.queue, self.timer.timed());
        }
        target.present();
        if !self.wired.in_secondary_camera {
            if let Some(budget) = self.asset_budget
                && self.gpu_asset_bytes() > budget
            {
                let _ = self.evict_unused(2, budget);
            }
            // Once background compiles end, what they added to the pipeline
            // cache is written, in the background too.
            if self.frames.is_multiple_of(30) {
                let busy = self.pending() > 0;
                if self.busy && !busy {
                    self.loader.jobs.push(Job::Persist, Priority::Background);
                }
                self.busy = busy;
            }
        }
        Ok(())
    }

    /// Fast path for split-screen frames: renders each `split_screen == true` camera into
    /// temporally-isolated offscreen targets BEFORE acquiring display/filter targets, then
    /// composites them once with scene-filter sprites, ordered insets, HUD sprites, startup veil,
    /// global color filter, and HDR display decode.
    fn draw_split_screen(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
        target: &mut dyn RenderTarget,
        scratch: &mut Scratch,
        now: f64,
        (field_upload_bytes, field_upload_calls, field_input_texels): (u64, u32, u64),
    ) -> Result<()> {
        let samples = self.quality(frame, catalog, scratch)?;
        let covered = self.veil.as_ref().is_some_and(|veil| veil.clearing.is_none());
        #[cfg(not(target_arch = "wasm32"))]
        while !covered && self.built_ins.coming(BuiltIn::all(samples)) && self.wait(std::time::Duration::from_secs(10))
        {
        }
        let (output_width, output_height) = target.size();
        self.sprite_batches(&frame.sprites, catalog, &mut scratch.sprites, &mut scratch.sprite_order)?;
        if covered && !self.built_ins.ready(BuiltIn::all(samples)) {
            self.waiting = true;
            self.last_stats = RenderStats::default();
            let Some(surface) = target.view() else { return Ok(()) };
            let output = self.screen.view(&self.device, &self.queue, &surface, (output_width, output_height));
            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.cover(&mut encoder, &output, frame, now, (output_width, output_height), true);
            if frame.loading && !scratch.sprites.is_empty() {
                let layout = self.context.sprite_group.clone();
                let sprites = self.sprites.get_or_insert_with(|| Sprites::new(&self.device, layout));
                sprites.draw(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &output,
                    (output_width, output_height),
                    &scratch.sprites,
                    &frame.sprites,
                    &scratch.sprite_order,
                    None,
                );
            }
            self.screen.decode(&mut encoder, &surface);
            self.queue.submit([encoder.finish()]);
            target.present();
            return Ok(());
        }
        // 1. Render all split-screen child views FIRST into presentation-format intermediates
        //    with display decode and global color filter bypassed, and isolated temporal state.
        let (rects, child_waiting, mut stats) =
            self.render_split_screen_views(frame, catalog, (output_width, output_height))?;

        // 2. Prepare parent-level insets (after child views so draw uniforms hold inset draws).
        let parent_waiting_before_insets = self.waiting;
        if !frame.insets.is_empty() {
            self.batch_insets_only(frame, scratch)?;
            self.prepare_items(frame, catalog, samples, scratch)?;
            self.prepare_insets(frame, (output_width, output_height), &mut scratch.insets);
        }
        let composite = (!scratch.insets.is_empty()).then(|| self.built_ins.get(BuiltIn::Composite).clone());

        // 3. Acquire parent display & accessibility filter targets AFTER child submissions.
        let Some(surface) = target.view() else { return Ok(()) };
        let device = self.device.clone();
        let screen = self.screen.view(&device, &self.queue, &surface, (output_width, output_height));
        let format = self.context.format;
        let filtered =
            self.filter.begin(&device, &self.queue, format, self.context.cache.get(), (output_width, output_height));
        let output = filtered.as_ref().unwrap_or(&screen).clone();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("split_screen_frame"),
        });
        self.timer.begin(self.timing || self.resolution.budgeted());
        if !scratch.items.is_empty() {
            self.morphing.dispatch(&device, &mut encoder, &self.assets);
        }

        // 4. Composite split views, running scene-filter sprites when present.
        let has_filters = scratch.sprites.iter().any(|batch| batch.scene_filter);
        let composite_scene_view = if has_filters {
            let layout = self.context.sprite_group.clone();
            let sprites = self.sprites.get_or_insert_with(|| Sprites::new(&device, layout));
            let images = sprites.filter_targets(&device, (output_width, output_height), format);
            self.composite_split_screen(&mut encoder, &images[0], (output_width, output_height), rects)?;
            let sprites = self.sprites.as_mut().unwrap();
            sprites.prepare(&device, &self.queue, (output_width, output_height), &frame.sprites, &scratch.sprite_order);
            let mut current = 0;
            for batch in scratch.sprites.iter().filter(|batch| batch.scene_filter) {
                for index in batch.first..batch.first + batch.count {
                    let next = 1 - current;
                    encoder.copy_texture_to_texture(
                        images[current].texture().as_image_copy(),
                        images[next].texture().as_image_copy(),
                        wgpu::Extent3d { width: output_width, height: output_height, depth_or_array_layers: 1 },
                    );
                    sprites.set_scene(Some(images[current].clone()));
                    sprites.draw_pass(
                        &device,
                        &mut encoder,
                        &images[next],
                        std::slice::from_ref(batch),
                        index..index + 1,
                        self.timer.pass(Span::Post),
                        true,
                    );
                    current = next;
                }
            }
            self.wired.blit(&device, &mut encoder, &images[current], &output, format);
            Some(images[current].clone())
        } else {
            self.composite_split_screen(&mut encoder, &output, (output_width, output_height), rects)?;
            None
        };

        // 5. Draw ordered insets, HUD sprites, and startup veil identically to single-view frames.
        if !scratch.sprites.is_empty() {
            let layout = self.context.sprite_group.clone();
            let sprites = self.sprites.get_or_insert_with(|| Sprites::new(&device, layout));
            sprites.set_scene(composite_scene_view);
            if !has_filters {
                sprites.prepare(
                    &device,
                    &self.queue,
                    (output_width, output_height),
                    &frame.sprites,
                    &scratch.sprite_order,
                );
            }
        }
        let drawing = Drawing {
            assets: &self.assets,
            items: &scratch.items,
            batches: &scratch.batches,
            solids: &self.solid_meshes,
            instances: &self.instance_buffer,
            zero_tangents: &self.zero_tangents,
            default_extra_attrs: &self.default_extra_attrs,
            skin: self.skinning.group(),
            morphed: self.morphing.output(),
            culling: &self.culling,
            meshlet_indirect: &[],
            dag_outputs: &[],
            multi_draw_indirect: false,
        };
        let split_line_scene = self.gather_line_scene(frame);
        let split_render2d_overlay = Self::encode_render2d_base_wired(
            &mut self.wired,
            &device,
            &self.queue,
            &mut encoder,
            frame,
            &output,
            format,
            (output_width, output_height),
        )?;
        let ready = &scratch.ready;
        scratch.insets.sort_by_key(|&(index, _)| frame.insets[index].after_sprites);
        let mut cursor = 0;
        for layer in 0..scratch.insets.len() {
            let index = scratch.insets[layer].0;
            let boundary = frame.insets[index].after_sprites;
            let end = scratch.sprite_order.partition_point(|&index| index < boundary);
            if end > cursor {
                self.sprites.as_mut().unwrap().draw_prepared(
                    &device,
                    &mut encoder,
                    &output,
                    &scratch.sprites,
                    u32::try_from(cursor).expect("bounded sprite count")
                        ..u32::try_from(end).expect("bounded sprite count"),
                    self.timer.pass(Span::Overlay),
                );
                cursor = end;
            }
            if let Some(composite) = &composite {
                self.draw_insets(
                    &mut encoder,
                    &output,
                    &drawing,
                    &scratch.insets[layer..layer + 1],
                    ready,
                    composite,
                    &mut stats,
                );
            }
        }
        if frame.loading {
            self.cover(&mut encoder, &output, frame, now, (output_width, output_height), false);
        }
        if cursor < scratch.sprite_order.len() {
            self.sprites.as_mut().unwrap().draw_prepared(
                &device,
                &mut encoder,
                &output,
                &scratch.sprites,
                u32::try_from(cursor).expect("bounded sprite count")
                    ..u32::try_from(scratch.sprite_order.len()).expect("bounded sprite count"),
                self.timer.pass(Span::Overlay),
            );
        }
        if !split_render2d_overlay.is_empty() {
            Self::encode_render2d_overlay_wired(
                &mut self.wired,
                &device,
                &mut encoder,
                &split_render2d_overlay,
                &output,
                format,
                (output_width, output_height),
            );
        }
        if let Some(lines) = &split_line_scene
            && lines.batch.has_2d()
        {
            Self::encode_lines_2d_wired(
                &mut self.wired,
                &device,
                &self.queue,
                &mut encoder,
                lines,
                &output,
                format,
                (output_width, output_height),
            )?;
        }
        if !frame.loading {
            self.cover(&mut encoder, &output, frame, now, (output_width, output_height), false);
        }

        // 6. Apply global accessibility filter and display decode ONCE to the full composite.
        if filtered.is_some() {
            self.filter.finish(&mut encoder, &screen);
        }
        self.screen.decode(&mut encoder, &surface);
        stats.field_upload_bytes = field_upload_bytes;
        stats.field_upload_calls = field_upload_calls;
        stats.field_input_texels = field_input_texels;
        stats.gpu = self.timer.latest();
        self.waiting = self.waiting || parent_waiting_before_insets || child_waiting;
        self.last_stats = stats;
        self.timer.resolve(&mut encoder);
        self.queue.submit([encoder.finish()]);
        self.timer.submitted();
        self.resolution.submitted(&self.queue, self.timer.timed());
        target.present();
        if let Some(budget) = self.asset_budget
            && self.gpu_asset_bytes() > budget
        {
            let _ = self.evict_unused(2, budget);
        }
        if self.frames.is_multiple_of(30) {
            let busy = self.pending() > 0;
            if self.busy && !busy {
                self.loader.jobs.push(Job::Persist, Priority::Background);
            }
            self.busy = busy;
        }
        Ok(())
    }
}
