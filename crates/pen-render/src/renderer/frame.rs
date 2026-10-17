// A frame drawn: what it draws worked out (batching.rs), then encoded: the
// shadow pass, the scene's passes (opaque surfaces, decals over them,
// translucent surfaces and overlays), the passes after the scene (post.rs),
// sprites, insets and the veil. And another program's sprites drawn over a
// finished frame.
use std::collections::HashSet;

use anyhow::{Result, anyhow};
use pen_protocol::{Catalog, RenderFrame, Sprite};

use super::timing::Span;
use super::validate::validate_frame;
use super::{Drawing, GpuAsset, ReadyEntry, Scratch, ShaderAsset, Shape, State, particles};
use crate::asset_store::{AssetKind, Material};
use crate::built_in::BuiltIn;
use crate::decals::{DECAL_MARK, Decals};
use crate::loader::{Job, Priority};
use crate::post::Post;
use crate::shader::Layer;
use crate::sprites::Sprites;
use crate::{RenderStats, RenderTarget};

// Bind groups no frame has drawn with for this many frames are let go.
const GROUP_LIFETIME: u64 = 120;

// How much of an overlay shows through the scene in front of it.
const OCCLUDED_STRENGTH: f64 = 0.3;

/// The stencil marks a material's surfaces stamp: the frame's aberration
/// splits them, and decals lie on them.
pub(super) fn stencil(material: &Material) -> u32 {
    (if material.aberration { 1 } else { 0 }) | if material.decals { DECAL_MARK } else { 0 }
}

impl Drawing<'_> {
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
                if let (Some(at), Some(skin), Some(group)) = (entry.skin, &mesh.skin, self.skin) {
                    pass.set_vertex_buffer(1, skin.vertices.slice(..));
                    pass.set_bind_group(3, group, &[at]);
                }
                let lod = mesh.lods.get(usize::from(item.lod)).unwrap_or(&mesh.lods[0]);
                pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(lod.first..lod.first + lod.count, 0, 0..1);
                if let Some(stats) = stats {
                    stats.draw_calls += 1;
                    stats.triangles += u64::from(lod.count / 3);
                }
            }
            Shape::Batch(index) => {
                let batch = &self.batches[index];
                let Some(GpuAsset::Mesh(mesh)) = self.assets.current(batch.mesh) else { return false };
                // Culled on the GPU: its instances and triangles are
                // counted as they are read back.
                if let Some(culled) = &batch.gpu {
                    let draws = self.culling.draw(pass, culled, view, mesh);
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
                pass.set_vertex_buffer(1, self.instances.slice(..));
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
                pass.draw(0..solid.count, 0..1);
                if let Some(stats) = stats {
                    stats.draw_calls += 1;
                }
            }
        }
        true
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
        for patch in &frame.fields {
            self.write_field(catalog.material(patch.material), patch);
        }

        // What the loader brought, and resources the GPU is done with.
        let _ = self.device.poll(wgpu::PollType::Poll);
        let now = self.now();
        self.receive(now);
        self.frames += 1;
        self.waiting = false;
        if self.frames.is_multiple_of(GROUP_LIFETIME) {
            let oldest = self.frames - GROUP_LIFETIME;
            self.groups.retain(|_, (_, used)| *used >= oldest);
        }
        self.fade(now);
        if self.post.is_none() {
            self.post = Some(Post::new(&self.device, &self.queue));
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
        // The render scale, less what holding a frame budget takes off it.
        let timed = self.timer.fresh();
        let scale = frame.quality.scale * self.resolution.factor(timed);
        let mut stats = RenderStats::default();
        // Only depth-writing objects can be regrouped freely. Transparent
        // objects retain their global back-to-front order.
        #[allow(clippy::cast_precision_loss, reason = "pixels")]
        let height = target.size().1 as f32 * scale;
        self.batch(frame, catalog, samples, height, scratch, &mut stats)?;
        let shadowed = self.light(frame);
        self.prepare_items(frame, catalog, samples, scratch)?;
        // Decals whose material and texture are ready, by material.
        self.decal_batches(frame, catalog, scratch)?;
        // Sprites over the finished image, in order.
        self.sprite_batches(&frame.sprites, catalog, &mut scratch.sprites, &mut scratch.sprite_order)?;
        // Particles whose material, texture and pipeline are ready.
        self.prepare_particles(frame, catalog, samples, scratch)?;
        self.sort_passes(frame, scratch, shadowed);
        let ready = &scratch.ready;
        // Until they are ready, only the veil draws.
        if covered && !self.built_ins.ready(BuiltIn::all(samples)) {
            self.waiting = true;
            self.last_stats = RenderStats::default();
            let Some(surface) = target.view() else { return Ok(()) };
            let output = self.screen.view(&self.device, &self.queue, &surface, target.size());
            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.cover(&mut encoder, &output, frame, now, target.size(), true);
            self.screen.decode(&mut encoder, &surface);
            self.queue.submit([encoder.finish()]);
            target.present();
            return Ok(());
        }
        // Each inset's targets and scene, ready to draw into.
        let (output_width, output_height) = target.size();
        self.prepare_insets(frame, (output_width, output_height), &mut scratch.insets);
        let composite = (!scratch.insets.is_empty()).then(|| self.built_ins.get(BuiltIn::Composite).clone());

        let quality = pen_protocol::Quality { msaa: samples, scale, ..frame.quality };
        let targets = self.post.as_mut().unwrap().begin(&mut self.built_ins, frame, quality, target.size());
        // With TAA the scene draws through a projection jittered within its
        // pixels: the scene uniform's, after its view.
        if let Some(projection) = self.post.as_ref().unwrap().jittered() {
            self.queue.write_buffer(&self.scene_uniform, 64, bytemuck::cast_slice(&projection));
        }
        self.clusters.update(&self.device, &self.queue, frame, targets.size);
        let device = self.device.clone();
        let mut encoder = device.create_command_encoder(&Default::default());
        // Timestamps around the passes, read back a few frames later, while
        // the host or a frame budget wants them.
        self.timer.begin(self.timing || self.resolution.budgeted());
        // Instances of large batches culled on the GPU, before any pass
        // draws them.
        self.culling.dispatch(&mut encoder, &self.timer, &self.instance_buffer);
        let drawing = Drawing {
            assets: &self.assets,
            items: &scratch.items,
            batches: &scratch.batches,
            solids: &self.solid_meshes,
            instances: &self.instance_buffer,
            skin: self.skinning.group(),
            culling: &self.culling,
        };
        if shadowed {
            self.cast_shadows(&mut encoder, &drawing, &scratch.casters, ready);
        }
        if let Some(particles) = &self.particles {
            particles.simulate(&mut encoder);
        }
        // Acquired as late as possible. An HDR surface that needs decoding
        // gets the frame through a picture of its own (display.rs); with a
        // colour filter on, the frame draws into the filter's texture first,
        // which its pass draws into that picture (or the surface) last.
        let Some(surface) = target.view() else { return Ok(()) };
        let screen = self.screen.view(&self.device, &self.queue, &surface, (output_width, output_height));
        let format = self.context.format;
        let filtered =
            self.filter.begin(&device, &self.queue, format, self.context.cache.get(), (output_width, output_height));
        let output = filtered.as_ref().unwrap_or(&screen).clone();
        // Opaque surfaces, then any decals over them, then translucent ones
        // and overlays; without decals, all in one pass.
        let layered = !scratch.decals.is_empty();
        let scene_group = &self.scene_group.as_ref().unwrap().0;
        let lights = self.clusters.group();
        let draw_layers = |pass: &mut wgpu::RenderPass<'_>, layers: &[(&[usize], Layer)], stats: &mut RenderStats| {
            pass.set_bind_group(0, scene_group, &[]);
            pass.set_bind_group(2, lights, &[]);
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
        let later: [(&[usize], Layer); 3] = [
            (&scratch.translucent, Layer::Translucent),
            (&scratch.overlays, Layer::Occluded),
            (&scratch.overlays, Layer::Visible),
        ];
        let clear = wgpu::Color {
            r: f64::from(frame.clear.r),
            g: f64::from(frame.clear.g),
            b: f64::from(frame.clear.b),
            a: f64::from(frame.clear.a),
        };
        {
            let mut first = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(targets.color(clear, true, !layered))],
                timestamp_writes: self.timer.pass(Span::Scene),
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0), store: wgpu::StoreOp::Store }),
                }),
                ..Default::default()
            });
            let opaque: (&[usize], Layer) = (&scratch.opaque, Layer::Opaque);
            // Particles come after the translucent surfaces, before overlays.
            if layered {
                draw_layers(&mut first, &[opaque], &mut stats);
            } else {
                draw_layers(&mut first, &[opaque, later[0]], &mut stats);
                particles::draw(&mut first, &scratch.emitters);
                draw_layers(&mut first, &later[1..], &mut stats);
            }
        }
        if layered {
            let decals = self.decals.get_or_insert_with(|| Decals::new(&device));
            {
                let mut decal_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("decals"),
                    color_attachments: &[Some(targets.color(clear, false, false))],
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
            let mut last = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene over decals"),
                color_attachments: &[Some(targets.color(clear, false, true))],
                timestamp_writes: self.timer.pass(Span::Scene),
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                }),
                ..Default::default()
            });
            draw_layers(&mut last, &later[..1], &mut stats);
            particles::draw(&mut last, &scratch.emitters);
            draw_layers(&mut last, &later[1..], &mut stats);
        }
        self.post.as_mut().unwrap().finish(&mut self.built_ins, &mut encoder, frame, &output, &self.timer);
        if !scratch.sprites.is_empty() {
            let layout = self.context.sprite_group.clone();
            let sprites = self.sprites.get_or_insert_with(|| Sprites::new(&device, layout));
            let size = (output_width, output_height);
            sprites.draw(
                &device,
                &self.queue,
                &mut encoder,
                &output,
                size,
                &scratch.sprites,
                &frame.sprites,
                &scratch.sprite_order,
                self.timer.pass(Span::Overlay),
            );
        }
        if let Some(composite) = &composite {
            self.draw_insets(&mut encoder, &output, &drawing, &scratch.insets, ready, composite, &mut stats);
        }
        self.cover(&mut encoder, &output, frame, now, (output_width, output_height), false);
        if filtered.is_some() {
            self.filter.finish(&mut encoder, &screen);
        }
        self.screen.decode(&mut encoder, &surface);
        // The scale the scene drew at, as `Post` sized it.
        #[allow(clippy::cast_precision_loss, reason = "a ratio of pixels")]
        let drawn = targets.size.0 as f32 / output_width.max(1) as f32;
        stats.scale = drawn;
        stats.gpu = self.timer.latest();
        let kept = self.culling.latest();
        stats.instances += kept.instances;
        stats.triangles += kept.triangles;
        stats.culled += kept.culled;
        stats.shadow_culled += kept.shadow_culled;
        self.last_stats = stats;
        self.timer.resolve(&mut encoder);
        self.culling.finish(&mut encoder);
        self.queue.submit([encoder.finish()]);
        self.timer.submitted();
        self.culling.submitted();
        self.resolution.submitted(&self.queue, self.timer.timed());
        target.present();
        // Once background compiles end, what they added to the pipeline
        // cache is written, in the background too.
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
