// Shadows: depth maps of the casters seen from a light, which the scene's
// surfaces compare their depth against. The sun's, the game's cascades (each
// an orthographic window of the frame's light view, cascades.rs), are layers
// of one array texture: shaders that read `shadow_cascades` (binding 3) take
// each surface from the map of its slice of the view, blending at the slice's
// end; others read the farthest map at binding 1. The local lights' faces
// (local_shadows.rs) are layers of another, bound with the lights at group 2.
// A frame that asks for contact shadows also draws its casters' depth from
// the camera into a texture of the scene's size (binding 9).
use pen_protocol::RenderFrame;

use super::cascades::{Cascade, cascades};
use super::timing::Span;
use super::{
    Drawing, Floats, GpuAsset, ReadyEntry, SCENE_BYTES, SCENE_FLOATS, Shadows, State, heavens, lighting, uniform_buffer,
};
use crate::environment::Environment;
use crate::local_shadows::{Held, MAX_FACES, assign, select};
use crate::matrix::{IDENTITY, multiply};
use pen_protocol::boundary::MAX_CASCADES;

/// WGSL a material shader adds to take the sun's shadows: `sun_shadow(world,
/// normal, position)`, `sun_visibility`, `contact_shadow` (shadows.wgsl).
pub const SHADOWS_WGSL: &str = include_str!("../shaders/shadows.wgsl");

// The share of a slice, at its far end, over which a surface blends into the
// next map's.
const BAND: f32 = 0.15;

/// The scene uniform for `frame`, its sun's `count` maps (`maps`) included:
/// what `shadows.wgsl` and materials read (materials.md lists the fields).
pub(super) fn scene_floats(frame: &RenderFrame, maps: &[Cascade; MAX_CASCADES], count: usize) -> Floats<SCENE_FLOATS> {
    let sun = frame.shadow.texels > 0 && count > 0;
    let texel = 1.0 / frame.shadow.texels.max(1) as f32;
    let mut scene = Floats::<SCENE_FLOATS>::new();
    scene.extend(frame.view).extend(frame.projection);
    lighting(&mut scene, &frame.ambient, &frame.light_direction, &frame.light_color);
    // Binding 1's map, the farthest, for shaders that read the game's alone.
    let last = &maps[count.max(1) - 1];
    scene.extend(if sun { multiply(&last.projection, &frame.shadow.view) } else { IDENTITY });
    scene.extend([if sun { 1.0 } else { 0.0 }, texel, if sun { last.texel * 1.5 } else { 0.0 }, 0.0]);
    heavens(&mut scene, &frame.sky, &frame.view);
    // The maps: how many, the band they blend over, the filter and the
    // contact shadow's steps; what each sees, and each one's texel in
    // texture coordinates and in the world, where its slice ends and its
    // depth range; then the soft shadows' settings.
    let contact = if sun { frame.shadow.contact } else { 0 };
    #[allow(clippy::cast_precision_loss, reason = "at most MAX_CASCADES")]
    scene.extend([count as f32, BAND, frame.shadow.filter as f32, contact as f32]);
    for map in maps {
        scene.extend(multiply(&map.projection, &frame.shadow.view));
    }
    for map in maps {
        let range = if map.projection[10] != 0.0 { 1.0 / map.projection[10].abs() } else { 0.0 };
        scene.extend([texel, map.texel, map.far, range]);
    }
    scene.extend([frame.shadow.softness, frame.shadow.contact_length, 0.0, 0.0]);
    scene
}

/// How many scene uniforms the passes that draw into maps use: the sun's
/// maps, the local faces, and the camera's depth.
const PASSES: usize = MAX_CASCADES + MAX_FACES + 1;
const CONTACT_PASS: usize = MAX_CASCADES + MAX_FACES;

/// A shadow map array's texture, a view of each layer to draw into, and the
/// view of them all.
pub(super) struct ShadowMaps {
    texture: wgpu::Texture,
    layers: Vec<wgpu::TextureView>,
    pub(super) all: wgpu::TextureView,
}

/// The camera's depth, for contact shadows.
pub(super) struct ContactDepth {
    view: wgpu::TextureView,
    size: (u32, u32),
}

/// The local lights' faces, and which light's what each layer holds.
pub(super) struct LocalMaps {
    maps: ShadowMaps,
    texels: u32,
    /// Which texture this is, to know when the lights' bind group must change.
    id: u64,
    /// What each layer holds: the id and kind of the light whose first face
    /// it is, if it is one's first, and a signature of what was drawn in it.
    held: [Held; MAX_FACES],
    signatures: [Option<u64>; MAX_FACES],
}

fn depth_array(device: &wgpu::Device, label: &str, texels: u32, layers: u32) -> ShadowMaps {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: texels, height: texels, depth_or_array_layers: layers },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let layer = |index: u32| {
        texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: index,
            array_layer_count: Some(1),
            ..Default::default()
        })
    };
    let layers = (0..layers).map(layer).collect();
    let all = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    ShadowMaps { texture, layers, all }
}

/// A scene bind group: its uniform, the game's map (`game`, else the first
/// layer of `maps`), the comparison sampler, every map and the camera's
/// depth (`contact`, else the stand-in's first layer).
#[allow(clippy::too_many_arguments, reason = "the group's resources")]
fn scene_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    maps: &ShadowMaps,
    game: Option<&wgpu::TextureView>,
    contact: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    sky: (&Environment, &wgpu::Buffer),
) -> wgpu::BindGroup {
    let [cube, table, sky_sampler, light, probes] = sky.0.entries(sky.1);
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene"),
        layout,
        entries: &[
            cube,
            table,
            sky_sampler,
            light,
            probes,
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(game.unwrap_or(&maps.layers[0])),
            },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&maps.all) },
            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(contact) },
        ],
    })
}

impl State {
    pub(super) fn shadow_resources(&mut self) -> &mut Shadows {
        let device = &self.device;
        let scene_layout = &self.context.scene_group;
        let environment = &self.environment;
        self.shadows.get_or_insert_with(|| {
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                compare: Some(wgpu::CompareFunction::Less),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            let stand_in = depth_array(device, "shadow stand-in", 1, 1);
            // The passes that draw into maps bind the stand-in, as they draw into the real ones.
            let (uniforms, groups) = (0..PASSES)
                .map(|_| {
                    let uniform = uniform_buffer(device, "shadow scene", SCENE_BYTES);
                    let group = scene_group(
                        device,
                        scene_layout,
                        &uniform,
                        &stand_in,
                        None,
                        &stand_in.layers[0],
                        &sampler,
                        (environment, environment.empty()),
                    );
                    (uniform, group)
                })
                .unzip();
            Shadows {
                uniforms,
                groups,
                sampler,
                stand_in,
                maps: None,
                map_id: 0,
                contact: None,
                contact_id: 0,
                contact_serial: 0,
                local: None,
            }
        })
    }

    /// A scene bind group for `uniform` that reads no shadow map (an inset's),
    /// lit by the sky light in `light`.
    pub(super) fn unshadowed_scene(&mut self, uniform: &wgpu::Buffer, light: &wgpu::Buffer) -> wgpu::BindGroup {
        let device = self.device.clone();
        let layout = self.context.scene_group.clone();
        self.shadow_resources();
        let resources = self.shadows.as_ref().unwrap();
        scene_group(
            &device,
            &layout,
            uniform,
            &resources.stand_in,
            None,
            &resources.stand_in.layers[0],
            &resources.sampler,
            (&self.environment, light),
        )
    }

    /// The frame's shadow maps: the sun's cascades (every one only if a
    /// loaded shader reads them), and the point and spot lights that cast.
    pub(super) fn fit_cascades(&mut self, frame: &RenderFrame) {
        let cascaded = self
            .assets
            .slots
            .iter()
            .any(|slot| matches!(&slot.current, Some(GpuAsset::Shader(shader)) if shader.cascaded));
        self.cascades = cascades(frame, cascaded);
        let mut local =
            select(&frame.lights, &frame.view, &frame.projection, frame.shadow.local, frame.shadow.local_texels);
        let held = self.shadows.as_ref().and_then(|shadows| shadows.local.as_ref()).map(|local| local.held);
        assign(&mut local, &held.unwrap_or([None; MAX_FACES]));
        self.local = local;
    }

    /// Writes the scene's uniform and each shadow pass's, making the maps if
    /// need be; whether anything casts a shadow: the sun or a local light.
    pub(super) fn light(&mut self, frame: &RenderFrame) -> bool {
        // The game fits its shadow maps' view; 0 texels for none.
        let shadow_texels = frame.shadow.texels;
        let sun = (shadow_texels > 0 && self.cascades.1 > 0).then_some(&frame.shadow);
        let (maps, count) = self.cascades;
        let contact = if sun.is_some() { frame.shadow.contact } else { 0 };
        let scene = scene_floats(frame, &maps, count);
        self.queue.write_buffer(&self.scene_uniform, 0, scene.bytes());
        self.environment.update(&self.queue, frame);
        // Each shadow pass sees the scene from its light.
        let device = self.device.clone();
        let queue = self.queue.clone();
        if let Some(sun) = sun {
            let layers = u32::try_from(count).expect("at most MAX_CASCADES");
            let resources = self.shadow_resources();
            let fits = resources.maps.as_ref().is_some_and(|maps| {
                maps.texture.width() == shadow_texels && maps.texture.depth_or_array_layers() >= layers
            });
            if !fits {
                // Room for every map a frame may draw, so that one fewer or
                // more makes no new texture.
                let layers = if count > 1 { u32::try_from(MAX_CASCADES).expect("a few maps") } else { 1 };
                resources.maps = Some(depth_array(&device, "shadow maps", shadow_texels, layers));
                resources.map_id += 1;
            }
            for (map, uniform) in maps[..count].iter().zip(&resources.uniforms) {
                let mut data = Floats::<SCENE_FLOATS>::new();
                data.extend(sun.view).extend(map.projection);
                lighting(&mut data, &frame.ambient, &frame.light_direction, &frame.light_color);
                queue.write_buffer(uniform, 0, data.bytes());
            }
        }
        let faces = self.local.faces.len();
        let local_views: Vec<_> = self.local.faces.iter().map(|face| (face.view, face.projection)).collect();
        if faces > 0 {
            let texels = frame.shadow.local_texels;
            let resources = self.shadow_resources();
            if resources.local.as_ref().is_none_or(|local| local.texels != texels) {
                let id = resources.local.as_ref().map_or(1, |local| local.id + 1);
                resources.local = Some(LocalMaps {
                    maps: depth_array(&device, "local shadow maps", texels, u32::try_from(MAX_FACES).unwrap_or(18)),
                    texels,
                    id,
                    held: [None; MAX_FACES],
                    signatures: [None; MAX_FACES],
                });
            }
            for ((view, projection), uniform) in local_views.iter().zip(&resources.uniforms[MAX_CASCADES..]) {
                let mut data = Floats::<SCENE_FLOATS>::new();
                data.extend(*view).extend(*projection);
                lighting(&mut data, &frame.ambient, &frame.light_direction, &frame.light_color);
                queue.write_buffer(uniform, 0, data.bytes());
            }
        }
        // The game's map, the farthest, at binding 1.
        self.bind_scene(sun.is_some(), count);
        sun.is_some() || faces > 0 || contact > 0
    }

    /// The scene's bind group, made again when the maps it reads changed.
    pub(super) fn bind_scene(&mut self, sun: bool, count: usize) {
        let bound = if sun {
            let resources = self.shadow_resources();
            (resources.map_id, count.max(1) - 1, resources.contact_id)
        } else {
            (0, 0, self.shadow_resources().contact_id)
        };
        if self.scene_group.as_ref().is_none_or(|(_, map)| *map != bound) {
            let resources = self.shadows.as_ref().unwrap();
            let maps = resources.maps.as_ref().filter(|_| bound.0 != 0);
            let contact = resources.contact.as_ref().filter(|_| bound.2 != 0);
            let group = scene_group(
                &self.device,
                &self.context.scene_group,
                &self.scene_uniform,
                maps.unwrap_or(&resources.stand_in),
                maps.map(|maps| &maps.layers[bound.1]),
                contact.map_or(&resources.stand_in.layers[0], |contact| &contact.view),
                &resources.sampler,
                (&self.environment, self.environment.light()),
            );
            self.scene_group = Some((group, bound));
        }
    }

    /// Makes (or drops) the camera's depth texture for a scene of `size`
    /// pixels, as `frame` asks for contact shadows.
    pub(super) fn prepare_contact(&mut self, frame: &RenderFrame, size: (u32, u32)) {
        let wanted = self.cascades.1 > 0 && frame.shadow.contact > 0 && frame.projection[11] < -0.5;
        let device = self.device.clone();
        let resources = self.shadow_resources();
        if !wanted {
            resources.contact_id = 0;
        } else {
            if resources.contact.as_ref().is_none_or(|contact| contact.size != size) {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("contact depth"),
                    size: wgpu::Extent3d { width: size.0.max(1), height: size.1.max(1), depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = texture.create_view(&Default::default());
                resources.contact = Some(ContactDepth { view, size });
                resources.contact_serial += 1;
            }
            resources.contact_id = resources.contact_serial;
        }
        let (sun, count) = (self.cascades.1 > 0, self.cascades.1);
        self.bind_scene(sun, count);
    }

    /// A signature for each face of the selected local lights: of the light
    /// and of everything that casts into it. A face whose signature is what
    /// it was when it was drawn holds what it would hold now.
    pub(super) fn local_signatures(&self, frame: &RenderFrame, scratch: &super::Scratch) -> Vec<u64> {
        use super::{Shape, Source};
        use std::hash::{Hash, Hasher};
        let count = self.cascades.1;
        let mut signatures = Vec::with_capacity(self.local.faces.len());
        let floats = |hasher: &mut std::collections::hash_map::DefaultHasher, values: &[f32]| {
            for value in values {
                value.to_bits().hash(hasher);
            }
        };
        for caster in &self.local.casters {
            let light = &frame.lights[caster.light];
            for face in 0..caster.faces() {
                let view = 1 + count + caster.first + face;
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                floats(
                    &mut hasher,
                    &[
                        light.position.x,
                        light.position.y,
                        light.position.z,
                        light.radius,
                        light.direction.x,
                        light.direction.y,
                        light.direction.z,
                        light.outer,
                    ],
                );
                (frame.shadow.local_texels, caster.id, face).hash(&mut hasher);
                for &index in &scratch.casters {
                    let entry = &scratch.ready[index];
                    let item = &scratch.items[entry.item];
                    if item.views & (1 << view) == 0 {
                        continue;
                    }
                    // What it is drawn with, and of what revision.
                    (entry.shader, self.assets.slots[entry.shader].revision, item.material, item.lod).hash(&mut hasher);
                    if let Some(mesh) = entry.mesh {
                        (mesh, self.assets.slots[mesh].revision).hash(&mut hasher);
                    }
                    match item.shape {
                        Shape::Mesh(mesh) => {
                            mesh.hash(&mut hasher);
                            if let Some(draw) = item.draw(frame) {
                                floats(&mut hasher, &draw.transform);
                                if let Some(parameters) = draw.parameters {
                                    floats(&mut hasher, &parameters);
                                }
                                if let Some(extra) = &draw.extra {
                                    floats(&mut hasher, extra);
                                }
                            }
                            // A posed draw's joints move it without its transform.
                            if matches!(item.source, Source::Draw(draw) if scratch.posed(draw)) {
                                self.frames.hash(&mut hasher);
                            }
                        }
                        Shape::Batch(batch) => {
                            let batch = &scratch.batches[batch];
                            (batch.mesh, self.assets.slots[batch.mesh].revision).hash(&mut hasher);
                            let ranges: Vec<(u32, u32)> = match batch.gpu {
                                Some(_) => vec![(batch.first, batch.count)],
                                None => batch.ranges[view].to_vec(),
                            };
                            for (first, count) in ranges {
                                count.hash(&mut hasher);
                                let from = first as usize * 20;
                                floats(&mut hasher, &self.instance_data[from..from + count as usize * 20]);
                            }
                        }
                        Shape::Solid(solid) => (solid, self.solids_revision).hash(&mut hasher),
                    }
                }
                signatures.push(hasher.finish());
            }
        }
        signatures
    }

    /// Which local faces (by their index in the selection) to draw this
    /// frame: all but those whose signature is the one they were drawn
    /// with, in the layer the same light held. Remembers what it decides.
    pub(super) fn plan_local_shadows(&mut self, signatures: &[u64], stats: &mut crate::RenderStats) -> Vec<bool> {
        let casters = &self.local.casters;
        let mut redraw = vec![true; self.local.faces.len()];
        stats.local_shadow_casters = u32::try_from(casters.len()).unwrap_or(u32::MAX);
        stats.local_shadow_faces = u32::try_from(redraw.len()).unwrap_or(u32::MAX);
        let Some(local) = self.shadows.as_mut().and_then(|shadows| shadows.local.as_mut()) else { return redraw };
        for caster in casters {
            let same = local.held[caster.layer] == Some((caster.id, caster.cube));
            for face in 0..caster.faces() {
                let layer = caster.layer + face;
                let signature = Some(signatures[caster.first + face]);
                if same && local.signatures[layer] == signature {
                    redraw[caster.first + face] = false;
                    stats.local_shadow_kept += 1;
                }
                local.signatures[layer] = signature;
            }
        }
        // The layers each light holds from now on; the rest hold nothing worth keeping.
        local.held = [None; MAX_FACES];
        let mut used = [false; MAX_FACES];
        for caster in casters {
            local.held[caster.layer] = Some((caster.id, caster.cube));
            used[caster.layer..caster.layer + caster.faces()].fill(true);
        }
        for (signature, used) in local.signatures.iter_mut().zip(used) {
            if !used {
                *signature = None;
            }
        }
        redraw
    }

    /// The shadow passes: the casters each map sees, from its light, into
    /// it; the local lights' faces `redraw` says to draw (the others hold
    /// what their casters still give); and the camera's depth.
    #[allow(clippy::too_many_arguments, reason = "what the passes draw")]
    pub(super) fn cast_shadows(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        drawing: &Drawing<'_>,
        frame: &RenderFrame,
        casters: &[usize],
        seen_by_camera: &[usize],
        redraw: &[bool],
        ready: &[ReadyEntry],
    ) {
        let count = self.cascades.1;
        let Some(shadows) = &self.shadows else { return };
        if let Some(maps) = &shadows.maps {
            for (map, layer) in maps.layers.iter().enumerate().take(count) {
                let mut casting = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow"),
                    color_attachments: &[],
                    timestamp_writes: self.timer.pass(Span::Shadow),
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: layer,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                casting.set_bind_group(0, &shadows.groups[map], &[]);
                casting.set_bind_group(2, self.clusters.empty(), &[]);
                // The maps are the views after the camera's.
                draw_casters(&mut casting, drawing, casters, ready, map + 1);
            }
        }
        // The local lights' faces: the views after the sun's maps'.
        if let Some(local) = &shadows.local {
            for caster in &self.local.casters {
                for face in 0..caster.faces() {
                    let index = caster.first + face;
                    if !redraw[index] {
                        continue;
                    }
                    let mut casting = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("local shadow"),
                        color_attachments: &[],
                        timestamp_writes: self.timer.pass(Span::Shadow),
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &local.maps.layers[caster.layer + face],
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                    casting.set_bind_group(0, &shadows.groups[MAX_CASCADES + index], &[]);
                    casting.set_bind_group(2, self.clusters.empty(), &[]);
                    draw_casters(&mut casting, drawing, casters, ready, 1 + count + index);
                }
            }
        }
        // The camera's depth, its casters drawn through the camera's own
        // projection, jitter and all, so that it lines up with the scene.
        if shadows.contact_id != 0 {
            let contact = shadows.contact.as_ref().expect("a contact id has a texture");
            let jittered = self.post.as_ref().and_then(|post| post.jittered());
            let mut data = Floats::<SCENE_FLOATS>::new();
            data.extend(frame.view).extend(jittered.unwrap_or(frame.projection));
            lighting(&mut data, &frame.ambient, &frame.light_direction, &frame.light_color);
            self.queue.write_buffer(&shadows.uniforms[CONTACT_PASS], 0, data.bytes());
            let mut depth = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("contact depth"),
                color_attachments: &[],
                timestamp_writes: self.timer.pass(Span::Shadow),
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &contact.view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            depth.set_bind_group(0, &shadows.groups[CONTACT_PASS], &[]);
            depth.set_bind_group(2, self.clusters.empty(), &[]);
            draw_casters(&mut depth, drawing, seen_by_camera, ready, 0);
        }
    }
}

/// Draws the casters (indices into `ready`) that `view` sees, depth only.
fn draw_casters(
    pass: &mut wgpu::RenderPass<'_>,
    drawing: &Drawing<'_>,
    casters: &[usize],
    ready: &[ReadyEntry],
    view: usize,
) {
    for &index in casters {
        let entry = &ready[index];
        let item = &drawing.items[entry.item];
        if item.views & (1 << view) == 0 {
            continue;
        }
        let stage = entry.stage(drawing.items) as usize;
        pass.set_pipeline(drawing.shader(entry).shadow[stage].as_ref().unwrap());
        pass.set_bind_group(1, &entry.group, &[entry.offset]);
        drawing.draw(pass, entry, view, None);
    }
}

/// The local lights' map faces, which the lights' bind group binds, and
/// which texture they are.
pub(super) fn local_atlas(shadows: &Option<Shadows>) -> Option<(&wgpu::TextureView, u64)> {
    let local = shadows.as_ref()?.local.as_ref()?;
    Some((&local.maps.all, local.id))
}

#[cfg(test)]
mod template {
    #[test]
    fn the_new_game_template_carries_the_current_shadows_block() {
        let template = include_str!("../../../pen/templates/new/src/assets/lit.wgsl");
        let begin = template.find("// BEGIN shadows\n").expect("a begin marker") + "// BEGIN shadows\n".len();
        let end = template.find("// END shadows").expect("an end marker");
        assert_eq!(&template[begin..end], super::SHADOWS_WGSL, "copy shaders/shadows.wgsl into lit.wgsl");
    }
}
