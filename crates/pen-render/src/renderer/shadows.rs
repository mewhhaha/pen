// The sun's shadows: depth maps of the casters seen from the light, which the
// scene's surfaces compare their depth against. The game's own map, and,
// where a loaded shader reads them, nearer ones fitted inside the camera's
// view (cascades.rs), all layers of one array texture: shaders that read
// `shadow_cascades` (binding 3) take each surface from the nearest map that
// holds it; others read the game's map at binding 1, as before.
use pen_protocol::RenderFrame;

use super::cascades::{MAX_CASCADES, cascades};
use super::timing::Span;
use super::{
    Drawing, Floats, GpuAsset, ReadyEntry, SCENE_BYTES, SCENE_FLOATS, Shadows, State, heavens, lighting, uniform_buffer,
};
use crate::matrix::{IDENTITY, multiply};

// The share of a map's width, at its edge, over which a surface fades into
// the next map.
const BAND: f32 = 0.1;

/// A shadow map array's texture, a view of each layer to draw into, and the
/// view of them all.
pub(super) struct ShadowMaps {
    texture: wgpu::Texture,
    layers: Vec<wgpu::TextureView>,
    all: wgpu::TextureView,
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
/// layer of `maps`), the comparison sampler, and every map.
fn scene_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    maps: &ShadowMaps,
    game: Option<&wgpu::TextureView>,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(game.unwrap_or(&maps.layers[0])),
            },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&maps.all) },
        ],
    })
}

impl State {
    pub(super) fn shadow_resources(&mut self) -> &mut Shadows {
        let device = &self.device;
        let scene_layout = &self.context.scene_group;
        self.shadows.get_or_insert_with(|| {
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                compare: Some(wgpu::CompareFunction::Less),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            let stand_in = depth_array(device, "shadow stand-in", 1, 1);
            // The shadow passes bind the stand-in, as they draw into the maps.
            let (uniforms, groups) = (0..MAX_CASCADES)
                .map(|_| {
                    let uniform = uniform_buffer(device, "shadow scene", SCENE_BYTES);
                    let group = scene_group(device, scene_layout, &uniform, &stand_in, None, &sampler);
                    (uniform, group)
                })
                .unzip();
            Shadows { uniforms, groups, sampler, stand_in, maps: None, map_id: 0 }
        })
    }

    /// A scene bind group for `uniform` that reads no shadow map (an inset's).
    pub(super) fn unshadowed_scene(&mut self, uniform: &wgpu::Buffer) -> wgpu::BindGroup {
        let device = self.device.clone();
        let layout = self.context.scene_group.clone();
        let resources = self.shadow_resources();
        scene_group(&device, &layout, uniform, &resources.stand_in, None, &resources.sampler)
    }

    /// The frame's shadow maps (`cascades`): nearer ones only if a loaded
    /// shader reads them.
    pub(super) fn fit_cascades(&mut self, frame: &RenderFrame) {
        let cascaded = self
            .assets
            .slots
            .iter()
            .any(|slot| matches!(&slot.current, Some(GpuAsset::Shader(shader)) if shader.cascaded));
        self.cascades = cascades(frame, cascaded);
    }

    /// Writes the scene's uniform and, when the sun casts shadows, each
    /// shadow pass's, making the maps if need be; whether it does.
    pub(super) fn light(&mut self, frame: &RenderFrame) -> bool {
        // The game fits its shadow map's view; 0 texels for none.
        let shadow_texels = frame.shadow.texels;
        let sun = (shadow_texels > 0).then_some(&frame.shadow);
        let (maps, count) = self.cascades;
        let mut scene = Floats::<SCENE_FLOATS>::new();
        scene.extend(frame.view).extend(frame.projection);
        lighting(&mut scene, &frame.ambient, &frame.light_direction, &frame.light_color);
        scene.extend(sun.map_or(IDENTITY, |sun| multiply(&sun.projection, &sun.view)));
        let texel = 1.0 / shadow_texels.max(1) as f32;
        scene.extend([if sun.is_some() { 1.0 } else { 0.0 }, texel, sun.map_or(0.0, |sun| sun.texel * 1.5), 0.0]);
        heavens(&mut scene, &frame.sky, &frame.view);
        // The maps: how many and the band they fade over, what each sees,
        // and each one's texel and how far off a surface to read it.
        #[allow(clippy::cast_precision_loss, reason = "at most MAX_CASCADES")]
        scene.extend([count as f32, BAND, 0.0, 0.0]);
        for map in &maps {
            scene.extend(multiply(&map.projection, &frame.shadow.view));
        }
        for map in &maps {
            scene.extend([texel, map.texel * 1.5, 0.0, 0.0]);
        }
        self.queue.write_buffer(&self.scene_uniform, 0, scene.bytes());
        // Each shadow pass sees the scene from the light.
        if let Some(sun) = sun {
            let device = self.device.clone();
            let queue = self.queue.clone();
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
        // The game's map, the last, at binding 1.
        let bound = if sun.is_some() { (self.shadow_resources().map_id, count.max(1) - 1) } else { (0, 0) };
        if self.scene_group.as_ref().is_none_or(|(_, map)| *map != bound) {
            self.shadow_resources();
            let resources = self.shadows.as_ref().unwrap();
            let maps = resources.maps.as_ref().filter(|_| bound.0 != 0);
            let group = scene_group(
                &self.device,
                &self.context.scene_group,
                &self.scene_uniform,
                maps.unwrap_or(&resources.stand_in),
                maps.map(|maps| &maps.layers[bound.1]),
                &resources.sampler,
            );
            self.scene_group = Some((group, bound));
        }
        sun.is_some()
    }

    /// The shadow passes: the casters each map sees, from the light, into it.
    pub(super) fn cast_shadows(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        drawing: &Drawing<'_>,
        casters: &[usize],
        ready: &[ReadyEntry],
    ) {
        let Some(shadows) = &self.shadows else { return };
        let Some(maps) = &shadows.maps else { return };
        let (_, count) = self.cascades;
        for (map, layer) in maps.layers.iter().enumerate().take(count) {
            let mut casting = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                timestamp_writes: self.timer.pass(Span::Shadow),
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: layer,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            casting.set_bind_group(0, &shadows.groups[map], &[]);
            casting.set_bind_group(2, self.clusters.empty(), &[]);
            // The maps are the views after the camera's.
            let view = map + 1;
            for &index in casters {
                let entry = &ready[index];
                let item = &drawing.items[entry.item];
                if item.views & (1 << view) == 0 {
                    continue;
                }
                let stage = entry.stage(drawing.items) as usize;
                casting.set_pipeline(drawing.shader(entry).shadow[stage].as_ref().unwrap());
                casting.set_bind_group(1, &entry.group, &[entry.offset]);
                drawing.draw(&mut casting, entry, view, None);
            }
        }
    }
}
