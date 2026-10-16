// The sun's shadows: a depth map of the casters seen from the light, which
// the scene's surfaces compare their depth against.
use pen_protocol::RenderFrame;

use super::{
    Drawing, Floats, ReadyEntry, SCENE_BYTES, Shadows, Shape, State, extent, heavens, lighting, uniform_buffer,
};
use crate::matrix::{IDENTITY, multiply};

impl State {
    pub(super) fn shadow_resources(&mut self) -> &mut Shadows {
        let device = &self.device;
        let scene_layout = &self.context.scene_group;
        self.shadows.get_or_insert_with(|| {
            let uniform = uniform_buffer(device, "shadow scene", SCENE_BYTES);
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                compare: Some(wgpu::CompareFunction::Less),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            let stand_in = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("shadow stand-in"),
                size: extent(1, 1),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let stand_in = stand_in.create_view(&Default::default());
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow scene"),
                layout: scene_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&stand_in) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                ],
            });
            Shadows { uniform, sampler, stand_in, group, map: None, map_id: 0 }
        })
    }

    /// Writes the scene's uniform and, when the sun casts shadows, the
    /// shadow pass's, making the map if need be; whether it does.
    pub(super) fn light(&mut self, frame: &RenderFrame) -> bool {
        // The game fits the shadow map's view; 0 texels for none.
        let shadow_texels = frame.shadow.texels;
        let sun = (shadow_texels > 0).then_some(&frame.shadow);
        let mut scene = Floats::<80>::new();
        scene.extend(frame.view).extend(frame.projection);
        lighting(&mut scene, &frame.ambient, &frame.light_direction, &frame.light_color);
        scene.extend(sun.map_or(IDENTITY, |sun| multiply(&sun.projection, &sun.view)));
        scene.extend([
            if sun.is_some() { 1.0 } else { 0.0 },
            1.0 / shadow_texels.max(1) as f32,
            sun.map_or(0.0, |sun| sun.texel * 1.5),
            0.0,
        ]);
        heavens(&mut scene, &frame.sky, &frame.view);
        self.queue.write_buffer(&self.scene_uniform, 0, scene.bytes());
        // The shadow pass sees the scene from the light.
        if let Some(sun) = sun {
            let device = self.device.clone();
            let queue = self.queue.clone();
            let resources = self.shadow_resources();
            if resources.map.as_ref().is_none_or(|(map, _)| map.width() != shadow_texels) {
                let map = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("shadow map"),
                    size: extent(shadow_texels, shadow_texels),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = map.create_view(&Default::default());
                resources.map = Some((map, view));
                resources.map_id += 1;
            }
            let mut data = Floats::<80>::new();
            data.extend(sun.view).extend(sun.projection);
            lighting(&mut data, &frame.ambient, &frame.light_direction, &frame.light_color);
            queue.write_buffer(&resources.uniform, 0, data.bytes());
        }
        let bound_map = if sun.is_some() { self.shadow_resources().map_id } else { 0 };
        if self.scene_group.as_ref().is_none_or(|(_, map)| *map != bound_map) {
            self.shadow_resources();
            let resources = self.shadows.as_ref().unwrap();
            let view = match (bound_map, &resources.map) {
                (0, _) | (_, None) => &resources.stand_in,
                (_, Some((_, view))) => view,
            };
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene"),
                layout: &self.context.scene_group,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.scene_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&resources.sampler) },
                ],
            });
            self.scene_group = Some((group, bound_map));
        }
        sun.is_some()
    }

    /// The shadow pass: the casters seen from the light, into the map.
    pub(super) fn cast_shadows(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        drawing: &Drawing<'_>,
        casters: &[usize],
        ready: &[ReadyEntry],
    ) {
        let Some(shadows) = &self.shadows else { return };
        let Some((_, map)) = &shadows.map else { return };
        let mut casting = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: map,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        casting.set_bind_group(0, &shadows.group, &[]);
        for &index in casters {
            let entry = &ready[index];
            let instanced = matches!(drawing.items[entry.item].shape, Shape::Batch(_));
            casting.set_pipeline(drawing.shader(entry).shadow[instanced as usize].as_ref().unwrap());
            casting.set_bind_group(1, &entry.group, &[entry.offset]);
            drawing.draw(&mut casting, entry, None);
        }
    }
}
