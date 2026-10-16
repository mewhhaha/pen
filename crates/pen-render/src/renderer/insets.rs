// Insets: a few draws seen by their own camera, lit by their own light
// without shadows, drawn supersampled and multisampled into targets of their
// own and laid over the output in their rectangle, toned as the scene is.
// Smooth insets use twice the display resolution; pixelated ones the
// requested block size and nearest sampling.
use pen_protocol::RenderFrame;

use super::{
    Drawing, Floats, INSET_SAMPLES, InsetScene, InsetTargets, ReadyEntry, SCENE_BYTES, State, extent, heavens,
    lighting, overlay_pipeline, uniform_buffer,
};
use crate::RenderStats;
use crate::built_in::BuiltIn;
use crate::matrix::IDENTITY;
use crate::post::SCENE_FORMAT;
use crate::shader::PipelineContext;

// Lays an inset's resolved image over the screen in its rectangle, toned as
// the scene is; its color is premultiplied by its coverage, so edges blend
// smoothly.
const COMPOSITE_SHADER: &str = include_str!("../shaders/composite.wgsl");

/// The pipeline laying an inset's image over the output.
pub(crate) fn composite_pipeline(context: &PipelineContext) -> wgpu::RenderPipeline {
    overlay_pipeline(context, COMPOSITE_SHADER, "inset composite")
}

impl State {
    /// The inset's targets, made anew when its size or look changes.
    pub(super) fn targets_for(&mut self, index: usize, width: u32, height: u32, pixelated: bool) -> &InsetTargets {
        if self.inset_targets.len() <= index {
            self.inset_targets.resize_with(index + 1, || None);
        }
        let limit = self.device.limits().max_texture_dimension_2d;
        // A quarter more than `size`, rounded up to 256.
        let roomy = |size: u32| limit.min(size.saturating_add(size.div_ceil(4)).div_ceil(256).saturating_mul(256));
        // Kept while they hold the inset and are not four times what it would
        // be given afresh.
        let found = self.inset_targets[index].as_ref().filter(|found| found.pixelated == pixelated);
        let fits = found.is_some_and(|found| {
            found.width >= width
                && found.height >= height
                && u64::from(found.width) * u64::from(found.height)
                    <= 4 * u64::from(roomy(width)) * u64::from(roomy(height))
        });
        if !fits {
            // The first targets fit the inset as it is: most keep their size,
            // and a full-screen one's would take hundreds of megabytes more.
            // One that outgrew them has room to grow.
            let (width, height) = if found.is_some() { (roomy(width), roomy(height)) } else { (width, height) };
            let composite = self.built_ins.get(BuiltIn::Composite).clone();
            let device = &self.device;
            let texture = |samples: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("inset"),
                        size: extent(width, height),
                        mip_level_count: 1,
                        sample_count: samples,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage,
                        view_formats: &[],
                    })
                    .create_view(&Default::default())
            };
            let resolved =
                texture(1, SCENE_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
            let placement = uniform_buffer(device, "inset placement", 32);
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("inset"),
                layout: &composite.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: placement.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&resolved) },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(if pixelated {
                            &self.inset_nearest_sampler
                        } else {
                            &self.inset_sampler
                        }),
                    },
                ],
            });
            self.inset_targets[index] = Some(InsetTargets {
                width,
                height,
                pixelated,
                color: texture(INSET_SAMPLES, SCENE_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT),
                depth: texture(
                    INSET_SAMPLES,
                    wgpu::TextureFormat::Depth24PlusStencil8,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                ),
                resolved,
                placement,
                group,
            });
        }
        self.inset_targets[index].as_ref().unwrap()
    }

    /// Each inset's targets and scene, ready to draw into, and how many
    /// pixels across and down each draws, into `insets`.
    pub(super) fn prepare_insets(
        &mut self,
        frame: &RenderFrame,
        (output_width, output_height): (u32, u32),
        insets: &mut Vec<(usize, (u32, u32))>,
    ) {
        for (index, inset) in frame.insets.iter().enumerate() {
            let (width, height) = (inset.width, inset.height);
            if width <= 0.0
                || height <= 0.0
                || inset.x >= output_width as f32
                || inset.y >= output_height as f32
                || inset.x + width <= 0.0
                || inset.y + height <= 0.0
            {
                continue;
            }
            let pixelated = inset.pixel_size > 0.0;
            let (width64, height64) = (f64::from(width), f64::from(height));
            let resolution = (if pixelated { 1.0 / f64::from(inset.pixel_size).max(1.0) } else { 2.0 })
                .min(4096.0 / width64)
                .min(4096.0 / height64)
                .min((3840.0 * 2160.0 / (width64 * height64)).sqrt());
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "at most 4096 by `resolution`")]
            let size = |value: f64| ((value * resolution).ceil() as u32).max(1);
            let drawn = (size(width64), size(height64));
            self.targets_for(index, drawn.0, drawn.1, pixelated);
            if self.inset_scenes.len() <= index {
                self.inset_scenes.resize_with(index + 1, || None);
            }
            if self.inset_scenes[index].is_none() {
                let uniform = uniform_buffer(&self.device, "inset scene", SCENE_BYTES);
                let device = self.device.clone();
                let scene_layout = self.context.scene_group.clone();
                let resources = self.shadow_resources();
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("inset scene"),
                    layout: &scene_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&resources.stand_in),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&resources.sampler),
                        },
                    ],
                });
                self.inset_scenes[index] = Some(InsetScene { uniform, group });
            }
            let mut data = Floats::<80>::new();
            data.extend(inset.view).extend(inset.projection);
            lighting(&mut data, &inset.ambient, &inset.light_direction, &inset.light_color);
            data.extend(IDENTITY).extend([0.0; 4]);
            heavens(&mut data, &frame.sky, &inset.view);
            let scene = self.inset_scenes[index].as_ref().unwrap();
            self.queue.write_buffer(&scene.uniform, 0, data.bytes());
            let targets = self.inset_targets[index].as_ref().unwrap();
            self.queue.write_buffer(
                &targets.placement,
                0,
                bytemuck::cast_slice(&[
                    inset.x,
                    inset.y,
                    width,
                    height,
                    output_width as f32,
                    output_height as f32,
                    drawn.0 as f32 / targets.width as f32,
                    drawn.1 as f32 / targets.height as f32,
                ]),
            );
            insets.push((index, drawn));
        }
    }

    /// Draws each inset (supersampled and multisampled, lit by its own light
    /// without shadows, seen by its own camera) and lays it over `output` in
    /// its rectangle.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_insets(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        drawing: &Drawing<'_>,
        insets: &[(usize, (u32, u32))],
        ready: &[ReadyEntry],
        composite: &wgpu::RenderPipeline,
        stats: &mut RenderStats,
    ) {
        for &(index, drawn) in insets {
            let targets = self.inset_targets[index].as_ref().unwrap();
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("inset"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &targets.color,
                        depth_slice: None,
                        resolve_target: Some(&targets.resolved),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Discard,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &targets.depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0),
                            store: wgpu::StoreOp::Discard,
                        }),
                    }),
                    ..Default::default()
                });
                pass.set_viewport(0.0, 0.0, drawn.0 as f32, drawn.1 as f32, 0.0, 1.0);
                pass.set_scissor_rect(0, 0, drawn.0, drawn.1);
                pass.set_bind_group(0, &self.inset_scenes[index].as_ref().unwrap().group, &[]);
                for entry in ready {
                    if drawing.items[entry.item].inset() != Some(index) {
                        continue;
                    }
                    pass.set_pipeline(&entry.pipeline);
                    pass.set_bind_group(1, &entry.group, &[entry.offset]);
                    drawing.draw(&mut pass, entry, Some(stats));
                }
            }
            let mut laid = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("inset composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            laid.set_pipeline(composite);
            laid.set_bind_group(0, &targets.group, &[]);
            laid.draw(0..6, 0..1);
        }
    }
}
