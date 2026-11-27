//! 2D radiance cascades over a scene in pixel coordinates. Each texel contains
//! linear HDR emissive RGB and opacity A. The resolved texture contains angular
//! mean radiance, suitable for direction-independent diffuse 2D lighting.

use anyhow::{Result, ensure};
use bytemuck::{Pod, Zeroable};

use super::buffer;

pub const RADIANCE_CASCADES_2D_WGSL: &str = crate::wgsl!("render2d_rc2d.wgsl");

/// Spacing doubles and direction count quadruples at each level.
#[derive(Clone, Copy, Debug)]
pub struct RadianceCascades2dConfig {
    pub c0_spacing: u32,
    pub c0_dirs: u32,
    /// None chooses enough cascades to reach the scene diagonal, up to six.
    /// An explicit count in 1..=6 extends its final interval to cover the scene.
    pub cascades: Option<u32>,
}

impl Default for RadianceCascades2dConfig {
    fn default() -> Self {
        Self { c0_spacing: 2, c0_dirs: 4, cascades: None }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    grid: [u32; 2],
    dirs: u32,
    spacing: f32,
    far_grid: [u32; 2],
    far_dirs: u32,
    far_spacing: f32,
    interval: [f32; 2],
    padding: [f32; 2],
}

struct Cascade {
    params: wgpu::Buffer,
    workgroups: [u32; 2],
}

/// A fixed-size GI pass with reusable cascade buffers and an HDR output texture.
/// Record a build before any use of the irradiance for that scene.
pub struct RadianceCascades2d {
    size: [u32; 2],
    cascades: Vec<Cascade>,
    packets: [wgpu::Buffer; 2],
    empty_packet: wgpu::Buffer,
    trace: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    resolve_bg: wgpu::BindGroup,
    composite: wgpu::ComputePipeline,
    composite_uniform: wgpu::Buffer,
    irradiance: wgpu::Texture,
    irradiance_view: wgpu::TextureView,
}

impl RadianceCascades2d {
    pub fn new(device: &wgpu::Device, size: [u32; 2], config: RadianceCascades2dConfig) -> Result<Self> {
        let limits = device.limits();
        ensure!(size.iter().all(|&side| side > 0 && side <= limits.max_texture_dimension_2d), "invalid RC scene size");
        ensure!(config.c0_spacing > 0 && config.c0_spacing <= limits.max_texture_dimension_2d, "invalid RC spacing");
        ensure!((1..=64).contains(&config.c0_dirs), "RC requires 1 to 64 base directions");
        let [width, height] = size.map(|side| side as f32);
        let diagonal = width.hypot(height);
        let automatic = config.cascades.is_none();
        let count = config.cascades.unwrap_or(6);
        ensure!((1..=6).contains(&count), "RC supports 1 to 6 cascades");

        let mut cascades = Vec::with_capacity(count as usize);
        let mut packet_bytes = 0;
        let mut start = 0.0;
        let mut interval_length = 2.0 * config.c0_spacing as f32;
        for k in 0..count {
            let spacing = config.c0_spacing << k;
            let dirs = config.c0_dirs << (2 * k);
            let grid = size.map(|side| side.div_ceil(spacing));
            let mut end = start + interval_length;
            let reach = diagonal + spacing as f32 * std::f32::consts::SQRT_2;
            let last = k + 1 == count || (automatic && end >= reach);
            if last {
                end = end.max(reach);
            }
            let params = Params {
                grid,
                dirs,
                spacing: spacing as f32,
                far_grid: size.map(|side| side.div_ceil(spacing * 2)),
                far_dirs: if last { 0 } else { dirs * 4 },
                far_spacing: (spacing * 2) as f32,
                interval: [start, end],
                padding: [0.0; 2],
            };
            let packet_count = u64::from(grid[0]) * u64::from(grid[1]) * u64::from(dirs);
            let bytes = packet_count * 16;
            ensure!(
                bytes <= limits.max_storage_buffer_binding_size && bytes <= limits.max_buffer_size,
                "RC cascade {k} exceeds the device's storage buffer limit"
            );
            packet_bytes = packet_bytes.max(bytes);
            let groups = u32::try_from(packet_count.div_ceil(64))?;
            let x = groups.min(limits.max_compute_workgroups_per_dimension);
            let y = groups.div_ceil(x);
            ensure!(y <= limits.max_compute_workgroups_per_dimension, "RC dispatch exceeds the device's limits");
            cascades.push(Cascade {
                params: buffer(device, "2D RC interval", bytemuck::bytes_of(&params), wgpu::BufferUsages::UNIFORM),
                workgroups: [x, y],
            });
            if last {
                break;
            }
            start = end;
            interval_length *= 4.0;
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("2D radiance cascades"),
            source: wgpu::ShaderSource::Wgsl(RADIANCE_CASCADES_2D_WGSL.into()),
        });
        let pipeline = |name| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &shader,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: crate::pipeline_cache::active().as_ref(),
            })
        };
        let packet_buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let irradiance = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("2D RC irradiance"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let irradiance_view = irradiance.create_view(&Default::default());
        let packets =
            [packet_buffer("2D RC even cascades", packet_bytes), packet_buffer("2D RC odd cascades", packet_bytes)];
        let resolve = pipeline("resolve_cascade");
        let resolve_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("2D RC resolve bg"),
            layout: &resolve.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cascades[0].params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: packets[0].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&irradiance_view) },
            ],
        });
        let composite_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("2D RC composite weights"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            size,
            cascades,
            packets,
            empty_packet: packet_buffer("2D RC far boundary", 16),
            trace: pipeline("trace_cascade"),
            resolve,
            resolve_bg,
            composite: pipeline("composite_irradiance"),
            composite_uniform,
            irradiance,
            irradiance_view,
        })
    }

    pub fn cascade_count(&self) -> usize {
        self.cascades.len()
    }

    /// The Rgba16Float texture written by build. RGB is irradiance and alpha is one.
    pub fn irradiance(&self) -> &wgpu::Texture {
        &self.irradiance
    }

    pub fn irradiance_view(&self) -> &wgpu::TextureView {
        &self.irradiance_view
    }

    /// Trace from the farthest interval to the nearest, then resolve at full
    /// scene resolution. Scene must be a matching single-sampled Rgba16Float
    /// 2D view with TEXTURE_BINDING usage.
    pub fn build(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, scene: &wgpu::TextureView) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("2D radiance cascades"),
            timestamp_writes: None,
        });
        for (level, cascade) in self.cascades.iter().enumerate().rev() {
            let far =
                if level + 1 == self.cascades.len() { &self.empty_packet } else { &self.packets[(level + 1) % 2] };
            dispatch(
                device,
                &mut pass,
                &self.trace,
                &[
                    (0, cascade.params.as_entire_binding()),
                    (1, wgpu::BindingResource::TextureView(scene)),
                    (2, far.as_entire_binding()),
                    (3, self.packets[level % 2].as_entire_binding()),
                ],
                cascade.workgroups,
            );
        }
        let resolve_groups = self.size.map(|side| side.div_ceil(8));
        pass.set_pipeline(&self.resolve);
        pass.set_bind_group(0, &self.resolve_bg, &[]);
        pass.dispatch_workgroups(resolve_groups[0], resolve_groups[1], 1);
    }

    /// Apply `albedo.rgb * (ambient + irradiance * multiply) + irradiance * add`
    /// and preserve the albedo alpha. Weights are [ambient, multiply, add].
    /// Albedo and output views must match the scene size. Output is a separate
    /// Rgba16Float storage view. For irradiance alone, use irradiance_view directly.
    pub fn composite(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        albedo: &wgpu::TextureView,
        output: &wgpu::TextureView,
        weights: [f32; 3],
    ) {
        self.composite_layer(device, encoder, albedo, output, weights, false);
    }

    pub(crate) fn composite_layer(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        albedo: &wgpu::TextureView,
        output: &wgpu::TextureView,
        weights: [f32; 3],
        premultiplied: bool,
    ) {
        let [ambient, multiply, add] = weights;
        let data = [ambient, multiply, add, f32::from(premultiplied)];
        let stage = buffer(device, "2D RC composite stage", bytemuck::bytes_of(&data), wgpu::BufferUsages::COPY_SRC);
        encoder.copy_buffer_to_buffer(&stage, 0, &self.composite_uniform, 0, 16);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("2D RC composite"),
            timestamp_writes: None,
        });
        dispatch(
            device,
            &mut pass,
            &self.composite,
            &[
                (4, wgpu::BindingResource::TextureView(output)),
                (5, wgpu::BindingResource::TextureView(&self.irradiance_view)),
                (6, wgpu::BindingResource::TextureView(albedo)),
                (7, self.composite_uniform.as_entire_binding()),
            ],
            self.size.map(|side| side.div_ceil(8)),
        );
    }
}

fn dispatch(
    device: &wgpu::Device,
    pass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    resources: &[(u32, wgpu::BindingResource<'_>)],
    workgroups: [u32; 2],
) {
    let entries: Vec<_> =
        resources.iter().cloned().map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource }).collect();
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(workgroups[0], workgroups[1], 1);
}

/// Axis-aligned scene primitives in texture pixel coordinates. The material is
/// [linear HDR emissive R/G/B, opacity]. Use [0, 0, 0, 1] for an opaque wall.
#[derive(Clone, Copy, Debug)]
pub enum Scene2dEmitter {
    Circle { center: [f32; 2], radius: f32, material: [f32; 4] },
    Rect { rect: [f32; 4], material: [f32; 4] },
}

/// Rasterize hard-edged primitives at texel centers into row-major RGBA f32
/// pixels, with later primitives composited over earlier ones. Upload as f16
/// for use as Rgba16Float; the empty scene is transparent black.
pub fn rasterize_scene_2d(size: [u32; 2], emitters: &[Scene2dEmitter]) -> Vec<[f32; 4]> {
    let width = size[0] as usize;
    let mut pixels = vec![[0.0; 4]; width * size[1] as usize];
    // Count bounding-box visits, including pixels rejected by the circle test.
    let mut remaining = (width * size[1] as usize).saturating_mul(4).clamp(1_048_576, 33_554_432);
    for &emitter in emitters {
        let ([left, top, right, bottom], material) = match emitter {
            Scene2dEmitter::Circle { center: [x, y], radius, material } => {
                ([x - radius, y - radius, x + radius, y + radius], material)
            }
            Scene2dEmitter::Rect { rect: [x, y, w, h], material } => ([x, y, x + w, y + h], material),
        };
        let x_range = pixel_bound(left, size[0])..pixel_bound(right, size[0]);
        let y_range = pixel_bound(top, size[1])..pixel_bound(bottom, size[1]);
        let area = x_range.len() * y_range.len();
        if area > remaining {
            continue;
        }
        remaining -= area;
        let opacity = material[3].clamp(0.0, 1.0);
        let transmittance = 1.0 - opacity;
        for y in y_range {
            for x in x_range.clone() {
                if let Scene2dEmitter::Circle { center, radius, .. } = emitter {
                    let dx = x as f32 + 0.5 - center[0];
                    let dy = y as f32 + 0.5 - center[1];
                    if dx * dx + dy * dy > radius * radius {
                        continue;
                    }
                }
                let pixel = &mut pixels[y * width + x];
                for channel in 0..3 {
                    pixel[channel] = material[channel] + pixel[channel] * transmittance;
                }
                pixel[3] = opacity + pixel[3] * transmittance;
            }
        }
    }
    pixels
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "quantizing clamped pixel bounds")]
fn pixel_bound(edge: f32, limit: u32) -> usize {
    ((edge - 0.5).ceil().clamp(0.0, limit as f32) as usize).min(limit as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_emitters_share_one_clipped_raster_budget() {
        let covering = Scene2dEmitter::Rect { rect: [-100.0, -100.0, 2000.0, 2000.0], material: [1.0, 0.0, 0.0, 0.0] };
        let image = rasterize_scene_2d([512, 512], &[covering; 64]);
        assert_eq!(image[0], [4.0, 0.0, 0.0, 0.0]);
        assert_eq!(image[image.len() - 1], image[0]);
    }
}
