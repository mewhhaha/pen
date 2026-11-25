//! GPU ReSTIR direct and diffuse indirect lighting from Pen's thin G-buffer.

use std::collections::HashMap;

use anyhow::{Result, ensure};
use wgpu::util::DeviceExt;

#[cfg(not(target_arch = "wasm32"))]
use crate::ray_tracing::TLAS_TRAVERSAL_HW_WGSL;
use crate::ray_tracing::{RtLocalLight, TLAS_TRAVERSAL_WGSL, TlasSceneGpu, device_supports_hardware_ray_query};

pub const RESTIR_WGSL: &str = include_str!("shaders/restir.wgsl");
pub const RESTIR_MAX_LIGHTS: usize = 256;

const OUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DI_STRIDE: u64 = 32;
const GI_STRIDE: u64 = 64;
const MIN_RAY_DISTANCE: f32 = 0.001;

#[derive(Clone, Copy, Debug)]
pub struct RestirParams {
    /// Current and previous column-major WebGPU view-projection matrices.
    pub view_projection: [f32; 16],
    pub prev_view_projection: [f32; 16],
    pub camera_pos: [f32; 3],
    pub depth_clear: f32,
    /// World-space direction toward the sun.
    pub sun_direction: [f32; 3],
    pub sun_color: [f32; 3],
    pub sky_zenith: [f32; 3],
    pub sky_horizon: [f32; 3],
    pub max_ray_distance: f32,
    pub normal_bias: f32,
    pub visibility_mask: u32,
    pub frame_index: u32,
    pub enable_di: bool,
    pub enable_gi: bool,
    pub reset_history: bool,
    /// Strength of optional DDGI irradiance at secondary hits.
    pub gi_bounce_scale: f32,
}

pub struct RestirInputs<'a> {
    pub depth: &'a wgpu::TextureView,
    pub normal_roughness: &'a wgpu::TextureView,
    pub albedo_metallic: &'a wgpu::TextureView,
    /// Current UV minus previous UV. Missing motion or abs(x) >= 100 uses matrix reprojection.
    pub motion: Option<&'a wgpu::TextureView>,
    /// Packed cascades from DDGI, with STORAGE usage.
    pub ddgi_packed: Option<&'a wgpu::Buffer>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    inverse_view_projection: [f32; 16],
    prev_view_projection: [f32; 16],
    camera_and_clear: [f32; 4],
    previous_camera: [f32; 4],
    sun_and_max_distance: [f32; 4],
    sun_color_and_bias: [f32; 4],
    sky_zenith_and_bounce: [f32; 4],
    sky_horizon: [f32; 4],
    scene_counts_and_mask: [u32; 4],
    size_lights_frame: [u32; 4],
    flags: [u32; 4],
}

struct Image {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Image {
    fn new(device: &wgpu::Device, size: (u32, u32), label: &str, extra_usage: wgpu::TextureUsages) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OUT_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | extra_usage,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        Self { texture, view }
    }
}

struct Reservoirs {
    scratch: wgpu::Buffer,
    history: [wgpu::Buffer; 2],
}

impl Reservoirs {
    fn new(device: &wgpu::Device, bytes: u64, labels: [&str; 3]) -> Self {
        let make = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        Self { scratch: make(labels[0]), history: [make(labels[1]), make(labels[2])] }
    }
}

struct Frame {
    size: (u32, u32),
    di: Image,
    gi: Image,
    depth_nrm: [Image; 2],
    di_reservoirs: Option<Reservoirs>,
    gi_reservoirs: Option<Reservoirs>,
    previous: usize,
    previous_camera: [f32; 3],
}

impl Frame {
    fn new(device: &wgpu::Device, size: (u32, u32)) -> Self {
        let out = |label| Image::new(device, size, label, wgpu::TextureUsages::COPY_SRC);
        let history = |label| Image::new(device, size, label, wgpu::TextureUsages::empty());
        Self {
            size,
            di: out("restir out_di"),
            gi: out("restir out_gi"),
            depth_nrm: [history("restir geometry A"), history("restir geometry B")],
            di_reservoirs: None,
            gi_reservoirs: None,
            previous: 0,
            previous_camera: [0.0; 3],
        }
    }
}

struct Pass {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    entry: &'static str,
    bindings: &'static [u32],
}

impl Pass {
    fn new(
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        entry: &'static str,
        bindings: &'static [u32],
        hardware: bool,
        native_depth: bool,
    ) -> Self {
        let buffer = |ty| wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None };
        let storage = |read_only| buffer(wgpu::BufferBindingType::Storage { read_only });
        let entries: Vec<_> = bindings
            .iter()
            .map(|&binding| {
                let ty = match binding {
                    0 => buffer(wgpu::BufferBindingType::Uniform),
                    #[cfg(not(target_arch = "wasm32"))]
                    1 if hardware => wgpu::BindingType::AccelerationStructure { vertex_return: false },
                    1..=4 | 9..=10 | 13..=14 | 17..=18 => storage(true),
                    15 | 19 => storage(false),
                    5..=8 | 11 => wgpu::BindingType::Texture {
                        sample_type: if binding == 5 && native_depth {
                            wgpu::TextureSampleType::Depth
                        } else {
                            wgpu::TextureSampleType::Float { filterable: false }
                        },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    12 | 16 | 20 => wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: OUT_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    _ => unreachable!("unknown ReSTIR binding {binding}"),
                };
                wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None }
            })
            .collect();
        #[cfg(target_arch = "wasm32")]
        let _ = hardware;
        let layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(entry), entries: &entries });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { layout, pipeline, entry, bindings }
    }

    fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &[Option<wgpu::BindingResource<'_>>; 21],
        size: (u32, u32),
    ) {
        let entries: Vec<_> = self
            .bindings
            .iter()
            .map(|&binding| wgpu::BindGroupEntry { binding, resource: resources[binding as usize].clone().unwrap() })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(self.entry),
            layout: &self.layout,
            entries: &entries,
        });
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(self.entry), timestamp_writes: None });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
    }
}

struct Pipelines {
    module: wgpu::ShaderModule,
    di: Option<[Pass; 3]>,
    gi: Option<[Pass; 3]>,
    geometry: Pass,
}

impl Pipelines {
    fn new(device: &wgpu::Device, hardware: bool, native_depth: bool) -> Self {
        let (traversal, declaration) =
            (TLAS_TRAVERSAL_WGSL, "@group(0) @binding(1) var<storage, read> tlas_nodes: array<BvhNode>;");
        #[cfg(not(target_arch = "wasm32"))]
        let (traversal, declaration) = if hardware {
            (TLAS_TRAVERSAL_HW_WGSL, "@group(0) @binding(1) var hw_tlas: acceleration_structure;")
        } else {
            (traversal, declaration)
        };
        let mut body = RESTIR_WGSL.replace("// __TLAS_BINDING_1__", declaration);
        if native_depth {
            body = body
                .replace("var depth_tex: texture_2d<f32>;", "var depth_tex: texture_depth_2d;")
                .replace("textureLoad(depth_tex, pixel, 0).x", "textureLoad(depth_tex, pixel, 0)");
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("restir"),
            source: wgpu::ShaderSource::Wgsl(format!("{traversal}\n{body}").into()),
        });
        let geometry =
            Pass::new(device, &module, "restir_store_geometry", &[0, 5, 6, 12, 16, 20], hardware, native_depth);
        Self { module, di: None, gi: None, geometry }
    }

    fn prepare(&mut self, device: &wgpu::Device, hardware: bool, native_depth: bool, di: bool, gi: bool) {
        let make = |entry, bindings| Pass::new(device, &self.module, entry, bindings, hardware, native_depth);
        if di && self.di.is_none() {
            self.di = Some([
                make("restir_di_initial", &[0, 1, 2, 3, 4, 5, 6, 7, 9, 15]),
                make("restir_di_temporal", &[0, 5, 6, 7, 8, 9, 11, 13, 14, 15]),
                make("restir_di_spatial", &[0, 1, 2, 3, 4, 5, 6, 7, 9, 13, 15, 16]),
            ]);
        }
        if gi && self.gi.is_none() {
            self.gi = Some([
                make("restir_gi_initial", &[0, 1, 2, 3, 4, 5, 6, 9, 10, 19]),
                make("restir_gi_temporal", &[0, 5, 6, 8, 11, 17, 18, 19]),
                make("restir_gi_spatial", &[0, 1, 2, 3, 4, 5, 6, 17, 20]),
            ]);
        }
    }
}

pub struct RestirRenderer {
    pipelines: HashMap<(bool, bool), Pipelines>,
    hardware_supported: bool,
    dummy: wgpu::Buffer,
    frame: Option<Frame>,
    di_history_light_count: Option<u32>,
    gi_history_valid: bool,
}

impl RestirRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            pipelines: HashMap::new(),
            hardware_supported: device_supports_hardware_ray_query(device),
            dummy: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("restir empty DDGI and local lights"),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            frame: None,
            di_history_light_count: None,
            gi_history_valid: false,
        }
    }

    /// Whether native ray queries are enabled on the device. Snapshots without a hardware TLAS use compute traversal.
    pub fn uses_hardware_ray_query(&self) -> bool {
        self.hardware_supported
    }

    pub fn reset_history(&mut self) {
        self.di_history_light_count = None;
        self.gi_history_valid = false;
    }

    /// Linear direct lighting with the primary surface BRDF applied.
    pub fn di_view(&self) -> Option<&wgpu::TextureView> {
        self.frame.as_ref().map(|frame| &frame.di.view)
    }

    pub fn di_texture(&self) -> Option<&wgpu::Texture> {
        self.frame.as_ref().map(|frame| &frame.di.texture)
    }

    /// Linear diffuse bounce lighting, without the primary surface albedo or metallic.
    pub fn gi_view(&self) -> Option<&wgpu::TextureView> {
        self.frame.as_ref().map(|frame| &frame.gi.view)
    }

    pub fn gi_texture(&self) -> Option<&wgpu::Texture> {
        self.frame.as_ref().map(|frame| &frame.gi.texture)
    }

    /// Encode one frame. Submit frames in dispatch order. Reset history on camera cuts,
    /// light-slot reassignment, or discontinuous changes to GI lighting or scene geometry.
    /// Inputs are single-sampled, full-size 2D views of mip level zero.
    #[allow(clippy::too_many_arguments, reason = "caller supplies frame inputs and command encoder")]
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene: &TlasSceneGpu,
        inputs: RestirInputs<'_>,
        size: (u32, u32),
        params: &RestirParams,
        local_lights: &[RtLocalLight],
    ) -> Result<()> {
        let limits = device.limits();
        ensure!(
            size.0 > 0 && size.1 > 0 && size.0.max(size.1) <= limits.max_texture_dimension_2d,
            "invalid ReSTIR extent {size:?}"
        );
        ensure!(local_lights.len() <= RESTIR_MAX_LIGHTS, "ReSTIR supports at most {RESTIR_MAX_LIGHTS} local lights");
        let count = u32::try_from(local_lights.len())?;
        let depth_format = inputs.depth.texture().format();
        let native_depth = depth_format == wgpu::TextureFormat::Depth32Float;
        ensure!(native_depth || depth_format == wgpu::TextureFormat::R32Float, "unsupported ReSTIR depth format");
        validate_input(inputs.depth, size, depth_format)?;
        validate_input(inputs.normal_roughness, size, OUT_FORMAT)?;
        validate_input(inputs.albedo_metallic, size, OUT_FORMAT)?;
        if let Some(motion) = inputs.motion {
            validate_input(motion, size, OUT_FORMAT)?;
        }
        if let Some(ddgi) = inputs.ddgi_packed {
            ensure!(
                ddgi.size() >= 16 && ddgi.usage().contains(wgpu::BufferUsages::STORAGE),
                "ReSTIR DDGI must bind a packed storage buffer"
            );
        }
        ensure!(
            params.max_ray_distance.is_finite() && params.max_ray_distance >= MIN_RAY_DISTANCE,
            "invalid ReSTIR ray distance"
        );
        ensure!(params.normal_bias.is_finite() && params.normal_bias >= 0.0, "invalid ReSTIR normal bias");
        let pixels = u64::from(size.0) * u64::from(size.1);
        for (enabled, stride) in [(params.enable_di, DI_STRIDE), (params.enable_gi, GI_STRIDE)] {
            if enabled {
                let bytes = pixels * stride;
                ensure!(
                    bytes <= limits.max_storage_buffer_binding_size && bytes <= limits.max_buffer_size,
                    "ReSTIR reservoir buffer of {bytes} bytes exceeds device limits"
                );
            }
        }
        let hardware = self.hardware_supported && scene.has_hw_tlas();
        let storage_needed = if params.enable_di || params.enable_gi { if hardware { 6 } else { 7 } } else { 0 };
        ensure!(
            limits.max_storage_buffers_per_shader_stage >= storage_needed,
            "ReSTIR requires {storage_needed} storage buffers per compute stage"
        );

        if params.reset_history || self.frame.as_ref().is_none_or(|frame| frame.size != size) {
            self.reset_history();
        }
        let frame = self.frame.get_or_insert_with(|| Frame::new(device, size));
        if frame.size != size {
            *frame = Frame::new(device, size);
        }
        let sun = params.sun_direction;
        let sun_length = sun.iter().map(|v| v * v).sum::<f32>().sqrt();
        let sun = if sun_length.is_finite() && sun_length > 0.0 { sun.map(|v| v / sun_length) } else { [0.0; 3] };
        let pack = |[x, y, z]: [f32; 3], w| [x, y, z, w];
        let uniform = Uniform {
            inverse_view_projection: crate::matrix::invert(&params.view_projection),
            prev_view_projection: params.prev_view_projection,
            camera_and_clear: pack(params.camera_pos, params.depth_clear),
            previous_camera: pack(frame.previous_camera, 0.0),
            sun_and_max_distance: pack(sun, params.max_ray_distance),
            sun_color_and_bias: pack(params.sun_color, params.normal_bias),
            sky_zenith_and_bounce: pack(params.sky_zenith, params.gi_bounce_scale.max(0.0)),
            sky_horizon: pack(params.sky_horizon, 0.0),
            scene_counts_and_mask: [
                scene.tlas_node_count,
                scene.tlas_instance_count,
                scene.blas_tri_count,
                params.visibility_mask,
            ],
            size_lights_frame: [size.0, size.1, count, params.frame_index],
            flags: [
                u32::from(self.di_history_light_count == Some(count)),
                u32::from(self.gi_history_valid),
                u32::from(inputs.motion.is_some()),
                u32::from(params.enable_di) | (u32::from(params.enable_gi) << 1),
            ],
        };
        // Immutable per-dispatch uploads preserve ordering even when several frames
        // are recorded before submission.
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("restir frame uniforms"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let lights = (!local_lights.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("restir local lights"),
                contents: bytemuck::cast_slice(local_lights),
                usage: wgpu::BufferUsages::STORAGE,
            })
        });

        let pipelines = self
            .pipelines
            .entry((hardware, native_depth))
            .or_insert_with(|| Pipelines::new(device, hardware, native_depth));
        pipelines.prepare(device, hardware, native_depth, params.enable_di, params.enable_gi);
        let prev = frame.previous;
        let next = 1 - prev;
        let view = |view| Some(wgpu::BindingResource::TextureView(view));
        let tlas = scene.tlas_nodes_buf.as_entire_binding();
        #[cfg(not(target_arch = "wasm32"))]
        let tlas = if hardware { scene.hw_tlas.as_ref().unwrap().as_binding() } else { tlas };
        let mut resources = [
            Some(buffer_binding(&uniform)),
            Some(tlas),
            Some(buffer_binding(&scene.tlas_instances_buf)),
            Some(buffer_binding(&scene.blas_nodes_buf)),
            Some(buffer_binding(&scene.blas_tris_buf)),
            view(inputs.depth),
            view(inputs.normal_roughness),
            view(inputs.albedo_metallic),
            view(inputs.motion.unwrap_or(inputs.normal_roughness)),
            Some(buffer_binding(lights.as_ref().unwrap_or(&self.dummy))),
            Some(buffer_binding(inputs.ddgi_packed.unwrap_or(&self.dummy))),
            view(&frame.depth_nrm[prev].view),
            view(&frame.depth_nrm[next].view),
            None,
            None,
            None,
            view(&frame.di.view),
            None,
            None,
            None,
            view(&frame.gi.view),
        ];

        if params.enable_di {
            let buffers = frame.di_reservoirs.get_or_insert_with(|| {
                Reservoirs::new(
                    device,
                    pixels * DI_STRIDE,
                    ["restir DI scratch", "restir DI history A", "restir DI history B"],
                )
            });
            let [initial, temporal, spatial] = pipelines.di.as_ref().unwrap();
            resources[15] = Some(buffer_binding(&buffers.scratch));
            initial.dispatch(device, encoder, &resources, size);
            resources[13] = Some(buffer_binding(&buffers.scratch));
            resources[14] = Some(buffer_binding(&buffers.history[prev]));
            resources[15] = Some(buffer_binding(&buffers.history[next]));
            temporal.dispatch(device, encoder, &resources, size);
            resources[13] = Some(buffer_binding(&buffers.history[next]));
            // The initial reservoir is dead here. Reuse its allocation for the spatial result.
            resources[15] = Some(buffer_binding(&buffers.scratch));
            spatial.dispatch(device, encoder, &resources, size);
        }
        if params.enable_gi {
            let buffers = frame.gi_reservoirs.get_or_insert_with(|| {
                Reservoirs::new(
                    device,
                    pixels * GI_STRIDE,
                    ["restir GI scratch", "restir GI history A", "restir GI history B"],
                )
            });
            let [initial, temporal, spatial] = pipelines.gi.as_ref().unwrap();
            resources[19] = Some(buffer_binding(&buffers.scratch));
            initial.dispatch(device, encoder, &resources, size);
            resources[17] = Some(buffer_binding(&buffers.scratch));
            resources[18] = Some(buffer_binding(&buffers.history[prev]));
            resources[19] = Some(buffer_binding(&buffers.history[next]));
            temporal.dispatch(device, encoder, &resources, size);
            resources[17] = Some(buffer_binding(&buffers.history[next]));
            spatial.dispatch(device, encoder, &resources, size);
        }
        pipelines.geometry.dispatch(device, encoder, &resources, size);
        frame.previous = next;
        frame.previous_camera = params.camera_pos;
        self.di_history_light_count = params.enable_di.then_some(count);
        self.gi_history_valid = params.enable_gi;
        Ok(())
    }
}

fn validate_input(view: &wgpu::TextureView, size: (u32, u32), format: wgpu::TextureFormat) -> Result<()> {
    let texture = view.texture();
    ensure!(
        texture.format() == format
            && texture.dimension() == wgpu::TextureDimension::D2
            && texture.sample_count() == 1
            && (texture.width(), texture.height()) == size
            && texture.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING),
        "ReSTIR input must be a single-sampled {format:?} texture of size {size:?} with TEXTURE_BINDING usage"
    );
    Ok(())
}

fn buffer_binding(buffer: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    buffer.as_entire_binding()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::IDENTITY;
    use crate::ray_tracing::{Blas, BlasTriangle, GpuBlasPool, TlasInstance, TlasScene};
    use crate::testing;

    const SIZE: (u32, u32) = (64, 32);
    // x covers [-2, 2], z covers [-0.5, 0.5]. The floor at y = 0 has clip depth 0.5.
    const FLOOR_VP: [f32; 16] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0];

    fn params() -> RestirParams {
        RestirParams {
            view_projection: FLOOR_VP,
            prev_view_projection: FLOOR_VP,
            camera_pos: [0.0, 12.0, 0.0],
            depth_clear: 0.0,
            sun_direction: [0.0, 1.0, 0.0],
            sun_color: [0.0; 3],
            sky_zenith: [0.0; 3],
            sky_horizon: [0.0; 3],
            max_ray_distance: 20.0,
            normal_bias: 0.001,
            visibility_mask: u32::MAX,
            frame_index: 0,
            enable_di: false,
            enable_gi: false,
            reset_history: false,
            gi_bounce_scale: 1.0,
        }
    }

    struct GBuffer {
        depth: wgpu::TextureView,
        normal: wgpu::TextureView,
        albedo: wgpu::TextureView,
        motion: wgpu::TextureView,
    }

    impl GBuffer {
        fn new(device: &wgpu::Device, queue: &wgpu::Queue, native_depth: bool) -> Self {
            let count = (SIZE.0 * SIZE.1) as usize;
            let desc = wgpu::TextureDescriptor {
                label: Some("ReSTIR witness depth"),
                size: wgpu::Extent3d { width: SIZE.0, height: SIZE.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: if native_depth { wgpu::TextureFormat::Depth32Float } else { wgpu::TextureFormat::R32Float },
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | if native_depth { wgpu::TextureUsages::RENDER_ATTACHMENT } else { wgpu::TextureUsages::COPY_DST },
                view_formats: &[],
            };
            let depth = if native_depth {
                let texture = device.create_texture(&desc);
                let view = texture.create_view(&Default::default());
                let mut encoder = device.create_command_encoder(&Default::default());
                let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.5),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                queue.submit([encoder.finish()]);
                view
            } else {
                device
                    .create_texture_with_data(
                        queue,
                        &desc,
                        wgpu::util::TextureDataOrder::LayerMajor,
                        bytemuck::cast_slice(&vec![0.5f32; count]),
                    )
                    .create_view(&Default::default())
            };
            let picture = |pixel| {
                testing::picture(device, queue, SIZE.0, SIZE.1, &vec![pixel; count]).create_view(&Default::default())
            };
            let motion: Vec<_> =
                (0..count).map(|i| if i % 2 == 0 { [0.0; 4] } else { [100.0, 0.0, 0.0, 0.0] }).collect();
            Self {
                depth,
                normal: picture([0.0, 1.0, 0.0, 0.8]),
                albedo: picture([0.8, 0.8, 0.8, 0.0]),
                motion: testing::picture(device, queue, SIZE.0, SIZE.1, &motion).create_view(&Default::default()),
            }
        }

        fn inputs(&self, motion: bool) -> RestirInputs<'_> {
            RestirInputs {
                depth: &self.depth,
                normal_roughness: &self.normal,
                albedo_metallic: &self.albedo,
                motion: motion.then_some(&self.motion),
                ddgi_packed: None,
            }
        }
    }

    fn render(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &mut RestirRenderer,
        scene: &TlasSceneGpu,
        gbuffer: &GBuffer,
        params: &RestirParams,
        lights: &[RtLocalLight],
    ) -> Vec<[f32; 4]> {
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer.dispatch(device, &mut encoder, scene, gbuffer.inputs(params.enable_gi), SIZE, params, lights).unwrap();
        let output = if params.enable_di { renderer.di_texture() } else { renderer.gi_texture() }.unwrap();
        let row_bytes = SIZE.0 * 8;
        assert_eq!(row_bytes % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
        let bytes = u64::from(row_bytes * SIZE.1);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReSTIR witness"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row_bytes), rows_per_image: None },
            },
            output.size(),
        );
        queue.submit([encoder.finish()]);
        let raw = testing::read(device, queue, &readback, bytes);
        raw.chunks_exact(8)
            .map(|pixel| {
                std::array::from_fn(|i| {
                    let bits = u16::from_le_bytes([pixel[2 * i], pixel[2 * i + 1]]);
                    assert_ne!(bits & 0x7c00, 0x7c00, "nonfinite ReSTIR output");
                    testing::unhalf(bits)
                })
            })
            .collect()
    }

    fn quad(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> [BlasTriangle; 2] {
        [BlasTriangle::flat(a, b, c), BlasTriangle::flat(a, c, d)]
    }

    fn divider(top: f32) -> [BlasTriangle; 2] {
        quad([0.0, 0.0, -2.0], [0.0, top, -2.0], [0.0, top, 2.0], [0.0, 0.0, 2.0])
    }

    fn region(columns: std::ops::Range<usize>) -> impl Iterator<Item = usize> {
        (4..28).flat_map(move |y| columns.clone().map(move |x| y * SIZE.0 as usize + x))
    }

    #[allow(clippy::cast_precision_loss, reason = "witness averages contain at most 2048 finite pixels")]
    fn mean(values: impl Iterator<Item = f32>) -> f32 {
        let (sum, count) = values.fold((0.0, 0u32), |(sum, n), val| (sum + val, n + 1));
        sum / count as f32
    }

    #[test]
    fn gpu_restir_di_many_lights_shadow_and_spatiotemporal_convergence_witness() {
        let (device, queue) = testing::device_with_optional().expect("headless GPU adapter required");
        let wall = Blas::build(&divider(4.0)).unwrap();
        let scene = TlasScene::build(&[wall], &[TlasInstance::new(0, IDENTITY)]).unwrap();
        // Static uploads intentionally exercise software fallback even on a hardware-capable device.
        let scene = TlasSceneGpu::upload(&device, &scene).unwrap();
        let gbuffer = GBuffer::new(&device, &queue, false);
        let mut renderer = RestirRenderer::new(&device);
        let mut params = params();
        params.enable_di = true;

        // Coincident emitters make the exact answer available using one equivalent light.
        // One bright emitter makes the eight-candidate starting estimate deliberately noisy.
        let point = |intensity| RtLocalLight {
            position_range: [-1.4, 2.0, 0.0, 14.0],
            color_outer_cos: [intensity, intensity, intensity, -1.0],
            direction_inner_cos: [0.0, -1.0, 0.0, -1.0],
        };
        let mut lights = [point(0.4); 32];
        lights[0] = point(26.0);
        let mut first = Vec::new();
        let mut last = Vec::new();
        for frame in 0..=8 {
            params.frame_index = frame;
            let pixels = render(&device, &queue, &mut renderer, &scene, &gbuffer, &params, &lights);
            if frame == 0 {
                first = pixels;
            } else if frame == 8 {
                last = pixels;
            }
        }
        renderer.reset_history();
        let reference = render(&device, &queue, &mut renderer, &scene, &gbuffer, &params, &[point(26.0 + 31.0 * 0.4)]);
        let mse = |pixels: &[[f32; 4]]| mean(region(4..25).map(|i| (pixels[i][0] / reference[i][0] - 1.0).powi(2)));
        let (first_error, last_error) = (mse(&first), mse(&last));
        let lit = mean(region(4..25).map(|i| last[i][0]));
        let expected = mean(region(4..25).map(|i| reference[i][0]));
        let shadow = mean(region(32..48).map(|i| last[i][0]));
        eprintln!("ReSTIR DI: lit={lit}, expected={expected}, shadow={shadow}, MSE ratio={}", first_error / last_error);
        assert!(lit > 0.75 && (lit / expected - 1.0).abs() < 0.2, "lit={lit}, exact={expected}");
        assert!(shadow < lit * 0.01, "occluded region={shadow}, lit={lit}");
        assert!(last_error * 3.0 < first_error, "normalized DI MSE: frame 0={first_error}, frame 8={last_error}");
    }

    #[test]
    fn gpu_restir_gi_cornell_color_bleed_and_thin_wall_visibility_witness() {
        let (device, queue) = testing::device_with_optional().expect("headless GPU adapter required");
        let quads = [
            quad([-2.2, 0.0, -2.0], [-2.2, 0.0, 2.0], [2.2, 0.0, 2.0], [2.2, 0.0, -2.0]),
            quad([-2.2, 2.4, -2.0], [2.2, 2.4, -2.0], [2.2, 2.4, 2.0], [-2.2, 2.4, 2.0]),
            quad([-2.2, 0.0, -2.0], [-2.2, 2.4, -2.0], [-2.2, 2.4, 2.0], [-2.2, 0.0, 2.0]),
            divider(2.4),
            quad([2.2, 0.0, -2.0], [2.2, 0.0, 2.0], [2.2, 2.4, 2.0], [2.2, 2.4, -2.0]),
            quad([-2.2, 0.0, -2.0], [2.2, 0.0, -2.0], [2.2, 2.4, -2.0], [-2.2, 2.4, -2.0]),
            quad([-2.2, 0.0, 2.0], [-2.2, 2.4, 2.0], [2.2, 2.4, 2.0], [2.2, 0.0, 2.0]),
        ];
        let mut pool = GpuBlasPool::new(&device);
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut instances = Vec::new();
        for (i, tris) in quads.iter().enumerate() {
            let blas = pool.ensure_blas(&device, &mut encoder, i as u64 + 1, tris).unwrap().unwrap();
            let mut instance = TlasInstance::new(blas, IDENTITY);
            if i == 2 {
                instance.emissive = [10.0, 0.08, 0.06];
            }
            instances.push(instance);
        }
        pool.update_tlas(&device, &mut encoder, &instances).unwrap();
        queue.submit([encoder.finish()]);
        let scene = pool.snapshot_scene();
        let gbuffer = GBuffer::new(&device, &queue, true);
        let mut renderer = RestirRenderer::new(&device);
        let mut params = params();
        params.enable_gi = true;
        params.depth_clear = 1.0;
        let frames: Vec<_> = (0..=8)
            .map(|frame| {
                params.frame_index = frame;
                render(&device, &queue, &mut renderer, &scene, &gbuffer, &params, &[])
            })
            .collect();
        let last = &frames[8];
        let lit_red = mean(region(5..22).map(|i| last[i][0]));
        let lit_green = mean(region(5..22).map(|i| last[i][1]));
        // The tested dark strip is directly across the wall, within the reuse disk.
        let dark_red = mean(region(32..36).map(|i| last[i][0]));
        assert!(lit_red > 0.45 && lit_red > 10.0 * lit_green, "lit red={lit_red}, green={lit_green}");
        assert!(dark_red < lit_red * 0.15, "red behind thin wall={dark_red}, lit red={lit_red}");

        let frame_delta = |a: usize, b: usize| mean(region(5..22).map(|i| (frames[a][i][0] - frames[b][i][0]).powi(2)));
        let (early, late) = (frame_delta(0, 1), frame_delta(7, 8));
        eprintln!(
            "ReSTIR GI (hardware TLAS={}): lit red={lit_red}, lit green={lit_green}, dark red={dark_red}, temporal delta ratio={}",
            scene.has_hw_tlas(),
            early / late
        );
        assert!(late < early * 0.65, "GI temporal variance: first delta={early}, final delta={late}");
    }
}
