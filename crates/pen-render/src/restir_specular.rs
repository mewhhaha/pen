//! GPU glossy reflection sampling and spatiotemporal reservoir reuse.

use std::collections::HashMap;

use anyhow::{Result, ensure};
#[cfg(test)]
use wgpu::util::DeviceExt;

#[cfg(not(target_arch = "wasm32"))]
use crate::ray_tracing::TLAS_TRAVERSAL_HW_WGSL;
use crate::ray_tracing::{RtLocalLight, TlasSceneGpu, device_supports_hardware_ray_query};
use crate::restir::RESTIR_MAX_LIGHTS;
use crate::sharc::{SHARC_HASH_WGSL, SHARC_QUERY_RO_FN_WGSL, SharcCache};

pub const RESTIR_SPECULAR_WGSL: &str = crate::wgsl!("restir_specular.wgsl");

const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const GEOMETRY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
pub(crate) const RESERVOIR_STRIDE: u64 = 64;
const LIGHT_BUFFER_BYTES: usize = RESTIR_MAX_LIGHTS * std::mem::size_of::<RtLocalLight>();
const BINDING_COUNT: usize = 33;

#[derive(Clone, Copy, Debug)]
pub struct RestirSpecularParams {
    /// Current and previous column-major WebGPU view-projection matrices.
    pub view_projection: [f32; 16],
    pub prev_view_projection: [f32; 16],
    pub camera_pos: [f32; 3],
    /// Sky depth, typically 1.0 for forward Z and 0.0 for reversed Z.
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
    /// Skip surfaces above this perceptual roughness.
    pub max_roughness: f32,
    pub reset_history: bool,
    /// Strength of optional DDGI and SHaRC lighting at secondary hits.
    pub gi_bounce_scale: f32,
}

impl Default for RestirSpecularParams {
    fn default() -> Self {
        Self {
            view_projection: crate::matrix::IDENTITY,
            prev_view_projection: crate::matrix::IDENTITY,
            camera_pos: [0.0; 3],
            depth_clear: 0.0,
            sun_direction: [0.0, 1.0, 0.0],
            sun_color: [0.0; 3],
            sky_zenith: [0.0; 3],
            sky_horizon: [0.0; 3],
            max_ray_distance: 100.0,
            normal_bias: 0.001,
            visibility_mask: u32::MAX,
            frame_index: 0,
            max_roughness: 0.75,
            reset_history: false,
            gi_bounce_scale: 1.0,
        }
    }
}

pub struct RestirSpecularInputs<'a> {
    /// Depth32Float or R32Float. The G-buffer views share a full-resolution extent.
    pub depth: &'a wgpu::TextureView,
    pub normal_roughness: &'a wgpu::TextureView,
    pub albedo_metallic: &'a wgpu::TextureView,
    /// Rg16Float or Rgba16Float, with current UV minus previous UV in xy.
    /// Missing motion or magnitude >= 100 in either lane uses matrix reprojection.
    pub motion: Option<&'a wgpu::TextureView>,
    /// Packed DDGI cascades with STORAGE usage.
    pub ddgi_buf: Option<&'a wgpu::Buffer>,
    /// Optional read-only radiance cache for points without full DDGI coverage.
    pub sharc: Option<&'a SharcCache>,
}

pub struct RestirSpecularOutputs<'a> {
    /// Rgba16Float with linear specular radiance in RGB and hit distance in meters in A.
    /// Sky and invalid samples have zero distance.
    pub specular_view: &'a wgpu::TextureView,
    /// Rgba16Float with surface motion in xy and virtual reflected-point motion in zw.
    pub motion_view: &'a wgpu::TextureView,
}

// Matches RestirUniform in the shared ReSTIR lighting shader. Specular uses
// sky_horizon.w for the roughness cutoff and flags.xy for history and SHaRC.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LightingUniform {
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SpecTransforms {
    view_projection: [f32; 16],
    inverse_prev_view_projection: [f32; 16],
}

fn storage_image(
    device: &wgpu::Device,
    size: (u32, u32),
    label: &str,
    format: wgpu::TextureFormat,
    extra_usage: wgpu::TextureUsages,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | extra_usage,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

struct Frame {
    size: (u32, u32),
    geometry: [wgpu::TextureView; 2],
    specular: wgpu::TextureView,
    motion: wgpu::TextureView,
    history: [wgpu::Buffer; 2],
    scratch: wgpu::Buffer,
    previous: usize,
    previous_camera: [f32; 3],
    previous_gbuffer_size: (u32, u32),
}

impl Frame {
    fn new(device: &wgpu::Device, size: (u32, u32), bytes: u64) -> Self {
        let reservoir = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let geometry = |label| storage_image(device, size, label, GEOMETRY_FORMAT, wgpu::TextureUsages::empty());
        let output = |label| storage_image(device, size, label, OUTPUT_FORMAT, wgpu::TextureUsages::COPY_SRC);
        Self {
            size,
            geometry: [geometry("specular geometry A"), geometry("specular geometry B")],
            specular: output("ReSTIR specular radiance"),
            motion: output("ReSTIR specular motion"),
            history: [reservoir("specular history A"), reservoir("specular history B")],
            scratch: reservoir("specular reservoir scratch"),
            previous: 0,
            previous_camera: [0.0; 3],
            previous_gbuffer_size: (0, 0),
        }
    }

    fn outputs(&self) -> RestirSpecularOutputs<'_> {
        RestirSpecularOutputs { specular_view: &self.specular, motion_view: &self.motion }
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
                    0 | 21 | 22 | 30 => buffer(wgpu::BufferBindingType::Uniform),
                    #[cfg(not(target_arch = "wasm32"))]
                    1 if hardware => wgpu::BindingType::AccelerationStructure { vertex_return: false },
                    1..=4 | 10 | 25 | 26 | 31 | 32 => storage(true),
                    27 => storage(false),
                    5..=8 | 23 => wgpu::BindingType::Texture {
                        sample_type: if binding == 5 && native_depth {
                            wgpu::TextureSampleType::Depth
                        } else {
                            wgpu::TextureSampleType::Float { filterable: false }
                        },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    24 | 28 | 29 => wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: if binding == 24 { GEOMETRY_FORMAT } else { OUTPUT_FORMAT },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    _ => unreachable!("unknown ReSTIR specular binding {binding}"),
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
            cache: crate::pipeline_cache::active().as_ref(),
        });
        Self { layout, pipeline, entry, bindings }
    }

    fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &[Option<wgpu::BindingResource<'_>>; BINDING_COUNT],
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
        let ts = crate::renderer::timing::scope(crate::renderer::timing::Span::Scene);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.entry),
            timestamp_writes: ts.compute(),
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
    }
}

fn pipelines(device: &wgpu::Device, hardware: bool, native_depth: bool, specular_wgsl: &str) -> [Pass; 3] {
    let (traversal, tlas_binding) = (
        crate::deferred_wgsl!("tlas_traversal.wgsl"),
        "@group(0) @binding(1) var<storage, read> tlas_nodes: array<BvhNode>;",
    );
    #[cfg(not(target_arch = "wasm32"))]
    let (traversal, tlas_binding) = if hardware {
        (TLAS_TRAVERSAL_HW_WGSL, "@group(0) @binding(1) var hw_tlas: acceleration_structure;")
    } else {
        (traversal, tlas_binding)
    };
    // WGSL binds only the globals reached by each entry point. Share the thin
    // G-buffer mapping, TLAS, light attenuation and packed DDGI decode with ReSTIR.
    let restir = crate::deferred_wgsl!("restir.wgsl").replace("// __TLAS_BINDING_1__", tlas_binding);
    let mut body = format!("{traversal}\n{restir}\n{SHARC_HASH_WGSL}\n{SHARC_QUERY_RO_FN_WGSL}\n{specular_wgsl}");
    if native_depth {
        body = body
            .replace("var depth_tex: texture_2d<f32>;", "var depth_tex: texture_depth_2d;")
            .replace("textureLoad(depth_tex, g_px, 0).x", "textureLoad(depth_tex, g_px, 0)");
    }
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ReSTIR specular"),
        source: wgpu::ShaderSource::Wgsl(body.into()),
    });
    let make = |entry, bindings| Pass::new(device, &module, entry, bindings, hardware, native_depth);
    // Hardware trace_tlas does not read the software BLAS-node binding.
    let initial = if hardware {
        &[0, 1, 2, 4, 5, 6, 7, 10, 22, 27, 30, 31, 32][..]
    } else {
        &[0, 1, 2, 3, 4, 5, 6, 7, 10, 22, 27, 30, 31, 32]
    };
    let spatial = if hardware {
        &[0, 1, 2, 4, 5, 6, 7, 8, 21, 25, 27, 28, 29][..]
    } else {
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 21, 25, 27, 28, 29]
    };
    [
        make("spec_initial", initial),
        make("spec_temporal", &[0, 5, 6, 7, 8, 21, 23, 24, 25, 26, 27]),
        make("spec_spatial_resolve", spatial),
    ]
}

pub struct RestirSpecularRenderer {
    pipelines: HashMap<(bool, bool), [Pass; 3]>,
    hardware_supported: bool,
    dummy: wgpu::Buffer,
    lighting_buf: wgpu::Buffer,
    transforms_buf: wgpu::Buffer,
    lights_buf: wgpu::Buffer,
    frame: Option<Frame>,
    history_valid: bool,
}

impl RestirSpecularRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReSTIR specular empty DDGI and SHaRC"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let lighting_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReSTIR specular lighting"),
            size: std::mem::size_of::<LightingUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let transforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReSTIR specular transforms"),
            size: std::mem::size_of::<SpecTransforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReSTIR specular lights"),
            size: LIGHT_BUFFER_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipelines: HashMap::new(),
            hardware_supported: device_supports_hardware_ray_query(device),
            dummy,
            lighting_buf,
            transforms_buf,
            lights_buf,
            frame: None,
            history_valid: false,
        }
    }

    /// Whether native ray queries are enabled. Software snapshots use compute traversal.
    pub fn uses_hardware_ray_query(&self) -> bool {
        self.hardware_supported
    }

    pub fn reset_history(&mut self) {
        self.history_valid = false;
    }

    pub fn outputs(&self) -> Option<RestirSpecularOutputs<'_>> {
        self.frame.as_ref().map(Frame::outputs)
    }

    /// Encode a frame at full or reduced resolution, and submit frames in dispatch
    /// order. Reset on camera cuts or discontinuous changes to scene or lighting.
    #[allow(clippy::too_many_arguments, reason = "matches ReSTIR scene and frame input conventions")]
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene: &TlasSceneGpu,
        inputs: RestirSpecularInputs<'_>,
        size: (u32, u32),
        params: &RestirSpecularParams,
        local_lights: &[RtLocalLight],
    ) -> Result<RestirSpecularOutputs<'_>> {
        let limits = device.limits();
        ensure!(
            size.0 > 0 && size.1 > 0 && size.0.max(size.1) <= limits.max_texture_dimension_2d,
            "invalid ReSTIR specular extent {size:?}"
        );
        let bytes = u64::from(size.0) * u64::from(size.1) * RESERVOIR_STRIDE;
        ensure!(
            bytes <= limits.max_storage_buffer_binding_size && bytes <= limits.max_buffer_size,
            "ReSTIR specular reservoir buffer of {bytes} bytes exceeds device limits"
        );
        let hardware = self.hardware_supported && scene.has_hw_tlas();
        let storage_needed = if hardware { 6 } else { 8 };
        ensure!(
            limits.max_storage_buffers_per_shader_stage >= storage_needed,
            "ReSTIR specular requires {storage_needed} storage buffers per compute stage"
        );
        ensure!(local_lights.len() <= RESTIR_MAX_LIGHTS, "ReSTIR specular supports up to {RESTIR_MAX_LIGHTS} lights");
        let depth_format = inputs.depth.texture().format();
        let gbuffer_size =
            validate_input(inputs.depth, &[wgpu::TextureFormat::Depth32Float, wgpu::TextureFormat::R32Float])?;
        let native_depth = depth_format == wgpu::TextureFormat::Depth32Float;
        ensure!(
            gbuffer_size.0 >= size.0
                && gbuffer_size.1 >= size.1
                && validate_input(inputs.normal_roughness, &[OUTPUT_FORMAT])? == gbuffer_size
                && validate_input(inputs.albedo_metallic, &[OUTPUT_FORMAT])? == gbuffer_size,
            "ReSTIR specular G-buffer textures must share an extent of at least {size:?}"
        );
        if let Some(motion) = inputs.motion {
            let motion_size = validate_input(motion, &[wgpu::TextureFormat::Rg16Float, OUTPUT_FORMAT])?;
            ensure!(motion_size.0 >= size.0 && motion_size.1 >= size.1, "motion extent must be at least {size:?}");
        }
        if let Some(ddgi) = inputs.ddgi_buf {
            ensure!(
                ddgi.size() >= 16 && ddgi.usage().contains(wgpu::BufferUsages::STORAGE),
                "ReSTIR specular DDGI must bind a packed storage buffer"
            );
        }
        ensure!(
            params.max_ray_distance.is_finite()
                && params.max_ray_distance >= 0.001
                && params.normal_bias.is_finite()
                && (0.0..params.max_ray_distance).contains(&params.normal_bias),
            "invalid ReSTIR specular ray distance or bias"
        );
        ensure!((0.0..=1.0).contains(&params.max_roughness), "invalid ReSTIR specular roughness cutoff");

        if self.frame.as_ref().is_none_or(|frame| frame.size != size) {
            self.frame = Some(Frame::new(device, size, bytes));
            self.history_valid = false;
        }
        let frame = self.frame.as_mut().unwrap();
        let use_history = self.history_valid && !params.reset_history && frame.previous_gbuffer_size == gbuffer_size;
        let sun_length = params.sun_direction.iter().map(|v| v * v).sum::<f32>().sqrt();
        let sun = if sun_length.is_finite() && sun_length > 0.0 {
            params.sun_direction.map(|v| v / sun_length)
        } else {
            [0.0; 3]
        };
        let pack = |[x, y, z]: [f32; 3], w| [x, y, z, w];
        let lighting = LightingUniform {
            inverse_view_projection: crate::matrix::invert(&params.view_projection),
            prev_view_projection: params.prev_view_projection,
            camera_and_clear: pack(params.camera_pos, params.depth_clear),
            previous_camera: pack(frame.previous_camera, 0.0),
            sun_and_max_distance: pack(sun, params.max_ray_distance),
            sun_color_and_bias: pack(params.sun_color, params.normal_bias),
            sky_zenith_and_bounce: pack(params.sky_zenith, params.gi_bounce_scale.max(0.0)),
            sky_horizon: pack(params.sky_horizon, params.max_roughness),
            scene_counts_and_mask: [
                scene.tlas_node_count,
                scene.tlas_instance_count,
                scene.blas_tri_count,
                params.visibility_mask,
            ],
            size_lights_frame: [size.0, size.1, u32::try_from(local_lights.len())?, params.frame_index],
            flags: [u32::from(use_history), u32::from(inputs.sharc.is_some()), u32::from(inputs.motion.is_some()), 0],
        };
        let transforms = SpecTransforms {
            view_projection: params.view_projection,
            inverse_prev_view_projection: crate::matrix::invert(&params.prev_view_projection),
        };
        // Encoder-ordered upload through the shared frame staging ring avoids per-frame buffer allocations.
        let lighting_bytes = bytemuck::bytes_of(&lighting);
        let transforms_bytes = bytemuck::bytes_of(&transforms);
        let live_lights: &[u8] = bytemuck::cast_slice(local_lights);
        crate::staging::encode_copy(device, encoder, &self.lighting_buf, 0, lighting_bytes);
        crate::staging::encode_copy(device, encoder, &self.transforms_buf, 0, transforms_bytes);
        if !live_lights.is_empty() {
            crate::staging::encode_copy(device, encoder, &self.lights_buf, 0, live_lights);
        }

        let [initial, temporal, spatial] = self.pipelines.entry((hardware, native_depth)).or_insert_with(|| {
            pipelines(device, hardware, native_depth, crate::deferred_wgsl!("restir_specular.wgsl"))
        });
        let previous = frame.previous;
        let next = 1 - previous;
        let tlas = scene.tlas_nodes_buf.as_entire_binding();
        #[cfg(not(target_arch = "wasm32"))]
        let tlas = if hardware { scene.hw_tlas.as_ref().unwrap().as_binding() } else { tlas };
        let view = wgpu::BindingResource::TextureView;
        let sharc = |get: for<'a> fn(&'a SharcCache) -> &'a wgpu::Buffer| {
            inputs.sharc.map_or(&self.dummy, get).as_entire_binding()
        };
        let mut resources = std::array::from_fn(|_| None);
        for (binding, resource) in [
            (0, self.lighting_buf.as_entire_binding()),
            (1, tlas),
            (2, scene.tlas_instances_buf.as_entire_binding()),
            (3, scene.blas_nodes_buf.as_entire_binding()),
            (4, scene.blas_tris_buf.as_entire_binding()),
            (5, view(inputs.depth)),
            (6, view(inputs.normal_roughness)),
            (7, view(inputs.albedo_metallic)),
            (8, view(inputs.motion.unwrap_or(inputs.normal_roughness))),
            (10, inputs.ddgi_buf.unwrap_or(&self.dummy).as_entire_binding()),
            (21, self.transforms_buf.as_entire_binding()),
            (22, self.lights_buf.as_entire_binding()),
            (23, view(&frame.geometry[previous])),
            (24, view(&frame.geometry[next])),
            (26, frame.history[previous].as_entire_binding()),
            (27, frame.scratch.as_entire_binding()),
            (28, view(&frame.specular)),
            (29, view(&frame.motion)),
            (30, sharc(SharcCache::uniform_buffer)),
            (31, sharc(SharcCache::keys_buffer)),
            (32, sharc(SharcCache::resolved_buffer)),
        ] {
            resources[binding] = Some(resource);
        }

        initial.dispatch(device, encoder, &resources, size);
        resources[25] = Some(frame.scratch.as_entire_binding());
        resources[27] = Some(frame.history[next].as_entire_binding());
        temporal.dispatch(device, encoder, &resources, size);
        resources[25] = Some(frame.history[next].as_entire_binding());
        // The initial reservoir is dead, so spatial resolve can overwrite scratch.
        resources[27] = Some(frame.scratch.as_entire_binding());
        spatial.dispatch(device, encoder, &resources, size);
        frame.previous = next;
        frame.previous_camera = params.camera_pos;
        frame.previous_gbuffer_size = gbuffer_size;
        self.history_valid = true;
        Ok(frame.outputs())
    }
}

fn validate_input(view: &wgpu::TextureView, formats: &[wgpu::TextureFormat]) -> Result<(u32, u32)> {
    let texture = view.texture();
    ensure!(
        formats.contains(&texture.format())
            && texture.dimension() == wgpu::TextureDimension::D2
            && texture.sample_count() == 1
            && texture.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING),
        "ReSTIR specular input must be a single-sampled 2D texture with format in {formats:?} and TEXTURE_BINDING usage"
    );
    Ok((texture.width(), texture.height()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::IDENTITY;
    use crate::ray_tracing::{Blas, BlasTriangle, GpuBlasPool, TlasInstance, TlasScene};
    use crate::testing;

    const SIZE: (u32, u32) = (64, 32);
    // x spans [-0.8, 0.8], z spans [-0.4, 0.4], and the floor at y = 0 has clip depth 0.5.
    const FLOOR_VP: [f32; 16] = [1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0];

    struct GBuffer {
        depth: wgpu::TextureView,
        normal: wgpu::TextureView,
        albedo: wgpu::TextureView,
        motion: wgpu::TextureView,
    }

    fn extent((width, height): (u32, u32)) -> wgpu::Extent3d {
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 }
    }

    impl GBuffer {
        fn new(device: &wgpu::Device, queue: &wgpu::Queue, roughness: f32, half_res: bool) -> Self {
            let depth = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("specular floor depth"),
                size: extent(SIZE),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: if half_res { wgpu::TextureFormat::Depth32Float } else { wgpu::TextureFormat::R32Float },
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | if half_res { wgpu::TextureUsages::RENDER_ATTACHMENT } else { wgpu::TextureUsages::COPY_DST },
                view_formats: &[],
            });
            let depth_view = depth.create_view(&Default::default());
            let count = (SIZE.0 * SIZE.1) as usize;
            if half_res {
                let mut encoder = device.create_command_encoder(&Default::default());
                drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.5),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                }));
                queue.submit([encoder.finish()]);
            } else {
                queue.write_texture(
                    depth.as_image_copy(),
                    bytemuck::cast_slice(&vec![0.5f32; count]),
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(SIZE.0 * 4), rows_per_image: None },
                    extent(SIZE),
                );
            }
            let pixels = (0..SIZE.1).flat_map(|y| (0..SIZE.0).map(move |x| (x, y)));
            let normals: Vec<_> = pixels
                .map(
                    |(x, y)| {
                        if half_res && (x % 2 == 0 || y % 2 == 0) { [0.0; 4] } else { [0.0, 1.0, 0.0, roughness] }
                    },
                )
                .collect();
            let picture = |data: &[[f32; 4]]| {
                testing::picture(device, queue, SIZE.0, SIZE.1, data).create_view(&Default::default())
            };
            let motion = if half_res {
                let offset = [testing::half(0.002), testing::half(-0.003)];
                device
                    .create_texture_with_data(
                        queue,
                        &wgpu::TextureDescriptor {
                            label: Some("specular Rg16Float motion"),
                            size: extent(SIZE),
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rg16Float,
                            usage: wgpu::TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        },
                        wgpu::util::TextureDataOrder::LayerMajor,
                        bytemuck::cast_slice(&vec![offset; count]),
                    )
                    .create_view(&Default::default())
            } else {
                picture(&vec![[0.0; 4]; count])
            };
            Self {
                depth: depth_view,
                normal: picture(&normals),
                albedo: picture(&vec![[0.9, 0.9, 0.9, 0.9]; count]),
                motion,
            }
        }

        fn inputs(&self) -> RestirSpecularInputs<'_> {
            RestirSpecularInputs {
                depth: &self.depth,
                normal_roughness: &self.normal,
                albedo_metallic: &self.albedo,
                motion: Some(&self.motion),
                ddgi_buf: None,
                sharc: None,
            }
        }
    }

    fn quad(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> [BlasTriangle; 2] {
        [BlasTriangle::flat(a, b, c), BlasTriangle::flat(a, c, d)]
    }

    fn read_rgba(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<[f32; 4]> {
        let row = texture.width() * 8;
        let stride = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let bytes = u64::from(stride) * u64::from(texture.height());
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("specular output readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(stride), rows_per_image: None },
            },
            texture.size(),
        );
        queue.submit([encoder.finish()]);
        testing::read(device, queue, &buffer, bytes)
            .chunks_exact(stride as usize)
            .flat_map(|scanline| scanline[..row as usize].as_chunks::<8>().0)
            .map(|px| {
                let [r, g, b, a] = px.as_chunks::<2>().0 else { unreachable!() };
                [*r, *g, *b, *a].map(|lane| {
                    let value = testing::unhalf(u16::from_le_bytes(lane));
                    assert!(value.is_finite(), "nonfinite specular output");
                    value
                })
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments, reason = "witness varies render mode, scene and frame settings")]
    fn render(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &mut RestirSpecularRenderer,
        scene: &TlasSceneGpu,
        floor: &GBuffer,
        size: (u32, u32),
        params: &RestirSpecularParams,
    ) -> Vec<[f32; 4]> {
        let mut encoder = device.create_command_encoder(&Default::default());
        let output = renderer.dispatch(device, &mut encoder, scene, floor.inputs(), size, params, &[]).unwrap();
        queue.submit([encoder.finish()]);
        read_rgba(device, queue, output.specular_view.texture())
    }

    fn mean(values: impl Iterator<Item = f32>) -> f32 {
        let (sum, count) = values.fold((0.0, 0u32), |(sum, n), val| (sum + val, n + 1));
        sum / count as f32
    }

    fn region(x: std::ops::Range<usize>) -> impl Iterator<Item = usize> {
        (4..28).flat_map(move |y| x.clone().map(move |x| y * SIZE.0 as usize + x))
    }

    #[test]
    fn gpu_restir_specular_glossy_convergence_and_divider_visibility() {
        let (device, queue) = testing::device_with_optional().expect("headless GPU adapter required");
        let wall = quad([-1.5, 0.0, -0.60], [-1.5, 5.0, -0.60], [-1.5, 5.0, 0.60], [-1.5, 0.0, 0.60]);
        let divider = quad([0.0, 0.0, -9.0], [0.0, 8.0, -9.0], [0.0, 8.0, 9.0], [0.0, 0.0, 9.0]);
        let mut red = TlasInstance::new(0, IDENTITY);
        red.albedo = [1.0, 0.0, 0.0];
        let mut black = TlasInstance::new(1, IDENTITY);
        black.albedo = [0.0; 3];
        black.metallic = 1.0;
        black.roughness = 1.0;
        let scene =
            TlasScene::build(&[Blas::build(&wall).unwrap(), Blas::build(&divider).unwrap()], &[red, black]).unwrap();
        let scene = TlasSceneGpu::upload(&device, &scene).unwrap();
        let floor = GBuffer::new(&device, &queue, 0.35, false);
        let mut params = RestirSpecularParams {
            view_projection: FLOOR_VP,
            prev_view_projection: FLOOR_VP,
            camera_pos: [7.2, 12.0, 0.0],
            sun_direction: [0.3, -1.0, 0.0],
            sun_color: [60.0; 3],
            sky_horizon: [0.0, 0.0, 0.15],
            sky_zenith: [0.0, 0.0, 0.15],
            ..Default::default()
        };
        let mut renderer = RestirSpecularRenderer::new(&device);
        let frames: Vec<_> = (0..8)
            .map(|frame| {
                params.frame_index = frame;
                render(&device, &queue, &mut renderer, &scene, &floor, SIZE, &params)
            })
            .collect();

        let unchecked_source = RESTIR_SPECULAR_WGSL
            .replace("const SPEC_VALIDATE_VISIBILITY: bool = true;", "const SPEC_VALIDATE_VISIBILITY: bool = false;");
        assert_ne!(unchecked_source, RESTIR_SPECULAR_WGSL);
        let mut unchecked = RestirSpecularRenderer::new(&device);
        unchecked.pipelines.insert((false, false), pipelines(&device, false, false, &unchecked_source));
        let unchecked_frames: Vec<_> = (0..8)
            .map(|frame| {
                params.frame_index = frame;
                render(&device, &queue, &mut unchecked, &scene, &floor, SIZE, &params)
            })
            .collect();

        let delta = |a: usize, b: usize| mean(region(5..24).map(|i| (frames[a][i][0] - frames[b][i][0]).powi(2)));
        let (early, late) = (delta(0, 1), delta(6, 7));
        let lit = mean(region(5..24).map(|i| frames[7][i][0]));
        let dark = mean(frames.iter().flat_map(|f| region(32..36).map(|i| f[i][0])));
        let leaked = mean(unchecked_frames.iter().flat_map(|f| region(32..36).map(|i| f[i][0])));
        let wall_hits: Vec<_> =
            region(5..24).filter_map(|i| (frames[7][i][0] > 0.5).then_some(frames[7][i][3])).collect();
        let sky = region(0..32).filter(|&i| frames[0][i][2] > 0.01 && frames[0][i][3] == 0.0).count();
        eprintln!(
            "glossy red={lit:.4}, occluded red={dark:.6}, bypass red={leaked:.4}, temporal variance ratio={:.2}, wall hits={}, sky={sky}",
            early / late,
            wall_hits.len()
        );
        assert!(lit > 0.5 && late * 1.8 <= early, "visible red={lit}, first delta={early}, final delta={late}");
        assert!(dark < lit * 0.01 && leaked > dark + lit * 0.015, "red with validation={dark}, bypass={leaked}");
        assert!(
            wall_hits.len() > 100 && wall_hits.iter().all(|&d| (0.8..6.0).contains(&d)),
            "wall distances range {:?}",
            wall_hits.iter().copied().fold((f32::INFINITY, 0.0f32), |(lo, hi), d| (lo.min(d), hi.max(d)))
        );
        assert!(sky > 0, "expected blue sky misses with zero hit distances");
        params.visibility_mask = 0;
        renderer.reset_history();
        let sky_only = render(&device, &queue, &mut renderer, &scene, &floor, SIZE, &params);
        assert!(sky_only.iter().all(|p| p[3] == 0.0) && sky_only.iter().any(|p| p[2] > 0.01));
    }

    #[test]
    fn gpu_restir_specular_mirror_preserves_the_split_target_at_half_resolution() {
        let (device, queue) = testing::device_with_optional().expect("headless GPU adapter required");
        let panel = |left, right| quad([left, 2.0, -2.0], [right, 2.0, -2.0], [right, 2.0, 2.0], [left, 2.0, 2.0]);
        let mut pool = GpuBlasPool::new(&device);
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut instances = Vec::new();
        // Center the projected split between the two full-resolution texels
        // sampled by the neighboring half-resolution pixels.
        let split = 0.0125 * (7.0 / 6.0);
        for (key, limits, color) in [(1, (-2.0, split), [3.0, 0.0, 0.0]), (2, (split, 2.0), [0.0, 0.0, 3.0])] {
            let id = pool.ensure_blas(&device, &mut encoder, key, &panel(limits.0, limits.1)).unwrap().unwrap();
            let mut instance = TlasInstance::new(id, IDENTITY);
            instance.emissive = color;
            instances.push(instance);
        }
        pool.update_tlas(&device, &mut encoder, &instances).unwrap();
        queue.submit([encoder.finish()]);
        let scene = pool.snapshot_scene();
        let floor = GBuffer::new(&device, &queue, 0.04, true);
        let size = (SIZE.0 / 2, SIZE.1 / 2);
        let mut renderer = RestirSpecularRenderer::new(&device);
        let mut params = RestirSpecularParams {
            view_projection: FLOOR_VP,
            prev_view_projection: FLOOR_VP,
            camera_pos: [0.0, 12.0, 0.0],
            depth_clear: 1.0,
            ..Default::default()
        };
        for frame in 0..8 {
            params.frame_index = frame;
            let pixels = render(&device, &queue, &mut renderer, &scene, &floor, size, &params);
            // Oblique VNDF tail samples travel farther than the perpendicular distance.
            let near = pixels.iter().filter(|p| (p[3] - 2.0).abs() < 0.025).count();
            assert!(
                near * 100 >= pixels.len() * 99 && pixels.iter().all(|p| p[3] >= 1.999),
                "mirror frame {frame}: {near}/{} hits near the two-meter panel",
                pixels.len()
            );
            let left = mean(pixels.chunks_exact(size.0 as usize).map(|row| row[size.0 as usize / 2 - 1][0]));
            let right = mean(pixels.chunks_exact(size.0 as usize).map(|row| row[size.0 as usize / 2][2]));
            let smear = mean(
                pixels
                    .chunks_exact(size.0 as usize)
                    .flat_map(|row| [row[size.0 as usize / 2 - 1][2], row[size.0 as usize / 2][0]]),
            );
            assert!(left > 2.1 && right > 2.1 && smear < 0.15, "split edge red={left}, blue={right}, smear={smear}");
        }
        let motion = read_rgba(&device, &queue, renderer.outputs().unwrap().motion_view.texture());
        assert!(motion.iter().all(|p| {
            (p[0] - 0.002).abs() < 1e-5
                && (p[1] + 0.003).abs() < 1e-5
                && (p[2] - p[0]).abs() < 1e-5
                && (p[3] - p[1]).abs() < 1e-5
        }));
        // Alter the previous transform's homogeneous y term. The floor does
        // not move in UV, while the virtual panel at y = -2 projects with w = 0.75.
        params.reset_history = true;
        params.prev_view_projection[7] = 0.125;
        render(&device, &queue, &mut renderer, &scene, &floor, size, &params);
        let parallax = read_rgba(&device, &queue, renderer.outputs().unwrap().motion_view.texture());
        let x = 23u32;
        let ndc_x = 2.0 * (2.0 * x as f32 + 1.5) / SIZE.0 as f32 - 1.0;
        let expected_x = 0.002 - ndc_x * (7.0 / 6.0) / 6.0;
        let px = parallax[(8 * size.0 + x) as usize];
        assert!(
            (px[0] - 0.002).abs() < 1e-5 && (px[2] - expected_x).abs() < 0.01,
            "parallax motion {px:?}, expected x={expected_x}"
        );
        eprintln!("half-resolution mirror split (hardware TLAS={})", scene.has_hw_tlas());
    }
}
