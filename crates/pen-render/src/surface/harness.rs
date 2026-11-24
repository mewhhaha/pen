//! Compute shaders for the standard surface's tests: a library's WGSL plus a
//! `kernel(i)` that reads case `i` of a float buffer and writes its results to
//! another, optionally with the real sky and lights bound as a mesh pipeline
//! has them (group 0 bindings 4 to 13, group 2).
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::float_cmp,
    reason = "test geometry and the shaders' own f32 against f64"
)]
use pen_protocol::{Light, ProbeVolume, Sky, Vector};
use wgpu::util::DeviceExt;

use crate::environment::{self, Environment};
use crate::lights::{self, Clusters};
use crate::local_shadows::Selection;
use crate::pipeline_cache::PipelineCache;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

pub fn gpu() -> Option<Gpu> {
    let (device, queue) = crate::testing::device()?;
    Some(Gpu { device, queue })
}

/// The sky and the lights of a frame, as a pipeline binds them.
pub struct World {
    environment: Environment,
    clusters: Clusters,
    depth: wgpu::TextureView,
    picture: wgpu::TextureView,
}

pub fn vector(x: f32, y: f32, z: f32) -> Vector {
    Vector { x, y, z }
}

/// A sky of three colors with a diffuse term of `diffuse` all round and a tint of 1.
pub fn sky(zenith: Vector, horizon: Vector, ground: Vector, diffuse: Vector) -> Sky {
    let scale = 1.0 / 0.282_094_8;
    Sky {
        zenith,
        horizon,
        ground,
        tint: vector(1.0, 1.0, 1.0),
        sh0: vector(diffuse.x * scale, diffuse.y * scale, diffuse.z * scale),
        // A little of the sky's direction in the diffuse light too.
        sh1: vector(diffuse.x * 0.3, diffuse.y * 0.3, diffuse.z * 0.3),
        sh3: vector(diffuse.x * 0.1, diffuse.y * 0.1, diffuse.z * 0.1),
        ..Sky::default()
    }
}

impl World {
    pub fn new(gpu: &Gpu, sky: &Sky, lights: &[Light]) -> World {
        let (device, queue) = (&gpu.device, &gpu.queue);
        let mut environment = Environment::new(device, PipelineCache::open(device, None));
        let volume = ProbeVolume::default();
        let using = environment.set_probes(queue, &volume, &[]);
        environment.set_sky(queue, sky, &Vector::default(), &volume, using);
        let mut encoder = device.create_command_encoder(&Default::default());
        environment.encode(device, &mut encoder);
        queue.submit([encoder.finish()]);
        let mut clusters = Clusters::new(device, &lights::group_layout(device));
        clusters.update_one(device, queue, lights, ([0.0; 3], 1000.0), &Selection::default(), None);
        let stand = |format, dimension, layers| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: layers },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor { dimension: Some(dimension), ..Default::default() })
        };
        World {
            environment,
            clusters,
            depth: stand(wgpu::TextureFormat::Depth32Float, wgpu::TextureViewDimension::D2Array, 2),
            picture: stand(wgpu::TextureFormat::Rgba16Float, wgpu::TextureViewDimension::D2, 1),
        }
    }
}

/// A height texture and its sampler, bound at 2 and 3.
pub struct Heights {
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

impl Heights {
    /// `width` by `height` heights in 0 to 1 (red of an Rgba8Unorm), filtered
    /// linearly, repeating.
    pub fn new(gpu: &Gpu, width: u32, height: u32, red: &[u8]) -> Heights {
        assert_eq!(red.len(), (width * height) as usize);
        let texels: Vec<u8> = red.iter().flat_map(|&r| [r, 0, 0, 255]).collect();
        let texture = gpu.device.create_texture_with_data(
            &gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some("heights"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &texels,
        );
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Heights { view: texture.create_view(&Default::default()), sampler }
    }
}

/// Runs `kernel` (WGSL defining `fn kernel(i: u32)`, which may call `arg`,
/// `arg3`, `put` and `put3`) after `library`, once for each of `count` cases
/// of `input` (`in_stride` floats each), and returns `count * out_stride` floats.
#[allow(clippy::too_many_arguments, reason = "what one kernel run is made of")]
pub fn compute(
    gpu: &Gpu,
    library: &str,
    kernel: &str,
    input: &[f32],
    in_stride: usize,
    out_stride: usize,
    world: Option<&World>,
    heights: Option<&Heights>,
) -> Vec<f32> {
    let (device, queue) = (&gpu.device, &gpu.queue);
    let count = input.len() / in_stride;
    assert_eq!(input.len(), count * in_stride);
    let source = format!(
        "{library}
@group(0) @binding(0) var<storage, read> case_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> case_output: array<f32>;
fn arg(i: u32, k: u32) -> f32 {{ return case_input[i * {in_stride}u + k]; }}
fn arg3(i: u32, k: u32) -> vec3f {{ return vec3f(arg(i, k), arg(i, k + 1u), arg(i, k + 2u)); }}
fn put(i: u32, k: u32, value: f32) {{ case_output[i * {out_stride}u + k] = value; }}
fn put3(i: u32, k: u32, value: vec3f) {{ put(i, k, value.x); put(i, k + 1u, value.y); put(i, k + 2u, value.z); }}
{kernel}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {{
  if (id.x < {count}u) {{ kernel(id.x); }}
}}"
    );
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("surface test"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let compute_only = wgpu::ShaderStages::COMPUTE;
    let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: compute_only,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let mut entries = vec![storage(0, true), storage(1, false)];
    if heights.is_some() {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: compute_only,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: compute_only,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
    }
    if world.is_some() {
        entries.extend(environment::layout_entries());
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 12,
            visibility: wgpu::ShaderStages::FRAGMENT | compute_only,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 13,
            visibility: wgpu::ShaderStages::FRAGMENT | compute_only,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
    }
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
    let lights_layout = world.map(|_| lights::group_layout(device));
    let layouts: Vec<Option<&wgpu::BindGroupLayout>> =
        if let Some(lights) = &lights_layout { vec![Some(&layout), None, Some(lights)] } else { vec![Some(&layout)] };
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &layouts,
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("surface test"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(error) = pollster::block_on(scope.pop()) {
        panic!("the test shader does not compile: {error}");
    }
    let inputs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bytes = (count * out_stride * 4) as u64;
    let outputs = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut group: Vec<wgpu::BindGroupEntry> = vec![
        wgpu::BindGroupEntry { binding: 0, resource: inputs.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: outputs.as_entire_binding() },
    ];
    if let Some(heights) = heights {
        group.push(wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&heights.view) });
        group.push(wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&heights.sampler) });
    }
    if let Some(world) = world {
        group.extend(world.environment.entries(world.environment.light()));
        group.push(wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&world.depth) });
        group.push(wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(&world.picture) });
    }
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &group });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        if let Some(world) = world {
            pass.set_bind_group(2, world.clusters.group(), &[]);
        }
        pass.dispatch_workgroups(u32::try_from(count.div_ceil(64)).unwrap(), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&outputs, 0, &readback, 0, bytes);
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mapped = readback.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice(&mapped).to_vec()
}
