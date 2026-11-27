//! World-space radiance cache for secondary ray hits.
//!
//! Encode frame uniforms before consumers and resolve after all consumers.
//! Consumers read the last resolved radiance and deposit into atomic sums.
//! Resolve and clear use separate compute passes with exclusive access.

use anyhow::{Result, ensure};
use wgpu::util::DeviceExt;

pub const SHARC_DEFAULT_CAPACITY: u32 = 1 << 17;
pub const SHARC_WGSL: &str = crate::wgsl!("sharc.wgsl");

/// Prefix this to a read-write shader with its own `sharc_u`, `sharc_keys`,
/// `sharc_accum` and `sharc_resolved` bindings. See the WGSL for their types.
pub const SHARC_COMMON_WGSL: &str = slice_before(SHARC_WGSL, ENTRYPOINTS);

/// Hash + key types only (no storage variable references), suitable for pairing with
/// [`SHARC_QUERY_RO_FN_WGSL`] in read-only consumers without an `sharc_accum` binding.
pub const SHARC_HASH_WGSL: &str = slice_before(SHARC_WGSL, MUTABLE);

/// Read-only `sharc_query_ro(pos, nrm)` helper expecting `sharc_u: SharcUniform`,
/// `var<storage, read> sharc_keys: array<u32>`, and `var<storage, read> sharc_resolved: array<vec4f>`.
pub const SHARC_QUERY_RO_FN_WGSL: &str = r"
fn sharc_query_ro(pos: vec3f, nrm: vec3f) -> vec4f {
    if (sharc_u.counts.z == 0u) {
        return vec4f(0.0);
    }
    let key = sharc_hash_key(pos, nrm, sharc_u);
    let mask = sharc_u.counts.x - 1u;
    for (var probe = 0u; probe < min(SHARC_PROBE_STEPS, sharc_u.counts.x); probe += 1u) {
        let slot = (key.home + probe) & mask;
        if (sharc_keys[slot * 2u] == key.tag && sharc_keys[slot * 2u + 1u] == key.fingerprint) {
            return sharc_resolved[slot];
        }
    }
    return vec4f(0.0);
}
";

const ENTRYPOINTS: &[u8] = b"// __SHARC_ENTRYPOINTS__";
const MUTABLE: &[u8] = b"// __SHARC_MUTABLE__";
const WORKGROUP_SIZE: u32 = 64;
const RADIANCE_SCALE: f32 = 1024.0;
const MAX_SAMPLES: u32 = 256;

const fn slice_before<'a>(source: &'a str, marker: &[u8]) -> &'a str {
    let bytes = source.as_bytes();
    let mut offset = 0;
    while offset + marker.len() <= bytes.len() {
        let mut matched = 0;
        while matched < marker.len() && bytes[offset + matched] == marker[matched] {
            matched += 1;
        }
        if matched == marker.len() {
            return source.split_at(offset).0;
        }
        offset += 1;
    }
    panic!("missing SHaRC WGSL section marker");
}

#[derive(Clone, Copy, Debug)]
pub struct SharcConfig {
    pub capacity: u32,
    pub base_voxel_size: f32,
    pub decay_alpha: f32,
    /// A fresh entry starts at age 1 and is evicted when its age exceeds this limit.
    pub max_age_frames: u32,
    pub max_radiance: f32,
}

impl Default for SharcConfig {
    fn default() -> Self {
        Self {
            capacity: SHARC_DEFAULT_CAPACITY,
            base_voxel_size: 0.25,
            decay_alpha: 0.15,
            max_age_frames: 120,
            max_radiance: 64.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SharcParams {
    pub camera_pos: [f32; 3],
    pub frame_index: u32,
    pub reset: bool,
    pub enabled: bool,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SharcUniform {
    camera_and_voxel: [f32; 4],
    params: [f32; 4],
    counts: [u32; 4],
    pad: [u32; 4],
}

impl SharcUniform {
    fn new(config: SharcConfig, params: &SharcParams) -> Self {
        let [x, y, z] = params.camera_pos;
        Self {
            camera_and_voxel: [x, y, z, config.base_voxel_size],
            params: [config.decay_alpha, RADIANCE_SCALE, config.max_radiance, params.frame_index as f32],
            counts: [config.capacity, config.max_age_frames, u32::from(params.enabled), 0],
            pad: [0; 4],
        }
    }
}

pub struct SharcCache {
    config: SharcConfig,
    uniform: wgpu::Buffer,
    keys: wgpu::Buffer,
    accum: wgpu::Buffer,
    resolved: wgpu::Buffer,
    group: wgpu::BindGroup,
    clear_pipeline: wgpu::ComputePipeline,
    resolve_pipeline: wgpu::ComputePipeline,
}

impl SharcCache {
    pub fn new(device: &wgpu::Device, config: SharcConfig) -> Result<Self> {
        let limits = device.limits();
        let bucket_bytes = u64::from(config.capacity) * 16;
        ensure!(
            config.capacity.is_power_of_two()
                && config.capacity <= 1 << 30
                && bucket_bytes <= limits.max_storage_buffer_binding_size
                && bucket_bytes <= limits.max_buffer_size
                && config.capacity.div_ceil(WORKGROUP_SIZE) <= limits.max_compute_workgroups_per_dimension,
            "invalid SHaRC capacity {} for device limits",
            config.capacity
        );
        ensure!(
            limits.max_storage_buffers_per_shader_stage >= 3
                && limits.max_uniform_buffers_per_shader_stage >= 1
                && limits.max_bindings_per_bind_group >= 4
                && limits.max_compute_workgroup_size_x >= WORKGROUP_SIZE
                && limits.max_compute_invocations_per_workgroup >= WORKGROUP_SIZE,
            "SHaRC compute and binding requirements exceed device limits"
        );
        ensure!(
            config.base_voxel_size > 0.0 && config.base_voxel_size <= f32::MAX / 4096.0,
            "invalid SHaRC base voxel size"
        );
        ensure!((0.0..=1.0).contains(&config.decay_alpha), "invalid SHaRC decay alpha");
        ensure!((1..(1 << 24)).contains(&config.max_age_frames), "invalid SHaRC max age for f32 storage");
        let max_radiance = (u32::MAX / MAX_SAMPLES) as f32 / RADIANCE_SCALE;
        ensure!(config.max_radiance > 0.0 && config.max_radiance <= max_radiance, "invalid SHaRC radiance cap");

        // The initial uniform is valid and disabled. WebGPU zero-initializes storage.
        let initial = SharcUniform::new(config, &SharcParams::default());
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sharc uniforms"),
            contents: bytemuck::bytes_of(&initial),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let storage = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let keys = storage("sharc keys", bucket_bytes / 2);
        let accum = storage("sharc accumulation", bucket_bytes);
        let resolved = storage("sharc resolved radiance", bucket_bytes);

        let buffers = [&uniform, &keys, &accum, &resolved];
        let layout_entries = std::array::from_fn::<_, 4, _>(|i| wgpu::BindGroupLayoutEntry {
            binding: u32::try_from(i).unwrap(),
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage { read_only: false }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sharc maintenance"),
            entries: &layout_entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sharc maintenance"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sharc maintenance"),
            layout: &layout,
            entries: &std::array::from_fn::<_, 4, _>(|i| wgpu::BindGroupEntry {
                binding: layout_entries[i].binding,
                resource: buffers[i].as_entire_binding(),
            }),
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sharc"),
            source: wgpu::ShaderSource::Wgsl(crate::deferred_wgsl!("sharc.wgsl").into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: crate::pipeline_cache::active().as_ref(),
            })
        };
        Ok(Self {
            config,
            uniform,
            keys,
            accum,
            resolved,
            group,
            clear_pipeline: pipeline("sharc_clear"),
            resolve_pipeline: pipeline("sharc_resolve"),
        })
    }

    /// Create a small cache whose initial uniform disables lookup and deposit.
    pub fn new_disabled(device: &wgpu::Device) -> Result<Self> {
        Self::new(device, SharcConfig { capacity: 64, ..Default::default() })
    }

    /// Upload for the next queue submission. For a reset, encode `clear`
    /// before the first consumer. Use `encode_uniform` for multiple frames
    /// in the same submission.
    pub fn update_uniform(&self, queue: &wgpu::Queue, params: &SharcParams) {
        let uniform = SharcUniform::new(self.config, params);
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// Begin a frame with an encoder-ordered upload and clear on reset.
    pub fn encode_uniform(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, params: &SharcParams) {
        self.upload(device, encoder, params);
        if params.reset {
            self.clear(encoder);
        }
    }

    fn upload(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, params: &SharcParams) {
        let uniform = SharcUniform::new(self.config, params);
        crate::staging::encode_copy(device, encoder, &self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    pub fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        self.dispatch(encoder, &self.clear_pipeline, "sharc clear");
    }

    /// Finish a frame after its consumers. Disabled frames pause aging.
    /// Apply reset before consumers via `encode_uniform` or `clear`.
    pub fn resolve(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, params: &SharcParams) {
        if params.enabled {
            self.upload(device, encoder, params);
            self.dispatch(encoder, &self.resolve_pipeline, "sharc resolve");
        }
    }

    /// Zero-allocation resolve using the uniform already written via [`Self::update_uniform`]
    /// or [`Self::encode_uniform`].
    pub fn resolve_uploaded(&self, encoder: &mut wgpu::CommandEncoder, enabled: bool) {
        if enabled {
            self.dispatch(encoder, &self.resolve_pipeline, "sharc resolve");
        }
    }

    fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline, label: &str) {
        let ts = crate::renderer::timing::scope(crate::renderer::timing::Span::Scene);
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: ts.compute() });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(self.config.capacity.div_ceil(WORKGROUP_SIZE), 1, 1);
    }

    pub fn uniform_buffer(&self) -> &wgpu::Buffer {
        &self.uniform
    }

    pub fn keys_buffer(&self) -> &wgpu::Buffer {
        &self.keys
    }

    pub fn accum_buffer(&self) -> &wgpu::Buffer {
        &self.accum
    }

    pub fn resolved_buffer(&self) -> &wgpu::Buffer {
        &self.resolved
    }

    pub fn config(&self) -> SharcConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing;

    const WITNESS_WGSL: &str = r"
@group(0) @binding(0) var<uniform> sharc_u: SharcUniform;
@group(0) @binding(1) var<storage, read_write> sharc_keys: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> sharc_accum: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> sharc_resolved: array<vec4f>;

struct Hit {
    pos: vec4f,
    nrm: vec4f,
    radiance: vec4f,
}

@group(0) @binding(4) var<storage, read> hits: array<Hit>;
@group(0) @binding(5) var<storage, read_write> queries: array<vec4f>;

@compute @workgroup_size(64)
fn deposit(@builtin(global_invocation_id) id: vec3u) {
    let hit = hits[id.y];
    sharc_accumulate(hit.pos.xyz, hit.nrm.xyz, hit.radiance.rgb);
}

@compute @workgroup_size(1)
fn query(@builtin(global_invocation_id) id: vec3u) {
    let hit = hits[id.x];
    queries[id.x] = sharc_query(hit.pos.xyz, hit.nrm.xyz);
}
";

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Hit {
        pos: [f32; 4],
        nrm: [f32; 4],
        radiance: [f32; 4],
    }

    struct Witness {
        device: wgpu::Device,
        queue: wgpu::Queue,
        cache: SharcCache,
        deposit: wgpu::ComputePipeline,
        query: wgpu::ComputePipeline,
    }

    impl Witness {
        fn new(config: SharcConfig) -> Self {
            let (device, queue) = testing::device().expect("headless GPU adapter required");
            let cache = SharcCache::new(&device, config).unwrap();
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sharc witness"),
                source: wgpu::ShaderSource::Wgsl(format!("{SHARC_COMMON_WGSL}{WITNESS_WGSL}").into()),
            });
            let pipeline = |entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: crate::pipeline_cache::active().as_ref(),
                })
            };
            let deposit = pipeline("deposit");
            let query = pipeline("query");
            Self { device, queue, cache, deposit, query }
        }

        fn hits(&self, hits: &[Hit]) -> wgpu::Buffer {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sharc witness hits"),
                contents: bytemuck::cast_slice(hits),
                usage: wgpu::BufferUsages::STORAGE,
            })
        }

        fn dispatch(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            pipeline: &wgpu::ComputePipeline,
            buffers: &[(u32, &wgpu::Buffer)],
            groups: (u32, u32),
        ) {
            let entries: Vec<_> = buffers
                .iter()
                .map(|&(binding, buffer)| wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() })
                .collect();
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sharc witness"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sharc witness"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }

        fn frame(&self, frame_index: u32, deposits: &[Hit], lookups: &[Hit]) -> Vec<[f32; 4]> {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            let params = SharcParams { frame_index, reset: frame_index == 0, enabled: true, ..Default::default() };
            self.cache.encode_uniform(&self.device, &mut encoder, &params);
            let uniform = self.cache.uniform_buffer();
            let keys = self.cache.keys_buffer();
            if !deposits.is_empty() {
                self.dispatch(
                    &mut encoder,
                    &self.deposit,
                    &[(0, uniform), (1, keys), (2, self.cache.accum_buffer()), (4, &self.hits(deposits))],
                    (8, u32::try_from(deposits.len()).unwrap()),
                );
            }
            self.cache.resolve(&self.device, &mut encoder, &params);

            let query_count = u32::try_from(lookups.len()).unwrap();
            let size = u64::from(query_count) * 16;
            let output = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sharc witness query output"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.dispatch(
                &mut encoder,
                &self.query,
                &[(0, uniform), (1, keys), (3, self.cache.resolved_buffer()), (4, &self.hits(lookups)), (5, &output)],
                (query_count, 1),
            );
            self.queue.submit([encoder.finish()]);
            testing::read(&self.device, &self.queue, &output, size)
                .as_chunks::<16>()
                .0
                .iter()
                .map(|row| bytemuck::pod_read_unaligned(row))
                .collect()
        }
    }

    #[test]
    fn gpu_sharc_unbounded_multi_room_normal_separation_and_convergence_witness() {
        let gpu = Witness::new(SharcConfig::default());
        let red = Hit { pos: [0.25, -0.25, 0.0, 0.0], nrm: [1.0, 0.0, 0.0, 0.0], radiance: [4.0, 0.1, 0.1, 0.0] };
        let blue = Hit { nrm: [-1.0, 0.0, 0.0, 0.0], radiance: [0.1, 0.1, 4.0, 0.0], ..red };
        let far = Hit { pos: [150.0, 1.0, 1.0, 0.0], radiance: [1.0, 2.0, 3.0, 0.0], ..red };
        // At the default 0.25 m base size, both far positions share a 16 m cell.
        // A fixed-resolution cache would miss the offset lookup.
        let far_lookup = Hit { pos: [153.0, 1.0, 1.0, 0.0], ..far };
        let far_miss = Hit { pos: [164.0, 1.0, 1.0, 0.0], ..far };
        let faint = 102.0 / RADIANCE_SCALE;
        let expected = [[4.0, faint, faint, 1.0], [faint, faint, 4.0, 1.0], [1.0, 2.0, 3.0, 1.0], [0.0; 4]];
        for frame in 0..4 {
            let actual = gpu.frame(frame, &[red, blue, far], &[red, blue, far_lookup, far_miss]);
            for (sample, exact) in actual.iter().zip(expected) {
                for (value, target) in sample.iter().zip(exact) {
                    assert!((value - target).abs() < 1e-6, "frame {frame}: {sample:?} vs {exact:?}");
                }
            }
        }
    }

    #[test]
    fn gpu_sharc_rejects_a_collision_on_the_original_fingerprint() {
        let gpu = Witness::new(SharcConfig { capacity: 1, ..Default::default() });
        // Distinct cells with +X normals and the same original tag, 0x5971df11.
        let stored = Hit { pos: [14.5, 2.5, -4.5, 0.0], nrm: [1.0, 0.0, 0.0, 0.0], radiance: [4.0, 0.0, 0.0, 0.0] };
        let colliding = Hit { pos: [7.0, -15.0, -27.0, 0.0], radiance: [0.0, 4.0, 0.0, 0.0], ..stored };
        assert_eq!(gpu.frame(0, &[stored], &[stored, colliding]), [[4.0, 0.0, 0.0, 1.0], [0.0; 4]]);
        assert_eq!(gpu.frame(1, &[colliding], &[stored, colliding]), [[4.0, 0.0, 0.0, 2.0], [0.0; 4]]);
    }

    #[test]
    fn gpu_sharc_temporal_ema_adaptation_and_stale_eviction_witness() {
        let gpu = Witness::new(SharcConfig {
            capacity: 64,
            decay_alpha: 0.25,
            max_age_frames: 3,
            // At the fixed-point limit, accepting one extra sample wraps a sum.
            max_radiance: (u32::MAX / MAX_SAMPLES) as f32 / RADIANCE_SCALE,
            ..Default::default()
        });
        let config = gpu.cache.config();
        let green = Hit {
            pos: [-2.0, 1.0, 0.0, 0.0],
            nrm: [0.0, 1.0, 0.0, 0.0],
            radiance: [0.0, config.max_radiance, 0.0, 0.0],
        };
        let orange = Hit { radiance: [16.0, 4.0, 1.0, 0.0], ..green };
        let mut expected = green.radiance;
        expected[3] = 1.0;
        let first = gpu.frame(0, &[green], &[green])[0];
        assert_eq!(first, expected);
        let mut last = first;
        for frame in 1..=6 {
            for (value, target) in expected[..3].iter_mut().zip(orange.radiance) {
                *value += config.decay_alpha * (target - *value);
            }
            let sample = gpu.frame(frame, &[orange], &[orange])[0];
            assert!(last[0] < sample[0] && sample[0] < orange.radiance[0]);
            assert!(orange.radiance[1] < sample[1] && sample[1] < last[1]);
            for (value, target) in sample.iter().zip(expected) {
                assert!((value - target).abs() < 0.002, "frame {frame}: {sample:?} vs {expected:?}");
            }
            last = sample;
        }
        for tick in 1..=config.max_age_frames + 2 {
            let mut aged = last;
            aged[3] += tick as f32;
            let expected = if tick < config.max_age_frames { aged } else { [0.0; 4] };
            assert_eq!(gpu.frame(6 + tick, &[], &[orange])[0], expected);
        }
        let keys = gpu.cache.keys_buffer();
        assert!(testing::read(&gpu.device, &gpu.queue, keys, keys.size()).iter().all(|&byte| byte == 0));
    }
}
