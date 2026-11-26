// Clustered forward lighting for the frame's point and spot lights (the
// render packet's lights section). The view is cut into clusters: 16 x 9
// tiles of the picture, and 24 slices of the depth ahead, closer together
// near the eye (evenly spaced for an orthographic camera). Each light's reach,
// a sphere (a spot's cone in the smallest sphere round it that this finds),
// goes into every cluster it may touch, and the shaders of lit materials read
// only their fragment's cluster's lights: `LIGHTS_WGSL`, bound as group 2 of
// every mesh pipeline. A frame without lights costs one small uniform write.
use anyhow::{Result, ensure};
use pen_protocol::{Light, Matrix, RenderFrame};

use crate::local_shadows::{MAX_FACES, Selection};

/// Tiles across and down the picture, and slices of depth.
pub(crate) const TILES: (u32, u32) = (16, 9);
pub(crate) const SLICES: u32 = 24;
const CLUSTERS: usize = 16 * 9 * 24;

/// The most lights one cluster takes; past them a cluster's lights are left
/// out, in the frame's order.
const PER_CLUSTER: u32 = 256;

/// WGSL a material shader adds to light with the frame's lights:
/// `clustered_light(world, normal, position, view_z)`.
pub const LIGHTS_WGSL: &str = include_str!("shaders/lights.wgsl");

/// How the depth ahead is cut into slices: a depth's slice is
/// `log(depth) * scale + bias`, or `depth * scale + bias` when `linear`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Slicing {
    pub near: f64,
    pub far: f64,
    pub linear: bool,
    pub scale: f64,
    pub bias: f64,
}

impl Slicing {
    fn slice(&self, depth: f64) -> u32 {
        let along = if self.linear { depth } else { depth.max(1e-6).ln() };
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped to the slices")]
        let slice = (along * self.scale + self.bias).floor().clamp(0.0, f64::from(SLICES - 1)) as u32;
        slice
    }
}

/// Each cluster's lights: its first index into `indices` and how many.
pub(crate) struct Assignment {
    pub ranges: Vec<[u32; 2]>,
    pub indices: Vec<u32>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub slicing: Slicing,
}

fn transform(m: &Matrix, point: [f64; 4]) -> [f64; 4] {
    let mut out = [0.0; 4];
    for (row, value) in out.iter_mut().enumerate() {
        *value = (0..4).map(|k| f64::from(m[k * 4 + row]) * point[k]).sum();
    }
    out
}

/// The slicing of the depth `projection` sees, as far as `farthest` when it
/// sees forever.
fn slicing(projection: &Matrix, farthest: f64) -> Slicing {
    let inverse = crate::matrix::invert(projection);
    // The depth ahead of the eye at a depth-buffer value.
    let depth = |value: f64| {
        let point = transform(&inverse, [0.0, 0.0, value, 1.0]);
        if point[3].abs() < 1e-12 { f64::INFINITY } else { -point[2] / point[3] }
    };
    let near = depth(0.0).max(1e-4);
    let mut far = depth(1.0);
    if !far.is_finite() || far <= near {
        far = farthest.max(near * 2.0);
    }
    let linear = projection[11] == 0.0;
    let slices = f64::from(SLICES);
    let (scale, bias) = if linear {
        (slices / (far - near), -near * slices / (far - near))
    } else {
        let scale = slices / (far / near).ln();
        (scale, -near.ln() * scale)
    };
    Slicing { near, far, linear, scale, bias }
}

/// The sphere round what `light` reaches, in view space.
fn reach(light: &Light, view: &Matrix) -> ([f64; 3], f64) {
    let position = [f64::from(light.position.x), f64::from(light.position.y), f64::from(light.position.z)];
    let radius = f64::from(light.radius);
    let outer = f64::from(light.outer);
    let direction = [f64::from(light.direction.x), f64::from(light.direction.y), f64::from(light.direction.z)];
    let length = direction.iter().map(|value| value * value).sum::<f64>().sqrt();
    // A spot's cone, its cap included, fits a smaller sphere along its axis.
    let (center, radius) = if outer > 0.0 && length > 0.0 {
        let along = |distance: f64| std::array::from_fn(|axis| position[axis] + direction[axis] / length * distance);
        if outer >= std::f64::consts::FRAC_1_SQRT_2 {
            let half = radius / (2.0 * outer);
            (along(half), half)
        } else {
            (along(radius * outer), radius * (1.0 - outer * outer).sqrt())
        }
    } else {
        (position, radius)
    };
    let seen = transform(view, [center[0], center[1], center[2], 1.0]);
    ([seen[0], seen[1], seen[2]], radius)
}

/// The clusters each of `lights` may light, seen through `view` and
/// `projection`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn assign(lights: &[Light], view: &Matrix, projection: &Matrix) -> Assignment {
    let mut scratch = AssignScratch::default();
    let slicing = assign_into(lights, view, projection, &mut scratch);
    Assignment { ranges: scratch.ranges, indices: scratch.indices, slicing }
}

#[derive(Default)]
struct AssignScratch {
    spheres: Vec<([f64; 3], f64)>,
    boxes: Vec<Option<[u32; 6]>>,
    counts: Vec<u32>,
    ranges: Vec<[u32; 2]>,
    indices: Vec<u32>,
}

fn assign_into(lights: &[Light], view: &Matrix, projection: &Matrix, scratch: &mut AssignScratch) -> Slicing {
    scratch.spheres.clear();
    scratch.spheres.extend(lights.iter().map(|light| reach(light, view)));
    let farthest = scratch.spheres.iter().map(|(center, radius)| radius - center[2]).fold(1.0, f64::max);
    let slicing = slicing(projection, farthest);
    // Each light's box of clusters, if it reaches any.
    scratch.boxes.clear();
    for &(center, radius) in &scratch.spheres {
        let (closest, farthest) = (-(center[2] + radius), -(center[2] - radius));
        if farthest < slicing.near || closest > slicing.far {
            scratch.boxes.push(None);
            continue;
        }
        // The corners of its box, the near ones no closer than the near
        // plane, on the picture.
        let near_z = -closest.max(slicing.near);
        let far_z = center[2] - radius;
        let (mut low, mut high) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for corner in 0..8 {
            let x = center[0] + if corner & 1 == 0 { -radius } else { radius };
            let y = center[1] + if corner & 2 == 0 { -radius } else { radius };
            let z = if corner & 4 == 0 { near_z } else { far_z };
            let clip = transform(projection, [x, y, z, 1.0]);
            let ndc = [clip[0] / clip[3], clip[1] / clip[3]];
            for axis in 0..2 {
                low[axis] = low[axis].min(ndc[axis]);
                high[axis] = high[axis].max(ndc[axis]);
            }
        }
        if high[0] < -1.0 || low[0] > 1.0 || high[1] < -1.0 || low[1] > 1.0 {
            scratch.boxes.push(None);
            continue;
        }
        // As the shaders find a fragment's tile; picture rows go down
        // while clip-space y goes up.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped to the tiles")]
        let tile = |along: f64, tiles: u32| (along * f64::from(tiles)).floor().clamp(0.0, f64::from(tiles - 1)) as u32;
        scratch.boxes.push(Some([
            tile(low[0] * 0.5 + 0.5, TILES.0),
            tile(high[0] * 0.5 + 0.5, TILES.0),
            tile(0.5 - high[1] * 0.5, TILES.1),
            tile(0.5 - low[1] * 0.5, TILES.1),
            slicing.slice(closest.max(slicing.near)),
            slicing.slice(farthest.min(slicing.far)),
        ]));
    }
    scratch.counts.resize(CLUSTERS, 0);
    scratch.counts.fill(0);
    for bounds in scratch.boxes.iter().flatten() {
        let [x0, x1, y0, y1, z0, z1] = *bounds;
        for z in z0..=z1 {
            let z_off = z * TILES.1;
            for y in y0..=y1 {
                let row = (z_off + y) * TILES.0;
                for x in x0..=x1 {
                    let cluster = (row + x) as usize;
                    scratch.counts[cluster] = (scratch.counts[cluster] + 1).min(PER_CLUSTER);
                }
            }
        }
    }
    scratch.ranges.clear();
    let mut first = 0;
    for &count in &scratch.counts {
        scratch.ranges.push([first, 0]);
        first += count;
    }
    scratch.indices.resize(usize::try_from(first).unwrap_or(0), 0);
    for (light, bounds) in (0u32..).zip(&scratch.boxes) {
        let Some([x0, x1, y0, y1, z0, z1]) = *bounds else { continue };
        for z in z0..=z1 {
            let z_off = z * TILES.1;
            for y in y0..=y1 {
                let row = (z_off + y) * TILES.0;
                for x in x0..=x1 {
                    let cluster = (row + x) as usize;
                    let [start, taken] = &mut scratch.ranges[cluster];
                    if *taken < scratch.counts[cluster] {
                        scratch.indices[(*start + *taken) as usize] = light;
                        *taken += 1;
                    }
                }
            }
        }
    }
    slicing
}

/// What a frame's lights must be.
pub(crate) fn validate(lights: &[Light]) -> Result<()> {
    for light in lights {
        ensure!(light.radius > 0.0, "game returned a light without reach");
        ensure!(
            (-1.0..=1.0).contains(&light.outer) && (-1.0..=1.0).contains(&light.inner),
            "game returned a light whose cone is not a cosine"
        );
    }
    Ok(())
}

const RANGE_OFFSET_BITS: u32 = 20;
const PACKED_RANGE_BYTES: u64 = (CLUSTERS * size_of::<u32>()) as u64;
const _: () = assert!(CLUSTERS * (PER_CLUSTER as usize) < 1 << RANGE_OFFSET_BITS && CLUSTERS.is_multiple_of(4));

pub(crate) fn compact_bindings(device: &wgpu::Device) -> bool {
    // The scene's probe data and the volume's fog data also consume storage bindings.
    device.limits().max_storage_buffers_per_shader_stage < 7
}

/// Adapts the standard lighting declarations to fixed-size uniforms on low-limit devices.
pub(crate) fn shader_for_device(device: &wgpu::Device, code: String) -> String {
    if !compact_bindings(device) {
        return code;
    }
    let code = code.replace(
        "var<storage, read> local_shadow_faces: array<mat4x4f>;",
        &format!("var<uniform> local_shadow_faces: array<mat4x4f, {MAX_FACES}>;"),
    );
    let declaration = "var<storage, read> cluster_ranges: array<vec2u>;";
    if !code.contains(declaration) {
        return code;
    }
    let mut adapted = String::with_capacity(code.len());
    let mut rest = code.as_str();
    while let Some((prefix, access)) = rest.split_once("cluster_ranges[") {
        let mut depth = 1;
        let Some(end) = access.find(|c| {
            depth += i32::from(c == '[') - i32::from(c == ']');
            depth == 0
        }) else {
            break;
        };
        adapted.push_str(prefix);
        adapted.push_str("cluster_range_at(");
        adapted.push_str(&access[..end]);
        adapted.push(')');
        rest = &access[end + 1..];
    }
    adapted.push_str(rest);
    adapted.replace(
        declaration,
        &format!(
            "var<uniform> cluster_ranges: array<vec4u, {}>;
fn cluster_range_at(index: u32) -> vec2u {{
    let packed = cluster_ranges[index / 4u][index % 4u];
    return vec2u(packed & ((1u << {RANGE_OFFSET_BITS}u) - 1u), packed >> {RANGE_OFFSET_BITS}u);
}}",
            CLUSTERS / 4,
        ),
    )
}

/// The layout of group 2 of every mesh pipeline: the grid, the lights, each
/// cluster's range and the lights' indices.
pub(crate) fn group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lights"),
        entries: &layout_entries(compact_bindings(device)),
    })
}

/// The entries of that layout.
pub(crate) fn layout_entries(compact: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let constant = |binding, size| wgpu::BindGroupLayoutEntry {
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        ..storage(binding)
    };
    vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(GRID_BYTES),
            },
            count: None,
        },
        storage(1),
        if compact { constant(2, PACKED_RANGE_BYTES) } else { storage(2) },
        storage(3),
        // The local lights' shadow maps, their comparison sampler and the
        // matrix of each face (local_shadows.rs).
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
            count: None,
        },
        if compact { constant(6, FACE_BYTES) } else { storage(6) },
        // The compact layout's compute stage only needs point and spot lights.
        if compact {
            wgpu::BindGroupLayoutEntry { visibility: wgpu::ShaderStages::FRAGMENT, ..storage(7) }
        } else {
            storage(7)
        },
    ]
}

const GRID_BYTES: u64 = 48;
const FACE_BYTES: u64 = (MAX_FACES * 64) as u64;
const LIGHT_FLOATS: usize = 16;
const MAX_AREA_LIGHTS: usize = 64;
const AREA_MAT_OFFSET: usize = MAX_AREA_LIGHTS * 5;
const AREA_LUT_TEXELS: usize = (crate::area_lights::LTC_LUT_SIZE * crate::area_lights::LTC_LUT_SIZE) as usize;
const AREA_AMP_OFFSET: usize = AREA_MAT_OFFSET + AREA_LUT_TEXELS;
pub(crate) const MAX_AREA_COOKIES: usize = 8;
const AREA_COOKIE_DIM: u32 = 32;
const AREA_COOKIE_TEXELS: usize = (AREA_COOKIE_DIM * AREA_COOKIE_DIM) as usize;
const AREA_COOKIE_OFFSET: usize = AREA_AMP_OFFSET + AREA_LUT_TEXELS;
const AREA_BUFFER_BYTES: u64 = ((AREA_COOKIE_OFFSET + MAX_AREA_COOKIES * AREA_COOKIE_TEXELS) * 16) as u64;

const COOKIE_PACK_WGSL: &str = r#"
struct PackParams {
  slot: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};
@group(0) @binding(0) var<uniform> params: PackParams;
@group(0) @binding(1) var cookie_tex: texture_2d<f32>;
@group(0) @binding(2) var cookie_samp: sampler;
@group(0) @binding(3) var<storage, read_write> area_buf: array<vec4f>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= 32u || id.y >= 32u) {
    return;
  }
  let uv = (vec2f(id.xy) + vec2f(0.5)) / 32.0;
  let c = textureSampleLevel(cookie_tex, cookie_samp, uv, 0.0);
  area_buf[8512u + params.slot * 1024u + id.y * 32u + id.x] = c;
}
"#;

struct CookiePacker {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params: [wgpu::Buffer; MAX_AREA_COOKIES],
}

impl CookiePacker {
    fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("area_cookie_pack"),
            source: wgpu::ShaderSource::Wgsl(COOKIE_PACK_WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("area_cookie_pack_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("area_cookie_pack_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("area_cookie_pack"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("area_cookie_pack_samp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let params = std::array::from_fn(|slot| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("area_cookie_pack_params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: true,
            });
            let words = [u32::try_from(slot).unwrap_or(0), 0u32, 0u32, 0u32];
            buf.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytemuck::cast_slice(&words));
            buf.unmap();
            buf
        });
        Self { pipeline, bgl, sampler, params }
    }
}

/// The grid's uniform: its counts, its scales, whether its slices are linear,
/// and the active LTC area light count in `options.y`.
fn grid(lights: u32, area_lights: u32, slicing: &Slicing, size: (u32, u32)) -> [u32; 12] {
    let float = |value: f64| crate::matrix::narrow(value).to_bits();
    [
        TILES.0,
        TILES.1,
        SLICES,
        lights,
        float(f64::from(TILES.0) / f64::from(size.0.max(1))),
        float(f64::from(TILES.1) / f64::from(size.1.max(1))),
        float(slicing.scale),
        float(slicing.bias),
        float(if slicing.linear { 1.0 } else { 0.0 }),
        float(f64::from(area_lights)),
        0,
        0,
    ]
}

fn storage(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The buffers group 2 binds, grown as frames need, and a group without
/// lights for what another camera sees (insets) or a shadow pass draws (it
/// binds a stand-in atlas, as it draws into the real one).
pub(crate) struct Clusters {
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    area_only_uniform: wgpu::Buffer,
    lights: wgpu::Buffer,
    ranges: wgpu::Buffer,
    packed_ranges: Option<Vec<u32>>,
    indices: wgpu::Buffer,
    /// The matrix of each atlas layer's face, as the lights' shadows read it.
    faces: wgpu::Buffer,
    /// Packed LTC area lights (`0..320` vec4f), 64x64 LTC LUTs (`320..8512`), and 8x 32x32 cookies (`8512..16704`).
    area: wgpu::Buffer,
    cookie_packer: Option<CookiePacker>,
    cookie_ids: [u64; MAX_AREA_COOKIES],
    sampler: wgpu::Sampler,
    /// The local lights' shadow maps the group binds, and which texture they are.
    atlas: wgpu::TextureView,
    atlas_id: u64,
    group: wgpu::BindGroup,
    area_only_group: wgpu::BindGroup,
    empty: wgpu::BindGroup,
    data: Vec<f32>,
    face_data: Vec<f32>,
    area_data: Vec<crate::area_lights::AreaLight>,
    assign_scratch: AssignScratch,
}

impl Clusters {
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let uniform = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: GRID_BYTES,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                // A grid of no lights is zeros.
                mapped_at_creation: false,
            })
        };
        let compact = compact_bindings(device);
        let constants = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let lights = storage(device, "lights", 4 * LIGHT_FLOATS as u64);
        let indices = storage(device, "light indices", 4);
        let (ranges, faces) = if compact {
            (constants("light clusters", PACKED_RANGE_BYTES), constants("light shadow faces", FACE_BYTES))
        } else {
            (storage(device, "light clusters", CLUSTERS as u64 * 8), storage(device, "light shadow faces", FACE_BYTES))
        };
        let area = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("area lights + LTC LUTs"),
            size: AREA_BUFFER_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            type LtcLuts = (Vec<[f32; 4]>, Vec<[f32; 4]>);
            static LTC_LUTS: std::sync::OnceLock<LtcLuts> = std::sync::OnceLock::new();
            let (ltc_mat, ltc_amp) = LTC_LUTS.get_or_init(crate::area_lights::generate_ltc_luts);
            let mat_bytes: &[u8] = bytemuck::cast_slice(ltc_mat);
            let amp_bytes: &[u8] = bytemuck::cast_slice(ltc_amp);
            let mat_off = (AREA_MAT_OFFSET * 16) as u64;
            let amp_off = (AREA_AMP_OFFSET * 16) as u64;
            area.slice(mat_off..mat_off + mat_bytes.len() as u64)
                .get_mapped_range_mut()
                .unwrap()
                .copy_from_slice(mat_bytes);
            area.slice(amp_off..amp_off + amp_bytes.len() as u64)
                .get_mapped_range_mut()
                .unwrap()
                .copy_from_slice(amp_bytes);
        }
        area.unmap();
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("local shadows"),
            compare: Some(wgpu::CompareFunction::Less),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("local shadow stand-in"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let stand_in = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let empty_uniform = uniform("no lights");
        let empty =
            Self::bind(device, layout, &empty_uniform, &lights, &ranges, &indices, &stand_in, &sampler, &faces, &area);
        let area_only_uniform = uniform("area-only light grid");
        let area_only_group = Self::bind(
            device,
            layout,
            &area_only_uniform,
            &lights,
            &ranges,
            &indices,
            &stand_in,
            &sampler,
            &faces,
            &area,
        );
        let uniform = uniform("light grid");
        let group =
            Self::bind(device, layout, &uniform, &lights, &ranges, &indices, &stand_in, &sampler, &faces, &area);
        Self {
            layout: layout.clone(),
            uniform,
            area_only_uniform,
            lights,
            ranges,
            packed_ranges: compact.then(Vec::new),
            indices,
            faces,
            area,
            cookie_packer: None,
            cookie_ids: [0; MAX_AREA_COOKIES],
            sampler,
            atlas: stand_in.clone(),
            atlas_id: 0,
            group,
            area_only_group,
            empty,
            data: Vec::new(),
            face_data: vec![0.0; MAX_FACES * 16],
            area_data: Vec::new(),
            assign_scratch: AssignScratch::default(),
        }
    }

    #[allow(clippy::too_many_arguments, reason = "the group's resources")]
    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        lights: &wgpu::Buffer,
        ranges: &wgpu::Buffer,
        indices: &wgpu::Buffer,
        atlas: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        faces: &wgpu::Buffer,
        area: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: lights.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ranges.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(atlas) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry { binding: 6, resource: faces.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: area.as_entire_binding() },
        ];
        device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("lights"), layout, entries: &entries })
    }

    /// Packs `frame.area_lights` (up to `MAX_AREA_LIGHTS`) into the area-light storage region.
    fn upload_area_lights(&mut self, queue: &wgpu::Queue, frame: &RenderFrame, cookie_slots: &[Option<u32>]) -> u32 {
        use crate::area_lights::AreaLight;
        self.area_data.clear();
        for (idx, spec) in frame.area_lights.iter().take(MAX_AREA_LIGHTS).enumerate() {
            let pos = [spec.position.x, spec.position.y, spec.position.z];
            let right = [spec.right.x, spec.right.y, spec.right.z];
            let up = [spec.up.x, spec.up.y, spec.up.z];
            let gain = spec.intensity.max(0.0);
            let col = [spec.color.x * gain, spec.color.y * gain, spec.color.z * gain];
            let reach = spec.range.max(0.1);
            let mut light = match spec.kind {
                1 => AreaLight::disk(pos, right, up, (spec.width * 0.5).max(0.01), col, reach, spec.two_sided),
                2 => AreaLight::tube(pos, right, up, (spec.width * 0.5).max(0.01), spec.height.max(0.01), col, reach),
                3 => AreaLight::cookie_projector(
                    pos,
                    right,
                    up,
                    spec.width.clamp(0.05, 4.0),
                    (spec.width / spec.height.max(0.01)).clamp(0.1, 10.0),
                    col,
                    reach,
                    0.05,
                ),
                _ => AreaLight::rect(
                    pos,
                    right,
                    up,
                    (spec.width * 0.5).max(0.01),
                    (spec.height * 0.5).max(0.01),
                    col,
                    reach,
                    spec.two_sided,
                ),
            };
            light.flags[1] = cookie_slots.get(idx).and_then(|s| *s).map_or(-1.0, |s| s as f32);
            self.area_data.push(light);
        }
        if !self.area_data.is_empty() {
            queue.write_buffer(&self.area, 0, bytemuck::cast_slice(&self.area_data));
        }
        u32::try_from(self.area_data.len()).unwrap_or(0)
    }

    /// Downsamples changed cookie projector textures (32x32 RGBA) into `self.area` storage slots.
    pub fn encode_cookies(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        cookies: &[(u32, wgpu::TextureView, u64)],
    ) {
        if cookies.is_empty() {
            return;
        }
        let mut any_dirty = false;
        for &(slot, _, view_id) in cookies {
            if (slot as usize) < MAX_AREA_COOKIES && self.cookie_ids[slot as usize] != view_id {
                any_dirty = true;
                break;
            }
        }
        if !any_dirty {
            return;
        }
        let packer = self.cookie_packer.get_or_insert_with(|| CookiePacker::new(device));
        for &(slot, ref view, view_id) in cookies {
            let s = slot as usize;
            if s >= MAX_AREA_COOKIES || self.cookie_ids[s] == view_id {
                continue;
            }
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("area_cookie_pack_bg"),
                layout: &packer.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: packer.params[s].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&packer.sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: self.area.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("area_cookie_pack"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&packer.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(AREA_COOKIE_DIM.div_ceil(8), AREA_COOKIE_DIM.div_ceil(8), 1);
            self.cookie_ids[s] = view_id;
        }
    }

    /// Sorts `frame`'s lights into the clusters of its view, on a picture
    /// of `size` pixels. The lights `shadows` picked read the maps of
    /// `atlas` (a view of its layers, and which texture it is).
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &RenderFrame,
        size: (u32, u32),
        shadows: &Selection,
        atlas: Option<(&wgpu::TextureView, u64)>,
        cookie_slots: &[Option<u32>],
    ) {
        let area_count = self.upload_area_lights(queue, frame, cookie_slots);
        let mut area_only_words = [0u32; 12];
        area_only_words[9] = crate::matrix::narrow(f64::from(area_count)).to_bits();
        queue.write_buffer(&self.area_only_uniform, 0, bytemuck::cast_slice(&area_only_words));
        let count = u32::try_from(frame.lights.len()).unwrap_or(u32::MAX);
        if count == 0 {
            queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&area_only_words));
            return;
        }
        let slicing = assign_into(&frame.lights, &frame.view, &frame.projection, &mut self.assign_scratch);
        let words = grid(count, area_count, &slicing, size);
        let ranges = std::mem::take(&mut self.assign_scratch.ranges);
        let indices = std::mem::take(&mut self.assign_scratch.indices);
        self.upload_parts(device, queue, &frame.lights, shadows, atlas, &ranges, &indices, words);
        self.assign_scratch.ranges = ranges;
        self.assign_scratch.indices = indices;
    }

    /// Puts the lights that reach within `far` of `position` in one cluster
    /// that every fragment reads, for what is seen from a point all round
    /// (a reflection probe's capture), not through the frame's camera.
    pub fn update_one(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lights: &[Light],
        (position, far): ([f32; 3], f32),
        shadows: &Selection,
        atlas: Option<(&wgpu::TextureView, u64)>,
    ) {
        let count = u32::try_from(lights.len()).unwrap_or(u32::MAX);
        if count == 0 {
            queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&[0u32; 12]));
            return;
        }
        let reaching: Vec<u32> = (0u32..)
            .zip(lights)
            .filter(|(_, light)| {
                let apart =
                    [light.position.x - position[0], light.position.y - position[1], light.position.z - position[2]];
                let distance = (apart[0] * apart[0] + apart[1] * apart[1] + apart[2] * apart[2]).sqrt();
                distance <= light.radius + far
            })
            .map(|(index, _)| index)
            .take(PER_CLUSTER as usize)
            .collect();
        let assignment = Assignment {
            ranges: vec![[0, u32::try_from(reaching.len()).unwrap_or(0)]],
            indices: reaching,
            slicing: Slicing { near: 0.0, far: 0.0, linear: false, scale: 0.0, bias: 0.0 },
        };
        // One tile and one slice, whatever the fragment's position.
        let words = [1, 1, 1, count, 0, 0, 0, 0, 0, 0, 0, 0];
        self.upload(device, queue, lights, shadows, atlas, &assignment, words);
    }

    #[allow(clippy::too_many_arguments, reason = "what the buffers are filled from")]
    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lights: &[Light],
        shadows: &Selection,
        atlas: Option<(&wgpu::TextureView, u64)>,
        assignment: &Assignment,
        words: [u32; 12],
    ) {
        self.upload_parts(device, queue, lights, shadows, atlas, &assignment.ranges, &assignment.indices, words);
    }

    #[allow(clippy::too_many_arguments, reason = "what the buffers are filled from")]
    fn upload_parts(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lights: &[Light],
        shadows: &Selection,
        atlas: Option<(&wgpu::TextureView, u64)>,
        ranges: &[[u32; 2]],
        indices: &[u32],
        words: [u32; 12],
    ) {
        self.data.clear();
        for (index, light) in lights.iter().enumerate() {
            let (p, c, d) = (&light.position, &light.color, &light.direction);
            self.data.extend([p.x, p.y, p.z, light.radius, c.x, c.y, c.z, light.outer, d.x, d.y, d.z, light.inner]);
            // The light's shadow: its first layer (-1 for none), whether it is
            // a cube, and a face texel's width at unit depth.
            #[allow(clippy::cast_precision_loss, reason = "at most MAX_FACES")]
            match shadows.of(index) {
                Some(caster) => {
                    self.data.extend([caster.layer as f32, if caster.cube { 1.0 } else { 0.0 }, caster.texel, 0.0])
                }
                None => self.data.extend([-1.0, 0.0, 0.0, 0.0]),
            }
        }
        let light_bytes: &[u8] = bytemuck::cast_slice(&self.data);
        let index_bytes: &[u8] = bytemuck::cast_slice(indices);
        let mut grown = false;
        for (buffer, label, bytes) in
            [(&mut self.lights, "lights", light_bytes.len()), (&mut self.indices, "light indices", index_bytes.len())]
        {
            let bytes = bytes as u64;
            if bytes > buffer.size() {
                *buffer = storage(device, label, bytes.next_power_of_two());
                grown = true;
            }
        }
        if let Some((view, id)) = atlas.filter(|_| !shadows.casters.is_empty())
            && id != self.atlas_id
        {
            self.atlas = view.clone();
            self.atlas_id = id;
            grown = true;
        }
        if grown {
            self.group = Self::bind(
                device,
                &self.layout,
                &self.uniform,
                &self.lights,
                &self.ranges,
                &self.indices,
                &self.atlas,
                &self.sampler,
                &self.faces,
                &self.area,
            );
            self.area_only_group = Self::bind(
                device,
                &self.layout,
                &self.area_only_uniform,
                &self.lights,
                &self.ranges,
                &self.indices,
                &self.atlas,
                &self.sampler,
                &self.faces,
                &self.area,
            );
        }
        if !shadows.casters.is_empty() {
            self.face_data.fill(0.0);
            for caster in &shadows.casters {
                for face in 0..caster.faces() {
                    let at = (caster.layer + face) * 16;
                    self.face_data[at..at + 16].copy_from_slice(&shadows.faces[caster.first + face].matrix);
                }
            }
            queue.write_buffer(&self.faces, 0, bytemuck::cast_slice(&self.face_data));
        }
        queue.write_buffer(&self.lights, 0, light_bytes);
        if !index_bytes.is_empty() {
            queue.write_buffer(&self.indices, 0, index_bytes);
        }
        if let Some(packed) = &mut self.packed_ranges {
            packed.clear();
            packed.extend(ranges.iter().map(|&[start, count]| start | (count << RANGE_OFFSET_BITS)));
            queue.write_buffer(&self.ranges, 0, bytemuck::cast_slice(packed));
        } else {
            queue.write_buffer(&self.ranges, 0, bytemuck::cast_slice(ranges));
        }
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&words));
    }

    /// The frame's lights.
    pub fn group(&self) -> &wgpu::BindGroup {
        &self.group
    }

    /// Zero clustered point/spot lights while keeping LTC area lights & cookies active
    /// (used by the Tier-3 opaque pass when ReSTIR DI shades all point & spot lights).
    pub fn area_only_group(&self) -> &wgpu::BindGroup {
        &self.area_only_group
    }

    /// No lights, for what another camera sees.
    pub fn empty(&self) -> &wgpu::BindGroup {
        &self.empty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::IDENTITY;
    use pen_protocol::Vector;

    fn perspective(near: f32, far: f32) -> Matrix {
        let mut m = [0.0; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = far / (near - far);
        m[11] = -1.0;
        m[14] = near * far / (near - far);
        m
    }

    fn point(x: f32, y: f32, z: f32, radius: f32) -> Light {
        Light {
            position: Vector { x, y, z },
            radius,
            color: Vector { x: 1.0, y: 1.0, z: 1.0 },
            direction: Vector::default(),
            inner: -1.0,
            outer: -1.0,
            shadow: false,
            id: 0,
        }
    }

    fn clusters_with(assignment: &Assignment, light: u32) -> Vec<usize> {
        (0..CLUSTERS)
            .filter(|&cluster| {
                let [first, count] = assignment.ranges[cluster];
                assignment.indices[first as usize..(first + count) as usize].contains(&light)
            })
            .collect()
    }

    #[test]
    fn slices_the_depth_the_projection_sees() {
        let slicing = slicing(&perspective(0.1, 500.0), 1.0);
        // As near and far as f32 matrices say.
        assert!((slicing.near - 0.1).abs() < 1e-6 && (slicing.far - 500.0).abs() < 0.1, "{slicing:?}");
        assert_eq!(slicing.slice(0.1), 0);
        assert_eq!(slicing.slice(499.0), SLICES - 1);
        // Each slice is as many times deeper than the last.
        let middle = (0.1f64 * 500.0).sqrt();
        assert_eq!(slicing.slice(middle * 1.01), SLICES / 2);
        // An orthographic camera's slices are even; WebGPU's depth runs 0..1.
        let mut orthographic = IDENTITY;
        orthographic[10] = -0.01;
        orthographic[14] = 0.0;
        let even = super::slicing(&orthographic, 1.0);
        assert!(even.linear && even.near.abs() < 1e-3 + 1e-4 && (even.far - 100.0).abs() < 1e-3, "{even:?}");
        assert_eq!(even.slice(50.5), SLICES / 2);
    }

    #[test]
    fn a_light_lands_in_the_clusters_round_it() {
        let projection = perspective(0.1, 100.0);
        // Straight ahead, 10 deep, reaching 1: the middle tiles, a few slices.
        let lights = [point(0.0, 0.0, -10.0, 1.0), point(0.0, 0.0, 5.0, 1.0), point(-30.0, 0.0, -10.0, 1.0)];
        let assignment = assign(&lights, &IDENTITY, &projection);
        let near = clusters_with(&assignment, 0);
        assert!(!near.is_empty());
        let tiles = |cluster: usize| (cluster % 16, (cluster / 16) % 9, cluster / (16 * 9));
        let (x, y, z): (Vec<_>, Vec<_>, Vec<_>) = near.iter().map(|&cluster| tiles(cluster)).fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut xs, mut ys, mut zs), (x, y, z)| {
                xs.push(x);
                ys.push(y);
                zs.push(z);
                (xs, ys, zs)
            },
        );
        assert!(x.iter().all(|&x| (7..=8).contains(&x)), "{x:?}");
        assert!(y.iter().all(|&y| (4..=5).contains(&y)), "{y:?}");
        let slicing = assignment.slicing;
        let expected = slicing.slice(9.0)..=slicing.slice(11.0);
        assert!(z.iter().all(|z| expected.contains(&u32::try_from(*z).unwrap())), "{z:?} {expected:?}");
        // Behind the eye, and off to the side, it lights nothing.
        assert!(clusters_with(&assignment, 1).is_empty());
        assert!(clusters_with(&assignment, 2).is_empty());
    }

    #[test]
    fn every_lit_point_finds_its_light_in_its_cluster() {
        // Brute force: points ahead of the camera, each light that reaches
        // one must be in the point's cluster.
        let projection = perspective(0.1, 60.0);
        let mut view = IDENTITY;
        view[13] = -2.0;
        let mut lights = Vec::new();
        let mut seed = 7u32;
        let mut random = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f64::from(seed >> 8) / f64::from(1u32 << 24)
        };
        for index in 0..200 {
            #[allow(clippy::cast_possible_truncation, reason = "test coordinates")]
            let mut light = point(
                (random() * 40.0 - 20.0) as f32,
                (random() * 10.0 - 3.0) as f32,
                (-random() * 50.0) as f32,
                (0.5 + random() * 4.0) as f32,
            );
            if index % 3 == 0 {
                light.direction = Vector { x: 0.0, y: -1.0, z: 0.0 };
                light.outer = if index % 2 == 0 { 0.9 } else { 0.3 };
                light.inner = light.outer + 0.05;
            }
            lights.push(light);
        }
        let assignment = assign(&lights, &view, &projection);
        let grid = grid(200, 0, &assignment.slicing, (320, 180));
        let scale = (f64::from(f32::from_bits(grid[4])), f64::from(f32::from_bits(grid[5])));
        for _ in 0..4000 {
            let (sx, sy, depth) = (random() * 320.0, random() * 180.0, 0.2 + random() * 55.0);
            // The view-space point at that pixel and depth.
            let ndc = (sx / 160.0 - 1.0, 1.0 - sy / 90.0);
            let seen = [ndc.0 * depth, ndc.1 * depth, -depth];
            let world = [seen[0], seen[1] + 2.0, seen[2]];
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "test tiles")]
            let tile = ((sx * scale.0) as u32, (sy * scale.1) as u32);
            let cluster = ((assignment.slicing.slice(depth) * TILES.1 + tile.1) * TILES.0 + tile.0) as usize;
            let [first, count] = assignment.ranges[cluster];
            let found = &assignment.indices[first as usize..(first + count) as usize];
            for (index, light) in (0u32..).zip(&lights) {
                let to = [
                    f64::from(light.position.x) - world[0],
                    f64::from(light.position.y) - world[1],
                    f64::from(light.position.z) - world[2],
                ];
                let distance = to.iter().map(|value| value * value).sum::<f64>().sqrt();
                let inside_cone = light.outer <= -1.0 || to[1] / distance >= f64::from(light.outer);
                if distance < f64::from(light.radius) && inside_cone {
                    assert!(found.contains(&index), "light {index} misses the point at {world:?}");
                }
            }
        }
    }

    #[test]
    #[ignore = "a measurement; run with --ignored --nocapture in release"]
    fn clustering_time() {
        let projection = perspective(0.1, 200.0);
        for count in [64u16, 512, 4096] {
            let lights: Vec<Light> = (0..count)
                .map(|index| {
                    let at = f32::from(index);
                    point((at * 0.37).sin() * 40.0, 1.0, -5.0 - (at * 0.11).cos().abs() * 80.0, 2.0)
                })
                .collect();
            let started = std::time::Instant::now();
            let assignment = assign(&lights, &IDENTITY, &projection);
            eprintln!(
                "{count} lights: {:.2} ms, {} indices",
                started.elapsed().as_secs_f64() * 1000.0,
                assignment.indices.len()
            );
        }
    }

    #[test]
    fn refuses_lights_without_reach() {
        assert!(validate(&[point(0.0, 0.0, 0.0, 1.0)]).is_ok());
        assert!(validate(&[point(0.0, 0.0, 0.0, 0.0)]).is_err());
        let mut cone = point(0.0, 0.0, 0.0, 1.0);
        cone.outer = 2.0;
        assert!(validate(&[cone]).is_err());
    }
}
