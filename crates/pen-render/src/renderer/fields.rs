// Fields: values the game keeps per material key and patches each frame, in
// one half-float texture each, which draws of that material read at binding
// 3 in place of the normal map (vertices too).
use pen_protocol::FieldPatch;

// Experimental control used to isolate field diffing from other host changes.
const DIRTY_FIELDS: bool = true;

use super::assets::whole;
use super::{State, extent};

/// The bits of the half float nearest `value`, ties rounded up.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "bit fields of a float"
)]
pub(crate) fn half_of(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xff) as i32 - 112;
    let mantissa = bits & 0x7fffff;
    if exponent >= 31 {
        return (sign | 0x7c00) as u16;
    }
    if exponent > 0 {
        return (sign | (((exponent as u32) << 10) + ((mantissa + 0x1000) >> 13))) as u16;
    }
    if exponent < -10 {
        return sign as u16;
    }
    (sign | ((((mantissa | 0x800000) >> (1 - exponent)) + 0x1000) >> 13)) as u16
}

/// CPU shadow of the uploaded half-float values. Comparison happens after
/// conversion: distinct F32 values can be the same actual texture sample.
#[derive(Default)]
pub(super) struct FieldShadow {
    width: usize,
    pixels: Vec<u16>,
    scratch: Vec<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Changed {
    x: u32,
    z: u32,
    columns: u32,
    rows: u32,
}

impl FieldShadow {
    fn new(width: u32, height: u32) -> Self {
        Self { width: width as usize, pixels: vec![0; width as usize * height as usize], scratch: Vec::new() }
    }

    fn apply(&mut self, patch: &FieldPatch) -> Option<Changed> {
        self.scratch.clear();
        self.scratch.extend(patch.values.iter().map(|&value| half_of(value)));
        let (mut left, mut top, mut right, mut bottom) = (patch.columns, patch.rows, 0, 0);
        for z in 0..patch.rows {
            let row = (patch.z + z) as usize * self.width + patch.x as usize;
            for x in 0..patch.columns {
                let value = self.scratch[z as usize * patch.columns as usize + x as usize];
                let destination = &mut self.pixels[row + x as usize];
                if *destination != value {
                    *destination = value;
                    left = left.min(x);
                    top = top.min(z);
                    right = right.max(x + 1);
                    bottom = bottom.max(z + 1);
                }
            }
        }
        (left < right && top < bottom).then_some(Changed {
            x: left,
            z: top,
            columns: right.saturating_sub(left),
            rows: bottom.saturating_sub(top),
        })
    }
}

impl State {
    /// Writes a patch into its material's field, made (zeroed) when first
    /// patched or when its size changes.
    pub(super) fn write_field(&mut self, material: &str, patch: &FieldPatch) -> (u64, u32) {
        let (width, height) = (patch.width.max(1), patch.height.max(1));
        let found = self.field_ids.get(material).copied();
        let index = match found {
            Some(index)
                if self.fields[index].1.texture.width() == width && self.fields[index].1.texture.height() == height =>
            {
                index
            }
            _ => {
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("field"),
                    size: extent(width, height),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let id = self.view_id();
                let field = whole(texture, id);
                match found {
                    Some(index) => {
                        let old = std::mem::replace(&mut self.fields[index].1, field);
                        self.fields[index].2 = FieldShadow::new(width, height);
                        self.forget_views(old.view_id);
                        index
                    }
                    None => {
                        self.fields.push((material.to_owned(), field, FieldShadow::new(width, height)));
                        self.field_ids.insert(material.to_owned(), self.fields.len() - 1);
                        self.fields.len() - 1
                    }
                }
            }
        };
        if patch.columns == 0 || patch.rows == 0 {
            return (0, 0);
        }
        let (_, field, shadow) = &mut self.fields[index];
        let changed = if DIRTY_FIELDS {
            let Some(changed) = shadow.apply(patch) else { return (0, 0) };
            changed
        } else {
            shadow.scratch.clear();
            shadow.scratch.extend(patch.values.iter().map(|&value| half_of(value)));
            Changed { x: 0, z: 0, columns: patch.columns, rows: patch.rows }
        };
        let first = changed.z as usize * patch.columns as usize + changed.x as usize;
        let last = first + (changed.rows - 1) as usize * patch.columns as usize + changed.columns as usize;
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &field.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: patch.x + changed.x, y: patch.z + changed.z, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&shadow.scratch[first..last]),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(patch.columns * 2),
                rows_per_image: Some(changed.rows),
            },
            extent(changed.columns, changed.rows),
        );
        (((last - first) * 2) as u64, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_round_half_way_up() {
        for (value, half) in [
            (0.0, 0x0000),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (0.5, 0x3800),
            (65504.0, 0x7bff),
            (1e6, 0x7c00),
            (6.1035156e-5, 0x0400),
            (3.0e-5, 0x01f7),
            (1e-9, 0x0000),
            (0.1, 0x2e66),
            (1.0009766, 0x3c01),
        ] {
            assert_eq!(half_of(value), half, "{value}");
        }
    }
    fn patch(width: u32, height: u32, x: u32, z: u32, columns: u32, rows: u32, values: Vec<f32>) -> FieldPatch {
        FieldPatch { material: 1, width, height, x, z, columns, rows, values }
    }

    #[test]
    fn field_shadow_skips_equal_halves_and_bounds_cropped_rows() {
        let mut shadow = FieldShadow::new(9, 7);
        let mut p = patch(9, 7, 2, 1, 5, 4, vec![0.0; 20]);
        assert_eq!(shadow.apply(&p), None);
        p.values[11] = 1.0;
        assert_eq!(shadow.apply(&p), Some(Changed { x: 1, z: 2, columns: 1, rows: 1 }));
        assert_eq!(shadow.apply(&p), None);
        p.values[11] = 1.00001;
        assert_eq!(shadow.apply(&p), None);
        p.values[0] = -0.0;
        p.values[19] = 2.0;
        assert_eq!(shadow.apply(&p), Some(Changed { x: 0, z: 0, columns: 5, rows: 4 }));
        p.values.fill(0.0);
        assert!(shadow.apply(&p).is_some());
        assert!(shadow.pixels.iter().all(|&value| value == 0));
    }

    #[test]
    fn cropped_field_uploads_match_full_reference_through_overlaps_and_restores() {
        let (width, height) = (27, 19);
        let mut shadow = FieldShadow::new(width, height);
        let mut uploaded = vec![0; width as usize * height as usize];
        let mut reference = uploaded.clone();
        let mut seed = 17_u32;
        let mut random = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };
        for step in 0..700 {
            let x = random() % width;
            let z = random() % height;
            let columns = 1 + random() % (width - x);
            let rows = 1 + random() % (height - z);
            let values = (0..columns * rows)
                .map(|_| if step % 19 == 0 { 0.0 } else { (random() % 4000) as f32 / 1000.0 })
                .collect();
            let p = patch(width, height, x, z, columns, rows, values);
            for row in 0..rows as usize {
                for column in 0..columns as usize {
                    reference[(z as usize + row) * width as usize + x as usize + column] =
                        half_of(p.values[row * columns as usize + column]);
                }
            }
            if let Some(changed) = shadow.apply(&p) {
                for row in changed.z..changed.z + changed.rows {
                    for column in changed.x..changed.x + changed.columns {
                        uploaded[(z + row) as usize * width as usize + (x + column) as usize] =
                            shadow.scratch[row as usize * columns as usize + column as usize];
                    }
                }
            }
            assert_eq!(uploaded, reference, "patch {step}");
            assert_eq!(shadow.pixels, reference);
            assert_eq!(shadow.apply(&p), None);
        }
    }

    #[test]
    fn field_dirty_uploads_reach_gpu_and_resize_invalidates_the_shadow() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter =
            pollster::block_on(instance.request_adapter(&Default::default())).expect("field regression requires a GPU");
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let root = std::env::temp_dir().join(format!("pen-field-test-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let mut state = State::new(
            device,
            queue,
            crate::RendererOptions {
                format: wgpu::TextureFormat::Bgra8Unorm,
                display: crate::Display::SDR,
                assets: root.clone(),
                veil: None,
                pipeline_cache: None,
            },
        )
        .unwrap();
        for (width, height) in [(9, 7), (9, 7), (13, 5)] {
            state.write_field("snow", &patch(width, height, 0, 0, width, height, vec![0.0; (width * height) as usize]));
            let p = patch(
                width,
                height,
                2,
                1,
                5,
                3,
                vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
            );
            state.write_field("snow", &p);
            state.write_field("snow", &p);
            let shader = state.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(
                    r#"
@group(0) @binding(0) var field: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> values: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(field);
  if (id.x < size.x * size.y) { values[id.x] = textureLoad(field, vec2i(i32(id.x % size.x), i32(id.x / size.x)), 0).r; }
}"#
                    .into(),
                ),
            });
            let pipeline = state.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let size = u64::from(width * height) * 4;
            let storage = state.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = state.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&state.fields[0].1.view),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: storage.as_entire_binding() },
                ],
            });
            let mut encoder = state.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups((width * height).div_ceil(64), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, size);
            state.queue.submit([encoder.finish()]);
            readback.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
            state.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let mapped = readback.slice(..).get_mapped_range().unwrap();
            let actual: Vec<u16> = bytemuck::cast_slice::<u8, f32>(&mapped).iter().map(|&v| half_of(v)).collect();
            let mut expected = vec![0; (width * height) as usize];
            for z in 0..p.rows {
                for x in 0..p.columns {
                    expected[((p.z + z) * width + p.x + x) as usize] = half_of(p.values[(z * p.columns + x) as usize]);
                }
            }
            assert_eq!(actual, expected);
            assert_eq!(state.fields.len(), 1);
            drop(mapped);
            readback.unmap();
        }
        drop(state);
        std::fs::remove_dir_all(root).unwrap();
    }
}
