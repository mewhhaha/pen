// The colour filter: an engine preference (ENGINE_PREFERENCE_COLOR_FILTER)
// for players with a colour vision deficiency, applied to the whole picture,
// the game's sprites and insets included. While one is on, the frame draws
// into a texture of the output's size and format, and a last pass copies it
// to the output through a 3x3 matrix in linear light: the output holds the
// screen's (sRGB-encoded) values, so the pass decodes, filters and encodes
// again. With no filter the frame draws straight to the output, as before.
//
// The matrices: Machado, Oliveira and Fernandes' simulation of protanopia,
// deuteranopia and tritanopia (2009, severity 1), and the correction made
// from each (Fidaner, Lin and Ozguven's daltonization): what the simulation
// loses, the difference between the colour and its simulation, is shifted
// into the channels the player still tells apart. Achromatopsia shows the
// picture's luminance (Rec. 709) as grey either way. `strength` blends from
// the picture as it is (0) to the full filter (1).
use pen_protocol::color_filter;

/// A 3x3 matrix, rows of linear RGB.
pub type Matrix3 = [[f32; 3]; 3];

const IDENTITY: Matrix3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// What a player without one kind of cone sees (Machado et al. 2009).
const PROTANOPIA: Matrix3 =
    [[0.152_286, 1.052_583, -0.204_868], [0.114_503, 0.786_281, 0.099_216], [-0.003_882, -0.048_116, 1.051_998]];
const DEUTERANOPIA: Matrix3 =
    [[0.367_322, 0.860_646, -0.227_968], [0.280_085, 0.672_501, 0.047_413], [-0.011_820, 0.042_940, 0.968_881]];
const TRITANOPIA: Matrix3 =
    [[1.255_528, -0.076_749, -0.178_779], [-0.078_411, 0.930_809, 0.147_602], [0.004_733, 0.691_367, 0.303_900]];
const LUMINANCE: Matrix3 = [[0.2126, 0.7152, 0.0722], [0.2126, 0.7152, 0.0722], [0.2126, 0.7152, 0.0722]];

/// Where each kind's lost difference goes: for a red or green deficiency,
/// into green and blue; for a blue one, into red and green.
const RED_GREEN_SHIFT: Matrix3 = [[0.0, 0.0, 0.0], [0.7, 1.0, 0.0], [0.7, 0.0, 1.0]];
const BLUE_SHIFT: Matrix3 = [[1.0, 0.0, 0.7], [0.0, 1.0, 0.7], [0.0, 0.0, 0.0]];

fn multiply(a: &Matrix3, b: &Matrix3) -> Matrix3 {
    std::array::from_fn(|row| std::array::from_fn(|column| (0..3).map(|k| a[row][k] * b[k][column]).sum()))
}

fn blend(a: &Matrix3, b: &Matrix3, amount: f32) -> Matrix3 {
    std::array::from_fn(|row| std::array::from_fn(|column| a[row][column] + (b[row][column] - a[row][column]) * amount))
}

/// A colour filter over the whole picture: which (COLOR_FILTER_*), how
/// much of it (0 to 1), and whether it shows what a player with that
/// deficiency sees instead of correcting for it (for developers checking
/// their colours).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ColorFilter {
    pub kind: u32,
    pub strength: f32,
    pub simulate: bool,
}

impl ColorFilter {
    /// The matrix it applies to linear RGB; None when it changes nothing.
    pub fn matrix(&self) -> Option<Matrix3> {
        let strength = if self.strength.is_finite() { self.strength.clamp(0.0, 1.0) } else { 0.0 };
        let (simulation, shift) = match self.kind {
            color_filter::PROTANOPIA => (PROTANOPIA, RED_GREEN_SHIFT),
            color_filter::DEUTERANOPIA => (DEUTERANOPIA, RED_GREEN_SHIFT),
            color_filter::TRITANOPIA => (TRITANOPIA, BLUE_SHIFT),
            color_filter::ACHROMATOPSIA => (LUMINANCE, IDENTITY),
            _ => return None,
        };
        if strength == 0.0 {
            return None;
        }
        let full = if self.simulate || self.kind == color_filter::ACHROMATOPSIA {
            simulation
        } else {
            // The colour plus its lost difference, shifted: I + E (I - S).
            let lost: Matrix3 = std::array::from_fn(|row| {
                std::array::from_fn(|column| IDENTITY[row][column] - simulation[row][column])
            });
            let shifted = multiply(&shift, &lost);
            std::array::from_fn(|row| std::array::from_fn(|column| IDENTITY[row][column] + shifted[row][column]))
        };
        Some(blend(&IDENTITY, &full, strength))
    }

    /// What the filter makes of one sRGB-encoded colour (each channel 0 to
    /// 1), as the pass does: for the audit and tests.
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let Some(matrix) = self.matrix() else { return rgb };
        let linear = rgb.map(decode);
        matrix.map(|row| encode((0..3).map(|k| row[k] * linear[k]).sum::<f32>().clamp(0.0, 1.0)))
    }
}

/// The sRGB transfer function and its inverse, as the pass's WGSL has them.
pub fn decode(value: f32) -> f32 {
    if value <= 0.04045 { value / 12.92 } else { ((value + 0.055) / 1.055).powf(2.4) }
}

pub fn encode(value: f32) -> f32 {
    if value <= 0.003_130_8 { value * 12.92 } else { 1.055 * value.powf(1.0 / 2.4) - 0.055 }
}

pub(crate) const SHADER: &str = r"
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<uniform> rows: array<vec4f, 3>;

@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
}

fn decode(value: vec3f) -> vec3f {
  return select(pow((value + 0.055) / 1.055, vec3f(2.4)), value / 12.92, value <= vec3f(0.04045));
}

fn encode(value: vec3f) -> vec3f {
  return select(1.055 * pow(value, vec3f(1.0 / 2.4)) - 0.055, value * 12.92, value <= vec3f(0.0031308));
}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let texel = textureLoad(source, vec2i(position.xy), 0);
  let linear = decode(texel.rgb);
  let filtered = vec3f(dot(rows[0].xyz, linear), dot(rows[1].xyz, linear), dot(rows[2].xyz, linear));
  return vec4f(encode(clamp(filtered, vec3f(0.0), vec3f(1.0))), texel.a);
}
";

/// The texture a filtered frame draws into, and the pass to the output.
struct Pass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    /// The texture of the output's size, and its bind group.
    target: Option<(wgpu::Texture, wgpu::BindGroup)>,
}

/// The renderer's colour filter: the one set, and its pass once used.
#[derive(Default)]
pub(crate) struct Filter {
    pub filter: ColorFilter,
    pass: Option<Pass>,
}

impl Filter {
    /// Where a frame of `size` draws, filtered: a texture the pass reads,
    /// or None while no filter is on (the frame draws to the output).
    pub fn begin(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        cache: Option<&wgpu::PipelineCache>,
        (width, height): (u32, u32),
    ) -> Option<wgpu::TextureView> {
        let matrix = self.filter.matrix()?;
        let pass = self.pass.get_or_insert_with(|| Pass::new(device, format, cache));
        let rows: Vec<f32> = matrix.iter().flat_map(|row| [row[0], row[1], row[2], 0.0]).collect();
        queue.write_buffer(&pass.uniform, 0, bytemuck::cast_slice(&rows));
        let fits =
            pass.target.as_ref().is_some_and(|(texture, _)| texture.width() == width && texture.height() == height);
        if !fits {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("colour filter"),
                size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("colour filter"),
                layout: &pass.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture.create_view(&Default::default())),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: pass.uniform.as_entire_binding() },
                ],
            });
            pass.target = Some((texture, group));
        }
        pass.target.as_ref().map(|(texture, _)| texture.create_view(&Default::default()))
    }

    /// Draws the filtered frame `begin` gave onto `output`.
    pub fn finish(&self, encoder: &mut wgpu::CommandEncoder, output: &wgpu::TextureView) {
        let Some(pass) = &self.pass else { return };
        let Some((_, group)) = &pass.target else { return };
        let mut render = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("colour filter"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        render.set_pipeline(&pass.pipeline);
        render.set_bind_group(0, group, &[]);
        render.draw(0..3, 0..1);
    }
}

impl Pass {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat, cache: Option<&wgpu::PipelineCache>) -> Pass {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("colour filter"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(48),
                    },
                    count: None,
                },
            ],
        });
        let module = crate::downlevel::shader_module(device, Some("colour filter"), SHADER);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("colour filter"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("colour filter"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: None,
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: None,
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("colour filter"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Pass { pipeline, layout, uniform, target: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-3)
    }

    #[test]
    fn no_filter_changes_nothing() {
        assert_eq!(ColorFilter::default().matrix(), None);
        let off = ColorFilter { kind: color_filter::DEUTERANOPIA, strength: 0.0, simulate: false };
        assert_eq!(off.matrix(), None);
        assert_eq!(off.apply([0.3, 0.6, 0.9]), [0.3, 0.6, 0.9]);
        let unknown = ColorFilter { kind: 99, strength: 1.0, simulate: false };
        assert_eq!(unknown.matrix(), None);
    }

    #[test]
    fn greys_stay_grey_and_white_stays_white() {
        // Each simulation keeps white white (Machado's rows sum to 1), and
        // so each correction keeps it too.
        for kind in [color_filter::PROTANOPIA, color_filter::DEUTERANOPIA, color_filter::TRITANOPIA] {
            for simulate in [false, true] {
                let filter = ColorFilter { kind, strength: 1.0, simulate };
                for grey in [0.0, 0.5, 1.0] {
                    assert!(close(filter.apply([grey; 3]), [grey; 3]), "{kind} {simulate} {grey}");
                }
            }
        }
    }

    #[test]
    fn simulation_confuses_red_and_green_and_correction_parts_them() {
        let red = [0.8, 0.25, 0.2];
        let green = [0.45, 0.55, 0.2];
        let distance = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        let seen = ColorFilter { kind: color_filter::DEUTERANOPIA, strength: 1.0, simulate: true };
        let corrected = ColorFilter { kind: color_filter::DEUTERANOPIA, strength: 1.0, simulate: false };
        // As a deuteranope sees them they lie closer than they are; after
        // correction, they lie farther apart for the deuteranope again.
        let apart = distance(red, green);
        let confused = distance(seen.apply(red), seen.apply(green));
        assert!(confused < apart * 0.6, "{confused} against {apart}");
        let parted = distance(seen.apply(corrected.apply(red)), seen.apply(corrected.apply(green)));
        assert!(parted > confused * 1.2, "{parted} against {confused}");
    }

    #[test]
    fn achromatopsia_shows_luminance_and_strength_blends() {
        let grey = ColorFilter { kind: color_filter::ACHROMATOPSIA, strength: 1.0, simulate: false };
        let [r, g, b] = grey.apply([1.0, 0.0, 0.0]);
        assert!((r - g).abs() < 1e-6 && (g - b).abs() < 1e-6);
        assert!((r - encode(0.2126)).abs() < 1e-3);
        let half = ColorFilter { strength: 0.5, ..grey };
        let matrix = half.matrix().unwrap();
        assert!((matrix[0][0] - (1.0 + 0.2126) / 2.0).abs() < 1e-6);
        assert!((matrix[0][1] - 0.7152 / 2.0).abs() < 1e-6);
    }

    #[test]
    fn the_transfer_functions_invert() {
        for step in 0..=20 {
            let value = step as f32 / 20.0;
            assert!((encode(decode(value)) - value).abs() < 1e-5, "{value}");
        }
    }
}
