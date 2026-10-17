// Drawing on a WebGL2-class device: wgpu's GL backend, in a browser without
// WebGPU. A device without storage buffers is taken to be one (`limited`),
// and on it:
//
// - An array a shader reads from a storage buffer (`var<storage, read> X:
//   array<T>`: sprites, decals) becomes a uniform array of as many elements as
//   a uniform binding holds, and a draw of more is split into windows of it,
//   each bound at its own offset (`Window`).
// - A depth texture a shader only loads from, which GLSL ES 3.0 cannot, is
//   declared as a float texture (`float_depth`), which a float copy of the
//   depth binds (`DepthCopy`).
// - Multisampled textures cannot be sampled, so the scene draws with one
//   sample a pixel; FXAA still smooths it, and insets, which are not sampled
//   multisampled, keep theirs.
// - A depth-stencil attachment cannot be read while attached, and GLSL ES
//   3.0 reads a depth texture only through a nearest sampler, so the passes
//   reading the scene's depth read a copy of it (`DepthCopy`); nor can a
//   stencil be sampled, so the aberration's marks are copied out of it by a
//   stencil-tested pass (`Marks`).
// - A texture cannot be viewed in another format, so an sRGB texture's linear
//   view (for normal maps) reads it decoded too.
use std::borrow::Cow;
use std::num::NonZeroU32;
use std::ops::Range;

const FULLSCREEN: &str = include_str!("shaders/fullscreen.wgsl");
const DEPTH_COPY_SHADER: &str = include_str!("shaders/depth_copy.wgsl");
const MARKS_SHADER: &str = include_str!("shaders/marks.wgsl");

/// Most bytes a uniform array standing in for a storage buffer takes, as big
/// as WebGL2's uniform blocks commonly are.
const MOST_ARRAY_BYTES: u64 = 1 << 16;

/// The stencil bit of surfaces the aberration splits (see `decals::DECAL_MARK`).
const ABERRATION_MARK: u32 = 1;

/// The format the aberration's marks are copied into.
pub(crate) const MARKS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Uint;

/// Whether `device` is WebGL2-class: it has no storage buffers.
pub(crate) fn limited(device: &wgpu::Device) -> bool {
    device.limits().max_storage_buffers_per_shader_stage == 0
}

/// The samples per pixel the scene draws with, of those `wanted`: one where
/// multisampled textures cannot be sampled.
pub(crate) fn samples(device: &wgpu::Device, wanted: u32) -> u32 {
    if limited(device) { 1 } else { wanted }
}

/// The formats a texture of `format` is also viewed in: an sRGB colour texture
/// (a compressed one too) is read as linear data too (normal maps), where the
/// device can.
pub(crate) fn view_formats(device: &wgpu::Device, format: wgpu::TextureFormat) -> Vec<wgpu::TextureFormat> {
    if format.is_srgb() && !limited(device) { vec![format.remove_srgb_suffix()] } else { Vec::new() }
}

/// The format a texture of `format` is read as linear data in.
pub(crate) fn linear_format(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::TextureFormat {
    view_formats(device, format).first().copied().unwrap_or(format)
}

/// Bytes of a uniform array standing in for a storage buffer: one binding's
/// worth, at most `MOST_ARRAY_BYTES`.
fn array_bytes(device: &wgpu::Device) -> u32 {
    let bytes = device.limits().max_uniform_buffer_binding_size.min(MOST_ARRAY_BYTES);
    u32::try_from(bytes).expect("at most 64 KiB")
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// How a limited device binds an array of `stride`-byte elements: `count` of
/// them at a time, from an element a multiple of `step` in, which puts the
/// binding's offset where uniform bindings may start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Window {
    pub count: u32,
    pub step: u32,
    pub stride: u32,
}

impl Window {
    /// The window `device` binds such an array through, or None if it has
    /// storage buffers.
    pub fn of(device: &wgpu::Device, stride: u32) -> Option<Window> {
        limited(device)
            .then(|| Window::new(array_bytes(device), device.limits().min_uniform_buffer_offset_alignment, stride))
    }

    fn new(bytes: u32, alignment: u32, stride: u32) -> Window {
        let window = Window { count: bytes / stride, step: alignment / gcd(alignment, stride), stride };
        debug_assert!(window.count >= window.step, "{window:?} cannot reach past a step");
        window
    }

    /// Bytes bound at once: the uniform array's size.
    pub fn bytes(self) -> u64 {
        u64::from(self.count) * u64::from(self.stride)
    }

    /// Elements a buffer holding `len` of them must have room for, so that
    /// the last window fits in it.
    pub fn room(self, len: usize) -> usize {
        len + self.count as usize
    }

    /// The draws of `instances` of the array: for each, the byte offset its
    /// window is bound at and its instances counted from there.
    pub fn draws(self, instances: Range<u32>) -> impl Iterator<Item = (u32, Range<u32>)> {
        let mut at = instances.start;
        std::iter::from_fn(move || {
            (at < instances.end).then(|| {
                let first = at - at % self.step;
                let end = instances.end.min(first + self.count);
                let draw = (first * self.stride, at - first..end - first);
                at = end;
                draw
            })
        })
    }
}

/// The binding of an array the vertex and fragment stages read: a read-only
/// storage buffer, or a `window` of a uniform one at a dynamic offset.
pub(crate) fn array_binding(window: Option<Window>) -> wgpu::BindingType {
    match window {
        None => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        Some(window) => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(window.bytes()),
        },
    }
}

/// The usage of a buffer holding such an array.
pub(crate) fn array_usage(window: Option<Window>) -> wgpu::BufferUsages {
    let usage = if window.is_some() { wgpu::BufferUsages::UNIFORM } else { wgpu::BufferUsages::STORAGE };
    usage | wgpu::BufferUsages::COPY_DST
}

/// A shader module of WGSL `code`; on a limited device, of the code as
/// `downleveled`.
pub(crate) fn shader_module(device: &wgpu::Device, label: Option<&str>, code: &str) -> wgpu::ShaderModule {
    let module = limited(device).then(|| downleveled(code, array_bytes(device))).flatten();
    let source = match module {
        Some(module) => wgpu::ShaderSource::Naga(Cow::Owned(module)),
        // Code that does not parse is wgpu's to report, as anywhere else.
        None => wgpu::ShaderSource::Wgsl(code.into()),
    };
    device.create_shader_module(wgpu::ShaderModuleDescriptor { label, source })
}

/// WGSL `code` as a limited device runs it: the depth textures it only loads
/// from are float textures, and its read-only storage arrays uniform arrays
/// of as many elements as `bytes` hold. None if it does not parse.
pub(crate) fn downleveled(code: &str, bytes: u32) -> Option<naga::Module> {
    let mut module = naga::front::wgsl::parse_str(&float_depth(code)).ok()?;
    uniform_arrays(&mut module, bytes);
    Some(module)
}

/// Makes each read-only storage array of `module` a uniform array of as many
/// elements as `bytes` hold. Its accesses read the same elements.
fn uniform_arrays(module: &mut naga::Module, bytes: u32) {
    for (_, global) in module.global_variables.iter_mut() {
        let naga::AddressSpace::Storage { access } = global.space else { continue };
        if access.contains(naga::StorageAccess::STORE) {
            continue;
        }
        let naga::TypeInner::Array { base, size: naga::ArraySize::Dynamic, stride } = module.types[global.ty].inner
        else {
            continue;
        };
        let Some(count) = NonZeroU32::new(bytes / stride) else { continue };
        let inner = naga::TypeInner::Array { base, size: naga::ArraySize::Constant(count), stride };
        global.ty = module.types.insert(naga::Type { name: None, inner }, naga::Span::UNDEFINED);
        global.space = naga::AddressSpace::Uniform;
    }
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Where `word` stands in `code` as a whole identifier.
fn whole_words<'a>(code: &'a str, word: &'a str) -> impl Iterator<Item = usize> + 'a {
    code.match_indices(word)
        .map(|(at, _)| at)
        .filter(move |&at| !code[..at].ends_with(is_word) && !code[at + word.len()..].starts_with(is_word))
}

/// The variable a type ends the declaration of: `before` ends with
/// `var NAME:`.
fn declared(before: &str) -> Option<&str> {
    let before = before.trim_end().strip_suffix(':')?.trim_end();
    let start = before.char_indices().rev().find(|&(_, c)| !is_word(c)).map_or(0, |(at, c)| at + c.len_utf8());
    let name = &before[start..];
    let keyword = before[..start].trim_end();
    (!name.is_empty() && keyword.ends_with("var") && !keyword[..keyword.len() - 3].ends_with(is_word)).then_some(name)
}

/// Where each `textureLoad(NAME, ...)` in `code` ends, after its `)`.
fn loads(code: &str, name: &str) -> Vec<usize> {
    whole_words(code, "textureLoad")
        .filter_map(|at| {
            let open = at + "textureLoad".len();
            let arguments = code[open..].trim_start().strip_prefix('(')?;
            let first = arguments.trim_start().strip_prefix(name)?;
            first.trim_start().strip_prefix(',')?;
            if first.starts_with(is_word) {
                return None;
            }
            let mut depth = 0;
            for (offset, c) in code[open..].char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' if depth == 1 => return Some(open + offset + 1),
                    ')' => depth -= 1,
                    _ => {}
                }
            }
            None
        })
        .collect()
}

/// `code` with each `texture_depth_2d` variable it reads only with
/// `textureLoad` declared as a `texture_2d<f32>`, whose loads give the depth
/// as x: GLSL ES 3.0 cannot load from a depth texture, and a float texture
/// binds the depth's float copy (or the depth view itself).
fn float_depth(code: &str) -> Cow<'_, str> {
    const DEPTH: &str = "texture_depth_2d";
    let mut edits: Vec<(Range<usize>, &str)> = Vec::new();
    for at in whole_words(code, DEPTH) {
        let Some(name) = declared(&code[..at]) else { continue };
        let loads = loads(code, name);
        // Its uses in code, not in comments, must be its declaration and loads.
        let in_comment = |at: usize| code[code[..at].rfind('\n').map_or(0, |start| start + 1)..at].contains("//");
        if whole_words(code, name).filter(|&at| !in_comment(at)).count() != loads.len() + 1 {
            continue;
        }
        edits.push((at..at + DEPTH.len(), "texture_2d<f32>"));
        edits.extend(loads.into_iter().map(|end| (end..end, ".x")));
    }
    if edits.is_empty() {
        return Cow::Borrowed(code);
    }
    edits.sort_by_key(|(range, _)| range.start);
    let mut rewritten = String::with_capacity(code.len() + 2 * edits.len());
    let mut done = 0;
    for (range, text) in edits {
        rewritten.push_str(&code[done..range.start]);
        rewritten.push_str(text);
        done = range.end;
    }
    rewritten.push_str(&code[done..]);
    Cow::Owned(rewritten)
}

fn fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    code: &str,
    layout: Option<&wgpu::PipelineLayout>,
    format: wgpu::TextureFormat,
    depth_stencil: Option<wgpu::DepthStencilState>,
) -> wgpu::RenderPipeline {
    let module = shader_module(device, Some(label), &code.replace("${fullscreen}", FULLSCREEN));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// A copy of the scene's depth, which a limited device's passes read for
/// it: it cannot read a depth-stencil texture while it is attached, even
/// read-only (decals), and GLSL ES 3.0 reads a depth texture only through a
/// nearest sampler, which the passes reading it have none of (ambient
/// occlusion, sky and fog). The copy is R32Float, exact for 24-bit depth.
pub(crate) struct DepthCopy {
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroupLayout,
    nearest: wgpu::Sampler,
    copy: Option<(wgpu::Texture, wgpu::TextureView)>,
}

impl DepthCopy {
    pub fn new(device: &wgpu::Device) -> Self {
        let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("depth copy"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("depth copy"),
            bind_group_layouts: &[Some(&group)],
            immediate_size: 0,
        });
        let format = wgpu::TextureFormat::R32Float;
        let pipeline = fullscreen_pipeline(device, "depth copy", DEPTH_COPY_SHADER, Some(&layout), format, None);
        Self { pipeline, group, nearest: device.create_sampler(&Default::default()), copy: None }
    }

    /// The copy of the depth of a `size` scene, made anew for a new size.
    pub fn target(&mut self, device: &wgpu::Device, size: (u32, u32)) -> wgpu::TextureView {
        if self.copy.as_ref().is_none_or(|(texture, _)| (texture.width(), texture.height()) != size) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("depth copy"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            self.copy = Some((texture, view));
        }
        self.copy.as_ref().unwrap().1.clone()
    }

    /// Copies `depth`, the scene's depth alone, into the copy `target` made.
    pub fn copy(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, depth: &wgpu::TextureView) {
        let (_, copy) = self.copy.as_ref().expect("DepthCopy::target comes first");
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.group,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.nearest) },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("depth copy"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: copy,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// The aberration's marks copied out of the scene's stencil into a
/// `MARKS_FORMAT` texture the present pass reads: a limited device cannot
/// sample a stencil.
pub(crate) struct Marks {
    pipeline: wgpu::RenderPipeline,
}

impl Marks {
    pub fn new(device: &wgpu::Device) -> Self {
        let test = wgpu::StencilFaceState {
            compare: wgpu::CompareFunction::Equal,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::Keep,
        };
        let depth_stencil = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState { front: test, back: test, read_mask: ABERRATION_MARK, write_mask: 0 },
            bias: Default::default(),
        };
        Self { pipeline: fullscreen_pipeline(device, "marks", MARKS_SHADER, None, MARKS_FORMAT, Some(depth_stencil)) }
    }

    /// Copies the marks of `depth`, the scene's depth and stencil, into
    /// `marks`: 1 where they are, 0 elsewhere.
    pub fn copy(&self, encoder: &mut wgpu::CommandEncoder, marks: &wgpu::TextureView, depth: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("marks"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: marks,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: None,
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_stencil_reference(ABERRATION_MARK);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_cover_every_instance_once_at_aligned_offsets() {
        // Sprites (64 bytes) in a WebGL2 uniform block, and decals (144) in
        // the smallest one WebGL2 allows.
        for (bytes, stride) in [(65536, 64), (16384, 144), (16384, 64)] {
            let window = Window::new(bytes, 256, stride);
            assert!(window.bytes() <= u64::from(bytes));
            for instances in [0..0, 0..1, 3..9, 5..2000, 1023..1025, 113..900] {
                let mut next = instances.start;
                for (offset, drawn) in window.draws(instances.clone()) {
                    assert_eq!(offset % 256, 0);
                    assert_eq!(offset % stride, 0);
                    assert!(drawn.start < drawn.end && drawn.end <= window.count);
                    assert_eq!(offset / stride + drawn.start, next);
                    next = offset / stride + drawn.end;
                    assert!((offset / stride + window.count) as usize <= window.room(instances.end as usize));
                }
                assert_eq!(next, instances.end);
            }
        }
        assert_eq!(Window::new(65536, 256, 64), Window { count: 1024, step: 4, stride: 64 });
        assert_eq!(Window::new(16384, 256, 144), Window { count: 113, step: 16, stride: 144 });
    }

    #[test]
    fn floats_the_depth_textures_only_loaded() {
        let code = "@group(0) @binding(2) var depth : texture_depth_2d;\n\
                    @group(0) @binding(3) var shadow: texture_depth_2d;\n\
                    fn at(p: vec2i) -> f32 { return textureLoad( depth ,p, min(0, 1)) + textureSampleCompare(shadow, s, \
                    vec2f(0.0), 0.5); }";
        assert_eq!(
            float_depth(code),
            "@group(0) @binding(2) var depth : texture_2d<f32>;\n\
             @group(0) @binding(3) var shadow: texture_depth_2d;\n\
             fn at(p: vec2i) -> f32 { return textureLoad( depth ,p, min(0, 1)).x + textureSampleCompare(shadow, s, \
             vec2f(0.0), 0.5); }"
        );
        // Read some other way too, or only declared: as it was.
        let passed = "var depth: texture_depth_2d;\nfn f() { g(depth); let d = textureLoad(depth, p, 0); }";
        assert!(matches!(float_depth(passed), Cow::Borrowed(_)));
        assert!(matches!(float_depth("var depthy: texture_depth_2d_array;"), Cow::Borrowed(_)));
    }

    #[test]
    fn storage_arrays_become_uniform_arrays() {
        let code = "struct Sprite { rect: vec4f, uv: vec4f, color: vec4f, parameters: vec4f };\n\
                    @group(0) @binding(1) var<storage,read> sprites: array<Sprite>;\n\
                    @group(0) @binding(2) var<storage, read_write> out: array<f32>;\n\
                    @vertex fn vertex(@builtin(instance_index) i: u32) -> @builtin(position) vec4f { return sprites[i].rect; }";
        let module = downleveled(code, 65536).unwrap();
        let spaces: Vec<_> = module.global_variables.iter().map(|(_, global)| (global.space, global.ty)).collect();
        assert_eq!(spaces[0].0, naga::AddressSpace::Uniform);
        assert!(matches!(
            module.types[spaces[0].1].inner,
            naga::TypeInner::Array { size: naga::ArraySize::Constant(count), stride: 64, .. } if count.get() == 1024
        ));
        // Written arrays are left as they are, for wgpu to refuse.
        assert!(matches!(spaces[1].0, naga::AddressSpace::Storage { .. }));
        assert!(downleveled("fn broken(", 65536).is_none());
    }

    /// Validates `code` as downleveled for a WebGL2 device (no features, the
    /// capabilities wgpu gives its GL backend) and translates each of its
    /// entry points into GLSL ES 3.0 as wgpu's WebGL2 backend does.
    fn webgl(name: &str, code: &str) -> Result<usize, String> {
        use naga::back::glsl;
        // The smallest uniform block WebGL2 allows.
        let module = downleveled(code, 16384).ok_or_else(|| format!("{name} does not parse"))?;
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        )
        .validate(&module)
        .map_err(|error| format!("{name}: {}", error.emit_to_string(&float_depth(code))))?;
        // Each kind of resource numbered from 0 in turn, as wgpu's layouts do.
        let mut binding_map = glsl::BindingMap::default();
        let mut counts = [0u8; 4];
        for (_, global) in module.global_variables.iter() {
            let Some(binding) = &global.binding else { continue };
            let kind = match (global.space, &module.types[global.ty].inner) {
                (naga::AddressSpace::Uniform, _) => 0,
                (naga::AddressSpace::Storage { .. }, _) => 1,
                (_, naga::TypeInner::Sampler { .. }) => 2,
                _ => 3,
            };
            binding_map.insert(*binding, counts[kind]);
            counts[kind] += 1;
        }
        let options = glsl::Options {
            version: glsl::Version::Embedded { version: 300, is_webgl: true },
            writer_flags: glsl::WriterFlags::ADJUST_COORDINATE_SPACE | glsl::WriterFlags::FORCE_POINT_SIZE,
            binding_map,
            zero_initialize_workgroup_memory: true,
        };
        for entry in &module.entry_points {
            let pipeline =
                glsl::PipelineOptions { shader_stage: entry.stage, entry_point: entry.name.clone(), multiview: None };
            let mut text = String::new();
            glsl::Writer::new(&mut text, &module, &info, &options, &pipeline, Default::default())
                .and_then(|mut writer| writer.write())
                .map_err(|error| format!("{name} {}: {error}", entry.name))?;
            // Storage blocks, which WebGL2 has none of, are laid out std430.
            if text.contains("std430") {
                return Err(format!("{name} {} still reads a storage buffer:\n{text}", entry.name));
            }
        }
        Ok(module.entry_points.len())
    }

    #[test]
    fn every_shader_translates_to_webgl2() {
        use crate::post::*;
        let toned = |code: &str| code.replace("${tonedWgsl}", TONED_WGSL);
        let fullscreen = |code: &str| code.replace("${fullscreen}", FULLSCREEN);
        // The renderer's own, as a limited device draws them: one sample a
        // pixel.
        let mut shaders = vec![
            ("occlusion".to_owned(), occlusion_shader(1)),
            ("atmosphere".to_owned(), atmosphere_shader(1)),
            ("present".to_owned(), present_shader(1, &crate::Display::SDR)),
            ("first bloom down".to_owned(), bloom_down_shader(true)),
            ("bloom down".to_owned(), bloom_down_shader(false)),
            ("bloom up".to_owned(), bloom_up_shader()),
            ("decals".to_owned(), crate::decals::shader(1)),
            ("mipmaps".to_owned(), include_str!("shaders/mipmaps.wgsl").to_owned()),
            ("veil".to_owned(), toned(include_str!("shaders/veil.wgsl"))),
            ("inset composite".to_owned(), toned(include_str!("shaders/composite.wgsl"))),
            ("depth copy".to_owned(), fullscreen(DEPTH_COPY_SHADER)),
            ("marks".to_owned(), fullscreen(MARKS_SHADER)),
            // The other lanes' passes: TAA, the colour filter and HDR decode.
            ("taa".to_owned(), crate::taa::shader(1)),
            ("colour filter".to_owned(), crate::filter::SHADER.to_owned()),
            ("decode".to_owned(), fullscreen(crate::display::DECODE)),
        ];
        // The game's material and sprite shaders: the bundle's, or the
        // sibling checkout's.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let game = [root.join("../../dist/gdev/assets"), root.join("../../../gdev/src/assets")]
            .into_iter()
            .find(|assets| assets.join("catalog.json").is_file() || assets.join("sprite.wgsl").is_file());
        match &game {
            Some(assets) => {
                let mut files: Vec<_> = std::fs::read_dir(assets)
                    .unwrap()
                    .map(|entry| entry.unwrap().path())
                    .filter(|path| path.extension().is_some_and(|extension| extension == "wgsl"))
                    .collect();
                files.sort();
                assert!(!files.is_empty(), "no shaders in {}", assets.display());
                for path in files {
                    let name = path.file_name().unwrap().to_string_lossy().into_owned();
                    shaders.push((name, std::fs::read_to_string(&path).unwrap()));
                }
            }
            None => eprintln!("skipped the game's shaders: no dist/gdev/assets (just gdev) or ../gdev checkout"),
        }
        let failures: Vec<String> = shaders.iter().filter_map(|(name, code)| webgl(name, code).err()).collect();
        assert!(failures.is_empty(), "{} of {} shaders fail:\n{}", failures.len(), shaders.len(), failures.join("\n"));
    }
}
