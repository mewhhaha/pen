//! Weighted Blended Order-Independent Transparency (McGuire & Bavoil 2013).
//!
//! Renders overlapping transparent meshes, foliage, hair cards, and layered VFX
//! into dual offscreen targets (`accum: Rgba16Float`, `reveal: R8Unorm`) so
//! final pixels are strictly independent of draw submission order.

/// WGSL helper snippet for translucent fragment shaders targeting WBOIT.
pub const OIT_SHADER_HELPERS_WGSL: &str = r#"
struct OitOutput {
  @location(0) accum: vec4f,
  @location(1) reveal: f32,
};

// McGuire & Bavoil (2013) Equation 9 depth-aware weight function (compatible with forward-Z and reversed-Z).
fn oit_weight(linear_depth: f32, alpha: f32) -> f32 {
  let z = max(abs(linear_depth), 0.01);
  let w = alpha / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 6.0));
  return clamp(w, 1e-2, 3000.0);
}

fn oit_pack(rgb: vec3f, alpha: f32, linear_depth: f32) -> OitOutput {
  let a = clamp(alpha, 0.0, 1.0);
  let w = oit_weight(linear_depth, a);
  return OitOutput(vec4f(rgb * a, a) * w, a);
}
"#;

/// Returns the two `ColorTargetState` entries required for WBOIT translucent render pipelines:
/// - `@location(0) accum`: `Rgba16Float`, additive (`One + One`)
/// - `@location(1) reveal`: `R8Unorm`, multiplicative (`Zero + OneMinusSrc`)
pub fn oit_color_targets() -> [Option<wgpu::ColorTargetState>; 2] {
    [
        Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::R8Unorm,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            write_mask: wgpu::ColorWrites::RED,
        }),
    ]
}

/// Derives a WBOIT-compatible WGSL module with `@fragment fn fragment_oit(...) -> OitOutput`
/// from any standard Pen mesh shader declaring `@fragment fn fragment(...) -> @location(0) vec4f`.
pub fn derive_oit_wgsl(code: &str) -> Option<String> {
    if code.contains("fn fragment_oit") {
        return Some(code.to_owned());
    }
    // Locate `@fragment` followed by `fn fragment(`.
    let mut search = 0usize;
    let (frag_start, paren_open) = loop {
        let rel = code[search..].find("@fragment")?;
        let at = search + rel;
        let after_attr = &code[at + "@fragment".len()..];
        let trimmed = after_attr.trim_start();
        if let Some(after_fn) = trimmed.strip_prefix("fn") {
            let after_fn_ws = after_fn.trim_start();
            if after_fn_ws.len() < after_fn.len()
                && let Some(after_name) = after_fn_ws.strip_prefix("fragment")
            {
                let next_c = after_name.chars().next();
                if next_c.is_some_and(|c| c == '(' || c.is_whitespace()) {
                    let paren_rel = after_name.find('(')?;
                    let open_idx = code.len() - after_name.len() + paren_rel;
                    break (at, open_idx);
                }
            }
        }
        search = at + "@fragment".len();
    };

    // Match closing `)` of parameter list.
    let mut depth = 0i32;
    let mut paren_close = None;
    for (off, ch) in code[paren_open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    paren_close = Some(paren_open + off);
                    break;
                }
            }
            _ => {}
        }
    }
    let paren_close = paren_close?;
    let params_raw = code[paren_open + 1..paren_close].trim();

    // Find opening `{` of function body and verify return type is `@location(0) vec4f` or `vec4<f32>`.
    let after_params = &code[paren_close + 1..];
    let body_rel = after_params.find('{')?;
    let ret_clause = after_params[..body_rel].trim();
    if !ret_clause.contains("@location(0)") || !(ret_clause.contains("vec4f") || ret_clause.contains("vec4<f32>")) {
        return None;
    }
    let body_open = paren_close + 1 + body_rel;

    // Split top-level comma-separated parameters.
    let mut params = Vec::new();
    if !params_raw.is_empty() {
        let mut start = 0usize;
        let mut p_depth = 0i32;
        let mut a_depth = 0i32;
        for (i, ch) in params_raw.char_indices() {
            match ch {
                '(' => p_depth += 1,
                ')' => p_depth -= 1,
                '<' => a_depth += 1,
                '>' => a_depth -= 1,
                ',' if p_depth == 0 && a_depth == 0 => {
                    let item = params_raw[start..i].trim();
                    if !item.is_empty() {
                        params.push(item);
                    }
                    start = i + 1;
                }
                _ => {}
            }
        }
        let tail = params_raw[start..].trim();
        if !tail.is_empty() {
            params.push(tail);
        }
    }

    fn strip_wgsl_attrs(mut s: &str) -> &str {
        loop {
            s = s.trim_start();
            let Some(rest) = s.strip_prefix('@') else { break };
            let name_end = rest.find(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap_or(rest.len());
            let mut after = &rest[name_end..];
            let trimmed = after.trim_start();
            if let Some(in_paren) = trimmed.strip_prefix('(') {
                let mut d = 1i32;
                let mut close = None;
                for (i, ch) in in_paren.char_indices() {
                    match ch {
                        '(' => d += 1,
                        ')' => {
                            d -= 1;
                            if d == 0 {
                                close = Some(i);
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                let Some(c) = close else { break };
                after = &in_paren[c + 1..];
            }
            s = after;
        }
        s.trim()
    }

    fn find_struct_position_field(code: &str, struct_name: &str) -> Option<String> {
        let needle = format!("struct {struct_name}");
        let mut pos = 0usize;
        while let Some(rel) = code[pos..].find(&needle) {
            let at = pos + rel;
            let after = &code[at + needle.len()..];
            if after.chars().next().is_some_and(|c| c.is_alphanumeric() || c == '_') {
                pos = at + needle.len();
                continue;
            }
            let open = after.find('{')?;
            let rest = &after[open + 1..];
            let close = rest.find('}')?;
            let body = &rest[..close];
            if let Some(b_at) = body.find("@builtin(position)") {
                let field_decl = strip_wgsl_attrs(&body[b_at..]);
                let colon = field_decl.find(':')?;
                let field_name = field_decl[..colon].trim();
                if !field_name.is_empty() {
                    return Some(field_name.to_owned());
                }
            }
            return None;
        }
        None
    }

    let mut inner_params = Vec::with_capacity(params.len());
    let mut call_args = Vec::with_capacity(params.len());
    let mut pos_expr: Option<String> = None;

    for param in &params {
        let clean = strip_wgsl_attrs(param);
        let colon = clean.find(':')?;
        let name = clean[..colon].trim();
        let ty = clean[colon + 1..].trim();
        if name.is_empty() || ty.is_empty() {
            return None;
        }
        inner_params.push(format!("{name}: {ty}"));
        call_args.push(name.to_owned());
        if pos_expr.is_none() {
            if param.contains("@builtin(position)") {
                pos_expr = Some(format!("{name}.z"));
            } else if ty.chars().all(|c| c.is_alphanumeric() || c == '_')
                && let Some(field) = find_struct_position_field(code, ty)
            {
                pos_expr = Some(format!("{name}.{field}.z"));
            }
        }
    }

    let mut wrapper_params: Vec<String> = params.iter().map(|s| (*s).to_owned()).collect();
    let z_expr = match pos_expr {
        Some(expr) => expr,
        None => {
            wrapper_params.push("@builtin(position) _pen_oit_pos: vec4f".to_owned());
            "_pen_oit_pos.z".to_owned()
        }
    };

    let mut out = String::with_capacity(code.len() + OIT_SHADER_HELPERS_WGSL.len() + 384);
    out.push_str(&code[..frag_start]);
    out.push_str("fn _pen_oit_inner(");
    out.push_str(&inner_params.join(", "));
    out.push_str(") -> vec4f ");
    out.push_str(&code[body_open..]);
    out.push('\n');
    out.push_str(OIT_SHADER_HELPERS_WGSL);
    out.push_str("\n@fragment fn fragment_oit(");
    out.push_str(&wrapper_params.join(", "));
    out.push_str(") -> OitOutput {\n  let _pen_oit_c = _pen_oit_inner(");
    out.push_str(&call_args.join(", "));
    out.push_str(");\n  let _pen_oit_z = clamp(0.1 / max(1.0 - ");
    out.push_str(&z_expr);
    out.push_str(", 5e-4), 0.1, 200.0);\n  return oit_pack(_pen_oit_c.rgb, _pen_oit_c.a, _pen_oit_z);\n}\n");
    Some(out)
}

/// Offscreen accumulation + revealage targets and fullscreen composite pass for WBOIT.
pub struct OitPass {
    module: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    target_format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    msaa_pipelines: std::collections::HashMap<u32, wgpu::RenderPipeline>,
    layout: wgpu::BindGroupLayout,
    size: (u32, u32),
    samples: u32,
    accum_msaa: Option<(wgpu::Texture, wgpu::TextureView)>,
    reveal_msaa: Option<(wgpu::Texture, wgpu::TextureView)>,
    accum: Option<(wgpu::Texture, wgpu::TextureView)>,
    reveal: Option<(wgpu::Texture, wgpu::TextureView)>,
    bind_group: Option<wgpu::BindGroup>,
}

impl OitPass {
    fn build_composite_pipeline(
        device: &wgpu::Device,
        pl: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        target_format: wgpu::TextureFormat,
        samples: u32,
        cache: Option<&wgpu::PipelineCache>,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("oit_composite"),
            layout: Some(pl),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: samples.max(1), ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache,
        })
    }

    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat, cache: Option<&wgpu::PipelineCache>) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("oit_composite"),
            source: wgpu::ShaderSource::Wgsl(crate::deferred_wgsl!("oit_composite.wgsl").into()),
        });
        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("oit_composite_layout"),
            entries: &[tex_entry(0), tex_entry(1)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("oit_composite_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = Self::build_composite_pipeline(device, &pipeline_layout, &module, target_format, 1, cache);
        Self {
            module,
            pipeline_layout,
            target_format,
            pipeline,
            msaa_pipelines: std::collections::HashMap::new(),
            layout,
            size: (0, 0),
            samples: 1,
            accum_msaa: None,
            reveal_msaa: None,
            accum: None,
            reveal: None,
            bind_group: None,
        }
    }

    /// Ensures offscreen `accum` and `reveal` textures match `(width, height)` at 1x MSAA.
    pub fn resize(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        self.resize_with_samples(device, size, 1);
    }

    /// Ensures offscreen `accum` and `reveal` textures match `(width, height)` and `samples` (resolving MSAA to 1x).
    pub fn resize_with_samples(&mut self, device: &wgpu::Device, size: (u32, u32), samples: u32) {
        let size = (size.0.max(1), size.1.max(1));
        let samples = samples.max(1);
        if self.size == size && self.samples == samples && self.bind_group.is_some() {
            return;
        }
        let make = |label: &str, format: wgpu::TextureFormat, sample_count: u32, usage: wgpu::TextureUsages| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            (tex, view)
        };
        let resolved_usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let accum = make("oit_accum", wgpu::TextureFormat::Rgba16Float, 1, resolved_usage);
        let reveal = make("oit_reveal", wgpu::TextureFormat::R8Unorm, 1, resolved_usage);
        self.accum_msaa = (samples > 1).then(|| {
            make("oit_accum_msaa", wgpu::TextureFormat::Rgba16Float, samples, wgpu::TextureUsages::RENDER_ATTACHMENT)
        });
        self.reveal_msaa = (samples > 1).then(|| {
            make("oit_reveal_msaa", wgpu::TextureFormat::R8Unorm, samples, wgpu::TextureUsages::RENDER_ATTACHMENT)
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oit_composite_bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&accum.1) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&reveal.1) },
            ],
        });
        self.accum = Some(accum);
        self.reveal = Some(reveal);
        self.bind_group = Some(bg);
        self.size = size;
        self.samples = samples;
    }

    /// Returns the two color attachments clearing `accum` to `(0,0,0,0)` and `reveal` to `1.0`,
    /// automatically resolving multisampled targets into the 1x textures when `samples > 1`.
    pub fn attachments(&self) -> [Option<wgpu::RenderPassColorAttachment<'_>>; 2] {
        let accum_resolved = &self.accum.as_ref().expect("call resize before attachments").1;
        let reveal_resolved = &self.reveal.as_ref().expect("call resize before attachments").1;
        match (&self.accum_msaa, &self.reveal_msaa) {
            (Some(a_msaa), Some(r_msaa)) => [
                Some(wgpu::RenderPassColorAttachment {
                    view: &a_msaa.1,
                    depth_slice: None,
                    resolve_target: Some(accum_resolved),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Discard,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &r_msaa.1,
                    depth_slice: None,
                    resolve_target: Some(reveal_resolved),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Discard,
                    },
                }),
            ],
            _ => [
                Some(wgpu::RenderPassColorAttachment {
                    view: accum_resolved,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: reveal_resolved,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
        }
    }

    /// Composites accumulated OIT transparency over `scene_view` (loading existing opaque pixels).
    pub fn composite(&self, encoder: &mut wgpu::CommandEncoder, scene_view: &wgpu::TextureView) {
        let Some(bg) = &self.bind_group else { return };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("oit_composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: scene_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Composites accumulated OIT transparency into a color attachment matching `samples` (1x or MSAA).
    pub fn composite_attachment(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        attachment: wgpu::RenderPassColorAttachment<'_>,
        samples: u32,
    ) {
        let Some(bg) = &self.bind_group else { return };
        let pipeline = if samples <= 1 {
            &self.pipeline
        } else {
            let (pl, module, fmt) = (&self.pipeline_layout, &self.module, self.target_format);
            self.msaa_pipelines
                .entry(samples)
                .or_insert_with(|| Self::build_composite_pipeline(device, pl, module, fmt, samples, None))
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("oit_composite"),
            color_attachments: &[Some(attachment)],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_wboit_produces_identical_pixels_regardless_of_draw_order() {
        let Some((device, queue)) = crate::testing::device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let quad_wgsl = format!(
            "{OIT_SHADER_HELPERS_WGSL}
            struct Quad {{ color: vec4f, depth: f32, _p0: f32, _p1: f32, _p2: f32 }};
            @group(0) @binding(0) var<uniform> q: Quad;
            @vertex fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {{
              let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
              let y = 1.0 - f32(idx & 2u) * 2.0;
              return vec4f(x, y, 0.5, 1.0);
            }}
            @fragment fn fs() -> OitOutput {{
              return oit_pack(q.color.rgb, q.color.a, q.depth);
            }}"
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("oit_test_quad"),
            source: wgpu::ShaderSource::Wgsl(quad_wgsl.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(32),
                },
                count: None,
            }],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("oit_quad_pipe"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &oit_color_targets(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let make_quad = |rgba: [f32; 4], depth: f32| {
            let data = [rgba[0], rgba[1], rgba[2], rgba[3], depth, 0.0, 0.0, 0.0];
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&data));
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
            })
        };
        let red_near = make_quad([1.0, 0.1, 0.0, 0.5], 5.0);
        let blue_far = make_quad([0.0, 0.2, 1.0, 0.6], 20.0);

        let mut oit = OitPass::new(&device, wgpu::TextureFormat::Rgba8Unorm, None);
        oit.resize(&device, (8, 8));

        let render_order = |first: &wgpu::BindGroup, second: &wgpu::BindGroup| -> [u8; 4] {
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = target.create_view(&Default::default());
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let _clear = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
            }
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("oit_accum_pass"),
                    color_attachments: &oit.attachments(),
                    ..Default::default()
                });
                pass.set_pipeline(&pipe);
                pass.set_bind_group(0, first, &[]);
                pass.draw(0..3, 0..1);
                pass.set_bind_group(0, second, &[]);
                pass.draw(0..3, 0..1);
            }
            oit.composite(&mut enc, &view);

            let rb = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256 * 8,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            enc.copy_texture_to_buffer(
                target.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &rb,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(8),
                    },
                },
                wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 },
            );
            queue.submit([enc.finish()]);
            let slice = rb.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range().unwrap();
            let px = [data[0], data[1], data[2], data[3]];
            drop(data);
            rb.unmap();
            px
        };

        let ab = render_order(&red_near, &blue_far);
        let ba = render_order(&blue_far, &red_near);
        assert_eq!(ab, ba, "WBOIT must be independent of draw submission order: {ab:?} vs {ba:?}");
        // Nearer red quad should dominate far blue quad due to depth weighting.
        assert!(ab[0] > ab[2], "nearer red quad should dominate far blue quad: {ab:?}");
    }

    #[test]
    fn derive_oit_wgsl_handles_struct_and_direct_fragment_inputs() {
        let with_struct = r#"
struct Out { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@fragment fn fragment(i: Out) -> @location(0) vec4f {
  return vec4f(i.uv, 0.0, 0.5);
}
"#;
        let derived = derive_oit_wgsl(with_struct).expect("struct fragment must derive OIT");
        assert!(derived.contains("fn _pen_oit_inner(i: Out) -> vec4f"));
        assert!(derived.contains("@fragment fn fragment_oit(i: Out) -> OitOutput"));
        assert!(derived.contains("i.pos.z"));
        assert!(!derived.contains("_pen_oit_pos"));

        let without_pos = r#"
@fragment fn fragment(@location(0) uv: vec2f) -> @location(0) vec4<f32> {
  return vec4f(uv, 1.0, 0.5);
}
"#;
        let derived2 = derive_oit_wgsl(without_pos).expect("direct fragment must derive OIT");
        assert!(derived2.contains("fn _pen_oit_inner(uv: vec2f) -> vec4f"));
        assert!(derived2.contains("@builtin(position) _pen_oit_pos: vec4f"));
        assert!(derived2.contains("_pen_oit_pos.z"));
    }
}
