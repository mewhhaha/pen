//! Self-contained GPU overlay HUD for `pen edit`:
//! Renders translucent toolbar, hierarchy tree, property inspector, timeline scrubber,
//! asset browser, crisp built-in 8x8 ASCII bitmap text, and 3D-projected X/Y/Z gizmo arrows
//! directly over Pen's 3D viewport with zero external font files required.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "UI pixel coordinates and atlas packing")]

use pen_scene::{Component, NodeId};

use crate::editor::EditorState;
use crate::editor::gizmo::gizmo_axes;

#[repr(C)]
#[derive(Clone, Copy)]
struct HudVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

fn hud_vertices_bytes(verts: &[HudVertex]) -> Vec<u8> {
    let mut out = Vec::with_capacity(verts.len() * std::mem::size_of::<HudVertex>());
    for v in verts {
        for f in [v.pos[0], v.pos[1], v.uv[0], v.uv[1], v.color[0], v.color[1], v.color[2], v.color[3]] {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
    out
}

const HUD_WGSL: &str = r#"
struct HudIn {
  @location(0) pos: vec2f,
  @location(1) uv: vec2f,
  @location(2) color: vec4f,
};
struct HudOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec4f,
};
@group(0) @binding(0) var font_tex: texture_2d<f32>;
@group(0) @binding(1) var font_samp: sampler;

@vertex fn vs(v: HudIn) -> HudOut {
  var o: HudOut;
  o.clip = vec4f(v.pos, 0.0, 1.0);
  o.uv = v.uv;
  o.color = v.color;
  return o;
}

@fragment fn fs(i: HudOut) -> @location(0) vec4f {
  let mask = textureSampleLevel(font_tex, font_samp, i.uv, 0.0).r;
  return vec4f(i.color.rgb, i.color.a * mask);
}
"#;

/// Compact 5x7 uppercase + digit + punctuation bitmap glyphs (stored in a 128x64 R8Unorm atlas).
fn glyph_bits(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x13, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x0E, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x0E],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '+' => [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x08],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        '/' => [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10],
        '[' => [0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E],
        ']' => [0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        '|' => [0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        '*' => [0x00, 0x15, 0x0E, 0x1F, 0x0E, 0x15, 0x00],
        '_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F],
        _ => [0x00; 7],
    }
}

/// Interactive click regions emitted during HUD layout for mouse hit-testing.
#[derive(Clone, Debug)]
pub struct HudClickZones {
    pub hierarchy_rows: Vec<(f32, f32, f32, f32, NodeId)>,
    pub timeline_bar: [f32; 4], // x, y, w, h
}

/// GPU-accelerated editor HUD & 3D gizmo overlay renderer.
pub struct EditorHud {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buf: wgpu::Buffer,
    vertex_cap: usize,
}

impl EditorHud {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        // 16 cols x 8 rows of 8x8 glyph cells (128x64). Cell (0,0) = solid white pixel for rects/lines!
        const ATLAS_W: u32 = 128;
        const ATLAS_H: u32 = 64;
        let mut pixels = vec![0u8; (ATLAS_W * ATLAS_H) as usize];
        // Cell 0 (code 32 space unused -> fill solid white block at cell 0 for untextured rects!).
        for y in 0..8 {
            for x in 0..8 {
                pixels[(y * ATLAS_W + x) as usize] = 255;
            }
        }
        for code in 33u8..127 {
            let cell = u32::from(code - 32);
            let cx = (cell % 16) * 8;
            let cy = (cell / 16) * 8;
            let rows = glyph_bits(code as char);
            for (ry, &bits) in rows.iter().enumerate() {
                for rx in 0..5u32 {
                    if ((bits >> (4 - rx)) & 1) != 0 {
                        pixels[((cy + ry as u32) * ATLAS_W + (cx + rx + 1)) as usize] = 255;
                    }
                }
            }
        }

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("editor_hud_font"),
            size: wgpu::Extent3d { width: ATLAS_W, height: ATLAS_H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            tex.as_image_copy(),
            &pixels,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(ATLAS_W), rows_per_image: Some(ATLAS_H) },
            wgpu::Extent3d { width: ATLAS_W, height: ATLAS_H, depth_or_array_layers: 1 },
        );
        let view = tex.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("editor_hud_samp"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("editor_hud_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("editor_hud_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("editor_hud_wgsl"),
            source: wgpu::ShaderSource::Wgsl(HUD_WGSL.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("editor_hud_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("editor_hud_pipe"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<HudVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4],
                })],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let vertex_cap = 16384;
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("editor_hud_vb"),
            size: (vertex_cap * std::mem::size_of::<HudVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, bind_group, vertex_buf, vertex_cap }
    }

    /// Renders the complete editor UI (toolbar, hierarchy, inspector, timeline, asset browser, and 3D gizmos)
    /// onto `target_view` using LoadOp::Load so the 3D viewport scene shines underneath.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        size: (u32, u32),
        state: &EditorState,
    ) -> HudClickZones {
        let wf = size.0.max(1) as f32;
        let hf = size.1.max(1) as f32;
        let mut verts: Vec<HudVertex> = Vec::with_capacity(4096);
        let to_ndc = |x: f32, y: f32| -> [f32; 2] { [(x / wf) * 2.0 - 1.0, 1.0 - (y / hf) * 2.0] };

        let push_quad = |verts: &mut Vec<HudVertex>, x: f32, y: f32, w: f32, h: f32, uv0: [f32; 2], uv1: [f32; 2], color: [f32; 4]| {
            let p0 = to_ndc(x, y);
            let p1 = to_ndc(x + w, y);
            let p2 = to_ndc(x + w, y + h);
            let p3 = to_ndc(x, y + h);
            verts.push(HudVertex { pos: p0, uv: uv0, color });
            verts.push(HudVertex { pos: p1, uv: [uv1[0], uv0[1]], color });
            verts.push(HudVertex { pos: p2, uv: uv1, color });
            verts.push(HudVertex { pos: p0, uv: uv0, color });
            verts.push(HudVertex { pos: p2, uv: uv1, color });
            verts.push(HudVertex { pos: p3, uv: [uv0[0], uv1[1]], color });
        };

        let push_rect = |verts: &mut Vec<HudVertex>, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]| {
            push_quad(verts, x, y, w, h, [0.02, 0.02], [0.04, 0.04], color);
        };

        let push_line = |verts: &mut Vec<HudVertex>, a: [f32; 2], b: [f32; 2], thickness: f32, color: [f32; 4]| {
            let dx = b[0] - a[0];
            let dy = b[1] - a[1];
            let len = (dx * dx + dy * dy).sqrt();
            if len < 0.5 {
                return;
            }
            let nx = -dy / len * (thickness * 0.5);
            let ny = dx / len * (thickness * 0.5);
            let p0 = to_ndc(a[0] + nx, a[1] + ny);
            let p1 = to_ndc(b[0] + nx, b[1] + ny);
            let p2 = to_ndc(b[0] - nx, b[1] - ny);
            let p3 = to_ndc(a[0] - nx, a[1] - ny);
            let uv = [0.03, 0.03];
            verts.push(HudVertex { pos: p0, uv, color });
            verts.push(HudVertex { pos: p1, uv, color });
            verts.push(HudVertex { pos: p2, uv, color });
            verts.push(HudVertex { pos: p0, uv, color });
            verts.push(HudVertex { pos: p2, uv, color });
            verts.push(HudVertex { pos: p3, uv, color });
        };

        let push_text = |verts: &mut Vec<HudVertex>, x: f32, y: f32, scale: f32, text: &str, color: [f32; 4]| {
            let mut cx = x;
            let cw = 8.0 * scale;
            let ch = 8.0 * scale;
            for ch_val in text.chars() {
                if ch_val != ' ' && ch_val.is_ascii() {
                    let b = ch_val.to_ascii_uppercase() as u8;
                    if (33..127).contains(&b) {
                        let cell = u32::from(b - 32);
                        let gx = (cell % 16) as f32 * 8.0;
                        let gy = (cell / 16) as f32 * 8.0;
                        push_quad(
                            verts,
                            cx,
                            y,
                            cw,
                            ch,
                            [gx / 128.0, gy / 64.0],
                            [(gx + 8.0) / 128.0, (gy + 8.0) / 64.0],
                            color,
                        );
                    }
                }
                cx += 7.0 * scale;
            }
        };

        // 1. 3D-projected X/Y/Z Interactive Transform Gizmo on selected node.
        if let Some(sel_id) = state.selected
            && let Some(rnode) = state.resolved.nodes.iter().find(|n| n.id == sel_id)
        {
            let pivot = [rnode.world_matrix[12], rnode.world_matrix[13], rnode.world_matrix[14]];
            let arm_len = (state.camera.distance * 0.18).clamp(0.6, 5.0);
            let axes = gizmo_axes(&rnode.local_transform, state.gizmo_space);
            if let Some(p0) = state.camera.project_point(pivot, wf, hf) {
                let colors = [
                    [0.95, 0.25, 0.25, 0.95], // +X red
                    [0.25, 0.92, 0.35, 0.95], // +Y green
                    [0.25, 0.55, 0.98, 0.95], // +Z blue
                ];
                let labels = ["X", "Y", "Z"];
                for i in 0..3 {
                    let tip = [
                        pivot[0] + axes[i][0] * arm_len,
                        pivot[1] + axes[i][1] * arm_len,
                        pivot[2] + axes[i][2] * arm_len,
                    ];
                    if let Some(p1) = state.camera.project_point(tip, wf, hf) {
                        push_line(&mut verts, p0, p1, 3.0, colors[i]);
                        push_rect(&mut verts, p1[0] - 4.0, p1[1] - 4.0, 8.0, 8.0, colors[i]);
                        push_text(&mut verts, p1[0] + 6.0, p1[1] - 4.0, 1.2, labels[i], colors[i]);
                    }
                }
            }
        }

        // 2. Top Toolbar.
        let panel_bg = [0.08, 0.09, 0.12, 0.86];
        let border = [0.22, 0.26, 0.34, 0.95];
        let text_hi = [0.92, 0.94, 0.98, 1.0];
        let text_dim = [0.64, 0.70, 0.78, 1.0];
        let accent = [0.24, 0.60, 0.98, 0.92];

        push_rect(&mut verts, 0.0, 0.0, wf, 32.0, panel_bg);
        push_rect(&mut verts, 0.0, 31.0, wf, 1.0, border);
        let tm_name = match state.scene.environment.tonemap {
            1 => "AGX",
            2 => "ACES",
            3 => "PBRN",
            _ => "SHLD",
        };
        let play_tag = if state.is_playing() { "PLAYING" } else { "EDIT" };
        let dirty_star = if state.dirty { "*" } else { "" };
        let undo_tag = match (state.history.can_undo(), state.history.can_redo()) {
            (true, true) => "U/R",
            (true, false) => "U/-",
            (false, true) => "-/R",
            (false, false) => "-/-",
        };
        let bar = format!(
            "PEN EDIT | {}{} | {} | [W/E/R] {} [G] {} | [T] {} | [P] {} | [CTRL+S] SAVE [Z/Y] {}",
            state.scene.name,
            dirty_star,
            play_tag,
            state.gizmo_mode.label(),
            state.gizmo_space.label(),
            tm_name,
            if state.is_playing() { "STOP" } else { "PLAY" },
            undo_tag,
        );
        push_text(&mut verts, 10.0, 10.0, 1.35, &bar, text_hi);

        // 3. Left Hierarchy Tree Panel.
        let left_w = 220.0f32.min(wf * 0.25);
        let bottom_h = 62.0;
        let body_h = (hf - 32.0 - bottom_h).max(0.0);
        push_rect(&mut verts, 0.0, 32.0, left_w, body_h, panel_bg);
        push_rect(&mut verts, left_w - 1.0, 32.0, 1.0, body_h, border);
        push_text(&mut verts, 10.0, 42.0, 1.3, &format!("HIERARCHY ({})", state.scene.nodes.len()), accent);

        let mut hierarchy_rows = Vec::new();
        for (i, node) in state.scene.nodes.iter().take(24).enumerate() {
            let ry = 62.0 + i as f32 * 20.0;
            if ry + 18.0 > 32.0 + body_h {
                break;
            }
            let is_sel = state.selected == Some(node.id);
            if is_sel {
                push_rect(&mut verts, 4.0, ry - 2.0, left_w - 8.0, 18.0, [0.18, 0.36, 0.62, 0.85]);
            }
            let indent = if node.parent.is_some() { "  " } else { "" };
            let vis = if node.visible { "+" } else { "-" };
            push_text(
                &mut verts,
                8.0,
                ry + 2.0,
                1.2,
                &format!("{indent}{vis} {}", node.name),
                if is_sel { text_hi } else { text_dim },
            );
            hierarchy_rows.push((0.0, ry - 2.0, left_w, 18.0, node.id));
        }

        // 4. Right Property Inspector Panel.
        let right_w = 260.0f32.min(wf * 0.28);
        let rx = wf - right_w;
        push_rect(&mut verts, rx, 32.0, right_w, body_h, panel_bg);
        push_rect(&mut verts, rx, 32.0, 1.0, body_h, border);
        push_text(&mut verts, rx + 10.0, 42.0, 1.3, "INSPECTOR", accent);

        let mut iy = 64.0;
        if let Some(sel_id) = state.selected
            && let Some(node) = state.scene.find_node(sel_id)
        {
            let p = node.transform.position;
            let s = node.transform.scale;
            push_text(&mut verts, rx + 10.0, iy, 1.25, &format!("NODE: {}", node.name), text_hi);
            iy += 18.0;
            push_text(&mut verts, rx + 10.0, iy, 1.15, &format!("POS: {:.2} {:.2} {:.2}", p[0], p[1], p[2]), text_dim);
            iy += 16.0;
            push_text(&mut verts, rx + 10.0, iy, 1.15, &format!("SCL: {:.2} {:.2} {:.2}", s[0], s[1], s[2]), text_dim);
            iy += 18.0;
            for comp in &node.components {
                let desc = match comp {
                    Component::MeshRenderer { mesh, material, .. } => format!("MESH: {mesh} ({material})"),
                    Component::Light { kind, intensity, .. } => format!("LIGHT: {kind:?} I={intensity:.1}"),
                    Component::Camera { fov_deg, .. } => format!("CAMERA: FOV={fov_deg:.0}"),
                    other => format!("COMP: {}", other.type_name()),
                };
                push_text(&mut verts, rx + 10.0, iy, 1.1, &desc, text_dim);
                iy += 16.0;
            }
        } else {
            push_text(&mut verts, rx + 10.0, iy, 1.15, "CLICK VIEWPORT OR TREE", text_dim);
            iy += 18.0;
        }
        iy += 10.0;
        push_text(&mut verts, rx + 10.0, iy, 1.2, "ENVIRONMENT", accent);
        iy += 16.0;
        push_text(
            &mut verts,
            rx + 10.0,
            iy,
            1.1,
            &format!(
                "SKY:{} DDGI:{} OIT:{}",
                if state.scene.environment.physical_sky { "ON" } else { "OFF" },
                if state.scene.environment.ddgi { "ON" } else { "OFF" },
                if state.scene.environment.oit { "ON" } else { "OFF" },
            ),
            text_dim,
        );
        iy += 16.0;
        if let Some(status) = &state.shader_graph_status {
            push_text(&mut verts, rx + 10.0, iy, 1.1, status, [0.4, 0.9, 0.55, 1.0]);
        }

        // 5. Bottom Timeline Scrubber & Asset Browser Bar.
        let by = hf - bottom_h;
        push_rect(&mut verts, 0.0, by, wf, bottom_h, panel_bg);
        push_rect(&mut verts, 0.0, by, wf, 1.0, border);
        let dur = state.sequence.duration.max(0.01);
        let frac = (state.sequencer.time / dur).clamp(0.0, 1.0);
        let bar_x = 12.0;
        let bar_y = by + 24.0;
        let bar_w = (wf * 0.55).max(80.0);
        let bar_h = 12.0;
        push_text(
            &mut verts,
            12.0,
            by + 8.0,
            1.15,
            &format!("TIMELINE: {:.2}S / {:.2}S | ASSETS: {}", state.sequencer.time, dur, state.asset_files.len()),
            text_hi,
        );
        push_rect(&mut verts, bar_x, bar_y, bar_w, bar_h, [0.16, 0.18, 0.24, 1.0]);
        push_rect(&mut verts, bar_x, bar_y, bar_w * frac, bar_h, accent);
        push_rect(&mut verts, bar_x + bar_w * frac - 2.0, bar_y - 2.0, 4.0, bar_h + 4.0, text_hi);
        let assets_preview: String = state.asset_files.iter().take(4).cloned().collect::<Vec<_>>().join(" | ");
        push_text(&mut verts, 12.0, by + 44.0, 1.05, &assets_preview, text_dim);

        // Upload vertices (growing buffer if needed).
        if !verts.is_empty() {
            if verts.len() > self.vertex_cap {
                self.vertex_cap = verts.len().next_power_of_two();
                self.vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("editor_hud_vb"),
                    size: (self.vertex_cap * std::mem::size_of::<HudVertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            let raw = hud_vertices_bytes(&verts);
            queue.write_buffer(&self.vertex_buf, 0, &raw);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("editor_hud"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buf.slice(0..(verts.len() * std::mem::size_of::<HudVertex>()) as u64));
            pass.draw(0..verts.len() as u32, 0..1);
        }

        HudClickZones { hierarchy_rows, timeline_bar: [bar_x, bar_y, bar_w, bar_h] }
    }
}
