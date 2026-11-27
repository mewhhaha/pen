//! End-to-end headless GPU & CPU integration tests proving standard `pen_protocol::RenderFrame`
//! packets genuinely drive:
//! - AgX vs Shoulder tonemapping & HDRI IBL
//! - Hillaire physical sky LUTs & raymarched volumetric clouds (`quality.physical_sky`)
//! - LTC area lights & authored cookie projectors (`frame.area_lights`)
//! - Stock `lit.wgsl` template applying `SurfaceMapAtlas` (baked lightmap + thin-film iridescence)
//! - Signed-phase thin-film iridescence zero-thickness exact bare Fresnel witness
//! - Full PBR deferred decals (`frame.pbr_decals`)
//! - Split-screen viewports with distinct cameras, `ColorFilter`, insets, and HUD sprites across frames
//! - MSAA 4x depth-resolved outlines isolated at reduced render scale + scene-filter composition
//! - Live `.matgraph.json` visual shader graphs (`Texture2D` lit/unlit/fallback, `sun_shadow`, alpha cutoff, `Time`)
//! - Pure CPU `validate_frame` checks (never skipped without GPU)

mod common;

use pen_protocol::{
    AreaLightSpec, CameraTargetSpec, Cascade, Catalog, Color, Draw, Extension, Inset, PbrDecalSpec, Sprite, Vector,
    VideoStreamSpec, color_filter,
};
use pen_render::{
    ColorFilter, DepthMode, ENV_FOLIAGE_EXT_TAG, ENV_WATER_EXT_TAG, EnvironmentScene, EnvironmentTerrain, HdriImage,
    NodeRef, Renderer, RendererOptions, ShaderGraph, ShaderNode, ShaderNodeKind, TerrainPatch, TerrainUniform,
    TonemapOperator, perspective_reversed_z,
};
use serde_json::json;

const STOCK_LIT_SHADER: &str = include_str!("../../pen/templates/new/src/assets/lit.wgsl");

const SIMPLE_SHADER: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct DrawU { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: DrawU;
@group(1) @binding(1) var base: texture_2d<f32>;
@group(1) @binding(2) var base_samp: sampler;

struct Out { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vertex(@location(0) p: vec3f, @location(1) _n: vec3f, @location(2) uv: vec2f) -> Out {
  var o: Out;
  o.pos = scene.projection * scene.view * draw.model * vec4f(p, 1.0);
  o.uv = uv;
  return o;
}

@fragment fn fragment(i: Out) -> @location(0) vec4f {
  let c = textureSample(base, base_samp, i.uv).rgb * draw.color.rgb * 3.0;
  return vec4f(c, draw.color.a);
}
"#;

const HUD_SPRITE_SHADER: &str = r#"
@group(0) @binding(0) var<uniform> screen: vec4f;
struct SpriteU { rect: vec4f, uv: vec4f, color: vec4f, parameters: vec4f };
@group(0) @binding(1) var<storage, read> sprites: array<SpriteU>;
struct Out { @builtin(position) pos: vec4f, @location(0) color: vec4f };
@vertex fn vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
  let corners = array<vec2f, 6>(vec2f(0,0), vec2f(1,0), vec2f(0,1), vec2f(0,1), vec2f(1,0), vec2f(1,1));
  let s = sprites[i];
  let at = s.rect.xy + corners[v] * s.rect.zw;
  return Out(vec4f(at.x / screen.x * 2.0 - 1.0, 1.0 - at.y / screen.y * 2.0, 0.0, 1.0), s.color);
}
@fragment fn fragment(v: Out) -> @location(0) vec4f {
  return v.color;
}
"#;

const FILTER_SPRITE_SHADER: &str = r#"
@group(0) @binding(0) var<uniform> screen: vec4f;
struct SpriteU { rect: vec4f, uv: vec4f, color: vec4f, parameters: vec4f };
@group(0) @binding(1) var<storage, read> sprites: array<SpriteU>;
@group(0) @binding(4) var scene: texture_2d<f32>;
struct Out { @builtin(position) pos: vec4f, @location(0) color: vec4f };
@vertex fn vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
  let corners = array<vec2f, 6>(vec2f(0,0), vec2f(1,0), vec2f(0,1), vec2f(0,1), vec2f(1,0), vec2f(1,1));
  let s = sprites[i];
  let at = s.rect.xy + corners[v] * s.rect.zw;
  return Out(vec4f(at.x / screen.x * 2.0 - 1.0, 1.0 - at.y / screen.y * 2.0, 0.0, 1.0), s.color);
}
@fragment fn fragment(v: Out) -> @location(0) vec4f {
  let size = vec2i(textureDimensions(scene));
  let p = clamp(vec2i(v.pos.xy / screen.xy * vec2f(size)), vec2i(0), size - vec2i(1));
  return vec4f(textureLoad(scene, p, 0).rgb * v.color.rgb, 1.0);
}
"#;

fn write_solid_png(path: &std::path::Path, rgba: [u8; 4]) {
    let img = image::RgbaImage::from_pixel(4, 4, image::Rgba(rgba));
    img.save(path).unwrap();
}

fn write_split_cookie_png(path: &std::path::Path) {
    // Left half bright red, right half black — proves cookie texture projection on GPU.
    let mut img = image::RgbaImage::new(16, 16);
    for y in 0..16 {
        for x in 0..16 {
            let px = if x < 8 { [255, 20, 20, 255] } else { [0, 0, 0, 255] };
            img.put_pixel(x, y, image::Rgba(px));
        }
    }
    img.save(path).unwrap();
}

fn translate_view(tx: f32, ty: f32, tz: f32) -> [f32; 16] {
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, tx, ty, tz, 1.0]
}

struct UnavailableTarget(u32);

impl pen_render::RenderTarget for UnavailableTarget {
    fn size(&self) -> (u32, u32) {
        (self.0, self.0)
    }

    fn view(&mut self) -> Option<wgpu::TextureView> {
        None
    }

    fn present(&mut self) {
        panic!("target has no view");
    }
}

fn wait_ready(
    renderer: &mut Renderer,
    frame: &pen_protocol::RenderFrame,
    catalog: &Catalog,
    target: &mut common::Offscreen,
) {
    let mut events = Vec::new();
    let mut steady = 0u32;
    for _ in 0..150 {
        renderer.submit(frame, catalog, target).unwrap();
        events.extend(renderer.take_events());
        if !renderer.waiting() && renderer.pending() == 0 {
            steady += 1;
            if steady >= 2 {
                break;
            }
        } else {
            steady = 0;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        !events.iter().any(|e| matches!(e, pen_protocol::GameEvent::AssetFailed { .. })),
        "unexpected asset failure: {events:?}"
    );
    assert!(steady >= 2, "assets failed to load in time: {events:?}");
}

/// Pure CPU validation witness — always executes even on machines without a GPU adapter.
#[test]
fn frame_validation_rejects_bad_video_singular_4x4_pbr_decal_and_oob_split_viewports() {
    let mut bad_video = common::frame(vec![]);
    bad_video.video_streams.push(VideoStreamSpec {
        material: 0,
        time: f32::NAN,
        playing: true,
        looping: true,
        speed: 1.0,
    });
    assert!(pen_render::validate_frame(&bad_video).is_err(), "NaN video time must fail validation");

    // Upper-left 3x3 is identity (det3 = 1), but bottom row is [0,0,0,0] -> singular 4x4!
    let mut bad_decal = common::frame(vec![]);
    bad_decal.pbr_decals.push(PbrDecalSpec {
        material: 0,
        transform: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        albedo: Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
        normal_strength: 1.0,
        roughness: 0.5,
        metallic: 0.0,
        emissive_strength: 0.0,
        fade: 0.1,
    });
    assert!(pen_render::validate_frame(&bad_decal).is_err(), "4x4 singular PBR decal must fail validation");

    // Split-screen viewport starting at 1.0 or with zero width must fail up-front validation.
    let mut bad_split = common::frame(vec![]);
    bad_split.camera_targets.push(CameraTargetSpec {
        material: 0,
        width: 32,
        height: 32,
        viewport_x: 1.0,
        viewport_y: 0.0,
        viewport_w: 0.25,
        viewport_h: 1.0,
        split_screen: true,
        view: bad_split.view,
        projection: bad_split.projection,
    });
    assert!(pen_render::validate_frame(&bad_split).is_err(), "OOB split-screen viewport must fail validation");

    bad_split.camera_targets[0].viewport_x = 0.0;
    bad_split.camera_targets[0].viewport_w = 0.0;
    assert!(pen_render::validate_frame(&bad_split).is_err(), "zero-width split-screen viewport must fail validation");
}

#[test]
fn gpu_thin_film_iridescence_zero_thickness_matches_bare_fresnel() {
    let Some((device, queue)) = common::device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let wgsl = format!(
        r#"
{lobes}
@group(0) @binding(0) var<storage, read_write> out: array<vec4f>;
@compute @workgroup_size(1)
fn main() {{
  // Case 0: n1=1.0, n2=1.3, n3=1.1 (f0 = ((1.1-1)/(1.1+1))^2 = 0.0022675737), thickness=0.
  let f0_a = vec3f(0.0022675737);
  let r_a = lobe_iridescence_fresnel(1.0, 1.3, 1.0, 0.0, f0_a);
  // Case 1: n1=1.0, n2=1.3, n3=1.5 (f0 = 0.04), thickness=0.
  let f0_b = vec3f(0.04);
  let r_b = lobe_iridescence_fresnel(1.0, 1.3, 1.0, 0.0, f0_b);
  // Case 2: n1=1.0, n2=1.5, n3=1.5 (film matches substrate), thickness=350 nm -> exact 0.04.
  let r_c = lobe_iridescence_fresnel(1.0, 1.5, 1.0, 350.0, f0_b);
  // Case 3: non-zero thickness 350 nm with n2=1.8 -> chromatic RGB dispersion (channels differ).
  let r_d = lobe_iridescence_fresnel(1.0, 1.8, 0.85, 350.0, f0_b);
  out[0] = vec4f(r_a, 0.0);
  out[1] = vec4f(r_b, 0.0);
  out[2] = vec4f(r_c, 0.0);
  out[3] = vec4f(r_d, 0.0);
}}
"#,
        lobes = pen_render::SURFACE_LOBES_WGSL,
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("irid_zero_thickness_test"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("irid_zero_thickness_pipe"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("irid_out"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("irid_read"),
        size: 64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("irid_bg"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: out_buf.as_entire_binding() }],
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, 64);
    queue.submit([enc.finish()]);
    let slice = read_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range().unwrap();
    let vals: &[[f32; 4]] = bytemuck::cast_slice(&data);
    assert!(
        (vals[0][0] - 0.002_267_6).abs() < 1e-4,
        "1.0/1.3/1.1 zero-thickness must match bare Fresnel 0.00227, got {:?}",
        vals[0]
    );
    assert!(
        (vals[1][0] - 0.04).abs() < 1e-4,
        "1.0/1.3/1.5 zero-thickness must match bare Fresnel 0.04, got {:?}",
        vals[1]
    );
    assert!((vals[2][0] - 0.04).abs() < 1e-4, "matched film/substrate IOR must stay 0.04, got {:?}", vals[2]);
    let spread = (vals[3][0] - vals[3][2]).abs().max((vals[3][1] - vals[3][2]).abs());
    assert!(spread > 0.02, "350 nm thin film must show chromatic interference spread, got {:?}", vals[3]);
    drop(data);
    read_buf.unmap();
}

#[test]
fn render_frame_end_to_end_drives_agx_hdri_physical_sky_pbr_decals_polish_and_split_screen() {
    let Some((device, queue)) = common::device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let dir = std::env::temp_dir().join(format!("pen-wired-pipeline-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    std::fs::write(dir.join("simple.wgsl"), SIMPLE_SHADER).unwrap();
    std::fs::write(dir.join("lit.wgsl"), STOCK_LIT_SHADER).unwrap();
    std::fs::write(dir.join("hud.wgsl"), HUD_SPRITE_SHADER).unwrap();
    std::fs::write(dir.join("filter.wgsl"), FILTER_SPRITE_SHADER).unwrap();
    write_solid_png(&dir.join("irid.png"), [255, 128, 0, 255]);
    write_solid_png(&dir.join("lightmap.png"), [40, 220, 60, 255]);
    write_solid_png(&dir.join("teal.png"), [20, 220, 230, 255]);
    write_split_cookie_png(&dir.join("cookie.png"));

    // Live .matgraph.json with Texture2D + Time + InstanceParam(0) + alpha cutoff + sun shadows!
    let tex_graph = ShaderGraph {
        name: "live_textured_graph".into(),
        nodes: vec![
            ShaderNode { id: "uv".into(), ui_pos: [0.0, 0.0], kind: ShaderNodeKind::Uv },
            ShaderNode {
                id: "tex".into(),
                ui_pos: [120.0, 0.0],
                kind: ShaderNodeKind::Texture2D { uv: Some(NodeRef::of("uv")), fallback_color: [0.95, 0.15, 0.1] },
            },
            ShaderNode {
                id: "rgb".into(),
                ui_pos: [240.0, 0.0],
                kind: ShaderNodeKind::Swizzle { input: NodeRef::of("tex"), mask: "xyz".into() },
            },
            ShaderNode { id: "t".into(), ui_pos: [120.0, 120.0], kind: ShaderNodeKind::Time },
            ShaderNode { id: "p0".into(), ui_pos: [120.0, 180.0], kind: ShaderNodeKind::InstanceParam { index: 0 } },
            ShaderNode {
                id: "rough".into(),
                ui_pos: [240.0, 150.0],
                kind: ShaderNodeKind::Clamp { x: NodeRef::of("p0"), min: 0.2, max: 0.9 },
            },
            ShaderNode { id: "alpha".into(), ui_pos: [240.0, 220.0], kind: ShaderNodeKind::InstanceParam { index: 1 } },
            ShaderNode {
                id: "out".into(),
                ui_pos: [400.0, 0.0],
                kind: ShaderNodeKind::StandardSurfaceOutput {
                    base_color: Some(NodeRef::of("rgb")),
                    metallic: None,
                    roughness: Some(NodeRef::of("rough")),
                    normal: None,
                    emissive: None,
                    ao: None,
                    alpha: Some(NodeRef::of("alpha")),
                    clearcoat: None,
                    iridescence: None,
                },
            },
        ],
    };
    std::fs::write(dir.join("tex_graph.matgraph.json"), serde_json::to_vec(&tex_graph).unwrap()).unwrap();

    let quad = json!({
        "positions": [-1.0, -1.0, 0.0,  1.0, -1.0, 0.0,  1.0, 1.0, 0.0,  -1.0, 1.0, 0.0],
        "normals": [0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0],
        "uvs": [0.0, 1.0,  1.0, 1.0,  1.0, 0.0,  0.0, 0.0],
        "uvs1": [0.0, 1.0,  1.0, 1.0,  1.0, 0.0,  0.0, 0.0],
        "indices": [0, 1, 2, 0, 2, 3]
    });
    std::fs::write(dir.join("quad.mesh.json"), serde_json::to_vec(&quad).unwrap()).unwrap();

    let write_mat = |name: &str, v: serde_json::Value| {
        std::fs::write(dir.join(name), serde_json::to_vec(&v).unwrap()).unwrap();
    };
    write_mat(
        "wall_red.mat.json",
        json!({ "shader": "simple.wgsl", "color": [1.0, 0.15, 0.1, 1.0], "lighting": "lit", "decals": true }),
    );
    write_mat(
        "wall_green.mat.json",
        json!({ "shader": "simple.wgsl", "color": [0.1, 0.95, 0.15, 1.0], "lighting": "lit" }),
    );
    write_mat("decal.mat.json", json!({ "shader": "simple.wgsl", "color": [0.1, 0.9, 0.2, 1.0], "lighting": "lit" }));
    write_mat("lit_plain.mat.json", json!({ "shader": "lit.wgsl", "color": [1.0, 1.0, 1.0, 1.0], "lighting": "lit" }));
    write_mat(
        "lit_lightmap.mat.json",
        json!({
            "shader": "lit.wgsl",
            "color": [1.0, 1.0, 1.0, 1.0],
            "lighting": "lit",
            "surface": { "lightmapTexture": "lightmap.png" }
        }),
    );
    write_mat(
        "lit_irid.mat.json",
        json!({
            "shader": "lit.wgsl",
            "color": [0.15, 0.15, 0.15, 1.0],
            "lighting": "lit",
            "surface": {
                "iridescence": 1.0,
                "iridescenceIor": 1.9,
                "iridescenceThicknessMin": 200.0,
                "iridescenceThicknessMax": 520.0,
                "iridescenceTexture": "irid.png"
            }
        }),
    );
    write_mat("hud.mat.json", json!({ "shader": "hud.wgsl", "color": [1.0, 1.0, 1.0, 1.0], "lighting": "unlit" }));
    write_mat(
        "filter.mat.json",
        json!({ "shader": "filter.wgsl", "color": [1.0, 1.0, 1.0, 1.0], "lighting": "unlit", "scene_filter": true }),
    );
    write_mat(
        "cookie.mat.json",
        json!({ "shader": "simple.wgsl", "texture": "cookie.png", "color": [1.0, 1.0, 1.0, 1.0], "lighting": "unlit" }),
    );
    write_mat(
        "graph_tex_unlit.mat.json",
        json!({ "shader": "tex_graph.matgraph.json", "texture": "teal.png", "color": [1.0, 1.0, 1.0, 1.0], "lighting": "unlit", "alphaCutoff": 0.5 }),
    );
    write_mat(
        "graph_fallback_unlit.mat.json",
        json!({ "shader": "tex_graph.matgraph.json", "color": [1.0, 1.0, 1.0, 1.0], "lighting": "unlit" }),
    );
    write_mat(
        "graph_lit.mat.json",
        json!({ "shader": "tex_graph.matgraph.json", "texture": "teal.png", "color": [1.0, 1.0, 1.0, 1.0], "lighting": "lit" }),
    );

    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions {
            format: common::FORMAT,
            display: pen_render::Display::SDR,
            assets: dir.clone(),
            veil: None,
            pipeline_cache: None,
        },
    )
    .unwrap();

    let hdri = HdriImage::from_rgba_f32(16, 8, vec![[1.5, 1.2, 1.0, 1.0]; 128]).unwrap();
    let sh = renderer.set_hdri_environment(&hdri, 1.0, 0.0).unwrap();
    assert!(sh[0].x > 1.0);

    let catalog = Catalog {
        meshes: vec!["quad.mesh.json".into()],
        materials: vec![
            "wall_red.mat.json".into(),             // 0
            "wall_green.mat.json".into(),           // 1
            "decal.mat.json".into(),                // 2
            "lit_plain.mat.json".into(),            // 3
            "lit_lightmap.mat.json".into(),         // 4
            "lit_irid.mat.json".into(),             // 5
            "hud.mat.json".into(),                  // 6
            "filter.mat.json".into(),               // 7
            "cookie.mat.json".into(),               // 8
            "graph_tex_unlit.mat.json".into(),      // 9
            "graph_fallback_unlit.mat.json".into(), // 10
            "graph_lit.mat.json".into(),            // 11
        ],
        sounds: vec![],
    };

    let wall_xform = [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -2.0, 1.0];
    let proj = pen_render::perspective_forward_z(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 50.0);

    let mut frame =
        common::frame(vec![Draw { mesh: 0, material: 0, transform: wall_xform, parameters: None, extra: None }]);
    frame.projection = proj;
    frame.light_direction = Vector { x: 0.2, y: -0.9, z: -0.3 };
    frame.grade.tonemap = TonemapOperator::AgX as u32;
    frame.pbr_decals.push(PbrDecalSpec {
        material: 2,
        transform: [0.8, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, -2.0, 1.0],
        albedo: Color { r: 0.1, g: 1.0, b: 0.2, a: 0.9 },
        normal_strength: 0.5,
        roughness: 0.4,
        metallic: 0.0,
        emissive_strength: 0.3,
        fade: 0.1,
    });
    let mut target = common::Offscreen::new(&device, 64);
    wait_ready(&mut renderer, &frame, &catalog, &mut target);

    // 1) AgX vs Shoulder tonemapping produces visibly different HDR colors.
    let mut shoulder_frame = frame.clone();
    shoulder_frame.grade.tonemap = TonemapOperator::Shoulder as u32;
    renderer.submit(&shoulder_frame, &catalog, &mut target).unwrap();
    let px_shoulder = common::middle(&device, &queue, &target.0);
    renderer.submit(&frame, &catalog, &mut target).unwrap();
    let px_agx = common::middle(&device, &queue, &target.0);
    assert_ne!(px_shoulder, px_agx, "AgX and Shoulder tonemappers should differ on HDR colors");

    // 2) Physical sky LUTs & volumetric clouds visibly replace the black clear background.
    let mut sky_off = common::frame(vec![]);
    sky_off.projection = proj;
    sky_off.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    sky_off.quality.sky = false;
    sky_off.quality.physical_sky = false;
    sky_off.light_direction = Vector { x: 0.0, y: -0.7, z: -0.7 };
    sky_off.light_color = Vector { x: 4.0, y: 3.8, z: 3.5 };
    renderer.submit(&sky_off, &catalog, &mut target).unwrap();
    let px_clear = common::pixels(&device, &queue, &target.0)[8 * 64 + 32];

    let mut sky_on = sky_off.clone();
    sky_on.quality.physical_sky = true;
    sky_on.clouds.cover = 0.0;
    renderer.submit(&sky_on, &catalog, &mut target).unwrap();
    let pixels_no_clouds = common::pixels(&device, &queue, &target.0);
    let px_phys_sky = pixels_no_clouds[8 * 64 + 32];
    assert!(
        px_phys_sky[2] > px_clear[2].saturating_add(15),
        "physical_sky LUT must brighten sky background: clear={px_clear:?}, phys={px_phys_sky:?}"
    );

    let mut sky_clouds = sky_on.clone();
    sky_clouds.clouds.cover = 0.9;
    renderer.submit(&sky_clouds, &catalog, &mut target).unwrap();
    let pixels_with_clouds = common::pixels(&device, &queue, &target.0);
    let cloud_diff_pixels = pixels_no_clouds
        .iter()
        .zip(&pixels_with_clouds)
        .filter(|(a, b)| a[0].abs_diff(b[0]) > 2 || a[1].abs_diff(b[1]) > 2 || a[2].abs_diff(b[2]) > 2)
        .count();
    assert!(cloud_diff_pixels > 16, "volumetric clouds must visibly composite over physical sky");

    // 3) LTC area light + authored CookieSpot projector through STOCK `lit.wgsl`!
    let mut area_base = common::frame(vec![Draw {
        mesh: 0,
        material: 3, // stock lit.wgsl plain
        transform: wall_xform,
        parameters: Some([0.0, 0.5, 0.0, 0.0]),
        extra: None,
    }]);
    area_base.projection = proj;
    area_base.light_color = Vector { x: 0.0, y: 0.0, z: 0.0 };
    area_base.ambient = Vector { x: 0.0, y: 0.0, z: 0.0 };
    area_base.sky = Default::default();
    area_base.quality.sky = false;
    area_base.quality.physical_sky = false;
    let black_hdri = HdriImage::from_rgba_f32(8, 4, vec![[0.0, 0.0, 0.0, 1.0]; 32]).unwrap();
    renderer.set_hdri_environment(&black_hdri, 0.0, 0.0).unwrap();
    wait_ready(&mut renderer, &area_base, &catalog, &mut target);
    let px_dark = common::middle(&device, &queue, &target.0);

    let mut area_lit = area_base.clone();
    area_lit.area_lights.push(AreaLightSpec {
        kind: 0,
        position: Vector { x: 0.0, y: 0.0, z: -0.6 },
        right: Vector { x: 1.0, y: 0.0, z: 0.0 },
        up: Vector { x: 0.0, y: 1.0, z: 0.0 },
        color: Vector { x: 1.0, y: 0.4, z: 0.1 },
        intensity: 8.0,
        width: 1.6,
        height: 1.6,
        range: 10.0,
        cookie_material: 0,
        two_sided: true,
    });
    renderer.submit(&area_lit, &catalog, &mut target).unwrap();
    let px_area = common::middle(&device, &queue, &target.0);
    assert!(
        px_area[0] > px_dark[0].saturating_add(40),
        "LTC area light must illuminate stock lit.wgsl: dark={px_dark:?}, lit={px_area:?}"
    );

    // CookieSpot projector using authored red/black split cookie material (material 8 -> 1-based index 9).
    let mut cookie_frame = area_base.clone();
    cookie_frame.area_lights.push(AreaLightSpec {
        kind: 3,
        position: Vector { x: 0.0, y: 0.0, z: 0.0 },
        right: Vector { x: 1.0, y: 0.0, z: 0.0 },
        up: Vector { x: 0.0, y: -1.0, z: 0.0 }, // cross(right, up) = (0, 0, -1) toward wall at z=-2
        color: Vector { x: 1.0, y: 1.0, z: 1.0 },
        intensity: 20.0,
        width: 1.0,
        height: 1.0,
        range: 12.0,
        cookie_material: 9, // 1-based -> catalog.materials[8] = cookie.mat.json
        two_sided: false,
    });
    wait_ready(&mut renderer, &cookie_frame, &catalog, &mut target);
    let cookie_px = common::pixels(&device, &queue, &target.0);
    let cookie_left = cookie_px[32 * 64 + 16];
    let cookie_right = cookie_px[32 * 64 + 48];
    assert!(
        cookie_left[0] > cookie_right[0].saturating_add(50) && cookie_left[0] > cookie_left[1].saturating_add(30),
        "authored cookie projector must illuminate left red and leave right dark: left={cookie_left:?}, right={cookie_right:?}"
    );

    // 4a) Stock `lit.wgsl` + ordinary `.mat.json` baked lightmap (NO custom tile mask parameter!).
    let mut lm_frame = area_base.clone();
    lm_frame.draws[0].material = 4; // lit_lightmap.mat.json
    wait_ready(&mut renderer, &lm_frame, &catalog, &mut target);
    let px_lm = common::middle(&device, &queue, &target.0);
    assert!(
        px_lm[1] > px_dark[1].saturating_add(40) && px_lm[1] > px_lm[0],
        "stock lit.wgsl must decode baked lightmap automatically: dark={px_dark:?}, lm={px_lm:?}"
    );

    // 4b) Isolated iridescence-only witness on stock `lit.wgsl` under white head-on sun!
    // Use moderate roughness and irradiance so neither plain nor iridescent specular clips to white.
    let mut plain_spec = area_base.clone();
    plain_spec.draws[0].material = 3;
    plain_spec.draws[0].parameters = Some([0.0, 0.65, 0.0, 0.0]);
    plain_spec.light_direction = Vector { x: 0.0, y: 0.0, z: -1.0 };
    plain_spec.light_color = Vector { x: 0.8, y: 0.8, z: 0.8 };
    renderer.submit(&plain_spec, &catalog, &mut target).unwrap();
    let px_no_irid = common::middle(&device, &queue, &target.0);

    let mut irid_only = plain_spec.clone();
    irid_only.draws[0].material = 5; // lit_irid.mat.json (dark albedo + iridescence=1.0 + irid.png)
    wait_ready(&mut renderer, &irid_only, &catalog, &mut target);
    let px_irid = common::middle(&device, &queue, &target.0);
    let irid_chroma = px_irid[0].abs_diff(px_irid[2]).max(px_irid[1].abs_diff(px_irid[2]));
    let plain_chroma = px_no_irid[0].abs_diff(px_no_irid[2]).max(px_no_irid[1].abs_diff(px_no_irid[2]));
    assert!(
        irid_chroma > plain_chroma + 10,
        "iridescence-only material on stock lit.wgsl must produce chromatic specular shift: plain={px_no_irid:?}, irid={px_irid:?}"
    );

    // 5) Split-screen with TWO DISTINCT cameras (left sees red quad at x=-10, right sees green quad at x=+10),
    //    global ColorFilter enabled, HUD sprite across seam, and an Inset!
    let mut split_frame = common::frame(vec![
        Draw {
            mesh: 0,
            material: 0, // red at x = -10
            transform: [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -10.0, 0.0, -2.0, 1.0],
            parameters: None,
            extra: None,
        },
        Draw {
            mesh: 0,
            material: 1, // green at x = +10
            transform: [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 0.0, -2.0, 1.0],
            parameters: None,
            extra: None,
        },
    ]);
    split_frame.projection = proj;
    split_frame.camera_targets.push(CameraTargetSpec {
        material: 0,
        width: 32,
        height: 64,
        viewport_x: 0.0,
        viewport_y: 0.0,
        viewport_w: 0.5,
        viewport_h: 1.0,
        split_screen: true,
        view: translate_view(10.0, 0.0, 0.0), // camera at x=-10 -> sees RED quad
        projection: proj,
    });
    split_frame.camera_targets.push(CameraTargetSpec {
        material: 0,
        width: 32,
        height: 64,
        viewport_x: 0.5,
        viewport_y: 0.0,
        viewport_w: 0.5,
        viewport_h: 1.0,
        split_screen: true,
        view: translate_view(-10.0, 0.0, 0.0), // camera at x=+10 -> sees GREEN quad
        projection: perspective_reversed_z(std::f32::consts::FRAC_PI_2, 1.0, 0.1, None),
    });
    split_frame.sprites.push(Sprite {
        x: 24.0,
        y: 24.0,
        width: 16.0,
        height: 16.0,
        u0: 0.0,
        v0: 0.0,
        u1: 1.0,
        v1: 1.0,
        color: [0.05, 0.15, 1.0, 1.0],
        material: 6,
        parameters: [0.0; 4],
    });
    split_frame.insets.push(Inset {
        x: 48.0,
        y: 0.0,
        width: 16.0,
        height: 16.0,
        view: translate_view(0.0, 0.0, 0.0),
        projection: proj,
        ambient: Vector { x: 1.0, y: 1.0, z: 1.0 },
        light_direction: Vector { x: 0.0, y: 0.0, z: -1.0 },
        light_color: Vector { x: 1.0, y: 1.0, z: 1.0 },
        after_sprites: 1,
        pixel_size: 0.0,
        draws: vec![Draw {
            mesh: 0,
            material: 0, // red inset in top-right corner over green right half!
            transform: wall_xform,
            parameters: None,
            extra: None,
        }],
    });
    // Enable an active accessibility ColorFilter (tritanopia at 0.25) to prove split-screen
    // does NOT overwrite the filter target and present only the last camera!
    renderer.set_color_filter(ColorFilter { kind: color_filter::TRITANOPIA, strength: 0.25, simulate: false });
    wait_ready(&mut renderer, &split_frame, &catalog, &mut target);
    assert_eq!(renderer.depth_mode(), DepthMode::ForwardZ);
    // Run a second frame to exercise cached split-screen temporal state across frames.
    renderer.submit(&split_frame, &catalog, &mut target).unwrap();
    let split_stats = renderer.stats();
    assert!(split_stats.draw_calls >= 3, "split stats must include child + inset draws: {split_stats:?}");
    assert!(split_stats.scale > 0.0, "split stats must report actual drawn scale");
    let pixels = common::pixels(&device, &queue, &target.0);
    let left_mid = pixels[32 * 64 + 8];
    let right_mid = pixels[32 * 64 + 56];
    let hud_mid = pixels[32 * 64 + 32];
    let inset_corner = pixels[8 * 64 + 56];
    assert!(
        left_mid[0] > left_mid[1].saturating_add(80),
        "left split camera must show RED quad (not overwritten by right camera): left={left_mid:?}, right={right_mid:?}"
    );
    assert!(
        right_mid[1] > right_mid[0].saturating_add(80),
        "right split camera must show GREEN quad: left={left_mid:?}, right={right_mid:?}"
    );
    assert!(
        hud_mid[2] > 180 && hud_mid[2] > hud_mid[0] + 100,
        "HUD sprite must survive on top of split-screen seam: hud={hud_mid:?}"
    );
    assert!(
        inset_corner[0] > inset_corner[1].saturating_add(80),
        "top-right RED inset must draw over GREEN right split viewport: inset={inset_corner:?}"
    );
    renderer.set_color_filter(ColorFilter::default());

    // 6) Isolated MSAA 4x outline-ONLY test at reduced scene scale (NO RCAS, NO grain, NO flare) + scene filter.
    let mut filter_msaa = common::frame(vec![Draw {
        mesh: 0,
        material: 0,
        transform: [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -2.0, 1.0],
        parameters: None,
        extra: None,
    }]);
    filter_msaa.projection = proj;
    filter_msaa.quality.msaa = 4;
    filter_msaa.quality.scale = 0.5;
    filter_msaa.sprites.push(Sprite {
        x: 0.0,
        y: 0.0,
        width: 64.0,
        height: 64.0,
        u0: 0.0,
        v0: 0.0,
        u1: 1.0,
        v1: 1.0,
        color: [0.5, 1.0, 1.0, 1.0],
        material: 7,
        parameters: [0.0; 4],
    });
    wait_ready(&mut renderer, &filter_msaa, &catalog, &mut target);
    let px_filter_plain = common::pixels(&device, &queue, &target.0);

    let mut filter_outline_only = filter_msaa.clone();
    filter_outline_only.grade.outline_strength = 1.0;
    filter_outline_only.grade.sharpness = 0.0; // Strictly outline-only!
    renderer.submit(&filter_outline_only, &catalog, &mut target).unwrap();
    let px_filter_outlined = common::pixels(&device, &queue, &target.0);
    assert_ne!(
        px_filter_plain, px_filter_outlined,
        "MSAA 4x depth-resolved outlines alone (no RCAS) at reduced scale must visibly modify edges"
    );

    // 7) Live .matgraph.json: Texture2D (unlit textured -> teal, unlit untextured -> red fallback),
    //    alpha cutoff discard, and sun_shadow reception!
    let mut graph_tex = common::frame(vec![Draw {
        mesh: 0,
        material: 9, // graph_tex_unlit with teal.png and alphaCutoff=0.5
        transform: wall_xform,
        parameters: Some([0.5, 1.0, 0.0, 0.0]), // p0=rough, p1=alpha=1.0
        extra: None,
    }]);
    graph_tex.projection = proj;
    graph_tex.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    graph_tex.quality.sky = false;
    wait_ready(&mut renderer, &graph_tex, &catalog, &mut target);
    let px_gt = common::middle(&device, &queue, &target.0);
    assert!(
        px_gt[1] > 140 && px_gt[2] > 140 && px_gt[0] < 80,
        "unlit textured ShaderGraph must sample Texture2D (teal), not fallback: {px_gt:?}"
    );

    // Alpha cutoff discards when alpha < 0.5!
    let mut graph_cutout = graph_tex.clone();
    graph_cutout.draws[0].parameters = Some([0.5, 0.2, 0.0, 0.0]); // alpha = 0.2 < 0.5
    renderer.submit(&graph_cutout, &catalog, &mut target).unwrap();
    let px_cutout = common::middle(&device, &queue, &target.0);
    assert!(
        px_cutout[0] < 10 && px_cutout[1] < 10 && px_cutout[2] < 10,
        "ShaderGraph alphaCutoff=0.5 must discard alpha=0.2 fragments: {px_cutout:?}"
    );

    // Untextured ShaderGraph uses fallback_color (red), NOT white stand-in!
    let mut graph_fb = graph_tex.clone();
    graph_fb.draws[0].material = 10; // graph_fallback_unlit (no texture)
    wait_ready(&mut renderer, &graph_fb, &catalog, &mut target);
    let px_fb = common::middle(&device, &queue, &target.0);
    assert!(
        px_fb[0] > 150 && px_fb[1] < 100 && px_fb[2] < 100,
        "untextured ShaderGraph must use Texture2D fallback_color (red): {px_fb:?}"
    );

    // Lit ShaderGraph receives real sun shadows (occluder quad in front darkens receiver)!
    let mut graph_lit = common::frame(vec![Draw {
        mesh: 0,
        material: 11, // graph_lit
        transform: wall_xform,
        parameters: Some([0.5, 1.0, 0.0, 0.0]),
        extra: None,
    }]);
    graph_lit.projection = proj;
    graph_lit.light_direction = Vector { x: 0.0, y: 0.0, z: -1.0 };
    graph_lit.light_color = Vector { x: 3.0, y: 3.0, z: 3.0 };
    graph_lit.ambient = Vector { x: 0.0, y: 0.0, z: 0.0 };
    graph_lit.sky = Default::default();
    graph_lit.quality.sky = false;
    graph_lit.shadow.texels = 256;
    graph_lit.shadow.view = translate_view(0.0, 0.0, -5.0);
    graph_lit.cascades = vec![Cascade {
        projection: [0.25, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, -0.1, 1.0],
        texel: 8.0 / 256.0,
        far: 50.0,
    }];
    wait_ready(&mut renderer, &graph_lit, &catalog, &mut target);
    let px_lit_unshadowed = common::middle(&device, &queue, &target.0);

    // Place a blocker behind the camera (z = +1.0, between sun at z=+5 and receiver at z=-2) so camera
    // doesn't see the blocker directly, only its sun shadow on the receiver!
    let mut graph_shadowed = graph_lit.clone();
    graph_shadowed.draws.push(Draw {
        mesh: 0,
        material: 11,
        transform: [1.5, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        parameters: Some([0.5, 1.0, 0.0, 0.0]),
        extra: None,
    });
    renderer.submit(&graph_shadowed, &catalog, &mut target).unwrap();
    let px_lit_shadowed = common::middle(&device, &queue, &target.0);
    assert!(
        px_lit_unshadowed[1] > px_lit_shadowed[1].saturating_add(40),
        "lit ShaderGraph must receive sun shadows: unshadowed={px_lit_unshadowed:?}, shadowed={px_lit_shadowed:?}"
    );

    // 8) Catalog out-of-range check on renderer.submit.
    let mut bad_mat = common::frame(vec![]);
    bad_mat.pbr_decals.push(PbrDecalSpec {
        material: 999,
        transform: wall_xform,
        albedo: Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
        normal_strength: 1.0,
        roughness: 0.5,
        metallic: 0.0,
        emissive_strength: 0.0,
        fade: 0.1,
    });
    assert!(
        renderer.submit(&bad_mat, &catalog, &mut target).is_err(),
        "out-of-range pbr_decal material must be rejected"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gpu_render_frame_drives_oit_ray_tracing_ddgi_and_temporal_upscale() {
    let Some((device, queue)) = common::device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let dir = std::env::temp_dir().join(format!("pen-wired-oit-rt-ddgi-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let quad = json!({
        "positions": [-1.0, -1.0, 0.0,  1.0, -1.0, 0.0,  1.0, 1.0, 0.0,  -1.0, 1.0, 0.0],
        "normals": [0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0],
        "uvs": [0.0, 1.0,  1.0, 1.0,  1.0, 0.0,  0.0, 0.0],
        "indices": [0, 1, 2, 0, 2, 3]
    });
    std::fs::write(dir.join("quad.mesh.json"), serde_json::to_vec(&quad).unwrap()).unwrap();
    std::fs::write(dir.join("simple.wgsl"), SIMPLE_SHADER).unwrap();
    std::fs::write(dir.join("lit.wgsl"), STOCK_LIT_SHADER).unwrap();
    write_solid_png(&dir.join("white.png"), [255, 255, 255, 255]);

    // Materials:
    // 0: opaque white lit (receiver, simple.wgsl)
    // 1: translucent red (alpha 0.5)
    // 2: translucent blue (alpha 0.6)
    // 3: unlit emissive red wall (for RT GI/reflection & DDGI color bleeding)
    // 4: opaque dark receiver / blocker (simple.wgsl)
    // 5: stock PBR lit white receiver (lit.wgsl — samples packed in-shader DDGI!)
    // 6: stock PBR lit dark charcoal receiver (lit.wgsl — verifies albedo-modulated DDGI!)
    let mats = [
        (
            "white.mat.json",
            json!({ "shader": "simple.wgsl", "texture": "white.png", "color": [0.85, 0.85, 0.85, 1.0], "lighting": "lit" }),
        ),
        (
            "red_trans.mat.json",
            json!({ "shader": "simple.wgsl", "texture": "white.png", "color": [0.95, 0.08, 0.05, 0.5], "blend": true, "lighting": "unlit" }),
        ),
        (
            "blue_trans.mat.json",
            json!({ "shader": "simple.wgsl", "texture": "white.png", "color": [0.05, 0.15, 0.95, 0.6], "blend": true, "lighting": "unlit" }),
        ),
        (
            "emissive_red.mat.json",
            json!({ "shader": "simple.wgsl", "texture": "white.png", "color": [1.0, 0.0, 0.0, 1.0], "lighting": "unlit", "surface": { "emissiveStrength": 6.0 } }),
        ),
        (
            "blocker.mat.json",
            json!({ "shader": "simple.wgsl", "texture": "white.png", "color": [0.06, 0.06, 0.06, 1.0], "lighting": "lit" }),
        ),
        (
            "lit_white.mat.json",
            json!({ "shader": "lit.wgsl", "texture": "white.png", "color": [0.85, 0.85, 0.85, 1.0], "lighting": "lit", "instanceParameters": [0.0, 0.85, 0.0, 0.0] }),
        ),
        (
            "lit_dark.mat.json",
            json!({ "shader": "lit.wgsl", "texture": "white.png", "color": [0.04, 0.04, 0.04, 1.0], "lighting": "lit", "instanceParameters": [0.0, 0.85, 0.0, 0.0] }),
        ),
    ];
    for (name, val) in &mats {
        std::fs::write(dir.join(name), val.to_string()).unwrap();
    }

    let catalog = Catalog {
        meshes: vec!["quad.mesh.json".into()],
        materials: mats.iter().map(|(n, _)| (*n).into()).collect(),
        sounds: vec![],
    };

    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions {
            format: common::FORMAT,
            display: pen_render::Display::SDR,
            assets: dir.clone(),
            veil: None,
            pipeline_cache: None,
        },
    )
    .unwrap();
    let mut target = common::Offscreen::new(&device, 64);

    let proj = pen_render::perspective_forward_z(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 50.0);
    let scale_at =
        |s: f32, z: f32| -> [f32; 16] { [s, 0.0, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, z, 1.0] };

    // 1) Weighted Blended OIT (`quality.oit = true`) produces identical pixels regardless of draw order!
    let red_near = Draw { mesh: 0, material: 1, transform: scale_at(3.0, -2.0), parameters: None, extra: None };
    let blue_far = Draw { mesh: 0, material: 2, transform: scale_at(3.0, -3.5), parameters: None, extra: None };

    let mut oit_ab = common::frame(vec![red_near.clone(), blue_far.clone()]);
    oit_ab.projection = proj;
    oit_ab.quality.sky = false;
    oit_ab.quality.oit = true;
    oit_ab.clear = Color { r: 0.05, g: 0.05, b: 0.05, a: 1.0 };
    wait_ready(&mut renderer, &oit_ab, &catalog, &mut target);
    let px_oit_ab = common::middle(&device, &queue, &target.0);

    let mut oit_ba = oit_ab.clone();
    oit_ba.draws = vec![blue_far.clone(), red_near.clone()];
    renderer.submit(&oit_ba, &catalog, &mut target).unwrap();
    let px_oit_ba = common::middle(&device, &queue, &target.0);

    assert_eq!(
        px_oit_ab, px_oit_ba,
        "quality.oit=true must render overlapping translucents identically regardless of submission order: {px_oit_ab:?} vs {px_oit_ba:?}"
    );
    assert!(
        px_oit_ab[0] > px_oit_ab[2],
        "WBOIT depth weighting must give nearer red quad stronger contribution than farther blue quad: {px_oit_ab:?}"
    );

    // 2) Ray Tracing (`quality.ray_tracing = 1` shadows/RTAO, `= 2` +reflections/GI) via live RenderFrame!
    let receiver = Draw { mesh: 0, material: 0, transform: scale_at(4.0, -2.0), parameters: None, extra: None };
    // Blocker sits behind the camera (z = +1.0) between sun at +z and receiver at z = -2.0.
    let blocker = Draw { mesh: 0, material: 4, transform: scale_at(4.0, 1.0), parameters: None, extra: None };
    let mut rt_off = common::frame(vec![receiver.clone(), blocker.clone()]);
    rt_off.projection = proj;
    rt_off.light_direction = Vector { x: 0.0, y: 0.0, z: -1.0 };
    rt_off.light_color = Vector { x: 2.5, y: 2.5, z: 2.5 };
    rt_off.sky = Default::default();
    rt_off.quality.sky = false;
    rt_off.quality.ray_tracing = 0;
    wait_ready(&mut renderer, &rt_off, &catalog, &mut target);
    let px_rt0 = common::middle(&device, &queue, &target.0);

    let mut rt_tier1 = rt_off.clone();
    rt_tier1.quality.ray_tracing = 1;
    renderer.submit(&rt_tier1, &catalog, &mut UnavailableTarget(64)).unwrap();
    renderer.submit(&rt_tier1, &catalog, &mut target).unwrap();
    let px_rt1 = common::middle(&device, &queue, &target.0);
    assert!(
        px_rt0[0] > px_rt1[0].saturating_add(40),
        "quality.ray_tracing=1 must cast BVH RT shadows onto receiver: off={px_rt0:?}, rt1={px_rt1:?}"
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let limits = wgpu::Limits { max_storage_buffers_per_shader_stage: 4, ..wgpu::Limits::default() };
    let (limited_device, limited_queue) = pollster::block_on(
        adapter.request_device(&wgpu::DeviceDescriptor { required_limits: limits, ..Default::default() }),
    )
    .unwrap();
    let mut limited = Renderer::new(
        limited_device.clone(),
        limited_queue.clone(),
        RendererOptions {
            format: common::FORMAT,
            display: pen_render::Display::SDR,
            assets: dir.clone(),
            veil: None,
            pipeline_cache: None,
        },
    )
    .unwrap();
    let mut limited_target = common::Offscreen::new(&limited_device, 64);
    let mut fallback = rt_off.clone();
    fallback.quality.ray_tracing = 3;
    fallback.quality.ddgi = true;
    wait_ready(&mut limited, &fallback, &catalog, &mut limited_target);
    assert_eq!(common::middle(&limited_device, &limited_queue, &limited_target.0), px_rt0);
    fallback.draws[0].material = 5;
    fallback.light_color = Vector::default();
    fallback.lights.push(pen_protocol::Light {
        position: Vector { z: -1.0, ..Default::default() },
        radius: 8.0,
        color: Vector { x: 10.0, y: 1.0, z: 0.0 },
        inner: -1.0,
        outer: -1.0,
        ..Default::default()
    });
    wait_ready(&mut limited, &fallback, &catalog, &mut limited_target);
    let compact_lit = common::middle(&limited_device, &limited_queue, &limited_target.0);
    fallback.quality.ray_tracing = 0;
    fallback.quality.ddgi = false;
    wait_ready(&mut renderer, &fallback, &catalog, &mut target);
    assert_eq!(compact_lit, common::middle(&device, &queue, &target.0));
    assert!(compact_lit[0] > compact_lit[2].saturating_add(40));
    drop(limited);

    // Swap blocker for bright emissive red panel facing a dark receiver; tier 2 adds RT reflections + 1-bounce GI!
    let dark_receiver = Draw { mesh: 0, material: 4, transform: scale_at(4.0, -2.0), parameters: None, extra: None };
    let emissive_panel = Draw { mesh: 0, material: 3, transform: scale_at(4.0, 1.0), parameters: None, extra: None };
    let mut rt_dark_tier1 = rt_tier1.clone();
    rt_dark_tier1.draws = vec![dark_receiver.clone(), emissive_panel.clone()];
    wait_ready(&mut renderer, &rt_dark_tier1, &catalog, &mut target);
    let px_dark_rt1 = common::middle(&device, &queue, &target.0);

    let mut rt_tier2 = rt_dark_tier1.clone();
    rt_tier2.quality.ray_tracing = 2;
    renderer.submit(&rt_tier2, &catalog, &mut target).unwrap();
    let px_rt2 = common::middle(&device, &queue, &target.0);
    assert!(
        px_rt2[0] > px_rt2[2].saturating_add(40) && px_rt2[0] > px_dark_rt1[0].saturating_add(40),
        "quality.ray_tracing=2 must inject colored RT reflection + 1-bounce GI from emissive wall: rt1={px_dark_rt1:?}, rt2={px_rt2:?}"
    );

    // 3) Dynamic Diffuse GI (`quality.ddgi = true`) evaluates inside forward PBR (`lit.wgsl` via packed
    //    `probe_data` at group 0 binding 8) with proper albedo modulation — bright diffuse receiver bleeds
    //    vivid red from the front-facing emissive wall while a dark charcoal receiver stays dark!
    let facing_emissive = Draw {
        mesh: 0,
        material: 3,
        transform: [-4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        parameters: None,
        extra: None,
    };
    let mut ddgi_off = common::frame(vec![
        Draw { mesh: 0, material: 5, transform: scale_at(4.0, -2.0), parameters: None, extra: None },
        facing_emissive.clone(),
    ]);
    ddgi_off.projection = proj;
    ddgi_off.light_direction = Vector { x: 0.0, y: -1.0, z: 0.0 };
    ddgi_off.light_color = Vector { x: 0.0, y: 0.0, z: 0.0 };
    ddgi_off.ambient = Vector { x: 0.0, y: 0.0, z: 0.0 };
    ddgi_off.sky = Default::default();
    ddgi_off.quality.sky = false;
    ddgi_off.probe_volume = pen_protocol::ProbeVolume {
        revision: 1,
        nx: 3,
        ny: 3,
        nz: 3,
        spacing: 1.5,
        origin: Vector { x: -1.5, y: -1.5, z: -2.0 },
    };
    ddgi_off.quality.ddgi = false;
    wait_ready(&mut renderer, &ddgi_off, &catalog, &mut target);
    let px_no_ddgi = common::middle(&device, &queue, &target.0);

    let mut ddgi_on = ddgi_off.clone();
    ddgi_on.quality.ddgi = true;
    for _ in 0..4 {
        renderer.submit(&ddgi_on, &catalog, &mut target).unwrap();
    }
    let px_ddgi = common::middle(&device, &queue, &target.0);
    assert!(
        px_ddgi[0] > px_no_ddgi[0].saturating_add(40) && px_ddgi[0] > px_ddgi[2].saturating_add(30),
        "quality.ddgi=true must illuminate lit PBR receiver with red indirect color bleed: off={px_no_ddgi:?}, on={px_ddgi:?}"
    );

    // Same DDGI field on dark charcoal albedo [0.04]: proves PBR albedo modulation (no additive washout!).
    let mut ddgi_dark = ddgi_on.clone();
    ddgi_dark.draws[0].material = 6;
    wait_ready(&mut renderer, &ddgi_dark, &catalog, &mut target);
    let px_dark_ddgi = common::middle(&device, &queue, &target.0);
    assert!(
        px_dark_ddgi[0] * 2 < px_ddgi[0],
        "in-shader PBR DDGI must modulate indirect light by surface albedo (no additive washout on dark material): white={px_ddgi:?}, dark={px_dark_ddgi:?}"
    );

    // Combined RT2 + DDGI: exercises half-res/full-res RT reflections sampling live packed DDGI with zero redundant RT GI.
    let mut combined = ddgi_on.clone();
    combined.quality.ray_tracing = 2;
    renderer.submit(&combined, &catalog, &mut target).unwrap();
    let px_combined = common::middle(&device, &queue, &target.0);
    assert!(
        px_combined[0] > px_combined[2].saturating_add(30),
        "combined RT2 + DDGI must render cleanly with vivid indirect illumination: {px_combined:?}"
    );

    // 3b) Tier-3 Ultra RT (`quality.ray_tracing = 3`, `ddgi = false`): Thin G-Buffer prepass +
    //     ReSTIR GI (spatiotemporal Jacobian Pairwise MIS + thin-wall visibility) + SVGF demodulated
    //     denoising + full-res Thin G-Buffer `albedo * (1 - metallic)` re-modulation!
    let mut restir_white = ddgi_off.clone();
    restir_white.quality.ddgi = false;
    restir_white.quality.ray_tracing = 3;
    restir_white.exposure.cut = true;
    renderer.submit(&restir_white, &catalog, &mut target).unwrap();
    restir_white.exposure.cut = false;
    for _ in 0..4 {
        renderer.submit(&restir_white, &catalog, &mut target).unwrap();
    }
    let px_restir_white = common::middle(&device, &queue, &target.0);
    assert!(
        px_restir_white[0] > px_no_ddgi[0].saturating_add(40)
            && px_restir_white[0] > px_restir_white[2].saturating_add(30),
        "quality.ray_tracing=3 must drive Thin G-Buffer + ReSTIR GI red color bleed onto white receiver: off={px_no_ddgi:?}, restir_white={px_restir_white:?}"
    );

    // Same Tier-3 ReSTIR GI scene on dark charcoal albedo [0.04]: proves demodulated SVGF + Thin G-Buffer albedo modulation!
    let mut restir_dark = restir_white.clone();
    restir_dark.draws[0].material = 6;
    restir_dark.exposure.cut = true;
    wait_ready(&mut renderer, &restir_dark, &catalog, &mut target);
    restir_dark.exposure.cut = false;
    for _ in 0..3 {
        renderer.submit(&restir_dark, &catalog, &mut target).unwrap();
    }
    let px_restir_dark = common::middle(&device, &queue, &target.0);
    assert!(
        px_restir_dark[0] * 2 < px_restir_white[0],
        "Tier-3 ReSTIR GI must modulate demodulated GI by Thin G-Buffer albedo (no washout on dark material): white={px_restir_white:?}, dark={px_restir_dark:?}"
    );

    // Reversed-Z compatibility for Tier-3 Thin G-Buffer + ReSTIR GI.
    let mut restir_rev = restir_white.clone();
    restir_rev.projection = perspective_reversed_z(std::f32::consts::FRAC_PI_2, 1.0, 0.1, None);
    restir_rev.exposure.cut = true;
    wait_ready(&mut renderer, &restir_rev, &catalog, &mut target);
    restir_rev.exposure.cut = false;
    for _ in 0..3 {
        renderer.submit(&restir_rev, &catalog, &mut target).unwrap();
    }
    let px_restir_rev = common::middle(&device, &queue, &target.0);
    assert!(
        px_restir_rev[0] > px_restir_rev[2].saturating_add(30),
        "Tier-3 Thin G-Buffer + ReSTIR GI must work identically under Reversed-Z: {px_restir_rev:?}"
    );

    // 4) Temporal Upscaling (`quality.upscale = true`): FSR Quality + translucent auto-reactive pass,
    //    seamless Dynamic Resolution Scaling (0.67 -> 0.50 preserving full-res history), and non-TAA
    //    spatial FSR 1 EASU fallback (`scale = 0.5`, `taa = false`, `upscale = false`)!
    let red_front = Draw { mesh: 0, material: 1, transform: scale_at(3.0, -1.5), parameters: None, extra: None };
    let dark_back = Draw { mesh: 0, material: 4, transform: scale_at(4.0, -2.0), parameters: None, extra: None };
    let mut upscale_frame = common::frame(vec![dark_back, red_front]);
    upscale_frame.projection = proj;
    upscale_frame.quality.sky = false;
    upscale_frame.quality.scale = 0.6667;
    upscale_frame.quality.upscale = true;
    wait_ready(&mut renderer, &upscale_frame, &catalog, &mut target);
    for _ in 0..5 {
        renderer.submit(&upscale_frame, &catalog, &mut target).unwrap();
    }
    let passes = renderer.passes();
    assert!(passes.contains(&"taa"), "quality.upscale=true must enable temporal accumulation pass: {passes:?}");
    let px_fsr_q = common::middle(&device, &queue, &target.0);
    assert!(
        px_fsr_q[0] > 100 && px_fsr_q[0] > px_fsr_q[2].saturating_add(30),
        "FSR Quality upscale with translucent auto-reactive overlay must render cleanly: {px_fsr_q:?}"
    );

    // Seamless DRS transition (0.6667 -> 0.50) preserves full-res history without artifacts.
    upscale_frame.quality.scale = 0.5;
    upscale_frame.draws = vec![receiver.clone()];
    for _ in 0..4 {
        renderer.submit(&upscale_frame, &catalog, &mut target).unwrap();
    }
    let px_up = common::middle(&device, &queue, &target.0);
    assert!(
        px_up[0] > 100 && px_up[1] > 100 && px_up[2] > 100,
        "DRS FSR Performance receiver must render cleanly: {px_up:?}"
    );

    // Non-TAA reduced-scale spatial FSR 1 EASU path in present.wgsl.
    let mut easu_frame = common::frame(vec![receiver]);
    easu_frame.projection = proj;
    easu_frame.quality.sky = false;
    easu_frame.quality.scale = 0.5;
    easu_frame.quality.taa = false;
    easu_frame.quality.upscale = false;
    renderer.submit(&easu_frame, &catalog, &mut target).unwrap();
    let px_easu = common::middle(&device, &queue, &target.0);
    assert!(
        px_easu[0] > 100 && px_easu[1] > 100 && px_easu[2] > 100,
        "non-TAA FSR 1 EASU present path must render cleanly: {px_easu:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gpu_render_frame_drives_svgf_meshlets_environment_hdri_and_reversed_z() {
    let Some((device, queue)) = common::device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let dir = std::env::temp_dir().join(format!("pen-wired-gpu-v2-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Build a dense 16x16 UV sphere (512 triangles >= 128 threshold) so meshlet cluster culling activates.
    let (mut pos, mut norm, mut uv, mut idx) =
        (Vec::<f32>::new(), Vec::<f32>::new(), Vec::<f32>::new(), Vec::<u32>::new());
    let stacks = 16u32;
    let slices = 16u32;
    for r in 0..=stacks {
        let v = r as f32 / stacks as f32;
        let phi = v * std::f32::consts::PI;
        for c in 0..=slices {
            let u = c as f32 / slices as f32;
            let theta = u * std::f32::consts::TAU;
            let x = phi.sin() * theta.cos();
            let y = phi.cos();
            let z = phi.sin() * theta.sin();
            pos.extend_from_slice(&[x, y, z]);
            norm.extend_from_slice(&[x, y, z]);
            uv.extend_from_slice(&[u, v]);
        }
    }
    let stride = slices + 1;
    for r in 0..stacks {
        for c in 0..slices {
            let i0 = r * stride + c;
            let i1 = i0 + 1;
            let i2 = (r + 1) * stride + c;
            let i3 = i2 + 1;
            idx.extend_from_slice(&[i0, i1, i2, i1, i3, i2]);
        }
    }
    let sphere_json = json!({ "positions": pos, "normals": norm, "uvs": uv, "indices": idx });
    let quad_json = json!({
        "positions": [-1.0, -1.0, 0.0,  1.0, -1.0, 0.0,  1.0, 1.0, 0.0,  -1.0, 1.0, 0.0],
        "normals": [0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0],
        "uvs": [0.0, 1.0,  1.0, 1.0,  1.0, 0.0,  0.0, 0.0],
        "indices": [0, 1, 2, 0, 2, 3]
    });
    std::fs::write(dir.join("sphere.mesh.json"), serde_json::to_vec(&sphere_json).unwrap()).unwrap();
    std::fs::write(dir.join("quad.mesh.json"), serde_json::to_vec(&quad_json).unwrap()).unwrap();
    std::fs::write(dir.join("simple.wgsl"), SIMPLE_SHADER).unwrap();
    std::fs::write(dir.join("lit.wgsl"), STOCK_LIT_SHADER).unwrap();
    write_solid_png(&dir.join("white.png"), [255, 255, 255, 255]);

    let mats = [
        (
            "green.mat.json",
            json!({ "shader": "simple.wgsl", "texture": "white.png", "color": [0.1, 0.9, 0.15, 1.0], "lighting": "unlit" }),
        ),
        (
            "red.mat.json",
            json!({ "shader": "simple.wgsl", "texture": "white.png", "color": [0.95, 0.1, 0.1, 1.0], "lighting": "unlit" }),
        ),
        (
            "chrome.mat.json",
            json!({ "shader": "lit.wgsl", "texture": "white.png", "color": [0.95, 0.95, 0.95, 1.0], "lighting": "lit", "instanceParameters": [0.1, 0.95, 1.0, 0.0] }),
        ),
        (
            "dark_lit.mat.json",
            json!({ "shader": "simple.wgsl", "texture": "white.png", "color": [0.05, 0.05, 0.05, 1.0], "lighting": "lit" }),
        ),
        (
            "emissive_red.mat.json",
            json!({ "shader": "simple.wgsl", "texture": "white.png", "color": [1.0, 0.0, 0.0, 1.0], "lighting": "unlit", "surface": { "emissiveStrength": 6.0 } }),
        ),
    ];
    for (name, val) in &mats {
        std::fs::write(dir.join(name), val.to_string()).unwrap();
    }

    let catalog = Catalog {
        meshes: vec!["sphere.mesh.json".into(), "quad.mesh.json".into()],
        materials: mats.iter().map(|(n, _)| (*n).into()).collect(),
        sounds: vec![],
    };
    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions {
            format: common::FORMAT,
            display: pen_render::Display::SDR,
            assets: dir.clone(),
            veil: None,
            pipeline_cache: None,
        },
    )
    .unwrap();
    let mut target = common::Offscreen::new(&device, 64);
    let proj_fwd = pen_render::perspective_forward_z(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);

    // 1) GPU meshlet cluster culling activates on the 512-triangle sphere and renders visibly.
    let mut ml_frame = common::frame(vec![Draw {
        mesh: 0,
        material: 0,
        transform: [1.5, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, -3.0, 1.0],
        parameters: None,
        extra: None,
    }]);
    ml_frame.projection = proj_fwd;
    ml_frame.quality.sky = false;
    ml_frame.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    wait_ready(&mut renderer, &ml_frame, &catalog, &mut target);
    assert!(renderer.last_meshlet_cull_draws() >= 1, "dense opaque LOD0 sphere must execute GPU meshlet culling");
    let px_ml = common::middle(&device, &queue, &target.0);
    assert!(
        px_ml[1] > 150 && px_ml[0] < 80,
        "GPU meshlet indirect draw must rasterize the green sphere visibly: {px_ml:?}"
    );

    // 2) SVGF spatiotemporal denoiser runs cleanly over 4 consecutive RT tier-2 frames.
    let scale_at =
        |s: f32, z: f32| -> [f32; 16] { [s, 0.0, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, z, 1.0] };
    let mut svgf_frame = common::frame(vec![
        Draw { mesh: 1, material: 3, transform: scale_at(4.0, -2.0), parameters: None, extra: None },
        Draw { mesh: 1, material: 4, transform: scale_at(4.0, 1.0), parameters: None, extra: None },
    ]);
    svgf_frame.projection = proj_fwd;
    svgf_frame.light_direction = Vector { x: 0.0, y: 0.0, z: -1.0 };
    svgf_frame.light_color = Vector { x: 2.0, y: 2.0, z: 2.0 };
    svgf_frame.sky = Default::default();
    svgf_frame.quality.sky = false;
    svgf_frame.quality.ray_tracing = 2;
    wait_ready(&mut renderer, &svgf_frame, &catalog, &mut target);
    for _ in 0..3 {
        renderer.submit(&svgf_frame, &catalog, &mut target).unwrap();
    }
    let px_svgf = common::middle(&device, &queue, &target.0);
    assert!(
        px_svgf[0] > px_svgf[2].saturating_add(40),
        "4-frame SVGF-denoised RT tier-2 pass must preserve warm red GI bounce: {px_svgf:?}"
    );

    // 3) HDRI Radiance `.hdr` bytes IBL (`Renderer::set_hdri_bytes` + `has_hdri_environment`).
    let mut hdr_frame = common::frame(vec![Draw {
        mesh: 0,
        material: 2,
        transform: [1.6, 0.0, 0.0, 0.0, 0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 1.6, 0.0, 0.0, 0.0, -3.0, 1.0],
        parameters: None,
        extra: None,
    }]);
    hdr_frame.projection = proj_fwd;
    hdr_frame.light_color = Vector { x: 0.0, y: 0.0, z: 0.0 };
    hdr_frame.ambient = Vector { x: 0.0, y: 0.0, z: 0.0 };
    hdr_frame.sky = Default::default();
    hdr_frame.quality.sky = false;
    assert!(!renderer.has_hdri_environment());
    wait_ready(&mut renderer, &hdr_frame, &catalog, &mut target);
    let px_no_hdri = common::middle(&device, &queue, &target.0);

    // Build a valid flat 4x4 Radiance .hdr file filled with bright warm amber (RGBE ~ [8.0, 4.0, 0.5]).
    let mut hdr_bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 4\n".to_vec();
    for _ in 0..16 {
        // E=132 -> scale = 2^(132-128)/256 = 16/256 = 0.0625; R=192 -> 12.0, G=96 -> 6.0, B=8 -> 0.5
        hdr_bytes.extend_from_slice(&[192, 96, 8, 132]);
    }
    renderer.set_hdri_bytes(&hdr_bytes, 1.0, 0.0).unwrap();
    assert!(renderer.has_hdri_environment());
    renderer.submit(&hdr_frame, &catalog, &mut target).unwrap();
    let px_with_hdri = common::middle(&device, &queue, &target.0);
    assert!(
        px_with_hdri[0] > px_no_hdri[0].saturating_add(50) && px_with_hdri[0] > px_with_hdri[2].saturating_add(40),
        "Radiance .hdr IBL must light metallic chrome sphere warmly: before={px_no_hdri:?}, after={px_with_hdri:?}"
    );
    renderer.clear_hdri_environment();
    assert!(!renderer.has_hdri_environment());

    // 4) Built-in outdoor environment passes: CDLOD terrain (`EnvironmentScene`), Gerstner water (`ENV_WATER_EXT_TAG`),
    //    and GPU wind-swaying foliage (`ENV_FOLIAGE_EXT_TAG`) rendered through standard `Renderer::submit`.
    let look_down: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -6.0, 1.0]; // camera at (0, 6, 0) looking straight down -Y at XZ plane y=0!
    let mut env_base = common::frame(vec![]);
    env_base.view = look_down;
    env_base.projection = proj_fwd;
    env_base.light_direction = Vector { x: 0.0, y: -1.0, z: 0.0 };
    env_base.light_color = Vector { x: 2.5, y: 2.4, z: 2.2 };
    env_base.sky.horizon = Vector { x: 0.65, y: 0.78, z: 0.92 };
    env_base.sky.zenith = Vector { x: 0.15, y: 0.38, z: 0.78 };
    env_base.quality.sky = false;
    env_base.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };

    // 4a. Host-configured CDLOD terrain heightmap.
    let hm = renderer.upload_terrain_heightmap(4, 4, &[0.5f32; 16]).unwrap();
    renderer.set_environment_scene(EnvironmentScene {
        terrain: Some(EnvironmentTerrain {
            heightmap: std::sync::Arc::new(hm),
            uniform: TerrainUniform {
                view_projection: [0.0; 16],
                camera_pos_and_fog: [0.0, 6.0, 0.0, 0.0],
                sun_dir_and_height_scale: [0.0, 1.0, 0.0, 1.0],
                sun_color_and_ambient: [2.2, 2.1, 2.0, 0.4],
                world_bounds: [20.0, 20.0, 40.0, 80.0],
                layer_heights: [0.1, 0.4, 0.8, 0.2],
                fog_color: [0.6, 0.7, 0.8, 0.0],
            },
            patches: vec![TerrainPatch {
                origin_scale_morph: [-10.0, -10.0, 20.0, 0.0],
                uv_rect_and_grid: [0.0, 0.0, 1.0, 8.0],
            }],
        }),
        water: vec![],
        foliage: None,
    });
    renderer.submit(&env_base, &catalog, &mut target).unwrap();
    let px_terrain = common::middle(&device, &queue, &target.0);
    assert!(
        px_terrain[0] > 20 || px_terrain[1] > 20 || px_terrain[2] > 20,
        "CDLOD terrain pass must rasterize terrain pixels: {px_terrain:?}"
    );
    renderer.clear_environment_scene();

    // 4b. Squid extension Gerstner water (`ENV_WATER_EXT_TAG`).
    let mut water_frame = env_base.clone();
    water_frame.extensions.push(Extension {
        tag: ENV_WATER_EXT_TAG,
        values: vec![
            0.0,  // water level y
            -2.0, // bed y
            0.0,  // bed slope
            16.0, // grid cells
            0.5,  // time
            0.45, 0.12, 0.08, // Beer-Lambert absorption
            0.04, 0.25, 0.42, // deep tint
            0.65, // foam threshold
            0.55, 0.48, 0.35, // bed color
            0.6,  // wave amp
            10.0, // half extent
        ],
    });
    renderer.submit(&water_frame, &catalog, &mut target).unwrap();
    let px_water = common::middle(&device, &queue, &target.0);
    assert!(
        px_water[2] > 30 && px_water[2] >= px_water[0],
        "ENV_WATER_EXT_TAG must draw blue-tinted Gerstner water: {px_water:?}"
    );

    // 4c. Squid extension GPU foliage (`ENV_FOLIAGE_EXT_TAG`).
    let mut foliage_frame = common::frame(vec![]);
    foliage_frame.projection = proj_fwd;
    foliage_frame.light_direction = Vector { x: 0.0, y: -1.0, z: -0.3 };
    foliage_frame.light_color = Vector { x: 2.4, y: 2.4, z: 2.2 };
    foliage_frame.quality.sky = false;
    foliage_frame.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    let mut fvals = vec![
        0.5, // time
        1.0, 0.0, // wind dir xz
        0.2, 1.5, 1.0, // wind strength, speed, gust
        0.45, 2.4, // blade half-width, height
        0.08, 0.42, 0.06, // root rgb
        0.25, 0.85, 0.12, // tip rgb
        0.4, 0.9, 0.15, 2.0, // sss rgb + power
    ];
    // Cluster blades right in front of the camera at z = -2.0 spanning y = -1.2..+1.2!
    for i in 0..8 {
        let x = -0.7 + i as f32 * 0.2;
        fvals.extend_from_slice(&[x, -1.2, -2.0, 1.0, 0.0, i as f32 * 0.4, 0.0, 1.0]);
    }
    foliage_frame.extensions.push(Extension { tag: ENV_FOLIAGE_EXT_TAG, values: fvals });
    renderer.submit(&foliage_frame, &catalog, &mut target).unwrap();
    let px_foliage = common::middle(&device, &queue, &target.0);
    assert!(
        px_foliage[1] > px_foliage[0].saturating_add(25) && px_foliage[1] > 40,
        "ENV_FOLIAGE_EXT_TAG must draw green wind-swaying grass blades: {px_foliage:?}"
    );

    // 5) Reversed-Z auto-detection & distant depth precision:
    //    Two quads 1m apart at 5 km (`z = -5000.0` GREEN in front of `z = -5001.0` RED)
    //    must always show GREEN even when RED is submitted AFTER GREEN!
    let proj_rev = perspective_reversed_z(std::f32::consts::FRAC_PI_2, 1.0, 0.1, None);
    let mut rev_frame = common::frame(vec![
        Draw { mesh: 1, material: 0, transform: scale_at(8000.0, -5000.0), parameters: None, extra: None }, // nearer GREEN
        Draw { mesh: 1, material: 1, transform: scale_at(8000.0, -5001.0), parameters: None, extra: None }, // farther RED drawn second
    ]);
    rev_frame.projection = proj_rev;
    rev_frame.quality.sky = false;
    rev_frame.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    wait_ready(&mut renderer, &rev_frame, &catalog, &mut target);
    assert_eq!(renderer.depth_mode(), DepthMode::ReversedZ);
    let px_rev = common::middle(&device, &queue, &target.0);
    assert!(
        px_rev[1] > px_rev[0].saturating_add(120),
        "Reversed-Z must resolve 1m depth separation at 5km cleanly (green in front of red): {px_rev:?}"
    );
    // Verify switching back to Forward-Z restores DepthMode::ForwardZ cleanly too.
    renderer.submit(&ml_frame, &catalog, &mut target).unwrap();
    assert_eq!(renderer.depth_mode(), DepthMode::ForwardZ);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn tier3_restir_di_many_light_shadows_sharc_and_two_phase_meshlet_disocclusion_witness() {
    let Some((device, queue)) = common::featured_device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let dir = std::env::temp_dir().join(format!("pen-tier3-restir-di-sharc-2p-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Dense 512-triangle UV sphere (>124 tris -> multiple meshlets!) + unit quad.
    let (stacks, slices) = (16u32, 16u32);
    let (mut pos, mut norm, mut uv, mut idx) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for r in 0..=stacks {
        let v = r as f32 / stacks as f32;
        let phi = v * std::f32::consts::PI;
        for c in 0..=slices {
            let u = c as f32 / slices as f32;
            let theta = u * std::f32::consts::TAU;
            let (x, y, z) = (phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
            pos.extend_from_slice(&[x, y, z]);
            norm.extend_from_slice(&[x, y, z]);
            uv.extend_from_slice(&[u, v]);
        }
    }
    let stride = slices + 1;
    for r in 0..stacks {
        for c in 0..slices {
            let (i0, i1, i2, i3) = (r * stride + c, r * stride + c + 1, (r + 1) * stride + c, (r + 1) * stride + c + 1);
            idx.extend_from_slice(&[i0, i1, i2, i1, i3, i2]);
        }
    }
    let sphere_json = json!({ "positions": pos, "normals": norm, "uvs": uv, "indices": idx });
    let quad_json = json!({
        "positions": [-1.0, -1.0, 0.0,  1.0, -1.0, 0.0,  1.0, 1.0, 0.0,  -1.0, 1.0, 0.0],
        "normals": [0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0],
        "uvs": [0.0, 1.0,  1.0, 1.0,  1.0, 0.0,  0.0, 0.0],
        "indices": [0, 1, 2, 0, 2, 3]
    });
    std::fs::write(dir.join("sphere.mesh.json"), serde_json::to_vec(&sphere_json).unwrap()).unwrap();
    std::fs::write(dir.join("quad.mesh.json"), serde_json::to_vec(&quad_json).unwrap()).unwrap();
    std::fs::write(dir.join("simple.wgsl"), SIMPLE_SHADER).unwrap();
    std::fs::write(dir.join("lit.wgsl"), STOCK_LIT_SHADER).unwrap();
    write_solid_png(&dir.join("white.png"), [255, 255, 255, 255]);

    let mats = [
        (
            "green.mat.json",
            json!({ "shader": "simple.wgsl", "texture": "white.png", "color": [0.1, 0.9, 0.15, 1.0], "lighting": "unlit" }),
        ),
        (
            "red.mat.json",
            json!({ "shader": "simple.wgsl", "texture": "white.png", "color": [0.95, 0.1, 0.1, 1.0], "lighting": "unlit" }),
        ),
        (
            "white_lit.mat.json",
            json!({ "shader": "lit.wgsl", "texture": "white.png", "color": [0.85, 0.85, 0.85, 1.0], "lighting": "lit", "instanceParameters": [0.0, 0.75, 1.0, 0.0] }),
        ),
    ];
    for (name, val) in &mats {
        std::fs::write(dir.join(name), val.to_string()).unwrap();
    }
    let catalog = Catalog {
        meshes: vec!["sphere.mesh.json".into(), "quad.mesh.json".into()],
        materials: mats.iter().map(|(n, _)| (*n).into()).collect(),
        sounds: vec![],
    };
    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions {
            format: common::FORMAT,
            display: pen_render::Display::SDR,
            assets: dir.clone(),
            veil: None,
            pipeline_cache: None,
        },
    )
    .unwrap();
    let mut target = common::Offscreen::new(&device, 64);

    // 1) Two-phase GPU meshlet occlusion culling: 0-frame disocclusion recovery in BOTH Forward-Z & Reversed-Z!
    let proj_fwd = pen_render::perspective_forward_z(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
    let proj_rev = perspective_reversed_z(std::f32::consts::FRAC_PI_2, 1.0, 0.1, None);
    let sphere_draw = Draw {
        mesh: 0,
        material: 0,
        transform: [1.4, 0.0, 0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 0.0, -5.0, 1.0],
        parameters: None,
        extra: None,
    };
    let blocker_draw = Draw {
        mesh: 1,
        material: 1,
        transform: [3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.5, 1.0],
        parameters: None,
        extra: None,
    };
    for (label, proj) in [("ForwardZ", proj_fwd), ("ReversedZ", proj_rev)] {
        let mut occluded = common::frame(vec![blocker_draw.clone(), sphere_draw.clone()]);
        occluded.projection = proj;
        occluded.quality.sky = false;
        occluded.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
        wait_ready(&mut renderer, &occluded, &catalog, &mut target);
        // Submit once more so Hi-Z definitely holds the foreground red blocker depth!
        renderer.submit(&occluded, &catalog, &mut target).unwrap();
        let px_occ = common::middle(&device, &queue, &target.0);
        assert!(px_occ[0] > 150 && px_occ[1] < 80, "{label}: blocker must occlude sphere: {px_occ:?}");

        // Next frame: remove blocker! Phase 1 rejects the sphere against prior-frame Hi-Z, and Phase 2 MUST
        // recover it on the EXACT SAME FRAME (0-frame disocclusion lag, no 1-frame black hole!).
        let mut revealed = common::frame(vec![sphere_draw.clone()]);
        revealed.projection = proj;
        revealed.quality.sky = false;
        revealed.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
        renderer.submit(&revealed, &catalog, &mut target).unwrap();
        assert!(
            renderer.last_meshlet_phase2_active(),
            "{label}: two-phase meshlet recovery must execute when prior-frame Hi-Z was active"
        );
        let px_rev = common::middle(&device, &queue, &target.0);
        assert!(
            px_rev[1] > 140 && px_rev[0] < 80,
            "{label}: disoccluded meshlet sphere must appear on the EXACT same frame (0-frame lag): {px_rev:?}"
        );
    }

    // 2) Live Tier-3 ReSTIR DI many-light ray-traced shadows + SHaRC multi-bounce GI:
    //    Backlit white wall at z=-3.0 facing +Z; vertical blocker at x=0 protruding toward +Z divides
    //    the wall into left (x < 0, lit by 16 point lights on the left) and right (x > 0, shadowed!).
    let back_wall = Draw {
        mesh: 1,
        material: 2,
        transform: [4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -3.0, 1.0],
        parameters: None,
        extra: None,
    };
    // Vertical divider perpendicular to the back wall at x=0, spanning z in [-3.0, -0.5]
    let divider = Draw {
        mesh: 1,
        material: 2,
        transform: [0.0, 0.0, -1.5, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.8, 1.0],
        parameters: None,
        extra: None,
    };
    let mut lights = Vec::new();
    for i in 0u32..16 {
        let f = i as f32;
        lights.push(pen_protocol::Light {
            position: Vector { x: -1.6 + (f % 4.0) * 0.15, y: -0.6 + (f / 4.0).floor() * 0.4, z: -1.8 },
            color: Vector { x: 1.4, y: 1.1, z: 0.7 },
            radius: 12.0,
            direction: Vector { x: 0.0, y: 0.0, z: -1.0 },
            inner: -1.0,
            outer: -1.0,
            shadow: false,
            id: i + 1,
        });
    }
    let mut base = common::frame(vec![back_wall, divider]);
    base.projection = proj_fwd;
    base.ambient = Vector { x: 0.0, y: 0.0, z: 0.0 };
    base.light_color = Vector { x: 0.0, y: 0.0, z: 0.0 };
    base.sky = Default::default();
    base.quality.sky = false;
    base.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    base.lights = lights;

    // RT=0 (Forward+ unshadowed point lights): right side is illuminated through the divider!
    let mut f_rt0 = base.clone();
    f_rt0.quality.ray_tracing = 0;
    wait_ready(&mut renderer, &f_rt0, &catalog, &mut target);
    let img_rt0 = common::pixels(&device, &queue, &target.0);
    let luma_at = |img: &[[u8; 4]], x: u32, y: u32| -> f32 {
        let p = img[(y * 64 + x) as usize];
        0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
    };
    let rt0_right = luma_at(&img_rt0, 48, 32);

    // RT=3 (ReSTIR DI + ReSTIR GI + SHaRC + SVGF, opaque clusters zeroed via area_only_group):
    // left side stays brightly lit while right side behind the divider receives crisp RT local shadows!
    let mut f_rt3 = base.clone();
    f_rt3.quality.ray_tracing = 3;
    f_rt3.quality.ddgi = false;
    f_rt3.exposure.cut = true;
    renderer.submit(&f_rt3, &catalog, &mut target).unwrap();
    f_rt3.exposure.cut = false;
    for _ in 0..5 {
        renderer.submit(&f_rt3, &catalog, &mut target).unwrap();
    }
    let img_rt3 = common::pixels(&device, &queue, &target.0);
    let rt3_left = luma_at(&img_rt3, 16, 32);
    let rt3_right = luma_at(&img_rt3, 48, 32);
    assert!(
        rt3_left > 80.0 && rt3_right < rt3_left * 0.45 && rt3_right < rt0_right * 0.65,
        "Tier-3 ReSTIR DI + SHaRC must illuminate lit side brightly and cast RT local shadow behind divider (zero Forward+ leak!): rt0_right={rt0_right:.1}, rt3_left={rt3_left:.1}, rt3_right={rt3_right:.1}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn live_renderer_wires_dag_lods_gpu_particles_lines_jfa_outlines_and_render2d() {
    use pen_render::{
        Camera2d, GPU_PARTICLES_EXT_TAG, GpuParticle, GpuParticleScene, JfaOutlineScene, JfaOutlineTarget,
        LINES_EXT_TAG, LineBatch, LineLayer, LineScene, LineStyle, OutlineStyle, RENDER2D_EXT_TAG,
        RENDER2D_WORLD_EXT_TAG, RadianceCascades2dConfig, RadianceCascades2dLayer, Scene2d, Scene2dEmitter, SdfShape,
        Sprite2d, SpriteLayer2d, TileCell, TilemapLayer2d, TilemapUniform, XRayStyle,
    };

    let Some((device, queue)) = common::featured_device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let dir = std::env::temp_dir().join(format!("pen-live-five-systems-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Dense 1024-triangle UV sphere (>=768 tris -> continuous MeshletDagGpu!) + unit quad + cube.
    let (stacks, slices) = (16u32, 32u32);
    let (mut pos, mut norm, mut uv, mut idx) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for r in 0..=stacks {
        let v = r as f32 / stacks as f32;
        let phi = v * std::f32::consts::PI;
        for c in 0..=slices {
            let u = c as f32 / slices as f32;
            let theta = u * std::f32::consts::TAU;
            let (x, y, z) = (phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
            pos.extend_from_slice(&[x, y, z]);
            norm.extend_from_slice(&[x, y, z]);
            uv.extend_from_slice(&[u, v]);
        }
    }
    let stride = slices + 1;
    for r in 0..stacks {
        for c in 0..slices {
            let (i0, i1, i2, i3) = (r * stride + c, r * stride + c + 1, (r + 1) * stride + c, (r + 1) * stride + c + 1);
            idx.extend_from_slice(&[i0, i1, i2, i1, i3, i2]);
        }
    }
    let sphere_json = json!({ "positions": pos, "normals": norm, "uvs": uv, "indices": idx });
    let quad_json = json!({
        "positions": [-1.0, -1.0, 0.0,  1.0, -1.0, 0.0,  1.0, 1.0, 0.0,  -1.0, 1.0, 0.0],
        "normals": [0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0],
        "uvs": [0.0, 1.0,  1.0, 1.0,  1.0, 0.0,  0.0, 0.0],
        "indices": [0, 1, 2, 0, 2, 3]
    });
    std::fs::write(dir.join("sphere_dag.mesh.json"), serde_json::to_vec(&sphere_json).unwrap()).unwrap();
    std::fs::write(dir.join("quad.mesh.json"), serde_json::to_vec(&quad_json).unwrap()).unwrap();
    std::fs::write(dir.join("simple.wgsl"), SIMPLE_SHADER).unwrap();
    write_solid_png(&dir.join("white.png"), [255, 255, 255, 255]);

    let mats = [
        (
            "blue.mat.json",
            json!({ "shader": "simple.wgsl", "texture": "white.png", "color": [0.08, 0.25, 0.9, 1.0], "lighting": "unlit" }),
        ),
        (
            "red.mat.json",
            json!({ "shader": "simple.wgsl", "texture": "white.png", "color": [0.9, 0.08, 0.08, 1.0], "lighting": "unlit" }),
        ),
    ];
    for (name, val) in &mats {
        std::fs::write(dir.join(name), val.to_string()).unwrap();
    }
    write_solid_png(&dir.join("world_only.png"), [20, 220, 255, 255]);
    std::fs::write(
        dir.join("world_only.mat.json"),
        json!({ "shader": "simple.wgsl", "texture": "world_only.png", "color": [1.0, 1.0, 1.0, 1.0], "lighting": "unlit" }).to_string(),
    )
    .unwrap();
    let catalog = Catalog {
        meshes: vec!["sphere_dag.mesh.json".into(), "quad.mesh.json".into()],
        materials: mats.iter().map(|(n, _)| (*n).into()).chain(["world_only.mat.json".into()]).collect(),
        sounds: vec![],
    };
    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions {
            format: common::FORMAT,
            display: pen_render::Display::SDR,
            assets: dir.clone(),
            veil: None,
            pipeline_cache: None,
        },
    )
    .unwrap();
    let mut target = common::Offscreen::new(&device, 128);
    let proj_fwd = pen_render::perspective_forward_z(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
    let proj_rev = perspective_reversed_z(std::f32::consts::FRAC_PI_2, 1.0, 0.1, None);

    // 1) Continuous Meshlet DAG LOD culling in BOTH Forward-Z & Reversed-Z + fallback when disabled.
    let sphere_draw = Draw {
        mesh: 0,
        material: 0,
        transform: [1.2, 0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, -4.0, 1.0],
        parameters: None,
        extra: None,
    };
    for (label, proj) in [("ForwardZ", proj_fwd), ("ReversedZ", proj_rev)] {
        let mut f = common::frame(vec![sphere_draw.clone()]);
        f.projection = proj;
        f.quality.sky = false;
        f.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
        wait_ready(&mut renderer, &f, &catalog, &mut target);
        assert_eq!(renderer.last_meshlet_dag_draws(), 1, "{label}: dense sphere must use continuous DAG LODs");
        let mid = common::middle(&device, &queue, &target.0);
        assert!(mid[2] > 140 && mid[0] < 90, "{label}: DAG sphere center must render blue: {mid:?}");
    }
    renderer.set_dag_lod_error_threshold(-1.0);
    let mut f_nodag = common::frame(vec![sphere_draw.clone()]);
    f_nodag.projection = proj_fwd;
    f_nodag.quality.sky = false;
    f_nodag.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    wait_ready(&mut renderer, &f_nodag, &catalog, &mut target);
    assert_eq!(renderer.last_meshlet_dag_draws(), 0);
    assert_eq!(renderer.last_meshlet_cull_draws(), 1, "disabling DAG must fall back to meshlet culling");
    renderer.set_dag_lod_error_threshold(1.5);

    // 2) Live JFA Silhouette Outlines (both 1x and MSAA 4x depth-resolve path!).
    let center_quad = Draw {
        mesh: 1,
        material: 0,
        transform: [0.45, 0.0, 0.0, 0.0, 0.0, 0.45, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -2.0, 1.0],
        parameters: None,
        extra: None,
    };
    renderer.set_jfa_outlines(JfaOutlineScene {
        styles: vec![OutlineStyle {
            color: [2.0, 1.8, 0.05, 1.0],
            occluded_color: [0.1, 0.5, 0.6, 1.0],
            width_px: 5.0,
            feather_px: 1.0,
            fill_alpha: 0.0,
        }],
        targets: vec![JfaOutlineTarget { draw_index: 0, style_index: 0 }],
        max_radius_px: 8,
        depth_bias: 1e-4,
    });
    for msaa in [1u32, 4u32] {
        let mut f_jfa = common::frame(vec![center_quad.clone()]);
        f_jfa.projection = proj_fwd;
        f_jfa.quality.sky = false;
        f_jfa.quality.msaa = msaa;
        f_jfa.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
        wait_ready(&mut renderer, &f_jfa, &catalog, &mut target);
        let px = common::pixels(&device, &queue, &target.0);
        let at = |x: u32, y: u32| px[(y * 128 + x) as usize];
        // Center (64,64) stays blue interior; quad half-width at z=-2 with fov=90 deg is 0.45/2 * 64 = 14.4 px -> x in [50..78].
        let interior = at(64, 64);
        assert!(interior[2] > 140 && interior[0] < 90, "MSAA {msaa}: JFA interior must stay blue: {interior:?}");
        // Ring pixel 2 px right of quad edge (x=81, y=64) must glow bright yellow!
        let ring = at(81, 64);
        assert!(ring[0] > 180 && ring[1] > 150 && ring[2] < 90, "MSAA {msaa}: JFA ring must be yellow: {ring:?}");
        // Far exterior (110, 64) must remain black.
        let far = at(110, 64);
        assert!(far[0] < 10 && far[1] < 10 && far[2] < 10, "MSAA {msaa}: far background must stay black: {far:?}");
    }
    renderer.clear_jfa_outlines();

    // 3) Live 3D Depth-Tested + X-Ray Lines & 2D Screen-Space Lines (both API & LINES_EXT_TAG).
    let right_blocker = Draw {
        mesh: 1,
        material: 1,
        transform: [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.5, 1.0],
        parameters: None,
        extra: None,
    };
    let mut batch = LineBatch::new();
    batch.segment(
        [-2.0, 0.0, -2.5],
        [2.0, 0.0, -2.5],
        [0.0, 1.6, 1.6, 1.0],
        [0.0, 1.6, 1.6, 1.0],
        LineStyle { width_px: 6.0, layer: LineLayer::XRay, ..Default::default() },
    );
    batch.segment_2d(
        [8.0, 12.0],
        [120.0, 12.0],
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        LineStyle { width_px: 4.0, ..Default::default() },
    );
    renderer.set_line_scene(LineScene {
        batch,
        xray: XRayStyle { alpha: 0.45, desaturate: 0.0, solid_line_dash_px: None },
    });
    let mut f_lines = common::frame(vec![right_blocker.clone()]);
    f_lines.projection = proj_fwd;
    f_lines.quality.sky = false;
    f_lines.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    wait_ready(&mut renderer, &f_lines, &catalog, &mut target);
    let px = common::pixels(&device, &queue, &target.0);
    let at = |x: u32, y: u32| px[(y * 128 + x) as usize];
    let left_line = at(32, 64);
    let right_xray = at(96, 64);
    let right_wall_only = at(96, 96);
    let top_2d = at(64, 12);
    assert!(left_line[1] > 180 && left_line[2] > 180, "unoccluded 3D line must be bright cyan: {left_line:?}");
    assert!(
        right_xray[1] > right_wall_only[1] + 25 && right_xray[2] > right_wall_only[2] + 25,
        "X-ray occluded portion must shine through red blocker: xray={right_xray:?} wall={right_wall_only:?}"
    );
    assert!(top_2d[0] > 200 && top_2d[1] > 200 && top_2d[2] < 40, "2D overlay line must be yellow: {top_2d:?}");
    renderer.clear_line_scene();

    // Verify LINES_EXT_TAG also renders a 2D line via RenderFrame::extensions alone.
    let mut f_line_ext = common::frame(vec![]);
    f_line_ext.projection = proj_fwd;
    f_line_ext.quality.sky = false;
    f_line_ext.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    f_line_ext.extensions.push(Extension {
        tag: LINES_EXT_TAG,
        values: vec![
            0.3, 0.0, // xray_alpha, desaturate
            10.0, 64.0, 0.0, 6.0, // p0, width_px
            118.0, 64.0, 0.0, 0.0, // p1, dash_len
            0.0, 1.0, 0.0, 1.0, // green
            0.0, 1.0, 0.0, 1.0, // gap, feather, layer, is_2d=1
        ],
    });
    renderer.submit(&f_line_ext, &catalog, &mut target).unwrap();
    let mid_ext_line = common::middle(&device, &queue, &target.0);
    assert!(mid_ext_line[1] > 180 && mid_ext_line[0] < 40, "LINES_EXT_TAG must draw green 2D line: {mid_ext_line:?}");

    // The first edge is 22 px, so the next edge starts partway into a 20 px dash period.
    f_line_ext.extensions[0].values = vec![
        0.3, 0.0, 10.0, 64.0, 0.0, 2.0, 32.0, 64.0, 0.0, 8.0, 0.0, 1.0, 0.0, 1.0, 12.0, 1.0, 0.0, 1.0, 32.0, 64.0, 0.0,
        2.0, 100.0, 64.0, 0.0, 8.0, 0.0, 1.0, 0.0, 1.0, 12.0, 1.0, 0.0, 3.0,
    ];
    renderer.submit(&f_line_ext, &catalog, &mut target).unwrap();
    let dashed = common::pixels(&device, &queue, &target.0);
    assert!(dashed[64 * 128 + 38][1] < 10, "continuation should already be in the gap");
    assert!(dashed[64 * 128 + 52][1] > 180, "next dash should start at the continued phase");

    // 4) Live GPU Mesh Particles (via API & GPU_PARTICLES_EXT_TAG).
    let mut p_scene = GpuParticleScene {
        max_particles: 64,
        gravity: [0.0, -25.0, 0.0],
        dt: 1.0 / 30.0,
        curl_strength: 0.0,
        depth_collide: true,
        ambient: 1.2,
        ..GpuParticleScene::default()
    };
    p_scene.pending_seed =
        Some(vec![GpuParticle::primary([0.0, 0.0, -2.2], [0.0, 0.0, 0.0], 5.0, 0.9, [1.8, 0.6, 0.05, 1.0])]);
    renderer.set_gpu_particle_scene(p_scene);
    let mut f_gp = common::frame(vec![]);
    f_gp.projection = proj_fwd;
    f_gp.quality.sky = false;
    f_gp.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    renderer.submit(&f_gp, &catalog, &mut target).unwrap();
    let p_mid = common::middle(&device, &queue, &target.0);
    assert!(p_mid[0] > 140 && p_mid[1] > 50 && p_mid[2] < 60, "live GPU particle cube must render orange: {p_mid:?}");
    renderer.clear_gpu_particle_scene();

    // Smoke-check GPU_PARTICLES_EXT_TAG extension decoding as well.
    let mut f_gp_ext = f_gp.clone();
    let mut ext_vals = vec![
        64.0,
        1.0 / 60.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5, // max, dt, gravity, drag, curl
        0.5,
        0.2,
        0.3,
        17.0,
        0.0,
        2.0,
        0.5,
        1.2, // rest, fric, thick, flags(collide+reseed), subemit, ambient
    ];
    ext_vals.extend_from_slice(&[
        0.0, 0.0, -2.2, 0.0, // pos_age
        0.0, 0.0, 0.0, 5.0, // vel_life
        0.1, 1.8, 0.2, 1.0, // green color
        0.9, 0.0, 0.0, 0.0, // size_rot_gen
    ]);
    f_gp_ext.extensions.push(Extension { tag: GPU_PARTICLES_EXT_TAG, values: ext_vals });
    renderer.submit(&f_gp_ext, &catalog, &mut target).unwrap();
    let p_ext_mid = common::middle(&device, &queue, &target.0);
    assert!(p_ext_mid[1] > 140 && p_ext_mid[0] < 70, "GPU_PARTICLES_EXT_TAG must render green cube: {p_ext_mid:?}");

    let mut split_particles = f_gp_ext.clone();
    let args = &mut split_particles.extensions[0].values;
    args[1] = 0.075;
    args[11] = 1.0;
    args[23] = 0.12;
    for (x, projection) in [(0.0, proj_fwd), (0.5, proj_rev)] {
        split_particles.camera_targets.push(CameraTargetSpec {
            viewport_x: x,
            viewport_y: 0.0,
            viewport_w: 0.5,
            viewport_h: 1.0,
            split_screen: true,
            view: common::IDENTITY,
            projection,
            ..Default::default()
        });
    }
    renderer.submit(&split_particles, &catalog, &mut target).unwrap();
    let split = common::pixels(&device, &queue, &target.0);
    for x in [32, 96] {
        assert!(split[64 * 128 + x][1] > 140, "both cameras must draw the same live particle");
    }

    let mut msaa_particles = f_gp_ext.clone();
    let args = &mut msaa_particles.extensions[0].values;
    args[1] = 0.1;
    args[11] = 1.0;
    args[20] = 10.0;
    renderer.submit(&msaa_particles, &catalog, &mut target).unwrap();
    let before_msaa = common::pixels(&device, &queue, &target.0);
    assert!(before_msaa[64 * 128 + 93][1] > 140);
    msaa_particles.quality.msaa = 4;
    msaa_particles.extensions[0].values[1] = 0.0;
    renderer.submit(&msaa_particles, &catalog, &mut target).unwrap();
    let after_msaa = common::pixels(&device, &queue, &target.0);
    assert!(after_msaa[64 * 128 + 93][1] > 140);
    assert!(after_msaa[64 * 128 + 64][1] < 10, "MSAA switch must preserve the simulated position");

    f_gp_ext.extensions[0].values.truncate(16);
    renderer.submit(&f_gp_ext, &catalog, &mut target).unwrap();
    let cleared = common::middle(&device, &queue, &target.0);
    assert!(cleared[..3].iter().all(|&channel| channel < 10), "empty reseed must clear the buffer");

    // With backface culling enabled, a front-only mesh must survive both basis and spin.
    let mut face_particle = GpuParticle::primary([0.0, 0.0, -2.2], [0.0; 3], 5.0, 1.0, [1.8, 0.4, 0.0, 1.0]);
    face_particle.size_rot_gen[1] = 0.4;
    renderer.set_gpu_particle_scene(GpuParticleScene {
        max_particles: 1,
        mesh_vertices: [[0.0, -0.5, 0.5], [0.0, -0.5, -0.5], [0.0, 0.5, 0.0]]
            .map(|position| pen_render::GpuParticleMeshVertex { position, normal: [1.0, 0.0, 0.0] })
            .into(),
        mesh_indices: vec![0, 1, 2],
        dt: 0.0,
        ambient: 1.2,
        pending_seed: Some(vec![face_particle]),
        ..Default::default()
    });
    renderer.submit(&f_gp, &catalog, &mut target).unwrap();
    assert!(common::middle(&device, &queue, &target.0)[0] > 180, "spun particle front face must survive culling");
    renderer.clear_gpu_particle_scene();

    // 5) Live First-Class 2D Renderer Suite (Tilemap + Sprites2d + 2D Radiance Cascades GI + SDF UI + RENDER2D_EXT_TAG).
    let white_atlas = renderer.upload_2d_texture_rgba8(2, 2, &[255u8; 16]);
    let cells = renderer.upload_tilemap_cells(&[TileCell::new(1, false, false, false, 1); 4]);
    let cam2d = Camera2d { center: [0.0, 0.0], zoom: 1.0, rotation: 0.0, viewport: [128.0, 128.0] };
    renderer.set_scene_2d(Scene2d {
        camera: cam2d,
        tilemaps: vec![TilemapLayer2d {
            uniform: TilemapUniform {
                camera_vp: cam2d.world_to_ndc(),
                origin_and_tile_size: [-64.0, 64.0, 64.0, 64.0],
                grid_size_and_atlas: [2, 2, 1, 1],
                anim_time_and_tint: [0.0, 1.0, 0.0, 0.0, 0.9, 0.9, 0.9, 1.0],
            },
            cells,
            atlas: white_atlas.clone(),
            sampler: None,
        }],
        sprite_layers: vec![SpriteLayer2d {
            sprites: vec![Sprite2d {
                pos: [-32.0, 0.0],
                size: [12.0, 12.0],
                color: [1.0, 1.0, 1.0, 1.0],
                ..Default::default()
            }],
            atlas: white_atlas,
            sampler: None,
            y_sort: true,
        }],
        radiance_cascades: Some(RadianceCascades2dLayer {
            emitters: vec![
                Scene2dEmitter::Circle { center: [24.0, 64.0], radius: 10.0, material: [5.0, 3.5, 1.0, 1.0] },
                Scene2dEmitter::Rect { rect: [60.0, 16.0, 10.0, 96.0], material: [0.0, 0.0, 0.0, 1.0] },
            ],
            scene_view: None,
            config: RadianceCascades2dConfig::default(),
            composite_weights: [0.0, 1.0, 0.0],
            opaque_backdrop: true,
        }),
        sdf_shapes: vec![SdfShape {
            rect: [8.0, 8.0, 32.0, 20.0],
            corner_radii: [6.0; 4],
            fill_color: [0.1, 0.2, 0.95, 1.0],
            border_color: [1.0, 1.0, 1.0, 1.0],
            border_width_px: 2.0,
            ..Default::default()
        }],
        sdf_overlay_shapes: vec![],
    });
    let mut f_2d = common::frame(vec![]);
    f_2d.projection = proj_fwd;
    f_2d.quality.sky = false;
    f_2d.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    // Also attach an SDF overlay badge in the bottom-right via RENDER2D_EXT_TAG!
    f_2d.extensions.push(Extension {
        tag: RENDER2D_EXT_TAG,
        values: vec![
            92.0, 92.0, 28.0, 28.0, // rect
            4.0, 4.0, 4.0, 4.0, // corner_radii
            0.0, 0.95, 0.2, 1.0, // fill_color (green)
            0.0, 0.95, 0.2, 1.0, // fill_color_end
            0.0, 0.0, 0.0, 0.0, // border_color
            0.0, 0.0, 0.0, 0.0, // shadow_color
            16.0, 0.0, 0.0, 0.0, // kind=16 (overlay rounded rect), border, grad
            0.0, 0.0, 0.0, 0.0, // shadow_off, blur, clip_r
            0.0, 0.0, 0.0, 0.0, // clip_rect
        ],
    });
    renderer.set_devtools_sdf_overlay(vec![SdfShape {
        rect: [48.0, 8.0, 32.0, 20.0],
        fill_color: [0.95, 0.0, 0.0, 1.0],
        ..Default::default()
    }]);
    renderer.submit(&f_2d, &catalog, &mut target).unwrap();
    let px_2d = common::pixels(&device, &queue, &target.0);
    let at2d = |x: u32, y: u32| px_2d[(y * 128 + x) as usize];
    let rc_lit = at2d(42, 64);
    let rc_umbra = at2d(90, 64);
    let sdf_card = at2d(24, 18);
    let sdf_ext = at2d(106, 106);
    let sdf_devtools = at2d(64, 18);
    assert!(sdf_devtools[0] > 180 && sdf_devtools[1] < 40);
    assert!(
        rc_lit[0] > 120 && rc_umbra[0] < 15,
        "2D RC GI must light left and shadow right behind wall: lit={rc_lit:?} umbra={rc_umbra:?}"
    );
    assert!(sdf_card[2] > 180 && sdf_card[0] < 60, "SDF UI card must render blue fill: {sdf_card:?}");
    assert!(sdf_ext[1] > 180 && sdf_ext[0] < 40, "RENDER2D_EXT_TAG SDF badge must render green: {sdf_ext:?}");
    renderer.clear_scene_2d();

    // 5b) Verify RENDER2D_WORLD_EXT_TAG (65_009) drives full 2D tilemap + sprite + 2D Radiance Cascades GI
    //     purely from RenderFrame::extensions (zero Rust set_scene_2d!).
    let mut world_ext: Vec<f32> = vec![
        0.0, 0.0, 1.0, 0.0, // camera center, zoom, rot
        1.0, 1.0, 2.0, 5.0, // 1 tilemap, 1 sprite, 2 RC emitters, flags=1(opaque RC) | 4(y_sort)
        2.0, 4.0, 0.0, 0.0, 1.0, 0.0, // RC config: spacing=2, dirs=4, auto cascades, ambient=0, mul=1, add=0
        // Tilemap header (15 floats): mat=-1 (white), origin(-64,64), tile(64,64), grid(2,2), atlas(1,1), anim(0,1), tint(0.9)
        -1.0, -64.0, 64.0, 64.0, 64.0, 2.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0.9, 0.9, 0.9, 1.0,
        // 4 cells (id=1, packed as f32)
        1.0, 1.0, 1.0, 1.0,
        // 1 sprite (25 floats): white mat=-1, pos(-32,0), size(12,12), pivot(0.5,0.5), rot=0, layer=0, uv(0,0,1,1), color(1), slice(0), flags(0)
        -1.0, -32.0, 0.0, 12.0, 12.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, // 2 RC emitters (9 floats each): Circle light left + opaque Rect wall center
        0.0, 24.0, 64.0, 10.0, 0.0, 5.0, 3.5, 1.0, 1.0, 1.0, 60.0, 16.0, 10.0, 96.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let mut f_2d_ext = common::frame(vec![]);
    f_2d_ext.projection = proj_fwd;
    f_2d_ext.quality.sky = false;
    f_2d_ext.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    f_2d_ext.extensions.push(Extension { tag: RENDER2D_WORLD_EXT_TAG, values: std::mem::take(&mut world_ext) });
    renderer.submit(&f_2d_ext, &catalog, &mut target).unwrap();
    let px_2d_ext = common::pixels(&device, &queue, &target.0);
    let ext_lit = px_2d_ext[(64 * 128 + 42) as usize];
    let ext_umbra = px_2d_ext[(64 * 128 + 90) as usize];
    assert!(px_2d_ext[18 * 128 + 64][0] > 180, "devtools overlay must coexist with the world extension");
    let values = &mut f_2d_ext.extensions[0].values;
    values[33] = 2.0;
    wait_ready(&mut renderer, &f_2d_ext, &catalog, &mut target);
    let atlas_pixel = common::pixels(&device, &queue, &target.0)[64 * 128 + 32];
    assert!(
        atlas_pixel[1] > atlas_pixel[0].saturating_add(50),
        "extension-only atlas must load while the devtools overlay is active: {atlas_pixel:?}"
    );
    renderer.clear_devtools_sdf_overlay();
    assert!(
        ext_lit[0] > 120 && ext_umbra[0] < 15,
        "RENDER2D_WORLD_EXT_TAG must drive 2D tilemap + RC2D GI purely from extensions: lit={ext_lit:?} umbra={ext_umbra:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn live_renderer_dag_two_phase_hiz_recovers_disocclusion_same_frame() {
    let Some((device, queue)) = common::featured_device() else {
        eprintln!("skipping live_renderer_dag_two_phase_hiz_recovers_disocclusion_same_frame: no Vulkan GPU");
        return;
    };
    let dir = std::env::temp_dir().join(format!("pen_dag2p_live_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // 1024-triangle UV sphere (16 stacks x 32 slices -> auto-builds continuous MeshletDagGpu).
    let (stacks, slices) = (16u32, 32u32);
    let mut pos = Vec::new();
    let mut norm = Vec::new();
    let mut uv = Vec::new();
    let mut idx = Vec::new();
    for r in 0..=stacks {
        let v = r as f32 / stacks as f32;
        let phi = v * std::f32::consts::PI;
        for c in 0..=slices {
            let u = c as f32 / slices as f32;
            let theta = u * std::f32::consts::TAU;
            let (x, y, z) = (phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
            pos.extend_from_slice(&[x, y, z]);
            norm.extend_from_slice(&[x, y, z]);
            uv.extend_from_slice(&[u, v]);
        }
    }
    let stride = slices + 1;
    for r in 0..stacks {
        for c in 0..slices {
            let (i0, i1, i2, i3) = (r * stride + c, r * stride + c + 1, (r + 1) * stride + c, (r + 1) * stride + c + 1);
            idx.extend_from_slice(&[i0, i1, i2, i1, i3, i2]);
        }
    }
    let sphere_json = json!({ "positions": pos, "normals": norm, "uvs": uv, "indices": idx });
    let quad_json = json!({
        "positions": [-1.0, -1.0, 0.0,  1.0, -1.0, 0.0,  1.0, 1.0, 0.0,  -1.0, 1.0, 0.0],
        "normals": [0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0],
        "uvs": [0.0, 1.0,  1.0, 1.0,  1.0, 0.0,  0.0, 0.0],
        "indices": [0, 1, 2, 0, 2, 3]
    });
    std::fs::write(dir.join("sphere_dag.mesh.json"), serde_json::to_vec(&sphere_json).unwrap()).unwrap();
    std::fs::write(dir.join("quad.mesh.json"), serde_json::to_vec(&quad_json).unwrap()).unwrap();
    std::fs::write(dir.join("simple.wgsl"), SIMPLE_SHADER).unwrap();
    write_solid_png(&dir.join("white.png"), [255, 255, 255, 255]);
    std::fs::write(
        dir.join("green.mat.json"),
        json!({ "shader": "simple.wgsl", "texture": "white.png", "color": [0.08, 0.92, 0.15, 1.0], "lighting": "unlit" })
            .to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.join("red.mat.json"),
        json!({ "shader": "simple.wgsl", "texture": "white.png", "color": [0.92, 0.08, 0.08, 1.0], "lighting": "unlit" })
            .to_string(),
    )
    .unwrap();

    let catalog = Catalog {
        meshes: vec!["sphere_dag.mesh.json".into(), "quad.mesh.json".into()],
        materials: vec!["green.mat.json".into(), "red.mat.json".into()],
        sounds: vec![],
    };
    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions {
            format: common::FORMAT,
            display: pen_render::Display::SDR,
            assets: dir.clone(),
            veil: None,
            pipeline_cache: None,
        },
    )
    .unwrap();
    let mut target = common::Offscreen::new(&device, 128);
    let proj_fwd = pen_render::perspective_forward_z(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
    let proj_rev = perspective_reversed_z(std::f32::consts::FRAC_PI_2, 1.0, 0.1, None);

    let dag_sphere = Draw {
        mesh: 0,
        material: 0,
        transform: [1.1, 0.0, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, -4.5, 1.0],
        parameters: None,
        extra: None,
    };
    let blocker = Draw {
        mesh: 1,
        material: 1,
        transform: [2.5, 0.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.5, 1.0],
        parameters: None,
        extra: None,
    };

    for (label, proj) in [("ForwardZ", proj_fwd), ("ReversedZ", proj_rev)] {
        // Frame 1: red wall completely occludes the green DAG sphere & primes Hi-Z with close depth.
        let mut f_occluded = common::frame(vec![blocker.clone(), dag_sphere.clone()]);
        f_occluded.projection = proj;
        f_occluded.quality.sky = false;
        f_occluded.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
        wait_ready(&mut renderer, &f_occluded, &catalog, &mut target);
        // Submit one settled occluded frame to ensure Hi-Z pyramid matches `proj`'s depth mode.
        renderer.submit(&f_occluded, &catalog, &mut target).unwrap();
        let occ_mid = common::middle(&device, &queue, &target.0);
        assert!(occ_mid[0] > 180 && occ_mid[1] < 90, "{label}: frame 1 must show red blocker: {occ_mid:?}");

        // Frame 2: blocker disappears. Phase 1 sees stale near Hi-Z and defers DAG clusters;
        // Phase 2 rebuilds current-frame Hi-Z and recovers the green DAG sphere on the SAME frame (0-frame lag).
        let mut f_disoccluded = common::frame(vec![dag_sphere.clone()]);
        f_disoccluded.projection = proj;
        f_disoccluded.quality.sky = false;
        f_disoccluded.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
        renderer.submit(&f_disoccluded, &catalog, &mut target).unwrap();
        assert_eq!(renderer.last_meshlet_dag_draws(), 1, "{label}: sphere must render via continuous DAG");
        assert!(
            renderer.last_meshlet_phase2_active(),
            "{label}: two-phase Hi-Z DAG disocclusion recovery must be active on frame 2"
        );
        let rec_mid = common::middle(&device, &queue, &target.0);
        assert!(
            rec_mid[1] > 160 && rec_mid[0] < 90,
            "{label}: disoccluded DAG sphere must appear green on the exact same frame: {rec_mid:?}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}
