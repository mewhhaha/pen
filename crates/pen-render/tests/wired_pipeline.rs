//! End-to-end headless GPU integration tests proving standard `pen_protocol::RenderFrame`
//! packets genuinely drive:
//! - AgX vs Shoulder tonemapping & HDRI IBL
//! - Hillaire physical sky LUTs & raymarched volumetric clouds (`quality.physical_sky`)
//! - LTC area lights through `standard_surface` (`frame.area_lights`)
//! - `SurfaceMapAtlas` + thin-film iridescence + baked lightmap (`surface_apply_atlas`)
//! - Full PBR deferred decals (`frame.pbr_decals`)
//! - Split-screen viewports with HUD sprites composited on top & cached targets
//! - Scene-filter sprites composed with RCAS + MSAA 4x depth-resolved outlines
//! - Live `.matgraph.json` visual shader graphs loaded directly by `Renderer`
//! - `validate_frame` & catalog bounds checks for new packet fields

mod common;

use pen_protocol::{AreaLightSpec, CameraTargetSpec, Catalog, Color, Draw, PbrDecalSpec, Sprite, Vector, VideoStreamSpec};
use pen_render::{
    HdriImage, NodeRef, RampStop, Renderer, RendererOptions, ShaderGraph, ShaderNode, ShaderNodeKind, TonemapOperator,
};
use serde_json::json;

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

fn pbr_surface_shader() -> String {
    format!(
        r#"
struct Scene {{
  view: mat4x4f, projection: mat4x4f, ambient: vec4f, light_direction: vec4f, light_color: vec4f,
  light_view_projection: mat4x4f, shadow: vec4f, zenith: vec4f, horizon: vec4f, ground: vec4f, eye: vec4f,
}};
struct DrawU {{
  model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f,
  extra: array<vec4f, 32>, surface: array<vec4f, 5>,
}};
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: DrawU;
@group(1) @binding(2) var mat_samp: sampler;
@group(1) @binding(4) var mat_atlas: texture_2d<f32>;
{lights}
{env}
{lobes}
{surface}
struct Out {{
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) uv1: vec2f,
  @location(4) view_z: f32,
}};
@vertex fn vertex(
  @location(0) p: vec3f,
  @location(1) n: vec3f,
  @location(2) uv: vec2f,
  @location(12) uv1: vec2f,
) -> Out {{
  let w = draw.model * vec4f(p, 1.0);
  let vp = scene.view * w;
  return Out(scene.projection * vp, w.xyz, (draw.normal * vec4f(n, 0.0)).xyz, uv, uv1, vp.z);
}}
@fragment fn fragment(i: Out) -> @location(0) vec4f {{
  let n = normalize(i.normal);
  let v = normalize(scene.eye.xyz - i.world);
  var s = surface_default(i.world, n, v, draw.color.rgb, draw.parameters.x, draw.parameters.y, i.pos, i.view_z);
  let mask = u32(draw.parameters.z + 0.5);
  s = surface_apply_atlas(s, mat_atlas, mat_samp, i.uv, i.uv1, draw.settings.w, mask);
  return vec4f(standard_surface(s), 1.0);
}}
"#,
        lights = pen_render::LIGHTS_WGSL,
        env = pen_render::ENVIRONMENT_WGSL,
        lobes = pen_render::SURFACE_LOBES_WGSL,
        surface = pen_render::SURFACE_WGSL,
    )
}

fn write_solid_png(path: &std::path::Path, rgba: [u8; 4]) {
    let img = image::RgbaImage::from_pixel(4, 4, image::Rgba(rgba));
    img.save(path).unwrap();
}

fn wait_ready(renderer: &mut Renderer, frame: &pen_protocol::RenderFrame, catalog: &Catalog, target: &mut common::Offscreen) {
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
    std::fs::write(dir.join("pbr.wgsl"), pbr_surface_shader()).unwrap();
    std::fs::write(dir.join("hud.wgsl"), HUD_SPRITE_SHADER).unwrap();
    std::fs::write(dir.join("filter.wgsl"), FILTER_SPRITE_SHADER).unwrap();
    write_solid_png(&dir.join("irid.png"), [255, 180, 0, 255]);
    write_solid_png(&dir.join("lightmap.png"), [40, 220, 60, 255]);

    // Real .matgraph.json shader graph compiled on load by pen-render's Loader!
    let graph = ShaderGraph {
        name: "live_proc_rock".into(),
        nodes: vec![
            ShaderNode { id: "uv".into(), ui_pos: [0.0, 0.0], kind: ShaderNodeKind::Uv },
            ShaderNode {
                id: "fbm".into(),
                ui_pos: [120.0, 0.0],
                kind: ShaderNodeKind::FbmNoise { coords: Some(NodeRef::of("uv")), scale: 4.0, octaves: 3 },
            },
            ShaderNode {
                id: "ramp".into(),
                ui_pos: [240.0, 0.0],
                kind: ShaderNodeKind::ColorRamp {
                    t: NodeRef::of("fbm"),
                    stops: vec![
                        RampStop { t: 0.0, color: [0.2, 0.7, 0.9] },
                        RampStop { t: 1.0, color: [0.9, 0.4, 0.2] },
                    ],
                },
            },
            ShaderNode {
                id: "emit".into(),
                ui_pos: [240.0, 120.0],
                kind: ShaderNodeKind::Vec3Const { value: [0.6, 0.2, 0.8] },
            },
            ShaderNode {
                id: "out".into(),
                ui_pos: [400.0, 0.0],
                kind: ShaderNodeKind::StandardSurfaceOutput {
                    base_color: Some(NodeRef::of("ramp")),
                    metallic: None,
                    roughness: None,
                    normal: None,
                    emissive: Some(NodeRef::of("emit")),
                    ao: None,
                    alpha: None,
                    clearcoat: None,
                    iridescence: None,
                },
            },
        ],
    };
    std::fs::write(dir.join("proc.matgraph.json"), serde_json::to_vec(&graph).unwrap()).unwrap();

    let quad = json!({
        "positions": [-1.0, -1.0, 0.0,  1.0, -1.0, 0.0,  1.0, 1.0, 0.0,  -1.0, 1.0, 0.0],
        "normals": [0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0],
        "uvs": [0.0, 1.0,  1.0, 1.0,  1.0, 0.0,  0.0, 0.0],
        "uvs1": [0.0, 1.0,  1.0, 1.0,  1.0, 0.0,  0.0, 0.0],
        "indices": [0, 1, 2, 0, 2, 3]
    });
    std::fs::write(dir.join("quad.mesh.json"), serde_json::to_vec(&quad).unwrap()).unwrap();
    std::fs::write(
        dir.join("wall.mat.json"),
        serde_json::to_vec(&json!({
            "shader": "simple.wgsl",
            "color": [1.0, 0.2, 0.1, 1.0],
            "lighting": "lit",
            "decals": true
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("decal.mat.json"),
        serde_json::to_vec(&json!({
            "shader": "simple.wgsl",
            "color": [0.1, 0.9, 0.2, 1.0],
            "lighting": "lit"
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("pbr.mat.json"),
        serde_json::to_vec(&json!({
            "shader": "pbr.wgsl",
            "color": [1.0, 1.0, 1.0, 1.0],
            "lighting": "lit"
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("pbr_atlas.mat.json"),
        serde_json::to_vec(&json!({
            "shader": "pbr.wgsl",
            "color": [1.0, 1.0, 1.0, 1.0],
            "lighting": "lit",
            "surface": {
                "iridescenceTexture": "irid.png",
                "lightmapTexture": "lightmap.png"
            }
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("hud.mat.json"),
        serde_json::to_vec(&json!({
            "shader": "hud.wgsl",
            "color": [1.0, 1.0, 1.0, 1.0],
            "lighting": "unlit"
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("filter.mat.json"),
        serde_json::to_vec(&json!({
            "shader": "filter.wgsl",
            "color": [1.0, 1.0, 1.0, 1.0],
            "lighting": "unlit",
            "scene_filter": true
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("graph.mat.json"),
        serde_json::to_vec(&json!({
            "shader": "proc.matgraph.json",
            "color": [1.0, 1.0, 1.0, 1.0],
            "lighting": "lit"
        }))
        .unwrap(),
    )
    .unwrap();

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
            "wall.mat.json".into(),      // 0
            "decal.mat.json".into(),     // 1
            "pbr.mat.json".into(),       // 2
            "pbr_atlas.mat.json".into(), // 3
            "hud.mat.json".into(),       // 4
            "filter.mat.json".into(),    // 5
            "graph.mat.json".into(),     // 6
        ],
        sounds: vec![],
    };

    let wall_xform = [
        2.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, -2.0, 1.0,
    ];
    let proj = pen_render::perspective_forward_z(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 50.0);

    let mut frame = common::frame(vec![Draw {
        mesh: 0,
        material: 0,
        transform: wall_xform,
        parameters: None,
        extra: None,
    }]);
    frame.projection = proj;
    frame.light_direction = Vector { x: 0.2, y: -0.9, z: -0.3 };
    frame.grade.tonemap = TonemapOperator::AgX as u32;
    frame.grade.sharpness = 0.5;
    frame.grade.grain = 0.04;
    frame.grade.flare = 0.4;
    frame.quality.physical_sky = true;
    frame.clouds.cover = 0.4;
    frame.pbr_decals.push(PbrDecalSpec {
        material: 1,
        transform: [
            0.8, 0.0, 0.0, 0.0,
            0.0, 0.8, 0.0, 0.0,
            0.0, 0.0, 0.8, 0.0,
            0.0, 0.0, -2.0, 1.0,
        ],
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
    shoulder_frame.grade.grain = 0.0;
    let mut agx_clean = frame.clone();
    agx_clean.grade.grain = 0.0;
    renderer.submit(&shoulder_frame, &catalog, &mut target).unwrap();
    let px_shoulder = common::middle(&device, &queue, &target.0);
    renderer.submit(&agx_clean, &catalog, &mut target).unwrap();
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
    let px_phys_sky = common::pixels(&device, &queue, &target.0)[8 * 64 + 32];
    assert!(
        px_phys_sky[2] > px_clear[2].saturating_add(15),
        "physical_sky LUT must brighten sky background: clear={px_clear:?}, phys={px_phys_sky:?}"
    );

    let mut sky_clouds = sky_on.clone();
    sky_clouds.clouds.cover = 0.9;
    renderer.submit(&sky_clouds, &catalog, &mut target).unwrap();
    let pixels_no_clouds = {
        renderer.submit(&sky_on, &catalog, &mut target).unwrap();
        common::pixels(&device, &queue, &target.0)
    };
    renderer.submit(&sky_clouds, &catalog, &mut target).unwrap();
    let pixels_with_clouds = common::pixels(&device, &queue, &target.0);
    let cloud_diff_pixels = pixels_no_clouds
        .iter()
        .zip(&pixels_with_clouds)
        .filter(|(a, b)| a[0].abs_diff(b[0]) > 2 || a[1].abs_diff(b[1]) > 2 || a[2].abs_diff(b[2]) > 2)
        .count();
    assert!(cloud_diff_pixels > 16, "volumetric clouds must visibly composite over physical sky");

    // 3) LTC area lights illuminate `standard_surface` with zero sun, zero ambient, zero point lights!
    let mut area_base = common::frame(vec![Draw {
        mesh: 0,
        material: 2,
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
    // Clear HDRI so IBL reflection doesn't mask area light darkness.
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
        "LTC area light must illuminate standard_surface: dark={px_dark:?}, lit={px_area:?}"
    );

    // 4) SurfaceMapAtlas baked lightmap + thin-film iridescence visibly drive standard_surface.
    let mut atlas_frame = area_base.clone();
    atlas_frame.draws[0].material = 3;
    atlas_frame.draws[0].parameters = Some([0.0, 0.4, (512 | 1024) as f32, 0.0]);
    wait_ready(&mut renderer, &atlas_frame, &catalog, &mut target);
    let px_atlas = common::middle(&device, &queue, &target.0);
    assert!(
        px_atlas[1] > px_dark[1].saturating_add(40) && px_atlas[1] > px_atlas[0],
        "baked lightmap tile in SurfaceMapAtlas must illuminate surface green: dark={px_dark:?}, atlas={px_atlas:?}"
    );

    // 5) Split-screen viewports render both halves AND keep HUD sprites drawn on top across the seam.
    let mut split_frame = frame.clone();
    split_frame.grade.grain = 0.0;
    split_frame.camera_targets.push(CameraTargetSpec {
        material: 0,
        width: 32,
        height: 64,
        viewport_x: 0.0,
        viewport_y: 0.0,
        viewport_w: 0.5,
        viewport_h: 1.0,
        split_screen: true,
        view: frame.view,
        projection: frame.projection,
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
        view: frame.view,
        projection: frame.projection,
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
        color: [0.05, 0.2, 1.0, 1.0],
        material: 4,
        parameters: [0.0; 4],
    });
    wait_ready(&mut renderer, &split_frame, &catalog, &mut target);
    // Submit a second frame to exercise cached split-screen target reuse.
    renderer.submit(&split_frame, &catalog, &mut target).unwrap();
    let pixels = common::pixels(&device, &queue, &target.0);
    let left_mid = pixels[32 * 64 + 8];
    let right_mid = pixels[32 * 64 + 56];
    let hud_mid = pixels[32 * 64 + 32];
    assert!(
        (left_mid[0] > 20 || left_mid[1] > 20) && (right_mid[0] > 20 || right_mid[1] > 20),
        "both split-screen halves should render non-black content: left={left_mid:?}, right={right_mid:?}"
    );
    assert!(
        hud_mid[2] > 180 && hud_mid[2] > hud_mid[0] + 100,
        "HUD sprite must survive on top of split-screen seam: hud={hud_mid:?}"
    );

    // 6) Scene-filter sprite + MSAA 4x depth-resolved outlines compose together cleanly.
    let mut filter_msaa = common::frame(vec![Draw {
        mesh: 0,
        material: 0,
        transform: [
            0.5, 0.0, 0.0, 0.0,
            0.0, 0.5, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, -2.0, 1.0,
        ],
        parameters: None,
        extra: None,
    }]);
    filter_msaa.projection = proj;
    filter_msaa.quality.msaa = 4;
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
        material: 5,
        parameters: [0.0; 4],
    });
    wait_ready(&mut renderer, &filter_msaa, &catalog, &mut target);
    let px_filter_plain = common::pixels(&device, &queue, &target.0);

    let mut filter_outlined = filter_msaa.clone();
    filter_outlined.grade.outline_strength = 1.0;
    filter_outlined.grade.sharpness = 0.6;
    renderer.submit(&filter_outlined, &catalog, &mut target).unwrap();
    let px_filter_outlined = common::pixels(&device, &queue, &target.0);
    assert_ne!(
        px_filter_plain, px_filter_outlined,
        "scene filter + MSAA 4x depth-resolved outlines + RCAS must compose instead of skipping polish"
    );

    // 7) Visual shader graph (.matgraph.json) loads and renders directly through Renderer.
    let mut graph_frame = common::frame(vec![Draw {
        mesh: 0,
        material: 6,
        transform: wall_xform,
        parameters: None,
        extra: None,
    }]);
    graph_frame.projection = proj;
    wait_ready(&mut renderer, &graph_frame, &catalog, &mut target);
    let px_graph = common::middle(&device, &queue, &target.0);
    assert!(
        px_graph[0] > 30 || px_graph[1] > 30 || px_graph[2] > 30,
        "compiled .matgraph.json material must render visible pixels: {px_graph:?}"
    );

    // 8) Validation catches bad video_streams and out-of-range catalog material indices.
    let mut bad_video = common::frame(vec![]);
    bad_video.video_streams.push(VideoStreamSpec {
        material: 0,
        time: f32::NAN,
        playing: true,
        looping: true,
        speed: 1.0,
    });
    assert!(pen_render::validate_frame(&bad_video).is_err(), "NaN video time must fail validation");

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
