#![cfg(not(target_arch = "wasm32"))]
mod common;
use common::*;
use pen_protocol::*;
use pen_render::{Display, Renderer, RendererOptions};

const FILTER: &str = r#"
@group(0) @binding(0) var<uniform> screen: vec4f;
struct Sprite { rect: vec4f, uv: vec4f, color: vec4f, parameters: vec4f };
@group(0) @binding(1) var<storage,read> sprites: array<Sprite>;
@group(0) @binding(4) var scene: texture_2d<f32>;
struct Output { @builtin(position) position: vec4f, @location(0) color: vec4f };
@vertex fn vertex(@builtin(vertex_index) v:u32, @builtin(instance_index) i:u32)->Output {
 let corners=array<vec2f,6>(vec2f(0,0),vec2f(1,0),vec2f(0,1),vec2f(0,1),vec2f(1,0),vec2f(1,1));
 let sprite=sprites[i]; let at=sprite.rect.xy+corners[v]*sprite.rect.zw;
 return Output(vec4f(at.x/screen.x*2-1,1-at.y/screen.y*2,0,1),sprite.color);
}
@fragment fn fragment(v:Output)->@location(0) vec4f {
 if(v.color.r>0.9 && v.color.g==0.) { return v.color; }
 let size=vec2i(textureDimensions(scene));
 let p=clamp(vec2i(v.position.xy/screen.xy*vec2f(size)),vec2i(0),size-vec2i(1));
 return vec4f(textureLoad(scene,p,0).rgb*v.color.rgb,v.color.a);
}
"#;
const WORLD: &str = r#"
@vertex fn vertex(@location(0) position:vec3f)->@builtin(position) vec4f { return vec4f(position,1); }
@fragment fn fragment()->@location(0) vec4f { return vec4f(3,1.2,0.2,1); }
"#;

fn sprite(material: u32, size: f32, color: [f32; 4]) -> Sprite {
    Sprite {
        x: 0.,
        y: 0.,
        width: size,
        height: size,
        u0: 0.,
        v0: 0.,
        u1: 1.,
        v1: 1.,
        color,
        material,
        parameters: [0.; 4],
    }
}
fn error(a: &[[u8; 4]], b: &[[u8; 4]]) -> f64 {
    let sum: u64 =
        a.iter().zip(b).flat_map(|(a, b)| a[..3].iter().zip(&b[..3])).map(|(a, b)| u64::from(a.abs_diff(*b))).sum();
    sum as f64 / (a.len() * 3) as f64
}

#[test]
fn filters_preserve_scene_effects_chain_before_ui_and_resize() {
    let (device, queue) = device().expect("scene filter regression needs a GPU adapter");
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("scene-filters-test");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("filter.wgsl"), FILTER).unwrap();
    std::fs::write(root.join("world.wgsl"), WORLD).unwrap();
    std::fs::write(root.join("shape.mesh.json"),r#"{"positions":[-0.6,-0.6,0.5,0.6,-0.6,0.5,0,0.6,0.5],"normals":[0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,0.5,1],"indices":[0,1,2]}"#).unwrap();
    std::fs::write(
        root.join("world.material.json"),
        r#"{"shader":"world.wgsl","aberration":true,"lighting":"unlit","color":[1,1,1,1]}"#,
    )
    .unwrap();
    std::fs::write(
        root.join("filter.material.json"),
        r#"{"shader":"filter.wgsl","scene_filter":true,"lighting":"unlit","color":[1,1,1,1]}"#,
    )
    .unwrap();
    std::fs::write(root.join("ui.material.json"), r#"{"shader":"filter.wgsl","lighting":"unlit","color":[1,1,1,1]}"#)
        .unwrap();
    let catalog = Catalog {
        sounds: vec![],
        meshes: vec!["shape.mesh.json".into()],
        materials: vec!["world.material.json".into(), "filter.material.json".into(), "ui.material.json".into()],
    };
    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions { format: FORMAT, display: Display::SDR, assets: root, veil: None, pipeline_cache: None },
    )
    .unwrap();
    let base = frame(vec![Draw { mesh: 0, material: 0, transform: IDENTITY, parameters: None, extra: None }]);
    let mut target = Offscreen::new(&device, 64);
    let mut warm = base.clone();
    warm.sprites = vec![sprite(1, 64., [1.; 4]), sprite(2, 8., [1., 0., 0., 1.])];
    pump(&mut renderer, &warm, &catalog, &mut target, |events| {
        ready(events, "world.wgsl", 1) && ready(events, "filter.wgsl", 1)
    });
    for size in [64, 96, 64] {
        target = Offscreen::new(&device, size);
        let mut baseline = Vec::new();
        for effect in 0..8 {
            let mut picture = base.clone();
            picture.quality.effects = true;
            picture.quality.scale = 0.75;
            match effect {
                1 => picture.aberration.x = 0.08,
                2 => picture.blur = 0.4,
                3 => picture.quality.bloom = true,
                4 => {
                    picture.fog.density = 0.7;
                    picture.fog.radius = 0.1;
                    picture.fog.height = 2.;
                }
                5 => picture.quality.fxaa = true,
                6 => {
                    picture.quality.sky = true;
                    picture.sky.zenith = Vector { x: 0.2, y: 0.4, z: 0.7 };
                    picture.sky.horizon = Vector { x: 0.4, y: 0.5, z: 0.7 };
                }
                7 => {
                    picture.quality.effects = false;
                    picture.aberration.x = 0.08;
                    picture.blur = 0.4;
                }
                _ => {}
            }
            renderer.submit(&picture, &catalog, &mut target).unwrap();
            let expected = pixels(&device, &queue, &target.0);
            if effect == 0 {
                baseline = expected.clone();
            }
            if effect == 1 || effect == 2 || effect == 3 || effect == 4 || effect == 6 {
                assert!(error(&baseline, &expected) > 0.05, "effect {effect} fixture must visibly change the scene");
            }
            picture.sprites = vec![sprite(1, size as f32, [1.; 4])];
            renderer.submit(&picture, &catalog, &mut target).unwrap();
            let filtered = pixels(&device, &queue, &target.0);
            // A filter adds a resample of already-toned pixels. HDR edges
            // may differ slightly from interpolating HDR before tone mapping.
            assert!(
                error(&expected, &filtered) < 2.0,
                "identity filter lost effect {effect} at {size}: {}",
                error(&expected, &filtered)
            );
            if effect == 7 {
                assert!(error(&baseline, &filtered) < 0.8, "effects-disabled frame must stay disabled");
            }
        }
        // Two adjacent passes sharing a pipeline must compose, while the UI
        // sharing that same shader/texture must remain a separate batch.
        let mut picture = frame(vec![]);
        picture.clear = Color { r: 0.8, g: 0.8, b: 0.8, a: 1. };
        picture.sprites = vec![
            sprite(1, size as f32, [0.5, 0.5, 0.5, 1.]),
            sprite(1, size as f32, [0.5, 0.5, 0.5, 1.]),
            sprite(2, 8., [1., 0., 0., 1.]),
        ];
        renderer.submit(&picture, &catalog, &mut target).unwrap();
        let shown = pixels(&device, &queue, &target.0);
        let center = shown[(size * size / 2 + size / 2) as usize];
        assert!(center[0].abs_diff(51) < 3 && center[1].abs_diff(51) < 3, "filters must chain: {center:?}");
        assert_eq!(shown[(4 * size + 4) as usize], [255, 0, 0, 255], "UI must remain outside filters");
        picture.sprites.clear();
        renderer.submit(&picture, &catalog, &mut target).unwrap();
        assert!(
            middle(&device, &queue, &target.0)[0].abs_diff(204) < 3,
            "disabling filters restores the original path"
        );
    }
}
