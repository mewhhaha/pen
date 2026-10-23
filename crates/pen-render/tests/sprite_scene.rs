#![cfg(not(target_arch = "wasm32"))]
mod common;
use common::*;
use pen_protocol::*;
use pen_render::{Display, Renderer, RendererOptions};

#[test]
fn sprite_scene_is_current_scaled_and_unaffected_by_other_ui() {
    let Some((device, queue)) = device() else { return };
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("sprite-scene-test");
    std::fs::create_dir_all(&root).unwrap();
    let shader = r#"
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
 if(v.color.r>.5) { return vec4f(1,0,0,1); }
 let dimensions=vec2i(textureDimensions(scene));
 let point=clamp(vec2i(v.position.xy/screen.xy*vec2f(dimensions)),vec2i(0),dimensions-vec2i(1));
 return vec4f(vec3f(1)-textureLoad(scene,point,0).rgb,1);
}
"#;
    std::fs::write(root.join("scene.wgsl"), shader).unwrap();
    std::fs::write(root.join("scene.material.json"), r#"{"shader":"scene.wgsl","lighting":"unlit","color":[1,1,1,1]}"#)
        .unwrap();
    let catalog = Catalog { sounds: vec![], meshes: vec![], materials: vec!["scene.material.json".into()] };
    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions { format: FORMAT, display: Display::SDR, assets: root, veil: None, pipeline_cache: None },
    )
    .unwrap();
    let mut target = Offscreen::new(&device, 64);
    let mut picture = frame(vec![]);
    let sprite = Sprite {
        x: 0.,
        y: 0.,
        width: 64.,
        height: 64.,
        u0: 0.,
        v0: 0.,
        u1: 1.,
        v1: 1.,
        color: [0., 0., 0., 1.],
        material: 0,
        parameters: [0.; 4],
    };
    let mut underneath = sprite;
    underneath.color = [1., 0., 0., 1.];
    picture.sprites = vec![underneath, sprite];
    pump(&mut renderer, &picture, &catalog, &mut target, |events| ready(events, "scene.wgsl", 1));
    for (size, scale, background, expected) in [(64, 1., 0., 255), (96, 0.5, 1., 0), (64, 0.75, 0., 255)] {
        target = Offscreen::new(&device, size);
        picture.quality.scale = scale;
        picture.clear = Color { r: background, g: background, b: background, a: 1. };
        for entry in &mut picture.sprites {
            entry.width = size as f32;
            entry.height = size as f32;
        }
        renderer.submit(&picture, &catalog, &mut target).unwrap();
        let actual = middle(&device, &queue, &target.0);
        for channel in &actual[..3] {
            assert!(i16::from(*channel).abs_diff(expected) <= 2, "{actual:?}");
        }
    }
}
