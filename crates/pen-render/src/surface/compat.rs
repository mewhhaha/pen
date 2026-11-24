//! The library's base lobe against what it replaced: the frozen
//! `surface_ambient` and `clustered_surface` of the parent commit
//! (baseline.wgsl) and the library's, on the real sky and lights of a mesh
//! pipeline, over grids of inputs, compared bit for bit. Where the two files'
//! own text is wanted whole (`PEN_BASELINE_SHADERS` names a directory holding
//! the parent's environment.wgsl and lights.wgsl), they are compared too.
use pen_protocol::Light;

use super::harness::{Gpu, World, compute, gpu, sky, vector};
use crate::{ENVIRONMENT_WGSL, LIGHTS_WGSL};

use super::{SURFACE_LOBES_WGSL, SURFACE_WGSL};

const BASELINE: &str = include_str!("baseline.wgsl");

fn library() -> String {
    format!("{ENVIRONMENT_WGSL}\n{LIGHTS_WGSL}\n{SURFACE_LOBES_WGSL}\n{SURFACE_WGSL}\n{BASELINE}")
}

/// A tiny deterministic generator, so that a grid's random part is the same every run.
struct Random(u64);

impl Random {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u32 << 24) as f32
    }
    fn unit(&mut self) -> [f32; 3] {
        loop {
            let v = [self.next() * 2.0 - 1.0, self.next() * 2.0 - 1.0, self.next() * 2.0 - 1.0];
            let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if length > 0.1 && length <= 1.0 {
                return [v[0] / length, v[1] / length, v[2] / length];
            }
        }
    }
}

/// Surfaces of every roughness and metallic on a grid, with random views, normals
/// and albedos: world (3), normal (3), view (3), albedo (3), metallic, roughness.
fn surfaces() -> Vec<[f32; 14]> {
    let mut random = Random(7);
    let mut cases = Vec::new();
    for roughness in [0.0, 0.03, 0.04, 0.12, 0.3, 0.5, 0.75, 0.95, 1.0, 1.3] {
        for metallic in [0.0, 0.1, 0.5, 0.9, 1.0] {
            for _ in 0..24 {
                let normal = random.unit();
                let mut view = random.unit();
                // Mostly facing the eye, some not.
                let facing = normal[0] * view[0] + normal[1] * view[1] + normal[2] * view[2];
                if facing < 0.0 && random.next() < 0.8 {
                    view = view.map(|c| -c);
                }
                let world = [random.next() * 6.0 - 3.0, random.next() * 3.0 - 0.5, random.next() * 6.0 - 3.0];
                cases.push([
                    world[0],
                    world[1],
                    world[2],
                    normal[0],
                    normal[1],
                    normal[2],
                    view[0],
                    view[1],
                    view[2],
                    random.next(),
                    random.next(),
                    random.next(),
                    metallic,
                    roughness,
                ]);
            }
        }
    }
    cases
}

fn lights() -> Vec<Light> {
    let point = |x, y, z, radius, color: [f32; 3]| Light {
        position: vector(x, y, z),
        radius,
        color: vector(color[0], color[1], color[2]),
        direction: vector(0.0, -1.0, 0.0),
        inner: -1.0,
        outer: -1.0,
        shadow: false,
        id: 0,
    };
    let mut spot = point(-2.0, 3.0, 1.0, 9.0, [3.0, 2.5, 1.5]);
    (spot.inner, spot.outer) = (0.95, 0.7);
    spot.direction = vector(0.4, -0.9, -0.2);
    vec![
        point(0.0, 1.5, 0.0, 7.0, [4.0, 3.0, 2.0]),
        point(2.0, 0.4, -1.5, 5.0, [0.5, 1.5, 3.0]),
        spot,
        point(40.0, 1.0, 0.0, 3.0, [9.0, 9.0, 9.0]),
    ]
}

fn bits(floats: &[f32]) -> Vec<u32> {
    floats.iter().map(|f| f.to_bits()).collect()
}

fn assert_same(what: &str, cases: &[[f32; 14]], width: usize, out: &[f32]) {
    // Each row is the old result then the library's, `width` floats each.
    let mut different = 0;
    for (index, row) in out.chunks(2 * width).enumerate() {
        let (old, new) = row.split_at(width);
        if bits(old) != bits(new) {
            different += 1;
            if different <= 3 {
                eprintln!("{what}: case {index} {:?}: old {old:?}, new {new:?}", cases[index]);
            }
        }
    }
    assert_eq!(different, 0, "{what}: {different} of {} cases differ in bits", out.len() / (2 * width));
}

fn worlds(gpu: &Gpu) -> Vec<(&'static str, World)> {
    let lit = sky(vector(0.2, 0.4, 1.0), vector(0.9, 0.8, 0.7), vector(0.15, 0.12, 0.1), vector(0.5, 0.6, 0.8));
    // A frame without a sky: no reflection to take light from the diffuse.
    let mut dark = lit;
    dark.tint = vector(0.0, 0.0, 0.0);
    dark.sh0 = vector(0.0, 0.0, 0.0);
    dark.sh1 = vector(0.0, 0.0, 0.0);
    dark.sh3 = vector(0.0, 0.0, 0.0);
    vec![("a sky", World::new(gpu, &lit, &lights())), ("no sky", World::new(gpu, &dark, &lights()))]
}

#[test]
fn the_base_lobes_sky_light_is_the_old_surface_ambient_to_the_bit() {
    let Some(gpu) = gpu() else { return };
    let cases = surfaces();
    let input: Vec<f32> = cases.iter().flatten().copied().collect();
    let kernel = "
fn kernel(i: u32) {
  let world = arg3(i, 0u); let normal = arg3(i, 3u); let view = arg3(i, 6u); let albedo = arg3(i, 9u);
  let metallic = arg(i, 12u); let roughness = arg(i, 13u);
  put3(i, 0u, old_surface_ambient(world, normal, view, albedo, metallic, roughness));
  put3(i, 3u, surface_ambient(world, normal, view, albedo, metallic, roughness));
}";
    for (name, world) in worlds(&gpu) {
        let out = compute(&gpu, &library(), kernel, &input, 14, 6, Some(&world), None);
        assert_same(&format!("surface_ambient under {name}"), &cases, 3, &out);
        // The light is there to be compared: not every result is zero.
        assert!(name == "no sky" || out.chunks(6).any(|r| r[0] > 0.1), "the sky lights nothing: {:?}", &out[..6]);
    }
}

#[test]
fn the_base_lobes_light_from_the_frames_lights_is_the_old_clustered_surface_to_the_bit() {
    let Some(gpu) = gpu() else { return };
    let cases = surfaces();
    // The cluster's lights, the same for every fragment.
    let input: Vec<f32> = cases.iter().flatten().copied().collect();
    let kernel = "
fn kernel(i: u32) {
  let world = arg3(i, 0u); let normal = arg3(i, 3u); let view = arg3(i, 6u);
  let f0 = mix(vec3f(0.04), arg3(i, 9u), arg(i, 12u));
  let roughness = arg(i, 13u);
  let position = vec4f(100.0, 100.0, 0.5, 1.0);
  let old = old_clustered_surface(world, normal, view, roughness, f0, position, -5.0);
  let fresh = clustered_surface(world, normal, view, roughness, f0, position, -5.0);
  put3(i, 0u, old.diffuse); put3(i, 3u, old.specular);
  put3(i, 6u, fresh.diffuse); put3(i, 9u, fresh.specular);
}";
    let (_, world) = worlds(&gpu).remove(0);
    let out = compute(&gpu, &library(), kernel, &input, 14, 12, Some(&world), None);
    assert_same("clustered_surface", &cases, 6, &out);
    let lit = out.chunks(12).filter(|r| r[0] > 0.0 && r[3] > 0.0).count();
    assert!(lit > cases.len() / 4, "the lights reach {lit} of {} cases", cases.len());
}

#[test]
fn standard_surface_with_the_base_lobe_is_how_shaders_combine_the_old_halves() {
    let Some(gpu) = gpu() else { return };
    let cases = surfaces();
    let input: Vec<f32> = cases.iter().flatten().copied().collect();
    let kernel = "
fn kernel(i: u32) {
  let world = arg3(i, 0u); let normal = arg3(i, 3u); let view = arg3(i, 6u); let albedo = arg3(i, 9u);
  let metallic = arg(i, 12u); let roughness = arg(i, 13u);
  let f0 = mix(vec3f(0.04), albedo, metallic);
  let position = vec4f(100.0, 100.0, 0.5, 1.0);
  // How a game's shader has lit the albedo: the sky, then the lights' two parts.
  var old = old_surface_ambient(world, normal, view, albedo, metallic, roughness);
  let local = old_clustered_surface(world, normal, view, roughness, f0, position, -5.0);
  old += local.diffuse * albedo * (1.0 - clamp(metallic, 0.0, 1.0));
  old += local.specular;
  let s = surface_default(world, normal, view, albedo, metallic, roughness, position, -5.0);
  put3(i, 0u, old); put3(i, 3u, standard_surface(s));
}";
    let (_, world) = worlds(&gpu).remove(0);
    let out = compute(&gpu, &library(), kernel, &input, 14, 6, Some(&world), None);
    // The two halves are each the same to the bit; their sum is the shader's
    // own two additions, which a compiler may contract differently in two places
    // (a final fused multiply-add), so this one is held to two units of the last place.
    let mut bit_differences = 0;
    for (index, row) in out.chunks(6).enumerate() {
        let (old, new) = row.split_at(3);
        bit_differences += usize::from(bits(old) != bits(new));
        for c in 0..3 {
            let (a, b) = (f64::from(old[c]), f64::from(new[c]));
            assert!(
                (a - b).abs() <= 2.5e-7 * a.abs().max(1e-3),
                "case {index} {:?}: old {old:?}, new {new:?}",
                cases[index]
            );
        }
    }
    eprintln!(
        "standard_surface against the old halves summed: {bit_differences} of {} cases differ in the last bit",
        cases.len()
    );
}

/// The parent's whole environment.wgsl and lights.wgsl against the library's, if
/// they are at hand: `PEN_BASELINE_SHADERS=dir cargo test ...`, where `dir` holds
/// `git show 8d5884d:crates/pen-render/src/shaders/{environment,lights}.wgsl`.
#[test]
fn the_parents_whole_files_give_the_same_bits() {
    let Some(directory) = std::env::var_os("PEN_BASELINE_SHADERS") else { return };
    let Some(gpu) = gpu() else { return };
    let read = |name: &str| std::fs::read_to_string(std::path::Path::new(&directory).join(name)).unwrap();
    let (old_environment, old_lights) = (read("environment.wgsl"), read("lights.wgsl"));
    let cases = surfaces();
    let input: Vec<f32> = cases.iter().flatten().copied().collect();
    let kernel = |call: &str| {
        format!(
            "
fn kernel(i: u32) {{
  let world = arg3(i, 0u); let normal = arg3(i, 3u); let view = arg3(i, 6u); let albedo = arg3(i, 9u);
  let metallic = arg(i, 12u); let roughness = arg(i, 13u);
  let f0 = mix(vec3f(0.04), albedo, metallic);
  let position = vec4f(100.0, 100.0, 0.5, 1.0);
  var light = surface_ambient(world, normal, view, albedo, metallic, roughness);
  let local = clustered_surface(world, normal, view, roughness, f0, position, -5.0);
  {call}
  put3(i, 0u, light); put3(i, 3u, local.specular);
}}"
        )
    };
    let (_, world) = worlds(&gpu).remove(0);
    let old = compute(
        &gpu,
        &format!("{old_environment}\n{old_lights}"),
        &kernel("light += local.diffuse * albedo * (1.0 - metallic);"),
        &input,
        14,
        6,
        Some(&world),
        None,
    );
    let new = compute(
        &gpu,
        &library(),
        &kernel("light += local.diffuse * albedo * (1.0 - metallic);"),
        &input,
        14,
        6,
        Some(&world),
        None,
    );
    let different = old.chunks(6).zip(new.chunks(6)).filter(|(a, b)| bits(a) != bits(b)).count();
    assert_eq!(different, 0, "{different} of {} cases differ against the parent's own files", cases.len());
}
