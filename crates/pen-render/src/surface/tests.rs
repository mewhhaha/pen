//! The standard surface's lobes against independent f64 references, on a real
//! (headless) GPU. Every test says its tolerance; compute shaders run the pure
//! lobes of `surface_lobes.wgsl` on their own, and the real sky and lights are
//! bound where a test needs them.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::float_cmp,
    reason = "test geometry and the shaders' own f32 against f64"
)]
use std::f64::consts::PI;

use super::harness::{Gpu, Heights, compute, gpu};
use super::reference::{self, Surface, V};
use super::{SURFACE_LOBES_WGSL, SURFACE_TRANSMISSION_WGSL, SURFACE_WGSL};

const COAT: u32 = 1;
const SHEEN: u32 = 2;
const ANISOTROPY: u32 = 4;
const TRANSMISSION: u32 = 8;
const SUBSURFACE: u32 = 16;

/// A unit vector at `theta` from +z, turned `phi` about it.
fn dir(theta: f64, phi: f64) -> V {
    V(theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos())
}

/// One surface and one light: the floats `SURFACE_OF` reads, and the
/// reference's surface.
#[derive(Clone, Debug)]
struct Case {
    surface: Surface,
    l: V,
    lobes: u32,
}

impl Default for Case {
    fn default() -> Self {
        Case {
            surface: Surface {
                n: V(0.0, 0.0, 1.0),
                v: dir(0.5, 0.3),
                roughness: 0.4,
                metallic: 0.0,
                base: V(0.8, 0.6, 0.4),
                f0: V::all(0.04),
                f90: 1.0,
                coat: 0.0,
                coat_roughness: 0.2,
                coat_normal: V(0.0, 0.0, 1.0),
                sheen: V::all(0.0),
                sheen_roughness: 0.5,
                anisotropy: 0.0,
                tangent: V(1.0, 0.0, 0.0),
                bitangent: V(0.0, 1.0, 0.0),
                subsurface: 0.0,
                subsurface_color: V(1.0, 0.5, 0.3),
                thin: 0.0,
                transmission: 0.0,
            },
            l: dir(0.6, 2.0),
            lobes: 0,
        }
    }
}

const STRIDE: usize = 48;

impl Case {
    fn floats(&self) -> [f32; STRIDE] {
        let s = &self.surface;
        let mut out = [0.0f32; STRIDE];
        let mut put = |at: usize, values: &[f64]| {
            for (k, value) in values.iter().enumerate() {
                out[at + k] = *value as f32;
            }
        };
        put(0, &s.n.array());
        put(3, &s.v.array());
        put(6, &self.l.array());
        put(9, &[s.roughness, s.metallic]);
        put(11, &s.base.array());
        put(14, &s.f0.array());
        put(17, &[s.f90, f64::from(self.lobes), s.coat, s.coat_roughness]);
        put(21, &s.coat_normal.array());
        put(24, &s.sheen.array());
        put(27, &[s.sheen_roughness, s.anisotropy]);
        put(29, &s.tangent.array());
        put(32, &s.bitangent.array());
        put(35, &[s.subsurface]);
        put(36, &s.subsurface_color.array());
        put(39, &[s.thin, s.transmission, 1.0]);
        out
    }
}

/// WGSL for `kernel`s that build the surface a `Case` describes.
const CASES: &str = "
fn surface_of(i: u32) -> StandardSurface {
  var s = surface_default(vec3f(0.0), arg3(i, 0u), arg3(i, 3u), arg3(i, 11u), arg(i, 10u), arg(i, 9u), vec4f(0.0), 0.0);
  s.f0 = arg3(i, 14u);
  s.f90 = arg(i, 17u);
  s.lobes = u32(arg(i, 18u));
  s.coat = arg(i, 19u);
  s.coat_roughness = arg(i, 20u);
  s.coat_normal = arg3(i, 21u);
  s.sheen_color = arg3(i, 24u);
  s.sheen_roughness = arg(i, 27u);
  s.anisotropy = arg(i, 28u);
  s.tangent = arg3(i, 29u);
  s.bitangent = arg3(i, 32u);
  s.subsurface = arg(i, 35u);
  s.subsurface_color = arg3(i, 36u);
  s.thin = arg(i, 39u);
  s.transmission = arg(i, 40u);
  return s;
}
";

/// What `surface_light` returns for each case (diffuse then specular, three
/// floats each) with a radiance of 1.
fn lights(gpu: &Gpu, cases: &[Case]) -> Vec<[f32; 6]> {
    let input: Vec<f32> = cases.iter().flat_map(Case::floats).collect();
    let kernel = format!(
        "{CASES}
fn kernel(i: u32) {{
  let r = surface_light(surface_of(i), normalize(arg3(i, 6u)), vec3f(arg(i, 41u)));
  put3(i, 0u, r.diffuse);
  put3(i, 3u, r.specular);
}}"
    );
    let out = compute(gpu, SURFACE_LOBES_WGSL, &kernel, &input, STRIDE, 6, None, None);
    out.chunks(6).map(|c| c.try_into().unwrap()).collect()
}

/// The table's text between its markers, as numbers.
fn table_in_shader() -> Vec<f64> {
    let begin = SURFACE_LOBES_WGSL.find("// SHEEN TABLE BEGIN").expect("table begin") + "// SHEEN TABLE BEGIN".len();
    let end = SURFACE_LOBES_WGSL.find("// SHEEN TABLE END").expect("table end");
    SURFACE_LOBES_WGSL[begin..end]
        .split(',')
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(|text| text.parse().expect("a number"))
        .collect()
}

/// The shader's table, written out for pasting between its markers:
/// `cargo test -p pen-render --release print_the_sheen_table -- --ignored --nocapture`.
#[test]
#[ignore = "prints the table the shader carries"]
fn print_the_sheen_table() {
    let table = reference::sheen_table();
    for row in table.chunks(8) {
        let line: Vec<String> = row.iter().map(|value| format!("{value:.8}")).collect();
        println!("  {},", line.join(", "));
    }
}

#[test]
fn the_sheen_table_in_the_shader_is_the_integral() {
    let in_shader = table_in_shader();
    assert_eq!(in_shader.len(), 512, "32 cosines by 16 roughnesses");
    let table = reference::sheen_table();
    for (index, (shader, want)) in in_shader.iter().zip(&table).enumerate() {
        assert!(
            (shader - want).abs() < 2e-8,
            "entry {index} (roughness row {}, cosine column {}): shader {shader} vs integral {want}",
            index / 32,
            index % 32
        );
    }
    // Ink's table, which this is: its entries sum to 220.61942091.
    let sum: f64 = in_shader.iter().sum();
    assert!((sum - 220.619_420_91).abs() < 1e-6, "the table's checksum is {sum}");
}

/// The relative difference of `got` from `want`, with `floor` keeping a
/// near-zero `want` from dividing.
fn error(got: f64, want: f64, floor: f64) -> f64 {
    (got - want).abs() / want.abs().max(floor)
}

/// The cases every lobe is tried on: surfaces of a few roughnesses viewed from
/// a few angles and lit from many (the lights' and the view's azimuths differ
/// so that anisotropy and the frame matter).
fn grid(configure: impl Fn(&mut Surface)) -> Vec<Case> {
    let mut cases = Vec::new();
    for roughness in [0.12, 0.35, 0.8] {
        for view in [0.2, 0.9, 1.35] {
            for (light, phi) in [(0.1, 0.0), (0.5, 1.0), (0.9, 2.5), (1.2, 4.0), (1.45, 5.5), (1.7, 0.7), (2.0, 3.0)] {
                let mut case = Case::default();
                case.surface.roughness = roughness;
                case.surface.v = dir(view, 0.7);
                case.l = dir(light, phi);
                configure(&mut case.surface);
                cases.push(case);
            }
        }
    }
    cases
}

/// Runs `cases` with `lobes` set and checks the shader's responses against the
/// reference's: `tolerance` of the larger of the two (floor `1e-3` of a
/// response, so that a lobe's far tail does not fail for a rounding).
fn check_against_reference(gpu: &Gpu, name: &str, cases: &mut [Case], lobes: u32, tolerance: f64) -> f64 {
    let table = reference::sheen_table();
    for case in cases.iter_mut() {
        case.lobes = lobes;
    }
    let got = lights(gpu, cases);
    let mut worst = 0.0f64;
    for (case, got) in cases.iter().zip(&got) {
        let want = reference::light(&case.surface, case.l, lobes, &table);
        for c in 0..3 {
            for (kind, g, w) in [
                ("diffuse", f64::from(got[c]), want.diffuse.array()[c] * PI),
                ("specular", f64::from(got[3 + c]), want.specular.array()[c] * PI),
            ] {
                let e = error(g, w, 1e-3);
                worst = worst.max(e);
                assert!(
                    e < tolerance,
                    "{name}: {kind} channel {c} is {g} where the f64 reference has {w} (relative {e}) for {case:?}"
                );
            }
        }
    }
    // The lobes are in play: switching them off would change the responses.
    let mut moved = 0.0f64;
    for case in cases.iter() {
        let with = reference::light(&case.surface, case.l, lobes, &table);
        let without = reference::light(&case.surface, case.l, 0, &table);
        moved = moved.max(error(with.specular.0 + with.diffuse.0, without.specular.0 + without.diffuse.0, 1e-2));
    }
    assert!(
        lobes == 0 || moved > 0.05,
        "{name}: the lobes change no response by more than {moved}; the test would not see them"
    );
    eprintln!("{name}: {} cases, the worst relative error against the f64 reference is {worst:.2e}", cases.len());
    worst
}

#[test]
fn the_base_lobe_matches_the_reference() {
    let Some(gpu) = gpu() else { return };
    let mut cases = grid(|s| {
        s.f0 = V(0.9, 0.7, 0.3);
        s.metallic = 1.0;
    });
    check_against_reference(&gpu, "base", &mut cases, 0, 2e-4);
}

#[test]
fn clear_coat_matches_the_reference() {
    let Some(gpu) = gpu() else { return };
    let mut cases = grid(|s| {
        s.coat = 0.8;
        s.coat_roughness = 0.15;
    });
    check_against_reference(&gpu, "coat", &mut cases, COAT, 3e-4);
    // A coat of its own normal.
    let mut tilted = grid(|s| {
        s.coat = 1.0;
        s.coat_roughness = 0.1;
        s.coat_normal = dir(0.3, 1.0);
    });
    check_against_reference(&gpu, "coat with its own normal", &mut tilted, COAT, 3e-4);
}

#[test]
fn sheen_matches_the_reference() {
    let Some(gpu) = gpu() else { return };
    let mut cases = grid(|s| {
        s.sheen = V(0.9, 0.6, 0.3);
        s.sheen_roughness = 0.45;
    });
    check_against_reference(&gpu, "sheen", &mut cases, SHEEN, 5e-4);
}

#[test]
fn anisotropy_matches_the_reference() {
    let Some(gpu) = gpu() else { return };
    for rotation in [0.0f64, 0.7, -1.9] {
        // A frame turned about the normal by `rotation`.
        let (t, b) = (dir(PI / 2.0, rotation), dir(PI / 2.0, rotation + PI / 2.0));
        let mut cases = grid(|s| {
            s.anisotropy = 0.8;
            s.tangent = t;
            s.bitangent = b;
        });
        check_against_reference(&gpu, &format!("anisotropy 0.8 turned {rotation}"), &mut cases, ANISOTROPY, 5e-4);
    }
    let mut negative = grid(|s| s.anisotropy = -0.6);
    check_against_reference(&gpu, "anisotropy -0.6", &mut negative, ANISOTROPY, 5e-4);
}

#[test]
fn subsurface_and_transmission_scale_the_diffuse_lobe() {
    let Some(gpu) = gpu() else { return };
    let mut cases = grid(|s| {
        s.subsurface = 0.7;
        s.thin = 0.2;
        s.transmission = 0.35;
    });
    check_against_reference(&gpu, "subsurface and transmission", &mut cases, SUBSURFACE | TRANSMISSION, 3e-4);
}

#[test]
fn every_lobe_at_once_matches_the_reference() {
    let Some(gpu) = gpu() else { return };
    let (t, b) = (dir(PI / 2.0, 0.4), dir(PI / 2.0, 0.4 + PI / 2.0));
    let mut cases = grid(|s| {
        s.coat = 0.6;
        s.coat_roughness = 0.2;
        s.coat_normal = dir(0.2, 2.0);
        s.sheen = V(0.5, 0.5, 0.8);
        s.sheen_roughness = 0.35;
        s.anisotropy = 0.5;
        s.tangent = t;
        s.bitangent = b;
        s.subsurface = 0.4;
        s.thin = 0.5;
        s.transmission = 0.2;
    });
    check_against_reference(&gpu, "all lobes", &mut cases, COAT | SHEEN | ANISOTROPY | SUBSURFACE | TRANSMISSION, 5e-4);
}

#[test]
fn lobes_of_weight_zero_are_the_base_lobe_to_the_bit() {
    let Some(gpu) = gpu() else { return };
    let mut plain = grid(|s| s.f0 = V(0.6, 0.5, 0.4));
    let mut zeroed = plain.clone();
    for case in &mut plain {
        case.lobes = 0;
    }
    for case in &mut zeroed {
        case.lobes = COAT | SHEEN | ANISOTROPY | TRANSMISSION | SUBSURFACE;
        // Every weight at zero, the other parameters left to be read.
        case.surface.coat = 0.0;
        case.surface.sheen = V::all(0.0);
        case.surface.anisotropy = 0.0;
        case.surface.subsurface = 0.0;
        case.surface.transmission = 0.0;
    }
    let (a, b) = (lights(&gpu, &plain), lights(&gpu, &zeroed));
    for (index, (a, b)) in a.iter().zip(&b).enumerate() {
        let (a, b): (Vec<u32>, Vec<u32>) =
            (a.iter().map(|v| v.to_bits()).collect(), b.iter().map(|v| v.to_bits()).collect());
        assert_eq!(a, b, "case {index}: lobes with no weight change the base lobe's bits");
    }
}

/// What a surface sends back from a white furnace (radiance 1 from every
/// direction), as `reference::furnace` computes it: the diffuse part by a
/// midpoint rule over the light's sphere, 96 by 192 cells; the specular lobes
/// by one over the half vector's hemisphere, 512 by 128 cells bunched toward
/// its poles. The diffuse and the specular albedos, three floats each.
fn furnace(gpu: &Gpu, cases: &[Case]) -> Vec<[f64; 6]> {
    let input: Vec<f32> = cases.iter().flat_map(Case::floats).collect();
    let kernel = format!(
        "{CASES}
fn kernel(i: u32) {{
  let s = surface_of(i);
  var diffuse = vec3f(0.0);
  var specular = vec3f(0.0);
  for (var a = 0u; a < 96u; a++) {{
    let t = (f32(a) + 0.5) / 96.0 * 3.14159265;
    let weight = sin(t) * (3.14159265 / 96.0) * (6.28318531 / 192.0);
    for (var b = 0u; b < 192u; b++) {{
      let p = (f32(b) + 0.5) / 192.0 * 6.28318531;
      let r = surface_light(s, vec3f(sin(t) * cos(p), sin(t) * sin(p), cos(t)), vec3f(1.0));
      diffuse += r.diffuse * weight;
    }}
  }}
  for (var a = 0u; a < 512u; a++) {{
    let u = (f32(a) + 0.5) / 512.0;
    let t = 3.14159265 / 4.0 * (1.0 - cos(3.14159265 * u));
    let weight = sin(t) * (3.14159265 / 4.0 * 3.14159265 * sin(3.14159265 * u) / 512.0) * (6.28318531 / 128.0);
    for (var b = 0u; b < 128u; b++) {{
      let p = (f32(b) + 0.5) / 128.0 * 6.28318531;
      let h = vec3f(sin(t) * cos(p), sin(t) * sin(p), cos(t));
      let vh = dot(s.view, h);
      if (vh > 0.0) {{
        let r = surface_light(s, h * (2.0 * vh) - s.view, vec3f(1.0));
        specular += r.specular * (weight * 4.0 * vh);
      }}
    }}
  }}
  // The lights' scale carries a factor of pi.
  put3(i, 0u, diffuse * s.base_color * (1.0 - s.metallic) / 3.14159265);
  put3(i, 3u, specular / 3.14159265);
}}"
    );
    let out = compute(gpu, SURFACE_LOBES_WGSL, &kernel, &input, STRIDE, 6, None, None);
    out.chunks(6).map(|c| std::array::from_fn(|k| f64::from(c[k]))).collect()
}

fn total(albedo: &[f64; 6]) -> f64 {
    albedo[0]
        .max(albedo[1])
        .max(albedo[2])
        .max(albedo[3] + albedo[0])
        .max(albedo[4] + albedo[1])
        .max(albedo[5] + albedo[2])
}

#[test]
fn a_coat_creates_no_energy() {
    let Some(gpu) = gpu() else { return };
    let table = reference::sheen_table();
    // A white metal (f0 = 1: the base lobe returns at most what it gets) and
    // a white dielectric, under a coat of every weight and roughness, seen from
    // straight on to a cosine of 0.02.
    let mut cases = Vec::new();
    for metal in [true, false] {
        // (Below a roughness of about 0.08 the half vector's own rounding in f32,
        // which the base lobe has always had, moves the narrow peak's integral by
        // percents: the f64 reference below takes the roughest and smoothest.)
        for roughness in [0.08, 0.15, 0.4, 0.9, 1.0] {
            for cosine in [1.0f64, 0.7, 0.3, 0.1, 0.05, 0.02] {
                for (coat, coat_roughness) in [(0.0, 0.1), (0.5, 0.1), (1.0, 0.08), (1.0, 0.15), (1.0, 0.5), (1.0, 1.0)]
                {
                    let mut case = Case::default();
                    case.surface.v = dir(cosine.acos(), 0.0);
                    case.surface.roughness = roughness;
                    case.surface.base = V::all(1.0);
                    case.surface.f0 = if metal { V::all(1.0) } else { V::all(0.04) };
                    case.surface.metallic = if metal { 1.0 } else { 0.0 };
                    case.surface.coat = coat;
                    case.surface.coat_roughness = coat_roughness;
                    case.lobes = COAT;
                    cases.push(case);
                }
            }
        }
    }
    let coated = furnace(&gpu, &cases);
    let mut uncoated_cases = cases.clone();
    let mut coat_alone = cases.clone();
    for case in &mut uncoated_cases {
        case.lobes = 0;
    }
    for case in &mut coat_alone {
        // No base lobes: the coat's own albedo.
        case.surface.base = V::all(0.0);
        case.surface.f0 = V::all(0.0);
        case.surface.f90 = 0.0;
    }
    let (uncoated, alone) = (furnace(&gpu, &uncoated_cases), furnace(&gpu, &coat_alone));
    let (mut worst_metal, mut worst_alone, mut worst_excess) = (0.0f64, 0.0f64, f64::MIN);
    for (index, case) in cases.iter().enumerate() {
        let (c, u, a) = (total(&coated[index]), total(&uncoated[index]), total(&alone[index]));
        let metal = case.surface.metallic > 0.5;
        // Never more than the layers beneath plus the coat's own reflection, nor
        // (the coat only moves light about) than the larger of 1 and the layers beneath.
        assert!(
            c <= u + a + 3e-3,
            "case {index}: coated {c} exceeds uncoated {u} plus the coat alone {a} for {case:?}"
        );
        assert!(c <= 1.0f64.max(u) + 1e-3, "case {index}: coated {c} exceeds max(1, uncoated {u}) for {case:?}");
        worst_alone = worst_alone.max(a);
        worst_excess = worst_excess.max(c - 1.0f64.max(u));
        if metal {
            worst_metal = worst_metal.max(c);
        }
        // The reference's furnace agrees with the shader's (a case in seven, for the CPU's time).
        if index % 7 == 0 {
            let want = reference::furnace(&case.surface, COAT, &table, (96, 192), (512, 128));
            let want = [want.0.0, want.0.1, want.0.2, want.1.0, want.1.1, want.1.2];
            let want_total = total(&want);
            assert!(
                (c - want_total).abs() < 4e-3 * want_total.max(1.0),
                "case {index}: furnace {c} vs the reference's {want_total} for {case:?}"
            );
        }
    }
    // The same in f64 at the smoothest roughness there is, 0.04, grazing included.
    let mut worst_smooth = 0.0f64;
    for cosine in [1.0f64, 0.3, 0.1, 0.02] {
        for roughness in [0.04, 1.0] {
            for coat_roughness in [0.04, 0.3, 1.0] {
                let mut case = Case::default();
                case.surface.v = dir(cosine.acos(), 0.0);
                case.surface.roughness = roughness;
                case.surface.base = V::all(1.0);
                case.surface.f0 = V::all(1.0);
                case.surface.metallic = 1.0;
                case.surface.coat = 1.0;
                case.surface.coat_roughness = coat_roughness;
                let albedo = reference::furnace(&case.surface, COAT, &table, (8, 8), (2048, 128));
                let returned = albedo.1.0;
                worst_smooth = worst_smooth.max(returned);
                assert!(
                    returned <= 1.0 + 1e-3,
                    "f64: view cos {cosine}, roughness {roughness}, coat roughness {coat_roughness}: {returned}"
                );
            }
        }
    }
    eprintln!(
        "coat furnace: {} cases down to a view cosine of 0.02, a coated white metal returns at most {worst_metal:.4}, the coat alone at most {worst_alone:.4}, coated minus max(1, uncoated) at most {worst_excess:.4}; in f64 down to roughness 0.04 at most {worst_smooth:.4}",
        cases.len()
    );
}

#[test]
fn sheen_returns_what_it_takes_from_the_base() {
    let Some(gpu) = gpu() else { return };
    // A white diffuse base without a highlight under a sheen of 1: the base
    // keeps 1 - Es of its light and the sheen adds its own albedo, so the
    // furnace gives back what it was given, up to the table's interpolation
    // error, or where the sharp lobe's own albedo passes 1 (a grazing view
    // of a sheen roughness under 0.45, which the table caps), that albedo.
    let mut cases = Vec::new();
    for sheen_roughness in [0.07, 0.15, 0.3, 0.5, 0.8, 1.0] {
        for view in [0.0, 0.5, 0.9, 1.2, 1.4, 1.55] {
            let mut case = Case::default();
            case.surface.v = dir(view, 0.0);
            case.surface.base = V::all(1.0);
            case.surface.f0 = V::all(0.0);
            case.surface.f90 = 0.0;
            case.surface.sheen = V::all(1.0);
            case.surface.sheen_roughness = sheen_roughness;
            case.lobes = SHEEN;
            cases.push(case);
        }
    }
    let got = furnace(&gpu, &cases);
    let table = reference::sheen_table();
    let mut worst = 0.0f64;
    for (case, albedo) in cases.iter().zip(&got) {
        let returned = total(albedo);
        let es = reference::sheen_albedo(case.surface.v.2, case.surface.sheen_roughness);
        let want = reference::furnace(&case.surface, SHEEN, &table, (96, 192), (512, 128));
        let want = total(&[want.0.0, want.0.1, want.0.2, want.1.0, want.1.1, want.1.2]);
        worst = worst.max((returned - 1.0).abs().min((returned - es).abs()));
        assert!(
            returned <= 1.0f64.max(es) + 0.02,
            "sheen {} seen at cos {}: the furnace returns {returned} (the integral's Es is {es})",
            case.surface.sheen_roughness,
            case.surface.v.2
        );
        assert!((returned - want).abs() < 0.01 * want.max(1.0), "sheen furnace {returned} vs the reference's {want}");
    }
    eprintln!("sheen furnace: the largest departure from 1 (or from the lobe's own albedo) is {worst:.4}");
}

#[test]
fn the_sheen_table_interpolates_to_a_few_percent() {
    let table = reference::sheen_table();
    let (mut worst_all, mut worst_cloth) = (0.0f64, 0.0f64);
    for rs in [0.1, 0.2, 0.33, 0.47, 0.61, 0.76, 0.9] {
        for nv in [0.03, 0.1, 0.2, 0.3, 0.45, 0.55, 0.7, 0.8, 0.95] {
            let exact = reference::sheen_albedo_with(nv, rs, 512, 512);
            let got = reference::table_albedo(&table, nv, rs);
            // Where it matters: what the sheen layers on can only be told
            // apart past a hundredth of the light.
            let e = (got.min(1.0) - exact.min(1.0)).abs();
            worst_all = worst_all.max(e);
            if rs >= 0.3 {
                worst_cloth = worst_cloth.max(e);
            }
        }
    }
    eprintln!(
        "sheen table: largest error of the capped albedo {worst_all:.4}, for sheen roughness 0.3 and up {worst_cloth:.4}"
    );
    assert!(worst_all < 0.025, "the table is off by {worst_all} somewhere");
    assert!(worst_cloth < 0.006, "the table is off by {worst_cloth} for cloth-like roughness");
    assert!(table.iter().all(|&x| (0.0..=1.0).contains(&x)), "the table is capped at 1");
}

/// A kernel that evaluates scalar functions, one case a row.
fn scalars(gpu: &Gpu, body: &str, input: &[f32], in_stride: usize, out_stride: usize) -> Vec<f32> {
    compute(
        gpu,
        SURFACE_LOBES_WGSL,
        &format!("fn kernel(i: u32) {{ {body} }}"),
        input,
        in_stride,
        out_stride,
        None,
        None,
    )
}

#[test]
fn the_sheen_lookup_is_the_tables_bilinear_interpolation() {
    let Some(gpu) = gpu() else { return };
    let table = reference::sheen_table();
    let mut input = Vec::new();
    for rs in [0.01, 0.07, 0.2, 0.55, 1.0, 1.5] {
        for nv in [-0.1, 0.0, 0.03, 0.4, 0.77, 1.0, 1.2] {
            input.extend([nv as f32, rs as f32]);
        }
    }
    let got = scalars(&gpu, "put(i, 0u, lobe_sheen_albedo(arg(i, 0u), arg(i, 1u)));", &input, 2, 1);
    for (case, got) in input.chunks(2).zip(&got) {
        let want = reference::table_albedo(&table, f64::from(case[0]), f64::from(case[1]));
        assert!(
            (f64::from(*got) - want).abs() < 1e-4 * want.max(1.0),
            "cos {} roughness {}: {got} vs {want}",
            case[0],
            case[1]
        );
    }
}

#[test]
fn anisotropy_of_zero_is_the_isotropic_lobe() {
    let Some(gpu) = gpu() else { return };
    // The anisotropic distribution and visibility, at an anisotropy of 0
    // (both widths alpha), against the isotropic ones at the same angles:
    // the same formulas rearranged, so equal to what f32 rounds them to.
    let mut input = Vec::new();
    for roughness in [0.06, 0.2, 0.5, 0.9] {
        for nl_angle in [0.1, 0.6, 1.1, 1.5] {
            for nv_angle in [0.2, 0.8, 1.3] {
                for (phi_l, phi_v) in [(0.0, 3.0), (1.0, 2.0), (2.5, 5.0)] {
                    let (l, v) = (dir(nl_angle, phi_l), dir(nv_angle, phi_v));
                    let h = (l + v).unit();
                    input.extend([
                        roughness as f32,
                        l.0 as f32,
                        l.1 as f32,
                        l.2 as f32,
                        v.0 as f32,
                        v.1 as f32,
                        v.2 as f32,
                        h.0 as f32,
                        h.1 as f32,
                        h.2 as f32,
                    ]);
                }
            }
        }
    }
    let body = "
      let alpha = lobe_aniso_alpha(arg(i, 0u), 0.0);
      let l = arg3(i, 1u); let v = arg3(i, 4u); let h = arg3(i, 7u);
      let a2 = clamp(arg(i, 0u), 0.04, 1.0) * clamp(arg(i, 0u), 0.04, 1.0) * clamp(arg(i, 0u), 0.04, 1.0) * clamp(arg(i, 0u), 0.04, 1.0);
      put(i, 0u, lobe_aniso_distribution(h.z, h.x, h.y, alpha));
      put(i, 1u, lobe_ggx_distribution(h.z, a2));
      put(i, 2u, lobe_aniso_visibility(l.z, v.z, l.x, l.y, v.x, v.y, alpha));
      put(i, 3u, lobe_smith_visibility(l.z, v.z, a2));
    ";
    let got = scalars(&gpu, body, &input, 10, 4);
    let mut worst = 0.0f64;
    for (case, got) in input.chunks(10).zip(got.chunks(4)) {
        for (a, b, what) in [(got[0], got[1], "distribution"), (got[2], got[3], "visibility")] {
            let e = error(f64::from(a), f64::from(b), 1e-6);
            worst = worst.max(e);
            assert!(
                e < 5e-4,
                "roughness {}: the anisotropic {what} {a} is not the isotropic {b} (relative {e})",
                case[0]
            );
        }
    }
    eprintln!("anisotropy 0 against the isotropic lobe: largest relative difference {worst:.2e}");
}

#[test]
fn the_coats_highlight_follows_its_own_normal() {
    let Some(gpu) = gpu() else { return };
    // The coat alone (no base lobe) lit from directions in the plane of the
    // view and a normal tilted 20 degrees: it peaks where the coat normal
    // mirrors the view, whatever the base's normal.
    let tilt = 20.0f64.to_radians();
    let view = 35.0f64.to_radians();
    let mut cases = Vec::new();
    for degrees in -800..=800 {
        let angle = f64::from(degrees) * 0.1f64.to_radians();
        let mut case = Case::default();
        case.surface.v = dir(view, 0.0);
        case.surface.base = V::all(0.0);
        case.surface.f0 = V::all(0.0);
        case.surface.f90 = 0.0;
        case.surface.coat = 1.0;
        case.surface.coat_roughness = 0.12;
        case.surface.coat_normal = dir(tilt.abs(), 0.0);
        case.l = V(angle.sin(), 0.0, angle.cos());
        case.lobes = COAT;
        cases.push(case);
    }
    let got = lights(&gpu, &cases);
    let table = reference::sheen_table();
    let peak_of =
        |value: &dyn Fn(usize) -> f64| (0..cases.len()).max_by(|&a, &b| value(a).total_cmp(&value(b))).unwrap();
    let shader_peak = peak_of(&|k| f64::from(got[k][3]));
    let reference_peak = peak_of(&|k| reference::light(&cases[k].surface, cases[k].l, COAT, &table).specular.0);
    let angle_of = |k: usize| cases[k].l.0.atan2(cases[k].l.2).to_degrees();
    // The mirror direction of the view about the coat's normal, in the plane.
    let (n, v) = (cases[0].surface.coat_normal, cases[0].surface.v);
    let mirror = n * (2.0 * n.dot(v)) - v;
    let analytic = mirror.0.atan2(mirror.2).to_degrees();
    eprintln!(
        "coat highlight: shader peak {:.2} degrees, f64 reference {:.2}, mirror of the view about the coat normal {analytic:.2}",
        angle_of(shader_peak),
        angle_of(reference_peak)
    );
    assert!(
        (angle_of(shader_peak) - angle_of(reference_peak)).abs() <= 0.3,
        "the shader's coat highlight is not where the reference's is"
    );
    // Kelemen's visibility moves the lobe's maximum a little off the exact mirror.
    assert!(
        (angle_of(shader_peak) - analytic).abs() < 2.5,
        "the coat's highlight is not where its own normal mirrors the view"
    );
    // And the base normal (z) would have put it at the view's mirror, -35 degrees.
    assert!((angle_of(shader_peak) + 35.0).abs() > 30.0);
}

#[test]
fn the_wrap_and_the_back_lit_term_match_the_reference() {
    let Some(gpu) = gpu() else { return };
    let mut input = Vec::new();
    for signed in [-1.0, -0.6, -0.5, -0.3, -0.05, 0.0, 0.2, 0.7, 1.0] {
        for amount in [0.0, 0.3, 1.0] {
            for color in [V(1.0, 0.5, 0.2), V(0.1, 0.1, 0.1), V(1.0, 1.0, 1.0)] {
                input.extend([signed as f32, amount as f32, color.0 as f32, color.1 as f32, color.2 as f32]);
            }
        }
    }
    let got = scalars(&gpu, "put3(i, 0u, lobe_subsurface_diffuse(arg(i, 0u), arg(i, 1u), arg3(i, 2u)));", &input, 5, 3);
    for (case, got) in input.chunks(5).zip(got.chunks(3)) {
        let want = reference::subsurface_diffuse(
            f64::from(case[0]),
            f64::from(case[1]),
            V::from32([case[2], case[3], case[4]]),
        ) * PI;
        #[allow(clippy::needless_range_loop, reason = "a channel of three arrays")]
        for c in 0..3 {
            assert!(
                (f64::from(got[c]) - want.array()[c]).abs() < 1e-5,
                "wrap at {} amount {}: channel {c} is {} vs {}",
                case[0],
                case[1],
                got[c],
                want.array()[c]
            );
        }
    }
    // The back-lit term: the light behind the surface and the eye in front.
    let mut input = Vec::new();
    for light in [dir(2.6, 0.0), dir(2.0, 1.0), dir(1.6, 2.0), dir(0.5, 3.0), dir(2.9, 4.0)] {
        for eye in [dir(0.3, 0.0), dir(1.0, 3.0), dir(1.4, 6.0)] {
            for (amount, thin) in [(1.0, 0.0), (0.5, 0.3), (1.0, 1.0), (0.0, 0.0)] {
                input.extend([light.0, light.1, light.2, eye.0, eye.1, eye.2, amount, thin].map(|x| x as f32));
            }
        }
    }
    let body = "put3(i, 0u, lobe_subsurface_back(arg3(i, 0u), vec3f(0.0, 0.0, 1.0), arg3(i, 3u), arg(i, 6u), vec3f(0.9, 0.6, 0.3), arg(i, 7u)));";
    let got = scalars(&gpu, body, &input, 8, 3);
    let mut peak = 0.0f64;
    for (case, got) in input.chunks(8).zip(got.chunks(3)) {
        let l = V(f64::from(case[0]), f64::from(case[1]), f64::from(case[2]));
        let v = V(f64::from(case[3]), f64::from(case[4]), f64::from(case[5]));
        let want = reference::subsurface_back(
            l,
            V(0.0, 0.0, 1.0),
            v,
            f64::from(case[6]),
            V(0.9, 0.6, 0.3),
            f64::from(case[7]),
        ) * PI;
        peak = peak.max(want.0);
        #[allow(clippy::needless_range_loop, reason = "a channel of three arrays")]
        for c in 0..3 {
            assert!(
                (f64::from(got[c]) - want.array()[c]).abs() < 1e-5,
                "back-lit term, case {case:?}: channel {c} is {} vs {}",
                got[c],
                want.array()[c]
            );
        }
    }
    assert!(peak > 0.2, "the back-lit cases reach {peak}: the test would see the term");
}

#[test]
fn the_wrap_diffuse_gives_back_no_more_than_its_albedo() {
    let Some(gpu) = gpu() else { return };
    let mut cases = Vec::new();
    for amount in [0.0, 0.25, 0.6, 1.0] {
        for color in [V::all(0.0), V(1.0, 0.5, 0.2), V::all(1.0)] {
            let mut case = Case::default();
            case.surface.base = V::all(1.0);
            case.surface.f0 = V::all(0.0);
            case.surface.f90 = 0.0;
            case.surface.v = dir(0.4, 0.0);
            case.surface.subsurface = amount;
            case.surface.subsurface_color = color;
            // Opaque: no back-lit term, only the wrapped diffuse.
            case.surface.thin = 1.0;
            case.lobes = SUBSURFACE;
            cases.push(case);
        }
    }
    let got = furnace(&gpu, &cases);
    for (case, albedo) in cases.iter().zip(&got) {
        #[allow(clippy::needless_range_loop, reason = "a channel of three arrays")]
        for c in 0..3 {
            // The sphere's integral of the wrap diffuse, light from all round.
            assert!(
                albedo[c] <= 1.0 + 3e-3,
                "subsurface {} {:?}: channel {c} returns {} of its light",
                case.surface.subsurface,
                case.surface.subsurface_color,
                albedo[c]
            );
        }
    }
    // With a color of 1 and all subsurface the wrap spends exactly its albedo.
    let full =
        cases.iter().position(|c| c.surface.subsurface == 1.0 && c.surface.subsurface_color == V::all(1.0)).unwrap();
    assert!((got[full][0] - 1.0).abs() < 4e-3, "a wrap of 0.5 conserves energy over the sphere: {}", got[full][0]);
    // Lambert's hemisphere alone is 1; the wrap moves some of it past the terminator.
    let lambert = cases.iter().position(|c| c.surface.subsurface == 0.0).unwrap();
    assert!((got[lambert][0] - 1.0).abs() < 4e-3, "lambert's albedo {}", got[lambert][0]);
}

#[test]
fn transmission_maths_matches_the_references() {
    let Some(gpu) = gpu() else { return };
    // The face-on reflectance from an index of refraction.
    let iors = [1.0f32, 1.33, 1.5, 2.4, 1.05];
    let got = scalars(&gpu, "put(i, 0u, lobe_ior_f0(arg(i, 0u)));", &iors, 1, 1);
    for (ior, got) in iors.iter().zip(&got) {
        let want = reference::ior_f0(f64::from(*ior));
        assert!((f64::from(*got) - want).abs() < 1e-6, "ior {ior}: f0 {got} vs {want}");
    }
    assert_eq!(got[0], 0.0, "an index of 1 reflects nothing");
    assert_eq!(got[2], 0.04f32, "glass of 1.5 reflects 4%, to the bit");
    assert!((f64::from(got[3]) - 0.1698).abs() < 1e-3, "ior 2.4: {}", got[3]);
    // Snell's law: the refracted angle's sine is the incident's over the
    // index; total internal reflection leaving glass at a grazing angle.
    let mut input = Vec::new();
    for ior in [1.0f64, 1.33, 1.5, 2.4] {
        for degrees in [0.0f64, 15.0, 40.0, 65.0, 85.0] {
            let angle = degrees.to_radians();
            // `view` toward the eye, 'angle' off the normal in the xz plane.
            input.extend([angle.sin() as f32, 0.0, angle.cos() as f32, (1.0 / ior) as f32]);
        }
    }
    // From glass to air (eta over 1): total internal reflection past the critical angle.
    for degrees in [10.0f64, 30.0, 41.0, 42.5, 60.0] {
        let angle = degrees.to_radians();
        input.extend([angle.sin() as f32, 0.0, angle.cos() as f32, 1.5]);
    }
    let got = scalars(&gpu, "put3(i, 0u, lobe_refract(arg3(i, 0u), vec3f(0.0, 0.0, 1.0), arg(i, 3u)));", &input, 4, 3);
    let (mut bent, mut reflected) = (0, 0);
    for (case, got) in input.chunks(4).zip(got.chunks(3)) {
        let sin_in = f64::from(case[0]);
        let eta = f64::from(case[3]);
        let sin_out = eta * sin_in;
        if sin_out > 1.0 {
            reflected += 1;
            assert!(got.iter().all(|&c| c == 0.0), "total internal reflection (sin {sin_in}, eta {eta}) gives {got:?}");
            continue;
        }
        bent += 1;
        // The ray continues into the surface (-z) with sin(out) = eta * sin(in), away from the view's side.
        let out = V::from32([got[0], got[1], got[2]]);
        assert!((out.length() - 1.0).abs() < 1e-5, "a refracted direction is a unit vector: {out:?}");
        assert!(
            (out.0 + sin_out).abs() < 1e-5 && out.1.abs() < 1e-6,
            "sin of the refracted angle {} vs Snell's {sin_out}",
            -out.0
        );
        assert!(out.2 < 0.0, "it enters the surface");
    }
    assert!(bent == 23 && reflected == 2, "{bent} refractions and {reflected} reflections tried");
    // Beer-Lambert attenuation in the pow form, and its two exceptions.
    let mut input = Vec::new();
    for (color, thickness, distance) in [
        (V(0.5, 0.25, 1.0), 2.0, 1.0),
        (V(0.9, 0.2, 0.05), 0.3, 1.0),
        (V(0.5, 0.25, 1.0), 0.0, 1.0),
        (V(0.5, 0.25, 1.0), 2.0, 0.0),
        (V(0.8, 0.6, 0.0), 1.0, 4.0),
    ] {
        input.extend([color.0, color.1, color.2, thickness, distance].map(|x| x as f32));
    }
    let got = scalars(&gpu, "put3(i, 0u, lobe_volume_attenuation(arg3(i, 0u), arg(i, 3u), arg(i, 4u)));", &input, 5, 3);
    for (case, got) in input.chunks(5).zip(got.chunks(3)) {
        let color = V::from32([case[0], case[1], case[2]]);
        let (thickness, distance) = (f64::from(case[3]), f64::from(case[4]));
        let want = if thickness <= 0.0 || distance <= 0.0 {
            V::all(1.0)
        } else {
            color.each(|c| c.powf(thickness / distance))
        };
        #[allow(clippy::needless_range_loop, reason = "a channel of three arrays")]
        for c in 0..3 {
            assert!(
                (f64::from(got[c]) - want.array()[c]).abs() < 1e-5,
                "attenuation {case:?}: channel {c} is {} vs {}",
                got[c],
                want.array()[c]
            );
        }
    }
    // The picture's mip level: rougher and denser glass blurs more, and it stops at the last level.
    let mut input = Vec::new();
    for roughness in [0.0, 0.1, 0.5, 1.0] {
        for ior in [1.0, 1.25, 1.5, 2.0] {
            input.extend([roughness, ior, 11.0, 10.0].map(|x| x as f32));
        }
    }
    let got = scalars(
        &gpu,
        "put(i, 0u, lobe_transmission_level(arg(i, 0u), arg(i, 1u), arg(i, 2u), arg(i, 3u)));",
        &input,
        4,
        1,
    );
    for (case, got) in input.chunks(4).zip(&got) {
        let (r, ior) = (f64::from(case[0]), f64::from(case[1]));
        let want = (11.0 * (r * (2.0 * ior - 2.0).clamp(0.0, 1.0)).clamp(0.0, 1.0)).min(10.0);
        assert!((f64::from(*got) - want).abs() < 1e-4, "level at roughness {r} ior {ior}: {got} vs {want}");
    }
    assert_eq!(got[3], 0.0, "ior 1 reads the sharp picture whatever the roughness");
    assert_eq!(*got.last().unwrap(), 10.0, "stops at the last level");
}

/// `parallax_uv` for each case: uv (2), the view's and the light's tangent-space
/// vectors (3 each), the scale and whether to shade; the displaced uv, the
/// height found and the shadow come back.
fn parallax(gpu: &Gpu, heights: &Heights, cases: &[[f32; 9]]) -> Vec<[f32; 4]> {
    let kernel = "
@group(0) @binding(2) var heights: texture_2d<f32>;
@group(0) @binding(3) var height_sampler: sampler;
fn kernel(i: u32) {
  let r = parallax_uv(heights, height_sampler, vec2f(arg(i, 0u), arg(i, 1u)), normalize(arg3(i, 2u)), normalize(arg3(i, 5u)), arg(i, 8u), 0.0, true);
  put(i, 0u, r.uv.x); put(i, 1u, r.uv.y); put(i, 2u, r.height); put(i, 3u, r.shadow);
}";
    let input: Vec<f32> = cases.iter().flatten().copied().collect();
    let out = compute(gpu, SURFACE_LOBES_WGSL, kernel, &input, 9, 4, None, Some(heights));
    out.chunks(4).map(|c| c.try_into().unwrap()).collect()
}

/// A view from `degrees` off the normal toward +x, in tangent space.
fn tilted(degrees: f64) -> [f32; 3] {
    let a = degrees.to_radians();
    [a.sin() as f32, 0.0, a.cos() as f32]
}

#[test]
fn a_height_step_shifts_the_uv_by_the_analytic_amount() {
    let Some(gpu) = gpu() else { return };
    // 64 texels across: the left half a plateau at height 102/255 = 0.4, the
    // right half one at 204/255 = 0.8.
    let (low, high) = (102u8, 204u8);
    let red: Vec<u8> = (0..64 * 64).map(|k| if k % 64 < 32 { low } else { high }).collect();
    let heights = Heights::new(&gpu, 64, 64, &red);
    let scale = 0.1f32;
    let (h_low, h_high) = (f64::from(low) / 255.0, f64::from(high) / 255.0);
    let mut worst = 0.0f64;
    for degrees in [20.0, 35.0, 50.0, 65.0] {
        let view = tilted(degrees);
        let cases = [
            [0.25, 0.5, view[0], view[1], view[2], 0.0, 0.0, 1.0, scale],
            [0.78, 0.5, view[0], view[1], view[2], 0.0, 0.0, 1.0, scale],
        ];
        let got = parallax(&gpu, &heights, &cases);
        let (on_low, on_high) = (f64::from(got[0][0]), f64::from(got[1][0]));
        // The ray sinks 1 - height along the way; each plateau's uv shifts by that
        // much of the view's tangent over its z, times the scale.
        let tan = degrees.to_radians().tan();
        let want_low = 0.25 - tan * f64::from(scale) * (1.0 - h_low);
        let want_high = 0.78 - tan * f64::from(scale) * (1.0 - h_high);
        assert!(
            (on_low - want_low).abs() < 1e-4,
            "{degrees} degrees: the low plateau reads uv {on_low}, expected {want_low}"
        );
        assert!(
            (on_high - want_high).abs() < 1e-4,
            "{degrees} degrees: the high plateau reads uv {on_high}, expected {want_high}"
        );
        // The lower plateau, relative to the higher: (h1 - h2) p tan(theta), back toward the view's far side.
        let relative = (on_low - 0.25) - (on_high - 0.78);
        let analytic = -(h_high - h_low) * f64::from(scale) * tan;
        worst = worst.max((relative - analytic).abs());
        assert!(
            (relative - analytic).abs() < 2e-4,
            "{degrees} degrees: relative shift {relative}, (h1 - h2) p tan = {analytic}"
        );
        assert!((f64::from(got[0][2]) - h_low).abs() < 1e-3 && (f64::from(got[1][2]) - h_high).abs() < 1e-3);
    }
    eprintln!("POM step: the relative shift of the plateaus is within {worst:.1e} of (h1 - h2) p tan(theta)");
    // At normal incidence nothing moves.
    let flat = parallax(
        &gpu,
        &heights,
        &[[0.25, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, scale], [0.9, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, scale]],
    );
    assert_eq!(
        (flat[0][0], flat[0][1], flat[1][0], flat[1][1]),
        (0.25, 0.5, 0.9, 0.3),
        "no parallax at normal incidence"
    );
}

#[test]
fn the_march_finds_where_a_sloped_surface_meets_the_ray() {
    let Some(gpu) = gpu() else { return };
    // A ramp from height 0.3 at u = 0 up to 0.9 at u = 1: the ray's depth and
    // the surface's meet at one place the refinement's linear step is exact for.
    let red: Vec<u8> = (0..256 * 4)
        .map(|k| {
            let u = f64::from(k % 256) / 255.0;
            ((0.3 + 0.6 * u) * 255.0).round() as u8
        })
        .collect();
    let heights = Heights::new(&gpu, 256, 4, &red);
    let scale = 0.12f64;
    let slope = 0.6f64;
    let mut cases = Vec::new();
    let mut want = Vec::new();
    for degrees in [10.0f64, 30.0, 55.0, 70.0] {
        for u0 in [0.3f64, 0.5, 0.7] {
            let view = tilted(degrees);
            cases.push([u0 as f32, 0.5, view[0], view[1], view[2], 0.0, 0.0, 1.0, scale as f32]);
            // uv(d) = u0 - m d (m the shift per unit depth); height(u) = 0.3 + slope u.
            let m = degrees.to_radians().tan() * scale;
            let depth = (1.0 - 0.3 - slope * u0) / (1.0 - slope * m);
            want.push(u0 - m * depth);
        }
    }
    let got = parallax(&gpu, &heights, &cases);
    for ((case, got), want) in cases.iter().zip(&got).zip(&want) {
        assert!(
            (f64::from(got[0]) - want).abs() < 2.5e-3,
            "uv {} view {:?}: the march lands at {} but the ray meets the ramp at {want}",
            case[0],
            &case[2..5],
            got[0]
        );
    }
}

#[test]
fn the_march_takes_fewer_layers_looking_straight_down() {
    let Some(gpu) = gpu() else { return };
    let input: Vec<f32> = (0..=40).map(|k| k as f32 / 40.0).collect();
    let got = scalars(&gpu, "put(i, 0u, parallax_layers(arg(i, 0u)));", &input, 1, 1);
    assert_eq!(got[0], 32.0, "a grazing view takes 32 layers");
    assert_eq!(*got.last().unwrap(), 8.0, "a view straight down takes 8");
    assert!(got.windows(2).all(|w| w[1] <= w[0]), "layers never grow as the view comes up: {got:?}");
    let negative = scalars(&gpu, "put(i, 0u, parallax_layers(arg(i, 0u)));", &[-1.0, -0.5], 1, 1);
    assert_eq!(negative, [8.0, 20.0]);
}

#[test]
fn a_groove_shades_itself_and_a_plateau_does_not() {
    let Some(gpu) = gpu() else { return };
    // 64 texels: a groove of height 0 eight texels wide in the middle of a
    // surface at height 1.
    let red: Vec<u8> = (0..64 * 64).map(|k| if (28..36).contains(&(k % 64)) { 0 } else { 255 }).collect();
    let heights = Heights::new(&gpu, 64, 64, &red);
    let scale = 0.1f32;
    // Looking straight down so that the point is where it is, the light 45 degrees up toward +x.
    let light = tilted(45.0);
    let up = [0.0, 0.0, 1.0];
    let at = |u: f32, light: [f32; 3]| [u, 0.5, up[0], up[1], up[2], light[0], light[1], light[2], scale];
    let got = parallax(
        &gpu,
        &heights,
        &[at(0.5, light), at(0.1, light), at(0.85, light), at(0.5, tilted(5.0)), at(0.5, [-light[0], 0.0, light[2]])],
    );
    assert!(got[0][3] < 0.02, "the groove's floor is in shadow: {}", got[0][3]);
    assert_eq!(got[1][3], 1.0, "a plateau's top is lit");
    assert_eq!(got[2][3], 1.0, "a plateau's top is lit away from the groove too");
    assert!(got[3][3] > 0.99, "a light almost overhead reaches the groove's floor: {}", got[3][3]);
    assert!(got[4][3] < 0.02, "and from the other side");
    assert!((got[0][2]).abs() < 1e-3, "the groove's floor is at height 0: {}", got[0][2]);
    // The more the light leans, the less of the groove's floor it reaches; the
    // wall (0.0625 away, a rise of 1 at scale 0.1) is cleared at about 32 degrees.
    let sweep: Vec<[f32; 9]> = (0..=80).map(|degrees| at(0.5, tilted(f64::from(degrees)))).collect();
    let shade: Vec<f32> = parallax(&gpu, &heights, &sweep).iter().map(|r| r[3]).collect();
    assert!(shade.windows(2).all(|w| w[1] <= w[0] + 1e-6), "the groove darkens as the light leans: {shade:?}");
    assert!(shade[20] > 0.99 && shade[45] < 0.01, "lit at 20 degrees and dark at 45: {} {}", shade[20], shade[45]);
}

#[test]
fn the_new_game_template_carries_the_current_lights_and_surface_blocks() {
    let template = include_str!("../../../pen/templates/new/src/assets/lit.wgsl");
    for (name, source) in
        [("lights", crate::LIGHTS_WGSL), ("surface_lobes", SURFACE_LOBES_WGSL), ("surface", SURFACE_WGSL)]
    {
        let marker = format!("// BEGIN {name}\n");
        let begin = template.find(&marker).unwrap_or_else(|| panic!("a begin marker for {name}")) + marker.len();
        let end = template.find(&format!("// END {name}\n")).unwrap_or_else(|| panic!("an end marker for {name}"));
        assert_eq!(&template[begin..end], source, "copy shaders/{name}.wgsl into lit.wgsl");
    }
    // The transmission glue is not in the template: it needs the translucency library.
    assert!(!template.contains("fn surface_transmit"));
    assert!(SURFACE_TRANSMISSION_WGSL.contains("fn surface_transmit"));
}

#[test]
fn the_sky_lookup_of_an_anisotropic_surface_bends_toward_the_view_across_the_grain() {
    let Some(gpu) = gpu() else { return };
    let mut input = Vec::new();
    let mut cases = Vec::new();
    for roughness in [0.05, 0.2, 0.6] {
        for anisotropy in [-0.9, -0.3, 0.4, 1.0] {
            for (theta, phi) in [(0.2, 0.0), (0.9, 1.0), (1.3, 4.0)] {
                let v = dir(theta, phi);
                let (t, b) = (V(0.8, 0.6, 0.0), V(-0.6, 0.8, 0.0));
                cases.push((roughness, anisotropy, v, t, b));
                input.extend([
                    roughness as f32,
                    anisotropy as f32,
                    v.0 as f32,
                    v.1 as f32,
                    v.2 as f32,
                    t.0 as f32,
                    t.1 as f32,
                    t.2 as f32,
                    b.0 as f32,
                    b.1 as f32,
                    b.2 as f32,
                ]);
            }
        }
    }
    let body = "put3(i, 0u, lobe_bent_normal(vec3f(0.0, 0.0, 1.0), arg3(i, 5u), arg3(i, 8u), arg3(i, 2u), arg(i, 0u), arg(i, 1u)));";
    let got = scalars(&gpu, body, &input, 11, 3);
    let mut moved = 0.0f64;
    for ((roughness, anisotropy, v, t, b), got) in cases.iter().zip(got.chunks(3)) {
        let want = reference::bent_normal(V(0.0, 0.0, 1.0), *t, *b, *v, *roughness, *anisotropy);
        moved = moved.max((want - V(0.0, 0.0, 1.0)).length());
        for c in 0..3 {
            assert!(
                (f64::from(got[c]) - want.array()[c]).abs() < 2e-5,
                "roughness {roughness} anisotropy {anisotropy}: {got:?} vs {want:?}"
            );
        }
    }
    assert!(moved > 0.1, "the lookups bend by up to {moved}: the test would see them");
}
