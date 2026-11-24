//! Filmic tonemapping operators (Shoulder, AgX, ACES Filmic, Khronos PBR Neutral)
//! with HDR display peak headroom support and f64 CPU references.

/// Selectable display tonemapping operator for `present.wgsl`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TonemapOperator {
    /// Pen's original 0.9-knee per-channel shoulder curve (default compatibility).
    #[default]
    Shoulder = 0,
    /// Troy Sobotka / Blender / Godot 4.3+ AgX filmic curve (hue-preserving highlight desaturation).
    AgX = 1,
    /// Stephen Hill fitted ACES RRT+ODT filmic curve.
    AcesFilmic = 2,
    /// Khronos PBR Neutral commerce tonemapper (exact mid-tone hue/albedo fidelity).
    PbrNeutral = 3,
}

impl TonemapOperator {
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::AgX,
            2 => Self::AcesFilmic,
            3 => Self::PbrNeutral,
            _ => Self::Shoulder,
        }
    }
}

/// WGSL module providing `toned_with_mode(srgb_encoded: vec3f, mode: u32, peak: f32) -> vec3f`
/// and legacy `toned(color: vec3f) -> vec3f`. Expects `srgb_decode` / `srgb_encode` from `transfer.wgsl`.
pub const TONEMAP_SUITE_WGSL: &str = r#"
fn tonemap_shoulder(color: vec3f, peak: f32) -> vec3f {
  let shoulder = peak * 0.9;
  let d = max(peak - shoulder, 0.0001);
  let rolled = peak - d * d / (max(color, vec3f(shoulder)) + d - shoulder);
  return select(color, rolled, color > vec3f(shoulder));
}

fn agx_default_contrast(x: vec3f) -> vec3f {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2
    - 40.14 * x4 * x
    + 31.96 * x4
    - 6.868 * x2 * x
    + 0.4298 * x2
    + 0.1191 * x
    - 0.00232;
}

fn tonemap_agx_linear(val_in: vec3f) -> vec3f {
  let val = max(val_in, vec3f(0.0));
  // AgX inset matrix (column-major).
  let inset = mat3x3f(
    vec3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
    vec3f(0.0784335999999992, 0.878468636469772, 0.0784336),
    vec3f(0.0792237451477643, 0.0791661274605434, 0.879142973793104)
  );
  // AgX outset matrix (column-major).
  let outset = mat3x3f(
    vec3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
    vec3f(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
    vec3f(-0.0990297440797205, -0.0989611768448433, 1.15107367264116)
  );
  let min_ev = -12.47393;
  let max_ev = 4.026069;
  var c = inset * val;
  c = max(c, vec3f(1e-10));
  c = log2(c);
  c = clamp((c - vec3f(min_ev)) / (max_ev - min_ev), vec3f(0.0), vec3f(1.0));
  c = agx_default_contrast(c);
  // AgX Punchy mild contrast/saturation look in log-encoded space.
  let luma = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  c = pow(max(c, vec3f(0.0)), vec3f(1.15));
  c = vec3f(luma) + (c - vec3f(luma)) * 1.15;
  c = outset * c;
  return clamp(c, vec3f(0.0), vec3f(1.0));
}

fn tonemap_aces_linear(val_in: vec3f) -> vec3f {
  let v = max(val_in, vec3f(0.0));
  let aces_in = mat3x3f(
    vec3f(0.59719, 0.07600, 0.02840),
    vec3f(0.35458, 0.90834, 0.13383),
    vec3f(0.04823, 0.01566, 0.83777)
  );
  let aces_out = mat3x3f(
    vec3f( 1.60475, -0.10208, -0.00327),
    vec3f(-0.53108,  1.10813, -0.07276),
    vec3f(-0.07367, -0.00605,  1.07602)
  );
  let x = aces_in * v;
  let a = x * (x + vec3f(0.0245786)) - vec3f(0.000090537);
  let b = x * (0.983729 * x + vec3f(0.4329510)) + vec3f(0.238081);
  return clamp(aces_out * (a / b), vec3f(0.0), vec3f(1.0));
}

fn tonemap_pbr_neutral_linear(color_in: vec3f) -> vec3f {
  var color = max(color_in, vec3f(0.0));
  let start_compression = 0.8 - 0.04;
  let desaturation = 0.15;
  let x = min(color.r, min(color.g, color.b));
  let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
  color -= vec3f(offset);
  let peak = max(color.r, max(color.g, color.b));
  if (peak < start_compression) {
    return color;
  }
  let d = 1.0 - start_compression;
  let new_peak = 1.0 - d * d / (peak + d - start_compression);
  color *= new_peak / peak;
  let g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
  return mix(color, vec3f(new_peak), g);
}

fn toned_with_mode(srgb_encoded: vec3f, mode: u32, peak_encoded: f32) -> vec3f {
  if (mode == 0u) {
    return tonemap_shoulder(srgb_encoded, peak_encoded);
  }
  let linear = srgb_decode(max(srgb_encoded, vec3f(0.0)));
  // On HDR displays (peak_encoded > 1.0), scale input down by linear peak before
  // SDR filmic compression and rescale back up so highlights smoothly reach display peak.
  let linear_peak = select(1.0, max(srgb_decode(vec3f(peak_encoded)).r, 1.0), peak_encoded > 1.001);
  let scaled = linear / linear_peak;
  var mapped: vec3f;
  if (mode == 1u) {
    // AgX outputs sRGB-ready contrast-encoded values in [0, 1].
    let agx_srgb = tonemap_agx_linear(scaled);
    mapped = srgb_decode(agx_srgb);
  } else if (mode == 2u) {
    mapped = tonemap_aces_linear(scaled);
  } else {
    mapped = tonemap_pbr_neutral_linear(scaled);
  }
  return srgb_encode(mapped * linear_peak);
}
"#;

/// Evaluates any `TonemapOperator` on linear RGB in f64 (returns linear output in `[0, 1]` for SDR).
pub fn tonemap_linear_f64(linear: [f64; 3], operator: TonemapOperator) -> [f64; 3] {
    let v = linear.map(|c| c.max(0.0));
    match operator {
        TonemapOperator::Shoulder => {
            let enc = v.map(srgb_encode_f64);
            let mapped = enc.map(|c| {
                let s = 0.9;
                let d = 0.1;
                if c > s { 1.0 - d * d / (c + d - s) } else { c }
            });
            mapped.map(srgb_decode_f64)
        }
        TonemapOperator::AgX => {
            let c0 = 0.842_479_062_253_094 * v[0] + 0.078_433_599_999_999_2 * v[1] + 0.079_223_745_147_764_3 * v[2];
            let c1 = 0.042_328_242_261_012_3 * v[0] + 0.878_468_636_469_772 * v[1] + 0.079_166_127_460_543_4 * v[2];
            let c2 = 0.042_375_654_905_705_1 * v[0] + 0.078_433_6 * v[1] + 0.879_142_973_793_104 * v[2];
            let poly = |x: f64| {
                let x = ((x.max(1e-10).log2() + 12.47393) / (4.026069 + 12.47393)).clamp(0.0, 1.0);
                let x2 = x * x;
                let x4 = x2 * x2;
                15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232
            };
            let p = [poly(c0), poly(c1), poly(c2)];
            let luma = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
            let q = p.map(|c| {
                let pc = c.max(0.0).powf(1.15);
                luma + (pc - luma) * 1.15
            });
            let o0 = (1.196_879_005_120_17 * q[0] - 0.098_020_881_140_136_8 * q[1] - 0.099_029_744_079_720_5 * q[2])
                .clamp(0.0, 1.0);
            let o1 = (-0.052_896_851_757_456_2 * q[0] + 1.151_903_129_904_17 * q[1] - 0.098_961_176_844_843_3 * q[2])
                .clamp(0.0, 1.0);
            let o2 = (-0.052_971_635_514_443_8 * q[0] - 0.098_043_450_117_124_1 * q[1] + 1.151_073_672_641_16 * q[2])
                .clamp(0.0, 1.0);
            [srgb_decode_f64(o0), srgb_decode_f64(o1), srgb_decode_f64(o2)]
        }
        TonemapOperator::AcesFilmic => {
            let x0 = 0.59719 * v[0] + 0.35458 * v[1] + 0.04823 * v[2];
            let x1 = 0.07600 * v[0] + 0.90834 * v[1] + 0.01566 * v[2];
            let x2 = 0.02840 * v[0] + 0.13383 * v[1] + 0.83777 * v[2];
            let rrt = |x: f64| (x * (x + 0.0245786) - 0.000090537) / (x * (0.983729 * x + 0.4329510) + 0.238081);
            let y = [rrt(x0), rrt(x1), rrt(x2)];
            [
                (1.60475 * y[0] - 0.53108 * y[1] - 0.07367 * y[2]).clamp(0.0, 1.0),
                (-0.10208 * y[0] + 1.10813 * y[1] - 0.00605 * y[2]).clamp(0.0, 1.0),
                (-0.00327 * y[0] - 0.07276 * y[1] + 1.07602 * y[2]).clamp(0.0, 1.0),
            ]
        }
        TonemapOperator::PbrNeutral => {
            let start = 0.76;
            let desat = 0.15;
            let x = v[0].min(v[1]).min(v[2]);
            let offset = if x < 0.08 { x - 6.25 * x * x } else { 0.04 };
            let mut c = v.map(|ch| ch - offset);
            let peak = c[0].max(c[1]).max(c[2]);
            if peak < start {
                return c;
            }
            let d = 1.0 - start;
            let new_peak = 1.0 - d * d / (peak + d - start);
            for ch in &mut c {
                *ch *= new_peak / peak;
            }
            let g = 1.0 - 1.0 / (desat * (peak - new_peak) + 1.0);
            c.map(|ch| ch * (1.0 - g) + new_peak * g)
        }
    }
}

fn srgb_encode_f64(l: f64) -> f64 {
    let l = l.max(0.0);
    if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 }
}

fn srgb_decode_f64(c: f64) -> f64 {
    let c = c.max(0.0);
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filmic_tonemappers_desaturate_bright_primaries_and_preserve_monotone_gray() {
        // Pure intense red [20, 0, 0]: Shoulder leaves G=B=0 (hue-skewed neon red), while
        // AgX, ACES, and PBR Neutral desaturate smoothly toward warm white.
        let bright_red = [20.0, 0.0, 0.0];
        let sh = tonemap_linear_f64(bright_red, TonemapOperator::Shoulder);
        let agx = tonemap_linear_f64(bright_red, TonemapOperator::AgX);
        let aces = tonemap_linear_f64(bright_red, TonemapOperator::AcesFilmic);
        let pbr = tonemap_linear_f64(bright_red, TonemapOperator::PbrNeutral);
        assert_eq!(sh[1], 0.0);
        assert!(agx[1] > 0.1 && agx[2] > 0.05, "AgX should desaturate intense red: {agx:?}");
        assert!(aces[1] > 0.05 && aces[2] > 0.01, "ACES should desaturate intense red: {aces:?}");
        assert!(pbr[1] > 0.2 && pbr[2] > 0.2, "PBR Neutral should desaturate intense red: {pbr:?}");

        // Mid-gray monotonicity across all 4 operators.
        for op in [
            TonemapOperator::Shoulder,
            TonemapOperator::AgX,
            TonemapOperator::AcesFilmic,
            TonemapOperator::PbrNeutral,
        ] {
            let low = tonemap_linear_f64([0.05; 3], op)[0];
            let mid = tonemap_linear_f64([0.18; 3], op)[0];
            let high = tonemap_linear_f64([2.0; 3], op)[0];
            assert!(low < mid && mid < high && high <= 1.0001, "{op:?}: {low}, {mid}, {high}");
        }
    }
}
