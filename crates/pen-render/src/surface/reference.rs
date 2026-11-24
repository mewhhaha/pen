//! Independent f64 references for the standard surface's lobes, written from
//! the material design (Ink's `lighting.brdf` conventions: a lobe's response is
//! per unit radiance, with the cosine in it) and not from the WGSL.
#![allow(clippy::many_single_char_names, reason = "the formulas' own symbols")]
use std::f64::consts::PI;
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V(pub f64, pub f64, pub f64);

impl Add for V {
    type Output = V;
    fn add(self, o: V) -> V {
        V(self.0 + o.0, self.1 + o.1, self.2 + o.2)
    }
}
impl Sub for V {
    type Output = V;
    fn sub(self, o: V) -> V {
        V(self.0 - o.0, self.1 - o.1, self.2 - o.2)
    }
}
impl Mul<f64> for V {
    type Output = V;
    fn mul(self, k: f64) -> V {
        V(self.0 * k, self.1 * k, self.2 * k)
    }
}
impl Mul<V> for V {
    type Output = V;
    fn mul(self, o: V) -> V {
        V(self.0 * o.0, self.1 * o.1, self.2 * o.2)
    }
}
impl Neg for V {
    type Output = V;
    fn neg(self) -> V {
        V(-self.0, -self.1, -self.2)
    }
}

impl V {
    pub fn all(x: f64) -> V {
        V(x, x, x)
    }
    pub fn from32(a: [f32; 3]) -> V {
        V(f64::from(a[0]), f64::from(a[1]), f64::from(a[2]))
    }
    pub fn dot(self, o: V) -> f64 {
        self.0 * o.0 + self.1 * o.1 + self.2 * o.2
    }
    pub fn cross(self, o: V) -> V {
        V(self.1 * o.2 - self.2 * o.1, self.2 * o.0 - self.0 * o.2, self.0 * o.1 - self.1 * o.0)
    }
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }
    pub fn unit(self) -> V {
        self * (1.0 / self.length())
    }
    pub fn max3(self) -> f64 {
        self.0.max(self.1).max(self.2)
    }
    pub fn each(self, f: impl Fn(f64) -> f64) -> V {
        V(f(self.0), f(self.1), f(self.2))
    }
    pub fn array(self) -> [f64; 3] {
        [self.0, self.1, self.2]
    }
}

pub fn clamp(x: f64, low: f64, high: f64) -> f64 {
    x.max(low).min(high)
}

pub fn mix(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

/// The GGX distribution for alpha `a` (the square of the perceptual roughness).
pub fn ggx_d(nh: f64, a: f64) -> f64 {
    let a2 = a * a;
    let den = nh * nh * (a2 - 1.0) + 1.0;
    a2 / (PI * den * den)
}

/// The height-correlated Smith visibility for alpha `a`.
pub fn smith_v(nl: f64, nv: f64, a: f64) -> f64 {
    let a2 = a * a;
    0.5 / (nl * (nv * nv * (1.0 - a2) + a2).sqrt() + nv * (nl * nl * (1.0 - a2) + a2).sqrt()).max(1e-6)
}

pub fn schlick(f0: V, f90: f64, vh: f64) -> V {
    f0 + (V::all(f90) - f0) * (1.0 - vh).powi(5)
}

/// The dielectric's face-on reflectance at index `ior` (at least 1).
pub fn ior_f0(ior: f64) -> f64 {
    let ior = ior.max(1.0);
    ((ior - 1.0) / (ior + 1.0)).powi(2)
}

/// A surface and everything its lobes read.
#[derive(Clone, Debug)]
pub struct Surface {
    pub n: V,
    pub v: V,
    pub roughness: f64,
    pub metallic: f64,
    pub base: V,
    pub f0: V,
    pub f90: f64,
    pub coat: f64,
    pub coat_roughness: f64,
    pub coat_normal: V,
    pub sheen: V,
    pub sheen_roughness: f64,
    pub anisotropy: f64,
    pub tangent: V,
    pub bitangent: V,
    pub subsurface: f64,
    pub subsurface_color: V,
    pub thin: f64,
    pub transmission: f64,
}

/// What one light brings, per unit radiance, in Ink's units: `diffuse` is
/// Ink's diffuse response over base * (1 - metallic), which multiplies it
/// (Lambert's is `max(nl, 0) / pi`), `specular` everything else.
#[derive(Clone, Copy, Debug)]
pub struct Response {
    pub diffuse: V,
    pub specular: V,
}

pub fn wrap(x: f64, w: f64) -> f64 {
    clamp((x + w) / ((1.0 + w) * (1.0 + w)), 0.0, 1.0)
}

/// The cheap subsurface's diffuse bracket over pi (Lambert's is `max(nl, 0) / pi`).
pub fn subsurface_diffuse(signed_nl: f64, amount: f64, color: V) -> V {
    let nl = signed_nl.max(0.0);
    let lit = color.each(|c| clamp(c + nl, 0.0, 1.0));
    (V::all((1.0 - amount) * nl) + lit * (amount * wrap(signed_nl, 0.5))) * (1.0 / PI)
}

/// The back-lit term, base and pi left out (Ink's is times base / pi * 2).
pub fn subsurface_back(l: V, n: V, v: V, amount: f64, color: V, thin: f64) -> V {
    let through = clamp(v.dot(-(l + n * 0.2).unit()), 0.0, 1.0);
    color * (amount * (1.0 - thin) * through.powi(4) * 2.0 / PI)
}

pub fn sheen_d(nh: f64, rs: f64) -> f64 {
    let a = rs * rs;
    (2.0 + 1.0 / a) * (1.0 - nh * nh).max(0.0).powf(0.5 / a) / (2.0 * PI)
}

pub fn sheen_v(nl: f64, nv: f64) -> f64 {
    1.0 / (4.0 * (nl + nv - nl * nv))
}

/// The sheen lobe's directional albedo for a sheen color of 1, integrated over
/// the hemisphere of light directions by a midpoint rule of `theta` by `phi`
/// cells: the integral of D * V * cos(theta) over the sphere's solid angle.
pub fn sheen_albedo_with(nv: f64, rs: f64, theta: usize, phi: usize) -> f64 {
    let rs = clamp(rs, 0.07, 1.0);
    let nv = nv.clamp(1e-5, 1.0);
    let sv = (1.0 - nv * nv).sqrt();
    let v = V(sv, 0.0, nv);
    let mut sum = 0.0;
    for i in 0..theta {
        let t = (i as f64 + 0.5) / theta as f64 * PI / 2.0;
        let nl = t.cos();
        let mut ring = 0.0;
        for j in 0..phi {
            // The light's azimuth from 0 to pi; the other half mirrors it.
            let p = (j as f64 + 0.5) / phi as f64 * PI;
            let l = V(t.sin() * p.cos(), t.sin() * p.sin(), nl);
            let h = (v + l).unit();
            ring += sheen_d(h.2, rs) * sheen_v(nl, nv) * nl;
        }
        sum += ring * 2.0 * PI / phi as f64 * t.sin() * (PI / 2.0 / theta as f64);
    }
    sum
}

pub fn sheen_albedo(nv: f64, rs: f64) -> f64 {
    sheen_albedo_with(nv, rs, 1024, 1024)
}

/// Sheen's directional albedo by the recipe of the table the shader carries (it
/// is Ink's): the integral of D * V * cos over the light's hemisphere for a
/// white tint, a midpoint rule of `steps` cells in the light's cosine and
/// `steps` in its azimuth (half of it, the lobe being mirror symmetric about
/// the view's plane, doubled), capped at 1.
pub fn sheen_albedo_capped(nv: f64, rs: f64, steps: usize) -> f64 {
    let nv = nv.clamp(1e-5, 1.0);
    let v = V((1.0 - nv * nv).sqrt(), 0.0, nv);
    let mut sum = 0.0;
    for i in 0..steps {
        let nl = (i as f64 + 0.5) / steps as f64;
        let ring = (1.0 - nl * nl).sqrt();
        for j in 0..steps {
            let phi = (j as f64 + 0.5) * PI / steps as f64;
            let l = V(ring * phi.cos(), ring * phi.sin(), nl);
            let h = (v + l).unit();
            sum += sheen_d(clamp(h.2, 0.0, 1.0), clamp(rs, 0.07, 1.0)) * sheen_v(nl, nv) * nl;
        }
    }
    (2.0 * sum * (PI / steps as f64) / steps as f64).min(1.0)
}

/// The cosine and the roughness at column `column` and row `row` of the table
/// the shader carries: 32 columns evenly spaced in the cosine's square root (0
/// to 1) by 16 rows in the square root of the roughness's share of 0.07 to 1.
pub fn table_point(column: usize, row: usize) -> (f64, f64) {
    ((column as f64 / 31.0).powi(2), 0.07 + 0.93 * (row as f64 / 15.0).powi(2))
}

/// The table the shader carries, rows of roughness by columns of cosine.
pub fn sheen_table() -> Vec<f64> {
    static TABLE: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
    TABLE
        .get_or_init(|| {
            std::thread::scope(|scope| {
                let rows: Vec<_> = (0..16)
                    .map(|row| {
                        scope.spawn(move || {
                            (0..32)
                                .map(|column| {
                                    let (nv, rs) = table_point(column, row);
                                    sheen_albedo_capped(nv, rs, 256)
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                rows.into_iter().flat_map(|row| row.join().unwrap()).collect()
            })
        })
        .clone()
}

/// Bilinear lookup of `table` as the shader does it.
pub fn table_albedo(table: &[f64], nv: f64, rs: f64) -> f64 {
    let x = clamp(nv, 0.0, 1.0).sqrt() * 31.0;
    let y = clamp((clamp(rs, 0.07, 1.0) - 0.07) / 0.93, 0.0, 1.0).sqrt() * 15.0;
    let (x0, y0) = ((x.floor() as usize).min(30), (y.floor() as usize).min(14));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let at = |row: usize, column: usize| table[row * 32 + column];
    let a = at(y0, x0) * (1.0 - fx) + at(y0, x0 + 1) * fx;
    let b = at(y0 + 1, x0) * (1.0 - fx) + at(y0 + 1, x0 + 1) * fx;
    a * (1.0 - fy) + b * fy
}

/// The anisotropic widths (tangent, bitangent) at perceptual roughness `r`.
pub fn aniso_alpha(r: f64, a: f64) -> (f64, f64) {
    let alpha = clamp(r, 0.04, 1.0).powi(2);
    (mix(alpha, 1.0, a * a), alpha)
}

pub fn aniso_d(nh: f64, th: f64, bh: f64, at: f64, ab: f64) -> f64 {
    let q = (th / at).powi(2) + (bh / ab).powi(2) + nh * nh;
    1.0 / (PI * at * ab) / (q * q)
}

#[allow(clippy::too_many_arguments, reason = "the formula's own terms")]
pub fn aniso_v(nl: f64, nv: f64, tl: f64, bl: f64, tv: f64, bv: f64, at: f64, ab: f64) -> f64 {
    0.5 / (nv * V(at * tl, ab * bl, nl).length() + nl * V(at * tv, ab * bv, nv).length()).max(1e-6)
}

/// The sky lookup's normal for an anisotropic reflection (bent toward the view).
pub fn bent_normal(n: V, t: V, b: V, v: V, r: f64, a: f64) -> V {
    let grain = if a >= 0.0 { b } else { t };
    let bent = grain.cross(v).cross(grain);
    (n * (1.0 - a.abs() * clamp(5.0 * r, 0.0, 1.0)) + bent * (a.abs() * clamp(5.0 * r, 0.0, 1.0))).unit()
}

pub fn coat_fresnel(vh: f64) -> f64 {
    0.04 + 0.96 * (1.0 - vh).powi(5)
}

/// One light on a surface, which `lobes` (`coat`, `sheen`, `aniso`,
/// `subsurface`, `transmission` as bits 0 to 4) says has which lobes, shining
/// from `l` (unit, toward the light): `(diffuse * (1 - transmission) +
/// specular + back light) * sheen_keep * coat_keep + sheen + coat`.
pub fn light(s: &Surface, l: V, lobes: u32, table: &[f64]) -> Response {
    let signed = s.n.dot(l);
    let nl = signed.max(0.0);
    let subsurface = lobes & 16 != 0 && s.subsurface > 0.0;
    let mut diffuse = V::all(nl / PI);
    let mut back = V::all(0.0);
    if subsurface {
        let amount = clamp(s.subsurface, 0.0, 1.0);
        diffuse = subsurface_diffuse(signed, amount, s.subsurface_color);
        back = s.base * subsurface_back(l, s.n, s.v, amount, s.subsurface_color, s.thin);
    }
    if lobes & 8 != 0 && s.transmission > 0.0 {
        let kept = 1.0 - clamp(s.transmission, 0.0, 1.0);
        diffuse = diffuse * kept;
        back = back * kept;
    }
    let nv = s.n.dot(s.v);
    if nl <= 0.0 || nv <= 0.0 {
        return Response { diffuse, specular: back };
    }
    let nv = nv.max(1e-5);
    let h = (s.v + l).unit();
    let nh = clamp(s.n.dot(h), 0.0, 1.0);
    let vh = clamp(s.v.dot(h), 0.0, 1.0);
    let r = clamp(s.roughness, 0.04, 1.0);
    let f = schlick(s.f0, s.f90, vh);
    let a = clamp(s.anisotropy, -1.0, 1.0);
    let dv = if lobes & 4 != 0 && a != 0.0 {
        let (at, ab) = aniso_alpha(r, a);
        aniso_d(nh, s.tangent.dot(h), s.bitangent.dot(h), at, ab)
            * aniso_v(nl, nv, s.tangent.dot(l), s.bitangent.dot(l), s.tangent.dot(s.v), s.bitangent.dot(s.v), at, ab)
    } else {
        ggx_d(nh, r * r) * smith_v(nl, nv, r * r)
    };
    let specular = f * (dv * nl);
    // What the sheen and the coat leave of the lobes under them, and their own.
    let mut keep = 1.0;
    let mut sheen = V::all(0.0);
    if lobes & 2 != 0 && s.sheen.max3() > 0.0 {
        let rs = clamp(s.sheen_roughness, 0.07, 1.0);
        let tint = s.sheen.each(|c| c.max(0.0));
        sheen = tint * (sheen_d(nh, rs) * sheen_v(nl, nv) * nl);
        keep *= (1.0 - tint.max3() * table_albedo(table, nv, rs)).max(0.0);
    }
    let mut coat = V::all(0.0);
    if lobes & 1 != 0 && s.coat > 0.0 {
        let weight = clamp(s.coat, 0.0, 1.0);
        let ac = clamp(s.coat_roughness, 0.04, 1.0).powi(2);
        let nhc = clamp(s.coat_normal.dot(h), 0.0, 1.0);
        let nlc = clamp(s.coat_normal.dot(l), 0.0, 1.0);
        let nvc = s.coat_normal.dot(s.v).max(1e-5);
        let fc = weight * coat_fresnel(vh);
        coat = V::all(fc * ggx_d(nhc, ac) * smith_v(nlc, nvc, ac) * nlc);
        keep *= 1.0 - weight * coat_fresnel(nvc.min(1.0));
    }
    Response { diffuse: diffuse * keep, specular: (specular + back) * keep + sheen + coat }
}

/// A white furnace around `s` (whose normal is +z): radiance 1 from every
/// direction. Ink's responses are per unit radiance, so their integral over
/// the sphere is what the surface sends back. The diffuse part is a midpoint
/// rule of `lights` cells (theta by phi) over the light's sphere; the specular
/// lobes (narrow) are integrated over the half vector instead, in `halves`
/// cells with the cells bunched toward both its poles, the light being the
/// view's reflection about it and the solid angle 4 (v . h) of the half vector's.
pub fn furnace(s: &Surface, lobes: u32, table: &[f64], lights: (usize, usize), halves: (usize, usize)) -> (V, V) {
    assert_eq!(s.n, V(0.0, 0.0, 1.0));
    let ((theta, phi), (h_theta, h_phi)) = (lights, halves);
    let mut diffuse = V::all(0.0);
    for i in 0..theta {
        let t = (i as f64 + 0.5) / theta as f64 * PI;
        let weight = t.sin() * (PI / theta as f64) * (2.0 * PI / phi as f64);
        for j in 0..phi {
            let p = (j as f64 + 0.5) / phi as f64 * 2.0 * PI;
            let l = V(t.sin() * p.cos(), t.sin() * p.sin(), t.cos());
            diffuse = diffuse + light(s, l, lobes, table).diffuse * weight;
        }
    }
    let mut specular = V::all(0.0);
    for i in 0..h_theta {
        let u = (i as f64 + 0.5) / h_theta as f64;
        let t = PI / 4.0 * (1.0 - (PI * u).cos());
        let weight = t.sin() * (PI / 4.0 * PI * (PI * u).sin() / h_theta as f64) * (2.0 * PI / h_phi as f64);
        for j in 0..h_phi {
            let p = (j as f64 + 0.5) / h_phi as f64 * 2.0 * PI;
            let h = V(t.sin() * p.cos(), t.sin() * p.sin(), t.cos());
            let vh = s.v.dot(h);
            if vh <= 0.0 {
                continue;
            }
            let l = h * (2.0 * vh) - s.v;
            specular = specular + light(s, l, lobes, table).specular * (weight * 4.0 * vh);
        }
    }
    (diffuse * s.base * (1.0 - s.metallic), specular)
}
