// The lobes of a standard surface, as pure functions with no bindings: the
// microfacet base (GGX, height-correlated Smith, Schlick), a clear coat, sheen
// (Charlie with Neubelt's visibility), anisotropic GGX, a cheap subsurface
// (wrap and a back-lit term), the transmission and volume maths, and parallax
// occlusion mapping. `SURFACE_WGSL` (the sky, the frame's lights and the sun)
// and `SURFACE_TRANSMISSION_WGSL` build on them; a test can run any of them in
// a compute shader on its own.
//
// Conventions are Ink's `lighting.brdf`: a lobe's response is per unit
// radiance with the cosine in it. The frame's lights are scaled so that a
// diffuse surface is albedo times their color times the cosine, with no
// division by pi, so every lobe that comes out of `surface_light` is Ink's
// times pi (the specular ones here; the diffuse one is Lambert's 1 / pi * pi).
// The lobes below return Ink's own values; the multiplication by pi is
// `surface_light`'s.
//
// A shader builds a `StandardSurface` (`surface_default`, then sets what its
// material has), says which lobes exist in `lobes` (constants, so that the
// compiler folds the ones it leaves out, and a check of the parameter, so a
// weight of 0 costs nothing either) and calls `standard_surface`.
//
// Names: `surface_` for what a shader calls, `lobe_` for the maths under it,
// `parallax_` for POM; `SURFACE_` for flags.

const SURFACE_COAT: u32 = 1u;
const SURFACE_SHEEN: u32 = 2u;
const SURFACE_ANISOTROPY: u32 = 4u;
const SURFACE_TRANSMISSION: u32 = 8u;
const SURFACE_SUBSURFACE: u32 = 16u;
// The face-on reflectance comes from the ior, the specular factor and its color
// (`surface_apply_block`); without it f0 is the dielectric 0.04 of ior 1.5.
const SURFACE_SPECULAR: u32 = 32u;

// What the frame's lights bring to a glossy surface, in two parts a material
// scales by its own: `diffuse` is for albedo (times 1 - metallic) to multiply,
// and `specular` is everything that does not take the albedo (the base lobe's
// highlight, already Fresnel-weighted, the coat, the sheen), to add as it is.
struct LightResponse {
  diffuse: vec3f,
  specular: vec3f,
};

// A surface seen from `view`: what `standard_surface` and `standard_sun` take.
// Build it with `surface_default` and set the fields of the lobes it has.
struct StandardSurface {
  // Where it is, which way it faces (unit) and the unit vector toward the eye.
  world: vec3f,
  normal: vec3f,
  view: vec3f,
  // The fragment's @builtin(position) and its view-space z (negative ahead of
  // the eye), which find its cluster of lights.
  position: vec4f,
  view_z: f32,
  // The base: albedo (or a metal's color), metallic and perceptual roughness (0
  // to 1), ambient occlusion (1 open to 0 closed), and the reflectance face on
  // (`f0`) and at grazing angles (`f90`).
  base_color: vec3f,
  metallic: f32,
  roughness: f32,
  occlusion: f32,
  f0: vec3f,
  f90: f32,
  // `SURFACE_*` flags: which lobes exist.
  lobes: u32,
  // Clear coat: its weight, roughness and normal (the mesh's own, with no
  // normal map, unless the material says).
  coat: f32,
  coat_roughness: f32,
  coat_normal: vec3f,
  // Sheen: color and roughness.
  sheen_color: vec3f,
  sheen_roughness: f32,
  // Anisotropy in -1 to 1 along `tangent`; with `bitangent`, an orthonormal
  // frame with `normal` (`surface_frame`).
  anisotropy: f32,
  tangent: vec3f,
  bitangent: vec3f,
  // Transmission: its weight, the index of refraction, the thickness in world
  // units (0 for a thin wall), the volume's color and the distance in which
  // it falls to that color (0 for never), and the light from behind
  // (`surface_transmit`, in `SURFACE_TRANSMISSION_WGSL`).
  transmission: f32,
  ior: f32,
  thickness: f32,
  attenuation_color: vec3f,
  attenuation_distance: f32,
  behind: vec3f,
  // Subsurface: how much light goes through (0 to 1), its color, and how
  // thin the surface is (0 a leaf, 1 opaque).
  subsurface: f32,
  subsurface_color: vec3f,
  thin: f32,
};

// A surface of albedo (or metal color) `base_color`, `metallic` and
// `roughness`: the base lobe alone, seen at the fragment `position` (its
// @builtin(position)) whose view-space z is `view_z`.
fn surface_default(world: vec3f, normal: vec3f, view: vec3f, base_color: vec3f, metallic: f32, roughness: f32, position: vec4f, view_z: f32) -> StandardSurface {
  var s: StandardSurface;
  s.world = world;
  s.normal = normal;
  s.view = view;
  s.position = position;
  s.view_z = view_z;
  s.base_color = base_color;
  s.metallic = metallic;
  s.roughness = roughness;
  s.occlusion = 1.0;
  s.f0 = mix(vec3f(0.04), base_color, clamp(metallic, 0.0, 1.0));
  s.f90 = 1.0;
  s.coat_normal = normal;
  s.sheen_roughness = 0.5;
  s.ior = 1.5;
  s.attenuation_color = vec3f(1.0);
  s.thin = 1.0;
  return s;
}

// Sets the lobes in `lobes` (`SURFACE_*` flags, constants the shader chooses)
// from a material's `surface` block, the draw uniform's last five `vec4f`s:
//   [0] clearcoat, clearcoat roughness, sheen roughness, anisotropy
//   [1] sheen color, anisotropy rotation (radians; for `surface_frame`)
//   [2] transmission, ior, thickness (model units), attenuation distance
//   [3] attenuation color, specular
//   [4] specular color, emissive strength
// `model_scale` is what the thickness is scaled by to reach world units
// (`length(draw.model[0].xyz)`). With `SURFACE_SPECULAR` or
// `SURFACE_TRANSMISSION` the reflectance (f0 and f90) follows the ior, the
// specular factor and its color; otherwise it stays the 0.04 of `surface_default`.
fn surface_apply_block(s: StandardSurface, block: array<vec4f, 5>, lobes: u32, model_scale: f32) -> StandardSurface {
  var r = s;
  r.lobes = lobes;
  if ((lobes & SURFACE_COAT) != 0u) {
    r.coat = block[0].x;
    r.coat_roughness = block[0].y;
  }
  if ((lobes & SURFACE_SHEEN) != 0u) {
    r.sheen_roughness = block[0].z;
    r.sheen_color = block[1].xyz;
  }
  if ((lobes & SURFACE_ANISOTROPY) != 0u) {
    r.anisotropy = block[0].w;
  }
  if ((lobes & (SURFACE_SPECULAR | SURFACE_TRANSMISSION)) != 0u) {
    r.ior = block[2].y;
    let specular = block[3].w;
    let dielectric = min(lobe_ior_f0(r.ior) * block[4].xyz, vec3f(1.0)) * specular;
    r.f0 = mix(dielectric, s.base_color, clamp(s.metallic, 0.0, 1.0));
    r.f90 = mix(specular, 1.0, clamp(s.metallic, 0.0, 1.0));
  }
  if ((lobes & SURFACE_TRANSMISSION) != 0u) {
    r.transmission = block[2].x;
    r.thickness = block[2].z * model_scale;
    r.attenuation_distance = block[2].w;
    r.attenuation_color = block[3].xyz;
  }
  return r;
}

// A mesh's tangent stream (xyz, w the handedness), or where it is zero (a mesh
// with none) `derived`, from `surface_derived_tangent`.
fn surface_tangent(stream: vec4f, derived: vec4f) -> vec4f {
  return select(derived, stream, dot(stream.xyz, stream.xyz) > 0.000001);
}

// ---- The base lobe.

// GGX's distribution for the squared alpha `a2` (alpha = roughness squared).
fn lobe_ggx_distribution(nh: f32, a2: f32) -> f32 {
  let denominator = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / (3.14159265359 * denominator * denominator);
}

// The height-correlated Smith visibility, which has the lobe's 1 / (4 nl nv).
fn lobe_smith_visibility(nl: f32, nv: f32, a2: f32) -> f32 {
  return 0.5 / max(0.000001, nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2));
}

// Schlick's Fresnel, from `f0` face on to `f90` at grazing angles.
fn lobe_schlick(f0: vec3f, f90: f32, vh: f32) -> vec3f {
  let fade = 1.0 - vh;
  return f0 + (vec3f(f90) - f0) * (fade * fade * fade * fade * fade);
}

// The dielectric's face-on reflectance from its index of refraction (at least
// 1), as two products and a division, so that an ior of 1.5 is the f32 nearest 0.04.
fn lobe_ior_f0(ior: f32) -> f32 {
  let i = max(ior, 1.0);
  return (i - 1.0) * (i - 1.0) / ((i + 1.0) * (i + 1.0));
}

// ---- Clear coat: a second, dielectric GGX layer over everything, with the
// base lobe's own Smith visibility at the coat's alpha and normal.

// Its Fresnel at `vh` (cosine between the view and the half vector), f0 0.04.
fn lobe_coat_fresnel(vh: f32) -> f32 {
  let fade = 1.0 - vh;
  return 0.04 + 0.96 * (fade * fade * fade * fade * fade);
}

// ---- Sheen: Charlie's distribution and Neubelt's visibility, for cloth.

// The sheen roughness a shader gives, held where the lobe stays well behaved.
fn lobe_sheen_roughness(roughness: f32) -> f32 {
  return clamp(roughness, 0.07, 1.0);
}

fn lobe_sheen_distribution(nh: f32, roughness: f32) -> f32 {
  let alpha = roughness * roughness;
  let sin2 = max(1.0 - nh * nh, 0.0);
  return (2.0 + 1.0 / alpha) * pow(sin2, 0.5 / alpha) / 6.28318530718;
}

fn lobe_sheen_visibility(nl: f32, nv: f32) -> f32 {
  return 1.0 / (4.0 * (nl + nv - nl * nv));
}

// The sheen lobe's directional albedo for a sheen color of 1, capped at 1:
// what the lobe takes from the light under it (`nv` the cosine of the view,
// `roughness` the sheen's, through `lobe_sheen_roughness`). It is the integral
// of the lobe's D * V * cos over the light's hemisphere, past 1 for a grazing
// view of a sharp sheen. A table of 32 columns by 16 rows, entry `32 row +
// column`, bilinear: the columns evenly spaced in the square root of nv and the
// rows in the square root of (roughness - 0.07) / 0.93. It is Ink's table
// (tools/native/src/surface.rs writes it; its sum is 220.61942091), and
// pen-render's tests integrate it again and compare it to this text.
var<private> lobe_sheen_table = array<f32, 512>(
  // SHEEN TABLE BEGIN
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
  1.00000000, 0.96609209, 0.66635188, 0.43914545, 0.27494452, 0.16243548, 0.08983930, 0.04608312,
  0.02168285, 0.00923651, 0.00350696, 0.00116469, 0.00033065, 0.00007799, 0.00001475, 0.00000213,
  0.00000022, 0.00000002, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
  1.00000000, 0.99849743, 0.70382454, 0.47581233, 0.30694153, 0.18781249, 0.10823914, 0.05826881,
  0.02901557, 0.01321075, 0.00542376, 0.00197461, 0.00062460, 0.00016736, 0.00003679, 0.00000637,
  0.00000082, 0.00000007, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
  1.00000000, 1.00000000, 0.79120667, 0.56797785, 0.39292684, 0.26082237, 0.16527702, 0.09938102,
  0.05630328, 0.02980166, 0.01458949, 0.00652583, 0.00262752, 0.00093481, 0.00028706, 0.00007381,
  0.00001526, 0.00000240, 0.00000026, 0.00000002, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
  1.00000000, 1.00000000, 0.87482038, 0.67105278, 0.50103574, 0.36310585, 0.25457281, 0.17199340,
  0.11146346, 0.06891247, 0.04037954, 0.02224737, 0.01141374, 0.00538712, 0.00230358, 0.00087483,
  0.00028734, 0.00007868, 0.00001703, 0.00000268, 0.00000027, 0.00000001, 0.00000000, 0.00000000,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
  1.00000000, 1.00000000, 0.91595046, 0.74369491, 0.59229018, 0.46187804, 0.35194184, 0.26139767,
  0.18869614, 0.13193549, 0.08898340, 0.05760236, 0.03557141, 0.02079630, 0.01139987, 0.00578623,
  0.00267422, 0.00109958, 0.00038892, 0.00011233, 0.00002426, 0.00000330, 0.00000018, 0.00000000,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
  1.00000000, 1.00000000, 0.90920665, 0.77150637, 0.64560985, 0.53219713, 0.43159173, 0.34378996,
  0.26849076, 0.20512874, 0.15291174, 0.11086346, 0.07787063, 0.05273348, 0.03421768, 0.02110542,
  0.01224284, 0.00658119, 0.00320904, 0.00137332, 0.00048788, 0.00012911, 0.00001932, 0.00000026,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
  1.00000000, 0.98102820, 0.86882158, 0.76205773, 0.66160758, 0.56814115, 0.48214657, 0.40394357,
  0.33369393, 0.27141090, 0.21696873, 0.17011324, 0.13047412, 0.09757910, 0.07087035, 0.04972273,
  0.03346384, 0.02139528, 0.01281466, 0.00703751, 0.00341839, 0.00137031, 0.00038154, 0.00002884,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
  0.98277184, 0.89634067, 0.81178558, 0.72985974, 0.65118486, 0.57627302, 0.50554118, 0.43932106,
  0.37786569, 0.32135404, 0.26989436, 0.22352695, 0.18222671, 0.14590578, 0.11441668, 0.08755592,
  0.06506831, 0.04665211, 0.03196491, 0.02063031, 0.01224544, 0.00638909, 0.00263055, 0.00053878,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 0.94502083,
  0.87992735, 0.81491643, 0.75060375, 0.68749593, 0.62601584, 0.56651993, 0.50931011, 0.45464221,
  0.40273167, 0.35375757, 0.30786537, 0.26516875, 0.22575081, 0.18966490, 0.15693516, 0.12755696,
  0.10149724, 0.07869493, 0.05906142, 0.04248124, 0.02881280, 0.01788945, 0.00952070, 0.00349374,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 0.98340343, 0.93834126, 0.89090605, 0.84200093,
  0.79230903, 0.74236191, 0.69258323, 0.64331760, 0.59485042, 0.54742171, 0.50123597, 0.45646924,
  0.41327429, 0.37178437, 0.33211591, 0.29437057, 0.25863659, 0.22498985, 0.19349450, 0.16420342,
  0.13715847, 0.11239058, 0.08991975, 0.06975496, 0.05189404, 0.03632342, 0.02301797, 0.01194067,
  0.96083395, 0.95130653, 0.92968785, 0.90165811, 0.86940935, 0.83430004, 0.79726916, 0.75899487,
  0.71998297, 0.68062055, 0.64121042, 0.60199396, 0.56316703, 0.52489117, 0.48730163, 0.45051341,
  0.41462561, 0.37972478, 0.34588747, 0.31318216, 0.28167071, 0.25140954, 0.22245055, 0.19484174,
  0.16862782, 0.14385056, 0.12054914, 0.09876037, 0.07851887, 0.05985716, 0.04280577, 0.02739325,
  0.85583864, 0.84815146, 0.83071649, 0.80812013, 0.78212586, 0.75382052, 0.72394998, 0.69304750,
  0.66150557, 0.62961961, 0.59761598, 0.56567085, 0.53392314, 0.50248383, 0.47144269, 0.44087321,
  0.41083638, 0.38138354, 0.35255851, 0.32439934, 0.29693967, 0.27020975, 0.24423731, 0.21904830,
  0.19466745, 0.17111876, 0.14842596, 0.12661285, 0.10570369, 0.08572354, 0.06669855, 0.04865633,
  0.77679509, 0.77036767, 0.75580614, 0.73695585, 0.71529142, 0.69171744, 0.66685106, 0.64112984,
  0.61487218, 0.58831418, 0.56163327, 0.53496417, 0.50840985, 0.48204943, 0.45594386, 0.43014016,
  0.40467461, 0.37957512, 0.35486313, 0.33055503, 0.30666328, 0.28319735, 0.26016440, 0.23756984,
  0.21541786, 0.19371178, 0.17245440, 0.15164825, 0.13129592, 0.11140024, 0.09196451, 0.07299265,
  0.71697164, 0.71142504, 0.69887748, 0.68266018, 0.66404822, 0.64382094, 0.62250753, 0.60048076,
  0.57800943, 0.55529039, 0.53246921, 0.50965395, 0.48692483, 0.46434101, 0.44194562, 0.41976940,
  0.39783347, 0.37615137, 0.35473071, 0.33357435, 0.31268135, 0.29204771, 0.27166689, 0.25153028,
  0.23162749, 0.21194660, 0.19247427, 0.17319591, 0.15409562, 0.13515622, 0.11635915, 0.09768477,
  0.67144043, 0.66652196, 0.65541313, 0.64108069, 0.62465867, 0.60683840, 0.58808760, 0.56873412,
  0.54901274, 0.52909382, 0.50910171, 0.48912709, 0.46923564, 0.44947405, 0.42987455, 0.41045811,
  0.39123696, 0.37221637, 0.35339603, 0.33477118, 0.31633332, 0.29807086, 0.27996951, 0.26201260,
  0.24418128, 0.22645456, 0.20880934, 0.19122031, 0.17365971, 0.15609709, 0.13849886, 0.12083050,
  0.63657542, 0.63211312, 0.62205054, 0.60909083, 0.59426633, 0.57820518, 0.56133123, 0.54394050,
  0.52624394, 0.50839359, 0.49049941, 0.47264056, 0.45487327, 0.43723644, 0.41975565, 0.40244617,
  0.38531519, 0.36836348, 0.35158661, 0.33497592, 0.31851919, 0.30220115, 0.28600380, 0.26990667,
  0.25388683, 0.23791896, 0.22197516, 0.20602478, 0.19003399, 0.17396535, 0.15777709, 0.14142957,
  // SHEEN TABLE END
);

fn lobe_sheen_albedo(nv: f32, roughness: f32) -> f32 {
  let x = sqrt(clamp(nv, 0.0, 1.0)) * 31.0;
  let y = sqrt(clamp((lobe_sheen_roughness(roughness) - 0.07) / 0.93, 0.0, 1.0)) * 15.0;
  let x0 = min(u32(floor(x)), 30u);
  let y0 = min(u32(floor(y)), 14u);
  let fx = x - f32(x0);
  let fy = y - f32(y0);
  let a = mix(lobe_sheen_table[y0 * 32u + x0], lobe_sheen_table[y0 * 32u + x0 + 1u], fx);
  let b = mix(lobe_sheen_table[(y0 + 1u) * 32u + x0], lobe_sheen_table[(y0 + 1u) * 32u + x0 + 1u], fx);
  return mix(a, b, fy);
}

// ---- Anisotropic GGX: the lobe stretched along the tangent (a > 0 widens it
// across the grain, a < 0 along it, as glTF has it).

// The widths along the tangent and the bitangent for perceptual `roughness`
// and anisotropy `a`: x along the tangent, y along the bitangent. They are
// both alpha (roughness squared) when `a` is 0.
fn lobe_aniso_alpha(roughness: f32, anisotropy: f32) -> vec2f {
  let rough = clamp(roughness, 0.04, 1.0);
  let alpha = rough * rough;
  return vec2f(mix(alpha, 1.0, anisotropy * anisotropy), alpha);
}

// th and bh: the half vector's parts along the tangent and the bitangent.
fn lobe_aniso_distribution(nh: f32, th: f32, bh: f32, alpha: vec2f) -> f32 {
  let ratio = vec2f(th, bh) / alpha;
  let d = dot(ratio, ratio) + nh * nh;
  return 1.0 / (3.14159265359 * alpha.x * alpha.y * d * d);
}

// tl, bl, tv, bv: the light's and the view's parts along the tangent and bitangent.
fn lobe_aniso_visibility(nl: f32, nv: f32, tl: f32, bl: f32, tv: f32, bv: f32, alpha: vec2f) -> f32 {
  let shadowing = nv * length(vec3f(alpha.x * tl, alpha.y * bl, nl)) + nl * length(vec3f(alpha.x * tv, alpha.y * bv, nv));
  return 0.5 / max(0.000001, shadowing);
}

// The normal a reflection of an anisotropic surface looks up the sky by:
// bent toward the view across the grain, by as much as the anisotropy and the
// roughness (the reflection stretches along the grain, Filament's trick).
fn lobe_bent_normal(normal: vec3f, tangent: vec3f, bitangent: vec3f, view: vec3f, roughness: f32, anisotropy: f32) -> vec3f {
  let grain = select(tangent, bitangent, anisotropy >= 0.0);
  let across = cross(grain, view);
  let bent = cross(across, grain);
  let mixed = mix(normal, bent, abs(anisotropy) * clamp(5.0 * roughness, 0.0, 1.0));
  if (dot(mixed, mixed) < 0.000001) {
    return normal;
  }
  return normalize(mixed);
}

// Sets the anisotropy's frame: `tangent` (any vector not along the normal; a
// mesh's tangent stream, or `surface_derived_tangent`) made orthogonal to the
// normal and turned about it by `rotation` (radians), and the bitangent across
// it, `handedness` (+1 or -1) giving its side.
fn surface_frame(s: StandardSurface, tangent: vec3f, handedness: f32, rotation: f32) -> StandardSurface {
  var r = s;
  let t0 = normalize(tangent - s.normal * dot(s.normal, tangent));
  r.tangent = t0 * cos(rotation) + cross(s.normal, t0) * sin(rotation);
  r.bitangent = cross(s.normal, r.tangent) * handedness;
  return r;
}

// ---- Cheap subsurface: wrapped lighting and a back-lit term.

fn lobe_wrap(x: f32, w: f32) -> f32 {
  return clamp((x + w) / ((1.0 + w) * (1.0 + w)), 0.0, 1.0);
}

// The diffuse lobe's response for `amount` of subsurface of `color`, for the
// unclamped cosine `signed` to the light: 1 / pi of Ink's, so that
// Lambert is `max(signed, 0)`.
fn lobe_subsurface_diffuse(signed: f32, amount: f32, color: vec3f) -> vec3f {
  let nl = max(signed, 0.0);
  return vec3f((1.0 - amount) * nl) + amount * lobe_wrap(signed, 0.5) * clamp(color + vec3f(nl), vec3f(0.0), vec3f(1.0));
}

// What leaks through to a viewer on the other side: the light `way` (unit,
// toward the light) bent by the normal, seen along `view` (toward the eye).
// Ink's term is times base / pi * 2; this has the base and pi left out.
fn lobe_subsurface_back(way: vec3f, normal: vec3f, view: vec3f, amount: f32, color: vec3f, thin: f32) -> vec3f {
  let bent = -normalize(way + 0.2 * normal);
  let through = clamp(dot(view, bent), 0.0, 1.0);
  let squared = through * through;
  return color * (amount * (1.0 - thin) * squared * squared * 2.0);
}

// ---- Transmission and volume (KHR_materials_transmission, _volume, _ior).

// The direction a ray arriving along -`view` takes entering a medium, as
// `refract` has it: `eta` is the index of the medium it comes from over the one
// it enters (1 / ior from the air into glass). The zero vector at total
// internal reflection, which only happens leaving a denser medium (eta over 1).
fn lobe_refract(view: vec3f, normal: vec3f, eta: f32) -> vec3f {
  return refract(-view, normal, eta);
}

// How much of its color a volume lets through over `thickness` (world units)
// when it lets `color` through over `distance`: Beer-Lambert, per channel. A
// distance of 0 is a medium that absorbs nothing, and so is no thickness.
fn lobe_volume_attenuation(color: vec3f, thickness: f32, distance: f32) -> vec3f {
  if (thickness <= 0.0 || distance <= 0.0) {
    return vec3f(1.0);
  }
  return pow(color, vec3f(thickness / distance));
}

// The mip level of the picture behind to read through a surface of
// `roughness` and index `ior` (as the glTF sample viewer picks it): `size_log2` is the base-2
// logarithm of the picture's width, `last` its last level.
fn lobe_transmission_level(roughness: f32, ior: f32, size_log2: f32, last: f32) -> f32 {
  return min(size_log2 * clamp(roughness * clamp(2.0 * ior - 2.0, 0.0, 1.0), 0.0, 1.0), last);
}

// ---- Parallax occlusion mapping: a heightfield the surface's uv are marched
// through. The heights are the red of the material's layer texture (binding 4
// of group 1; pass it with its sampler): 1 is the surface, 0 `scale` (in uv
// units) below it.

struct Parallax {
  // Where every texture read of the material goes instead of the mesh's uv.
  uv: vec2f,
  // The height of the surface there (1 at the top).
  height: f32,
  // The light's visibility: 1 lit, 0 in a groove (1 when not asked for).
  shadow: f32,
};

// How many layers the march takes along a direction whose tangent-space z is
// `z` (|z| of 1 straight down, 0 across): floor(32 - 24 |z|), 8 to 32.
fn parallax_layers(z: f32) -> f32 {
  return floor(32.0 - 24.0 * min(abs(z), 1.0));
}

// The view `v` (world, toward the eye) in the tangent space of `tangent`,
// `bitangent` and `normal` (unit and orthogonal; the uv's axes): the transpose
// of the frame times `v`, normalized.
fn parallax_tangent_space(tangent: vec3f, bitangent: vec3f, normal: vec3f, v: vec3f) -> vec3f {
  return normalize(vec3f(dot(tangent, v), dot(bitangent, v), dot(normal, v)));
}

fn parallax_height(layers: texture_2d<f32>, layer_sampler: sampler, uv: vec2f, lod: f32) -> f32 {
  return clamp(textureSampleLevel(layers, layer_sampler, uv, lod).r, 0.0, 1.0);
}

// How much of the light reaches the surface at `uv` where its height is
// `height`, along `light` (unit, tangent space, toward the light): the ray
// rises from the surface to the top in as many layers as the march takes,
// sliding toward the light by `light.xy / z * scale` of the height it climbs,
// and the most any sample stands above it, times 8, darkens the light. 0 for a
// light below the surface.
fn parallax_shadow(layers: texture_2d<f32>, layer_sampler: sampler, uv: vec2f, height: f32, light: vec3f, scale: f32, lod: f32) -> f32 {
  if (light.z <= 0.0) {
    return 0.0;
  }
  let count = parallax_layers(light.z);
  let rise = 1.0 - height;
  let slide = light.xy / max(light.z, 0.0001) * (scale * rise);
  var above = 0.0;
  for (var k = 1.0; k <= count; k += 1.0) {
    let along = k / count;
    above = max(above, parallax_height(layers, layer_sampler, uv + slide * along, lod) - (height + rise * along));
  }
  return 1.0 - clamp(8.0 * above, 0.0, 1.0);
}

// Marches the view `view` (unit, tangent space, toward the eye, z up) down
// through the heightfield from `uv` in layers of equal depth, to the first
// layer where the surface is at or above the ray, and puts the hit where the
// line between that sample and the one before it crosses. `scale` is the height
// range in uv units; `lod` the layer texture's level (0 for the base). A view
// at or below the surface, or a hit at the first layer, leaves `uv` alone. With
// `with_shadow`, `light` (tangent space, toward the light) gives `shadow` too.
fn parallax_uv(layers: texture_2d<f32>, layer_sampler: sampler, uv: vec2f, view: vec3f, light: vec3f, scale: f32, lod: f32, with_shadow: bool) -> Parallax {
  var result = Parallax(uv, parallax_height(layers, layer_sampler, uv, lod), 1.0);
  if (view.z > 0.0) {
    let count = parallax_layers(view.z);
    let step = (-view.xy / max(view.z, 0.0001)) * scale / count;
    var before = uv;
    var before_gap = 0.0;
    var at = uv;
    for (var k = 0.0; k <= count; k += 1.0) {
      at = uv + step * k;
      let gap = parallax_height(layers, layer_sampler, at, lod) - (1.0 - k / count);
      if (gap >= 0.0 || k >= count) {
        if (k > 0.0) {
          result.uv = at + (before - at) * (gap / (gap - before_gap));
        }
        break;
      }
      before = at;
      before_gap = gap;
    }
    result.height = parallax_height(layers, layer_sampler, result.uv, lod);
  }
  if (with_shadow) {
    result.shadow = parallax_shadow(layers, layer_sampler, result.uv, result.height, light, scale, lod);
  }
  return result;
}

// ---- One light on a surface.

// The response of `s` to one light shining from the unit direction `way` (toward
// the light) with `radiance` (its color times its strength and its visibility,
// scaled as the frame's lights are). `diffuse` and `specular` are what
// `LightResponse` says. With only the base lobe, `diffuse` is `radiance` times
// the cosine and `specular` the GGX highlight, as the frame's lights have
// always been summed. With lobes, the diffuse and specular lobes (and the
// subsurface's back light, in `specular` for it takes the albedo but not
// 1 - metallic) are scaled by what the sheen and the coat leave of them, and
// the sheen and the coat add theirs: `(diffuse * (1 - transmission) + specular) *
// sheen_keep * coat_keep + sheen + coat`. The sheen is not dimmed by the coat.
fn surface_light(s: StandardSurface, way: vec3f, radiance: vec3f) -> LightResponse {
  var response = LightResponse(vec3f(0.0), vec3f(0.0));
  let signed = dot(s.normal, way);
  let nl = max(signed, 0.0);
  let wrapping = (s.lobes & SURFACE_SUBSURFACE) != 0u && s.subsurface > 0.0;
  if (nl <= 0.0 && !wrapping) {
    return response;
  }
  var diffuse = vec3f(nl);
  var back = vec3f(0.0);
  if (wrapping) {
    let amount = clamp(s.subsurface, 0.0, 1.0);
    diffuse = lobe_subsurface_diffuse(signed, amount, s.subsurface_color);
    back = s.base_color * lobe_subsurface_back(way, s.normal, s.view, amount, s.subsurface_color, s.thin);
  }
  if ((s.lobes & SURFACE_TRANSMISSION) != 0u && s.transmission > 0.0) {
    let kept = 1.0 - clamp(s.transmission, 0.0, 1.0);
    diffuse *= kept;
    back *= kept;
  }
  let facing = dot(s.normal, s.view);
  if (nl <= 0.0 || facing <= 0.0) {
    response.diffuse = radiance * diffuse;
    response.specular = radiance * back;
    return response;
  }
  let nv = max(facing, 0.00001);
  let half_way = normalize(s.view + way);
  let nh = clamp(dot(s.normal, half_way), 0.0, 1.0);
  let vh = clamp(dot(s.view, half_way), 0.0, 1.0);
  var distribution = 0.0;
  var visibility = 0.0;
  let anisotropy = clamp(s.anisotropy, -1.0, 1.0);
  if ((s.lobes & SURFACE_ANISOTROPY) != 0u && anisotropy != 0.0) {
    let alpha = lobe_aniso_alpha(s.roughness, anisotropy);
    distribution = lobe_aniso_distribution(nh, dot(s.tangent, half_way), dot(s.bitangent, half_way), alpha);
    visibility = lobe_aniso_visibility(nl, nv, dot(s.tangent, way), dot(s.bitangent, way), dot(s.tangent, s.view), dot(s.bitangent, s.view), alpha);
  } else {
    let rough = clamp(s.roughness, 0.04, 1.0);
    let a2 = rough * rough * rough * rough;
    distribution = lobe_ggx_distribution(nh, a2);
    visibility = lobe_smith_visibility(nl, nv, a2);
  }
  let fresnel = lobe_schlick(s.f0, s.f90, vh);
  var specular = radiance * (fresnel * (distribution * visibility * nl * 3.14159265359));
  // The layers over the base lobes take their share of the light, and add theirs.
  var layers = 1.0;
  var added = vec3f(0.0);
  if ((s.lobes & SURFACE_SHEEN) != 0u && dot(s.sheen_color, s.sheen_color) > 0.0) {
    let tint = max(s.sheen_color, vec3f(0.0));
    let rough = lobe_sheen_roughness(s.sheen_roughness);
    let lobe = lobe_sheen_distribution(nh, rough) * lobe_sheen_visibility(nl, nv) * nl;
    added = radiance * tint * (lobe * 3.14159265359);
    layers = max(0.0, 1.0 - max(tint.x, max(tint.y, tint.z)) * lobe_sheen_albedo(nv, rough));
  }
  var coat = vec3f(0.0);
  if ((s.lobes & SURFACE_COAT) != 0u && s.coat > 0.0) {
    let weight = clamp(s.coat, 0.0, 1.0);
    let rc = clamp(s.coat_roughness, 0.04, 1.0);
    let a2 = rc * rc * rc * rc;
    let nhc = clamp(dot(s.coat_normal, half_way), 0.0, 1.0);
    let nlc = clamp(dot(s.coat_normal, way), 0.0, 1.0);
    let nvc = max(dot(s.coat_normal, s.view), 0.00001);
    // The coat's own lobe has its Fresnel at the half vector; what it leaves of
    // the lobes under it is the same for every light, its Fresnel at the view.
    let fc = weight * lobe_coat_fresnel(vh);
    layers *= 1.0 - weight * lobe_coat_fresnel(min(nvc, 1.0));
    coat = radiance * (fc * lobe_ggx_distribution(nhc, a2) * lobe_smith_visibility(nlc, nvc, a2) * nlc * 3.14159265359);
  }
  response.diffuse = radiance * diffuse * layers;
  response.specular = (specular + radiance * back) * layers + added + coat;
  return response;
}

// The light of the sun (`sun_color` times its visibility, scaled as the
// frame's lights are; `sun_dir` the way its light travels, as the scene's
// `light_direction`) on `s`: the same lobes the frame's lights get, diffuse
// times the albedo and (1 - metallic), and the specular ones as they are.
// Lambert alone is `base_color * sun_color * visibility * max(dot(normal, -sun_dir), 0)`.
fn standard_sun(s: StandardSurface, sun_dir: vec3f, sun_color: vec3f, visibility: f32) -> vec3f {
  let response = surface_light(s, -normalize(sun_dir), sun_color * visibility);
  return response.diffuse * s.base_color * (1.0 - clamp(s.metallic, 0.0, 1.0)) + response.specular;
}

// Screen-door 4x4 Bayer dither predicate for smooth LOD crossfading without
// transparency sorting (returns true when the fragment should be discarded).
// Pass `fade` in (0, 1] for the outgoing LOD and `-fade` in [-1, 0) for the
// incoming LOD so both halves cover complementary pixel masks with zero holes
// and zero double-drawn pixels. `fade >= 1.0` never discards; `fade == 0.0` or
// `fade <= -1.0` always discards.
fn surface_lod_dither(pixel: vec2f, fade: f32) -> bool {
  if (fade >= 1.0) {
    return false;
  }
  if (fade == 0.0 || fade <= -1.0) {
    return true;
  }
  let p = vec2u(floor(max(pixel, vec2f(0.0)))) & vec2u(3u, 3u);
  let idx = p.y * 4u + p.x;
  let bayer = array<f32, 16>(
     0.0,  8.0,  2.0, 10.0,
    12.0,  4.0, 14.0,  6.0,
     3.0, 11.0,  1.0,  9.0,
    15.0,  7.0, 13.0,  5.0,
  );
  let threshold = (bayer[idx] + 0.5) / 16.0;
  if (fade > 0.0) {
    return threshold >= fade;
  }
  return threshold < (-fade);
}

// Applies glTF `KHR_texture_transform` (scale -> rotation -> translation) to `uv`.
// `offset_scale` holds `(offset.x, offset.y, scale.x, scale.y)` and `rot_cs` holds `(cos(r), sin(r))`.
fn surface_transform_uv(uv: vec2f, offset_scale: vec4f, rot_cs: vec2f) -> vec2f {
  let s = uv * offset_scale.zw;
  let r = vec2f(s.x * rot_cs.x + s.y * rot_cs.y, -s.x * rot_cs.y + s.y * rot_cs.x);
  return r + offset_scale.xy;
}

// Applies a linear ORM texture sample (`R = occlusion`, `G = roughness`, `B = metallic`)
// onto `s` when `enabled > 0.5`.
fn surface_apply_orm(s: StandardSurface, orm: vec4f, enabled: f32) -> StandardSurface {
  var out = s;
  if (enabled > 0.5) {
    out.occlusion *= clamp(orm.r, 0.0, 1.0);
    out.roughness = clamp(out.roughness * orm.g, 0.04, 1.0);
    out.metallic = clamp(out.metallic * orm.b, 0.0, 1.0);
  }
  return out;
}
