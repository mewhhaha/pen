// The frame's point and spot lights, sorted into clusters of the view: the
// picture cut into tiles across and down, and the depth ahead into slices,
// closer together near the eye. A fragment reads only the lights whose reach
// overlaps its cluster. Material shaders that light with them declare these
// bindings (group 2 of every mesh pipeline) and call `clustered_light`.
struct ClusterGrid {
  // x, y, z: tiles across, down and slices deep; w: the frame's lights.
  counts: vec4u,
  // xy: tiles per pixel across and down; z, w: a depth's slice is
  // log(depth) * z + w, or depth * z + w when options.x is 1.
  scale: vec4f,
  options: vec4f,
};
struct ClusterLight {
  // xyz: where it is; w: how far it reaches.
  position: vec4f,
  // rgb: its color times its intensity; w: the cosine of a spot's cone, -1 for a point light.
  color: vec4f,
  // xyz: the way a spot shines; w: the cosine within which it is at full strength.
  direction: vec4f,
  // x: the first layer of its shadow map in `local_shadows`, -1 for none (a
  // point light's six faces follow it); y: 1 for a cube, 0 for a spot's one
  // face; z: a face texel's width at unit depth, in world units.
  shadow: vec4f,
};
@group(2) @binding(0) var<uniform> cluster_grid: ClusterGrid;
@group(2) @binding(1) var<storage, read> cluster_lights: array<ClusterLight>;
@group(2) @binding(2) var<storage, read> cluster_ranges: array<vec2u>;
@group(2) @binding(3) var<storage, read> cluster_indices: array<u32>;
// The shadow maps of the lights that cast (a layer a face), their comparison
// sampler and each layer's world-to-clip matrix.
@group(2) @binding(4) var local_shadows: texture_depth_2d_array;
@group(2) @binding(5) var local_shadow_sampler: sampler_comparison;
@group(2) @binding(6) var<storage, read> local_shadow_faces: array<mat4x4f>;
// Packed LTC area lights (up to 64 lights * 5 vec4f at offset 0) followed by
// 64x64 LTC inverse-M matrix LUT (offset 320) and 64x64 LTC amplitude LUT (offset 4416).
@group(2) @binding(7) var<storage, read> area_light_data: array<vec4f>;

// Eight points of a Poisson disk of radius 1.
const local_shadow_taps = array<vec2f, 8>(
  vec2f(-0.7071, -0.7071), vec2f(0.0, -0.1), vec2f(0.7071, -0.7071), vec2f(-0.4, 0.2),
  vec2f(0.45, 0.25), vec2f(-0.85, 0.5), vec2f(0.1, 0.9), vec2f(0.9, 0.35));
// How far a surface is moved along its normal (texels, at a right angle to
// the light), and toward the light (texels: a base, and more with the slope
// of the surface to the light), before it reads its map; and the filter's reach.
const local_normal_offset: f32 = 1.5;
const local_bias_base: f32 = 1.5;
const local_bias_slope: f32 = 0.75;
const local_filter_radius: f32 = 1.5;

// How much of `light` reaches `world`, a surface facing `normal`, where `way`
// is the unit vector toward the light and `distance` how far it is: 1 where
// nothing the light's map holds lies between, 0 behind a caster, filtered over
// a small disk turned by a number from the surface's position. A point
// light's face is the one its direction from the light leans on most; each
// face's window reaches a few texels past it, so the filter never leaves it.
fn local_shadow(light: ClusterLight, world: vec3f, normal: vec3f, way: vec3f, distance: f32) -> f32 {
  if (light.shadow.x < 0.0) {
    return 1.0;
  }
  var layer = u32(light.shadow.x);
  if (light.shadow.y > 0.5) {
    let out = -way * distance;
    let a = abs(out);
    var face = 0u;
    if (a.x >= a.y && a.x >= a.z) {
      face = select(0u, 1u, out.x < 0.0);
    } else if (a.y >= a.z) {
      face = select(2u, 3u, out.y < 0.0);
    } else {
      face = select(4u, 5u, out.z < 0.0);
    }
    layer += face;
  }
  let matrix = local_shadow_faces[layer];
  let ahead = (matrix * vec4f(world, 1.0)).w;
  let texel = light.shadow.z * max(ahead, 0.0001);
  let facing = clamp(dot(normal, way), 0.0, 1.0);
  let sine = sqrt(clamp(1.0 - facing * facing, 0.0, 1.0));
  let tangent = min(sine / max(facing, 0.05), 4.0);
  let shifted = world + normal * (texel * local_normal_offset * sine) + way * (texel * (local_bias_base + local_bias_slope * tangent));
  let clip = matrix * vec4f(shifted, 1.0);
  if (clip.w <= 0.0) {
    return 1.0;
  }
  let ndc = clip.xyz / clip.w;
  if (ndc.z >= 1.0) {
    return 1.0;
  }
  let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let angle = 6.2831853 * fract(sin(dot(world, vec3f(12.9898, 78.233, 37.719))) * 43758.5453);
  let spin = vec2f(cos(angle), sin(angle));
  let step = local_filter_radius / f32(textureDimensions(local_shadows).x);
  var lit = 0.0;
  for (var i = 0u; i < 8u; i++) {
    let tap = local_shadow_taps[i];
    let offset = vec2f(tap.x * spin.x - tap.y * spin.y, tap.x * spin.y + tap.y * spin.x) * step;
    lit += textureSampleCompareLevel(local_shadows, local_shadow_sampler, uv + offset, layer, ndc.z);
  }
  return lit * 0.125;
}

// How much of `light` reaches `world` on a surface facing `normal`: its color
// times the cosine of its angle to the surface, falling smoothly to nothing
// at its reach (and, for a spot, outside its cone).
fn cluster_light(light: ClusterLight, world: vec3f, normal: vec3f) -> vec3f {
  let toward = light.position.xyz - world;
  let reach = light.position.w;
  let dist2 = dot(toward, toward);
  if (reach <= 0.0 || dist2 >= reach * reach) {
    return vec3f(0.0);
  }
  if (dot(normal, toward) <= 0.0) {
    return vec3f(0.0);
  }
  let distance = sqrt(dist2);
  let way = toward / max(distance, 0.0001);
  let ratio = distance / reach;
  let window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
  var strength = window * window / (distance * distance + 1.0);
  if (light.color.w > -1.0) {
    strength *= smoothstep(light.color.w, max(light.direction.w, light.color.w + 0.0001), dot(-way, light.direction.xyz));
  }
  let nl = max(dot(normal, way), 0.0);
  return light.color.rgb * strength * nl * local_shadow(light, world, normal, way, distance);
}

// The cluster of the fragment at `position` (its @builtin(position)), whose
// distance ahead of the eye is `depth`.
fn cluster_of(position: vec4f, depth: f32) -> u32 {
  let counts = cluster_grid.counts;
  let tile = min(vec2u(max(position.xy * cluster_grid.scale.xy, vec2f(0.0))), counts.xy - 1u);
  let along = select(log(max(depth, 0.000001)), depth, cluster_grid.options.x > 0.5);
  let slice = u32(clamp(along * cluster_grid.scale.z + cluster_grid.scale.w, 0.0, f32(counts.z - 1u)));
  return (slice * counts.y + tile.y) * counts.x + tile.x;
}

// The light the frame's point and spot lights bring to `world`, on a surface
// facing `normal` (unit length), seen at the fragment `position` (its
// @builtin(position)) whose view-space z is `view_z` (negative ahead of the
// eye): their colors summed, each times the cosine of its angle to the
// surface, as a diffuse surface's irradiance.
fn clustered_light(world: vec3f, normal: vec3f, position: vec4f, view_z: f32) -> vec3f {
  if (cluster_grid.counts.w == 0u) {
    return vec3f(0.0);
  }
  let range = cluster_ranges[cluster_of(position, -view_z)];
  var sum = vec3f(0.0);
  for (var index = range.x; index < range.x + range.y; index++) {
    sum += cluster_light(cluster_lights[cluster_indices[index]], world, normal);
  }
  return sum;
}

// One light as a surface sees it: the unit direction toward it and the
// radiance it brings there (its color times its strength and its visibility,
// the frame's light scale, with no cosine), zero where it does not reach or
// the surface faces away from it.
struct ClusterSample {
  way: vec3f,
  radiance: vec3f,
};

// `light` at `world` for a surface facing `normal`. A surface that faces away
// from it takes nothing (and the shadow map is not read), unless `behind`: a
// subsurface that lets light through from the back.
fn cluster_light_sample(light: ClusterLight, world: vec3f, normal: vec3f, behind: bool) -> ClusterSample {
  var taken = ClusterSample(vec3f(0.0), vec3f(0.0));
  let toward = light.position.xyz - world;
  let reach = light.position.w;
  let dist2 = dot(toward, toward);
  if (reach <= 0.0 || dist2 >= reach * reach) {
    return taken;
  }
  if (!behind && dot(normal, toward) <= 0.0) {
    return taken;
  }
  let distance = sqrt(dist2);
  let way = toward / max(distance, 0.0001);
  let ratio = distance / reach;
  let window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
  var strength = window * window / (distance * distance + 1.0);
  if (light.color.w > -1.0) {
    strength *= smoothstep(light.color.w, max(light.direction.w, light.color.w + 0.0001), dot(-way, light.direction.xyz));
  }
  taken.way = way;
  taken.radiance = light.color.rgb * strength * local_shadow(light, world, normal, way, distance);
  return taken;
}

// ---- Linearly Transformed Cosines (LTC) Area Lights via @group(2) @binding(7).
const area_max_lights: u32 = 64u;
const area_mat_offset: u32 = 320u;
const area_amp_offset: u32 = 4416u;
const area_pi: f32 = 3.14159265359;
const area_octagon_scale: f32 = 1.0539073;

fn area_sample_lut(base_offset: u32, rough: f32, q: f32) -> vec4f {
  let fx = clamp(rough, 0.0, 1.0) * 63.0;
  let fy = clamp(q, 0.0, 1.0) * 63.0;
  let x0 = min(u32(floor(fx)), 62u);
  let y0 = min(u32(floor(fy)), 62u);
  let tx = fx - f32(x0);
  let ty = fy - f32(y0);
  let i00 = base_offset + y0 * 64u + x0;
  let i10 = i00 + 1u;
  let i01 = i00 + 64u;
  let i11 = i01 + 1u;
  let row0 = mix(area_light_data[i00], area_light_data[i10], tx);
  let row1 = mix(area_light_data[i01], area_light_data[i11], tx);
  return mix(row0, row1, ty);
}

fn area_integrate_edge(v1: vec3f, v2: vec3f) -> vec3f {
  let c = cross(v1, v2);
  let sin_t = length(c);
  if (sin_t <= 1e-6) {
    return vec3f(0.0);
  }
  let cos_t = clamp(dot(v1, v2), -1.0, 1.0);
  let theta = atan2(sin_t, cos_t);
  return c * (theta / (2.0 * area_pi * sin_t));
}

fn area_eval_quad(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f) -> f32 {
  let poly = array<vec3f, 4>(p0, p1, p2, p3);
  var clipped = array<vec3f, 5>(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
  var out_n = 0u;
  for (var i = 0u; i < 4u; i++) {
    let cur = poly[i];
    let nxt = poly[(i + 1u) % 4u];
    let cur_in = cur.z >= 0.0;
    let nxt_in = nxt.z >= 0.0;
    if (cur_in && out_n < 5u) {
      clipped[out_n] = cur;
      out_n += 1u;
    }
    if (cur_in != nxt_in && out_n < 5u) {
      let denom = cur.z - nxt.z;
      let t = select(0.5, cur.z / denom, abs(denom) > 1e-7);
      clipped[out_n] = mix(cur, nxt, clamp(t, 0.0, 1.0));
      out_n += 1u;
    }
  }
  if (out_n < 3u) {
    return 0.0;
  }
  var dirs = array<vec3f, 5>(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
  var valid_n = 0u;
  for (var i = 0u; i < out_n; i++) {
    let len2 = dot(clipped[i], clipped[i]);
    if (len2 > 1e-10) {
      let d = clipped[i] * inverseSqrt(len2);
      if (valid_n == 0u || dot(d - dirs[valid_n - 1u], d - dirs[valid_n - 1u]) > 1e-10) {
        dirs[valid_n] = d;
        valid_n += 1u;
      }
    }
  }
  if (valid_n < 3u) {
    return 0.0;
  }
  var sum = vec3f(0.0);
  for (var i = 0u; i < valid_n; i++) {
    sum += area_integrate_edge(dirs[i], dirs[(i + 1u) % valid_n]);
  }
  return clamp(abs(sum.z), 0.0, 1.0);
}

fn area_basis(n: vec3f, v: vec3f) -> mat3x3f {
  let v_proj = v - n * dot(v, n);
  let len2 = dot(v_proj, v_proj);
  var t1: vec3f;
  if (len2 > 1e-6) {
    t1 = v_proj * inverseSqrt(len2);
  } else {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    t1 = normalize(cross(up, n));
  }
  let t2 = cross(n, t1);
  return transpose(mat3x3f(t1, t2, n));
}

const area_cookie_offset: u32 = 8512u;
const area_cookie_dim: u32 = 32u;
const area_cookie_texels: u32 = 1024u;
const area_max_cookies: u32 = 8u;

fn area_sample_cookie(slot: u32, uv: vec2f) -> vec3f {
  let c = clamp(uv, vec2f(0.0), vec2f(1.0)) * f32(area_cookie_dim - 1u);
  let x0 = u32(floor(c.x));
  let y0 = u32(floor(c.y));
  let x1 = min(x0 + 1u, area_cookie_dim - 1u);
  let y1 = min(y0 + 1u, area_cookie_dim - 1u);
  let fx = c.x - f32(x0);
  let fy = c.y - f32(y0);
  let base = area_cookie_offset + min(slot, area_max_cookies - 1u) * area_cookie_texels;
  let t00 = area_light_data[base + y0 * area_cookie_dim + x0].rgb;
  let t10 = area_light_data[base + y0 * area_cookie_dim + x1].rgb;
  let t01 = area_light_data[base + y1 * area_cookie_dim + x0].rgb;
  let t11 = area_light_data[base + y1 * area_cookie_dim + x1].rgb;
  return mix(mix(t00, t10, fx), mix(t01, t11, fx), fy);
}

fn area_range_window(dist: f32, reach: f32) -> f32 {
  if (reach <= 0.0) {
    return 1.0;
  }
  if (dist >= reach) {
    return 0.0;
  }
  let r = dist / reach;
  let w = clamp(1.0 - r * r * r * r, 0.0, 1.0);
  return w * w;
}

// Evaluates all active LTC area lights (Rect, Disk, Tube, CookieSpot) in `area_light_data`
// with full Fresnel endpoints (`f0`, `f90`), lobe energy sharing (transmission/sheen/coat),
// packed RGBA cookies, and directional GGX BRDF for cookie projectors.
// Emitter `col_reach.rgb` is radiance in scene-linear units (a unit-radiance full hemisphere
// produces irradiance 1.0 in Pen's pi-scaled Lambert convention).
fn clustered_area_lights_surface(
  world: vec3f,
  normal: vec3f,
  view: vec3f,
  base_color: vec3f,
  metallic: f32,
  f0: vec3f,
  f90: f32,
  roughness: f32,
  diff_scale: f32,
  spec_scale: f32,
  coat_weight: f32,
  coat_roughness: f32,
  coat_normal: vec3f,
  sheen_tint: vec3f,
) -> vec3f {
  let count = min(u32(max(cluster_grid.options.y, 0.0) + 0.5), area_max_lights);
  if (count == 0u) {
    return vec3f(0.0);
  }
  let n = normalize(normal);
  let v = normalize(view);
  let ndotv = clamp(dot(n, v), 0.0, 1.0);
  let rough = clamp(roughness, 0.04, 1.0);
  let metal = clamp(metallic, 0.0, 1.0);
  let has_sheen = dot(sheen_tint, sheen_tint) > 0.0;

  var ltc_ready = false;
  var basis = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
  var spec_xf = basis;
  var spec_fresnel = vec3f(0.0);
  var diff_albedo = vec3f(0.0);
  var coat_xf = basis;
  var coat_fresnel = 0.0;
  var total = vec3f(0.0);

  for (var i = 0u; i < count; i++) {
    let base = i * 5u;
    let pos_kind = area_light_data[base];
    let col_reach = area_light_data[base + 3u];
    let center = pos_kind.xyz;
    let dist_center = length(center - world);
    let atten = area_range_window(dist_center, col_reach.w);
    if (atten <= 0.0) {
      continue;
    }
    let right_ext = area_light_data[base + 1u];
    let up_ext = area_light_data[base + 2u];
    let flags = area_light_data[base + 4u];

    let kind = u32(round(pos_kind.w));
    let right_len = length(right_ext.xyz);
    let up_len = length(up_ext.xyz);
    if (right_len <= 1e-5 || up_len <= 1e-5) {
      continue;
    }
    let right = right_ext.xyz / right_len;
    let up = up_ext.xyz / up_len;
    let two_sided = flags.x > 0.5;

    if (kind == 0u || kind == 1u) {
      let emit_raw = cross(right, up);
      let emit_len = length(emit_raw);
      if (emit_len <= 1e-5) {
        continue;
      }
      let emit_n = emit_raw / emit_len;
      if (!two_sided && dot(emit_n, world - center) <= 0.0) {
        continue;
      }
    }

    if (!ltc_ready && kind != 3u) {
      let q = sqrt(max(1.0 - ndotv, 0.0));
      let t1 = area_sample_lut(area_mat_offset, rough, q);
      let t2 = area_sample_lut(area_amp_offset, rough, q);
      let minv = mat3x3f(
        vec3f(t1.x, 0.0, t1.y),
        vec3f(0.0, 1.0, 0.0),
        vec3f(t1.z, 0.0, t1.w),
      );
      basis = area_basis(n, v);
      spec_xf = minv * basis;
      spec_fresnel = f0 * t2.x + vec3f(f90 * t2.y);
      // Conserve energy against the LTC directional specular albedo so a unit-radiance hemisphere stays <= 1.
      diff_albedo = base_color * (1.0 - metal) * max(vec3f(1.0) - spec_fresnel, vec3f(0.0)) * diff_scale;
      coat_xf = spec_xf;
      if (coat_weight > 0.0) {
        let cn = normalize(coat_normal);
        let cndv = clamp(dot(cn, v), 0.0, 1.0);
        let cq = sqrt(max(1.0 - cndv, 0.0));
        let cr = clamp(coat_roughness, 0.04, 1.0);
        let ct1 = area_sample_lut(area_mat_offset, cr, cq);
        let ct2 = area_sample_lut(area_amp_offset, cr, cq);
        let cminv = mat3x3f(
          vec3f(ct1.x, 0.0, ct1.y),
          vec3f(0.0, 1.0, 0.0),
          vec3f(ct1.z, 0.0, ct1.w),
        );
        coat_xf = cminv * area_basis(cn, v);
        coat_fresnel = coat_weight * (0.04 * ct2.x + ct2.y);
      }
      ltc_ready = true;
    }

    if (kind == 3u) {
      // Cookie spotlight cone projector evaluated with directional GGX microfacet BRDF.
      let forward = normalize(cross(right, up));
      let rel = world - center;
      let lz = dot(rel, forward);
      if (lz <= 1e-4) {
        continue;
      }
      let tan_fov = max(flags.w, 0.05);
      let aspect = max(up_ext.w, 0.1);
      let u = 0.5 + dot(rel, right) / (2.0 * lz * tan_fov * aspect);
      let v_coord = 0.5 - dot(rel, up) / (2.0 * lz * tan_fov);
      if (u < 0.0 || u > 1.0 || v_coord < 0.0 || v_coord > 1.0) {
        continue;
      }
      let edge = max(flags.z, 0.005);
      let mask = smoothstep(0.0, edge, u) * smoothstep(0.0, edge, 1.0 - u)
        * smoothstep(0.0, edge, v_coord) * smoothstep(0.0, edge, 1.0 - v_coord);
      let cookie_rgb = select(vec3f(1.0), area_sample_cookie(u32(flags.y + 0.5), vec2f(u, v_coord)), flags.y >= 0.0);
      let to_l = -rel / max(dist_center, 1e-4);
      let ndl = max(dot(n, to_l), 0.0);
      if (ndl <= 0.0) {
        continue;
      }
      let falloff = mask / (dist_center * dist_center + 1.0);
      let radiance = col_reach.rgb * cookie_rgb * (falloff * atten);
      let h = normalize(v + to_l);
      let nh = clamp(dot(n, h), 0.0, 1.0);
      let vh = clamp(dot(v, h), 0.0, 1.0);
      let nv = max(ndotv, 1e-5);
      let a = rough * rough;
      let a2 = a * a;
      let denom = nh * nh * (a2 - 1.0) + 1.0;
      let ggx_d = a2 / (3.14159265359 * denom * denom);
      let vis = 0.5 / max(1e-6, ndl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(ndl * ndl * (1.0 - a2) + a2));
      let fade = 1.0 - vh;
      let schlick = f0 + (vec3f(f90) - f0) * (fade * fade * fade * fade * fade);
      let dir_diff = base_color * (1.0 - metal) * max(vec3f(1.0) - schlick, vec3f(0.0)) * (ndl * diff_scale);
      let dir_spec = schlick * (ggx_d * vis * ndl * 3.14159265359 * spec_scale);
      var dir_coat = vec3f(0.0);
      if (coat_weight > 0.0) {
        let cn = normalize(coat_normal);
        let nlc = max(dot(cn, to_l), 0.0);
        let nhc = clamp(dot(cn, h), 0.0, 1.0);
        let nvc = max(dot(cn, v), 1e-5);
        let ca = coat_roughness * coat_roughness;
        let ca2 = ca * ca;
        let cd = nhc * nhc * (ca2 - 1.0) + 1.0;
        let c_ggx = ca2 / (3.14159265359 * cd * cd);
        let c_vis = 0.5 / max(1e-6, nlc * sqrt(nvc * nvc * (1.0 - ca2) + ca2) + nvc * sqrt(nlc * nlc * (1.0 - ca2) + ca2));
        let fc = coat_weight * (0.04 + 0.96 * fade * fade * fade * fade * fade);
        dir_coat = vec3f(fc * c_ggx * c_vis * nlc * 3.14159265359);
      }
      var dir_sheen = vec3f(0.0);
      if (has_sheen) {
        dir_sheen = sheen_tint * ndl;
      }
      total += (dir_diff + dir_spec + dir_sheen + dir_coat) * radiance;
      continue;
    }

    var diff_form = 0.0;
    var spec_form = 0.0;
    var coat_form = 0.0;

    if (kind == 0u) {
      let ex = right * right_ext.w;
      let ey = up * up_ext.w;
      let q0 = (center - ex - ey) - world;
      let q1 = (center + ex - ey) - world;
      let q2 = (center + ex + ey) - world;
      let q3 = (center - ex + ey) - world;
      diff_form = area_eval_quad(basis * q0, basis * q1, basis * q2, basis * q3);
      spec_form = area_eval_quad(spec_xf * q0, spec_xf * q1, spec_xf * q2, spec_xf * q3);
      if (coat_weight > 0.0) {
        coat_form = area_eval_quad(coat_xf * q0, coat_xf * q1, coat_xf * q2, coat_xf * q3);
      }
    } else if (kind == 1u) {
      let rad = max(right_ext.w, 1e-4) * area_octagon_scale;
      let d = rad * 0.70710678;
      let c0 = center + right * rad - world;
      let c1 = center + right * d + up * d - world;
      let c2 = center + up * rad - world;
      let c3 = center - right * d + up * d - world;
      let c4 = center - right * rad - world;
      let c5 = center - right * d - up * d - world;
      let c6 = center - up * rad - world;
      let c7 = center + right * d - up * d - world;
      diff_form = area_eval_quad(basis * c0, basis * c1, basis * c2, basis * c3)
        + area_eval_quad(basis * c4, basis * c5, basis * c6, basis * c7)
        + area_eval_quad(basis * c0, basis * c3, basis * c4, basis * c7);
      spec_form = area_eval_quad(spec_xf * c0, spec_xf * c1, spec_xf * c2, spec_xf * c3)
        + area_eval_quad(spec_xf * c4, spec_xf * c5, spec_xf * c6, spec_xf * c7)
        + area_eval_quad(spec_xf * c0, spec_xf * c3, spec_xf * c4, spec_xf * c7);
      if (coat_weight > 0.0) {
        coat_form = area_eval_quad(coat_xf * c0, coat_xf * c1, coat_xf * c2, coat_xf * c3)
          + area_eval_quad(coat_xf * c4, coat_xf * c5, coat_xf * c6, coat_xf * c7)
          + area_eval_quad(coat_xf * c0, coat_xf * c3, coat_xf * c4, coat_xf * c7);
      }
    } else {
      let half_len = max(right_ext.w, 1e-4);
      let tube_r = max(up_ext.w, 0.01);
      let to_p = world - center;
      let perp_raw = to_p - right * dot(to_p, right);
      let perp_dir = select(up, normalize(perp_raw), dot(perp_raw, perp_raw) > 1e-6);
      let side = normalize(cross(right, perp_dir));
      let ex = right * half_len;
      let ey = side * tube_r;
      let q0 = (center - ex - ey) - world;
      let q1 = (center + ex - ey) - world;
      let q2 = (center + ex + ey) - world;
      let q3 = (center - ex + ey) - world;
      diff_form = area_eval_quad(basis * q0, basis * q1, basis * q2, basis * q3);
      spec_form = area_eval_quad(spec_xf * q0, spec_xf * q1, spec_xf * q2, spec_xf * q3);
      if (coat_weight > 0.0) {
        coat_form = area_eval_quad(coat_xf * q0, coat_xf * q1, coat_xf * q2, coat_xf * q3);
      }
    }

    var contrib = diff_albedo * diff_form + spec_fresnel * (spec_form * spec_scale);
    if (has_sheen) {
      contrib += sheen_tint * diff_form;
    }
    if (coat_weight > 0.0) {
      contrib += vec3f(coat_fresnel * coat_form);
    }
    total += contrib * col_reach.rgb * atten;
  }
  return total;
}

// Convenience wrapper for shaders calling the 7-argument LTC area-light entry point.
fn clustered_area_lights(
  world: vec3f,
  normal: vec3f,
  view: vec3f,
  base_color: vec3f,
  metallic: f32,
  f0: vec3f,
  roughness: f32,
) -> vec3f {
  return clustered_area_lights_surface(
    world, normal, view, base_color, metallic, f0, 1.0, roughness, 1.0, 1.0, 0.0, 0.04, normal, vec3f(0.0),
  );
}
