// Prefix with one of Pen's TLAS traversal implementations.
struct RestirUniform {
  inverse_view_projection: mat4x4f,
  prev_view_projection: mat4x4f,
  camera_and_clear: vec4f,
  previous_camera: vec4f,
  sun_and_max_distance: vec4f,
  sun_color_and_bias: vec4f,
  sky_zenith_and_bounce: vec4f,
  sky_horizon: vec4f,
  scene_counts_and_mask: vec4u,
  size_lights_frame: vec4u,
  // x = DI history, y = GI history, z = has motion, w = enabled (DI bit 0, GI bit 1).
  flags: vec4u,
};

struct RtLocalLight {
  position_range: vec4f,
  color_outer_cos: vec4f,
  direction_inner_cos: vec4f,
};

struct DiReservoir {
  light_idx: u32,
  w_sum: f32,
  weight: f32,
  m: f32,
  surface_pos_roughness: vec4f,
};

struct GiReservoir {
  sample_pos_m: vec4f,
  sample_nrm_w: vec4f,
  radiance_wsum: vec4f,
  vis_pad: vec4f,
};

@group(0) @binding(0) var<uniform> u: RestirUniform;
// __TLAS_BINDING_1__
@group(0) @binding(2) var<storage, read> tlas_instances: array<TlasInstance>;
@group(0) @binding(3) var<storage, read> blas_nodes: array<BvhNode>;
@group(0) @binding(4) var<storage, read> blas_tris: array<BlasTri>;
@group(0) @binding(5) var depth_tex: texture_2d<f32>;
@group(0) @binding(6) var normal_roughness: texture_2d<f32>;
@group(0) @binding(7) var albedo_metallic: texture_2d<f32>;
@group(0) @binding(8) var motion_tex: texture_2d<f32>;
@group(0) @binding(9) var<storage, read> local_lights: array<RtLocalLight>;
@group(0) @binding(10) var<storage, read> ddgi_packed: array<vec4f>;
@group(0) @binding(11) var prev_depth_nrm: texture_2d<f32>;
@group(0) @binding(12) var next_depth_nrm: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var<storage, read> di_input: array<DiReservoir>;
@group(0) @binding(14) var<storage, read> di_prev: array<DiReservoir>;
@group(0) @binding(15) var<storage, read_write> di_output: array<DiReservoir>;
@group(0) @binding(16) var out_di: texture_storage_2d<rgba16float, write>;
@group(0) @binding(17) var<storage, read> gi_input: array<GiReservoir>;
@group(0) @binding(18) var<storage, read> gi_prev: array<GiReservoir>;
@group(0) @binding(19) var<storage, read_write> gi_output: array<GiReservoir>;
@group(0) @binding(20) var out_gi: texture_storage_2d<rgba16float, write>;
// __RESTIR_SHARC_DECLS__

const PI: f32 = 3.14159265359;
const TAU: f32 = 2.0 * PI;
const EPS: f32 = 1e-6;
const RAY_MIN: f32 = 1e-4;
const EMPTY_DI: DiReservoir = DiReservoir(0u, 0.0, 0.0, 0.0, vec4f(0.0));
const EMPTY_GI: GiReservoir = GiReservoir(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));

fn finite_nonnegative(x: f32, ceiling: f32) -> f32 {
  // Check the exponent bits so NaN/Inf rejection also survives relaxed float math.
  return select(clamp(x, 0.0, ceiling), 0.0, (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u);
}

fn safe_rgb(rgb: vec3f) -> vec3f {
  return vec3f(finite_nonnegative(rgb.x, 16384.0), finite_nonnegative(rgb.y, 16384.0),
               finite_nonnegative(rgb.z, 16384.0));
}

fn luma(rgb: vec3f) -> f32 {
  return dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
}

fn hash(value: u32) -> u32 {
  var x = value;
  x = (x ^ (x >> 16u)) * 0x7feb352du;
  x = (x ^ (x >> 15u)) * 0x846ca68bu;
  return x ^ (x >> 16u);
}

fn random(state: ptr<function, u32>) -> f32 {
  *state += 0x9e3779b9u;
  return f32(hash(*state) >> 8u) * (1.0 / 16777216.0);
}

fn rng_for(pixel: u32, stream: u32) -> u32 {
  return hash(pixel ^ hash(u.size_lights_frame.w) ^ hash(stream));
}

fn extent() -> vec2u {
  return u.size_lights_frame.xy;
}

fn pixel_index(pixel: vec2i) -> u32 {
  return u32(pixel.y) * extent().x + u32(pixel.x);
}

fn in_bounds(pixel: vec2i) -> bool {
  return all(pixel >= vec2i(0)) && all(pixel < vec2i(extent()));
}

fn input_pixel(pixel: vec2i, dims: vec2i) -> vec2i {
  let dispatch_dims = vec2i(extent());
  let scaled = vec2i((vec2f(pixel) + 0.5) * vec2f(dims) / vec2f(dispatch_dims));
  return select(clamp(scaled, vec2i(0), dims - 1), pixel, all(dims == dispatch_dims));
}

fn load_material(pixel: vec2i) -> vec4f {
  let dims = vec2i(textureDimensions(normal_roughness, 0));
  return textureLoad(albedo_metallic, input_pixel(pixel, dims), 0);
}

struct Geometry {
  pos: vec3f,
  nrm: vec3f,
  distance: f32,
  roughness: f32,
  camera: vec3f,
  valid: bool,
  uv: vec2f,
};

fn load_geometry(pixel: vec2i) -> Geometry {
  let dims = vec2i(textureDimensions(normal_roughness, 0));
  let g_px = input_pixel(pixel, dims);
  let depth = textureLoad(depth_tex, g_px, 0).x;
  let nr = textureLoad(normal_roughness, g_px, 0);
  if (!(dot(nr.xyz, nr.xyz) >= 0.01) || !(abs(depth - u.camera_and_clear.w) > 1e-6)) {
    return Geometry(vec3f(0.0), vec3f(0.0), 0.0, 1.0, vec3f(0.0), false, vec2f(0.0));
  }
  let uv = (vec2f(g_px) + 0.5) / vec2f(dims);
  let clip = vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), depth, 1.0);
  let world = u.inverse_view_projection * clip;
  if (!(abs(world.w) > 1e-8)) {
    return Geometry(vec3f(0.0), vec3f(0.0), 0.0, 1.0, vec3f(0.0), false, vec2f(0.0));
  }
  let pos = world.xyz / world.w;
  return Geometry(pos, normalize(nr.xyz), length(pos - u.camera_and_clear.xyz),
                  clamp(nr.w, 0.045, 1.0), u.camera_and_clear.xyz, true, uv);
}

fn similar(geometry: Geometry, distance: f32, normal: vec3f) -> bool {
  return abs(geometry.distance - distance) / max(geometry.distance, 0.1) < 0.08
      && dot(geometry.nrm, normal) > 0.85;
}

fn oct_decode(encoded: vec2f) -> vec3f {
  let z = 1.0 - abs(encoded.x) - abs(encoded.y);
  let xy = select(encoded, (1.0 - abs(encoded.yx)) * rt_ddgi_sign_nz(encoded), z < 0.0);
  return normalize(vec3f(xy, z));
}

fn history_pixel(pixel: vec2i, geometry: Geometry) -> vec2i {
  var uv = geometry.uv;
  var mv = vec2f(100.0, 0.0);
  if (u.flags.z != 0u) {
    let dims = vec2i(textureDimensions(motion_tex, 0));
    mv = textureLoad(motion_tex, input_pixel(pixel, dims), 0).xy;
  }
  if (abs(mv.x) < 100.0) {
    uv -= mv;
  } else {
    let clip = u.prev_view_projection * vec4f(geometry.pos, 1.0);
    if (!(clip.w > 1e-8)) { return vec2i(-1); }
    uv = clip.xy / clip.w * vec2f(0.5, -0.5) + 0.5;
  }
  if (!all(uv >= vec2f(0.0)) || !all(uv < vec2f(1.0))) { return vec2i(-1); }
  let prev = vec2i(uv * vec2f(extent()));
  let packed = textureLoad(prev_depth_nrm, prev, 0);
  if (packed.w < 0.5 || !similar(geometry, packed.x, oct_decode(packed.yz))) {
    return vec2i(-1);
  }
  return prev;
}

fn neighbor_pixel(pixel: vec2i, rotation: f32, i: u32, radius: f32, rng: ptr<function, u32>) -> vec2i {
  let angle = rotation + f32(i) * (TAU / 4.0);
  let r = radius * sqrt(random(rng));
  return pixel + vec2i(round(vec2f(cos(angle), sin(angle)) * r));
}

fn trace(origin: vec3f, dir: vec3f, distance: f32, any_hit: bool) -> TlasHit {
  return trace_tlas(origin, dir, RAY_MIN, distance, u.scene_counts_and_mask.w, any_hit,
                    u.scene_counts_and_mask.x, u.scene_counts_and_mask.y, u.scene_counts_and_mask.z);
}

fn ray_bias() -> f32 {
  return max(u.sun_color_and_bias.w, RAY_MIN);
}

fn visible(pos: vec3f, nrm: vec3f, endpoint: vec3f) -> bool {
  let bias = ray_bias();
  let origin = pos + nrm * bias;
  let delta = endpoint - origin;
  let distance = length(delta);
  if (distance <= 2.0 * RAY_MIN) { return true; }
  let limit = max(distance - max(bias * 0.5, 2.0 * RAY_MIN), RAY_MIN);
  return !trace(origin, delta / distance, limit, true).hit;
}

struct LightEval {
  dir: vec3f,
  rgb: vec3f,
  cosine: f32,
};

fn local_light(pos: vec3f, nrm: vec3f, light: RtLocalLight) -> LightEval {
  let delta = light.position_range.xyz - pos;
  let distance = length(delta);
  let reach = light.position_range.w;
  if (!(distance > RAY_MIN && distance < reach)) {
    return LightEval(vec3f(0.0), vec3f(0.0), 0.0);
  }
  let dir = delta / distance;
  let cosine = max(dot(nrm, dir), 0.0);
  let range_ratio2 = (distance / reach) * (distance / reach);
  let window = 1.0 - range_ratio2 * range_ratio2;
  var attenuation = window * window / (distance * distance + 1.0);
  let outer = light.color_outer_cos.w;
  if (outer > -0.999) {
    attenuation *= smoothstep(outer, max(light.direction_inner_cos.w, outer + 1e-4),
                              dot(-dir, light.direction_inner_cos.xyz));
  }
  return LightEval(dir, light.color_outer_cos.rgb * attenuation, cosine);
}

fn eval_local_light_brdf(geometry: Geometry, material: vec4f, light_idx: u32) -> vec3f {
  let incident = local_light(geometry.pos, geometry.nrm, local_lights[light_idx]);
  if (incident.cosine <= 0.0) { return vec3f(0.0); }
  let view_delta = geometry.camera - geometry.pos;
  let view = view_delta / max(length(view_delta), EPS);
  let half_delta = incident.dir + view;
  let half_dir = half_delta / max(length(half_delta), EPS);
  let ndv = max(dot(geometry.nrm, view), EPS);
  let ndh = clamp(dot(geometry.nrm, half_dir), 0.0, 1.0);
  let vdh = clamp(dot(view, half_dir), 0.0, 1.0);
  let rough2 = geometry.roughness * geometry.roughness;
  let a2 = rough2 * rough2;
  let denom = ndh * ndh * (a2 - 1.0) + 1.0;
  let distribution = a2 / max(PI * denom * denom, 1e-10);
  let k = (geometry.roughness + 1.0) * (geometry.roughness + 1.0) * 0.125;
  let masking = ndv / mix(ndv, 1.0, k) * incident.cosine / mix(incident.cosine, 1.0, k);
  let metal = clamp(material.w, 0.0, 1.0);
  let albedo = clamp(material.rgb, vec3f(0.0), vec3f(1.0));
  let f0 = mix(vec3f(0.04), albedo, metal);
  let fresnel = mix(f0, vec3f(1.0), pow(1.0 - vdh, 5.0));
  let spec = fresnel * distribution * masking / max(4.0 * ndv * incident.cosine, EPS);
  let diffuse = (1.0 - fresnel) * albedo * ((1.0 - metal) / PI);
  return safe_rgb(incident.rgb * incident.cosine * (diffuse + spec));
}

fn stratified_light(i: u32, count: u32, candidates: u32, offset: u32, rng: ptr<function, u32>) -> u32 {
  if (count == candidates) { return (i + offset) % count; }
  let pick = u32((f32(i) + random(rng)) * f32(count) / f32(candidates));
  return (min(pick, count - 1u) + offset) % count;
}

fn di_add(r: ptr<function, DiReservoir>, light_idx: u32, mass: f32, rng: ptr<function, u32>) {
  (*r).w_sum += mass;
  if (mass > 0.0 && random(rng) * (*r).w_sum < mass) {
    (*r).light_idx = light_idx;
  }
}

fn di_finish(r: DiReservoir, geometry: Geometry, material: vec4f) -> DiReservoir {
  var result = r;
  result.surface_pos_roughness = vec4f(geometry.pos, geometry.roughness);
  if (r.w_sum > 0.0) {
    let p_hat = luma(eval_local_light_brdf(geometry, material, r.light_idx));
    result.weight = r.w_sum / max(r.m * p_hat, EPS);
  }
  return result;
}

fn pairwise_mass(p_source: f32, p_other: f32, m_source: f32, m_other: f32) -> f32 {
  let weighted = m_source * p_source;
  if (!(weighted > 0.0)) { return 0.0; }
  return (m_source + m_other) * weighted / (weighted + m_other * p_other);
}

@compute @workgroup_size(8, 8, 1)
fn restir_di_initial(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= extent())) { return; }
  let pixel = vec2i(id.xy);
  let index = pixel_index(pixel);
  let g = load_geometry(pixel);
  var r = EMPTY_DI;
  let count = u.size_lights_frame.z;
  if ((u.flags.w & 1u) != 0u && g.valid && count > 0u) {
    let mat = load_material(pixel);
    let candidates = min(8u, count);
    var rng = rng_for(index, 1u);
    let offset = u32(random(&rng) * f32(count));
    for (var i = 0u; i < candidates; i++) {
      let light_idx = stratified_light(i, count, candidates, offset, &rng);
      let p_hat = luma(eval_local_light_brdf(g, mat, light_idx));
      di_add(&r, light_idx, p_hat * f32(count), &rng);
    }
    r.m = f32(candidates);
    r = di_finish(r, g, mat);
    if (r.weight > 0.0 && !visible(g.pos, g.nrm, local_lights[r.light_idx].position_range.xyz)) {
      r.weight = 0.0;
    }
  }
  di_output[index] = r;
}

@compute @workgroup_size(8, 8, 1)
fn restir_di_temporal(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= extent())) { return; }
  let pixel = vec2i(id.xy);
  let index = pixel_index(pixel);
  let current = di_input[index];
  let g = load_geometry(pixel);
  var result = current;
  if (g.valid && u.flags.x != 0u && current.m > 0.0) {
    let prev = history_pixel(pixel, g);
    if (in_bounds(prev)) {
      let mat = load_material(pixel);
      let old = di_prev[pixel_index(prev)];
      let m_old = min(old.m, 20.0);
      if (m_old > 0.0) {
        let previous_geometry = textureLoad(prev_depth_nrm, prev, 0);
        let qg = Geometry(old.surface_pos_roughness.xyz, oct_decode(previous_geometry.yz),
                          previous_geometry.x, old.surface_pos_roughness.w, u.previous_camera.xyz, true, vec2f(0.0));
        let current_p = luma(eval_local_light_brdf(g, mat, current.light_idx));
        let old_at_current = luma(eval_local_light_brdf(g, mat, old.light_idx));
        // The validated reprojection is assumed to retain its albedo and metallic.
        let old_p = luma(eval_local_light_brdf(qg, mat, old.light_idx));
        let current_at_old = luma(eval_local_light_brdf(qg, mat, current.light_idx));
        let current_mass = pairwise_mass(current_p, current_at_old, current.m, m_old);
        let old_mass = pairwise_mass(old_p, old_at_current, m_old, current.m);
        var rng = rng_for(index, 2u);
        result = EMPTY_DI;
        result.m = current.m + m_old;
        di_add(&result, current.light_idx, current.weight * current_p * current_mass, &rng);
        di_add(&result, old.light_idx, old.weight * old_at_current * old_mass, &rng);
        result = di_finish(result, g, mat);
      }
    }
  }
  di_output[index] = result;
}

@compute @workgroup_size(8, 8, 1)
fn restir_di_spatial(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= extent())) { return; }
  let pixel = vec2i(id.xy);
  let index = pixel_index(pixel);
  let g = load_geometry(pixel);
  let center = di_input[index];
  var result = EMPTY_DI;
  var rgb = vec3f(0.0);
  if (g.valid && center.m > 0.0) {
    let mat = load_material(pixel);
    let center_p = luma(eval_local_light_brdf(g, mat, center.light_idx));
    let m_center = center.m / 4.0;
    var center_mass = 0.0;
    var rng = rng_for(index, 3u);
    let rotation = random(&rng) * TAU;
    result.m = center.m;
    for (var i = 0u; i < 4u; i++) {
      let q = neighbor_pixel(pixel, rotation, i, 20.0, &rng);
      if (in_bounds(q) && any(q != pixel)) {
        let qg = load_geometry(q);
        let other = di_input[pixel_index(q)];
        if (qg.valid && similar(g, qg.distance, qg.nrm) && other.m > 0.0) {
          let qmat = load_material(q);
          let other_at_q = luma(eval_local_light_brdf(qg, qmat, other.light_idx));
          let other_at_center = luma(eval_local_light_brdf(g, mat, other.light_idx));
          let center_at_q = luma(eval_local_light_brdf(qg, qmat, center.light_idx));
          let neighbor_mass = pairwise_mass(other_at_q, other_at_center, other.m, m_center);
          center_mass += pairwise_mass(center_p, center_at_q, m_center, other.m);
          di_add(&result, other.light_idx, other.weight * other_at_center * neighbor_mass, &rng);
          result.m += other.m;
          continue;
        }
      }
      center_mass += m_center;
    }
    di_add(&result, center.light_idx, center.weight * center_p * center_mass, &rng);
    result = di_finish(result, g, mat);
    result.weight = finite_nonnegative(result.weight, 1024.0);
    if (result.weight > 0.0 && visible(g.pos, g.nrm, local_lights[result.light_idx].position_range.xyz)) {
      rgb = safe_rgb(eval_local_light_brdf(g, mat, result.light_idx) * result.weight);
    } else {
      result.weight = 0.0;
    }
  }
  di_output[index] = result;
  textureStore(out_di, pixel, vec4f(rgb, f32(g.valid && (u.flags.w & 1u) != 0u)));
}
// Packed atlas decode matches tlas_raytrace.wgsl (cascade headers, offsets, and visibility moments).
fn rt_ddgi_sign_nz(v: vec2f) -> vec2f {
  return vec2f(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
}

fn rt_ddgi_oct_encode(v: vec3f) -> vec2f {
  let l1 = abs(v.x) + abs(v.y) + abs(v.z);
  var p = v.xy / max(l1, 1e-8);
  if (v.z < 0.0) {
    p = (1.0 - abs(p.yx)) * rt_ddgi_sign_nz(p);
  }
  return p;
}

fn rt_ddgi_wrap(v: i32, n: u32) -> u32 {
  let ni = max(i32(n), 1);
  return u32(((v % ni) + ni) % ni);
}

fn rt_ddgi_sample_irr(irr_off: u32, w: u32, h: u32, fpx: vec2f) -> vec4f {
  let p = fpx - vec2f(0.5);
  let x0 = u32(clamp(floor(p.x), 0.0, f32(w - 1u)));
  let y0 = u32(clamp(floor(p.y), 0.0, f32(h - 1u)));
  let x1 = min(x0 + 1u, w - 1u);
  let y1 = min(y0 + 1u, h - 1u);
  let fx = clamp(p.x - f32(x0), 0.0, 1.0);
  let fy = clamp(p.y - f32(y0), 0.0, 1.0);
  let c00 = ddgi_packed[irr_off + y0 * w + x0];
  let c10 = ddgi_packed[irr_off + y0 * w + x1];
  let c01 = ddgi_packed[irr_off + y1 * w + x0];
  let c11 = ddgi_packed[irr_off + y1 * w + x1];
  return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

fn rt_ddgi_vis_at(vis_off: u32, w: u32, x: u32, y: u32) -> vec2f {
  let idx = y * w + x;
  let packed = ddgi_packed[vis_off + (idx >> 1u)];
  return select(packed.xy, packed.zw, (idx & 1u) != 0u);
}

fn rt_ddgi_sample_vis(vis_off: u32, w: u32, h: u32, fpx: vec2f) -> vec2f {
  let p = fpx - vec2f(0.5);
  let x0 = u32(clamp(floor(p.x), 0.0, f32(w - 1u)));
  let y0 = u32(clamp(floor(p.y), 0.0, f32(h - 1u)));
  let x1 = min(x0 + 1u, w - 1u);
  let y1 = min(y0 + 1u, h - 1u);
  let fx = clamp(p.x - f32(x0), 0.0, 1.0);
  let fy = clamp(p.y - f32(y0), 0.0, 1.0);
  let c00 = rt_ddgi_vis_at(vis_off, w, x0, y0);
  let c10 = rt_ddgi_vis_at(vis_off, w, x1, y0);
  let c01 = rt_ddgi_vis_at(vis_off, w, x0, y1);
  let c11 = rt_ddgi_vis_at(vis_off, w, x1, y1);
  return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

fn rt_ddgi_cascade(cascade_idx: u32, world: vec3f, normal: vec3f) -> vec4f {
  let base = 1u + cascade_idx * 6u;
  let h0 = ddgi_packed[base + 0u];
  let h1 = ddgi_packed[base + 1u];
  let h2 = ddgi_packed[base + 2u];
  let h3 = ddgi_packed[base + 3u];
  let h4 = ddgi_packed[base + 4u];
  let h5 = ddgi_packed[base + 5u];
  let dims = vec3u(u32(max(h2.x, 0.0)), u32(max(h2.y, 0.0)), u32(max(h2.z, 0.0)));
  if (dims.x < 2u || dims.y < 2u || dims.z < 2u) {
    return vec4f(0.0);
  }
  let origin = h0.xyz;
  let gamma = max(h0.w, 1.0);
  let spacing = max(h1.xyz, vec3f(1e-4));
  let min_sp = min(spacing.x, min(spacing.y, spacing.z));
  let biased_pos = world + normal * (h1.w * min_sp);
  let rel = (biased_pos - origin) / spacing;
  let max_c = vec3f(f32(dims.x - 1u), f32(dims.y - 1u), f32(dims.z - 1u));
  let edge = min(min(min(rel.x, max_c.x - rel.x), min(rel.y, max_c.y - rel.y)), min(rel.z, max_c.z - rel.z));
  let reach = clamp(edge + 0.5, 0.0, 1.0);
  if (reach <= 0.0) {
    return vec4f(0.0);
  }
  let scroll = vec3i(i32(round(h3.x)), i32(round(h3.y)), i32(round(h3.z)));
  let irr_off = u32(max(h3.w, 0.0));
  let vis_off = u32(max(h4.x, 0.0));
  let strength = max(h4.y, 0.0);
  let irr_w = u32(max(h4.z, 1.0));
  let irr_h = u32(max(h4.w, 1.0));
  let vis_w = u32(max(h5.x, 1.0));
  let vis_h = u32(max(h5.y, 1.0));
  let offsets_off = u32(max(h5.z, 0.0));

  let clamped = clamp(rel, vec3f(0.0), max_c);
  let base_i = vec3u(floor(clamped));
  let alpha = clamped - vec3f(base_i);
  let oct_n = rt_ddgi_oct_encode(normalize(normal)) * 0.5 + vec2f(0.5);

  var sum_irr = vec3f(0.0);
  var sum_w = 0.0;
  for (var i = 0u; i < 8u; i++) {
    let off = vec3u(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u);
    let coord = min(base_i + off, dims - vec3u(1u));
    let sc = vec3u(
      rt_ddgi_wrap(i32(coord.x) + scroll.x, dims.x),
      rt_ddgi_wrap(i32(coord.y) + scroll.y, dims.y),
      rt_ddgi_wrap(i32(coord.z) + scroll.z, dims.z),
    );
    let tri_v = mix(vec3f(1.0) - alpha, alpha, vec3f(off));
    var w = max(tri_v.x * tri_v.y * tri_v.z, 1e-4);
    let s_idx = (sc.y * dims.z + sc.z) * dims.x + sc.x;
    let probe_off = select(vec3f(0.0), ddgi_packed[offsets_off + s_idx].xyz, offsets_off > 0u);
    let probe_pos = origin + vec3f(coord) * spacing + probe_off;
    let to_probe = probe_pos - world;
    let d_u = length(to_probe);
    let dir_p = select(normal, to_probe / d_u, d_u > 1e-5);
    let wrap = (dot(dir_p, normal) + 1.0) * 0.5;
    w *= wrap * wrap + 0.05;

    let s_col = sc.z * dims.x + sc.x;
    let s_row = sc.y;
    // Chebyshev visibility test from probe toward biased surface point (prevents thin-wall GI leaks in reflections).
    if (vis_off > 0u) {
      let p2pt = biased_pos - probe_pos;
      let d_pt = length(p2pt);
      let dir_from_probe = select(-normal, p2pt / d_pt, d_pt > 1e-5);
      let oct_v = rt_ddgi_oct_encode(dir_from_probe) * 0.5 + vec2f(0.5);
      let vis_px = vec2f(f32(s_col * 18u + 1u), f32(s_row * 18u + 1u)) + oct_v * 16.0;
      let moments = rt_ddgi_sample_vis(vis_off, vis_w, vis_h, vis_px);
      if (d_pt > moments.x) {
        let variance = max(abs(moments.y - moments.x * moments.x), 1e-4);
        let d = d_pt - moments.x;
        var cheb = variance / (variance + d * d);
        cheb = max(cheb * cheb * cheb, 0.0);
        w *= cheb;
      }
    }

    let irr_px = vec2f(f32(s_col * 10u + 1u), f32(s_row * 10u + 1u)) + oct_n * 8.0;
    let texel = rt_ddgi_sample_irr(irr_off, irr_w, irr_h, irr_px);
    w *= clamp(texel.a, 0.0, 1.0);
    if (w <= 1e-6) {
      continue;
    }
    let irr = pow(max(texel.rgb, vec3f(0.0)), vec3f(gamma * 0.5));
    sum_irr += irr * w;
    sum_w += w;
  }
  if (sum_w <= 1e-6) {
    return vec4f(0.0);
  }
  let avg = sum_irr / sum_w;
  return vec4f(avg * avg * strength, reach);
}

fn rt_sample_ddgi(world: vec3f, normal: vec3f) -> vec4f {
  let count = u32(clamp(round(ddgi_packed[0u].x), 0.0, 2.0));
  if (count == 0u) {
    return vec4f(0.0);
  }
  let c0 = rt_ddgi_cascade(0u, world, normal);
  if (count == 1u || c0.a >= 0.999) {
    return c0;
  }
  let c1 = rt_ddgi_cascade(1u, world, normal);
  return vec4f(mix(c1.rgb, c0.rgb, c0.a), max(c0.a, c1.a));
}

fn sky(dir: vec3f) -> vec3f {
  let lower = u.sky_horizon.rgb * mix(0.35, 1.0, clamp(1.0 + dir.y, 0.0, 1.0));
  let upper = mix(u.sky_horizon.rgb, u.sky_zenith_and_bounce.rgb, pow(max(dir.y, 0.0), 0.6));
  return select(lower, upper, dir.y >= 0.0);
}

fn shade_bounce(pos: vec3f, nrm: vec3f, hit: TlasHit, rng: ptr<function, u32>) -> vec4f {
  let diffuse_albedo = hit.albedo * (1.0 - hit.metallic);
  let diffuse = diffuse_albedo / PI;
  let sun = u.sun_and_max_distance.xyz;
  let cosine = max(dot(nrm, sun), 0.0);
  let max_dist = u.sun_and_max_distance.w;
  var radiance = hit.emissive;
  if (cosine > 0.0 && luma(u.sun_color_and_bias.rgb) > 0.0 &&
      visible(pos, nrm, pos + sun * max_dist)) {
    radiance += diffuse * u.sun_color_and_bias.rgb * cosine;
  }

  let count = u.size_lights_frame.z;
  let candidates = min(8u, count);
  let offset = u32(random(rng) * f32(count));
  for (var i = 0u; i < candidates; i++) {
    let light = local_lights[stratified_light(i, count, candidates, offset, rng)];
    let incident = local_light(pos, nrm, light);
    if (incident.cosine > 0.0 && luma(incident.rgb) > 0.0 &&
        visible(pos, nrm, light.position_range.xyz)) {
      radiance += diffuse * incident.rgb * (incident.cosine * f32(count) / f32(candidates));
    }
  }

  var ddgi_cov = 0.0;
  if (u.sky_zenith_and_bounce.w > 0.0) {
    let ddgi = rt_sample_ddgi(pos, nrm);
    ddgi_cov = clamp(ddgi.a, 0.0, 1.0);
    radiance += diffuse_albedo * ddgi.rgb * (ddgi_cov * u.sky_zenith_and_bounce.w);
  }
  return vec4f(safe_rgb(radiance), ddgi_cov);
}

fn cosine_direction(nrm: vec3f, rng: ptr<function, u32>) -> vec3f {
  let r2 = random(rng);
  let angle = TAU * random(rng);
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(nrm.y) > 0.9);
  let tangent = normalize(cross(up, nrm));
  let bitangent = cross(nrm, tangent);
  return normalize(sqrt(r2) * (cos(angle) * tangent + sin(angle) * bitangent)
                   + sqrt(1.0 - r2) * nrm);
}

fn gi_cosine(g: Geometry, sample: GiReservoir) -> f32 {
  let delta = sample.sample_pos_m.xyz - g.pos;
  let dir = delta / max(length(delta), EPS);
  // A surface sample stores radiance on the side seen by the generating ray.
  if (sample.vis_pad.y < u.sun_and_max_distance.w && dot(sample.sample_nrm_w.xyz, -dir) <= 0.0) {
    return 0.0;
  }
  return max(dot(g.nrm, dir), 0.0);
}

fn gi_target(g: Geometry, sample: GiReservoir) -> f32 {
  return luma(sample.radiance_wsum.rgb) * gi_cosine(g, sample);
}

fn gi_add(r: ptr<function, GiReservoir>, sample: GiReservoir, m: f32, mass: f32, rng: ptr<function, u32>) {
  let total = (*r).radiance_wsum.w + mass;
  let total_m = (*r).sample_pos_m.w + m;
  if (sample.vis_pad.x > 0.5 && mass > 0.0 && random(rng) * total < mass) {
    *r = sample;
  }
  (*r).sample_pos_m.w = total_m;
  (*r).radiance_wsum.w = total;
}

fn gi_finish(r: GiReservoir, g: Geometry) -> GiReservoir {
  var result = r;
  result.sample_nrm_w.w = r.radiance_wsum.w / max(r.sample_pos_m.w * gi_target(g, r), EPS);
  return result;
}

@compute @workgroup_size(8, 8, 1)
fn restir_gi_initial(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= extent())) { return; }
  let pixel = vec2i(id.xy);
  let index = pixel_index(pixel);
  let g = load_geometry(pixel);
  var r = EMPTY_GI;
  if ((u.flags.w & 2u) != 0u && g.valid) {
    var rng = rng_for(index, 4u);
    let dir = cosine_direction(g.nrm, &rng);
    let max_dist = u.sun_and_max_distance.w;
    let origin = g.pos + g.nrm * ray_bias();
    let hit = trace(origin, dir, max_dist, false);
    var pos = g.pos + dir * max_dist;
    var nrm = -dir;
    var radiance = safe_rgb(sky(dir));
    var hit_albedo_ddgi = 0.0;
    if (hit.hit) {
      pos = origin + dir * hit.t;
      nrm = select(-hit.world_normal, hit.world_normal, dot(hit.world_normal, dir) < 0.0);
      let shaded = shade_bounce(pos, nrm, hit, &rng);
      radiance = shaded.rgb;
      let diff_albedo = clamp(hit.albedo * (1.0 - hit.metallic), vec3f(0.0), vec3f(1.0));
      // __RESTIR_SHARC_BOUNCE__
      hit_albedo_ddgi = bitcast<f32>(pack4x8unorm(vec4f(diff_albedo, shaded.a)));
    }
    // A one-candidate reservoir uses the inverse cosine sampling density.
    // The luminance-weighted target enters w_sum and cancels when M = 1.
    let pdf = max(dot(g.nrm, dir) / PI, EPS);
    r = GiReservoir(vec4f(pos, 1.0), vec4f(nrm, 1.0 / pdf), vec4f(radiance, 0.0),
                    vec4f(1.0, select(max_dist, hit.t, hit.hit), hit_albedo_ddgi, 0.0));
    r.radiance_wsum.w = gi_target(g, r) / pdf;
  }
  gi_output[index] = r;
}

@compute @workgroup_size(8, 8, 1)
fn restir_gi_temporal(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= extent())) { return; }
  let pixel = vec2i(id.xy);
  let index = pixel_index(pixel);
  let g = load_geometry(pixel);
  let current = gi_input[index];
  var result = current;
  if (g.valid && u.flags.y != 0u && current.sample_pos_m.w > 0.0) {
    let prev = history_pixel(pixel, g);
    if (in_bounds(prev)) {
      let old = gi_prev[pixel_index(prev)];
      let m_old = min(old.sample_pos_m.w, 24.0);
      var rng = rng_for(index, 5u);
      result = EMPTY_GI;
      gi_add(&result, current, current.sample_pos_m.w, current.radiance_wsum.w, &rng);
      gi_add(&result, old, m_old, gi_target(g, old) * old.sample_nrm_w.w * m_old, &rng);
      result = gi_finish(result, g);
    }
  }
  gi_output[index] = result;
}

fn gi_jacobian(neighbor: Geometry, receiver: Geometry, sample: GiReservoir) -> f32 {
  if (sample.vis_pad.y >= u.sun_and_max_distance.w) { return 1.0; }
  let q = receiver.pos - sample.sample_pos_m.xyz;
  let r = neighbor.pos - sample.sample_pos_m.xyz;
  let dq2 = dot(q, q);
  let dr2 = dot(r, r);
  let cos_q = abs(dot(q, sample.sample_nrm_w.xyz)) / max(sqrt(dq2), EPS);
  let cos_r = abs(dot(r, sample.sample_nrm_w.xyz)) / max(sqrt(dr2), EPS);
  return clamp(cos_q / max(cos_r, 0.05) * dr2 / max(dq2, 0.01), 0.1, 10.0);
}

fn gi_sample_visible(g: Geometry, sample: GiReservoir) -> bool {
  let bias = ray_bias();
  let origin = g.pos + g.nrm * bias;
  let max_dist = u.sun_and_max_distance.w;
  if (sample.vis_pad.y >= max_dist) {
    let dir = normalize(sample.sample_pos_m.xyz - g.pos);
    return !trace(origin, dir, max_dist, true).hit;
  }
  // Offset the endpoint onto the radiating side of the hit surface.
  let endpoint = sample.sample_pos_m.xyz + sample.sample_nrm_w.xyz * min(bias, 0.01);
  let delta = endpoint - origin;
  let distance = length(delta);
  if (distance <= RAY_MIN) { return true; }
  return !trace(origin, delta / distance, distance, true).hit;
}

@compute @workgroup_size(8, 8, 1)
fn restir_gi_spatial(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= extent())) { return; }
  let pixel = vec2i(id.xy);
  let index = pixel_index(pixel);
  let g = load_geometry(pixel);
  let center = gi_input[index];
  var rgb = vec3f(0.0);
  if (g.valid && center.sample_pos_m.w > 0.0) {
    var result = EMPTY_GI;
    var rng = rng_for(index, 6u);
    let rotation = random(&rng) * TAU;
    let p_c_at_c = gi_target(g, center);
    let m_center = center.sample_pos_m.w / 4.0;
    var center_mass = 0.0;
    for (var i = 0u; i < 4u; i++) {
      let q = neighbor_pixel(pixel, rotation, i, 24.0, &rng);
      if (in_bounds(q) && any(q != pixel)) {
        let qg = load_geometry(q);
        let other = gi_input[pixel_index(q)];
        let m_other = other.sample_pos_m.w;
        if (qg.valid && similar(g, qg.distance, qg.nrm) && m_other > 0.0) {
          let p_q_at_q = gi_target(qg, other);
          let p_q_at_c = gi_target(g, other) * gi_jacobian(qg, g, other);
          let p_c_at_q = gi_target(qg, center) * gi_jacobian(g, qg, center);
          let neighbor_mass = pairwise_mass(p_q_at_q, p_q_at_c, m_other, m_center);
          center_mass += pairwise_mass(p_c_at_c, p_c_at_q, m_center, m_other);
          // Zero-radiance reservoirs still represent draws. Only their selection mass is zero.
          gi_add(&result, other, m_other, p_q_at_c * other.sample_nrm_w.w * neighbor_mass, &rng);
          continue;
        }
      }
      center_mass += m_center;
    }
    gi_add(&result, center, center.sample_pos_m.w, p_c_at_c * center.sample_nrm_w.w * center_mass, &rng);
    result = gi_finish(result, g);
    let weight = finite_nonnegative(result.sample_nrm_w.w, 64.0);
    if (result.vis_pad.x > 0.5 && weight > 0.0 && gi_sample_visible(g, result)) {
      rgb = safe_rgb(result.radiance_wsum.rgb * (gi_cosine(g, result) * weight / PI));
    }
  }
  // __RESTIR_SHARC_DEPOSIT__
  textureStore(out_gi, pixel, vec4f(rgb, f32(g.valid && (u.flags.w & 2u) != 0u)));
}

@compute @workgroup_size(8, 8, 1)
fn restir_store_geometry(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= extent())) { return; }
  let pixel = vec2i(id.xy);
  let g = load_geometry(pixel);
  var packed = vec4f(0.0);
  if (g.valid && g.distance <= 65504.0) {
    packed = vec4f(g.distance, rt_ddgi_oct_encode(g.nrm), 1.0);
  }
  textureStore(next_depth_nrm, pixel, packed);
  if ((u.flags.w & 1u) == 0u) { textureStore(out_di, pixel, vec4f(0.0)); }
  if ((u.flags.w & 2u) == 0u) { textureStore(out_gi, pixel, vec4f(0.0)); }
}
