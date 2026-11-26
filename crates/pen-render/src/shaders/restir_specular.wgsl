// Appended to Pen's TLAS, ReSTIR lighting, and read-only SHaRC helpers.
struct SpecTransforms {
  view_projection: mat4x4f,
  inverse_prev_view_projection: mat4x4f,
};

struct SpecReservoir {
  hit_pos_m: vec4f,       // World position or sky direction, and candidate count.
  hit_nrm_w: vec4f,       // Secondary normal and normalized contribution weight.
  radiance_wsum: vec4f,   // Incoming radiance and scalar resampling mass.
  sample_meta: vec4f,            // Hit distance, surface flag, source roughness, reuse flag.
};

@group(0) @binding(21) var<uniform> spec_transforms: SpecTransforms;
@group(0) @binding(22) var<uniform> spec_lights: array<RtLocalLight, 256>;
@group(0) @binding(23) var spec_prev_geometry: texture_2d<f32>;
@group(0) @binding(24) var spec_next_geometry: texture_storage_2d<rgba32float, write>;
@group(0) @binding(25) var<storage, read> spec_input: array<SpecReservoir>;
@group(0) @binding(26) var<storage, read> spec_prev: array<SpecReservoir>;
@group(0) @binding(27) var<storage, read_write> spec_output: array<SpecReservoir>;
@group(0) @binding(28) var out_specular: texture_storage_2d<rgba16float, write>;
@group(0) @binding(29) var out_motion: texture_storage_2d<rgba16float, write>;
@group(0) @binding(30) var<uniform> sharc_u: SharcUniform;
@group(0) @binding(31) var<storage, read> sharc_keys: array<u32>;
@group(0) @binding(32) var<storage, read> sharc_resolved: array<vec4f>;

const SPEC_EMPTY: SpecReservoir = SpecReservoir(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
const SPEC_VALIDATE_VISIBILITY: bool = true;

struct SpecSurface {
  pos: vec3f,
  nrm: vec3f,
  view: vec3f,
  roughness: f32,
  f0: vec3f,
  depth: f32,
  valid: bool,
};

fn spec_clip(uv: vec2f, depth: f32) -> vec4f {
  return vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), depth, 1.0);
}

fn spec_gbuffer_uv(pixel: vec2i, dims: vec2i) -> vec2f {
  return (vec2f(input_pixel(pixel, dims)) + 0.5) / vec2f(dims);
}

fn spec_load_surface(pixel: vec2i) -> SpecSurface {
  let dims = vec2i(textureDimensions(normal_roughness, 0));
  let g_px = input_pixel(pixel, dims);
  let depth = textureLoad(depth_tex, g_px, 0).x;
  if (!(abs(depth - u.camera_and_clear.w) > EPS)) {
    return SpecSurface(vec3f(0.0), vec3f(0.0), vec3f(0.0), 1.0, vec3f(0.0), 0.0, false);
  }
  let nr = textureLoad(normal_roughness, g_px, 0);
  if (!(dot(nr.xyz, nr.xyz) >= 0.01 && nr.w >= 0.0 && nr.w <= u.sky_horizon.w)) {
    return SpecSurface(vec3f(0.0), vec3f(0.0), vec3f(0.0), 1.0, vec3f(0.0), 0.0, false);
  }
  let g_uv = (vec2f(g_px) + 0.5) / vec2f(dims);
  let world = u.inverse_view_projection * spec_clip(g_uv, depth);
  if (!(abs(world.w) > 1e-8)) {
    return SpecSurface(vec3f(0.0), vec3f(0.0), vec3f(0.0), 1.0, vec3f(0.0), 0.0, false);
  }
  let pos = world.xyz / world.w;
  let nrm = normalize(nr.xyz);
  let view = normalize(u.camera_and_clear.xyz - pos);
  let mat = textureLoad(albedo_metallic, g_px, 0);
  let f0 = mix(vec3f(0.04), clamp(mat.rgb, vec3f(0.0), vec3f(1.0)), clamp(mat.a, 0.0, 1.0));
  return SpecSurface(pos, nrm, view, clamp(nr.w, 0.04, 1.0), f0, depth, dot(nrm, view) > EPS);
}

// Heitz 2018, JCGT 7(4), sampling the projected area of the stretched hemisphere.
fn spec_vndf(n: vec3f, v: vec3f, alpha: f32, xi: vec2f) -> vec3f {
  let up = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(n.z) > 0.99);
  let tangent = normalize(cross(up, n));
  let bitangent = cross(n, tangent);
  let vh = normalize(vec3f(alpha * dot(v, tangent), alpha * dot(v, bitangent), dot(v, n)));
  let lensq = dot(vh.xy, vh.xy);
  var t1 = vec3f(1.0, 0.0, 0.0);
  if (lensq > 0.0) {
    t1 = vec3f(-vh.y, vh.x, 0.0) * inverseSqrt(lensq);
  }
  let t2 = cross(vh, t1);
  let r = sqrt(xi.x);
  let phi = TAU * xi.y;
  let x = r * cos(phi);
  let y = mix(sqrt(max(1.0 - x * x, 0.0)), r * sin(phi), 0.5 * (1.0 + vh.z));
  let nh = x * t1 + y * t2 + sqrt(max(1.0 - x * x - y * y, 0.0)) * vh;
  let h = normalize(vec3f(alpha * nh.xy, max(nh.z, 0.0)));
  return normalize(tangent * h.x + bitangent * h.y + n * h.z);
}

fn spec_d(n: vec3f, h: vec3f, alpha2: f32) -> f32 {
  // The cross product avoids cancellation near the peak of very smooth GGX lobes.
  let sin_h = cross(n, h);
  let nh = clamp(dot(n, h), 0.0, 1.0);
  let d = dot(sin_h, sin_h) + alpha2 * nh * nh;
  return alpha2 / (PI * d * d);
}

// G1(cosine) / cosine, with the common factor of cosine cancelled.
fn spec_g1_over_cos(cosine: f32, alpha2: f32) -> f32 {
  return 2.0 / (cosine + sqrt(alpha2 + (1.0 - alpha2) * cosine * cosine));
}

// GGX BRDF times the receiver cosine.
fn spec_response(n: vec3f, v: vec3f, l: vec3f, f0: vec3f, roughness: f32) -> vec3f {
  let nl = dot(n, l);
  let nv = dot(n, v);
  if (!(nl > 0.0 && nv > 0.0)) { return vec3f(0.0); }
  let h = normalize(v + l);
  let one_minus_vh = 1.0 - clamp(dot(v, h), 0.0, 1.0);
  let fresnel = mix(f0, vec3f(1.0), pow(one_minus_vh, 5.0));
  let alpha = roughness * roughness;
  let alpha2 = alpha * alpha;
  let masking = spec_g1_over_cos(nv, alpha2) * (spec_g1_over_cos(nl, alpha2) * nl);
  return fresnel * (0.25 * spec_d(n, h, alpha2) * masking);
}

fn spec_hit_radiance(pos: vec3f, n: vec3f, v: vec3f, hit: TlasHit, rng: ptr<function, u32>) -> vec3f {
  let albedo = clamp(hit.albedo, vec3f(0.0), vec3f(1.0));
  let metal = clamp(hit.metallic, 0.0, 1.0);
  let diffuse_albedo = albedo * (1.0 - metal);
  let diffuse = diffuse_albedo / PI;
  let f0 = mix(vec3f(0.04), albedo, metal);
  let rough = clamp(hit.roughness, 0.04, 1.0);
  let sun = u.sun_and_max_distance.xyz;
  let nl = dot(n, sun);
  var radiance = hit.emissive;
  if (nl > 0.0 && luma(u.sun_color_and_bias.rgb) > 0.0 &&
      visible(pos, n, pos + sun * u.sun_and_max_distance.w)) {
    radiance += (diffuse * nl + spec_response(n, v, sun, f0, rough)) * u.sun_color_and_bias.rgb;
  }
  let count = u.size_lights_frame.z;
  if (count > 0u) {
    let candidates = min(8u, count);
    let offset = u32(random(rng) * f32(count));
    let scale = f32(count) / f32(candidates);
    for (var i = 0u; i < candidates; i++) {
      let light = spec_lights[stratified_light(i, count, candidates, offset, rng)];
      let incident = local_light(pos, n, light);
      if (incident.cosine > 0.0 && luma(incident.rgb) > 0.0 &&
          visible(pos, n, light.position_range.xyz)) {
        let response = spec_response(n, v, incident.dir, f0, rough);
        radiance += (diffuse * incident.cosine + response) * (incident.rgb * scale);
      }
    }
  }
  if (u.sky_zenith_and_bounce.w > 0.0) {
    let ddgi = rt_sample_ddgi(pos, n);
    let coverage = clamp(ddgi.a, 0.0, 1.0);
    var indirect = ddgi.rgb * coverage;
    if (u.flags.y != 0u && coverage < 1.0) {
      let cached = sharc_query_ro(pos, n);
      if (cached.a > 0.5) {
        indirect += cached.rgb * (1.0 - coverage);
      }
    }
    radiance += diffuse_albedo * indirect * u.sky_zenith_and_bounce.w;
  }
  return safe_rgb(radiance);
}

fn spec_direction(g: SpecSurface, r: SpecReservoir) -> vec3f {
  if (r.sample_meta.y == 0.0) { return r.hit_pos_m.xyz; }
  let delta = r.hit_pos_m.xyz - g.pos;
  return delta / max(length(delta), EPS);
}

fn spec_integrand(g: SpecSurface, r: SpecReservoir) -> vec3f {
  let l = spec_direction(g, r);
  return spec_response(g.nrm, g.view, l, g.f0, g.roughness) * r.radiance_wsum.rgb;
}

fn spec_target(g: SpecSurface, r: SpecReservoir) -> f32 {
  return finite_nonnegative(luma(spec_integrand(g, r)), 1e12);
}

fn spec_jacobian(source_pos: vec3f, receiver: SpecSurface, r: SpecReservoir) -> f32 {
  if (!(dot(receiver.nrm, spec_direction(receiver, r)) > 0.0)) { return 0.0; }
  if (r.sample_meta.y == 0.0) { return 1.0; }

  let from_source = source_pos - r.hit_pos_m.xyz;
  let from_receiver = receiver.pos - r.hit_pos_m.xyz;
  let dr2 = dot(from_source, from_source);
  let dq2 = dot(from_receiver, from_receiver);
  let cos_r = dot(from_source, r.hit_nrm_w.xyz) / max(sqrt(dr2), EPS);
  let cos_q = dot(from_receiver, r.hit_nrm_w.xyz) / max(sqrt(dq2), EPS);
  if (!(cos_q > 0.0 && cos_r > 0.0)) { return 0.0; }
  let j = (cos_q / max(cos_r, 1e-4)) * (dr2 / max(dq2, 1e-4));
  return select(0.0, j, j >= 0.1 && j <= 10.0);
}

fn spec_uv(clip: vec4f) -> vec2f {
  return clip.xy / clip.w * vec2f(0.5, -0.5) + 0.5;
}

fn spec_motion(pixel: vec2i, g: SpecSurface, r: SpecReservoir) -> vec4f {
  let prev_clip = u.prev_view_projection * vec4f(g.pos, 1.0);
  let uv = spec_gbuffer_uv(pixel, vec2i(textureDimensions(normal_roughness, 0)));
  var static_motion = vec2f(0.0);
  if (prev_clip.w > EPS) {
    let mv = uv - spec_uv(prev_clip);
    if (all(abs(mv) < vec2f(100.0))) { static_motion = mv; }
  }
  var surface_motion = static_motion;
  if (u.flags.z != 0u) {
    let motion_px = input_pixel(pixel, vec2i(textureDimensions(motion_tex, 0)));
    let mv = textureLoad(motion_tex, motion_px, 0).xy;
    if (all(abs(mv) < vec2f(100.0))) { surface_motion = mv; }
  }
  var parallax_motion = surface_motion;
  if (r.sample_meta.x > 0.0 && prev_clip.w > EPS) {
    // Reflect the hit across the tangent plane to project the virtual image.
    let virtual_pos = g.pos + reflect(spec_direction(g, r), g.nrm) * r.sample_meta.x;
    let curr_virtual = spec_transforms.view_projection * vec4f(virtual_pos, 1.0);
    let prev_virtual = u.prev_view_projection * vec4f(virtual_pos, 1.0);
    if (min(curr_virtual.w, prev_virtual.w) > EPS) {
      let mv = spec_uv(curr_virtual) - spec_uv(prev_virtual);
      if (all(abs(mv) < vec2f(100.0))) {
        parallax_motion = surface_motion + mv - static_motion;
      }
    }
  }
  return vec4f(surface_motion, parallax_motion);
}

fn spec_compatible(g: SpecSurface, q_pos: vec3f, q_nrm: vec3f, q_rough: f32, camera: vec3f) -> bool {
  let rough = min(g.roughness, q_rough);
  let t = smoothstep(0.04, 0.45, rough);
  let depth = length(g.pos - camera);
  let q_depth = length(q_pos - camera);
  return abs(g.roughness - q_rough) <= mix(0.025, 0.12, t) &&
      dot(g.nrm, q_nrm) >= mix(0.995, 0.90, t) &&
      abs(depth - q_depth) <= mix(0.012, 0.075, t) * max(depth, 0.1) &&
      abs(dot(q_pos - g.pos, g.nrm)) <= mix(0.004, 0.025, t) * max(depth, 0.1);
}

fn spec_stream(result: ptr<function, SpecReservoir>, candidate: SpecReservoir, mass: f32,
               reused: bool, rng: ptr<function, u32>) {
  let sum = (*result).radiance_wsum.w + mass;
  let m = (*result).hit_pos_m.w;
  if (mass > 0.0 && random(rng) * sum < mass) {
    *result = candidate;
    if (reused) { (*result).sample_meta.w = 1.0; }
  }
  (*result).radiance_wsum.w = sum;
  (*result).hit_pos_m.w = m;
}

fn spec_finish(result: SpecReservoir, g: SpecSurface) -> SpecReservoir {
  var r = result;
  let p = spec_target(g, r);
  r.hit_nrm_w.w = 0.0;
  if (p > 0.0 && r.hit_pos_m.w > 0.0) {
    r.hit_nrm_w.w = finite_nonnegative(r.radiance_wsum.w / (r.hit_pos_m.w * p), 1e12);
  }
  r.sample_meta.z = g.roughness;
  if (r.sample_meta.y > 0.0) {
    r.sample_meta.x = length(r.hit_pos_m.xyz - g.pos);
  }
  return r;
}

fn spec_trace(origin: vec3f, l: vec3f, limit: f32, any_hit: bool) -> TlasHit {
  return trace_tlas(origin, l, ray_bias(), limit, u.scene_counts_and_mask.w,
                    any_hit, u.scene_counts_and_mask.x, u.scene_counts_and_mask.y, u.scene_counts_and_mask.z);
}

@compute @workgroup_size(8, 8, 1)
fn spec_initial(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= extent())) { return; }
  let pixel = vec2i(id.xy);
  let index = pixel_index(pixel);
  let g = spec_load_surface(pixel);
  var r = SPEC_EMPTY;
  if (g.valid) {
    // Count a valid draw even if it lands below the horizon or sees zero radiance.
    r.hit_pos_m.w = 1.0;
    r.sample_meta.z = g.roughness;
    var rng = rng_for(index, 101u);
    let alpha = g.roughness * g.roughness;
    let xi = vec2f(random(&rng), random(&rng));
    let h = spec_vndf(g.nrm, g.view, alpha, xi);
    let l = reflect(-g.view, h);
    if (dot(g.nrm, l) > 0.0) {
      let hit = spec_trace(g.pos, l, u.sun_and_max_distance.w, false);
      r.hit_pos_m = vec4f(l, 1.0);
      if (hit.hit) {
        let pos = g.pos + l * hit.t;
        let n = select(-hit.world_normal, hit.world_normal, dot(hit.world_normal, l) < 0.0);
        r.hit_pos_m = vec4f(pos, 1.0);
        r.hit_nrm_w = vec4f(n, 0.0);
        r.radiance_wsum = vec4f(spec_hit_radiance(pos, n, -l, hit, &rng), 0.0);
        r.sample_meta.x = length(pos - g.pos);
        r.sample_meta.y = 1.0;
      } else {
        r.radiance_wsum = vec4f(safe_rgb(sky(l)), 0.0);
      }

      // The initial RIS mass is p-hat / pdf; 0.25 * D(n, h) * G1(nv) / nv cancels
      // algebraically, avoiding 0/0 on mirror lobes and skipping spec_d + G1(nv).
      let nl = dot(g.nrm, l);
      let one_minus_vh = 1.0 - clamp(dot(g.view, h), 0.0, 1.0);
      let fresnel = mix(g.f0, vec3f(1.0), pow(one_minus_vh, 5.0));
      let weight_rgb = fresnel * (spec_g1_over_cos(nl, alpha * alpha) * nl) * r.radiance_wsum.rgb;
      let mass = finite_nonnegative(luma(weight_rgb), 1e12);
      r.radiance_wsum.w = mass;
      r.hit_nrm_w.w = mass;
    }
  }
  spec_output[index] = r;
}

@compute @workgroup_size(8, 8, 1)
fn spec_temporal(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= extent())) { return; }
  let pixel = vec2i(id.xy);
  let index = pixel_index(pixel);
  let g = spec_load_surface(pixel);
  if (!g.valid) {
    spec_output[index] = SPEC_EMPTY;
    textureStore(spec_next_geometry, pixel, vec4f(0.0));
    return;
  }

  textureStore(spec_next_geometry, pixel, vec4f(g.depth, rt_ddgi_oct_encode(g.nrm), g.roughness));
  var result = spec_input[index];
  if (u.flags.x != 0u) {
    let mv = spec_motion(pixel, g, result);
    let mirror = 1.0 - smoothstep(0.04, 0.45, g.roughness);
    let prev_uv = (vec2f(pixel) + 0.5) / vec2f(extent()) - mix(mv.xy, mv.zw, mirror);
    if (all(prev_uv >= vec2f(0.0)) && all(prev_uv < vec2f(1.0))) {
      let prev_px = vec2i(prev_uv * vec2f(extent()));
      let packed = textureLoad(spec_prev_geometry, prev_px, 0);
      if (packed.w > 0.0) {
        let prev_g_uv = spec_gbuffer_uv(prev_px, vec2i(textureDimensions(normal_roughness, 0)));
        let prev_world = spec_transforms.inverse_prev_view_projection * spec_clip(prev_g_uv, packed.x);
        if (abs(prev_world.w) > EPS) {
          let prev_pos = prev_world.xyz / prev_world.w;
          if (spec_compatible(g, prev_pos, oct_decode(packed.yz), packed.w, u.previous_camera.xyz)) {
            let prev = spec_prev[pixel_index(prev_px)];
            let j = spec_jacobian(prev_pos, g, prev);
            let cap = mix(4.0, 16.0, smoothstep(0.04, 0.45, g.roughness));
            let m = min(prev.hit_pos_m.w, cap - 1.0);
            if (prev.radiance_wsum.w == 0.0 || j > 0.0) {
              var rng = rng_for(index, 102u);
              result.hit_pos_m.w += m;
              let mass = finite_nonnegative(spec_target(g, prev) * prev.hit_nrm_w.w * m * j, 1e12);
              spec_stream(&result, prev, mass, true, &rng);
            }
          }
        }
      }
    }
  }
  spec_output[index] = spec_finish(result, g);
}

fn spec_trace_any(origin: vec3f, l: vec3f, limit: f32) -> bool {
  return trace_tlas_any_hit(origin, l, ray_bias(), limit, u.scene_counts_and_mask.w,
                            u.scene_counts_and_mask.x, u.scene_counts_and_mask.y, u.scene_counts_and_mask.z);
}

fn spec_visible(g: SpecSurface, r: SpecReservoir) -> bool {
  if (r.sample_meta.y == 0.0) {
    return !spec_trace_any(g.pos, r.hit_pos_m.xyz, u.sun_and_max_distance.w);
  }
  let delta = r.hit_pos_m.xyz - g.pos;
  let dist = length(delta);
  // Stop just short of the hit, keeping the segment on the actual connection.
  let limit = dist - max(2.0 * RAY_MIN, dist * 1e-6);
  return limit <= ray_bias() || !spec_trace_any(g.pos, delta / dist, limit);
}

@compute @workgroup_size(8, 8, 1)
fn spec_spatial_resolve(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= extent())) { return; }
  let pixel = vec2i(id.xy);
  let index = pixel_index(pixel);
  let g = spec_load_surface(pixel);
  if (!g.valid) {
    spec_output[index] = SPEC_EMPTY;
    textureStore(out_specular, pixel, vec4f(0.0));
    textureStore(out_motion, pixel, vec4f(0.0));
    return;
  }

  let center = spec_input[index];
  let center_p = spec_target(g, center);
  let dist = select(u.sun_and_max_distance.w, center.sample_meta.x, center.sample_meta.y > 0.0);
  let cone = g.roughness * g.roughness * clamp(dist, 0.02, 20.0);
  let radius = mix(0.45, 10.0, smoothstep(0.04, 0.5, g.roughness) * smoothstep(0.01, 0.65, cone));
  var rng = rng_for(index, 103u);
  let rotation = random(&rng) * TAU;
  let center_m = center.hit_pos_m.w / 5.0;
  var center_mass = 0.0;
  var result = center;
  result.radiance_wsum.w = 0.0;
  var visited: array<vec2i, 5>;
  for (var i = 0u; i < 5u; i++) {
    let angle = rotation + f32(i) * (TAU / 5.0);
    let offset = vec2i(round(vec2f(cos(angle), sin(angle)) * radius * sqrt(random(&rng))));
    let q = pixel + offset;
    visited[i] = q;
    var duplicate = false;
    for (var j = 0u; j < i; j++) {
      duplicate = duplicate || all(visited[j] == q);
    }
    if (!in_bounds(q) || all(q == pixel) || duplicate) {
      center_mass += center_m;
      continue;
    }

    let other = spec_input[pixel_index(q)];
    if (other.hit_pos_m.w <= 0.0) {
      center_mass += center_m;
      continue;
    }
    let qg = spec_load_surface(q);
    if (!qg.valid ||
        !spec_compatible(g, qg.pos, qg.nrm, other.sample_meta.z, u.camera_and_clear.xyz)) {
      center_mass += center_m;
      continue;
    }

    let jacobian = spec_jacobian(qg.pos, g, other);
    if (other.radiance_wsum.w > 0.0 && jacobian == 0.0) {
      center_mass += center_m;
      continue;
    }
    let other_m = other.hit_pos_m.w;
    let reverse_j = spec_jacobian(g.pos, qg, center);
    // Compare target densities in each sample's original solid-angle measure.
    let other_p = spec_target(g, other);
    let neighbor_mass = pairwise_mass(spec_target(qg, other), other_p * jacobian, other_m, center_m);
    center_mass += pairwise_mass(center_p, spec_target(qg, center) * reverse_j, center_m, other_m);
    result.hit_pos_m.w += other_m;
    let mass = finite_nonnegative(other.hit_nrm_w.w * other_p * jacobian * neighbor_mass, 1e12);
    spec_stream(&result, other, mass, true, &rng);
  }
  spec_stream(&result, center, center.hit_nrm_w.w * center_p * center_mass, false, &rng);
  result = spec_finish(result, g);

  var rgb = vec3f(0.0);
  if (SPEC_VALIDATE_VISIBILITY && result.sample_meta.w != 0.0 && !spec_visible(g, result)) {
    result = SPEC_EMPTY;
  } else {
    rgb = safe_rgb(spec_integrand(g, result) * result.hit_nrm_w.w);
  }
  spec_output[index] = result;
  textureStore(out_specular, pixel, vec4f(rgb, finite_nonnegative(result.sample_meta.x, 65504.0)));
  textureStore(out_motion, pixel, spec_motion(pixel, g, result));
}
