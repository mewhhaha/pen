// Physically Based Deferred Decals (albedo + tangent-space normal + ORM roughness/metallic/AO + emissive).
// Reconstructs world position and geometric normal per-sample (MSAA-aware) from the scene depth buffer,
// derives an exact decal-aligned (+U/+V) tangent basis, and evaluates GGX Cook-Torrance shading.

struct View {
  view_projection: mat4x4f,
  inverse_view_projection: mat4x4f,
  camera_pos_and_clear: vec4f, // xyz = camera world pos, w = depth clear value (1.0 standard, 0.0 reversed-Z)
  sun_dir_and_unused: vec4f,   // xyz = normalized toward-sun dir, w = 0
  sun_color_intensity: vec4f,  // rgb = sun radiance, w = ambient intensity
  sky_ambient: vec4f,          // rgb = upper hemisphere ambient, w = viewport origin x
  ground_ambient: vec4f,       // rgb = lower hemisphere ambient, w = viewport origin y
  viewport_size: vec4f,        // xy = viewport width/height (<=0 falls back to depth dims), zw = unused
};

struct PbrDecal {
  model: mat4x4f,
  inverse: mat4x4f,
  albedo_opacity: vec4f,       // rgb = linear tint, a = opacity
  emissive_normal: vec4f,      // rgb = linear emissive, w = normal_strength
  pbr_params: vec4f,           // x = roughness_scale, y = metallic_scale, z = angle_fade_min_dot, w = y_fade_start
};

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> decals: array<PbrDecal>;
@group(0) @binding(2) var depth_tex: ${depth_texture};
@group(0) @binding(3) var albedo_tex: texture_2d<f32>;
@group(0) @binding(4) var normal_tex: texture_2d<f32>;
@group(0) @binding(5) var orm_tex: texture_2d<f32>;
@group(0) @binding(6) var decal_sampler: sampler;

const PI: f32 = 3.14159265359;

const CORNERS = array<vec3f, 8>(
  vec3f(-1.0, -1.0, -1.0), vec3f(1.0, -1.0, -1.0), vec3f(-1.0, 1.0, -1.0), vec3f(1.0, 1.0, -1.0),
  vec3f(-1.0, -1.0, 1.0), vec3f(1.0, -1.0, 1.0), vec3f(-1.0, 1.0, 1.0), vec3f(1.0, 1.0, 1.0),
);
const FACES = array<u32, 36>(
  0u, 4u, 6u, 0u, 6u, 2u, 1u, 3u, 7u, 1u, 7u, 5u,
  0u, 1u, 5u, 0u, 5u, 4u, 2u, 6u, 7u, 2u, 7u, 3u,
  0u, 2u, 3u, 0u, 3u, 1u, 4u, 5u, 7u, 4u, 7u, 6u,
);

struct BoxOut {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) index: u32,
};

@vertex fn vertex(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> BoxOut {
  let corner = CORNERS[FACES[vid]];
  return BoxOut(view.view_projection * decals[iid].model * vec4f(corner, 1.0), iid);
}

struct DepthSample {
  world: vec3f,
  valid: bool,
};

fn load_world_sample(pixel: vec2i, sample_idx: i32) -> DepthSample {
  let dims = vec2i(textureDimensions(depth_tex));
  if (pixel.x < 0 || pixel.y < 0 || pixel.x >= dims.x || pixel.y >= dims.y) {
    return DepthSample(vec3f(0.0), false);
  }
  let d = textureLoad(depth_tex, pixel, sample_idx);
  if (abs(d - view.camera_pos_and_clear.w) <= 1e-6) {
    return DepthSample(vec3f(0.0), false);
  }
  let vp_origin = vec2f(view.sky_ambient.w, view.ground_ambient.w);
  let vp_size = select(vec2f(dims), view.viewport_size.xy, view.viewport_size.x > 0.5 && view.viewport_size.y > 0.5);
  let local_px = (vec2f(pixel) + vec2f(0.5)) - vp_origin;
  let ndc = vec2f(
    local_px.x / max(vp_size.x, 1.0) * 2.0 - 1.0,
    1.0 - local_px.y / max(vp_size.y, 1.0) * 2.0,
  );
  let clip = view.inverse_view_projection * vec4f(ndc, d, 1.0);
  if (abs(clip.w) <= 1e-6) {
    return DepthSample(vec3f(0.0), false);
  }
  return DepthSample(clip.xyz / clip.w, true);
}

fn ggx_d(ndoth: f32, a2: f32) -> f32 {
  let denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
  return a2 / max(PI * denom * denom, 1e-6);
}

fn smith_g(ndotv: f32, ndotl: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) * 0.125;
  let gv = ndotv / max(ndotv * (1.0 - k) + k, 1e-5);
  let gl = ndotl / max(ndotl * (1.0 - k) + k, 1e-5);
  return gv * gl;
}

fn shade_pbr_decal(in: BoxOut, sample_idx: i32) -> vec4f {
  let decal = decals[in.index];
  let pixel = vec2i(in.position.xy);
  let center = load_world_sample(pixel, sample_idx);
  if (!center.valid) {
    discard;
  }
  let world_pos = center.world;
  let local = (decal.inverse * vec4f(world_pos, 1.0)).xyz;
  if (any(abs(local) > vec3f(1.0))) {
    discard;
  }

  // Reconstruct geometric surface normal using only valid in-bounds non-sky neighbors.
  let sr = load_world_sample(pixel + vec2i(1, 0), sample_idx);
  let sl = load_world_sample(pixel - vec2i(1, 0), sample_idx);
  let sd = load_world_sample(pixel + vec2i(0, 1), sample_idx);
  let su = load_world_sample(pixel - vec2i(0, 1), sample_idx);

  let dx_r = sr.world - world_pos;
  let dx_l = world_pos - sl.world;
  let dy_d = sd.world - world_pos;
  let dy_u = world_pos - su.world;

  var has_dx = true;
  var dx = vec3f(0.0);
  if (sr.valid && sl.valid) {
    dx = select(dx_l, dx_r, dot(dx_r, dx_r) < dot(dx_l, dx_l));
  } else if (sr.valid) {
    dx = dx_r;
  } else if (sl.valid) {
    dx = dx_l;
  } else {
    has_dx = false;
  }

  var has_dy = true;
  var dy = vec3f(0.0);
  if (sd.valid && su.valid) {
    dy = select(dy_u, dy_d, dot(dy_d, dy_d) < dot(dy_u, dy_u));
  } else if (sd.valid) {
    dy = dy_d;
  } else if (su.valid) {
    dy = dy_u;
  } else {
    has_dy = false;
  }

  let decal_up_raw = (decal.model * vec4f(0.0, 1.0, 0.0, 0.0)).xyz;
  let decal_up = select(vec3f(0.0, 1.0, 0.0), normalize(decal_up_raw), length(decal_up_raw) > 1e-6);
  var n_geo = decal_up;
  if (has_dx && has_dy) {
    let raw_n = cross(dx, dy);
    let n_len = length(raw_n);
    if (n_len > 1e-6) {
      n_geo = raw_n / n_len;
    }
  }
  let view_raw = view.camera_pos_and_clear.xyz - world_pos;
  let view_dir = select(n_geo, normalize(view_raw), length(view_raw) > 1e-6);
  if (dot(n_geo, view_dir) < 0.0) {
    n_geo = -n_geo;
  }

  let angle_min = decal.pbr_params.z;
  let angle_dot = abs(dot(n_geo, decal_up));
  let angle_fade = smoothstep(angle_min, min(angle_min + 0.18, 1.0), angle_dot);
  let y_fade_start = clamp(decal.pbr_params.w, 0.0, 0.98);
  let y_fade = 1.0 - smoothstep(y_fade_start, 1.0, abs(local.y));

  let uv = local.xz * 0.5 + 0.5;
  let albedo_sample = textureSampleLevel(albedo_tex, decal_sampler, uv, 0.0);
  let cover = albedo_sample.a * decal.albedo_opacity.a * y_fade * angle_fade;
  if (cover <= 0.001) {
    discard;
  }

  // Orthonormal tangent basis aligned to UV +U (decal +X) and UV +V (decal +Z).
  let decal_right_raw = (decal.model * vec4f(1.0, 0.0, 0.0, 0.0)).xyz;
  let decal_fwd_raw = (decal.model * vec4f(0.0, 0.0, 1.0, 0.0)).xyz;
  let decal_right = select(vec3f(1.0, 0.0, 0.0), normalize(decal_right_raw), length(decal_right_raw) > 1e-6);
  let decal_fwd = select(vec3f(0.0, 0.0, 1.0), normalize(decal_fwd_raw), length(decal_fwd_raw) > 1e-6);

  let t_raw = decal_right - n_geo * dot(decal_right, n_geo);
  let t_len = length(t_raw);
  let fallback_axis = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(n_geo.z) > 0.9);
  let tangent = select(normalize(cross(fallback_axis, n_geo)), t_raw / t_len, t_len > 1e-5);
  // Orient bitangent so +V aligns with decal +Z (local.z), matching `uv = local.xz * 0.5 + 0.5`.
  let b_ortho = normalize(cross(n_geo, tangent));
  let b_sign = select(-1.0, 1.0, dot(b_ortho, decal_fwd) >= 0.0);
  let bitangent = b_ortho * b_sign;

  let n_tex = textureSampleLevel(normal_tex, decal_sampler, uv, 0.0).xyz * 2.0 - 1.0;
  let n_str = decal.emissive_normal.w;
  let n_ts = normalize(vec3f(n_tex.xy * n_str, max(n_tex.z, 0.05)));
  let n_world = normalize(tangent * n_ts.x + bitangent * n_ts.y + n_geo * n_ts.z);

  let orm = textureSampleLevel(orm_tex, decal_sampler, uv, 0.0);
  let ao = clamp(orm.r, 0.0, 1.0);
  let roughness = clamp(orm.g * decal.pbr_params.x, 0.04, 1.0);
  let metallic = clamp(orm.b * decal.pbr_params.y, 0.0, 1.0);
  let base_color = albedo_sample.rgb * decal.albedo_opacity.rgb;

  let sun_raw = view.sun_dir_and_unused.xyz;
  let l_dir = select(vec3f(0.0, 1.0, 0.0), normalize(sun_raw), length(sun_raw) > 1e-6);
  let h_dir = normalize(view_dir + l_dir);
  let ndotl = max(dot(n_world, l_dir), 0.0);
  let ndotv = max(dot(n_world, view_dir), 0.001);
  let ndoth = max(dot(n_world, h_dir), 0.0);
  let vdoth = max(dot(view_dir, h_dir), 0.0);

  let f0 = mix(vec3f(0.04), base_color, metallic);
  let fresnel = f0 + (1.0 - f0) * pow(1.0 - vdoth, 5.0);
  let alpha = roughness * roughness;
  let d_term = ggx_d(ndoth, alpha * alpha);
  let g_term = smith_g(ndotv, ndotl, roughness);
  let spec = (d_term * g_term * fresnel) / max(4.0 * ndotv * max(ndotl, 0.001), 1e-5);
  let kd = (1.0 - fresnel) * (1.0 - metallic);
  let direct = (kd * base_color / PI + spec) * view.sun_color_intensity.rgb * ndotl;

  let hemi = n_world.y * 0.5 + 0.5;
  let amb_irrad = mix(view.ground_ambient.rgb, view.sky_ambient.rgb, hemi) * view.sun_color_intensity.w;
  let r_vec = reflect(-view_dir, n_world);
  let env_refl = mix(view.ground_ambient.rgb, view.sky_ambient.rgb, r_vec.y * 0.5 + 0.5) * view.sun_color_intensity.w;
  let fresnel_amb = f0 + (max(vec3f(1.0 - roughness), f0) - f0) * pow(1.0 - ndotv, 5.0);
  let ambient = ((1.0 - metallic) * base_color * amb_irrad + fresnel_amb * env_refl) * ao;

  let shaded = direct + ambient + decal.emissive_normal.rgb;
  return vec4f(shaded * cover, cover);
}

@fragment fn pbr_fragment(in: BoxOut ${sample_param}) -> @location(0) vec4f {
  return shade_pbr_decal(in, ${sample_arg});
}
