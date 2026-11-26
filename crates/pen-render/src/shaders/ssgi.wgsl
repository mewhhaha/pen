// Screen-Space Global Illumination (SSGI): single-bounce diffuse indirect
// lighting via depth-buffer hemisphere raymarching in linear HDR space.
struct Ssgi {
  projection: mat4x4f,
  inverse_projection: mat4x4f,
  // x: world radius, y: bounce intensity, z: hit thickness, w: max screen radius (px)
  settings: vec4f,
  // xy: texture size, zw: 1 / texture size
  size: vec4f,
};

@group(0) @binding(0) var<uniform> ssgi: Ssgi;
@group(0) @binding(1) var scene_color: texture_2d<f32>;
@group(0) @binding(2) var scene_depth: ${depth_texture};
@group(0) @binding(3) var color_sampler: sampler;
${fullscreen}

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

fn depth_at(pixel: vec2i) -> f32 {
  let size = vec2i(ssgi.size.xy);
  return textureLoad(scene_depth, clamp(pixel, vec2i(0), size - 1), 0);
}

fn view_at(pixel: vec2f, depth: f32) -> vec3f {
  let ndc = vec2f(pixel.x * ssgi.size.z * 2.0 - 1.0, 1.0 - pixel.y * ssgi.size.w * 2.0);
  let point = ssgi.inverse_projection * vec4f(ndc, depth, 1.0);
  return point.xyz / point.w;
}

fn view_z_at(pixel: vec2f, depth: f32) -> f32 {
  let ndc = vec2f(pixel.x * ssgi.size.z * 2.0 - 1.0, 1.0 - pixel.y * ssgi.size.w * 2.0);
  let m = ssgi.inverse_projection;
  let z = m[0].z * ndc.x + m[1].z * ndc.y + m[2].z * depth + m[3].z;
  let w = m[0].w * ndc.x + m[1].w * ndc.y + m[2].w * depth + m[3].w;
  return z / w;
}

fn neighbour(pixel: vec2f, step: vec2i) -> vec3f {
  return view_at(pixel + vec2f(step), depth_at(vec2i(pixel) + step));
}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(position.xy);
  let base = textureLoad(scene_color, pixel, 0);
  let depth = depth_at(pixel);
  if (depth >= 1.0) {
    return base;
  }

  let middle = view_at(position.xy, depth);
  let view_dir = normalize(-middle);
  let right = neighbour(position.xy, vec2i(1, 0)) - middle;
  let left = middle - neighbour(position.xy, vec2i(-1, 0));
  let down = neighbour(position.xy, vec2i(0, 1)) - middle;
  let up = middle - neighbour(position.xy, vec2i(0, -1));
  let across = select(left, right, abs(right.z) < abs(left.z));
  let along = select(up, down, abs(down.z) < abs(up.z));
  var normal = normalize(cross(across, along));
  if (dot(normal, view_dir) < 0.0) {
    normal = -normal;
  }

  // Orthonormal tangent basis around the view-space surface normal.
  let helper = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
  let t = normalize(cross(helper, normal));
  let b = cross(normal, t);

  let ign = fract(52.9829189 * fract(dot(floor(position.xy), vec2f(0.06711056, 0.00583715))));
  let radius = ssgi.settings.x;
  let intensity = ssgi.settings.y;
  let thickness = ssgi.settings.z;

  let rays = 8;
  let steps = 6;
  let origin = middle + normal * 0.04;
  let clip_origin = ssgi.projection * vec4f(origin, 1.0);
  let max_hit_dist = radius * 1.35;
  let inv_max_hit_dist = 1.0 / max(max_hit_dist, 1e-5);
  var bounce = vec3f(0.0);
  for (var r = 0; r < rays; r++) {
    let u1 = (f32(r) + ign) / f32(rays);
    let u2 = fract(ign * 13.7 + f32(r) * 0.61803399);
    // Cosine-weighted hemisphere ray direction.
    let phi = u1 * TWO_PI;
    let sin_t = sqrt(clamp(u2, 0.04, 0.96));
    let cos_t = sqrt(max(1.0 - sin_t * sin_t, 0.0));
    let ray_dir = normalize(t * (cos(phi) * sin_t) + b * (sin(phi) * sin_t) + normal * cos_t);
    let clip_dir = ssgi.projection * vec4f(ray_dir, 0.0);

    for (var s = 1; s <= steps; s++) {
      let frac = (f32(s) - 0.5 + 0.5 * u2) / f32(steps);
      let march_dist = radius * frac * frac;
      let probe_z = origin.z + ray_dir.z * march_dist;
      if (probe_z >= -0.05) {
        break;
      }
      let clip = clip_origin + clip_dir * march_dist;
      let inv_w = 1.0 / max(clip.w, 1e-5);
      let proj_xy = vec2f((clip.x * inv_w * 0.5 + 0.5) * ssgi.size.x, (0.5 - clip.y * inv_w * 0.5) * ssgi.size.y);
      if (proj_xy.x < 1.0 || proj_xy.x >= ssgi.size.x - 1.0 || proj_xy.y < 1.0 || proj_xy.y >= ssgi.size.y - 1.0) {
        break;
      }
      let sample_depth = depth_at(vec2i(proj_xy));
      if (sample_depth >= 1.0) {
        continue;
      }
      // In RH view space (z < 0), sample_z > probe_z means the surface is closer to the eye than the ray.
      let sample_z = view_z_at(proj_xy, sample_depth);
      let dz = sample_z - probe_z;
      if (dz > 0.01 && dz < thickness) {
        let sample_pos = view_at(proj_xy, sample_depth);
        let to_hit = sample_pos - middle;
        let hit_dist = length(to_hit);
        if (hit_dist > 0.05 && hit_dist < max_hit_dist) {
          let ndotl = max(dot(normal, to_hit / hit_dist), 0.0);
          let atten = clamp(1.0 - hit_dist * inv_max_hit_dist, 0.0, 1.0);
          let uv = proj_xy * ssgi.size.zw;
          let radiance = min(textureSampleLevel(scene_color, color_sampler, uv, 0.0).rgb, vec3f(6.0));
          bounce += radiance * (ndotl * atten * atten);
        }
        break;
      }
    }
  }

  let indirect = bounce * (intensity / f32(rays));
  // Modulate bounce conservatively by approximate surface reflectance so dark materials
  // do not glow while shadowed white/neutral walls still receive clear color bleed.
  let luma = dot(base.rgb, vec3f(0.2126, 0.7152, 0.0722));
  let chroma = clamp(base.rgb / max(luma, 0.15), vec3f(0.0), vec3f(1.0));
  let reflectance = mix(vec3f(0.55), chroma, clamp(luma * 2.0, 0.0, 1.0));
  return vec4f(base.rgb + indirect * reflectance, base.a);
}
