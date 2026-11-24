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

fn neighbour(pixel: vec2f, step: vec2i) -> vec3f {
  return view_at(pixel + vec2f(step), depth_at(vec2i(pixel) + step));
}

fn screen_of(view_pos: vec3f) -> vec3f {
  let clip = ssgi.projection * vec4f(view_pos, 1.0);
  let ndc = clip.xyz / max(clip.w, 1e-5);
  return vec3f((ndc.x * 0.5 + 0.5) * ssgi.size.x, (0.5 - ndc.y * 0.5) * ssgi.size.y, ndc.z);
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
  var bounce = vec3f(0.0);
  for (var r = 0; r < rays; r++) {
    let u1 = (f32(r) + ign) / f32(rays);
    let u2 = fract(ign * 13.7 + f32(r) * 0.61803399);
    // Cosine-weighted hemisphere ray direction.
    let phi = u1 * TWO_PI;
    let sin_t = sqrt(clamp(u2, 0.04, 0.96));
    let cos_t = sqrt(max(1.0 - sin_t * sin_t, 0.0));
    let ray_dir = normalize(t * (cos(phi) * sin_t) + b * (sin(phi) * sin_t) + normal * cos_t);

    // Offset start slightly along normal to avoid self-intersection.
    let origin = middle + normal * 0.04;
    for (var s = 1; s <= steps; s++) {
      let frac = (f32(s) - 0.5 + 0.5 * u2) / f32(steps);
      let march_dist = radius * frac * frac;
      let probe = origin + ray_dir * march_dist;
      if (probe.z >= -0.05) {
        break;
      }
      let proj = screen_of(probe);
      if (proj.x < 1.0 || proj.x >= ssgi.size.x - 1.0 || proj.y < 1.0 || proj.y >= ssgi.size.y - 1.0) {
        break;
      }
      let sample_depth = depth_at(vec2i(proj.xy));
      if (sample_depth >= 1.0) {
        continue;
      }
      let sample_pos = view_at(proj.xy, sample_depth);
      // In RH view space (z < 0), sample_pos.z > probe.z means the surface is closer to the eye than the ray.
      let dz = sample_pos.z - probe.z;
      if (dz > 0.01 && dz < thickness) {
        let to_hit = sample_pos - middle;
        let hit_dist = length(to_hit);
        if (hit_dist > 0.05 && hit_dist < radius * 1.35) {
          let ndotl = max(dot(normal, to_hit / hit_dist), 0.0);
          let atten = clamp(1.0 - hit_dist / (radius * 1.35), 0.0, 1.0);
          let uv = proj.xy * ssgi.size.zw;
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
