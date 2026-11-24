@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var scene_color: texture_2d<f32>;
@group(0) @binding(2) var scene_depth: ${depth_texture};
@group(0) @binding(3) var occlusion_map: texture_2d<f32>;
@group(0) @binding(4) var occlusion_sampler: sampler;
@group(0) @binding(5) var<uniform> volume: Volume;
@group(0) @binding(6) var volume_integrated: texture_3d<f32>;
@group(0) @binding(7) var physical_sky_lut: texture_2d<f32>;
@group(0) @binding(8) var physical_clouds: texture_2d<f32>;
${fullscreen}
${common}
${volume}

// Samples the Hillaire 2020 physical sky-view LUT (with solar disk) and optionally
// composites the raymarched volumetric cloud buffer (`cloud.rgb + sky * cloud.a`).
fn physical_sky_background(direction: vec3f, screen_uv: vec2f, with_clouds: bool) -> vec3f {
  let rb = 6360.0;
  let r = rb + 0.2;
  let pi = 3.14159265359;
  let d = normalize(direction);
  let sun_raw = atmosphere.sun.xyz;
  let sun_len = length(sun_raw);
  let sun_dir = select(vec3f(0.0, 0.85, -0.5268), sun_raw / sun_len, sun_len > 1e-5);

  let horizon_angle = -acos(clamp(rb / r, 0.0, 1.0));
  let lat = asin(clamp(d.y, -1.0, 1.0));
  var v = 0.5;
  if (lat < horizon_angle) {
    let span = max(horizon_angle + 0.5 * pi, 1e-4);
    v = 0.5 * (1.0 - sqrt(clamp((horizon_angle - lat) / span, 0.0, 1.0)));
  } else {
    let span = max(0.5 * pi - horizon_angle, 1e-4);
    v = 0.5 * (1.0 + sqrt(clamp((lat - horizon_angle) / span, 0.0, 1.0)));
  }
  let sun_xz = vec2f(sun_dir.x, sun_dir.z);
  let sun_xz_len = length(sun_xz);
  let fwd = select(vec2f(0.0, -1.0), sun_xz / sun_xz_len, sun_xz_len > 1e-5);
  let right = vec2f(-fwd.y, fwd.x);
  let az = atan2(dot(d.xz, right), dot(d.xz, fwd));
  let uv = vec2f(fract(az / (2.0 * pi) + 0.5), clamp(v, 0.0, 1.0));

  let dims = max(vec2f(textureDimensions(physical_sky_lut, 0)), vec2f(2.0));
  let safe_uv = clamp(uv * ((dims - 1.0) / dims) + 0.5 / dims, vec2f(0.0), vec2f(1.0));
  var col = textureSampleLevel(physical_sky_lut, occlusion_sampler, safe_uv, 0.0).rgb;

  let toward = max(dot(d, sun_dir), 0.0);
  let above = smoothstep(-0.02, 0.02, d.y);
  col += atmosphere.sun_color.rgb * 4.0 * smoothstep(0.99945, 0.9997, toward) * above;

  if (with_clouds) {
    let c = textureSampleLevel(physical_clouds, occlusion_sampler, clamp(screen_uv, vec2f(0.0), vec2f(1.0)), 0.0);
    col = c.rgb + col * clamp(c.a, 0.0, 1.0);
  }
  return col;
}

// The ambient occlusion here, blurred over the 4 x 4 pixels its noise repeats
// in: four filtered reads, each averaging 2 x 2.
fn occlusion(pixel: vec2f) -> f32 {
  let texel = 1.0 / atmosphere.size.xy;
  let at = pixel * texel;
  var total = 0.0;
  for (var corner = 0; corner < 4; corner++) {
    let offset = vec2f(select(-0.5, 1.5, (corner & 1) == 1), select(-0.5, 1.5, corner > 1));
    total += textureSampleLevel(occlusion_map, occlusion_sampler, at + offset * texel, 0.0).r;
  }
  return total * 0.25;
}

fn unproject(pixel: vec2f, depth: f32) -> vec3f {
  let ndc = vec2f(pixel.x / atmosphere.size.x * 2.0 - 1.0, 1.0 - pixel.y / atmosphere.size.y * 2.0);
  let world = atmosphere.inverse_view_projection * vec4f(ndc, depth, 1.0);
  return world.xyz / world.w;
}


// The colour the mist takes beyond the froxel volume, or everywhere without it:
// the light it scatters toward the eye per unit of optical depth, unshadowed,
// the sun in the phase the mist's anisotropy gives it and the ambient light
// the sky gives all round or, by `volume.scatter.z`, the sky's own colour in
// the direction looked.
fn mist_light(direction: vec3f, point: vec3f) -> vec3f {
  let around = mix(volume.ambient.rgb, mist_sky(direction), volume.scatter.z) * volume.scatter.y;
  let sun = atmosphere.sun_color.rgb * (volume.scatter.x * phase_scaled(volume.medium.w, dot(atmosphere.sun.xyz, direction)) * cloud_shadow(point));
  return volume.medium.rgb * (around + sun);
}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(position.xy);
  let scene = textureLoad(scene_color, pixel, 0);
  var empty = 0.0;
  var nearest = 1.0;
  for (var sample = 0; sample < ${samples}; sample++) {
    let depth = textureLoad(scene_depth, pixel, sample);
    if (depth >= 1.0) {
      empty += ${share};
    } else {
      nearest = min(nearest, depth);
    }
  }
  // The clear colour fills what the samples left empty, in what the resolve
  // averaged: take it back out, to keep the surfaces' own share.
  let color = scene.rgb - empty * atmosphere.clear.rgb;
  let camera = atmosphere.camera.xyz;
  let direction = normalize(unproject(position.xy, 0.5) - camera);
  let sky_mode = atmosphere.camera.w;
  var background = atmosphere.clear.rgb;
  if (sky_mode > 1.5) {
    background = physical_sky_background(direction, position.xy / atmosphere.size.xy, sky_mode > 2.5);
  } else if (sky_mode > 0.5) {
    background = sky(direction, true);
  }
  if (empty > 0.999) {
    return vec4f(color + empty * background, 1.0);
  }
  let surface = color / (1.0 - empty) *
    mix(1.0, occlusion(position.xy), atmosphere.fog.w);
  let point = unproject(position.xy, nearest);
  // The mist between the eye and the surface: what the froxel volume holds up
  // to its reach, then the same medium analytically, in the colour of what the
  // volume ends on.
  var scattered = vec3f(0.0);
  var optical = 0.0;
  var start = camera;
  var tint = mist_light(direction, point);
  if (volume.ambient.w > 0.5) {
    let depth = -(volume.view * vec4f(point, 1.0)).z;
    let reach = min(depth, volume.grid.w);
    let at = vec3f(position.xy / atmosphere.size.xy, (volume_boundary(reach) + 0.5) / (volume.grid.z + 1.0));
    let held = textureSampleLevel(volume_integrated, occlusion_sampler, at, 0.0);
    scattered = held.rgb;
    optical = held.a;
    if (depth > volume.grid.w) {
      start = camera + (point - camera) * (volume.grid.w / depth);
    } else {
      start = point;
    }
    let seen = 1.0 - exp(-optical);
    tint = mix(tint, scattered / max(seen, 0.0001), smoothstep(0.0, 0.05, seen));
  }
  let beyond = mist_beyond_arena(start, point) * atmosphere.fog.y;
  let transmitted = exp(-optical);
  let light = scattered + transmitted * tint * (1.0 - exp(-beyond));
  let misted = surface * cloud_shadow(point) * (transmitted * exp(-beyond)) + light;
  return vec4f(misted * (1.0 - empty) + empty * background, 1.0);
}
