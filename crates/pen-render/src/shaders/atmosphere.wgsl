@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var scene_color: texture_2d<f32>;
@group(0) @binding(2) var scene_depth: ${depth_texture};
@group(0) @binding(3) var occlusion_map: texture_2d<f32>;
@group(0) @binding(4) var occlusion_sampler: sampler;
@group(0) @binding(5) var<uniform> volume: Volume;
@group(0) @binding(6) var volume_integrated: texture_3d<f32>;
${fullscreen}
${common}
${volume}

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
  let remaining = scene.a;
  let color = scene.rgb - remaining * atmosphere.clear.rgb;
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
  let camera = atmosphere.camera.xyz;
  let direction = normalize(unproject(position.xy, 0.5) - camera);
  let with_sky = atmosphere.camera.w > 0.5;
  let background = select(atmosphere.clear.rgb, sky(direction, true), with_sky);
  if (empty > 0.999) {
    return vec4f(color + remaining * background, 1.0);
  }
  // Depth counts opaque coverage; alpha also accounts for translucent
  // surfaces and particles covering the background without writing depth.
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
  return vec4f(misted * (1.0 - empty) + remaining * background, 1.0);
}
