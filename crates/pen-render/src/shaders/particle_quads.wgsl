// The frame's particles, each a square facing the camera, tinted by its
// color and shaped by its emitter's material's texture (a soft round dot
// without one), supporting flipbook sub-UV sheets, soft opaque-depth fading,
// directional sun + ambient lighting, and volumetric mist.
struct View {
  view_projection: mat4x4f,
  inverse_projection: mat4x4f,
  // The camera's right and up, in world space.
  right: vec4f,
  up: vec4f,
  // Directional sun direction (toward surface), sun color, ambient color, screen size.
  sun_dir: vec4f,
  sun_color: vec4f,
  ambient: vec4f,
  screen_size: vec4f,
};

struct Particle {
  place: vec4f,
  color: vec4f,
  info: vec4f,
};

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var shape: texture_2d<f32>;
@group(0) @binding(3) var shape_sampler: sampler;
@group(0) @binding(6) var sky_sampler: sampler;
${translucency}

// Whether the particles add light (fire, sparks) rather than cover what is behind them.
const additive = ${additive};

struct Corner {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec4f,
  // The mist before the particle: rgb its in-scatter, a its transmittance.
  @location(2) @interpolate(flat) fog: vec4f,
  // x: soft depth fade distance; y: lighting weight; zw: quad corner (-1..1)
  @location(3) effects: vec4f,
};

fn linear_depth(ndc_z: f32) -> f32 {
  let clip = view.inverse_projection * vec4f(0.0, 0.0, ndc_z, 1.0);
  return -clip.z / max(abs(clip.w), 1e-5);
}

@vertex fn vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> Corner {
  let corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let particle = particles[index];
  let corner = corners[vertex];
  // A particle of no width is gone: its square has no area.
  let world = particle.place.xyz + (view.right.xyz * corner.x + view.up.xyz * corner.y) * (particle.place.w * 0.5);
  var out: Corner;
  out.position = view.view_projection * vec4f(world, 1.0);
  let base_uv = vec2f(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
  let cols = max(floor(particle.info.w / 64.0), 1.0);
  let rows = max(particle.info.w - cols * 64.0, 1.0);
  if (cols > 1.0 || rows > 1.0) {
    let frame = particle.info.z;
    let col = frame % cols;
    let row = floor(frame / cols);
    out.uv = (vec2f(col, row) + base_uv) / vec2f(cols, rows);
  } else {
    out.uv = base_uv;
  }
  out.color = particle.color;
  out.fog = fog_terms(particle.place.xyz);
  out.effects = vec4f(particle.info.x, particle.info.y, corner.x, corner.y);
  return out;
}

@fragment fn fragment(input: Corner) -> @location(0) vec4f {
  var color = input.color * textureSample(shape, shape_sampler, input.uv);
  let lit = input.effects.y;
  if (lit > 0.0001) {
    let to_camera = normalize(cross(view.right.xyz, view.up.xyz));
    let n_approx = normalize(view.right.xyz * input.effects.z + view.up.xyz * input.effects.w + to_camera * 0.6);
    let to_sun = -normalize(view.sun_dir.xyz);
    let wrap = clamp((dot(n_approx, to_sun) + 0.5) / 1.5, 0.0, 1.0);
    let lighting = view.ambient.rgb + view.sun_color.rgb * wrap;
    color = vec4f(color.rgb * mix(vec3f(1.0), lighting, lit), color.a);
  }
  let soft = input.effects.x;
  if (soft > 0.0001 && opaque_available()) {
    let dims = vec2i(textureDimensions(opaque_depth, 0));
    let coord = clamp(vec2i(input.position.xy), vec2i(0), dims - vec2i(1));
    let scene_d = textureLoad(opaque_depth, coord, 0).r;
    if (scene_d < 1.0) {
      let gap = linear_depth(scene_d) - linear_depth(input.position.z);
      let fade = clamp(gap / soft, 0.0, 1.0);
      color.a *= fade;
    }
  }
  let light = color.rgb * color.a * input.fog.a;
  return vec4f(select(light + input.fog.rgb * color.a, light, additive), color.a);
}
