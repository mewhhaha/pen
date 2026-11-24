// The frame's particles, each a square facing the camera, tinted by its
// color and shaped by its emitter's material's texture (a soft round dot
// without one), drawn over the scene's opaque and translucent surfaces
// without writing depth. The mist between the eye and a particle, at the
// depth of its middle, is its own: a particle covering what is behind it
// takes the mist's transmittance and its in-scatter by its alpha, light that
// only adds takes the transmittance alone (translucency.wgsl).
struct View {
  view_projection: mat4x4f,
  // The camera's right and up, in world space.
  right: vec4f,
  up: vec4f,
};

struct Particle {
  place: vec4f,
  color: vec4f,
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
};

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
  out.uv = vec2f(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
  out.color = particle.color;
  out.fog = fog_terms(particle.place.xyz);
  return out;
}

// Premultiplied, for both blends. Over what is behind it, which holds the mist
// to its own depth already, a particle of colour c and alpha a gives
// a * (c * T + L) + (1 - a) * behind: the premultiplied colour times T, and the
// in-scatter by its alpha. Added light is `c * a * T` and no mist of its own,
// since it hides nothing.
@fragment fn fragment(input: Corner) -> @location(0) vec4f {
  let color = input.color * textureSample(shape, shape_sampler, input.uv);
  let light = color.rgb * color.a * input.fog.a;
  return vec4f(select(light + input.fog.rgb * color.a, light, additive), color.a);
}
