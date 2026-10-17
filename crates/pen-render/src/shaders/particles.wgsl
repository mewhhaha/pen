// Where each emitter's live particles are, worked out afresh every frame from
// the emitter's clock and seed alone: particle k is born k / rate seconds
// into the emitter's life, at a random point of the ball round it, with a
// random velocity added to the emitter's, and flies under gravity and drag
// in closed form. So the same emitter at the same time gives the same
// particles, whatever the frame rate, and rewinding needs no history.
struct Emitter {
  transform: mat4x4f,
  // xyz: velocity at birth, in the emitter's space; w: the most random speed
  // added to it.
  velocity: vec4f,
  // xyz: gravity, in world space; w: drag, a share of velocity a second.
  gravity: vec4f,
  color_start: vec4f,
  color_end: vec4f,
  // Birth ball's radius, width at birth and at death, lifetime in seconds.
  shape: vec4f,
  // x: the age of its first slot's particle (the oldest), y: seconds between
  // births.
  timing: vec4f,
  // x: its first slot among the frame's, y: its slots, z: how many of them
  // are alive, w: the first slot's particle's birth number.
  counts: vec4u,
  // x: its seed.
  seed: vec4u,
};

struct Particle {
  // xyz: where it is; w: its width, 0 for none.
  place: vec4f,
  color: vec4f,
};

@group(0) @binding(0) var<storage, read> emitters: array<Emitter>;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

// PCG's output permutation of an LCG step: integer only, so every GPU draws
// the same numbers.
fn hash(value: u32) -> u32 {
  let state = value * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// A number in 0..1 for particle `index` of `seed`: one of its streams.
fn random(seed: u32, index: u32, stream: u32) -> f32 {
  return f32(hash(seed ^ hash(index ^ hash(stream))) >> 8u) / 16777216.0;
}

// A point of the unit ball, evenly spread.
fn ball(seed: u32, index: u32, stream: u32) -> vec3f {
  let z = random(seed, index, stream) * 2.0 - 1.0;
  let angle = random(seed, index, stream + 1u) * 6.28318531;
  let reach = pow(random(seed, index, stream + 2u), 1.0 / 3.0);
  let around = sqrt(max(0.0, 1.0 - z * z));
  return vec3f(around * cos(angle), around * sin(angle), z) * reach;
}

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) id: vec3u) {
  let slot = id.x;
  // The emitter whose slots hold this one: the last starting at or before it.
  var low = 0u;
  var high = arrayLength(&emitters);
  while (low + 1u < high) {
    let middle = (low + high) / 2u;
    if (emitters[middle].counts.x <= slot) {
      low = middle;
    } else {
      high = middle;
    }
  }
  let emitter = emitters[low];
  let local = slot - emitter.counts.x;
  if (local >= emitter.counts.y) {
    return;
  }
  if (local >= emitter.counts.z) {
    particles[slot] = Particle(vec4f(0.0), vec4f(0.0));
    return;
  }
  let index = emitter.counts.w + local;
  let seed = emitter.seed.x;
  let age = max(emitter.timing.x - f32(local) * emitter.timing.y, 0.0);
  let life = clamp(age / emitter.shape.w, 0.0, 1.0);
  let born = (emitter.transform * vec4f(ball(seed, index, 0u) * emitter.shape.x, 1.0)).xyz;
  let launched = emitter.velocity.xyz + ball(seed, index, 3u) * emitter.velocity.w;
  let velocity = (emitter.transform * vec4f(launched, 0.0)).xyz;
  let gravity = emitter.gravity.xyz;
  let drag = emitter.gravity.w;
  var moved = velocity * age + 0.5 * gravity * age * age;
  if (drag > 0.0001) {
    // dv/dt = gravity - drag v, solved.
    let carried = (1.0 - exp(-drag * age)) / drag;
    moved = velocity * carried + gravity * (age - carried) / drag;
  }
  let width = mix(emitter.shape.y, emitter.shape.z, life);
  particles[slot] = Particle(vec4f(born + moved, width), mix(emitter.color_start, emitter.color_end, life));
}
