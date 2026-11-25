// Race-Free 3-Pass Stateful GPU Mesh Particle Simulation Compute Shaders:
// Pass 1 (`simulate_step`): advances live particles (3D divergence-free curl noise + swept
//   world-space depth-buffer collision) and appends child particles to a dedicated `spawn_queue`.
// Pass 2 (`apply_spawns`): claims dead slots (`vel_life.w <= 0`) on the GPU to materialize queued
//   child particles — zero data races and automatic GPU-resident slot reclamation.
// Pass 3 (`compact_alive`): compacts live particle indices into `alive_indices` and writes exact
//   `DrawIndexedIndirect.instance_count` (guaranteed <= max_particles).

struct GpuParticle {
  pos_age: vec4f,      // xyz = world position, w = age (seconds)
  vel_life: vec4f,     // xyz = world velocity, w = lifetime (seconds; <= 0 means dead/free slot)
  color: vec4f,        // rgba = linear color
  size_rot_gen: vec4f, // x = size, y = rotation_rad, z = spin_speed_rad_s, w = generation (0 = primary, 1 = child)
};

struct SimParams {
  view_projection: mat4x4f,
  inverse_view_projection: mat4x4f,
  camera_pos_and_clear: vec4f, // xyz = camera world pos, w = depth clear value (1.0 standard, 0.0 rev-Z)
  gravity_dt: vec4f,           // xyz = gravity (m/s^2), w = dt (seconds)
  curl_drag: vec4f,            // x = drag, y = curl_strength, z = curl_frequency, w = time (seconds)
  collision: vec4f,            // x = restitution (0..1), y = friction (0..1), z = world_thickness_m, w = reversed_z (>0.5)
  sub_emit: vec4f,             // x = max_particles, y = sub_emit_count, z = sub_emit_speed, w = sub_emit_life
  flags: vec4f,                // x = depth_collide (>0.5), y = sub_emit_on_bounce (>0.5), z = sub_emit_on_death (>0.5), w = index_count
};

struct DrawIndexedIndirect {
  index_count: u32,
  instance_count: atomic<u32>,
  first_index: u32,
  base_vertex: i32,
  first_instance: u32,
};

struct SimCounters {
  spawn_queued: atomic<u32>,
  spawn_consumed: atomic<u32>,
  sub_emitted: atomic<u32>,
  bounces: atomic<u32>,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read_write> spawn_queue: array<GpuParticle>;
@group(0) @binding(3) var<storage, read_write> alive_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> indirect: DrawIndexedIndirect;
@group(0) @binding(5) var<storage, read_write> counters: SimCounters;
@group(0) @binding(6) var depth_tex: texture_2d<f32>;

fn hash3(p: vec3f) -> vec3f {
  let q = vec3f(
    dot(p, vec3f(127.1, 311.7, 74.7)),
    dot(p, vec3f(269.5, 183.3, 246.1)),
    dot(p, vec3f(113.5, 271.9, 124.6)),
  );
  return fract(sin(q) * 43758.5453);
}

fn value_noise3_vec(p: vec3f) -> vec3f {
  let i = floor(p);
  let f = p - i;
  let u = f * f * (3.0 - 2.0 * f);
  let n000 = hash3(i + vec3f(0.0, 0.0, 0.0));
  let n100 = hash3(i + vec3f(1.0, 0.0, 0.0));
  let n010 = hash3(i + vec3f(0.0, 1.0, 0.0));
  let n110 = hash3(i + vec3f(1.0, 1.0, 0.0));
  let n001 = hash3(i + vec3f(0.0, 0.0, 1.0));
  let n101 = hash3(i + vec3f(1.0, 0.0, 1.0));
  let n011 = hash3(i + vec3f(0.0, 1.0, 1.0));
  let n111 = hash3(i + vec3f(1.0, 1.0, 1.0));
  let nx00 = mix(n000, n100, u.x);
  let nx10 = mix(n010, n110, u.x);
  let nx01 = mix(n001, n101, u.x);
  let nx11 = mix(n011, n111, u.x);
  return mix(mix(nx00, nx10, u.y), mix(nx01, nx11, u.y), u.z) * 2.0 - 1.0;
}

// Divergence-free 3D curl of a smooth 3-component noise vector potential Psi(p).
fn curl_noise3(p: vec3f) -> vec3f {
  let e = 0.05;
  let dx = vec3f(e, 0.0, 0.0);
  let dy = vec3f(0.0, e, 0.0);
  let dz = vec3f(0.0, 0.0, e);

  let p_x1 = value_noise3_vec(p + dx);
  let p_x0 = value_noise3_vec(p - dx);
  let p_y1 = value_noise3_vec(p + dy);
  let p_y0 = value_noise3_vec(p - dy);
  let p_z1 = value_noise3_vec(p + dz);
  let p_z0 = value_noise3_vec(p - dz);

  let inv2e = 0.5 / e;
  let cx = (p_y1.z - p_y0.z) - (p_z1.y - p_z0.y);
  let cy = (p_z1.x - p_z0.x) - (p_x1.z - p_x0.z);
  let cz = (p_x1.y - p_x0.y) - (p_y1.x - p_y0.x);
  return vec3f(cx, cy, cz) * inv2e;
}

struct DepthHit {
  world: vec3f,
  depth: f32,
  valid: bool,
};

fn load_depth_world(px: vec2i, dims: vec2i) -> DepthHit {
  if (px.x < 0 || px.y < 0 || px.x >= dims.x || px.y >= dims.y) {
    return DepthHit(vec3f(0.0), 0.0, false);
  }
  let d = textureLoad(depth_tex, px, 0).r;
  if (abs(d - params.camera_pos_and_clear.w) <= 1e-6) {
    return DepthHit(vec3f(0.0), d, false);
  }
  let ndc = vec2f(
    (f32(px.x) + 0.5) / f32(dims.x) * 2.0 - 1.0,
    1.0 - (f32(px.y) + 0.5) / f32(dims.y) * 2.0,
  );
  let h = params.inverse_view_projection * vec4f(ndc, d, 1.0);
  if (abs(h.w) <= 1e-6) {
    return DepthHit(vec3f(0.0), d, false);
  }
  return DepthHit(h.xyz / h.w, d, true);
}

fn enqueue_children(origin: vec3f, normal: vec3f, base_vel: vec3f, base_color: vec4f, base_size: f32, seed_id: u32) {
  let count = u32(clamp(round(params.sub_emit.y), 0.0, 8.0));
  let max_p = u32(params.sub_emit.x);
  let spd = max(params.sub_emit.z, 0.5);
  let life = max(params.sub_emit.w, 0.15);

  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
  let t1 = normalize(cross(up, normal));
  let t2 = cross(normal, t1);

  for (var k = 0u; k < count; k++) {
    let q = atomicAdd(&counters.spawn_queued, 1u);
    if (q >= max_p) {
      break;
    }
    let rnd = hash3(vec3f(f32(seed_id) * 0.13 + f32(k) * 7.1, params.curl_drag.w * 3.7, f32(q) * 0.29));
    let angle = rnd.x * 6.2831853;
    let spread = 0.35 + 0.55 * rnd.y;
    let dir = normalize(normal * (0.6 + 0.4 * rnd.z) + (t1 * cos(angle) + t2 * sin(angle)) * spread);
    let child_vel = dir * spd * (0.7 + 0.6 * rnd.z) + base_vel * 0.25;
    spawn_queue[q] = GpuParticle(
      vec4f(origin + normal * 0.03, 0.0),
      vec4f(child_vel, life * (0.75 + 0.5 * rnd.y)),
      vec4f(min(base_color.rgb * 1.4 + vec3f(0.3, 0.2, 0.05), vec3f(2.0)), 1.0),
      vec4f(max(base_size * 0.45, 0.02), angle, (rnd.x - 0.5) * 12.0, 1.0),
    );
  }
}

@compute @workgroup_size(64)
fn simulate_step(@builtin(global_invocation_id) id: vec3u) {
  let idx = id.x;
  let max_p = u32(params.sub_emit.x);
  if (idx >= max_p) {
    return;
  }
  let life = particles[idx].vel_life.w;
  if (life <= 0.0) {
    return;
  }
  var p = particles[idx];
  let dt = clamp(params.gravity_dt.w, 0.0, 0.1);
  let new_age = p.pos_age.w + dt;
  let is_primary = p.size_rot_gen.w < 0.5;
  if (new_age >= life) {
    p.vel_life.w = 0.0;
    particles[idx] = p;
    if (is_primary && params.flags.z > 0.5) {
      enqueue_children(p.pos_age.xyz, vec3f(0.0, 1.0, 0.0), p.vel_life.xyz, p.color, p.size_rot_gen.x, idx);
    }
    return;
  }

  let prev_pos = p.pos_age.xyz;
  var vel = p.vel_life.xyz;

  vel += params.gravity_dt.xyz * dt;
  if (abs(params.curl_drag.y) > 1e-5) {
    let freq = max(params.curl_drag.z, 0.05);
    let curl = curl_noise3(prev_pos * freq + vec3f(0.0, params.curl_drag.w * 0.35, 0.0));
    vel += curl * params.curl_drag.y * dt;
  }
  let drag = clamp(1.0 - params.curl_drag.x * dt, 0.0, 1.0);
  vel *= drag;
  var next_pos = prev_pos + vel * dt;

  // Swept world-space depth-buffer collision test (prevents tunneling and never bounces exiting particles).
  if (params.flags.x > 0.5) {
    for (var sample_index = 1u; sample_index <= 2u; sample_index += 1u) {
      let sample_pos = mix(prev_pos, next_pos, f32(sample_index) * 0.5);
      let clip_next = params.view_projection * vec4f(sample_pos, 1.0);
      if (clip_next.w > 1e-4) {
        let ndc = clip_next.xyz / clip_next.w;
        if (ndc.x >= -1.0 && ndc.x <= 1.0 && ndc.y >= -1.0 && ndc.y <= 1.0 && ndc.z >= 0.0 && ndc.z <= 1.0) {
          let dims = vec2i(textureDimensions(depth_tex, 0));
          let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
          let px = clamp(vec2i(floor(uv * vec2f(dims))), vec2i(0), dims - vec2i(1));
          let surf = load_depth_world(px, dims);
          if (surf.valid) {
            // Reconstruct surface normal from valid neighbors oriented toward the visible (camera) side.
            let sr = load_depth_world(px + vec2i(1, 0), dims);
            let sl = load_depth_world(px - vec2i(1, 0), dims);
            let sd = load_depth_world(px + vec2i(0, 1), dims);
            let su = load_depth_world(px - vec2i(0, 1), dims);
            var dx = vec3f(1.0, 0.0, 0.0);
            if (sr.valid && sl.valid) {
              let dr = sr.world - surf.world;
              let dl = surf.world - sl.world;
              dx = select(dl, dr, dot(dr, dr) < dot(dl, dl));
            } else if (sr.valid) {
              dx = sr.world - surf.world;
            } else if (sl.valid) {
              dx = surf.world - sl.world;
            }
            var dy = vec3f(0.0, 0.0, 1.0);
            if (sd.valid && su.valid) {
              let dd = sd.world - surf.world;
              let du = surf.world - su.world;
              dy = select(du, dd, dot(dd, dd) < dot(du, du));
            } else if (sd.valid) {
              dy = sd.world - surf.world;
            } else if (su.valid) {
              dy = surf.world - su.world;
            }
            let raw_n = cross(dx, dy);
            var n = select(vec3f(0.0, 1.0, 0.0), normalize(raw_n), dot(raw_n, raw_n) > 1e-12 * dot(dx, dx) * dot(dy, dy));
            let to_cam = params.camera_pos_and_clear.xyz - surf.world;
            if (dot(n, to_cam) < 0.0) {
              n = -n;
            }

            // World-space signed distances to surface plane along outward normal `n`.
            let radius = max(p.size_rot_gen.x * 0.5, 0.01);
            let thick = max(params.collision.z, radius);
            let d_prev = dot(prev_pos - surf.world, n);
            let d_next = dot(sample_pos - surf.world, n);
            let vn = dot(vel, n);
            // Accept collision only when moving into the visible surface (vn < 0) and crossing or entering the thickness slab.
            let crossed = d_prev >= -radius && d_next <= radius && d_next >= -thick;
            if (vn < 0.0 && crossed) {
              let restitution = clamp(params.collision.x, 0.0, 1.0);
              let friction = clamp(params.collision.y, 0.0, 1.0);
              let v_normal = n * vn;
              let v_tangent = vel - v_normal;
              vel = v_tangent * (1.0 - friction) - v_normal * restitution;
              next_pos = surf.world + n * (radius + 0.01);
              atomicAdd(&counters.bounces, 1u);
              if (is_primary && params.flags.y > 0.5) {
                enqueue_children(next_pos, n, vel, p.color, p.size_rot_gen.x, idx);
              }
              break;
            }
          }
        }
      }
    }
  }

  p.pos_age = vec4f(next_pos, new_age);
  p.vel_life = vec4f(vel, life);
  p.size_rot_gen.y += p.size_rot_gen.z * dt;
  particles[idx] = p;
}

@compute @workgroup_size(64)
fn apply_spawns(@builtin(global_invocation_id) id: vec3u) {
  let idx = id.x;
  let max_p = u32(params.sub_emit.x);
  if (idx >= max_p) {
    return;
  }
  let total_queued = min(atomicLoad(&counters.spawn_queued), max_p);
  if (total_queued == 0u) {
    return;
  }
  // Claim any dead/retired slot (`vel_life.w <= 0`) to recycle capacity automatically.
  if (particles[idx].vel_life.w <= 0.0) {
    let q = atomicAdd(&counters.spawn_consumed, 1u);
    if (q < total_queued) {
      particles[idx] = spawn_queue[q];
      atomicAdd(&counters.sub_emitted, 1u);
    }
  }
}

var<workgroup> wg_alive_count: atomic<u32>;
var<workgroup> wg_alive_base: u32;

@compute @workgroup_size(64)
fn compact_alive(
  @builtin(global_invocation_id) id: vec3u,
  @builtin(local_invocation_index) lane: u32,
) {
  if (lane == 0u) {
    atomicStore(&wg_alive_count, 0u);
  }
  workgroupBarrier();
  let idx = id.x;
  let max_p = u32(params.sub_emit.x);
  let alive = idx < max_p && particles[idx].vel_life.w > 0.0;
  var local_slot = 0u;
  if (alive) {
    local_slot = atomicAdd(&wg_alive_count, 1u);
  }
  workgroupBarrier();
  if (lane == 0u) {
    let count = atomicLoad(&wg_alive_count);
    if (count > 0u) {
      wg_alive_base = atomicAdd(&indirect.instance_count, count);
    }
  }
  workgroupBarrier();
  if (alive) {
    alive_indices[wg_alive_base + local_slot] = idx;
  }
}
