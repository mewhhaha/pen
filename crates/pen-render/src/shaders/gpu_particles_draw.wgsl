// Instanced 3D Mesh Particle Renderer driven by GPU-compacted alive indices and DrawIndexedIndirect.

struct GpuParticle {
  pos_age: vec4f,
  vel_life: vec4f,
  color: vec4f,
  size_rot_gen: vec4f,
};

struct DrawUniform {
  view_projection: mat4x4f,
  sun_dir_and_ambient: vec4f, // xyz = toward sun, w = ambient level
  sun_color: vec4f,           // rgb = sun radiance, w = velocity_align (>0.5 aligns mesh +Y with velocity)
};

@group(0) @binding(0) var<uniform> u: DrawUniform;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read> alive_indices: array<u32>;

struct VertexIn {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
};

struct VsOut {
  @builtin(position) clip_pos: vec4f,
  @location(0) world_normal: vec3f,
  @location(1) color: vec4f,
};

@vertex fn vs_main(v: VertexIn, @builtin(instance_index) iid: u32) -> VsOut {
  let p = particles[alive_indices[iid]];
  let center = p.pos_age.xyz;
  let age_frac = clamp(p.pos_age.w / max(p.vel_life.w, 1e-4), 0.0, 1.0);
  // Smooth birth/death scale envelope.
  let scale = p.size_rot_gen.x * smoothstep(0.0, 0.08, age_frac + 0.08) * (1.0 - smoothstep(0.85, 1.0, age_frac));
  let rot = p.size_rot_gen.y;

  // Orthonormal basis: either velocity-aligned (+Y along velocity) or spinning around +Y.
  let speed = length(p.vel_life.xyz);
  var up = vec3f(0.0, 1.0, 0.0);
  if (u.sun_color.w > 0.5 && speed > 1e-3) {
    up = p.vel_life.xyz / speed;
  }
  let ref_axis = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(up.y) > 0.9);
  let r0 = normalize(cross(ref_axis, up));
  let f0 = cross(r0, up);
  let c = cos(rot);
  let s = sin(rot);
  let right = r0 * c + f0 * s;
  let fwd = f0 * c - r0 * s;

  let local_pos = v.position * scale;
  let world_pos = center + right * local_pos.x + up * local_pos.y + fwd * local_pos.z;
  let world_normal = normalize(right * v.normal.x + up * v.normal.y + fwd * v.normal.z);
  let alpha = p.color.a * (1.0 - smoothstep(0.8, 1.0, age_frac));
  return VsOut(u.view_projection * vec4f(world_pos, 1.0), world_normal, vec4f(p.color.rgb, alpha));
}

@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
  let n = normalize(in.world_normal);
  let l = normalize(u.sun_dir_and_ambient.xyz);
  let ndotl = max(dot(n, l), 0.0);
  let lit = in.color.rgb * (u.sun_color.rgb * ndotl + vec3f(u.sun_dir_and_ambient.w));
  return vec4f(lit, in.color.a);
}
