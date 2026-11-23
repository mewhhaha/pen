// Auto exposure's first half: a histogram of the picture's luminance. Each pixel
// falls in one of 256 bins, evenly spaced in log2 of its linear luminance over the
// frame's range (the first and last bins also take what lies beyond), counted by
// its weight in 1/256ths (1 everywhere, or falling off toward the edges of the
// picture as the frame's centre weighting asks). A workgroup counts in its own
// shared memory first, then adds its bins to the buffer's.
${transfer}

struct Meter {
  // x the lowest bin's log2 luminance, y bins per stop, z the centre weighting.
  range: vec4f,
}
@group(0) @binding(0) var<uniform> meter: Meter;
@group(0) @binding(1) var scene: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>, 256>;

var<workgroup> local: array<atomic<u32>, 256>;

@compute @workgroup_size(16, 16)
fn bins(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) index: u32) {
  atomicStore(&local[index], 0u);
  workgroupBarrier();
  let size = textureDimensions(scene);
  if (id.x < size.x && id.y < size.y) {
    let light = luminance(srgb_decode(textureLoad(scene, vec2i(id.xy), 0).rgb));
    // A pixel that is not a number counts for nothing; black and darker fall in the
    // first bin, brighter in the last.
    if (light == light) {
      var bin = 0u;
      if (light > 0.0) {
        bin = u32(clamp((log2(light) - meter.range.x) * meter.range.y, 0.0, 255.0));
      }
      let from_centre = (vec2f(id.xy) + 0.5) / vec2f(size) - 0.5;
      let weight = mix(1.0, exp(-6.0 * dot(from_centre, from_centre)), meter.range.z);
      atomicAdd(&local[bin], u32(weight * 256.0 + 0.5));
    }
  }
  workgroupBarrier();
  atomicAdd(&histogram[index], atomicLoad(&local[index]));
}
