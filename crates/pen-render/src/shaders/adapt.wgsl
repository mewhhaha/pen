// Auto exposure's second half, one invocation: the histogram's trimmed average,
// the exposure it asks for, and a step toward it.
//
// The average is of the bins' log2 luminance (the geometric mean of the light)
// over the part of the weight between two percentiles, a bin that straddles one
// counting for the share inside. Exposure in stops brings that average to the
// frame's middle grey, plus the compensation, within the limits. Each frame closes
// 1 - exp(-rate * dt) of the distance to it, at the rate for brightening or for
// darkening, so splitting a span of time into frames changes nothing; the first
// frame, and a cut, jump there.
struct Adapt {
  // x range low, y range high (log2 luminance), z the low percentile, w the high.
  range: vec4f,
  // x compensation, y least, z most exposure, w middle grey (linear).
  limits: vec4f,
  // x brighten rate, y darken rate, z seconds, w 1 to jump.
  speeds: vec4f,
}
@group(0) @binding(0) var<uniform> adapt: Adapt;
@group(0) @binding(1) var<storage, read> histogram: array<u32, 256>;
// Exposure in stops (what the frame is shown with), the metered average, 1 once it
// holds an exposure, and the exposure asked for.
@group(0) @binding(2) var<storage, read_write> state: array<f32, 4>;

@compute @workgroup_size(1)
fn exposure_step() {
  var sum = 0u;
  for (var bin = 0u; bin < 256u; bin++) { sum += histogram[bin]; }
  if (sum == 0u) { return; }
  let total = f32(sum);
  let width = (adapt.range.y - adapt.range.x) / 256.0;
  var seen = 0u;
  var weight = 0.0;
  var logs = 0.0;
  for (var bin = 0u; bin < 256u; bin++) {
    let before = f32(seen) / total;
    seen += histogram[bin];
    let after = f32(seen) / total;
    let inside = max(0.0, min(after, adapt.range.w) - max(before, adapt.range.z));
    weight += inside;
    logs += inside * (adapt.range.x + (f32(bin) + 0.5) * width);
  }
  if (weight <= 0.0) { return; }
  let average = logs / weight;
  let wanted = clamp(log2(adapt.limits.w) - average + adapt.limits.x, adapt.limits.y, adapt.limits.z);
  var stops = wanted;
  if (state[2] > 0.5 && adapt.speeds.w < 0.5) {
    let rate = select(adapt.speeds.y, adapt.speeds.x, wanted > state[0]);
    stops = state[0] + (wanted - state[0]) * (1.0 - exp(-rate * adapt.speeds.z));
  }
  state[0] = clamp(stops, adapt.limits.y, adapt.limits.z);
  state[1] = average;
  state[2] = 1.0;
  state[3] = wanted;
}
