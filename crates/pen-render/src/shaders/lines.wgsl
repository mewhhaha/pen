struct Uniform {
    view_proj: mat4x4f,
    viewport: vec2f,
    xray_alpha: f32,
    xray_desaturate: f32,
    reversed_z: u32,
}

struct Segment {
    p0: vec3f,
    width_px: f32,
    p1: vec3f,
    dash_len: f32,
    color0: vec4f,
    color1: vec4f,
    params: vec4f,
}

@group(0) @binding(0) var<uniform> frame: Uniform;
@group(0) @binding(1) var<storage, read> segments: array<Segment>;

const W_EPSILON: f32 = 1e-6;
const SCREEN_FLAG: u32 = 2u;

struct LineOut {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(linear) local: vec2f,
    @location(1) @interpolate(linear) color: vec4f,
    @location(2) @interpolate(flat) shape: vec3f, // length, radius, feather
    @location(3) @interpolate(flat) dash: vec3f, // dash, gap, phase
}

fn clip_plane(t: vec2f, d0: f32, d1: f32) -> vec2f {
    if d0 < 0.0 {
        if d1 < 0.0 {
            return vec2f(1.0, 0.0);
        }
        return vec2f(max(t.x, d0 / (d0 - d1)), t.y);
    }
    if d1 < 0.0 {
        return vec2f(t.x, min(t.y, d0 / (d0 - d1)));
    }
    return t;
}

fn screen_clip(p: vec2f) -> vec4f {
    return vec4f(p / frame.viewport * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
}

@vertex
fn vs_line(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> LineOut {
    let line = segments[index];
    var a: vec4f;
    var b: vec4f;
    var t = vec2f(0.0, 1.0);
    if (u32(line.params.w) & SCREEN_FLAG) != 0u {
        a = screen_clip(line.p0.xy);
        b = screen_clip(line.p1.xy);
    } else {
        a = frame.view_proj * vec4f(line.p0, 1.0);
        b = frame.view_proj * vec4f(line.p1, 1.0);
        // Clip the centerline in homogeneous space. The two z planes bracket
        // WebGPU depth for either projection convention.
        t = clip_plane(t, a.w - W_EPSILON, b.w - W_EPSILON);
        t = clip_plane(t, a.z, b.z);
        t = clip_plane(t, a.w - a.z, b.w - b.z);
    }

    var out: LineOut;
    out.position = vec4f(2.0, 2.0, 0.0, 1.0);
    if t.x > t.y {
        return out;
    }

    let ca = mix(a, b, t.x);
    let cb = mix(a, b, t.y);
    let wa = max(ca.w, W_EPSILON);
    let wb = max(cb.w, W_EPSILON);
    let ndc_a = ca.xy / wa;
    let ndc_b = cb.xy / wb;
    let scale = frame.viewport * vec2f(0.5, -0.5);
    let delta = (ndc_b - ndc_a) * scale;
    // Reject unrepresentable projections before squaring or extruding.
    if !all(abs(delta) <= vec2f(1e18)) || !all(abs(ndc_a) <= vec2f(1e18)) || !all(abs(ndc_b) <= vec2f(1e18)) {
        return out;
    }
    let len = length(delta);
    var tangent = vec2f(1.0, 0.0);
    if len > 0.0 {
        tangent = delta / len;
    }
    let normal = vec2f(-tangent.y, tangent.x);
    let radius = line.width_px * 0.5;
    let feather = max(line.params.z, 1e-4);
    let pad = radius + feather * 0.5;

    // Four pairs across the strip separate the two caps from the segment body.
    // Depth is linear along the projected centerline and constant in each cap.
    let column = vertex / 2u;
    let at_end = column >= 2u;
    let along = select(0.0, len, at_end);
    let extension = select(0.0, pad, column == 3u) - select(0.0, pad, column == 0u);
    let across = select(-pad, pad, (vertex & 1u) != 0u);
    let ndc = select(ndc_a, ndc_b, at_end) + (tangent * extension + normal * across) / scale;
    let clipped = select(ca, cb, at_end);
    let w = select(wa, wb, at_end);
    var z = clipped.z;
    if len <= 1e-4 {
        let a_depth = ca.z / wa;
        let b_depth = cb.z / wb;
        z = select(min(a_depth, b_depth), max(a_depth, b_depth), frame.reversed_z != 0u) * w;
    }
    out.position = vec4f(ndc * w, clamp(z, 0.0, w), w);
    out.local = vec2f(along + extension, across);
    out.color = mix(line.color0, line.color1, select(t.x, t.y, at_end));
    out.shape = vec3f(len, radius, feather);
    out.dash = vec3f(line.dash_len, line.params.x, line.params.y);
    return out;
}

fn dash_integral(x: f32, period: f32, dash: f32) -> f32 {
    let periods = floor(x / period);
    return periods * dash + clamp(x - periods * period, 0.0, dash);
}

fn coverage(in: LineOut) -> f32 {
    let along = clamp(in.local.x, 0.0, in.shape.x);
    let distance = length(in.local - vec2f(along, 0.0));
    let radius = in.shape.y;
    let feather = in.shape.z;
    let half_feather = 0.5 * feather;
    // Subtract the opposite edge's filter when the stroke is subpixel.
    var alpha = smoothstep(-half_feather, half_feather, radius - distance)
              - smoothstep(-half_feather, half_feather, -radius - distance);
    let dash = in.dash.x;
    let gap = in.dash.y;
    if dash > 0.0 && gap > 0.0 {
        let period = dash + gap;
        if period < feather * 0.001 {
            return alpha * (dash / period);
        }
        let phase = ((along % period) + in.dash.z) % period;
        let filtered = (dash_integral(phase + half_feather, period, dash)
                      - dash_integral(phase - half_feather, period, dash)) / feather;
        alpha *= clamp(filtered, 0.0, 1.0);
    }
    return alpha;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4f {
    return vec4f(in.color.rgb, clamp(in.color.a, 0.0, 1.0) * coverage(in));
}

@fragment
fn fs_xray(in: LineOut) -> @location(0) vec4f {
    let luma = dot(in.color.rgb, vec3f(0.2126, 0.7152, 0.0722));
    let rgb = mix(in.color.rgb, vec3f(luma), frame.xray_desaturate);
    return vec4f(rgb, clamp(in.color.a, 0.0, 1.0) * coverage(in) * frame.xray_alpha);
}

struct FastOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
}

@vertex
fn vs_fast(@location(0) position: vec3f, @location(1) color: vec4f) -> FastOut {
    return FastOut(frame.view_proj * vec4f(position, 1.0), color);
}

@fragment
fn fs_fast(in: FastOut) -> @location(0) vec4f {
    return vec4f(in.color.rgb, clamp(in.color.a, 0.0, 1.0));
}
