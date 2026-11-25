// Packets hold emitted radiance (RGB) and remaining transmittance (A).
struct Params {
    grid: vec2u,
    dirs: u32,
    spacing: f32,
    far_grid: vec2u,
    far_dirs: u32,
    far_spacing: f32,
    interval: vec2f,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var scene: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> far_packets: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> packets: array<vec4f>;
@group(0) @binding(4) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var irradiance: texture_2d<f32>;
@group(0) @binding(6) var albedo: texture_2d<f32>;
@group(0) @binding(7) var<uniform> weights: vec4f;

const VACUUM = vec4f(0, 0, 0, 1);
const TAU = 6.283185307179586;

fn direction(index: u32, count: u32) -> vec2f {
    let angle = TAU * (f32(index) + 0.5) / f32(count);
    return vec2f(cos(angle), sin(angle));
}

fn compose(near: vec4f, far: vec4f) -> vec4f {
    return vec4f(near.rgb + near.a * far.rgb, near.a * far.a);
}

// Walk every intersected texel with a grid DDA so solid texels cannot be
// skipped, even along diagonals. Clip the segment to the texture first.
fn trace_segment(start: vec2f, end: vec2f) -> vec4f {
    let delta = end - start;
    let segment_length = length(delta);
    if segment_length < 1e-5 {
        return VACUUM;
    }
    let dir = delta / segment_length;
    let positive = dir >= vec2f(0);
    let inv_dir = select(vec2f(-1), vec2f(1), positive) / max(abs(dir), vec2f(1e-8));
    let size = vec2i(textureDimensions(scene));
    let low = -start * inv_dir;
    let high = (vec2f(size) - start) * inv_dir;
    let enter = min(low, high);
    let leave = max(low, high);
    var t = max(0.0, max(enter.x, enter.y));
    let stop = min(segment_length, min(leave.x, leave.y));
    let parallel_outside = (abs(dir) < vec2f(1e-8))
                           & ((start < vec2f(0)) | (start >= vec2f(size)));
    if t >= stop || any(parallel_outside) {
        return VACUUM;
    }

    let at = start + t * dir;
    var cell = clamp(vec2i(select(ceil(at) - 1.0, floor(at), positive)), vec2i(0), size - 1);
    let step = select(vec2i(-1), vec2i(1), positive);
    let edge = vec2f(cell) + select(vec2f(0), vec2f(1), positive);
    var crossing = select((edge - start) * inv_dir, vec2f(1e30), abs(dir) < vec2f(1e-8));
    var packet = VACUUM;
    while t < stop && all(cell >= vec2i(0)) && all(cell < size) {
        let next = min(crossing.x, crossing.y);
        let chord = min(next, stop) - t;
        if chord > 0.0 {
            let material = textureLoad(scene, cell, 0);
            if material.a > 0.0 || any(material.rgb != vec3f(0.0)) {
                let opacity = clamp(material.a, 0.0, 1.0);
                let transmission = select(pow(max(1.0 - opacity, 1e-6), chord), 0.0, opacity >= 1.0);
                let emission_scale = select(chord, (1.0 - transmission) / max(opacity, 1e-5), opacity >= 1e-5);
                packet = compose(packet, vec4f(material.rgb * emission_scale, transmission));
                if packet.a <= 0.0 {
                    break;
                }
            }
        }
        let advance = crossing <= vec2f(next);
        cell += select(vec2i(0), step, advance);
        crossing += select(vec2f(0), abs(inv_dir), advance);
        t = max(t, next);
    }
    return packet;
}

fn position(probe: vec2u, spacing: f32, extent: vec2f) -> vec2f {
    // Keep the last probe of a partial cell inside the scene.
    return min((vec2f(probe) + 0.5) * spacing, extent - 0.5);
}

fn probe_tap(base: vec2i, tap: u32, grid: vec2u) -> vec2u {
    let offset = vec2i(i32(tap & 1u), i32(tap >> 1u));
    return vec2u(clamp(base + offset, vec2i(0), vec2i(grid) - 1));
}

fn probe_blend(point: vec2f, base: vec2i, grid: vec2u, spacing: f32, extent: vec2f) -> vec2f {
    let near = position(probe_tap(base, 0u, grid), spacing, extent);
    let far = position(probe_tap(base, 3u, grid), spacing, extent);
    return clamp((point - near) / max(far - near, vec2f(1e-6)), vec2f(0), vec2f(1));
}

fn tap_weight(blend: vec2f, tap: u32) -> f32 {
    let use_high = (vec2u(tap) & vec2u(1u, 2u)) != vec2u(0);
    let w = select(1.0 - blend, blend, use_high);
    return w.x * w.y;
}

fn packet_index(probe: vec2u, grid: vec2u, dirs: u32, bin: u32) -> u32 {
    return (probe.y * grid.x + probe.x) * dirs + bin;
}

@compute @workgroup_size(64)
fn trace_cascade(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let index = id.x + id.y * groups.x * 64u;
    let probe_index = index / params.dirs;
    if probe_index >= params.grid.x * params.grid.y {
        return;
    }
    let bin = index % params.dirs;
    let probe = vec2u(probe_index % params.grid.x, probe_index / params.grid.x);
    let extent = vec2f(textureDimensions(scene));
    let point = position(probe, params.spacing, extent);
    let ray = direction(bin, params.dirs);
    let start = point + ray * params.interval.x;
    if params.far_dirs == 0u {
        packets[index] = trace_segment(start, point + ray * params.interval.y);
        return;
    }

    let grid_position = point / params.far_spacing - 0.5;
    let base = vec2i(floor(grid_position));
    let blend = probe_blend(point, base, params.far_grid, params.far_spacing, extent);
    var merged = vec4f(0);
    for (var child = 0u; child < 4u; child++) {
        let child_bin = bin * 4u + child;
        let child_ray = direction(child_bin, params.far_dirs);
        var angular = vec4f(0);
        for (var tap = 0u; tap < 4u; tap++) {
            let far_probe = probe_tap(base, tap, params.far_grid);
            // Each near segment meets the start of the selected far segment.
            // Merge transmittance before bilinear filtering so blocked taps
            // cannot borrow radiance from paths that miss the blocker.
            let end = position(far_probe, params.far_spacing, extent) + child_ray * params.interval.y;
            let near = trace_segment(start, end);
            let sample_pkt = select(
                near,
                compose(near, far_packets[packet_index(far_probe, params.far_grid, params.far_dirs, child_bin)]),
                near.a > 0.0,
            );
            angular += sample_pkt * tap_weight(blend, tap);
        }
        merged += angular;
    }
    packets[index] = merged * 0.25;
}

@compute @workgroup_size(8, 8)
fn resolve_cascade(@builtin(global_invocation_id) id: vec3u) {
    if any(id.xy >= textureDimensions(output)) {
        return;
    }
    let point = vec2f(id.xy) + 0.5;
    let base = vec2i(floor(point / params.spacing - 0.5));
    let blend = probe_blend(point, base, params.grid, params.spacing, vec2f(textureDimensions(output)));
    var rgb = vec3f(0);
    for (var tap = 0u; tap < 4u; tap++) {
        let probe = probe_tap(base, tap, params.grid);
        var sum = vec3f(0);
        for (var bin = 0u; bin < params.dirs; bin++) {
            sum += far_packets[packet_index(probe, params.grid, params.dirs, bin)].rgb;
        }
        rgb += sum * tap_weight(blend, tap);
    }
    textureStore(output, id.xy, vec4f(rgb / f32(params.dirs), 1));
}

@compute @workgroup_size(8, 8)
fn composite_irradiance(@builtin(global_invocation_id) id: vec3u) {
    if any(id.xy >= textureDimensions(output)) {
        return;
    }
    let xy = vec2i(id.xy);
    let light = textureLoad(irradiance, xy, 0).rgb;
    let base = textureLoad(albedo, xy, 0);
    let rgb = base.rgb * (weights.x + light * weights.y) + light * weights.z * mix(1.0, base.a, weights.w);
    textureStore(output, id.xy, vec4f(rgb, base.a));
}
