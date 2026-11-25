const STYLE_MASK: u32 = 0xffffu;
const OCCLUDED: u32 = 0x10000u;
const STYLE_COUNT: u32 = 256u;

struct MeshDraw {
    model: mat4x4f,
    view_proj: mat4x4f,
    packed_style: u32,
    reversed_z: u32,
    depth_bias: f32,
    _pad: u32,
}

struct Flood {
    size: vec2u,
    step: i32,
    _pad: u32,
}

struct OutlineStyleGpu {
    color: vec4f,
    occluded_color: vec4f,
    width_px: f32,
    feather_px: f32,
    fill_alpha: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> mesh: MeshDraw;
@group(0) @binding(1) var scene_depth: texture_depth_2d;
@group(0) @binding(2) var mask_tex: texture_2d<u32>;
@group(0) @binding(3) var<storage, read> source: array<vec4u>;
@group(0) @binding(4) var<storage, read_write> destination: array<vec4u>;
@group(0) @binding(5) var<uniform> flood: Flood;
@group(0) @binding(6) var<uniform> styles: array<OutlineStyleGpu, 256>;
@group(0) @binding(7) var scene_depth_r32: texture_2d<f32>;
@group(0) @binding(8) var<storage, read> joints: array<mat4x4f>;
@group(0) @binding(9) var<uniform> max_radius: u32;

@vertex
fn mask_vs(@location(0) position: vec3f) -> @builtin(position) vec4f {
    return mesh.view_proj * (mesh.model * vec4f(position, 1.0));
}

@vertex
fn mask_skinned_vs(
    @location(0) position: vec3f, @location(9) indices: vec4u, @location(10) weights: vec4f,
) -> @builtin(position) vec4f {
    let skin = joints[indices.x] * weights.x + joints[indices.y] * weights.y
        + joints[indices.z] * weights.z + joints[indices.w] * weights.w;
    return mesh.view_proj * (mesh.model * (skin * vec4f(position, 1.0)));
}

@fragment
fn mask_fs() -> @location(0) u32 {
    return mesh.packed_style;
}

@fragment
fn depth_mask_fs(@builtin(position) position: vec4f) -> @location(0) u32 {
    let depth = textureLoad(scene_depth, vec2i(position.xy), 0);
    let behind = select(position.z - depth, depth - position.z, mesh.reversed_z != 0u);
    let slope_bias = max(abs(dpdx(position.z)), abs(dpdy(position.z))) * 1.5;
    let bias = max(mesh.depth_bias, slope_bias);
    return mesh.packed_style | select(0u, OCCLUDED, behind > bias);
}

@fragment
fn depth_r32_mask_fs(@builtin(position) position: vec4f) -> @location(0) u32 {
    let depth = textureLoad(scene_depth_r32, vec2i(position.xy), 0).r;
    let behind = select(position.z - depth, depth - position.z, mesh.reversed_z != 0u);
    let slope_bias = max(abs(dpdx(position.z)), abs(dpdy(position.z))) * 1.5;
    let bias = max(mesh.depth_bias, slope_bias);
    return mesh.packed_style | select(0u, OCCLUDED, behind > bias);
}

fn pixel_index(pixel: vec2u, width: u32) -> u32 {
    return pixel.y * width + pixel.x;
}

fn has_style(packed: u32) -> bool {
    let style = packed & STYLE_MASK;
    return style > 0u && style <= STYLE_COUNT;
}

// Each seed stores integer pixel coordinates and its original mask value.
// A zero mask value denotes an empty seed.
@compute @workgroup_size(8, 8)
fn init_seeds(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(mask_tex);
    if any(id.xy >= size) {
        return;
    }
    let packed = textureLoad(mask_tex, vec2i(id.xy), 0).r;
    destination[pixel_index(id.xy, size.x)] = select(
        vec4u(0u), vec4u(id.xy, packed, 0u), has_style(packed)
    );
}

fn distance_squared(pixel: vec2i, seed: vec2u) -> u32 {
    let delta = pixel - vec2i(seed);
    return u32(delta.x * delta.x) + u32(delta.y * delta.y);
}

@compute @workgroup_size(8, 8)
fn jfa_step(@builtin(global_invocation_id) id: vec3u) {
    if any(id.xy >= flood.size) {
        return;
    }
    let index = pixel_index(id.xy, flood.size.x);
    var best = source[index];
    if best.z != 0u && all(best.xy == id.xy) {
        destination[index] = best;
        return;
    }
    let pixel = vec2i(id.xy);
    var best_distance = select(0xffffffffu, distance_squared(pixel, best.xy), best.z != 0u);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor = pixel + vec2i(x, y) * flood.step;
            if (x == 0 && y == 0) || any(neighbor < vec2i(0)) || any(neighbor >= vec2i(flood.size)) {
                continue;
            }
            let candidate = source[pixel_index(vec2u(neighbor), flood.size.x)];
            if candidate.z == 0u {
                continue;
            }
            let distance = distance_squared(pixel, candidate.xy);
            // Break exact ties in row-major order, independent of style and dispatch order.
            let earlier = pixel_index(candidate.xy, flood.size.x) < pixel_index(best.xy, flood.size.x);
            if distance < best_distance || (distance == best_distance && earlier) {
                best = candidate;
                best_distance = distance;
            }
        }
    }
    destination[index] = best;
}

@vertex
fn fullscreen_vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
}

fn style_color(style: OutlineStyleGpu, packed: u32) -> vec4f {
    return select(style.color, style.occluded_color, (packed & OCCLUDED) != 0u);
}

@fragment
fn composite(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let pixel = vec2u(position.xy);
    let size = textureDimensions(mask_tex);
    if any(pixel >= size) {
        return vec4f(0.0);
    }
    let mask = textureLoad(mask_tex, vec2i(pixel), 0).r;
    if has_style(mask) {
        let style = styles[(mask & STYLE_MASK) - 1u];
        let color = style_color(style, mask);
        return vec4f(color.rgb, clamp(color.a, 0.0, 1.0) * style.fill_alpha);
    }

    let seed = source[pixel_index(pixel, size.x)];
    if seed.z == 0u {
        return vec4f(0.0);
    }
    let style = styles[(seed.z & STYLE_MASK) - 1u];
    if style.width_px <= 0.0 {
        return vec4f(0.0);
    }
    let distance_sq = distance_squared(vec2i(pixel), seed.xy);
    if distance_sq > max_radius * max_radius {
        return vec4f(0.0);
    }
    let distance = sqrt(f32(distance_sq));
    var coverage = select(0.0, 1.0, distance <= style.width_px);
    if style.feather_px > 0.0 {
        let inner = style.width_px - style.feather_px;
        let outer = style.width_px + style.feather_px;
        if outer > inner {
            coverage = 1.0 - smoothstep(inner, outer, distance);
        }
    }
    let color = style_color(style, seed.z);
    return vec4f(color.rgb, clamp(color.a, 0.0, 1.0) * coverage);
}
