struct Sprite {
    pos: vec2f,
    size: vec2f,
    pivot_rot_layer: vec4f,
    uv_rect: vec4f,
    color: vec4f,
    nine_slice: vec4f,
    flags: vec4f,
}

@group(0) @binding(0) var<uniform> camera: mat4x4f;
@group(0) @binding(1) var<storage, read> sprites: array<Sprite>;
@group(0) @binding(2) var atlas: texture_2d<f32>;
@group(0) @binding(3) var atlas_sampler: sampler;

const QUAD = array<vec2f, 6>(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1),
                            vec2f(0, 1), vec2f(1, 0), vec2f(1, 1));

struct Vertex {
    @builtin(position) position: vec4f,
    @location(0) local: vec2f,
    @location(1) @interpolate(flat) index: u32,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> Vertex {
    let sprite = sprites[index];
    let local = QUAD[vertex];
    let offset = (local - sprite.pivot_rot_layer.xy) * sprite.size * vec2f(1, -1);
    let c = cos(sprite.pivot_rot_layer.z);
    let s = sin(sprite.pivot_rot_layer.z);
    let world = sprite.pos + mat2x2f(vec2f(c, -s), vec2f(s, c)) * offset;
    return Vertex(camera * vec4f(world, 0, 1), local, index);
}

fn fit_borders(borders: vec2f, span: f32) -> vec2f {
    let positive = max(borders, vec2f(0));
    return positive * min(1.0, span / max(positive.x + positive.y, 1e-6));
}

fn slice_axis(uv: f32, dest: f32, source: f32, borders: vec2f) -> f32 {
    let d = fit_borders(borders, dest);
    let s = fit_borders(borders, source);
    let x = uv * dest;
    if max(borders.x, 0.0) + max(borders.y, 0.0) >= dest {
        if d.x > 0.0 && (x <= d.x || d.y == 0.0) {
            let inset = min(0.5, s.x * 0.5);
            return clamp(x / d.x * s.x, inset, s.x - inset);
        }
        if d.y > 0.0 {
            let inset = min(0.5, s.y * 0.5);
            return clamp(source - (dest - x) / d.y * s.y, source - s.y + inset, source - inset);
        }
        return uv * source;
    }
    if x < d.x {
        return x / d.x * s.x;
    }
    if x > dest - d.y {
        return source - (dest - x) / d.y * s.y;
    }
    return s.x + (x - d.x) / max(dest - d.x - d.y, 1e-6) * (source - s.x - s.y);
}

@fragment
fn fs_main(vertex: Vertex) -> @location(0) vec4f {
    let sprite = sprites[vertex.index];
    let uv = select(vertex.local, 1.0 - vertex.local, sprite.flags.xy > vec2f(0.5));
    let atlas_size = select(vec2f(textureDimensions(atlas)), sprite.flags.zw, sprite.flags.zw > vec2f(0));
    let uv_span = sprite.uv_rect.zw - sprite.uv_rect.xy;
    let source_size = abs(uv_span) * atlas_size;
    var texel = uv * source_size;
    if any(sprite.nine_slice > vec4f(0)) {
        texel = vec2f(
            slice_axis(uv.x, abs(sprite.size.x), source_size.x, sprite.nine_slice.xz),
            slice_axis(uv.y, abs(sprite.size.y), source_size.y, sprite.nine_slice.yw),
        );
    }
    let inset = min(vec2f(0.5), source_size * 0.5);
    let sample_uv = sprite.uv_rect.xy + sign(uv_span) * clamp(texel, inset, source_size - inset) / atlas_size;
    return textureSampleLevel(atlas, atlas_sampler, sample_uv, 0.0) * sprite.color;
}
