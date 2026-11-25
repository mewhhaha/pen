struct Shape {
    rect: vec4f,
    corner_radii: vec4f,
    fill_color: vec4f,
    fill_color_end: vec4f,
    border_color: vec4f,
    shadow_color: vec4f,
    kind: u32,
    border_width_px: f32,
    gradient_kind: u32,
    gradient_angle: f32,
    shadow_offset: vec2f,
    shadow_blur_px: f32,
    clip_radius_px: f32,
    clip_rect: vec4f,
    gradient_center: vec2f,
    gradient_radius_px: f32,
    rotation: f32,
}

@group(0) @binding(0) var<uniform> viewport: vec4f;
@group(0) @binding(1) var<storage, read> shapes: array<Shape>;

const QUAD = array<vec2f, 6>(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1),
                            vec2f(0, 1), vec2f(1, 0), vec2f(1, 1));

struct Vertex {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) index: u32,
}

fn rotation(angle: f32) -> mat2x2f {
    let c = cos(angle);
    let s = sin(angle);
    return mat2x2f(vec2f(c, s), vec2f(-s, c));
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> Vertex {
    let shape = shapes[index];
    let half_size = shape.rect.zw * 0.5;
    let r = rotation(shape.rotation);
    let bounds = abs(r[0]) * half_size.x + abs(r[1]) * half_size.y;
    let shadow_extent = 4.0 * max(shape.shadow_blur_px, 0.0);
    let low = -bounds + min(vec2f(0), shape.shadow_offset - shadow_extent) - 1.0;
    let high = bounds + max(vec2f(0), shape.shadow_offset + shadow_extent) + 1.0;
    let screen = shape.rect.xy + half_size + mix(low, high, QUAD[vertex]);
    let ndc = screen / viewport.xy * vec2f(2, -2) + vec2f(-1, 1);
    return Vertex(vec4f(ndc, 0, 1), index);
}

fn segment_distance(p: vec2f, a: vec2f, b: vec2f) -> f32 {
    let edge = b - a;
    let t = clamp(dot(p - a, edge) / max(dot(edge, edge), 1e-12), 0.0, 1.0);
    return length(p - a - t * edge);
}

fn box_distance(p: vec2f, half_size: vec2f, radius: f32) -> f32 {
    let q = abs(p) - half_size + radius;
    return length(max(q, vec2f(0))) + min(max(q.x, q.y), 0.0) - radius;
}

fn rounded_rect(p: vec2f, size: vec2f, radii: vec4f) -> f32 {
    if all(radii == vec4f(radii.x)) {
        return box_distance(p, size * 0.5, clamp(radii.x, 0.0, min(size.x, size.y) * 0.5));
    }
    // CSS-style normalization preserves relative radii when adjacent arcs meet.
    let positive = max(radii, vec4f(0));
    let sums = positive + positive.yzwx;
    let fit = size.xyxy / max(sums, vec4f(1e-6));
    let r = positive * min(1.0, min(min(fit.x, fit.y), min(fit.z, fit.w)));
    let q = p + size * 0.5;
    let centers = array<vec2f, 4>(vec2f(r.x), vec2f(size.x - r.y, r.y),
                                size - r.z, vec2f(r.w, size.y - r.w));
    let signs = array<vec2f, 4>(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, 1));
    let starts = array<vec2f, 4>(vec2f(r.x, 0), vec2f(size.x, r.y),
                               vec2f(size.x - r.z, size.y), vec2f(0, size.y - r.w));
    let ends = array<vec2f, 4>(vec2f(size.x - r.y, 0), size - vec2f(0, r.z),
                             vec2f(r.w, size.y), vec2f(0, r.x));
    var distance = 1e20;
    var outside = any(q < vec2f(0)) || any(q > size);
    for (var i = 0u; i < 4u; i++) {
        distance = min(distance, segment_distance(q, starts[i], ends[i]));
        let d = (q - centers[i]) * signs[i];
        if all(d >= vec2f(0)) {
            let arc = length(d) - r[i];
            distance = min(distance, abs(arc));
            outside = outside || arc > 0.0;
        }
    }
    return select(-distance, distance, outside);
}

fn ellipse_distance(point: vec2f, half_size: vec2f) -> f32 {
    let swap = half_size.x < half_size.y;
    let axes = max(select(half_size, half_size.yx, swap), vec2f(1e-4));
    let p = abs(select(point, point.yx, swap));
    if axes.x == axes.y {
        return length(p) - axes.x;
    }

    let uv = p / axes;
    let inside = dot(uv, uv) < 1.0;
    let ratio = axes.x / axes.y;
    let r = ratio * ratio;
    var closest = vec2f(axes.x, 0);
    if uv.y > 0.0 {
        // Solve the closest-point Lagrange multiplier on the first quadrant.
        // Shift by one to keep the root bound positive even near the major axis.
        let n = r * uv.x;
        var lo = max(uv.y, n - (r - 1.0));
        var hi = select(max(1.0, length(vec2f(n, uv.y))), 1.0, inside);
        // The major-axis term decreases with the root, so its value at the
        // lower bound gives an upper bound on the remaining minor-axis term.
        let major = n / (lo + (r - 1.0));
        if major < 1.0 {
            hi = min(hi, uv.y / sqrt((1.0 - major) * (1.0 + major)));
        }
        hi = max(lo, hi);
        for (var i = 0u; i < 48u; i++) {
            let mid = (lo + hi) * 0.5;
            if mid == lo || mid == hi {
                break;
            }
            let projection = vec2f(n / (mid + (r - 1.0)), uv.y / mid);
            let too_small = dot(projection, projection) > 1.0;
            lo = select(lo, mid, too_small);
            hi = select(mid, hi, too_small);
        }
        let root = (lo + hi) * 0.5;
        closest = axes * vec2f(n / (root + (r - 1.0)), uv.y / root);
    } else if uv.x < 1.0 - 1.0 / r {
        let x = uv.x / (1.0 - 1.0 / r);
        closest = axes * vec2f(x, sqrt(max(1.0 - x * x, 0.0)));
    }
    return length(p - closest) * select(1.0, -1.0, inside);
}

fn shape_distance(shape: Shape, p: vec2f, half_size: vec2f) -> f32 {
    if shape.kind == 1u {
        return ellipse_distance(p, half_size);
    }
    if shape.kind == 2u {
        let radius = min(half_size.x, half_size.y);
        let line = half_size - radius;
        return length(p - clamp(p, -line, line)) - radius;
    }
    return rounded_rect(p, half_size * 2.0, shape.corner_radii);
}

fn coverage(distance: f32) -> f32 {
    let width = max(fwidth(distance), 1.0);
    return clamp(0.5 - distance / width, 0.0, 1.0);
}

fn gradient(shape: Shape, p: vec2f, half_size: vec2f) -> vec4f {
    var t = 0.0;
    if shape.gradient_kind == 1u {
        let axis = vec2f(cos(shape.gradient_angle), sin(shape.gradient_angle));
        t = 0.5 + 0.5 * dot(p, axis) / max(dot(half_size, abs(axis)), 1e-6);
    } else if shape.gradient_kind == 2u {
        let center = (shape.gradient_center * 2.0 - 1.0) * half_size;
        let specified = shape.gradient_radius_px;
        let radius = select(length(half_size + abs(center)), specified, specified > 0.0);
        t = length(p - center) / max(radius, 1e-6);
    }
    return mix(shape.fill_color, shape.fill_color_end, clamp(t, 0.0, 1.0));
}

// Gaussian convolution of the tangent half-plane, separated from its interior
// by the shape's signed distance. Abramowitz & Stegun's approximation to erfc.
fn shadow_coverage(distance: f32, sigma: f32) -> f32 {
    let x = distance / (sqrt(2.0) * sigma);
    let t = 1.0 / (1.0 + 0.3275911 * abs(x));
    let tail = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                 - 0.284496736) * t + 0.254829592) * t * exp(-x * x) * 0.5;
    return select(tail, 1.0 - tail, x < 0.0);
}

fn premult(color: vec4f, opacity: f32) -> vec4f {
    let alpha = color.a * opacity;
    return vec4f(color.rgb * alpha, alpha);
}

@fragment
fn fs_main(vertex: Vertex) -> @location(0) vec4f {
    let shape = shapes[vertex.index];
    let half_size = max(shape.rect.zw * 0.5, vec2f(0));
    let delta = vertex.position.xy - shape.rect.xy - half_size;
    let inverse = rotation(-shape.rotation);
    let p = inverse * delta;
    let distance = shape_distance(shape, p, half_size);
    let outer = coverage(distance);
    let inner = coverage(distance + max(shape.border_width_px, 0.0));
    let paint = premult(gradient(shape, p, half_size), inner) + premult(shape.border_color, outer - inner);

    var shadow_distance = 1e20;
    if shape.shadow_color.a > 0.0 {
        shadow_distance = shape_distance(shape, inverse * (delta - shape.shadow_offset), half_size);
    }
    let hard_shadow = coverage(shadow_distance);
    var shadow_opacity = hard_shadow;
    if shape.shadow_blur_px > 0.0 && shape.shadow_color.a > 0.0 {
        shadow_opacity = shadow_coverage(shadow_distance, shape.shadow_blur_px);
    }
    let shadow = premult(shape.shadow_color, shadow_opacity);

    let clip_half = max(shape.clip_rect.zw * 0.5, vec2f(0));
    let q = vertex.position.xy - shape.clip_rect.xy - clip_half;
    let radius = clamp(shape.clip_radius_px, 0.0, min(clip_half.x, clip_half.y));
    let sd = box_distance(q, clip_half, radius);
    // AA extends inward only, keeping every pixel outside the clip clear.
    let inner_clip = clamp(-2.0 * sd / max(fwidth(sd), 1.0), 0.0, 1.0);
    let clip = select(select(inner_clip, 0.0, shape.clip_rect.w <= 0.0),
                      1.0, shape.clip_rect.z <= 0.0);
    return (paint + shadow * (1.0 - paint.a)) * clip;
}
