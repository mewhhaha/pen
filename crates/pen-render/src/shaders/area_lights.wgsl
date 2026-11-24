// Linearly Transformed Cosines (LTC) Rectangular, Disk, and Tube Area Lights
// (Eric Heitz et al., SIGGRAPH 2016) + Projected 2D Light Cookies / Gobo Projectors.

const LTC_PI: f32 = 3.14159265359;
const LTC_LUT_SIZE: f32 = 64.0;
const LTC_LUT_SCALE: f32 = (64.0 - 1.0) / 64.0;
const LTC_LUT_BIAS: f32 = 0.5 / 64.0;
// Area-matched regular octagon radius multiplier: sqrt(pi / (2 * sqrt(2))).
const LTC_OCTAGON_RADIUS_SCALE: f32 = 1.0539073;

// Kind constants: 0 = Rect, 1 = Disk, 2 = Tube, 3 = Cookie Spotlight
struct AreaLight {
    // xyz: world center, w: kind (0..3)
    position_kind: vec4f,
    // xyz: unit right/axis vector, w: half-width (rect), radius (disk), half-length (tube), tan_half_fov (cookie)
    right_extent: vec4f,
    // xyz: unit up vector, w: half-height (rect), radius (disk), tube_radius (tube), aspect (cookie)
    up_extent: vec4f,
    // rgb: linear color * intensity, w: max reach (<=0 unbounded)
    color_reach: vec4f,
    // x: two_sided (>0.5), y: cookie_enabled (>0.5), z: cookie_soft_edge, w: cookie_fov_scale
    flags: vec4f,
};

// Exact spherical polygon edge integral vector: cross(v1, v2) * theta / (2 * pi * sin(theta)).
// Uses atan2(|v1 x v2|, dot(v1, v2)) so near-coincident and near-antiparallel edges stay exact.
fn ltc_integrate_edge(v1: vec3f, v2: vec3f) -> vec3f {
    let c = cross(v1, v2);
    let sin_t = length(c);
    if (sin_t <= 1e-6) {
        return vec3f(0.0);
    }
    let cos_t = clamp(dot(v1, v2), -1.0, 1.0);
    let theta = atan2(sin_t, cos_t);
    return c * (theta / (2.0 * LTC_PI * sin_t));
}

// Clips a 4-vertex polygon against the z >= 0 horizon plane and integrates its projected solid angle.
// Caller performs one-sided emitter facing cull in world space before LTC transform.
fn ltc_eval_quad_polygon(
    p0: vec3f,
    p1: vec3f,
    p2: vec3f,
    p3: vec3f,
) -> f32 {
    let poly = array<vec3f, 4>(p0, p1, p2, p3);
    var clipped = array<vec3f, 5>(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
    var out_n = 0u;
    for (var i = 0u; i < 4u; i++) {
        let cur = poly[i];
        let nxt = poly[(i + 1u) % 4u];
        let cur_in = cur.z >= 0.0;
        let nxt_in = nxt.z >= 0.0;
        if (cur_in && out_n < 5u) {
            clipped[out_n] = cur;
            out_n += 1u;
        }
        if (cur_in != nxt_in && out_n < 5u) {
            let denom = cur.z - nxt.z;
            let t = select(0.5, cur.z / denom, abs(denom) > 1e-7);
            clipped[out_n] = mix(cur, nxt, clamp(t, 0.0, 1.0));
            out_n += 1u;
        }
    }
    if (out_n < 3u) {
        return 0.0;
    }
    // Filter degenerate/coincident vertices near the shading point to prevent normalize(0).
    var dirs = array<vec3f, 5>(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
    var valid_n = 0u;
    for (var i = 0u; i < out_n; i++) {
        let len2 = dot(clipped[i], clipped[i]);
        if (len2 > 1e-10) {
            let d = clipped[i] * inverseSqrt(len2);
            if (valid_n == 0u || dot(d - dirs[valid_n - 1u], d - dirs[valid_n - 1u]) > 1e-10) {
                dirs[valid_n] = d;
                valid_n += 1u;
            }
        }
    }
    if (valid_n < 3u) {
        return 0.0;
    }
    var sum = vec3f(0.0);
    for (var i = 0u; i < valid_n; i++) {
        sum += ltc_integrate_edge(dirs[i], dirs[(i + 1u) % valid_n]);
    }
    // `ltc_integrate_edge` already includes the 1 / (2 * PI) normalization factor.
    return clamp(abs(sum.z), 0.0, 1.0);
}

fn ltc_basis(n: vec3f, v: vec3f) -> mat3x3f {
    let v_proj = v - n * dot(v, n);
    let len2 = dot(v_proj, v_proj);
    var t1: vec3f;
    if (len2 > 1e-6) {
        t1 = v_proj * inverseSqrt(len2);
    } else {
        let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
        t1 = normalize(cross(up, n));
    }
    let t2 = cross(n, t1);
    return transpose(mat3x3f(t1, t2, n));
}

fn ltc_range_window(dist: f32, reach: f32) -> f32 {
    if (reach <= 0.0) {
        return 1.0;
    }
    if (dist >= reach) {
        return 0.0;
    }
    let r = dist / reach;
    let w = clamp(1.0 - r * r * r * r, 0.0, 1.0);
    return w * w;
}

fn ltc_sample_cookie(
    light: AreaLight,
    world: vec3f,
    cookie_tex: texture_2d<f32>,
    cookie_sampler: sampler,
) -> vec3f {
    if (light.flags.y <= 0.5) {
        return vec3f(1.0);
    }
    let right = normalize(light.right_extent.xyz);
    let up = normalize(light.up_extent.xyz);
    let forward = normalize(cross(right, up));
    let rel = world - light.position_kind.xyz;
    let lz = dot(rel, forward);
    if (lz <= 1e-4) {
        return vec3f(0.0);
    }
    let tan_fov = max(light.flags.w, 0.05);
    let aspect = max(light.up_extent.w, 0.1);
    let u = 0.5 + dot(rel, right) / (2.0 * lz * tan_fov * aspect);
    let v = 0.5 - dot(rel, up) / (2.0 * lz * tan_fov);
    if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) {
        return vec3f(0.0);
    }
    let edge = max(light.flags.z, 0.005);
    let mask = smoothstep(0.0, edge, u) * smoothstep(0.0, edge, 1.0 - u)
        * smoothstep(0.0, edge, v) * smoothstep(0.0, edge, 1.0 - v);
    return textureSampleLevel(cookie_tex, cookie_sampler, vec2f(u, v), 0.0).rgb * mask;
}

// Evaluates combined diffuse + specular irradiance from one AreaLight using LTC LUTs.
// Emitter front facing direction is `normalize(cross(right, up))`.
fn eval_area_light(
    light: AreaLight,
    world: vec3f,
    normal: vec3f,
    view: vec3f,
    albedo: vec3f,
    f0: vec3f,
    roughness: f32,
    ltc_mat_tex: texture_2d<f32>,
    ltc_amp_tex: texture_2d<f32>,
    cookie_tex: texture_2d<f32>,
    lut_sampler: sampler,
) -> vec3f {
    let n = normalize(normal);
    let v = normalize(view);
    let ndotv = clamp(dot(n, v), 0.0, 1.0);
    let r_clamped = clamp(roughness, 0.04, 1.0);
    let uv = vec2f(r_clamped, sqrt(max(1.0 - ndotv, 0.0))) * LTC_LUT_SCALE + vec2f(LTC_LUT_BIAS);

    let t1 = textureSampleLevel(ltc_mat_tex, lut_sampler, uv, 0.0);
    let t2 = textureSampleLevel(ltc_amp_tex, lut_sampler, uv, 0.0);
    let minv = mat3x3f(
        vec3f(t1.x, 0.0, t1.y),
        vec3f(0.0, 1.0, 0.0),
        vec3f(t1.z, 0.0, t1.w),
    );
    let basis = ltc_basis(n, v);
    let spec_xf = minv * basis;
    let two_sided = light.flags.x > 0.5;
    let kind = u32(round(light.position_kind.w));

    let center = light.position_kind.xyz;
    let right_len = length(light.right_extent.xyz);
    let up_len = length(light.up_extent.xyz);
    if (right_len <= 1e-5 || up_len <= 1e-5) {
        return vec3f(0.0);
    }
    let right = light.right_extent.xyz / right_len;
    let up = light.up_extent.xyz / up_len;
    let reach = light.color_reach.w;
    let dist_center = length(center - world);
    let atten = ltc_range_window(dist_center, reach);
    if (atten <= 0.0) {
        return vec3f(0.0);
    }

    var diff_form = 0.0;
    var spec_form = 0.0;

    if (kind == 0u || kind == 1u) {
        // One-sided facing test against emitter normal = cross(right, up).
        let emit_raw = cross(right, up);
        let emit_len = length(emit_raw);
        if (emit_len <= 1e-5) {
            return vec3f(0.0);
        }
        let emit_n = emit_raw / emit_len;
        if (!two_sided && dot(emit_n, world - center) <= 0.0) {
            return vec3f(0.0);
        }
    }

    if (kind == 0u) {
        let ex = right * light.right_extent.w;
        let ey = up * light.up_extent.w;
        let q0 = (center - ex - ey) - world;
        let q1 = (center + ex - ey) - world;
        let q2 = (center + ex + ey) - world;
        let q3 = (center - ex + ey) - world;
        diff_form = ltc_eval_quad_polygon(basis * q0, basis * q1, basis * q2, basis * q3);
        spec_form = ltc_eval_quad_polygon(spec_xf * q0, spec_xf * q1, spec_xf * q2, spec_xf * q3);
    } else if (kind == 1u) {
        // Area-matched regular octagon decomposed into 3 non-overlapping LTC quads.
        let rad = max(light.right_extent.w, 1e-4) * LTC_OCTAGON_RADIUS_SCALE;
        let d = rad * 0.70710678;
        let c0 = center + right * rad - world;
        let c1 = center + right * d + up * d - world;
        let c2 = center + up * rad - world;
        let c3 = center - right * d + up * d - world;
        let c4 = center - right * rad - world;
        let c5 = center - right * d - up * d - world;
        let c6 = center - up * rad - world;
        let c7 = center + right * d - up * d - world;
        diff_form = ltc_eval_quad_polygon(basis * c0, basis * c1, basis * c2, basis * c3)
            + ltc_eval_quad_polygon(basis * c4, basis * c5, basis * c6, basis * c7)
            + ltc_eval_quad_polygon(basis * c0, basis * c3, basis * c4, basis * c7);
        spec_form = ltc_eval_quad_polygon(spec_xf * c0, spec_xf * c1, spec_xf * c2, spec_xf * c3)
            + ltc_eval_quad_polygon(spec_xf * c4, spec_xf * c5, spec_xf * c6, spec_xf * c7)
            + ltc_eval_quad_polygon(spec_xf * c0, spec_xf * c3, spec_xf * c4, spec_xf * c7);
    } else if (kind == 2u) {
        let half_len = max(light.right_extent.w, 1e-4);
        let tube_r = max(light.up_extent.w, 0.01);
        let to_p = world - center;
        let perp_raw = to_p - right * dot(to_p, right);
        let perp_dir = select(up, normalize(perp_raw), dot(perp_raw, perp_raw) > 1e-6);
        let side = normalize(cross(right, perp_dir));
        let ex = right * half_len;
        let ey = side * tube_r;
        let q0 = (center - ex - ey) - world;
        let q1 = (center + ex - ey) - world;
        let q2 = (center + ex + ey) - world;
        let q3 = (center - ex + ey) - world;
        diff_form = ltc_eval_quad_polygon(basis * q0, basis * q1, basis * q2, basis * q3);
        spec_form = ltc_eval_quad_polygon(spec_xf * q0, spec_xf * q1, spec_xf * q2, spec_xf * q3);
    } else {
        let to_l = center - world;
        let dist = max(length(to_l), 1e-4);
        let l = to_l / dist;
        let ndotl = max(dot(n, l), 0.0);
        let inv_sq = 1.0 / (dist * dist + 0.25);
        diff_form = ndotl * inv_sq;
        let h = normalize(l + v);
        let ndoth = max(dot(n, h), 0.0);
        let a2 = max(r_clamped * r_clamped * r_clamped * r_clamped, 1e-4);
        let denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
        let ggx = a2 / max(LTC_PI * denom * denom, 1e-5);
        spec_form = ggx * ndotl * inv_sq * 0.25;
    }

    let cookie = ltc_sample_cookie(light, world, cookie_tex, lut_sampler);
    // Split-sum Fresnel: t2.x = E[(1 - Fc) G_vis], t2.y = E[Fc G_vis] -> F0 * t2.x + t2.y.
    let spec_tint = f0 * t2.x + vec3f(t2.y);
    let diffuse = albedo * diff_form;
    let specular = spec_tint * spec_form;
    return light.color_reach.rgb * cookie * atten * (diffuse + specular);
}
