// Raymarched 3D volumetric clouds with procedural Perlin-Worley FBM density,
// cumulus height shaping, dual-lobe Henyey-Greenstein silver lining, Beer-Powder
// multi-scattering energy conservation, and sun light marching toward shell exit.
// Supports cameras below, inside, and above the cloud shell (including downward views).
// Outputs Rgba16Float: rgb = in-scattered cloud radiance, a = transmittance (1 = clear sky).

const PI: f32 = 3.14159265359;

struct CloudUniform {
    inverse_view_projection: mat4x4f,
    // xyz: camera world position (km), w: planet_radius_km (6360)
    camera: vec4f,
    // xyz: unit direction toward sun, w: coverage in 0..1
    sun_dir: vec4f,
    // rgb: solar radiance, w: density scale
    sun_color: vec4f,
    // rgb: sky ambient tint, w: cloud_bottom_km (1.5)
    ambient: vec4f,
    // xyz: wind offset (km), w: cloud_top_km (4.0)
    wind: vec4f,
    // xy: output dimensions, z: detail frequency, w: silver_lining_g (0.8)
    params: vec4f,
    // x: primary ray steps (8..64), y: max march distance km, zw: reserved
    quality: vec4f,
};

@group(0) @binding(0) var<uniform> cloud: CloudUniform;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba16float, write>;

fn hash3(p: vec3f) -> vec3f {
    let q = vec3f(
        dot(p, vec3f(127.1, 311.7, 74.7)),
        dot(p, vec3f(269.5, 183.3, 246.1)),
        dot(p, vec3f(113.5, 271.9, 124.6)),
    );
    return fract(sin(q) * 43758.5453);
}

fn value_noise3(p: vec3f) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash3(i + vec3f(0.0, 0.0, 0.0)).x;
    let n100 = hash3(i + vec3f(1.0, 0.0, 0.0)).x;
    let n010 = hash3(i + vec3f(0.0, 1.0, 0.0)).x;
    let n110 = hash3(i + vec3f(1.0, 1.0, 0.0)).x;
    let n001 = hash3(i + vec3f(0.0, 0.0, 1.0)).x;
    let n101 = hash3(i + vec3f(1.0, 0.0, 1.0)).x;
    let n011 = hash3(i + vec3f(0.0, 1.0, 1.0)).x;
    let n111 = hash3(i + vec3f(1.0, 1.0, 1.0)).x;
    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);
    return mix(mix(nx00, nx10, u.y), mix(nx01, nx11, u.y), u.z);
}

fn worley3(p: vec3f) -> f32 {
    let i = floor(p);
    let f = p - i;
    var min_d2 = 4.0;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let nb = vec3f(f32(x), f32(y), f32(z));
                let jitter = hash3(i + nb);
                let diff = nb + jitter - f;
                min_d2 = min(min_d2, dot(diff, diff));
            }
        }
    }
    return clamp(1.0 - sqrt(min_d2), 0.0, 1.0);
}

fn cloud_density(pos_km: vec3f) -> f32 {
    let rb = cloud.camera.w;
    let bottom = cloud.ambient.w;
    let top = max(cloud.wind.w, bottom + 0.1);
    let alt = length(pos_km) - rb;
    if (alt <= bottom || alt >= top) {
        return 0.0;
    }
    let h = clamp((alt - bottom) / (top - bottom), 0.0, 1.0);
    let height_profile = smoothstep(0.0, 0.18, h) * smoothstep(1.0, 0.65, h);
    let p = (pos_km + cloud.wind.xyz) * max(cloud.params.z, 0.05);
    let base_perlin = 0.55 * value_noise3(p) + 0.30 * value_noise3(p * 2.07 + 3.1) + 0.15 * value_noise3(p * 4.13 - 1.7);
    let billows = worley3(p * 1.45);
    let shape = mix(base_perlin, billows, 0.45);
    let coverage = clamp(cloud.sun_dir.w, 0.0, 1.0);
    let threshold = 1.0 - coverage;
    let raw = clamp((shape - threshold) / max(1.0 - threshold, 0.05), 0.0, 1.0);
    let detail = worley3(p * 3.8 + vec3f(1.3, 2.7, 0.9));
    let eroded = max(0.0, raw - (1.0 - detail) * 0.25 * (1.0 - raw));
    return eroded * height_profile * max(cloud.sun_color.w, 0.0);
}

fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4), 1.5));
}

fn dual_lobe_phase(cos_theta: f32, g_fwd: f32) -> f32 {
    return mix(hg_phase(cos_theta, g_fwd), hg_phase(cos_theta, -0.3), 0.3);
}

fn ray_sphere(origin: vec3f, dir: vec3f, radius: f32) -> vec2f {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let d = b * b - c;
    if (d < 0.0) {
        return vec2f(-1.0, -1.0);
    }
    let s = sqrt(d);
    return vec2f(-b - s, -b + s);
}

// 6-step quadratic-spaced light march to the sun-facing cloud shell exit, zeroing direct
// sunlight when the ray toward the sun is occluded by the planet body.
fn sun_light_energy(pos_km: vec3f, sun_dir: vec3f, cos_theta: f32) -> vec3f {
    let rb = cloud.camera.w;
    let hit_planet = ray_sphere(pos_km, sun_dir, rb);
    if (hit_planet.x > 0.0) {
        return vec3f(0.0);
    }
    let r_top = rb + max(cloud.wind.w, cloud.ambient.w + 0.1);
    let r_bot = rb + cloud.ambient.w;
    let hit_top = ray_sphere(pos_km, sun_dir, r_top);
    let hit_bot = ray_sphere(pos_km, sun_dir, r_bot);
    let thickness = max(r_top - r_bot, 0.2);
    var t_exit = max(hit_top.y, 0.0);
    if (hit_bot.x > 0.0) {
        t_exit = min(t_exit, hit_bot.x);
    }
    t_exit = min(t_exit, thickness * 10.0);
    if (t_exit <= 1e-4) {
        return cloud.sun_color.rgb * dual_lobe_phase(cos_theta, clamp(cloud.params.w, 0.0, 0.92));
    }

    var od = 0.0;
    var t_prev = 0.0;
    for (var i = 1u; i <= 6u; i++) {
        let frac = f32(i) / 6.0;
        let t_next = t_exit * (frac * frac);
        let ds = t_next - t_prev;
        let t_mid = 0.5 * (t_prev + t_next);
        od += cloud_density(pos_km + sun_dir * t_mid) * ds;
        t_prev = t_next;
    }
    let g = clamp(cloud.params.w, 0.0, 0.92);
    let direct = exp(-od) * dual_lobe_phase(cos_theta, g);
    let multi = exp(-od * 0.28) * 0.45 * dual_lobe_phase(cos_theta, g * 0.45);
    let powder = 1.0 - 0.65 * exp(-od * 2.0);
    return cloud.sun_color.rgb * (direct + multi) * powder;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dims = vec2u(cloud.params.xy);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    if (cloud.sun_dir.w <= 0.001 || cloud.sun_color.w <= 0.0001) {
        textureStore(out_tex, vec2i(id.xy), vec4f(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let world_h = cloud.inverse_view_projection * vec4f(ndc, 0.5, 1.0);
    if (abs(world_h.w) <= 1e-6) {
        textureStore(out_tex, vec2i(id.xy), vec4f(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let ray_raw = world_h.xyz / world_h.w - cloud.camera.xyz;
    let ray_len = length(ray_raw);
    if (ray_len <= 1e-6) {
        textureStore(out_tex, vec2i(id.xy), vec4f(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let dir = ray_raw / ray_len;

    let rb = cloud.camera.w;
    let r_bot = rb + cloud.ambient.w;
    let r_top = rb + max(cloud.wind.w, cloud.ambient.w + 0.1);
    let origin = vec3f(cloud.camera.x, rb + max(cloud.camera.y, 0.002), cloud.camera.z);
    let cam_r = length(origin);

    let hit_planet = ray_sphere(origin, dir, rb);
    let hit_bot = ray_sphere(origin, dir, r_bot);
    let hit_top = ray_sphere(origin, dir, r_top);
    let t_planet = select(1e9, hit_planet.x, hit_planet.x > 0.0);

    // Outer shell interval [outer_a, outer_b] clipped to forward ray and planet hit.
    if (hit_top.y <= 0.0) {
        textureStore(out_tex, vec2i(id.xy), vec4f(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let outer_a = max(hit_top.x, 0.0);
    let outer_b = min(hit_top.y, t_planet);
    if (outer_b <= outer_a) {
        textureStore(out_tex, vec2i(id.xy), vec4f(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Subtract inner sphere [hit_bot.x, hit_bot.y] from [outer_a, outer_b] -> up to two segments.
    var seg0 = vec2f(0.0, 0.0);
    var seg1 = vec2f(0.0, 0.0);
    if (cam_r < r_bot) {
        // Camera below clouds: starts when exiting inner sphere (if before planet/outer exit).
        if (hit_bot.y > outer_a && hit_bot.y < outer_b) {
            seg0 = vec2f(hit_bot.y, outer_b);
        }
    } else if (hit_bot.y > outer_a && hit_bot.x < outer_b) {
        // Ray pierces inner sphere: split into entry segment before hit_bot.x and optional exit after hit_bot.y.
        if (hit_bot.x > outer_a) {
            seg0 = vec2f(outer_a, min(hit_bot.x, outer_b));
        }
        if (hit_bot.y < outer_b) {
            seg1 = vec2f(max(hit_bot.y, outer_a), outer_b);
        }
    } else {
        // Ray stays purely in the shell [outer_a, outer_b].
        seg0 = vec2f(outer_a, outer_b);
    }

    let max_march = clamp(cloud.quality.y, 5.0, 120.0);
    if (seg0.y > seg0.x) {
        seg0.y = min(seg0.y, seg0.x + max_march);
    }
    let len0 = max(seg0.y - seg0.x, 0.0);
    let rem = max(max_march - len0, 0.0);
    if (seg1.y > seg1.x) {
        seg1.y = min(seg1.y, seg1.x + rem);
    }
    let len1 = max(seg1.y - seg1.x, 0.0);
    let total_len = len0 + len1;
    if (total_len <= 1e-4) {
        textureStore(out_tex, vec2i(id.xy), vec4f(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let sun_dir = normalize(cloud.sun_dir.xyz);
    let cos_theta = dot(dir, sun_dir);
    let steps = clamp(u32(round(cloud.quality.x)), 8u, 64u);
    let dt = total_len / f32(steps);
    var transmittance = 1.0;
    var scattered = vec3f(0.0);
    let jitter = fract(sin(dot(vec2f(id.xy), vec2f(12.9898, 78.233))) * 43758.5453);

    for (var i = 0u; i < steps; i++) {
        if (transmittance < 0.01) {
            break;
        }
        let s = (f32(i) + jitter) * dt;
        let t = select(seg1.x + (s - len0), seg0.x + s, s < len0);
        let pos = origin + dir * t;
        let dens = cloud_density(pos);
        if (dens > 0.0005) {
            let step_t = exp(-dens * dt);
            let h = clamp((length(pos) - r_bot) / max(r_top - r_bot, 0.1), 0.0, 1.0);
            let sun_l = sun_light_energy(pos, sun_dir, cos_theta);
            let amb_l = cloud.ambient.rgb * mix(0.45, 1.0, h);
            let step_lum = (sun_l + amb_l) * (1.0 - step_t);
            scattered += transmittance * step_lum;
            transmittance *= step_t;
        }
    }
    // Distance-based aerial fade works identically whether looking up, across, or down from above.
    let nearest_t = select(seg1.x, seg0.x, len0 > 0.0);
    let dist_fade = exp(-max(nearest_t - 20.0, 0.0) * 0.015);
    let final_t = mix(1.0, transmittance, dist_fade);
    let final_s = scattered * dist_fade;
    textureStore(out_tex, vec2i(id.xy), vec4f(max(final_s, vec3f(0.0)), clamp(final_t, 0.0, 1.0)));
}
