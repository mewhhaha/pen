// Physical Rayleigh + Mie + Ozone multi-scattering atmosphere LUT compute
// shaders following Sebastian Hillaire (EGSR 2020), "A Scalable and Production
// Ready Sky and Atmosphere Rendering Technique".
// All lengths and altitudes are in kilometers; outputs are linear radiance /
// dimensionless transmittance stored in Rgba16Float textures.

const PI: f32 = 3.14159265359;

struct AtmoUniform {
    // x: planet_radius_km (6360), y: atmo_radius_km (6460), z: camera_alt_km, w: rayleigh_scale_km (8.0)
    radii: vec4f,
    // xyz: rayleigh scattering (km^-1), w: mie_scale_km (1.2)
    rayleigh: vec4f,
    // xyz: mie scattering (km^-1), w: mie_extinction (km^-1)
    mie: vec4f,
    // xyz: ozone absorption (km^-1), w: mie_g (0.8)
    ozone: vec4f,
    // xyz: unit direction toward sun, w: sun_angular_cos (cos 0.267 deg)
    sun_dir: vec4f,
    // rgb: solar irradiance, w: ozone_center_km (25.0)
    sun_color: vec4f,
    // rgb: ground albedo, w: ozone_half_width_km (15.0)
    ground: vec4f,
    // xy: target dims, zw: transmittance dims
    dims: vec4f,
};

@group(0) @binding(0) var<uniform> atmo: AtmoUniform;
@group(0) @binding(1) var trans_lut: texture_2d<f32>;
@group(0) @binding(2) var ms_lut: texture_2d<f32>;
@group(0) @binding(3) var lut_sampler: sampler;
@group(0) @binding(4) var out_tex: texture_storage_2d<rgba16float, write>;

// Ray-sphere intersection from origin at radius `r` along direction with zenith cosine `mu`
// against a sphere of radius `target_r`. Returns nearest positive distance or -1.
fn ray_sphere_dist(r: f32, mu: f32, target_r: f32) -> f32 {
    let disc = r * r * (mu * mu - 1.0) + target_r * target_r;
    if (disc < 0.0) {
        return -1.0;
    }
    let s = sqrt(disc);
    let t0 = -r * mu - s;
    let t1 = -r * mu + s;
    if (t0 >= 0.0) {
        return t0;
    }
    if (t1 >= 0.0) {
        return t1;
    }
    return -1.0;
}

fn ray_sphere_3d(origin: vec3f, dir: vec3f, radius: f32) -> vec2f {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let d = b * b - c;
    if (d < 0.0) {
        return vec2f(-1.0, -1.0);
    }
    let s = sqrt(d);
    return vec2f(-b - s, -b + s);
}

struct Medium {
    scatter_r: vec3f,
    scatter_m: vec3f,
    extinction: vec3f,
};

fn sample_medium(altitude_km: f32) -> Medium {
    let h = max(altitude_km, 0.0);
    let d_r = exp(-h / max(atmo.radii.w, 0.1));
    let d_m = exp(-h / max(atmo.rayleigh.w, 0.05));
    let oz_center = max(atmo.sun_color.w, 1.0);
    let oz_width = max(atmo.ground.w, 1.0);
    let d_o = max(0.0, 1.0 - abs(h - oz_center) / oz_width);

    let scatter_r = atmo.rayleigh.xyz * d_r;
    let scatter_m = atmo.mie.xyz * d_m;
    let ext_m = vec3f(atmo.mie.w) * d_m;
    let abs_o = atmo.ozone.xyz * d_o;
    return Medium(scatter_r, scatter_m, scatter_r + ext_m + abs_o);
}

// Numerically stable (1 - exp(-ext * dt)) / ext using Taylor series for optically thin cells.
fn segment_integral(ext: vec3f, dt: f32) -> vec3f {
    let x = max(ext * dt, vec3f(0.0));
    let taylor = vec3f(dt) * (vec3f(1.0) - x * 0.5 + x * x * (1.0 / 6.0));
    let exact = (vec3f(1.0) - exp(-x)) / max(ext, vec3f(1e-7));
    return select(exact, taylor, x < vec3f(1e-3));
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let num = 1.0 - g2;
    let denom = 4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4), 1.5);
    return num / denom;
}

// Hillaire/Bruneton transmittance LUT UV parameterization (maps physical endpoints to [0, 1]).
fn transmittance_uv(r: f32, mu: f32) -> vec2f {
    let rb = atmo.radii.x;
    let rt = atmo.radii.y;
    let h_max = sqrt(max(rt * rt - rb * rb, 1e-4));
    let rho = sqrt(max(r * r - rb * rb, 0.0));
    let disc = r * r * (mu * mu - 1.0) + rt * rt;
    let d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
    let d_min = rt - r;
    let d_max = rho + h_max;
    let x_mu = clamp((d - d_min) / max(d_max - d_min, 1e-5), 0.0, 1.0);
    let x_r = clamp(rho / h_max, 0.0, 1.0);
    return vec2f(x_mu, x_r);
}

fn sample_transmittance(r: f32, mu: f32) -> vec3f {
    let uv = transmittance_uv(r, mu);
    let dims = max(atmo.dims.zw, vec2f(2.0));
    let safe_uv = clamp(uv * ((dims - 1.0) / dims) + 0.5 / dims, vec2f(0.0), vec2f(1.0));
    return textureSampleLevel(trans_lut, lut_sampler, safe_uv, 0.0).rgb;
}

fn sample_multiscatter(alt_km: f32, sun_mu: f32) -> vec3f {
    let u = clamp(sun_mu * 0.5 + 0.5, 0.0, 1.0);
    let v = clamp(alt_km / max(atmo.radii.y - atmo.radii.x, 1.0), 0.0, 1.0);
    let safe_uv = vec2f(u, v) * (31.0 / 32.0) + 0.5 / 32.0;
    return textureSampleLevel(ms_lut, lut_sampler, safe_uv, 0.0).rgb;
}

// Camera-height-aware non-linear horizon latitude mapping (Hillaire EGSR 2020 §5.3).
// Concentrates sky-view LUT texels tightly around the true curved planet limb.
fn sky_view_v_to_lat(v: f32, r: f32, rb: f32) -> f32 {
    let horizon_angle = -acos(clamp(rb / max(r, rb), 0.0, 1.0));
    if (v < 0.5) {
        let k = 1.0 - 2.0 * v;
        return horizon_angle - k * k * (horizon_angle + 0.5 * PI);
    }
    let k = 2.0 * v - 1.0;
    return horizon_angle + k * k * (0.5 * PI - horizon_angle);
}

// Inverse mapping for consumers sampling `sky_view` at output resolution.
fn sky_view_dir_to_uv(view_dir: vec3f, sun_dir: vec3f, r: f32, rb: f32) -> vec2f {
    let d = normalize(view_dir);
    let horizon_angle = -acos(clamp(rb / max(r, rb), 0.0, 1.0));
    let lat = asin(clamp(d.y, -1.0, 1.0));
    var v = 0.5;
    if (lat < horizon_angle) {
        let span = max(horizon_angle + 0.5 * PI, 1e-4);
        let k = sqrt(clamp((horizon_angle - lat) / span, 0.0, 1.0));
        v = 0.5 * (1.0 - k);
    } else {
        let span = max(0.5 * PI - horizon_angle, 1e-4);
        let k = sqrt(clamp((lat - horizon_angle) / span, 0.0, 1.0));
        v = 0.5 * (1.0 + k);
    }
    let sun_xz = vec2f(sun_dir.x, sun_dir.z);
    let sun_xz_len = length(sun_xz);
    let fwd = select(vec2f(0.0, -1.0), sun_xz / sun_xz_len, sun_xz_len > 1e-5);
    let right = vec2f(-fwd.y, fwd.x);
    let az = atan2(dot(d.xz, right), dot(d.xz, fwd));
    let u = fract(az / (2.0 * PI) + 0.5);
    return vec2f(u, clamp(v, 0.0, 1.0));
}

// Output-resolution planet-occluded solar disk contribution attenuated by transmittance.
fn eval_solar_disk(view_dir: vec3f, sun_dir: vec3f, r: f32, rb: f32) -> vec3f {
    let d = normalize(view_dir);
    let s = normalize(sun_dir);
    let cos_theta = dot(d, s);
    if (cos_theta < atmo.sun_dir.w) {
        return vec3f(0.0);
    }
    if (ray_sphere_dist(r, d.y, rb) > 0.0) {
        return vec3f(0.0);
    }
    return sample_transmittance(r, d.y) * atmo.sun_color.rgb * 12.0;
}

@compute @workgroup_size(8, 8, 1)
fn transmittance_lut(@builtin(global_invocation_id) id: vec3u) {
    let dims = vec2u(atmo.dims.xy);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    // Map texel centers i in [0, N-1] directly onto [0, 1] endpoints to match `sample_transmittance`.
    let uv = vec2f(id.xy) / max(vec2f(dims) - vec2f(1.0), vec2f(1.0));
    let rb = atmo.radii.x;
    let rt = atmo.radii.y;
    let h_max = sqrt(max(rt * rt - rb * rb, 1e-4));
    let rho = h_max * uv.y;
    let r = sqrt(rho * rho + rb * rb);
    let d_min = rt - r;
    let d_max = rho + h_max;
    let d = d_min + uv.x * (d_max - d_min);
    var mu = 1.0;
    if (d > 1e-5) {
        mu = clamp((h_max * h_max - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
    }

    let steps = 40u;
    let dt = d / f32(steps);
    var od = vec3f(0.0);
    for (var i = 0u; i < steps; i++) {
        let t = (f32(i) + 0.5) * dt;
        let ri = sqrt(max(r * r + t * t + 2.0 * r * mu * t, rb * rb));
        let m = sample_medium(ri - rb);
        od += m.extinction * dt;
    }
    let trans = exp(-od);
    textureStore(out_tex, vec2i(id.xy), vec4f(trans, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn multiscatter_lut(@builtin(global_invocation_id) id: vec3u) {
    let dims = vec2u(atmo.dims.xy);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    let uv = vec2f(id.xy) / max(vec2f(dims) - vec2f(1.0), vec2f(1.0));
    let rb = atmo.radii.x;
    let rt = atmo.radii.y;
    let sun_mu = uv.x * 2.0 - 1.0;
    let alt = uv.y * (rt - rb);
    let r = rb + clamp(alt, 0.002, rt - rb - 0.002);
    let origin = vec3f(0.0, r, 0.0);
    let sun_dir = normalize(vec3f(sqrt(max(0.0, 1.0 - sun_mu * sun_mu)), sun_mu, 0.0));

    let dir_count = 64u;
    let golden = 2.39996323;
    let inv_sphere = 1.0 / f32(dir_count);
    let iso_phase = 1.0 / (4.0 * PI);
    var l_2nd = vec3f(0.0);
    var f_ms = vec3f(0.0);

    for (var d_idx = 0u; d_idx < dir_count; d_idx++) {
        let y = 1.0 - (f32(d_idx) + 0.5) * 2.0 / f32(dir_count);
        let rad = sqrt(max(0.0, 1.0 - y * y));
        let phi = golden * f32(d_idx);
        let dir = vec3f(cos(phi) * rad, y, sin(phi) * rad);

        let hit_atmo = ray_sphere_3d(origin, dir, rt);
        let hit_ground = ray_sphere_3d(origin, dir, rb);
        var t_max = max(hit_atmo.y, 0.0);
        var hits_ground = false;
        if (hit_ground.x > 0.0) {
            t_max = min(t_max, hit_ground.x);
            hits_ground = true;
        }
        let steps = 20u;
        let dt = t_max / f32(steps);
        var throughput = vec3f(1.0);
        var lum = vec3f(0.0);
        var f_ray = vec3f(0.0);
        for (var s = 0u; s < steps; s++) {
            let t = (f32(s) + 0.5) * dt;
            let p = origin + dir * t;
            let pi_r = length(p);
            let m = sample_medium(pi_r - rb);
            let step_t = exp(-m.extinction * dt);
            let scatter = m.scatter_r + m.scatter_m;
            let integ = segment_integral(m.extinction, dt);

            let up = p / max(pi_r, 1e-4);
            let local_sun_mu = dot(up, sun_dir);
            let sun_hit_ground = ray_sphere_dist(pi_r, local_sun_mu, rb);
            let sun_vis = select(1.0, 0.0, sun_hit_ground > 0.0);
            let t_sun = sample_transmittance(pi_r, local_sun_mu) * sun_vis;

            lum += throughput * scatter * iso_phase * t_sun * integ;
            f_ray += throughput * scatter * integ;
            throughput *= step_t;
        }
        if (hits_ground) {
            let p_g = origin + dir * t_max;
            let n_g = normalize(p_g);
            let n_dot_l = max(dot(n_g, sun_dir), 0.0);
            let t_sun = sample_transmittance(rb + 0.001, n_dot_l);
            lum += throughput * (atmo.ground.rgb / PI) * n_dot_l * t_sun;
        }
        l_2nd += lum * inv_sphere;
        f_ms += f_ray * inv_sphere;
    }
    let psi_ms = l_2nd / max(vec3f(1.0) - f_ms, vec3f(0.05));
    textureStore(out_tex, vec2i(id.xy), vec4f(max(psi_ms, vec3f(0.0)), 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn sky_view_lut(@builtin(global_invocation_id) id: vec3u) {
    let dims = vec2u(atmo.dims.xy);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    let uv = vec2f(id.xy) / max(vec2f(dims) - vec2f(1.0), vec2f(1.0));
    let rb = atmo.radii.x;
    let rt = atmo.radii.y;
    let cam_alt = clamp(atmo.radii.z, 0.002, rt - rb - 0.1);
    let r = rb + cam_alt;
    let origin = vec3f(0.0, r, 0.0);

    let lat = sky_view_v_to_lat(uv.y, r, rb);
    let azimuth = (uv.x - 0.5) * 2.0 * PI;
    let cos_lat = cos(lat);
    let dir = normalize(vec3f(cos_lat * sin(azimuth), sin(lat), -cos_lat * cos(azimuth)));

    // Align sun into the XZ azimuth=0 meridian plane; consumers match via `sky_view_dir_to_uv`.
    let sun_y = clamp(atmo.sun_dir.y, -1.0, 1.0);
    let sun_horiz = sqrt(max(0.0, 1.0 - sun_y * sun_y));
    let sun_dir = normalize(vec3f(0.0, sun_y, -sun_horiz));

    let hit_atmo = ray_sphere_3d(origin, dir, rt);
    let hit_ground = ray_sphere_3d(origin, dir, rb);
    var t_max = max(hit_atmo.y, 0.0);
    var hits_ground = false;
    if (hit_ground.x > 0.0) {
        t_max = min(t_max, hit_ground.x);
        hits_ground = true;
    }

    let cos_theta = dot(dir, sun_dir);
    let p_r = rayleigh_phase(cos_theta);
    let p_m = mie_phase(cos_theta, clamp(atmo.ozone.w, -0.95, 0.95));

    let steps = 32u;
    let dt = t_max / f32(steps);
    var throughput = vec3f(1.0);
    var radiance = vec3f(0.0);

    for (var s = 0u; s < steps; s++) {
        let t = (f32(s) + 0.5) * dt;
        let p = origin + dir * t;
        let pi_r = length(p);
        let alt = max(pi_r - rb, 0.0);
        let m = sample_medium(alt);
        let step_t = exp(-m.extinction * dt);
        let integ = segment_integral(m.extinction, dt);

        let up = p / max(pi_r, 1e-4);
        let sun_mu = dot(up, sun_dir);
        let sun_blocked = ray_sphere_dist(pi_r, sun_mu, rb) > 0.0;
        let t_sun = select(sample_transmittance(pi_r, sun_mu), vec3f(0.0), sun_blocked);
        let psi_ms = sample_multiscatter(alt, sun_mu);

        let single = (m.scatter_r * p_r + m.scatter_m * p_m) * t_sun;
        let multi = (m.scatter_r + m.scatter_m) * psi_ms;
        radiance += throughput * (single + multi) * integ * atmo.sun_color.rgb;
        throughput *= step_t;
    }

    if (hits_ground) {
        let p_g = origin + dir * t_max;
        let n_g = normalize(p_g);
        // Preserve signed sun zenith cosine for night-side twilight multi-scattering lookup.
        let raw_sun_mu = dot(n_g, sun_dir);
        let n_dot_l = max(raw_sun_mu, 0.0);
        let t_sun = select(vec3f(0.0), sample_transmittance(rb + 0.001, n_dot_l), raw_sun_mu > 0.0);
        let psi_ms = sample_multiscatter(0.0, raw_sun_mu);
        radiance += throughput * (atmo.ground.rgb / PI) * (t_sun * n_dot_l + psi_ms) * atmo.sun_color.rgb;
    }

    // Smooth atmospheric sky stored here; composite sharp solar disk at output resolution via `eval_solar_disk`.
    textureStore(out_tex, vec2i(id.xy), vec4f(max(radiance, vec3f(0.0)), 1.0));
}
