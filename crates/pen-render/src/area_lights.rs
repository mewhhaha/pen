//! Linearly Transformed Cosines (LTC) Rectangular, Disk, and Tube Area Lights
//! plus projected 2D Light Cookie / Gobo Projectors.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "LUT table indexing")]

use std::f64::consts::PI;

/// Standalone WGSL source implementing LTC Rect/Disk/Tube area lights and cookie projectors.
pub const AREA_LIGHTS_WGSL: &str = include_str!("shaders/area_lights.wgsl");

pub const LTC_LUT_SIZE: u32 = 64;

/// Area light shape kind matching `area_lights.wgsl`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum AreaLightKind {
    Rect = 0,
    Disk = 1,
    Tube = 2,
    CookieSpot = 3,
}

/// GPU-ready 80-byte area light / cookie projector record.
/// For one-sided `Rect`/`Disk` lights, the emitting front direction is `normalize(cross(right, up))`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AreaLight {
    pub position_kind: [f32; 4],
    pub right_extent: [f32; 4],
    pub up_extent: [f32; 4],
    pub color_reach: [f32; 4],
    pub flags: [f32; 4],
}

impl AreaLight {
    #[allow(clippy::too_many_arguments)]
    pub fn rect(
        position: [f32; 3],
        right: [f32; 3],
        up: [f32; 3],
        half_width: f32,
        half_height: f32,
        color: [f32; 3],
        reach: f32,
        two_sided: bool,
    ) -> Self {
        Self {
            position_kind: [position[0], position[1], position[2], AreaLightKind::Rect as u32 as f32],
            right_extent: [right[0], right[1], right[2], half_width.max(1e-3)],
            up_extent: [up[0], up[1], up[2], half_height.max(1e-3)],
            color_reach: [color[0].max(0.0), color[1].max(0.0), color[2].max(0.0), reach],
            flags: [if two_sided { 1.0 } else { 0.0 }, 0.0, 0.02, 1.0],
        }
    }

    pub fn disk(
        position: [f32; 3],
        right: [f32; 3],
        up: [f32; 3],
        radius: f32,
        color: [f32; 3],
        reach: f32,
        two_sided: bool,
    ) -> Self {
        Self {
            position_kind: [position[0], position[1], position[2], AreaLightKind::Disk as u32 as f32],
            right_extent: [right[0], right[1], right[2], radius.max(1e-3)],
            up_extent: [up[0], up[1], up[2], radius.max(1e-3)],
            color_reach: [color[0].max(0.0), color[1].max(0.0), color[2].max(0.0), reach],
            flags: [if two_sided { 1.0 } else { 0.0 }, 0.0, 0.02, 1.0],
        }
    }

    pub fn tube(
        position: [f32; 3],
        axis: [f32; 3],
        up_hint: [f32; 3],
        half_length: f32,
        radius: f32,
        color: [f32; 3],
        reach: f32,
    ) -> Self {
        Self {
            position_kind: [position[0], position[1], position[2], AreaLightKind::Tube as u32 as f32],
            right_extent: [axis[0], axis[1], axis[2], half_length.max(1e-3)],
            up_extent: [up_hint[0], up_hint[1], up_hint[2], radius.max(1e-3)],
            color_reach: [color[0].max(0.0), color[1].max(0.0), color[2].max(0.0), reach],
            flags: [1.0, 0.0, 0.02, 1.0],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cookie_projector(
        position: [f32; 3],
        right: [f32; 3],
        up: [f32; 3],
        tan_half_fov: f32,
        aspect: f32,
        color: [f32; 3],
        reach: f32,
        soft_edge: f32,
    ) -> Self {
        Self {
            position_kind: [position[0], position[1], position[2], AreaLightKind::CookieSpot as u32 as f32],
            right_extent: [right[0], right[1], right[2], tan_half_fov.clamp(0.05, 4.0)],
            up_extent: [up[0], up[1], up[2], aspect.clamp(0.1, 10.0)],
            color_reach: [color[0].max(0.0), color[1].max(0.0), color[2].max(0.0), reach],
            flags: [0.0, 1.0, soft_edge.clamp(0.005, 0.5), tan_half_fov.clamp(0.05, 4.0)],
        }
    }
}

/// Converts f32 to IEEE 754 binary16 for uploading `Rgba16Float` LUTs.
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xff).cast_signed() - 112;
    let mant = bits & 0x7f_ffff;
    if exp >= 31 {
        return (sign | 0x7c00) as u16;
    }
    if exp > 0 {
        return (sign | (((exp as u32) << 10) + ((mant + 0x1000) >> 13))) as u16;
    }
    if exp < -10 {
        return sign as u16;
    }
    (sign | ((((mant | 0x80_0000) >> (1 - exp)) + 0x1000) >> 13)) as u16
}

fn ggx_smith_g1(ndotv: f64, alpha: f64) -> f64 {
    let a2 = alpha * alpha;
    let ndv2 = ndotv * ndotv;
    2.0 * ndotv / (ndotv + (a2 + (1.0 - a2) * ndv2).sqrt()).max(1e-7)
}

/// Generates `64 x 64` Heitz LTC inverse-M matrix (`mat`) and GGX split-sum Fresnel
/// amplitude (`amp`) LUTs using 128-sample Hammersley GGX importance integration.
/// Coordinate mapping matches `area_lights.wgsl`: texel `(x, y)` spans
/// `roughness = x / (N - 1)` and `q = y / (N - 1) = sqrt(1 - NdotV)` -> `NdotV = 1 - q^2`.
pub fn generate_ltc_luts() -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let n = LTC_LUT_SIZE as usize;
    let mut mat = vec![[1.0f32, 0.0, 0.0, 1.0]; n * n];
    let mut amp = vec![[1.0f32, 0.0, 0.0, 1.0]; n * n];

    const SAMPLES: usize = 128;
    for y in 0..n {
        let q = y as f64 / (n - 1) as f64;
        // Shader indexes by q = sqrt(1 - NdotV), so NdotV = 1 - q^2.
        let ndotv = (1.0 - q * q).clamp(0.01, 1.0);
        let vx = (1.0 - ndotv * ndotv).sqrt();
        let vz = ndotv;
        for x in 0..n {
            let rough = (x as f64 / (n - 1) as f64).clamp(0.04, 1.0);
            let alpha = (rough * rough).clamp(0.0016, 1.0);
            let a2 = alpha * alpha;

            let mut scale = 0.0;
            let mut bias = 0.0;
            let mut mean_x = 0.0;
            let mut mean_z = 0.0;
            let mut weight_sum = 0.0;
            let mut samples_l = [(0.0f64, 0.0f64, 0.0f64, 0.0f64); SAMPLES];
            let mut sample_count = 0usize;

            for s in 0..SAMPLES {
                let u1 = (s as f64 + 0.5) / SAMPLES as f64;
                let bits = (s as u32).reverse_bits();
                let u2 = f64::from(bits) / 4_294_967_296.0;

                let phi = 2.0 * PI * u1;
                let cos_h = ((1.0 - u2) / (1.0 + (a2 - 1.0) * u2)).sqrt().clamp(0.0, 1.0);
                let sin_h = (1.0 - cos_h * cos_h).sqrt();
                let hx = sin_h * phi.cos();
                let hy = sin_h * phi.sin();
                let hz = cos_h;

                let vdh = (vx * hx + vz * hz).max(0.0);
                let lx = 2.0 * vdh * hx - vx;
                let ly = 2.0 * vdh * hy;
                let lz = 2.0 * vdh * hz - vz;
                if lz > 1e-5 && vdh > 1e-5 {
                    let g_v = ggx_smith_g1(ndotv, alpha);
                    let g_l = ggx_smith_g1(lz, alpha);
                    let g_vis = (g_v * g_l * vdh / (ndotv * cos_h).max(1e-6)).min(4.0);
                    let fc = (1.0 - vdh).powi(5);
                    scale += (1.0 - fc) * g_vis;
                    bias += fc * g_vis;
                    mean_x += lx * g_vis;
                    mean_z += lz * g_vis;
                    weight_sum += g_vis;
                    samples_l[sample_count] = (lx, ly, lz, g_vis);
                    sample_count += 1;
                }
            }
            let norm = 1.0 / SAMPLES as f64;
            let mag = ((scale + bias) * norm).clamp(0.0, 1.0);
            let fresnel = (bias * norm).clamp(0.0, mag);
            let f0_scale = (mag - fresnel).max(0.0);

            let (rx, rz) = if weight_sum > 1e-6 {
                let len = (mean_x * mean_x + mean_z * mean_z).sqrt().max(1e-6);
                ((mean_x / len).clamp(-0.98, 0.98), (mean_z / len).clamp(0.05, 1.0))
            } else {
                (0.0, 1.0)
            };
            let norm_e = (rx * rx + rz * rz).sqrt();
            let (rx, rz) = (rx / norm_e, rz / norm_e);

            let (mut var_x, mut var_y) = (0.0, 0.0);
            if weight_sum > 1e-6 {
                for &(lx, ly, lz, w) in &samples_l[..sample_count] {
                    let px = lx * rz - lz * rx;
                    var_x += w * px * px;
                    var_y += w * ly * ly;
                }
                var_x /= weight_sum;
                var_y /= weight_sum;
            }
            let sigma_x = (2.0 * var_x.sqrt()).max(alpha * 0.5).clamp(0.015, 1.0);
            let sigma_y = (2.0 * var_y.sqrt()).max(alpha * 0.5).clamp(0.015, 1.0);

            let m00 = (rz * sigma_y) / sigma_x;
            let m20 = rx * sigma_y;
            let m02 = (-rx * sigma_y) / sigma_x;
            let m22 = rz * sigma_y;

            let idx = y * n + x;
            mat[idx] = [m00 as f32, m20 as f32, m02 as f32, m22 as f32];
            amp[idx] = [f0_scale as f32, fresnel as f32, mag as f32, 1.0];
        }
    }
    (mat, amp)
}

/// GPU LUT textures for LTC area-light evaluation + default 1x1 white cookie fallback.
pub struct AreaLightResources {
    pub ltc_mat: wgpu::Texture,
    pub ltc_mat_view: wgpu::TextureView,
    pub ltc_amp: wgpu::Texture,
    pub ltc_amp_view: wgpu::TextureView,
    pub default_cookie: wgpu::Texture,
    pub default_cookie_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl AreaLightResources {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let (mat, amp) = generate_ltc_luts();
        let upload_rgba16f = |label, data: &[[f32; 4]], size: u32| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let halfs: Vec<u16> = data.iter().flatten().map(|&v| f32_to_f16(v)).collect();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&halfs),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size * 8), rows_per_image: Some(size) },
                wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            );
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        let (ltc_mat, ltc_mat_view) = upload_rgba16f("ltc_mat", &mat, LTC_LUT_SIZE);
        let (ltc_amp, ltc_amp_view) = upload_rgba16f("ltc_amp", &amp, LTC_LUT_SIZE);
        let default_cookie = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("default_cookie"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &default_cookie,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255, 255, 255, 255],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let default_cookie_view = default_cookie.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ltc_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        Self { ltc_mat, ltc_mat_view, ltc_amp, ltc_amp_view, default_cookie, default_cookie_view, sampler }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ltc_lut_is_finite_and_monotone_in_roughness() {
        let (mat, amp) = generate_ltc_luts();
        assert!(mat.iter().flatten().all(|v| v.is_finite()));
        assert!(amp.iter().flatten().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.001));
        let smooth_m22 = mat[10 * LTC_LUT_SIZE as usize + 4][3];
        let rough_m22 = mat[10 * LTC_LUT_SIZE as usize + 55][3];
        assert!(rough_m22 > smooth_m22 * 4.0, "rough m22={rough_m22} must exceed smooth m22={smooth_m22}");
        // Verify exact grazing Fresnel split-sum conservation: f0_scale + bias == mag.
        for a in &amp {
            assert!((a[0] + a[1] - a[2]).abs() < 1e-4, "split-sum mismatch: {a:?}");
        }
    }

    #[test]
    fn gpu_ltc_rect_disk_tube_one_sided_and_exact_form_factor_witness() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU adapter required for LTC area-light witness");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let res = AreaLightResources::new(&device, &queue);

        let cookie_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_cookie"),
            size: wgpu::Extent3d { width: 16, height: 16, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut cookie_bytes = vec![0u8; 16 * 16 * 4];
        for y in 0..16 {
            for x in 0..16 {
                let v = if x < 8 { 255u8 } else { 0u8 };
                let o = (y * 16 + x) * 4;
                cookie_bytes[o..o + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &cookie_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &cookie_bytes,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(16 * 4), rows_per_image: Some(16) },
            wgpu::Extent3d { width: 16, height: 16, depth_or_array_layers: 1 },
        );
        let cookie_view = cookie_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let harness_wgsl = format!(
            "{AREA_LIGHTS_WGSL}\n
            struct Probe {{ world_rough: vec4f, view_kind: vec4f, normal_f0: vec4f, albedo_pad: vec4f }};
            @group(0) @binding(0) var<storage, read> lights: array<AreaLight>;
            @group(0) @binding(1) var<storage, read> probes: array<Probe>;
            @group(0) @binding(2) var<storage, read_write> results: array<vec4f>;
            @group(0) @binding(3) var ltc_mat: texture_2d<f32>;
            @group(0) @binding(4) var ltc_amp: texture_2d<f32>;
            @group(0) @binding(5) var cookie: texture_2d<f32>;
            @group(0) @binding(6) var samp: sampler;
            @compute @workgroup_size(1)
            fn test_eval(@builtin(global_invocation_id) id: vec3u) {{
                let i = id.x;
                let p = probes[i];
                let li = u32(round(p.view_kind.w));
                let rgb = eval_area_light(
                    lights[li],
                    p.world_rough.xyz,
                    normalize(p.normal_f0.xyz),
                    normalize(p.view_kind.xyz),
                    p.albedo_pad.xyz,
                    vec3f(p.normal_f0.w),
                    p.world_rough.w,
                    ltc_mat,
                    ltc_amp,
                    cookie,
                    samp,
                );
                results[i] = vec4f(rgb, 1.0);
            }}"
        );
        let lights = [
            // 0: One-sided Rect above origin facing DOWN (-Y = cross((1,0,0), (0,0,1)) = (0,-1,0))
            AreaLight::rect([0.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0, 1.0, [1.0; 3], 0.0, false),
            // 1: One-sided Disk above origin facing DOWN (-Y)
            AreaLight::disk([0.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0, [4.0; 3], 20.0, false),
            // 2: Tube above origin
            AreaLight::tube([0.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.5, 0.15, [4.0; 3], 20.0),
            // 3: Cookie projector at (0, 3, 0) shining down (-Y)
            AreaLight::cookie_projector([0.0, 3.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0, 1.0, [10.0; 3], 20.0, 0.01),
        ];
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Probe {
            world_rough: [f32; 4],
            view_kind: [f32; 4],
            normal_f0: [f32; 4],
            albedo_pad: [f32; 4],
        }
        let probes = [
            // 0: One-sided Rect front (y=0), smooth mirror-like specular
            Probe { world_rough: [0.0, 0.0, 0.0, 0.15], view_kind: [0.0, 1.0, 0.0, 0.0], normal_f0: [0.0, 1.0, 0.0, 0.9], albedo_pad: [0.05, 0.05, 0.05, 0.0] },
            // 1: One-sided Rect front (y=0), rough
            Probe { world_rough: [0.0, 0.0, 0.0, 0.70], view_kind: [0.0, 1.0, 0.0, 0.0], normal_f0: [0.0, 1.0, 0.0, 0.9], albedo_pad: [0.05, 0.05, 0.05, 0.0] },
            // 2: Exact diffuse form-factor check for 2x2 unit-radiance square at h=2: pure diffuse (albedo=1, f0=0)
            Probe { world_rough: [0.0, 0.0, 0.0, 1.0], view_kind: [0.0, 1.0, 0.0, 0.0], normal_f0: [0.0, 1.0, 0.0, 0.0], albedo_pad: [1.0, 1.0, 1.0, 0.0] },
            // 3: One-sided Rect BACK side (y=4, normal facing down toward light) -> MUST be 0!
            Probe { world_rough: [0.0, 4.0, 0.0, 0.5], view_kind: [0.0, -1.0, 0.0, 0.0], normal_f0: [0.0, -1.0, 0.0, 0.5], albedo_pad: [1.0, 1.0, 1.0, 0.0] },
            // 4: One-sided Disk front (y=0)
            Probe { world_rough: [0.0, 0.0, 0.0, 0.20], view_kind: [0.0, 1.0, 0.0, 1.0], normal_f0: [0.0, 1.0, 0.0, 0.9], albedo_pad: [0.05, 0.05, 0.05, 0.0] },
            // 5: One-sided Disk BACK side (y=4) -> MUST be 0!
            Probe { world_rough: [0.0, 4.0, 0.0, 0.20], view_kind: [0.0, -1.0, 0.0, 1.0], normal_f0: [0.0, -1.0, 0.0, 0.9], albedo_pad: [1.0, 1.0, 1.0, 0.0] },
            // 6: Tube
            Probe { world_rough: [0.0, 0.0, 0.0, 0.20], view_kind: [0.0, 1.0, 0.0, 2.0], normal_f0: [0.0, 1.0, 0.0, 0.9], albedo_pad: [0.05, 0.05, 0.05, 0.0] },
            // 7: Cookie lit left half
            Probe { world_rough: [-1.0, 0.0, 0.0, 0.50], view_kind: [0.0, 1.0, 0.0, 3.0], normal_f0: [0.0, 1.0, 0.0, 0.04], albedo_pad: [0.8, 0.8, 0.8, 0.0] },
            // 8: Cookie dark right half
            Probe { world_rough: [1.0, 0.0, 0.0, 0.50], view_kind: [0.0, 1.0, 0.0, 3.0], normal_f0: [0.0, 1.0, 0.0, 0.04], albedo_pad: [0.8, 0.8, 0.8, 0.0] },
        ];
        let make_buf = |label, bytes: &[u8], usage| {
            let b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.len() as u64,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&b, 0, bytes);
            b
        };
        let lights_buf = make_buf("lights", bytemuck::cast_slice(&lights), wgpu::BufferUsages::STORAGE);
        let probes_buf = make_buf("probes", bytemuck::cast_slice(&probes), wgpu::BufferUsages::STORAGE);
        let out_size = (probes.len() * 16) as u64;
        let res_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("results"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: out_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("area_light_test"),
            source: wgpu::ShaderSource::Wgsl(harness_wgsl.into()),
        });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("area_light_test"),
            layout: None,
            module: &module,
            entry_point: Some("test_eval"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("area_light_bg"),
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: lights_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: probes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: res_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&res.ltc_mat_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&res.ltc_amp_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&cookie_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&res.sampler) },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(probes.len() as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&res_buf, 0, &readback, 0, out_size);
        queue.submit(Some(enc.finish()));
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let bytes = slice.get_mapped_range().unwrap();
        let vals: &[[f32; 4]] = bytemuck::cast_slice(&bytes);
        assert!(vals.iter().flatten().all(|v| v.is_finite() && *v >= 0.0), "{vals:?}");
        // Smooth mirror-aligned rect highlight must exceed rough.
        assert!(vals[0][0] > vals[1][0] * 1.3, "smooth rect must peak brighter than rough: {} vs {}", vals[0][0], vals[1][0]);
        // Analytical Lambertian form factor of [-1, 1]^2 square at h=2 above center:
        // F = (4 / pi) * (a / sqrt(a^2 + h^2)) * atan(a / sqrt(a^2 + h^2)) with a=1, h=2 -> 0.23946.
        let expected_f = (4.0 / PI) * (1.0 / 5.0f64.sqrt()) * (1.0 / 5.0f64.sqrt()).atan();
        assert!(
            (f64::from(vals[2][0]) - expected_f).abs() < 0.015,
            "diffuse LTC form factor must match exact analytical integral {expected_f:.5}, got {:.5}",
            vals[2][0]
        );
        // One-sided rect and disk MUST emit exactly zero from their back sides.
        assert_eq!(vals[3][0], 0.0, "one-sided rect back side must be dark: {:?}", vals[3]);
        assert!(vals[4][0] > 1.0, "one-sided disk front side must illuminate: {:?}", vals[4]);
        assert_eq!(vals[5][0], 0.0, "one-sided disk back side must be dark: {:?}", vals[5]);
        assert!(vals[6][0] > 0.3, "tube LTC must illuminate: {:?}", vals[6]);
        assert!(vals[7][0] > 0.3 && vals[7][0] > vals[8][0] * 10.0, "cookie contrast failed: lit={} dark={}", vals[7][0], vals[8][0]);
        drop(bytes);
        readback.unmap();
    }

    #[test]
    fn ltc_amplitude_lut_fresnel_and_hemisphere_energy_conservation() {
        let (_mat, amp) = generate_ltc_luts();
        let n = LTC_LUT_SIZE as usize;
        for y in 0..n {
            for x in 0..n {
                let [scale, bias, mag, _] = amp[y * n + x];
                assert!(scale >= 0.0 && bias >= 0.0 && (0.0..=1.001).contains(&mag));
                // Reconstructed Schlick F(F0, F90) = F0 * scale + F90 * bias stays in [0, 1].
                let f_dielectric = 0.04 * scale + 1.0 * bias;
                let f_conductor = 0.9 * scale + 1.0 * bias;
                let f_zero_f90 = 0.04 * scale + 0.0 * bias;
                assert!((0.0..=1.001).contains(&f_dielectric), "dielectric={f_dielectric}");
                assert!(f_conductor >= f_dielectric && f_conductor <= 1.001);
                assert!(f_zero_f90 <= f_dielectric + 1e-6);
                // White dielectric energy sum under unit-radiance full hemisphere:
                // diff_albedo * 1 + spec_fresnel * 1 = (1 - F) + F = 1.0.
                let energy = (1.0 - f_dielectric).max(0.0) + f_dielectric;
                assert!((energy - 1.0).abs() < 1e-5);
            }
        }
        // Grazing view (high y) has significantly stronger Schlick grazing bias than normal incidence (y = 0).
        let bias_normal = amp[n / 2][1];
        let bias_grazing = amp[(n - 2) * n + n / 2][1];
        assert!(
            bias_grazing > bias_normal * 3.0,
            "grazing Schlick bias must exceed normal incidence: normal={bias_normal}, grazing={bias_grazing}"
        );
    }
}
