//! Physical Rayleigh + Mie + Ozone multi-scattering atmosphere LUTs (Hillaire 2020)
//! and raymarched 3D volumetric clouds with dual-lobe Henyey-Greenstein silver lining.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "GPU texture dimensions")]

use anyhow::{Result, ensure};

/// Standalone WGSL source for the transmittance, multi-scattering, and sky-view LUT compute passes.
pub const ATMOSPHERE_LUT_WGSL: &str = crate::wgsl!("atmosphere_lut.wgsl");
/// Standalone WGSL source for raymarched 3D volumetric clouds.
pub const CLOUDS_WGSL: &str = crate::wgsl!("clouds.wgsl");

pub const TRANSMITTANCE_SIZE: (u32, u32) = (256, 64);
pub const MULTISCATTER_SIZE: (u32, u32) = (32, 32);
pub const SKY_VIEW_SIZE: (u32, u32) = (192, 108);

/// Physical Earth-like atmosphere & volumetric cloud configuration (distances in km).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtmosphereParams {
    pub planet_radius_km: f32,
    pub atmo_radius_km: f32,
    pub camera_altitude_km: f32,
    pub rayleigh_scale_km: f32,
    pub rayleigh_scattering: [f32; 3],
    pub mie_scale_km: f32,
    pub mie_scattering: [f32; 3],
    pub mie_extinction: f32,
    pub mie_g: f32,
    pub ozone_absorption: [f32; 3],
    pub ozone_center_km: f32,
    pub ozone_half_width_km: f32,
    pub sun_direction: [f32; 3],
    pub sun_irradiance: [f32; 3],
    pub ground_albedo: [f32; 3],
    pub cloud_coverage: f32,
    pub cloud_density: f32,
    pub cloud_bottom_km: f32,
    pub cloud_top_km: f32,
    pub cloud_wind_km: [f32; 3],
    pub cloud_detail_freq: f32,
    pub cloud_silver_g: f32,
    pub cloud_ambient: [f32; 3],
    /// Primary volumetric cloud march steps (clamped to 8..=64; default 32).
    pub cloud_steps: u32,
    /// Maximum volumetric cloud raymarch distance in kilometers (clamped to 5..=120; default 45 km).
    pub cloud_max_march_km: f32,
}

impl Default for AtmosphereParams {
    fn default() -> Self {
        // Sebastian Hillaire EGSR 2020 Table 1 Earth atmosphere constants (km^-1).
        Self {
            planet_radius_km: 6360.0,
            atmo_radius_km: 6460.0,
            camera_altitude_km: 0.2,
            rayleigh_scale_km: 8.0,
            rayleigh_scattering: [5.802e-3, 13.558e-3, 33.1e-3],
            mie_scale_km: 1.2,
            mie_scattering: [3.996e-3, 3.996e-3, 3.996e-3],
            mie_extinction: 4.440e-3,
            mie_g: 0.8,
            ozone_absorption: [0.650e-3, 1.881e-3, 0.085e-3],
            ozone_center_km: 25.0,
            ozone_half_width_km: 15.0,
            sun_direction: [0.0, 0.85, -0.5268],
            sun_irradiance: [1.0, 0.98, 0.94],
            ground_albedo: [0.22, 0.24, 0.20],
            cloud_coverage: 0.55,
            cloud_density: 0.65,
            cloud_bottom_km: 1.5,
            cloud_top_km: 4.0,
            cloud_wind_km: [0.0, 0.0, 0.0],
            cloud_detail_freq: 0.35,
            cloud_silver_g: 0.8,
            cloud_ambient: [0.28, 0.36, 0.48],
            cloud_steps: 32,
            cloud_max_march_km: 45.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AtmoUniform {
    radii: [f32; 4],
    rayleigh: [f32; 4],
    mie: [f32; 4],
    ozone: [f32; 4],
    sun_dir: [f32; 4],
    sun_color: [f32; 4],
    ground: [f32; 4],
    dims: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CloudUniform {
    inverse_view_projection: [f32; 16],
    camera: [f32; 4],
    sun_dir: [f32; 4],
    sun_color: [f32; 4],
    ambient: [f32; 4],
    wind: [f32; 4],
    params: [f32; 4],
    quality: [f32; 4],
}

/// World-space scene units (meters) converted to atmosphere kilometers (`1 unit = 1 meter`).
pub const WORLD_METERS_TO_KM: f32 = 0.001;
/// Base observer elevation above sea level (km) when scene camera `y == 0`.
pub const BASE_CAMERA_ALTITUDE_KM: f32 = 0.2;

/// Converts a scene world camera position (meters) to physical atmosphere coordinates (kilometers),
/// anchored at `BASE_CAMERA_ALTITUDE_KM` and clamped above planet ground (`>= 0.002 km`).
pub fn physical_camera_km(world_eye: [f32; 3]) -> [f32; 3] {
    [
        finite_or(world_eye[0], 0.0) * WORLD_METERS_TO_KM,
        (BASE_CAMERA_ALTITUDE_KM + finite_or(world_eye[1], 0.0) * WORLD_METERS_TO_KM).max(0.002),
        finite_or(world_eye[2], 0.0) * WORLD_METERS_TO_KM,
    ]
}

/// Normalizes a sun direction vector consistently across LUTs, clouds, and sky composite.
pub fn normalize_sun_dir(v: [f32; 3]) -> [f32; 3] {
    normalize3(v)
}

fn finite_or(v: f32, fallback: f32) -> f32 {
    if v.is_finite() { v } else { fallback }
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let x = finite_or(v[0], 0.0);
    let y = finite_or(v[1], 1.0);
    let z = finite_or(v[2], 0.0);
    let len = (f64::from(x).powi(2) + f64::from(y).powi(2) + f64::from(z).powi(2)).sqrt();
    if len <= 1e-8 || !len.is_finite() {
        [0.0, 1.0, 0.0]
    } else {
        [(f64::from(x) / len) as f32, (f64::from(y) / len) as f32, (f64::from(z) / len) as f32]
    }
}

/// Encodes a command-ordered uniform upload via a mapped-at-creation staging buffer so multiple
/// dispatches recorded into the same `CommandEncoder` never overwrite each other's inputs.
fn encode_uniform_upload(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, dst: &wgpu::Buffer, bytes: &[u8]) {
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("atmo_stage"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    stage.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytes);
    stage.unmap();
    encoder.copy_buffer_to_buffer(&stage, 0, dst, 0, bytes.len() as u64);
}

/// GPU compute generator for physical Transmittance, Multi-Scattering, and Sky-View LUTs,
/// plus a raymarched 3D volumetric cloud buffer.
pub struct PhysicalAtmosphere {
    transmittance: wgpu::Texture,
    transmittance_view: wgpu::TextureView,
    multiscatter: wgpu::Texture,
    multiscatter_view: wgpu::TextureView,
    sky_view: wgpu::Texture,
    sky_view_view: wgpu::TextureView,
    dummy_tex: wgpu::Texture,
    dummy_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    uniform_trans: wgpu::Buffer,
    uniform_ms: wgpu::Buffer,
    uniform_sky: wgpu::Buffer,
    bgl: wgpu::BindGroupLayout,
    bg_trans: wgpu::BindGroup,
    bg_ms: wgpu::BindGroup,
    bg_sky: wgpu::BindGroup,
    pipe_trans: wgpu::ComputePipeline,
    pipe_ms: wgpu::ComputePipeline,
    pipe_sky: wgpu::ComputePipeline,
    cloud_bgl: wgpu::BindGroupLayout,
    cloud_pipe: wgpu::ComputePipeline,
    cloud_uniform: wgpu::Buffer,
    cloud_tex: Option<(u32, u32, wgpu::Texture, wgpu::TextureView, wgpu::BindGroup)>,
    cloud_serial: u64,
    last_medium_key: Option<[u32; 19]>,
    last_sky_key: Option<[u32; 8]>,
}

impl PhysicalAtmosphere {
    pub fn new(device: &wgpu::Device) -> Self {
        let make_lut = |label, (w, h)| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        let (transmittance, transmittance_view) = make_lut("atmo_transmittance", TRANSMITTANCE_SIZE);
        let (multiscatter, multiscatter_view) = make_lut("atmo_multiscatter", MULTISCATTER_SIZE);
        let (sky_view, sky_view_view) = make_lut("atmo_sky_view", SKY_VIEW_SIZE);
        let (dummy_tex, dummy_view) = make_lut("atmo_dummy", (1, 1));
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atmo_lut_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let make_ubuf = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let uniform_trans = make_ubuf("atmo_u_trans", std::mem::size_of::<AtmoUniform>() as u64);
        let uniform_ms = make_ubuf("atmo_u_ms", std::mem::size_of::<AtmoUniform>() as u64);
        let uniform_sky = make_ubuf("atmo_u_sky", std::mem::size_of::<AtmoUniform>() as u64);

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atmo_lut_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let make_bg = |label,
                       ubuf: &wgpu::Buffer,
                       t_in: &wgpu::TextureView,
                       m_in: &wgpu::TextureView,
                       out: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(t_in) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(m_in) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(out) },
                ],
            })
        };
        let bg_trans = make_bg("atmo_bg_trans", &uniform_trans, &dummy_view, &dummy_view, &transmittance_view);
        let bg_ms = make_bg("atmo_bg_ms", &uniform_ms, &transmittance_view, &dummy_view, &multiscatter_view);
        let bg_sky = make_bg("atmo_bg_sky", &uniform_sky, &transmittance_view, &multiscatter_view, &sky_view_view);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("atmosphere_lut"),
            source: wgpu::ShaderSource::Wgsl(crate::deferred_wgsl!("atmosphere_lut.wgsl").into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("atmo_lut_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let make_pipe = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipe_trans = make_pipe("transmittance_lut");
        let pipe_ms = make_pipe("multiscatter_lut");
        let pipe_sky = make_pipe("sky_view_lut");

        let cloud_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clouds_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let cloud_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clouds"),
            source: wgpu::ShaderSource::Wgsl(crate::deferred_wgsl!("clouds.wgsl").into()),
        });
        let cloud_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clouds_pl"),
            bind_group_layouts: &[Some(&cloud_bgl)],
            immediate_size: 0,
        });
        let cloud_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clouds_main"),
            layout: Some(&cloud_pl),
            module: &cloud_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let cloud_uniform = make_ubuf("clouds_u", std::mem::size_of::<CloudUniform>() as u64);

        Self {
            transmittance,
            transmittance_view,
            multiscatter,
            multiscatter_view,
            sky_view,
            sky_view_view,
            dummy_tex,
            dummy_view,
            sampler,
            uniform_trans,
            uniform_ms,
            uniform_sky,
            bgl,
            bg_trans,
            bg_ms,
            bg_sky,
            pipe_trans,
            pipe_ms,
            pipe_sky,
            cloud_bgl,
            cloud_pipe,
            cloud_uniform,
            cloud_tex: None,
            cloud_serial: 0,
            last_medium_key: None,
            last_sky_key: None,
        }
    }

    pub fn transmittance_texture(&self) -> &wgpu::Texture {
        &self.transmittance
    }
    pub fn transmittance_view(&self) -> &wgpu::TextureView {
        &self.transmittance_view
    }
    pub fn multiscatter_texture(&self) -> &wgpu::Texture {
        &self.multiscatter
    }
    pub fn multiscatter_view(&self) -> &wgpu::TextureView {
        &self.multiscatter_view
    }
    pub fn sky_view_texture(&self) -> &wgpu::Texture {
        &self.sky_view
    }
    pub fn sky_view_view(&self) -> &wgpu::TextureView {
        &self.sky_view_view
    }
    pub fn cloud_texture(&self) -> Option<&wgpu::Texture> {
        self.cloud_tex.as_ref().map(|t| &t.2)
    }
    pub fn cloud_view(&self) -> Option<&wgpu::TextureView> {
        self.cloud_tex.as_ref().map(|t| &t.3)
    }
    pub fn cloud_serial(&self) -> u64 {
        self.cloud_serial
    }

    /// Ensures the volumetric cloud target exists at `(width, height)`, bumping `cloud_serial`
    /// whenever the texture is freshly allocated or resized.
    pub fn ensure_clouds(&mut self, device: &wgpu::Device, (width, height): (u32, u32)) -> Result<u64> {
        let max_dim = device.limits().max_texture_dimension_2d.min(2048);
        ensure!(
            width >= 1 && height >= 1 && width <= max_dim && height <= max_dim,
            "invalid cloud size {width}x{height} (max {max_dim})"
        );
        let need_alloc = !matches!(&self.cloud_tex, Some((w, h, _, _, _)) if *w == width && *h == height);
        if need_alloc {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("volumetric_clouds"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("clouds_bg"),
                layout: &self.cloud_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.cloud_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                ],
            });
            self.cloud_tex = Some((width, height, tex, view, bg));
            self.cloud_serial += 1;
        }
        Ok(self.cloud_serial)
    }

    /// Dispatches Transmittance -> Multi-Scattering -> Sky-View LUT updates in command order into `encoder`,
    /// skipping unchanged medium LUTs (transmittance + multi-scattering) and unchanged sky-view frames.
    pub fn update_luts(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &AtmosphereParams,
    ) {
        let sun = normalize3(params.sun_direction);
        let rb = finite_or(params.planet_radius_km, 6360.0).max(100.0);
        let rt = finite_or(params.atmo_radius_km, rb + 100.0).max(rb + 10.0);
        let mie_s = [
            finite_or(params.mie_scattering[0], 0.0).max(0.0),
            finite_or(params.mie_scattering[1], 0.0).max(0.0),
            finite_or(params.mie_scattering[2], 0.0).max(0.0),
        ];
        // Scalar Mie extinction must be at least the max of all 3 RGB Mie scattering channels.
        let mie_ext = finite_or(params.mie_extinction, 0.0).max(mie_s[0]).max(mie_s[1]).max(mie_s[2]).max(1e-5);
        let make_u = |(w, h): (u32, u32)| AtmoUniform {
            radii: [
                rb,
                rt,
                finite_or(params.camera_altitude_km, 0.2).max(0.002),
                finite_or(params.rayleigh_scale_km, 8.0).max(0.1),
            ],
            rayleigh: [
                finite_or(params.rayleigh_scattering[0], 0.0).max(0.0),
                finite_or(params.rayleigh_scattering[1], 0.0).max(0.0),
                finite_or(params.rayleigh_scattering[2], 0.0).max(0.0),
                finite_or(params.mie_scale_km, 1.2).max(0.05),
            ],
            mie: [mie_s[0], mie_s[1], mie_s[2], mie_ext],
            ozone: [
                finite_or(params.ozone_absorption[0], 0.0).max(0.0),
                finite_or(params.ozone_absorption[1], 0.0).max(0.0),
                finite_or(params.ozone_absorption[2], 0.0).max(0.0),
                finite_or(params.mie_g, 0.8).clamp(-0.95, 0.95),
            ],
            sun_dir: [sun[0], sun[1], sun[2], 0.999989], // cos(0.267 deg) solar angular radius
            sun_color: [
                finite_or(params.sun_irradiance[0], 1.0).max(0.0),
                finite_or(params.sun_irradiance[1], 1.0).max(0.0),
                finite_or(params.sun_irradiance[2], 1.0).max(0.0),
                finite_or(params.ozone_center_km, 25.0).max(1.0),
            ],
            ground: [
                finite_or(params.ground_albedo[0], 0.2).clamp(0.0, 1.0),
                finite_or(params.ground_albedo[1], 0.2).clamp(0.0, 1.0),
                finite_or(params.ground_albedo[2], 0.2).clamp(0.0, 1.0),
                finite_or(params.ozone_half_width_km, 15.0).max(1.0),
            ],
            dims: [w as f32, h as f32, TRANSMITTANCE_SIZE.0 as f32, TRANSMITTANCE_SIZE.1 as f32],
        };
        let u_sky = make_u(SKY_VIEW_SIZE);
        let medium_key = [
            u_sky.radii[0].to_bits(),
            u_sky.radii[1].to_bits(),
            u_sky.radii[3].to_bits(),
            u_sky.rayleigh[0].to_bits(),
            u_sky.rayleigh[1].to_bits(),
            u_sky.rayleigh[2].to_bits(),
            u_sky.rayleigh[3].to_bits(),
            u_sky.mie[0].to_bits(),
            u_sky.mie[1].to_bits(),
            u_sky.mie[2].to_bits(),
            u_sky.mie[3].to_bits(),
            u_sky.ozone[0].to_bits(),
            u_sky.ozone[1].to_bits(),
            u_sky.ozone[2].to_bits(),
            u_sky.sun_color[3].to_bits(),
            u_sky.ground[0].to_bits(),
            u_sky.ground[1].to_bits(),
            u_sky.ground[2].to_bits(),
            u_sky.ground[3].to_bits(),
        ];
        let sky_key = [
            u_sky.radii[2].to_bits(),
            u_sky.ozone[3].to_bits(),
            u_sky.sun_dir[0].to_bits(),
            u_sky.sun_dir[1].to_bits(),
            u_sky.sun_dir[2].to_bits(),
            u_sky.sun_color[0].to_bits(),
            u_sky.sun_color[1].to_bits(),
            u_sky.sun_color[2].to_bits(),
        ];
        let medium_dirty = self.last_medium_key != Some(medium_key);
        let sky_dirty = medium_dirty || self.last_sky_key != Some(sky_key);
        if !sky_dirty {
            return;
        }

        let dispatch = |enc: &mut wgpu::CommandEncoder,
                        label,
                        pipe: &wgpu::ComputePipeline,
                        bg: &wgpu::BindGroup,
                        (w, h): (u32, u32)| {
            let mut pass =
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        };
        if medium_dirty {
            let u_trans = make_u(TRANSMITTANCE_SIZE);
            let u_ms = make_u(MULTISCATTER_SIZE);
            let u_sz = std::mem::size_of::<AtmoUniform>() as u64;
            let stage = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("atmo_lut_stage"),
                size: u_sz * 3,
                usage: wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            stage.slice(0..u_sz).get_mapped_range_mut().unwrap().copy_from_slice(bytemuck::bytes_of(&u_trans));
            stage.slice(u_sz..u_sz * 2).get_mapped_range_mut().unwrap().copy_from_slice(bytemuck::bytes_of(&u_ms));
            stage.slice(u_sz * 2..u_sz * 3).get_mapped_range_mut().unwrap().copy_from_slice(bytemuck::bytes_of(&u_sky));
            stage.unmap();
            encoder.copy_buffer_to_buffer(&stage, 0, &self.uniform_trans, 0, u_sz);
            encoder.copy_buffer_to_buffer(&stage, u_sz, &self.uniform_ms, 0, u_sz);
            encoder.copy_buffer_to_buffer(&stage, u_sz * 2, &self.uniform_sky, 0, u_sz);
            dispatch(encoder, "atmo_transmittance", &self.pipe_trans, &self.bg_trans, TRANSMITTANCE_SIZE);
            dispatch(encoder, "atmo_multiscatter", &self.pipe_ms, &self.bg_ms, MULTISCATTER_SIZE);
            dispatch(encoder, "atmo_sky_view", &self.pipe_sky, &self.bg_sky, SKY_VIEW_SIZE);
        } else {
            encode_uniform_upload(device, encoder, &self.uniform_sky, bytemuck::bytes_of(&u_sky));
            dispatch(encoder, "atmo_sky_view", &self.pipe_sky, &self.bg_sky, SKY_VIEW_SIZE);
        }
        self.last_medium_key = Some(medium_key);
        self.last_sky_key = Some(sky_key);
    }

    /// Raymarches the 3D volumetric cloud layer at `(width, height)` into an `Rgba16Float` texture.
    /// `camera_pos_km` is in kilometers (matching `inverse_view_projection` in km space).
    /// For scene world-meter projections, prefer [`Self::render_clouds_world`].
    #[allow(clippy::too_many_arguments)]
    pub fn render_clouds(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &AtmosphereParams,
        inverse_view_projection: [f32; 16],
        camera_pos_km: [f32; 3],
        size: (u32, u32),
    ) -> Result<()> {
        self.render_clouds_world(device, encoder, params, inverse_view_projection, camera_pos_km, 1.0, 0.0, size)
    }

    /// Raymarches the 3D volumetric cloud layer using a scene world-space `inverse_view_projection`
    /// and `world_camera_pos` (converted to atmosphere km via `world_to_km` and `base_altitude_km`).
    #[allow(clippy::too_many_arguments)]
    pub fn render_clouds_world(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &AtmosphereParams,
        inverse_view_projection: [f32; 16],
        world_camera_pos: [f32; 3],
        world_to_km: f32,
        base_altitude_km: f32,
        (width, height): (u32, u32),
    ) -> Result<()> {
        self.ensure_clouds(device, (width, height))?;
        ensure!(inverse_view_projection.iter().all(|v| v.is_finite()), "non-finite inverse_view_projection");
        ensure!(world_camera_pos.iter().all(|v| v.is_finite()), "non-finite camera position");
        let sun = normalize3(params.sun_direction);
        let bottom = finite_or(params.cloud_bottom_km, 1.5).max(0.1);
        let top = finite_or(params.cloud_top_km, bottom + 2.5).max(bottom + 0.2);
        let u = CloudUniform {
            inverse_view_projection,
            camera: [
                world_camera_pos[0],
                world_camera_pos[1],
                world_camera_pos[2],
                finite_or(params.planet_radius_km, 6360.0).max(100.0),
            ],
            sun_dir: [sun[0], sun[1], sun[2], finite_or(params.cloud_coverage, 0.5).clamp(0.0, 1.0)],
            sun_color: [
                finite_or(params.sun_irradiance[0], 1.0).max(0.0),
                finite_or(params.sun_irradiance[1], 1.0).max(0.0),
                finite_or(params.sun_irradiance[2], 1.0).max(0.0),
                finite_or(params.cloud_density, 0.65).max(0.0),
            ],
            ambient: [
                finite_or(params.cloud_ambient[0], 0.3).max(0.0),
                finite_or(params.cloud_ambient[1], 0.35).max(0.0),
                finite_or(params.cloud_ambient[2], 0.45).max(0.0),
                bottom,
            ],
            wind: [
                finite_or(params.cloud_wind_km[0], 0.0),
                finite_or(params.cloud_wind_km[1], 0.0),
                finite_or(params.cloud_wind_km[2], 0.0),
                top,
            ],
            params: [
                width as f32,
                height as f32,
                finite_or(params.cloud_detail_freq, 0.35).max(0.05),
                finite_or(params.cloud_silver_g, 0.8).clamp(0.0, 0.92),
            ],
            quality: [
                params.cloud_steps.clamp(8, 64) as f32,
                finite_or(params.cloud_max_march_km, 45.0).clamp(5.0, 120.0),
                finite_or(world_to_km, 1.0).max(1e-6),
                finite_or(base_altitude_km, 0.0),
            ],
        };
        encode_uniform_upload(device, encoder, &self.cloud_uniform, bytemuck::bytes_of(&u));
        let (_, _, _, _, bg) = self.cloud_tex.as_ref().unwrap();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("volumetric_clouds"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.cloud_pipe);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        Ok(())
    }

    #[allow(dead_code)]
    fn _keep_alive(&self) -> (&wgpu::Texture, &wgpu::TextureView, &wgpu::Sampler, &wgpu::BindGroupLayout) {
        (&self.dummy_tex, &self.dummy_view, &self.sampler, &self.bgl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_to_f32(bits: u16) -> f32 {
        let sign = u32::from((bits >> 15) & 1) << 31;
        let exp = u32::from((bits >> 10) & 0x1f);
        let frac = u32::from(bits & 0x3ff);
        if exp == 0 {
            if frac == 0 {
                f32::from_bits(sign)
            } else {
                let mut f = frac;
                let mut e = -14i32;
                while (f & 0x400) == 0 {
                    f <<= 1;
                    e -= 1;
                }
                f &= 0x3ff;
                f32::from_bits(sign | (((e + 127) as u32) << 23) | (f << 13))
            }
        } else if exp == 31 {
            f32::from_bits(sign | (0xff << 23) | (frac << 13))
        } else {
            f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13))
        }
    }

    fn read_rgba16f(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture) -> Vec<[f32; 4]> {
        let (w, h) = (tex.width(), tex.height());
        let bpr = (w * 8).next_multiple_of(256);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atmo_readback"),
            size: u64::from(bpr * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().unwrap();
        let mut out = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let off = (y * bpr + x * 8) as usize;
                out.push([
                    f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]])),
                    f16_to_f32(u16::from_le_bytes([data[off + 2], data[off + 3]])),
                    f16_to_f32(u16::from_le_bytes([data[off + 4], data[off + 5]])),
                    f16_to_f32(u16::from_le_bytes([data[off + 6], data[off + 7]])),
                ]);
            }
        }
        drop(data);
        buf.unmap();
        out
    }

    #[test]
    fn gpu_physical_atmosphere_luts_and_volumetric_clouds_match_physics() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU adapter required for atmosphere witness");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let mut atmo = PhysicalAtmosphere::new(&device);

        // 1. Noon sky: zenith must be distinctly Rayleigh blue (B > R * 1.4), finite everywhere.
        let noon = AtmosphereParams { sun_direction: [0.0, 0.9, -0.4359], ..AtmosphereParams::default() };
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        atmo.update_luts(&device, &mut enc, &noon);
        queue.submit(Some(enc.finish()));
        let ms = read_rgba16f(&device, &queue, atmo.multiscatter_texture());
        assert!(ms.iter().all(|p| p.iter().all(|c| c.is_finite() && *c >= 0.0)));
        assert!(ms[16 * 32 + 24][2] > 0.005, "multi-scattering LUT must be positive in daylight");
        let sky_noon = read_rgba16f(&device, &queue, atmo.sky_view_texture());
        assert!(sky_noon.iter().all(|p| p.iter().all(|c| c.is_finite() && *c >= 0.0)));
        let zenith =
            sky_noon[(SKY_VIEW_SIZE.1 as usize - 1) * SKY_VIEW_SIZE.0 as usize + (SKY_VIEW_SIZE.0 as usize / 2)];
        assert!(zenith[2] > zenith[0] * 1.4, "Rayleigh noon zenith must be blue: {zenith:?}");

        // 2. Sunset sky: sun near horizon -> horizon toward sun shifts warm red/orange (R > B).
        let sunset = AtmosphereParams { sun_direction: [0.0, 0.04, -0.9992], ..AtmosphereParams::default() };
        let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        atmo.update_luts(&device, &mut enc2, &sunset);
        queue.submit(Some(enc2.finish()));
        let sky_sunset = read_rgba16f(&device, &queue, atmo.sky_view_texture());
        let sunset_horiz = sky_sunset
            [(SKY_VIEW_SIZE.1 as usize * 56 / 100) * SKY_VIEW_SIZE.0 as usize + (SKY_VIEW_SIZE.0 as usize / 2)];
        assert!(sunset_horiz[0] > sunset_horiz[2] * 1.2, "sunset horizon toward sun must redden: {sunset_horiz:?}");

        // 3. Volumetric clouds viewed from below: optical absorption + forward silver lining > backscatter.
        let cloud_params = AtmosphereParams {
            sun_direction: [0.0, 0.45, -0.893],
            cloud_coverage: 0.78,
            cloud_density: 1.2,
            ..AtmosphereParams::default()
        };
        let inv_vp_toward_sun: [f32; 16] =
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.6, -1.0, 0.0, 0.0, 0.2, 0.0, 1.0];
        let mut enc3 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        atmo.render_clouds(&device, &mut enc3, &cloud_params, inv_vp_toward_sun, [0.0, 0.2, 0.0], (64, 64)).unwrap();
        queue.submit(Some(enc3.finish()));
        let toward = read_rgba16f(&device, &queue, atmo.cloud_texture().unwrap());
        let min_t = toward.iter().map(|p| p[3]).fold(1.0f32, f32::min);
        let avg_lum_fwd: f32 = toward.iter().map(|p| p[0] + p[1] + p[2]).sum::<f32>() / toward.len() as f32;
        assert!(min_t < 0.85, "clouds with coverage=0.78 must absorb sky light, min_t={min_t}");
        assert!(avg_lum_fwd > 0.05 && avg_lum_fwd.is_finite(), "clouds must scatter finite light: {avg_lum_fwd}");

        let back_params = AtmosphereParams { sun_direction: [0.0, 0.45, 0.893], ..cloud_params };
        let mut enc4 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        atmo.render_clouds(&device, &mut enc4, &back_params, inv_vp_toward_sun, [0.0, 0.2, 0.0], (64, 64)).unwrap();
        queue.submit(Some(enc4.finish()));
        let away = read_rgba16f(&device, &queue, atmo.cloud_texture().unwrap());
        let avg_lum_back: f32 = away.iter().map(|p| p[0] + p[1] + p[2]).sum::<f32>() / away.len() as f32;
        assert!(
            avg_lum_fwd > avg_lum_back * 1.15,
            "forward Henyey-Greenstein silver lining must exceed backscatter: fwd={avg_lum_fwd} back={avg_lum_back}"
        );

        // 4. Clouds viewed from ABOVE (altitude 6 km looking straight down -Y through 1.5..4.0 km shell).
        // World point for NDC (x, y, 0.5): (x, 5.0, y), camera at (0, 6.0, 0) -> ray dir = (x, -1, y).
        let inv_vp_downward: [f32; 16] =
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 6.0, 0.0, 1.0];
        let mut enc5 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        atmo.render_clouds(&device, &mut enc5, &cloud_params, inv_vp_downward, [0.0, 6.0, 0.0], (32, 32)).unwrap();
        queue.submit(Some(enc5.finish()));
        let from_above = read_rgba16f(&device, &queue, atmo.cloud_texture().unwrap());
        let min_t_above = from_above.iter().map(|p| p[3]).fold(1.0f32, f32::min);
        let avg_lum_above: f32 = from_above.iter().map(|p| p[0] + p[1] + p[2]).sum::<f32>() / from_above.len() as f32;
        assert!(
            min_t_above < 0.85 && avg_lum_above > 0.05,
            "clouds viewed downward from above shell must render: min_t={min_t_above} lum={avg_lum_above}"
        );
    }
}
