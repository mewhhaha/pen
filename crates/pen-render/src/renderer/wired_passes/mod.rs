//! End-to-end wiring of high-fidelity rendering passes into `Renderer::draw`:
//! - Hillaire physical atmosphere LUTs & volumetric clouds (`quality.physical_sky`)
//! - Full PBR deferred decals (`frame.pbr_decals`)
//! - Weighted Blended Order-Independent Transparency (`quality.oit`)
//! - Portable 2-level TLAS/BLAS ray tracing (`quality.ray_tracing` 1 = shadows+RTAO, 2 = +reflections+GI)
//! - Dynamic Diffuse Global Illumination probe grid (`quality.ddgi`)
//! - AMD FidelityFX RCAS sharpening & Post-FX polish (`grade.{sharpness, grain, flare, outline_strength}`)
//! - Render-to-texture & split-screen secondary cameras (`frame.camera_targets`)

mod ddgi;
mod decals_oit;
mod meshlets;
mod post_split;
mod rt;
mod sky_env;

use crate::atmosphere_lut::PhysicalAtmosphere;
use crate::built_in::{BuiltIn, BuiltIns};
use crate::ddgi::DdgiVolume;
use crate::decals::PbrDecalPass;
use crate::environment_passes::{
    FoliageInstance, FoliageUniform, TerrainHeightmap, TerrainPatch, TerrainUniform, WaterUniform,
};
use crate::meshlets::MeshletCuller;
use crate::oit::OitPass;
use crate::post_fx::{PostFxPass, RcasSharpener};
use crate::ray_tracing::{GpuBlasPool, TlasRayTracer, TlasSceneGpu};
use crate::render_targets::SplitScreenCompositor;
use crate::restir::RestirRenderer;
use crate::restir_specular::RestirSpecularRenderer;
use crate::sharc::SharcCache;
use pen_protocol::{Matrix, RenderFrame};
use rt::{RtCompositePipelines, SvgfRtState};
use std::collections::HashMap;
use std::sync::Arc;

pub(super) use ddgi::has_active_point_spot_lights;
pub(super) use decals_oit::PreparedPbrDecalBatch;
pub(super) use post_split::SplitViewSlot;
pub(super) use sky_env::EnvPipelines;

/// Built-in optional `RenderFrame::extensions` tags driving outdoor GPU environment passes from Blot.
pub const ENV_TERRAIN_EXT_TAG: u32 = 65_001;
pub const ENV_WATER_EXT_TAG: u32 = 65_002;
pub const ENV_FOLIAGE_EXT_TAG: u32 = 65_003;

/// CDLOD heightfield terrain descriptor attached to [`EnvironmentScene`].
pub struct EnvironmentTerrain {
    pub heightmap: Arc<TerrainHeightmap>,
    pub patches: Vec<TerrainPatch>,
    pub uniform: TerrainUniform,
}

/// GPU-instanced wind-swayed foliage descriptor attached to [`EnvironmentScene`].
pub struct EnvironmentFoliage {
    pub uniform: FoliageUniform,
    pub instances: Vec<FoliageInstance>,
}

/// Outdoor GPU environment passes (CDLOD terrain, Gerstner water, instanced foliage) rendered inside
/// `Renderer::draw`'s opaque HDR scene pass.
#[derive(Default)]
pub struct EnvironmentScene {
    pub terrain: Option<EnvironmentTerrain>,
    pub water: Vec<WaterUniform>,
    pub foliage: Option<EnvironmentFoliage>,
}

const COPY_BLIT_WGSL: &str = r#"
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;

@vertex fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
  let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
  let y = 1.0 - f32(idx & 2u) * 2.0;
  return vec4f(x, y, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(src_tex, 0));
  let uv = pos.xy / max(dims, vec2f(1.0));
  return textureSampleLevel(src_tex, src_samp, uv, 0.0);
}
"#;

struct BlitShared {
    module: wgpu::ShaderModule,
    bgl: wgpu::BindGroupLayout,
    pl: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
}

#[derive(Clone, Copy)]
pub(super) struct PlannedMeshletDraw {
    pub slot: usize,
    pub mesh_slot: usize,
    pub mesh_key: u64,
    pub ml_count: u32,
    pub model: Matrix,
    pub backface_cull: bool,
}

#[derive(Default)]
pub(super) struct WiredPasses {
    pub physical_atmosphere: Option<PhysicalAtmosphere>,
    pub physical_sky_serial: u64,
    pub physical_sky_last_key: (bool, u64),
    pub pbr_decals: HashMap<u32, PbrDecalPass>,
    pub oit_pass: Option<OitPass>,
    pub oit_pipelines: HashMap<(usize, u64, u32, bool), Option<wgpu::RenderPipeline>>,
    pub blas_pool: Option<GpuBlasPool>,
    pub tlas_hash: u64,
    pub tlas_gpu: Option<TlasSceneGpu>,
    pub tlas_bounds: Option<([f32; 3], [f32; 3])>,
    pub ray_tracer: Option<TlasRayTracer>,
    pub restir: Option<RestirRenderer>,
    pub restir_specular: Option<RestirSpecularRenderer>,
    pub sharc: Option<SharcCache>,
    pub sharc_needs_clear: bool,
    svgf_rt: Option<SvgfRtState>,
    rt_composite: Option<RtCompositePipelines>,
    pub ddgi_near: Option<DdgiVolume>,
    pub ddgi_far: Option<DdgiVolume>,
    pub ddgi_active_this_frame: bool,
    pub meshlet_culler: Option<MeshletCuller>,
    pub meshlet_indirect_serial: u64,
    pub last_meshlet_cull_draws: u32,
    pub meshlet_phase1_used_hiz: bool,
    pub last_meshlet_phase2_active: bool,
    pub meshlet_plan: Vec<PlannedMeshletDraw>,
    pub env_pipelines: HashMap<(u32, bool), EnvPipelines>,
    pub env_scene: Option<EnvironmentScene>,
    pub ext_terrain_hm: Option<(u64, TerrainHeightmap)>,
    pub rcas: Option<RcasSharpener>,
    pub post_fx: Option<PostFxPass>,
    pub split_screen: Option<(wgpu::TextureFormat, SplitScreenCompositor)>,
    pub split_views: Vec<SplitViewSlot>,
    pub polish_input: Option<(u32, u32, wgpu::TextureFormat, wgpu::Texture, wgpu::TextureView)>,
    pub msaa_depth_resolve: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
    pub dummy_depth: Option<(wgpu::Texture, wgpu::TextureView)>,
    pub dummy_vel: Option<(wgpu::Texture, wgpu::TextureView)>,
    pub dummy_rgba16f: Option<(wgpu::Texture, wgpu::TextureView)>,
    blit_shared: Option<BlitShared>,
    blit_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    pub in_secondary_camera: bool,
    pub bypass_display_output: bool,
}

/// Builds a lightweight secondary-camera `RenderFrame` cloning only 3D scene fields and
/// omitting audio, UI, services, haptics, video streams, camera targets, sprites, and insets.
pub(super) fn secondary_frame(frame: &RenderFrame, view: Matrix, projection: Matrix) -> RenderFrame {
    RenderFrame {
        view,
        projection,
        clear: frame.clear,
        ambient: frame.ambient,
        light_direction: frame.light_direction,
        light_color: frame.light_color,
        aberration: frame.aberration,
        quality: frame.quality,
        sky: frame.sky,
        probe_volume: frame.probe_volume,
        fog: frame.fog,
        shadow: frame.shadow,
        blur: frame.blur,
        clouds: frame.clouds,
        exposure: frame.exposure,
        grade: frame.grade,
        reflection: frame.reflection,
        motion_blur: frame.motion_blur,
        dof: frame.dof,
        draws: frame.draws.clone(),
        solids: frame.solids.clone(),
        decals: frame.decals.clone(),
        poses: frame.poses.clone(),
        emitters: frame.emitters.clone(),
        lights: frame.lights.clone(),
        cascades: frame.cascades.clone(),
        probes: frame.probes.clone(),
        reflection_probes: frame.reflection_probes.clone(),
        fog_volumes: frame.fog_volumes.clone(),
        bone_matrices: frame.bone_matrices.clone(),
        morph_weights: frame.morph_weights.clone(),
        motions: frame.motions.clone(),
        ribbons: frame.ribbons.clone(),
        area_lights: frame.area_lights.clone(),
        pbr_decals: frame.pbr_decals.clone(),
        ..RenderFrame::default()
    }
}

impl WiredPasses {
    pub fn reset_svgf_history(&mut self) {
        if let Some(svgf) = &mut self.svgf_rt {
            svgf.reset_history();
        }
        if let Some(restir) = &mut self.restir {
            restir.reset_history();
        }
        self.sharc_needs_clear = true;
    }

    pub fn wants_post_polish(frame: &RenderFrame) -> bool {
        frame.grade.sharpness > 0.0
            || frame.quality.upscale
            || frame.grade.grain > 0.0
            || frame.grade.flare > 0.0
            || frame.grade.outline_strength > 0.0
    }

    pub fn polish_target(
        &mut self,
        device: &wgpu::Device,
        size: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureView {
        let size = (size.0.max(1), size.1.max(1));
        if !matches!(&self.polish_input, Some((w, h, fmt, _, _)) if *w == size.0 && *h == size.1 && *fmt == format) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("post_polish_input"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.polish_input = Some((size.0, size.1, format, tex, view));
        }
        self.polish_input.as_ref().unwrap().4.clone()
    }

    pub fn dummy_depth_view(&mut self, device: &wgpu::Device) -> wgpu::TextureView {
        if self.dummy_depth.is_none() {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("wired_dummy_depth"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.dummy_depth = Some((tex, view));
        }
        self.dummy_depth.as_ref().unwrap().1.clone()
    }

    pub fn dummy_velocity_view(&mut self, device: &wgpu::Device) -> wgpu::TextureView {
        if self.dummy_vel.is_none() {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("wired_dummy_vel"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.dummy_vel = Some((tex, view));
        }
        self.dummy_vel.as_ref().unwrap().1.clone()
    }

    pub fn dummy_rgba16f_view(&mut self, device: &wgpu::Device) -> wgpu::TextureView {
        if self.dummy_rgba16f.is_none() {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("wired_dummy_rgba16f"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.dummy_rgba16f = Some((tex, view));
        }
        self.dummy_rgba16f.as_ref().unwrap().1.clone()
    }

    /// Resolves a multisampled depth view into a cached 1x `R32Float` texture via `BuiltIn::OpaqueDepth`
    /// so screen-space outlines work identically under MSAA.
    pub fn resolve_msaa_depth(
        &mut self,
        device: &wgpu::Device,
        built_ins: &mut BuiltIns,
        encoder: &mut wgpu::CommandEncoder,
        depth_only: &wgpu::TextureView,
        size: (u32, u32),
        samples: u32,
    ) -> wgpu::TextureView {
        let size = (size.0.max(1), size.1.max(1));
        if !matches!(&self.msaa_depth_resolve, Some((w, h, _, _)) if *w == size.0 && *h == size.1) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("wired_msaa_depth_resolve"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.msaa_depth_resolve = Some((size.0, size.1, tex, view));
        }
        let dst = self.msaa_depth_resolve.as_ref().unwrap().3.clone();
        let pipeline = built_ins.get(BuiltIn::OpaqueDepth(samples));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wired_msaa_depth_resolve_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_only) }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("wired_msaa_depth_resolve"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
        dst
    }

    fn ensure_blit(&mut self, device: &wgpu::Device, target_format: wgpu::TextureFormat) {
        let shared = self.blit_shared.get_or_insert_with(|| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("wired_copy_blit"),
                source: wgpu::ShaderSource::Wgsl(COPY_BLIT_WGSL.into()),
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("wired_copy_blit_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("wired_copy_blit_pl"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("wired_copy_blit_samp"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            BlitShared { module, bgl, pl, sampler }
        });
        self.blit_pipelines.entry(target_format).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("wired_copy_blit"),
                layout: Some(&shared.pl),
                vertex: wgpu::VertexState {
                    module: &shared.module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shared.module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
    }

    pub fn blit(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
    ) {
        self.ensure_blit(device, target_format);
        let shared = self.blit_shared.as_ref().unwrap();
        let pipeline = self.blit_pipelines.get(&target_format).unwrap();
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wired_copy_blit_bg"),
            layout: &shared.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shared.sampler) },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("wired_copy_blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
