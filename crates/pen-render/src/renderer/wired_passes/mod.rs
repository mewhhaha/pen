//! End-to-end wiring of high-fidelity rendering passes into `Renderer::draw`:
//! - Hillaire physical atmosphere LUTs & volumetric clouds (`quality.physical_sky`)
//! - Full PBR deferred decals (`frame.pbr_decals`)
//! - Weighted Blended Order-Independent Transparency (`quality.oit`)
//! - Portable 2-level TLAS/BLAS ray tracing (`quality.ray_tracing` 1 = shadows+RTAO, 2 = +reflections+GI)
//! - Dynamic Diffuse Global Illumination probe grid (`quality.ddgi`)
//! - AMD FidelityFX RCAS sharpening & Post-FX polish (`grade.{sharpness, grain, flare, outline_strength}`)
//! - Render-to-texture & split-screen secondary cameras (`frame.camera_targets`)

#[cfg(feature = "render-rt")]
mod ddgi;
mod decals_oit;
mod gpu_particles;
mod jfa_outline;
mod lines;
mod meshlets;
mod post_split;
mod render2d;
#[cfg(feature = "render-rt")]
mod rt;
mod sky_env;

#[cfg(feature = "render-atmosphere")]
use crate::atmosphere_lut::PhysicalAtmosphere;
use crate::built_in::{BuiltIn, BuiltIns};
#[cfg(feature = "render-rt")]
use crate::ddgi::DdgiVolume;
#[cfg(feature = "render-pbr-decals")]
use crate::decals::PbrDecalPass;
use crate::environment_passes::{
    FoliageInstance, FoliageUniform, TerrainHeightmap, TerrainPatch, TerrainUniform, WaterUniform,
};
#[cfg(feature = "render-meshlets")]
use crate::meshlets::MeshletCuller;
#[cfg(feature = "render-oit")]
use crate::oit::OitPass;
#[cfg(feature = "render-post-fx")]
use crate::post_fx::{PostFxPass, RcasSharpener};
#[cfg(feature = "render-rt")]
use crate::ray_tracing::{GpuBlasPool, TlasRayTracer, TlasSceneGpu};
use crate::render_targets::SplitScreenCompositor;
#[cfg(feature = "render-rt")]
use crate::restir::RestirRenderer;
#[cfg(feature = "render-rt")]
use crate::restir_specular::RestirSpecularRenderer;
#[cfg(feature = "render-rt")]
use crate::sharc::SharcCache;
use pen_protocol::{Matrix, RenderFrame};
#[cfg(feature = "render-rt")]
use rt::{RtCompositePipelines, SvgfRtState};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "render-rt")]
pub(super) use ddgi::has_active_point_spot_lights;
pub(super) use decals_oit::PreparedPbrDecalBatch;
pub use gpu_particles::GpuParticleScene;
pub(super) use gpu_particles::GpuParticleWiredState;
pub use jfa_outline::{JfaOutlineScene, JfaOutlineTarget};
pub(super) use jfa_outline::{JfaOutlineWiredState, OutlineScenes, outlined};
pub use lines::LineScene;
pub(super) use lines::LineWiredState;
pub(super) use post_split::SplitViewSlot;
pub(super) use render2d::Render2dWiredState;
pub use render2d::{RadianceCascades2dLayer, Scene2d, SpriteLayer2d, TilemapLayer2d};
#[cfg(feature = "render-env")]
pub(super) use sky_env::EnvPipelines;

/// Built-in optional `RenderFrame::extensions` tags driving outdoor GPU environment passes from Blot.
pub const ENV_TERRAIN_EXT_TAG: u32 = 65_001;
pub const ENV_WATER_EXT_TAG: u32 = 65_002;
pub const ENV_FOLIAGE_EXT_TAG: u32 = 65_003;
// Note: tag 65_004 is reserved by `pen_runtime::reverb::ACOUSTICS_EXT_TAG`.
/// Built-in optional `RenderFrame::extensions` tags for GPU mesh particles, antialiased lines,
/// JFA silhouette outlines, vector SDF UI, full 2D world/GI scenes, and Nanite meshlet/DAG LOD control.
pub const GPU_PARTICLES_EXT_TAG: u32 = 65_005;
pub const LINES_EXT_TAG: u32 = 65_006;
pub const JFA_OUTLINE_EXT_TAG: u32 = 65_007;
pub const RENDER2D_EXT_TAG: u32 = 65_008;
pub const RENDER2D_WORLD_EXT_TAG: u32 = 65_009;
/// Per-frame GPU meshlet / Cluster DAG ("Nanite") override:
/// `[mode]` or `[mode, pixel_error_threshold]`:
/// - `mode == 0`: Off (bypass GPU meshlet & Cluster DAG culling; draw standard indexed meshes)
/// - `mode == 1`: Meshlets only (two-phase GPU meshlet culling at full LOD0, no DAG simplification)
/// - `mode >= 2`: Cluster DAG continuous LOD + two-phase Hi-Z occlusion culling (optional `pixel_error_threshold` in px)
pub const NANITE_EXT_TAG: u32 = 65_010;

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

#[cfg_attr(not(feature = "render-meshlets"), allow(dead_code))]
#[derive(Clone, Copy)]
pub(super) struct PlannedMeshletDraw {
    pub slot: usize,
    pub mesh_slot: usize,
    pub mesh_key: u64,
    pub ml_count: u32,
    pub model: Matrix,
    pub backface_cull: bool,
}

#[cfg_attr(not(feature = "render-meshlets"), allow(dead_code))]
#[derive(Clone, Copy)]
pub(super) struct PlannedDagDraw {
    pub slot: usize,
    pub mesh_slot: usize,
    pub model: Matrix,
    pub backface_cull: bool,
}

#[derive(Clone, Copy)]
pub(super) struct DagLodSettings {
    pub enabled: bool,
    pub pixel_error_threshold: f32,
    #[cfg_attr(not(feature = "render-meshlets"), allow(dead_code))]
    pub frame_pixel_error: f32,
}

impl Default for DagLodSettings {
    fn default() -> Self {
        Self { enabled: true, pixel_error_threshold: 1.5, frame_pixel_error: 1.5 }
    }
}

#[cfg(feature = "render-rt")]
#[derive(Default)]
pub(super) struct RtHistory {
    restir: Option<RestirRenderer>,
    restir_specular: Option<RestirSpecularRenderer>,
    svgf_rt: Option<Box<SvgfRtState>>,
}

#[cfg(not(feature = "render-rt"))]
#[derive(Default)]
pub(super) struct RtHistory;

#[derive(Default)]
pub(super) struct WiredPasses {
    #[cfg(feature = "render-atmosphere")]
    pub physical_atmosphere: Option<PhysicalAtmosphere>,
    #[cfg(feature = "render-atmosphere")]
    pub physical_sky_serial: u64,
    #[cfg(feature = "render-atmosphere")]
    pub physical_sky_last_key: (bool, u64),
    pub scene_depth_serial: u64,
    #[cfg(feature = "render-pbr-decals")]
    pub pbr_decals: HashMap<u32, PbrDecalPass>,
    #[cfg(feature = "render-oit")]
    pub oit_pass: Option<OitPass>,
    #[cfg(feature = "render-oit")]
    pub oit_pipelines: HashMap<(usize, u64, u32, bool), Option<wgpu::RenderPipeline>>,
    #[cfg(feature = "render-rt")]
    pub blas_pool: Option<GpuBlasPool>,
    #[cfg(feature = "render-rt")]
    pub tlas_hash: u64,
    #[cfg(feature = "render-rt")]
    pub tlas_gpu: Option<TlasSceneGpu>,
    #[cfg(feature = "render-rt")]
    pub tlas_bounds: Option<([f32; 3], [f32; 3])>,
    #[cfg(feature = "render-rt")]
    pub tlas_retained_epoch: u64,
    #[cfg(feature = "render-rt")]
    pub tlas_retained_revision: u64,
    #[cfg(feature = "render-rt")]
    pub tlas_asset_revision: u64,
    #[cfg(feature = "render-rt")]
    pub tlas_solids_revision: u64,
    #[cfg(feature = "render-rt")]
    pub tlas_instances_scratch: Vec<crate::ray_tracing::TlasInstance>,
    #[cfg(feature = "render-rt")]
    pub ray_tracer: Option<TlasRayTracer>,
    pub rt_history: RtHistory,
    #[cfg(feature = "render-rt")]
    pub sharc: Option<SharcCache>,
    #[cfg(feature = "render-rt")]
    pub sharc_needs_clear: bool,
    #[cfg(feature = "render-rt")]
    sharc_frame: Option<u64>,
    pub split_child: bool,
    #[cfg(feature = "render-rt")]
    rt_composite: Option<RtCompositePipelines>,
    #[cfg(feature = "render-rt")]
    pub ddgi_near: Option<DdgiVolume>,
    #[cfg(feature = "render-rt")]
    pub ddgi_far: Option<DdgiVolume>,
    #[cfg(feature = "render-rt")]
    pub ddgi_active_this_frame: bool,
    #[cfg(feature = "render-meshlets")]
    pub meshlet_culler: Option<MeshletCuller>,
    #[cfg_attr(not(feature = "render-meshlets"), allow(dead_code))]
    pub meshlet_indirect_serial: u64,
    pub last_meshlet_cull_draws: u32,
    pub meshlet_phase1_used_hiz: bool,
    pub last_meshlet_phase2_active: bool,
    pub meshlet_plan: Vec<PlannedMeshletDraw>,
    #[cfg(feature = "render-meshlets")]
    pub meshlet_dag_culler: Option<crate::meshlet_dag::MeshletDagCuller>,
    pub last_meshlet_dag_draws: u32,
    pub dag_phase1_used_hiz: bool,
    pub dag_plan: Vec<PlannedDagDraw>,
    pub dag_lod: DagLodSettings,
    #[cfg(feature = "render-env")]
    pub env_pipelines: HashMap<(u32, bool), EnvPipelines>,
    pub env_scene: Option<EnvironmentScene>,
    #[cfg(feature = "render-env")]
    pub ext_terrain_hm: Option<(u64, TerrainHeightmap)>,
    pub gpu_particles: GpuParticleWiredState,
    pub lines: LineWiredState,
    pub jfa_outline: JfaOutlineWiredState,
    pub render2d: Render2dWiredState,
    #[cfg(feature = "render-post-fx")]
    pub rcas: Option<RcasSharpener>,
    #[cfg(feature = "render-post-fx")]
    pub post_fx: Option<PostFxPass>,
    pub split_screen: Option<(wgpu::TextureFormat, SplitScreenCompositor)>,
    pub split_views: Vec<SplitViewSlot>,
    pub polish_input: Option<(u32, u32, wgpu::TextureFormat, wgpu::Texture, wgpu::TextureView)>,
    #[allow(dead_code)]
    pub msaa_depth_resolve: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
    #[allow(dead_code)]
    msaa_depth_resolve_bg: Option<(wgpu::TextureView, u32, wgpu::BindGroup)>,
    #[allow(dead_code)]
    pub dummy_depth: Option<(wgpu::Texture, wgpu::TextureView)>,
    #[cfg_attr(not(feature = "render-rt"), allow(dead_code))]
    pub dummy_vel: Option<(wgpu::Texture, wgpu::TextureView)>,
    #[cfg_attr(not(feature = "render-rt"), allow(dead_code))]
    pub dummy_rgba16f: Option<(wgpu::Texture, wgpu::TextureView)>,
    blit_shared: Option<BlitShared>,
    blit_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    blit_bg: Option<(wgpu::TextureView, wgpu::BindGroup)>,
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
    #[cfg(feature = "render-post-fx")]
    pub fn wants_post_polish(frame: &RenderFrame) -> bool {
        frame.grade.sharpness > 0.0
            || frame.quality.upscale
            || frame.grade.grain > 0.0
            || frame.grade.flare > 0.0
            || frame.grade.outline_strength > 0.0
    }

    #[cfg(not(feature = "render-post-fx"))]
    pub fn wants_post_polish(_frame: &RenderFrame) -> bool {
        false
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

    #[allow(dead_code)]
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

    #[cfg_attr(not(feature = "render-rt"), allow(dead_code))]
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

    #[cfg_attr(not(feature = "render-rt"), allow(dead_code))]
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
    #[allow(dead_code)]
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
        if self.msaa_depth_resolve_bg.as_ref().is_none_or(|(view, s, _)| view != depth_only || *s != samples) {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wired_msaa_depth_resolve_bg"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_only),
                }],
            });
            self.msaa_depth_resolve_bg = Some((depth_only.clone(), samples, bg));
        }
        let bg = &self.msaa_depth_resolve_bg.as_ref().unwrap().2;
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
        pass.set_bind_group(0, bg, &[]);
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
                cache: crate::pipeline_cache::active().as_ref(),
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
        if self.blit_bg.as_ref().is_none_or(|(view, _)| view != src) {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wired_copy_blit_bg"),
                layout: &shared.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shared.sampler) },
                ],
            });
            self.blit_bg = Some((src.clone(), bg));
        }
        let bg = &self.blit_bg.as_ref().unwrap().1;
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
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
