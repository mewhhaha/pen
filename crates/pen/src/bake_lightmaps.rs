//! `pen bake-lightmaps`: headless GPU multi-bounce lightmap baker CLI.
//!
//! Reads a JSON bake job describing scene geometry (or `.mesh.json` paths + transforms + materials),
//! automatically unwraps/packs UV2 lightmap charts when absent, traces multi-bounce path-traced
//! irradiance, directional dominant-light, and sun shadowmask maps on the GPU, and writes
//! `lightmap_irradiance.exr`, `lightmap_directional.png`, `lightmap_shadowmask.png`, and
//! `lightmap_manifest.json`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use pen_render::{
    Blas, BlasTriangle, GpuLightmapBaker, LightmapBakeParams, LightmapTriangle, TlasInstance, TlasScene, TlasSceneGpu,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct MeshJson {
    #[serde(default)]
    positions: Vec<[f32; 3]>,
    #[serde(default)]
    normals: Vec<[f32; 3]>,
    #[serde(default)]
    indices: Vec<u32>,
    #[serde(default)]
    uvs1: Vec<[f32; 2]>,
}

#[derive(Deserialize)]
struct BakeInstanceSpec {
    /// Either a path to a `.mesh.json` file (relative to job JSON) or inline triangles.
    mesh: Option<String>,
    #[serde(default)]
    triangles: Vec<[[f32; 3]; 3]>,
    #[serde(default = "identity_matrix")]
    transform: [f32; 16],
    #[serde(default = "default_albedo")]
    albedo: [f32; 3],
    #[serde(default = "default_roughness")]
    roughness: f32,
    #[serde(default)]
    emissive: [f32; 3],
    #[serde(default)]
    metallic: f32,
    #[serde(default = "default_true")]
    bake_target: bool,
}

#[derive(Deserialize)]
struct BakeJob {
    instances: Vec<BakeInstanceSpec>,
    #[serde(default = "default_sun_dir")]
    sun_direction: [f32; 3],
    #[serde(default = "default_sun_color")]
    sun_color: [f32; 3],
    #[serde(default = "default_sun_radius")]
    sun_angular_radius: f32,
    #[serde(default = "default_sky_zenith")]
    sky_zenith: [f32; 3],
    #[serde(default = "default_sky_horizon")]
    sky_horizon: [f32; 3],
    #[serde(default = "default_size")]
    width: u32,
    #[serde(default = "default_size")]
    height: u32,
    #[serde(default = "default_samples")]
    samples: u32,
    #[serde(default = "default_bounces")]
    bounces: u32,
    #[serde(default = "default_gutter")]
    gutter: u32,
}

#[derive(Serialize)]
struct BakeManifest {
    width: u32,
    height: u32,
    samples: u32,
    bounces: u32,
    chart_count: u32,
    utilization: f32,
    irradiance: String,
    directional: String,
    shadowmask: String,
    triangle_uv2: Vec<[[f32; 2]; 3]>,
}

fn identity_matrix() -> [f32; 16] {
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
}
fn default_albedo() -> [f32; 3] {
    [0.8, 0.8, 0.8]
}
fn default_roughness() -> f32 {
    0.5
}
fn default_true() -> bool {
    true
}
fn default_sun_dir() -> [f32; 3] {
    [0.4, 0.8, 0.45]
}
fn default_sun_color() -> [f32; 3] {
    [3.0, 2.9, 2.7]
}
fn default_sun_radius() -> f32 {
    0.02
}
fn default_sky_zenith() -> [f32; 3] {
    [0.15, 0.25, 0.5]
}
fn default_sky_horizon() -> [f32; 3] {
    [0.35, 0.42, 0.55]
}
fn default_size() -> u32 {
    256
}
fn default_samples() -> u32 {
    64
}
fn default_bounces() -> u32 {
    2
}
fn default_gutter() -> u32 {
    2
}

fn xform_pt(m: &[f32; 16], p: [f32; 3]) -> [f32; 3] {
    [
        m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
        m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
        m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    ]
}

fn xform_vec(m: &[f32; 16], v: [f32; 3]) -> [f32; 3] {
    let x = m[0] * v[0] + m[4] * v[1] + m[8] * v[2];
    let y = m[1] * v[0] + m[5] * v[1] + m[9] * v[2];
    let z = m[2] * v[0] + m[6] * v[1] + m[10] * v[2];
    let len = (x * x + y * y + z * z).sqrt();
    if len > 1e-6 { [x / len, y / len, z / len] } else { [0.0, 1.0, 0.0] }
}

pub struct BakeCliOptions {
    pub job_path: PathBuf,
    pub out_dir: PathBuf,
    pub size_override: Option<u32>,
    pub samples_override: Option<u32>,
    pub bounces_override: Option<u32>,
}

pub fn run(opts: &BakeCliOptions) -> Result<()> {
    let raw = std::fs::read_to_string(&opts.job_path)
        .with_context(|| format!("reading bake job {}", opts.job_path.display()))?;
    let job: BakeJob = serde_json::from_str(&raw).context("parsing lightmap bake job JSON")?;
    ensure!(!job.instances.is_empty(), "bake job must list at least one instance");

    let base_dir = opts.job_path.parent().unwrap_or_else(|| Path::new("."));
    let mut blases = Vec::with_capacity(job.instances.len());
    let mut tlas_instances = Vec::with_capacity(job.instances.len());
    let mut target_tris = Vec::new();

    for spec in &job.instances {
        let mut local_tris = Vec::new();
        let mut local_uv2: Vec<Option<[[f32; 2]; 3]>> = Vec::new();

        if let Some(mesh_rel) = &spec.mesh {
            let mesh_path = base_dir.join(mesh_rel);
            let mraw = std::fs::read_to_string(&mesh_path)
                .with_context(|| format!("reading mesh {}", mesh_path.display()))?;
            let mesh: MeshJson = serde_json::from_str(&mraw)?;
            let idx_list: Vec<u32> = if mesh.indices.is_empty() {
                (0..mesh.positions.len() as u32).collect()
            } else {
                mesh.indices
            };
            for tri in idx_list.chunks_exact(3) {
                let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                if i0 >= mesh.positions.len() || i1 >= mesh.positions.len() || i2 >= mesh.positions.len() {
                    continue;
                }
                let (p0, p1, p2) = (mesh.positions[i0], mesh.positions[i1], mesh.positions[i2]);
                let bt = if i0 < mesh.normals.len() && i1 < mesh.normals.len() && i2 < mesh.normals.len() {
                    BlasTriangle::smooth(p0, p1, p2, mesh.normals[i0], mesh.normals[i1], mesh.normals[i2])
                } else {
                    BlasTriangle::flat(p0, p1, p2)
                };
                let uv2 = if i0 < mesh.uvs1.len() && i1 < mesh.uvs1.len() && i2 < mesh.uvs1.len() {
                    Some([mesh.uvs1[i0], mesh.uvs1[i1], mesh.uvs1[i2]])
                } else {
                    None
                };
                local_tris.push(bt);
                local_uv2.push(uv2);
            }
        } else {
            for t in &spec.triangles {
                local_tris.push(BlasTriangle::flat(t[0], t[1], t[2]));
                local_uv2.push(None);
            }
        }
        if local_tris.is_empty() {
            continue;
        }
        let blas_idx = u32::try_from(blases.len())?;
        blases.push(Blas::build(&local_tris)?);
        let mut ti = TlasInstance::new(blas_idx, spec.transform);
        ti.albedo = spec.albedo;
        ti.roughness = spec.roughness;
        ti.emissive = spec.emissive;
        ti.metallic = spec.metallic;
        tlas_instances.push(ti);

        if spec.bake_target {
            for (bt, uv2) in local_tris.iter().zip(local_uv2) {
                target_tris.push(LightmapTriangle {
                    positions: [
                        xform_pt(&spec.transform, bt.v0),
                        xform_pt(&spec.transform, bt.v1),
                        xform_pt(&spec.transform, bt.v2),
                    ],
                    normals: [
                        xform_vec(&spec.transform, bt.n0),
                        xform_vec(&spec.transform, bt.n1),
                        xform_vec(&spec.transform, bt.n2),
                    ],
                    uv2,
                });
            }
        }
    }
    if target_tris.is_empty() {
        bail!("no bake_target triangles found in {}", opts.job_path.display());
    }

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .context("headless GPU adapter required for pen bake-lightmaps")?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;

    let tlas = TlasScene::build(&blases, &tlas_instances)?;
    let tlas_gpu = TlasSceneGpu::upload(&device, &tlas)?;
    let baker = GpuLightmapBaker::new(&device);

    let width = opts.size_override.unwrap_or(job.width);
    let height = opts.size_override.unwrap_or(job.height);
    let samples = opts.samples_override.unwrap_or(job.samples);
    let bounces = opts.bounces_override.unwrap_or(job.bounces);

    let params = LightmapBakeParams {
        width,
        height,
        samples_per_texel: samples,
        max_bounces: bounces,
        gutter_texels: job.gutter,
        denoise: true,
        sun_direction: job.sun_direction,
        sun_angular_radius: job.sun_angular_radius,
        sun_color: job.sun_color,
        sky_zenith: job.sky_zenith,
        sky_horizon: job.sky_horizon,
        ..Default::default()
    };

    let (packed, baked) = baker.bake(&device, &queue, &tlas_gpu, &target_tris, &params)?;
    baked.write_to_dir(&opts.out_dir)?;

    let manifest = BakeManifest {
        width,
        height,
        samples,
        bounces,
        chart_count: packed.chart_count,
        utilization: packed.utilization,
        irradiance: "lightmap_irradiance.exr".into(),
        directional: "lightmap_directional.png".into(),
        shadowmask: "lightmap_shadowmask.png".into(),
        triangle_uv2: packed.triangle_uv2,
    };
    std::fs::write(opts.out_dir.join("lightmap_manifest.json"), serde_json::to_string_pretty(&manifest)?)?;
    println!(
        "Baked {}x{} lightmaps ({} charts, {:.0}% UV2 utilization, {} spp, {} bounces) -> {}",
        width,
        height,
        packed.chart_count,
        packed.utilization * 100.0,
        samples,
        bounces,
        opts.out_dir.display()
    );
    Ok(())
}
