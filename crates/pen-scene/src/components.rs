//! Typed scene components covering rendering, lighting, cameras, GI/lightmaps, physics, audio, and particles.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

fn default_true() -> bool {
    true
}
fn default_one() -> f32 {
    1.0
}
fn default_fov() -> f32 {
    60.0
}
fn default_near() -> f32 {
    0.1
}
fn default_far() -> f32 {
    500.0
}
fn default_white3() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}
fn default_clear() -> [f32; 4] {
    [0.08, 0.10, 0.14, 1.0]
}
fn default_uv_scale_offset() -> [f32; 4] {
    [1.0, 1.0, 0.0, 0.0]
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LodLevel {
    pub mesh: String,
    pub material: String,
    /// Relative screen-height threshold below which this LOD activates (`1.0` = LOD0).
    pub screen_threshold: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightKind {
    Directional,
    Point,
    Spot,
    AreaRect,
    AreaDisk,
    AreaTube,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum ColliderShape {
    Box { half_extents: [f32; 3] },
    Sphere { radius: f32 },
    Capsule { radius: f32, half_height: f32 },
    ConvexHull { points: Vec<[f32; 3]> },
    TriMesh { mesh: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BodyMode {
    Static,
    Dynamic,
    Kinematic,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Component {
    MeshRenderer {
        mesh: String,
        material: String,
        #[serde(default = "default_true")]
        cast_shadows: bool,
        #[serde(default = "default_true")]
        receive_shadows: bool,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        lod_group: Vec<LodLevel>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parameters: Option<[f32; 4]>,
    },
    Light {
        kind: LightKind,
        #[serde(default = "default_white3")]
        color: [f32; 3],
        #[serde(default = "default_one")]
        intensity: f32,
        #[serde(default)]
        range: f32,
        #[serde(default)]
        inner_cone: f32,
        #[serde(default)]
        outer_cone: f32,
        #[serde(default = "default_one")]
        width: f32,
        #[serde(default = "default_one")]
        height: f32,
        #[serde(default)]
        cast_shadows: bool,
    },
    Camera {
        #[serde(default = "default_fov")]
        fov_deg: f32,
        #[serde(default = "default_near")]
        near: f32,
        #[serde(default = "default_far")]
        far: f32,
        #[serde(default = "default_true")]
        active: bool,
        #[serde(default = "default_clear")]
        clear_color: [f32; 4],
        #[serde(default, skip_serializing_if = "Option::is_none")]
        target_material: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        viewport: Option<[f32; 4]>,
    },
    ReflectionProbe {
        id: u32,
        extent: [f32; 3],
        #[serde(default)]
        blend: f32,
        #[serde(default)]
        priority: u32,
    },
    DdgiVolume {
        spacing: [f32; 3],
        dims: [u32; 3],
        #[serde(default)]
        rays_per_probe: u32,
    },
    LightmapRef {
        irradiance_tex: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        directional_tex: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        shadowmask_tex: Option<String>,
        #[serde(default = "default_uv_scale_offset")]
        uv_scale_offset: [f32; 4],
    },
    Collider {
        #[serde(flatten)]
        shape: ColliderShape,
        #[serde(default)]
        is_trigger: bool,
        #[serde(default = "default_layer")]
        layer: u32,
    },
    RigidBody {
        mode: BodyMode,
        #[serde(default = "default_one")]
        mass: f32,
        #[serde(default)]
        linear_damping: f32,
        #[serde(default)]
        angular_damping: f32,
    },
    AudioSource {
        clip: String,
        #[serde(default)]
        bus: u32,
        #[serde(default = "default_one")]
        volume: f32,
        #[serde(default = "default_one")]
        pitch: f32,
        #[serde(default)]
        looping: bool,
        #[serde(default = "default_true")]
        spatial: bool,
        #[serde(default = "default_one")]
        ref_distance: f32,
        #[serde(default = "default_far")]
        max_distance: f32,
    },
    ReverbZone {
        half_extents: [f32; 3],
        #[serde(default = "default_one")]
        blend: f32,
        #[serde(default = "default_one")]
        decay: f32,
        #[serde(default)]
        damping: f32,
        #[serde(default = "default_one")]
        wet: f32,
    },
    ParticleEmitter {
        material: String,
        rate: f32,
        lifetime: f32,
        radius: f32,
        velocity: [f32; 3],
        gravity: [f32; 3],
        size_start: f32,
        size_end: f32,
        color_start: [f32; 4],
        color_end: [f32; 4],
        #[serde(default)]
        additive: bool,
    },
    AnimationPlayer {
        #[serde(default)]
        clips: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        active_clip: Option<String>,
        #[serde(default = "default_one")]
        speed: f32,
        #[serde(default = "default_true")]
        looping: bool,
    },
    Custom {
        kind: String,
        #[serde(default)]
        properties: BTreeMap<String, serde_json::Value>,
    },
}

fn default_layer() -> u32 {
    1
}

impl Component {
    /// Stable component type discriminant name used by sparse prefab overrides.
    pub fn type_name(&self) -> &'static str {
        match self {
            Component::MeshRenderer { .. } => "MeshRenderer",
            Component::Light { .. } => "Light",
            Component::Camera { .. } => "Camera",
            Component::ReflectionProbe { .. } => "ReflectionProbe",
            Component::DdgiVolume { .. } => "DdgiVolume",
            Component::LightmapRef { .. } => "LightmapRef",
            Component::Collider { .. } => "Collider",
            Component::RigidBody { .. } => "RigidBody",
            Component::AudioSource { .. } => "AudioSource",
            Component::ReverbZone { .. } => "ReverbZone",
            Component::ParticleEmitter { .. } => "ParticleEmitter",
            Component::AnimationPlayer { .. } => "AnimationPlayer",
            Component::Custom { .. } => "Custom",
        }
    }
}
