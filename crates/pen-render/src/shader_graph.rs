//! Visual PBR Material / Shader Node Graph Compiler (`.matgraph.json` -> validated WGSL).
//!
//! Compiles a typed directed acyclic graph (DAG) of procedural noise, triplanar/UV2 textures,
//! Fresnel, color ramps, normal blending, and math nodes into a complete GGX PBR WGSL shader
//! with cycle detection, automatic scalar/vector type promotion, and live GPU compilation.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "shader node indices and pixel readback")]

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

/// Value type flowing along a shader graph edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortType {
    Float,
    Vec2,
    Vec3,
    Vec4,
}

impl PortType {
    fn wgsl(self) -> &'static str {
        match self {
            PortType::Float => "f32",
            PortType::Vec2 => "vec2f",
            PortType::Vec3 => "vec3f",
            PortType::Vec4 => "vec4f",
        }
    }
}

/// Reference to another node's output by unique string id.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeRef {
    pub node: String,
    #[serde(default)]
    pub output: u32,
}

impl NodeRef {
    pub fn of(node: impl Into<String>) -> Self {
        Self { node: node.into(), output: 0 }
    }
}

/// Color stop along a [`ShaderNodeKind::ColorRamp`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RampStop {
    pub t: f32,
    pub color: [f32; 3],
}

/// Supported visual material graph node kinds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ShaderNodeKind {
    // Constants & vertex/scene inputs
    FloatConst { value: f32 },
    Vec2Const { value: [f32; 2] },
    Vec3Const { value: [f32; 3] },
    Vec4Const { value: [f32; 4] },
    Uv,
    Uv1,
    WorldPos,
    WorldNormal,
    ViewDir,
    VertexColor,
    Time,
    InstanceParam { index: u32 },

    // Coordinate transforms & textures
    UvTransform {
        #[serde(default)]
        uv: Option<NodeRef>,
        #[serde(default = "default_one2")]
        scale: [f32; 2],
        #[serde(default)]
        offset: [f32; 2],
        #[serde(default)]
        rotation_rad: f32,
    },
    Texture2D {
        #[serde(default)]
        uv: Option<NodeRef>,
        #[serde(default = "default_white3")]
        fallback_color: [f32; 3],
    },
    Triplanar {
        #[serde(default = "default_one")]
        scale: f32,
        #[serde(default = "default_four")]
        sharpness: f32,
    },

    // Procedural noise
    SimplexNoise {
        #[serde(default)]
        coords: Option<NodeRef>,
        #[serde(default = "default_four")]
        scale: f32,
    },
    WorleyNoise {
        #[serde(default)]
        coords: Option<NodeRef>,
        #[serde(default = "default_four")]
        scale: f32,
    },
    FbmNoise {
        #[serde(default)]
        coords: Option<NodeRef>,
        #[serde(default = "default_four")]
        scale: f32,
        #[serde(default = "default_octaves")]
        octaves: u32,
    },

    // Shading helpers
    Fresnel {
        #[serde(default = "default_five")]
        power: f32,
    },
    ColorRamp {
        t: NodeRef,
        stops: Vec<RampStop>,
    },
    NormalBlend {
        base: NodeRef,
        detail: NodeRef,
    },

    // Math & vector operations
    Add { a: NodeRef, b: NodeRef },
    Sub { a: NodeRef, b: NodeRef },
    Mul { a: NodeRef, b: NodeRef },
    Div { a: NodeRef, b: NodeRef },
    Pow { base: NodeRef, exp: NodeRef },
    Sin { x: NodeRef },
    Cos { x: NodeRef },
    Clamp { x: NodeRef, min: f32, max: f32 },
    Smoothstep { edge0: f32, edge1: f32, x: NodeRef },
    Lerp { a: NodeRef, b: NodeRef, t: NodeRef },
    Dot { a: NodeRef, b: NodeRef },
    Normalize { x: NodeRef },
    Swizzle { input: NodeRef, mask: String },
    Combine3 { x: NodeRef, y: NodeRef, z: NodeRef },

    // Master PBR output node
    StandardSurfaceOutput {
        #[serde(default)]
        base_color: Option<NodeRef>,
        #[serde(default)]
        metallic: Option<NodeRef>,
        #[serde(default)]
        roughness: Option<NodeRef>,
        #[serde(default)]
        normal: Option<NodeRef>,
        #[serde(default)]
        emissive: Option<NodeRef>,
        #[serde(default)]
        ao: Option<NodeRef>,
        #[serde(default)]
        alpha: Option<NodeRef>,
        #[serde(default)]
        clearcoat: Option<NodeRef>,
        #[serde(default)]
        iridescence: Option<NodeRef>,
    },
}

fn default_one() -> f32 {
    1.0
}
fn default_four() -> f32 {
    4.0
}
fn default_five() -> f32 {
    5.0
}
fn default_octaves() -> u32 {
    4
}
fn default_one2() -> [f32; 2] {
    [1.0, 1.0]
}
fn default_white3() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

/// One positioned node inside a `.matgraph.json` visual shader graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShaderNode {
    pub id: String,
    #[serde(default)]
    pub ui_pos: [f32; 2],
    #[serde(flatten)]
    pub kind: ShaderNodeKind,
}

/// Complete visual material node graph document (`.matgraph.json`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShaderGraph {
    pub name: String,
    pub nodes: Vec<ShaderNode>,
}

/// Compiled WGSL output plus metadata from [`ShaderGraph::compile_wgsl`].
#[derive(Clone, Debug)]
pub struct CompiledShaderGraph {
    pub name: String,
    pub wgsl: String,
    pub node_count: usize,
    pub uses_time: bool,
}

fn collect_deps(kind: &ShaderNodeKind, out: &mut Vec<String>) {
    let mut push = |r: &NodeRef| out.push(r.node.clone());
    match kind {
        ShaderNodeKind::FloatConst { .. }
        | ShaderNodeKind::Vec2Const { .. }
        | ShaderNodeKind::Vec3Const { .. }
        | ShaderNodeKind::Vec4Const { .. }
        | ShaderNodeKind::Uv
        | ShaderNodeKind::Uv1
        | ShaderNodeKind::WorldPos
        | ShaderNodeKind::WorldNormal
        | ShaderNodeKind::ViewDir
        | ShaderNodeKind::VertexColor
        | ShaderNodeKind::Time
        | ShaderNodeKind::InstanceParam { .. }
        | ShaderNodeKind::Triplanar { .. }
        | ShaderNodeKind::Fresnel { .. } => {}
        ShaderNodeKind::UvTransform { uv, .. } | ShaderNodeKind::Texture2D { uv, .. } => {
            if let Some(u) = uv {
                push(u);
            }
        }
        ShaderNodeKind::SimplexNoise { coords, .. }
        | ShaderNodeKind::WorleyNoise { coords, .. }
        | ShaderNodeKind::FbmNoise { coords, .. } => {
            if let Some(c) = coords {
                push(c);
            }
        }
        ShaderNodeKind::ColorRamp { t, .. } => push(t),
        ShaderNodeKind::NormalBlend { base, detail } => {
            push(base);
            push(detail);
        }
        ShaderNodeKind::Add { a, b }
        | ShaderNodeKind::Sub { a, b }
        | ShaderNodeKind::Mul { a, b }
        | ShaderNodeKind::Div { a, b }
        | ShaderNodeKind::Dot { a, b } => {
            push(a);
            push(b);
        }
        ShaderNodeKind::Pow { base, exp } => {
            push(base);
            push(exp);
        }
        ShaderNodeKind::Sin { x }
        | ShaderNodeKind::Cos { x }
        | ShaderNodeKind::Clamp { x, .. }
        | ShaderNodeKind::Smoothstep { x, .. }
        | ShaderNodeKind::Normalize { x } => push(x),
        ShaderNodeKind::Lerp { a, b, t } => {
            push(a);
            push(b);
            push(t);
        }
        ShaderNodeKind::Swizzle { input, .. } => push(input),
        ShaderNodeKind::Combine3 { x, y, z } => {
            push(x);
            push(y);
            push(z);
        }
        ShaderNodeKind::StandardSurfaceOutput {
            base_color,
            metallic,
            roughness,
            normal,
            emissive,
            ao,
            alpha,
            clearcoat,
            iridescence,
        } => {
            for r in [base_color, metallic, roughness, normal, emissive, ao, alpha, clearcoat, iridescence]
                .into_iter()
                .flatten()
            {
                push(r);
            }
        }
    }
}

fn promote(expr: &str, from: PortType, to: PortType) -> String {
    if from == to {
        return expr.to_owned();
    }
    match (from, to) {
        (PortType::Float, PortType::Vec2) => format!("vec2f({expr})"),
        (PortType::Float, PortType::Vec3) => format!("vec3f({expr})"),
        (PortType::Float, PortType::Vec4) => format!("vec4f({expr})"),
        (PortType::Vec2, PortType::Float) => format!("({expr}).x"),
        (PortType::Vec2, PortType::Vec3) => format!("vec3f({expr}, 0.0)"),
        (PortType::Vec2, PortType::Vec4) => format!("vec4f({expr}, 0.0, 1.0)"),
        (PortType::Vec3, PortType::Float) => format!("dot({expr}, vec3f(0.2126, 0.7152, 0.0722))"),
        (PortType::Vec3, PortType::Vec2) => format!("({expr}).xy"),
        (PortType::Vec3, PortType::Vec4) => format!("vec4f({expr}, 1.0)"),
        (PortType::Vec4, PortType::Float) => format!("({expr}).x"),
        (PortType::Vec4, PortType::Vec2) => format!("({expr}).xy"),
        (PortType::Vec4, PortType::Vec3) => format!("({expr}).xyz"),
        _ => expr.to_owned(),
    }
}

fn wider_type(a: PortType, b: PortType) -> PortType {
    match (a, b) {
        (PortType::Vec4, _) | (_, PortType::Vec4) => PortType::Vec4,
        (PortType::Vec3, _) | (_, PortType::Vec3) => PortType::Vec3,
        (PortType::Vec2, _) | (_, PortType::Vec2) => PortType::Vec2,
        _ => PortType::Float,
    }
}

fn swizzle_type(mask: &str) -> Result<PortType> {
    ensure!((1..=4).contains(&mask.len()), "invalid swizzle mask '{mask}'");
    ensure!(mask.chars().all(|c| matches!(c, 'x' | 'y' | 'z' | 'w' | 'r' | 'g' | 'b' | 'a')), "invalid swizzle '{mask}'");
    Ok(match mask.len() {
        1 => PortType::Float,
        2 => PortType::Vec2,
        3 => PortType::Vec3,
        _ => PortType::Vec4,
    })
}

fn sanitize_ident(id: &str, idx: usize) -> String {
    let clean: String = id.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    format!("n_{idx}_{clean}")
}

fn fmt_f32(v: f32) -> String {
    if v.is_finite() { format!("{v:?}") } else { "0.0".to_owned() }
}

const PRELUDE_WGSL: &str = r#"// Generated by Pen Visual Shader Node Graph Compiler.
struct GraphUniform {
  time_and_params: vec4f, // x = time, yzw = reserved
  sun_dir_and_amb: vec4f, // xyz = normalized sun direction, w = ambient scale
  sun_color: vec4f,       // rgb = sun radiance
  cam_pos: vec4f,         // xyz = camera world pos
  instance_params: vec4f, // per-instance parameters[0..4]
};

@group(0) @binding(0) var<uniform> sg_u: GraphUniform;

struct VsOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) world_pos: vec3f,
  @location(2) world_normal: vec3f,
  @location(3) vertex_color: vec4f,
  @location(4) uv1: vec2f,
};

@vertex fn vs_preview(@builtin(vertex_index) vid: u32) -> VsOut {
  var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let p = pos[vid];
  let uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  var out: VsOut;
  out.clip = vec4f(p, 0.0, 1.0);
  out.uv = uv;
  out.uv1 = uv;
  out.world_pos = vec3f((uv.x - 0.5) * 2.0, (0.5 - uv.y) * 2.0, 0.0);
  out.world_normal = normalize(vec3f(out.world_pos.x * 0.35, out.world_pos.y * 0.35, 1.0));
  out.vertex_color = vec4f(1.0, 1.0, 1.0, 1.0);
  return out;
}

fn sg_hash2(p: vec2f) -> vec2f {
  let q = vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3)));
  return fract(sin(q) * 43758.5453);
}

fn sg_value_noise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = fract(sin(dot(i + vec2f(0.0, 0.0), vec2f(127.1, 311.7))) * 43758.5453);
  let b = fract(sin(dot(i + vec2f(1.0, 0.0), vec2f(127.1, 311.7))) * 43758.5453);
  let c = fract(sin(dot(i + vec2f(0.0, 1.0), vec2f(127.1, 311.7))) * 43758.5453);
  let d = fract(sin(dot(i + vec2f(1.0, 1.0), vec2f(127.1, 311.7))) * 43758.5453);
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn sg_worley(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  var md = 1.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let cell = vec2f(f32(dx), f32(dy));
      let pt = sg_hash2(i + cell);
      let d = length(cell + pt - f);
      md = min(md, d);
    }
  }
  return clamp(md, 0.0, 1.0);
}

fn sg_fbm(p: vec2f, octaves: u32) -> f32 {
  var val = 0.0;
  var amp = 0.5;
  var freq = p;
  let oct = clamp(octaves, 1u, 8u);
  for (var i = 0u; i < oct; i++) {
    val += amp * sg_value_noise(freq);
    freq = freq * 2.02 + vec2f(17.1, 9.2);
    amp *= 0.5;
  }
  return val;
}

fn sg_triplanar(wp: vec3f, wn: vec3f, scale: f32, sharp: f32) -> vec3f {
  var w = pow(abs(wn), vec3f(max(sharp, 1.0)));
  w /= max(w.x + w.y + w.z, 1e-5);
  let cx = vec3f(sg_value_noise(wp.yz * scale));
  let cy = vec3f(sg_value_noise(wp.xz * scale));
  let cz = vec3f(sg_value_noise(wp.xy * scale));
  return cx * w.x + cy * w.y + cz * w.z;
}

fn sg_pbr_eval(
  base_color: vec3f,
  metallic: f32,
  roughness: f32,
  n: vec3f,
  v: vec3f,
  emissive: vec3f,
  ao: f32,
  clearcoat: f32,
  iridescence: f32,
) -> vec3f {
  let l = normalize(sg_u.sun_dir_and_amb.xyz);
  let h = normalize(v + l);
  let ndl = max(dot(n, l), 0.0);
  let ndv = max(dot(n, v), 1e-4);
  let ndh = max(dot(n, h), 0.0);
  let vdh = max(dot(v, h), 0.0);

  let r = clamp(roughness, 0.04, 1.0);
  let a = r * r;
  let a2 = a * a;
  let denom = ndh * ndh * (a2 - 1.0) + 1.0;
  let d = a2 / max(3.14159265 * denom * denom, 1e-5);
  let k = (r + 1.0) * (r + 1.0) * 0.125;
  let gv = ndv / (ndv * (1.0 - k) + k);
  let gl = ndl / (ndl * (1.0 - k) + k);
  let f0 = mix(vec3f(0.04), base_color, clamp(metallic, 0.0, 1.0));
  var f = f0 + (vec3f(1.0) - f0) * pow(1.0 - vdh, 5.0);
  if (iridescence > 0.0) {
    let phase = (1.0 - ndv) * 6.283185;
    let tint = vec3f(0.5 + 0.5 * cos(phase), 0.5 + 0.5 * cos(phase + 2.094), 0.5 + 0.5 * cos(phase + 4.188));
    f = mix(f, f * tint * 1.4, clamp(iridescence, 0.0, 1.0));
  }
  let spec = (d * gv * gl * f) / max(4.0 * ndv * ndl, 1e-4);
  let kd = (vec3f(1.0) - f) * (1.0 - clamp(metallic, 0.0, 1.0));
  var lit = (kd * base_color / 3.14159265 + spec) * sg_u.sun_color.rgb * ndl;
  if (clearcoat > 0.0) {
    let cc_a2 = 0.0025;
    let cc_den = ndh * ndh * (cc_a2 - 1.0) + 1.0;
    let cc_d = cc_a2 / max(3.14159265 * cc_den * cc_den, 1e-5);
    let cc_f = 0.04 + 0.96 * pow(1.0 - vdh, 5.0);
    lit += vec3f(cc_d * cc_f * clamp(clearcoat, 0.0, 1.0) * 0.25) * sg_u.sun_color.rgb * ndl;
  }
  let amb = base_color * max(ao, 0.0) * max(sg_u.sun_dir_and_amb.w, 0.0);
  return lit + amb + emissive;
}
"#;

/// Returns true when an asset shader key names a JSON visual material graph
/// (`.matgraph.json` or `.shader-graph.json`).
pub fn is_shader_graph_key(key: &str) -> bool {
    key.ends_with(".matgraph.json") || key.ends_with(".shader-graph.json")
}

const MATERIAL_HELPERS_WGSL: &str = r#"// Generated by Pen Visual Shader Node Graph Compiler (live material shader).
struct Scene {
  view: mat4x4f,
  projection: mat4x4f,
  ambient: vec4f,
  light_direction: vec4f,
  light_color: vec4f,
  light_view_projection: mat4x4f,
  shadow: vec4f,
  zenith: vec4f,
  horizon: vec4f,
  ground: vec4f,
  eye: vec4f,
  cascades: vec4f,
  cascade_view_projection: array<mat4x4f, 4>,
  cascade_texels: array<vec4f, 4>,
  shadow_params: vec4f,
};
struct Draw {
  model: mat4x4f,
  normal: mat4x4f,
  color: vec4f,
  settings: vec4f,
  parameters: vec4f,
  extra: array<vec4f, 32>,
  surface: array<vec4f, 5>,
  uv_transform: vec4f,
  maps: vec4f,
  atlas: vec4f,
  iridescence: vec4f,
};
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@group(1) @binding(1) var sg_base_tex: texture_2d<f32>;
@group(1) @binding(2) var sg_base_samp: sampler;

struct SgVsOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) world_pos: vec3f,
  @location(2) world_normal: vec3f,
  @location(3) vertex_color: vec4f,
  @location(4) uv1: vec2f,
  @location(5) view_z: f32,
};

@vertex fn vertex(
  @location(0) pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(12) uv1: vec2f,
  @location(13) color: vec4f,
) -> SgVsOut {
  let world = draw.model * vec4f(pos, 1.0);
  let view_pos = scene.view * world;
  var out: SgVsOut;
  out.clip = scene.projection * view_pos;
  out.uv = uv;
  out.world_pos = world.xyz;
  out.world_normal = (draw.normal * vec4f(normal, 0.0)).xyz;
  out.vertex_color = color;
  out.uv1 = uv1;
  out.view_z = view_pos.z;
  return out;
}

fn sg_hash2(p: vec2f) -> vec2f {
  let q = vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3)));
  return fract(sin(q) * 43758.5453);
}

fn sg_value_noise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = fract(sin(dot(i + vec2f(0.0, 0.0), vec2f(127.1, 311.7))) * 43758.5453);
  let b = fract(sin(dot(i + vec2f(1.0, 0.0), vec2f(127.1, 311.7))) * 43758.5453);
  let c = fract(sin(dot(i + vec2f(0.0, 1.0), vec2f(127.1, 311.7))) * 43758.5453);
  let d = fract(sin(dot(i + vec2f(1.0, 1.0), vec2f(127.1, 311.7))) * 43758.5453);
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn sg_worley(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  var md = 1.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let cell = vec2f(f32(dx), f32(dy));
      let pt = sg_hash2(i + cell);
      let d = length(cell + pt - f);
      md = min(md, d);
    }
  }
  return clamp(md, 0.0, 1.0);
}

fn sg_fbm(p: vec2f, octaves: u32) -> f32 {
  var val = 0.0;
  var amp = 0.5;
  var freq = p;
  let oct = clamp(octaves, 1u, 8u);
  for (var i = 0u; i < oct; i++) {
    val += amp * sg_value_noise(freq);
    freq = freq * 2.02 + vec2f(17.1, 9.2);
    amp *= 0.5;
  }
  return val;
}

fn sg_triplanar(wp: vec3f, wn: vec3f, scale: f32, sharp: f32) -> vec3f {
  var w = pow(abs(wn), vec3f(max(sharp, 1.0)));
  w /= max(w.x + w.y + w.z, 1e-5);
  let cx = vec3f(sg_value_noise(wp.yz * scale));
  let cy = vec3f(sg_value_noise(wp.xz * scale));
  let cz = vec3f(sg_value_noise(wp.xy * scale));
  return cx * w.x + cy * w.y + cz * w.z;
}
"#;

impl ShaderGraph {
    /// Compiles this visual material graph into standalone preview WGSL (`vs_preview` + `fs_main`),
    /// verifying acyclicity, valid node references, and typed port promotion.
    pub fn compile_wgsl(&self) -> Result<CompiledShaderGraph> {
        self.compile_internal(false)
    }

    /// Compiles this visual material graph into a full Pen mesh material shader (`vertex` + `fragment`)
    /// bound against Pen's live Scene/Draw/Lights groups (`standard_surface` + `standard_sun`).
    pub fn compile_material_wgsl(&self) -> Result<CompiledShaderGraph> {
        self.compile_internal(true)
    }

    fn compile_internal(&self, material_mode: bool) -> Result<CompiledShaderGraph> {
        ensure!(!self.nodes.is_empty(), "shader graph '{}' has no nodes", self.name);

        let mut by_id: HashMap<&str, (usize, &ShaderNode)> = HashMap::with_capacity(self.nodes.len());
        let mut output_node_id: Option<&str> = None;
        for (i, node) in self.nodes.iter().enumerate() {
            ensure!(!node.id.is_empty(), "shader node at index {i} has empty id");
            ensure!(by_id.insert(node.id.as_str(), (i, node)).is_none(), "duplicate shader node id '{}'", node.id);
            if matches!(node.kind, ShaderNodeKind::StandardSurfaceOutput { .. }) {
                ensure!(output_node_id.is_none(), "shader graph '{}' has multiple StandardSurfaceOutput nodes", self.name);
                output_node_id = Some(node.id.as_str());
            }
        }
        let root_id = output_node_id
            .with_context(|| format!("shader graph '{}' is missing a StandardSurfaceOutput node", self.name))?;

        // DFS topological sort starting from StandardSurfaceOutput (dead-code eliminates unused branches
        // and rejects any cycle cleanly!).
        let mut state: HashMap<&str, u8> = HashMap::new(); // 0=unvisited, 1=visiting, 2=done
        let mut order: Vec<&str> = Vec::new();
        fn visit<'a>(
            id: &'a str,
            by_id: &HashMap<&'a str, (usize, &'a ShaderNode)>,
            state: &mut HashMap<&'a str, u8>,
            order: &mut Vec<&'a str>,
        ) -> Result<()> {
            match state.get(id).copied().unwrap_or(0) {
                2 => return Ok(()),
                1 => bail!("cycle detected in shader graph at node '{id}'"),
                _ => {}
            }
            state.insert(id, 1);
            let (_, node) = by_id.get(id).with_context(|| format!("dangling shader node reference '{id}'"))?;
            let mut deps = Vec::new();
            collect_deps(&node.kind, &mut deps);
            for d in &deps {
                let d_ref = by_id
                    .get(d.as_str())
                    .map(|(_, n)| n.id.as_str())
                    .with_context(|| format!("node '{}' references unknown node '{d}'", node.id))?;
                visit(d_ref, by_id, state, order)?;
            }
            state.insert(id, 2);
            order.push(id);
            Ok(())
        }
        visit(root_id, &by_id, &mut state, &mut order)?;

        let mut vars: HashMap<&str, (String, PortType)> = HashMap::new();
        let mut body = String::new();
        let mut uses_time = false;
        let mut emitted = HashSet::new();

        let get_var = |vars: &HashMap<&str, (String, PortType)>, r: &NodeRef| -> Result<(String, PortType)> {
            vars.get(r.node.as_str())
                .cloned()
                .with_context(|| format!("unresolved node reference '{}'", r.node))
        };

        let mut out_exprs = (
            "vec3f(0.8, 0.8, 0.8)".to_owned(), // base_color
            "0.0".to_owned(),                  // metallic
            "0.5".to_owned(),                  // roughness
            "in_n".to_owned(),                 // normal
            "vec3f(0.0)".to_owned(),           // emissive
            "1.0".to_owned(),                  // ao
            "1.0".to_owned(),                  // alpha
            "0.0".to_owned(),                  // clearcoat
            "0.0".to_owned(),                  // iridescence
        );

        for (step, &id) in order.iter().enumerate() {
            if !emitted.insert(id) {
                continue;
            }
            let (_, node) = by_id[id];
            let vname = sanitize_ident(id, step);

            match &node.kind {
                ShaderNodeKind::FloatConst { value } => {
                    writeln!(body, "  let {vname}: f32 = {};", fmt_f32(*value))?;
                    vars.insert(id, (vname, PortType::Float));
                }
                ShaderNodeKind::Vec2Const { value } => {
                    writeln!(body, "  let {vname}: vec2f = vec2f({}, {});", fmt_f32(value[0]), fmt_f32(value[1]))?;
                    vars.insert(id, (vname, PortType::Vec2));
                }
                ShaderNodeKind::Vec3Const { value } => {
                    writeln!(
                        body,
                        "  let {vname}: vec3f = vec3f({}, {}, {});",
                        fmt_f32(value[0]),
                        fmt_f32(value[1]),
                        fmt_f32(value[2])
                    )?;
                    vars.insert(id, (vname, PortType::Vec3));
                }
                ShaderNodeKind::Vec4Const { value } => {
                    writeln!(
                        body,
                        "  let {vname}: vec4f = vec4f({}, {}, {}, {});",
                        fmt_f32(value[0]),
                        fmt_f32(value[1]),
                        fmt_f32(value[2]),
                        fmt_f32(value[3])
                    )?;
                    vars.insert(id, (vname, PortType::Vec4));
                }
                ShaderNodeKind::Uv => {
                    writeln!(body, "  let {vname}: vec2f = in_v.uv;")?;
                    vars.insert(id, (vname, PortType::Vec2));
                }
                ShaderNodeKind::Uv1 => {
                    writeln!(body, "  let {vname}: vec2f = in_v.uv1;")?;
                    vars.insert(id, (vname, PortType::Vec2));
                }
                ShaderNodeKind::WorldPos => {
                    writeln!(body, "  let {vname}: vec3f = in_v.world_pos;")?;
                    vars.insert(id, (vname, PortType::Vec3));
                }
                ShaderNodeKind::WorldNormal => {
                    writeln!(body, "  let {vname}: vec3f = in_n;")?;
                    vars.insert(id, (vname, PortType::Vec3));
                }
                ShaderNodeKind::ViewDir => {
                    writeln!(body, "  let {vname}: vec3f = in_view;")?;
                    vars.insert(id, (vname, PortType::Vec3));
                }
                ShaderNodeKind::VertexColor => {
                    writeln!(body, "  let {vname}: vec4f = in_v.vertex_color;")?;
                    vars.insert(id, (vname, PortType::Vec4));
                }
                ShaderNodeKind::Time => {
                    uses_time = true;
                    if material_mode {
                        writeln!(body, "  let {vname}: f32 = scene.shadow_params.z;")?;
                    } else {
                        writeln!(body, "  let {vname}: f32 = sg_u.time_and_params.x;")?;
                    }
                    vars.insert(id, (vname, PortType::Float));
                }
                ShaderNodeKind::InstanceParam { index } => {
                    let lane = match index % 4 {
                        0 => "x",
                        1 => "y",
                        2 => "z",
                        _ => "w",
                    };
                    if material_mode {
                        writeln!(body, "  let {vname}: f32 = draw.parameters.{lane};")?;
                    } else {
                        writeln!(body, "  let {vname}: f32 = sg_u.instance_params.{lane};")?;
                    }
                    vars.insert(id, (vname, PortType::Float));
                }
                ShaderNodeKind::UvTransform { uv, scale, offset, rotation_rad } => {
                    let base = if let Some(u) = uv {
                        let (e, t) = get_var(&vars, u)?;
                        promote(&e, t, PortType::Vec2)
                    } else {
                        "in_v.uv".to_owned()
                    };
                    let c = rotation_rad.cos();
                    let s = rotation_rad.sin();
                    writeln!(
                        body,
                        "  let {vname}_raw: vec2f = {base} * vec2f({}, {}) + vec2f({}, {});",
                        fmt_f32(scale[0]),
                        fmt_f32(scale[1]),
                        fmt_f32(offset[0]),
                        fmt_f32(offset[1])
                    )?;
                    writeln!(
                        body,
                        "  let {vname}: vec2f = vec2f({vname}_raw.x * {} - {vname}_raw.y * {}, {vname}_raw.x * {} + {vname}_raw.y * {});",
                        fmt_f32(c),
                        fmt_f32(s),
                        fmt_f32(s),
                        fmt_f32(c)
                    )?;
                    vars.insert(id, (vname, PortType::Vec2));
                }
                ShaderNodeKind::Texture2D { uv, fallback_color } => {
                    let uv_e = if let Some(u) = uv {
                        let (e, t) = get_var(&vars, u)?;
                        promote(&e, t, PortType::Vec2)
                    } else {
                        "in_v.uv".to_owned()
                    };
                    if material_mode {
                        writeln!(
                            body,
                            "  let {vname}_tex: vec4f = textureSample(sg_base_tex, sg_base_samp, {uv_e});"
                        )?;
                        writeln!(
                            body,
                            "  let {vname}: vec4f = select(vec4f({}, {}, {}, 1.0), {vname}_tex, draw.atlas.y > 0.5);",
                            fmt_f32(fallback_color[0]),
                            fmt_f32(fallback_color[1]),
                            fmt_f32(fallback_color[2])
                        )?;
                    } else {
                        // Procedural checker/tile preview fallback when standalone without external texture slot.
                        writeln!(
                            body,
                            "  let {vname}_chk: f32 = 0.85 + 0.15 * sg_value_noise({uv_e} * 8.0);"
                        )?;
                        writeln!(
                            body,
                            "  let {vname}: vec4f = vec4f(vec3f({}, {}, {}) * {vname}_chk, 1.0);",
                            fmt_f32(fallback_color[0]),
                            fmt_f32(fallback_color[1]),
                            fmt_f32(fallback_color[2])
                        )?;
                    }
                    vars.insert(id, (vname, PortType::Vec4));
                }
                ShaderNodeKind::Triplanar { scale, sharpness } => {
                    writeln!(
                        body,
                        "  let {vname}: vec3f = sg_triplanar(in_v.world_pos, in_n, {}, {});",
                        fmt_f32(*scale),
                        fmt_f32(*sharpness)
                    )?;
                    vars.insert(id, (vname, PortType::Vec3));
                }
                ShaderNodeKind::SimplexNoise { coords, scale } => {
                    let c = if let Some(cr) = coords {
                        let (e, t) = get_var(&vars, cr)?;
                        promote(&e, t, PortType::Vec2)
                    } else {
                        "in_v.uv".to_owned()
                    };
                    writeln!(body, "  let {vname}: f32 = sg_value_noise({c} * {});", fmt_f32(*scale))?;
                    vars.insert(id, (vname, PortType::Float));
                }
                ShaderNodeKind::WorleyNoise { coords, scale } => {
                    let c = if let Some(cr) = coords {
                        let (e, t) = get_var(&vars, cr)?;
                        promote(&e, t, PortType::Vec2)
                    } else {
                        "in_v.uv".to_owned()
                    };
                    writeln!(body, "  let {vname}: f32 = sg_worley({c} * {});", fmt_f32(*scale))?;
                    vars.insert(id, (vname, PortType::Float));
                }
                ShaderNodeKind::FbmNoise { coords, scale, octaves } => {
                    let c = if let Some(cr) = coords {
                        let (e, t) = get_var(&vars, cr)?;
                        promote(&e, t, PortType::Vec2)
                    } else {
                        "in_v.uv".to_owned()
                    };
                    writeln!(body, "  let {vname}: f32 = sg_fbm({c} * {}, {}u);", fmt_f32(*scale), (*octaves).clamp(1, 8))?;
                    vars.insert(id, (vname, PortType::Float));
                }
                ShaderNodeKind::Fresnel { power } => {
                    writeln!(
                        body,
                        "  let {vname}: f32 = pow(clamp(1.0 - max(dot(in_n, in_view), 0.0), 0.0, 1.0), {});",
                        fmt_f32(power.max(0.01))
                    )?;
                    vars.insert(id, (vname, PortType::Float));
                }
                ShaderNodeKind::ColorRamp { t, stops } => {
                    ensure!(!stops.is_empty(), "ColorRamp node '{id}' requires at least 1 stop");
                    let (te, tt) = get_var(&vars, t)?;
                    let tf = promote(&te, tt, PortType::Float);
                    let mut sorted = stops.clone();
                    sorted.sort_by(|a, b| a.t.total_cmp(&b.t));
                    let first = sorted[0].color;
                    writeln!(
                        body,
                        "  var {vname}: vec3f = vec3f({}, {}, {});",
                        fmt_f32(first[0]),
                        fmt_f32(first[1]),
                        fmt_f32(first[2])
                    )?;
                    for pair in sorted.windows(2) {
                        let (s0, s1) = (pair[0], pair[1]);
                        let span = (s1.t - s0.t).max(1e-5);
                        writeln!(
                            body,
                            "  {vname} = mix({vname}, vec3f({}, {}, {}), clamp(({tf} - {}) / {}, 0.0, 1.0));",
                            fmt_f32(s1.color[0]),
                            fmt_f32(s1.color[1]),
                            fmt_f32(s1.color[2]),
                            fmt_f32(s0.t),
                            fmt_f32(span)
                        )?;
                    }
                    vars.insert(id, (vname, PortType::Vec3));
                }
                ShaderNodeKind::NormalBlend { base, detail } => {
                    let (be, bt) = get_var(&vars, base)?;
                    let (de, dt) = get_var(&vars, detail)?;
                    let b3 = promote(&be, bt, PortType::Vec3);
                    let d3 = promote(&de, dt, PortType::Vec3);
                    writeln!(body, "  let {vname}: vec3f = normalize(vec3f(({b3}).xy + ({d3}).xy, ({b3}).z * ({d3}).z));")?;
                    vars.insert(id, (vname, PortType::Vec3));
                }
                ShaderNodeKind::Add { a, b }
                | ShaderNodeKind::Sub { a, b }
                | ShaderNodeKind::Mul { a, b }
                | ShaderNodeKind::Div { a, b } => {
                    let (ae, at) = get_var(&vars, a)?;
                    let (be, bt) = get_var(&vars, b)?;
                    let wt = wider_type(at, bt);
                    let pa = promote(&ae, at, wt);
                    let pb = promote(&be, bt, wt);
                    let op = match &node.kind {
                        ShaderNodeKind::Add { .. } => "+",
                        ShaderNodeKind::Sub { .. } => "-",
                        ShaderNodeKind::Mul { .. } => "*",
                        _ => "/",
                    };
                    writeln!(body, "  let {vname}: {} = ({pa}) {op} ({pb});", wt.wgsl())?;
                    vars.insert(id, (vname, wt));
                }
                ShaderNodeKind::Pow { base, exp } => {
                    let (ae, at) = get_var(&vars, base)?;
                    let (be, bt) = get_var(&vars, exp)?;
                    let wt = wider_type(at, bt);
                    let pa = promote(&ae, at, wt);
                    let pb = promote(&be, bt, wt);
                    let zero = promote("0.0001", PortType::Float, wt);
                    writeln!(body, "  let {vname}: {} = pow(max({pa}, {zero}), {pb});", wt.wgsl())?;
                    vars.insert(id, (vname, wt));
                }
                ShaderNodeKind::Sin { x } => {
                    let (xe, xt) = get_var(&vars, x)?;
                    writeln!(body, "  let {vname}: {} = sin({xe});", xt.wgsl())?;
                    vars.insert(id, (vname, xt));
                }
                ShaderNodeKind::Cos { x } => {
                    let (xe, xt) = get_var(&vars, x)?;
                    writeln!(body, "  let {vname}: {} = cos({xe});", xt.wgsl())?;
                    vars.insert(id, (vname, xt));
                }
                ShaderNodeKind::Clamp { x, min, max } => {
                    let (xe, xt) = get_var(&vars, x)?;
                    let lo = promote(&fmt_f32(*min), PortType::Float, xt);
                    let hi = promote(&fmt_f32(*max), PortType::Float, xt);
                    writeln!(body, "  let {vname}: {} = clamp({xe}, {lo}, {hi});", xt.wgsl())?;
                    vars.insert(id, (vname, xt));
                }
                ShaderNodeKind::Smoothstep { edge0, edge1, x } => {
                    let (xe, xt) = get_var(&vars, x)?;
                    let e0 = promote(&fmt_f32(*edge0), PortType::Float, xt);
                    let e1 = promote(&fmt_f32(*edge1), PortType::Float, xt);
                    writeln!(body, "  let {vname}: {} = smoothstep({e0}, {e1}, {xe});", xt.wgsl())?;
                    vars.insert(id, (vname, xt));
                }
                ShaderNodeKind::Lerp { a, b, t } => {
                    let (ae, at) = get_var(&vars, a)?;
                    let (be, bt) = get_var(&vars, b)?;
                    let (te, tt) = get_var(&vars, t)?;
                    let wt = wider_type(at, bt);
                    let pa = promote(&ae, at, wt);
                    let pb = promote(&be, bt, wt);
                    let pt = promote(&te, tt, PortType::Float);
                    writeln!(body, "  let {vname}: {} = mix({pa}, {pb}, clamp({pt}, 0.0, 1.0));", wt.wgsl())?;
                    vars.insert(id, (vname, wt));
                }
                ShaderNodeKind::Dot { a, b } => {
                    let (ae, at) = get_var(&vars, a)?;
                    let (be, bt) = get_var(&vars, b)?;
                    let wt = wider_type(at, bt);
                    let wt = if wt == PortType::Float { PortType::Vec3 } else { wt };
                    let pa = promote(&ae, at, wt);
                    let pb = promote(&be, bt, wt);
                    writeln!(body, "  let {vname}: f32 = dot({pa}, {pb});")?;
                    vars.insert(id, (vname, PortType::Float));
                }
                ShaderNodeKind::Normalize { x } => {
                    let (xe, xt) = get_var(&vars, x)?;
                    let wt = if xt == PortType::Float { PortType::Vec3 } else { xt };
                    let px = promote(&xe, xt, wt);
                    writeln!(body, "  let {vname}: {} = normalize({px});", wt.wgsl())?;
                    vars.insert(id, (vname, wt));
                }
                ShaderNodeKind::Swizzle { input, mask } => {
                    let (ie, it) = get_var(&vars, input)?;
                    let out_t = swizzle_type(mask)?;
                    let promoted = if it == PortType::Float { promote(&ie, it, PortType::Vec4) } else { ie };
                    writeln!(body, "  let {vname}: {} = ({promoted}).{mask};", out_t.wgsl())?;
                    vars.insert(id, (vname, out_t));
                }
                ShaderNodeKind::Combine3 { x, y, z } => {
                    let (xe, xt) = get_var(&vars, x)?;
                    let (ye, yt) = get_var(&vars, y)?;
                    let (ze, zt) = get_var(&vars, z)?;
                    writeln!(
                        body,
                        "  let {vname}: vec3f = vec3f({}, {}, {});",
                        promote(&xe, xt, PortType::Float),
                        promote(&ye, yt, PortType::Float),
                        promote(&ze, zt, PortType::Float)
                    )?;
                    vars.insert(id, (vname, PortType::Vec3));
                }
                ShaderNodeKind::StandardSurfaceOutput {
                    base_color,
                    metallic,
                    roughness,
                    normal,
                    emissive,
                    ao,
                    alpha,
                    clearcoat,
                    iridescence,
                } => {
                    if let Some(r) = base_color {
                        let (e, t) = get_var(&vars, r)?;
                        out_exprs.0 = promote(&e, t, PortType::Vec3);
                    }
                    if let Some(r) = metallic {
                        let (e, t) = get_var(&vars, r)?;
                        out_exprs.1 = promote(&e, t, PortType::Float);
                    }
                    if let Some(r) = roughness {
                        let (e, t) = get_var(&vars, r)?;
                        out_exprs.2 = promote(&e, t, PortType::Float);
                    }
                    if let Some(r) = normal {
                        let (e, t) = get_var(&vars, r)?;
                        out_exprs.3 = format!("normalize({})", promote(&e, t, PortType::Vec3));
                    }
                    if let Some(r) = emissive {
                        let (e, t) = get_var(&vars, r)?;
                        out_exprs.4 = promote(&e, t, PortType::Vec3);
                    }
                    if let Some(r) = ao {
                        let (e, t) = get_var(&vars, r)?;
                        out_exprs.5 = promote(&e, t, PortType::Float);
                    }
                    if let Some(r) = alpha {
                        let (e, t) = get_var(&vars, r)?;
                        out_exprs.6 = promote(&e, t, PortType::Float);
                    }
                    if let Some(r) = clearcoat {
                        let (e, t) = get_var(&vars, r)?;
                        out_exprs.7 = promote(&e, t, PortType::Float);
                    }
                    if let Some(r) = iridescence {
                        let (e, t) = get_var(&vars, r)?;
                        out_exprs.8 = promote(&e, t, PortType::Float);
                    }
                }
            }
        }

        if material_mode {
            let mut wgsl = String::with_capacity(
                MATERIAL_HELPERS_WGSL.len()
                    + crate::SHADOWS_WGSL.len()
                    + crate::LIGHTS_WGSL.len()
                    + crate::ENVIRONMENT_WGSL.len()
                    + crate::SURFACE_LOBES_WGSL.len()
                    + crate::SURFACE_WGSL.len()
                    + body.len()
                    + 1024,
            );
            wgsl.push_str(MATERIAL_HELPERS_WGSL);
            wgsl.push('\n');
            wgsl.push_str(crate::SHADOWS_WGSL);
            wgsl.push('\n');
            wgsl.push_str(crate::LIGHTS_WGSL);
            wgsl.push('\n');
            wgsl.push_str(crate::ENVIRONMENT_WGSL);
            wgsl.push('\n');
            wgsl.push_str(crate::SURFACE_LOBES_WGSL);
            wgsl.push('\n');
            wgsl.push_str(crate::SURFACE_WGSL);
            writeln!(wgsl, "\n@fragment fn fragment(in_v: SgVsOut) -> @location(0) vec4f {{")?;
            writeln!(wgsl, "  let in_n = normalize(in_v.world_normal);")?;
            writeln!(wgsl, "  let in_view = normalize(scene.eye.xyz - in_v.world_pos);")?;
            wgsl.push_str(&body);
            writeln!(wgsl, "  let sg_alpha = clamp(({}) * draw.color.a, 0.0, 1.0);", out_exprs.6)?;
            writeln!(wgsl, "  if (draw.settings.y > 0.0 && sg_alpha < draw.settings.y) {{ discard; }}")?;
            writeln!(wgsl, "  let sg_base = ({}) * draw.color.rgb;", out_exprs.0)?;
            writeln!(wgsl, "  let sg_emissive = {};", out_exprs.4)?;
            writeln!(wgsl, "  if (draw.settings.x <= 0.5) {{ return vec4f(sg_base + sg_emissive, sg_alpha); }}")?;
            writeln!(
                wgsl,
                "  let sg_normal = {};",
                out_exprs.3
            )?;
            writeln!(
                wgsl,
                "  var s = surface_default(in_v.world_pos, sg_normal, in_view, sg_base, clamp({}, 0.0, 1.0), clamp({}, 0.04, 1.0), in_v.clip, in_v.view_z);",
                out_exprs.1, out_exprs.2
            )?;
            writeln!(wgsl, "  s.occlusion = clamp({}, 0.0, 1.0);", out_exprs.5)?;
            writeln!(wgsl, "  let sg_coat = clamp({}, 0.0, 1.0);", out_exprs.7)?;
            writeln!(wgsl, "  if (sg_coat > 0.0) {{ s.lobes |= SURFACE_COAT; s.coat = sg_coat; s.coat_roughness = 0.08; }}")?;
            writeln!(wgsl, "  let sg_irid = clamp({}, 0.0, 1.0);", out_exprs.8)?;
            writeln!(wgsl, "  if (sg_irid > 0.0) {{ s.lobes |= SURFACE_IRIDESCENCE; s.iridescence = sg_irid; }}")?;
            writeln!(wgsl, "  let sg_vis = sun_shadow(in_v.world_pos, sg_normal, in_v.clip);")?;
            writeln!(
                wgsl,
                "  let lit = standard_surface(s) + standard_sun(s, scene.light_direction.xyz, scene.light_color.rgb, sg_vis) + sg_emissive;"
            )?;
            writeln!(wgsl, "  return vec4f(lit, sg_alpha);")?;
            writeln!(wgsl, "}}")?;
            return Ok(CompiledShaderGraph { name: self.name.clone(), wgsl, node_count: order.len(), uses_time });
        }

        let mut wgsl = String::with_capacity(PRELUDE_WGSL.len() + body.len() + 512);
        wgsl.push_str(PRELUDE_WGSL);
        writeln!(wgsl, "\n@fragment fn fs_main(in_v: VsOut) -> @location(0) vec4f {{")?;
        writeln!(wgsl, "  let in_n = normalize(in_v.world_normal);")?;
        writeln!(wgsl, "  let in_view = normalize(sg_u.cam_pos.xyz - in_v.world_pos);")?;
        wgsl.push_str(&body);
        writeln!(
            wgsl,
            "  let final_rgb = sg_pbr_eval({}, {}, {}, {}, in_view, {}, {}, {}, {});",
            out_exprs.0, out_exprs.1, out_exprs.2, out_exprs.3, out_exprs.4, out_exprs.5, out_exprs.7, out_exprs.8
        )?;
        writeln!(wgsl, "  return vec4f(final_rgb, clamp({}, 0.0, 1.0));", out_exprs.6)?;
        writeln!(wgsl, "}}")?;

        Ok(CompiledShaderGraph { name: self.name.clone(), wgsl, node_count: order.len(), uses_time })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GraphUniform {
        time_and_params: [f32; 4],
        sun_dir_and_amb: [f32; 4],
        sun_color: [f32; 4],
        cam_pos: [f32; 4],
        instance_params: [f32; 4],
    }

    #[test]
    fn gpu_shader_graph_compiles_procedural_pbr_dag_renders_and_rejects_cycles() {
        let graph = ShaderGraph {
            name: "iridescent_rock".into(),
            nodes: vec![
                ShaderNode { id: "uv".into(), ui_pos: [0.0, 0.0], kind: ShaderNodeKind::Uv },
                ShaderNode {
                    id: "fbm".into(),
                    ui_pos: [120.0, 0.0],
                    kind: ShaderNodeKind::FbmNoise { coords: Some(NodeRef::of("uv")), scale: 6.0, octaves: 4 },
                },
                ShaderNode {
                    id: "ramp".into(),
                    ui_pos: [240.0, 0.0],
                    kind: ShaderNodeKind::ColorRamp {
                        t: NodeRef::of("fbm"),
                        stops: vec![
                            RampStop { t: 0.2, color: [0.15, 0.35, 0.85] },
                            RampStop { t: 0.8, color: [0.95, 0.55, 0.15] },
                        ],
                    },
                },
                ShaderNode { id: "fresnel".into(), ui_pos: [120.0, 120.0], kind: ShaderNodeKind::Fresnel { power: 3.0 } },
                ShaderNode {
                    id: "glow_col".into(),
                    ui_pos: [240.0, 120.0],
                    kind: ShaderNodeKind::Vec3Const { value: [0.2, 0.6, 1.0] },
                },
                ShaderNode {
                    id: "emissive".into(),
                    ui_pos: [360.0, 120.0],
                    kind: ShaderNodeKind::Mul { a: NodeRef::of("glow_col"), b: NodeRef::of("fresnel") },
                },
                ShaderNode { id: "rough".into(), ui_pos: [240.0, 220.0], kind: ShaderNodeKind::FloatConst { value: 0.25 } },
                ShaderNode { id: "irid".into(), ui_pos: [240.0, 280.0], kind: ShaderNodeKind::FloatConst { value: 0.7 } },
                ShaderNode {
                    id: "out".into(),
                    ui_pos: [500.0, 100.0],
                    kind: ShaderNodeKind::StandardSurfaceOutput {
                        base_color: Some(NodeRef::of("ramp")),
                        metallic: None,
                        roughness: Some(NodeRef::of("rough")),
                        normal: None,
                        emissive: Some(NodeRef::of("emissive")),
                        ao: None,
                        alpha: None,
                        clearcoat: Some(NodeRef::of("fresnel")),
                        iridescence: Some(NodeRef::of("irid")),
                    },
                },
            ],
        };

        // JSON round-trip check.
        let json = serde_json::to_string_pretty(&graph).unwrap();
        let reloaded: ShaderGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(reloaded, graph);

        let compiled = reloaded.compile_wgsl().unwrap();
        assert_eq!(compiled.node_count, 9);

        // Compile & render live on headless GPU!
        let (device, queue) = crate::testing::device().expect("GPU adapter required");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader_graph_preview"),
            source: wgpu::ShaderSource::Wgsl(compiled.wgsl.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sg_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<GraphUniform>() as u64),
                },
                count: None,
            }],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sg_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sg_pipe"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_preview"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let u = GraphUniform {
            time_and_params: [1.0, 0.0, 0.0, 0.0],
            sun_dir_and_amb: [0.4, 0.6, 0.7, 0.25],
            sun_color: [2.5, 2.4, 2.2, 0.0],
            cam_pos: [0.0, 0.0, 3.0, 0.0],
            instance_params: [0.0; 4],
        };
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sg_u"),
            size: std::mem::size_of::<GraphUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        ubuf.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytemuck::bytes_of(&u));
        ubuf.unmap();
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sg_bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }],
        });

        let (w, h) = (32u32, 32u32);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sg_out"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let bpr = (w * 4).next_multiple_of(256);
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sg_rb"),
            size: u64::from(bpr * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sg_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        enc.copy_texture_to_buffer(
            tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &rb,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().unwrap();
        let mut min_r = 255u8;
        let mut max_r = 0u8;
        for y in 0..h {
            for x in 0..w {
                let r = data[(y * bpr + x * 4) as usize];
                min_r = min_r.min(r);
                max_r = max_r.max(r);
            }
        }
        assert!(max_r > 80 && max_r > min_r + 15, "procedural shader graph output must show lit variation: min={min_r}, max={max_r}");
        drop(data);
        rb.unmap();

        // Cycle rejection witness.
        let cyclic = ShaderGraph {
            name: "cyclic".into(),
            nodes: vec![
                ShaderNode {
                    id: "a".into(),
                    ui_pos: [0.0, 0.0],
                    kind: ShaderNodeKind::Add { a: NodeRef::of("b"), b: NodeRef::of("b") },
                },
                ShaderNode {
                    id: "b".into(),
                    ui_pos: [100.0, 0.0],
                    kind: ShaderNodeKind::Add { a: NodeRef::of("a"), b: NodeRef::of("a") },
                },
                ShaderNode {
                    id: "out".into(),
                    ui_pos: [200.0, 0.0],
                    kind: ShaderNodeKind::StandardSurfaceOutput {
                        base_color: Some(NodeRef::of("a")),
                        metallic: None,
                        roughness: None,
                        normal: None,
                        emissive: None,
                        ao: None,
                        alpha: None,
                        clearcoat: None,
                        iridescence: None,
                    },
                },
            ],
        };
        assert!(cyclic.compile_wgsl().is_err(), "cyclic shader graph must be rejected");
    }
}
