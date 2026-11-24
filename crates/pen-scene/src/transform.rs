//! Local TRS transforms and column-major 4x4 matrix math for scene hierarchies.

use serde::{Deserialize, Serialize};

pub const IDENTITY_MAT4: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

/// Local node transform: translation, unit quaternion `(x, y, z, w)`, and non-uniform scale.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    #[serde(default)]
    pub position: [f32; 3],
    #[serde(default = "default_quat")]
    pub rotation: [f32; 4],
    #[serde(default = "default_scale")]
    pub scale: [f32; 3],
}

fn default_quat() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

fn default_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

impl Default for Transform {
    fn default() -> Self {
        Self { position: [0.0, 0.0, 0.0], rotation: default_quat(), scale: default_scale() }
    }
}

/// Sparse override applied to a prefab node's transform.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TransformOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation: Option<[f32; 4]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<[f32; 3]>,
}

impl Transform {
    pub fn from_position(position: [f32; 3]) -> Self {
        Self { position, ..Default::default() }
    }

    pub fn apply_override(&mut self, ov: &TransformOverride) {
        if let Some(p) = ov.position {
            self.position = p;
        }
        if let Some(r) = ov.rotation {
            self.rotation = normalize_quat(r);
        }
        if let Some(s) = ov.scale {
            self.scale = s;
        }
    }

    /// Converts TRS into a column-major 4x4 affine matrix (`T * R * S`).
    pub fn to_matrix(&self) -> [f32; 16] {
        let [qx, qy, qz, qw] = normalize_quat(self.rotation);
        let [sx, sy, sz] = self.scale;
        let xx = qx * qx;
        let yy = qy * qy;
        let zz = qz * qz;
        let xy = qx * qy;
        let xz = qx * qz;
        let yz = qy * qz;
        let wx = qw * qx;
        let wy = qw * qy;
        let wz = qw * qz;

        [
            (1.0 - 2.0 * (yy + zz)) * sx,
            (2.0 * (xy + wz)) * sx,
            (2.0 * (xz - wy)) * sx,
            0.0,
            (2.0 * (xy - wz)) * sy,
            (1.0 - 2.0 * (xx + zz)) * sy,
            (2.0 * (yz + wx)) * sy,
            0.0,
            (2.0 * (xz + wy)) * sz,
            (2.0 * (yz - wx)) * sz,
            (1.0 - 2.0 * (xx + yy)) * sz,
            0.0,
            self.position[0],
            self.position[1],
            self.position[2],
            1.0,
        ]
    }

    /// Forward direction (`-Z` in right-handed OpenGL/glTF/WebGPU coordinates).
    pub fn forward(&self) -> [f32; 3] {
        let m = self.to_matrix();
        normalize3([-m[8], -m[9], -m[10]])
    }
}

pub fn normalize_quat(q: [f32; 4]) -> [f32; 4] {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len > 1e-6 && len.is_finite() {
        [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

pub fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-6 && len.is_finite() {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, -1.0]
    }
}

/// Column-major 4x4 matrix multiplication `a * b`.
pub fn mul_mat4(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = a[row] * b[col * 4]
                + a[4 + row] * b[col * 4 + 1]
                + a[8 + row] * b[col * 4 + 2]
                + a[12 + row] * b[col * 4 + 3];
        }
    }
    out
}

/// Decomposes a column-major affine matrix (with positive scale) back into TRS.
pub fn mat4_to_trs(m: &[f32; 16]) -> Transform {
    let position = [m[12], m[13], m[14]];
    let sx = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
    let sy = (m[4] * m[4] + m[5] * m[5] + m[6] * m[6]).sqrt();
    let sz = (m[8] * m[8] + m[9] * m[9] + m[10] * m[10]).sqrt();
    let inv_sx = if sx > 1e-6 { 1.0 / sx } else { 0.0 };
    let inv_sy = if sy > 1e-6 { 1.0 / sy } else { 0.0 };
    let inv_sz = if sz > 1e-6 { 1.0 / sz } else { 0.0 };

    let m00 = m[0] * inv_sx;
    let m10 = m[1] * inv_sx;
    let m20 = m[2] * inv_sx;
    let m01 = m[4] * inv_sy;
    let m11 = m[5] * inv_sy;
    let m21 = m[6] * inv_sy;
    let m02 = m[8] * inv_sz;
    let m12 = m[9] * inv_sz;
    let m22 = m[10] * inv_sz;

    let trace = m00 + m11 + m22;
    let rotation = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s]
    };

    Transform { position, rotation: normalize_quat(rotation), scale: [sx, sy, sz] }
}

/// Builds a right-handed look-at view matrix.
pub fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let f = normalize3([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    let back = [-f[0], -f[1], -f[2]];
    let right = normalize3([
        up[1] * back[2] - up[2] * back[1],
        up[2] * back[0] - up[0] * back[2],
        up[0] * back[1] - up[1] * back[0],
    ]);
    let u = [
        back[1] * right[2] - back[2] * right[1],
        back[2] * right[0] - back[0] * right[2],
        back[0] * right[1] - back[1] * right[0],
    ];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    [
        right[0], u[0], back[0], 0.0,
        right[1], u[1], back[1], 0.0,
        right[2], u[2], back[2], 0.0,
        -dot(right, eye), -dot(u, eye), -dot(back, eye), 1.0,
    ]
}

/// Builds a WebGPU 0..1 depth perspective projection matrix.
pub fn perspective(fov_deg: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let fov_rad = fov_deg.clamp(1.0, 175.0).to_radians();
    let f = 1.0 / (0.5 * fov_rad).tan();
    let a = aspect.max(0.05);
    let n = near.max(1e-4);
    let fa = far.max(n + 0.01);
    [
        f / a, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, fa / (n - fa), -1.0,
        0.0, 0.0, n * fa / (n - fa), 0.0,
    ]
}

/// Converts a camera world transform matrix into a view matrix (`inverse` of rigid camera pose).
pub fn camera_world_to_view(world: &[f32; 16]) -> [f32; 16] {
    let eye = [world[12], world[13], world[14]];
    let forward = normalize3([-world[8], -world[9], -world[10]]);
    let up = normalize3([world[4], world[5], world[6]]);
    let target = [eye[0] + forward[0], eye[1] + forward[1], eye[2] + forward[2]];
    look_at(eye, target, up)
}
