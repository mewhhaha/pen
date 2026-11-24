//! Interactive 3D Translate / Rotate / Scale gizmo solver (World/Local space + snapping).

use pen_scene::{Transform, normalize_quat};

use crate::editor::picking::closest_axis_param;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

impl GizmoMode {
    pub fn label(self) -> &'static str {
        match self {
            GizmoMode::Translate => "MOVE",
            GizmoMode::Rotate => "ROTATE",
            GizmoMode::Scale => "SCALE",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoSpace {
    World,
    Local,
}

impl GizmoSpace {
    pub fn label(self) -> &'static str {
        match self {
            GizmoSpace::World => "WORLD",
            GizmoSpace::Local => "LOCAL",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
}

/// Active gizmo drag session tracking initial transform & grab anchor.
#[derive(Clone, Copy, Debug)]
pub struct GizmoDrag {
    pub axis: GizmoAxis,
    pub axis_dir: [f32; 3],
    pub pivot: [f32; 3],
    pub start_param: f32,
    pub initial: Transform,
}

/// Computes the 3 orthonormal gizmo axis vectors for `transform` in `space`.
pub fn gizmo_axes(transform: &Transform, space: GizmoSpace) -> [[f32; 3]; 3] {
    match space {
        GizmoSpace::World => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        GizmoSpace::Local => {
            let m = transform.to_matrix();
            let norm = |v: [f32; 3]| {
                let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                if l > 1e-6 { [v[0] / l, v[1] / l, v[2] / l] } else { [1.0, 0.0, 0.0] }
            };
            [norm([m[0], m[1], m[2]]), norm([m[4], m[5], m[6]]), norm([m[8], m[9], m[10]])]
        }
    }
}

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    normalize_quat([
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ])
}

fn quat_axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let half = angle * 0.5;
    let s = half.sin();
    normalize_quat([axis[0] * s, axis[1] * s, axis[2] * s, half.cos()])
}

impl GizmoDrag {
    pub fn begin(
        axis: GizmoAxis,
        axis_dir: [f32; 3],
        pivot: [f32; 3],
        ray_orig: [f32; 3],
        ray_dir: [f32; 3],
        initial: Transform,
    ) -> Option<Self> {
        let (start_param, _) = closest_axis_param(ray_orig, ray_dir, pivot, axis_dir)?;
        Some(Self { axis, axis_dir, pivot, start_param, initial })
    }

    /// Evaluates the updated transform along this drag for cursor ray `(ray_orig, ray_dir)`.
    pub fn evaluate(&self, mode: GizmoMode, ray_orig: [f32; 3], ray_dir: [f32; 3], snap: bool) -> Transform {
        let Some((curr_param, _)) = closest_axis_param(ray_orig, ray_dir, self.pivot, self.axis_dir) else {
            return self.initial;
        };
        let raw_delta = curr_param - self.start_param;
        let mut out = self.initial;
        match mode {
            GizmoMode::Translate => {
                let d = if snap { (raw_delta * 2.0).round() * 0.5 } else { raw_delta };
                for k in 0..3 {
                    out.position[k] = self.initial.position[k] + self.axis_dir[k] * d;
                }
            }
            GizmoMode::Rotate => {
                let mut angle = raw_delta * 0.9;
                if snap {
                    let step = 15.0f32.to_radians();
                    angle = (angle / step).round() * step;
                }
                let q_delta = quat_axis_angle(self.axis_dir, angle);
                out.rotation = quat_mul(q_delta, self.initial.rotation);
            }
            GizmoMode::Scale => {
                let mut delta = raw_delta * 0.5;
                if snap {
                    delta = (delta * 4.0).round() * 0.25;
                }
                let idx = match self.axis {
                    GizmoAxis::X => 0,
                    GizmoAxis::Y => 1,
                    GizmoAxis::Z => 2,
                };
                out.scale[idx] = (self.initial.scale[idx] + delta).max(0.05);
            }
        }
        out
    }
}
