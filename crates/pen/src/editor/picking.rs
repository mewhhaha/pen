//! Ray-vs-node picking and gizmo handle hit testing for the interactive viewport editor.

use pen_scene::{NodeId, ResolvedScene};

use crate::editor::gizmo::GizmoAxis;

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn ray_sphere(origin: [f32; 3], dir: [f32; 3], center: [f32; 3], radius: f32) -> Option<f32> {
    let oc = sub3(origin, center);
    let b = dot3(oc, dir);
    let c = dot3(oc, oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    if t > 0.01 {
        Some(t)
    } else {
        let t2 = -b + disc.sqrt();
        (t2 > 0.01).then_some(t2)
    }
}

/// Picks the nearest visible node hit by ray `(origin, dir)`.
pub fn pick_node(scene: &ResolvedScene, origin: [f32; 3], dir: [f32; 3]) -> Option<NodeId> {
    let mut best: Option<(f32, NodeId)> = None;
    for node in &scene.nodes {
        if !node.visible {
            continue;
        }
        let m = &node.world_matrix;
        let center = [m[12], m[13], m[14]];
        let sx = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
        let sy = (m[4] * m[4] + m[5] * m[5] + m[6] * m[6]).sqrt();
        let sz = (m[8] * m[8] + m[9] * m[9] + m[10] * m[10]).sqrt();
        let radius = (sx.max(sy).max(sz) * 0.65).max(0.35);
        if let Some(t) = ray_sphere(origin, dir, center, radius)
            && best.is_none_or(|(bt, _)| t < bt)
        {
            best = Some((t, node.id));
        }
    }
    best.map(|(_, id)| id)
}

/// Closest parameter `s` along line `a_orig + s * a_dir` to ray `r_orig + t * r_dir`,
/// plus the squared separation distance between the two lines.
pub fn closest_axis_param(
    r_orig: [f32; 3],
    r_dir: [f32; 3],
    a_orig: [f32; 3],
    a_dir: [f32; 3],
) -> Option<(f32, f32)> {
    let w0 = sub3(r_orig, a_orig);
    let a = dot3(r_dir, r_dir);
    let b = dot3(r_dir, a_dir);
    let c = dot3(a_dir, a_dir);
    let d = dot3(r_dir, w0);
    let e = dot3(a_dir, w0);
    let denom = a * c - b * b;
    if denom.abs() < 1e-6 {
        return None;
    }
    let t_ray = (b * e - c * d) / denom;
    let s_axis = (a * e - b * d) / denom;
    if t_ray < 0.0 {
        return None;
    }
    let pr = [r_orig[0] + r_dir[0] * t_ray, r_orig[1] + r_dir[1] * t_ray, r_orig[2] + r_dir[2] * t_ray];
    let pa = [a_orig[0] + a_dir[0] * s_axis, a_orig[1] + a_dir[1] * s_axis, a_orig[2] + a_dir[2] * s_axis];
    let diff = sub3(pr, pa);
    Some((s_axis, dot3(diff, diff)))
}

/// Picks an interactive 3D gizmo handle arm (`X`, `Y`, or `Z`) near `pivot` with `arm_len`.
pub fn pick_gizmo_axis(
    origin: [f32; 3],
    dir: [f32; 3],
    pivot: [f32; 3],
    axes: [[f32; 3]; 3],
    arm_len: f32,
) -> Option<GizmoAxis> {
    let threshold = (arm_len * 0.16).max(0.12);
    let threshold2 = threshold * threshold;
    let mut best: Option<(f32, GizmoAxis)> = None;
    for (axis, axis_dir) in [(GizmoAxis::X, axes[0]), (GizmoAxis::Y, axes[1]), (GizmoAxis::Z, axes[2])] {
        if let Some((s, dist2)) = closest_axis_param(origin, dir, pivot, axis_dir)
            && s >= 0.0
            && s <= arm_len * 1.15
            && dist2 <= threshold2
            && best.is_none_or(|(bd, _)| dist2 < bd)
        {
            best = Some((dist2, axis));
        }
    }
    best.map(|(_, a)| a)
}
