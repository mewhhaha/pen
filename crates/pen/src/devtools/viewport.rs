//! Pointer inspection and renderer scenes for the built-in devtools viewport.

use pen_protocol::{Matrix, RenderFrame};
use pen_render::{
    DepthMode, JfaOutlineScene, JfaOutlineTarget, LineBatch, LineLayer, LineScene, LineStyle, OutlineStyle,
    RenderStats, Renderer, Scene2d, SdfShape, detect_depth_mode,
};

const GOLD: [f32; 4] = [1.6, 1.25, 0.15, 1.0];

/// Live inspection state. Draw indices refer to the frame supplied by the caller.
pub struct ViewportOverlay {
    pub enabled: bool,
    pub show_outlines: bool,
    pub show_gizmos: bool,
    pub show_ui_bounds: bool,
    pub show_hud: bool,
    pub hovered_draw: Option<usize>,
    pub selected_draw: Option<usize>,
    active_on_renderer: bool,
}

impl Default for ViewportOverlay {
    fn default() -> Self {
        Self {
            enabled: false,
            show_outlines: true,
            show_gizmos: true,
            show_ui_bounds: true,
            show_hud: true,
            hovered_draw: None,
            selected_draw: None,
            active_on_renderer: false,
        }
    }
}

impl ViewportOverlay {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn toggle(&mut self) -> bool {
        self.enabled = !self.enabled;
        self.enabled
    }

    pub fn apply_command(&mut self, cmd: &str) -> Option<String> {
        let mut words = cmd.split_ascii_whitespace();
        if words.next()? != "/viewport" {
            return None;
        }
        let action = words.next();
        if words.next().is_none() {
            match action {
                Some("on") => {
                    self.enabled = true;
                    return Some("Viewport overlay on".into());
                }
                Some("off") => {
                    self.enabled = false;
                    return Some("Viewport overlay off".into());
                }
                Some("toggle") => return Some(status("Viewport overlay", self.toggle())),
                Some("outlines") => {
                    self.show_outlines = !self.show_outlines;
                    return Some(status("Viewport outlines", self.show_outlines));
                }
                Some("gizmos") => {
                    self.show_gizmos = !self.show_gizmos;
                    return Some(status("Viewport gizmos", self.show_gizmos));
                }
                Some("ui") => {
                    self.show_ui_bounds = !self.show_ui_bounds;
                    return Some(status("Viewport UI bounds", self.show_ui_bounds));
                }
                Some("hud") => {
                    self.show_hud = !self.show_hud;
                    return Some(status("Viewport HUD", self.show_hud));
                }
                Some("clear") => {
                    self.hovered_draw = None;
                    self.selected_draw = None;
                    return Some("Viewport selection cleared".into());
                }
                _ => {}
            }
        }
        Some("Usage: /viewport on|off|toggle|outlines|gizmos|ui|hud|clear".into())
    }

    pub fn update_pointer(&mut self, frame: &RenderFrame, pointer_px: [f32; 2], viewport: [f32; 2]) {
        self.hovered_draw = pick_draw(frame, pointer_px, viewport);
    }

    pub fn click_select(&mut self, frame: &RenderFrame, pointer_px: [f32; 2], viewport: [f32; 2]) -> Option<usize> {
        self.update_pointer(frame, pointer_px, viewport);
        self.selected_draw = self.hovered_draw;
        self.selected_draw
    }

    pub fn build_jfa_outlines(&self, frame: &RenderFrame) -> Option<JfaOutlineScene> {
        if !self.enabled || !self.show_outlines {
            return None;
        }
        let targets: Vec<_> =
            [self.selected_draw, self.hovered_draw.filter(|hover| Some(*hover) != self.selected_draw)]
                .into_iter()
                .enumerate()
                .filter_map(|(style_index, draw)| {
                    draw.filter(|&draw_index| draw_index < frame.draws.len())
                        .map(|draw_index| JfaOutlineTarget { draw_index, style_index })
                })
                .collect();
        if targets.is_empty() {
            return None;
        }

        Some(JfaOutlineScene {
            targets,
            styles: vec![
                OutlineStyle {
                    color: GOLD,
                    occluded_color: [0.2, 0.65, 0.85, 0.7],
                    width_px: 4.0,
                    feather_px: 1.0,
                    fill_alpha: 0.0,
                },
                OutlineStyle {
                    color: [0.25, 0.9, 1.4, 0.95],
                    occluded_color: [0.15, 0.45, 0.65, 0.5],
                    width_px: 2.5,
                    feather_px: 1.0,
                    fill_alpha: 0.0,
                },
            ],
            ..Default::default()
        })
    }

    pub fn build_line_scene(&self, frame: &RenderFrame, viewport: [f32; 2]) -> Option<LineScene> {
        if !self.enabled {
            return None;
        }

        let mut scene = LineScene::default();
        if self.show_gizmos {
            let style = LineStyle { width_px: 1.5, layer: LineLayer::XRay, ..Default::default() };
            scene.batch.axes([0.0; 3], 1.0, style);
            if let Some(draw) = self.selected_draw.and_then(|index| frame.draws.get(index)) {
                selected_gizmo(&mut scene.batch, &draw.transform, LineStyle { width_px: 2.5, ..style });
            }
            light_gizmos(&mut scene.batch, frame, style);
        }
        if self.show_ui_bounds && valid_viewport(viewport) {
            ui_bounds(&mut scene.batch, frame, viewport);
        }
        (!scene.batch.is_empty()).then_some(scene)
    }

    pub fn build_sdf_hud(&self, frame: &RenderFrame, stats: &RenderStats, viewport: [f32; 2]) -> Option<Scene2d> {
        if !self.enabled || !self.show_hud || !valid_viewport(viewport) {
            return None;
        }

        let [width, height] = viewport;
        let scale = (width / 104.0).min(height / 88.0).min(1.0);
        let margin = 12.0 * scale;
        let card_width = (width - 2.0 * margin).min(216.0 * scale);
        let card_height = 64.0 * scale;
        let card_x = width - margin - card_width;
        let padding = 12.0 * scale;
        let x = card_x + padding;
        let inner_width = card_width - 2.0 * padding;
        let mut shapes = vec![SdfShape {
            rect: [card_x, margin, card_width, card_height],
            corner_radii: [10.0; 4],
            fill_color: [0.024, 0.036, 0.060, 0.88],
            border_color: [0.24, 0.37, 0.51, 0.55],
            border_width_px: 1.5,
            shadow_color: [0.0, 0.0, 0.0, 0.55],
            shadow_offset: [0.0, 3.0],
            shadow_blur_px: 8.0,
            ..Default::default()
        }];

        let track_color = [0.13, 0.18, 0.25, 0.85];
        let load = (stats.draw_calls as f32 / frame.draws.len().max(1) as f32).min(1.0);
        let load_rect = [x, margin + padding, inner_width, 6.0 * scale];
        shapes.push(pill(load_rect, track_color));
        load_gradient(&mut shapes, load_rect, load);

        let selection_color = if self.selected_draw.is_some_and(|index| index < frame.draws.len()) {
            GOLD
        } else {
            [0.23, 0.29, 0.39, 0.85]
        };
        shapes.push(pill([x, margin + 27.0 * scale, 28.0 * scale, 11.0 * scale], selection_color));

        let ui_rect = [x, margin + 46.0 * scale, inner_width, 5.0 * scale];
        shapes.push(pill(ui_rect, track_color));
        if !frame.ui.is_empty() {
            let ui_load = frame.ui.len().min(64) as f32 / 64.0;
            shapes.push(pill([x, ui_rect[1], inner_width * ui_load, ui_rect[3]], [0.24, 0.70, 0.92, 0.9]));
        }

        Some(Scene2d { sdf_overlay_shapes: shapes, ..Default::default() })
    }

    /// Apply on the same renderer each frame using dedicated devtools slots so
    /// game-owned scenes and wire extensions stay intact.
    pub fn apply_to_renderer(
        &mut self,
        renderer: &mut Renderer,
        frame: &RenderFrame,
        stats: &RenderStats,
        viewport: [f32; 2],
    ) {
        if self.enabled {
            self.active_on_renderer = true;
            if let Some(scene) = self.build_jfa_outlines(frame) {
                renderer.set_devtools_jfa_outlines(scene);
            } else {
                renderer.clear_devtools_jfa_outlines();
            }
            if let Some(scene) = self.build_line_scene(frame, viewport) {
                renderer.set_devtools_line_scene(scene);
            } else {
                renderer.clear_devtools_line_scene();
            }
            if let Some(scene) = self.build_sdf_hud(frame, stats, viewport) {
                renderer.set_devtools_sdf_overlay(scene.sdf_overlay_shapes);
            } else {
                renderer.clear_devtools_sdf_overlay();
            }
        } else if std::mem::take(&mut self.active_on_renderer) {
            renderer.clear_devtools_jfa_outlines();
            renderer.clear_devtools_line_scene();
            renderer.clear_devtools_sdf_overlay();
        }
    }

    /// Temporarily removes devtools overlay scenes from `renderer` (e.g. for clean
    /// captures) without disabling the overlay or disturbing game-owned scenes.
    pub fn suspend_on_renderer(&mut self, renderer: &mut Renderer) {
        if std::mem::take(&mut self.active_on_renderer) {
            renderer.clear_devtools_jfa_outlines();
            renderer.clear_devtools_line_scene();
            renderer.clear_devtools_sdf_overlay();
        }
    }
}

fn status(label: &str, shown: bool) -> String {
    format!("{label} {}", if shown { "on" } else { "off" })
}

fn valid_viewport(viewport: [f32; 2]) -> bool {
    viewport.into_iter().all(|extent| extent.is_finite() && extent > 0.0)
}

/// Pick the nearest positive intersection with a draw's local unit cube
/// `[-0.5, 0.5]^3` within the camera's near-to-far clip interval.
pub fn pick_draw(frame: &RenderFrame, pointer_px: [f32; 2], viewport: [f32; 2]) -> Option<usize> {
    if frame.draws.is_empty()
        || !valid_viewport(viewport)
        || !pointer_px.into_iter().zip(viewport).all(|(p, extent)| (0.0..extent).contains(&p))
    {
        return None;
    }

    let (origin, direction, max_t) = pointer_ray(frame, pointer_px, viewport)?;
    frame
        .draws
        .iter()
        .enumerate()
        .filter_map(|(index, draw)| draw_hit(&draw.transform, origin, direction, max_t).map(|t| (index, t)))
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(index, _)| index)
}

fn pointer_ray(
    frame: &RenderFrame,
    [x, y]: [f32; 2],
    [width, height]: [f32; 2],
) -> Option<([f64; 3], [f64; 3], Option<f64>)> {
    let projection = frame.projection;
    let view_projection = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..4).map(|k| f64::from(projection[k * 4 + row]) * f64::from(frame.view[column * 4 + k])).sum()
        })
    });
    let inverse = invert(view_projection)?;
    let [near_z, far_z] = match detect_depth_mode(&projection) {
        Some(DepthMode::ReversedZ) => [1.0, 0.0],
        // Orthographic depth increases with view-space Z in right-handed Reversed-Z.
        None if projection[11] == 0.0 && projection[15] != 0.0 && projection[10] / projection[15] > 0.0 => [1.0, 0.0],
        _ => [0.0, 1.0],
    };
    let x = 2.0 * f64::from(x) / f64::from(width) - 1.0;
    let y = 1.0 - 2.0 * f64::from(y) / f64::from(height);
    let unproject = |z| inverse.map(|row| dot(row, [x, y, z, 1.0]));
    let near = unproject(near_z);
    let far = unproject(far_z);
    if near[3] == 0.0 || !near.into_iter().chain(far).all(f64::is_finite) {
        return None;
    }

    let origin = std::array::from_fn(|i| near[i] / near[3]);
    let (delta, finite_far): ([f64; 3], bool) = if far[3] == 0.0 {
        // The infinite far plane unprojects to a homogeneous direction.
        (std::array::from_fn(|i| far[i] * near[3].signum()), false)
    } else {
        (std::array::from_fn(|i| far[i] / far[3] - origin[i]), true)
    };
    let length = delta[0].hypot(delta[1]).hypot(delta[2]);
    if length == 0.0 || !length.is_finite() || !origin.into_iter().all(f64::is_finite) {
        return None;
    }
    Some((origin, delta.map(|lane| lane / length), finite_far.then_some(length)))
}

fn invert(matrix: [[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    if !matrix.iter().flatten().all(|value| value.is_finite()) {
        return None;
    }
    let mut rows: [[f64; 8]; 4] = std::array::from_fn(|row| {
        std::array::from_fn(|column| if column < 4 { matrix[row][column] } else { f64::from(row + 4 == column) })
    });
    for column in 0..4 {
        let pivot = (column..4).max_by(|&a, &b| rows[a][column].abs().total_cmp(&rows[b][column].abs()))?;
        rows.swap(column, pivot);
        let divisor = rows[column][column];
        if divisor == 0.0 || !divisor.is_finite() {
            return None;
        }
        let pivot_row = rows[column].map(|value| value / divisor);
        rows[column] = pivot_row;
        for (index, row) in rows.iter_mut().enumerate() {
            if index != column {
                let factor = row[column];
                for (value, pivot_value) in row.iter_mut().zip(pivot_row) {
                    *value -= factor * pivot_value;
                }
            }
        }
    }
    Some(rows.map(|row| [row[4], row[5], row[6], row[7]]))
}

fn dot<const N: usize>(a: [f64; N], b: [f64; N]) -> f64 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}

fn draw_hit(transform: &Matrix, origin: [f64; 3], direction: [f64; 3], max_t: Option<f64>) -> Option<f64> {
    if !transform.iter().all(|lane| lane.is_finite()) {
        return None;
    }
    // Conservative bounding-sphere reject for standard affine transforms before 4x4 inversion:
    // half the sum of column norms bounds every corner of `[-0.5, 0.5]^3` even under shear.
    if transform[3] == 0.0
        && transform[7] == 0.0
        && transform[11] == 0.0
        && transform[15] == 1.0
        && !sphere_overlaps_ray(transform, origin, direction, max_t)
    {
        return None;
    }
    let matrix = std::array::from_fn(|row| std::array::from_fn(|column| f64::from(transform[column * 4 + row])));
    if let Some(inverse) = invert(matrix) {
        let local_origin: [f64; 3] =
            std::array::from_fn(|row| dot(inverse[row], [origin[0], origin[1], origin[2], 1.0]));
        let local_direction: [f64; 3] =
            std::array::from_fn(|row| dot(inverse[row], [direction[0], direction[1], direction[2], 0.0]));
        if local_origin.into_iter().chain(local_direction).all(f64::is_finite) {
            return local_unit_box_hit(local_origin, local_direction, max_t);
        }
    }
    sphere_hit(transform, origin, direction, max_t)
}

fn sphere_overlaps_ray(transform: &Matrix, origin: [f64; 3], direction: [f64; 3], max_t: Option<f64>) -> bool {
    let transform = transform.map(f64::from);
    let (columns, _) = transform[..12].as_chunks::<4>();
    let radius = (columns.iter().map(|&[x, y, z, _]| x.hypot(y).hypot(z)).sum::<f64>() * 0.5).max(0.25) * (1.0 + 1e-6);
    let offset = std::array::from_fn(|axis| transform[12 + axis] - origin[axis]);
    let along = dot(offset, direction);
    let perpendicular: [f64; 3] = std::array::from_fn(|axis| {
        let b = (axis + 1) % 3;
        let c = (axis + 2) % 3;
        offset[b] * direction[c] - offset[c] * direction[b]
    });
    let discriminant = radius * radius - dot(perpendicular, perpendicular);
    if discriminant < 0.0 {
        return false;
    }
    let half_chord = discriminant.sqrt();
    if along + half_chord <= 0.0 {
        return false;
    }
    max_t.is_none_or(|far| along - half_chord <= far)
}

fn local_unit_box_hit(origin: [f64; 3], direction: [f64; 3], max_t: Option<f64>) -> Option<f64> {
    let mut t_min = 0.0_f64;
    let mut t_max = max_t.unwrap_or(f64::INFINITY);
    for axis in 0..3 {
        let dir = direction[axis];
        let orig = origin[axis];
        if dir == 0.0 {
            if !(-0.5..=0.5).contains(&orig) {
                return None;
            }
        } else {
            let inv = 1.0 / dir;
            let t0 = (-0.5 - orig) * inv;
            let t1 = (0.5 - orig) * inv;
            let (near, far) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
            t_min = t_min.max(near);
            t_max = t_max.min(far);
            if t_min > t_max {
                return None;
            }
        }
    }
    let hit = if t_min > 0.0 { t_min } else { t_max };
    (hit.is_finite() && hit > 0.0 && max_t.is_none_or(|far| hit <= far)).then_some(hit)
}

fn sphere_hit(transform: &Matrix, origin: [f64; 3], direction: [f64; 3], max_t: Option<f64>) -> Option<f64> {
    if !transform.iter().all(|lane| lane.is_finite()) {
        return None;
    }
    let transform = transform.map(f64::from);
    let (columns, _) = transform[..12].as_chunks::<4>();
    // Half the sum of column norms bounds every corner of `[-0.5, 0.5]^3`, even under combined shear.
    let radius = (columns.iter().map(|&[x, y, z, _]| x.hypot(y).hypot(z)).sum::<f64>() * 0.5).max(0.25);
    let offset = std::array::from_fn(|axis| transform[12 + axis] - origin[axis]);
    let along = dot(offset, direction);
    // |offset × direction|² avoids subtracting large nearly equal squared lengths.
    let perpendicular: [f64; 3] = std::array::from_fn(|axis| {
        let b = (axis + 1) % 3;
        let c = (axis + 2) % 3;
        offset[b] * direction[c] - offset[c] * direction[b]
    });
    let discriminant = radius * radius - dot(perpendicular, perpendicular);
    if discriminant < 0.0 {
        return None;
    }
    let half_chord = discriminant.sqrt();
    [along - half_chord, along + half_chord]
        .into_iter()
        .find(|t| t.is_finite() && *t > 0.0 && max_t.is_none_or(|far| *t <= far))
}

fn selected_gizmo(batch: &mut LineBatch, transform: &Matrix, style: LineStyle) {
    if !transform.iter().all(|lane| lane.is_finite()) {
        return;
    }
    let point = |[x, y, z]: [f32; 3]| {
        std::array::from_fn(|row| {
            transform[row] * x + transform[4 + row] * y + transform[8 + row] * z + transform[12 + row]
        })
    };
    let origin = point([0.0; 3]);
    let corners: [[f32; 3]; 8] =
        std::array::from_fn(|mask| point(std::array::from_fn(|axis| if mask & (1 << axis) == 0 { -0.5 } else { 0.5 })));
    for axis in 0..3 {
        let mut end = [0.0; 3];
        end[axis] = 1.0;
        batch.segment(origin, point(end), GOLD, GOLD, style);
        let bit = 1 << axis;
        for mask in 0..8 {
            if mask & bit == 0 {
                batch.segment(corners[mask], corners[mask | bit], GOLD, GOLD, style);
            }
        }
    }
}

fn light_gizmos(batch: &mut LineBatch, frame: &RenderFrame, style: LineStyle) {
    let color = [1.25, 0.83, 0.27, 0.9];
    let dashed = LineStyle { width_px: 1.0, dash_len_px: 5.0, gap_len_px: 4.0, ..style };
    for light in frame.lights.iter().take(32) {
        let position = [light.position.x, light.position.y, light.position.z];
        if !position.into_iter().all(f32::is_finite) {
            continue;
        }
        for axis in 0..3 {
            let mut start = position;
            let mut end = position;
            start[axis] -= 0.15;
            end[axis] += 0.15;
            batch.segment(start, end, color, color, style);
        }
        batch.circle(position, [0.0, 1.0, 0.0], light.radius, 48, color, dashed);
    }
}

fn ui_bounds(batch: &mut LineBatch, frame: &RenderFrame, [width, height]: [f32; 2]) {
    for (node, visible) in frame.ui.iter().zip(crate::accessibility::included(&frame.ui)) {
        if !visible || node.width <= 0.0 || node.height <= 0.0 {
            continue;
        }
        let [left, top, right, bottom] =
            [node.x * width, node.y * height, (node.x + node.width) * width, (node.y + node.height) * height];
        if ![left, top, right, bottom].into_iter().all(f32::is_finite)
            || right <= left
            || bottom <= top
            || right <= 0.0
            || bottom <= 0.0
            || left >= width
            || top >= height
        {
            continue;
        }
        let (color, style) = if node.focused {
            ([1.0, 0.18, 0.85, 1.0], LineStyle { width_px: 2.0, ..Default::default() })
        } else {
            (
                [0.34, 0.82, 0.89, 0.75],
                LineStyle { width_px: 1.25, dash_len_px: 5.0, gap_len_px: 3.0, ..Default::default() },
            )
        };
        batch.polyline_2d(&[[left, top], [right, top], [right, bottom], [left, bottom]], color, true, style);
    }
}

fn pill(rect: [f32; 4], fill_color: [f32; 4]) -> SdfShape {
    SdfShape { rect, fill_color, kind: 2, ..Default::default() }
}

fn load_gradient(shapes: &mut Vec<SdfShape>, rect: [f32; 4], load: f32) {
    let [x, y, width, height] = rect;
    let colors = [[0.22, 0.81, 0.40, 1.0], [1.0, 0.65, 0.16, 1.0], [1.0, 0.22, 0.22, 1.0]];
    let clip_rect = [x, y, width * load, height];
    for (start, pair) in [0.0, 0.5].into_iter().zip(colors.windows(2)) {
        if load > start {
            shapes.push(SdfShape {
                rect: [x + width * start, y, width * 0.5, height],
                fill_color: pair[0],
                fill_color_end: pair[1],
                gradient_kind: 1,
                clip_rect,
                clip_radius_px: height * 0.5,
                ..Default::default()
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::f32::consts::FRAC_PI_2;

    use pen_protocol::{Draw, UiNode};
    use pen_render::{
        ENV_FOLIAGE_EXT_TAG, ENV_TERRAIN_EXT_TAG, ENV_WATER_EXT_TAG, GPU_PARTICLES_EXT_TAG, JFA_OUTLINE_EXT_TAG,
        LINES_EXT_TAG, RENDER2D_EXT_TAG, RENDER2D_WORLD_EXT_TAG, perspective_forward_z, perspective_reversed_z,
    };
    use pen_runtime::reverb::ACOUSTICS_EXT_TAG;

    use super::*;

    const VIEWPORT: [f32; 2] = [200.0, 200.0];
    const IDENTITY: Matrix = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    fn draw_at(x: f32, z: f32) -> Draw {
        let mut transform = IDENTITY;
        transform[12] = x;
        transform[14] = z;
        Draw { transform, ..Default::default() }
    }

    fn picking_frame() -> RenderFrame {
        RenderFrame {
            view: IDENTITY,
            projection: perspective_forward_z(FRAC_PI_2, 1.0, 0.1, 100.0),
            draws: vec![draw_at(-1.5, -5.0), draw_at(1.5, -5.0), draw_at(1.5, -3.0)],
            ..Default::default()
        }
    }

    #[test]
    fn extension_tags_are_unique_and_contiguous() {
        let tags = HashSet::from([
            ENV_TERRAIN_EXT_TAG,
            ENV_WATER_EXT_TAG,
            ENV_FOLIAGE_EXT_TAG,
            ACOUSTICS_EXT_TAG,
            GPU_PARTICLES_EXT_TAG,
            LINES_EXT_TAG,
            JFA_OUTLINE_EXT_TAG,
            RENDER2D_EXT_TAG,
            RENDER2D_WORLD_EXT_TAG,
        ]);
        assert_eq!(tags.len(), 9);
        assert_eq!(tags, (65_001..=65_009).collect());
    }

    #[test]
    fn picking_uses_near_to_far_rays_in_both_depth_modes() {
        let mut frame = picking_frame();
        let mut forward_ortho = IDENTITY;
        forward_ortho[0] = 0.25;
        forward_ortho[5] = 0.25;
        forward_ortho[10] = -1.0 / 100.0;
        let mut reversed_ortho = forward_ortho;
        reversed_ortho[10] = -forward_ortho[10];
        reversed_ortho[14] = 1.0;

        for projection in [
            frame.projection,
            perspective_reversed_z(FRAC_PI_2, 1.0, 0.1, Some(100.0)),
            perspective_reversed_z(FRAC_PI_2, 1.0, 0.1, None),
            forward_ortho,
            reversed_ortho,
        ] {
            frame.projection = projection;
            assert_eq!(pick_draw(&frame, [65.0, 100.0], VIEWPORT), Some(0));
            assert_eq!(pick_draw(&frame, [140.0, 100.0], VIEWPORT), Some(2));
            assert_eq!(pick_draw(&frame, [2.0, 2.0], VIEWPORT), None);
        }

        assert_eq!(pick_draw(&frame, [f32::NAN, 100.0], VIEWPORT), None);
        assert_eq!(pick_draw(&frame, [200.0, 100.0], VIEWPORT), None);
        assert_eq!(pick_draw(&frame, [140.0, 100.0], [200.0, 0.0]), None);
        frame.view = [0.0; 16];
        assert_eq!(pick_draw(&frame, [140.0, 100.0], VIEWPORT), None);
    }

    #[test]
    fn commands_and_selection_control_the_overlay_scenes() {
        let mut frame = picking_frame();
        frame.ui.push(UiNode { id: 1, x: 0.1, y: 0.2, width: 0.25, height: 0.1, focused: true, ..Default::default() });
        let stats = RenderStats { draw_calls: 2, ..Default::default() };

        let mut overlay = ViewportOverlay::new();
        assert!(!overlay.is_enabled());
        assert_eq!(overlay.apply_command("/viewport on").as_deref(), Some("Viewport overlay on"));
        assert!(overlay.is_enabled());
        assert_eq!(overlay.click_select(&frame, [65.0, 100.0], VIEWPORT), Some(0));
        overlay.update_pointer(&frame, [140.0, 100.0], VIEWPORT);

        let outlines = overlay.build_jfa_outlines(&frame).unwrap();
        assert_eq!(
            outlines.targets,
            [JfaOutlineTarget { draw_index: 0, style_index: 0 }, JfaOutlineTarget { draw_index: 2, style_index: 1 }]
        );
        assert_eq!(outlines.styles[0].width_px, 4.0);
        assert_eq!(outlines.styles[1].width_px, 2.5);
        overlay.hovered_draw = overlay.selected_draw;
        assert_eq!(overlay.build_jfa_outlines(&frame).unwrap().targets.len(), 1);
        overlay.hovered_draw = Some(frame.draws.len());
        assert_eq!(overlay.build_jfa_outlines(&frame).unwrap().targets.len(), 1);
        assert_eq!(overlay.apply_command("/viewport outlines").as_deref(), Some("Viewport outlines off"));
        assert!(overlay.build_jfa_outlines(&frame).is_none());
        overlay.apply_command("/viewport outlines");
        assert!(overlay.build_jfa_outlines(&frame).is_some());

        let lines = overlay.build_line_scene(&frame, VIEWPORT).unwrap();
        assert!(lines.batch.has_3d());
        assert!(lines.batch.has_2d());
        overlay.apply_command("/viewport gizmos");
        let lines = overlay.build_line_scene(&frame, VIEWPORT).unwrap();
        assert!(!lines.batch.has_3d());
        assert!(lines.batch.has_2d());
        frame.ui[0].hidden = true;
        assert!(overlay.build_line_scene(&frame, VIEWPORT).is_none());
        frame.ui[0].hidden = false;
        overlay.apply_command("/viewport ui");
        assert!(overlay.build_line_scene(&frame, VIEWPORT).is_none());
        overlay.apply_command("/viewport ui");
        overlay.apply_command("/viewport gizmos");

        let hud = overlay.build_sdf_hud(&frame, &stats, VIEWPORT).unwrap();
        let card = hud.sdf_overlay_shapes[0];
        assert_eq!(card.rect[0] + card.rect[2], VIEWPORT[0] - 12.0);
        assert_eq!(card.corner_radii, [10.0; 4]);
        assert_eq!(card.shadow_blur_px, 8.0);
        assert_eq!(card.border_width_px, 1.5);
        assert_eq!(hud.sdf_overlay_shapes.iter().filter(|shape| shape.gradient_kind == 1).count(), 2);
        assert!(hud.sdf_overlay_shapes.iter().any(|shape| shape.kind == 2 && shape.fill_color == GOLD));
        assert_eq!(hud.sdf_overlay_shapes.len(), 7);
        overlay.apply_command("/viewport hud");
        assert!(overlay.build_sdf_hud(&frame, &stats, VIEWPORT).is_none());
        overlay.apply_command("/viewport hud");

        assert_eq!(overlay.click_select(&frame, [2.0, 2.0], VIEWPORT), None);
        assert_eq!(overlay.click_select(&frame, [65.0, 100.0], VIEWPORT), Some(0));
        overlay.apply_command("/viewport clear");
        assert_eq!((overlay.selected_draw, overlay.hovered_draw), (None, None));
        overlay.click_select(&frame, [65.0, 100.0], VIEWPORT);

        assert!(overlay.apply_command("/viewport toggle").is_some());
        assert!(!overlay.is_enabled());
        overlay.apply_command("/viewport on");
        assert_eq!(overlay.apply_command("/viewport off").as_deref(), Some("Viewport overlay off"));
        assert!(!overlay.is_enabled());
        assert!(overlay.build_jfa_outlines(&frame).is_none());
        assert!(overlay.build_line_scene(&frame, VIEWPORT).is_none());
        assert!(overlay.build_sdf_hud(&frame, &stats, VIEWPORT).is_none());
        assert!(overlay.apply_command("/other on").is_none());
    }

    #[test]
    fn picking_respects_finite_far_plane_and_combined_shear() {
        // Finite far plane at 10.0 must reject a draw sitting beyond it at z = -12.0.
        let clipped_frame = RenderFrame {
            view: IDENTITY,
            projection: perspective_forward_z(FRAC_PI_2, 1.0, 0.1, 10.0),
            draws: vec![draw_at(0.0, -12.0)],
            ..Default::default()
        };
        assert_eq!(pick_draw(&clipped_frame, [100.0, 100.0], VIEWPORT), None);

        // Reversed-Z with finite far plane at 10.0 must also reject z = -12.0.
        let clipped_rz = RenderFrame {
            projection: perspective_reversed_z(FRAC_PI_2, 1.0, 0.1, Some(10.0)),
            ..clipped_frame.clone()
        };
        assert_eq!(pick_draw(&clipped_rz, [100.0, 100.0], VIEWPORT), None);

        // Unit cube sheared by x' = x + y + z under orthographic [-2, 2] x [-2, 2].
        let mut ortho = IDENTITY;
        ortho[0] = 0.5;
        ortho[5] = 0.5;
        ortho[10] = -0.1;
        ortho[14] = 0.5;
        // Column-major: col0=(1,0,0), col1=(1,1,0), col2=(1,0,1), col3=(0,0,0)
        let sheared_transform: Matrix =
            [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let sheared_frame = RenderFrame {
            view: IDENTITY,
            projection: ortho,
            draws: vec![Draw { transform: sheared_transform, ..Default::default() }],
            ..Default::default()
        };
        // World (1.4, 0.49) maps to NDC (0.7, 0.245) -> pixel (170.0, 75.5) on 200x200.
        assert_eq!(pick_draw(&sheared_frame, [170.0, 75.5], VIEWPORT), Some(0));
    }
}
