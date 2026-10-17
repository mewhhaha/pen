// Cascaded shadow maps: beside the game's own map (`frame.shadow`, which
// covers the floor where the camera looks), the renderer fits a nearer map
// of the same size inside the camera's view, just wide enough for the
// stretch of the view nearer than its split, so that shadows close to the
// camera get finer texels. They look along the game's light,
// with its depth range, and their windows move in whole texels of their own,
// so their shadows hold still as the camera moves, as the game's do. Shaders
// that read them (`shadow_cascades`) take each surface from the nearest map
// that holds it, fading into the next toward its edge; shaders that do not
// read the game's map as before.
use pen_protocol::{Matrix, RenderFrame};

use crate::matrix::{invert, multiply};

/// The most maps a frame draws, and so the most the scene uniform describes.
pub(crate) const MAX_CASCADES: usize = 4;

/// Maps fitted nearer the camera than the game's. Each costs about as much
/// GPU time as the game's own, as gdev's largest casters (the crater, the
/// rocks) reach into every map: in its play view on an RTX 2070 SUPER at
/// 1920x1280 the shadow passes took 1.1 ms with the game's map alone, 2.0
/// with one nearer and 3.0 with two.
const NEARER: usize = 1;

// Where the splits fall between linear (0) and logarithmic (1) spacing.
const LOGARITHMIC: f64 = 0.6;

// A nearer map is fitted only if it is at most this share of the game's.
const SMALLER: f64 = 0.8;

/// A shadow map: what it sees (in the game's light view), and how wide a
/// texel of it is in world units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cascade {
    pub projection: Matrix,
    pub texel: f32,
}

/// The maps `frame` draws, nearest first, the game's own last: a nearer one
/// when `nearer`, where the game's map is an orthographic one
/// along its light (the shape `ink/graphics` fits) and the camera's view
/// reaches into it. Empty without shadows.
pub(crate) fn cascades(frame: &RenderFrame, nearer: bool) -> ([Cascade; MAX_CASCADES], usize) {
    let shadow = &frame.shadow;
    let game = Cascade { projection: shadow.projection, texel: shadow.texel };
    let mut maps = [game; MAX_CASCADES];
    if shadow.texels == 0 {
        return (maps, 0);
    }
    let fitted = if nearer { fit(frame) } else { Vec::new() };
    maps[..fitted.len()].copy_from_slice(&fitted);
    maps[fitted.len()] = game;
    (maps, fitted.len() + 1)
}

/// Whether `p` is an orthographic projection along its axes.
fn orthographic(p: &Matrix) -> bool {
    [1, 2, 3, 4, 6, 7, 8, 9, 11].iter().all(|&at| p[at] == 0.0) && p[15] == 1.0 && p[0] > 0.0 && p[5] > 0.0
}

fn fit(frame: &RenderFrame) -> Vec<Cascade> {
    let shadow = &frame.shadow;
    let p = shadow.projection.map(f64::from);
    if !orthographic(&shadow.projection) || shadow.texels < 2 {
        return Vec::new();
    }
    // The game's window in the light's view: its half width and middle.
    let reach = (1.0 / p[0]).max(1.0 / p[5]);
    let middle = [-p[12] / p[0], -p[13] / p[5], (0.5 - p[14]) / p[10]];
    let light = shadow.view.map(f64::from);
    let to_world = invert(&shadow.view).map(f64::from);
    let transform = |m: &[f64; 16], v: [f64; 3]| {
        [0, 1, 2].map(|row| m[row] * v[0] + m[4 + row] * v[1] + m[8 + row] * v[2] + m[12 + row])
    };
    // The camera's view volume: its near and far corners, by NDC x and y.
    let unproject = invert(&multiply(&frame.projection, &frame.view)).map(f64::from);
    let corner = |x: f64, y: f64, z: f64| {
        let m = &unproject;
        let point = [0, 1, 2, 3].map(|row| m[row] * x + m[4 + row] * y + m[8 + row] * z + m[12 + row]);
        [point[0] / point[3], point[1] / point[3], point[2] / point[3]]
    };
    let edges: Vec<([f64; 3], [f64; 3])> = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
        .into_iter()
        .map(|(x, y)| (corner(x, y, 0.0), corner(x, y, 1.0)))
        .collect();
    if edges.iter().flat_map(|(near, far)| near.iter().chain(far)).any(|value| !value.is_finite()) {
        return Vec::new();
    }
    // Distances ahead of the camera (its view's -z), which runs linearly
    // along each edge from its near corner to its far one.
    let view = frame.view.map(f64::from);
    let ahead = |point: [f64; 3]| -(view[2] * point[0] + view[6] * point[1] + view[10] * point[2] + view[14]);
    let (near, far) = (ahead(edges[0].0), ahead(edges[0].1));
    // As far as the game's window reaches from the camera.
    let window = ahead(transform(&to_world, middle)) + reach;
    let farthest = window.min(far);
    if !(near > 0.0 && farthest > near) {
        return Vec::new();
    }
    let texels = f64::from(shadow.texels);
    let mut fitted = Vec::new();
    for split in 1..=NEARER {
        let share = split as f64 / (NEARER + 1) as f64;
        let distance = LOGARITHMIC * near * (farthest / near).powf(share)
            + (1.0 - LOGARITHMIC) * (near + (farthest - near) * share);
        let along = (distance - near) / (far - near);
        // The slice of the view nearer than the split, round its corners'
        // middle, its radius rounded up to a sixteenth so it keeps its size.
        let corners: Vec<[f64; 3]> = edges
            .iter()
            .flat_map(|(start, end)| [*start, [0, 1, 2].map(|axis| start[axis] + (end[axis] - start[axis]) * along)])
            .collect();
        let centre = [0, 1, 2].map(|axis| corners.iter().map(|corner| corner[axis]).sum::<f64>() / 8.0);
        let radius = corners
            .iter()
            .map(|corner| (0..3).map(|axis| (corner[axis] - centre[axis]).powi(2)).sum::<f64>().sqrt())
            .fold(0.0, f64::max);
        // Room for the snapping to move it half a texel either way.
        let radius = (radius * 16.0).ceil() / 16.0 * texels / (texels - 1.0);
        if radius > reach * SMALLER {
            break;
        }
        // Its middle in the light's view, moved in whole texels.
        let texel = 2.0 * radius / texels;
        let [x, y, _] = transform(&light, centre);
        let (x, y) = ((x / texel).round() * texel, (y / texel).round() * texel);
        let mut projection = shadow.projection;
        projection[0] = narrow(1.0 / radius);
        projection[5] = narrow(1.0 / radius);
        projection[12] = narrow(-x / radius);
        projection[13] = narrow(-y / radius);
        fitted.push(Cascade { projection, texel: narrow(texel) });
    }
    fitted
}

fn narrow(value: f64) -> f32 {
    crate::matrix::narrow(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::IDENTITY;

    /// A frame whose camera stands at `eye` looking 60 degrees down along
    /// -z, and whose sun shines straight down on a game map `reach` wide
    /// round the origin.
    fn frame(eye: [f32; 3], reach: f32) -> RenderFrame {
        let mut frame = RenderFrame::default();
        let (near, far) = (0.5f32, 500.0f32);
        let mut projection = [0.0; 16];
        projection[0] = 1.0;
        projection[5] = 1.5;
        projection[10] = far / (near - far);
        projection[11] = -1.0;
        projection[14] = near * far / (near - far);
        frame.projection = projection;
        let (sin, cos) = ((-60f32).to_radians().sin(), (-60f32).to_radians().cos());
        let rotation = [1.0, 0.0, 0.0, 0.0, 0.0, cos, -sin, 0.0, 0.0, sin, cos, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut translation = IDENTITY;
        translation[12] = -eye[0];
        translation[13] = -eye[1];
        translation[14] = -eye[2];
        frame.view = multiply(&rotation, &translation);
        // The sun's view looks down -y from 1000 up: x stays, -z is up the
        // map, and depth is 1000 - y.
        frame.shadow.view = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1000.0, 1.0];
        let mut sun = IDENTITY;
        sun[0] = 1.0 / reach;
        sun[5] = 1.0 / reach;
        sun[10] = -1.0 / 200.0;
        sun[14] = -900.0 / 200.0;
        frame.shadow.projection = sun;
        frame.shadow.texels = 2048;
        frame.shadow.texel = 2.0 * reach / 2048.0;
        frame
    }

    #[test]
    fn nearer_maps_fit_inside_the_game_s_and_move_in_whole_texels() {
        let frame = frame([0.0, 30.0, 17.0], 60.0);
        let (maps, count) = cascades(&frame, true);
        assert_eq!(count, NEARER + 1);
        // The game's own is last, as it gave it.
        assert_eq!(maps[NEARER].projection, frame.shadow.projection);
        assert_eq!(maps[NEARER].texel, frame.shadow.texel);
        // Each nearer one finer than the next, keeping the game's depth.
        assert!(maps[..=NEARER].windows(2).all(|pair| pair[0].texel < pair[1].texel), "{maps:?}");
        for map in &maps[..NEARER] {
            assert_eq!(
                (map.projection[10], map.projection[14]),
                (frame.shadow.projection[10], frame.shadow.projection[14])
            );
            // Its middle on its own texel grid.
            let middle = -map.projection[12] / map.projection[0];
            assert!(((middle / map.texel).round() - middle / map.texel).abs() < 1e-3, "{map:?}");
        }
        // The camera moving keeps their sizes, and moves them in whole texels.
        let (moved, _) = cascades(&frame_moved(&frame, 0.37), true);
        for (map, moved) in maps.iter().zip(&moved).take(NEARER) {
            assert_eq!((moved.projection[0], moved.texel), (map.projection[0], map.texel));
            let shift = (map.projection[12] - moved.projection[12]) / map.projection[0] / map.texel;
            assert!((shift - shift.round()).abs() < 1e-2 && shift.round() != 0.0, "{shift}");
        }
        // Without asking, or without shadows, or with a map that is not
        // orthographic: the game's alone, or none.
        assert_eq!(cascades(&frame, false).1, 1);
        let mut dark = frame.clone();
        dark.shadow.texels = 0;
        assert_eq!(cascades(&dark, true).1, 0);
        let mut perspective = frame.clone();
        perspective.shadow.projection[11] = -1.0;
        assert_eq!(cascades(&perspective, true).1, 1);
        // A game's map too small to split: its own alone.
        assert_eq!(cascades(&super::tests::frame([0.0, 30.0, 17.0], 3.0), true).1, 1);
    }

    fn frame_moved(frame: &RenderFrame, by: f32) -> RenderFrame {
        let mut moved = frame.clone();
        moved.view[12] -= by;
        moved
    }
}
