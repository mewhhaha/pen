// Shadows of point and spot lights. The game marks the lights that may cast
// (`Light::shadow`) and gives a budget (`shadow.local`); the renderer picks
// the casters, deterministically, and gives each its map faces in one
// depth-texture array shared by every local light: a spot is one perspective
// face along its cone, a point light six, the faces of a cube. Every face's
// window is a little wider than the face itself (`GUARD` texels), so that a
// filter at a face's edge reads what the same face holds past it and no seam
// shows between faces: a surface always reads the face of the axis its
// direction from the light leans on most.
use pen_protocol::{Light, Matrix, Vector};

use crate::culling::{Frustum, Sphere};
use crate::matrix::{invert, narrow};

/// The most map faces a frame draws: three point lights, or eighteen spots.
pub(crate) const MAX_FACES: usize = 18;

/// Texels past a face's edge its window holds, for filters to read.
const GUARD: f64 = 4.0;

/// A spot whose cone is wider than this half-angle (cosine) gets a cube.
const SPOT_LIMIT: f64 = 0.17364817766693033;

/// One map face: the matrices it is drawn with and read through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Face {
    pub view: Matrix,
    pub projection: Matrix,
    /// `projection * view`, narrowed.
    pub matrix: Matrix,
}

/// A light picked to cast, and where its faces are.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Caster {
    /// Its index in the frame's lights.
    pub light: usize,
    pub id: u32,
    /// Six faces of a cube, else one along its cone.
    pub cube: bool,
    /// Its first face's index among the selection's.
    pub first: usize,
    /// Its first layer of the atlas: its faces are the layers from there.
    pub layer: usize,
    /// The width of a face texel at unit depth along its axis, in world units.
    pub texel: f32,
}

/// The lights that cast this frame and their faces.
#[derive(Clone, Debug, Default)]
pub(crate) struct Selection {
    pub casters: Vec<Caster>,
    pub faces: Vec<Face>,
}

impl Selection {
    /// The caster of frame light `light`, if it was picked.
    pub fn of(&self, light: usize) -> Option<&Caster> {
        self.casters.iter().find(|caster| caster.light == light)
    }
}

type Vec3 = [f64; 3];

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn unit(a: Vec3) -> Vec3 {
    let length = dot(a, a).sqrt();
    [a[0] / length, a[1] / length, a[2] / length]
}

fn vec3(vector: &Vector) -> Vec3 {
    [vector.x, vector.y, vector.z].map(f64::from)
}

/// The cube faces' axes: +x, -x, +y, -y, +z, -z, as the shaders pick them,
/// and each one's up.
pub(crate) const CUBE: [([f64; 3], [f64; 3]); 6] = [
    ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
];

/// The face looking from `eye` along `forward` with `up`, seeing `half` (the
/// tangent of its half-angle) at depth 1, from `near` to `far`.
fn face(eye: Vec3, forward: Vec3, up: Vec3, half: f64, near: f64, far: f64) -> Face {
    let right = unit(cross(forward, up));
    let above = cross(right, forward);
    let view = [
        right[0],
        above[0],
        -forward[0],
        0.0,
        right[1],
        above[1],
        -forward[1],
        0.0,
        right[2],
        above[2],
        -forward[2],
        0.0,
        -dot(right, eye),
        -dot(above, eye),
        dot(forward, eye),
        1.0,
    ];
    let mut projection = [0.0f64; 16];
    projection[0] = 1.0 / half;
    projection[5] = 1.0 / half;
    projection[10] = far / (near - far);
    projection[11] = -1.0;
    projection[14] = near * far / (near - far);
    let mut matrix = [0.0f64; 16];
    for column in 0..4 {
        for row in 0..4 {
            matrix[column * 4 + row] = (0..4).map(|k| projection[k * 4 + row] * view[column * 4 + k]).sum();
        }
    }
    Face { view: view.map(narrow), projection: projection.map(narrow), matrix: matrix.map(narrow) }
}

/// The tangent of the half-angle that leaves `GUARD` texels past an angle's
/// edge on a map of `texels`.
fn widened(tangent: f64, texels: u32) -> f64 {
    let texels = f64::from(texels.max(1));
    tangent * texels / (texels - 2.0 * GUARD).max(1.0)
}

/// The faces of `light`: six round a point light, one along a spot's cone.
fn faces_of(light: &Light, texels: u32) -> (Vec<Face>, bool, f32) {
    let eye = vec3(&light.position);
    let far = f64::from(light.radius);
    let near = (far * 0.004).clamp(0.02, 0.25).min(far * 0.5);
    let axis = vec3(&light.direction);
    let cone = f64::from(light.outer);
    let spot = cone > SPOT_LIMIT && dot(axis, axis) > 1e-12;
    if spot {
        let forward = unit(axis);
        let up = if forward[1].abs() > 0.99 { [0.0, 0.0, 1.0] } else { [0.0, 1.0, 0.0] };
        let half = cone.acos().tan();
        let half = widened(half, texels);
        let texel = 2.0 * half / f64::from(texels.max(1));
        (vec![face(eye, forward, up, half, near, far)], false, narrow(texel))
    } else {
        let half = widened(1.0, texels);
        let faces = CUBE.iter().map(|&(forward, up)| face(eye, forward, up, half, near, far)).collect();
        (faces, true, narrow(2.0 * half / f64::from(texels.max(1))))
    }
}

/// The light at `index`'s priority among the casters: bigger on screen first
/// (its reach over its distance from the eye), then the lower id, then the
/// earlier index, so that equal lights always rank the same way.
fn rank(lights: &[Light], eye: Vec3, a: usize, b: usize) -> std::cmp::Ordering {
    let size = |index: usize| {
        let light = &lights[index];
        let distance = dot(sub(vec3(&light.position), eye), sub(vec3(&light.position), eye)).sqrt();
        f64::from(light.radius) / distance.max(f64::from(light.radius) * 0.05).max(1e-6)
    };
    size(b).total_cmp(&size(a)).then(lights[a].id.cmp(&lights[b].id)).then(a.cmp(&b))
}

/// Picks the casters among `lights` as seen by the camera: those that ask,
/// reach into its view and are within `budget`, by screen size (then id), as
/// many as fit `MAX_FACES`. Each gets its faces, on maps of `texels`.
pub(crate) fn select(lights: &[Light], view: &Matrix, projection: &Matrix, budget: u32, texels: u32) -> Selection {
    let mut selection = Selection::default();
    if budget == 0 || texels < 16 {
        return selection;
    }
    let seen = Frustum::of(projection, view);
    let inverse = invert(view);
    let eye = [f64::from(inverse[12]), f64::from(inverse[13]), f64::from(inverse[14])];
    let mut asking: Vec<usize> = (0..lights.len())
        .filter(|&index| {
            let light = &lights[index];
            light.shadow
                && light.radius > 0.0
                && seen.sees(&Sphere {
                    center: [light.position.x, light.position.y, light.position.z],
                    radius: light.radius,
                })
        })
        .collect();
    crate::sort::insertion_sort_by(&mut asking, |&a, &b| rank(lights, eye, a, b));
    for index in asking {
        if selection.casters.len() >= budget as usize {
            break;
        }
        let light = &lights[index];
        let (faces, cube, texel) = faces_of(light, texels);
        if selection.faces.len() + faces.len() > MAX_FACES {
            // A lesser light may still fit the faces that are left.
            continue;
        }
        selection.casters.push(Caster {
            light: index,
            id: light.id,
            cube,
            first: selection.faces.len(),
            layer: 0,
            texel,
        });
        selection.faces.extend(faces);
    }
    selection
}

impl Caster {
    /// How many faces it has.
    pub fn faces(&self) -> usize {
        if self.cube { 6 } else { 1 }
    }
}

/// Which light's first face each atlas layer holds from the frame before:
/// its id and whether it is a cube.
pub(crate) type Held = Option<(u32, bool)>;

/// Gives each caster its layers of the atlas: the ones it held last frame
/// when they are free (so that its map may be kept), else the first free run.
/// A cube's faces are the six layers in a row.
pub(crate) fn assign(selection: &mut Selection, held: &[Held; MAX_FACES]) {
    let mut taken = [false; MAX_FACES];
    let mut placed: Vec<Option<usize>> = vec![None; selection.casters.len()];
    let free =
        |taken: &[bool; MAX_FACES], start: usize, length: usize| !taken[start..start + length].iter().any(|&t| t);
    for (index, caster) in selection.casters.iter().enumerate() {
        let length = caster.faces();
        let kept = (0..=MAX_FACES - length)
            .find(|&start| held[start] == Some((caster.id, caster.cube)) && free(&taken, start, length));
        if let Some(start) = kept {
            taken[start..start + length].fill(true);
            placed[index] = Some(start);
        }
    }
    for (index, caster) in selection.casters.iter().enumerate() {
        if placed[index].is_some() {
            continue;
        }
        let length = caster.faces();
        if let Some(start) = (0..=MAX_FACES - length).find(|&start| free(&taken, start, length)) {
            taken[start..start + length].fill(true);
            placed[index] = Some(start);
        }
    }
    // A cube that finds no run of six, though the layers are enough: start over, in order.
    if placed.iter().any(Option::is_none) {
        let mut next = 0;
        for (index, caster) in selection.casters.iter().enumerate() {
            placed[index] = Some(next);
            next += caster.faces();
        }
    }
    for (caster, layer) in selection.casters.iter_mut().zip(placed) {
        caster.layer = layer.unwrap_or(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::IDENTITY;

    fn light(id: u32, at: [f32; 3], radius: f32) -> Light {
        Light {
            position: Vector { x: at[0], y: at[1], z: at[2] },
            radius,
            color: Vector { x: 1.0, y: 1.0, z: 1.0 },
            direction: Vector { x: 0.0, y: -1.0, z: 0.0 },
            inner: -1.0,
            outer: -1.0,
            shadow: true,
            id,
        }
    }

    fn camera() -> (Matrix, Matrix) {
        let mut projection = [0.0; 16];
        projection[0] = 1.0;
        projection[5] = 1.0;
        projection[10] = -1.001;
        projection[11] = -1.0;
        projection[14] = -0.1001;
        (IDENTITY, projection)
    }

    fn ids(selection: &Selection) -> Vec<u32> {
        selection.casters.iter().map(|caster| caster.id).collect()
    }

    #[test]
    fn nearer_and_bigger_lights_win_and_ties_break_by_id() {
        let (view, projection) = camera();
        let spot = |id, at, radius| {
            let mut spot = light(id, at, radius);
            spot.outer = 0.8;
            spot.inner = 0.9;
            spot
        };
        let lights = [
            spot(7, [0.0, 0.0, -20.0], 4.0),
            spot(3, [0.0, 0.0, -10.0], 4.0),
            spot(9, [1.0, 0.0, -10.0], 4.0),
            spot(5, [0.0, 0.0, -10.0], 4.0),
            spot(1, [0.0, 0.0, -30.0], 4.0),
        ];
        let picked = select(&lights, &view, &projection, 2, 512);
        // Two equally near lights: id 3 before id 5; then the dearer ones.
        assert_eq!(ids(&picked), vec![3, 5]);
        // The order the frame lists them in changes nothing.
        let mut shuffled = lights;
        shuffled.reverse();
        let again = select(&shuffled, &view, &projection, 2, 512);
        assert_eq!(ids(&again), vec![3, 5]);
        // A bigger budget takes more, in the same order.
        let all = select(&lights, &view, &projection, 16, 512);
        assert_eq!(ids(&all), vec![3, 5, 9, 7, 1]);
    }

    #[test]
    fn lights_that_do_not_ask_or_cannot_be_seen_cast_nothing() {
        let (view, projection) = camera();
        let mut quiet = light(1, [0.0, 0.0, -5.0], 3.0);
        quiet.shadow = false;
        let behind = light(2, [0.0, 0.0, 50.0], 3.0);
        let beside = light(3, [100.0, 0.0, -5.0], 3.0);
        assert!(select(&[quiet, behind, beside], &view, &projection, 8, 512).casters.is_empty());
        assert!(select(&[light(4, [0.0, 0.0, -5.0], 3.0)], &view, &projection, 0, 512).casters.is_empty());
    }

    #[test]
    fn faces_run_out_and_a_smaller_light_may_still_fit() {
        let (view, projection) = camera();
        let mut lights: Vec<Light> = (0..4).map(|id| light(id, [0.0, 0.0, -5.0 - id as f32], 3.0)).collect();
        // Three point lights fill the 18 faces; a spot behind them still fits nowhere.
        let spot = {
            let mut spot = light(9, [0.0, 0.0, -20.0], 3.0);
            spot.outer = 0.9;
            spot.inner = 0.95;
            spot
        };
        lights.push(spot);
        let picked = select(&lights, &view, &projection, 16, 512);
        assert_eq!(picked.faces.len(), 18);
        assert_eq!(ids(&picked), vec![0, 1, 2]);
        // With room for one more face, the spot fits where the fourth cube could not.
        let fewer = [lights[0], lights[1], spot];
        let picked = select(&fewer, &view, &projection, 16, 512);
        assert_eq!((picked.faces.len(), picked.casters.len()), (13, 3));
        assert!(!picked.casters[2].cube && picked.casters[2].first == 12);
    }

    /// Where `face` puts `point`: its texture coordinates and depth.
    fn project(face: &Face, point: Vec3) -> Option<(f64, f64, f64)> {
        let m = face.matrix.map(f64::from);
        let clip =
            [0, 1, 2, 3].map(|row| m[row] * point[0] + m[4 + row] * point[1] + m[8 + row] * point[2] + m[12 + row]);
        (clip[3] > 0.0).then(|| (clip[0] / clip[3] * 0.5 + 0.5, 0.5 - clip[1] / clip[3] * 0.5, clip[2] / clip[3]))
    }

    #[test]
    fn a_cube_sees_every_direction_with_a_guard_band_past_each_face() {
        let at = light(1, [1.0, 2.0, 3.0], 10.0);
        let (faces, cube, texel) = faces_of(&at, 512);
        assert!(cube && faces.len() == 6);
        for (index, &(axis, _)) in CUBE.iter().enumerate() {
            // Straight along an axis: the middle of its face.
            let point = [1.0 + axis[0] * 5.0, 2.0 + axis[1] * 5.0, 3.0 + axis[2] * 5.0];
            let (u, v, _) = project(&faces[index], point).unwrap();
            assert!((u - 0.5).abs() < 1e-6 && (v - 0.5).abs() < 1e-6, "face {index}: {u} {v}");
            // The face's own edge, 45 degrees off, still lies inside by the guard.
            let across = if axis[0] != 0.0 {
                [0.0, 1.0, 0.0]
            } else if axis[1] != 0.0 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let at = [
                1.0 + (axis[0] + across[0]) * 5.0,
                2.0 + (axis[1] + across[1]) * 5.0,
                3.0 + (axis[2] + across[2]) * 5.0,
            ];
            let (u, v, _) = project(&faces[index], at).unwrap();
            // The face's own edge lies inside its window by the guard's texels.
            let outside = (u - 0.5).abs().max((v - 0.5).abs()) - 0.5;
            assert!((outside + GUARD / 512.0).abs() < 1e-6, "face {index} edge at {u} {v}");
        }
        // Each texel at unit depth is as wide as 2 * half / texels.
        assert!((f64::from(texel) - 2.0 * widened(1.0, 512) / 512.0).abs() < 1e-7);
    }

    #[test]
    fn layers_stay_with_their_light_while_it_stays_picked() {
        let (view, projection) = camera();
        let lights = [light(1, [0.0, 0.0, -5.0], 3.0), light(2, [0.0, 0.0, -9.0], 3.0)];
        let mut picked = select(&lights, &view, &projection, 4, 256);
        let mut held: [Held; MAX_FACES] = [None; MAX_FACES];
        assign(&mut picked, &held);
        assert_eq!((picked.casters[0].layer, picked.casters[1].layer), (0, 6));
        for caster in &picked.casters {
            held[caster.layer] = Some((caster.id, caster.cube));
        }
        // The nearer light leaves: the other keeps its layers, and a new one
        // takes the freed ones.
        let later = [light(2, [0.0, 0.0, -9.0], 3.0), light(3, [0.0, 0.0, -5.0], 3.0)];
        let mut again = select(&later, &view, &projection, 4, 256);
        assign(&mut again, &held);
        let by_id = |id: u32| again.casters.iter().find(|caster| caster.id == id).unwrap().layer;
        assert_eq!((by_id(2), by_id(3)), (6, 0));
        // A cube that finds no run of six among spots starts everything over.
        let mut spots = Vec::new();
        for (id, at) in [(10u32, -5.0f32), (11, -5.5), (12, -6.0)] {
            let mut spot = light(id, [0.0, 0.0, at], 3.0);
            spot.outer = 0.9;
            spots.push(spot);
        }
        let mut scattered = select(&spots, &view, &projection, 4, 256);
        let mut held: [Held; MAX_FACES] = [None; MAX_FACES];
        for (id, layer) in [(10u32, 0usize), (11, 6), (12, 12)] {
            held[layer] = Some((id, false));
        }
        scattered.casters.push(Caster { light: 9, id: 20, cube: true, first: 3, layer: 0, texel: 0.0 });
        assign(&mut scattered, &held);
        let mut used = [0u8; MAX_FACES];
        for caster in &scattered.casters {
            for slot in &mut used[caster.layer..caster.layer + caster.faces()] {
                *slot += 1;
            }
        }
        assert!(used.iter().all(|&count| count <= 1), "{used:?}");
    }

    #[test]
    fn a_spot_looks_down_its_cone_and_a_wide_one_becomes_a_cube() {
        let mut spot = light(1, [0.0, 4.0, 0.0], 12.0);
        spot.outer = 0.5;
        spot.inner = 0.6;
        let (faces, cube, _) = faces_of(&spot, 256);
        assert!(!cube && faces.len() == 1);
        // The cone's axis meets the middle; the cone's edge just inside the window.
        let (u, v, depth) = project(&faces[0], [0.0, 0.0, 0.0]).unwrap();
        assert!((u - 0.5).abs() < 1e-6 && (v - 0.5).abs() < 1e-6 && depth > 0.0 && depth < 1.0);
        let edge = 4.0 * 0.5f64.acos().tan();
        let (u, _, _) = project(&faces[0], [edge, 0.0, 0.0]).unwrap();
        // The cone's edge lies just inside the window, the guard's texels from its edge.
        assert!(((u - 0.5).abs() - (0.5 - GUARD / 256.0)).abs() < 1e-3, "{u}");
        let mut wide = spot;
        wide.outer = 0.1;
        assert!(faces_of(&wide, 256).1);
    }
}
