//! Tangents for the primitives that need a tangent frame (a normal map, an
//! anisotropic material) and come without a `TANGENT` attribute: the
//! MikkTSpace algorithm (Morten Mikkelsen, "Simulation of Wrinkled Surfaces
//! Revisited", 2008; the reference implementation `mikktspace.c`, which the
//! glTF specification names), made here because no implementation of it is
//! among pen's dependencies.
//!
//! What is the reference's, step by step:
//!
//! 1. Vertices equal in position, normal and texture coordinates are welded.
//! 2. Each triangle's first order derivatives give its tangent `s` (the
//!    direction `u` grows in) and bitangent `t` (the direction `v` grows in),
//!    unit vectors, and whether its texture coordinates keep its winding
//!    (`ORIENT_PRESERVING`). A triangle with no area in texture space has no
//!    say (`GROUP_WITH_ANY`).
//! 3. Triangles sharing an edge are neighbours. Around each welded vertex they
//!    form groups by the reference's four rules: neighbours that agree in
//!    orientation join (`GROUP_WITH_ANY` ones join the first group that
//!    reaches them and take its orientation).
//! 4. Within a group, the faces whose `s` and `t` lie within the angular
//!    threshold of each other's (the reference's default threshold is 180
//!    degrees, so all of them) make a subgroup, whose tangent space is the sum
//!    of its triangles' `s` and `t` projected into the vertex's tangent plane,
//!    each weighted by the triangle's angle at the vertex, normalised.
//! 5. The tangent is that `s`; the handedness is +1 where the group preserves
//!    orientation and -1 where it does not.
//!
//! What differs from the reference, and why:
//!
//! - Triangles only (the reference also takes quads, and glTF has none), so a
//!   corner is in one group and the reference's averaging of a corner's two
//!   tangent spaces never happens.
//! - Arithmetic in f64 rather than f32, narrowed at the end; the reference's
//!   "not zero" tests (`fabsf(x) > FLT_MIN`) are kept as they are.
//! - An edge shared by more than two triangles pairs them in face order (the
//!   reference pairs whichever its sort puts together).
//! - The reference's degenerate-triangle prologue and epilogue (moving
//!   triangles with a repeated position to the end and giving them a
//!   neighbour's tangent space) is the `GROUP_WITH_ANY` handling above, and a
//!   corner that no group reaches (an isolated degenerate triangle, a vertex
//!   no triangle uses) gets an arbitrary unit tangent perpendicular to its
//!   normal and handedness +1 instead of the reference's default `(1, 0, 0)`,
//!   which is not perpendicular to anything in particular.
//! - Where the reference leaves a vertex of two tangent spaces to its caller
//!   (mirrored texture coordinates over a shared vertex), [`split`] makes the
//!   vertex two, since a glTF attribute has one tangent a vertex.
//! - Texture coordinates are glTF's, `v` down the image; the algorithm runs on
//!   `1 - v`, so that the bitangent `cross(normal, tangent) * w` points where the
//!   texture's green channel grows (up the image), as the glTF specification's
//!   normal maps have it.
use std::collections::HashMap;

use crate::Geometry;

type Vector = [f64; 3];

const GROUP_WITH_ANY: u8 = 1;
const ORIENT_PRESERVING: u8 = 2;

fn sub(a: Vector, b: Vector) -> Vector {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: Vector, b: Vector) -> Vector {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(factor: f64, a: Vector) -> Vector {
    [factor * a[0], factor * a[1], factor * a[2]]
}

fn dot(a: Vector, b: Vector) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: Vector) -> f64 {
    dot(a, a).sqrt()
}

/// The reference's `NotZero`.
fn not_zero(value: f64) -> bool {
    value.abs() > f64::from(f32::MIN_POSITIVE)
}

fn vector_not_zero(a: Vector) -> bool {
    a.into_iter().any(not_zero)
}

/// `a` made one long, unless it has no length.
fn normalized(a: Vector) -> Vector {
    if vector_not_zero(a) { scale(1.0 / length(a), a) } else { a }
}

/// `a` with what lies along the unit vector `n` taken out, one long again.
fn projected(a: Vector, n: Vector) -> Vector {
    normalized(sub(a, scale(dot(n, a), n)))
}

/// Any unit vector perpendicular to the unit vector `n`.
fn perpendicular(n: Vector) -> Vector {
    let axis = if n[0].abs() <= n[1].abs() && n[0].abs() <= n[2].abs() {
        [1.0, 0.0, 0.0]
    } else if n[1].abs() <= n[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    normalized(sub(axis, scale(dot(n, axis), n)))
}

struct Triangle {
    /// The welded vertices of its corners.
    corners: [usize; 3],
    flags: u8,
    /// Its tangent and bitangent, unit vectors (zero where it has none).
    s: Vector,
    t: Vector,
    /// The triangle across the edge from each corner to the next.
    neighbours: [Option<usize>; 3],
    /// The group each corner is in.
    groups: [Option<usize>; 3],
}

struct Group {
    /// The welded vertex it is around.
    vertex: usize,
    preserving: bool,
    faces: Vec<usize>,
}

/// Each corner's tangent (xyz, then 1 or -1 as the handedness of the frame
/// `cross(normal, tangent) * w`) for the triangles of `indices` over these
/// vertices, in the order of `indices`. `normals` must be unit vectors.
pub fn corners(positions: &[[f32; 3]], normals: &[[f32; 3]], uvs: &[[f32; 2]], indices: &[u32]) -> Vec<[f32; 4]> {
    let position = |vertex: usize| positions[vertex].map(f64::from);
    let normal = |vertex: usize| normals[vertex].map(f64::from);
    // The texture's `v` runs up, as the normal map's green channel does.
    let uv = |vertex: usize| [f64::from(uvs[vertex][0]), 1.0 - f64::from(uvs[vertex][1])];

    // 1. Welding.
    let mut welded: HashMap<[u32; 8], usize> = HashMap::new();
    let mut representative: Vec<usize> = Vec::new();
    let weld: Vec<usize> = (0..positions.len())
        .map(|vertex| {
            // Negative zero is zero.
            let bits = |value: f32| (value + 0.0).to_bits();
            let [x, y, z] = positions[vertex].map(bits);
            let [nx, ny, nz] = normals[vertex].map(bits);
            let [u, v] = uvs[vertex].map(bits);
            *welded.entry([x, y, z, nx, ny, nz, u, v]).or_insert_with(|| {
                representative.push(vertex);
                representative.len() - 1
            })
        })
        .collect();

    // 2. Each triangle's derivatives.
    let mut triangles: Vec<Triangle> = indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|corners| {
            let [a, b, c] = corners.map(|corner| corner as usize);
            let (p1, p2, p3) = (position(a), position(b), position(c));
            let (t1, t2, t3) = (uv(a), uv(b), uv(c));
            let (t21x, t21y) = (t2[0] - t1[0], t2[1] - t1[1]);
            let (t31x, t31y) = (t3[0] - t1[0], t3[1] - t1[1]);
            let (d1, d2) = (sub(p2, p1), sub(p3, p1));
            let signed_area = t21x * t31y - t21y * t31x;
            let os = sub(scale(t31y, d1), scale(t21y, d2));
            let ot = add(scale(-t31x, d1), scale(t21x, d2));
            let mut triangle = Triangle {
                corners: [weld[a], weld[b], weld[c]],
                flags: GROUP_WITH_ANY | if signed_area > 0.0 { ORIENT_PRESERVING } else { 0 },
                s: [0.0; 3],
                t: [0.0; 3],
                neighbours: [None; 3],
                groups: [None; 3],
            };
            if not_zero(signed_area) {
                let area = signed_area.abs();
                let (length_s, length_t) = (length(os), length(ot));
                let sign = if triangle.flags & ORIENT_PRESERVING == 0 { -1.0 } else { 1.0 };
                if not_zero(length_s) {
                    triangle.s = scale(sign / length_s, os);
                }
                if not_zero(length_t) {
                    triangle.t = scale(sign / length_t, ot);
                }
                // A good triangle has a say.
                if not_zero(length_s / area) && not_zero(length_t / area) {
                    triangle.flags &= !GROUP_WITH_ANY;
                }
            }
            triangle
        })
        .collect();

    // 3. Neighbours across shared edges, and the groups around each vertex.
    let mut edges: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for (face, triangle) in triangles.iter().enumerate() {
        for corner in 0..3 {
            let (a, b) = (triangle.corners[corner], triangle.corners[(corner + 1) % 3]);
            if a != b {
                edges.entry((a.min(b), a.max(b))).or_default().push((face, corner));
            }
        }
    }
    for sharing in edges.values() {
        for pair in sharing.as_chunks::<2>().0 {
            let ((first, at), (second, other)) = (pair[0], pair[1]);
            if first != second {
                triangles[first].neighbours[at] = Some(second);
                triangles[second].neighbours[other] = Some(first);
            }
        }
    }
    let mut groups: Vec<Group> = Vec::new();
    for face in 0..triangles.len() {
        for corner in 0..3 {
            if triangles[face].flags & GROUP_WITH_ANY != 0 || triangles[face].groups[corner].is_some() {
                continue;
            }
            let group = groups.len();
            groups.push(Group {
                vertex: triangles[face].corners[corner],
                preserving: triangles[face].flags & ORIENT_PRESERVING != 0,
                faces: vec![face],
            });
            triangles[face].groups[corner] = Some(group);
            // The neighbour across the edge leaving the corner, then the one
            // across the edge arriving at it, each followed as far as it goes.
            let (left, right) = (triangles[face].neighbours[corner], triangles[face].neighbours[(corner + 2) % 3]);
            let mut pending: Vec<usize> = right.into_iter().chain(left).collect();
            while let Some(next) = pending.pop() {
                let triangle = &mut triangles[next];
                let Some(at) = triangle.corners.iter().position(|&corner| corner == groups[group].vertex) else {
                    continue;
                };
                if triangle.groups[at].is_some() {
                    continue;
                }
                if triangle.flags & GROUP_WITH_ANY != 0 && triangle.groups.iter().all(Option::is_none) {
                    // The first group to reach a triangle with no say gives it
                    // its orientation: the reference's one dependency on order.
                    triangle.flags &= !ORIENT_PRESERVING;
                    if groups[group].preserving {
                        triangle.flags |= ORIENT_PRESERVING;
                    }
                }
                if (triangle.flags & ORIENT_PRESERVING != 0) != groups[group].preserving {
                    continue;
                }
                groups[group].faces.push(next);
                triangle.groups[at] = Some(group);
                pending.extend(triangle.neighbours[(at + 2) % 3]);
                pending.extend(triangle.neighbours[at]);
            }
        }
    }

    // 4. Each group's subgroups' tangent spaces, to the corners they are for.
    let mut spaces: Vec<Option<(Vector, bool)>> = vec![None; triangles.len() * 3];
    // The angular threshold of the default: 180 degrees, whose cosine is -1.
    let threshold = -1.0;
    for (group_index, group) in groups.iter().enumerate() {
        let n = normal(representative[group.vertex]);
        let mut subgroups: Vec<(Vec<usize>, Vector)> = Vec::new();
        // Each face's tangent and bitangent in the vertex's tangent plane.
        let flats: Vec<(Vector, Vector)> = group
            .faces
            .iter()
            .map(|&face| (projected(triangles[face].s, n), projected(triangles[face].t, n)))
            .collect();
        for (place, &face) in group.faces.iter().enumerate() {
            let corner = (0..3).find(|&corner| triangles[face].groups[corner] == Some(group_index)).expect("in group");
            let (s, t) = flats[place];
            let mut members: Vec<usize> = group
                .faces
                .iter()
                .zip(&flats)
                .filter(|&(&other, &(s2, t2))| {
                    let any = (triangles[face].flags | triangles[other].flags) & GROUP_WITH_ANY != 0;
                    any || face == other || (dot(s, s2) > threshold && dot(t, t2) > threshold)
                })
                .map(|(&other, _)| other)
                .collect();
            members.sort_unstable();
            let found = subgroups.iter().position(|(known, _)| *known == members);
            let at = found.unwrap_or_else(|| {
                let space = evaluate(&members, &triangles, group.vertex, &representative, positions, normals);
                subgroups.push((members, space));
                subgroups.len() - 1
            });
            spaces[face * 3 + corner] = Some((subgroups[at].1, group.preserving));
        }
    }

    // 5. The tangents, those no group reached from their normals.
    indices
        .iter()
        .enumerate()
        .map(|(at, &vertex)| {
            let n = normal(vertex as usize);
            match spaces[at] {
                Some((space, preserving)) if vector_not_zero(space) && space.iter().all(|value| value.is_finite()) => {
                    let [x, y, z] = space.map(narrow);
                    [x, y, z, if preserving { 1.0 } else { -1.0 }]
                }
                _ => {
                    let [x, y, z] = perpendicular(n).map(narrow);
                    [x, y, z, 1.0]
                }
            }
        })
        .collect()
}

/// A subgroup's tangent around `vertex`: its faces' tangents projected into
/// the vertex's tangent plane, summed by the angle each face has at the vertex,
/// and made a unit vector. (The reference does the same for the bitangents, which
/// a glTF tangent leaves to `cross(normal, tangent) * w`.)
fn evaluate(
    faces: &[usize],
    triangles: &[Triangle],
    vertex: usize,
    representative: &[usize],
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
) -> Vector {
    let n = normals[representative[vertex]].map(f64::from);
    let mut s = [0.0; 3];
    for &face in faces {
        let triangle = &triangles[face];
        // Only triangles with a say add to it.
        if triangle.flags & GROUP_WITH_ANY != 0 {
            continue;
        }
        let Some(at) = triangle.corners.iter().position(|&corner| corner == vertex) else { continue };
        let weight = {
            // The angle at the vertex between its triangle's two edges, in the vertex's tangent plane.
            let place = |welded: usize| positions[representative[welded]].map(f64::from);
            let before = place(triangle.corners[(at + 2) % 3]);
            let here = place(triangle.corners[at]);
            let after = place(triangle.corners[(at + 1) % 3]);
            let edge_before = projected(sub(before, here), n);
            let edge_after = projected(sub(after, here), n);
            dot(edge_before, edge_after).clamp(-1.0, 1.0).acos()
        };
        s = add(s, scale(weight, projected(triangle.s, n)));
    }
    normalized(s)
}

#[allow(clippy::cast_possible_truncation, reason = "narrowing f64 to the f32 a mesh holds")]
fn narrow(value: f64) -> f32 {
    value as f32
}

/// Gives `geometry` the tangents of its `corners` (as [`corners`] makes
/// them, for its own triangles): a vertex whose corners disagree (mirrored
/// texture coordinates meeting along it) is made as many as they differ, its
/// other attributes copied, and the triangles that used it moved to the one
/// with their tangent. A vertex no triangle uses gets a tangent from its normal.
pub fn split(geometry: &mut Geometry, corners: &[[f32; 4]]) {
    let count = geometry.positions.len();
    let mut tangents: Vec<[f32; 4]> = vec![[0.0; 4]; count];
    let mut assigned = vec![false; count];
    // The vertices a vertex was split into, by the bits of their tangent.
    let mut copies: HashMap<(usize, [u32; 4]), u32> = HashMap::new();
    let bits = |tangent: [f32; 4]| tangent.map(|value| (value + 0.0).to_bits());
    for (corner, &tangent) in corners.iter().enumerate() {
        let vertex = geometry.indices[corner] as usize;
        if !assigned[vertex] {
            assigned[vertex] = true;
            tangents[vertex] = tangent;
        } else if bits(tangents[vertex]) != bits(tangent) {
            let copy = *copies.entry((vertex, bits(tangent))).or_insert_with(|| {
                let made = geometry.positions.len();
                geometry.positions.push(geometry.positions[vertex]);
                geometry.normals.push(geometry.normals[vertex]);
                geometry.uvs.push(geometry.uvs[vertex]);
                if !geometry.joints.is_empty() {
                    geometry.joints.push(geometry.joints[vertex]);
                    geometry.weights.push(geometry.weights[vertex]);
                }
                tangents.push(tangent);
                u32::try_from(made).expect("a primitive's vertices are indexed by u32")
            });
            geometry.indices[corner] = copy;
        }
    }
    for vertex in 0..count {
        if !assigned[vertex] {
            let [x, y, z] = perpendicular(geometry.normals[vertex].map(f64::from)).map(narrow);
            tangents[vertex] = [x, y, z, 1.0];
        }
    }
    geometry.tangents = tangents;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit square in the xy plane facing +z, u along +x and v down the image
    /// (so up the plane): corners 0 to 3 from the bottom left, counterclockwise.
    struct Square {
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<u32>,
    }

    fn square() -> Square {
        Square {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            uvs: vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            indices: vec![0, 1, 2, 0, 2, 3],
        }
    }

    #[test]
    fn a_square_has_its_tangent_along_u_and_a_right_handed_frame() {
        let Square { positions, normals, uvs, indices } = square();
        for tangent in corners(&positions, &normals, &uvs, &indices) {
            assert_eq!(tangent, [1.0, 0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn mirroring_u_turns_the_tangent_along_and_flips_the_handedness() {
        let Square { positions, normals, mut uvs, indices } = square();
        for uv in &mut uvs {
            uv[0] = 1.0 - uv[0];
        }
        for tangent in corners(&positions, &normals, &uvs, &indices) {
            assert_eq!(tangent, [-1.0, 0.0, 0.0, -1.0]);
        }
        // Mirroring v instead leaves the tangent where it was.
        let Square { positions, normals, mut uvs, indices } = square();
        for uv in &mut uvs {
            uv[1] = 1.0 - uv[1];
        }
        for tangent in corners(&positions, &normals, &uvs, &indices) {
            assert_eq!(tangent, [1.0, 0.0, 0.0, -1.0]);
        }
    }

    #[test]
    fn a_triangle_with_no_area_in_texture_space_gives_a_frame_without_nans() {
        // Three coincident texture coordinates; a second triangle with a say shares an edge.
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
        let normals = vec![[0.0, 0.0, 1.0]; 4];
        let uvs = vec![[0.3, 0.3], [0.3, 0.3], [0.3, 0.3], [0.3, 0.3]];
        let indices = vec![0, 1, 2, 1, 3, 2];
        for tangent in corners(&positions, &normals, &uvs, &indices) {
            assert!(tangent.iter().all(|value| value.is_finite()), "{tangent:?}");
            let length = tangent[..3].iter().map(|value| f64::from(*value).powi(2)).sum::<f64>().sqrt();
            assert!((length - 1.0).abs() < 1e-6, "{tangent:?}");
            assert!(tangent[2].abs() < 1e-6 && tangent[3].abs() == 1.0, "perpendicular to the normal: {tangent:?}");
        }
    }
}
