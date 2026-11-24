//! Tangents: MikkTSpace-compatible ones made for a primitive that has a normal
//! map or an anisotropic material and no `TANGENT` attribute, unit, perpendicular
//! to the normal, along +u and right-handed (flipped where the texture is mirrored)
//! as a hand computation has them, a vertex split where mirrored coordinates meet
//! in it; the file's own kept, moved with the placement; skinned primitives too.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "test geometry: JSON numbers and indices to the f32 and u32 a mesh holds"
)]
mod common;

use common::*;
use pen_gltf::{Options, import, tangents};
use serde_json::{Value, json};

type Vector = [f64; 3];

fn sub(a: Vector, b: Vector) -> Vector {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(factor: f64, a: Vector) -> Vector {
    a.map(|value| factor * value)
}

fn dot(a: Vector, b: Vector) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vector, b: Vector) -> Vector {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn normalize(a: Vector) -> Vector {
    scale(1.0 / dot(a, a).sqrt(), a)
}

fn wide(a: [f32; 3]) -> Vector {
    a.map(f64::from)
}

/// A mesh's JSON arrays as vectors of arrays.
struct Mesh {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    tangents: Option<Vec<[f32; 4]>>,
    indices: Vec<u32>,
    json: Value,
}

fn arrays<const N: usize>(value: &Value) -> Vec<[f32; N]> {
    let numbers: Vec<f32> = value.as_array().unwrap().iter().map(|n| n.as_f64().unwrap() as f32).collect();
    numbers.as_chunks::<N>().0.to_vec()
}

fn mesh_of(path: &std::path::Path) -> Mesh {
    let json = read_json(path);
    Mesh {
        positions: arrays(&json["positions"]),
        normals: arrays(&json["normals"]),
        tangents: json.get("tangents").map(arrays::<4>),
        indices: json["indices"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap() as u32).collect(),
        json,
    }
}

/// The mesh of `surface` imported with the glTF `material` and node `placement`.
fn imported(
    name: &str,
    surface: &Surface,
    material: Value,
    extensions: &[&str],
    placement: Value,
) -> (Mesh, Vec<String>) {
    let directory = directory(name);
    let out = directory.join("assets");
    let file = model_file(&directory, name, model(surface, material, extensions, placement));
    let imported = import(&file, &out, &Options::default()).unwrap();
    (mesh_of(&out.join(&imported.meshes[0].mesh)), imported.notes)
}

fn normal_mapped() -> Value {
    json!({"name": "mapped", "normalTexture": {"index": 0}})
}

fn length(tangent: [f32; 4]) -> f64 {
    dot(wide([tangent[0], tangent[1], tangent[2]]), wide([tangent[0], tangent[1], tangent[2]])).sqrt()
}

/// Every tangent of `mesh` a unit vector (to 1e-5) perpendicular to its vertex's normal (to 1e-5), of handedness 1 or -1.
fn assert_frames(mesh: &Mesh) {
    let tangents = mesh.tangents.as_ref().expect("the mesh has tangents");
    assert_eq!(tangents.len(), mesh.positions.len());
    for (vertex, tangent) in tangents.iter().enumerate() {
        assert!((length(*tangent) - 1.0).abs() < 1e-5, "vertex {vertex}: {tangent:?} is not a unit vector");
        let along = dot(wide([tangent[0], tangent[1], tangent[2]]), wide(mesh.normals[vertex]));
        assert!(along.abs() < 1e-5, "vertex {vertex}: {tangent:?} is not perpendicular to its normal ({along})");
        assert!(tangent[3] == 1.0 || tangent[3] == -1.0, "vertex {vertex}: handedness {}", tangent[3]);
    }
}

#[test]
fn a_normal_mapped_primitive_gets_tangents_along_u_and_one_without_a_map_does_not() {
    let (mesh, notes) = imported("square", &square(), normal_mapped(), &[], json!({}));
    assert!(notes.is_empty(), "{notes:?}");
    assert_frames(&mesh);
    assert_eq!(mesh.tangents.unwrap(), vec![[1.0, 0.0, 0.0, 1.0]; 4], "u grows along +x and the frame is right-handed");
    // Nothing reads them without a normal map or anisotropy.
    let (mesh, _) = imported("no-map", &square(), json!({"name": "plain"}), &[], json!({}));
    assert!(mesh.tangents.is_none() && mesh.json.get("tangents").is_none());
    // Anisotropy wants them.
    let anisotropic = json!({"extensions": {"KHR_materials_anisotropy": {"anisotropyStrength": 0.5}}});
    let (mesh, _) = imported("aniso", &square(), anisotropic, &["KHR_materials_anisotropy"], json!({}));
    assert_eq!(mesh.tangents.unwrap(), vec![[1.0, 0.0, 0.0, 1.0]; 4]);
    // An anisotropy of 0 has no direction to read.
    let isotropic = json!({"extensions": {"KHR_materials_anisotropy": {"anisotropyStrength": 0.0}}});
    let (mesh, _) = imported("aniso-zero", &square(), isotropic, &["KHR_materials_anisotropy"], json!({}));
    assert!(mesh.tangents.is_none());
}

#[test]
fn the_renderer_reads_the_tangents_the_import_wrote() {
    let (mesh, _) = imported("read", &square(), normal_mapped(), &[], json!({}));
    let bytes = serde_json::to_vec(&mesh.json).unwrap();
    // pack_mesh checks them, and its binary form (flag 32) carries them.
    let packed = pen_render::pack_mesh(&bytes).unwrap();
    assert_eq!(packed[5] & 32, 32);
}

#[test]
fn mirroring_the_texture_flips_the_handedness_and_turns_the_tangent_along_u() {
    let mut mirrored = square();
    for uv in &mut mirrored.uvs {
        uv[0] = 1.0 - uv[0];
    }
    let (mesh, _) = imported("mirrored", &mirrored, normal_mapped(), &[], json!({}));
    assert_eq!(mesh.tangents.unwrap(), vec![[-1.0, 0.0, 0.0, -1.0]; 4]);
    // Flipping v alone leaves the direction along u and flips the frame.
    let mut flipped = square();
    for uv in &mut flipped.uvs {
        uv[1] = 1.0 - uv[1];
    }
    let (mesh, _) = imported("flipped", &flipped, normal_mapped(), &[], json!({}));
    assert_eq!(mesh.tangents.unwrap(), vec![[1.0, 0.0, 0.0, -1.0]; 4]);
}

#[test]
fn a_sphere_has_unit_tangents_perpendicular_to_its_normals_along_the_longitude() {
    let (mesh, _) = imported("sphere", &sphere(12, 16), normal_mapped(), &[], json!({}));
    assert_frames(&mesh);
    let tangents = mesh.tangents.as_ref().unwrap();
    let width = 17;
    let mut checked = 0;
    for (vertex, tangent) in tangents.iter().enumerate() {
        // u runs along the longitude: the tangent is dP/du, (-sin phi, 0, -cos phi), away from the poles.
        let (ring, segment) = (vertex / width, vertex % width);
        let phi = segment as f64 / 16.0 * std::f64::consts::TAU;
        let normal = wide(mesh.normals[vertex]);
        if ring == 0 || ring == 12 {
            continue;
        }
        let along = [-phi.sin(), 0.0, -phi.cos()];
        let got = wide([tangent[0], tangent[1], tangent[2]]);
        // A vertex has the mean of its triangles' tangents: on the seam, only the triangles on one
        // side of it, a half segment (11 degrees) off; elsewhere the two sides agree.
        let (seam, near_pole) = (segment == 0 || segment == 16, ring == 1 || ring == 11);
        let least = if seam {
            0.97
        } else if near_pole {
            0.99
        } else {
            0.999
        };
        assert!(dot(got, along) > least, "vertex {vertex} (ring {ring}, segment {segment}): {tangent:?} vs {along:?}");
        // The bitangent is where v' grows: up the sphere, toward the north pole.
        let bitangent = scale(f64::from(tangent[3]), cross(normal, got));
        assert!(bitangent[1] > 0.0, "vertex {vertex}: bitangent {bitangent:?}");
        checked += 1;
    }
    assert!(checked > 100, "{checked} vertices checked");
    // Every vertex has the right-handed frame: the sphere's texture is not mirrored anywhere.
    assert!(tangents.iter().all(|tangent| tangent[3] == 1.0));
}

/// Each corner's tangent a hand computation gives, for a surface whose vertices all lie in
/// one group each (every triangle keeps its winding in texture space): per triangle the
/// tangent `(e1 dv2 - e2 dv1) / det` and bitangent `(e2 du1 - e1 du2) / det` of the
/// textbook derivation (texture `v` flipped, so that it runs up the picture), projected into
/// each vertex's tangent plane, summed over the vertex's triangles by the angle each has
/// there, normalised; the handedness the side of `cross(normal, tangent)` the bitangent is on.
fn by_hand(positions: &[[f32; 3]], normals: &[[f32; 3]], uvs: &[[f32; 2]], indices: &[u32]) -> Vec<[f64; 4]> {
    let mut sums = vec![[0.0f64; 3]; positions.len()];
    let mut handed = vec![0.0f64; positions.len()];
    for triangle in indices.as_chunks::<3>().0.iter() {
        let p = [0, 1, 2].map(|corner| wide(positions[triangle[corner] as usize]));
        let uv = [0, 1, 2].map(|corner| {
            let [u, v] = uvs[triangle[corner] as usize];
            [f64::from(u), 1.0 - f64::from(v)]
        });
        let (e1, e2) = (sub(p[1], p[0]), sub(p[2], p[0]));
        let (du1, dv1) = (uv[1][0] - uv[0][0], uv[1][1] - uv[0][1]);
        let (du2, dv2) = (uv[2][0] - uv[0][0], uv[2][1] - uv[0][1]);
        let det = du1 * dv2 - du2 * dv1;
        let tangent = scale(1.0 / det, sub(scale(dv2, e1), scale(dv1, e2)));
        let bitangent = scale(1.0 / det, sub(scale(du1, e2), scale(du2, e1)));
        for corner in 0..3 {
            let vertex = triangle[corner] as usize;
            let n = wide(normals[vertex]);
            let flat = |a: Vector| normalize(sub(a, scale(dot(n, a), n)));
            let (before, after) = (p[(corner + 2) % 3], p[(corner + 1) % 3]);
            let angle = dot(flat(sub(before, p[corner])), flat(sub(after, p[corner]))).clamp(-1.0, 1.0).acos();
            let t = flat(tangent);
            sums[vertex] = [0, 1, 2].map(|axis| sums[vertex][axis] + angle * t[axis]);
            handed[vertex] += dot(cross(n, t), bitangent).signum();
        }
    }
    triangle_corners(indices)
        .map(|vertex| {
            let t = normalize(sums[vertex]);
            [t[0], t[1], t[2], handed[vertex].signum()]
        })
        .collect()
}

fn triangle_corners(indices: &[u32]) -> impl Iterator<Item = usize> + '_ {
    indices.iter().map(|&vertex| vertex as usize)
}

/// Two triangles of a bent quad, with texture coordinates that are neither axis aligned nor alike.
fn bent() -> Surface {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [1.2, 1.0, 0.5], [0.0, 1.0, 0.1]];
    let indices = vec![0u16, 1, 2, 0, 2, 3];
    // Each vertex faces as the mean of its triangles' faces.
    let mut sums = [[0.0f64; 3]; 4];
    for triangle in indices.as_chunks::<3>().0.iter() {
        let p = [0, 1, 2].map(|corner| wide(positions[triangle[corner] as usize]));
        let face = normalize(cross(sub(p[1], p[0]), sub(p[2], p[0])));
        for &vertex in triangle {
            sums[vertex as usize] = [0, 1, 2].map(|axis| sums[vertex as usize][axis] + face[axis]);
        }
    }
    let normals = sums.map(|sum| normalize(sum).map(|value| value as f32)).to_vec();
    Surface {
        positions,
        normals,
        uvs: vec![[0.1, 0.9], [0.8, 0.85], [0.9, 0.2], [0.05, 0.15]],
        indices,
        tangents: None,
    }
}

#[test]
fn tangents_follow_a_hand_computation_over_a_bent_quad_and_a_single_triangle() {
    let surface = bent();
    let indices: Vec<u32> = surface.indices.iter().map(|&index| u32::from(index)).collect();
    let made = tangents::corners(&surface.positions, &surface.normals, &surface.uvs, &indices);
    let wanted = by_hand(&surface.positions, &surface.normals, &surface.uvs, &indices);
    assert_eq!(made.len(), 6);
    for (corner, (got, want)) in made.iter().zip(&wanted).enumerate() {
        for axis in 0..4 {
            assert!(
                (f64::from(got[axis]) - want[axis]).abs() < 1e-5,
                "corner {corner}: made {got:?}, by hand {want:?}"
            );
        }
    }
    // The triangles really differ there: it is the angles' weighing that gives the shared vertices theirs.
    let alone = |triangle: [u32; 3]| by_hand(&surface.positions, &surface.normals, &surface.uvs, &triangle)[0];
    assert!((0..3).any(|axis| (alone([0, 1, 2])[axis] - alone([0, 2, 3])[axis]).abs() > 1e-3));
    // One triangle alone: its own tangent, projected.
    let single = tangents::corners(&surface.positions, &surface.normals, &surface.uvs, &[0, 1, 2]);
    let want = by_hand(&surface.positions, &surface.normals, &surface.uvs, &[0, 1, 2]);
    for (got, want) in single.iter().zip(&want) {
        for axis in 0..4 {
            assert!((f64::from(got[axis]) - want[axis]).abs() < 1e-5, "{got:?} vs {want:?}");
        }
    }
}

#[test]
fn texture_coordinates_that_meet_mirrored_split_the_vertices_where_they_meet() {
    // Two squares side by side, the right one's texture mirrored: u runs 0 to 0.5 over the left and
    // 0.5 back to 0 over the right, so the middle vertices (at x = 1) are equal in every attribute.
    let positions =
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]];
    let uvs = vec![[0.0, 1.0], [0.5, 1.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.0], [0.0, 0.0]];
    let surface = Surface {
        positions,
        normals: vec![[0.0, 0.0, 1.0]; 6],
        uvs,
        indices: vec![0, 1, 4, 0, 4, 3, 1, 2, 5, 1, 5, 4],
        tangents: None,
    };
    let (mesh, _) = imported("seam", &surface, normal_mapped(), &[], json!({}));
    assert_frames(&mesh);
    // The two middle vertices are two each; the corners unchanged.
    assert_eq!(mesh.positions.len(), 8);
    let tangents = mesh.tangents.as_ref().unwrap();
    for triangle in mesh.indices.as_chunks::<3>().0.iter() {
        let x = triangle.iter().map(|&vertex| mesh.positions[vertex as usize][0]).fold(0.0, f32::max);
        // The left square's triangles have x up to 1, the right's up to 2.
        let want = if x <= 1.0 { [1.0, 0.0, 0.0, 1.0] } else { [-1.0, 0.0, 0.0, -1.0] };
        for &vertex in triangle {
            assert_eq!(tangents[vertex as usize], want, "vertex {vertex} at {:?}", mesh.positions[vertex as usize]);
        }
    }
    // The unsplit vertices keep their places in the arrays.
    assert_eq!(&mesh.positions[..6], &surface.positions[..]);
}

#[test]
fn a_triangle_with_no_area_in_texture_space_gives_finite_unit_tangents() {
    let mut surface = square();
    // The second triangle's three texture coordinates are one point.
    surface.uvs = vec![[0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [0.5, 0.5]];
    let (mesh, _) = imported("degenerate", &surface, normal_mapped(), &[], json!({}));
    let tangents = mesh.tangents.as_ref().unwrap();
    assert!(tangents.iter().all(|tangent| tangent.iter().all(|value| value.is_finite())));
    assert_frames(&mesh);
    // And one with all of them one point has frames too, from the normals alone.
    surface.uvs = vec![[0.2, 0.2]; 4];
    let (mesh, _) = imported("all-degenerate", &surface, normal_mapped(), &[], json!({}));
    assert_frames(&mesh);
}

#[test]
fn a_primitive_without_texture_coordinates_gets_no_tangents_and_is_reported() {
    let directory = directory("untextured");
    let out = directory.join("assets");
    let (mut document, bytes) = model(&square(), normal_mapped(), &[], json!({}));
    document["meshes"][0]["primitives"][0]["attributes"].as_object_mut().unwrap().remove("TEXCOORD_0");
    let file = gltf_file(&directory.join("untextured.gltf"), document, &bytes);
    let imported = import(&file, &out, &Options::default()).unwrap();
    assert!(mesh_of(&out.join(&imported.meshes[0].mesh)).tangents.is_none());
    assert!(imported.notes.iter().any(|note| note.contains("no texture coordinates")), "{:?}", imported.notes);
}

fn with_tangents() -> Surface {
    let mut surface = square();
    // Tangents of the file's own, which are not what would be made.
    surface.tangents =
        Some(vec![[0.0, 1.0, 0.0, 1.0], [0.6, 0.8, 0.0, -1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, -1.0]]);
    surface
}

#[test]
fn the_files_own_tangents_are_kept_exactly() {
    // With a normal map, which would have them made...
    let (mesh, _) = imported("kept", &with_tangents(), normal_mapped(), &[], json!({}));
    assert_eq!(mesh.tangents, with_tangents().tangents);
    // ...and without one, as the file has them.
    let (mesh, _) = imported("kept-plain", &with_tangents(), json!({}), &[], json!({}));
    assert_eq!(mesh.tangents, with_tangents().tangents);
}

#[test]
fn the_files_own_tangents_move_with_the_placement_and_a_mirror_flips_the_handedness() {
    // Turned a quarter about z (x to y, y to -x) and moved: tangents turn, they do not move.
    let half = std::f64::consts::FRAC_1_SQRT_2;
    let turn = json!({"rotation": [0.0, 0.0, half, half], "translation": [5.0, 6.0, 7.0]});
    let (mesh, _) = imported("turned", &with_tangents(), normal_mapped(), &[], turn);
    let tangents = mesh.tangents.unwrap();
    let want = [[-1.0, 0.0, 0.0, 1.0], [-0.8, 0.6, 0.0, -1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, -1.0]];
    for (got, want) in tangents.iter().zip(want) {
        for axis in 0..4 {
            assert!((got[axis] - want[axis]).abs() < 1e-6, "{got:?} vs {want:?}");
        }
    }
    // Mirrored in x: the tangent's x flips, the handedness with it, and the winding is put right.
    let (mesh, _) =
        imported("mirrored-placement", &with_tangents(), normal_mapped(), &[], json!({"scale": [-1.0, 1.0, 1.0]}));
    let tangents = mesh.tangents.unwrap();
    let want = [[0.0, 1.0, 0.0, -1.0], [-0.6, 0.8, 0.0, 1.0], [0.0, 0.0, 1.0, -1.0], [-1.0, 0.0, 0.0, 1.0]];
    for (got, want) in tangents.iter().zip(want) {
        for axis in 0..4 {
            assert!((got[axis] - want[axis]).abs() < 1e-6, "{got:?} vs {want:?}");
        }
    }
    assert_eq!(mesh.indices[..3], [0, 2, 1], "the mirrored triangles are wound back");
    // Scaled: the length is made one again.
    let (mesh, _) = imported("scaled", &with_tangents(), normal_mapped(), &[], json!({"scale": [3.0, 3.0, 3.0]}));
    assert!(mesh.tangents.unwrap().iter().all(|tangent| (length(*tangent) - 1.0).abs() < 1e-6));
}

#[test]
fn a_mirrored_placement_of_made_tangents_agrees_with_the_mirrored_surface() {
    // Made after the placement, from the mirrored positions and normals and the wound-back
    // triangles: the frame is still right-handed against the texture.
    let (mesh, _) = imported("mirror-made", &square(), normal_mapped(), &[], json!({"scale": [-1.0, 1.0, 1.0]}));
    // u still grows along the surface's +x as the file's: now toward -x.
    assert_eq!(mesh.tangents.unwrap(), vec![[-1.0, 0.0, 0.0, -1.0]; 4]);
}

#[test]
fn skinned_primitives_get_tangents_too() {
    let directory = directory("skinned");
    let out = directory.join("assets");
    let (mut document, bytes) = arm();
    // The arm's red material has a normal map.
    document["materials"][0]["normalTexture"] = json!({"index": 0});
    let file = gltf_file(&directory.join("arm.gltf"), document, &bytes);
    let imported = import(&file, &out, &Options::default()).unwrap();
    let column = mesh_of(&out.join(&imported.meshes[0].mesh));
    assert!(column.json.get("skin").is_some());
    assert_eq!(column.positions.len(), 10);
    assert_frames(&column);
    // Its joints and weights are still a vertex's four.
    assert_eq!(column.json["joints"].as_array().unwrap().len(), 40);
    assert_eq!(column.json["weights"].as_array().unwrap().len(), 40);
    // u grows along +x across the column.
    assert!(column.tangents.unwrap().iter().all(|tangent| (tangent[0] - 1.0).abs() < 1e-6 && tangent[3] == 1.0));
    // The renderer takes it, rig and tangents.
    let packed = pen_render::pack_mesh(&std::fs::read(out.join(&imported.meshes[0].mesh)).unwrap()).unwrap();
    assert_eq!(packed[5] & (4 | 32), 4 | 32);
    // The plinth has no normal map: no tangents.
    assert!(mesh_of(&out.join(&imported.meshes[1].mesh)).tangents.is_none());
}
