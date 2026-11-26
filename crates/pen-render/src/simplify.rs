// Simpler levels of detail for meshes, made where a game's assets are
// packaged (`pen build`), with meshoptimizer's simplifier (the pure Rust
// port, meshopt-rs). Each level keeps the mesh's vertices and leaves out
// triangles, collapsing edges while the surface stays within an error of the
// whole mesh's; frames draw the simplest level whose error, seen from where
// they look, is under a pixel.
use anyhow::Result;

use crate::asset_store::{
    MAX_LODS, MESH_VERTEX_FLOATS, Mesh, is_packed_mesh, pack, parse_mesh, push_lod, quantize_mesh,
};

// The errors the simpler levels are made within, a share of the mesh's size
// (its bounding box's longest side): a level is drawn once its error is
// under a pixel, so these serve meshes under 400, 100 and 25 pixels across.
const ERRORS: [f32; MAX_LODS - 1] = [0.0025, 0.01, 0.04];

// Meshes with fewer triangles draw whole.
const LEAST_TRIANGLES: usize = 256;

// A level is kept only with at most this share of the last one's triangles.
const KEPT: f64 = 0.75;

/// Like `pack_mesh`, with up to three simpler levels of detail and pre-baked
/// GPU meshlets + continuous LOD Cluster DAG for static meshes so shipped native
/// and browser builds skip runtime meshlet/DAG construction. A mesh with an
/// explicit JSON `lods` list (including empty) keeps its authored LOD levels
/// while still baking meshlets/DAG; persistent instances and already-packed
/// binary `PMSH` meshes are preserved as authored.
pub fn pack_mesh_with_lods(bytes: &[u8]) -> Result<Vec<u8>> {
    if is_packed_mesh(bytes)? {
        return Ok(bytes.to_vec());
    }
    let mut mesh = parse_mesh(bytes)?;
    quantize_mesh(&mut mesh);
    if mesh.instances.is_some() {
        return pack(&mesh);
    }
    if !mesh.authored_lods && mesh.lods.len() == 1 && mesh.lods[0].count as usize / 3 >= LEAST_TRIANGLES {
        mesh = simplified(mesh)?;
    }
    if mesh.skin.is_none() && mesh.morphs.is_none() && mesh.meshlets.is_none() && mesh.meshlet_dag.is_none() {
        let (ml, dag) = crate::mesh_cache::build_meshlets_and_dag(&mesh);
        mesh.meshlets = ml;
        mesh.meshlet_dag = dag;
    }
    pack(&mesh)
}

fn simplified(mut mesh: Mesh) -> Result<Mesh> {
    let positions: Vec<[f32; 3]> = mesh
        .vertices
        .as_chunks::<MESH_VERTEX_FLOATS>()
        .0
        .iter()
        .map(|vertex| [vertex[0], vertex[1], vertex[2]])
        .collect();
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in &positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(p[axis]);
            max[axis] = max[axis].max(p[axis]);
        }
    }
    let size = (0..3).map(|axis| max[axis] - min[axis]).fold(0.0, f32::max);
    let whole = mesh.lods[0].count as usize;
    let mut last = whole;
    let mut scratch = vec![0; whole];
    for error in ERRORS {
        let count = meshopt_rs::simplify::simplify(&mut scratch, &mesh.indices[..whole], &positions, 0, error);
        #[allow(clippy::cast_precision_loss, reason = "a ratio of index counts")]
        let fewer = (count as f64) <= last as f64 * KEPT;
        if count >= 3 && fewer {
            push_lod(&mut mesh.lods, mesh.indices.len(), count, error * size)?;
            mesh.indices.extend_from_slice(&scratch[..count]);
            last = count;
        }
    }
    // Every LOD addresses the original vertices and all of their attributes.
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bumpy grid of `side` x `side` quads, as mesh JSON.
    pub(super) fn grid(side: u32) -> Vec<u8> {
        let (mut positions, mut normals, mut uvs, mut indices) = (vec![], vec![], vec![], vec![]);
        for z in 0..=side {
            for x in 0..=side {
                let (u, v) = (f64::from(x) / f64::from(side), f64::from(z) / f64::from(side));
                positions.extend([u, 0.001 * (u * 40.0).sin() * (v * 30.0).cos(), v]);
                normals.extend([0.0, 1.0, 0.0]);
                uvs.extend([u, v]);
            }
        }
        let at = |x: u32, z: u32| z * (side + 1) + x;
        for z in 0..side {
            for x in 0..side {
                indices.extend([at(x, z), at(x, z + 1), at(x + 1, z), at(x + 1, z), at(x, z + 1), at(x + 1, z + 1)]);
            }
        }
        serde_json::to_vec(
            &serde_json::json!({"positions": positions, "normals": normals, "uvs": uvs, "indices": indices}),
        )
        .unwrap()
    }

    #[test]
    fn packs_simpler_levels_of_large_meshes_only() {
        let packed = pack_mesh_with_lods(&grid(40)).unwrap();
        let mesh = parse_mesh(&packed).unwrap();
        assert!(mesh.lods.len() > 1, "{:?}", mesh.lods);
        // Each level has fewer triangles than the last, and a larger error,
        // over the same vertices.
        for pair in mesh.lods.windows(2) {
            assert!(f64::from(pair[1].count) <= f64::from(pair[0].count) * KEPT, "{:?}", mesh.lods);
            assert!(pair[1].error > pair[0].error);
        }
        assert_eq!(mesh.lods[0].count, 40 * 40 * 6);
        assert_eq!(mesh.vertices.len(), 41 * 41 * MESH_VERTEX_FLOATS);
        // Pre-baked meshlets and continuous LOD Cluster DAG are embedded in the packaged PMSH binary.
        assert!(mesh.meshlets.as_ref().is_some_and(|ml| ml.meshlets.len() > 1));
        assert!(mesh.meshlet_dag.as_ref().is_some_and(|dag| dag.level_count > 1 && dag.clusters.len() > 1));
        // Packing it again keeps its levels, meshlets, and DAG bit-for-bit.
        assert_eq!(pack_mesh_with_lods(&packed).unwrap(), packed);
        // A 128-triangle mesh keeps one discrete LOD, bakes meshlets, and skips the DAG (< 768 tris).
        let small = parse_mesh(&pack_mesh_with_lods(&grid(8)).unwrap()).unwrap();
        assert_eq!(small.lods.len(), 1);
        assert!(small.meshlets.as_ref().is_some_and(|ml| ml.meshlets.len() > 1));
        assert!(small.meshlet_dag.is_none());
        // A tiny mesh (< 128 tris) is left whole without meshlets or a DAG.
        let tiny = parse_mesh(&pack_mesh_with_lods(&grid(6)).unwrap()).unwrap();
        assert_eq!(tiny.lods.len(), 1);
        assert!(tiny.meshlets.is_none() && tiny.meshlet_dag.is_none());
    }

    #[test]
    fn simpler_levels_keep_the_vertices_tangents() {
        let mut json: serde_json::Value = serde_json::from_slice(&grid(40)).unwrap();
        let vertices = 41 * 41;
        let tangents: Vec<f32> =
            (0..vertices).flat_map(|n| [1.0, 0.0, 0.0, if n % 2 == 0 { 1.0 } else { -1.0 }]).collect();
        json["tangents"] = serde_json::json!(tangents);
        let packed = pack_mesh_with_lods(&serde_json::to_vec(&json).unwrap()).unwrap();
        let mesh = parse_mesh(&packed).unwrap();
        assert!(mesh.lods.len() > 1, "{:?}", mesh.lods);
        assert_eq!(mesh.tangents.as_deref(), Some(&tangents[..]));
        assert_eq!(pack_mesh_with_lods(&packed).unwrap(), packed);
    }

    #[test]
    fn precision_16_bakes_bounds_and_cones_from_the_decoded_geometry() {
        let mut json: serde_json::Value = serde_json::from_slice(&grid(40)).unwrap();
        // Tilt the vertex normals so the precision pass changes those as well as the positions.
        json["normals"] = serde_json::json!([0.001234567, 1.0, -0.002345678].repeat(41 * 41));
        json["precision"] = 16.into();
        let json = serde_json::to_vec(&json).unwrap();
        let original = parse_mesh(&json).unwrap();
        let packed = pack_mesh_with_lods(&json).unwrap();
        let mesh = parse_mesh(&packed).unwrap();
        let vertices = mesh.vertices.as_chunks::<MESH_VERTEX_FLOATS>().0;
        assert!(
            original.vertices.as_chunks::<MESH_VERTEX_FLOATS>().0.iter().zip(vertices).any(|(a, b)| a[..3] != b[..3])
        );
        assert_ne!(original.vertices[3..6], mesh.vertices[3..6]);

        let position = |i: u32| {
            let v = vertices[i as usize];
            [v[0], v[1], v[2]].map(f64::from)
        };
        let sub = |a: [f64; 3], b: [f64; 3]| std::array::from_fn(|axis| a[axis] - b[axis]);
        let length = |v: [f64; 3]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let epsilon = f64::from(f32::EPSILON);
        let check_sphere = |indices: &[u32], sphere: [f32; 4]| {
            let center = [sphere[0], sphere[1], sphere[2]].map(f64::from);
            for &index in indices {
                let distance = length(sub(position(index), center));
                assert!(distance <= f64::from(sphere[3]) + epsilon, "vertex {index} lies outside {sphere:?}");
            }
        };
        let check_cone = |indices: &[u32], cone: [f32; 4]| {
            // The culling cone bounds triangle normals from positions, not shading normals.
            if cone[3] >= 1.0 {
                return false;
            }
            let axis = [cone[0], cone[1], cone[2]].map(f64::from);
            for &[a, b, c] in indices.as_chunks::<3>().0 {
                let [x, y, z] = sub(position(b), position(a));
                let [u, v, w] = sub(position(c), position(a));
                let normal = [y * w - z * v, z * u - x * w, x * v - y * u];
                let cos_angle =
                    axis.iter().zip(normal).map(|(a, b)| a * b).sum::<f64>() / (length(normal) * length(axis));
                let sin_angle = (1.0 - cos_angle * cos_angle).max(0.0).sqrt();
                assert!(
                    cos_angle > 0.0 && sin_angle <= f64::from(cone[3]) + epsilon,
                    "triangle [{a}, {b}, {c}] lies outside {cone:?}"
                );
            }
            true
        };
        let range = |counts: [u32; 4]| counts[0] as usize..(counts[0] + counts[1]) as usize;
        let meshlets = mesh.meshlets.unwrap();
        let mut covered = vec![false; vertices.len()];
        let mut active_meshlet_cones = 0;
        let mut active_dag_cones = 0;
        for cluster in meshlets.meshlets {
            let indices = &meshlets.indices[range(cluster.counts)];
            check_sphere(indices, cluster.sphere);
            active_meshlet_cones += usize::from(check_cone(indices, cluster.cone_axis_cutoff));
            for &index in indices {
                covered[index as usize] = true;
            }
        }
        assert!(covered.into_iter().all(|seen| seen));

        let dag = mesh.meshlet_dag.unwrap();
        assert!(dag.level_count > 1);
        for cluster in dag.clusters {
            let indices = &dag.indices[range(cluster.counts)];
            check_sphere(indices, cluster.bounds_sphere);
            check_sphere(indices, cluster.parent_sphere);
            active_dag_cones += usize::from(check_cone(indices, cluster.cone_axis_cutoff));
        }
        assert!(active_meshlet_cones > 0 && active_dag_cones > 0);
    }

    #[test]
    fn simpler_levels_keep_secondary_uvs_and_vertex_colors() {
        let mut json: serde_json::Value = serde_json::from_slice(&grid(40)).unwrap();
        let uvs1: Vec<_> = (0..41 * 41).map(|i| [f64::from(i) / 1681.0, 0.25]).collect();
        let colors: Vec<_> = (0..41 * 41).map(|i| [0.25, f64::from(i) / 1681.0, 0.5, 1.0]).collect();
        json["uvs1"] = serde_json::json!(uvs1.as_flattened());
        json["colors"] = serde_json::json!(colors.as_flattened());
        let mesh = parse_mesh(&serde_json::to_vec(&json).unwrap()).unwrap();
        let (vertices, uvs1, colors) =
            (mesh.vertices.clone(), mesh.uvs1.clone().unwrap(), mesh.colors.clone().unwrap());
        let reduced = simplified(mesh).unwrap();
        assert!(reduced.lods.len() > 1);
        assert_eq!(reduced.vertices, vertices);
        assert_eq!(reduced.uvs1.unwrap(), uvs1);
        assert_eq!(reduced.colors.unwrap(), colors);
    }

    /// The levels the game's meshes get, and how long making them takes
    /// (docs/render-performance.md).
    #[test]
    #[ignore = "measures dist/gdev/assets; run with --ignored --nocapture in release"]
    fn game_mesh_levels() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dist/gdev/assets");
        let mut files: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.to_string_lossy().ends_with(".mesh.json"))
            .collect();
        files.sort();
        let (mut whole, mut packed, mut seconds) = (0, 0, 0.0);
        for path in files {
            let bytes = std::fs::read(&path).unwrap();
            let started = std::time::Instant::now();
            let lods = pack_mesh_with_lods(&bytes).unwrap();
            seconds += started.elapsed().as_secs_f64();
            whole += crate::pack_mesh(&bytes).unwrap().len();
            packed += lods.len();
            let mesh = parse_mesh(&lods).unwrap();
            if mesh.lods.len() > 1 {
                let levels: Vec<String> =
                    mesh.lods.iter().map(|lod| format!("{} ({:.4})", lod.count / 3, lod.error)).collect();
                eprintln!("{}: {}", path.file_name().unwrap().to_string_lossy(), levels.join(", "));
            }
        }
        eprintln!(
            "packed {:.1} MB without levels, {:.1} MB with, in {:.2} s",
            whole as f64 / 1e6,
            packed as f64 / 1e6,
            seconds
        );
    }
}

#[cfg(test)]
mod authored_tests {
    use super::*;

    #[test]
    fn explicit_empty_detail_list_and_repacked_binary_preserve_triangles_and_bounds() {
        let json = super::tests::grid(40);
        let plain = parse_mesh(&json).unwrap();
        assert!(plain.indices.len() / 3 > LEAST_TRIANGLES);
        let mut authored: serde_json::Value = serde_json::from_slice(&json).unwrap();
        authored["lods"] = serde_json::json!([]);
        let packed = pack_mesh_with_lods(&serde_json::to_vec(&authored).unwrap()).unwrap();
        let kept = parse_mesh(&packed).unwrap();
        assert_eq!(kept.lods.len(), 1);
        assert_eq!(kept.indices, plain.indices);
        assert_eq!(kept.vertices, plain.vertices);
        assert_eq!(kept.bounds.center, plain.bounds.center);
        assert_eq!(kept.bounds.radius, plain.bounds.radius);
        assert!(kept.meshlets.is_some() && kept.meshlet_dag.is_some());
        assert_eq!(pack_mesh_with_lods(&packed).unwrap(), packed);
        // A packed, unsimplified asset is authoritative even if its original
        // JSON did not opt out. Repackaging must not invent new geometry.
        let raw = crate::asset_store::pack(&plain).unwrap();
        assert_eq!(pack_mesh_with_lods(&raw).unwrap(), raw);
    }
}
