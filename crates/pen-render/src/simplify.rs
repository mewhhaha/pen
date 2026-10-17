// Simpler levels of detail for meshes, made where a game's assets are
// packaged (`pen build`), with meshoptimizer's simplifier (the pure Rust
// port, meshopt-rs). Each level keeps the mesh's vertices and leaves out
// triangles, collapsing edges while the surface stays within an error of the
// whole mesh's; frames draw the simplest level whose error, seen from where
// they look, is under a pixel.
use anyhow::Result;

use crate::asset_store::{MAX_LODS, MESH_VERTEX_FLOATS, Mesh, checked_mesh, pack, parse_mesh};

// The errors the simpler levels are made within, a share of the mesh's size
// (its bounding box's longest side): a level is drawn once its error is
// under a pixel, so these serve meshes under 400, 100 and 25 pixels across.
const ERRORS: [f32; MAX_LODS - 1] = [0.0025, 0.01, 0.04];

// Meshes with fewer triangles draw whole.
const LEAST_TRIANGLES: usize = 256;

// A level is kept only with at most this share of the last one's triangles.
const KEPT: f64 = 0.75;

/// Like `pack_mesh`, with up to three simpler levels of detail: each as
/// simple as the simplifier makes it within its error (a quarter percent,
/// one percent and four percent of the mesh's size), kept if it leaves out a
/// quarter of the last kept one's triangles. A mesh of fewer than 256
/// triangles, or one that has levels already, is packed as it is.
pub fn pack_mesh_with_lods(bytes: &[u8]) -> Result<Vec<u8>> {
    let mesh = parse_mesh(bytes)?;
    if mesh.lods.len() > 1 || mesh.lods[0].count as usize / 3 < LEAST_TRIANGLES {
        return pack(&mesh);
    }
    pack(&simplified(mesh)?)
}

fn simplified(mesh: Mesh) -> Result<Mesh> {
    let positions: Vec<[f32; 3]> = mesh
        .vertices
        .as_chunks::<MESH_VERTEX_FLOATS>()
        .0
        .iter()
        .map(|vertex| [vertex[0], vertex[1], vertex[2]])
        .collect();
    let size = (0..3)
        .map(|axis| {
            let along = positions.iter().map(|position| position[axis]);
            along.clone().fold(f32::NEG_INFINITY, f32::max) - along.fold(f32::INFINITY, f32::min)
        })
        .fold(0.0, f32::max);
    let whole = &mesh.indices[..mesh.lods[0].count as usize];
    let mut simpler: Vec<(f32, Vec<u32>)> = Vec::new();
    let mut last = whole.len();
    let mut scratch = vec![0; whole.len()];
    for error in ERRORS {
        let count = meshopt_rs::simplify::simplify(&mut scratch, whole, &positions, 0, error);
        #[allow(clippy::cast_precision_loss, reason = "a ratio of index counts")]
        let fewer = (count as f64) <= last as f64 * KEPT;
        if count >= 3 && fewer {
            simpler.push((error * size, scratch[..count].to_vec()));
            last = count;
        }
    }
    let whole = whole.to_vec();
    checked_mesh(mesh.vertices, whole, &simpler, mesh.skin)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bumpy grid of `side` x `side` quads, as mesh JSON.
    fn grid(side: u32) -> Vec<u8> {
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
        // Packing it again keeps its levels.
        assert_eq!(pack_mesh_with_lods(&packed).unwrap(), packed);
        // A small mesh is left whole.
        let small = parse_mesh(&pack_mesh_with_lods(&grid(8)).unwrap()).unwrap();
        assert_eq!(small.lods.len(), 1);
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
