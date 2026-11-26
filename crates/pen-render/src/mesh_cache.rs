//! Content-hashed disk cache and background CPU preparation for mesh assets.
//!
//! Loader workers build meshlet partitions and QEM Cluster DAGs. Cached
//! meshlets and DAGs are keyed by a 128-bit content hash
//! of the LOD0 vertex positions, indices, and builder configuration, stored in
//! `~/.cache/pen/meshlets/` (when a cache directory is configured) via atomic
//! rename so concurrent runs never observe partial files.
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicU64, Ordering};

use crate::asset_store::{MESH_VERTEX_FLOATS, Mesh};
#[cfg(any(not(target_arch = "wasm32"), test))]
use crate::meshlet_dag::DagClusterGpu;
use crate::meshlet_dag::{MeshletDagConfig, MeshletDagMesh, build_meshlet_dag, build_meshlet_dag_from_leaves};
#[cfg(any(not(target_arch = "wasm32"), test))]
use crate::meshlets::{DEFAULT_MAX_MESHLET_TRIANGLES, DEFAULT_MAX_MESHLET_VERTICES, MeshletGpu};
use crate::meshlets::{MeshletMesh, build_meshlets};

#[cfg(any(not(target_arch = "wasm32"), test))]
const CACHE_MAGIC: [u8; 8] = *b"PENMLC01";
#[cfg(any(not(target_arch = "wasm32"), test))]
const FLAG_MESHLETS: u32 = 1 << 0;
#[cfg(any(not(target_arch = "wasm32"), test))]
const FLAG_MESHLET_DAG: u32 = 1 << 1;

/// A parsed mesh together with its CPU-derived acceleration and LOD structures,
/// built off the main thread and ready for GPU buffer upload.
pub(crate) struct PreparedMesh {
    pub mesh: Mesh,
    pub meshlets: Option<MeshletMesh>,
    pub meshlet_dag: Option<MeshletDagMesh>,
}

/// Cheap-to-clone handle to the on-disk meshlet/DAG cache directory.
#[derive(Clone, Default)]
pub(crate) struct MeshCache {
    #[cfg(not(target_arch = "wasm32"))]
    dir: Option<Arc<PathBuf>>,
}

impl MeshCache {
    /// Derives the sibling `meshlets/` cache directory from `RendererOptions::pipeline_cache`
    /// (`~/.cache/pen/pipelines` -> `~/.cache/pen/meshlets`), or disables caching when None.
    pub fn from_pipeline_cache(pipeline_cache: Option<&Path>) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let dir = pipeline_cache.map(|path| {
                if path.file_name().is_some_and(|name| name == "pipelines")
                    && let Some(parent) = path.parent()
                {
                    parent.join("meshlets")
                } else {
                    path.join("meshlets")
                }
            });
            Self::open(dir.as_deref())
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = pipeline_cache;
            Self {}
        }
    }

    #[cfg(any(not(target_arch = "wasm32"), test))]
    pub fn open(dir: Option<&Path>) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self { dir: dir.map(|path| Arc::new(path.to_path_buf())) }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = dir;
            Self {}
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn enabled(&self) -> bool {
        self.dir.is_some()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn entry_path(&self, hash: &[u8; 16]) -> Option<PathBuf> {
        let dir = self.dir.as_deref()?;
        Some(dir.join(format!("{}.bin", hex16(hash))))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load(&self, hash: &[u8; 16]) -> Option<(Option<MeshletMesh>, Option<MeshletDagMesh>)> {
        let path = self.entry_path(hash)?;
        let bytes = std::fs::read(&path).ok()?;
        decode_entry(&bytes, hash)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn store(&self, hash: &[u8; 16], meshlets: Option<&MeshletMesh>, dag: Option<&MeshletDagMesh>) {
        let Some(path) = self.entry_path(hash) else { return };
        let Some(bytes) = encode_entry(hash, meshlets, dag) else { return };
        static PARTIAL_SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = PARTIAL_SEQ.fetch_add(1, Ordering::Relaxed);
        let partial = path.with_extension(format!("{}.{seq}.partial", std::process::id()));
        let written = (|| {
            if let Some(directory) = path.parent() {
                std::fs::create_dir_all(directory)?;
            }
            std::fs::write(&partial, &bytes)?;
            std::fs::rename(&partial, &path)
        })();
        if let Err(error) = written {
            let _ = std::fs::remove_file(&partial);
            log::warn!("meshlet cache {}: {error}", path.display());
        }
    }
}

/// Builds GPU meshlets and (for meshes >= 768 triangles) a continuous LOD Cluster DAG
/// from a static unskinned mesh's LOD0 geometry. Shared by offline packaging (`pack_mesh_with_lods`)
/// and runtime cache misses (`prepare_mesh`).
#[cfg(any(feature = "simplify", test))]
pub(crate) fn build_meshlets_and_dag(mesh: &Mesh) -> (Option<MeshletMesh>, Option<MeshletDagMesh>) {
    if mesh.skin.is_some() || mesh.morphs.is_some() || mesh.instances.is_some() {
        return (None, None);
    }
    let Some(idx_slice) = mesh.lods.first().and_then(|lod0| {
        let start = lod0.first as usize;
        let end = start.checked_add(lod0.count as usize)?;
        mesh.indices.get(start..end)
    }) else {
        return (None, None);
    };
    if idx_slice.len() / 3 < 128 {
        return (None, None);
    }
    build_meshlets_and_dag_from_slice(&mesh_positions(&mesh.vertices), idx_slice, &MeshletDagConfig::default())
}

fn mesh_positions(vertices: &[f32]) -> Vec<[f32; 3]> {
    vertices.as_chunks::<MESH_VERTEX_FLOATS>().0.iter().map(|v| [v[0], v[1], v[2]]).collect()
}

fn build_meshlets_and_dag_from_slice(
    positions: &[[f32; 3]],
    idx_slice: &[u32],
    config: &MeshletDagConfig,
) -> (Option<MeshletMesh>, Option<MeshletDagMesh>) {
    let ml = build_meshlets(positions, idx_slice, config.max_vertices, config.max_triangles)
        .ok()
        .filter(|ml| ml.meshlets.len() > 1);
    let dag = (idx_slice.len() / 3 >= 768)
        .then(|| {
            let usable = |&[a, b, c]: &[u32; 3]| {
                let (Some(&pa), Some(&pb), Some(&pc)) =
                    (positions.get(a as usize), positions.get(b as usize), positions.get(c as usize))
                else {
                    return false;
                };
                let ab = std::array::from_fn::<_, 3, _>(|axis| f64::from(pb[axis]) - f64::from(pa[axis]));
                let ac = std::array::from_fn::<_, 3, _>(|axis| f64::from(pc[axis]) - f64::from(pa[axis]));
                let nx = ab[1] * ac[2] - ab[2] * ac[1];
                let ny = ab[2] * ac[0] - ab[0] * ac[2];
                let nz = ab[0] * ac[1] - ab[1] * ac[0];
                nx != 0.0 || ny != 0.0 || nz != 0.0
            };
            let mut triangles = idx_slice.as_chunks::<3>().0.iter();
            if let Some(first_bad) = triangles.position(|tri| !usable(tri)) {
                let mut clean_indices = Vec::with_capacity(idx_slice.len() - 3);
                clean_indices.extend_from_slice(&idx_slice[..first_bad * 3]);
                clean_indices.extend(triangles.filter(|tri| usable(tri)).flatten().copied());
                if clean_indices.len() / 3 < 768 {
                    return None;
                }
                build_meshlet_dag(positions, &clean_indices, config)
            } else if let Some(leaves) = &ml {
                build_meshlet_dag_from_leaves(positions, leaves, config)
            } else {
                build_meshlet_dag(positions, idx_slice, config)
            }
            .ok()
            .filter(|d| d.level_count > 1 && d.clusters.len() > 1)
        })
        .flatten();
    (ml, dag)
}

/// Prepares meshlets and the Cluster DAG for GPU upload.
/// Packaged `PMSH` meshes that already carry pre-baked meshlets/DAG skip building and
/// disk-cache I/O completely; unpacked dev meshes use `cache` on worker threads.
pub(crate) fn prepare_mesh(mut mesh: Mesh, cache: &MeshCache) -> PreparedMesh {
    let baked_ml = mesh.meshlets.take();
    let baked_dag = mesh.meshlet_dag.take();
    let (meshlets, meshlet_dag) = if baked_ml.is_some() || baked_dag.is_some() {
        (baked_ml, baked_dag)
    } else if mesh.skin.is_none() && mesh.morphs.is_none() && mesh.instances.is_none() {
        let lod0_slice = mesh.lods.first().and_then(|lod0| {
            let start = lod0.first as usize;
            let end = start.checked_add(lod0.count as usize)?;
            mesh.indices.get(start..end)
        });
        if let Some(idx_slice) = lod0_slice
            && idx_slice.len() / 3 >= 128
        {
            let config = MeshletDagConfig::default();
            #[cfg(not(target_arch = "wasm32"))]
            let hash = cache.enabled().then(|| hash_mesh_inputs(&mesh.vertices, idx_slice, &config));
            #[cfg(not(target_arch = "wasm32"))]
            let cached = hash.as_ref().and_then(|hash| cache.load(hash));
            #[cfg(target_arch = "wasm32")]
            let cached: Option<(Option<MeshletMesh>, Option<MeshletDagMesh>)> = {
                let _ = cache;
                None
            };
            if let Some(cached) = cached {
                cached
            } else {
                let (ml, dag) = build_meshlets_and_dag_from_slice(&mesh_positions(&mesh.vertices), idx_slice, &config);
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(hash) = &hash {
                    cache.store(hash, ml.as_ref(), dag.as_ref());
                }
                (ml, dag)
            }
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    PreparedMesh { mesh, meshlets, meshlet_dag }
}

#[cfg(not(target_arch = "wasm32"))]
fn hex16(bytes: &[u8; 16]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(32);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

/// Streaming MurmurHash3_x64_128 (stable across Rust versions, zero extra deps).
#[cfg(any(not(target_arch = "wasm32"), test))]
struct Hasher128 {
    h1: u64,
    h2: u64,
    len: u64,
    tail: [u8; 16],
    tail_len: usize,
}

#[cfg(any(not(target_arch = "wasm32"), test))]
impl Hasher128 {
    const C1: u64 = 0x87c3_7b91_1142_53d5;
    const C2: u64 = 0x4cf5_ad43_2745_937f;

    fn new(seed: u64) -> Self {
        Self { h1: seed, h2: seed, len: 0, tail: [0; 16], tail_len: 0 }
    }

    #[inline]
    fn fmix64(mut k: u64) -> u64 {
        k ^= k >> 33;
        k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
        k ^= k >> 33;
        k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        k ^ (k >> 33)
    }

    #[inline]
    fn mix_block(&mut self, block: &[u8; 16]) {
        let mut k1 = u64::from_le_bytes(block[0..8].try_into().unwrap());
        let mut k2 = u64::from_le_bytes(block[8..16].try_into().unwrap());
        k1 = k1.wrapping_mul(Self::C1).rotate_left(31).wrapping_mul(Self::C2);
        self.h1 ^= k1;
        self.h1 = self.h1.rotate_left(27).wrapping_add(self.h2).wrapping_mul(5).wrapping_add(0x52dc_e729);
        k2 = k2.wrapping_mul(Self::C2).rotate_left(33).wrapping_mul(Self::C1);
        self.h2 ^= k2;
        self.h2 = self.h2.rotate_left(31).wrapping_add(self.h1).wrapping_mul(5).wrapping_add(0x3849_5ab5);
    }

    fn update(&mut self, mut bytes: &[u8]) {
        self.len = self.len.wrapping_add(bytes.len() as u64);
        if self.tail_len > 0 {
            let take = (16 - self.tail_len).min(bytes.len());
            self.tail[self.tail_len..self.tail_len + take].copy_from_slice(&bytes[..take]);
            self.tail_len += take;
            bytes = &bytes[take..];
            if self.tail_len == 16 {
                let block = self.tail;
                self.tail_len = 0;
                self.mix_block(&block);
            }
        }
        let (chunks, rem) = bytes.as_chunks::<16>();
        for chunk in chunks {
            self.mix_block(chunk);
        }
        if !rem.is_empty() {
            self.tail[..rem.len()].copy_from_slice(rem);
            self.tail_len = rem.len();
        }
    }

    fn finish(mut self) -> [u8; 16] {
        let mut k1 = 0u64;
        let mut k2 = 0u64;
        for i in (8..self.tail_len).rev() {
            k2 = (k2 << 8) | u64::from(self.tail[i]);
        }
        if self.tail_len > 8 {
            k2 = k2.wrapping_mul(Self::C2).rotate_left(33).wrapping_mul(Self::C1);
            self.h2 ^= k2;
        }
        for i in (0..self.tail_len.min(8)).rev() {
            k1 = (k1 << 8) | u64::from(self.tail[i]);
        }
        if self.tail_len > 0 {
            k1 = k1.wrapping_mul(Self::C1).rotate_left(31).wrapping_mul(Self::C2);
            self.h1 ^= k1;
        }
        self.h1 ^= self.len;
        self.h2 ^= self.len;
        self.h1 = self.h1.wrapping_add(self.h2);
        self.h2 = self.h2.wrapping_add(self.h1);
        self.h1 = Self::fmix64(self.h1);
        self.h2 = Self::fmix64(self.h2);
        self.h1 = self.h1.wrapping_add(self.h2);
        self.h2 = self.h2.wrapping_add(self.h1);
        let mut out = [0u8; 16];
        out[0..8].copy_from_slice(&self.h1.to_le_bytes());
        out[8..16].copy_from_slice(&self.h2.to_le_bytes());
        out
    }
}

#[cfg(any(not(target_arch = "wasm32"), test))]
fn hash_mesh_inputs(vertices: &[f32], indices: &[u32], config: &MeshletDagConfig) -> [u8; 16] {
    let positions = vertices.as_chunks::<MESH_VERTEX_FLOATS>().0;
    let mut hasher = Hasher128::new(0x5045_4e4d_4c43_3031);
    hasher.update(&CACHE_MAGIC);
    let header: [u64; 8] = [
        DEFAULT_MAX_MESHLET_VERTICES as u64,
        DEFAULT_MAX_MESHLET_TRIANGLES as u64,
        config.max_vertices as u64,
        config.max_triangles as u64,
        u64::from(config.max_levels),
        config.target_group_size as u64,
        positions.len() as u64,
        indices.len() as u64,
    ];
    hasher.update(bytemuck::cast_slice(&header));
    for vertex in positions {
        hasher.update(bytemuck::cast_slice(&vertex[..3]));
    }
    hasher.update(bytemuck::cast_slice(indices));
    hasher.finish()
}

#[cfg(any(not(target_arch = "wasm32"), test))]
fn encode_entry(hash: &[u8; 16], meshlets: Option<&MeshletMesh>, dag: Option<&MeshletDagMesh>) -> Option<Vec<u8>> {
    let ml_bytes = meshlets.map_or(0, |m| 8 + m.meshlets.len() * size_of::<MeshletGpu>() + m.indices.len() * 4);
    let dag_bytes = dag.map_or(0, |d| 20 + d.clusters.len() * size_of::<DagClusterGpu>() + d.indices.len() * 4);
    let mut out = Vec::with_capacity(8 + 16 + 4 + ml_bytes + dag_bytes);
    out.extend_from_slice(&CACHE_MAGIC);
    out.extend_from_slice(hash);
    let flags = (u32::from(meshlets.is_some()) * FLAG_MESHLETS) | (u32::from(dag.is_some()) * FLAG_MESHLET_DAG);
    out.extend_from_slice(&flags.to_le_bytes());
    if let Some(m) = meshlets {
        out.extend_from_slice(&u32::try_from(m.meshlets.len()).ok()?.to_le_bytes());
        out.extend_from_slice(&u32::try_from(m.indices.len()).ok()?.to_le_bytes());
        out.extend_from_slice(bytemuck::cast_slice(&m.meshlets));
        out.extend_from_slice(bytemuck::cast_slice(&m.indices));
    }
    if let Some(d) = dag {
        out.extend_from_slice(&d.level_count.to_le_bytes());
        out.extend_from_slice(&d.leaf_cluster_count.to_le_bytes());
        out.extend_from_slice(&d.root_cluster_count.to_le_bytes());
        out.extend_from_slice(&u32::try_from(d.clusters.len()).ok()?.to_le_bytes());
        out.extend_from_slice(&u32::try_from(d.indices.len()).ok()?.to_le_bytes());
        out.extend_from_slice(bytemuck::cast_slice(&d.clusters));
        out.extend_from_slice(bytemuck::cast_slice(&d.indices));
    }
    Some(out)
}

#[cfg(any(not(target_arch = "wasm32"), test))]
fn decode_entry(bytes: &[u8], expected_hash: &[u8; 16]) -> Option<(Option<MeshletMesh>, Option<MeshletDagMesh>)> {
    fn take<'a>(cursor: &mut &'a [u8], len: usize) -> Option<&'a [u8]> {
        let (head, rest) = cursor.split_at_checked(len)?;
        *cursor = rest;
        Some(head)
    }
    fn take_u32(cursor: &mut &[u8]) -> Option<u32> {
        let head = take(cursor, 4)?;
        Some(u32::from_le_bytes(head.try_into().ok()?))
    }
    fn take_pod_vec<T: bytemuck::Pod>(cursor: &mut &[u8], count: usize) -> Option<Vec<T>> {
        let byte_len = count.checked_mul(size_of::<T>())?;
        let (head, rest) = cursor.split_at_checked(byte_len)?;
        *cursor = rest;
        let mut out = vec![T::zeroed(); count];
        bytemuck::cast_slice_mut(&mut out).copy_from_slice(head);
        Some(out)
    }

    let mut cursor = bytes;

    if take(&mut cursor, 8)? != CACHE_MAGIC {
        return None;
    }
    if take(&mut cursor, 16)? != expected_hash {
        return None;
    }
    let flags = take_u32(&mut cursor)?;
    if flags & !(FLAG_MESHLETS | FLAG_MESHLET_DAG) != 0 {
        return None;
    }
    let meshlets = if flags & FLAG_MESHLETS != 0 {
        let ml_count = take_u32(&mut cursor)? as usize;
        let idx_count = take_u32(&mut cursor)? as usize;
        let meshlets = take_pod_vec::<MeshletGpu>(&mut cursor, ml_count)?;
        let indices = take_pod_vec::<u32>(&mut cursor, idx_count)?;
        Some(MeshletMesh { meshlets, indices })
    } else {
        None
    };
    let dag = if flags & FLAG_MESHLET_DAG != 0 {
        let level_count = take_u32(&mut cursor)?;
        let leaf_cluster_count = take_u32(&mut cursor)?;
        let root_cluster_count = take_u32(&mut cursor)?;
        let cluster_count = take_u32(&mut cursor)? as usize;
        let idx_count = take_u32(&mut cursor)? as usize;
        let clusters = take_pod_vec::<DagClusterGpu>(&mut cursor, cluster_count)?;
        let indices = take_pod_vec::<u32>(&mut cursor, idx_count)?;
        Some(MeshletDagMesh { clusters, indices, level_count, leaf_cluster_count, root_cluster_count })
    } else {
        None
    };
    if !cursor.is_empty() {
        return None;
    }
    Some((meshlets, dag))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_store::{Lod, Mesh};
    use crate::culling::Sphere;

    fn sphere_mesh(rings: u32, sectors: u32) -> Mesh {
        let mut vertices = Vec::new();
        let push_vert = |vertices: &mut Vec<f32>, p: [f32; 3]| {
            vertices.extend_from_slice(&[p[0], p[1], p[2], p[0], p[1], p[2], 0.0, 0.0, 0.0, 0.0]);
        };
        push_vert(&mut vertices, [0.0, 0.0, 1.0]);
        for ring in 1..rings {
            let (sin_phi, cos_phi) = (std::f32::consts::PI * ring as f32 / rings as f32).sin_cos();
            for sector in 0..sectors {
                let (sin_theta, cos_theta) = (std::f32::consts::TAU * sector as f32 / sectors as f32).sin_cos();
                push_vert(&mut vertices, [sin_phi * cos_theta, sin_phi * sin_theta, cos_phi]);
            }
        }
        let south = u32::try_from(vertices.len() / MESH_VERTEX_FLOATS).unwrap();
        push_vert(&mut vertices, [0.0, 0.0, -1.0]);
        let index = |r, s| 1 + (r - 1) * sectors + s % sectors;
        let mut indices = Vec::new();
        for s in 0..sectors {
            indices.extend([0, index(1, s), index(1, s + 1)]);
            for r in 1..rings - 1 {
                let (a, b, c, d) = (index(r, s), index(r, s + 1), index(r + 1, s), index(r + 1, s + 1));
                indices.extend([a, c, b, b, c, d]);
            }
            indices.extend([index(rings - 1, s), south, index(rings - 1, s + 1)]);
        }
        let count = u32::try_from(indices.len()).unwrap();
        Mesh {
            vertices,
            indices,
            bounds: Sphere { center: [0.0; 3], radius: 1.0 },
            lods: vec![Lod { first: 0, count, error: 0.0 }],
            authored_lods: false,
            skin: None,
            instances: None,
            precision: None,
            tangents: None,
            morphs: None,
            uvs1: None,
            colors: None,
            meshlets: None,
            meshlet_dag: None,
        }
    }

    #[test]
    fn cache_round_trip_matches_cold_build_and_rejects_corruption() {
        let dir = std::env::temp_dir().join(format!("pen-mesh-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cache = MeshCache::open(Some(&dir));
        let cold = prepare_mesh(sphere_mesh(28, 32), &cache);
        assert!(cold.meshlets.is_some());
        assert!(cold.meshlet_dag.is_some());

        let warm = prepare_mesh(sphere_mesh(28, 32), &cache);
        let (ml_cold, ml_warm) = (cold.meshlets.as_ref().unwrap(), warm.meshlets.as_ref().unwrap());
        assert_eq!(bytemuck::cast_slice::<_, u8>(&ml_cold.meshlets), bytemuck::cast_slice::<_, u8>(&ml_warm.meshlets));
        assert_eq!(ml_cold.indices, ml_warm.indices);

        let (dag_cold, dag_warm) = (cold.meshlet_dag.as_ref().unwrap(), warm.meshlet_dag.as_ref().unwrap());
        assert_eq!(dag_cold.level_count, dag_warm.level_count);
        assert_eq!(dag_cold.leaf_cluster_count, dag_warm.leaf_cluster_count);
        assert_eq!(dag_cold.root_cluster_count, dag_warm.root_cluster_count);
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&dag_cold.clusters),
            bytemuck::cast_slice::<_, u8>(&dag_warm.clusters)
        );
        assert_eq!(dag_cold.indices, dag_warm.indices);

        // Corrupt the cache file and verify decode rejects it and prepare_mesh rebuilds cleanly.
        let entry = std::fs::read_dir(&dir).unwrap().next().unwrap().unwrap().path();
        let mut corrupted = std::fs::read(&entry).unwrap();
        corrupted.truncate(corrupted.len() / 2);
        std::fs::write(&entry, &corrupted).unwrap();
        let rebuilt = prepare_mesh(sphere_mesh(28, 32), &cache);
        assert_eq!(rebuilt.meshlet_dag.as_ref().unwrap().indices, dag_cold.indices);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_pipeline_cache_uses_sibling_meshlets_dir() {
        let dir = PathBuf::from("/tmp/example/pen/pipelines");
        let cache = MeshCache::from_pipeline_cache(Some(&dir));
        assert_eq!(cache.dir.as_deref().map(PathBuf::as_path), Some(Path::new("/tmp/example/pen/meshlets")));
        let custom = PathBuf::from("/tmp/custom-cache");
        let custom_cache = MeshCache::from_pipeline_cache(Some(&custom));
        assert_eq!(custom_cache.dir.as_deref().map(PathBuf::as_path), Some(Path::new("/tmp/custom-cache/meshlets")));
    }

    #[cfg(feature = "simplify")]
    #[test]
    fn prepare_mesh_uses_prebaked_meshlets_without_cache() {
        let mesh = sphere_mesh(28, 32);
        let mut positions = Vec::with_capacity(mesh.vertices.len() / MESH_VERTEX_FLOATS * 3);
        let mut normals = Vec::with_capacity(mesh.vertices.len() / MESH_VERTEX_FLOATS * 3);
        let mut uvs = Vec::with_capacity(mesh.vertices.len() / MESH_VERTEX_FLOATS * 2);
        for v in mesh.vertices.as_chunks::<MESH_VERTEX_FLOATS>().0 {
            positions.extend_from_slice(&v[0..3]);
            normals.extend_from_slice(&v[3..6]);
            uvs.extend_from_slice(&v[6..8]);
        }
        let json = serde_json::to_vec(&serde_json::json!({
            "positions": positions,
            "normals": normals,
            "uvs": uvs,
            "indices": mesh.indices,
        }))
        .unwrap();

        let packed = crate::simplify::pack_mesh_with_lods(&json).unwrap();
        let parsed = crate::asset_store::parse_mesh(&packed).unwrap();
        assert!(parsed.meshlets.is_some());
        assert!(parsed.meshlet_dag.is_some());

        let dir = std::env::temp_dir().join(format!("pen-mesh-prebaked-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cache = MeshCache::open(Some(&dir));
        let prepared = prepare_mesh(parsed, &cache);
        let cold = prepare_mesh(mesh, &MeshCache::default());

        assert!(!dir.exists(), "pre-baked meshes must not write to the runtime meshlet cache");
        let (ml_baked, ml_cold) = (prepared.meshlets.as_ref().unwrap(), cold.meshlets.as_ref().unwrap());
        assert_eq!(bytemuck::cast_slice::<_, u8>(&ml_baked.meshlets), bytemuck::cast_slice::<_, u8>(&ml_cold.meshlets));
        assert_eq!(ml_baked.indices, ml_cold.indices);

        let (dag_baked, dag_cold) = (prepared.meshlet_dag.as_ref().unwrap(), cold.meshlet_dag.as_ref().unwrap());
        assert_eq!(dag_baked.level_count, dag_cold.level_count);
        assert_eq!(dag_baked.leaf_cluster_count, dag_cold.leaf_cluster_count);
        assert_eq!(dag_baked.root_cluster_count, dag_cold.root_cluster_count);
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&dag_baked.clusters),
            bytemuck::cast_slice::<_, u8>(&dag_cold.clusters)
        );
        assert_eq!(dag_baked.indices, dag_cold.indices);
    }
}
