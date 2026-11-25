//! Endpoint-preserving meshlet simplification and a GPU-selected continuous LOD cut.
//!
//! Each replacement group is atomic: its clusters share both LOD tests, and a
//! later replacement consumes the entire group. Group IDs are the first cluster
//! index in the corresponding group. Follow parent group IDs to walk ancestry.
#![allow(clippy::cast_possible_truncation, reason = "outward rounding and validated GPU counts")]

use std::collections::{BTreeSet, HashMap, HashSet};
use std::ops::Range;

use anyhow::{Result, ensure};
use wgpu::util::DeviceExt;

use crate::meshlets::{
    DEFAULT_MAX_MESHLET_TRIANGLES, DEFAULT_MAX_MESHLET_VERTICES, MeshletCullParams, MeshletIndirectDraw, MeshletMesh,
    build_meshlets,
};

pub const MESHLET_DAG_CUT_WGSL: &str = include_str!("shaders/meshlet_dag_cut.wgsl");
const NO_PARENT: u32 = u32::MAX;
const ROOT_ERROR: f32 = 1e30;
const ROOT_ERROR_MIN: f32 = 1e29;
const WORKGROUP_SIZE: u32 = 64;

#[derive(Clone, Copy, Debug)]
pub struct MeshletDagConfig {
    pub max_vertices: usize,
    pub max_triangles: usize,
    /// Maximum number of levels, counting the leaves.
    pub max_levels: u32,
    pub target_group_size: usize,
}

impl Default for MeshletDagConfig {
    fn default() -> Self {
        Self {
            max_vertices: DEFAULT_MAX_MESHLET_VERTICES,
            max_triangles: DEFAULT_MAX_MESHLET_TRIANGLES,
            max_levels: 8,
            target_group_size: 4,
        }
    }
}

/// 112-byte descriptor matching the shader's seven 16-byte vectors.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DagClusterGpu {
    pub bounds_sphere: [f32; 4],
    pub cone_axis_cutoff: [f32; 4],
    pub lod_sphere: [f32; 4],
    pub parent_sphere: [f32; 4],
    /// [lod_error, parent_error, lod_level, 0]
    pub errors: [f32; 4],
    /// [first_index, index_count, base_vertex_bits, tri_count]
    pub counts: [u32; 4],
    /// [group_id, parent_group_id, lod_level, unique_verts]
    pub meta: [u32; 4],
}

#[derive(Clone, Debug)]
pub struct MeshletDagMesh {
    pub clusters: Vec<DagClusterGpu>,
    pub indices: Vec<u32>,
    pub level_count: u32,
    pub leaf_cluster_count: u32,
    pub root_cluster_count: u32,
}

type Point = [f64; 3];
type Triangle = [u32; 3];
type Edge = [u32; 2];

fn sub(a: Point, b: Point) -> Point {
    std::array::from_fn(|i| a[i] - b[i])
}

fn dot(a: Point, b: Point) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}

fn cross(a: Point, b: Point) -> Point {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn triangle_normal(positions: &[[f32; 3]], [a, b, c]: Triangle) -> Point {
    let [a, b, c] = [a, b, c].map(|i| positions[i as usize].map(f64::from));
    cross(sub(b, a), sub(c, a))
}

fn center(sphere: [f32; 4]) -> Point {
    [sphere[0], sphere[1], sphere[2]].map(f64::from)
}

fn enclosing_sphere(spheres: &[[f32; 4]]) -> [f32; 4] {
    let midpoint = std::array::from_fn::<_, 3, _>(|axis| {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for s in spheres {
            min = min.min(f64::from(s[axis]) - f64::from(s[3]));
            max = max.max(f64::from(s[axis]) + f64::from(s[3]));
        }
        (0.5 * (min + max)) as f32
    });
    let mut radius: f64 = 0.0;
    for &s in spheres {
        let delta = sub(center(s), midpoint.map(f64::from));
        radius = radius.max(dot(delta, delta).sqrt() + f64::from(s[3]));
    }
    [midpoint[0], midpoint[1], midpoint[2], ((radius * (1.0 + 1e-6)) as f32).next_up()]
}

fn edges([a, b, c]: Triangle) -> [Edge; 3] {
    [[a.min(b), a.max(b)], [b.min(c), b.max(c)], [c.min(a), c.max(a)]]
}

fn triangles(indices: &[u32]) -> impl Iterator<Item = Triangle> + '_ {
    indices.as_chunks::<3>().0.iter().copied()
}

fn cluster_indices(indices: &[u32], counts: [u32; 4]) -> &[u32] {
    let start = counts[0] as usize;
    &indices[start..start + counts[1] as usize]
}

// Verify the reused meshlet cone against f64 normals and its stored f32 axis.
// This also covers tiny triangles omitted by the meshlet builder's normal epsilon.
fn conservative_cone(positions: &[[f32; 3]], indices: &[u32], mut cone: [f32; 4]) -> [f32; 4] {
    if cone[3] >= 1.0 {
        return cone;
    }
    let axis = center(cone);
    let length = dot(axis, axis).sqrt();
    let min_dot = triangles(indices)
        .map(|tri| {
            let normal = triangle_normal(positions, tri);
            dot(normal, axis) / (length * dot(normal, normal).sqrt())
        })
        .fold(1.0, f64::min);
    cone[3] = if min_dot <= 0.0 || length > 1.000_001 {
        1.0
    } else {
        // Bound the angular cutoff and accommodate rounding of the axis and shader dot products.
        ((length * (1.0 - min_dot * min_dot).max(0.0).sqrt() + 2e-6) as f32).next_up().max(cone[3]).min(1.0)
    };
    cone
}

// An active bundle is one complete replacement group, possibly at an earlier
// level if it could not merge in the previous round.
type Bundle = Range<usize>;

struct Frontier {
    edge_counts: HashMap<Edge, usize>,
    vertex_owners: HashMap<u32, usize>,
    vertices: Vec<HashSet<u32>>,
    neighbors: Vec<HashMap<usize, usize>>,
}

impl Frontier {
    fn new(mesh: &MeshletDagMesh, active: &[Bundle]) -> Self {
        let mut edge_counts = HashMap::new();
        let mut vertex_owners = HashMap::new();
        let mut vertices = vec![HashSet::new(); active.len()];
        let mut edge_owners = HashMap::<Edge, Vec<usize>>::new();
        for (owner, bundle) in active.iter().enumerate() {
            for cluster in &mesh.clusters[bundle.clone()] {
                for tri in triangles(cluster_indices(&mesh.indices, cluster.counts)) {
                    for vertex in tri {
                        if vertices[owner].insert(vertex) {
                            *vertex_owners.entry(vertex).or_insert(0) += 1;
                        }
                    }
                    for edge in edges(tri) {
                        *edge_counts.entry(edge).or_insert(0) += 1;
                        let owners = edge_owners.entry(edge).or_default();
                        if owners.last() != Some(&owner) {
                            owners.push(owner);
                        }
                    }
                }
            }
        }
        let mut neighbors = vec![HashMap::new(); active.len()];
        for (edge, owners) in edge_owners {
            // Manifold shared edges drive grouping. Nonmanifold seams stay locked.
            if edge_counts[&edge] == 2
                && let [a, b] = owners[..]
            {
                *neighbors[a].entry(b).or_insert(0) += 1;
                *neighbors[b].entry(a).or_insert(0) += 1;
            }
        }
        Self { edge_counts, vertex_owners, vertices, neighbors }
    }

    fn locked_vertices(&self, tris: &[Triangle], group: &[usize]) -> HashSet<u32> {
        let mut local = HashMap::new();
        for &triangle in tris {
            for edge in edges(triangle) {
                *local.entry(edge).or_insert(0) += 1;
            }
        }
        let mut locked = HashSet::new();
        for (edge, count) in local {
            if count != 2 || self.edge_counts[&edge] != 2 {
                locked.extend(edge);
            }
        }
        // Also preserve point contacts shared by otherwise edge-disconnected bundles.
        let mut owners = HashMap::new();
        for &bundle in group {
            for &vertex in &self.vertices[bundle] {
                *owners.entry(vertex).or_insert(0) += 1;
            }
        }
        for (vertex, count) in owners {
            if count != self.vertex_owners[&vertex] {
                locked.insert(vertex);
            }
        }
        locked
    }
}

#[derive(Clone, Copy, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    fn add_plane(&mut self, plane: [f64; 4]) {
        let mut lane = 0;
        for (i, &a) in plane.iter().enumerate() {
            for &b in &plane[i..] {
                self.0[lane] += a * b;
                lane += 1;
            }
        }
    }

    fn merged(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] + other.0[i]))
    }

    fn error(self, [x, y, z]: Point) -> f64 {
        let [xx, xy, xz, xw, yy, yz, yw, zz, zw, ww] = self.0;
        (xx * x * x
            + yy * y * y
            + zz * z * z
            + ww
            + 2.0 * (xy * x * y + xz * x * z + yz * y * z + xw * x + yw * y + zw * z))
            .max(0.0)
    }
}

// Pack group vertices locally for both collapse adjacency and build_meshlets,
// which allocates vertex adjacency proportional to the input vertex buffer.
struct GroupGeometry {
    original_ids: Vec<u32>,
    positions: Vec<[f32; 3]>,
    triangles: Vec<Triangle>,
}

impl GroupGeometry {
    fn new(positions: &[[f32; 3]], tris: Vec<Triangle>) -> Self {
        let mut original_ids = Vec::new();
        let mut local_ids = HashMap::new();
        let triangles = tris
            .into_iter()
            .map(|tri| {
                tri.map(|original| {
                    *local_ids.entry(original).or_insert_with(|| {
                        let local = original_ids.len() as u32;
                        original_ids.push(original);
                        local
                    })
                })
            })
            .collect();
        let positions = original_ids.iter().map(|&i| positions[i as usize]).collect();
        Self { original_ids, positions, triangles }
    }

    fn simplify(&mut self, locked: &HashSet<u32>, origin: Point) -> Option<f64> {
        let points: Vec<_> = self.positions.iter().map(|p| sub(p.map(f64::from), origin)).collect();
        let locked: Vec<_> = self.original_ids.iter().map(|id| locked.contains(id)).collect();
        let mut quadrics = vec![Quadric::default(); points.len()];
        for &tri in &self.triangles {
            let n = triangle_normal(&self.positions, tri);
            let length = dot(n, n).sqrt();
            let n = n.map(|x| x / length);
            let plane = [n[0], n[1], n[2], -dot(n, points[tri[0] as usize])];
            for vertex in tri {
                quadrics[vertex as usize].add_plane(plane);
            }
        }

        let original = self.triangles.len();
        let target = original.div_ceil(2);
        let mut max_error: f64 = 0.0;
        while self.triangles.len() > target {
            let mut edge_counts = HashMap::new();
            let mut incident = vec![Vec::new(); points.len()];
            for (i, &tri) in self.triangles.iter().enumerate() {
                for v in tri {
                    incident[v as usize].push(i);
                }
                for edge in edges(tri) {
                    *edge_counts.entry(edge).or_insert(0) += 1;
                }
            }
            let mut candidates = Vec::new();
            for ([a, b], count) in edge_counts {
                if count != 2 || (locked[a as usize] && locked[b as usize]) {
                    continue;
                }
                let quadric = quadrics[a as usize].merged(quadrics[b as usize]);
                for (keep, remove) in [(a, b), (b, a)] {
                    if !locked[remove as usize] {
                        candidates.push((quadric.error(points[keep as usize]), keep, remove));
                    }
                }
            }
            candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| (a.1, a.2).cmp(&(b.1, b.2))));
            let Some((error, keep, remove)) =
                candidates.into_iter().find(|&(_, keep, remove)| self.can_collapse(&incident, &locked, keep, remove))
            else {
                break;
            };
            quadrics[keep as usize] = quadrics[keep as usize].merged(quadrics[remove as usize]);
            max_error = max_error.max(error.sqrt());
            self.triangles.retain_mut(|tri| {
                if tri.contains(&remove) && tri.contains(&keep) {
                    return false;
                }
                *tri = tri.map(|v| if v == remove { keep } else { v });
                true
            });
        }
        (self.triangles.len() < original).then_some(max_error)
    }

    fn can_collapse(&self, incident: &[Vec<usize>], locked: &[bool], keep: u32, remove: u32) -> bool {
        let around = |vertex: u32| incident[vertex as usize].iter().map(|&i| self.triangles[i]);
        let keep_neighbors: HashSet<_> = around(keep).flatten().filter(|&v| v != keep).collect();
        let remove_neighbors: HashSet<_> = around(remove).flatten().filter(|&v| v != remove).collect();
        // A new edge between two locked vertices could already exist in a
        // neighboring group or be created there by a concurrent replacement.
        if locked[keep as usize]
            && remove_neighbors.iter().any(|&v| v != keep && locked[v as usize] && !keep_neighbors.contains(&v))
        {
            return false;
        }
        // The link condition rules out pinching unrelated parts of the surface.
        let opposite: HashSet<_> =
            around(remove).filter(|tri| tri.contains(&keep)).flatten().filter(|&v| v != keep && v != remove).collect();
        if opposite.len() != 2 || keep_neighbors.intersection(&remove_neighbors).any(|v| !opposite.contains(v)) {
            return false;
        }

        let face_key = |mut tri: Triangle| {
            tri.sort_unstable();
            tri
        };
        let mut faces: HashSet<_> = around(keep).filter(|tri| !tri.contains(&remove)).map(face_key).collect();
        for tri in around(remove).filter(|tri| !tri.contains(&keep)) {
            let updated = tri.map(|v| if v == remove { keep } else { v });
            if !faces.insert(face_key(updated))
                || dot(triangle_normal(&self.positions, tri), triangle_normal(&self.positions, updated)) <= 0.0
            {
                return false;
            }
        }
        true
    }
}

fn take_group(
    free: &mut BTreeSet<usize>,
    active: &[Bundle],
    frontier: &Frontier,
    mesh: &MeshletDagMesh,
    budget: usize,
) -> Vec<usize> {
    let seed = free.pop_first().unwrap();
    let mut group = vec![seed];
    let mut size = active[seed].len();
    let mut adjacent = HashMap::new();
    while size < budget {
        for (&neighbor, &edges) in &frontier.neighbors[*group.last().unwrap()] {
            *adjacent.entry(neighbor).or_insert(0) += edges;
        }
        let origin = center(mesh.clusters[active[seed].start].lod_sphere);
        let fits = |id: usize| active[id].len() <= budget - size;
        // Consider edge-connected candidates first. The frontier inherits the
        // meshlet builder's spatial ordering for the bounded proximity fallback.
        let candidates = adjacent.keys().copied().filter(|&id| free.contains(&id) && fits(id));
        let distance = |id: usize| {
            let delta = sub(center(mesh.clusters[active[id].start].lod_sphere), origin);
            dot(delta, delta)
        };
        let proximity = |a: &usize, b: &usize| distance(*a).total_cmp(&distance(*b)).then_with(|| a.cmp(b));
        let best = candidates
            .min_by(|a, b| adjacent[b].cmp(&adjacent[a]).then_with(|| proximity(a, b)))
            .or_else(|| free.iter().take(32).copied().filter(|&id| fits(id)).min_by(proximity));
        let Some(best) = best else { break };
        free.remove(&best);
        size += active[best].len();
        group.push(best);
    }
    group
}

fn append_group(
    mesh: &mut MeshletDagMesh,
    partition: MeshletMesh,
    positions: &[[f32; 3]],
    lod: Option<([f32; 4], f32)>,
    level: u32,
) -> Result<Bundle> {
    let start = mesh.clusters.len();
    let offset = u32::try_from(mesh.indices.len())?;
    ensure!(u32::try_from(start + partition.meshlets.len()).is_ok(), "DAG cluster count exceeds u32");
    ensure!(u32::try_from(mesh.indices.len() + partition.indices.len()).is_ok(), "DAG index count exceeds u32");
    for cluster in partition.meshlets {
        let bounds = cluster.sphere;
        ensure!(bounds.into_iter().all(f32::is_finite), "meshlet bounds exceed f32");
        let indices = cluster_indices(&partition.indices, cluster.counts);
        let index = mesh.clusters.len() as u32;
        let (lod_sphere, error) = lod.unwrap_or((bounds, 0.0));
        let mut counts = cluster.counts;
        counts[0] += offset;
        mesh.clusters.push(DagClusterGpu {
            bounds_sphere: bounds,
            cone_axis_cutoff: conservative_cone(positions, indices, cluster.cone_axis_cutoff),
            lod_sphere,
            parent_sphere: lod_sphere,
            errors: [error, ROOT_ERROR, level as f32, 0.0],
            counts,
            meta: [if lod.is_some() { start as u32 } else { index }, NO_PARENT, level, cluster.meta[0]],
        });
    }
    mesh.indices.extend(partition.indices);
    Ok(start..mesh.clusters.len())
}

fn replace_group(
    mesh: &mut MeshletDagMesh,
    positions: &[[f32; 3]],
    config: &MeshletDagConfig,
    frontier: &Frontier,
    active: &[Bundle],
    group: &[usize],
    level: u32,
) -> Result<Option<Bundle>> {
    let children: Vec<_> = group.iter().flat_map(|&id| active[id].clone()).collect();
    let mut spheres = Vec::new();
    let mut child_error = 0.0_f32;
    let mut tris = Vec::new();
    for &i in &children {
        let child = &mesh.clusters[i];
        spheres.push(child.lod_sphere);
        child_error = child_error.max(child.errors[0]);
        tris.extend(triangles(cluster_indices(&mesh.indices, child.counts)));
    }
    let locked = frontier.locked_vertices(&tris, group);
    let mut geometry = GroupGeometry::new(positions, tris);
    let input_sphere = enclosing_sphere(&spheres);
    let Some(collapse_error) = geometry.simplify(&locked, center(input_sphere)) else { return Ok(None) };
    let simplified = geometry.triangles.as_flattened();
    if simplified.len() / 3 > 2 * config.max_triangles {
        return Ok(None);
    }
    let mut partition = build_meshlets(&geometry.positions, simplified, config.max_vertices, config.max_triangles)?;
    if partition.meshlets.len() > 2 {
        return Ok(None);
    }
    spheres.extend(partition.meshlets.iter().map(|m| m.sphere));
    let sphere = enclosing_sphere(&spheres);
    let error = ((f64::from(child_error) + collapse_error.max(1e-4 * f64::from(sphere[3]))) as f32).next_up();
    ensure!(error < ROOT_ERROR_MIN, "DAG error exceeds the supported f32 range");
    for i in &mut partition.indices {
        *i = geometry.original_ids[*i as usize];
    }
    let parents = append_group(mesh, partition, positions, Some((sphere, error)), level)?;
    for i in children {
        let child = &mut mesh.clusters[i];
        child.parent_sphere = sphere;
        child.errors[1] = error;
        child.meta[1] = parents.start as u32;
    }
    Ok(Some(parents))
}

/// Build a DAG whose index buffer addresses the caller's original vertices.
/// Open edges, shared group seams and point contacts are preserved at all cuts.
pub fn build_meshlet_dag(positions: &[[f32; 3]], indices: &[u32], config: &MeshletDagConfig) -> Result<MeshletDagMesh> {
    ensure!(config.max_levels >= 1, "max_levels must be at least 1");
    ensure!(config.target_group_size >= 2, "target_group_size must be at least 2");
    ensure!(
        indices.iter().all(|&i| { positions.get(i as usize).is_some_and(|p| p.iter().all(|x| x.is_finite())) }),
        "mesh indices must address finite positions"
    );
    ensure!(triangles(indices).all(|tri| triangle_normal(positions, tri) != [0.0; 3]), "mesh has degenerate triangles");
    let leaves = build_meshlets(positions, indices, config.max_vertices, config.max_triangles)?;
    let mut mesh = MeshletDagMesh {
        clusters: Vec::new(),
        indices: Vec::new(),
        level_count: 0,
        leaf_cluster_count: 0,
        root_cluster_count: 0,
    };
    let range = append_group(&mut mesh, leaves, positions, None, 0)?;
    mesh.leaf_cluster_count = range.len() as u32;
    mesh.level_count = u32::from(!range.is_empty());
    let mut active: Vec<_> = range.map(|i| i..i + 1).collect();
    for level in 1..config.max_levels {
        if active.first().is_none_or(|b| active.len() == 1 && b.len() == 1) {
            break;
        }
        let frontier = Frontier::new(&mesh, &active);
        let mut free: BTreeSet<_> = (0..active.len()).collect();
        let mut next = Vec::new();
        let before = mesh.clusters.len();
        while !free.is_empty() {
            let mut group = take_group(&mut free, &active, &frontier, &mesh, config.target_group_size);
            loop {
                if (group.len() > 1 || active[group[0]].len() > 1)
                    && let Some(parents) =
                        replace_group(&mut mesh, positions, config, &frontier, &active, &group, level)?
                {
                    next.push(parents);
                    break;
                }
                if group.len() == 1 {
                    next.push(active[group[0]].clone());
                    break;
                }
                // A smaller replacement can fit when boundary locks prevent a
                // large group from reaching the two-parent budget.
                free.insert(group.pop().unwrap());
            }
        }
        if mesh.clusters.len() == before {
            break;
        }
        mesh.level_count = level + 1;
        active = next;
    }
    mesh.root_cluster_count = active.iter().map(Bundle::len).sum::<usize>() as u32;
    Ok(mesh)
}

pub struct MeshletDagGpu {
    pub clusters_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    pub cluster_count: u32,
    pub index_count: u32,
    pub level_count: u32,
    pub leaf_cluster_count: u32,
    pub root_cluster_count: u32,
}

impl MeshletDagGpu {
    pub fn upload(device: &wgpu::Device, mesh: &MeshletDagMesh) -> Result<Self> {
        let clusters = bytemuck::cast_slice(&mesh.clusters);
        let indices = bytemuck::cast_slice(&mesh.indices);
        let limits = device.limits();
        ensure!(
            clusters.len() as u64 <= limits.max_storage_buffer_binding_size.min(limits.max_buffer_size),
            "DAG cluster buffer too large"
        );
        ensure!(indices.len() as u64 <= limits.max_buffer_size, "DAG index buffer too large");
        let cluster_count = u32::try_from(mesh.clusters.len())?;
        let index_count = u32::try_from(mesh.indices.len())?;
        let upload = |label, bytes: &[u8], empty: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if bytes.is_empty() { empty } else { bytes },
                usage,
            })
        };
        Ok(Self {
            clusters_buf: upload(
                "meshlet_dag_clusters",
                clusters,
                &[0; size_of::<DagClusterGpu>()],
                wgpu::BufferUsages::STORAGE,
            ),
            index_buf: upload("meshlet_dag_indices", indices, &[0; 4], wgpu::BufferUsages::INDEX),
            cluster_count,
            index_count,
            level_count: mesh.level_count,
            leaf_cluster_count: mesh.leaf_cluster_count,
            root_cluster_count: mesh.root_cluster_count,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MeshletDagCullParams {
    pub cull: MeshletCullParams,
    /// Column-major perspective projection, used to derive pixels per unit of projected error.
    pub proj: [f32; 16],
    pub viewport_height: f32,
    pub near_plane: f32,
    pub pixel_error_threshold: f32,
}

/// Dispatch outputs. The first draw_count records are packed in GPU execution
/// order. The remaining indirect records are zeroed for fixed-capacity draws.
pub struct MeshletDagCullOutput {
    pub draws_buf: wgpu::Buffer,
    pub draw_count_buf: wgpu::Buffer,
    /// One u32 cluster index per emitted draw, in the same order. Read only the live prefix.
    pub selected_clusters_buf: wgpu::Buffer,
    capacity: u32,
}

impl MeshletDagCullOutput {
    pub fn new(device: &wgpu::Device, mesh: &MeshletDagGpu) -> Self {
        let capacity = mesh.cluster_count;
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let indirect = storage | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST;
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        let slots = u64::from(capacity.max(1));
        Self {
            draws_buf: buffer("meshlet_dag_draws", slots * size_of::<MeshletIndirectDraw>() as u64, indirect),
            draw_count_buf: buffer("meshlet_dag_draw_count", 4, indirect),
            selected_clusters_buf: buffer("meshlet_dag_selected_clusters", slots * 4, storage),
            capacity,
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CullUniform {
    model: [f32; 16],
    inv_model: [f32; 16],
    hiz_vp: [f32; 16],
    planes: [[f32; 4]; 6],
    camera_pos: [f32; 4],
    counts_flags: [u32; 4],
    projection: [f32; 4],
}

// The maximum absolute row sum of M^T M bounds its largest eigenvalue.
// Its square root bounds the largest singular value of M, including shear.
fn model_scale(model: &[f32; 16]) -> f32 {
    let columns = [0, 4, 8].map(|i| [model[i], model[i + 1], model[i + 2]].map(f64::from));
    let row_sum = |a| columns.iter().map(|&b| dot(a, b).abs()).sum::<f64>();
    let bound = columns.into_iter().map(row_sum).fold(0.0, f64::max).sqrt();
    (bound as f32).next_up()
}

pub struct MeshletDagCuller {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    dummy_hiz: wgpu::TextureView,
}

impl MeshletDagCuller {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let storage = |binding, read_only| buffer(binding, wgpu::BufferBindingType::Storage { read_only });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("meshlet_dag_bindings"),
            entries: &[
                buffer(0, wgpu::BufferBindingType::Uniform),
                storage(1, true),
                storage(2, false),
                storage(3, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage(5, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("meshlet_dag_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("meshlet_dag_cut.wgsl"),
            source: wgpu::ShaderSource::Wgsl(MESHLET_DAG_CUT_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("meshlet_dag_cut"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cull_meshlet_dag"),
            compilation_options: Default::default(),
            cache: None,
        });
        let dummy_hiz = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("meshlet_dag_dummy_hiz"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        Self { layout, pipeline, dummy_hiz }
    }

    /// Encode one cut for an affine model. A Hi-Z view, when supplied, uses the
    /// same depth convention and world-to-clip matrix as the meshlet culler.
    pub fn cull(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &MeshletDagGpu,
        output: &MeshletDagCullOutput,
        params: &MeshletDagCullParams,
        hiz: Option<&wgpu::TextureView>,
    ) -> Result<()> {
        ensure!(output.capacity >= mesh.cluster_count, "DAG output buffer has insufficient capacity");
        let p = &params.cull;
        let scale = model_scale(&p.model);
        let proj_scale = 0.5 * params.viewport_height * params.proj[5].abs();
        ensure!(
            scale.is_finite()
                && proj_scale.is_finite()
                && proj_scale > 0.0
                && params.near_plane.is_finite()
                && params.near_plane > 0.0
                && params.pixel_error_threshold.is_finite()
                && params.pixel_error_threshold >= 0.0,
            "invalid DAG cull scale, projection, near plane or error threshold"
        );
        encoder.clear_buffer(&output.draw_count_buf, 0, None);
        encoder.clear_buffer(&output.draws_buf, 0, None);
        if mesh.cluster_count == 0 {
            return Ok(());
        }
        let hiz = hiz.filter(|_| p.hiz_enabled);
        let hiz_mips = hiz.map_or(0, |v| p.hiz_mips.min(v.texture().mip_level_count()));
        let uniform = CullUniform {
            model: p.model,
            inv_model: crate::matrix::invert(&p.model),
            hiz_vp: p.hiz_vp,
            planes: p.frustum_planes,
            camera_pos: [p.camera_pos[0], p.camera_pos[1], p.camera_pos[2], 0.0],
            counts_flags: [
                mesh.cluster_count,
                u32::from(hiz.is_some()),
                hiz_mips,
                u32::from(p.backface_cull) | (u32::from(p.reversed_z) << 1),
            ],
            projection: [scale, proj_scale, params.near_plane, params.pixel_error_threshold],
        };
        // A separate immutable uniform per dispatch supports multiple cuts in one encoder.
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("meshlet_dag_uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meshlet_dag_cut"),
            layout: &self.layout,
            entries: &[
                entry(0, &uniform),
                entry(1, &mesh.clusters_buf),
                entry(2, &output.draws_buf),
                entry(3, &output.draw_count_buf),
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(hiz.unwrap_or(&self.dummy_hiz)),
                },
                entry(5, &output.selected_clusters_buf),
            ],
        });
        let groups = mesh.cluster_count.div_ceil(WORKGROUP_SIZE);
        let width = groups.min(device.limits().max_compute_workgroups_per_dimension);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width, groups.div_ceil(width), 1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::IDENTITY;

    fn uv_sphere() -> (Vec<[f32; 3]>, Vec<u32>) {
        let (rings, sectors) = (28u32, 32u32);
        let mut positions = vec![[0.0, 0.0, 1.0]];
        for ring in 1..rings {
            let (sin_phi, cos_phi) = (std::f32::consts::PI * ring as f32 / rings as f32).sin_cos();
            for sector in 0..sectors {
                let (sin_theta, cos_theta) = (std::f32::consts::TAU * sector as f32 / sectors as f32).sin_cos();
                positions.push([sin_phi * cos_theta, sin_phi * sin_theta, cos_phi]);
            }
        }
        let south = positions.len() as u32;
        positions.push([0.0, 0.0, -1.0]);
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
        (positions, indices)
    }

    fn closed_cut(mesh: &MeshletDagMesh, ids: &[u32]) {
        let mut incidence = HashMap::new();
        for &i in ids {
            for [a, b, c] in triangles(cluster_indices(&mesh.indices, mesh.clusters[i as usize].counts)) {
                for (a, b) in [(a, b), (b, c), (c, a)] {
                    let count = incidence.entry((a.min(b), a.max(b))).or_insert([0, 0]);
                    count[usize::from(a < b)] += 1;
                }
            }
        }
        assert!(incidence.values().all(|&count| count == [1, 1]), "cut has a crack or an overlapping edge");
    }

    #[test]
    fn cpu_hierarchy_preserves_geometry_and_monotonic_replacement_groups() {
        let (positions, indices) = uv_sphere();
        assert!(indices.len() / 3 >= 1600);
        let config = MeshletDagConfig::default();
        let mesh = build_meshlet_dag(&positions, &indices, &config).unwrap();
        assert!(mesh.leaf_cluster_count >= 12);
        assert!(mesh.level_count >= 3, "levels: {}", mesh.level_count);
        assert!(mesh.indices.iter().all(|&i| (i as usize) < positions.len()));
        assert!(triangles(&mesh.indices).all(|tri| triangle_normal(&positions, tri) != [0.0; 3]));
        assert_eq!(size_of::<DagClusterGpu>(), 112);
        assert_eq!(align_of::<DagClusterGpu>(), 16);

        let mut roots = Vec::new();
        for (i, cluster) in mesh.clusters.iter().enumerate() {
            let leader = &mesh.clusters[cluster.meta[0] as usize];
            assert_eq!(cluster.meta[..3], leader.meta[..3]);
            assert_eq!(cluster.lod_sphere, leader.lod_sphere);
            assert_eq!(cluster.parent_sphere, leader.parent_sphere);
            assert_eq!(cluster.errors, leader.errors);
            assert!(cluster.counts[3] as usize <= config.max_triangles);
            assert!(cluster.meta[3] as usize <= config.max_vertices);
            if i < mesh.leaf_cluster_count as usize {
                assert_eq!(cluster.errors[0], 0.0);
            }
            if cluster.meta[1] == NO_PARENT {
                roots.push(i as u32);
                assert!(cluster.errors[1] >= ROOT_ERROR_MIN);
                continue;
            }
            let parent = &mesh.clusters[cluster.meta[1] as usize];
            assert_eq!(parent.meta[0], cluster.meta[1]);
            assert!(parent.meta[2] > cluster.meta[2]);
            assert_eq!(cluster.parent_sphere, parent.lod_sphere);
            assert_eq!(cluster.errors[1], parent.errors[0]);
            assert!(cluster.errors[1] > cluster.errors[0]);
            let delta = sub(center(cluster.parent_sphere), center(cluster.lod_sphere));
            assert!(dot(delta, delta).sqrt() + f64::from(cluster.lod_sphere[3]) < f64::from(cluster.parent_sphere[3]));
        }
        assert_eq!(roots.len(), mesh.root_cluster_count as usize);
        closed_cut(&mesh, &roots);
    }

    fn vulkan_device() -> (wgpu::Device, wgpu::Queue) {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("headless Vulkan adapter required");
        assert_eq!(adapter.get_info().backend, wgpu::Backend::Vulkan);
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap()
    }

    fn view(distance: f32, x: f32) -> MeshletDagCullParams {
        let (near, far) = (0.01, 30_000.0);
        let proj = [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            far / (far - near),
            1.0,
            0.0,
            0.0,
            -near * far / (far - near),
            0.0,
        ];
        MeshletDagCullParams {
            cull: MeshletCullParams {
                model: IDENTITY,
                hiz_vp: IDENTITY,
                frustum_planes: [
                    [1.0, 0.0, 1.0, distance - x],
                    [-1.0, 0.0, 1.0, distance + x],
                    [0.0, 1.0, 1.0, distance],
                    [0.0, -1.0, 1.0, distance],
                    [0.0, 0.0, 1.0, distance - near],
                    [0.0, 0.0, -1.0, far - distance],
                ],
                camera_pos: [x, 0.0, -distance],
                hiz_enabled: false,
                hiz_mips: 0,
                backface_cull: false,
                reversed_z: false,
            },
            proj,
            viewport_height: 4096.0,
            near_plane: near,
            pixel_error_threshold: 1.5,
        }
    }

    fn readback(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, src: &wgpu::Buffer) -> wgpu::Buffer {
        let dst = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: src.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(src, 0, &dst, 0, src.size());
        dst
    }

    fn read<T: bytemuck::Pod>(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<T> {
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range().unwrap();
        let values = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        buffer.unmap();
        values
    }

    fn dispatch(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &MeshletDagMesh,
        views: &[(MeshletDagCullParams, Option<&wgpu::TextureView>)],
    ) -> Vec<(Vec<u32>, u32)> {
        let gpu = MeshletDagGpu::upload(device, mesh).unwrap();
        let output = MeshletDagCullOutput::new(device, &gpu);
        let culler = MeshletDagCuller::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let results: Vec<_> = views
            .iter()
            .map(|(params, hiz)| {
                culler.cull(device, &mut encoder, &gpu, &output, params, *hiz).unwrap();
                [
                    readback(device, &mut encoder, &output.draw_count_buf),
                    readback(device, &mut encoder, &output.selected_clusters_buf),
                    readback(device, &mut encoder, &output.draws_buf),
                ]
            })
            .collect();
        queue.submit(Some(encoder.finish()));
        results
            .iter()
            .map(|[count, ids, draws]| {
                let count = read::<u32>(device, count)[0] as usize;
                let ids = read::<u32>(device, ids)[..count].to_vec();
                let draws = read::<MeshletIndirectDraw>(device, draws);
                let mut tris = 0;
                for (&id, draw) in ids.iter().zip(&draws) {
                    let cluster = mesh.clusters[id as usize];
                    assert_ne!(draw.index_count, 0);
                    assert_eq!(draw.index_count, cluster.counts[1]);
                    assert_eq!(draw.first_index, cluster.counts[0]);
                    assert_eq!(draw.base_vertex, 0);
                    assert_eq!(draw.instance_count, 1);
                    assert_eq!(draw.first_instance, 0);
                    tris += draw.index_count / 3;
                }
                assert!(draws[count..].iter().all(|draw| draw.instance_count == 0));
                (ids, tris)
            })
            .collect()
    }

    fn exact_cut(mesh: &MeshletDagMesh, selected: &[u32]) {
        let selected_set: HashSet<_> = selected.iter().copied().collect();
        assert_eq!(selected_set.len(), selected.len());
        let groups: HashSet<_> = selected.iter().map(|&i| mesh.clusters[i as usize].meta[0]).collect();
        for (i, cluster) in mesh.clusters.iter().enumerate() {
            // A multi-cluster replacement must select every sibling together.
            assert_eq!(groups.contains(&cluster.meta[0]), selected_set.contains(&(i as u32)));
        }
        for leaf in &mesh.clusters[..mesh.leaf_cluster_count as usize] {
            let mut group = leaf.meta[0];
            let mut coverage = 0;
            while group != NO_PARENT {
                coverage += usize::from(groups.contains(&group));
                group = mesh.clusters[group as usize].meta[1];
            }
            assert_eq!(coverage, 1, "leaf group {} is uncovered or overlaps an ancestor", leaf.meta[0]);
        }
        closed_cut(mesh, selected);
    }

    #[test]
    fn vulkan_selects_exact_near_mixed_far_and_offscreen_cuts() {
        let (positions, indices) = uv_sphere();
        let mesh = build_meshlet_dag(&positions, &indices, &MeshletDagConfig::default()).unwrap();
        let (device, queue) = vulkan_device();
        // Disable facing for this witness to inspect the complete cut of the closed mesh.
        let mut views =
            [(3.0, 0.0), (250.0, 0.0), (10_000.0, 0.0), (3.0, 20.0), (250.0, 0.0)].map(|(d, x)| (view(d, x), None));
        views[4].0.cull.model[4] = 3.0;
        let results = dispatch(&device, &queue, &mesh, &views);
        let [(near, near_tris), (mid, mid_tris), (far, far_tris), (offscreen, _), (sheared, sheared_tris)] =
            results.as_slice()
        else {
            unreachable!()
        };
        eprintln!(
            "levels={}, roots={}, near={near_tris}, mid={mid_tris}, far={far_tris}",
            mesh.level_count, mesh.root_cluster_count
        );
        assert_eq!(near.len(), mesh.leaf_cluster_count as usize);
        assert!(near.iter().all(|&i| i < mesh.leaf_cluster_count));
        assert_eq!(*near_tris as usize, indices.len() / 3);
        let mid_levels: HashSet<_> = mid.iter().map(|&i| mesh.clusters[i as usize].meta[2]).collect();
        assert!(mid_levels.len() > 1, "middle distance must select mixed LODs: {mid_levels:?}");
        assert!(near_tris > mid_tris && mid_tris > far_tris);
        assert_eq!(far.len(), mesh.root_cluster_count as usize);
        assert!(far.iter().all(|&i| mesh.clusters[i as usize].meta[1] == NO_PARENT));
        assert!(offscreen.is_empty());
        assert!(sheared_tris > mid_tris, "shear must increase the projected error and refine the cut");
        for cut in [near, mid, far, sheared] {
            exact_cut(&mesh, cut);
        }
    }

    #[test]
    fn vulkan_culls_backfaces_and_respects_hiz_holes_in_both_depth_conventions() {
        let (device, queue) = vulkan_device();
        let positions = [[-0.05, -0.05, 0.0], [0.0, 0.05, 0.0], [0.05, -0.05, 0.0]];
        let mesh = build_meshlet_dag(&positions, &[0, 1, 2], &MeshletDagConfig::default()).unwrap();
        let texture = |depths: [f32; 4]| {
            let size = wgpu::Extent3d { width: 2, height: 2, depth_or_array_layers: 1 };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                texture.as_image_copy(),
                bytemuck::cast_slice(&depths),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8), rows_per_image: None },
                size,
            );
            texture.create_view(&wgpu::TextureViewDescriptor::default())
        };
        for reversed_z in [false, true] {
            let (near_depth, clear_depth) = if reversed_z { (0.9, 0.0) } else { (0.1, 1.0) };
            let occluder = texture([near_depth; 4]);
            let hole = texture([near_depth, near_depth, near_depth, clear_depth]);
            let mut front = view(1.0, 0.0);
            front.cull.model[0] = 2.0;
            front.cull.model[4] = 1.0;
            front.cull.model[14] = 0.5;
            front.cull.backface_cull = true;
            front.cull.reversed_z = reversed_z;
            if reversed_z {
                front.cull.hiz_vp[10] = -1.0;
                front.cull.hiz_vp[14] = 1.0;
            }
            let mut back = front;
            back.cull.camera_pos[2] = 1.0;
            let mut depth = front;
            depth.cull.hiz_mips = 1;
            depth.cull.hiz_enabled = true;
            let views = [(front, None), (back, None), (depth, Some(&occluder)), (depth, Some(&hole))];
            let results = dispatch(&device, &queue, &mesh, &views);
            assert_eq!(results.iter().map(|(_, tris)| *tris).collect::<Vec<_>>(), [1, 0, 0, 1]);
        }
    }
}
