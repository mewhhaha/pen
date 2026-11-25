use anyhow::{Result, ensure};

/// World-space triangle fed into the BVH builder.
#[derive(Clone, Copy, Debug)]
pub struct BvhTriangle {
    pub v0: [f32; 3],
    pub v1: [f32; 3],
    pub v2: [f32; 3],
}

/// Cache-aligned 32-byte GPU BVH node with explicit `u32` child/leaf fields (no float/u32 bitcasts).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BvhNodeGpu {
    pub aabb_min: [f32; 3],
    pub left_or_first: u32,
    pub aabb_max: [f32; 3],
    pub tri_count: u32,
}

/// GPU triangle record storing all 3 vertices for watertight Woop-Benthin-Wald intersection.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BvhTriGpu {
    pub v0: [f32; 4],
    pub v1: [f32; 4],
    pub v2: [f32; 4],
}

#[derive(Clone, Copy)]
pub(super) struct Aabb {
    pub(super) min: [f32; 3],
    pub(super) max: [f32; 3],
}

impl Aabb {
    pub(super) fn empty() -> Self {
        Self { min: [f32::INFINITY; 3], max: [f32::NEG_INFINITY; 3] }
    }
    pub(super) fn grow_pt(&mut self, p: [f32; 3]) {
        for (k, &v) in p.iter().enumerate() {
            self.min[k] = self.min[k].min(v);
            self.max[k] = self.max[k].max(v);
        }
    }
    pub(super) fn grow_box(&mut self, o: &Aabb) {
        for k in 0..3 {
            self.min[k] = self.min[k].min(o.min[k]);
            self.max[k] = self.max[k].max(o.max[k]);
        }
    }
    fn surface_area(&self) -> f32 {
        let ex = (self.max[0] - self.min[0]).max(0.0);
        let ey = (self.max[1] - self.min[1]).max(0.0);
        let ez = (self.max[2] - self.min[2]).max(0.0);
        2.0 * (ex * ey + ey * ez + ez * ex)
    }
}

fn tri_area2(t: &BvhTriangle) -> f64 {
    let e1 = [f64::from(t.v1[0] - t.v0[0]), f64::from(t.v1[1] - t.v0[1]), f64::from(t.v1[2] - t.v0[2])];
    let e2 = [f64::from(t.v2[0] - t.v0[0]), f64::from(t.v2[1] - t.v0[1]), f64::from(t.v2[2] - t.v0[2])];
    let cx = e1[1] * e2[2] - e1[2] * e2[1];
    let cy = e1[2] * e2[0] - e1[0] * e2[2];
    let cz = e1[0] * e2[1] - e1[1] * e2[0];
    cx * cx + cy * cy + cz * cz
}

/// Validated, acyclic flattened SAH-binned BVH (depth `<= 28`, guaranteed safe for 32-deep GPU stack).
#[derive(Clone, Debug)]
pub struct BvhScene {
    nodes: Vec<BvhNodeGpu>,
    tris: Vec<BvhTriGpu>,
}

impl BvhScene {
    pub fn nodes(&self) -> &[BvhNodeGpu] {
        &self.nodes
    }

    pub fn tris(&self) -> &[BvhTriGpu] {
        &self.tris
    }

    /// Builds a validated flat SAH-binned BVH over `triangles`, filtering zero-area degenerates
    /// and falling back to balanced median splits on coincident centroids.
    pub fn build(triangles: &[BvhTriangle]) -> Result<Self> {
        let mut valid_tris = Vec::with_capacity(triangles.len());
        for t in triangles {
            ensure!(
                t.v0.iter().chain(t.v1.iter()).chain(t.v2.iter()).all(|v| v.is_finite()),
                "BVH triangle contains non-finite vertex coordinate"
            );
            if tri_area2(t) > 1e-24 {
                valid_tris.push(*t);
            }
        }
        if valid_tris.is_empty() {
            return Ok(Self { nodes: Vec::new(), tris: Vec::new() });
        }
        let mut order: Vec<usize> = (0..valid_tris.len()).collect();
        let centroids: Vec<[f32; 3]> = valid_tris
            .iter()
            .map(|t| {
                [
                    (t.v0[0] + t.v1[0] + t.v2[0]) / 3.0,
                    (t.v0[1] + t.v1[1] + t.v2[1]) / 3.0,
                    (t.v0[2] + t.v1[2] + t.v2[2]) / 3.0,
                ]
            })
            .collect();
        let tri_boxes: Vec<Aabb> = valid_tris
            .iter()
            .map(|t| {
                let mut b = Aabb::empty();
                b.grow_pt(t.v0);
                b.grow_pt(t.v1);
                b.grow_pt(t.v2);
                b
            })
            .collect();

        let mut nodes = vec![BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 }];
        let mut out_tris = Vec::with_capacity(valid_tris.len());

        #[allow(clippy::too_many_arguments)]
        fn build_rec(
            node_idx: usize,
            start: usize,
            end: usize,
            depth: u32,
            order: &mut [usize],
            centroids: &[[f32; 3]],
            tri_boxes: &[Aabb],
            triangles: &[BvhTriangle],
            nodes: &mut Vec<BvhNodeGpu>,
            out_tris: &mut Vec<BvhTriGpu>,
        ) {
            let mut bounds = Aabb::empty();
            let mut c_bounds = Aabb::empty();
            for &ti in &order[start..end] {
                bounds.grow_box(&tri_boxes[ti]);
                c_bounds.grow_pt(centroids[ti]);
            }
            let count = end - start;
            if count <= 4 || depth >= 28 {
                let first = out_tris.len() as u32;
                for &ti in &order[start..end] {
                    let t = triangles[ti];
                    out_tris.push(BvhTriGpu {
                        v0: [t.v0[0], t.v0[1], t.v0[2], 0.0],
                        v1: [t.v1[0], t.v1[1], t.v1[2], 0.0],
                        v2: [t.v2[0], t.v2[1], t.v2[2], 0.0],
                    });
                }
                nodes[node_idx] = BvhNodeGpu {
                    aabb_min: bounds.min,
                    left_or_first: first,
                    aabb_max: bounds.max,
                    tri_count: count as u32,
                };
                return;
            }
            let ext = [
                c_bounds.max[0] - c_bounds.min[0],
                c_bounds.max[1] - c_bounds.min[1],
                c_bounds.max[2] - c_bounds.min[2],
            ];
            let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
                0
            } else if ext[1] >= ext[2] {
                1
            } else {
                2
            };

            let mid = if ext[axis] <= 1e-6 {
                // Balanced fallback prevents arbitrarily huge leaves when centroids coincide.
                start + count / 2
            } else {
                const BINS: usize = 8;
                let mut bin_box = [Aabb::empty(); BINS];
                let mut bin_cnt = [0usize; BINS];
                let scale = BINS as f32 / ext[axis];
                for &ti in &order[start..end] {
                    let b = (((centroids[ti][axis] - c_bounds.min[axis]) * scale) as usize).min(BINS - 1);
                    bin_cnt[b] += 1;
                    bin_box[b].grow_box(&tri_boxes[ti]);
                }
                let mut best_cost = f32::INFINITY;
                let mut best_split = 0usize;
                for split in 1..BINS {
                    let (mut l_box, mut r_box) = (Aabb::empty(), Aabb::empty());
                    let (mut l_cnt, mut r_cnt) = (0usize, 0usize);
                    for b in 0..split {
                        l_cnt += bin_cnt[b];
                        l_box.grow_box(&bin_box[b]);
                    }
                    for b in split..BINS {
                        r_cnt += bin_cnt[b];
                        r_box.grow_box(&bin_box[b]);
                    }
                    if l_cnt == 0 || r_cnt == 0 {
                        continue;
                    }
                    let cost = l_cnt as f32 * l_box.surface_area() + r_cnt as f32 * r_box.surface_area();
                    if cost < best_cost {
                        best_cost = cost;
                        best_split = split;
                    }
                }
                let split_pos = if best_split == 0 {
                    c_bounds.min[axis] + 0.5 * ext[axis]
                } else {
                    c_bounds.min[axis] + (best_split as f32 / BINS as f32) * ext[axis]
                };
                let mut m = start;
                for i in start..end {
                    if centroids[order[i]][axis] < split_pos {
                        order.swap(i, m);
                        m += 1;
                    }
                }
                if m == start || m == end { start + count / 2 } else { m }
            };

            let left_idx = nodes.len();
            nodes.push(BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 });
            nodes.push(BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 });
            nodes[node_idx] =
                BvhNodeGpu { aabb_min: bounds.min, left_or_first: left_idx as u32, aabb_max: bounds.max, tri_count: 0 };
            build_rec(left_idx, start, mid, depth + 1, order, centroids, tri_boxes, triangles, nodes, out_tris);
            build_rec(left_idx + 1, mid, end, depth + 1, order, centroids, tri_boxes, triangles, nodes, out_tris);
        }

        build_rec(
            0,
            0,
            valid_tris.len(),
            0,
            &mut order,
            &centroids,
            &tri_boxes,
            &valid_tris,
            &mut nodes,
            &mut out_tris,
        );
        Ok(Self { nodes, tris: out_tris })
    }
}

// ============================================================================
// 2-Level Dynamic TLAS/BLAS Ray-Tracing Engine (Shadows, RTAO, Reflections, GI)
// ============================================================================

/// Mesh-local triangle with per-vertex normals for a reusable Bottom-Level Acceleration Structure ([`Blas`]).
#[derive(Clone, Copy, Debug)]
pub struct BlasTriangle {
    pub v0: [f32; 3],
    pub v1: [f32; 3],
    pub v2: [f32; 3],
    pub n0: [f32; 3],
    pub n1: [f32; 3],
    pub n2: [f32; 3],
}

impl BlasTriangle {
    /// Creates a flat-shaded triangle whose vertex normals equal the normalized geometric face normal.
    pub fn flat(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> Self {
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let len = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        let n = if len > 1e-8 { [c[0] / len, c[1] / len, c[2] / len] } else { [0.0, 1.0, 0.0] };
        Self { v0, v1, v2, n0: n, n1: n, n2: n }
    }

    /// Creates a smooth-shaded triangle with explicit per-vertex normals.
    pub fn smooth(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], n0: [f32; 3], n1: [f32; 3], n2: [f32; 3]) -> Self {
        Self { v0, v1, v2, n0, n1, n2 }
    }
}

/// GPU-packed BLAS triangle record (96 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlasTriGpu {
    pub v0: [f32; 4],
    pub v1: [f32; 4],
    pub v2: [f32; 4],
    pub n0: [f32; 4],
    pub n1: [f32; 4],
    pub n2: [f32; 4],
}

impl BlasTriGpu {
    fn from_tri(t: &BlasTriangle) -> Self {
        Self {
            v0: [t.v0[0], t.v0[1], t.v0[2], 0.0],
            v1: [t.v1[0], t.v1[1], t.v1[2], 0.0],
            v2: [t.v2[0], t.v2[1], t.v2[2], 0.0],
            n0: [t.n0[0], t.n0[1], t.n0[2], 0.0],
            n1: [t.n1[0], t.n1[1], t.n1[2], 0.0],
            n2: [t.n2[0], t.n2[1], t.n2[2], 0.0],
        }
    }
}

fn blas_tri_area2(t: &BlasTriangle) -> f64 {
    let e1 = [f64::from(t.v1[0] - t.v0[0]), f64::from(t.v1[1] - t.v0[1]), f64::from(t.v1[2] - t.v0[2])];
    let e2 = [f64::from(t.v2[0] - t.v0[0]), f64::from(t.v2[1] - t.v0[1]), f64::from(t.v2[2] - t.v0[2])];
    let cx = e1[1] * e2[2] - e1[2] * e2[1];
    let cy = e1[2] * e2[0] - e1[0] * e2[2];
    let cz = e1[0] * e2[1] - e1[1] * e2[0];
    cx * cx + cy * cy + cz * cz
}

/// Reusable per-mesh Bottom-Level Acceleration Structure supporting O(N) in-place [`Blas::refit`]
/// for deformed or skinned geometry without topology rebuild.
#[derive(Clone, Debug)]
pub struct Blas {
    nodes: Vec<BvhNodeGpu>,
    tris: Vec<BlasTriGpu>,
    /// Maps each packed leaf triangle index back to its index in the original input slice.
    source_indices: Vec<usize>,
    source_len: usize,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
}

impl Blas {
    pub fn nodes(&self) -> &[BvhNodeGpu] {
        &self.nodes
    }

    pub fn tris(&self) -> &[BlasTriGpu] {
        &self.tris
    }

    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        (self.aabb_min, self.aabb_max)
    }

    /// Builds a SAH-binned BLAS (depth `<= 24`, guaranteed safe for the 28-deep GPU BLAS stack).
    pub fn build(triangles: &[BlasTriangle]) -> Result<Self> {
        let mut valid_indices = Vec::with_capacity(triangles.len());
        for (i, t) in triangles.iter().enumerate() {
            ensure!(
                t.v0.iter()
                    .chain(t.v1.iter())
                    .chain(t.v2.iter())
                    .chain(t.n0.iter())
                    .chain(t.n1.iter())
                    .chain(t.n2.iter())
                    .all(|v| v.is_finite()),
                "BLAS triangle contains non-finite coordinate or normal"
            );
            if blas_tri_area2(t) > 1e-24 {
                valid_indices.push(i);
            }
        }
        if valid_indices.is_empty() {
            return Ok(Self {
                nodes: Vec::new(),
                tris: Vec::new(),
                source_indices: Vec::new(),
                source_len: triangles.len(),
                aabb_min: [0.0; 3],
                aabb_max: [0.0; 3],
            });
        }
        let centroids: Vec<[f32; 3]> = triangles
            .iter()
            .map(|t| {
                [
                    (t.v0[0] + t.v1[0] + t.v2[0]) / 3.0,
                    (t.v0[1] + t.v1[1] + t.v2[1]) / 3.0,
                    (t.v0[2] + t.v1[2] + t.v2[2]) / 3.0,
                ]
            })
            .collect();
        let tri_boxes: Vec<Aabb> = triangles
            .iter()
            .map(|t| {
                let mut b = Aabb::empty();
                b.grow_pt(t.v0);
                b.grow_pt(t.v1);
                b.grow_pt(t.v2);
                b
            })
            .collect();

        let mut nodes = vec![BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 }];
        let mut out_tris = Vec::with_capacity(valid_indices.len());
        let mut source_indices = Vec::with_capacity(valid_indices.len());

        #[allow(clippy::too_many_arguments)]
        fn build_rec(
            node_idx: usize,
            start: usize,
            end: usize,
            depth: u32,
            order: &mut [usize],
            centroids: &[[f32; 3]],
            tri_boxes: &[Aabb],
            triangles: &[BlasTriangle],
            nodes: &mut Vec<BvhNodeGpu>,
            out_tris: &mut Vec<BlasTriGpu>,
            source_indices: &mut Vec<usize>,
        ) {
            let mut bounds = Aabb::empty();
            let mut c_bounds = Aabb::empty();
            for &ti in &order[start..end] {
                bounds.grow_box(&tri_boxes[ti]);
                c_bounds.grow_pt(centroids[ti]);
            }
            let count = end - start;
            if count <= 4 || depth >= 24 {
                let first = out_tris.len() as u32;
                for &ti in &order[start..end] {
                    out_tris.push(BlasTriGpu::from_tri(&triangles[ti]));
                    source_indices.push(ti);
                }
                nodes[node_idx] = BvhNodeGpu {
                    aabb_min: bounds.min,
                    left_or_first: first,
                    aabb_max: bounds.max,
                    tri_count: count as u32,
                };
                return;
            }
            let ext = [
                c_bounds.max[0] - c_bounds.min[0],
                c_bounds.max[1] - c_bounds.min[1],
                c_bounds.max[2] - c_bounds.min[2],
            ];
            let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
                0
            } else if ext[1] >= ext[2] {
                1
            } else {
                2
            };
            let mid = if ext[axis] <= 1e-6 {
                start + count / 2
            } else {
                const BINS: usize = 8;
                let mut bin_box = [Aabb::empty(); BINS];
                let mut bin_cnt = [0usize; BINS];
                let scale = BINS as f32 / ext[axis];
                for &ti in &order[start..end] {
                    let b = (((centroids[ti][axis] - c_bounds.min[axis]) * scale) as usize).min(BINS - 1);
                    bin_cnt[b] += 1;
                    bin_box[b].grow_box(&tri_boxes[ti]);
                }
                let mut best_cost = f32::INFINITY;
                let mut best_split = 0usize;
                for split in 1..BINS {
                    let (mut l_box, mut r_box) = (Aabb::empty(), Aabb::empty());
                    let (mut l_cnt, mut r_cnt) = (0usize, 0usize);
                    for b in 0..split {
                        l_cnt += bin_cnt[b];
                        l_box.grow_box(&bin_box[b]);
                    }
                    for b in split..BINS {
                        r_cnt += bin_cnt[b];
                        r_box.grow_box(&bin_box[b]);
                    }
                    if l_cnt == 0 || r_cnt == 0 {
                        continue;
                    }
                    let cost = l_cnt as f32 * l_box.surface_area() + r_cnt as f32 * r_box.surface_area();
                    if cost < best_cost {
                        best_cost = cost;
                        best_split = split;
                    }
                }
                let split_pos = if best_split == 0 {
                    c_bounds.min[axis] + 0.5 * ext[axis]
                } else {
                    c_bounds.min[axis] + (best_split as f32 / BINS as f32) * ext[axis]
                };
                let mut m = start;
                for i in start..end {
                    if centroids[order[i]][axis] < split_pos {
                        order.swap(i, m);
                        m += 1;
                    }
                }
                if m == start || m == end { start + count / 2 } else { m }
            };

            let left_idx = nodes.len();
            nodes.push(BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 });
            nodes.push(BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 });
            nodes[node_idx] =
                BvhNodeGpu { aabb_min: bounds.min, left_or_first: left_idx as u32, aabb_max: bounds.max, tri_count: 0 };
            build_rec(
                left_idx,
                start,
                mid,
                depth + 1,
                order,
                centroids,
                tri_boxes,
                triangles,
                nodes,
                out_tris,
                source_indices,
            );
            build_rec(
                left_idx + 1,
                mid,
                end,
                depth + 1,
                order,
                centroids,
                tri_boxes,
                triangles,
                nodes,
                out_tris,
                source_indices,
            );
        }

        let n_valid = valid_indices.len();
        build_rec(
            0,
            0,
            n_valid,
            0,
            &mut valid_indices,
            &centroids,
            &tri_boxes,
            triangles,
            &mut nodes,
            &mut out_tris,
            &mut source_indices,
        );
        let root_min = nodes[0].aabb_min;
        let root_max = nodes[0].aabb_max;
        Ok(Self {
            nodes,
            tris: out_tris,
            source_indices,
            source_len: triangles.len(),
            aabb_min: root_min,
            aabb_max: root_max,
        })
    }

    /// Refits deformed/skinned triangle vertices and every BVH node AABB in O(N) bottom-up
    /// without rebuilding tree topology.
    pub fn refit(&mut self, updated_triangles: &[BlasTriangle]) -> Result<()> {
        ensure!(
            updated_triangles.len() == self.source_len,
            "BLAS refit triangle count {} does not match original {}",
            updated_triangles.len(),
            self.source_len
        );
        if self.nodes.is_empty() {
            return Ok(());
        }
        for (dst_idx, &src_idx) in self.source_indices.iter().enumerate() {
            let t = &updated_triangles[src_idx];
            ensure!(
                t.v0.iter()
                    .chain(t.v1.iter())
                    .chain(t.v2.iter())
                    .chain(t.n0.iter())
                    .chain(t.n1.iter())
                    .chain(t.n2.iter())
                    .all(|v| v.is_finite()),
                "BLAS refit triangle contains non-finite value"
            );
            self.tris[dst_idx] = BlasTriGpu::from_tri(t);
        }
        // Reverse index order visits every child strictly before its parent.
        for idx in (0..self.nodes.len()).rev() {
            let node = self.nodes[idx];
            let mut b = Aabb::empty();
            if node.tri_count > 0 {
                let first = node.left_or_first as usize;
                let end = (first + node.tri_count as usize).min(self.tris.len());
                for tri in &self.tris[first..end] {
                    b.grow_pt([tri.v0[0], tri.v0[1], tri.v0[2]]);
                    b.grow_pt([tri.v1[0], tri.v1[1], tri.v1[2]]);
                    b.grow_pt([tri.v2[0], tri.v2[1], tri.v2[2]]);
                }
            } else {
                let left = node.left_or_first as usize;
                let right = left + 1;
                if left < self.nodes.len() {
                    b.grow_box(&Aabb { min: self.nodes[left].aabb_min, max: self.nodes[left].aabb_max });
                }
                if right < self.nodes.len() {
                    b.grow_box(&Aabb { min: self.nodes[right].aabb_min, max: self.nodes[right].aabb_max });
                }
            }
            self.nodes[idx].aabb_min = b.min;
            self.nodes[idx].aabb_max = b.max;
        }
        self.aabb_min = self.nodes[0].aabb_min;
        self.aabb_max = self.nodes[0].aabb_max;
        Ok(())
    }
}

/// Dynamic scene instance referencing a shared [`Blas`] with a world transform and shading parameters.
#[derive(Clone, Copy, Debug)]
pub struct TlasInstance {
    pub blas_index: u32,
    pub transform: [f32; 16],
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub metallic: f32,
    pub mask: u32,
}

impl TlasInstance {
    pub fn new(blas_index: u32, transform: [f32; 16]) -> Self {
        Self {
            blas_index,
            transform,
            albedo: [0.8, 0.8, 0.8],
            roughness: 0.5,
            emissive: [0.0, 0.0, 0.0],
            metallic: 0.0,
            mask: u32::MAX,
        }
    }
}

/// GPU-packed TLAS instance record (160 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TlasInstanceGpu {
    pub inv_transform: [f32; 16],
    pub normal_matrix_0: [f32; 4],
    pub normal_matrix_1: [f32; 4],
    pub normal_matrix_2: [f32; 4],
    pub albedo_roughness: [f32; 4],
    pub emissive_metallic: [f32; 4],
    pub blas_node_offset: u32,
    pub blas_node_count: u32,
    pub blas_tri_offset: u32,
    pub mask: u32,
}

pub(super) fn transform_aabb(m: &[f32; 16], bmin: [f32; 3], bmax: [f32; 3]) -> Aabb {
    let mut out = Aabb::empty();
    for &x in &[bmin[0], bmax[0]] {
        for &y in &[bmin[1], bmax[1]] {
            for &z in &[bmin[2], bmax[2]] {
                let wx = m[0] * x + m[4] * y + m[8] * z + m[12];
                let wy = m[1] * x + m[5] * y + m[9] * z + m[13];
                let wz = m[2] * x + m[6] * y + m[10] * z + m[14];
                out.grow_pt([wx, wy, wz]);
            }
        }
    }
    out
}

/// Assembled 2-level acceleration structure combining packed BLAS pools with a dynamic SAH TLAS.
#[derive(Clone, Debug)]
pub struct TlasScene {
    tlas_nodes: Vec<BvhNodeGpu>,
    tlas_instances: Vec<TlasInstanceGpu>,
    blas_nodes: Vec<BvhNodeGpu>,
    blas_tris: Vec<BlasTriGpu>,
}

impl TlasScene {
    pub fn tlas_nodes(&self) -> &[BvhNodeGpu] {
        &self.tlas_nodes
    }

    pub fn tlas_instances(&self) -> &[TlasInstanceGpu] {
        &self.tlas_instances
    }

    pub fn blas_nodes(&self) -> &[BvhNodeGpu] {
        &self.blas_nodes
    }

    pub fn blas_tris(&self) -> &[BlasTriGpu] {
        &self.blas_tris
    }

    /// Builds a 2-level TLAS over `instances` referencing `blases`.
    /// Fast enough to rebuild every frame when dynamic transforms move without touching BLAS trees.
    pub fn build(blases: &[Blas], instances: &[TlasInstance]) -> Result<Self> {
        let mut blas_offsets = Vec::with_capacity(blases.len());
        let mut blas_nodes = Vec::new();
        let mut blas_tris = Vec::new();
        for b in blases {
            blas_offsets.push((blas_nodes.len() as u32, b.nodes.len() as u32, blas_tris.len() as u32));
            blas_nodes.extend_from_slice(&b.nodes);
            blas_tris.extend_from_slice(&b.tris);
        }

        let mut prepared = Vec::with_capacity(instances.len());
        let mut inst_boxes = Vec::with_capacity(instances.len());
        let mut centroids = Vec::with_capacity(instances.len());

        for inst in instances {
            ensure!(inst.transform.iter().all(|v| v.is_finite()), "TLAS instance transform contains non-finite value");
            let bi = inst.blas_index as usize;
            ensure!(bi < blases.len(), "TLAS instance blas_index {bi} out of bounds ({})", blases.len());
            let blas = &blases[bi];
            if blas.nodes.is_empty() {
                continue;
            }
            let inv = crate::matrix::invert(&inst.transform);
            ensure!(inv.iter().all(|v| v.is_finite()), "TLAS instance transform is singular / non-invertible");
            // Columns of transpose(inv_3x3): column c = [inv[0*4 + c], inv[1*4 + c], inv[2*4 + c]].
            let nm0 = [inv[0], inv[4], inv[8], 0.0];
            let nm1 = [inv[1], inv[5], inv[9], 0.0];
            let nm2 = [inv[2], inv[6], inv[10], 0.0];
            let (node_off, node_cnt, tri_off) = blas_offsets[bi];
            let wb = transform_aabb(&inst.transform, blas.aabb_min, blas.aabb_max);
            centroids.push([
                0.5 * (wb.min[0] + wb.max[0]),
                0.5 * (wb.min[1] + wb.max[1]),
                0.5 * (wb.min[2] + wb.max[2]),
            ]);
            inst_boxes.push(wb);
            prepared.push(TlasInstanceGpu {
                inv_transform: inv,
                normal_matrix_0: nm0,
                normal_matrix_1: nm1,
                normal_matrix_2: nm2,
                albedo_roughness: [
                    inst.albedo[0].max(0.0),
                    inst.albedo[1].max(0.0),
                    inst.albedo[2].max(0.0),
                    inst.roughness.clamp(0.0, 1.0),
                ],
                emissive_metallic: [
                    inst.emissive[0].max(0.0),
                    inst.emissive[1].max(0.0),
                    inst.emissive[2].max(0.0),
                    inst.metallic.clamp(0.0, 1.0),
                ],
                blas_node_offset: node_off,
                blas_node_count: node_cnt,
                blas_tri_offset: tri_off,
                mask: inst.mask,
            });
        }

        if prepared.is_empty() {
            return Ok(Self { tlas_nodes: Vec::new(), tlas_instances: Vec::new(), blas_nodes, blas_tris });
        }

        let mut order: Vec<usize> = (0..prepared.len()).collect();
        let mut tlas_nodes =
            vec![BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 }];
        let mut tlas_instances = Vec::with_capacity(prepared.len());

        #[allow(clippy::too_many_arguments)]
        fn build_tlas_rec(
            node_idx: usize,
            start: usize,
            end: usize,
            depth: u32,
            order: &mut [usize],
            centroids: &[[f32; 3]],
            inst_boxes: &[Aabb],
            prepared: &[TlasInstanceGpu],
            nodes: &mut Vec<BvhNodeGpu>,
            out_insts: &mut Vec<TlasInstanceGpu>,
        ) {
            let mut bounds = Aabb::empty();
            let mut c_bounds = Aabb::empty();
            for &ii in &order[start..end] {
                bounds.grow_box(&inst_boxes[ii]);
                c_bounds.grow_pt(centroids[ii]);
            }
            let count = end - start;
            if count <= 2 || depth >= 20 {
                let first = out_insts.len() as u32;
                for &ii in &order[start..end] {
                    out_insts.push(prepared[ii]);
                }
                nodes[node_idx] = BvhNodeGpu {
                    aabb_min: bounds.min,
                    left_or_first: first,
                    aabb_max: bounds.max,
                    tri_count: count as u32,
                };
                return;
            }
            let ext = [
                c_bounds.max[0] - c_bounds.min[0],
                c_bounds.max[1] - c_bounds.min[1],
                c_bounds.max[2] - c_bounds.min[2],
            ];
            let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
                0
            } else if ext[1] >= ext[2] {
                1
            } else {
                2
            };
            let mid = if ext[axis] <= 1e-6 {
                start + count / 2
            } else {
                let split_pos = c_bounds.min[axis] + 0.5 * ext[axis];
                let mut m = start;
                for i in start..end {
                    if centroids[order[i]][axis] < split_pos {
                        order.swap(i, m);
                        m += 1;
                    }
                }
                if m == start || m == end { start + count / 2 } else { m }
            };
            let left_idx = nodes.len();
            nodes.push(BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 });
            nodes.push(BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 });
            nodes[node_idx] =
                BvhNodeGpu { aabb_min: bounds.min, left_or_first: left_idx as u32, aabb_max: bounds.max, tri_count: 0 };
            build_tlas_rec(left_idx, start, mid, depth + 1, order, centroids, inst_boxes, prepared, nodes, out_insts);
            build_tlas_rec(left_idx + 1, mid, end, depth + 1, order, centroids, inst_boxes, prepared, nodes, out_insts);
        }

        build_tlas_rec(
            0,
            0,
            prepared.len(),
            0,
            &mut order,
            &centroids,
            &inst_boxes,
            &prepared,
            &mut tlas_nodes,
            &mut tlas_instances,
        );
        Ok(Self { tlas_nodes, tlas_instances, blas_nodes, blas_tris })
    }
}

/// Uploaded GPU buffers for a [`TlasScene`], shareable across [`TlasRayTracer`](super::TlasRayTracer), DDGI, and lightmap baking.
#[derive(Clone)]
pub struct TlasSceneGpu {
    pub tlas_nodes_buf: wgpu::Buffer,
    pub tlas_instances_buf: wgpu::Buffer,
    pub blas_nodes_buf: wgpu::Buffer,
    pub blas_tris_buf: wgpu::Buffer,
    pub tlas_node_count: u32,
    pub tlas_instance_count: u32,
    pub blas_tri_count: u32,
    #[cfg(not(target_arch = "wasm32"))]
    pub hw_tlas: Option<wgpu::Tlas>,
}

impl TlasSceneGpu {
    pub fn has_hw_tlas(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.hw_tlas.is_some()
        }
        #[cfg(target_arch = "wasm32")]
        {
            false
        }
    }

    pub fn upload(device: &wgpu::Device, scene: &TlasScene) -> Result<Self> {
        let max_storage = device.limits().max_storage_buffer_binding_size;
        let t_nodes = if scene.tlas_nodes.is_empty() {
            vec![BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 }]
        } else {
            scene.tlas_nodes.clone()
        };
        let t_insts = if scene.tlas_instances.is_empty() {
            vec![bytemuck::Zeroable::zeroed()]
        } else {
            scene.tlas_instances.clone()
        };
        let b_nodes = if scene.blas_nodes.is_empty() {
            vec![BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 }]
        } else {
            scene.blas_nodes.clone()
        };
        let b_tris =
            if scene.blas_tris.is_empty() { vec![bytemuck::Zeroable::zeroed()] } else { scene.blas_tris.clone() };

        let make_buf = |label: &str, bytes: &[u8]| -> Result<wgpu::Buffer> {
            ensure!(bytes.len() as u64 <= max_storage, "{label} exceeds device storage limit");
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: true,
            });
            buf.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytes);
            buf.unmap();
            Ok(buf)
        };

        Ok(Self {
            tlas_nodes_buf: make_buf("tlas_nodes", bytemuck::cast_slice(&t_nodes))?,
            tlas_instances_buf: make_buf("tlas_instances", bytemuck::cast_slice(&t_insts))?,
            blas_nodes_buf: make_buf("blas_nodes", bytemuck::cast_slice(&b_nodes))?,
            blas_tris_buf: make_buf("blas_tris", bytemuck::cast_slice(&b_tris))?,
            tlas_node_count: scene.tlas_nodes.len() as u32,
            tlas_instance_count: scene.tlas_instances.len() as u32,
            blas_tri_count: scene.blas_tris.len() as u32,
            #[cfg(not(target_arch = "wasm32"))]
            hw_tlas: None,
        })
    }
}
