use std::collections::HashMap;

use anyhow::{Result, ensure};

use super::bvh::{
    Aabb, Blas, BlasTriGpu, BlasTriangle, BvhNodeGpu, TlasInstance, TlasInstanceGpu, TlasSceneGpu, transform_aabb,
};
use super::{device_supports_hardware_ray_query, encode_copy};

struct PoolBlasEntry {
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    node_offset: u32,
    node_count: u32,
    tri_offset: u32,
    tri_count: u32,
    #[cfg(not(target_arch = "wasm32"))]
    hw_blas: Option<wgpu::Blas>,
    #[cfg(not(target_arch = "wasm32"))]
    _hw_vbuf: Option<wgpu::Buffer>,
}

/// Persistent GPU BLAS pool with incremental append and TLAS-only per-frame updates.
///
/// Static mesh BLAS geometry is built and uploaded to the GPU **once** per unique mesh ID;
/// moving objects only rebuild and write the lightweight TLAS node/instance buffers (and
/// optional native hardware `wgpu::Tlas` when `EXPERIMENTAL_RAY_QUERY` is enabled).
pub struct GpuBlasPool {
    id_to_index: HashMap<u64, u32>,
    entries: Vec<PoolBlasEntry>,
    blas_nodes_buf: wgpu::Buffer,
    blas_nodes_cap: u32,
    blas_nodes_len: u32,
    blas_tris_buf: wgpu::Buffer,
    blas_tris_cap: u32,
    blas_tris_len: u32,
    tlas_nodes_buf: wgpu::Buffer,
    tlas_nodes_cap: u32,
    tlas_nodes_len: u32,
    tlas_instances_buf: wgpu::Buffer,
    tlas_instances_cap: u32,
    tlas_instances_len: u32,
    use_hw_rt: bool,
    #[cfg(not(target_arch = "wasm32"))]
    hw_tlas: Option<wgpu::Tlas>,
    #[cfg(not(target_arch = "wasm32"))]
    hw_tlas_cap: u32,
    pub blas_upload_count: u64,
    pub tlas_update_count: u64,
}

impl GpuBlasPool {
    pub fn new(device: &wgpu::Device) -> Self {
        let make_storage = |label: &str, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(64),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let blas_nodes_cap = 1024u32;
        let blas_tris_cap = 2048u32;
        let tlas_nodes_cap = 256u32;
        let tlas_instances_cap = 256u32;
        Self {
            id_to_index: HashMap::new(),
            entries: Vec::new(),
            blas_nodes_buf: make_storage(
                "pool_blas_nodes",
                u64::from(blas_nodes_cap) * std::mem::size_of::<BvhNodeGpu>() as u64,
            ),
            blas_nodes_cap,
            blas_nodes_len: 0,
            blas_tris_buf: make_storage(
                "pool_blas_tris",
                u64::from(blas_tris_cap) * std::mem::size_of::<BlasTriGpu>() as u64,
            ),
            blas_tris_cap,
            blas_tris_len: 0,
            tlas_nodes_buf: make_storage(
                "pool_tlas_nodes",
                u64::from(tlas_nodes_cap) * std::mem::size_of::<BvhNodeGpu>() as u64,
            ),
            tlas_nodes_cap,
            tlas_nodes_len: 0,
            tlas_instances_buf: make_storage(
                "pool_tlas_instances",
                u64::from(tlas_instances_cap) * std::mem::size_of::<TlasInstanceGpu>() as u64,
            ),
            tlas_instances_cap,
            tlas_instances_len: 0,
            use_hw_rt: device_supports_hardware_ray_query(device),
            #[cfg(not(target_arch = "wasm32"))]
            hw_tlas: None,
            #[cfg(not(target_arch = "wasm32"))]
            hw_tlas_cap: 0,
            blas_upload_count: 0,
            tlas_update_count: 0,
        }
    }

    pub fn uses_hardware_ray_query(&self) -> bool {
        self.use_hw_rt
    }

    #[allow(clippy::too_many_arguments)]
    fn grow_buffer(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buf: &mut wgpu::Buffer,
        cap: &mut u32,
        used: u32,
        needed: u32,
        elem_size: usize,
        label: &str,
    ) -> Result<()> {
        if needed <= *cap {
            return Ok(());
        }
        let mut next_cap = (*cap).max(64);
        while next_cap < needed {
            next_cap = next_cap.saturating_mul(2);
        }
        let next_bytes = u64::from(next_cap) * elem_size as u64;
        ensure!(
            next_bytes <= device.limits().max_storage_buffer_binding_size,
            "{label} ({next_bytes} bytes) exceeds device max_storage_buffer_binding_size"
        );
        let next_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: next_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let copy_bytes = u64::from(used) * elem_size as u64;
        if copy_bytes > 0 {
            encoder.copy_buffer_to_buffer(buf, 0, &next_buf, 0, copy_bytes);
        }
        *buf = next_buf;
        *cap = next_cap;
        Ok(())
    }

    /// Ensures mesh `id` has a resident BLAS in the persistent GPU pool.
    /// Subsequent calls with the same `id` are O(1) lookups with zero CPU builds or GPU uploads.
    pub fn ensure_blas(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        id: u64,
        triangles: &[BlasTriangle],
    ) -> Result<Option<u32>> {
        if let Some(&idx) = self.id_to_index.get(&id) {
            return Ok((self.entries[idx as usize].node_count > 0).then_some(idx));
        }
        let blas = Blas::build(triangles)?;
        let (aabb_min, aabb_max) = blas.bounds();
        let node_count = blas.nodes().len() as u32;
        let tri_count = blas.tris().len() as u32;
        if node_count == 0 || tri_count == 0 {
            let idx = self.entries.len() as u32;
            self.entries.push(PoolBlasEntry {
                aabb_min: [0.0; 3],
                aabb_max: [0.0; 3],
                node_offset: 0,
                node_count: 0,
                tri_offset: 0,
                tri_count: 0,
                #[cfg(not(target_arch = "wasm32"))]
                hw_blas: None,
                #[cfg(not(target_arch = "wasm32"))]
                _hw_vbuf: None,
            });
            self.id_to_index.insert(id, idx);
            return Ok(None);
        }

        let node_offset = self.blas_nodes_len;
        let tri_offset = self.blas_tris_len;
        Self::grow_buffer(
            device,
            encoder,
            &mut self.blas_nodes_buf,
            &mut self.blas_nodes_cap,
            self.blas_nodes_len,
            node_offset + node_count,
            std::mem::size_of::<BvhNodeGpu>(),
            "pool_blas_nodes",
        )?;
        Self::grow_buffer(
            device,
            encoder,
            &mut self.blas_tris_buf,
            &mut self.blas_tris_cap,
            self.blas_tris_len,
            tri_offset + tri_count,
            std::mem::size_of::<BlasTriGpu>(),
            "pool_blas_tris",
        )?;

        let node_bytes: &[u8] = bytemuck::cast_slice(blas.nodes());
        let stage_nodes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool_blas_nodes_stage"),
            size: node_bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        stage_nodes.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(node_bytes);
        stage_nodes.unmap();
        encoder.copy_buffer_to_buffer(
            &stage_nodes,
            0,
            &self.blas_nodes_buf,
            u64::from(node_offset) * std::mem::size_of::<BvhNodeGpu>() as u64,
            node_bytes.len() as u64,
        );

        let tri_bytes: &[u8] = bytemuck::cast_slice(blas.tris());
        let stage_tris = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool_blas_tris_stage"),
            size: tri_bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        stage_tris.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(tri_bytes);
        stage_tris.unmap();
        encoder.copy_buffer_to_buffer(
            &stage_tris,
            0,
            &self.blas_tris_buf,
            u64::from(tri_offset) * std::mem::size_of::<BlasTriGpu>() as u64,
            tri_bytes.len() as u64,
        );

        self.blas_nodes_len += node_count;
        self.blas_tris_len += tri_count;
        self.blas_upload_count += 1;

        #[cfg(not(target_arch = "wasm32"))]
        let (hw_blas, _hw_vbuf) = if self.use_hw_rt {
            let mut verts: Vec<[f32; 3]> = Vec::with_capacity(blas.tris().len() * 3);
            for t in blas.tris() {
                verts.push([t.v0[0], t.v0[1], t.v0[2]]);
                verts.push([t.v1[0], t.v1[1], t.v1[2]]);
                verts.push([t.v2[0], t.v2[1], t.v2[2]]);
            }
            let vbytes: &[u8] = bytemuck::cast_slice(&verts);
            let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hw_blas_verts"),
                size: vbytes.len() as u64,
                usage: wgpu::BufferUsages::BLAS_INPUT,
                mapped_at_creation: true,
            });
            vbuf.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(vbytes);
            vbuf.unmap();
            let size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: verts.len() as u32,
                index_format: None,
                index_count: None,
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            };
            let b = device.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: Some("hw_blas"),
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size_desc.clone()] },
            );
            let geom = wgpu::BlasTriangleGeometry {
                size: &size_desc,
                vertex_buffer: &vbuf,
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: None,
                first_index: None,
                transform_buffer: None,
                transform_buffer_offset: None,
            };
            let entry =
                wgpu::BlasBuildEntry { blas: &b, geometry: wgpu::BlasGeometries::TriangleGeometries(vec![geom]) };
            encoder.build_acceleration_structures(std::iter::once(&entry), std::iter::empty());
            (Some(b), Some(vbuf))
        } else {
            (None, None)
        };

        let idx = self.entries.len() as u32;
        self.entries.push(PoolBlasEntry {
            aabb_min,
            aabb_max,
            node_offset,
            node_count,
            tri_offset,
            tri_count,
            #[cfg(not(target_arch = "wasm32"))]
            hw_blas,
            #[cfg(not(target_arch = "wasm32"))]
            _hw_vbuf,
        });
        self.id_to_index.insert(id, idx);
        Ok(Some(idx))
    }

    pub fn blas_bounds(&self, pool_index: u32) -> ([f32; 3], [f32; 3]) {
        let e = &self.entries[pool_index as usize];
        (e.aabb_min, e.aabb_max)
    }

    /// Rebuilds ONLY the lightweight top-level acceleration structure over `instances`
    /// referencing already-uploaded BLAS pool slots — zero BLAS copies or re-uploads.
    pub fn update_tlas(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        instances: &[TlasInstance],
    ) -> Result<()> {
        let mut prepared = Vec::with_capacity(instances.len());
        #[cfg(not(target_arch = "wasm32"))]
        let hw_only = self.use_hw_rt;
        #[cfg(target_arch = "wasm32")]
        let hw_only = false;
        let mut inst_boxes = Vec::with_capacity(if hw_only { 0 } else { instances.len() });
        let mut centroids = Vec::with_capacity(if hw_only { 0 } else { instances.len() });
        let mut world_transforms = Vec::with_capacity(instances.len());
        let mut blas_indices = Vec::with_capacity(instances.len());

        for inst in instances {
            ensure!(inst.transform.iter().all(|v| v.is_finite()), "TLAS instance transform contains non-finite value");
            let bi = inst.blas_index as usize;
            ensure!(bi < self.entries.len(), "TLAS instance pool blas_index {bi} out of bounds");
            let entry = &self.entries[bi];
            if entry.node_count == 0 || entry.tri_count == 0 {
                continue;
            }
            let inv = crate::matrix::invert(&inst.transform);
            ensure!(inv.iter().all(|v| v.is_finite()), "TLAS instance transform is singular / non-invertible");
            let nm0 = [inv[0], inv[4], inv[8], 0.0];
            let nm1 = [inv[1], inv[5], inv[9], 0.0];
            let nm2 = [inv[2], inv[6], inv[10], 0.0];
            if !hw_only {
                let wb = transform_aabb(&inst.transform, entry.aabb_min, entry.aabb_max);
                centroids.push([
                    0.5 * (wb.min[0] + wb.max[0]),
                    0.5 * (wb.min[1] + wb.max[1]),
                    0.5 * (wb.min[2] + wb.max[2]),
                ]);
                inst_boxes.push(wb);
            }
            world_transforms.push(inst.transform);
            blas_indices.push(bi);
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
                blas_node_offset: entry.node_offset,
                blas_node_count: entry.node_count,
                blas_tri_offset: entry.tri_offset,
                mask: inst.mask,
            });
        }

        if prepared.is_empty() {
            self.tlas_nodes_len = 0;
            self.tlas_instances_len = 0;
            return Ok(());
        }

        // Native hardware ray-query (`tlas_traversal_hw.wgsl`) indexes `tlas_instances[custom_data]`
        // directly from `hw_tlas` and never reads `tlas_nodes`, so skip the CPU SW BVH build & upload!
        #[cfg(not(target_arch = "wasm32"))]
        if hw_only {
            let n_insts = prepared.len() as u32;
            Self::grow_buffer(
                device,
                encoder,
                &mut self.tlas_instances_buf,
                &mut self.tlas_instances_cap,
                0,
                n_insts,
                std::mem::size_of::<TlasInstanceGpu>(),
                "pool_tlas_instances",
            )?;
            encode_copy(device, encoder, &self.tlas_instances_buf, bytemuck::cast_slice(&prepared));
            self.tlas_nodes_len = 0;
            self.tlas_instances_len = n_insts;
            self.tlas_update_count += 1;

            let needed = n_insts.max(1);
            if self.hw_tlas.is_none() || self.hw_tlas_cap < needed {
                let cap = needed.next_power_of_two().max(64);
                self.hw_tlas = Some(device.create_tlas(&wgpu::CreateTlasDescriptor {
                    label: Some("hw_tlas"),
                    max_instances: cap,
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                }));
                self.hw_tlas_cap = cap;
            }
            if let Some(hw_tlas) = &mut self.hw_tlas {
                let total_slots = self.hw_tlas_cap as usize;
                for slot in 0..total_slots {
                    if slot < prepared.len() {
                        let bi = blas_indices[slot];
                        let m = world_transforms[slot];
                        let row_major_3x4 =
                            [m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14]];
                        let mask = (prepared[slot].mask & 0xff) as u8;
                        let mask = if mask == 0 { 0xff } else { mask };
                        if let Some(hw_blas) = &self.entries[bi].hw_blas {
                            *hw_tlas.get_mut_single(slot).unwrap() =
                                Some(wgpu::TlasInstance::new(hw_blas, row_major_3x4, slot as u32, mask));
                        } else {
                            *hw_tlas.get_mut_single(slot).unwrap() = None;
                        }
                    } else {
                        *hw_tlas.get_mut_single(slot).unwrap() = None;
                    }
                }
                encoder.build_acceleration_structures(std::iter::empty(), std::iter::once(&*hw_tlas));
            }
            return Ok(());
        }

        let mut order: Vec<usize> = (0..prepared.len()).collect();
        let mut tlas_nodes =
            vec![BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 }];
        let mut tlas_instances = Vec::with_capacity(prepared.len());
        let mut ordered_src = Vec::with_capacity(prepared.len());

        #[allow(clippy::too_many_arguments)]
        fn build_rec(
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
            out_src: &mut Vec<usize>,
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
                    out_src.push(ii);
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
            build_rec(
                left_idx,
                start,
                mid,
                depth + 1,
                order,
                centroids,
                inst_boxes,
                prepared,
                nodes,
                out_insts,
                out_src,
            );
            build_rec(
                left_idx + 1,
                mid,
                end,
                depth + 1,
                order,
                centroids,
                inst_boxes,
                prepared,
                nodes,
                out_insts,
                out_src,
            );
        }

        build_rec(
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
            &mut ordered_src,
        );

        let n_nodes = tlas_nodes.len() as u32;
        let n_insts = tlas_instances.len() as u32;
        Self::grow_buffer(
            device,
            encoder,
            &mut self.tlas_nodes_buf,
            &mut self.tlas_nodes_cap,
            0,
            n_nodes,
            std::mem::size_of::<BvhNodeGpu>(),
            "pool_tlas_nodes",
        )?;
        Self::grow_buffer(
            device,
            encoder,
            &mut self.tlas_instances_buf,
            &mut self.tlas_instances_cap,
            0,
            n_insts,
            std::mem::size_of::<TlasInstanceGpu>(),
            "pool_tlas_instances",
        )?;

        encode_copy(device, encoder, &self.tlas_nodes_buf, bytemuck::cast_slice(&tlas_nodes));
        encode_copy(device, encoder, &self.tlas_instances_buf, bytemuck::cast_slice(&tlas_instances));
        self.tlas_nodes_len = n_nodes;
        self.tlas_instances_len = n_insts;
        self.tlas_update_count += 1;

        #[cfg(not(target_arch = "wasm32"))]
        if self.use_hw_rt {
            let needed = n_insts.max(1);
            if self.hw_tlas.is_none() || self.hw_tlas_cap < needed {
                let cap = needed.next_power_of_two().max(64);
                self.hw_tlas = Some(device.create_tlas(&wgpu::CreateTlasDescriptor {
                    label: Some("hw_tlas"),
                    max_instances: cap,
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                }));
                self.hw_tlas_cap = cap;
            }
            if let Some(hw_tlas) = &mut self.hw_tlas {
                let total_slots = self.hw_tlas_cap as usize;
                for slot in 0..total_slots {
                    if slot < ordered_src.len() {
                        let src_i = ordered_src[slot];
                        let bi = blas_indices[src_i];
                        let m = world_transforms[src_i];
                        let row_major_3x4 =
                            [m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14]];
                        let mask = (tlas_instances[slot].mask & 0xff) as u8;
                        let mask = if mask == 0 { 0xff } else { mask };
                        if let Some(hw_blas) = &self.entries[bi].hw_blas {
                            *hw_tlas.get_mut_single(slot).unwrap() =
                                Some(wgpu::TlasInstance::new(hw_blas, row_major_3x4, slot as u32, mask));
                        } else {
                            *hw_tlas.get_mut_single(slot).unwrap() = None;
                        }
                    } else {
                        *hw_tlas.get_mut_single(slot).unwrap() = None;
                    }
                }
                encoder.build_acceleration_structures(std::iter::empty(), std::iter::once(&*hw_tlas));
            }
        }

        Ok(())
    }

    /// Cheaply clones GPU resource handles into a [`TlasSceneGpu`] snapshot for DDGI / RT passes.
    pub fn snapshot_scene(&self) -> TlasSceneGpu {
        TlasSceneGpu {
            tlas_nodes_buf: self.tlas_nodes_buf.clone(),
            tlas_instances_buf: self.tlas_instances_buf.clone(),
            blas_nodes_buf: self.blas_nodes_buf.clone(),
            blas_tris_buf: self.blas_tris_buf.clone(),
            tlas_node_count: self.tlas_nodes_len,
            tlas_instance_count: self.tlas_instances_len,
            blas_tri_count: self.blas_tris_len,
            #[cfg(not(target_arch = "wasm32"))]
            hw_tlas: self.hw_tlas.clone(),
        }
    }
}
