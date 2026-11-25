//! The checked PMSH v1 layout shared by the reader and byte-plane codec.
//! Count tables and bounds stay untransposed, so this also describes the
//! inflated v2 stream before its records have been restored.
use std::ops::Range;

use anyhow::{Context, Result, ensure};

use super::{
    MAX_LODS, MAX_MORPH_TARGETS, MESH_ATTRIBUTES, MESH_HEADER, MESH_INSTANCES, MESH_LODS, MESH_MAGIC, MESH_MESHLETS,
    MESH_MORPH_FLOATS, MESH_MORPHS, MESH_SHORT_INDICES, MESH_SKIN, MESH_TANGENT_FLOATS, MESH_TANGENTS, MESH_VERSION,
    MESH_VERTEX_FLOATS, MESHLET_SUB_CLUSTERS, MESHLET_SUB_DAG,
};
use crate::meshlet_dag::DagClusterGpu;
use crate::meshlets::MeshletGpu;

pub(super) const INSTANCE_FLOATS: usize = 20;
const SKIN_STRIDE: usize = 24;

pub(super) struct Block {
    pub start: usize,
    pub count: usize,
    pub stride: usize,
}

impl Block {
    pub fn range(&self) -> Range<usize> {
        self.start..self.start + self.count * self.stride
    }

    fn take(bytes: &[u8], at: &mut usize, count: usize, stride: usize) -> Result<Self> {
        let end = count.checked_mul(stride).and_then(|size| at.checked_add(size));
        let end = end.filter(|&end| end <= bytes.len()).context("binary mesh size does not match its counts")?;
        let block = Self { start: *at, count, stride };
        *at = end;
        Ok(block)
    }
}

pub(super) struct Instances {
    pub bounds: Range<usize>,
    pub records: Block,
}

pub(super) struct DagLayout {
    pub level_count: u32,
    pub leaf_cluster_count: u32,
    pub root_cluster_count: u32,
    pub clusters: Block,
    pub indices: Block,
}

pub(super) struct MeshletsLayout {
    pub meshlets: Option<(Block, Block)>,
    pub dag: Option<DagLayout>,
}

pub(super) struct Layout {
    pub vertices: Block,
    pub tangents: Option<Block>,
    pub indices: Block,
    pub lods: Vec<(f32, Block)>,
    pub morphs: Option<(u32, Block)>,
    pub skin: Option<(Block, Range<usize>)>,
    pub instances: Option<Instances>,
    pub meshlets: Option<MeshletsLayout>,
}

pub(super) fn word(bytes: &[u8], at: usize) -> Result<usize> {
    let value = bytes.get(at..).and_then(|tail| tail.get(..4)).context("binary mesh ends in its counts")?;
    Ok(u32::from_le_bytes(value.try_into()?) as usize)
}

impl Layout {
    pub fn read(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() >= MESH_HEADER, "binary mesh ends in its header");
        ensure!(&bytes[..4] == MESH_MAGIC && bytes[4] == MESH_VERSION, "unsupported binary mesh version");
        ensure!(bytes[6..8] == [0; 2], "unsupported binary mesh header");
        const ALL_FLAGS: u8 = MESH_ATTRIBUTES
            | MESH_SHORT_INDICES
            | MESH_SKIN
            | MESH_LODS
            | MESH_INSTANCES
            | MESH_TANGENTS
            | MESH_MORPHS
            | MESH_MESHLETS;
        const _: () = assert!(ALL_FLAGS == u8::MAX);
        let flags = bytes[5];
        ensure!(
            flags & (MESH_SKIN | MESH_INSTANCES) != (MESH_SKIN | MESH_INSTANCES),
            "a mesh cannot have both a skin and persistent instances"
        );
        ensure!(
            flags & MESH_MESHLETS == 0 || flags & (MESH_SKIN | MESH_INSTANCES | MESH_MORPHS) == 0,
            "baked meshlets cannot be combined with skinning, morphs, or persistent instances"
        );
        let count = word(bytes, 8)?;
        let floats = if flags & MESH_ATTRIBUTES != 0 { MESH_VERTEX_FLOATS } else { MESH_VERTEX_FLOATS - 2 };
        let index_bytes = if flags & MESH_SHORT_INDICES != 0 { 2 } else { 4 };
        let mut at = MESH_HEADER;
        let vertices = Block::take(bytes, &mut at, count, floats * 4)?;
        let tangents = if flags & MESH_TANGENTS != 0 {
            Some(Block::take(bytes, &mut at, count, MESH_TANGENT_FLOATS * 4)?)
        } else {
            None
        };
        let indices = Block::take(bytes, &mut at, word(bytes, 12)?, index_bytes)?;
        let mut lods = Vec::new();
        if flags & MESH_LODS != 0 {
            let count = word(bytes, at)?;
            ensure!(count < MAX_LODS, "binary mesh has too many levels of detail");
            at += 4;
            let table = Block::take(bytes, &mut at, count, 8)?;
            for row in bytes[table.range()].as_chunks::<8>().0 {
                let count = word(row, 0)?;
                let error = f32::from_le_bytes(row[4..8].try_into()?);
                lods.push((error, Block::take(bytes, &mut at, count, index_bytes)?));
            }
        }
        let morphs = if flags & MESH_MORPHS != 0 {
            let targets = word(bytes, at)?;
            ensure!((1..=MAX_MORPH_TARGETS).contains(&targets), "binary mesh has invalid morph target count");
            at += 4;
            let records = count.checked_mul(targets).context("binary mesh morph count overflows")?;
            let block = Block::take(bytes, &mut at, records, MESH_MORPH_FLOATS * 4)?;
            Some((u32::try_from(targets)?, block))
        } else {
            None
        };
        let skin = if flags & MESH_SKIN != 0 {
            let records = Block::take(bytes, &mut at, count, SKIN_STRIDE)?;
            let rig = at..bytes.len();
            at = bytes.len();
            Some((records, rig))
        } else {
            None
        };
        let instances = if flags & MESH_INSTANCES != 0 {
            let count = word(bytes, at)?;
            at += 4;
            let bounds = Block::take(bytes, &mut at, 6, 4)?.range();
            let records = Block::take(bytes, &mut at, count, INSTANCE_FLOATS * 4)?;
            Some(Instances { bounds, records })
        } else {
            None
        };
        let meshlets = if flags & MESH_MESHLETS != 0 {
            let subflags = u32::try_from(word(bytes, at)?)?;
            at += 4;
            ensure!(
                subflags != 0 && subflags & !(MESHLET_SUB_CLUSTERS | MESHLET_SUB_DAG) == 0,
                "unsupported binary meshlet subflags"
            );
            let ml = if subflags & MESHLET_SUB_CLUSTERS != 0 {
                let ml_count = word(bytes, at)?;
                let idx_count = word(bytes, at + 4)?;
                at += 8;
                let records = Block::take(bytes, &mut at, ml_count, size_of::<MeshletGpu>())?;
                let idx = Block::take(bytes, &mut at, idx_count, index_bytes)?;
                Some((records, idx))
            } else {
                None
            };
            let dag = if subflags & MESHLET_SUB_DAG != 0 {
                let cluster_count = word(bytes, at)?;
                let idx_count = word(bytes, at + 4)?;
                let level_count = u32::try_from(word(bytes, at + 8)?)?;
                let leaf_cluster_count = u32::try_from(word(bytes, at + 12)?)?;
                let root_cluster_count = u32::try_from(word(bytes, at + 16)?)?;
                at += 20;
                let clusters = Block::take(bytes, &mut at, cluster_count, size_of::<DagClusterGpu>())?;
                let idx = Block::take(bytes, &mut at, idx_count, index_bytes)?;
                Some(DagLayout { level_count, leaf_cluster_count, root_cluster_count, clusters, indices: idx })
            } else {
                None
            };
            Some(MeshletsLayout { meshlets: ml, dag })
        } else {
            None
        };
        ensure!(at == bytes.len(), "binary mesh has trailing bytes");
        Ok(Self { vertices, tangents, indices, lods, morphs, skin, instances, meshlets })
    }

    /// Record blocks only: counts, errors, bounds and a skin's variable
    /// rig data retain their original byte order.
    pub fn blocks(&self) -> impl Iterator<Item = &Block> {
        [&self.vertices, &self.indices]
            .into_iter()
            .chain(self.tangents.iter())
            .chain(self.lods.iter().map(|(_, block)| block))
            .chain(self.morphs.iter().map(|(_, block)| block))
            .chain(self.skin.iter().map(|(block, _)| block))
            .chain(self.instances.iter().map(|instances| &instances.records))
            .chain(self.meshlets.iter().flat_map(|ml| {
                ml.meshlets
                    .iter()
                    .flat_map(|(records, idx)| [records, idx])
                    .chain(ml.dag.iter().flat_map(|dag| [&dag.clusters, &dag.indices]))
            }))
    }
}
