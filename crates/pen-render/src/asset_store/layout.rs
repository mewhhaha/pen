//! The checked PMSH v1 layout shared by the reader and byte-plane codec.
//! Count tables and bounds stay untransposed, so this also describes the
//! inflated v2 stream before its records have been restored.
use std::ops::Range;

use anyhow::{Context, Result, ensure};

use super::{
    MAX_LODS, MESH_ATTRIBUTES, MESH_HEADER, MESH_INSTANCES, MESH_LODS, MESH_MAGIC, MESH_SHORT_INDICES, MESH_SKIN,
    MESH_VERSION, MESH_VERTEX_FLOATS,
};

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

pub(super) struct Layout {
    pub vertices: Block,
    pub indices: Block,
    pub lods: Vec<(f32, Block)>,
    pub skin: Option<(Block, Range<usize>)>,
    pub instances: Option<Instances>,
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
        let flags = bytes[5];
        ensure!(
            flags & !(MESH_ATTRIBUTES | MESH_SHORT_INDICES | MESH_SKIN | MESH_LODS | MESH_INSTANCES) == 0,
            "unsupported binary mesh flags"
        );
        ensure!(
            flags & (MESH_SKIN | MESH_INSTANCES) != (MESH_SKIN | MESH_INSTANCES),
            "a mesh cannot have both a skin and persistent instances"
        );
        let count = word(bytes, 8)?;
        let floats = if flags & MESH_ATTRIBUTES != 0 { MESH_VERTEX_FLOATS } else { MESH_VERTEX_FLOATS - 2 };
        let index_bytes = if flags & MESH_SHORT_INDICES != 0 { 2 } else { 4 };
        let mut at = MESH_HEADER;
        let vertices = Block::take(bytes, &mut at, count, floats * 4)?;
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
        ensure!(at == bytes.len(), "binary mesh has trailing bytes");
        Ok(Self { vertices, indices, lods, skin, instances })
    }

    /// Record blocks only: counts, errors, bounds and a skin's variable
    /// rig data retain their original byte order.
    pub fn blocks(&self) -> impl Iterator<Item = &Block> {
        [&self.vertices, &self.indices]
            .into_iter()
            .chain(self.lods.iter().map(|(_, block)| block))
            .chain(self.skin.iter().map(|(block, _)| block))
            .chain(self.instances.iter().map(|instances| &instances.records))
    }
}
