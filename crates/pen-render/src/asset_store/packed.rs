//! PMSH v2 wraps v1 in zlib after grouping matching bytes of its records.
//! All floats, indices, LODs, skin data and instance records stay bit exact.
//! The 16-byte outer header is magic, version, mesh flags, two zero bytes,
//! expanded v1 length (u32 LE), and four zero bytes. Tiny/incompressible
//! meshes keep v1. Both native and browser loaders use this decoder.
use std::borrow::Cow;
use std::io::Write;

use anyhow::{Context, Result, ensure};
use flate2::{Compression, Decompress, FlushDecompress, Status, write::ZlibEncoder};

use super::{
    MAX_LODS, MESH_ATTRIBUTES, MESH_HEADER, MESH_INSTANCES, MESH_LODS, MESH_MAGIC, MESH_SHORT_INDICES, MESH_SKIN,
};

const VERSION: u8 = 2;
// Bound allocations before trusting a compressed asset's claimed length.
const MAX_EXPANDED: usize = 256 * 1024 * 1024;

pub(super) fn encode(mut raw: Vec<u8>) -> Result<Vec<u8>> {
    if raw.len() > MAX_EXPANDED {
        return Ok(raw);
    }
    let blocks = blocks(&raw)?;
    let mut scratch = Vec::new();
    transpose(&mut raw, &blocks, &mut scratch, false);
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(&raw)?;
    let compressed = encoder.finish()?;
    if compressed.len() + MESH_HEADER >= raw.len() {
        transpose(&mut raw, &blocks, &mut scratch, true);
        return Ok(raw);
    }
    let mut out = Vec::with_capacity(MESH_HEADER + compressed.len());
    out.extend(MESH_MAGIC);
    out.extend([VERSION, raw[5], 0, 0]);
    out.extend(u32::try_from(raw.len())?.to_le_bytes());
    out.extend([0; 4]);
    out.extend(compressed);
    Ok(out)
}

pub(super) fn decode(bytes: &[u8]) -> Result<Cow<'_, [u8]>> {
    if bytes.get(4) != Some(&VERSION) {
        return Ok(Cow::Borrowed(bytes));
    }
    ensure!(bytes.len() >= MESH_HEADER, "compressed mesh ends in its header");
    ensure!(bytes[6..8] == [0; 2] && bytes[12..16] == [0; 4], "unsupported compressed mesh header");
    let length = word(bytes, 8)?;
    ensure!((MESH_HEADER..=MAX_EXPANDED).contains(&length), "compressed mesh expands beyond its size limit");
    // One extra byte detects expansion beyond the declared length, including
    // a stream whose checksum is valid but whose header underreports its size.
    let mut raw = Vec::new();
    raw.try_reserve_exact(length + 1).context("allocating decoded mesh")?;
    raw.resize(length + 1, 0);
    let mut decoder = Decompress::new(true);
    let payload = &bytes[MESH_HEADER..];
    let status = decoder.decompress(payload, &mut raw, FlushDecompress::Finish)?;
    ensure!(
        status == Status::StreamEnd
            && decoder.total_out() == length as u64
            && decoder.total_in() == payload.len() as u64,
        "compressed mesh is truncated, has trailing bytes, or does not match its declared size"
    );
    raw.truncate(length);
    ensure!(&raw[..4] == MESH_MAGIC && raw[4] == 1 && raw[5] == bytes[5], "invalid compressed mesh contents");
    let blocks = blocks(&raw)?;
    transpose(&mut raw, &blocks, &mut Vec::new(), true);
    Ok(Cow::Owned(raw))
}

struct Block {
    start: usize,
    count: usize,
    stride: usize,
}

fn word(bytes: &[u8], at: usize) -> Result<usize> {
    let value = bytes.get(at..at + 4).context("binary mesh ends in its counts")?;
    Ok(u32::from_le_bytes(value.try_into()?) as usize)
}

/// Count tables and bounds are left in place, so the inverse can find each
/// block without decoding its contents. A skin's variable-size rig stays as is.
fn blocks(bytes: &[u8]) -> Result<Vec<Block>> {
    ensure!(bytes.len() >= MESH_HEADER, "binary mesh ends in its header");
    let flags = bytes[5];
    ensure!(flags & MESH_SKIN == 0 || flags & MESH_INSTANCES == 0, "mesh has both skin and instances");
    let vertices = word(bytes, 8)?;
    let indices = word(bytes, 12)?;
    let index_bytes = if flags & MESH_SHORT_INDICES != 0 { 2 } else { 4 };
    let mut blocks = Vec::new();
    let mut at = MESH_HEADER;
    let mut add = |at: &mut usize, count: usize, stride: usize| -> Result<()> {
        let end = count.checked_mul(stride).and_then(|size| at.checked_add(size));
        let end = end.filter(|&end| end <= bytes.len()).context("binary mesh size does not match its counts")?;
        blocks.push(Block { start: *at, count, stride });
        *at = end;
        Ok(())
    };
    add(&mut at, vertices, if flags & MESH_ATTRIBUTES != 0 { 40 } else { 32 })?;
    add(&mut at, indices, index_bytes)?;
    if flags & MESH_LODS != 0 {
        let levels = word(bytes, at)?;
        ensure!(levels < MAX_LODS, "binary mesh has too many levels of detail");
        let table = at + 4;
        at = table + levels * 8;
        for level in 0..levels {
            add(&mut at, word(bytes, table + level * 8)?, index_bytes)?;
        }
    }
    if flags & MESH_SKIN != 0 {
        add(&mut at, vertices, 24)?;
    } else if flags & MESH_INSTANCES != 0 {
        let count = word(bytes, at)?;
        at += 28;
        add(&mut at, count, 80)?;
    }
    ensure!(flags & MESH_SKIN != 0 || at == bytes.len(), "binary mesh has trailing bytes");
    Ok(blocks)
}

fn transpose(bytes: &mut [u8], blocks: &[Block], scratch: &mut Vec<u8>, inverse: bool) {
    for block in blocks {
        let data = &mut bytes[block.start..block.start + block.count * block.stride];
        scratch.clear();
        scratch.extend_from_slice(data);
        for column in 0..block.stride {
            for row in 0..block.count {
                let record = row * block.stride + column;
                let plane = column * block.count + row;
                if inverse {
                    data[record] = scratch[plane];
                } else {
                    data[plane] = scratch[record];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_store::{pack_mesh, pack_uncompressed, parse_mesh, tests::skinned_triangle};

    #[test]
    fn records_indices_lods_and_skin_round_trip_bit_exactly() {
        let mut mesh = parse_mesh(skinned_triangle().as_bytes()).unwrap();
        // Include signed zero and the smallest subnormal; these must not be
        // rounded or treated as equivalent by a lossless file codec.
        mesh.vertices[0] = -0.0;
        mesh.vertices[1] = f32::from_bits(1);
        let raw = pack_uncompressed(&mesh).unwrap();
        let packed = encode(raw.clone()).unwrap();
        assert_eq!(packed[4], VERSION);
        assert!(packed.len() < raw.len());
        assert_eq!(decode(&packed).unwrap().as_ref(), raw);
        let parsed = parse_mesh(&packed).unwrap();
        assert_eq!(parsed.vertices[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(parsed.vertices[1].to_bits(), 1);
        assert_eq!(pack_mesh(&packed).unwrap(), packed);
    }

    #[test]
    fn rejects_truncation_corruption_trailers_and_false_expanded_sizes() {
        let raw = pack_uncompressed(&parse_mesh(skinned_triangle().as_bytes()).unwrap()).unwrap();
        let packed = encode(raw.clone()).unwrap();
        for end in 0..packed.len() {
            assert!(parse_mesh(&packed[..end]).is_err(), "truncated at {end}");
        }
        let mut extra = packed.clone();
        extra.push(0);
        assert!(parse_mesh(&extra).is_err());
        let mut corrupt = packed.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(parse_mesh(&corrupt).is_err());
        for length in [0, raw.len() - 1, raw.len() + 1, MAX_EXPANDED + 1, u32::MAX as usize] {
            let mut broken = packed.clone();
            broken[8..12].copy_from_slice(&u32::try_from(length).unwrap().to_le_bytes());
            assert!(parse_mesh(&broken).is_err(), "expanded length {length}");
        }
        // A valid zlib stream must not permit forged internal count tables.
        let mut forged = raw;
        forged[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut compressor = ZlibEncoder::new(Vec::new(), Compression::best());
        compressor.write_all(&forged).unwrap();
        let mut broken = packed[..MESH_HEADER].to_vec();
        broken.extend(compressor.finish().unwrap());
        assert!(parse_mesh(&broken).is_err());
    }
}
