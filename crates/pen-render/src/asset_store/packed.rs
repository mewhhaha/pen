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
    MESH_HEADER, MESH_MAGIC,
    layout::{Layout, word},
};

const VERSION: u8 = 2;
// Bound allocations before trusting a compressed asset's claimed length.
const MAX_EXPANDED: usize = 256 * 1024 * 1024;

pub(super) fn encode(mut raw: Vec<u8>) -> Result<Vec<u8>> {
    if raw.len() > MAX_EXPANDED {
        return Ok(raw);
    }
    let layout = Layout::read(&raw)?;
    let mut scratch = Vec::new();
    transpose(&mut raw, &layout, &mut scratch, false);
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(&raw)?;
    let compressed = encoder.finish()?;
    if compressed.len() + MESH_HEADER >= raw.len() {
        transpose(&mut raw, &layout, &mut scratch, true);
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
    let layout = Layout::read(&raw)?;
    transpose(&mut raw, &layout, &mut Vec::new(), true);
    Ok(Cow::Owned(raw))
}

fn transpose(bytes: &mut [u8], layout: &Layout, scratch: &mut Vec<u8>, inverse: bool) {
    for block in layout.blocks() {
        let data = &mut bytes[block.range()];
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
        for bytes in [&raw, &packed] {
            for end in 0..bytes.len() {
                assert!(parse_mesh(&bytes[..end]).is_err(), "version {} truncated at {end}", bytes[4]);
            }
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
