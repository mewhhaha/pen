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

pub(super) fn encode_recipe(mesh: &super::Mesh) -> Result<Vec<u8>> {
    ensure!(mesh.lods.len() == 1, "instance recipe meshes cannot have extra LODs");
    let instances = mesh.instances.as_ref().context("missing instance recipe")?;
    let component = |start: usize, count: usize| {
        mesh.vertices
            .as_chunks::<{ super::MESH_VERTEX_FLOATS }>()
            .0
            .iter()
            .flat_map(|v| v[start..start + count].iter().copied())
            .collect::<Vec<_>>()
    };
    let json = serde_json::to_vec(&serde_json::json!({
        "positions":component(0,3),"normals":component(3,3),"uvs":component(6,2),"attributes":component(8,2),
        "tangents":mesh.tangents,"indices":mesh.indices,"lods":[],
        "instances":{"recipe":instances.recipe,"min":instances.min,"max":instances.max}
    }))?;
    ensure!(json.len() <= 64 * 1024 * 1024, "instance recipe JSON exceeds its size limit");
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(&json)?;
    let mut out = b"PMRP".to_vec();
    out.extend(u32::try_from(json.len())?.to_le_bytes());
    out.extend(encoder.finish()?);
    Ok(out)
}

pub(super) fn decode_recipe(bytes: &[u8]) -> Result<Vec<u8>> {
    ensure!(bytes.len() >= 8, "instance recipe header ends early");
    let length = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    ensure!((2..=64 * 1024 * 1024).contains(&length), "instance recipe expands beyond its size limit");
    let mut out = vec![0; length + 1];
    let mut decoder = Decompress::new(true);
    let status = decoder.decompress(&bytes[8..], &mut out, FlushDecompress::Finish)?;
    ensure!(
        status == Status::StreamEnd
            && decoder.total_out() == length as u64
            && decoder.total_in() == (bytes.len() - 8) as u64,
        "compressed instance recipe is truncated or has trailing data"
    );
    out.truncate(length);
    ensure!(out.starts_with(b"{"), "instance recipe must contain mesh JSON");
    Ok(out)
}

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

pub(super) fn decode(bytes: &[u8]) -> Result<(Cow<'_, [u8]>, Layout)> {
    if bytes.get(4) != Some(&VERSION) {
        return Ok((Cow::Borrowed(bytes), Layout::read(bytes)?));
    }
    let mut raw = inflate(bytes)?;
    let layout = Layout::read(&raw)?;
    transpose(&mut raw, &layout, &mut Vec::new(), true);
    Ok((Cow::Owned(raw), layout))
}

#[cfg(feature = "simplify")]
pub(super) fn validate_layout(bytes: &[u8]) -> Result<()> {
    if bytes.get(4) == Some(&VERSION) {
        Layout::read(&inflate(bytes)?)?;
    } else {
        Layout::read(bytes)?;
    }
    Ok(())
}

fn inflate(bytes: &[u8]) -> Result<Vec<u8>> {
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
    Ok(raw)
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
        assert_eq!(decode(&packed).unwrap().0.as_ref(), raw);
        let parsed = parse_mesh(&packed).unwrap();
        assert_eq!(parsed.vertices[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(parsed.vertices[1].to_bits(), 1);
        assert_eq!(pack_mesh(&packed).unwrap(), packed);
    }

    #[cfg(feature = "simplify")]
    #[test]
    fn packaging_keeps_existing_packed_layout_and_compression() {
        use crate::pack_mesh_with_lods;

        let raw = pack_uncompressed(&parse_mesh(skinned_triangle().as_bytes()).unwrap()).unwrap();
        let compressed = encode(raw.clone()).unwrap();
        assert_eq!(raw[4], 1);
        assert_eq!(compressed[4], VERSION);
        // Re-encoding this v1 asset normally chooses v2.
        assert_eq!(pack_mesh(&raw).unwrap(), compressed);
        for bytes in [raw, compressed] {
            assert_eq!(pack_mesh_with_lods(&bytes).unwrap(), bytes);
            let mut bad_header = bytes.clone();
            bad_header[6] = 1;
            assert!(pack_mesh_with_lods(&bad_header).is_err());

            let mut trailing = bytes;
            trailing.push(0);
            if trailing[4] == 1 {
                // A v1 skin's variable-length rig occupies the rest of the layout.
                trailing[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
            }
            assert!(pack_mesh_with_lods(&trailing).is_err());
        }
    }

    #[test]
    fn tangents_round_trip_bit_exactly_through_the_byte_planes() {
        let mut mesh = parse_mesh(skinned_triangle().as_bytes()).unwrap();
        // Signed zero, a subnormal and an ordinary value of each sign.
        let tangents = vec![-0.0, f32::from_bits(1), 0.6, 1.0, 0.0, -1.0, 0.0, -1.0, 0.8, 0.0, 0.6, 1.0];
        mesh.set_tangents(Some(tangents.clone())).unwrap();
        let raw = pack_uncompressed(&mesh).unwrap();
        let packed = encode(raw.clone()).unwrap();
        assert_eq!(decode(&packed).unwrap().0.as_ref(), raw);
        let parsed = parse_mesh(&packed).unwrap();
        let bits = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(parsed.tangents.as_ref().unwrap()), bits(&tangents));
        assert_eq!(pack_mesh(&packed).unwrap(), packed);
    }

    #[test]
    fn morph_targets_round_trip_bit_exactly_beside_a_skin() {
        let mut mesh = parse_mesh(skinned_triangle().as_bytes()).unwrap();
        let deltas = vec![
            -0.0,
            f32::from_bits(1),
            0.25,
            0.0,
            0.0,
            0.5,
            0.5,
            -0.25,
            0.0,
            0.1,
            0.2,
            0.3,
            0.0,
            1.0,
            -0.5,
            0.0,
            0.0,
            0.0,
        ];
        mesh.set_morphs(Some(crate::asset_store::MorphTargets { targets: 1, deltas: deltas.clone() })).unwrap();
        let raw = pack_uncompressed(&mesh).unwrap();
        let packed = encode(raw.clone()).unwrap();
        assert_eq!(decode(&packed).unwrap().0.as_ref(), raw);
        let parsed = parse_mesh(&packed).unwrap();
        let bits = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        assert_eq!(parsed.morphs.as_ref().unwrap().targets, 1);
        assert_eq!(bits(&parsed.morphs.as_ref().unwrap().deltas), bits(&deltas));
        assert!(parsed.skin.is_some());
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
