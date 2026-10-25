//! Optional author-approved rounding before the existing byte-plane codec.
//! No new runtime format: discarded fraction bits become compressible zero
//! planes. UVs, shader attributes, indices, skinning and instances stay exact.
use anyhow::Result;

use super::layout::Layout;

fn rounded(value: f32, bits: u8) -> f32 {
    let dropped = 23 - u32::from(bits);
    let mask = (1u32 << dropped) - 1;
    let raw = value.to_bits();
    // Nearest, ties to even; preserve sign, including negative zero.
    let half = mask.div_ceil(2);
    let low = raw & mask;
    let up = low > half || (low == half && (raw >> dropped) & 1 != 0);
    let reduced = f32::from_bits((raw & !mask) + if up { mask + 1 } else { 0 });
    if reduced.is_finite() { reduced } else { value }
}

pub(super) fn reduce(mut raw: Vec<u8>, bits: u8) -> Result<Vec<u8>> {
    let layout = Layout::read(&raw)?;
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let coordinate = |row: &[u8], axis: usize| f32::from_le_bytes(row[axis * 4..axis * 4 + 4].try_into().unwrap());
    for row in raw[layout.vertices.range()].chunks_exact(layout.vertices.stride) {
        for axis in 0..3 {
            min[axis] = min[axis].min(coordinate(row, axis));
            max[axis] = max[axis].max(coordinate(row, axis));
        }
    }
    for row in raw[layout.vertices.range()].chunks_exact_mut(layout.vertices.stride) {
        for axis in 0..6 {
            let value = coordinate(row, axis);
            // Retaining extrema keeps the authored bounding box exactly.
            if axis < 3 && (value == min[axis] || value == max[axis]) {
                continue;
            }
            row[axis * 4..axis * 4 + 4].copy_from_slice(&rounded(value, bits).to_le_bytes());
        }
    }
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_store::{MESH_VERTEX_FLOATS, pack_mesh, pack_uncompressed, parse_mesh, tests::skinned_triangle};

    #[test]
    fn relative_error_is_bounded_and_rounding_is_stable() {
        for bits in [16, 20, 22] {
            for raw in (0..=u32::MAX).step_by(65537) {
                let value = f32::from_bits(raw);
                if !value.is_finite() {
                    continue;
                }
                let reduced = rounded(value, bits);
                assert!(reduced.is_finite());
                assert_eq!(reduced.to_bits(), rounded(reduced, bits).to_bits());
                let bound = f64::from(value.abs()) * 2f64.powi(-i32::from(bits) - 1)
                    + f64::from(f32::from_bits(1 << (22 - u32::from(bits))));
                assert!((f64::from(value) - f64::from(reduced)).abs() <= bound);
            }
        }
        assert_eq!(rounded(-0.0, 16).to_bits(), (-0.0f32).to_bits());
        assert_eq!(rounded(f32::MAX, 16), f32::MAX);
    }

    #[test]
    fn only_opted_in_coordinates_change_and_skin_data_stays_exact() {
        let mut source: serde_json::Value = serde_json::from_str(&skinned_triangle()).unwrap();
        source["positions"] = serde_json::json!([0, 0, 0, 1, 1, 0, 0.123456, 0.654321, 0]);
        source["normals"] = serde_json::json!([0.123456, 1, 0, 0.123456, 1, 0, 0.123456, 1, 0]);
        let original = parse_mesh(&serde_json::to_vec(&source).unwrap()).unwrap();
        let raw = pack_uncompressed(&original).unwrap();
        let reduced = reduce(raw.clone(), 16).unwrap();
        let layout = Layout::read(&raw).unwrap();
        assert_eq!(&raw[layout.vertices.range().end..], &reduced[layout.vertices.range().end..]);
        let decoded = parse_mesh(&reduced).unwrap();
        for (a, b) in original
            .vertices
            .as_chunks::<MESH_VERTEX_FLOATS>()
            .0
            .iter()
            .zip(decoded.vertices.as_chunks::<MESH_VERTEX_FLOATS>().0)
        {
            assert_eq!(&a[6..], &b[6..]);
        }
        assert_ne!(original.vertices, decoded.vertices);
        let plain = pack_mesh(&serde_json::to_vec(&source).unwrap()).unwrap();
        assert_eq!(parse_mesh(&plain).unwrap().vertices, original.vertices);
        source["precision"] = 16.into();
        let packed = pack_mesh(&serde_json::to_vec(&source).unwrap()).unwrap();
        assert!(packed.len() <= plain.len());
        assert_eq!(pack_mesh(&packed).unwrap(), packed);
        for value in [0, 15, 24, 255] {
            source["precision"] = value.into();
            assert!(pack_mesh(&serde_json::to_vec(&source).unwrap()).is_err());
        }
    }
}
