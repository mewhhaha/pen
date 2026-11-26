//! Optional author-approved rounding before the existing byte-plane codec.
//! No new runtime format: discarded fraction bits become compressible zero
//! planes. UVs, shader attributes, indices, skinning and instances stay exact.
use super::{MESH_VERTEX_FLOATS, Mesh, Sphere, valid_normal};

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

pub(super) fn reduce(mesh: &mut Mesh, bits: u8) {
    let (vertices, _) = mesh.vertices.as_chunks_mut::<MESH_VERTEX_FLOATS>();
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for row in &*vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(row[axis]);
            max[axis] = max[axis].max(row[axis]);
        }
    }
    for row in vertices {
        for axis in 0..3 {
            let value = row[axis];
            // Retaining extrema keeps the authored bounding box exactly.
            if value != min[axis] && value != max[axis] {
                row[axis] = rounded(value, bits).clamp(min[axis], max[axis]);
            }
        }
        let normal = [row[3], row[4], row[5]].map(|value| rounded(value, bits));
        if valid_normal(&normal) {
            row[3..6].copy_from_slice(&normal);
        }
    }
    if mesh.instances.is_none() {
        mesh.bounds = Sphere::around(&mesh.vertices, MESH_VERTEX_FLOATS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_store::{layout::Layout, pack_mesh, pack_uncompressed, parse_mesh, tests::skinned_triangle};

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
        let mut mesh = parse_mesh(&serde_json::to_vec(&source).unwrap()).unwrap();
        reduce(&mut mesh, 16);
        let reduced = pack_uncompressed(&mesh).unwrap();
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
        assert_eq!(parse_mesh(&packed).unwrap().vertices, decoded.vertices);
        assert_eq!(pack_mesh(&packed).unwrap(), packed);
        for value in [0, 15, 24, 255] {
            source["precision"] = value.into();
            assert!(pack_mesh(&serde_json::to_vec(&source).unwrap()).is_err());
        }
    }
}
