//! Persistent instance records belong to the mesh asset, not a frame packet.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use super::Mesh;
use crate::culling::Sphere;

/// Five vec4s at shader locations 3..7. Their meaning belongs to the shader.
/// Bounds enclose all instances in mesh space, before shader deformation.
#[derive(Deserialize)]
pub(crate) struct MeshInstances {
    pub data: Vec<f32>,
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl MeshInstances {
    pub fn count(&self) -> u32 {
        u32::try_from(self.data.len() / 20).expect("validated instance count")
    }

    pub(super) fn write(&self, bytes: &mut Vec<u8>) -> Result<()> {
        bytes.extend(self.count().to_le_bytes());
        for value in self.min.iter().chain(&self.max).chain(&self.data) {
            bytes.extend(value.to_le_bytes());
        }
        Ok(())
    }

    pub(super) fn read(bytes: &[u8]) -> Result<Self> {
        let header = bytes.get(..28).context("binary mesh ends in its instance header")?;
        let count = u32::from_le_bytes(header[..4].try_into()?) as usize;
        let end = count.checked_mul(80).and_then(|size| size.checked_add(28));
        ensure!(end == Some(bytes.len()), "binary mesh instance size does not match its count");
        let values: Vec<f32> = bytes[4..].as_chunks::<4>().0.iter().map(|&value| f32::from_le_bytes(value)).collect();
        Ok(Self { min: values[..3].try_into()?, max: values[3..6].try_into()?, data: values[6..].to_vec() })
    }
}

impl Mesh {
    pub(super) fn set_instances(&mut self, instances: Option<MeshInstances>) -> Result<()> {
        if let Some(instances) = &instances {
            ensure!(self.skin.is_none(), "a mesh cannot have both a skin and persistent instances");
            ensure!(
                !instances.data.is_empty() && instances.data.len().is_multiple_of(20),
                "mesh instances require 20 floats per record"
            );
            u32::try_from(instances.data.len() / 20).context("mesh has more instances than a u32 counts")?;
            ensure!(instances.data.iter().all(|v| v.is_finite()), "mesh instance data must be finite");
            ensure!(
                (0..3).all(|axis| instances.min[axis].is_finite()
                    && instances.max[axis].is_finite()
                    && instances.min[axis] <= instances.max[axis]),
                "mesh instance bounds must be finite and ordered"
            );
            self.bounds = Sphere::around(&[instances.min, instances.max].concat(), 3);
            ensure!(self.bounds.radius.is_finite(), "mesh instance bounds exceed float32 range");
        }
        self.instances = instances;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::asset_store::{mesh_bounds, pack_mesh, parse_mesh};
    use serde_json::json;

    fn fixture() -> serde_json::Value {
        json!({"positions":[9.,0.,0.,10.,0.,0.,9.,1.,0.], "normals":[0.,0.,1.,0.,0.,1.,0.,0.,1.],
            "uvs":[0.,0.,1.,0.,0.,1.], "indices":[0,1,2],
            "instances":{"data": vec![0.; 40], "min":[-1.,-2.,-3.], "max":[1.,2.,3.]}})
    }

    #[test]
    fn records_and_aggregate_bounds_survive_packing() {
        let bytes = serde_json::to_vec(&fixture()).unwrap();
        let packed = pack_mesh(&bytes).unwrap();
        assert_eq!(pack_mesh(&packed).unwrap(), packed);
        for bytes in [&bytes, &packed] {
            let mesh = parse_mesh(bytes).unwrap();
            assert_eq!(mesh.instances.unwrap().count(), 2);
            assert_eq!(mesh_bounds(bytes).unwrap(), ([-1., -2., -3.], [1., 2., 3.]));
            assert_eq!(mesh.bounds.center, [0.; 3]);
            assert!(mesh.bounds.radius >= 14f32.sqrt());
        }
        #[cfg(feature = "simplify")]
        assert_eq!(crate::simplify::pack_mesh_with_lods(&bytes).unwrap(), packed);
    }

    #[test]
    fn rejects_incomplete_records_bounds_and_binary_trailers() {
        let mut malformed = fixture();
        malformed["instances"]["data"] = json!([1.]);
        assert!(parse_mesh(&serde_json::to_vec(&malformed).unwrap()).is_err());
        malformed["instances"]["data"] = json!([]);
        assert!(parse_mesh(&serde_json::to_vec(&malformed).unwrap()).is_err());
        let mut malformed = fixture();
        malformed["instances"]["min"][0] = json!(2.);
        assert!(parse_mesh(&serde_json::to_vec(&malformed).unwrap()).is_err());
        let bytes = pack_mesh(&serde_json::to_vec(&fixture()).unwrap()).unwrap();
        for end in 0..bytes.len() {
            assert!(parse_mesh(&bytes[..end]).is_err(), "truncation at {end}");
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(parse_mesh(&extra).is_err());
        let mut both = bytes;
        both[5] |= 4;
        assert!(parse_mesh(&both).is_err());
    }

    #[test]
    fn rejects_nonfinite_records_in_binary() {
        let mut bytes = pack_mesh(&serde_json::to_vec(&fixture()).unwrap()).unwrap();
        let end = bytes.len();
        bytes[end - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(parse_mesh(&bytes).is_err());
    }
}
