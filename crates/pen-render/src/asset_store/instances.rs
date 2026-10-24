//! Persistent instance records belong to the mesh asset, not a frame packet.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use super::Mesh;
use super::layout::{INSTANCE_FLOATS, Instances};
use crate::culling::Sphere;

/// Five vec4s at shader locations 3..7. Their meaning belongs to the shader.
/// Bounds enclose all instances in mesh space, before shader deformation.
#[derive(Deserialize)]
pub(crate) struct MeshInstances {
    pub data: Vec<f32>,
    /// JSON records can use fixed-point integers. Bounds stay in mesh units;
    /// binary files and GPU records always contain the expanded values.
    #[serde(default = "unit_scale")]
    pub scale: f32,
    pub min: [f32; 3],
    pub max: [f32; 3],
}

fn unit_scale() -> f32 {
    1.0
}

impl MeshInstances {
    pub fn count(&self) -> u32 {
        u32::try_from(self.data.len() / INSTANCE_FLOATS).expect("validated instance count")
    }

    pub(super) fn write(&self, bytes: &mut Vec<u8>) -> Result<()> {
        bytes.extend(self.count().to_le_bytes());
        for value in self.min.iter().chain(&self.max).chain(&self.data) {
            bytes.extend(value.to_le_bytes());
        }
        Ok(())
    }

    pub(super) fn read(bytes: &[u8], layout: &Instances) -> Result<Self> {
        let values =
            |bytes: &[u8]| bytes.as_chunks::<4>().0.iter().map(|&value| f32::from_le_bytes(value)).collect::<Vec<_>>();
        let bounds = values(&bytes[layout.bounds.clone()]);
        let data = values(&bytes[layout.records.range()]);
        Ok(Self { min: bounds[..3].try_into()?, max: bounds[3..].try_into()?, data, scale: 1.0 })
    }
}

impl Mesh {
    pub(super) fn set_instances(&mut self, mut instances: Option<MeshInstances>) -> Result<()> {
        if let Some(instances) = &mut instances {
            ensure!(self.skin.is_none(), "a mesh cannot have both a skin and persistent instances");
            ensure!(
                !instances.data.is_empty() && instances.data.len().is_multiple_of(INSTANCE_FLOATS),
                "mesh instances require 20 floats per record"
            );
            u32::try_from(instances.data.len() / INSTANCE_FLOATS)
                .context("mesh has more instances than a u32 counts")?;
            ensure!(
                instances.scale.is_finite() && instances.scale > 0.0,
                "mesh instance scale must be finite and positive"
            );
            if instances.scale != 1.0 {
                for value in &mut instances.data {
                    *value *= instances.scale;
                }
                instances.scale = 1.0;
            }
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
    fn fixed_point_records_expand_once_without_scaling_bounds() {
        let mut source = fixture();
        source["instances"]["data"] = json!((-20..20).collect::<Vec<_>>());
        source["instances"]["scale"] = json!(1.0 / 4096.0);
        let json = serde_json::to_vec(&source).unwrap();
        let packed = pack_mesh(&json).unwrap();
        assert_eq!(pack_mesh(&packed).unwrap(), packed);
        for bytes in [&json, &packed] {
            let instance = parse_mesh(bytes).unwrap().instances.unwrap();
            assert_eq!(instance.scale, 1.0);
            assert_eq!(instance.data, (-20..20).map(|n| n as f32 / 4096.0).collect::<Vec<_>>());
            assert_eq!((instance.min, instance.max), ([-1., -2., -3.], [1., 2., 3.]));
        }
        for scale in [0.0, -1.0, 1e100] {
            source["instances"]["scale"] = json!(scale);
            assert!(parse_mesh(&serde_json::to_vec(&source).unwrap()).is_err());
        }
        source["instances"]["scale"] = json!(f32::MAX);
        assert!(parse_mesh(&serde_json::to_vec(&source).unwrap()).is_err());
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
        let mesh = parse_mesh(&serde_json::to_vec(&fixture()).unwrap()).unwrap();
        let mut bytes = super::super::pack_uncompressed(&mesh).unwrap();
        let end = bytes.len();
        bytes[end - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(parse_mesh(&bytes).is_err());
    }
}
