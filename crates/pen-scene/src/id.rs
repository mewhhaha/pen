//! Stable deterministic 64-bit GUIDs for scenes, prefabs, assets, and hierarchy nodes.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Deterministic 64-bit FNV-1a hash helper.
pub const fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        i += 1;
    }
    h
}

/// Stable 64-bit asset identifier (serialized as 16-char lowercase hex for clean JSON diffs).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct AssetId(pub u64);

impl AssetId {
    pub fn from_path(path: &str) -> Self {
        Self(fnv1a64(path.as_bytes()))
    }
}

/// Stable 64-bit node identifier inside a scene or prefab hierarchy.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct NodeId(pub u64);

impl NodeId {
    pub fn from_seed(name: &str, parent: Option<NodeId>, ordinal: u32) -> Self {
        let mut buf = Vec::with_capacity(name.len() + 16);
        buf.extend_from_slice(&parent.map_or(0u64, |p| p.0).to_le_bytes());
        buf.extend_from_slice(&ordinal.to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        let h = fnv1a64(&buf);
        Self(if h == 0 { 1 } else { h })
    }

    /// Derives a collision-free instance node id when instantiating a prefab child under `instance_root`.
    pub fn scoped(instance_root: NodeId, prefab_node: NodeId) -> Self {
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(&instance_root.0.to_le_bytes());
        buf[8..].copy_from_slice(&prefab_node.0.to_le_bytes());
        let h = fnv1a64(&buf);
        Self(if h == 0 { 1 } else { h })
    }
}

macro_rules! impl_hex_id {
    ($ty:ident) => {
        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{:016x}", self.0)
            }
        }
        impl fmt::Debug for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($ty), "({:016x})"), self.0)
            }
        }
        impl Serialize for $ty {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(&format!("{:016x}", self.0))
            }
        }
        impl<'de> Deserialize<'de> for $ty {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let s = String::deserialize(deserializer)?;
                let trimmed = s.strip_prefix("0x").unwrap_or(&s);
                let val = u64::from_str_radix(trimmed, 16).map_err(serde::de::Error::custom)?;
                Ok(Self(val))
            }
        }
    };
}

impl_hex_id!(AssetId);
impl_hex_id!(NodeId);
