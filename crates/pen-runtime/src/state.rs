//! Optional interpretation of guest-owned state. Saving, loading and replay
//! need no adapter; inspection and host-assisted migration opt in to one.
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;

/// Adapter selection, shipped beside the asset catalog.
pub const CONFIG_FILE: &str = "saved.json";
pub const OPAQUE_CONFIG: &str = "{\"format\":\"opaque\"}\n";

/// A view supplied by an adapter, not a layout imposed on guest state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Section<'a> {
    pub key: u32,
    pub version: u32,
    pub width: u32,
    pub lanes: &'a [f32],
}

/// One proposed retry after the guest could not migrate a saved state.
/// The runtime validates the payload and restores it in a fresh guest.
pub trait Migration {
    fn state(&self) -> &[f32];
    /// Describe the actual second restore, including any migration done here.
    fn report(&self, response: &[f32]) -> Option<String>;
}

/// A format adapter selected by the application. `None` means unsupported
/// or unreadable, distinct from a supported view with no entries.
/// Implementations must not infer a format from magic numbers alone.
pub trait StateAdapter: Send + Sync {
    /// Reject an explicitly declared but unusable configuration before state
    /// operations. Ordinary frames can continue while the declaration is fixed.
    fn validate(&self) -> Result<()> {
        Ok(())
    }
    fn sections<'a>(&self, _state: &'a [f32]) -> Option<Vec<Section<'a>>> {
        None
    }
    /// Optional names for numeric lanes, used in divergence reports.
    fn lane_label(&self, _section: &Section<'_>, offset: usize) -> String {
        format!("lane {offset}")
    }
    fn tuned_values(&self, _state: &[f32]) -> Option<HashMap<u32, f32>> {
        None
    }
    fn report(&self, _response: &[f32]) -> Option<String> {
        None
    }
    fn migrate(&self, _state: &[f32], _response: &[f32]) -> Result<Option<Box<dyn Migration>>> {
        Ok(None)
    }
}

/// The default: guest state and restore responses have no host interpretation.
pub struct Opaque;
impl StateAdapter for Opaque {}

pub fn opaque() -> Arc<dyn StateAdapter> {
    Arc::new(Opaque)
}

struct Invalid(String);
impl StateAdapter for Invalid {
    fn validate(&self) -> Result<()> {
        anyhow::bail!("{}", self.0)
    }
}

pub fn invalid(message: String) -> Arc<dyn StateAdapter> {
    Arc::new(Invalid(message))
}
