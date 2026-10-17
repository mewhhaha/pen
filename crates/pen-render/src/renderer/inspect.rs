// What pen's dev tools ask of a renderer: where a catalog asset stands, and
// a material's texture, loaded, for them to draw pictures into.
use anyhow::Result;
use pen_protocol::{Catalog, devtools_asset, devtools_asset_state};

use super::{GpuAsset, State};
use crate::asset_store::AssetKind;

impl State {
    /// Where the catalog's mesh or material (DevtoolsAsset `kind`) `key`
    /// stands (DevtoolsAssetState) and its revision; sounds are not the
    /// renderer's, and stand unused.
    pub fn standing(&self, kind: u32, key: &str) -> (u32, u64) {
        match kind {
            devtools_asset::MESH => self.assets.standing(AssetKind::Mesh, key),
            devtools_asset::MATERIAL => self.assets.standing(AssetKind::Material, key),
            _ => (devtools_asset_state::UNUSED, 0),
        }
    }

    /// Material `index`'s texture once it is loaded (starting to load the
    /// material and the texture until then).
    pub fn material_texture(&mut self, catalog: &Catalog, index: u32) -> Result<Option<wgpu::Texture>> {
        if index as usize >= catalog.materials.len() {
            anyhow::bail!("there is no material {index}");
        }
        let Some(material) = self.material(catalog, index)? else { return Ok(None) };
        let Some(key) = &material.material.texture else { return Ok(None) };
        let slot = self.resolve(AssetKind::Texture, key)?;
        Ok(match self.assets.current(slot) {
            Some(GpuAsset::Texture(texture)) => Some(texture.texture.clone()),
            _ => None,
        })
    }
}
