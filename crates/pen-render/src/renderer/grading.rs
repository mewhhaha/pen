// The grade's look-up tables: the frame names each by a material whose texture is
// the table's strip (grade.rs). A table is used once its finest level is resident
// and its shape is a strip's; until then the grade goes without it.
use anyhow::Result;
use pen_protocol::{Catalog, RenderFrame};

use super::{GpuAsset, State};
use crate::asset_store::AssetKind;
use crate::grade::{self, Lut};

impl State {
    /// The tables the frame's grade uses, as `grade::tables` makes them.
    pub(super) fn look_up_tables(&mut self, frame: &RenderFrame, catalog: &Catalog) -> Result<[Option<Lut>; 2]> {
        let grade = &frame.grade;
        let mut found = [None, None];
        for (found, named) in found.iter_mut().zip([grade.lut_a, grade.lut_b]) {
            if named == 0 {
                continue;
            }
            let Some(material) = self.material(catalog, named - 1)? else { continue };
            let Some(key) = &material.material.texture else { continue };
            let slot = self.resolve(AssetKind::Texture, key)?;
            let Some(GpuAsset::Texture(texture)) = self.assets.current_mut(slot) else { continue };
            let (width, size) = (texture.texture.width(), texture.texture.height());
            if texture.resident != 0 || !(2..=64).contains(&size) || width != size * size {
                continue;
            }
            texture.seen = true;
            *found = Some(Lut { view: texture.linear.clone(), id: texture.view_id, size });
        }
        let [a, b] = found;
        Ok(grade::tables(grade, a, b))
    }
}
