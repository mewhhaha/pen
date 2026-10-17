// Fields: values the game keeps per material key and patches each frame, in
// one half-float texture each, which draws of that material read at binding
// 3 in place of the normal map (vertices too).
use pen_protocol::FieldPatch;

use super::assets::whole;
use super::{State, extent};

/// The bits of the half float nearest `value`, ties rounded up.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "bit fields of a float"
)]
pub(crate) fn half_of(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xff) as i32 - 112;
    let mantissa = bits & 0x7fffff;
    if exponent >= 31 {
        return (sign | 0x7c00) as u16;
    }
    if exponent > 0 {
        return (sign | (((exponent as u32) << 10) + ((mantissa + 0x1000) >> 13))) as u16;
    }
    if exponent < -10 {
        return sign as u16;
    }
    (sign | ((((mantissa | 0x800000) >> (1 - exponent)) + 0x1000) >> 13)) as u16
}

impl State {
    /// Writes a patch into its material's field, made (zeroed) when first
    /// patched or when its size changes.
    pub(super) fn write_field(&mut self, material: &str, patch: &FieldPatch) {
        let (width, height) = (patch.width.max(1), patch.height.max(1));
        let found = self.field_ids.get(material).copied();
        let index = match found {
            Some(index)
                if self.fields[index].1.texture.width() == width && self.fields[index].1.texture.height() == height =>
            {
                index
            }
            _ => {
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("field"),
                    size: extent(width, height),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let id = self.view_id();
                let field = whole(texture, id);
                match found {
                    Some(index) => {
                        let old = std::mem::replace(&mut self.fields[index].1, field);
                        self.forget_views(old.view_id);
                        index
                    }
                    None => {
                        self.fields.push((material.to_owned(), field));
                        self.field_ids.insert(material.to_owned(), self.fields.len() - 1);
                        self.fields.len() - 1
                    }
                }
            }
        };
        if patch.columns == 0 || patch.rows == 0 {
            return;
        }
        let halves: Vec<u16> = patch.values.iter().map(|&value| half_of(value)).collect();
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.fields[index].1.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: patch.x, y: patch.z, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&halves),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(patch.columns * 2),
                rows_per_image: Some(patch.rows),
            },
            extent(patch.columns, patch.rows),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_round_half_way_up() {
        for (value, half) in [
            (0.0, 0x0000),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (0.5, 0x3800),
            (65504.0, 0x7bff),
            (1e6, 0x7c00),
            (6.1035156e-5, 0x0400),
            (3.0e-5, 0x01f7),
            (1e-9, 0x0000),
            (0.1, 0x2e66),
            (1.0009766, 0x3c01),
        ] {
            assert_eq!(half_of(value), half, "{value}");
        }
    }
}
