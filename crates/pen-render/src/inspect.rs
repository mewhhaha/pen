// What an asset file is, read as the renderer would read it but without a
// GPU, for tools that check or show assets: `pen check`'s asset lints, an
// asset browser. A file the renderer would refuse is refused here with the
// same message.
use anyhow::{Result, bail, ensure};

use crate::asset_store::{parse_material, parse_mesh};
use crate::packed_texture::{PackedReader, is_packed_texture, mip_levels};

/// A texture's size and levels.
#[derive(Clone, Debug, PartialEq)]
pub struct TextureInfo {
    pub width: u32,
    pub height: u32,
    /// What it is: `png`, `jpeg`, `gtex color` (and the other kinds),
    /// `ktx2` and its format.
    pub format: String,
    /// The levels the file holds.
    pub levels: u32,
    /// The levels the renderer samples: the file's own, or for an image a
    /// full chain it draws on the GPU.
    pub sampled_levels: u32,
}

/// The texture in `bytes`: its header read, and a packed texture's every
/// level inflated, so that a damaged one is refused.
pub fn inspect_texture(bytes: &[u8]) -> Result<TextureInfo> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        ensure!(bytes.len() >= 29 && &bytes[12..16] == b"IHDR", "PNG has no header");
        let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
        let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
        let depth = bytes[24];
        let color = match bytes[25] {
            0 => "grey",
            2 => "RGB",
            3 => "indexed",
            4 => "grey and alpha",
            6 => "RGBA",
            other => bail!("PNG has an unknown colour type {other}"),
        };
        ensure!(width > 0 && height > 0, "PNG is empty");
        let interlaced = if bytes[28] == 1 { ", interlaced" } else { "" };
        return Ok(TextureInfo {
            width,
            height,
            format: format!("png {color} {depth}-bit{interlaced}"),
            levels: 1,
            sampled_levels: mip_levels(width, height),
        });
    }
    if bytes.starts_with(&[0xff, 0xd8]) {
        let (width, height, progressive) = jpeg_size(bytes)?;
        return Ok(TextureInfo {
            width,
            height,
            format: if progressive { "jpeg progressive".into() } else { "jpeg".into() },
            levels: 1,
            sampled_levels: mip_levels(width, height),
        });
    }
    if is_packed_texture(bytes) {
        ensure!(bytes.len() >= 16, "packed texture requires a GTEX header");
        let kind = crate::packed_texture::kind_name(bytes[6]).unwrap_or("unknown");
        let mut reader = PackedReader::open(bytes.to_vec())?;
        let header = reader.header;
        while reader.next_level()?.is_some() {}
        return Ok(TextureInfo {
            width: header.width,
            height: header.height,
            format: format!("gtex {kind}"),
            levels: header.levels,
            sampled_levels: header.levels,
        });
    }
    if bytes.starts_with(KTX2) {
        return ktx2(bytes);
    }
    bail!("not an image the renderer reads (PNG, JPEG or GTEX)")
}

const KTX2: &[u8; 12] = b"\xabKTX 20\xbb\r\n\x1a\n";

/// A KTX2 texture's header: its size, format and levels (none means the
/// loader makes them).
fn ktx2(bytes: &[u8]) -> Result<TextureInfo> {
    ensure!(bytes.len() >= 80, "KTX2 header ends early");
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let (format, width, height, levels) = (word(12), word(20), word(24), word(40));
    ensure!(width > 0, "KTX2 texture is empty");
    let height = height.max(1);
    let name = match format {
        0 => "basis".to_owned(),
        37 => "rgba8".to_owned(),
        43 => "rgba8 srgb".to_owned(),
        139 => "bc4".to_owned(),
        141 => "bc5".to_owned(),
        145 => "bc7".to_owned(),
        146 => "bc7 srgb".to_owned(),
        147 | 148 => "etc2 rgb".to_owned(),
        151 | 152 => "etc2 rgba".to_owned(),
        157 => "astc 4x4".to_owned(),
        158 => "astc 4x4 srgb".to_owned(),
        other => format!("vkFormat {other}"),
    };
    let sampled = if levels == 0 { mip_levels(width, height) } else { levels };
    Ok(TextureInfo { width, height, format: format!("ktx2 {name}"), levels: levels.max(1), sampled_levels: sampled })
}

/// A baseline or progressive JPEG's size, from its frame header.
fn jpeg_size(bytes: &[u8]) -> Result<(u32, u32, bool)> {
    let mut at = 2;
    while at + 4 <= bytes.len() {
        ensure!(bytes[at] == 0xff, "JPEG markers are damaged");
        let marker = bytes[at + 1];
        if marker == 0xff {
            at += 1;
            continue;
        }
        let length = usize::from(u16::from_be_bytes([bytes[at + 2], bytes[at + 3]]));
        // Start of frame: baseline, extended or progressive.
        if matches!(marker, 0xc0..=0xc2) {
            ensure!(at + 9 <= bytes.len(), "JPEG frame header ends early");
            let height = u32::from(u16::from_be_bytes([bytes[at + 5], bytes[at + 6]]));
            let width = u32::from(u16::from_be_bytes([bytes[at + 7], bytes[at + 8]]));
            ensure!(width > 0 && height > 0, "JPEG is empty");
            return Ok((width, height, marker == 0xc2));
        }
        ensure!(
            !matches!(marker, 0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf),
            "lossless and arithmetic-coded JPEG are not read"
        );
        at += 2 + length;
    }
    bail!("JPEG has no frame header")
}

/// A mesh's vertex and triangle counts, as the renderer parses it.
pub fn inspect_mesh(bytes: &[u8]) -> Result<(usize, usize)> {
    let mesh = parse_mesh(bytes)?;
    Ok((mesh.vertices.len() / crate::asset_store::MESH_VERTEX_FLOATS, mesh.indices.len() / 3))
}

/// What a material names: its shader, then its textures.
pub fn inspect_material(bytes: &[u8]) -> Result<(String, Vec<String>)> {
    let material = parse_material(bytes)?;
    let textures = [material.texture, material.normal_texture, material.layer_texture].into_iter().flatten().collect();
    Ok((material.shader, textures))
}

/// Whether a shader draws sprites (its vertex stage takes only builtins) or
/// meshes; None when it has no `vertex` entry point.
pub fn shader_draws_sprites(code: &str) -> Option<bool> {
    crate::shader::draws_sprites(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn textures_are_read_from_their_headers() {
        let mut png = Vec::new();
        png.extend(b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR");
        png.extend(300u32.to_be_bytes());
        png.extend(200u32.to_be_bytes());
        png.extend([8, 6, 0, 0, 0]);
        let info = inspect_texture(&png).unwrap();
        assert_eq!((info.width, info.height, info.levels, info.sampled_levels), (300, 200, 1, 9));
        assert_eq!(info.format, "png RGBA 8-bit");
        // A baseline JPEG's frame header after an APP0 segment.
        let jpeg = [0xff, 0xd8, 0xff, 0xe0, 0, 4, 0, 0, 0xff, 0xc0, 0, 11, 8, 0, 64, 0, 128, 3, 0, 0, 0];
        let info = inspect_texture(&jpeg).unwrap();
        assert_eq!((info.width, info.height, info.format.as_str()), (128, 64, "jpeg"));
        let mut ktx = KTX2.to_vec();
        ktx.resize(80, 0);
        ktx[12..16].copy_from_slice(&145u32.to_le_bytes());
        ktx[20..24].copy_from_slice(&256u32.to_le_bytes());
        ktx[24..28].copy_from_slice(&256u32.to_le_bytes());
        ktx[40..44].copy_from_slice(&1u32.to_le_bytes());
        let info = inspect_texture(&ktx).unwrap();
        assert_eq!((info.format.as_str(), info.levels, info.sampled_levels), ("ktx2 bc7", 1, 1));
        assert!(inspect_texture(b"GIF89a").is_err());
        assert!(inspect_texture(&png[..20]).is_err());
    }
}
