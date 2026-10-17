// Packed textures (pen-render's packed_texture.rs reads them; gdev's
// models/gtex.py writes the others): "GTEX", version 2, channels, kind, level
// count, width and height (u32s), then each level's deflated byte count
// (u32) and its deflated pixels, all little-endian.
use std::io::Write;

use anyhow::Result;

// packed_texture.rs's KINDS: "raw" is exact linear data, one level.
const RAW: u8 = 5;

/// Four channels a texel of exact linear data, one level.
pub(crate) fn raw(width: u32, height: u32, pixels: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
    encoder.write_all(pixels)?;
    let deflated = encoder.finish()?;
    let mut bytes = b"GTEX".to_vec();
    bytes.extend([2, 4, RAW, 1]);
    bytes.extend(width.to_le_bytes());
    bytes.extend(height.to_le_bytes());
    bytes.extend(u32::try_from(deflated.len())?.to_le_bytes());
    bytes.extend(deflated);
    Ok(bytes)
}
