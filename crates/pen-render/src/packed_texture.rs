// Packed textures, as models/gtex.py writes them: compact channels (palette
// indices and data stay one or two bytes a texel instead of being expanded to
// RGBA) and every mip level worked out offline, smallest first, so a texture
// streams in: a whole small picture at once, sharper with every level after.
//
// Header, little-endian: "GTEX", version 2, channels (1, 2 or 4), kind (see
// `KINDS`), level count, width and height (u32s). Then each level from 1 x 1
// up to the full size, as WebGPU sizes them: its deflated byte count (u32),
// then its deflated pixels. "exact" (colour) and "raw" (linear data) have
// the full size alone: neither may be averaged.
use std::io::Read;

use anyhow::{Result, bail, ensure};

const KINDS: [&str; 6] = ["exact", "color", "cutout", "data", "normals", "raw"];

#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedHeader {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
    /// Its mip levels: 1, or all of them down to 1 x 1.
    pub levels: u32,
}

/// One mip level; level 0 is the full size.
pub(crate) struct PackedLevel {
    pub level: u32,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Levels in a full mip chain for a `width` x `height` image.
pub(crate) fn mip_levels(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

pub(crate) fn level_size(size: u32, level: u32) -> u32 {
    (size >> level).max(1)
}

pub(crate) fn is_packed_texture(bytes: &[u8]) -> bool {
    bytes.starts_with(b"GTEX")
}

/// Bytes a texel of a packed texture's format takes.
pub(crate) fn texel_bytes(format: wgpu::TextureFormat) -> u32 {
    match format {
        wgpu::TextureFormat::R8Unorm => 1,
        wgpu::TextureFormat::Rg8Unorm => 2,
        _ => 4,
    }
}

fn parse_header(bytes: &[u8]) -> Result<PackedHeader> {
    ensure!(is_packed_texture(bytes), "packed texture requires a GTEX header");
    ensure!(bytes[4] == 2, "unsupported packed texture version");
    let channels = bytes[5];
    let Some(&kind) = KINDS.get(bytes[6] as usize) else {
        bail!("unsupported packed texture format");
    };
    ensure!([1, 2, 4].contains(&channels), "unsupported packed texture format");
    let width = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let height = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    ensure!(
        (1..=16384).contains(&width) && (1..=16384).contains(&height),
        "packed texture dimensions must be 1..16384"
    );
    let levels = u32::from(bytes[7]);
    let expected = if kind == "exact" || kind == "raw" { 1 } else { mip_levels(width, height) };
    ensure!(levels == expected, "packed texture level count does not match its kind");
    ensure!(!((kind == "color" || kind == "cutout") && channels != 4), "packed color texture needs four channels");
    let format = match channels {
        1 => wgpu::TextureFormat::R8Unorm,
        2 => wgpu::TextureFormat::Rg8Unorm,
        _ if matches!(kind, "data" | "normals" | "raw") => wgpu::TextureFormat::Rgba8Unorm,
        _ => wgpu::TextureFormat::Rgba8UnormSrgb,
    };
    Ok(PackedHeader { width, height, format, levels })
}

/// A packed texture being read from its bytes: its header, then its levels,
/// smallest first, one inflated per `next_level`.
pub(crate) struct PackedReader {
    bytes: Vec<u8>,
    at: usize,
    pub header: PackedHeader,
    /// The next level to read; it counts down to 0, then past it.
    next: i64,
}

impl PackedReader {
    pub fn open(bytes: Vec<u8>) -> Result<Self> {
        ensure!(bytes.len() >= 16, "packed texture requires a GTEX header");
        let header = parse_header(&bytes[..16])?;
        Ok(Self { bytes, at: 16, header, next: i64::from(header.levels) - 1 })
    }

    fn read(&mut self, count: usize) -> Result<&[u8]> {
        ensure!(self.bytes.len() - self.at >= count, "packed texture ends early");
        let start = self.at;
        self.at += count;
        Ok(&self.bytes[start..self.at])
    }

    /// The next level, or None once every level is read.
    pub fn next_level(&mut self) -> Result<Option<PackedLevel>> {
        let Ok(level) = u32::try_from(self.next) else {
            ensure!(self.at == self.bytes.len(), "packed texture runs past its last level");
            return Ok(None);
        };
        self.next -= 1;
        let width = level_size(self.header.width, level);
        let height = level_size(self.header.height, level);
        let length = (width * height * texel_bytes(self.header.format)) as usize;
        let deflated = u32::from_le_bytes(self.read(4)?.try_into().unwrap()) as usize;
        // Deflate never grows data by more than a little.
        ensure!(deflated <= length + 1024 + (length >> 6), "packed texture level is too long");
        let pixels = inflate(self.read(deflated)?, length)?;
        Ok(Some(PackedLevel { level, width, height, pixels }))
    }
}

fn inflate(deflated: &[u8], length: usize) -> Result<Vec<u8>> {
    let mut decoder = flate2::read::ZlibDecoder::new(deflated);
    let mut pixels = vec![0; length];
    let mut size = 0;
    while size < length {
        let read = decoder.read(&mut pixels[size..])?;
        if read == 0 {
            break;
        }
        size += read;
    }
    ensure!(size == length, "packed texture level does not match its size");
    let mut extra = [0u8; 1];
    ensure!(decoder.read(&mut extra)? == 0, "packed texture expands beyond its dimensions");
    Ok(pixels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn deflate(bytes: &[u8]) -> Vec<u8> {
        let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(bytes).unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn reads_levels_smallest_first() {
        let mut bytes = b"GTEX".to_vec();
        bytes.extend([2, 1, 3, 2]);
        bytes.extend(2u32.to_le_bytes());
        bytes.extend(1u32.to_le_bytes());
        for level in [vec![7u8], vec![1, 2]] {
            let packed = deflate(&level);
            bytes.extend(u32::try_from(packed.len()).unwrap().to_le_bytes());
            bytes.extend(packed);
        }
        let mut reader = PackedReader::open(bytes.clone()).unwrap();
        assert_eq!(reader.header.format, wgpu::TextureFormat::R8Unorm);
        let first = reader.next_level().unwrap().unwrap();
        assert_eq!((first.level, first.width, first.pixels.clone()), (1, 1, vec![7]));
        let second = reader.next_level().unwrap().unwrap();
        assert_eq!((second.level, second.width, second.pixels), (0, 2, vec![1, 2]));
        assert!(reader.next_level().unwrap().is_none());
        bytes.push(0);
        let mut reader = PackedReader::open(bytes).unwrap();
        reader.next_level().unwrap();
        reader.next_level().unwrap();
        assert!(reader.next_level().is_err());
    }

    #[test]
    fn mip_chain_lengths() {
        assert_eq!(mip_levels(1, 1), 1);
        assert_eq!(mip_levels(1024, 1024), 11);
        assert_eq!(mip_levels(512, 1152), 11);
        assert_eq!(level_size(5, 1), 2);
        assert_eq!(level_size(1, 4), 1);
    }
}
