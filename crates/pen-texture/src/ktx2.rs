// KTX2 files (Khronos Texture 2.0): a texture's format by its Vulkan number,
// its size and mip levels, a data format descriptor, and each level's texels,
// smallest level first in the file so a texture can stream in. pen writes
// every level deflated on its own (supercompression scheme 3, ZLIB) and reads
// that or uncompressed levels.
//
// A texture ships as one file per format family (`name.bc7.ktx2`,
// `name.astc.ktx2`, `name.etc2.ktx2`, `name.rgba.ktx2`) beside the key a
// material names (`name.ktx2`); the renderer reads the family its device
// samples directly (`variant_key`).
use std::io::{Read, Write};

use anyhow::{Context, Result, bail, ensure};

const IDENTIFIER: [u8; 12] = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];

// The header and the index after it, then 24 bytes for each level.
const HEADER_BYTES: usize = 80;
const LEVEL_INDEX_BYTES: usize = 24;

const SCHEME_NONE: u32 = 0;
const SCHEME_ZLIB: u32 = 3;

/// The widest or tallest texture read.
pub const MAX_SIZE: u32 = 16384;

/// The texel formats pen reads and writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    Rgba8Unorm,
    Rgba8Srgb,
    Bc7Unorm,
    Bc7Srgb,
    Etc2Rgba8Unorm,
    Etc2Rgba8Srgb,
    Astc4x4Unorm,
    Astc4x4Srgb,
}

const FORMATS: [(Format, u32); 8] = [
    (Format::Rgba8Unorm, 37),
    (Format::Rgba8Srgb, 43),
    (Format::Bc7Unorm, 145),
    (Format::Bc7Srgb, 146),
    (Format::Etc2Rgba8Unorm, 151),
    (Format::Etc2Rgba8Srgb, 152),
    (Format::Astc4x4Unorm, 157),
    (Format::Astc4x4Srgb, 158),
];

impl Format {
    /// Its `VkFormat`.
    pub fn vk(self) -> u32 {
        FORMATS.iter().find(|(format, _)| *format == self).map(|&(_, vk)| vk).unwrap()
    }

    pub fn from_vk(vk: u32) -> Option<Format> {
        FORMATS.iter().find(|&&(_, number)| number == vk).map(|&(format, _)| format)
    }

    pub fn family(self) -> Family {
        match self {
            Format::Rgba8Unorm | Format::Rgba8Srgb => Family::Rgba,
            Format::Bc7Unorm | Format::Bc7Srgb => Family::Bc7,
            Format::Etc2Rgba8Unorm | Format::Etc2Rgba8Srgb => Family::Etc2,
            Format::Astc4x4Unorm | Format::Astc4x4Srgb => Family::Astc,
        }
    }

    pub fn is_srgb(self) -> bool {
        matches!(self, Format::Rgba8Srgb | Format::Bc7Srgb | Format::Etc2Rgba8Srgb | Format::Astc4x4Srgb)
    }

    /// Texels across and down a block: 1 x 1 uncompressed, 4 x 4 otherwise.
    pub fn block(self) -> u32 {
        if self.family() == Family::Rgba { 1 } else { 4 }
    }

    /// Bytes a block takes.
    pub fn block_bytes(self) -> u32 {
        if self.family() == Family::Rgba { 4 } else { 16 }
    }

    /// Bytes a level of this size takes: whole blocks, as many as cover it.
    pub fn level_bytes(self, width: u32, height: u32) -> usize {
        let block = self.block();
        let blocks = u64::from(width.div_ceil(block)) * u64::from(height.div_ceil(block));
        usize::try_from(blocks * u64::from(self.block_bytes())).unwrap_or(usize::MAX)
    }
}

/// A family of formats some GPUs sample directly: BC7 (desktop GPUs), ASTC
/// (Apple and most mobile GPUs), ETC2 (OpenGL ES 3 class mobile GPUs), and
/// plain RGBA (any GPU), in the order a renderer prefers them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    Bc7,
    Astc,
    Etc2,
    Rgba,
}

impl Family {
    pub const ALL: [Family; 4] = [Family::Bc7, Family::Astc, Family::Etc2, Family::Rgba];

    /// Its name in a variant's file name.
    pub fn name(self) -> &'static str {
        match self {
            Family::Bc7 => "bc7",
            Family::Astc => "astc",
            Family::Etc2 => "etc2",
            Family::Rgba => "rgba",
        }
    }

    /// Its format for colour (sRGB) or for data read as it is.
    pub fn format(self, srgb: bool) -> Format {
        match (self, srgb) {
            (Family::Bc7, false) => Format::Bc7Unorm,
            (Family::Bc7, true) => Format::Bc7Srgb,
            (Family::Astc, false) => Format::Astc4x4Unorm,
            (Family::Astc, true) => Format::Astc4x4Srgb,
            (Family::Etc2, false) => Format::Etc2Rgba8Unorm,
            (Family::Etc2, true) => Format::Etc2Rgba8Srgb,
            (Family::Rgba, false) => Format::Rgba8Unorm,
            (Family::Rgba, true) => Format::Rgba8Srgb,
        }
    }
}

/// The file holding `key`'s texture in `family`: `name.bc7.ktx2` for
/// `name.ktx2`. None for a key that is not a KTX2 file's, or is a variant's.
pub fn variant_key(key: &str, family: Family) -> Option<String> {
    let name = key.strip_suffix(".ktx2")?;
    (variant_of(key).is_none()).then(|| format!("{name}.{}.ktx2", family.name()))
}

/// The key a variant's file stands for: `name.ktx2` for `name.bc7.ktx2`.
pub fn variant_of(key: &str) -> Option<String> {
    let name = key.strip_suffix(".ktx2")?;
    Family::ALL.iter().find_map(|family| {
        let base = name.strip_suffix(family.name())?.strip_suffix('.')?;
        (!base.is_empty() && !base.ends_with('/')).then(|| format!("{base}.ktx2"))
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub format: Format,
    pub width: u32,
    pub height: u32,
    /// Its mip levels, from the full size down; 1 for none.
    pub levels: u32,
}

/// One mip level's texels (whole blocks); level 0 is the full size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Level {
    pub level: u32,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// Levels in a full mip chain for a `width` x `height` image.
pub fn mip_levels(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

pub fn level_size(size: u32, level: u32) -> u32 {
    (size >> level).max(1)
}

pub fn is_ktx2(bytes: &[u8]) -> bool {
    bytes.starts_with(&IDENTIFIER)
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

/// A KTX2 file being read: its header, then its levels, smallest first, one
/// inflated per `next_level`.
pub struct Reader {
    bytes: Vec<u8>,
    header: Header,
    scheme: u32,
    /// Each level's offset and length in the file, level 0 first.
    index: Vec<(usize, usize)>,
    /// The next level to read; it counts down to 0, then past it.
    next: i64,
}

impl Reader {
    pub fn open(bytes: Vec<u8>) -> Result<Self> {
        ensure!(bytes.len() >= HEADER_BYTES && is_ktx2(&bytes), "texture requires a KTX2 header");
        let vk = u32_at(&bytes, 12);
        let Some(format) = Format::from_vk(vk) else { bail!("unsupported KTX2 format (VkFormat {vk})") };
        let (type_size, width, height) = (u32_at(&bytes, 16), u32_at(&bytes, 20), u32_at(&bytes, 24));
        let (depth, layers, faces) = (u32_at(&bytes, 28), u32_at(&bytes, 32), u32_at(&bytes, 36));
        let (levels, scheme) = (u32_at(&bytes, 40), u32_at(&bytes, 44));
        ensure!(type_size == 1, "KTX2 type size must be 1 for its format");
        ensure!(
            (1..=MAX_SIZE).contains(&width) && (1..=MAX_SIZE).contains(&height),
            "KTX2 texture dimensions must be 1..{MAX_SIZE}"
        );
        ensure!(depth == 0 && layers <= 1 && faces == 1, "KTX2 texture must be a single 2D image");
        ensure!((1..=mip_levels(width, height)).contains(&levels), "KTX2 level count must be 1 to a full mip chain");
        ensure!(scheme == SCHEME_NONE || scheme == SCHEME_ZLIB, "unsupported KTX2 supercompression {scheme}");
        let count = usize::try_from(levels)?;
        ensure!(bytes.len() >= HEADER_BYTES + count * LEVEL_INDEX_BYTES, "KTX2 level index ends early");
        let mut index = Vec::with_capacity(count);
        for level in 0..levels {
            let at = HEADER_BYTES + usize::try_from(level)? * LEVEL_INDEX_BYTES;
            let (offset, length, uncompressed) = (u64_at(&bytes, at), u64_at(&bytes, at + 8), u64_at(&bytes, at + 16));
            let expected = format.level_bytes(level_size(width, level), level_size(height, level));
            ensure!(usize::try_from(uncompressed).ok() == Some(expected), "KTX2 level {level} has the wrong size");
            ensure!(scheme != SCHEME_NONE || length == uncompressed, "KTX2 level {level} has the wrong length");
            let end = offset.checked_add(length).context("KTX2 level runs past the file")?;
            ensure!(end <= bytes.len() as u64, "KTX2 level {level} runs past the file");
            index.push((usize::try_from(offset)?, usize::try_from(length)?));
        }
        let header = Header { format, width, height, levels };
        Ok(Self { bytes, header, scheme, index, next: i64::from(levels) - 1 })
    }

    pub fn header(&self) -> Header {
        self.header
    }

    /// The next level, smallest first, or None once every level is read.
    pub fn next_level(&mut self) -> Result<Option<Level>> {
        let Ok(level) = u32::try_from(self.next) else { return Ok(None) };
        self.next -= 1;
        let Header { format, width, height, .. } = self.header;
        let (width, height) = (level_size(width, level), level_size(height, level));
        let (offset, length) = self.index[usize::try_from(level)?];
        let stored = &self.bytes[offset..offset + length];
        let expected = format.level_bytes(width, height);
        let data = if self.scheme == SCHEME_ZLIB { inflate(stored, expected)? } else { stored.to_vec() };
        Ok(Some(Level { level, width, height, data }))
    }
}

fn inflate(deflated: &[u8], length: usize) -> Result<Vec<u8>> {
    // Deflate never grows data by more than a little.
    ensure!(deflated.len() <= length + 1024 + (length >> 6), "KTX2 level is too long");
    let mut decoder = flate2::read::ZlibDecoder::new(deflated);
    let mut data = vec![0; length];
    let mut size = 0;
    while size < length {
        let read = decoder.read(&mut data[size..])?;
        if read == 0 {
            break;
        }
        size += read;
    }
    ensure!(size == length, "KTX2 level does not match its size");
    let mut extra = [0u8; 1];
    ensure!(decoder.read(&mut extra)? == 0, "KTX2 level expands beyond its size");
    Ok(data)
}

/// The data format descriptor of `format`: one basic block, with a sample
/// for each channel (or for each block's colour and alpha halves).
fn descriptor(format: Format) -> Vec<u8> {
    const LINEAR: u8 = 0x10;
    let srgb = format.is_srgb();
    // An sRGB texture's alpha is still linear.
    let alpha_linear = if srgb { LINEAR } else { 0 };
    // Colour model, then each sample's bit offset, bit count, channel and
    // upper value.
    let (model, samples): (u8, Vec<(u16, u8, u8, u32)>) = match format.family() {
        Family::Rgba => (1, vec![(0, 8, 0, 255), (8, 8, 1, 255), (16, 8, 2, 255), (24, 8, 15 | alpha_linear, 255)]),
        Family::Bc7 => (134, vec![(0, 128, 0, u32::MAX)]),
        Family::Etc2 => (161, vec![(0, 64, 15 | alpha_linear, u32::MAX), (64, 64, 2, u32::MAX)]),
        Family::Astc => (162, vec![(0, 128, 0, u32::MAX)]),
    };
    let block = u8::try_from(format.block() - 1).unwrap();
    let block_size = 24 + 16 * samples.len();
    let mut out = Vec::with_capacity(4 + block_size);
    out.extend(u32::try_from(4 + block_size).unwrap().to_le_bytes());
    // Khronos, basic descriptor block, version 2.
    out.extend(0u32.to_le_bytes());
    out.extend(2u16.to_le_bytes());
    out.extend(u16::try_from(block_size).unwrap().to_le_bytes());
    // BT.709 primaries; sRGB or linear transfer; straight alpha.
    out.extend([model, 1, if srgb { 2 } else { 1 }, 0]);
    out.extend([block, block, 0, 0]);
    out.extend([u8::try_from(format.block_bytes()).unwrap(), 0, 0, 0, 0, 0, 0, 0]);
    for (offset, bits, channel, upper) in samples {
        out.extend(offset.to_le_bytes());
        out.extend([bits - 1, channel, 0, 0, 0, 0]);
        out.extend(0u32.to_le_bytes());
        out.extend(upper.to_le_bytes());
    }
    out
}

/// Key/value data: who wrote the file.
fn key_values() -> Vec<u8> {
    let entry = b"KTXwriter\0pen-texture\0";
    let mut out = u32::try_from(entry.len()).unwrap().to_le_bytes().to_vec();
    out.extend(entry);
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
    out
}

fn deflate(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
    encoder.write_all(bytes)?;
    Ok(encoder.finish()?)
}

/// A KTX2 file of `levels` (level 0, the full size, first; each whole blocks
/// of `format`), each deflated on its own and stored smallest first.
pub fn write(format: Format, width: u32, height: u32, levels: &[Vec<u8>]) -> Result<Vec<u8>> {
    ensure!(
        (1..=MAX_SIZE).contains(&width) && (1..=MAX_SIZE).contains(&height),
        "KTX2 texture dimensions must be 1..{MAX_SIZE}"
    );
    let count = u32::try_from(levels.len())?;
    ensure!((1..=mip_levels(width, height)).contains(&count), "KTX2 level count must be 1 to a full mip chain");
    for (level, data) in (0..).zip(levels) {
        let expected = format.level_bytes(level_size(width, level), level_size(height, level));
        ensure!(data.len() == expected, "level {level} holds {} bytes, not {expected}", data.len());
    }
    let deflated = levels.iter().map(|level| deflate(level)).collect::<Result<Vec<_>>>()?;
    let dfd = descriptor(format);
    let kvd = key_values();
    let dfd_offset = HEADER_BYTES + levels.len() * LEVEL_INDEX_BYTES;
    let kvd_offset = dfd_offset + dfd.len();
    let data_offset = kvd_offset + kvd.len();
    let mut out = Vec::with_capacity(data_offset + deflated.iter().map(Vec::len).sum::<usize>());
    out.extend(IDENTIFIER);
    for value in [format.vk(), 1, width, height, 0, 0, 1, count, SCHEME_ZLIB] {
        out.extend(value.to_le_bytes());
    }
    for value in [dfd_offset, dfd.len(), kvd_offset, kvd.len()] {
        out.extend(u32::try_from(value)?.to_le_bytes());
    }
    // No supercompression global data.
    out.extend([0u8; 16]);
    // Levels are stored smallest first.
    let mut offsets = vec![0; levels.len()];
    let mut at = data_offset;
    for level in (0..levels.len()).rev() {
        offsets[level] = at;
        at += deflated[level].len();
    }
    for (level, data) in levels.iter().enumerate() {
        for value in [offsets[level], deflated[level].len(), data.len()] {
            out.extend(u64::try_from(value)?.to_le_bytes());
        }
    }
    out.extend(dfd);
    out.extend(kvd);
    for level in deflated.iter().rev() {
        out.extend(level);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_and_reads_levels_smallest_first() {
        let levels = vec![(0..64u8).collect::<Vec<_>>(), (0..16u8).collect(), vec![9, 8, 7, 6]];
        let bytes = write(Format::Rgba8Srgb, 4, 4, &levels).unwrap();
        assert!(is_ktx2(&bytes));
        let mut reader = Reader::open(bytes.clone()).unwrap();
        assert_eq!(reader.header(), Header { format: Format::Rgba8Srgb, width: 4, height: 4, levels: 3 });
        let mut read = Vec::new();
        while let Some(level) = reader.next_level().unwrap() {
            read.push((level.level, level.width, level.data));
        }
        assert_eq!(read, vec![(2, 1, levels[2].clone()), (1, 2, levels[1].clone()), (0, 4, levels[0].clone())]);
        // The smallest level's data comes first in the file.
        let first = usize::try_from(u64_at(&bytes, HEADER_BYTES + 2 * LEVEL_INDEX_BYTES)).unwrap();
        let second = usize::try_from(u64_at(&bytes, HEADER_BYTES + LEVEL_INDEX_BYTES)).unwrap();
        assert!(first < second);
    }

    #[test]
    fn block_levels_cover_whole_blocks() {
        assert_eq!(Format::Bc7Srgb.level_bytes(8, 4), 32);
        assert_eq!(Format::Bc7Srgb.level_bytes(2, 1), 16);
        assert_eq!(Format::Rgba8Unorm.level_bytes(3, 2), 24);
        let levels: Vec<Vec<u8>> = (0..4)
            .map(|level| vec![7; Format::Astc4x4Unorm.level_bytes(level_size(8, level), level_size(4, level))])
            .collect();
        let bytes = write(Format::Astc4x4Unorm, 8, 4, &levels).unwrap();
        let mut reader = Reader::open(bytes).unwrap();
        assert_eq!(reader.header().levels, 4);
        let smallest = reader.next_level().unwrap().unwrap();
        assert_eq!((smallest.level, smallest.width, smallest.height, smallest.data.len()), (3, 1, 1, 16));
    }

    #[test]
    fn descriptor_describes_the_format() {
        let bc7 = descriptor(Format::Bc7Srgb);
        assert_eq!(u32_at(&bc7, 0), 44);
        // Model BC7, BT.709, sRGB; 4 x 4 blocks of 16 bytes.
        assert_eq!(&bc7[12..16], &[134, 1, 2, 0]);
        assert_eq!(&bc7[16..21], &[3, 3, 0, 0, 16]);
        let rgba = descriptor(Format::Rgba8Unorm);
        assert_eq!(u32_at(&rgba, 0), 4 + 24 + 64);
        assert_eq!(&rgba[12..16], &[1, 1, 1, 0]);
    }

    #[test]
    fn refuses_what_it_cannot_read() {
        let good = write(Format::Bc7Unorm, 4, 4, &[vec![0; 16]]).unwrap();
        assert!(Reader::open(good.clone()).is_ok());
        let mut format = good.clone();
        format[12..16].copy_from_slice(&999u32.to_le_bytes());
        assert!(Reader::open(format).is_err());
        let mut cut = good.clone();
        cut.truncate(cut.len() - 1);
        assert!(Reader::open(cut).is_err());
        let mut size = good.clone();
        // Level 0's uncompressed length.
        size[HEADER_BYTES + 16..HEADER_BYTES + 24].copy_from_slice(&15u64.to_le_bytes());
        assert!(Reader::open(size).is_err());
        assert!(Reader::open(b"GTEX".to_vec()).is_err());
        assert!(write(Format::Bc7Unorm, 4, 4, &[vec![0; 15]]).is_err());
    }

    #[test]
    fn variants_name_their_family() {
        assert_eq!(variant_key("textures/crater.ktx2", Family::Bc7).as_deref(), Some("textures/crater.bc7.ktx2"));
        assert_eq!(variant_key("crater.png", Family::Bc7), None);
        assert_eq!(variant_key("crater.bc7.ktx2", Family::Astc), None);
        assert_eq!(variant_of("textures/crater.etc2.ktx2").as_deref(), Some("textures/crater.ktx2"));
        assert_eq!(variant_of("crater.ktx2"), None);
        assert_eq!(variant_of("rgba.ktx2"), None);
        assert_eq!(variant_of("dir/.rgba.ktx2"), None);
    }
}
