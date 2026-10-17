// KTX2 textures (pen-texture's `ktx2`): which variant of a texture to read,
// by the block-compression families the device samples, and their levels,
// which stream in smallest first as packed textures' do.
use anyhow::{Result, ensure};
use pen_texture::ktx2::{self, Family, Format};

use crate::packed_texture::{PackedHeader, PackedLevel, PackedReader, is_packed_texture};

pub(crate) fn texture_format(format: Format) -> wgpu::TextureFormat {
    let astc = |channel| wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B4x4, channel };
    match format {
        Format::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        Format::Rgba8Srgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        Format::Bc7Unorm => wgpu::TextureFormat::Bc7RgbaUnorm,
        Format::Bc7Srgb => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
        Format::Etc2Rgba8Unorm => wgpu::TextureFormat::Etc2Rgba8Unorm,
        Format::Etc2Rgba8Srgb => wgpu::TextureFormat::Etc2Rgba8UnormSrgb,
        Format::Astc4x4Unorm => astc(wgpu::AstcChannel::Unorm),
        Format::Astc4x4Srgb => astc(wgpu::AstcChannel::UnormSrgb),
    }
}

/// The device feature a family's formats need; none for plain RGBA.
fn feature(family: Family) -> wgpu::Features {
    match family {
        Family::Bc7 => wgpu::Features::TEXTURE_COMPRESSION_BC,
        Family::Astc => wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        Family::Etc2 => wgpu::Features::TEXTURE_COMPRESSION_ETC2,
        Family::Rgba => wgpu::Features::empty(),
    }
}

/// The families a device with `features` samples, best first: BC7, ASTC,
/// ETC2, then plain RGBA, which every device does.
pub(crate) fn families(features: wgpu::Features) -> impl Iterator<Item = Family> {
    Family::ALL.into_iter().filter(move |&family| features.contains(feature(family)))
}

/// The files to read for texture `key`, in order until one is there: for a
/// KTX2 key, its variants the device samples, then the key's own file.
pub(crate) fn candidates(key: &str, features: wgpu::Features) -> Vec<String> {
    let mut keys: Vec<String> = families(features).filter_map(|family| ktx2::variant_key(key, family)).collect();
    keys.push(key.to_owned());
    keys
}

/// Whether a texture file has levels of its own, which stream in.
pub(crate) fn has_levels(bytes: &[u8]) -> bool {
    is_packed_texture(bytes) || ktx2::is_ktx2(bytes)
}

/// A texture's levels, smallest first: a packed texture's, or a KTX2 file's.
pub(crate) enum LevelReader {
    Packed(PackedReader),
    Ktx2(ktx2::Reader),
}

impl LevelReader {
    /// For a device with `features`, which must sample the file's format.
    pub fn open(bytes: Vec<u8>, features: wgpu::Features) -> Result<Self> {
        if !ktx2::is_ktx2(&bytes) {
            return Ok(LevelReader::Packed(PackedReader::open(bytes)?));
        }
        let reader = ktx2::Reader::open(bytes)?;
        let format = reader.header().format;
        ensure!(features.contains(feature(format.family())), "the device cannot sample {format:?} textures");
        Ok(LevelReader::Ktx2(reader))
    }

    pub fn header(&self) -> PackedHeader {
        match self {
            LevelReader::Packed(reader) => reader.header,
            LevelReader::Ktx2(reader) => {
                let header = reader.header();
                PackedHeader {
                    width: header.width,
                    height: header.height,
                    format: texture_format(header.format),
                    levels: header.levels,
                }
            }
        }
    }

    pub fn next_level(&mut self) -> Result<Option<PackedLevel>> {
        match self {
            LevelReader::Packed(reader) => reader.next_level(),
            LevelReader::Ktx2(reader) => Ok(reader.next_level()?.map(|level| PackedLevel {
                level: level.level,
                width: level.width,
                height: level.height,
                pixels: level.data,
            })),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_follow_the_device() {
        let all = wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::TEXTURE_COMPRESSION_ASTC
            | wgpu::Features::TEXTURE_COMPRESSION_ETC2;
        assert_eq!(
            candidates("rock.ktx2", all),
            ["rock.bc7.ktx2", "rock.astc.ktx2", "rock.etc2.ktx2", "rock.rgba.ktx2", "rock.ktx2"]
        );
        // A phone's GPU, and one with none of them.
        assert_eq!(
            candidates("rock.ktx2", wgpu::Features::TEXTURE_COMPRESSION_ETC2),
            ["rock.etc2.ktx2", "rock.rgba.ktx2", "rock.ktx2"]
        );
        assert_eq!(candidates("rock.ktx2", wgpu::Features::empty()), ["rock.rgba.ktx2", "rock.ktx2"]);
        assert_eq!(candidates("rock.png", all), ["rock.png"]);
        assert_eq!(candidates("rock.bc7.ktx2", all), ["rock.bc7.ktx2"]);
    }

    #[test]
    fn reads_ktx2_levels_the_device_samples() {
        let levels: Vec<Vec<u8>> = vec![vec![1; 16], vec![2; 16], vec![3; 16]];
        let bytes = ktx2::write(Format::Bc7Srgb, 4, 4, &levels).unwrap();
        assert!(has_levels(&bytes));
        let error = LevelReader::open(bytes.clone(), wgpu::Features::empty()).err().unwrap();
        assert_eq!(error.to_string(), "the device cannot sample Bc7Srgb textures");
        let mut reader = LevelReader::open(bytes, wgpu::Features::TEXTURE_COMPRESSION_BC).unwrap();
        let header = reader.header();
        assert_eq!((header.format, header.levels), (wgpu::TextureFormat::Bc7RgbaUnormSrgb, 3));
        let first = reader.next_level().unwrap().unwrap();
        assert_eq!((first.level, first.width, first.pixels), (2, 1, vec![3; 16]));
    }
}
