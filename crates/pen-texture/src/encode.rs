// The asset pipeline's texture encoder: an RGBA8 image to the KTX2 file of
// one format family, its mip chain filtered (`mips`) and each level
// block-encoded (`bc7`, `astc`, `etc2`) or kept as plain RGBA. A big level
// encodes on a few threads, a band of block rows each, into the very bytes
// one thread would write.
//
// The helpers the block encoders share live here too: a block's texels with
// the image's edge repeated, a bit writer, and fitting two endpoints to some
// texels (their principal axis, then least squares given the step each
// texel picks, then each texel's nearest step between the endpoints).
use std::num::NonZero;
use std::ops::Range;
use std::thread;

use anyhow::{Result, ensure};

use crate::ktx2::{self, Family, Format};
use crate::{astc, bc7, etc2, mips};

/// How a texture's texels are read: colour (sRGB-encoded RGB, straight
/// alpha) or data read as it is (normals, masks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Color,
    Data,
}

/// A 4 x 4 block's RGBA8 texels, row-major.
pub(crate) type Block = [[u8; 4]; 16];

/// A block encoder: 16 texels to the 16 bytes of one block.
pub(crate) type BlockEncoder = fn(&Block) -> [u8; 16];

/// The most threads a level encodes on: the machine is shared.
const MAX_THREADS: usize = 4;

/// Blocks too few to be worth a thread of their own.
const BLOCKS_PER_THREAD: usize = 256;

/// Power iteration steps to a principal axis: plenty for 16 texels.
pub(crate) const POWER_STEPS: usize = 8;

/// One KTX2 file of `family` holding `rgba` (`width` x `height` RGBA8):
/// sRGB formats for colour, UNORM for data; every mip level down to 1 x 1
/// when `mipmaps`, else level 0 alone. A block family needs a width and
/// height that are multiples of 4 (WebGPU's rule for compressed textures);
/// an image that is not is written in the family's file as plain RGBA, which
/// every GPU reads.
pub fn encode(rgba: &[u8], width: u32, height: u32, kind: Kind, family: Family, mipmaps: bool) -> Result<Vec<u8>> {
    let levels = levels(rgba, width, height, kind, mipmaps)?;
    write(&levels, width, height, kind, family, threads())
}

/// Supported families in `Family::ALL` order, encoded from one mip chain.
/// Dimensions outside whole blocks need only the universal RGBA fallback.
pub fn encode_all(rgba: &[u8], width: u32, height: u32, kind: Kind, mipmaps: bool) -> Result<Vec<(Family, Vec<u8>)>> {
    let levels = levels(rgba, width, height, kind, mipmaps)?;
    let threads = threads();
    let families: &[_] =
        if width.is_multiple_of(4) && height.is_multiple_of(4) { &Family::ALL } else { &[Family::Rgba] };
    families.iter().map(|&family| Ok((family, write(&levels, width, height, kind, family, threads)?))).collect()
}

/// The image's levels, the full size first: the image alone, or its whole
/// mip chain.
fn levels(rgba: &[u8], width: u32, height: u32, kind: Kind, mipmaps: bool) -> Result<Vec<Vec<u8>>> {
    ensure!(
        (1..=ktx2::MAX_SIZE).contains(&width) && (1..=ktx2::MAX_SIZE).contains(&height),
        "texture dimensions must be 1..{}",
        ktx2::MAX_SIZE
    );
    let expected = texel_count(width, height) * 4;
    ensure!(rgba.len() == expected, "a {width} x {height} RGBA8 image holds {expected} bytes, not {}", rgba.len());
    Ok(if mipmaps { mips::mip_chain(rgba, width, height, kind == Kind::Color) } else { vec![rgba.to_vec()] })
}

/// The KTX2 file of `levels` in `family`'s format, or plain RGBA when the
/// size is not whole blocks; each level encoded on up to `threads` threads.
fn write(levels: &[Vec<u8>], width: u32, height: u32, kind: Kind, family: Family, threads: usize) -> Result<Vec<u8>> {
    let srgb = kind == Kind::Color;
    let whole_blocks = width.is_multiple_of(4) && height.is_multiple_of(4);
    let format = if whole_blocks { family.format(srgb) } else { Family::Rgba.format(srgb) };
    let data: Vec<Vec<u8>> = (0..)
        .zip(levels)
        .map(|(level, texels)| {
            let (width, height) = (ktx2::level_size(width, level), ktx2::level_size(height, level));
            encode_level(format, texels, width, height, threads)
        })
        .collect();
    ktx2::write(format, width, height, &data)
}

/// One level's texels as `format` stores them.
fn encode_level(format: Format, rgba: &[u8], width: u32, height: u32, threads: usize) -> Vec<u8> {
    let encode_block: BlockEncoder = match format.family() {
        Family::Bc7 => bc7::encode_block,
        Family::Astc => astc::encode_block,
        Family::Etc2 => etc2::encode_block,
        Family::Rgba => return rgba.to_vec(),
    };
    encode_blocks(rgba, width, height, encode_block, threads)
}

fn threads() -> usize {
    thread::available_parallelism().map_or(1, NonZero::get).min(MAX_THREADS)
}

fn texel_count(width: u32, height: u32) -> usize {
    width as usize * height as usize
}

/// `rgba` (`width` x `height` RGBA8) as blocks, row-major, each from
/// `encode_block`, on up to `threads` threads, a band of block rows each.
/// The bytes are the same however many threads write them.
pub(crate) fn encode_blocks(
    rgba: &[u8],
    width: u32,
    height: u32,
    encode_block: BlockEncoder,
    threads: usize,
) -> Vec<u8> {
    assert_eq!(rgba.len(), texel_count(width, height) * 4, "a {width} x {height} RGBA8 image");
    let (columns, rows) = (width.div_ceil(4) as usize, height.div_ceil(4) as usize);
    let row_bytes = columns * 16;
    let mut out = vec![0; rows * row_bytes];
    let threads = threads.min(rows * columns / BLOCKS_PER_THREAD).clamp(1, rows.max(1));
    if threads == 1 {
        encode_rows(rgba, width, height, encode_block, 0, &mut out);
        return out;
    }
    let band = rows.div_ceil(threads);
    thread::scope(|scope| {
        for (index, chunk) in out.chunks_mut(band * row_bytes).enumerate() {
            scope.spawn(move || encode_rows(rgba, width, height, encode_block, index * band, chunk));
        }
    });
    out
}

/// Fills `out` with whole rows of blocks, from block row `first_row` on.
fn encode_rows(rgba: &[u8], width: u32, height: u32, encode_block: BlockEncoder, first_row: usize, out: &mut [u8]) {
    let columns = width.div_ceil(4) as usize;
    for (index, bytes) in out.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        *bytes = encode_block(&block(rgba, width, height, index % columns, first_row + index / columns));
    }
}

/// The block at block `column` and `row` of `rgba`; texels past the image's
/// right or bottom edge repeat its last column or row.
pub(crate) fn block(rgba: &[u8], width: u32, height: u32, column: usize, row: usize) -> Block {
    let (width, height) = (width as usize, height as usize);
    std::array::from_fn(|texel| {
        let x = (column * 4 + texel % 4).min(width - 1);
        let y = (row * 4 + texel / 4).min(height - 1);
        let at = (y * width + x) * 4;
        [rgba[at], rgba[at + 1], rgba[at + 2], rgba[at + 3]]
    })
}

/// `value` rounded to the nearest whole number in 0..=255.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "rounded and clamped to 0..=255 first")]
pub(crate) fn to_u8(value: f32) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

/// Bits written least significant first into a 128-bit block.
#[derive(Default)]
pub(crate) struct Bits {
    value: u128,
    len: u32,
}

impl Bits {
    /// Appends the low `count` bits of `value`, which holds no more.
    pub(crate) fn put(&mut self, value: impl Into<u32>, count: u32) {
        let value = value.into();
        debug_assert!(count < 32 && value >> count == 0 && self.len + count <= 128, "{value} in {count} bits");
        self.value |= u128::from(value) << self.len;
        self.len += count;
    }

    pub(crate) fn len(&self) -> u32 {
        self.len
    }

    pub(crate) fn value(&self) -> u128 {
        self.value
    }
}

/// 1 in each of `channels` and 0 in the rest, so loops run over all four
/// channels (which the compiler unrolls) and count only those.
fn mask(channels: &Range<usize>) -> [i32; 4] {
    std::array::from_fn(|c| i32::from(channels.contains(&c)))
}

fn dot(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// The largest eigenvalue of symmetric `matrix` and its unit eigenvector,
/// after `steps` of power iteration. It starts from the column of the
/// largest diagonal entry, which is never square to the principal axis.
pub(crate) fn principal_axis(matrix: &[[f32; 4]; 4], steps: usize) -> (f32, [f32; 4]) {
    let start = (0..4).fold(0, |best, c| if matrix[c][c] > matrix[best][best] { c } else { best });
    let mut axis = matrix[start];
    let mut length = dot(&axis, &axis).sqrt();
    if length <= f32::EPSILON {
        let mut unit = [0.0; 4];
        unit[start] = 1.0;
        return (0.0, unit);
    }
    for _ in 0..steps {
        let unit = axis.map(|value| value / length);
        let next = matrix.map(|row| dot(&row, &unit));
        length = dot(&next, &next).sqrt();
        if length <= f32::EPSILON {
            break;
        }
        axis = next;
    }
    let length = dot(&axis, &axis).sqrt();
    let axis = axis.map(|value| value / length);
    (dot(&axis, &matrix.map(|row| dot(&row, &axis))), axis)
}

/// The two ends of `texels`' spread along their principal axis in
/// `channels`: the line two endpoints fit first. Channels outside are 0.
pub(crate) fn line_endpoints(texels: &[[u8; 4]], channels: Range<usize>) -> [[f32; 4]; 2] {
    let mask = mask(&channels).map(|bit| bit as f32);
    let count = texels.len() as f32;
    let mut mean = [0.0; 4];
    for texel in texels {
        for c in 0..4 {
            mean[c] += f32::from(texel[c]) * mask[c];
        }
    }
    mean = mean.map(|sum| sum / count);
    let mut scatter = [[0.0; 4]; 4];
    for texel in texels {
        let offset: [f32; 4] = std::array::from_fn(|c| (f32::from(texel[c]) - mean[c]) * mask[c]);
        for c in 0..4 {
            for d in 0..4 {
                scatter[c][d] += offset[c] * offset[d];
            }
        }
    }
    let (_, axis) = principal_axis(&scatter, POWER_STEPS);
    let axis: [f32; 4] = std::array::from_fn(|c| axis[c] * mask[c]);
    let (mut low, mut high) = (f32::MAX, f32::MIN);
    for texel in texels {
        let along: f32 = (0..4).map(|c| (f32::from(texel[c]) - mean[c]) * axis[c]).sum();
        (low, high) = (low.min(along), high.max(along));
    }
    [low, high].map(|along| std::array::from_fn(|c| (mean[c] + along * axis[c]).clamp(0.0, 255.0) * mask[c]))
}

/// The endpoints that best fit `texels` in `channels` (least squares) given
/// how far each texel sits between them (`along`, 0 at the first, 1 at the
/// second). None when every texel sits at the same place, which pins no line.
pub(crate) fn least_squares(texels: &[[u8; 4]], along: &[f32], channels: Range<usize>) -> Option<[[f32; 4]; 2]> {
    let mask = mask(&channels).map(|bit| bit as f32);
    let (mut first, mut both, mut second) = (0.0, 0.0, 0.0);
    let (mut toward_first, mut toward_second) = ([0.0; 4], [0.0; 4]);
    for (texel, &t) in texels.iter().zip(along) {
        let s = 1.0 - t;
        (first, both, second) = (first + s * s, both + s * t, second + t * t);
        for c in 0..4 {
            toward_first[c] += s * f32::from(texel[c]);
            toward_second[c] += t * f32::from(texel[c]);
        }
    }
    let determinant = first * second - both * both;
    if determinant <= 1e-3 * first.max(second) {
        return None;
    }
    Some([
        std::array::from_fn(|c| {
            ((second * toward_first[c] - both * toward_second[c]) / determinant).clamp(0.0, 255.0) * mask[c]
        }),
        std::array::from_fn(|c| {
            ((first * toward_second[c] - both * toward_first[c]) / determinant).clamp(0.0, 255.0) * mask[c]
        }),
    ])
}

/// Each texel's nearest `palette` entry in `channels` (its index), and the
/// squared error they leave. The palette runs along a line from its first
/// entry to its last, so a texel's projection on that line guesses its
/// index and the entries either side check the guess.
pub(crate) fn nearest(texels: &[[u8; 4]], channels: Range<usize>, palette: &[[i32; 4]]) -> ([u8; 16], u32) {
    let mask = mask(&channels);
    let top = palette.len() - 1;
    let steps = i32::try_from(top).unwrap_or(i32::MAX);
    let first = palette[0];
    let direction: [i32; 4] = std::array::from_fn(|c| (palette[top][c] - first[c]) * mask[c]);
    let length: i32 = direction.iter().map(|d| d * d).sum();
    let distance = |texel: &[i32; 4], color: &[i32; 4]| -> u32 {
        (0..4).map(|c| ((texel[c] - color[c]) * mask[c]).unsigned_abs().pow(2)).sum()
    };
    let mut indices = [0; 16];
    let mut error = 0;
    for (index, texel) in indices.iter_mut().zip(texels) {
        let texel = texel.map(i32::from);
        let guess = if length == 0 {
            0
        } else {
            let along: i32 = (0..4).map(|c| (texel[c] - first[c]) * direction[c]).sum();
            usize::try_from((2 * along * steps + length).div_euclid(2 * length).clamp(0, steps)).unwrap_or(0)
        };
        let (mut best, mut least) = (guess, distance(&texel, &palette[guess]));
        for k in [guess.wrapping_sub(1), guess + 1] {
            if let Some(color) = palette.get(k) {
                let near = distance(&texel, color);
                if near < least {
                    (best, least) = (k, near);
                }
            }
        }
        *index = u8::try_from(best).unwrap_or(0);
        error += least;
    }
    (indices, error)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ktx2::Reader;

    /// A deterministic test image: smooth gradients, two hard-edged shapes,
    /// a noisy quarter and alpha that ramps, steps and cuts out.
    pub(crate) fn test_image(width: u32, height: u32) -> Vec<u8> {
        let mut seed = 0x2545_f491_u32;
        let mut noise = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            i32::try_from(seed >> 24).unwrap() % 17 - 8
        };
        let (w, h) = (i32::try_from(width).unwrap(), i32::try_from(height).unwrap());
        let mut out = Vec::with_capacity(texel_count(width, height) * 4);
        for y in 0..h {
            for x in 0..w {
                let bowl = ((x - w / 2).pow(2) + (y - h / 2).pow(2)) * 255 / (w * w / 4 + h * h / 4).max(1);
                let mut texel = [x * 255 / w.max(2), y * 255 / h.max(2), 255 - bowl, 255];
                if (x - w * 5 / 8).pow(2) + (y - h * 3 / 8).pow(2) < (w / 5).pow(2) {
                    texel[..3].copy_from_slice(&[220, 30, 60]);
                }
                if (w / 8..w * 3 / 8).contains(&x) && (h * 5 / 8..h * 7 / 8).contains(&y) {
                    texel[..3].copy_from_slice(&[20, 200, 90]);
                }
                if x >= w / 2 && y >= h / 2 {
                    for value in &mut texel[..3] {
                        *value += noise();
                    }
                }
                if x >= w / 2 {
                    texel[3] = 255 - y * 200 / h.max(1);
                }
                if (w * 3 / 4..w * 7 / 8).contains(&x) && (h / 8..h / 4).contains(&y) {
                    texel[3] = 0;
                }
                out.extend(texel.map(|value| u8::try_from(value.clamp(0, 255)).unwrap()));
            }
        }
        out
    }

    /// A 6 x 6 corner of the test image across a gradient, the disc's edge
    /// and alpha: an image of partial blocks.
    pub(crate) fn small_image() -> Vec<u8> {
        let image = test_image(64, 64);
        (20..26).flat_map(|y| image[(y * 64 + 26) * 4..(y * 64 + 32) * 4].to_vec()).collect()
    }

    /// texture2ddecoder's pixels (BGRA packed in a u32) as RGBA8.
    pub(crate) fn rgba_of(pixels: &[u32]) -> Vec<u8> {
        pixels
            .iter()
            .flat_map(|pixel| {
                let [b, g, r, a] = pixel.to_le_bytes();
                [r, g, b, a]
            })
            .collect()
    }

    /// Peak signal to noise ratio over every channel, in dB.
    pub(crate) fn psnr(original: &[u8], decoded: &[u8]) -> f64 {
        assert_eq!(original.len(), decoded.len());
        let error: f64 = original.iter().zip(decoded).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
        let mean = error / original.len() as f64;
        if mean == 0.0 { f64::INFINITY } else { 10.0 * (255.0f64.powi(2) / mean).log10() }
    }

    /// The largest difference in any channel.
    pub(crate) fn max_difference(original: &[u8], decoded: &[u8]) -> u8 {
        original.iter().zip(decoded).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0)
    }

    /// Colours to fill a block with, which a block encoder must give back
    /// exactly or within a level or two.
    pub(crate) const SOLID_COLORS: [[u8; 4]; 8] = [
        [0, 0, 0, 0],
        [255, 255, 255, 255],
        [0, 0, 0, 255],
        [17, 130, 201, 77],
        [1, 254, 128, 3],
        [255, 0, 127, 128],
        [93, 94, 95, 96],
        [200, 13, 77, 255],
    ];

    fn read(bytes: Vec<u8>) -> (ktx2::Header, Vec<ktx2::Level>) {
        let mut reader = Reader::open(bytes).unwrap();
        let mut levels = Vec::new();
        while let Some(level) = reader.next_level().unwrap() {
            levels.push(level);
        }
        levels.reverse();
        (reader.header(), levels)
    }

    #[test]
    fn writes_every_family_with_its_levels() {
        let image = test_image(20, 12);
        let files = encode_all(&image, 20, 12, Kind::Color, true).unwrap();
        assert_eq!(files.iter().map(|(family, _)| *family).collect::<Vec<_>>(), Family::ALL);
        for (family, bytes) in files {
            let (header, levels) = read(bytes);
            assert_eq!(header.format, family.format(true));
            assert_eq!((header.width, header.height, header.levels), (20, 12, 5));
            for (level, data) in (0..).zip(&levels) {
                let (width, height) = (ktx2::level_size(20, level), ktx2::level_size(12, level));
                assert_eq!((data.level, data.width, data.height), (level, width, height));
                assert_eq!(data.data.len(), header.format.level_bytes(width, height));
            }
            if family == Family::Rgba {
                assert_eq!(levels[0].data, image);
            }
        }
        let (header, levels) = read(encode(&image, 20, 12, Kind::Data, Family::Astc, false).unwrap());
        assert_eq!((header.format, header.levels, levels.len()), (Format::Astc4x4Unorm, 1, 1));
    }

    #[test]
    fn level_zero_decodes_close_to_the_image() {
        let image = test_image(32, 32);
        for family in [Family::Bc7, Family::Etc2, Family::Astc] {
            let (header, levels) = read(encode(&image, 32, 32, Kind::Color, family, false).unwrap());
            let mut pixels = vec![0; 32 * 32];
            match header.format {
                Format::Bc7Srgb => texture2ddecoder::decode_bc7(&levels[0].data, 32, 32, &mut pixels),
                Format::Etc2Rgba8Srgb => texture2ddecoder::decode_etc2_rgba8(&levels[0].data, 32, 32, &mut pixels),
                Format::Astc4x4Srgb => texture2ddecoder::decode_astc(&levels[0].data, 32, 32, 4, 4, &mut pixels),
                format => panic!("unexpected {format:?}"),
            }
            .unwrap();
            assert!(psnr(&image, &rgba_of(&pixels)) > 30.0, "{family:?}");
        }
    }

    #[test]
    fn sizes_not_whole_blocks_fall_back_to_rgba() {
        let image = test_image(6, 8);
        for family in Family::ALL {
            let (header, levels) = read(encode(&image, 6, 8, Kind::Data, family, true).unwrap());
            assert_eq!((header.format, header.levels), (Format::Rgba8Unorm, 4));
            assert_eq!(levels[0].data, image);
            assert_eq!(levels[1].data.len(), 3 * 4 * 4);
        }
        let files = encode_all(&image, 6, 8, Kind::Data, true).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].0, Family::Rgba);
        let (header, levels) = read(files.into_iter().next().unwrap().1);
        assert_eq!((header.format, header.levels), (Format::Rgba8Unorm, 4));
        assert_eq!(levels[0].data, image);
    }

    #[test]
    fn refuses_what_it_cannot_encode() {
        let image = test_image(4, 4);
        assert!(encode(&image, 4, 4, Kind::Color, Family::Bc7, true).is_ok());
        assert!(encode(&image[..60], 4, 4, Kind::Color, Family::Bc7, true).is_err());
        assert!(encode(&image, 8, 2, Kind::Color, Family::Bc7, true).is_ok());
        assert!(encode(&[], 0, 4, Kind::Color, Family::Bc7, true).is_err());
        assert!(encode(&image, 16, 1, Kind::Color, Family::Etc2, false).is_ok());
        let wide = ktx2::MAX_SIZE + 1;
        assert!(encode(&vec![0; wide as usize * 4], wide, 1, Kind::Data, Family::Rgba, false).is_err());
    }

    #[test]
    fn encodes_the_same_bytes_every_time_on_any_threads() {
        let image = test_image(96, 136);
        let levels = levels(&image, 96, 136, Kind::Color, true).unwrap();
        for family in Family::ALL {
            let serial = write(&levels, 96, 136, Kind::Color, family, 1).unwrap();
            let parallel = write(&levels, 96, 136, Kind::Color, family, 4).unwrap();
            assert_eq!(serial, parallel, "{family:?}");
            assert_eq!(serial, encode(&image, 96, 136, Kind::Color, family, true).unwrap(), "{family:?}");
        }
        let big = test_image(128, 128);
        assert_eq!(bc7::encode(&big, 128, 128), encode_blocks(&big, 128, 128, bc7::encode_block, 4));
        assert_eq!(etc2::encode(&big, 128, 128), encode_blocks(&big, 128, 128, etc2::encode_block, 3));
        assert_eq!(astc::encode(&big, 128, 128), encode_blocks(&big, 128, 128, astc::encode_block, 2));
    }

    #[test]
    fn edge_blocks_repeat_the_last_texels() {
        let image: Vec<u8> = (0..6 * 6 * 4).map(|value| u8::try_from(value).unwrap()).collect();
        let corner = block(&image, 6, 6, 1, 1);
        // Texel (4, 4) starts this block; (5, 5) is the image's last.
        assert_eq!(corner[0], [112, 113, 114, 115]);
        assert_eq!(corner[1], [116, 117, 118, 119]);
        assert_eq!(corner[2], corner[1]);
        assert_eq!(corner[15], [140, 141, 142, 143]);
        assert_eq!(corner[12], corner[4]);
    }

    #[test]
    fn fits_a_line_through_texels() {
        let texels: Vec<[u8; 4]> = (0..16u8).map(|i| [i * 10, 200 - i * 10, 50, 255]).collect();
        let [first, second] = line_endpoints(&texels, 0..4);
        let (low, high) = if first[0] < second[0] { (first, second) } else { (second, first) };
        for (end, expected) in [(low, [0.0, 200.0, 50.0, 255.0]), (high, [150.0, 50.0, 50.0, 255.0])] {
            for c in 0..4 {
                assert!((end[c] - expected[c]).abs() < 0.5, "{end:?}");
            }
        }
        let along: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let ends = least_squares(&texels, &along, 0..3).unwrap();
        assert!(ends[0][0].abs() < 0.01 && (ends[1][0] - 150.0).abs() < 0.01, "{ends:?}");
        assert!(least_squares(&texels, &[0.5; 16], 0..3).is_none());
    }
}
