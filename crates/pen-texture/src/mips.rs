// Mip chains: each level half the size of the one above (WebGPU's sizes,
// `ktx2::level_size`) down to 1 x 1, by a box filter that averages the texels
// each new texel covers. Across an odd size a new texel covers one texel
// above whole and parts of the two beside it, weighted by how much of each,
// so nothing shifts. Colour averages in linear light, weighted by alpha so
// transparent texels lend no colour (no dark fringes); data averages as
// stored.
use crate::encode::to_u8;
use crate::ktx2;

/// Every mip level of `rgba` (`width` x `height` RGBA8), level 0 (a copy)
/// down to 1 x 1, each from the one above. `color`: RGB is sRGB-encoded and
/// alpha straight; otherwise every channel is data.
pub fn mip_chain(rgba: &[u8], width: u32, height: u32, color: bool) -> Vec<Vec<u8>> {
    let linear: [f32; 256] = std::array::from_fn(|value| srgb_to_linear(value as f32 / 255.0));
    let mut levels = vec![rgba.to_vec()];
    for level in 1..ktx2::mip_levels(width, height) {
        let (above_width, above_height) = (ktx2::level_size(width, level - 1), ktx2::level_size(height, level - 1));
        let above = &levels[levels.len() - 1];
        let next = downsample(above, above_width, above_height, color.then_some(&linear));
        levels.push(next);
    }
    levels
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 { value / 12.92 } else { ((value + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 { value * 12.92 } else { 1.055 * value.powf(1.0 / 2.4) - 0.055 }
}

/// What each texel along an axis of the next level takes from `size` texels
/// above: up to three texels and their weights, which sum to 1.
fn taps(size: u32) -> Vec<[(usize, f32); 3]> {
    let half = (size >> 1).max(1) as usize;
    let size_f = size as f32;
    (0..half)
        .map(|i| match size {
            1 => [(0, 1.0), (0, 0.0), (0, 0.0)],
            _ if size.is_multiple_of(2) => [(2 * i, 0.5), (2 * i + 1, 0.5), (2 * i, 0.0)],
            // `half` texels span `size`: each covers 2 + 1 / half texels above.
            _ => [
                (2 * i, (half - i) as f32 / size_f),
                (2 * i + 1, half as f32 / size_f),
                (2 * i + 2, (i + 1) as f32 / size_f),
            ],
        })
        .collect()
}

/// The level below `rgba` (`width` x `height`). With `linear` (sRGB to
/// linear for each 8-bit value), RGB averages in linear light weighted by
/// alpha; without, every channel averages as it is.
fn downsample(rgba: &[u8], width: u32, height: u32, linear: Option<&[f32; 256]>) -> Vec<u8> {
    let (columns, rows) = (taps(width), taps(height));
    let width = width as usize;
    let mut out = Vec::with_capacity(columns.len() * rows.len() * 4);
    for row in &rows {
        for column in &columns {
            // Weighted sums: of each channel, or for colour of alpha-weighted
            // linear RGB and alpha, with plain linear RGB for when every
            // texel covered is transparent.
            let mut sum = [0.0f32; 4];
            let mut plain = [0.0f32; 3];
            for &(y, row_weight) in row {
                for &(x, column_weight) in column {
                    let weight = row_weight * column_weight;
                    if weight == 0.0 {
                        continue;
                    }
                    let texel = &rgba[(y * width + x) * 4..][..4];
                    let Some(linear) = linear else {
                        for (sum, &value) in sum.iter_mut().zip(texel) {
                            *sum += weight * f32::from(value);
                        }
                        continue;
                    };
                    let alpha = weight * f32::from(texel[3]) / 255.0;
                    for c in 0..3 {
                        let value = linear[usize::from(texel[c])];
                        sum[c] += alpha * value;
                        plain[c] += weight * value;
                    }
                    sum[3] += alpha;
                }
            }
            if linear.is_none() {
                out.extend(sum.map(to_u8));
                continue;
            }
            let alpha = sum[3];
            for c in 0..3 {
                let value = if alpha > 0.0 { sum[c] / alpha } else { plain[c] };
                out.push(to_u8(linear_to_srgb(value) * 255.0));
            }
            out.push(to_u8(alpha * 255.0));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texel(level: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
        let at = ((y * width + x) * 4) as usize;
        level[at..at + 4].try_into().unwrap()
    }

    #[test]
    fn levels_halve_down_to_one_texel() {
        let chain = mip_chain(&[9; 5 * 3 * 4], 5, 3, true);
        assert_eq!(chain.iter().map(Vec::len).collect::<Vec<_>>(), [5 * 3 * 4, 2 * 4, 4]);
        let chain = mip_chain(&[9; 8 * 2 * 4], 8, 2, false);
        assert_eq!(chain.iter().map(Vec::len).collect::<Vec<_>>(), [64, 16, 8, 4]);
        // One colour stays that colour.
        assert!(chain.iter().all(|level| level.iter().all(|&value| value == 9)));
        assert_eq!(mip_chain(&[1, 2, 3, 4], 1, 1, true), vec![vec![1, 2, 3, 4]]);
    }

    #[test]
    fn colour_averages_in_linear_light() {
        let black = [0, 0, 0, 255];
        let white = [255, 255, 255, 255];
        let image = [black, white, white, black].concat();
        let chain = mip_chain(&image, 2, 2, true);
        // Half of white's light is sRGB 188, not 128.
        assert_eq!(chain[1], [188, 188, 188, 255]);
        let data = mip_chain(&image, 2, 2, false);
        assert_eq!(data[1], [128, 128, 128, 255]);
        let known = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]].concat();
        assert_eq!(mip_chain(&known, 2, 2, false)[1], [70, 80, 90, 100]);
    }

    #[test]
    fn transparent_texels_lend_no_colour() {
        let red = [255, 0, 0, 255];
        let clear = [0, 0, 0, 0];
        let chain = mip_chain(&[red, clear, clear, red].concat(), 2, 2, true);
        assert_eq!(chain[1], [255, 0, 0, 128]);
        // Wholly transparent texels still average their colour.
        let ghost = [200, 100, 50, 0];
        let chain = mip_chain(&[ghost, ghost, ghost, ghost].concat(), 2, 2, true);
        assert_eq!(chain[1], ghost);
        // Data is averaged as it is.
        let chain = mip_chain(&[red, clear, clear, red].concat(), 2, 2, false);
        assert_eq!(chain[1], [128, 0, 0, 128]);
    }

    #[test]
    fn odd_sizes_keep_a_centred_feature_centred() {
        // A bright texel in the middle of a 5 x 3 image.
        let mut image = vec![0u8; 5 * 3 * 4];
        image[(5 + 2) * 4..(5 + 2) * 4 + 4].copy_from_slice(&[255, 255, 255, 255]);
        let chain = mip_chain(&image, 5, 3, false);
        let (left, right) = (texel(&chain[1], 2, 0, 0), texel(&chain[1], 2, 1, 0));
        assert_eq!(left, right);
        // Each of the two covers 2.5 texels across and all 3 down; the bright
        // one lies half in each: (255 / 2.5 / 2) / 3.
        assert_eq!(left, [17, 17, 17, 17]);
        // The middle of 7 texels lands wholly in the middle of 3, at 3 / 7.
        let mut row = vec![0u8; 7 * 4];
        row[12..16].copy_from_slice(&[255, 255, 255, 255]);
        assert_eq!(mip_chain(&row, 7, 1, false)[1], [[0; 4], [109; 4], [0; 4]].concat());
        // A centred texel of a 3 x 3 image weighs a third each way.
        let mut middle = vec![0u8; 9 * 4];
        middle[16..20].copy_from_slice(&[90, 90, 90, 90]);
        assert_eq!(mip_chain(&middle, 3, 3, false)[1], [10, 10, 10, 10]);
    }

    #[test]
    fn mips_are_deterministic() {
        let image = crate::encode::tests::test_image(37, 21);
        assert_eq!(mip_chain(&image, 37, 21, true), mip_chain(&image, 37, 21, true));
    }
}
