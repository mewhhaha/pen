//! Image processing recipes stored in GTEX v3. The game authors the graph;
//! this decoder supplies only image operations, with bounded dimensions,
//! graph size and filter radii. Evaluation occurs once, before uploading.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "bounded pixel coordinates and channels")]
use crate::packed_texture::{PackedHeader, PackedLevel};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Recipe {
    nodes: Vec<Node>,
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Node {
    Source,
    Tile { input: usize, exponent: f64 },
    Power { input: usize, exponent: f64, clamp: bool },
    Scale { input: usize, scale: Vec<f64>, offset: Vec<f64>, clamp: bool },
    Luma { input: usize, weights: [f64; 3] },
    Blur { input: usize, sigma: f64 },
    Subtract { a: usize, b: usize },
    Abs { input: usize },
    Join { a: usize, b: usize },
    Normal { input: usize, strength: f64 },
}

impl Node {
    fn inputs(&self) -> Vec<usize> {
        match *self {
            Self::Source => vec![],
            Self::Subtract { a, b } | Self::Join { a, b } => vec![a, b],
            Self::Tile { input, .. }
            | Self::Power { input, .. }
            | Self::Scale { input, .. }
            | Self::Luma { input, .. }
            | Self::Blur { input, .. }
            | Self::Abs { input }
            | Self::Normal { input, .. } => vec![input],
        }
    }
}

struct Field {
    channels: usize,
    pixels: Vec<f64>,
}
impl Field {
    fn map(&self, f: impl Fn(f64, usize) -> f64) -> Self {
        Self {
            channels: self.channels,
            pixels: self.pixels.iter().enumerate().map(|(i, &v)| f(v, i % self.channels)).collect(),
        }
    }
}

pub(crate) fn generate(bytes: &[u8], header: PackedHeader, kind: u8) -> Result<Vec<PackedLevel>> {
    let (w, h) = (header.width as usize, header.height as usize);
    ensure!(
        w.is_power_of_two() && h.is_power_of_two() && w * h <= 1024 * 1024,
        "recipe texture needs power-of-two dimensions, at most 1M pixels"
    );
    ensure!([1, 3, 4].contains(&kind), "recipe texture kind must be color, data or normals");
    let length = u32::from_le_bytes(bytes.get(..4).context("missing texture recipe length")?.try_into()?) as usize;
    ensure!(
        length <= 64 * 1024 && length + 4 < bytes.len() && bytes.len() <= 8 * 1024 * 1024,
        "invalid texture recipe size"
    );
    let recipe: Recipe = serde_json::from_slice(&bytes[4..4 + length])?;
    ensure!(!recipe.nodes.is_empty() && recipe.nodes.len() <= 32, "texture recipe needs 1..32 nodes");
    let mut uses = vec![0usize; recipe.nodes.len()];
    let mut work = 0u64;
    for (i, node) in recipe.nodes.iter().enumerate() {
        for input in node.inputs() {
            ensure!(input < i, "texture recipe has a forward reference");
            uses[input] += 1;
        }
        if let Node::Blur { sigma, .. } = node {
            ensure!(sigma.is_finite() && *sigma > 0.0 && *sigma <= 24.0, "texture blur sigma must be 0..24");
            work += (6.0 * sigma + 3.0) as u64 * 2 * (w * h) as u64;
        }
    }
    ensure!(work <= 400_000_000, "texture recipe exceeds its filter work limit");
    let image = &bytes[4 + length..];
    // Read dimensions before allocating the source image.
    let reader = image::ImageReader::with_format(std::io::Cursor::new(image), image::ImageFormat::Jpeg);
    let (iw, ih) = reader.into_dimensions()?;
    ensure!(iw > 0 && ih > 0 && u64::from(iw) * u64::from(ih) <= 4096 * 4096, "recipe source image is too large");
    let mut fields: Vec<Option<Field>> = Vec::new();
    for (i, node) in recipe.nodes.iter().enumerate() {
        let field = |index: usize| fields[index].as_ref().expect("referenced fields are retained");
        let made = match node {
            Node::Source => {
                ensure!(i == 0, "texture recipe source must be its first node");
                let source = image::load_from_memory_with_format(image, image::ImageFormat::Jpeg)?.into_rgb8();
                let source = image::imageops::resize(
                    &source,
                    header.width,
                    header.height,
                    image::imageops::FilterType::Lanczos3,
                );
                Field { channels: 3, pixels: source.into_raw().into_iter().map(|v| f64::from(v) / 255.0).collect() }
            }
            Node::Tile { input, exponent } => {
                ensure!(exponent.is_finite() && *exponent > 0.0, "invalid tile exponent");
                let f = field(*input);
                let wx: Vec<_> = (0..w)
                    .map(|x| (std::f64::consts::PI * (x as f64 + 0.5) / w as f64).sin().powf(*exponent))
                    .collect();
                let wy: Vec<_> = (0..h)
                    .map(|y| (std::f64::consts::PI * (y as f64 + 0.5) / h as f64).sin().powf(*exponent))
                    .collect();
                let mut pixels = f.pixels.clone();
                for (y, &wy) in wy.iter().enumerate() {
                    for (x, &wx) in wx.iter().enumerate() {
                        for c in 0..f.channels {
                            let at = (y * w + x) * f.channels + c;
                            let shifted = (((y + h / 2) % h) * w + (x + w / 2) % w) * f.channels + c;
                            let weight = wx * wy;
                            pixels[at] = f.pixels[at] * weight + f.pixels[shifted] * (1.0 - weight);
                        }
                    }
                }
                Field { channels: f.channels, pixels }
            }
            Node::Power { input, exponent, clamp } => {
                ensure!(exponent.is_finite() && *exponent > 0.0, "invalid texture power");
                field(*input).map(|v, _| if *clamp { v.clamp(0.0, 1.0) } else { v }.powf(*exponent))
            }
            Node::Scale { input, scale, offset, clamp } => {
                let f = field(*input);
                ensure!(
                    scale.len() == f.channels
                        && offset.len() == f.channels
                        && scale.iter().chain(offset).all(|v| v.is_finite()),
                    "texture scale does not match its channels"
                );
                f.map(|v, c| {
                    let v = v * scale[c] + offset[c];
                    if *clamp { v.clamp(0.0, 1.0) } else { v }
                })
            }
            Node::Luma { input, weights } => {
                let f = field(*input);
                ensure!(
                    f.channels == 3 && weights.iter().all(|v| v.is_finite()),
                    "luma requires three finite weights and RGB"
                );
                Field {
                    channels: 1,
                    pixels: f
                        .pixels
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|p| p.iter().zip(weights).map(|(a, b)| a * b).sum())
                        .collect(),
                }
            }
            Node::Blur { input, sigma } => {
                let f = field(*input);
                ensure!(f.channels == 1, "blur requires a scalar image");
                let radius = (3.0 * sigma).floor().max(1.0) as usize;
                let mut kernel: Vec<f64> = (0..=radius * 2)
                    .map(|i| (-((i as f64 - radius as f64).powi(2)) / (2.0 * sigma * sigma)).exp())
                    .collect();
                let sum: f64 = kernel.iter().sum();
                kernel.iter_mut().for_each(|v| *v /= sum);
                let mut pixels = f.pixels.clone();
                for vertical in [true, false] {
                    let mut next = vec![0.0; w * h];
                    for (tap, &weight) in kernel.iter().enumerate() {
                        for y in 0..h {
                            let sy = if vertical { (y + h + radius - tap % h) % h } else { y };
                            for x in 0..w {
                                let sx = if vertical { x } else { (x + w + radius - tap % w) % w };
                                next[y * w + x] += pixels[sy * w + sx] * weight;
                            }
                        }
                    }
                    pixels = next;
                }
                Field { channels: 1, pixels }
            }
            Node::Subtract { a, b } => {
                let (a, b) = (field(*a), field(*b));
                ensure!(a.channels == b.channels, "texture difference channels disagree");
                Field { channels: a.channels, pixels: a.pixels.iter().zip(&b.pixels).map(|(a, b)| a - b).collect() }
            }
            Node::Abs { input } => field(*input).map(|v, _| v.abs()),
            Node::Join { a, b } => {
                let (a, b) = (field(*a), field(*b));
                ensure!(a.channels + b.channels <= 4, "too many texture channels");
                Field {
                    channels: a.channels + b.channels,
                    pixels: a
                        .pixels
                        .chunks_exact(a.channels)
                        .zip(b.pixels.chunks_exact(b.channels))
                        .flat_map(|(a, b)| a.iter().chain(b).copied())
                        .collect(),
                }
            }
            Node::Normal { input, strength } => {
                let f = field(*input);
                ensure!(f.channels == 1 && strength.is_finite(), "normal needs scalar height and finite strength");
                let mut pixels = Vec::with_capacity(w * h * 2);
                for y in 0..h {
                    for x in 0..w {
                        let gx = (f.pixels[y * w + (x + 1) % w] - f.pixels[y * w + (x + w - 1) % w]) * 0.5 * strength;
                        let gy =
                            (f.pixels[((y + 1) % h) * w + x] - f.pixels[((y + h - 1) % h) * w + x]) * 0.5 * strength;
                        let length = (gx * gx + gy * gy + 1.0).sqrt();
                        pixels.extend([-gx / length * 0.5 + 0.5, gy / length * 0.5 + 0.5]);
                    }
                }
                Field { channels: 2, pixels }
            }
        };
        ensure!(made.pixels.iter().all(|v| v.is_finite()), "texture recipe produced nonfinite pixels");
        fields.push(Some(made));
        for input in node.inputs() {
            uses[input] -= 1;
            if uses[input] == 0 {
                fields[input] = None;
            }
        }
        // Dead nodes need not retain their storage either.
        if uses[i] == 0 && i + 1 < recipe.nodes.len() {
            fields[i] = None;
        }
    }
    let output = fields.pop().flatten().context("texture recipe has no output")?;
    ensure!(
        output.channels == crate::packed_texture::texel_bytes(header.format) as usize,
        "recipe output channel count disagrees with its header"
    );
    ensure!(kind != 4 || output.channels == 2, "recipe normals require two channels");
    Ok(mips(output, header, kind))
}

fn to8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
fn linear(v: f64) -> f64 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}
fn srgb(v: f64) -> f64 {
    if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.max(0.0).powf(1.0 / 2.4) - 0.055 }
}

fn mips(field: Field, header: PackedHeader, kind: u8) -> Vec<PackedLevel> {
    let mut w = header.width as usize;
    let mut h = header.height as usize;
    let pixels: Vec<u8> = field.pixels.into_iter().map(to8).collect();
    let mut levels = vec![PackedLevel { level: 0, width: header.width, height: header.height, pixels: pixels.clone() }];
    let channels = if kind == 4 { 3 } else { field.channels };
    let mut values: Vec<f64> = pixels
        .chunks_exact(field.channels)
        .flat_map(|p| {
            let mut v: Vec<_> = p.iter().map(|&v| f64::from(v) / 255.0).collect();
            if kind == 1 {
                v[..3].iter_mut().for_each(|v| *v = linear(*v));
            }
            if kind == 4 {
                v[0] = v[0] * 2.0 - 1.0;
                v[1] = v[1] * 2.0 - 1.0;
                v.push((1.0 - v[0] * v[0] - v[1] * v[1]).max(0.0).sqrt());
            }
            v
        })
        .collect();
    for level in 1..header.levels {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut next = vec![0.0; nw * nh * channels];
        for y in 0..nh {
            for x in 0..nw {
                for dy in 0..2 {
                    for dx in 0..2 {
                        for c in 0..channels {
                            next[(y * nw + x) * channels + c] +=
                                values[(((2 * y + dy).min(h - 1)) * w + (2 * x + dx).min(w - 1)) * channels + c] * 0.25;
                        }
                    }
                }
            }
        }
        let pixels = next
            .chunks_exact(channels)
            .flat_map(|p| {
                let mut v = p.to_vec();
                if kind == 1 {
                    v[..3].iter_mut().for_each(|v| *v = srgb(*v));
                }
                if kind == 4 {
                    let length = v.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-6);
                    v.truncate(2);
                    v.iter_mut().for_each(|v| *v = *v / length * 0.5 + 0.5);
                }
                v.into_iter().map(to8)
            })
            .collect();
        levels.push(PackedLevel { level, width: nw as u32, height: nh as u32, pixels });
        values = next;
        w = nw;
        h = nh;
    }
    levels.reverse();
    levels
}

#[cfg(test)]
mod tests {
    use crate::packed_texture::PackedReader;
    use serde_json::json;

    fn asset(nodes: serde_json::Value, channels: u8, kind: u8) -> Vec<u8> {
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 100)
            .encode(&[128; 8 * 8 * 3], 8, 8, image::ExtendedColorType::Rgb8)
            .unwrap();
        let graph = serde_json::to_vec(&json!({"nodes":nodes})).unwrap();
        let mut bytes = b"GTEX".to_vec();
        bytes.extend([3, channels, kind, 4]);
        bytes.extend(8u32.to_le_bytes());
        bytes.extend(8u32.to_le_bytes());
        bytes.extend(u32::try_from(graph.len()).unwrap().to_le_bytes());
        bytes.extend(graph);
        bytes.extend(jpeg);
        bytes
    }

    #[test]
    fn reconstructs_independent_alpha_and_unit_normals_at_every_mip() {
        let bytes = asset(
            json!([{"op":"source"},{"op":"luma","input":0,"weights":[0,0,0]},
            {"op":"join","a":0,"b":1}]),
            4,
            1,
        );
        let mut reader = PackedReader::open(bytes).unwrap();
        let mut levels = 0;
        while let Some(level) = reader.next_level().unwrap() {
            assert!(
                level.pixels.as_chunks::<4>().0.iter().all(|p| *p == [128, 128, 128, 0]),
                "color must survive zero gloss"
            );
            levels += 1;
        }
        assert_eq!(levels, 4);
        let bytes = asset(
            json!([{"op":"source"},{"op":"luma","input":0,"weights":[0.299,0.587,0.114]},
            {"op":"blur","input":1,"sigma":2.0},{"op":"normal","input":2,"strength":80}]),
            2,
            4,
        );
        let mut reader = PackedReader::open(bytes).unwrap();
        while let Some(level) = reader.next_level().unwrap() {
            assert!(level.pixels.iter().all(|&v| v == 128), "flat surface must stay flat");
        }
    }

    #[test]
    fn refuses_cyclic_overbudget_and_truncated_recipes() {
        for nodes in [
            json!([{"op":"source"},{"op":"abs","input":1}]),
            json!([{"op":"source"},{"op":"blur","input":0,"sigma":100}]),
            json!([{"op":"source"},{"op":"scale","input":0,"scale":[1],"offset":[0],"clamp":false}]),
        ] {
            assert!(PackedReader::open(asset(nodes, 4, 1)).is_err());
        }
        let bytes = asset(
            json!([{"op":"source"},{"op":"luma","input":0,"weights":[0,0,0]},
            {"op":"join","a":0,"b":1}]),
            4,
            1,
        );
        for end in 0..20 {
            assert!(PackedReader::open(bytes[..end].to_vec()).is_err());
        }
        let mut oversized = bytes;
        oversized[8..12].copy_from_slice(&16384u32.to_le_bytes());
        assert!(PackedReader::open(oversized).is_err());
    }
}
