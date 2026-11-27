//! Image processing recipes stored in GTEX v3. The game authors the graph;
//! this decoder supplies only image operations, with bounded dimensions,
//! graph size and filter radii. Evaluation occurs once, before uploading.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "bounded pixel coordinates and channels")]
use crate::packed_texture::{PackedHeader, PackedLevel};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Deserializer};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Recipe {
    nodes: Vec<Node>,
}

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

impl<'de> Deserialize<'de> for Node {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let map = serde_json::Map::<String, serde_json::Value>::deserialize(deserializer)?;
        parse_node(map).map_err(serde::de::Error::custom)
    }
}

fn parse_node(mut map: serde_json::Map<String, serde_json::Value>) -> std::result::Result<Node, String> {
    let take_usize =
        |map: &mut serde_json::Map<String, serde_json::Value>, key: &str| -> std::result::Result<usize, String> {
            let raw = map.remove(key).ok_or_else(|| format!("missing field `{key}`"))?;
            let n = raw.as_u64().ok_or_else(|| format!("expected unsigned integer for `{key}`"))?;
            usize::try_from(n).map_err(|_| format!("integer out of range for `{key}`"))
        };
    let take_f64 =
        |map: &mut serde_json::Map<String, serde_json::Value>, key: &str| -> std::result::Result<f64, String> {
            let raw = map.remove(key).ok_or_else(|| format!("missing field `{key}`"))?;
            raw.as_f64().ok_or_else(|| format!("expected number for `{key}`"))
        };
    let take_bool =
        |map: &mut serde_json::Map<String, serde_json::Value>, key: &str| -> std::result::Result<bool, String> {
            let raw = map.remove(key).ok_or_else(|| format!("missing field `{key}`"))?;
            raw.as_bool().ok_or_else(|| format!("expected bool for `{key}`"))
        };
    let take_f64_vec =
        |map: &mut serde_json::Map<String, serde_json::Value>, key: &str| -> std::result::Result<Vec<f64>, String> {
            let raw = map.remove(key).ok_or_else(|| format!("missing field `{key}`"))?;
            let arr = raw.as_array().ok_or_else(|| format!("expected array for `{key}`"))?;
            arr.iter().map(|v| v.as_f64().ok_or_else(|| format!("expected number in `{key}`"))).collect()
        };
    let op = map
        .remove("op")
        .and_then(|v| match v {
            serde_json::Value::String(s) => Some(s),
            _ => None,
        })
        .ok_or_else(|| "missing field `op`".to_owned())?;
    let node = match op.as_str() {
        "source" => Node::Source,
        "tile" => Node::Tile { input: take_usize(&mut map, "input")?, exponent: take_f64(&mut map, "exponent")? },
        "power" => Node::Power {
            input: take_usize(&mut map, "input")?,
            exponent: take_f64(&mut map, "exponent")?,
            clamp: take_bool(&mut map, "clamp")?,
        },
        "scale" => Node::Scale {
            input: take_usize(&mut map, "input")?,
            scale: take_f64_vec(&mut map, "scale")?,
            offset: take_f64_vec(&mut map, "offset")?,
            clamp: take_bool(&mut map, "clamp")?,
        },
        "luma" => {
            let input = take_usize(&mut map, "input")?;
            let weights: [f64; 3] =
                take_f64_vec(&mut map, "weights")?.try_into().map_err(|_| "expected 3 weights".to_owned())?;
            Node::Luma { input, weights }
        }
        "blur" => Node::Blur { input: take_usize(&mut map, "input")?, sigma: take_f64(&mut map, "sigma")? },
        "subtract" => Node::Subtract { a: take_usize(&mut map, "a")?, b: take_usize(&mut map, "b")? },
        "abs" => Node::Abs { input: take_usize(&mut map, "input")? },
        "join" => Node::Join { a: take_usize(&mut map, "a")?, b: take_usize(&mut map, "b")? },
        "normal" => Node::Normal { input: take_usize(&mut map, "input")?, strength: take_f64(&mut map, "strength")? },
        other => return Err(format!("unknown texture recipe op `{other}`")),
    };
    if let Some(extra) = map.keys().next() {
        return Err(format!("unknown field `{extra}`"));
    }
    Ok(node)
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

fn parse_recipe(bytes: &[u8], header: PackedHeader, kind: u8) -> Result<(Recipe, Vec<usize>, &[u8])> {
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
    Ok((recipe, uses, &bytes[4 + length..]))
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn generate(bytes: &[u8], header: PackedHeader, kind: u8) -> Result<Vec<PackedLevel>> {
    let (recipe, uses, image) = parse_recipe(bytes, header, kind)?;
    // Read dimensions before allocating the source image.
    let reader = image::ImageReader::with_format(std::io::Cursor::new(image), image::ImageFormat::Jpeg);
    let (iw, ih) = reader.into_dimensions()?;
    ensure!(iw > 0 && ih > 0 && u64::from(iw) * u64::from(ih) <= 4096 * 4096, "recipe source image is too large");
    let source_rgb = if matches!(recipe.nodes.first(), Some(Node::Source)) {
        let source = image::load_from_memory_with_format(image, image::ImageFormat::Jpeg)?.into_rgb8();
        let source =
            image::imageops::resize(&source, header.width, header.height, image::imageops::FilterType::Lanczos3);
        Some(source.into_raw())
    } else {
        None
    };
    evaluate(recipe, uses, source_rgb, header, kind)
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn generate_async(bytes: &[u8], header: PackedHeader, kind: u8) -> Result<Vec<PackedLevel>> {
    let (recipe, uses, image) = parse_recipe(bytes, header, kind)?;
    let decoded = crate::platform::decode_image(image.to_vec()).await?;
    let (iw, ih) = (decoded.width, decoded.height);
    ensure!(iw > 0 && ih > 0 && u64::from(iw) * u64::from(ih) <= 4096 * 4096, "recipe source image is too large");
    let source_rgb = if matches!(recipe.nodes.first(), Some(Node::Source)) {
        let mut rgb = Vec::with_capacity((iw as usize) * (ih as usize) * 3);
        for px in decoded.pixels.as_chunks::<4>().0 {
            rgb.extend_from_slice(&px[..3]);
        }
        Some(if iw == header.width && ih == header.height {
            rgb
        } else {
            resize_rgb_lanczos3(&rgb, iw, ih, header.width, header.height)
        })
    } else {
        None
    };
    evaluate(recipe, uses, source_rgb, header, kind)
}

#[cfg(any(target_arch = "wasm32", test))]
fn lanczos3(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 1e-12 {
        1.0
    } else if ax >= 3.0 {
        0.0
    } else {
        let px = std::f64::consts::PI * ax;
        (px.sin() / px) * ((px / 3.0).sin() / (px / 3.0))
    }
}

#[cfg(any(target_arch = "wasm32", test))]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "clamped to 4096x4096 dimensions and 0..=255 pixel range"
)]
fn resize_rgb_lanczos3(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let (sw, sh, dw, dh) = (src_w as usize, src_h as usize, dst_w as usize, dst_h as usize);
    // Horizontal pass into f64 buffer (sh x dw x 3).
    let scale_x = (sw as f64 / dw as f64).max(1.0);
    let support_x = 3.0 * scale_x;
    let mut horiz = vec![0.0f64; sh * dw * 3];
    let mut weights = Vec::new();
    for dx in 0..dw {
        let input = (dx as f64 + 0.5) * (sw as f64 / dw as f64);
        let left = (input - support_x).floor().clamp(0.0, (sw - 1) as f64) as usize;
        let right = (input + support_x).ceil().clamp((left + 1) as f64, sw as f64) as usize;
        let center = input - 0.5;
        weights.clear();
        let mut sum = 0.0;
        for sx in left..right {
            let w = lanczos3((sx as f64 - center) / scale_x);
            weights.push((sx, w));
            sum += w;
        }
        let inv = if sum.abs() > 1e-12 { 1.0 / sum } else { 1.0 };
        for y in 0..sh {
            let mut acc = [0.0f64; 3];
            for &(sx, w) in &weights {
                let p = (y * sw + sx) * 3;
                for c in 0..3 {
                    acc[c] += f64::from(src[p + c]) * w;
                }
            }
            let out = (y * dw + dx) * 3;
            for c in 0..3 {
                horiz[out + c] = acc[c] * inv;
            }
        }
    }
    // Vertical pass into u8 RGB buffer (dh x dw x 3).
    let scale_y = (sh as f64 / dh as f64).max(1.0);
    let support_y = 3.0 * scale_y;
    let mut out = vec![0u8; dh * dw * 3];
    for dy in 0..dh {
        let input = (dy as f64 + 0.5) * (sh as f64 / dh as f64);
        let top = (input - support_y).floor().clamp(0.0, (sh - 1) as f64) as usize;
        let bottom = (input + support_y).ceil().clamp((top + 1) as f64, sh as f64) as usize;
        let center = input - 0.5;
        weights.clear();
        let mut sum = 0.0;
        for sy in top..bottom {
            let w = lanczos3((sy as f64 - center) / scale_y);
            weights.push((sy, w));
            sum += w;
        }
        let inv = if sum.abs() > 1e-12 { 1.0 / sum } else { 1.0 };
        for x in 0..dw {
            let mut acc = [0.0f64; 3];
            for &(sy, w) in &weights {
                let p = (sy * dw + x) * 3;
                for c in 0..3 {
                    acc[c] += horiz[p + c] * w;
                }
            }
            let dst = (dy * dw + x) * 3;
            for c in 0..3 {
                out[dst + c] = (acc[c] * inv).clamp(0.0, 255.0).round() as u8;
            }
        }
    }
    out
}

fn evaluate(
    recipe: Recipe,
    mut uses: Vec<usize>,
    mut source_rgb: Option<Vec<u8>>,
    header: PackedHeader,
    kind: u8,
) -> Result<Vec<PackedLevel>> {
    let (w, h) = (header.width as usize, header.height as usize);
    let mut fields: Vec<Option<Field>> = Vec::new();
    for (i, node) in recipe.nodes.iter().enumerate() {
        let field = |index: usize| fields[index].as_ref().expect("referenced fields are retained");
        let made = match node {
            Node::Source => {
                ensure!(i == 0, "texture recipe source must be its first node");
                let raw = source_rgb.take().context("missing decoded recipe source image")?;
                Field { channels: 3, pixels: raw.into_iter().map(|v| f64::from(v) / 255.0).collect() }
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

    #[test]
    fn lanczos3_resizer_matches_image_crate() {
        let (sw, sh) = (17u32, 13u32);
        let mut src = Vec::with_capacity((sw * sh * 3) as usize);
        for y in 0..sh {
            for x in 0..sw {
                src.push(((x * 15 + y * 7) % 256) as u8);
                src.push(((x * 9 + y * 19 + 30) % 256) as u8);
                src.push(((x * 23 + y * 5 + 90) % 256) as u8);
            }
        }
        let img = image::RgbImage::from_raw(sw, sh, src.clone()).unwrap();
        for (dw, dh) in [(8u32, 8u32), (32, 24), (11, 19)] {
            let expected = image::imageops::resize(&img, dw, dh, image::imageops::FilterType::Lanczos3);
            let actual = super::resize_rgb_lanczos3(&src, sw, sh, dw, dh);
            assert_eq!(actual.len(), expected.as_raw().len());
            let max_diff = actual.iter().zip(expected.as_raw()).map(|(&a, &b)| a.abs_diff(b)).max().unwrap_or(0);
            assert!(max_diff <= 3, "resize {sw}x{sh} -> {dw}x{dh}: max diff {max_diff} > 3");
        }
    }
}
