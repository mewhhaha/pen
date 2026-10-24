//! The asset pipeline's texture tools (pen-texture): `pen texture` turns a
//! PNG into a KTX2 file for each format family a GPU may sample directly,
//! which the renderer picks between by its device's features; `pen font`
//! makes a font's multi-channel signed distance field atlas, for text that
//! stays sharp at any size, with each glyph's metrics beside it as JSON.
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use pen_texture::encode::{Kind, encode_all};
use pen_texture::ktx2::{self, Family};

/// `out` (`name.ktx2`, the key a material names) written as its variants:
/// `name.bc7.ktx2`, `name.astc.ktx2`, `name.etc2.ktx2` and `name.rgba.ktx2`.
/// Non-block-aligned dimensions emit only RGBA and remove old variants.
pub fn texture(image: &Path, out: Option<PathBuf>, linear: bool, mipmaps: bool) -> Result<()> {
    let out = out.unwrap_or_else(|| image.with_extension("ktx2"));
    let name =
        out.to_str().and_then(|out| out.strip_suffix(".ktx2")).context("the texture's file must end in .ktx2")?;
    ensure!(ktx2::variant_of(&out.to_string_lossy()).is_none(), "the texture's file must not name a format family");
    let decoded = image::open(image).with_context(|| format!("reading {}", image.display()))?.into_rgba8();
    let (width, height) = decoded.dimensions();
    let started = Instant::now();
    let kind = if linear { Kind::Data } else { Kind::Color };
    let files = encode_all(decoded.as_raw(), width, height, kind, mipmaps)?;
    for (family, bytes) in &files {
        let path = format!("{name}.{}.ktx2", family.name());
        write(Path::new(&path), bytes)?;
        let format = ktx2::Reader::open(bytes.clone())?.header().format;
        println!("{path}: {format:?}, {} KB", bytes.len().div_ceil(1024));
    }
    // Publish the fallback before removing old, preferred variants. A later
    // run with new dimensions must never load an earlier image's pixels.
    for family in Family::ALL {
        if !files.iter().any(|(kept, _)| *kept == family) {
            let path = PathBuf::from(format!("{name}.{}.ktx2", family.name()));
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error).with_context(|| format!("removing {}", path.display())),
            }
        }
    }
    println!("texture: {width} x {height} in {:.2} s", started.elapsed().as_secs_f64());
    Ok(())
}

/// Writes in one rename, so a reader never sees half a file.
fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    let partial = path.with_extension("ktx2.partial");
    std::fs::write(&partial, bytes).with_context(|| format!("writing {}", partial.display()))?;
    std::fs::rename(&partial, path).with_context(|| format!("writing {}", path.display()))
}

/// What `pen font` draws.
pub struct FontOptions {
    /// Pixels per em.
    pub em: f32,
    /// The field's distance either side of an edge, in pixels.
    pub range: f32,
    pub width: u32,
    pub weight: Option<f32>,
    pub chars: Option<String>,
}

/// The atlas of `font` as `out` (a `.ktx2` key, whose texels are read as data,
/// never as sRGB colour or in compressed blocks, which would move the edges),
/// and its metrics as the same name's `.json`.
pub fn font(font: &Path, out: Option<PathBuf>, options: FontOptions) -> Result<()> {
    let out = out.unwrap_or_else(|| font.with_extension("ktx2"));
    ensure!(out.extension().is_some_and(|extension| extension == "ktx2"), "the atlas's file must end in .ktx2");
    ensure!(ktx2::variant_of(&out.to_string_lossy()).is_none(), "the atlas's file must not name a format family");
    let data = std::fs::read(font).with_context(|| format!("reading {}", font.display()))?;
    let chars: Vec<char> = match &options.chars {
        Some(chars) => chars.chars().collect(),
        None => (' '..='~').collect(),
    };
    let started = Instant::now();
    let atlas = pen_texture::msdf::atlas(
        &data,
        &pen_texture::msdf::Options {
            size: options.em,
            range: options.range,
            width: options.width,
            variations: options.weight.map(|weight| (*b"wght", weight)).into_iter().collect(),
            chars,
        },
    )?;
    let bytes = ktx2::write(ktx2::Format::Rgba8Unorm, atlas.width, atlas.height, std::slice::from_ref(&atlas.pixels))?;
    write(&out, &bytes)?;
    // No family variants: a texture key with none reads its own file.
    for family in Family::ALL {
        let stale = ktx2::variant_key(&out.to_string_lossy(), family).map(PathBuf::from);
        if let Some(stale) = stale.filter(|stale| stale.exists()) {
            std::fs::remove_file(&stale).with_context(|| format!("removing {}", stale.display()))?;
        }
    }
    let glyphs: Vec<serde_json::Value> = atlas
        .glyphs
        .iter()
        .map(|glyph| {
            serde_json::json!({
                "code": u32::from(glyph.code),
                "u0": glyph.u0, "v0": glyph.v0, "u1": glyph.u1, "v1": glyph.v1,
                "left": glyph.left, "top": glyph.top, "width": glyph.width, "height": glyph.height,
                "advance": glyph.advance,
            })
        })
        .collect();
    let metrics = serde_json::json!({
        "atlas": out.file_name().map(|name| name.to_string_lossy().into_owned()),
        "width": atlas.width,
        "height": atlas.height,
        "em": options.em,
        "range": options.range,
        "spread": atlas.spread,
        "ascent": atlas.ascent,
        "descent": atlas.descent,
        "glyphs": glyphs,
    });
    let json = out.with_extension("json");
    std::fs::write(&json, serde_json::to_string_pretty(&metrics)? + "\n")
        .with_context(|| format!("writing {}", json.display()))?;
    println!(
        "font: {} glyphs in {} x {}, {} and {}, in {:.2} s",
        atlas.glyphs.len(),
        atlas.width,
        atlas.height,
        out.display(),
        json.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resizing_a_texture_replaces_preferred_variants_with_the_new_fallback() {
        let root = crate::project::temporary_dir().unwrap().join(format!("texture-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("sample.png");
        let out = root.join("sample.ktx2");
        for width in [8, 6] {
            image::RgbaImage::from_pixel(width, 8, image::Rgba([24, 65, 128, 255])).save(&source).unwrap();
            texture(&source, Some(out.clone()), false, true).unwrap();
            for family in Family::ALL {
                let file = root.join(format!("sample.{}.ktx2", family.name()));
                assert_eq!(file.is_file(), width == 8 || family == Family::Rgba);
            }
            let bytes = std::fs::read(root.join("sample.rgba.ktx2")).unwrap();
            let mut reader = ktx2::Reader::open(bytes).unwrap();
            assert_eq!(reader.header().width, width);
            while let Some(level) = reader.next_level().unwrap() {
                if level.level == 0 {
                    assert!(level.data.as_chunks::<4>().0.iter().all(|pixel| *pixel == [24, 65, 128, 255]));
                }
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
