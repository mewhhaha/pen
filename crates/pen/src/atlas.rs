//! Texture atlases, packed by the asset pipeline: the built-in step `atlas`
//! (asset_graph.rs) packs its source images, sprites and interface pieces,
//! into a few pages, so that a frame's sprites share one texture, and so one
//! draw, instead of a texture each.
//!
//! ```json
//! { "recipe": "ui-atlas", "builtin": "atlas", "sources": ["art/ui/*.png"],
//!   "outputs": ["src/assets/ui_atlas*", "src/ui_atlas.blot"],
//!   "options": { "out": "src/assets/ui_atlas", "blot": "src/ui_atlas.blot" } }
//! ```
//!
//! writes `ui_atlas.png` (and `ui_atlas_1.png` and on when one page is not
//! enough), `ui_atlas.atlas.json`, where each sprite is, and, asked for, a
//! Blot module with each sprite's texture coordinates for ink's `ui.image`.
//! Each image is trimmed of its fully transparent border (the frame says how
//! much, so it can be put back where it was), and its edge pixels are
//! repeated round it (`extrude`, 1 by default), so that filtering at its edge
//! never reaches into its neighbours. Options: `out`, `blot`, `maxSize`
//! (2048), `extrude` (1), `spacing` (1), `trim` (true) and `powerOfTwo`
//! (false: pages are cut to what they hold, rounded up to 4 pixels).
//!
//! Packing is MaxRects with the best short side fit, largest images first:
//! the same sources always give the same pages, byte for byte.
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::asset_graph::Context as StepContext;
use crate::pipeline::matches;
use crate::store::write_atomically;

/// A rectangle of pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    fn right(&self) -> u32 {
        self.x + self.width
    }

    fn bottom(&self) -> u32 {
        self.y + self.height
    }

    fn contains(&self, other: &Rect) -> bool {
        other.x >= self.x && other.y >= self.y && other.right() <= self.right() && other.bottom() <= self.bottom()
    }

    fn overlaps(&self, other: &Rect) -> bool {
        self.x < other.right() && other.x < self.right() && self.y < other.bottom() && other.y < self.bottom()
    }
}

/// One page being packed: the free space as maximal rectangles.
struct Bin {
    free: Vec<Rect>,
}

impl Bin {
    fn new(width: u32, height: u32) -> Bin {
        Bin { free: vec![Rect { x: 0, y: 0, width, height }] }
    }

    /// Places a `width` x `height` rectangle where it leaves the least on
    /// its shorter side, if it fits.
    fn insert(&mut self, width: u32, height: u32) -> Option<Rect> {
        let place = self
            .free
            .iter()
            .filter(|free| free.width >= width && free.height >= height)
            .min_by_key(|free| {
                let (across, down) = (free.width - width, free.height - height);
                (across.min(down), across.max(down), free.y, free.x)
            })
            .map(|free| Rect { x: free.x, y: free.y, width, height })?;
        let mut free = Vec::with_capacity(self.free.len() + 4);
        for space in self.free.drain(..) {
            if !space.overlaps(&place) {
                free.push(space);
                continue;
            }
            // What is left of it on each side of the placed rectangle.
            if place.x > space.x {
                free.push(Rect { width: place.x - space.x, ..space });
            }
            if place.right() < space.right() {
                free.push(Rect { x: place.right(), width: space.right() - place.right(), ..space });
            }
            if place.y > space.y {
                free.push(Rect { height: place.y - space.y, ..space });
            }
            if place.bottom() < space.bottom() {
                free.push(Rect { y: place.bottom(), height: space.bottom() - place.bottom(), ..space });
            }
        }
        // Only maximal rectangles stay.
        let mut index = 0;
        while index < free.len() {
            let inside = free.iter().enumerate().any(|(other, space)| {
                other != index && space.contains(&free[index]) && (space != &free[index] || other < index)
            });
            if inside {
                free.swap_remove(index);
            } else {
                index += 1;
            }
        }
        free.sort_by_key(|space| (space.y, space.x, space.width, space.height));
        self.free = free;
        Some(place)
    }
}

/// Packed pages: each page's size, and each rectangle's page and place.
#[derive(Debug)]
pub struct Packing {
    pub pages: Vec<(u32, u32)>,
    pub places: Vec<(usize, Rect)>,
}

/// Packs rectangles of `sizes` into as few pages of at most `max` a side as
/// it can, each page as small as holds what it got: power-of-two sides, or
/// with `tight` cut to what it holds, rounded up to 4. Largest first, so the
/// same sizes always pack the same way.
pub fn pack(sizes: &[(u32, u32)], max: u32, tight: bool) -> Result<Packing> {
    for (index, &(width, height)) in sizes.iter().enumerate() {
        ensure!(width > 0 && height > 0, "rectangle {index} is empty");
        ensure!(width <= max && height <= max, "a {width} x {height} image does not fit in a page of {max} a side");
    }
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&index| {
        let (width, height) = sizes[index];
        (std::cmp::Reverse(width.max(height)), std::cmp::Reverse(u64::from(width) * u64::from(height)), index)
    });
    let mut packing =
        Packing { pages: Vec::new(), places: vec![(0, Rect { x: 0, y: 0, width: 0, height: 0 }); sizes.len()] };
    while !order.is_empty() {
        let page = packing.pages.len();
        // The smallest power-of-two page that holds all that is left, else
        // as much as the largest holds.
        let area: u64 = order.iter().map(|&index| u64::from(sizes[index].0) * u64::from(sizes[index].1)).sum();
        let mut candidates: Vec<(u32, u32)> = Vec::new();
        let mut width = 1;
        while width <= max {
            let mut height = 1;
            while height <= max {
                if u64::from(width) * u64::from(height) >= area {
                    candidates.push((width, height));
                }
                height *= 2;
            }
            width *= 2;
        }
        let largest = (max, max);
        candidates
            .sort_by_key(|&(width, height)| (u64::from(width) * u64::from(height), width.abs_diff(height), height));
        let fitted = candidates.into_iter().chain([largest]).find_map(|(width, height)| {
            let mut bin = Bin::new(width, height);
            let placed: Option<Vec<Rect>> =
                order.iter().map(|&index| bin.insert(sizes[index].0, sizes[index].1)).collect();
            placed.map(|placed| ((width, height), placed))
        });
        let (mut size, placed) = match fitted {
            Some(fitted) => (fitted.0, order.drain(..).zip(fitted.1).collect::<Vec<_>>()),
            None => {
                let mut bin = Bin::new(max, max);
                let mut placed = Vec::new();
                order.retain(|&index| match bin.insert(sizes[index].0, sizes[index].1) {
                    Some(rect) => {
                        placed.push((index, rect));
                        false
                    }
                    None => true,
                });
                ((max, max), placed)
            }
        };
        if tight {
            let right = placed.iter().map(|(_, rect)| rect.right()).max().unwrap_or(1);
            let bottom = placed.iter().map(|(_, rect)| rect.bottom()).max().unwrap_or(1);
            size =
                (right.next_multiple_of(4).min(size.0.max(right)), bottom.next_multiple_of(4).min(size.1.max(bottom)));
        }
        for (index, rect) in placed {
            packing.places[index] = (page, rect);
        }
        packing.pages.push(size);
    }
    Ok(packing)
}

/// An RGBA image.
struct Image {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl Image {
    fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let at = ((y * self.width + x) * 4) as usize;
        self.pixels[at..at + 4].try_into().unwrap()
    }

    fn set(&mut self, x: u32, y: u32, pixel: [u8; 4]) {
        let at = ((y * self.width + x) * 4) as usize;
        self.pixels[at..at + 4].copy_from_slice(&pixel);
    }

    /// The smallest rectangle holding every pixel that is not fully
    /// transparent; a fully transparent image keeps one pixel.
    fn opaque_bounds(&self) -> Rect {
        let (mut left, mut top, mut right, mut bottom) = (u32::MAX, u32::MAX, 0, 0);
        for y in 0..self.height {
            for x in 0..self.width {
                if self.pixel(x, y)[3] != 0 {
                    (left, top, right, bottom) = (left.min(x), top.min(y), right.max(x + 1), bottom.max(y + 1));
                }
            }
        }
        if left == u32::MAX {
            return Rect { x: 0, y: 0, width: 1, height: 1 };
        }
        Rect { x: left, y: top, width: right - left, height: bottom - top }
    }
}

/// What the atlas step was asked for.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Options {
    /// The pages' and map's path, without extension, from the checkout.
    out: String,
    /// A Blot module of the sprites' frames to write, from the checkout.
    #[serde(default)]
    blot: Option<String>,
    #[serde(default = "default_max")]
    max_size: u32,
    #[serde(default = "default_extrude")]
    extrude: u32,
    #[serde(default = "default_spacing")]
    spacing: u32,
    #[serde(default = "default_trim")]
    trim: bool,
    #[serde(default)]
    power_of_two: bool,
}

fn default_max() -> u32 {
    2048
}

fn default_extrude() -> u32 {
    1
}

fn default_spacing() -> u32 {
    1
}

fn default_trim() -> bool {
    true
}

/// The atlas's map, `NAME.atlas.json`.
#[derive(Debug, Deserialize, Serialize)]
pub struct Map {
    pub pages: Vec<Page>,
    pub sprites: Vec<Frame>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Page {
    /// Its asset key: its path from the map's directory.
    pub image: String,
    pub width: u32,
    pub height: u32,
}

/// Where one sprite is.
#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Frame {
    /// Its image's file name without extension, made a name: letters,
    /// digits and underscores.
    pub name: String,
    /// Its image, from the checkout.
    pub source: String,
    pub page: usize,
    /// Its trimmed pixels in the page.
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// The same as texture coordinates, at the pixels' outer edges.
    pub u0: f64,
    pub v0: f64,
    pub u1: f64,
    pub v1: f64,
    /// Where the trimmed pixels were in the image, and its whole size.
    pub trim_x: u32,
    pub trim_y: u32,
    pub source_width: u32,
    pub source_height: u32,
}

/// A file name made a Blot name, as the catalog's are: letters, digits and
/// underscores, not starting with a digit.
fn name_of(path: &str) -> String {
    let file = path.rsplit('/').next().unwrap_or(path);
    let stem = file.split('.').next().unwrap_or(file);
    let name: String =
        stem.chars().map(|character| if character.is_ascii_alphanumeric() { character } else { '_' }).collect();
    if name.is_empty() || name.starts_with(|character: char| character.is_ascii_digit()) {
        format!("_{name}")
    } else {
        name
    }
}

fn read_image(path: &Path) -> Result<Image> {
    let image = image::open(path).with_context(|| format!("reading {}", path.display()))?.into_rgba8();
    Ok(Image { width: image.width(), height: image.height(), pixels: image.into_raw() })
}

/// Packs `images` (each named by its path) into pages as `options` say,
/// returning the pages and the map, whose pages are named after `stem`.
fn build(images: Vec<(String, Image)>, options: &Options, stem: &str) -> Result<(Vec<Image>, Map)> {
    let mut names = BTreeMap::new();
    for (path, _) in &images {
        if let Some(other) = names.insert(name_of(path), path) {
            bail!("{other} and {path} would both be called {}", name_of(path));
        }
    }
    let bounds: Vec<Rect> = images
        .iter()
        .map(|(_, image)| {
            if options.trim {
                image.opaque_bounds()
            } else {
                Rect { x: 0, y: 0, width: image.width, height: image.height }
            }
        })
        .collect();
    let border = options.extrude * 2 + options.spacing;
    let sizes: Vec<(u32, u32)> = bounds.iter().map(|rect| (rect.width + border, rect.height + border)).collect();
    let packing = pack(&sizes, options.max_size, !options.power_of_two)?;
    let mut pages: Vec<Image> = packing
        .pages
        .iter()
        .map(|&(width, height)| Image { width, height, pixels: vec![0; (width * height * 4) as usize] })
        .collect();
    let mut sprites = Vec::new();
    for (((path, image), bounds), (page, place)) in images.iter().zip(&bounds).zip(&packing.places) {
        let target = &mut pages[*page];
        let extrude = options.extrude;
        let (left, top) = (place.x + extrude, place.y + extrude);
        // Each pixel of the cell, from the nearest of the image's pixels:
        // the image itself, and its edges repeated outwards.
        for y in 0..bounds.height + extrude * 2 {
            for x in 0..bounds.width + extrude * 2 {
                let from_x = x.saturating_sub(extrude).min(bounds.width - 1) + bounds.x;
                let from_y = y.saturating_sub(extrude).min(bounds.height - 1) + bounds.y;
                target.set(place.x + x, place.y + y, image.pixel(from_x, from_y));
            }
        }
        let (width, height) = (f64::from(target.width), f64::from(target.height));
        sprites.push(Frame {
            name: name_of(path),
            source: path.clone(),
            page: *page,
            x: left,
            y: top,
            width: bounds.width,
            height: bounds.height,
            u0: f64::from(left) / width,
            v0: f64::from(top) / height,
            u1: f64::from(left + bounds.width) / width,
            v1: f64::from(top + bounds.height) / height,
            trim_x: bounds.x,
            trim_y: bounds.y,
            source_width: image.width,
            source_height: image.height,
        });
    }
    let file = stem.rsplit('/').next().unwrap_or(stem);
    let map = Map {
        pages: pages
            .iter()
            .enumerate()
            .map(|(index, page)| Page {
                image: if index == 0 { format!("{file}.png") } else { format!("{file}_{index}.png") },
                width: page.width,
                height: page.height,
            })
            .collect(),
        sprites,
    };
    Ok((pages, map))
}

fn png(image: &Image) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    image::ImageEncoder::write_image(
        image::codecs::png::PngEncoder::new(&mut bytes),
        &image.pixels,
        image.width,
        image.height,
        image::ExtendedColorType::Rgba8,
    )?;
    Ok(bytes)
}

/// A float as Blot reads it: always with a point, a negative one in brackets.
fn blot_number(value: f64) -> String {
    let text = format!("{value:.6}");
    if value < 0.0 { format!("({text})") } else { text }
}

/// The Blot module of the map's frames: one constant per sprite.
fn blot_module(map: &Map, step: &str, out: &str) -> String {
    let mut text = format!(
        "// Generated by pen's atlas step ({step} in pipeline.json); do not edit.\n\
         // Each sprite's place in its page of {out}.png (page 1 is {out}_1.png,\n\
         // and so on): texture coordinates for ink's `ui.image`, and its size in\n\
         // pixels, trimmed, with where its pixels were in the whole image.\n\n\
         type Frame is data = $Frame {{ page: U32, u0: F32, v0: F32, u1: F32, v1: F32, width: F32, height: F32, \
         left: F32, top: F32, full_width: F32, full_height: F32 }}\n\n"
    );
    for frame in &map.sprites {
        text += &format!(
            "const {} = $Frame {{ page: {}, u0: {}, v0: {}, u1: {}, v1: {}, width: {}, height: {}, left: {}, top: {}, \
             full_width: {}, full_height: {} }}\n",
            frame.name,
            frame.page,
            blot_number(frame.u0),
            blot_number(frame.v0),
            blot_number(frame.u1),
            blot_number(frame.v1),
            blot_number(f64::from(frame.width)),
            blot_number(f64::from(frame.height)),
            blot_number(f64::from(frame.trim_x)),
            blot_number(f64::from(frame.trim_y)),
            blot_number(f64::from(frame.source_width)),
            blot_number(f64::from(frame.source_height)),
        );
    }
    text
}

/// The `atlas` step.
pub fn step(context: &StepContext) -> Result<()> {
    let options: Options = serde_json::from_value(context.step.options.clone()).context("the atlas step's options")?;
    ensure!(options.max_size > 0 && options.max_size <= 16384, "maxSize must be 1 to 16384");
    let mut written = vec![format!("{}.png", options.out), format!("{}.atlas.json", options.out)];
    written.extend(options.blot.clone());
    for path in &written {
        ensure!(
            context.step.outputs.iter().any(|glob| matches(glob, path)),
            "it writes {path}, which its outputs do not name"
        );
    }
    let mut images = Vec::new();
    for source in &context.sources {
        images.push((source.clone(), read_image(&context.root.join(source))?));
    }
    ensure!(!images.is_empty(), "its sources name no images");
    let (pages, map) = build(images, &options, &options.out)?;
    let directory = Path::new(&options.out).parent().unwrap_or(Path::new(""));
    for (page, image) in map.pages.iter().zip(&pages) {
        let bytes = png(image)?;
        write_atomically(&context.root.join(directory).join(&page.image), |file| Ok(file.write_all(&bytes)?))?;
    }
    let json = serde_json::to_vec_pretty(&map)?;
    write_atomically(&context.root.join(format!("{}.atlas.json", options.out)), |file| Ok(file.write_all(&json)?))?;
    if let Some(blot) = &options.blot {
        let text = blot_module(&map, &context.step.recipe, &options.out);
        write_atomically(&context.root.join(blot), |file| Ok(file.write_all(text.as_bytes())?))?;
    }
    let area: u64 = map.pages.iter().map(|page| u64::from(page.width) * u64::from(page.height)).sum();
    let used: u64 = map.sprites.iter().map(|frame| u64::from(frame.width) * u64::from(frame.height)).sum();
    eprintln!(
        "atlas: {} sprites in {} page{} ({}), {:.0}% of it theirs",
        map.sprites.len(),
        map.pages.len(),
        if map.pages.len() == 1 { "" } else { "s" },
        map.pages.iter().map(|page| format!("{} x {}", page.width, page.height)).collect::<Vec<_>>().join(", "),
        used as f64 * 100.0 / area.max(1) as f64,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> Options {
        serde_json::from_value(serde_json::json!({ "out": "src/assets/ui" })).unwrap()
    }

    /// A `width` x `height` image of one color, with a transparent border
    /// `margin` wide.
    fn image(width: u32, height: u32, margin: u32, color: [u8; 4]) -> Image {
        let mut image = Image { width, height, pixels: vec![0; (width * height * 4) as usize] };
        for y in margin..height - margin {
            for x in margin..width - margin {
                image.set(x, y, color);
            }
        }
        image
    }

    #[test]
    fn rectangles_pack_without_overlapping_into_small_pages() {
        // Deterministic sizes of every shape.
        let mut seed = 7u32;
        let mut next = |most: u32| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 16) % most + 1
        };
        let sizes: Vec<(u32, u32)> = (0..120).map(|_| (next(60), next(60))).collect();
        let packing = pack(&sizes, 512, false).unwrap();
        assert_eq!(packing.pages.len(), 1);
        let (width, height) = packing.pages[0];
        assert!(width.is_power_of_two() && height.is_power_of_two() && width * height <= 512 * 512);
        let area: u32 = sizes.iter().map(|(width, height)| width * height).sum();
        assert!(area as f64 / f64::from(width * height) > 0.6, "{area} in {width} x {height}");
        for (index, (_, rect)) in packing.places.iter().enumerate() {
            assert_eq!((rect.width, rect.height), sizes[index]);
            assert!(rect.right() <= width && rect.bottom() <= height);
            for (_, other) in &packing.places[index + 1..] {
                assert!(!rect.overlaps(other), "{rect:?} {other:?}");
            }
        }
        // The same sizes pack the same way; too many for a page fill more.
        assert_eq!(pack(&sizes, 512, false).unwrap().places, packing.places);
        let packing = pack(&sizes, 128, true).unwrap();
        assert!(packing.pages.len() > 1);
        assert!(packing.pages.iter().all(|&(width, height)| width <= 128 && height <= 128));
        for page in 0..packing.pages.len() {
            let on: Vec<&Rect> = packing.places.iter().filter(|(at, _)| *at == page).map(|(_, rect)| rect).collect();
            assert!(!on.is_empty());
            for (index, rect) in on.iter().enumerate() {
                assert!(on[index + 1..].iter().all(|other| !rect.overlaps(other)));
            }
        }
        assert!(pack(&[(129, 1)], 128, true).is_err());
    }

    #[test]
    fn images_are_trimmed_extruded_and_mapped() {
        let red = [255, 0, 0, 255];
        let green = [0, 255, 0, 255];
        let images = vec![
            ("art/ui/play button.png".to_owned(), image(12, 10, 2, red)),
            ("art/ui/2x.png".to_owned(), image(4, 4, 0, green)),
            ("art/ui/clear.png".to_owned(), image(3, 3, 0, [0; 4])),
        ];
        let (pages, map) = build(images, &options(), "src/assets/ui").unwrap();
        assert_eq!(pages.len(), 1);
        assert_eq!(map.pages[0].image, "ui.png");
        let names: Vec<&str> = map.sprites.iter().map(|frame| frame.name.as_str()).collect();
        assert_eq!(names, ["play_button", "_2x", "clear"]);
        let play = &map.sprites[0];
        // Trimmed to its 8 x 6 opaque middle, which was 2 pixels in.
        assert_eq!((play.width, play.height, play.trim_x, play.trim_y), (8, 6, 2, 2));
        assert_eq!((play.source_width, play.source_height), (12, 10));
        let page = &pages[0];
        assert_eq!(play.u0, f64::from(play.x) / f64::from(page.width));
        assert_eq!(play.v1, f64::from(play.y + play.height) / f64::from(page.height));
        for y in play.y..play.y + play.height {
            for x in play.x..play.x + play.width {
                assert_eq!(page.pixel(x, y), red);
            }
        }
        // A pixel of its edge repeated round it, corners too.
        for (x, y) in [(play.x - 1, play.y), (play.x + play.width, play.y + 2), (play.x - 1, play.y - 1)] {
            assert_eq!(page.pixel(x, y), red, "{x} {y}");
        }
        // A fully transparent image keeps one pixel.
        assert_eq!((map.sprites[2].width, map.sprites[2].height), (1, 1));
        // Two images one name would have are refused.
        let twins = vec![("a/x.png".to_owned(), image(2, 2, 0, red)), ("b/x.png".to_owned(), image(2, 2, 0, red))];
        assert!(build(twins, &options(), "ui").is_err());
        let module = blot_module(&map, "ui-atlas", "src/assets/ui");
        assert!(module.contains("const play_button = $Frame { page: 0, u0: "), "{module}");
        assert!(module.contains("width: 8.000000, height: 6.000000, left: 2.000000"), "{module}");
    }

    /// A sprite shader that draws its texture between the sprite's
    /// coordinates, as gdev's does.
    const SPRITE: &str = r#"
struct Screen { size: vec4f };
struct Sprite { rect: vec4f, uv: vec4f, color: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> screen: Screen;
@group(0) @binding(1) var<storage, read> sprites: array<Sprite>;
@group(0) @binding(2) var image: texture_2d<f32>;
@group(0) @binding(3) var image_sampler: sampler;
struct Corner { @builtin(position) position: vec4f, @location(0) uv: vec2f };
@vertex fn vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> Corner {
  let sprite = sprites[index];
  let corner = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0))[vertex];
  let pixel = sprite.rect.xy + corner * sprite.rect.zw;
  return Corner(vec4f(pixel.x / screen.size.x * 2.0 - 1.0, 1.0 - pixel.y / screen.size.y * 2.0, 0.0, 1.0), mix(sprite.uv.xy, sprite.uv.zw, corner));
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  return textureSampleLevel(image, image_sampler, corner.uv, 0.0);
}
"#;

    /// Each sprite of an atlas `options` packs, drawn four times its size
    /// with linear filtering: the pixels of each that are not its color.
    #[allow(clippy::cast_possible_truncation, reason = "texture coordinates, as sprites carry them")]
    fn stray_pixels(options: serde_json::Value) -> Option<Vec<usize>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).ok()?;
        let root = crate::store::scratch("atlas-gpu");
        let colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 255, 255]];
        let images =
            colors.iter().enumerate().map(|(index, color)| (format!("{index}.png"), image(8, 8, 0, *color))).collect();
        let options: Options = serde_json::from_value(options).unwrap();
        let (pages, map) = build(images, &options, "atlas").unwrap();
        std::fs::write(root.join("atlas.png"), png(&pages[0]).unwrap()).unwrap();
        std::fs::write(root.join("sprite.wgsl"), SPRITE).unwrap();
        std::fs::write(
            root.join("atlas.material.json"),
            r#"{"shader": "sprite.wgsl", "texture": "atlas.png", "color": [1,1,1,1], "lighting": "unlit"}"#,
        )
        .unwrap();
        let mut renderer = pen_render::Renderer::new(
            device.clone(),
            queue.clone(),
            pen_render::RendererOptions {
                display: pen_render::Display::SDR,
                format: wgpu::TextureFormat::Rgba8Unorm,
                assets: root.clone(),
                veil: None,
                pipeline_cache: None,
            },
        )
        .unwrap();
        let catalog = pen_protocol::Catalog { materials: vec!["atlas.material.json".into()], ..Default::default() };
        // Each 32 x 32 at x = 40 * its index, with a gap between them.
        let sprites: Vec<pen_protocol::Sprite> = map
            .sprites
            .iter()
            .enumerate()
            .map(|(index, frame)| pen_protocol::Sprite {
                x: 40.0 * index as f32,
                y: 0.0,
                width: 32.0,
                height: 32.0,
                u0: frame.u0 as f32,
                v0: frame.v0 as f32,
                u1: frame.u1 as f32,
                v1: frame.v1 as f32,
                color: [1.0; 4],
                material: 0,
                parameters: [0.0; 4],
            })
            .collect();
        let (width, height) = (192u32, 32u32);
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let started = std::time::Instant::now();
        while !renderer.prepare_sprites(&sprites, &catalog).unwrap() {
            assert!(started.elapsed() < std::time::Duration::from_secs(30), "the atlas loads");
            renderer.wait(std::time::Duration::from_millis(50));
        }
        renderer.overlay(&sprites, &catalog, &target.create_view(&Default::default()), (width, height)).unwrap();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(width * height * 4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width * 4), rows_per_image: None },
            },
            target.size(),
        );
        queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let pixels = buffer.slice(..).get_mapped_range().unwrap().to_vec();
        std::fs::remove_dir_all(root).unwrap();
        let mut stray = vec![0; map.sprites.len()];
        for (index, frame) in map.sprites.iter().enumerate() {
            let color = colors[frame.source.trim_end_matches(".png").parse::<usize>().unwrap()];
            for y in 0..32 {
                for x in 40 * index..40 * index + 32 {
                    let at = (y * width as usize + x) * 4;
                    if pixels[at..at + 4].iter().zip(color).any(|(got, want)| got.abs_diff(want) > 2) {
                        stray[index] += 1;
                    }
                }
            }
        }
        Some(stray)
    }

    #[test]
    fn sprites_drawn_from_an_atlas_keep_to_their_own_pixels() {
        let Some(extruded) = stray_pixels(serde_json::json!({ "out": "atlas" })) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert_eq!(extruded, [0, 0, 0, 0], "with their edges repeated, sprites show only their own color");
        // Packed edge to edge, filtering reaches into the neighbours (or the
        // empty page) at the edges: what extruding is for.
        let bare = stray_pixels(serde_json::json!({ "out": "atlas", "extrude": 0, "spacing": 0 })).unwrap();
        assert!(bare.iter().all(|&count| count > 0), "{bare:?}");
    }

    #[test]
    fn the_step_writes_only_what_its_outputs_name() {
        let root = crate::store::scratch("atlas-step");
        std::fs::create_dir_all(root.join("art")).unwrap();
        for (name, color) in [("a", [255, 0, 0, 255]), ("b", [0, 0, 255, 255])] {
            let image = image(6, 6, 1, color);
            std::fs::write(root.join(format!("art/{name}.png")), png(&image).unwrap()).unwrap();
        }
        let mut graph_step: crate::asset_graph::Step = serde_json::from_value(serde_json::json!({
            "recipe": "ui-atlas", "builtin": "atlas", "sources": ["art/*.png"],
            "outputs": ["out/ui*"], "options": { "out": "out/ui", "blot": "src/ui.blot" }
        }))
        .unwrap();
        let sources = vec!["art/a.png".to_owned(), "art/b.png".to_owned()];
        let refused = step(&StepContext { root: &root, step: &graph_step, sources: sources.clone() }).unwrap_err();
        assert!(format!("{refused:#}").contains("src/ui.blot"), "{refused:#}");
        graph_step.outputs.push("src/ui.blot".into());
        step(&StepContext { root: &root, step: &graph_step, sources }).unwrap();
        let map: Map = serde_json::from_slice(&std::fs::read(root.join("out/ui.atlas.json")).unwrap()).unwrap();
        assert_eq!(map.sprites.len(), 2);
        let page = image::open(root.join("out/ui.png")).unwrap().into_rgba8();
        assert_eq!((page.width(), page.height()), (map.pages[0].width, map.pages[0].height));
        assert!(std::fs::read_to_string(root.join("src/ui.blot")).unwrap().contains("const b = $Frame"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
