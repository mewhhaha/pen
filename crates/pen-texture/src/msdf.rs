// Multi-channel signed distance field (MSDF) font atlases, after Chlumsky's
// "Shape Decomposition for Multi-channel Distance Fields" (2015) and his
// msdfgen. A one-channel distance field rounds a glyph's sharp corners when
// magnified; here each glyph's edges are coloured so the two edges meeting at
// a corner share only one of three channels, each channel holds the distance
// to its own edges, and the median of the three keeps corners sharp at any
// scale. The alpha channel holds the plain distance (an MTSDF) for effects
// that want a smooth field, such as outlines and shadows.
//
// `atlas` lays a font's glyphs out as gdev's models/font/font.py does its
// one-channel field: cells padded by the field's range plus a pixel, packed
// in shelves tallest first into an atlas a fixed width wide and a power of
// two high, with each glyph's place and metrics in ems. `generate` is the
// font-independent core: the field of any outline in pixels.
mod coloring;
mod correction;
mod field;
mod geometry;
#[cfg(test)]
mod tests;

use std::num::NonZero;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, anyhow, bail, ensure};
use skrifa::instance::Size;
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::{FontRef, GlyphId, MetadataProvider, Tag};

pub use geometry::{Point, Segment};

/// Glyph cells are made on at most this many threads.
const MAX_THREADS: usize = 4;

/// What `atlas` makes: which glyphs, drawn how large, with how wide a
/// field, in how wide an atlas.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// Pixels per em the glyphs are drawn at.
    pub size: f32,
    /// Distance in pixels either side of an edge the field holds (the "pixel
    /// range" or spread).
    pub range: f32,
    /// The atlas's width in pixels; its height is the power of two that fits.
    pub width: u32,
    /// Variable-font axis values, e.g. ([b'w', b'g', b'h', b't'], 600.0).
    pub variations: Vec<([u8; 4], f32)>,
    /// The characters to put in the atlas; ones the font lacks are left out.
    pub chars: Vec<char>,
}

/// A glyph's place in the atlas and its metrics, in ems: its box from the
/// pen at the baseline (`top` negative above it), padding included, and how
/// far the pen moves after it. A glyph with no ink (a space) has an empty
/// box and no place.
#[derive(Clone, Debug, PartialEq)]
pub struct Glyph {
    pub code: char,
    pub u0: f32,
    pub v0: f32,
    pub u1: f32,
    pub v1: f32,
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
    pub advance: f32,
}

/// A font's glyphs' fields packed in one RGBA8 texture, where each is, and
/// the font's vertical metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct Atlas {
    pub width: u32,
    pub height: u32,
    /// RGBA8 rows, top first: RGB the multi-channel field, A the true
    /// single-channel distance (an MTSDF), both 0.5 at the edge and above 0.5
    /// inside.
    pub pixels: Vec<u8>,
    /// The glyphs in the order of `Options::chars`.
    pub glyphs: Vec<Glyph>,
    /// In ems, as gdev's metrics are: how far the font rises above the
    /// baseline and falls below it (positive).
    pub ascent: f32,
    pub descent: f32,
    /// The field's range either side of an edge, in ems (range / size).
    pub spread: f32,
}

/// An outline in pixels, x right and y down from a cell's top-left corner:
/// closed contours, each a loop of segments, each starting where the one
/// before it ends. Filled by the non-zero rule, as fonts are, whichever way
/// its contours run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Shape {
    pub contours: Vec<Vec<Segment>>,
}

/// The field of `shape` over a `width` x `height` cell, RGBA8 rows top
/// first, texel (x, y) sampling the point (x + 0.5, y + 0.5): RGB the
/// multi-channel field, A the true distance, each encoded as
/// 0.5 + distance / (2 `range`), clamped, so 0.5 at the edge, above it inside,
/// and 0 or 1 at `range` pixels or more from it.
pub fn generate(shape: &Shape, width: u32, height: u32, range: f32) -> Vec<u8> {
    assert!(range > 0.0, "an MSDF's range must be positive");
    let (width, height) = (usize::try_from(width).unwrap(), usize::try_from(height).unwrap());
    let field = distances(shape, width, height, true);
    let scale = 1.0 / (2.0 * f64::from(range));
    field.iter().flat_map(|texel| texel.map(|distance| encode(0.5 + distance * scale))).collect()
}

/// Each texel's distances in pixels, positive inside: the three channels,
/// error corrected if `correct`, then the true distance.
fn distances(shape: &Shape, width: usize, height: usize, correct: bool) -> Vec<[f64; 4]> {
    let prepared = field::Prepared::new(shape);
    let mut field = field::distances(&prepared, width, height);
    if correct {
        correction::correct(&mut field, width, height, &prepared);
    }
    field
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped to 0..=255 first")]
fn encode(value: f64) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// A glyph's outline as a font draws it, turned into pixels from the pen at
/// the baseline, y down.
struct Outline {
    scale: f64,
    contours: Vec<Vec<Segment>>,
    contour: Vec<Segment>,
    start: Point,
    current: Point,
}

impl Outline {
    fn new(scale: f64) -> Self {
        Self { scale, contours: Vec::new(), contour: Vec::new(), start: Point::default(), current: Point::default() }
    }

    fn point(&self, x: f32, y: f32) -> Point {
        Point::new(f64::from(x) * self.scale, -f64::from(y) * self.scale)
    }

    fn push(&mut self, segment: Segment) {
        self.contour.push(segment);
        self.current = segment.end();
    }

    /// Closes the contour being drawn, with a line back to its start.
    fn finish(&mut self) {
        if self.contour.is_empty() {
            return;
        }
        if self.current != self.start {
            self.contour.push(Segment::Line(self.current, self.start));
        }
        self.contours.push(std::mem::take(&mut self.contour));
    }
}

impl OutlinePen for Outline {
    fn move_to(&mut self, x: f32, y: f32) {
        self.finish();
        self.start = self.point(x, y);
        self.current = self.start;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.push(Segment::Line(self.current, self.point(x, y)));
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.push(Segment::Quad(self.current, self.point(x1, y1), self.point(x, y)));
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.push(Segment::Cubic(self.current, self.point(x1, y1), self.point(x2, y2), self.point(x, y)));
    }

    fn close(&mut self) {
        self.finish();
    }
}

/// A glyph's cell: its outline placed in it, and where the cell sits from
/// the pen at the baseline, in pixels, y down.
struct Cell {
    shape: Shape,
    left: i64,
    top: i64,
    width: u32,
    height: u32,
}

/// A glyph as the font has it: its advance in ems, and its cell unless it
/// has no ink.
struct Entry {
    code: char,
    advance: f32,
    cell: Option<Cell>,
}

/// The font's vertical metrics in ems, and its glyphs for `options.chars`.
struct Font {
    ascent: f32,
    descent: f32,
    entries: Vec<Entry>,
}

#[allow(clippy::cast_possible_truncation, reason = "whole pixels of a glyph's bounds, far inside i64")]
fn whole(pixels: f64) -> i64 {
    pixels as i64
}

/// The cell around an outline in pixels from the pen: its bounds rounded
/// out to whole pixels, padded by `pad`. None for an outline with no ink.
fn cell(mut shape: Shape, pad: i64) -> Result<Option<Cell>> {
    let mut bounds: Option<(Point, Point)> = None;
    for segment in shape.contours.iter().flatten().filter(|segment| !segment.is_degenerate()) {
        let (min, max) = segment.bounds();
        bounds = Some(match bounds {
            None => (min, max),
            Some((low, high)) => {
                (Point::new(low.x.min(min.x), low.y.min(min.y)), Point::new(high.x.max(max.x), high.y.max(max.y)))
            }
        });
    }
    let Some((min, max)) = bounds else { return Ok(None) };
    ensure!(min.x.is_finite() && min.y.is_finite() && max.x.is_finite() && max.y.is_finite(), "outline is not finite");
    let (left, top) = (whole(min.x.floor()) - pad, whole(min.y.floor()) - pad);
    let (right, bottom) = (whole(max.x.ceil()) + pad, whole(max.y.ceil()) + pad);
    let offset = Point::new(left as f64, top as f64);
    for segment in shape.contours.iter_mut().flatten() {
        *segment = match *segment {
            Segment::Line(p0, p1) => Segment::Line(p0 - offset, p1 - offset),
            Segment::Quad(p0, p1, p2) => Segment::Quad(p0 - offset, p1 - offset, p2 - offset),
            Segment::Cubic(p0, p1, p2, p3) => Segment::Cubic(p0 - offset, p1 - offset, p2 - offset, p3 - offset),
        };
    }
    let (width, height) = (u32::try_from(right - left)?, u32::try_from(bottom - top)?);
    Ok(Some(Cell { shape, left, top, width, height }))
}

/// Reads `data`'s outlines for `options.chars` (each once), set to
/// `options.variations`.
fn font(data: &[u8], options: &Options) -> Result<Font> {
    ensure!(options.size > 0.0 && options.size.is_finite(), "MSDF glyph size must be positive");
    ensure!(options.range > 0.0 && options.range.is_finite(), "MSDF range must be positive");
    let face = FontRef::from_index(data, 0).map_err(|error| anyhow!("font could not be read: {error}"))?;
    let axes = face.axes();
    for (tag, _) in &options.variations {
        let tag = Tag::new(tag);
        ensure!(axes.iter().any(|axis| axis.tag() == tag), "font has no '{tag}' axis");
    }
    let location = axes.location(options.variations.iter().map(|&(tag, value)| (Tag::new(&tag), value)));
    // Font units: the outlines are scaled here, in f64.
    let metrics = face.metrics(Size::unscaled(), &location);
    let advances = face.glyph_metrics(Size::unscaled(), &location);
    let (charmap, outlines) = (face.charmap(), face.outline_glyphs());
    let units = f32::from(metrics.units_per_em);
    ensure!(units > 0.0, "font has no units per em");
    let scale = f64::from(options.size) / f64::from(units);
    let pad = whole(f64::from(options.range).ceil()) + 1;
    let mut entries = Vec::new();
    for &code in &options.chars {
        if entries.iter().any(|entry: &Entry| entry.code == code) {
            continue;
        }
        let Some(id) = charmap.map(code).filter(|&id| id != GlyphId::NOTDEF) else { continue };
        let advance = advances.advance_width(id).unwrap_or(0.0) / units;
        let mut outline = Outline::new(scale);
        let drawn = outlines
            .get(id)
            .map(|glyph| glyph.draw(DrawSettings::unhinted(Size::unscaled(), &location), &mut outline))
            .transpose()
            .map_err(|error| anyhow!("glyph {code:?}: {error}"))?;
        let cell = if drawn.is_some() {
            outline.finish();
            cell(Shape { contours: outline.contours }, pad).with_context(|| format!("glyph {code:?}"))?
        } else {
            None
        };
        entries.push(Entry { code, advance, cell });
    }
    Ok(Font { ascent: metrics.ascent / units, descent: -metrics.descent / units, entries })
}

/// `work` done on each of `items` on up to `MAX_THREADS` threads, the
/// results in the items' order.
fn in_parallel<T: Sync, R: Send>(items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(1, NonZero::get).min(MAX_THREADS).min(items.len());
    if threads <= 1 {
        return items.iter().map(work).collect();
    }
    let next = AtomicUsize::new(0);
    let mut results: Vec<Option<R>> = std::iter::repeat_with(|| None).take(items.len()).collect();
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(i) else { break };
                        done.push((i, work(item)));
                    }
                    done
                })
            })
            .collect();
        for worker in workers {
            let done = worker.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            for (i, result) in done {
                results[i] = Some(result);
            }
        }
    });
    results.into_iter().map(|result| result.expect("every item is worked")).collect()
}

/// Places cells of `sizes` (width, height) in shelves `width` wide, tallest
/// first and otherwise in order, as font.py does: each cell's top-left corner,
/// and the power of two high that holds them all.
fn pack(sizes: &[(u32, u32)], width: u32) -> Result<(Vec<(u32, u32)>, u32)> {
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(sizes[i].1));
    let mut places = vec![(0, 0); sizes.len()];
    let (mut x, mut y, mut shelf) = (0u32, 0u32, 0u32);
    for i in order {
        let (cell_width, cell_height) = sizes[i];
        ensure!(cell_width <= width, "a {cell_width} pixel wide cell does not fit a {width} pixel wide atlas");
        if cell_width > width - x {
            (x, y, shelf) = (0, y.checked_add(shelf).context("MSDF atlas is too tall")?, 0);
        }
        places[i] = (x, y);
        x += cell_width;
        shelf = shelf.max(cell_height);
    }
    let used = y.checked_add(shelf).context("MSDF atlas is too tall")?;
    Ok((places, used.max(1).checked_next_power_of_two().context("MSDF atlas is too tall")?))
}

/// An atlas of the glyphs of the font in `data` for `options.chars`: each
/// glyph's field in a cell padded by the range plus a pixel, the cells packed
/// in shelves tallest first, and each glyph's place and metrics.
pub fn atlas(data: &[u8], options: &Options) -> Result<Atlas> {
    ensure!(options.width > 0, "MSDF atlas width must be positive");
    let font = font(data, options)?;
    for entry in &font.entries {
        if let Some(cell) = entry.cell.as_ref().filter(|cell| cell.width > options.width) {
            bail!("glyph {:?} is {} pixels wide, wider than the atlas's {}", entry.code, cell.width, options.width);
        }
    }
    let cells: Vec<&Cell> = font.entries.iter().filter_map(|entry| entry.cell.as_ref()).collect();
    let fields = in_parallel(&cells, |cell| generate(&cell.shape, cell.width, cell.height, options.range));
    let sizes: Vec<(u32, u32)> = cells.iter().map(|cell| (cell.width, cell.height)).collect();
    let (places, height) = pack(&sizes, options.width)?;

    let row = usize::try_from(options.width)? * 4;
    let mut pixels = vec![0u8; row * usize::try_from(height)?];
    for ((cell, field), &(x, y)) in cells.iter().zip(&fields).zip(&places) {
        let cell_row = usize::try_from(cell.width)? * 4;
        for (r, line) in field.chunks_exact(cell_row).enumerate() {
            let at = (usize::try_from(y)? + r) * row + usize::try_from(x)? * 4;
            pixels[at..at + cell_row].copy_from_slice(line);
        }
    }

    let (atlas_width, atlas_height, size) = (options.width as f32, height as f32, options.size);
    let mut placed = places.iter();
    let glyphs = font
        .entries
        .iter()
        .map(|entry| {
            let (code, advance) = (entry.code, entry.advance);
            let Some(cell) = &entry.cell else {
                let [u0, v0, u1, v1, left, top, width, height] = [0.0; 8];
                return Glyph { code, u0, v0, u1, v1, left, top, width, height, advance };
            };
            let &(x, y) = placed.next().expect("a place for every cell");
            Glyph {
                code,
                u0: x as f32 / atlas_width,
                v0: y as f32 / atlas_height,
                u1: (x + cell.width) as f32 / atlas_width,
                v1: (y + cell.height) as f32 / atlas_height,
                left: cell.left as f32 / size,
                top: cell.top as f32 / size,
                width: cell.width as f32 / size,
                height: cell.height as f32 / size,
                advance,
            }
        })
        .collect();
    Ok(Atlas {
        width: options.width,
        height,
        pixels,
        glyphs,
        ascent: font.ascent,
        descent: font.descent,
        spread: options.range / options.size,
    })
}
