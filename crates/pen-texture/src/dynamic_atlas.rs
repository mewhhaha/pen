//! Runtime dynamic LRU MSDF glyph atlas cache wired to [`crate::shaping`]:
//! shapes arbitrary Unicode/BiDi/Arabic/CJK runs via `FontChain`, rasterizes
//! uncached `(font_id, glyph_id)` outlines into MTSDF cells on demand via
//! [`crate::msdf::generate`], pins active draw-batch cells so a single
//! `layout()` call can never overwrite its own quads, keeps whitespace out of
//! pixel slots, evicts least-recently-used cells when full, and coalesces dirty
//! pixel upload rectangles by slot.
use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Result, ensure};
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::{FontRef, GlyphId, MetadataProvider};

use crate::msdf::{Point, Segment, Shape, generate};
use crate::shaping::{Direction, FontChain, shape};

const MAX_ATLAS_DIM: u32 = 16_384;
const MAX_ATLAS_SLOTS: usize = 65_536;

/// Positioned textured quad for one shaped glyph in em units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtlasQuad {
    pub font_index: usize,
    pub glyph_id: u32,
    pub cluster: u32,
    pub rtl: bool,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub u0: f32,
    pub v0: f32,
    pub u1: f32,
    pub v1: f32,
}

/// Dirty sub-rectangle to upload into the GPU atlas texture via `write_texture`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirtyRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// Tightly packed RGBA8 rows (`width * height * 4` bytes).
    pub pixels: Vec<u8>,
}

/// Result of shaping and caching a line of text.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapedQuads {
    pub quads: Vec<AtlasQuad>,
    pub width: f32,
    pub base_rtl: bool,
    /// True if the run required more unique inked glyphs than atlas capacity
    /// (earlier quads in the batch remain valid and pinned; overflowing glyphs
    /// were skipped rather than overwriting earlier quads).
    pub overflowed: bool,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct GlyphKey {
    font_id: u64,
    glyph_id: u32,
}

#[derive(Clone, Copy, Debug)]
struct CachedSlot {
    last_used: u64,
    pinned_batch: u64,
    u0: f32,
    v0: f32,
    u1: f32,
    v1: f32,
    left: f32,
    top: f32,
    width: f32,
    height: f32,
}

/// Fixed-grid LRU dynamic MTSDF glyph atlas.
pub struct DynamicGlyphAtlas {
    cols: u32,
    rows: u32,
    cell_size: u32,
    px_per_em: f32,
    range: f32,
    clock: u64,
    batch_id: u64,
    slots: Vec<Option<GlyphKey>>,
    entries: HashMap<GlyphKey, CachedSlot>,
    no_ink: HashSet<GlyphKey>,
    dirty: BTreeMap<usize, DirtyRect>,
}

impl DynamicGlyphAtlas {
    /// Creates a validated atlas of `cols * cell_size` by `rows * cell_size` pixels.
    pub fn try_new(cols: u32, rows: u32, cell_size: u32, px_per_em: f32, range: f32) -> Result<Self> {
        ensure!(cols > 0 && rows > 0, "atlas grid dimensions must be non-zero");
        ensure!(cell_size >= 8, "atlas cell_size must be at least 8");
        ensure!(px_per_em.is_finite() && px_per_em > 0.0, "px_per_em must be finite and positive");
        ensure!(range.is_finite() && range > 0.0, "MSDF range must be finite and positive");
        let border = f64::from(range) + 1.0;
        ensure!(
            f64::from(cell_size) > 2.0 * border + 1.0,
            "cell_size ({cell_size}) must exceed 2 * (range + 1) + 1 to preserve MSDF border"
        );
        let width = cols.checked_mul(cell_size).ok_or_else(|| anyhow::anyhow!("atlas width overflow"))?;
        let height = rows.checked_mul(cell_size).ok_or_else(|| anyhow::anyhow!("atlas height overflow"))?;
        ensure!(width <= MAX_ATLAS_DIM && height <= MAX_ATLAS_DIM, "atlas exceeds {MAX_ATLAS_DIM}px dimension limit");
        let total = usize::try_from(cols)
            .ok()
            .and_then(|c| usize::try_from(rows).ok().and_then(|r| c.checked_mul(r)))
            .ok_or_else(|| anyhow::anyhow!("atlas slot count overflow"))?;
        ensure!(total <= MAX_ATLAS_SLOTS, "atlas slot count {total} exceeds limit {MAX_ATLAS_SLOTS}");
        Ok(Self {
            cols,
            rows,
            cell_size,
            px_per_em,
            range,
            clock: 0,
            batch_id: 0,
            slots: vec![None; total],
            entries: HashMap::new(),
            no_ink: HashSet::new(),
            dirty: BTreeMap::new(),
        })
    }

    /// Creates an atlas, panicking if dimensions or parameters are invalid.
    pub fn new(cols: u32, rows: u32, cell_size: u32, px_per_em: f32, range: f32) -> Self {
        Self::try_new(cols, rows, cell_size, px_per_em, range).expect("valid DynamicGlyphAtlas parameters")
    }

    pub fn width(&self) -> u32 {
        self.cols * self.cell_size
    }

    pub fn height(&self) -> u32 {
        self.rows * self.cell_size
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn cached_count(&self) -> usize {
        self.entries.len()
    }

    /// Drains coalesced pending RGBA8 cell uploads generated since the last call.
    pub fn take_dirty(&mut self) -> Vec<DirtyRect> {
        std::mem::take(&mut self.dirty).into_values().collect()
    }

    /// Shapes `text` with `chain`, rasterizing missing glyphs into the LRU cache
    /// while pinning every cell referenced by this batch so earlier quads remain valid.
    pub fn layout(&mut self, chain: &FontChain<'_>, text: &str, direction: Direction) -> ShapedQuads {
        self.batch_id = self.batch_id.wrapping_add(1).max(1);
        let batch = self.batch_id;
        let shaped = shape(chain, text, direction);
        let mut quads = Vec::with_capacity(shaped.glyphs.len());
        let mut pen_x = 0.0f32;
        let mut overflowed = false;
        for g in &shaped.glyphs {
            if let Some(font_id) = chain.font_id(g.font_index) {
                let key = GlyphKey { font_id, glyph_id: g.glyph_id };
                match self.ensure_glyph(chain, g.font_index, key, batch) {
                    GlyphLookup::Hit(entry) => {
                        quads.push(AtlasQuad {
                            font_index: g.font_index,
                            glyph_id: g.glyph_id,
                            cluster: g.cluster,
                            rtl: g.rtl,
                            x: pen_x + g.x_offset + entry.left,
                            y: g.y_offset + entry.top,
                            width: entry.width,
                            height: entry.height,
                            u0: entry.u0,
                            v0: entry.v0,
                            u1: entry.u1,
                            v1: entry.v1,
                        });
                    }
                    GlyphLookup::NoInk => {}
                    GlyphLookup::Overflow => overflowed = true,
                }
            }
            pen_x += g.x_advance;
        }
        ShapedQuads { quads, width: shaped.width, base_rtl: shaped.base_rtl, overflowed }
    }

    fn ensure_glyph(&mut self, chain: &FontChain<'_>, font_index: usize, key: GlyphKey, batch: u64) -> GlyphLookup {
        self.clock += 1;
        let now = self.clock;
        if self.no_ink.contains(&key) {
            return GlyphLookup::NoInk;
        }
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used = now;
            entry.pinned_batch = batch;
            return GlyphLookup::Hit(*entry);
        }
        // Rasterize BEFORE allocating/evicting a slot so whitespace or missing outlines
        // never evict visible glyphs!
        let Some(font) = chain.font(font_index) else {
            self.no_ink.insert(key);
            return GlyphLookup::NoInk;
        };
        let Some(rendered) = self.rasterize_outline(font, GlyphId::new(key.glyph_id)) else {
            self.no_ink.insert(key);
            return GlyphLookup::NoInk;
        };
        let Some(slot) = self.allocate_slot(batch) else {
            return GlyphLookup::Overflow;
        };
        #[allow(clippy::cast_possible_truncation, reason = "validated slot <= 65536")]
        let sx = (slot as u32 % self.cols) * self.cell_size;
        #[allow(clippy::cast_possible_truncation, reason = "validated slot <= 65536")]
        let sy = (slot as u32 / self.cols) * self.cell_size;
        let aw = self.width() as f32;
        let ah = self.height() as f32;
        let entry = CachedSlot {
            last_used: now,
            pinned_batch: batch,
            u0: sx as f32 / aw,
            v0: sy as f32 / ah,
            u1: (sx + self.cell_size) as f32 / aw,
            v1: (sy + self.cell_size) as f32 / ah,
            left: rendered.left,
            top: rendered.top,
            width: rendered.width,
            height: rendered.height,
        };
        let rect = DirtyRect { x: sx, y: sy, width: self.cell_size, height: self.cell_size, pixels: rendered.pixels };
        // Coalesce pending uploads per cell slot.
        self.dirty.insert(slot, rect);
        self.slots[slot] = Some(key);
        self.entries.insert(key, entry);
        GlyphLookup::Hit(entry)
    }

    fn allocate_slot(&mut self, batch: u64) -> Option<usize> {
        if let Some(free) = self.slots.iter().position(Option::is_none) {
            return Some(free);
        }
        // Evict least-recently-used slot NOT pinned by the current draw batch.
        let mut oldest_slot = None;
        let mut oldest_time = u64::MAX;
        for (slot, occupant) in self.slots.iter().enumerate() {
            if let Some(key) = occupant
                && let Some(entry) = self.entries.get(key)
                && entry.pinned_batch != batch
                && entry.last_used < oldest_time
            {
                oldest_time = entry.last_used;
                oldest_slot = Some(slot);
            }
        }
        let slot = oldest_slot?;
        if let Some(evicted) = self.slots[slot].take() {
            self.entries.remove(&evicted);
        }
        Some(slot)
    }

    fn rasterize_outline(&self, font: &FontRef<'_>, gid: GlyphId) -> Option<RasterizedGlyph> {
        let upem = f64::from(font.metrics(Size::unscaled(), LocationRef::default()).units_per_em.max(1));
        let scale = f64::from(self.px_per_em) / upem;
        let outlines = font.outline_glyphs();
        let glyph = outlines.get(gid)?;
        let mut pen = Pen::new(scale);
        glyph.draw(DrawSettings::unhinted(Size::unscaled(), LocationRef::default()), &mut pen).ok()?;
        pen.finish();
        if pen.contours.is_empty() {
            return None;
        }
        let (mut min_x, mut min_y, mut max_x, mut max_y) =
            (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for contour in &pen.contours {
            for seg in contour {
                for p in seg_points(*seg) {
                    min_x = min_x.min(p.x);
                    min_y = min_y.min(p.y);
                    max_x = max_x.max(p.x);
                    max_y = max_y.max(p.y);
                }
            }
        }
        if !min_x.is_finite() || max_x <= min_x || max_y <= min_y {
            return None;
        }
        // Reserve a fixed pixel border around the cell for MSDF range + linear filtering
        // so large outlines shrink into the inner box without losing their sampling margin.
        let border = f64::from(self.range) + 1.0;
        let inner = (f64::from(self.cell_size) - 2.0 * border).max(1.0);
        let bw = max_x - min_x;
        let bh = max_y - min_y;
        let eff_scale = (inner / bw.max(bh)).min(1.0);
        let ox = (f64::from(self.cell_size) - bw * eff_scale) * 0.5 - min_x * eff_scale;
        let oy = (f64::from(self.cell_size) - bh * eff_scale) * 0.5 - min_y * eff_scale;

        let shape = Shape {
            contours: pen
                .contours
                .into_iter()
                .map(|c| c.into_iter().map(|s| transform_seg(s, eff_scale, ox, oy)).collect())
                .collect(),
        };
        let pixels = generate(&shape, self.cell_size, self.cell_size, self.range);
        #[allow(clippy::cast_possible_truncation, reason = "glyph metrics fit f32")]
        let inv_px = 1.0 / (self.px_per_em * eff_scale as f32);
        #[allow(clippy::cast_possible_truncation, reason = "glyph offsets fit f32")]
        Some(RasterizedGlyph {
            left: (-ox as f32) * inv_px,
            top: (-oy as f32) * inv_px,
            width: self.cell_size as f32 * inv_px,
            height: self.cell_size as f32 * inv_px,
            pixels,
        })
    }
}

enum GlyphLookup {
    Hit(CachedSlot),
    NoInk,
    Overflow,
}

struct RasterizedGlyph {
    left: f32,
    top: f32,
    width: f32,
    height: f32,
    pixels: Vec<u8>,
}

fn seg_points(seg: Segment) -> [Point; 4] {
    match seg {
        Segment::Line(a, b) => [a, b, a, b],
        Segment::Quad(a, b, c) => [a, b, c, c],
        Segment::Cubic(a, b, c, d) => [a, b, c, d],
    }
}

fn transform_seg(seg: Segment, s: f64, ox: f64, oy: f64) -> Segment {
    let tx = |p: Point| Point::new(p.x * s + ox, p.y * s + oy);
    match seg {
        Segment::Line(a, b) => Segment::Line(tx(a), tx(b)),
        Segment::Quad(a, b, c) => Segment::Quad(tx(a), tx(b), tx(c)),
        Segment::Cubic(a, b, c, d) => Segment::Cubic(tx(a), tx(b), tx(c), tx(d)),
    }
}

struct Pen {
    scale: f64,
    contours: Vec<Vec<Segment>>,
    contour: Vec<Segment>,
    start: Point,
    current: Point,
}

impl Pen {
    fn new(scale: f64) -> Self {
        Self { scale, contours: Vec::new(), contour: Vec::new(), start: Point::default(), current: Point::default() }
    }

    fn pt(&self, x: f32, y: f32) -> Point {
        Point::new(f64::from(x) * self.scale, -f64::from(y) * self.scale)
    }

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

impl OutlinePen for Pen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.finish();
        self.start = self.pt(x, y);
        self.current = self.start;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.pt(x, y);
        self.contour.push(Segment::Line(self.current, p));
        self.current = p;
    }

    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let c = self.pt(cx, cy);
        let p = self.pt(x, y);
        self.contour.push(Segment::Quad(self.current, c, p));
        self.current = p;
    }

    fn curve_to(&mut self, c1x: f32, c1y: f32, c2x: f32, c2y: f32, x: f32, y: f32) {
        let c1 = self.pt(c1x, c1y);
        let c2 = self.pt(c2x, c2y);
        let p = self.pt(x, y);
        self.contour.push(Segment::Cubic(self.current, c1, c2, p));
        self.current = p;
    }

    fn close(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn system_font() -> Option<Vec<u8>> {
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        ]
        .iter()
        .find_map(|p| std::fs::read(p).ok())
    }

    #[test]
    fn dynamic_atlas_validates_dimensions_and_handles_rectangular_last_slot() {
        assert!(DynamicGlyphAtlas::try_new(0, 4, 16, 12.0, 2.0).is_err());
        assert!(DynamicGlyphAtlas::try_new(4, 4, 16, f32::INFINITY, 2.0).is_err());
        assert!(DynamicGlyphAtlas::try_new(4, 4, 8, 12.0, 4.0).is_err(), "range leaves no inner box");
        assert!(DynamicGlyphAtlas::try_new(1000, 1000, 32, 16.0, 2.0).is_err(), "exceeds max dim/slots");

        let atlas = DynamicGlyphAtlas::try_new(3, 2, 16, 12.0, 2.0).unwrap();
        assert_eq!((atlas.width(), atlas.height(), atlas.capacity()), (48, 32, 6));
    }

    #[test]
    fn dynamic_atlas_pins_current_batch_skips_whitespace_eviction_and_coalesces_dirty() {
        let Some(bytes) = system_font() else {
            return;
        };
        let chain = FontChain::new(&[&bytes]).unwrap();
        // Tiny 2x1 atlas (capacity 2).
        let mut atlas = DynamicGlyphAtlas::new(2, 1, 24, 16.0, 2.5);
        // Layout "A   B" -> spaces must NOT occupy or evict either of the 2 pixel slots!
        let r1 = atlas.layout(&chain, "A   B", Direction::Ltr);
        assert!(!r1.overflowed);
        assert_eq!(r1.quads.len(), 2);
        assert_eq!(atlas.cached_count(), 2);
        // Single batch requiring 3 unique inked glyphs ("A B C") pins A and B and flags overflow
        // rather than silently overwriting A's slot before returning.
        let r2 = atlas.layout(&chain, "A B C", Direction::Ltr);
        assert!(r2.overflowed);
        assert_eq!(r2.quads.len(), 2, "only A and B emitted; A's cell stays intact");
        assert_eq!(r2.quads[0].u0, r1.quads[0].u0);
        // Next batch containing only "C" can evict LRU slot cleanly and coalesces dirty uploads.
        let r3 = atlas.layout(&chain, "C", Direction::Ltr);
        assert!(!r3.overflowed);
        assert_eq!(r3.quads.len(), 1);
        let dirty = atlas.take_dirty();
        assert!(dirty.len() <= 2, "dirty rects coalesced per slot");
    }

    #[test]
    fn dynamic_atlas_rasterizes_system_font_and_hits_cache_on_repeat() {
        let Some(bytes) = system_font() else {
            return;
        };
        let chain = FontChain::new(&[&bytes]).unwrap();
        let mut atlas = DynamicGlyphAtlas::new(3, 2, 24, 16.0, 2.5);
        let first = atlas.layout(&chain, "ABCDEF", Direction::Ltr);
        assert!(!first.overflowed);
        assert_eq!(first.quads.len(), 6);
        let dirty1 = atlas.take_dirty();
        assert_eq!(dirty1.len(), 6, "all 6 slots including rectangular last slot (2,1) filled");
        assert_eq!((dirty1[5].x, dirty1[5].y), (48, 24));
        let second = atlas.layout(&chain, "FA", Direction::Ltr);
        assert_eq!(second.quads.len(), 2);
        assert!(atlas.take_dirty().is_empty());
    }
}
