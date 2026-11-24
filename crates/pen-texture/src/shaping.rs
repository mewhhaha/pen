//! Complex text shaping: Unicode BiDi / RTL visual reordering (UAX #9 core),
//! Arabic contextual joining & mandatory Lam-Alef ligatures, OpenType `GSUB`
//! substitutions/ligatures, `GPOS` / legacy `kern` pair kerning, and ordered
//! multi-font fallback chains.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "glyph metrics and cluster indices"
)]
use std::collections::HashSet;

use anyhow::{Context, Result};
use read_fonts::tables::gpos::{PairPos, PositionLookup};
use read_fonts::tables::gsub::{SingleSubst, SubstitutionLookup};
use read_fonts::tables::kern::SubtableKind;
use read_fonts::{FontRef, TableProvider};
use skrifa::instance::{LocationRef, Size};
use skrifa::{GlyphId, GlyphId16, MetadataProvider, Tag};

/// Paragraph base direction for BiDi resolution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Direction {
    /// Detect from the first strong directional character (defaults to LTR).
    #[default]
    Auto,
    Ltr,
    Rtl,
}

/// One positioned glyph produced by [`shape`], in visual left-to-right order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapedGlyph {
    /// Index into the [`FontChain`] that supplied this glyph.
    pub font_index: usize,
    /// Glyph ID inside that font (`0` for `.notdef`).
    pub glyph_id: u32,
    /// Byte offset of the originating character cluster in the input UTF-8 string.
    pub cluster: u32,
    /// Horizontal pen advance in ems (`units / upem`).
    pub x_advance: f32,
    /// Vertical pen advance in ems.
    pub y_advance: f32,
    /// Horizontal placement offset in ems.
    pub x_offset: f32,
    /// Vertical placement offset in ems.
    pub y_offset: f32,
    /// True when this glyph belongs to a right-to-left visual run.
    pub rtl: bool,
}

/// Result of shaping one paragraph/line.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapedLine {
    pub glyphs: Vec<ShapedGlyph>,
    /// Total horizontal advance in ems.
    pub width: f32,
    /// Resolved paragraph base direction (`true` = RTL).
    pub base_rtl: bool,
}

/// Primary font followed by zero or more fallback fonts.
pub struct FontChain<'a> {
    fonts: Vec<FontRef<'a>>,
}

impl<'a> FontChain<'a> {
    pub fn new(font_bytes: &[&'a [u8]]) -> Result<Self> {
        let mut fonts = Vec::with_capacity(font_bytes.len());
        for (i, bytes) in font_bytes.iter().enumerate() {
            fonts.push(FontRef::new(bytes).with_context(|| format!("parsing font {i} in fallback chain"))?);
        }
        Ok(Self { fonts })
    }

    pub fn len(&self) -> usize {
        self.fonts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fonts.is_empty()
    }

    fn resolve_char(&self, ch: char) -> (usize, GlyphId) {
        for (idx, font) in self.fonts.iter().enumerate() {
            if let Some(gid) = font.charmap().map(ch)
                && gid.to_u32() != 0
            {
                return (idx, gid);
            }
        }
        (0, GlyphId::NOTDEF)
    }
}

// ---------------------------------------------------------------------------
// Unicode BiDi (UAX #9 core): classification, level resolution, mirroring, L2
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BidiClass {
    L,
    R,
    Al,
    En,
    An,
    Es,
    Et,
    Cs,
    Nsm,
    Ws,
    On,
    Lri,
    Rli,
    Fsi,
    Pdi,
    Lro,
    Rlo,
    Lre,
    Rle,
    Pdf,
}

fn bidi_class(ch: char) -> BidiClass {
    let cp = ch as u32;
    match cp {
        0x2066 => return BidiClass::Lri,
        0x2067 => return BidiClass::Rli,
        0x2068 => return BidiClass::Fsi,
        0x2069 => return BidiClass::Pdi,
        0x202D => return BidiClass::Lro,
        0x202E => return BidiClass::Rlo,
        0x202A => return BidiClass::Lre,
        0x202B => return BidiClass::Rle,
        0x202C => return BidiClass::Pdf,
        0x200E => return BidiClass::L,
        0x200F => return BidiClass::R,
        0x061C => return BidiClass::Al,
        _ => {}
    }
    if ch.is_ascii_digit() {
        return BidiClass::En;
    }
    if matches!(ch, '+' | '-') {
        return BidiClass::Es;
    }
    if matches!(ch, '#' | '$' | '%' | '€' | '£' | '¥') {
        return BidiClass::Et;
    }
    if matches!(ch, ',' | '.' | ':' | '/') {
        return BidiClass::Cs;
    }
    if ch.is_whitespace() {
        return BidiClass::Ws;
    }
    // Arabic-Indic and Extended Arabic-Indic digits.
    if (0x0660..=0x0669).contains(&cp) || (0x06F0..=0x06F9).contains(&cp) {
        return BidiClass::An;
    }
    // Combining marks (Latin, Hebrew, Arabic diacritics).
    if (0x0300..=0x036F).contains(&cp)
        || (0x0591..=0x05BD).contains(&cp)
        || (0x0610..=0x061A).contains(&cp)
        || (0x064B..=0x065F).contains(&cp)
        || cp == 0x0670
    {
        return BidiClass::Nsm;
    }
    // Hebrew, NKo, Samaritan, Mandaic, etc.
    if (0x0590..=0x05FF).contains(&cp) || (0x07C0..=0x085F).contains(&cp) || (0xFB1D..=0xFB4F).contains(&cp) {
        return BidiClass::R;
    }
    // Arabic, Syriac, Thaana, Arabic Presentation Forms.
    if (0x0600..=0x07BF).contains(&cp)
        || (0x0860..=0x08FF).contains(&cp)
        || (0xFB50..=0xFDFF).contains(&cp)
        || (0xFE70..=0xFEFF).contains(&cp)
    {
        return BidiClass::Al;
    }
    if ch.is_alphabetic() {
        return BidiClass::L;
    }
    BidiClass::On
}

fn mirror_bracket(ch: char) -> char {
    match ch {
        '(' => ')',
        ')' => '(',
        '[' => ']',
        ']' => '[',
        '{' => '}',
        '}' => '{',
        '<' => '>',
        '>' => '<',
        '«' => '»',
        '»' => '«',
        '‹' => '›',
        '›' => '‹',
        _ => ch,
    }
}

fn is_bidi_control(ch: char) -> bool {
    matches!(
        ch as u32,
        0x200E | 0x200F | 0x061C | 0x202A..=0x202E | 0x2066..=0x2069
    )
}

#[derive(Clone, Debug)]
struct LogicalChar {
    ch: char,
    cluster: u32,
    level: u8,
    class: BidiClass,
}

/// Resolves BiDi embedding levels and returns characters in visual LTR order.
pub fn reorder_visual(text: &str, direction: Direction) -> (Vec<(char, u32, bool)>, bool) {
    let raw: Vec<(char, u32, BidiClass)> =
        text.char_indices().map(|(idx, ch)| (ch, idx as u32, bidi_class(ch))).collect();
    if raw.is_empty() {
        return (Vec::new(), direction == Direction::Rtl);
    }
    let base_rtl = match direction {
        Direction::Ltr => false,
        Direction::Rtl => true,
        Direction::Auto => raw
            .iter()
            .find_map(|(_, _, c)| match c {
                BidiClass::L => Some(false),
                BidiClass::R | BidiClass::Al => Some(true),
                _ => None,
            })
            .unwrap_or(false),
    };
    let base_level: u8 = u8::from(base_rtl);

    // Track explicit embedding/override/isolate stack.
    let mut stack: Vec<(u8, Option<BidiClass>, bool)> = vec![(base_level, None, false)];
    let mut items: Vec<LogicalChar> = Vec::with_capacity(raw.len());
    for &(ch, cluster, orig_class) in &raw {
        let &(cur_level, cur_override, _) = stack.last().unwrap();
        match orig_class {
            BidiClass::Lre | BidiClass::Lri => {
                let next = (cur_level + 2) & !1;
                if next <= 125 {
                    stack.push((next, None, orig_class == BidiClass::Lri));
                }
            }
            BidiClass::Rle | BidiClass::Rli => {
                let next = (cur_level + 1) | 1;
                if next <= 125 {
                    stack.push((next, None, orig_class == BidiClass::Rli));
                }
            }
            BidiClass::Fsi => {
                let next = if base_rtl { (cur_level + 1) | 1 } else { (cur_level + 2) & !1 };
                if next <= 125 {
                    stack.push((next, None, true));
                }
            }
            BidiClass::Lro => {
                let next = (cur_level + 2) & !1;
                if next <= 125 {
                    stack.push((next, Some(BidiClass::L), false));
                }
            }
            BidiClass::Rlo => {
                let next = (cur_level + 1) | 1;
                if next <= 125 {
                    stack.push((next, Some(BidiClass::R), false));
                }
            }
            BidiClass::Pdf => {
                if stack.len() > 1 && !stack.last().unwrap().2 {
                    stack.pop();
                }
            }
            BidiClass::Pdi => {
                while stack.len() > 1 {
                    let (_, _, is_iso) = stack.pop().unwrap();
                    if is_iso {
                        break;
                    }
                }
            }
            _ => {
                if !is_bidi_control(ch) {
                    let class = cur_override.unwrap_or(orig_class);
                    items.push(LogicalChar { ch, cluster, level: cur_level, class });
                }
            }
        }
    }
    if items.is_empty() {
        return (Vec::new(), base_rtl);
    }

    // W1-W7: weak types (NSM inherits previous, AL->R with EN->AN, ES/CS between digits, ET adjacent to EN).
    let mut prev_strong = if base_rtl { BidiClass::R } else { BidiClass::L };
    for i in 0..items.len() {
        let c = items[i].class;
        match c {
            BidiClass::Nsm => {
                items[i].class = if i > 0 { items[i - 1].class } else { BidiClass::On };
            }
            BidiClass::Al => {
                prev_strong = BidiClass::Al;
                items[i].class = BidiClass::R;
            }
            BidiClass::L | BidiClass::R => {
                prev_strong = c;
            }
            BidiClass::En if prev_strong == BidiClass::Al => {
                items[i].class = BidiClass::An;
            }
            _ => {}
        }
    }
    for i in 1..items.len().saturating_sub(1) {
        let prev = items[i - 1].class;
        let next = items[i + 1].class;
        if items[i].class == BidiClass::Es && prev == BidiClass::En && next == BidiClass::En {
            items[i].class = BidiClass::En;
        } else if items[i].class == BidiClass::Cs {
            if prev == BidiClass::En && next == BidiClass::En {
                items[i].class = BidiClass::En;
            } else if prev == BidiClass::An && next == BidiClass::An {
                items[i].class = BidiClass::An;
            }
        }
    }
    // ET adjacent to EN -> EN.
    for i in 0..items.len() {
        if items[i].class == BidiClass::Et {
            let adj_en = (i > 0 && items[i - 1].class == BidiClass::En)
                || (i + 1 < items.len() && items[i + 1].class == BidiClass::En);
            if adj_en {
                items[i].class = BidiClass::En;
            } else {
                items[i].class = BidiClass::On;
            }
        }
    }
    // Remaining ES/CS -> ON.
    for item in &mut items {
        if matches!(item.class, BidiClass::Es | BidiClass::Cs) {
            item.class = BidiClass::On;
        }
    }

    // N1-N2: resolve neutrals (WS, ON) to surrounding strong direction, else embedding direction.
    let strong_of = |c: BidiClass| match c {
        BidiClass::L => Some(BidiClass::L),
        BidiClass::R | BidiClass::En | BidiClass::An => Some(BidiClass::R),
        _ => None,
    };
    let mut i = 0;
    while i < items.len() {
        if matches!(items[i].class, BidiClass::Ws | BidiClass::On) {
            let start = i;
            let lvl = items[i].level;
            while i < items.len() && items[i].level == lvl && matches!(items[i].class, BidiClass::Ws | BidiClass::On) {
                i += 1;
            }
            let embed_dir = if lvl % 2 == 1 { BidiClass::R } else { BidiClass::L };
            let before = if start > 0 { strong_of(items[start - 1].class).unwrap_or(embed_dir) } else { embed_dir };
            let after = if i < items.len() { strong_of(items[i].class).unwrap_or(embed_dir) } else { embed_dir };
            let resolved = if before == after { before } else { embed_dir };
            for item in &mut items[start..i] {
                item.class = resolved;
            }
        } else {
            i += 1;
        }
    }

    // I1-I2: implicit level increments.
    for item in &mut items {
        if item.level % 2 == 0 {
            match item.class {
                BidiClass::R => item.level += 1,
                BidiClass::An | BidiClass::En => item.level += 2,
                _ => {}
            }
        } else if matches!(item.class, BidiClass::L | BidiClass::En | BidiClass::An) {
            item.level += 1;
        }
    }

    // L1: trailing whitespace resets to paragraph embedding level.
    let orig_classes: Vec<BidiClass> = raw
        .iter()
        .filter(|(ch, _, _)| !is_bidi_control(*ch))
        .map(|&(_, _, orig_class)| orig_class)
        .collect();
    for (idx, &orig_class) in orig_classes.iter().enumerate().rev() {
        if idx < items.len() && orig_class == BidiClass::Ws {
            items[idx].level = base_level;
        } else {
            break;
        }
    }

    // L4: mirror paired brackets on odd (RTL) levels.
    for item in &mut items {
        if item.level % 2 == 1 {
            item.ch = mirror_bracket(item.ch);
        }
    }

    // L2: reverse contiguous runs from max level down to min odd level.
    let max_level = items.iter().map(|it| it.level).max().unwrap_or(0);
    let min_odd = items.iter().map(|it| it.level).filter(|&l| l % 2 == 1).min();
    if let Some(min_odd) = min_odd {
        for lvl in (min_odd..=max_level).rev() {
            let mut pos = 0;
            while pos < items.len() {
                if items[pos].level >= lvl {
                    let start = pos;
                    while pos < items.len() && items[pos].level >= lvl {
                        pos += 1;
                    }
                    items[start..pos].reverse();
                } else {
                    pos += 1;
                }
            }
        }
    }

    let out = items.into_iter().map(|it| (it.ch, it.cluster, it.level % 2 == 1)).collect();
    (out, base_rtl)
}

// ---------------------------------------------------------------------------
// Arabic contextual joining (Presentation Forms-B) + Lam-Alef ligatures
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum JoinType {
    None,
    Right,
    Dual,
    Transparent,
}

/// Returns `(JoinType, [isolated, final, initial, medial])` for core Arabic letters.
fn arabic_forms(ch: char) -> Option<(JoinType, [char; 4])> {
    let entry = match ch {
        '\u{0622}' => (JoinType::Right, ['\u{FE81}', '\u{FE82}', '\u{FE81}', '\u{FE82}']), // Alef Madda
        '\u{0623}' => (JoinType::Right, ['\u{FE83}', '\u{FE84}', '\u{FE83}', '\u{FE84}']), // Alef Hamza Above
        '\u{0624}' => (JoinType::Right, ['\u{FE85}', '\u{FE86}', '\u{FE85}', '\u{FE86}']), // Waw Hamza
        '\u{0625}' => (JoinType::Right, ['\u{FE87}', '\u{FE88}', '\u{FE87}', '\u{FE88}']), // Alef Hamza Below
        '\u{0626}' => (JoinType::Dual, ['\u{FE89}', '\u{FE8A}', '\u{FE8B}', '\u{FE8C}']),  // Yeh Hamza
        '\u{0627}' => (JoinType::Right, ['\u{FE8D}', '\u{FE8E}', '\u{FE8D}', '\u{FE8E}']), // Alef
        '\u{0628}' => (JoinType::Dual, ['\u{FE8F}', '\u{FE90}', '\u{FE91}', '\u{FE92}']),  // Beh
        '\u{0629}' => (JoinType::Right, ['\u{FE93}', '\u{FE94}', '\u{FE93}', '\u{FE94}']), // Teh Marbuta
        '\u{062A}' => (JoinType::Dual, ['\u{FE95}', '\u{FE96}', '\u{FE97}', '\u{FE98}']),  // Teh
        '\u{062B}' => (JoinType::Dual, ['\u{FE99}', '\u{FE9A}', '\u{FE9B}', '\u{FE9C}']),  // Theh
        '\u{062C}' => (JoinType::Dual, ['\u{FE9D}', '\u{FE9E}', '\u{FE9F}', '\u{FEA0}']),  // Jeem
        '\u{062D}' => (JoinType::Dual, ['\u{FEA1}', '\u{FEA2}', '\u{FEA3}', '\u{FEA4}']),  // Hah
        '\u{062E}' => (JoinType::Dual, ['\u{FEA5}', '\u{FEA6}', '\u{FEA7}', '\u{FEA8}']),  // Khah
        '\u{062F}' => (JoinType::Right, ['\u{FEA9}', '\u{FEAA}', '\u{FEA9}', '\u{FEAA}']), // Dal
        '\u{0630}' => (JoinType::Right, ['\u{FEAB}', '\u{FEAC}', '\u{FEAB}', '\u{FEAC}']), // Thal
        '\u{0631}' => (JoinType::Right, ['\u{FEAD}', '\u{FEAE}', '\u{FEAD}', '\u{FEAE}']), // Reh
        '\u{0632}' => (JoinType::Right, ['\u{FEAF}', '\u{FEB0}', '\u{FEAF}', '\u{FEB0}']), // Zain
        '\u{0633}' => (JoinType::Dual, ['\u{FEB1}', '\u{FEB2}', '\u{FEB3}', '\u{FEB4}']),  // Seen
        '\u{0634}' => (JoinType::Dual, ['\u{FEB5}', '\u{FEB6}', '\u{FEB7}', '\u{FEB8}']),  // Sheen
        '\u{0635}' => (JoinType::Dual, ['\u{FEB9}', '\u{FEBA}', '\u{FEBB}', '\u{FEBC}']),  // Sad
        '\u{0636}' => (JoinType::Dual, ['\u{FEBD}', '\u{FEBE}', '\u{FEBF}', '\u{FEC0}']),  // Dad
        '\u{0637}' => (JoinType::Dual, ['\u{FEC1}', '\u{FEC2}', '\u{FEC3}', '\u{FEC4}']),  // Tah
        '\u{0638}' => (JoinType::Dual, ['\u{FEC5}', '\u{FEC6}', '\u{FEC7}', '\u{FEC8}']),  // Zah
        '\u{0639}' => (JoinType::Dual, ['\u{FEC9}', '\u{FECA}', '\u{FECB}', '\u{FECC}']),  // Ain
        '\u{063A}' => (JoinType::Dual, ['\u{FECD}', '\u{FECE}', '\u{FECF}', '\u{FED0}']),  // Ghain
        '\u{0641}' => (JoinType::Dual, ['\u{FED1}', '\u{FED2}', '\u{FED3}', '\u{FED4}']),  // Feh
        '\u{0642}' => (JoinType::Dual, ['\u{FED5}', '\u{FED6}', '\u{FED7}', '\u{FED8}']),  // Qaf
        '\u{0643}' => (JoinType::Dual, ['\u{FED9}', '\u{FEDA}', '\u{FEDB}', '\u{FEDC}']),  // Kaf
        '\u{0644}' => (JoinType::Dual, ['\u{FEDD}', '\u{FEDE}', '\u{FEDF}', '\u{FEE0}']),  // Lam
        '\u{0645}' => (JoinType::Dual, ['\u{FEE1}', '\u{FEE2}', '\u{FEE3}', '\u{FEE4}']),  // Meem
        '\u{0646}' => (JoinType::Dual, ['\u{FEE5}', '\u{FEE6}', '\u{FEE7}', '\u{FEE8}']),  // Noon
        '\u{0647}' => (JoinType::Dual, ['\u{FEE9}', '\u{FEEA}', '\u{FEEB}', '\u{FEEC}']),  // Heh
        '\u{0648}' => (JoinType::Right, ['\u{FEED}', '\u{FEEE}', '\u{FEED}', '\u{FEEE}']), // Waw
        '\u{0649}' => (JoinType::Right, ['\u{FEEF}', '\u{FEF0}', '\u{FEEF}', '\u{FEF0}']), // Alef Maksura
        '\u{064A}' => (JoinType::Dual, ['\u{FEF1}', '\u{FEF2}', '\u{FEF3}', '\u{FEF4}']),  // Yeh
        _ => return None,
    };
    Some(entry)
}

fn join_type(ch: char) -> JoinType {
    if ch == '\u{0640}' {
        return JoinType::Dual; // Tatweel
    }
    if matches!(bidi_class(ch), BidiClass::Nsm) {
        return JoinType::Transparent;
    }
    if matches!(ch, '\u{FEF5}'..='\u{FEFC}') {
        return JoinType::Right; // Folded Lam-Alef ligatures connect to preceding Dual letters.
    }
    arabic_forms(ch).map_or(JoinType::None, |(jt, _)| jt)
}

/// Applies Arabic contextual joining & mandatory Lam-Alef ligatures on logical text.
pub fn shape_arabic_logical(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    if !chars.iter().any(|&c| arabic_forms(c).is_some()) {
        return text.to_owned();
    }
    // Step 1: fold mandatory Lam + Alef variants into Presentation Forms-B ligatures.
    let mut folded: Vec<char> = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\u{0644}' && i + 1 < chars.len() {
            let lig = match chars[i + 1] {
                '\u{0622}' => Some(('\u{FEF5}', '\u{FEF6}')),
                '\u{0623}' => Some(('\u{FEF7}', '\u{FEF8}')),
                '\u{0625}' => Some(('\u{FEF9}', '\u{FEFA}')),
                '\u{0627}' => Some(('\u{FEFB}', '\u{FEFC}')),
                _ => None,
            };
            if let Some((iso, fin)) = lig {
                let prev_connects = folded
                    .iter()
                    .rev()
                    .find(|&&c| join_type(c) != JoinType::Transparent)
                    .is_some_and(|&c| join_type(c) == JoinType::Dual);
                folded.push(if prev_connects { fin } else { iso });
                i += 2;
                continue;
            }
        }
        folded.push(chars[i]);
        i += 1;
    }
    // Step 2: select isolated / final / initial / medial forms.
    let mut out = String::with_capacity(folded.len() * 3);
    for idx in 0..folded.len() {
        let ch = folded[idx];
        let Some((jt, forms)) = arabic_forms(ch) else {
            out.push(ch);
            continue;
        };
        let prev_connects = folded[..idx]
            .iter()
            .rev()
            .find(|&&c| join_type(c) != JoinType::Transparent)
            .is_some_and(|&c| join_type(c) == JoinType::Dual);
        let next_connects = jt == JoinType::Dual
            && folded[idx + 1..]
                .iter()
                .find(|&&c| join_type(c) != JoinType::Transparent)
                .is_some_and(|&c| matches!(join_type(c), JoinType::Dual | JoinType::Right));
        let form = match (prev_connects, next_connects) {
            (false, false) => forms[0], // isolated
            (true, false) => forms[1],  // final
            (false, true) => forms[2],  // initial
            (true, true) => forms[3],   // medial
        };
        out.push(form);
    }
    out
}

// ---------------------------------------------------------------------------
// OpenType GSUB (SingleSubst, LigatureSubst) & GPOS / kern pair kerning
// ---------------------------------------------------------------------------

fn feature_lookups(features: &read_fonts::tables::layout::FeatureList<'_>, tags: &[Tag]) -> Vec<u16> {
    let mut seen = HashSet::new();
    let mut lookups = Vec::new();
    for record in features.feature_records() {
        if !tags.contains(&record.feature_tag()) {
            continue;
        }
        if let Ok(feature) = record.feature(features.offset_data()) {
            for idx in feature.lookup_list_indices() {
                let id = idx.get();
                if seen.insert(id) {
                    lookups.push(id);
                }
            }
        }
    }
    lookups.sort_unstable();
    lookups
}

fn apply_gsub(font: &FontRef<'_>, glyphs: &mut Vec<(GlyphId, u32)>) {
    let Ok(gsub) = font.gsub() else { return };
    let Ok(features) = gsub.feature_list() else { return };
    let Ok(lookup_list) = gsub.lookup_list() else { return };
    let active_tags = [Tag::new(b"ccmp"), Tag::new(b"rlig"), Tag::new(b"liga"), Tag::new(b"calt")];
    let lookup_indices = feature_lookups(&features, &active_tags);

    for idx in lookup_indices {
        let Ok(lookup) = lookup_list.lookups().get(usize::from(idx)) else { continue };
        match lookup {
            SubstitutionLookup::Single(table) => {
                for subtable in table.subtables().iter().flatten() {
                    for (gid, _) in glyphs.iter_mut() {
                        let Ok(g16) = GlyphId16::try_from(*gid) else { continue };
                        let mapped = match &subtable {
                            SingleSubst::Format1(f1) => f1.coverage().ok().and_then(|c| c.get(g16)).map(|_| {
                                let next = (i32::from(g16.to_u16()) + i32::from(f1.delta_glyph_id())) as u16;
                                GlyphId::from(next)
                            }),
                            SingleSubst::Format2(f2) => f2
                                .coverage()
                                .ok()
                                .and_then(|c| c.get(g16))
                                .and_then(|pos| f2.substitute_glyph_ids().get(usize::from(pos)))
                                .map(|sub| GlyphId::from(sub.get())),
                        };
                        if let Some(next) = mapped {
                            *gid = next;
                        }
                    }
                }
            }
            SubstitutionLookup::Ligature(table) => {
                for subtable in table.subtables().iter().flatten() {
                    let Ok(coverage) = subtable.coverage() else { continue };
                    let mut i = 0;
                    while i < glyphs.len() {
                        let Ok(first) = GlyphId16::try_from(glyphs[i].0) else {
                            i += 1;
                            continue;
                        };
                        let Some(cov_idx) = coverage.get(first) else {
                            i += 1;
                            continue;
                        };
                        let Ok(set) = subtable.ligature_sets().get(usize::from(cov_idx)) else {
                            i += 1;
                            continue;
                        };
                        let mut matched = None;
                        for lig in set.ligatures().iter().flatten() {
                            let comps = lig.component_glyph_ids();
                            if i + 1 + comps.len() <= glyphs.len()
                                && comps.iter().enumerate().all(|(k, c)| glyphs[i + 1 + k].0 == GlyphId::from(c.get()))
                            {
                                matched = Some((GlyphId::from(lig.ligature_glyph()), comps.len()));
                                break;
                            }
                        }
                        if let Some((lig_gid, consumed)) = matched {
                            glyphs[i].0 = lig_gid;
                            glyphs.drain(i + 1..i + 1 + consumed);
                        }
                        i += 1;
                    }
                }
            }
            _ => {}
        }
    }
}

/// Reads pair kerning in font design units between `(left, right)` from `GPOS` (`kern`)
/// or legacy TrueType `kern` Format 0.
fn pair_kern_units(font: &FontRef<'_>, left: GlyphId, right: GlyphId) -> i32 {
    let (Ok(l16), Ok(r16)) = (GlyphId16::try_from(left), GlyphId16::try_from(right)) else {
        return 0;
    };
    if let Ok(gpos) = font.gpos()
        && let (Ok(features), Ok(lookup_list)) = (gpos.feature_list(), gpos.lookup_list())
    {
        let kern_lookups = feature_lookups(&features, &[Tag::new(b"kern")]);
        let mut total = 0i32;
        let mut found = false;
        for idx in kern_lookups {
            let Ok(PositionLookup::Pair(table)) = lookup_list.lookups().get(usize::from(idx)) else { continue };
            for subtable in table.subtables().iter().flatten() {
                match subtable {
                    PairPos::Format1(f1) => {
                        let Some(cov_idx) = f1.coverage().ok().and_then(|c| c.get(l16)) else { continue };
                        let Ok(pair_set) = f1.pair_sets().get(usize::from(cov_idx)) else { continue };
                        for rec in pair_set.pair_value_records().iter().flatten() {
                            if rec.second_glyph() == r16 {
                                if let Some(dx) = rec.value_record1().x_advance() {
                                    total += i32::from(dx);
                                    found = true;
                                }
                                break;
                            }
                        }
                    }
                    PairPos::Format2(f2) => {
                        if f2.coverage().ok().and_then(|c| c.get(l16)).is_none() {
                            continue;
                        }
                        let (Ok(cd1), Ok(cd2)) = (f2.class_def1(), f2.class_def2()) else { continue };
                        let (c1, c2) = (usize::from(cd1.get(l16)), usize::from(cd2.get(r16)));
                        if let Ok(class1) = f2.class1_records().get(c1)
                            && let Ok(class2) = class1.class2_records().get(c2)
                            && let Some(dx) = class2.value_record1().x_advance()
                        {
                            total += i32::from(dx);
                            found = true;
                        }
                    }
                }
            }
        }
        if found {
            return total;
        }
    }
    // Fallback to legacy TrueType `kern` Format 0 subtables.
    if let Ok(kern) = font.kern() {
        for subtable in kern.subtables().flatten() {
            if let Ok(SubtableKind::Format0(f0)) = subtable.kind()
                && let Some(val) = f0.kerning(left, right)
            {
                return val;
            }
        }
    }
    0
}

/// Shapes `text` using `chain` with BiDi reordering, Arabic contextual joining,
/// OpenType `GSUB` ligatures/substitutions, `GPOS`/`kern` kerning, and fallback font resolution.
pub fn shape(chain: &FontChain<'_>, text: &str, direction: Direction) -> ShapedLine {
    if chain.is_empty() || text.is_empty() {
        return ShapedLine { glyphs: Vec::new(), width: 0.0, base_rtl: direction == Direction::Rtl };
    }
    let arabic_shaped = shape_arabic_logical(text);
    let (visual, base_rtl) = reorder_visual(&arabic_shaped, direction);

    // Resolve each visual character through the fallback chain, falling back to base
    // Arabic character if the font lacks Presentation Forms-B glyphs.
    let resolved: Vec<(usize, GlyphId, u32, bool)> = visual
        .into_iter()
        .map(|(ch, cluster, rtl)| {
            let (font_idx, gid) = chain.resolve_char(ch);
            (font_idx, gid, cluster, rtl)
        })
        .collect();

    let mut out = Vec::with_capacity(resolved.len());
    let mut width = 0.0f32;
    let mut cursor = 0;
    while cursor < resolved.len() {
        let font_index = resolved[cursor].0;
        let rtl = resolved[cursor].3;
        let mut end = cursor + 1;
        while end < resolved.len() && resolved[end].0 == font_index && resolved[end].3 == rtl {
            end += 1;
        }
        let font = &chain.fonts[font_index];
        let upem = f32::from(font.head().map_or(1000, |h| h.units_per_em()).max(1));
        let metrics = font.glyph_metrics(Size::unscaled(), LocationRef::default());

        // For GSUB ligatures, run in logical order within the run (reverse back for RTL).
        let mut run_glyphs: Vec<(GlyphId, u32)> = resolved[cursor..end].iter().map(|&(_, g, c, _)| (g, c)).collect();
        if rtl {
            run_glyphs.reverse();
        }
        apply_gsub(font, &mut run_glyphs);
        if rtl {
            run_glyphs.reverse();
        }

        for i in 0..run_glyphs.len() {
            let (gid, cluster) = run_glyphs[i];
            let base_adv = metrics.advance_width(gid).unwrap_or(0.0);
            let kern = if i + 1 < run_glyphs.len() {
                pair_kern_units(font, gid, run_glyphs[i + 1].0) as f32
            } else {
                0.0
            };
            let x_advance = (base_adv + kern) / upem;
            width += x_advance;
            out.push(ShapedGlyph {
                font_index,
                glyph_id: gid.to_u32(),
                cluster,
                x_advance,
                y_advance: 0.0,
                x_offset: 0.0,
                y_offset: 0.0,
                rtl,
            });
        }
        cursor = end;
    }

    ShapedLine { glyphs: out, width, base_rtl }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal valid OpenType SFNT font containing `head`, `hhea`, `maxp`,
    /// `hmtx`, `cmap` (format 4), optional `kern` (format 0), optional `GSUB` (`liga`),
    /// and optional `GPOS` (`kern` PairPos format 1) for deterministic offline testing.
    fn build_test_font(
        chars: &[(char, u16, u16)], // (char, glyph_id, advance_units)
        legacy_kern: &[(u16, u16, i16)],
        gsub_ligas: &[(u16, u16, u16)], // (first, second, ligature_gid)
        gpos_kerns: &[(u16, u16, i16)],
    ) -> Vec<u8> {
        let num_glyphs = chars
            .iter()
            .map(|&(_, g, _)| g)
            .chain(gsub_ligas.iter().map(|&(_, _, l)| l))
            .max()
            .unwrap_or(0)
            + 1;

        let mut head = vec![0u8; 54];
        head[0..4].copy_from_slice(&0x0001_0000u32.to_be_bytes());
        head[12..16].copy_from_slice(&0x5F0F_3CF5u32.to_be_bytes());
        head[18..20].copy_from_slice(&1000u16.to_be_bytes()); // upem = 1000

        let mut hhea = vec![0u8; 36];
        hhea[0..4].copy_from_slice(&0x0001_0000u32.to_be_bytes());
        hhea[4..6].copy_from_slice(&800i16.to_be_bytes());
        hhea[6..8].copy_from_slice(&(-200i16).to_be_bytes());
        hhea[34..36].copy_from_slice(&num_glyphs.to_be_bytes());

        let mut maxp = vec![0u8; 6];
        maxp[0..4].copy_from_slice(&0x0000_5000u32.to_be_bytes());
        maxp[4..6].copy_from_slice(&num_glyphs.to_be_bytes());

        let mut adv = vec![500u16; usize::from(num_glyphs)];
        for &(_, g, a) in chars {
            adv[usize::from(g)] = a;
        }
        let mut hmtx = Vec::with_capacity(usize::from(num_glyphs) * 4);
        for a in adv {
            hmtx.extend_from_slice(&a.to_be_bytes());
            hmtx.extend_from_slice(&0i16.to_be_bytes());
        }

        // cmap format 4 (one single-char segment per BMP char + 0xFFFF terminator).
        let mut sorted: Vec<(u16, u16)> = chars.iter().map(|&(c, g, _)| (c as u16, g)).collect();
        sorted.sort_by_key(|&(c, _)| c);
        sorted.push((0xFFFF, 0));
        let seg_count = sorted.len() as u16;
        let seg_x2 = seg_count * 2;
        let mut fmt4 = Vec::new();
        fmt4.extend_from_slice(&4u16.to_be_bytes());
        let len_pos = fmt4.len();
        fmt4.extend_from_slice(&0u16.to_be_bytes());
        fmt4.extend_from_slice(&0u16.to_be_bytes()); // lang
        fmt4.extend_from_slice(&seg_x2.to_be_bytes());
        fmt4.extend_from_slice(&2u16.to_be_bytes()); // searchRange
        fmt4.extend_from_slice(&0u16.to_be_bytes()); // entrySelector
        fmt4.extend_from_slice(&0u16.to_be_bytes()); // rangeShift
        for &(c, _) in &sorted {
            fmt4.extend_from_slice(&c.to_be_bytes());
        }
        fmt4.extend_from_slice(&0u16.to_be_bytes()); // reservedPad
        for &(c, _) in &sorted {
            fmt4.extend_from_slice(&c.to_be_bytes());
        }
        for &(c, g) in &sorted {
            let delta = if c == 0xFFFF { 1i16 } else { (i32::from(g) - i32::from(c)) as i16 };
            fmt4.extend_from_slice(&delta.to_be_bytes());
        }
        for _ in &sorted {
            fmt4.extend_from_slice(&0u16.to_be_bytes());
        }
        let fmt4_len = fmt4.len() as u16;
        fmt4[len_pos..len_pos + 2].copy_from_slice(&fmt4_len.to_be_bytes());
        let mut cmap = Vec::new();
        cmap.extend_from_slice(&0u16.to_be_bytes());
        cmap.extend_from_slice(&1u16.to_be_bytes());
        cmap.extend_from_slice(&3u16.to_be_bytes()); // platform Windows
        cmap.extend_from_slice(&1u16.to_be_bytes()); // encoding BMP
        cmap.extend_from_slice(&12u32.to_be_bytes());
        cmap.extend_from_slice(&fmt4);

        let mut tables: Vec<([u8; 4], Vec<u8>)> = vec![
            (*b"cmap", cmap),
            (*b"head", head),
            (*b"hhea", hhea),
            (*b"hmtx", hmtx),
            (*b"maxp", maxp),
        ];

        if !legacy_kern.is_empty() {
            let mut pairs = legacy_kern.to_vec();
            pairs.sort_by_key(|&(l, r, _)| (l, r));
            let mut sub = Vec::new();
            let sub_len = (14 + pairs.len() * 6) as u16;
            sub.extend_from_slice(&0u16.to_be_bytes()); // version
            sub.extend_from_slice(&sub_len.to_be_bytes());
            sub.extend_from_slice(&1u16.to_be_bytes()); // coverage: horizontal format 0
            sub.extend_from_slice(&(pairs.len() as u16).to_be_bytes());
            sub.extend_from_slice(&0u16.to_be_bytes());
            sub.extend_from_slice(&0u16.to_be_bytes());
            sub.extend_from_slice(&0u16.to_be_bytes());
            for (l, r, v) in pairs {
                sub.extend_from_slice(&l.to_be_bytes());
                sub.extend_from_slice(&r.to_be_bytes());
                sub.extend_from_slice(&v.to_be_bytes());
            }
            let mut kern = Vec::new();
            kern.extend_from_slice(&0u16.to_be_bytes()); // version 0 (OT)
            kern.extend_from_slice(&1u16.to_be_bytes()); // nTables
            kern.extend_from_slice(&sub);
            tables.push((*b"kern", kern));
        }

        if !gsub_ligas.is_empty() {
            // Minimal GSUB v1.0 with 1 feature 'liga' -> 1 LigatureSubst (type 4) lookup.
            let mut ligs = gsub_ligas.to_vec();
            ligs.sort_by_key(|&(a, _, _)| a);
            let mut sub = Vec::new();
            let n = ligs.len() as u16;
            let cov_off = 6 + n * 2;
            sub.extend_from_slice(&1u16.to_be_bytes()); // format 1
            sub.extend_from_slice(&cov_off.to_be_bytes());
            sub.extend_from_slice(&n.to_be_bytes());
            let cov_len = 4 + n * 2;
            for i in 0..n {
                let set_off = cov_off + cov_len + i * 10;
                sub.extend_from_slice(&set_off.to_be_bytes());
            }
            // Coverage format 1
            sub.extend_from_slice(&1u16.to_be_bytes());
            sub.extend_from_slice(&n.to_be_bytes());
            for &(first, _, _) in &ligs {
                sub.extend_from_slice(&first.to_be_bytes());
            }
            // Each LigatureSet (4 bytes) + 1 Ligature (6 bytes) = 10 bytes.
            for &(_, second, lig_gid) in &ligs {
                sub.extend_from_slice(&1u16.to_be_bytes()); // ligatureCount
                sub.extend_from_slice(&4u16.to_be_bytes()); // offset to Ligature
                sub.extend_from_slice(&lig_gid.to_be_bytes());
                sub.extend_from_slice(&2u16.to_be_bytes()); // componentCount
                sub.extend_from_slice(&second.to_be_bytes());
            }
            tables.push((*b"GSUB", build_ot_layout_table(b"liga", 4, &sub)));
        }

        if !gpos_kerns.is_empty() {
            // Minimal GPOS v1.0 with 1 feature 'kern' -> 1 PairPosFormat1 (type 2) lookup.
            let mut pairs = gpos_kerns.to_vec();
            pairs.sort_by_key(|&(a, _, _)| a);
            let n = pairs.len() as u16;
            let cov_off = 10 + n * 2;
            let cov_len = 4 + n * 2;
            let mut sub = Vec::new();
            sub.extend_from_slice(&1u16.to_be_bytes()); // format 1
            sub.extend_from_slice(&cov_off.to_be_bytes());
            sub.extend_from_slice(&0x0004u16.to_be_bytes()); // valueFormat1 = X_ADVANCE
            sub.extend_from_slice(&0x0000u16.to_be_bytes()); // valueFormat2 = none
            sub.extend_from_slice(&n.to_be_bytes());
            for i in 0..n {
                let set_off = cov_off + cov_len + i * 6;
                sub.extend_from_slice(&set_off.to_be_bytes());
            }
            sub.extend_from_slice(&1u16.to_be_bytes());
            sub.extend_from_slice(&n.to_be_bytes());
            for &(first, _, _) in &pairs {
                sub.extend_from_slice(&first.to_be_bytes());
            }
            // Each PairSet: pairValueCount(2) + secondGlyph(2) + xAdvance(2) = 6 bytes.
            for &(_, second, val) in &pairs {
                sub.extend_from_slice(&1u16.to_be_bytes());
                sub.extend_from_slice(&second.to_be_bytes());
                sub.extend_from_slice(&val.to_be_bytes());
            }
            tables.push((*b"GPOS", build_ot_layout_table(b"kern", 2, &sub)));
        }

        tables.sort_by_key(|(tag, _)| *tag);
        let num_tables = tables.len() as u16;
        let mut sfnt = Vec::new();
        sfnt.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        sfnt.extend_from_slice(&num_tables.to_be_bytes());
        sfnt.extend_from_slice(&0u16.to_be_bytes());
        sfnt.extend_from_slice(&0u16.to_be_bytes());
        sfnt.extend_from_slice(&0u16.to_be_bytes());
        let mut offset = 12 + usize::from(num_tables) * 16;
        for (tag, data) in &tables {
            sfnt.extend_from_slice(tag);
            sfnt.extend_from_slice(&0u32.to_be_bytes()); // checksum
            sfnt.extend_from_slice(&(offset as u32).to_be_bytes());
            sfnt.extend_from_slice(&(data.len() as u32).to_be_bytes());
            offset += (data.len() + 3) & !3;
        }
        for (_, data) in tables {
            sfnt.extend_from_slice(&data);
            while sfnt.len() % 4 != 0 {
                sfnt.push(0);
            }
        }
        sfnt
    }

    fn build_ot_layout_table(feature_tag: &[u8; 4], lookup_type: u16, subtable: &[u8]) -> Vec<u8> {
        let mut t = Vec::new();
        // Header (10 bytes): major 1, minor 0, scriptList=10, featureList=12, lookupList=26
        t.extend_from_slice(&1u16.to_be_bytes());
        t.extend_from_slice(&0u16.to_be_bytes());
        t.extend_from_slice(&10u16.to_be_bytes());
        t.extend_from_slice(&12u16.to_be_bytes());
        t.extend_from_slice(&26u16.to_be_bytes());
        // ScriptList at 10 (2 bytes): 0 scripts
        t.extend_from_slice(&0u16.to_be_bytes());
        // FeatureList at 12 (14 bytes): 1 record (6 bytes) + FeatureTable (6 bytes) + header (2)
        t.extend_from_slice(&1u16.to_be_bytes());
        t.extend_from_slice(feature_tag);
        t.extend_from_slice(&8u16.to_be_bytes()); // offset to FeatureTable from FeatureList
        t.extend_from_slice(&0u16.to_be_bytes()); // featureParamsOffset
        t.extend_from_slice(&1u16.to_be_bytes()); // lookupIndexCount
        t.extend_from_slice(&0u16.to_be_bytes()); // lookupListIndices[0]
        // LookupList at 26: count(2) + offset(2) + Lookup header(8) + subtable
        t.extend_from_slice(&1u16.to_be_bytes());
        t.extend_from_slice(&4u16.to_be_bytes());
        t.extend_from_slice(&lookup_type.to_be_bytes());
        t.extend_from_slice(&0u16.to_be_bytes()); // lookupFlag
        t.extend_from_slice(&1u16.to_be_bytes()); // subTableCount
        t.extend_from_slice(&8u16.to_be_bytes()); // subtableOffset from Lookup
        t.extend_from_slice(subtable);
        t
    }

    #[test]
    fn bidi_reorders_mixed_ltr_rtl_keeps_digits_ltr_and_mirrors_brackets() {
        // Pure Hebrew RTL reverses visually, while embedded European digits stay LTR.
        let (vis, rtl) = reorder_visual("שלום 123", Direction::Auto);
        assert!(rtl);
        let chars: String = vis.iter().map(|&(c, _, _)| c).collect();
        assert_eq!(chars, "123 םולש", "Hebrew reversed while digits stay LTR: {chars:?}");

        // Paired brackets mirror inside RTL context.
        let (vis_b, rtl_b) = reorder_visual("(אב)", Direction::Rtl);
        assert!(rtl_b);
        let chars_b: String = vis_b.iter().map(|&(c, _, _)| c).collect();
        assert_eq!(chars_b, "(בא)", "mirrored brackets wrap reversed Hebrew: {chars_b:?}");

        // LTR paragraph with embedded Hebrew run.
        let (vis_m, rtl_m) = reorder_visual("Hi אב!", Direction::Ltr);
        assert!(!rtl_m);
        let chars_m: String = vis_m.iter().map(|&(c, _, _)| c).collect();
        assert_eq!(chars_m, "Hi בא!");
    }

    #[test]
    fn arabic_contextual_joining_and_lam_alef_ligature() {
        // "سلام" (Seen-Lam-Alef-Meem): Seen initial, Lam+Alef joined final ligature, Meem isolated!
        let shaped = shape_arabic_logical("\u{0633}\u{0644}\u{0627}\u{0645}");
        let got: Vec<char> = shaped.chars().collect();
        assert_eq!(got, vec!['\u{FEB3}', '\u{FEFC}', '\u{FEE1}'], "Seen-init + LamAlef-final + Meem-iso");
    }

    #[test]
    fn opentype_gsub_ligatures_gpos_kerning_and_legacy_kern() {
        // Font with 'A'(1, 600), 'V'(2, 600), 'f'(3, 400), 'i'(4, 300), ligature 'fi'(5, 550),
        // GSUB liga (3,4)->5 and GPOS kern (1,2) -> -120.
        let bytes = build_test_font(
            &[('A', 1, 600), ('V', 2, 600), ('f', 3, 400), ('i', 4, 300), ('\u{FB01}', 5, 550)],
            &[],
            &[(3, 4, 5)],
            &[(1, 2, -120)],
        );
        let chain = FontChain::new(&[&bytes]).unwrap();
        let line = shape(&chain, "AVfi", Direction::Ltr);
        assert_eq!(line.glyphs.len(), 3, "fi must collapse into 1 ligature glyph: {:?}", line.glyphs);
        assert_eq!(line.glyphs[0].glyph_id, 1);
        assert!((line.glyphs[0].x_advance - 0.48).abs() < 1e-4, "A+V GPOS kerned 600-120=480: {:?}", line.glyphs[0]);
        assert_eq!(line.glyphs[2].glyph_id, 5, "ligature glyph id 5");
        assert!((line.glyphs[2].x_advance - 0.55).abs() < 1e-4);

        // Legacy TrueType `kern` Format 0 fallback when no GPOS table exists.
        let legacy = build_test_font(&[('T', 1, 600), ('o', 2, 500)], &[(1, 2, -80)], &[], &[]);
        let chain2 = FontChain::new(&[&legacy]).unwrap();
        let line2 = shape(&chain2, "To", Direction::Ltr);
        assert!((line2.glyphs[0].x_advance - 0.52).abs() < 1e-4, "legacy kern 600-80=520");
    }

    #[test]
    fn font_chain_falls_back_for_missing_scripts() {
        // Primary font only has Latin 'A'; fallback font has Hebrew 'א' and 'ב'.
        let latin = build_test_font(&[('A', 1, 600), (' ', 2, 250)], &[], &[], &[]);
        let hebrew = build_test_font(&[('א', 10, 550), ('ב', 11, 500)], &[], &[], &[]);
        let chain = FontChain::new(&[&latin, &hebrew]).unwrap();
        let line = shape(&chain, "A אב", Direction::Ltr);
        assert_eq!(line.glyphs.len(), 4);
        assert_eq!((line.glyphs[0].font_index, line.glyphs[0].glyph_id, line.glyphs[0].rtl), (0, 1, false));
        assert_eq!((line.glyphs[1].font_index, line.glyphs[1].glyph_id, line.glyphs[1].rtl), (0, 2, false));
        // Hebrew run visually reversed: 'ב'(11) then 'א'(10), both resolved from fallback font 1!
        assert_eq!((line.glyphs[2].font_index, line.glyphs[2].glyph_id, line.glyphs[2].rtl), (1, 11, true));
        assert_eq!((line.glyphs[3].font_index, line.glyphs[3].glyph_id, line.glyphs[3].rtl), (1, 10, true));
    }
}
