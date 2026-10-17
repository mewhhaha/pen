//! Text laid out for the game's glyph drawer, which draws code points left
//! to right one glyph each: Arabic joined into its presentation forms, then
//! each line put in visual order by the Unicode bidirectional algorithm
//! (UAX #9, through `unicode-bidi`), its marks after their letters and its
//! brackets mirrored in right-to-left runs.
use unicode_bidi::{BidiClass, BidiInfo, bidi_class};

/// `text` in visual order for a left-to-right glyph drawer, line by line
/// (each line between `\n`s its own paragraph), and whether the first line
/// reads right to left: its base direction from its first strong character
/// (UAX #9 P2 and P3), left to right without one. Arabic letters become
/// their isolated, final, initial or medial presentation forms first, as
/// they join in logical order. Text without right-to-left characters comes
/// back as it is; in text with them, the invisible controls that steer
/// joining and direction (directional marks, embeddings, isolates, ZWJ,
/// ZWNJ) are left out, as the font has no glyphs for them.
pub fn display(text: &str) -> (String, bool) {
    use BidiClass::{AL, AN, R, RLE, RLI, RLO};
    if !text.chars().any(|char| matches!(bidi_class(char), R | AL | AN | RLE | RLO | RLI)) {
        return (text.to_owned(), false);
    }
    let mut shown = String::with_capacity(text.len() * 3 / 2);
    let mut first = None;
    for (index, line) in text.split('\n').enumerate() {
        if index > 0 {
            shown.push('\n');
        }
        let right_to_left = reorder(&shape(line), &mut shown);
        first.get_or_insert(right_to_left);
    }
    (shown, first.unwrap_or(false))
}

/// Appends `line` in visual order, and says whether its first paragraph
/// reads right to left. Right-to-left runs go backwards by clusters (a
/// character and the marks after it, UAX #9 L3) with their brackets
/// mirrored (L4), which `unicode-bidi` leaves to its caller.
fn reorder(line: &str, shown: &mut String) -> bool {
    let info = BidiInfo::new(line, None);
    for paragraph in &info.paragraphs {
        let (levels, runs) = info.visual_runs(paragraph, paragraph.range.clone());
        for run in runs {
            let text = &line[run.clone()];
            if !levels[run.start].is_rtl() {
                shown.extend(text.chars().filter(|char| !invisible(*char)));
                continue;
            }
            let mut marks = Vec::new();
            for char in text.chars().rev() {
                if bidi_class(char) == BidiClass::NSM {
                    marks.push(char);
                    continue;
                }
                if !invisible(char) {
                    shown.push(mirror(char));
                }
                shown.extend(marks.drain(..).rev());
            }
            shown.extend(marks.drain(..).rev());
        }
    }
    info.paragraphs.first().is_some_and(|paragraph| paragraph.level.is_rtl())
}

/// Characters without a glyph that steer joining and direction: explicit
/// embeddings, overrides and isolates, the directional marks, ZWJ and ZWNJ
/// and the other boundary-neutral format characters.
fn invisible(char: char) -> bool {
    use BidiClass::{BN, FSI, LRE, LRI, LRO, PDF, PDI, RLE, RLI, RLO};
    matches!(char, '\u{200E}' | '\u{200F}' | '\u{061C}')
        || matches!(bidi_class(char), BN | LRE | RLE | LRO | RLO | PDF | LRI | RLI | FSI | PDI)
}

/// A bracket's mirror image, which a right-to-left run shows in its place.
fn mirror(char: char) -> char {
    match char {
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
        _ => char,
    }
}

/// How a character joins its neighbours (Unicode's Joining_Type).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Joining {
    /// Joins neither side (U): a space, a digit, Latin, hamza, ZWNJ.
    None,
    /// Joins the letter before it only (R): alef, dal, reh, waw, ...
    Right,
    /// Joins both sides (D): beh, seen, lam, meem, ...
    Dual,
    /// Makes both neighbours join it (C): tatweel and ZWJ.
    Causing,
    /// Lets its neighbours join across it (T): marks and invisible format
    /// characters.
    Transparent,
}

/// The joining type of a character, from Unicode's ArabicShaping.txt for
/// the Arabic and Arabic Supplement blocks.
fn joining(char: char) -> Joining {
    match char {
        '\u{0620}'
        | '\u{0626}'
        | '\u{0628}'
        | '\u{062A}'..='\u{062E}'
        | '\u{0633}'..='\u{063F}'
        | '\u{0641}'..='\u{0647}'
        | '\u{0649}'..='\u{064A}'
        | '\u{066E}'..='\u{066F}'
        | '\u{0678}'..='\u{0687}'
        | '\u{069A}'..='\u{06BF}'
        | '\u{06C1}'..='\u{06C2}'
        | '\u{06CC}'
        | '\u{06CE}'
        | '\u{06D0}'..='\u{06D1}'
        | '\u{06FA}'..='\u{06FC}'
        | '\u{06FF}'
        | '\u{0750}'..='\u{0758}'
        | '\u{075C}'..='\u{076A}'
        | '\u{076D}'..='\u{0770}'
        | '\u{0772}'
        | '\u{0775}'..='\u{0777}'
        | '\u{077A}'..='\u{077F}' => Joining::Dual,
        '\u{0622}'..='\u{0625}'
        | '\u{0627}'
        | '\u{0629}'
        | '\u{062F}'..='\u{0632}'
        | '\u{0648}'
        | '\u{0671}'..='\u{0673}'
        | '\u{0675}'..='\u{0677}'
        | '\u{0688}'..='\u{0699}'
        | '\u{06C0}'
        | '\u{06C3}'..='\u{06CB}'
        | '\u{06CD}'
        | '\u{06CF}'
        | '\u{06D2}'..='\u{06D3}'
        | '\u{06D5}'
        | '\u{06EE}'..='\u{06EF}'
        | '\u{0759}'..='\u{075B}'
        | '\u{076B}'..='\u{076C}'
        | '\u{0771}'
        | '\u{0773}'..='\u{0774}'
        | '\u{0778}'..='\u{0779}' => Joining::Right,
        '\u{0640}' | '\u{200D}' => Joining::Causing,
        '\u{200C}' => Joining::None,
        _ if bidi_class(char) == BidiClass::NSM || invisible(char) => Joining::Transparent,
        _ => Joining::None,
    }
}

/// Arabic letters and their presentation forms (Arabic Presentation
/// Forms-B, and Forms-A for Persian, Urdu and other letters): isolated,
/// final, initial and medial, 0 where Unicode has none. By letter.
const FORMS: [(u16, [u16; 4]); 76] = [
    (0x0621, [0xFE80, 0, 0, 0]),
    (0x0622, [0xFE81, 0xFE82, 0, 0]),
    (0x0623, [0xFE83, 0xFE84, 0, 0]),
    (0x0624, [0xFE85, 0xFE86, 0, 0]),
    (0x0625, [0xFE87, 0xFE88, 0, 0]),
    (0x0626, [0xFE89, 0xFE8A, 0xFE8B, 0xFE8C]),
    (0x0627, [0xFE8D, 0xFE8E, 0, 0]),
    (0x0628, [0xFE8F, 0xFE90, 0xFE91, 0xFE92]),
    (0x0629, [0xFE93, 0xFE94, 0, 0]),
    (0x062A, [0xFE95, 0xFE96, 0xFE97, 0xFE98]),
    (0x062B, [0xFE99, 0xFE9A, 0xFE9B, 0xFE9C]),
    (0x062C, [0xFE9D, 0xFE9E, 0xFE9F, 0xFEA0]),
    (0x062D, [0xFEA1, 0xFEA2, 0xFEA3, 0xFEA4]),
    (0x062E, [0xFEA5, 0xFEA6, 0xFEA7, 0xFEA8]),
    (0x062F, [0xFEA9, 0xFEAA, 0, 0]),
    (0x0630, [0xFEAB, 0xFEAC, 0, 0]),
    (0x0631, [0xFEAD, 0xFEAE, 0, 0]),
    (0x0632, [0xFEAF, 0xFEB0, 0, 0]),
    (0x0633, [0xFEB1, 0xFEB2, 0xFEB3, 0xFEB4]),
    (0x0634, [0xFEB5, 0xFEB6, 0xFEB7, 0xFEB8]),
    (0x0635, [0xFEB9, 0xFEBA, 0xFEBB, 0xFEBC]),
    (0x0636, [0xFEBD, 0xFEBE, 0xFEBF, 0xFEC0]),
    (0x0637, [0xFEC1, 0xFEC2, 0xFEC3, 0xFEC4]),
    (0x0638, [0xFEC5, 0xFEC6, 0xFEC7, 0xFEC8]),
    (0x0639, [0xFEC9, 0xFECA, 0xFECB, 0xFECC]),
    (0x063A, [0xFECD, 0xFECE, 0xFECF, 0xFED0]),
    (0x0641, [0xFED1, 0xFED2, 0xFED3, 0xFED4]),
    (0x0642, [0xFED5, 0xFED6, 0xFED7, 0xFED8]),
    (0x0643, [0xFED9, 0xFEDA, 0xFEDB, 0xFEDC]),
    (0x0644, [0xFEDD, 0xFEDE, 0xFEDF, 0xFEE0]),
    (0x0645, [0xFEE1, 0xFEE2, 0xFEE3, 0xFEE4]),
    (0x0646, [0xFEE5, 0xFEE6, 0xFEE7, 0xFEE8]),
    (0x0647, [0xFEE9, 0xFEEA, 0xFEEB, 0xFEEC]),
    (0x0648, [0xFEED, 0xFEEE, 0, 0]),
    (0x0649, [0xFEEF, 0xFEF0, 0xFBE8, 0xFBE9]),
    (0x064A, [0xFEF1, 0xFEF2, 0xFEF3, 0xFEF4]),
    (0x0671, [0xFB50, 0xFB51, 0, 0]),
    (0x0677, [0xFBDD, 0, 0, 0]),
    (0x0679, [0xFB66, 0xFB67, 0xFB68, 0xFB69]),
    (0x067A, [0xFB5E, 0xFB5F, 0xFB60, 0xFB61]),
    (0x067B, [0xFB52, 0xFB53, 0xFB54, 0xFB55]),
    (0x067E, [0xFB56, 0xFB57, 0xFB58, 0xFB59]),
    (0x067F, [0xFB62, 0xFB63, 0xFB64, 0xFB65]),
    (0x0680, [0xFB5A, 0xFB5B, 0xFB5C, 0xFB5D]),
    (0x0683, [0xFB76, 0xFB77, 0xFB78, 0xFB79]),
    (0x0684, [0xFB72, 0xFB73, 0xFB74, 0xFB75]),
    (0x0686, [0xFB7A, 0xFB7B, 0xFB7C, 0xFB7D]),
    (0x0687, [0xFB7E, 0xFB7F, 0xFB80, 0xFB81]),
    (0x0688, [0xFB88, 0xFB89, 0, 0]),
    (0x068C, [0xFB84, 0xFB85, 0, 0]),
    (0x068D, [0xFB82, 0xFB83, 0, 0]),
    (0x068E, [0xFB86, 0xFB87, 0, 0]),
    (0x0691, [0xFB8C, 0xFB8D, 0, 0]),
    (0x0698, [0xFB8A, 0xFB8B, 0, 0]),
    (0x06A4, [0xFB6A, 0xFB6B, 0xFB6C, 0xFB6D]),
    (0x06A6, [0xFB6E, 0xFB6F, 0xFB70, 0xFB71]),
    (0x06A9, [0xFB8E, 0xFB8F, 0xFB90, 0xFB91]),
    (0x06AD, [0xFBD3, 0xFBD4, 0xFBD5, 0xFBD6]),
    (0x06AF, [0xFB92, 0xFB93, 0xFB94, 0xFB95]),
    (0x06B1, [0xFB9A, 0xFB9B, 0xFB9C, 0xFB9D]),
    (0x06B3, [0xFB96, 0xFB97, 0xFB98, 0xFB99]),
    (0x06BA, [0xFB9E, 0xFB9F, 0, 0]),
    (0x06BB, [0xFBA0, 0xFBA1, 0xFBA2, 0xFBA3]),
    (0x06BE, [0xFBAA, 0xFBAB, 0xFBAC, 0xFBAD]),
    (0x06C0, [0xFBA4, 0xFBA5, 0, 0]),
    (0x06C1, [0xFBA6, 0xFBA7, 0xFBA8, 0xFBA9]),
    (0x06C5, [0xFBE0, 0xFBE1, 0, 0]),
    (0x06C6, [0xFBD9, 0xFBDA, 0, 0]),
    (0x06C7, [0xFBD7, 0xFBD8, 0, 0]),
    (0x06C8, [0xFBDB, 0xFBDC, 0, 0]),
    (0x06C9, [0xFBE2, 0xFBE3, 0, 0]),
    (0x06CB, [0xFBDE, 0xFBDF, 0, 0]),
    (0x06CC, [0xFBFC, 0xFBFD, 0xFBFE, 0xFBFF]),
    (0x06D0, [0xFBE4, 0xFBE5, 0xFBE6, 0xFBE7]),
    (0x06D2, [0xFBAE, 0xFBAF, 0, 0]),
    (0x06D3, [0xFBB0, 0xFBB1, 0, 0]),
];

/// Where a letter stands in its word, which picks its form.
#[derive(Clone, Copy)]
enum Position {
    Isolated,
    Final,
    Initial,
    Medial,
}

/// A letter's presentation form at `position`, if Unicode has one.
fn form(letter: char, position: Position) -> Option<char> {
    let code = u16::try_from(u32::from(letter)).ok()?;
    let at = FORMS.binary_search_by_key(&code, |(letter, _)| *letter).ok()?;
    let form = FORMS[at].1[position as usize];
    (form != 0).then(|| char::from_u32(u32::from(form))).flatten()
}

/// The lam-alef ligature of lam and `alef`, isolated or final, if `alef`
/// is one of the four alefs that make one.
fn lam_alef(alef: char, joined: bool) -> Option<char> {
    let isolated = match alef {
        '\u{0622}' => '\u{FEF5}',
        '\u{0623}' => '\u{FEF7}',
        '\u{0625}' => '\u{FEF9}',
        '\u{0627}' => '\u{FEFB}',
        _ => return None,
    };
    if joined { char::from_u32(u32::from(isolated) + 1) } else { Some(isolated) }
}

/// `line` with its Arabic letters in the presentation forms their joins
/// give them, in logical order: each joins the nearest letter before it
/// and after it that is not transparent, when both sides join that way.
/// Lam then alef becomes the lam-alef ligature, its marks after it.
fn shape(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    if !chars.iter().any(|char| form(*char, Position::Isolated).is_some()) {
        return line.to_owned();
    }
    let kinds: Vec<Joining> = chars.iter().map(|char| joining(*char)).collect();
    // Where the nearest character after each that is not transparent is.
    let mut after = vec![None; chars.len()];
    let mut next = None;
    for at in (0..chars.len()).rev() {
        after[at] = next;
        if kinds[at] != Joining::Transparent {
            next = Some(at);
        }
    }
    let mut shaped = String::with_capacity(line.len() * 3 / 2);
    let mut before = Joining::None;
    let mut at = 0;
    while at < chars.len() {
        let (char, kind) = (chars[at], kinds[at]);
        if kind == Joining::Transparent {
            shaped.push(char);
            at += 1;
            continue;
        }
        let joins_before = matches!(kind, Joining::Dual | Joining::Right | Joining::Causing)
            && matches!(before, Joining::Dual | Joining::Causing);
        let next = after[at];
        let joins_after = matches!(kind, Joining::Dual | Joining::Causing)
            && next.is_some_and(|next| matches!(kinds[next], Joining::Dual | Joining::Right | Joining::Causing));
        if char == '\u{0644}'
            && let Some(alef) = next
            && let Some(ligature) = lam_alef(chars[alef], joins_before)
        {
            shaped.push(ligature);
            shaped.extend(&chars[at + 1..alef]);
            before = Joining::Right;
            at = alef + 1;
            continue;
        }
        let position = match (joins_before, joins_after) {
            (false, false) => Position::Isolated,
            (true, false) => Position::Final,
            (false, true) => Position::Initial,
            (true, true) => Position::Medial,
        };
        shaped.push(form(char, position).unwrap_or(char));
        before = kind;
        at += 1;
    }
    shaped
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The code points of a displayed text, and its direction.
    fn shown(text: &str) -> (Vec<u32>, bool) {
        let (shown, right_to_left) = display(text);
        (shown.chars().map(u32::from).collect(), right_to_left)
    }

    #[test]
    fn arabic_joins_into_presentation_forms_in_visual_order() {
        // Seen initial, lam-alef final, meem isolated, drawn from the left.
        assert_eq!(shown("سلام"), (vec![0xFEE1, 0xFEFC, 0xFEB3], true));
        assert_eq!(shown("لا"), (vec![0xFEFB], true));
        assert_eq!(shown("لأ"), (vec![0xFEF7], true));
        assert_eq!(shown("بلا"), (vec![0xFEFC, 0xFE91], true));
        // Meem initial, reh final; hah initial, beh medial, alef final.
        assert_eq!(shown("مرحبا"), (vec![0xFE8E, 0xFE92, 0xFEA3, 0xFEAE, 0xFEE3], true));
        // Hamza never joins.
        assert_eq!(shown("ماء"), (vec![0xFE80, 0xFE8E, 0xFEE3], true));
        // Marks follow their letter and do not break its joins.
        assert_eq!(shown("بَب"), (vec![0xFE90, 0xFE91, 0x064E], true));
        assert_eq!(shown("لَا"), (vec![0xFEFB, 0x064E], true));
        // Tatweel joins both sides; ZWNJ joins neither and is left out.
        assert_eq!(shown("بـ"), (vec![0x0640, 0xFE91], true));
        assert_eq!(shown("ک\u{200C}ک"), (vec![0xFB8E, 0xFB8E], true));
    }

    #[test]
    fn persian_and_urdu_letters_take_forms_a() {
        // Peh initial, farsi yeh medial, keheh final.
        assert_eq!(shown("پیک"), (vec![0xFB8F, 0xFBFF, 0xFB58], true));
        // Tcheh initial, jeh final; gaf isolated.
        assert_eq!(shown("چژ گ"), (vec![0xFB92, 0x20, 0xFB8B, 0xFB7C], true));
    }

    #[test]
    fn mixed_lines_keep_left_to_right_runs_in_order() {
        // Noon initial, qaf medial, tah medial, teh marbuta final.
        let (score, right_to_left) = display("Score: 12 نقطة");
        assert_eq!(score, "Score: 12 \u{FE94}\u{FEC4}\u{FED8}\u{FEE7}");
        assert!(!right_to_left);
        let (sentence, right_to_left) = display("نقطة 12");
        assert_eq!(sentence, "12 \u{FE94}\u{FEC4}\u{FED8}\u{FEE7}");
        assert!(right_to_left);
    }

    #[test]
    fn hebrew_reverses_without_shaping_and_brackets_mirror() {
        assert_eq!(display("שלום"), ("םולש".to_owned(), true));
        assert_eq!(shown("(مرحبا)"), (vec![0x28, 0xFE8E, 0xFE92, 0xFEA3, 0xFEAE, 0xFEE3, 0x29], true));
        assert_eq!(display("«שלום» [א]"), ("[א] «םולש»".to_owned(), true));
    }

    #[test]
    fn each_line_is_its_own_paragraph() {
        assert_eq!(display("שלום\nabc דג"), ("םולש\nabc גד".to_owned(), true));
        assert_eq!(display("abc\nשלום"), ("abc\nםולש".to_owned(), false));
        assert_eq!(display("\u{200F}abc\nx"), ("abc\nx".to_owned(), true));
    }

    #[test]
    fn left_to_right_text_comes_back_as_it_is() {
        for text in ["Continue", "Fortsätt", "MSAA 4× + FXAA", "  •  ", "", "a\u{200D}b", "Привет, мир"] {
            assert_eq!(display(text), (text.to_owned(), false));
        }
    }
}
