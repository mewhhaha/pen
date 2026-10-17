//! Unicode CLDR's cardinal plural rules, and counts as the game shows them.
//! The rules read a count's operands from its digits as shown (up to three
//! decimals, trailing zeros dropped), so `1.5` has one fraction digit and
//! `1.0` shows as `1`, an integer. Without trailing zeros CLDR's `w` equals
//! `v` and `t` equals `f`, and without compact notation (`1.2M`) `e` is 0.
use pen_protocol::plural_category::{FEW, MANY, ONE, OTHER, TWO, ZERO};

/// A language's rule set, named by a language that uses it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Rules {
    /// CLDR's root: every count is `other` (ja, zh, ko, th, vi, ...).
    Root,
    /// one: i = 1 and v = 0 (en, de, nl, sv, fi, et, ...).
    English,
    /// one: n = 1 (nb, bg, el, hu, tr, ...).
    Single,
    Danish,
    Icelandic,
    /// one: n = 1; many: a whole number of millions.
    Spanish,
    /// one: i = 1 and v = 0; many: a whole number of millions (it, ca, pt-PT).
    Italian,
    /// one: i = 0,1; many: a whole number of millions (fr, pt).
    French,
    /// ru, uk, be.
    Russian,
    Polish,
    /// cs, sk.
    Czech,
    Lithuanian,
    Latvian,
    Romanian,
    Slovenian,
    /// hr, sr, bs.
    Serbian,
    Arabic,
    Hebrew,
    Irish,
    Welsh,
    /// one: i = 0 or n = 1 (hi, fa, bn, ...).
    Hindi,
    Macedonian,
    Filipino,
    Maltese,
    /// one: i = 0,1 (hy).
    Armenian,
}

/// The rule set of a BCP 47 tag: by its primary subtag, except Portugal's
/// Portuguese (`pt-PT`), which has its own.
fn rules(tag: &str) -> Rules {
    let mut subtags = tag.split(['-', '_']);
    let language = subtags.next().unwrap_or_default().to_ascii_lowercase();
    match language.as_str() {
        "pt" if subtags.any(|subtag| subtag.eq_ignore_ascii_case("pt")) => Rules::Italian,
        "en" | "de" | "nl" | "sv" | "fi" | "et" | "ur" | "yi" | "sw" | "gl" | "ast" | "fy" | "ia" | "io" | "lij"
        | "sc" => Rules::English,
        "nb" | "no" | "nn" | "bg" | "el" | "hu" | "tr" | "ps" | "dv" | "ug" | "ckb" | "sd" | "sq" | "af" | "az"
        | "eu" | "ka" | "kk" | "ky" | "mn" | "ml" | "mr" | "ta" | "te" | "lb" | "fo" | "uz" | "tk" | "ne" | "or"
        | "so" => Rules::Single,
        "da" => Rules::Danish,
        "is" => Rules::Icelandic,
        "es" => Rules::Spanish,
        "it" | "ca" => Rules::Italian,
        "fr" | "pt" => Rules::French,
        "ru" | "uk" | "be" => Rules::Russian,
        "pl" => Rules::Polish,
        "cs" | "sk" => Rules::Czech,
        "lt" => Rules::Lithuanian,
        "lv" => Rules::Latvian,
        "ro" | "mo" => Rules::Romanian,
        "sl" => Rules::Slovenian,
        "hr" | "sr" | "bs" | "sh" => Rules::Serbian,
        "ar" => Rules::Arabic,
        "he" | "iw" => Rules::Hebrew,
        "ga" => Rules::Irish,
        "cy" => Rules::Welsh,
        "hi" | "fa" | "bn" | "gu" | "kn" | "zu" | "am" | "as" => Rules::Hindi,
        "mk" => Rules::Macedonian,
        "fil" | "tl" => Rules::Filipino,
        "mt" => Rules::Maltese,
        "hy" | "kab" => Rules::Armenian,
        _ => Rules::Root,
    }
}

/// The plural categories (`pen_protocol::plural_category`) the language of
/// BCP 47 tag `tag` uses, in code order; `other` is always one.
pub fn categories(tag: &str) -> &'static [u32] {
    match rules(tag) {
        Rules::Root => &[OTHER],
        Rules::English
        | Rules::Single
        | Rules::Danish
        | Rules::Icelandic
        | Rules::Hindi
        | Rules::Macedonian
        | Rules::Filipino
        | Rules::Armenian => &[ONE, OTHER],
        Rules::Spanish | Rules::Italian | Rules::French => &[ONE, MANY, OTHER],
        Rules::Russian | Rules::Polish | Rules::Czech | Rules::Lithuanian => &[ONE, FEW, MANY, OTHER],
        Rules::Latvian => &[ZERO, ONE, OTHER],
        Rules::Romanian | Rules::Serbian => &[ONE, FEW, OTHER],
        Rules::Slovenian => &[ONE, TWO, FEW, OTHER],
        Rules::Hebrew => &[ONE, TWO, OTHER],
        Rules::Irish | Rules::Maltese => &[ONE, TWO, FEW, MANY, OTHER],
        Rules::Arabic | Rules::Welsh => &[ZERO, ONE, TWO, FEW, MANY, OTHER],
    }
}

/// A count's operands as CLDR's rules read them.
struct Operands {
    /// The integer digits. One longer than 15 digits keeps its last 15
    /// above 10^15: the rules test the last six digits and small values.
    i: u64,
    /// How many fraction digits show.
    v: usize,
    /// The fraction digits as a number.
    f: u64,
}

impl Operands {
    fn of(shown: &str) -> Operands {
        let (whole, fraction) = shown.split_once('.').unwrap_or((shown, ""));
        let i = match whole.len().checked_sub(15) {
            Some(cut) if cut > 0 => 10u64.pow(15) + whole[cut..].parse::<u64>().unwrap_or(0),
            _ => whole.parse().unwrap_or(0),
        };
        Operands { i, v: fraction.len(), f: fraction.parse().unwrap_or(0) }
    }

    /// Whether `n`, the count, is the whole number `value`.
    fn is(&self, value: u64) -> bool {
        self.v == 0 && self.i == value
    }

    /// Whether `n` is a whole number in `low..=high`.
    fn range(&self, low: u64, high: u64) -> bool {
        self.v == 0 && (low..=high).contains(&self.i)
    }

    /// Whether `n % modulus` is a whole number in `low..=high`: never for a
    /// count with fraction digits.
    fn remainder(&self, modulus: u64, low: u64, high: u64) -> bool {
        self.v == 0 && (low..=high).contains(&(self.i % modulus))
    }

    /// A whole number of millions, CLDR's `many` for French, Spanish,
    /// Italian and Portuguese ("un million de").
    fn millions(&self) -> bool {
        self.v == 0 && self.i != 0 && self.i.is_multiple_of(1_000_000)
    }
}

fn category(rules: Rules, count: &Operands) -> u32 {
    let Operands { i, v, f } = *count;
    let (i10, i100, f10, f100) = (i % 10, i % 100, f % 10, f % 100);
    let pick = |choices: &[(bool, u32)]| choices.iter().find(|(yes, _)| *yes).map_or(OTHER, |(_, category)| *category);
    match rules {
        Rules::Root => OTHER,
        Rules::English => pick(&[(i == 1 && v == 0, ONE)]),
        Rules::Single => pick(&[(count.is(1), ONE)]),
        Rules::Danish => pick(&[(count.is(1) || (f != 0 && i <= 1), ONE)]),
        Rules::Icelandic => pick(&[((f == 0 && i10 == 1 && i100 != 11) || (f10 == 1 && f100 != 11), ONE)]),
        Rules::Spanish => pick(&[(count.is(1), ONE), (count.millions(), MANY)]),
        Rules::Italian => pick(&[(i == 1 && v == 0, ONE), (count.millions(), MANY)]),
        Rules::French => pick(&[(i <= 1, ONE), (count.millions(), MANY)]),
        Rules::Russian => pick(&[
            (v == 0 && i10 == 1 && i100 != 11, ONE),
            (v == 0 && (2..=4).contains(&i10) && !(12..=14).contains(&i100), FEW),
            (v == 0 && (i10 == 0 || (5..=9).contains(&i10) || (11..=14).contains(&i100)), MANY),
        ]),
        Rules::Polish => pick(&[
            (i == 1 && v == 0, ONE),
            (v == 0 && (2..=4).contains(&i10) && !(12..=14).contains(&i100), FEW),
            (v == 0 && ((i != 1 && i10 <= 1) || (5..=9).contains(&i10) || (12..=14).contains(&i100)), MANY),
        ]),
        Rules::Czech => pick(&[(i == 1 && v == 0, ONE), ((2..=4).contains(&i) && v == 0, FEW), (v != 0, MANY)]),
        Rules::Lithuanian => pick(&[
            (count.remainder(10, 1, 1) && !count.remainder(100, 11, 19), ONE),
            (count.remainder(10, 2, 9) && !count.remainder(100, 11, 19), FEW),
            (f != 0, MANY),
        ]),
        Rules::Latvian => pick(&[
            (count.remainder(10, 0, 0) || count.remainder(100, 11, 19) || (v == 2 && (11..=19).contains(&f100)), ZERO),
            (
                (count.remainder(10, 1, 1) && i100 != 11) || (v == 2 && f10 == 1 && f100 != 11) || (v != 2 && f10 == 1),
                ONE,
            ),
        ]),
        Rules::Romanian => pick(&[
            (i == 1 && v == 0, ONE),
            (v != 0 || count.is(0) || (!count.is(1) && count.remainder(100, 1, 19)), FEW),
        ]),
        Rules::Slovenian => pick(&[
            (v == 0 && i100 == 1, ONE),
            (v == 0 && i100 == 2, TWO),
            ((v == 0 && (3..=4).contains(&i100)) || v != 0, FEW),
        ]),
        Rules::Serbian => pick(&[
            ((v == 0 && i10 == 1 && i100 != 11) || (f10 == 1 && f100 != 11), ONE),
            (
                (v == 0 && (2..=4).contains(&i10) && !(12..=14).contains(&i100))
                    || ((2..=4).contains(&f10) && !(12..=14).contains(&f100)),
                FEW,
            ),
        ]),
        Rules::Arabic => pick(&[
            (count.is(0), ZERO),
            (count.is(1), ONE),
            (count.is(2), TWO),
            (count.remainder(100, 3, 10), FEW),
            (count.remainder(100, 11, 99), MANY),
        ]),
        Rules::Hebrew => pick(&[((i == 1 && v == 0) || (i == 0 && v != 0), ONE), (i == 2 && v == 0, TWO)]),
        Rules::Irish => {
            pick(&[(count.is(1), ONE), (count.is(2), TWO), (count.range(3, 6), FEW), (count.range(7, 10), MANY)])
        }
        Rules::Welsh => pick(&[
            (count.is(0), ZERO),
            (count.is(1), ONE),
            (count.is(2), TWO),
            (count.is(3), FEW),
            (count.is(6), MANY),
        ]),
        Rules::Hindi => pick(&[(i == 0 || count.is(1), ONE)]),
        Rules::Macedonian => pick(&[((v == 0 && i10 == 1 && i100 != 11) || (f10 == 1 && f100 != 11), ONE)]),
        Rules::Filipino => {
            let unlucky = |digit: u64| matches!(digit, 4 | 6 | 9);
            pick(&[((v == 0 && (1..=3).contains(&i)) || (v == 0 && !unlucky(i10)) || (v != 0 && !unlucky(f10)), ONE)])
        }
        Rules::Maltese => pick(&[
            (count.is(1), ONE),
            (count.is(2), TWO),
            (count.is(0) || count.remainder(100, 3, 10), FEW),
            (count.remainder(100, 11, 19), MANY),
        ]),
        Rules::Armenian => pick(&[(i <= 1, ONE)]),
    }
}

/// The plural category (`pen_protocol::plural_category`) of `count` in the
/// language of BCP 47 tag `language`, by CLDR's cardinal rules, read from
/// the count as shown: up to three decimals, trailing zeros dropped. A
/// negative count is read as its size; one that is not finite is `other`,
/// as is every count in a language without rules here (CLDR's root).
pub fn plural(language: &str, count: f64) -> u32 {
    if !count.is_finite() {
        return OTHER;
    }
    category(rules(language), &Operands::of(&shown(count.abs())))
}

/// A size as shown: up to three decimals, trailing zeros dropped.
fn shown(size: f64) -> String {
    let text = format!("{size:.3}");
    text.trim_end_matches('0').trim_end_matches('.').to_owned()
}

/// Whether the language of BCP 47 tag `tag` writes a decimal comma, as CLDR
/// writes its numbers in Latin digits. Arabic and Persian write a point
/// here, as CLDR does in Latin digits, rather than Arabic-Indic digits with
/// U+066B, which the game's font would have to hold.
fn decimal_comma(tag: &str) -> bool {
    let language = tag.split(['-', '_']).next().unwrap_or_default().to_ascii_lowercase();
    matches!(
        language.as_str(),
        "de" | "nl"
            | "sv"
            | "da"
            | "nb"
            | "no"
            | "nn"
            | "fi"
            | "et"
            | "is"
            | "it"
            | "es"
            | "ca"
            | "pt"
            | "fr"
            | "ru"
            | "uk"
            | "be"
            | "pl"
            | "cs"
            | "sk"
            | "lt"
            | "lv"
            | "ro"
            | "sl"
            | "hr"
            | "sr"
            | "bs"
            | "bg"
            | "el"
            | "hu"
            | "tr"
            | "vi"
            | "id"
            | "ps"
            | "mk"
            | "sq"
            | "af"
            | "az"
            | "eu"
            | "gl"
            | "ka"
            | "kk"
            | "hy"
            | "lo"
            | "lb"
            | "fo"
    )
}

/// `count` as the game shows it in the language of BCP 47 tag `language`:
/// up to three decimals, trailing zeros dropped, with the language's decimal
/// separator and no grouping (`1500`, `2,5` in Swedish, `2.5` in English).
/// A count that is not finite shows as Rust writes it (`inf`, `NaN`).
pub fn count_text(language: &str, count: f64) -> String {
    if !count.is_finite() {
        return count.to_string();
    }
    let size = shown(count.abs());
    let size = if decimal_comma(language) { size.replace('.', ",") } else { size };
    if count < 0.0 && size != "0" { format!("-{size}") } else { size }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Checks each count of `samples` is `category` in `language`.
    fn expect(language: &str, category: u32, samples: &[f64]) {
        for &count in samples {
            let name = pen_protocol::plural_category::NAMES[category as usize];
            let got = pen_protocol::plural_category::NAMES[plural(language, count) as usize];
            assert_eq!(got, name, "{language} {count}");
        }
    }

    #[test]
    fn operands_come_from_the_count_as_shown() {
        assert_eq!(shown(1.0), "1");
        assert_eq!(shown(1.5), "1.5");
        assert_eq!(shown(1.25), "1.25");
        assert_eq!(shown(1.0004), "1");
        assert_eq!(shown(0.1 + 0.2), "0.3");
        assert_eq!(shown(1000.0), "1000");
        let large = Operands::of(&shown(1e20));
        assert_eq!((large.i % 1_000_000, large.i > 1000, large.v), (0, true, 0));
        let half = Operands::of("1.5");
        assert_eq!((half.i, half.v, half.f), (1, 1, 5));
    }

    #[test]
    fn germanic_and_romance_languages() {
        for language in ["en", "en-US", "de", "nl", "sv", "sv-SE", "fi", "et", "ur"] {
            expect(language, ONE, &[1.0]);
            expect(language, OTHER, &[0.0, 2.0, 16.0, 100.0, 1.5, 0.1, 1000000.0]);
        }
        for language in ["nb", "no", "nn", "bg", "el", "hu", "tr"] {
            expect(language, ONE, &[1.0]);
            expect(language, OTHER, &[0.0, 2.0, 1.5, 0.1, 11.0]);
        }
        expect("da", ONE, &[1.0, 0.1, 0.5, 1.5]);
        expect("da", OTHER, &[0.0, 2.0, 2.5, 11.0]);
        expect("is", ONE, &[1.0, 21.0, 31.0, 101.0, 0.1, 1.1, 10.1]);
        expect("is", OTHER, &[0.0, 2.0, 11.0, 111.0, 0.11, 1.5, 10.0]);
        expect("es", ONE, &[1.0]);
        expect("es", MANY, &[1000000.0, 2000000.0]);
        expect("es", OTHER, &[0.0, 2.0, 1.5, 1000.0, 1000001.0]);
        for language in ["it", "ca", "pt-PT", "pt_pt"] {
            expect(language, ONE, &[1.0]);
            expect(language, MANY, &[1000000.0]);
            expect(language, OTHER, &[0.0, 2.0, 1.5, 0.5, 100.0]);
        }
        for language in ["fr", "pt", "pt-BR"] {
            expect(language, ONE, &[0.0, 1.0, 1.5, 0.5]);
            expect(language, MANY, &[1000000.0, 3000000.0]);
            expect(language, OTHER, &[2.0, 2.5, 100.0, 1000001.0]);
        }
    }

    #[test]
    fn slavic_and_baltic_languages() {
        for language in ["ru", "uk", "be"] {
            expect(language, ONE, &[1.0, 21.0, 31.0, 101.0, 1001.0]);
            expect(language, FEW, &[2.0, 3.0, 4.0, 22.0, 24.0, 102.0]);
            expect(language, MANY, &[0.0, 5.0, 11.0, 12.0, 14.0, 20.0, 100.0, 111.0]);
            expect(language, OTHER, &[1.5, 0.1, 2.5]);
        }
        expect("pl", ONE, &[1.0]);
        expect("pl", FEW, &[2.0, 4.0, 22.0, 104.0]);
        expect("pl", MANY, &[0.0, 5.0, 11.0, 12.0, 14.0, 21.0, 100.0]);
        expect("pl", OTHER, &[1.5, 0.1]);
        for language in ["cs", "sk"] {
            expect(language, ONE, &[1.0]);
            expect(language, FEW, &[2.0, 3.0, 4.0]);
            expect(language, MANY, &[1.5, 0.1, 10.5]);
            expect(language, OTHER, &[0.0, 5.0, 22.0, 100.0]);
        }
        expect("lt", ONE, &[1.0, 21.0, 101.0]);
        expect("lt", FEW, &[2.0, 9.0, 22.0, 102.0]);
        expect("lt", MANY, &[0.1, 1.5, 2.25]);
        expect("lt", OTHER, &[0.0, 10.0, 11.0, 19.0, 20.0, 111.0]);
        expect("lv", ZERO, &[0.0, 10.0, 11.0, 19.0, 20.0, 100.0, 0.11, 1.15]);
        expect("lv", ONE, &[1.0, 21.0, 101.0, 0.1, 1.1, 0.21, 1.01]);
        expect("lv", OTHER, &[2.0, 9.0, 22.0, 0.2, 1.5, 0.22]);
        for language in ["hr", "sr", "bs"] {
            expect(language, ONE, &[1.0, 21.0, 0.1, 1.1, 2.21]);
            expect(language, FEW, &[2.0, 4.0, 22.0, 0.2, 1.4, 2.32]);
            expect(language, OTHER, &[0.0, 5.0, 11.0, 12.0, 14.0, 0.5, 1.11, 1.12]);
        }
        expect("sl", ONE, &[1.0, 101.0, 201.0]);
        expect("sl", TWO, &[2.0, 102.0]);
        expect("sl", FEW, &[3.0, 4.0, 103.0, 0.5, 1.5]);
        expect("sl", OTHER, &[0.0, 5.0, 11.0, 100.0]);
        expect("ro", ONE, &[1.0]);
        expect("ro", FEW, &[0.0, 2.0, 19.0, 101.0, 119.0, 1.5]);
        expect("ro", OTHER, &[20.0, 99.0, 120.0]);
    }

    #[test]
    fn semitic_celtic_and_other_languages() {
        expect("ar", ZERO, &[0.0]);
        expect("ar", ONE, &[1.0]);
        expect("ar", TWO, &[2.0]);
        expect("ar", FEW, &[3.0, 10.0, 103.0, 110.0]);
        expect("ar", MANY, &[11.0, 26.0, 99.0, 111.0]);
        expect("ar", OTHER, &[100.0, 101.0, 102.0, 200.0, 0.5, 1.5]);
        expect("he", ONE, &[1.0, 0.5, 0.1]);
        expect("he", TWO, &[2.0]);
        expect("he", OTHER, &[0.0, 3.0, 10.0, 20.0, 1.5]);
        expect("ga", ONE, &[1.0]);
        expect("ga", TWO, &[2.0]);
        expect("ga", FEW, &[3.0, 6.0]);
        expect("ga", MANY, &[7.0, 10.0]);
        expect("ga", OTHER, &[0.0, 11.0, 100.0, 1.5]);
        expect("cy", ZERO, &[0.0]);
        expect("cy", ONE, &[1.0]);
        expect("cy", TWO, &[2.0]);
        expect("cy", FEW, &[3.0]);
        expect("cy", MANY, &[6.0]);
        expect("cy", OTHER, &[4.0, 5.0, 7.0, 100.0, 1.5]);
        for language in ["hi", "fa"] {
            expect(language, ONE, &[0.0, 1.0, 0.5]);
            expect(language, OTHER, &[2.0, 1.5, 10.0]);
        }
        for language in ["ja", "zh", "zh-Hant-TW", "ko", "th", "vi", "id", "ms", "tlh", ""] {
            expect(language, OTHER, &[0.0, 1.0, 2.0, 1.5, 100.0]);
        }
    }

    #[test]
    fn odd_counts_are_read_as_their_size() {
        expect("ru", ONE, &[-1.0, -21.0]);
        expect("en", ONE, &[-1.0, 1.0004]);
        expect("ar", OTHER, &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
        assert_eq!(categories("ru"), [ONE, FEW, MANY, OTHER]);
        assert_eq!(categories("ja"), [OTHER]);
    }

    #[test]
    fn counts_show_with_the_languages_decimal_separator() {
        assert_eq!(count_text("en", 1500.0), "1500");
        assert_eq!(count_text("en", 2.5), "2.5");
        assert_eq!(count_text("sv-SE", 2.5), "2,5");
        assert_eq!(count_text("de", 0.125), "0,125");
        assert_eq!(count_text("ar", 2.5), "2.5");
        assert_eq!(count_text("fr", -3.0), "-3");
        assert_eq!(count_text("fr", -0.0001), "0");
        assert_eq!(count_text("en", 1.0004), "1");
        assert_eq!(count_text("en", f64::INFINITY), "inf");
    }
}
