//! The player's language (`pen_protocol::service::LOCALE`, LocaleOperation
//! in `protocol/schema.json`): the game's strings in the player's language,
//! laid out for the game to draw.
//!
//! The strings come from string tables ([`Table`]). The game's own
//! `src/text.json` is the base table: every key the game asks for, in the
//! base language (English unless the game says otherwise). Beside it,
//! `src/locales/TAG.json` translates it into the language of BCP 47 tag TAG
//! (`sv.json`, `pt-BR.json`), keyed as `text.json` is; a key a translation
//! lacks shows from the base. A string may be one form for each CLDR plural
//! category, chosen by the language's rules for the count the game gives
//! ([`plural`]), with `{count}` filled in as the language writes numbers.
//!
//! The host chooses the table for the tag it is given (the player's choice,
//! or the system's locale) by BCP 47 lookup ([`Locales::select`]): the
//! exact tag, then shorter ones (`zh-Hant-TW`, `zh-Hant`, `zh`), then the
//! base.
//!
//! The host lays the text out because the game cannot: it draws code points
//! one glyph after another, left to right, from an SDF font atlas, without
//! shaping or reordering. So answers come in visual order, Arabic joined
//! into its presentation forms and right-to-left runs reversed ([`display`]),
//! with value 1 when the text reads right to left, for the game to align it.
//! [`Locales::check`] lists what a translator must fix, down to the glyphs
//! a translation needs that the game's font lacks.
//!
//! Answers depend on the tables and the tag only, with no calls to the
//! system, so every machine answers alike; a replay reads them from the
//! recording like any service's.
use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::Context;
use pen_protocol::plural_category::{self, OTHER};
use pen_protocol::{locale_op, service};

use crate::services::{Answer, Request, Service, code_points};

mod display;
mod plural;
mod table;

pub use display::display;
pub use plural::{categories, count_text, plural};
pub use table::{Entry, Table};

/// A string as the game draws it: in visual order, and whether it reads
/// right to left (its first line's direction); and as assistive technology
/// reads it, in reading order and unshaped.
#[derive(Clone, Debug, PartialEq)]
pub struct Shown {
    pub text: String,
    pub right_to_left: bool,
    pub spoken: String,
}

/// The base table and its translations.
#[derive(Clone, Debug, Default)]
pub struct Locales {
    base: Table,
    base_language: String,
    /// The translations by tag, in tag order.
    tables: Vec<(String, Table)>,
}

/// The tables that loaded and the errors encountered while reading them.
#[derive(Debug, Default)]
pub struct LocaleLoad {
    pub locales: Option<Locales>,
    pub errors: Vec<LoadError>,
}

/// A problem reading or parsing a particular locale file or directory.
#[derive(Debug)]
pub struct LoadError {
    pub path: PathBuf,
    source: anyhow::Error,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:#}", self.path.display(), self.source)
    }
}

impl LocaleLoad {
    fn record<T>(&mut self, path: impl Into<PathBuf>, result: anyhow::Result<T>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(source) => {
                self.errors.push(LoadError { path: path.into(), source });
                None
            }
        }
    }
}

/// Tags compare without case and with `_` as `-`.
fn normal(tag: &str) -> String {
    tag.trim().replace('_', "-").to_ascii_lowercase()
}

impl Locales {
    /// The base table in `base_language`, and translations by tag.
    pub fn new(base: Table, base_language: &str, mut tables: Vec<(String, Table)>) -> Locales {
        tables.sort_by(|(one, _), (other, _)| one.cmp(other));
        Locales { base, base_language: base_language.to_owned(), tables }
    }

    /// Parses the base table and translations from JSON held in memory.
    pub fn from_json(base_json: &str, base_language: &str, translations: &[(&str, &str)]) -> LocaleLoad {
        let mut loaded = LocaleLoad::default();
        let base = loaded.record("text.json", Table::parse(base_json));
        let tables = translations
            .iter()
            .filter_map(|&(tag, json)| {
                let table = loaded.record(format!("locales/{tag}.json"), Table::parse(json))?;
                Some((tag.to_owned(), table))
            })
            .collect();
        loaded.locales = base.map(|base| Locales::new(base, base_language, tables));
        loaded
    }

    /// Reads the game's base table and each translation from the locales directory.
    /// The base is required for lookup. Each file's failure is reported independently.
    pub fn load(base: &Path, language: &str, directory: &Path) -> LocaleLoad {
        let read = |path: &Path| -> anyhow::Result<Table> {
            let text = std::fs::read_to_string(path).context("reading file")?;
            Table::parse(&text)
        };
        let mut loaded = LocaleLoad::default();
        let base = loaded.record(base, read(base));
        let entries = match std::fs::read_dir(directory) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            entries => loaded.record(directory, entries.context("reading directory")),
        };
        let mut paths = Vec::new();
        for entry in entries.into_iter().flatten() {
            if let Some(entry) = loaded.record(directory, entry.context("reading directory entry")) {
                paths.push(entry.path());
            }
        }
        paths.sort();
        let mut tables = Vec::new();
        for path in paths {
            // Exported sites reserve index.json for translation discovery.
            if path.extension().is_none_or(|extension| extension != "json")
                || path.file_name().is_some_and(|name| name == "index.json")
                || path.is_dir()
            {
                continue;
            }
            if let Some(tag) = path.file_stem().and_then(|stem| stem.to_str())
                && let Some(table) = loaded.record(&path, read(&path))
            {
                tables.push((tag.to_owned(), table));
            }
        }
        loaded.locales = base.map(|base| Locales::new(base, language, tables));
        loaded
    }

    /// The tags of the translations, in order.
    pub fn tags(&self) -> impl Iterator<Item = &str> {
        self.tables.iter().map(|(tag, _)| tag.as_str())
    }

    /// The language for BCP 47 tag `wanted`: the translation with that tag,
    /// else the one with the longest prefix of it (`zh-Hant-TW`, `zh-Hant`,
    /// `zh`), else the base. A POSIX locale (`sv_SE.UTF-8`) reads as its tag.
    pub fn select(&self, wanted: &str) -> Locale<'_> {
        let mut wanted = normal(wanted.split(['.', '@']).next().unwrap_or_default());
        loop {
            if let Some((tag, table)) = self.tables.iter().find(|(tag, _)| normal(tag) == wanted) {
                return Locale { tag, table: Some(table), locales: self };
            }
            let Some(cut) = wanted.rfind('-') else { break };
            wanted.truncate(cut);
        }
        Locale { tag: &self.base_language, table: None, locales: self }
    }

    /// What a translator must fix, table by table (the base first) and key
    /// by key: keys the base lacks, plural forms the language never uses or
    /// lacks, placeholders a translation drops or adds, and characters the
    /// game's font lacks (`covered` says which it holds) as the strings are
    /// drawn, Arabic in its presentation forms. The keys each translation
    /// leaves to the base come last, as information ([`Severity`]), for
    /// counting how much of the base it covers.
    pub fn check(&self, covered: &dyn Fn(char) -> bool) -> Vec<Problem> {
        let mut problems = Vec::new();
        let base = std::iter::once((self.base_language.as_str(), &self.base, true));
        let tables = self.tables.iter().map(|(tag, table)| (tag.as_str(), table, false));
        for (tag, table, is_base) in base.chain(tables) {
            let mut report = |key: &str, kind: ProblemKind| {
                problems.push(Problem { table: tag.to_owned(), base: is_base, key: key.to_owned(), kind });
            };
            let used = categories(tag);
            // Every digit and the decimal separator, where a count shows.
            let count = count_text(tag, 9_876_543_210.5);
            for (key, entry) in table.entries() {
                let Some(original) = self.base.get(key) else {
                    report(key, ProblemKind::Unknown);
                    continue;
                };
                let has = entry.categories();
                for category in has.iter().filter(|category| !used.contains(category)) {
                    report(key, ProblemKind::UnusedForm(*category));
                }
                if matches!(entry, Entry::Plural(_)) || matches!(original, Entry::Plural(_)) {
                    for category in used.iter().filter(|category| !has.contains(category)) {
                        report(key, ProblemKind::MissingForm(*category));
                    }
                }
                if !is_base {
                    let (wanted, given) = (original.placeholders(), entry.placeholders());
                    for name in wanted.difference(&given) {
                        report(key, ProblemKind::MissingPlaceholder((*name).to_owned()));
                    }
                    for name in given.difference(&wanted) {
                        report(key, ProblemKind::ExtraPlaceholder((*name).to_owned()));
                    }
                }
                let mut lacking = BTreeSet::new();
                for text in entry.strings() {
                    let (shown, _) = display(&text.replace("{count}", &count));
                    lacking.extend(shown.chars().filter(|char| *char != '\n' && !covered(*char)));
                }
                for char in lacking {
                    report(key, ProblemKind::Uncovered(char));
                }
            }
            if !is_base {
                for (key, _) in self.base.entries().filter(|(key, _)| table.get(key).is_none()) {
                    report(key, ProblemKind::Untranslated);
                }
            }
        }
        problems
    }
}

/// Whether the language of BCP 47 tag `tag` reads right to left: Arabic,
/// Persian, Hebrew, Urdu, Pashto, Yiddish, Dhivehi, Uyghur, Sorani or
/// Sindhi, unless a script subtag after the language says otherwise
/// (`ur-Latn`), or a language written in a right-to-left script (`-Arab`,
/// `-Hebr`, `-Thaa`, `-Syrc`, `-Nkoo`, `-Adlm`, `-Rohg`).
pub fn right_to_left(tag: &str) -> bool {
    let mut subtags = tag.split(['-', '_']);
    let language = subtags.next().unwrap_or_default().to_ascii_lowercase();
    if let Some(script) = subtags.next().filter(|subtag| subtag.len() == 4 && subtag.chars().all(char::is_alphabetic)) {
        return ["arab", "hebr", "thaa", "syrc", "nkoo", "adlm", "rohg"]
            .contains(&script.to_ascii_lowercase().as_str());
    }
    matches!(language.as_str(), "ar" | "fa" | "he" | "iw" | "ur" | "ps" | "yi" | "dv" | "ug" | "ckb" | "sd")
}

/// Substitutes named or positional placeholders, copying argument values verbatim.
/// Unknown placeholders and an unfinished placeholder stay as written.
pub fn format_args(template: &str, args: &[(&str, &str)]) -> String {
    substitute(template, args, None)
}

fn substitute(template: &str, args: &[(&str, &str)], count: Option<&str>) -> String {
    if args.is_empty() && count.is_none() {
        return template.to_owned();
    }
    let mut text = String::with_capacity(template.len());
    let mut open = None;
    let mut copied = 0;
    for (index, brace) in template.match_indices(['{', '}']) {
        if brace == "{" {
            open = Some(index);
        } else if let Some(start) = open.take() {
            let name = &template[start + 1..index];
            let value = count
                .filter(|_| name == "count")
                .or_else(|| args.iter().find_map(|&(key, value)| (key == name).then_some(value)));
            if let Some(value) = value {
                text.push_str(&template[copied..start]);
                text.push_str(value);
                copied = index + 1;
            }
        }
    }
    text.push_str(&template[copied..]);
    text
}

/// A language chosen from [`Locales`]: a translation, or the base.
#[derive(Clone, Copy, Debug)]
pub struct Locale<'a> {
    tag: &'a str,
    table: Option<&'a Table>,
    locales: &'a Locales,
}

impl<'a> Locale<'a> {
    /// Its BCP 47 tag: the translation's, or the base language.
    pub fn tag(&self) -> &'a str {
        self.tag
    }

    pub fn right_to_left(&self) -> bool {
        right_to_left(self.tag)
    }

    /// The plural category of `count` in it.
    pub fn plural(&self, count: f64) -> u32 {
        plural(self.tag, count)
    }

    /// The string for `key`, from its table or else the base, as the game
    /// draws it ([`display`]); none when neither has the key. With a count
    /// it is the form for the count's plural category (or `other`) with
    /// `{count}` filled in, both by the language of the table the string
    /// came from; without one, the `other` form as written.
    pub fn text(&self, key: &str, count: Option<f64>) -> Option<Shown> {
        self.text_with_args(key, count, &[])
    }

    /// Resolves the string and formats its arguments before shaping it.
    /// A supplied count takes precedence over a `count` argument.
    pub fn text_with_args(&self, key: &str, count: Option<f64>, args: &[(&str, &str)]) -> Option<Shown> {
        let (entry, language) = match self.table.and_then(|table| table.get(key)) {
            Some(entry) => (entry, self.tag),
            None => (self.locales.base.get(key)?, self.locales.base_language.as_str()),
        };
        let template = entry.form(count.map_or(OTHER, |count| plural(language, count)));
        let text = match count {
            Some(count) => substitute(template, args, Some(&count_text(language, count))),
            None => format_args(template, args),
        };
        let (shown, right_to_left) = display(&text);
        Some(Shown { text: shown, right_to_left, spoken: text })
    }
}

/// Something [`Locales::check`] found in a table.
#[derive(Clone, Debug, PartialEq)]
pub struct Problem {
    /// The table's tag; the base's is its language.
    pub table: String,
    /// Whether it is in the base table.
    pub base: bool,
    pub key: String,
    pub kind: ProblemKind,
}

/// What is wrong with a table's string.
#[derive(Clone, Debug, PartialEq)]
pub enum ProblemKind {
    /// A key the base lacks: the game never asks for it.
    Unknown,
    /// A plural form for a category (`pen_protocol::plural_category`) the
    /// language never uses.
    UnusedForm(u32),
    /// No form for a plural category the language uses, where the string
    /// (or the base's) has plural forms: its counts show the `other` form.
    MissingForm(u32),
    /// A placeholder the base string has and the translation lacks.
    MissingPlaceholder(String),
    /// A placeholder the translation has and the base string lacks.
    ExtraPlaceholder(String),
    /// A character the game's font lacks, as the string is drawn.
    Uncovered(char),
    /// A key of the base the translation lacks: it shows from the base.
    Untranslated,
}

/// How much a [`Problem`] matters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// The game shows something wrong.
    Error,
    /// Likely a mistake.
    Warning,
    /// Worth counting: untranslated keys.
    Information,
}

impl Problem {
    pub fn severity(&self) -> Severity {
        match self.kind {
            ProblemKind::Unknown | ProblemKind::ExtraPlaceholder(_) | ProblemKind::Uncovered(_) => Severity::Error,
            // A form may say its count in words ("one gnorp", Arabic's dual).
            ProblemKind::MissingPlaceholder(_) | ProblemKind::UnusedForm(_) | ProblemKind::MissingForm(_) => {
                Severity::Warning
            }
            ProblemKind::Untranslated => Severity::Information,
        }
    }
}

impl fmt::Display for Problem {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let name = |category: u32| plural_category::NAMES.get(category as usize).copied().unwrap_or("?");
        let (table, key) = (&self.table, &self.key);
        let base = if self.base { " (base)" } else { "" };
        write!(formatter, "{table}{base} \"{key}\": ")?;
        match &self.kind {
            ProblemKind::Unknown => write!(formatter, "no such key in the base table"),
            ProblemKind::UnusedForm(category) => {
                write!(formatter, "the \"{}\" form is never used in {table}", name(*category))
            }
            ProblemKind::MissingForm(category) => {
                write!(formatter, "no \"{}\" form; those counts show the \"other\" form", name(*category))
            }
            ProblemKind::MissingPlaceholder(placeholder) => {
                write!(formatter, "lacks {{{placeholder}}}, which the base string has")
            }
            ProblemKind::ExtraPlaceholder(placeholder) => {
                write!(formatter, "has {{{placeholder}}}, which the base string lacks")
            }
            ProblemKind::Uncovered(char) => {
                write!(formatter, "the font lacks U+{:04X} {char:?}", u32::from(*char))
            }
            ProblemKind::Untranslated => write!(formatter, "untranslated"),
        }
    }
}

/// The locale service: strings, the language and plural categories, from
/// [`Locales`] in the language chosen for tag `wanted`.
pub struct LocaleService {
    locales: Locales,
    wanted: String,
}

impl LocaleService {
    pub fn new(locales: Locales, wanted: &str) -> LocaleService {
        LocaleService { locales, wanted: wanted.to_owned() }
    }

    /// Changes the requested language for subsequent answers.
    pub fn set_tag(&mut self, tag: &str) {
        tag.clone_into(&mut self.wanted);
    }

    pub fn wanted(&self) -> &str {
        &self.wanted
    }

    /// The base language followed by the loaded translation tags.
    pub fn available_tags(&self) -> Vec<&str> {
        std::iter::once(self.locales.base_language.as_str()).chain(self.locales.tags()).collect()
    }

    /// The language it answers in.
    pub fn locale(&self) -> Locale<'_> {
        self.locales.select(&self.wanted)
    }
}

impl<Input> Service<Input> for LocaleService {
    fn code(&self) -> u32 {
        service::LOCALE
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        let locale = self.locale();
        let flag = |yes: bool| f32::from(u8::from(yes));
        Some(match request.operation {
            locale_op::TEXT => {
                let count = (request.numbers[1] == 1.0).then(|| f64::from(request.numbers[0]));
                match locale.text(&request.text(), count) {
                    // What is drawn, then, where it differs, a 0 and what is read.
                    Some(shown) if shown.spoken != shown.text => {
                        let mut values = code_points(&shown.text);
                        values.push(0.0);
                        values.extend(code_points(&shown.spoken));
                        Answer { value: flag(shown.right_to_left), values, ..Answer::ok(0.0) }
                    }
                    Some(shown) => Answer { value: flag(shown.right_to_left), ..Answer::text(&shown.text) },
                    None => Answer::unavailable(),
                }
            }
            locale_op::LANGUAGE => Answer { value: flag(locale.right_to_left()), ..Answer::text(locale.tag()) },
            locale_op::PLURAL => Answer::ok(locale.plural(f64::from(request.numbers[0])) as f32),
            _ => Answer::unknown(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::{code_points, text};
    use pen_protocol::plural_category::{FEW, MANY, ONE};
    use pen_protocol::service_status;

    #[test]
    fn parses_in_memory_tables_with_their_language_and_tag() {
        let base = r#"{"title":"Pausa","settings":"Inställningar"}"#;
        let loaded = Locales::from_json(base, "sv", &[("fr", r#"{"title":"Pause"}"#), ("ar", "{}")]);
        assert!(loaded.errors.is_empty());
        let locales = loaded.locales.unwrap();
        assert_eq!(locales.tags().collect::<Vec<_>>(), ["ar", "fr"]);
        assert_eq!(locales.select("de").tag(), "sv");
        assert_eq!(locales.select("fr-CA").text("title", None).unwrap().spoken, "Pause");
        assert_eq!(locales.select("fr").text("settings", None).unwrap().spoken, "Inställningar");
        let broken_base = Locales::from_json("{", "sv", &[]);
        assert!(broken_base.locales.is_none());
        assert_eq!(broken_base.errors[0].path, Path::new("text.json"));
    }

    #[test]
    fn malformed_translation_leaves_base_and_valid_translations_usable() {
        let loaded = Locales::from_json(
            r#"{"title":"Continue","settings":"Settings"}"#,
            "en",
            &[("ar", r#"{"title":"متابعة"}"#), ("sv", "{")],
        );
        let [error] = loaded.errors.as_slice() else { panic!("{:?}", loaded.errors) };
        assert_eq!(error.path, Path::new("locales/sv.json"));
        assert!(error.to_string().contains("not JSON"));
        let locales = loaded.locales.unwrap();
        assert_eq!(locales.tags().collect::<Vec<_>>(), ["ar"]);
        assert_eq!(locales.select("ar").text("title", None).unwrap().spoken, "متابعة");
        assert_eq!(locales.select("ar").text("settings", None).unwrap().spoken, "Settings");
        assert_eq!(locales.select("sv").text("title", None).unwrap().spoken, "Continue");
    }

    #[test]
    fn substitutes_named_and_positional_arguments_once() {
        let args = [("name", "{0}"), ("0", "world"), ("1", ""), ("other", "🌍")];
        assert_eq!(
            format_args("Hello {name}, {0}! {1}{other} {missing} {name} {unfinished", &args),
            "Hello {0}, world! 🌍 {missing} {0} {unfinished"
        );
        assert_eq!(format_args("An unclosed { before {0}", &args), "An unclosed { before world");
        assert_eq!(format_args("Already plain } and {missing}", &[]), "Already plain } and {missing}");
    }

    #[test]
    fn formats_count_and_arguments_before_shaping_each_string() {
        let tables = Locales::from_json(
            r#"{"coins":{"one":"{0} has {count} coin","other":"{0} has {count} coins"},"name":"{name}"}"#,
            "en",
            &[("ar", r#"{"name":"مرحبا {name}"}"#)],
        )
        .locales
        .unwrap();
        let locale = tables.select("ar");
        let args = [("0", "{count}"), ("count", "many")];
        assert_eq!(locale.text_with_args("coins", Some(1.0), &args).unwrap().spoken, "{count} has 1 coin");
        assert_eq!(locale.text_with_args("coins", Some(2.0), &args).unwrap().spoken, "{count} has 2 coins");
        assert_eq!(locale.text_with_args("coins", None, &args).unwrap().spoken, "{count} has many coins");
        assert!(locale.text_with_args("missing", None, &[]).is_none());
        let shown = locale.text_with_args("name", None, &[("name", "علي")]).unwrap();
        assert_eq!(shown.spoken, "مرحبا علي");
        assert_eq!(shown.text, display("مرحبا علي").0);
        assert_ne!(shown.text, shown.spoken);
        assert!(shown.right_to_left);
    }

    #[test]
    fn switching_the_service_tag_updates_its_answers() {
        let mut service = LocaleService::new(locales(), "sv-SE");
        assert_eq!(service.available_tags(), ["en", "ar", "ru", "sv", "zh-Hant"]);
        assert_eq!(service.wanted(), "sv-SE");
        assert_eq!(service.locale().text("resume", None).unwrap().spoken, "Fortsätt");
        service.set_tag("ar");
        assert_eq!(service.wanted(), "ar");
        let answer = ask(&mut service, locale_op::LANGUAGE, [0.0; 4], "");
        assert_eq!(text(&answer.values), "ar");
        assert_eq!(answer.value, 1.0);
        service.set_tag("de-DE");
        assert_eq!(service.locale().tag(), "en");
        assert_eq!(service.locale().text("resume", None).unwrap().spoken, "Continue");
    }

    fn table(json: &str) -> Table {
        Table::parse(json).unwrap()
    }

    fn locales() -> Locales {
        let base = table(
            r#"{"resume": "Continue", "coins": {"one": "{count} coin", "other": "{count} coins"},
                "greeting": "Hello, {name}!", "speed": "{count} m/s", "credits": "Credits"}"#,
        );
        let tables = vec![
            (
                "sv".to_owned(),
                table(r#"{"resume": "Fortsätt", "coins": {"one": "{count} mynt", "other": "{count} mynt"}}"#),
            ),
            (
                "ru".to_owned(),
                table(
                    r#"{"coins": {"one": "{count} монета", "few": "{count} монеты", "many": "{count} монет", "other": "{count} монеты"}}"#,
                ),
            ),
            ("ar".to_owned(), table(r#"{"resume": "استمر", "coins": {"other": "{count} عملة"}}"#)),
            ("zh-Hant".to_owned(), table(r#"{"resume": "繼續"}"#)),
        ];
        Locales::new(base, "en", tables)
    }

    #[test]
    fn languages_are_chosen_by_bcp_47_lookup() {
        let locales = locales();
        assert_eq!(locales.tags().collect::<Vec<_>>(), ["ar", "ru", "sv", "zh-Hant"]);
        for (wanted, tag) in [
            ("sv", "sv"),
            ("sv-SE", "sv"),
            ("SV_se", "sv"),
            ("sv_SE.UTF-8", "sv"),
            ("zh-Hant-TW", "zh-Hant"),
            ("zh-hant", "zh-Hant"),
            ("zh", "en"),
            ("zh-Hans", "en"),
            ("de-DE", "en"),
            ("", "en"),
        ] {
            assert_eq!(locales.select(wanted).tag(), tag, "{wanted}");
        }
        assert!(locales.select("ar-EG").right_to_left());
        assert!(!locales.select("sv").right_to_left());
        for (tag, expected) in [("he", true), ("fa-IR", true), ("ur-Latn", false), ("pa-Arab", true), ("sv", false)] {
            assert_eq!(right_to_left(tag), expected, "{tag}");
        }
    }

    #[test]
    fn strings_take_their_plural_form_and_count() {
        let locales = locales();
        let shown = |wanted: &str, key: &str, count: Option<f64>| {
            locales.select(wanted).text(key, count).map(|shown| shown.text)
        };
        assert_eq!(shown("sv", "resume", None).as_deref(), Some("Fortsätt"));
        assert_eq!(shown("sv", "coins", Some(2.5)).as_deref(), Some("2,5 mynt"));
        assert_eq!(shown("en", "coins", Some(1.0)).as_deref(), Some("1 coin"));
        assert_eq!(shown("en", "coins", Some(1.5)).as_deref(), Some("1.5 coins"));
        assert_eq!(shown("en", "coins", None).as_deref(), Some("{count} coins"));
        assert_eq!(shown("ru", "coins", Some(21.0)).as_deref(), Some("21 монета"));
        assert_eq!(shown("ru", "coins", Some(3.0)).as_deref(), Some("3 монеты"));
        assert_eq!(shown("ru", "coins", Some(11.0)).as_deref(), Some("11 монет"));
        assert_eq!(shown("ru", "coins", Some(1.5)).as_deref(), Some("1,5 монеты"));
        // A key only the base has shows in the base language, by its rules.
        assert_eq!(shown("ru", "resume", None).as_deref(), Some("Continue"));
        assert_eq!(shown("ru", "speed", Some(2.5)).as_deref(), Some("2.5 m/s"));
        assert_eq!(shown("ru", "greeting", None).as_deref(), Some("Hello, {name}!"));
        assert_eq!(shown("sv", "missing", None), None);
        let arabic = locales.select("ar").text("coins", Some(3.0)).unwrap();
        assert!(arabic.right_to_left);
        assert!(arabic.text.starts_with("\u{FE94}"), "{:?}", arabic.text);
        assert!(arabic.text.ends_with(" 3"), "{:?}", arabic.text);
    }

    /// A request for the locale service.
    fn ask(service: &mut LocaleService, operation: u32, numbers: [f32; 4], key: &str) -> Answer {
        let values = code_points(key);
        Service::<()>::call(service, &Request { id: 1, operation, numbers, values: &values }).unwrap()
    }

    #[test]
    fn right_to_left_answers_carry_the_text_read_after_the_text_drawn() {
        let base = Table::parse(r#"{"resume": "Continue"}"#).unwrap();
        let arabic = Table::parse(r#"{"resume": "متابعة"}"#).unwrap();
        let mut service = LocaleService::new(Locales::new(base, "en", vec![("ar".into(), arabic)]), "ar");
        let answer = ask(&mut service, locale_op::TEXT, [0.0; 4], "resume");
        let split = answer.values.iter().position(|value| *value == 0.0).unwrap();
        let (drawn, read) = (text(&answer.values[..split]), text(&answer.values[split + 1..]));
        assert_eq!(read, "متابعة");
        assert_eq!(drawn, display("متابعة").0);
        assert_ne!(drawn, read);
        assert_eq!(answer.value, 1.0);
        // Left to right, drawn and read are one.
        let mut english = LocaleService::new(locales(), "en");
        assert!(!ask(&mut english, locale_op::TEXT, [0.0; 4], "resume").values.contains(&0.0));
    }

    #[test]
    fn the_service_answers_text_language_and_plural() {
        let mut swedish = LocaleService::new(locales(), "sv-SE");
        let answer = ask(&mut swedish, locale_op::TEXT, [0.0; 4], "resume");
        assert_eq!((answer.status, answer.value, text(&answer.values)), (service_status::OK, 0.0, "Fortsätt".into()));
        let answer = ask(&mut swedish, locale_op::TEXT, [3.0, 1.0, 0.0, 0.0], "coins");
        assert_eq!(text(&answer.values), "3 mynt");
        // `b` is not 1: no count, so the `other` form as written.
        let answer = ask(&mut swedish, locale_op::TEXT, [3.0, 0.0, 0.0, 0.0], "coins");
        assert_eq!(text(&answer.values), "{count} mynt");
        assert_eq!(ask(&mut swedish, locale_op::TEXT, [0.0; 4], "nothing"), Answer::unavailable());
        let answer = ask(&mut swedish, locale_op::LANGUAGE, [0.0; 4], "");
        assert_eq!((text(&answer.values), answer.value), ("sv".into(), 0.0));
        assert_eq!(ask(&mut swedish, 99, [0.0; 4], ""), Answer::unknown());

        let mut russian = LocaleService::new(locales(), "ru");
        for (count, category) in [(1.0, ONE), (3.0, FEW), (5.0, MANY), (1.5, OTHER)] {
            assert_eq!(ask(&mut russian, locale_op::PLURAL, [count, 0.0, 0.0, 0.0], ""), Answer::ok(category as f32));
        }

        let mut arabic = LocaleService::new(locales(), "ar");
        let answer = ask(&mut arabic, locale_op::TEXT, [0.0; 4], "resume");
        assert_eq!(answer.value, 1.0);
        // Reh final, meem medial, teh medial, seen initial, alef isolated;
        // then a 0 and the text as it reads.
        assert_eq!(answer.values[..6], [0xFEAE, 0xFEE4, 0xFE98, 0xFEB3, 0xFE8D, 0].map(|code| code as f32));
        let answer = ask(&mut arabic, locale_op::LANGUAGE, [0.0; 4], "");
        assert_eq!((text(&answer.values), answer.value), ("ar".into(), 1.0));
        assert_eq!(Service::<()>::code(&arabic), service::LOCALE);
    }

    #[test]
    fn checks_name_the_table_and_key_of_each_problem() {
        let base = table(
            r#"{"coins": {"one": "{count} coin", "other": "{count} coins"}, "greeting": "Hello, {name}!",
                "resume": "Continue", "zero": {"zero": "none", "one": "one", "other": "some"}}"#,
        );
        let tables = vec![
            (
                "ru".to_owned(),
                table(
                    r#"{"coins": {"one": "{count} монета", "two": "{count} монеты", "other": "{count} монет"},
                        "greeting": "Привет, {nmae}!", "resum": "Продолжить"}"#,
                ),
            ),
            ("ar".to_owned(), table(r#"{"resume": "سلام"}"#)),
        ];
        let locales = Locales::new(base, "en", tables);
        // A font with Latin, Cyrillic and the Arabic presentation forms but no seen.
        let covered = |char: char| char.is_ascii() || ('\u{0400}'..='\u{04FF}').contains(&char) || char > '\u{FEB4}';
        let problems: Vec<String> = locales.check(&covered).iter().map(ToString::to_string).collect();
        assert_eq!(
            problems,
            [
                "en (base) \"zero\": the \"zero\" form is never used in en",
                "ar \"resume\": the font lacks U+FEB3 'ﺳ'",
                "ar \"coins\": untranslated",
                "ar \"greeting\": untranslated",
                "ar \"zero\": untranslated",
                "ru \"coins\": the \"two\" form is never used in ru",
                "ru \"coins\": no \"few\" form; those counts show the \"other\" form",
                "ru \"coins\": no \"many\" form; those counts show the \"other\" form",
                "ru \"greeting\": lacks {name}, which the base string has",
                "ru \"greeting\": has {nmae}, which the base string lacks",
                "ru \"resum\": no such key in the base table",
                "ru \"resume\": untranslated",
                "ru \"zero\": untranslated",
            ]
        );
        let severities: Vec<Severity> = locales.check(&covered).iter().map(Problem::severity).collect();
        assert_eq!(severities[..2], [Severity::Warning, Severity::Error]);
        assert_eq!(severities.last(), Some(&Severity::Information));
    }

    #[test]
    fn tables_load_from_the_game_and_its_locales_directory() {
        let root = std::env::temp_dir().join(format!("pen-locale-test-{}", std::process::id()));
        let directory = root.join("locales");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(root.join("text.json"), r#"{"resume": "Continue"}"#).unwrap();
        let missing = Locales::load(&root.join("text.json"), "en", &root.join("nowhere"));
        assert!(missing.errors.is_empty());
        assert_eq!(missing.locales.unwrap().tags().count(), 0);
        std::fs::write(directory.join("pt-BR.json"), r#"{"resume": "Continuar"}"#).unwrap();
        std::fs::write(directory.join("sv.json"), r#"{"resume": "Fortsätt"}"#).unwrap();
        std::fs::write(directory.join("index.json"), r#"["pt-BR","sv"]"#).unwrap();
        std::fs::write(directory.join("notes.txt"), "not a table").unwrap();
        let loaded = Locales::load(&root.join("text.json"), "en", &directory);
        assert!(loaded.errors.is_empty());
        let locales = loaded.locales.unwrap();
        assert_eq!(locales.tags().collect::<Vec<_>>(), ["pt-BR", "sv"]);
        assert_eq!(locales.select("pt-BR").text("resume", None).unwrap().text, "Continuar");
        std::fs::write(directory.join("de.json"), r#"{"resume": 3}"#).unwrap();
        let loaded = Locales::load(&root.join("text.json"), "en", &directory);
        assert_eq!(loaded.locales.unwrap().tags().collect::<Vec<_>>(), ["pt-BR", "sv"]);
        let [error] = loaded.errors.as_slice() else { panic!("{:?}", loaded.errors) };
        assert_eq!(error.path, directory.join("de.json"));
        assert_eq!(
            format!("{error:#}"),
            format!(
                "{}: \"resume\": a value is a string, an object of plural forms or a list of strings, not a number",
                directory.join("de.json").display()
            )
        );
        std::fs::remove_dir_all(&root).unwrap();
    }
}
