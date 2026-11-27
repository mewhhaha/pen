//! The player's language here (`pen_runtime::locale`): the game's string
//! tables beside its assets, `text.json` (the base) and `locales/TAG.json`
//! (a checkout's `src/`, a package's `game/`), shown in PEN_LOCALE's
//! language, else the system's (`platform::locale`), else the base's.
use std::path::Path;

use pen_runtime::locale::{LocaleService, Locales};

/// Finds the sorted translation tags beside the base table in `root`.
#[cfg(feature = "compiler")]
pub fn locale_tags(root: &Path) -> Vec<String> {
    let Ok(source) = root.canonicalize() else { return Vec::new() };
    let Ok(entries) = std::fs::read_dir(root.join("locales")) else { return Vec::new() };
    let mut tags: Vec<_> = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let tag = path.file_stem()?.to_str()?;
            if tag == "index" || path.extension()? != "json" {
                return None;
            }
            let file = path.canonicalize().ok()?;
            (file.starts_with(&source) && file.is_file()).then(|| tag.to_owned())
        })
        .collect();
    tags.sort();
    tags
}

/// Copies the string tables beside `assets` into `into`, with an index for
/// browser hosts to discover the translations.
#[cfg(feature = "compiler")]
pub fn ship_tables(assets: &Path, into: &Path) -> std::io::Result<()> {
    let Some(root) = assets.parent() else { return Ok(()) };
    if !root.join("text.json").is_file() {
        return Ok(());
    }
    std::fs::copy(root.join("text.json"), into.join("text.json"))?;
    let tags = locale_tags(root);
    let directory = into.join("locales");
    std::fs::create_dir_all(&directory)?;
    for tag in &tags {
        let file = format!("{tag}.json");
        std::fs::copy(root.join("locales").join(&file), directory.join(file))?;
    }
    let index = std::fs::File::create(directory.join("index.json"))?;
    serde_json::to_writer(index, &tags)?;
    Ok(())
}

/// The language the game shows in.
pub fn wanted() -> String {
    std::env::var("PEN_LOCALE").ok().filter(|tag| !tag.is_empty()).or_else(super::platform::locale).unwrap_or_default()
}

/// The locale service for the game whose assets are in `assets`, or None
/// for a game without string tables (its requests are answered unknown, and
/// it keeps its own words). Invalid translations are reported and skipped.
pub fn service(assets: &Path) -> Option<LocaleService> {
    let root = assets.parent()?;
    let base = root.join("text.json");
    if !base.is_file() {
        return None;
    }
    let loaded = Locales::load(&base, "en", &root.join("locales"));
    for error in loaded.errors {
        eprintln!("the game's string tables: {error}");
    }
    let service = LocaleService::new(loaded.locales?, &wanted());
    let locale = service.locale();
    if !locale.tag().is_empty() && std::env::var_os("PEN_SERVICES_LOG").is_some() {
        eprintln!("locale: {} (right to left: {})", locale.tag(), locale.right_to_left());
    }
    Some(service)
}

#[cfg(test)]
mod tests {
    use pen_protocol::{GameEvent, ServiceRequest, locale_op, service, service_status};
    use pen_runtime::services::{Services, code_points};

    use super::*;

    #[cfg(feature = "compiler")]
    #[test]
    fn ships_the_base_and_an_index_of_the_copied_translations() {
        let root = crate::store::scratch("locale-shipping");
        let source = root.join("src");
        let site = root.join("site");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::create_dir_all(&site).unwrap();
        ship_tables(&source.join("assets"), &site).unwrap();
        assert!(!site.join("locales").exists());

        std::fs::write(source.join("text.json"), r#"{"title":"Continue"}"#).unwrap();
        ship_tables(&source.join("assets"), &site).unwrap();
        assert_eq!(std::fs::read_to_string(site.join("locales/index.json")).unwrap(), "[]");

        std::fs::create_dir_all(source.join("locales/dir.json")).unwrap();
        std::fs::write(source.join("locales/sv.json"), r#"{"title":"Fortsätt"}"#).unwrap();
        std::fs::write(source.join("locales/ar.json"), "{}").unwrap();
        std::fs::write(source.join("locales/index.json"), "[]").unwrap();
        std::fs::write(source.join("locales/notes.txt"), "notes").unwrap();
        ship_tables(&source.join("assets"), &site).unwrap();
        assert_eq!(std::fs::read_to_string(site.join("locales/index.json")).unwrap(), r#"["ar","sv"]"#);
        let locales = Locales::load(&site.join("text.json"), "en", &site.join("locales")).locales.unwrap();
        assert_eq!(locales.tags().collect::<Vec<_>>(), ["ar", "sv"]);
        assert_eq!(locales.select("sv").text("title", None).unwrap().spoken, "Fortsätt");
        assert_eq!(locales.select("en").text("title", None).unwrap().spoken, "Continue");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn answers_from_the_tables_beside_the_assets() {
        let root = std::env::temp_dir().join(format!("pen-locale-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("assets")).unwrap();
        std::fs::create_dir_all(root.join("locales")).unwrap();
        std::fs::write(
            root.join("text.json"),
            r#"{"resume": "Continue", "gnorps": {"one": "{count} gnorp", "other": "{count} gnorps"}}"#,
        )
        .unwrap();
        std::fs::write(
            root.join("locales/sv.json"),
            r#"{"resume": "Fortsätt", "gnorps": {"one": "{count} gnorp", "other": "{count} gnorper"}}"#,
        )
        .unwrap();
        std::fs::write(root.join("locales/ar.json"), "{").unwrap();
        assert!(service(&root.join("assets")).is_some());
        let locales = Locales::load(&root.join("text.json"), "en", &root.join("locales")).locales.unwrap();
        let mut services: Services = Services::new();
        services.register(Box::new(LocaleService::new(locales, "sv-SE")));
        let ask = |id, key: &str, count: Option<f32>| ServiceRequest {
            request: id,
            service: service::LOCALE,
            operation: locale_op::TEXT,
            a: count.unwrap_or(0.0),
            b: f32::from(u8::from(count.is_some())),
            value_count: u32::try_from(key.chars().count()).unwrap(),
            values: code_points(key),
            ..ServiceRequest::default()
        };
        services.answer(&[ask(1, "resume", None), ask(2, "gnorps", Some(3.0)), ask(3, "missing", None)]);
        let answers = services.take_answers();
        let text = |id| -> String {
            let values: Vec<f32> = answers
                .iter()
                .filter_map(|event| match event {
                    GameEvent::ServiceData { request, a, b, c } if *request == id => Some([*a, *b, *c]),
                    _ => None,
                })
                .flatten()
                .collect();
            pen_runtime::services::text(&values).trim_end_matches('\0').to_owned()
        };
        assert!(text(1).starts_with("Fortsätt"), "{}", text(1));
        assert!(text(2).starts_with("3 gnorper"), "{}", text(2));
        assert!(answers.contains(&GameEvent::ServiceAnswer {
            request: 3,
            status: service_status::UNAVAILABLE,
            value: 0.0,
            length: 0
        }));
        // No tables, no service.
        assert!(service(&root.join("nowhere/assets")).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }
    /// The checkout's own tables (`../gdev`): nothing a translator must fix,
    /// and every glyph they need, shaped, in the font its `src/font.sq`
    /// says it has.
    #[test]
    fn the_games_tables_have_no_errors_and_the_font_has_their_glyphs() {
        let checkout = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../gdev");
        let Ok(font) = std::fs::read_to_string(checkout.join("src/font.sq")) else {
            crate::fixtures::missing("no ../gdev checkout");
            return;
        };
        let mut covered: std::collections::HashSet<char> = (32u8..127).map(char::from).collect();
        for line in font.lines() {
            if let Some((code, _)) = line.trim().split_once(" => $metrics.Glyph")
                && let Some(character) = code.parse::<u32>().ok().and_then(char::from_u32)
            {
                covered.insert(character);
            }
        }
        let source = checkout.join("src");
        let locales = Locales::load(&source.join("text.json"), "en", &source.join("locales")).locales.unwrap();
        assert!(locales.tags().count() >= 3, "{:?}", locales.tags().collect::<Vec<_>>());
        let problems = locales.check(&|character| covered.contains(&character) || character == '\n');
        let errors: Vec<String> = problems
            .iter()
            .filter(|problem| problem.severity() == pen_runtime::locale::Severity::Error)
            .map(ToString::to_string)
            .collect();
        assert!(errors.is_empty(), "{errors:#?}");
        // The menu's words are translated into each language.
        for tag in locales.tags() {
            let locale = locales.select(tag);
            let resume = locale.text("resume", None).unwrap();
            assert!(
                !resume.text.is_empty() && resume.text.chars().all(|character| covered.contains(&character)),
                "{tag}"
            );
        }
        assert!(locales.select("ar").text("resume", None).unwrap().right_to_left);
        assert_eq!(locales.select("ru").text("gnorps_counted", Some(21.0)).unwrap().text, "21 гнорп");
        assert_eq!(locales.select("ru").text("gnorps_counted", Some(12.0)).unwrap().text, "12 гнорпов");
        assert_eq!(locales.select("en-GB").text("gnorps_counted", Some(1.0)).unwrap().text, "1 gnorp");
    }
}
