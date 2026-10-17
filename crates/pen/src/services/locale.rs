//! The player's language here (`pen_runtime::locale`): the game's string
//! tables beside its assets, `text.json` (the base) and `locales/TAG.json`
//! (a checkout's `src/`, a package's `game/`), shown in PEN_LOCALE's
//! language, else the system's (`platform::locale`), else the base's.
use std::path::Path;

use pen_runtime::locale::{LocaleService, Locales};

/// Copies the string tables beside `assets` (a checkout's `src/`) into
/// `into`, beside the bundle's own assets, for a package or bundle to
/// answer from: `text.json`, and `locales/` when there is one.
pub fn ship_tables(assets: &Path, into: &Path) -> std::io::Result<()> {
    let Some(root) = assets.parent() else { return Ok(()) };
    if !root.join("text.json").is_file() {
        return Ok(());
    }
    std::fs::copy(root.join("text.json"), into.join("text.json"))?;
    let Ok(entries) = std::fs::read_dir(root.join("locales")) else { return Ok(()) };
    std::fs::create_dir_all(into.join("locales"))?;
    for entry in entries {
        let path = entry?.path();
        if path.extension().is_some_and(|extension| extension == "json") {
            std::fs::copy(&path, into.join("locales").join(path.file_name().unwrap_or_default()))?;
        }
    }
    Ok(())
}

/// The language the game shows in.
pub fn wanted() -> String {
    std::env::var("PEN_LOCALE").ok().filter(|tag| !tag.is_empty()).or_else(super::platform::locale).unwrap_or_default()
}

/// The locale service for the game whose assets are in `assets`, or None
/// for a game without string tables (its requests are answered unknown, and
/// it keeps its own words). Tables that do not read are reported, and the
/// game keeps its words then too.
pub fn service(assets: &Path) -> Option<LocaleService> {
    let root = assets.parent()?;
    let base = root.join("text.json");
    if !base.is_file() {
        return None;
    }
    match Locales::load(&base, "en", &root.join("locales")) {
        Ok(locales) => {
            let service = LocaleService::new(locales, &wanted());
            let locale = service.locale();
            if !locale.tag().is_empty() && std::env::var_os("PEN_SERVICES_LOG").is_some() {
                eprintln!("locale: {} (right to left: {})", locale.tag(), locale.right_to_left());
            }
            Some(service)
        }
        Err(error) => {
            eprintln!("the game's string tables: {error:#}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use pen_protocol::{GameEvent, ServiceRequest, locale_op, service, service_status};
    use pen_runtime::services::{Services, code_points};

    use super::*;

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
        let locales = Locales::load(&root.join("text.json"), "en", &root.join("locales")).unwrap();
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
    /// and every glyph they need, shaped, in the font its `src/font.blot`
    /// says it has.
    #[test]
    fn the_games_tables_have_no_errors_and_the_font_has_their_glyphs() {
        let checkout = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../gdev");
        let Ok(font) = std::fs::read_to_string(checkout.join("src/font.blot")) else {
            crate::fixtures::missing("no ../gdev checkout");
            return;
        };
        let mut covered: std::collections::HashSet<char> = (32u8..127).map(char::from).collect();
        for line in font.lines() {
            if let Some((code, _)) = line.trim().split_once(" => #metrics.Glyph")
                && let Some(character) = code.parse::<u32>().ok().and_then(char::from_u32)
            {
                covered.insert(character);
            }
        }
        let source = checkout.join("src");
        let locales = Locales::load(&source.join("text.json"), "en", &source.join("locales")).unwrap();
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
