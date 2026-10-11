//! A Generator, kept between builds, makes a file again when its bytes
//! change, and only then.
mod common;

use common::{Memory, program, source};
use pen_plugin_gdev_assets::Generator;

#[test]
fn a_generator_makes_a_changed_file_again() {
    let catalog = ("src/assets/catalog.json", r#"{"sounds": ["a.wav"]}"#);
    let program = program("/checkout", "src/main.blot");
    let mut generator = Generator::default();
    let text = |words: &str| Memory::new("/checkout", [catalog, ("src/text.json", words)]);

    let first = generator.generate(&program, &text(r#"{"a": "A"}"#)).unwrap();
    assert_eq!(generator.generate(&program, &text(r#"{"a": "A"}"#)).unwrap(), first);

    let changed = generator.generate(&program, &text(r#"{"b": "B"}"#)).unwrap();
    assert_eq!(
        source(&changed, "/checkout/src/text.json.blot"),
        "type Text is data = #Text { b: (Array U32) }\nconst text: Text = (#Text { b: #[66] })\n"
    );
    assert_eq!(
        source(&changed, "/checkout/src/assets/catalog.json.blot"),
        source(&first, "/checkout/src/assets/catalog.json.blot")
    );

    let broken = generator.generate(&program, &text(r#"{"b": "✓"}"#)).unwrap_err();
    assert_eq!(broken.to_string(), "/checkout/src/text.json: text b: the font has no ✓");
    assert_eq!(generator.generate(&program, &text(r#"{"b": "✓"}"#)).unwrap_err(), broken);
    assert_eq!(generator.generate(&program, &text(r#"{"b": "B"}"#)).unwrap(), changed);

    // A file it no longer imports is gone from what it makes.
    let without = generator.generate(&program, &Memory::new("/checkout", [catalog])).unwrap();
    assert_eq!(without.modules.len(), 1);
}
