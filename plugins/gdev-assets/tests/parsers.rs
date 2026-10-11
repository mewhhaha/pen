//! Each of gdev's parsers on small checkouts: what it makes of a file, and
//! the errors asset_parsers.ts reports. tests/typescript.rs checks these and
//! more against the TypeScript itself; here they are spelled out, with what
//! differs from it.
mod common;

use common::{Memory, modules, program, source};
use pen_plugin_gdev_assets::{Program, generate};

const CATALOG: (&str, &str) = ("src/assets/catalog.json", r#"{"meshes": [], "materials": [], "sounds": []}"#);
const VERTEX: &str = "\n@vertex fn main() -> @builtin(position) vec4f { return vec4f(0.0); }\n";

/// The error the program at src/main.blot fails with: its file and message.
fn error(files: &[(&str, &str)]) -> (String, String) {
    let error = modules("src/main.blot", files).expect_err("an error");
    (error.file, error.message)
}

/// The module a shader makes, or why it does not.
fn shader(code: &str) -> Result<String, String> {
    let code = format!("{code}{VERTEX}");
    let generated = modules("src/main.blot", &[CATALOG, ("src/assets/case.wgsl", &code)]).map_err(|error| {
        assert_eq!(error.file, "/checkout/src/assets/case.wgsl");
        error.message
    })?;
    Ok(source(&generated, "/checkout/src/assets/case.wgsl.blot").to_string())
}

#[test]
fn catalog_names_each_file_by_its_index() {
    let catalog =
        r#"{"meshes": ["rock_menhir.mesh.json", "cube.mesh.json"], "sounds": ["blip.wav"], "textures": ["x.png"]}"#;
    let generated = modules("src/main.blot", &[("src/assets/catalog.json", catalog)]).unwrap();
    assert_eq!(
        source(&generated, "/checkout/src/assets/catalog.json.blot"),
        "type Meshes is data = #Meshes { rock_menhir: U32, cube: U32 }\n\
         type Materials is data = #Materials {  }\n\
         type Sounds is data = #Sounds { blip: U32 }\n\
         const meshes: Meshes = (#Meshes { rock_menhir: 0, cube: 1 })\n\
         const materials: Materials = (#Materials {  })\n\
         const sounds: Sounds = (#Sounds { blip: 0 })\n"
    );
}

#[test]
fn catalog_refuses_two_files_of_one_name_and_names_blot_cannot_have() {
    let file = "/checkout/src/assets/catalog.json".to_string();
    let shared = r#"{"meshes": ["a.mesh.json", "a.json"]}"#;
    assert_eq!(
        error(&[("src/assets/catalog.json", shared)]),
        (file.clone(), "catalog meshes: two files share a name".to_string())
    );
    let digit = r#"{"sounds": ["1st.wav"]}"#;
    assert_eq!(
        error(&[("src/assets/catalog.json", digit)]),
        (file.clone(), "Invalid Blot value/field name: 1st".to_string())
    );
    assert_eq!(
        error(&[("src/assets/catalog.json", "{ meshes: [] }")]),
        (file, "Expected property name or '}' in JSON at position 2 (line 1 column 3)".to_string())
    );
}

#[test]
fn text_is_code_points() {
    let words = r#"{"title": "Åsa", "menu": ["Go", ""]}"#;
    let generated = modules("src/main.blot", &[CATALOG, ("src/text.json", words)]).unwrap();
    assert_eq!(
        source(&generated, "/checkout/src/text.json.blot"),
        "type Text is data = #Text { title: (Array U32), menu: (Array (Array U32)) }\n\
         const text: Text = (#Text { title: #[197, 115, 97], menu: #[#[71, 111], #[]] })\n"
    );
}

#[test]
fn text_refuses_a_glyph_the_font_lacks() {
    let file = "/checkout/src/text.json".to_string();
    assert_eq!(
        error(&[CATALOG, ("src/text.json", r#"{"hello": "héllo ✓ ✓ ♥"}"#)]),
        (file.clone(), "text hello: the font has no ✓♥".to_string())
    );
    assert_eq!(
        error(&[CATALOG, ("src/text.json", r#"{"menu": ["ok", "naïve"]}"#)]),
        (file.clone(), "text menu: the font has no ï".to_string())
    );
    assert_eq!(
        error(&[CATALOG, ("src/text.json", r#"{"Title": "x"}"#)]),
        (file, "Invalid Blot value/field name: Title".to_string())
    );
}

#[test]
fn tunables_are_a_record_and_declarations() {
    let tunables = r#"{"about": "x", "tunables": [{"name": "camera.zoom_speed", "default": 0.5, "min": 0, "max": 1, "step": 0.1, "doc": "Zoom ✓"}]}"#;
    let generated = modules("src/main.blot", &[CATALOG, ("src/tunables.json", tunables)]).unwrap();
    assert_eq!(
        source(&generated, "/checkout/src/tunables.json.blot"),
        "type Tunables is data = #Tunables { camera_zoom_speed: (U32, F32, F32, F32) }\n\
         type Declaration is data = #Declaration { name: (Array U32), id: U32, initial: F32, least: F32, most: F32, doc: (Array U32) }\n\
         const tunables: Tunables = (#Tunables { camera_zoom_speed: (11175412, 0.5, 0.0, 1.0) })\n\
         const declarations: (Array Declaration) = #[(#Declaration { name: #[99, 97, 109, 101, 114, 97, 46, 122, 111, 111, 109, 95, 115, 112, 101, 101, 100], id: 11175412, initial: 0.5, least: 0.0, most: 1.0, doc: #[90, 111, 111, 109, 32, 63] })]\n"
    );
}

#[test]
fn tunables_say_what_is_wrong() {
    let cases = [
        (r#"{"tunable": []}"#, r#"tunables.json: expected { "tunables": [...] }"#),
        (r#"{"tunables": [3]}"#, "tunables.json: tunable 0 is not an object"),
        (
            r#"{"tunables": [{"name": "Camera"}]}"#,
            r#"tunables.json: tunable 0 needs a name of lowercase words joined by dots, such as "camera.zoom_speed""#,
        ),
        (
            r#"{"tunables": [{"name": "a", "default": 0, "min": 0, "max": 1, "low": 0}]}"#,
            "tunables.json: a: unknown low (a tunable has name, default, min, max, step, doc)",
        ),
        (
            r#"{"tunables": [{"name": "a", "default": 1e39, "min": 0, "max": 1}]}"#,
            "tunables.json: a: default must be a finite number",
        ),
        (
            r#"{"tunables": [{"name": "a", "default": 1, "min": 1, "max": 1.00000001}]}"#,
            "tunables.json: a: min must be below max",
        ),
        (
            r#"{"tunables": [{"name": "a", "default": 0.30000000000000004, "min": 0.5, "max": 1e21}]}"#,
            "tunables.json: a: its default 0.30000000000000004 is outside 0.5..1e+21",
        ),
        (
            r#"{"tunables": [{"name": "a", "default": 0, "min": 0, "max": 1, "step": 0}]}"#,
            "tunables.json: a: step must be positive",
        ),
        (
            r#"{"tunables": [{"name": "a", "default": 0, "min": 0, "max": 1, "doc": 3}]}"#,
            "tunables.json: a: doc must be text",
        ),
        (
            r#"{"tunables": [{"name": "a.b", "default": 0, "min": 0, "max": 1}, {"name": "a_b", "default": 0, "min": 0, "max": 1}]}"#,
            "tunables.json: a_b: reads as a_b, as a.b does; rename one",
        ),
        (
            r#"{"tunables": [{"name": "t858", "default": 0, "min": 0, "max": 1}, {"name": "t8662", "default": 0, "min": 0, "max": 1}]}"#,
            "tunables.json: t8662: its id 6937671 is t858's too; rename one",
        ),
        (
            r#"{"tunables": [{"name": "type", "default": 0, "min": 0, "max": 1}]}"#,
            "Invalid Blot value/field name: type",
        ),
    ];
    for (tunables, message) in cases {
        assert_eq!(
            error(&[CATALOG, ("src/tunables.json", tunables)]),
            ("/checkout/src/tunables.json".to_string(), message.to_string()),
            "{tunables}"
        );
    }
}

#[test]
fn shaders_give_their_parameters_and_scalar_constants() {
    let code = "struct Parameters { width: f32, height: f32 }\n\
                const scale = 0.25;\nconst offset = -2.05;\nconst count = 9u;\nconst on = true;\n\
                const whole = 3;\nconst signed = 4i;\nconst tint = vec3f(1.0, 0.5, 0.0);\nconst half = 1.0 / 2.0;";
    assert_eq!(
        shader(code).unwrap(),
        "type Parameters is data = #Parameters { width: F32, height: F32 }\n\
         const scale: F32 = 0.25\nconst offset: F32 = (-2.049999952316284)\nconst count: U32 = 9\n\
         const on: Bool = #True\nconst half: F32 = 0.5\n"
    );
    assert_eq!(shader("struct Other { x: u32 }").unwrap(), "");
}

#[test]
fn shaders_refuse_what_blot_cannot_take() {
    let parameters = "Parameters must be up to four f32s, a draw's parameters";
    assert_eq!(shader("struct Parameters { a: f32, b: f32, c: f32, d: f32, e: f32 }").unwrap_err(), parameters);
    assert_eq!(shader("struct Parameters { a: vec2f }").unwrap_err(), parameters);
    assert_eq!(shader("const SCALE = 2.0;").unwrap_err(), "Invalid Blot value/field name: SCALE");
    assert_eq!(shader("const huge = 1e39;").unwrap_err(), "Asset number does not fit finite F32");
}

#[test]
fn a_shader_that_does_not_parse_says_its_line() {
    let message = shader("const a = 1.0;\nfn broken( {").unwrap_err();
    assert!(message.ends_with(". Line: 2"), "{message}");
}

/// Where naga, which reads WGSL as the standard says, and wgsl_reflect part
/// ways (tests/fixtures/cases.json has wgsl_reflect's side).
#[test]
fn shaders_as_naga_reads_them() {
    // wgsl_reflect left out a typed constant with an integer literal.
    assert_eq!(shader("const a: u32 = 3;\nconst b: f32 = 2;").unwrap(), "const a: U32 = 3\nconst b: F32 = 2.0\n");
    // wgsl_reflect rounded each literal to f32 first, and made this 0.
    assert_eq!(shader("const a = 16777217.0 - 16777216.0;").unwrap(), "const a: F32 = 1.0\n");
    // wgsl_reflect left out a constant that uses a later one.
    assert_eq!(shader("const b = a * 2.0;\nconst a = 1.0;").unwrap(), "const b: F32 = 2.0\nconst a: F32 = 1.0\n");
    // wgsl_reflect read a hexadecimal float as 0.
    assert_eq!(shader("const h = 0x1p4;").unwrap(), "const h: F32 = 16.0\n");
    // wgsl_reflect only parsed; naga also resolves names as it makes the
    // module.
    let message = shader("const a = 1.0;\nfn f() -> f32 { return missing; }").unwrap_err();
    assert!(message.ends_with(". Line: 2"), "{message}");
    // Neither validates the module, as wgpu does.
    assert_eq!(shader("const a = 1.0;\nfn f() -> f32 { return 1u; }").unwrap(), "const a: F32 = 1.0\n");
}

#[test]
fn abstract_integers_are_left_out_as_wgsl_reflect_left_them() {
    let code = "const a = 3;\nconst b = -3;\nconst c = 3000000000;\nconst d = 5000000000;\nconst e = -3000000000;\n\
                const f = 5e9;\nconst g = vec2(1e39, 0.0);\nconst h = array(1, 2);";
    assert_eq!(shader(code).unwrap(), "const f: F32 = 5000000000.0\n");
}

#[test]
fn a_program_beside_src_imports_its_files_and_its_own() {
    let files = [
        ("src/assets/catalog.json", r#"{"sounds": ["a.wav"]}"#),
        ("src/text.json", r#"{"game": "Game"}"#),
        ("src/assets/a.wgsl", "const a = 1.0;"),
        ("tools/text.json", r#"{"tool": "Tool"}"#),
        ("tools/assets/b.wgsl", "const b = 2u;"),
    ];
    let generated = modules("tools/main.blot", &files).unwrap();
    let paths: Vec<&str> = generated.modules.iter().map(|module| module.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "/checkout/src/assets/a.wgsl.blot",
            "/checkout/src/assets/catalog.json.blot",
            "/checkout/src/text.json.blot",
            "/checkout/tools/assets/b.wgsl.blot",
            "/checkout/tools/text.json.blot",
        ]
    );
    // The files read, and where a missing catalog, tunables or shader of
    // either directory's would appear.
    assert_eq!(
        generated.dependencies,
        [
            "/checkout/src",
            "/checkout/src/assets",
            "/checkout/src/assets/a.wgsl",
            "/checkout/src/assets/catalog.json",
            "/checkout/src/text.json",
            "/checkout/tools",
            "/checkout/tools/assets",
            "/checkout/tools/assets/b.wgsl",
            "/checkout/tools/text.json",
        ]
    );
}

#[test]
fn a_program_without_a_catalog_imports_its_own_directory() {
    let files = [
        ("game/text.json", r#"{"word": "Word"}"#),
        ("game/assets/c.wgsl", "const c = true;"),
        ("game/assets/notes.txt", ""),
    ];
    let generated = modules("game/main.blot", &files).unwrap();
    assert_eq!(source(&generated, "/checkout/game/assets/c.wgsl.blot"), "const c: Bool = #True\n");
    assert_eq!(generated.modules.len(), 2);
    // A catalog in src/ would change where the files come from.
    assert_eq!(
        generated.dependencies,
        [
            "/checkout",
            "/checkout/game",
            "/checkout/game/assets",
            "/checkout/game/assets/c.wgsl",
            "/checkout/game/text.json"
        ]
    );
}

#[test]
fn the_plugin_takes_no_settings() {
    let memory = Memory::new("/checkout", [CATALOG]);
    let with = |settings: &str| Program { settings: settings.to_string(), ..program("/checkout", "src/main.blot") };
    assert!(generate(&with(" { } "), &memory).is_ok());
    let error = generate(&with(r#"{"source": "src"}"#), &memory).unwrap_err();
    assert_eq!(error.file, "/checkout/src/main.blot");
    assert_eq!(error.message, r#"the gdev assets plugin takes no settings, but has {"source": "src"}"#);
}
