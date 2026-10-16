//! Golden-image tests: scripted scenarios run headless on a steady clock
//! after the assets settle, and each shot is compared with a stored image.
//!
//! A scenario file is JSON:
//!
//! ```json
//! {
//!   "size": [960, 640],
//!   "fps": 60,
//!   "threshold": 0.01,
//!   "scenarios": {
//!     "play": { "about": "the rock after Tab", "script": "key=Tab wait=60 shot=play" }
//!   }
//! }
//! ```
//!
//! Each scenario starts a fresh game from the bundle with a fresh data
//! directory (no saves or preferences). Shots land in `OUT/NAME/SHOT.png`,
//! goldens live in `GOLDEN/NAME/SHOT.png`; a shot whose RMSE over the colour
//! channels exceeds the threshold fails and leaves `SHOT.diff.png` beside it.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use blot_host::GuestModule;
use pen_protocol::Catalog;
use serde::Deserialize;

use crate::script;
use crate::session::{Gpu, Options, Paths, Session};

#[derive(Deserialize)]
struct Scenarios {
    #[serde(default = "default_size")]
    size: (u32, u32),
    #[serde(default = "default_fps")]
    fps: f64,
    /// Largest RMSE (0..1 over the colour channels) a shot may differ by.
    #[serde(default = "default_threshold")]
    threshold: f64,
    scenarios: BTreeMap<String, Scenario>,
}

fn default_size() -> (u32, u32) {
    (960, 640)
}
fn default_fps() -> f64 {
    60.0
}
fn default_threshold() -> f64 {
    0.01
}

#[derive(Deserialize)]
struct Scenario {
    #[serde(default)]
    about: String,
    /// Steps as one string or a list of them.
    script: StepList,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StepList {
    One(String),
    Many(Vec<String>),
}

impl StepList {
    fn words(&self) -> Vec<String> {
        match self {
            StepList::One(text) => vec![text.clone()],
            StepList::Many(steps) => steps.clone(),
        }
    }
}

pub struct TestOptions {
    pub file: PathBuf,
    pub golden: PathBuf,
    pub out: PathBuf,
    pub update: bool,
    /// Only scenarios whose names contain one of these.
    pub only: Vec<String>,
    /// Draw on a software adapter (`--software`).
    pub software: bool,
}

/// How far apart two images are: RMSE and the largest difference, both over
/// the colour channels in 0..1, and the share of pixels that differ visibly.
struct Difference {
    rmse: f64,
    largest: f64,
    changed: f64,
}

fn compare(actual: &image::RgbaImage, golden: &image::RgbaImage, diff: &Path) -> Result<Difference> {
    let mut squares = 0.0f64;
    let mut largest = 0u8;
    let mut changed = 0usize;
    let mut picture = image::RgbaImage::new(actual.width(), actual.height());
    for ((a, g), d) in actual.pixels().zip(golden.pixels()).zip(picture.pixels_mut()) {
        let mut pixel_largest = 0u8;
        for channel in 0..3 {
            let delta = a[channel].abs_diff(g[channel]);
            squares += f64::from(delta) * f64::from(delta);
            pixel_largest = pixel_largest.max(delta);
        }
        largest = largest.max(pixel_largest);
        if pixel_largest > 8 {
            changed += 1;
        }
        // Differences in red over a dimmed copy of the golden image.
        let base = (u16::from(g[0]) + u16::from(g[1]) + u16::from(g[2])) / 12;
        let base = u8::try_from(base).expect("765 / 12 fits a byte");
        *d = image::Rgba([base.saturating_add(pixel_largest.saturating_mul(4)), base, base, 255]);
    }
    let pixels = f64::from(actual.width()) * f64::from(actual.height());
    let difference = Difference {
        rmse: (squares / (pixels * 3.0)).sqrt() / 255.0,
        largest: f64::from(largest) / 255.0,
        changed: changed as f64 / pixels,
    };
    if difference.largest > 0.0 {
        picture.save(diff).with_context(|| format!("writing {}", diff.display()))?;
    }
    Ok(difference)
}

/// The game a test runs: its compiled program's directory, its assets and
/// the scene a new game starts from.
pub struct Game<'a> {
    pub directory: &'a Path,
    pub assets: &'a Path,
    pub scene: &'a Path,
}

pub fn run(
    test: TestOptions,
    game: Game,
    engine: wasmtime::Engine,
    module: GuestModule,
    catalog: Catalog,
) -> Result<()> {
    let text = std::fs::read_to_string(&test.file).with_context(|| format!("reading {}", test.file.display()))?;
    let file: Scenarios = serde_json::from_str(&text).with_context(|| format!("parsing {}", test.file.display()))?;
    ensure!(file.size.0 > 0 && file.size.1 > 0, "visual size must not be empty");
    ensure!(file.fps.is_finite() && file.fps > 0.0, "visual fps must be finite and positive");
    ensure!((0.0..=1.0).contains(&file.threshold), "visual threshold must be within 0..1");
    ensure!(!file.scenarios.is_empty(), "visual scenarios must not be empty");
    ensure!(
        test.only.is_empty()
            || file.scenarios.keys().any(|name| test.only.iter().any(|only| name.contains(only.as_str()))),
        "no visual scenario matches --only"
    );
    let mut gpu = Some(Gpu::headless(file.size, game.assets, test.software)?);
    let scratch = std::env::temp_dir().join(format!("pen-visual-{}", std::process::id()));
    let (mut passed, mut failed, mut updated) = (0, Vec::new(), 0);
    for (name, scenario) in &file.scenarios {
        if !test.only.is_empty() && !test.only.iter().any(|only| name.contains(only.as_str())) {
            continue;
        }
        eprintln!("\n== {name}: {}", scenario.about);
        let shots = test.out.join(name);
        let _ = std::fs::remove_dir_all(&shots);
        let data = scratch.join(name);
        let _ = std::fs::remove_dir_all(&data);
        let options = Options {
            frames: 0,
            screenshot: None,
            record: None,
            fps: Some(file.fps),
            mute: true,
            dev: false,
            script: script::parse(&scenario.script.words())?,
            headless: true,
            settle: true,
        };
        // The scene is copied so the test never writes the game's own.
        std::fs::create_dir_all(&data)?;
        let scene = data.join("start.scene");
        if game.scene.exists() {
            std::fs::copy(game.scene, &scene)?;
        }
        let paths = Paths::new(
            game.directory.to_owned(),
            Some(game.assets.to_owned()),
            Some(scene),
            Some(shots.clone()),
            Some(data.clone()),
        );
        let mut session = Session::start(
            options,
            paths,
            engine.clone(),
            module.clone(),
            catalog.clone(),
            gpu.take().expect("the GPU comes back after each scenario"),
            None,
        )?;
        while session.step(None)? {}
        gpu = Some(session.finish()?);
        let _ = std::fs::remove_dir_all(&data);

        let mut taken: Vec<PathBuf> = std::fs::read_dir(&shots)
            .map(|entries| entries.filter_map(|entry| entry.ok().map(|entry| entry.path())).collect())
            .unwrap_or_default();
        taken.retain(|path| path.extension().is_some_and(|extension| extension == "png"));
        taken.sort();
        if taken.is_empty() {
            failed.push(format!("{name}: took no shots (add shot=NAME steps)"));
            continue;
        }
        for actual_path in taken {
            let shot = actual_path.file_name().unwrap().to_string_lossy().into_owned();
            let label = format!("{name}/{shot}");
            let golden_path = test.golden.join(name).join(&shot);
            if test.update {
                std::fs::create_dir_all(golden_path.parent().unwrap())?;
                std::fs::copy(&actual_path, &golden_path)?;
                eprintln!("  {label}: updated");
                updated += 1;
                continue;
            }
            if !golden_path.is_file() {
                failed.push(format!("{label}: missing golden {}; use --update to create it", golden_path.display()));
                continue;
            }
            let actual = image::open(&actual_path)?.to_rgba8();
            let golden = image::open(&golden_path)?.to_rgba8();
            if actual.dimensions() != golden.dimensions() {
                failed.push(format!(
                    "{label}: {:?} but the golden image is {:?}",
                    actual.dimensions(),
                    golden.dimensions()
                ));
                continue;
            }
            let diff = actual_path.with_extension("diff.png");
            let difference = compare(&actual, &golden, &diff)?;
            let summary = format!(
                "RMSE {:.4}%, largest {:.1}%, {:.3}% of pixels visibly changed",
                difference.rmse * 100.0,
                difference.largest * 100.0,
                difference.changed * 100.0
            );
            if difference.rmse > file.threshold {
                eprintln!("  {label}: FAILED, {summary} (see {})", diff.display());
                failed.push(format!("{label}: {summary}"));
            } else {
                eprintln!("  {label}: ok, {summary}");
                passed += 1;
            }
        }
    }
    let _ = std::fs::remove_dir_all(&scratch);
    eprintln!("\n{passed} passed, {} failed, {updated} written", failed.len());
    if !failed.is_empty() {
        bail!("visual tests failed:\n  {}", failed.join("\n  "));
    }
    Ok(())
}
