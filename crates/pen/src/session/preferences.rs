//! The player's preferences (`Preferences`), kept between runs: the game's
//! own, and the engine's (`pen_runtime::engine_preferences`).
use std::path::{Path, PathBuf};

use anyhow::Result;
use pen_protocol::{GameEvent, color_filter, engine_preference};
use pen_runtime::engine_preferences::{EnginePreferences, PreferencesService, Shared};

use crate::input::Input;

/// The player's preferences, as the game last sent them, and the engine's.
/// The game keeps its own; the host only stores them and plays them back
/// as Preference events at startup and after every restore, the engine's
/// after them (from index ENGINE_PREFERENCES). The engine's are kept in
/// `engine-preferences.json` beside them, set by the game's settings
/// through the `preferences` service, and for one run by PEN_COLOR_FILTER,
/// PEN_TEXT_SCALE and PEN_REDUCED_MOTION. Until the game has stored any
/// of its own, the host plays back a graphics preset for the GPU instead
/// (`crate::presets`), if given one.
pub struct Preferences {
    path: PathBuf,
    values: Vec<f32>,
    preset: Vec<(u32, f32)>,
    engine: Shared,
    engine_path: PathBuf,
    /// The engine's preferences' change count when they were last applied
    /// and kept; None before the first frame.
    applied: Option<u64>,
    /// PEN_COLOR_FILTER's `simulate-` filters: what a player with that
    /// deficiency sees, for checking a game's colours (never kept).
    simulate: Option<u32>,
}

/// The engine's preferences from `engine-preferences.json`: an object of
/// them by name (`{"text_scale": 1.5}`), as `keep` writes it.
fn read_engine(path: &Path) -> EnginePreferences {
    let mut preferences = EnginePreferences::default();
    let text = std::fs::read_to_string(path).unwrap_or_default();
    if let Ok(serde_json::Value::Object(values)) = serde_json::from_str(&text) {
        for (name, value) in values {
            let text = match value {
                serde_json::Value::String(text) => text,
                other => other.to_string(),
            };
            if let Err(error) = preferences.set_named(&name, &text) {
                eprintln!("{}: {error}", path.display());
            }
        }
    }
    preferences
}

/// The run's own settings from the environment, over the kept ones;
/// `PEN_COLOR_FILTER` takes a filter's name and an optional strength
/// (`deuteranopia:0.5`), or `simulate-NAME`.
fn from_environment(preferences: &mut EnginePreferences, simulate: &mut Option<u32>) {
    let read = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
    if let Some(filter) = read("PEN_COLOR_FILTER") {
        let (name, strength) = filter.split_once(':').unwrap_or((&filter, "1"));
        let shown = name.strip_prefix("simulate-");
        let kind = (0..).zip(color_filter::NAMES).find(|(_, known)| *known == shown.unwrap_or(name));
        match (kind, shown) {
            (Some((kind, _)), Some(_)) => *simulate = Some(kind),
            (Some(_), None) => {
                let set = preferences
                    .set_named("color_filter", name)
                    .and_then(|_| preferences.set_named("filter_strength", strength));
                if let Err(error) = set {
                    eprintln!("PEN_COLOR_FILTER: {error}");
                }
            }
            (None, _) => eprintln!("PEN_COLOR_FILTER: no filter {name:?} (one of {})", color_filter::NAMES.join(", ")),
        }
    }
    for (variable, name) in [("PEN_TEXT_SCALE", "text_scale"), ("PEN_REDUCED_MOTION", "reduced_motion")] {
        if let Some(value) = read(variable)
            && let Err(error) = preferences.set_named(name, &value)
        {
            eprintln!("{variable}: {error}");
        }
    }
}

impl Preferences {
    pub fn load(path: PathBuf) -> Preferences {
        let values = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<Vec<f32>>(&text).ok())
            .unwrap_or_default();
        let engine_path = path.with_file_name("engine-preferences.json");
        let mut engine = read_engine(&engine_path);
        let mut simulate = None;
        from_environment(&mut engine, &mut simulate);
        Preferences {
            path,
            values,
            preset: Vec::new(),
            engine: Shared::new(engine),
            engine_path,
            applied: None,
            simulate,
        }
    }

    /// Whether none are stored: the game's first run, or the player has
    /// changed none since.
    pub(super) fn untouched(&self) -> bool {
        self.values.is_empty()
    }

    /// Lanes and values to play back while none are stored.
    pub(super) fn preset(&mut self, preset: Vec<(u32, f32)>) {
        self.preset = preset;
    }

    pub(super) fn replay(&self, input: &mut Input) {
        if self.values.is_empty() {
            for (index, value) in &self.preset {
                input.enqueue(GameEvent::Preference { index: *index as f32, value: *value });
            }
        }
        for (index, value) in self.values.iter().enumerate() {
            input.enqueue(GameEvent::Preference { index: index as f32, value: *value });
        }
        for event in self.engine().events() {
            input.enqueue(event);
        }
    }

    pub(super) fn store(&mut self, values: &[f32]) -> Result<()> {
        if values == self.values {
            return Ok(());
        }
        self.values = values.to_vec();
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.path, serde_json::to_string(values)?)?;
        Ok(())
    }

    /// The `preferences` service, setting the engine's preferences here.
    pub(super) fn service(&self) -> PreferencesService {
        PreferencesService(self.engine.clone())
    }

    /// The engine's preferences now.
    pub fn engine(&self) -> EnginePreferences {
        self.engine.get().0
    }

    /// Before a frame draws: the colour filter to the renderer when the
    /// engine's preferences changed (or on the first frame), and those the
    /// game's settings changed kept.
    pub(super) fn apply(&mut self, renderer: &mut pen_render::Renderer) -> Result<()> {
        let (engine, changes) = self.engine.get();
        if self.applied == Some(changes) {
            return Ok(());
        }
        let filter = match self.simulate {
            Some(kind) => pen_render::ColorFilter { kind, strength: 1.0, simulate: true },
            None => {
                pen_render::ColorFilter { kind: engine.color_filter, strength: engine.filter_strength, simulate: false }
            }
        };
        renderer.set_color_filter(filter);
        if self.applied.replace(changes).is_some() {
            self.keep(&engine)?;
        }
        Ok(())
    }

    /// Writes the engine's preferences to `engine-preferences.json`.
    fn keep(&self, engine: &EnginePreferences) -> Result<()> {
        let values: serde_json::Map<String, serde_json::Value> = engine_preference::NAMES
            .iter()
            .zip(engine.values())
            .map(|(name, value)| {
                let value = match *name {
                    "color_filter" => serde_json::Value::from(color_filter::NAMES[engine.color_filter as usize]),
                    "reduced_motion" => serde_json::Value::from(engine.reduced_motion),
                    _ => serde_json::Value::from(value),
                };
                ((*name).to_owned(), value)
            })
            .collect();
        if let Some(parent) = self.engine_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.engine_path, serde_json::to_string_pretty(&values)? + "\n")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use pen_protocol::Viewport;
    use pen_protocol::boundary::ENGINE_PREFERENCES;

    use super::*;

    #[test]
    fn keeps_what_the_game_sends_and_plays_it_back() {
        let directory = std::env::temp_dir().join(format!("pen-preferences-{}", std::process::id()));
        let path = directory.join("data/preferences.json");
        let _ = std::fs::remove_dir_all(&directory);
        let mut preferences = Preferences::load(path.clone());
        let mut input = Input::default();
        preferences.replay(&mut input);
        // Only the engine's, at their defaults.
        let events = input.next_frame(0.0, Viewport { width: 1, height: 1 }).events;
        assert_eq!(events, EnginePreferences::default().events());
        preferences.store(&[0.5, 1.0]).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "[0.5,1.0]");
        // The same again writes nothing.
        std::fs::remove_file(&path).unwrap();
        preferences.store(&[0.5, 1.0]).unwrap();
        assert!(!path.exists());
        preferences.store(&[0.25]).unwrap();
        let loaded = Preferences::load(path);
        loaded.replay(&mut input);
        let events = input.next_frame(0.0, Viewport { width: 1, height: 1 }).events;
        assert_eq!(events[0], GameEvent::Preference { index: 0.0, value: 0.25 });
        assert_eq!(events.len(), 1 + engine_preference::NAMES.len());
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn keeps_the_engines_preferences_the_game_sets() {
        let directory = std::env::temp_dir().join(format!("pen-engine-preferences-{}", std::process::id()));
        let path = directory.join("preferences.json");
        let _ = std::fs::remove_dir_all(&directory);
        let preferences = Preferences::load(path.clone());
        let service = preferences.service();
        // The game's settings page sets two, as the service does.
        assert_eq!(service.0.set(engine_preference::TEXT_SCALE, 1.5), Some(1.5));
        assert_eq!(service.0.set(engine_preference::COLOR_FILTER, color_filter::PROTANOPIA as f32), Some(1.0));
        preferences.keep(&preferences.engine()).unwrap();
        let kept = std::fs::read_to_string(directory.join("engine-preferences.json")).unwrap();
        assert!(kept.contains("\"color_filter\": \"protanopia\"") && kept.contains("\"text_scale\": 1.5"), "{kept}");
        // The next run plays them back after the game's own.
        let loaded = Preferences::load(path);
        assert_eq!(loaded.engine().text_scale, 1.5);
        let mut input = Input::default();
        loaded.replay(&mut input);
        let events = input.next_frame(0.0, Viewport { width: 1, height: 1 }).events;
        let base = ENGINE_PREFERENCES as f32;
        assert!(events.contains(&GameEvent::Preference { index: base, value: 1.0 }));
        assert!(events.contains(&GameEvent::Preference { index: base + 2.0, value: 1.5 }));
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_preset_plays_until_preferences_are_stored() {
        let directory = std::env::temp_dir().join(format!("pen-preferences-preset-{}", std::process::id()));
        let path = directory.join("preferences.json");
        let _ = std::fs::remove_dir_all(&directory);
        let mut preferences = Preferences::load(path.clone());
        assert!(preferences.untouched());
        preferences.preset(vec![(0, 1.0), (4, 1024.0)]);
        let mut input = Input::default();
        preferences.replay(&mut input);
        let viewport = Viewport { width: 1, height: 1 };
        // The game's own lanes; the engine's preferences follow them.
        let game = |events: Vec<GameEvent>| -> Vec<GameEvent> {
            let engine = ENGINE_PREFERENCES as f32;
            events
                .into_iter()
                .filter(|event| matches!(event, GameEvent::Preference { index, .. } if *index < engine))
                .collect()
        };
        let events = game(input.next_frame(0.0, viewport).events);
        let preset =
            [GameEvent::Preference { index: 0.0, value: 1.0 }, GameEvent::Preference { index: 4.0, value: 1024.0 }];
        assert_eq!(events, preset);
        // Once the game sends its own, they play back alone.
        preferences.store(&[4.0, 0.0]).unwrap();
        assert!(!preferences.untouched());
        preferences.replay(&mut input);
        let events = game(input.next_frame(0.0, viewport).events);
        assert_eq!(
            events,
            [GameEvent::Preference { index: 0.0, value: 4.0 }, GameEvent::Preference { index: 1.0, value: 0.0 }]
        );
        assert!(!Preferences::load(path).untouched());
        let _ = std::fs::remove_dir_all(&directory);
    }
}
