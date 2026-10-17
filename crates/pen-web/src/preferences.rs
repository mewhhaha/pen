//! The player's preferences, kept in the page's localStorage between visits:
//! the game's own and the engine's (`pen_runtime::engine_preferences`), and
//! the game's actions with the player's bindings (`pen_runtime::actions`).
use pen_protocol::GameEvent;
use pen_runtime::actions::{self, Actions};
use pen_runtime::engine_preferences::{EnginePreferences, PreferencesService, Shared};

const KEY: &str = "pen.preferences";
const ENGINE: &str = "pen.engine-preferences";
const BINDINGS: &str = "pen.bindings";

fn storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

/// The player's preferences, as the game last sent them. The game keeps
/// them; the page only stores them and plays them back as Preference events
/// at startup and after every restore, the engine's after them. The
/// engine's are set by the game's settings (the `preferences` service), and
/// reduced motion follows the page's `prefers-reduced-motion` until they
/// are.
pub struct Preferences {
    values: Vec<f32>,
    engine: Shared,
    /// The engine's change count when they were last applied and kept.
    applied: Option<u64>,
}

/// The engine's preferences the page kept, else reduced motion as the
/// browser's settings ask.
fn load_engine() -> EnginePreferences {
    let text = storage().and_then(|storage| storage.get_item(ENGINE).ok().flatten());
    if let Some(values) = text.and_then(|text| serde_json::from_str::<Vec<f32>>(&text).ok()) {
        return EnginePreferences::from_values(&values);
    }
    let mut preferences = EnginePreferences::default();
    let reduce = web_sys::window()
        .and_then(|window| window.match_media("(prefers-reduced-motion: reduce)").ok().flatten())
        .is_some_and(|query| query.matches());
    preferences.reduced_motion = reduce;
    preferences
}

impl Preferences {
    pub fn load() -> Preferences {
        let text = storage().and_then(|storage| storage.get_item(KEY).ok().flatten()).unwrap_or_default();
        let values: Vec<f32> = serde_json::from_str(&text).unwrap_or_default();
        Preferences { values, engine: Shared::new(load_engine()), applied: None }
    }

    /// The events that play them back to the game.
    pub fn events(&self) -> impl Iterator<Item = GameEvent> + '_ {
        self.values
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .map(|(index, value)| GameEvent::Preference { index: index as f32, value: *value })
            .chain(self.engine.get().0.events())
    }

    /// The `preferences` service, setting the engine's preferences here.
    pub fn service(&self) -> PreferencesService {
        PreferencesService(self.engine.clone())
    }

    /// Before a frame draws: the colour filter to the renderer when the
    /// engine's preferences changed, and those the game's settings set kept.
    pub fn apply(&mut self, renderer: &mut pen_render::Renderer) {
        let (engine, changes) = self.engine.get();
        if self.applied == Some(changes) {
            return;
        }
        renderer.set_color_filter(pen_render::ColorFilter {
            kind: engine.color_filter,
            strength: engine.filter_strength,
            simulate: false,
        });
        if self.applied.replace(changes).is_some()
            && let Some(storage) = storage()
            && let Ok(text) = serde_json::to_string(&engine.values())
            && let Err(error) = storage.set_item(ENGINE, &text)
        {
            web_sys::console::warn_1(&format!("keeping the engine's preferences: {error:?}").into());
        }
    }

    /// Keeps what the game sent, if it changed.
    pub fn store(&mut self, values: &[f32]) {
        if values == self.values {
            return;
        }
        self.values = values.to_vec();
        if let Some(storage) = storage()
            && let Ok(text) = serde_json::to_string(values)
            && let Err(error) = storage.set_item(KEY, &text)
        {
            web_sys::console::warn_1(&format!("keeping preferences: {error:?}").into());
        }
    }
}

/// The actions and bindings the page kept, or none.
pub fn load_bindings() -> Actions {
    let text = storage().and_then(|storage| storage.get_item(BINDINGS).ok().flatten()).unwrap_or_default();
    Actions::load(&text)
}

/// Keeps the actions and bindings, if they changed.
pub fn keep_bindings(actions: &actions::Shared) {
    let mut actions = actions.borrow_mut();
    if !actions.take_changed() {
        return;
    }
    if let Some(storage) = storage()
        && let Err(error) = storage.set_item(BINDINGS, &actions.save())
    {
        web_sys::console::warn_1(&format!("keeping the player's bindings: {error:?}").into());
    }
}
