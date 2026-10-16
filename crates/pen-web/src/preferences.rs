//! The player's preferences, kept in the page's localStorage between visits.
use pen_protocol::GameEvent;

const KEY: &str = "pen.preferences";

fn storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

/// The player's preferences, as the game last sent them. The game keeps
/// them; the page only stores them and plays them back as Preference events
/// at startup and after every restore.
pub struct Preferences {
    values: Vec<f32>,
}

impl Preferences {
    pub fn load() -> Preferences {
        let text = storage().and_then(|storage| storage.get_item(KEY).ok().flatten()).unwrap_or_default();
        let values: Vec<f32> = serde_json::from_str(&text).unwrap_or_default();
        Preferences { values }
    }

    /// The events that play them back to the game.
    pub fn events(&self) -> impl Iterator<Item = GameEvent> + '_ {
        self.values
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .map(|(index, value)| GameEvent::Preference { index: index as f32, value: *value })
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
