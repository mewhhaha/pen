//! The player's preferences (`Preferences`), kept between runs.
use std::path::PathBuf;

use anyhow::Result;
use pen_protocol::GameEvent;

use crate::input::Input;

/// The player's preferences, as the game last sent them. The game keeps
/// them; the host only stores them and plays them back as Preference events
/// at startup and after every restore.
pub struct Preferences {
    path: PathBuf,
    values: Vec<f32>,
}

impl Preferences {
    pub fn load(path: PathBuf) -> Preferences {
        let values = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<Vec<f32>>(&text).ok())
            .unwrap_or_default();
        Preferences { path, values }
    }

    pub(super) fn replay(&self, input: &mut Input) {
        for (index, value) in self.values.iter().enumerate() {
            input.enqueue(GameEvent::Preference { index: index as f32, value: *value });
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
}

#[cfg(test)]
mod tests {
    use pen_protocol::Viewport;

    use super::*;

    #[test]
    fn keeps_what_the_game_sends_and_plays_it_back() {
        let directory = std::env::temp_dir().join(format!("pen-preferences-{}", std::process::id()));
        let path = directory.join("data/preferences.json");
        let _ = std::fs::remove_dir_all(&directory);
        let mut preferences = Preferences::load(path.clone());
        let mut input = Input::default();
        preferences.replay(&mut input);
        assert!(input.next_frame(0.0, Viewport { width: 1, height: 1 }).events.is_empty());
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
        assert_eq!(events, [GameEvent::Preference { index: 0.0, value: 0.25 }]);
        let _ = std::fs::remove_dir_all(&directory);
    }
}
