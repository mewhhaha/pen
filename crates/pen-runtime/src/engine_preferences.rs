//! The engine's own preferences (ENGINE_PREFERENCE_* in
//! `protocol/schema.json`): a colour filter, how strong it is, a text scale
//! and reduced motion. They are the player's, the same for every game, and
//! the host keeps them; a player sets them through the host (its options,
//! the platform's settings) or a game's settings page, which asks the
//! `preferences` service ([`PreferencesService`]). The renderer applies the
//! colour filter to the whole picture; the game honors the rest where it
//! draws text and moves things, as it learns them from Preference events at
//! index `ENGINE_PREFERENCES` + the preference, played back with the game's
//! own preferences at startup and after every restore.
use std::sync::{Arc, Mutex, PoisonError};

use pen_protocol::boundary::{ENGINE_PREFERENCES, whole};
use pen_protocol::{GameEvent, color_filter, engine_preference, preferences_op, service};

use crate::services::{Answer, Request, Service};

/// The engine's preferences, as the player set them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnginePreferences {
    /// A COLOR_FILTER_*.
    pub color_filter: u32,
    /// How much of the filter shows, 0 to 1.
    pub filter_strength: f32,
    /// How much larger the game draws its text, 1 to 2.
    pub text_scale: f32,
    pub reduced_motion: bool,
}

impl Default for EnginePreferences {
    fn default() -> EnginePreferences {
        EnginePreferences {
            color_filter: color_filter::NONE,
            filter_strength: 1.0,
            text_scale: 1.0,
            reduced_motion: false,
        }
    }
}

/// The largest text scale; games lay their text out to fit at it.
pub const MAX_TEXT_SCALE: f32 = 2.0;

impl EnginePreferences {
    /// Sets `preference` to `value` as the engine keeps it (clamped to its
    /// range), and answers that; None for a preference or value it does not
    /// take.
    pub fn set(&mut self, preference: u32, value: f32) -> Option<f32> {
        if !value.is_finite() {
            return None;
        }
        match preference {
            engine_preference::COLOR_FILTER => {
                let kind = whole(value).filter(|kind| (*kind as usize) < color_filter::NAMES.len())?;
                self.color_filter = kind;
            }
            engine_preference::FILTER_STRENGTH => self.filter_strength = value.clamp(0.0, 1.0),
            engine_preference::TEXT_SCALE => self.text_scale = value.clamp(1.0, MAX_TEXT_SCALE),
            engine_preference::REDUCED_MOTION => self.reduced_motion = value >= 0.5,
            _ => return None,
        }
        Some(self.get(preference))
    }

    /// A preference's value, as the game gets it.
    pub fn get(&self, preference: u32) -> f32 {
        match preference {
            engine_preference::COLOR_FILTER => self.color_filter as f32,
            engine_preference::FILTER_STRENGTH => self.filter_strength,
            engine_preference::TEXT_SCALE => self.text_scale,
            engine_preference::REDUCED_MOTION => f32::from(u8::from(self.reduced_motion)),
            _ => 0.0,
        }
    }

    /// Every preference's value, in ENGINE_PREFERENCE's order.
    pub fn values(&self) -> Vec<f32> {
        (0..).zip(engine_preference::NAMES).map(|(preference, _)| self.get(preference)).collect()
    }

    /// The preferences `values` (in ENGINE_PREFERENCE's order) set, those
    /// it lacks or that do not fit at their defaults.
    pub fn from_values(values: &[f32]) -> EnginePreferences {
        let mut preferences = EnginePreferences::default();
        for ((preference, _), value) in (0..).zip(engine_preference::NAMES).zip(values) {
            preferences.set(preference, *value);
        }
        preferences
    }

    /// The Preference events that tell the game.
    pub fn events(&self) -> Vec<GameEvent> {
        self.values()
            .into_iter()
            .enumerate()
            .map(|(preference, value)| GameEvent::Preference { index: (ENGINE_PREFERENCES + preference) as f32, value })
            .collect()
    }

    /// A preference by its name (`text_scale`) set from text: a number, a
    /// colour filter's name, or on/off, true/false and yes/no for reduced
    /// motion. For hosts' options and environment variables.
    pub fn set_named(&mut self, name: &str, text: &str) -> Result<f32, String> {
        let (preference, _) = (0..)
            .zip(engine_preference::NAMES)
            .find(|(_, known)| *known == name)
            .ok_or_else(|| format!("no engine preference {name:?} (one of {})", engine_preference::NAMES.join(", ")))?;
        let text = text.trim().to_ascii_lowercase();
        let value = match text.as_str() {
            "on" | "true" | "yes" => 1.0,
            "off" | "false" | "no" => 0.0,
            named => match color_filter::NAMES.iter().position(|filter| *filter == named) {
                Some(kind) => kind as f32,
                None => named.parse::<f32>().map_err(|_| format!("{name}: {text:?} is not a value it takes"))?,
            },
        };
        self.set(preference, value).ok_or_else(|| format!("{name}: {text:?} is not a value it takes"))
    }
}

/// The engine's preferences as a host and its `preferences` service share
/// them, with a count of the changes, so the host sees when one came in
/// (to apply the filter and keep them).
#[derive(Clone, Debug, Default)]
pub struct Shared(Arc<Mutex<(EnginePreferences, u64)>>);

impl Shared {
    pub fn new(preferences: EnginePreferences) -> Shared {
        Shared(Arc::new(Mutex::new((preferences, 0))))
    }

    /// The preferences now, and how many changes came in so far.
    pub fn get(&self) -> (EnginePreferences, u64) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Sets one, as the service does; the kept value, if it took it.
    pub fn set(&self, preference: u32, value: f32) -> Option<f32> {
        let mut shared = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let kept = shared.0.set(preference, value)?;
        shared.1 += 1;
        Some(kept)
    }
}

/// The `preferences` service: a game's settings page setting one of the
/// engine's preferences (`set`: preference `a` to `b`). The host applies it
/// and keeps it; the answer is the value kept.
pub struct PreferencesService(pub Shared);

impl<Input> Service<Input> for PreferencesService {
    fn code(&self) -> u32 {
        service::PREFERENCES
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        if request.operation != preferences_op::SET {
            return Some(Answer::unknown());
        }
        let [preference, value, ..] = request.numbers;
        let Some(preference) = whole(preference) else { return Some(Answer::failed()) };
        Some(self.0.set(preference, value).map_or_else(Answer::failed, Answer::ok))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_values_in_range_and_refuses_others() {
        let mut preferences = EnginePreferences::default();
        assert_eq!(preferences.set(engine_preference::TEXT_SCALE, 3.0), Some(2.0));
        assert_eq!(preferences.set(engine_preference::TEXT_SCALE, 0.5), Some(1.0));
        assert_eq!(preferences.set(engine_preference::FILTER_STRENGTH, -1.0), Some(0.0));
        assert_eq!(preferences.set(engine_preference::REDUCED_MOTION, 0.7), Some(1.0));
        assert_eq!(preferences.set(engine_preference::COLOR_FILTER, 2.0), Some(2.0));
        assert_eq!(preferences.color_filter, color_filter::DEUTERANOPIA);
        // Not a filter, not a whole number, not a number, not a preference.
        assert_eq!(preferences.set(engine_preference::COLOR_FILTER, 9.0), None);
        assert_eq!(preferences.set(engine_preference::COLOR_FILTER, 1.5), None);
        assert_eq!(preferences.set(engine_preference::TEXT_SCALE, f32::NAN), None);
        assert_eq!(preferences.set(99, 1.0), None);
        assert_eq!(preferences.color_filter, color_filter::DEUTERANOPIA);
        assert_eq!(EnginePreferences::from_values(&preferences.values()), preferences);
        // What does not fit stays at its default.
        assert_eq!(EnginePreferences::from_values(&[7.0, f32::NAN]), EnginePreferences::default());
    }

    #[test]
    fn tells_the_game_past_its_own_preferences() {
        let preferences = EnginePreferences { text_scale: 1.5, reduced_motion: true, ..EnginePreferences::default() };
        let base = ENGINE_PREFERENCES as f32;
        assert_eq!(
            preferences.events(),
            [
                GameEvent::Preference { index: base, value: 0.0 },
                GameEvent::Preference { index: base + 1.0, value: 1.0 },
                GameEvent::Preference { index: base + 2.0, value: 1.5 },
                GameEvent::Preference { index: base + 3.0, value: 1.0 },
            ]
        );
    }

    #[test]
    fn takes_names_from_options() {
        let mut preferences = EnginePreferences::default();
        assert_eq!(preferences.set_named("color_filter", "Tritanopia"), Ok(3.0));
        assert_eq!(preferences.set_named("reduced_motion", "on"), Ok(1.0));
        assert_eq!(preferences.set_named("text_scale", "1.25"), Ok(1.25));
        assert!(preferences.set_named("text_size", "2").is_err());
        assert!(preferences.set_named("color_filter", "sepia").is_err());
    }

    #[test]
    fn the_service_sets_and_counts_changes() {
        let shared = Shared::default();
        let mut service = PreferencesService(shared.clone());
        let ask = |service: &mut PreferencesService, numbers: [f32; 4]| {
            Service::<()>::call(service, &Request { id: 1, operation: preferences_op::SET, numbers, values: &[] })
                .unwrap()
        };
        assert_eq!(ask(&mut service, [2.0, 1.75, 0.0, 0.0]), Answer::ok(1.75));
        assert_eq!(ask(&mut service, [0.0, 12.0, 0.0, 0.0]), Answer::failed());
        assert_eq!(ask(&mut service, [0.5, 1.0, 0.0, 0.0]), Answer::failed());
        let (preferences, changes) = shared.get();
        assert_eq!((preferences.text_scale, changes), (1.75, 1));
    }
}
