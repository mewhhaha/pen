//! Captions: text a sound carries, for players who cannot hear it. The
//! assets' `captions.json` gives a sound (by its catalog key) its caption,
//! `"kaching.wav": "A till rings"`, or with how long it shows,
//! `"kaching.wav": {"text": "A till rings", "seconds": 1.5}`. As frames play
//! sounds, a [`Track`] keeps the captions showing: a sound's from when it
//! plays, a voice's from when it starts, each for its seconds, the same
//! caption again showing longer rather than twice. Hosts surface them to
//! assistive technology (a polite live region), and to the game through the
//! captions service ([`CaptionsService`], CAPTIONS_OP_*), which both hosts
//! answer alike.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use anyhow::{Context, Result, bail};
use pen_protocol::{Catalog, RenderFrame, captions_op, service};
use web_time::Instant;

use crate::services::{Answer, Request, Service};

/// How long a caption shows when captions.json does not say.
pub const SECONDS: f64 = 3.0;

/// A sound's caption.
#[derive(Clone, Debug, PartialEq)]
pub struct Caption {
    pub text: String,
    pub seconds: f64,
}

/// The captions of the catalog's sounds, by key.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Captions {
    by_sound: HashMap<String, Caption>,
}

impl Captions {
    /// captions.json: each sound's key to its caption's text, or to its
    /// text and seconds.
    pub fn parse(json: &str) -> Result<Captions> {
        let value: serde_json::Value = serde_json::from_str(json).context("captions.json is not JSON")?;
        let Some(sounds) = value.as_object() else { bail!("captions.json must map sounds to their captions") };
        let mut by_sound = HashMap::with_capacity(sounds.len());
        for (key, caption) in sounds {
            let (text, seconds) = match caption {
                serde_json::Value::String(text) => (text.clone(), SECONDS),
                serde_json::Value::Object(fields) => {
                    let text = fields.get("text").and_then(|text| text.as_str());
                    let seconds = fields.get("seconds").map(|seconds| seconds.as_f64());
                    match (text, seconds) {
                        (Some(text), None) => (text.to_owned(), SECONDS),
                        (Some(text), Some(Some(seconds))) if seconds.is_finite() && seconds > 0.0 => {
                            (text.to_owned(), seconds)
                        }
                        _ => bail!("captions.json: {key}'s caption needs a text, and positive seconds if any"),
                    }
                }
                _ => bail!("captions.json: {key}'s caption must be text or {{\"text\", \"seconds\"}}"),
            };
            by_sound.insert(key.clone(), Caption { text, seconds });
        }
        Ok(Captions { by_sound })
    }

    pub fn get(&self, sound: &str) -> Option<&Caption> {
        self.by_sound.get(sound)
    }

    pub fn is_empty(&self) -> bool {
        self.by_sound.is_empty()
    }

    /// The captions with a track, for a host and its captions service.
    pub fn shared(self) -> Shared {
        Arc::new(Mutex::new(State { captions: self, ..State::default() }))
    }
}

/// A caption showing, until when.
#[derive(Clone, Debug, PartialEq)]
struct Showing {
    text: String,
    until: f64,
}

/// The captions showing as frames play, the newest last.
#[derive(Debug, Default)]
pub struct Track {
    showing: Vec<Showing>,
    /// The last frame's voices: their clips, by id.
    voices: HashMap<u32, u32>,
}

impl Track {
    fn show(&mut self, caption: &Caption, now: f64) {
        let until = now + caption.seconds;
        match self.showing.iter().position(|showing| showing.text == caption.text) {
            Some(index) => {
                let mut again = self.showing.remove(index);
                again.until = again.until.max(until);
                self.showing.push(again);
            }
            None => self.showing.push(Showing { text: caption.text.clone(), until }),
        }
    }

    /// A frame plays at `now` seconds: the captions of its sounds, and of
    /// its voices that start (a new id, or another clip), show.
    pub fn frame(&mut self, frame: &RenderFrame, catalog: &Catalog, captions: &Captions, now: f64) {
        let caption = |clip: u32| catalog.sounds.get(clip as usize).and_then(|key| captions.get(key));
        for sound in &frame.sounds {
            if let Some(caption) = caption(sound.clip) {
                self.show(caption, now);
            }
        }
        let mut voices = HashMap::with_capacity(frame.voices.len());
        for voice in &frame.voices {
            if self.voices.get(&voice.id) != Some(&voice.clip)
                && let Some(caption) = caption(voice.clip)
            {
                self.show(caption, now);
            }
            voices.insert(voice.id, voice.clip);
        }
        self.voices = voices;
        self.showing.retain(|showing| showing.until > now);
    }

    /// The captions showing at `now`, newest first.
    pub fn active(&self, now: f64) -> Vec<&str> {
        self.showing.iter().rev().filter(|showing| showing.until > now).map(|showing| showing.text.as_str()).collect()
    }

    /// A world was replaced (time travel, a load): its voices start again.
    pub fn forget(&mut self) {
        self.voices.clear();
    }
}

/// The captions and the track, as a host and its captions service share
/// them, and the clock captions show by.
#[derive(Debug)]
pub struct State {
    pub captions: Captions,
    pub track: Track,
    /// The catalog's sounds, by clip, as the last frame had them.
    sounds: Vec<String>,
    started: Instant,
}

impl Default for State {
    fn default() -> State {
        State { captions: Captions::default(), track: Track::default(), sounds: Vec::new(), started: Instant::now() }
    }
}

impl State {
    fn now(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// A frame shown: the captions its sounds and voices bring.
    pub fn frame(&mut self, frame: &RenderFrame, catalog: &Catalog) {
        if self.sounds != catalog.sounds {
            self.sounds.clone_from(&catalog.sounds);
        }
        let now = self.now();
        self.track.frame(frame, catalog, &self.captions, now);
    }

    /// The captions showing now, newest first, a line each.
    pub fn text(&self) -> String {
        self.track.active(self.now()).join("\n")
    }
}

pub type Shared = Arc<Mutex<State>>;

pub fn lock(shared: &Shared) -> MutexGuard<'_, State> {
    shared.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The captions service: a clip's caption (CAPTIONS_OP_TEXT) and those
/// showing now (CAPTIONS_OP_ACTIVE).
pub struct CaptionsService(pub Shared);

impl<Input> Service<Input> for CaptionsService {
    fn code(&self) -> u32 {
        service::CAPTIONS
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        let state = lock(&self.0);
        Some(match request.operation {
            captions_op::TEXT => {
                let caption = pen_protocol::boundary::whole(request.numbers[0])
                    .and_then(|clip| state.sounds.get(clip as usize))
                    .and_then(|key| state.captions.get(key));
                match caption {
                    #[allow(clippy::cast_possible_truncation, reason = "seconds to show")]
                    Some(caption) => Answer { value: caption.seconds as f32, ..Answer::text(&caption.text) },
                    None => Answer::unavailable(),
                }
            }
            captions_op::ACTIVE => {
                let active = state.track.active(state.now());
                Answer { value: active.len() as f32, ..Answer::text(&active.join("\n")) }
            }
            _ => Answer::unknown(),
        })
    }

    fn rendered(&mut self, frame: &RenderFrame, catalog: &Catalog) {
        lock(&self.0).frame(frame, catalog);
    }

    fn restored(&mut self) {
        lock(&self.0).track.forget();
    }
}

#[cfg(test)]
mod tests {
    use pen_protocol::{ServiceRequest, Sound, Voice};

    use super::*;
    use crate::services::Services;

    const JSON: &str = r#"{"kaching.wav": "A till rings", "wind.wav": {"text": "Wind blows", "seconds": 5}}"#;

    fn catalog() -> Catalog {
        Catalog { sounds: vec!["kaching.wav".into(), "wind.wav".into(), "blip.wav".into()], ..Catalog::default() }
    }

    #[test]
    fn captions_read_as_text_or_with_seconds() {
        let captions = Captions::parse(JSON).unwrap();
        assert_eq!(captions.get("kaching.wav"), Some(&Caption { text: "A till rings".into(), seconds: SECONDS }));
        assert_eq!(captions.get("wind.wav").map(|caption| caption.seconds), Some(5.0));
        assert_eq!(captions.get("blip.wav"), None);
        assert!(Captions::parse(r#"{"a.wav": {"seconds": 2}}"#).is_err());
        assert!(Captions::parse(r#"{"a.wav": {"text": "x", "seconds": -1}}"#).is_err());
        assert!(Captions::parse("[]").is_err());
    }

    /// A sound's caption shows for its seconds, again for longer; a
    /// voice's when it starts, not each frame it plays, and again when it
    /// comes back or changes clip.
    #[test]
    fn the_track_shows_what_plays() {
        let (catalog, captions) = (catalog(), Captions::parse(JSON).unwrap());
        let mut track = Track::default();
        let sound = |clip: u32| Sound { clip, ..Sound::default() };
        let voice = |id: u32, clip: u32| Voice { id, clip, ..Voice::default() };
        let frame = |sounds: Vec<Sound>, voices: Vec<Voice>| RenderFrame { sounds, voices, ..RenderFrame::default() };
        track.frame(&frame(vec![sound(0), sound(2)], vec![voice(7, 1)]), &catalog, &captions, 0.0);
        assert_eq!(track.active(0.0), ["Wind blows", "A till rings"]);
        // The wind keeps blowing, its caption not shown again; the till
        // rings again, longer.
        track.frame(&frame(vec![sound(0)], vec![voice(7, 1)]), &catalog, &captions, 3.0);
        assert_eq!(track.active(3.0), ["A till rings", "Wind blows"]);
        assert_eq!(track.active(4.5), ["A till rings", "Wind blows"]);
        track.frame(&frame(vec![], vec![voice(7, 1)]), &catalog, &captions, 5.5);
        assert_eq!(track.active(5.5), ["A till rings"]);
        track.frame(&frame(vec![], vec![voice(7, 1)]), &catalog, &captions, 6.0);
        assert!(track.active(6.0).is_empty());
        // Left out and back, the wind starts again.
        track.frame(&frame(vec![], vec![]), &catalog, &captions, 7.0);
        track.frame(&frame(vec![], vec![voice(7, 1)]), &catalog, &captions, 8.0);
        assert_eq!(track.active(8.0), ["Wind blows"]);
    }

    /// The service answers a clip's caption and those showing.
    #[test]
    fn the_service_answers_captions() {
        let shared = Shared::default();
        lock(&shared).captions = Captions::parse(JSON).unwrap();
        let mut services: Services = Services::new();
        services.register(Box::new(CaptionsService(shared.clone())));
        let frame = RenderFrame { sounds: vec![Sound { clip: 0, ..Sound::default() }], ..RenderFrame::default() };
        services.rendered(&frame, &catalog());
        assert_eq!(lock(&shared).text(), "A till rings");
        let ask = |id: u32, operation: u32, a: f32| ServiceRequest {
            request: id,
            service: service::CAPTIONS,
            operation,
            a,
            ..ServiceRequest::default()
        };
        services.answer(&[
            ask(1, captions_op::TEXT, 1.0),
            ask(2, captions_op::TEXT, 2.0),
            ask(3, captions_op::ACTIVE, 0.0),
        ]);
        let answers = services.take_answers();
        let wind = Answer { value: 5.0, ..Answer::text("Wind blows") }.events(1);
        assert_eq!(answers[..wind.len()], wind[..]);
        let rest = &answers[wind.len()..];
        assert_eq!(rest[..1], Answer::unavailable().events(2)[..]);
        assert_eq!(rest[1..], Answer { value: 1.0, ..Answer::text("A till rings") }.events(3)[..]);
    }
}
