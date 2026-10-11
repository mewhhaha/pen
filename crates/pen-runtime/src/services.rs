use pen_protocol::boundary::MAX_SERVICE_DATA;
use pen_protocol::{GameEvent, service_status};

/// A request as a service sees it.
pub struct Request<'a> {
    pub id: u32,
    pub operation: u32,
    /// The request's four numbers, `a` to `d`.
    pub numbers: [f32; 4],
    /// Its values: text as code points, or a compute service's numbers.
    pub values: &'a [f32],
}

impl Request<'_> {
    /// The values as text; values that are no code point become U+FFFD.
    pub fn text(&self) -> String {
        text(self.values)
    }
}

/// Code points as text; values that are no code point become U+FFFD.
pub fn text(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| {
            let code = if value.fract() == 0.0 && *value >= 0.0 { *value as u32 } else { u32::MAX };
            char::from_u32(code).unwrap_or(char::REPLACEMENT_CHARACTER)
        })
        .collect()
}

/// Text as code points.
pub fn code_points(text: &str) -> Vec<f32> {
    text.chars().map(|char| char as u32 as f32).collect()
}

/// A service's answer: how it went (`pen_protocol::service_status`), a
/// number, and values (text as code points, or numbers).
#[derive(Clone, Debug, PartialEq)]
pub struct Answer {
    pub status: u32,
    pub value: f32,
    pub values: Vec<f32>,
}

impl Answer {
    pub fn ok(value: f32) -> Answer {
        Answer { status: service_status::OK, value, values: Vec::new() }
    }

    pub fn text(text: &str) -> Answer {
        Answer { status: service_status::OK, value: 0.0, values: code_points(text) }
    }

    pub fn unavailable() -> Answer {
        Answer { status: service_status::UNAVAILABLE, value: 0.0, values: Vec::new() }
    }

    pub fn failed() -> Answer {
        Answer { status: service_status::FAILED, value: 0.0, values: Vec::new() }
    }

    pub fn unknown() -> Answer {
        Answer { status: service_status::UNKNOWN, value: 0.0, values: Vec::new() }
    }

    /// The events that carry it to the game: a ServiceAnswer, then its
    /// values three to a ServiceData, the last padded with zeros.
    pub fn events(&self, request: u32) -> Vec<GameEvent> {
        let values = &self.values[..self.values.len().min(MAX_SERVICE_DATA)];
        let value = if self.value.is_finite() { self.value } else { 0.0 };
        let mut events = vec![GameEvent::ServiceAnswer { request, status: self.status, value, length: values.len() as u32 }];
        for chunk in values.chunks(3) {
            let at = |index: usize| chunk.get(index).copied().filter(|value| value.is_finite()).unwrap_or(0.0);
            events.push(GameEvent::ServiceData { request, a: at(0), b: at(1), c: at(2) });
        }
        events
    }
}
