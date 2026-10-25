//! Stateless noise service shared by both hosts.
//! Physics, navigation and collider queries belong to guest libraries or plugins; the
//! host does not embed a rigid-body engine. Their former service IDs remain
//! reserved and receive the registry's ordinary unknown answer.
//! Noise requests carry the whole problem, keeping replay and
//! save state in the game. `conformance` checks equal native/browser answers.
use pen_protocol::boundary::MAX_SERVICE_DATA;
use pen_runtime::services::{Answer, Services};

pub mod conformance;
mod noise;

pub use noise::Noise;

/// Adds the simulation services to a host's registry.
pub fn register<Input>(services: &mut Services<Input>) {
    services.register(Box::new(Noise));
}

/// A request's values, read in order. Each read is None past the end, or
/// for a value that is not what it should be: a number that is not finite,
/// or a count that is no whole number or is too large.
struct Lanes<'a> {
    values: &'a [f32],
    at: usize,
}

impl<'a> Lanes<'a> {
    fn new(values: &'a [f32]) -> Lanes<'a> {
        Lanes { values, at: 0 }
    }

    /// The next `count` numbers, every one finite.
    fn numbers(&mut self, count: usize) -> Option<&'a [f32]> {
        let numbers = self.values.get(self.at..self.at.checked_add(count)?)?;
        self.at += count;
        numbers.iter().all(|value| value.is_finite()).then_some(numbers)
    }

    /// Whether every value has been read.
    fn done(&self) -> bool {
        self.at >= self.values.len()
    }
}

/// An answer of `count` records laid out in `values`, its value the count,
/// or `failed` when they would not fit in one answer.
fn counted(count: usize, values: Vec<f32>) -> Answer {
    if values.len() > MAX_SERVICE_DATA {
        return Answer::failed();
    }
    Answer { values, ..Answer::ok(count as f32) }
}
