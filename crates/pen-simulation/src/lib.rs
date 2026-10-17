//! The simulation services: rigid-body physics (rapier), paths over a
//! navmesh, ray and shape queries against colliders, and noise and
//! heightfields. Both hosts register them (`register`), and both answer a
//! request with the same bits: x86_64 and wasm32 run the same arithmetic,
//! rapier with libm and without SIMD (`enhanced-determinism`), the rest with
//! nothing but IEEE operations that round alike everywhere. `conformance`
//! checks that against fixed expected bytes on each host.
//!
//! Each request carries the whole problem (the bodies, the colliders, the
//! areas, the noise's settings) and its answer is a function of it alone:
//! no service keeps a world between requests. That keeps them inside the
//! determinism contract (docs/determinism.md): the game's state is all of
//! the game, so after a rewind or reload it asks the same question and gets
//! the same answer, and a replay reads the answer from the recording.
//!
//! They speak in the services boundary's numbers, as
//! `pen_runtime::services::Pathfinding` does: four numbers and the values,
//! laid out as `protocol/schema.json`'s `PhysicsOperation`,
//! `ColliderShape`, `NavmeshOperation`, `SpatialOperation` and
//! `NoiseOperation` say. A request that does not follow its layout is
//! answered `failed`, one for an operation the service lacks `unknown`.
use pen_protocol::boundary::{MAX_SERVICE_DATA, whole};
use pen_runtime::services::{Answer, Services};

pub mod conformance;
mod navmesh;
mod noise;
mod physics;
mod shapes;
mod spatial;

pub use navmesh::Navmesh;
pub use noise::Noise;
pub use physics::Physics;
pub use spatial::Spatial;

/// Adds the simulation services to a host's registry.
pub fn register<Input>(services: &mut Services<Input>) {
    services.register(Box::new(Physics));
    services.register(Box::new(Navmesh));
    services.register(Box::new(Spatial));
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

    fn number(&mut self) -> Option<f32> {
        let value = *self.values.get(self.at)?;
        self.at += 1;
        value.is_finite().then_some(value)
    }

    /// A whole number up to `most`.
    fn count(&mut self, most: usize) -> Option<usize> {
        whole(self.number()?).map(|value| value as usize).filter(|value| *value <= most)
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
