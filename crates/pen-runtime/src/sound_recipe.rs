//! Bounded, deterministic mono sound recipes, rendered once when loaded.
//! Games own the graph, seeds and timing. Neither host evaluates it in the
//! audio callback. Node indices refer only to earlier nodes; lowpass nodes
//! retain one accumulator across samples.
#![allow(clippy::cast_possible_truncation, reason = "bounded audio samples")]
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pen_sound: u32,
    pub rate: u32,
    frames: u32,
    crossfade: u32,
    seed: u32,
    peak: f64,
    nodes: Vec<Node>,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Node {
    Constant { value: f64 },
    Time,
    Noise,
    Add { a: usize, b: usize },
    Subtract { a: usize, b: usize },
    Multiply { a: usize, b: usize },
    Sin { input: usize },
    Exp { input: usize },
    Lowpass { input: usize, coefficient: usize },
}

impl Recipe {
    pub fn scaled(&mut self, gain: f64) -> Result<Vec<u8>> {
        let peak = self.peak * gain;
        ensure!(peak.is_finite() && peak > 0.0 && peak <= 1.0, "invalid sound recipe gain");
        self.peak = peak;
        Ok(serde_json::to_vec(self)?)
    }
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() <= 64 * 1024, "sound recipe is too large");
        let recipe: Self = serde_json::from_slice(bytes)?;
        ensure!(recipe.pen_sound == 1, "unsupported sound recipe version");
        ensure!((8000..=48000).contains(&recipe.rate), "sound recipe rate must be 8000..48000");
        ensure!(recipe.frames > 0 && recipe.frames <= recipe.rate * 60, "sound recipe must last at most 60 seconds");
        ensure!(recipe.crossfade <= recipe.frames / 2, "sound crossfade exceeds half its duration");
        ensure!(recipe.seed != 0, "sound recipe seed must be nonzero");
        ensure!(recipe.peak.is_finite() && recipe.peak > 0.0 && recipe.peak <= 1.0, "invalid sound recipe peak");
        ensure!(!recipe.nodes.is_empty() && recipe.nodes.len() <= 128, "sound recipe needs 1..128 nodes");
        ensure!(
            u64::from(recipe.frames + recipe.crossfade) * recipe.nodes.len() as u64 <= 64_000_000,
            "sound recipe exceeds its work limit"
        );
        for (index, node) in recipe.nodes.iter().enumerate() {
            let valid = match *node {
                Node::Constant { value } => value.is_finite(),
                Node::Time | Node::Noise => true,
                Node::Sin { input } | Node::Exp { input } => input < index,
                Node::Add { a, b } | Node::Subtract { a, b } | Node::Multiply { a, b } => a < index && b < index,
                Node::Lowpass { input, coefficient } => input < index && coefficient < index,
            };
            ensure!(valid, "invalid sound recipe node {index}");
        }
        Ok(recipe)
    }

    pub fn render(&self) -> Result<Vec<f32>> {
        let mut values = vec![0.0f64; self.nodes.len()];
        let mut state = values.clone();
        let mut seed = self.seed;
        let mut samples = Vec::with_capacity((self.frames + self.crossfade) as usize);
        for frame in 0..self.frames + self.crossfade {
            for (index, node) in self.nodes.iter().enumerate() {
                let value = match *node {
                    Node::Constant { value } => value,
                    Node::Time => f64::from(frame) / f64::from(self.frames),
                    Node::Noise => {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        f64::from(seed) / f64::from(u32::MAX) * 2.0 - 1.0
                    }
                    Node::Add { a, b } => values[a] + values[b],
                    Node::Subtract { a, b } => values[a] - values[b],
                    Node::Multiply { a, b } => values[a] * values[b],
                    Node::Sin { input } => values[input].sin(),
                    Node::Exp { input } => values[input].exp(),
                    Node::Lowpass { input, coefficient } => {
                        ensure!((0.0..=1.0).contains(&values[coefficient]), "invalid sound lowpass coefficient");
                        state[index] += values[coefficient] * (values[input] - state[index]);
                        state[index]
                    }
                };
                ensure!(value.is_finite(), "sound recipe produced a nonfinite value");
                values[index] = value;
            }
            samples.push(*values.last().unwrap());
        }
        for frame in 0..self.crossfade as usize {
            let phase = std::f64::consts::FRAC_PI_2 * frame as f64 / f64::from(self.crossfade);
            samples[frame] = samples[frame] * phase.sin() + samples[self.frames as usize + frame] * phase.cos();
        }
        samples.truncate(self.frames as usize);
        let peak = samples.iter().fold(1e-9f64, |peak, value| peak.max(value.abs()));
        Ok(samples.into_iter().map(|sample| (sample * (self.peak / peak)) as f32).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> serde_json::Value {
        json!({"pen_sound":1,"rate":8000,"frames":8000,"crossfade":400,"seed":81,"peak":0.7,
            "nodes":[{"op":"noise"},{"op":"constant","value":0.02},
                {"op":"lowpass","input":0,"coefficient":1}]})
    }

    #[test]
    fn seeded_filtered_loop_is_repeatable_bounded_and_continuous() {
        let recipe = Recipe::parse(&serde_json::to_vec(&fixture()).unwrap()).unwrap();
        let a = recipe.render().unwrap();
        assert_eq!(a, recipe.render().unwrap());
        assert_eq!(a.len(), 8000);
        assert!((a.iter().fold(0f32, |v, x| v.max(x.abs())) - 0.7).abs() < 1e-6);
        assert!((a[0] - a[a.len() - 1]).abs() < 0.1);
        let mut other = fixture();
        other["seed"] = 82.into();
        assert_ne!(a, Recipe::parse(&serde_json::to_vec(&other).unwrap()).unwrap().render().unwrap());
    }

    #[test]
    fn rejects_unbounded_cyclic_and_nonfinite_recipes() {
        for (key, value) in [
            ("frames", json!(u32::MAX)),
            ("crossfade", json!(8000)),
            ("seed", json!(0)),
            ("rate", json!(0)),
            ("peak", json!(2)),
            ("nodes", json!([{"op":"sin","input":0}])),
        ] {
            let mut invalid = fixture();
            invalid[key] = value;
            assert!(Recipe::parse(&serde_json::to_vec(&invalid).unwrap()).is_err(), "{key}");
        }
        let mut invalid = fixture();
        invalid["nodes"] = json!([{"op":"constant","value":1000},{"op":"exp","input":0}]);
        assert!(Recipe::parse(&serde_json::to_vec(&invalid).unwrap()).unwrap().render().is_err());
    }
}
