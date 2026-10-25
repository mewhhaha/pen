//! Seeded instance recipes. The game supplies a bounded scalar graph and a
//! planting mask; both hosts expand it once into ordinary instance records.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "validated recipe indices and finite float output"
)]
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Recipe {
    version: u32,
    seed: u32,
    random_draws: usize,
    stride: usize,
    inputs: Vec<f64>,
    tables: Vec<Vec<f64>>,
    nodes: Vec<Node>,
    output: [usize; 20],
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Node {
    Constant { value: f64 },
    Input { lane: usize },
    Random { index: usize },
    Lookup { table: usize, index: usize },
    Add { a: usize, b: usize },
    Subtract { a: usize, b: usize },
    Multiply { a: usize, b: usize },
    Divide { a: usize, b: usize },
    Min { a: usize, b: usize },
    Max { a: usize, b: usize },
    Less { a: usize, b: usize },
    Sin { input: usize },
    Cos { input: usize },
    Exp { input: usize },
    Sqrt { input: usize },
    Floor { input: usize },
    Round { input: usize, units: f64 },
    Select { test: usize, yes: usize, no: usize },
    MixBits { input: usize, shift: u32, multiplier: u32, final_shift: u32 },
}

impl Node {
    fn inputs(&self) -> Vec<usize> {
        match *self {
            Self::Constant { .. } | Self::Input { .. } => vec![],
            Self::Random { index } | Self::Lookup { index, .. } => vec![index],
            Self::Add { a, b }
            | Self::Subtract { a, b }
            | Self::Multiply { a, b }
            | Self::Divide { a, b }
            | Self::Min { a, b }
            | Self::Max { a, b }
            | Self::Less { a, b } => vec![a, b],
            Self::Sin { input }
            | Self::Cos { input }
            | Self::Exp { input }
            | Self::Sqrt { input }
            | Self::Floor { input }
            | Self::Round { input, .. }
            | Self::MixBits { input, .. } => vec![input],
            Self::Select { test, yes, no } => vec![test, yes, no],
        }
    }
}

impl Recipe {
    pub fn generate(&self) -> Result<Vec<f32>> {
        ensure!(self.version == 1, "unsupported instance recipe version");
        ensure!(
            (1..=16).contains(&self.stride) && self.inputs.len().is_multiple_of(self.stride),
            "invalid instance recipe input stride"
        );
        let count = self.inputs.len() / self.stride;
        ensure!(
            (1..=100_000).contains(&count)
                && (1..=2048).contains(&self.nodes.len())
                && count * self.nodes.len() <= 150_000_000,
            "instance recipe exceeds its work limit"
        );
        ensure!(
            self.random_draws <= 4_000_000
                && self.tables.len() <= 16
                && self.tables.iter().map(Vec::len).sum::<usize>() <= 1_000_000,
            "instance recipe data exceeds its limit"
        );
        ensure!(
            self.inputs.iter().chain(self.tables.iter().flatten()).all(|v| v.is_finite()),
            "instance recipe data must be finite"
        );
        ensure!(self.output.iter().all(|&v| v < self.nodes.len()), "invalid instance recipe output");
        for (index, node) in self.nodes.iter().enumerate() {
            ensure!(node.inputs().iter().all(|&v| v < index), "instance recipe has a forward reference");
            let valid = match *node {
                Node::Constant { value } => value.is_finite(),
                Node::Input { lane } => lane < self.stride,
                Node::Lookup { table, .. } => table < self.tables.len(),
                Node::Round { units, .. } => units.is_finite() && units > 0.0,
                Node::MixBits { shift, final_shift, .. } => shift < 32 && final_shift < 32,
                _ => true,
            };
            ensure!(valid, "invalid instance recipe node {index}");
        }
        let mut rng = Random::new(self.seed);
        let random: Vec<_> = (0..self.random_draws).map(|_| rng.sample()).collect();
        let mut values = vec![0.0; self.nodes.len()];
        let mut out = Vec::with_capacity(count * 20);
        let index = |v: f64, len: usize| -> Result<usize> {
            ensure!(v >= 0.0 && v < len as f64 && v.fract() == 0.0, "instance recipe lookup outside its data");
            Ok(v as usize)
        };
        for row in self.inputs.chunks_exact(self.stride) {
            for (i, node) in self.nodes.iter().enumerate() {
                let value = match *node {
                    Node::Constant { value } => value,
                    Node::Input { lane } => row[lane],
                    Node::Random { index: at } => random[index(values[at], random.len())?],
                    Node::Lookup { table, index: at } => {
                        self.tables[table][index(values[at], self.tables[table].len())?]
                    }
                    Node::Add { a, b } => values[a] + values[b],
                    Node::Subtract { a, b } => values[a] - values[b],
                    Node::Multiply { a, b } => values[a] * values[b],
                    Node::Divide { a, b } => values[a] / values[b],
                    Node::Min { a, b } => values[a].min(values[b]),
                    Node::Max { a, b } => values[a].max(values[b]),
                    Node::Less { a, b } => {
                        if values[a] < values[b] {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    Node::Sin { input } => values[input].sin(),
                    Node::Cos { input } => values[input].cos(),
                    Node::Exp { input } => values[input].exp(),
                    Node::Sqrt { input } => values[input].sqrt(),
                    Node::Floor { input } => values[input].floor(),
                    Node::Round { input, units } => (values[input] * units).round_ties_even() / units,
                    Node::Select { test, yes, no } => values[if values[test] != 0.0 { yes } else { no }],
                    Node::MixBits { input, shift, multiplier, final_shift } => {
                        let v = values[input].rem_euclid(4294967296.0) as u32;
                        let v = (v ^ (v >> shift)).wrapping_mul(multiplier);
                        f64::from(v ^ (v >> final_shift))
                    }
                };
                ensure!(value.is_finite(), "instance recipe produced nonfinite data");
                values[i] = value;
            }
            for &output in &self.output {
                let value = values[output] as f32;
                ensure!(value.is_finite(), "instance recipe output exceeds float32");
                out.push(value);
            }
        }
        Ok(out)
    }
}

/// MT19937's init-by-array with one U32 seed and 53-bit samples. This is
/// Python's integer-seeded Random stream, so authoring can retain its draws.
struct Random {
    state: [u32; 624],
    at: usize,
}
impl Random {
    fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = 19650218;
        for i in 1..624 {
            state[i] = 1812433253u32.wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30)).wrapping_add(i as u32);
        }
        let mut i = 1;
        for _ in 0..624 {
            state[i] = (state[i] ^ (state[i - 1] ^ (state[i - 1] >> 30)).wrapping_mul(1664525)).wrapping_add(seed);
            i += 1;
            if i >= 624 {
                state[0] = state[623];
                i = 1;
            }
        }
        for _ in 0..623 {
            state[i] =
                (state[i] ^ (state[i - 1] ^ (state[i - 1] >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                state[0] = state[623];
                i = 1;
            }
        }
        state[0] = 0x80000000;
        Self { state, at: 624 }
    }
    fn next(&mut self) -> u32 {
        if self.at == 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x80000000) | (self.state[(i + 1) % 624] & 0x7fffffff);
                self.state[i] = self.state[(i + 397) % 624] ^ (y >> 1) ^ if y & 1 != 0 { 0x9908b0df } else { 0 };
            }
            self.at = 0;
        }
        let mut y = self.state[self.at];
        self.at += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }
    fn sample(&mut self) -> f64 {
        let a = self.next() >> 5;
        let b = self.next() >> 6;
        (f64::from(a) * 67108864.0 + f64::from(b)) / 9007199254740992.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_store::{pack_mesh, parse_mesh};
    use serde_json::json;

    #[test]
    fn random_stream_matches_integer_seeded_python_authoring() {
        let mut random = Random::new(7319);
        for expected in
            [0.27746334049176347, 0.7853518940996139, 0.5979703527912313, 0.20969823463412351, 0.8839413120985173]
        {
            assert_eq!(random.sample(), expected);
        }
    }

    fn fixture() -> serde_json::Value {
        json!({"positions":[0,0,0,1,0,0,0,1,0],"normals":[0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,0,1],"indices":[0,1,2],"lods":[],
          "instances":{"min":[0,0,0],"max":[1,1,1],"recipe":{"version":1,"seed":7319,"random_draws":5,"stride":1,"inputs":[0,1,2],"tables":[],
            "nodes":[{"op":"input","lane":0},{"op":"random","index":0}],"output":vec![1;20]}}})
    }

    #[test]
    fn packing_keeps_recipes_and_expands_identical_records() {
        let json = serde_json::to_vec(&fixture()).unwrap();
        let mesh = parse_mesh(&json).unwrap();
        let packed = pack_mesh(&json).unwrap();
        assert!(packed.starts_with(b"PMRP"));
        assert_eq!(pack_mesh(&packed).unwrap(), packed);
        let decoded = parse_mesh(&packed).unwrap();
        assert_eq!(mesh.instances.unwrap().data, decoded.instances.unwrap().data);
        for end in 0..packed.len() {
            assert!(parse_mesh(&packed[..end]).is_err(), "truncated at {end}");
        }
        let mut extra = packed;
        extra.push(0);
        assert!(parse_mesh(&extra).is_err());
        let mut invalid = fixture();
        invalid["instances"]["recipe"]["inputs"] = json!([5]);
        assert!(parse_mesh(&serde_json::to_vec(&invalid).unwrap()).is_err());
        invalid["instances"]["recipe"]["inputs"] = json!([0]);
        invalid["instances"]["recipe"]["nodes"][0] = json!({"op":"sin","input":1});
        assert!(parse_mesh(&serde_json::to_vec(&invalid).unwrap()).is_err());
    }
}
