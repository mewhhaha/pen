//! Graphics presets picked from the GPU when a game first runs. The game
//! owns its preferences (gdev's are `ink/preferences`' lanes) and declares
//! its presets beside its assets, in `presets.json`: the checkout's
//! `src/presets.json`, which `pen build` ships as `game/presets.json`.
//!
//! ```json
//! {"about": "...", "lanes": {"msaa": 0, "shadows": 4},
//!  "presets": {"low": {"msaa": 1, "shadows": 1024}, "high": {"msaa": 4, "shadows": 4096}}}
//! ```
//!
//! The host only tells GPUs apart: [`tier`] puts the adapter wgpu opened in
//! a [`Tier`] by its type, vendor and name. While the player has no stored
//! preferences (the first run, until they change a setting and the game
//! sends its preferences to be stored) the tier's preset plays to the game
//! as Preference events, as stored preferences do. A tier the file leaves
//! out falls back to the next lower one it has.
//!
//! Headless runs (tests, golden images, replays) keep the game's own
//! defaults, so that they draw alike on every machine, unless
//! PEN_GRAPHICS_TIER asks: `auto` for the GPU's tier, or a tier's name.
use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// How much GPU a machine has, as far as its adapter tells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// A software rasterizer: lavapipe, llvmpipe, WARP, SwiftShader.
    Software,
    /// A handheld PC's APU: the Steam Deck.
    Handheld,
    /// Integrated graphics that struggle (Intel's UHD and Iris), and virtual GPUs.
    Low,
    /// Capable integrated graphics (Apple silicon, AMD's Radeon iGPUs, Intel Arc), and the unknown.
    Medium,
    /// A discrete GPU.
    High,
}

impl Tier {
    pub fn name(self) -> &'static str {
        match self {
            Tier::Software => "software",
            Tier::Handheld => "handheld",
            Tier::Low => "low",
            Tier::Medium => "medium",
            Tier::High => "high",
        }
    }

    fn named(name: &str) -> Option<Tier> {
        [Tier::Software, Tier::Handheld, Tier::Low, Tier::Medium, Tier::High]
            .into_iter()
            .find(|tier| tier.name() == name)
    }

    /// The presets to look for, this tier's first, then lower ones.
    fn fallbacks(self) -> &'static [&'static str] {
        match self {
            Tier::Software => &["software", "low"],
            Tier::Handheld => &["handheld", "medium", "low"],
            Tier::Low => &["low"],
            Tier::Medium => &["medium", "low"],
            Tier::High => &["high", "medium", "low"],
        }
    }
}

const APPLE: u32 = 0x106b;
const AMD: u32 = 0x1002;
const INTEL: u32 = 0x8086;

/// The tier of the adapter `info` describes; `deck`, on a Steam Deck.
pub fn tier(info: &wgpu::AdapterInfo, deck: bool) -> Tier {
    let name = info.name.to_lowercase();
    let software = ["llvmpipe", "lavapipe", "swiftshader", "basic render driver"];
    if info.device_type == wgpu::DeviceType::Cpu || software.iter().any(|word| name.contains(word)) {
        return Tier::Software;
    }
    // The Deck's APU, "AMD Custom GPU 0405 (RADV VANGOGH)", on SteamOS or not.
    if deck || name.contains("vangogh") {
        return Tier::Handheld;
    }
    match info.device_type {
        wgpu::DeviceType::DiscreteGpu => Tier::High,
        wgpu::DeviceType::IntegratedGpu if info.vendor == APPLE || info.vendor == AMD => Tier::Medium,
        wgpu::DeviceType::IntegratedGpu if info.vendor == INTEL && name.contains("arc") => Tier::Medium,
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::VirtualGpu => Tier::Low,
        _ => Tier::Medium,
    }
}

/// A game's `presets.json`: its preferences' lanes by name, and each
/// tier's values for some of them (and words for whoever reads it, which
/// are not read here).
#[derive(Debug, Deserialize)]
pub struct Presets {
    lanes: BTreeMap<String, u32>,
    presets: BTreeMap<String, BTreeMap<String, f32>>,
}

impl Presets {
    pub fn parse(text: &str) -> Result<Presets> {
        let presets: Presets = serde_json::from_str(text)?;
        for (tier, values) in &presets.presets {
            if Tier::named(tier).is_none() {
                bail!("{tier} is not a tier: software, handheld, low, medium or high");
            }
            if let Some(name) = values.keys().find(|name| !presets.lanes.contains_key(*name)) {
                bail!("{tier}'s {name} is not among the lanes");
            }
            if let Some((name, _)) = values.iter().find(|(_, value)| !value.is_finite()) {
                bail!("{tier}'s {name} is not a number");
            }
        }
        Ok(presets)
    }

    /// The preset for `tier` (or the next lower one there is) as
    /// preference lanes and values, and its name.
    pub fn pick(&self, tier: Tier) -> Option<(&str, Vec<(u32, f32)>)> {
        let (name, values) = tier.fallbacks().iter().find_map(|name| self.presets.get_key_value(*name))?;
        let mut lanes: Vec<(u32, f32)> = values.iter().map(|(lane, value)| (self.lanes[lane], *value)).collect();
        lanes.sort_by_key(|(lane, _)| *lane);
        Some((name.as_str(), lanes))
    }
}

/// What PEN_GRAPHICS_TIER asks for: None to leave the game's defaults be.
fn asked(headless: bool) -> Result<Option<Option<Tier>>> {
    match std::env::var("PEN_GRAPHICS_TIER") {
        Err(_) if headless => Ok(None),
        Err(_) => Ok(Some(None)),
        Ok(name) if name == "auto" => Ok(Some(None)),
        Ok(name) => match Tier::named(&name) {
            Some(tier) => Ok(Some(Some(tier))),
            None => bail!("PEN_GRAPHICS_TIER={name}: auto, software, handheld, low, medium or high"),
        },
    }
}

/// The preset for a player without stored preferences, from the game's
/// `presets.json` beside `assets`, if it has one; says which and why.
pub fn first_run(assets: &Path, info: &wgpu::AdapterInfo, headless: bool) -> Vec<(u32, f32)> {
    match choose(assets, info, headless) {
        Ok(Some(chosen)) => {
            eprintln!("graphics: the {} preset for {} ({} tier)", chosen.preset, info.name, chosen.tier.name());
            chosen.values
        }
        Ok(None) => Vec::new(),
        Err(error) => {
            eprintln!("graphics: no preset: {error:#}");
            Vec::new()
        }
    }
}

/// A preset picked for a tier.
struct Chosen {
    tier: Tier,
    /// The tier's own, or the next lower one's.
    preset: String,
    values: Vec<(u32, f32)>,
}

fn choose(assets: &Path, info: &wgpu::AdapterInfo, headless: bool) -> Result<Option<Chosen>> {
    let Some(asked) = asked(headless)? else { return Ok(None) };
    let Some(directory) = assets.parent() else { return Ok(None) };
    let path = directory.join("presets.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    let presets = Presets::parse(&text).with_context(|| format!("reading {}", path.display()))?;
    let tier = asked.unwrap_or_else(|| tier(info, crate::services::platform::steam_deck()));
    Ok(presets.pick(tier).map(|(preset, values)| Chosen { tier, preset: preset.to_owned(), values }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(name: &str, vendor: u32, device_type: wgpu::DeviceType) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo { name: name.into(), vendor, ..wgpu::AdapterInfo::new(device_type, wgpu::Backend::Vulkan) }
    }

    #[test]
    fn adapters_fall_into_tiers() {
        use wgpu::DeviceType::{Cpu, DiscreteGpu, IntegratedGpu, Other, VirtualGpu};
        let cases = [
            ("NVIDIA GeForce RTX 2070 SUPER", 0x10de, DiscreteGpu, Tier::High),
            ("AMD Radeon RX 6600 (RADV NAVI23)", AMD, DiscreteGpu, Tier::High),
            ("AMD Custom GPU 0405 (RADV VANGOGH)", AMD, IntegratedGpu, Tier::Handheld),
            ("AMD Radeon 780M (RADV PHOENIX)", AMD, IntegratedGpu, Tier::Medium),
            ("Apple M2", APPLE, IntegratedGpu, Tier::Medium),
            ("Intel(R) Arc(tm) Graphics (MTL)", INTEL, IntegratedGpu, Tier::Medium),
            ("Intel(R) Iris(R) Xe Graphics", INTEL, IntegratedGpu, Tier::Low),
            ("Intel(R) UHD Graphics 620", INTEL, IntegratedGpu, Tier::Low),
            ("llvmpipe (LLVM 21.1.0, 256 bits)", 0x10005, Cpu, Tier::Software),
            ("Microsoft Basic Render Driver", 0x1414, Other, Tier::Software),
            ("SwiftShader Device (Subzero)", 0x1ae0, Cpu, Tier::Software),
            ("Virtio-GPU Venus", 0x1af4, VirtualGpu, Tier::Low),
            ("Some GPU", 0x1234, Other, Tier::Medium),
        ];
        for (name, vendor, kind, expected) in cases {
            assert_eq!(tier(&adapter(name, vendor, kind), false), expected, "{name}");
        }
        // SteamOS says it is a Deck, whatever the adapter's name.
        assert_eq!(tier(&adapter("AMD Custom GPU 0932", AMD, IntegratedGpu), true), Tier::Handheld);
        assert_eq!(tier(&adapter("llvmpipe", 0, Cpu), true), Tier::Software);
    }

    #[test]
    fn presets_pick_lanes_for_a_tier_or_the_next_lower() {
        let presets = Presets::parse(
            r#"{"about": "words", "lanes": {"msaa": 0, "shadows": 4, "bloom": 12}, "presets": {
                "low": {"msaa": 1, "shadows": 1024, "bloom": 0},
                "high": {"msaa": 4, "shadows": 4096}}}"#,
        )
        .unwrap();
        assert_eq!(presets.pick(Tier::High), Some(("high", vec![(0, 4.0), (4, 4096.0)])));
        // Neither medium nor handheld: low, the next lower one there is.
        assert_eq!(presets.pick(Tier::Medium), Some(("low", vec![(0, 1.0), (4, 1024.0), (12, 0.0)])));
        assert_eq!(presets.pick(Tier::Handheld).unwrap().0, "low");
        assert_eq!(presets.pick(Tier::Software).unwrap().0, "low");
        let only_high = Presets::parse(r#"{"lanes": {"msaa": 0}, "presets": {"high": {"msaa": 4}}}"#).unwrap();
        assert_eq!(only_high.pick(Tier::Low), None);

        for wrong in [
            r#"{"lanes": {"msaa": 0}, "presets": {"ultra": {"msaa": 4}}}"#,
            r#"{"lanes": {"msaa": 0}, "presets": {"low": {"shadows": 4}}}"#,
            r#"{"presets": {"low": {"msaa": 1}}}"#,
            r#"{"lanes": {"msaa": -1}, "presets": {}}"#,
            r#"{"lanes": {"msaa": 0}, "presets": {"low": {"msaa": "x"}}}"#,
        ] {
            assert!(Presets::parse(wrong).is_err(), "{wrong}");
        }
    }
}
