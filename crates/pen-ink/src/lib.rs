//! Optional Ink persistence tooling for Pen hosts. Ink owns the wire format;
//! pen-runtime only sees the StateAdapter contract. No compiler is required.
mod inspect;
mod report;
pub mod saved;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use pen_protocol::boundary::whole;
use pen_runtime::state::{self, Migration, Section, StateAdapter};
use report::{DECLARED, SectionOutcome, describe_restore, restore_report};
use saved::SavedTypes;

pub const FORMAT: &str = "ink-persist-v1";

/// Select explicitly; missing configuration is handled by hosts as opaque.
/// Old saved.json declarations need a `format` entry before being interpreted.
pub fn adapter(text: &str) -> Result<Arc<dyn StateAdapter>> {
    let json: serde_json::Value = serde_json::from_str(text)?;
    match json.get("format").and_then(serde_json::Value::as_str).context("saved.json needs an explicit format")? {
        "opaque" => {
            ensure!(json.get("types").is_none(), "opaque state cannot declare Ink saved types");
            Ok(state::opaque())
        }
        FORMAT => {
            let types = if json.get("types").is_some() { SavedTypes::parse(text)? } else { SavedTypes::default() };
            saved::check(&types)?;
            Ok(Arc::new(Ink { types }))
        }
        other => bail!("unsupported state format {other}"),
    }
}

#[derive(Default)]
pub struct Ink {
    pub types: SavedTypes,
}

impl StateAdapter for Ink {
    fn sections<'a>(&self, state: &'a [f32]) -> Option<Vec<Section<'a>>> {
        inspect::sections(state)
    }

    fn lane_label(&self, section: &Section<'_>, offset: usize) -> String {
        let Ok(stride) = usize::try_from(section.width) else { return format!("lane {offset}") };
        if stride == 0 {
            return format!("lane {offset}");
        }
        let (entity, within) = (offset / (stride + 1), offset % (stride + 1));
        if within == 0 { format!("entity {entity}'s flag") } else { format!("entity {entity} lane {}", within - 1) }
    }

    fn tuned_values(&self, state: &[f32]) -> Option<HashMap<u32, f32>> {
        let mut tuned = HashMap::new();
        // Ink's persist section 11 holds (tunable ID, value) pairs.
        for section in self.sections(state)?.into_iter().filter(|section| section.key == 11) {
            let (pairs, tail) = section.lanes.as_chunks::<2>();
            if !tail.is_empty() {
                return None;
            }
            for &[id, value] in pairs {
                tuned.insert(whole(id)?, value);
            }
        }
        Some(tuned)
    }

    fn report(&self, response: &[f32]) -> Option<String> {
        describe_restore(restore_report(response).as_deref())
    }

    fn migrate(&self, state: &[f32], response: &[f32]) -> Result<Option<Box<dyn Migration>>> {
        let Some(outcomes) = restore_report(response) else { return Ok(None) };
        let targets = saved::declared_resets(&self.types, &outcomes);
        if targets.is_empty() {
            return Ok(None);
        }
        let (state, keys) = saved::migrate(state, &self.types, &targets)?;
        if keys.is_empty() {
            return Ok(None);
        }
        Ok(Some(Box::new(InkMigration { state, keys, outcomes })))
    }
}

struct InkMigration {
    state: Vec<f32>,
    keys: Vec<u32>,
    outcomes: Vec<SectionOutcome>,
}

impl Migration for InkMigration {
    fn state(&self) -> &[f32] {
        &self.state
    }

    fn report(&self, response: &[f32]) -> Option<String> {
        let mut outcomes = restore_report(response)?;
        for outcome in &mut outcomes {
            if outcome.fate == 0
                && self.keys.contains(&outcome.key)
                && let Some(original) = self.outcomes.iter().find(|before| before.key == outcome.key)
            {
                outcome.fate = DECLARED;
                outcome.saved = original.saved;
            }
        }
        describe_restore(Some(&outcomes))
    }
}

#[cfg(test)]
mod runtime_tests;

#[cfg(test)]
mod tests;
