//! A game's own events and render sections, which its `pen.json` declares
//! under `protocol`, so that it adds to the packets without a protocol of
//! its own:
//!
//! ```json
//! "protocol": {
//!   "events": { "Kick": { "tag": 1, "fields": { "x": "f32", "z": "f32" } } },
//!   "sections": { "Score": { "tag": 1, "fields": { "points": "integer", "multiplier": "f32" } } }
//! }
//! ```
//!
//! pen's protocol carries them generically: an event as the `Game` input
//! event (its tag and up to three values), a section after the render
//! packet, in its extensions block (its tag and its lanes). `pen codegen`
//! writes the game's typed bindings for them into `pen/extensions.blot`,
//! and pen's tools (script steps, `pen packets`) name them by these
//! declarations.
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

/// The most fields one of a game's events has: the event lanes left after
/// its tag.
pub const EVENT_FIELDS: usize = 3;
/// The highest tag (pen's MAX_EXTENSION_TAG).
pub const MAX_TAG: u32 = 65_535;
/// The largest whole number an `integer` field holds exactly in a lane.
pub const MAX_INTEGER: u32 = 16_777_215;

/// What one of a declaration's fields holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
    /// Any finite number.
    F32,
    /// A whole number from 0 to [`MAX_INTEGER`].
    Integer,
    /// 0 or 1, a Bool in Blot.
    Flag,
}

impl FieldKind {
    fn parse(text: &str) -> Option<FieldKind> {
        Some(match text {
            "f32" => FieldKind::F32,
            "integer" => FieldKind::Integer,
            "flag" => FieldKind::Flag,
            _ => return None,
        })
    }

    fn blot_type(self) -> &'static str {
        match self {
            FieldKind::F32 => "F32",
            FieldKind::Integer => "U32",
            FieldKind::Flag => "Bool",
        }
    }

    /// Whether a lane holds a value of this kind.
    fn holds(self, lane: f32) -> bool {
        match self {
            FieldKind::F32 => lane.is_finite(),
            FieldKind::Integer => lane.fract() == 0.0 && (0.0..=MAX_INTEGER as f32).contains(&lane),
            FieldKind::Flag => lane == 0.0 || lane == 1.0,
        }
    }
}

/// A lane [`FieldKind::Integer`] holds, as the whole number it is.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "checked whole and in range")]
fn whole(lane: f32) -> u32 {
    lane as u32
}

/// One declared event or section: its name, tag, words and fields.
#[derive(Clone, Debug, PartialEq)]
pub struct Declared {
    pub name: String,
    pub tag: u32,
    pub doc: Option<String>,
    pub fields: Vec<(String, FieldKind)>,
}

impl Declared {
    /// Its lanes from `values`, one per field, checked.
    pub fn lanes(&self, values: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            values.len() == self.fields.len(),
            "{} takes {} value{} ({}), not {}",
            self.name,
            self.fields.len(),
            if self.fields.len() == 1 { "" } else { "s" },
            self.fields.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>().join(", "),
            values.len()
        );
        for ((name, kind), value) in self.fields.iter().zip(values) {
            ensure!(kind.holds(*value), "{}.{name} takes {kind:?} values, not {value}", self.name);
        }
        Ok(values.to_vec())
    }

    /// A section's lanes as its records, each an object of its fields.
    pub fn records(&self, values: &[f32]) -> Result<Vec<Value>> {
        let width = self.fields.len().max(1);
        ensure!(values.len().is_multiple_of(width), "{} has {} lanes, not records of {width}", self.name, values.len());
        values
            .chunks(width)
            .map(|record| {
                let mut object = serde_json::Map::new();
                for ((name, kind), value) in self.fields.iter().zip(record) {
                    ensure!(kind.holds(*value), "{}.{name} holds {value}, not a {kind:?}", self.name);
                    let value = match kind {
                        FieldKind::F32 => serde_json::json!(value),
                        FieldKind::Integer => serde_json::json!(whole(*value)),
                        FieldKind::Flag => Value::Bool(*value == 1.0),
                    };
                    object.insert(name.clone(), value);
                }
                Ok(Value::Object(object))
            })
            .collect()
    }
}

/// A checkout's declarations: its events and its sections.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Declarations {
    pub events: Vec<Declared>,
    pub sections: Vec<Declared>,
}

/// Names the bindings or Blot's prelude have already: the constructor of
/// pen's events, the bindings' own types and Blot's.
const TAKEN: [&str; 11] = ["Pen", "Event", "Input", "Array", "List", "Maybe", "Result", "Bool", "U32", "F32", "Unit"];

/// A Blot type name: a capital, then letters and digits.
fn type_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_uppercase()) && chars.all(|c| c.is_ascii_alphanumeric())
}

/// Blot's words, which name no field (blot2's grammar.baba), and `self`.
const KEYWORDS: &str = "as borrow break case cleanup code complete concurrent const data descriptor do effect else \
                        entry ever exists fn for forall from if import in infix infixl infixr is let many of once \
                        pack region return row self shared then type unique unpack use where";

/// A Blot value name: lowercase letters, digits and underscores, and not
/// one of its words.
fn value_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        && !KEYWORDS.split_whitespace().any(|word| word == name)
}

fn declared(what: &str, value: &Value, most: Option<usize>) -> Result<Vec<Declared>> {
    let Some(entries) = value.as_object() else {
        ensure!(value.is_null(), "protocol.{what} is an object of declarations by name");
        return Ok(Vec::new());
    };
    let mut all: Vec<Declared> = Vec::new();
    for (name, entry) in entries {
        let place = format!("protocol.{what}.{name}");
        ensure!(type_name(name), "{place}: a name is a capital, then letters and digits");
        ensure!(!TAKEN.contains(&name.as_str()), "{place}: {name} is taken by the bindings or Blot");
        let tag = entry["tag"].as_u64().filter(|tag| (1..=u64::from(MAX_TAG)).contains(tag));
        let tag = u32::try_from(tag.with_context(|| format!("{place}: tag is a whole number from 1 to {MAX_TAG}"))?)?;
        ensure!(all.iter().all(|other| other.tag != tag), "{place}: tag {tag} is taken");
        let doc = match &entry["doc"] {
            Value::Null => None,
            Value::String(text) => Some(text.clone()),
            _ => bail!("{place}: doc is text"),
        };
        let mut fields = Vec::new();
        if let Some(object) = entry["fields"].as_object() {
            for (field, kind) in object {
                ensure!(
                    value_name(field),
                    "{place}.{field}: a field's name is lowercase, digits and _, and no Blot word"
                );
                let kind = kind.as_str().and_then(FieldKind::parse);
                fields.push((field.clone(), kind.with_context(|| format!("{place}.{field} is f32, integer or flag"))?));
            }
        } else {
            ensure!(entry["fields"].is_null(), "{place}: fields is an object of kinds by name");
        }
        if let Some(most) = most {
            ensure!(fields.len() <= most, "{place}: an event has at most {most} fields");
        } else {
            ensure!(!fields.is_empty(), "{place}: a section has fields");
        }
        let known = ["tag", "doc", "fields"];
        if let Some(key) = entry.as_object().and_then(|entry| entry.keys().find(|key| !known.contains(&key.as_str()))) {
            bail!("{place}: unknown key {key}");
        }
        all.push(Declared { name: name.clone(), tag, doc, fields });
    }
    Ok(all)
}

impl Declarations {
    /// A `pen.json`'s `protocol` block.
    pub fn parse(value: &Value) -> Result<Declarations> {
        if value.is_null() {
            return Ok(Declarations::default());
        }
        let Some(object) = value.as_object() else { bail!("protocol is an object of events and sections") };
        if let Some(key) = object.keys().find(|key| !["events", "sections"].contains(&key.as_str())) {
            bail!("protocol: unknown key {key}");
        }
        let declarations = Declarations {
            events: declared("events", &value["events"], Some(EVENT_FIELDS))?,
            sections: declared("sections", &value["sections"], None)?,
        };
        for event in &declarations.events {
            ensure!(
                declarations.sections.iter().all(|section| section.name != event.name),
                "protocol: {} names both an event and a section",
                event.name
            );
        }
        Ok(declarations)
    }

    /// The declarations in the `pen.json` of the checkout at `root`; none
    /// without one.
    pub fn of_checkout(root: &Path) -> Result<Declarations> {
        let file = root.join("pen.json");
        let text = match std::fs::read_to_string(&file) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Declarations::default()),
            Err(error) => return Err(error).with_context(|| format!("reading {}", file.display())),
        };
        let config: Value = serde_json::from_str(&text).with_context(|| format!("reading {}", file.display()))?;
        Declarations::parse(&config["protocol"]).with_context(|| format!("{}", file.display()))
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty() && self.sections.is_empty()
    }

    /// The event called `name` (as declared, or in lowercase).
    pub fn event(&self, name: &str) -> Option<&Declared> {
        self.events.iter().find(|event| event.name == name || event.name.eq_ignore_ascii_case(name))
    }

    /// The section with `tag`.
    pub fn section(&self, tag: u32) -> Option<&Declared> {
        self.sections.iter().find(|section| section.tag == tag)
    }

    /// A section's lanes, by its declaration: its name and its records.
    pub fn describe(&self, tag: u32, values: &[f32]) -> Result<Value> {
        let section = self.section(tag).with_context(|| format!("no section of the game's has tag {tag}"))?;
        Ok(serde_json::json!({ "section": section.name, "records": section.records(values)? }))
    }
}

fn comment(text: &str) -> Vec<String> {
    crate::model::wrap(text, 78).iter().map(|line| format!("// {line}")).collect()
}

/// `pen/extensions.blot` for `declarations`, under `header`'s lines.
pub fn emit(declarations: &Declarations, header: &[String]) -> String {
    let snake = |name: &str| crate::model::snake(name);
    let mut out = header.to_vec();
    out.extend(
        "//\n// The game's own events and render sections, which its pen.json declares\n// under `protocol`, beside pen's. Its owner-aware host serves these packets.\nimport * as array from \"std/array\"\nimport * as lanes from \"./lanes\"\nimport * as protocol from \"./protocol\"\nimport * as types from \"./types\""
            .lines()
            .map(str::to_owned),
    );
    out.push(String::new());
    out.push("// Each event's and section's tag.".to_owned());
    for declared in declarations.events.iter().chain(&declarations.sections) {
        out.push(format!("const {}_tag = {}", snake(&declared.name), declared.tag));
    }
    out.push(String::new());
    out.extend(comment("The events the host sends: pen's, and the game's own."));
    for event in &declarations.events {
        if let Some(doc) = &event.doc {
            out.extend(comment(&format!("{}: {doc}", event.name)));
        }
    }
    out.push("type Event is data =".to_owned());
    out.push("  | $Pen protocol.Event".to_owned());
    for event in &declarations.events {
        if event.fields.is_empty() {
            out.push(format!("  | ${}", event.name));
        } else {
            let fields = event
                .fields
                .iter()
                .map(|(name, kind)| format!("{name}: {}", kind.blot_type()))
                .collect::<Vec<_>>()
                .join(", ");
            out.push(format!("  | ${} {{ {fields} }}", event.name));
        }
    }
    out.push(String::new());
    out.push(
        "type Input is data = $Input { elapsed: F32, width: F32, height: F32, events: &[Event], state: &[F32] }"
            .to_owned(),
    );
    out.push(String::new());
    out.extend(comment(
        "An event's lanes from `start`: one of pen's, or one of the game's, which travels as pen's Game event: \
         its tag, then its values in the lanes after, those it has no field for zero.",
    ));
    out.push("const decode_event = func (values: Array F32) => return func (start: U32) => return do:".to_owned());
    out.push("  if values[start] != U32.to_f32 protocol.game_tag:".to_owned());
    out.push("    return $Pen (protocol.decode_event values start)".to_owned());
    out.push("  let tag = lanes.integer values[start + 1] protocol.max_extension_tag".to_owned());
    out.push("  let lanes_after = #[values[start + 2], values[start + 3], values[start + 4]]".to_owned());
    out.push("  return case tag of".to_owned());
    for event in &declarations.events {
        out.push(format!("    {} => do:", event.tag));
        for unused in event.fields.len()..EVENT_FIELDS {
            out.push(format!("      use lanes.require (lanes_after[{unused}] == 0.0)"));
        }
        let fields = event
            .fields
            .iter()
            .enumerate()
            .map(|(index, (name, kind))| {
                let lane = format!("lanes_after[{index}]");
                let value = match kind {
                    FieldKind::F32 => lane,
                    FieldKind::Integer => format!("lanes.integer {lane} lanes.max_integer"),
                    FieldKind::Flag => format!("lanes.integer {lane} 1 == 1"),
                };
                format!("{name}: {value}")
            })
            .collect::<Vec<_>>();
        if fields.is_empty() {
            out.push(format!("      return ${}", event.name));
        } else {
            out.push(format!("      return ${} {{ {} }}", event.name, fields.join(", ")));
        }
    }
    out.push("    _ => @panic \"unknown game event\"".to_owned());
    out.push(String::new());
    out.extend(comment("The input packet, as protocol.decode_input_into reads it, with the game's own events."));
    out.extend(
        "const decode_input_into = func (values: Array F32) -> (events: ResizeArray Event) => return do:\n  use lanes.require (values.length >= protocol.frame_header_floats)\n  use lanes.finite values\n  let elapsed = values[0]\n  use lanes.require (elapsed >= 0.0)\n  let width = values[1]\n  let height = values[2]\n  use lanes.require (width > 0.0)\n  use lanes.require (height > 0.0)\n  let event_count = lanes.integer values[3] protocol.max_events\n  let state_count = lanes.integer values[4] protocol.max_state_floats\n  let state_start = protocol.frame_header_floats + event_count * protocol.event_floats\n  use lanes.require (values.length == state_start + state_count)\n  @storage.truncate events 0\n  for index in 0 .. event_count:\n    @storage.push events (decode_event values (protocol.frame_header_floats + index * protocol.event_floats))\n  return $Input {\n    state: &values[state_start..state_start + state_count],\n    elapsed, width, height,\n    events: &events[0..events.length],\n  }"
            .lines()
            .map(str::to_owned),
    );
    for section in &declarations.sections {
        let name = snake(&section.name);
        out.push(String::new());
        out.extend(comment(&match &section.doc {
            Some(doc) => format!("{}: {doc}", section.name),
            None => format!("{}: one of the game's own sections.", section.name),
        }));
        let fields = section
            .fields
            .iter()
            .map(|(name, kind)| format!("{name}: {}", kind.blot_type()))
            .collect::<Vec<_>>()
            .join(", ");
        out.push(format!("type {} is data = ${} {{ {fields} }}", section.name, section.name));
        out.push(format!("const {name}_floats = {}", section.fields.len()));
        let lanes = section
            .fields
            .iter()
            .map(|(field, kind)| match kind {
                FieldKind::F32 => format!("record.{field}"),
                FieldKind::Integer => format!("U32.to_f32 record.{field}"),
                FieldKind::Flag => format!("lanes.bit record.{field}"),
            })
            .collect::<Vec<_>>()
            .join(", ");
        out.extend(comment(&format!("The section of these {}s, for the frame's extensions.", section.name)));
        out.push(format!("const encode_{name} = func (records: Array {}) => return $types.Extension {{", section.name));
        out.push(format!("  tag: {name}_tag,"));
        out.push(format!("  values: #[lane | record <- records, lane <- #[{lanes}]],"));
        out.push("}".to_owned());
    }
    out.push(String::new());
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declarations() -> Declarations {
        Declarations::parse(&serde_json::json!({
            "events": {
                "Kick": { "tag": 1, "doc": "A push.", "fields": { "x": "f32", "z": "f32" } },
                "Reset": { "tag": 2 },
                "Hire": { "tag": 3, "fields": { "count": "integer", "skilled": "flag", "pay": "f32" } }
            },
            "sections": {
                "Score": { "tag": 1, "fields": { "points": "integer", "multiplier": "f32", "over": "flag" } }
            }
        }))
        .unwrap()
    }

    #[test]
    fn declarations_parse_and_check_values() {
        let declared = declarations();
        assert_eq!(declared.events.len(), 3);
        let kick = declared.event("kick").unwrap();
        assert_eq!((kick.tag, kick.fields.len()), (1, 2));
        assert_eq!(kick.lanes(&[1.5, -2.0]).unwrap(), [1.5, -2.0]);
        assert!(kick.lanes(&[1.5]).unwrap_err().to_string().contains("takes 2 values (x, z)"));
        let hire = declared.event("Hire").unwrap();
        assert!(hire.lanes(&[2.5, 1.0, 3.0]).is_err(), "integer");
        assert!(hire.lanes(&[2.0, 0.5, 3.0]).is_err(), "flag");
        assert!(hire.lanes(&[2.0, 1.0, f32::NAN]).is_err(), "finite");
        let described = declared.describe(1, &[3.0, 1.5, 0.0, 4.0, 2.0, 1.0]).unwrap();
        assert_eq!(
            described,
            serde_json::json!({ "section": "Score", "records": [
                { "points": 3, "multiplier": 1.5, "over": false },
                { "points": 4, "multiplier": 2.0, "over": true },
            ]})
        );
        assert!(declared.describe(1, &[3.0, 1.5]).is_err(), "partial record");
        assert!(declared.describe(2, &[]).is_err(), "unknown tag");
    }

    #[test]
    fn malformed_declarations_are_refused() {
        for (protocol, error) in [
            (serde_json::json!({ "event": {} }), "unknown key event"),
            (serde_json::json!({ "events": { "kick": { "tag": 1 } } }), "a name is a capital"),
            (serde_json::json!({ "events": { "Kick": { "tag": 0 } } }), "tag is a whole number"),
            (serde_json::json!({ "events": { "Kick": { "tag": 70000 } } }), "tag is a whole number"),
            (serde_json::json!({ "events": { "A": { "tag": 1 }, "B": { "tag": 1 } } }), "tag 1 is taken"),
            (
                serde_json::json!({ "events": { "A": { "tag": 1, "fields": { "a": "f32", "b": "f32", "c": "f32", "d": "f32" } } } }),
                "at most 3 fields",
            ),
            (serde_json::json!({ "events": { "A": { "tag": 1, "fields": { "a": "u8" } } } }), "f32, integer or flag"),
            (serde_json::json!({ "events": { "A": { "tag": 1, "fields": { "Big": "f32" } } } }), "lowercase"),
            (serde_json::json!({ "sections": { "A": { "tag": 1 } } }), "a section has fields"),
            (serde_json::json!({ "events": { "Pen": { "tag": 1 } } }), "Pen is taken"),
            (serde_json::json!({ "events": { "A": { "tag": 1, "fields": { "type": "f32" } } } }), "no Blot word"),
            (serde_json::json!({ "sections": { "Input": { "tag": 1, "fields": { "a": "f32" } } } }), "Input is taken"),
            (
                serde_json::json!({ "sections": { "A": { "tag": 1, "fields": { "a": "f32" }, "size": 2 } } }),
                "unknown key",
            ),
            (
                serde_json::json!({ "events": { "A": { "tag": 1 } }, "sections": { "A": { "tag": 1, "fields": { "a": "f32" } } } }),
                "both an event and a section",
            ),
        ] {
            let refused = Declarations::parse(&protocol).unwrap_err().to_string();
            assert!(refused.contains(error), "{protocol}: {refused}");
        }
        assert!(Declarations::parse(&Value::Null).unwrap().is_empty());
    }

    #[test]
    fn bindings_decode_events_and_encode_sections() {
        let source = emit(&declarations(), &["// header".to_owned()]);
        for expected in [
            "const kick_tag = 1",
            "  | $Pen protocol.Event",
            "  | $Kick { x: F32, z: F32 }",
            "  | $Reset",
            "    1 => do:\n      use lanes.require (lanes_after[2] == 0.0)\n      return $Kick { x: lanes_after[0], z: lanes_after[1] }",
            "return $Hire { count: lanes.integer lanes_after[0] lanes.max_integer, skilled: lanes.integer lanes_after[1] 1 == 1, pay: lanes_after[2] }",
            "type Score is data = $Score { points: U32, multiplier: F32, over: Bool }",
            "  values: #[lane | record <- records, lane <- #[U32.to_f32 record.points, record.multiplier, lanes.bit record.over]],",
        ] {
            assert!(source.contains(expected), "missing {expected:?} in\n{source}");
        }
    }
}
