//! The packets' canonical walks: their values in the schema's order, laid out
//! by none of the packets' tricks, so that pen's tests can compare what the
//! Rust and the Squid codecs make of the same values (pen's
//! `codec_round_trip`). Rust gets `canonical_input` and `canonical_render`
//! beside the codecs; the game's `pen/protocol_canonical.sq`, which only
//! tests import, gets the same input walk and a flat render packet writer.
//!
//! The walk of a value, by its kind:
//!
//! - a number (`f32`) is itself; a whole number (a count, reference, choice,
//!   the schema, a key's code) is its value; a flag is 1 or 0;
//! - a vector is its components, a matrix its 16 lanes column by column, a
//!   fixed run of lanes those lanes;
//! - an optional value (Rust's Option) is 0, or 1 then its walk;
//! - an array is its count, then each element's walk (a tail's elements
//!   are its lanes);
//! - a record, a group of header lanes and a frame are their fields' walks
//!   in the schema's order: a group at its first lane's place, a record's
//!   children and tails inside it, and a variable record's rest after its
//!   head;
//! - an event is its tag, then its fields' walks in the schema's order.
//!
//! The render frame's walk leaves out the layout's own lanes (the magic and
//! sizes) and the sections whose values their owners carry, and takes the
//! schema and state, which Rust's frame holds apart, in their places.
use crate::model::*;
use crate::squid::{chunk, comment};
use serde_json::Value;
use std::collections::HashSet;

/// One value of the render frame's walk.
enum Member<'a> {
    /// A header lane outside the groups.
    Lane(&'a Laid),
    /// A group of header lanes, walked at its first lane's place.
    Group(&'a str),
    /// A section the frame holds, the state among them.
    Section(&'a Section),
}

/// The render frame's values, in the walk's order.
fn members(model: &Model) -> Vec<Member<'_>> {
    let mut members = Vec::new();
    let mut groups = HashSet::new();
    for entry in &model.header {
        if matches!(kind(&entry.laid.field), "magic" | "size") {
            continue;
        }
        match &entry.group {
            Some(group) => {
                if groups.insert(group.as_str()) {
                    members.push(Member::Group(group));
                }
            }
            None => members.push(Member::Lane(&entry.laid)),
        }
    }
    for entry in &model.sections {
        if entry.owner.is_none() && kind(&entry.section) != "trailing" {
            members.push(Member::Section(entry));
        }
    }
    members
}

/// A group's lanes, in the header's order.
fn group_lanes<'a>(model: &'a Model, group: &str) -> Vec<&'a Laid> {
    model.header.iter().filter(|entry| entry.group.as_deref() == Some(group)).map(|entry| &entry.laid).collect()
}

/// The lane that says whether a group is there (Rust's Option), if it has one.
fn presence(lanes: &[&Laid]) -> Option<usize> {
    lanes.iter().position(|laid| kind(&laid.field) == "present")
}

/// The record a section holds, or a children field's section holds.
fn section_record<'a>(model: &'a Model, name: &str) -> &'a Value {
    &model.section(name).section["record"]
}

/// Whether an event's field crosses as a number (Squid's F32), not a whole one.
fn crosses_as_f32(field: &Value) -> bool {
    field["wire"] == "f32"
}

/// Rust statements that push a walk's lanes onto `out`, runs of single lanes
/// joined into one `extend`.
struct Pushes {
    indent: String,
    lines: Vec<String>,
    lanes: Vec<String>,
}

impl Pushes {
    fn new(indent: usize) -> Pushes {
        Pushes { indent: " ".repeat(indent), lines: Vec::new(), lanes: Vec::new() }
    }

    fn lane(&mut self, value: String) {
        self.lanes.push(value);
    }

    fn line(&mut self, line: String) {
        self.flush();
        self.lines.push(format!("{}{line}", self.indent));
    }

    fn block(&mut self, lines: Vec<String>) {
        self.flush();
        self.lines.extend(lines);
    }

    fn flush(&mut self) {
        let indent = &self.indent;
        match self.lanes.len() {
            0 => {}
            1 => self.lines.push(format!("{indent}out.push({});", self.lanes[0])),
            _ => {
                let joined = self.lanes.join(", ");
                if indent.len() + joined.len() + 14 <= 112 {
                    self.lines.push(format!("{indent}out.extend([{joined}]);"));
                } else {
                    self.lines.push(format!("{indent}out.extend(["));
                    for line in chunk(&self.lanes, 112 - indent.len() - 4) {
                        self.lines.push(format!("{indent}    {line},"));
                    }
                    self.lines.push(format!("{indent}]);"));
                }
            }
        }
        self.lanes.clear();
    }

    fn finish(mut self) -> Vec<String> {
        self.flush();
        self.lines
    }
}

/// A Rust record's walk function: `canonical_draw`.
fn rust_walker(record: &Value) -> String {
    format!("canonical_{}", snake(s(&record["name"])))
}

/// The statements walking a list of `record`s at `path`; `out` is how the
/// walk is passed to the record's function.
fn rust_list(pushes: &mut Pushes, record: &Value, path: &str, out: &str) {
    let item = snake(s(&record["name"]));
    pushes.lane(format!("{path}.len() as f32"));
    pushes.line(format!("for {item} in &{path} {{"));
    pushes.line(format!("    {}({out}, {item});", rust_walker(record)));
    pushes.line("}".to_owned());
}

/// `values`, the lanes of an Option at `path`: 0, or 1 then their count (when
/// `counted`) and lanes.
fn rust_optional(pushes: &mut Pushes, path: &str, counted: bool) {
    let indent = pushes.indent.clone();
    let mut some = Pushes::new(indent.len() + 8);
    some.lane("1.0".to_owned());
    if counted {
        some.lane("values.len() as f32".to_owned());
    }
    some.line("out.extend_from_slice(values);".to_owned());
    let mut lines = vec![format!("{indent}match &{path} {{"), format!("{indent}    Some(values) => {{")];
    lines.extend(some.finish());
    lines.extend([format!("{indent}    }}"), format!("{indent}    None => out.push(0.0),"), format!("{indent}}}")]);
    pushes.block(lines);
}

/// The walk of a field of `kind` at `path`.
fn rust_field(model: &Model, pushes: &mut Pushes, field: &Value, path: &str, out: &str) {
    match kind(field) {
        "f32" => pushes.lane(path.to_owned()),
        "count" | "choice" | "reference" | "present" => pushes.lane(format!("{path} as f32")),
        "flag" => pushes.lane(format!("f32::from({path})")),
        "vector" => {
            for component in array(&field["type"]["components"]) {
                pushes.lane(format!("{path}.{}", s(component)));
            }
        }
        "matrix" | "lanes" => pushes.line(format!("out.extend_from_slice(&{path});")),
        "optional" => rust_optional(pushes, path, false),
        "tail" => rust_optional(pushes, path, true),
        "children" => rust_list(pushes, section_record(model, s(&field["section"])), path, out),
        _ => unreachable!(),
    }
}

/// The walk of an input packet and of a render packet, for pen-protocol's
/// codec module.
pub fn rust(model: &Model) -> Vec<String> {
    let mut out = "    /// The walk of an input packet's values that pen's tests compare with the game's\n    /// (`pen/protocol_canonical.sq`): elapsed, the viewport's width and height, the events'\n    /// count, each event's tag and then its fields in the schema's order (a key as its code),\n    /// then the state's count and lanes. It lays nothing out as the packet does.\n    pub fn canonical_input(frame: &Frame, elapsed: f64, state: &[f32]) -> Vec<f32> {\n        let viewport = frame.viewport;\n        let mut out = vec![elapsed as f32, viewport.width as f32, viewport.height as f32, frame.events.len() as f32];\n        for event in &frame.events {\n            match event {".lines().map(str::to_owned).collect::<Vec<_>>();
    for event in model.events() {
        let fields = event_fields(event);
        for host in hosts(event) {
            let name = s(&host["name"]);
            let mut lanes = vec![format!("{}.0", event["tag"])];
            let pattern = if host["fields"].is_object() {
                // A host's own fields do not cross: the event's lanes are zero.
                lanes.extend(fields.iter().map(|_| "0.0".to_owned()));
                format!("GameEvent::{name} {{ .. }}")
            } else {
                for (name, field) in &fields {
                    lanes.push(match field["wire"].as_str() {
                        Some("key") => format!("key_code({name})"),
                        _ if field["rust"] == "f32" => format!("*{name}"),
                        _ => format!("*{name} as f32"),
                    });
                }
                if !event["value"].is_null() {
                    format!("GameEvent::{name}(value)")
                } else if !fields.is_empty() {
                    let names = fields.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>().join(", ");
                    format!("GameEvent::{name} {{ {names} }}")
                } else {
                    format!("GameEvent::{name}")
                }
            };
            if lanes.len() == 1 {
                out.push(format!("                {pattern} => out.push({}),", lanes[0]));
            } else {
                out.push(format!("                {pattern} => out.extend([{}]),", lanes.join(", ")));
            }
        }
    }
    out.extend("            }\n        }\n        out.push(state.len() as f32);\n        out.extend_from_slice(state);\n        out\n    }\n".split('\n').map(str::to_owned));

    out.extend(
        "    mod canonical_sink {\n        pub(super) trait CanonicalSink {\n            fn push(&mut self, value: f32);\n            fn extend<const N: usize>(&mut self, values: [f32; N]);\n            fn extend_from_slice(&mut self, values: &[f32]);\n        }\n\n        impl CanonicalSink for Vec<f32> {\n            #[inline]\n            fn push(&mut self, value: f32) {\n                Vec::push(self, value);\n            }\n            #[inline]\n            fn extend<const N: usize>(&mut self, values: [f32; N]) {\n                Vec::extend_from_slice(self, &values);\n            }\n            #[inline]\n            fn extend_from_slice(&mut self, values: &[f32]) {\n                Vec::extend_from_slice(self, values);\n            }\n        }\n\n        pub(super) struct Fnv1aSink(pub(super) u64);\n\n        impl Fnv1aSink {\n            #[inline]\n            fn write_lane(&mut self, lane: f32) {\n                for byte in lane.to_bits().to_le_bytes() {\n                    self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);\n                }\n            }\n        }\n\n        impl CanonicalSink for Fnv1aSink {\n            #[inline]\n            fn push(&mut self, value: f32) {\n                self.write_lane(value);\n            }\n            #[inline]\n            fn extend<const N: usize>(&mut self, values: [f32; N]) {\n                for value in values {\n                    self.write_lane(value);\n                }\n            }\n            #[inline]\n            fn extend_from_slice(&mut self, values: &[f32]) {\n                for &value in values {\n                    self.write_lane(value);\n                }\n            }\n        }\n    }\n\n    /// The walk of a render packet's values that pen's tests compare with the game's\n    /// (`pen/protocol_canonical.sq`): the header's values and then the sections, each in the\n    /// schema's order, as described in pen-codegen's canonical.rs. It takes the schema and\n    /// the state, which the frame does not hold, where the packet has them.\n    pub fn canonical_render(frame: &RenderFrame, schema: u32, state: &[f32]) -> Vec<f32> {\n        let mut out = Vec::new();\n        walk_canonical_render(&mut out, frame, schema, state);\n        out\n    }\n\n    /// Streaming FNV-1a hash of [`canonical_render`] without allocating a buffer.\n    pub fn hash_canonical_render(frame: &RenderFrame, schema: u32, state: &[f32]) -> u64 {\n        let mut out = canonical_sink::Fnv1aSink(0xcbf2_9ce4_8422_2325);\n        walk_canonical_render(&mut out, frame, schema, state);\n        out.0\n    }\n\n    fn walk_canonical_render<S: canonical_sink::CanonicalSink>(out: &mut S, frame: &RenderFrame, schema: u32, state: &[f32]) {"
            .lines()
            .map(str::to_owned),
    );
    let mut pushes = Pushes::new(8);
    for member in members(model) {
        match member {
            Member::Lane(laid) if kind(&laid.field) == "schema" => pushes.lane("schema as f32".to_owned()),
            Member::Lane(laid) => rust_field(model, &mut pushes, &laid.field, &format!("frame.{}", laid.name), "out"),
            Member::Group(group) => {
                let lanes = group_lanes(model, group);
                if presence(&lanes).is_none() {
                    for laid in &lanes {
                        rust_field(model, &mut pushes, &laid.field, &format!("frame.{group}.{}", laid.name), "out");
                    }
                    continue;
                }
                // A group with a presence lane is an Option, its presence lane the index
                // Rust's group holds.
                let mut some = Pushes::new(16);
                some.lane("1.0".to_owned());
                for laid in &lanes {
                    rust_field(model, &mut some, &laid.field, &format!("{group}.{}", laid.name), "out");
                }
                let mut lines =
                    vec![format!("        match &frame.{group} {{"), format!("            Some({group}) => {{")];
                lines.extend(some.finish());
                lines.extend([
                    "            }".to_owned(),
                    "            None => out.push(0.0),".to_owned(),
                    "        }".to_owned(),
                ]);
                pushes.block(lines);
            }
            Member::Section(entry) => {
                let section = &entry.section;
                let name = s(&section["name"]);
                match kind(section) {
                    "lanes" if section["apart"] == true => {
                        pushes.lane("state.len() as f32".to_owned());
                        pushes.line("out.extend_from_slice(state);".to_owned());
                    }
                    "lanes" if section["nullable"] == true => {
                        rust_optional(&mut pushes, &format!("frame.{name}"), true)
                    }
                    "lanes" => {
                        pushes.lane(format!("frame.{name}.len() as f32"));
                        pushes.line(format!("out.extend_from_slice(&frame.{name});"));
                    }
                    _ => rust_list(&mut pushes, &section["record"], &format!("frame.{name}"), "out"),
                }
            }
        }
    }
    out.extend(pushes.finish());
    out.push("    }".to_owned());

    // Each record's walk.
    let mut emitted = HashSet::new();
    for entry in &model.sections {
        let mut records: Vec<(&Value, Vec<Laid>)> =
            entry.fixed.iter().chain(entry.item.iter()).map(|layout| (&layout.record, layout.fields.clone())).collect();
        if kind(&entry.section) == "variable" {
            records.push((&entry.section["record"], entry.head.clone()));
        }
        for (record, fields) in records {
            if !emitted.insert(s(&record["name"]).to_owned()) {
                continue;
            }
            let variable = snake(s(&record["name"]));
            out.push(String::new());
            out.push(format!(
                "    fn {}<S: canonical_sink::CanonicalSink>(out: &mut S, {variable}: &{}) {{",
                rust_walker(record),
                s(&record["name"])
            ));
            let mut pushes = Pushes::new(8);
            for laid in &fields {
                rust_field(model, &mut pushes, &laid.field, &format!("{variable}.{}", laid.name), "out");
            }
            let rest = &record["rest"];
            if rest.is_object() {
                let path = format!("{variable}.{}", s(&rest["name"]));
                if kind(rest) == "repeat" {
                    rust_list(&mut pushes, &rest["record"], &path, "out");
                } else {
                    pushes.lane(format!("{path}.len() as f32"));
                    pushes.line(format!("out.extend_from_slice(&{path});"));
                }
            }
            out.extend(pushes.finish());
            out.push("    }".to_owned());
        }
    }
    out
}

/// Copy a fixed-width canonical value into its schema-defined packet offset.
fn stream_field(out: &mut Vec<String>, field: &Laid, pool: &str, indent: &str) {
    let offset = field.offset;
    match kind(&field.field) {
        "optional" => {
            let width = n(&field.field["count"]);
            out.push(format!("{indent}let present = lanes.integer canonical[at] 1"));
            out.push(format!("{indent}at := at + 1"));
            out.push(format!("{indent}if present == 1:"));
            out.push(format!("{indent}  stream.{pool}[base + {offset}] := {width}.0"));
            for index in 0..width {
                out.push(format!("{indent}  stream.{pool}[base + {}] := canonical[at + {index}]", offset + index + 1));
            }
            out.push(format!("{indent}  at := at + {width}"));
        }
        "tail" => {
            let unit = n(&field.field["unit"]);
            out.push(format!("{indent}let present = lanes.integer canonical[at] 1"));
            out.push(format!("{indent}at := at + 1"));
            out.push(format!("{indent}if present == 1:"));
            out.push(format!("{indent}  let count = lanes.integer canonical[at] lanes.max_integer"));
            out.push(format!("{indent}  lanes.require (count % {unit} == 0)"));
            out.push(format!("{indent}  at := at + 1"));
            out.push(format!("{indent}  stream.{pool}[base + {offset}] := U32.to_f32 (count / {unit})"));
            out.push(format!("{indent}  for index in 0 .. count:"));
            out.push(format!("{indent}    @storage.push stream.extras canonical[at + index]"));
            out.push(format!("{indent}  at := at + count"));
        }
        "children" => {
            // The schema currently has one child-record section, inset_draws.
            let child = snake(&singular(s(&field.field["section"])));
            out.push(format!("{indent}let count = lanes.integer canonical[at] lanes.max_integer"));
            out.push(format!("{indent}at := at + 1"));
            out.push(format!("{indent}stream.{pool}[base + {offset}] := U32.to_f32 count"));
            out.push(format!("{indent}stream.at := at"));
            out.push(format!("{indent}for 0 .. count:"));
            out.push(format!("{indent}  stream := {child}_into canonical stream"));
            out.push(format!("{indent}at := stream.at"));
        }
        _ => {
            for index in 0..field.lanes {
                out.push(format!("{indent}stream.{pool}[base + {}] := canonical[at + {index}]", offset + index));
            }
            out.push(format!("{indent}at := at + {}", field.lanes));
        }
    }
}

fn stream_record(entry: &Section) -> Vec<String> {
    let name = singular(s(&entry.section["name"]));
    let (fields, width) =
        entry.fixed.as_ref().map_or((&entry.head, entry.head_lanes), |layout| (&layout.fields, layout.lanes));
    // A record another record owns is written to the children's pool.
    let pool = if entry.owner.is_some() { "children" } else { "output" };
    let mut out = vec![
        String::new(),
        format!("func {name}_into (canonical: #[F32]) -> (stream: own Stream): Stream =>"),
        "  let at = stream.at".to_owned(),
        format!("  let base = stream.{pool}.length"),
        format!("  for 0 .. {width}:"),
        format!("    @storage.push stream.{pool} 0.0"),
    ];
    for field in fields {
        stream_field(&mut out, field, pool, "  ");
    }
    if kind(&entry.section) == "variable" {
        out.push("  let count = lanes.integer canonical[at] lanes.max_integer".to_owned());
        out.push("  at := at + 1".to_owned());
        if entry.item.is_some() {
            out.push(format!("  stream.{pool}[base + {}] := U32.to_f32 count", width - 1));
        }
        out.push("  for 0 .. count:".to_owned());
        if let Some(layout) = &entry.item {
            out.push(format!("    let base = stream.{pool}.length"));
            out.push(format!("    for 0 .. {}:", layout.lanes));
            out.push(format!("      @storage.push stream.{pool} 0.0"));
            for field in &layout.fields {
                stream_field(&mut out, field, pool, "    ");
            }
        } else {
            out.push(format!("    @storage.push stream.{pool} canonical[at]"));
            out.push("    at := at + 1".to_owned());
        }
    }
    out.push("  stream.at := at".to_owned());
    out.push("  return stream".to_owned());
    out
}

fn stream_header(out: &mut Vec<String>, field: &Laid, indent: &str) {
    for index in 0..field.lanes {
        let value = if kind(&field.field) == "present" {
            // Canonical optional groups use a separate presence flag and zero-based reference.
            "canonical[at] + 1.0".to_owned()
        } else {
            format!("canonical[at + {index}]")
        };
        out.push(format!("{indent}stream.output[{}] := {value}", field.offset + index));
    }
    out.push(format!("{indent}at := at + {}", field.lanes));
}

/// Decode a canonical walk into flat packet lanes. Deferred child records and
/// tail lanes have separate owners so no nested owner/borrow escapes a reader.
fn render_stream(model: &Model) -> Vec<String> {
    let mut out = vec![
        String::new(),
        "// What a walk has made so far: where it reads next, the packet's lanes, and".to_owned(),
        "// the child records and tail lanes that follow them. A writer takes it over".to_owned(),
        "// and returns it.".to_owned(),
        "type Stream = { at: U32, output: #[F32], children: #[F32], extras: #[F32] }".to_owned(),
    ];
    for entry in &model.sections {
        if matches!(kind(&entry.section), "records" | "variable") {
            out.extend(stream_record(entry));
        }
    }
    out.extend([
        String::new(),
        "// Canonical render values streamed into the packet it is given and returns.".to_owned(),
        "func render_packet_from_canonical (canonical: #[F32]) -> (output: own #[F32]): #[F32] =>".to_owned(),
        "  lanes.finite canonical".to_owned(),
        "  @storage.truncate output 0".to_owned(),
        "  for 0 .. protocol.packet_header_floats:".to_owned(),
        "    @storage.push output 0.0".to_owned(),
        "  let children: #[F32] = #[]".to_owned(),
        "  let extras: #[F32] = #[]".to_owned(),
        "  let stream: Stream = { at: 0, output, children, extras }".to_owned(),
        "  stream.output[0] := protocol.packet_magic".to_owned(),
        "  let at: U32 = 0".to_owned(),
    ]);
    for member in members(model) {
        match member {
            Member::Lane(field) => stream_header(&mut out, field, "  "),
            Member::Group(group) => {
                let fields = group_lanes(model, group);
                let indent = if presence(&fields).is_some() {
                    out.push("  let present = lanes.integer canonical[at] 1".to_owned());
                    out.push("  at := at + 1".to_owned());
                    out.push("  if present == 1:".to_owned());
                    "    "
                } else {
                    "  "
                };
                for field in fields {
                    stream_header(&mut out, field, indent);
                }
            }
            Member::Section(_) => {}
        }
    }
    for entry in &model.sections {
        let name = s(&entry.section["name"]);
        out.push(format!("  let {name}_start: U32 = stream.output.length"));
        match kind(&entry.section) {
            "trailing" => {
                out.push("  for value in stream.extras:".to_owned());
                out.push("    @storage.push stream.output value".to_owned());
            }
            _ if entry.owner.is_some() => {
                let width = entry.fixed.as_ref().unwrap().lanes;
                out.push(format!("  let {name}_count: U32 = stream.children.length / {width}"));
                out.push("  for value in stream.children:".to_owned());
                out.push("    @storage.push stream.output value".to_owned());
            }
            "lanes" => {
                out.push(format!("  let {name}_count: U32 = 0"));
                let indent = if entry.section["nullable"] == true {
                    out.push("  let present = lanes.integer canonical[at] 1".to_owned());
                    out.push("  at := at + 1".to_owned());
                    out.push("  if present == 1:".to_owned());
                    "    "
                } else {
                    "  "
                };
                out.push(format!("{indent}{name}_count := lanes.integer canonical[at] lanes.max_integer"));
                out.push(format!("{indent}at := at + 1"));
                out.push(format!("{indent}for index in 0 .. {name}_count:"));
                out.push(format!("{indent}  @storage.push stream.output canonical[at + index]"));
                out.push(format!("{indent}at := at + {name}_count"));
            }
            _ => {
                out.push(format!("  let {name}_count = lanes.integer canonical[at] lanes.max_integer"));
                out.push("  at := at + 1".to_owned());
                let record = singular(name);
                out.push("  stream.at := at".to_owned());
                out.push(format!("  for 0 .. {name}_count:"));
                out.push(format!("    stream := {record}_into canonical stream"));
                out.push("  at := stream.at".to_owned());
            }
        }
        for header in &model.header {
            let field = &header.laid;
            if kind(&field.field) == "size" && s(&field.field["section"]) == name {
                let count = if field.field["of"] == "count" {
                    format!("{name}_count")
                } else {
                    format!("(stream.output.length - {name}_start)")
                };
                out.push(format!("  stream.output[{}] := U32.to_f32 {count}", field.offset));
            }
        }
    }
    out.push("  lanes.require (at == canonical.length)".to_owned());
    out.push("  return stream.output".to_owned());
    out
}

/// The game's `pen/protocol_canonical.sq`: the input walk, and the render
/// walk streamed into a packet, with its child draws and extras gathered beside it.
pub fn squid(model: &Model, top: &[String]) -> String {
    let mut out = top.to_vec();
    out.extend([
        "//".to_owned(),
        "// The packets' canonical walks (pen-codegen's canonical.rs), which pen's tests".to_owned(),
        "// compare with Rust's. Only tests import this; the game must not.".to_owned(),
        "import * as array from \"std/array\"".to_owned(),
        "import * as lanes from \"./lanes\"".to_owned(),
        "import * as protocol from \"./protocol\"".to_owned(),
        String::new(),
    ]);
    out.extend(comment(Some("An event's walk: its tag, then its fields in the schema's order.")));
    out.push("func canonical_event (event: protocol.Event) => case event of".to_owned());
    for event in model.core_events() {
        let name = s(&event["name"]);
        let fields = event_fields(event);
        let value = |local: &str, field: &Value| {
            if crosses_as_f32(field) { local.to_owned() } else { format!("U32.to_f32 {local}") }
        };
        let mut lanes = vec![format!("{}.0", event["tag"])];
        let pattern = if !event["value"].is_null() {
            lanes.push(value("value", &event["value"]));
            format!("$protocol.{name} value")
        } else if !fields.is_empty() {
            lanes.extend(fields.iter().map(|(name, field)| value(name, field)));
            let names = fields.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>().join(", ");
            format!("$protocol.{name} {{ {names} }}")
        } else {
            format!("$protocol.{name}")
        };
        out.push(format!("  {pattern} => #[{}]", lanes.join(", ")));
    }
    out.push(String::new());
    out.extend(comment(Some(
        "The input's walk, as pen-protocol's canonical_input makes it: elapsed, the viewport's width and height, the events' count and walks, then the state's count and lanes.",
    )));
    out.extend(
        "func canonical_input (input: protocol.Input) =>\n  let $protocol.Input { elapsed, width, height, events, state } = input\n  let output: #[F32] = #[elapsed, width, height, U32.to_f32 events.length]\n  for event in events:\n    output := array.extend self (canonical_event event)\n  output := self.push (U32.to_f32 state.length)\n  output := array.extend self state\n  return output"
            .lines()
            .map(str::to_owned),
    );
    out.push(String::new());
    out.extend(render_stream(model));
    out.push(String::new());
    out.join("\n")
}
