//! The packets' canonical walks: their values in the schema's order, laid out
//! by none of the packets' tricks, so that pen's tests can compare what the
//! Rust and the Blot codecs make of the same values (pen's
//! `codec_round_trip`). Rust gets `canonical_input` and `canonical_render`
//! beside the codecs; the game's `pen/protocol_canonical.blot`, which only
//! tests import, gets the same input walk and the render walk's inverse.
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
use crate::blot::{chunk, comment};
use crate::model::*;
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

/// Whether an event's field crosses as a number (Blot's F32), not a whole one.
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
    let mut out = "    /// The walk of an input packet's values that pen's tests compare with the game's\n    /// (`pen/protocol_canonical.blot`): elapsed, the viewport's width and height, the events'\n    /// count, each event's tag and then its fields in the schema's order (a key as its code),\n    /// then the state's count and lanes. It lays nothing out as the packet does.\n    pub fn canonical_input(frame: &Frame, elapsed: f64, state: &[f32]) -> Vec<f32> {\n        let viewport = frame.viewport;\n        let mut out = vec![elapsed as f32, viewport.width as f32, viewport.height as f32, frame.events.len() as f32];\n        for event in &frame.events {\n            match event {".lines().map(str::to_owned).collect::<Vec<_>>();
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
        "    /// The walk of a render packet's values that pen's tests compare with the game's\n    /// (`pen/protocol_canonical.blot`): the header's values and then the sections, each in the\n    /// schema's order, as described in pen-codegen's canonical.rs. It takes the schema and\n    /// the state, which the frame does not hold, where the packet has them.\n    pub fn canonical_render(frame: &RenderFrame, schema: u32, state: &[f32]) -> Vec<f32> {\n        let mut out = Vec::new();"
            .lines()
            .map(str::to_owned),
    );
    let mut pushes = Pushes::new(8);
    for member in members(model) {
        match member {
            Member::Lane(laid) if kind(&laid.field) == "schema" => pushes.lane("schema as f32".to_owned()),
            Member::Lane(laid) => {
                rust_field(model, &mut pushes, &laid.field, &format!("frame.{}", laid.name), "&mut out")
            }
            Member::Group(group) => {
                let lanes = group_lanes(model, group);
                if presence(&lanes).is_none() {
                    for laid in &lanes {
                        rust_field(
                            model,
                            &mut pushes,
                            &laid.field,
                            &format!("frame.{group}.{}", laid.name),
                            "&mut out",
                        );
                    }
                    continue;
                }
                // A group with a presence lane is an Option, its presence lane the index
                // Rust's group holds.
                let mut some = Pushes::new(16);
                some.lane("1.0".to_owned());
                for laid in &lanes {
                    rust_field(model, &mut some, &laid.field, &format!("{group}.{}", laid.name), "&mut out");
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
                    _ => rust_list(&mut pushes, &section["record"], &format!("frame.{name}"), "&mut out"),
                }
            }
        }
    }
    out.extend(pushes.finish());
    out.extend(["        out".to_owned(), "    }".to_owned()]);

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
                "    fn {}(out: &mut Vec<f32>, {variable}: &{}) {{",
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

/// The Blot function reading a value of `field`'s kind back from the walk.
fn blot_reader(model: &Model, field: &Value) -> String {
    match kind(field) {
        "f32" => "f32_from".to_owned(),
        "count" | "reference" | "choice" | "present" | "schema" => "u32_from".to_owned(),
        "flag" => "bool_from".to_owned(),
        "vector" => format!("{}_from", snake(s(&field["type"]["name"]))),
        "matrix" => "mat4_from".to_owned(),
        "lanes" => "vec4_from".to_owned(),
        "optional" => "maybe_vec4_from".to_owned(),
        "tail" => "maybe_vec4s_from".to_owned(),
        "children" => format!("{}_list_from", snake(s(&section_record(model, s(&field["section"]))["name"]))),
        _ => unreachable!(),
    }
}

/// A Blot value of `field`'s kind that is all zeros, for a group that is not
/// there.
fn blot_zero(field: &Value) -> String {
    match kind(field) {
        "f32" => "0.0".to_owned(),
        "count" | "reference" | "choice" | "present" | "schema" => "0".to_owned(),
        "flag" => "false".to_owned(),
        "vector" => format!(
            "$types.{} {{ {} }}",
            s(&field["type"]["name"]),
            array(&field["type"]["components"]).iter().map(|c| format!("{}: 0.0", s(c))).collect::<Vec<_>>().join(", ")
        ),
        "matrix" => "$types.Mat4 { x: types.zero4, y: types.zero4, z: types.zero4, w: types.zero4 }".to_owned(),
        "lanes" => "types.zero4".to_owned(),
        "optional" => "$Nothing".to_owned(),
        _ => "#[]".to_owned(),
    }
}

/// `let (name, name_end) = reader canonical at` for each value, reading on
/// from `start`; returns the lines and where the last one ends.
fn blot_reads(reads: &[(String, String)], start: &str, indent: &str) -> (Vec<String>, String) {
    let mut lines = Vec::new();
    let mut at = start.to_owned();
    for (name, reader) in reads {
        lines.push(format!("{indent}let ({name}, {name}_end) = {reader} canonical {at}"));
        at = format!("{name}_end");
    }
    (lines, at)
}

/// `#types.Name { fields }`, in lines no longer than the others.
fn blot_record(name: &str, fields: &[String], indent: &str, end: &str) -> Vec<String> {
    let one = format!("{indent}return ($types.{name} {{ {} }}, {end})", fields.join(", "));
    if one.len() <= 100 {
        return vec![one];
    }
    let mut lines = vec![format!("{indent}let built = $types.{name} {{")];
    lines.extend(chunk(fields, 96 - indent.len()).iter().map(|line| format!("{indent}  {line},")));
    lines.push(format!("{indent}}}"));
    lines.push(format!("{indent}return (built, {end})"));
    lines
}

/// The readers of a record's values and of a list of them.
fn blot_record_readers(model: &Model, record: &Value, fields: &[Laid]) -> Vec<String> {
    let name = s(&record["name"]);
    let reader = snake(name);
    let mut reads: Vec<(String, String)> =
        fields.iter().map(|laid| (laid.name.clone(), blot_reader(model, &laid.field))).collect();
    let rest = &record["rest"];
    if rest.is_object() {
        let read = if kind(rest) == "repeat" {
            format!("{}_list_from", snake(s(&rest["record"]["name"])))
        } else {
            "floats_from".to_owned()
        };
        reads.push((s(&rest["name"]).to_owned(), read));
    }
    let mut out = vec![
        String::new(),
        format!("const {reader}_from = func (canonical: Array F32) => return func (at: U32) => return do:"),
    ];
    let (lines, end) = blot_reads(&reads, "at", "  ");
    out.extend(lines);
    let names: Vec<String> = reads.iter().map(|(name, _)| name.clone()).collect();
    out.extend(blot_record(name, &names, "  ", &end));
    out.extend([
        String::new(),
        format!("const {reader}_list_from = func (canonical: Array F32) => return func (at: U32) => return do:"),
        "  let count = lanes.integer canonical[at] lanes.max_integer".to_owned(),
        "  let items = #[]".to_owned(),
        "  let next = at + 1".to_owned(),
        "  for 0 .. count:".to_owned(),
        format!("    let (item, item_end) = {reader}_from canonical next"),
        "    items := array.push item self".to_owned(),
        "    next := item_end".to_owned(),
        "  return (items, next)".to_owned(),
    ]);
    out
}

const BLOT_READERS: &str = r#"// Each reader takes the walk and where a value starts in it, and returns the
// value and where the next one starts. Counts and flags must be whole.
const f32_from = func (canonical: Array F32) => return func (at: U32) => return (canonical[at], at + 1)

const u32_from = func (canonical: Array F32) => return func (at: U32) => return (lanes.integer canonical[at] lanes.max_integer, at + 1)

const bool_from = func (canonical: Array F32) => return func (at: U32) => return (lanes.integer canonical[at] 1 == 1, at + 1)

const vec4_at = func (canonical: Array F32) => return func (at: U32) => return
  $types.Vec4 { x: canonical[at], y: canonical[at + 1], z: canonical[at + 2], w: canonical[at + 3] }

const vec4_from = func (canonical: Array F32) => return func (at: U32) => return (vec4_at canonical at, at + 4)

// A matrix's columns, x first.
const mat4_from = func (canonical: Array F32) => return func (at: U32) => return do:
  let x = vec4_at canonical at
  let y = vec4_at canonical (at + 4)
  let z = vec4_at canonical (at + 8)
  let w = vec4_at canonical (at + 12)
  return ($types.Mat4 { x, y, z, w }, at + 16)

const maybe_vec4_from = func (canonical: Array F32) => return func (at: U32) => return do:
  if lanes.integer canonical[at] 1 == 1:
    return ($Some (vec4_at canonical (at + 1)), at + 5)
  return ($Nothing, at + 1)

const floats_from = func (canonical: Array F32) => return func (at: U32) => return do:
  let count = lanes.integer canonical[at] lanes.max_integer
  return (array.slice (at + 1) count canonical, at + 1 + count)

const maybe_floats_from = func (canonical: Array F32) => return func (at: U32) => return do:
  if lanes.integer canonical[at] 1 == 1:
    return floats_from canonical (at + 1)
  return (#[], at + 1)

// A tail: 0, or 1, its lanes' count and its lanes, four to a Vec4.
const maybe_vec4s_from = func (canonical: Array F32) => return func (at: U32) => return do:
  if lanes.integer canonical[at] 1 == 1:
    let count = lanes.integer canonical[at + 1] lanes.max_integer
    use lanes.require (count % 4 == 0)
    return (array.generate (count / 4) (func index => return vec4_at canonical (at + 2 + index * 4)), at + 2 + count)
  return (#[], at + 1)"#;

/// The game's `pen/protocol_canonical.blot`: the input walk, and the render
/// walk read back into the record the game encodes.
pub fn blot(model: &Model, top: &[String]) -> String {
    let mut out = top.to_vec();
    out.extend([
        "//".to_owned(),
        "// The packets' canonical walks (pen-codegen's canonical.rs), which pen's tests".to_owned(),
        "// compare with Rust's. Only tests import this; the game must not.".to_owned(),
        "import * as array from \"std/array\"".to_owned(),
        "import * as lanes from \"./lanes\"".to_owned(),
        "import * as protocol from \"./protocol\"".to_owned(),
        "import * as types from \"./types\"".to_owned(),
        String::new(),
    ]);
    out.extend(comment(Some("An event's walk: its tag, then its fields in the schema's order.")));
    out.push("const canonical_event = func (event: protocol.Event) => return case event of".to_owned());
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
        "const canonical_input = func (input: protocol.Input) => return do:\n  let $protocol.Input { elapsed, width, height, events, state } = input\n  let output: GrowArray F32 = #[elapsed, width, height, U32.to_f32 events.length]\n  for event in events:\n    array.append_into output (canonical_event event)\n  output.push (U32.to_f32 state.length)\n  array.append_into output state\n  return @storage.freeze output"
            .lines()
            .map(str::to_owned),
    );
    out.push(String::new());
    out.extend(BLOT_READERS.lines().map(str::to_owned));
    for vector in array(&model.protocol["vectors"]) {
        let components: Vec<_> = array(&vector["components"])
            .iter()
            .enumerate()
            .map(|(offset, component)| match offset {
                0 => format!("{}: canonical[at]", s(component)),
                _ => format!("{}: canonical[at + {offset}]", s(component)),
            })
            .collect();
        out.push(String::new());
        out.push(format!(
            "const {}_from = func (canonical: Array F32) => return func (at: U32) => return",
            snake(s(&vector["name"]))
        ));
        out.push(format!(
            "  ($types.{} {{ {} }}, at + {})",
            s(&vector["name"]),
            components.join(", "),
            components.len()
        ));
    }
    // Records, each after those it holds.
    let mut records: Vec<(&Value, Vec<Laid>)> = Vec::new();
    for entry in &model.sections {
        for layout in entry.fixed.iter().chain(entry.item.iter()) {
            records.push((&layout.record, layout.fields.clone()));
        }
        if kind(&entry.section) == "variable" {
            records.push((&entry.section["record"], entry.head.clone()));
        }
    }
    let mut emitted = HashSet::new();
    fn emit<'a>(
        model: &'a Model,
        records: &[(&'a Value, Vec<Laid>)],
        index: usize,
        emitted: &mut HashSet<&'a str>,
        out: &mut Vec<String>,
    ) {
        let (record, fields) = &records[index];
        if !emitted.insert(s(&record["name"])) {
            return;
        }
        let mut held: Vec<&str> = fields
            .iter()
            .filter(|laid| kind(&laid.field) == "children")
            .map(|laid| s(&section_record(model, s(&laid.field["section"]))["name"]))
            .collect();
        if record["rest"]["kind"] == "repeat" {
            held.push(s(&record["rest"]["record"]["name"]));
        }
        for name in held {
            if let Some(child) = records.iter().position(|(other, _)| s(&other["name"]) == name) {
                emit(model, records, child, emitted, out);
            }
        }
        out.extend(blot_record_readers(model, record, fields));
    }
    for index in 0..records.len() {
        emit(model, &records, index, &mut emitted, &mut out);
    }
    // Groups.
    for group in object(&model.protocol["render"]["groups"]).keys() {
        let lanes = group_lanes(model, group);
        let ty = s(&model.protocol["render"]["groups"][group]["type"]);
        let reads: Vec<(String, String)> =
            lanes.iter().map(|laid| (laid.name.clone(), blot_reader(model, &laid.field))).collect();
        out.push(String::new());
        out.push(format!(
            "const {}_from = func (canonical: Array F32) => return func (at: U32) => return do:",
            snake(ty)
        ));
        let Some(present) = presence(&lanes) else {
            let (lines, end) = blot_reads(&reads, "at", "  ");
            out.extend(lines);
            let names: Vec<String> = reads.iter().map(|(name, _)| name.clone()).collect();
            out.extend(blot_record(ty, &names, "  ", &end));
            continue;
        };
        // 0 when the group is not there, else 1 and its values; its presence lane
        // holds the index plus one, 0 meaning none.
        out.push("  if lanes.integer canonical[at] 1 == 1:".to_owned());
        let (lines, end) = blot_reads(&reads, "(at + 1)", "    ");
        out.extend(lines);
        let mut names: Vec<String> = reads.iter().map(|(name, _)| name.clone()).collect();
        names[present] = format!("{}: {} + 1", names[present], names[present]);
        out.extend(blot_record(ty, &names, "    ", &end));
        let zeros: Vec<String> =
            lanes.iter().map(|laid| format!("{}: {}", laid.name, blot_zero(&laid.field))).collect();
        out.extend(blot_record(ty, &zeros, "  ", "at + 1"));
    }
    // The frame.
    let mut reads = Vec::new();
    for member in members(model) {
        match member {
            Member::Lane(laid) => reads.push((laid.name.clone(), blot_reader(model, &laid.field))),
            Member::Group(group) => {
                let ty = s(&model.protocol["render"]["groups"][group]["type"]);
                reads.push((group.to_owned(), format!("{}_from", snake(ty))));
            }
            Member::Section(entry) => {
                let section = &entry.section;
                let reader = match kind(section) {
                    "lanes" if section["nullable"] == true => "maybe_floats_from".to_owned(),
                    "lanes" => "floats_from".to_owned(),
                    _ => format!("{}_list_from", snake(s(&section["record"]["name"]))),
                };
                reads.push((s(&section["name"]).to_owned(), reader));
            }
        }
    }
    out.push(String::new());
    out.extend(comment(Some(
        "The record the game encodes, from the render walk pen-protocol's canonical_render makes of a frame, its schema and its state. Anything else traps.",
    )));
    out.push("const render_from_canonical = func (canonical: Array F32) => return do:".to_owned());
    let (lines, end) = blot_reads(&reads, "0", "  ");
    out.extend(lines);
    out.push(format!("  use lanes.require ({end} == canonical.length)"));
    out.push("  return $types.RenderFrame {".to_owned());
    let names: Vec<String> = reads.iter().map(|(name, _)| name.clone()).collect();
    out.extend(chunk(&names, 76).iter().map(|line| format!("    {line},")));
    out.push("  }".to_owned());
    out.push(String::new());
    out.join("\n")
}
