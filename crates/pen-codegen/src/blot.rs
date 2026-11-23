use crate::model::*;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

pub(crate) fn comment(text: Option<&str>) -> Vec<String> {
    text.map(|text| wrap(text, 78).iter().map(|line| format!("// {line}")).collect()).unwrap_or_default()
}
fn count(record: &Value) -> String {
    floats(record).to_lowercase()
}
fn bound_blot(value: &Value) -> String {
    if value.is_object() {
        return format!("protocol.{}", s(&value["limit"]).to_lowercase());
    }
    value.to_string()
}
pub(crate) fn chunk(lanes: &[String], width: usize) -> Vec<String> {
    let mut result = Vec::new();
    let mut line = String::new();
    for lane in lanes {
        if !line.is_empty() && line.len() + 2 + lane.len() > width {
            result.push(line);
            line = String::new();
        }
        if !line.is_empty() {
            line.push_str(", ");
        }
        line.push_str(lane);
    }
    if !line.is_empty() {
        result.push(line);
    }
    result
}
fn encode(field: &Value, path: &str, local: &str) -> (Vec<String>, Vec<String>) {
    let xyzw = ["x", "y", "z", "w"];
    match kind(field) {
        "f32" => (vec![], vec![path.to_owned()]),
        "vector" => (
            vec![],
            array(&field["type"]["components"]).iter().map(|component| format!("{path}.{}", s(component))).collect(),
        ),
        "matrix" => {
            let columns: Vec<_> = xyzw.iter().map(|column| format!("{local}_{column}")).collect();
            let bindings = xyzw
                .iter()
                .zip(&columns)
                .map(|(column, local)| format!("{column}: {local}"))
                .collect::<Vec<_>>()
                .join(", ");
            (
                vec![format!("let $types.Mat4 {{ {bindings} }} = {path}")],
                columns.iter().flat_map(|column| xyzw.iter().map(move |row| format!("{column}.{row}"))).collect(),
            )
        }
        "lanes" => (vec![], xyzw.iter().map(|component| format!("{path}.{component}")).collect()),
        "count" | "reference" | "choice" | "present" => (vec![], vec![format!("U32.to_f32 ({path})")]),
        "flag" => (vec![], vec![format!("lanes.bit {path}")]),
        "optional" => {
            let mut lanes = vec![format!("{local}_count")];
            lanes.extend(xyzw.iter().map(|component| format!("{local}.{component}")));
            (
                vec![
                    format!("let ({local}_count, {local}) = case {path} of"),
                    "  $Some value => (4.0, value)".to_owned(),
                    "  $Nothing => (0.0, types.zero4)".to_owned(),
                ],
                lanes,
            )
        }
        "children" => (vec![], vec![format!("U32.to_f32 ({path}.length)")]),
        "tail" => (vec![], vec![format!("U32.to_f32 (min {path}.length {})", bound_blot(&field["max"]))]),
        _ => unreachable!(),
    }
}
fn field_path(variable: &str, laid: &Laid) -> String {
    format!("{variable}.{}", laid.name)
}
fn record(layout: &Layout) -> Vec<String> {
    let name = s(&layout.record["name"]);
    let variable = singular(&snake(name));
    let about = match layout.record["doc"].as_str() {
        Some(doc) => format!("{name}: {doc}"),
        None => format!("A {name}'s lanes."),
    };
    let mut out = comment(Some(&about));
    out.push(format!("func {} ({variable}: types.{name}) =>", encoder(&layout.record)));
    let mut lets = Vec::new();
    let mut lanes = Vec::new();
    for laid in &layout.fields {
        let (local, values) = encode(&laid.field, &field_path(&variable, laid), &laid.name);
        lets.extend(local);
        lanes.extend(values);
    }
    out.extend(lets.iter().map(|line| format!("  {line}")));
    out.push("  return #[".to_owned());
    out.extend(chunk(&lanes, 76).iter().map(|line| format!("    {line},")));
    out.push("  ]".to_owned());
    out
}
fn constants(model: &Model) -> Vec<String> {
    let mut out = vec!["// Limits.".to_owned()];
    for limit in array(&model.protocol["limits"]) {
        out.extend(comment(limit["doc"].as_str()));
        out.push(format!("const {} = {}", s(&limit["limit"]).to_lowercase(), limit["value"]));
    }
    for enumeration in array(&model.protocol["enumerations"]) {
        out.push(String::new());
        out.extend(comment(enumeration["doc"].as_str()));
        for (code, value) in array(&enumeration["values"]).iter().enumerate() {
            out.push(format!("const {}_{} = {code}", s(&enumeration["constant"]).to_lowercase(), s(value)));
        }
    }
    out.extend([
        String::new(),
        "// Sizes in lanes.".to_owned(),
        format!("const packet_magic = {}.0", model.magic),
        format!("const packet_header_floats = {}", model.header_lanes),
        "const frame_header_floats = 5".to_owned(),
        format!("const event_floats = {}", model.event_lanes()),
    ]);
    if let Some(extensions) = model.extensions() {
        out.push(format!("const extensions_marker = {}.0", n(&extensions["marker"])));
    }
    let mut seen = HashSet::new();
    for entry in &model.sections {
        let mut values = Vec::new();
        if let Some(layout) = &entry.fixed {
            values.push((count(&layout.record), layout.lanes));
        }
        if kind(&entry.section) == "variable" {
            values.push((count(&entry.section["record"]), entry.head_lanes));
        }
        if let Some(layout) = &entry.item {
            values.push((count(&layout.record), layout.lanes));
        }
        if kind(&entry.section) == "lanes"
            && let Some(constant) = entry.section["constant"].as_str()
        {
            values.push((format!("{}_floats", constant.to_lowercase()), n(&entry.section["max"])));
        }
        for (name, value) in values {
            if seen.insert(name.clone()) {
                out.push(format!("const {name} = {value}"));
            }
        }
    }
    out.extend([String::new(), "// Where each header lane and record field starts.".to_owned()]);
    for entry in &model.header {
        out.push(format!("const header_{}_lane = {}", entry.key.replace('.', "_"), entry.laid.offset));
    }
    for layout in &model.records {
        for field in &layout.fields {
            out.push(format!("const {}_{}_lane = {}", snake(s(&layout.record["name"])), field.name, field.offset));
        }
    }
    for entry in &model.sections {
        if kind(&entry.section) != "variable" {
            continue;
        }
        let name = snake(s(&entry.section["record"]["name"]));
        for field in &entry.head {
            out.push(format!("const {name}_{}_lane = {}", field.name, field.offset));
        }
        let rest = &entry.section["record"]["rest"];
        if kind(rest) == "repeat" {
            out.push(format!("const {name}_{}_lane = {}", s(&rest["name"]), entry.head_lanes - 1));
        }
    }
    out.extend([String::new(), "// Each event's tag.".to_owned()]);
    for event in model.events() {
        out.push(format!("const {}_tag = {}", snake(s(&event["name"])), event["tag"]));
    }
    out.push(String::new());
    out.extend(comment(Some(
        "The codes KeyDown and KeyUp carry for named keys; a key that types a character carries its code point, \
         a letter its capital's.",
    )));
    for (key, code) in object(&model.protocol["input"]["keys"]) {
        out.push(format!("const key_{} = {}", snake(key), code));
    }
    out
}
fn input(model: &Model) -> Vec<String> {
    let input = &model.protocol["input"];
    let input_lanes = array(&input["lanes"]);
    let mut out = comment(Some("The events the host sends, as their lanes hold them."));
    out.push("type Event is data =".to_owned());
    let wire_type = |field: &Value| -> &str {
        if field["wire"] == "f32" {
            return "F32";
        }
        "U32"
    };
    for event in model.core_events() {
        let name = s(&event["name"]);
        let fields = event_fields(event);
        if !event["value"].is_null() {
            out.push(format!("  | ${name} {}", wire_type(&event["value"])));
        } else if !fields.is_empty() {
            out.push(format!(
                "  | ${name} {{ {} }}",
                fields
                    .iter()
                    .map(|(name, field)| format!("{name}: {}", wire_type(field)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        } else {
            out.push(format!("  | ${name}"));
        }
    }
    out.extend([
        String::new(),
        "type Input is data = $Input { elapsed: F32, width: F32, height: F32, events: &[Event], state: &[F32] }"
            .to_owned(),
        String::new(),
    ]);
    out.extend(comment(Some(&format!("An event's lanes from `start`: its tag, then {}. The {} lane holds a whole number; lanes an event does not use are zero.", input_lanes.iter().map(s).collect::<Vec<_>>().join(", "), s(&input_lanes[0])))));
    out.push("func decode_event values => func (start: U32) =>".to_owned());
    out.push(format!(
        "  let tag = lanes.integer (values.at (start)) {}",
        model.core_events().iter().map(|event| n(&event["tag"])).max().unwrap_or(0)
    ));
    for (index, lane) in input_lanes.iter().enumerate() {
        if index == 0 {
            out.push(format!("  let {} = lanes.integer (values.at (start + 1)) lanes.max_integer", s(lane)));
        } else {
            out.push(format!("  let {} = (values.at (start + {}))", s(lane), index + 1));
        }
    }
    out.push("  case tag of".to_owned());
    for event in model.core_events() {
        let fields = event_fields(event);
        let name = s(&event["name"]);
        out.push(format!("    {} =>", event["tag"]));
        for (index, lane) in input_lanes.iter().enumerate() {
            if let Some((_, field)) = fields.iter().find(|(_, field)| &field["lane"] == lane) {
                if !field["max"].is_null() && field["wire"] != "f32" {
                    out.push(format!("      use lanes.require ({} <= {})", s(lane), field["max"]));
                }
            } else {
                let mut zero = "0.0";
                if index == 0 {
                    zero = "0";
                }
                out.push(format!("      use lanes.require ({} == {zero})", s(lane)));
            }
        }
        let value = |field: &Value| -> String {
            if field["wire"] != "f32" && field["lane"] != input_lanes[0] {
                return format!("lanes.integer {} lanes.max_integer", s(&field["lane"]));
            }
            s(&field["lane"]).to_owned()
        };
        if !event["value"].is_null() {
            out.push(format!("      return ${name} {}", value(&event["value"])));
        } else if !fields.is_empty() {
            let fields = fields
                .iter()
                .map(|(name, field)| {
                    let lane = value(field);
                    if lane == *name {
                        return lane;
                    }
                    format!("{name}: {lane}")
                })
                .collect::<Vec<_>>()
                .join(", ");
            out.push(format!("      return ${name} {{ {fields} }}"));
        } else {
            out.push(format!("      return ${name}"));
        }
    }
    out.push(format!("    _ => panic {}", quote(&input["messages"]["tag"])));
    out.push(String::new());
    out.extend(comment(Some("The input packet: the seconds since the last frame, the viewport's width and height, the event and state counts, the events and the state. Anything else traps.")));
    out.extend("func decode_input_into values -> (events: #[~Event]) =>\n  use lanes.require (values.length >= frame_header_floats)\n  use lanes.finite values\n  let elapsed = (values.at (0))\n  use lanes.require (elapsed >= 0.0)\n  let width = (values.at (1))\n  let height = (values.at (2))\n  use lanes.require (width > 0.0)\n  use lanes.require (height > 0.0)".lines().map(str::to_owned));
    out.push(format!(
        "  let event_count = lanes.integer (values.at (3)) {}",
        s(&input["maxEvents"]["limit"]).to_lowercase()
    ));
    out.push(format!(
        "  let state_count = lanes.integer (values.at (4)) {}",
        s(&model.protocol["state"]["max"]["limit"]).to_lowercase()
    ));
    out.extend("  let state_start = frame_header_floats + event_count * event_floats\n  use lanes.require (values.length == state_start + state_count)\n  @storage.truncate events 0\n  for index in 0 .. event_count:\n    @storage.push events (decode_event values (frame_header_floats + index * event_floats))\n  return $Input {\n    state: values.view state_start (state_start + state_count),\n    elapsed, width, height,\n    events: &events[0..events.length],\n  }".lines().map(str::to_owned));
    out
}
// Emit schema offsets directly into the named packet destination.
fn direct_fields(out: &mut Vec<String>, fields: &[Laid], variable: &str, indent: &str) {
    for field in fields {
        let (locals, values) = encode(&field.field, &field_path(variable, field), &field.name);
        out.extend(locals.iter().map(|line| format!("{indent}{line}")));
        for (lane, value) in values.iter().enumerate() {
            let offset = field.offset + lane;
            out.push(format!("{indent}encoded[at + {offset}] := {value}"));
        }
    }
}

// Emit lanes in schema order. Only section lengths that depend on the rows
// need backpatching; all other header values are known before the first row.
fn append_fields(out: &mut Vec<String>, fields: &[Laid], row: &str, indent: &str) {
    for field in fields {
        let (locals, values) = encode(&field.field, &field_path(row, field), &field.name);
        out.extend(locals.into_iter().map(|line| format!("{indent}{line}")));
        out.extend(values.into_iter().map(|value| format!("{indent}@storage.push encoded ({value})")));
    }
}

fn render_views(model: &Model) -> Vec<String> {
    let mut out = vec![String::new(),
        "func encode_render_views (packet: types.RenderHeader) -> collections -> project -> context -> (encoded: own #[~F32]) =>".into(),
        "  @storage.truncate encoded 0".into(),
        "  return append_render_views packet collections project context encoded".into(),
        "func append_render_views (packet: types.RenderHeader) -> collections -> project -> context -> (encoded: own #[~F32]) =>".into(),
        "  let start = encoded.length".into()];
    for section in &model.sections {
        if kind(&section.section) != "trailing" && section.owner.is_none() {
            let name = s(&section.section["name"]);
            out.push(format!("  let {name}_count: U32 = collections.{name}.length"));
        }
    }
    let mut patches = HashMap::new();
    for entry in &model.header {
        let field = &entry.laid.field;
        let (locals, values) = match kind(field) {
            "magic" => (vec![], vec!["protocol.packet_magic".into()]),
            "schema" => (vec![], vec!["U32.to_f32 packet.schema".into()]),
            "size" => {
                let name = s(&field["section"]);
                let dynamic = field["of"] == "lanes" || model.section(name).owner.is_some();
                let value = if dynamic {
                    patches.insert(name, (entry.laid.offset, s(&field["of"])));
                    "0.0".into()
                } else {
                    format!("U32.to_f32 {name}_count")
                };
                (vec![], vec![value])
            }
            _ => {
                let prefix = entry.group.as_ref().map_or(String::new(), |group| format!("{group}."));
                encode(field, &format!("packet.{prefix}{}", entry.laid.name), &entry.key.replace('.', "_"))
            }
        };
        out.extend(locals.into_iter().map(|line| format!("  {line}")));
        out.extend(values.into_iter().map(|value| format!("  @storage.push encoded ({value})")));
    }
    for section in &model.sections {
        let name = s(&section.section["name"]);
        if patches.contains_key(name) {
            out.push(format!("  let {name}_start = encoded.length"));
        }
        if kind(&section.section) == "trailing" {
            let (_, owners) = model.tails.iter().find(|(tail, _)| tail == name).unwrap();
            for owner in owners {
                let section = model.section(&owner.section);
                let indent = view_rows(&mut out, section);
                out.push(format!("{indent}project.append_extra context source encoded"));
            }
            continue;
        }
        let indent = view_rows(&mut out, section);
        if kind(&section.section) == "lanes" {
            out.push(format!("{indent}@storage.push encoded source"));
        } else if name == "draws" || name == "inset_draws" {
            out.push(format!("{indent}let (mesh, material) = project.draw_metadata context source"));
            let matrix = (0..16).map(|i| format!("m{i}")).collect::<Vec<_>>();
            out.push(format!("{indent}let ({}) = project.draw_transform context source", matrix.join(", ")));
            out.push(format!("{indent}let (present, p0, p1, p2, p3) = project.draw_parameters context source"));
            for field in &section.fixed.as_ref().unwrap().fields {
                let values = match kind(&field.field) {
                    "reference" => vec![format!("U32.to_f32 {}", field.name)],
                    "matrix" => matrix.clone(),
                    "optional" => ["present", "p0", "p1", "p2", "p3"].map(str::to_owned).to_vec(),
                    "tail" => vec!["U32.to_f32 (project.extra_count context source)".into()],
                    _ => unreachable!(),
                };
                out.extend(values.into_iter().map(|value| format!("{indent}@storage.push encoded ({value})")));
            }
        } else {
            let row = singular(name);
            let nominal = section
                .fixed
                .as_ref()
                .filter(|layout| layout.fields.iter().all(|f| !matches!(kind(&f.field), "children" | "tail")));
            let annotation = nominal.map_or(String::new(), |l| format!(": types.{}", s(&l.record["name"])));
            out.push(format!("{indent}let {row}{annotation} = project.{row} context source"));
            if let Some(layout) = &section.fixed {
                append_fields(&mut out, &layout.fields, &row, indent);
            } else {
                append_fields(&mut out, &section.head, &row, indent);
                let rest = s(&section.section["record"]["rest"]["name"]);
                out.push(format!("{indent}let count: U32 = {row}.{rest}.length"));
                if section.item.is_some() {
                    out.push(format!("{indent}@storage.push encoded (U32.to_f32 count)"));
                }
                out.push(format!("{indent}for i in 0 .. count:"));
                out.push(format!("{indent}  let item = {row}.{rest}.at i"));
                if let Some(layout) = &section.item {
                    append_fields(&mut out, &layout.fields, "item", &format!("{indent}  "));
                } else {
                    out.push(format!("{indent}  @storage.push encoded item"));
                }
            }
        }
        if let Some((offset, unit)) = patches.get(name) {
            let mut size = format!("(encoded.length - {name}_start)");
            if *unit == "count" {
                size.push_str(&format!(" / {}", section.fixed.as_ref().unwrap().lanes));
            }
            out.push(format!("  encoded[start + {offset}] := U32.to_f32 ({size})"));
        }
    }
    out.push("  return encoded".into());
    out.push(String::new());
    out.push("func measure_render_floats collections -> project -> context =>".into());
    out.push("  let total: U32 = protocol.packet_header_floats".into());
    for section in &model.sections {
        if kind(&section.section) != "trailing" && section.owner.is_none() {
            let name = s(&section.section["name"]);
            out.push(format!("  let {name}_count: U32 = collections.{name}.length"));
        }
    }
    for section in &model.sections {
        let name = s(&section.section["name"]);
        if kind(&section.section) == "trailing" {
            let (_, owners) = model.tails.iter().find(|(tail, _)| tail == name).unwrap();
            for owner in owners {
                let owner_section = model.section(&owner.section);
                let indent = view_rows(&mut out, owner_section);
                out.push(format!("{indent}total := self + (project.extra_count context source) * 4"));
            }
            continue;
        }
        if kind(&section.section) == "lanes" {
            out.push(format!("  total := self + {name}_count"));
        } else if let Some(layout) = &section.fixed {
            if let Some((owners, children)) = &section.owner {
                out.push(format!("  for parent in 0 .. {owners}_count:"));
                out.push(format!(
                    "    let children = project.{}_{} context collections.{owners}[parent]",
                    singular(owners),
                    children
                ));
                out.push(format!("    total := self + children.length * {}", layout.lanes));
            } else {
                out.push(format!("  total := self + {name}_count * {}", layout.lanes));
            }
        } else {
            let indent = view_rows(&mut out, section);
            let row = singular(name);
            let rest = s(&section.section["record"]["rest"]["name"]);
            out.push(format!("{indent}let {row} = project.{row} context source"));
            out.push(format!("{indent}let count: U32 = {row}.{rest}.length"));
            if let Some(item) = &section.item {
                out.push(format!("{indent}total := self + {} + count * {}", section.head_lanes, item.lanes));
            } else {
                out.push(format!("{indent}total := self + {} + count", section.head_lanes));
            }
        }
    }
    out.push("  return total".into());
    out
}

fn view_rows(out: &mut Vec<String>, section: &Section) -> &'static str {
    if let Some((owners, children)) = &section.owner {
        out.push(format!("  for parent in 0 .. {owners}_count:"));
        out.push(format!(
            "    let children = project.{}_{} context collections.{owners}[parent]",
            singular(owners),
            children
        ));
        out.push("    let count: U32 = children.length".into());
        out.push("    for i in 0 .. count:".into());
        out.push("      let source = children.at i".into());
        "      "
    } else {
        let name = s(&section.section["name"]);
        out.push(format!("  for i in 0 .. {name}_count:"));
        out.push(format!("    let source = collections.{name}[i]"));
        "    "
    }
}

// Small games keep flat mesh and sound pools without owning nested wire records.
// This encoder shares the schema's header and record layouts with the full view path.
fn render_meshes(model: &Model) -> Vec<String> {
    let draws = model.section("draws").fixed.as_ref().unwrap();
    let sounds = model.section("sounds").fixed.as_ref().unwrap();
    let mut out = vec![String::new(),
        "// Flat mesh draws (no parameters/extras) and sounds, into the packet it is given and returns.".to_owned(),
        "type MeshDraw = { mesh: U32, material: U32, transform: types.Mat4 }".to_owned(),
        "func encode_mesh_frame (packet: types.RenderHeader) -> (draws: &[MeshDraw]) -> (sounds: &[types.Sound]) -> (encoded: own #[~F32]) =>".to_owned(),
        "  @storage.truncate encoded 0".to_owned(),
        format!("  for 0 .. (protocol.packet_header_floats + draws.length * {} + sounds.length * {}):", draws.lanes, sounds.lanes),
        "    @storage.push encoded 0.0".to_owned()];
    for entry in &model.header {
        let field = &entry.laid;
        let values = match kind(&field.field) {
            "magic" => vec!["protocol.packet_magic".to_owned()],
            "schema" => vec!["U32.to_f32 packet.schema".to_owned()],
            "size" => vec![match (s(&field.field["section"]), s(&field.field["of"])) {
                ("draws", "count") => "U32.to_f32 draws.length".to_owned(),
                ("sounds", "count") => "U32.to_f32 sounds.length".to_owned(),
                _ => "0.0".to_owned(),
            }],
            _ => {
                let prefix = entry.group.as_ref().map_or(String::new(), |group| format!("{group}."));
                let (locals, values) =
                    encode(&field.field, &format!("packet.{prefix}{}", field.name), &entry.key.replace('.', "_"));
                out.extend(locals.iter().map(|line| format!("  {line}")));
                values
            }
        };
        for (index, value) in values.iter().enumerate() {
            out.push(format!("  encoded[{}] := {value}", field.offset + index));
        }
    }
    out.push("  let at: U32 = protocol.packet_header_floats".to_owned());
    out.push("  for draw in draws:".to_owned());
    let mesh_fields: Vec<_> =
        draws.fields.iter().filter(|field| !matches!(kind(&field.field), "optional" | "tail")).cloned().collect();
    direct_fields(&mut out, &mesh_fields, "draw", "    ");
    out.push(format!("    at := self + {}", draws.lanes));
    out.push("  for sound in sounds:".to_owned());
    direct_fields(&mut out, &sounds.fields, "sound", "    ");
    out.push(format!("    at := self + {}", sounds.lanes));
    out.push("  return encoded".to_owned());
    out
}

fn render(model: &Model) -> Vec<String> {
    let mut out = Vec::new();
    let mut emitted = HashSet::new();
    for entry in &model.sections {
        if let Some(layout) = &entry.fixed
            && emitted.insert(s(&layout.record["name"]))
        {
            out.push(String::new());
            out.extend(record(layout));
        }
        if kind(&entry.section) != "variable" {
            continue;
        }
        let record = &entry.section["record"];
        let rest = &record["rest"];
        let name = singular(&snake(s(&record["name"])));
        let rest_name = s(&rest["name"]);
        out.push(String::new());
        if kind(rest) == "values" {
            let mut lets = Vec::new();
            let mut lanes = Vec::new();
            for laid in &entry.head {
                let (local, values) = encode(&laid.field, &field_path(&name, laid), &laid.name);
                lets.extend(local);
                lanes.extend(values);
            }
            out.extend(comment(Some(&format!(
                "{}: its {}, then its {rest_name}.",
                s(&record["name"]),
                entry.head.iter().map(|field| field.name.as_str()).collect::<Vec<_>>().join(", ")
            ))));
            out.push(format!("func {} ({name}: types.{}) =>", encoder(record), s(&record["name"])));
            out.extend(lets.iter().map(|line| format!("  {line}")));
            out.push("  return array.concat #[".to_owned());
            out.extend(chunk(&lanes, 76).iter().map(|line| format!("    {line},")));
            out.push(format!("  ] {name}.{rest_name}"));
        } else {
            let item = entry.item.as_ref().unwrap();
            let item_name = singular(rest_name);
            let mut lines = Vec::new();
            let mut lane = 0;
            for laid in &entry.head {
                let (local, values) = encode(&laid.field, &field_path(&name, laid), &laid.name);
                lines.extend(local);
                for value in values {
                    lines.push(format!("encoded[{lane}] := {value}"));
                    lane += 1;
                }
            }
            lines.push(format!("encoded[{lane}] := U32.to_f32 {rest_name}.length"));
            let mut writes = Vec::new();
            let mut offset = 0;
            for laid in &item.fields {
                let (local, values) = encode(&laid.field, &field_path(&item_name, laid), &laid.name);
                writes.extend(local);
                for value in values {
                    let mut index = "start".to_owned();
                    if offset != 0 {
                        index = format!("start + {offset}");
                    }
                    writes.push(format!("encoded[{index}] := {value}"));
                    offset += 1;
                }
            }
            let mut head = String::new();
            if !entry.head.is_empty() {
                head = format!(
                    "its {}, ",
                    entry.head.iter().map(|field| field.name.as_str()).collect::<Vec<_>>().join(", ")
                );
            }
            out.extend(comment(Some(&format!(
                "{}: {head}its {rest_name}' count, then each of them.",
                s(&record["name"])
            ))));
            out.push(format!("func {} ({name}: types.{}) =>", encoder(record), s(&record["name"])));
            out.push(format!("  let {rest_name} = {name}.{rest_name}"));
            out.push(format!(
                "  let encoded = @array.fill (protocol.{} + {rest_name}.length * protocol.{}) 0.0",
                count(record),
                count(&item.record)
            ));
            out.extend(lines.iter().map(|line| format!("  {line}")));
            out.push(format!("  for index in 0 .. {rest_name}.length:"));
            out.push(format!("    let {item_name} = {rest_name}[index]"));
            out.push(format!("    let start = protocol.{} + index * protocol.{}", count(record), count(&item.record)));
            out.extend(writes.iter().map(|line| format!("    {line}")));
            out.push("  return encoded".to_owned());
        }
    }
    let mut tail_encoders = HashMap::new();
    for (_, owners) in &model.tails {
        for owner in owners {
            let key = format!("{}.{}", s(&owner.layout.record["name"]), owner.field.name);
            if tail_encoders.contains_key(&key) {
                continue;
            }
            let name = format!("encode_{}_{}", snake(s(&owner.layout.record["name"])), owner.field.name);
            tail_encoders.insert(key, name.clone());
            let variable = singular(&snake(s(&owner.layout.record["name"])));
            let path = field_path(&variable, &owner.field);
            let max = bound_blot(&owner.field.field["max"]);
            out.push(String::new());
            out.extend(comment(Some(&format!(
                "A {}'s {}, at most {max} of them, for the packet's last section.",
                s(&owner.layout.record["name"]),
                owner.field.name
            ))));
            out.push(format!("func {name} ({variable}: types.{}) =>", s(&owner.layout.record["name"])));
            out.push(format!("  let kept = min {path}.length {max}"));
            out.push(format!(
                "  return #[lane | index <- array.generate kept identity, lane <- vec4_lanes {path}[index]]"
            ));
        }
    }
    if !tail_encoders.is_empty() {
        out.extend([String::new(), "func vec4_lanes (v: types.Vec4) => #[v.x, v.y, v.z, v.w]".to_owned()]);
    }
    out.extend(render_views(model));
    out.extend(render_meshes(model));

    out
}
fn field_type(model: &Model, field: &Value) -> String {
    match kind(field) {
        "f32" => "F32".to_owned(),
        "count" | "reference" | "choice" | "present" | "schema" => "U32".to_owned(),
        "flag" => "Bool".to_owned(),
        "vector" => s(&field["type"]["name"]).to_owned(),
        "matrix" => "Mat4".to_owned(),
        "lanes" => "Vec4".to_owned(),
        "optional" => "Maybe Vec4".to_owned(),
        "tail" => "&[Vec4]".to_owned(),
        "children" => format!("&[{}]", s(&model.section(s(&field["section"])).section["record"]["name"])),
        _ => unreachable!(),
    }
}

fn structure(out: &mut Vec<String>, name: &str, fields: Vec<String>) {
    out.push(String::new());
    out.push(format!("type {name} is data = ${name} {{"));
    out.extend(fields.into_iter().map(|field| format!("  {field},")));
    out.push("}".to_owned());
}

fn member(model: &Model, laid: &Laid) -> String {
    format!("{}: {}", laid.name, field_type(model, &laid.field))
}

fn types(model: &Model) -> Vec<String> {
    let mut out = vec![
        "// Wire records. Mat4 stores four columns; vectors follow schema lane order.".to_owned(),
        "type Vec4 is data = $Vec4 { x: F32, y: F32, z: F32, w: F32 }".to_owned(),
        "type Mat4 is data = $Mat4 { x: Vec4, y: Vec4, z: Vec4, w: Vec4 }".to_owned(),
        "const zero4 = $Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }".to_owned(),
    ];
    for vector in array(&model.protocol["vectors"]) {
        structure(
            &mut out,
            s(&vector["name"]),
            array(&vector["components"]).iter().map(|component| format!("{}: F32", s(component))).collect(),
        );
    }
    for layout in &model.records {
        structure(&mut out, s(&layout.record["name"]), layout.fields.iter().map(|laid| member(model, laid)).collect());
    }
    for entry in &model.sections {
        if kind(&entry.section) != "variable" {
            continue;
        }
        let record = &entry.section["record"];
        let rest = &record["rest"];
        let mut fields: Vec<_> = entry.head.iter().map(|laid| member(model, laid)).collect();
        let mut ty = "F32";
        if kind(rest) == "repeat" {
            ty = s(&rest["record"]["name"]);
        }
        fields.push(format!("{}: &[{ty}]", s(&rest["name"])));
        structure(&mut out, s(&record["name"]), fields);
    }
    for (group, value) in object(&model.protocol["render"]["groups"]) {
        structure(
            &mut out,
            s(&value["type"]),
            model
                .header
                .iter()
                .filter(|entry| entry.group.as_ref() == Some(group))
                .map(|entry| member(model, &entry.laid))
                .collect(),
        );
    }
    let mut fields = Vec::new();
    let mut groups = HashSet::new();
    for entry in &model.header {
        if matches!(kind(&entry.laid.field), "magic" | "size") {
            continue;
        }
        if let Some(group) = &entry.group {
            if groups.insert(group) {
                fields.push(format!("{group}: {}", s(&model.protocol["render"]["groups"][group]["type"])));
            }
        } else {
            fields.push(member(model, &entry.laid));
        }
    }
    // Structural, fixed header shared by the borrowed source encoder and
    // adapters. No section collections or owners enter this alias.
    out.push(String::new());
    out.push("type RenderHeader = {".to_owned());
    out.extend(fields.iter().map(|field| format!("  {field},")));
    out.push("}".to_owned());

    out
}

const NUMERIC: &str = r#"import * as array from "std/array"

const max_integer = 16777215

func require (condition: Bool) =>
  if condition:
    return ()
  return panic "invalid Pen numeric packet"

func finite values => require (array.all is_finite values)

func integer (value: F32) => func (maximum: U32) =>
  let number = F32.to_u32 value
  use require (value == U32.to_f32 number)
  use require (number <= maximum)
  return number

func bit (value: Bool) =>
  if value:
    return 1.0
  return 0.0"#;

pub fn emit(model: &Model, header: &str) -> Vec<(&'static str, String)> {
    let top: Vec<_> = header.lines().map(|line| format!("// {line}")).collect();
    let mut protocol = top.clone();
    protocol.extend(
        r#"//
// Packet limits, offsets and input decoding.
import * as array from "std/array"
import * as lanes from "./lanes"
"#
        .split('\n')
        .map(str::to_owned),
    );
    protocol.extend(constants(model));
    protocol.push(String::new());
    protocol.extend(input(model));
    protocol.push(String::new());
    let mut render_source = top.clone();
    render_source.extend(
        r#"//
// Encode Pen wire records without application dependencies.
import * as array from "std/array"
import * as lanes from "./lanes"
import * as protocol from "./protocol"
import * as types from "./types""#
            .lines()
            .map(str::to_owned),
    );
    render_source.extend(render(model));
    render_source.push(String::new());
    let mut wire_types = top.clone();
    wire_types.extend(types(model));
    wire_types.push(String::new());
    let canonical = crate::canonical::blot(model, &top);
    let bridge = crate::bridge::emit(model, &top);
    let program = crate::program::program(&top);
    let frame = crate::program::frame(model, &top);
    let check = crate::program::check(&top);
    let testing = crate::program::testing(&top);
    let mut numeric = top;
    numeric.extend(NUMERIC.lines().map(str::to_owned));
    numeric.push(String::new());
    vec![
        ("protocol.blot", protocol.join("\n")),
        ("protocol_render.blot", render_source.join("\n")),
        ("types.blot", wire_types.join("\n")),
        ("lanes.blot", numeric.join("\n")),
        ("protocol_canonical.blot", canonical),
        ("protocol_views_test.blot", bridge),
        ("program.blot", program),
        ("frame.blot", frame),
        ("check.blot", check),
        ("test.blot", testing),
    ]
}
