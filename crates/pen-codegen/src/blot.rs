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
                vec![format!("let #types.Mat4 {{ {bindings} }} = {path}")],
                columns.iter().flat_map(|column| xyzw.iter().map(move |row| format!("{column}.{row}"))).collect(),
            )
        }
        "lanes" => (vec![], xyzw.iter().map(|component| format!("{path}.{component}")).collect()),
        "count" | "reference" | "choice" | "present" => (vec![], vec![format!("from {path}")]),
        "flag" => (vec![], vec![format!("lanes.bit {path}")]),
        "optional" => {
            let mut lanes = vec![format!("{local}_count")];
            lanes.extend(xyzw.iter().map(|component| format!("{local}.{component}")));
            (
                vec![
                    format!("let ({local}_count, {local}) = case {path} of"),
                    "  #Some value => (4.0, value)".to_owned(),
                    "  #Nothing => (0.0, types.zero4)".to_owned(),
                ],
                lanes,
            )
        }
        "children" => (vec![], vec![format!("from {path}.length")]),
        "tail" => (vec![], vec![format!("from (min {path}.length {})", bound_blot(&field["max"]))]),
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
    out.push(format!("const {} = fn ({variable}: types.{name}) => do:", encoder(&layout.record)));
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
            out.push(format!("  | #{name} {}", wire_type(&event["value"])));
        } else if !fields.is_empty() {
            out.push(format!(
                "  | #{name} {{ {} }}",
                fields
                    .iter()
                    .map(|(name, field)| format!("{name}: {}", wire_type(field)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        } else {
            out.push(format!("  | #{name}"));
        }
    }
    out.extend([
        String::new(),
        "type Input is data = #Input { elapsed: F32, width: F32, height: F32, events: Array Event, state: Array F32 }"
            .to_owned(),
        String::new(),
    ]);
    out.extend(comment(Some(&format!("An event's lanes from `start`: its tag, then {}. The {} lane holds a whole number; lanes an event does not use are zero.", input_lanes.iter().map(s).collect::<Vec<_>>().join(", "), s(&input_lanes[0])))));
    out.push("const decode_event = fn (values: Array F32) => fn (start: U32) => do:".to_owned());
    out.push(format!(
        "  let tag = lanes.integer values[start] {}",
        model.core_events().iter().map(|event| n(&event["tag"])).max().unwrap_or(0)
    ));
    for (index, lane) in input_lanes.iter().enumerate() {
        if index == 0 {
            out.push(format!("  let {} = lanes.integer values[start + 1] lanes.max_integer", s(lane)));
        } else {
            out.push(format!("  let {} = values[start + {}]", s(lane), index + 1));
        }
    }
    out.push("  return case tag of".to_owned());
    for event in model.core_events() {
        let fields = event_fields(event);
        let name = s(&event["name"]);
        out.push(format!("    {} => do:", event["tag"]));
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
            out.push(format!("      return #{name} {}", value(&event["value"])));
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
            out.push(format!("      return #{name} {{ {fields} }}"));
        } else {
            out.push(format!("      return #{name}"));
        }
    }
    out.push(format!("    _ => @panic {}", quote(&input["messages"]["tag"])));
    out.push(String::new());
    out.extend(comment(Some("The input packet: the seconds since the last frame, the viewport's width and height, the event and state counts, the events and the state. Anything else traps.")));
    out.extend("const decode_input = fn (values: Array F32) => do:\n  use lanes.require (values.length >= frame_header_floats)\n  use lanes.finite values\n  let elapsed = values[0]\n  use lanes.require (elapsed >= 0.0)\n  let width = values[1]\n  let height = values[2]\n  use lanes.require (width > 0.0)\n  use lanes.require (height > 0.0)".lines().map(str::to_owned));
    out.push(format!("  let event_count = lanes.integer values[3] {}", s(&input["maxEvents"]["limit"]).to_lowercase()));
    out.push(format!(
        "  let state_count = lanes.integer values[4] {}",
        s(&model.protocol["state"]["max"]["limit"]).to_lowercase()
    ));
    out.extend("  let state_start = frame_header_floats + event_count * event_floats\n  use lanes.require (values.length == state_start + state_count)\n  return #Input {\n    state: array.slice state_start state_count values,\n    elapsed, width, height,\n    events: array.generate event_count (fn index => decode_event values (frame_header_floats + index * event_floats)),\n  }".lines().map(str::to_owned));
    out
}
// Emit writes in the frame function itself: passing its mutable output to a
// per-record helper would introduce aliases and defeat unique-array reuse.
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

fn direct_records(out: &mut Vec<String>, entry: &Section, body: impl Fn(&mut Vec<String>, &str, &str)) {
    let variable = singular(s(&entry.section["name"]));
    if let Some((owners, children)) = &entry.owner {
        let owner = singular(owners);
        out.push(format!("  for {owner} in packet.{owners}:"));
        out.push(format!("    for {variable} in {owner}.{children}:"));
        body(out, &variable, "      ");
    } else {
        let section = s(&entry.section["name"]);
        let values = if section == "draws" { "draws".to_owned() } else { format!("packet.{section}") };
        out.push(format!("  for {variable} in {values}:"));
        if section == "draws" {
            out.push(format!("    let extra = extra_of {variable}"));
        }
        body(out, &variable, "    ");
    }
}

fn render_direct(model: &Model) -> Vec<String> {
    let mut out = vec![String::new(),
        "// Count first, then write every lane into one uniquely owned buffer.".to_owned(),
        "const draw_metadata = fn (draw: types.Draw) => (draw.mesh, draw.material)".to_owned(),
        "const draw_transform = fn (draw: types.Draw) => do:".to_owned(),
        "  let #types.Mat4 { x, y, z, w } = draw.transform".to_owned(),
        "  return (x.x, x.y, x.z, x.w, y.x, y.y, y.z, y.w, z.x, z.y, z.z, z.w, w.x, w.y, w.z, w.w)".to_owned(),
        "const draw_parameters = fn (draw: types.Draw) => case draw.parameters of".to_owned(),
        "  #Some p => (4.0, p.x, p.y, p.z, p.w)".to_owned(),
        "  #Nothing => (0.0, 0.0, 0.0, 0.0, 0.0)".to_owned(),
        "const encode_render = fn (packet: types.RenderFrame) =>".to_owned(),
        "  encode_render_draws packet packet.draws draw_metadata draw_transform draw_parameters encode_draw_extra".to_owned(),
        String::new(),
        "// Adapters supply scalar projections without constructing nominal wire copies.".to_owned(),
        "const encode_render_draws = fn (packet: types.RenderFrame) => fn (draws: Array d) => fn (metadata_of: d -> (U32, U32)) => fn (matrix_of: d -> (F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, F32)) => fn (parameters_of: d -> (F32, F32, F32, F32, F32)) => fn (extra_of: d -> Array F32) => do:".to_owned()];
    let mut counts = HashMap::new();
    let mut lengths = HashMap::new();
    let mut sizes = vec!["protocol.packet_header_floats".to_owned()];
    for entry in &model.sections {
        let name = s(&entry.section["name"]);
        if kind(&entry.section) == "trailing" {
            continue;
        }
        let count = if name == "draws" {
            "draws.length".to_owned()
        } else if let Some((owners, children)) = &entry.owner {
            let count = format!("{name}_count");
            out.push(format!("  let {count} = 0"));
            let owner = singular(owners);
            out.push(format!("  for {owner} in packet.{owners}:"));
            out.push(format!("    {count} := self + {owner}.{children}.length"));
            count
        } else {
            format!("packet.{name}.length")
        };
        counts.insert(name, count.clone());
        let length = match kind(&entry.section) {
            "lanes" => count,
            "records" => format!("{count} * {}", entry.fixed.as_ref().unwrap().lanes),
            "variable" => {
                let length = format!("{name}_lanes");
                out.push(format!("  let {length} = 0"));
                let rest = &entry.section["record"]["rest"];
                let rest_name = s(&rest["name"]);
                let width = entry.item.as_ref().map_or(1, |item| item.lanes);
                direct_records(&mut out, entry, |out, variable, indent| {
                    out.push(format!(
                        "{indent}{length} := self + {} + {variable}.{rest_name}.length * {width}",
                        entry.head_lanes
                    ));
                });
                length
            }
            _ => unreachable!(),
        };
        sizes.push(length.clone());
        lengths.insert(name, length);
    }
    for (name, owners) in &model.tails {
        let length = format!("{name}_lanes");
        out.push(format!("  let {length} = 0"));
        for owner in owners {
            let entry = model.section(&owner.section);
            direct_records(&mut out, entry, |out, variable, indent| {
                if owner.section == "draws" {
                    out.push(format!("{indent}{length} := self + extra.length"));
                } else {
                    let path = field_path(variable, &owner.field);
                    let max = bound_blot(&owner.field.field["max"]);
                    out.push(format!("{indent}{length} := self + min {path}.length {max} * 4"));
                }
            });
        }
        sizes.push(length);
    }
    out.push(format!("  let encoded = @array.fill ({}) 0.0", sizes.join(" + ")));
    out.push("  let at = 0".to_owned());
    for entry in &model.header {
        let field = &entry.laid;
        let lane = &field.field;
        let values = match kind(lane) {
            "magic" => vec!["protocol.packet_magic".to_owned()],
            "schema" => vec!["from packet.schema".to_owned()],
            "size" => {
                let section = s(&lane["section"]);
                vec![format!("from {}", if lane["of"] == "lanes" { &lengths[section] } else { &counts[section] })]
            }
            _ => {
                let prefix = entry.group.as_ref().map_or(String::new(), |group| format!("{group}."));
                let path = format!("packet.{prefix}{}", field.name);
                let (locals, values) = encode(lane, &path, &entry.key.replace('.', "_"));
                out.extend(locals.iter().map(|line| format!("  {line}")));
                values
            }
        };
        for (index, value) in values.iter().enumerate() {
            out.push(format!("  encoded[{}] := {value}", field.offset + index));
        }
    }
    out.push("  at := protocol.packet_header_floats".to_owned());
    for entry in &model.sections {
        let name = s(&entry.section["name"]);
        match kind(&entry.section) {
            "lanes" => {
                out.push(format!("  for value in packet.{name}:"));
                out.push("    encoded[at] := value".to_owned());
                out.push("    at := self + 1".to_owned());
            }
            "records" => {
                let layout = entry.fixed.as_ref().unwrap();
                direct_records(&mut out, entry, |out, variable, indent| {
                    if name == "draws" {
                        out.push(format!("{indent}let (mesh, material) = metadata_of {variable}"));
                        let matrix: Vec<_> = (0..16).map(|i| format!("m{i}")).collect();
                        out.push(format!("{indent}let ({}) = matrix_of {variable}", matrix.join(", ")));
                        out.push(format!("{indent}let (present, p0, p1, p2, p3) = parameters_of {variable}"));
                        for field in &layout.fields {
                            let values = match kind(&field.field) {
                                "reference" => vec![format!("U32.to_f32 {}", field.name)],
                                "matrix" => matrix.clone(),
                                "optional" => {
                                    vec!["present".into(), "p0".into(), "p1".into(), "p2".into(), "p3".into()]
                                }
                                "tail" => vec!["from (extra.length / 4)".into()],
                                _ => unreachable!(),
                            };
                            for (lane, value) in values.iter().enumerate() {
                                out.push(format!("{indent}encoded[at + {}] := {value}", field.offset + lane));
                            }
                        }
                    } else {
                        direct_fields(out, &layout.fields, variable, indent);
                    }
                    out.push(format!("{indent}at := self + {}", layout.lanes));
                });
            }
            "variable" => {
                let rest = &entry.section["record"]["rest"];
                let rest_name = s(&rest["name"]);
                direct_records(&mut out, entry, |out, variable, indent| {
                    direct_fields(out, &entry.head, variable, indent);
                    if let Some(layout) = &entry.item {
                        out.push(format!(
                            "{indent}encoded[at + {}] := from {variable}.{rest_name}.length",
                            entry.head_lanes - 1
                        ));
                        out.push(format!("{indent}at := self + {}", entry.head_lanes));
                        let item = singular(rest_name);
                        out.push(format!("{indent}for {item} in {variable}.{rest_name}:"));
                        direct_fields(out, &layout.fields, &item, &format!("{indent}  "));
                        out.push(format!("{indent}  at := self + {}", layout.lanes));
                    } else {
                        out.push(format!("{indent}at := self + {}", entry.head_lanes));
                        out.push(format!("{indent}for value in {variable}.{rest_name}:"));
                        out.push(format!("{indent}  encoded[at] := value"));
                        out.push(format!("{indent}  at := self + 1"));
                    }
                });
            }
            "trailing" => {
                if let Some((_, owners)) = model.tails.iter().find(|(section, _)| section == name) {
                    for owner in owners {
                        direct_records(&mut out, model.section(&owner.section), |out, variable, indent| {
                            if owner.section == "draws" {
                                out.push(format!("{indent}for value in extra:"));
                                out.push(format!("{indent}  encoded[at] := value"));
                                out.push(format!("{indent}  at := self + 1"));
                            } else {
                                let path = field_path(variable, &owner.field);
                                let max = bound_blot(&owner.field.field["max"]);
                                out.push(format!("{indent}for index in 0 .. min {path}.length {max}:"));
                                for (lane, component) in ["x", "y", "z", "w"].iter().enumerate() {
                                    out.push(format!("{indent}  encoded[at + {lane}] := {path}[index].{component}"));
                                }
                                out.push(format!("{indent}  at := self + 4"));
                            }
                        });
                    }
                }
            }
            _ => unreachable!(),
        }
    }
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
            out.push(format!("const {} = fn ({name}: types.{}) => do:", encoder(record), s(&record["name"])));
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
            lines.push(format!("encoded[{lane}] := from {rest_name}.length"));
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
            out.push(format!("const {} = fn ({name}: types.{}) => do:", encoder(record), s(&record["name"])));
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
            out.push(format!("const {name} = fn ({variable}: types.{}) => do:", s(&owner.layout.record["name"])));
            out.push(format!("  let kept = min {path}.length {max}"));
            out.push(format!(
                "  return #[lane | index <- array.generate kept identity, lane <- vec4_lanes {path}[index]]"
            ));
        }
    }
    if !tail_encoders.is_empty() {
        out.extend([String::new(), "const vec4_lanes = fn (v: types.Vec4) => #[v.x, v.y, v.z, v.w]".to_owned()]);
    }
    out.extend(render_direct(model));
    if let Some(extensions) = model.extensions() {
        out.push(String::new());
        out.extend(comment(Some(
            "The lanes of the game's own sections, to follow encode_render's: none without any, else the \
             marker, their count, then each one's tag, length and values. A game's generated \
             pen/extensions.blot makes them from its declarations.",
        )));
        out.push("const encode_extensions = fn (extensions: Array types.Extension) => do:".to_owned());
        out.push("  if extensions.is_empty:".to_owned());
        out.push("    return #[]".to_owned());
        out.push(format!("  use lanes.require (extensions.length <= {})", bound_blot(&extensions["max"])));
        out.push("  return array.flatten #[".to_owned());
        out.push("    #[protocol.extensions_marker, from extensions.length],".to_owned());
        // A spread inside a comprehension's arrays is past blotc's evaluator
        // (`constant_expression`), so each one's lanes are concatenated.
        out.push(
            "    array.flatten #[array.concat #[from extension.tag, from extension.values.length] extension.values | extension <- extensions],"
                .to_owned(),
        );
        out.push("  ]".to_owned());
    }
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
        "tail" => "Array Vec4".to_owned(),
        "children" => format!("Array {}", s(&model.section(s(&field["section"])).section["record"]["name"])),
        _ => unreachable!(),
    }
}

fn structure(out: &mut Vec<String>, name: &str, fields: Vec<String>) {
    out.push(String::new());
    out.push(format!("type {name} is data = #{name} {{"));
    out.extend(fields.into_iter().map(|field| format!("  {field},")));
    out.push("}".to_owned());
}

fn member(model: &Model, laid: &Laid) -> String {
    format!("{}: {}", laid.name, field_type(model, &laid.field))
}

fn types(model: &Model) -> Vec<String> {
    let mut out = vec![
        "// Wire records. Mat4 stores four columns; vectors follow schema lane order.".to_owned(),
        "type Vec4 is data = #Vec4 { x: F32, y: F32, z: F32, w: F32 }".to_owned(),
        "type Mat4 is data = #Mat4 { x: Vec4, y: Vec4, z: Vec4, w: Vec4 }".to_owned(),
        "const zero4 = #Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }".to_owned(),
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
        fields.push(format!("{}: Array {ty}", s(&rest["name"])));
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
    for entry in &model.sections {
        if kind(&entry.section) == "trailing" || entry.owner.is_some() {
            continue;
        }
        let mut ty = "F32";
        if kind(&entry.section) != "lanes" {
            ty = s(&entry.section["record"]["name"]);
        }
        fields.push(format!("{}: Array {ty}", s(&entry.section["name"])));
    }
    structure(&mut out, "RenderFrame", fields);
    if model.extensions().is_some() {
        out.push(String::new());
        out.extend(comment(Some(
            "One of the game's own sections (its pen.json `protocol`): its tag and lanes. They travel after the \
             render packet, which RenderFrame leaves them out of: protocol_render's encode_extensions lays them out.",
        )));
        out.push("type Extension is data = #Extension { tag: U32, values: Array F32 }".to_owned());
    }
    out
}

const NUMERIC: &str = r#"import * as array from "std/array"

const max_integer = 16777215

const require = fn (condition: Bool) => do:
  if condition:
    return ()
  return @panic "invalid Pen numeric packet"

const finite = fn (values: Array F32) => require (array.all is_finite values)

const integer = fn (value: F32) => fn (maximum: U32) => do:
  let number = F32.to_u32 value
  use require (value == U32.to_f32 number)
  use require (number <= maximum)
  return number

const bit = fn (value: Bool) => do:
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
    let program = crate::program::program(&top);
    let frame = crate::program::frame(model, &top);
    let check = crate::program::check(&top);
    let mut numeric = top;
    numeric.extend(NUMERIC.lines().map(str::to_owned));
    numeric.push(String::new());
    vec![
        ("protocol.blot", protocol.join("\n")),
        ("protocol_render.blot", render_source.join("\n")),
        ("types.blot", wire_types.join("\n")),
        ("lanes.blot", numeric.join("\n")),
        ("protocol_canonical.blot", canonical),
        ("program.blot", program),
        ("frame.blot", frame),
        ("check.blot", check),
    ]
}
