//! Test-only flat packet projections into the production view encoder.
use crate::model::*;
use std::collections::{HashMap, HashSet};

fn scalar(field: &serde_json::Value, at: &str) -> String {
    match kind(field) {
        "count" | "reference" | "choice" | "present" | "schema" => format!("F32.to_u32 packet[{at}]"),
        "flag" => format!("packet[{at}] == 1.0"),
        "matrix" => format!(
            "$types.Mat4 {{ {} }}",
            ["x", "y", "z", "w"]
                .iter()
                .enumerate()
                .map(|(i, c)| format!("{c}: vec4_at packet ({at} + {})", i * 4))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        "vector" => format!(
            "$types.{} {{ {} }}",
            s(&field["type"]["name"]),
            array(&field["type"]["components"])
                .iter()
                .enumerate()
                .map(|(i, c)| format!("{}: packet[{at} + {i}]", s(c)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        "lanes" => format!("vec4_at packet ({at})"),
        "optional" => format!("if packet[{at}] == 4.0 then $Some (vec4_at packet ({at} + 1)) else $Nothing"),
        "tail" => format!(
            "array.generate (F32.to_u32 packet[{at}]) (func i => vec4_at packet (extra_start context at + i * 4))"
        ),
        "children" => {
            format!("array.generate (F32.to_u32 packet[{at}]) (func i => child_start context at + i * 24)")
        }
        _ => format!("packet[{at}]"),
    }
}

fn projected(section: &Section, out: &mut Vec<String>) {
    let name = singular(s(&section.section["name"]));
    let (fields, width) =
        section.fixed.as_ref().map_or((&section.head, section.head_lanes), |layout| (&layout.fields, layout.lanes));
    out.push(format!("func project_{name} (context: Context) -> (at: U32) =>"));
    out.push("  let packet: #[F32] = context.packet".to_owned());
    let mut values: Vec<_> = fields
        .iter()
        .map(|field| format!("{}: {}", field.name, scalar(&field.field, &format!("at + {}", field.offset))))
        .collect();
    let rest = &section.section["record"]["rest"];
    if rest.is_object() {
        let value = if let Some(item) = &section.item {
            let elements: Vec<_> = item
                .fields
                .iter()
                .map(|field| {
                    format!(
                        "{}: {}",
                        field.name,
                        scalar(&field.field, &format!("at + {width} + i * {} + {}", item.lanes, field.offset))
                    )
                })
                .collect();
            format!(
                "array.generate (F32.to_u32 packet[at + {}]) (func i => $types.{} {{ {} }})",
                width - 1,
                s(&item.record["name"]),
                elements.join(", ")
            )
        } else {
            let count = array(&rest["of"])
                .iter()
                .map(|name| {
                    let offset = fields.iter().find(|field| field.name == s(name)).unwrap().offset;
                    format!("F32.to_u32 packet[at + {offset}]")
                })
                .collect::<Vec<_>>()
                .join(" * ");
            format!("packet.view (at + {width}) (at + {width} + {count})")
        };
        values.push(format!("{}: {value}", s(&rest["name"])));
    }
    let nominal = if rest.is_object() || fields.iter().any(|field| matches!(kind(&field.field), "tail" | "children")) {
        String::new()
    } else {
        format!("$types.{} ", s(&section.section["record"]["name"]))
    };
    out.push(format!("  return {nominal}{{ {} }}", values.join(", ")));
}

pub fn emit(model: &Model, header: &[String]) -> String {
    let mut out = header.to_vec();
    out.extend([
        "// Test-only numeric packet projections into the production render encoder.".to_owned(),
        "import * as array from \"std/array\"".to_owned(),
        "import * as types from \"./types\"".to_owned(),
        "import * as renderer from \"./protocol_render\"".to_owned(),
        String::new(),
        "type Context = {packet: #[F32], draws_start: U32, draws_count: U32, inset_draws_start: U32, inset_draws_count: U32, insets_start: U32, insets_count: U32, extras_start: U32}".to_owned(),
        "func vec4_at (packet: #[F32]) -> (at: U32) => $types.Vec4 {x: packet[at], y: packet[at + 1], z: packet[at + 2], w: packet[at + 3]}".to_owned(),
        "func extra_start (context: Context) -> (at: U32) =>".to_owned(),
        "  let packet: #[F32] = context.packet".to_owned(),
        "  let start: U32 = context.extras_start".to_owned(),
    ]);
    let draws = model.section("draws").fixed.as_ref().unwrap();
    let tail = draws.fields.iter().find(|field| kind(&field.field) == "tail").unwrap();
    for section in ["draws", "inset_draws"] {
        out.push(format!("  for i in 0 .. context.{section}_count:"));
        out.push(format!("    let offset = context.{section}_start + i * {}", draws.lanes));
        out.push("    if offset < at:".to_owned());
        out.push(format!("      start := start + F32.to_u32 packet[offset + {}] * 4", tail.offset));
    }
    out.push("  return start".to_owned());
    let insets = model.section("insets").fixed.as_ref().unwrap();
    let children = insets.fields.iter().find(|field| kind(&field.field) == "children").unwrap();
    out.extend([
        "func child_start (context: Context) -> (at: U32) =>".to_owned(),
        "  let packet: #[F32] = context.packet".to_owned(),
        "  let start: U32 = context.inset_draws_start".to_owned(),
        format!("  for i in 0 .. ((at - context.insets_start) / {}):", insets.lanes),
        format!(
            "    start := start + F32.to_u32 packet[context.insets_start + i * {} + {}] * {}",
            insets.lanes, children.offset, draws.lanes
        ),
        "  return start".to_owned(),
        "func inset_draws (context: Context) -> (at: U32) =>".to_owned(),
        format!("  let count = F32.to_u32 context.packet[at + {}]", children.offset),
        "  let first = child_start context at".to_owned(),
        format!("  return array.generate count (func i => first + i * {})", draws.lanes),
        "func draw_metadata (context: Context) -> (at: U32) =>".to_owned(),
        "  let packet: #[F32] = context.packet".to_owned(),
        "  return (F32.to_u32 packet[at], F32.to_u32 packet[at + 1])".to_owned(),
        "func draw_transform (context: Context) -> (at: U32) =>".to_owned(),
        "  let packet: #[F32] = context.packet".to_owned(),
        format!("  return ({})", (2..18).map(|offset| format!("packet[at + {offset}]")).collect::<Vec<_>>().join(", ")),
        "func draw_parameters (context: Context) -> (at: U32) =>".to_owned(),
        "  let packet: #[F32] = context.packet".to_owned(),
        "  return (packet[at + 18], packet[at + 19], packet[at + 20], packet[at + 21], packet[at + 22])".to_owned(),
        "func extra_count (context: Context) -> (at: U32) =>".to_owned(),
        format!("  return F32.to_u32 context.packet[at + {}]", tail.offset),
        "func append_extra (context: Context) -> (at: U32) -> (encoded: #[~F32]) =>".to_owned(),
        "  let first = extra_start context at".to_owned(),
        "  for lane in 0 .. extra_count context at * 4:".to_owned(),
        "    @storage.push encoded context.packet[first + lane]".to_owned(),
        "  return ()".to_owned(),
    ]);
    for section in &model.sections {
        if matches!(kind(&section.section), "records" | "variable") && s(&section.section["name"]) != "draws" {
            projected(section, &mut out);
        }
    }
    out.push("func reencode_views (packet: #[F32]) -> (destination: own #[~F32]) =>".to_owned());
    let mut start = model.header_lanes.to_string();
    for section in &model.sections {
        let name = s(&section.section["name"]);
        out.push(format!("  let {name}_start: U32 = {start}"));
        if kind(&section.section) == "trailing" {
            break;
        }
        let sizes: HashMap<_, _> = model
            .header
            .iter()
            .filter(|entry| kind(&entry.laid.field) == "size" && entry.laid.field["section"] == name)
            .map(|entry| (s(&entry.laid.field["of"]), entry.laid.offset))
            .collect();
        let offset = sizes.get("count").unwrap_or_else(|| sizes.get("lanes").unwrap());
        out.push(format!("  let {name}_count: U32 = F32.to_u32 packet[{offset}]"));
        let length = match kind(&section.section) {
            "variable" => format!("F32.to_u32 packet[{}]", sizes["lanes"]),
            "lanes" => format!("{name}_count"),
            _ => format!("{name}_count * {}", section.fixed.as_ref().unwrap().lanes),
        };
        start = format!("{name}_start + {length}");
    }
    out.push("  let context: Context = {packet, draws_start, draws_count, inset_draws_start, inset_draws_count, insets_start, insets_count, extras_start}".to_owned());
    let mut fields = Vec::new();
    let mut groups = HashSet::new();
    for entry in &model.header {
        if matches!(kind(&entry.laid.field), "magic" | "size") {
            continue;
        }
        if let Some(group) = &entry.group {
            if !groups.insert(group) {
                continue;
            }
            let values: Vec<_> = model
                .header
                .iter()
                .filter(|entry| entry.group.as_ref() == Some(group))
                .map(|entry| {
                    format!("{}: {}", entry.laid.name, scalar(&entry.laid.field, &entry.laid.offset.to_string()))
                })
                .collect();
            fields.push(format!(
                "{group}: $types.{} {{ {} }}",
                s(&model.protocol["render"]["groups"][group]["type"]),
                values.join(", ")
            ));
        } else {
            fields.push(format!("{}: {}", entry.laid.name, scalar(&entry.laid.field, &entry.laid.offset.to_string())));
        }
    }
    out.push(format!("  let header: types.RenderHeader = {{ {} }}", fields.join(", ")));
    let mut collections = Vec::new();
    for section in &model.sections {
        if section.owner.is_some() || kind(&section.section) == "trailing" {
            continue;
        }
        let name = s(&section.section["name"]);
        collections.push(name);
        match kind(&section.section) {
            "lanes" => out.push(format!("  let {name} = packet.view {name}_start ({name}_start + {name}_count)")),
            "variable" => {
                let width = section.head_lanes;
                let length = if let Some(item) = &section.item {
                    format!("F32.to_u32 packet[at + {}] * {}", width - 1, item.lanes)
                } else {
                    array(&section.section["record"]["rest"]["of"])
                        .iter()
                        .map(|field| {
                            let offset = section.head.iter().find(|entry| entry.name == s(field)).unwrap().offset;
                            format!("F32.to_u32 packet[at + {offset}]")
                        })
                        .collect::<Vec<_>>()
                        .join(" * ")
                };
                out.extend([
                    format!("  let {name}: #[+U32] = #[]"),
                    format!("  let at = {name}_start"),
                    format!("  for 0 .. {name}_count:"),
                    format!("    @storage.push {name} at"),
                    format!("    at := at + {width} + {length}"),
                ]);
            }
            _ => out.push(format!(
                "  let {name} = array.generate {name}_count (func i => {name}_start + i * {})",
                section.fixed.as_ref().unwrap().lanes
            )),
        }
    }
    out.push(format!("  let collections = {{ {} }}", collections.join(", ")));
    let mut projections = vec![
        "draw_metadata".to_owned(),
        "draw_transform".to_owned(),
        "draw_parameters".to_owned(),
        "extra_count".to_owned(),
        "append_extra".to_owned(),
        "inset_draws".to_owned(),
    ];
    for section in &model.sections {
        if matches!(kind(&section.section), "records" | "variable") && s(&section.section["name"]) != "draws" {
            let name = singular(s(&section.section["name"]));
            projections.push(format!("{name}: project_{name}"));
        }
    }
    out.extend([
        format!("  let project = {{ {} }}", projections.join(", ")),
        "  return renderer.encode_render_views header collections project context destination".to_owned(),
        String::new(),
    ]);
    out.join("\n")
}
