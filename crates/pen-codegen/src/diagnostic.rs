//! A defensive error-only walk of the same model used by the fast decoder.
use crate::model::*;

fn offset(base: &str, lane: usize) -> String {
    if base == "0" {
        lane.to_string()
    } else if lane == 0 {
        base.into()
    } else {
        format!("{base} + {lane}")
    }
}

fn field(model: &Model, laid: &Laid, base: &str, path: &str) -> Vec<String> {
    let value = &laid.field;
    let name = &laid.name;
    let lane = offset(base, laid.offset);
    let at = format!("&{path}.field({name:?})");
    let count = |maximum: String| format!("d.count({at}, {lane}, {maximum})?");
    let statement = match kind(value) {
        "f32" | "choice" => format!("d.finite({at}, {lane})?;"),
        "matrix" | "lanes" | "vector" => {
            let components = if kind(value) == "vector" { quote(&value["type"]["components"]) } else { "[]".into() };
            format!("d.values({at}, {lane}, {}, &{components})?;", laid.lanes)
        }
        "count" => format!("let {name} = {};", count(bound(&value["max"]))),
        "flag" => format!("{};", count("1".into())),
        "reference" | "present" => {
            let catalog = &model.protocol["catalog"][s(&value["to"])];
            let list = s(&catalog["list"]);
            if kind(value) == "present" {
                format!("{};", count(format!("catalog.{list}.len()")))
            } else {
                format!("d.reference({at}, {lane}, catalog.{list}.len(), {})?;", !catalog["empty"].is_null())
            }
        }
        "optional" => {
            let size = n(&value["count"]);
            format!(
                "let count = d.count(&{path}.field({name:?}).field(\"count\"), {lane}, {size})?;\nif count != 0 && count != {size} {{ return Err({path}.field({name:?}).field(\"count\").error({lane}, \"0 or {size}\", Actual::Value(count as f32))); }}\nd.values({at}, {lane} + 1, {size}, &[])?;"
            )
        }
        "children" => {
            let section = s(&value["section"]);
            let record = &model.section(section).fixed.as_ref().unwrap().record;
            format!(
                "let count = {};\nlet end = d.end({at}, {section}_cursor, count, {})?;\nd.within({at}, {section}_cursor, end, {section}_end)?;\nfor child in 0..count {{\n    inspect_{}(d, catalog, {section}_cursor, &{path}.field({name:?}).index(child), extras_cursor)?;\n    {section}_cursor += {};\n}}",
                count(bound(&value["max"])),
                floats(record),
                snake(s(&record["name"])),
                floats(record)
            )
        }
        "tail" => format!(
            "let count = {};\nlet end = d.end({at}, *extras_cursor, count, {})?;\nd.values({at}, *extras_cursor, end - *extras_cursor, &[])?;\n*extras_cursor = end;",
            count(bound(&value["max"])),
            value["unit"]
        ),
        _ => unreachable!("{}", kind(value)),
    };
    statement.lines().map(str::to_owned).collect()
}

pub fn emit(model: &Model) -> Vec<String> {
    let mut out = vec!["    use crate::diagnostic::{Actual, Check, FrameDiagnostic, Inspector, Location};".into()];
    let push = |out: &mut Vec<String>, indent: usize, lines: Vec<String>| {
        out.extend(lines.into_iter().map(|line| format!("{}{line}", " ".repeat(indent))));
    };
    // These readers run only after a failure, and do not allocate frame vectors.
    for layout in &model.records {
        if !model.sections.iter().any(|section| {
            section.owner.is_some()
                && section.fixed.as_ref().is_some_and(|value| value.record["name"] == layout.record["name"])
                || section.item.as_ref().is_some_and(|value| value.record["name"] == layout.record["name"])
        }) {
            continue;
        }
        let record = snake(s(&layout.record["name"]));
        out.push(format!("    #[allow(unused_variables)]\n    fn inspect_{record}(d: &Inspector<'_>, catalog: &Catalog, base: usize, path: &Location, extras_cursor: &mut usize) -> Check {{"));
        for laid in &layout.fields {
            // Children need section cursors from the enclosing packet walk.
            if kind(&laid.field) != "children" {
                push(&mut out, 8, field(model, laid, "base", "path"));
            }
        }
        out.push("        Ok(())\n    }".into());
    }
    out.push("    #[allow(unused_variables)]\n    pub(crate) fn inspect_frame(packet: &[f32], schema: u32, catalog: &Catalog) -> Check {\n        let d = &Inspector { packet };\n        let header = Location::section(\"header\");\n        d.equal(&header.field(\"magic\"), 0, PACKET_MAGIC)?;\n        d.equal(&header.field(\"schema\"), 1, schema as f32)?;".into());
    let layout_start = out.len();
    for entry in &model.header {
        let value = &entry.laid.field;
        if kind(value) != "size" {
            continue;
        }
        let section = &model.section(s(&value["section"])).section;
        let name = s(&value["section"]);
        let of = s(&value["of"]);
        let max = if of == "count" {
            bound(&section["max"])
        } else if kind(section) == "variable" {
            bound(&section["maxLanes"])
        } else {
            bound(&section["max"])
        };
        out.push(format!(
            "        let {name}_{of} = d.count(&header.field({:?}), {}, {max})?;",
            entry.key, entry.laid.offset
        ));
        if let Some(counts) = section["exactly"]["counts"].as_array() {
            let counts = counts.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
            out.push(format!("        if ![{counts}].contains(&{name}_{of}) {{ return Err(header.field({:?}).error({}, \"one of [{counts}]\", Actual::Value({name}_{of} as f32))); }}", entry.key, entry.laid.offset));
        }
    }
    let mut previous = "PACKET_HEADER_FLOATS".to_owned();
    for entry in &model.sections {
        let name = s(&entry.section["name"]);
        out.push(format!("        let {name}_start = {previous};"));
        if kind(&entry.section) == "trailing" {
            continue;
        }
        let (count, width) = if let Some(layout) = &entry.fixed {
            (format!("{name}_count"), floats(&layout.record))
        } else {
            (format!("{name}_lanes"), "1".into())
        };
        out.push(format!(
            "        let {name}_end = d.end(&Location::section({name:?}), {name}_start, {count}, {width})?;"
        ));
        previous = format!("{name}_end");
        if entry.owner.is_some() {
            out.push(format!("        let mut {name}_cursor = {name}_start;"));
        }
    }
    let layout = out[layout_start..].to_vec();
    out.push("        let extras_cursor = &mut { extras_start };".into());
    for entry in &model.header {
        if matches!(kind(&entry.laid.field), "magic" | "schema" | "size") {
            continue;
        }
        let mut laid = entry.laid.clone();
        laid.name = entry.key.clone();
        // Dotted header names are paths, not Rust variable names.
        let lines =
            field(model, &laid, "0", "header")
                .into_iter()
                .map(|line| {
                    if line.starts_with("let ") {
                        line.replace(&format!("let {} =", laid.name), "let _ =")
                    } else {
                        line
                    }
                })
                .collect();
        push(&mut out, 8, lines);
    }
    for entry in &model.sections {
        let section = &entry.section;
        let name = s(&section["name"]);
        if entry.owner.is_some() || kind(section) == "trailing" {
            continue;
        }
        if kind(section) == "lanes" {
            out.push(format!("        d.values(&Location::section({name:?}), {name}_start, {name}_lanes, &[])?;"));
            continue;
        }
        out.push(format!("        let mut base = {name}_start;\n        for item in 0..{name}_count {{\n            let path = &Location::section({name:?}).index(item);"));
        if let Some(layout) = &entry.fixed {
            // Inline parents: their children share packet-level section cursors.
            for laid in &layout.fields {
                push(&mut out, 12, field(model, laid, "base", "path"));
            }
            out.push(format!("            base += {};", floats(&layout.record)));
        } else {
            let record = &section["record"];
            let rest = &record["rest"];
            let rest_name = s(&rest["name"]);
            out.push(format!(
                "            d.within(path, base, d.end(path, base, 1, {})?, {name}_end)?;",
                entry.head_lanes
            ));
            for laid in &entry.head {
                push(&mut out, 12, field(model, laid, "base", "path"));
            }
            let count;
            let width;
            if let Some(layout) = &entry.item {
                count = format!("{rest_name}_count");
                width = floats(&layout.record);
                out.push(format!(
                    "            let {count} = d.count(&path.field(\"{rest_name}.count\"), {}, {})?;",
                    offset("base", entry.head_lanes - 1),
                    bound(&rest["max"])
                ));
                out.push(format!("            if {count} < {} {{ return Err(path.field(\"{rest_name}.count\").error({}, \"at least {}\", Actual::Value({count} as f32))); }}", rest["min"], offset("base", entry.head_lanes - 1), rest["min"]));
            } else {
                count = array(&rest["of"]).iter().map(s).collect::<Vec<_>>().join(" * ");
                width = "1".into();
            }
            out.push(format!("            let first = base + {};\n            let end = d.end(&path.field({rest_name:?}), first, {count}, {width})?;\n            d.within(&path.field({rest_name:?}), first, end, {name}_end)?;", entry.head_lanes));
            if let Some(layout) = &entry.item {
                out.push(format!("            for child in 0..{count} {{\n                inspect_{}(d, catalog, first + child * {width}, &path.field({rest_name:?}).index(child), extras_cursor)?;\n            }}", snake(s(&layout.record["name"]))));
            } else {
                out.push(format!("            d.values(&path.field({rest_name:?}), first, end - first, &[])?;"));
            }
            out.push("            base = end;".into());
        }
        out.push(format!("        }}\n        d.exhausted(&Location::section({name:?}), base, {name}_end)?;"));
        if let Some(layout) = &entry.fixed {
            for laid in &layout.fields {
                if kind(&laid.field) == "children" {
                    let child = s(&laid.field["section"]);
                    out.push(format!(
                        "        d.exhausted(&Location::section({child:?}), {child}_cursor, {child}_end)?;"
                    ));
                }
            }
        }
    }
    out.push("        inspect_extensions(d, *extras_cursor)?;\n        Ok(())\n    }".into());
    out.push("    #[cold]\n    fn explain_frame(packet: &[f32], schema: u32, catalog: &Catalog) -> FrameDiagnostic {\n        let mut diagnostic = inspect_frame(packet, schema, catalog).err().unwrap_or_else(|| Location::section(\"packet\").error(0, \"a valid render packet (structural context unavailable)\", Actual::Length(packet.len())));\n        if diagnostic.section != \"extensions\" && let Ok(start) = provenance_start(packet, schema) {\n            diagnostic.origin = crate::diagnostic::origin(packet, start, diagnostic.lane).map(Box::new);\n        }\n        diagnostic\n    }".into());
    out.push("    #[allow(unused_variables, unused_mut)]\n    fn provenance_start(packet: &[f32], schema: u32) -> Check<usize> {\n        let d = &Inspector { packet };\n        let header = Location::section(\"header\");\n        d.equal(&header.field(\"magic\"), 0, PACKET_MAGIC)?;\n        d.equal(&header.field(\"schema\"), 1, schema as f32)?;".into());
    out.extend(layout);
    out.push("        let mut end = extras_start;".into());
    for (_, owners) in &model.tails {
        for owner in owners {
            out.push(format!("        for item in 0..{}_count {{\n            let count = d.count(&Location::section({:?}).index(item).field({:?}), {}_start + item * {} + {}, {})?;\n            end = d.end(&Location::section(\"extras\"), end, count, {})?;\n        }}", owner.section, owner.section, owner.field.name, owner.section, floats(&owner.layout.record), owner.field.offset, bound(&owner.field.field["max"]), owner.field.field["unit"]));
        }
    }
    out.push("        Ok(end)\n    }".into());
    out.push("    fn inspect_extensions(d: &Inspector<'_>, mut at: usize) -> Check {\n        let start = at;\n        let path = &Location::section(\"extensions\");\n        let mut provenance_seen = false;\n        if at < d.packet.len() {\n            d.equal(&path.field(\"marker\"), at, EXTENSIONS_MARKER)?;\n            let count = d.count(&path.field(\"count\"), at + 1, MAX_EXTENSIONS)?;\n            at += 2;\n            for item in 0..count {\n                let path = &path.index(item);\n                let tag = d.count(&path.field(\"tag\"), at, MAX_EXTENSION_TAG)?;\n                let length = d.count(&path.field(\"length\"), at + 1, if tag == PROVENANCE_TAG { MAX_PROVENANCE_LANES } else { MAX_EXTENSION_LANES })?;\n                d.values(&path.field(\"values\"), at + 2, length, &[])?;\n                if tag == PROVENANCE_TAG {\n                    if provenance_seen || crate::diagnostic::read_provenance(&d.packet[at + 2..at + 2 + length], start).is_err() {\n                        return Err(path.field(\"provenance\").error(at + 2, \"one bounded version-1 provenance JSON object\", Actual::Length(length)));\n                    }\n                    provenance_seen = true;\n                }\n                at = d.end(path, at + 2, length, 1)?;\n            }\n        }\n        d.exhausted(path, at, d.packet.len())\n    }".into());
    out
}
