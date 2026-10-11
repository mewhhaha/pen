use crate::model::*;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

fn doc(text: Option<&str>, indent: &str) -> Vec<String> {
    text.map(|text| {
        wrap(text, 96 - indent.len() - 4)
            .iter()
            .map(|line| format!("{indent}/// {line}"))
            .collect()
    })
    .unwrap_or_default()
}
fn field_type(model: &Model, field: &Value) -> (String, bool) {
    let ty = match kind(field) {
        "f32" => "f32".to_owned(),
        "count" | "choice" | "reference" | "present" => "u32".to_owned(),
        "flag" => "bool".to_owned(),
        "vector" => s(&field["type"]["name"]).to_owned(),
        "matrix" => "Matrix".to_owned(),
        "lanes" => format!("[f32; {}]", field["count"]),
        "optional" => format!("Option<[f32; {}]>", field["count"]),
        "tail" => return ("Option<Vec<f32>>".to_owned(), false),
        "children" => {
            return (
                format!(
                    "Vec<{}>",
                    s(&model
                        .section(s(&field["section"]))
                        .fixed
                        .as_ref()
                        .unwrap()
                        .record["name"])
                ),
                false,
            );
        }
        _ => unreachable!(),
    };
    (ty, true)
}
struct Member {
    name: String,
    ty: String,
    copy: bool,
    doc: Option<String>,
}
fn member(model: &Model, laid: &Laid) -> Member {
    let (ty, copy) = field_type(model, &laid.field);
    Member {
        name: laid.name.clone(),
        ty,
        copy,
        doc: text(&laid.field, "doc"),
    }
}
fn structure(name: &str, about: Option<&str>, fields: &[Member]) -> Vec<String> {
    let mut out = doc(about, "    ");
    let mut copy = "";
    if fields.iter().all(|field| field.copy) {
        copy = "Copy, ";
    }
    out.push(format!(
        "    #[derive(Clone, {copy}Debug, Default, PartialEq)]"
    ));
    out.push(format!("    pub struct {name} {{"));
    for field in fields {
        out.extend(doc(field.doc.as_deref(), "        "));
        out.push(format!("        pub {}: {},", field.name, field.ty));
    }
    out.push("    }".to_owned());
    out
}
fn emit_record(
    out: &mut Vec<String>,
    model: &Model,
    record: &Value,
    fields: &[Laid],
    extra: Option<Member>,
    emitted: &mut HashSet<String>,
) {
    if !emitted.insert(s(&record["name"]).to_owned()) {
        return;
    }
    let mut fields: Vec<_> = fields.iter().map(|field| member(model, field)).collect();
    if let Some(field) = extra {
        fields.push(field);
    }
    out.extend(structure(
        s(&record["name"]),
        record["doc"].as_str(),
        &fields,
    ));
}
fn types(model: &Model) -> Vec<String> {
    let protocol = &model.protocol;
    let mut out = "/// The packets' meaning, as hosts see it. Asset references are indices into the\n/// [`Catalog`]; the decoder has checked each one is in range.\npub mod types {\n    /// A column-major 4x4 matrix, as WGSL and the packet lay it out.\n    pub type Matrix = [f32; 16];".lines().map(str::to_owned).collect::<Vec<_>>();
    for enumeration in array(&protocol["enumerations"]) {
        out.extend(doc(enumeration["doc"].as_str(), "    "));
        out.push(format!(
            "    pub mod {} {{",
            s(&enumeration["constant"]).to_lowercase()
        ));
        for (code, value) in array(&enumeration["values"]).iter().enumerate() {
            out.push(format!(
                "        pub const {}: u32 = {code};",
                s(value).to_uppercase()
            ));
        }
        out.push("        /// The names of the codes, by code.".to_owned());
        out.push(format!(
            "        pub const NAMES: [&str; {}] = [{}];",
            array(&enumeration["values"]).len(),
            array(&enumeration["values"])
                .iter()
                .map(quote)
                .collect::<Vec<_>>()
                .join(", ")
        ));
        out.push("    }".to_owned());
    }
    for vector in array(&protocol["vectors"]) {
        let fields = array(&vector["components"])
            .iter()
            .map(|name| Member {
                name: s(name).to_owned(),
                ty: "f32".to_owned(),
                copy: true,
                doc: None,
            })
            .collect::<Vec<_>>();
        out.extend(structure(
            s(&vector["name"]),
            vector["doc"].as_str(),
            &fields,
        ));
    }
    out.extend(structure(
        "Viewport",
        None,
        &[
            Member {
                name: "width".to_owned(),
                ty: "u32".to_owned(),
                copy: true,
                doc: None,
            },
            Member {
                name: "height".to_owned(),
                ty: "u32".to_owned(),
                copy: true,
                doc: None,
            },
        ],
    ));
    out.extend([
        "    #[derive(Clone, Debug, PartialEq)]".to_owned(),
        "    pub enum GameEvent {".to_owned(),
    ]);
    for event in model.events() {
        for host in hosts(event) {
            let name = s(&host["name"]);
            out.extend(doc(host["doc"].as_str(), "        "));
            let fields = if host["fields"].is_object() {
                &host["fields"]
            } else {
                &event["fields"]
            };
            if fields.is_object() {
                out.push(format!(
                    "        {name} {{ {} }},",
                    object(fields)
                        .iter()
                        .map(|(name, field)| format!("{name}: {}", s(&field["rust"])))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            } else if !event["value"].is_null() {
                out.push(format!("        {name}({}),", s(&event["value"]["rust"])));
            } else {
                out.push(format!("        {name},"));
            }
        }
    }
    out.extend("    }\n    #[derive(Clone, Debug, PartialEq)]\n    pub struct Frame {\n        /// Seconds on a monotonic clock.\n        pub timestamp: f64,\n        pub viewport: Viewport,\n        pub events: Vec<GameEvent>,\n    }".lines().map(str::to_owned));
    let fields = object(&protocol["catalog"])
        .values()
        .map(|list| Member {
            name: s(&list["list"]).to_owned(),
            ty: "Vec<String>".to_owned(),
            copy: false,
            doc: text(list, "doc"),
        })
        .collect::<Vec<_>>();
    out.extend(structure(
        "Catalog",
        Some("The game's assets, by the IDs the packet uses: each list's index is its ID."),
        &fields,
    ));
    out.push("    impl Catalog {".to_owned());
    for (name, list) in object(&protocol["catalog"]) {
        out.push(format!(
            "        pub fn {name}(&self, index: u32) -> &str {{"
        ));
        out.push(format!(
            "            &self.{}[index as usize]",
            s(&list["list"])
        ));
        out.push("        }".to_owned());
    }
    out.push("    }".to_owned());
    let mut emitted = HashSet::new();
    for entry in &model.sections {
        if let Some(layout) = &entry.fixed {
            emit_record(
                &mut out,
                model,
                &layout.record,
                &layout.fields,
                None,
                &mut emitted,
            );
        }
        if kind(&entry.section) == "variable" {
            if let Some(layout) = &entry.item {
                emit_record(
                    &mut out,
                    model,
                    &layout.record,
                    &layout.fields,
                    None,
                    &mut emitted,
                );
            }
            let record = &entry.section["record"];
            let rest = &record["rest"];
            let mut ty = "Vec<f32>".to_owned();
            if kind(rest) == "repeat" {
                ty = format!("Vec<{}>", s(&rest["record"]["name"]));
            }
            emit_record(
                &mut out,
                model,
                record,
                &entry.head,
                Some(Member {
                    name: s(&rest["name"]).to_owned(),
                    ty,
                    copy: false,
                    doc: text(rest, "doc"),
                }),
                &mut emitted,
            );
        }
    }
    for (group, value) in object(&protocol["render"]["groups"]) {
        let fields = model
            .header
            .iter()
            .filter(|entry| entry.group.as_ref() == Some(group))
            .map(|entry| member(model, &entry.laid))
            .collect::<Vec<_>>();
        out.extend(structure(s(&value["type"]), value["doc"].as_str(), &fields));
    }
    let mut frame = Vec::new();
    let mut groups = HashSet::new();
    for entry in &model.header {
        if matches!(kind(&entry.laid.field), "magic" | "schema" | "size") {
            continue;
        }
        if let Some(group) = &entry.group {
            if !groups.insert(group) {
                continue;
            }
            let value = &protocol["render"]["groups"][group];
            let mut ty = s(&value["type"]).to_owned();
            if model.header.iter().any(|other| {
                other.group.as_ref() == Some(group) && kind(&other.laid.field) == "present"
            }) {
                ty = format!("Option<{ty}>");
            }
            frame.push(Member {
                name: group.clone(),
                ty,
                copy: true,
                doc: text(value, "doc"),
            });
        } else {
            frame.push(member(model, &entry.laid));
        }
    }
    for entry in &model.sections {
        let section = &entry.section;
        if entry.owner.is_some()
            || kind(section) == "trailing"
            || (kind(section) == "lanes" && section["apart"] == true)
        {
            continue;
        }
        let ty = match kind(section) {
            "lanes" if section["nullable"] == true => "Option<Vec<f32>>".to_owned(),
            "lanes" => "Vec<f32>".to_owned(),
            _ => format!("Vec<{}>", s(&section["record"]["name"])),
        };
        frame.push(Member {
            name: s(&section["name"]).to_owned(),
            ty,
            copy: false,
            doc: text(section, "doc"),
        });
    }
    out.extend(structure(
        "RenderFrame",
        protocol["render"]["doc"].as_str(),
        &frame,
    ));
    out.push("}".to_owned());
    out
}
fn add_constant(
    out: &mut Vec<String>,
    seen: &mut HashSet<String>,
    name: String,
    value: usize,
    about: String,
) {
    if !seen.insert(name.clone()) {
        return;
    }
    out.extend(doc(Some(&format!("Lanes: {about}.")), "    "));
    out.push(format!("    pub const {name}: usize = {value};"));
}
fn offsets(out: &mut Vec<String>, name: &str, fields: &[(String, usize)]) {
    out.push(format!("        pub mod {} {{", snake(name)));
    for (name, offset) in fields {
        out.push(format!(
            "            pub const {}: usize = {offset};",
            upper(name)
        ));
    }
    out.push("        }".to_owned());
}
fn constants(model: &Model) -> Vec<String> {
    let mut out = Vec::new();
    for limit in array(&model.protocol["limits"]) {
        out.extend(doc(limit["doc"].as_str(), "    "));
        out.push(format!(
            "    pub const {}: usize = {};",
            s(&limit["limit"]),
            limit["value"]
        ));
    }
    out.push(format!(
        "    pub const PACKET_MAGIC: f32 = 0x{:x} as f32;",
        model.magic
    ));
    let header_fields = model
        .header
        .iter()
        .map(|entry| {
            let mut field = entry.laid.clone();
            field.name.clone_from(&entry.key);
            field
        })
        .collect::<Vec<_>>();
    out.extend(doc(
        Some(&format!(
            "The render packet header: {}.",
            describe(&header_fields)
        )),
        "    ",
    ));
    out.push(format!(
        "    pub const PACKET_HEADER_FLOATS: usize = {};",
        model.header_lanes
    ));
    out.extend([
        "    /// The input packet header: elapsed, width, height, event count and state count."
            .to_owned(),
        "    pub const FRAME_HEADER_FLOATS: usize = 5;".to_owned(),
        format!(
            "    /// An event: its tag, then {}.",
            array(&model.protocol["input"]["lanes"])
                .iter()
                .map(s)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        format!(
            "    pub const EVENT_FLOATS: usize = {};",
            model.event_lanes()
        ),
    ]);
    let mut seen = HashSet::new();
    for entry in &model.sections {
        if let Some(layout) = &entry.fixed {
            add_constant(
                &mut out,
                &mut seen,
                floats(&layout.record),
                layout.lanes,
                describe(&layout.fields),
            );
        }
        if kind(&entry.section) == "variable" {
            let rest = &entry.section["record"]["rest"];
            let mut about = describe(&entry.head);
            if kind(rest) == "repeat" {
                if !entry.head.is_empty() {
                    about.push_str(", ");
                }
                about.push_str(&format!("{} count", s(&rest["name"])));
            }
            about.push_str(&format!(
                ", before a {}'s {}",
                singular(s(&entry.section["name"])),
                s(&rest["name"])
            ));
            add_constant(
                &mut out,
                &mut seen,
                floats(&entry.section["record"]),
                entry.head_lanes,
                about,
            );
            if let Some(layout) = &entry.item {
                add_constant(
                    &mut out,
                    &mut seen,
                    floats(&layout.record),
                    layout.lanes,
                    describe(&layout.fields),
                );
            }
        }
        if kind(&entry.section) == "lanes"
            && let Some(name) = entry.section["constant"].as_str()
        {
            add_constant(
                &mut out,
                &mut seen,
                format!("{name}_FLOATS"),
                n(&entry.section["max"]),
                format!("{} lanes", s(&entry.section["name"])),
            );
        }
    }
    out.extend("    /// Where each header lane and record field starts; an optional field's lanes\n    /// follow its count.\n    pub mod lanes {\n        pub mod header {".lines().map(str::to_owned));
    for entry in &model.header {
        out.push(format!(
            "            pub const {}: usize = {};",
            upper(&entry.key.replace('.', "_")),
            entry.laid.offset
        ));
    }
    out.push("        }".to_owned());
    for layout in &model.records {
        offsets(
            &mut out,
            s(&layout.record["name"]),
            &layout
                .fields
                .iter()
                .map(|field| (field.name.clone(), field.offset))
                .collect::<Vec<_>>(),
        );
    }
    for entry in &model.sections {
        if kind(&entry.section) != "variable" {
            continue;
        }
        let mut fields = entry
            .head
            .iter()
            .map(|field| (field.name.clone(), field.offset))
            .collect::<Vec<_>>();
        let rest = &entry.section["record"]["rest"];
        if kind(rest) == "repeat" {
            fields.push((s(&rest["name"]).to_owned(), entry.head_lanes - 1));
        }
        offsets(&mut out, s(&entry.section["record"]["name"]), &fields);
    }
    out.extend([
        "    }".to_owned(),
        "    /// Each event's tag.".to_owned(),
        "    pub mod event_tags {".to_owned(),
    ]);
    for event in model.events() {
        out.push(format!(
            "        pub const {}: u32 = {};",
            upper(s(&event["name"])),
            event["tag"]
        ));
    }
    out.push("    }".to_owned());
    out
}
fn lines(out: &mut Vec<String>, value: &str) {
    out.extend(value.split('\n').map(str::to_owned));
}
fn input(model: &Model) -> Vec<String> {
    let input = &model.protocol["input"];
    let state = &model.protocol["state"];
    let mut out = Vec::new();
    lines(
        &mut out,
        "    pub fn validate_state(state: &[f32]) -> Result<()> {\n        ensure!(",
    );
    out.push(format!(
        "            state.len() <= {} && state.iter().all(|value| value.is_finite()),",
        s(&state["max"]["limit"])
    ));
    out.push(format!("            {}", quote(&state["message"])));
    lines(
        &mut out,
        "        );\n        Ok(())\n    }\n\n    pub fn validate_schema(schema: f32) -> Result<u32> {\n        ensure!(",
    );
    out.push(format!(
        "            schema.fract() == 0.0 && (1.0..={} as f32).contains(&schema),",
        s(&state["max"]["limit"])
    ));
    out.push(format!("            {}", quote(&state["schemaMessage"])));
    lines(
        &mut out,
        "        );\n        Ok(schema as u32)\n    }\n\n    /// The code a key travels as: named keys by their browser names, other single\n    /// UTF-16 code units as themselves (as `charCodeAt(0)` reads them), lowercase\n    /// ASCII letters as their capitals, and anything else as 0.\n    pub fn key_code(key: &str) -> f32 {\n        let named = match key {",
    );
    for (key, code) in object(&input["keys"]) {
        out.push(format!(
            "            {} => 0x{:x},",
            quote(&Value::String(key.clone())),
            n(code)
        ));
    }
    lines(
        &mut out,
        "            _ => 0,\n        };\n        if named != 0 {\n            return named as f32;\n        }\n        let mut units = key.encode_utf16();\n        match (units.next(), units.next()) {\n            (Some(code), None) => {\n                if (97..=122).contains(&code) {\n                    (code - 32) as f32\n                } else {\n                    code as f32\n                }\n            }\n            _ => 0.0,\n        }\n    }\n",
    );
    out.push(format!(
        "    fn event_record(event: &GameEvent) -> [f32; {}] {{",
        model.event_lanes()
    ));
    out.push("        match event {".to_owned());
    for event in model.events() {
        for host in hosts(event) {
            let mut lanes = vec!["0.0".to_owned(); model.event_lanes()];
            lanes[0] = format!("{}.0", event["tag"]);
            let name = s(&host["name"]);
            let pattern;
            if host["fields"].is_object() {
                pattern = format!("GameEvent::{name} {{ .. }}");
            } else {
                let fields = event_fields(event);
                for (name, field) in &fields {
                    let lane = 1 + array(&input["lanes"])
                        .iter()
                        .position(|lane| lane == &field["lane"])
                        .unwrap();
                    lanes[lane] = match field["wire"].as_str() {
                        Some("key") => format!("key_code({name})"),
                        _ if field["rust"] == "f32" => format!("*{name}"),
                        _ => format!("*{name} as f32"),
                    };
                }
                if !event["value"].is_null() {
                    pattern = format!("GameEvent::{name}(value)");
                } else if !fields.is_empty() {
                    pattern = format!(
                        "GameEvent::{name} {{ {} }}",
                        fields
                            .iter()
                            .map(|(name, _)| name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                } else {
                    pattern = format!("GameEvent::{name}");
                }
            }
            out.push(format!("            {pattern} => [{}],", lanes.join(", ")));
        }
    }
    lines(&mut out, "        }\n    }\n");
    out.extend(doc(
        Some(&format!(
            "{} Unused event lanes are zero.",
            input["doc"].as_str().unwrap_or("")
        )),
        "    ",
    ));
    lines(
        &mut out,
        "    pub fn encode_frame(frame: &Frame, elapsed: f64, state: &[f32]) -> Result<Vec<f32>> {\n        validate_state(state)?;\n        ensure!(\n            frame.timestamp.is_finite()\n                && elapsed.is_finite()\n                && elapsed >= 0.0",
    );
    out.push(format!(
        "                && (1..=0x{:x}).contains(&frame.viewport.width)",
        n(&input["maxExtent"])
    ));
    out.push(format!(
        "                && (1..=0x{:x}).contains(&frame.viewport.height)",
        n(&input["maxExtent"])
    ));
    out.push(format!(
        "                && frame.events.len() <= {},",
        s(&input["maxEvents"]["limit"])
    ));
    out.push(format!(
        "            {}",
        quote(&input["messages"]["frame"])
    ));
    lines(
        &mut out,
        "        );\n        let mut packet = Vec::with_capacity(FRAME_HEADER_FLOATS + frame.events.len() * EVENT_FLOATS + state.len());\n        packet.extend_from_slice(&[\n            elapsed as f32,\n            frame.viewport.width as f32,\n            frame.viewport.height as f32,\n            frame.events.len() as f32,\n            state.len() as f32,\n        ]);\n        for event in &frame.events {\n            packet.extend_from_slice(&event_record(event));\n        }\n        packet.extend_from_slice(state);",
    );
    out.push(format!(
        "        ensure!(packet.iter().all(|value| value.is_finite()), {});",
        quote(&input["messages"]["nonfinite"])
    ));
    lines(&mut out, "        Ok(packet)\n    }");
    out
}
fn at(base: &str, offset: usize) -> String {
    if base.is_empty() {
        return offset.to_string();
    }
    if offset == 0 {
        return base.to_owned();
    }
    format!("{base} + {offset}")
}
fn convert(field: &Value, local: &str) -> String {
    if kind(field) == "count" {
        return format!("{local} as u32");
    }
    local.to_owned()
}
fn decode_field(model: &Model, base: &str, laid: &Laid, reader: &str) -> (Vec<String>, String) {
    let field = &laid.field;
    let index = at(base, laid.offset);
    let name = &laid.name;
    let lane = |offset| format!("{reader}packet[{}]", at(base, laid.offset + offset));
    match kind(field) {
        "f32" => (vec![], lane(0)),
        "vector" => (
            vec![],
            format!(
                "{} {{ {} }}",
                s(&field["type"]["name"]),
                array(&field["type"]["components"])
                    .iter()
                    .enumerate()
                    .map(|(offset, component)| format!("{}: {}", s(component), lane(offset)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ),
        "matrix" => (vec![], format!("{reader}matrix({index})")),
        "lanes" => (
            vec![],
            format!(
                "{reader}packet[{index}..{}].try_into().unwrap()",
                at(base, laid.offset + n(&field["count"]))
            ),
        ),
        "count" => (
            vec![],
            format!("{reader}count({index}, {})?", bound(&field["max"])),
        ),
        "flag" => (vec![], format!("{reader}flag({index})?")),
        "reference" => (vec![], format!("{reader}{}_at({index})?", s(&field["to"]))),
        "choice" => {
            let mut value = array(&field["values"])
                .iter()
                .map(|value| format!("if {} == {value}.0 {{ {value} }} else ", lane(0)))
                .collect::<Vec<_>>()
                .join("");
            value.push_str(&format!("{{ {} }}", field["otherwise"]));
            (vec![], value)
        }
        "optional" => {
            let count = n(&field["count"]);
            (
                vec![
                    format!("let {name}_count = {reader}count({index}, {count})?;"),
                    format!(
                        "ensure!({name}_count == 0 || {name}_count == {count}, {});",
                        quote(&field["message"])
                    ),
                ],
                format!(
                    "({name}_count == {count}).then(|| {reader}packet[{}..{}].try_into().unwrap())",
                    at(base, laid.offset + 1),
                    at(base, laid.offset + 1 + count)
                ),
            )
        }
        "children" => {
            let section = s(&field["section"]);
            let record = &model.section(section).fixed.as_ref().unwrap().record;
            (
                vec![
                    format!(
                        "let {name}_count = {reader}count({index}, {})?;",
                        bound(&field["max"])
                    ),
                    format!("let {name}_first = {reader}{section}_at;"),
                    format!("{reader}{section}_at += {name}_count * {};", floats(record)),
                    format!(
                        "ensure!({reader}{section}_at <= {reader}{section}_end, {});",
                        quote(&field["over"])
                    ),
                    format!("let mut {name} = Vec::with_capacity({name}_count);"),
                    format!("for item in 0..{name}_count {{"),
                    format!(
                        "    {name}.push({reader}{}({name}_first + item * {})?);",
                        decoder(record),
                        floats(record)
                    ),
                    "}".to_owned(),
                ],
                name.clone(),
            )
        }
        "tail" => {
            let section = s(&field["section"]);
            (
                vec![
                    "// Counted with the packet's length.".to_owned(),
                    format!(
                        "let {name}_lanes = {} as usize * {};",
                        lane(0),
                        field["unit"]
                    ),
                    format!("let {name}_first = {reader}{section}_at;"),
                    format!("{reader}{section}_at += {name}_lanes;"),
                ],
                format!(
                    "({name}_lanes != 0).then(|| {reader}packet[{name}_first..{name}_first + {name}_lanes].to_vec())"
                ),
            )
        }
        "present" => (
            vec![],
            format!(
                "{reader}count({index}, {reader}catalog.{}.len())?",
                s(&model.protocol["catalog"][s(&field["to"])]["list"])
            ),
        ),
        _ => unreachable!(),
    }
}
fn render(model: &Model) -> Vec<String> {
    let protocol = &model.protocol;
    let messages = &protocol["render"]["messages"];
    let cursors = model
        .sections
        .iter()
        .filter(|entry| entry.owner.is_some() || kind(&entry.section) == "trailing")
        .collect::<Vec<_>>();
    let mut out = Vec::new();
    lines(
        &mut out,
        "    /// A decoded render packet and the state lanes it carried.\n    pub struct Decoded {\n        pub state: Vec<f32>,\n        pub render: RenderFrame,\n    }\n\n    /// The packet, the catalog, and where the next owned record or tail starts.\n    struct Reader<'a> {\n        packet: &'a [f32],\n        catalog: &'a Catalog,",
    );
    for entry in &cursors {
        let name = s(&entry.section["name"]);
        out.push(format!("        {name}_at: usize,"));
        if entry.owner.is_some() {
            out.push(format!("        {name}_end: usize,"));
        }
    }
    lines(
        &mut out,
        "    }\n\n    impl Reader<'_> {\n        #[inline]\n        fn count(&self, index: usize, maximum: usize) -> Result<usize> {\n            let value = self.packet[index];\n            if value.fract() != 0.0 || value < 0.0 || value > maximum as f32 {",
    );
    out.push(format!(
        "                bail!({});",
        quote(&messages["count"])
    ));
    lines(
        &mut out,
        "            }\n            Ok(value as usize)\n        }\n\n        #[inline]\n        fn flag(&self, index: usize) -> Result<bool> {\n            Ok(self.count(index, 1)? == 1)\n        }\n\n        #[inline]\n        fn matrix(&self, index: usize) -> Matrix {\n            self.packet[index..index + 16].try_into().unwrap()\n        }",
    );
    for (name, list) in object(&protocol["catalog"]) {
        out.extend([
            String::new(),
            "        #[inline]".to_owned(),
            format!("        fn {name}_at(&self, index: usize) -> Result<u32> {{"),
        ]);
        if !list["empty"].is_null() {
            out.push(format!(
                "            ensure!(!self.catalog.{}.is_empty(), {});",
                s(&list["list"]),
                quote(&list["empty"])
            ));
            out.push(format!(
                "            Ok(self.count(index, self.catalog.{}.len() - 1)? as u32)",
                s(&list["list"])
            ));
        } else {
            out.push(format!("            Ok(self.count(index, self.catalog.{}.len().saturating_sub(1))? as u32)", s(&list["list"])));
        }
        out.push("        }".to_owned());
    }
    let mut emitted = HashSet::new();
    for entry in &model.sections {
        for layout in entry.fixed.iter().chain(entry.item.iter()) {
            if !emitted.insert(s(&layout.record["name"])) {
                continue;
            }
            out.push(String::new());
            out.push(format!(
                "        fn {}(&mut self, lane: usize) -> Result<{}> {{",
                decoder(&layout.record),
                s(&layout.record["name"])
            ));
            let mut fields = Vec::new();
            for laid in &layout.fields {
                let (statements, value) = decode_field(model, "lane", laid, "self.");
                out.extend(statements.iter().map(|line| format!("            {line}")));
                if value == laid.name {
                    fields.push(laid.name.clone());
                } else {
                    out.push(format!("            let {} = {value};", laid.name));
                    let value = convert(&laid.field, &laid.name);
                    if value == laid.name {
                        fields.push(laid.name.clone());
                    } else {
                        fields.push(format!("{}: {value}", laid.name));
                    }
                }
            }
            out.push(format!(
                "            Ok({} {{ {} }})",
                s(&layout.record["name"]),
                fields.join(", ")
            ));
            out.push("        }".to_owned());
        }
    }
    out.extend(["    }".to_owned(), String::new()]);
    out.extend(doc(
        Some(&format!(
            "Decodes a render packet. {} With several faults, it reports the first in lane order.",
            protocol["render"]["doc"].as_str().unwrap_or("")
        )),
        "    ",
    ));
    lines(
        &mut out,
        "    pub fn decode_frame(packet: &[f32], schema: u32, catalog: &Catalog) -> Result<Decoded> {\n        ensure!(\n            packet.len() >= PACKET_HEADER_FLOATS && packet[0] == PACKET_MAGIC && packet[1] == schema as f32,",
    );
    out.push(format!("            {}", quote(&messages["header"])));
    lines(
        &mut out,
        "        );\n        let mut reader = Reader {\n            packet,\n            catalog,",
    );
    for entry in &cursors {
        let name = s(&entry.section["name"]);
        out.push(format!("            {name}_at: 0,"));
        if entry.owner.is_some() {
            out.push(format!("            {name}_end: 0,"));
        }
    }
    out.extend([
        "        };".to_owned(),
        "        // The sections' sizes.".to_owned(),
    ]);
    for entry in &model.header {
        let lane = &entry.laid.field;
        if kind(lane) != "size" {
            continue;
        }
        let section = &model.section(s(&lane["section"])).section;
        let name = s(&lane["section"]);
        let of = s(&lane["of"]);
        let local = format!("{name}_{of}");
        let maximum = match (of, kind(section)) {
            ("count", _) => bound(&section["max"]),
            (_, "variable") => bound(&section["maxLanes"]),
            (_, "lanes") if section["constant"].is_string() => {
                format!("{}_FLOATS", s(&section["constant"]))
            }
            _ => bound(&section["max"]),
        };
        out.push(format!(
            "        let {local} = reader.count({}, {maximum})?;",
            entry.laid.offset
        ));
        if kind(section) == "lanes" && !section["exactly"].is_null() {
            let conditions = array(&section["exactly"]["counts"])
                .iter()
                .map(|count| format!("{local} == {count}"))
                .collect::<Vec<_>>()
                .join(" || ");
            out.push(format!(
                "        ensure!({conditions}, {});",
                quote(&section["exactly"]["message"])
            ));
        }
    }
    out.push("        // Where the sections lie.".to_owned());
    let mut previous = "PACKET_HEADER_FLOATS".to_owned();
    let mut trailing = None;
    for entry in &model.sections {
        let section = &entry.section;
        let name = s(&section["name"]);
        out.push(format!("        let {name}_start = {previous};"));
        if kind(section) == "trailing" {
            trailing = Some(name);
            continue;
        }
        let mut length = format!("{name}_lanes");
        if kind(section) == "records" {
            length = format!("{name}_count * {}", floats(&section["record"]));
        }
        out.push(format!("        let {name}_end = {name}_start + {length};"));
        previous = format!("{name}_end");
    }
    out.push(format!(
        "        let truncated = {};",
        quote(&messages["length"])
    ));
    out.push(format!("        ensure!(packet.len() >= {previous} && packet.iter().all(|value| value.is_finite()), truncated);"));
    let mut packet_end = previous;
    if let Some(trailing) = trailing {
        out.push(format!(
            "        // What the records' tails claim of {trailing}, in order."
        ));
        out.push(format!("        let mut {trailing}_lanes = 0;"));
        if let Some((_, owners)) = model.tails.iter().find(|(name, _)| name == trailing) {
            for owner in owners {
                out.push(format!(
                    "        for index in 0..{}_count {{",
                    owner.section
                ));
                out.push(format!("            {trailing}_lanes += reader.count({}_start + index * {} + {}, {})? * {};", owner.section, floats(&owner.layout.record), owner.field.offset, bound(&owner.field.field["max"]), owner.field.field["unit"]));
                out.push("        }".to_owned());
            }
        }
        out.push(format!(
            "        let {trailing}_end = {trailing}_start + {trailing}_lanes;"
        ));
        packet_end = format!("{trailing}_end");
    }
    out.push(format!(
        "        ensure!(packet.len() == {packet_end}, truncated);"
    ));
    for entry in &cursors {
        let name = s(&entry.section["name"]);
        out.push(format!("        reader.{name}_at = {name}_start;"));
        if entry.owner.is_some() {
            out.push(format!("        reader.{name}_end = {name}_end;"));
        }
    }
    out.push("        // The header's values.".to_owned());
    let mut header_values = HashMap::new();
    for entry in &model.header {
        let lane = &entry.laid.field;
        if matches!(kind(lane), "magic" | "schema" | "size") {
            continue;
        }
        let local = entry.key.replace('.', "_");
        let (statements, value) = decode_field(model, "", &entry.laid, "reader.");
        out.extend(statements.iter().map(|line| format!("        {line}")));
        out.push(format!("        let {local} = {value};"));
        header_values.insert(entry.key.as_str(), convert(lane, &local));
    }
    out.push("        // The sections.".to_owned());
    for entry in &model.sections {
        let section = &entry.section;
        let name = s(&section["name"]);
        match kind(section) {
            "lanes" => {
                let mut slice = format!("packet[{name}_start..{name}_end].to_vec()");
                if section["nullable"] == true {
                    slice = format!("({name}_lanes != 0).then(|| {slice})");
                }
                out.push(format!("        let {name} = {slice};"));
            }
            "records" => {
                if entry.owner.is_some() {
                    continue;
                }
                let layout = entry.fixed.as_ref().unwrap();
                out.push(format!(
                    "        let mut {name} = Vec::with_capacity({name}_count);"
                ));
                out.push(format!("        for index in 0..{name}_count {{"));
                out.push(format!(
                    "            {name}.push(reader.{}({name}_start + index * {})?);",
                    decoder(&layout.record),
                    floats(&layout.record)
                ));
                out.push("        }".to_owned());
                for field in &layout.fields {
                    if kind(&field.field) == "children" {
                        let child = s(&field.field["section"]);
                        out.push(format!(
                            "        ensure!(reader.{child}_at == {child}_end, {});",
                            quote(&field.field["under"])
                        ));
                    }
                }
            }
            "variable" => {
                let record = &section["record"];
                let rest = &record["rest"];
                let rest_name = s(&rest["name"]);
                let local = record_local(name);
                let lane = format!("{local}_lane");
                let after = format!("{local}_end");
                let first = format!("{rest_name}_first");
                out.push(format!(
                    "        let mut {name} = Vec::with_capacity({name}_count);"
                ));
                out.push(format!("        let mut {lane} = {name}_start;"));
                out.push(format!("        for _ in 0..{name}_count {{"));
                out.push(format!(
                    "            ensure!({lane} + {} <= {name}_end, {});",
                    floats(record),
                    quote(&section["messages"]["truncated"])
                ));
                let mut fields = Vec::new();
                for laid in &entry.head {
                    let (statements, value) = decode_field(model, &lane, laid, "reader.");
                    out.extend(statements.iter().map(|line| format!("            {line}")));
                    out.push(format!("            let {} = {value};", laid.name));
                    let value = convert(&laid.field, &laid.name);
                    if value == laid.name {
                        fields.push(laid.name.clone());
                    } else {
                        fields.push(format!("{}: {value}", laid.name));
                    }
                }
                let rest_lanes;
                let mut condition = String::new();
                if kind(rest) == "repeat" {
                    out.push(format!(
                        "            let {rest_name}_count = reader.count({}, {})?;",
                        at(&lane, entry.head_lanes - 1),
                        bound(&rest["max"])
                    ));
                    rest_lanes = format!("{rest_name}_count * {}", floats(&rest["record"]));
                    condition = format!("{rest_name}_count >= {} && ", rest["min"]);
                } else {
                    rest_lanes = array(&rest["of"])
                        .iter()
                        .map(s)
                        .collect::<Vec<_>>()
                        .join(" * ");
                }
                out.push(format!(
                    "            let {first} = {lane} + {};",
                    floats(record)
                ));
                out.push(format!("            let {after} = {first} + {rest_lanes};"));
                out.push(format!(
                    "            ensure!({condition}{after} <= {name}_end, {});",
                    quote(&section["messages"]["invalid"])
                ));
                if kind(rest) == "repeat" {
                    let item = entry.item.as_ref().unwrap();
                    out.push(format!(
                        "            let mut {rest_name} = Vec::with_capacity({rest_name}_count);"
                    ));
                    out.push(format!("            for item in 0..{rest_name}_count {{"));
                    out.push(format!(
                        "                {rest_name}.push(reader.{}({first} + item * {})?);",
                        decoder(&item.record),
                        floats(&item.record)
                    ));
                    out.push("            }".to_owned());
                } else {
                    out.push(format!(
                        "            let {rest_name} = packet[{first}..{after}].to_vec();"
                    ));
                }
                fields.push(rest_name.to_owned());
                out.push(format!(
                    "            {name}.push({} {{ {} }});",
                    s(&record["name"]),
                    fields.join(", ")
                ));
                out.push(format!("            {lane} = {after};"));
                out.push("        }".to_owned());
                out.push(format!(
                    "        ensure!({lane} == {name}_end, {});",
                    quote(&section["messages"]["mismatch"])
                ));
            }
            _ => {}
        }
    }
    out.push("        let render = RenderFrame {".to_owned());
    let mut groups = HashSet::new();
    for entry in &model.header {
        let lane = &entry.laid.field;
        if matches!(kind(lane), "magic" | "schema" | "size") {
            continue;
        }
        if let Some(group) = &entry.group {
            if !groups.insert(group) {
                continue;
            }
            let members = model
                .header
                .iter()
                .filter(|entry| entry.group.as_ref() == Some(group))
                .collect::<Vec<_>>();
            let values = members
                .iter()
                .map(|member| {
                    let value = &header_values[member.key.as_str()];
                    if kind(&member.laid.field) == "present" {
                        return format!(
                            "{}: ({} - 1) as u32",
                            member.laid.name,
                            value.trim_end_matches(" as u32")
                        );
                    }
                    format!("{}: {value}", member.laid.name)
                })
                .collect::<Vec<_>>()
                .join(", ");
            let mut object = format!(
                "{} {{ {values} }}",
                s(&protocol["render"]["groups"][group]["type"])
            );
            if let Some(presence) = members
                .iter()
                .find(|member| kind(&member.laid.field) == "present")
            {
                object = format!(
                    "({} != 0).then(|| {object})",
                    header_values[presence.key.as_str()].trim_end_matches(" as u32")
                );
            }
            out.push(format!("            {group}: {object},"));
        } else {
            let value = &header_values[entry.key.as_str()];
            if *value == entry.laid.name {
                out.push(format!("            {value},"));
            } else {
                out.push(format!("            {}: {value},", entry.laid.name));
            }
        }
    }
    let mut state = "state";
    for entry in &model.sections {
        let section = &entry.section;
        if kind(section) == "lanes" && section["apart"] == true {
            state = s(&section["name"]);
            continue;
        }
        if entry.owner.is_some() || kind(section) == "trailing" {
            continue;
        }
        out.push(format!("            {},", s(&section["name"])));
    }
    out.push("        };".to_owned());
    if state == "state" {
        out.push("        Ok(Decoded { state, render })".to_owned());
    } else {
        out.push(format!("        Ok(Decoded {{ state: {state}, render }})"));
    }
    out.push("    }".to_owned());
    out
}
pub fn emit(model: &Model, header: &str) -> String {
    let mut out = header
        .lines()
        .map(|line| format!("// {line}").trim_end().to_owned())
        .collect::<Vec<_>>();
    lines(
        &mut out,
        "//! The packets crossing between the host and a game: their types,\n//! constants, the render packet decoder and the input packet encoder.\n",
    );
    out.extend(types(model));
    out.push(String::new());
    lines(
        &mut out,
        "/// The packets' layout: constants, the input packet encoder and the render\n/// packet decoder.\npub mod codec {\n    use anyhow::{Result, bail, ensure};\n\n    use super::types::*;\n",
    );
    out.extend(constants(model));
    out.push(String::new());
    out.extend(input(model));
    out.push(String::new());
    out.extend(render(model));
    out.extend(["}".to_owned(), String::new()]);
    out.join("\n")
}
