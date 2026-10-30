//! The Blot a game needs besides its packets' codecs: the program loop pen
//! runs it through (`program.blot`), a frame that draws nothing for it to
//! start from (`frame.blot`) and the checks its tests report to `pen
//! test-blot` (`check.blot`), so that a game builds on pen's bindings
//! alone.
use crate::model::*;
use serde_json::Value;

/// What a game program says first (pen-runtime's `HANDSHAKE_MAGIC`), and
/// the version of the program protocol it speaks (`PROGRAM_VERSION`).
pub const PROGRAM_MAGIC: u32 = 4_670_544;
pub const PROGRAM_VERSION: u32 = 2;

const PROGRAM: &str = include_str!("program.blot");
const CHECK: &str = include_str!("check.blot");

/// The session driver's library uses operation codes generated from the schema.
pub fn testing(header: &[String]) -> String {
    under(header, include_str!("testing.blot"))
}

fn under(header: &[String], body: &str) -> String {
    let mut out = header.to_vec();
    out.extend(body.lines().map(str::to_owned));
    out.push(String::new());
    out.join("\n")
}

/// `program.blot` under `header`'s lines.
pub fn program(header: &[String]) -> String {
    let body =
        PROGRAM.replace("${MAGIC}", &PROGRAM_MAGIC.to_string()).replace("${VERSION}", &PROGRAM_VERSION.to_string());
    under(header, &body)
}

/// `check.blot`, the checks of `pen test-blot`'s report format, under
/// `header`'s lines.
pub fn check(header: &[String]) -> String {
    under(header, CHECK)
}

/// A Blot F32 literal; a negative one in parentheses.
fn float(value: f64) -> String {
    let text = if value.fract() == 0.0 { format!("{value:.1}") } else { value.to_string() };
    if value < 0.0 { format!("({text})") } else { text }
}

/// The value a blank frame gives a header lane: its `default`, else nothing
/// (false, zero, the identity matrix, the choice's otherwise).
fn blank(field: &Value) -> String {
    let default = &field["default"];
    let number = |index: Option<usize>| {
        let value = match index {
            Some(index) => &default[index],
            None => default,
        };
        float(value.as_f64().unwrap_or(0.0))
    };
    match kind(field) {
        "flag" if default == true => "true".to_owned(),
        "flag" => "false".to_owned(),
        "f32" => number(None),
        "count" => default.as_u64().unwrap_or(0).to_string(),
        "choice" => field["otherwise"].to_string(),
        "matrix" => "identity".to_owned(),
        "vector" => {
            let components = array(&field["type"]["components"])
                .iter()
                .enumerate()
                .map(|(index, component)| format!("{}: {}", s(component), number(Some(index))))
                .collect::<Vec<_>>()
                .join(", ");
            format!("$types.{} {{ {components} }}", s(&field["type"]["name"]))
        }
        _ => "0".to_owned(),
    }
}

/// `frame.blot`: the identity matrix, and `blank`, a frame that draws,
/// plays and asks nothing, with the header's defaults.
pub fn frame(model: &Model, header: &[String]) -> String {
    let mut out = header.to_vec();
    out.extend(
        "//\n// A frame to start from: one that draws, plays and asks nothing, lit from\n// above. A game sets what it needs (`frame.draws := #[...]`) and leaves the\n// rest.\nimport * as types from \"./types\"\n\nconst identity = $types.Mat4 {\n  x: $types.Vec4 { x: 1.0, y: 0.0, z: 0.0, w: 0.0 },\n  y: $types.Vec4 { x: 0.0, y: 1.0, z: 0.0, w: 0.0 },\n  z: $types.Vec4 { x: 0.0, y: 0.0, z: 1.0, w: 0.0 },\n  w: $types.Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },\n}\n\n// A frame of state `schema` that draws nothing.\nconst blank = func (schema: U32) => return $types.RenderFrame {"
            .lines()
            .map(str::to_owned),
    );
    let mut groups: Vec<&str> = Vec::new();
    for entry in &model.header {
        let lane = &entry.laid.field;
        match kind(lane) {
            "magic" | "size" => {}
            "schema" => out.push("  schema,".to_owned()),
            _ => match &entry.group {
                Some(group) if groups.contains(&group.as_str()) => {}
                Some(group) => {
                    groups.push(group);
                    let ty = s(&model.protocol["render"]["groups"][group]["type"]);
                    out.push(format!("  {group}: $types.{ty} {{"));
                    for member in model.header.iter().filter(|member| member.group.as_ref() == Some(group)) {
                        out.push(format!("    {}: {},", member.laid.name, blank(&member.laid.field)));
                    }
                    out.push("  },".to_owned());
                }
                None => out.push(format!("  {}: {},", entry.laid.name, blank(lane))),
            },
        }
    }
    // A header contains fixed data only; source groups are separate borrowed
    // views for encode_render_views. Never allocate an owning RenderFrame.
    let header_start = out.iter().position(|line| line.starts_with("const blank =")).unwrap();
    let mut header = out[header_start..].to_vec();
    header[0] = "const header = func (schema: U32): types.RenderHeader => return {".to_owned();
    header.push("}".to_owned());
    for entry in &model.sections {
        if kind(&entry.section) == "trailing" || entry.owner.is_some() {
            continue;
        }
        out.push(format!("  {}: #[],", s(&entry.section["name"])));
    }
    out.push("}".to_owned());
    out.push(String::new());
    out.extend(header);
    out.push(String::new());
    out.join("\n")
}
