use anyhow::{Result, bail, ensure};
use serde_json::Value;
use std::collections::HashSet;

pub fn s(value: &Value) -> &str {
    value.as_str().unwrap()
}
pub fn n(value: &Value) -> usize {
    value.as_u64().unwrap() as usize
}
pub fn array(value: &Value) -> &[Value] {
    value.as_array().unwrap()
}
pub fn object(value: &Value) -> &serde_json::Map<String, Value> {
    value.as_object().unwrap()
}
pub fn kind(value: &Value) -> &str {
    s(&value["kind"])
}
pub fn text(value: &Value, key: &str) -> Option<String> {
    value[key].as_str().map(str::to_owned)
}
pub fn quote(value: &Value) -> String {
    serde_json::to_string(value).unwrap()
}
pub fn snake(name: &str) -> String {
    let mut result = String::new();
    let mut previous = ' ';
    for c in name.chars() {
        if c.is_ascii_uppercase() && (previous.is_ascii_lowercase() || previous.is_ascii_digit()) {
            result.push('_');
        }
        result.extend(c.to_lowercase());
        previous = c;
    }
    result
}
pub fn upper(name: &str) -> String {
    snake(name).to_uppercase()
}
pub fn singular(name: &str) -> String {
    if let Some(prefix) = name.strip_suffix("ies") {
        return format!("{prefix}y");
    }
    if let Some(prefix) = name.strip_suffix('s') {
        return prefix.to_owned();
    }
    name.to_owned()
}
pub fn record_local(name: &str) -> String {
    let one = singular(name);
    if one == name {
        return format!("{name}_record");
    }
    one
}
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + word.len() > width {
            lines.push(line);
            line = String::new();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}
pub fn bound(value: &Value) -> String {
    if value.is_object() {
        return s(&value["limit"]).to_owned();
    }
    if let Some(text) = value.as_str() {
        return text.to_owned();
    }
    value.to_string()
}
pub fn floats(record: &Value) -> String {
    format!("{}_FLOATS", s(&record["constant"]))
}
pub fn decoder(record: &Value) -> String {
    format!("decode_{}", snake(s(&record["name"])))
}
pub fn encoder(record: &Value) -> String {
    format!("encode_{}", snake(s(&record["name"])))
}

#[derive(Clone)]
pub struct Laid {
    pub name: String,
    pub offset: usize,
    pub lanes: usize,
    pub field: Value,
}
#[derive(Clone)]
pub struct Layout {
    pub record: Value,
    pub fields: Vec<Laid>,
    pub lanes: usize,
}
pub struct Header {
    pub key: String,
    pub group: Option<String>,
    pub laid: Laid,
}
pub struct Section {
    pub section: Value,
    pub fixed: Option<Layout>,
    pub head: Vec<Laid>,
    pub head_lanes: usize,
    pub item: Option<Layout>,
    pub owner: Option<(String, String)>,
}
pub struct Tail {
    pub section: String,
    pub layout: Layout,
    pub field: Laid,
}
pub struct Model {
    pub protocol: Value,
    pub header: Vec<Header>,
    pub header_lanes: usize,
    pub magic: usize,
    pub sections: Vec<Section>,
    pub records: Vec<Layout>,
    pub tails: Vec<(String, Vec<Tail>)>,
}
impl Model {
    pub fn section(&self, name: &str) -> &Section {
        self.sections
            .iter()
            .find(|entry| s(&entry.section["name"]) == name)
            .unwrap()
    }
    pub fn events(&self) -> &[Value] {
        array(&self.protocol["input"]["events"])
    }
    pub fn event_lanes(&self) -> usize {
        1 + array(&self.protocol["input"]["lanes"]).len()
    }
}
pub fn event_fields(event: &Value) -> Vec<(String, Value)> {
    if !event["value"].is_null() {
        return vec![("value".to_owned(), event["value"].clone())];
    }
    if let Some(fields) = event["fields"].as_object() {
        return fields
            .iter()
            .map(|(name, field)| (name.clone(), field.clone()))
            .collect();
    }
    Vec::new()
}
pub fn hosts(event: &Value) -> Vec<Value> {
    if let Some(hosts) = event["hosts"].as_array() {
        return hosts.clone();
    }
    vec![serde_json::json!({"name": event["name"], "doc": event["doc"]})]
}
pub fn lanes(field: &Value) -> usize {
    match kind(field) {
        "vector" => array(&field["type"]["components"]).len(),
        "matrix" => 16,
        "lanes" => n(&field["count"]),
        "optional" => 1 + n(&field["count"]),
        _ => 1,
    }
}
fn layout_fields(fields: &Value) -> Vec<Laid> {
    let mut offset = 0;
    object(fields)
        .iter()
        .map(|(name, field)| {
            let laid = Laid {
                name: name.clone(),
                offset,
                lanes: lanes(field),
                field: field.clone(),
            };
            offset += laid.lanes;
            laid
        })
        .collect()
}
fn layout(record: &Value) -> Layout {
    let fields = layout_fields(&record["fields"]);
    let lanes = fields.iter().map(|field| field.lanes).sum();
    Layout {
        record: record.clone(),
        fields,
        lanes,
    }
}
pub fn describe(fields: &[Laid]) -> String {
    fields
        .iter()
        .map(|laid| match kind(&laid.field) {
            "optional" => format!(
                "{} count and {}{}",
                laid.name, laid.name, laid.field["count"]
            ),
            "children" | "tail" => format!("{} count", laid.name),
            _ if laid.lanes == 1 => laid.name.clone(),
            _ => format!("{}{}", laid.name, laid.lanes),
        })
        .collect::<Vec<_>>()
        .join(", ")
}
fn string(value: &Value) -> Result<()> {
    ensure!(value.is_string(), "expected string: {value}");
    Ok(())
}
fn integer(value: &Value) -> Result<()> {
    ensure!(
        value.as_u64().is_some_and(|value| value <= u32::MAX as u64),
        "expected nonnegative 32-bit integer: {value}"
    );
    Ok(())
}
fn strings(value: &Value) -> Result<()> {
    ensure!(value.is_array(), "expected array: {value}");
    for entry in array(value) {
        string(entry)?;
    }
    Ok(())
}
fn fields(value: &Value, header: bool) -> Result<()> {
    ensure!(value.is_object(), "expected fields object: {value}");
    for (name, field) in object(value) {
        ensure!(name.split('.').all(symbol), "invalid field name {name}");
        lane(field, header)?;
    }
    Ok(())
}
fn validate_bound(value: &Value) -> Result<()> {
    if let Some(expression) = value.as_str() {
        let terms: Vec<_> = expression.split('-').map(str::trim).collect();
        ensure!(
            terms.len() <= 2 && terms.iter().all(|term| identifier(term)),
            "invalid bound {expression}"
        );
    } else if value.is_object() {
        string(&value["limit"])?;
        integer(&value["value"])?;
    } else {
        integer(value)?;
    }
    Ok(())
}
fn identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_lowercase() || c == '_')
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}
fn symbol(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}
fn lane(value: &Value, header: bool) -> Result<()> {
    string(&value["kind"])?;
    for key in ["doc"] {
        if !value[key].is_null() {
            string(&value[key])?;
        }
    }
    match kind(value) {
        "f32" | "matrix" | "flag" => {}
        "vector" => {
            string(&value["type"]["name"])?;
            strings(&value["type"]["components"])?;
            ensure!(
                !array(&value["type"]["components"]).is_empty(),
                "invalid vector components"
            );
        }
        "lanes" | "optional" => {
            integer(&value["count"])?;
            ensure!(n(&value["count"]) == 4, "Blot requires four-lane values");
            if kind(value) == "optional" {
                string(&value["message"])?;
            }
        }
        "count" => validate_bound(&value["max"])?,
        "reference" => string(&value["to"])?,
        "choice" => {
            ensure!(value["values"].is_array(), "choice requires values");
            for choice in array(&value["values"]) {
                integer(choice)?;
            }
            integer(&value["otherwise"])?;
        }
        "children" | "tail" => {
            string(&value["section"])?;
            validate_bound(&value["max"])?;
            if kind(value) == "tail" {
                integer(&value["unit"])?;
                ensure!(n(&value["unit"]) == 4, "Blot requires four-lane tails");
            } else {
                string(&value["over"])?;
                string(&value["under"])?;
            }
        }
        "magic" if header => integer(&value["value"])?,
        "schema" if header => {}
        "present" if header => string(&value["to"])?,
        "size" if header => {
            string(&value["section"])?;
            ensure!(
                matches!(value["of"].as_str(), Some("count" | "lanes")),
                "invalid size lane"
            );
        }
        other => bail!("unsupported lane kind {other}"),
    }
    Ok(())
}
fn record(value: &Value, variable: bool) -> Result<()> {
    string(&value["name"])?;
    string(&value["constant"])?;
    ensure!(
        symbol(s(&value["name"])) && symbol(s(&value["constant"])),
        "invalid record symbol"
    );
    if variable {
        fields(&value["head"], false)?;
        let rest = &value["rest"];
        string(&rest["name"])?;
        match rest["kind"].as_str() {
            Some("repeat") => {
                integer(&rest["min"])?;
                validate_bound(&rest["max"])?;
                record(&rest["record"], false)?;
            }
            Some("values") => strings(&rest["of"])?,
            _ => bail!("unsupported variable record rest"),
        }
    } else {
        fields(&value["fields"], false)?;
    }
    Ok(())
}
fn check_bounds(owner: &str, fields: &[Laid]) -> Result<()> {
    let mut seen = HashSet::new();
    for field in fields {
        if kind(&field.field) == "count"
            && let Some(expression) = field.field["max"].as_str()
        {
            for term in expression.split('-').map(str::trim) {
                ensure!(
                    seen.contains(term),
                    "{owner}.{} needs {term} before it",
                    field.name
                );
            }
        }
        seen.insert(field.name.as_str());
    }
    Ok(())
}
pub fn build(protocol: Value) -> Result<Model> {
    // Validate shapes before calculating offsets. Malformed data must return an error,
    // rather than reaching the emitters' assumptions.
    for key in ["catalog", "state", "input", "render"] {
        ensure!(protocol[key].is_object(), "{key} requires an object");
    }
    for key in ["vectors", "limits", "enumerations"] {
        ensure!(protocol[key].is_array(), "{key} requires an array");
    }
    for value in array(&protocol["vectors"]) {
        string(&value["name"])?;
        strings(&value["components"])?;
    }
    for value in array(&protocol["limits"]) {
        string(&value["limit"])?;
        integer(&value["value"])?;
    }
    for (name, value) in object(&protocol["catalog"]) {
        string(&value["list"])?;
        ensure!(
            symbol(name) && symbol(s(&value["list"])),
            "invalid catalog symbol"
        );
    }
    validate_bound(&protocol["state"]["max"])?;
    for key in ["message", "schemaMessage"] {
        string(&protocol["state"][key])?;
    }
    let input = &protocol["input"];
    strings(&input["lanes"])?;
    ensure!(
        array(&input["lanes"]).len() == 4,
        "input requires four event lanes"
    );
    ensure!(
        input["events"].is_array() && input["keys"].is_object(),
        "input requires events and keys"
    );
    validate_bound(&input["maxEvents"])?;
    integer(&input["maxExtent"])?;
    for value in object(&input["keys"]).values() {
        integer(value)?;
    }
    for key in ["frame", "nonfinite", "tag"] {
        string(&input["messages"][key])?;
    }
    let mut tags = HashSet::new();
    let mut names = HashSet::new();
    for event in array(&input["events"]) {
        integer(&event["tag"])?;
        string(&event["name"])?;
        ensure!(symbol(s(&event["name"])), "invalid event name");
        ensure!(
            n(&event["tag"]) >= 1 && tags.insert(n(&event["tag"])),
            "invalid or duplicate event tag"
        );
        ensure!(names.insert(s(&event["name"])), "duplicate event name");
        ensure!(
            event["value"].is_null() || event["fields"].is_null(),
            "event has both value and fields"
        );
        if !event["fields"].is_null() {
            ensure!(event["fields"].is_object(), "event fields require object");
        }
        let mut used = HashSet::new();
        for (name, field) in event_fields(event) {
            ensure!(symbol(&name), "invalid event field name");
            string(&field["lane"])?;
            ensure!(
                array(&input["lanes"]).contains(&field["lane"])
                    && used.insert(s(&field["lane"]).to_owned()),
                "unknown or duplicate event lane"
            );
            ensure!(
                matches!(field["wire"].as_str(), Some("key" | "integer" | "f32")),
                "invalid event wire type"
            );
            ensure!(
                matches!(
                    field["rust"].as_str(),
                    Some("f32" | "u32" | "u64" | "String")
                ),
                "invalid event Rust type"
            );
            if !field["max"].is_null() {
                integer(&field["max"])?;
                ensure!(
                    field["lane"] == input["lanes"][0],
                    "only code lane has maximum"
                );
            }
        }
        if !event["hosts"].is_null() {
            ensure!(event["hosts"].is_array(), "hosts require array");
            for host in array(&event["hosts"]) {
                string(&host["name"])?;
                ensure!(host["fields"].is_object(), "host fields require object");
                for field in object(&host["fields"]).values() {
                    ensure!(
                        matches!(
                            field["rust"].as_str(),
                            Some("f32" | "u32" | "u64" | "String")
                        ),
                        "invalid host Rust type"
                    );
                }
            }
        }
    }
    ensure!(!tags.is_empty(), "input requires events");
    let mut enumerations = HashSet::new();
    for value in array(&protocol["enumerations"]) {
        string(&value["name"])?;
        string(&value["constant"])?;
        strings(&value["values"])?;
        ensure!(
            enumerations.insert(s(&value["name"])) && enumerations.insert(s(&value["constant"])),
            "duplicate enumeration"
        );
        let values: HashSet<_> = array(&value["values"]).iter().map(s).collect();
        ensure!(
            !values.is_empty()
                && values.len() == array(&value["values"]).len()
                && values.iter().all(|name| identifier(name)),
            "enumeration requires distinct snake_case values"
        );
    }
    let render = &protocol["render"];
    fields(&render["header"], true)?;
    ensure!(
        render["groups"].is_object() && render["sections"].is_array(),
        "render requires groups and sections"
    );
    for group in object(&render["groups"]).values() {
        string(&group["type"])?;
    }
    for key in ["header", "count", "length"] {
        string(&render["messages"][key])?;
    }
    let mut header_lanes = 0;
    let header: Vec<_> = object(&render["header"])
        .iter()
        .map(|(key, field)| {
            let (group, name) = match key.split_once('.') {
                Some((group, name)) => (Some(group.to_owned()), name.to_owned()),
                None => (None, key.clone()),
            };
            let laid = Laid {
                name,
                offset: header_lanes,
                lanes: lanes(field),
                field: field.clone(),
            };
            header_lanes += laid.lanes;
            Header {
                key: key.clone(),
                group,
                laid,
            }
        })
        .collect();
    ensure!(
        header.len() >= 2
            && kind(&header[0].laid.field) == "magic"
            && kind(&header[1].laid.field) == "schema",
        "header must start with magic and schema"
    );
    for entry in &header {
        if let Some(group) = &entry.group {
            ensure!(
                render["groups"][group].is_object(),
                "unknown header group {group}"
            );
        }
    }
    for group in object(&render["groups"]).keys() {
        let members: Vec<_> = header
            .iter()
            .filter(|entry| entry.group.as_ref() == Some(group))
            .collect();
        ensure!(
            !members.is_empty()
                && members
                    .iter()
                    .filter(|entry| kind(&entry.laid.field) == "present")
                    .count()
                    <= 1,
            "group needs lanes and at most one presence flag"
        );
    }
    let magic = n(&header[0].laid.field["value"]);
    let mut sections = Vec::new();
    let mut records: Vec<Layout> = Vec::new();
    let mut section_names = HashSet::new();
    for section in array(&render["sections"]) {
        string(&section["name"])?;
        string(&section["kind"])?;
        ensure!(symbol(s(&section["name"])), "invalid section name");
        ensure!(
            section_names.insert(s(&section["name"])),
            "duplicate section"
        );
        let mut entry = Section {
            section: section.clone(),
            fixed: None,
            head: Vec::new(),
            head_lanes: 0,
            item: None,
            owner: None,
        };
        let expected = match kind(section) {
            "lanes" => {
                validate_bound(&section["max"])?;
                for key in ["nullable", "apart"] {
                    ensure!(
                        section[key].is_null() || section[key].is_boolean(),
                        "section {key} requires boolean"
                    );
                }
                if !section["constant"].is_null() {
                    string(&section["constant"])?;
                    integer(&section["max"])?;
                }
                if !section["exactly"].is_null() {
                    ensure!(
                        section["exactly"]["counts"].is_array(),
                        "exactly needs counts"
                    );
                    for count in array(&section["exactly"]["counts"]) {
                        integer(count)?;
                    }
                    string(&section["exactly"]["message"])?;
                }
                (0, 1)
            }
            "records" => {
                validate_bound(&section["max"])?;
                record(&section["record"], false)?;
                entry.fixed = Some(layout(&section["record"]));
                if let Some(owner) = section["owner"].as_str() {
                    let (name, field) = owner
                        .split_once('.')
                        .ok_or_else(|| anyhow::anyhow!("owner requires section.field"))?;
                    entry.owner = Some((name.to_owned(), field.to_owned()));
                }
                (1, 0)
            }
            "variable" => {
                validate_bound(&section["max"])?;
                validate_bound(&section["maxLanes"])?;
                record(&section["record"], true)?;
                for key in ["truncated", "invalid", "mismatch"] {
                    string(&section["messages"][key])?;
                }
                entry.head = layout_fields(&section["record"]["head"]);
                entry.head_lanes = entry.head.iter().map(|field| field.lanes).sum();
                check_bounds(s(&section["record"]["name"]), &entry.head)?;
                let rest = &section["record"]["rest"];
                if kind(rest) == "repeat" {
                    entry.head_lanes += 1;
                    entry.item = Some(layout(&rest["record"]));
                } else {
                    for term in array(&rest["of"]) {
                        ensure!(
                            entry.head.iter().any(|field| field.name == s(term)),
                            "rest refers to unknown head field"
                        );
                    }
                }
                (1, 1)
            }
            "trailing" => (0, 0),
            other => bail!("unknown section kind {other}"),
        };
        for layout in entry.fixed.iter().chain(entry.item.iter()) {
            check_bounds(s(&layout.record["name"]), &layout.fields)?;
            if let Some(old) = records
                .iter()
                .find(|old| old.record["name"] == layout.record["name"])
            {
                ensure!(old.record == layout.record, "two records have same name");
            } else {
                records.push(layout.clone());
            }
        }
        let sizes: Vec<_> = header
            .iter()
            .filter(|entry| {
                kind(&entry.laid.field) == "size" && entry.laid.field["section"] == section["name"]
            })
            .collect();
        ensure!(
            sizes.len() == expected.0 + expected.1
                && sizes
                    .iter()
                    .filter(|entry| entry.laid.field["of"] == "count")
                    .count()
                    == expected.0
                && sizes
                    .iter()
                    .filter(|entry| entry.laid.field["of"] == "lanes")
                    .count()
                    == expected.1,
            "section {} needs its size lanes",
            section["name"]
        );
        sections.push(entry);
    }
    let mut tails: Vec<(String, Vec<Tail>)> = Vec::new();
    for entry in &sections {
        for layout in entry.fixed.iter().chain(entry.item.iter()) {
            for field in &layout.fields {
                let lane = &field.field;
                if kind(lane) == "reference" {
                    ensure!(
                        !protocol["catalog"][s(&lane["to"])].is_null(),
                        "unknown catalog reference"
                    );
                }
                if matches!(kind(lane), "children" | "tail") {
                    let target = sections
                        .iter()
                        .find(|other| other.section["name"] == lane["section"])
                        .ok_or_else(|| anyhow::anyhow!("unknown target section"))?;
                    if kind(lane) == "children" {
                        ensure!(
                            target.owner
                                == Some((s(&entry.section["name"]).to_owned(), field.name.clone())),
                            "child section has wrong owner"
                        );
                    } else {
                        ensure!(
                            kind(&target.section) == "trailing" && entry.fixed.is_some(),
                            "only fixed records own trailing lanes"
                        );
                        let name = s(&lane["section"]);
                        let tail = Tail {
                            section: s(&entry.section["name"]).to_owned(),
                            layout: layout.clone(),
                            field: field.clone(),
                        };
                        if let Some((_, owners)) =
                            tails.iter_mut().find(|(section, _)| section == name)
                        {
                            owners.push(tail);
                        } else {
                            tails.push((name.to_owned(), vec![tail]));
                        }
                    }
                }
            }
        }
    }
    for entry in &header {
        let lane = &entry.laid.field;
        if kind(lane) == "size" {
            ensure!(
                section_names.contains(s(&lane["section"])),
                "header sizes unknown section"
            );
        }
        if matches!(kind(lane), "present" | "reference") {
            ensure!(
                !protocol["catalog"][s(&lane["to"])].is_null(),
                "header refers to unknown catalog"
            );
        }
    }
    for (index, entry) in sections.iter().enumerate() {
        if kind(&entry.section) == "trailing" {
            ensure!(
                index == sections.len() - 1,
                "trailing section must come last"
            );
        }
    }
    ensure!(
        sections
            .iter()
            .filter(|entry| kind(&entry.section) == "lanes" && entry.section["apart"] == true)
            .count()
            == 1,
        "protocol requires one state section"
    );
    Ok(Model {
        protocol,
        header,
        header_lanes,
        magic,
        sections,
        records,
        tails,
    })
}
