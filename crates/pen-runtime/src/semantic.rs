//! Optional application observations, separate from simulation and service answers.
//! The application validates its own values; Pen validates this bounded transport.
//! See docs/semantic-inspection.md for the wire contract and Ink projections.
use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;

use anyhow::{Result, ensure};
use pen_protocol::{Extension, RenderFrame, boundary::MAX_EXTENSION_TAG};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value as Json, json};

use crate::services::{Answer, code_points};

pub const MAX_PAYLOAD: usize = 65_536;
pub const MAX_FRAMES: usize = 3600;
pub const MAX_RETAINED_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_OBJECTS: usize = 128;
pub const MAX_FIELDS: usize = 256;
pub const MAX_DECISIONS: usize = 64;
const MAX_TEXT: usize = 256;
const MAX_WORDS: usize = 128;
const MAX_REQUEST: usize = 4096;
const PAGE_BYTES: usize = 10_000;

/// Exact identity words, including store/branch/generation where applicable.
/// No object identity is inferred from a dense row or a display label.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub kind: String,
    pub words: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Unsigned,
    Float,
    Boolean,
    Encoded,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Field {
    pub id: u32,
    pub label: String,
    pub type_name: String,
    pub kind: Kind,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub tag: u32,
    pub label: String,
    pub fields: Vec<Field>,
}

/// Mirrors the application descriptor, without depending on Ink's codecs.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Descriptor {
    pub version: u32,
    pub variants: Vec<Variant>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Type {
    pub id: u32,
    pub label: String,
    pub descriptor: Descriptor,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Schema {
    pub id: String,
    pub types: Vec<Type>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Value {
    Unsigned(u32),
    Float(f32),
    Boolean(bool),
    Encoded(Vec<u32>),
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let map = serde_json::Map::<String, Json>::deserialize(deserializer)?;
        parse_value_map(map).map_err(serde::de::Error::custom)
    }
}

fn json_u32(val: &Json, key: &str) -> std::result::Result<u32, String> {
    let n = val.as_u64().ok_or_else(|| format!("expected u32 for `{key}`"))?;
    u32::try_from(n).map_err(|_| format!("u32 out of range for `{key}`"))
}

fn json_u32_vec(val: &Json, key: &str) -> std::result::Result<Vec<u32>, String> {
    let arr = val.as_array().ok_or_else(|| format!("expected array for `{key}`"))?;
    arr.iter().map(|v| json_u32(v, key)).collect()
}

fn parse_value_map(mut map: serde_json::Map<String, Json>) -> std::result::Result<Value, String> {
    let kind = match map.remove("kind") {
        Some(Json::String(s)) => s,
        Some(_) => return Err("expected string for `kind`".to_owned()),
        None => return Err("missing field `kind`".to_owned()),
    };
    let raw = map.remove("value").ok_or_else(|| "missing field `value`".to_owned())?;
    if let Some(extra) = map.keys().next() {
        return Err(format!("unknown field `{extra}`"));
    }
    match kind.as_str() {
        "unsigned" => Ok(Value::Unsigned(json_u32(&raw, "value")?)),
        "float" =>
        {
            #[allow(clippy::cast_possible_truncation, reason = "JSON numbers narrow to f32 for semantic Value::Float")]
            raw.as_f64().map(|v| Value::Float(v as f32)).ok_or_else(|| "expected number for `value`".to_owned())
        }
        "boolean" => raw.as_bool().map(Value::Boolean).ok_or_else(|| "expected bool for `value`".to_owned()),
        "encoded" => Ok(Value::Encoded(json_u32_vec(&raw, "value")?)),
        other => Err(format!("unknown value kind `{other}`")),
    }
}

impl Value {
    fn kind(&self) -> Kind {
        match self {
            Self::Unsigned(_) => Kind::Unsigned,
            Self::Float(_) => Kind::Float,
            Self::Boolean(_) => Kind::Boolean,
            Self::Encoded(_) => Kind::Encoded,
        }
    }

    fn valid(&self) -> bool {
        match self {
            Self::Float(value) => value.is_finite(),
            Self::Encoded(words) => words.len() <= MAX_WORDS,
            _ => true,
        }
    }

    fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            _ => self == other,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FieldValue {
    pub id: u32,
    pub value: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Object {
    pub id: Identity,
    pub type_id: u32,
    pub tag: u32,
    pub fields: Vec<FieldValue>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub file: String,
    /// UTF-8 byte offsets, never character indices.
    pub start: u32,
    pub end: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Decision {
    /// Decimal u64 text keeps browser consumers from rounding large ticks.
    pub tick: String,
    pub object: Identity,
    pub system: String,
    pub source: Option<Source>,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub version: u32,
    pub schema: Schema,
    pub tick: String,
    pub objects: Vec<Object>,
    /// None means decision instrumentation was disabled. Empty means no events.
    pub decisions: Option<Vec<Decision>>,
    pub omitted_objects: u32,
    pub omitted_decisions: u32,
}

fn label(text: &str) -> bool {
    !text.is_empty() && text.len() <= MAX_TEXT && !text.chars().any(char::is_control)
}

fn identity(id: &Identity) -> bool {
    label(&id.kind) && !id.words.is_empty() && id.words.len() <= 8
}

pub fn position(text: &str) -> Option<u64> {
    if text.is_empty() || text.len() > 20 || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok().filter(|value: &u64| value.to_string() == text)
}

impl Schema {
    fn variant(&self, object: &Object) -> Option<&Variant> {
        self.types.iter().find(|ty| ty.id == object.type_id)?.descriptor.variants.iter().find(|v| v.tag == object.tag)
    }

    fn validate(&self) -> Result<()> {
        ensure!(label(&self.id) && !self.types.is_empty() && self.types.len() <= 16, "invalid schema");
        let mut types = BTreeSet::new();
        let mut fields = 0;
        let mut variants = 0;
        for ty in &self.types {
            ensure!(types.insert(ty.id) && label(&ty.label), "invalid type");
            ensure!(ty.descriptor.version > 0 && !ty.descriptor.variants.is_empty(), "invalid descriptor");
            let mut tags = BTreeSet::new();
            for variant in &ty.descriptor.variants {
                variants += 1;
                ensure!(variants <= 64 && tags.insert(variant.tag) && label(&variant.label), "invalid variant");
                let mut ids = BTreeSet::new();
                for field in &variant.fields {
                    fields += 1;
                    ensure!(fields <= MAX_FIELDS && ids.insert(field.id), "invalid field id/count");
                    ensure!(label(&field.label) && label(&field.type_name), "invalid field label");
                }
            }
        }
        Ok(())
    }
}

impl Observation {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.version == 1, "unsupported observation version");
        let tick = position(&self.tick).ok_or_else(|| anyhow::anyhow!("invalid simulation tick"))?;
        self.schema.validate()?;
        ensure!(self.objects.len() <= MAX_OBJECTS, "too many objects");
        let mut ids = BTreeSet::new();
        let mut fields = 0;
        for object in &self.objects {
            ensure!(identity(&object.id) && ids.insert(&object.id), "invalid object identity");
            let variant = self.schema.variant(object).ok_or_else(|| anyhow::anyhow!("unknown object type/variant"))?;
            ensure!(object.fields.len() == variant.fields.len(), "incomplete object");
            let mut ids = BTreeSet::new();
            for value in &object.fields {
                fields += 1;
                let field = variant.fields.iter().find(|field| field.id == value.id);
                ensure!(fields <= MAX_FIELDS && ids.insert(value.id), "invalid object field id/count");
                ensure!(
                    field.is_some_and(|field| field.kind == value.value.kind()) && value.value.valid(),
                    "invalid field value"
                );
            }
        }
        if let Some(decisions) = &self.decisions {
            ensure!(decisions.len() <= MAX_DECISIONS, "too many decisions");
            let mut previous = 0;
            for decision in decisions {
                let at = position(&decision.tick).ok_or_else(|| anyhow::anyhow!("invalid decision tick"))?;
                ensure!(at >= previous && at <= tick, "unordered decision ticks");
                previous = at;
                ensure!(
                    identity(&decision.object) && label(&decision.system) && label(&decision.reason),
                    "invalid decision"
                );
                if let Some(source) = &decision.source {
                    ensure!(label(&source.file) && source.start <= source.end, "invalid source span");
                }
            }
        }
        Ok(())
    }

    /// Development-only producer helper. The application chooses a declared tag.
    pub fn extension(&self, tag: u32) -> Result<Extension> {
        ensure!(tag != 0 && tag as usize <= MAX_EXTENSION_TAG, "invalid observation tag");
        self.validate()?;
        let bytes = serde_json::to_vec(self)?;
        ensure!(bytes.len() <= MAX_PAYLOAD, "observation too large");
        Ok(Extension { tag, values: bytes.into_iter().map(f32::from).collect() })
    }
}

fn decode(values: &[f32]) -> Result<Observation> {
    ensure!(values.len() <= MAX_PAYLOAD, "observation too large");
    let bytes: Option<Vec<u8>> = values
        .iter()
        .map(|&lane| {
            if lane.is_finite() && (0.0..=255.0).contains(&lane) && lane.fract() == 0.0 {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                Some(lane as u8)
            } else {
                None
            }
        })
        .collect();
    let mut observation: Observation =
        serde_json::from_slice(&bytes.ok_or_else(|| anyhow::anyhow!("invalid UTF-8 bytes"))?)?;
    observation.validate()?;
    // Descriptor order is presentation, not identity or compatibility.
    crate::sort::sort_by_u64_key(&mut observation.schema.types, |ty| u64::from(ty.id));
    for ty in &mut observation.schema.types {
        crate::sort::sort_by_u64_key(&mut ty.descriptor.variants, |variant| u64::from(variant.tag));
        for variant in &mut ty.descriptor.variants {
            crate::sort::sort_by_u64_key(&mut variant.fields, |field| u64::from(field.id));
        }
    }
    for object in &mut observation.objects {
        crate::sort::sort_by_u64_key(&mut object.fields, |field| u64::from(field.id));
    }
    Ok(observation)
}

// Wire values are independent complete observations, so capture can start or
// seek anywhere. Retention shares descriptors and unchanged object values.
#[derive(Debug)]
struct Sample {
    schema: Arc<Schema>,
    tick: String,
    objects: Vec<Arc<Object>>,
    decisions: Option<Vec<Decision>>,
    omitted_objects: u32,
    omitted_decisions: u32,
}

fn identity_bytes(id: &Identity) -> usize {
    id.kind.capacity() + id.words.capacity() * size_of::<u32>()
}

fn object_bytes(object: &Object) -> usize {
    size_of::<Object>()
        + 2 * size_of::<usize>()
        + identity_bytes(&object.id)
        + object.fields.capacity() * size_of::<FieldValue>()
        + object
            .fields
            .iter()
            .map(|field| match &field.value {
                Value::Encoded(words) => words.capacity() * size_of::<u32>(),
                _ => 0,
            })
            .sum::<usize>()
}

fn schema_bytes(schema: &Schema) -> usize {
    size_of::<Schema>()
        + 2 * size_of::<usize>()
        + schema.id.capacity()
        + schema.types.capacity() * size_of::<Type>()
        + schema
            .types
            .iter()
            .map(|ty| {
                ty.label.capacity()
                    + ty.descriptor.variants.capacity() * size_of::<Variant>()
                    + ty.descriptor
                        .variants
                        .iter()
                        .map(|variant| {
                            variant.label.capacity()
                                + variant.fields.capacity() * size_of::<Field>()
                                + variant
                                    .fields
                                    .iter()
                                    .map(|field| field.label.capacity() + field.type_name.capacity())
                                    .sum::<usize>()
                        })
                        .sum::<usize>()
            })
            .sum::<usize>()
}

fn same_object(a: &Object, b: &Object) -> bool {
    a.id == b.id
        && a.type_id == b.type_id
        && a.tag == b.tag
        && a.fields.len() == b.fields.len()
        && a.fields.iter().zip(&b.fields).all(|(a, b)| a.id == b.id && a.value.same(&b.value))
}

impl Sample {
    // Allocation capacities, including inline frame storage, not JSON size.
    fn bytes(&self) -> usize {
        size_of::<Kept>()
            + self.tick.capacity()
            + self.objects.capacity() * size_of::<Arc<Object>>()
            + self.decisions.as_ref().map_or(0, |decisions| {
                decisions.capacity() * size_of::<Decision>()
                    + decisions
                        .iter()
                        .map(|decision| {
                            decision.tick.capacity()
                                + identity_bytes(&decision.object)
                                + decision.system.capacity()
                                + decision.reason.capacity()
                                + decision.source.as_ref().map_or(0, |source| source.file.capacity())
                        })
                        .sum::<usize>()
            })
    }
}

#[derive(Debug)]
struct Kept {
    position: u64,
    bytes: usize,
    restored: bool,
    observation: Option<Sample>,
    unavailable: &'static str,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Watch {
    pub object: Identity,
    pub field: Option<u32>,
    /// None means any change; a value means transition to that value.
    pub equals: Option<Value>,
}

#[derive(Debug)]
pub enum Request {
    Capture { tag: Option<u32> },
    Status,
    Schema { position: String, first: u32 },
    Objects { position: String, first: u32 },
    Values { position: String, object: Identity, first: u32 },
    Diff { before: String, after: String, object: Option<Identity>, first: u32 },
    Decisions { from: String, through: String, object: Option<Identity>, first: u32 },
    Watch { watch: Watch },
    Unwatch,
    Seek { position: String },
}

impl<'de> Deserialize<'de> for Request {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let map = serde_json::Map::<String, Json>::deserialize(deserializer)?;
        parse_request_map(map).map_err(serde::de::Error::custom)
    }
}

fn req_str(map: &mut serde_json::Map<String, Json>, key: &str) -> std::result::Result<String, String> {
    match map.remove(key) {
        Some(Json::String(s)) => Ok(s),
        Some(_) => Err(format!("expected string for `{key}`")),
        None => Err(format!("missing field `{key}`")),
    }
}

fn req_opt_u32(map: &mut serde_json::Map<String, Json>, key: &str) -> std::result::Result<Option<u32>, String> {
    match map.remove(key) {
        None | Some(Json::Null) => Ok(None),
        Some(v) => json_u32(&v, key).map(Some),
    }
}

fn req_first(map: &mut serde_json::Map<String, Json>) -> std::result::Result<u32, String> {
    Ok(req_opt_u32(map, "first")?.unwrap_or(0))
}

fn parse_identity_val(val: Json, key: &str) -> std::result::Result<Identity, String> {
    let mut map = match val {
        Json::Object(m) => m,
        _ => return Err(format!("expected object for `{key}`")),
    };
    let kind = req_str(&mut map, "kind")?;
    let words_val = map.remove("words").ok_or_else(|| "missing field `words`".to_owned())?;
    let words = json_u32_vec(&words_val, "words")?;
    if let Some(extra) = map.keys().next() {
        return Err(format!("unknown field `{extra}`"));
    }
    Ok(Identity { kind, words })
}

fn req_identity(map: &mut serde_json::Map<String, Json>, key: &str) -> std::result::Result<Identity, String> {
    let val = map.remove(key).ok_or_else(|| format!("missing field `{key}`"))?;
    parse_identity_val(val, key)
}

fn req_opt_identity(
    map: &mut serde_json::Map<String, Json>,
    key: &str,
) -> std::result::Result<Option<Identity>, String> {
    match map.remove(key) {
        None | Some(Json::Null) => Ok(None),
        Some(val) => parse_identity_val(val, key).map(Some),
    }
}

fn parse_watch_val(val: Json) -> std::result::Result<Watch, String> {
    let mut map = match val {
        Json::Object(m) => m,
        _ => return Err("expected object for `watch`".to_owned()),
    };
    let object = req_identity(&mut map, "object")?;
    let field = req_opt_u32(&mut map, "field")?;
    let equals = match map.remove("equals") {
        None | Some(Json::Null) => None,
        Some(Json::Object(m)) => Some(parse_value_map(m)?),
        Some(_) => return Err("expected object for `equals`".to_owned()),
    };
    if let Some(extra) = map.keys().next() {
        return Err(format!("unknown field `{extra}`"));
    }
    Ok(Watch { object, field, equals })
}

fn parse_request_map(mut map: serde_json::Map<String, Json>) -> std::result::Result<Request, String> {
    let op = req_str(&mut map, "op")?;
    let req = match op.as_str() {
        "capture" => Request::Capture { tag: req_opt_u32(&mut map, "tag")? },
        "status" => Request::Status,
        "schema" => Request::Schema { position: req_str(&mut map, "position")?, first: req_first(&mut map)? },
        "objects" => Request::Objects { position: req_str(&mut map, "position")?, first: req_first(&mut map)? },
        "values" => Request::Values {
            position: req_str(&mut map, "position")?,
            object: req_identity(&mut map, "object")?,
            first: req_first(&mut map)?,
        },
        "diff" => Request::Diff {
            before: req_str(&mut map, "before")?,
            after: req_str(&mut map, "after")?,
            object: req_opt_identity(&mut map, "object")?,
            first: req_first(&mut map)?,
        },
        "decisions" => Request::Decisions {
            from: req_str(&mut map, "from")?,
            through: req_str(&mut map, "through")?,
            object: req_opt_identity(&mut map, "object")?,
            first: req_first(&mut map)?,
        },
        "watch" => {
            let val = map.remove("watch").ok_or_else(|| "missing field `watch`".to_owned())?;
            Request::Watch { watch: parse_watch_val(val)? }
        }
        "unwatch" => Request::Unwatch,
        "seek" => Request::Seek { position: req_str(&mut map, "position")? },
        other => return Err(format!("unknown semantic op `{other}`")),
    };
    if let Some(extra) = map.keys().next() {
        return Err(format!("unknown field `{extra}`"));
    }
    Ok(req)
}

pub fn request(values: &[f32]) -> Result<Request> {
    ensure!(values.len() <= MAX_REQUEST, "request too large");
    let mut text = String::new();
    for &value in values {
        let point = pen_protocol::boundary::whole(value).and_then(char::from_u32);
        text.push(point.ok_or_else(|| anyhow::anyhow!("invalid request text"))?);
    }
    ensure!(text.len() <= MAX_REQUEST, "request too large");
    Ok(serde_json::from_str(&text)?)
}

pub fn reply(status: u32, value: Json) -> Answer {
    Answer { status, value: 0.0, values: code_points(&value.to_string()) }
}

pub fn unavailable(reason: &str) -> Answer {
    reply(pen_protocol::service_status::UNAVAILABLE, json!({"status":"unavailable", "reason":reason}))
}

pub fn failed(reason: &str) -> Answer {
    reply(pen_protocol::service_status::FAILED, json!({"status":"failed", "reason":reason}))
}

fn ok(value: Json) -> Answer {
    reply(pen_protocol::service_status::OK, value)
}

fn page(rows: impl IntoIterator<Item = Json>, first: u32, mut metadata: Json) -> Answer {
    let first = first as usize;
    let mut selected = Vec::new();
    let mut total = 0;
    let mut bytes = 0;
    let mut full = false;
    for row in rows {
        if total >= first && !full {
            let size = row.to_string().len();
            if bytes + size > PAGE_BYTES || selected.len() == 64 {
                full = true;
            } else {
                bytes += size;
                selected.push(row);
            }
        }
        total += 1;
    }
    let next = first.saturating_add(selected.len());
    metadata["status"] = json!("ok");
    metadata["rows"] = json!(selected);
    metadata["total"] = json!(total);
    metadata["next"] = if next < total { json!(next) } else { Json::Null };
    ok(metadata)
}

/// A bounded observation ring indexed by existing timeline positions, not a
/// second simulation history. Defaults to no capture and no per-frame work.
#[derive(Debug, Default)]
pub struct Observations {
    tag: Option<u32>,
    frames: VecDeque<Kept>,
    bytes: usize,
    schema: Option<Arc<Schema>>,
    epoch: u64,
    invalidated: Option<&'static str>,
    evicted: u64,
    watch: Option<Watch>,
    broke: Option<u64>,
    pending_break: bool,
}

impl Observations {
    fn retained_bytes(&self) -> usize {
        self.bytes + (self.frames.capacity() - self.frames.len()) * size_of::<Kept>()
    }

    pub fn enabled(&self) -> bool {
        self.tag.is_some()
    }

    pub fn watching(&self) -> bool {
        self.watch.is_some()
    }

    pub fn take_break(&mut self) -> Option<u64> {
        std::mem::take(&mut self.pending_break).then_some(self.broke).flatten()
    }

    pub fn invalidate(&mut self, why: &'static str) {
        self.frames.clear();
        self.bytes = 0;
        self.schema = None;
        self.watch = None;
        self.broke = None;
        self.pending_break = false;
        self.epoch = self.epoch.saturating_add(1);
        self.invalidated = Some(why);
    }

    pub fn branch(&mut self, position: u64) {
        while self.frames.back().is_some_and(|frame| frame.position > position) {
            let removed = self.frames.pop_back().unwrap();
            self.release(removed);
        }
        self.broke = None;
        self.pending_break = false;
    }

    pub fn retain_from(&mut self, position: u64) {
        while self.frames.front().is_some_and(|frame| frame.position < position) {
            let removed = self.frames.pop_front().unwrap();
            self.release(removed);
            self.evicted = self.evicted.saturating_add(1);
        }
    }

    /// Called only for an accepted frame, at the timeline position after it.
    /// Invalid observation payloads never reject a simulation frame.
    pub fn ran(&mut self, position: u64, frame: &RenderFrame, restored: bool) {
        self.record(position, frame, restored, true);
    }

    /// Seeking recomputes observations but cannot consume an armed watch.
    pub fn replayed(&mut self, position: u64, frame: &RenderFrame, restored: bool) {
        self.record(position, frame, restored, false);
    }

    fn release(&mut self, kept: Kept) {
        self.bytes -= kept.bytes;
        if let Some(sample) = kept.observation {
            for object in sample.objects {
                if Arc::strong_count(&object) == 1 {
                    self.bytes -= object_bytes(&object);
                }
            }
        }
    }

    fn record(&mut self, position: u64, frame: &RenderFrame, restored: bool, watching: bool) {
        let Some(tag) = self.tag else { return };
        let mut extensions = frame.extensions.iter().filter(|extension| extension.tag == tag);
        let found = extensions.next();
        let (observation, unavailable) = match found {
            None => (None, "instrumentation_disabled"),
            Some(_) if extensions.next().is_some() => (None, "invalid_observation"),
            Some(extension) => match decode(&extension.values) {
                Ok(value) => (Some(value), ""),
                Err(_) => (None, "invalid_observation"),
            },
        };
        if let Some(observation) = &observation {
            if self.schema.as_ref().is_some_and(|schema| schema.as_ref() != &observation.schema) {
                self.invalidate("schema_changed");
            }
            if self.schema.is_none() {
                let schema = Arc::new(observation.schema.clone());
                self.bytes += schema_bytes(&schema);
                self.schema = Some(schema);
            }
        }
        let at = if self.frames.back().is_none_or(|frame| frame.position < position) {
            self.frames.len()
        } else {
            self.frames.partition_point(|frame| frame.position < position)
        };
        let observation = observation.map(|value| {
            let previous = self
                .frames
                .get(at)
                .filter(|frame| frame.position == position)
                .or_else(|| at.checked_sub(1).and_then(|index| self.frames.get(index)))
                .and_then(|frame| frame.observation.as_ref());
            let objects = value
                .objects
                .into_iter()
                .map(|object| {
                    if let Some(shared) =
                        previous.and_then(|sample| sample.objects.iter().find(|old| same_object(old, &object)))
                    {
                        shared.clone()
                    } else {
                        self.bytes += object_bytes(&object);
                        Arc::new(object)
                    }
                })
                .collect();
            Sample {
                schema: self.schema.as_ref().unwrap().clone(),
                tick: value.tick,
                objects,
                decisions: value.decisions,
                omitted_objects: value.omitted_objects,
                omitted_decisions: value.omitted_decisions,
            }
        });
        // Replaying a position replaces it, preserving the future until branching.
        if self.frames.get(at).is_some_and(|frame| frame.position == position) {
            let removed = self.frames.remove(at).unwrap();
            self.release(removed);
        }
        if watching
            && !restored
            && let (Some(watch), Some(now)) = (&self.watch, &observation)
        {
            let before = at
                .checked_sub(1)
                .and_then(|index| self.frames.get(index))
                .filter(|frame| frame.position.checked_add(1) == Some(position));
            if let Some(before) = before.and_then(|frame| frame.observation.as_ref())
                && transitioned(watch, before, now)
            {
                self.broke = Some(position);
                self.pending_break = true;
                self.watch = None;
            }
        }
        let bytes = observation.as_ref().map_or(size_of::<Kept>(), Sample::bytes);
        self.bytes += bytes;
        self.frames.insert(at, Kept { position, observation, bytes, restored, unavailable });
        while self.frames.len() > MAX_FRAMES || self.retained_bytes() > MAX_RETAINED_BYTES {
            // A seek keeps the observation it just reconstructed, evicting
            // distant future rows first when the ring is already full.
            let removed = if self.frames.back().is_some_and(|frame| frame.position > position) {
                self.frames.pop_back()
            } else {
                self.frames.pop_front()
            };
            self.release(removed.unwrap());
            self.evicted = self.evicted.saturating_add(1);
        }
    }

    fn frame(&self, at: &str) -> Result<&Kept, Answer> {
        let at = position(at).ok_or_else(|| failed("invalid_position"))?;
        if self.tag.is_none() {
            return Err(unavailable("capture_disabled"));
        }
        self.frames.iter().find(|frame| frame.position == at).ok_or_else(|| unavailable("not_retained"))
    }

    fn observed(&self, at: &str) -> Result<(&Kept, &Sample), Answer> {
        let frame = self.frame(at)?;
        frame.observation.as_ref().map(|observation| (frame, observation)).ok_or_else(|| unavailable(frame.unavailable))
    }

    pub fn answer(&mut self, request: Request) -> Answer {
        self.query(request).unwrap_or_else(|answer| answer)
    }

    fn query(&mut self, request: Request) -> Result<Answer, Answer> {
        match request {
            Request::Capture { tag } => {
                if tag.is_some_and(|tag| tag == 0 || tag as usize > MAX_EXTENSION_TAG) {
                    return Err(failed("invalid_extension_tag"));
                }
                if self.tag != tag {
                    self.invalidate(if tag.is_some() { "capture_started" } else { "capture_disabled" });
                    self.tag = tag;
                    self.evicted = 0;
                }
                self.query(Request::Status)
            }
            Request::Status => Ok(ok(json!({
                "status":"ok", "tag":self.tag, "epoch":self.epoch.to_string(),
                "earliest":self.frames.front().map(|frame| frame.position.to_string()),
                "latest":self.frames.back().map(|frame| frame.position.to_string()),
                "retained":self.frames.len(), "retained_bytes":self.retained_bytes(),
                "max_frames":MAX_FRAMES, "max_bytes":MAX_RETAINED_BYTES,
                "evicted":self.evicted.to_string(), "invalidated":self.invalidated,
                "watch":self.watch, "paused_at":self.broke.map(|at| at.to_string()),
            }))),
            Request::Schema { position, first } => {
                let (_, sample) = self.observed(&position)?;
                let rows = sample.schema.types.iter().flat_map(|ty| {
                    ty.descriptor.variants.iter().flat_map(move |variant| {
                        // Include a row for nullary variants, too.
                        std::iter::once(json!({"type":ty.id, "label":ty.label, "version":ty.descriptor.version,
                        "tag":variant.tag, "variant":variant.label, "field":null}))
                        .chain(
                            variant
                                .fields
                                .iter()
                                .map(move |field| json!({"type":ty.id,"tag":variant.tag,"field":field})),
                        )
                    })
                });
                Ok(page(rows, first, json!({"schema":sample.schema.id})))
            }
            Request::Objects { position, first } => {
                let (frame, sample) = self.observed(&position)?;
                Ok(page(
                    sample.objects.iter().map(|object| json!({"id":object.id,"type":object.type_id,"tag":object.tag})),
                    first,
                    json!({"tick":sample.tick,"omitted":sample.omitted_objects,"restored":frame.restored}),
                ))
            }
            Request::Values { position, object, first } => {
                let (_, sample) = self.observed(&position)?;
                let found = sample.objects.iter().find(|row| row.id == object).ok_or_else(|| {
                    unavailable(if sample.omitted_objects > 0 { "objects_omitted" } else { "object_absent" })
                })?;
                let variant = sample.schema.variant(found).unwrap();
                let rows = found.fields.iter().map(|value| json!({"field":variant.fields.iter().find(|field| field.id == value.id).unwrap(),"value":value.value}));
                Ok(page(rows, first, json!({"tick":sample.tick,"object":object,"variant":variant.label})))
            }
            Request::Diff { before, after, object, first } => {
                let (a, before) = self.observed(&before)?;
                let (b, after) = self.observed(&after)?;
                if before.schema != after.schema {
                    return Err(unavailable("schema_changed"));
                }
                Ok(page(
                    diff(before, after, object.as_ref()),
                    first,
                    json!({"before":a.position.to_string(),"after":b.position.to_string(),
                    "omitted_before":before.omitted_objects,"omitted_after":after.omitted_objects,"restored":b.restored}),
                ))
            }
            Request::Decisions { from, through, object, first } => {
                let from = position(&from).ok_or_else(|| failed("invalid_position"))?;
                let through = position(&through).ok_or_else(|| failed("invalid_position"))?;
                if through < from || through - from >= MAX_FRAMES as u64 {
                    return Err(failed("invalid_range"));
                }
                let mut rows = Vec::new();
                let mut omitted = 0u64;
                for at in from..=through {
                    let (_, sample) = self.observed(&at.to_string())?;
                    let decisions = sample.decisions.as_ref().ok_or_else(|| unavailable("decisions_disabled"))?;
                    omitted += u64::from(sample.omitted_decisions);
                    rows.extend(
                        decisions
                            .iter()
                            .enumerate()
                            .filter(|(_, event)| object.as_ref().is_none_or(|id| &event.object == id))
                            .map(|(index, event)| json!({"position":at.to_string(),"index":index,"decision":event})),
                    );
                }
                Ok(page(rows, first, json!({"omitted":omitted.to_string()})))
            }
            Request::Watch { watch } => {
                if self.tag.is_none() {
                    return Err(unavailable("capture_disabled"));
                }
                if !identity(&watch.object)
                    || watch.equals.as_ref().is_some_and(|value| !value.valid() || watch.field.is_none())
                {
                    return Err(failed("invalid_watch"));
                }
                self.watch = Some(watch);
                self.broke = None;
                self.pending_break = false;
                self.query(Request::Status)
            }
            Request::Unwatch => {
                self.watch = None;
                self.broke = None;
                self.pending_break = false;
                self.query(Request::Status)
            }
            Request::Seek { .. } => Err(failed("seek_requires_host")),
        }
    }
}

fn transitioned(watch: &Watch, before: &Sample, after: &Sample) -> bool {
    let a = before.objects.iter().find(|object| object.id == watch.object);
    let b = after.objects.iter().find(|object| object.id == watch.object);
    if (a.is_none() && before.omitted_objects > 0) || (b.is_none() && after.omitted_objects > 0) {
        return false;
    }
    let Some(field) = watch.field else {
        return !diff(before, after, Some(&watch.object)).is_empty();
    };
    fn value(object: Option<&Object>, field: u32) -> Option<&Value> {
        object.and_then(|object| object.fields.iter().find(|value| value.id == field)).map(|field| &field.value)
    }
    let (a, b) = (value(a.map(AsRef::as_ref), field), value(b.map(AsRef::as_ref), field));
    let changed = match (a, b) {
        (Some(a), Some(b)) => !a.same(b),
        (None, None) => false,
        _ => true,
    };
    changed && watch.equals.as_ref().is_none_or(|expected| b.is_some_and(|value| value.same(expected)))
}

fn diff(before: &Sample, after: &Sample, only: Option<&Identity>) -> Vec<Json> {
    let mut rows = Vec::new();
    let ids: BTreeSet<_> = before
        .objects
        .iter()
        .chain(&after.objects)
        .map(|object| &object.id)
        .filter(|id| only.is_none_or(|only| only == *id))
        .collect();
    for id in ids {
        let a = before.objects.iter().find(|object| &object.id == id);
        let b = after.objects.iter().find(|object| &object.id == id);
        match (a, b) {
            (None, Some(_)) => rows.push(
                json!({"object":id,"change":if before.omitted_objects == 0 {"created"} else {"previously_unobserved"}}),
            ),
            (Some(_), None) => {
                rows.push(json!({"object":id,"change":if after.omitted_objects == 0 {"destroyed"} else {"unobserved"}}))
            }
            (Some(a), Some(b)) => {
                if a.type_id != b.type_id || a.tag != b.tag {
                    rows.push(
                        json!({"object":id,"change":"variant","before":[a.type_id,a.tag],"after":[b.type_id,b.tag]}),
                    );
                }
                let fields: BTreeSet<_> = a.fields.iter().chain(&b.fields).map(|field| field.id).collect();
                for id in fields {
                    let old = a.fields.iter().find(|field| field.id == id);
                    let new = b.fields.iter().find(|field| field.id == id);
                    if old.zip(new).is_some_and(|(old, new)| old.value.same(&new.value)) {
                        continue;
                    }
                    let label = after
                        .schema
                        .variant(b)
                        .and_then(|variant| variant.fields.iter().find(|field| field.id == id))
                        .or_else(|| {
                            before
                                .schema
                                .variant(a)
                                .and_then(|variant| variant.fields.iter().find(|field| field.id == id))
                        });
                    rows.push(
                        json!({"object":b.id,"change":"field","field":id,"label":label.map(|field| &field.label),
                        "before":old.map(|field| &field.value),"after":new.map(|field| &field.value)}),
                    );
                }
            }
            (None, None) => unreachable!(),
        }
    }
    rows
}

#[cfg(test)]
mod tests;
