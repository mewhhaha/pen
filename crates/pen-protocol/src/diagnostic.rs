//! Structural explanations for rejected render packets. The generated walk
//! runs only after decoding fails; it never changes which packets are accepted.
use crate::boundary::{MAX_PROVENANCE_ENTRIES, MAX_PROVENANCE_LANES, MAX_PROVENANCE_TEXT, PROVENANCE_TAG};
use serde::{Deserialize, Serialize};
use std::fmt;

/// An application's own explanation, scoped to this packet only.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub version: u32,
    pub entries: Vec<Origin>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Origin {
    /// Half-open absolute packet lane range, excluding the extension block.
    pub first: u32,
    pub end: u32,
    /// Opaque stable identity, including store/generation if slots are reused.
    pub object: Option<String>,
    pub system: Option<String>,
    pub source: Option<SourceSpan>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceSpan {
    pub file: String,
    /// Half-open UTF-8 byte offsets in the application's source.
    pub start: u32,
    pub end: u32,
}

impl Provenance {
    fn validate(&self, packet_end: usize) -> anyhow::Result<()> {
        anyhow::ensure!(self.version == 1, "unsupported provenance version");
        anyhow::ensure!(self.entries.len() <= MAX_PROVENANCE_ENTRIES, "too many provenance entries");
        let text = |value: &str| !value.is_empty() && value.len() <= MAX_PROVENANCE_TEXT;
        let mut previous_end = 0;
        for entry in &self.entries {
            anyhow::ensure!(
                entry.first >= previous_end && entry.first < entry.end && u64::from(entry.end) <= packet_end as u64,
                "provenance ranges must be ordered, disjoint and inside the packet"
            );
            anyhow::ensure!(
                entry.object.is_some() || entry.system.is_some() || entry.source.is_some(),
                "empty provenance entry"
            );
            anyhow::ensure!(
                entry.object.as_deref().is_none_or(text) && entry.system.as_deref().is_none_or(text),
                "invalid provenance label"
            );
            if let Some(source) = &entry.source {
                anyhow::ensure!(text(&source.file) && source.start <= source.end, "invalid provenance source span");
            }
            previous_end = entry.end;
        }
        Ok(())
    }

    /// Makes the reserved extension; applications opt in when constructing a
    /// development packet. `packet_end` excludes the extensions marker.
    pub fn extension(&self, packet_end: usize) -> anyhow::Result<crate::Extension> {
        self.validate(packet_end)?;
        let bytes = serde_json::to_vec(self)?;
        anyhow::ensure!(bytes.len() <= MAX_PROVENANCE_LANES, "provenance payload is too large");
        Ok(crate::Extension {
            tag: u32::try_from(PROVENANCE_TAG).unwrap(),
            values: bytes.into_iter().map(f32::from).collect(),
        })
    }
}

pub(crate) fn read_provenance(values: &[f32], packet_end: usize) -> anyhow::Result<Provenance> {
    anyhow::ensure!(values.len() <= MAX_PROVENANCE_LANES, "provenance payload is too large");
    let bytes: Option<Vec<u8>> =
        values.iter().map(|value| crate::boundary::whole(*value).and_then(|value| u8::try_from(value).ok())).collect();
    let provenance: Provenance =
        serde_json::from_slice(&bytes.ok_or_else(|| anyhow::anyhow!("provenance must contain UTF-8 bytes"))?)?;
    provenance.validate(packet_end)?;
    Ok(provenance)
}

/// Untrusted or ambiguous metadata is ignored when explaining another fault.
pub(crate) fn origin(packet: &[f32], start: usize, lane: usize) -> Option<Origin> {
    use crate::boundary::{EXTENSIONS_MARKER, MAX_EXTENSION_LANES, MAX_EXTENSION_TAG, MAX_EXTENSIONS, whole};
    if packet.get(start) != Some(&EXTENSIONS_MARKER) {
        return None;
    }
    let read = |at: usize, max: usize| {
        whole(*packet.get(at)?).and_then(|value| usize::try_from(value).ok()).filter(|value| *value <= max)
    };
    let count = read(start.checked_add(1)?, MAX_EXTENSIONS)?;
    let mut at = start.checked_add(2)?;
    let mut found = None;
    let mut seen = false;
    for _ in 0..count {
        let tag = read(at, MAX_EXTENSION_TAG)?;
        let length = read(at.checked_add(1)?, MAX_EXTENSION_LANES)?;
        let first = at.checked_add(2)?;
        let end = first.checked_add(length)?;
        let values = packet.get(first..end)?;
        if tag == PROVENANCE_TAG {
            if seen {
                return None;
            }
            seen = true;
            found = read_provenance(values, start)
                .ok()?
                .entries
                .into_iter()
                .find(|entry| u64::from(entry.first) <= lane as u64 && (lane as u64) < u64::from(entry.end));
        }
        at = end;
    }
    (at == packet.len()).then_some(found).flatten()
}

/// The value at a rejected boundary, including boundaries with no value.
#[derive(Clone, Debug, PartialEq)]
pub enum Actual {
    Value(f32),
    Missing,
    Length(usize),
}

impl fmt::Display for Actual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(value) => write!(f, "{value}"),
            Self::Missing => f.write_str("missing"),
            Self::Length(length) => write!(f, "{length} lanes"),
        }
    }
}

/// Where the host rejected a value, rather than an inferred arithmetic origin.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameDiagnostic {
    pub section: &'static str,
    pub item: Option<usize>,
    pub path: String,
    pub lane: usize,
    pub expected: String,
    pub actual: Actual,
    pub origin: Option<Box<Origin>>,
}

impl fmt::Display for FrameDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at frame lane {}: expected {}, got {}", self.path, self.lane, self.expected, self.actual)?;
        if let Some(origin) = &self.origin {
            f.write_str("; application provenance:")?;
            if let Some(object) = &origin.object {
                write!(f, " object {object:?}")?;
            }
            if let Some(system) = &origin.system {
                write!(f, " system {system:?}")?;
            }
            if let Some(source) = &origin.source {
                write!(f, " source {:?}:{}..{} (UTF-8 bytes)", source.file, source.start, source.end)?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for FrameDiagnostic {}

pub(crate) type Check<T = ()> = Result<T, FrameDiagnostic>;

pub(crate) struct Location {
    section: &'static str,
    item: Option<usize>,
    path: String,
}

impl Location {
    pub(crate) fn section(section: &'static str) -> Self {
        Self { section, item: None, path: section.into() }
    }

    pub(crate) fn field(&self, name: &str) -> Self {
        Self { section: self.section, item: self.item, path: format!("{}.{name}", self.path) }
    }

    pub(crate) fn index(&self, item: usize) -> Self {
        Self { section: self.section, item: self.item.or(Some(item)), path: format!("{}[{item}]", self.path) }
    }

    fn element(&self, index: usize) -> Self {
        Self { section: self.section, item: self.item, path: format!("{}[{index}]", self.path) }
    }

    pub(crate) fn error(&self, lane: usize, expected: impl Into<String>, actual: Actual) -> FrameDiagnostic {
        FrameDiagnostic {
            section: self.section,
            item: self.item,
            path: self.path.clone(),
            lane,
            expected: expected.into(),
            actual,
            origin: None,
        }
    }
}

pub(crate) struct Inspector<'a> {
    pub(crate) packet: &'a [f32],
}

impl Inspector<'_> {
    pub(crate) fn value(&self, at: &Location, lane: usize) -> Check<f32> {
        self.packet.get(lane).copied().ok_or_else(|| at.error(lane, "a float32 lane", Actual::Missing))
    }

    pub(crate) fn equal(&self, at: &Location, lane: usize, expected: f32) -> Check {
        let actual = self.value(at, lane)?;
        if actual != expected {
            return Err(at.error(lane, expected.to_string(), Actual::Value(actual)));
        }
        Ok(())
    }

    pub(crate) fn count(&self, at: &Location, lane: usize, maximum: usize) -> Check<usize> {
        let value = self.value(at, lane)?;
        if let Some(value) = crate::boundary::whole(value)
            && let Ok(value) = usize::try_from(value)
            && value <= maximum
        {
            return Ok(value);
        }
        Err(at.error(lane, format!("a whole number in 0..={maximum}"), Actual::Value(value)))
    }

    pub(crate) fn reference(&self, at: &Location, lane: usize, count: usize, required: bool) -> Check {
        if required && count == 0 {
            return Err(at.error(lane, "an index in a nonempty catalog", Actual::Value(self.value(at, lane)?)));
        }
        self.count(at, lane, count.saturating_sub(1)).map(|_| ())
    }

    pub(crate) fn finite(&self, at: &Location, lane: usize) -> Check {
        let value = self.value(at, lane)?;
        if !value.is_finite() {
            return Err(at.error(lane, "a finite float32", Actual::Value(value)));
        }
        Ok(())
    }

    /// A vector uses component names, other runs use zero-based lane indices.
    pub(crate) fn values(&self, at: &Location, first: usize, count: usize, components: &[&str]) -> Check {
        let end =
            first.checked_add(count).ok_or_else(|| at.error(first, "a bounded lane range", Actual::Length(count)))?;
        let available = self.packet.get(first..end.min(self.packet.len())).unwrap_or_default();
        if let Some(index) = available.iter().position(|value| !value.is_finite()) {
            let at = components.get(index).map_or_else(|| at.element(index), |name| at.field(name));
            return self.finite(&at, first + index);
        }
        if end > self.packet.len() {
            let index = self.packet.len().saturating_sub(first);
            let at = components.get(index).map_or_else(|| at.element(index), |name| at.field(name));
            return Err(at.error(first + index, "a finite float32", Actual::Missing));
        }
        Ok(())
    }

    pub(crate) fn end(&self, at: &Location, first: usize, count: usize, width: usize) -> Check<usize> {
        count
            .checked_mul(width)
            .and_then(|length| first.checked_add(length))
            .ok_or_else(|| at.error(first, "a bounded lane range", Actual::Length(count)))
    }

    pub(crate) fn within(&self, at: &Location, first: usize, end: usize, limit: usize) -> Check {
        if end > limit {
            return Err(at.error(first, format!("a record ending by lane {limit}"), Actual::Length(end - first)));
        }
        Ok(())
    }

    pub(crate) fn exhausted(&self, at: &Location, end: usize, expected: usize) -> Check {
        if end != expected {
            return Err(at.error(end.min(expected), format!("section ending at lane {expected}"), Actual::Length(end)));
        }
        Ok(())
    }
}

#[cfg(test)]
mod measurements {
    use super::*;
    use crate::boundary::*;
    use crate::packets::codec::decode_raw;
    use std::hint::black_box;
    use std::time::Instant;

    #[test]
    fn diagnostic_walk_accepts_the_real_packets_the_decoder_accepts() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dist");
        let catalog = read_catalog(&root.join("gdev/assets")).unwrap();
        let schema = std::fs::read_to_string(root.join("packets/schema.txt")).unwrap().trim().parse().unwrap();
        let mut count = 0;
        for entry in std::fs::read_dir(root.join("packets")).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|extension| extension != "f32") {
                continue;
            }
            let bytes = std::fs::read(&path).unwrap();
            let packet: Vec<_> = bytes.as_chunks::<4>().0.iter().map(|bytes| f32::from_le_bytes(*bytes)).collect();
            decode_frame(&packet, schema, &catalog).unwrap();
            crate::packets::codec::inspect_frame(&packet, schema, &catalog)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            count += 1;
        }
        assert!(count > 0, "just packets must prepare real fixtures");
    }

    /// Compare the raw decoder and its diagnostic wrapper in the same binary,
    /// alternating measurement order. No GPU or compiler runs during capture.
    #[test]
    #[ignore = "release-mode diagnostic overhead measurement; needs just packets"]
    fn diagnostic_overhead() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dist");
        let catalog = read_catalog(&root.join("gdev/assets")).unwrap();
        let schema = std::fs::read_to_string(root.join("packets/schema.txt")).unwrap().trim().parse().unwrap();
        let path = std::fs::read_dir(root.join("packets"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "f32"))
            .min()
            .unwrap();
        let bytes = std::fs::read(path).unwrap();
        let packet: Vec<_> = bytes.as_chunks::<4>().0.iter().map(|bytes| f32::from_le_bytes(*bytes)).collect();
        let provenance = Provenance {
            version: 1,
            entries: vec![Origin {
                first: u32::try_from(lanes::header::CLEAR).unwrap(),
                end: u32::try_from(lanes::header::CLEAR + 4).unwrap(),
                object: Some("fixture:1:1".into()),
                system: Some("lighting".into()),
                source: None,
            }],
        }
        .extension(packet.len())
        .unwrap();
        let mut observed = packet.clone();
        observed.extend([EXTENSIONS_MARKER, 1.0, provenance.tag as f32, provenance.values.len() as f32]);
        observed.extend(provenance.values);
        let mut invalid = observed.clone();
        invalid[lanes::header::CLEAR] = f32::NAN;
        let names = ["raw", "diagnostics_disabled", "provenance_enabled", "rejected_with_provenance"];
        let mut samples: [Vec<f64>; 4] = Default::default();
        let mut frame = crate::RenderFrame::default();
        let iterations = 2000;
        for round in 0..7 {
            for step in 0..4 {
                let mode = if round % 2 == 0 { step } else { 3 - step };
                let packet = match mode {
                    2 => &observed,
                    3 => &invalid,
                    _ => &packet,
                };
                let decoder = black_box(if mode == 0 { decode_raw } else { decode_into });
                for _ in 0..20 {
                    let _ = decoder(black_box(packet), schema, &catalog, &mut frame);
                }
                let started = Instant::now();
                for _ in 0..iterations {
                    let result = decoder(black_box(packet), schema, &catalog, &mut frame);
                    if mode == 3 {
                        black_box(format!("{:#}", result.unwrap_err()));
                    } else {
                        black_box(result.unwrap());
                    }
                }
                samples[mode].push(started.elapsed().as_secs_f64() * 1e6 / f64::from(iterations));
            }
        }
        let mut report = serde_json::Map::new();
        for (name, mut values) in names.into_iter().zip(samples) {
            values.sort_by(f64::total_cmp);
            report.insert(name.into(), serde_json::json!({"median_us": values[3], "samples_us": values}));
        }
        report.insert("packet_lanes".into(), packet.len().into());
        report.insert("iterations_per_sample".into(), iterations.into());
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        if let Some(directory) = std::env::var_os("PEN_DIAGNOSTIC_FIXTURES") {
            std::fs::write(
                std::path::Path::new(&directory).join("timing.json"),
                serde_json::to_vec_pretty(&report).unwrap(),
            )
            .unwrap();
        }
    }
}
