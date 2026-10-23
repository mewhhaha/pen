mod fixtures;
use pen_protocol::boundary::*;
use pen_protocol::diagnostic::{Actual, FrameDiagnostic, Origin, Provenance, SourceSpan};

#[test]
fn malformed_packets_have_structural_context_and_reuse_recovers() {
    let mut frame = pen_protocol::RenderFrame::default();
    let mut exported = Vec::new();
    for case in fixtures::cases() {
        let error = decode_into(&case.packet, 1, &fixtures::catalog(), &mut frame).unwrap_err();
        let diagnostic = error.downcast_ref::<FrameDiagnostic>().expect("structured diagnostic");
        assert_eq!(diagnostic.path, case.path, "{}: {error:#}", case.name);
        assert_eq!(diagnostic.lane, case.lane, "{}: {error:#}", case.name);
        assert!(format!("{error:#}").len() < 8192);
        assert!(!diagnostic.expected.is_empty());
        assert!(diagnostic.origin.is_none());
        if case.name == "nested" {
            assert_eq!((diagnostic.section, diagnostic.item), ("insets", Some(1)));
        }
        if case.name == "truncated" {
            assert_eq!(diagnostic.actual, Actual::Missing);
        }
        exported.push(serde_json::json!({"name": case.name, "bits": case.packet.iter().map(|value| value.to_bits()).collect::<Vec<_>>(), "diagnostic": format!("{error:#}")}));
        decode_into(&fixtures::blank(), 1, &fixtures::catalog(), &mut frame).unwrap();
        assert!(frame.draws.is_empty() && frame.insets.is_empty() && frame.extensions.is_empty());
    }
    let mut packet = fixtures::blank();
    attach(&mut packet, &provenance());
    packet[lanes::header::CLEAR] = f32::NAN;
    let error = decode_frame(&packet, 1, &fixtures::catalog()).err().unwrap();
    exported.push(serde_json::json!({"name": "provenance", "bits": packet.iter().map(|value| value.to_bits()).collect::<Vec<_>>(), "diagnostic": format!("{error:#}")}));
    if let Some(path) = std::env::var_os("PEN_DIAGNOSTIC_FIXTURES") {
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(std::path::Path::new(&path).join("cases.json"), serde_json::to_vec(&exported).unwrap()).unwrap();
    }
}

fn provenance() -> Provenance {
    Provenance {
        version: 1,
        entries: vec![Origin {
            first: u32::try_from(lanes::header::CLEAR).unwrap(),
            end: u32::try_from(lanes::header::CLEAR + 4).unwrap(),
            object: Some("building:7:42".into()),
            system: Some("lighting".into()),
            source: Some(SourceSpan { file: "src/świat.blot".into(), start: 19, end: 47 }),
        }],
    }
}

fn attach(packet: &mut Vec<f32>, provenance: &Provenance) {
    let extension = provenance.extension(packet.len()).unwrap();
    packet.extend([EXTENSIONS_MARKER, 1.0, extension.tag as f32, extension.values.len() as f32]);
    packet.extend(extension.values);
}

#[test]
fn provenance_is_optional_bounded_and_scoped_to_the_rejected_packet() {
    let mut packet = fixtures::blank();
    attach(&mut packet, &provenance());
    let observed = decode_frame(&packet, 1, &fixtures::catalog()).unwrap().render;
    assert_eq!(observed, decode_frame(&fixtures::blank(), 1, &fixtures::catalog()).unwrap().render);
    packet[lanes::header::CLEAR] = f32::NAN;
    let error = decode_frame(&packet, 1, &fixtures::catalog()).err().unwrap();
    assert_eq!(error.downcast_ref::<FrameDiagnostic>().unwrap().origin, Some(Box::new(provenance().entries.remove(0))));
    assert!(format!("{error:#}").contains("UTF-8 bytes"));
    let mut next = provenance();
    next.entries[0].object = Some("building:7:43".into());
    next.entries[0].source.as_mut().unwrap().start = u32::MAX - 1;
    next.entries[0].source.as_mut().unwrap().end = u32::MAX;
    let mut replaced = fixtures::blank();
    attach(&mut replaced, &next);
    replaced[lanes::header::CLEAR] = f32::NAN;
    assert_eq!(
        decode_frame(&replaced, 1, &fixtures::catalog())
            .err()
            .unwrap()
            .downcast_ref::<FrameDiagnostic>()
            .unwrap()
            .origin
            .as_deref(),
        next.entries.first(),
        "a reused slot and new source span never inherit an earlier packet's provenance"
    );
    let mut plain = fixtures::blank();
    plain[lanes::header::CLEAR] = f32::NAN;
    assert!(
        decode_frame(&plain, 1, &fixtures::catalog())
            .err()
            .unwrap()
            .downcast_ref::<FrameDiagnostic>()
            .unwrap()
            .origin
            .is_none()
    );
    // A malformed layout must never cause a scan for a magic number in payloads.
    packet[lanes::header::DRAW_COUNT] = f32::NAN;
    assert!(
        decode_frame(&packet, 1, &fixtures::catalog())
            .err()
            .unwrap()
            .downcast_ref::<FrameDiagnostic>()
            .unwrap()
            .origin
            .is_none()
    );
    let mut invalid = provenance();
    invalid.entries[0].object = Some("x".repeat(MAX_PROVENANCE_TEXT + 1));
    assert!(invalid.extension(PACKET_HEADER_FLOATS).is_err());
    invalid = provenance();
    invalid.entries.push(invalid.entries[0].clone());
    assert!(invalid.extension(PACKET_HEADER_FLOATS).is_err(), "ambiguous overlapping origins");
    invalid = provenance();
    invalid.version = 2;
    assert!(invalid.extension(PACKET_HEADER_FLOATS).is_err());
    let mut malformed = fixtures::blank();
    malformed.extend([EXTENSIONS_MARKER, 1.0, PROVENANCE_TAG as f32, 1.0, 0.0]);
    let error = decode_frame(&malformed, 1, &fixtures::catalog()).err().unwrap();
    assert_eq!(error.downcast_ref::<FrameDiagnostic>().unwrap().path, "extensions[0].provenance");

    // Validate the receiver independently of the well-behaved sender helper.
    let rejected = |value: serde_json::Value, bytes: Option<usize>| {
        let mut bytes_out = serde_json::to_vec(&value).unwrap();
        if let Some(length) = bytes {
            bytes_out.resize(length, b' ');
        }
        let mut packet = fixtures::blank();
        packet.extend([EXTENSIONS_MARKER, 1.0, PROVENANCE_TAG as f32, bytes_out.len() as f32]);
        packet.extend(bytes_out.into_iter().map(f32::from));
        let error = decode_frame(&packet, 1, &fixtures::catalog()).err().expect("invalid metadata rejected");
        assert!(format!("{error:#}").len() < 8192);
        assert_eq!(error.downcast_ref::<FrameDiagnostic>().unwrap().section, "extensions");
    };
    rejected(serde_json::to_value(provenance()).unwrap(), Some(MAX_PROVENANCE_LANES + 1));
    rejected(serde_json::json!({"version":1,"entries":[],"unknown": "x".repeat(12000)}), None);
    let entries: Vec<_> = (0..=MAX_PROVENANCE_ENTRIES)
        .map(|index| serde_json::json!({"first": index, "end": index + 1, "object":"x"}))
        .collect();
    rejected(serde_json::json!({"version":1,"entries":entries}), None);
    let mut duplicate = fixtures::blank();
    let extension = provenance().extension(duplicate.len()).unwrap();
    duplicate.extend([EXTENSIONS_MARKER, 2.0]);
    for _ in 0..2 {
        duplicate.extend([extension.tag as f32, extension.values.len() as f32]);
        duplicate.extend(&extension.values);
    }
    assert!(decode_frame(&duplicate, 1, &fixtures::catalog()).is_err());
    duplicate[lanes::header::CLEAR] = f32::NAN;
    assert!(
        decode_frame(&duplicate, 1, &fixtures::catalog())
            .err()
            .unwrap()
            .downcast_ref::<FrameDiagnostic>()
            .unwrap()
            .origin
            .is_none()
    );
}

#[test]
fn every_truncation_and_hostile_numeric_mutation_is_safe() {
    for case in fixtures::cases() {
        for end in 0..case.packet.len() {
            let _ = decode_frame(&case.packet[..end], 1, &fixtures::catalog());
        }
        for lane in 0..case.packet.len() {
            for value in [f32::NAN, f32::INFINITY, f32::MAX, -1.0, 0.5] {
                let mut packet = case.packet.clone();
                packet[lane] = value;
                if let Err(error) = decode_frame(&packet, 1, &fixtures::catalog()) {
                    assert!(error.downcast_ref::<FrameDiagnostic>().is_some());
                    assert!(format!("{error:#}").len() < 8192);
                }
            }
        }
    }
}
