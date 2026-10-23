use pen_protocol::Catalog;
use pen_protocol::boundary::*;

pub struct Case {
    pub name: &'static str,
    pub packet: Vec<f32>,
    pub path: &'static str,
    pub lane: usize,
}

pub fn catalog() -> Catalog {
    Catalog { meshes: vec!["mesh".into()], materials: vec!["material".into()], sounds: vec!["sound".into()] }
}

pub fn blank() -> Vec<f32> {
    let mut packet = vec![0.0; PACKET_HEADER_FLOATS];
    packet[0] = PACKET_MAGIC;
    packet[1] = 1.0;
    packet
}

pub fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let mut add = |name, mut packet: Vec<f32>, lane, value, path| {
        assert!(decode_frame(&packet, 1, &catalog()).is_ok(), "{name}: fixture must start valid");
        packet[lane] = value;
        cases.push(Case { name, packet, path, lane });
    };
    add("header", blank(), lanes::header::CLEAR + 2, f32::NAN, "header.clear.b");
    add("matrix", blank(), lanes::header::SHADOW_VIEW + 7, f32::INFINITY, "header.shadow.view[7]");
    add("count", blank(), lanes::header::DRAW_COUNT, MAX_DRAWS as f32 + 1.0, "header.draw_count");
    add("magic", blank(), 0, 123.0, "header.magic");
    let mut draws = blank();
    draws[lanes::header::DRAW_COUNT] = 2.0;
    draws.resize(PACKET_HEADER_FLOATS + DRAW_FLOATS * 2, 0.0);
    add(
        "draw",
        draws.clone(),
        PACKET_HEADER_FLOATS + DRAW_FLOATS + lanes::draw::TRANSFORM + 9,
        f32::NEG_INFINITY,
        "draws[1].transform[9]",
    );
    add("catalog", draws.clone(), PACKET_HEADER_FLOATS + lanes::draw::MESH, 9.0, "draws[0].mesh");
    add("optional", draws.clone(), PACKET_HEADER_FLOATS + lanes::draw::PARAMETERS, 3.0, "draws[0].parameters.count");
    draws[PACKET_HEADER_FLOATS + lanes::draw::EXTRA] = 1.0;
    draws.extend([0.0; 4]);
    add("tail", draws, PACKET_HEADER_FLOATS + DRAW_FLOATS * 2 + 3, f32::NAN, "draws[0].extra[3]");
    let mut insets = blank();
    insets[lanes::header::INSET_COUNT] = 2.0;
    insets[lanes::header::INSET_DRAW_COUNT] = 3.0;
    insets.resize(PACKET_HEADER_FLOATS + INSET_FLOATS * 2 + DRAW_FLOATS * 3, 0.0);
    insets[PACKET_HEADER_FLOATS + lanes::inset::DRAWS] = 1.0;
    insets[PACKET_HEADER_FLOATS + INSET_FLOATS + lanes::inset::DRAWS] = 2.0;
    add(
        "nested",
        insets.clone(),
        PACKET_HEADER_FLOATS + INSET_FLOATS * 2 + DRAW_FLOATS * 2 + lanes::draw::TRANSFORM + 4,
        f32::NAN,
        "insets[1].draws[1].transform[4]",
    );
    let mut solids = blank();
    solids[lanes::header::SOLID_COUNT] = 1.0;
    solids[lanes::header::SOLID_LANES] = (1 + SOLID_FACE_FLOATS * 4) as f32;
    solids.push(4.0);
    solids.extend(vec![0.0; SOLID_FACE_FLOATS * 4]);
    add(
        "faces",
        solids.clone(),
        PACKET_HEADER_FLOATS + 1 + SOLID_FACE_FLOATS * 2 + 1,
        f32::NAN,
        "solids[0].faces[2].normal.y",
    );
    add("faces_count", solids.clone(), PACKET_HEADER_FLOATS, 2.0, "solids[0].faces.count");
    let mut fields = blank();
    fields[lanes::header::FIELD_COUNT] = 1.0;
    fields[lanes::header::FIELD_LANES] = 11.0;
    fields.extend([0.0, 4.0, 4.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
    add("field", fields.clone(), PACKET_HEADER_FLOATS + FIELD_FLOATS + 2, f32::NAN, "fields[0].values[2]");
    add("field_bounds", fields, PACKET_HEADER_FLOATS + lanes::field_patch::COLUMNS, 4.0, "fields[0].columns");
    let mut services = blank();
    services[lanes::header::SERVICE_COUNT] = 1.0;
    services[lanes::header::SERVICE_LANES] = 10.0;
    services.extend([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
    add(
        "service_tag",
        services.clone(),
        PACKET_HEADER_FLOATS + lanes::service_request::SERVICE,
        99.0,
        "services[0].service",
    );
    add(
        "service_values",
        services,
        PACKET_HEADER_FLOATS + SERVICE_REQUEST_FLOATS + 1,
        f32::NAN,
        "services[0].values[1]",
    );
    let mut extensions = blank();
    extensions.extend([EXTENSIONS_MARKER, 1.0, 1.0, 2.0, 0.0, 0.0]);
    add("extension_tag", extensions.clone(), PACKET_HEADER_FLOATS + 2, 65536.0, "extensions[0].tag");
    add("extension_values", extensions, PACKET_HEADER_FLOATS + 5, f32::NAN, "extensions[0].values[1]");
    // A bad structural count takes precedence over an unrelated NaN later.
    let mut corrupt = insets;
    corrupt[lanes::header::INSET_DRAW_COUNT] = f32::NAN;
    corrupt[PACKET_HEADER_FLOATS + INSET_FLOATS * 2 + 3] = f32::NAN;
    cases.push(Case {
        name: "untrusted",
        packet: corrupt,
        path: "header.inset_draw_count",
        lane: lanes::header::INSET_DRAW_COUNT,
    });
    let mut truncated = blank();
    truncated[lanes::header::DRAW_COUNT] = 1.0;
    truncated.extend([0.0; 5]);
    cases.push(Case {
        name: "truncated",
        packet: truncated,
        path: "draws[0].transform[3]",
        lane: PACKET_HEADER_FLOATS + 5,
    });
    let mut oversized = blank();
    oversized.push(0.0);
    cases.push(Case { name: "oversized", packet: oversized, path: "extensions.marker", lane: PACKET_HEADER_FLOATS });
    cases
}
