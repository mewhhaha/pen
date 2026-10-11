//! The packets crossing between host and game, as gdev's `boundary.ts` lays
//! them out: frames in, render packets out. Their layout, the frame encoder
//! and the render packet decoder are generated from gdev's protocol/schema.json
//! into packets.rs (`just sync-protocol`); this adds reading the asset catalog.
use anyhow::{Context, Result, anyhow};

pub use crate::packets::codec::*;
use crate::protocol::*;

pub fn parse_catalog(value: &serde_json::Value) -> Result<Catalog> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("asset catalog must be an object"))?;
    let paths = |kind: &str, empty: bool| -> Result<Vec<String>> {
        let entries = object.get(kind).and_then(|entries| entries.as_array());
        let invalid = || anyhow!("asset catalog {kind} must contain relative paths");
        let entries = entries.ok_or_else(invalid)?;
        if entries.is_empty() && !empty {
            return Err(invalid());
        }
        entries
            .iter()
            .map(|entry| {
                let path = entry.as_str().ok_or_else(invalid)?;
                if path.is_empty()
                    || path.starts_with('/')
                    || path.split(['/', '\\']).any(|part| part == "..")
                {
                    return Err(invalid());
                }
                Ok(path.to_owned())
            })
            .collect()
    };
    Ok(Catalog {
        meshes: paths("meshes", false)?,
        materials: paths("materials", false)?,
        sounds: if object.contains_key("sounds") { paths("sounds", true)? } else { Vec::new() },
    })
}

/// Reads `catalog.json` from an asset directory.
pub fn read_catalog(assets: &std::path::Path) -> Result<Catalog> {
    let path = assets.join("catalog.json");
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    parse_catalog(&serde_json::from_str(&text)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_encode_as_the_browser_host_does() {
        let frame = Frame {
            timestamp: 1.0,
            viewport: Viewport { width: 640, height: 480 },
            events: vec![
                GameEvent::KeyDown("a".into()),
                GameEvent::KeyUp("ArrowLeft".into()),
                GameEvent::PointerDown { button: 2, x: 3.0, y: 4.0 },
                GameEvent::Wheel { x: 1.0, y: 2.0, delta: -100.0 },
            ],
        };
        let packet = encode_frame(&frame, 0.5, &[7.0]).unwrap();
        assert_eq!(
            packet,
            [
                0.5, 640.0, 480.0, 4.0, 1.0, //
                1.0, 65.0, 0.0, 0.0, 0.0, //
                2.0, 0x110100 as f32, 0.0, 0.0, 0.0, //
                3.0, 2.0, 3.0, 4.0, 0.0, //
                6.0, 0.0, 1.0, 2.0, -100.0, //
                7.0,
            ]
        );
        assert!(encode_frame(&frame, -1.0, &[]).is_err());
    }

    #[test]
    fn events_travel_with_their_schema_tags() {
        let events = [
            (GameEvent::Console { command: 4.0, argument: 600.0 }, [event_tags::CONSOLE as f32, 4.0, 600.0, 0.0, 0.0]),
            (GameEvent::Preference { index: 9.0, value: 0.75 }, [event_tags::PREFERENCE as f32, 9.0, 0.75, 0.0, 0.0]),
            (GameEvent::Tune { index: 3, value: 0.25 }, [event_tags::TUNE as f32, 3.0, 0.25, 0.0, 0.0]),
            (GameEvent::FocusLost, [event_tags::FOCUS_LOST as f32, 0.0, 0.0, 0.0, 0.0]),
            (
                GameEvent::AssetFailed { key: "k".into(), kind: "mesh".into(), revision: 2, message: "m".into() },
                [event_tags::ASSET_CHANGED as f32, 0.0, 0.0, 0.0, 0.0],
            ),
        ];
        for (event, lanes) in events {
            let frame = Frame { timestamp: 0.0, viewport: Viewport { width: 1, height: 1 }, events: vec![event] };
            let packet = encode_frame(&frame, 0.0, &[]).unwrap();
            assert_eq!(packet[FRAME_HEADER_FLOATS..], lanes);
        }
        assert_eq!(event_tags::TUNE, 11);
    }

    #[test]
    fn rejects_truncated_packets() {
        let catalog = Catalog { meshes: vec!["m".into()], materials: vec!["x".into()], sounds: vec![] };
        let mut packet = vec![0.0; PACKET_HEADER_FLOATS];
        packet[0] = PACKET_MAGIC;
        packet[1] = 3.0;
        assert!(decode_frame(&packet, 3, &catalog).is_ok());
        assert!(decode_frame(&packet, 4, &catalog).is_err());
        packet[3] = 1.0; // one draw, but no lanes for it
        assert!(decode_frame(&packet, 3, &catalog).is_err());
    }

    #[test]
    fn field_patches_follow_the_preferences() {
        let catalog = Catalog { meshes: vec!["m".into()], materials: vec!["a".into(), "b".into()], sounds: vec![] };
        let mut packet = vec![0.0; PACKET_HEADER_FLOATS];
        packet[0] = PACKET_MAGIC;
        packet[1] = 3.0;
        packet[lanes::header::PREFERENCE_COUNT] = PREFERENCE_FLOATS as f32;
        packet.extend((0..PREFERENCE_FLOATS).map(|lane| lane as f32));
        // Material 1's 4 x 3 field, its 2 x 2 patch at column 1, row 1; then
        // an empty patch of material 0's 8 x 8 field.
        let patches = [vec![1.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0, 0.5, 1.5, 2.5, 3.5], vec![0.0, 8.0, 8.0, 8.0, 0.0, 0.0, 8.0]];
        packet[lanes::header::FIELD_COUNT] = 2.0;
        packet[lanes::header::FIELD_LANES] = patches.iter().map(Vec::len).sum::<usize>() as f32;
        packet.extend(patches.concat());
        let decoded = decode_frame(&packet, 3, &catalog).unwrap();
        assert_eq!(decoded.render.preferences.as_deref(), Some(&packet[PACKET_HEADER_FLOATS..PACKET_HEADER_FLOATS + PREFERENCE_FLOATS]));
        assert_eq!(
            decoded.render.fields,
            [
                FieldPatch { material: 1, width: 4, height: 3, x: 1, z: 1, columns: 2, rows: 2, values: vec![0.5, 1.5, 2.5, 3.5] },
                FieldPatch { material: 0, width: 8, height: 8, x: 8, z: 0, columns: 0, rows: 8, values: vec![] },
            ]
        );
        // A patch past its field's edge, and lanes that do not add up.
        let mut outside = packet.clone();
        outside[PACKET_HEADER_FLOATS + PREFERENCE_FLOATS + 5] = 4.0;
        assert!(decode_frame(&outside, 3, &catalog).is_err());
        let mut short = packet.clone();
        short[lanes::header::FIELD_LANES] -= 1.0;
        short.pop();
        assert!(decode_frame(&short, 3, &catalog).is_err());
    }

    /// The interface's nodes: parents by place, their text one run of code
    /// points the lengths split, and lengths that do not add up refused.
    #[test]
    fn interface_nodes_follow_the_fields() {
        let catalog = Catalog { meshes: vec!["m".into()], materials: vec!["x".into()], sounds: vec![] };
        let mut packet = vec![0.0; PACKET_HEADER_FLOATS];
        packet[0] = PACKET_MAGIC;
        packet[1] = 3.0;
        let node = |id: f32, parent: f32, role: u32, text: &str, name: usize, description: usize| {
            let mut lanes = vec![0.0; UI_NODE_FLOATS];
            lanes[lanes::ui_node::ID] = id;
            lanes[lanes::ui_node::PARENT] = parent;
            lanes[lanes::ui_node::ROLE] = role as f32;
            lanes[lanes::ui_node::WIDTH] = 40.0;
            lanes[lanes::ui_node::FOCUSABLE] = 1.0;
            lanes[lanes::ui_node::CHECKED] = ui_state::TRUE as f32;
            lanes[lanes::ui_node::ACTIONS] = ((1 << ui_action::FOCUS) | (1 << ui_action::ACTIVATE)) as f32;
            let code_points: Vec<f32> = text.chars().map(|character| character as u32 as f32).collect();
            lanes[lanes::ui_node::TEXT_LENGTH] = code_points.len() as f32;
            lanes[lanes::ui_node::NAME_LENGTH] = name as f32;
            lanes[lanes::ui_node::DESCRIPTION_LENGTH] = description as f32;
            lanes.extend(code_points);
            lanes
        };
        let nodes = [node(0x40001 as f32, 0.0, ui_role::DIALOG, "Paused", 6, 0), node(0x40100 as f32, 1.0, ui_role::BUTTON, "Go×now", 2, 1)];
        packet[lanes::header::UI_COUNT] = 2.0;
        packet[lanes::header::UI_LANES] = nodes.iter().map(Vec::len).sum::<usize>() as f32;
        packet.extend(nodes.concat());
        let decoded = decode_frame(&packet, 3, &catalog).unwrap();
        let ui = &decoded.render.ui;
        assert_eq!(ui.len(), 2);
        assert_eq!((ui[0].id, ui[0].parent, ui[0].role), (0x40001, 0, ui_role::DIALOG));
        assert_eq!((ui[1].id, ui[1].parent, ui[1].role, ui[1].checked), (0x40100, 1, ui_role::BUTTON, ui_state::TRUE));
        assert_eq!((ui[1].name_length, ui[1].description_length, ui[1].text_length), (2, 1, 6));
        let text: String = ui[1].text.iter().map(|code| char::from_u32(*code as u32).unwrap()).collect();
        assert_eq!(text, "Go×now");
        assert!(ui[1].focusable && ui[1].actions & (1 << ui_action::ACTIVATE) != 0);
        assert_eq!(ui_role::NAMES[ui[0].role as usize], "dialog");
        // A name longer than the text, and a description past it.
        let mut long = packet.clone();
        long[PACKET_HEADER_FLOATS + lanes::ui_node::NAME_LENGTH] = 7.0;
        assert!(decode_frame(&long, 3, &catalog).is_err());
        let mut past = packet.clone();
        past[PACKET_HEADER_FLOATS + UI_NODE_FLOATS + 6 + lanes::ui_node::DESCRIPTION_LENGTH] = 5.0;
        assert!(decode_frame(&past, 3, &catalog).is_err());
    }

    /// Every packet a headless run wrote decodes (`just packets` first).
    #[test]
    fn real_packets_decode() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dist");
        let (Ok(catalog), Ok(schema)) = (
            read_catalog(&root.join("gdev/assets")),
            std::fs::read_to_string(root.join("packets/schema.txt")),
        ) else {
            return;
        };
        let schema: u32 = schema.trim().parse().unwrap();
        for entry in std::fs::read_dir(root.join("packets")).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|extension| extension != "f32") {
                continue;
            }
            let bytes = std::fs::read(&path).unwrap();
            let lanes: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect();
            let decoded = decode_frame(&lanes, schema, &catalog).unwrap();
            assert!(!decoded.render.draws.is_empty(), "{} has no draws", path.display());
        }
    }
}
