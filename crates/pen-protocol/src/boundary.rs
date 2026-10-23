//! The packets crossing between host and game: frames in, render packets
//! out. Their layout, the frame encoder and the render packet decoder are
//! generated from protocol/schema.json into packets.rs (`just
//! sync-protocol`); this adds reading the asset catalog, and the numbers
//! in a packet's lanes.
use anyhow::{Context, Result, anyhow};

pub use crate::packets::codec::*;
use crate::protocol::*;

/// The whole number a lane holds (an id, count, index or code point), or
/// None for a fraction, a negative, a non-finite value or one past u32.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "checked whole and in range")]
pub fn whole(lane: f32) -> Option<u32> {
    (lane.fract() == 0.0 && (0.0..4_294_967_296.0).contains(&lane)).then_some(lane as u32)
}

/// A game's catalog: its `meshes` and `materials`, and `sounds` if it has
/// any, each a list of relative paths, which may be empty (a game that
/// draws only sprites has no meshes).
pub fn parse_catalog(value: &serde_json::Value) -> Result<Catalog> {
    let object = value.as_object().ok_or_else(|| anyhow!("asset catalog must be an object"))?;
    let paths = |kind: &str| -> Result<Vec<String>> {
        let entries = object.get(kind).and_then(|entries| entries.as_array());
        let invalid = || anyhow!("asset catalog {kind} must contain relative paths");
        let entries = entries.ok_or_else(invalid)?;
        entries
            .iter()
            .map(|entry| {
                let path = entry.as_str().ok_or_else(invalid)?;
                if path.is_empty() || path.starts_with('/') || path.split(['/', '\\']).any(|part| part == "..") {
                    return Err(invalid());
                }
                Ok(path.to_owned())
            })
            .collect()
    };
    Ok(Catalog {
        meshes: paths("meshes")?,
        materials: paths("materials")?,
        sounds: if object.contains_key("sounds") { paths("sounds")? } else { Vec::new() },
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
    fn catalogs_list_relative_paths_or_none() {
        let parse = |text: &str| parse_catalog(&serde_json::from_str(text).unwrap());
        let sprites = parse(r#"{"meshes": [], "materials": ["ui.material.json"]}"#).unwrap();
        assert_eq!((sprites.meshes.len(), sprites.materials.len(), sprites.sounds.len()), (0, 1, 0));
        assert!(parse(r#"{"meshes": [], "materials": [], "sounds": []}"#).is_ok());
        for wrong in [
            r#"{"materials": []}"#,
            r#"{"meshes": "cube.mesh.json", "materials": []}"#,
            r#"{"meshes": ["../cube.mesh.json"], "materials": []}"#,
            r#"{"meshes": ["/cube.mesh.json"], "materials": []}"#,
            r#"{"meshes": [""], "materials": []}"#,
            r#"[]"#,
        ] {
            assert!(parse(wrong).is_err(), "{wrong}");
        }
    }

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
                0.5,
                640.0,
                480.0,
                4.0,
                1.0, //
                1.0,
                65.0,
                0.0,
                0.0,
                0.0, //
                2.0,
                0x110100 as f32,
                0.0,
                0.0,
                0.0, //
                3.0,
                2.0,
                3.0,
                4.0,
                0.0, //
                6.0,
                0.0,
                1.0,
                2.0,
                -100.0, //
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
    fn lanes_hold_whole_numbers() {
        assert_eq!(
            [0.0, 7.0, 16_777_216.0, 4_294_967_040.0].map(whole),
            [Some(0), Some(7), Some(16_777_216), Some(4_294_967_040)]
        );
        let not = [-1.0, 0.5, 4_294_967_296.0, 1e30, f32::INFINITY, f32::NAN];
        assert!(not.into_iter().all(|lane| whole(lane).is_none()));
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
        let patches =
            [vec![1.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0, 0.5, 1.5, 2.5, 3.5], vec![0.0, 8.0, 8.0, 8.0, 0.0, 0.0, 8.0]];
        packet[lanes::header::FIELD_COUNT] = 2.0;
        packet[lanes::header::FIELD_LANES] = patches.iter().map(Vec::len).sum::<usize>() as f32;
        packet.extend(patches.concat());
        let decoded = decode_frame(&packet, 3, &catalog).unwrap();
        assert_eq!(
            decoded.render.preferences.as_deref(),
            Some(&packet[PACKET_HEADER_FLOATS..PACKET_HEADER_FLOATS + PREFERENCE_FLOATS])
        );
        assert_eq!(
            decoded.render.fields,
            [
                FieldPatch {
                    material: 1,
                    width: 4,
                    height: 3,
                    x: 1,
                    z: 1,
                    columns: 2,
                    rows: 2,
                    values: vec![0.5, 1.5, 2.5, 3.5]
                },
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

    /// Pose layers and particle emitters follow the voices, each in its
    /// record's lanes; a material the catalog lacks is refused.
    #[test]
    fn poses_and_emitters_follow_the_voices() {
        let catalog = Catalog { meshes: vec!["m".into()], materials: vec!["a".into(), "b".into()], sounds: vec![] };
        let mut packet = vec![0.0; PACKET_HEADER_FLOATS];
        packet[0] = PACKET_MAGIC;
        packet[1] = 3.0;
        packet[lanes::header::POSE_COUNT] = 2.0;
        packet[lanes::header::EMITTER_COUNT] = 1.0;
        packet.extend([0.0, 1.0, 0.25, 1.0, 0.0, 0.0, 0.5, 0.5]);
        let mut emitter = vec![0.0; EMITTER_FLOATS];
        emitter[lanes::emitter::MATERIAL] = 1.0;
        emitter[lanes::emitter::TRANSFORM..lanes::emitter::TRANSFORM + 16]
            .copy_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 1.0]);
        emitter[lanes::emitter::SEED] = 9.0;
        emitter[lanes::emitter::RATE] = 30.0;
        emitter[lanes::emitter::GRAVITY + 1] = -9.8;
        emitter[lanes::emitter::COLOR_END + 3] = 0.5;
        emitter[lanes::emitter::ADDITIVE] = 1.0;
        packet.extend(&emitter);
        let render = decode_frame(&packet, 3, &catalog).unwrap().render;
        assert_eq!(
            render.poses,
            [Pose { draw: 0, clip: 1, time: 0.25, weight: 1.0 }, Pose { draw: 0, clip: 0, time: 0.5, weight: 0.5 }]
        );
        let decoded = &render.emitters[0];
        assert_eq!((decoded.material, decoded.seed, decoded.rate, decoded.additive), (1, 9, 30.0, true));
        assert_eq!((decoded.transform[12], decoded.gravity.y, decoded.color_end.a), (2.0, -9.8, 0.5));
        let mut unknown = packet.clone();
        unknown[PACKET_HEADER_FLOATS + 8 + lanes::emitter::MATERIAL] = 2.0;
        assert!(decode_frame(&unknown, 3, &catalog).is_err());
    }

    /// Decoding into a frame that held another replaces all of it, and the
    /// state comes back where it lies in the packet.
    #[test]
    fn decoding_into_a_frame_replaces_what_it_held() {
        let catalog =
            Catalog { meshes: vec!["m".into(), "n".into()], materials: vec!["a".into(), "b".into()], sounds: vec![] };
        let packet = |state: &[f32], extras: &[usize], preferences: bool, field: &[f32]| {
            let mut packet = vec![0.0; PACKET_HEADER_FLOATS];
            packet[0] = PACKET_MAGIC;
            packet[1] = 3.0;
            packet[lanes::header::STATE_COUNT] = state.len() as f32;
            packet[lanes::header::DRAW_COUNT] = extras.len() as f32;
            packet.extend(state);
            for (index, extra) in extras.iter().enumerate() {
                let mut draw = vec![index as f32; DRAW_FLOATS];
                draw[lanes::draw::PARAMETERS] = 0.0;
                draw[lanes::draw::EXTRA] = *extra as f32;
                packet.extend(draw);
            }
            if preferences {
                packet[lanes::header::PREFERENCE_COUNT] = PREFERENCE_FLOATS as f32;
                packet.extend((0..PREFERENCE_FLOATS).map(|lane| lane as f32));
            }
            if !field.is_empty() {
                packet[lanes::header::FIELD_COUNT] = 1.0;
                packet[lanes::header::FIELD_LANES] = field.len() as f32;
                packet.extend(field);
            }
            packet.extend((0..extras.iter().sum::<usize>() * 4).map(|lane| lane as f32 + 0.5));
            packet
        };
        let first = packet(&[7.0, 8.0], &[0, 2], true, &[1.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0, 0.5, 1.5, 2.5, 3.5]);
        let second = packet(&[], &[1], false, &[]);
        let mut frame = RenderFrame::default();
        assert_eq!(decode_into(&first, 3, &catalog, &mut frame).unwrap(), [7.0, 8.0]);
        assert_eq!(frame, decode_frame(&first, 3, &catalog).unwrap().render);
        assert_eq!(frame.draws[1].extra.as_ref().map(Vec::len), Some(8));
        assert!(decode_into(&second, 3, &catalog, &mut frame).unwrap().is_empty());
        assert_eq!(frame, decode_frame(&second, 3, &catalog).unwrap().render);
        assert_eq!(frame.draws[0].extra.as_deref(), Some(&[0.5, 1.5, 2.5, 3.5][..]));
        decode_into(&first, 3, &catalog, &mut frame).unwrap();
        assert_eq!(frame, decode_frame(&first, 3, &catalog).unwrap().render);
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
        let nodes = [
            node(0x40001 as f32, 0.0, ui_role::DIALOG, "Paused", 6, 0),
            node(0x40100 as f32, 1.0, ui_role::BUTTON, "Go×now", 2, 1),
        ];
        packet[lanes::header::UI_COUNT] = 2.0;
        packet[lanes::header::UI_LANES] = nodes.iter().map(Vec::len).sum::<usize>() as f32;
        packet.extend(nodes.concat());
        let decoded = decode_frame(&packet, 3, &catalog).unwrap();
        let ui = &decoded.render.ui;
        assert_eq!(ui.len(), 2);
        assert_eq!((ui[0].id, ui[0].parent, ui[0].role), (0x40001, 0, ui_role::DIALOG));
        assert_eq!((ui[1].id, ui[1].parent, ui[1].role, ui[1].checked), (0x40100, 1, ui_role::BUTTON, ui_state::TRUE));
        assert_eq!((ui[1].name_length, ui[1].description_length, ui[1].text_length), (2, 1, 6));
        let text: Vec<f32> = "Go×now".chars().map(|character| u32::from(character) as f32).collect();
        assert_eq!(ui[1].text, text);
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
        let (Ok(catalog), Ok(schema)) =
            (read_catalog(&root.join("gdev/assets")), std::fs::read_to_string(root.join("packets/schema.txt")))
        else {
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

    /// The game's own sections come after everything else, the draws'
    /// extra lanes included, and a packet without them ends before them.
    #[test]
    fn the_games_own_sections_follow_the_packet() {
        let catalog = Catalog { meshes: vec!["m".into()], materials: vec!["x".into()], sounds: vec![] };
        let mut packet = vec![0.0; PACKET_HEADER_FLOATS];
        packet[0] = PACKET_MAGIC;
        packet[1] = 3.0;
        packet[lanes::header::DRAW_COUNT] = 1.0;
        let mut draw = vec![0.0; DRAW_FLOATS];
        draw[lanes::draw::TRANSFORM] = 1.0;
        draw[lanes::draw::EXTRA] = 2.0;
        packet.extend(draw);
        packet.extend((0..8).map(|lane| lane as f32));
        let plain = packet.clone();
        packet.extend([EXTENSIONS_MARKER, 2.0, 1.0, 2.0, 1.5, 2.5, 7.0, 0.0]);
        let mut frame = decode_frame(&packet, 3, &catalog).unwrap().render;
        assert_eq!(frame.draws[0].extra.as_ref().map(Vec::len), Some(8));
        assert_eq!(
            frame.extensions,
            [Extension { tag: 1, values: vec![1.5, 2.5] }, Extension { tag: 7, values: vec![] }]
        );
        // One without them leaves none behind.
        decode_into(&plain, 3, &catalog, &mut frame).unwrap();
        assert!(frame.extensions.is_empty());
        let block = PACKET_HEADER_FLOATS + DRAW_FLOATS + 8;
        let changed = |at: usize, value: f32| {
            let mut changed = packet.clone();
            changed[block + at] = value;
            decode_frame(&changed, 3, &catalog).map(|decoded| decoded.render)
        };
        // Lanes that are no block make an oversized packet, as they did.
        let oversized = "game returned a truncated, oversized or nonfinite frame";
        assert_eq!(changed(0, 1.0).unwrap_err().root_cause().to_string(), oversized, "the marker");
        assert_eq!(
            changed(2, 0.0).unwrap_err().root_cause().to_string(),
            "invalid development provenance",
            "tag 0 requires provenance"
        );
        assert!(changed(2, MAX_EXTENSION_TAG as f32 + 1.0).is_err(), "a tag past the most");
        assert!(changed(1, MAX_EXTENSIONS as f32 + 1.0).is_err(), "too many sections");
        assert!(changed(1, 3.0).is_err(), "a section the packet lacks");
        assert!(changed(3, 3.0).is_err(), "lanes the packet lacks");
        let mut longer = packet.clone();
        longer.push(0.0);
        assert!(decode_frame(&longer, 3, &catalog).is_err(), "lanes after the last section");
    }
}
