//! The packets crossing between host and game, as gdev's `boundary.ts` lays
//! them out: frames in, render packets out.
use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::protocol::*;

pub const PACKET_MAGIC: f32 = 0x474446 as f32;
/// Mesh, material, matrix16, parameter count and four parameters, and how
/// many extra vec4s it has in the packet's last section.
pub const DRAW_FLOATS: usize = 24;
/// The most extra vec4s one draw may carry.
pub const MAX_DRAW_EXTRA: usize = 32;
/// Material, matrix16 and opacity.
pub const DECAL_FLOATS: usize = 18;
/// Clip, gain, pan and pitch.
pub const SOUND_FLOATS: usize = 4;
/// x, y, width, height, u0, v0, u1, v1, color4, material and parameters4.
pub const SPRITE_FLOATS: usize = 17;
/// x, y, width, height, view16, projection16, ambient3, light direction3,
/// light color3, draw count and pixel size.
pub const INSET_FLOATS: usize = 47;
/// Preference lanes (packages/preferences.blot).
pub const PREFERENCE_FLOATS: usize = 14;
pub const PACKET_HEADER_FLOATS: usize = 132;
/// Counts travel in float32 lanes: the wire format's exact-integer range.
pub const MAX_STATE_FLOATS: usize = 0xffffff;
const MAX_DRAWS: usize = 100_000;
const MAX_SOLIDS: usize = 100_000;
const MAX_DECALS: usize = 4096;
const MAX_SOUNDS: usize = 256;
const MAX_SPRITES: usize = 65_536;
const MAX_INSETS: usize = 16;
/// Faces of one convex solid; games may use fewer.
pub const MAX_SOLID_FACES: usize = 64;
const MAX_EVENTS: usize = 100_000;

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

pub fn validate_state(state: &[f32]) -> Result<()> {
    ensure!(
        state.len() <= MAX_STATE_FLOATS && state.iter().all(|value| value.is_finite()),
        "game state must be a bounded array of finite float32 values"
    );
    Ok(())
}

pub fn validate_schema(schema: f32) -> Result<u32> {
    ensure!(
        schema.fract() == 0.0 && (1.0..=0xffffff as f32).contains(&schema),
        "state schema must be an integer in 1..16777215"
    );
    Ok(schema as u32)
}

fn key_code(key: &str) -> f32 {
    let named = match key {
        "ArrowLeft" => 0x110100,
        "ArrowRight" => 0x110101,
        "ArrowUp" => 0x110102,
        "ArrowDown" => 0x110103,
        "Tab" => 0x110104,
        "Delete" => 0x110105,
        "Escape" => 0x110106,
        "Backspace" => 0x110107,
        "Enter" => 0x110108,
        "PageUp" => 0x110109,
        "PageDown" => 0x11010a,
        "F2" => 0x110202,
        "F6" => 0x110206,
        "F9" => 0x110209,
        "Shift" => 0x110300,
        "Control" => 0x110301,
        "Alt" => 0x110302,
        "Meta" => 0x110303,
        _ => 0,
    };
    if named != 0 {
        return named as f32;
    }
    // A single UTF-16 code unit, as `charCodeAt(0)` reads it; lowercase ASCII
    // letters read as their capitals.
    let mut units = key.encode_utf16();
    match (units.next(), units.next()) {
        (Some(code), None) => {
            if (97..=122).contains(&code) {
                (code - 32) as f32
            } else {
                code as f32
            }
        }
        _ => 0.0,
    }
}

fn event_record(event: &GameEvent) -> [f32; 5] {
    match event {
        GameEvent::KeyDown(key) => [1.0, key_code(key), 0.0, 0.0, 0.0],
        GameEvent::KeyUp(key) => [2.0, key_code(key), 0.0, 0.0, 0.0],
        GameEvent::PointerDown { button, x, y } => [3.0, *button as f32, *x, *y, 0.0],
        GameEvent::PointerUp { button, x, y } => [4.0, *button as f32, *x, *y, 0.0],
        GameEvent::PointerMove { x, y } => [5.0, 0.0, *x, *y, 0.0],
        GameEvent::Wheel { x, y, delta } => [6.0, 0.0, *x, *y, *delta],
        GameEvent::FocusLost => [7.0, 0.0, 0.0, 0.0, 0.0],
        GameEvent::Console { command, argument } => [9.0, *command, *argument, 0.0, 0.0],
        GameEvent::Preference { index, value } => [10.0, *index, *value, 0.0, 0.0],
        GameEvent::AssetReady { .. } | GameEvent::AssetFailed { .. } => {
            [8.0, 0.0, 0.0, 0.0, 0.0]
        }
    }
}

pub fn encode_frame(frame: &Frame, elapsed: f64, state: &[f32]) -> Result<Vec<f32>> {
    validate_state(state)?;
    ensure!(
        frame.timestamp.is_finite()
            && elapsed.is_finite()
            && elapsed >= 0.0
            && (1..=0xffffff).contains(&frame.viewport.width)
            && (1..=0xffffff).contains(&frame.viewport.height)
            && frame.events.len() <= MAX_EVENTS,
        "invalid frame clock, viewport or event count"
    );
    let mut packet = Vec::with_capacity(5 + frame.events.len() * 5 + state.len());
    packet.extend_from_slice(&[
        elapsed as f32,
        frame.viewport.width as f32,
        frame.viewport.height as f32,
        frame.events.len() as f32,
        state.len() as f32,
    ]);
    for event in &frame.events {
        packet.extend_from_slice(&event_record(event));
    }
    packet.extend_from_slice(state);
    ensure!(
        packet.iter().all(|value| value.is_finite()),
        "frame contains a nonfinite float32 value"
    );
    Ok(packet)
}

/// A decoded render packet and the state lanes it carried.
pub struct Decoded {
    pub state: Vec<f32>,
    pub render: RenderFrame,
}

pub fn decode_frame(packet: &[f32], schema: u32, catalog: &Catalog) -> Result<Decoded> {
    ensure!(
        packet.len() >= PACKET_HEADER_FLOATS
            && packet[0] == PACKET_MAGIC
            && packet[1] == schema as f32,
        "game returned an invalid frame header or state schema"
    );
    let count = |index: usize, maximum: usize| -> Result<usize> {
        let value = packet[index];
        if value.fract() != 0.0 || value < 0.0 || value > maximum as f32 {
            bail!("game returned an invalid frame count or flag");
        }
        Ok(value as usize)
    };
    let flag = |index: usize| -> Result<bool> { Ok(count(index, 1)? == 1) };
    let state_length = count(2, MAX_STATE_FLOATS)?;
    let draw_count = count(3, MAX_DRAWS)?;
    let solid_count = count(4, MAX_SOLIDS)?;
    let solid_lanes = count(5, MAX_SOLIDS * (1 + MAX_SOLID_FACES * 5))?;
    let decal_count = count(6, MAX_DECALS)?;
    let sound_count = count(7, MAX_SOUNDS)?;
    let sprite_count = count(8, MAX_SPRITES)?;
    let inset_count = count(9, MAX_INSETS)?;
    let inset_draw_count = count(10, MAX_DRAWS)?;
    let preference_count = count(11, PREFERENCE_FLOATS)?;
    ensure!(
        preference_count == 0 || preference_count == PREFERENCE_FLOATS,
        "game returned partial preferences"
    );
    let lengths = [
        state_length,
        draw_count * DRAW_FLOATS,
        solid_lanes,
        decal_count * DECAL_FLOATS,
        sound_count * SOUND_FLOATS,
        sprite_count * SPRITE_FLOATS,
        inset_count * INSET_FLOATS,
        inset_draw_count * DRAW_FLOATS,
        preference_count,
    ];
    let fixed = PACKET_HEADER_FLOATS + lengths.iter().sum::<usize>();
    let truncated = "game returned a truncated, oversized or nonfinite frame";
    ensure!(
        packet.len() >= fixed && packet.iter().all(|value| value.is_finite()),
        truncated
    );
    let mut starts = [0usize; 9];
    let mut at = PACKET_HEADER_FLOATS;
    for (start, length) in starts.iter_mut().zip(lengths) {
        *start = at;
        at += length;
    }
    let [
        state_start,
        draw_start,
        solid_start,
        decal_start,
        sound_start,
        sprite_start,
        inset_start,
        inset_draw_start,
        preference_start,
    ] = starts;

    // The draws' extra vectors follow, in draw order: the scene's draws, then the insets'.
    let mut extra_lanes = 0usize;
    let mut extras = Vec::with_capacity(draw_count + inset_draw_count);
    for draw in (0..draw_count)
        .map(|index| draw_start + index * DRAW_FLOATS)
        .chain((0..inset_draw_count).map(|index| inset_draw_start + index * DRAW_FLOATS))
    {
        let own = count(draw + 23, MAX_DRAW_EXTRA)? * 4;
        let first = fixed + extra_lanes;
        extra_lanes += own;
        extras.push((own != 0).then_some((first, own)));
    }
    ensure!(packet.len() == fixed + extra_lanes, truncated);

    let vector = |index: usize| Vector {
        x: packet[index],
        y: packet[index + 1],
        z: packet[index + 2],
    };
    let matrix = |index: usize| -> Matrix { packet[index..index + 16].try_into().unwrap() };
    let material = |index: usize| -> Result<u32> {
        Ok(count(index, catalog.materials.len().saturating_sub(1))? as u32)
    };
    let draw = |index: usize, extra: Option<(usize, usize)>| -> Result<Draw> {
        let mesh = count(index, catalog.meshes.len().saturating_sub(1))? as u32;
        let parameter_count = count(index + 18, 4)?;
        ensure!(
            parameter_count == 0 || parameter_count == 4,
            "draw parameters must contain zero or four lanes"
        );
        Ok(Draw {
            mesh,
            material: material(index + 1)?,
            transform: matrix(index + 2),
            parameters: (parameter_count == 4)
                .then(|| packet[index + 19..index + 23].try_into().unwrap()),
            extra: extra.map(|(first, own)| packet[first..first + own].to_vec()),
        })
    };

    let draws = (0..draw_count)
        .map(|index| draw(draw_start + index * DRAW_FLOATS, extras[index]))
        .collect::<Result<Vec<_>>>()?;

    // Each solid is a face count, then normal3, distance, material per face.
    let mut solids = Vec::with_capacity(solid_count);
    let mut index = solid_start;
    for _ in 0..solid_count {
        ensure!(index < decal_start, "game returned truncated solids");
        let faces = count(index, MAX_SOLID_FACES)?;
        ensure!(
            faces >= 4 && index + 1 + faces * 5 <= decal_start,
            "game returned an invalid solid"
        );
        solids.push(Solid {
            faces: (0..faces)
                .map(|face| {
                    let lane = index + 1 + face * 5;
                    Ok(SolidFace {
                        normal: vector(lane),
                        distance: packet[lane + 3],
                        material: material(lane + 4)?,
                    })
                })
                .collect::<Result<_>>()?,
        });
        index += 1 + faces * 5;
    }
    ensure!(index == decal_start, "game returned solid lanes that do not match");

    let mut inset_draw = inset_draw_start;
    let mut insets = Vec::with_capacity(inset_count);
    for inset in 0..inset_count {
        let lane = inset_start + inset * INSET_FLOATS;
        let own = count(lane + 45, MAX_DRAWS)?;
        let first = inset_draw;
        let first_index = (first - inset_draw_start) / DRAW_FLOATS;
        inset_draw += own * DRAW_FLOATS;
        ensure!(
            inset_draw <= preference_start,
            "game returned more inset draws than it counted"
        );
        insets.push(Inset {
            x: packet[lane],
            y: packet[lane + 1],
            width: packet[lane + 2],
            height: packet[lane + 3],
            pixel_size: packet[lane + 46],
            view: matrix(lane + 4),
            projection: matrix(lane + 20),
            ambient: vector(lane + 36),
            light_direction: vector(lane + 39),
            light_color: vector(lane + 42),
            draws: (0..own)
                .map(|item| {
                    draw(first + item * DRAW_FLOATS, extras[draw_count + first_index + item])
                })
                .collect::<Result<_>>()?,
        });
    }
    ensure!(
        inset_draw == preference_start,
        "game returned inset draws that do not match"
    );

    let ambience = count(124, catalog.sounds.len())?;
    let sound = |index: usize| -> Result<u32> {
        ensure!(
            !catalog.sounds.is_empty(),
            "game played a sound with no sounds in the catalog"
        );
        Ok(count(index, catalog.sounds.len() - 1)? as u32)
    };
    let render = RenderFrame {
        view: matrix(20),
        projection: matrix(36),
        clear: Color { r: packet[52], g: packet[53], b: packet[54], a: packet[55] },
        ambient: vector(56),
        light_direction: vector(59),
        light_color: vector(62),
        aberration: Aberration { x: packet[65], y: packet[66] },
        blur: packet[126],
        hide_cursor: flag(131)?,
        quality: Quality {
            msaa: if packet[67] == 4.0 { 4 } else { 1 },
            fxaa: flag(68)?,
            scale: packet[69],
            filtering: count(70, 16)? as u32,
            occlusion: flag(71)?,
            sky: flag(72)?,
            effects: flag(73)?,
            bloom: flag(127)?,
        },
        sky: Sky {
            zenith: vector(74),
            horizon: vector(77),
            ground: vector(80),
            sun: vector(83),
            night: packet[86],
        },
        fog: Fog { radius: packet[87], density: packet[88], height: packet[89] },
        clouds: Clouds {
            cover: packet[128],
            drift: Drift { x: packet[129], z: packet[130] },
        },
        shadow: Shadow {
            view: matrix(90),
            projection: matrix(106),
            texel: packet[122],
            texels: count(123, 8192)? as u32,
        },
        draws,
        solids,
        decals: (0..decal_count)
            .map(|decal| {
                let lane = decal_start + decal * DECAL_FLOATS;
                Ok(Decal {
                    material: material(lane)?,
                    transform: matrix(lane + 1),
                    opacity: packet[lane + 17],
                })
            })
            .collect::<Result<_>>()?,
        sounds: (0..sound_count)
            .map(|played| {
                let lane = sound_start + played * SOUND_FLOATS;
                Ok(Sound {
                    clip: sound(lane)?,
                    gain: packet[lane + 1],
                    pan: packet[lane + 2],
                    pitch: packet[lane + 3],
                })
            })
            .collect::<Result<_>>()?,
        ambience: (ambience != 0).then(|| Ambience {
            clip: (ambience - 1) as u32,
            gain: packet[125],
        }),
        sprites: (0..sprite_count)
            .map(|sprite| {
                let lane = sprite_start + sprite * SPRITE_FLOATS;
                Ok(Sprite {
                    x: packet[lane],
                    y: packet[lane + 1],
                    width: packet[lane + 2],
                    height: packet[lane + 3],
                    u0: packet[lane + 4],
                    v0: packet[lane + 5],
                    u1: packet[lane + 6],
                    v1: packet[lane + 7],
                    color: packet[lane + 8..lane + 12].try_into().unwrap(),
                    material: material(lane + 12)?,
                    parameters: packet[lane + 13..lane + 17].try_into().unwrap(),
                })
            })
            .collect::<Result<_>>()?,
        insets,
        preferences: (preference_count != 0)
            .then(|| packet[preference_start..preference_start + preference_count].to_vec()),
        save_requested: flag(12)?,
        load_requested: flag(13)?,
        scene_save_requested: flag(14)?,
        scene_load_requested: flag(15)?,
        checkpoint_requested: flag(16)?,
        undo_requested: flag(17)?,
        redo_requested: flag(18)?,
        exit_requested: flag(19)?,
    };
    Ok(Decoded {
        state: packet[state_start..state_start + state_length].to_vec(),
        render,
    })
}

/// Reads `catalog.json` from an asset directory.
pub fn read_catalog(assets: &std::path::Path) -> Result<Catalog> {
    let path = assets.join("catalog.json");
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    parse_catalog(&serde_json::from_str(&text)?)
}
