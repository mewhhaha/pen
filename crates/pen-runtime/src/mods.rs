//! Mods: extra guest programs that run beside the game, each in a sandbox
//! of its own and with only the capabilities its manifest asks for
//! (docs/mods.md). A mod is a directory holding `mod.json`, its manifest,
//! and `mod.wasm`, a module of the guest ABI (blot2's
//! `compiler/guest-abi.md`) exporting `frame: Array F32 -> Array F32` and
//! importing nothing: it reaches nothing but what each call hands it.
//!
//! Each frame the player plays live, the host calls every mod with the
//! frame (and, with `observe`, its events) and splices the events a mod
//! with `input` answers into the frame's input, after the player's. The
//! game cannot tell them from the player's, and input recordings keep them,
//! so replays, rewinds and reloads hold without the mods. A mod that asks
//! for what it was not granted, answers malformed, traps, or runs out of
//! time or memory is stopped; the game plays on.
//!
//! This is the protocol and its checks, which native pen and the browser
//! share; the sandboxes are the hosts'.
use anyhow::{Result, anyhow, bail, ensure};
use pen_protocol::boundary::{EVENT_FLOATS, FRAME_HEADER_FLOATS, event_tags, whole};

use crate::abi::{GuestAbi, Parameter, ValueType};

/// The export a host calls each frame.
pub const MOD_ENTRY: &str = "frame";
/// The version of the arrays a mod and the host exchange.
pub const MOD_PROTOCOL: f32 = 1.0;
/// The most events a mod may add to one frame.
pub const MAX_MOD_EVENTS: usize = 64;

/// What a mod may do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Capability {
    /// See the frame's events: the player's input as the game gets it.
    Observe,
    /// Add input events to the frame: keys, the pointer, the wheel and
    /// console commands, as a player could.
    Input,
}

impl Capability {
    fn named(name: &str) -> Result<Capability> {
        Ok(match name {
            "observe" => Capability::Observe,
            "input" => Capability::Input,
            _ => bail!("unknown capability `{name}` (a mod may ask for observe and input)"),
        })
    }
}

/// A mod's `mod.json`: `{"name": "sticky-arrows", "capabilities":
/// ["observe", "input"]}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModManifest {
    pub name: String,
    pub capabilities: Vec<Capability>,
}

impl ModManifest {
    pub fn parse(text: &str) -> Result<ModManifest> {
        let json: serde_json::Value = serde_json::from_str(text)?;
        let name = json.get("name").and_then(serde_json::Value::as_str).ok_or_else(|| anyhow!("a mod needs a name"))?;
        ensure!(
            !name.is_empty() && name.len() <= 64 && name.chars().all(|c| c.is_alphanumeric() || "-_ ".contains(c)),
            "`{name}` is not a mod's name: up to 64 letters, digits, spaces, `-` and `_`"
        );
        let mut capabilities = Vec::new();
        for capability in json.get("capabilities").and_then(serde_json::Value::as_array).into_iter().flatten() {
            let capability = capability.as_str().ok_or_else(|| anyhow!("a capability is a name"))?;
            capabilities.push(Capability::named(capability)?);
        }
        capabilities.sort();
        capabilities.dedup();
        Ok(ModManifest { name: name.to_owned(), capabilities })
    }

    pub fn may(&self, capability: Capability) -> bool {
        self.capabilities.contains(&capability)
    }
}

/// Checks a mod's module: it exports [`MOD_ENTRY`], an `Array F32 ->
/// Array F32` that takes no host callback.
pub fn check_mod(abi: &GuestAbi) -> Result<()> {
    let entry = abi.function(MOD_ENTRY).ok_or_else(|| anyhow!("a mod must export {MOD_ENTRY}"))?;
    ensure!(
        entry.parameter == Parameter::Value(ValueType::ArrayF32) && entry.result == ValueType::ArrayF32,
        "a mod must export {MOD_ENTRY}: Array F32 -> Array F32"
    );
    ensure!(abi.callbacks().is_empty(), "a mod takes no host callbacks");
    Ok(())
}

/// What a mod is called with for frame `frame`, whose input (pen-protocol's
/// `encode_frame` lanes) is `input`: the protocol's version, the frame's
/// number, elapsed seconds, width and height, then its events (with
/// `observe`; none without).
pub fn request(manifest: &ModManifest, frame: u64, input: &[f32]) -> Vec<f32> {
    let [elapsed, width, height, ..] = input[..] else { return vec![MOD_PROTOCOL, frame as f32, 0.0, 0.0, 0.0, 0.0] };
    let events = if manifest.may(Capability::Observe) { frame_events(input) } else { &[] };
    let mut request = Vec::with_capacity(6 + events.len());
    request.extend([MOD_PROTOCOL, frame as f32, elapsed, width, height, (events.len() / EVENT_FLOATS) as f32]);
    request.extend_from_slice(events);
    request
}

/// An input packet's events' lanes.
fn frame_events(input: &[f32]) -> &[f32] {
    let count = input.get(3).copied().and_then(whole).map_or(0, |count| count as usize);
    input.get(FRAME_HEADER_FLOATS..FRAME_HEADER_FLOATS + count * EVENT_FLOATS).unwrap_or(&[])
}

/// The events a mod's answer adds, once checked against what it may do:
/// the version, a count, then each event's five lanes (tag, code, x, y,
/// delta, as the game gets events). Only a mod with `input` may add any,
/// and only events a player could make.
pub fn granted(manifest: &ModManifest, answer: &[f32]) -> Result<Vec<[f32; EVENT_FLOATS]>> {
    let [version, count, events @ ..] = answer else { bail!("its answer has no header") };
    ensure!(*version == MOD_PROTOCOL, "it answers in protocol {version}, not {MOD_PROTOCOL}");
    let count = whole(*count).map(|count| count as usize).filter(|count| *count <= MAX_MOD_EVENTS);
    let count = count.ok_or_else(|| anyhow!("it adds {} events (at most {MAX_MOD_EVENTS})", answer[1]))?;
    ensure!(events.len() == count * EVENT_FLOATS, "its answer's length does not match its {count} events");
    ensure!(count == 0 || manifest.may(Capability::Input), "it adds input, and it may not (no `input` capability)");
    let mut granted = Vec::with_capacity(count);
    for event in events.as_chunks::<EVENT_FLOATS>().0 {
        ensure!(event.iter().all(|lane| lane.is_finite()), "it adds an event with a nonfinite lane");
        let tag = whole(event[0]).unwrap_or(0);
        let allowed = matches!(
            tag,
            event_tags::KEY_DOWN
                | event_tags::KEY_UP
                | event_tags::POINTER_DOWN
                | event_tags::POINTER_UP
                | event_tags::POINTER_MOVE
                | event_tags::WHEEL
                | event_tags::CONSOLE
        );
        ensure!(allowed, "it adds an event of tag {}, which only the host makes", event[0]);
        if matches!(tag, event_tags::KEY_DOWN | event_tags::KEY_UP | event_tags::POINTER_DOWN | event_tags::POINTER_UP)
        {
            ensure!(whole(event[1]).is_some(), "it adds a key or button that is no whole number");
        }
        granted.push(*event);
    }
    Ok(granted)
}

/// Adds `events` to an input packet, after its own.
pub fn splice(input: &mut Vec<f32>, events: &[[f32; EVENT_FLOATS]]) -> Result<()> {
    if events.is_empty() {
        return Ok(());
    }
    let count = input.get(3).copied().and_then(whole).ok_or_else(|| anyhow!("an input packet without its header"))?;
    let end = FRAME_HEADER_FLOATS + count as usize * EVENT_FLOATS;
    ensure!(end <= input.len(), "an input packet shorter than its events");
    input.splice(end..end, events.iter().flatten().copied());
    input[3] = (count as usize + events.len()) as f32;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_protocol::boundary::encode_frame;
    use pen_protocol::{Frame, GameEvent, Viewport};

    fn input(events: Vec<GameEvent>) -> Vec<f32> {
        let frame = Frame { timestamp: 0.0, viewport: Viewport { width: 640, height: 480 }, events };
        encode_frame(&frame, 1.0 / 60.0, &[7.0, 8.0]).unwrap()
    }

    fn manifest(capabilities: &str) -> ModManifest {
        ModManifest::parse(&format!(r#"{{"name": "sticky", "capabilities": {capabilities}}}"#)).unwrap()
    }

    #[test]
    fn manifests_name_their_capabilities() {
        let both = manifest(r#"["input", "observe", "input"]"#);
        assert_eq!(both.capabilities, [Capability::Observe, Capability::Input]);
        assert!(manifest("[]").capabilities.is_empty());
        assert!(ModManifest::parse(r#"{"name": "x", "capabilities": ["network"]}"#).is_err());
        assert!(ModManifest::parse(r#"{"name": "../x"}"#).is_err());
        assert!(ModManifest::parse(r#"{"capabilities": []}"#).is_err());
    }

    #[test]
    fn mods_see_the_frame_and_its_events_only_when_they_may() {
        let input = input(vec![GameEvent::KeyDown("ArrowLeft".into())]);
        let blind = request(&manifest(r#"["input"]"#), 9, &input);
        assert_eq!(blind, [1.0, 9.0, 1.0 / 60.0, 640.0, 480.0, 0.0]);
        let seeing = request(&manifest(r#"["observe"]"#), 9, &input);
        assert_eq!(seeing[5..], [1.0, 1.0, 0x110100 as f32, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn answers_add_only_what_a_player_could_and_the_mod_may() {
        let input_mod = manifest(r#"["input"]"#);
        let key = [event_tags::KEY_DOWN as f32, 0x110101 as f32, 0.0, 0.0, 0.0];
        let console = [event_tags::CONSOLE as f32, 3.0, 12.0, 0.0, 0.0];
        let answer = [&[1.0, 2.0][..], &key, &console].concat();
        assert_eq!(granted(&input_mod, &answer).unwrap(), [key, console]);
        assert!(granted(&manifest("[]"), &[1.0, 0.0]).unwrap().is_empty(), "nothing to grant");
        let refused = |manifest: &ModManifest, answer: &[f32]| format!("{:#}", granted(manifest, answer).unwrap_err());
        assert!(refused(&manifest(r#"["observe"]"#), &answer).contains("no `input` capability"));
        let forged = [event_tags::SERVICE_ANSWER as f32, 0.0, 1.0, 0.0, 0.0];
        assert!(refused(&input_mod, &[&[1.0, 1.0][..], &forged].concat()).contains("only the host makes"));
        assert!(refused(&input_mod, &[1.0, 2.0, 1.0]).contains("does not match"));
        assert!(refused(&input_mod, &[2.0, 0.0]).contains("protocol 2"));
        assert!(refused(&input_mod, &[1.0, 65.0]).contains("at most 64"));
        let fraction = [event_tags::KEY_UP as f32, 0.5, 0.0, 0.0, 0.0];
        assert!(refused(&input_mod, &[&[1.0, 1.0][..], &fraction].concat()).contains("whole number"));
    }

    #[test]
    fn spliced_events_follow_the_players() {
        let mut spliced = input(vec![GameEvent::KeyDown("a".into())]);
        let key = [event_tags::KEY_UP as f32, 65.0, 0.0, 0.0, 0.0];
        splice(&mut spliced, &[key]).unwrap();
        assert_eq!(spliced, input(vec![GameEvent::KeyDown("a".into()), GameEvent::KeyUp("a".into())]));
        assert_eq!(spliced[spliced.len() - 2..], [7.0, 8.0], "the state lanes stay last");
    }
}
