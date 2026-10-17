//! Mods beside the game (`pen --mod DIR`; pen_runtime::mods and
//! docs/mods.md): each loaded from its directory, compiled by the
//! development engine (whose code checks a deadline) and run in a sandbox of
//! its own, [`MEMORY`] of memory, [`DEADLINE`] a call, with no imports.
//! Each frame the player plays live, before its input goes to the game,
//! every running mod sees it and may add events; one that fails is stopped,
//! and the log says why.
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use blot_host::{Guest, GuestModule, Limits, Value};
use pen_runtime::mods::{MOD_ENTRY, ModManifest, check_mod, granted, request, splice};

use crate::devlog::{self, Source};

/// A mod's linear memory, at most.
pub const MEMORY: usize = 64 << 20;
/// How long one call of a mod may run.
pub const DEADLINE: Duration = Duration::from_millis(100);

struct Mod {
    manifest: ModManifest,
    guest: Guest,
    stopped: bool,
}

pub struct Mods {
    mods: Vec<Mod>,
    /// The live frames they have seen.
    frame: u64,
}

impl Mods {
    /// The mods in `directories`, each a `mod.json` and `mod.wasm`.
    pub fn load(directories: &[PathBuf]) -> Result<Mods> {
        let engine = blot_host::dev_engine().context("mods are compiled, and this pen has no compiler")?;
        let mut mods = Vec::new();
        for directory in directories {
            let loaded = load(&engine, directory).with_context(|| format!("the mod in {}", directory.display()))?;
            let capabilities: Vec<_> = loaded
                .manifest
                .capabilities
                .iter()
                .map(|capability| format!("{capability:?}").to_lowercase())
                .collect();
            devlog::info(
                Source::Session,
                format!("mod {} loaded, may {}", loaded.manifest.name, capabilities.join(" and ")),
            );
            mods.push(loaded);
        }
        Ok(Mods { mods, frame: 0 })
    }

    /// Shows each running mod a live frame's input, and adds what they may
    /// add to it, after the player's events.
    pub fn frame(&mut self, input: &mut Vec<f32>) {
        let frame = self.frame;
        self.frame += 1;
        let mut added = Vec::new();
        for running in self.mods.iter_mut().filter(|running| !running.stopped) {
            let asked = Value::ArrayF32(request(&running.manifest, frame, input));
            let answer = running.guest.call(MOD_ENTRY, asked).and_then(Value::into_f32_array);
            match answer.and_then(|answer| granted(&running.manifest, &answer)) {
                Ok(events) => added.extend(events),
                Err(error) => {
                    running.stopped = true;
                    devlog::error(Source::Session, format!("mod {} stopped: {error:#}", running.manifest.name));
                }
            }
        }
        if let Err(error) = splice(input, &added) {
            devlog::error(Source::Session, format!("mods: {error:#}"));
        }
    }

    /// The mods still running, by name.
    #[cfg(all(test, feature = "compiler"))]
    pub fn running(&self) -> Vec<&str> {
        self.mods.iter().filter(|running| !running.stopped).map(|running| running.manifest.name.as_str()).collect()
    }
}

fn load(engine: &wasmtime::Engine, directory: &Path) -> Result<Mod> {
    let manifest = directory.join("mod.json");
    let text = std::fs::read_to_string(&manifest).with_context(|| format!("reading {}", manifest.display()))?;
    let manifest = ModManifest::parse(&text).with_context(|| format!("reading {}", manifest.display()))?;
    let wasm = directory.join("mod.wasm");
    let wasm = std::fs::read(&wasm).with_context(|| format!("reading {}", wasm.display()))?;
    let module = GuestModule::new(engine, &wasm, None)?;
    check_mod(module.abi())?;
    let mut guest = Guest::sandboxed(engine, &module, Limits { memory: MEMORY })?;
    guest.set_deadline(DEADLINE);
    Ok(Mod { manifest, guest, stopped: false })
}

#[cfg(all(test, feature = "compiler"))]
mod tests {
    use super::*;
    use pen_protocol::boundary::{encode_frame, event_tags};
    use pen_protocol::{Frame, GameEvent, Viewport};

    fn section(id: u8, contents: &[u8]) -> Vec<u8> {
        [&[id, u8::try_from(contents.len()).unwrap()], contents].concat()
    }

    fn name(text: &str) -> Vec<u8> {
        [&[u8::try_from(text.len()).unwrap()], text.as_bytes()].concat()
    }

    /// A mod whose `frame` answers the same every time: it presses the
    /// right arrow. With `spin` it never answers.
    fn presser(spin: bool) -> Vec<u8> {
        let abi = [&[0x02, 0x01][..], &name("frame"), &[0x06, 0x06, 0x00]].concat();
        let lanes = [1.0, 1.0, event_tags::KEY_DOWN as f32, 0x110101 as f32, 0.0, 0.0, 0.0f32];
        let answer: Vec<u8> =
            [7u32.to_le_bytes().to_vec(), lanes.iter().flat_map(|lane| lane.to_le_bytes()).collect()].concat();
        let frame: &[u8] =
            if spin { &[0x00, 0x03, 0x40, 0x0c, 0x00, 0x0b, 0x41, 16, 0x0b] } else { &[0x00, 0x41, 16, 0x0b] };
        let allocate = [0x00, 0x41, 0x80, 0x08, 0x0b];
        let reset = [0x00, 0x41, 0x00, 0x0b];
        let mut code = vec![0x03];
        for body in [frame, &allocate, &reset] {
            code.push(u8::try_from(body.len()).unwrap());
            code.extend(body);
        }
        let exports = [
            &[0x04][..],
            &name("frame"),
            &[0x00, 0x00],
            &name("blot:memory"),
            &[0x02, 0x00],
            &name("blot:allocate"),
            &[0x00, 0x01],
            &name("blot:reset"),
            &[0x00, 0x02],
        ]
        .concat();
        let data = [&[0x01, 0x00, 0x41, 16, 0x0b, u8::try_from(answer.len()).unwrap()][..], &answer].concat();
        [
            &b"\0asm\x01\0\0\0"[..],
            &section(0, &[&name(pen_runtime::abi::ABI_SECTION)[..], &abi].concat()),
            &section(1, &[0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]),
            &section(3, &[0x03, 0x00, 0x00, 0x00]),
            &section(5, &[0x01, 0x00, 0x01]),
            &section(7, &exports),
            &section(10, &code),
            &section(11, &data),
        ]
        .concat()
    }

    fn mod_directory(name: &str, capabilities: &str, wasm: &[u8]) -> PathBuf {
        let directory = std::env::temp_dir().join(format!("pen-mod-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let manifest = format!(r#"{{"name": "{name}", "capabilities": {capabilities}}}"#);
        std::fs::write(directory.join("mod.json"), manifest).unwrap();
        std::fs::write(directory.join("mod.wasm"), wasm).unwrap();
        directory
    }

    fn input(events: Vec<GameEvent>) -> Vec<f32> {
        let frame = Frame { timestamp: 0.0, viewport: Viewport { width: 640, height: 480 }, events };
        encode_frame(&frame, 1.0 / 60.0, &[]).unwrap()
    }

    #[test]
    fn mods_add_what_they_may_and_stop_otherwise() {
        let pressing = mod_directory("pressing", r#"["input"]"#, &presser(false));
        let unasked = mod_directory("unasked", "[]", &presser(false));
        let spinning = mod_directory("spinning", r#"["input"]"#, &presser(true));
        let mut mods = Mods::load(&[pressing.clone(), unasked.clone(), spinning.clone()]).unwrap();
        assert_eq!(mods.running(), ["pressing", "unasked", "spinning"]);
        let mut frame = input(vec![GameEvent::KeyDown("a".into())]);
        mods.frame(&mut frame);
        let expected = input(vec![GameEvent::KeyDown("a".into()), GameEvent::KeyDown("ArrowRight".into())]);
        assert_eq!(frame, expected, "the pressing mod's key, after the player's; nothing from the others");
        assert_eq!(mods.running(), ["pressing"], "the one that may not add input, and the one that ran too long, stop");
        let mut frame = input(Vec::new());
        mods.frame(&mut frame);
        assert_eq!(frame, input(vec![GameEvent::KeyDown("ArrowRight".into())]));
        for directory in [pressing, unasked, spinning] {
            std::fs::remove_dir_all(directory).unwrap();
        }
        let missing = std::env::temp_dir().join("pen-mod-missing");
        assert!(format!("{:#}", Mods::load(&[missing]).err().unwrap()).contains("mod.json"));
    }
}
